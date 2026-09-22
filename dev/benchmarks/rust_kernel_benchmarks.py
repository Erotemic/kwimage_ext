#!/usr/bin/env python3
"""Deterministic microbenchmarks for kwimage_ext's Rust kernels.

The benchmark surface is intentionally Python-facing: these numbers include the
PyO3/NumPy boundary that real callers pay, but exclude deterministic fixture
construction.  A separate perf workload mode repeats one named case long enough
for hardware-counter and sampling profilers.

The script has no benchmark-framework dependency.  It needs only NumPy and an
installed kwimage_ext Rust extension, which keeps evidence collection usable in
release environments.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CaseSpec:
    name: str
    family: str
    description: str
    perf_default: bool = False
    quick: bool = True


@dataclass
class PreparedCase:
    spec: CaseSpec
    backend: str
    call: Callable[[], object]
    prepare: Callable[[], None] | None = None
    verify: Callable[[], dict] | None = None
    work_items: int | None = None
    work_unit: str | None = None
    traits: dict | None = None

    @property
    def mutates(self) -> bool:
        return self.prepare is not None


CASE_SPECS = (
    CaseSpec(
        'boxes_iou_small', 'boxes',
        '64 x 64 pairwise box IoU; boundary-sensitive small workload.',
    ),
    CaseSpec(
        'boxes_iou_large', 'boxes',
        '2048 x 512 pairwise box IoU; arithmetic/cache throughput workload.',
        perf_default=True,
    ),
    CaseSpec(
        'cpu_nms_sparse_1000', 'nms',
        '1000 mostly-disjoint boxes; near worst-case pair scanning.',
        perf_default=True,
    ),
    CaseSpec(
        'cpu_nms_dense_1000', 'nms',
        '1000 heavily overlapping boxes; suppression-heavy branch behavior.',
    ),
    CaseSpec(
        'soft_nms_gaussian_512', 'nms',
        '512 boxes with Gaussian Soft-NMS; mutating/transcendental workload.',
        perf_default=True,
    ),
    CaseSpec(
        'mask_encode_sparse_128x128x16', 'mask',
        'Sixteen sparse 128 x 128 masks; small-mask / per-instance overhead workload.',
    ),
    CaseSpec(
        'mask_encode_512x512x4', 'mask',
        'Four sparse 512 x 512 masks encoded to COCO RLE.',
        perf_default=True,
    ),
    CaseSpec(
        'mask_encode_dense_512x512x4', 'mask',
        'Four 50%-dense random 512 x 512 masks; transition-heavy RLE workload.',
    ),
    CaseSpec(
        'mask_encode_structured_512x512x4', 'mask',
        'Four structured 512 x 512 masks; realistic long-run rectangle workload.',
        perf_default=True,
    ),
    CaseSpec(
        'mask_encode_solid_512x512x4', 'mask',
        'Four solid/half-plane 512 x 512 masks; extreme long-run workload.',
    ),
    CaseSpec(
        'mask_encode_many_64x64x128', 'mask',
        '128 small 64 x 64 masks; Python/RLE-object creation overhead workload.',
    ),
    CaseSpec(
        'mask_encode_odd_513x509x4', 'mask',
        'Four structured odd-sized 513 x 509 masks; non-power-of-two dimensions.',
    ),
    CaseSpec(
        'mask_encode_uint8_255_512x512x4', 'mask',
        'Four structured masks stored as 0/255 uint8; common OpenCV-style representation.',
    ),
    CaseSpec(
        'mask_encode_hd_720x1280x1', 'mask',
        'One structured 720 x 1280 mask; common HD-scale single-instance workload.',
        quick=False,
    ),
    CaseSpec(
        'mask_iou_fragmented_48', 'mask',
        '48 x 48 fragmented-mask IoU matrix; RLE run-scanning workload.',
        perf_default=True,
    ),
    CaseSpec(
        'mask_iou_structured_64', 'mask',
        '64 x 64 structured-mask IoU matrix; long-run bbox-prefilter workload.',
    ),
    CaseSpec(
        'mask_iou_crowd_32', 'mask',
        '32 x 32 structured-mask IoU matrix with alternating COCO crowd flags.',
    ),
    CaseSpec(
        'assignment_sparse', 'assignment',
        '1000 predictions, 1000 truths, 32 candidates, 10 thresholds.',
        perf_default=True,
    ),
    CaseSpec(
        'assignment_grid', 'assignment',
        'Area-batched assignment over 4 areas and 10 thresholds.',
    ),
)
CASE_BY_NAME = {spec.name: spec for spec in CASE_SPECS}


def _rust_module():
    return importlib.import_module('kwimage_ext._rust')


def _optional_module(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _optional_legacy_module(name):
    return _optional_module(name)


def _optional_pycocotools_mask():
    return _optional_module('pycocotools.mask')


def _random_boxes(rng, n, extent=2048.0, min_size=2.0, max_size=128.0):
    xy = rng.uniform(0.0, extent, size=(n, 2)).astype(np.float32)
    wh = rng.uniform(min_size, max_size, size=(n, 2)).astype(np.float32)
    return np.ascontiguousarray(np.hstack([xy, xy + wh]), dtype=np.float32)


def _cluster_boxes(rng, n):
    xy = rng.normal(loc=128.0, scale=3.0, size=(n, 2)).astype(np.float32)
    wh = rng.uniform(80.0, 120.0, size=(n, 2)).astype(np.float32)
    return np.ascontiguousarray(np.hstack([xy, xy + wh]), dtype=np.float32)


def _make_assignment_problem(seed=0, n_pred=1000, n_true=1000, candidates=32):
    rng = np.random.RandomState(seed)
    offsets = [0]
    flat_txs = []
    flat_ious = []
    for _ in range(n_pred):
        count = min(n_true, candidates)
        local_txs = rng.choice(n_true, size=count, replace=False)
        local_ious = rng.uniform(0.35, 1.0, size=count)
        order = np.lexsort((-local_txs, -local_ious))
        flat_txs.extend(local_txs[order])
        flat_ious.extend(local_ious[order])
        offsets.append(len(flat_txs))
    return {
        'offsets': np.asarray(offsets, dtype=np.int64),
        'txs': np.asarray(flat_txs, dtype=np.int64),
        'ious': np.asarray(flat_ious, dtype=np.float64),
        'pred_order': rng.permutation(n_pred).astype(np.int64),
        'thresholds': np.linspace(0.5, 0.95, 10, dtype=np.float64),
        'true_ignore': np.zeros(n_true, dtype=np.uint8),
        'annotation_ignore': np.zeros(n_true, dtype=np.uint8),
        'crowd': np.zeros(n_true, dtype=np.uint8),
    }


def _assignment_python_reference(data):
    offsets = data['offsets']
    txs = data['txs']
    ious = data['ious']
    pred_order = data['pred_order']
    thresholds = data['thresholds']
    crowd = data['crowd']
    n_true = len(crowd)
    out = np.full((len(thresholds), len(pred_order)), -1, dtype=np.int64)
    for ti, threshold in enumerate(thresholds):
        available = np.ones(n_true, dtype=bool)
        for oi, px in enumerate(pred_order):
            for ci in range(offsets[px], offsets[px + 1]):
                iou = ious[ci]
                if iou < threshold:
                    break
                tx = txs[ci]
                if not available[tx] and not crowd[tx]:
                    continue
                out[ti, oi] = tx
                if not crowd[tx]:
                    available[tx] = False
                break
    return out


def _digest_value(value):
    h = hashlib.sha256()

    def update(item):
        if isinstance(item, np.ndarray):
            contiguous = np.ascontiguousarray(item)
            h.update(b'array\0')
            h.update(str(contiguous.dtype).encode())
            h.update(repr(contiguous.shape).encode())
            h.update(contiguous.tobytes())
        elif isinstance(item, (bytes, bytearray, memoryview)):
            h.update(b'bytes\0')
            h.update(bytes(item))
        elif isinstance(item, dict):
            h.update(b'dict\0')
            for key in sorted(item, key=repr):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode() + b'\0')
            for subitem in item:
                update(subitem)
        else:
            h.update(type(item).__name__.encode() + b'\0')
            h.update(repr(item).encode())

    update(value)
    return h.hexdigest()


def _prepare_boxes_case(spec, backend, seed):
    rng = np.random.RandomState(seed)
    if spec.name == 'boxes_iou_small':
        n, k = 64, 64
    else:
        n, k = 2048, 512
    boxes = _random_boxes(rng, n)
    query = _random_boxes(rng, k)
    rust = _rust_module()
    legacy = _optional_legacy_module(
        'kwimage_ext.structs._boxes_backend.cython_boxes_legacy')
    modules = {'rust': rust, 'legacy': legacy}
    module = modules.get(backend)
    if module is None:
        raise LookupError(f'backend {backend!r} unavailable for {spec.name}')

    def call():
        return module.bbox_ious_c(boxes, query, 1.0)

    def verify():
        got = rust.bbox_ious_c(boxes, query, 1.0)
        info = {'digest': _digest_value(got)}
        if legacy is not None:
            want = legacy.bbox_ious_c(boxes, query, 1.0)
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
            info['comparator'] = 'legacy-cython'
        else:
            diag = np.diag(rust.bbox_ious_c(boxes[:8], boxes[:8], 1.0))
            np.testing.assert_allclose(diag, 1.0, rtol=1e-6, atol=1e-6)
            info['comparator'] = 'self-iou-invariant'
        return info

    return PreparedCase(
        spec, backend, call, verify=verify,
        work_items=n * k, work_unit='box-pairs')


def _prepare_nms_case(spec, backend, seed):
    rng = np.random.RandomState(seed)
    rust = _rust_module()
    if spec.name == 'soft_nms_gaussian_512':
        legacy = _optional_legacy_module(
            'kwimage_ext.algo._nms_backend.cpu_soft_nms_legacy')
        modules = {'rust': rust, 'legacy': legacy}
        module = modules.get(backend)
        if module is None:
            raise LookupError(f'backend {backend!r} unavailable for {spec.name}')
        base_boxes = _random_boxes(rng, 512, extent=512, min_size=30, max_size=160)
        base_scores = rng.uniform(0.01, 1.0, size=512).astype(np.float32)
        boxes = base_boxes.copy()
        scores = base_scores.copy()

        def prepare():
            np.copyto(boxes, base_boxes)
            np.copyto(scores, base_scores)

        def call():
            return module.soft_nms(
                boxes, scores, thresh=0.001, overlap_thresh=0.3,
                sigma=0.5, bias=0.0, method=2)

        def verify():
            rust_boxes = base_boxes.copy()
            rust_scores = base_scores.copy()
            got = rust.soft_nms(
                rust_boxes, rust_scores, thresh=0.001, overlap_thresh=0.3,
                sigma=0.5, bias=0.0, method=2)
            info = {
                'digest': _digest_value((got, rust_boxes, rust_scores)),
            }
            if legacy is not None:
                legacy_boxes = base_boxes.copy()
                legacy_scores = base_scores.copy()
                want = legacy.soft_nms(
                    legacy_boxes, legacy_scores, thresh=0.001,
                    overlap_thresh=0.3, sigma=0.5, bias=0.0, method=2)
                np.testing.assert_array_equal(got, want)
                np.testing.assert_allclose(rust_boxes, legacy_boxes, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(rust_scores, legacy_scores, rtol=1e-6, atol=1e-6)
                info['comparator'] = 'legacy-cython'
            else:
                assert len(got) <= len(base_scores)
                assert len(set(map(int, got))) == len(got)
                info['comparator'] = 'index-invariants'
            return info

        return PreparedCase(
            spec, backend, call, prepare=prepare, verify=verify,
            work_items=len(base_scores), work_unit='input-boxes')

    legacy = _optional_legacy_module('kwimage_ext.algo._nms_backend.cpu_nms_legacy')
    modules = {'rust': rust, 'legacy': legacy}
    module = modules.get(backend)
    if module is None:
        raise LookupError(f'backend {backend!r} unavailable for {spec.name}')
    if spec.name == 'cpu_nms_sparse_1000':
        boxes = _random_boxes(rng, 1000, extent=100000, min_size=2, max_size=16)
    else:
        boxes = _cluster_boxes(rng, 1000)
    scores = rng.uniform(0.01, 1.0, size=len(boxes)).astype(np.float32)

    def call():
        return module.cpu_nms(boxes, scores, 0.5, 0.0)

    def verify():
        got = rust.cpu_nms(boxes, scores, 0.5, 0.0)
        assert len(got) == len(set(got))
        assert all(0 <= int(idx) < len(boxes) for idx in got)
        info = {'digest': _digest_value(got)}
        if legacy is not None:
            want = legacy.cpu_nms(boxes, scores, 0.5, 0.0)
            assert list(got) == list(want)
            info['comparator'] = 'legacy-cython'
        else:
            info['comparator'] = 'index-invariants'
        return info

    return PreparedCase(
        spec, backend, call, verify=verify,
        work_items=len(boxes), work_unit='input-boxes')


def _structured_masks(rng, shape, rectangles=24):
    h, w, n = shape
    masks = np.zeros(shape, dtype=np.uint8)
    max_h = max(2, min(h, max(8, h // 5)))
    max_w = max(2, min(w, max(8, w // 5)))
    for chan in range(n):
        for _ in range(rectangles):
            y1 = int(rng.randint(0, max(1, h)))
            x1 = int(rng.randint(0, max(1, w)))
            rh = int(rng.randint(1, max_h + 1))
            rw = int(rng.randint(1, max_w + 1))
            masks[y1:min(h, y1 + rh), x1:min(w, x1 + rw), chan] = 1
    return masks


def _make_mask_encode_fixture(name, rng):
    if name == 'mask_encode_sparse_128x128x16':
        masks = (rng.random_sample((128, 128, 16)) > 0.94).astype(np.uint8)
    elif name == 'mask_encode_512x512x4':
        masks = (rng.random_sample((512, 512, 4)) > 0.94).astype(np.uint8)
    elif name == 'mask_encode_dense_512x512x4':
        masks = (rng.random_sample((512, 512, 4)) > 0.50).astype(np.uint8)
    elif name == 'mask_encode_structured_512x512x4':
        masks = _structured_masks(rng, (512, 512, 4), rectangles=24)
    elif name == 'mask_encode_solid_512x512x4':
        masks = np.zeros((512, 512, 4), dtype=np.uint8)
        masks[:, :, 1] = 1
        masks[:256, :, 2] = 1
        masks[:, :256, 3] = 1
    elif name == 'mask_encode_many_64x64x128':
        masks = _structured_masks(rng, (64, 64, 128), rectangles=4)
    elif name == 'mask_encode_odd_513x509x4':
        masks = _structured_masks(rng, (513, 509, 4), rectangles=24)
    elif name == 'mask_encode_uint8_255_512x512x4':
        masks = _structured_masks(rng, (512, 512, 4), rectangles=24) * np.uint8(255)
    elif name == 'mask_encode_hd_720x1280x1':
        masks = _structured_masks(rng, (720, 1280, 1), rectangles=48)
    else:
        raise AssertionError(name)
    return np.asfortranarray(masks, dtype=np.uint8)


def _make_mask_iou_fixture(name, rng):
    if name == 'mask_iou_fragmented_48':
        masks = (rng.random_sample((96, 96, 48)) > 0.90).astype(np.uint8)
        crowd = [0] * 48
    elif name == 'mask_iou_structured_64':
        masks = _structured_masks(rng, (256, 256, 64), rectangles=8)
        crowd = [0] * 64
    elif name == 'mask_iou_crowd_32':
        masks = _structured_masks(rng, (256, 256, 32), rectangles=8)
        crowd = [idx % 2 for idx in range(32)]
    else:
        raise AssertionError(name)
    return np.asfortranarray(masks, dtype=np.uint8), crowd


def _mask_traits(masks):
    logical = masks != 0
    h, w, n = masks.shape
    transitions = 0
    comparisons = 0
    for idx in range(n):
        flat = np.ravel(logical[:, :, idx], order='F')
        if flat.size > 1:
            transitions += int(np.count_nonzero(flat[1:] != flat[:-1]))
            comparisons += flat.size - 1
    unique_values = np.unique(masks)
    if unique_values.size <= 8:
        stored_values = [int(v) for v in unique_values]
    else:
        stored_values = ['many']
    return {
        'shape': list(map(int, masks.shape)),
        'dtype': str(masks.dtype),
        'f_contiguous': bool(masks.flags.f_contiguous),
        'c_contiguous': bool(masks.flags.c_contiguous),
        'foreground_fraction': float(np.mean(logical)) if logical.size else 0.0,
        'logical_transition_fraction': (
            float(transitions / comparisons) if comparisons else 0.0
        ),
        'stored_values': stored_values,
    }


def _rle_signature(rles):
    signature = []
    for item in rles:
        counts = item['counts']
        if isinstance(counts, str):
            counts = counts.encode()
        else:
            counts = bytes(counts)
        signature.append((tuple(map(int, item['size'])), counts))
    return signature


def _prepare_mask_case(spec, backend, seed):
    rng = np.random.RandomState(seed)
    rust = _rust_module()
    legacy = _optional_legacy_module(
        'kwimage_ext.structs._mask_backend.cython_mask_legacy')
    pycocotools = _optional_pycocotools_mask()
    modules = {
        'rust': rust,
        'legacy': legacy,
        'pycocotools': pycocotools,
    }
    module = modules.get(backend)
    if module is None:
        raise LookupError(f'backend {backend!r} unavailable for {spec.name}')

    if spec.name.startswith('mask_encode_'):
        masks = _make_mask_encode_fixture(spec.name, rng)
        traits = _mask_traits(masks)

        def call():
            return module.encode(masks)

        def verify():
            got = rust.encode(masks)
            got_sig = _rle_signature(got)
            decoded = rust.decode(got)
            expected_binary = (masks != 0).astype(np.uint8)
            np.testing.assert_array_equal(decoded, expected_binary)
            info = {
                'digest': _digest_value(got),
                'comparators': [],
            }
            if legacy is not None:
                want = legacy.encode(masks)
                assert got_sig == _rle_signature(want)
                info['comparators'].append('legacy-cython')
            if pycocotools is not None:
                want = pycocotools.encode(masks)
                assert got_sig == _rle_signature(want)
                info['comparators'].append('pycocotools')
            info['comparator'] = (
                info['comparators'][0] if info['comparators'] else 'roundtrip'
            )
            return info

        return PreparedCase(
            spec, backend, call, verify=verify,
            work_items=masks.size, work_unit='pixels', traits=traits)

    masks, crowd = _make_mask_iou_fixture(spec.name, rng)
    traits = _mask_traits(masks)
    traits['matrix_shape'] = [int(masks.shape[2]), int(masks.shape[2])]
    traits['crowd_fraction'] = float(np.mean(crowd)) if crowd else 0.0
    rust_rles = rust.encode(masks)
    # RLE dictionaries are the interchange representation. Use each backend's
    # encoder for its timed IoU call so the benchmark also catches backend-
    # specific assumptions about canonical COCO RLE objects, while encoding
    # itself remains outside the timed region.
    if backend == 'rust':
        rles = rust_rles
    else:
        rles = module.encode(masks)

    def call():
        return module.iou(rles, rles, crowd)

    def verify():
        got = np.asarray(rust.iou(rust_rles, rust_rles, crowd))
        info = {'digest': _digest_value(got), 'comparators': []}
        if not any(crowd):
            np.testing.assert_allclose(np.diag(got), 1.0, rtol=0, atol=0)
        if legacy is not None:
            legacy_rles = legacy.encode(masks)
            want = np.asarray(legacy.iou(legacy_rles, legacy_rles, crowd))
            np.testing.assert_allclose(got, want, rtol=0, atol=0)
            info['comparators'].append('legacy-cython')
        if pycocotools is not None:
            pycoco_rles = pycocotools.encode(masks)
            want = np.asarray(pycocotools.iou(pycoco_rles, pycoco_rles, crowd))
            np.testing.assert_allclose(got, want, rtol=0, atol=0)
            info['comparators'].append('pycocotools')
        info['comparator'] = (
            info['comparators'][0] if info['comparators'] else 'self-iou-invariant'
        )
        return info

    return PreparedCase(
        spec, backend, call, verify=verify,
        work_items=len(rles) * len(rles), work_unit='rle-pairs', traits=traits)

def _prepare_assignment_case(spec, backend, seed):
    rust = _rust_module()
    data = _make_assignment_problem(seed=seed)
    if backend not in {'rust', 'python'}:
        raise LookupError(f'backend {backend!r} unavailable for {spec.name}')

    args = (
        data['offsets'], data['txs'], data['ious'], data['pred_order'],
        data['thresholds'], data['true_ignore'], data['annotation_ignore'],
        data['crowd'],
    )
    if spec.name == 'assignment_sparse':
        if backend == 'rust':
            def call():
                return rust.coco_greedy_match(*args, inclusive=True)
        else:
            def call():
                return _assignment_python_reference(data)

        def verify():
            got = rust.coco_greedy_match(*args, inclusive=True)[0]
            want = _assignment_python_reference(data)
            np.testing.assert_array_equal(got, want)
            return {'digest': _digest_value(got), 'comparator': 'python-reference'}

        return PreparedCase(
            spec, backend, call, verify=verify,
            work_items=len(data['txs']) * len(data['thresholds']),
            work_unit='candidate-threshold-visits-upper-bound')

    if backend != 'rust':
        raise LookupError('assignment_grid currently has only a Rust benchmark backend')
    grid = np.zeros((4, len(data['true_ignore'])), dtype=np.uint8)
    grid_args = (
        data['offsets'], data['txs'], data['ious'], data['pred_order'],
        data['thresholds'], data['true_ignore'], grid, data['crowd'],
    )

    def call():
        return rust.coco_greedy_match_grid(*grid_args, inclusive=True)

    def verify():
        got = rust.coco_greedy_match_grid(*grid_args, inclusive=True)[0]
        repeated = np.stack([
            rust.coco_greedy_match(
                data['offsets'], data['txs'], data['ious'], data['pred_order'],
                data['thresholds'], data['true_ignore'], grid[ai], data['crowd'],
                inclusive=True,
            )[0]
            for ai in range(len(grid))
        ])
        np.testing.assert_array_equal(got, repeated)
        return {'digest': _digest_value(got), 'comparator': 'repeated-rust-single-area'}

    return PreparedCase(
        spec, backend, call, verify=verify,
        work_items=len(data['txs']) * len(data['thresholds']) * len(grid),
        work_unit='candidate-area-threshold-visits-upper-bound')


def prepare_case(name, backend='rust', seed=0):
    spec = CASE_BY_NAME[name]
    if spec.family == 'boxes':
        return _prepare_boxes_case(spec, backend, seed)
    if spec.family == 'nms':
        return _prepare_nms_case(spec, backend, seed)
    if spec.family == 'mask':
        return _prepare_mask_case(spec, backend, seed)
    if spec.family == 'assignment':
        return _prepare_assignment_case(spec, backend, seed)
    raise AssertionError(spec.family)


def available_backends(name, seed=0):
    backends = []
    family = CASE_BY_NAME[name].family
    candidates = ['rust']
    if family == 'mask':
        candidates.extend(['legacy', 'pycocotools'])
    elif family != 'assignment':
        candidates.append('legacy')
    elif name == 'assignment_sparse':
        candidates.append('python')
    for backend in candidates:
        try:
            prepare_case(name, backend=backend, seed=seed)
        except LookupError:
            continue
        backends.append(backend)
    return backends


def _percentile(values, q):
    if not values:
        return math.nan
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _one_timed_call(case):
    if case.prepare is not None:
        case.prepare()
    start = time.perf_counter_ns()
    result = case.call()
    elapsed = time.perf_counter_ns() - start
    return result, elapsed


def benchmark_case(case, *, samples=9, target_sample_seconds=0.08, warmup=3):
    verify_info = case.verify() if case.verify is not None else {}
    for _ in range(warmup):
        _one_timed_call(case)

    _, probe_ns = _one_timed_call(case)
    if case.mutates:
        loops = 1
    else:
        target_ns = max(1, int(target_sample_seconds * 1e9))
        loops = max(1, min(1_000_000, target_ns // max(probe_ns, 1)))

    sample_ns = []
    last_result = None
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(samples):
            if case.mutates:
                last_result, elapsed = _one_timed_call(case)
            else:
                start = time.perf_counter_ns()
                for _loop_idx in range(loops):
                    last_result = case.call()
                elapsed = time.perf_counter_ns() - start
            sample_ns.append(elapsed / loops)
    finally:
        if gc_was_enabled:
            gc.enable()

    result = {
        'case': case.spec.name,
        'family': case.spec.family,
        'description': case.spec.description,
        'backend': case.backend,
        'samples': samples,
        'loops_per_sample': loops,
        'median_ns': statistics.median(sample_ns),
        'mean_ns': statistics.fmean(sample_ns),
        'min_ns': min(sample_ns),
        'p05_ns': _percentile(sample_ns, 0.05),
        'p95_ns': _percentile(sample_ns, 0.95),
        'max_ns': max(sample_ns),
        'stdev_ns': statistics.pstdev(sample_ns) if len(sample_ns) > 1 else 0.0,
        'sample_ns': sample_ns,
        'work_items': case.work_items,
        'work_unit': case.work_unit,
        'result_digest': _digest_value(last_result),
        'verification': verify_info,
        'traits': case.traits or {},
    }
    if case.work_items:
        result['median_ns_per_work_item'] = result['median_ns'] / case.work_items
    return result


def _extension_metadata():
    rust = _rust_module()
    path = Path(rust.__file__).resolve()
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {
        'module': rust.__name__,
        'path': str(path),
        'sha256': sha256,
        'version': rust.version() if hasattr(rust, 'version') else None,
        'capabilities': list(rust.capabilities()) if hasattr(rust, 'capabilities') else None,
    }


def _optional_distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata():
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        with contextlib.suppress(Exception):
            affinity = sorted(os.sched_getaffinity(0))
    return {
        'python': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'numpy_version': np.__version__,
        'optional_backend_versions': {
            'pycocotools': _optional_distribution_version('pycocotools'),
        },
        'cpu_affinity': affinity,
        'rust_extension': _extension_metadata(),
    }


def run_suite(case_names=None, *, backends='auto', seed=0, samples=9,
              target_sample_seconds=0.08, warmup=3):
    if case_names is None:
        case_names = [spec.name for spec in CASE_SPECS]
    rows = []
    skipped = []
    for name in case_names:
        if backends == 'auto':
            selected = available_backends(name, seed=seed)
        else:
            selected = list(backends)
        for backend in selected:
            try:
                case = prepare_case(name, backend=backend, seed=seed)
            except LookupError as ex:
                skipped.append({'case': name, 'backend': backend, 'reason': str(ex)})
                continue
            rows.append(benchmark_case(
                case, samples=samples,
                target_sample_seconds=target_sample_seconds,
                warmup=warmup,
            ))
    return {
        'schema_version': 2,
        'generated_at_unix': time.time(),
        'seed': seed,
        'environment': environment_metadata(),
        'results': rows,
        'skipped': skipped,
    }


def run_perf_workload(name, *, backend='rust', seed=0, seconds=2.0,
                      warmup=3):
    case = prepare_case(name, backend=backend, seed=seed)
    if case.verify is not None:
        case.verify()
    for _ in range(warmup):
        _one_timed_call(case)
    deadline = time.perf_counter() + seconds
    calls = 0
    last_result = None
    while time.perf_counter() < deadline:
        last_result, _elapsed = _one_timed_call(case)
        calls += 1
    # Keep validation outside the sampled hot loop. Hashing every result made
    # perf attribute a substantial fraction of cycles to SHA256 rather than the
    # Rust kernel we intended to profile.
    digest = _digest_value(last_result) if calls else None
    payload = {
        'case': name,
        'backend': backend,
        'seconds_requested': seconds,
        'calls': calls,
        'last_result_digest': digest,
        'mutating_prepare_in_profile_scope': case.mutates,
    }
    print(json.dumps(payload, sort_keys=True))
    return payload


def _parse_case_names(text):
    if not text or text == 'all':
        return [spec.name for spec in CASE_SPECS]
    names = [part.strip() for part in text.split(',') if part.strip()]
    unknown = sorted(set(names) - set(CASE_BY_NAME))
    if unknown:
        raise SystemExit(f'unknown benchmark case(s): {unknown}')
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List benchmark cases and exit.')
    parser.add_argument('--cases', default='all', help='Comma-separated cases, or all.')
    parser.add_argument('--backend', choices=['auto', 'rust', 'legacy', 'pycocotools', 'python'], default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--target-sample-seconds', type=float, default=0.08)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--perf-workload', choices=sorted(CASE_BY_NAME))
    parser.add_argument('--perf-seconds', type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.list:
        for spec in CASE_SPECS:
            marker = '*' if spec.perf_default else ' '
            print(f'{marker} {spec.name:28s} [{spec.family}] {spec.description}')
        return 0

    if args.perf_workload:
        backend = 'rust' if args.backend == 'auto' else args.backend
        run_perf_workload(
            args.perf_workload, backend=backend, seed=args.seed,
            seconds=args.perf_seconds, warmup=args.warmup)
        return 0

    if args.quick:
        samples = min(args.samples, 3)
        target_seconds = min(args.target_sample_seconds, 0.02)
    else:
        samples = args.samples
        target_seconds = args.target_sample_seconds
    case_names = _parse_case_names(args.cases)
    if args.quick and args.cases == 'all':
        case_names = [name for name in case_names if CASE_BY_NAME[name].quick]
    selected_backends = args.backend if args.backend == 'auto' else [args.backend]
    payload = run_suite(
        case_names,
        backends=selected_backends,
        seed=args.seed,
        samples=samples,
        target_sample_seconds=target_seconds,
        warmup=args.warmup,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + '\n')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
