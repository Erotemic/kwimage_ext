"""Direct Rust-vs-historical-C/Cython parity tests.

Normal wheels intentionally do not ship the legacy extensions, so this module
skips when the reference stack is absent.  The dedicated parity command sets
``KWIMAGE_EXT_REQUIRE_LEGACY_PARITY=1``; under that mode a missing legacy
backend is a hard failure rather than a skip.
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import pytest


REQUIRE_PARITY = os.environ.get(
    'KWIMAGE_EXT_REQUIRE_LEGACY_PARITY', '').lower() in {'1', 'true', 'yes', 'on'}


def _import_or_skip(name):
    try:
        return importlib.import_module(name)
    except ImportError as ex:
        if REQUIRE_PARITY:
            pytest.fail(
                f'required parity backend {name!r} is not importable: {ex}',
                pytrace=False,
            )
        pytest.skip(f'parity backend {name!r} is not built')


@pytest.fixture(scope='module')
def backends():
    return {
        'rust': _import_or_skip('kwimage_ext._rust'),
        'boxes': _import_or_skip(
            'kwimage_ext.structs._boxes_backend.cython_boxes_legacy'),
        'mask': _import_or_skip(
            'kwimage_ext.structs._mask_backend.cython_mask_legacy'),
        'nms': _import_or_skip(
            'kwimage_ext.algo._nms_backend.cpu_nms_legacy'),
        'soft_nms': _import_or_skip(
            'kwimage_ext.algo._nms_backend.cpu_soft_nms_legacy'),
    }


def _assert_array_parity(got, want, *, rtol=1e-6, atol=1e-7):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, equal_nan=True)


def _assert_rles_equal(got, want):
    assert len(got) == len(want)
    for got_rle, want_rle in zip(got, want):
        assert list(got_rle['size']) == list(want_rle['size'])
        assert got_rle['counts'] == want_rle['counts']


def _box_cases():
    rng = np.random.RandomState(104729)
    cases = [
        np.empty((0, 4), dtype=np.float32),
        np.array([[0, 0, 9, 9]], dtype=np.float32),
        np.array([
            [0, 0, 9, 9],
            [9, 9, 12, 12],
            [10, 10, 10, 10],
            [-5, -7, 3, 2],
            [4, 5, 3, 9],
        ], dtype=np.float32),
    ]
    xy1 = rng.uniform(-1e3, 1e3, size=(37, 2)).astype(np.float32)
    wh = rng.uniform(0, 300, size=(37, 2)).astype(np.float32)
    cases.append(np.hstack([xy1, xy1 + wh]).astype(np.float32))
    # A strided Nx4 view checks that neither backend accidentally requires a
    # C-contiguous input when the historical API did not.
    raw = rng.uniform(-100, 100, size=(11, 8)).astype(np.float32)
    view = raw[:, ::2]
    assert view.shape == (11, 4) and not view.flags.c_contiguous
    cases.append(view)
    return cases


@pytest.mark.parametrize('name', [
    'bbox_ious_c',
    'bbox_overlaps',
    'bbox_intersections',
])
def test_boxes_pairwise_parity(backends, name):
    rust = backends['rust']
    legacy = backends['boxes']
    cases = _box_cases()
    for boxes in cases:
        for query in cases[:4]:
            if name == 'bbox_ious_c':
                for bias in [0.0, 1.0, 0.5]:
                    got = rust.bbox_ious_c(boxes, query, bias=bias)
                    want = legacy.bbox_ious_c(boxes, query, bias=bias)
                    _assert_array_parity(got, want)
            else:
                got = getattr(rust, name)(boxes, query)
                want = getattr(legacy, name)(boxes, query)
                _assert_array_parity(got, want)


def _bbox_similarities_reference(boxes, query_boxes):
    """Independent floating-point reference for the documented formula."""
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    out = np.zeros((len(boxes), len(query_boxes)), dtype=np.float32)
    for n, box in enumerate(boxes):
        cx1 = np.float32((box[0] + box[2]) * np.float32(0.5))
        cy1 = np.float32((box[1] + box[3]) * np.float32(0.5))
        w1 = np.float32(box[2] - box[0] + np.float32(1.0))
        h1 = np.float32(box[3] - box[1] + np.float32(1.0))
        for k, query in enumerate(query_boxes):
            cx2 = np.float32((query[0] + query[2]) * np.float32(0.5))
            cy2 = np.float32((query[1] + query[3]) * np.float32(0.5))
            w2 = np.float32(query[2] - query[0] + np.float32(1.0))
            h2 = np.float32(query[3] - query[1] + np.float32(1.0))
            loc_dist = np.float32(
                np.abs(np.float32(cx1 - cx2)) / np.float32(w1 + w2) +
                np.abs(np.float32(cy1 - cy2)) / np.float32(h1 + h2)
            )
            shape_dist = np.float32(np.abs(
                np.float32(w2 * h2) / np.float32(w1 * h1) - np.float32(1.0)
            ))
            out[n, k] = np.float32(
                -np.log(np.float32(loc_dist + np.float32(0.001))) -
                np.float32(shape_dist * shape_dist) + np.float32(1.0)
            )
    return out


def test_bbox_similarities_rust_uses_float_abs_not_legacy_integer_abs(backends):
    """Keep the Rust correction for a historical Cython ``abs`` defect.

    The legacy Cython source calls C ``abs`` on floating-point distances.  On
    the reference build that truncates fractional values to integers before
    taking the absolute value.  Rust implements the intended floating-point
    formula instead; this is an intentional semantic correction, not a parity
    regression.
    """
    rust = backends['rust']
    legacy = backends['boxes']
    boxes = np.array([[0, 0, 9, 9]], dtype=np.float32)
    query = np.array([
        [0, 0, 9, 9],
        [9, 9, 12, 12],
        [10, 10, 10, 10],
        [-5, -7, 3, 2],
        [4, 5, 3, 9],
    ], dtype=np.float32)

    got = rust.bbox_similarities(boxes, query)
    reference = _bbox_similarities_reference(boxes, query)
    historical = legacy.bbox_similarities(boxes, query)

    _assert_array_parity(got, reference)
    assert not np.allclose(historical[:, 1:], reference[:, 1:])


def test_anchor_intersections_parity(backends):
    rust = backends['rust']
    legacy = backends['boxes']
    anchors = np.array([
        [1, 1], [8, 12], [50, 30], [0.5, 7.25], [1e3, 2e3],
    ], dtype=np.float32)
    query = np.array([
        [0, 0, 0, 0],
        [0, 0, 9, 9],
        [-10, 5, 4, 20],
        [100, 100, 130, 140],
    ], dtype=np.float32)
    _assert_array_parity(
        rust.anchor_intersections(anchors, query),
        legacy.anchor_intersections(anchors, query),
    )


def test_bbox_intersections_self_parity(backends):
    rust = backends['rust']
    legacy = backends['boxes']
    for boxes in _box_cases():
        _assert_array_parity(
            rust.bbox_intersections_self(boxes),
            legacy.bbox_intersections_self(boxes),
        )


def _mask_cases():
    rng = np.random.RandomState(65537)
    cases = []
    for h, w, n in [(1, 1, 1), (3, 5, 2), (17, 19, 4), (32, 31, 3)]:
        zero = np.zeros((h, w, n), dtype=np.uint8, order='F')
        one = np.ones((h, w, n), dtype=np.uint8, order='F')
        noisy = (rng.rand(h, w, n) > 0.72).astype(np.uint8, order='F')
        cases.extend([zero, one, noisy])
    rectangle = np.zeros((23, 29, 3), dtype=np.uint8, order='F')
    rectangle[2:17, 3:21, 0] = 1
    rectangle[8:9, :, 1] = 1
    rectangle[:, 13:14, 2] = 1
    cases.append(rectangle)
    return cases


@pytest.mark.parametrize('mask_index', range(13))
def test_mask_encode_decode_area_bbox_parity(backends, mask_index):
    rust = backends['rust']
    legacy = backends['mask']
    mask = _mask_cases()[mask_index]

    rust_rles = rust.encode(mask)
    legacy_rles = legacy.encode(mask)
    _assert_rles_equal(rust_rles, legacy_rles)

    # Cross-decode canonical COCO RLEs as well as decoding each backend's own
    # output.  This catches codec compatibility that a same-backend roundtrip
    # would miss.
    _assert_array_parity(rust.decode(rust_rles), legacy.decode(legacy_rles))
    _assert_array_parity(rust.decode(legacy_rles), legacy.decode(rust_rles))
    _assert_array_parity(rust.area(rust_rles), legacy.area(legacy_rles))
    _assert_array_parity(rust.toBbox(rust_rles), legacy.toBbox(legacy_rles))


def test_mask_merge_and_iou_parity(backends):
    rust = backends['rust']
    legacy = backends['mask']
    mask = _mask_cases()[-1]
    rust_rles = rust.encode(mask)
    legacy_rles = legacy.encode(mask)

    for intersect in [0, 1]:
        rust_merged = rust.merge(rust_rles, intersect=intersect)
        legacy_merged = legacy.merge(legacy_rles, intersect=intersect)
        _assert_rles_equal([rust_merged], [legacy_merged])

    for crowd in ([0, 0, 0], [1, 0, 1]):
        _assert_array_parity(
            rust.iou(rust_rles, rust_rles, crowd),
            legacy.iou(legacy_rles, legacy_rles, crowd),
            rtol=1e-12,
            atol=0,
        )


def test_mask_bbox_iou_parity(backends):
    rust = backends['rust']
    legacy = backends['mask']
    dt = np.array([
        [0, 0, 10, 10],
        [5.5, 2.25, 8.0, 4.0],
        [-3, -4, 2, 7],
    ], dtype=np.float64)
    gt = np.array([
        [0, 0, 10, 10],
        [8, 1, 5, 6],
    ], dtype=np.float64)
    for crowd in ([0, 0], [1, 0]):
        _assert_array_parity(
            rust.iou(dt, gt, crowd),
            legacy.iou(dt, gt, crowd),
            rtol=1e-12,
            atol=0,
        )


def test_mask_constructors_and_codec_parity(backends):
    rust = backends['rust']
    legacy = backends['mask']
    h, w = 67, 71
    boxes = np.array([
        [3.2, 4.7, 20.3, 15.8],
        [0, 0, 1, 1],
        [12.0, 18.25, 31.4, 28.6],
    ], dtype=np.float64)
    polygons = [
        [3.2, 4.7, 31.6, 6.1, 29.4, 28.8, 5.1, 30.2],
        [18.25, 12.75, 55.4, 13.1, 61.8, 46.6, 20.2, 50.3],
        [7.9, 40.2, 14.1, 35.6, 25.7, 42.8, 22.4, 57.1, 9.0, 55.0],
    ]

    _assert_rles_equal(rust.frBbox(boxes, h, w), legacy.frBbox(boxes, h, w))
    _assert_rles_equal(rust.frPoly(polygons, h, w), legacy.frPoly(polygons, h, w))

    encoded = rust.frPoly(polygons, h, w)
    uncompressed = [
        {
            'size': list(rle['size']),
            'counts': rust._rle_bytes_to_array(rle['counts']).tolist(),
        }
        for rle in encoded
    ]
    _assert_rles_equal(
        rust.frUncompressedRLE(uncompressed, h, w),
        legacy.frUncompressedRLE(uncompressed, h, w),
    )

    counts = np.array([0, 3, 2, 5, 7, 1, 11, 0, 19], dtype=np.uint32)
    rust_bytes = rust._rle_array_to_bytes(counts)
    legacy_bytes = legacy._rle_array_to_bytes(counts)
    assert rust_bytes == legacy_bytes
    _assert_array_parity(
        rust._rle_bytes_to_array(rust_bytes),
        legacy._rle_bytes_to_array(legacy_bytes),
    )


def test_rust_empty_decode_is_deterministic(backends):
    """Document a safety fix rather than invoking legacy undefined behavior.

    The historical decoder indexes ``Rs._R[0]`` even when the RLE list is
    empty.  Rust deliberately returns a deterministic empty tensor instead of
    reproducing that unsafe access.
    """
    rust = backends['rust']
    got = rust.decode([])
    assert got.shape == (0, 0, 0)
    assert got.dtype == np.uint8


def _nms_cases():
    return [
        (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ),
        (
            np.array([[0, 0, 10, 10]], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
        ),
        (
            np.array([
                [0, 0, 10, 10],
                [0, 0, 10, 10],
                [10, 10, 20, 20],
                [50, 50, 60, 60],
            ], dtype=np.float32),
            np.array([0.9, 0.9, 0.8, 0.1], dtype=np.float32),
        ),
    ]


@pytest.mark.parametrize('thresh', [0.0, 0.3, 0.5, 1.0])
@pytest.mark.parametrize('bias', [0.0, 1.0])
def test_cpu_nms_parity(backends, thresh, bias):
    rust = backends['rust']
    legacy = backends['nms']
    for boxes, scores in _nms_cases():
        got = rust.cpu_nms(boxes, scores, thresh, bias=bias)
        want = legacy.cpu_nms(boxes, scores, thresh, bias=bias)
        assert got == want


@pytest.mark.parametrize('method', [0, 1, 2])
@pytest.mark.parametrize('bias', [0.0, 1.0])
def test_soft_nms_parity(backends, method, bias):
    rust = backends['rust']
    legacy = backends['soft_nms']
    # Scores remain comfortably above the pruning threshold so this test covers
    # the shared algorithm rather than the documented legacy threshold bug.
    boxes0 = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],
        [4, 3, 13, 12],
        [50, 50, 60, 60],
    ], dtype=np.float32)
    scores0 = np.array([0.95, 0.82, 0.73, 0.61], dtype=np.float32)
    kwargs = {
        'thresh': 1e-8,
        'overlap_thresh': 0.3,
        'sigma': 0.5,
        'bias': bias,
        'method': method,
    }
    rust_boxes = boxes0.copy()
    rust_scores = scores0.copy()
    legacy_boxes = boxes0.copy()
    legacy_scores = scores0.copy()
    got = rust.soft_nms(rust_boxes, rust_scores, **kwargs)
    want = legacy.soft_nms(legacy_boxes, legacy_scores, **kwargs)

    assert got.dtype == np.dtype(np.intp)
    assert want.dtype == np.dtype(np.intp)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(rust_boxes, legacy_boxes, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(rust_scores, legacy_scores, rtol=1e-6, atol=1e-7)


def test_soft_nms_does_not_preserve_legacy_disjoint_threshold_bug(backends):
    """Low-score disjoint boxes are pruned by Rust, unlike the legacy bug.

    Legacy performs the score-threshold check only inside the positive-overlap
    branch.  A disjoint box below ``thresh`` therefore survives.  That is not a
    compatibility behavior we want to retain in the production backend.
    """
    rust = backends['rust']
    legacy = backends['soft_nms']
    boxes = np.array([
        [0, 0, 10, 10],
        [100, 100, 110, 110],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.0001], dtype=np.float32)
    kwargs = {
        'thresh': 0.001,
        'overlap_thresh': 0.3,
        'sigma': 0.5,
        'bias': 0.0,
        'method': 0,
    }
    rust_keep = rust.soft_nms(boxes.copy(), scores.copy(), **kwargs)
    legacy_keep = legacy.soft_nms(boxes.copy(), scores.copy(), **kwargs)
    np.testing.assert_array_equal(rust_keep, np.array([0], dtype=np.intp))
    np.testing.assert_array_equal(legacy_keep, np.array([0, 1], dtype=np.intp))
