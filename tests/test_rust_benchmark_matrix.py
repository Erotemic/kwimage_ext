from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = REPO_ROOT / 'dev' / 'benchmarks' / 'rust_kernel_benchmarks.py'
PROFILE_PATH = REPO_ROOT / 'dev' / 'profiling' / 'profile_rust_kernels.py'


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _variation(kind):
    rng = np.random.RandomState(24680)
    if kind == 'empty':
        data = np.zeros((17, 19, 3), dtype=np.uint8, order='F')
    elif kind == 'solid':
        data = np.ones((17, 19, 3), dtype=np.uint8, order='F')
    elif kind == 'sparse':
        data = np.asfortranarray(
            (rng.random_sample((37, 41, 5)) > 0.94).astype(np.uint8))
    elif kind == 'dense':
        data = np.asfortranarray(
            (rng.random_sample((37, 41, 5)) > 0.50).astype(np.uint8))
    elif kind == 'checkerboard':
        yy, xx = np.indices((33, 35))
        plane = ((yy + xx) % 2).astype(np.uint8)
        data = np.asfortranarray(np.stack([plane, 1 - plane], axis=2))
    elif kind == 'uint8_255':
        data = np.zeros((65, 67, 3), dtype=np.uint8, order='F')
        data[3:61, 5:49, 0] = 255
        data[::3, 9:54, 1] = 255
        data[11:40, ::4, 2] = 255
    elif kind == 'c_order':
        data = (rng.random_sample((31, 29, 4)) > 0.80).astype(np.uint8)
        assert data.flags.c_contiguous
        assert not data.flags.f_contiguous
    elif kind == 'strided':
        base = (rng.random_sample((62, 58, 4)) > 0.80).astype(np.uint8)
        data = base[::2, ::2, :]
        assert not data.flags.c_contiguous
        assert not data.flags.f_contiguous
    elif kind == 'single_row':
        data = np.asfortranarray(
            (rng.random_sample((1, 257, 2)) > 0.75).astype(np.uint8))
    elif kind == 'single_col':
        data = np.asfortranarray(
            (rng.random_sample((257, 1, 2)) > 0.75).astype(np.uint8))
    else:
        raise AssertionError(kind)
    return data


@pytest.mark.parametrize('kind', [
    'empty',
    'solid',
    'sparse',
    'dense',
    'checkerboard',
    'uint8_255',
    'c_order',
    'strided',
    'single_row',
    'single_col',
])
def test_mask_encode_common_variations_match_canonical_backends(kind):
    rust = pytest.importorskip('kwimage_ext._rust')
    masks = _variation(kind)
    canonical = np.asfortranarray((masks != 0).astype(np.uint8))

    got = rust.encode(masks)
    canonical_got = rust.encode(canonical)
    assert _rle_signature(got) == _rle_signature(canonical_got)
    np.testing.assert_array_equal(rust.decode(got), canonical)

    try:
        from kwimage_ext.structs._mask_backend import cython_mask_legacy
    except Exception:
        cython_mask_legacy = None
    if cython_mask_legacy is not None:
        want = cython_mask_legacy.encode(canonical)
        assert _rle_signature(got) == _rle_signature(want)

    pycoco = pytest.importorskip('pycocotools.mask')
    want = pycoco.encode(canonical)
    assert _rle_signature(got) == _rle_signature(want)


def test_benchmark_matrix_declares_common_mask_variations():
    bench = _load_module(BENCH_PATH, '_kwimage_benchmark_matrix_test')
    names = {spec.name for spec in bench.CASE_SPECS}
    expected = {
        'mask_encode_sparse_128x128x16',
        'mask_encode_512x512x4',
        'mask_encode_dense_512x512x4',
        'mask_encode_structured_512x512x4',
        'mask_encode_solid_512x512x4',
        'mask_encode_many_64x64x128',
        'mask_encode_odd_513x509x4',
        'mask_encode_uint8_255_512x512x4',
        'mask_encode_hd_720x1280x1',
        'mask_iou_fragmented_48',
        'mask_iou_structured_64',
        'mask_iou_crowd_32',
    }
    assert expected <= names
    assert bench.CASE_BY_NAME['mask_encode_hd_720x1280x1'].quick is False


def test_mask_benchmark_fixtures_record_useful_traits():
    bench = _load_module(BENCH_PATH, '_kwimage_benchmark_traits_test')
    rng = np.random.RandomState(0)
    for name in [
        'mask_encode_sparse_128x128x16',
        'mask_encode_dense_512x512x4',
        'mask_encode_structured_512x512x4',
        'mask_encode_uint8_255_512x512x4',
    ]:
        masks = bench._make_mask_encode_fixture(name, rng)
        traits = bench._mask_traits(masks)
        assert traits['shape'] == list(masks.shape)
        assert traits['dtype'] == 'uint8'
        assert traits['f_contiguous'] is True
        assert 0.0 <= traits['foreground_fraction'] <= 1.0
        assert 0.0 <= traits['logical_transition_fraction'] <= 1.0
    values_255 = bench._mask_traits(
        bench._make_mask_encode_fixture(
            'mask_encode_uint8_255_512x512x4', np.random.RandomState(0)))
    assert values_255['stored_values'] == [0, 255]


def test_direct_pycocotools_backend_is_exposed_when_installed():
    pytest.importorskip('pycocotools.mask')
    pytest.importorskip('kwimage_ext._rust')
    bench = _load_module(BENCH_PATH, '_kwimage_benchmark_pycoco_test')
    backends = bench.available_backends('mask_encode_structured_512x512x4')
    assert 'pycocotools' in backends


def test_comparison_rows_keep_every_comparator_and_use_conservative_band():
    profile = _load_module(PROFILE_PATH, '_kwimage_profile_claim_test')
    assert profile._comparison_band(0.95) == 'faster_by_at_least_5pct'
    assert profile._comparison_band(0.951) == 'within_5pct'
    assert profile._comparison_band(1.049) == 'within_5pct'
    assert profile._comparison_band(1.05) == 'slower_by_at_least_5pct'

    payload = {
        'results': [
            {'case': 'mask', 'family': 'mask', 'backend': 'rust',
             'median_ns': 90.0, 'p05_ns': 88.0, 'p95_ns': 92.0,
             'traits': {'shape': [1, 1, 1]}},
            {'case': 'mask', 'family': 'mask', 'backend': 'legacy',
             'median_ns': 100.0, 'p05_ns': 98.0, 'p95_ns': 102.0},
            {'case': 'mask', 'family': 'mask', 'backend': 'pycocotools',
             'median_ns': 120.0, 'p05_ns': 118.0, 'p95_ns': 122.0},
        ],
    }
    rows = profile._comparison_rows(payload)
    assert [row['comparator'] for row in rows] == ['legacy', 'pycocotools']
    assert rows[0]['rust_over_comparator'] == pytest.approx(0.9)
    assert rows[1]['rust_over_comparator'] == pytest.approx(0.75)
    assert all(row['observed_band'] == 'faster_by_at_least_5pct' for row in rows)
