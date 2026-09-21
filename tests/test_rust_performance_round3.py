import numpy as np
import pytest


rust = pytest.importorskip('kwimage_ext._rust')


def _counts(items):
    return [item['counts'] for item in items]


def test_mask_encode_contiguous_fast_path_matches_legacy_patterns_if_available():
    legacy = pytest.importorskip(
        'kwimage_ext.structs._mask_backend.cython_mask_legacy')
    rng = np.random.RandomState(48291)

    cases = []
    cases.append(np.zeros((31, 37, 3), dtype=np.uint8))
    cases.append(np.ones((31, 37, 3), dtype=np.uint8))

    checker = np.indices((31, 37, 3)).sum(axis=0) % 2
    cases.append(checker.astype(np.uint8))

    stripes = np.zeros((127, 131, 2), dtype=np.uint8)
    stripes[:, 16:112, 0] = 1
    stripes[8:120, :, 1] = 1
    cases.append(stripes)

    sparse_lines = np.zeros((129, 133, 2), dtype=np.uint8)
    sparse_lines[20:110, 31, 0] = 1
    sparse_lines[64, 10:120, 1] = 1
    cases.append(sparse_lines)
    cases.append((rng.random_sample((47, 53, 4)) > 0.94).astype(np.uint8))
    cases.append((rng.random_sample((47, 53, 4)) > 0.50).astype(np.uint8))

    structured = np.zeros((64, 72, 3), dtype=np.uint8)
    structured[4:40, 7:31, 0] = 1
    structured[20:61, 15:69, 1] = 1
    structured[::5, 3:60, 2] = 1
    cases.append(structured)

    for mask in cases:
        mask = np.asfortranarray(mask)
        got = rust.encode(mask)
        want = legacy.encode(mask)
        assert _counts(got) == _counts(want)
        assert [item['size'] for item in got] == [item['size'] for item in want]
        np.testing.assert_array_equal(rust.decode(got), mask)


def test_mask_encode_strided_fallback_matches_contiguous_fast_path():
    rng = np.random.RandomState(91231)
    mask = (rng.random_sample((41, 43, 3)) > 0.83).astype(np.uint8)
    fast = np.asfortranarray(mask)

    backing = np.zeros((82, 86, 3), dtype=np.uint8)
    backing[::2, ::2, :] = mask
    strided = backing[::2, ::2, :]
    assert not strided.flags.c_contiguous
    assert not strided.flags.f_contiguous

    got_fast = rust.encode(fast)
    got_strided = rust.encode(strided)
    assert _counts(got_fast) == _counts(got_strided)


def test_bbox_iou_contiguous_fast_path_matches_strided_and_legacy_if_available():
    legacy = pytest.importorskip(
        'kwimage_ext.structs._boxes_backend.cython_boxes_legacy')
    rng = np.random.RandomState(3817)
    xy = rng.uniform(-20, 100, size=(73, 2)).astype(np.float32)
    wh = rng.uniform(0.1, 30, size=(73, 2)).astype(np.float32)
    boxes = np.ascontiguousarray(np.hstack([xy, xy + wh]), dtype=np.float32)
    qxy = rng.uniform(-20, 100, size=(29, 2)).astype(np.float32)
    qwh = rng.uniform(0.1, 30, size=(29, 2)).astype(np.float32)
    query = np.ascontiguousarray(np.hstack([qxy, qxy + qwh]), dtype=np.float32)

    box_backing = np.empty((len(boxes), 8), dtype=np.float32)
    box_backing[:, ::2] = boxes
    boxes_strided = box_backing[:, ::2]
    query_backing = np.empty((len(query), 8), dtype=np.float32)
    query_backing[:, ::2] = query
    query_strided = query_backing[:, ::2]

    got_fast = rust.bbox_ious_c(boxes, query, 1.0)
    got_strided = rust.bbox_ious_c(boxes_strided, query_strided, 1.0)
    want = legacy.bbox_ious_c(boxes, query, 1.0)
    np.testing.assert_array_equal(got_fast, got_strided)
    np.testing.assert_array_equal(got_fast, want)
