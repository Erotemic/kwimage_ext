"""Parity tests for optimized contiguous Rust kernel fast paths."""

import numpy as np
import pytest


rust = pytest.importorskip('kwimage_ext._rust')


def _padded_box_view(boxes, sentinel=-999.0):
    parent = np.full((len(boxes), 8), sentinel, dtype=np.float32)
    view = parent[:, ::2]
    view[...] = boxes
    assert not view.flags.c_contiguous
    return parent, view


def _padded_score_view(scores, sentinel=-999.0):
    parent = np.full(len(scores) * 2, sentinel, dtype=np.float32)
    view = parent[::2]
    view[...] = scores
    assert not view.flags.c_contiguous
    return parent, view


def test_bbox_ious_contiguous_fast_path_matches_strided_fallback():
    rng = np.random.RandomState(1729)
    boxes = rng.rand(41, 4).astype(np.float32)
    query = rng.rand(37, 4).astype(np.float32)
    boxes[:, 2:] += boxes[:, :2]
    query[:, 2:] += query[:, :2]

    _, boxes_view = _padded_box_view(boxes)
    _, query_view = _padded_box_view(query)

    for bias in [0.0, 0.5, 1.0]:
        fast = rust.bbox_ious_c(boxes, query, bias)
        generic = rust.bbox_ious_c(boxes_view, query_view, bias)
        np.testing.assert_allclose(fast, generic, rtol=0, atol=0)


def test_cpu_nms_contiguous_fast_path_matches_strided_fallback():
    rng = np.random.RandomState(1730)
    boxes = rng.rand(80, 4).astype(np.float32)
    boxes[:, 2:] += boxes[:, :2]
    scores = rng.rand(len(boxes)).astype(np.float32)
    # Exercise deterministic tie handling too.
    scores[5:8] = 0.75

    _, boxes_view = _padded_box_view(boxes)
    _, scores_view = _padded_score_view(scores)

    for bias in [0.0, 1.0]:
        fast = rust.cpu_nms(boxes, scores, 0.35, bias)
        generic = rust.cpu_nms(boxes_view, scores_view, 0.35, bias)
        assert fast == generic


def test_soft_nms_contiguous_fast_path_matches_strided_fallback():
    rng = np.random.RandomState(1731)
    base_boxes = rng.rand(64, 4).astype(np.float32)
    base_boxes[:, 2:] += base_boxes[:, :2]
    base_scores = rng.rand(len(base_boxes)).astype(np.float32)

    for method in [0, 1, 2]:
        boxes_fast = base_boxes.copy()
        scores_fast = base_scores.copy()
        keep_fast = rust.soft_nms(
            boxes_fast,
            scores_fast,
            thresh=0.03,
            overlap_thresh=0.3,
            sigma=0.5,
            bias=0.0,
            method=method,
        )

        box_parent, boxes_generic = _padded_box_view(base_boxes)
        score_parent, scores_generic = _padded_score_view(base_scores)
        keep_generic = rust.soft_nms(
            boxes_generic,
            scores_generic,
            thresh=0.03,
            overlap_thresh=0.3,
            sigma=0.5,
            bias=0.0,
            method=method,
        )

        np.testing.assert_array_equal(keep_fast, keep_generic)
        np.testing.assert_allclose(boxes_fast, boxes_generic, rtol=0, atol=0)
        np.testing.assert_allclose(scores_fast, scores_generic, rtol=0, atol=0)
        # The generic strided path must not touch storage outside the views.
        assert np.all(box_parent[:, 1::2] == -999.0)
        assert np.all(score_parent[1::2] == -999.0)


def test_mask_encode_fortran_fast_path_matches_general_layout():
    rng = np.random.RandomState(1732)
    logical = (rng.rand(33, 29, 5) > 0.82).astype(np.uint8)
    fortran = np.asfortranarray(logical)
    c_order = np.ascontiguousarray(logical)

    fast = rust.encode(fortran)
    generic = rust.encode(c_order)

    assert [item['size'] for item in fast] == [item['size'] for item in generic]
    assert [item['counts'] for item in fast] == [item['counts'] for item in generic]


def test_soft_nms_area_cache_matches_legacy_if_available():
    legacy = pytest.importorskip(
        'kwimage_ext.algo._nms_backend.cpu_soft_nms_legacy')
    rng = np.random.RandomState(1733)
    base_boxes = rng.rand(96, 4).astype(np.float32)
    base_boxes[:, 2:] += base_boxes[:, :2]
    base_scores = rng.rand(len(base_boxes)).astype(np.float32)

    for method in [0, 1, 2]:
        rust_boxes = base_boxes.copy()
        rust_scores = base_scores.copy()
        legacy_boxes = base_boxes.copy()
        legacy_scores = base_scores.copy()
        got = rust.soft_nms(
            rust_boxes, rust_scores,
            thresh=0.03, overlap_thresh=0.3, sigma=0.5,
            bias=0.0, method=method)
        want = legacy.soft_nms(
            legacy_boxes, legacy_scores,
            thresh=0.03, overlap_thresh=0.3, sigma=0.5,
            bias=0.0, method=method)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_allclose(rust_boxes, legacy_boxes, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(rust_scores, legacy_scores, rtol=1e-6, atol=1e-6)


def test_bbox_iou_nan_behavior_matches_legacy_if_available():
    legacy = pytest.importorskip(
        'kwimage_ext.structs._boxes_backend.cython_boxes_legacy')

    base_box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    base_query = np.array([1.0, 1.0, 8.0, 8.0], dtype=np.float32)

    # This is an observed compatibility contract rather than an assumption
    # about how Cython lowers min/max: a NaN in any candidate-box coordinate
    # leaves that output row at zero, while a NaN in a query box propagates to
    # NaN for otherwise-valid candidate boxes.
    boxes = [base_box.copy()]
    query = [base_query.copy()]
    for coord in range(4):
        item = base_box.copy()
        item[coord] = np.nan
        boxes.append(item)

        item = base_query.copy()
        item[coord] = np.nan
        query.append(item)

    boxes = np.asarray(boxes, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)

    def check_pair(test_boxes, test_query):
        got = rust.bbox_ious_c(test_boxes, test_query, 1.0)
        want = legacy.bbox_ious_c(test_boxes, test_query, 1.0)
        np.testing.assert_allclose(got, want, rtol=0, atol=0, equal_nan=True)

    # Standard C-order arrays exercise the optimized flat-slice path.
    check_pair(boxes, query)

    # Non-standard views exercise the generic ndarray fallback path while
    # preserving the exact same logical values.
    boxes_storage = np.empty((len(boxes), 8), dtype=np.float32)
    boxes_storage[:, ::2] = boxes
    query_storage = np.empty((len(query), 8), dtype=np.float32)
    query_storage[:, ::2] = query
    strided_boxes = boxes_storage[:, ::2]
    strided_query = query_storage[:, ::2]
    assert not strided_boxes.flags.c_contiguous
    assert not strided_query.flags.c_contiguous
    check_pair(strided_boxes, strided_query)
