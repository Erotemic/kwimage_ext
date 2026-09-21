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
