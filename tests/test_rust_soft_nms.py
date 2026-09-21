import numpy as np
import pytest


rust = pytest.importorskip('kwimage_ext._rust')


def _soft_nms_case():
    """Case that exercises reordering, suppression, pruning, and tail copying."""
    boxes = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],
        [100, 100, 110, 110],
        [50, 50, 60, 60],
    ], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.4, 0.8], dtype=np.float32)
    return boxes, scores


def test_soft_nms_mutates_in_place_and_tracks_original_indices():
    boxes, scores = _soft_nms_case()

    keep = rust.soft_nms(
        boxes,
        scores,
        thresh=0.5,
        overlap_thresh=0.3,
        sigma=0.5,
        bias=0.0,
        method=0,
    )

    np.testing.assert_array_equal(keep, np.array([1, 3], dtype=np.intp))

    # The highest-scoring input is moved to the front.  The overlapping input
    # is suppressed, then the last active detection is copied over its slot.
    # The inactive tail remains externally observable because Soft-NMS mutates
    # its NumPy arguments in place.
    expected_boxes = np.array([
        [1, 1, 11, 11],
        [50, 50, 60, 60],
        [100, 100, 110, 110],
        [50, 50, 60, 60],
    ], dtype=np.float32)
    expected_scores = np.array([0.9, 0.8, 0.4, 0.8], dtype=np.float32)
    np.testing.assert_array_equal(boxes, expected_boxes)
    np.testing.assert_array_equal(scores, expected_scores)


def test_soft_nms_strided_views_mutate_original_backing_storage():
    source_boxes, source_scores = _soft_nms_case()

    # Put the real inputs in every other element so the views are writable but
    # non-contiguous.  This exercises rust-numpy's safe zero-copy mutable view
    # path rather than only contiguous allocations.
    box_sentinel = np.float32(-12345.0)
    score_sentinel = np.float32(-23456.0)
    box_storage = np.full((len(source_boxes), 8), box_sentinel, dtype=np.float32)
    score_storage = np.full(len(source_scores) * 2, score_sentinel, dtype=np.float32)
    boxes = box_storage[:, ::2]
    scores = score_storage[::2]
    boxes[...] = source_boxes
    scores[...] = source_scores

    assert not boxes.flags.c_contiguous
    assert not scores.flags.c_contiguous
    assert np.shares_memory(boxes, box_storage)
    assert np.shares_memory(scores, score_storage)

    contiguous_boxes = source_boxes.copy()
    contiguous_scores = source_scores.copy()
    kwargs = {
        'thresh': 0.5,
        'overlap_thresh': 0.3,
        'sigma': 0.5,
        'bias': 0.0,
        'method': 0,
    }
    expected_keep = rust.soft_nms(contiguous_boxes, contiguous_scores, **kwargs)
    got_keep = rust.soft_nms(boxes, scores, **kwargs)

    np.testing.assert_array_equal(got_keep, expected_keep)
    np.testing.assert_array_equal(boxes, contiguous_boxes)
    np.testing.assert_array_equal(scores, contiguous_scores)

    # A copied temporary could produce the right return value while failing to
    # update the caller.  Check the base allocations directly, including that
    # the interleaved padding was not touched.
    np.testing.assert_array_equal(box_storage[:, ::2], contiguous_boxes)
    np.testing.assert_array_equal(score_storage[::2], contiguous_scores)
    np.testing.assert_array_equal(
        box_storage[:, 1::2],
        np.full((len(source_boxes), 4), box_sentinel, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        score_storage[1::2],
        np.full(len(source_scores), score_sentinel, dtype=np.float32),
    )


def test_soft_nms_validates_shapes_before_mutation():
    bad_boxes = np.arange(15, dtype=np.float32).reshape(3, 5)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    boxes_before = bad_boxes.copy()
    scores_before = scores.copy()

    with pytest.raises(ValueError, match='ltrb must have shape Nx4'):
        rust.soft_nms(bad_boxes, scores)

    np.testing.assert_array_equal(bad_boxes, boxes_before)
    np.testing.assert_array_equal(scores, scores_before)

    boxes = np.arange(12, dtype=np.float32).reshape(3, 4)
    bad_scores = np.array([0.9, 0.8], dtype=np.float32)
    boxes_before = boxes.copy()
    scores_before = bad_scores.copy()

    with pytest.raises(ValueError, match='scores length must equal number of boxes'):
        rust.soft_nms(boxes, bad_scores)

    np.testing.assert_array_equal(boxes, boxes_before)
    np.testing.assert_array_equal(bad_scores, scores_before)


@pytest.mark.parametrize('num_boxes', [0, 1])
def test_soft_nms_degenerate_sizes(num_boxes):
    boxes = np.zeros((num_boxes, 4), dtype=np.float32)
    scores = np.ones(num_boxes, dtype=np.float32)

    keep = rust.soft_nms(boxes, scores)

    np.testing.assert_array_equal(keep, np.arange(num_boxes, dtype=np.intp))


def _reference_soft_nms(
        boxes, scores, *, thresh=0.001, overlap_thresh=0.3,
        sigma=0.5, bias=0.0, method=0):
    """Small literal Python model of the Rust/Cython in-place algorithm."""
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    scores = np.asarray(scores, dtype=np.float32).copy()
    inds = np.arange(len(scores), dtype=np.intp)
    active_n = len(scores)
    i = 0

    while i < active_n:
        maxpos = i
        maxscore = scores[i]
        for pos in range(i + 1, active_n):
            if scores[pos] > maxscore:
                maxscore = scores[pos]
                maxpos = pos

        if maxpos != i:
            boxes[[i, maxpos]] = boxes[[maxpos, i]]
            scores[i], scores[maxpos] = scores[maxpos], scores[i]
            inds[i], inds[maxpos] = inds[maxpos], inds[i]

        tx1, ty1, tx2, ty2 = boxes[i]
        pos = i + 1
        while pos < active_n:
            x1, y1, x2, y2 = boxes[pos]
            area = (x2 - x1 + bias) * (y2 - y1 + bias)
            iw = min(tx2, x2) - max(tx1, x1) + bias
            if iw > 0.0:
                ih = min(ty2, y2) - max(ty1, y1) + bias
                if ih > 0.0:
                    denom = (
                        (tx2 - tx1 + bias) * (ty2 - ty1 + bias)
                        + area - iw * ih
                    )
                    ov = np.float32(0.0) if denom == 0.0 else iw * ih / denom
                    if method == 1:
                        weight = 1.0 - ov if ov > overlap_thresh else 1.0
                    elif method == 2:
                        weight = np.exp(np.float32(-(ov * ov) / sigma))
                    else:
                        weight = 0.0 if ov > overlap_thresh else 1.0
                    scores[pos] *= np.float32(weight)

            if scores[pos] < thresh:
                last = active_n - 1
                if pos != last:
                    boxes[pos] = boxes[last]
                    scores[pos] = scores[last]
                    inds[pos] = inds[last]
                active_n -= 1
                continue
            pos += 1
        i += 1

    return inds[:active_n].copy(), boxes, scores


@pytest.mark.parametrize('method', [0, 1, 2])
@pytest.mark.parametrize('bias', [0.0, 1.0])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_soft_nms_matches_reference_on_randomized_cases(method, bias, seed):
    rng = np.random.RandomState(seed)
    num = 12

    xy1 = rng.uniform(0, 40, size=(num, 2)).astype(np.float32)
    wh = rng.uniform(1, 20, size=(num, 2)).astype(np.float32)
    boxes = np.concatenate([xy1, xy1 + wh], axis=1).astype(np.float32)
    scores = rng.uniform(0.05, 1.0, size=num).astype(np.float32)

    kwargs = {
        'thresh': 0.15,
        'overlap_thresh': 0.3,
        'sigma': 0.5,
        'bias': bias,
        'method': method,
    }
    expected_keep, expected_boxes, expected_scores = _reference_soft_nms(
        boxes, scores, **kwargs)

    got_boxes = boxes.copy()
    got_scores = scores.copy()
    got_keep = rust.soft_nms(got_boxes, got_scores, **kwargs)

    np.testing.assert_array_equal(got_keep, expected_keep)
    np.testing.assert_array_equal(got_boxes, expected_boxes)
    np.testing.assert_allclose(
        got_scores, expected_scores, rtol=2e-6, atol=2e-7)
