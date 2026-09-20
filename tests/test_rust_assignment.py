"""Direct tests for the Rust sparse greedy assignment kernel."""
from __future__ import annotations

import numpy as np
import pytest


rust = pytest.importorskip('kwimage_ext._rust')


def _reference(
    offsets,
    txs,
    ious,
    pred_order,
    thresholds,
    true_ignore,
    annotation_ignore,
    crowd,
    *,
    inclusive=True,
):
    offsets = np.asarray(offsets, dtype=np.int64)
    txs = np.asarray(txs, dtype=np.int64)
    ious = np.asarray(ious, dtype=float)
    pred_order = np.asarray(pred_order, dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=float)
    true_ignore = np.asarray(true_ignore, dtype=bool)
    annotation_ignore = np.asarray(annotation_ignore, dtype=bool)
    crowd = np.asarray(crowd, dtype=bool)
    n_true = len(true_ignore)
    match_txs = np.full((len(thresholds), len(pred_order)), -1, dtype=np.int64)
    match_ious = np.full(match_txs.shape, -1.0, dtype=float)
    true_nmatches = np.zeros((len(thresholds), n_true), dtype=np.int64)
    for ti, threshold in enumerate(thresholds):
        available = ~true_ignore.copy()
        for oi, px in enumerate(pred_order):
            fallback = -1
            fallback_iou = -1.0
            chosen = -1
            chosen_iou = -1.0
            for ci in range(offsets[px], offsets[px + 1]):
                iou = float(ious[ci])
                if inclusive:
                    if iou < threshold:
                        break
                else:
                    if iou <= threshold:
                        break
                tx = int(txs[ci])
                if true_ignore[tx]:
                    continue
                if not available[tx] and not crowd[tx]:
                    continue
                if annotation_ignore[tx]:
                    if fallback < 0:
                        fallback = tx
                        fallback_iou = iou
                    continue
                chosen = tx
                chosen_iou = iou
                break
            if chosen < 0 and fallback >= 0:
                chosen = fallback
                chosen_iou = fallback_iou
            if chosen >= 0:
                if not crowd[chosen]:
                    available[chosen] = False
                true_nmatches[ti, chosen] += 1
                match_txs[ti, oi] = chosen
                match_ious[ti, oi] = chosen_iou
    return match_txs, match_ious, true_nmatches


def _run(*args, inclusive=True):
    from kwimage_ext.algo.assignment import coco_greedy_match
    return coco_greedy_match(*args, inclusive=inclusive)


def test_assignment_capability_is_advertised():
    from kwimage_ext.algo.assignment import backend_metadata

    assert 'coco-assignment' in set(rust.capabilities())
    assert hasattr(rust, 'coco_greedy_match')
    meta = backend_metadata()
    assert meta['kind'] == 'rust'
    assert meta['capability'] == 'coco-assignment'


def test_assignment_known_cases():
    # Three predictions.  p0/p1 both prefer truth 0, p2 can only use truth 1.
    offsets = np.array([0, 2, 4, 5], dtype=np.int64)
    txs = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    ious = np.array([.9, .7, .8, .6, .55], dtype=float)
    pred_order = np.array([0, 1, 2], dtype=np.int64)
    thresholds = np.array([.5, .75, .9], dtype=float)
    flags = np.zeros(2, dtype=np.uint8)

    got = _run(offsets, txs, ious, pred_order, thresholds, flags, flags, flags)
    want = _reference(offsets, txs, ious, pred_order, thresholds, flags, flags, flags)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a, b)

    # At .5: p0->t0, p1->t1, p2 unmatched.
    # At .75 and .9 only p0->t0 (the .9 boundary is inclusive).
    np.testing.assert_array_equal(got[0][0], [0, 1, -1])
    np.testing.assert_array_equal(got[0][1], [0, -1, -1])
    np.testing.assert_array_equal(got[0][2], [0, -1, -1])


def test_crowd_reuse_and_annotation_fallback():
    offsets = np.array([0, 2, 4], dtype=np.int64)
    txs = np.array([0, 1, 0, 1], dtype=np.int64)
    ious = np.array([.95, .80, .90, .70], dtype=float)
    pred_order = np.array([0, 1], dtype=np.int64)
    thresholds = np.array([.5], dtype=float)
    true_ignore = np.array([0, 0], dtype=np.uint8)

    # Higher-overlap ignored truth 0 is a fallback; ordinary truth 1 wins.
    annotation_ignore = np.array([1, 0], dtype=np.uint8)
    crowd = np.array([0, 0], dtype=np.uint8)
    got = _run(
        offsets, txs, ious, pred_order, thresholds,
        true_ignore, annotation_ignore, crowd,
    )
    np.testing.assert_array_equal(got[0][0], [1, 0])

    # A crowd truth can be reused by both predictions.
    annotation_ignore = np.array([1, 0], dtype=np.uint8)
    crowd = np.array([1, 0], dtype=np.uint8)
    offsets2 = np.array([0, 1, 2], dtype=np.int64)
    txs2 = np.array([0, 0], dtype=np.int64)
    ious2 = np.array([.9, .8], dtype=float)
    got = _run(
        offsets2, txs2, ious2, pred_order, thresholds,
        true_ignore, annotation_ignore, crowd,
    )
    np.testing.assert_array_equal(got[0][0], [0, 0])
    np.testing.assert_array_equal(got[2][0], [2, 0])


def test_true_ignore_and_threshold_boundary():
    offsets = np.array([0, 2], dtype=np.int64)
    txs = np.array([0, 1], dtype=np.int64)
    ious = np.array([.8, .5], dtype=float)
    pred_order = np.array([0], dtype=np.int64)
    thresholds = np.array([.5], dtype=float)
    true_ignore = np.array([1, 0], dtype=np.uint8)
    annotation_ignore = np.zeros(2, dtype=np.uint8)
    crowd = np.zeros(2, dtype=np.uint8)

    inclusive = _run(
        offsets, txs, ious, pred_order, thresholds,
        true_ignore, annotation_ignore, crowd, inclusive=True,
    )
    strict = _run(
        offsets, txs, ious, pred_order, thresholds,
        true_ignore, annotation_ignore, crowd, inclusive=False,
    )
    np.testing.assert_array_equal(inclusive[0][0], [1])
    np.testing.assert_array_equal(strict[0][0], [-1])


def test_randomized_reference_parity():
    rng = np.random.RandomState(20260920)
    for _ in range(100):
        n_pred = int(rng.randint(0, 24))
        n_true = int(rng.randint(0, 18))
        candidate_txs = []
        candidate_ious = []
        offsets = [0]
        for _px in range(n_pred):
            if n_true:
                count = int(rng.randint(0, n_true + 1))
                local_txs = rng.choice(n_true, size=count, replace=False)
                local_ious = rng.uniform(.45, 1.0, size=count)
                order = np.lexsort((-local_txs, -local_ious))
                candidate_txs.extend(local_txs[order].tolist())
                candidate_ious.extend(local_ious[order].tolist())
            offsets.append(len(candidate_txs))
        offsets = np.asarray(offsets, dtype=np.int64)
        candidate_txs = np.asarray(candidate_txs, dtype=np.int64)
        candidate_ious = np.asarray(candidate_ious, dtype=float)
        pred_order = rng.permutation(n_pred).astype(np.int64)
        thresholds = np.array([.5, .55, .75, .95], dtype=float)
        true_ignore = (rng.rand(n_true) < .08).astype(np.uint8)
        annotation_ignore = (rng.rand(n_true) < .18).astype(np.uint8)
        crowd = (rng.rand(n_true) < .12).astype(np.uint8)
        for inclusive in [False, True]:
            got = _run(
                offsets, candidate_txs, candidate_ious, pred_order, thresholds,
                true_ignore, annotation_ignore, crowd, inclusive=inclusive,
            )
            want = _reference(
                offsets, candidate_txs, candidate_ious, pred_order, thresholds,
                true_ignore, annotation_ignore, crowd, inclusive=inclusive,
            )
            np.testing.assert_array_equal(got[0], want[0])
            np.testing.assert_allclose(got[1], want[1], rtol=0, atol=0)
            np.testing.assert_array_equal(got[2], want[2])


def test_assignment_input_validation():
    flags = np.zeros(1, dtype=np.uint8)
    with pytest.raises(ValueError, match='candidate_offsets'):
        _run([], [], [], [], [.5], flags, flags, flags)
    with pytest.raises(ValueError, match='duplicate'):
        _run([0, 0, 0], [], [], [0, 0], [.5], flags, flags, flags)
    with pytest.raises(ValueError, match='out-of-range truth'):
        _run([0, 1], [3], [.9], [0], [.5], flags, flags, flags)


def _run_grid(*args, inclusive=True):
    from kwimage_ext.algo.assignment import coco_greedy_match_grid
    return coco_greedy_match_grid(*args, inclusive=inclusive)


def _reference_grid(
    offsets,
    txs,
    ious,
    pred_order,
    thresholds,
    true_ignore,
    annotation_ignore_grid,
    crowd,
    *,
    inclusive=True,
):
    annotation_ignore_grid = np.asarray(annotation_ignore_grid, dtype=np.uint8)
    parts = [
        _reference(
            offsets,
            txs,
            ious,
            pred_order,
            thresholds,
            true_ignore,
            annotation_ignore,
            crowd,
            inclusive=inclusive,
        )
        for annotation_ignore in annotation_ignore_grid
    ]
    if not parts:
        n_thresh = len(thresholds)
        n_pred = len(pred_order)
        n_true = len(true_ignore)
        return (
            np.empty((0, n_thresh, n_pred), dtype=np.int64),
            np.empty((0, n_thresh, n_pred), dtype=float),
            np.empty((0, n_thresh, n_true), dtype=np.int64),
        )
    return tuple(np.stack([p[idx] for p in parts], axis=0) for idx in range(3))


def test_assignment_grid_capability_is_advertised():
    from kwimage_ext.algo.assignment import backend_metadata

    assert 'coco-assignment-grid' in set(rust.capabilities())
    assert hasattr(rust, 'coco_greedy_match_grid')
    meta = backend_metadata()
    assert 'coco-assignment-grid' in meta['capabilities']


def test_assignment_grid_matches_repeated_single_area_calls():
    offsets = np.array([0, 3, 5, 7], dtype=np.int64)
    txs = np.array([0, 1, 2, 0, 2, 1, 2], dtype=np.int64)
    ious = np.array([.95, .85, .60, .90, .55, .80, .70], dtype=float)
    pred_order = np.array([0, 1, 2], dtype=np.int64)
    thresholds = np.array([.5, .75, .9], dtype=float)
    true_ignore = np.array([0, 0, 0], dtype=np.uint8)
    annotation_ignore_grid = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    crowd = np.array([0, 1, 0], dtype=np.uint8)

    got = _run_grid(
        offsets,
        txs,
        ious,
        pred_order,
        thresholds,
        true_ignore,
        annotation_ignore_grid,
        crowd,
    )
    want = _reference_grid(
        offsets,
        txs,
        ious,
        pred_order,
        thresholds,
        true_ignore,
        annotation_ignore_grid,
        crowd,
    )
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_allclose(got[1], want[1], rtol=0, atol=0)
    np.testing.assert_array_equal(got[2], want[2])


def test_assignment_grid_randomized_reference_parity():
    rng = np.random.RandomState(20260921)
    for _ in range(50):
        n_pred = int(rng.randint(0, 20))
        n_true = int(rng.randint(0, 16))
        n_area = int(rng.randint(1, 6))
        candidate_txs = []
        candidate_ious = []
        offsets = [0]
        for _px in range(n_pred):
            if n_true:
                count = int(rng.randint(0, n_true + 1))
                local_txs = rng.choice(n_true, size=count, replace=False)
                local_ious = rng.uniform(.45, 1.0, size=count)
                order = np.lexsort((-local_txs, -local_ious))
                candidate_txs.extend(local_txs[order].tolist())
                candidate_ious.extend(local_ious[order].tolist())
            offsets.append(len(candidate_txs))
        offsets = np.asarray(offsets, dtype=np.int64)
        candidate_txs = np.asarray(candidate_txs, dtype=np.int64)
        candidate_ious = np.asarray(candidate_ious, dtype=float)
        pred_order = rng.permutation(n_pred).astype(np.int64)
        thresholds = np.array([.5, .55, .75, .95], dtype=float)
        true_ignore = (rng.rand(n_true) < .08).astype(np.uint8)
        annotation_ignore_grid = (
            rng.rand(n_area, n_true) < .20
        ).astype(np.uint8)
        crowd = (rng.rand(n_true) < .12).astype(np.uint8)
        for inclusive in [False, True]:
            got = _run_grid(
                offsets,
                candidate_txs,
                candidate_ious,
                pred_order,
                thresholds,
                true_ignore,
                annotation_ignore_grid,
                crowd,
                inclusive=inclusive,
            )
            want = _reference_grid(
                offsets,
                candidate_txs,
                candidate_ious,
                pred_order,
                thresholds,
                true_ignore,
                annotation_ignore_grid,
                crowd,
                inclusive=inclusive,
            )
            np.testing.assert_array_equal(got[0], want[0])
            np.testing.assert_allclose(got[1], want[1], rtol=0, atol=0)
            np.testing.assert_array_equal(got[2], want[2])
