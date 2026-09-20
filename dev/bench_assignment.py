#!/usr/bin/env python3
"""Microbenchmark the Rust sparse greedy assignment kernel against Python."""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from kwimage_ext.algo.assignment import coco_greedy_match
from kwimage_ext.algo.assignment import coco_greedy_match_grid


def python_reference(offsets, txs, ious, pred_order, thresholds, crowd):
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


def make_problem(n_pred, n_true, candidates, seed):
    rng = np.random.RandomState(seed)
    offsets = [0]
    flat_txs = []
    flat_ious = []
    for _ in range(n_pred):
        count = min(n_true, candidates)
        local_txs = rng.choice(n_true, size=count, replace=False)
        local_ious = rng.uniform(.45, 1.0, size=count)
        order = np.lexsort((-local_txs, -local_ious))
        flat_txs.extend(local_txs[order])
        flat_ious.extend(local_ious[order])
        offsets.append(len(flat_txs))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(flat_txs, dtype=np.int64),
        np.asarray(flat_ious, dtype=np.float64),
        rng.permutation(n_pred).astype(np.int64),
        np.linspace(.5, .95, 10),
        np.zeros(n_true, dtype=np.uint8),
    )


def time_call(func, repeat):
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = func()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-pred', type=int, default=100)
    parser.add_argument('--n-true', type=int, default=100)
    parser.add_argument('--candidates', type=int, default=20)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--areas', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    offsets, txs, ious, pred_order, thresholds, crowd = make_problem(
        args.n_pred, args.n_true, args.candidates, args.seed)
    flags = np.zeros(args.n_true, dtype=np.uint8)
    rust_call = lambda: coco_greedy_match(
        offsets, txs, ious, pred_order, thresholds,
        flags, flags, crowd, inclusive=True)[0]
    py_call = lambda: python_reference(
        offsets, txs, ious, pred_order, thresholds, crowd)
    rust_result, rust_time = time_call(rust_call, args.repeat)
    py_result, py_time = time_call(py_call, args.repeat)
    assert np.array_equal(rust_result, py_result)

    annotation_grid = np.zeros((args.areas, args.n_true), dtype=np.uint8)
    rust_grid_call = lambda: coco_greedy_match_grid(
        offsets, txs, ious, pred_order, thresholds,
        flags, annotation_grid, crowd, inclusive=True)[0]
    rust_repeated_call = lambda: np.stack([
        coco_greedy_match(
            offsets, txs, ious, pred_order, thresholds,
            flags, annotation_grid[ai], crowd, inclusive=True)[0]
        for ai in range(args.areas)
    ], axis=0)
    rust_grid_result, rust_grid_time = time_call(rust_grid_call, args.repeat)
    rust_repeated_result, rust_repeated_time = time_call(
        rust_repeated_call, args.repeat)
    assert np.array_equal(rust_grid_result, rust_repeated_result)

    print(f'python median:          {py_time:.6f}s')
    print(f'rust median:            {rust_time:.6f}s')
    print(f'rust/python:            {rust_time / py_time:.3f}x')
    print(f'rust grid median:       {rust_grid_time:.6f}s')
    print(f'rust repeated areas:    {rust_repeated_time:.6f}s')
    print(f'grid/repeated ({args.areas}): {rust_grid_time / rust_repeated_time:.3f}x')


if __name__ == '__main__':
    main()
