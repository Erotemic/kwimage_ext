"""Rust acceleration for sparse COCO-style greedy assignment.

This module intentionally exposes a narrow kernel rather than an evaluator.
Callers provide an already-built sparse candidate graph and stable prediction
order.  Geometry, class compatibility, area policy, and result accumulation
remain owned by the caller.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def _rust_backend() -> Any:
    try:
        rust = importlib.import_module('kwimage_ext._rust')
    except ImportError as ex:  # pragma: no cover - exercised by downstream fallback
        raise ImportError(
            'kwimage_ext Rust assignment support is unavailable. Install a '
            'Rust-enabled kwimage_ext wheel or use the caller\'s Python fallback.'
        ) from ex
    if not hasattr(rust, 'coco_greedy_match'):
        raise ImportError(
            'The installed kwimage_ext Rust backend predates the '
            'coco_greedy_match capability.'
        )
    return rust


def backend_metadata():
    """Return stable provenance for the assignment kernel."""
    rust = _rust_backend()
    capabilities = ['coco-assignment']
    if hasattr(rust, 'coco_greedy_match_grid'):
        capabilities.append('coco-assignment-grid')
    return {
        'name': 'kwimage_ext.coco_greedy_match',
        'kind': 'rust',
        'module': rust.__name__,
        'file': getattr(rust, '__file__', None),
        'version': rust.version() if hasattr(rust, 'version') else None,
        'capability': 'coco-assignment',
        'capabilities': capabilities,
    }


def coco_greedy_match(
    candidate_offsets,
    candidate_txs,
    candidate_ious,
    pred_order,
    thresholds,
    true_ignore_flags,
    annotation_ignore_flags,
    crowd_flags,
    *,
    inclusive=True,
):
    """Run batched greedy matching for multiple IoU thresholds.

    Args:
        candidate_offsets (ArrayLike): CSR offsets of length ``n_pred + 1``.
        candidate_txs (ArrayLike): Flat truth indexes for all candidate lists.
        candidate_ious (ArrayLike): Flat overlaps aligned with ``candidate_txs``.
        pred_order (ArrayLike): Stable score order after maxDet/filtering.
        thresholds (ArrayLike): IoU thresholds to evaluate independently.
        true_ignore_flags (ArrayLike): Truths excluded from ordinary matching.
        annotation_ignore_flags (ArrayLike): Matchable-but-ignored truths.
        crowd_flags (ArrayLike): Reusable truths.
        inclusive (bool): Use ``IoU >= threshold`` rather than ``>``.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``match_txs``,
        ``match_ious``, and ``true_nmatches``.  The first two have shape
        ``(n_threshold, len(pred_order))``; the third has shape
        ``(n_threshold, n_truth)``.

    Notes:
        Candidate lists must already be sorted by descending overlap with the
        desired equal-overlap tie policy encoded in their order.  This keeps the
        compiled kernel independent of kwcoco's geometry and hierarchy logic.
    """
    rust = _rust_backend()
    return rust.coco_greedy_match(
        np.ascontiguousarray(candidate_offsets, dtype=np.int64),
        np.ascontiguousarray(candidate_txs, dtype=np.int64),
        np.ascontiguousarray(candidate_ious, dtype=np.float64),
        np.ascontiguousarray(pred_order, dtype=np.int64),
        np.ascontiguousarray(thresholds, dtype=np.float64),
        np.ascontiguousarray(true_ignore_flags, dtype=np.uint8),
        np.ascontiguousarray(annotation_ignore_flags, dtype=np.uint8),
        np.ascontiguousarray(crowd_flags, dtype=np.uint8),
        inclusive=bool(inclusive),
    )


def coco_greedy_match_grid(
    candidate_offsets,
    candidate_txs,
    candidate_ious,
    pred_order,
    thresholds,
    true_ignore_flags,
    annotation_ignore_flags,
    crowd_flags,
    *,
    inclusive=True,
):
    """Run greedy matching for every area-range / threshold combination.

    This is the area-batched companion to :func:`coco_greedy_match`.
    ``annotation_ignore_flags`` has shape ``(n_area, n_true)`` and each row is
    evaluated independently while candidate geometry and prediction ordering
    are shared.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``match_txs``,
        ``match_ious``, and ``true_nmatches`` with shapes
        ``(n_area, n_threshold, n_pred_order)``,
        ``(n_area, n_threshold, n_pred_order)``, and
        ``(n_area, n_threshold, n_true)`` respectively.
    """
    rust = _rust_backend()
    if not hasattr(rust, 'coco_greedy_match_grid'):
        raise ImportError(
            'The installed kwimage_ext Rust backend predates the '
            'coco_greedy_match_grid capability.'
        )
    return rust.coco_greedy_match_grid(
        np.ascontiguousarray(candidate_offsets, dtype=np.int64),
        np.ascontiguousarray(candidate_txs, dtype=np.int64),
        np.ascontiguousarray(candidate_ious, dtype=np.float64),
        np.ascontiguousarray(pred_order, dtype=np.int64),
        np.ascontiguousarray(thresholds, dtype=np.float64),
        np.ascontiguousarray(true_ignore_flags, dtype=np.uint8),
        np.ascontiguousarray(annotation_ignore_flags, dtype=np.uint8),
        np.ascontiguousarray(crowd_flags, dtype=np.uint8),
        inclusive=bool(inclusive),
    )
