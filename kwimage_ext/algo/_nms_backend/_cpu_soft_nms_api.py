"""Rust-first compatibility wrapper for CPU Soft-NMS."""
from __future__ import annotations

from pathlib import Path

from kwimage_ext._rust_dispatch import backend_info, import_backend

_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = import_backend(
            'kwimage_ext._rust',
            'kwimage_ext.algo._nms_backend.cpu_soft_nms_legacy',
            required_symbols={'soft_nms'},
            legacy_extension_stem='cpu_soft_nms_legacy',
            legacy_search_dir=str(Path(__file__).parent),
        )
    return _BACKEND


def backend_metadata():
    return backend_info(_backend())


def soft_nms(
    ltrb,
    scores,
    thresh=0.001,
    overlap_thresh=0.3,
    sigma=0.5,
    bias=0.0,
    method=0,
):
    return _backend().soft_nms(
        ltrb,
        scores,
        thresh=thresh,
        overlap_thresh=overlap_thresh,
        sigma=sigma,
        bias=bias,
        method=method,
    )
