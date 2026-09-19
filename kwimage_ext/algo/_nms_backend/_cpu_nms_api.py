"""Rust-first compatibility wrapper for CPU NMS."""
from __future__ import annotations

from pathlib import Path

from kwimage_ext._rust_dispatch import backend_info, import_backend

_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = import_backend(
            'kwimage_ext._rust',
            'kwimage_ext.algo._nms_backend.cpu_nms_legacy',
            required_symbols={'cpu_nms'},
            legacy_extension_stem='cpu_nms_legacy',
            legacy_search_dir=str(Path(__file__).parent),
        )
    return _BACKEND


def backend_metadata():
    return backend_info(_backend())


def cpu_nms(ltrb, scores, thresh, bias=0.0):
    return _backend().cpu_nms(ltrb, scores, thresh, bias=bias)
