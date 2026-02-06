"""Python shim for mask backend."""
from __future__ import annotations

from pathlib import Path

from kwimage_ext._rust_dispatch import import_backend

_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = import_backend(
            "kwimage_ext._rust",
            "kwimage_ext.structs._mask_backend.cython_mask_legacy",
            legacy_extension_stem="cython_mask",
            legacy_search_dir=str(Path(__file__).parent),
        )
    return _BACKEND


def __getattr__(name):
    if name in {
        "encode", "decode", "merge", "area", "iou", "toBbox", "frBbox",
        "frPoly", "frUncompressedRLE", "frPyObjects",
        "_rle_bytes_to_array", "_rle_array_to_bytes",
    }:
        return getattr(_backend(), name)
    raise AttributeError(name)


__all__ = [
    "encode", "decode", "merge", "area", "iou", "toBbox", "frBbox",
    "frPoly", "frUncompressedRLE", "frPyObjects",
    "_rle_bytes_to_array", "_rle_array_to_bytes",
]
