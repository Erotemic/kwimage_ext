"""Compatibility frontend for the COCO-style mask backend.

This module also normalizes historically ambiguous Python inputs before they
cross the binary boundary.  The normalization fixes the inherited one-dimensional
bbox reshape bug and avoids treating an ``N x 2`` polygon as a list of bboxes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from kwimage_ext._rust_dispatch import backend_info, import_backend

_REQUIRED = {
    'encode', 'decode', 'merge', 'area', 'iou', 'toBbox', 'frBbox',
    'frPoly', 'frUncompressedRLE', '_rle_bytes_to_array',
    '_rle_array_to_bytes',
}
_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = import_backend(
            'kwimage_ext._rust',
            'kwimage_ext.structs._mask_backend.cython_mask_legacy',
            required_symbols=_REQUIRED,
            legacy_extension_stem='cython_mask_legacy',
            legacy_search_dir=str(Path(__file__).parent),
        )
    return _BACKEND


def backend_metadata():
    return backend_info(_backend())


def encode(mask):
    mask = np.asarray(mask, dtype=np.uint8, order='F')
    if mask.ndim == 2:
        mask = mask[:, :, None]
    if mask.ndim != 3:
        raise ValueError(f'encode expects HxW or HxWxN uint8 data, got shape={mask.shape!r}')
    return _backend().encode(mask)


def decode(rleObjs):
    return _backend().decode(rleObjs)


def merge(rleObjs, intersect=0):
    return _backend().merge(rleObjs, intersect=intersect)


def area(rleObjs):
    return _backend().area(rleObjs)


def _normalize_iou_arg(objs):
    if isinstance(objs, np.ndarray):
        if objs.ndim == 1:
            if objs.size != 4:
                raise ValueError('1D ndarray input to iou must contain exactly one bbox')
            objs = objs.reshape(1, 4)
        if objs.ndim != 2 or objs.shape[1] != 4:
            raise ValueError('ndarray input to iou is only for bounding boxes with shape Nx4')
        return np.asarray(objs, dtype=np.float64)
    if isinstance(objs, tuple):
        objs = list(objs)
    if isinstance(objs, list):
        if not objs:
            return objs
        if all(isinstance(item, dict) for item in objs):
            return objs
        arr = np.asarray(objs)
        if arr.ndim == 1:
            if arr.size != 4:
                raise ValueError('flat list input to iou must contain exactly one bbox')
            arr = arr.reshape(1, 4)
        if arr.ndim == 2 and arr.shape[1] == 4:
            return np.asarray(arr, dtype=np.float64)
        raise ValueError('list input to iou must be Nx4 bboxes or a list of RLE dictionaries')
    raise TypeError(f'unsupported iou input type: {type(objs)!r}')


def iou(dt, gt, pyiscrowd):
    dt = _normalize_iou_arg(dt)
    gt = _normalize_iou_arg(gt)
    if len(pyiscrowd) != len(gt):
        raise ValueError(
            f'iscrowd length ({len(pyiscrowd)}) must equal number of ground truths ({len(gt)})')
    return _backend().iou(dt, gt, pyiscrowd)


def toBbox(rleObjs):
    return _backend().toBbox(rleObjs)


def frBbox(bb, h, w):
    bb = np.asarray(bb, dtype=np.float64)
    if bb.ndim == 1:
        if bb.size != 4:
            raise ValueError('bbox must have four coordinates')
        bb = bb.reshape(1, 4)
    if bb.ndim != 2 or bb.shape[1] != 4:
        raise ValueError('bbox input must have shape Nx4')
    return _backend().frBbox(bb, h, w)


def frPoly(poly, h, w):
    # Preserve one output RLE per input polygon part.  Degenerate parts are
    # represented as explicit empty masks instead of being dropped; callers
    # often rely on positional correspondence between inputs and returned RLEs.
    result = []
    backend = _backend()
    for part in poly:
        arr = np.asarray(part, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[1] != 2:
                raise ValueError('2D polygon arrays must have shape Nx2')
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise ValueError('polygon data must be flat or Nx2')
        if arr.size % 2:
            raise ValueError('polygon coordinate vector must have even length')
        if arr.size < 6:
            empty = {'size': [int(h), int(w)], 'counts': [int(h) * int(w)]}
            result.append(backend.frUncompressedRLE([empty], h, w)[0])
        else:
            result.append(backend.frPoly([arr.tolist()], h, w)[0])
    return result


def frUncompressedRLE(ucRles, h, w):
    return _backend().frUncompressedRLE(ucRles, h, w)


def frPyObjects(pyobj, h, w):
    """Robust version of the historical pycocotools input dispatcher."""
    if isinstance(pyobj, np.ndarray):
        arr = np.asarray(pyobj)
        if arr.ndim == 1 and arr.size == 4:
            return frBbox(arr, h, w)[0]
        if arr.ndim == 2 and arr.shape[1] == 4:
            return frBbox(arr, h, w)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return frPoly([arr], h, w)[0]
        raise ValueError(f'unsupported ndarray shape for frPyObjects: {arr.shape!r}')

    if isinstance(pyobj, dict):
        if 'counts' not in pyobj or 'size' not in pyobj:
            raise ValueError('RLE dictionaries require counts and size')
        if isinstance(pyobj['counts'], (list, tuple, np.ndarray)):
            return frUncompressedRLE([pyobj], h, w)[0]
        return pyobj

    if not isinstance(pyobj, (list, tuple)):
        raise TypeError(f'unsupported frPyObjects input type: {type(pyobj)!r}')
    pyobj = list(pyobj)
    if not pyobj:
        return []

    if all(isinstance(item, dict) for item in pyobj):
        if all(isinstance(item.get('counts'), (list, tuple, np.ndarray)) for item in pyobj):
            return frUncompressedRLE(pyobj, h, w)
        return pyobj

    # Flat numeric object: one bbox or one polygon.
    if all(np.isscalar(item) for item in pyobj):
        if len(pyobj) == 4:
            return frBbox(pyobj, h, w)[0]
        if len(pyobj) >= 6 and len(pyobj) % 2 == 0:
            return frPoly([pyobj], h, w)[0]
        raise ValueError('flat object must be a four-value bbox or polygon with >=3 vertices')

    # Nx2 is unambiguously a single polygon, including a quadrilateral.
    arr = np.asarray(pyobj)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return frPoly([arr], h, w)[0]

    # A list of flattened parts or bboxes.
    lengths = [len(item) for item in pyobj]
    if all(length == 4 for length in lengths):
        return frBbox(pyobj, h, w)
    if all(length >= 6 and length % 2 == 0 for length in lengths):
        return frPoly(pyobj, h, w)
    raise ValueError('nested input must be Nx4 bboxes, Nx2 polygon points, or flattened polygon parts')


def _rle_bytes_to_array(counts_str):
    return _backend()._rle_bytes_to_array(counts_str)


def _rle_array_to_bytes(counts_arr):
    return _backend()._rle_array_to_bytes(np.asarray(counts_arr, dtype=np.uint32))


__all__ = [
    'encode', 'decode', 'merge', 'area', 'iou', 'toBbox', 'frBbox',
    'frPoly', 'frUncompressedRLE', 'frPyObjects',
    '_rle_bytes_to_array', '_rle_array_to_bytes', 'backend_metadata',
]
