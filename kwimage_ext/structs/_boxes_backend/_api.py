"""Compatibility frontend for the box extension backend."""
from __future__ import annotations

from pathlib import Path

from kwimage_ext._rust_dispatch import backend_info, import_backend

_REQUIRED = {
    'bbox_ious_c', 'bbox_overlaps', 'bbox_intersections',
    'anchor_intersections', 'bbox_intersections_self', 'bbox_similarities',
}
_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = import_backend(
            'kwimage_ext._rust',
            'kwimage_ext.structs._boxes_backend.cython_boxes_legacy',
            required_symbols=_REQUIRED,
            legacy_extension_stem='cython_boxes_legacy',
            legacy_search_dir=str(Path(__file__).parent),
        )
    return _BACKEND


def backend_metadata():
    return backend_info(_backend())


def bbox_ious_c(boxes, query_boxes, bias=1.0):
    return _backend().bbox_ious_c(boxes, query_boxes, bias=bias)


def bbox_overlaps(boxes, query_boxes):
    return _backend().bbox_overlaps(boxes, query_boxes)


def bbox_intersections(boxes, query_boxes):
    return _backend().bbox_intersections(boxes, query_boxes)


def anchor_intersections(anchors, query_boxes):
    return _backend().anchor_intersections(anchors, query_boxes)


def bbox_intersections_self(boxes):
    return _backend().bbox_intersections_self(boxes)


def bbox_similarities(boxes, query_boxes):
    return _backend().bbox_similarities(boxes, query_boxes)


__all__ = [
    'bbox_ious_c', 'bbox_overlaps', 'bbox_intersections',
    'anchor_intersections', 'bbox_intersections_self', 'bbox_similarities',
    'backend_metadata',
]
