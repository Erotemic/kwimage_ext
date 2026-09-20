#!/usr/bin/env python3
"""Remove in-place binary artifacts before rebuilding the legacy oracle.

The public compatibility modules deliberately occupy the historical import
names.  Old editable-build binaries with those names can otherwise shadow the
Python shims before import dispatch gets a chance to choose a backend.

This helper also removes the explicitly renamed ``*_legacy`` binaries so a
parity run cannot accidentally compare Rust against an extension left over
from an older source tree.
"""
from __future__ import annotations

import importlib.machinery
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / 'kwimage_ext'
ARTIFACT_STEMS = {
    PACKAGE_ROOT / 'structs' / '_boxes_backend': {
        'cython_boxes',
        'cython_boxes_legacy',
    },
    PACKAGE_ROOT / 'structs' / '_mask_backend': {
        'cython_mask',
        'cython_mask_legacy',
    },
    PACKAGE_ROOT / 'algo' / '_nms_backend': {
        'cpu_nms',
        'cpu_nms_legacy',
        'cpu_soft_nms',
        'cpu_soft_nms_legacy',
    },
}


def is_extension_artifact(path: Path, stems: set[str]) -> bool:
    name = path.name
    return any(
        name.startswith(stem + '.') and name.endswith(suffix)
        for stem in stems
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def main() -> int:
    removed = []
    for directory, stems in ARTIFACT_STEMS.items():
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and is_extension_artifact(path, stems):
                path.unlink()
                removed.append(path.relative_to(PACKAGE_ROOT.parent))
    if removed:
        print('removed stale extension artifacts:')
        for path in removed:
            print(f'  {path}')
    else:
        print('no stale extension artifacts found')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
