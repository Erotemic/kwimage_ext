#!/usr/bin/env python3
"""Validate that a release wheel is Rust-first and genuinely abi3.

This intentionally inspects the built artifact rather than the checkout.  It
catches missing extension payloads, accidentally bundled legacy binary stacks,
and regressions back to per-minor CPython wheel tags.
"""
from __future__ import annotations

import argparse
import email.parser
import re
import sys
import zipfile
from pathlib import Path


LEGACY_BINARY_STEMS = (
    'cython_boxes',
    'cython_boxes_legacy',
    'cython_mask',
    'cython_mask_legacy',
    'cpu_nms',
    'cpu_nms_legacy',
    'cpu_soft_nms',
    'cpu_soft_nms_legacy',
)
REQUIRED_PYTHON_FILES = {
    'kwimage_ext/__init__.py',
    'kwimage_ext/_rust_dispatch.py',
    'kwimage_ext/structs/_boxes_backend/_api.py',
    'kwimage_ext/structs/_mask_backend/_api.py',
    'kwimage_ext/algo/_nms_backend/_cpu_nms_api.py',
    'kwimage_ext/algo/_nms_backend/_cpu_soft_nms_api.py',
}


def _is_binary_member(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(('.so', '.pyd', '.dll', '.dylib'))


def check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    if not re.search(r'-cp310-abi3-', path.name):
        errors.append(
            f'{path.name}: expected a cp310-abi3 wheel tag, not a per-minor ABI tag')

    with zipfile.ZipFile(path) as zfile:
        names = set(zfile.namelist())
        rust_members = [
            name for name in names
            if name.startswith('kwimage_ext/_rust') and _is_binary_member(name)
        ]
        if len(rust_members) != 1:
            errors.append(
                f'{path.name}: expected exactly one kwimage_ext._rust binary, '
                f'found {rust_members!r}')

        legacy_members = []
        for name in names:
            base = Path(name).name
            if _is_binary_member(name) and any(
                base.startswith(stem + '.') for stem in LEGACY_BINARY_STEMS
            ):
                legacy_members.append(name)
        if legacy_members:
            errors.append(
                f'{path.name}: production wheel contains legacy binaries: '
                f'{legacy_members!r}')

        missing = sorted(REQUIRED_PYTHON_FILES - names)
        if missing:
            errors.append(
                f'{path.name}: missing compatibility modules: {missing!r}')

        wheel_meta_names = [
            name for name in names if name.endswith('.dist-info/WHEEL')]
        if len(wheel_meta_names) != 1:
            errors.append(
                f'{path.name}: expected one WHEEL metadata file, '
                f'found {wheel_meta_names!r}')
        else:
            text = zfile.read(wheel_meta_names[0]).decode('utf8')
            tags = [
                line.split(':', 1)[1].strip()
                for line in text.splitlines()
                if line.startswith('Tag:')
            ]
            if not tags or not all(tag.startswith('cp310-abi3-') for tag in tags):
                errors.append(
                    f'{path.name}: WHEEL metadata does not advertise only '
                    f'cp310-abi3 tags: {tags!r}')

        metadata_names = [
            name for name in names if name.endswith('.dist-info/METADATA')]
        if len(metadata_names) != 1:
            errors.append(
                f'{path.name}: expected one METADATA file, found {metadata_names!r}')
        else:
            parser = email.parser.Parser()
            msg = parser.parsestr(zfile.read(metadata_names[0]).decode('utf8'))
            requires_python = msg.get('Requires-Python')
            if requires_python != '>=3.10':
                errors.append(
                    f'{path.name}: expected Requires-Python >=3.10, '
                    f'got {requires_python!r}')

    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('wheels', nargs='+', type=Path)
    args = parser.parse_args(argv)

    errors = []
    for wheel in args.wheels:
        if not wheel.exists():
            errors.append(f'missing wheel: {wheel}')
            continue
        wheel_errors = check_wheel(wheel)
        if wheel_errors:
            errors.extend(wheel_errors)
        else:
            print(f'OK: {wheel}')

    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
