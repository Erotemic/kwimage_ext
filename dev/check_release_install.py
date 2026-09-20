#!/usr/bin/env python3
"""Audit an installed kwimage_ext release for downstream metric readiness.

This complements ``check_wheel_artifact.py``.  The wheel check validates the
archive structure and ABI3 tag; this script imports the installed artifact,
requires the production Rust capabilities used by kwcoco, and executes a small
area-grid assignment smoke/parity check.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import numpy as np


REQUIRED_CAPABILITIES = {
    'boxes',
    'mask-rle',
    'mask-iou',
    'cpu-nms',
    'soft-nms',
    'coco-assignment',
    'coco-assignment-grid',
}


def _time_call(func, repeat):
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = func()
        samples.append(time.perf_counter() - start)
    return result, {
        'repeat': int(repeat),
        'median_seconds': float(statistics.median(samples)),
        'min_seconds': float(min(samples)),
        'max_seconds': float(max(samples)),
        'samples_seconds': list(map(float, samples)),
    }


def audit(benchmark=False, repeat=50, require_install_root=None):
    import kwimage_ext
    from kwimage_ext import _rust
    from kwimage_ext.algo.assignment import backend_metadata
    from kwimage_ext.algo.assignment import coco_greedy_match
    from kwimage_ext.algo.assignment import coco_greedy_match_grid

    capabilities = set(_rust.capabilities())
    missing = sorted(REQUIRED_CAPABILITIES - capabilities)
    if missing:
        raise AssertionError(f'missing Rust release capabilities: {missing!r}')

    metadata = backend_metadata()
    if metadata.get('kind') != 'rust':
        raise AssertionError(f'assignment backend is not Rust: {metadata!r}')
    if 'coco-assignment-grid' not in set(metadata.get('capabilities', [])):
        raise AssertionError(
            'grid capability missing from assignment metadata: '
            f'{metadata!r}')

    # Two predictions, two truths. p0 prefers t0; p1 can only take t0 at the
    # lower threshold once t0 is already occupied, so it remains unmatched.
    offsets = np.array([0, 2, 3], dtype=np.int64)
    txs = np.array([0, 1, 0], dtype=np.int64)
    ious = np.array([0.90, 0.60, 0.80], dtype=np.float64)
    pred_order = np.array([0, 1], dtype=np.int64)
    thresholds = np.array([0.50, 0.75, 0.90], dtype=np.float64)
    true_ignore = np.zeros(2, dtype=np.uint8)
    crowd = np.zeros(2, dtype=np.uint8)
    area_grid = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=np.uint8)

    grid = coco_greedy_match_grid(
        offsets, txs, ious, pred_order, thresholds,
        true_ignore, area_grid, crowd, inclusive=True)
    repeated = [
        coco_greedy_match(
            offsets, txs, ious, pred_order, thresholds,
            true_ignore, area_grid[ai], crowd, inclusive=True)
        for ai in range(len(area_grid))
    ]
    for idx, repeated_tuple in enumerate(repeated):
        for got, want in zip((part[idx] for part in grid), repeated_tuple):
            np.testing.assert_array_equal(got, want)

    package_version = importlib.metadata.version('kwimage_ext')
    module_version = getattr(kwimage_ext, '__version__', None)
    rust_version = _rust.version() if hasattr(_rust, 'version') else None
    version_values = {package_version, module_version, rust_version}
    if None in version_values or len(version_values) != 1:
        raise AssertionError(
            'kwimage_ext release version mismatch: '
            f'package={package_version!r}, module={module_version!r}, '
            f'rust={rust_version!r}')

    module_paths = {
        'kwimage_ext': Path(kwimage_ext.__file__).resolve(),
        'rust': Path(_rust.__file__).resolve(),
    }
    import kwimage_ext.algo.assignment as assignment_module
    module_paths['assignment'] = Path(assignment_module.__file__).resolve()
    if require_install_root is not None:
        root = Path(require_install_root).resolve()
        leaked = {
            name: str(path)
            for name, path in module_paths.items()
            if not path.is_relative_to(root)
        }
        if leaked:
            raise AssertionError(
                'release audit imported kwimage_ext outside the isolated wheel '
                f'install root {root}: {leaked!r}')
    else:
        root = None

    result = {
        'schema': 'kwimage_ext_release_install_audit_v2',
        'status': 'ok',
        'package_version': package_version,
        'module_version': module_version,
        'module_file': str(module_paths['kwimage_ext']),
        'rust_file': str(module_paths['rust']),
        'assignment_file': str(module_paths['assignment']),
        'required_install_root': None if root is None else str(root),
        'artifact_isolated': root is not None,
        'rust_version': rust_version,
        'capabilities': sorted(capabilities),
        'assignment_backend': metadata,
        'grid_smoke': {
            'areas': int(len(area_grid)),
            'thresholds': thresholds.tolist(),
            'predictions': int(len(pred_order)),
            'truths': int(len(true_ignore)),
            'repeated_parity': True,
        },
    }

    if benchmark:
        # Reuse the same small deterministic call. This is an install smoke
        # timing, not the publication benchmark; downstream kwcoco owns the
        # end-to-end comparison against pycocotools.
        call = lambda: coco_greedy_match_grid(
            offsets, txs, ious, pred_order, thresholds,
            true_ignore, area_grid, crowd, inclusive=True)
        _result, timing = _time_call(call, repeat)
        result['microbenchmark'] = timing
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--require-install-root', type=Path)
    parser.add_argument('--repeat', type=int, default=50)
    args = parser.parse_args(argv)
    result = audit(
        benchmark=args.benchmark, repeat=args.repeat,
        require_install_root=args.require_install_root)
    text = json.dumps(result, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
