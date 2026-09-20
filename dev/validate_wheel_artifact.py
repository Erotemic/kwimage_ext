#!/usr/bin/env python3
"""Validate a built wheel structurally and from an isolated artifact install.

Unlike editable/source-tree validation, this script installs each wheel into a
fresh temporary target directory, runs the release capability audit from a
different working directory, and requires every imported ``kwimage_ext`` file
to resolve underneath that target.  This prevents the checkout (or an editable
``.pth`` file) from making an incomplete wheel look healthy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from check_wheel_artifact import check_wheel


REPO_DPATH = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO_DPATH / 'dev' / 'check_release_install.py'


def _run(cmd, **kwargs):
    return subprocess.run(cmd, check=False, text=True, **kwargs)


def validate_wheel(wheel: Path, benchmark: bool = False) -> dict:
    wheel = wheel.resolve()
    errors = check_wheel(wheel)
    if errors:
        raise AssertionError('\n'.join(errors))

    with tempfile.TemporaryDirectory(prefix='kwimage_ext_wheel_audit_') as temp:
        temp_dpath = Path(temp)
        target = temp_dpath / 'site'
        target.mkdir()

        install_cmd = [
            sys.executable, '-m', 'pip', 'install', '--target',
            str(target), str(wheel),
        ]
        install = _run(
            install_cmd,
            cwd=temp_dpath,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if install.returncode:
            raise RuntimeError(
                f'isolated wheel install failed for {wheel.name}:\n{install.stdout}')

        audit_json = temp_dpath / 'release_install.json'
        audit_cmd = [
            sys.executable,
            str(AUDIT_SCRIPT),
            '--require-install-root', str(target),
            '--output', str(audit_json),
        ]
        if benchmark:
            audit_cmd.append('--benchmark')

        env = os.environ.copy()
        # Put the isolated wheel first.  Dependencies such as NumPy may still
        # come from the caller's environment, but kwimage_ext itself must come
        # entirely from ``target`` and check_release_install enforces that.
        old_pythonpath = env.get('PYTHONPATH')
        env['PYTHONPATH'] = (
            str(target) if not old_pythonpath
            else str(target) + os.pathsep + old_pythonpath
        )
        env['PYTHONNOUSERSITE'] = '1'
        env['KWIMAGE_EXT_FORCE_RUST'] = '1'
        audit = _run(
            audit_cmd,
            cwd=temp_dpath,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if audit.returncode:
            raise RuntimeError(
                f'isolated release audit failed for {wheel.name}:\n{audit.stdout}')
        data = json.loads(audit_json.read_text())
        data['wheel'] = str(wheel)
        data['wheel_name'] = wheel.name
        data['isolated_install'] = True
        return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('wheels', nargs='+', type=Path)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)

    results = []
    failures = []
    for wheel in args.wheels:
        try:
            info = validate_wheel(wheel, benchmark=args.benchmark)
        except Exception as ex:
            failures.append(f'{wheel}: {ex}')
        else:
            results.append(info)
            print(f'OK isolated wheel: {wheel}')

    payload = {
        'schema': 'kwimage_ext_wheel_validation_v1',
        'status': 'ok' if not failures else 'error',
        'results': results,
        'errors': failures,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + '\n')

    if failures:
        for error in failures:
            print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
