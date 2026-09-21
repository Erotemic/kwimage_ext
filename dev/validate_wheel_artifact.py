#!/usr/bin/env python3
"""Validate a built wheel structurally and from an isolated artifact install.

Unlike editable/source-tree validation, this script structurally inspects every
wheel and, when the wheel is compatible with the current host, installs it into
a fresh temporary target directory and runs the release capability audit from a
different working directory.  Incompatible sibling artifacts (notably
musllinux wheels built from an Ubuntu host) are runtime-smoke-tested by
cibuildwheel in their target container instead.  This prevents the checkout
(or an editable ``.pth`` file) from making an incomplete wheel look healthy.
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


def _wheel_is_compatible_with_host(wheel: Path) -> bool:
    """Return whether this interpreter can install the wheel tag.

    Release Linux jobs intentionally build both manylinux and musllinux wheels.
    The GitHub host itself is Ubuntu/glibc, so the musllinux sibling cannot be
    installed there even though cibuildwheel already tested it inside its
    Alpine container.  Use packaging's tag machinery (falling back to pip's
    vendored copy) instead of treating that expected incompatibility as a
    release failure.
    """
    try:
        from packaging import tags
        from packaging.utils import parse_wheel_filename
    except ImportError:
        from pip._vendor.packaging import tags
        from pip._vendor.packaging.utils import parse_wheel_filename

    _, _, _, wheel_tags = parse_wheel_filename(wheel.name)
    return not set(wheel_tags).isdisjoint(tags.sys_tags())


def _run(cmd, **kwargs):
    return subprocess.run(cmd, check=False, text=True, **kwargs)


def validate_wheel(wheel: Path, benchmark: bool = False) -> dict:
    wheel = wheel.resolve()
    errors = check_wheel(wheel)
    if errors:
        raise AssertionError('\n'.join(errors))

    host_compatible = _wheel_is_compatible_with_host(wheel)
    if not host_compatible:
        return {
            'wheel': str(wheel),
            'wheel_name': wheel.name,
            'host_compatible': False,
            'isolated_install': False,
            'runtime_validation': 'cibuildwheel-container',
        }

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
        # Do not inherit the caller's PYTHONPATH.  The exact wheel and its
        # declared dependencies were installed into ``target`` above; the
        # release audit must not gain modules from an editable checkout or
        # another developer-controlled path.
        env['PYTHONPATH'] = str(target)
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
        data['host_compatible'] = True
        data['isolated_install'] = True
        data['runtime_validation'] = 'isolated-host-install'
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
            if info['isolated_install']:
                print(f'OK isolated wheel: {wheel}')
            else:
                print(
                    f'OK structurally; runtime tested by cibuildwheel container: '
                    f'{wheel}'
                )

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
