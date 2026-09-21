#!/usr/bin/env python3
"""Collect a self-contained kwimage_ext Rust performance evidence bundle.

The front door is intentionally non-interactive.  It runs a correctness gate,
deterministic Python-facing microbenchmarks, Linux perf counters and sampled
profiles when available, records machine/toolchain provenance, snapshots the
relevant source, and writes one .tar.gz bundle suitable for handing to a
reviewer or optimization agent.

Missing optional profilers degrade to status files; correctness or benchmark
failures make the command fail after still writing the evidence bundle.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = REPO_ROOT / 'dev' / 'benchmarks' / 'rust_kernel_benchmarks.py'
DEFAULT_PERF_CASES = (
    'boxes_iou_large',
    'cpu_nms_sparse_1000',
    'mask_iou_fragmented_48',
    'assignment_sparse',
)
PERF_EVENTS = (
    'task-clock',
    'cycles',
    'instructions',
    'branches',
    'branch-misses',
    'cache-references',
    'cache-misses',
    'page-faults',
    'context-switches',
    'cpu-migrations',
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _run(command, *, cwd=REPO_ROOT, env=None, stdout_path=None,
         stderr_path=None, timeout=None):
    start = time.monotonic()
    if stdout_path is None:
        stdout = subprocess.PIPE
    else:
        Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
        stdout = open(stdout_path, 'wb')
    if stderr_path is None:
        stderr = subprocess.PIPE
    else:
        Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)
        stderr = open(stderr_path, 'wb')
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            check=False,
        )
        stdout_text = None if stdout_path else proc.stdout.decode(errors='replace')
        stderr_text = None if stderr_path else proc.stderr.decode(errors='replace')
        return {
            'command': list(map(str, command)),
            'returncode': proc.returncode,
            'duration_seconds': time.monotonic() - start,
            'stdout': stdout_text,
            'stderr': stderr_text,
        }
    except subprocess.TimeoutExpired as ex:
        return {
            'command': list(map(str, command)),
            'returncode': 124,
            'duration_seconds': time.monotonic() - start,
            'stdout': None,
            'stderr': f'timed out after {ex.timeout} seconds',
        }
    finally:
        if stdout_path is not None:
            stdout.close()
        if stderr_path is not None:
            stderr.close()


def _capture_text(bundle, name, command, records, *, env=None):
    path = bundle / name
    result = _run(command, env=env)
    text = result['stdout'] or ''
    if result['stderr']:
        if text:
            text += '\n--- stderr ---\n'
        text += result['stderr']
    text += f'\n--- returncode: {result["returncode"]} ---\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    records.append({k: v for k, v in result.items() if k not in {'stdout', 'stderr'}})
    return result


def _read_optional(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _source_version():
    init_path = REPO_ROOT / 'kwimage_ext' / '__init__.py'
    try:
        tree = ast.parse(init_path.read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(getattr(target, 'id', None) == '__version__' for target in node.targets):
                try:
                    return ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    return None
    return None


def _basic_metadata(argv):
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return {
        'schema_version': 1,
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'argv': argv,
        'repo_root': str(REPO_ROOT),
        'hostname': platform.node(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': sys.version,
        'python_executable': sys.executable,
        'source_version': _source_version(),
        'build_env': {
            key: os.environ.get(key)
            for key in ['RUSTFLAGS', 'CFLAGS', 'CXXFLAGS', 'CARGO_PROFILE_RELEASE_DEBUG']
            if os.environ.get(key) is not None
        },
        'cpu_affinity': affinity,
        'machine_id_sha256': (
            hashlib.sha256((_read_optional('/etc/machine-id') or '').encode()).hexdigest()
            if _read_optional('/etc/machine-id') else None
        ),
        'perf_event_paranoid': _read_optional('/proc/sys/kernel/perf_event_paranoid'),
        'cpu_governor': _read_optional('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
        'intel_pstate_no_turbo': _read_optional('/sys/devices/system/cpu/intel_pstate/no_turbo'),
    }


def _snapshot_source(bundle):
    dest = bundle / 'source'
    dest.mkdir(parents=True, exist_ok=True)
    paths = [
        REPO_ROOT / 'rust' / 'Cargo.toml',
        REPO_ROOT / 'rust' / 'Cargo.lock',
        REPO_ROOT / 'pyproject.toml',
        REPO_ROOT / 'dev' / 'benchmarks' / 'rust_kernel_benchmarks.py',
        REPO_ROOT / 'dev' / 'profiling' / 'profile_rust_kernels.py',
    ]
    paths.extend(sorted((REPO_ROOT / 'rust' / 'src').glob('*.rs')))
    for src in paths:
        if not src.exists():
            continue
        rel = src.relative_to(REPO_ROOT)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)


def _capture_environment(bundle, records, env):
    commands = [
        ('environment/uname.txt', ['uname', '-a']),
        ('environment/lscpu.txt', ['lscpu']),
        ('environment/python-version.txt', [sys.executable, '-VV']),
        ('environment/pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
        ('environment/numpy-config.txt', [
            sys.executable, '-c',
            'import numpy as np; print(np.__version__); np.__config__.show()',
        ]),
        ('environment/rustc.txt', ['rustc', '-vV']),
        ('environment/cargo.txt', ['cargo', '-V']),
        ('environment/perf-version.txt', ['perf', '--version']),
        ('git/head.txt', ['git', 'rev-parse', 'HEAD']),
        ('git/log.txt', ['git', 'log', '-1', '--decorate=short', '--format=fuller']),
        ('git/status.txt', ['git', 'status', '--short']),
        ('git/diff.txt', ['git', 'diff', '--no-ext-diff', '--binary']),
    ]
    for name, command in commands:
        if shutil.which(command[0]) is None:
            (bundle / name).parent.mkdir(parents=True, exist_ok=True)
            (bundle / name).write_text(f'unavailable: {command[0]} not found\n')
            continue
        _capture_text(bundle, name, command, records, env=env)


def _rebuild_profiled(bundle, records, env):
    result_path = bundle / 'build' / 'maturin-develop.txt'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    build_env = dict(env)
    build_env['CARGO_PROFILE_RELEASE_DEBUG'] = '1'
    prior_rustflags = build_env.get('RUSTFLAGS', '').strip()
    frame_pointer_flag = '-C force-frame-pointers=yes'
    build_env['RUSTFLAGS'] = (
        f'{prior_rustflags} {frame_pointer_flag}'.strip()
        if frame_pointer_flag not in prior_rustflags else prior_rustflags
    )
    maturin = shutil.which('maturin')
    if maturin is None:
        result_path.write_text('maturin executable not found on PATH\n')
        result = {
            'command': ['maturin', 'develop', '--release'],
            'returncode': 127,
            'duration_seconds': 0.0,
            'stdout': None,
            'stderr': 'maturin executable not found on PATH',
        }
    else:
        result = _run(
            [maturin, 'develop', '--release'],
            env=build_env,
            stdout_path=result_path,
            stderr_path=result_path.with_suffix('.stderr.txt'),
            timeout=900,
        )
    records.append({k: v for k, v in result.items() if k not in {'stdout', 'stderr'}})
    (bundle / 'build' / 'profile-build-env.json').write_text(json.dumps({
        'CARGO_PROFILE_RELEASE_DEBUG': build_env['CARGO_PROFILE_RELEASE_DEBUG'],
        'RUSTFLAGS': build_env['RUSTFLAGS'],
    }, indent=2) + '\n')
    return result


def _probe_version_state(bundle, records, env):
    # Record source/package/distribution/extension version identities.
    code = r'''import importlib.metadata
import json
import pathlib
import kwimage_ext
from kwimage_ext import _rust

try:
    distribution_version = importlib.metadata.version('kwimage_ext')
except importlib.metadata.PackageNotFoundError:
    distribution_version = None

path = pathlib.Path(_rust.__file__).resolve()
payload = {
    'package_version': getattr(kwimage_ext, '__version__', None),
    'distribution_version': distribution_version,
    'rust_extension_version': _rust.version() if hasattr(_rust, 'version') else None,
    'rust_extension_path': str(path),
}
print(json.dumps(payload, sort_keys=True))
'''
    result = _run([sys.executable, '-c', code], env=env, timeout=60)
    records.append({k: v for k, v in result.items() if k not in {'stdout', 'stderr'}})
    payload = {
        'command_result': {
            k: v for k, v in result.items()
            if k not in {'stdout', 'stderr'}
        },
    }
    if result['returncode'] == 0:
        try:
            payload.update(json.loads((result['stdout'] or '').strip()))
        except json.JSONDecodeError:
            payload['parse_error'] = 'version probe did not emit valid JSON'
    else:
        payload['stderr'] = result.get('stderr')
    source_version = _source_version()
    payload['source_version'] = source_version
    payload['source_matches_package'] = (
        source_version is not None and
        payload.get('package_version') == source_version
    )
    payload['source_matches_extension'] = (
        source_version is not None and
        payload.get('rust_extension_version') == source_version
    )
    payload['distribution_matches_source'] = (
        source_version is not None and
        payload.get('distribution_version') == source_version
    )
    path = bundle / 'correctness' / 'version-probe.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload


def _run_correctness_gate(bundle, records, env, quick, version_state):
    tests = [
        'tests/test_rust_backend.py',
        'tests/test_rust_assignment.py',
        'tests/test_rust_mask_and_softnms.py',
        'tests/test_rust_profiling_harness.py',
    ]
    if quick:
        tests = [
            'tests/test_rust_backend.py',
            'tests/test_rust_profiling_harness.py',
        ]
    full_result = _run(
        [sys.executable, '-m', 'pytest', '-q', *tests],
        env=env,
        stdout_path=bundle / 'correctness' / 'pytest.txt',
        stderr_path=bundle / 'correctness' / 'pytest.stderr.txt',
        timeout=900,
    )
    records.append({k: v for k, v in full_result.items() if k not in {'stdout', 'stderr'}})

    kernel_result = None
    metadata_only_failure = False
    required_passed = full_result['returncode'] == 0
    if not required_passed:
        # A stale dist-info record can disagree with the checkout even though the
        # source package and freshly built extension are current. Keep that failure
        # visible, but distinguish it from a Rust/kernel correctness failure by
        # rerunning everything except the packaging-metadata assertion.
        kernel_result = _run(
            [
                sys.executable, '-m', 'pytest', '-q', *tests,
                '-k', 'not test_release_version_metadata_is_consistent',
            ],
            env=env,
            stdout_path=bundle / 'correctness' / 'kernel-pytest.txt',
            stderr_path=bundle / 'correctness' / 'kernel-pytest.stderr.txt',
            timeout=900,
        )
        records.append({k: v for k, v in kernel_result.items() if k not in {'stdout', 'stderr'}})
        metadata_only_failure = (
            kernel_result['returncode'] == 0 and
            version_state.get('source_matches_package') is True and
            version_state.get('source_matches_extension') is True and
            version_state.get('distribution_matches_source') is False
        )
        required_passed = metadata_only_failure

    status = {
        'full': full_result,
        'kernel_without_distribution_metadata_test': kernel_result,
        'metadata_only_failure': metadata_only_failure,
        'required_passed': required_passed,
        'version_state': version_state,
    }
    status_path = bundle / 'correctness' / 'status.json'
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    return status


def _benchmark_provenance(bundle, benchmark_json):
    # Require benchmark source and compiled extension versions to agree.
    status = {
        'ok': False,
        'source_version': _source_version(),
        'extension_version': None,
        'extension_path': None,
        'extension_sha256': None,
    }
    if benchmark_json.exists():
        payload = json.loads(benchmark_json.read_text())
        ext = payload.get('environment', {}).get('rust_extension', {})
        status['extension_version'] = ext.get('version')
        status['extension_path'] = ext.get('path')
        status['extension_sha256'] = ext.get('sha256')
        status['ok'] = (
            status['source_version'] is not None and
            status['extension_version'] == status['source_version']
        )
    path = bundle / 'benchmarks' / 'provenance.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    return status

def _run_benchmarks(bundle, records, env, quick, cases):
    output = bundle / 'benchmarks' / 'results.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(BENCH_SCRIPT),
        '--backend', 'auto',
        '--output-json', str(output),
    ]
    if cases:
        command.extend(['--cases', ','.join(cases)])
    if quick:
        command.append('--quick')
    result = _run(
        command,
        env=env,
        stdout_path=bundle / 'benchmarks' / 'stdout.txt',
        stderr_path=bundle / 'benchmarks' / 'stderr.txt',
        timeout=900,
    )
    records.append({k: v for k, v in result.items() if k not in {'stdout', 'stderr'}})
    return result, output


def _write_benchmark_csv(bundle, benchmark_json):
    if not benchmark_json.exists():
        return
    payload = json.loads(benchmark_json.read_text())
    rows = payload.get('results', [])
    columns = [
        'case', 'family', 'backend', 'samples', 'loops_per_sample',
        'median_ns', 'mean_ns', 'min_ns', 'p05_ns', 'p95_ns', 'max_ns',
        'stdev_ns', 'work_items', 'work_unit', 'median_ns_per_work_item',
        'result_digest', 'verification_comparator', 'verification_digest',
    ]
    out = bundle / 'benchmarks' / 'results.csv'
    with out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            verification = row.get('verification', {})
            flat = {key: row.get(key) for key in columns}
            flat['verification_comparator'] = verification.get('comparator')
            flat['verification_digest'] = verification.get('digest')
            writer.writerow(flat)


def _profile_command(case, seconds):
    return [
        sys.executable, str(BENCH_SCRIPT),
        '--backend', 'rust',
        '--perf-workload', case,
        '--perf-seconds', str(seconds),
    ]


def _perf_stat(bundle, case, records, env, seconds, repeats):
    outdir = bundle / 'perf' / case
    outdir.mkdir(parents=True, exist_ok=True)
    command = [
        'perf', 'stat', '-x', ';', '-r', str(repeats),
        '-e', ','.join(PERF_EVENTS), '--',
        *_profile_command(case, seconds),
    ]
    result = _run(
        command,
        env=env,
        stdout_path=outdir / 'workload-stat-stdout.txt',
        stderr_path=outdir / 'perf-stat.txt',
        timeout=max(120, int(seconds * repeats * 5 + 60)),
    )
    records.append({k: v for k, v in result.items() if k not in {'stdout', 'stderr'}})
    (outdir / 'perf-stat.status.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def _perf_record(bundle, case, records, env, seconds, frequency):
    outdir = bundle / 'perf' / case
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = outdir / 'perf.data'
    command = [
        'perf', 'record', '-F', str(frequency), '-g', '--call-graph', 'dwarf',
        '-o', str(data_path), '--',
        *_profile_command(case, seconds),
    ]
    result = _run(
        command,
        env=env,
        stdout_path=outdir / 'workload-record-stdout.txt',
        stderr_path=outdir / 'perf-record.txt',
        timeout=max(120, int(seconds * 5 + 60)),
    )
    records.append({k: v for k, v in result.items() if k not in {'stdout', 'stderr'}})
    (outdir / 'perf-record.status.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['returncode'] == 0 and data_path.exists():
        report = _run(
            [
                'perf', 'report', '--stdio', '--no-children',
                '--percent-limit', '0.2',
                '--sort', 'comm,dso,symbol',
                '-i', str(data_path),
            ],
            env=env,
            stdout_path=outdir / 'perf-report.txt',
            stderr_path=outdir / 'perf-report.stderr.txt',
            timeout=120,
        )
        records.append({k: v for k, v in report.items() if k not in {'stdout', 'stderr'}})
    return result


def _capture_extension_artifact(bundle, benchmark_json, records, env):
    if not benchmark_json.exists():
        return
    payload = json.loads(benchmark_json.read_text())
    ext = payload.get('environment', {}).get('rust_extension', {})
    ext_path = Path(ext.get('path') or '')
    if not ext_path.is_file():
        return
    artifact_dir = bundle / 'extension'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied = artifact_dir / ext_path.name
    shutil.copy2(ext_path, copied)
    (artifact_dir / 'metadata.json').write_text(json.dumps({
        **ext,
        'copied_sha256': _sha256(copied),
    }, indent=2, sort_keys=True) + '\n')
    for name, command in [
        ('file.txt', ['file', str(ext_path)]),
        ('ldd.txt', ['ldd', str(ext_path)]),
        ('readelf-sections.txt', ['readelf', '-WS', str(ext_path)]),
        ('readelf-symbols.txt', ['readelf', '-Ws', str(ext_path)]),
    ]:
        if shutil.which(command[0]):
            _capture_text(bundle, f'extension/{name}', command, records, env=env)


def _format_us(ns):
    return f'{ns / 1000.0:.3f}'


def _write_summary(bundle, benchmark_json, correctness_result, perf_results):
    lines = [
        '# kwimage_ext Rust performance evidence',
        '',
        f'Generated: `{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}`',
        '',
        '## Correctness gate',
        '',
    ]
    if correctness_result:
        full = correctness_result['full']
        lines.append(f'- full pytest return code: `{full["returncode"]}`')
        if correctness_result.get('metadata_only_failure'):
            version_state = correctness_result.get('version_state', {})
            lines.extend([
                '- **The full suite failed only because installed distribution metadata is stale.**',
                '- Kernel/parity tests pass when the distribution-metadata assertion is excluded.',
                f'- checkout/package/extension version: `{version_state.get("source_version")}`',
                f'- installed distribution record: `{version_state.get("distribution_version")}`',
                '- This is recorded as an environment packaging issue, not a kernel correctness failure.',
            ])
        kernel_result = correctness_result.get('kernel_without_distribution_metadata_test')
        if kernel_result is not None:
            lines.append(f'- kernel-only pytest return code: `{kernel_result["returncode"]}`')
        lines.append(f'- required correctness gate passed: `{correctness_result.get("required_passed")}`')
    else:
        lines.append('- pytest: skipped')
    lines.append('')
    if benchmark_json.exists():
        payload = json.loads(benchmark_json.read_text())
        metadata_path = bundle / 'metadata.json'
        source_version = None
        if metadata_path.exists():
            source_version = json.loads(metadata_path.read_text()).get('source_version')
        extension_version = payload.get('environment', {}).get('rust_extension', {}).get('version')
        if source_version and extension_version and source_version != extension_version:
            lines.extend([
                '## Provenance warning',
                '',
                f'- **Source version `{source_version}` does not match imported Rust extension version `{extension_version}`.**',
                '- Re-run with `--rebuild-profiled` before using these numbers to optimize this checkout.',
                '',
            ])
        results = payload.get('results', [])
        by_case = {}
        for row in results:
            by_case.setdefault(row['case'], {})[row['backend']] = row
        lines.extend([
            '## Microbenchmarks',
            '',
            '| case | rust median us | comparator | comparator median us | rust/comparator | rust ns/work item |',
            '| --- | ---: | --- | ---: | ---: | ---: |',
        ])
        for case, backend_rows in by_case.items():
            rust = backend_rows.get('rust')
            if rust is None:
                continue
            comparator_name = ''
            comparator = None
            for name in ('legacy', 'python'):
                if name in backend_rows:
                    comparator_name = name
                    comparator = backend_rows[name]
                    break
            if comparator:
                ratio = rust['median_ns'] / comparator['median_ns']
                comp_us = _format_us(comparator['median_ns'])
                ratio_text = f'{ratio:.3f}x'
            else:
                comp_us = ''
                ratio_text = ''
            ns_item = rust.get('median_ns_per_work_item')
            ns_item_text = f'{ns_item:.4f}' if ns_item is not None else ''
            lines.append(
                f'| `{case}` | {_format_us(rust["median_ns"])} | {comparator_name} | '
                f'{comp_us} | {ratio_text} | {ns_item_text} |')
        lines.extend(['', 'Lower `rust/comparator` is faster.', ''])
    lines.extend(['## perf collection', ''])
    if not perf_results:
        lines.append('- perf: skipped or unavailable')
    else:
        for case, status in perf_results.items():
            lines.append(
                f'- `{case}`: stat={status.get("stat")}, record={status.get("record")}')
    lines.extend([
        '',
        '## Important provenance',
        '',
        '- `benchmarks/results.json` records the exact imported Rust extension path and SHA-256.',
        '- `extension/` contains that compiled extension and ELF metadata when available.',
        '- `source/` snapshots the Rust kernels and benchmark/profiling front doors.',
        '- `git/` records HEAD, dirty status, and the working-tree diff.',
        '- Raw perf counter output and sampled reports are under `perf/<case>/`.',
        '- Compare timing results only on equivalent machines/toolchains/workloads.',
        '',
    ])
    (bundle / 'summary.md').write_text('\n'.join(lines))


def _write_bundle_readme(bundle):
    (bundle / 'README.txt').write_text(
        'kwimage_ext Rust profiling evidence bundle\n'
        '========================================\n\n'
        'Start with summary.md and benchmarks/results.json.\n'
        'Raw Linux perf captures live under perf/<case>/.\n'
        'The imported extension binary and its SHA-256 are under extension/.\n'
        'Source files used to interpret the measurements are under source/.\n'
        'Command statuses are recorded in commands.json.\n'
    )


def _tar_bundle(bundle, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as tar:
        tar.add(bundle, arcname=bundle.name)


def _parse_cases(text):
    if not text:
        return None
    return [part.strip() for part in text.split(',') if part.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Output .tar.gz path.')
    parser.add_argument('--quick', action='store_true', help='Short smoke campaign.')
    parser.add_argument('--skip-tests', action='store_true')
    parser.add_argument('--no-perf', action='store_true')
    parser.add_argument('--rebuild-profiled', action='store_true',
                        help='Run maturin develop --release with debug info and frame pointers first.')
    parser.add_argument('--cases', help='Comma-separated benchmark cases; default all.')
    parser.add_argument('--perf-cases', default=','.join(DEFAULT_PERF_CASES),
                        help='Comma-separated cases for perf stat/record.')
    parser.add_argument('--perf-seconds', type=float, default=2.0)
    parser.add_argument('--perf-repeats', type=int, default=3)
    parser.add_argument('--perf-frequency', type=int, default=999)
    parser.add_argument('--keep-directory', action='store_true')
    args = parser.parse_args(argv)

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    output = args.output
    if output is None:
        output = REPO_ROOT / f'kwimage-ext-rust-profile-{stamp}.tar.gz'
    output = output.expanduser().resolve()
    if output.suffixes[-2:] != ['.tar', '.gz']:
        parser.error('--output must end in .tar.gz')

    env = dict(os.environ)
    env['KWIMAGE_EXT_FORCE_RUST'] = '1'
    env.pop('KWIMAGE_EXT_FORCE_LEGACY', None)

    commands = []
    hard_failure = False
    perf_results = {}
    temp_root = Path(tempfile.mkdtemp(prefix='kwimage-ext-profile-'))
    bundle = temp_root / f'kwimage-ext-rust-profile-{stamp}'
    bundle.mkdir()
    try:
        metadata = _basic_metadata(sys.argv if argv is None else [sys.argv[0], *argv])
        (bundle / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
        _write_bundle_readme(bundle)
        _snapshot_source(bundle)
        _capture_environment(bundle, commands, env)

        if args.rebuild_profiled:
            rebuild = _rebuild_profiled(bundle, commands, env)
            if rebuild['returncode'] != 0:
                hard_failure = True

        version_state = None
        if not hard_failure:
            version_state = _probe_version_state(bundle, commands, env)
            if (
                version_state.get('source_matches_package') is not True or
                version_state.get('source_matches_extension') is not True
            ):
                hard_failure = True

        correctness_result = None
        if not args.skip_tests and not hard_failure:
            correctness_result = _run_correctness_gate(
                bundle, commands, env, args.quick, version_state)
            if correctness_result['required_passed'] is not True:
                hard_failure = True

        benchmark_result = None
        benchmark_json = bundle / 'benchmarks' / 'results.json'
        if not hard_failure:
            benchmark_result, benchmark_json = _run_benchmarks(
                bundle, commands, env, args.quick, _parse_cases(args.cases))
            if benchmark_result['returncode'] != 0 or not benchmark_json.exists():
                hard_failure = True

        if benchmark_json.exists():
            provenance = _benchmark_provenance(bundle, benchmark_json)
            if provenance['ok'] is not True:
                hard_failure = True

        _write_benchmark_csv(bundle, benchmark_json)
        _capture_extension_artifact(bundle, benchmark_json, commands, env)

        can_perf = (
            not args.no_perf and not hard_failure and platform.system() == 'Linux'
            and shutil.which('perf') is not None
        )
        if can_perf:
            for case in _parse_cases(args.perf_cases) or DEFAULT_PERF_CASES:
                seconds = min(args.perf_seconds, 0.5) if args.quick else args.perf_seconds
                repeats = 1 if args.quick else args.perf_repeats
                stat = _perf_stat(bundle, case, commands, env, seconds, repeats)
                record = _perf_record(
                    bundle, case, commands, env, seconds, args.perf_frequency)
                perf_results[case] = {
                    'stat': stat['returncode'],
                    'record': record['returncode'],
                }
        else:
            reason = []
            if args.no_perf:
                reason.append('--no-perf')
            if hard_failure:
                reason.append('prior hard failure')
            if platform.system() != 'Linux':
                reason.append(f'platform={platform.system()}')
            if shutil.which('perf') is None:
                reason.append('perf unavailable')
            (bundle / 'perf' / 'SKIPPED.txt').parent.mkdir(parents=True, exist_ok=True)
            (bundle / 'perf' / 'SKIPPED.txt').write_text(', '.join(reason) + '\n')

        (bundle / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        _write_summary(bundle, benchmark_json, correctness_result, perf_results)
        _tar_bundle(bundle, output)
        digest = _sha256(output)
        print(f'evidence bundle: {output}')
        print(f'sha256: {digest}')
        if hard_failure:
            print('evidence bundle created, but a required correctness/build/benchmark check failed')
            return 1
        return 0
    finally:
        if args.keep_directory:
            print(f'evidence directory: {bundle}')
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
