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
CANONICAL_BENCHMARK_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_PERF_CASES = (
    'boxes_iou_large',
    'cpu_nms_sparse_1000',
    'soft_nms_gaussian_512',
    'mask_encode_sparse_128x128x16',
    'mask_encode_512x512x4',
    'mask_encode_structured_512x512x4',
    'mask_area_structured_256x256x32',
    'mask_area_fragmented_256x256x32',
    'mask_merge_structured_256x256x32',
    'mask_merge_fragmented_256x256x32',
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

PORTABLE_PROFILE_RUSTFLAGS = '-C target-cpu=generic'


BENCHMARK_THREAD_ENV = {
    # kwimage_ext kernels are single-threaded. Imported numeric libraries may
    # otherwise start worker pools that perf counts as part of the process,
    # obscuring the kernel counters and increasing run-to-run noise.
    'OPENBLAS_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
}


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
        'benchmark_thread_env': dict(BENCHMARK_THREAD_ENV),
        'build_env': {
            key: os.environ.get(key)
            for key in [
                'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS',
                'CFLAGS', 'CXXFLAGS', 'CARGO_PROFILE_RELEASE_DEBUG',
            ]
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
        REPO_ROOT / 'setup.py',
        REPO_ROOT / 'dev' / 'build_legacy.sh',
        REPO_ROOT / 'dev' / 'clean_stale_legacy_artifacts.py',
        REPO_ROOT / 'dev' / 'benchmarks' / 'rust_kernel_benchmarks.py',
        REPO_ROOT / 'dev' / 'profiling' / 'README.md',
        REPO_ROOT / 'dev' / 'profiling' / 'profile_rust_kernels.py',
    ]
    paths.extend(sorted((REPO_ROOT / 'rust' / 'src').glob('*.rs')))
    paths.extend(sorted((REPO_ROOT / 'tests').glob('test_rust_*.py')))
    backend_parity = REPO_ROOT / 'tests' / 'test_backend_parity.py'
    if backend_parity.exists():
        paths.append(backend_parity)
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


def _portable_profile_build_env(env):
    """Create a production-like portable Rust build environment.

    Profiling evidence must describe the ISA policy used by published wheels,
    not whatever CPU happens to run the benchmark.  In particular, never
    inherit ambient ``target-cpu=native`` / ``target-feature`` flags.

    ``target-cpu=generic`` is rustc's portable target baseline.  Debug info is
    enabled through the Cargo profile rather than architecture-affecting
    rustflags, and perf uses DWARF call graphs so frame pointers are not forced
    into the timed binary.
    """
    build_env = dict(env)
    ignored = {}
    architecture_flag_keys = {
        'RUSTFLAGS',
        'CARGO_ENCODED_RUSTFLAGS',
        'CARGO_BUILD_RUSTFLAGS',
    }
    for key in list(build_env):
        if (
            key in architecture_flag_keys or
            (key.startswith('CARGO_TARGET_') and key.endswith('_RUSTFLAGS'))
        ):
            ignored[key] = build_env.pop(key)
    build_env['CARGO_PROFILE_RELEASE_DEBUG'] = '1'
    build_env['RUSTFLAGS'] = PORTABLE_PROFILE_RUSTFLAGS
    return build_env, ignored


def _rebuild_profiled(bundle, records, env):
    result_path = bundle / 'build' / 'maturin-develop.txt'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    build_env, ignored_rustflags = _portable_profile_build_env(env)
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
        'policy': 'portable-published-wheel-isa',
        'target_cpu': 'generic',
        'CARGO_PROFILE_RELEASE_DEBUG': build_env['CARGO_PROFILE_RELEASE_DEBUG'],
        'RUSTFLAGS': build_env['RUSTFLAGS'],
        'ignored_ambient_rustflags': ignored_rustflags,
        'notes': [
            'Architecture-affecting ambient Cargo/Rust flags are intentionally ignored.',
            'Debug info does not change the intended CPU ISA baseline.',
            'No frame-pointer forcing is used in the timed binary.',
        ],
    }, indent=2, sort_keys=True) + '\n')
    return result


def _legacy_source_hashes():
    paths = [
        REPO_ROOT / 'setup.py',
        REPO_ROOT / 'CMakeLists.txt',
        REPO_ROOT / 'kwimage_ext' / 'structs' / '_boxes_backend' / 'cython_boxes.pyx',
        REPO_ROOT / 'kwimage_ext' / 'structs' / '_mask_backend' / 'cython_mask.pyx',
        REPO_ROOT / 'kwimage_ext' / 'structs' / '_mask_backend' / 'maskApi.c',
        REPO_ROOT / 'kwimage_ext' / 'structs' / '_mask_backend' / 'maskApi.h',
        REPO_ROOT / 'kwimage_ext' / 'algo' / '_nms_backend' / 'cpu_nms.pyx',
        REPO_ROOT / 'kwimage_ext' / 'algo' / '_nms_backend' / 'cpu_soft_nms.pyx',
    ]
    return {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in paths if path.is_file()
    }


def _rebuild_legacy_reference(bundle, records, env):
    """Build and fingerprint the historical Cython/C comparator in-place."""
    build_dir = bundle / 'build'
    build_dir.mkdir(parents=True, exist_ok=True)
    build_env = dict(env)
    ignored_build_flags = {}
    for key in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'CMAKE_ARGS'):
        if key in build_env:
            ignored_build_flags[key] = build_env.pop(key)
    # Pin the helper to the same interpreter that will execute the benchmark.
    # This avoids accidentally building the comparator with a different
    # ``python`` found earlier on PATH.
    build_env['KWIMAGE_EXT_LEGACY_PYTHON'] = sys.executable
    result = _run(
        ['bash', str(REPO_ROOT / 'dev' / 'build_legacy.sh')],
        env=build_env,
        stdout_path=build_dir / 'legacy-reference.txt',
        stderr_path=build_dir / 'legacy-reference.stderr.txt',
        timeout=900,
    )
    records.append({k: v for k, v in result.items() if k not in {'stdout', 'stderr'}})

    artifact_probe = {}
    if result['returncode'] == 0:
        code = r'''import hashlib
import importlib
import json
from pathlib import Path

names = [
    'kwimage_ext.structs._boxes_backend.cython_boxes_legacy',
    'kwimage_ext.structs._mask_backend.cython_mask_legacy',
    'kwimage_ext.algo._nms_backend.cpu_nms_legacy',
    'kwimage_ext.algo._nms_backend.cpu_soft_nms_legacy',
]
payload = {}
for name in names:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    payload[name] = {
        'path': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
print(json.dumps(payload, sort_keys=True))
'''
        probe = _run([sys.executable, '-c', code], env=build_env, timeout=60)
        records.append({k: v for k, v in probe.items() if k not in {'stdout', 'stderr'}})
        if probe['returncode'] == 0:
            try:
                artifact_probe = json.loads((probe['stdout'] or '').strip())
            except json.JSONDecodeError:
                artifact_probe = {'parse_error': probe.get('stdout')}
        else:
            artifact_probe = {'probe_error': probe.get('stderr')}

    payload = {
        'ok': result['returncode'] == 0 and bool(artifact_probe)
              and 'parse_error' not in artifact_probe
              and 'probe_error' not in artifact_probe,
        'build_returncode': result['returncode'],
        'repo_root': str(REPO_ROOT),
        'python_executable': sys.executable,
        'source_hashes': _legacy_source_hashes(),
        'artifacts': artifact_probe,
        'ignored_ambient_build_flags': ignored_build_flags,
        'policy': 'same-checkout-cpu-only-legacy-reference',
    }
    (build_dir / 'legacy-reference.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload


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
    # Keep the evidence gate in sync as new Rust regression tests are added.
    # The earlier hard-coded list accidentally omitted the fast-path parity
    # tests added by the first optimization round.
    tests = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / 'tests').glob('test_rust_*.py')
    )
    backend_parity = REPO_ROOT / 'tests' / 'test_backend_parity.py'
    if not quick and backend_parity.exists():
        tests.append(str(backend_parity.relative_to(REPO_ROOT)))
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


def _benchmark_provenance(
        bundle, benchmark_json, required_pycocotools_version=None):
    # Require benchmark source and compiled extension versions to agree.
    build_policy_path = bundle / 'build' / 'profile-build-env.json'
    build_policy = (
        json.loads(build_policy_path.read_text())
        if build_policy_path.exists() else None
    )
    status = {
        'ok': False,
        'source_version': _source_version(),
        'extension_version': None,
        'extension_path': None,
        'extension_sha256': None,
        'portable_build_verified': (
            build_policy is not None and
            build_policy.get('policy') == 'portable-published-wheel-isa' and
            build_policy.get('target_cpu') == 'generic' and
            build_policy.get('RUSTFLAGS') == PORTABLE_PROFILE_RUSTFLAGS
        ),
        'build_policy': build_policy,
        'legacy_reference_rebuilt': False,
        'legacy_claim_verified': False,
        'legacy_artifact_matches': {},
        'pycocotools_claim_verified': False,
        'pycocotools_version': None,
        'pycocotools_extension_sha256': None,
        'required_pycocotools_version': required_pycocotools_version,
        'pycocotools_version_requirement_met': None,
    }
    if benchmark_json.exists():
        payload = json.loads(benchmark_json.read_text())
        benchmark_env = payload.get('environment', {})
        ext = benchmark_env.get('rust_extension', {})
        status['extension_version'] = ext.get('version')
        status['extension_path'] = ext.get('path')
        status['extension_sha256'] = ext.get('sha256')
        status['ok'] = (
            status['source_version'] is not None and
            status['extension_version'] == status['source_version']
        )

        legacy_build_path = bundle / 'build' / 'legacy-reference.json'
        legacy_build = (
            json.loads(legacy_build_path.read_text())
            if legacy_build_path.exists() else None
        )
        status['legacy_reference_rebuilt'] = bool(
            legacy_build and legacy_build.get('ok'))
        benchmark_legacy = (
            benchmark_env.get('backend_artifacts', {}).get('legacy', {}))
        built_legacy = (legacy_build or {}).get('artifacts', {})
        for name, built in built_legacy.items():
            measured = benchmark_legacy.get(name) or {}
            status['legacy_artifact_matches'][name] = bool(
                built.get('sha256') and
                built.get('sha256') == measured.get('sha256')
            )
        has_legacy_rows = any(
            row.get('backend') == 'legacy' for row in payload.get('results', []))
        status['legacy_claim_verified'] = bool(
            has_legacy_rows and
            status['legacy_reference_rebuilt'] and
            status['legacy_artifact_matches'] and
            all(status['legacy_artifact_matches'].values())
        )

        pycoco = benchmark_env.get('backend_artifacts', {}).get('pycocotools', {})
        pycoco_ext = pycoco.get('_mask') or {}
        status['pycocotools_version'] = (
            benchmark_env.get('optional_backend_versions', {}).get('pycocotools'))
        status['pycocotools_extension_sha256'] = pycoco_ext.get('sha256')
        has_pycoco_rows = any(
            row.get('backend') == 'pycocotools'
            for row in payload.get('results', []))
        status['pycocotools_claim_verified'] = bool(
            has_pycoco_rows and
            status['pycocotools_version'] and
            status['pycocotools_extension_sha256']
        )
        if required_pycocotools_version is not None:
            status['pycocotools_version_requirement_met'] = bool(
                status['pycocotools_claim_verified'] and
                status['pycocotools_version'] == required_pycocotools_version
            )
            status['ok'] = bool(
                status['ok'] and
                status['pycocotools_version_requirement_met']
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
        '--seeds', ','.join(map(str, CANONICAL_BENCHMARK_SEEDS[:1] if quick else CANONICAL_BENCHMARK_SEEDS)),
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
        'seeds_json', 'loops_per_seed_json',
        'median_ns', 'mean_ns', 'min_ns', 'p05_ns', 'p95_ns', 'max_ns',
        'stdev_ns', 'work_items', 'work_unit', 'median_ns_per_work_item',
        'result_digest', 'verification_comparator',
        'verification_comparators', 'verification_digest', 'traits_json',
    ]
    out = bundle / 'benchmarks' / 'results.csv'
    with out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            verification = row.get('verification', {})
            flat = {key: row.get(key) for key in columns}
            flat['seeds_json'] = json.dumps(row.get('seeds', []))
            flat['loops_per_seed_json'] = json.dumps(
                row.get('loops_per_seed', {}), sort_keys=True)
            flat['verification_comparator'] = verification.get('comparator')
            flat['verification_comparators'] = json.dumps(
                verification.get('comparators', []), sort_keys=True)
            flat['verification_digest'] = verification.get('digest')
            flat['traits_json'] = json.dumps(row.get('traits', {}), sort_keys=True)
            writer.writerow(flat)



def _comparison_band(ratio):
    """Describe an observed median ratio without claiming significance."""
    if ratio <= 0.95:
        return 'faster_by_at_least_5pct'
    if ratio >= 1.05:
        return 'slower_by_at_least_5pct'
    return 'within_5pct'


def _comparison_rows(payload):
    by_case = {}
    for row in payload.get('results', []):
        by_case.setdefault(row['case'], {})[row['backend']] = row

    rows = []
    paired_by_key = {
        (item['case'], item['comparator']): item
        for item in payload.get('paired_comparisons', [])
    }
    comparator_order = ('legacy', 'pycocotools', 'python')
    for case in [row['case'] for row in payload.get('results', []) if row['backend'] == 'rust']:
        backend_rows = by_case[case]
        rust = backend_rows.get('rust')
        if rust is None:
            continue
        for comparator_name in comparator_order:
            comparator = backend_rows.get(comparator_name)
            if comparator is None:
                continue
            paired = paired_by_key.get((case, comparator_name))
            if paired is None:
                ratio = rust['median_ns'] / comparator['median_ns']
                ratio_method = 'ratio-of-independent-medians'
                ratio_p05 = None
                ratio_p95 = None
                ratio_samples = None
                ratio_seeds = None
            else:
                ratio = paired['rust_over_comparator_median']
                ratio_method = paired.get('ratio_method')
                ratio_p05 = paired.get('rust_over_comparator_p05')
                ratio_p95 = paired.get('rust_over_comparator_p95')
                ratio_samples = paired.get('samples')
                ratio_seeds = paired.get('seeds')
            rows.append({
                'case': case,
                'family': rust.get('family'),
                'comparator': comparator_name,
                'rust_median_ns': rust['median_ns'],
                'comparator_median_ns': comparator['median_ns'],
                'rust_over_comparator': ratio,
                'ratio_method': ratio_method,
                'ratio_p05': ratio_p05,
                'ratio_p95': ratio_p95,
                'ratio_samples': ratio_samples,
                'ratio_seeds': ratio_seeds,
                'observed_band': _comparison_band(ratio),
                'rust_p05_ns': rust.get('p05_ns'),
                'rust_p95_ns': rust.get('p95_ns'),
                'comparator_p05_ns': comparator.get('p05_ns'),
                'comparator_p95_ns': comparator.get('p95_ns'),
                'traits': rust.get('traits', {}),
            })
    return rows


def _comparator_claim_provenance_verified(provenance, comparator):
    """Whether the native artifacts have enough provenance for a claim."""
    rust_verified = bool(
        provenance.get('ok') and provenance.get('portable_build_verified'))
    if comparator == 'legacy':
        comparator_verified = provenance.get('legacy_claim_verified') is True
    elif comparator == 'pycocotools':
        comparator_verified = provenance.get('pycocotools_claim_verified') is True
    elif comparator == 'python':
        comparator_verified = True
    else:
        comparator_verified = False
    return rust_verified and comparator_verified


def _comparison_claim_verified(provenance, row):
    """Whether one timing row is suitable for the canonical release claim."""
    seeds = row.get('ratio_seeds') or []
    paired_multi_seed = bool(
        row.get('ratio_method') == 'paired-rotating-order-samples-multi-seed'
        and len(set(seeds)) >= 2
        and row.get('ratio_samples')
        and row.get('ratio_p05') is not None
        and row.get('ratio_p95') is not None
    )
    return bool(
        paired_multi_seed and
        _comparator_claim_provenance_verified(
            provenance, row.get('comparator'))
    )


def _write_comparison_outputs(bundle, benchmark_json):
    if not benchmark_json.exists():
        return []
    payload = json.loads(benchmark_json.read_text())
    rows = _comparison_rows(payload)
    provenance_path = bundle / 'benchmarks' / 'provenance.json'
    provenance = (
        json.loads(provenance_path.read_text())
        if provenance_path.exists() else {}
    )
    for row in rows:
        row['claim_verified'] = _comparison_claim_verified(provenance, row)
    out_json = bundle / 'benchmarks' / 'comparisons.json'
    out_json.write_text(json.dumps({
        'schema_version': 2,
        'interpretation': {
            'faster_by_at_least_5pct': 'rust/comparator <= 0.95',
            'within_5pct': '0.95 < rust/comparator < 1.05',
            'slower_by_at_least_5pct': 'rust/comparator >= 1.05',
            'note': (
                'Canonical schema-v3 benchmark rows use the median of paired '
                'Rust/comparator sample ratios collected in rotating backend '
                'order across multiple fixture seeds. Bands remain descriptive '
                'and are not statistical significance tests. claim_verified '
                'also requires a portable same-source Rust build and verified '
                'comparator provenance.'
            ),
        },
        'rows': rows,
    }, indent=2, sort_keys=True) + '\n')

    out_csv = bundle / 'benchmarks' / 'comparisons.csv'
    columns = [
        'case', 'family', 'comparator', 'rust_median_ns',
        'comparator_median_ns', 'rust_over_comparator', 'observed_band',
        'ratio_method', 'ratio_p05', 'ratio_p95', 'ratio_samples',
        'ratio_seeds_json', 'claim_verified',
        'rust_p05_ns', 'rust_p95_ns', 'comparator_p05_ns',
        'comparator_p95_ns', 'traits_json',
    ]
    with out_csv.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in columns}
            flat['ratio_seeds_json'] = json.dumps(row.get('ratio_seeds'))
            flat['traits_json'] = json.dumps(row.get('traits', {}), sort_keys=True)
            writer.writerow(flat)
    return rows


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


def _capture_comparator_artifacts(bundle, benchmark_json):
    if not benchmark_json.exists():
        return
    payload = json.loads(benchmark_json.read_text())
    backend_artifacts = payload.get('environment', {}).get('backend_artifacts', {})
    out_root = bundle / 'comparators'
    metadata = {}
    for group_name, group in backend_artifacts.items():
        group_meta = {}
        for name, item in (group or {}).items():
            if not item:
                group_meta[name] = None
                continue
            path = Path(item.get('path') or '')
            copied_rel = None
            copied_sha256 = None
            if path.is_file():
                safe_name = name.replace('.', '_') + path.suffix
                copied = out_root / group_name / safe_name
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copied)
                copied_rel = str(copied.relative_to(bundle))
                copied_sha256 = _sha256(copied)
            group_meta[name] = {
                **item,
                'copied_path': copied_rel,
                'copied_sha256': copied_sha256,
            }
        metadata[group_name] = group_meta
    if metadata:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / 'metadata.json').write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + '\n')


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
        provenance_path = bundle / 'benchmarks' / 'provenance.json'
        provenance = (
            json.loads(provenance_path.read_text())
            if provenance_path.exists() else {}
        )
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
        comparison_rows = _comparison_rows(payload)
        pycoco_version = (
            payload.get('environment', {})
            .get('optional_backend_versions', {})
            .get('pycocotools')
        )
        lines.extend([
            '## Comparator availability',
            '',
            '- legacy Cython/C: same-checkout rebuild verified'
            if provenance.get('legacy_claim_verified')
            else '- legacy Cython/C: not source-current verified; do not use legacy rows for a same-checkout performance claim',
            f'- pycocotools: `{pycoco_version}`' if pycoco_version else '- pycocotools: unavailable; no direct pycocotools performance claim can be made from this bundle.',
            '- pycocotools binary fingerprint captured'
            if provenance.get('pycocotools_claim_verified')
            else '- pycocotools binary fingerprint unavailable',
            (
                f'- required pycocotools version: '
                f'`{provenance.get("required_pycocotools_version")}`; '
                f'matched: `{provenance.get("pycocotools_version_requirement_met")}`'
            )
            if provenance.get('required_pycocotools_version') is not None
            else '- required pycocotools version: not specified',
            f'- fixture seeds: `{payload.get("seeds")}`',
            f'- comparison timing: `{payload.get("timing_method", "independent backend blocks")}`',
            '',
            '## Microbenchmarks',
            '',
            '| case | rust median us | comparator | comparator median us | rust/comparator | observed band |',
            '| --- | ---: | --- | ---: | ---: | --- |',
        ])
        for row in comparison_rows:
            claim_status = (
                'yes' if _comparison_claim_verified(provenance, row) else 'no'
            )
            lines.append(
                f'| `{row["case"]}` | {_format_us(row["rust_median_ns"])} | '
                f'{row["comparator"]} | {_format_us(row["comparator_median_ns"])} | '
                f'{row["rust_over_comparator"]:.3f}x | '
                f'{row["observed_band"]} (claim-ready: {claim_status}) |')
        if not comparison_rows:
            lines.append('| _no comparator rows available_ | | | | | |')
        lines.extend([
            '',
            'Lower `rust/comparator` is faster.',
            'For schema-v3 evidence, `rust/comparator` is the median of paired '
            'sample ratios gathered in rotating backend order across the recorded seeds.',
            'The ±5% band is a conservative descriptive tolerance around the observed ratio, not a significance test.',
            '',
            '## Comparator coverage',
            '',
        ])
        for comparator in ('legacy', 'pycocotools', 'python'):
            rows = [row for row in comparison_rows if row['comparator'] == comparator]
            if not rows:
                continue
            counts = {
                band: sum(row['observed_band'] == band for row in rows)
                for band in (
                    'faster_by_at_least_5pct',
                    'within_5pct',
                    'slower_by_at_least_5pct',
                )
            }
            worst = max(rows, key=lambda row: row['rust_over_comparator'])
            claim_ready = bool(rows) and all(
                _comparison_claim_verified(provenance, row) for row in rows)
            lines.append(
                f'- `{comparator}` ({"claim-ready" if claim_ready else "provenance-unverified"}): '
                f'{len(rows)} compared cases; '
                f'{counts["faster_by_at_least_5pct"]} >=5% faster, '
                f'{counts["within_5pct"]} within 5%, '
                f'{counts["slower_by_at_least_5pct"]} >=5% slower. '
                f'Highest observed rust/comparator ratio: '
                f'`{worst["rust_over_comparator"]:.3f}x` on `{worst["case"]}`.')
        lines.extend([
            '',
            'Any performance statement should be scoped to this recorded benchmark matrix and machine/toolchain.',
            '',
        ])
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
        '- `benchmarks/comparisons.{json,csv}` records every Rust/comparator pair and the paired-ratio method used for claims.',
        '- `extension/` contains that compiled extension and ELF metadata when available.',
        '- `comparators/` contains copied legacy/pycocotools comparator artifacts and SHA-256 fingerprints when available.',
        '- `build/legacy-reference.json` records the same-checkout legacy rebuild and source hashes.',
        '- `source/` snapshots the Rust kernels and benchmark/profiling front doors.',
        '- `git/` records HEAD, dirty status, and the working-tree diff.',
        '- Raw perf counter output and sampled reports are under `perf/<case>/`.',
        '- Profile rebuilds use `-C target-cpu=generic`; ambient Rust/Cargo ISA flags are ignored.',
        '- The timed profiled binary does not force frame pointers; perf uses DWARF call graphs.',
        '- No runtime SIMD/multiversion path is implied by this profiling policy.',
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
                        help='Run maturin develop --release with portable target-cpu=generic and debug info first.')
    parser.add_argument(
        '--skip-legacy-rebuild', action='store_true',
        help='With --rebuild-profiled, do not rebuild the same-checkout legacy Cython/C reference. '
             'Legacy benchmark rows will then be marked unverified for source-current claims.',
    )
    parser.add_argument(
        '--require-pycocotools-version',
        help='Fail the evidence run unless this exact pycocotools version is '
             'installed, fingerprinted, and directly benchmarked.',
    )
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
    env.update(BENCHMARK_THREAD_ENV)

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
            # Build the Cython reference first because its helper may install
            # build requirements (including NumPy). Rust is then compiled
            # against the final environment that will actually be benchmarked.
            if not args.skip_legacy_rebuild:
                legacy_rebuild = _rebuild_legacy_reference(bundle, commands, env)
                if legacy_rebuild['ok'] is not True:
                    hard_failure = True
            if not hard_failure:
                rebuild = _rebuild_profiled(bundle, commands, env)
                if rebuild['returncode'] != 0:
                    hard_failure = True
            if not hard_failure:
                _capture_text(
                    bundle,
                    'environment/pip-freeze-post-build.txt',
                    [sys.executable, '-m', 'pip', 'freeze'],
                    commands,
                    env=env,
                )

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
            provenance = _benchmark_provenance(
                bundle,
                benchmark_json,
                required_pycocotools_version=args.require_pycocotools_version,
            )
            if provenance['ok'] is not True:
                hard_failure = True

        _write_benchmark_csv(bundle, benchmark_json)
        _write_comparison_outputs(bundle, benchmark_json)
        _capture_extension_artifact(bundle, benchmark_json, commands, env)
        _capture_comparator_artifacts(bundle, benchmark_json)

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
