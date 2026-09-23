from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = REPO_ROOT / 'dev' / 'benchmarks' / 'rust_kernel_benchmarks.py'
PROFILE_PATH = REPO_ROOT / 'dev' / 'profiling' / 'profile_rust_kernels.py'


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_case_registry_is_stable_and_unique():
    bench = _load_module(BENCH_PATH, '_kwimage_ext_bench_test_registry')
    names = [spec.name for spec in bench.CASE_SPECS]
    assert len(names) == len(set(names))
    assert {
        'boxes_iou_large',
        'cpu_nms_sparse_1000',
        'mask_iou_fragmented_48',
        'assignment_sparse',
    } <= set(names)
    assert all(bench.CASE_BY_NAME[name].name == name for name in names)


def test_digest_is_content_sensitive_and_layout_independent():
    bench = _load_module(BENCH_PATH, '_kwimage_ext_bench_test_digest')
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.asfortranarray(a)
    c = a.copy()
    c[1, 2] += 1
    assert bench._digest_value(a) == bench._digest_value(b)
    assert bench._digest_value(a) != bench._digest_value(c)


def test_percentile_interpolates_deterministically():
    bench = _load_module(BENCH_PATH, '_kwimage_ext_bench_test_percentile')
    values = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert bench._percentile(values, 0.0) == 0.0
    assert bench._percentile(values, 0.5) == 20.0
    assert bench._percentile(values, 1.0) == 40.0
    assert bench._percentile([10.0, 20.0], 0.25) == 12.5


def test_group_benchmark_uses_paired_sample_ratios(monkeypatch):
    bench = _load_module(BENCH_PATH, '_kwimage_ext_bench_test_paired_group')
    spec = bench.CaseSpec('fake_case', 'test', 'paired timing test')
    cases = [
        bench.PreparedCase(spec, 'rust', lambda: np.array([1])),
        bench.PreparedCase(spec, 'legacy', lambda: np.array([1])),
    ]
    elapsed = {
        'rust': iter([10, 20, 30, 40]),
        'legacy': iter([20, 40, 60, 80]),
    }

    monkeypatch.setattr(
        bench, '_calibrate_case',
        lambda case, **kwargs: ({'comparator': 'test'}, 1))

    def fake_time(case, loops):
        assert loops == 1
        return np.array([1]), next(elapsed[case.backend])

    monkeypatch.setattr(bench, '_time_case_sample', fake_time)
    rows, pairs = bench.benchmark_case_group(
        cases, samples=4, target_sample_seconds=0.001, warmup=0)
    assert [row['backend'] for row in rows] == ['rust', 'legacy']
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair['rust_over_comparator_sample_ratios'] == [0.5] * 4
    assert pair['rust_over_comparator_median'] == 0.5


def test_run_suite_aggregates_paired_ratios_across_seeds(monkeypatch):
    bench = _load_module(BENCH_PATH, '_kwimage_ext_bench_test_multiseed_suite')
    spec = bench.CaseSpec('fake_case', 'test', 'multi-seed timing test')
    monkeypatch.setitem(bench.CASE_BY_NAME, 'fake_case', spec)
    monkeypatch.setattr(
        bench, 'available_backends',
        lambda name, seed=0: ['rust', 'legacy'])
    monkeypatch.setattr(
        bench, 'prepare_case',
        lambda name, backend='rust', seed=0: bench.PreparedCase(
            spec, backend, lambda: np.array([seed], dtype=np.int64)))
    monkeypatch.setattr(bench, 'environment_metadata', lambda: {'test': True})

    def fake_group(cases, **kwargs):
        seed = int(cases[0].call()[0])
        rows = []
        medians = {'rust': 10.0 + seed, 'legacy': 20.0 + seed}
        for case in cases:
            median = medians[case.backend]
            rows.append({
                'case': 'fake_case',
                'family': 'test',
                'description': 'multi-seed timing test',
                'backend': case.backend,
                'samples': 2,
                'loops_per_sample': 3,
                'median_ns': median,
                'mean_ns': median,
                'min_ns': median,
                'p05_ns': median,
                'p95_ns': median,
                'max_ns': median,
                'stdev_ns': 0.0,
                'sample_ns': [median, median],
                'work_items': 1,
                'work_unit': 'item',
                'median_ns_per_work_item': median,
                'result_digest': f'{case.backend}-{seed}',
                'verification': {'comparator': 'test'},
                'traits': {'seed': seed},
            })
        ratio = medians['rust'] / medians['legacy']
        pairs = [{
            'case': 'fake_case',
            'family': 'test',
            'comparator': 'legacy',
            'ratio_method': 'paired-rotating-order-samples',
            'samples': 2,
            'rust_over_comparator_sample_ratios': [ratio, ratio],
            'rust_over_comparator_median': ratio,
            'rust_over_comparator_p05': ratio,
            'rust_over_comparator_p95': ratio,
        }]
        return rows, pairs

    monkeypatch.setattr(bench, 'benchmark_case_group', fake_group)
    payload = bench.run_suite(
        ['fake_case'], seeds=[0, 1], samples=2,
        target_sample_seconds=0.001, warmup=0)
    assert payload['schema_version'] == 3
    assert payload['seeds'] == [0, 1]
    assert len(payload['results']) == 2
    rust = next(row for row in payload['results'] if row['backend'] == 'rust')
    assert rust['samples'] == 4
    assert rust['loops_per_seed'] == {'0': 3, '1': 3}
    pair = payload['paired_comparisons'][0]
    assert pair['samples'] == 4
    assert pair['seeds'] == [0, 1]


def test_list_mode_does_not_require_importable_extension():
    proc = subprocess.run(
        [sys.executable, str(BENCH_PATH), '--list'],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'boxes_iou_large' in proc.stdout
    assert 'assignment_sparse' in proc.stdout


def test_profile_summary_renders_comparator_ratio(tmp_path):
    profile = _load_module(PROFILE_PATH, '_kwimage_ext_profile_test_summary')
    bench_json = tmp_path / 'benchmarks.json'
    bench_json.write_text(json.dumps({
        'environment': {'rust_extension': {'version': '9.9.8'}},
        'results': [
            {
                'case': 'boxes_iou_small',
                'backend': 'rust',
                'median_ns': 1000.0,
                'median_ns_per_work_item': 1.0,
            },
            {
                'case': 'boxes_iou_small',
                'backend': 'legacy',
                'median_ns': 2000.0,
            },
        ]
    }))
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'metadata.json').write_text(json.dumps({'source_version': '9.9.9'}))
    profile._write_summary(
        bundle,
        bench_json,
        {
            'full': {'returncode': 0},
            'kernel_without_distribution_metadata_test': None,
            'metadata_only_failure': False,
            'required_passed': True,
            'version_state': {},
        },
        {'boxes_iou_large': {'stat': 0, 'record': 0}},
    )
    text = (bundle / 'summary.md').read_text()
    assert '0.500x' in text
    assert 'boxes_iou_small' in text
    assert 'stat=0, record=0' in text
    assert 'Source version `9.9.9` does not match' in text


def test_benchmark_csv_flattens_verification(tmp_path):
    profile = _load_module(PROFILE_PATH, '_kwimage_ext_profile_test_csv')
    benchmark_json = tmp_path / 'results.json'
    benchmark_json.write_text(json.dumps({
        'results': [{
            'case': 'boxes_iou_small',
            'family': 'boxes',
            'backend': 'rust',
            'samples': 3,
            'loops_per_sample': 10,
            'median_ns': 123.0,
            'mean_ns': 124.0,
            'min_ns': 120.0,
            'p05_ns': 120.5,
            'p95_ns': 130.0,
            'max_ns': 131.0,
            'stdev_ns': 3.0,
            'work_items': 64,
            'work_unit': 'box-pairs',
            'median_ns_per_work_item': 1.921875,
            'result_digest': 'abc',
            'verification': {'comparator': 'legacy-cython', 'digest': 'def'},
        }]
    }))
    bundle = tmp_path / 'bundle'
    (bundle / 'benchmarks').mkdir(parents=True)
    profile._write_benchmark_csv(bundle, benchmark_json)
    text = (bundle / 'benchmarks' / 'results.csv').read_text()
    assert 'verification_comparator' in text
    assert 'legacy-cython' in text
    assert ',def' in text


def test_perf_default_cases_are_valid_benchmark_names():
    bench = _load_module(BENCH_PATH, '_kwimage_ext_bench_test_perf_names')
    profile = _load_module(PROFILE_PATH, '_kwimage_ext_profile_test_perf_names')
    assert set(profile.DEFAULT_PERF_CASES) <= set(bench.CASE_BY_NAME)
    assert all(bench.CASE_BY_NAME[name].perf_default for name in profile.DEFAULT_PERF_CASES)


def test_quick_rust_microbenchmark_if_extension_is_available():
    pytest.importorskip('kwimage_ext._rust')
    bench = _load_module(BENCH_PATH, '_kwimage_ext_bench_test_integration')
    payload = bench.run_suite(
        ['boxes_iou_small'],
        backends=['rust'],
        seed=0,
        samples=2,
        target_sample_seconds=0.001,
        warmup=1,
    )
    assert not payload['skipped']
    assert len(payload['results']) == 1
    row = payload['results'][0]
    assert row['backend'] == 'rust'
    assert row['median_ns'] > 0
    assert row['verification']['digest']



def test_correctness_gate_allows_only_stale_distribution_metadata(tmp_path, monkeypatch):
    profile = _load_module(PROFILE_PATH, '_kwimage_ext_profile_test_correctness_classify')
    results = iter([
        {
            'command': ['pytest'],
            'returncode': 1,
            'duration_seconds': 1.0,
            'stdout': None,
            'stderr': None,
        },
        {
            'command': ['pytest'],
            'returncode': 0,
            'duration_seconds': 1.0,
            'stdout': None,
            'stderr': None,
        },
    ])
    monkeypatch.setattr(profile, '_run', lambda *args, **kwargs: next(results))
    version_state = {
        'source_version': '0.4.1',
        'package_version': '0.4.1',
        'rust_extension_version': '0.4.1',
        'distribution_version': '0.3.2',
        'source_matches_package': True,
        'source_matches_extension': True,
        'distribution_matches_source': False,
    }
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    status = profile._run_correctness_gate(
        bundle, [], {}, quick=True, version_state=version_state)
    assert status['metadata_only_failure'] is True
    assert status['required_passed'] is True
    saved = json.loads((bundle / 'correctness' / 'status.json').read_text())
    assert saved['required_passed'] is True


def test_correctness_gate_keeps_kernel_failure_required(tmp_path, monkeypatch):
    profile = _load_module(PROFILE_PATH, '_kwimage_ext_profile_test_correctness_hard')
    results = iter([
        {
            'command': ['pytest'],
            'returncode': 1,
            'duration_seconds': 1.0,
            'stdout': None,
            'stderr': None,
        },
        {
            'command': ['pytest'],
            'returncode': 1,
            'duration_seconds': 1.0,
            'stdout': None,
            'stderr': None,
        },
    ])
    monkeypatch.setattr(profile, '_run', lambda *args, **kwargs: next(results))
    version_state = {
        'source_matches_package': True,
        'source_matches_extension': True,
        'distribution_matches_source': False,
    }
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    status = profile._run_correctness_gate(
        bundle, [], {}, quick=True, version_state=version_state)
    assert status['metadata_only_failure'] is False
    assert status['required_passed'] is False


def test_benchmark_provenance_requires_current_extension(tmp_path, monkeypatch):
    profile = _load_module(PROFILE_PATH, '_kwimage_ext_profile_test_provenance')
    monkeypatch.setattr(profile, '_source_version', lambda: '0.4.1')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    benchmark_json = tmp_path / 'results.json'
    benchmark_json.write_text(json.dumps({
        'environment': {
            'rust_extension': {
                'version': '0.4.0',
                'path': '/tmp/_rust.so',
                'sha256': 'abc',
            },
        },
        'results': [],
    }))
    status = profile._benchmark_provenance(bundle, benchmark_json)
    assert status['ok'] is False
    payload = json.loads(benchmark_json.read_text())
    payload['environment']['rust_extension']['version'] = '0.4.1'
    benchmark_json.write_text(json.dumps(payload))
    status = profile._benchmark_provenance(bundle, benchmark_json)
    assert status['ok'] is True


def test_benchmark_provenance_verifies_rebuilt_legacy_artifacts(tmp_path, monkeypatch):
    profile = _load_module(PROFILE_PATH, '_kwimage_ext_profile_test_legacy_provenance')
    monkeypatch.setattr(profile, '_source_version', lambda: '0.4.1')
    bundle = tmp_path / 'bundle'
    (bundle / 'build').mkdir(parents=True)
    benchmark_json = tmp_path / 'results.json'
    legacy_name = 'kwimage_ext.structs._mask_backend.cython_mask_legacy'
    (bundle / 'build' / 'legacy-reference.json').write_text(json.dumps({
        'ok': True,
        'artifacts': {
            legacy_name: {'path': '/tmp/legacy.so', 'sha256': 'legacy-hash'},
        },
    }))
    benchmark_json.write_text(json.dumps({
        'environment': {
            'rust_extension': {
                'version': '0.4.1',
                'path': '/tmp/_rust.so',
                'sha256': 'rust-hash',
            },
            'backend_artifacts': {
                'legacy': {
                    legacy_name: {
                        'path': '/tmp/legacy.so',
                        'sha256': 'legacy-hash',
                    },
                },
                'pycocotools': {},
            },
            'optional_backend_versions': {},
        },
        'results': [
            {'case': 'mask', 'backend': 'rust'},
            {'case': 'mask', 'backend': 'legacy'},
        ],
    }))
    status = profile._benchmark_provenance(bundle, benchmark_json)
    assert status['ok'] is True
    assert status['legacy_reference_rebuilt'] is True
    assert status['legacy_artifact_matches'][legacy_name] is True
    assert status['legacy_claim_verified'] is True


def test_benchmark_provenance_can_require_exact_pycocotools(tmp_path, monkeypatch):
    profile = _load_module(
        PROFILE_PATH, '_kwimage_ext_profile_test_pycoco_version_requirement')
    monkeypatch.setattr(profile, '_source_version', lambda: '0.4.1')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    benchmark_json = tmp_path / 'results.json'
    benchmark_json.write_text(json.dumps({
        'environment': {
            'rust_extension': {
                'version': '0.4.1',
                'path': '/tmp/_rust.so',
                'sha256': 'rust-hash',
            },
            'backend_artifacts': {
                'legacy': {},
                'pycocotools': {
                    '_mask': {
                        'path': '/tmp/_mask.so',
                        'sha256': 'pycoco-hash',
                    },
                },
            },
            'optional_backend_versions': {'pycocotools': '2.0.11'},
        },
        'results': [
            {'case': 'mask', 'backend': 'rust'},
            {'case': 'mask', 'backend': 'pycocotools'},
        ],
    }))
    matched = profile._benchmark_provenance(
        bundle, benchmark_json, required_pycocotools_version='2.0.11')
    assert matched['ok'] is True
    assert matched['pycocotools_version_requirement_met'] is True
    mismatched = profile._benchmark_provenance(
        bundle, benchmark_json, required_pycocotools_version='2.0.10')
    assert mismatched['ok'] is False
    assert mismatched['pycocotools_version_requirement_met'] is False


def test_comparison_claim_requires_portable_and_comparator_provenance():
    profile = _load_module(
        PROFILE_PATH, '_kwimage_ext_profile_test_claim_provenance')
    provenance = {
        'ok': True,
        'portable_build_verified': True,
        'legacy_claim_verified': True,
        'pycocotools_claim_verified': True,
    }
    base_row = {
        'ratio_method': 'paired-rotating-order-samples-multi-seed',
        'ratio_seeds': [0, 1, 2],
        'ratio_samples': 27,
        'ratio_p05': 0.8,
        'ratio_p95': 0.9,
    }
    for comparator in ['legacy', 'pycocotools', 'python']:
        row = {**base_row, 'comparator': comparator}
        assert profile._comparison_claim_verified(provenance, row) is True
    provenance['portable_build_verified'] = False
    row = {**base_row, 'comparator': 'legacy'}
    assert profile._comparison_claim_verified(provenance, row) is False
    provenance['portable_build_verified'] = True
    provenance['legacy_claim_verified'] = False
    assert profile._comparison_claim_verified(provenance, row) is False
    provenance['legacy_claim_verified'] = True
    one_seed = {**row, 'ratio_seeds': [0]}
    assert profile._comparison_claim_verified(provenance, one_seed) is False


def test_legacy_rebuild_pins_benchmark_python(tmp_path, monkeypatch):
    profile = _load_module(
        PROFILE_PATH, '_kwimage_ext_profile_test_legacy_python')
    seen = []
    artifact_names = [
        'kwimage_ext.structs._boxes_backend.cython_boxes_legacy',
        'kwimage_ext.structs._mask_backend.cython_mask_legacy',
        'kwimage_ext.algo._nms_backend.cpu_nms_legacy',
        'kwimage_ext.algo._nms_backend.cpu_soft_nms_legacy',
    ]

    def fake_run(command, **kwargs):
        seen.append((list(command), dict(kwargs.get('env') or {})))
        if command[0] == sys.executable:
            stdout = json.dumps({
                name: {'path': f'/tmp/{idx}.so', 'sha256': f'hash-{idx}'}
                for idx, name in enumerate(artifact_names)
            })
        else:
            stdout = ''
        return {
            'command': list(command),
            'returncode': 0,
            'duration_seconds': 0.0,
            'stdout': stdout,
            'stderr': '',
        }

    monkeypatch.setattr(profile, '_run', fake_run)
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    payload = profile._rebuild_legacy_reference(bundle, [], {'PATH': '/tmp'})
    assert payload['ok'] is True
    assert payload['python_executable'] == sys.executable
    assert seen[0][1]['KWIMAGE_EXT_LEGACY_PYTHON'] == sys.executable


def test_perf_workload_hashes_only_after_sampled_loop(monkeypatch):
    bench = _load_module(BENCH_PATH, '_kwimage_ext_bench_test_perf_digest_scope')
    spec = bench.CaseSpec('fake_case', 'test', 'test case')
    calls = []

    def call():
        calls.append(len(calls))
        return np.array([len(calls)], dtype=np.int64)

    case = bench.PreparedCase(spec, 'rust', call)
    monkeypatch.setattr(bench, 'prepare_case', lambda *args, **kwargs: case)
    clock = iter([0.0, 0.1, 0.2, 0.6])
    monkeypatch.setattr(bench.time, 'perf_counter', lambda: next(clock))
    digest_calls = []
    original_digest = bench._digest_value

    def counted_digest(value):
        digest_calls.append(value)
        return original_digest(value)

    monkeypatch.setattr(bench, '_digest_value', counted_digest)
    payload = bench.run_perf_workload(
        'fake_case', backend='rust', seconds=0.5, warmup=0)
    assert payload['calls'] == 2
    assert len(digest_calls) == 1


def test_profile_limits_unrelated_numeric_worker_pools():
    profile = _load_module(PROFILE_PATH, '_kwimage_ext_profile_test_thread_env')
    expected = {
        'OPENBLAS_NUM_THREADS',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'BLIS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
    }
    assert expected <= set(profile.BENCHMARK_THREAD_ENV)
    assert all(profile.BENCHMARK_THREAD_ENV[key] == '1' for key in expected)


def test_correctness_gate_discovers_fastpath_regressions(tmp_path, monkeypatch):
    profile = _load_module(PROFILE_PATH, '_kwimage_ext_profile_test_dynamic_tests')
    seen_commands = []

    def fake_run(command, **kwargs):
        seen_commands.append(list(command))
        return {
            'command': list(command),
            'returncode': 0,
            'duration_seconds': 0.0,
            'stdout': None,
            'stderr': None,
        }

    monkeypatch.setattr(profile, '_run', fake_run)
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    status = profile._run_correctness_gate(
        bundle, [], {}, quick=False, version_state={})
    assert status['required_passed'] is True
    command = seen_commands[0]
    fastpath_test = str(
        (REPO_ROOT / 'tests' / 'test_rust_performance_fastpaths.py')
        .relative_to(REPO_ROOT)
    )
    assert fastpath_test in command
    backend_parity = REPO_ROOT / 'tests' / 'test_backend_parity.py'
    if backend_parity.exists():
        assert str(backend_parity.relative_to(REPO_ROOT)) in command
