"""Unit tests for portable Rust profiling-build policy."""
import importlib.util
from pathlib import Path


def _load_profile_module():
    root = Path(__file__).resolve().parents[1]
    path = root / 'dev' / 'profiling' / 'profile_rust_kernels.py'
    spec = importlib.util.spec_from_file_location('profile_rust_kernels_portability', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profile_build_uses_explicit_generic_cpu_and_ignores_native_flags():
    mod = _load_profile_module()
    env = {
        'PATH': '/usr/bin',
        'RUSTFLAGS': '-C target-cpu=native -C target-feature=+avx2',
        'CARGO_ENCODED_RUSTFLAGS': '-C\x1ftarget-feature=+avx512f',
        'CARGO_BUILD_RUSTFLAGS': '-C target-cpu=haswell',
        'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS': '-C target-feature=+fma',
        'CFLAGS': '-O3',
    }
    got, ignored = mod._portable_profile_build_env(env)

    assert got['RUSTFLAGS'] == '-C target-cpu=generic'
    assert got['CARGO_PROFILE_RELEASE_DEBUG'] == '1'
    assert got['CFLAGS'] == '-O3'
    assert 'CARGO_ENCODED_RUSTFLAGS' not in got
    assert 'CARGO_BUILD_RUSTFLAGS' not in got
    assert 'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS' not in got
    assert ignored == {
        'RUSTFLAGS': '-C target-cpu=native -C target-feature=+avx2',
        'CARGO_ENCODED_RUSTFLAGS': '-C\x1ftarget-feature=+avx512f',
        'CARGO_BUILD_RUSTFLAGS': '-C target-cpu=haswell',
        'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS': '-C target-feature=+fma',
    }


def test_profile_build_does_not_force_frame_pointers():
    mod = _load_profile_module()
    got, _ = mod._portable_profile_build_env({})
    assert 'force-frame-pointers' not in got['RUSTFLAGS']
    assert 'native' not in got['RUSTFLAGS']
    assert 'target-feature' not in got['RUSTFLAGS']
