"""Utilities for choosing Rust or legacy extension backends.

Backend selection is capability-aware.  A partially implemented Rust module is
never allowed to shadow a working legacy extension for an API it does not yet
provide.

Environment forcing is read dynamically instead of once at module import time.
This is useful both for tests and for diagnostic processes that deliberately
exercise Rust and legacy backends in the same interpreter.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
from types import ModuleType


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _force_legacy() -> bool:
    return _env_truthy('KWIMAGE_EXT_FORCE_LEGACY')


def _force_rust() -> bool:
    return _env_truthy('KWIMAGE_EXT_FORCE_RUST')


def _missing_symbols(module: ModuleType, required_symbols):
    return tuple(name for name in required_symbols if not hasattr(module, name))


def _load_extension_from_path(module_name: str, ext_stem: str, search_dir: str) -> ModuleType | None:
    """Load a compiled extension from a specific package path, if it exists."""
    base = Path(search_dir)
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for candidate in base.glob(f'{ext_stem}*{suffix}'):
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


def import_backend(
    rust_module: str,
    legacy_module: str,
    *,
    required_symbols=(),
    legacy_extension_stem: str | None = None,
    legacy_search_dir: str | None = None,
) -> ModuleType:
    """Import the first backend that implements all requested capabilities.

    ``KWIMAGE_EXT_FORCE_LEGACY=1`` skips Rust.  ``KWIMAGE_EXT_FORCE_RUST=1``
    refuses fallback and is intended for CI parity tests of Rust-only wheels.
    Setting both variables is an error.
    """
    required_symbols = tuple(required_symbols)
    errors = []
    force_legacy = _force_legacy()
    force_rust = _force_rust()
    if force_legacy and force_rust:
        raise RuntimeError(
            'KWIMAGE_EXT_FORCE_LEGACY and KWIMAGE_EXT_FORCE_RUST cannot both be enabled')

    if not force_legacy:
        try:
            module = importlib.import_module(rust_module)
        except Exception as ex:  # pragma: no cover - diagnostic fallback
            errors.append((rust_module, ex))
        else:
            missing = _missing_symbols(module, required_symbols)
            if not missing:
                return module
            errors.append((rust_module, RuntimeError(
                f'backend is missing required symbols: {missing!r}')))
            if force_rust:
                attempted = ', '.join(f'{name}: {err}' for name, err in errors)
                raise ImportError(
                    f'Rust backend lacks required capability. Attempted: {attempted}')

    if force_rust:
        attempted = ', '.join(f'{name}: {err}' for name, err in errors)
        raise ImportError(
            f'Rust backend is required but unavailable. Attempted: {attempted}')

    if legacy_extension_stem and legacy_search_dir:
        try:
            module = _load_extension_from_path(
                legacy_module, legacy_extension_stem, legacy_search_dir)
            if module is not None:
                missing = _missing_symbols(module, required_symbols)
                if not missing:
                    return module
                errors.append((legacy_module, RuntimeError(
                    f'legacy extension is missing required symbols: {missing!r}')))
        except Exception as ex:  # pragma: no cover
            errors.append((f'{legacy_search_dir}/{legacy_extension_stem}*', ex))

    try:
        module = importlib.import_module(legacy_module)
    except Exception as ex:
        errors.append((legacy_module, ex))
    else:
        missing = _missing_symbols(module, required_symbols)
        if not missing:
            return module
        errors.append((legacy_module, RuntimeError(
            f'legacy module is missing required symbols: {missing!r}')))

    attempted = ', '.join(f'{name}: {err}' for name, err in errors)
    raise ImportError(f'Unable to import a capable backend. Attempted: {attempted}')


def backend_info(module: ModuleType) -> dict:
    """Return stable diagnostic metadata for benchmark provenance."""
    name = getattr(module, '__name__', type(module).__name__)
    kind = 'rust' if name == 'kwimage_ext._rust' else 'legacy'
    return {
        'module': name,
        'kind': kind,
        'file': getattr(module, '__file__', None),
    }
