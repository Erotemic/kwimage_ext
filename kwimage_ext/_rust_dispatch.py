"""Utilities for choosing Rust or legacy extension backends."""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
from types import ModuleType

_FORCE_LEGACY = os.environ.get("KWIMAGE_EXT_FORCE_LEGACY", "").strip().lower() in {
    "1", "true", "yes", "on"
}


def _load_extension_from_path(module_name: str, ext_stem: str, search_dir: str) -> ModuleType | None:
    """Load a compiled extension from a specific package path, if it exists."""
    base = Path(search_dir)
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for candidate in base.glob(f"{ext_stem}*{suffix}"):
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
    legacy_extension_stem: str | None = None,
    legacy_search_dir: str | None = None,
) -> ModuleType:
    """Import Rust backend unless legacy mode is requested."""
    errors = []
    if not _FORCE_LEGACY:
        try:
            return importlib.import_module(rust_module)
        except Exception as ex:  # pragma: no cover - diagnostic fallback
            errors.append((rust_module, ex))

    if legacy_extension_stem and legacy_search_dir:
        try:
            module = _load_extension_from_path(legacy_module, legacy_extension_stem, legacy_search_dir)
            if module is not None:
                return module
        except Exception as ex:  # pragma: no cover
            errors.append((f"{legacy_search_dir}/{legacy_extension_stem}*", ex))

    try:
        return importlib.import_module(legacy_module)
    except Exception as ex:
        errors.append((legacy_module, ex))

    attempted = ", ".join(f"{name}: {err}" for name, err in errors)
    raise ImportError(f"Unable to import backend. Attempted: {attempted}")
