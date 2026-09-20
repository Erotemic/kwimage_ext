"""Public mask backend compatibility package.

The historical public import is ``kwimage_ext.structs._mask_backend.cython_mask``.
Editable source trees may still contain an old in-place ``cython_mask*.so`` from
pre-Rust builds.  CPython normally prefers extension modules over ``.py`` files,
which would silently bypass the Rust-first compatibility shim.

Register the non-colliding Python API module under the historical submodule name
while the package is initialized.  This makes the public import deterministic
and leaves legacy binary kernels available only through explicit ``*_legacy``
names during parity testing.
"""
from __future__ import annotations

import sys

from . import _api as cython_mask

sys.modules[__name__ + '.cython_mask'] = cython_mask

__all__ = ['cython_mask']
