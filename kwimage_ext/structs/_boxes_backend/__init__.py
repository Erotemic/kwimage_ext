"""Public box backend compatibility package.

See the mask backend package for why the public historical submodule name is
registered explicitly.  This prevents stale in-place Cython extension files
from shadowing the Rust-first Python API in editable checkouts.
"""
from __future__ import annotations

import sys

from . import _api as cython_boxes

sys.modules[__name__ + '.cython_boxes'] = cython_boxes

__all__ = ['cython_boxes']
