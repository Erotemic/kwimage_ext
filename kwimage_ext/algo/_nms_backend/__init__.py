"""Public NMS backend compatibility package."""
from __future__ import annotations

import sys

from . import _cpu_nms_api as cpu_nms

# Preserve the historical CPU NMS import while preventing a stale in-place
# ``cpu_nms*.so`` from shadowing the Rust-first shim.
sys.modules[__name__ + '.cpu_nms'] = cpu_nms

__all__ = ['cpu_nms']
