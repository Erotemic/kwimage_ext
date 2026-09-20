"""Public NMS backend compatibility package."""
from __future__ import annotations

import sys

from . import _cpu_nms_api as cpu_nms
from . import _cpu_soft_nms_api as cpu_soft_nms

# Preserve historical import names while preventing stale in-place extension
# modules from shadowing the Rust-first API shims in editable checkouts.
sys.modules[__name__ + '.cpu_nms'] = cpu_nms
sys.modules[__name__ + '.cpu_soft_nms'] = cpu_soft_nms

__all__ = ['cpu_nms', 'cpu_soft_nms']
