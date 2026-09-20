<<<<<<< HEAD
"""Historical import name; implementation lives in :mod:`._cpu_nms_api`."""
from ._cpu_nms_api import *  # noqa: F401,F403
from ._cpu_nms_api import backend_metadata  # noqa: F401
||||||| a8e6a11
=======
from kwimage_ext._rust_dispatch import import_backend

_backend = import_backend("kwimage_ext._rust", "kwimage_ext.algo._nms_backend.cpu_nms_legacy")
cpu_nms = _backend.cpu_nms
>>>>>>> e609a9f61ad9b84d176f20f02d2a8ba42e2b42a5
