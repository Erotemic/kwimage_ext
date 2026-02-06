from kwimage_ext._rust_dispatch import import_backend

_backend = import_backend("kwimage_ext._rust", "kwimage_ext.algo._nms_backend.cpu_soft_nms_legacy")
soft_nms = _backend.soft_nms
