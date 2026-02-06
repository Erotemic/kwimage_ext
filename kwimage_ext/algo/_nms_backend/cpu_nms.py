from kwimage_ext._rust_dispatch import import_backend

_backend = import_backend("kwimage_ext._rust", "kwimage_ext.algo._nms_backend.cpu_nms_legacy")
cpu_nms = _backend.cpu_nms
