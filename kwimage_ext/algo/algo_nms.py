"""
Python frontend for binary backends with tests
"""
# import numpy as np
# import warnings
try:
    import torch
except Exception:
    torch = None


# class _NMS_Impls():
#     # TODO: could make this prettier
#     def __init__(self):
#         self._funcs = None
#     def _lazy_init(self):
#         _funcs = {}
#         try:
#             from kwimage_ext.algo._nms_backend import cpu_nms
#             _funcs['cython_cpu'] = cpu_nms.cpu_nms
#         except Exception as ex:
#             warnings.warn(
#                 'optional cpu_nms is not available: {}'.format(str(ex)))
#         try:
#             if torch is not None and torch.cuda.is_available():
#                 from kwimage_ext.algo._nms_backend import gpu_nms
#                 _funcs['cython_gpu'] = gpu_nms.gpu_nms
#         except Exception as ex:
#             warnings.warn
#             ('optional gpu_nms is not available: {}'.format(str(ex)))
#         self._funcs = _funcs
#         self._valid = frozenset(_impls._funcs.keys())
# _impls = _NMS_Impls()


def cython_cpu_nms(ltrb, scores, thresh, bias=0.0):
    """
    Example:
        >>> import numpy as np
        >>> from kwimage_ext.algo.algo_nms import *  # NOQA
        >>> ltrb = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>>     [100, 100, 150, 101],
        >>>     [120, 100, 180, 101],
        >>>     [150, 100, 200, 101],
        >>> ], dtype=np.float32)
        >>> scores = np.linspace(0, 1, len(ltrb)).astype(np.float32)
        >>> thresh = .2
        >>> bias = 0.0
        >>> keep = cython_cpu_nms(ltrb, scores, thresh, bias)
        >>> assert set(keep) == {6, 4, 3, 1}
    """
    from kwimage_ext.algo._nms_backend import cpu_nms
    keep = cpu_nms.cpu_nms(ltrb=ltrb, scores=scores, thresh=thresh, bias=bias)
    return keep


def cython_gpu_gtms(ltrb, scores, thresh, bias=0.0, device_id=None):
    """
    FIXME: Broken

    Example:
        >>> # xdoctest: +SKIP
        >>> from kwimage_ext.algo.algo_nms import *  # NOQA
        >>> ltrb = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>>     [100, 100, 150, 101],
        >>>     [120, 100, 180, 101],
        >>>     [150, 100, 200, 101],
        >>> ], dtype=np.float32)
        >>> scores = np.linspace(0, 1, len(ltrb)).astype(np.float32)
        >>> thresh = .2
        >>> bias = 0.0
        >>> keep = cython_gpu_nms(ltrb, scores, thresh, bias)
        >>> device_id = None
        >>> assert set(keep) == {6, 4, 3, 1}
    """
    from kwimage_ext.algo._nms_backend import gpu_nms
    # TODO: if the data is already on a torch GPU can we just
    # use it?
    # HACK: we should parameterize which device is used
    if device_id is None:
        device_id = torch.cuda.current_device()
    keep = gpu_nms.gpu_nms(ltrb, scores, float(thresh), bias=float(bias),
                           device_id=device_id)
    return keep
