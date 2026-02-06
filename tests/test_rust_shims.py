import importlib
import numpy as np


def test_shim_modules_importable_without_extensions():
    boxes = importlib.import_module('kwimage_ext.structs._boxes_backend.cython_boxes')
    masks = importlib.import_module('kwimage_ext.structs._mask_backend.cython_mask')
    assert boxes is not None
    assert masks is not None


def test_boxes_overlap_shape_and_dtype():
    boxes_mod = importlib.import_module('kwimage_ext.structs._boxes_backend.cython_boxes')
    boxes = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=np.float32)
    q = np.array([[0, 0, 10, 10]], dtype=np.float32)
    got = boxes_mod.bbox_overlaps(boxes, q)
    assert got.shape == (2, 1)
    assert got.dtype == np.float32
    assert np.isclose(got[0, 0], 1.0)


def test_cpu_nms_smoke():
    mod = importlib.import_module('kwimage_ext.algo._nms_backend.cpu_nms')
    ltrb = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = mod.cpu_nms(ltrb, scores, 0.5, 0.0)
    assert keep[0] == 0
    assert 2 in keep
