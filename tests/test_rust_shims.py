import importlib


def test_shim_modules_importable_without_extensions():
    boxes = importlib.import_module('kwimage_ext.structs._boxes_backend.cython_boxes')
    masks = importlib.import_module('kwimage_ext.structs._mask_backend.cython_mask')
    assert boxes is not None
    assert masks is not None


def test_boxes_shim_reports_missing_backend_on_call():
    boxes = importlib.import_module('kwimage_ext.structs._boxes_backend.cython_boxes')
    try:
        boxes.bbox_overlaps([], [])
    except ImportError as ex:
        assert 'Unable to import backend' in str(ex)
    else:  # pragma: no cover
        raise AssertionError('expected ImportError when no backend is available')
