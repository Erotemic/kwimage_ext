import importlib

import numpy as np
import pytest


rust = pytest.importorskip('kwimage_ext._rust')


def test_rust_capabilities_are_complete_for_shims():
    caps = set(rust.capabilities())
    assert {'boxes', 'mask-rle', 'mask-iou', 'cpu-nms'} <= caps


def test_public_shims_are_actually_using_rust_when_forced():
    from kwimage_ext.structs._boxes_backend import cython_boxes
    from kwimage_ext.structs._mask_backend import cython_mask
    from kwimage_ext.algo._nms_backend import cpu_nms

    assert cython_boxes.backend_metadata()['kind'] == 'rust'
    assert cython_mask.backend_metadata()['kind'] == 'rust'
    assert cpu_nms.backend_metadata()['kind'] == 'rust'


def test_box_iou_known_values():
    boxes = np.array([[0, 0, 9, 9], [0, 0, 4, 4]], dtype=np.float32)
    query = np.array([[0, 0, 9, 9]], dtype=np.float32)
    got = rust.bbox_ious_c(boxes, query, bias=1.0)
    assert got.dtype == np.float32
    assert np.allclose(got[:, 0], [1.0, 0.25])


def test_cpu_nms_known_case():
    ltrb = np.array([
        [0, 0, 10, 10],
        [0, 0, 10, 10],
        [10, 10, 20, 20],
    ], dtype=np.float32)
    scores = np.array([0.1, 0.9, 0.8], dtype=np.float32)
    keep = rust.cpu_nms(ltrb, scores, 0.5, bias=0.0)
    assert keep == [1, 2]


def test_mask_roundtrip_and_area():
    from kwimage_ext.structs._mask_backend import cython_mask

    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[1:5, 2:7] = 1
    rle = cython_mask.encode(np.asfortranarray(mask))[0]
    recon = cython_mask.decode([rle])[:, :, 0]
    assert np.array_equal(mask, recon)
    assert int(cython_mask.area([rle])[0]) == int(mask.sum())
    assert np.array_equal(cython_mask.toBbox([rle])[0], [2, 1, 5, 4])


def test_rle_codec_roundtrip():
    counts = np.array([0, 3, 2, 5, 7, 1, 11], dtype=np.uint32)
    encoded = rust._rle_array_to_bytes(counts)
    decoded = rust._rle_bytes_to_array(encoded)
    assert np.array_equal(counts, decoded)


def test_iou_accepts_one_dimensional_bbox_and_validates_crowd():
    from kwimage_ext.structs._mask_backend import cython_mask

    dt = np.array([0, 0, 10, 10], dtype=np.float64)
    gt = np.array([0, 0, 10, 10], dtype=np.float64)
    got = cython_mask.iou(dt, gt, [0])
    assert np.allclose(got, [[1.0]])
    with pytest.raises(ValueError):
        cython_mask.iou(dt, gt, [])


def test_frpyobjects_nx2_quad_is_polygon_not_four_bboxes():
    from kwimage_ext.structs._mask_backend import cython_mask

    quad = np.array([[1, 1], [1, 5], [5, 5], [5, 1]], dtype=float)
    rle = cython_mask.frPyObjects(quad, 8, 8)
    mask = cython_mask.decode([rle])[:, :, 0]
    assert mask.sum() > 0
    bbox = cython_mask.toBbox([rle])[0]
    assert bbox[2] > 1 and bbox[3] > 1


def test_degenerate_polygon_is_empty_not_crash():
    from kwimage_ext.structs._mask_backend import cython_mask

    rle = cython_mask.frPoly([[1, 1, 3, 3]], 8, 8)[0]
    mask = cython_mask.decode([rle])[:, :, 0]
    assert mask.sum() == 0


def test_pycocotools_mask_parity_if_available():
    pycoco = pytest.importorskip('pycocotools.mask')
    from kwimage_ext.structs._mask_backend import cython_mask

    rng = np.random.RandomState(0)
    masks = (rng.rand(32, 24, 6) > 0.8).astype(np.uint8, order='F')
    ours = cython_mask.encode(masks)
    theirs = pycoco.encode(masks)
    # Compression bytes should be canonical for the same uncompressed RLE.
    assert [r['counts'] for r in ours] == [r['counts'] for r in theirs]
    crowd = [0] * len(ours)
    got = cython_mask.iou(ours, ours, crowd)
    want = pycoco.iou(theirs, theirs, crowd)
    assert np.array_equal(got, want)


def test_rust_mask_iou_fragmented_matrix_parity_if_available():
    """Exercise the matrix path where bbox setup must be amortized."""
    pycoco = pytest.importorskip('pycocotools.mask')
    from kwimage_ext.structs._mask_backend import cython_mask

    rng = np.random.RandomState(12345)
    masks = (rng.rand(64, 64, 24) > 0.92).astype(np.uint8, order='F')
    ours = cython_mask.encode(masks)
    theirs = pycoco.encode(masks)
    crowd = [0] * len(ours)
    got = cython_mask.iou(ours, ours, crowd)
    want = pycoco.iou(theirs, theirs, crowd)
    assert np.array_equal(got, want)
