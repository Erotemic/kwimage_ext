import importlib

import numpy as np
import pytest


rust = pytest.importorskip('kwimage_ext._rust')


def test_rust_capabilities_are_complete_for_shims():
    caps = set(rust.capabilities())
    assert {'boxes', 'mask-rle', 'mask-iou', 'cpu-nms', 'soft-nms'} <= caps


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


def test_bbox_similarities_known_formula():
    boxes = np.array([[0, 0, 9, 9]], dtype=np.float32)
    query = np.array([
        [9, 9, 12, 12],
        [10, 10, 10, 10],
    ], dtype=np.float32)
    got = rust.bbox_similarities(boxes, query)
    # These values use floating-point absolute values.  The legacy Cython
    # implementation accidentally truncated fractional distances via C abs().
    want = np.array([[0.4473846, 0.018900394]], dtype=np.float32)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


def test_cpu_nms_known_case():
    ltrb = np.array([
        [0, 0, 10, 10],
        [0, 0, 10, 10],
        [10, 10, 20, 20],
    ], dtype=np.float32)
    scores = np.array([0.1, 0.9, 0.8], dtype=np.float32)
    keep = rust.cpu_nms(ltrb, scores, 0.5, bias=0.0)
    assert keep == [1, 2]



def test_cpu_nms_equal_scores_prefer_earlier_index():
    ltrb = np.array([
        [0, 0, 10, 10],
        [0, 0, 10, 10],
        [50, 50, 60, 60],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.9, 0.8], dtype=np.float32)
    keep = rust.cpu_nms(ltrb, scores, 0.5, bias=0.0)
    assert keep == [0, 2]


def test_soft_nms_known_case():
    from kwimage_ext.algo._nms_backend import cpu_soft_nms

    ltrb = np.array([
        [0, 0, 10, 10],
        [0, 0, 10, 10],
        [50, 50, 60, 60],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = cpu_soft_nms.soft_nms(ltrb, scores)
    assert isinstance(keep, np.ndarray)
    assert keep.ndim == 1
    assert keep[0] == 0
    assert 2 in keep

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


def test_rust_polygon_rasterization_matches_pycocotools_if_available():
    pycoco = pytest.importorskip('pycocotools.mask')
    from kwimage_ext.structs._mask_backend import cython_mask

    h, w = 64, 80
    polygons = [
        [3.2, 4.7, 31.6, 6.1, 29.4, 28.8, 5.1, 30.2],
        [18.25, 12.75, 55.4, 13.1, 61.8, 46.6, 20.2, 50.3],
        [7.9, 40.2, 14.1, 35.6, 25.7, 42.8, 22.4, 57.1, 9.0, 55.0],
    ]
    ours = cython_mask.frPoly(polygons, h, w)
    theirs = pycoco.frPyObjects(polygons, h, w)
    assert [r['counts'] for r in ours] == [r['counts'] for r in theirs]
    assert np.array_equal(cython_mask.decode(ours), pycoco.decode(theirs))
    assert np.array_equal(cython_mask.area(ours), pycoco.area(theirs))
    assert np.array_equal(cython_mask.toBbox(ours), pycoco.toBbox(theirs))


def test_rust_polygon_merge_matches_pycocotools_if_available():
    pycoco = pytest.importorskip('pycocotools.mask')
    from kwimage_ext.structs._mask_backend import cython_mask

    h, w = 48, 48
    polygons = [
        [2.2, 3.3, 23.8, 4.1, 20.2, 25.7, 4.0, 22.9],
        [16.5, 14.2, 40.2, 15.8, 39.1, 39.4, 18.4, 37.6],
    ]
    ours_parts = cython_mask.frPoly(polygons, h, w)
    their_parts = pycoco.frPyObjects(polygons, h, w)
    ours = cython_mask.merge(ours_parts)
    theirs = pycoco.merge(their_parts)
    assert ours['counts'] == theirs['counts']
    assert np.array_equal(cython_mask.decode([ours])[:, :, 0], pycoco.decode(theirs))


def test_rust_bbox_rasterization_matches_pycocotools_if_available():
    pycoco = pytest.importorskip('pycocotools.mask')
    from kwimage_ext.structs._mask_backend import cython_mask

    h, w = 64, 80
    boxes = np.array([
        [3.2, 4.7, 20.3, 15.8],
        [12.0, 18.25, 31.4, 28.6],
        [0.0, 0.0, 8.5, 9.5],
    ], dtype=np.float64)
    ours = cython_mask.frBbox(boxes, h, w)
    theirs = pycoco.frPyObjects(boxes, h, w)
    assert [r['counts'] for r in ours] == [r['counts'] for r in theirs]
    assert np.array_equal(cython_mask.decode(ours), pycoco.decode(theirs))
