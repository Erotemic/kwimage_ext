import numpy as np


def test_soft_nms_smoke():
    from kwimage_ext.algo._nms_backend import cpu_soft_nms
    ltrb = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = cpu_soft_nms.soft_nms(ltrb, scores)
    assert keep.shape[0] >= 1


def test_mask_encode_decode_roundtrip():
    from kwimage_ext.structs._mask_backend import cython_mask
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    rle = cython_mask.encode(np.asfortranarray(mask))
    dec = cython_mask.decode(rle)
    assert dec.shape[0] == 4 and dec.shape[1] == 4
