import numpy as np
import pytest


rust = pytest.importorskip('kwimage_ext._rust')


def _assert_encode_roundtrip_and_legacy(mask):
    mask = np.asfortranarray(mask, dtype=np.uint8)
    got = rust.encode(mask)
    decoded = rust.decode(got)
    np.testing.assert_array_equal(decoded, mask)

    legacy = pytest.importorskip(
        'kwimage_ext.structs._mask_backend.cython_mask_legacy')
    want = legacy.encode(mask)
    assert [item['counts'] for item in got] == [item['counts'] for item in want]


@pytest.mark.parametrize('shape', [
    (512, 512, 4),
    (17, 19, 3),
    (1, 257, 2),
    (257, 1, 2),
])
def test_structured_encode_long_runs_match_legacy(shape):
    h, w, n = shape
    mask = np.zeros(shape, dtype=np.uint8)
    if n > 0:
        mask[:, :, 0] = 1
    if n > 1:
        mask[:max(1, h // 2), :, 1] = 1
    if n > 2:
        mask[:, :max(1, w // 2), 2] = 1
    _assert_encode_roundtrip_and_legacy(mask)


def test_structured_encode_rectangles_match_legacy():
    rng = np.random.RandomState(54321)
    mask = np.zeros((513, 509, 4), dtype=np.uint8)
    for chan in range(mask.shape[2]):
        for _ in range(31):
            y1 = int(rng.randint(0, 500))
            x1 = int(rng.randint(0, 496))
            rh = int(rng.randint(1, 91))
            rw = int(rng.randint(1, 91))
            mask[y1:min(mask.shape[0], y1 + rh),
                 x1:min(mask.shape[1], x1 + rw), chan] = 1
    _assert_encode_roundtrip_and_legacy(mask)


def test_structured_encode_transitions_around_word_boundaries():
    # Exercise run lengths immediately around the 8/32-byte scan widths.
    lengths = [1, 7, 8, 9, 31, 32, 33, 63, 64, 65]
    flat = []
    value = 0
    for length in lengths:
        flat.extend([value] * length)
        value ^= 1
    data = np.asarray(flat, dtype=np.uint8)
    mask = np.asfortranarray(data.reshape((-1, 1, 1), order='F'))
    _assert_encode_roundtrip_and_legacy(mask)
