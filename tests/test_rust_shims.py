import importlib
import os
import sys
import types


def test_shim_modules_importable_without_extensions():
    boxes = importlib.import_module('kwimage_ext.structs._boxes_backend.cython_boxes')
    masks = importlib.import_module('kwimage_ext.structs._mask_backend.cython_mask')
    assert boxes is not None
    assert masks is not None


def test_historical_import_names_resolve_to_unshadowable_api_modules():
    from kwimage_ext.structs._boxes_backend import cython_boxes as boxes_attr
    from kwimage_ext.structs._mask_backend import cython_mask as masks_attr
    from kwimage_ext.algo._nms_backend import cpu_nms as nms_attr

    boxes_import = importlib.import_module(
        'kwimage_ext.structs._boxes_backend.cython_boxes')
    masks_import = importlib.import_module(
        'kwimage_ext.structs._mask_backend.cython_mask')
    nms_import = importlib.import_module(
        'kwimage_ext.algo._nms_backend.cpu_nms')

    assert boxes_import is boxes_attr
    assert masks_import is masks_attr
    assert nms_import is nms_attr
    # These are deliberately non-colliding Python API modules.  In particular,
    # a stale cython_mask*.so in an editable checkout must not win the import.
    assert boxes_attr.__name__.endswith('._api')
    assert masks_attr.__name__.endswith('._api')
    assert nms_attr.__name__.endswith('._cpu_nms_api')


def test_capability_aware_dispatch_rejects_partial_rust(monkeypatch):
    from kwimage_ext import _rust_dispatch

    monkeypatch.delenv('KWIMAGE_EXT_FORCE_RUST', raising=False)
    monkeypatch.delenv('KWIMAGE_EXT_FORCE_LEGACY', raising=False)

    rust = types.ModuleType('fake_rust')
    rust.only_one = lambda: None
    legacy = types.ModuleType('fake_legacy')
    legacy.need_a = lambda: 'a'
    legacy.need_b = lambda: 'b'

    modules = {
        'fake_rust': rust,
        'fake_legacy': legacy,
    }

    monkeypatch.setattr(
        _rust_dispatch.importlib,
        'import_module',
        lambda name: modules[name],
    )
    chosen = _rust_dispatch.import_backend(
        'fake_rust',
        'fake_legacy',
        required_symbols={'need_a', 'need_b'},
    )
    assert chosen is legacy


def test_capability_aware_dispatch_can_force_rust(monkeypatch):
    from kwimage_ext import _rust_dispatch

    monkeypatch.setenv('KWIMAGE_EXT_FORCE_RUST', '1')
    monkeypatch.delenv('KWIMAGE_EXT_FORCE_LEGACY', raising=False)

    rust = types.ModuleType('fake_rust')
    legacy = types.ModuleType('fake_legacy')
    legacy.need_a = lambda: 'a'
    monkeypatch.setattr(
        _rust_dispatch.importlib,
        'import_module',
        lambda name: {'fake_rust': rust, 'fake_legacy': legacy}[name],
    )
    try:
        _rust_dispatch.import_backend(
            'fake_rust', 'fake_legacy', required_symbols={'need_a'}
        )
    except ImportError as ex:
        assert 'missing required symbols' in str(ex)
    else:
        raise AssertionError('forcing Rust must reject a partial Rust backend')


def test_conflicting_force_flags_are_rejected(monkeypatch):
    from kwimage_ext import _rust_dispatch

    monkeypatch.setenv('KWIMAGE_EXT_FORCE_RUST', '1')
    monkeypatch.setenv('KWIMAGE_EXT_FORCE_LEGACY', '1')
    try:
        _rust_dispatch.import_backend('unused.rust', 'unused.legacy')
    except RuntimeError as ex:
        assert 'cannot both be enabled' in str(ex)
    else:
        raise AssertionError('conflicting backend force flags must be rejected')
