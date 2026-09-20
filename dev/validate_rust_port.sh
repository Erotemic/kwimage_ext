#!/usr/bin/env bash
set -uo pipefail

OUT=${1:-_rust_validation}
rm -rf "$OUT"
mkdir -p "$OUT"

{
    echo "date=$(date -Iseconds)"
    echo "python=$(python -VV 2>&1)"
    command -v rustc >/dev/null && rustc --version || echo "rustc=missing"
    command -v cargo >/dev/null && cargo --version || echo "cargo=missing"
    python -m maturin --version 2>&1 || true
    git rev-parse HEAD 2>&1 || true
    git status --short 2>&1 || true
    echo "legacy_artifacts_before:"
    find kwimage_ext -type f \( \
        -name 'cython_mask*.so' -o \
        -name 'cython_boxes*.so' -o \
        -name 'cpu_nms*.so' \
    \) -print 2>/dev/null || true
} > "$OUT/environment.txt"

# Record Cargo's resolved view of the crate before invoking the Python build
# backend.  This makes workspace/configuration failures immediately visible in
# the validation bundle.
if command -v cargo >/dev/null; then
    cargo metadata --format-version 1 --no-deps --manifest-path rust/Cargo.toml \
        > "$OUT/cargo_metadata.json" 2> "$OUT/cargo_metadata.stderr"
    CARGO_METADATA_RC=$?
else
    CARGO_METADATA_RC=127
    echo 'cargo unavailable' > "$OUT/cargo_metadata.stderr"
    : > "$OUT/cargo_metadata.json"
fi
echo "$CARGO_METADATA_RC" > "$OUT/cargo_metadata.returncode"

python -m pip install -v -e . > "$OUT/build.log" 2>&1
BUILD_RC=$?
echo "$BUILD_RC" > "$OUT/build.returncode"

if [ "$BUILD_RC" -eq 0 ]; then
    KWIMAGE_EXT_FORCE_RUST=1 python -m pytest -q \
        tests/test_rust_backend.py tests/test_rust_shims.py tests/test_rust_mask_and_softnms.py tests/test_rust_assignment.py \
        > "$OUT/tests.log" 2>&1
    TEST_RC=$?
else
    TEST_RC=99
    echo "tests skipped because build failed" > "$OUT/tests.log"
fi
echo "$TEST_RC" > "$OUT/tests.returncode"

KWIMAGE_EXT_FORCE_RUST=1 python - <<'PY' > "$OUT/imports.txt" 2>&1
try:
    import kwimage_ext
    print('kwimage_ext=', kwimage_ext.__file__)
    print('version=', kwimage_ext.__version__)
except Exception as ex:
    print('kwimage_ext import failed:', repr(ex))
try:
    import kwimage_ext._rust as rust
    print('rust=', rust.__file__)
    print('caps=', rust.capabilities())
except Exception as ex:
    print('rust import failed:', repr(ex))
try:
    from kwimage_ext.structs._boxes_backend import cython_boxes
    print('boxes_api=', cython_boxes.__file__)
    print('boxes_api_name=', cython_boxes.__name__)
    print('boxes_backend=', cython_boxes.backend_metadata())
except Exception as ex:
    print('boxes import failed:', repr(ex))
try:
    from kwimage_ext.structs._mask_backend import cython_mask
    print('mask_api=', cython_mask.__file__)
    print('mask_api_name=', cython_mask.__name__)
    print('mask_backend=', cython_mask.backend_metadata())
except Exception as ex:
    print('mask import failed:', repr(ex))
try:
    from kwimage_ext.algo._nms_backend import cpu_nms
    print('nms_api=', cpu_nms.__file__)
    print('nms_api_name=', cpu_nms.__name__)
    print('nms_backend=', cpu_nms.backend_metadata())
except Exception as ex:
    print('nms import failed:', repr(ex))
try:
    from kwimage_ext.algo import assignment
    print('assignment_api=', assignment.__file__)
    print('assignment_backend=', assignment.backend_metadata())
except Exception as ex:
    print('assignment import failed:', repr(ex))
PY

python -m pip freeze > "$OUT/pip_freeze.txt" 2>&1 || true

tar -czf "$OUT/kwimage_ext_rust_validation.tar.gz" \
    -C "$OUT" \
    environment.txt cargo_metadata.json cargo_metadata.stderr cargo_metadata.returncode build.log build.returncode tests.log tests.returncode imports.txt pip_freeze.txt

echo "build_rc=$BUILD_RC test_rc=$TEST_RC"
echo "upload: $OUT/kwimage_ext_rust_validation.tar.gz"

if [ "$BUILD_RC" -ne 0 ]; then
    exit "$BUILD_RC"
fi
exit "$TEST_RC"
