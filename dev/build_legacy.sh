#!/usr/bin/env bash
set -euo pipefail

# Temporary parity/reference build for the historical Cython/C extensions.
# PEP-517 / normal wheel builds are Rust-first; this path exists only while the
# Rust port is being validated against the legacy kernels. Legacy modules are
# built under explicit *_legacy names so they can never shadow the public
# Rust-first compatibility modules.
python dev/clean_stale_legacy_artifacts.py
python -m pip install -r requirements/build.txt

# scikit-build caches interpreter, Cython, and NumPy discovery under _skbuild.
# Reusing that directory after switching virtualenvs can silently combine the
# active Python interpreter with Cython / NumPy paths from another environment.
# A parity oracle must never be built from that mixed state.
rm -rf _skbuild _cmake_test_compile

PYTHON_EXE="$(python -c 'import sys; print(sys.executable)')"
PYTHON_SCRIPTS="$(python -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
if [[ "${OS:-}" == "Windows_NT" ]]; then
    CYTHON_EXE="${PYTHON_SCRIPTS}/cython.exe"
else
    CYTHON_EXE="${PYTHON_SCRIPTS}/cython"
fi

if [[ ! -x "$CYTHON_EXE" ]]; then
    echo "error: Cython executable for the active Python was not found: $CYTHON_EXE" >&2
    echo "active Python: $PYTHON_EXE" >&2
    exit 2
fi

# Validate that Cython is importable by the same interpreter before asking
# CMake to use its console script. This produces a direct diagnostic instead
# of letting FindCython discover a stale executable elsewhere on PATH.
python - <<'PY'
import Cython
import numpy
import sys
print(f'legacy reference Python: {sys.executable}')
print(f'legacy reference Cython: {Cython.__version__} ({Cython.__file__})')
print(f'legacy reference NumPy: {numpy.__version__} ({numpy.__file__})')
PY
printf 'legacy reference Cython executable: %s\n' "$CYTHON_EXE"

# The parity oracle is CPU-only. CMAKE_ARGS is honored by scikit-build and
# prevents a CUDA-capable developer machine from turning this reference build
# into an unrelated GPU build as a side effect. Pin FindCython to the active
# interpreter environment as an additional guard against PATH/cache leakage.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DUSE_CUDA=OFF -DCYTHON_EXECUTABLE:FILEPATH=${CYTHON_EXE}"
python setup.py build_ext --inplace

# Fail here, with one direct diagnostic, if the Cython-generated initializer
# does not agree with the intentionally renamed extension filename.  The
# parity suite should only start after all four reference modules are proven
# importable from this source checkout.
python - <<'PY'
import importlib

module_names = [
    'kwimage_ext.structs._boxes_backend.cython_boxes_legacy',
    'kwimage_ext.structs._mask_backend.cython_mask_legacy',
    'kwimage_ext.algo._nms_backend.cpu_nms_legacy',
    'kwimage_ext.algo._nms_backend.cpu_soft_nms_legacy',
]
for name in module_names:
    module = importlib.import_module(name)
    print(f'legacy reference import OK: {name} -> {module.__file__}')
PY
