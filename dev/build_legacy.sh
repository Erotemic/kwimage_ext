#!/usr/bin/env bash
set -euo pipefail

# Temporary parity/reference build for the historical Cython/C extensions.
# PEP-517 / normal wheel builds are Rust-first; this path exists only while the
# Rust port is being validated against the legacy kernels.  Legacy modules are
# built under explicit *_legacy names so they can never shadow the public
# Rust-first compatibility modules.
python dev/clean_stale_legacy_artifacts.py
python -m pip install -r requirements/build.txt

# The parity oracle is CPU-only.  CMAKE_ARGS is honored by scikit-build and
# prevents a CUDA-capable developer machine from turning this reference build
# into an unrelated GPU build as a side effect.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DUSE_CUDA=OFF"
python setup.py build_ext --inplace
