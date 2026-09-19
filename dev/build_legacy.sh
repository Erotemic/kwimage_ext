#!/usr/bin/env bash
set -euo pipefail

# Temporary parity/reference build for the historical Cython/C extensions.
# PEP-517 / normal wheel builds are Rust-first; this path exists only while the
# Rust port is being validated against the legacy kernels.  Legacy modules are
# built under explicit *_legacy names so they can never shadow the public
# Rust-first compatibility modules.
python dev/clean_stale_legacy_artifacts.py
python -m pip install -r requirements/build.txt
python setup.py build_ext --inplace
