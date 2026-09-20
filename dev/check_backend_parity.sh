#!/usr/bin/env bash
set -euo pipefail

# Build the production Rust extension, build the historical CPU reference
# extensions under non-colliding *_legacy names, then require direct parity.
python -m pip install -e .
./dev/build_legacy.sh
KWIMAGE_EXT_REQUIRE_LEGACY_PARITY=1 \
    python -m pytest -q tests/test_backend_parity.py "$@"
