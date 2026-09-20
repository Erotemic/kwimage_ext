#!/usr/bin/env bash
set -euo pipefail

# The caller must install the project test environment first. Generated CI does
# this explicitly via tool.xcookie.ci_source_checks, which prevents pytest (or
# any other test dependency) from being supplied accidentally by the runner.
# Build the historical CPU reference extensions under non-colliding *_legacy
# names, then require direct Rust-vs-legacy parity.
./dev/build_legacy.sh
KWIMAGE_EXT_REQUIRE_LEGACY_PARITY=1 \
    python -m pytest -q tests/test_backend_parity.py "$@"
