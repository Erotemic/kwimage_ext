#!/usr/bin/env bash
set -euo pipefail

__doc__="
Runs cibuildwheel to create linux binary wheels.

Requirements:
    pip install cibuildwheel

SeeAlso:
    pyproject.toml
"

if ! command -v docker >/dev/null 2>&1 ; then
    echo "Missing requirement: docker. Please install docker before running build_wheels.sh"
    exit 1
fi
if ! command -v cibuildwheel >/dev/null 2>&1 ; then
    echo "The cibuildwheel module is not installed. Please pip install cibuildwheel before running build_wheels.sh"
    exit 1
fi

# Reusable/stable-ABI wheel selection is packaging policy.
# Honor [tool.cibuildwheel].build unless the caller explicitly
# supplied CIBW_BUILD in the environment.
if [[ -n "${CIBW_BUILD:-}" ]]; then
    echo "CIBW_BUILD override = $CIBW_BUILD"
else
    echo "CIBW_BUILD = <from pyproject.toml>"
fi

# Recreate the output directory so stale wheels cannot satisfy
# post-build validation or be mistaken for this build.
rm -rf wheelhouse
mkdir -p wheelhouse

cibuildwheel --config-file pyproject.toml --platform linux --archs x86_64

# Project-owned artifact validation.
python dev/validate_wheel_artifact.py wheelhouse/kwimage_ext*.whl
