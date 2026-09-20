#!/usr/bin/env bash
__doc__="
Runs cibuildwheel to create linux binary wheels.

Requirements:
    pip install cibuildwheel

SeeAlso:
    pyproject.toml
"

if ! which docker ; then
    echo "Missing requirement: docker. Please install docker before running build_wheels.sh"
    exit 1
fi
if ! which cibuildwheel ; then
    echo "The cibuildwheel module is not installed. Please pip install cibuildwheel before running build_wheels.sh"
    exit 1
fi

# The package uses PyO3 abi3-py310.  Always build from the CPython 3.10
# baseline unless the caller explicitly overrides CIBW_BUILD.
export CIBW_BUILD="${CIBW_BUILD:-cp310-*}"
echo "CIBW_BUILD = $CIBW_BUILD"

# Never let an older same-version wheel survive into release validation.
mkdir -p wheelhouse
rm -f wheelhouse/kwimage_ext*.whl

cibuildwheel --config-file pyproject.toml --platform linux --archs x86_64  #  kwimage_ext: +UNCOMMENT_IF(binpy)
python dev/validate_wheel_artifact.py wheelhouse/kwimage_ext*.whl
