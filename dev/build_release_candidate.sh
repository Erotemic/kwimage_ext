#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-_release_candidate}
rm -rf "$OUT"
mkdir -p "$OUT/wheelhouse"

# Build a fresh non-editable PEP-517 artifact.  pip creates an isolated build
# environment, so maturin does not have to be installed in the developer env.
python -m pip wheel --no-deps --wheel-dir "$OUT/wheelhouse" . \
    > "$OUT/wheel_build.log" 2>&1

WHEELS=("$OUT"/wheelhouse/kwimage_ext*.whl)
if [ ! -e "${WHEELS[0]}" ]; then
    echo "No kwimage_ext wheel was produced" >&2
    exit 1
fi

python dev/validate_wheel_artifact.py --benchmark \
    --output "$OUT/wheel_validation.json" "${WHEELS[@]}" \
    > "$OUT/wheel_validation.log" 2>&1

python -m pytest -q tests/test_release_packaging.py \
    > "$OUT/release_packaging_tests.log" 2>&1

sha256sum "${WHEELS[@]}" > "$OUT/SHA256SUMS"
python - <<'PY' "$OUT" > "$OUT/wheel_inventory.txt"
import sys, zipfile
from pathlib import Path
out = Path(sys.argv[1])
for wheel in sorted((out / 'wheelhouse').glob('kwimage_ext*.whl')):
    print(f'[{wheel.name}]')
    with zipfile.ZipFile(wheel) as zf:
        for name in sorted(zf.namelist()):
            print(name)
PY

tar -czf "$OUT/kwimage_ext_release_candidate_evidence.tar.gz" \
    -C "$OUT" \
    wheel_build.log wheel_validation.json wheel_validation.log \
    release_packaging_tests.log SHA256SUMS wheel_inventory.txt

echo "release candidate validated"
echo "wheel(s): ${WHEELS[*]}"
echo "upload: $OUT/kwimage_ext_release_candidate_evidence.tar.gz"
