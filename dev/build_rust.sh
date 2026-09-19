#!/usr/bin/env bash
set -euo pipefail

# Rust-first local development build.  The resulting extension is installed as
# kwimage_ext._rust while the Python compatibility shims keep the public API.
python -m pip install 'maturin>=1.7,<2.0'
python -m maturin develop --manifest-path rust/Cargo.toml
python - <<'PY'
import kwimage_ext._rust as rust
print('rust version =', rust.version())
print('rust capabilities =', rust.capabilities())
PY
