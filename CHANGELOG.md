# Changelog

* Added direct Rust-vs-pycocotools parity coverage for fractional polygon rasterization, multi-polygon union, bbox rasterization, RLE bytes, decoded masks, areas, and derived boxes.

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed
* Optimize the Rust RLE IoU kernel by precomputing per-instance bounding boxes and detection areas once per matrix call, matching the original COCO C kernel structure instead of recomputing geometry for every mask pair.
* Add a strict Rust-vs-C/Cython CPU parity suite and make release CI gate on it.
* Build one `cp310-abi3` Rust wheel per platform and smoke-test that same artifact across supported Python versions instead of rebuilding per Python minor.
* Keep historical C/Cython extensions as development-only `*_legacy` reference modules rather than shipping both binary stacks.

### Fixed
* Preserve the historical `numpy.intp` return dtype for Soft-NMS indices in the Rust backend.
* Keep Soft-NMS's Rust-side pruning of disjoint boxes whose scores are already below threshold; the legacy Cython implementation only checked the threshold inside its overlap branch.

## Version 0.3.2 - Unreleased


## Version 0.3.1 - Released 2025-09-15

### Fixed

* Fix issue with deprecated numpy API 


## Version 0.2.1 - Released 2025-08-23

### Added
* Add 3.11 support

### Changes
* Added i686 wheels

## Version 0.2.0 - Released 2022-06-21

### Fixed
* Python 3.10 wheels

## Version 0.1.0 - Released 2022-06-21

### Added
* Initial port from kwimage

## Unreleased

### Rust backend transition

* Make PEP-517 builds Rust-first via maturin / PyO3 with an abi3 Python 3.10+
  extension, while retaining the Cython/C implementation as a temporary parity
  reference under `dev/build_legacy.sh`.
* Implement Rust box IoU/intersection/similarity kernels, COCO RLE
  encode/decode/area/merge/IoU/bbox/polygon conversion, and CPU NMS.
* Make backend dispatch capability-aware so a partial Rust module cannot shadow
  a working legacy backend.
* Harden mask input normalization: fix one-dimensional bbox handling, validate
  `iscrowd` length, distinguish Nx2 polygons from bbox lists, handle degenerate
  polygons as empty masks, and support NumPy 2 `__array__(dtype=..., copy=...)`
  in the legacy reference implementation.
* Prevent stale in-place pre-Rust extension modules from shadowing the Rust-first
  public API in editable checkouts by registering non-colliding compatibility
  modules under the historical import names.
* Rename future Cython parity builds to explicit `*_legacy` extension names,
  make backend forcing dynamic/testable, and make Rust validation fail when
  parity tests fail rather than reporting build success alone.
