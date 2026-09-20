# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 0.3.2 - Unreleased

### Added
* Add an area-by-threshold Rust COCO greedy-assignment grid kernel so candidate geometry and prediction order cross PyO3 once per image instead of once per area range.
* Add a batched Rust COCO-style greedy assignment kernel over sparse CSR candidate graphs, with crowd reuse, annotation-ignore fallback, threshold-boundary control, and direct randomized Python-reference parity tests.
* Add a direct assignment microbenchmark for measuring PyO3 kernel speed independently from detector geometry and COCO accumulation.
* Add direct Rust-vs-pycocotools parity coverage for fractional polygon rasterization, multi-polygon union, bbox rasterization, RLE bytes, decoded masks, areas, and derived boxes.
* Add a strict Rust-vs-C/Cython CPU parity suite and make release CI gate on it.

### Changed
* Make PEP-517 builds Rust-first via maturin / PyO3 with an abi3 Python 3.10+ extension, while retaining the Cython/C implementation as a temporary parity reference under `dev/build_legacy.sh`.
* Implement Rust box IoU/intersection/similarity kernels, COCO RLE encode/decode/area/merge/IoU/bbox/polygon conversion, CPU NMS, and Soft-NMS.
* Optimize the Rust RLE IoU kernel by precomputing per-instance bounding boxes and detection areas once per matrix call, matching the original COCO C kernel structure instead of recomputing geometry for every mask pair.
* Build one `cp310-abi3` Rust wheel per platform and smoke-test that same artifact across supported Python versions instead of rebuilding per Python minor.
* Keep historical C/Cython extensions as development-only `*_legacy` reference modules rather than shipping both binary stacks.
* Make `pyproject.toml` the single source of truth for production package metadata while keeping `setup.py` only as a legacy parity-build entry point.
* Install the exact wheel artifact in CI instead of resolving `kwimage_ext` by version from package indexes during artifact tests.
* Make the Rust-first parity, reusable ABI3 wheel matrix, artifact validation, and GitHub/GitLab workflows declarative through xcookie configuration rather than hand-maintained CI edits.
* Make backend dispatch capability-aware so a partial Rust module cannot shadow a working legacy backend.
* Rename future Cython parity builds to explicit `*_legacy` extension names, make backend forcing dynamic/testable, and make Rust validation fail when parity tests fail rather than reporting build success alone.

### Fixed
* Preserve the historical `numpy.intp` return dtype for Soft-NMS indices in the Rust backend.
* Match historical CPU-NMS equal-score behavior by preferring the earlier input index deterministically.
* Match the legacy Soft-NMS in-place mutation semantics when pruning overlapping boxes, including the observable inactive array tail.
* Keep Soft-NMS's Rust-side pruning of disjoint boxes whose scores are already below threshold; the legacy Cython implementation only checked the threshold inside its overlap branch.
* Keep Rust `bbox_similarities` on the intended floating-point formula instead of reproducing the legacy Cython integer-`abs` truncation defect; direct parity tests document this intentional difference.
* Harden mask input normalization: fix one-dimensional bbox handling, validate `iscrowd` length, distinguish Nx2 polygons from bbox lists, handle degenerate polygons as empty masks, and support NumPy 2 `__array__(dtype=..., copy=...)` in the legacy reference implementation.
* Prevent stale in-place pre-Rust extension modules from shadowing the Rust-first public API in editable checkouts by registering non-colliding compatibility modules under the historical import names.

## Version 0.3.1 - Released 2025-09-15

### Fixed
* Fix issue with deprecated numpy API.

## Version 0.2.1 - Released 2025-08-23

### Added
* Add Python 3.11 support.

### Changed
* Add i686 wheels.

## Version 0.2.0 - Released 2022-06-21

### Fixed
* Add Python 3.10 wheels.

## Version 0.1.0 - Released 2022-06-21

### Added
* Initial port from kwimage.
