# Rust kernel profiling

`profile_rust_kernels.py` produces one self-contained evidence tarball for
performance review. It is designed to answer both "how fast is this Python API
call?" and "why is the native kernel spending those cycles?".

Typical full capture:

```bash
python dev/profiling/profile_rust_kernels.py \
    --output ~/Downloads/kwimage-ext-rust-profile.tar.gz
```

For authoritative portable performance evidence, rebuild the local Rust
extension and same-checkout legacy Cython/C reference first:

```bash
python dev/profiling/profile_rust_kernels.py \
    --rebuild-profiled \
    --output ~/Downloads/kwimage-ext-rust-profile.tar.gz
```

`--rebuild-profiled` uses `maturin develop --release`, release optimization,
debug information, and `-C target-cpu=generic`. Ambient architecture-changing
Rust flags are ignored. It also runs `dev/build_legacy.sh` so legacy comparison
rows are tied to the historical Cython/C source in the same checkout. The exact
Rust, legacy, and pycocotools binaries that ran are SHA-256 fingerprinted and
copied into the bundle when available.

The legacy helper is pinned to the same Python executable used by the evidence
runner. Its Cython/NumPy build requirements are settled before the Rust rebuild,
so both native backends are built against the final benchmark environment.

When the evidence is intended to support a claim against a particular
pycocotools release, add `--require-pycocotools-version X.Y.Z`. The run then
fails unless that exact version is installed, its native `_mask` artifact is
fingerprinted, and direct pycocotools benchmark rows were collected.

A short smoke capture is available as:

```bash
python dev/profiling/profile_rust_kernels.py --quick --no-perf
```

## What the bundle contains

- `summary.md`: first-pass timing and profiler status summary.
- `benchmarks/results.json`: deterministic multi-seed microbenchmark samples,
  paired rotating-order Rust/comparator ratios, and backend provenance.
- `benchmarks/comparisons.{json,csv}`: claim-oriented paired ratios with
  descriptive ±5% bands.
- `correctness/`: pytest gate output. Benchmarks do not run after a failed gate.
- `perf/<case>/perf-stat.txt`: cycles, instructions, branches, branch misses,
  cache events, page faults, context switches, and migrations.
- `perf/<case>/perf.data`: sampled call stacks from `perf record`.
- `perf/<case>/perf-report.txt`: text rendering of the sampled profile.
- `extension/`: the exact native extension that ran plus ELF/symbol metadata.
- `comparators/`: exact legacy and pycocotools artifacts that ran, when
  available.
- `build/legacy-reference.json`: same-checkout legacy source hashes and rebuilt
  binary fingerprints.
- `environment/`: CPU, Python, NumPy, Rust, package, and profiler provenance.
- `git/`: source HEAD, status, and working-tree diff.
- `source/`: Rust kernel and profiling source snapshot.
- `commands.json`: commands, return codes, and wall durations.

`perf` is optional. A machine whose `perf_event_paranoid` policy prevents
hardware counters still produces timing/correctness/environment evidence and
records the profiler failure rather than silently substituting fake counters.

## Benchmark philosophy

Fixture generation is deterministic and occurs outside timed regions. For
mutating APIs such as Soft-NMS, restoring the input is also excluded from each
microbenchmark timing sample. The standalone perf workload necessarily profiles
input restoration for mutating cases and records that fact; the default perf
case list therefore favors non-mutating kernels.

Canonical benchmark captures use five deterministic fixture seeds. Competing
backends are timed in rotating order inside each sample round, and performance
claims use the median of paired Rust/comparator sample ratios rather than the
ratio of two independently collected medians. This reduces sensitivity to
frequency or load drift during a run. The reported ±5% bands remain descriptive
thresholds, not statistical-significance tests.

Mask comparisons include direct pycocotools timings where the APIs are shared:
encode, decode, area, `toBbox`, merge, and IoU. RLE-consuming APIs include both
structured/low-transition and fragmented/high-transition workloads so claims do
not depend on one favorable run-length regime.

Do not compare wall-clock rows from different machines or materially different
toolchains. The evidence bundle records enough provenance to detect those
mistakes. Hardware-counter deltas such as instructions/call can often be more
informative than small wall-clock changes.
