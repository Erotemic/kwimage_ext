# Rust kernel profiling

`profile_rust_kernels.py` produces one self-contained evidence tarball for
performance review. It is designed to answer both "how fast is this Python API
call?" and "why is the native kernel spending those cycles?".

Typical full capture:

```bash
python dev/profiling/profile_rust_kernels.py \
    --output ~/Downloads/kwimage-ext-rust-profile.tar.gz
```

For the strongest native stack traces, rebuild the local extension as an
optimized symbolized build first:

```bash
python dev/profiling/profile_rust_kernels.py \
    --rebuild-profiled \
    --output ~/Downloads/kwimage-ext-rust-profile.tar.gz
```

`--rebuild-profiled` uses `maturin develop --release`, release optimization,
debug information, and frame pointers. The exact imported `_rust` shared object
is copied into the bundle and SHA-256 hashed, so measurements can be tied to the
binary that actually ran.

A short smoke capture is available as:

```bash
python dev/profiling/profile_rust_kernels.py --quick --no-perf
```

## What the bundle contains

- `summary.md`: first-pass timing and profiler status summary.
- `benchmarks/results.json`: deterministic microbenchmark samples and
  Rust/legacy (or Python reference) comparisons when a comparator is installed.
- `correctness/`: pytest gate output. Benchmarks do not run after a failed gate.
- `perf/<case>/perf-stat.txt`: cycles, instructions, branches, branch misses,
  cache events, page faults, context switches, and migrations.
- `perf/<case>/perf.data`: sampled call stacks from `perf record`.
- `perf/<case>/perf-report.txt`: text rendering of the sampled profile.
- `extension/`: the exact native extension that ran plus ELF/symbol metadata.
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

Do not compare wall-clock rows from different machines or materially different
toolchains. The evidence bundle records enough provenance to detect those
mistakes. Hardware-counter deltas such as instructions/call can often be more
informative than small wall-clock changes.
