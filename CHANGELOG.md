# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-07-07

### Changed

- Renamed `parallax.decode_available` to `parallax.cute_decode_available`:
  the flag only reports the optional CuTeDSL SM90 kernel — the Triton (and,
  with the `[helion]` extra, Helion) decode kernels are always importable.
  `decode_available` remains as a deprecated alias and emits a
  `DeprecationWarning`.

### Fixed

- `import parallax` no longer crashes when the cutedsl stack is installed
  but fails to load (missing NVIDIA driver, non-SM90 GPU, torch/cutlass ABI
  mismatch, ...). The optional import now tolerates any exception — not just
  `ImportError` — and the CuTeDSL entry points raise a helpful `ImportError`
  on call instead.

## [0.1.0] - 2026-07-07

First PyPI release, published as `parallax-kernel` (imports as `parallax`).

### Added

- **Triton kernels** (any CUDA GPU): causal training with autograd
  (`parallax_func`, raw `parallax_fwd`/`parallax_bwd` with exposed stats),
  variable-length packed training (`parallax_varlen_func`), and single-token
  decode (`parallax.triton.parallax_decode`).
- **Helion kernels** (experimental, `[helion]` extra): training, varlen, and
  decode under `parallax.helion` with the same entry-point names — autotuned,
  compiled to Triton. On H100 vs the Triton kernels: 0.84x geomean
  training-step latency across a 17-shape grid, 0.53x varlen, 1.8-6.4x faster
  decode.
- **CuTeDSL SM90 decode kernel** (`[cutedsl]` extra, Hopper only):
  `parallax_attn_with_kvcache` against a KV cache, `GraphedDecode` for
  CUDA-graph capture, and the deprecated `parallax_decode` alias. Beats FA2
  and matches or beats FA3 on H200 across the benchmarked grid.
- **fp32 PyTorch reference** (`parallax_reference`) for correctness checks on
  any hardware.
- `parallax.__version__` and `parallax.decode_available`.
- Bench harness (`scripts/bench_decode.py`) with FA2/FA3 speed + precision
  comparison, and a parity test suite against the fp32 reference.
