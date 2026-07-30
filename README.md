# Parallax: Parameterized Local Linear Attention

This repository provides the reference implementation of Parallax, accompanying an
anonymous submission currently under review. Author, affiliation, and publication
details are omitted for double-blind review.

Parallax is an upgrade to Softmax Attention. It is a scalable form of Local Linear
Attention (LLA), a mechanism with provable theoretical advantages over Softmax
Attention. Parallax and LLA are **not** linear complexity attention mechanisms.
They share the computational structure of Softmax Attention and require KV cache
for decoding. Optimizations such as sliding window and block-sparsity are
structurally compatible with Parallax.

## Install

Triton and Helion each provide training (dense + varlen) and decode kernels.</br>
CuTeDSL provides the decode kernel only.

For development or the bench and parity harnesses, install from source:

```bash
uv sync                # core; add --extra helion / --extra cutedsl as needed
uv sync --group bench  # bench harness: pinned stack + FA2 + pytest
```

The `[helion]` extra adds the [Helion](https://github.com/pytorch/helion) kernels
(experimental); the `[cutedsl]` extra adds the SM90 CuTeDSL decode kernels and
pins torch 2.9.1 / triton 3.5.1.

## Quickstart

> Note: the kernels are developed and tested on NVIDIA Hopper GPUs.
> A reference PyTorch implementation is provided in `parallax/reference.py` for correctness verification and as a starting point for custom implementations on other hardware.

### Training (Triton)

```python
import torch
from parallax import parallax_func

B, H, L, D = 2, 8, 1024, 128
q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
r = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

o = parallax_func(q, r, k, v) # (B, H, L, D), causal
o.float().pow(2).mean().backward()
```

### Decoding (CuTeDSL)

```python
import math
import torch
from parallax import parallax_decode

B, H, D = 4, 8, 128
kv_len = 4096
q = torch.randn(B, 1, H, D, device="cuda", dtype=torch.bfloat16)
r = torch.randn_like(q)
k = torch.randn(B, kv_len, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)

o = parallax_decode(q, r, k, v, qk_scale=1.0 / math.sqrt(D)) # (B, 1, H, D)
```

### Helion kernels (experimental)

[Helion](https://github.com/pytorch/helion) implementations of all three kernels
— autotuned, compiled to Triton — live under `parallax.helion` with the same
entry-point names and signatures:

```python
from parallax.helion import parallax_func, parallax_varlen_func, parallax_decode
```

On H100 (full autotune + CUDA-graph replay) vs the Triton kernels above:
training step (fwd+bwd) **0.84×** geomean latency across a 17-shape grid,
varlen **0.53×**, decode **1.8–6.4× faster**. Precision: q50 max-norm relative
error < 1e-2 vs the fp32 reference for the output and all four gradients
(`scripts/test_*_helion.py`).

Helion autotunes on first call per shape — minutes per new shape with
`HELION_AUTOTUNE_EFFORT=full` (cached via `HELION_CACHE_DIR`), immediate but
slower with `=none`. For production, pin tuned configs — see
[Helion's deployment docs](https://github.com/pytorch/helion/blob/main/docs/deployment_autotuning.md).

## Benchmark

`scripts/bench_decode.py` benchmarks the decode kernel against FA2 and
FA3 with combined speed + precision reporting:

```bash
python scripts/bench_decode.py                       # example sweep
python scripts/bench_decode.py --include-fa3         # add the FA3 column
python scripts/bench_decode.py --parallax-grid \
                               --csv runs/bench.csv  # 216-shape grid, save to CSV
```

The numbers below are measured on a single NVIDIA H200 SXM (132 SMs)
with bf16 inputs and head dimension `D = 128`. Latency is the q50
over a CUDA-graph replay sweep (`q05` and `q95` are within ±1% on
every row). Accuracy is the worst per-element relative error against
the fp32 torch reference (`parallax.parallax_reference`).

**Small batch (B = 1, H = 8, D = 128)**

| L | FA2 (µs) | FA3 (µs) | Parallax (µs) | Parallax max-rel-err |
|---:|---:|---:|---:|---:|
|   512 |  8.38 | 10.64 | **5.79** | 2.1e-3 |
|  1024 |  9.45 |  9.10 | **6.48** | 4.0e-3 |
|  4096 | 17.07 | 11.90 | **8.61** | 2.0e-3 |
| 16384 | 29.82 | 24.46 | **21.53** | 2.7e-3 |

**Large batch (B = 32, H = 8, D = 128)**

| L | FA2 (µs) | FA3 (µs) | Parallax (µs) | Parallax max-rel-err |
|---:|---:|---:|---:|---:|
|   512 |   27.73 |   **23.48** |    24.02 | 3.6e-3 |
|  1024 |   99.73 |   **39.16** |    39.55 | 3.4e-3 |
|  4096 |  384.90 |    281.64  | **279.96** | 3.6e-3 |
| 16384 | 1574.94 |   1096.76  | **1094.37** | 3.2e-3 |

Reproduce the small-batch table with:

```bash
python scripts/bench_decode.py --include-fa3 \
    --shape 1,512,8,128  --shape 1,1024,8,128 \
    --shape 1,4096,8,128 --shape 1,16384,8,128 \
    --warmup 100 --iters 50 --trials 20
```

Reproduce the large-batch table with:

```bash
python scripts/bench_decode.py --include-fa3 \
    --shape 32,512,8,128  --shape 32,1024,8,128 \
    --shape 32,4096,8,128 --shape 32,16384,8,128 \
    --warmup 100 --iters 50 --trials 20
```

## License

MIT. See `LICENSE`.
