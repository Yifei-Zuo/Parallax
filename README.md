# Parallax: Parameterized Local Linear Attention

[![arXiv](https://img.shields.io/badge/-arXiv-000000.svg?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2605.29157)
[![HF Papers](https://img.shields.io/badge/-Huggingface-000000.svg?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/papers/2605.29157)
[![X](https://img.shields.io/badge/-Post-000000.svg?logo=x&logoColor=white)](https://x.com/YifeiZuoX/status/2060499152791077082)
[![Blog](https://img.shields.io/badge/-Blog-000000.svg?logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSItMSAtMSAzNCAzNSIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMjIuNTk3NCAxNi42MjJMMjkuMTczMSAxMi4zNjYxQzI5Ljk0NTkgMTEuODY1NyAzMC45NzczIDEyLjA4NjQgMzEuNDc3OCAxMi44NTkyQzMxLjk3ODMgMTMuNjMyMSAzMS43NTc1IDE0LjY2MzQgMzAuOTg0NyAxNS4xNjM5TDIzLjA0MzQgMjAuMzA3NEw5LjUxODczIDE1LjU5OTRMMi41Mzk0MSAxOS44ODU2QzEuNzU0OTkgMjAuMzY3MiAwLjcyODgwOSAyMC4xMjIgMC4yNDY4NTcgMTkuMzM3OEMtMC4yMzQ5MjMgMTguNTUzMiAwLjAwOTk0MTI2IDE3LjUyNTYgMC43OTQ1ODIgMTcuMDQzOEw5LjEyMjczIDExLjkzMDdMMjIuNTk3NCAxNi42MjJaIiBmaWxsPSIjRjNGM0Y0Ii8%2BPHBhdGggZD0iTTIyLjU5NzQgNC42OTEzMkwyOS4xNzMxIDAuNDM1NDQ5QzI5Ljk0NTkgLTAuMDY1MDA5NSAzMC45NzczIDAuMTU1NzQxIDMxLjQ3NzggMC45Mjg1NTNDMzEuOTc4MyAxLjcwMTQxIDMxLjc1NzUgMi43MzI3OCAzMC45ODQ3IDMuMjMzMjVMMjMuMDQzNCA4LjM3NjdMOS41MTg3MyAzLjY2ODY5TDIuNTM5NDEgNy45NTQ5MUMxLjc1NDk5IDguNDM2NTUgMC43Mjg4MDkgOC4xOTEyOSAwLjI0Njg1NyA3LjQwNzE4Qy0wLjIzNDkyMyA2LjYyMjU0IDAuMDA5OTQxMjYgNS41OTQ4OSAwLjc5NDU4MiA1LjExMzExTDkuMTIyNzMgMEwyMi41OTc0IDQuNjkxMzJaIiBmaWxsPSIjRjNGM0Y0Ii8%2BPHBhdGggZD0iTTIyLjU5NzQgMjguNTUzNkwyOS4xNzMxIDI0LjI5NzhDMjkuOTQ1OSAyMy43OTczIDMwLjk3NzMgMjQuMDE4IDMxLjQ3NzggMjQuNzkwOUMzMS45NzgzIDI1LjU2MzcgMzEuNzU3NSAyNi41OTUxIDMwLjk4NDcgMjcuMDk1NUwyMy4wNDM0IDMyLjIzOUw5LjUxODczIDI3LjUzMUwyLjUzOTQxIDMxLjgxNzJDMS43NTQ5OSAzMi4yOTg5IDAuNzI4ODA5IDMyLjA1MzYgMC4yNDY4NTcgMzEuMjY5NUMtMC4yMzQ5MjMgMzAuNDg0OCAwLjAwOTk0MTI2IDI5LjQ1NzIgMC43OTQ1ODIgMjguOTc1NEw5LjEyMjczIDIzLjg2MjNMMjIuNTk3NCAyOC41NTM2WiIgZmlsbD0iI0YzRjNGNCIvPjwvc3ZnPgo%3D)](https://blog.tilderesearch.com/blog/parallax)
[![License](https://img.shields.io/badge/-license-000000.svg)](LICENSE)

This repository provides the official implementation of Parallax from the following paper:

> **Parallax: Parameterized Local Linear Attention for Language Modeling.**</br>
> Yifei Zuo, Dhruv Pai, Zhichen Zeng, Alec Dewulf, Shuming Hu, and Zhaoran Wang.
> arXiv preprint, 2026.

Parallax is an upgrade to Softmax Attention. It is a scalable form of Local Linear Attention (LLA), a mechanism with provable theoretical advantages over Softmax Attention (see [FlashLLA](https://github.com/Yifei-Zuo/FlashLLA) for the LLA kernels). Parallax and LLA are **not** linear complexity attention mechanisms. They share the computational structure of Softmax Attention and require KV cache for decoding. Optimizations such as sliding window and block-sparsity are structurally compatible with Parallax.

## Integrations

- [Flash-Linear-Attention](https://github.com/fla-org/flash-linear-attention): Parallax kernels available in the `fla` library.
- [Modded-NanoGPT-plx](https://github.com/Yifei-Zuo/modded-nanogpt-plx/tree/master/parallax): Parallax for the `Modded-NanoGPT` speedrun.

<p align="left">
  <img src="assets/hyperball_zoom.png" width="40%" />
  <img src="assets/pema_zoom.png" width="40%" />
</p>

## Install

```bash
git clone https://github.com/Yifei-Zuo/Parallax.git
cd Parallax

# Training only (Triton + reference)
uv sync
# Or with pip:
pip install -e .
```

Add the SM90 decode kernels:

```bash
uv sync --extra decode
# Or with pip:
pip install -e '.[decode]'
```

For the bench harness:

```bash
uv sync --extra bench
# Or with pip:
pip install -e '.[bench]'
```

Add the [Helion](https://github.com/pytorch/helion) kernels (experimental):

```bash
uv sync --extra helion
# Or with pip:
pip install -e '.[helion]'
```

## Quickstart

> Note: our current kernels are developed and tested on NVIDIA Hopper GPUs.
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

## Citation

```bibtex
@misc{zuo2026parallaxparameterizedlocallinear,
      title={Parallax: Parameterized Local Linear Attention for Language Modeling}, 
      author={Yifei Zuo and Dhruv Pai and Zhichen Zeng and Alec Dewulf and Shuming Hu and Zhaoran Wang},
      year={2026},
      eprint={2605.29157},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.29157}, 
}
```

## License

MIT. See [LICENSE](LICENSE).
