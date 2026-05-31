# Parallax: Parameterized Local Linear Attention

[![arXiv](https://img.shields.io/badge/arXiv-2605.29157-b31b1b.svg)](https://arxiv.org/abs/2605.29157)
[![HF Papers](https://img.shields.io/badge/HuggingFace-Papers-FFD21E.svg)](https://huggingface.co/papers/2605.29157)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository provides the official implementation of Parallax from the following paper:

> **Parallax: Parameterized Local Linear Attention.**</br>
> Yifei Zuo, Dhruv Pai, Zhichen Zeng, Alec Dewulf, Shuming Hu, and Zhaoran Wang.
> arXiv preprint, 2026.

Parallax is an upgrade to Softmax Attention. It is a scalable form of Local Linear Attention (LLA), a mechanism with provable theoretical advantages over Softmax Attention (see [FlashLLA](https://github.com/Yifei-Zuo/FlashLLA) for the LLA kernels). Parallax and LLA are **not** linear complexity attention mechanisms. They share the computational structure of Softmax Attention and require KV cache for decoding. Optimizations such as sliding window and block-sparsity are structurally compatible with Parallax.

## Install

```bash
git clone https://github.com/Yifei-Zuo/Parallax.git
cd Parallax

uv sync

# Or with pip:
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

For the bench harness (adds the FA2 baseline + `rich` Live table):

```bash
uv sync --extra bench
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