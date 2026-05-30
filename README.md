# Parallax: Parameterized Local Linear Attention

[![arXiv](https://img.shields.io/badge/arXiv-2605.29157-b31b1b.svg)](https://arxiv.org/abs/2605.29157)
[![HF Papers](https://img.shields.io/badge/HuggingFace-Papers-FFD21E.svg)](https://huggingface.co/papers/2605.29157)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository provides the official implementation of Parallax from the following paper:

**Parallax: Parameterized Local Linear Attention.**
Yifei Zuo, Dhruv Pai, Zhichen Zeng, Alec Dewulf, Shuming Hu, and Zhaoran Wang.

## Install

```bash
git clone https://github.com/Yifei-Zuo/Parallax.git
cd Parallax

uv sync

# Or with pip:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

For the bench harness:

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

`scripts/bench_decode.py` benchmarks the decode kernel against FA2 (and
optionally FA3) with combined speed + precision reporting:

```bash
python scripts/bench_decode.py                       # example sweep
python scripts/bench_decode.py --include-fa3         # add the FA3 column
python scripts/bench_decode.py --parallax-grid \
                               --csv runs/bench.csv  # 216-shape grid → CSV
```

For each shape it prints CUDA-graph replay latency (q05/q50/q95) per
backend and a per-row precision check.

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