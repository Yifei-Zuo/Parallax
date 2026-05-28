"""Shared shape grids + input factory for the verify and bench scripts.

Mirrors ``flashlla/dev_ops/decode_h200_v4/_test_utils.py`` from the upstream
dev tree so both scripts use a single source of truth for ``(B, K, H, D)``
cases.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


REFERENCE_SHAPES = [
    # (B, K, H, D)
    (1,   128,  1,  64),
    (1,   128,  1, 128),
    (4,  1024,  8,  64),
    (4,  1024,  8, 128),
    (4,  4096, 16, 128),
    (1,  8192,  8, 128),
    (1,  4096,  1,  64),
    (1,  8192,  1,  64),
    (1, 16384,  1,  64),
    (1,  8192,  8, 128),
    (1, 16384,  8, 128),
    (4, 16384, 16, 128),
]


def extended_grid():
    """288-shape sweep (kept for cross-comparison with the upstream v3 run)."""
    shapes = []
    for B in (1, 2, 4, 8):
        for H in (1, 4, 8, 16):
            for K in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
                for D in (64, 128):
                    shapes.append((B, K, H, D))
    return shapes


def parallax_grid():
    """Dedup ``BH x K x D`` grid for the Parallax bench (BH-invariant axis).

    Axes:
        BH: 1..2048 (12 values), K: 128..32768 (9 values), D: 64, 128.

    Skips shapes whose bf16 K/V tensor would exceed 2 GiB (peak during the
    fp32 RMS-norm intermediates inside :func:`make_inputs`).
    """
    BHs = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    Ks  = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    Ds  = (64, 128)
    CAP_BYTES = 2 * (1 << 30)
    shapes = []
    for BH in BHs:
        B, H = 1, BH
        for K in Ks:
            for D in Ds:
                if BH * K * D * 2 > CAP_BYTES:
                    continue
                shapes.append((B, K, H, D))
    return shapes


def make_inputs(B, K, H, D, *, dtype=torch.bfloat16, device="cuda", seed=None):
    """Canonical Parallax decode inputs.

    Returns ``(q, r, k, v, qk_scale)`` where ``q, r`` are ``(B, 1, H, D)`` and
    ``k, v`` are ``(B, K, H, D)``. ``q, r, k`` are RMS-normed (matching what the
    kernel would see mid-training); ``v`` is left raw.
    """
    if seed is not None:
        torch.manual_seed(seed)
    q = torch.randn(B, 1, H, D, device=device, dtype=dtype)
    r = torch.randn_like(q)
    k = torch.randn(B, K, H, D, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q = F.rms_norm(q.float(), (D,)).to(dtype).contiguous()
    r = F.rms_norm(r.float(), (D,)).to(dtype).contiguous()
    k = F.rms_norm(k.float(), (D,)).to(dtype).contiguous()
    v = v.contiguous()
    qk_scale = 1.0 / math.sqrt(D)
    return q, r, k, v, qk_scale
