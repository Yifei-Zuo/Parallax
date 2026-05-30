# Copyright (c) 2026 Yifei Zuo.
# SPDX-License-Identifier: MIT
"""Autograd wrapper for the Parallax Triton training kernels.

Exposes :func:`parallax_func` (functional API, the canonical entry point)
and :class:`ParallaxFunction` (the underlying ``torch.autograd.Function``).
Both back onto :func:`parallax.triton.parallax_fwd` and
:func:`parallax.triton.parallax_bwd`.
"""

from __future__ import annotations

import torch

from parallax.triton.parallax_bwd import parallax_bwd
from parallax.triton.parallax_fwd import parallax_fwd


class ParallaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q: torch.Tensor,
                r: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                qk_scale: float) -> torch.Tensor:
        o, barv, d1, bart, m = parallax_fwd(q, r, k, v, qk_scale)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.qk_scale = qk_scale
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        grad_q, grad_r, grad_k, grad_v = parallax_bwd(
            q, r, k, v, o, barv, d1, bart, m, grad_o, ctx.qk_scale,
        )
        return grad_q, grad_r, grad_k, grad_v, None


def parallax_func(q: torch.Tensor,
                  r: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor,
                  qk_scale: float | None = None) -> torch.Tensor:
    """Causal Parallax attention with autograd, backed by Triton kernels.

    Args:
        q, r: ``(B, H, L_q, D)`` bf16 or fp16 tensors.
        k, v: ``(B, H, L_kv, D)`` tensors with the same dtype as ``q``.
        qk_scale: defaults to ``1 / sqrt(D)``.

    Returns:
        ``(B, H, L_q, D)`` tensor with the same dtype as ``q``.
    """
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"parallax_func requires bf16 or fp16 inputs, got q.dtype={q.dtype}"
        )
    B, H, L_q, D = q.shape
    L_kv = k.shape[2]
    if qk_scale is None:
        qk_scale = D ** -0.5
    o = ParallaxFunction.apply(
        q.reshape(B * H, L_q, D),
        r.reshape(B * H, L_q, D),
        k.reshape(B * H, L_kv, D),
        v.reshape(B * H, L_kv, D),
        float(qk_scale),
    )
    return o.reshape(B, H, L_q, D)
