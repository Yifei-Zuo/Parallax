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
                qk_scale: float,
                n_rep: int,
                window_size_left: int) -> torch.Tensor:
        o, barv, d1, bart, m = parallax_fwd(q, r, k, v, qk_scale, n_rep, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.qk_scale = qk_scale
        ctx.n_rep = n_rep
        ctx.window_size_left = window_size_left
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        grad_q, grad_r, grad_k, grad_v = parallax_bwd(
            q, r, k, v, o, barv, d1, bart, m, grad_o,
            ctx.qk_scale, ctx.n_rep, ctx.window_size_left,
        )
        return grad_q, grad_r, grad_k, grad_v, None, None, None


def parallax_func(q: torch.Tensor,
                  r: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor,
                  qk_scale: float | None = None,
                  window_size_left: int = -1) -> torch.Tensor:
    """Causal Parallax with autograd, backed by Triton kernels.

    Shape convention follows :func:`torch.nn.functional.scaled_dot_product_attention`
    — heads as the second axis. Heads are folded into the batch dimension
    internally before the kernel call and unfolded on return. GQA is
    supported by passing ``k, v`` with fewer heads than ``q, r``
    (``H_q % H_kv == 0``); ``n_rep`` is derived as ``H_q // H_kv``.

    Args:
        q, r: ``(B, H_q, L_q, D)`` bf16 or fp16 tensors.
        k, v: ``(B, H_kv, L_kv, D)`` tensors with the same dtype as ``q``;
            ``H_q`` must be divisible by ``H_kv``.
        qk_scale: defaults to ``1 / sqrt(D)``.
        window_size_left: causal sliding-window length (FA2 convention).
            ``-1`` (default) disables; ``>= 0`` restricts row ``i`` to cols
            ``[i - window_size_left + 1, i]``.

    Returns:
        ``(B, H_q, L_q, D)`` tensor with the same dtype as ``q``.
    """
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"parallax_func requires bf16 or fp16 inputs, got q.dtype={q.dtype}"
        )
    B, H_q, L_q, D = q.shape
    _, H_kv, L_kv, _ = k.shape
    if H_q % H_kv != 0:
        raise ValueError(
            f"H_q ({H_q}) must be divisible by H_kv ({H_kv}) for GQA"
        )
    n_rep = H_q // H_kv
    if qk_scale is None:
        qk_scale = D ** -0.5
    o = ParallaxFunction.apply(
        q.reshape(B * H_q, L_q, D),
        r.reshape(B * H_q, L_q, D),
        k.reshape(B * H_kv, L_kv, D),
        v.reshape(B * H_kv, L_kv, D),
        float(qk_scale),
        n_rep,
        window_size_left,
    )
    return o.reshape(B, H_q, L_q, D)
