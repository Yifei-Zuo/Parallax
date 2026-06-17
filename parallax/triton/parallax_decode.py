# Copyright (c) 2026 Yifei Zuo.
# SPDX-License-Identifier: MIT
"""Single-token Parallax decode in Triton.

Pure-Triton autoregressive (``Sq == 1``) decode for Parallax, runnable on any
CUDA GPU. Layout matches :func:`parallax.parallax_reference`: ``q, r`` are
``(B, 1, H_q, D)`` and ``k, v`` are ``(B, L, H_kv, D)`` with ``H_q % H_kv == 0``
(GQA). See :func:`parallax.parallax_reference` for the math.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from parallax.triton._kernel_utils import exp2


@triton.heuristics({
    'USE_CACHE_START': lambda args: args['cache_start'] is not None,
})
@triton.jit(do_not_specialize=['Skv'])
def parallel_parallax_onestep_kernel(
    q,
    r,
    k,
    v,
    o,
    scale,
    cache_start,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BD: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_CACHE_START: tl.constexpr,
    BS: tl.constexpr,
):
    """One query per (batch, head), reduced against the cache via online softmax."""
    i_bh = tl.program_id(0)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G                               # kv head shared by this q head (GQA)
    RCP_LN2: tl.constexpr = 1.4426950216

    # Lowest key the query attends to: left-padding start, raised to the window edge.
    kv_lo = 0
    if USE_CACHE_START:
        kv_lo = tl.load(cache_start + i_b).to(tl.int32)
    if WINDOW_SIZE_LEFT >= 0:
        kv_lo = tl.maximum(kv_lo, Skv - WINDOW_SIZE_LEFT)
    kv_lo = tl.maximum(kv_lo, 0)

    p_q = tl.make_block_ptr(q + i_bh * K, (K,), (1,), (0,), (BD,), (0,))
    p_r = tl.make_block_ptr(r + i_bh * K, (K,), (1,), (0,), (BD,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * K, (K,), (1,), (0,), (BD,), (0,))
    b_q = tl.load(p_q, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_r = tl.load(p_r, boundary_check=(0,), padding_option="zero").to(tl.float32)
    scale_log2 = scale * RCP_LN2

    # Online-softmax state: pivot m, denominators d1/d2 = sum(p1)/sum(p2),
    # and unnormalized outputs o1/o2 = p1@v / p2@v.
    m = tl.full((1,), -float("inf"), dtype=tl.float32)
    d1 = tl.zeros((1,), dtype=tl.float32)
    d2 = tl.zeros((1,), dtype=tl.float32)
    o1 = tl.zeros((BD,), dtype=tl.float32)
    o2 = tl.zeros((BD,), dtype=tl.float32)

    start_block = kv_lo // BS
    p_k = tl.make_block_ptr(k + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (start_block * BS, 0), (BS, BD), (1, 0))
    p_v = tl.make_block_ptr(v + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (start_block * BS, 0), (BS, BD), (1, 0))
    for i_s in range(start_block * BS, tl.cdiv(Skv, BS) * BS, BS):
        col = i_s + tl.arange(0, BS)
        mask = (col >= kv_lo) & (col < Skv)                          # [BS] valid keys
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        s1 = tl.sum(b_q[None, :] * b_k, axis=1) * scale_log2          # scale * (q . k), base-2
        s2 = tl.sum(b_r[None, :] * b_k, axis=1)                        # r . k (unscaled)
        s1 = tl.where(mask, s1, -float("inf"))
        m_new = tl.maximum(m, tl.max(s1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)         # finite pivot (empty cache -> 0)
        alpha = exp2(m - m_safe)
        p1 = exp2(s1 - m_safe)
        p2 = p1 * s2
        d1 = d1 * alpha + tl.sum(p1)
        d2 = d2 * alpha + tl.sum(p2)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)                        # 0 when no valid key (avoid NaN)
    out = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1             # O1/d1*(1 + d2/d1) - O2/d1
    tl.store(p_o, out.to(p_o.dtype.element_ty), boundary_check=(0,))


def parallax_decode(q: torch.Tensor,
                    r: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    qk_scale: float | None = None,
                    *,
                    window_size_left: int = -1,
                    cache_start: torch.LongTensor | None = None) -> torch.Tensor:
    """Single-token Parallax decode (``Sq == 1``), Triton. Forward-only.

    Args:
        q, r: ``(B, 1, H_q, D)`` query / secondary query (bf16/fp16); ``r`` is
            not pre-scaled.
        k, v: ``(B, L, H_kv, D)`` cache; ``H_q % H_kv == 0`` (GQA).
        qk_scale: scale applied to ``q @ k^T`` only. Defaults to ``1 / sqrt(D)``.
        window_size_left: causal sliding-window length (FA2 convention). ``-1``
            (default) disables; ``>= 0`` restricts the query to the most recent
            ``window_size_left`` keys.
        cache_start: per-batch ``(B,)`` first valid key index (left-padding mask);
            ``None`` means the whole cache is valid.

    Returns:
        ``(B, 1, H_q, D)`` output with the same dtype as ``q``.
    """
    B, Sq, HQ, K = q.shape
    if Sq != 1:
        raise ValueError(f"parallax_decode requires a single query (Sq=1), got Sq={Sq}.")
    Skv, H = k.shape[1], k.shape[2]
    G = HQ // H
    if qk_scale is None:
        qk_scale = K ** -0.5

    q, r, k, v = (x.contiguous() for x in (q, r, k, v))
    if cache_start is not None:
        cache_start = cache_start.to(device=q.device, dtype=torch.int32).contiguous()
    BD = triton.next_power_of_2(K)
    o = torch.empty_like(q)
    grid = (B * HQ,)
    parallel_parallax_onestep_kernel[grid](
        q, r, k, v, o, float(qk_scale), cache_start, Skv,
        HQ=HQ, H=H, G=G, K=K, BD=BD,
        WINDOW_SIZE_LEFT=window_size_left,
        USE_CACHE_START=cache_start is not None,
        BS=128,
        num_warps=4, num_stages=2,
    )
    return o
