# Copyright (c) 2026 Yifei Zuo.
# SPDX-License-Identifier: MIT
"""Parallax dense causal training pass (forward + backward) in Helion.

Drop-in for :func:`parallax.parallax_func`, with autograd. The backward mirrors
:func:`parallax.triton.parallax_bwd`: a preprocess kernel (per-row
``t = Σ grad_o·o``, ``b = Σ grad_o·barv``), a grad_q/grad_r kernel (parallel over
query rows) and a grad_k/grad_v kernel (parallel over KV; no atomics — two passes
with swapped parallel axes). GQA folds per-q-head dK/dV back to the kv-head axis.
The forward is routed by head dim (see ``parallax_fwd``).

Layout: q, r are ``(B, H_q, L, D)``; k, v are ``(B, H_kv, L, D)`` with
``H_q % H_kv == 0``. Causal-with-prefix: query row ``i`` attends keys
``j <= L_kv - L_q + i``.
"""
from __future__ import annotations

import torch
import helion
import helion.language as hl

_LOG2E = 1.4426950216  # 1 / ln(2); scores accumulated in base-2 units.


# --------------------------------------------------------------------------- #
# Forward: masked single KV loop (used for head_dim >= 128).
# --------------------------------------------------------------------------- #
@helion.kernel(static_shapes=True)
def _fwd_kernel(
    q: torch.Tensor,        # (B, H_q, L_q, D)
    r: torch.Tensor,        # (B, H_q, L_q, D)
    k: torch.Tensor,        # (B, H_kv, L_kv, D)
    v: torch.Tensor,        # (B, H_kv, L_kv, D)
    scale_log2: float,
    WINDOW_LEFT: int,
):
    B, Hq, Lq, D = q.shape
    Hk = k.size(1)
    Lk = k.size(2)
    D = hl.specialize(D)
    n_rep = Hq // Hk
    o = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1o = torch.empty([B, Hq, Lq, 1], device=q.device, dtype=torch.float32)
    barto = torch.empty([B, Hq, Lq, 1], device=q.device, dtype=torch.float32)
    mo = torch.empty([B, Hq, Lq, 1], device=q.device, dtype=torch.float32)

    block_m = hl.register_block_size(Lq)
    block_n = hl.register_block_size(Lk)
    for tile_b, tile_h, tile_m in hl.tile([B, Hq, Lq], block_size=[1, 1, block_m]):
        b = tile_b.begin
        hq = tile_h.begin
        hk = hq // n_rep
        q_i = q[b, hq, tile_m, :]
        r_i = r[b, hq, tile_m, :]
        ti = tile_m.index[:, None]
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        d1_i = hl.zeros([tile_m], dtype=torch.float32)
        d2_i = hl.zeros([tile_m], dtype=torch.float32)
        o1 = hl.zeros([tile_m, D], dtype=torch.float32)
        o2 = hl.zeros([tile_m, D], dtype=torch.float32)

        # Single masked KV loop, bounded by the tile's causal range (skips fully
        # future blocks). At D >= 128 this beats the 2-phase split below: the
        # matmuls dominate, so skipping the cheap per-block mask is not worth the
        # two-loop overhead.
        last_row = torch.amax(tile_m.index, 0)
        hi_key = last_row + (Lk - Lq + 1)
        hi_key = torch.minimum(hi_key, hi_key.new_full([], Lk))
        for tile_n in hl.tile(0, hi_key, block_size=block_n):
            kt = k[b, hk, tile_n, :]
            vt = v[b, hk, tile_n, :]
            tj = tile_n.index[None, :]
            s1 = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            s2 = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            causal = tj <= (Lk - Lq + ti)
            if WINDOW_LEFT >= 0:
                mask = causal & (tj >= (Lk - Lq + ti) - WINDOW_LEFT + 1)
            else:
                mask = causal
            s1 = torch.where(mask, s1, float("-inf"))
            m_new = torch.maximum(m_i, torch.amax(s1, -1))
            m_safe = torch.where(m_new == float("-inf"), 0.0, m_new)
            alpha = torch.exp2(m_i - m_safe)
            p1 = torch.exp2(s1 - m_safe[:, None])
            p2 = p1 * s2
            d1_i = d1_i * alpha + torch.sum(p1, -1)
            d2_i = d2_i * alpha + torch.sum(p2, -1)
            o1 = o1 * alpha[:, None] + hl.dot(p1.to(v.dtype), vt, out_dtype=torch.float32)
            o2 = o2 * alpha[:, None] + hl.dot(p2.to(v.dtype), vt, out_dtype=torch.float32)
            m_i = m_new

        inv = torch.where(d1_i > 0.0, 1.0 / d1_i, 0.0)
        barv_t = o1 * inv[:, None]
        bart_t = d2_i * inv
        o_t = barv_t + bart_t[:, None] * barv_t - o2 * inv[:, None]
        o[tile_b, tile_h, tile_m, :] = o_t[None, None, :, :].to(o.dtype)
        barv[tile_b, tile_h, tile_m, :] = barv_t[None, None, :, :].to(barv.dtype)
        d1o[tile_b, tile_h, tile_m, :] = d1_i[None, None, :, None]
        barto[tile_b, tile_h, tile_m, :] = bart_t[None, None, :, None]
        mo[tile_b, tile_h, tile_m, :] = m_i[None, None, :, None]
    return o, barv, d1o, barto, mo


# 2-phase causal-split forward: safe interior (no mask) + masked diagonal border.
# Wins at small head_dim (D <= 64), where matmuls are small and the per-block
# mask ALU is a large fraction; the wrapper routes D <= 64 here. block_m is
# floored at 64 (block_m=32 autotuned pathologically).
@helion.kernel(static_shapes=True)
def _fwd_kernel_split(q, r, k, v, scale_log2: float, WINDOW_LEFT: int):
    B, Hq, Lq, D = q.shape
    Hk = k.size(1)
    Lk = k.size(2)
    D = hl.specialize(D)
    n_rep = Hq // Hk
    o = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1o = torch.empty([B, Hq, Lq, 1], device=q.device, dtype=torch.float32)
    barto = torch.empty([B, Hq, Lq, 1], device=q.device, dtype=torch.float32)
    mo = torch.empty([B, Hq, Lq, 1], device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(64, Lq)
    block_n = hl.register_block_size(Lk)
    for tile_b, tile_h, tile_m in hl.tile([B, Hq, Lq], block_size=[1, 1, block_m]):
        b = tile_b.begin
        hq = tile_h.begin
        hk = hq // n_rep
        q_i = q[b, hq, tile_m, :]
        r_i = r[b, hq, tile_m, :]
        ti = tile_m.index[:, None]
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        d1_i = hl.zeros([tile_m], dtype=torch.float32)
        d2_i = hl.zeros([tile_m], dtype=torch.float32)
        o1 = hl.zeros([tile_m, D], dtype=torch.float32)
        o2 = hl.zeros([tile_m, D], dtype=torch.float32)
        first_row = torch.amin(tile_m.index, 0)
        last_row = torch.amax(tile_m.index, 0)
        hi_key = last_row + (Lk - Lq + 1)
        hi_key = torch.minimum(hi_key, hi_key.new_full([], Lk))
        if WINDOW_LEFT >= 0:
            safe_hi = first_row.new_full([], 0)
        else:
            safe_hi = first_row + (Lk - Lq + 1)
            safe_hi = torch.maximum(safe_hi, safe_hi.new_full([], 0))
            safe_hi = torch.minimum(safe_hi, hi_key)
        # Safe phase (no mask).
        for tile_n in hl.tile(0, safe_hi, block_size=block_n):
            kt = k[b, hk, tile_n, :]
            vt = v[b, hk, tile_n, :]
            s1 = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            s2 = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            m_new = torch.maximum(m_i, torch.amax(s1, -1))
            alpha = torch.exp2(m_i - m_new)
            p1 = torch.exp2(s1 - m_new[:, None])
            p2 = p1 * s2
            d1_i = d1_i * alpha + torch.sum(p1, -1)
            d2_i = d2_i * alpha + torch.sum(p2, -1)
            o1 = o1 * alpha[:, None] + hl.dot(p1.to(v.dtype), vt, out_dtype=torch.float32)
            o2 = o2 * alpha[:, None] + hl.dot(p2.to(v.dtype), vt, out_dtype=torch.float32)
            m_i = m_new
        # Border phase (masked).
        for tile_n in hl.tile(safe_hi, hi_key, block_size=block_n):
            kt = k[b, hk, tile_n, :]
            vt = v[b, hk, tile_n, :]
            tj = tile_n.index[None, :]
            s1 = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            s2 = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            causal = tj <= (Lk - Lq + ti)
            if WINDOW_LEFT >= 0:
                mask = causal & (tj >= (Lk - Lq + ti) - WINDOW_LEFT + 1)
            else:
                mask = causal
            s1 = torch.where(mask, s1, float("-inf"))
            m_new = torch.maximum(m_i, torch.amax(s1, -1))
            m_safe = torch.where(m_new == float("-inf"), 0.0, m_new)
            alpha = torch.exp2(m_i - m_safe)
            p1 = torch.exp2(s1 - m_safe[:, None])
            p2 = p1 * s2
            d1_i = d1_i * alpha + torch.sum(p1, -1)
            d2_i = d2_i * alpha + torch.sum(p2, -1)
            o1 = o1 * alpha[:, None] + hl.dot(p1.to(v.dtype), vt, out_dtype=torch.float32)
            o2 = o2 * alpha[:, None] + hl.dot(p2.to(v.dtype), vt, out_dtype=torch.float32)
            m_i = m_new
        inv = torch.where(d1_i > 0.0, 1.0 / d1_i, 0.0)
        barv_t = o1 * inv[:, None]
        bart_t = d2_i * inv
        o_t = barv_t + bart_t[:, None] * barv_t - o2 * inv[:, None]
        o[tile_b, tile_h, tile_m, :] = o_t[None, None, :, :].to(o.dtype)
        barv[tile_b, tile_h, tile_m, :] = barv_t[None, None, :, :].to(barv.dtype)
        d1o[tile_b, tile_h, tile_m, :] = d1_i[None, None, :, None]
        barto[tile_b, tile_h, tile_m, :] = bart_t[None, None, :, None]
        mo[tile_b, tile_h, tile_m, :] = m_i[None, None, :, None]
    return o, barv, d1o, barto, mo


# --------------------------------------------------------------------------- #
# Backward: preprocess + grad_q/grad_r + grad_k/grad_v (mirrors Triton math).
# --------------------------------------------------------------------------- #
@helion.kernel(static_shapes=True,
               config=helion.Config(block_sizes=[128], num_warps=4, num_stages=2))
def _bwd_preprocess(grad_o, o, barv):
    B, Hq, Lq, D = grad_o.shape
    D = hl.specialize(D)
    t_out = torch.empty([B, Hq, Lq, 1], device=o.device, dtype=torch.float32)
    b_out = torch.empty([B, Hq, Lq, 1], device=o.device, dtype=torch.float32)
    for tile_b, tile_h, tile_m in hl.tile([B, Hq, Lq], block_size=[1, 1, hl.register_block_size(Lq)]):
        b = tile_b.begin
        hq = tile_h.begin
        go = grad_o[b, hq, tile_m, :].to(torch.float32)
        oo = o[b, hq, tile_m, :].to(torch.float32)
        bv = barv[b, hq, tile_m, :].to(torch.float32)
        t_out[b, hq, tile_m, :] = torch.sum(go * oo, -1)[:, None]
        b_out[b, hq, tile_m, :] = torch.sum(go * bv, -1)[:, None]
    return t_out, b_out


@helion.kernel(static_shapes=True)
def _bwd_rq(q, r, k, v, d1o, barto, mo, to, bo, grad_o, qk_scale: float, WINDOW_LEFT: int):
    B, Hq, Lq, D = q.shape
    Hk = k.size(1)
    Lk = k.size(2)
    D = hl.specialize(D)
    n_rep = Hq // Hk
    scale_log2 = qk_scale * _LOG2E
    grad_q = torch.empty([B, Hq, Lq, D], device=q.device, dtype=torch.bfloat16)
    grad_r = torch.empty([B, Hq, Lq, D], device=q.device, dtype=torch.bfloat16)
    block_m = hl.register_block_size(Lq)
    block_n = hl.register_block_size(Lk)
    for tile_b, tile_h, tile_m in hl.tile([B, Hq, Lq], block_size=[1, 1, block_m]):
        b = tile_b.begin
        hq = tile_h.begin
        hk = hq // n_rep
        q_i = q[b, hq, tile_m, :]
        r_i = r[b, hq, tile_m, :]
        go_i = grad_o[b, hq, tile_m, :]
        ti = tile_m.index[:, None]
        d1_i = d1o[b, hq, tile_m, :]
        bart_i = barto[b, hq, tile_m, :]
        m_i = mo[b, hq, tile_m, :]
        t_i = to[b, hq, tile_m, :]
        b_i = bo[b, hq, tile_m, :]
        inv_d1 = torch.where(d1_i > 0.0, 1.0 / d1_i, 0.0)
        acc_q = hl.zeros([tile_m, D], dtype=torch.float32)
        acc_r = hl.zeros([tile_m, D], dtype=torch.float32)
        first_row = torch.amin(tile_m.index, 0)
        last_row = torch.amax(tile_m.index, 0)
        hi_key = last_row + (Lk - Lq + 1)
        hi_key = torch.minimum(hi_key, hi_key.new_full([], Lk))
        # 2-phase causal split: safe interior (no mask) + diagonal border (mask).
        # SWA breaks interior safety, so disable the safe phase there.
        if WINDOW_LEFT >= 0:
            safe_hi = first_row.new_full([], 0)
        else:
            safe_hi = first_row + (Lk - Lq + 1)
            safe_hi = torch.maximum(safe_hi, safe_hi.new_full([], 0))
            safe_hi = torch.minimum(safe_hi, hi_key)

        for tile_n in hl.tile(0, safe_hi, block_size=block_n):
            kt = k[b, hk, tile_n, :]
            vt = v[b, hk, tile_n, :]
            qk = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            rk = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            w = torch.exp2(qk - m_i)
            a = hl.dot(go_i, vt.T, out_dtype=torch.float32)
            p = w * inv_d1
            bart_minus_rk = bart_i - rk
            delta = a - b_i
            gl = p * (a - t_i + bart_minus_rk * delta)
            gu = -p * delta
            acc_q = hl.dot(gl.to(k.dtype), kt, acc=acc_q)
            acc_r = hl.dot(gu.to(k.dtype), kt, acc=acc_r)

        for tile_n in hl.tile(safe_hi, hi_key, block_size=block_n):
            kt = k[b, hk, tile_n, :]
            vt = v[b, hk, tile_n, :]
            tj = tile_n.index[None, :]
            qk = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            rk = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            causal = tj <= (Lk - Lq + ti)
            if WINDOW_LEFT >= 0:
                mask = causal & (tj >= (Lk - Lq + ti) - WINDOW_LEFT + 1)
            else:
                mask = causal
            w = torch.where(mask, torch.exp2(qk - m_i), 0.0)
            a = hl.dot(go_i, vt.T, out_dtype=torch.float32)
            p = w * inv_d1
            bart_minus_rk = bart_i - rk
            delta = a - b_i
            gl = p * (a - t_i + bart_minus_rk * delta)
            gu = -p * delta
            acc_q = hl.dot(gl.to(k.dtype), kt, acc=acc_q)
            acc_r = hl.dot(gu.to(k.dtype), kt, acc=acc_r)
        grad_q[b, hq, tile_m, :] = (acc_q * qk_scale).to(torch.bfloat16)
        grad_r[b, hq, tile_m, :] = acc_r.to(torch.bfloat16)
    return grad_q, grad_r


@helion.kernel(static_shapes=True)
def _bwd_kv(q, r, k, v, d1o, barto, mo, to, bo, grad_o, qk_scale: float, WINDOW_LEFT: int):
    B, Hq, Lq, D = q.shape
    Hk = k.size(1)
    Lk = k.size(2)
    D = hl.specialize(D)
    n_rep = Hq // Hk
    scale_log2 = qk_scale * _LOG2E
    gk_buf = torch.empty([B, Hq, Lk, D], device=q.device, dtype=torch.float32)
    gv_buf = torch.empty([B, Hq, Lk, D], device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(Lq)
    block_n = hl.register_block_size(Lk)
    for tile_b, tile_h, tile_n in hl.tile([B, Hq, Lk], block_size=[1, 1, block_n]):
        b = tile_b.begin
        hq = tile_h.begin
        hk = hq // n_rep
        kt = k[b, hk, tile_n, :]
        vt = v[b, hk, tile_n, :]
        tj = tile_n.index[None, :]
        acc_k = hl.zeros([tile_n, D], dtype=torch.float32)
        acc_v = hl.zeros([tile_n, D], dtype=torch.float32)
        first_col = torch.amin(tile_n.index, 0)
        last_col = torch.amax(tile_n.index, 0)
        lo_row = first_col - (Lk - Lq)
        lo_row = torch.maximum(lo_row, lo_row.new_full([], 0))
        # 2-phase split over query blocks: diagonal border (mask) then safe
        # interior (no mask). A query row i is safe for this KV block when
        # i >= last_col - (Lk-Lq). SWA breaks interior safety -> mask all.
        if WINDOW_LEFT >= 0:
            safe_row = lo_row.new_full([], Lq)
        else:
            safe_row = last_col - (Lk - Lq)
            safe_row = torch.maximum(safe_row, lo_row)
            safe_row = torch.minimum(safe_row, safe_row.new_full([], Lq))

        for tile_m in hl.tile(lo_row, safe_row, block_size=block_m):
            q_i = q[b, hq, tile_m, :]
            r_i = r[b, hq, tile_m, :]
            go_i = grad_o[b, hq, tile_m, :]
            ti = tile_m.index[:, None]
            d1_i = d1o[b, hq, tile_m, :]
            bart_i = barto[b, hq, tile_m, :]
            m_i = mo[b, hq, tile_m, :]
            t_i = to[b, hq, tile_m, :]
            b_i = bo[b, hq, tile_m, :]
            inv_d1 = torch.where(d1_i > 0.0, 1.0 / d1_i, 0.0)
            qk = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            rk = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            causal = tj <= (Lk - Lq + ti)
            if WINDOW_LEFT >= 0:
                mask = causal & (tj >= (Lk - Lq + ti) - WINDOW_LEFT + 1)
            else:
                mask = causal
            w = torch.where(mask, torch.exp2(qk - m_i), 0.0)
            p = w * inv_d1
            a = hl.dot(go_i, vt.T, out_dtype=torch.float32)
            delta = a - b_i
            bart_minus_rk = bart_i - rk
            gl = p * (a - t_i + bart_minus_rk * delta) * qk_scale
            gu = -p * delta
            weights = p * (1.0 + bart_minus_rk)
            acc_k = hl.dot(gl.T.to(k.dtype), q_i, acc=acc_k)
            acc_k = hl.dot(gu.T.to(k.dtype), r_i, acc=acc_k)
            acc_v = hl.dot(weights.T.to(k.dtype), go_i, acc=acc_v)

        for tile_m in hl.tile(safe_row, Lq, block_size=block_m):
            q_i = q[b, hq, tile_m, :]
            r_i = r[b, hq, tile_m, :]
            go_i = grad_o[b, hq, tile_m, :]
            d1_i = d1o[b, hq, tile_m, :]
            bart_i = barto[b, hq, tile_m, :]
            m_i = mo[b, hq, tile_m, :]
            t_i = to[b, hq, tile_m, :]
            b_i = bo[b, hq, tile_m, :]
            inv_d1 = torch.where(d1_i > 0.0, 1.0 / d1_i, 0.0)
            qk = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            rk = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            w = torch.exp2(qk - m_i)
            p = w * inv_d1
            a = hl.dot(go_i, vt.T, out_dtype=torch.float32)
            delta = a - b_i
            bart_minus_rk = bart_i - rk
            gl = p * (a - t_i + bart_minus_rk * delta) * qk_scale
            gu = -p * delta
            weights = p * (1.0 + bart_minus_rk)
            acc_k = hl.dot(gl.T.to(k.dtype), q_i, acc=acc_k)
            acc_k = hl.dot(gu.T.to(k.dtype), r_i, acc=acc_k)
            acc_v = hl.dot(weights.T.to(k.dtype), go_i, acc=acc_v)
        gk_buf[b, hq, tile_n, :] = acc_k
        gv_buf[b, hq, tile_n, :] = acc_v
    return gk_buf, gv_buf


# --------------------------------------------------------------------------- #
# Host wrappers + autograd Function.
# --------------------------------------------------------------------------- #
def parallax_fwd(q, r, k, v, qk_scale=None, window_size_left=-1):
    """Raw Helion Parallax forward. Returns (o, barv, d1, bart, m).

    Shapes: q, r ``(B, H_q, L, D)``; k, v ``(B, H_kv, L, D)``.
    """
    B, Hq, Lq, D = q.shape
    Hk = k.shape[1]
    if Hq % Hk != 0:
        raise ValueError(f"H_q ({Hq}) must be divisible by H_kv ({Hk}) for GQA")
    if qk_scale is None:
        qk_scale = D ** -0.5
    scale_log2 = float(qk_scale) * _LOG2E
    q, r, k, v = (x.contiguous() for x in (q, r, k, v))
    # Route by head_dim: the causal split wins at small D (per-block mask ALU is
    # a large fraction); at D >= 128 the masked single loop is faster (matmuls
    # dominate, so the two-loop overhead isn't worth the mask savings).
    if D <= 64:
        return _fwd_kernel_split(q, r, k, v, scale_log2, int(window_size_left))
    return _fwd_kernel(q, r, k, v, scale_log2, int(window_size_left))


def parallax_bwd(q, r, k, v, o, barv, d1, bart, m, grad_o,
                        qk_scale, window_size_left=-1):
    """Raw Helion Parallax backward. Returns (grad_q, grad_r, grad_k, grad_v).

    All (B, H, L, D); dK/dV are GQA-folded to the kv-head axis.
    """
    B, Hq, Lq, D = q.shape
    Hk = k.shape[1]
    n_rep = Hq // Hk
    Lk = k.shape[2]
    q, r, k, v, grad_o = (x.contiguous() for x in (q, r, k, v, grad_o))
    t_stat, b_stat = _bwd_preprocess(grad_o, o, barv)
    grad_q, grad_r = _bwd_rq(q, r, k, v, d1, bart, m, t_stat, b_stat, grad_o,
                             float(qk_scale), int(window_size_left))
    gk_buf, gv_buf = _bwd_kv(q, r, k, v, d1, bart, m, t_stat, b_stat, grad_o,
                             float(qk_scale), int(window_size_left))
    if n_rep == 1:
        grad_k, grad_v = gk_buf.to(torch.bfloat16), gv_buf.to(torch.bfloat16)
    else:
        grad_k = gk_buf.view(B, Hk, n_rep, Lk, D).sum(2).to(torch.bfloat16)
        grad_v = gv_buf.view(B, Hk, n_rep, Lk, D).sum(2).to(torch.bfloat16)
    return grad_q, grad_r, grad_k, grad_v


class ParallaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, r, k, v, qk_scale, window_size_left):
        o, barv, d1, bart, m = parallax_fwd(
            q, r, k, v, qk_scale=qk_scale, window_size_left=window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.qk_scale = qk_scale
        ctx.window_size_left = window_size_left
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        grad_q, grad_r, grad_k, grad_v = parallax_bwd(
            q, r, k, v, o, barv, d1, bart, m, grad_o.contiguous(),
            ctx.qk_scale, ctx.window_size_left)
        # Kernels accumulate/store grads in bf16; cast back so autograd sees
        # grads matching the input dtypes (fp16 inputs otherwise break).
        return (grad_q.to(q.dtype), grad_r.to(r.dtype),
                grad_k.to(k.dtype), grad_v.to(v.dtype), None, None)


def parallax_func(q, r, k, v, qk_scale=None, window_size_left=-1):
    """Parallax training pass (Helion), ``(B, H, L, D)`` API, with autograd.

    Drop-in for :func:`parallax.parallax_func` (same shapes / default scale).
    ``qk_scale`` defaults to ``1 / sqrt(D)``.
    """
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"parallax_func requires bf16 or fp16 inputs, got q.dtype={q.dtype}"
        )
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"H_q ({q.shape[1]}) must be divisible by H_kv ({k.shape[1]}) for GQA"
        )
    D = q.shape[-1]
    if qk_scale is None:
        qk_scale = D ** -0.5
    return ParallaxFunction.apply(q, r, k, v, qk_scale, window_size_left)
