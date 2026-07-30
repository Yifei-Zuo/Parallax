# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: MIT
"""Variable-length (packed) Parallax training pass in Helion (fwd + backward).

Drop-in for :func:`parallax.parallax_varlen_func`. Sequences are packed along the
token axis (batch size 1) and addressed via ``cu_seqlens``; heads-last layout
``(B, T, HQ, D)``. Query token ``i`` attends key ``j`` iff ``seq_start[i] <= j <= i``
(plus the optional sliding window), with loop bounds derived from per-token
``seq_start`` / ``seq_end`` tensors. The backward mirrors the dense
:mod:`parallax.helion.parallax_train` structure: preprocess + grad_q/grad_r +
grad_k/grad_v (no atomics); GQA folds per-q-head dK/dV back to the kv-head axis.
"""
from __future__ import annotations

import torch
import helion
import helion.language as hl

_LOG2E = 1.4426950216


@helion.kernel(static_shapes=True)
def _varlen_fwd_kernel(q, r, k, v, seq_start, scale_log2: float, WINDOW_LEFT: int):
    B, T, HQ, D = q.shape
    H = k.size(2)
    D = hl.specialize(D)
    G = HQ // H
    o = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1o = torch.empty([B, T, HQ], device=q.device, dtype=torch.float32)
    barto = torch.empty([B, T, HQ], device=q.device, dtype=torch.float32)
    mo = torch.empty([B, T, HQ], device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(T)
    block_n = hl.register_block_size(T)
    for tile_b, tile_h, tile_m in hl.tile([B, HQ, T], block_size=[1, 1, block_m]):
        b = tile_b.begin
        hq = tile_h.begin
        hk = hq // G
        q_i = q[b, tile_m, hq, :]
        r_i = r[b, tile_m, hq, :]
        ti = tile_m.index[:, None]
        ss = seq_start[b, tile_m]
        ss_i = ss[:, None]
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        d1_i = hl.zeros([tile_m], dtype=torch.float32)
        d2_i = hl.zeros([tile_m], dtype=torch.float32)
        o1 = hl.zeros([tile_m, D], dtype=torch.float32)
        o2 = hl.zeros([tile_m, D], dtype=torch.float32)
        lo = torch.amin(ss, 0)
        hi = torch.amax(tile_m.index, 0) + 1
        hi = torch.minimum(hi, hi.new_full([], T))
        for tile_n in hl.tile(lo, hi, block_size=block_n):
            kt = k[b, tile_n, hk, :]
            vt = v[b, tile_n, hk, :]
            tj = tile_n.index[None, :]
            s1 = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            s2 = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            causal = (tj <= ti) & (tj >= ss_i)
            if WINDOW_LEFT >= 0:
                mask = causal & (tj >= ti - WINDOW_LEFT + 1)
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
        o[b, tile_m, hq, :] = o_t.to(o.dtype)
        barv[b, tile_m, hq, :] = barv_t.to(barv.dtype)
        d1o[b, tile_m, hq] = d1_i
        barto[b, tile_m, hq] = bart_t
        mo[b, tile_m, hq] = m_i
    return o, barv, d1o, barto, mo


@helion.kernel(static_shapes=True,
               config=helion.Config(block_sizes=[128], num_warps=4, num_stages=2))
def _varlen_preprocess(grad_o, o, barv):
    B, T, HQ, D = grad_o.shape
    D = hl.specialize(D)
    t_out = torch.empty([B, T, HQ], device=o.device, dtype=torch.float32)
    b_out = torch.empty([B, T, HQ], device=o.device, dtype=torch.float32)
    for tile_b, tile_h, tile_m in hl.tile([B, HQ, T], block_size=[1, 1, hl.register_block_size(T)]):
        b = tile_b.begin
        hq = tile_h.begin
        go_t = grad_o[b, tile_m, hq, :].to(torch.float32)
        oo = o[b, tile_m, hq, :].to(torch.float32)
        bv = barv[b, tile_m, hq, :].to(torch.float32)
        t_out[b, tile_m, hq] = torch.sum(go_t * oo, -1)
        b_out[b, tile_m, hq] = torch.sum(go_t * bv, -1)
    return t_out, b_out


@helion.kernel(static_shapes=True)
def _varlen_dqr(q, r, k, v, d1o, barto, mo, to, bo, grad_o, seq_start,
                qk_scale: float, WINDOW_LEFT: int):
    B, T, HQ, D = q.shape
    H = k.size(2)
    D = hl.specialize(D)
    G = HQ // H
    scale_log2 = qk_scale * _LOG2E
    grad_q = torch.empty([B, T, HQ, D], device=q.device, dtype=torch.bfloat16)
    grad_r = torch.empty([B, T, HQ, D], device=q.device, dtype=torch.bfloat16)
    block_m = hl.register_block_size(T)
    block_n = hl.register_block_size(T)
    for tile_b, tile_h, tile_m in hl.tile([B, HQ, T], block_size=[1, 1, block_m]):
        b = tile_b.begin
        hq = tile_h.begin
        hk = hq // G
        q_i = q[b, tile_m, hq, :]
        r_i = r[b, tile_m, hq, :]
        go_i = grad_o[b, tile_m, hq, :]
        ti = tile_m.index[:, None]
        ss = seq_start[b, tile_m]
        ss_i = ss[:, None]
        d1_i = d1o[b, tile_m, hq][:, None]
        bart_i = barto[b, tile_m, hq][:, None]
        m_i = mo[b, tile_m, hq][:, None]
        t_i = to[b, tile_m, hq][:, None]
        b_i = bo[b, tile_m, hq][:, None]
        inv_d1 = torch.where(d1_i > 0.0, 1.0 / d1_i, 0.0)
        acc_q = hl.zeros([tile_m, D], dtype=torch.float32)
        acc_r = hl.zeros([tile_m, D], dtype=torch.float32)
        lo = torch.amin(ss, 0)
        hi = torch.amax(tile_m.index, 0) + 1
        hi = torch.minimum(hi, hi.new_full([], T))
        for tile_n in hl.tile(lo, hi, block_size=block_n):
            kt = k[b, tile_n, hk, :]
            vt = v[b, tile_n, hk, :]
            tj = tile_n.index[None, :]
            qk = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            rk = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            causal = (tj <= ti) & (tj >= ss_i)
            if WINDOW_LEFT >= 0:
                mask = causal & (tj >= ti - WINDOW_LEFT + 1)
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
        grad_q[b, tile_m, hq, :] = (acc_q * qk_scale).to(torch.bfloat16)
        grad_r[b, tile_m, hq, :] = acc_r.to(torch.bfloat16)
    return grad_q, grad_r


@helion.kernel(static_shapes=True)
def _varlen_dkv(q, r, k, v, d1o, barto, mo, to, bo, grad_o, seq_start, seq_end,
                qk_scale: float, WINDOW_LEFT: int):
    B, T, HQ, D = q.shape
    H = k.size(2)
    D = hl.specialize(D)
    G = HQ // H
    scale_log2 = qk_scale * _LOG2E
    gk_buf = torch.empty([B, T, HQ, D], device=q.device, dtype=torch.float32)
    gv_buf = torch.empty([B, T, HQ, D], device=q.device, dtype=torch.float32)
    block_m = hl.register_block_size(T)
    block_n = hl.register_block_size(T)
    for tile_b, tile_h, tile_n in hl.tile([B, HQ, T], block_size=[1, 1, block_n]):
        b = tile_b.begin
        hq = tile_h.begin
        hk = hq // G
        kt = k[b, tile_n, hk, :]
        vt = v[b, tile_n, hk, :]
        tj = tile_n.index[None, :]
        se = seq_end[b, tile_n]
        acc_k = hl.zeros([tile_n, D], dtype=torch.float32)
        acc_v = hl.zeros([tile_n, D], dtype=torch.float32)
        lo_row = torch.amin(tile_n.index, 0)
        hi_row = torch.amax(se, 0)
        hi_row = torch.minimum(hi_row, hi_row.new_full([], T))
        for tile_m in hl.tile(lo_row, hi_row, block_size=block_m):
            q_i = q[b, tile_m, hq, :]
            r_i = r[b, tile_m, hq, :]
            go_i = grad_o[b, tile_m, hq, :]
            ti = tile_m.index[:, None]
            ss_i = seq_start[b, tile_m][:, None]
            d1_i = d1o[b, tile_m, hq][:, None]
            bart_i = barto[b, tile_m, hq][:, None]
            m_i = mo[b, tile_m, hq][:, None]
            t_i = to[b, tile_m, hq][:, None]
            b_i = bo[b, tile_m, hq][:, None]
            inv_d1 = torch.where(d1_i > 0.0, 1.0 / d1_i, 0.0)
            qk = hl.dot(q_i, kt.T, out_dtype=torch.float32) * scale_log2
            rk = hl.dot(r_i, kt.T, out_dtype=torch.float32)
            causal = (tj <= ti) & (tj >= ss_i)
            if WINDOW_LEFT >= 0:
                mask = causal & (tj >= ti - WINDOW_LEFT + 1)
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
        gk_buf[b, tile_n, hq, :] = acc_k
        gv_buf[b, tile_n, hq, :] = acc_v
    return gk_buf, gv_buf


# --------------------------------------------------------------------------- #
# Host: seq bounds, raw fwd/bwd, autograd Function, public entry.
# --------------------------------------------------------------------------- #
def _seq_bounds(cu_seqlens, B, T, device):
    """Per-token (B, T) int32 seq_start / seq_end (first / one-past-last token of
    each token's sequence). Packed (cu_seqlens given, B==1) or dense (each of the
    B rows is one length-T causal sequence)."""
    if cu_seqlens is None:
        ss = torch.zeros((B, T), device=device, dtype=torch.int32)
        se = torch.full((B, T), T, device=device, dtype=torch.int32)
        return ss, se
    cu = cu_seqlens.to(device=device, dtype=torch.int32)
    lens = (cu[1:] - cu[:-1]).to(torch.long)
    ss = torch.repeat_interleave(cu[:-1], lens).view(1, T)
    se = torch.repeat_interleave(cu[1:], lens).view(1, T)
    return ss, se


def parallax_varlen_fwd(q, r, k, v, seq_start, qk_scale, window_size_left=-1):
    """Raw varlen forward. Returns (o, barv, d1, bart, m); ``seq_start`` is (B, T)."""
    scale_log2 = float(qk_scale) * _LOG2E
    return _varlen_fwd_kernel(q, r, k, v, seq_start, scale_log2, int(window_size_left))


def parallax_varlen_bwd(q, r, k, v, o, barv, d1, bart, m, grad_o,
                        seq_start, seq_end, qk_scale, window_size_left=-1):
    """Raw varlen backward. Returns (grad_q, grad_r, grad_k, grad_v), GQA-folded."""
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    t_stat, b_stat = _varlen_preprocess(grad_o, o, barv)
    grad_q, grad_r = _varlen_dqr(q, r, k, v, d1, bart, m, t_stat, b_stat, grad_o,
                                 seq_start, float(qk_scale), int(window_size_left))
    gk_buf, gv_buf = _varlen_dkv(q, r, k, v, d1, bart, m, t_stat, b_stat, grad_o,
                                 seq_start, seq_end, float(qk_scale), int(window_size_left))
    if G == 1:
        grad_k, grad_v = gk_buf.to(torch.bfloat16), gv_buf.to(torch.bfloat16)
    else:
        grad_k = gk_buf.view(B, T, H, G, D).sum(3).to(torch.bfloat16)
        grad_v = gv_buf.view(B, T, H, G, D).sum(3).to(torch.bfloat16)
    return grad_q, grad_r, grad_k, grad_v


class ParallaxVarlenFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, r, k, v, qk_scale, window_size_left, cu_seqlens):
        B, T = q.shape[0], q.shape[1]
        seq_start, seq_end = _seq_bounds(cu_seqlens, B, T, q.device)
        o, barv, d1, bart, m = parallax_varlen_fwd(q, r, k, v, seq_start, qk_scale, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m, seq_start, seq_end)
        ctx.qk_scale = qk_scale
        ctx.window_size_left = window_size_left
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, r, k, v, o, barv, d1, bart, m, seq_start, seq_end = ctx.saved_tensors
        gq, gr, gk, gv = parallax_varlen_bwd(
            q, r, k, v, o, barv, d1, bart, m, grad_o.contiguous(),
            seq_start, seq_end, ctx.qk_scale, ctx.window_size_left)
        return gq.to(q.dtype), gr.to(r.dtype), gk.to(k.dtype), gv.to(v.dtype), None, None, None


def parallax_varlen_func(q, r, k, v, qk_scale=None, window_size_left=-1, cu_seqlens=None):
    """Variable-length (packed) causal Parallax training (Helion), with autograd.

    Drop-in for :func:`parallax.parallax_varlen_func`. Heads-last ``(B, T, HQ, D)``
    layout (``k, v`` are ``(B, T, H, D)``).

    Args:
        q, r: ``(B, T, HQ, D)`` bf16/fp16. k, v: ``(B, T, H, D)``, GQA when
            ``HQ % H == 0``.
        qk_scale: defaults to ``1 / sqrt(D)``.
        window_size_left: causal sliding-window length (FA2 convention); ``-1`` off.
        cu_seqlens: cumulative sequence lengths ``[N+1]`` (int32/int64) for packed
            varlen (batch size must be 1). ``None`` runs the dense path (each of
            the ``B`` rows is one length-``T`` causal sequence).
    """
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallax_varlen_func requires bf16/fp16 inputs, got {q.dtype}")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"H_q ({q.shape[2]}) must be divisible by H_kv ({k.shape[2]}) for GQA")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"batch size must be 1 (got {q.shape[0]}) with cu_seqlens; pack "
                f"variable-length inputs into one sequence first.")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")
    if qk_scale is None:
        qk_scale = q.shape[-1] ** -0.5
    q, r, k, v = (t.contiguous() for t in (q, r, k, v))
    return ParallaxVarlenFunction.apply(q, r, k, v, float(qk_scale), window_size_left, cu_seqlens)
