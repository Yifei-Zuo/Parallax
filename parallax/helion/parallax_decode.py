# Copyright (c) 2026 Yifei Zuo.
# SPDX-License-Identifier: MIT
"""Single-token Parallax decode in Helion.

A Helion (PyTorch-embedded, autotuned, compiles-to-Triton) implementation of the
single-token (``Sq == 1``) Parallax decode, for benchmarking against the
hand-written pure-Triton decode (:func:`parallax.triton.parallax_decode`) and the
CuTeDSL SM90 decode (:func:`parallax.parallax_decode`). Forward-only.

Layout matches :func:`parallax.parallax_reference`: ``q, r`` are ``(B, 1, H_q, D)``
and ``k, v`` are ``(B, L, H_kv, D)`` with ``H_q % H_kv == 0`` (GQA).

Two paths, chosen by the wrapper:
  * **base** — one program per ``(batch, query-head)`` = ``B*H_q`` CTAs, serial KV
    reduction. Best when ``B*H_q`` fills the GPU. Handles SWA + cache_start.
  * **split-KV** (flash-decoding) — grid ``B*H_q*num_splits``: each program reduces
    a KV slice to per-split ``(m,d1,d2,O1,O2)``, then a merge kernel LSE-combines
    the splits. Best for small ``B*H_q`` / long ``L`` (fills the GPU); ncu showed
    the base kernel there lights only ``B*H_q`` of 132 SMs (DRAM ~1.7% of peak).
    Used only for the plain full-cache path (no window / no cache_start).
"""
from __future__ import annotations

import os
import torch
import helion
import helion.language as hl

_LOG2E = 1.4426950216  # 1 / ln(2); s1 is accumulated in base-2 units.
_NUM_SMS = 132         # H100


# --------------------------------------------------------------------------- #
# Base kernel: one program per (batch, query head). Handles SWA + cache_start.
# --------------------------------------------------------------------------- #
@helion.kernel(static_shapes=True)
def _parallax_decode_kernel(
    q_in: torch.Tensor,        # (B, 1, H_q, D)
    r_in: torch.Tensor,        # (B, 1, H_q, D)
    k_in: torch.Tensor,        # (B, L, H_kv, D)
    v_in: torch.Tensor,        # (B, L, H_kv, D)
    cache_start: torch.Tensor,  # (B,) int32, first valid key per batch
    scale_log2: float,
    WINDOW_LEFT: int,
):
    B, Sq, Hq, D = q_in.shape
    L = k_in.size(1)
    Hk = k_in.size(2)
    D = hl.specialize(D)
    n_rep = Hq // Hk
    out = torch.empty_like(q_in)
    for tile_b, tile_h in hl.tile([B, Hq], block_size=[1, 1]):
        b = tile_b.begin
        hq = tile_h.begin
        hk = hq // n_rep
        lo = cache_start[b]
        if WINDOW_LEFT >= 0:
            lo = torch.maximum(lo, lo.new_full([], L - WINDOW_LEFT))
        lo = torch.maximum(lo, lo.new_full([], 0))

        q_i = q_in[b, :, hq, :].to(torch.float32)   # (1, D)
        r_i = r_in[b, :, hq, :].to(torch.float32)   # (1, D)
        m = hl.full([1], float("-inf"), dtype=torch.float32)
        d1 = hl.zeros([1], dtype=torch.float32)
        d2 = hl.zeros([1], dtype=torch.float32)
        o1 = hl.zeros([1, D], dtype=torch.float32)
        o2 = hl.zeros([1, D], dtype=torch.float32)
        for tile_n in hl.tile(L):
            col = tile_n.index
            valid = (col >= lo) & (col < L)
            k = k_in[b, tile_n, hk, :].to(torch.float32)
            v = v_in[b, tile_n, hk, :]
            s1 = torch.matmul(q_i, k.T) * scale_log2
            s2 = torch.matmul(r_i, k.T)
            s1 = torch.where(valid[None, :], s1, float("-inf"))
            m_new = torch.maximum(m, torch.amax(s1, -1))
            m_safe = torch.where(m_new == float("-inf"), 0.0, m_new)
            alpha = torch.exp2(m - m_safe)
            p1 = torch.exp2(s1 - m_safe[:, None])
            p2 = p1 * s2
            d1 = d1 * alpha + torch.sum(p1, -1)
            d2 = d2 * alpha + torch.sum(p2, -1)
            o1 = o1 * alpha[:, None] + torch.matmul(p1.to(v.dtype), v).to(torch.float32)
            o2 = o2 * alpha[:, None] + torch.matmul(p2.to(v.dtype), v).to(torch.float32)
            m = m_new
        inv = torch.where(d1 > 0.0, 1.0 / d1, 0.0)
        c = d2 * inv
        o = o1 * inv[:, None] * (1.0 + c[:, None]) - o2 * inv[:, None]
        out[b, :, hq, :] = o.to(out.dtype)
    return out


# --------------------------------------------------------------------------- #
# Split-KV (flash-decoding): partial reduction per KV slice + LSE merge.
# --------------------------------------------------------------------------- #
@helion.kernel(static_shapes=True)
def _parallax_decode_partial(
    q_in, r_in, k_in, v_in,
    bounds,               # (S+1,) int32 split boundaries, clamped to L (no OOB)
    scale_log2: float,
):
    B, Sq, Hq, D = q_in.shape
    Hk = k_in.size(2)
    S = bounds.size(0) - 1
    D = hl.specialize(D)
    n_rep = Hq // Hk
    pm = torch.empty([B, Hq, S], device=q_in.device, dtype=torch.float32)
    pd1 = torch.empty([B, Hq, S], device=q_in.device, dtype=torch.float32)
    pd2 = torch.empty([B, Hq, S], device=q_in.device, dtype=torch.float32)
    pO1 = torch.empty([B, Hq, S, D], device=q_in.device, dtype=torch.float32)
    pO2 = torch.empty([B, Hq, S, D], device=q_in.device, dtype=torch.float32)
    for tb, th, ts in hl.tile([B, Hq, S], block_size=[1, 1, 1]):
        b = tb.begin
        hq = th.begin
        s = ts.begin
        hk = hq // n_rep
        n_lo = bounds[s]          # tensor scalars; n_hi <= L (clamped) -> no OOB
        n_hi = bounds[s + 1]
        q_i = q_in[b, :, hq, :].to(torch.float32)
        r_i = r_in[b, :, hq, :].to(torch.float32)
        m = hl.full([1], float("-inf"), dtype=torch.float32)
        d1 = hl.zeros([1], dtype=torch.float32)
        d2 = hl.zeros([1], dtype=torch.float32)
        o1 = hl.zeros([1, D], dtype=torch.float32)
        o2 = hl.zeros([1, D], dtype=torch.float32)
        for tn in hl.tile(n_lo, n_hi):
            kt = k_in[b, tn, hk, :].to(torch.float32)
            vt = v_in[b, tn, hk, :]
            s1 = torch.matmul(q_i, kt.T) * scale_log2
            s2 = torch.matmul(r_i, kt.T)
            m_new = torch.maximum(m, torch.amax(s1, -1))
            m_safe = torch.where(m_new == float("-inf"), 0.0, m_new)
            alpha = torch.exp2(m - m_safe)
            p1 = torch.exp2(s1 - m_safe[:, None])
            p2 = p1 * s2
            d1 = d1 * alpha + torch.sum(p1, -1)
            d2 = d2 * alpha + torch.sum(p2, -1)
            o1 = o1 * alpha[:, None] + torch.matmul(p1.to(vt.dtype), vt).to(torch.float32)
            o2 = o2 * alpha[:, None] + torch.matmul(p2.to(vt.dtype), vt).to(torch.float32)
            m = m_new
        pm[b, hq, ts] = m
        pd1[b, hq, ts] = d1
        pd2[b, hq, ts] = d2
        pO1[b, hq, ts, :] = o1
        pO2[b, hq, ts, :] = o2
    return pm, pd1, pd2, pO1, pO2


@helion.kernel(static_shapes=True)
def _parallax_decode_merge(pm, pd1, pd2, pO1, pO2, out_dtype: hl.constexpr):
    B, Hq, S = pm.shape
    D = hl.specialize(pO1.size(-1))
    out = torch.empty([B, 1, Hq, D], device=pm.device, dtype=out_dtype)
    for tb, th in hl.tile([B, Hq], block_size=[1, 1]):
        b = tb.begin
        hq = th.begin
        m_all = pm[b, hq, :]                        # (S,)
        mg = torch.amax(m_all, -1, keepdim=True)
        mg_safe = torch.where(mg == float("-inf"), 0.0, mg)
        scl = torch.exp2(m_all - mg_safe)           # (S,) cross-split rescale
        d1 = torch.sum(pd1[b, hq, :] * scl, -1, keepdim=True)
        d2 = torch.sum(pd2[b, hq, :] * scl, -1, keepdim=True)
        O1 = torch.sum(pO1[b, hq, :, :] * scl[:, None], 0)   # (D,)
        O2 = torch.sum(pO2[b, hq, :, :] * scl[:, None], 0)
        inv = torch.where(d1 > 0.0, 1.0 / d1, 0.0)
        res = O1 * inv * (1.0 + d2 * inv) - O2 * inv         # (D,)
        out[b, :, hq, :] = res[None, :].to(out_dtype)
    return out


# --------------------------------------------------------------------------- #
# Host wrapper (Triton-decode-compatible signature) + base/split routing.
# --------------------------------------------------------------------------- #
_CS_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}
_BOUNDS_CACHE: dict[tuple, torch.Tensor] = {}


def _zeros_cache_start(B: int, device: torch.device) -> torch.Tensor:
    key = (B, device)
    buf = _CS_CACHE.get(key)
    if buf is None or buf.numel() < B:
        buf = torch.zeros(B, device=device, dtype=torch.int32)
        _CS_CACHE[key] = buf
    return buf[:B]


def _split_bounds(L: int, num_splits: int, device: torch.device) -> torch.Tensor:
    """Cached (num_splits+1,) int32 split boundaries, clamped to L (graph-safe)."""
    key = (L, num_splits, device)
    b = _BOUNDS_CACHE.get(key)
    if b is None:
        seg = (L + num_splits - 1) // num_splits
        b = (torch.arange(num_splits + 1, device=device, dtype=torch.int32) * seg).clamp_(max=L)
        _BOUNDS_CACHE[key] = b
    return b


def _choose_num_splits(B: int, Hq: int, L: int) -> int:
    """Split only when the base grid (B*Hq CTAs) underfills the GPU; else 1.

    ``PLX_DECODE_SPLITS`` env overrides the count (experimental autoresearch knob):
    an int forces that many splits; ``waves<N>`` aims for N waves of CTAs.
    """
    bh = B * Hq
    if bh >= _NUM_SMS or L < 1024:
        return 1
    ov = os.environ.get("PLX_DECODE_SPLITS")
    if ov:
        if ov.startswith("waves"):
            waves = int(ov[5:] or "1")
            ns = -(-(_NUM_SMS * waves) // bh)
        else:
            ns = int(ov)
        return max(1, min(ns, L // 128))
    # Aim for ~2 waves of CTAs: the extra concurrency hides HBM latency better
    # than 1 wave (autoresearch iter 2: 2 waves matched/beat CuTe on 8-head
    # long-context shapes; 3 waves regressed via merge overhead).
    ns = -(-(_NUM_SMS * 2) // bh)     # ceil(2*SMs / bh)
    ns = min(ns, L // 128)           # keep >= ~128 keys/split
    return max(1, ns)


def parallax_decode(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float | None = None,
    *,
    window_size_left: int = -1,
    cache_start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-token Parallax decode (``Sq == 1``), Helion. Forward-only.

    Drop-in for :func:`parallax.triton.parallax_decode`. Routes to a split-KV
    (flash-decoding) path for small ``B*H_q`` / long ``L`` full-cache decode; falls
    back to the base kernel for windowed / left-padded / large-batch cases.
    """
    B, Sq, Hq, D = q.shape
    if Sq != 1:
        raise ValueError(f"parallax_decode requires a single query (Sq=1), got Sq={Sq}.")
    Skv, Hk = k.shape[1], k.shape[2]
    if Hq % Hk != 0:
        raise ValueError(f"H_q ({Hq}) must be divisible by H_kv ({Hk}) for GQA")
    if qk_scale is None:
        qk_scale = D ** -0.5
    scale_log2 = float(qk_scale) * _LOG2E
    q, r, k, v = (x.contiguous() for x in (q, r, k, v))

    # Split-KV fast path: plain full cache (no window, no left-padding), small B*Hq.
    if window_size_left < 0 and cache_start is None:
        ns = _choose_num_splits(B, Hq, Skv)
        if ns > 1:
            bounds = _split_bounds(Skv, ns, q.device)
            pm, pd1, pd2, pO1, pO2 = _parallax_decode_partial(q, r, k, v, bounds, scale_log2)
            return _parallax_decode_merge(pm, pd1, pd2, pO1, pO2, q.dtype)

    if cache_start is None:
        cs = _zeros_cache_start(B, q.device)
    else:
        cs = cache_start.to(device=q.device, dtype=torch.int32).contiguous()
    return _parallax_decode_kernel(q, r, k, v, cs, scale_log2, int(window_size_left))
