# Copyright (c) 2026 Yifei Zuo.
# SPDX-License-Identifier: MIT
"""Forward Triton kernel for Parallax.

Returns the bf16 ``o`` and ``barv`` plus the fp32 per-row scalars
``d1, bart, m`` consumed by the backward. Supports GQA via ``n_rep``
(callers pass un-replicated K/V; loads rebase to ``pid_batch // n_rep``)
and causal sliding-window attention via ``window_size_left`` (FA2
convention; ``-1`` disables).
"""

import math

import torch
import triton
import triton.language as tl


_TILE_SIZES = (32, 64, 128)
_WARP_COUNTS = (4, 8)
_STAGE_COUNTS = (2, 4)
_CONFIGS = [
    triton.Config({"ROW_TILE_SIZE": r, "COL_TILE_SIZE": c}, num_warps=w, num_stages=s)
    for r in _TILE_SIZES
    for c in _TILE_SIZES
    for w in _WARP_COUNTS
    for s in _STAGE_COUNTS
]


@triton.autotune(
    configs=_CONFIGS,
    key=["N_QUERIES", "N_KEYVALS", "HEAD_DIM", "N_REP", "WINDOW_SIZE_LEFT"],
)
@triton.jit
def _fwd_kernel(
    q_ptr,
    r_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    barv_ptr,
    d1_ptr,
    bart_ptr,
    m_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_rb, stride_rq, stride_rd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_barv_b, stride_barv_q, stride_barv_d,
    stride_d1_b, stride_d1_q,
    stride_bart_b, stride_bart_q,
    stride_m_b, stride_m_q,
    qk_scale,
    N_QUERIES,
    N_KEYVALS,
    HEAD_DIM: tl.constexpr,
    N_REP: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1)
    kv_batch_idx = pid_batch // N_REP
    pid_row = tl.program_id(0)
    row_offset = pid_row * ROW_TILE_SIZE
    row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
    row_mask = row_indices[:, None] < N_QUERIES
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(N_KEYVALS, row_offset + ROW_TILE_SIZE), COL_TILE_SIZE)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, N_KEYVALS) // COL_TILE_SIZE

    # SWA col-block boundaries. WINDOW_SIZE_LEFT < 0 disables SWA.
    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // COL_TILE_SIZE
        SAFE_LEFT_START = (leftmost_valid + COL_TILE_SIZE - 1) // COL_TILE_SIZE
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + pid_batch * stride_qb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_qq, stride_qd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    r_block_ptr = tl.make_block_ptr(
        base=r_ptr + pid_batch * stride_rb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_rq, stride_rd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + kv_batch_idx * stride_kb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_kk, stride_kd),
        offsets=(FIRST_COL_BLOCK * COL_TILE_SIZE, 0), block_shape=(COL_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + kv_batch_idx * stride_vb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_vk, stride_vd),
        offsets=(FIRST_COL_BLOCK * COL_TILE_SIZE, 0), block_shape=(COL_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + pid_batch * stride_ob,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_oq, stride_od),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    barv_block_ptr = tl.make_block_ptr(
        base=barv_ptr + pid_batch * stride_barv_b,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_barv_q, stride_barv_d),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    d1_block_ptr = tl.make_block_ptr(
        base=d1_ptr + pid_batch * stride_d1_b,
        shape=(N_QUERIES, 1), strides=(stride_d1_q, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    bart_block_ptr = tl.make_block_ptr(
        base=bart_ptr + pid_batch * stride_bart_b,
        shape=(N_QUERIES, 1), strides=(stride_bart_q, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    m_block_ptr = tl.make_block_ptr(
        base=m_ptr + pid_batch * stride_m_b,
        shape=(N_QUERIES, 1), strides=(stride_m_q, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )

    Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m_acc = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32)
    d2_acc = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32)
    barv_acc = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    Rv_acc = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    qk_scale_log2 = qk_scale * 1.44269504

    # Phase 0: left-border blocks (SWA only). Window mask only — below the
    # causal diagonal but straddling the left window edge.
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_offset = col_block_id * COL_TILE_SIZE
        col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < N_KEYVALS)
        )
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.max(qk, axis=1, keep_dims=True)
        m_new = tl.maximum(m_acc, m_new)
        # Rows whose window has not started yet stay at m_new == -inf;
        # force alpha=0, w=0 to avoid exp2(-inf - -inf) = NaN.
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_acc - safe_m)
        w = tl.math.exp2(qk - safe_m)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(tl.bfloat16), V, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(tl.bfloat16), V, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    # Phase A: safe blocks (no mask).
    for col_block_id in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        m_new = tl.max(qk, axis=1, keep_dims=True)
        m_new = tl.maximum(m_acc, m_new)
        alpha = tl.math.exp2(m_acc - m_new)
        w = tl.math.exp2(qk - m_new)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(tl.bfloat16), V, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(tl.bfloat16), V, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_offset = col_block_id * COL_TILE_SIZE
        col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < N_KEYVALS)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < N_KEYVALS)
            )
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.max(qk, axis=1, keep_dims=True)
        m_new = tl.maximum(m_acc, m_new)
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_acc - safe_m)
        w = tl.math.exp2(qk - safe_m)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(tl.bfloat16), V, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(tl.bfloat16), V, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    barv = barv_acc * inv_d1
    bart = d2_acc * inv_d1
    o = barv + bart * barv - Rv_acc * inv_d1

    tl.store(o_block_ptr, o.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(barv_block_ptr, barv.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(d1_block_ptr, d1_acc, boundary_check=(0, 1))
    tl.store(bart_block_ptr, bart, boundary_check=(0, 1))
    tl.store(m_block_ptr, m_acc, boundary_check=(0, 1))


def parallax_fwd(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float | torch.Tensor,
    n_rep: int = 1,
    window_size_left: int = -1,
):
    """Parallax forward pass (Triton, training).

    Args:
        q, r: ``(B * H_q, L_q, D)`` bf16/fp16.
        k, v: ``(B * H_kv, L_kv, D)`` un-replicated K/V, where ``H_kv = H_q // n_rep``.
        qk_scale: typically ``1 / sqrt(D)``.
        n_rep: GQA group size (``H_q // H_kv``). Default 1 = MHA.
        window_size_left: causal sliding-window length (FA2 convention).
            ``-1`` disables; ``>= 0`` restricts row ``i`` to cols
            ``[i - window_size_left + 1, i]``.

    Returns:
        ``(o, barv, d1, bart, m)`` — ``o`` and ``barv`` are ``(B * H_q, L_q, D)``
        bf16; ``d1``, ``bart``, ``m`` are ``(B * H_q, L_q, 1)`` fp32 per-row
        scalars for the backward.
    """
    batch_size, n_queries, head_dim = q.shape
    n_keyvals = k.shape[1]
    assert batch_size % n_rep == 0, (
        f"q batch axis {batch_size} must be divisible by n_rep={n_rep}"
    )
    assert k.shape[0] == batch_size // n_rep, (
        f"k batch axis {k.shape[0]} must equal q batch axis // n_rep "
        f"({batch_size}//{n_rep}={batch_size // n_rep})"
    )
    o = torch.empty((batch_size, n_queries, head_dim), device=q.device, dtype=q.dtype)
    barv = torch.empty((batch_size, n_queries, head_dim), device=q.device, dtype=q.dtype)
    d1 = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    bart = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    m = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    grid = lambda META: (math.ceil(n_queries / META["ROW_TILE_SIZE"]), batch_size)
    _fwd_kernel[grid](
        q, r, k, v, o, barv, d1, bart, m,
        q.stride(0), q.stride(1), q.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        barv.stride(0), barv.stride(1), barv.stride(2),
        d1.stride(0), d1.stride(1),
        bart.stride(0), bart.stride(1),
        m.stride(0), m.stride(1),
        qk_scale,
        n_queries,
        n_keyvals,
        head_dim,
        n_rep,
        window_size_left,
    )
    return o, barv, d1, bart, m
