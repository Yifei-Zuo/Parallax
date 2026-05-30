# Copyright (c) 2026 Yifei Zuo.
# SPDX-License-Identifier: MIT
"""Forward Triton kernel for Parallax attention (Algorithm 1).

Returns ``(o, barv, d1, bart, m)`` — bf16 outputs ``o`` and ``barv`` plus the
fp32 per-row scalars ``d1``, ``bart``, and ``m`` that the backward consumes.
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


@triton.autotune(configs=_CONFIGS, key=["N_QUERIES", "N_KEYVALS", "HEAD_DIM"])
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
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1)
    pid_row = tl.program_id(0)
    row_offset = pid_row * ROW_TILE_SIZE
    row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
    row_mask = row_indices[:, None] < N_QUERIES
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(N_KEYVALS, row_offset + ROW_TILE_SIZE), COL_TILE_SIZE)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, N_KEYVALS) // COL_TILE_SIZE

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
        base=k_ptr + pid_batch * stride_kb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_kk, stride_kd),
        offsets=(0, 0), block_shape=(COL_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + pid_batch * stride_vb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_vk, stride_vd),
        offsets=(0, 0), block_shape=(COL_TILE_SIZE, HEAD_DIM), order=(1, 0),
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

    # Phase A: safe blocks (no mask).
    for col_block_id in range(NUM_SAFE_BLOCKS):
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

    # Phase B: border blocks (causal + boundary mask).
    for col_block_id in range(NUM_SAFE_BLOCKS, NUM_TOTAL_BLOCKS):
        col_offset = col_block_id * COL_TILE_SIZE
        col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        mask = (
            (row_indices[:, None] >= col_indices[None, :])
            & row_mask
            & (col_indices[None, :] < N_KEYVALS)
        )
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        qk = tl.where(mask, qk, -float("inf"))
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
):
    """Parallax forward pass (Triton, training).

    Args:
        q, r: ``(B, L_q, D)`` bf16 / fp16.
        k, v: ``(B, L_kv, D)`` same dtype as ``q``.
        qk_scale: typically ``1 / sqrt(D)``.

    Returns:
        ``(o, barv, d1, bart, m)`` — ``o`` and ``barv`` are ``(B, L_q, D)`` bf16;
        ``d1``, ``bart``, ``m`` are ``(B, L_q, 1)`` fp32 per-row scalars for the
        backward.
    """
    batch_size, n_queries, head_dim = q.shape
    n_keyvals = k.shape[1]
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
    )
    return o, barv, d1, bart, m
