# Copyright (c) 2026 Yifei Zuo.
# SPDX-License-Identifier: MIT
"""Parallax dense causal training pass in Triton (forward + backward + autograd).

Forward kernel returns the bf16 ``o`` and ``barv`` plus the fp32 per-row scalars
``d1, bart, m`` consumed by the backward. Backward runs three kernels in
sequence: ``_bwd_preprocess_kernel`` (per-row ``t = Σ_d grad_o·o`` and
``b = Σ_d grad_o·barv``), ``_bwd_rq_kernel`` (``grad_q``/``grad_r``, parallel
over query rows) and ``_bwd_kv_kernel`` (``grad_k``/``grad_v``, parallel over
KV). GQA is supported via ``n_rep`` (callers pass un-replicated K/V; loads
rebase to ``pid_batch // n_rep``, and the backward folds per-q-head dK/dV back
to the kv-head axis). Causal sliding-window attention via ``window_size_left``
(FA2 convention; ``-1`` disables).

:func:`parallax_func` / :class:`ParallaxFunction` wrap the raw
:func:`parallax_fwd` / :func:`parallax_bwd` with autograd; ``parallax_func``
is the canonical ``(B, H, L, D)`` entry point.
"""

import math

import torch
import triton
import triton.language as tl


_TILE_SIZES = (32, 64, 128)
_WARP_COUNTS = (4, 8)
_STAGE_COUNTS = (2, 4)
# Each autotuner below takes its own list(_CONFIGS) copy — sharing one list
# across @triton.autotune decorators trips Dynamo ("ListVariable already
# tracked for mutation").
_CONFIGS = [
    triton.Config({"ROW_TILE_SIZE": r, "COL_TILE_SIZE": c}, num_warps=w, num_stages=s)
    for r in _TILE_SIZES
    for c in _TILE_SIZES
    for w in _WARP_COUNTS
    for s in _STAGE_COUNTS
]


@triton.autotune(
    configs=list(_CONFIGS),
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

# --------------------------------------------------------------------------- #
# Backward: preprocess + grad_q/grad_r + grad_k/grad_v.
# --------------------------------------------------------------------------- #

_PREPROCESS_CONFIGS = [
    triton.Config({"ROW_TILE_SIZE": r}, num_warps=w, num_stages=s)
    for r in (64, 128, 256)
    for w in (4, 8)
    for s in (2, 4)
]


def _prune_bwd_configs_by_head_dim(configs, named_args, **kwargs):
    # At HEAD_DIM >= 256, ROW_TILE_SIZE=128 spills registers.
    head_dim = named_args.get("HEAD_DIM", 0)
    if head_dim >= 256:
        pruned = [c for c in configs if c.kwargs["ROW_TILE_SIZE"] <= 64]
        return pruned if pruned else configs[:1]
    return configs


@triton.autotune(configs=_PREPROCESS_CONFIGS, key=["N_QUERIES", "HEAD_DIM"])
@triton.jit
def _bwd_preprocess_kernel(
    grad_o_ptr,
    o_ptr,
    barv_ptr,
    t_ptr,
    b_ptr,
    stride_gob, stride_goq, stride_god,
    stride_ob, stride_oq, stride_od,
    stride_barv_b, stride_barv_q, stride_barv_d,
    stride_tb, stride_tq,
    stride_bb, stride_bq,
    N_QUERIES,
    HEAD_DIM: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1)
    pid_row = tl.program_id(0)
    row_offset = pid_row * ROW_TILE_SIZE

    grad_o_block_ptr = tl.make_block_ptr(
        base=grad_o_ptr + pid_batch * stride_gob,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_goq, stride_god),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
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
    t_block_ptr = tl.make_block_ptr(
        base=t_ptr + pid_batch * stride_tb,
        shape=(N_QUERIES, 1), strides=(stride_tq, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + pid_batch * stride_bb,
        shape=(N_QUERIES, 1), strides=(stride_bq, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )

    grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_tile = tl.load(o_block_ptr, boundary_check=(0, 1), padding_option="zero")
    barv = tl.load(barv_block_ptr, boundary_check=(0, 1), padding_option="zero")

    grad_o_f32 = grad_o.to(tl.float32)
    t = tl.sum(grad_o_f32 * O_tile.to(tl.float32), axis=1, keep_dims=True)
    b = tl.sum(grad_o_f32 * barv.to(tl.float32), axis=1, keep_dims=True)

    tl.store(t_block_ptr, t, boundary_check=(0, 1))
    tl.store(b_block_ptr, b, boundary_check=(0, 1))


@triton.autotune(
    configs=list(_CONFIGS),
    key=["N_QUERIES", "N_KEYVALS", "HEAD_DIM", "N_REP", "WINDOW_SIZE_LEFT"],
    prune_configs_by={"early_config_prune": _prune_bwd_configs_by_head_dim},
)
@triton.jit
def _bwd_rq_kernel(
    q_ptr,
    r_ptr,
    k_ptr,
    v_ptr,
    d1_ptr,
    bart_ptr,
    m_ptr,
    t_ptr,
    b_ptr,
    grad_o_ptr,
    grad_q_ptr,
    grad_r_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_rb, stride_rq, stride_rd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_d1_b, stride_d1_q,
    stride_bart_b, stride_bart_q,
    stride_m_b, stride_m_q,
    stride_tb, stride_tq,
    stride_bb, stride_bq,
    stride_gob, stride_goq, stride_god,
    stride_gqb, stride_gqq, stride_gqd,
    stride_grb, stride_grq, stride_grd,
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

    # SWA col-block boundaries (see _fwd_kernel for derivation).
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
    t_block_ptr = tl.make_block_ptr(
        base=t_ptr + pid_batch * stride_tb,
        shape=(N_QUERIES, 1), strides=(stride_tq, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + pid_batch * stride_bb,
        shape=(N_QUERIES, 1), strides=(stride_bq, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    grad_o_block_ptr = tl.make_block_ptr(
        base=grad_o_ptr + pid_batch * stride_gob,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_goq, stride_god),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    grad_q_block_ptr = tl.make_block_ptr(
        base=grad_q_ptr + pid_batch * stride_gqb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_gqq, stride_gqd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    grad_r_block_ptr = tl.make_block_ptr(
        base=grad_r_ptr + pid_batch * stride_grb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_grq, stride_grd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )

    Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
    d1 = tl.load(d1_block_ptr, boundary_check=(0, 1), padding_option="zero")
    bart = tl.load(bart_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option="zero")
    t = tl.load(t_block_ptr, boundary_check=(0, 1), padding_option="zero")
    b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
    grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")
    grad_q_acc = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    grad_r_acc = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    qk_scale_log2 = qk_scale * 1.44269504

    inv_d1 = tl.where(row_mask, 1.0 / d1, 0.0)

    # Phase 0: left-border blocks (SWA only).
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
        w = tl.math.exp2(qk - m)
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = bart - rk
        delta = a - b
        gl = p * (a - t + bart_minus_rk * delta)
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(tl.bfloat16), K, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(tl.bfloat16), K, out_dtype=tl.float32, acc=grad_r_acc)
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    # Phase A: safe blocks (no mask).
    for col_block_id in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        w = tl.math.exp2(qk - m)
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = bart - rk
        delta = a - b
        gl = p * (a - t + bart_minus_rk * delta)
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(tl.bfloat16), K, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(tl.bfloat16), K, out_dtype=tl.float32, acc=grad_r_acc)
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
        w = tl.math.exp2(qk - m)
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = bart - rk
        delta = a - b
        gl = p * (a - t + bart_minus_rk * delta)
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(tl.bfloat16), K, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(tl.bfloat16), K, out_dtype=tl.float32, acc=grad_r_acc)
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    grad_q_acc = qk_scale * grad_q_acc

    tl.store(grad_q_block_ptr, grad_q_acc.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(grad_r_block_ptr, grad_r_acc.to(tl.bfloat16), boundary_check=(0, 1))


@triton.autotune(
    configs=list(_CONFIGS),
    key=["N_QUERIES", "N_KEYVALS", "HEAD_DIM", "N_REP", "WINDOW_SIZE_LEFT"],
    prune_configs_by={"early_config_prune": _prune_bwd_configs_by_head_dim},
)
@triton.jit
def _bwd_kv_kernel(
    q_ptr,
    r_ptr,
    k_ptr,
    v_ptr,
    d1_ptr,
    bart_ptr,
    m_ptr,
    t_ptr,
    b_ptr,
    grad_o_ptr,
    grad_k_ptr,
    grad_v_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_rb, stride_rq, stride_rd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_d1_b, stride_d1_q,
    stride_bart_b, stride_bart_q,
    stride_m_b, stride_m_q,
    stride_tb, stride_tq,
    stride_bb, stride_bq,
    stride_gob, stride_goq, stride_god,
    stride_gkb, stride_gkk, stride_gkd,
    stride_gvb, stride_gvk, stride_gvd,
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
    pid_col = tl.program_id(0)
    col_offset = pid_col * COL_TILE_SIZE
    col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)

    start_row_block = col_offset // ROW_TILE_SIZE
    start_row_offset = start_row_block * ROW_TILE_SIZE

    # SWA row-block boundaries:
    #   - num_row_blocks: cap after which rows can no longer reach this col block
    #   - WINDOW_SAFE_END: last row-block fully within W of every col in the block
    num_row_blocks_qbound = tl.cdiv(N_QUERIES, ROW_TILE_SIZE)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + COL_TILE_SIZE + WINDOW_SIZE_LEFT - 1, ROW_TILE_SIZE)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // ROW_TILE_SIZE
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + pid_batch * stride_qb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_qq, stride_qd),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    r_block_ptr = tl.make_block_ptr(
        base=r_ptr + pid_batch * stride_rb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_rq, stride_rd),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + kv_batch_idx * stride_kb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_kk, stride_kd),
        offsets=(col_offset, 0), block_shape=(COL_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + kv_batch_idx * stride_vb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_vk, stride_vd),
        offsets=(col_offset, 0), block_shape=(COL_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    d1_block_ptr = tl.make_block_ptr(
        base=d1_ptr + pid_batch * stride_d1_b,
        shape=(N_QUERIES, 1), strides=(stride_d1_q, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    bart_block_ptr = tl.make_block_ptr(
        base=bart_ptr + pid_batch * stride_bart_b,
        shape=(N_QUERIES, 1), strides=(stride_bart_q, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    m_block_ptr = tl.make_block_ptr(
        base=m_ptr + pid_batch * stride_m_b,
        shape=(N_QUERIES, 1), strides=(stride_m_q, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    t_block_ptr = tl.make_block_ptr(
        base=t_ptr + pid_batch * stride_tb,
        shape=(N_QUERIES, 1), strides=(stride_tq, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + pid_batch * stride_bb,
        shape=(N_QUERIES, 1), strides=(stride_bq, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    grad_o_block_ptr = tl.make_block_ptr(
        base=grad_o_ptr + pid_batch * stride_gob,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_goq, stride_god),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    grad_k_block_ptr = tl.make_block_ptr(
        base=grad_k_ptr + pid_batch * stride_gkb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_gkk, stride_gkd),
        offsets=(col_offset, 0), block_shape=(COL_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )
    grad_v_block_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_batch * stride_gvb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_gvk, stride_gvd),
        offsets=(col_offset, 0), block_shape=(COL_TILE_SIZE, HEAD_DIM), order=(1, 0),
    )

    K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
    grad_k_acc = tl.zeros((COL_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    grad_v_acc = tl.zeros((COL_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    qk_scale_log2 = qk_scale * 1.44269504

    # Safe phase starts when min(row) > max(col) — i.e.
    # row_offset >= col_offset + COL_TILE_SIZE.
    first_safe_row_block = tl.cdiv(col_offset + COL_TILE_SIZE, ROW_TILE_SIZE)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)

    # Phase A: causal-border row blocks. Apply causal mask plus window mask
    # when SWA is on.
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    for row_block_id in range(start_row_block, causal_end):
        row_offset = row_block_id * ROW_TILE_SIZE
        row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
        row_mask = row_indices[:, None] < N_QUERIES
        Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
        d1 = tl.load(d1_block_ptr, boundary_check=(0, 1), padding_option="zero")
        bart = tl.load(bart_block_ptr, boundary_check=(0, 1), padding_option="zero")
        m = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option="zero")
        t = tl.load(t_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / d1, 0.0)
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
        qk = tl.where(mask, qk, -float("inf"))
        w = tl.math.exp2(qk - m)
        p = w * inv_d1
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        delta = a - b
        bart_minus_rk = bart - rk
        gl = p * (a - t + bart_minus_rk * delta) * qk_scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(tl.bfloat16), Q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(tl.bfloat16), R, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(tl.bfloat16), grad_o, out_dtype=tl.float32, acc=grad_v_acc)

        q_block_ptr = tl.advance(q_block_ptr, (ROW_TILE_SIZE, 0))
        r_block_ptr = tl.advance(r_block_ptr, (ROW_TILE_SIZE, 0))
        d1_block_ptr = tl.advance(d1_block_ptr, (ROW_TILE_SIZE, 0))
        bart_block_ptr = tl.advance(bart_block_ptr, (ROW_TILE_SIZE, 0))
        m_block_ptr = tl.advance(m_block_ptr, (ROW_TILE_SIZE, 0))
        t_block_ptr = tl.advance(t_block_ptr, (ROW_TILE_SIZE, 0))
        b_block_ptr = tl.advance(b_block_ptr, (ROW_TILE_SIZE, 0))
        grad_o_block_ptr = tl.advance(grad_o_block_ptr, (ROW_TILE_SIZE, 0))

    # Phase B: safe row blocks (no causal/col/window mask).
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
        row_offset = row_block_id * ROW_TILE_SIZE
        row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
        row_mask = row_indices[:, None] < N_QUERIES
        Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
        d1 = tl.load(d1_block_ptr, boundary_check=(0, 1), padding_option="zero")
        bart = tl.load(bart_block_ptr, boundary_check=(0, 1), padding_option="zero")
        m = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option="zero")
        t = tl.load(t_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / d1, 0.0)
        w = tl.math.exp2(qk - m)
        p = w * inv_d1
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        delta = a - b
        bart_minus_rk = bart - rk
        gl = p * (a - t + bart_minus_rk * delta) * qk_scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(tl.bfloat16), Q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(tl.bfloat16), R, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(tl.bfloat16), grad_o, out_dtype=tl.float32, acc=grad_v_acc)

        q_block_ptr = tl.advance(q_block_ptr, (ROW_TILE_SIZE, 0))
        r_block_ptr = tl.advance(r_block_ptr, (ROW_TILE_SIZE, 0))
        d1_block_ptr = tl.advance(d1_block_ptr, (ROW_TILE_SIZE, 0))
        bart_block_ptr = tl.advance(bart_block_ptr, (ROW_TILE_SIZE, 0))
        m_block_ptr = tl.advance(m_block_ptr, (ROW_TILE_SIZE, 0))
        t_block_ptr = tl.advance(t_block_ptr, (ROW_TILE_SIZE, 0))
        b_block_ptr = tl.advance(b_block_ptr, (ROW_TILE_SIZE, 0))
        grad_o_block_ptr = tl.advance(grad_o_block_ptr, (ROW_TILE_SIZE, 0))

    # Phase C: window-border row blocks (SWA only). Rows past the causal
    # diagonal but straddling the right window edge.
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)
    for row_block_id in range(window_border_start, num_row_blocks):
        row_offset = row_block_id * ROW_TILE_SIZE
        row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
        row_mask = row_indices[:, None] < N_QUERIES
        Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
        d1 = tl.load(d1_block_ptr, boundary_check=(0, 1), padding_option="zero")
        bart = tl.load(bart_block_ptr, boundary_check=(0, 1), padding_option="zero")
        m = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option="zero")
        t = tl.load(t_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / d1, 0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < N_KEYVALS)
        )
        qk = tl.where(mask, qk, -float("inf"))
        w = tl.math.exp2(qk - m)
        p = w * inv_d1
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        delta = a - b
        bart_minus_rk = bart - rk
        gl = p * (a - t + bart_minus_rk * delta) * qk_scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(tl.bfloat16), Q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(tl.bfloat16), R, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(tl.bfloat16), grad_o, out_dtype=tl.float32, acc=grad_v_acc)

        q_block_ptr = tl.advance(q_block_ptr, (ROW_TILE_SIZE, 0))
        r_block_ptr = tl.advance(r_block_ptr, (ROW_TILE_SIZE, 0))
        d1_block_ptr = tl.advance(d1_block_ptr, (ROW_TILE_SIZE, 0))
        bart_block_ptr = tl.advance(bart_block_ptr, (ROW_TILE_SIZE, 0))
        m_block_ptr = tl.advance(m_block_ptr, (ROW_TILE_SIZE, 0))
        t_block_ptr = tl.advance(t_block_ptr, (ROW_TILE_SIZE, 0))
        b_block_ptr = tl.advance(b_block_ptr, (ROW_TILE_SIZE, 0))
        grad_o_block_ptr = tl.advance(grad_o_block_ptr, (ROW_TILE_SIZE, 0))

    tl.store(grad_k_block_ptr, grad_k_acc.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(grad_v_block_ptr, grad_v_acc.to(tl.bfloat16), boundary_check=(0, 1))


def parallax_bwd(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    barv: torch.Tensor,
    d1: torch.Tensor,
    bart: torch.Tensor,
    m: torch.Tensor,
    grad_o: torch.Tensor,
    qk_scale: float | torch.Tensor,
    n_rep: int = 1,
    window_size_left: int = -1,
):
    """Parallax backward pass (Triton, training).

    Args:
        q, r, k, v: the same tensors passed to :func:`parallax_fwd`.
        o, barv, d1, bart, m: tensors returned by :func:`parallax_fwd`.
        grad_o: gradient of the loss w.r.t. ``o``.
        qk_scale: matches the value passed to :func:`parallax_fwd`.
        n_rep: GQA group size (``H_q // H_kv``). Default 1 = MHA.
        window_size_left: causal sliding-window length (FA2 convention).

    Returns:
        ``(grad_q, grad_r, grad_k, grad_v)`` — bf16 tensors with the same
        shapes as ``q, r, k, v``. Under GQA the per-q-head dK/dV slots are
        folded back to the kv-head axis with a sum reduce.
    """
    batch_size, n_queries, head_dim = q.shape
    n_keyvals = k.shape[1]
    assert batch_size % n_rep == 0
    kv_batch_size = batch_size // n_rep
    assert k.shape[0] == kv_batch_size

    grad_q = torch.empty_like(q, dtype=torch.bfloat16)
    grad_r = torch.empty_like(r, dtype=torch.bfloat16)
    grad_k_buf = torch.empty(
        (batch_size, n_keyvals, head_dim), device=q.device, dtype=torch.bfloat16
    )
    grad_v_buf = torch.empty(
        (batch_size, n_keyvals, head_dim), device=q.device, dtype=torch.bfloat16
    )

    t = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    b = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    pre_grid = lambda META: (math.ceil(n_queries / META["ROW_TILE_SIZE"]), batch_size)
    _bwd_preprocess_kernel[pre_grid](
        grad_o, o, barv, t, b,
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        barv.stride(0), barv.stride(1), barv.stride(2),
        t.stride(0), t.stride(1),
        b.stride(0), b.stride(1),
        n_queries,
        head_dim,
    )

    rq_grid = lambda META: (math.ceil(n_queries / META["ROW_TILE_SIZE"]), batch_size)
    kv_grid = lambda META: (math.ceil(n_keyvals / META["COL_TILE_SIZE"]), batch_size)

    _bwd_rq_kernel[rq_grid](
        q, r, k, v, d1, bart, m, t, b, grad_o, grad_q, grad_r,
        q.stride(0), q.stride(1), q.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        d1.stride(0), d1.stride(1),
        bart.stride(0), bart.stride(1),
        m.stride(0), m.stride(1),
        t.stride(0), t.stride(1),
        b.stride(0), b.stride(1),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2),
        grad_r.stride(0), grad_r.stride(1), grad_r.stride(2),
        qk_scale,
        n_queries,
        n_keyvals,
        head_dim,
        n_rep,
        window_size_left,
    )

    _bwd_kv_kernel[kv_grid](
        q, r, k, v, d1, bart, m, t, b, grad_o, grad_k_buf, grad_v_buf,
        q.stride(0), q.stride(1), q.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        d1.stride(0), d1.stride(1),
        bart.stride(0), bart.stride(1),
        m.stride(0), m.stride(1),
        t.stride(0), t.stride(1),
        b.stride(0), b.stride(1),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        grad_k_buf.stride(0), grad_k_buf.stride(1), grad_k_buf.stride(2),
        grad_v_buf.stride(0), grad_v_buf.stride(1), grad_v_buf.stride(2),
        qk_scale,
        n_queries,
        n_keyvals,
        head_dim,
        n_rep,
        window_size_left,
    )

    if n_rep == 1:
        grad_k = grad_k_buf
        grad_v = grad_v_buf
    else:
        # Fold n_rep per-q-head slots back to the kv-head axis (same sum as
        # autograd through repeat_kv).
        grad_k = grad_k_buf.view(kv_batch_size, n_rep, n_keyvals, head_dim).sum(dim=1)
        grad_v = grad_v_buf.view(kv_batch_size, n_rep, n_keyvals, head_dim).sum(dim=1)
    return grad_q, grad_r, grad_k, grad_v

# --------------------------------------------------------------------------- #
# Autograd wrapper (canonical (B, H, L, D) entry point).
# --------------------------------------------------------------------------- #

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
