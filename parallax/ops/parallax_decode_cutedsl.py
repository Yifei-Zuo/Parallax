# Copyright (c) 2026 Zhichen Zeng.
# Reference: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd.py
# SPDX-License-Identifier: MIT
"""Parallax decoding kernel on NVIDIA Hopper (SM90).

Persistent split-K, warp-specialized streaming CuTeDSL kernel that
implements Algorithm 1 of the Parallax paper. The per-tile online
softmax, the per-tile composite-score state, and the cross-split
log-sum-exp merge all run inside a single kernel launch via an
atomic-last-CTA-wins finalize, with no separate reduction-epilogue
kernel.

Warp specialization (one producer warpgroup issuing TMA loads of
K_c / V_c tiles into shared memory, one consumer warpgroup driving
the QK and PV WGMMA pipeline plus the online softmax) follows the
FlashAttention 3 CuTeDSL kernel referenced above. Parallax extends
that streaming structure with one extra GEMM (R_r * K_c^T) and one
extra weighted PV accumulation (P_2 * V_c) per tile, both packed
into the same WGMMA pair that already produces FlashAttention's
S_1 = Q_r * K_c^T * s and O_1 = sum_j P_1_j * V_j. The two
branches share their K_c, V_c reads, the online-softmax max, and
the rescaling factor, so the covariance branch costs zero extra
HBM traffic and one extra row of register accumulators per CTA.

Variable names inside this file follow Algorithm 1 of the paper
directly (Q_r, R_r, K, V, S_1, S_2, P_1, P_2, m_r, alpha, d_1,
d_2, O_1, O_2, B_c, s); a short variable mapping table is provided
below.

Public entry point: ``parallax_decode(q, r, k, v, qk_scale, out=None)``.

Restrictions:
  * SM90 (H100 / H200) only
  * bf16 or fp16 input
  * seqlen_q = 1 (decoding only)
  * head_dim in {64, 128}
  * kv_len must be a multiple of B_c = 64
"""

from __future__ import annotations

import math
import operator
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import nvvm
from cutlass._mlir.dialects import llvm as _mlir_llvm

import cutlass.utils.hopper_helpers as sm90_utils_basic
from flash_attn.cute import hopper_helpers as sm90_utils
from flash_attn.cute import pipeline
from flash_attn.cute import utils


@cute.jit
def _atom_acq_rel_gpu_add_u32(counter_ptr: cute.Pointer) -> Int32:
    """`atom.acq_rel.gpu.global.add.u32` — returns OLD pre-increment value.

    Uses acq_rel ordering at GPU scope: this RMW is both a release of
    prior-program-order memory ops AND an acquire of any happens-before
    release by another GPU thread. That makes it the proper sync point
    for atomic-last-CTA-wins fan-in.
    """
    ptr_i64 = counter_ptr.toint().ir_value()
    res = _mlir_llvm.inline_asm(
        Int32.mlir_type,
        [ptr_i64],
        "atom.acq_rel.gpu.global.add.u32 $0, [$1], 1;",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,  # AT&T
    )
    return Int32(res)


@cute.jit
def _st_global_cg_f32(gmem_ptr: cute.Pointer, val: Float32) -> None:
    """`st.global.cg.f32 [ptr], val` — cache-global write, bypasses L1.

    Goes directly to L2 so peer CTAs reading via `ld.global.cg` see the
    write without depending on stale L1 lines being evicted.
    """
    ptr_i64 = gmem_ptr.toint().ir_value()
    _mlir_llvm.inline_asm(
        None,
        [ptr_i64, val.ir_value()],
        "st.global.cg.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )


@cute.jit
def _ld_global_cg_f32(gmem_ptr: cute.Pointer) -> Float32:
    """`ld.global.cv.f32 dst, [ptr]` — cache-volatile load: never cached,
    always reads from L2 (or memory). Strongest hint to avoid stale
    cached values. We use .cv on the reader side (not .cg) to make
    absolutely sure peer-CTA writes are not shadowed by an L1-cached
    line from a prior call."""
    ptr_i64 = gmem_ptr.toint().ir_value()
    res = _mlir_llvm.inline_asm(
        Float32.mlir_type,
        [ptr_i64],
        "ld.global.cv.f32 $0, [$1];",
        "=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )
    return Float32(res)


_compile_cache: dict[tuple, Callable] = {}
_output_cache: dict[tuple, tuple] = {}
_cute_input_cache: dict[tuple, tuple] = {}


class ParallaxDecodePersistentSplit:
    def __init__(self, dtype, head_dim: int, *, n_block_size: int = 64, debug_stage: int = 0):
        if head_dim > 128:
            raise ValueError("SM90 TMA/WGMMA prototype requires head_dim <= 128")
        if n_block_size != 64:
            raise ValueError("SM90 TMA/WGMMA prototype currently hard-codes N=64")
        self.dtype = dtype
        self.head_dim = head_dim
        self.head_dim_padded = int(math.ceil(head_dim / 16) * 16)
        self.m_block_size = 64
        self.n_block_size = n_block_size
        self.num_stages = 2  # was 3; reduced to 2 to close a producer-consumer
        # pipeline race that caused ~10% of calls on large-BH × mid-K × D=128
        # shapes to have ONE random (B, H) row computed with a stale K-tile
        # (max output diff ~0.2-0.8 vs output norm ~3-10). At num_stages=3
        # the producer can get 3 K/V tiles ahead of the consumer and the
        # `PipelineTmaAsyncNoCluster` mbarrier protocol intermittently flips
        # the stage barrier before TMA fully commits to SMEM. num_stages=2
        # drops the affected-shape count from 10/83 → 2/83 in the sweep and
        # has zero measurable speed cost on H200 (graph-replay kernel time
        # identical within ±0.3 µs).
        self.num_threads = 256
        self.num_threads_per_warp_group = 128
        self.debug_stage = debug_stage

    def _get_layouts(self):
        qk_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.dtype,
                self.head_dim_padded,
            ),
            self.dtype,
        )
        v_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.dtype,
                self.head_dim_padded,
            ),
            self.dtype,
        )
        sQ_layout = cute.tile_to_shape(qk_atom, (self.m_block_size, self.head_dim_padded), (0, 1))
        sK_layout = cute.tile_to_shape(qk_atom, (self.n_block_size, self.head_dim_padded, self.num_stages), (0, 1, 2))
        sV_layout = cute.tile_to_shape(v_atom, (self.n_block_size, self.head_dim_padded, self.num_stages), (0, 1, 2))
        return sQ_layout, sK_layout, sV_layout

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            cutlass.Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self.n_block_size),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            cutlass.Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self.head_dim_padded),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mR: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        mWs_counter: cute.Tensor,
        kv_len: cutlass.Constexpr[int],
        softmax_scale_log2: Float32,
        stream: cuda.CUstream,
        num_k_splits: cutlass.Constexpr[int] = 1,
    ):
        mK_tma, mV_tma = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0]))
            for t in (mK, mV)
        ]

        sQ_layout, sK_layout, sV_layout = self._get_layouts()
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()

        copy_atom_kv = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            copy_atom_kv,
            mK_tma,
            cute.select(sK_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_padded),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            copy_atom_kv,
            mV_tma,
            cute.select(sV_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_padded),
        )

        self.tma_copy_k_bytes = cute.size_in_bytes(mK.element_type, cute.select(sK_layout, mode=[0, 1]))
        self.tma_copy_v_bytes = cute.size_in_bytes(mV.element_type, cute.select(sV_layout, mode=[0, 1]))

        sQ_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)], 128]
        sK_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)], 128]
        sV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sV_layout)], 128]
        p_row_struct = cute.struct.Align[cute.struct.MemRange[Float32, 128], 128]
        stats_struct = cute.struct.Align[cute.struct.MemRange[Float32, 128], 128]
        mbar_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorage:
            mbar_ptr_K: mbar_struct
            mbar_ptr_V: mbar_struct
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct
            p_row: p_row_struct
            stats: stats_struct

        self.kernel(
            mQ,
            mR,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            mWs_m,
            mWs_d1,
            mWs_d2,
            mWs_O1,
            mWs_O2,
            mWs_counter,
            tma_atom_K,
            tma_atom_V,
            kv_len,
            softmax_scale_log2,
            sQ_layout,
            sK_layout,
            sV_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            num_k_splits,
        ).launch(
            grid=[cute.size(mQ.shape[0]), cute.size(mQ.shape[2]), num_k_splits],
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mR: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        mWs_counter: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        kv_len: cutlass.Constexpr[int],
        softmax_scale_log2: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr[Callable],
        num_k_splits: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        batch_idx, head_idx, k_split_id = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Per-split K-tile range. The K-tile loop iterates over
        # [k_start_tile, k_end_tile) instead of [0, tiles_total). At
        # num_k_splits == 1 this collapses to the full range, matching
        # the unsplit baseline. tiles_total and tiles_per_split are
        # constexpr (derived from kv_len, n_block_size, num_k_splits);
        # k_start_tile/k_end_tile are runtime values (depend on the
        # grid-Z block index).
        tiles_total: cutlass.Constexpr[int] = kv_len // self.n_block_size
        tiles_per_split: cutlass.Constexpr[int] = (tiles_total + num_k_splits - 1) // num_k_splits
        k_start_tile = k_split_id * tiles_per_split
        k_end_tile_uncapped = k_start_tile + tiles_per_split
        # Cap the last split at tiles_total when tiles_total is not a multiple of
        # tiles_per_split. cute.arch.min handles the runtime min.
        k_end_tile = cutlass.min(k_end_tile_uncapped, Int32(tiles_total))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP_row = storage.p_row.get_tensor(cute.make_layout(64))
        sStats = storage.stats.get_tensor(cute.make_layout(128))
        sVt = utils.transpose_view(sV)

        pipeline_group_producer = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_group_consumer = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
        pipeline_k = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_group_producer,
            consumer_group=pipeline_group_consumer,
            tx_count=self.tma_copy_k_bytes,
            init_wait=False,
        )
        pipeline_v = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_group_producer,
            consumer_group=pipeline_group_consumer,
            tx_count=self.tma_copy_v_bytes,
        )

        # Q/R fill + barrier moved inside the consumer branch so the producer warpgroup
        # can start TMA loads immediately instead of waiting on an all-block barrier.
        mK_cur = mK[None, None, head_idx, batch_idx]
        mV_cur = mV[None, None, head_idx, batch_idx]
        gK = cute.local_tile(mK_cur, (self.n_block_size, self.head_dim_padded), (None, 0))
        gV = cute.local_tile(mV_cur, (self.n_block_size, self.head_dim_padded), (None, 0))
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_V,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )

        if warp_idx < 4:
            cute.arch.warpgroup_reg_dealloc(24)
            producer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.num_stages)
            if warp_idx == 0:
                for n_tile in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                    self._load_tile(tma_atom_K, tKgK, tKsK, pipeline_k, n_tile, producer_state)
                    self._load_tile(tma_atom_V, tVgV, tVsV, pipeline_v, n_tile, producer_state)
                    producer_state.advance()
        else:
            cute.arch.warpgroup_reg_alloc(240)
            tidx_mma = tidx - self.num_threads_per_warp_group
            self._fill_qr_smem(mQ, mR, sQ, batch_idx, head_idx, tidx_mma)
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads_per_warp_group)
            self._mma_consumer(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mWs_m,
                mWs_d1,
                mWs_d2,
                mWs_O1,
                mWs_O2,
                mWs_counter,
                sQ,
                sK,
                sVt,
                sP_row,
                sStats,
                pipeline_k,
                pipeline_v,
                batch_idx,
                head_idx,
                k_split_id,
                tidx_mma,
                kv_len,
                softmax_scale_log2,
                num_k_splits,
                k_start_tile,
                k_end_tile,
            )

    @cute.jit
    def _fill_qr_smem(
        self,
        mQ: cute.Tensor,
        mR: cute.Tensor,
        sQ: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        tidx: Int32,
    ) -> None:
        if tidx < self.head_dim:
            sQ[0, tidx] = mQ[batch_idx, 0, head_idx, tidx]
            sQ[1, tidx] = mR[batch_idx, 0, head_idx, tidx]
        if tidx >= self.head_dim and tidx < self.head_dim_padded:
            sQ[0, tidx] = self.dtype(0.0)
            sQ[1, tidx] = self.dtype(0.0)
        # Rows 2..63 of sQ intentionally left uninitialized. The QK WGMMA writes acc_QR
        # rows 2..63 but nothing downstream reads them: the softmax only touches row 0,
        # _scale_output_rows01 touches rows 0/1 of acc_O, and the PV WGMMA's row-0/1
        # output depends only on row-0/1 of tOrP. Rows 2..63 of acc_O are discarded by
        # _finalize_and_store.

    @cute.jit
    def _load_tile(
        self,
        tma_atom: cute.CopyAtom,
        tG: cute.Tensor,
        tS: cute.Tensor,
        pipe: cutlass.pipeline.PipelineAsync,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
    ) -> None:
        pipe.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            tG[None, block],
            tS[None, producer_state.index],
            tma_bar_ptr=pipe.producer_get_barrier(producer_state),
        )

    @cute.jit
    def _mma_consumer(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        mWs_counter: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP_row: cute.Tensor,
        sStats: cute.Tensor,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        batch_idx: Int32,
        head_idx: Int32,
        k_split_id: Int32,
        tidx: Int32,
        kv_len: cutlass.Constexpr[int],
        softmax_scale_log2: Float32,
        num_k_splits: cutlass.Constexpr[int],
        k_start_tile: Int32,
        k_end_tile: Int32,
    ) -> None:
        wg_layout = cute.make_layout(1, stride=self.num_threads_per_warp_group)
        wg_mma_qk = tiled_mma_qk.get_slice(wg_layout(0))
        wg_mma_pv = tiled_mma_pv.get_slice(wg_layout(0))
        tSrQ = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sK))
        tOrVt = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sVt))
        acc_S_shape = tiled_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size))
        acc_O_shape = tiled_mma_pv.partition_shape_C((self.m_block_size, self.head_dim_padded))
        tOrP = cute.make_fragment(utils.convert_layout_acc_frgA(cute.make_layout(acc_S_shape)), self.dtype)
        acc_O = cute.make_fragment(acc_O_shape, Float32)

        m_r = -Float32.inf
        d1 = Float32(0.0)
        d2 = Float32(0.0)
        consumer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.num_stages)

        if const_expr(self.debug_stage == 1):
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                pipeline_k.consumer_release(consumer_state)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(1.0)
            return

        if const_expr(self.debug_stage == 11):
            if tidx == 0:
                sP_row[0] = Float32(0.0)
                sP_row[1] = Float32(0.0)
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads_per_warp_group)
            acc_QR = cute.make_fragment(acc_S_shape, Float32)
            acc_mn = utils.make_acc_tensor_mn_view(acc_QR)
            thr_mma = tiled_mma_qk.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
            cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
            cS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
            for r_i in cutlass.range(cute.size(acc_mn, mode=[0]), unroll_full=True):
                for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
                    if cS_mn[r_i, c_i][0] == 0:
                        sP_row[0] = Float32(1.0)
                    if cS_mn[r_i, c_i][0] == 1:
                        sP_row[1] = Float32(1.0)
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads_per_warp_group)
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                pipeline_k.consumer_release(consumer_state)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = sP_row[0] + Float32(2.0) * sP_row[1]
            return

        if const_expr(self.debug_stage == 17):
            acc_QR = cute.make_fragment(acc_S_shape, Float32)
            acc_mn = utils.make_acc_tensor_mn_view(acc_QR)
            thr_mma = tiled_mma_qk.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
            cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
            cS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
            for r_i in cutlass.range(cute.size(acc_mn, mode=[0]), unroll_full=True):
                for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
                    if cS_mn[r_i, c_i][0] == 0 and cS_mn[r_i, c_i][1] < 64:
                        mO[batch_idx, 0, head_idx, cS_mn[r_i, c_i][1]] = (r_i * 100 + c_i).to(Float32)
                    if cS_mn[r_i, c_i][0] == 1 and cS_mn[r_i, c_i][1] < 64:
                        mO[batch_idx, 0, head_idx, cS_mn[r_i, c_i][1] + 64] = (r_i * 100 + c_i).to(Float32)
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                pipeline_k.consumer_release(consumer_state)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            return

        if const_expr(self.debug_stage == 2):
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(2.0)
            return

        if const_expr(self.debug_stage == 3):
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                m_r, d1, d2, alpha = self._row0_online_softmax_and_make_p(
                    acc_QR,
                    tiled_mma_qk,
                    sP_row,
                    sStats,
                    m_r,
                    d1,
                    d2,
                    softmax_scale_log2,
                )
                tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
                utils.cvt_f16(tOrP_acc, tOrP)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(3.0)
            return

        if const_expr(self.debug_stage == 4):
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                m_r, d1, d2, alpha = self._row0_online_softmax_and_make_p(
                    acc_QR,
                    tiled_mma_qk,
                    sP_row,
                    sStats,
                    m_r,
                    d1,
                    d2,
                    softmax_scale_log2,
                )
                self._scale_output_rows01(acc_O, alpha, tiled_mma_pv)
                tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
                utils.cvt_f16(tOrP_acc, tOrP)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=0,
                )
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(4.0)
            return

        if const_expr(self.debug_stage == 5):
            tOrP.fill(0.0)
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                pipeline_k.consumer_release(consumer_state)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=0,
                )
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(5.0)
            return

        if const_expr(self.debug_stage == 6):
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                tOrP.fill(0.0)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=0,
                )
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(6.0)
            return

        if const_expr(self.debug_stage == 7):
            tOrP.fill(1.0)
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                pipeline_k.consumer_release(consumer_state)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=0,
                )
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(7.0)
            return

        if const_expr(self.debug_stage == 8):
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                m_r, d1, d2, alpha = self._row0_online_softmax_and_make_p(
                    acc_QR,
                    tiled_mma_qk,
                    sP_row,
                    sStats,
                    m_r,
                    d1,
                    d2,
                    softmax_scale_log2,
                )
                self._scale_output_rows01(acc_O, alpha, tiled_mma_pv)
                tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
                for i in cutlass.range_constexpr(cute.size(tOrP)):
                    tOrP[i] = tOrP_acc[i].to(self.dtype)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=0,
                )
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(8.0)
            return

        if const_expr(self.debug_stage == 9):
            acc_QR = cute.make_fragment(acc_S_shape, Float32)
            acc_QR.fill(0.0)
            acc_mn = utils.make_acc_tensor_mn_view(acc_QR)
            thr_mma = tiled_mma_qk.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
            cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
            cS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
            for r_i in cutlass.range(cute.size(acc_mn, mode=[0]), unroll_full=True):
                for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
                    if cS_mn[r_i, c_i][0] == 0:
                        acc_mn[r_i, c_i] = 1.0
            tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
            for i in cutlass.range_constexpr(cute.size(tOrP)):
                tOrP[i] = tOrP_acc[i].to(self.dtype)
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                pipeline_k.consumer_release(consumer_state)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=0,
                )
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(9.0)
            return

        if const_expr(self.debug_stage == 10):
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                acc_QR.fill(0.0)
                acc_mn = utils.make_acc_tensor_mn_view(acc_QR)
                thr_mma = tiled_mma_qk.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
                cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
                cS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
                for r_i in cutlass.range(cute.size(acc_mn, mode=[0]), unroll_full=True):
                    for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
                        if cS_mn[r_i, c_i][0] == 0:
                            acc_mn[r_i, c_i] = 1.0
                tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
                for i in cutlass.range_constexpr(cute.size(tOrP)):
                    tOrP[i] = tOrP_acc[i].to(self.dtype)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=0,
                )
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = Float32(10.0)
            return

        if const_expr(self.debug_stage == 12):
            for n_tile in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                m_r, d1, d2, alpha = self._row0_online_softmax_and_make_p(
                    acc_QR,
                    tiled_mma_qk,
                    sP_row,
                    sStats,
                    m_r,
                    d1,
                    d2,
                    softmax_scale_log2,
                )
                self._scale_output_rows01(acc_O, alpha, tiled_mma_pv)
                tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
                utils.cvt_f16(tOrP_acc, tOrP)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt[None, None, None, consumer_state.index],
                    zero_init=n_tile == 0,
                    wg_wait=0,
                )
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = d2 * cute.arch.rcp_approx(d1)
            return

        if const_expr(self.debug_stage == 13):
            rk_sum = Float32(0.0)
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                acc_mn = utils.make_acc_tensor_mn_view(acc_QR)
                thr_mma = tiled_mma_qk.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
                cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
                cS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
                for r_i in cutlass.range(cute.size(acc_mn, mode=[0]), unroll_full=True):
                    for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
                        if cS_mn[r_i, c_i][0] == 1:
                            rk_sum += acc_mn[r_i, c_i]
                rk_sum = utils.warp_reduce(rk_sum, operator.add, width=4)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = rk_sum / Float32(64.0)
            return

        if const_expr(self.debug_stage == 14):
            all_sum = Float32(0.0)
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                acc_mn = utils.make_acc_tensor_mn_view(acc_QR)
                for r_i in cutlass.range(cute.size(acc_mn, mode=[0]), unroll_full=True):
                    for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
                        all_sum += acc_mn[r_i, c_i]
                all_sum = utils.warp_reduce(all_sum, operator.add, width=4)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = all_sum
            return

        if const_expr(self.debug_stage == 16):
            for n_tile in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                acc_QR = cute.make_fragment(acc_S_shape, Float32)
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_QR,
                    tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(consumer_state)
                m_r, d1, d2, alpha = self._row0_online_softmax_and_make_p(
                    acc_QR,
                    tiled_mma_qk,
                    sP_row,
                    sStats,
                    m_r,
                    d1,
                    d2,
                    softmax_scale_log2,
                )
                self._scale_output_rows01(acc_O, alpha, tiled_mma_pv)
                tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
                utils.cvt_f16(tOrP_acc, tOrP)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt[None, None, None, consumer_state.index],
                    zero_init=n_tile == 0,
                    wg_wait=0,
                )
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            acc_mn = utils.make_acc_tensor_mn_view(acc_O)
            thr_mma = tiled_mma_pv.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
            cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
            cO_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
            row0_dim0 = Float32(0.0)
            row1_dim0 = Float32(0.0)
            all_sum = Float32(0.0)
            for r_i in cutlass.range(cute.size(acc_mn, mode=[0]), unroll_full=True):
                for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
                    all_sum += acc_mn[r_i, c_i]
                    if cO_mn[r_i, c_i][0] == 0 and cO_mn[r_i, c_i][1] == 0:
                        row0_dim0 = acc_mn[r_i, c_i]
                    if cO_mn[r_i, c_i][0] == 1 and cO_mn[r_i, c_i][1] == 0:
                        row1_dim0 = acc_mn[r_i, c_i]
            sStats[tidx] = all_sum
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads_per_warp_group)
            if tidx == 0:
                total = Float32(0.0)
                for i in cutlass.range(128, unroll=1):
                    total += sStats[i]
                mO[batch_idx, 0, head_idx, 0] = row0_dim0 * cute.arch.rcp_approx(d1)
                mO[batch_idx, 0, head_idx, 1] = row1_dim0 * cute.arch.rcp_approx(d1)
                mO[batch_idx, 0, head_idx, 2] = total * cute.arch.rcp_approx(d1)
            return

        if const_expr(self.debug_stage == 15):
            q_sum = Float32(0.0)
            r_sum = Float32(0.0)
            lane = tidx % 32
            d0 = lane
            d1 = lane + 32
            if d0 < self.head_dim:
                q_sum += sQ[0, d0].to(Float32)
                r_sum += sQ[1, d0].to(Float32)
            if d1 < self.head_dim:
                q_sum += sQ[0, d1].to(Float32)
                r_sum += sQ[1, d1].to(Float32)
            q_sum = utils.warp_reduce(q_sum, operator.add, width=32)
            r_sum = utils.warp_reduce(r_sum, operator.add, width=32)
            for _ in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
                pipeline_k.consumer_release(consumer_state)
                pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
                pipeline_v.consumer_release(consumer_state)
                consumer_state.advance()
            if tidx == 0:
                mO[batch_idx, 0, head_idx, 0] = q_sum
                mO[batch_idx, 0, head_idx, 1] = r_sum
            return

        acc_QR = cute.make_fragment(acc_S_shape, Float32)
        pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
        sm90_utils.gemm(
            tiled_mma_qk,
            acc_QR,
            tSrQ,
            tSrK[None, None, None, consumer_state.index],
            zero_init=True,
            wg_wait=-1,
        )
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(consumer_state)

        m_r, d1, d2, _alpha0 = self._row0_online_softmax_and_make_p(
            acc_QR,
            tiled_mma_qk,
            sP_row,
            sStats,
            m_r,
            d1,
            d2,
            softmax_scale_log2,
        )
        tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
        utils.cvt_f16(tOrP_acc, tOrP)

        pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
        sm90_utils.gemm(
            tiled_mma_pv,
            acc_O,
            tOrP,
            tOrVt[None, None, None, consumer_state.index],
            zero_init=True,
            wg_wait=-1,
        )
        # Track V slot to release after the next PV sync (deferred release so PV_n overlaps
        # with QK_{n+1} in the WGMMA pipeline).
        consumer_state_v = consumer_state.clone()
        consumer_state.advance()

        for _ in cutlass.range(k_start_tile, k_end_tile - 1, unroll=1):
            acc_QR = cute.make_fragment(acc_S_shape, Float32)
            pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
            sm90_utils.gemm(
                tiled_mma_qk,
                acc_QR,
                tSrQ,
                tSrK[None, None, None, consumer_state.index],
                zero_init=True,
                wg_wait=-1,
            )
            # Waits for both the current QK WGMMA and the prior iteration's PV WGMMA.
            warpgroup.wait_group(0)
            pipeline_k.consumer_release(consumer_state)
            pipeline_v.consumer_release(consumer_state_v)

            m_r, d1, d2, alpha = self._row0_online_softmax_and_make_p(
                acc_QR,
                tiled_mma_qk,
                sP_row,
                sStats,
                m_r,
                d1,
                d2,
                softmax_scale_log2,
            )
            self._scale_output_rows01(acc_O, alpha, tiled_mma_pv)
            tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
            utils.cvt_f16(tOrP_acc, tOrP)

            pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
            sm90_utils.gemm(
                tiled_mma_pv,
                acc_O,
                tOrP,
                tOrVt[None, None, None, consumer_state.index],
                zero_init=False,
                wg_wait=-1,
            )
            consumer_state_v.advance()
            consumer_state.advance()

        warpgroup.wait_group(0)
        pipeline_v.consumer_release(consumer_state_v)

        # Finalize: two paths, picked at JIT time on num_k_splits.
        #
        # **num_k_splits == 1 (short-K / SM-saturated grid)**: this CTA owns
        # the entire (B, H) row by itself, so it just casts in-register and
        # writes the bf16/fp16 row to mO directly via `_finalize_and_store`.
        # No HBM workspace round-trip, no fence, no atomic, no merge. This
        # is the same code path v1 (clla-cute) uses and matches its short-K
        # latency. (~1-2 µs saved per call.)
        #
        # **num_k_splits > 1 (split-K / narrow-batch long-K)**: every CTA
        # writes per-CTA un-normalized fp32 partials to HBM workspace,
        # then atomic-last-CTA-wins picks the merger:
        #   1. consumer-warpgroup-barrier (all 128 threads finished writes)
        #   2. fence_acq_rel_gpu — publish partial writes to peer CTAs
        #   3. tidx==0 atomic_add(counter[B, H], 1) returns OLD; the CTA
        #      that observes OLD == num_k_splits - 1 is the last arriver.
        #      Broadcast that via sStats[0].
        #   4. consumer-warpgroup-barrier (every thread sees the broadcast)
        #   5. last CTA only: 128 threads each handle one output column,
        #      read partials from HBM, run LSE merge in fp32, cast + store
        #      to mO. Then reset counter[B, H] = 0 for the next call.
        if const_expr(num_k_splits == 1):
            self._finalize_and_store(
                acc_O,
                d2,
                d1,
                tiled_mma_pv,
                mO,
                sP_row,
                batch_idx,
                head_idx,
            )
        else:
            self._store_split_partials(
                acc_O,
                d2,
                m_r,
                d1,
                softmax_scale_log2,
                tiled_mma_pv,
                mWs_m,
                mWs_d1,
                mWs_d2,
                mWs_O1,
                mWs_O2,
                batch_idx,
                head_idx,
                k_split_id,
            )
            self._fused_epilogue_atomic_last_wins(
                mO,
                mWs_m,
                mWs_d1,
                mWs_d2,
                mWs_O1,
                mWs_O2,
                mWs_counter,
                sStats,
                batch_idx,
                head_idx,
                tidx,
                num_k_splits,
            )

    @cute.jit
    def _row0_online_softmax_and_make_p(
        self,
        acc_qr: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        sP_row: cute.Tensor,
        sStats: cute.Tensor,
        m_r: Float32,
        d1: Float32,
        d2: Float32,
        softmax_scale_log2: Float32,
    ) -> tuple[Float32, Float32, Float32, Float32]:
        acc_mn = utils.make_acc_tensor_mn_view(acc_qr)
        thr_mma = tiled_mma_qk.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        tidx = cute.arch.thread_idx()[0] - self.num_threads_per_warp_group
        lane = tidx % 32
        warp = tidx // 32

        alpha = Float32(1.0)
        m_r_new = m_r
        d1_new = d1
        d2_new = d2
        m_cur = m_r
        # sm90 wgmma m64n64k16 fp32 C layout: each thread holds 2 rows stored at r_i=0 and
        # r_i=1; the r_i=1 slot corresponds to this-thread-row + 8, so rows 0 and 1 NEVER
        # appear at r_i=1 (they only appear at r_i=0 for lanes 0..7 of warp 0). Iterate only
        # r_i=0 to cut the inner-loop shuffle_sync count by half.
        r0 = 0
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            if cS_mn[r0, c_i][0] == 0:
                qk_val = acc_mn[r0, c_i]
                m_cur = qk_val if qk_val > m_cur else m_cur
        # Only lanes 0..7 of warp 0 hold useful row-0/row-1 state; width=8 produces the
        # correct reduced max on those lanes (lanes 8..31 retain stale values which are
        # ignored downstream by _finalize_and_store).
        m_cur = utils.warp_reduce(m_cur, cute.arch.fmax, width=8)
        m_r_safe = Float32(0.0) if m_cur == -Float32.inf else m_cur
        m_r_new = m_r_safe
        alpha = utils.exp2f((m_r - m_r_safe) * softmax_scale_log2)

        tile_sum = Float32(0.0)
        tile_c = Float32(0.0)
        # Hoist the constant factor out of the exp2f argument.
        scaled_max = m_r_safe * softmax_scale_log2
        # src_lane is loop-invariant per lane.
        src_lane = lane - 4 if lane >= 4 else lane
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            qk_exp = Float32(0.0)
            if cS_mn[r0, c_i][0] == 0:
                qk_exp = utils.exp2f(acc_mn[r0, c_i] * softmax_scale_log2 - scaled_max)
                tile_sum += qk_exp
                acc_mn[r0, c_i] = qk_exp
            qk_exp_from_row0 = utils.shuffle_sync(qk_exp, src_lane, width=32)
            if cS_mn[r0, c_i][0] == 1:
                wr = qk_exp_from_row0 * acc_mn[r0, c_i]
                tile_c += wr
                acc_mn[r0, c_i] = wr
        tile_sum = utils.warp_reduce(tile_sum, operator.add, width=8)
        d1_new = d1 * alpha + tile_sum
        tile_c = utils.warp_reduce(tile_c, operator.add, width=8)
        d2_new = d2 * alpha + tile_c

        return m_r_new, d1_new, d2_new, alpha

    @cute.jit
    def _scale_output_rows01(self, acc: cute.Tensor, alpha: Float32, tiled_mma: cute.TiledMma) -> None:
        acc_mn = utils.make_acc_tensor_mn_view(acc)
        thr_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        cO_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
        # sm90 wgmma m64 fp32 C layout: rows 0 and 1 appear only at r_i=0 (for lanes
        # 0..3 and 4..7 of warp 0 respectively); r_i=1 holds thread-row+8 (rows 8..63
        # which we never keep). Iterate r_i=0 only to halve the scan.
        r0 = 0
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            is_keep_row = cO_mn[r0, c_i][0] == 0 or cO_mn[r0, c_i][0] == 1
            if is_keep_row:
                acc_mn[r0, c_i] = acc_mn[r0, c_i] * alpha

    @cute.jit
    def _store_split_partials(
        self,
        acc_o: cute.Tensor,
        d2: Float32,
        m_r: Float32,
        d1: Float32,
        softmax_scale_log2: Float32,
        tiled_mma: cute.TiledMma,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        k_split_id: Int32,
    ) -> None:
        # Workspace tensors are shaped (B, H, num_k_splits) for scalars and
        # (B, H, num_k_splits, head_dim) for vector partials, so the kernel
        # can index them with (batch_idx, head_idx, k_split_id[, d]) directly.
        # The underlying memory layout is identical to the dispatcher's
        # (num_bh, num_k_splits[, head_dim]) flat allocation -- the .view()
        # is purely a CuTe-side reshape for clean indexing.
        acc_mn = utils.make_acc_tensor_mn_view(acc_o)
        thr_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        cO_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
        tidx = cute.arch.thread_idx()[0] - self.num_threads_per_warp_group
        # Scalar partials: one writer per CTA (lane 0 of warp 0 holds the
        # canonical reduced value because width=8 warp_reduce broadcasts
        # within the first 8 lanes of warp 0 -- lane 0 is included).
        # The kernel carries m_r in raw QK units (unscaled). The
        # Triton epilogue uses tl.exp for the log-sum-exp weight, so we
        # rescale m_r into "natural-base" units (matches the PyTorch
        # reference's m = (qk * qk_scale).max()). With qk_scale =
        # softmax_scale_log2 * ln(2), the conversion is m_r * scale_log2
        # * ln2.
        _LN2 = Float32(0.6931471805599453)
        # All partials writes go via `st.global.cg.f32` (cache-global) so
        # the data bypasses L1 and lands in L2 — peer CTAs reading via
        # `ld.global.cg` will see the latest write without depending on
        # stale L1 lines being evicted.
        if tidx == 0:
            _st_global_cg_f32(utils.elem_pointer(mWs_m,     (batch_idx, head_idx, k_split_id)), m_r * softmax_scale_log2 * _LN2)
            _st_global_cg_f32(utils.elem_pointer(mWs_d1,     (batch_idx, head_idx, k_split_id)), d1)
            _st_global_cg_f32(utils.elem_pointer(mWs_d2, (batch_idx, head_idx, k_split_id)), d2)
        # Vector partials: row 0 of acc_o is O_1 = Sum_j P_1_j * v_j,
        # row 1 of acc_o is O_2 = Sum_j P_2_j * v_j.
        r0 = 0
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            if cO_mn[r0, c_i][1] < self.head_dim:
                if cO_mn[r0, c_i][0] == 0:
                    _st_global_cg_f32(utils.elem_pointer(mWs_O1,  (batch_idx, head_idx, k_split_id, cO_mn[r0, c_i][1])), acc_mn[r0, c_i])
                if cO_mn[r0, c_i][0] == 1:
                    _st_global_cg_f32(utils.elem_pointer(mWs_O2, (batch_idx, head_idx, k_split_id, cO_mn[r0, c_i][1])), acc_mn[r0, c_i])

    @cute.jit
    def _finalize_and_store(
        self,
        acc_o: cute.Tensor,
        d2: Float32,
        denom: Float32,
        tiled_mma: cute.TiledMma,
        mO: cute.Tensor,
        sP_row: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
    ) -> None:
        acc_mn = utils.make_acc_tensor_mn_view(acc_o)
        thr_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        cO_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
        tidx = cute.arch.thread_idx()[0] - self.num_threads_per_warp_group
        lane = tidx % 32
        inv_d1 = cute.arch.rcp_approx(denom)
        c_norm = d2 * inv_d1
        # sm90 wgmma m64 fp32 C layout: rows 0/1 live only at r_i=0. The r_i=1 slot
        # holds thread-row+8 (rows 8..63) which we discard. Iterate r_i=0 only.
        r0 = 0
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            O2_local = Float32(0.0)
            if cO_mn[r0, c_i][0] == 1 and cO_mn[r0, c_i][1] < self.head_dim:
                O2_local = acc_mn[r0, c_i]
            src_lane = lane + 4 if lane < 4 else lane
            O2_from_row1 = utils.shuffle_sync(O2_local, src_lane, width=32)
            if cO_mn[r0, c_i][0] == 0 and cO_mn[r0, c_i][1] < self.head_dim:
                O1_d = acc_mn[r0, c_i] * inv_d1
                O2_d = O2_from_row1 * inv_d1
                o_fp32 = O1_d + c_norm * O1_d - O2_d
                # Cast to mO's element type (bf16/fp16 from the caller, or
                # fp32 if mO is the legacy internal buffer). Doing the
                # cancellation in fp32 and casting at the store preserves
                # the precision lever — same pattern as the fused-merge
                # path's `_merge_and_store_inkernel`.
                mO[batch_idx, 0, head_idx, cO_mn[r0, c_i][1]] = mO.element_type(o_fp32)

    @cute.jit
    def _fused_epilogue_atomic_last_wins(
        self,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        mWs_counter: cute.Tensor,
        sStats: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        tidx: Int32,
        num_k_splits: cutlass.Constexpr[int],
    ) -> None:
        # Runs on the consumer warpgroup only (128 threads, tidx in [0, 128)).
        # Pre-condition: _store_split_partials just wrote this CTA's partials
        # to the HBM workspace.
        #
        # Protocol:
        #   1. Consumer-wg barrier to ensure all 128 threads finished their
        #      per-CTA partial writes.
        #   2. fence_acq_rel_gpu so other CTAs observe our partials before
        #      our atomic-inc.
        #   3. tidx==0 atomic-adds 1 to counter[batch, head] (i32 in HBM),
        #      reads back OLD value. If OLD == num_k_splits-1 we're the
        #      last arriver for this (B, H). Broadcast that decision via
        #      sStats[0] (1.0=last, 0.0=not).
        #   4. Consumer-wg barrier so all threads see the broadcast.
        #   5. If last: every thread does the LSE-style merge in fp32 (each
        #      thread handles one output column when tidx < head_dim), then
        #      writes mO[batch, 0, head, :] and resets the counter so the
        #      next call starts from 0.
        # Release-side fence: every consumer thread publishes its partial
        # writes (mWs_O1/Rv, plus tidx==0's scalars) so peer CTAs that
        # later acquire from our atomic-inc observe those writes. The
        # barrier first ensures every thread has issued its store; the
        # fence then makes those stores GPU-visible.
        cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads_per_warp_group)
        cute.arch.fence_acq_rel_gpu()
        if tidx == 0:
            counter_ptr = utils.elem_pointer(mWs_counter, (batch_idx, head_idx))
            # Inline PTX `atom.acq_rel.gpu.global.add.u32` — proper acq_rel
            # ordering. The previous nvvm.atomicrmw without mem_order
            # defaulted to LLVM "monotonic" (relaxed) and did not establish
            # a sync-with relationship with the prior fence, so peer CTAs
            # could observe the increment without observing the partials.
            old = _atom_acq_rel_gpu_add_u32(counter_ptr)
            is_last = old == Int32(num_k_splits - 1)
            sStats[0] = Float32(1.0) if is_last else Float32(0.0)
        cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads_per_warp_group)
        is_last_cta = sStats[0] > Float32(0.5)
        if is_last_cta:
            # Acquire-side fence: every consumer thread is about to read
            # mWs_O1/Rv[:, :, s, tidx] for s in [0, num_k_splits) and
            # needs its own acquire to see the prior CTAs' released writes.
            # The CTA-level barrier above propagates sStats[0] but not
            # global-memory acquire semantics for arbitrary peer-CTA stores.
            cute.arch.fence_acq_rel_gpu()
            self._merge_and_store_inkernel(
                mO,
                mWs_m,
                mWs_d1,
                mWs_d2,
                mWs_O1,
                mWs_O2,
                batch_idx,
                head_idx,
                tidx,
                num_k_splits,
            )
            # Reset counter so the next call sees 0 without needing a
            # separate reset kernel. One thread does the regular store —
            # we're the only CTA still touching this counter (we're the
            # last for this (B, H)).
            if tidx == 0:
                mWs_counter[batch_idx, head_idx] = Int32(0)

    @cute.jit
    def _merge_and_store_inkernel(
        self,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        tidx: Int32,
        num_k_splits: cutlass.Constexpr[int],
    ) -> None:
        # Port of `reduction_epilogue` from Triton to CuTe DSL. Each of the
        # 128 consumer threads handles ONE output column (if tidx<head_dim).
        # Scalars (m, d1, d2) are read by every thread — that's NUM_K_SPLITS
        # * 3 redundant HBM reads but they're all L2-cached (we just wrote
        # them) and broadcast-friendly. The per-column O1/O2 reads are
        # unique per thread.
        #
        # Math (Algorithm 1 streaming final reduction):
        #   m_global  = max_s m_s                       (stored in natural-base)
        #   w[s]      = exp(m_s - m_global)
        #   d1_global = Σ_s d1_s * w[s]
        #   d2_global = Σ_s d2_s * w[s]
        #   O1_d      = (Σ_s O1_{s,d} * w[s]) / d1_global
        #   O2_d      = (Σ_s O2_{s,d} * w[s]) / d1_global
        #   c_norm    = d2_global / d1_global
        #   out[d]    = (1 + c_norm) * O1_d - O2_d
        _LOG2_E: cutlass.Constexpr[float] = 1.4426950408889634
        # All partials reads go via `ld.global.cg.f32` (cache-global) so we
        # bypass stale L1 cache lines and pick up the L2-resident writes
        # published by peer CTAs through the acq_rel atomic + their
        # `st.global.cg` stores.
        m_global = -Float32.inf
        for s in cutlass.range(num_k_splits, unroll_full=True):
            m_s = _ld_global_cg_f32(utils.elem_pointer(mWs_m, (batch_idx, head_idx, s)))
            m_global = m_s if m_s > m_global else m_global

        d1_global = Float32(0.0)
        d2_global = Float32(0.0)
        O1_acc = Float32(0.0)
        O2_acc = Float32(0.0)
        for s in cutlass.range(num_k_splits, unroll_full=True):
            m_s = _ld_global_cg_f32(utils.elem_pointer(mWs_m,  (batch_idx, head_idx, s)))
            d1_s = _ld_global_cg_f32(utils.elem_pointer(mWs_d1, (batch_idx, head_idx, s)))
            d2_s = _ld_global_cg_f32(utils.elem_pointer(mWs_d2, (batch_idx, head_idx, s)))
            # exp(m_s - m_global) = exp2((m_s - m_global) * log2(e)).
            # m is stored in natural base (* ln2), see _store_split_partials.
            w = utils.exp2f((m_s - m_global) * Float32(_LOG2_E))
            d1_global += d1_s * w
            d2_global += d2_s * w
            if tidx < self.head_dim:
                O1_s = _ld_global_cg_f32(utils.elem_pointer(mWs_O1, (batch_idx, head_idx, s, tidx)))
                O2_s = _ld_global_cg_f32(utils.elem_pointer(mWs_O2, (batch_idx, head_idx, s, tidx)))
                O1_acc += O1_s * w
                O2_acc += O2_s * w
        inv_d1 = cute.arch.rcp_approx(d1_global)
        c_norm = d2_global * inv_d1
        if tidx < self.head_dim:
            O1_d = O1_acc * inv_d1
            O2_d = O2_acc * inv_d1
            o_fp32 = O1_d + c_norm * O1_d - O2_d
            # Cast to mO's element type (bf16 / fp16 on the fused path,
            # fp32 if we ever pass mO=internal-fp32-buffer). Doing the
            # cancellation in fp32 and the cast at the store is the same
            # precision pattern as the Triton epilogue.
            mO[batch_idx, 0, head_idx, tidx] = mO.element_type(o_fp32)


def _to_cute_tensor(t: torch.Tensor):
    return from_dlpack(t.detach(), assumed_align=16)


def _cached_cute_tensor(t: torch.Tensor):
    """from_dlpack wrapper + memoization keyed on the tensor's data pointer, shape, dtype.
    The first-seen torch.Tensor is strong-ref'd in the cache so the underlying memory
    outlives the returned cute tensor view. For the benchmark harness this bounds the
    cache to ~O(#tensors in CASES * #input slots) = O(16) and is fair versus triton's
    DecodeContext which pre-allocates and retains its tensors.
    """
    key = (t.data_ptr(), tuple(t.shape), t.dtype)
    entry = _cute_input_cache.get(key)
    if entry is not None:
        return entry[1]
    ct = _to_cute_tensor(t)
    _cute_input_cache[key] = (t, ct)
    return ct


def parallax_decode_cutedsl_sm90(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float,
    *,
    debug_stage: int = 0,
    ws: dict | None = None,
    num_k_splits: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.ndim != 4 or q.shape[1] != 1:
        raise ValueError("expected q/r shape (B, 1, H, D)")
    if q.shape != r.shape or k.shape != v.shape:
        raise ValueError("q/r or k/v shape mismatch")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError(f"incompatible q/k shapes: {q.shape} vs {k.shape}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("SM90 TMA/WGMMA CuTe backend requires fp16/bf16 inputs")
    if not q.is_cuda:
        raise ValueError("SM90 TMA/WGMMA CuTe backend requires CUDA tensors")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise RuntimeError("SM90 TMA/WGMMA CuTe backend requires compute capability 9.x")
    if q.shape[-1] > 128:
        raise ValueError("SM90 TMA/WGMMA CuTe backend currently supports head_dim <= 128")
    if k.shape[1] % 64 != 0:
        raise ValueError("SM90 TMA/WGMMA prototype currently requires kv_len % 64 == 0")
    if ws is None:
        raise ValueError(
            "Parallax always uses the in-kernel fused epilogue and requires the "
            "caller to supply a full workspace including the atomic counter. "
            "Use parallax_decode(...) (the public dispatcher) which builds the "
            "workspace via _get_workspace()."
        )

    q = q.contiguous()
    r = r.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    head_dim = q.shape[-1]
    kv_len = k.shape[1]
    # The fused path writes the merged bf16/fp16 row to mO from the last
    # CTA. mO MUST be the caller's output tensor (correct dtype + layout).
    if out is not None:
        out_t_cached = _cached_cute_tensor(out)
    else:
        out_key = (q.shape[0], q.shape[2], q.shape[3], q.device, q.dtype)
        cached = _output_cache.get(out_key)
        if cached is None:
            out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
            out_t_cached = _to_cute_tensor(out)
            _output_cache[out_key] = (out, out_t_cached)
        else:
            out, out_t_cached = cached

    dtype = cutlass.BFloat16 if q.dtype is torch.bfloat16 else cutlass.Float16
    kernel = ParallaxDecodePersistentSplit(dtype, head_dim, debug_stage=debug_stage)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    scale_log2 = float(qk_scale) * math.log2(math.e)

    q_t, r_t, k_t, v_t = [_cached_cute_tensor(t) for t in (q, r, k, v)]
    out_t = out_t_cached

    # Workspace plumbing: reshape (num_bh, S[, D]) → (B, H, S[, D]) so the
    # kernel can index with (batch_idx, head_idx, k_split_id[, d]). The
    # counter is reshaped to (B, H) i32 for the atomic-add.
    B, H = q.shape[0], q.shape[2]
    ws_use = {
        "m":       ws["m"].view(B, H, num_k_splits),
        "d1":      ws["d1"].view(B, H, num_k_splits),
        "d2":      ws["d2"].view(B, H, num_k_splits),
        "O1":      ws["O1"].view(B, H, num_k_splits, head_dim),
        "O2":      ws["O2"].view(B, H, num_k_splits, head_dim),
        "counter": ws["counter"].view(B, H),
    }
    ws_m_t       = _cached_cute_tensor(ws_use["m"])
    ws_d1_t      = _cached_cute_tensor(ws_use["d1"])
    ws_d2_t      = _cached_cute_tensor(ws_use["d2"])
    ws_O1_t      = _cached_cute_tensor(ws_use["O1"])
    ws_O2_t      = _cached_cute_tensor(ws_use["O2"])
    ws_counter_t = _cached_cute_tensor(ws_use["counter"])

    # out.dtype is in the cache key because the kernel specializes the
    # final fp32 → mO.element_type cast.
    key = (q.dtype, out.dtype, head_dim, kv_len, q.shape[0], q.shape[2], debug_stage, num_k_splits)
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(
            kernel,
            q_t, r_t, k_t, v_t, out_t,
            ws_m_t, ws_d1_t, ws_d2_t, ws_O1_t, ws_O2_t, ws_counter_t,
            kv_len, scale_log2, stream,
            num_k_splits,
        )
    _compile_cache[key](
        q_t, r_t, k_t, v_t, out_t,
        ws_m_t, ws_d1_t, ws_d2_t, ws_O1_t, ws_O2_t, ws_counter_t,
        scale_log2, stream,
    )
    return out


# Wave-aware split-count rounding (the in-kernel merge unrolls over the
# split dim, so we need a power-of-two count).
def _round_to_pow2_wave_aware(s: int, num_bh: int, num_sms: int) -> int:
    if s <= 1:
        return 1
    if (s & (s - 1)) == 0:
        return s
    next_pow2 = 1
    while next_pow2 < s:
        next_pow2 <<= 1
    prev_pow2 = next_pow2 >> 1
    waves_prev = (num_bh * prev_pow2 + num_sms - 1) // num_sms
    waves_next = (num_bh * next_pow2 + num_sms - 1) // num_sms
    if waves_prev < waves_next:
        return prev_pow2
    return next_pow2


def _choose_num_k_splits(num_bh: int, kv_len: int, num_sms: int) -> int:
    """Pick a split count S over the L axis so the (B, H, S) grid fits one wave."""
    if num_bh >= num_sms:
        return 1
    K_SEG = 64  # smallest L slice we will ever assign to a single CTA
    MAX_K_SPLITS = 256
    if kv_len < K_SEG:
        return 1
    max_splits = min(kv_len // K_SEG, MAX_K_SPLITS)
    needed = math.ceil(num_sms / num_bh)
    num_k_splits = max(1, min(needed, max_splits))
    tiles_total = (kv_len + 63) // 64
    return min(num_k_splits, max(1, tiles_total))


# Module-level cache for the per-split HBM workspace. Keyed by the launch
# shape so distinct (num_bh, S, head_dim, device) configurations get their
# own tensors and we never reallocate on the hot path.
_WORKSPACE_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def _get_workspace(num_bh: int,
                   num_k_splits: int,
                   head_dim: int,
                   device: torch.device,
                   dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """fp32 workspace for the cross-split log-sum-exp merge.

    Layout (per-split partials of Algorithm 1):
      m       : (num_bh, S)              per-split running max
      d1      : (num_bh, S)              per-split d_1 = sum_j P_1_j
      d2      : (num_bh, S)              per-split d_2 = sum_j P_2_j
      O1      : (num_bh, S, head_dim)    per-split O_1 = sum_j P_1_j v_j
      O2      : (num_bh, S, head_dim)    per-split O_2 = sum_j P_2_j v_j
      counter : (num_bh,) i32            atomic last-CTA detector

    The counter starts at zero; every CTA atomic-adds 1 once its partials are
    published; the CTA that reads OLD == S - 1 is elected the merger. It runs
    the log-sum-exp merge plus the (1 + d_2/d_1) O_1/d_1 - O_2/d_1
    cancellation in fp32 and writes the bf16/fp16 output row. The merger
    also resets the counter to zero so the next call starts clean.
    """
    device_index = device.index if device.index is not None else (
        torch.cuda.current_device() if device.type == "cuda" else -1
    )
    key = (num_bh, num_k_splits, head_dim, device_index)
    cached = _WORKSPACE_CACHE.get(key)
    if cached is not None:
        return cached
    ws = {
        "m":       torch.full((num_bh, num_k_splits),            -float("inf"), dtype=dtype, device=device),
        "d1":      torch.zeros((num_bh, num_k_splits),           dtype=dtype, device=device),
        "d2":      torch.zeros((num_bh, num_k_splits),           dtype=dtype, device=device),
        "O1":      torch.zeros((num_bh, num_k_splits, head_dim), dtype=dtype, device=device),
        "O2":      torch.zeros((num_bh, num_k_splits, head_dim), dtype=dtype, device=device),
        "counter": torch.zeros((num_bh,),                        dtype=torch.int32, device=device),
    }
    _WORKSPACE_CACHE[key] = ws
    return ws


# Cache for caller-output buffers keyed by (B, H, D, device, dtype).
_OUTPUT_CACHE: dict[tuple, torch.Tensor] = {}


def _get_output_buffer(B: int, H: int, D: int,
                       device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    device_index = device.index if device.index is not None else (
        torch.cuda.current_device() if device.type == "cuda" else -1
    )
    key = (B, H, D, device_index, dtype)
    cached = _OUTPUT_CACHE.get(key)
    if cached is not None:
        return cached
    out = torch.empty((B, 1, H, D), device=device, dtype=dtype)
    _OUTPUT_CACHE[key] = out
    return out


def parallax_decode(q: torch.Tensor,
                    r: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    qk_scale: float,
                    *,
                    out: torch.Tensor | None = None) -> torch.Tensor:
    """Parallax forward decode on NVIDIA Hopper.

    Args:
        q, r: ``(B, 1, H, D)`` bf16 or fp16, matching dtypes.
        k, v: ``(B, L, H, D)`` same dtype as q.
        qk_scale: typically ``1 / sqrt(D)``.
        out: optional output tensor; if provided must match
            ``(B, 1, H, D)`` and the dtype of q.

    Returns:
        ``(B, 1, H, D)`` bf16/fp16 tensor implementing the forward of
        Algorithm 1 in the Parallax paper. The composite kernel weight
        rk_j = r dot k_j is computed inline; the caller does not need
        to materialize it.
    """
    assert q.is_contiguous() and r.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.dtype in (torch.bfloat16, torch.float16), (
        f"parallax_decode requires bf16 or fp16 input, got {q.dtype}"
    )
    B, _, H, D = q.shape
    assert r.shape == (B, 1, H, D), f"r shape {tuple(r.shape)} != q shape {tuple(q.shape)}"
    assert k.shape == (B, k.shape[1], H, D)
    assert v.shape == k.shape
    assert D in (64, 128), f"head_dim must be 64 or 128, got {D}"

    kv_len = k.shape[1]
    num_bh = B * H
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_k_splits = _choose_num_k_splits(num_bh, kv_len, num_sms)
    if num_k_splits > 1:
        num_k_splits = _round_to_pow2_wave_aware(num_k_splits, num_bh, num_sms)

    if out is None:
        out = _get_output_buffer(B, H, D, q.device, q.dtype)
    else:
        assert out.shape == (B, 1, H, D), (
            f"out shape mismatch: expected {(B, 1, H, D)}, got {tuple(out.shape)}"
        )
        assert out.dtype == q.dtype, (
            f"out dtype mismatch: expected {q.dtype}, got {out.dtype}"
        )

    ws = _get_workspace(num_bh, num_k_splits, D, q.device)
    parallax_decode_cutedsl_sm90(
        q, r, k, v, qk_scale, ws=ws, num_k_splits=num_k_splits, out=out,
    )
    return out
