# Copyright (c) 2026 Anonymous Authors.
# SPDX-License-Identifier: MIT
"""Small torch + triton helpers shared by the Triton kernels."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def exp2(x):
    """Base-2 exponential in fp32 (logits are pre-scaled by ``1 / ln 2``)."""
    return tl.math.exp2(x.to(tl.float32))


@functools.lru_cache(maxsize=None)
def _is_hopper_class(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] == 9


def varlen_block_size(head_dim: int, device_index: int) -> int:
    """Square tile size (``BT == BS``) shared by every varlen kernel launch.

    128 rows on Hopper with a small head dim, otherwise 64. Kept modest to bound
    the fp32 accumulator footprint.
    """
    if _is_hopper_class(device_index) and head_dim <= 64:
        return 128
    return 64


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Map each row-tile program id to ``(seq_id, chunk_id_within_seq)`` for varlen.

    Given cumulative sequence lengths ``cu_seqlens`` ``[N + 1]`` and a tile size,
    returns an int tensor ``[sum_i ceil(len_i / chunk_size), 2]`` on
    ``cu_seqlens.device``; ``len(result)`` is the number of row-tile programs.
    """
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_counts = (lens + chunk_size - 1) // chunk_size
    seg_id = torch.repeat_interleave(
        torch.arange(chunk_counts.numel(), device=cu_seqlens.device),
        chunk_counts,
    )
    seg_start = F.pad(chunk_counts.cumsum(0), (1, 0))[:-1]
    intra = torch.arange(seg_id.numel(), device=cu_seqlens.device) - seg_start[seg_id]
    return torch.stack([seg_id, intra], dim=1).to(cu_seqlens)
