"""fp32 PyTorch reference implementation of Parallax mechanism.

Implements Algorithm 1 of the Parallax paper (https://arxiv.org/abs/2605.29157) in pure PyTorch.
Works for both training (multi-query causal attention) and decoding (single-query, full-KV)
- for decoding pass ``q, r`` of shape ``(B, 1, H, D)``;
- for training pass ``q, r`` of shape ``(B, Q, H, D)`` with ``Q == L`` and ``causal=True``.
"""

from __future__ import annotations

import torch


def parallax_reference(q: torch.Tensor,
                       r: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor,
                       qk_scale: float,
                       *,
                       causal: bool = True) -> torch.Tensor:
    """fp32 PyTorch reference for Parallax (Algorithm 1).

    Args:
        q, r: ``(B, Q, H, D)`` tensors.
        k, v: ``(B, L, H, D)`` tensors.
        qk_scale: typically ``1 / sqrt(D)``.
        causal: if True (default), apply causal masking aligned to the *end*
            of the KV sequence — query position ``i`` (``0 <= i < Q``) attends
            to key positions ``j`` with ``j <= L - Q + i``. This is the
            standard convention used by FlashAttention with kvcache: when
            ``Q == L`` it is the usual triangular causal mask, and when
            ``Q == 1`` (decode) the single query attends to all L keys.

    Returns:
        ``(B, Q, H, D)`` fp32 tensor.
    """
    q_ = q.permute(0, 2, 1, 3).float()  # (B, H, Q, D)
    r_ = r.permute(0, 2, 1, 3).float()
    k_ = k.permute(0, 2, 1, 3).float()  # (B, H, L, D)
    v_ = v.permute(0, 2, 1, 3).float()

    _, _, Q, _ = q_.shape
    L = k_.shape[2]

    s1 = torch.einsum("bhqd,bhld->bhql", q_, k_) * qk_scale  # (B, H, Q, L)
    s2 = torch.einsum("bhqd,bhld->bhql", r_, k_)             # (B, H, Q, L)

    if causal:
        i_idx = torch.arange(Q, device=q.device).view(Q, 1)
        j_idx = torch.arange(L, device=q.device).view(1, L)
        mask = j_idx <= (L - Q + i_idx)                       # (Q, L)
        s1 = s1.masked_fill(~mask, float("-inf"))

    m = s1.amax(dim=-1, keepdim=True)
    p1 = (s1 - m).exp()
    d1 = p1.sum(dim=-1, keepdim=True)
    p2 = p1 * s2                                              # masked rows of s1 zero p1, so p2 is 0 there too
    d2 = p2.sum(dim=-1, keepdim=True)
    O1 = torch.einsum("bhql,bhld->bhqd", p1, v_)
    O2 = torch.einsum("bhql,bhld->bhqd", p2, v_)

    c_norm = d2 / d1
    out = O1 / d1 * (1.0 + c_norm) - O2 / d1
    return out.permute(0, 2, 1, 3).contiguous()
