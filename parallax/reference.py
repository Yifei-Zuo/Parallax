"""fp32 PyTorch reference implementation of Parallax.

Implements Algorithm 1 of the Parallax paper (https://arxiv.org/abs/2605.29157)
in pure PyTorch. Works for both training (multi-query causal) and decoding
(single-query, full-KV); supports GQA via ``H_q != H_kv`` (with
``H_q % H_kv == 0``) and causal sliding-window attention via
``window_size_left`` (FA2 convention).
"""

from __future__ import annotations

import torch


def parallax_reference(q: torch.Tensor,
                       r: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor,
                       qk_scale: float,
                       *,
                       causal: bool = True,
                       window_size_left: int = -1) -> torch.Tensor:
    """fp32 PyTorch reference for Parallax (Algorithm 1).

    Args:
        q, r: ``(B, Q, H_q, D)`` tensors.
        k, v: ``(B, L, H_kv, D)`` tensors; ``H_q`` must be divisible by ``H_kv``.
        qk_scale: typically ``1 / sqrt(D)``.
        causal: if True (default), apply causal masking aligned to the *end*
            of the KV sequence — query position ``i`` (``0 <= i < Q``) attends
            to key positions ``j`` with ``j <= L - Q + i``. When ``Q == L`` this
            is the standard triangular causal mask; when ``Q == 1`` (decode) the
            single query attends to all L keys.
        window_size_left: causal sliding-window length (FA2 convention).
            ``-1`` (default) disables; ``>= 0`` additionally restricts each query
            to keys at most ``window_size_left + 1`` positions behind it.

    Returns:
        ``(B, Q, H_q, D)`` fp32 tensor.
    """
    H_q = q.shape[2]
    H_kv = k.shape[2]
    if H_q % H_kv != 0:
        raise ValueError(
            f"H_q ({H_q}) must be divisible by H_kv ({H_kv}) for GQA"
        )
    n_rep = H_q // H_kv

    q_ = q.permute(0, 2, 1, 3).float()  # (B, H_q, Q, D)
    r_ = r.permute(0, 2, 1, 3).float()
    k_ = k.permute(0, 2, 1, 3).float()  # (B, H_kv, L, D)
    v_ = v.permute(0, 2, 1, 3).float()

    if n_rep > 1:
        k_ = k_.repeat_interleave(n_rep, dim=1)  # (B, H_q, L, D)
        v_ = v_.repeat_interleave(n_rep, dim=1)

    _, _, Q, _ = q_.shape
    L = k_.shape[2]

    s1 = torch.einsum("bhqd,bhld->bhql", q_, k_) * qk_scale  # (B, H_q, Q, L)
    s2 = torch.einsum("bhqd,bhld->bhql", r_, k_)

    if causal or window_size_left >= 0:
        i_idx = torch.arange(Q, device=q.device).view(Q, 1)
        j_idx = torch.arange(L, device=q.device).view(1, L)
        # Causal-with-prefix: row i in absolute terms sits at L - Q + i.
        causal_mask = j_idx <= (L - Q + i_idx) if causal else torch.ones_like(j_idx).bool()
        if window_size_left >= 0:
            window_mask = j_idx >= (L - Q + i_idx) - window_size_left + 1
            mask = causal_mask & window_mask
        else:
            mask = causal_mask
        s1 = s1.masked_fill(~mask, float("-inf"))

    m = s1.amax(dim=-1, keepdim=True)
    p1 = (s1 - m).exp()
    d1 = p1.sum(dim=-1, keepdim=True)
    p2 = p1 * s2
    d2 = p2.sum(dim=-1, keepdim=True)
    O1 = torch.einsum("bhql,bhld->bhqd", p1, v_)
    O2 = torch.einsum("bhql,bhld->bhqd", p2, v_)

    c_norm = d2 / d1
    out = O1 / d1 * (1.0 + c_norm) - O2 / d1
    return out.permute(0, 2, 1, 3).contiguous()
