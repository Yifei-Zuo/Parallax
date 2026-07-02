# Copyright (c) 2026 Yifei Zuo.
# SPDX-License-Identifier: MIT
"""Parity / smoke tests for the Helion kernels (decode, dense train, varlen).

Small shapes, default kernel configs (autotune forced off below) so the module
stays fast; the exhaustive sweeps live in ``scripts/test_*_helion.py``. Skipped
automatically when the ``helion`` package is not installed (the ``[helion]``
extra) — the rest of the suite's SM90 gate applies as usual via conftest.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

# Force default configs before any helion import (settings snapshot the env).
# Deliberately unconditional: an inherited EFFORT=full would autotune for
# minutes per kernel and read as a CI hang; the sweeps that do want tuned
# configs live in scripts/test_*_helion.py.
os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
pytest.importorskip("helion")

from conftest import make_decode_inputs  # noqa: E402
from parallax import parallax_reference  # noqa: E402
from parallax.helion import (  # noqa: E402
    parallax_decode,
    parallax_func,
    parallax_varlen_func,
)

REL_TOL = 1e-2  # q50 max-norm rel err gate; bf16 floor is ~2-5e-3.


def _make_train_inputs(B, H_q, H_kv, L, D, dtype=torch.bfloat16, seed=0):
    """RMS-normed (B, H, L, D) inputs — parallax_func's convention."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H_q, L, D, device="cuda", dtype=dtype, generator=g)
    r = torch.randn_like(q)
    k = torch.randn(B, H_kv, L, D, device="cuda", dtype=dtype, generator=g)
    v = torch.randn_like(k)
    q = F.rms_norm(q.float(), (D,)).to(dtype).contiguous()
    r = F.rms_norm(r.float(), (D,)).to(dtype).contiguous()
    k = F.rms_norm(k.float(), (D,)).to(dtype).contiguous()
    return q, r, k, v.contiguous()


def _q50(out, ref):
    rel = ((out.float() - ref.float()).abs()
           / max(ref.float().abs().max().item(), 1e-12)).flatten()
    return torch.quantile(rel, 0.5).item()


@pytest.mark.parametrize("B,H_q,H_kv,L,D,W", [
    (1, 4, 2, 256, 128, -1),   # GQA, masked-forward route (D >= 128)
    (2, 4, 4, 256, 64, -1),    # MHA, split-forward route (D <= 64)
    (1, 4, 4, 256, 128, 96),   # SWA
])
def test_train_fwd_bwd_parity(B, H_q, H_kv, L, D, W):
    q, r, k, v = _make_train_inputs(B, H_q, H_kv, L, D)
    for t in (q, r, k, v):
        t.requires_grad_(True)
    scale = D ** -0.5
    o = parallax_func(q, r, k, v, scale, window_size_left=W)
    go = torch.randn(o.shape, device=o.device, dtype=o.dtype,
                     generator=torch.Generator(device=o.device).manual_seed(1))
    o.backward(go)

    q2 = q.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    r2 = r.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    k2 = k.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    v2 = v.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    o_ref = parallax_reference(q2, r2, k2, v2, scale, causal=True,
                               window_size_left=W).permute(0, 2, 1, 3)
    o_ref.backward(go.float())

    assert _q50(o, o_ref) < REL_TOL
    assert _q50(q.grad, q2.grad.permute(0, 2, 1, 3)) < REL_TOL
    assert _q50(r.grad, r2.grad.permute(0, 2, 1, 3)) < REL_TOL
    assert _q50(k.grad, k2.grad.permute(0, 2, 1, 3)) < REL_TOL
    assert _q50(v.grad, v2.grad.permute(0, 2, 1, 3)) < REL_TOL


def test_train_fp16_grad_dtypes():
    """fp16 inputs must yield fp16 grads (kernels accumulate in bf16 internally)."""
    q, r, k, v = _make_train_inputs(1, 2, 2, 128, 64, dtype=torch.float16)
    for t in (q, r, k, v):
        t.requires_grad_(True)
    o = parallax_func(q, r, k, v)
    assert o.dtype == torch.float16
    o.backward(torch.randn_like(o))
    for t in (q, r, k, v):
        assert t.grad is not None and t.grad.dtype == torch.float16
        assert torch.isfinite(t.grad).all()


def test_train_input_validation():
    q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="bf16 or fp16"):
        parallax_func(q, q, q, q)
    qh = q.to(torch.bfloat16)
    kh = torch.randn(1, 3, 128, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible"):
        parallax_func(qh, qh, kh, kh)


def test_varlen_parity_small():
    lens = [1, 130, 256]
    T = sum(lens)
    H_q, H_kv, D, W = 4, 2, 64, -1
    g = torch.Generator(device="cuda").manual_seed(0)
    mk = lambda h: F.rms_norm(
        torch.randn(1, T, h, D, device="cuda", dtype=torch.bfloat16,
                    generator=g).float(), (D,)).bfloat16().contiguous()
    q, r, k = mk(H_q), mk(H_q), mk(H_kv)
    v = torch.randn(1, T, H_kv, D, device="cuda", dtype=torch.bfloat16,
                    generator=g).contiguous()
    cu = F.pad(torch.tensor(lens, device="cuda").cumsum(0), (1, 0)).to(torch.int32)
    scale = D ** -0.5
    for t in (q, r, k, v):
        t.requires_grad_(True)
    o = parallax_varlen_func(q, r, k, v, scale, window_size_left=W, cu_seqlens=cu)
    go = torch.randn(o.shape, device=o.device, dtype=o.dtype,
                     generator=torch.Generator(device=o.device).manual_seed(1))
    o.backward(go)

    o_ref = torch.empty_like(o, dtype=torch.float32)
    grads_ref = [torch.empty_like(t, dtype=torch.float32) for t in (q, r, k, v)]
    for i in range(len(lens)):
        bos, eos = int(cu[i]), int(cu[i + 1])
        leaves = [t[:, bos:eos].detach().float().requires_grad_(True)
                  for t in (q, r, k, v)]
        o_s = parallax_reference(*leaves, scale, causal=True, window_size_left=W)
        o_s.backward(go[:, bos:eos].float())
        o_ref[:, bos:eos] = o_s.detach()
        for gr, leaf in zip(grads_ref, leaves):
            gr[:, bos:eos] = leaf.grad

    assert _q50(o, o_ref) < REL_TOL
    for t, gr in zip((q, r, k, v), grads_ref):
        assert _q50(t.grad, gr) < REL_TOL


def test_decode_parity_small():
    q, r, k, v = make_decode_inputs(4, 8, 512, D=128)  # heads-last (B, 1, H, D)
    scale = 128 ** -0.5
    out = parallax_decode(q, r, k, v, scale)
    ref = parallax_reference(q, r, k, v, scale, causal=True)
    assert _q50(out, ref) < REL_TOL
