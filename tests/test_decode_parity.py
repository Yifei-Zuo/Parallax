"""Parity + serving matrix for the Parallax CuTeDSL decode kernel.

Two tiers:

* **fast** (default): a representative matrix — baseline / sliding-window /
  partial-tile / moderate split-K / larger-batch / head_dim 64 — that compiles
  in a few minutes and runs on every PR.
* **stress** (``-m stress``, deselected by default): the exhaustive
  ``(B, H, kv_len, window, D)`` sweep, including the long-``kv_len`` small-``B*H``
  split-K shapes whose in-kernel merge unrolls over up to 256 splits and costs
  minutes to JIT-compile. This is the shape-sparse race catcher; run it nightly
  / on demand.

Everything is checked against the fp32 ``parallax_reference`` oracle; GQA
(head-packing ``pack_n ∈ {1, 2, 4, 8}``) is now supported. Not-yet-supported
serving features (paged KV, varlen prefill/extend) raise ``NotImplementedError``
and are covered in ``test_api_kvcache.py``.
"""
from __future__ import annotations

import pytest
import torch

from parallax import parallax_attn_with_kvcache, parallax_reference
from conftest import REL_TOL, make_decode_inputs, rel_err, scale_for


def _win_kwarg(window_size_left):
    """Map the FA2 int convention used by the case tables to the API kwarg."""
    return None if window_size_left < 0 else (window_size_left, 0)


def _check_parity(B, H, kv_len, win, D):
    q, r, k, v = make_decode_inputs(B, H, kv_len, D=D,
                                    seed=B * 131 + H * 7 + kv_len + D)
    out = parallax_attn_with_kvcache(q, r, k, v, window_size=_win_kwarg(win),
                                     scale=scale_for(D))
    ref = parallax_reference(q, r, k, v, scale_for(D), causal=True,
                             window_size_left=win)
    assert not torch.isnan(out).any(), f"spurious NaN: B{B} H{H} L{kv_len} win{win} D{D}"
    assert rel_err(out, ref) < REL_TOL, f"rel-err high: B{B} H{H} L{kv_len} win{win} D{D}"


# --- fast default matrix (D=128 unless the tuple carries its own D) -----------
# (B, H, kv_len, window_size_left, D)
_FAST_CASES = [
    # baseline (SWA disabled)
    (1, 8, 512, -1, 128), (1, 8, 1024, -1, 128), (1, 8, 4096, -1, 128),
    # aligned window
    (1, 8, 4096, 256, 128), (1, 8, 4096, 64, 128),
    # unaligned window (first-tile partial mask)
    (1, 8, 4096, 100, 128),
    # partial first + last tile
    (1, 8, 1000, 128, 128), (1, 8, 65, 1024, 128),
    # moderate split-K (small B*H, with + without SWA)
    (2, 4, 8192, -1, 128), (2, 4, 8192, 1024, 128), (2, 4, 8192, 130, 128),
    # larger batch (no-split occupancy path)
    (8, 8, 2048, -1, 128), (32, 8, 1024, -1, 128),
    # head_dim = 64 path
    (1, 8, 1024, -1, 64), (4, 8, 2048, 256, 64),
]


@pytest.mark.sm90
@pytest.mark.parametrize("B,H,kv_len,win,D", _FAST_CASES)
def test_decode_parity(B, H, kv_len, win, D):
    _check_parity(B, H, kv_len, win, D)


# --- exhaustive stress sweep (opt-in: -m stress) -----------------------------
def _stress_cases():
    cases = []
    for B, H in [(1, 1), (1, 8), (2, 4), (8, 8), (32, 8)]:
        for kv_len in [512, 1000, 4096, 8192, 16384]:
            for win in [-1, 64, 256, 1024]:
                for D in [128, 64]:
                    cases.append((B, H, kv_len, win, D))
    return cases


@pytest.mark.sm90
@pytest.mark.stress
@pytest.mark.parametrize("B,H,kv_len,win,D", _stress_cases())
def test_decode_parity_stress(B, H, kv_len, win, D):
    _check_parity(B, H, kv_len, win, D)


@pytest.mark.sm90
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_dtypes(dtype):
    q, r, k, v = make_decode_inputs(1, 8, 1024, dtype=dtype)
    out = parallax_attn_with_kvcache(q, r, k, v)
    ref = parallax_reference(q, r, k, v, scale_for(128), causal=True)
    assert out.dtype == dtype
    assert rel_err(out, ref) < REL_TOL


# --- finite-padding contract -------------------------------------------------
@pytest.mark.sm90
@pytest.mark.parametrize("kv_len", [65, 100, 1000])
def test_partial_last_tile_is_finite(kv_len):
    """kv_len not a multiple of 64 -> partial last tile must not leak NaN.

    Validates the dense side of the finite-padding contract: the masked columns
    of the trailing partial tile do not poison the PV matmul.
    """
    q, r, k, v = make_decode_inputs(1, 8, kv_len)
    out = parallax_attn_with_kvcache(q, r, k, v)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


@pytest.mark.sm90
def test_nan_in_valid_kv_propagates():
    """Documents that the kernel does not sanitize inputs: a NaN in a *valid*
    KV position propagates to the output (NaN-in -> NaN-out is correct)."""
    q, r, k, v = make_decode_inputs(1, 8, 1024)
    v[0, 10, 0, :] = float("nan")
    out = parallax_attn_with_kvcache(q, r, k, v)
    assert torch.isnan(out).any()


# --- GQA pack_n parity --------------------------------------------------------
@pytest.mark.sm90
@pytest.mark.parametrize("pack_n", [2, 4, 8])
@pytest.mark.parametrize("B", [1, 2, 4])
@pytest.mark.parametrize("kv_len", [1024, 4096])
def test_gqa_parity(B, kv_len, pack_n):
    """GQA via head-packing: H_q = H_kv * pack_n, one CTA emits pack_n query
    heads per kv head. Output must match the fp32 reference (which already
    handles GQA via head-replication) within the bf16 noise floor."""
    H_kv = 4
    H_q = H_kv * pack_n
    D = 128
    q, r, k, v = make_decode_inputs(B, H_q, kv_len, H_kv=H_kv, D=D)
    out = parallax_attn_with_kvcache(q, r, k, v, scale=scale_for(D))
    ref = parallax_reference(q, r, k, v, scale_for(D), causal=True)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    assert rel_err(out, ref) < REL_TOL


# --- not-yet-supported GQA ratios (raise pre-compile) -------------------------
@pytest.mark.sm90
@pytest.mark.parametrize("ratio", [3, 5, 6, 7])
def test_unsupported_gqa_ratio_raises(ratio):
    """pack_n not in {1, 2, 4, 8} must raise NotImplementedError pre-compile."""
    H_kv = 4
    H_q = H_kv * ratio
    q, r, k, v = make_decode_inputs(1, H_q, 1024, H_kv=H_kv)
    with pytest.raises(NotImplementedError, match="pack_n"):
        parallax_attn_with_kvcache(q, r, k, v)


@pytest.mark.sm90
def test_paged_kv_xfail():
    """Paged KV (page_table) — lands with the paged-TMA prefill/extend kernel."""
    q, r, k, v = make_decode_inputs(1, 8, 1024)
    page_table = torch.zeros(1, 16, dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError):
        parallax_attn_with_kvcache(q, r, k, v, page_table=page_table)


@pytest.mark.sm90
@pytest.mark.parametrize("active_len", [128, 500, 999, 1024])
def test_seqused_k_runtime_length(active_len):
    """seqused_k unlocks compile-once across decode steps: the K/V cache is
    allocated at its max length once and seqused_k controls the per-batch
    active length read at runtime by the kernel. Verify the output matches
    the same call with K/V sliced to the active length."""
    B, H, kv_max, D = 1, 8, 1024, 128
    q, r, k, v = make_decode_inputs(B, H, kv_max, D=D)
    seqused = torch.tensor([active_len], dtype=torch.int32, device="cuda")
    out = parallax_attn_with_kvcache(q, r, k, v, seqused_k=seqused,
                                      scale=scale_for(D))
    ref = parallax_reference(q, r, k[:, :active_len], v[:, :active_len],
                              scale_for(D), causal=True)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    assert rel_err(out, ref) < REL_TOL
