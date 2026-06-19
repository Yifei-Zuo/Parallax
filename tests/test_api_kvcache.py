"""API contract tests for ``parallax_attn_with_kvcache``.

Covers the unified FA-style entry, explicit buffer ownership (no silent
output reuse / no aliasing across ``out=None`` calls), the ``window_size``
mapping, the default ``scale``, caller-owned ``out``, and the
``NotImplementedError`` surfaces for not-yet-supported features.

Run with pytest (``pytest tests/test_api_kvcache.py -v``) or standalone
(``python tests/test_api_kvcache.py``).
"""
from __future__ import annotations

import math

import pytest
import torch

from parallax import (
    parallax_attn_with_kvcache,
    parallax_decode,
    parallax_reference,
)

_NO_SM90 = (
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9
)
pytestmark = pytest.mark.skipif(_NO_SM90, reason="requires SM90 (Hopper) GPU")

D = 128


def _qrkv(B, H, kv_len, *, H_kv=None, dtype=torch.bfloat16, seed=0):
    H_kv = H_kv or H
    torch.manual_seed(seed)
    q = torch.randn(B, 1, H, D, device="cuda", dtype=dtype)
    r = torch.randn_like(q) * 0.5
    k = torch.randn(B, kv_len, H_kv, D, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    return q, r, k, v


def _rel_err(out, ref):
    diff = (out.float() - ref.float()).abs().max().item()
    return diff / max(ref.float().abs().max().item(), 1e-6)


def test_matches_reference():
    q, r, k, v = _qrkv(1, 8, 4096)
    out = parallax_attn_with_kvcache(q, r, k, v)
    ref = parallax_reference(q, r, k, v, 1.0 / math.sqrt(D), causal=True)
    assert _rel_err(out, ref) < 1e-2


def test_no_silent_output_reuse():
    """Two out=None calls must not alias, and the first result must survive."""
    q1, r1, k1, v1 = _qrkv(2, 8, 1024, seed=1)
    q2, r2, k2, v2 = _qrkv(2, 8, 1024, seed=2)  # same shape, different data

    o1 = parallax_attn_with_kvcache(q1, r1, k1, v1)
    o1_snapshot = o1.clone()
    o2 = parallax_attn_with_kvcache(q2, r2, k2, v2)

    assert o1.data_ptr() != o2.data_ptr(), "out=None calls aliased the same buffer"
    assert torch.equal(o1, o1_snapshot), "first result was clobbered by the second call"
    # And o2 is itself correct.
    ref2 = parallax_reference(q2, r2, k2, v2, 1.0 / math.sqrt(D), causal=True)
    assert _rel_err(o2, ref2) < 1e-2


def test_caller_owned_out_is_returned_and_written():
    q, r, k, v = _qrkv(1, 8, 2048)
    out = torch.empty(1, 1, 8, D, device="cuda", dtype=torch.bfloat16)
    ret = parallax_attn_with_kvcache(q, r, k, v, out=out)
    assert ret.data_ptr() == out.data_ptr(), "did not write into caller-provided out"
    ref = parallax_reference(q, r, k, v, 1.0 / math.sqrt(D), causal=True)
    assert _rel_err(out, ref) < 1e-2


def test_default_scale_matches_explicit():
    q, r, k, v = _qrkv(1, 8, 1024)
    auto = parallax_attn_with_kvcache(q, r, k, v)              # scale=None -> 1/sqrt(D)
    explicit = parallax_decode(q, r, k, v, 1.0 / math.sqrt(D))  # deprecated alias
    assert torch.equal(auto, explicit)


@pytest.mark.parametrize("win", [64, 128, 256, 333])
def test_window_size_tuple_maps_to_window_size_left(win):
    q, r, k, v = _qrkv(1, 8, 4096)
    via_tuple = parallax_attn_with_kvcache(q, r, k, v, window_size=(win, 0))
    via_int = parallax_decode(q, r, k, v, 1.0 / math.sqrt(D), window_size_left=win)
    assert torch.equal(via_tuple, via_int)


def test_window_size_none_and_negative_disable_swa():
    q, r, k, v = _qrkv(1, 8, 2048)
    a = parallax_attn_with_kvcache(q, r, k, v, window_size=None)
    b = parallax_attn_with_kvcache(q, r, k, v, window_size=(-1, -1))
    c = parallax_decode(q, r, k, v, 1.0 / math.sqrt(D), window_size_left=-1)
    assert torch.equal(a, c) and torch.equal(b, c)


def test_window_size_positive_right_rejected():
    q, r, k, v = _qrkv(1, 8, 512)
    with pytest.raises(ValueError, match="right"):
        parallax_attn_with_kvcache(q, r, k, v, window_size=(128, 4))


def test_page_table_not_implemented():
    q, r, k, v = _qrkv(1, 8, 512)
    page_table = torch.zeros(1, 8, dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError, match="[Pp]aged"):
        parallax_attn_with_kvcache(q, r, k, v, page_table=page_table)


def test_seqused_k_supported():
    """seqused_k now controls the active KV length at runtime — kernel reads
    it per-batch, no per-step recompile. K/V can be a pre-allocated full cache."""
    q, r, k, v = _qrkv(1, 8, 4096)
    seqused = torch.tensor([2000], dtype=torch.int32, device="cuda")
    out = parallax_attn_with_kvcache(q, r, k, v, seqused_k=seqused)
    ref = parallax_reference(q, r, k[:, :2000], v[:, :2000],
                              1.0 / math.sqrt(D), causal=True)
    assert _rel_err(out, ref) < 1e-2


def test_seqused_k_dtype_rejected():
    q, r, k, v = _qrkv(1, 8, 512)
    bad = torch.full((1,), 300, dtype=torch.int64, device="cuda")
    with pytest.raises(TypeError, match="int32"):
        parallax_attn_with_kvcache(q, r, k, v, seqused_k=bad)


def test_seqused_k_shape_rejected():
    q, r, k, v = _qrkv(2, 8, 512)
    bad = torch.full((1,), 300, dtype=torch.int32, device="cuda")  # B=1 vs q has B=2
    with pytest.raises(ValueError, match="seqused_k must have shape"):
        parallax_attn_with_kvcache(q, r, k, v, seqused_k=bad)


def test_seqused_k_over_ceiling_rejected():
    """seqused_k values exceeding k_cache.shape[1] must raise in eager mode."""
    q, r, k, v = _qrkv(1, 8, 512)
    over = torch.tensor([999], dtype=torch.int32, device="cuda")  # 999 > 512
    with pytest.raises(ValueError, match="exceeds k_cache.shape"):
        parallax_attn_with_kvcache(q, r, k, v, seqused_k=over)


def test_seqused_k_zero_rejected():
    """seqused_k=0 must raise in eager mode (kernel would clamp to 1)."""
    q, r, k, v = _qrkv(1, 8, 512)
    zero = torch.tensor([0], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="values < 1"):
        parallax_attn_with_kvcache(q, r, k, v, seqused_k=zero)


def test_seqused_k_negative_rejected():
    """Negative seqused_k values must raise in eager mode."""
    q, r, k, v = _qrkv(1, 8, 512)
    neg = torch.tensor([-1], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="values < 1"):
        parallax_attn_with_kvcache(q, r, k, v, seqused_k=neg)


@pytest.mark.parametrize("ratio", [2, 4, 8])
def test_gqa_supported(ratio):
    """GQA via head-packing lands at pack_n in {2, 4, 8}: one CTA emits
    pack_n query-head rows per kv head. Output must match the fp32
    reference within bf16 noise floor (~3-4e-3)."""
    q, r, k, v = _qrkv(1, 8, 1024, H_kv=8 // ratio)  # pack_n = ratio
    out = parallax_attn_with_kvcache(q, r, k, v)
    ref = parallax_reference(q, r, k, v, 1.0 / math.sqrt(D), causal=True)
    assert _rel_err(out, ref) < 1e-2
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


def test_unsupported_gqa_ratio_raises():
    """Odd or unsupported pack_n (e.g. 3) must raise NotImplementedError."""
    q, r, k, v = _qrkv(1, 6, 512, H_kv=2)  # H_q=6, H_kv=2 → pack_n=3
    with pytest.raises(NotImplementedError, match="pack_n"):
        parallax_attn_with_kvcache(q, r, k, v)


if __name__ == "__main__":
    import sys
    if _NO_SM90:
        print("SKIP: requires SM90 GPU")
        sys.exit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        # crude param expansion for the standalone runner
        params = getattr(fn, "pytestmark", None)
        if fn.__name__ == "test_window_size_tuple_maps_to_window_size_left":
            cases = [64, 128, 256, 333]
        else:
            cases = [None]
        for c in cases:
            try:
                fn(c) if c is not None else fn()
                print(f"PASS {fn.__name__}{f'[{c}]' if c is not None else ''}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {fn.__name__}{f'[{c}]' if c is not None else ''}: {e}")
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
