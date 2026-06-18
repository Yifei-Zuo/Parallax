"""End-to-end check for the Parallax CuTe decode kernel.

One self-contained script that exercises every capability of
``parallax_attn_with_kvcache`` and ``GraphedDecode``, reporting PASS/FAIL
with worst observed bf16 rel-err vs the fp32 ``parallax_reference`` oracle.
Exits non-zero on any failure. Runs in ~1 minute on an H200.

Coverage:
  - Unified entry: ``parallax_attn_with_kvcache``; ``out=None`` never aliases.
  - Large-batch determinism: race-free over many launches (B=32, H=8).
  - GraphedDecode: capture/replay correctness.
  - GQA via head-packing pack_n in {1, 2, 4, 8}; SWA x GQA combined.
  - Runtime kv_len: ONE compiled kernel serves a whole decode trajectory
    (1 -> max_kv_len) without recompile.
  - GraphedDecode + seqused_k: ONE captured graph serves every kv_len at
    replay time.

Usage:  python scripts/check_decode.py
"""
from __future__ import annotations

import gc
import math
import sys

import torch

from parallax import (
    GraphedDecode,
    parallax_attn_with_kvcache,
    parallax_reference,
)

D = 128
REL_TOL = 1e-2
DEV = "cuda"
DT = torch.bfloat16


def _rand(*shape, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, device=DEV, dtype=DT) * scale


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    if torch.isnan(out).any():
        return float("nan")
    return (
        (out.float() - ref.float()).abs().max().item()
        / max(ref.float().abs().max().item(), 1e-6)
    )


def _ref(q, r, k, v, *, window: int = -1) -> torch.Tensor:
    return parallax_reference(
        q, r, k, v, 1.0 / math.sqrt(D),
        causal=True, window_size_left=window,
    )


def _run(name: str, fn):
    print(f"  {name:<44} ", end="", flush=True)
    try:
        detail = fn() or ""
        print(f"PASS  {detail}")
        return True
    except AssertionError as e:
        print(f"FAIL  {e}")
        return False
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def test_basic_api():
    B, kv_len, H_q, H_kv = 2, 1024, 8, 8
    torch.manual_seed(0)
    q = _rand(B, 1, H_q, D)
    r = _rand(B, 1, H_q, D, scale=0.5)
    k = _rand(B, kv_len, H_kv, D)
    v = _rand(B, kv_len, H_kv, D)

    out_a = parallax_attn_with_kvcache(q, r, k, v)
    out_b = parallax_attn_with_kvcache(q, r, k, v)
    assert out_a.data_ptr() != out_b.data_ptr(), "out=None aliased across calls"

    ref = _ref(q, r, k, v)
    re = _rel_err(out_a, ref)
    assert re < REL_TOL, f"rel_err={re:.2e}"
    return f"rel_err={re:.2e}"


def test_large_batch_no_race():
    B, kv_len, H = 32, 4096, 8
    torch.manual_seed(1)
    q = _rand(B, 1, H, D)
    r = _rand(B, 1, H, D, scale=0.5)
    k = _rand(B, kv_len, H, D)
    v = _rand(B, kv_len, H, D)
    ref = _ref(q, r, k, v)

    worst = 0.0
    for _ in range(60):
        re = _rel_err(parallax_attn_with_kvcache(q, r, k, v), ref)
        assert re < REL_TOL, f"race surfaced rel_err={re:.2e}"
        worst = max(worst, re)
    return f"60 launches, worst {worst:.2e}"


def test_gqa_pack_n():
    worst = 0.0
    for pack_n in (1, 2, 4, 8):
        H_kv = 4
        H_q = H_kv * pack_n
        B, kv_len = 2, 2048
        torch.manual_seed(2 + pack_n)
        q = _rand(B, 1, H_q, D)
        r = _rand(B, 1, H_q, D, scale=0.5)
        k = _rand(B, kv_len, H_kv, D)
        v = _rand(B, kv_len, H_kv, D)
        re = _rel_err(parallax_attn_with_kvcache(q, r, k, v), _ref(q, r, k, v))
        assert re < REL_TOL, f"pack_n={pack_n} rel_err={re:.2e}"
        worst = max(worst, re)
    return f"pack_n=1/2/4/8 all green, worst {worst:.2e}"


def test_swa_gqa():
    pack_n, H_kv = 8, 4
    H_q = H_kv * pack_n
    B, kv_len, win = 1, 4096, 256
    torch.manual_seed(7)
    q = _rand(B, 1, H_q, D)
    r = _rand(B, 1, H_q, D, scale=0.5)
    k = _rand(B, kv_len, H_kv, D)
    v = _rand(B, kv_len, H_kv, D)
    out = parallax_attn_with_kvcache(q, r, k, v, window_size=(win, 0))
    re = _rel_err(out, _ref(q, r, k, v, window=win))
    assert re < REL_TOL, f"rel_err={re:.2e}"
    return f"win={win}, pack_n=8, rel_err={re:.2e}"


def test_graphed_decode_runtime_kv_len():
    B, H_q, H_kv, max_kv = 2, 8, 8, 8192
    gd = GraphedDecode(B, H_q, max_kv_len=max_kv, head_dim=D, H_kv=H_kv)

    torch.manual_seed(9)
    k_full = _rand(B, max_kv, H_kv, D)
    v_full = _rand(B, max_kv, H_kv, D)
    gd.k.copy_(k_full)
    gd.v.copy_(v_full)

    worst = 0.0
    lengths = (1, 64, 257, 1024, 4096, max_kv)
    for kv_len in lengths:
        q = _rand(B, 1, H_q, D)
        r = _rand(B, 1, H_q, D, scale=0.5)
        gd.q.copy_(q)
        gd.r.copy_(r)
        gd.cache_seqlens.fill_(kv_len)
        out = gd.replay()
        ref = _ref(q, r, k_full[:, :kv_len], v_full[:, :kv_len])
        re = _rel_err(out, ref)
        assert re < REL_TOL, f"kv_len={kv_len} rel_err={re:.2e}"
        worst = max(worst, re)
    return f"{len(lengths)} lengths from one capture, worst {worst:.2e}"


def test_compile_once_across_kv_lens():
    import importlib
    pd_mod = importlib.import_module("parallax.cute.parallax_decode")
    pd_mod._compile_cache.clear()

    B, H_q, H_kv, max_kv = 2, 8, 8, 8192
    torch.manual_seed(11)
    k_full = _rand(B, max_kv, H_kv, D)
    v_full = _rand(B, max_kv, H_kv, D)
    seqused = torch.empty(B, device=DEV, dtype=torch.int32)

    for kv_len in (1, 128, 1024, 4096, max_kv):
        q = _rand(B, 1, H_q, D)
        r = _rand(B, 1, H_q, D, scale=0.5)
        seqused.fill_(kv_len)
        parallax_attn_with_kvcache(q, r, k_full, v_full, seqused_k=seqused)

    n = len(pd_mod._compile_cache)
    assert n == 1, f"expected 1 compiled kernel for the whole bucket, got {n}"
    return "5 kv_lens served by 1 compiled kernel"


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: requires CUDA")
        return 0
    cc = torch.cuda.get_device_capability()
    if cc != (9, 0):
        print(f"SKIP: requires SM90 (Hopper), got {cc}")
        return 0
    print(f"Running on {torch.cuda.get_device_name()} (rel-err tol {REL_TOL})")
    print()

    tests = [
        ("unified API, no silent reuse        ", test_basic_api),
        ("no large-batch decode race          ", test_large_batch_no_race),
        ("GQA pack_n in {1, 2, 4, 8}          ", test_gqa_pack_n),
        ("SWA x GQA pack_n=8                  ", test_swa_gqa),
        ("GraphedDecode + seqused_k           ", test_graphed_decode_runtime_kv_len),
        ("compile-once across kv_len          ", test_compile_once_across_kv_lens),
    ]
    failed = sum(0 if _run(name, fn) else 1 for name, fn in tests)
    print()
    print("ALL PASS" if failed == 0 else f"{failed} FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
