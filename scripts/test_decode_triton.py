"""Precision test for the Triton single-token decode kernel.

Compares :func:`parallax.triton.parallax_decode` against the fp32 reference
(:func:`parallax.parallax_reference`) over the shared ``REFERENCE_SHAPES``, a
sliding-window sweep, GQA/MQA shapes, and left-padding (``cache_start``) cases.
The query is a single token, so the reference is called with ``causal=True``
(equivalent to attending to all valid keys, window aligned to the last position).

Usage:
    python scripts/test_decode_triton.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import REFERENCE_SHAPES  # noqa: E402

from parallax import parallax_reference  # noqa: E402
from parallax.triton import parallax_decode  # noqa: E402


REL_TOL = 1e-2  # bf16 noise floor ~2-5e-3; same threshold as test_decode_swa.py.


def _make(B, H_q, H_kv, kv_len, D, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, 1, H_q, D, device="cuda", dtype=dtype, generator=g)
    r = torch.randn_like(q)
    k = torch.randn(B, kv_len, H_kv, D, device="cuda", dtype=dtype, generator=g)
    v = torch.randn_like(k)
    q = F.rms_norm(q.float(), (D,)).to(dtype).contiguous()
    r = F.rms_norm(r.float(), (D,)).to(dtype).contiguous()
    k = F.rms_norm(k.float(), (D,)).to(dtype).contiguous()
    return q, r, k, v.contiguous(), 1.0 / math.sqrt(D)


def _run(B, H_q, H_kv, kv_len, D, win, cache_start=None, seed=0):
    q, r, k, v, scale = _make(B, H_q, H_kv, kv_len, D, seed=seed)
    cs = None
    if cache_start is not None:
        cs = torch.tensor(cache_start, device="cuda", dtype=torch.long)

    out = parallax_decode(q, r, k, v, scale,
                          window_size_left=win, cache_start=cs)

    if cs is None:
        ref = parallax_reference(q, r, k, v, scale,
                                 causal=True, window_size_left=win)
    else:
        # Left-padding is exactly "drop keys before cache_start" — slice the
        # valid tail per row; the window stays measured from the last position.
        parts = []
        for b in range(B):
            lo = int(cs[b].item())
            parts.append(parallax_reference(
                q[b:b + 1], r[b:b + 1], k[b:b + 1, lo:], v[b:b + 1, lo:],
                scale, causal=True, window_size_left=win))
        ref = torch.cat(parts, dim=0)

    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    rel = max_abs / max(ref.float().abs().max().item(), 1e-6)
    nan = bool(torch.isnan(out).any().item())
    return max_abs, rel, nan


# (label, B, H_q, H_kv, kv_len, D, win, cache_start)
_CASES: list[tuple] = []
# Baseline shape sweep (MHA), SWA disabled.
for (B, K, H, D) in REFERENCE_SHAPES:
    _CASES.append(("shape", B, H, H, K, D, -1, None))
# Sliding-window sweep at D=128 (aligned / unaligned / >= kv_len / narrow).
for win in (-1, 64, 128, 256, 512, 1024, 33, 100, 200):
    _CASES.append(("swa", 1, 8, 8, 4096, 128, win, None))
_CASES += [
    ("swa", 1, 8, 8, 1024, 128, 4096, None),   # window >= kv_len
    ("swa", 1, 8, 8, 1000, 128, 300, None),    # partial first + last tile
    ("swa", 1, 8, 8, 65, 128, 1024, None),
    ("swa", 1, 1, 1, 16384, 128, 333, None),   # small B*H, large kv_len
    ("swa", 2, 4, 4, 8192, 128, 130, None),
    ("swa", 1, 8, 8, 4096, 64, 128, None),     # D=64
]
# GQA / MQA.
_CASES += [
    ("gqa", 4, 16, 4, 1024, 128, -1, None),
    ("gqa", 2, 8, 2, 2048, 128, 256, None),
    ("mqa", 1, 32, 1, 4096, 64, -1, None),
]
# Left-padding (cache_start): uniform, mid-tile, non-uniform per-row.
_CASES += [
    ("cstart", 2, 8, 8, 1024, 128, -1, [300, 300]),
    ("cstart", 2, 8, 8, 1024, 128, 128, [300, 300]),
    ("cstart", 1, 8, 8, 1024, 128, -1, [333]),          # mid-BS-tile start
    ("cstart", 4, 8, 2, 1024, 128, -1, [0, 256, 512, 700]),
    ("cstart", 4, 8, 2, 1024, 128, 128, [0, 256, 512, 700]),
]


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    print(f"GPU: {torch.cuda.get_device_name(0)}   tol(max-norm rel) < {REL_TOL:g}\n")
    hdr = (f"{'kind':>6} {'B':>2} {'H_q':>3} {'H_kv':>4} {'kv_len':>6} {'D':>3} "
           f"{'win':>5} {'cstart':>14}  {'max_abs':>10} {'max_rel':>10} {'nan':>4}")
    print(hdr)
    print("-" * len(hdr))

    fails = 0
    for (label, B, H_q, H_kv, kv_len, D, win, cs) in _CASES:
        try:
            max_abs, rel, nan = _run(B, H_q, H_kv, kv_len, D, win, cache_start=cs)
        except Exception as e:
            print(f"{label:>6} {B:>2} {H_q:>3} {H_kv:>4} {kv_len:>6} {D:>3} "
                  f"{win:>5} {str(cs):>14}  ERROR: {e!r}")
            fails += 1
            continue
        ok = (not nan) and (rel < REL_TOL)
        tag = "" if ok else "  <-- FAIL"
        cs_s = "-" if cs is None else (str(cs) if len(cs) <= 4 else f"[{len(cs)}]")
        print(f"{label:>6} {B:>2} {H_q:>3} {H_kv:>4} {kv_len:>6} {D:>3} "
              f"{win:>5} {cs_s:>14}  {max_abs:>10.3e} {rel:>10.3e} "
              f"{'T' if nan else 'F':>4}{tag}")
        if not ok:
            fails += 1

    print()
    if fails == 0:
        print(f"ALL {len(_CASES)} PASS")
        return 0
    print(f"{fails}/{len(_CASES)} FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
