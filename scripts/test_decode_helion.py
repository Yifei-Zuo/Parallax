"""Precision test for the Helion single-token decode kernel.

Mirrors ``scripts/test_decode_triton.py`` exactly (same ``_CASES``, same fp32
reference oracle, same ``REL_TOL = 1e-2``) but exercises
:func:`parallax.helion.parallax_decode`.

Usage:
    python scripts/test_decode_helion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_decode_triton import _CASES, _make, REL_TOL  # noqa: E402

from parallax import parallax_reference  # noqa: E402
from parallax.helion import parallax_decode  # noqa: E402


def _run(B, H_q, H_kv, kv_len, D, win, cache_start=None, seed=0):
    q, r, k, v, scale = _make(B, H_q, H_kv, kv_len, D, seed=seed)
    cs = None
    if cache_start is not None:
        cs = torch.tensor(cache_start, device="cuda", dtype=torch.long)

    out = parallax_decode(q, r, k, v, scale, window_size_left=win, cache_start=cs)

    if cs is None:
        ref = parallax_reference(q, r, k, v, scale, causal=True, window_size_left=win)
    else:
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
