"""Precision test for sliding-window attention in ``parallax_decode``.

Compares the CuTeDSL decode kernel against the fp32 PyTorch reference
(``parallax.parallax_reference``) across baseline + SWA-active shapes,
including the split-K path. The reference's ``window_size_left``
follows the FA2 convention: ``-1`` disables SWA; ``>= 0`` restricts the
decode query to the most recent ``window_size_left`` keys.

Usage:
    python scripts/test_decode_swa.py
"""

from __future__ import annotations

import math
import sys

import torch

from parallax import parallax_decode, parallax_reference


def _run(B: int, H: int, D: int, kv_len: int, window_size_left: int,
         dtype: torch.dtype = torch.bfloat16, seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn(B, 1, H, D, device="cuda", dtype=dtype)
    r = torch.randn_like(q) * 0.5
    k = torch.randn(B, kv_len, H, D, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(D)

    out_cute = parallax_decode(
        q, r, k, v, scale, window_size_left=window_size_left
    )
    out_ref = parallax_reference(
        q, r, k, v, scale, causal=True, window_size_left=window_size_left
    )

    diff = (out_cute.float() - out_ref.float()).abs()
    max_abs = diff.max().item()
    ref_scale = out_ref.float().abs().max().item()
    # bf16 noise floor is best gauged against the output magnitude (the
    # composite formula O1/d1 * (1 + d2/d1) - O2/d1 cancels heavily, so
    # element-wise relative error blows up near zeros without telling us
    # anything useful).
    rel = max_abs / max(ref_scale, 1e-6)
    has_nan_cute = bool(torch.isnan(out_cute).any().item())
    has_nan_ref = bool(torch.isnan(out_ref).any().item())
    return max_abs, rel, has_nan_cute, has_nan_ref


# Each case: (B, H, kv_len, window_size_left). D = 128.
_CASES = [
    # ---- baseline (SWA disabled) ----
    (1, 8, 512, -1),
    (1, 8, 1024, -1),
    (1, 8, 4096, -1),
    (1, 8, 16384, -1),
    # ---- aligned window ----
    (1, 8, 4096, 128),
    (1, 8, 4096, 256),
    (1, 8, 4096, 512),
    (1, 8, 4096, 1024),
    (1, 8, 4096, 64),
    # ---- unaligned window (first-tile partial mask) ----
    (1, 8, 4096, 33),
    (1, 8, 4096, 100),
    (1, 8, 4096, 200),
    # ---- window >= kv_len (SWA effectively disabled) ----
    (1, 8, 1024, 4096),
    # ---- partial first tile + partial last tile ----
    (1, 8, 1000, 128),
    (1, 8, 1000, 300),
    (1, 8, 65, 1024),
    (1, 8, 100, 64),
    # ---- narrow window ----
    (1, 8, 8192, 65),
    # ---- split-K path: small B*H, large kv_len, with SWA ----
    (1, 1, 16384, -1),
    (1, 1, 16384, 4096),
    (1, 1, 16384, 2048),
    (1, 1, 16384, 1000),
    (1, 1, 16384, 333),
    (2, 4, 8192, 1024),
    (2, 4, 8192, 130),
]


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    # rel-error tolerance: bf16 noise floor sits at 2-5e-3 against fp32
    # ref across the cases below; 1e-2 leaves headroom for the composite
    # formula O1/d1 * (1 + d2/d1) - O2/d1, which cancels heavily and
    # amplifies bf16 rounding when the surviving KV count is very small.
    REL_TOL = 1e-2

    print(f"{'B':>2} {'H':>2} {'kv_len':>6}  {'win':>5}  "
          f"{'max_abs':>10}  {'max_rel':>10}  {'nan?':>6}")
    print("-" * 60)
    fails = 0
    for B, H, kv_len, win in _CASES:
        max_abs, max_rel, nan_c, nan_r = _run(B, H, 128, kv_len, win)
        ok = (not nan_c) and (max_rel < REL_TOL)
        tag = "" if ok else "  <-- FAIL"
        nan_tag = f"C={'T' if nan_c else 'F'},R={'T' if nan_r else 'F'}"
        print(f"{B:>2} {H:>2} {kv_len:>6}  {win:>5}  "
              f"{max_abs:>10.3e}  {max_rel:>10.3e}  {nan_tag:>6}{tag}")
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
