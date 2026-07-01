"""Precision test for the Helion training kernel (forward + backward).

Exercises :func:`parallax.helion.parallax_func` end-to-end (autograd) and
compares the output plus all four gradients against autograd through the fp32
:func:`parallax.parallax_reference`. Gate = q50 max-norm relative error < 1e-2
(the repo convention; the bf16 grad floor is ~3-6e-3). Covers MHA / GQA / MQA /
SWA and head_dim 64/128 (which routes through the split forward).

Usage:
    python scripts/test_train_helion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_train import _make_inputs  # noqa: E402

from parallax import parallax_reference  # noqa: E402
from parallax.helion import parallax_func  # noqa: E402

REL_TOL = 1e-2

# (B, H_q, H_kv, L, D, window_size_left)
_CASES = [
    (2, 8, 8, 1024, 128, -1),
    (2, 8, 8, 4096, 128, -1),
    (4, 8, 8, 2048, 128, -1),
    (1, 16, 4, 2048, 128, -1),   # GQA 4:1
    (1, 32, 1, 2048, 128, -1),   # MQA
    (2, 8, 8, 1024, 64, -1),     # D=64 (split-forward route)
    (2, 8, 8, 2048, 128, 256),   # SWA
]


def _qmax(out, ref):
    rel = ((out.float() - ref.float()).abs()
           / max(ref.float().abs().max().item(), 1e-12)).flatten()
    return torch.quantile(rel, 0.5).item(), rel.max().item()


def _run(B, H_q, H_kv, L, D, win, seed=0):
    q, r, k, v = _make_inputs(B, H_q, H_kv, L, D, seed=seed)
    for t in (q, r, k, v):
        t.requires_grad_(True)
    scale = D ** -0.5

    o = parallax_func(q, r, k, v, scale, window_size_left=win)
    go = torch.randn(o.shape, device=o.device, dtype=o.dtype,
                     generator=torch.Generator(device=o.device).manual_seed(1))
    o.backward(go)

    q2 = q.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    r2 = r.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    k2 = k.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    v2 = v.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    o_ref = parallax_reference(q2, r2, k2, v2, scale, causal=True,
                               window_size_left=win).permute(0, 2, 1, 3)
    o_ref.backward(go.float())

    tensors = {
        "o": (o, o_ref),
        "gq": (q.grad, q2.grad.permute(0, 2, 1, 3)),
        "gr": (r.grad, r2.grad.permute(0, 2, 1, 3)),
        "gk": (k.grad, k2.grad.permute(0, 2, 1, 3)),
        "gv": (v.grad, v2.grad.permute(0, 2, 1, 3)),
    }
    return {name: _qmax(a, b) for name, (a, b) in tensors.items()}


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    print(f"GPU: {torch.cuda.get_device_name(0)}   tol(q50 max-norm rel) < {REL_TOL:g}\n")
    hdr = (f"{'B':>2} {'H_q':>3} {'H_kv':>4} {'L':>5} {'D':>3} {'W':>4} | "
           + " ".join(f"{n:>17}" for n in ("o", "gq", "gr", "gk", "gv")) + "  gate")
    print(hdr)
    print("-" * len(hdr))

    fails = 0
    for (B, H_q, H_kv, L, D, win) in _CASES:
        try:
            res = _run(B, H_q, H_kv, L, D, win)
        except Exception as e:
            print(f"{B:>2} {H_q:>3} {H_kv:>4} {L:>5} {D:>3} {win:>4} | ERROR: {e!r}")
            fails += 1
            continue
        gate = max(v[0] for v in res.values()) < REL_TOL
        cells = " ".join(f"{res[n][0]:.1e}|{res[n][1]:.1e}" for n in ("o", "gq", "gr", "gk", "gv"))
        print(f"{B:>2} {H_q:>3} {H_kv:>4} {L:>5} {D:>3} {win:>4} | {cells}  "
              f"{'PASS' if gate else 'FAIL'}")
        if not gate:
            fails += 1

    print()
    if fails == 0:
        print(f"ALL {len(_CASES)} PASS  (cell = q50|max)")
        return 0
    print(f"{fails}/{len(_CASES)} FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
