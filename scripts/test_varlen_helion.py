"""Precision test for the Helion variable-length (packed) training kernel.

Exercises :func:`parallax.helion.parallax_varlen_func` end-to-end (autograd) on
packed multi-sequence inputs and compares the output + all four gradients against
autograd through the fp32 :func:`parallax.parallax_reference` applied per packed
sequence. Gate = q50 max-norm relative error < 1e-2. Covers MHA / GQA / MQA / SWA.

Usage:
    python scripts/test_varlen_helion.py
    python scripts/test_varlen_helion.py --nseq 6
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from parallax import parallax_reference
from parallax.helion import parallax_varlen_func

REL_TOL = 1e-2

# (H_q, H_kv, L_max, D, window_size_left)  — packed into one sequence (B=1).
_CASES = [
    (8, 8, 1024, 128, -1),    # MHA
    (8, 8, 1024, 64, -1),     # MHA, D=64
    (16, 4, 512, 128, -1),    # GQA 4:1
    (32, 1, 512, 64, -1),     # MQA
    (8, 8, 1024, 128, 256),   # SWA
    (8, 2, 512, 128, 128),    # GQA + SWA
    (16, 4, 512, 128, 64),    # GQA + SWA, tight window
]


def _make_packed(lens, H_q, H_kv, D, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    T = sum(lens)
    q = F.rms_norm(torch.randn(1, T, H_q, D, device=device, dtype=torch.bfloat16, generator=g).float(), (D,)).bfloat16().contiguous()
    r = F.rms_norm(torch.randn(1, T, H_q, D, device=device, dtype=torch.bfloat16, generator=g).float(), (D,)).bfloat16().contiguous()
    k = F.rms_norm(torch.randn(1, T, H_kv, D, device=device, dtype=torch.bfloat16, generator=g).float(), (D,)).bfloat16().contiguous()
    v = torch.randn(1, T, H_kv, D, device=device, dtype=torch.bfloat16, generator=g).contiguous()
    cu = F.pad(torch.tensor(lens, device=device).cumsum(0), (1, 0)).to(torch.int32)
    return q, r, k, v, cu


def _qmax(out, ref):
    rel = ((out.float() - ref.float()).abs() / max(ref.float().abs().max().item(), 1e-12)).flatten()
    return torch.quantile(rel, 0.5).item(), rel.max().item()


def _run(H_q, H_kv, L_max, D, W, nseq, seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(1, L_max + 1, (nseq,), generator=g).tolist()
    lens[0] = 1
    lens[-1] = L_max
    q, r, k, v, cu = _make_packed(lens, H_q, H_kv, D, seed=seed)
    scale = D ** -0.5
    qg, rg, kg, vg = (t.detach().requires_grad_(True) for t in (q, r, k, v))
    o = parallax_varlen_func(qg, rg, kg, vg, scale, window_size_left=W, cu_seqlens=cu)
    go = torch.randn(o.shape, device=o.device, dtype=o.dtype,
                     generator=torch.Generator(device=o.device).manual_seed(1))
    o.backward(go)

    o_ref = torch.empty_like(o, dtype=torch.float32)
    gq_r = torch.empty_like(q, dtype=torch.float32); gr_r = torch.empty_like(r, dtype=torch.float32)
    gk_r = torch.empty_like(k, dtype=torch.float32); gv_r = torch.empty_like(v, dtype=torch.float32)
    for i in range(nseq):
        bos, eos = int(cu[i].item()), int(cu[i + 1].item())
        qs = q[:, bos:eos].detach().float().requires_grad_(True)
        rs = r[:, bos:eos].detach().float().requires_grad_(True)
        ks = k[:, bos:eos].detach().float().requires_grad_(True)
        vs = v[:, bos:eos].detach().float().requires_grad_(True)
        os = parallax_reference(qs, rs, ks, vs, scale, causal=True, window_size_left=W)
        os.backward(go[:, bos:eos].float())
        o_ref[:, bos:eos] = os.detach()
        gq_r[:, bos:eos] = qs.grad; gr_r[:, bos:eos] = rs.grad
        gk_r[:, bos:eos] = ks.grad; gv_r[:, bos:eos] = vs.grad

    return {
        "o": _qmax(o, o_ref),
        "gq": _qmax(qg.grad, gq_r), "gr": _qmax(rg.grad, gr_r),
        "gk": _qmax(kg.grad, gk_r), "gv": _qmax(vg.grad, gv_r),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseq", type=int, default=4)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}   nseq={args.nseq}   tol(q50) < {REL_TOL:g}\n")
    hdr = (f"{'H_q':>3} {'H_kv':>4} {'L_max':>5} {'D':>3} {'W':>4} | "
           + " ".join(f"{n:>17}" for n in ("o", "gq", "gr", "gk", "gv")) + "  gate")
    print(hdr)
    print("-" * len(hdr))

    fails = 0
    for (H_q, H_kv, L_max, D, W) in _CASES:
        try:
            res = _run(H_q, H_kv, L_max, D, W, args.nseq)
        except Exception as e:
            print(f"{H_q:>3} {H_kv:>4} {L_max:>5} {D:>3} {W:>4} | ERROR: {e!r}")
            fails += 1
            continue
        gate = max(v[0] for v in res.values()) < REL_TOL
        cells = " ".join(f"{res[n][0]:.1e}|{res[n][1]:.1e}" for n in ("o", "gq", "gr", "gk", "gv"))
        print(f"{H_q:>3} {H_kv:>4} {L_max:>5} {D:>3} {W:>4} | {cells}  {'PASS' if gate else 'FAIL'}")
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
