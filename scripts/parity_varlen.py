"""Forward + backward parity check for :func:`parallax.parallax_varlen_func`.

Across a sweep of ``(B, H_q, H_kv, L_max, D, window)``, compares the packed
varlen kernel (bf16 Triton) against two references: (1) the fp32
:func:`parallax.parallax_reference` applied per packed sequence (output + all
four gradients), and (2) the same kernel run per-sequence on the dense path
(``cu_seqlens=None``), which must match the packed path exactly.

Run:
    python scripts/parity_varlen.py
    python scripts/parity_varlen.py --shape 1,8,2,1024,128,256 --nseq 6
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from parallax import parallax_reference, parallax_varlen_func


# Default sweep — (B, H_q, H_kv, L_max, D, window_size_left).
DEFAULT_CASES = [
    (1,  8,  8, 1024, 128, -1),    # MHA
    (1,  8,  8, 1024,  64, -1),    # MHA, D=64 (Hopper 128-row tile)
    (1, 16,  4,  512, 128, -1),    # GQA 4:1
    (1, 32,  1,  512,  64, -1),    # MQA
    (1,  8,  8, 1024, 128, 256),   # SWA
    (1,  8,  2,  512, 128, 128),   # GQA + SWA
    (1, 16,  4,  512, 128,  64),   # GQA + SWA, tight window
]

REL_TOL = 1e-2  # bf16 noise floor is ~2-5e-3; matches parity_train.py.


def _seqlens(nseq: int, l_max: int, seed: int) -> list[int]:
    """Random per-sequence lengths in [1, l_max], with the edge cases pinned."""
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(1, l_max + 1, (nseq,), generator=g).tolist()
    lens[0] = 1                      # single-token sequence
    lens[-1] = l_max                 # full-length sequence
    return lens


def _make_packed(lens, H_q, H_kv, D, dtype=torch.bfloat16, device="cuda", seed=0):
    """RMS-normed packed bf16 tensors (1, sum(lens), H, D) + int32 cu_seqlens."""
    g = torch.Generator(device=device).manual_seed(seed)
    T = sum(lens)
    q = torch.randn(1, T, H_q, D, device=device, dtype=dtype, generator=g)
    r = torch.randn_like(q)
    k = torch.randn(1, T, H_kv, D, device=device, dtype=dtype, generator=g)
    v = torch.randn_like(k)
    q = F.rms_norm(q.float(), (D,)).to(dtype).contiguous()
    r = F.rms_norm(r.float(), (D,)).to(dtype).contiguous()
    k = F.rms_norm(k.float(), (D,)).to(dtype).contiguous()
    v = v.contiguous()
    cu = F.pad(torch.tensor(lens, device=device).cumsum(0), (1, 0)).to(torch.int32)
    return q, r, k, v, cu


def _rel(out, ref):
    """``|out - ref| / max|ref|`` — returns (q50, max)."""
    diff = (out.float() - ref.float()).abs()
    relerr = (diff / max(ref.float().abs().max().item(), 1e-12)).flatten()
    return torch.quantile(relerr, 0.5).item(), relerr.max().item()


def _check_one(B, H_q, H_kv, L_max, D, W, nseq, seed):
    qk_scale = D ** -0.5
    lens = _seqlens(nseq, L_max, seed)
    q, r, k, v, cu = _make_packed(lens, H_q, H_kv, D, seed=seed)

    qg = q.detach().requires_grad_(True)
    rg = r.detach().requires_grad_(True)
    kg = k.detach().requires_grad_(True)
    vg = v.detach().requires_grad_(True)
    o = parallax_varlen_func(qg, rg, kg, vg, qk_scale, window_size_left=W, cu_seqlens=cu)
    grad_o = torch.randn_like(o)
    o.backward(grad_o)
    gq, gr, gk, gv = qg.grad, rg.grad, kg.grad, vg.grad

    # ── Gold: per-sequence fp32 reference, concatenated. ─────────────────────
    o_ref = torch.empty_like(o, dtype=torch.float32)
    gq_ref = torch.empty_like(q, dtype=torch.float32)
    gr_ref = torch.empty_like(r, dtype=torch.float32)
    gk_ref = torch.empty_like(k, dtype=torch.float32)
    gv_ref = torch.empty_like(v, dtype=torch.float32)
    # ── Cross-check: per-sequence dense Triton (cu_seqlens=None). ────────────
    o_dense = torch.empty_like(o)
    gq_dn = torch.empty_like(q); gr_dn = torch.empty_like(r)
    gk_dn = torch.empty_like(k); gv_dn = torch.empty_like(v)

    for i in range(nseq):
        bos, eos = int(cu[i].item()), int(cu[i + 1].item())
        go_s = grad_o[:, bos:eos]

        qs = q[:, bos:eos].detach().float().requires_grad_(True)
        rs = r[:, bos:eos].detach().float().requires_grad_(True)
        ks = k[:, bos:eos].detach().float().requires_grad_(True)
        vs = v[:, bos:eos].detach().float().requires_grad_(True)
        os = parallax_reference(qs, rs, ks, vs, qk_scale, causal=True, window_size_left=W)
        os.backward(go_s.float())
        o_ref[:, bos:eos] = os.detach()
        gq_ref[:, bos:eos] = qs.grad; gr_ref[:, bos:eos] = rs.grad
        gk_ref[:, bos:eos] = ks.grad; gv_ref[:, bos:eos] = vs.grad

        qd = q[:, bos:eos].detach().requires_grad_(True)
        rd = r[:, bos:eos].detach().requires_grad_(True)
        kd = k[:, bos:eos].detach().requires_grad_(True)
        vd = v[:, bos:eos].detach().requires_grad_(True)
        od = parallax_varlen_func(qd, rd, kd, vd, qk_scale, window_size_left=W)
        od.backward(go_s)
        o_dense[:, bos:eos] = od.detach()
        gq_dn[:, bos:eos] = qd.grad; gr_dn[:, bos:eos] = rd.grad
        gk_dn[:, bos:eos] = kd.grad; gv_dn[:, bos:eos] = vd.grad

    ref_errs = {
        "o":  _rel(o, o_ref),  "gq": _rel(gq, gq_ref), "gr": _rel(gr, gr_ref),
        "gk": _rel(gk, gk_ref), "gv": _rel(gv, gv_ref),
    }
    # Single worst-case number for the varlen-vs-dense kernel cross-check.
    xcheck = max(
        _rel(o, o_dense)[1], _rel(gq, gq_dn)[1], _rel(gr, gr_dn)[1],
        _rel(gk, gk_dn)[1], _rel(gv, gv_dn)[1],
    )
    return ref_errs, xcheck, lens


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=lambda s: tuple(int(x) for x in s.split(",")),
                    action="append",
                    help="B,H_q,H_kv,L_max,D,window — repeatable. B must be 1.")
    ap.add_argument("--nseq", type=int, default=4, help="packed sequences per case.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cases = args.shape if args.shape else DEFAULT_CASES

    print(f"GPU: {torch.cuda.get_device_name(0)}   reference=parallax_reference (fp32)")
    print(f"nseq={args.nseq}   tol(q50 vs ref) < {REL_TOL:g}\n")
    hdr = (f"{'H_q':>3} {'H_kv':>4} {'L_max':>5} {'D':>3} {'W':>4}  "
           f"{'o':>9} {'gq':>9} {'gr':>9} {'gk':>9} {'gv':>9}  {'vs-dense':>9}")
    print(hdr)
    print("-" * len(hdr))

    fails = 0
    for (B, H_q, H_kv, L_max, D, W) in cases:
        try:
            ref_errs, xcheck, lens = _check_one(B, H_q, H_kv, L_max, D, W, args.nseq, args.seed)
        except Exception as e:
            print(f"{H_q:>3} {H_kv:>4} {L_max:>5} {D:>3} {W:>4}  ERROR: {e!r}")
            fails += 1
            continue
        q50s = {kk: v[0] for kk, v in ref_errs.items()}
        ok = all(v < REL_TOL for v in q50s.values()) and xcheck < REL_TOL
        tag = "" if ok else "  <-- FAIL"
        print(f"{H_q:>3} {H_kv:>4} {L_max:>5} {D:>3} {W:>4}  "
              f"{q50s['o']:>9.2e} {q50s['gq']:>9.2e} {q50s['gr']:>9.2e} "
              f"{q50s['gk']:>9.2e} {q50s['gv']:>9.2e}  {xcheck:>9.2e}{tag}")
        if not ok:
            fails += 1

    print()
    if fails == 0:
        print(f"ALL {len(cases)} PASS")
        return 0
    print(f"{fails}/{len(cases)} FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
