"""Forward + backward parity check for the Parallax Triton training kernel.

Compares :func:`parallax.parallax_func` (bf16 Triton) against
:func:`parallax.parallax_reference` (fp32 PyTorch) across a sweep of
``(B, H_q, H_kv, L, D, window)`` shapes — covering MHA, GQA, MQA, and
sliding-window cases. For each shape it reports the max-norm relative
error (q50 + max) for the output and all four gradients.

Run:
    CUDA_VISIBLE_DEVICES=0 python scripts/parity_train.py
    CUDA_VISIBLE_DEVICES=0 python scripts/parity_train.py --shape 2,8,2,1024,128,256
    CUDA_VISIBLE_DEVICES=0 python scripts/parity_train.py --csv runs/parity.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

# Sibling-file imports (scripts/ is a flat directory, not a package).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parallax import parallax_func, parallax_reference  # noqa: E402


# Default sweep — MHA / GQA / MQA / SWA / GQA+SWA.
DEFAULT_CASES = [
    # (B, H_q, H_kv,  L,    D,  window_size_left)
    (1,  1,  1,   128,  64, -1),
    (1,  1,  1,   128, 128, -1),
    (2,  8,  8,  1024,  64, -1),
    (2,  8,  8,  1024, 128, -1),
    (1, 16,  4,   512, 128, -1),     # GQA 4:1
    (1, 32,  1,  1024,  64, -1),     # MQA
    (2,  8,  8,  1024, 128, 256),    # SWA
    (2,  8,  2,   512, 128, 128),    # GQA + SWA
    (1, 16,  4,   512, 128,  64),    # GQA + SWA, tight window
]


QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
ERROR = "ERROR"


def _make_inputs(B, H_q, H_kv, L, D, dtype=torch.bfloat16, device="cuda", seed=0):
    """RMS-normed bf16 tensors in (B, H, L, D) — parallax_func's convention."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, H_q, L, D, device=device, dtype=dtype, generator=g)
    r = torch.randn_like(q)
    k = torch.randn(B, H_kv, L, D, device=device, dtype=dtype, generator=g)
    v = torch.randn_like(k)
    q = F.rms_norm(q.float(), (D,)).to(dtype).contiguous()
    r = F.rms_norm(r.float(), (D,)).to(dtype).contiguous()
    k = F.rms_norm(k.float(), (D,)).to(dtype).contiguous()
    return q, r, k, v.contiguous()


def _rel_err(out, ref):
    """``|out - ref| / max|ref|`` quantiles + max. Returns 8-tuple."""
    if out is None or ref is None:
        return (float("nan"),) * 8
    diff = (out.float() - ref.float()).abs()
    ref_max = ref.float().abs().max().item()
    eps = 1e-12
    relerr = (diff / max(ref_max, eps)).flatten()
    qs = torch.tensor(QUANTILES, device=relerr.device, dtype=relerr.dtype)
    vals = torch.quantile(relerr, qs).tolist()
    max_rel = float(relerr.max().item())
    return (*vals, max_rel)


def _err_style(rel_err: float) -> str:
    """Color thresholds — bf16 noise floor is ~2-5e-3 for outputs, slightly
    higher for grads after accumulation. Tight threshold gets green, loose
    gets yellow, anything past 1e-1 is red."""
    if rel_err != rel_err:  # NaN
        return "dim"
    if rel_err < 1e-2:
        return "green"
    if rel_err < 1e-1:
        return "yellow"
    return "bold red"


def _check_one(B, H_q, H_kv, L, D, window_size_left):
    """Run parallax_func + parallax_reference once, return per-tensor rel-err."""
    qk_scale = D ** -0.5
    q, r, k, v = _make_inputs(B, H_q, H_kv, L, D)
    q.requires_grad_(True); r.requires_grad_(True)
    k.requires_grad_(True); v.requires_grad_(True)

    # Triton path.
    o_triton = parallax_func(q, r, k, v, qk_scale,
                             window_size_left=window_size_left)
    grad_o = torch.randn_like(o_triton)
    o_triton.backward(grad_o)
    gq_t = q.grad.clone(); gr_t = r.grad.clone()
    gk_t = k.grad.clone(); gv_t = v.grad.clone()

    # Reference path — fresh fp32 leaves, permute to FA convention (B, L, H, D).
    q2 = q.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    r2 = r.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    k2 = k.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    v2 = v.detach().permute(0, 2, 1, 3).contiguous().float().requires_grad_(True)
    o_ref = parallax_reference(q2, r2, k2, v2, qk_scale,
                               causal=True, window_size_left=window_size_left)
    # Output back to (B, H_q, L, D) to compare against o_triton.
    o_ref_t = o_ref.permute(0, 2, 1, 3)
    o_ref_t.backward(grad_o.float())
    gq_r = q2.grad.permute(0, 2, 1, 3)
    gr_r = r2.grad.permute(0, 2, 1, 3)
    gk_r = k2.grad.permute(0, 2, 1, 3)
    gv_r = v2.grad.permute(0, 2, 1, 3)

    return {
        "o":  _rel_err(o_triton, o_ref_t),
        "gq": _rel_err(gq_t, gq_r),
        "gr": _rel_err(gr_t, gr_r),
        "gk": _rel_err(gk_t, gk_r),
        "gv": _rel_err(gv_t, gv_r),
    }


def _make_table():
    t = Table(
        box=box.MINIMAL_HEAVY_HEAD,
        show_header=True,
        header_style="bold",
        title="Parallax training-kernel parity (bf16 Triton vs fp32 reference)",
        title_style="bold cyan",
        caption="cell = q50 rel err (max-norm).  "
                "[green]green[/green] < 1e-2,  [yellow]yellow[/yellow] < 1e-1,  "
                "[bold red]red[/bold red] >= 1e-1.",
        caption_style="dim",
    )
    for col in ("B", "H_q", "H_kv", "L", "D", "W"):
        t.add_column(col, justify="right", style="dim", no_wrap=True)
    for col in ("o", "grad_q", "grad_r", "grad_k", "grad_v"):
        t.add_column(col, justify="right", no_wrap=True)
    return t


def parse_shape(s):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"shape must be B,H_q,H_kv,L,D,window — got {s!r}"
        )
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=parse_shape, action="append",
                    help="B,H_q,H_kv,L,D,window — repeatable. "
                         "window=-1 disables SWA. Default: built-in sweep.")
    ap.add_argument("--csv", type=Path,
                    help="Optional CSV dump (full quantiles for every tensor).")
    ap.add_argument("--single-config", action="store_true",
                    help="Bypass triton autotune — use a single known-good config "
                         "(ROW=128, COL=64, num_warps=4, num_stages=2). Skips the "
                         "lengthy first-time autotune compile sweep.")
    args = ap.parse_args()

    if args.single_config:
        import importlib
        import triton
        cfg = [triton.Config({"ROW_TILE_SIZE": 64, "COL_TILE_SIZE": 64},
                             num_warps=4, num_stages=2)]
        pre_cfg = [triton.Config({"ROW_TILE_SIZE": 64},
                                 num_warps=4, num_stages=2)]
        fwd_mod = importlib.import_module("parallax.triton.parallax_fwd")
        bwd_mod = importlib.import_module("parallax.triton.parallax_bwd")
        fwd_mod._fwd_kernel.configs = cfg
        bwd_mod._bwd_rq_kernel.configs = cfg
        bwd_mod._bwd_kv_kernel.configs = cfg
        bwd_mod._bwd_preprocess_kernel.configs = pre_cfg

    cases = args.shape if args.shape else DEFAULT_CASES
    console = Console()
    console.rule("[bold cyan]Parallax training-kernel parity[/bold cyan]")
    console.print(
        f"GPU: [bold]{torch.cuda.get_device_name(0)}[/bold]   "
        f"shapes=[bold]{len(cases)}[/bold]   "
        f"reference=[bold]fp32 PyTorch (parallax_reference)[/bold]",
    )
    console.print()

    table = _make_table()
    csv_rows = []

    def _err_cell(err_tuple):
        q50 = err_tuple[3]
        return f"[{_err_style(q50)}]{q50:.2e}[/]"

    with Live(table, console=console, refresh_per_second=4, vertical_overflow="visible"):
        for (B, H_q, H_kv, L, D, W) in cases:
            try:
                errs = _check_one(B, H_q, H_kv, L, D, W)
            except Exception as e:
                # Render the failing row in red and continue.
                row = [str(B), str(H_q), str(H_kv), str(L), str(D),
                       str(W) if W >= 0 else "—"]
                row += [f"[bold red]{ERROR}[/bold red]"] * 5
                table.add_row(*row)
                console.print(f"  [red]{ERROR}[/red] at shape "
                              f"(B={B}, H_q={H_q}, H_kv={H_kv}, L={L}, D={D}, W={W}): {e!r}",
                              style="dim")
                continue

            row = [str(B), str(H_q), str(H_kv), str(L), str(D),
                   str(W) if W >= 0 else "—"]
            for key in ("o", "gq", "gr", "gk", "gv"):
                row.append(_err_cell(errs[key]))
            table.add_row(*row)

            # CSV row per tensor.
            for tensor_name, err_tuple in errs.items():
                eq05, eq10, eq25, eq50, eq75, eq90, eq95, emax = err_tuple
                csv_rows.append(dict(
                    B=B, H_q=H_q, H_kv=H_kv, L=L, D=D, window=W,
                    tensor=tensor_name,
                    q05=eq05, q10=eq10, q25=eq25, q50=eq50,
                    q75=eq75, q90=eq90, q95=eq95, max=emax,
                ))

            gc.collect(); torch.cuda.empty_cache()

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["B", "H_q", "H_kv", "L", "D", "window", "tensor",
                  "q05", "q10", "q25", "q50", "q75", "q90", "q95", "max"]
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in csv_rows:
                w.writerow({k: row[k] for k in fields})
        console.print()
        console.print(f"[green]✓[/green] wrote [bold]{args.csv}[/bold]  "
                      f"({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
