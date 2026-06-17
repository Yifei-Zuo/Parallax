"""Benchmark the Parallax SM90 decode kernel against FA2 / (optional) FA3.

Backends:

  - fa-decode     ``flash_attn.flash_attn_with_kvcache``             — FA2 C++ kvcache.
  - fa3-decode    ``flash_attn_interface.flash_attn_with_kvcache``   — FA3 hopper kvcache;
                                                                       opt-in via ``--include-fa3``.
  - parallax-cute ``parallax.parallax_decode``                       — the SM90 CuTeDSL kernel.

For every shape this script runs **both** an end-to-end precision pass and a
speed pass:

  * **Precision** — each backend is invoked once and compared to its fp32
    reference (pure softmax for FA2/FA3; :func:`parallax.parallax_reference`
    for parallax-cute). The reported error is the per-element relative
    error distribution normalised by ``max|ref|`` (avoids the near-zero
    singularity of per-element division). Reported as quantiles q05..q95
    plus max. With bf16 inputs the noise floor lands at ~2-4e-3.

  * **Speed** — each backend is captured into a CUDA graph and replayed
    so the per-call number matches what a CUDA-graphed inference engine
    (vLLM / SGLang / TRT-LLM) would see. Reported as quantiles
    q05..q95 over ``iters * trials`` per-call samples.

Run:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_decode.py
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_decode.py --include-fa3
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_decode.py --shape 4,1024,8,128 \\
                                                          --shape 4,4096,16,128
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_decode.py --parallax-grid \\
                                                          --include-fa3 \\
                                                          --csv /tmp/parallax_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

# torch must come first so libc10.so is on the linker path before flash_attn_3._C.
import torch  # noqa: F401

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

# ── Optional baselines (graceful absence: backend → NOT-AVAIL column) ─────
try:
    from flash_attn import flash_attn_with_kvcache as _real_fa2
    HAS_FA2 = True
except ImportError:
    _real_fa2 = None
    HAS_FA2 = False

try:
    from flash_attn_interface import flash_attn_with_kvcache as _real_fa3
    HAS_FA3 = True
except ImportError:
    _real_fa3 = None
    HAS_FA3 = False

# Sibling-file imports (scripts/ is a flat directory, not a package).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _test_utils import (  # noqa: E402
    REFERENCE_SHAPES as DEFAULT_CASES,
    make_inputs,
    parallax_grid,
)

from parallax import parallax_decode, parallax_reference  # noqa: E402
from parallax.triton import parallax_decode as parallax_decode_triton  # noqa: E402


# ── References ──────────────────────────────────────────────────────────
def _ref_std_attn(q, k, v, scale):
    """fp32 pure-softmax attention reference (for the FA2/FA3 comparison)."""
    qf = q.float(); kf = k.float(); vf = v.float()
    s = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", p, vf)


# ── Quantile profile ────────────────────────────────────────────────────
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

NOT_AVAIL = "NOT-AVAIL"
NOT_IMPL  = "NOT-IMPL"
ERROR     = "ERROR"


@dataclass
class QStats:
    q05_ms: float
    q10_ms: float
    q25_ms: float
    q50_ms: float
    q75_ms: float
    q90_ms: float
    q95_ms: float
    n: int


def _qstats(samples) -> QStats:
    s = sorted(samples)
    n = len(s)

    def _q(p):
        if n == 0:
            return float("nan")
        idx = max(0, min(n - 1, int(p * (n - 1))))
        return s[idx]

    return QStats(
        q05_ms=_q(0.05), q10_ms=_q(0.10), q25_ms=_q(0.25),
        q50_ms=_q(0.50), q75_ms=_q(0.75), q90_ms=_q(0.90),
        q95_ms=_q(0.95),
        n=n,
    )


def _l2_flush_buffer():
    return torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.float32)


def _time_fn(fn, warmup, iters, trials, l2_flush_buf) -> QStats:
    """CUDA-graph-replay timing.

    Captures ``iters_in_graph`` back-to-back ``fn()`` calls in a single CUDA
    graph (warm path) or 1 call + L2 flush (cold path), then replays
    ``n_replays`` times. Each replay produces one per-call latency sample.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    cold = l2_flush_buf is not None
    iters_in_graph = 1 if cold else iters
    n_replays = iters * trials if cold else trials

    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            fn()
        capture_stream.synchronize()
        with torch.cuda.graph(g, stream=capture_stream):
            for _ in range(iters_in_graph):
                fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    samples = []
    for _ in range(n_replays):
        if cold:
            l2_flush_buf.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) / iters_in_graph)
    return _qstats(samples)


def _err(out, ref):
    """Per-element relative-error quantile tuple (q05..q95, max).

    Relative error is ``|out - ref| / max|ref|`` (one distribution per
    output tensor, normalised by the global reference magnitude).
    ``max|ref|`` rather than per-element division avoids the singularity
    at near-zero reference entries.

    With bf16 inputs the noise floor lands at ~2-4e-3 — this is reported,
    not asserted; the operator reads the columns.
    """
    nan8 = (float("nan"),) * 8
    if out is None or ref is None:
        return nan8
    diff = (out.float() - ref).abs()
    ref_max = ref.abs().max().item()
    eps = 1e-12
    relerr = (diff / max(ref_max, eps)).flatten()
    qs = torch.tensor(QUANTILES, device=relerr.device, dtype=relerr.dtype)
    vals = torch.quantile(relerr, qs).tolist()
    max_rel = float(relerr.max().item())
    return (*vals, max_rel)


def _try_time(name, fn, warmup, iters, trials, buf, error_log):
    try:
        _probe = fn()
        del _probe
    except NotImplementedError:
        return NOT_IMPL
    except Exception as e:
        error_log.setdefault(name, repr(e))
        return ERROR
    return _time_fn(fn, warmup, iters, trials, buf)


def bench_one(batch, kv_len, nheads, head_dim, *, warmup, iters, trials,
              l2_flush, include_fa3, error_log):
    q, r, k, v, scale = make_inputs(batch, kv_len, nheads, head_dim)

    buf = _l2_flush_buffer() if l2_flush else None
    results: dict[str, object] = {}
    errs: dict[str, tuple] = {}

    # ── e2e precision pass ───────────────────────────────────────────────
    with torch.no_grad():
        need_std_ref = HAS_FA2 or (include_fa3 and HAS_FA3)
        ref_std = _ref_std_attn(q, k, v, scale) if need_std_ref else None
        # Q == 1 → causal mask is a no-op for the decode reference.
        ref_plx = parallax_reference(q, r, k, v, scale, causal=False).float()

        def _probe_err(name, fn, ref):
            try:
                out = fn()
            except Exception as e:
                error_log.setdefault(name, repr(e))
                return (float("nan"),) * 8
            err = _err(out, ref)
            del out
            return err

        if HAS_FA2:
            errs["fa-decode"] = _probe_err(
                "fa-decode",
                lambda: _real_fa2(q, k, v, softmax_scale=scale, causal=True),
                ref_std,
            )
        if include_fa3 and HAS_FA3:
            errs["fa3-decode"] = _probe_err(
                "fa3-decode",
                lambda: _real_fa3(q, k, v, softmax_scale=scale, causal=True),
                ref_std,
            )
        errs["parallax-cute"] = _probe_err(
            "parallax-cute",
            lambda: parallax_decode(q, r, k, v, scale),
            ref_plx,
        )
        errs["parallax-triton"] = _probe_err(
            "parallax-triton",
            lambda: parallax_decode_triton(q, r, k, v, scale),
            ref_plx,
        )
        del ref_std, ref_plx
    torch.cuda.synchronize()

    # ── timing pass ──────────────────────────────────────────────────────
    if HAS_FA2:
        fn_fa2 = lambda: _real_fa2(q, k, v, softmax_scale=scale, causal=True)
        results["fa-decode"] = _try_time("fa-decode", fn_fa2,
                                         warmup, iters, trials, buf, error_log)
    else:
        results["fa-decode"] = NOT_AVAIL

    if include_fa3:
        if HAS_FA3:
            fn_fa3 = lambda: _real_fa3(q, k, v, softmax_scale=scale, causal=True)
            results["fa3-decode"] = _try_time("fa3-decode", fn_fa3,
                                              warmup, iters, trials, buf, error_log)
        else:
            results["fa3-decode"] = NOT_AVAIL

    fn_plx = lambda: parallax_decode(q, r, k, v, scale)
    results["parallax-cute"] = _try_time("parallax-cute", fn_plx,
                                         warmup, iters, trials, buf, error_log)

    fn_plx_tri = lambda: parallax_decode_triton(q, r, k, v, scale)
    results["parallax-triton"] = _try_time("parallax-triton", fn_plx_tri,
                                           warmup, iters, trials, buf, error_log)

    return results, errs


def _fmt_speed(val) -> str:
    """Render a QStats cell as 'q50 (q05-q95)' or the error string."""
    if isinstance(val, str):
        return val
    return f"{val.q50_ms:.3f} ({val.q05_ms:.3f}-{val.q95_ms:.3f})"


def _err_style(rel_err: float) -> str:
    """Color thresholds for parallax-cute's q50 relative error.

    bf16 noise floor lives at ~2-4e-3. Cross 1e-2 and we want a soft warning;
    cross 1e-1 and something is genuinely off.
    """
    if rel_err != rel_err:  # NaN
        return "dim"
    if rel_err < 1e-2:
        return "green"
    if rel_err < 1e-1:
        return "yellow"
    return "bold red"


def _fastest_backend(res, backends):
    """Pick the backend with the lowest q50 ms among those that produced a QStats."""
    times = {b: res[b].q50_ms for b in backends if not isinstance(res[b], str)}
    if not times:
        return None
    return min(times, key=times.get)


def _make_table(backends, mode_str):
    t = Table(
        box=box.MINIMAL_HEAVY_HEAD,
        show_header=True,
        header_style="bold",
        title=f"Decode latency  ({mode_str})",
        title_style="bold cyan",
        caption="ms cell = median (q05-q95) over CUDA-graph replays.\n"
                "[green]green[/green] = fastest in row, [bold red]red rel err[/bold red] ≥ 1e-1",
        caption_style="dim",
    )
    for col in ("B", "K", "H", "D"):
        t.add_column(col, justify="right", style="dim", no_wrap=True)
    pretty = {
        "fa-decode":       "FA2 (ms)",
        "fa3-decode":      "FA3 (ms)",
        "parallax-cute":   "PLX (ms)",
        "parallax-triton": "PLX-T (ms)",
    }
    for b in backends:
        t.add_column(pretty[b], justify="right", no_wrap=True)
    t.add_column("PLX rel-err", justify="right", no_wrap=True)
    t.add_column("PLX-T rel-err", justify="right", no_wrap=True)
    return t


def parse_shape(s):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"shape must be B,K,H,D — got {s!r}")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=parse_shape, action="append",
                    help="B,K,H,D - repeatable. Default: built-in REFERENCE_SHAPES.")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters",  type=int, default=30)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--l2-flush", action="store_true",
                    help="Cold-cache mode: zero a 256MiB buffer between replays.")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--include-fa3", action="store_true",
                    help="Also bench FA3 hopper kvcache (flash_attn_3, source-built).")
    ap.add_argument("--parallax-grid", action="store_true",
                    help="216-shape Parallax dedup grid (BH x K x D).")
    args = ap.parse_args()

    if args.parallax_grid:
        args.shape = [tuple(s) for s in parallax_grid()]
    cases = args.shape if args.shape else DEFAULT_CASES

    if args.include_fa3 and not HAS_FA3:
        print("warning: --include-fa3 set but flash_attn_interface is not installed; "
              "fa3-decode column will print NOT-AVAIL.")

    backends = ["fa-decode"]
    if args.include_fa3:
        backends.append("fa3-decode")
    backends.append("parallax-cute")
    backends.append("parallax-triton")

    console = Console()
    cache_mode = "cold" if args.l2_flush else "warm"
    console.rule(f"[bold cyan]Parallax decode benchmark")
    console.print(
        f"GPU: [bold]{torch.cuda.get_device_name(0)}[/bold]\n"
        f"warmup={args.warmup}, iters={args.iters}, trials={args.trials}, n={args.iters * args.trials}, cache=[bold]{cache_mode}[/bold]\n"
        f"backends=[bold]{', '.join(backends)}[/bold]"
    )
    console.print(f"HAS_FA2={HAS_FA2}, HAS_FA3={HAS_FA3}", style="dim")
    console.print()

    error_log: dict = {}
    with console.status(
        f"[cyan]global warmup[/cyan] — {len(cases)} shapes x {len(backends)} backends ...",
        spinner="dots",
    ):
        for b, k, h, d in cases:
            bench_one(b, k, h, d, warmup=5, iters=1, trials=1,
                      l2_flush=False, include_fa3=args.include_fa3, error_log=error_log)
            gc.collect(); torch.cuda.empty_cache()

    mode_str = (f"warmup={args.warmup}, iters={args.iters}, trials={args.trials}, "
                f"cache={cache_mode}")
    table = _make_table(backends, mode_str)

    rows = []
    with Live(table, console=console, refresh_per_second=4, vertical_overflow="visible"):
        for b, k, h, d in cases:
            res, errs = bench_one(b, k, h, d,
                                  warmup=args.warmup, iters=args.iters, trials=args.trials,
                                  l2_flush=args.l2_flush, include_fa3=args.include_fa3,
                                  error_log=error_log)

            fastest = _fastest_backend(res, backends)
            plx_err_q50 = errs.get("parallax-cute", (float("nan"),) * 8)[3]
            err_cell = f"[{_err_style(plx_err_q50)}]{plx_err_q50:.2e}[/]"
            plxt_err_q50 = errs.get("parallax-triton", (float("nan"),) * 8)[3]
            err_cell_t = f"[{_err_style(plxt_err_q50)}]{plxt_err_q50:.2e}[/]"

            row = [str(b), str(k), str(h), str(d)]
            for backend in backends:
                cell = _fmt_speed(res[backend])
                if backend == fastest:
                    cell = f"[bold green]{cell}[/bold green]"
                elif isinstance(res[backend], str):  # NOT-AVAIL / NOT-IMPL / ERROR
                    cell = f"[dim]{cell}[/dim]"
                row.append(cell)
            row.append(err_cell)
            row.append(err_cell_t)
            table.add_row(*row)

            for backend in backends:
                s = res[backend]
                be_err = errs.get(backend, (float("nan"),) * 8)
                eq05, eq10, eq25, eq50, eq75, eq90, eq95, emax = be_err
                common = dict(
                    B=b, K=k, H=h, D=d, backend=backend,
                    cache=cache_mode,
                    q05_rel_err=eq05, q10_rel_err=eq10, q25_rel_err=eq25,
                    q50_rel_err=eq50, q75_rel_err=eq75, q90_rel_err=eq90,
                    q95_rel_err=eq95, max_rel_err=emax,
                )
                if isinstance(s, str):
                    rows.append(dict(common,
                        q05_ms=s, q10_ms=s, q25_ms=s, q50_ms=s,
                        q75_ms=s, q90_ms=s, q95_ms=s, n=0,
                    ))
                else:
                    rows.append(dict(common,
                        q05_ms=s.q05_ms, q10_ms=s.q10_ms, q25_ms=s.q25_ms,
                        q50_ms=s.q50_ms, q75_ms=s.q75_ms, q90_ms=s.q90_ms,
                        q95_ms=s.q95_ms, n=s.n,
                    ))

            gc.collect(); torch.cuda.empty_cache()

    if error_log:
        console.print()
        console.print("[bold red]backend errors[/bold red] (first occurrence only):")
        for name, err in error_log.items():
            console.print(f"  [yellow]{name}[/yellow]: {err}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["B", "K", "H", "D", "backend", "cache",
                  "q05_ms", "q10_ms", "q25_ms", "q50_ms",
                  "q75_ms", "q90_ms", "q95_ms", "n",
                  "q05_rel_err", "q10_rel_err", "q25_rel_err", "q50_rel_err",
                  "q75_rel_err", "q90_rel_err", "q95_rel_err", "max_rel_err"]
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fields})
        console.print()
        console.print(f"[green]✓[/green] wrote [bold]{args.csv}[/bold]  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
