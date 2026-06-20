"""CUDA graph capture/replay for the Parallax decode call.

Small-batch decode is host-bound (~ms/step of launch + Python overhead). The
fix is to capture the launch in a CUDA graph and replay it. The unified
``parallax_attn_with_kvcache`` entry takes a caller-provided ``out`` and
stable input tensors, so there is no allocation, host sync, or hidden buffer
swap inside the call and the kernel launch captures cleanly. The internal
split-K workspace is persistent (allocated on the warmup call, reused at
fixed addresses), and the cute tensor views are memoized by data pointer.

Contract under test: after capturing the graph, mutating the input tensors
*in place* and replaying must reproduce what a fresh eager call would compute
for those new inputs.
"""
from __future__ import annotations

import pytest
import torch

from parallax import GraphedDecode, parallax_attn_with_kvcache, parallax_reference
from conftest import REL_TOL, make_decode_inputs, rel_err, scale_for


@pytest.mark.sm90
@pytest.mark.parametrize("B,H,kv_len", [(1, 8, 4096), (2, 8, 1024), (1, 1, 8192)])
def test_cuda_graph_capture_replay(B, H, kv_len):
    D = 128
    q, r, k, v = make_decode_inputs(B, H, kv_len, seed=7)
    out = torch.empty(B, 1, H, D, device="cuda", dtype=torch.bfloat16)

    # --- warmup on a side stream (compiles the kernel, allocates the split-K
    # workspace, memoizes the cute views) — required before graph capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            parallax_attn_with_kvcache(q, r, k, v, scale=scale_for(D), out=out)
    torch.cuda.current_stream().wait_stream(s)

    # --- capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        parallax_attn_with_kvcache(q, r, k, v, scale=scale_for(D), out=out)

    # --- replay with fresh inputs written into the *same* tensors
    q2, r2, k2, v2 = make_decode_inputs(B, H, kv_len, seed=99)
    q.copy_(q2); r.copy_(r2); k.copy_(k2); v.copy_(v2)
    g.replay()
    torch.cuda.synchronize()
    out_graph = out.clone()

    ref = parallax_reference(q2, r2, k2, v2, scale_for(D), causal=True)
    assert not torch.isnan(out_graph).any()
    assert rel_err(out_graph, ref) < REL_TOL, "graph replay diverged from reference"

    # --- replaying again with yet-newer inputs must track them (proves replay
    # reads the live input buffers, not a captured snapshot)
    q3, r3, k3, v3 = make_decode_inputs(B, H, kv_len, seed=123)
    q.copy_(q3); r.copy_(r3); k.copy_(k3); v.copy_(v3)
    g.replay()
    torch.cuda.synchronize()
    ref3 = parallax_reference(q3, r3, k3, v3, scale_for(D), causal=True)
    assert rel_err(out.clone(), ref3) < REL_TOL


@pytest.mark.sm90
def test_cuda_graph_matches_eager():
    """A captured replay equals a fresh eager call on identical inputs."""
    B, H, kv_len, D = 1, 8, 4096, 128
    q, r, k, v = make_decode_inputs(B, H, kv_len, seed=11)
    out = torch.empty(B, 1, H, D, device="cuda", dtype=torch.bfloat16)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            parallax_attn_with_kvcache(q, r, k, v, scale=scale_for(D), out=out)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        parallax_attn_with_kvcache(q, r, k, v, scale=scale_for(D), out=out)
    g.replay()
    torch.cuda.synchronize()
    graph_out = out.clone()

    # fresh eager call (out=None -> its own fresh buffer; no aliasing with `out`)
    eager = parallax_attn_with_kvcache(q, r, k, v, scale=scale_for(D))
    torch.cuda.synchronize()
    assert torch.equal(graph_out, eager), "graph replay differs from eager on same inputs"


@pytest.mark.sm90
@pytest.mark.parametrize("B,H,kv_len,win", [(1, 8, 4096, -1), (1, 8, 4096, 256)])
def test_graphed_decode_helper_static(B, H, kv_len, win):
    """Static-length GraphedDecode (legacy `kv_len=` constructor): one
    instance per kv_len, attends every position in the buffer."""
    D = 128
    win_kw = None if win < 0 else (win, 0)
    gd = GraphedDecode(B, H, kv_len=kv_len, head_dim=D, window_size=win_kw)

    q, r, k, v = make_decode_inputs(B, H, kv_len, seed=5)
    out = gd(q=q, r=r, k=k, v=v).clone()
    ref = parallax_reference(q, r, k, v, scale_for(D), causal=True, window_size_left=win)
    assert not torch.isnan(out).any()
    assert rel_err(out, ref) < REL_TOL

    # a second step with new inputs replays correctly (proves buffer reuse)
    q2, r2, k2, v2 = make_decode_inputs(B, H, kv_len, seed=6)
    out2 = gd(q=q2, r=r2, k=k2, v=v2).clone()
    ref2 = parallax_reference(q2, r2, k2, v2, scale_for(D), causal=True, window_size_left=win)
    assert rel_err(out2, ref2) < REL_TOL


@pytest.mark.sm90
@pytest.mark.parametrize("active_len", [16, 256, 1000, 4096, 8000, 16384])
def test_graphed_decode_dynamic_kvlen(active_len):
    """Dynamic-length GraphedDecode (seqused_k): ONE capture covers every
    decode step. K/V are pre-allocated at max_kv_len; the graph reads
    cache_seqlens at replay time so the same graph handles any active length.
    """
    B, H, D = 1, 8, 128
    max_kv_len = 16384
    gd = GraphedDecode(B, H, max_kv_len=max_kv_len, head_dim=D)

    q, r, k, v = make_decode_inputs(B, H, max_kv_len, seed=11)
    # Mutate the graph's stable buffers in place.
    gd.q.copy_(q); gd.r.copy_(r); gd.k.copy_(k); gd.v.copy_(v)
    out = gd(cache_seqlens=active_len).clone()
    ref = parallax_reference(q, r, k[:, :active_len], v[:, :active_len],
                              scale_for(D), causal=True)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    assert rel_err(out, ref) < REL_TOL


@pytest.mark.sm90
def test_graphed_decode_dynamic_sweep():
    """One captured graph replays cleanly across an ascending decode
    trajectory — the serving pattern. Verifies the same graph produces
    correct outputs at every step from kv_len=1 to max_kv_len."""
    B, H, D = 1, 8, 128
    max_kv_len = 2048
    gd = GraphedDecode(B, H, max_kv_len=max_kv_len, head_dim=D)

    q, r, k, v = make_decode_inputs(B, H, max_kv_len, seed=17)
    gd.q.copy_(q); gd.r.copy_(r); gd.k.copy_(k); gd.v.copy_(v)
    for kv_len in [1, 16, 64, 100, 512, 1024, 2000, 2048]:
        out = gd(cache_seqlens=kv_len).clone()
        ref = parallax_reference(q, r, k[:, :kv_len], v[:, :kv_len],
                                  scale_for(D), causal=True)
        assert not torch.isnan(out).any()
        # kv_len=1 lives at the bf16 noise floor (single-position attention);
        # allow a slightly looser bound there. Everything else is comfortably
        # inside REL_TOL.
        tol = 1.5e-2 if kv_len <= 1 else REL_TOL
        assert rel_err(out, ref) < tol, f"step kv_len={kv_len}: rel_err too high"


@pytest.mark.sm90
@pytest.mark.parametrize("pack_n", [2, 4, 8])
def test_graphed_decode_gqa(pack_n):
    """GQA via GraphedDecode: H_q = H_kv * pack_n, one capture handles the
    packed-head decode for every active length up to max_kv_len."""
    B, H_kv, max_kv_len, D = 1, 4, 4096, 128
    H_q = H_kv * pack_n
    gd = GraphedDecode(B, H_q, max_kv_len=max_kv_len, head_dim=D, H_kv=H_kv)

    q, r, k, v = make_decode_inputs(B, H_q, max_kv_len, H_kv=H_kv, seed=23)
    gd.q.copy_(q); gd.r.copy_(r); gd.k.copy_(k); gd.v.copy_(v)
    for kv_len in [256, 1024, 4096]:
        out = gd(cache_seqlens=kv_len).clone()
        ref = parallax_reference(q, r, k[:, :kv_len], v[:, :kv_len],
                                  scale_for(D), causal=True)
        assert not torch.isnan(out).any()
        assert rel_err(out, ref) < REL_TOL, f"GQA pack_n={pack_n} kv_len={kv_len}: rel_err too high"


@pytest.mark.sm90
def test_graphed_decode_non_pow2_clamp():
    """Non-pow2 cache_len: in-kernel clamp uses real cache extent, not bucket.

    k_cache.shape[1]=1500 (non-pow2), bucket=2048 (pow2). Under graph replay,
    an out-of-range seqused_k > k.shape[1] must be clamped to 1500 by the
    in-kernel clamp. The band (k.shape[1], bucket] no longer silently attends
    uninitialized memory — the real-cache-extent clamp closes that hole.

    We manually capture a graph (not via GraphedDecode) to inject an
    out-of-range seqused_k into the stable seqlen buffer, bypassing the
    host-side validation.
    """
    B, H, D = 1, 8, 128
    cache_len = 1500  # non-pow2, bucket = 2048

    q = torch.randn(B, 1, H, D, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(q) * 0.5
    # K/V sized at the real cache extent (not bucket)
    k = torch.randn(B, cache_len, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    out = torch.empty_like(q)
    # Stable seqlen buffer (same idiom as the dispatcher)
    seqlen_buf = torch.full((B,), cache_len, dtype=torch.int32, device="cuda")

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            parallax_attn_with_kvcache(q, r, k, v, seqused_k=seqlen_buf,
                                       scale=scale_for(D), out=out)
    torch.cuda.current_stream().wait_stream(s)

    # Capture with valid seqused_k
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        parallax_attn_with_kvcache(q, r, k, v, seqused_k=seqlen_buf,
                                   scale=scale_for(D), out=out)

    # Inject out-of-range seqused_k into the stable buffer and replay
    # 1800 > cache_len=1500, should clamp to 1500
    seqlen_buf.fill_(1800)
    g.replay()
    torch.cuda.synchronize()
    graph_out = out.clone()

    # Reference: attend only up to cache_len (1500)
    ref = parallax_reference(q, r, k, v, scale_for(D), causal=True)
    assert not torch.isnan(graph_out).any() and not torch.isinf(graph_out).any()
    assert rel_err(graph_out, ref) < REL_TOL, (
        f"seqused_k=1800 on cache_len=1500 should clamp to 1500, "
        f"but output differs from reference (rel_err={rel_err(graph_out, ref):.4f})"
    )
