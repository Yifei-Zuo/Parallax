"""Shared pytest fixtures/helpers for the Parallax decode test suite.

The CuTeDSL decode kernel is SM90 (Hopper) only, so the whole suite is
skipped on non-Hopper / CPU-only machines. Helpers here mirror the input
construction used by ``scripts/_test_utils.py`` and the parity oracle in
``scripts/test_decode_swa.py`` so tests stay consistent with the benches.
"""
from __future__ import annotations

import math

import pytest
import torch

# --- hardware gate -----------------------------------------------------------
_NO_CUDA = not torch.cuda.is_available()
_NOT_SM90 = _NO_CUDA or torch.cuda.get_device_capability()[0] != 9


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "sm90: test requires an SM90 (Hopper) GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Skip the whole suite on non-SM90 hardware (kernels need Hopper)."""
    if not _NOT_SM90:
        return
    reason = "no CUDA device" if _NO_CUDA else "requires SM90 (Hopper) GPU"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        item.add_marker(skip)


# --- input / oracle helpers --------------------------------------------------
def make_decode_inputs(B, H, kv_len, *, H_kv=None, D=128,
                       dtype=torch.bfloat16, seed=0, device="cuda"):
    """``(q, r, k, v)`` for a decode step. ``H`` is H_q; ``H_kv`` defaults to H."""
    H_kv = H_kv if H_kv is not None else H
    torch.manual_seed(seed)
    q = torch.randn(B, 1, H, D, device=device, dtype=dtype)
    r = torch.randn_like(q) * 0.5
    k = torch.randn(B, kv_len, H_kv, D, device=device, dtype=dtype)
    v = torch.randn_like(k)
    return q, r, k, v


def scale_for(D):
    return 1.0 / math.sqrt(D)


def rel_err(out, ref):
    """Worst element abs error normalized by the output magnitude.

    bf16 noise floor sits at 2-5e-3; the composite formula cancels heavily so
    element-wise *relative* error blows up near zeros — we gauge against the
    output max magnitude instead (same convention as test_decode_swa.py).
    """
    diff = (out.float() - ref.float()).abs().max().item()
    ref_scale = max(ref.float().abs().max().item(), 1e-6)
    return diff / ref_scale


# bf16 rel-error tolerance against the fp32 reference (matches test_decode_swa).
REL_TOL = 1e-2
