# Copyright (c) 2026 Yifei Zuo.
# SPDX-License-Identifier: MIT
"""Helion implementations of Parallax kernels (evaluation track).

Helion is a PyTorch-embedded, autotuned DSL that compiles to Triton. These
kernels benchmark Helion against the hand-written Triton / CuTeDSL kernels.
Requires the ``[helion]`` extra (``uv sync --extra helion``).

  * ``parallax_decode``      — single-token decode (base + split-KV).
  * ``parallax_func`` — dense causal training pass (fwd+bwd, autograd),
                               drop-in for :func:`parallax.parallax_func`.
"""
from parallax.helion.parallax_decode import parallax_decode
from parallax.helion.parallax_train import (
    parallax_bwd,
    parallax_func,
    parallax_fwd,
)

__all__ = [
    "parallax_decode",
    "parallax_func",
    "parallax_fwd",
    "parallax_bwd",
]
