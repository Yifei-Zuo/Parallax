"""SM90 CuTeDSL kernels.

Importing this subpackage requires ``nvidia-cutlass-dsl`` and
``cuda-python`` — install the top-level ``[cutedsl]`` extra.
On a training-only install (torch + triton only), or wherever the cute
stack fails to load (no NVIDIA driver, non-SM90 machine), the import
fails; the top-level :mod:`parallax` package catches any failure and
substitutes a stub that raises a helpful error on call.
"""

from parallax.cute.parallax_decode import (
    GraphedDecode,
    parallax_attn_with_kvcache,
    parallax_decode,
)

__all__ = ["GraphedDecode", "parallax_attn_with_kvcache", "parallax_decode"]
