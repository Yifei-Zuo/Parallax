"""SM90 CuTeDSL kernels.

Importing this subpackage requires ``nvidia-cutlass-dsl`` and
``nvidia-cuda-python`` — install the top-level ``[decode]`` extra.
On a training-only install (torch + triton only) the import will fail;
the top-level :mod:`parallax` package catches that and substitutes a
stub that raises a helpful error on call.
"""

from parallax.cute.parallax_decode import parallax_decode

__all__ = ["parallax_decode"]
