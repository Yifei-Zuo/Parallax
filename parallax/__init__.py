"""Parallax — Parameterized Local Linear Attention.

Public entry points:
  * ``parallax_func``       — Triton training (causal fwd+bwd, autograd).
  * ``parallax_fwd``, ``parallax_bwd`` — raw Triton kernels with the
                                          intermediate stats exposed.
  * ``parallax_reference``  — fp32 PyTorch reference, runs anywhere.
  * ``parallax_decode``     — SM90 CuTeDSL decode kernel  (extras: [decode]).

The first three entry points work on any CUDA GPU and only require torch +
triton. The cute-based decode kernel additionally needs
``nvidia-cutlass-dsl`` and ``nvidia-cuda-python``; install the ``[decode]``
extra to get it.
"""

from parallax.reference import parallax_reference
from parallax.triton import parallax_func, parallax_bwd, parallax_fwd

# Optional extra: the cute decode kernel needs the [decode] stack. Substitute
# a stub that raises on call so ``from parallax import parallax_decode`` still
# works on a training-only install.
try:
    from parallax.cute import parallax_decode
    decode_available: bool = True
except ImportError as _cute_err:
    decode_available = False
    _cute_err_msg = (
        "Parallax decode kernel requires the [decode] extra "
        "(nvidia-cutlass-dsl + nvidia-cuda-python, Hopper SM90 only). "
        "Install with:  pip install 'parallax[decode]'  "
        "or  uv sync --extra decode\n"
        f"Underlying import error: {_cute_err}"
    )

    def parallax_decode(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(_cute_err_msg)


__all__ = [
    "parallax_func",
    "parallax_fwd",
    "parallax_bwd",
    "parallax_reference",
    "parallax_decode",
    "decode_available",
]
