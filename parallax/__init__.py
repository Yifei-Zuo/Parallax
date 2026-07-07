"""Parallax — Parameterized Local Linear Attention.

Public entry points:
  * ``parallax_func``       — Triton training (causal fwd+bwd, autograd).
  * ``parallax_varlen_func`` — Triton variable-length (packed) training example.
  * ``parallax_fwd``, ``parallax_bwd`` — raw Triton kernels with the
                                          intermediate stats exposed.
  * ``parallax_reference``  — fp32 PyTorch reference, runs anywhere.
  * ``parallax.triton.parallax_decode`` — pure-Triton single-token decode
                                          (any CUDA GPU; no extra deps).
  * ``parallax_attn_with_kvcache`` — SM90 CuTeDSL decode against a KV cache,
                                     canonical FA-style entry (extras: [cutedsl]).
  * ``parallax_decode``     — deprecated alias of the above (extras: [cutedsl]).

All entry points except the cute decode kernel work on any CUDA GPU and only
require torch + triton. The cute-based decode kernel additionally needs
``nvidia-cutlass-dsl`` and ``cuda-python``; install the ``[cutedsl]``
extra to get it.
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("parallax-kernel")
except PackageNotFoundError:  # source tree without installed dist metadata
    __version__ = "0.0.0+unknown"

from parallax.reference import parallax_reference
from parallax.triton import (
    parallax_func,
    parallax_bwd,
    parallax_fwd,
    parallax_varlen_func,
)

# Optional extra: the cute decode kernel needs the [cutedsl] stack. Substitute
# a stub that raises on call so ``from parallax import parallax_decode`` still
# works on a training-only install.
try:
    from parallax.cute import (
        GraphedDecode,
        parallax_attn_with_kvcache,
        parallax_decode,
    )
    decode_available: bool = True
except ImportError as _cute_err:
    decode_available = False
    _cute_err_msg = (
        "Parallax decode kernel requires the [cutedsl] extra "
        "(nvidia-cutlass-dsl + cuda-python, Hopper SM90 only). "
        "Install with:  pip install 'parallax-kernel[cutedsl]'  "
        "or  uv sync --extra cutedsl\n"
        f"Underlying import error: {_cute_err}"
    )

    def parallax_attn_with_kvcache(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(_cute_err_msg)

    def parallax_decode(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(_cute_err_msg)

    class GraphedDecode:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_cute_err_msg)


__all__ = [
    "__version__",
    "parallax_func",
    "parallax_varlen_func",
    "parallax_fwd",
    "parallax_bwd",
    "parallax_reference",
    "parallax_attn_with_kvcache",
    "parallax_decode",
    "GraphedDecode",
    "decode_available",
]
