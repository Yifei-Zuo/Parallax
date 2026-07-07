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

# Optional [cutedsl] stack: substitute stubs that raise on call so these names
# still import on a training-only install. Catch Exception, not ImportError —
# a present-but-broken stack (no NVIDIA driver, non-SM90 GPU) raises
# RuntimeError/OSError at import and must not take down the Triton/Helion paths.
try:
    from parallax.cute import (
        GraphedDecode,
        parallax_attn_with_kvcache,
        parallax_decode,
    )
    cute_decode_available: bool = True
except Exception as _cute_err:
    cute_decode_available = False
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


def __getattr__(name):
    if name == "decode_available":  # deprecated 0.1.0 alias
        import warnings

        warnings.warn(
            "parallax.decode_available is deprecated; use "
            "parallax.cute_decode_available. It only reports the optional "
            "CuTeDSL SM90 kernel — the Triton and Helion decode kernels do "
            "not depend on it.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cute_decode_available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "cute_decode_available",
]
