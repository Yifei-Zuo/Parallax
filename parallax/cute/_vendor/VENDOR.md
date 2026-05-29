# Vendored from `flash_attn.cute`

Three pure-Python cutlass-DSL helper modules vendored verbatim from
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

## Provenance

| Item | Value |
|---|---|
| Upstream package | `flash-attn` |
| Upstream version | `2.8.3` |
| Upstream license | BSD-3-Clause (Copyright (c) 2025, Tri Dao.) |
| Source path | `flash_attn/cute/{hopper_helpers,pipeline,utils}.py` |

The verbatim copyright header `# Copyright (c) 2025, Tri Dao.` is preserved at
the top of each file.

## Why these are vendored

The Parallax SM90 decode kernel uses 11 symbols from this thin (~733-line)
helper layer over the cutlass-dsl. Vendoring lets the kernel install as a
pure-Python wheel: end users only need `nvidia-cutlass-dsl` and `torch`, not
the full `flash-attn` package (which builds C++/CUDA via nvcc and takes
20-40 min to install).

The helper layer has **no** transitive imports into the rest of `flash_attn`
— only into `cutlass.*`. See `parallax/cute/parallax_decode.py` for the
exact call sites.

## Symbols actually used by `parallax_decode.py`

| Module | Symbols |
|---|---|
| `hopper_helpers` | `gemm` |
| `pipeline` | `make_pipeline_state`, `PipelineTmaAsyncNoCluster` |
| `utils` | `convert_layout_acc_frgA`, `cvt_f16`, `elem_pointer`, `exp2f`, `make_acc_tensor_mn_view`, `shuffle_sync`, `transpose_view`, `warp_reduce` |

## Resync procedure

1. `pip install -U flash-attn==X.Y.Z` in a scratch venv.
2. `cp <site-packages>/flash_attn/cute/{hopper_helpers,pipeline,utils}.py parallax/cute/_vendor/`
3. `git diff parallax/cute/_vendor/` — review changes, especially to the 11
   symbols above. If signatures or behavior changed, update the kernel.
4. Update the Upstream version row in this file.
5. Run `python scripts/bench_decode.py` and confirm the precision pass still
   reports rel-err at the bf16 noise floor (~2-4e-3).

Do **not** modify the vendored files in place; if you need a behavior change,
either upstream the fix to flash-attn or fork to a separate parallax-owned
module — keeping the vendor copy verbatim makes resyncs a clean diff review.
