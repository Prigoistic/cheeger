# extensions/

Native C++ / CUDA hot-path kernels, compiled via `torch.utils.cpp_extension`
(no cmake needed — clang/nvcc invoked directly).

These are an **optimisation layer added only after the pure-torch math is proven
and benchmarked**. The Python implementation in `src/fiedler/spectral/` stays the
reference; an extension must produce bit-comparable forward output and pass the
same `gradcheck` before it replaces the Python path.

Planned:
- `lanczos_cuda/` — fused sparse Lanczos (SpMV + reorthogonalisation) for the
  top-k eigensolver on large pixel-affinity graphs.
