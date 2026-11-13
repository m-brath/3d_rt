# plan/kokkos-plan.md — Kokkos GPU backend for the rt3d sweeps

Goal: run the performance-critical transport kernels on GPU (and many-core CPU) via a
**Kokkos** C++ backend, while keeping the existing pure-Python solver structure, the
NumPy reference paths, and the numba fast path untouched. Driven by the expectation of
**much larger grids/cases** (100^3+ cells, deep octrees, many ordinates) where the
current CPU numbers (24^3 / 642 ordinates full Ng run: 84 s) stop being acceptable.

## 1. Why Kokkos

- One source, multiple backends: **OpenMP** (CPU), **CUDA** (NVIDIA), **SYCL** (this
  machine's Intel Arc 130V iGPU). CUDA enablement waits for NVIDIA hardware access.
- The kernels in `rt3d/jit.py` are already isolated, ray-parallel (`prange` over rays),
  and operate on flattened SoA arrays (`_oct_arrays`) — exactly the layout a Kokkos
  `RangePolicy` over rays consumes. No solver restructuring needed.
- Rejected alternatives: numba-cuda / CuPy (NVIDIA-only, this machine has no NVIDIA
  GPU), PyTorch XPU (awkward for branchy DDA), Intel numba-dpex (less mature, vendor
  lock-in).

## 2. Scope — kernels to port (priority order)

All from `rt3d/jit.py`, in order of GPU-friendliness:

1. `chain_sweep_jit` / `sc_sweep_jit` — uniform-grid chain and short-characteristics
   sweeps. Fully regular, no branching; dominant cost at large grids. First target.
2. `_march_ray` + `march_octree_jit` — DDA march over the flattened octree arrays.
   Branch-heavy but fine with one ray per thread.
3. `sweep_direction_octree_exact_jit` — fused march + formal solution. On GPU the
   avoided segment materialisation (26M+ segments/ordinate on CPU) matters even more.
4. `beam_optical_depth_octree_jit`, `locate_jit` — small helpers, last.

**Stays in Python/BLAS:** the lambda-iteration loop, Ng acceleration (dense Nz x Nz
solve), the batched scattering matmul, octree construction/refinement, all I/O. Only
the per-ordinate sweeps move to C++.

## 3. Parallel model

- `Kokkos::RangePolicy` over rays (one ray per thread), mirroring the `prange(n)`
  structure operation-for-operation.
- Same flat parameter tuples (`prm`, `prm_i`) and same SoA octree arrays as the numba
  kernels; `double` precision throughout (Kokkos default).
- Octree descent (`_locate_scalar` clone) runs per-ray inside the kernel — no
  host-side tree walking.

## 4. Packaging and dependencies (new — install into `pyarts` first)

| Package | Source | Purpose |
|---|---|---|
| `cmake`, `ninja` | conda-forge | build system |
| `kokkos` | conda-forge | core backend library (OpenMP backend first) |
| `pybind11` | conda-forge | Python bindings |
| `scikit-build-core` | pip/conda-forge | PEP 517 build backend for the extension |

- Layout: `rt3d/_kokkos/` with C++ sources + `CMakeLists.txt`, compiled into an
  extension module `rt3d._kokkos`, editable-installable into `pyarts`.
- **CUDA backend:** needs `nvcc`; only on a machine with an NVIDIA GPU. Treated as
  "enable when hardware arrives" (phase K5).
- **SYCL backend:** needs the oneAPI/ICX compiler for the Intel Arc 130V here; late
  phase (K6), not a blocker for the rest.

## 5. Backend selection

Extend the existing fallback pattern, do not replace it:

```
backend: "auto" | "numpy" | "numba" | "kokkos"     (default "auto")
```

- `rt3d.kokkos_backend.HAVE_KOKKOS` flag, set by trying to import `rt3d._kokkos`.
- `rt3d.transport` dispatch order in auto mode: kokkos -> numba -> NumPy reference.
- Recorded in every output file alongside `n_refine`.
- All three paths must pass the existing tests and the five physics validation
  anchors (AGENTS.md).

## 6. Validation policy (decision)

The numba kernels are bit-comparable to the NumPy reference. **On GPU this is not
achievable** (different transcendental implementations, different reduction order).

- Kokkos-**OpenMP** path: keep near-bitwise agreement with the numba path.
- Kokkos-**CUDA/SYCL** paths: `np.allclose` with explicit tight tolerance
  (rtol 1e-12, atol scaled per quantity) against the NumPy reference in the tests.
- Anchor 3 (`exp(-tau)` transmittance) must hold to the same tolerance on every
  backend.

## 7. Phases

| Phase | Deliverable |
|---|---|
| K0 | Spike: one chain-sweep kernel, OpenMP backend, minimal pybind11 build; prove agreement with `chain_sweep_jit` |
| K1 | scikit-build-core packaging, editable install into `pyarts`, build docs |
| K2 | Uniform-grid sweeps (`chain_sweep`, `sc_sweep`) + `tests/test_kokkos.py` |
| K3 | Octree march + fused exact sweep + tests |
| K4 | `backend` config option, auto-detect, fallback wiring in `transport.py` / `solver.py`, output-file recording |
| K5 | CUDA backend enablement **when NVIDIA hardware is available** + benchmarks vs numba |
| K6 | SYCL backend for the Intel Arc here + benchmarks; decision on default backend |

## 8. Cost / risk

- K0-K4 is roughly a few days of work; K5/K6 depend on hardware/toolchain access.
- Main ongoing cost: **dual implementation** — every future kernel change needs a
  NumPy, numba, and Kokkos version, tied together by cross-backend tests. This is the
  price of the route (the numba-only design avoided it).
- Windows build risk: MSVC + Kokkos via conda-forge is the first thing K0 must prove;
  if it fails, fall back to building Kokkos from source with the scikit-build-core
  FetchContent flow before considering any change of direction.
- The Arc 130V is an iGPU sharing memory bandwidth with the CPU — expected speedups
  there are modest; the real payoff target is K5 hardware.
