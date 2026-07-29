# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NMarkov.jl — Julia package for numerical analysis of continuous-time Markov chains (CTMCs): stationary/quasi-stationary distributions, transient rewards, matrix exponentials, and sensitivity analysis. Not registered; depends on two unregistered JuliaReliab packages (`ZeroOrigin`, `DEQuadrature`). `compat` targets Julia 1.6; CI runs 1.6/1.9/1.10.

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # setup (DEQuadrature comes from [sources] git rev)
julia --project=. -e 'using Pkg; Pkg.test()'          # full test suite
julia --project=. test/runtests.jl                    # same, faster (no sandbox)
julia --project=. examples/02_transient_analysis.jl    # run an example
```

To run a single test file, run `runtests.jl` with the other `include` lines skipped, or:

```bash
julia --project=. -e 'using NMarkov, Test; include("test/test_mexp.jl")'
```

There is no linter or formatter configured.

## Architecture

Everything is one module, `NMarkov`, with a nested submodule `NMarkov.SparseMatrix`. Include order in `src/NMarkov.jl` is load-bearing: `SparseMatrix` → `utils.jl` → analysis files → `poisson.jl` → `mexp.jl` → `transient_analysis.jl` → `conv.jl`.

**Uniformization is the backbone.** Nearly every transient computation goes through `unif(Q, ufact) -> (P, qv)` (`src/utils.jl`), which builds the DTMC `P = I + Q/qv`, then weights powers of `P` by Poisson pmf values from `poipmf`/`cpoipmf`/`rightbound` (`src/poisson.jl`). `rightbound(qt, eps)` decides the truncation point; callers assert it against `rmax` (default 500) and error with "Time interval is too large" rather than silently degrading. When touching `mexp.jl`, `transient_analysis.jl`, or `conv.jl`, keep this pattern — do not substitute a dense `exp(Q*t)`.

**Regression tests.** `test/test_regression.jl` pins the bugs found in the 0.4.0 review; each testset names the behaviour it guards (absorbing-state uniformization, Poisson truncation order, time-vector validation, element-type genericity, the `AbstractMatrix` contract). Treat a failure there as a re-introduced bug, not a stale expectation.

**Layered API files:**
- `stationary_analysis.jl` — `gth`/`gth!` (direct GTH elimination), `stgs` (Gauss–Seidel via `gsstep!`), `stpower`, `stguess`.
- `sensitivity_analysis.jl` — `stsen` (QR-based), `stsengs`, `stsenpower`; mirrors the stationary solvers one derivative up.
- `quasistationary_analysis.jl` — `qstgs`, `qstpower`.
- `mexp.jl` (largest file) — `mexp`/`mexpc` for a single time, a vector of times, and mixture/`Distribution` variants (`mexpmix`, `mexpcmix`, which integrate over a density with `DEQuadrature.deint`).
- `transient_analysis.jl` — `tran(Q, x, r, ts)` returns `(inst, cum, x_final, r_final)`; computes the whole time series in one uniformization sweep, which is why it beats looping `mexp`.
- `conv.jl` — `convunifstep!`, the uniformized convolution `∫ exp(Q's) x y' exp(Q'(t-s)) ds` used for two-variable reward functionals.

**Where the formulas come from.** `conv.jl` implements the convolution integral of Okamura, Dohi & Trivedi, *A Refined EM Algorithm for PH Distributions* (Performance Evaluation); the file header maps each variable to its equation number. Before changing a summation range there, check the paper: `H` is eq.(24) with `U = right-left-1` (because `β_U` needs `π_{U+1}`), while `z` is the plain matrix-exponential truncation whose `U` is `rightbound(...)` itself — the two ranges differ *on purpose*, and one past reviewer "fixed" that apparent inconsistency in the wrong direction. Related: `poipmf!` returns values scaled by a constant Stirling factor (its `weight` is ~1.017, not 1), so dividing by `weight` is mandatory, not cosmetic.

**Type-flexibility convention.** Public entry points (`mexp`, `mexpc`, `tran`) are a *single* method that accepts loose types (`x::AbstractArray`, `t::Real`, `ts::AbstractVector`), converts everything to `Q`'s element type with `asarray`/`asvector` (`utils.jl`), validates times with `checktime`/`checktimes`, and then calls the internal kernel (`_mexp`, `_mexpc`, `_tran`). The split is deliberate: a public method that could also match the kernel's signature would dispatch back to itself for some element types. Never give the public name a second method that overlaps the kernel. `asarray` preserves shape — `tran` reaches its matrix methods only because the conversion does not `vec`.

**Element-type genericity is a supported feature**, not an accident: `Float32`, `Float16`, `Int` inputs and `BigFloat` sparse matrices all work. Inside a `Tv`-parameterized function never write a bare float literal where a scalar of `Tv` is expected — use `one(Tv)`/`zero(Tv)`/`convert(Tv, …)`. Tolerance defaults that must scale with precision go through helpers (`_stcheck_tol` in `sensitivity_analysis.jl`); a hardcoded `1.0e-8` is below Float32 resolution.

**Matrix genericity.** Algorithms are written against `AbstractMatrix` and dispatch through two helpers in `utils.jl`: `matmul!` (wraps BLAS `gemv!`/`gemm!` for dense, custom kernels for sparse) and `spdiag` (diagonal *view*, aliasing the storage, that works for all formats). `SparseMatrix` supplies `SparseCSR`, `SparseCSC`, `SparseCOO`, `SparseELL1/2`, and `BlockCOO`, each with its own `blas_level1/2/3.jl` implementations. Adding a sparse format means adding a `unif` method (via the `UnifMatrix` union), `spdiag` and `adddiag` support, and the relevant BLAS-level methods — not just the struct.

**Overloading BLAS names.** `blas_level1/2/3.jl` extend `LinearAlgebra.BLAS.scal!`/`axpy!`/`gemv!`/`gemm!` for the sparse types. Because those types are `AbstractMatrix` subtypes, a single method with a free element type is *ambiguous* with BLAS's own methods for `Float32`/`Float64`/`ComplexF32`/`ComplexF64`. The pattern used throughout: put the body in an internal `_spscal!`/`_spaxpy!`/`_spgemv!`/`_spgemm!`, then generate one generic forwarding method plus exact-element-type forwarders for the four BLAS types. Follow it when adding operations, and check whether `LinearAlgebra`'s same-named generic function (e.g. `LinearAlgebra.axpy!`, distinct from `BLAS.axpy!`) also needs a method.

**`unif` and structurally-absent diagonals.** `unif` adds 1 to every diagonal entry through `spdiag`, which can only write positions the sparsity pattern stores. A CTMC with an absorbing state has a zero diagonal that dense→sparse conversion drops, so `unif` calls `adddiag` first and scales a structure-preserving `copy` (`A / qv` goes through the generic sparse fallback, which prunes the structural zeros again). `spdiag`'s `setindex!` throws rather than warning when the entry is missing.

**`@origin` and `@inbounds`.** `ZeroOrigin.@origin` rewrites literal index expressions on the named arrays, shifting them to physical indices (`a[i]` → `a[i - origin + 1]`); the underlying arrays stay 1-based. Don't "fix" apparent off-by-ones inside those blocks. Two rules follow:

- Every `@origin` block here is also `@inbounds`, so a logical index outside the declared domain is an *unchecked* out-of-bounds access — a segfault, or a silent buffer under/overflow that still returns plausible numbers. Validate the domain once before entering the block; `_checkpoirange` (`poisson.jl`) and `_checkconvrange` (`conv.jl`) are the existing guards, and any new `@origin` kernel needs the equivalent. `poipmf!` is the cautionary case: it seeds at `prob[floor(lambda)]`, so the domain must contain the mode.
- Only the arrays listed in the macro call are rewritten, and a listed *matrix* would share one origin across all dimensions. The matrices in these kernels (`H`, `y0`, `y1`, `cy`) are deliberately not listed.
