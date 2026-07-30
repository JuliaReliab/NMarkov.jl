# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NMarkov.jl — Julia package for numerical analysis of continuous-time Markov chains (CTMCs): stationary/quasi-stationary distributions, transient rewards, matrix exponentials, and sensitivity analysis. Registered in the JuliaReliab registry (`https://github.com/JuliaReliab/Registry.git`, maintained with LocalRegistry.jl), not in General; so is its `DEQuadrature` dependency. The other dependency, `ZeroOrigin`, is in General. Adding that registry once is all the setup a checkout needs — there is deliberately no `[sources]` entry, since a registered package's `[sources]` is not honoured for consumers. `compat` targets Julia 1.10 (the current LTS); CI runs `min` and `1`, i.e. that lower bound and the latest stable release.

## Commands

```bash
# One-time: the JuliaReliab registry provides DEQuadrature (and NMarkov itself).
# Without it, instantiate cannot resolve DEQuadrature — there is no [sources] entry.
julia -e 'using Pkg;
  names = [r.name for r in Pkg.Registry.reachable_registries()]
  "General" in names || Pkg.Registry.add("General")
  "Registry" in names ||
      Pkg.Registry.add(RegistrySpec(url="https://github.com/JuliaReliab/Registry.git"))'

julia --project=. -e 'using Pkg; Pkg.instantiate()'   # setup
julia --project=. -e 'using Pkg; Pkg.test()'          # full test suite
julia --project=. test/runtests.jl                    # same, faster (no sandbox)
julia --project=. examples/02_transient_analysis.jl   # run an example
for f in examples/0*.jl; do julia --project=. $f; done  # CI does NOT run these
```

To run a single test file, run `runtests.jl` with the other `include` lines skipped, or:

```bash
julia --project=. -e 'using NMarkov, Test; include("test/test_mexp.jl")'
```

There is no linter or formatter configured. `Manifest.toml` is untracked, so
regenerating it is always safe.

**Releasing.** Bump `version` in `Project.toml`, merge to `master`, then from a
`master` checkout: register with LocalRegistry (`register("<path>";
registry="Registry")` — it commits to `~/.julia/registries/Registry` and pushes to
`JuliaReliab/Registry`), tag `vX.Y.Z`, and cut a GitHub release titled
`NMarkov X.Y.Z`. Register before releasing, so a release never points at an
unregistered version. Verify from an empty depot:
`JULIA_DEPOT_PATH=$(mktemp -d) julia -e '...Pkg.add("NMarkov")...'`.

## Architecture

Everything is one module, `NMarkov`, with a nested submodule `NMarkov.SparseMatrix`. Include order in `src/NMarkov.jl` is load-bearing: `SparseMatrix` → `utils.jl` → analysis files → `poisson.jl` → `mexp.jl` → `transient_analysis.jl` → `conv.jl`.

**Uniformization is the backbone.** Nearly every transient computation goes through `unif(Q, ufact) -> (P, qv)` (`src/utils.jl`), which builds the DTMC `P = I + Q/qv`, then weights powers of `P` by Poisson pmf values from `poipmf`/`cpoipmf`/`rightbound` (`src/poisson.jl`). `rightbound(qt, eps)` decides the truncation point, which is simultaneously a buffer length and a loop count; callers check it against `rmax` (default 500) and raise rather than silently degrading. When touching `mexp.jl`, `transient_analysis.jl`, or `conv.jl`, keep this pattern — do not substitute a dense `exp(Q*t)`.

**`dropzero` for the mixture functions.** `mexpmix`/`mexpcmix` build their time grid from the nodes `DEQuadrature.deint` returns, and the truncation point grows with `qv * maxt` where `maxt` is the *widest interval* of that grid. The DE transform spaces nodes multiplicatively, so one node far out in the tail makes `maxt` as large as that node. `deint` keeps every node whose weight is not exactly zero, and tail weights underflow to *subnormals* rather than to zero — with `dropzero = 0` the grid reached `4e15` for `LogNormal(0,1)` and `6.8e128` for `Pareto(1.5,1)`. Hence the `dropzero` keyword, default `eps(Tv)`; the discarded contributions are of order `1e-299`, so the integral is unchanged (for `exp(-u)` it still matches `inv(I - Q') * x0`, while the term count drops 828 → 64).

When a term count exceeds `rmax`, **read `maxt` from the error before reaching for a larger `rmax`** — `_checkmixrmax` (`mexp.jl`) prints it for exactly this reason. A moderate `maxt` just needs more terms; an enormous one means the tail cannot be followed over an unbounded range, and raising `rmax` would ask for a buffer of that many elements. A genuinely heavy-tailed density needs finite `bounds` regardless of `dropzero`. This is not a DEQuadrature bug: returning every nonzero-weight node is right for `deint`; trimming the tail is the caller's job because the caller uses those nodes as a time series.

**Tests.** `test/runtests.jl` includes one file per area (`test_sparsematrix`, `test_stationary`, `test_poisson`, `test_matrix`, `test_mexp`, `test_mix`, `test_conv`, `test_transient`) plus `test_regression.jl`. That last one pins the bugs found in the 0.4.0 review and after; each testset names the behaviour it guards (absorbing-state uniformization, the `z`/`H` summation ranges in `conv`, time-vector validation, `poipmf` domain validation, Gauss-Seidel diagonal preconditions, element-type genericity, the `AbstractMatrix` contract, mixture tail nodes). Treat a failure there as a re-introduced bug, not a stale expectation. The seven `examples/*.jl` are **not** run by CI — check them by hand after touching `mexp.jl`/`transient_analysis.jl`, since they exercise paths the tests do not (the `dropzero` bug surfaced only there).

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

## Session log

### 2026-07-29 / 30 — full code review of 0.4.0, then packaging (0.4.1 → 0.5.2)

Started as "review this codebase", grew into four releases. All merged to `master`
via PRs #15–#19; 0.5.1 and 0.5.2 are registered, tagged and released.

**Correctness fixes (0.4.1).** The one defect on a well-trodden path: `unif`
returned a non-stochastic `P` for a *sparse* generator whose diagonal was not fully
stored (an absorbing state's zero diagonal is dropped by dense→sparse conversion,
and `spdiag`'s `setindex!` discarded the write with only a warning) — so every
downstream `mexp`/`mexpc`/`tran`/`conv` result was wrong. Fixed with `adddiag`.
Also: `convunifstep!` normalised `z` by a weight covering a wider range than it
summed; `rightbound` looped forever below Float64; unsorted/negative time vectors
returned `NaN`; `stguess`/`gth`/the Gauss-Seidel solvers returned `NaN` on
absorbing states; `poipmf!`/`cpoipmf!` wrote outside their buffer (segfault, or
silent heap corruption returning plausible numbers); a `Base.iszero(::Float64)`
definition was pirating `iszero` for every `Float64` in the session. Plus the
`Float32`/`Float16`/`Int`/`BigFloat` paths (infinite recursion, `Float64`-only
sparse BLAS, ambiguous scalar `*`/`/`), the CSR `gsstep!` `UndefVarError`, and the
`AbstractMatrix` contract on `AbstractSparseM` (`length` was `nnz`).

**Packaging (0.5.0 → 0.5.2).** Julia compat raised 1.6 → 1.10 (LTS); CI matrix
`1.6/1.9/1.10` → `["min","1"]` (it had been testing an EOL version and *no*
shipping Julia); `concurrency` added so superseded PR runs are cancelled;
`julia-actions/cache@v3`; coverage wired up; registered in the JuliaReliab registry
so `Pkg.add("NMarkov")` works; `[sources]` dropped; required status checks
(`Julia min`, `Julia 1`, `strict=true`) enabled on `master`. 0.5.2 fixed the
`dropzero` blow-up described above.

**Three mistakes I made, for calibration.** (1) Reported `conv.jl`'s `H` as buggy;
it matches the paper bit-for-bit and I had only changed `z`. (2) "Verified" that the
old `z` was more accurate — using a matrix whose rows did not sum to zero, so
`unif`'s `P` was not stochastic and the whole comparison was vacuous. Always test
CTMC code with a real generator. (3) `adddiag` initially converted to COO just to
*test* the diagonal, making `unif` 2–4× slower; caught only because the
before/after benchmark was actually run.

**Verification habits that paid off.** Benchmarking before/after caught (3).
Running the examples caught the `dropzero` blow-up and three broken example
scripts. Executing every README snippet caught six wrong statements (including
`tran`'s return values and the sign of the uniformization formula). Installing from
an empty depot caught that the README's install instructions did not work.

**Pending / notes for next session.**
- No open work items. `refactor` and `master` are level.
- `CODECOV_TOKEN` is **not** set as a repository secret, so the coverage upload
  fails silently (`fail_ci_if_error: false`) and the Codecov badge stays stale.
  Adding it is a repository-settings task.
- `mexp`/`mexpc`/`tran` still use `@assert` for the `rmax` check with the old
  "Time interval is too large" text; only the mixture functions were converted to
  `ArgumentError` with `maxt` reported. Worth unifying if that message ever
  misleads someone.
- `test/test_matrix.jl`'s `unif1`/`unif2`/`unif3` only `println` — zero assertions.
  The real coverage is in `test_regression.jl`.
