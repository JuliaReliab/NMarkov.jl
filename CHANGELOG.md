# NMarkov 0.5.0

## Breaking

- **Requires Julia 1.10 or later** (`[compat] julia = "1.10"`, previously
  `"1.6"`). 1.10 is the current LTS; 1.6–1.9 are no longer supported.

## CI

- The test matrix is now `["min", "1"]` — the lower bound declared in
  `Project.toml` and the latest stable release, both resolved by
  `julia-actions/setup-julia`. It was `1.6 / 1.9 / 1.10`, which tested an EOL
  version (1.9) and **no currently released Julia at all**: 1.11 and 1.12 were
  never exercised. Two jobs instead of three.
- Added a `concurrency` group so a new push to a pull request cancels the
  superseded run. Runs on a branch are never cancelled, so the master run — the
  one the README badge reads — always completes.
- Replaced the hand-rolled `actions/cache` step (which cached only
  `~/.julia/artifacts`) with `julia-actions/cache@v3`, which caches the whole
  depot.
- The `DEQuadrature` URL install is now conditional on `VERSION < v"1.11"`:
  1.11+ resolves it from the `[sources]` entry in `Project.toml`, earlier
  versions ignore that section.
- Coverage is now processed and uploaded (`julia-processcoverage` +
  `codecov-action`) from the latest-stable job. `julia-runtest` was already
  collecting it; the report was simply discarded. Uploading needs a
  `CODECOV_TOKEN` repository secret.

## Documentation

Verified every code example in `README.md` against the current API by running
it. Corrected:

- `stgs(coo)` was documented as working; `stgs`, `stsengs` and `qstgs` accept
  only `SparseMatrixCSC` and `SparseCSC`. (The transient functions do accept
  every format.)
- `tran`'s third and fourth return values were described as one state vector per
  time point. They are single vectors: the state at the last time point and the
  cumulative time in each state over the whole interval. Only the first two
  returns are per time point; `mexpc` gives the per-time-point vectors.
- The uniformization formula was written `P = I - Q / q`; it is `P = I + Q / q`.
  With the minus sign the entries fall outside `[0,1]`. Also documented that
  `q = ufact * max|Q_ii|` with `ufact` defaulting to 1.01, which was omitted.
- The quasi-stationary power-method example passed the unscaled exit vector to
  `qstpower`; it takes `xi / q`, and the eigenvalue it returns is scaled to
  match.
- `SparseELL` is not a type name; the exported types are `SparseELL1` and
  `SparseELL2`.
- Both badges pointed at `okamumu/NMarkov.jl`, which does not exist. The
  repository is `JuliaReliab/NMarkov.jl`.
- Dropped the `ZeroOrigin` install line: it is registered in General and
  resolves on its own. `DEQuadrature` and NMarkov itself are still unregistered
  and need their URLs.

# NMarkov 0.4.1

Fixes from a full code review of 0.4.0. Behaviour changes are noted explicitly.

## Wrong results

- `unif` now returns a stochastic `P` for sparse matrices whose diagonal is not
  fully stored. A CTMC with an absorbing state has a zero diagonal entry that
  dense-to-sparse conversion drops; `unif` used to emit a warning, skip that
  entry, and return a non-stochastic matrix, so every downstream `mexp`/`mexpc`/
  `tran`/`conv` result was silently wrong. Added `adddiag` to fill in the
  missing structural zeros.
- `convunifstep!` dropped the last Poisson term from the instantaneous result
  `z` while still normalising by a weight that included it, biasing `z` by
  O(eps) instead of the expected truncation error.
- `rightbound` no longer loops forever for element types narrower than Float64
  (the running sum stops short of `1 - q` in Float32 and the tail test never
  fired).
- `mexpc(Q, x, ts)` accepted an unsorted time vector and returned `NaN` after an
  out-of-bounds write; `mexp(Q, x, ts)` silently sorted and returned results in
  a different order than requested. Unsorted or negative time points are now an
  `ArgumentError` in `mexp`, `mexpc` and `tran`.
- `stguess` returned `NaN` for a matrix with a zero diagonal entry (an ordinary
  periodic DTMC), which no convergence test could satisfy; it now falls back to
  the uniform guess.
- `gth`/`gth!` throw `ArgumentError` for a chain with an absorbing state instead
  of returning `NaN`.
- The Gauss-Seidel solvers `stgs`, `stsengs` and `qstgs` reject a zero diagonal
  entry (`ArgumentError`) instead of dividing by it and returning an all-`NaN`
  vector with `conv = false`. Gauss-Seidel inverts the diagonal, so a state that
  never leaves makes the iteration undefined — and structurally so: the column
  equation of such a state does not contain its own unknown, which the
  normalisation fixes instead, so there is no fixed point to iterate towards.
  Storing an explicit zero on the diagonal does not help. Chains with an
  absorbing state need a reducible-chain method.
  For `qstgs` the usual trigger is passing the full generator rather than the
  block restricted to the transient states; the docstring example did exactly
  that and returned `NaN`. It now shows the transient block plus the exit rates,
  matching `examples/05_quasi_stationary_analysis.jl`.
- `qstgs` returned a `gam` computed from the iterate before the last sweep; it
  is now consistent with the returned vector, as in `qstpower`.
- Removed a definition of `Base.iszero(::Float64)`, which redefined `iszero` for
  every `Float64` in the session, for all packages.

## Memory safety

`poipmf!`/`cpoipmf!` seed the recurrence at `prob[mode]` with
`mode = floor(lambda)` and walk it outwards. Under `@origin (prob => left)` that
is the physical index `mode - left + 1`, and the bodies run under `@inbounds`,
so a `mode` outside the requested domain wrote past the buffer. The domain and
`lambda` are now validated up front (`ArgumentError`), which turns these into
clean errors:

- `poipmf(5.05, 0)` and `cpoipmf(5.05, 2)` — domain ends below the mode;
  previously a segmentation fault.
- `poipmf(-1.0, 5)` — a negative mean gives `mode = -1`; previously a
  segmentation fault. `rightbound` also rejects a negative mean now.
- `poipmf(5.05, 20, left=10)` — domain starts above the mode; previously wrote
  five elements *in front of* the array and returned plausible numbers, so the
  heap corruption was silent.

`convunifstep!` likewise validates its `range` against the length of the Poisson
vector, which it reads as `poi[left]..poi[right]` inside an `@inbounds` block.

Calls made inside the package were never affected: they use `left = 0` and
`right = rightbound(lambda, eps)`, which is monotone in `lambda` and always at
least the mode.

## Broken code paths

- `gsstep!` for `SparseCSR` read an undefined variable and raised
  `UndefVarError` on every call.
- `mexp`/`mexpc` recursed infinitely (`StackOverflowError`) when the element
  type was `Float32` or `Float16`, and rejected a `Float64` time when the other
  arguments needed conversion. The public methods are now a single converting
  entry point delegating to an internal kernel.
- The type-conversion path of `tran` flattened matrix arguments with `vec`, so
  its matrix methods were unreachable for arguments needing conversion.
- Added the missing `mexpc(Q, x, ts)` conversion wrapper and the missing
  `tran(Q, x::Vector, r::Matrix, ts, forward=:N)` method.
- `eye(A, Tv)` ignored `Tv`.
- `stsengs` dropped the element type through its default `x0`.
- `_tocsr`/`_tocsc`/`_tocoo` failed for index types other than `Int`.

## Element-type genericity

- `Float32`, `Float16`, `Int` and `BigFloat` now work end to end. Removed the
  `Float64`-only method generation in the `SparseMatrix` BLAS routines and the
  hardcoded `Float64` literals in the `mexp`/`tran`/`conv` kernels.
- `_todense` no longer rounds through `Float64`, so a dense→sparse→dense round
  trip is exact for wider element types.
- The default tolerance of `dtmcstcheck`/`ctmcstcheck` follows the element type;
  the fixed `1.0e-8` was below Float32 resolution, so no Float32 stationary
  vector could pass.

## Behaviour changes to be aware of

- `AbstractSparseM` now honours the `AbstractMatrix` contract: `length(A)` is
  `m*n` (not `nnz`), `eachindex` walks all positions, and `A[i,j]` works. Code
  that used `length(A)` or linear indexing to reach the stored entries must use
  `nnz(A)` and `A.val` instead.
- Writing an element that is not in the sparsity pattern raises `ArgumentError`
  instead of being silently dropped with a warning.
- `unif` of the zero matrix returns `qv = 1` and `P = I` (previously
  `qv = 1.0e-12`); results are unchanged up to rounding.
- The iterative solvers (`stgs`, `stpower`, `stsengs`, `stsenpower`, `qstgs`,
  `qstpower`) emit a warning when they hit `maxiter`. The returned `conv` flag
  is unchanged.
- `stsengs`/`stsenpower` measure the relative error against the largest
  magnitude of the iterate rather than its maximum; a sensitivity vector sums to
  zero, so the old normaliser was not a meaningful scale and was `0/0` for a
  zero vector.
- `trans` throws `ArgumentError` for a symbol other than `:N`/`:T` instead of
  returning `nothing`.
- Corrected the `:N`/`:T` descriptions in the `mexp`/`mexpc` docstrings, which
  had the two directions the wrong way round.
- Removed ~350 lines of commented-out dead code and the unused `src/Sojourn.jl`.

# NMarkov 0.4.0

- Integrate SparseMatrix module into NMarkov
- Add type flexibility to `mexp`, `mexpc`, and `tran` functions
  - Now accept `Int`, `Float32`, and other numeric types for time and vector arguments
  - Automatically convert to the matrix element type
  - Support the vectors with `Int` elements (e.g., `[1, 0, 0]`)
- Add comprehensive examples to README.md
  - Steady-state analysis (GTH, GS, power, sensitivity, quasi-stationary)
  - Transient analysis and reward models
- Move notebook to examples directory
- Add GitHub Actions CI/CD workflow for automated testing on Julia 1.9 and 1.10
- Add type flexibility tests for `mexp` and `tran` functions

# NMarkov 0.3.6

- fix a bug in stsen for dense Q
- use ZeroOrigin.jl instead of Origin.jl (The latter is deprecated)

# NMarkov 0.3.5

- Change the version of sparsematrix

# NMarkov 0.3.4

- remove unifstep function

# NMarkov 0.3.3

- add stsen; sensitivity vector for CTMC with QR

# NMarkov 0.3.2

- change the computation of rerror for iterative methods
    - if the probability is zero, the corresponding element becomes NaN in rerror

# NMarkov 0.3.1

- add the case where Q is zero matrix
- add eye function

