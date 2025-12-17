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

