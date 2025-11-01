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

