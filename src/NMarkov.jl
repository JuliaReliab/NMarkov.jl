"""
    NMarkov

A Julia package for numerical computation of Markov chains.

## Features
- CTMC (Continuous-Time Markov Chain) analysis
- Matrix exponential computation with uniformization
- Stationary distribution calculation
- Sensitivity analysis
- Quasi-stationary analysis
- Sparse matrix operations (CSR, CSC, COO formats)

## Main Functions
- `mexp`: Matrix exponential computation
- `gth`: Stationary distribution (GTH algorithm)
- `tran`: Transient analysis with rewards
- `stsen`: Sensitivity analysis
- `qstgs`: Quasi-stationary distribution

## Usage
```julia
using NMarkov

# Define CTMC kernel matrix
Q = [-2.0 2.0; 1.0 -1.0]

# Compute stationary distribution
pi = gth(Q)

# Compute matrix exponential
x0 = [1.0, 0.0]
t = 1.0
result = mexp(Q, x0, t)
```
"""
module NMarkov

include("sparsematrix/SparseMatrix.jl")

using .SparseMatrix
using .SparseMatrix: SparseCSR, SparseCSC, SparseCOO, spdiag, spger!

using ZeroOrigin: @origin
using Distributions: Normal, cquantile, UnivariateDistribution, pdf
using DEQuadrature: deint
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: qr
using LinearAlgebra.BLAS: axpy!, gemm!, gemv!, scal!

include("utils.jl")
include("stationary_analysis.jl")
include("sensitivity_analysis.jl")
include("quasistationary_analysis.jl")

include("poisson.jl")
include("mexp.jl")
include("transient_analysis.jl")

include("conv.jl")

# Export main functions
export gth!, gth, stguess, stgs, stpower
export stsen, stsenguess, stsengs, stsenpower
export qstgs, qstpower
export rightbound, poipmf, cpoipmf, convunifstep!
export mexp, mexpc, mexpmix, mexpcmix
export unif, eye, tran

end # module
