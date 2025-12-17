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
