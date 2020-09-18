module NMarkov

# using LinearAlgebra
# using Printf

using Origin: @origin
using SparseArrays: SparseMatrixCSC, nnz
using SparseMatrix: SparseCSR, SparseCSC, SparseCOO, spdiag
using Distributions: UnivariateDistribution, pdf, cquantile, Normal, Poisson
using Deformula: deint

export @axpy, @scal, @dot
export itime
export poipmf!, poipmf, cpoipmf!, cpoipmf, rightbound
export unifstep!, cunifstep!, convunifstep!

export unif
export gth!, gth
export stsenguess, stsengs, stsenpower
export stguess, stgs, stpower
export qstgs, qstpower
export mexp, mexpc
export mexp, mexpmix, mexpc, mexpcmix

include("_common.jl")
include("_unif.jl")
include("_gth.jl")
include("_gsstep.jl")
include("_stationary_iterative.jl")
include("_sensitivity_iterative.jl")
include("_quasistationary_iterative.jl")

include("_poisson.jl")
include("_forward_backward.jl")
include("_transient.jl")

include("_mix.jl")

include("_conv.jl")

end # module
