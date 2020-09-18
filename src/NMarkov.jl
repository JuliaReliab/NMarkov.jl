module NMarkov

# using LinearAlgebra
# using Printf

using SparseArrays: SparseMatrixCSC
using SparseMatrix: SparseCSR, SparseCSC, SparseCOO, spdiag, nnz
using Distributions: UnivariateDistribution, pdf
using Deformula: deint

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
