module NMarkov

# using LinearAlgebra
# using Printf

using SparseMatrix: SparseCSR, SparseCSC, SparseCOO, spdiag
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
include("_transient.jl")

include("_mix.jl")

end # module
