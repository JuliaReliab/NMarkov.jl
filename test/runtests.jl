using NMarkov
using Test
using Printf
using Distributions
using SparseMatrix
using SparseArrays

import NMarkov: rightbound, poipmf, cpoipmf, convunifstep!

include("test_stationary.jl")

include("test_poisson.jl")
include("test_matrix.jl")

include("test_mexp.jl")
include("test_mix.jl")

include("test_conv.jl")

include("test_transient.jl")

