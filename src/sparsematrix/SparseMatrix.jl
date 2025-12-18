module SparseMatrix

# import LinearAlgebra #: Adjoint
# import LinearAlgebra.BLAS
# import SparseArrays #: SparseMatrixCSC, nnz

# Sparse matrix types
export AbstractSparseM
export SparseCSR, SparseCSC, SparseCOO
export SparseELL1, SparseELL2

# BLAS operations
export fill!, spger!

# Diagonal and block operations
export spdiag
export BlockCOO, block

include("sparse.jl")

include("blas_level1.jl")
include("blas_level2.jl")
include("blas_level3.jl")

include("mul.jl")
include("spdiag.jl")

include("block.jl")

include("ell.jl")
end
