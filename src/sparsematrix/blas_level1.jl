import LinearAlgebra.BLAS
import Base: fill!

## overload LinearAlgebra.BLAS.scal!

# blas level 1

"""
    fill!(x, A)

Put a value x to all the elements of array A.

### Arguments
- `x`: scalar value to fill
- `A`: array/matrix to fill (modified in-place)

### Returns
- The array `A` is directly changed and returned as output
"""
fill!

"""
    scal!(a, A)

Scale all elements of array A by scalar value a: `A := a * A`.

Works with vectors, dense matrices, and sparse matrices (SparseCSR, SparseCSC, SparseCOO, SparseMatrixCSC).

### Arguments
- `a`: scalar multiplier
- `A`: array/matrix to scale (modified in-place)

### Returns
- Scaled array `A`

### Algorithm
For each element: `A[i] = a * A[i]`

Optimized for sparse matrices by only scaling non-zero elements.
"""
scal!

"""
    axpy!(a, X, Y)

Compute `Y := a*X + Y` for arrays X and Y.

### Arguments
- `a`: scalar multiplier
- `X`: source array
- `Y`: destination array (modified in-place)

### Returns
- Modified array `Y`
"""
axpy!

function fill!(a::Union{Tv,Bool}, X::AbstractVector{Tv}) where {Tv}
    @. X = a
    X
end

function fill!(a::Union{Tv,Bool}, A::Matrix{Tv}) where {Tv}
    @. A = a
    A
end

function fill!(a::Union{Tv,Bool}, A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    @. A.nzval = a
    A
end

function fill!(a::Union{Tv,Bool}, A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    @. A.val = a
    A
end

_spscal!(a, X::AbstractVector) = (@. X *= a; X)
_spscal!(a, A::Matrix) = (@. A *= a; A)
_spscal!(a, A::SparseArrays.SparseMatrixCSC) = (@. A.nzval *= a; A)
_spscal!(a, A::AbstractSparseM) = (@. A.val *= a; A)

_spaxpy!(a, X::SparseArrays.SparseMatrixCSC, Y::SparseArrays.SparseMatrixCSC) =
    (LinearAlgebra.BLAS.axpy!(a, X.nzval, Y.nzval); Y)
_spaxpy!(a, X::AbstractSparseM, Y::AbstractSparseM) =
    (LinearAlgebra.BLAS.axpy!(a, X.val, Y.val); Y)

# BLAS already owns `scal!`/`axpy!` for its own element types over AbstractArray,
# and these containers are AbstractArrays, so a method whose element type is a
# free parameter is not resolvable against it (the parameter appears in more
# than one argument, which makes specificity inconclusive). Give each BLAS
# element type an exact-type method that wins outright, and keep one generic
# method for the element types BLAS does not cover.
# BLAS declares `scal!(a::T, X::AbstractArray{T})` with a bare `T` for the
# scalar, so a `Union{T,Bool}` scalar is wider there and narrower in the array
# argument: neither method wins. Emit a bare-scalar method per BLAS element type
# to settle it, plus a Bool one (Bool never overlaps BLAS's scalar type).
for cont in (:AbstractVector, :Matrix)
    @eval function LinearAlgebra.BLAS.scal!(a::Union{Tv,Bool}, A::$cont{Tv}) where {Tv}
        _spscal!(a, A)
    end
    for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
        @eval begin
            LinearAlgebra.BLAS.scal!(a::$elty, A::$cont{$elty}) = _spscal!(a, A)
            LinearAlgebra.BLAS.scal!(a::Bool, A::$cont{$elty}) = _spscal!(a, A)
        end
    end
end

for cont in (:(SparseArrays.SparseMatrixCSC), :AbstractSparseM)
    @eval function LinearAlgebra.BLAS.scal!(a::Union{Tv,Bool}, A::$cont{Tv,Ti}) where {Tv,Ti}
        _spscal!(a, A)
    end
    for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
        @eval begin
            LinearAlgebra.BLAS.scal!(a::$elty, A::$cont{$elty,Ti}) where {Ti} = _spscal!(a, A)
            LinearAlgebra.BLAS.scal!(a::Bool, A::$cont{$elty,Ti}) where {Ti} = _spscal!(a, A)
        end
    end
end

function LinearAlgebra.BLAS.axpy!(a::Union{Tv,Bool}, X::SparseArrays.SparseMatrixCSC{Tv,Ti}, Y::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    _spaxpy!(a, X, Y)
end

function LinearAlgebra.BLAS.axpy!(a::Union{Tv,Bool}, X::AbstractSparseM{Tv,Ti}, Y::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    _spaxpy!(a, X, Y)
end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function LinearAlgebra.BLAS.axpy!(a::Union{$elty,Bool}, X::SparseArrays.SparseMatrixCSC{$elty,Ti}, Y::SparseArrays.SparseMatrixCSC{$elty,Ti}) where {Ti}
            _spaxpy!(a, X, Y)
        end
        function LinearAlgebra.BLAS.axpy!(a::Union{$elty,Bool}, X::AbstractSparseM{$elty,Ti}, Y::AbstractSparseM{$elty,Ti}) where {Ti}
            _spaxpy!(a, X, Y)
        end
    end
end

# `LinearAlgebra.axpy!` is a different function from `LinearAlgebra.BLAS.axpy!`,
# and its generic fallback would walk every position of the matrix and try to
# write the structural zeros. Send it to the same stored-entry implementation.
function LinearAlgebra.axpy!(a::Number, X::AbstractSparseM{Tv,Ti}, Y::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    _spaxpy!(convert(Tv, a), X, Y)
end
