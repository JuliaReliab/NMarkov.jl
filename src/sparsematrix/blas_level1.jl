import LinearAlgebra.BLAS

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

for Tv in [:Float64]
    @eval begin
        function fill!(a::Union{$Tv,Bool}, X::AbstractVector{$Tv})
            @. X = a
            X
        end

        function fill!(a::Union{$Tv,Bool}, A::Matrix{$Tv})
            @. A = a
            A
        end

        function fill!(a::Union{$Tv,Bool}, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            @. A.nzval = a
            A
        end

        function fill!(a::Union{$Tv,Bool}, A::AbstractSparseM{$Tv,Ti}) where {Ti}
            @. A.val = a
            A
        end
    end

    @eval begin
        function LinearAlgebra.BLAS.scal!(a::Union{$Tv,Bool}, X::AbstractVector{$Tv})
            @. X *= a
            X
        end

        function LinearAlgebra.BLAS.scal!(a::$Tv, X::AbstractVector{$Tv})
            @. X *= a
            X
        end

        function LinearAlgebra.BLAS.scal!(a::Union{$Tv,Bool}, A::Matrix{$Tv})
            @. A *= a
            A
        end

        function LinearAlgebra.BLAS.scal!(a::$Tv, A::Matrix{$Tv})
            @. A *= a
            A
        end

        function LinearAlgebra.BLAS.scal!(a::Union{$Tv,Bool}, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            @. A.nzval *= a
            A
        end

        function LinearAlgebra.BLAS.scal!(a::$Tv, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            @. A.nzval *= a
            A
        end

        function LinearAlgebra.BLAS.scal!(a::Union{$Tv,Bool}, A::AbstractSparseM{$Tv,Ti}) where {Ti}
            @. A.val *= a
            A
        end

        function LinearAlgebra.BLAS.scal!(a::$Tv, A::AbstractSparseM{$Tv,Ti}) where {Ti}
            @. A.val *= a
            A
        end
    end

    @eval begin
        function LinearAlgebra.BLAS.axpy!(a::Union{$Tv,Bool}, X::SparseArrays.SparseMatrixCSC{$Tv,Ti}, Y::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            LinearAlgebra.BLAS.axpy!(a, X.nzval, Y.nzval)
            Y
        end

        function LinearAlgebra.BLAS.axpy!(a::Union{$Tv,Bool}, X::AbstractSparseM{$Tv,Ti}, Y::AbstractSparseM{$Tv,Ti}) where {Ti}
            LinearAlgebra.BLAS.axpy!(a, X.val, Y.val)
            Y
        end
    end
end
