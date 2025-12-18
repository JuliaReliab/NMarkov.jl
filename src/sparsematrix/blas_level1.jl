import LinearAlgebra.BLAS

## overload LinearAlgebra.BLAS.scal!

# blas level 1

"""
    scal!(a, A)

Scale all elements of array A by scalar value a: ``A := a * A``.

Works with vectors, dense matrices, and sparse matrices (SparseCSR, SparseCSC, SparseCOO, SparseMatrixCSC).

### Arguments
- `a`: scalar multiplier
- `A`: array/matrix to scale (modified in-place)

### Returns
- Scaled array `A`
"""
for Tv in [:Float64]
    @eval begin
        ## Matrix
        """
            fill!(x, A)
        
        Put a value x to all the elements of array of A.
        
        ### Output

        The array A is directly changed and it is returned as on output.
        """
        function fill!(a::Union{$Tv,Bool}, X::AbstractVector{$Tv})
            @. X = a
            X
        end

        ## Matrix
        function fill!(a::Union{$Tv,Bool}, A::Matrix{$Tv})
            @. A = a
            A
        end

        ## SparseArrays.SparseMatrixCSC
        function fill!(a::Union{$Tv,Bool}, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            @. A.nzval = a
            A
        end

        ## CSR/CSC/COO
        function fill!(a::Union{$Tv,Bool}, A::AbstractSparseM{$Tv,Ti}) where {Ti}
            @. A.val = a
            A
        end
    end

    @eval begin
        ## Vector
        function LinearAlgebra.BLAS.scal!(a::Union{$Tv,Bool}, X::AbstractVector{$Tv})
            @. X *= a
            X
        end

        ## Vector
        function LinearAlgebra.BLAS.scal!(a::$Tv, X::AbstractVector{$Tv})
            @. X *= a
            X
        end

        ## Matrix
        function LinearAlgebra.BLAS.scal!(a::Union{$Tv,Bool}, A::Matrix{$Tv})
            @. A *= a
            A
        end

        ## Matrix
        function LinearAlgebra.BLAS.scal!(a::$Tv, A::Matrix{$Tv})
            @. A *= a
            A
        end

        ## SparseArrays.SparseMatrixCSC
        function LinearAlgebra.BLAS.scal!(a::Union{$Tv,Bool}, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            @. A.nzval *= a
            A
        end

        ## SparseArrays.SparseMatrixCSC
        function LinearAlgebra.BLAS.scal!(a::$Tv, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            @. A.nzval *= a
            A
        end

        ## CSR/CSC/COO
        function LinearAlgebra.BLAS.scal!(a::Union{$Tv,Bool}, A::AbstractSparseM{$Tv,Ti}) where {Ti}
            @. A.val *= a
            A
        end

        ## CSR/CSC/COO
        function LinearAlgebra.BLAS.scal!(a::$Tv, A::AbstractSparseM{$Tv,Ti}) where {Ti}
            @. A.val *= a
            A
        end
    end

    @eval begin
        ## SparseArrays.SparseMatrixCSC
        function LinearAlgebra.BLAS.axpy!(a::Union{$Tv,Bool}, X::SparseArrays.SparseMatrixCSC{$Tv,Ti}, Y::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            LinearAlgebra.BLAS.axpy!(a, X.nzval, Y.nzval)
            Y
        end

        ## CSR/CSC/COO
        function LinearAlgebra.BLAS.axpy!(a::Union{$Tv,Bool}, X::AbstractSparseM{$Tv,Ti}, Y::AbstractSparseM{$Tv,Ti}) where {Ti}
            LinearAlgebra.BLAS.axpy!(a, X.val, Y.val)
            Y
        end
    end
end
