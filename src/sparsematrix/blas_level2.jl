import LinearAlgebra.BLAS

## overload LinearAlgebra.BLAS.gemv!

# blas level 2

# gemv!(trans::Char, alpha, A, x, beta, y)
#
# Update `y` with matrix-vector multiplication: ``y := \alpha * A^{trans} * x + \beta * y``.
#
# Supports sparse matrix formats (SparseCSR, SparseCSC, SparseCOO) as well as standard Matrix types.
#
# Arguments:
# - `trans::Char`: 'N' for non-transpose or 'T' for transpose
# - `alpha`: scalar multiplier
# - `A`: matrix (sparse or dense)
# - `x`: vector
# - `beta`: scalar multiplier for y
# - `y`: output vector (modified in-place)
#
# Returns:
# - Updated vector `y`

# spger!(alpha, X, Y, beta, A)
#
# Compute
#    A = alpha * X * Y + beta * A
# where A is a matrix, X and Y are column and row vectors respectively.
#
# Output: The matrix A is directly changed and also it is returned as an output.


## SparseArrays.SparseMatrixCSC
@inbounds function _spgemv!(trans::AbstractChar, alpha::Union{Tv,Bool}, A::SparseArrays.SparseMatrixCSC{Tv,Ti}, X::AbstractVector{Tv},
    beta::Union{Tv,Bool}, Y::AbstractVector{Tv}) where {Tv,Ti}
    if trans == 'N'
        m, n = size(A)
        @. Y *= beta
        for j = 1:n
            for z = A.colptr[j]:A.colptr[j+1]-1
                i = A.rowval[z]
                Y[i] += alpha * A.nzval[z] * X[j]
            end
        end
        Y
    elseif trans == 'T'
        n, m = size(A)
        @. Y *= beta
        for i = 1:m
            for z = A.colptr[i]:A.colptr[i+1]-1
                j = A.rowval[z]
                Y[i] += alpha * A.nzval[z] * X[j]
            end
        end
        Y
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

## CSR
@inbounds function _spgemv!(trans::AbstractChar, alpha::Union{Tv,Bool}, A::SparseCSR{Tv,Ti}, X::AbstractVector{Tv},
    beta::Union{Tv,Bool}, Y::AbstractVector{Tv}) where {Tv,Ti}
    if trans == 'N'
        m, n = size(A)
        @. Y *= beta
        for i = 1:m
            for z = A.rowptr[i]:A.rowptr[i+1]-1
                j = A.colind[z]
                Y[i] += alpha * A.val[z] * X[j]
            end
        end
        Y
    elseif trans == 'T'
        n, m = size(A)
        @. Y *= beta
        for j = 1:n
            for z = A.rowptr[j]:A.rowptr[j+1]-1
                i = A.colind[z]
                Y[i] += alpha * A.val[z] * X[j]
            end
        end
        Y
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

## CSC
@inbounds function _spgemv!(trans::AbstractChar, alpha::Union{Tv,Bool}, A::SparseCSC{Tv,Ti}, X::AbstractVector{Tv},
    beta::Union{Tv,Bool}, Y::AbstractVector{Tv}) where {Tv,Ti}
    if trans == 'N'
        m, n = size(A)
        @. Y *= beta
        for j = 1:n
            for z = A.colptr[j]:A.colptr[j+1]-1
                i = A.rowind[z]
                Y[i] += alpha * A.val[z] * X[j]
            end
        end
        Y
    elseif trans == 'T'
        n, m = size(A)
        @. Y *= beta
        for i = 1:m
            for z = A.colptr[i]:A.colptr[i+1]-1
                j = A.rowind[z]
                Y[i] += alpha * A.val[z] * X[j]
            end
        end
        Y
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

## COO
@inbounds function _spgemv!(trans::AbstractChar, alpha::Union{Tv,Bool}, A::SparseCOO{Tv,Ti}, X::AbstractVector{Tv},
    beta::Union{Tv,Bool}, Y::AbstractVector{Tv}) where {Tv,Ti}
    if trans == 'N'
        m, n = size(A)
        @. Y *= beta
        for z = 1:SparseArrays.nnz(A)
            i = A.rowind[z]
            j = A.colind[z]
            Y[i] += alpha * A.val[z] * X[j]
        end
        Y
    elseif trans == 'T'
        n, m = size(A)
        @. Y *= beta
        for z = 1:SparseArrays.nnz(A)
            j = A.rowind[z]
            i = A.colind[z]
            Y[i] += alpha * A.val[z] * X[j]
        end
        Y
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end


@inbounds begin
    function spger!(alpha::Union{Tv,Bool}, X::AbstractVector{Tv}, Y::AbstractVector{Tv}, beta::Union{Tv,Bool}, A::Matrix{Tv}) where {Tv}
        m, n = size(A)
        @. A *= beta
        for j = 1:n
            for i = 1:m
                A[i,j] += alpha * X[i] * Y[j]
            end
        end
        A
    end
end

## SparseArrays.SparseMatrixCSC
@inbounds function spger!(alpha::Union{Tv,Bool}, X::AbstractVector{Tv}, Y::AbstractVector{Tv}, beta::Union{Tv,Bool}, A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    @. A.nzval *= beta
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[z]
            A.nzval[z] += alpha * X[i] * Y[j]
        end
    end
    A
end

## CSR
@inbounds function spger!(alpha::Union{Tv,Bool}, X::AbstractVector{Tv}, Y::AbstractVector{Tv}, beta::Union{Tv,Bool}, A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    @. A.val *= beta
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            A.val[z] += alpha * X[i] * Y[j]
        end
    end
    A
end

## CSC
@inbounds function spger!(alpha::Union{Tv,Bool}, X::AbstractVector{Tv}, Y::AbstractVector{Tv}, beta::Union{Tv,Bool}, A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    @. A.val *= beta
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            A.val[z] += alpha * X[i] * Y[j]
        end
    end
    A
end

## COO
@inbounds function spger!(alpha::Union{Tv,Bool}, X::AbstractVector{Tv}, Y::AbstractVector{Tv}, beta::Union{Tv,Bool}, A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    @. A.val *= beta
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        A.val[z] += alpha * X[i] * Y[j]
    end
    A
end


# BLAS defines gemv! for its own element types over AbstractVecOrMat, and our
# sparse types are AbstractMatrix subtypes, so a single fully generic method
# would be ambiguous with it for Float32/Float64. Forward from an exact-type
# method (which wins on specificity) as well as from the generic one.
for mat in (:(SparseArrays.SparseMatrixCSC), :SparseCSR, :SparseCSC, :SparseCOO)
    @eval function LinearAlgebra.BLAS.gemv!(trans::AbstractChar, alpha::Union{Tv,Bool},
        A::$mat{Tv,Ti}, X::AbstractVector{Tv}, beta::Union{Tv,Bool},
        Y::AbstractVector{Tv}) where {Tv,Ti}
        _spgemv!(trans, alpha, A, X, beta, Y)
    end
    for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
        @eval function LinearAlgebra.BLAS.gemv!(trans::AbstractChar, alpha::Union{$elty,Bool},
            A::$mat{$elty,Ti}, X::AbstractVector{$elty}, beta::Union{$elty,Bool},
            Y::AbstractVector{$elty}) where {Ti}
            _spgemv!(trans, alpha, A, X, beta, Y)
        end
    end
end
