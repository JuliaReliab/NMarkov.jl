import LinearAlgebra.BLAS

## overload LinearAlgebra.BLAS.gemm!

# original interface

# function gemm!(transA::AbstractChar, transB::AbstractChar,
#     alpha::Union{($elty), Bool},
#     A::AbstractVecOrMat{$elty}, B::AbstractVecOrMat{$elty},
#     beta::Union{($elty), Bool},
#     C::AbstractVecOrMat{$elty})

# blas level 3

# gemm!(transA::Char, transB::Char, alpha, A, B, beta, C)
#
# Update `C` with matrix-matrix multiplication: ``C := \alpha * A^{transA} * B^{transB} + \beta * C``.
#
# Supports sparse matrix formats (SparseCSR, SparseCSC, SparseCOO) combined with dense matrices.
#
# Arguments:
# - `transA::Char`: 'N' for non-transpose or 'T' for transpose of A
# - `transB::Char`: 'N' for non-transpose or 'T' for transpose of B
# - `alpha`: scalar multiplier
# - `A`: matrix (sparse or dense)
# - `B`: matrix (sparse or dense)
# - `beta`: scalar multiplier for C
# - `C`: output matrix (modified in-place)
#
# Returns:
# - Updated matrix `C`


### CSR

@inbounds function _spgemm!(transA::AbstractChar, transB::AbstractChar,
    alpha::Union{Tv,Bool}, A::SparseCSR{Tv,Ti}, B::Matrix{Tv},
    beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
    if transA == 'N' && transB == 'N'
        m, k = size(A)
        k, n = size(B)
        @. C *= beta
        for i = 1:m
            for z = A.rowptr[i]:A.rowptr[i+1]-1
                l = A.colind[z]
                for j = 1:n
                    C[i,j] += alpha * A.val[z] * B[l,j]
                end
            end
        end
        C
    elseif transA == 'N' && transB == 'T'
        m, k = size(A)
        n, k = size(B)
        @. C *= beta
        for i = 1:m
            for z = A.rowptr[i]:A.rowptr[i+1]-1
                l = A.colind[z]
                for j = 1:n
                    C[i,j] += alpha * A.val[z] * B[j,l]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'N'
        k, m = size(A)
        k, n = size(B)
        @. C *= beta
        for l = 1:k
            for z = A.rowptr[l]:A.rowptr[l+1]-1
                i = A.colind[z]
                for j = 1:n
                    C[i,j] += alpha * A.val[z] * B[l,j]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'T'
        k, m = size(A)
        n, k = size(B)
        @. C *= beta
        for l = 1:k
            for z = A.rowptr[l]:A.rowptr[l+1]-1
                i = A.colind[z]
                for j = 1:n
                    C[i,j] += alpha * A.val[z] * B[j,l]
                end
            end
        end
        C
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

### CSR2

@inbounds function _spgemm!(transA::AbstractChar, transB::AbstractChar,
    alpha::Union{Tv,Bool}, B::Matrix{Tv}, A::SparseCSR{Tv,Ti},
    beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
    if transA == 'N' && transB == 'N'
        m, k = size(B)
        k, n = size(A)
        @. C *= beta
        for l = 1:k
            for z = A.rowptr[l]:A.rowptr[l+1]-1
                j = A.colind[z]
                for i = 1:m
                    C[i,j] += alpha * A.val[z] * B[i,l]
                end
            end
        end
        C
    elseif transA == 'N' && transB == 'T'
        m, k = size(B)
        n, k = size(A)
        @. C *= beta
        for j = 1:n
            for z = A.rowptr[j]:A.rowptr[j+1]-1
                l = A.colind[z]
                for i = 1:m
                    C[i,j] += alpha * A.val[z] * B[i,l]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'N'
        k, m = size(B)
        k, n = size(A)
        @. C *= beta
        for l = 1:k
            for z = A.rowptr[l]:A.rowptr[l+1]-1
                j = A.colind[z]
                for i = 1:m
                    C[i,j] += alpha * A.val[z] * B[l,i]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'T'
        k, m = size(B)
        n, k = size(A)
        @. C *= beta
        for j = 1:n
            for z = A.rowptr[j]:A.rowptr[j+1]-1
                l = A.colind[z]
                for i = 1:m
                    C[i,j] += alpha * A.val[z] * B[l,i]
                end
            end
        end
        C
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

### CSC

@inbounds function _spgemm!(transA::AbstractChar, transB::AbstractChar,
    alpha::Union{Tv,Bool}, A::SparseCSC{Tv,Ti}, B::Matrix{Tv},
    beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
    if transA == 'N' && transB == 'N'
        m, k = size(A)
        k, n = size(B)
        @. C *= beta
        for l = 1:k
            for z = A.colptr[l]:A.colptr[l+1]-1
                i = A.rowind[z]
                for j = 1:n
                    C[i,j] += alpha * A.val[z] * B[l,j]
                end
            end
        end
        C
    elseif transA == 'N' && transB == 'T'
        m, k = size(A)
        n, k = size(B)
        @. C *= beta
        for l = 1:k
            for z = A.colptr[l]:A.colptr[l+1]-1
                i = A.rowind[z]
                for j = 1:n
                    C[i,j] += alpha * A.val[z] * B[j,l]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'N'
        k, m = size(A)
        k, n = size(B)
        @. C *= beta
        for i = 1:m
            for z = A.colptr[i]:A.colptr[i+1]-1
                l = A.rowind[z]
                for j = 1:n
                    C[i,j] += alpha * A.val[z] * B[l,j]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'T'
        k, m = size(A)
        n, k = size(B)
        @. C *= beta
        for i = 1:m
            for z = A.colptr[i]:A.colptr[i+1]-1
                l = A.rowind[z]
                for j = 1:n
                    C[i,j] += alpha * A.val[z] * B[j,l]
                end
            end
        end
        C
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

### CSC2

@inbounds function _spgemm!(transA::AbstractChar, transB::AbstractChar,
    alpha::Union{Tv,Bool}, B::Matrix{Tv}, A::SparseCSC{Tv,Ti},
    beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
    if transA == 'N' && transB == 'N'
        m, k = size(B)
        k, n = size(A)
        @. C *= beta
        for j = 1:n
            for z = A.colptr[j]:A.colptr[j+1]-1
                l = A.rowind[z]
                for i = 1:m
                    C[i,j] += alpha * A.val[z] * B[i,l]
                end
            end
        end
        C
    elseif transA == 'N' && transB == 'T'
        m, k = size(B)
        n, k = size(A)
        @. C *= beta
        for l = 1:k
            for z = A.colptr[l]:A.colptr[l+1]-1
                j = A.rowind[z]
                for i = 1:m
                    C[i,j] += alpha * A.val[z] * B[i,l]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'N'
        k, m = size(B)
        k, n = size(A)
        @. C *= beta
        for j = 1:n
            for z = A.colptr[j]:A.colptr[j+1]-1
                l = A.rowind[z]
                for i = 1:m
                    C[i,j] += alpha * A.val[z] * B[l,i]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'T'
        k, m = size(B)
        n, k = size(A)
        @. C *= beta
        for l = 1:k
            for z = A.colptr[l]:A.colptr[l+1]-1
                j = A.rowind[z]
                for i = 1:m
                    C[i,j] += alpha * A.val[z] * B[l,i]
                end
            end
        end
        C
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

### COO

@inbounds function _spgemm!(transA::AbstractChar, transB::AbstractChar,
    alpha::Union{Tv,Bool}, A::SparseCOO{Tv,Ti}, B::Matrix{Tv},
    beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
    if transA == 'N' && transB == 'N'
        m, k = size(A)
        k, n = size(B)
        @. C *= beta
        for z = 1:SparseArrays.nnz(A)
            i = A.rowind[z]
            l = A.colind[z]
            for j = 1:n
                C[i,j] += alpha * A.val[z] * B[l,j]
            end
        end
        C
    elseif transA == 'N' && transB == 'T'
        m, k = size(A)
        n, k = size(B)
        @. C *= beta
        for z = 1:SparseArrays.nnz(A)
            i = A.rowind[z]
            l = A.colind[z]
            for j = 1:n
                C[i,j] += alpha * A.val[z] * B[j,l]
            end
        end
        C
    elseif transA == 'T' && transB == 'N'
        k, m = size(A)
        k, n = size(B)
        @. C *= beta
        for z = 1:SparseArrays.nnz(A)
            l = A.rowind[z]
            i = A.colind[z]
            for j = 1:n
                C[i,j] += alpha * A.val[z] * B[l,j]
            end
        end
        C
    elseif transA == 'T' && transB == 'T'
        k, m = size(A)
        n, k = size(B)
        @. C *= beta
        for z = 1:SparseArrays.nnz(A)
            l = A.rowind[z]
            i = A.colind[z]
            for j = 1:n
                C[i,j] += alpha * A.val[z] * B[j,l]
            end
        end
        C
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

### COO2

@inbounds function _spgemm!(transA::AbstractChar, transB::AbstractChar,
    alpha::Union{Tv,Bool}, B::Matrix{Tv}, A::SparseCOO{Tv,Ti},
    beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
    if transA == 'N' && transB == 'N'
        m, k = size(B)
        k, n = size(A)
        @. C *= beta
        for z = 1:SparseArrays.nnz(A)
            l = A.rowind[z]
            j = A.colind[z]
            for i = 1:m
                C[i,j] += alpha * A.val[z] * B[i,l]
            end
        end
        C
    elseif transA == 'N' && transB == 'T'
        m, k = size(B)
        n, k = size(A)
        @. C *= beta
        for z = 1:SparseArrays.nnz(A)
            j = A.rowind[z]
            l = A.colind[z]
            for i = 1:m
                C[i,j] += alpha * A.val[z] * B[i,l]
            end
        end
        C
    elseif transA == 'T' && transB == 'N'
        k, m = size(B)
        k, n = size(A)
        @. C *= beta
        for z = 1:SparseArrays.nnz(A)
            l = A.rowind[z]
            j = A.colind[z]
            for i = 1:m
                C[i,j] += alpha * A.val[z] * B[l,i]
            end
        end
        C
    elseif transA == 'T' && transB == 'T'
        k, m = size(B)
        n, k = size(A)
        @. C *= beta
        for z = 1:SparseArrays.nnz(A)
            j = A.rowind[z]
            l = A.colind[z]
            for i = 1:m
                C[i,j] += alpha * A.val[z] * B[l,i]
            end
        end
        C
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

### SparseArrays.SparseMatrixCSC{Tv,Ti}

@inbounds function _spgemm!(transA::AbstractChar, transB::AbstractChar,
    alpha::Union{Tv,Bool}, A::SparseArrays.SparseMatrixCSC{Tv,Ti}, B::Matrix{Tv},
    beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
    if transA == 'N' && transB == 'N'
        m, k = size(A)
        k, n = size(B)
        @. C *= beta
        for l = 1:k
            for z = A.colptr[l]:A.colptr[l+1]-1
                i = A.rowval[z]
                for j = 1:n
                    C[i,j] += alpha * A.nzval[z] * B[l,j]
                end
            end
        end
        C
    elseif transA == 'N' && transB == 'T'
        m, k = size(A)
        n, k = size(B)
        @. C *= beta
        for l = 1:k
            for z = A.colptr[l]:A.colptr[l+1]-1
                i = A.rowval[z]
                for j = 1:n
                    C[i,j] += alpha * A.nzval[z] * B[j,l]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'N'
        k, m = size(A)
        k, n = size(B)
        @. C *= beta
        for i = 1:m
            for z = A.colptr[i]:A.colptr[i+1]-1
                l = A.rowval[z]
                for j = 1:n
                    C[i,j] += alpha * A.nzval[z] * B[l,j]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'T'
        k, m = size(A)
        n, k = size(B)
        @. C *= beta
        for i = 1:m
            for z = A.colptr[i]:A.colptr[i+1]-1
                l = A.rowval[z]
                for j = 1:n
                    C[i,j] += alpha * A.nzval[z] * B[j,l]
                end
            end
        end
        C
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end

### SparseArrays.SparseMatrixCSC{Tv,Ti}2

@inbounds function _spgemm!(transA::AbstractChar, transB::AbstractChar,
    alpha::Union{Tv,Bool}, B::Matrix{Tv}, A::SparseArrays.SparseMatrixCSC{Tv,Ti},
    beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
    if transA == 'N' && transB == 'N'
        m, k = size(B)
        k, n = size(A)
        @. C *= beta
        for j = 1:n
            for z = A.colptr[j]:A.colptr[j+1]-1
                l = A.rowval[z]
                for i = 1:m
                    C[i,j] += alpha * A.nzval[z] * B[i,l]
                end
            end
        end
        C
    elseif transA == 'N' && transB == 'T'
        m, k = size(B)
        n, k = size(A)
        @. C *= beta
        for l = 1:k
            for z = A.colptr[l]:A.colptr[l+1]-1
                j = A.rowval[z]
                for i = 1:m
                    C[i,j] += alpha * A.nzval[z] * B[i,l]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'N'
        k, m = size(B)
        k, n = size(A)
        @. C *= beta
        for j = 1:n
            for z = A.colptr[j]:A.colptr[j+1]-1
                l = A.rowval[z]
                for i = 1:m
                    C[i,j] += alpha * A.nzval[z] * B[l,i]
                end
            end
        end
        C
    elseif transA == 'T' && transB == 'T'
        k, m = size(B)
        n, k = size(A)
        @. C *= beta
        for l = 1:k
            for z = A.colptr[l]:A.colptr[l+1]-1
                j = A.rowval[z]
                for i = 1:m
                    C[i,j] += alpha * A.nzval[z] * B[l,i]
                end
            end
        end
        C
    else
        throw(ErrorException("trans should be 'N' or 'T'"))
    end
end



# Same ambiguity as gemv! (see blas_level2.jl): disambiguate against BLAS's own
# gemm! by providing exact-element-type forwarding methods alongside the
# generic one. The sparse operand may be on either side of the product.
for mat in (:(SparseArrays.SparseMatrixCSC), :SparseCSR, :SparseCSC, :SparseCOO)
    @eval begin
        function LinearAlgebra.BLAS.gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{Tv,Bool}, A::$mat{Tv,Ti}, B::Matrix{Tv},
            beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
            _spgemm!(transA, transB, alpha, A, B, beta, C)
        end
        function LinearAlgebra.BLAS.gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{Tv,Bool}, B::Matrix{Tv}, A::$mat{Tv,Ti},
            beta::Union{Tv,Bool}, C::Matrix{Tv}) where {Tv,Ti}
            _spgemm!(transA, transB, alpha, B, A, beta, C)
        end
    end
    for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
        @eval begin
            function LinearAlgebra.BLAS.gemm!(transA::AbstractChar, transB::AbstractChar,
                alpha::Union{$elty,Bool}, A::$mat{$elty,Ti}, B::Matrix{$elty},
                beta::Union{$elty,Bool}, C::Matrix{$elty}) where {Ti}
                _spgemm!(transA, transB, alpha, A, B, beta, C)
            end
            function LinearAlgebra.BLAS.gemm!(transA::AbstractChar, transB::AbstractChar,
                alpha::Union{$elty,Bool}, B::Matrix{$elty}, A::$mat{$elty,Ti},
                beta::Union{$elty,Bool}, C::Matrix{$elty}) where {Ti}
                _spgemm!(transA, transB, alpha, B, A, beta, C)
            end
        end
    end
end
