"""
    Diag{Tv,Ti}

Type representing the diagonal elements of a matrix with efficient storage.

### Fields
- `index::Vector{Ti}`: indices into the value array for diagonal positions (0 if element is not present)
- `val::Vector{Tv}`: actual values from the source matrix

### Example
```julia
A = Matrix([1.0 0.0; 0.0 2.0])
d = spdiag(A)  # Extract diagonal
d[1]  # Returns 1.0
```
"""
struct Diag{Tv <: AbstractFloat, Ti <: Integer} <: AbstractArray{Tv,1}
    index::Vector{Ti}
    val::Vector{Tv}
end

"""
    spdiag(A)

Create a type of Diag to represent the diagonal matrix of A.
The type of A is allowed to be Matrix, SparseArrays.SparseMatrixCSC, SparseCSR,
SparseCSC, SparseCOO.

### Input

- `A` -- an instance of matrix; Matrix, SparseArrays.SparseMatrixCSC, SparseCSR, SparseCSC, SparseCOO.

### Output

An instance of Diag which a structure.
"""
function spdiag(A::Matrix{Tv}) where {Tv}
    m, n = size(A)
    index = Vector{Int}(undef, min(m,n))
    z = 1
    @inbounds for i = 1:min(m,n)
        index[i] = z
        z += m+1
    end
    val = reshape(A, length(A))
    Diag(index, val)
end

function spdiag(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    index = zeros(Ti, min(m,n))
    val = reshape(A.nzval, length(A.nzval))
    @inbounds for j = 1:min(m,n)
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[z]
            if i == j
                index[i] = z
                break
            elseif i > j
                break
            end
        end
    end
    Diag(index, val)
end

function spdiag(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    index = zeros(Ti, min(m,n))
    val = reshape(A.val, length(A.val))
    @inbounds for i = 1:min(m,n)
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            if i == j
                index[i] = z
                break
            elseif i < j
                break
            end
        end
    end
    Diag(index, val)
end

function spdiag(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    index = zeros(Ti, min(m,n))
    val = reshape(A.val, length(A.val))
    @inbounds for j = 1:min(m,n)
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            if i == j
                index[i] = z
                break
            elseif i > j
                break
            end
        end
    end
    Diag(index, val)
end

function spdiag(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    index = zeros(Ti, min(m,n))
    val = reshape(A.val, length(A.val))
    @inbounds for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        if i == j
            index[i] = z
        end
    end
    Diag(index, val)
end

"""
    hasfulldiag(A)
    hasfulldiag(d::Diag)

Whether every diagonal position of `A` is present in its sparsity pattern.

`Diag` records 0 in `index` for a position the pattern does not store, so this is
a scan of that index vector. A dense `Matrix` always answers `true`.

Use this to skip `adddiag` when there is nothing to add: checking is O(n) on top
of the `spdiag` the caller usually needs anyway, whereas `adddiag` on a CSR/CSC
matrix has to rebuild the index arrays.
"""
hasfulldiag(d::Diag) = all(!iszero, d.index)
hasfulldiag(A) = hasfulldiag(spdiag(A))

function Base.length(A::Diag{Tv,Ti}) where {Tv,Ti}
    return length(A.index)
end

function Base.size(A::Diag{Tv,Ti}) where {Tv,Ti}
    return (length(A.index),)
end

function Base.getindex(A::Diag{Tv,Ti}, i::Integer) where {Tv,Ti}
    z = A.index[i]
    if z == 0
        return Tv(0)
    else
        return A.val[z]
    end
end

function Base.setindex!(A::Diag{Tv,Ti}, value, i::Integer) where {Tv,Ti}
    z = A.index[i]
    if z == 0
        throw(ArgumentError(
            "cannot write the diagonal element ($i,$i): it is not stored in the " *
            "sparsity pattern. Use `adddiag` to add the missing structural zeros first."))
    end
    A.val[z] = convert(Tv, value)
end
