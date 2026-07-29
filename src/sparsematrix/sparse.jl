import LinearAlgebra
import SparseArrays

"""
    AbstractSparseM{Tv,Ti} <: AbstractMatrix{Tv}

Abstract tyoe for sparse matrix.

### Notes

Every concrete `AbstractSparseM` must have the following fields:
- `m`: the number of rows whose type is Ti
- `n`: the number of columns whose type is Ti
- `val`: a vector of non-zero elements whose type is Tv
"""
abstract type AbstractSparseM{Tv,Ti} <: AbstractMatrix{Tv} end

"""
    SparseCSR{Tv,Ti} <: AbstractSparseM{Tv,Ti}

Type that represents a sparse matrix with CSR (Compressed Sparse Row) format.

### Fields
- `m::Ti`: the number of rows
- `n::Ti`: the number of columns
- `val::Vector{Tv}`: non-zero elements
- `rowptr::Vector{Ti}`: row pointer array indicating where each row starts in val and colind arrays
- `colind::Vector{Ti}`: column indices for each non-zero element

### Example
```julia
# Create a CSR matrix from COO format
coo = SparseCOO(3, 3, [1.0, 2.0, 3.0], [1, 2, 3], [1, 2, 3])
csr = SparseCSR(coo)
```
"""
struct SparseCSR{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{Tv}
    rowptr::Vector{Ti}
    colind::Vector{Ti}
end

"""
    SparseCSC{Tv,Ti} <: AbstractSparseM{Tv,Ti}

Type that represents a sparse matrix with CSC (Compressed Sparse Column) format.

### Fields
- `m::Ti`: the number of rows
- `n::Ti`: the number of columns
- `val::Vector{Tv}`: non-zero elements
- `colptr::Vector{Ti}`: column pointer array indicating where each column starts in val and rowind arrays
- `rowind::Vector{Ti}`: row indices for each non-zero element

### Example
```julia
coo = SparseCOO(3, 3, [1.0, 2.0, 3.0], [1, 2, 3], [1, 2, 3])
csc = SparseCSC(coo)
```
"""
struct SparseCSC{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{Tv}
    colptr::Vector{Ti}
    rowind::Vector{Ti}
end

"""
    SparseCOO{Tv,Ti} <: AbstractSparseM{Tv,Ti}

Type that represents a sparse matrix with COO (Coordinate) format.

### Fields
- `m::Ti`: the number of rows
- `n::Ti`: the number of columns
- `val::Vector{Tv}`: non-zero elements
- `rowind::Vector{Ti}`: row indices for each non-zero element
- `colind::Vector{Ti}`: column indices for each non-zero element

### Notes
COO format is useful for efficient construction of sparse matrices. Convert to CSR or CSC for efficient computation.

### Example
```julia
# Create a 3x3 sparse identity matrix in COO format
rowind = [1, 2, 3]
colind = [1, 2, 3]
val = [1.0, 1.0, 1.0]
coo = SparseCOO(3, 3, val, rowind, colind)
```
"""
struct SparseCOO{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{Tv}
    rowind::Vector{Ti}
    colind::Vector{Ti}
end

"""
    SparseCSR(A)

Create a sparse matrix with CSR format from the matrix `A`.
The matrix `A` is allowed to be Matrix, SparseArrays.SparseMatrixCSC, SparseCSR, SparseCSC, SparseCOO and BlockCOO.
"""
SparseCSR(A::Matrix{Tv}) where {Tv} = _tocsr(A, Int)
SparseCSR(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = copy(A)
SparseCSR(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _tocsr(_tocoo(A))
SparseCSR(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _tocsr(A)
SparseCSR(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocsr(_tocoo(_tocsc(A)))

"""
    SparseCSC(A)

Create a sparse matrix with CSC format from the matrix `A`.
The matrix `A` is allowed to be Matrix, SparseArrays.SparseMatrixCSC, SparseCSR, SparseCSC, SparseCOO and BlockCOO.
"""
SparseCSC(A::Matrix{Tv}) where {Tv} = _tocsc(A, Int)
SparseCSC(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _tocsc(_tocoo(A))
SparseCSC(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = copy(A)
SparseCSC(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _tocsc(A)
SparseCSC(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocsc(A)

"""
    SparseCOO(A)

Create a sparse matrix with COO format from the matrix `A`.
The matrix `A` is allowed to be Matrix, SparseArrays.SparseMatrixCSC, SparseCSR, SparseCSC, SparseCOO and BlockCOO.
"""
SparseCOO(A::Matrix{Tv}) where {Tv} = _tocoo(A, Int)
SparseCOO(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _tocoo(A)
SparseCOO(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _tocoo(A)
SparseCOO(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = copy(A)
SparseCOO(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocoo(_tocsc(A))

"""
    Matrix(A)

Create a dense matrix Matrix{Tv} from the matrix `A`.
The matrix `A` is allowed to be SparseCSR, SparseCSC, SparseCOO and BlockCOO.
"""
Base.Matrix(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _todense(A)
Base.Matrix(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _todense(A)
Base.Matrix(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _todense(A)

SparseArrays.sparse(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = SparseArrays.sparse(_tocsc(_tocoo(A)))
SparseArrays.sparse(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = SparseArrays.SparseMatrixCSC{Tv,Ti}(A.m, A.n, copy(A.colptr), copy(A.rowind), copy(A.val))
SparseArrays.sparse(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = SparseArrays.sparse(_tocsc(A))

"""
    SparseCOO(m::Ti, n::Ti, elem::AbstractArray{Tuple{Ti,Ti,Tv},1}) where {Tv,Ti}

Create a m-by-n sparse matrix with COO format from the list of tuple (rowindex, colindex, value).
"""
function SparseCOO(m::Ti, n::Ti, elem::AbstractArray{Tuple{Ti,Ti,Tv},1}) where {Tv,Ti}
    rowind = Vector{Ti}()
    colind = Vector{Ti}()
    val = Vector{Tv}()
    prev_index::Tuple{Ti,Ti} = (0,0)
    for (i,j,u) in sort(elem)
        if m < i
            m = i
        end
        if n < j
            n = j
        end
        if prev_index != (i,j)
            push!(rowind, i)
            push!(colind, j)
            push!(val, u)
            prev_index = (i,j)
        else
            val[end] += u
        end
    end
    SparseCOO(m, n, val, rowind, colind)
end

### overload

function Base.show(io::IO, A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(m)*"x"*string(n)*" CSR-SparseMatrix")
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
        end
    end
end

function Base.show(io::IO, A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" CSC-SparseMatrix")
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
        end
    end
end

function Base.show(io::IO, A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" COO-SparseMatrix")
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
    end
end

function Base.show(io::IO, ::MIME"text/plain", A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(m)*"x"*string(n)*" CSR-SparseMatrix")
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" CSC-SparseMatrix")
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" COO-SparseMatrix")
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
    end
end

function SparseArrays.nnz(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    return length(A.val)
end

####

function Base.size(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    return (A.m, A.n)
end

# `size` reports the logical shape, so `length`/`eachindex`/`getindex` must
# agree with it: they are the AbstractMatrix contract that every generic
# fallback in Base and LinearAlgebra relies on. Use `nnz(A)` and `A.val` to
# reach the stored entries instead.

function Base.length(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    return m * n
end

function Base.getindex(A::AbstractSparseM{Tv,Ti}, i::Integer, j::Integer) where {Tv,Ti}
    @boundscheck checkbounds(A, i, j)
    z = _findz(A, i, j)
    z == 0 ? zero(Tv) : A.val[z]
end

function Base.getindex(A::AbstractSparseM{Tv,Ti}, k::Integer) where {Tv,Ti}
    m, n = size(A)
    @boundscheck checkbounds(A, k)
    i = mod1(k, m)
    j = div(k - i, m) + 1
    A[i, j]
end

# Writing is only possible where the sparsity pattern already stores an entry;
# creating a new one would have to rebuild the index arrays.
function Base.setindex!(A::AbstractSparseM{Tv,Ti}, value, i::Integer, j::Integer) where {Tv,Ti}
    @boundscheck checkbounds(A, i, j)
    z = _findz(A, i, j)
    z == 0 && throw(ArgumentError(
        "cannot write the element ($i,$j): it is not stored in the sparsity pattern"))
    A.val[z] = convert(Tv, value)
end

"""
    _findz(A, i, j)

Index into `A.val` of the stored entry at `(i,j)`, or 0 when the sparsity
pattern does not hold that position.
"""
function _findz(A::SparseCSR{Tv,Ti}, i::Integer, j::Integer) where {Tv,Ti}
    @inbounds for z = A.rowptr[i]:A.rowptr[i+1]-1
        A.colind[z] == j && return z
    end
    0
end

function _findz(A::SparseCSC{Tv,Ti}, i::Integer, j::Integer) where {Tv,Ti}
    @inbounds for z = A.colptr[j]:A.colptr[j+1]-1
        A.rowind[z] == i && return z
    end
    0
end

function _findz(A::SparseCOO{Tv,Ti}, i::Integer, j::Integer) where {Tv,Ti}
    @inbounds for z = 1:length(A.val)
        A.rowind[z] == i && A.colind[z] == j && return z
    end
    0
end

# function Base.iterate(A::AbstractSparseM{Tv,Ti}, i::Ti = 1) where {Tv,Ti}
#     i == length(A)+1 && return nothing
#     return (A.val[i], i+1)
# end

####

function Base.copy(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    SparseCSR(A.m, A.n, copy(A.val), A.rowptr, A.colind)
end

function Base.copy(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    SparseCSC(A.m, A.n, copy(A.val), A.colptr, A.rowind)
end

function Base.copy(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    SparseCOO(A.m, A.n, copy(A.val), A.rowind, A.colind)
end

####

function Base.similar(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    SparseCSR(A.m, A.n, similar(A.val), A.rowptr, A.colind)
end

function Base.similar(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    SparseCSC(A.m, A.n, similar(A.val), A.colptr, A.rowind)
end

function Base.similar(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    SparseCOO(A.m, A.n, similar(A.val), A.rowind, A.colind)
end

####

function Base.zero(::Type{AbstractMatrix{Tv}}) where Tv
    0
end

function Base.zero(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    SparseCSR(A.m, A.n, zero(A.val), A.rowptr, A.colind)
end

function Base.zero(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    SparseCSC(A.m, A.n, zero(A.val), A.colptr, A.rowind)
end

function Base.zero(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    SparseCOO(A.m, A.n, zero(A.val), A.rowind, A.colind)
end

function Base.zero(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    SparseArrays.SparseMatrixCSC{Tv,Ti}(A.m, A.n, A.colptr, A.rowval, zero(A.nzval))
end

###

"""
    _tocsr(A, Ti)

Internal function to convert a matrix to CSR (Compressed Sparse Row) format.

### Input
- `A`: Matrix to convert (Matrix or SparseCOO)
- `Ti`: Target integer type for indices

### Returns
- `SparseCSR{Tv,Ti}`: Matrix in CSR format
"""
function _tocsr(A::Matrix{Tv}, ::Type{Ti})::SparseCSR{Tv,Ti} where {Tv, Ti}
    # size returns Int, but the struct stores the dimensions as Ti
    m, n = Ti.(size(A))
    rowptr = Vector{Ti}(undef, m+1)
    colind = Vector{Ti}()
    val = Vector{Tv}()
    rowptr[1] = 1
    for i = 1:m
        for j = 1:n
            if !(iszero(A[i,j]))
                push!(colind, j)
                push!(val, A[i,j])
            end
        end
        rowptr[i+1] = length(val)+1
    end
    SparseCSR(m, n, val, rowptr, colind)
end

function _tocsr(A::SparseCOO{Tv,Ti})::SparseCSR{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    rowptr = Vector{Ti}(undef, m+1)
    colind = Vector{Ti}()
    val = Vector{Tv}()
    p = 1
    i = 1
    rowptr[i] = p
    for x in sort(collect(zip(A.rowind, A.colind, A.val)))
        if i != x[1]
            for u = i+1:x[1]
                rowptr[u] = p
            end
            i = x[1]
        end
        push!(colind, x[2])
        push!(val, x[3])
        p += 1
    end
    for u = i+1:m+1
        rowptr[u] = p
    end
    SparseCSR(m, n, val, rowptr, colind)    
end

"""
    _tocsc(A, Ti)

Internal function to convert a matrix to CSC (Compressed Sparse Column) format.

### Input
- `A`: Matrix to convert (Matrix, SparseCOO, or SparseArrays.SparseMatrixCSC)
- `Ti`: Target integer type for indices

### Returns
- `SparseCSC{Tv,Ti}`: Matrix in CSC format
"""
function _tocsc(A::Matrix{Tv}, ::Type{Ti})::SparseCSC{Tv,Ti} where {Tv, Ti}
    m, n = Ti.(size(A))
    colptr = Vector{Ti}(undef, n+1)
    rowind = Vector{Ti}()
    val = Vector{Tv}()
    colptr[1] = 1
    for j = 1:n
        for i = 1:m
            if !(iszero(A[i,j]))
                push!(rowind, i)
                push!(val, A[i,j])
            end
        end
        colptr[j+1] = length(val)+1
    end
    SparseCSC(m, n, val, colptr, rowind)
end

function _tocsc(A::SparseCOO{Tv,Ti})::SparseCSC{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    colptr = Vector{Ti}(undef, n+1)
    rowind = Vector{Ti}()
    val = Vector{Tv}()
    p = 1
    j = 1
    colptr[j] = p
    for x in sort(collect(zip(A.colind, A.rowind, A.val)))
        if j != x[1]
            for u = j+1:x[1]
                colptr[u] = p
            end
            j = x[1]
        end
        push!(rowind, x[2])
        push!(val, x[3])
        p += 1
    end
    for u = j+1:n+1
        colptr[u] = p
    end
    SparseCSC(m, n, val, colptr, rowind)
end

function _tocsc(A::SparseArrays.SparseMatrixCSC{Tv,Ti})::SparseCSC{Tv,Ti} where {Tv, Ti}
    SparseCSC(A.m, A.n, copy(A.nzval), copy(A.colptr), copy(A.rowval))
end

"""
    _tocoo(A, Ti)

Internal function to convert a matrix to COO (Coordinate) format.

### Input
- `A`: Matrix to convert (Matrix, SparseCSR, or SparseCSC)
- `Ti`: Target integer type for indices

### Returns
- `SparseCOO{Tv,Ti}`: Matrix in COO format
"""
function _tocoo(A::Matrix{Tv}, ::Type{Ti})::SparseCOO{Tv,Ti} where {Tv, Ti}
    m, n = Ti.(size(A))
    rowind = Vector{Ti}()
    colind = Vector{Ti}()
    val = Vector{Tv}()
    for j = 1:n
        for i = 1:m
            if !(iszero(A[i,j]))
                push!(rowind, i)
                push!(colind, j)
                push!(val, A[i,j])
            end
        end
    end
    SparseCOO(m, n, val, rowind, colind)
end

function _tocoo(A::SparseCSR{Tv,Ti})::SparseCOO{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    rowind = Vector{Ti}()
    colind = Vector{Ti}()
    val = Vector{Tv}()
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            push!(rowind, i)
            push!(colind, j)
            push!(val, A.val[z])
        end
    end
    SparseCOO(m, n, val, rowind, colind)
end

function _tocoo(A::SparseCSC{Tv,Ti})::SparseCOO{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    rowind = Vector{Ti}()
    colind = Vector{Ti}()
    val = Vector{Tv}()
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            push!(rowind, i)
            push!(colind, j)
            push!(val, A.val[z])
        end
    end
    SparseCOO(m, n, val, rowind, colind)
end

"""
    adddiag(A)

Return a matrix equal to `A` whose diagonal entries are all present in the
sparsity pattern, adding structural zeros where they are missing.

A dense `Matrix` is returned unchanged, and so is a sparse matrix that already
stores its whole diagonal — that case costs one `hasfulldiag` scan and the
length-n index vector `spdiag` builds, not a rebuild of the sparsity pattern.

This matters for uniformization: `unif` adds 1 to every diagonal entry through
`spdiag`, which can only write to entries the pattern actually stores. A CTMC
with an absorbing state has a zero diagonal entry that dense-to-sparse
conversion drops, so without this the uniformized matrix would not be
stochastic.
"""
adddiag(A::Matrix) = A

function adddiag(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    k = min(m, n)
    stored = falses(k)
    @inbounds for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        if i == A.colind[z] && i <= k
            stored[i] = true
        end
    end
    all(stored) && return A
    rowind = copy(A.rowind)
    colind = copy(A.colind)
    val = copy(A.val)
    @inbounds for i = 1:k
        if !stored[i]
            push!(rowind, Ti(i))
            push!(colind, Ti(i))
            push!(val, zero(Tv))
        end
    end
    SparseCOO(A.m, A.n, val, rowind, colind)
end

# Check first, convert only if something is missing: rebuilding the index arrays
# via COO costs O(nnz) allocations, while hasfulldiag is an O(n) scan.
function adddiag(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    hasfulldiag(A) && return A
    _tocsr(adddiag(_tocoo(A)))
end

function adddiag(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    hasfulldiag(A) && return A
    _tocsc(adddiag(_tocoo(A)))
end

function adddiag(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    hasfulldiag(A) && return A
    SparseArrays.sparse(_tocsc(adddiag(_tocoo(_tocsc(A)))))
end

"""
    _todense(A)

Internal function to convert a sparse matrix to dense Matrix format.

### Input
- `A`: Sparse matrix in CSR, CSC, or COO format

### Returns
- `Matrix{Tv}`: Dense matrix representation
"""
function _todense(A::SparseCSR{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, n = size(A)
    M = zeros(Tv, m, n)
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            M[i,j] = A.val[z]
        end
    end
    return M
end

function _todense(A::SparseCSC{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, n = size(A)
    M = zeros(Tv, m, n)
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            M[i,j] = A.val[z]
        end
    end
    return M
end

function _todense(A::SparseCOO{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, n = size(A)
    M = zeros(Tv, m, n)
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        M[i,j] = A.val[z]
    end
    return M
end
