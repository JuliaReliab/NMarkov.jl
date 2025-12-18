"""
    SparseELL1{Tv,Ti}

Type that represents an ELL (Ellpack) format of sparse matrix from CSR (row-major) format.

### Fields
- `m::Ti`: the number of rows
- `n::Ti`: the number of columns
- `k::Int`: the maximum number of non-zero elements in any row
- `idx::Matrix{Ti}`: matrix of column indices (0 indicates padding)
- `val::Matrix{Tv}`: matrix of non-zero elements

### Notes
ELL format is efficient for SpMV operations when the number of non-zeros per row is regular.
"""
struct SparseELL1{Tv,Ti}
    m::Ti
    n::Ti
    k::Int
    idx::Array{Ti,2}
    val::Array{Tv,2}
end

"""
    SparseELL2{Tv,Ti}

Type that represents an ELL (Ellpack) format of sparse matrix from CSC (column-major) format.

### Fields
- `m::Ti`: the number of rows
- `n::Ti`: the number of columns
- `k::Int`: the maximum number of non-zero elements in any column
- `idx::Matrix{Ti}`: matrix of row indices (0 indicates padding)
- `val::Matrix{Tv}`: matrix of non-zero elements

### Notes
ELL format is efficient for SpMV operations when the number of non-zeros per column is regular.
"""
struct SparseELL2{Tv,Ti}
    m::Ti
    n::Ti
    k::Int
    idx::Array{Ti,2}
    val::Array{Tv,2}
end

"""
    SparseELL1(A)

Create a sparse matrix with ELL format from row-based structure.

### Input
- `A`: A matrix in various formats (Matrix, SparseCSR, SparseCSC, SparseCOO, or SparseMatrixCSC)

### Returns
- `SparseELL1{Tv,Ti}`: ELL format sparse matrix

### Example
```julia
A = SparseCSR(...)
ell = SparseELL1(A)  # Convert to row-based ELL format
```
"""
function SparseELL1(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m = A.m
    n = A.n
    k = maximum(diff(A.rowptr))
    elem = zeros(Tv, m, k)
    idx = zeros(Ti, m, k)
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            v = z - A.rowptr[i] + 1
            j = A.colind[z]
            idx[i,v] = j
            elem[i,v] = A.val[z]
        end
    end
    SparseELL1{Tv,Ti}(m, n, k, idx, elem)
end

"""
    SparseELL2(A)

Create a sparse matrix with ELL format from column-based structure.

### Input
- `A`: A matrix in various formats (Matrix, SparseCSR, SparseCSC, SparseCOO, or SparseMatrixCSC)

### Returns
- `SparseELL2{Tv,Ti}`: ELL format sparse matrix (column-based)

### Example
```julia
A = SparseCSC(...)
ell = SparseELL2(A)  # Convert to column-based ELL format
```
"""
function SparseELL2(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    m = A.m
    n = A.n
    k = maximum(diff(A.colptr))
    elem = zeros(Tv, n, k)
    idx = zeros(Ti, n, k)
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            v = z - A.colptr[j] + 1
            i = A.rowind[z]
            idx[j,v] = i
            elem[j,v] = A.val[z]
        end
    end
    SparseELL2{Tv,Ti}(m, n, k, idx, elem)
end

SparseELL1(A::Matrix{Tv}) where Tv = SparseELL1(SparseCSR(A))
SparseELL1(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = SparseELL1(SparseCSR(A))
SparseELL1(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = SparseELL1(SparseCSR(A))
SparseELL1(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = SparseELL1(SparseCSR(A))

SparseELL2(A::Matrix{Tv}) where Tv = SparseELL2(SparseCSC(A))
SparseELL2(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = SparseELL2(SparseCSC(A))
SparseELL2(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = SparseELL2(SparseCSC(A))
SparseELL2(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = SparseELL2(SparseCSC(A))
