

"""
    @dot(x, y)

Compute the dot product of vectors x and y.

### Example
```julia
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
result = @dot(x, y)  # Returns 32.0
```
"""
macro dot(x, y)
    expr = quote
        s = 0
        for i in eachindex($(esc(x)))
            @inbounds s += $(esc(x))[i] * $(esc(y))[i]
        end
        s
    end
    expr
end

"""
    trans(transpose::Symbol)

Convert transpose symbol to character for BLAS operations.

### Arguments
- `transpose::Symbol`: `:N` for non-transpose, `:T` for transpose

### Returns
- Character ('N' or 'T') for use in BLAS functions

### Example
```julia
t = trans(:T)  # Returns 'T'
```
"""
function trans(transpose::Symbol)
    transpose == :N && return 'N'
    transpose == :T && return 'T'
    throw(ArgumentError("transpose must be :N or :T, got :$transpose"))
end

"""
    matmul!(transpose, alpha, A, B, beta, C)

Wrapper function for matrix-vector or matrix-matrix multiplication.

Performs: `C := alpha * A^transpose * B + beta * C`

### Arguments
- `transpose::Symbol`: `:N` for non-transpose, `:T` for transpose of A
- `alpha`: Scalar multiplier
- `A`: Matrix
- `B`: Vector or Matrix
- `beta`: Scalar multiplier for C
- `C`: Output vector or matrix (modified in-place)

### Notes
This function dispatches to either `gemv!` or `gemm!` depending on B type.
"""
function matmul!(transpose::Symbol, alpha::Union{Tv,Bool}, A::AbstractMatrix{Tv}, B::AbstractMatrix{Tv}, beta::Union{Tv,Bool}, C::AbstractMatrix{Tv}) where Tv
    gemm!(trans(transpose), 'N', alpha, A, B, beta, C)
end

function matmul!(transpose::Symbol, alpha::Union{Tv,Bool}, A::AbstractMatrix{Tv}, B::AbstractVector{Tv}, beta::Union{Tv,Bool}, C::AbstractVector{Tv}) where Tv
    gemv!(trans(transpose), alpha, A, B, beta, C)
end

"""
    asarray(Tv, x)
    asvector(Tv, ts)

Convert an argument to element type `Tv`, returning it untouched when it already
has that element type.

`asarray` keeps the shape of `x`, so a matrix argument stays a matrix; the
matrix methods of `tran`/`mexp` are only reachable that way.
"""
asarray(::Type{Tv}, x::AbstractArray{Tv}) where {Tv} = x
asarray(::Type{Tv}, x::AbstractArray) where {Tv} = convert(Array{Tv}, x)

asvector(::Type{Tv}, ts::AbstractVector{Tv}) where {Tv} = ts
asvector(::Type{Tv}, ts::AbstractVector) where {Tv} = convert(Vector{Tv}, ts)

"""
    checktime(t)
    checktimes(ts)

Validate the time argument of a transient computation.

A single time must be non-negative; a time series must in addition be sorted in
ascending order. Uniformization walks the series interval by interval, so a
descending step would ask for a Poisson p.m.f. with a negative mean, which
writes outside the p.m.f. buffer and yields `NaN`.
"""
function checktime(t::Real)
    t >= 0 || throw(ArgumentError("the time must be non-negative, got $t"))
    t
end

function checktimes(ts::AbstractVector)
    isempty(ts) && return ts
    issorted(ts) || throw(ArgumentError(
        "the time points must be sorted in ascending order, got $ts"))
    checktime(first(ts))
    ts
end

"""
    itime(t)

Compute interval times from a cumulative time vector.

### Arguments
- `t::AbstractVector`: Cumulative time points where t[1] is the first time

### Returns
- `dt::Vector`: Interval times (differences between consecutive points)
- `maxt::Number`: Maximum interval time

### Example
```julia
t = [0.0, 1.0, 2.5, 4.0]
dt, maxt = itime(t)  # dt = [0, 1.0, 1.5, 1.5], maxt = 1.5
```
"""
function itime(t::AbstractVector{Tv}) where Tv
    dt = similar(t)
    prev = Tv(0)
    maxt = Tv(0)
    @inbounds for i = eachindex(t)
        dt[i] = t[i] - prev
        prev = t[i]
        if dt[i] > maxt
            maxt = dt[i]
        end
    end
    return dt, maxt
end

"""
    eye(n, ::Type{Tv} = Float64)
    eye(A::AbstractMatrix, ::Type{Tv} = Float64)

Create an identity matrix.

### Arguments
- `n::Int`: Size of the identity matrix
- `A::AbstractMatrix`: Matrix whose size is used to determine identity matrix size
- `Tv::Type`: Element type (default: Float64)

### Returns
- Identity matrix of specified size and type
"""
function eye(n, ::Type{Tv} = Float64)::Matrix{Tv} where {Tv}
    m = zeros(Tv, n,n)
    @inbounds for i = 1:n
        m[i,i] = Tv(1)
    end
    m
end

function eye(A::AbstractMatrix, ::Type{Tv} = Float64)::Matrix{Tv} where {Tv}
    eye(size(A, 1), Tv)
end

"""
    unif(Q, ufact = 1.01)

Get an uniformized transition probability matrix from a CTMC kernel.

Computes: `P = I + Q / qv` where `qv = max(abs(diag(Q))) * ufact`

### Arguments
- `Q`: CTMC kernel matrix (sparse or dense)
- `ufact`: Uniformization factor (default: 1.01)

### Returns
- `P`: Uniformized transition probability matrix
- `qv`: The uniformization rate (maximum event rate)

### Supported Matrix Types
- SparseMatrixCSC
- SparseCSR
- SparseCSC
- SparseCOO
- Matrix (dense)

### Notes
A sparse `Q` whose diagonal is not fully stored (a CTMC with an absorbing state
has a zero diagonal entry, which dense-to-sparse conversion drops) is passed
through `adddiag` first, so `P` is stochastic in every case.

If `Q` is the zero matrix there is no event to uniformize against; `qv` is then
1 and `P` is the identity, which is the correct kernel for a chain that never
moves.
"""
const UnifMatrix{Tv} = Union{Matrix{Tv},
    SparseMatrixCSC{Tv},
    SparseCSR{Tv},
    SparseCSC{Tv},
    SparseCOO{Tv}}

function unif(Q::UnifMatrix{Tv}, ufact::Real = 1.01) where {Tv}
    # One spdiag serves both purposes: the largest |diagonal| and the test for a
    # complete pattern. Reading an absent diagonal entry already yields zero, so
    # adddiag cannot change the maximum.
    dq = spdiag(Q)
    qv = maximum(abs.(dq)) * convert(Tv, ufact)
    if iszero(qv)
        qv = one(Tv)
    end
    A = hasfulldiag(dq) ? Q : adddiag(Q)
    P = _scaledcopy(A, qv)
    d = spdiag(P)
    d .+= one(Tv)
    (P, qv)
end

# `A / qv` is right for a dense matrix and for the SparseMatrix types, whose `/`
# is already a structure-preserving `scal!` on a copy. SparseMatrixCSC is the
# exception: it falls back to the generic sparse `/`, which prunes the structural
# zeros on the diagonal, so scale a copy in place instead.
_scaledcopy(A::Matrix{Tv}, qv::Tv) where {Tv} = A / qv
_scaledcopy(A::AbstractSparseM{Tv,Ti}, qv::Tv) where {Tv,Ti} = A / qv

function _scaledcopy(A::SparseMatrixCSC{Tv,Ti}, qv::Tv) where {Tv,Ti}
    P = copy(A)
    scal!(one(Tv) / qv, P)
    P
end
