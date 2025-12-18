

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
    nothing
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
    eye(size(A)[1])
end

"""
    Uniformed matrix for CTMC

Internal macro for uniformization computation.
"""
macro unif(Q, ufact)
    expr = quote
        qv = maximum(abs.(spdiag($Q))) * $ufact
        if iszero(qv)
            qv = 1.0e-12
        end
        P = $Q / qv
        d = spdiag(P)
        d .+= 1
        (P, qv)
    end
    esc(expr)
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
"""
function unif(Q::SparseMatrixCSC{Tv,Ti}, ufact::Tv = 1.01) where {Tv, Ti}
    @unif(Q, ufact)
end

function unif(Q::SparseCSR{Tv,Ti}, ufact::Tv = 1.01) where {Tv, Ti}
    @unif(Q, ufact)
end

function unif(Q::SparseCSC{Tv,Ti}, ufact::Tv = 1.01) where {Tv, Ti}
    @unif(Q, ufact)
end

function unif(Q::SparseCOO{Tv,Ti}, ufact::Tv = 1.01) where {Tv, Ti}
    @unif(Q, ufact)
end

function unif(Q::Matrix{Tv}, ufact::Tv = 1.01) where {Tv}
    @unif(Q, ufact)
end
