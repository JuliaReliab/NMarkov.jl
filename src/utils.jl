

"""
@axpy
@ascal
@dot

BLAS Level 1 functions.
"""

# macro axpy(a, x, y)
#     expr = quote
#         let u = $(esc(a))
#             for i in eachindex($(esc(x)))
#                 @inbounds $(esc(y))[i] += u * $(esc(x))[i]
#             end
#         end
#     end
#     expr
# end

# macro scal(a, x)
#     expr = quote
#         let u = $(esc(a))
#             for i in eachindex($(esc(x)))
#                 @inbounds $(esc(x))[i] *= u
#             end
#         end
#     end
#     expr
# end

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

function trans(transpose::Symbol)
    transpose == :N && return 'N'
    transpose == :T && return 'T'
    nothing
end

function matmul!(transpose::Symbol, alpha::Union{Tv,Bool}, A::AbstractMatrix{Tv}, B::AbstractMatrix{Tv}, beta::Union{Tv,Bool}, C::AbstractMatrix{Tv}) where Tv
    gemm!(trans(transpose), 'N', alpha, A, B, beta, C)
end

function matmul!(transpose::Symbol, alpha::Union{Tv,Bool}, A::AbstractMatrix{Tv}, B::AbstractVector{Tv}, beta::Union{Tv,Bool}, C::AbstractVector{Tv}) where Tv
    gemv!(trans(transpose), alpha, A, B, beta, C)
end

"""
itime(t)

Get interval time from a given cumulative time vector t.
The first element is t[1]

Retuen value:
dt: interval time vector
maxt: maximum interval time
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
eye(n, ::Type{Tv} = Float64)::Matrix{Tv}
eye(A::AbstractMatrix, ::Type{Tv} = Float64)::Matrix{Tv}

Make an indentity matrix
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
Uniformed Matrix for CTMC
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
unif(Q::AbstractSparseM{Tv,Ti}, ufact::Tv = 1.01)
unif(Q::Matrix{Tv}, ufact::Tv = 1.01)

Get an uniformed transition probability matrix from a CTMC kernel.

   P = I + Q / qv
   qv = max(abs(diag(Q))) * ufact

Parameters:
- Q: CTMC Kernel
- ufact: uniformization factor
Return value:
A tuple of
- P: The uniformed transition probability matrix
- qv: The maximum event rate

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
