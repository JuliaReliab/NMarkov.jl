
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
