
"""
@daxpy
@dascal

BLAS Level 1 functions.
"""

macro daxpy(a, x, y)
    expr = quote
        let u = $a
            for i in 1:length($x)
                $y[i] += u * $x[i]
            end
        end
    end
    esc(expr)
end

macro dscal(a, x)
    expr = quote
        let u = $a
            for i in 1:length($x)
                $x[i] *= u
            end
        end
    end
    esc(expr)
end

# function spsum(A::SparseMatrixCSC{T}, axis::Int)::Vector{T} where {T <: AbstractFloat}
#     if axis == 0
#         return _spcolsum(A)
#     elseif axis == 1
#         return _sprowsum(A)
#     else
#         return sum(A.nzval)
#     end
# end

# function _sprowsum(A::SparseMatrixCSC{T})::Vector{T} where {T <: AbstractFloat}
#     x = zeros(A.m)
#     for j = 1:A.n
#         for z = A.colptr[j]:(A.colptr[j+1]-1)
#             x[A.rowval[z]] += A.nzval[z]
#         end
#     end
#     x
# end

# function _spcolsum(A::SparseMatrixCSC{T})::Vector{T} where {T <: AbstractFloat}
#     x = zeros(A.n)
#     for j = 1:A.n
#         for z = A.colptr[j]:(A.colptr[j+1]-1)
#             x[j] += A.nzval[z]
#         end
#     end
#     x
# end


