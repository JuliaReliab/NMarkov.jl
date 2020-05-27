# function invC(x::Vector{Float64}, Q::AbstractSparseMatrix{T})
#     v = x
#     for j in 1:Q.n
#         tmpx = 0.0
#         for z in Q.colptr[j]:(Q.colptr[j+1]-1)
#             i = Q.rowval[z]
#             if i == j
#                 tmpx += v[j]
#                 v[j] = tmpx / (-Q.nzval[z])
#                 break
#             else
#                 tmpx -= (-Q.nzval[z]) * v[i]
#             end
#         end
#     end
#     v
# end

# function ex(x::Vector{Float64}, Q::SparseMatrixCSC;
#     maxiter=5000, tol=1.0e-16)::Vector{Float64}
#     b = zeros(Q.n)
#     v = x
#     y = invC(v, Q)
#     while maximum(v) > tol
#         gsstep!(v, Q, b)
#         y += v
#     end
#     y
# end

# function ex2(Q::SparseMatrixCSC, b::Vector{Float64};
#     x0::Vector{Float64}=fill(1/Q.n, Q.n),
#     maxiter=5000, steps=50, rtol=1.0e-6)::Vector{Float64}
#     x = x0
#     iter = 0
#     conv = false
#     rerror = 0.0
#     while true
#         prevx = x
#         for i in 1:steps
#             gsstep!(x, Q, b)
#         end
#         rerror = maximum(abs.((x - prevx) ./ x))
#         iter += steps
#         if rerror < rtol
#             conv = true
#             break
#         end
#         if iter >= maxiter
#             break
#         end
#     end
#     @printf "convergence   : %s\n" conv ? "true" : "false"
#     @printf "iteration     : %d / %d\n" iter maxiter
#     @printf "relative error: %e < %e\n" rerror rtol
#     x
# end

# function test2(Q, x)
#     C = triu(Q, 0)
#     D = tril(Q, -1)
#     y = zeros(size(x))
#     x = (-C)' \ x
#     y += x
#     for i in 1:100
#         x = D' * x
#         x = (-C)' \ x
#         y += x
#     end
#     y
# end

# function test(Q::SparseMatrixCSC;
#     x0::Vector{Float64}=fill(1/Q.n, Q.n),
#     maxiter=5000, steps=50, rtol=1.0e-6)::Vector{Float64}
#     x = x0
#     iter = 0
#     conv = false
#     rerror = 0.0
#     while true
#         prevx = x
#         for i in 1:steps
#             b = x
#             gsstep!(x, Q, b)
#             x /= sum(x)
#         end
#         rerror = maximum(abs.((x - prevx) ./ x))
#         iter += steps
#         if rerror < rtol
#             conv = true
#             break
#         end
#         if iter >= maxiter
#             break
#         end
#     end
#     @printf "convergence   : %s\n" conv ? "true" : "false"
#     @printf "iteration     : %d / %d\n" iter maxiter
#     @printf "relative error: %e < %e\n" rerror rtol
#     x
# end
