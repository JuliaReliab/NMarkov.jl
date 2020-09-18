"""
Quasi-stationary vector with iterative methods
"""

"""
qstgs(Q::SparseCSC{Tv,Ti}, xi::Vector{Tv};
      x0::Vector{Tv}=stguess(Q,Tv), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6))

Get a quasi-stationary vector of CTMC.

Parameters:
- Q: CTMC Kernal
- xi: Exit vector
- x0: Initial vector for iteration
- maxiter: The maximum number of iteration. The algorithm stops when the number of iteration becomes maxiter.
- steps: The number of steps to check the convergence
- rtol: the tolerance error. When the relative errors of two successive vectors with steps attains rtol, the algorithm stops.
Return value:
A tuple of
- x: quasi-stationary vector
- gam: minimum eigen value
- conv: A boolean whether the algorithm converges or not
- iter: The number of iterations
- rerror: The relative error when the algorithm stops
"""

function qstgs(Q::SparseMatrixCSC{Tv,Ti}, xi::Vector{Tv}; x0::Vector{Tv}=stguess(Q,Tv),
        maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    qstgs(SparseCSC(Q), xi, x0=x0, maxiter=maxiter, steps=steps, rtol=rtol)
end

function qstgs(Q::SparseCSC{Tv,Ti}, xi::Vector{Tv}; x0::Vector{Tv}=stguess(Q,Tv),
        maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    m, n = size(Q)
    @assert m == n
    b = zeros(Tv, n)
    x = copy(x0)
    iter = 0
    conv = false
    rerror::Tv = 0
    gam::Tv = 0
    prevx = similar(x)
    while true
        prevx .= x
        for i in 1:steps
            gam = @dot(x, xi)
            gsstep!(x, Q, b, sigma=-gam)
            x ./= sum(x)
        end
        rerror = maximum(abs.((x - prevx) ./ x))
        iter += steps
        if rerror < rtol
            conv = true
            break
        end
        if iter >= maxiter
            break
        end
    end
    return x, gam, conv, iter, rerror
end

"""
qstpower(P::AbstractMatrix{Tv}; x0::Vector{Tv}=stguess(Q,Tv), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6))

Get a quasi-stationary vector of DTMC with power method.

Parameters:
- P: The transition probability matrix for DTMC
- xi: Exit vector
- x0: Initial vector for iteration
- maxiter: The maximum number of iteration. The algorithm stops when the number of iteration becomes maxiter.
- steps: The number of steps to check the convergence
- rtol: the tolerance error. When the relative errors of two successive vectors with steps attains rtol, the algorithm stops.
Return value:
A tuple of
- x: quasi-stationary vector
- gam: maximimum eigen value
- conv: A boolean whether the algorithm converges or not
- iter: The number of iterations
- rerror: The relative error when the algorithm stops
"""

function qstpower(P::AbstractMatrix{Tv}, xi::Vector{Tv};
    x0::Vector{Tv}=stguess(P,Tv), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv}
    m, n = size(P)
    @assert m == n
    Pdash = P'
    x = copy(x0)
    iter = 0
    conv = false
    rerror::Tv = 0
    prevx = similar(x)
    while true
        prevx .= x
        for i in 1:steps
            x = Pdash * x
            x ./= sum(x)
        end
        rerror = maximum(abs.((x - prevx) ./ x))
        iter += steps
        if rerror < rtol
            conv = true
            break
        end
        if iter >= maxiter
            break
        end
    end
    nu = @dot(x, xi)
    return x, nu, conv, iter, rerror
end

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

