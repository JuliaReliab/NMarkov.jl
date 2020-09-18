
"""
Sensitivity vector with iterative methods
"""

"""
stsenguess(Q::MatT, ::Type{Tv} = Float64)::Vector{Tv}

Get a vector which guesses the stationary vector of CTMC.
This is used for the iterative methods solving the stationary vector.
"""

function stsenguess(Q::MatT, ::Type{Tv} = Float64)::Vector{Tv} where {Tv,MatT}
    m, n = size(Q)
    @assert m == n
    fill(Tv(0), n)
end

"""
dtcmstcheck(Q::MatT, pis::Vector{Tv}; eps = Tv(1.0e-8))
ctmcstcheck(Q::MatT, pis::Vector{Tv}; eps = Tv(1.0e-8))

Check whether a given vector pis is a stationary vector of P ro Q
"""

function dtmcstcheck(P::MatT, pis::Vector{Tv}; eps = Tv(1.0e-8)) where {Tv,MatT}
    v = P' * pis - pis
    maximum(abs.(v)) < eps
end

function ctmcstcheck(Q::MatT, pis::Vector{Tv}; eps = Tv(1.0e-8)) where {Tv,MatT}
    v = Q' * pis
    maximum(abs.(v)) < eps
end

"""
stsengs(Q::SparseCSC{Tv,Ti}, pis::Vector{Tv}, b::Vector{Tv}:
  x0::Vector{Tv}=stsenguess(Q,Tv), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6))

Get a sensitivity vector for stationary vector of CTMC.

Parameters:
- Q: CTMC Kernal
- pis: A stationary vector, i.e., pis * Q = 0
- b: A vector. In the case of first derivative, b = pis * Qdash where Qdash is the first derivate of Q. It may change when we want to the high-order derivatives.
- x0: Initial vector for iteration
- maxiter: The maximum number of iteration. The algorithm stops when the number of iteration becomes maxiter.
- steps: The number of steps to check the convergence
- rtol: the tolerance error. When the relative errors of two successive vectors with steps attains rtol, the algorithm stops.
Return value:
A tuple of
- x: sensitivity vector
- conv: A boolean whether the algorithm converges or not
- iter: The number of iterations
- rerror: The relative error when the algorithm stops
"""

function stsengs(Q::SparseMatrixCSC{Tv,Ti}, pis::Vector{Tv}, b::Vector{Tv};
    x0::Vector{Tv}=stsenguess(Q), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    stsengs(SparseCSC(Q), pis, b, x0=x0, maxiter=maxiter, steps=steps, rtol=rtol)
end

function stsengs(Q::SparseCSC{Tv,Ti}, pis::Vector{Tv}, b::Vector{Tv};
    x0::Vector{Tv}=stsenguess(Q), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    m, n = size(Q)
    @assert m == n
    @assert ctmcstcheck(Q, pis)
    x = copy(x0)
    iter = 0
    conv = false
    rerror::Tv = Tv(0)
    prevx = similar(x)
    while true
        prevx .= x
        for i in 1:steps
            gsstep!(x, Q, b, alpha=-Tv(1))
            @axpy(-sum(x), pis, x)
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
    return x, conv, iter, rerror
end

"""
stsengs(Q::SparseCSC{Tv,Ti}, pis::Vector{Tv}, b::Vector{Tv}:
  x0::Vector{Tv}=stsenguess(Q,Tv), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6))

Get a sensitivity vector for stationary vector of CTMC.
For instance, the first derivative of stationary vector with power method

  s_n = (s_{n-1} P + pis * Pdash)(I - 1 pis)
    
Parameters:
- P: DTMC Kernal
- pis: A stationary vector, i.e., pis * P = pis
- b: A vector. In the case of first derivative, b = pis * Pdash where Pdash is the first derivate of P. It may change when we want to the high-order derivatives.
- x0: Initial vector for iteration
- maxiter: The maximum number of iteration. The algorithm stops when the number of iteration becomes maxiter.
- steps: The number of steps to check the convergence
- rtol: the tolerance error. When the relative errors of two successive vectors with steps attains rtol, the algorithm stops.
Return value:
A tuple of
- x: sensitivity vector
- conv: A boolean whether the algorithm converges or not
- iter: The number of iterations
- rerror: The relative error when the algorithm stops
"""

function stsenpower(P::AbstractMatrix{Tv}, pis::Vector{Tv}, b::Vector{Tv};
    x0::Vector{Tv}=stsenguess(P,Tv), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv}
    m, n = size(P)
    @assert m == n
    @assert dtmcstcheck(P, pis)
    Pdash = P'
    x = copy(x0)
    iter = 0
    conv = false
    rerror::Tv = 0
    prevx = similar(x)
    while true
        prevx .= x
        for i in 1:steps
            x = Pdash * x + b
            @axpy(-sum(x), pis, x)
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
    return x, conv, iter, rerror
end


