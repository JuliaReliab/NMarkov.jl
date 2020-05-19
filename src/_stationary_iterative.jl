
"""
Stationary vector with iterative methods
"""

export stguess, stgs, stpower

"""
stguess(Q::MatT, ::Type{Tv} = Float64)::Vector{Tv}

Get a vector which is guessed as the sensitivity function for the stationary vector of CTMC.
This is used as the initial vector for the iterative methods.
"""

function stguess(Q::MatT, ::Type{Tv} = Float64)::Vector{Tv} where {Tv,MatT}
    m, n = size(Q)
    @assert m == n
    result = Vector{Tv}(undef, n)
    for (i,x) in enumerate(spdiag(Q))
        result[i] = 1/x
    end
    result ./= sum(result)
end

"""
stgs(Q::SparseCSC{Tv,Ti}, x0::Vector{Tv}=stguess(Q,Tv); maxiter=5000, steps=20, reltol::Tv=Tv(1.0e-6))

Get a stationary vector of CTMC.

Parameters:
- Q: CTMC Kernal
- x0: Initial vector for iteration
- maxiter: The maximum number of iteration. The algorithm stops when the number of iteration becomes maxiter.
- steps: The number of steps to check the convergence
- reltol: the tolerance error. When the relative errors of two successive vectors with steps attains reltol, the algorithm stops.
Return value:
A tuple of
- x: stationary vector
- conv: A boolean whether the algorithm converges or not
- iter: The number of iterations
- rerror: The relative error when the algorithm stops
"""

function stgs(Q::SparseCSC{Tv,Ti}; x0::Vector{Tv}=stguess(Q,Tv),
        maxiter=5000, steps=20, reltol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    m, n = size(Q)
    @assert m == n
    b = zeros(Tv, n)
    x = copy(x0)
    iter = 0
    conv = false
    rerror::Tv = 0
    prevx = similar(x)
    while true
        prevx .= x
        for i in 1:steps
            gsstep!(x, Q, b)
            x ./= sum(x)
        end
        rerror = maximum(abs.((x - prevx) ./ x))
        iter += steps
        if rerror < reltol
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
stpower(P::AbstractMatrix{Tv}, x0::Vector{Tv}=stguess(Q,Tv); maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6))

Get a stationary vector of DTMC with power method.

Parameters:
- P: The transition probability matrix for DTMC
- x0: Initial vector for iteration
- maxiter: The maximum number of iteration. The algorithm stops when the number of iteration becomes maxiter.
- steps: The number of steps to check the convergence
- reltol: the tolerance error. When the relative errors of two successive vectors with steps attains reltol, the algorithm stops.
Return value:
A tuple of
- x: stationary vector
- conv: A boolean whether the algorithm converges or not
- iter: The number of iterations
- rerror: The relative error when the algorithm stops
"""

function stpower(P::AbstractMatrix{Tv}; x0::Vector{Tv}=stguess(P,Tv),
    maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
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
    return x, conv, iter, rerror
end
