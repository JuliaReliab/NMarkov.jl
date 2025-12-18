
"""
Stationary vector with iterative methods
"""

"""
GTH algorithm
"""

"""
gth!(Q::Matrix{Tv})
gth(Q::Matrix{Tv})

Compute the stationary vector for the CTMC with kernel Q with GTH algorithm.
Both of gth! and gth return a vector.
In gth!, the matrix Q is used as a workspace.
Note that Q does not have any absorbing state.
"""

function gth!(Q::Matrix{Tv})::Vector{Tv} where {Tv}
    @inbounds begin
        m, n = size(Q)
        @assert m == n
        for l = n:-1:2
            tmp::Tv = 0
            for u = 1:l-1
                tmp += Q[l,u]
            end
            for j = 1:l-1
                for i = 1:l-1
                    if i != j
                        Q[i,j] += Q[l,j] * Q[i,l] / tmp
                    end
                end
            end
            for i = 1:l-1
                Q[i,l] /= tmp
            end
            for i = 1:l-1
                Q[l,i] = 0
            end
            Q[l,l] = -1
        end
        x = Vector{Tv}(undef, n)
        x[1] = 1.0
        for l = 2:n
            x[l] = 0.0
            for i = 1:l-1
                x[l] += x[i] * Q[i,l]
            end
        end
        x /= sum(x)
    end
end

function gth(Q::Matrix{Tv}) where {Tv}
    gth!(copy(Q))
end

"""
gth!(Q::Matrix{Tv}, index::Vector{Ti})
gth(Q::Matrix{Tv}, index::Vector{Ti})

Compute the stationary vector for the CTMC with kernel Q with GTH algorithm.
Both of gth! and gth return a vector.
The CTMC states are permutated with an index vector.
In gth!, the matrix Q is used as a workspace.
Note that Q does not have any absorbing state.
"""

function gth!(Q::Matrix{Tv}, index::Vector{Ti})::Vector{Tv} where {Tv,Ti}
    m, n = size(Q)
    @assert m == n
    for l = n:-1:2
        tmp::Tv = 0
        for u = 1:l-1
            tmp += Q[index[l],index[u]]
        end
        for j = 1:l-1
            for i = 1:l-1
                if i != j
                    Q[index[i],index[j]] += Q[index[l],index[j]] * Q[index[i],index[l]] / tmp
                end
            end
        end
        for i = 1:l-1
            Q[index[i],index[l]] /= tmp
        end
        for i = 1:l-1
            Q[index[l],index[i]] = 0
        end
        Q[index[l],index[l]] = -1
    end
    x = Vector{Tv}(undef, n)
    x[index[1]] = 1.0
    for l = 2:n
        x[index[l]] = 0.0
        for i = 1:l-1
            x[index[l]] += x[index[i]] * Q[index[i],index[l]]
        end
    end
    x /= sum(x)
end

function gth(Q::Matrix{Tv}, index::Vector{Ti}) where {Tv, Ti}
    gth!(copy(Q), index)
end

"""
stguess(Q::MatT, ::Type{Tv} = Float64)::Vector{Tv}

Get a vector which is guessed as the stationary vector of CTMC.
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
stgs(Q::SparseCSC{Tv,Ti}, x0::Vector{Tv}=stguess(Q,Tv); maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6))

Get a stationary vector of CTMC.

Parameters:
- Q: CTMC Kernal
- x0: Initial vector for iteration
- maxiter: The maximum number of iteration. The algorithm stops when the number of iteration becomes maxiter.
- steps: The number of steps to check the convergence
- rtol: the tolerance error. When the relative errors of two successive vectors with steps attains rtol, the algorithm stops.
Return value:
A tuple of
- x: stationary vector
- conv: A boolean whether the algorithm converges or not
- iter: The number of iterations
- rerror: The relative error when the algorithm stops
"""

function stgs(Q::SparseMatrixCSC{Tv,Ti}; x0::Vector{Tv}=stguess(Q,Tv),
        maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    stgs(SparseCSC(Q), x0=x0, maxiter=maxiter, steps=steps, rtol=rtol)
end

function stgs(Q::SparseCSC{Tv,Ti}; x0::Vector{Tv}=stguess(Q,Tv),
        maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
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
        # rerror = maximum(abs.((x - prevx) ./ x))
        rerror = maximum(abs.(x - prevx)) / maximum(x)
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
stpower(P::AbstractMatrix{Tv}, x0::Vector{Tv}=stguess(Q,Tv); maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6))

Get a stationary vector of DTMC with power method.

Parameters:
- P: The transition probability matrix for DTMC
- x0: Initial vector for iteration
- maxiter: The maximum number of iteration. The algorithm stops when the number of iteration becomes maxiter.
- steps: The number of steps to check the convergence
- rtol: the tolerance error. When the relative errors of two successive vectors with steps attains rtol, the algorithm stops.
Return value:
A tuple of
- x: stationary vector
- conv: A boolean whether the algorithm converges or not
- iter: The number of iterations
- rerror: The relative error when the algorithm stops
"""

function stpower(P::AbstractMatrix{Tv}; x0::Vector{Tv}=stguess(P,Tv),
    maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv}
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
        # rerror = maximum(abs.((x - prevx) ./ x))
        rerror = maximum(abs.(x - prevx)) / maximum(x)
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
gsstep!(x::Vector{Tv}, Q::SparseMatrix.SparseCSC{Tv,Ti}, b::Vector{Tv}; alpha::Tv=Tv(1), sigma::Tv=Tv(0), omega::Tv=Tv(1))::Nothing
gsstep!(x::Vector{Tv}, Q::SparseMatrix.SparseCSR{Tv,Ti}, b::Vector{Tv}; alpha::Tv=Tv(1), sigma::Tv=Tv(0), omega::Tv=Tv(1))::Nothing

GS (Gauss-Seidal) or SOR (Successive Over Relaxation) step for the following linear equation

    alpha * trans(A - sigma I) * x = b

    notrans:
        x := (D/omega + L)^(-1) (b/alpha - (U - D (1-omega)/omega - sigma I) * x)
    trans:
        x := (D/omega + tr(U))^(-1) (b/alpha - (tr(L) - D (1-omega)/omega - sigma I) * x)
        
    where
        A: square matrix
        x: vector (in; initial vector for the step, out; updated vector)
        b: constant vector

Note that notrans and trans are determined by the type of matrix Q.
If Q is SparseCSC, gstep! provides the step for trans.
If Q is SparseCSR, gsstep! provides the step for notrans.
"""

function gsstep!(x::Vector{Tv}, Q::SparseCSC{Tv,Ti}, b::Vector{Tv};
        alpha::Tv=Tv(1), sigma::Tv=Tv(0), omega::Tv=Tv(1))::Nothing where {Tv, Ti}
    m, n = size(Q)
    @assert m == n
    @inbounds for j = 1:n
        tmpd::Tv = 0
        tmpx::Tv = b[j] / alpha
        for z = Q.colptr[j]:Q.colptr[j+1]-1
            i = Q.rowind[z]
            if i == j
                tmpd = Q.val[z]
                tmpx += sigma * x[i]
            else
                tmpx -= Q.val[z] * x[i]
            end
        end
        x[j] = omega / tmpd * tmpx + (1 - omega) * x[j]
    end
    nothing
end

function gsstep!(x::Vector{Tv}, Q::SparseCSR{Tv,Ti}, b::Vector{Tv};
        alpha::Tv=Tv(1), sigma::Tv=Tv(0), omega::Tv=Tv(1))::Nothing where {Tv, Ti}
    m, n = size(Q)
    @assert m == n
    @inbounds for i = 1:m
        tmpd::Tv = 0
        tmpx::Tv = b[j] / alpha
        for z = Q.rowptr[i]:Q.rowptr[i+1]-1
            j = Q.colind[z]
            if i == j
                tmpd = Q.val[z]
                tmpx += sigma * x[j]
            else
                tmpx -= Q.val[z] * x[j]
            end
        end
        x[i] = omega / tmpd * tmpx + (1 - omega) * x[i]
    end
    nothing
end

