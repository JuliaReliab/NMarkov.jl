"""
Stationary distribution computation for CTMC

This module provides algorithms for computing the stationary (steady-state) distribution of continuous-time Markov chains.
Supported methods include:
- GTH algorithm (direct method for dense matrices)
- Gauss-Seidel/SOR iterative methods (for sparse matrices)
- Power method (for discrete-time Markov chains)
"""

"""
    GTH (Grassmann-Taksar-Heyman) algorithm

Direct method for computing the stationary distribution of CTMC.
Suitable for dense infinitesimal generators.

Complexity: O(n³)
"""

"""
    gth!(Q)

Compute the stationary distribution of a CTMC using the GTH algorithm (in-place).

The matrix Q is used as workspace and is modified during computation.

### Arguments
- `Q::Matrix`: Infinitesimal generator matrix (must be square)

### Returns
- `pi::Vector`: Stationary probability vector (normalized to sum to 1)

### Algorithm
- GTH (Grassmann-Taksar-Heyman) direct method
- Works for dense matrices
- Requires Q to have no absorbing states

### Example
```julia
Q = [-2.0 2.0; 1.0 -1.0]
pi = gth!(copy(Q))  # Use copy since Q is modified
```

### Notes
- This in-place version modifies Q for efficiency
- Use `gth(Q)` for non-destructive computation
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

"""
    gth(Q)

Compute the stationary distribution of a CTMC using the GTH algorithm.

This non-destructive version makes a copy of Q before computation.

### Arguments
- `Q::Matrix`: Infinitesimal generator matrix (must be square)

### Returns
- `pi::Vector`: Stationary probability vector (normalized to sum to 1)

### Algorithm
- GTH (Grassmann-Taksar-Heyman) direct method
- Works for dense matrices
- Suitable for small to medium-sized matrices

### Example
```julia
Q = [-2.0 2.0; 1.0 -1.0]
pi = gth(Q)
```

### Notes
- Original matrix Q is not modified (makes a copy internally)
- Use `gth!(copy(Q))` if you want to avoid the extra copy
"""
function gth(Q::Matrix{Tv}) where {Tv}
    gth!(copy(Q))
end

"""
    gth!(Q, index)

Compute the stationary distribution of a CTMC using the GTH algorithm with state reordering (in-place).

Allows computation with states reordered according to an index vector.

### Arguments
- `Q::Matrix`: Infinitesimal generator matrix (must be square)
- `index::Vector{Int}`: Permutation vector specifying state reordering

### Returns
- `pi::Vector`: Stationary probability vector (normalized to sum to 1)

### Notes
- States are accessed in the order specified by the index vector
- Useful for reducing numerical errors by processing states in a specific order
- Q is modified as a workspace
- Requires no absorbing states

### Example
```julia
Q = [-2.0 2.0; 1.0 -1.0]
index = [2, 1]  # Process state 2 before state 1
pi = gth!(copy(Q), index)
```
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

"""
    gth(Q, index)

Compute the stationary distribution of a CTMC using the GTH algorithm with state reordering.

Non-destructive version that makes a copy of Q before computation.

### Arguments
- `Q::Matrix`: Infinitesimal generator matrix (must be square)
- `index::Vector{Int}`: Permutation vector specifying state reordering

### Returns
- `pi::Vector`: Stationary probability vector (normalized to sum to 1)

### Notes
- Original matrix Q is not modified
- States are accessed in the order specified by the index vector
- Useful for improving numerical stability

### Example
```julia
Q = [-2.0 2.0; 1.0 -1.0]
index = [2, 1]
pi = gth(Q, index)
```
"""
function gth(Q::Matrix{Tv}, index::Vector{Ti}) where {Tv, Ti}
    gth!(copy(Q), index)
end

"""
    stguess(Q, Tv = Float64)

Generate an initial guess for the stationary distribution.

Provides a reasonable starting vector for iterative methods based on the diagonal elements of Q.

### Arguments
- `Q`: Matrix (CTMC kernel or transition matrix)
- `Tv::Type`: Element type (default: Float64)

### Returns
- `x0::Vector`: Initial guess vector (normalized to sum to 1)

### Notes
- Uses diagonal elements of Q to construct the guess
- Typically provides faster convergence in iterative methods
- Normalized to be a probability distribution

### Example
```julia
Q = [-2.0 2.0; 1.0 -1.0]
x0 = stguess(Q)
```
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
    stgs(Q; x0 = stguess(Q, Tv), maxiter = 5000, steps = 20, rtol = 1.0e-6)

Compute the stationary distribution of a CTMC using Gauss-Seidel iterative method.

Supports both `SparseMatrixCSC` and `SparseCSC` matrix types.

### Arguments
- `Q`: Infinitesimal generator (sparse matrix)
- `x0::Vector`: Initial guess vector (default: `stguess(Q)`)
- `maxiter::Int`: Maximum number of iterations (default: 5000)
- `steps::Int`: Check convergence every n steps (default: 20)
- `rtol::Float`: Relative error tolerance (default: 1.0e-6)

### Returns
- `x::Vector`: Stationary probability vector
- `conv::Bool`: Whether the algorithm converged
- `iter::Int`: Number of iterations performed
- `rerror::Float`: Final relative error

### Algorithm
- Gauss-Seidel successive iteration
- Suitable for sparse matrices
- Converges for ergodic CTMCs

### Example
```julia
Q = SparseCSC(...)
pi, conv, iter, err = stgs(Q, maxiter=5000)
```

### Notes
- Convergence speed depends on matrix conditioning
- Use `steps` parameter to balance frequency of convergence checks
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
    stpower(P; x0 = stguess(P, Tv), maxiter = 5000, steps = 20, rtol = 1.0e-6)

Compute the stationary distribution of a DTMC using the power method.

### Arguments
- `P::AbstractMatrix`: Transition probability matrix for a discrete-time Markov chain
- `x0::Vector`: Initial guess vector (default: `stguess(P)`)
- `maxiter::Int`: Maximum number of iterations (default: 5000)
- `steps::Int`: Check convergence every n steps (default: 20)
- `rtol::Float`: Relative error tolerance (default: 1.0e-6)

### Returns
- `x::Vector`: Stationary probability vector
- `conv::Bool`: Whether the algorithm converged
- `iter::Int`: Number of iterations performed
- `rerror::Float`: Final relative error

### Algorithm
- Power method (iteration with transpose of transition matrix)
- Suitable for discrete-time Markov chains
- Converges for aperiodic, irreducible DTMCs

### Example
```julia
P = [0.9 0.1; 0.2 0.8]
pi, conv, iter, err = stpower(P, maxiter=5000)
```

### Notes
- Convergence depends on the spectral gap of P
- DTMC equivalent of `stgs` for CTMCs
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
    gsstep!(x, Q, b; alpha = 1, sigma = 0, omega = 1)

Perform one Gauss-Seidel or SOR iteration step for linear systems.

Solves the system: `alpha * (A - sigma*I) * x = b` using Gauss-Seidel or SOR method.

### Arguments
- `x::Vector`: Solution vector (modified in-place)
- `Q`: Coefficient matrix (SparseCSC or SparseCSR)
- `b::Vector`: Right-hand side vector
- `alpha::Float`: Scaling factor (default: 1.0)
- `sigma::Float`: Shift parameter (default: 0.0, use for eigenvalue problems)
- `omega::Float`: Over-relaxation parameter (default: 1.0, use 1.0 for GS, 1.0 < ω < 2.0 for SOR)

### Algorithm Details
For **SparseCSC** (column-wise storage):
- Computes: `x[j] := ω/d_j * (b[j]/α - Σ_{i≠j} Q[i,j]*x[i] + σ*x[j]) + (1-ω)*x[j]`

For **SparseCSR** (row-wise storage):
- Computes: `x[i] := ω/d_i * (b[i]/α - Σ_{j≠i} Q[i,j]*x[j] + σ*x[i]) + (1-ω)*x[i]`

### Returns
- Nothing (modifies `x` in-place)

### Notes
- Uses the diagonal element as `d_j` or `d_i`
- Suitable for solving the system `π*Q = 0` (stationary distribution)
- SparseCSC format processes column-wise
- SparseCSR format processes row-wise
- SOR improves convergence when properly tuned

### Example
```julia
Q = SparseCSC(...)
b = zeros(n)
x = ones(n) ./ n  # Initial guess
for step = 1:100
    gsstep!(x, Q, b, alpha=1.0, sigma=0.0)
    x ./= sum(x)
end
```
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

