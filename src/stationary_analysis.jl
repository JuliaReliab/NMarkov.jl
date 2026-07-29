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
            # The total rate out of state l into the remaining block is the
            # pivot; a zero pivot means state l cannot leave, i.e. the chain has
            # an absorbing state and no stationary distribution exists. Without
            # this check the elimination divides by zero and returns NaN.
            iszero(tmp) && throw(ArgumentError(
                "state $l cannot reach the remaining states: the chain has an " *
                "absorbing state, so it has no unique stationary distribution"))
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
        x[1] = one(Tv)
        for l = 2:n
            x[l] = zero(Tv)
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
        iszero(tmp) && throw(ArgumentError(
            "state $(index[l]) cannot reach the remaining states: the chain has " *
            "an absorbing state, so it has no unique stationary distribution"))
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
    x[index[1]] = one(Tv)
    for l = 2:n
        x[index[l]] = zero(Tv)
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
    d = spdiag(Q)
    for i = 1:n
        x = d[i]
        # A zero diagonal entry carries no rate information (a DTMC state with
        # no self-loop, or an absorbing state). 1/x would be Inf and the
        # normalisation below would turn the whole guess into NaN, which no
        # convergence test can ever satisfy, so fall back to the uniform guess.
        if iszero(x)
            result .= one(Tv) / n
            return result
        end
        result[i] = one(Tv) / x
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

"""
    checkgsdiag(name, Q, hint)

Check that every diagonal entry of `Q` is non-zero, which is a precondition of
the Gauss-Seidel iteration. `hint` is appended to the error message to say what
the caller expected of `Q`.

`gsstep!` divides by the diagonal, and Gauss-Seidel in general splits
`Q = D + (L + U)` and inverts `D`, which is singular as soon as one diagonal
entry vanishes. For a generator `diag(Q)[j] = -sum_{k != j} Q[j,k]` with
non-negative off-diagonals, so a zero diagonal entry means the whole row is
zero: state `j` never leaves.

The obstacle is structural, not numerical. The column equation of such a state
`a`, `sum_{i != a} Q[i,a] pi_i + Q[a,a] pi_a = 0`, does not contain `pi_a` at
all once `Q[a,a] = 0`; that component is fixed by the normalisation rather than
by any sweep, so there is no fixed point to iterate towards. Storing an explicit
zero on the diagonal (`adddiag`) therefore does not help either — zero is still
zero when you divide by it.
"""
function checkgsdiag(name::String, Q::MatT,
        hint::String = "A chain with an absorbing state has no unique stationary " *
                       "distribution reachable this way; use a method for " *
                       "reducible chains.") where {MatT}
    d = spdiag(Q)
    @inbounds for i in eachindex(d)
        iszero(d[i]) && throw(ArgumentError(
            "$name: diagonal entry ($i,$i) is zero, so state $i never leaves and the " *
            "Gauss-Seidel iteration is not defined (it divides by the diagonal). " *
            hint))
    end
    nothing
end

function stgs(Q::SparseCSC{Tv,Ti}; x0::Vector{Tv}=stguess(Q,Tv),
        maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    m, n = size(Q)
    @assert m == n
    checkgsdiag("stgs", Q)
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
            @warn "stgs did not converge within $maxiter iterations "  *
                  "(relative error $rerror, tolerance $rtol); the returned " *
                  "value is the last iterate"
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
            @warn "stpower did not converge within $maxiter iterations "  *
                  "(relative error $rerror, tolerance $rtol); the returned " *
                  "value is the last iterate"
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
        tmpx::Tv = b[i] / alpha
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

