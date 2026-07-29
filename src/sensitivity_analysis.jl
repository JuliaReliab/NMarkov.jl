"""
    stsen(Q, pis, b)
    stsengs(Q, pis, b; x0, maxiter, steps, rtol)
    stsenpower(P, pis, b; x0, maxiter, steps, rtol)
    stsenguess(Q, Tv)
    dtmcstcheck(P, pis; eps)
    ctmcstcheck(Q, pis; eps)

Sensitivity analysis for Markov chain stationary distributions.

This module provides methods to compute sensitivity vectors (Jacobian derivatives)
of the stationary distribution with respect to perturbations in the generator or
transition matrix. Sensitivity analysis is essential for:

- Parameter sensitivity in Markov models
- Derivative-based optimization of system parameters
- Robustness and stability analysis
- First-order and higher-order sensitivity computation

## Supported Methods

- **QR factorization**: `stsen` - Direct method using QR decomposition
- **Gauss-Seidel iteration**: `stsengs` - Iterative method for CTMCs with sparse matrices
- **Power method**: `stsenpower` - Iterative method for DTMCs

## Key Concepts

**Sensitivity Vector**: For CTMC with stationary distribution π, the sensitivity
vector s represents how π changes with perturbations: ∂π/∂θ where θ is a parameter.

**Computation**: Given perturbation vector b = π * ∂Q/∂θ, the sensitivity vector
satisfies: s * Q + b = -π * (s * 1), with constraint sum(s) = 0

## Example

```julia
using NMarkov

# 3-state CTMC generator
Q = [-2.0  1.0  1.0;
      0.5 -1.0  0.5;
      1.0  1.0 -2.0]

# Compute stationary distribution
pi = gth(Q)

# First-order perturbation in Q (e.g., change first diagonal)
dQ = [0.1  0.0  0.0;
      0.0  0.0  0.0;
      0.0  0.0  0.0]
b = pi' * dQ  # perturbation vector

# Sensitivity vector (direct QR method)
s = stsen(Q, pi, b)

# Sensitivity vector (iterative GS method)
s_iter, conv, iter, rerr = stsengs(Q, pi, b)
```
"""

"""
    stsen(Q, pis, b)

Compute the sensitivity vector for stationary distribution using QR factorization.

For CTMC with generator Q and stationary distribution π, computes sensitivity
vector s = ∂π/∂θ where perturbation is b = π * ∂Q/∂θ.

## Arguments

- `Q::Matrix`: CTMC generator matrix of size (n, n)
- `pis::Vector`: Stationary distribution (π*Q = 0, sum(π) = 1)
- `b::Vector`: Perturbation vector b = π * ∂Q/∂θ, where ∂Q/∂θ is the sensitivity of Q

## Returns

- `x::Vector`: Sensitivity vector s such that s*Q + b = -π*(s'*1) and sum(s) = 0

## Algorithm

Uses QR factorization of Q' to solve the constrained linear system:

1. Compute QR decomposition: Q' = QR (R is upper triangular)
2. Set R[n,n] = 1 (replace rank-deficiency row with constraint sum(s) = 0)
3. Solve: s = inv(R) * (-(Q'*b) with constraint)
4. Orthogonalize: s <- s - sum(s)*π

## Sensitivity Equation

The sensitivity vector satisfies:
`∂(π*Q = 0)/∂θ => s*Q + π*∂Q/∂θ = 0`

Equivalently:
`s*Q = -b` where `b = π*∂Q/∂θ`

With normalization constraint: `s'*1 = 0` (orthogonal to stationary vector)

## Numerical Notes

- Direct method: computationally stable for small-to-medium matrices (n ≤ 1000)
- Suitable for one-time sensitivity computation
- Time complexity: O(n³) due to QR factorization
- No iteration tolerance issues

## Example

```julia
Q = [-1.5  1.0  0.5;
      1.0 -1.5  0.5;
      0.5  0.5 -1.0]
pis = gth(Q)
dQ = [0.1  0.0  0.0;
      0.0  0.0  0.0;
      0.0  0.0  0.0]
b = pis' * dQ
s = stsen(Q, pis, b)
```
"""

function stsen(Q::Matrix{Tv}, pis::Vector{Tv}, b::Vector{Tv})::Vector{Tv} where Tv
    m, n = size(Q)
    @assert m == n
    @assert ctmcstcheck(Q, pis)
    qm, rm = qr(Q')
    rm[m,n] = Tv(1)
    xx = rm \ (-(qm' * b))
    xx - sum(xx) * pis
end


"""
    stsenguess(Q, Tv=Float64)

Get an initial guess vector for iterative sensitivity computation.

Returns a zero vector of appropriate type and length for iterative methods
solving the sensitivity equation.

## Arguments

- `Q::AbstractMatrix`: CTMC generator matrix (used only for dimension)
- `Tv::Type=Float64`: Element type for the guess vector

## Returns

- Zero vector of size (n,) with element type Tv

## Notes

- Suitable for iterative methods (Gauss-Seidel, Power method)
- Zero vector is conservative initial guess for convergence analysis
- Can be overridden with better guesses if available

## Example

```julia
Q = [-1.0  1.0;  0.5  -0.5]
x0 = stsenguess(Q, Float64)  # returns [0.0, 0.0]
```
"""

function stsenguess(Q::MatT, ::Type{Tv} = Float64)::Vector{Tv} where {Tv,MatT}
    m, n = size(Q)
    @assert m == n
    fill(Tv(0), n)
end

"""
    dtmcstcheck(P, pis; eps=1.0e-8)
    ctmcstcheck(Q, pis; eps=1.0e-8)

Check whether a given vector is a stationary distribution.

Verifies if pis satisfies the stationary condition within tolerance eps.

## Arguments

- `P::AbstractMatrix` or `Q::AbstractMatrix`: Transition/generator matrix
- `pis::Vector`: Candidate stationary distribution
- `eps::Real=1.0e-8`: Tolerance for verification

## Returns

- `true` if residual is below tolerance, `false` otherwise

## Stationary Conditions

- **DTMC**: `P'*pis = pis` (equivalently `pis'*(P-I) = 0`)
- **CTMC**: `Q'*pis = 0` (equivalently `pis'*Q = 0`)

Verification checks: `max(|P'*pis - pis|) < eps` or `max(|Q'*pis|) < eps`

## Notes

- Does NOT verify normalization: `sum(pis) = 1`
- Useful for sanity-checking computed solutions
- Essential before computing sensitivity vectors

## Example

```julia
Q = [-2.0  1.0  1.0;
      0.5 -1.0  0.5;
      1.0  1.0 -2.0]
pi = gth(Q)
is_valid = ctmcstcheck(Q, pi, eps=1.0e-8)  # should be true
```
"""

# The default tolerance follows the element type: a fixed 1.0e-8 sits below the
# resolution of Float32 (about 1.2e-7), so no Float32 stationary vector could
# ever pass. sqrt(eps(Tv)) is 1.5e-8 for Float64, i.e. the previous behaviour.
_stcheck_tol(::Type{Tv}) where {Tv<:AbstractFloat} = sqrt(Base.eps(Tv))
_stcheck_tol(::Type{Tv}) where {Tv} = Tv(1.0e-8)

function dtmcstcheck(P::MatT, pis::Vector{Tv}; eps = _stcheck_tol(Tv)) where {Tv,MatT}
    v = P' * pis - pis
    maximum(abs.(v)) < eps
end

function ctmcstcheck(Q::MatT, pis::Vector{Tv}; eps = _stcheck_tol(Tv)) where {Tv,MatT}
    v = Q' * pis
    maximum(abs.(v)) < eps
end

"""
    stsengs(Q, pis, b; x0, maxiter, steps, rtol)

Compute the sensitivity vector for CTMC using Gauss-Seidel iteration.

Iteratively solves the sensitivity equation s*Q = -b + λ*pis using
Gauss-Seidel method on the sparse linear system with diagonal dominance.

## Arguments

- `Q::Union{SparseMatrixCSC, SparseCSC}`: CTMC generator matrix
- `pis::Vector`: Stationary distribution (π*Q = 0, sum(π) = 1)
- `b::Vector`: Perturbation vector b = π*∂Q/∂θ
- `x0::Vector=stsenguess(Q)`: Initial guess vector (default: zero)
- `maxiter::Int=5000`: Maximum number of iterations
- `steps::Int=20`: Number of GS steps between convergence checks
- `rtol::Real=1.0e-6`: Relative error tolerance for convergence

## Returns

Tuple of four elements:
1. `x::Vector`: Computed sensitivity vector
2. `conv::Bool`: Convergence flag (true if `rerror < rtol`)
3. `iter::Int`: Number of iterations performed
4. `rerror::Real`: Relative error at termination

## Algorithm

Implements Gauss-Seidel iteration with spectral scaling:

1. Initialize: x <- x0
2. For iteration k:
   - Apply `steps` GS sweeps: x <- gsstep!(x, Q, b, alpha=-1)
   - Orthogonalize: x <- x - sum(x)*π
   - Check convergence: rerror = max(|x_new - x_old|) / max(x_new)
3. Stop when: rerror < rtol or iter >= maxiter

The orthogonalization step enforces constraint sum(s) = 0.

## Sensitivity Equation

Solves: `s*Q + b = -λ*π` where λ = s'*1

Equivalently, using matrix form: `Q'*s^T = -b^T` with constraint.

## Convergence

For CTMC with stochastically dominant generator (diagonally dominant):
- Typical convergence: 10-100 iterations
- Rate determined by eigenvalue gap of Q
- May require loose tolerance (`rtol = 1.0e-4`) for ill-conditioned matrices

## Computational Notes

- Time complexity per iteration: O(nnz(Q)) where nnz is nonzero count
- Memory: O(n) for vectors, no intermediate matrix storage
- Suitable for large sparse matrices (n >= 1000)
- Convergence may require parameter tuning for near-singular systems

## Example

```julia
using NMarkov
using SparseArrays

Q = spdiagm(-2.0*ones(3), 0, 3, 3) + spdiagm(ones(2), 1, 3, 3) + spdiagm(ones(2), -1, 3, 3)
pis = gth(Q)
dQ = sparse([1], [1], [0.1], 3, 3)
b = pis' * dQ |> vec

s, conv, iter, rerr = stsengs(Q, pis, b; rtol=1.0e-6)
println("Convergence: \$conv, Iterations: \$iter, Error: \$rerr")
```
"""

function stsengs(Q::SparseMatrixCSC{Tv,Ti}, pis::Vector{Tv}, b::Vector{Tv};
    x0::Vector{Tv}=stsenguess(Q,Tv), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    stsengs(SparseCSC(Q), pis, b, x0=x0, maxiter=maxiter, steps=steps, rtol=rtol)
end

function stsengs(Q::SparseCSC{Tv,Ti}, pis::Vector{Tv}, b::Vector{Tv};
    x0::Vector{Tv}=stsenguess(Q,Tv), maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    m, n = size(Q)
    @assert m == n
    # Same Gauss-Seidel precondition as stgs; see checkgsdiag.
    checkgsdiag("stsengs", Q)
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
            axpy!(-sum(x), pis, x)
        end
        # A sensitivity vector sums to zero, so it has negative entries and
        # maximum(x) is not a meaningful scale for it (it is 0/0 when the
        # vector is identically zero, which never satisfies rtol). Normalise by
        # the largest magnitude instead.
        den = maximum(abs.(x))
        rerror = iszero(den) ? zero(Tv) : maximum(abs.(x - prevx)) / den
        iter += steps
        if rerror < rtol
            conv = true
            break
        end
        if iter >= maxiter
            @warn "stsengs did not converge within $maxiter iterations "  *
                  "(relative error $rerror, tolerance $rtol); the returned " *
                  "value is the last iterate"
            break
        end
    end
    return x, conv, iter, rerror
end

"""
    stsenpower(P, pis, b; x0, maxiter, steps, rtol)

Compute the sensitivity vector for DTMC using Power method iteration.

Iteratively solves the sensitivity equation s*P = -b + λ*pis using
power method on the sparse system with matrix-vector multiplication.

## Arguments

- `P::AbstractMatrix`: DTMC transition probability matrix
- `pis::Vector`: Stationary distribution (π*P = π, sum(π) = 1)
- `b::Vector`: Perturbation vector b = π*∂P/∂θ
- `x0::Vector=stsenguess(P)`: Initial guess vector (default: zero)
- `maxiter::Int=5000`: Maximum number of iterations
- `steps::Int=20`: Number of power steps between convergence checks
- `rtol::Real=1.0e-6`: Relative error tolerance for convergence

## Returns

Tuple of four elements:
1. `x::Vector`: Computed sensitivity vector
2. `conv::Bool`: Convergence flag (true if `rerror < rtol`)
3. `iter::Int`: Number of iterations performed
4. `rerror::Real`: Relative error at termination

## Algorithm

Implements power method iteration for sensitivity computation:

1. Initialize: x <- x0
2. For iteration k:
   - Apply `steps` power iterations: x <- P'*x + b
   - Orthogonalize: x <- x - sum(x)*π
   - Check convergence: rerror = max(|x_new - x_old|) / max(x_new)
3. Stop when: rerror < rtol or iter >= maxiter

Each power iteration computes matrix-vector product P'*x.

## Sensitivity Equation

Solves: `s*P + b = -λ*π` where λ = s'*1

Equivalently: `P'*s^T = -b^T` with normalization constraint.

## Convergence

For DTMC with spectral gap δ (gap between 1 and second-largest eigenvalue):
- Convergence rate: O(ρ^k) where ρ < 1
- Faster for well-separated eigenvalues (ρ ≈ 0)
- Slower for nearly-stochastic matrices (ρ ≈ 1)

## Computational Notes

- Time complexity per iteration: O(n²) or O(nnz(P)) if sparse
- Memory: O(n) for vectors only
- Suitable for dense or sparse DTMC transition matrices
- Power method: simple but convergence may be slow for ill-conditioned systems

## Comparison with stsengs

- **stsengs** (Gauss-Seidel): Better for CTMCs, often faster
- **stsenpower** (Power method): Better for DTMCs, simpler implementation

## Example

```julia
using NMarkov

# DTMC transition matrix
P = [0.9  0.1  0.0;
     0.2  0.7  0.1;
     0.0  0.2  0.8]

# Stationary distribution
pis, conv, iter, rerr = stpower(P)

# Perturbation
dP = [0.05  -0.05  0.0;
      0.0   0.0   0.0;
      0.0   0.0   0.0]
b = pis' * dP |> vec

# Sensitivity (power method)
s, conv, iter, rerr = stsenpower(P, pis, b; rtol=1.0e-6)
println("Convergence: \$conv, Iterations: \$iter")
```
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
            axpy!(-sum(x), pis, x)
        end
        # A sensitivity vector sums to zero, so it has negative entries and
        # maximum(x) is not a meaningful scale for it (it is 0/0 when the
        # vector is identically zero, which never satisfies rtol). Normalise by
        # the largest magnitude instead.
        den = maximum(abs.(x))
        rerror = iszero(den) ? zero(Tv) : maximum(abs.(x - prevx)) / den
        iter += steps
        if rerror < rtol
            conv = true
            break
        end
        if iter >= maxiter
            @warn "stsenpower did not converge within $maxiter iterations "  *
                  "(relative error $rerror, tolerance $rtol); the returned " *
                  "value is the last iterate"
            break
        end
    end
    return x, conv, iter, rerror
end


