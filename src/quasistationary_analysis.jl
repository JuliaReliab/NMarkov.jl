"""
    qstgs(Q, xi; x0, maxiter, steps, rtol)
    qstpower(P, xi; x0, maxiter, steps, rtol)

Quasi-stationary analysis for Markov chains with absorbing boundaries.

This module provides methods to compute quasi-stationary distributions (QSD)
and absorption rates for Markov chains with one or more absorbing states.

The quasi-stationary distribution describes the conditional distribution of
the chain states given non-absorption up to time t, in the limit t→∞.

## Applications

- Reliability and survival analysis: behavior conditioned on no failure
- Population dynamics: conditioned on non-extinction
- Fluid flow: conditioned on non-overflow
- Exit rates and quasi-equilibrium analysis

## Supported Methods

- **Gauss-Seidel iteration**: `qstgs` - For CTMCs with sparse matrices
- **Power method**: `qstpower` - For DTMCs with eigenvalue computation

## Key Concepts

**Quasi-stationary distribution (QSD)**: The limiting conditional distribution
of states when conditioned on non-absorption:

`π_QST = lim_{t→∞} P(X(t)=i | τ > t)` where τ is absorption time

**Exit rate (absorption rate)**: The conditional absorption rate given QSD:

`γ = ∫ π_QST * e dt` where e is the exit vector

## Example

```julia
using NMarkov

# 4-state CTMC with absorption
Q = [-2.0  1.0  0.5  0.5;    # transient states
      0.5 -1.0  0.3  0.2;
      0.5  0.2 -1.0  0.3;
      0.0  0.0  0.0  0.0]    # absorbing state

# Exit vector (rates to absorption)
xi = [1.0; 1.0; 1.0; 0.0]

# Quasi-stationary distribution (GS iteration)
qst, gamma, conv, iter, rerr = qstgs(Q, xi)

# Interpretation:
# qst = QSD (conditioned on non-absorption)
# gamma = absorption rate for chain in QSD
```
"""

"""
    qstgs(Q, xi; x0, maxiter, steps, rtol)

Compute the quasi-stationary distribution for CTMC using Gauss-Seidel iteration.

Iteratively solves for the QSD vector q and exit rate γ such that:
`(Q - γ*I) * q = 0` with normalization `sum(q) = 1`

## Arguments

- `Q::Union{SparseMatrixCSC, SparseCSC}`: generator restricted to the **transient**
  states (the block usually written `T`), not the full generator. Every diagonal
  entry must be non-zero; the absorbing state's zero row must not be included.
- `xi::Vector`: Exit vector (rates to absorption); must be non-negative, and
  satisfies `Q * ones + xi == 0`
- `x0::Vector=stguess(Q)`: Initial guess for QSD (default: uniform/diagonal-based)
- `maxiter::Int=5000`: Maximum number of iterations
- `steps::Int=20`: Number of GS steps between convergence checks
- `rtol::Real=1.0e-6`: Relative error tolerance for convergence

## Returns

Tuple of five elements:
1. `x::Vector`: Quasi-stationary distribution (normalized to sum(x) = 1)
2. `gam::Real`: Absorption rate for chain in QSD (eigenvalue)
3. `conv::Bool`: Convergence flag (true if `rerror < rtol`)
4. `iter::Int`: Number of iterations performed
5. `rerror::Real`: Relative error at termination

## Algorithm

Implements GS iteration with power method eigenvalue tracking:

1. Initialize: x <- x0
2. For iteration k:
   - Compute absorption probability: γ = dot(x, ξ)
   - Apply GS step: x <- gsstep!(x, Q, 0, sigma=-γ)
   - Normalize: x <- x / sum(x)
   - Check convergence: rerror = max(|x_new - x_old|) / max(x_new)
3. Stop when: rerror < rtol or iter >= maxiter

The parameter σ = -γ implements the shifted operator (Q - γ*I).

## Mathematical Formulation

The QSD satisfies the eigenvalue problem:
`Q * q = -γ * q` where γ is the conditional absorption rate

Subject to normalization: `sum(q) = 1`

The exit rate is computed as: `γ = q' * ξ`

## Convergence

For CTMCs with well-separated spectral gap:
- Typical convergence: 50-500 iterations
- Depends on proximity of second-largest eigenvalue to 0
- May require parameter tuning for nearly-reversible chains

## Physical Interpretation

- **x vector**: Long-term state distribution conditioned on non-absorption
- **gam value**: Rate at which chain exits (absorbed) from QSD
- **High gam**: Chain absorbs quickly from QSD
- **Low gam**: Chain persists long before absorption

## Computational Notes

- Time complexity per iteration: O(nnz(Q))
- Memory: O(n) for vectors only
- Suitable for large sparse matrices (n >= 1000)
- Requires stable initialization for convergence

## Example

```julia
using NMarkov
using SparseArrays

# 4-state CTMC: 3 transient states plus 1 absorbing state.
# Pass only the transient block T and the exit rates xi. Handing over the full
# 4-by-4 generator would include the absorbing state's zero row, whose zero
# diagonal Gauss-Seidel cannot divide by.
T = [-4.0  1.0  0.0;
      0.0 -1.0  0.1;
      3.0  0.5 -3.5]
xi = -vec(sum(T, dims=2))        # [3.0, 0.9, 0.0]; satisfies T*ones + xi == 0

qst, gamma, conv, iter, rerr = qstgs(sparse(T), xi; rtol=1.0e-6)
println("QSD: \$qst")
println("Absorption rate: \$gamma")
println("Converged: \$conv in \$iter iterations")
```
"""

function qstgs(Q::SparseMatrixCSC{Tv,Ti}, xi::Vector{Tv}; x0::Vector{Tv}=stguess(Q,Tv),
        maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    qstgs(SparseCSC(Q), xi, x0=x0, maxiter=maxiter, steps=steps, rtol=rtol)
end

function qstgs(Q::SparseCSC{Tv,Ti}, xi::Vector{Tv}; x0::Vector{Tv}=stguess(Q,Tv),
        maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv,Ti}
    m, n = size(Q)
    @assert m == n
    # For the transient block T of an absorbing chain,
    # diag(T)[j] = -(sum_{k != j} T[j,k] + xi[j]), so a zero diagonal entry means
    # state j has neither an onward transition nor an exit rate: it is absorbing,
    # hence not a transient state at all. The usual cause is passing the full
    # generator, whose absorbing state contributes a zero row.
    checkgsdiag("qstgs", Q,
        "Quasi-stationary analysis takes the generator restricted to the transient " *
        "states together with their exit rates, not the full generator including " *
        "the absorbing state's zero row.")
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
        # gam above belongs to the iterate that entered the last sweep, not to
        # the x being returned; recompute it so the pair is consistent (this is
        # what qstpower does).
        gam = @dot(x, xi)
        # rerror = maximum(abs.((x - prevx) ./ x))
        rerror = maximum(abs.(x - prevx)) / maximum(x)
        iter += steps
        if rerror < rtol
            conv = true
            break
        end
        if iter >= maxiter
            @warn "qstgs did not converge within $maxiter iterations "  *
                  "(relative error $rerror, tolerance $rtol); the returned " *
                  "value is the last iterate"
            break
        end
    end
    return x, gam, conv, iter, rerror
end

"""
    qstpower(P, xi; x0, maxiter, steps, rtol)

Compute the quasi-stationary distribution for DTMC using Power method.

Iteratively solves for the QSD vector q and dominant eigenvalue λ such that:
`P' * q = λ * q` with normalization `sum(q) = 1`

## Arguments

- `P::AbstractMatrix`: DTMC transition probability matrix
- `xi::Vector`: Exit vector (absorption/transition rates); must be non-negative
- `x0::Vector=stguess(P)`: Initial guess for QSD (default: uniform)
- `maxiter::Int=5000`: Maximum number of iterations
- `steps::Int=20`: Number of power steps between convergence checks
- `rtol::Real=1.0e-6`: Relative error tolerance for convergence

## Returns

Tuple of five elements:
1. `x::Vector`: Quasi-stationary distribution (normalized to sum(x) = 1)
2. `nu::Real`: Quasi-stationary parameter (inner product dot(x, ξ))
3. `conv::Bool`: Convergence flag (true if `rerror < rtol`)
4. `iter::Int`: Number of iterations performed
5. `rerror::Real`: Relative error at termination

## Algorithm

Implements power method for QSD computation:

1. Initialize: x <- x0
2. For iteration k:
   - Apply `steps` power iterations: x <- P'*x
   - Normalize: x <- x / sum(x)
   - Check convergence: rerror = max(|x_new - x_old|) / max(x_new)
3. After convergence, compute: ν = dot(x, ξ)
4. Stop when: rerror < rtol or iter >= maxiter

Each power iteration multiplies by transpose of transition matrix P'.

## Mathematical Formulation

The QSD satisfies the eigenvector equation:
`P' * q = λ * q` with `sum(q) = 1`

where λ is the dominant eigenvalue (< 1 for absorbing chains).

The quasi-stationary parameter:
`ν = q' * ξ` measures relative absorption rate

## Convergence Rate

For DTMC with spectral gap δ = 1 - λ₂ (gap to second-largest eigenvalue):
- Convergence rate: O(λ₂^k)
- Faster for larger gaps (λ₂ << 1)
- Slower for nearly-periodic matrices (λ₂ ≈ 1)

## Physical Interpretation

- **x vector**: Quasi-stationary distribution (long-term conditioned distribution)
- **nu value**: Expected absorption rate from QSD
- **Eigenvalue λ**: Probability of non-absorption in one step (from QSD)

## Computational Notes

- Time complexity per iteration: O(n²) or O(nnz(P)) for sparse
- Memory: O(n) for vectors only
- Suitable for small-to-medium dense DTMCs
- Faster than dense linear algebra for sparse matrices

## Comparison with qstgs

- **qstgs** (GS): Better for CTMCs, faster convergence typically
- **qstpower** (Power): Better for DTMCs, simpler implementation, direct eigenvalue

## Example

```julia
using NMarkov

# DTMC with absorption: P includes transition to absorbing state
P = [0.7  0.2  0.0  0.1;    # state 1
     0.1  0.7  0.1  0.1;    # state 2
     0.0  0.2  0.7  0.1;    # state 3
     0.0  0.0  0.0  1.0]    # absorbing state

xi = [1.0; 1.0; 1.0; 0.0]   # exit rates (0 for absorbing)

qst, nu, conv, iter, rerr = qstpower(P, xi; rtol=1.0e-6)

println("QSD: \$qst")
println("Quasi-stationary parameter: \$nu")
println("Absorbed in \$iter iterations (converged: \$conv)")
```
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
        # rerror = maximum(abs.((x - prevx) ./ x))
        rerror = maximum(abs.(x - prevx)) / maximum(x)
        iter += steps
        if rerror < rtol
            conv = true
            break
        end
        if iter >= maxiter
            @warn "qstpower did not converge within $maxiter iterations "  *
                  "(relative error $rerror, tolerance $rtol); the returned " *
                  "value is the last iterate"
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

