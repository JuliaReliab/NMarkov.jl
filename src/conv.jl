# Convolution for CTMC.
#
# This file implements the uniformization-based convolution integral of
# H. Okamura, T. Dohi and K. S. Trivedi, "A Refined EM Algorithm for PH
# Distributions", Performance Evaluation. Equation numbers below refer to that
# paper. Check the paper before changing any summation range here.
#
#   H = int_0^y exp(T tau) v1 v2 exp(T (y-tau)) d tau                    eq.(2)
#     = (1/r) sum_{m=0}^{U} alpha_m beta_m                              eq.(24)
#   alpha_0 = v1,      alpha_m = P alpha_{m-1}                          eq.(25)
#   beta_U  = pi_{U+1} v2                                               eq.(18)
#   beta_m  = beta_{m+1} P + pi_{m+1} v2,   m = U-1 .. 0                eq.(17)
#
# Correspondence with the code (`range = (left, right)`, `poi[k] = pi_k`):
#
#   paper                     code
#   ------------------------  --------------------------------------------
#   beta_m                    vc[left + m + 1]
#   alpha_m                   x, multiplied by P once per iteration
#   U                         right - left - 1
#   (1/r) sum alpha_m beta_m  the spger! accumulation, divided by qv_weight
#
# U is `right - left - 1` rather than `right` because beta_U needs pi_{U+1}:
# with a single Poisson vector covering [left, right], the highest weight the
# H series can reach is poi[right], which pins U one below the top.
#
# Dividing by `weight` is not an optional normalisation. `poipmf!` seeds its
# recurrence at the mode with a Stirling approximation, so every returned value
# carries the same multiplicative error (weight is about 1.017, i.e. 1+1/(12*mode),
# for lambda = 5); dividing by the sum removes it. It also compensates the
# truncated tail, which is an O(eps) effect on top.

"""
    convunifstep!(trQ, trH, P, poi, range, weight, qv_weight, x, y, z, H)

Matrix exponential convolution operation using uniformization.

Computes the time-convolved integral of matrix exponentials:

`H = int_0^t exp(Q'*s) * x * y' * exp(Q'*(t-s)) ds`

simultaneously with the instantaneous:

`z = exp(Q'*t) * x`

This is essential for computing cumulative two-variable reward functionals
in transient analysis of CTMCs.

## Arguments

- `trQ::Symbol`: Transposition mode for Q (`:N` for forward, `:T` for backward)
- `trH::Symbol`: Transposition mode for H result (`:N` for row-wise, `:T` for column-wise)
- `P::AbstractMatrix`: Uniformized transition matrix (from `unif()`)
- `poi::Vector`: Poisson probability mass function values
- `range::Tuple{Int,Int}`: `(left, right)` indices for Poisson summation limits
- `weight::Real`: Normalization constant for instantaneous: `exp(-q*t) * sum_k (q*t)^k/k!`
- `qv_weight::Real`: Normalization constant for convolution: same or adjusted
- `x::Array`: Initial state vector (modified during execution)
- `y::Array`: Second vector for convolution (not modified)
- `z::Array`: Output array for instantaneous result (must be zero-initialized)
- `H::Matrix`: Output array for convolution integral (must be zero-initialized)

## Returns

- Nothing; results stored in-place in `z` and `H`

## Algorithm

Implements the uniformization-based convolution formula:

1. Prepare backward vectors: `vc[k]` = reversed Poisson-weighted P' powers
   - `vc[right] = poi[right] * y`
   - `vc[k] = P' * vc[k+1] + poi[k] * y`

2. Forward computation with outer product accumulation:
   - Initialize: `z += poi[0] * x`
   - `H += outer(x, vc[left+1])`
   - For each forward step k: apply P*x, accumulate weighted contributions
   - `z += poi[k] * P^k*x`
   - `H += outer(P^k*x, vc[k+1])`

3. Normalization: `z /= weight`, `H /= qv_weight`

## Mathematical Background

The convolution integral satisfies:

`d/dt H(t) = exp(Q'*t)*x*y' + Q'*H(t)`

Solution: `H(t) = int_0^t exp(Q'*s)*x*y'*exp(Q'*(t-s)) ds`

Uniformization converts this to:

`H(t) = (1/q*weight) * sum_i sum_j poi[i]*poi[j] * P^i*x*y'*P'^j`

where the backward vectors vc[j] pre-accumulate the P' powers with Poisson weights.

## Numerical Stability

- Separate normalization (weight, qv_weight) for accuracy
- Backward vector computation prevents numerical overflow
- Outer product operations use robust BLAS routines

## Memory Requirements

- Intermediate: O(right-left+1) * n for backward vectors vc
- In-place: no additional matrix allocation beyond outputs z, H

## Example

```julia
using NMarkov

Q = [-2.0  1.0  1.0;
      0.5 -1.0  0.5;
      1.0  1.0 -2.0]

x = [1.0; 0.0; 0.0]
y = [0.0; 1.0; 0.0]

P, qv = unif(Q, 1.01)
t = 1.5
right = rightbound(qv*t, 1.0e-8)
weight, poi = poipmf(qv*t, right, left=0)

z = zero(x)
H = zeros(3, 3)

convunifstep!(:N, :N, P, poi, (0, right), weight, weight, x, y, z, H)

# z = exp(Q'*1.5) * x
# H = integral of exp(Q'*s)*x*y'*exp(Q'*(1.5-s)) ds for s in [0, 1.5]
```
"""

function _checkconvrange(poi::Vector, range::Tuple{Ti,Ti}) where {Ti}
    left, right = range
    left >= 0 || throw(ArgumentError("the left Poisson bound must be non-negative, got $left"))
    right >= left || throw(ArgumentError("the Poisson range must satisfy left <= right, got ($left, $right)"))
    # poi is read as poi[left]..poi[right] under `@origin (poi => left)`, i.e.
    # physical 1..right-left+1, inside an @inbounds block.
    length(poi) >= right - left + 1 || throw(ArgumentError(
        "the Poisson vector holds $(length(poi)) elements, but the range " *
        "[$left, $right] needs $(right - left + 1)"))
    nothing
end

function convunifstep!(trQ::Symbol, trH::Symbol,
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    _convunifstep!(Val(trQ), Val(trH), P, poi, range, weight, qv_weight, x, y, z, H)
end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:N}, ::Val{:N},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    @inbounds begin
        left, right = range
        _checkconvrange(poi, range)
        tmpv = similar(x)
        vc = Vector{Vector{Tv}}(undef, right - left + 1)
        vc[right] = zero(x)
        axpy!(poi[right], y, vc[right])
        for l = right-1:-1:left+1
            vc[l] = similar(x)
            gemv!('T', one(Tv), P, vc[l+1], false, vc[l])
            axpy!(poi[l], y, vc[l])
        end

        # The two accumulations below deliberately stop at different points,
        # because they are truncations of two different series.
        #
        #   z is the matrix exponential itself, eq.(apppp):
        #       exp(T t) ~ sum_{m=0}^{U} pi_m P^m,  U = rightbound(r t, eps)
        #     and `right` IS that U, so z runs m = left .. right.
        #
        #   H is the convolution integral, eq.(24), whose own truncation point
        #     is U = right-left-1 (see the header: beta_U needs pi_{U+1}), so it
        #     runs m = 0 .. right-left-1, i.e. it pairs step m with vc[m+1] and
        #     stops one short of `right`.
        #
        # Giving z the H range would leave it summing poi[left..right-1] while
        # `weight` covers poi[left..right]: the numerator and the denominator
        # would no longer be the same range, z would not be the normalised
        # average of anything, and with a stochastic P it would lose
        # poi[right]/weight of the probability mass.
        axpy!(poi[left], x, z)
        right > left && spger!(one(Tv), x, vc[left+1], one(Tv), H)
        for l = left+1:right
            gemv!('N', one(Tv), P, x, false, tmpv)
            @. x = tmpv
            axpy!(poi[l], x, z)
            l < right && spger!(one(Tv), x, vc[l+1], one(Tv), H)
        end
        scal!(one(Tv)/weight, z)
        scal!(one(Tv)/qv_weight, H)
        nothing
    end
end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:T}, ::Val{:N},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    @inbounds begin
        left, right = range
        _checkconvrange(poi, range)
        tmpv = similar(x)
        vc = Vector{Vector{Tv}}(undef, right - left + 1)
        vc[right] = zero(x)
        axpy!(poi[right], y, vc[right])
        for l = right-1:-1:left+1
            vc[l] = similar(x)
            gemv!('N', one(Tv), P, vc[l+1], false, vc[l])
            axpy!(poi[l], y, vc[l])
        end

        # z runs to `right`, H stops at right-1: see the (:N,:N) method above.
        axpy!(poi[left], x, z)
        right > left && spger!(one(Tv), x, vc[left+1], one(Tv), H)
        for l = left+1:right
            gemv!('T', one(Tv), P, x, false, tmpv)
            @. x = tmpv
            axpy!(poi[l], x, z)
            l < right && spger!(one(Tv), x, vc[l+1], one(Tv), H)
        end
        scal!(one(Tv)/weight, z)
        scal!(one(Tv)/qv_weight, H)
        nothing
    end
end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:N}, ::Val{:T},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    @inbounds begin
        left, right = range
        _checkconvrange(poi, range)
        tmpv = similar(x)
        vc = Vector{Vector{Tv}}(undef, right - left + 1)
        vc[right] = zero(x)
        axpy!(poi[right], y, vc[right])
        for l = right-1:-1:left+1
            vc[l] = similar(x)
            gemv!('T', one(Tv), P, vc[l+1], false, vc[l])
            axpy!(poi[l], y, vc[l])
        end

        # z runs to `right`, H stops at right-1: see the (:N,:N) method above.
        axpy!(poi[left], x, z)
        right > left && spger!(one(Tv), vc[left+1], x, one(Tv), H)
        for l = left+1:right
            gemv!('N', one(Tv), P, x, false, tmpv)
            @. x = tmpv
            axpy!(poi[l], x, z)
            l < right && spger!(one(Tv), vc[l+1], x, one(Tv), H)
        end
        scal!(one(Tv)/weight, z)
        scal!(one(Tv)/qv_weight, H)
        nothing
    end
end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:T}, ::Val{:T},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    @inbounds begin
        left, right = range
        _checkconvrange(poi, range)
        tmpv = similar(x)
        vc = Vector{Vector{Tv}}(undef, right - left + 1)
        vc[right] = zero(x)
        axpy!(poi[right], y, vc[right])
        for l = right-1:-1:left+1
            vc[l] = similar(x)
            gemv!('N', one(Tv), P, vc[l+1], false, vc[l])
            axpy!(poi[l], y, vc[l])
        end

        # z runs to `right`, H stops at right-1: see the (:N,:N) method above.
        axpy!(poi[left], x, z)
        right > left && spger!(one(Tv), vc[left+1], x, one(Tv), H)
        for l = left+1:right
            gemv!('T', one(Tv), P, x, false, tmpv)
            @. x = tmpv
            axpy!(poi[l], x, z)
            l < right && spger!(one(Tv), vc[l+1], x, one(Tv), H)
        end
        scal!(one(Tv)/weight, z)
        scal!(one(Tv)/qv_weight, H)
        nothing
    end
end