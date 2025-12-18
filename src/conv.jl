# """
# Convolution for CTMC
# """

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

function convunifstep!(trQ::Symbol, trH::Symbol,
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    _convunifstep!(Val(trQ), Val(trH), P, poi, range, weight, qv_weight, x, y, z, H)
end

# @origin (vc => left, poi => left) function _convunifstep!(::Val{:N}, ::Val{:N},
#     P::AbstractMatrix{Tv},
#     poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
#     x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
#     H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
#     @inbounds begin
#         left, right = range
#         Pdash = P'
#         vc = Vector{Vector{Tv}}(undef, right - left + 1)
#         vc[right] = zero(x)
#         @axpy(poi[right], y, vc[right])
#         for l = right-1:-1:left+1
#             vc[l] = Pdash * vc[l+1]
#             @axpy(poi[l], y, vc[l])
#         end

#         @axpy(poi[left], x, z)
#         _dger!(x, vc[left+1], H)
#         for l = left+1:right-1
#             x .= P * x
#             @axpy(poi[l], x, z)
#             _dger!(x, vc[l+1], H)
#         end
#         @scal(1/weight, z)
#         @scal(1/qv_weight, H)
#         nothing
#     end
# end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:N}, ::Val{:N},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    @inbounds begin
        left, right = range
        tmpv = similar(x)
        vc = Vector{Vector{Tv}}(undef, right - left + 1)
        vc[right] = zero(x)
        axpy!(poi[right], y, vc[right])
        for l = right-1:-1:left+1
            vc[l] = similar(x)
            gemv!('T', 1.0, P, vc[l+1], false, vc[l])
            axpy!(poi[l], y, vc[l])
        end

        axpy!(poi[left], x, z)
        spger!(1.0, x, vc[left+1], 1.0, H)
        for l = left+1:right-1
            gemv!('N', 1.0, P, x, false, tmpv)
            @. x = tmpv
            axpy!(poi[l], x, z)
            spger!(1.0, x, vc[l+1], 1.0, H)
        end
        scal!(1/weight, z)
        scal!(1/qv_weight, H)
        nothing
    end
end

# @origin (vc => left, poi => left) function _convunifstep!(::Val{:T}, ::Val{:N},
#     P::AbstractMatrix{Tv},
#     poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
#     x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
#     H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
#     @inbounds begin
#         left, right = range
#         Pdash = P'
#         vc = Vector{Vector{Tv}}(undef, right - left + 1)
#         vc[right] = zero(x)
#         @axpy(poi[right], y, vc[right])
#         for l = right-1:-1:left+1
#             vc[l] = P * vc[l+1]
#             @axpy(poi[l], y, vc[l])
#         end

#         @axpy(poi[left], x, z)
#         _dger!(x, vc[left+1], H)
#         for l = left+1:right-1
#             x .= Pdash * x
#             @axpy(poi[l], x, z)
#             _dger!(x, vc[l+1], H)
#         end
#         @scal(1/weight, z)
#         @scal(1/qv_weight, H)
#         nothing
#     end
# end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:T}, ::Val{:N},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    @inbounds begin
        left, right = range
        tmpv = similar(x)
        vc = Vector{Vector{Tv}}(undef, right - left + 1)
        vc[right] = zero(x)
        axpy!(poi[right], y, vc[right])
        for l = right-1:-1:left+1
            vc[l] = similar(x)
            gemv!('N', 1.0, P, vc[l+1], false, vc[l])
            axpy!(poi[l], y, vc[l])
        end

        axpy!(poi[left], x, z)
        spger!(1.0, x, vc[left+1], 1.0, H)
        for l = left+1:right-1
            gemv!('T', 1.0, P, x, false, tmpv)
            @. x = tmpv
            axpy!(poi[l], x, z)
            spger!(1.0, x, vc[l+1], 1.0, H)
        end
        scal!(1/weight, z)
        scal!(1/qv_weight, H)
        nothing
    end
end

# @origin (vc => left, poi => left) function _convunifstep!(::Val{:N}, ::Val{:T},
#     P::AbstractMatrix{Tv},
#     poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
#     x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
#     H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
#     @inbounds begin
#         left, right = range
#         Pdash = P'
#         vc = Vector{Vector{Tv}}(undef, right - left + 1)
#         vc[right] = zero(x)
#         @axpy(poi[right], y, vc[right])
#         for l = right-1:-1:left+1
#             vc[l] = Pdash * vc[l+1]
#             @axpy(poi[l], y, vc[l])
#         end

#         @axpy(poi[left], x, z)
#         _dger!(vc[left+1], x, H)
#         for l = left+1:right-1
#             x .= P * x
#             @axpy(poi[l], x, z)
#             _dger!(vc[l+1], x, H)
#         end
#         @scal(1/weight, z)
#         @scal(1/qv_weight, H)
#         nothing
#     end
# end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:N}, ::Val{:T},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    @inbounds begin
        left, right = range
        tmpv = similar(x)
        vc = Vector{Vector{Tv}}(undef, right - left + 1)
        vc[right] = zero(x)
        axpy!(poi[right], y, vc[right])
        for l = right-1:-1:left+1
            vc[l] = similar(x)
            gemv!('T', 1.0, P, vc[l+1], false, vc[l])
            axpy!(poi[l], y, vc[l])
        end

        axpy!(poi[left], x, z)
        spger!(1.0, vc[left+1], x, 1.0, H)
        for l = left+1:right-1
            gemv!('N', 1.0, P, x, false, tmpv)
            @. x = tmpv
            axpy!(poi[l], x, z)
            spger!(1.0, vc[l+1], x, 1.0, H)
        end
        scal!(1/weight, z)
        scal!(1/qv_weight, H)
        nothing
    end
end

# @origin (vc => left, poi => left) function _convunifstep!(::Val{:T}, ::Val{:T},
#     P::AbstractMatrix{Tv},
#     poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
#     x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
#     H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
#     @inbounds begin
#         left, right = range
#         Pdash = P'
#         vc = Vector{Vector{Tv}}(undef, right - left + 1)
#         vc[right] = zero(x)
#         @axpy(poi[right], y, vc[right])
#         for l = right-1:-1:left+1
#             vc[l] = P * vc[l+1]
#             @axpy(poi[l], y, vc[l])
#         end

#         @axpy(poi[left], x, z)
#         _dger!(vc[left+1], x, H)
#         for l = left+1:right-1
#             x .= Pdash * x
#             @axpy(poi[l], x, z)
#             _dger!(vc[l+1], x, H)
#         end
#         @scal(1/weight, z)
#         @scal(1/qv_weight, H)
#         nothing
#     end
# end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:T}, ::Val{:T},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    @inbounds begin
        left, right = range
        tmpv = similar(x)
        vc = Vector{Vector{Tv}}(undef, right - left + 1)
        vc[right] = zero(x)
        axpy!(poi[right], y, vc[right])
        for l = right-1:-1:left+1
            vc[l] = similar(x)
            gemv!('N', 1.0, P, vc[l+1], false, vc[l])
            axpy!(poi[l], y, vc[l])
        end

        axpy!(poi[left], x, z)
        spger!(1.0, vc[left+1], x, 1.0, H)
        for l = left+1:right-1
            gemv!('T', 1.0, P, x, false, tmpv)
            @. x = tmpv
            axpy!(poi[l], x, z)
            spger!(1.0, vc[l+1], x, 1.0, H)
        end
        scal!(1/weight, z)
        scal!(1/qv_weight, H)
        nothing
    end
end