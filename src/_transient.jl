"""
Transient analysis for CTMC
"""

export unifforward!, unifbackward!
export cunifforward!, cunifbackward!
export mexp, mexpc
export Trans, NoTrans

"""
Trans
NoTrans

Types to represent the forward or backward computation for CTMC
"""

abstract type AbstractTranspose end
struct Trans <: AbstractTranspose end
struct NoTrans <: AbstractTranspose end

"""
unifforward!(P, poi, weight, x, y)
unifbackward!(P, poi, weight, x, y)

Compute the probability vector using the uniformized CTMC.

y = y + exp(tr(Q)*t) * x

where Q is unifomed by P = I - Q/qv. In the computation, Poisson p.m.f. with mean qv*t is used.

Parameters:
- P: The uniformed matrix
- poi: Poisson p.m.f.
- weight: The normalizing constant for Poisson p.m.f.
- x: Array (input)
- y: Array (output)
Return value: nothing
"""

function unifforward!(P::AbstractMatrix{Tv}, poi::Vector{Tv}, weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N})::Nothing where {Tv,N}
    xi = copy(x)
    Pdash = P'
    @daxpy(poi[1], xi, y)
    for i = 2:length(poi)
        xi .= Pdash * xi
        @daxpy(poi[i], xi, y)
    end
    @dscal(1/weight, y)
    nothing
end

function unifbackward!(P::AbstractMatrix{Tv}, poi::Vector{Tv}, weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N})::Nothing where {Tv,N}
    xi = copy(x)
    @daxpy(poi[1], xi, y)
    for i = 2:length(poi)
        xi .= P * xi
        @daxpy(poi[i], xi, y)
    end
    @dscal(1/weight, y)
    nothing
end

"""
cunifforward!(P, poi, weight, x, y)
cunifbackward!(P, poi, weight, x, y)

Compute the probability vector and cmulative one using the uniformized CTMC.

y = y + exp(tr(Q)*t) * x
cy = cy + int_0^t exp(tr(Q)*u) * x du

where Q is unifomed by P = I - Q/qv. In the computation, Poisson p.m.f. with mean qv*t is used.

Parameters:
- P: The uniformed matrix
- poi: Poisson p.m.f.
- cpoi: Poisson c.c.d.f.
- weight: The normalizing constant for Poisson p.m.f.
- qv_weight: The normalizing constant for Poisson c.c.d.f.
- x: Array (input)
- y: Array (output)
- cy: Array (output)
Return value: nothing
"""

function cunifforward!(P::MatT, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x0::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,MatT,N}
    x = copy(x0)
    Pdash = P'
    @daxpy(poi[1], x, y)
    @daxpy(cpoi[1], x, cy)
    for i = 2:length(poi)
        x .= Pdash * x
        @daxpy(poi[i], x, y)
        @daxpy(cpoi[i], x, cy)
    end
    @dscal(1/weight, y)
    @dscal(1/qv_weight, cy)
    nothing
end

function cunifbackward!(P::MatT, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x0::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,MatT,N}
    x = copy(x0)
    @daxpy(poi[1], x, y)
    @daxpy(cpoi[1], x, cy)
    for i = 2:length(poi)
        x .= P * x
        @daxpy(poi[i], x, y)
        @daxpy(cpoi[i], x, cy)
    end
    @dscal(1/weight, y)
    @dscal(1/qv_weight, cy)
    nothing
end

"""
mexp(Q, t, x, transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC.

exp(tr(Q)*t) * x

Parameters:
- Q: CTMC Kernel
- t: time
- x: Array
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value:
- probability vector
"""

function mexp(Q::AbstractMatrix{Tv}, t::Tv, x::Array{Tv,N};
    transpose::AbstractTranspose = NoTrans(),
    ufact::Tv = Tv(1.01),
    eps::Tv=Tv(1.0e-8),
    rmax=500) where {Tv,N}
    _mexp(Q, t, x, transpose, ufact, eps, rmax)
end

function _mexp(Q::AbstractMatrix{Tv}, t::Tv, x::Array{Tv,N}, ::Trans, ufact::Tv, eps::Tv, rmax) where {Tv,N}
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps)
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    weight, poi = poipmf(qv*t, right, left = 0)
    y = zero(x)
    unifforward!(P, poi, weight, x, y)
    return y
end

function _mexp(Q::AbstractMatrix{Tv}, t::Tv, x::Array{Tv,N}, ::NoTrans, ufact::Tv, eps::Tv, rmax) where {Tv,N}
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps)
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    weight, poi = poipmf(qv*t, right, left = 0)
    y = zero(x)
    unifbackward!(P, poi, weight, x, y)
    return y
end

"""
mexpc(Q, t, x; transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC and the cumulative value.

exp(tr(Q)*t) * x
int_0^t exp(tr(Q)*u) * x du

Parameters:
- Q: CTMC Kernel
- t: time
- x: Array
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value:
- probability vector
"""

function mexpc(Q::MatT, t::Tv, x::Array{Tv,N};
    transpose::AbstractTranspose = NoTrans(), ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N,MatT}
    _mexpc(Q, t, x, transpose, ufact, eps, rmax)
end

function _mexpc(Q::MatT, t::Tv, x::Array{Tv,N}, ::Trans, ufact::Tv, eps::Tv, rmax) where {Tv,N,MatT}
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps) + 1
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    weight, poi, cpoi = cpoipmf(qv*t, right, left = 0)
    y = zero(x)
    cy = zero(x)
    cunifforward!(P, poi, cpoi, weight, qv*weight, x, y, cy)
    return y, cy
end

function _mexpc(Q::MatT, t::Tv, x::Array{Tv,N}, ::NoTrans, ufact::Tv, eps::Tv, rmax) where {Tv,N,MatT}
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps) + 1
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    weight, poi, cpoi = cpoipmf(qv*t, right, left = 0)
    y = zero(x)
    cy = zero(x)
    cunifbackward!(P, poi, cpoi, weight, qv*weight, x, y, cy)
    return y, cy
end

# function mexp(Q::AbstractMatrix{Tv}, t::AbstractVector{Tv}, x::Array{Tv,N};
#         transpose::AbstractTranspose = NoTrans(), ufact::Tv = Tv(1.01),
#         eps::Tv=Tv(1.0e-8), abstol::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
#     dt = diff(t)
#     @assert all(dt .>= zero(Tv))
#     pushfirst!(dt, t[1])
#     maxt = maximum(dt)
#     qv, P = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps, abstol=abstol)
#     if right > rmax
#         error("Time interval is too large: right = ", right, " (rmax: ", rmax, ").")
#     end
#     prob = Vector{Tv}(undef, right+1)
#     result = Vector{Array{Tv,N}}()
#     push!(result, x)
#     y0 = x
#     for tau in dt
#         right = rightbound(qv*tau, eps, abstol=abstol)
#         weight = poipmf!(qv*tau, prob; left=0, right=right)
#         y1 = zero(y0)
#         _mexp!(transpose, P, prob, weight, copy(y0), y1)
#         push!(result, y1)
#         y0 = y1
#     end
#     return result
# end

# function cmexp(Q::AbstractMatrix{Tv}, t::AbstractVector{Tv}, x::Array{Tv,N};
#         transpose::AbstractTranspose = NoTrans(), ufact::Tv = Tv(1.01),
#         eps::Tv=Tv(1.0e-8), abstol::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
#     dt = diff(t)
#     @assert all(dt .>= zero(Tv))
#     pushfirst!(dt, t[1])
#     maxt = maximum(dt)
#     qv, P = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps, abstol=abstol) + 1
#     if right > rmax
#         error("Time interval is too large: right = ", right, " (rmax: ", rmax, ").")
#     end
#     prob = Vector{Tv}(undef, right+1)
#     cprob = Vector{Tv}(undef, right+1)
#     result = Vector{Array{Tv,N}}()
#     cresult = Vector{Array{Tv,N}}()
#     tmp = similar(x)
#     y = copy(x)
#     cy = zero(x)
#     push!(result, copy(y))
#     push!(cresult, copy(cy))
#     for tau in dt
#         right = rightbound(qv*tau, eps, abstol=abstol) + 1
#         weight = cpoipmf!(qv*tau, prob, cprob; left=0, right=right)
#         tmp .= y
#         _cmexp!(transpose, P, prob, cprob, weight, qv*weight, tmp, y, cy)
#         push!(result, copy(y))
#         push!(cresult, copy(cy))
#     end
#     return result, cresult
# end
