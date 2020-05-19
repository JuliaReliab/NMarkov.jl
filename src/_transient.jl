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

y = exp(tr(Q)*t) * x

where Q is unifomed by P = I - Q/qv. In the computation, Poisson p.m.f. with mean qv*t is used.

Parameters:
- P: The uniformed matrix
- poi: Poisson p.m.f.
- weight: The normalizing constant for Poisson p.m.f.
- x: Array (in)
- y: Array (out)
Return value: nothing
"""

function unifforward!(P::AbstractMatrix{Tv}, poi::Vector{Tv}, weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N})::Nothing where {Tv,N}
    y .= Tv(0)
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
    y .= Tv(0)
    xi = copy(x)
    @daxpy(poi[1], xi, y)
    for i = 2:length(poi)
        xi .= P * xi
        @daxpy(poi[i], xi, y)
    end
    @dscal(1/weight, y)
    nothing
end

function _unifstep!(::Trans, P::AbstractMatrix{Tv}, poi::Vector{Tv}, weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N})::Nothing where {Tv,N}
    unifforward!(P, poi, weight, x, y)
end

function _unifstep!(::NoTrans, P::AbstractMatrix{Tv}, poi::Vector{Tv}, weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N})::Nothing where {Tv,N}
    unifbackward!(P, poi, weight, x, y)
end

"""
cunifforward!(P, poi, weight, x, y)
cunifbackward!(P, poi, weight, x, y)

Compute the probability vector and cmulative one using the uniformized CTMC.

y = exp(tr(Q)*t) * x
cy = cy + int_0^t exp(tr(Q)*u) * x du

where Q is unifomed by P = I - Q/qv. In the computation, Poisson p.m.f. with mean qv*t is used.

Parameters:
- P: The uniformed matrix
- poi: Poisson p.m.f.
- cpoi: Poisson c.c.d.f.
- weight: The normalizing constant for Poisson p.m.f.
- qv_weight: The normalizing constant for Poisson c.c.d.f.
- x: Array (in)
- y: Array (out)
- cy: Array (inout)
Return value: nothing
"""

function cunifforward!(P::AbstractMatrix{Tv}, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,N}
    y .= Tv(0)
    xi = copy(x)
    cxi = zero(x)
    Pdash = P'
    @daxpy(poi[1], xi, y)
    @daxpy(cpoi[1], xi, cxi)
    for i = 2:length(poi)
        xi .= Pdash * xi
        @daxpy(poi[i], xi, y)
        @daxpy(cpoi[i], xi, cxi)
    end
    @dscal(1/weight, y)
    @dscal(1/qv_weight, cxi)
    cy .+= cxi
    nothing
end

function cunifbackward!(P::AbstractMatrix{Tv}, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,N}
    y .= Tv(0)
    xi = copy(x)
    cxi = zero(x)
    Pdash = P'
    @daxpy(poi[1], xi, y)
    @daxpy(cpoi[1], xi, cxi)
    for i = 2:length(poi)
        xi .= P * xi
        @daxpy(poi[i], xi, y)
        @daxpy(cpoi[i], xi, cxi)
    end
    @dscal(1/weight, y)
    @dscal(1/qv_weight, cxi)
    cy .+= cxi
    nothing
end

function _cunifstep!(::Trans, P::AbstractMatrix{Tv}, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,N}
    cunifforward!(P, poi, cpoi, weight, qv_weight, x, y, cy)
end

function _cunifstep!(::NoTrans, P::AbstractMatrix{Tv}, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,N}
    cunifbackward!(P, poi, cpoi, weight, qv_weight, x, y, cy)
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
    transpose::AbstractTranspose = NoTrans(), ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps)
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    weight, poi = poipmf(qv*t, right, left = 0)
    y = similar(x)
    _unifstep!(transpose, P, poi, weight, x, y)
    return y
end

# function _mexp(Q::AbstractMatrix{Tv}, t::Tv, x::Array{Tv,N}, ::Trans, ufact::Tv, eps::Tv, rmax) where {Tv,N}
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*t, eps)
#     @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
#     weight, poi = poipmf(qv*t, right, left = 0)
#     y = zero(x)
#     _unifstep!(P, poi, weight, x, y)
#     return y
# end

# function _mexp(Q::AbstractMatrix{Tv}, t::Tv, x::Array{Tv,N}, ::NoTrans, ufact::Tv, eps::Tv, rmax) where {Tv,N}
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*t, eps)
#     @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
#     weight, poi = poipmf(qv*t, right, left = 0)
#     y = zero(x)
#     unifbackward!(P, poi, weight, x, y)
#     return y
# end

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

Return value (tuple)
- probability vector
- cumulative value
"""

function mexpc(Q::AbstractMatrix{Tv}, t::Tv, x::Array{Tv,N};
    transpose::AbstractTranspose = NoTrans(), ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps) + 1
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    weight, poi, cpoi = cpoipmf(qv*t, right, left = 0)
    y = similar(x)
    cy = zero(x)
    _cunifstep!(transpose, P, poi, cpoi, weight, qv*weight, x, y, cy)
    return y, cy
end

# function _mexpc(Q::AbstractMatrix{Tv}, t::Tv, x::Array{Tv,N}, ::Trans, ufact::Tv, eps::Tv, rmax) where {Tv,N}
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*t, eps) + 1
#     @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
#     weight, poi, cpoi = cpoipmf(qv*t, right, left = 0)
#     y = zero(x)
#     cy = zero(x)
#     cunifforward!(P, poi, cpoi, weight, qv*weight, x, y, cy)
#     return y, cy
# end

# function _mexpc(Q::AbstractMatrix{Tv}, t::Tv, x::Array{Tv,N}, ::NoTrans, ufact::Tv, eps::Tv, rmax) where {Tv,N}
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*t, eps) + 1
#     @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
#     weight, poi, cpoi = cpoipmf(qv*t, right, left = 0)
#     y = zero(x)
#     cy = zero(x)
#     cunifbackward!(P, poi, cpoi, weight, qv*weight, x, y, cy)
#     return y, cy
# end

"""
mexp(Q, ts, x, transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC for time series

exp(tr(Q)*t) * x for t = ts

Parameters:
- Q: CTMC Kernel
- ts: time series
- x: Array
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value:
- probability vector
"""

function mexp(Q::AbstractMatrix{Tv}, ts::AbstractVector{Tv}, x::Array{Tv,N};
    transpose::AbstractTranspose = NoTrans(), ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    dt = diff(ts)
    @assert all(dt .>= zero(Tv))
    pushfirst!(dt, ts[1])
    maxt = maximum(dt)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps)
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    result = Vector{Array{Tv,N}}()
    y0 = x
    for tau in dt
        right = rightbound(qv*tau, eps)
        weight = poipmf!(qv*tau, prob; left=0, right=right)
        y1 = similar(y0)
        _unifstep!(transpose, P, prob, weight, y0, y1)
        push!(result, y1)
        y0 = y1
    end
    return result
end

"""
mexpc(Q, ts, x; transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC and the cumulative value for time series.

exp(tr(Q)*t) * x for t = ts
int_0^t exp(tr(Q)*u) * x du for t = ts

Parameters:
- Q: CTMC Kernel
- t: time
- x: Array
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value (tuple)
- probability vector
- cumulative value
"""

function mexpc(Q::AbstractMatrix{Tv}, ts::AbstractVector{Tv}, x::Array{Tv,N};
    transpose::AbstractTranspose = NoTrans(), ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    dt = diff(ts)
    @assert all(dt .>= zero(Tv))
    pushfirst!(dt, ts[1])
    maxt = maximum(dt)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)
    result = Vector{Array{Tv,N}}()
    cresult = Vector{Array{Tv,N}}()
    y0, cy0 = x, zero(x)
    for tau in dt
        right = rightbound(qv*tau, eps) + 1
        weight = cpoipmf!(qv*tau, prob, cprob; left=0, right=right)
        y1, cy1 = similar(y0), copy(cy0)
        _cunifstep!(transpose, P, prob, cprob, weight, qv*weight, y0, y1, cy1)
        push!(result, y1)
        push!(cresult, cy1)
        y0, cy0 = y1, cy1
    end
    return result, cresult
end
