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
- x: Array (inout). x may be changed after executing
- y: Array (out). y should be zero before executing
Return value: nothing
"""

function unifforward!(P::AbstractMatrix{Tv}, poi::Vector{Tv}, weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N})::Nothing where {Tv,N}
    Pdash = P'
    @daxpy(poi[1], x, y)
    for i = 2:length(poi)
        x .= Pdash * x
        @daxpy(poi[i], x, y)
    end
    @dscal(1/weight, y)
    nothing
end

function unifbackward!(P::AbstractMatrix{Tv}, poi::Vector{Tv}, weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N})::Nothing where {Tv,N}
    @daxpy(poi[1], x, y)
    for i = 2:length(poi)
        x .= P * x
        @daxpy(poi[i], x, y)
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
- x: Array (inout). x may be changed after executing
- y: Array (out). y should be zero before executing
- cy: Array (inout). cy should be zero before executing
Return value: nothing
"""

function cunifforward!(P::AbstractMatrix{Tv}, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,N}
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

function cunifbackward!(P::AbstractMatrix{Tv}, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,N}
    Pdash = P'
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

function _cunifstep!(::Trans, P::AbstractMatrix{Tv}, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,N}
    cunifforward!(P, poi, cpoi, weight, qv_weight, x, y, cy)
end

function _cunifstep!(::NoTrans, P::AbstractMatrix{Tv}, poi::Vector{Tv}, cpoi::Vector{Tv}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Tv,N}
    cunifbackward!(P, poi, cpoi, weight, qv_weight, x, y, cy)
end

"""
mexp(Q, x, t; transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC.

exp(tr(Q)*t) * x

Parameters:
- Q: CTMC Kernel
- x: Array
- t: time
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value:
- probability vector
"""

function mexp(Q::AbstractMatrix{Tv}, x::Array{Tv,N}, t::Tv;
    transpose::AbstractTranspose = NoTrans(), ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps)
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    weight, poi = poipmf(qv*t, right, left = 0)
    y = zero(x)
    _unifstep!(transpose, P, poi, weight, copy(x), y)
    return y
end

"""
mexpc(Q, x, t; transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC and the cumulative value.

exp(tr(Q)*t) * x
int_0^t exp(tr(Q)*u) * x du

Parameters:
- Q: CTMC Kernel
- x: Array
- t: time
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value (tuple)
- probability vector
- cumulative value
"""

function mexpc(Q::AbstractMatrix{Tv}, x::Array{Tv,N}, t::Tv;
    transpose::AbstractTranspose = NoTrans(), ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps) + 1
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    weight, poi, cpoi = cpoipmf(qv*t, right, left = 0)
    y = zero(x)
    cy = zero(x)
    _cunifstep!(transpose, P, poi, cpoi, weight, qv*weight, copy(x), y, cy)
    return y, cy
end

"""
mexp(Q, x, ts; transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC for time series

exp(tr(Q)*t) * x for t = ts

Parameters:
- Q: CTMC Kernel
- x: Array
- ts: time series
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value:
- probability vector
"""

function mexp(Q::AbstractMatrix{Tv}, x::Array{Tv,N}, ts::AbstractVector{Tv};
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
    y0 = copy(x)
    for tau in dt
        right = rightbound(qv*tau, eps)
        weight = poipmf!(qv*tau, prob; left=0, right=right)
        y1 = zero(y0)
        _unifstep!(transpose, P, prob, weight, y0, y1)
        push!(result, y1)
        y0 .= y1
    end
    return result
end

"""
mexpc(Q, x, ts; transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC and the cumulative value for time series.

exp(tr(Q)*t) * x for t = ts
int_0^t exp(tr(Q)*u) * x du for t = ts

Parameters:
- Q: CTMC Kernel
- x: Array
- ts: time series
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value (tuple)
- probability vector
- cumulative value
"""

function mexpc(Q::AbstractMatrix{Tv}, x::Array{Tv,N}, ts::AbstractVector{Tv};
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
    y0 = copy(x)
    cy = zero(x)
    tmp = similar(x)
    for tau in dt
        right = rightbound(qv*tau, eps) + 1
        weight = cpoipmf!(qv*tau, prob, cprob; left=0, right=right)
        y1 = zero(y0)
        tmp .= Tv(0)
        _cunifstep!(transpose, P, prob, cprob, weight, qv*weight, y0, y1, tmp)
        cy .+= tmp
        push!(result, y1)
        push!(cresult, copy(cy))
        y0 .= y1
    end
    return result, cresult
end
