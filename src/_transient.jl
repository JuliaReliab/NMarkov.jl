"""
Transient analysis for CTMC
"""

"""
tran(Q, x, r, ts; forward = :T, ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the instantaneous and cumulative rewards for CTMC on time series.

instantaneous reward: x * exp(Q*t) * r for t = ts
cumulative reward: x * int_0^t exp(Q*u) * r du for t = ts

Parameters:
- Q: CTMC Kernel
- x: initial vector.
- r: reward vector.
- ts: time series
- forward: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value (tuple)
- instantaneous reward
- cumulative reward
- probability vector at the last time (forward is :T)
- reward vector at the initial time (forward is :N)
"""

function tran(Q::AbstractMatrix{Tv}, x::ArrayT1, r::ArrayT2, ts::AbstractVector{Tv};
    forward::Symbol=:T, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT1<:AbstractArray{Tv},ArrayT2<:AbstractArray{Tv}}
    _tran(Q, x, r, ts, Val{forward}, ufact, eps, rmax)
end

function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::Array{Tv,1}, ts::AbstractVector{Tv},
    ::Type{Val{:T}}, ufact::Tv, eps::Tv, rmax) where Tv
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(ts)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)
    result = Vector{Tv}(undef, length(dt))
    cresult = Vector{Tv}(undef, length(dt))
    y0 = copy(x)
    y1 = similar(x)
    cy = zero(x)
    tmp = similar(x)
    @inbounds for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
        y1 .= Tv(0)
        tmp .= Tv(0)
        cunifstep!(:T, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
        cy .+= tmp
        result[k] = @dot(y1, r)
        cresult[k] = @dot(cy, r)
        y0 .= y1
    end
    return result, cresult, y1, cy
end

function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::Array{Tv,1}, ts::AbstractVector{Tv},
    ::Type{Val{:N}}, ufact::Tv, eps::Tv, rmax) where Tv
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(ts)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)
    result = Vector{Tv}(undef, length(dt))
    cresult = Vector{Tv}(undef, length(dt))
    y0 = copy(r)
    y1 = similar(r)
    cy = zero(r)
    tmp = similar(r)
    @inbounds for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
        y1 .= Tv(0)
        tmp .= Tv(0)
        cunifstep!(:N, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
        cy .+= tmp
        result[k] = @dot(x, y1)
        cresult[k] = @dot(x, cy)
        y0 .= y1
    end
    return result, cresult, y1, cy
end

function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::ArrayT2, ts::AbstractVector{Tv},
    ::Type{Val{:T}}, ufact::Tv, eps::Tv, rmax) where {Tv,ArrayT2<:AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(ts)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)
    result = Vector{Any}(undef, length(dt))
    cresult = Vector{Any}(undef, length(dt))
    y0 = copy(x)
    y1 = similar(x)
    cy = zero(x)
    tmp = similar(x)
    @inbounds for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
        y1 .= Tv(0)
        tmp .= Tv(0)
        cunifstep!(:T, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
        cy .+= tmp
        result[k] = r' * y1
        cresult[k] = r' * cy
        y0 .= y1
    end
    return result, cresult, y1, cy
end

function _tran(Q::AbstractMatrix{Tv}, x::ArrayT1, r::ArrayT2, ts::AbstractVector{Tv},
    ::Type{Val{:T}}, ufact::Tv, eps::Tv, rmax) where {Tv,ArrayT1<:AbstractArray{Tv},ArrayT2<:AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(ts)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)
    xdash = x'
    result = Vector{Any}(undef, length(dt))
    cresult = Vector{Any}(undef, length(dt))
    y0 = copy(xdash)
    y1 = similar(xdash)
    cy = zero(xdash)
    tmp = similar(xdash)
    @inbounds for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
        y1 .= Tv(0)
        tmp .= Tv(0)
        cunifstep!(:T, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
        cy .+= tmp
        result[k] = y1' * r
        cresult[k] = cy' * r
        y0 .= y1
    end
    return result, cresult, y1', cy'
end

function _tran(Q::AbstractMatrix{Tv}, x::ArrayT1, r::ArrayT2, ts::AbstractVector{Tv},
    ::Type{Val{:N}}, ufact::Tv, eps::Tv, rmax) where {Tv,ArrayT1<:AbstractArray{Tv},ArrayT2<:AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(ts)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)
    result = Vector{Any}(undef, length(dt))
    cresult = Vector{Any}(undef, length(dt))
    y0 = copy(r)
    y1 = similar(r)
    cy = zero(r)
    tmp = similar(r)
    @inbounds for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
        y1 .= Tv(0)
        tmp .= Tv(0)
        cunifstep!(:N, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
        cy .+= tmp
        result[k] = x * y1
        cresult[k] = x * cy
        y0 .= y1
    end
    return result, cresult, y1, cy
end
