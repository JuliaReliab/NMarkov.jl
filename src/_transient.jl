"""
Transient analysis for CTMC
"""

"""
mexp(Q, x, t; transpose = :N, ufact = 1.01, eps = 1.0e-8, rmax = 500)

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
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    m, n = size(Q)
    @assert m == n
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps)
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    weight, poi = poipmf(qv*t, right, left = 0)
    y = zero(x)
    unifstep!(transpose, P, poi, (0, right), weight, copy(x), y)
    return y
end

"""
mexpc(Q, x, t; transpose = :N, ufact = 1.01, eps = 1.0e-8, rmax = 500)

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
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    m, n = size(Q)
    @assert m == n
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    weight, poi, cpoi = cpoipmf(qv*t, right, left = 0)
    y = zero(x)
    cy = zero(x)
    cunifstep!(transpose, P, poi, cpoi, (0, right), weight, qv*weight, copy(x), y, cy)
    return y, cy
end

"""
mexp(Q, x, ts; transpose = :N, ufact = 1.01, eps = 1.0e-8, rmax = 500)

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
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(sort(ts))
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps)
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    result = Vector{Array{Tv,N}}(undef, length(dt))
    y0 = copy(x)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps)
        weight = poipmf!(qv*dt[k], prob; left=0, right=right)
        y1 = zero(y0)
        unifstep!(transpose, P, prob, (0, right), weight, y0, y1)
        result[k] = y1
        y0 .= y1
    end
    return result
end

"""
mexpc(Q, x, ts; transpose = :N, ufact = 1.01, eps = 1.0e-8, rmax = 500)

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
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(ts)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)
    result = Vector{Array{Tv,N}}(undef, length(dt))
    cresult = Vector{Array{Tv,N}}(undef, length(dt))
    y0 = copy(x)
    cy = zero(x)
    tmp = similar(x)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
        y1 = zero(y0)
        tmp .= Tv(0)
        cunifstep!(transpose, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
        cy .+= tmp
        result[k] = copy(y1)
        cresult[k] = copy(cy)
        y0 .= y1
    end
    return result, cresult
end

