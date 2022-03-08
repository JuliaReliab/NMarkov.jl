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

# function mexp(Q::AbstractMatrix{Tv}, x::ArrayT, t::Tv;
#     transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv, ArrayT <: AbstractArray{Tv}}
#     m, n = size(Q)
#     @assert m == n
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*t, eps)
#     @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
#     weight, poi = poipmf(qv*t, right, left = 0)
#     y = zero(x)
#     unifstep!(transpose, P, poi, (0, right), weight, copy(x), y)
#     return y
# end

@inbounds function mexp(Q::AbstractMatrix{Tv}, x::ArrayT, t::Tv;
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv, ArrayT <: AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps)
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    weight, poi = poipmf(qv*t, right, left = 0)

    y = zero(x)
    xtmp = copy(x)
    tmpv = similar(x)
    @origin (poi => 0) begin
        axpy!(poi[0], xtmp, y)
        for i = 1:right
            matmul!(transpose, 1.0, P, xtmp, false, tmpv)
            @. xtmp = tmpv
            axpy!(poi[i], xtmp, y)
        end
    end
    scal!(1/weight, y)
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

# function mexpc(Q::AbstractMatrix{Tv}, x::ArrayT, t::Tv;
#     transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv, ArrayT <: AbstractArray{Tv}}
#     m, n = size(Q)
#     @assert m == n
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*t, eps) + 1
#     @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
#     weight, poi, cpoi = cpoipmf(qv*t, right, left = 0)
#     y = zero(x)
#     cy = zero(x)
#     cunifstep!(transpose, P, poi, cpoi, (0, right), weight, qv*weight, copy(x), y, cy)
#     return y, cy
# end

@inbounds function mexpc(Q::AbstractMatrix{Tv}, x::ArrayT, t::Tv;
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv, ArrayT <: AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    P, qv = unif(Q, ufact)
    right = rightbound(qv*t, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    weight, poi, cpoi = cpoipmf(qv*t, right, left = 0)

    y = zero(x)
    cy = zero(x)
    xtmp = copy(x)
    tmpv = similar(x)
    @origin (poi=>0, cpoi=>0) begin
        axpy!(poi[0], xtmp, y)
        axpy!(cpoi[0], xtmp, cy)
        for i = 1:right
            matmul!(transpose, 1.0, P, xtmp, false, tmpv)
            @. xtmp = tmpv
            axpy!(poi[i], xtmp, y)
            axpy!(cpoi[i], xtmp, cy)
        end
    end
    scal!(1/weight, y), scal!(1/(qv*weight), cy)
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

# function mexp(Q::AbstractMatrix{Tv}, x::ArrayT, ts::AbstractVector{Tv};
#     transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
#     m, n = size(Q)
#     @assert m == n
#     dt, maxt = itime(sort(ts))
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps)
#     @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
#     prob = Vector{Tv}(undef, right+1)

#     result = Vector{Any}(undef, length(dt)) # TODO: memory usage?
#     y0 = copy(x)
#     for k = eachindex(dt)
#         right = rightbound(qv*dt[k], eps)
#         weight = poipmf!(qv*dt[k], prob; left=0, right=right)
#         y1 = zero(y0)
#         unifstep!(transpose, P, prob, (0, right), weight, y0, y1)
#         result[k] = y1
#         y0 .= y1
#     end
#     return result
# end

@inbounds function mexp(Q::AbstractMatrix{Tv}, x::ArrayT, ts::AbstractVector{Tv};
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(sort(ts))
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps)
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)

    result = Vector{Any}(undef, length(dt)) # TODO: memory usage?
    y0 = copy(x)
    xtmp = similar(x)
    tmpv = similar(x)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps)
        weight = poipmf!(qv*dt[k], prob; left=0, right=right)

        y1 = zero(y0)
        @. xtmp = y0
        @origin (prob => 0) begin
            axpy!(prob[0], xtmp, y1)
            for i = 1:right
                matmul!(transpose, 1.0, P, xtmp, false, tmpv)
                @. xtmp = tmpv
                axpy!(prob[i], xtmp, y1)
            end
        end
        result[k] = scal!(1/weight, y1)
        y0 = y1
    end
    result
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

@inbounds function mexpc(Q::AbstractMatrix{Tv}, x::ArrayT, ts::AbstractVector{Tv};
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(ts)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)

    result = Vector{Any}(undef, length(dt)) # TODO: memory usage?
    cresult = Vector{Any}(undef, length(dt)) # TODO: memory usage?
    y0 = copy(x)
    cy = zero(x)
    xtmp = similar(x)
    tmpv = similar(x)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)

        y1 = zero(y0)
        @. xtmp = y0
        @origin (prob=>0, cprob=>0) begin
            axpy!(prob[0], xtmp, y1)
            axpy!(cprob[0]/(qv*weight), xtmp, cy)
            for i = 1:right
                matmul!(transpose, 1.0, P, xtmp, false, tmpv)
                @. xtmp = tmpv
                axpy!(prob[i], xtmp, y1)
                axpy!(cprob[i]/(qv*weight), xtmp, cy)
            end
        end
        result[k] = scal!(1/weight, y1)
        cresult[k] = copy(cy)
        y0 = y1
    end
    result, cresult
end

# function mexpc(Q::AbstractMatrix{Tv}, x::ArrayT, ts::AbstractVector{Tv};
#     transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
#     m, n = size(Q)
#     @assert m == n
#     dt, maxt = itime(ts)
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps) + 1
#     @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
#     prob = Vector{Tv}(undef, right+1)
#     cprob = Vector{Tv}(undef, right+1)

#     result = Vector{Any}(undef, length(dt)) # TODO: memory usage?
#     cresult = Vector{Any}(undef, length(dt)) # TODO: memory usage?
#     y0 = copy(x)
#     cy = zero(x)
#     tmp = similar(x)
#     xtmp = similar(x)
#     tmpv = similar(x)
#     for k = eachindex(dt)
#         right = rightbound(qv*dt[k], eps) + 1
#         weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)

#         y1 = zero(y0)


#         tmp .= Tv(0)
#         cunifstep!(transpose, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
#         cy .+= tmp
#         result[k] = copy(y1)
#         cresult[k] = copy(cy)
#         y0 .= y1
#     end
#     return result, cresult
# end
