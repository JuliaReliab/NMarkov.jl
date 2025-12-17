

"""
Transient analysis for CTMC
"""

"""
mexp(Q, x, t; transpose = :N, ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC.

exp(tr(Q)*t) * x

Parameters:
- Q: CTMC Kernel
- x: Array (any numeric type, will be converted to Float64)
- t: time (any numeric type, will be converted to Float64)
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value:
- probability vector
"""

# Wrapper function to handle type conversions (for mixed types)
function mexp(Q::AbstractMatrix{Tv}, x::AbstractArray, t::Union{Int, Float32, Float16};
    transpose::Symbol=:N, ufact::Real=1.01, eps::Real=1.0e-8, rmax=500) where {Tv}
    x_float = vec(convert(Array{Tv}, x))
    t_float = convert(Tv, t)
    ufact_float = convert(Tv, ufact)
    eps_float = convert(Tv, eps)
    return mexp(Q, x_float, t_float; transpose=transpose, ufact=ufact_float, eps=eps_float, rmax=rmax)
end

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
- x: Array (any numeric type, will be converted to Float64)
- t: time (any numeric type, will be converted to Float64)
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value (tuple)
- probability vector
- cumulative value
"""

# Wrapper function to handle type conversions (for mixed types)
function mexpc(Q::AbstractMatrix{Tv}, x::AbstractArray, t::Union{Int, Float32, Float16};
    transpose::Symbol=:N, ufact::Real=1.01, eps::Real=1.0e-8, rmax=500) where {Tv}
    x_float = vec(convert(Array{Tv}, x))
    t_float = convert(Tv, t)
    ufact_float = convert(Tv, ufact)
    eps_float = convert(Tv, eps)
    return mexpc(Q, x_float, t_float; transpose=transpose, ufact=ufact_float, eps=eps_float, rmax=rmax)
end

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
- x: Array (any numeric type, will be converted to Float64)
- ts: time series (any numeric type, will be converted to Float64)
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value:
- probability vector
"""

# Wrapper function to handle type conversions (for mixed types)
function mexp(Q::AbstractMatrix{Tv}, x::AbstractArray, ts::AbstractVector;
    transpose::Symbol=:N, ufact::Real=1.01, eps::Real=1.0e-8, rmax=500) where {Tv}
    if !(eltype(x) <: Tv && eltype(ts) <: Tv)
        x_float = vec(convert(Array{Tv}, x))
        ts_float = convert(Vector{Tv}, ts)
        ufact_float = convert(Tv, ufact)
        eps_float = convert(Tv, eps)
        return mexp(Q, x_float, ts_float; transpose=transpose, ufact=ufact_float, eps=eps_float, rmax=rmax)
    end
    error("Method not found for these exact types")
end

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

"""
Mixed Matrix Exponential Function
"""

"""
mexpmix(f, Q, x; bounds = (0, Inf), transpose = :N, ufact = 1.01, eps = 1.0e-8, rmax = 500)
mexp(Q, x, dist, bounds = (minimum(dist), maximum(dist)), transpose = :N, ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC mixed with dist

int_bound[1]^bound[2] exp(tr(Q)*t) f(t) dt * x
int_bound[1]^bound[2] exp(tr(Q)*t) pdf(dist, t) dt * x

Parameters:
- Q: CTMC Kernel
- x: Array
- f: pdf of distribution
- dist: distribution (UnivariateDistribution)
- bounds: a tuple of domain of distribution
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value:
- probability vector
"""

# function mexpmix(f::Any, Q::AbstractMatrix{Tv}, x::ArrayT;
#     bounds=(Tv(0.0), Tv(Inf)), transpose::Symbol=:N,
#     ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
#     m, n = size(Q)
#     @assert m == n
#     de = deint(f, bounds[1], bounds[2])
#     dt, maxt = itime(de.x)
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps)
#     @assert right <= rmax "Time interval is too large. rmax should be changed: right = $right (rmax: $rmax)."
#     prob = Vector{Tv}(undef, right+1)
#     y0, y1 = copy(x), similar(x)
#     result = zero(x)
#     @inbounds for i in eachindex(dt)
#         right = rightbound(qv*dt[i], eps)
#         weight = poipmf!(qv*dt[i], prob; left=0, right=right)
#         y1 .= Tv(0)
#         unifstep!(transpose, P, prob, (0, right), weight, y0, y1)
#         @axpy(de.w[i], y1, result)
#         y0 .= y1
#     end
#     @scal(de.h, result)
#     return result
# end

@inbounds function mexpmix(f::Any, Q::AbstractMatrix{Tv}, x::ArrayT;
    bounds=(Tv(0.0), Tv(Inf)), transpose::Symbol=:N,
    ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    de = deint(f, bounds[1], bounds[2])
    dt, maxt = itime(de.x)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps)
    @assert right <= rmax "Time interval is too large. rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)

    y0 = copy(x)
    y1 = similar(x)
    tmpv = similar(x)
    result = zero(x)
    for k in eachindex(dt)
        right = rightbound(qv*dt[k], eps)
        weight = poipmf!(qv*dt[k], prob; left=0, right=right)

        @. y1 = zero(Tv)
        @origin (prob => 0) begin
            axpy!(prob[0], y0, y1)
            for i = 1:right
                matmul!(transpose, 1.0, P, y0, false, tmpv)
                @. y0 = tmpv
                axpy!(prob[i], y0, y1)
            end
        end
        scal!(1/weight, y1)
        axpy!(de.w[k], y1, result)
        @. y0 = y1
    end
    scal!(de.h, result)
end

function mexp(Q::AbstractMatrix{Tv}, x::ArrayT, dist::UnivariateDistribution;
    bounds=(minimum(dist), maximum(dist)), transpose::Symbol=:N,
    ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
    mexpmix(Q, x, bounds=bounds, transpose=transpose, ufact=ufact, eps=eps, rmax=rmax) do x
        pdf(dist, x)
    end
end

"""
mexpcmix(f, Q, x; bounds = (0, Inf), transpose = :N, ufact = 1.01, eps = 1.0e-8, rmax = 500)
mexpc(Q, x, dist, bounds = (minimum(dist), maximum(dist)), transpose = :N, ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute the probability vector for CTMC and the cumulative value for time series.
Compute the probability vector for CTMC and the cumulative value which are mixed with dist

int exp(tr(Q)*t) * f(t) dt * x
int int_0^t exp(tr(Q)*u) * x du f(t) dt 

Parameters:
- Q: CTMC Kernel
- x: Array
- f: pdf of distribution
- dist: distribution (UnivariateDistribution)
- bounds: a tuple of domain of distribution
- transpose: forward or backward
- ufact: uniformization factor
- eps: tolerance error for Poisson p.m.f.
- rmax: The maximum number of uniformization steps

Return value (tuple)
- probability vector
- cumulative value
"""

# function mexpcmix(f::Any, Q::AbstractMatrix{Tv}, x::ArrayT;
#     bounds=(Tv(0.0), Tv(Inf)), transpose::Symbol=:N,
#     ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
#     m, n = size(Q)
#     @assert m == n
#     de = deint(f, bounds[1], bounds[2])
#     dt, maxt = itime(de.x)
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps) + 1
#     @assert right <= rmax "Time interval is too large. rmax should be changed: right = $right (rmax: $rmax)."
#     prob = Vector{Tv}(undef, right+1)
#     cprob = Vector{Tv}(undef, right+1)
#     y0, y1 = copy(x), similar(x)
#     cy = zero(x)
#     tmp = similar(x)
#     result, cresult = zero(x), zero(x)
#     @inbounds for i in eachindex(dt)
#         right = rightbound(qv*dt[i], eps) + 1
#         weight = cpoipmf!(qv*dt[i], prob, cprob; left=0, right=right)
#         tmp .= Tv(0)
#         y1 .= Tv(0)
#         cunifstep!(transpose, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
#         cy .+= tmp
#         @axpy(de.w[i], y1, result)
#         @axpy(de.w[i], cy, cresult)
#         y0 .= y1
#     end
#     @scal(de.h, result)
#     @scal(de.h, cresult)
#     return result, cresult
# end

@inbounds function mexpcmix(f::Any, Q::AbstractMatrix{Tv}, x::ArrayT;
    bounds=(Tv(0.0), Tv(Inf)), transpose::Symbol=:N,
    ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    de = deint(f, bounds[1], bounds[2])
    dt, maxt = itime(de.x)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large. rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)

    y0 = copy(x)
    y1 = similar(x)
    cy = zero(x)
    tmpv = similar(x)
    result = zero(x)
    cresult = zero(x)
    for k in eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)

        @. y1 = zero(Tv)
        @origin (prob=>0, cprob=>0) begin
            axpy!(prob[0], y0, y1)
            axpy!(cprob[0]/(qv*weight), y0, cy)
            for i = 1:right
                matmul!(transpose, 1.0, P, y0, false, tmpv)
                @. y0 = tmpv
                axpy!(prob[i], y0, y1)
                axpy!(cprob[i]/(qv*weight), y0, cy)
            end
        end
        scal!(1/weight, y1)
        axpy!(de.w[k], y1, result)
        axpy!(de.w[k], cy, cresult)
        @. y0 = y1
    end
    scal!(de.h, result), scal!(de.h, cresult)
end

function mexpc(Q::AbstractMatrix{Tv}, x::ArrayT, dist::UnivariateDistribution;
    bounds = (minimum(dist), maximum(dist)), transpose::Symbol=:N,
    ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT<:AbstractArray{Tv}}
    mexpcmix(Q, x, bounds=bounds, transpose=transpose, ufact=ufact, eps=eps, rmax=rmax) do x
        pdf(dist, x)
    end
end