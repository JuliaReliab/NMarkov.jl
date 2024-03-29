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