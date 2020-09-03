"""
Mixed Matrix Exponential Function
"""

export mexp, mexpmix, mexpc, mexpcmix

"""
mexpmix(f, Q, x; bounds = (0, Inf), transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)
mexp(Q, x, dist, bounds = (minimum(dist), maximum(dist)), transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

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

function mexpmix(f::Any, Q::AbstractMatrix{Tv}, x::Array{Tv,N};
    bounds = (Tv(0.0), Tv(Inf)), transpose::AbstractTranspose = NoTrans(),
    ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    m, n = size(Q)
    @assert m == n
    de = deint(f, bounds[1], bounds[2])
    dt, maxt = itime(de.x)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps)
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    y0, y1 = copy(x), similar(x)
    result = zero(x)
    for i in eachindex(dt)
        right = rightbound(qv*dt[i], eps)
        weight = poipmf!(qv*dt[i], prob; left=0, right=right)
        y1 .= Tv(0)
        unifstep!(transpose, P, prob, (0, right), weight, y0, y1)
        @axpy(de.w[i], y1, result)
        y0 .= y1
    end
    @scal(de.h, result)
    return result
end

function mexp(Q::AbstractMatrix{Tv}, x::Array{Tv,N}, dist::UnivariateDistribution;
    bounds = (minimum(dist), maximum(dist)), transpose::AbstractTranspose = NoTrans(),
    ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    mexpmix(Q, x, bounds=bounds, transpose=transpose, ufact=ufact, eps=eps, rmax=rmax) do x
        pdf(dist, x)
    end
end

"""
mexpcmix(f, Q, x; bounds = (0, Inf), transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)
mexpc(Q, x, dist, bounds = (minimum(dist), maximum(dist)), transpose = NoTrans(), ufact = 1.01, eps = 1.0e-8, rmax = 500)

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

function mexpcmix(f::Any, Q::AbstractMatrix{Tv}, x::Array{Tv,N};
    bounds = (Tv(0.0), Tv(Inf)), transpose::AbstractTranspose = NoTrans(),
    ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    m, n = size(Q)
    @assert m == n
    de = deint(f, bounds[1], bounds[2])
    dt, maxt = itime(de.x)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)
    y0, y1 = copy(x), similar(x)
    cy = zero(x)
    tmp = similar(x)
    result, cresult = zero(x), zero(x)
    for i in eachindex(dt)
        right = rightbound(qv*dt[i], eps) + 1
        weight = cpoipmf!(qv*dt[i], prob, cprob; left=0, right=right)
        tmp .= Tv(0)
        y1 .= Tv(0)
        cunifstep!(transpose, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
        cy .+= tmp
        @axpy(de.w[i], y1, result)
        @axpy(de.w[i], cy, cresult)
        y0 .= y1
    end
    @scal(de.h, result)
    @scal(de.h, cresult)
    return result, cresult
end

function mexpc(Q::AbstractMatrix{Tv}, x::Array{Tv,N}, dist::UnivariateDistribution;
    bounds = (minimum(dist), maximum(dist)), transpose::AbstractTranspose = NoTrans(),
    ufact::Tv = Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,N}
    mexpcmix(Q, x, bounds=bounds, transpose=transpose, ufact=ufact, eps=eps, rmax=rmax) do x
        pdf(dist, x)
    end
end