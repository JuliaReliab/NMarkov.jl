"""
Fundamental computation for CTMC
"""

using Origin

export unifstep!
export cunifstep!

"""
unifstep!(tr, P, poi, range, weight, x, y)

Compute the probability vector using the uniformized CTMC.

y = exp(tr(Q)*t) * x

where Q is unifomed by P = I - Q/qv. In the computation, Poisson p.m.f. with mean qv*t is used.

Parameters:
- tr: transpose operator
- P: The uniformed matrix
- poi: Poisson p.m.f.
- range: domain of Poisson p.m.f
- weight: The normalizing constant for Poisson p.m.f.
- x: Array (inout). x may be changed after executing
- y: Array (out). y should be zero before executing
Return value: nothing
"""
@origin (poi => left) function unifstep!(::Trans, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N})::Nothing where {Ti,Tv,N}
    left, right = range
    Pdash = P'
    @daxpy(poi[left], x, y)
    for i = left+1:right
        x .= Pdash * x
        @daxpy(poi[i], x, y)
    end
    @dscal(1/weight, y)
    nothing
end

@origin (poi => left) function unifstep!(::NoTrans, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N})::Nothing where {Ti,Tv,N}
    left, right = range
    @daxpy(poi[left], x, y)
    for i = left+1:right
        x .= P * x
        @daxpy(poi[i], x, y)
    end
    @dscal(1/weight, y)
    nothing
end

"""
cunifstep!(tr, P, poi, cpoi, range, weight, qv_weight, x, y)

Compute the probability vector and cmulative one using the uniformized CTMC.

y = exp(tr(Q)*t) * x
cy = cy + int_0^t exp(tr(Q)*u) * x du

where Q is unifomed by P = I - Q/qv. In the computation, Poisson p.m.f. with mean qv*t is used.

Parameters:
- tr: transpose operator
- P: The uniformed matrix
- poi: Poisson p.m.f.
- cpoi: Poisson c.c.d.f.
- range: domain of Poisson p.m.f
- weight: The normalizing constant for Poisson p.m.f.
- qv_weight: The normalizing constant for Poisson c.c.d.f.
- x: Array (inout). x may be changed after executing
- y: Array (out). y should be zero before executing
- cy: Array (inout). cy should be zero before executing
Return value: nothing
"""

@origin (poi => left, cpoi => left) function cunifstep!(::Trans, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, cpoi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Ti,Tv,N}
    left, right = range
    Pdash = P'
    @daxpy(poi[left], x, y)
    @daxpy(cpoi[left], x, cy)
    for i = left+1:right
        x .= Pdash * x
        @daxpy(poi[i], x, y)
        @daxpy(cpoi[i], x, cy)
    end
    @dscal(1/weight, y)
    @dscal(1/qv_weight, cy)
    nothing
end

@origin (poi => left, cpoi => left) function cunifstep!(::NoTrans, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, cpoi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, cy::Array{Tv,N})::Nothing where {Ti,Tv,N}
    left, right = range
    @daxpy(poi[left], x, y)
    @daxpy(cpoi[left], x, cy)
    for i = left+1:right
        x .= P * x
        @daxpy(poi[i], x, y)
        @daxpy(cpoi[i], x, cy)
    end
    @dscal(1/weight, y)
    @dscal(1/qv_weight, cy)
    nothing
end

