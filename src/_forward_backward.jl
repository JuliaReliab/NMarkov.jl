"""
Fundamental computation for CTMC
"""

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

function unifstep!(tr::Symbol, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv,
    x::ArrayT, y::ArrayT)::Nothing where {Ti,Tv,ArrayT<:AbstractArray{Tv}}
    _unifstep!(Val(tr), P, poi, range, weight, x, y)
end

@origin (poi => left) function _unifstep!(::Val{:T}, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv,
    x::ArrayT, y::ArrayT)::Nothing where {Ti,Tv,ArrayT<:AbstractArray{Tv}}
    left, right = range
    Pdash = P'
    @axpy(poi[left], x, y)
    for i = left+1:right
        x .= Pdash * x
        @axpy(poi[i], x, y)
    end
    @scal(1/weight, y)
    nothing
end

@origin (poi => left) function _unifstep!(::Val{:N}, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv,
    x::ArrayT, y::ArrayT)::Nothing where {Ti,Tv,ArrayT<:AbstractArray{Tv}}
    left, right = range
    @axpy(poi[left], x, y)
    for i = left+1:right
        x .= P * x
        @axpy(poi[i], x, y)
    end
    @scal(1/weight, y)
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

function cunifstep!(tr::Symbol, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, cpoi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::ArrayT, y::ArrayT, cy::ArrayT)::Nothing where {Ti,Tv,ArrayT<:AbstractArray{Tv}}
    _cunifstep!(Val(tr), P, poi, cpoi, range, weight, qv_weight, x, y, cy)
end

@origin (poi => left, cpoi => left) function _cunifstep!(::Val{:T}, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, cpoi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::ArrayT, y::ArrayT, cy::ArrayT)::Nothing where {Ti,Tv,ArrayT<:AbstractArray{Tv}}
    left, right = range
    Pdash = P'
    @axpy(poi[left], x, y)
    @axpy(cpoi[left], x, cy)
    for i = left+1:right
        x .= Pdash * x
        @axpy(poi[i], x, y)
        @axpy(cpoi[i], x, cy)
    end
    @scal(1/weight, y)
    @scal(1/qv_weight, cy)
    nothing
end

@origin (poi => left, cpoi => left) function _cunifstep!(::Val{:N}, P::AbstractMatrix{Tv},
    poi::Vector{Tv}, cpoi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::ArrayT, y::ArrayT, cy::ArrayT)::Nothing where {Ti,Tv,ArrayT<:AbstractArray{Tv}}
    left, right = range
    @axpy(poi[left], x, y)
    @axpy(cpoi[left], x, cy)
    for i = left+1:right
        x .= P * x
        @axpy(poi[i], x, y)
        @axpy(cpoi[i], x, cy)
    end
    @scal(1/weight, y)
    @scal(1/qv_weight, cy)
    nothing
end

