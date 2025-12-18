"""
Transient analysis for CTMC

This module provides functions for computing instantaneous and cumulative rewards
for continuous-time Markov chains over a time series.

Main functions:
- `tran()`: Compute transient rewards with matrix exponential computation
- `mexp()`, `mexpc()`: Matrix exponential related functions (see mexp.jl)
- Type-flexible wrappers for mixed numeric types
"""

"""
    tran(Q, x, r, ts; forward = :T, ufact = 1.01, eps = 1.0e-8, rmax = 500)

Compute instantaneous and cumulative rewards for a CTMC over a time series.

This function computes rewards over multiple time points efficiently by using
the uniformization method with Poisson probability mass functions.

### Arguments
- `Q::AbstractMatrix`: Infinitesimal generator matrix (n×n)
- `x::AbstractArray`: Initial vector or row vector (any numeric type, auto-converted)
- `r::AbstractArray`: Reward vector or reward matrix (any numeric type, auto-converted)
- `ts::AbstractVector`: Time points where rewards are computed
- `forward::Symbol`: Direction of computation (`:T` for forward, `:N` for backward; default: `:T`)
- `ufact::Real`: Uniformization factor (default: 1.01)
- `eps::Real`: Tolerance for Poisson truncation (default: 1.0e-8)
- `rmax::Int`: Maximum Poisson truncation (default: 500)

### Returns
A tuple of four elements:
- `inst_reward`: Instantaneous reward at each time point
- `cum_reward`: Cumulative reward up to each time point
- `x_final`: Final state distribution (when `forward=:T`)
- `r_final`: Final reward distribution (when `forward=:N`)

### Reward Computations
**Forward direction** (`forward=:T`, default):
- Instantaneous: ``r_k = x \\cdot e^{Q t_k} \\cdot r^T`` 
- Cumulative: ``\\int_0^{t_k} x \\cdot e^{Q u} \\cdot r^T \\, du``
- Returns final probability vector

**Backward direction** (`forward=:N`):
- Instantaneous: ``r_k = x^T \\cdot e^{Q t_k} \\cdot r``
- Cumulative: ``\\int_0^{t_k} x^T \\cdot e^{Q u} \\cdot r \\, du``
- Returns final reward vector

### Supported Input Types
- `x` and `r` can be vectors or matrices
- `ts` can be any numeric vector (auto-converted to element type of Q)
- Automatic type conversion ensures flexibility

### Algorithm
- Uniformization with Poisson probability computations
- Efficient time-series computation avoiding repeated matrix exponentials
- Numerical accuracy controlled by `eps` and `rmax` parameters

### Example
```julia
Q = [-2.0 2.0; 1.0 -1.0]
x = [1.0, 0.0]  # Initial state
r = [1.0, 2.0]  # Rewards for states
ts = [0.5, 1.0, 2.0]  # Time points

inst, cum, x_final, r_final = tran(Q, x, r, ts)
```

### Performance Notes
- Time-series computation is more efficient than calling `mexp` separately
- Tolerance `eps` affects computation time: smaller ε requires larger Poisson truncation
- Adjust `ufact` if Poisson truncation becomes too large

### Errors
- Throws error if `rmax` is exceeded (time interval too large)
- Requires Q to be square
"""
function tran(Q::AbstractMatrix{Tv}, x::AbstractArray, r::AbstractArray, ts::AbstractVector;
    forward::Symbol=:T, ufact::Real=1.01, eps::Real=1.0e-8, rmax=500) where {Tv}
    if !(eltype(x) <: Tv && eltype(r) <: Tv && eltype(ts) <: Tv)
        x_float = vec(convert(Array{Tv}, x))
        r_float = vec(convert(Array{Tv}, r))
        ts_float = convert(Vector{Tv}, ts)
        ufact_float = convert(Tv, ufact)
        eps_float = convert(Tv, eps)
        return tran(Q, x_float, r_float, ts_float; forward=forward, ufact=ufact_float, eps=eps_float, rmax=rmax)
    end
    error("Method not found for these exact types")
end

function tran(Q::AbstractMatrix{Tv}, x::ArrayT1, r::ArrayT2, ts::AbstractVector{Tv};
    forward::Symbol=:T, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500) where {Tv,ArrayT1<:AbstractArray{Tv},ArrayT2<:AbstractArray{Tv}}
    _tran(Q, x, r, ts, Val{forward}, ufact, eps, rmax)
end

# function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::Array{Tv,1}, ts::AbstractVector{Tv},
#     ::Type{Val{:T}}, ufact::Tv, eps::Tv, rmax) where Tv
#     m, n = size(Q)
#     @assert m == n
#     dt, maxt = itime(ts)
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps) + 1
#     @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
#     prob = Vector{Tv}(undef, right+1)
#     cprob = Vector{Tv}(undef, right+1)
#     result = Vector{Tv}(undef, length(dt))
#     cresult = Vector{Tv}(undef, length(dt))
#     y0 = copy(x)
#     y1 = similar(x)
#     cy = zero(x)
#     tmp = similar(x)
#     @inbounds for k = eachindex(dt)
#         right = rightbound(qv*dt[k], eps) + 1
#         weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
#         y1 .= Tv(0)
#         tmp .= Tv(0)
#         cunifstep!(:T, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
#         cy .+= tmp
#         result[k] = @dot(y1, r)
#         cresult[k] = @dot(cy, r)
#         y0 .= y1
#     end
#     return result, cresult, y1, cy
# end

### vec * vec

@inbounds function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::Array{Tv,1}, ts::AbstractVector{Tv},
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
    tmpv = similar(x)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)

        @. y1 = zero(Tv)
        @origin (prob=>0, cprob=>0) begin
            axpy!(prob[0]/weight, y0, y1)
            axpy!(cprob[0]/(qv*weight), y0, cy)
            for i = 1:right
                matmul!(:T, 1.0, P, y0, false, tmpv)
                @. y0 = tmpv
                axpy!(prob[i]/weight, y0, y1)
                axpy!(cprob[i]/(qv*weight), y0, cy)
            end
        end
        result[k] = @dot(y1, r)
        cresult[k] = @dot(cy, r)
        @. y0 = y1
    end
    result, cresult, y1, cy
end

# function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::Array{Tv,1}, ts::AbstractVector{Tv},
#     ::Type{Val{:N}}, ufact::Tv, eps::Tv, rmax) where Tv
#     m, n = size(Q)
#     @assert m == n
#     dt, maxt = itime(ts)
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps) + 1
#     @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
#     prob = Vector{Tv}(undef, right+1)
#     cprob = Vector{Tv}(undef, right+1)
#     result = Vector{Tv}(undef, length(dt))
#     cresult = Vector{Tv}(undef, length(dt))
#     y0 = copy(r)
#     y1 = similar(r)
#     cy = zero(r)
#     tmp = similar(r)
#     @inbounds for k = eachindex(dt)
#         right = rightbound(qv*dt[k], eps) + 1
#         weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
#         y1 .= Tv(0)
#         tmp .= Tv(0)
#         cunifstep!(:N, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
#         cy .+= tmp
#         result[k] = @dot(x, y1)
#         cresult[k] = @dot(x, cy)
#         y0 .= y1
#     end
#     return result, cresult, y1, cy
# end

@inbounds function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::Array{Tv,1}, ts::AbstractVector{Tv},
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
    tmpv = similar(r)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)

        @. y1 = zero(Tv)
        @origin (prob=>0, cprob=>0) begin
            axpy!(prob[0]/weight, y0, y1)
            axpy!(cprob[0]/(qv*weight), y0, cy)
            for i = 1:right
                matmul!(:N, 1.0, P, y0, false, tmpv)
                @. y0 = tmpv
                axpy!(prob[i]/weight, y0, y1)
                axpy!(cprob[i]/(qv*weight), y0, cy)
            end
        end
        result[k] = @dot(x, y1)
        cresult[k] = @dot(x, cy)
        @. y0 = y1
    end
    result, cresult, y1, cy
end

### vec * mat

# function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::ArrayT2, ts::AbstractVector{Tv},
#     ::Type{Val{:T}}, ufact::Tv, eps::Tv, rmax) where {Tv,ArrayT2<:AbstractArray{Tv}}
#     m, n = size(Q)
#     @assert m == n
#     dt, maxt = itime(ts)
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps) + 1
#     @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
#     prob = Vector{Tv}(undef, right+1)
#     cprob = Vector{Tv}(undef, right+1)
#     result = Vector{Any}(undef, length(dt))
#     cresult = Vector{Any}(undef, length(dt))
#     y0 = copy(x)
#     y1 = similar(x)
#     cy = zero(x)
#     tmp = similar(x)
#     @inbounds for k = eachindex(dt)
#         right = rightbound(qv*dt[k], eps) + 1
#         weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
#         y1 .= Tv(0)
#         tmp .= Tv(0)
#         cunifstep!(:T, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
#         cy .+= tmp
#         result[k] = r' * y1
#         cresult[k] = r' * cy
#         y0 .= y1
#     end
#     return result, cresult, y1, cy
# end

@inbounds function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::ArrayT2, ts::AbstractVector{Tv},
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
    tmpv = similar(x)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)

        @. y1 = zero(Tv)
        @origin (prob=>0, cprob=>0) begin
            axpy!(prob[0]/weight, y0, y1)
            axpy!(cprob[0]/(qv*weight), y0, cy)
            for i = 1:right
                matmul!(:T, 1.0, P, y0, false, tmpv)
                @. y0 = tmpv
                axpy!(prob[i]/weight, y0, y1)
                axpy!(cprob[i]/(qv*weight), y0, cy)
            end
        end
        result[k] = r' * y1
        cresult[k] = r' * cy
        @. y0 = y1
    end
    result, cresult, y1, cy
end

### mat * mat

# function _tran(Q::AbstractMatrix{Tv}, x::ArrayT1, r::ArrayT2, ts::AbstractVector{Tv},
#     ::Type{Val{:T}}, ufact::Tv, eps::Tv, rmax) where {Tv,ArrayT1<:AbstractArray{Tv},ArrayT2<:AbstractArray{Tv}}
#     m, n = size(Q)
#     @assert m == n
#     dt, maxt = itime(ts)
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps) + 1
#     @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
#     prob = Vector{Tv}(undef, right+1)
#     cprob = Vector{Tv}(undef, right+1)
#     xdash = x'
#     result = Vector{Any}(undef, length(dt))
#     cresult = Vector{Any}(undef, length(dt))
#     y0 = copy(xdash)
#     y1 = similar(xdash)
#     cy = zero(xdash)
#     tmp = similar(xdash)
#     @inbounds for k = eachindex(dt)
#         right = rightbound(qv*dt[k], eps) + 1
#         weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
#         y1 .= Tv(0)
#         tmp .= Tv(0)
#         cunifstep!(:T, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
#         cy .+= tmp
#         result[k] = y1' * r
#         cresult[k] = cy' * r
#         y0 .= y1
#     end
#     return result, cresult, y1', cy'
# end

@inbounds function _tran(Q::AbstractMatrix{Tv}, x::ArrayT1, r::ArrayT2, ts::AbstractVector{Tv},
    ::Type{Val{:T}}, ufact::Tv, eps::Tv, rmax) where {Tv,ArrayT1<:AbstractArray{Tv},ArrayT2<:AbstractArray{Tv}}
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

    xdash = x'
    y0 = copy(xdash)
    y1 = similar(xdash)
    cy = zero(xdash)
    tmpv = similar(xdash)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)

        @. y1 = zero(Tv)
        @origin (prob=>0, cprob=>0) begin
            axpy!(prob[0]/weight, y0, y1)
            axpy!(cprob[0]/(qv*weight), y0, cy)
            for i = 1:right
                matmul!(:T, 1.0, P, y0, false, tmpv)
                @. y0 = tmpv
                axpy!(prob[i]/weight, y0, y1)
                axpy!(cprob[i]/(qv*weight), y0, cy)
            end
        end
        result[k] = y1' * r
        cresult[k] = cy' * r
        @. y0 = y1
    end
    result, cresult, y1, cy
end

# function _tran(Q::AbstractMatrix{Tv}, x::ArrayT1, r::ArrayT2, ts::AbstractVector{Tv},
#     ::Type{Val{:N}}, ufact::Tv, eps::Tv, rmax) where {Tv,ArrayT1<:AbstractArray{Tv},ArrayT2<:AbstractArray{Tv}}
#     m, n = size(Q)
#     @assert m == n
#     dt, maxt = itime(ts)
#     P, qv = unif(Q, ufact)
#     right = rightbound(qv*maxt, eps) + 1
#     @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
#     prob = Vector{Tv}(undef, right+1)
#     cprob = Vector{Tv}(undef, right+1)
#     result = Vector{Any}(undef, length(dt))
#     cresult = Vector{Any}(undef, length(dt))
#     y0 = copy(r)
#     y1 = similar(r)
#     cy = zero(r)
#     tmp = similar(r)
#     @inbounds for k = eachindex(dt)
#         right = rightbound(qv*dt[k], eps) + 1
#         weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)
#         y1 .= Tv(0)
#         tmp .= Tv(0)
#         cunifstep!(:N, P, prob, cprob, (0, right), weight, qv*weight, y0, y1, tmp)
#         cy .+= tmp
#         result[k] = x * y1
#         cresult[k] = x * cy
#         y0 .= y1
#     end
#     return result, cresult, y1, cy
# end

@inbounds function _tran(Q::AbstractMatrix{Tv}, x::ArrayT1, r::ArrayT2, ts::AbstractVector{Tv},
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
    tmpv = similar(r)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)

        @. y1 = zero(Tv)
        @origin (prob=>0, cprob=>0) begin
            axpy!(prob[0]/weight, y0, y1)
            axpy!(cprob[0]/(qv*weight), y0, cy)
            for i = 1:right
                matmul!(:N, 1.0, P, y0, false, tmpv)
                @. y0 = tmpv
                axpy!(prob[i]/weight, y0, y1)
                axpy!(cprob[i]/(qv*weight), y0, cy)
            end
        end
        result[k] = x * y1
        cresult[k] = x * cy
        @. y0 = y1
    end
    result, cresult, y1, cy
end
