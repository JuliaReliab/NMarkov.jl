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
    # asarray keeps the shape: `x` and `r` may be matrices, and the matrix
    # methods of _tran are only reachable if that shape survives conversion.
    _tran(Q, asarray(Tv, x), asarray(Tv, r), asvector(Tv, checktimes(ts)),
        Val{forward}, convert(Tv, ufact), convert(Tv, eps), rmax)
end


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
                matmul!(:T, one(Tv), P, y0, false, tmpv)
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
                matmul!(:N, one(Tv), P, y0, false, tmpv)
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

# x is a single initial vector and r holds several reward vectors as columns,
# so each time point yields one reward per column of r.

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
    result = Vector{Vector{Tv}}(undef, length(dt))
    cresult = Vector{Vector{Tv}}(undef, length(dt))

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
                matmul!(:T, one(Tv), P, y0, false, tmpv)
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

@inbounds function _tran(Q::AbstractMatrix{Tv}, x::Array{Tv,1}, r::ArrayT2, ts::AbstractVector{Tv},
    ::Type{Val{:N}}, ufact::Tv, eps::Tv, rmax) where {Tv,ArrayT2<:AbstractArray{Tv}}
    m, n = size(Q)
    @assert m == n
    dt, maxt = itime(ts)
    P, qv = unif(Q, ufact)
    right = rightbound(qv*maxt, eps) + 1
    @assert right <= rmax "Time interval is too large. t or rmax should be changed: right = $right (rmax: $rmax)."
    prob = Vector{Tv}(undef, right+1)
    cprob = Vector{Tv}(undef, right+1)
    result = Vector{Vector{Tv}}(undef, length(dt))
    cresult = Vector{Vector{Tv}}(undef, length(dt))

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
                matmul!(:N, one(Tv), P, y0, false, tmpv)
                @. y0 = tmpv
                axpy!(prob[i]/weight, y0, y1)
                axpy!(cprob[i]/(qv*weight), y0, cy)
            end
        end
        result[k] = y1' * x
        cresult[k] = cy' * x
        @. y0 = y1
    end
    result, cresult, y1, cy
end

### mat * mat

# x holds initial vectors as rows (a-by-n, matching the "row vector" wording of
# the docstring) and r holds reward vectors as columns (n-by-b), so each time
# point yields an a-by-b reward matrix.

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
    # x's rows are the initial vectors, so propagate its transpose: y0 is n-by-a
    # and matmul!(:T, ...) computes P' * y0.
    xdash = collect(x')
    y0 = copy(xdash)
    y1 = similar(xdash)
    cy = zero(xdash)
    tmpv = similar(xdash)

    # r may be a matrix (one reward per column) or a single reward vector, so
    # the per-time result is a matrix or a vector accordingly.
    RT = Base.promote_op(*, typeof(y1'), ArrayT2)
    result = Vector{RT}(undef, length(dt))
    cresult = Vector{RT}(undef, length(dt))
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps) + 1
        weight = cpoipmf!(qv*dt[k], prob, cprob; left=0, right=right)

        @. y1 = zero(Tv)
        @origin (prob=>0, cprob=>0) begin
            axpy!(prob[0]/weight, y0, y1)
            axpy!(cprob[0]/(qv*weight), y0, cy)
            for i = 1:right
                matmul!(:T, one(Tv), P, y0, false, tmpv)
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
    RT = Base.promote_op(*, ArrayT1, ArrayT2)
    result = Vector{RT}(undef, length(dt))
    cresult = Vector{RT}(undef, length(dt))

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
                matmul!(:N, one(Tv), P, y0, false, tmpv)
                @. y0 = tmpv
                axpy!(prob[i]/weight, y0, y1)
                axpy!(cprob[i]/(qv*weight), y0, cy)
            end
        end
        # x is a-by-n (rows are initial vectors), y1 is n-by-b.
        result[k] = x * y1
        cresult[k] = x * cy
        @. y0 = y1
    end
    result, cresult, y1, cy
end
