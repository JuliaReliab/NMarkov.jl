
export Trans, NoTrans
export @axpy, @scal, @dot
export itime

"""
Trans
NoTrans

Types to represent the forward or backward computation for CTMC
"""

abstract type AbstractTranspose end
struct Trans <: AbstractTranspose end
struct NoTrans <: AbstractTranspose end

"""
@axpy
@ascal
@dot

BLAS Level 1 functions.
"""

macro axpy(a, x, y)
    expr = quote
        let u = $a
            for i in 1:length($x)
                $y[i] += u * $x[i]
            end
        end
    end
    esc(expr)
end

macro scal(a, x)
    expr = quote
        let u = $a
            for i in 1:length($x)
                $x[i] *= u
            end
        end
    end
    esc(expr)
end

macro dot(x, y)
    expr = quote
        s = 0
        for i in 1:length($x)
            s += $x[i] * $y[i]
        end
        s
    end
    esc(expr)
end

"""
itime(t)

Get interval time from a given cumulative time vector t.
The first element is t[1]

Retuen value:
dt: interval time vector
maxt: maximum interval time
"""

function itime(t::AbstractVector{Tv}) where Tv
    dt = similar(t)
    prev = Tv(0)
    maxt = Tv(0)
    for i = eachindex(t)
        dt[i] = t[i] - prev
        prev = t[i]
        if dt[i] > maxt
            maxt = dt[i]
        end
    end
    return dt, maxt
end
