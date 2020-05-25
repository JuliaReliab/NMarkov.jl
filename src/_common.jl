
export Trans, NoTrans
export @daxpy, @dscal, @ddot

"""
Trans
NoTrans

Types to represent the forward or backward computation for CTMC
"""

abstract type AbstractTranspose end
struct Trans <: AbstractTranspose end
struct NoTrans <: AbstractTranspose end

"""
@daxpy
@dascal

BLAS Level 1 functions.
"""

macro daxpy(a, x, y)
    expr = quote
        let u = $a
            for i in 1:length($x)
                $y[i] += u * $x[i]
            end
        end
    end
    esc(expr)
end

macro dscal(a, x)
    expr = quote
        let u = $a
            for i in 1:length($x)
                $x[i] *= u
            end
        end
    end
    esc(expr)
end

macro ddot(x, y)
    expr = quote
        s = 0
        for i in 1:length($x)
            s += $x[i] * $y[i]
        end
        s
    end
    esc(expr)
end
