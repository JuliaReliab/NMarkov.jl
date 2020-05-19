"""
Uniformed Matrix for CTMC
"""

export unif

macro unif(Q, ufact)
    expr = quote
        qv = maximum(abs.(spdiag($Q))) * $ufact
        P = $Q / qv
        d = spdiag(P)
        d .+= 1
        (P, qv)
    end
    esc(expr)
end

"""
unif(Q::AbstractSparseM{Tv,Ti}, ufact::Tv = 1.01)
unif(Q::Matrix{Tv}, ufact::Tv = 1.01)

Get an uniformed transition probability matrix from a CTMC kernel.

   P = I + Q / qv
   qv = max(abs(diag(Q))) * ufact

Parameters:
- Q: CTMC Kernel
- ufact: uniformization factor
Return value:
A tuple of
- P: The uniformed transition probability matrix
- qv: The maximum event rate

"""

function unif(Q::SparseCSR{Tv,Ti}, ufact::Tv = 1.01) where {Tv, Ti}
    @unif(Q, ufact)
end

function unif(Q::SparseCSC{Tv,Ti}, ufact::Tv = 1.01) where {Tv, Ti}
    @unif(Q, ufact)
end

function unif(Q::SparseCOO{Tv,Ti}, ufact::Tv = 1.01) where {Tv, Ti}
    @unif(Q, ufact)
end

function unif(Q::Matrix{Tv}, ufact::Tv = 1.01) where {Tv, Ti}
    @unif(Q, ufact)
end
