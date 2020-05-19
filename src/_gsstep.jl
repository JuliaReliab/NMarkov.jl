"""
GS step for CSC
"""

"""
gsstep!(x::Vector{Tv}, Q::SparseMatrix.SparseCSC{Tv,Ti}, b::Vector{Tv}; alpha::Tv=Tv(1), sigma::Tv=Tv(0), omega::Tv=Tv(1))::Nothing
gsstep!(x::Vector{Tv}, Q::SparseMatrix.SparseCSR{Tv,Ti}, b::Vector{Tv}; alpha::Tv=Tv(1), sigma::Tv=Tv(0), omega::Tv=Tv(1))::Nothing

GS (Gauss-Seidal) or SOR (Successive Over Relaxation) step for the following linear equation

    alpha * trans(A - sigma I) * x = b

    notrans:
        x := (D/omega + L)^(-1) (b/alpha - (U - D (1-omega)/omega - sigma I) * x)
    trans:
        x := (D/omega + tr(U))^(-1) (b/alpha - (tr(L) - D (1-omega)/omega - sigma I) * x)
        
    where
        A: square matrix
        x: vector (in; initial vector for the step, out; updated vector)
        b: constant vector

Note that notrans and trans are determined by the type of matrix Q.
If Q is SparseCSC, gstep! provides the step for trans.
If Q is SparseCSR, gsstep! provides the step for notrans.
"""

function gsstep!(x::Vector{Tv}, Q::SparseCSC{Tv,Ti}, b::Vector{Tv};
        alpha::Tv=Tv(1), sigma::Tv=Tv(0), omega::Tv=Tv(1))::Nothing where {Tv, Ti}
    m, n = size(Q)
    @assert m == n
    for j = 1:n
        tmpd::Tv = 0
        tmpx::Tv = b[j] / alpha
        for z = Q.colptr[j]:Q.colptr[j+1]-1
            i = Q.rowind[z]
            if i == j
                tmpd = Q.val[z]
                tmpx += sigma * x[i]
            else
                tmpx -= Q.val[z] * x[i]
            end
        end
        x[j] = omega / tmpd * tmpx + (1 - omega) * x[j]
    end
    nothing
end

function gsstep!(x::Vector{Tv}, Q::SparseCSR{Tv,Ti}, b::Vector{Tv};
        alpha::Tv=Tv(1), sigma::Tv=Tv(0), omega::Tv=Tv(1))::Nothing where {Tv, Ti}
    m, n = size(Q)
    @assert m == n
    for i = 1:m
        tmpd::Tv = 0
        tmpx::Tv = b[j] / alpha
        for z = Q.rowptr[i]:Q.rowptr[i+1]-1
            j = Q.colind[z]
            if i == j
                tmpd = Q.val[z]
                tmpx += sigma * x[j]
            else
                tmpx -= Q.val[z] * x[j]
            end
        end
        x[i] = omega / tmpd * tmpx + (1 - omega) * x[i]
    end
    nothing
end

