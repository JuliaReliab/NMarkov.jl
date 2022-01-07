"""
Sensitivity vector with QR
"""

"""
stsen(Q::Matrix{Tv}, pis::Vector{Tv}, b::Vector{Tv})

Get a sensitivity vector for stationary vector of CTMC.

Parameters:
- Q: CTMC Kernal
- pis: A stationary vector, i.e., pis * Q = 0
- b: A vector. In the case of first derivative, b = pis * Qdash where Qdash is the first derivate of Q. It may change when we want to the high-order derivatives.
Return value:
A tuple of
- x: sensitivity vector
"""

function stsen(Q::Matrix{Tv}, pis::Vector{Tv}, b::Vector{Tv})::Vector{Tv} where Tv
    m, n = size(Q)
    @assert m == n
    @assert ctmcstcheck(Q, pis)
    qm, rm = qr(Q')
    xx = rm \ (- qm' * b)
    xx - sum(xx) * pis
end
