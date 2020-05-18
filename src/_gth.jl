"""
GTH algorithm
"""

export gth!, gth

"""
gth!(Q::Matrix{Tv})
gth(Q::Matrix{Tv})

Compute the stationary vector for the CTMC with kernel Q with GTH algorithm.
Both of gth! and gth return a vector.
In gth!, the matrix Q is used as a workspace.
Note that Q does not have any absorbing state.
"""

function gth!(Q::Matrix{Tv})::Vector{Tv} where {Tv}
    m, n = size(Q)
    @assert m == n
    for l = n:-1:2
        tmp::Tv = 0
        for u = 1:l-1
            tmp += Q[l,u]
        end
        for j = 1:l-1
            for i = 1:l-1
                if i != j
                    Q[i,j] += Q[l,j] * Q[i,l] / tmp
                end
            end
        end
        for i = 1:l-1
            Q[i,l] /= tmp
        end
        for i = 1:l-1
            Q[l,i] = 0
        end
        Q[l,l] = -1
    end
    x = Vector{Tv}(undef, n)
    x[1] = 1.0
    for l = 2:n
        x[l] = 0.0
        for i = 1:l-1
            x[l] += x[i] * Q[i,l]
        end
    end
    x /= sum(x)
end

function gth(Q::Matrix{Tv}) where {Tv}
    gth!(copy(Q))
end

"""
gth!(Q::Matrix{Tv}, index::Vector{Ti})
gth(Q::Matrix{Tv}, index::Vector{Ti})

Compute the stationary vector for the CTMC with kernel Q with GTH algorithm.
Both of gth! and gth return a vector.
The CTMC states are permutated with an index vector.
In gth!, the matrix Q is used as a workspace.
Note that Q does not have any absorbing state.
"""

function gth!(Q::Matrix{Tv}, index::Vector{Ti})::Vector{Tv} where {Tv,Ti}
    m, n = size(Q)
    @assert m == n
    for l = n:-1:2
        tmp::Tv = 0
        for u = 1:l-1
            tmp += Q[index[l],index[u]]
        end
        for j = 1:l-1
            for i = 1:l-1
                if i != j
                    Q[index[i],index[j]] += Q[index[l],index[j]] * Q[index[i],index[l]] / tmp
                end
            end
        end
        for i = 1:l-1
            Q[index[i],index[l]] /= tmp
        end
        for i = 1:l-1
            Q[index[l],index[i]] = 0
        end
        Q[index[l],index[l]] = -1
    end
    x = Vector{Tv}(undef, n)
    x[index[1]] = 1.0
    for l = 2:n
        x[index[l]] = 0.0
        for i = 1:l-1
            x[index[l]] += x[index[i]] * Q[index[i],index[l]]
        end
    end
    x /= sum(x)
end

function gth(Q::Matrix{Tv}, index::Vector{Ti}) where {Tv, Ti}
    gth!(copy(Q), index)
end

