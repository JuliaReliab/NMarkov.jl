"""
Special matrix
"""

"""
eye(n, ::Type{Tv} = Float64)::Matrix{Tv}
eye(A::AbstractMatrix, ::Type{Tv} = Float64)::Matrix{Tv}

Make an indentity matrix
"""
function eye(n, ::Type{Tv} = Float64)::Matrix{Tv} where {Tv}
    m = zeros(Tv, n,n)
    @inbounds for i = 1:n
        m[i,i] = Tv(1)
    end
    m
end

function eye(A::AbstractMatrix, ::Type{Tv} = Float64)::Matrix{Tv} where {Tv}
    eye(size(A)[1])
end
