"""
Convolution for CTMC
"""

function _dger!(x::Array{Tv,1}, y::Array{Tv,1}, A::Matrix{Tv}) where {Tv}
    m, n = size(A)
    for j = 1:n
        for i = 1:m
            A[i,j] += x[i] * y[j]
        end
    end
    nothing
end

function _dger!(x::Array{Tv,1}, y::Array{Tv,1}, A::SparseCSR{Tv,Ti}) where {Ti,Tv}
    m, n = size(A)
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            A.val[z] += x[i] * y[j]
        end
    end
    nothing
end

function _dger!(x::Array{Tv,1}, y::Array{Tv,1}, A::SparseCSC{Tv,Ti}) where {Ti,Tv}
    m, n = size(A)
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            A.val[z] += x[i] * y[j]
        end
    end
    nothing
end

function _dger!(x::Array{Tv,1}, y::Array{Tv,1}, A::SparseCOO{Tv,Ti}) where {Ti,Tv}
    for z = 1:nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        A.val[z] += x[i] * y[j]
    end
    nothing
end

function _dger!(x::Array{Tv,1}, y::Array{Tv,1}, A::SparseMatrixCSC{Tv,Ti}) where {Ti,Tv}
    m, n = size(A)
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[z]
            A.nzval[z] += x[i] * y[j]
        end
    end
    nothing
end

"""
convunifstep!(trQ, trH, P, poi, range, weight, qv_weight, x, y, z, H)

! Description: convolution integral operation for matrix exp form;
!
!          |t
! trH(H) = | exp(trQ(Q)*s) * x * y' * exp(trQ(Q)*(t-s)) ds
!          |0
!
!        and
!
!        z = exp(trQ(Q)*t) * x
!
!        t is involved in the Poisson probability vector.
!        qv is an uniformed parameter
!        return value is z

Parameters:
- trQ: transpose operator
- trH: transpose operator
- P: The uniformed matrix
- poi: Poisson p.m.f.
- range: domain of Poisson p.m.f
- weight: The normalizing constant for Poisson p.m.f.
- qv_weight: The normalizing constant for Poisson c.c.d.f.
- x: Vector (inout). x may be changed after executing
- y: Vector (out). y should be zero before executing
- z: Vector (out). cy should be zero before executing
- H: Array (out). H should be zero before executing
Return value: nothing
"""

function convunifstep!(trQ::Symbol, trH::Symbol,
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    _convunifstep!(Val(trQ), Val(trH), P, poi, range, weight, qv_weight, x, y, z, H)
end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:N}, ::Val{:N},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    left, right = range
    Pdash = P'
    vc = Vector{Vector{Tv}}(undef, right - left + 1)
    vc[right] = zero(x)
    @axpy(poi[right], y, vc[right])
    for l = right-1:-1:left+1
        vc[l] = Pdash * vc[l+1]
        @axpy(poi[l], y, vc[l])
    end

    @axpy(poi[left], x, z)
    _dger!(x, vc[left+1], H)
    for l = left+1:right-1
        x .= P * x
        @axpy(poi[l], x, z)
        _dger!(x, vc[l+1], H)
    end
    @scal(1/weight, z)
    @scal(1/qv_weight, H)
    nothing
end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:T}, ::Val{:N},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    left, right = range
    Pdash = P'
    vc = Vector{Vector{Tv}}(undef, right - left + 1)
    vc[right] = zero(x)
    @axpy(poi[right], y, vc[right])
    for l = right-1:-1:left+1
        vc[l] = P * vc[l+1]
        @axpy(poi[l], y, vc[l])
    end

    @axpy(poi[left], x, z)
    _dger!(x, vc[left+1], H)
    for l = left+1:right-1
        x .= Pdash * x
        @axpy(poi[l], x, z)
        _dger!(x, vc[l+1], H)
    end
    @scal(1/weight, z)
    @scal(1/qv_weight, H)
    nothing
end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:N}, ::Val{:T},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    left, right = range
    Pdash = P'
    vc = Vector{Vector{Tv}}(undef, right - left + 1)
    vc[right] = zero(x)
    @axpy(poi[right], y, vc[right])
    for l = right-1:-1:left+1
        vc[l] = Pdash * vc[l+1]
        @axpy(poi[l], y, vc[l])
    end

    @axpy(poi[left], x, z)
    _dger!(vc[left+1], x, H)
    for l = left+1:right-1
        x .= P * x
        @axpy(poi[l], x, z)
        _dger!(vc[l+1], x, H)
    end
    @scal(1/weight, z)
    @scal(1/qv_weight, H)
    nothing
end

@origin (vc => left, poi => left) function _convunifstep!(::Val{:T}, ::Val{:T},
    P::AbstractMatrix{Tv},
    poi::Vector{Tv}, range::Tuple{Ti,Ti}, weight::Tv, qv_weight::Tv,
    x::Array{Tv,N}, y::Array{Tv,N}, z::Array{Tv,N},
    H::AbstractMatrix{Tv})::Nothing where {Ti,Tv,N}
    left, right = range
    Pdash = P'
    vc = Vector{Vector{Tv}}(undef, right - left + 1)
    vc[right] = zero(x)
    @axpy(poi[right], y, vc[right])
    for l = right-1:-1:left+1
        vc[l] = P * vc[l+1]
        @axpy(poi[l], y, vc[l])
    end

    @axpy(poi[left], x, z)
    _dger!(vc[left+1], x, H)
    for l = left+1:right-1
        x .= Pdash * x
        @axpy(poi[l], x, z)
        _dger!(vc[l+1], x, H)
    end
    @scal(1/weight, z)
    @scal(1/qv_weight, H)
    nothing
end
