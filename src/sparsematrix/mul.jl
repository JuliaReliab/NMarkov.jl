"""
    Adjoint{T}

Lazy representation of the adjoint (transpose for real matrices) of a sparse matrix.

### Example
```julia
A = SparseCSR(...)
At = A'  # Creates Adjoint(A)
result = At * x  # Matrix-vector multiplication with transpose
```
"""
struct Adjoint{T <: AbstractSparseM}
    parent::T
end

"""
    adjoint(A::AbstractSparseM)

Return the adjoint (conjugate transpose) of sparse matrix A. For real matrices, this is equivalent to transpose.

### Returns
- `Adjoint{T}`: lazy representation of the adjoint
"""
function Base.adjoint(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    Adjoint(A)
end

function Base.adjoint(A::Adjoint{T}) where {T}
    A.parent
end

for mat in (:SparseCSR, :SparseCSC, :SparseCOO)
    @eval begin
        (Base.:*)(A::$mat{Tv,Ti}, x::Vector{Tv}) where {Tv,Ti} = LinearAlgebra.BLAS.gemv!('N', true, A, x, false, zeros(Tv, size(A,1)))
        (Base.:*)(At::Adjoint{$mat{Tv,Ti}}, x::Vector{Tv}) where {Tv,Ti} = LinearAlgebra.BLAS.gemv!('T', true, At.parent, x, false, zeros(Tv, size(At.parent,2)))
    end

    @eval begin
        (Base.:*)(A::$mat{Tv,Ti}, B::Matrix{Tv}) where {Tv,Ti} = LinearAlgebra.BLAS.gemm!('N', 'N', true, A, B, false, zeros(Tv, size(A,1), size(B,2)))
        (Base.:*)(A::$mat{Tv,Ti}, Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}) where {Tv,Ti} = LinearAlgebra.BLAS.gemm!('N', 'T', true, A, Bt.parent, false, zeros(Tv, size(A,1), size(Bt.parent,1)))
        (Base.:*)(At::Adjoint{$mat{Tv,Ti}}, B::Matrix{Tv}) where {Tv,Ti} = LinearAlgebra.BLAS.gemm!('T', 'N', true, At.parent, B, false, zeros(Tv, size(At.parent,2), size(B,2)))
        (Base.:*)(At::Adjoint{$mat{Tv,Ti}}, Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}) where {Tv,Ti} = LinearAlgebra.BLAS.gemm!('T', 'T', true, At.parent, Bt.parent, false, zeros(Tv, size(At.parent,2), size(Bt.parent,1)))

        (Base.:*)(B::Matrix{Tv}, A::$mat{Tv,Ti}) where {Tv,Ti} = LinearAlgebra.BLAS.gemm!('N', 'N', true, B, A, false, zeros(Tv, size(B,1), size(A,2)))
        (Base.:*)(Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::$mat{Tv,Ti}) where {Tv,Ti} = LinearAlgebra.BLAS.gemm!('T', 'N', true, Bt.parent, A, false, zeros(Tv, size(Bt.parent,2), size(A,2)))
        (Base.:*)(B::Matrix{Tv}, At::Adjoint{$mat{Tv,Ti}}) where {Tv,Ti} = LinearAlgebra.BLAS.gemm!('N', 'T', true, B, At.parent, false, zeros(Tv, size(B,1), size(At.parent,1)))
        (Base.:*)(Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, At::Adjoint{$mat{Tv,Ti}}) where {Tv,Ti} = LinearAlgebra.BLAS.gemm!('T', 'T', true, Bt.parent, At.parent, false, zeros(Tv, size(Bt.parent,2), size(At.parent,1)))
    end

    # Take the scalar as `Number` rather than `Union{Tv,Bool}`: with Tv appearing
    # in both arguments the method is a diagonal constraint, which leaves it
    # ambiguous against Base's `*(::AbstractArray, ::Number)` and
    # `/(::AbstractArray, ::Number)` (these types are AbstractArrays). With the
    # scalar unconstrained, ours is strictly the more specific method.
    @eval begin
        (Base.:*)(x::Number, A::$mat{Tv,Ti}) where {Tv,Ti} = LinearAlgebra.BLAS.scal!(convert(Tv, x), copy(A))
        (Base.:*)(A::$mat{Tv,Ti}, x::Number) where {Tv,Ti} = LinearAlgebra.BLAS.scal!(convert(Tv, x), copy(A))
        (Base.:/)(A::$mat{Tv,Ti}, x::Number) where {Tv,Ti} = LinearAlgebra.BLAS.scal!(Base.one(Tv)/convert(Tv, x), copy(A))

        (Base.:*)(x::Number, At::Adjoint{$mat{Tv,Ti}}) where {Tv,Ti} = LinearAlgebra.BLAS.scal!(convert(Tv, x), copy(At.parent))'
        (Base.:*)(At::Adjoint{$mat{Tv,Ti}}, x::Number) where {Tv,Ti} = LinearAlgebra.BLAS.scal!(convert(Tv, x), copy(At.parent))'
        (Base.:/)(At::Adjoint{$mat{Tv,Ti}}, x::Number) where {Tv,Ti} = LinearAlgebra.BLAS.scal!(Base.one(Tv)/convert(Tv, x), copy(At.parent))'
    end
end
