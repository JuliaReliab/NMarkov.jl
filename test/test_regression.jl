## Regression tests for bugs found in the 0.4.0 code review.
## Each testset name is keyed to the finding it pins down.

using LinearAlgebra: I, exp
using SparseArrays: sparse

@testset "regression" begin

## --- A: unif must produce a stochastic P even when a diagonal entry is
##        structurally absent from the sparse pattern (absorbing state).
@testset "unif with absorbing state" begin
    Q = [
        -3.0 3.0 0.0;
        0.0 -5.0 5.0;
        0.0 0.0 0.0
    ]
    Pd, qvd = unif(Q)
    @test all(sum(Pd, dims=2) .≈ 1.0)
    @test Pd ≈ Matrix(1.0I, 3, 3) + Q / qvd

    for S in (sparse(Q), SparseCSR(Q), SparseCSC(Q), SparseCOO(Q))
        P, qv = unif(S)
        @test qv ≈ qvd
        @test Matrix(P) ≈ Pd
        @test all(sum(Matrix(P), dims=2) .≈ 1.0)
    end
end

## Uniformization of an absorbing chain must give the right transient answer.
@testset "mexp with absorbing state" begin
    Q = [
        -3.0 3.0 0.0;
        0.0 -5.0 5.0;
        0.0 0.0 0.0
    ]
    x = [1.0, 0.0, 0.0]
    t = 0.7
    expected = exp(Q * t)' * x   # transpose=:T
    for S in (Q, sparse(Q), SparseCSR(Q), SparseCSC(Q), SparseCOO(Q))
        @test mexp(S, x, t, transpose=:T) ≈ expected
    end
end

## --- B: z and H are truncations of two different series and stop at different
##        points. z is the matrix exponential (paper eq.(apppp)), truncated at
##        U = rightbound(...) = `right`, so it must sum poi[0..right]. H is the
##        convolution integral (paper eq.(24)), whose U is right-1 because
##        beta_U needs pi_{U+1}. The old code applied H's range to z as well, so
##        z summed poi[0..right-1] while `weight` covered poi[0..right].
##
##        The generator below has row sums 0, so unif gives a stochastic P and
##        the mass-conservation check is meaningful. (A matrix without that
##        property makes P non-stochastic and the whole comparison vacuous.)
@testset "convunifstep! z summation range" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        2.0 1.0 -3.0
    ]
    n = 3
    x = [0.3, 0.5, 0.2]          # a probability vector
    y = [1.0, 0.0, 0.0]
    tau = 1.0
    P, qv = unif(Q)
    @test all(sum(P, dims=2) .≈ 1.0)

    for eps in (1.0e-2, 1.0e-4, 1.0e-6)
        right = rightbound(qv * tau, eps)
        weight, poi = poipmf(qv * tau, right, left=0)

        for trQ in (:N, :T)
            z = zeros(n)
            H = zero(Q)
            convunifstep!(trQ, :N, P, poi, (0, right), weight, qv * weight,
                copy(x), y, z, H)

            # The summation range itself: z must be the normalised partial sum
            # over the FULL domain [0, right]. Stopping at right-1 fails here.
            ref = zeros(n)
            pk = copy(x)
            for k = 0:right
                ref .+= poi[k+1] .* pk
                pk = trQ === :N ? P * pk : P' * pk
            end
            ref ./= weight
            @test z ≈ ref rtol = 1.0e-14

            # Accuracy against the exact matrix exponential.
            exact = (trQ === :N ? exp(Q * tau) : exp(Q' * tau)) * x
            @test maximum(abs.(z - exact)) < eps
        end

        # Mass conservation: with a stochastic P, a probability vector x and
        # trQ=:T, sum(z) is exactly 1 whatever eps is, because the weights are
        # normalised over the same range they are summed over. The old code
        # returned 1 - poi[right]/weight, i.e. it leaked O(eps) of the mass.
        z = zeros(n)
        H = zero(Q)
        convunifstep!(:T, :N, P, poi, (0, right), weight, qv * weight,
            copy(x), y, z, H)
        @test sum(z) ≈ 1.0 atol = 1.0e-12
    end
end

## --- C/D: unsorted or negative time points must be rejected, not silently
##          turned into NaN or into a reordered answer.
@testset "time vector validation" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        0.0 1.0 -1.0
    ]
    x = [1.0, 0.0, 0.0]
    r = [1.0, 2.0, 3.0]

    @test_throws ArgumentError mexp(Q, x, [1.0, 0.5])
    @test_throws ArgumentError mexpc(Q, x, [1.0, 0.5])
    @test_throws ArgumentError tran(Q, x, r, [1.0, 0.5])
    @test_throws ArgumentError mexp(Q, x, [-1.0, 0.5])
    @test_throws ArgumentError mexp(Q, x, -1.0)

    # sorted input still works and matches the single-time results
    ts = [0.5, 1.0, 2.0]
    res = mexp(Q, x, ts)
    for (k, t) in enumerate(ts)
        @test res[k] ≈ mexp(Q, x, t)
    end
end

## --- E: stguess must not produce NaN for a matrix with a zero diagonal.
@testset "stguess with zero diagonal" begin
    P = [0.0 1.0; 1.0 0.0]
    g = stguess(P)
    @test all(isfinite.(g))
    @test sum(g) ≈ 1.0

    pi, conv, iter, rerror = stpower(P)
    @test all(isfinite.(pi))
    @test pi ≈ [0.5, 0.5]
end

## gth must reject an absorbing chain instead of returning NaN.
@testset "gth with absorbing state" begin
    Q = [
        -3.0 3.0 0.0;
        0.0 -5.0 5.0;
        0.0 0.0 0.0
    ]
    @test_throws ArgumentError gth(Q)
    @test_throws ArgumentError gth(Q, [1, 2, 3])
end

## --- F: no type piracy on Base.iszero.
@testset "no type piracy on iszero" begin
    # `m.module` rather than `parentmodule(m)`: the latter has no Method method
    # before Julia 1.10, and this package supports 1.6.
    for m in methods(iszero, (Float64,))
        @test m.module !== NMarkov.SparseMatrix
    end
end

## --- G: the CSR Gauss-Seidel step must run at all (it used to read an
##       undefined variable) and must match a plain dense sweep.
##       The CSR sweep solves Q*x = b by rows; the CSC sweep solves x'Q = b'
##       by columns. They are different operations, so each gets its own
##       reference.
@testset "gsstep! matches a dense sweep" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        2.0 1.0 -3.0
    ]
    b = [0.1, 0.2, 0.3]
    x0 = [0.3, 0.3, 0.4]

    byrow = copy(x0)
    for i = 1:3
        s = b[i]
        for j = 1:3
            j != i && (s -= Q[i, j] * byrow[j])
        end
        byrow[i] = s / Q[i, i]
    end

    bycol = copy(x0)
    for j = 1:3
        s = b[j]
        for i = 1:3
            i != j && (s -= Q[i, j] * bycol[i])
        end
        bycol[j] = s / Q[j, j]
    end

    xr = copy(x0)
    NMarkov.gsstep!(xr, SparseCSR(Q), copy(b))
    @test xr ≈ byrow

    xc = copy(x0)
    NMarkov.gsstep!(xc, SparseCSC(Q), copy(b))
    @test xc ≈ bycol
end

## stgs must reach the same stationary vector regardless of storage format.
@testset "stgs across sparse formats" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        2.0 1.0 -3.0
    ]
    expected = gth(Q)
    for A in (sparse(Q), SparseCSC(Q))
        pi, conv, iter, rerror = stgs(A)
        @test conv
        @test pi ≈ expected
    end
end

## --- H/I/J: type flexibility must actually work, for both t and x, and for
##            element types other than Float64.
@testset "mexp/mexpc type flexibility" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        0.0 1.0 -1.0
    ]
    x = [1.0, 0.0, 0.0]
    ref_mexp = mexp(Q, x, 1.0)
    ref_mexpc = mexpc(Q, x, 1.0)

    # integer / small-float times and integer initial vectors
    @test mexp(Q, [1, 0, 0], 1) ≈ ref_mexp
    @test mexp(Q, [1, 0, 0], 1.0) ≈ ref_mexp
    @test mexp(Q, x, Int32(1)) ≈ ref_mexp
    @test mexp(Q, [1, 0, 0], 1.0f0) ≈ ref_mexp
    @test all(mexpc(Q, [1, 0, 0], 1) .≈ ref_mexpc)
    @test all(mexpc(Q, x, 1.0f0) .≈ ref_mexpc)
    @test mexpc(Q, [1, 0, 0], [1.0]) == mexpc(Q, x, [1.0])
    @test mexp(Q, [1, 0, 0], [1.0])[1] ≈ ref_mexp

    # Float32 end to end: the wrapper must not recurse into itself
    Q32 = Float32.(Q)
    x32 = Float32.(x)
    r32 = mexp(Q32, x32, 1.0f0)
    @test eltype(r32) === Float32
    @test r32 ≈ Float32.(ref_mexp) rtol = 1.0e-4
    @test eltype(mexpc(Q32, x32, 1.0f0)[1]) === Float32
    @test eltype(mexp(Q32, x32, [1.0f0])[1]) === Float32
end

@testset "tran type flexibility" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        0.0 1.0 -1.0
    ]
    x = [1.0, 0.0, 0.0]
    r = [1.0, 2.0, 3.0]
    ts = [0.5, 1.0]
    ref = tran(Q, x, r, ts)

    got = tran(Q, [1, 0, 0], [1, 2, 3], [0.5, 1.0])
    @test got[1] ≈ ref[1]
    @test got[2] ≈ ref[2]

    Q32, x32, r32, ts32 = Float32.(Q), Float32.(x), Float32.(r), Float32.(ts)
    got32 = tran(Q32, x32, r32, ts32)
    @test eltype(got32[1]) === Float32
    @test got32[1] ≈ Float32.(ref[1]) rtol = 1.0e-4
end

## --- L: the matrix methods of tran must agree with the vector methods, and
##       the type-conversion wrapper must not flatten a matrix argument.
##       x holds initial vectors as ROWS (a x n); r holds reward vectors as
##       COLUMNS (n x b).
@testset "tran with matrix arguments" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        0.0 1.0 -1.0
    ]
    ts = [0.5, 1.0]
    xm = [1.0 0.0 0.0; 0.0 0.5 0.5]    # a x n, two initial vectors as rows
    rm = [1.0 0.0; 2.0 1.0; 3.0 0.0]   # n x b, two reward vectors as columns

    for fwd in (:T, :N)
        inst, cum, = tran(Q, xm, rm, ts, forward=fwd)
        @test size(inst[1]) == (2, 2)
        for a = 1:2, b = 1:2
            vi, vc, = tran(Q, xm[a, :], rm[:, b], ts, forward=fwd)
            for k = 1:length(ts)
                @test inst[k][a, b] ≈ vi[k]
                @test cum[k][a, b] ≈ vc[k]
            end
        end
    end

    # a single initial vector with several reward vectors, both directions
    for fwd in (:T, :N)
        inst, cum, = tran(Q, xm[1, :], rm, ts, forward=fwd)
        @test length(inst[1]) == 2
        for b = 1:2
            vi, vc, = tran(Q, xm[1, :], rm[:, b], ts, forward=fwd)
            for k = 1:length(ts)
                @test inst[k][b] ≈ vi[k]
                @test cum[k][b] ≈ vc[k]
            end
        end
    end

    # the type-conversion wrapper must not flatten matrix arguments
    inst2, = tran(Q, [1 0 0; 0 0 1], rm, ts)
    @test size(inst2[1]) == (2, 2)
end

## --- M: eye must honour the requested element type.
@testset "eye element type" begin
    A = zeros(Float32, 3, 3)
    @test eltype(NMarkov.eye(3, Float32)) === Float32
    @test eltype(NMarkov.eye(A, Float32)) === Float32
    @test NMarkov.eye(A, Float32) == Matrix{Float32}(I, 3, 3)
end

## --- P: stsengs must not lose the element type through its default x0.
@testset "stsengs element type" begin
    Qd = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        2.0 1.0 -3.0
    ]
    pis = Float32.(gth(Qd))
    b = Float32[0.1, -0.05, -0.05]
    # the default x0 comes from stsenguess, which must follow Q's element type
    s, conv, = stsengs(SparseCSC(Float32.(Qd)), pis, b)
    @test eltype(s) === Float32
end

## --- N/O: sparse types must work for element types other than Float64 and
##          must honour the AbstractMatrix interface.
@testset "sparse matrix genericity" begin
    A = [
        1.0 0.0 2.0;
        0.0 3.0 0.0;
        4.0 0.0 5.0
    ]
    for S in (SparseCSR(A), SparseCSC(A), SparseCOO(A))
        @test size(S) == (3, 3)
        @test length(S) == 9              # AbstractMatrix contract, not nnz
        @test S[1, 3] == 2.0
        @test S[1, 2] == 0.0
        @test Matrix(S) == A
    end

    # element types other than Float64
    A32 = Float32.(A)
    x32 = Float32[1.0, 2.0, 3.0]
    for S in (SparseCSR(A32), SparseCSC(A32), SparseCOO(A32))
        @test S * x32 ≈ A32 * x32
        @test Matrix(S) == A32
    end

    # dense -> sparse -> dense must be exact even above Float64 precision
    Ab = BigFloat[BigFloat(1)/3 0; 0 2]
    for S in (SparseCSR(Ab), SparseCSC(Ab), SparseCOO(Ab))
        @test Matrix(S)[1, 1] == Ab[1, 1]
    end
end

## Scalar `*` and `/` on the sparse types must not be ambiguous with Base's
## `*(::AbstractArray, ::Number)` / `/(::AbstractArray, ::Number)`, and must
## preserve the sparsity pattern (including structural zeros).
@testset "sparse scalar multiply and divide" begin
    A = [
        1.0 0.0 2.0;
        0.0 3.0 0.0;
        4.0 0.0 5.0
    ]
    for S in (SparseCSR(A), SparseCSC(A), SparseCOO(A))
        n0 = length(S.val)
        @test Matrix(S / 2.0) ≈ A / 2.0
        @test Matrix(S * 2.0) ≈ A * 2.0
        @test Matrix(2.0 * S) ≈ 2.0 * A
        @test Matrix(S / 2) ≈ A / 2          # Int scalar
        @test Matrix(S * true) ≈ A
        @test length((S / 2.0).val) == n0    # pattern preserved
        @test Matrix(S) == A                 # the original is untouched
        # The adjoint path scales the parent and re-wraps it. (There is no
        # Matrix(::Adjoint{<:AbstractSparseM}) method, so go through the parent.)
        @test Matrix((S' / 2.0).parent) ≈ A / 2.0
        @test Matrix((2.0 * S').parent) ≈ 2.0 * A
    end

    # a stored structural zero must survive scaling: `unif` depends on it
    val = [-3.0, 3.0, -5.0, 5.0, 0.0]
    Qz = SparseCSC(3, 3, val, [1, 2, 4, 6], [1, 1, 2, 2, 3])
    @test length((Qz / 2.0).val) == 5
    @test spdiag(Qz / 2.0)[3] == 0.0
end

## Int32 index types must work through spdiag and stguess.
@testset "Int32 index types" begin
    Q = sparse(Int32[1, 1, 2, 2], Int32[1, 2, 1, 2], [-1.0, 1.0, 1.0, -1.0])
    @test eltype(Q.rowval) === Int32
    d = spdiag(Q)
    @test d[1] ≈ -1.0
    g = stguess(Q)
    @test all(isfinite.(g))
    @test sum(g) ≈ 1.0
end

## --- @origin bounds: poipmf!/cpoipmf! seed the recurrence at prob[mode] with
##     mode = floor(lambda). Under `@origin (prob => left)` that is the physical
##     index mode-left+1, so a mode outside [left,right] writes past the buffer
##     inside an @inbounds block. These four cases used to segfault or silently
##     corrupt the heap.
@testset "poipmf domain validation" begin
    # right below the mode
    @test_throws ArgumentError poipmf(5.05, 0)
    @test_throws ArgumentError cpoipmf(5.05, 2)
    # left above the mode (this one used to return plausible numbers while
    # writing five Float64s in front of the array)
    @test_throws ArgumentError poipmf(5.05, 20, left=10)
    # negative mean gives mode = -1
    @test_throws ArgumentError poipmf(-1.0, 5)
    @test_throws ArgumentError cpoipmf(-1.0, 5)
    @test_throws ArgumentError rightbound(-1.0, 1.0e-8)
    # an output vector too short for the requested domain
    @test_throws ArgumentError NMarkov.poipmf!(5.05, Vector{Float64}(undef, 3); left=0, right=20)

    # the documented usage keeps working and still sums to one
    for lambda in (0.5, 3.0, 10.0, 100.0)
        right = rightbound(lambda, 1.0e-8)
        @test right >= floor(Int, lambda)
        weight, prob = poipmf(lambda, right)
        @test sum(prob) / weight ≈ 1.0
        w2, p2, cp2 = cpoipmf(lambda, right)
        @test w2 ≈ weight
        @test p2 ≈ prob
    end

    # rightbound must stay monotone: mexp/tran size the p.m.f. buffer from the
    # largest interval and reuse it for the shorter ones.
    prev = -1
    for i = 1:2000
        r = rightbound(i / 100.0, 1.0e-8)
        @test r >= prev
        prev = r
    end
end

## convunifstep! must reject a Poisson vector too short for its range instead of
## reading past it under @inbounds.
@testset "convunifstep! range validation" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        0.0 1.0 -2.0
    ]
    x = [0.3, 0.5, 0.2]
    y = [1.0, 0.0, 0.0]
    P, qv = unif(Q)
    weight, poi = poipmf(qv * 1.0, rightbound(qv * 1.0, 1.0e-8))

    z = zeros(3)
    H = zero(Q)
    @test_throws ArgumentError convunifstep!(:N, :N, P, poi, (0, length(poi) + 5),
        weight, qv * weight, copy(x), y, z, H)
    @test_throws ArgumentError convunifstep!(:N, :N, P, poi, (-1, 3),
        weight, qv * weight, copy(x), y, z, H)
    @test_throws ArgumentError convunifstep!(:N, :N, P, poi, (5, 2),
        weight, qv * weight, copy(x), y, z, H)

    # the degenerate single-term range is legal: z is just x, H stays zero
    w0, poi0 = poipmf(0.5, 0)
    z0 = zeros(3)
    H0 = zero(Q)
    convunifstep!(:N, :N, P, poi0, (0, 0), w0, qv * w0, copy(x), y, z0, H0)
    @test z0 ≈ x
    @test all(H0 .== 0)
end

## --- Gauss-Seidel needs an invertible diagonal. An absorbing state makes the
##     diagonal singular, so stgs/stsengs must say so rather than dividing by
##     zero and returning NaN (which is what they did, silently before the
##     non-convergence warning was added).
@testset "stgs/stsengs reject absorbing states" begin
    Q = [
        -3.0 3.0 0.0;
        0.0 -5.0 5.0;
        0.0 0.0 0.0
    ]
    for A in (sparse(Q), SparseCSC(Q))
        @test_throws ArgumentError stgs(A)
    end

    # The point mass on the absorbing state IS a stationary distribution, so the
    # rejection is about Gauss-Seidel being inapplicable, not about the problem
    # being unsolvable.
    pis = [0.0, 0.0, 1.0]
    @test maximum(abs.(Q' * pis)) < 1.0e-14

    b = [0.1, -0.05, -0.05]
    for A in (sparse(Q), SparseCSC(Q))
        @test_throws ArgumentError stsengs(A, pis, b)
    end

    # A stored-but-zero diagonal must be rejected too, not just a structurally
    # absent one: `adddiag` cannot fix this, because 0 is still 0 when divided by.
    #        col 1 | col 2      | col 3            (the (3,3) zero is stored)
    val = [-3.0, 3.0, -5.0, 5.0, 0.0]
    rowind = [1, 1, 2, 2, 3]
    colptr = [1, 2, 4, 6]
    Qz = SparseCSC(3, 3, val, colptr, rowind)
    @test Matrix(Qz) == Q
    @test length(Qz.val) == 5                # one more entry than nnz(sparse(Q))
    @test spdiag(Qz)[3] == 0.0
    @test_throws ArgumentError stgs(Qz)

    # An irreducible chain is unaffected.
    Qok = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        2.0 1.0 -3.0
    ]
    for A in (sparse(Qok), SparseCSC(Qok))
        p, conv, = stgs(A)
        @test conv
        @test p ≈ gth(Qok)
    end
end

## qstgs has the same Gauss-Seidel precondition. Its `Q` is the generator
## restricted to the transient states; handing it the full generator brings the
## absorbing state's zero row along and used to yield an all-NaN answer (the
## docstring example did exactly that).
@testset "qstgs rejects the full generator" begin
    T = [
        -4.0 1.0 0.0;
        0.0 -1.0 0.1;
        3.0 0.5 -3.5
    ]
    xi = -vec(sum(T, dims=2))
    @test xi ≈ [3.0, 0.9, 0.0]
    @test all(xi .>= 0)

    # correct usage: the transient block only
    for A in (sparse(T), SparseCSC(T))
        x, gam, conv, = qstgs(A, xi)
        @test conv
        @test all(isfinite.(x))
        @test sum(x) ≈ 1.0
        # the QSD eigen-relation: x' T = -gam * x'
        @test maximum(abs.(T' * x + gam * x)) < 1.0e-6
        # gam must be consistent with the returned x, not one sweep stale
        @test gam ≈ sum(x .* xi) rtol = 1.0e-6
    end

    # the full generator, including the absorbing state's zero row, is rejected
    Qfull = [T xi; zeros(1, 4)]
    xifull = [xi; 0.0]
    for A in (sparse(Qfull), SparseCSC(Qfull))
        @test_throws ArgumentError qstgs(A, xifull)
    end
end

## --- trans() must reject an unknown symbol rather than returning nothing.
@testset "trans rejects unknown symbol" begin
    @test NMarkov.trans(:N) == 'N'
    @test NMarkov.trans(:T) == 'T'
    @test_throws ArgumentError NMarkov.trans(:X)
end

end
