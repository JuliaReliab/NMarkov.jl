using NMarkov
using Test
using Printf
using Distributions
using SparseMatrix

@testset "PoissonRight" begin
    for q in [1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12]
        for lambda in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
            @test 0 <= rightbound(lambda, q) - cquantile(Poisson(lambda), q) <= 6
        end
    end
    @time for q in [1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12]
        for lambda in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
            rightbound(lambda, q)
        end
    end
    @time for q in [1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12]
        for lambda in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
            cquantile(Poisson(lambda), q)
        end
    end
end

@testset "PoissonPMF" begin
    for q in [1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12]
        for lambda in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
            right = rightbound(lambda, q)
            weight,p = poipmf(lambda, right)
            v1 = p / weight
            v2 = pdf.(Poisson(lambda), 0:right)
            # println(v1)
            # println(v2)
            @test v1 ≈ v2
        end
    end
    @time for q in [1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12]
        for lambda in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
            right = rightbound(lambda, q)
            weight,p = poipmf(lambda, right)
        end
    end
    @time for q in [1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12]
        for lambda in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
            right = rightbound(lambda, q)
            v2 = pdf.(Poisson(lambda), 0:right)
        end
    end
end

@testset "PoissonCCDF" begin
    for q in [1.0e-10, 1.0e-11, 1.0e-12]
        for lambda in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
            right = rightbound(lambda, q)
            weight,p,cp = cpoipmf(lambda, right)
            v1 = cp / weight
            v2 = ccdf.(Poisson(lambda), 0:right)
            # println(v1)
            # println(v2)
            @test v1 ≈ v2
        end
    end
    @time for q in [1.0e-10, 1.0e-11, 1.0e-12]
        for lambda in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
            right = rightbound(lambda, q)
            weight,p,cp = cpoipmf(lambda, right)
            v1 = cp / weight
        end
    end
    @time for q in [1.0e-10, 1.0e-11, 1.0e-12]
        for lambda in [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 100000.0, 1000000.0]
            right = rightbound(lambda, q)
            v2 = ccdf.(Poisson(lambda), 0:right)
        end
    end
end

@testset "GTH" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    @time x = gth(Q)
    Q1 = copy(Q)
    Q1[:,1] .= 1
    b = zeros(size(Q1)[1])
    b[1] = 1.0
    @time x0 = Q1' \ b
    @test x0 ≈ x
    @time x = gth(Q, [2,3,1]) # permutation
    @test x0 ≈ x
end

@testset "GS" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    @time x, = stgs(SparseCSC(Q), x0 = [0.3, 0.4, 0.3])
    Q1 = copy(Q)
    Q1[:,1] .= 1
    b = zeros(size(Q1)[1])
    b[1] = 1.0
    @time x0 = Q1' \ b
    @test x0 ≈ x
    @time x, = stgs(SparseCSC(Q))
    @test x0 ≈ x
end

@testset "Power" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    @time x, = stpower(unif(Q)[1], x0 = [0.3, 0.4, 0.3])
    Q1 = copy(Q)
    Q1[:,1] .= 1
    b = zeros(size(Q1)[1])
    b[1] = 1.0
    @time x0 = Q1' \ b
    @test x0 ≈ x
    @time x, = stpower(unif(Q)[1])
    @test x0 ≈ x
end

@testset "STSENGS" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    Qdash = [
        -1.0 1.0 0.0;
        0.0 0.0 0.0;
        0.0 0.0 0.0
    ]
    pis = gth(Q)
    b = Qdash' * pis
    y0 = stsengs(SparseCSC(Q), pis, b)
    
    Q1 = copy(Q)
    Q1[:,1] .= 1
    b[1] = 0.0
    x0 = Q1' \ (-b)

    @test x0 ≈ y0[1]
end

@testset "STSENPower" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    Qdash = [
        -1.0 1.0 0.0;
        0.0 0.0 0.0;
        0.0 0.0 0.0
    ]
    P, qv = unif(SparseCSC(Q))
    Pdash = Qdash / qv
    pis = gth(Q)
    y0 = stsenpower(P, pis, Pdash' * pis)
    
    Q1 = copy(Q)
    Q1[:,1] .= 1
    b = Qdash' * pis
    b[1] = 0.0
    x0 = Q1' \ (-b)

    @test x0 ≈ y0[1]
end

@testset "QSTGS" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    xi = [1.0, 0.0, 0.0]
    @time x = qstgs(SparseCSC(Q), xi)

    P, qv = unif(Q)
    pxi = xi / qv
    @time x2 = qstpower(P, pxi)

    @test x[1] ≈ x2[1]
    @test x[2] ≈ x2[2]*qv
end

@testset "mexp1" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x0 = [
        0.5 0.0;
        0.5 0.5;
        0.0 0.5
    ]
    t = 10.0
    res0 = exp(Q*t) * x0
    res1 = mexp(Q, t, x0)
    res2 = mexp(SparseCSR(Q), t, x0)
    res3 = mexp(SparseCSC(Q), t, x0)
    res4 = mexp(SparseCOO(Q), t, x0)
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    res0 = exp(Q'*t) * x0
    res1 = mexp(Q, t, x0, transpose=Trans())
    res2 = mexp(SparseCSR(Q), t, x0, transpose=Trans())
    res3 = mexp(SparseCSC(Q), t, x0, transpose=Trans())
    res4 = mexp(SparseCOO(Q), t, x0, transpose=Trans())
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
end

@testset "mexpc1" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    I = [
        1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0
    ]
    x0 = [
        0.5 0.0;
        0.5 0.5;
        0.0 0.5
    ]
    t = 10.0

    Qdash = [
        Q zeros(3,3);
        I zeros(3,3)
    ]
    x0dash = [
        x0;
        zeros(3,2)
    ]
    res0 = exp(Qdash*t) * x0dash
    res1 = mexpc(Q, t, x0)
    res2 = mexpc(SparseCSR(Q), t, x0)
    res3 = mexpc(SparseCSC(Q), t, x0)
    res4 = mexpc(SparseCOO(Q), t, x0)
    @test res0[1:3,1:2] ≈ res1[1]
    @test res0[1:3,1:2] ≈ res2[1]
    @test res0[1:3,1:2] ≈ res3[1]
    @test res0[1:3,1:2] ≈ res4[1]
    @test res0[4:6,1:2] ≈ res1[2]
    @test res0[4:6,1:2] ≈ res2[2]
    @test res0[4:6,1:2] ≈ res3[2]
    @test res0[4:6,1:2] ≈ res4[2]

    Qdash = [
        Q' zeros(3,3);
        I zeros(3,3)
    ]
    x0dash = [
        x0;
        zeros(3,2)
    ]
    res0 = exp(Qdash*t) * x0dash
    res1 = mexpc(Q, t, x0, transpose=Trans())
    res2 = mexpc(SparseCSR(Q), t, x0, transpose=Trans())
    res3 = mexpc(SparseCSC(Q), t, x0, transpose=Trans())
    res4 = mexpc(SparseCOO(Q), t, x0, transpose=Trans())
    @test res0[1:3,1:2] ≈ res1[1]
    @test res0[1:3,1:2] ≈ res2[1]
    @test res0[1:3,1:2] ≈ res3[1]
    @test res0[1:3,1:2] ≈ res4[1]
    @test res0[4:6,1:2] ≈ res1[2]
    @test res0[4:6,1:2] ≈ res2[2]
    @test res0[4:6,1:2] ≈ res3[2]
    @test res0[4:6,1:2] ≈ res4[2]
end

@testset "mexp2" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x0 = [
        0.5 0.0;
        0.5 0.5;
        0.0 0.5
    ]
    ts = LinRange(0.0, 10.0, 10)
    res0 = [exp(Q*t) * x0 for t = ts]
    res1 = mexp(Q, ts, x0)
    res2 = mexp(SparseCSR(Q), ts, x0)
    res3 = mexp(SparseCSC(Q), ts, x0)
    res4 = mexp(SparseCOO(Q), ts, x0)
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    res0 = [exp(Q'*t) * x0 for t = ts]
    res1 = mexp(Q, ts, x0, transpose=Trans())
    res2 = mexp(SparseCSR(Q), ts, x0, transpose=Trans())
    res3 = mexp(SparseCSC(Q), ts, x0, transpose=Trans())
    res4 = mexp(SparseCOO(Q), ts, x0, transpose=Trans())
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
end

@testset "mexpc2" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    I = [
        1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0
    ]
    x0 = [
        0.5 0.0;
        0.5 0.5;
        0.0 0.5
    ]
    ts = LinRange(0.0, 10.0, 10)

    Qdash = [
        Q zeros(3,3);
        I zeros(3,3)
    ]
    x0dash = [
        x0;
        zeros(3,2)
    ]
    res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    res1 = mexpc(Q, ts, x0)
    res2 = mexpc(SparseCSR(Q), ts, x0)
    res3 = mexpc(SparseCSC(Q), ts, x0)
    res4 = mexpc(SparseCOO(Q), ts, x0)
    @test res0[1] ≈ res1[1]
    @test res0[1] ≈ res2[1]
    @test res0[1] ≈ res3[1]
    @test res0[1] ≈ res4[1]
    @test res0[2] ≈ res1[2]
    @test res0[2] ≈ res2[2]
    @test res0[2] ≈ res3[2]
    @test res0[2] ≈ res4[2]

    Qdash = [
        Q' zeros(3,3);
        I zeros(3,3)
    ]
    x0dash = [
        x0;
        zeros(3,2)
    ]
    res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    res1 = mexpc(Q, ts, x0, transpose=Trans())
    res2 = mexpc(SparseCSR(Q), ts, x0, transpose=Trans())
    res3 = mexpc(SparseCSC(Q), ts, x0, transpose=Trans())
    res4 = mexpc(SparseCOO(Q), ts, x0, transpose=Trans())
    @test res0[1] ≈ res1[1]
    @test res0[1] ≈ res2[1]
    @test res0[1] ≈ res3[1]
    @test res0[1] ≈ res4[1]
    @test res0[2] ≈ res1[2]
    @test res0[2] ≈ res2[2]
    @test res0[2] ≈ res3[2]
    @test res0[2] ≈ res4[2]
end
