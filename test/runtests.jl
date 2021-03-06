using NMarkov
using Test
using Printf
using Distributions
using SparseMatrix
using SparseArrays

import NMarkov: rightbound, poipmf, cpoipmf, convunifstep!

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

@testset "GS1" begin
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

@testset "GS2" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    @time x, = stgs(SparseMatrixCSC(Q), x0 = [0.3, 0.4, 0.3])
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

@testset "STSENGS1" begin
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

@testset "STSENGS2" begin
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
    y0 = stsengs(SparseMatrixCSC(Q), pis, b)
    
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

@testset "QSTGS1" begin
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

@testset "QSTGS2" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    xi = [1.0, 0.0, 0.0]
    @time x = qstgs(SparseMatrixCSC(Q), xi)

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
    res1 = mexp(Q, x0, t)
    res2 = mexp(SparseCSR(Q), x0, t)
    res3 = mexp(SparseCSC(Q), x0, t)
    res4 = mexp(SparseCOO(Q), x0, t)
    res5 = mexp(SparseMatrixCSC(Q), x0, t)
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    @test res0 ≈ res5
    res0 = exp(Q'*t) * x0
    res1 = mexp(Q, x0, t, transpose=:T)
    res2 = mexp(SparseCSR(Q), x0, t, transpose=:T)
    res3 = mexp(SparseCSC(Q), x0, t, transpose=:T)
    res4 = mexp(SparseCOO(Q), x0, t, transpose=:T)
    res5 = mexp(SparseMatrixCSC(Q), x0, t, transpose=:T)
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    @test res0 ≈ res5
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
    res1 = mexpc(Q, x0, t)
    res2 = mexpc(SparseCSR(Q), x0, t)
    res3 = mexpc(SparseCSC(Q), x0, t)
    res4 = mexpc(SparseCOO(Q), x0, t)
    res5 = mexpc(SparseMatrixCSC(Q), x0, t)
    @test res0[1:3,1:2] ≈ res1[1]
    @test res0[1:3,1:2] ≈ res2[1]
    @test res0[1:3,1:2] ≈ res3[1]
    @test res0[1:3,1:2] ≈ res4[1]
    @test res0[1:3,1:2] ≈ res5[1]
    @test res0[4:6,1:2] ≈ res1[2]
    @test res0[4:6,1:2] ≈ res2[2]
    @test res0[4:6,1:2] ≈ res3[2]
    @test res0[4:6,1:2] ≈ res4[2]
    @test res0[4:6,1:2] ≈ res5[2]

    Qdash = [
        Q' zeros(3,3);
        I zeros(3,3)
    ]
    x0dash = [
        x0;
        zeros(3,2)
    ]
    res0 = exp(Qdash*t) * x0dash
    res1 = mexpc(Q, x0, t, transpose=:T)
    res2 = mexpc(SparseCSR(Q), x0, t, transpose=:T)
    res3 = mexpc(SparseCSC(Q), x0, t, transpose=:T)
    res4 = mexpc(SparseCOO(Q), x0, t, transpose=:T)
    res5 = mexpc(SparseMatrixCSC(Q), x0, t, transpose=:T)
    @test res0[1:3,1:2] ≈ res1[1]
    @test res0[1:3,1:2] ≈ res2[1]
    @test res0[1:3,1:2] ≈ res3[1]
    @test res0[1:3,1:2] ≈ res4[1]
    @test res0[1:3,1:2] ≈ res5[1]
    @test res0[4:6,1:2] ≈ res1[2]
    @test res0[4:6,1:2] ≈ res2[2]
    @test res0[4:6,1:2] ≈ res3[2]
    @test res0[4:6,1:2] ≈ res4[2]
    @test res0[4:6,1:2] ≈ res5[2]
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
    res1 = mexp(Q, x0, ts)
    res2 = mexp(SparseCSR(Q), x0, ts)
    res3 = mexp(SparseCSC(Q), x0, ts)
    res4 = mexp(SparseCOO(Q), x0, ts)
    res5 = mexp(SparseMatrixCSC(Q), x0, ts)
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    @test res0 ≈ res5
    @time res0 = [exp(Q'*t) * x0 for t = ts]
    @time res0 = [exp(Q'*t) * x0 for t = ts]
    @time res1 = mexp(Q, x0, ts, transpose=:T)
    @time res1 = mexp(Q, x0, ts, transpose=:T)
    @time res2 = mexp(SparseCSR(Q), x0, ts, transpose=:T)
    @time res2 = mexp(SparseCSR(Q), x0, ts, transpose=:T)
    @time res3 = mexp(SparseCSC(Q), x0, ts, transpose=:T)
    @time res3 = mexp(SparseCSC(Q), x0, ts, transpose=:T)
    @time res4 = mexp(SparseCOO(Q), x0, ts, transpose=:T)
    @time res4 = mexp(SparseCOO(Q), x0, ts, transpose=:T)
    @time res5 = mexp(SparseMatrixCSC(Q), x0, ts, transpose=:T)
    @time res5 = mexp(SparseMatrixCSC(Q), x0, ts, transpose=:T)
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    @test res0 ≈ res5
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
    @time res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    @time res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    @time res1 = mexpc(Q, x0, ts)
    @time res1 = mexpc(Q, x0, ts)
    @time res2 = mexpc(SparseCSR(Q), x0, ts)
    @time res2 = mexpc(SparseCSR(Q), x0, ts)
    @time res3 = mexpc(SparseCSC(Q), x0, ts)
    @time res3 = mexpc(SparseCSC(Q), x0, ts)
    @time res4 = mexpc(SparseCOO(Q), x0, ts)
    @time res4 = mexpc(SparseCOO(Q), x0, ts)
    @time res5 = mexpc(SparseMatrixCSC(Q), x0, ts)
    @time res5 = mexpc(SparseMatrixCSC(Q), x0, ts)
    @test res0[1] ≈ res1[1]
    @test res0[1] ≈ res2[1]
    @test res0[1] ≈ res3[1]
    @test res0[1] ≈ res4[1]
    @test res0[1] ≈ res5[1]
    @test res0[2] ≈ res1[2]
    @test res0[2] ≈ res2[2]
    @test res0[2] ≈ res3[2]
    @test res0[2] ≈ res4[2]
    @test res0[2] ≈ res5[2]

    Qdash = [
        Q' zeros(3,3);
        I zeros(3,3)
    ]
    x0dash = [
        x0;
        zeros(3,2)
    ]
    @time res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    @time res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    @time res1 = mexpc(Q, x0, ts, transpose=:T)
    @time res1 = mexpc(Q, x0, ts, transpose=:T)
    @time res2 = mexpc(SparseCSR(Q), x0, ts, transpose=:T)
    @time res2 = mexpc(SparseCSR(Q), x0, ts, transpose=:T)
    @time res3 = mexpc(SparseCSC(Q), x0, ts, transpose=:T)
    @time res3 = mexpc(SparseCSC(Q), x0, ts, transpose=:T)
    @time res4 = mexpc(SparseCOO(Q), x0, ts, transpose=:T)
    @time res4 = mexpc(SparseCOO(Q), x0, ts, transpose=:T)
    @time res5 = mexpc(SparseMatrixCSC(Q), x0, ts, transpose=:T)
    @time res5 = mexpc(SparseMatrixCSC(Q), x0, ts, transpose=:T)
    @test res0[1] ≈ res1[1]
    @test res0[1] ≈ res2[1]
    @test res0[1] ≈ res3[1]
    @test res0[1] ≈ res4[1]
    @test res0[1] ≈ res5[1]
    @test res0[2] ≈ res1[2]
    @test res0[2] ≈ res2[2]
    @test res0[2] ≈ res3[2]
    @test res0[2] ≈ res4[2]
    @test res0[2] ≈ res5[2]
end

@testset "mixexp1" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x0 = [
        1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0
    ]
    ans = [
        0.2510060736298584394000 0.1554210097708885740531 0.2424928563850387575052;
        0.1989569330779636657791 0.2168314920514892163439 0.4320884477342963880808;
        0.1989569330779636657791 0.1686453260298337586409 0.4802746137559519845617
    ]
    @time res1 = mexp(Q, x0, Weibull(2.0, 1.0))
    @time res1 = mexp(Q, x0, Weibull(2.0, 1.0))
    for i = eachindex(ans)
        @test abs(ans[i] - res1[i]) < 1.0e-8
    end
end

@testset "mixexpc1" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x0 = [
        1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0
    ]
    ans_y = [
        0.3209844251001398029999 0.2157169520527697503809 0.4632986228470717948724;
        0.2263385249666138854252 0.2406732432647746833254 0.5329882317685930015472;
        0.2263385249666138854252 0.1924870824122626444819 0.5811743926211048183461
    ]
    ans_cy = [
        0.3913106241978086541344 0.1769700990060719680841 0.3179461986875124490659;
        0.1649720992311948519760 0.3017542963096322417016 0.4195005263505657833178;
        0.1649720992311948519760 0.1431186564517208936742 0.5781361662084771868564
    ]
    # println(Q)
    # println(x0)
    @time y, cy = mexpc(Q, x0, Weibull(2.0, 1.0))
    # println(Q)
    # println(x0)
    @time y, cy = mexpc(Q, x0, Weibull(2.0, 1.0))
    for i = eachindex(ans_y)
        @test abs(ans_y[i] - y[i]) < 1.0e-8
    end
    for i = eachindex(ans_cy)
        @test abs(ans_cy[i] - cy[i]) < 1.0e-8
    end
end

@testset "conv" begin
    @testset "conv notrans notrans" begin
        Q = [
            -3.0 2.0 0.0;
            1.0 -5.0 4.0;
            0.0 1.0 -2.0
        ]
        x = rand(3)
        y = rand(3)
        A = [
            Q reshape(x, (3,1)) * reshape(y, (1,3));
            zeros(3,3) Q
        ]

        tau = 0.5
        X = exp(A*tau)
        z0 = X[1:3,1:3] * x

        P, qv = unif(Q)
        right = rightbound(qv*tau, 1.0e-8)
        weight, poi = poipmf(qv*tau, right, left = 0)
        z = zeros(3)
        H = zero(Q)
        convunifstep!(:N, :N,
            P, poi, (0, right), weight, qv * weight, copy(x), y, z, H)
        @test z ≈ z0
        @test X[1:3,4:6] ≈ H

        P, qv = unif(SparseCSR(Q))
        right = rightbound(qv*tau, 1.0e-8)
        weight, poi = poipmf(qv*tau, right, left = 0)
        z = zeros(3)
        H = zero(P)
        convunifstep!(:N, :N,
            P, poi, (0, right), weight, qv * weight, copy(x), y, z, H)
        @test z ≈ z0
        for i = 1:3
            for z = H.rowptr[i]:H.rowptr[i+1]-1
                j = H.colind[z]
                @test X[i, j+3] ≈ H.val[z]
            end
        end

        P, qv = unif(SparseCSC(Q))
        right = rightbound(qv*tau, 1.0e-8)
        weight, poi = poipmf(qv*tau, right, left = 0)
        z = zeros(3)
        H = zero(P)
        convunifstep!(:N, :N,
            P, poi, (0, right), weight, qv * weight, copy(x), y, z, H)
        @test z ≈ z0
        for j = 1:3
            for z = H.colptr[j]:H.colptr[j+1]-1
                i = H.rowind[z]
                @test X[i, j+3] ≈ H.val[z]
            end
        end

        P, qv = unif(SparseCOO(Q))
        right = rightbound(qv*tau, 1.0e-8)
        weight, poi = poipmf(qv*tau, right, left = 0)
        z = zeros(3)
        H = zero(P)
        convunifstep!(:N, :N,
            P, poi, (0, right), weight, qv * weight, copy(x), y, z, H)
        @test z ≈ z0
        for z = 1:nnz(H)
            i = H.rowind[z]
            j = H.colind[z]
            @test X[i, j+3] ≈ H.val[z]
        end

        P, qv = unif(SparseMatrixCSC(Q))
        right = rightbound(qv*tau, 1.0e-8)
        weight, poi = poipmf(qv*tau, right, left = 0)
        z = zeros(3)
        H = zero(P)
        convunifstep!(:N, :N,
            P, poi, (0, right), weight, qv * weight, copy(x), y, z, H)
        @test z ≈ z0
        for j = 1:3
            for z = H.colptr[j]:H.colptr[j+1]-1
                i = H.rowval[z]
                @test X[i, j+3] ≈ H.nzval[z]
            end
        end
    end

    @testset "conv notrans trans" begin
        Q = [
            -3.0 2.0 0.0;
            1.0 -5.0 4.0;
            0.0 1.0 -2.0
        ]
        x = rand(3)
        y = rand(3)
        A = [
            Q reshape(x, (3,1)) * reshape(y, (1,3));
            zeros(3,3) Q
        ]

        tau = 0.5
        X = exp(A*tau)
        z0 = X[1:3,1:3] * x

        P, qv = unif(Q)
        right = rightbound(qv*tau, 1.0e-8)
        weight, poi = poipmf(qv*tau, right, left = 0)
        z = zeros(3)
        H = zero(Q)
        convunifstep!(:N, :T,
            P, poi, (0, right), weight, qv * weight, x, y, z, H)
        @test z ≈ z0
        @test X[1:3,4:6]' ≈ H
    end

    @testset "conv trans notrans" begin
        Q = [
            -3.0 2.0 0.0;
            1.0 -5.0 4.0;
            0.0 1.0 -2.0
        ]
        x = rand(3)
        y = rand(3)
        A = [
            Q' reshape(x, (3,1)) * reshape(y, (1,3));
            zeros(3,3) Q'
        ]

        tau = 0.5
        X = exp(A*tau)
        z0 = X[1:3,1:3] * x

        P, qv = unif(Q)
        right = rightbound(qv*tau, 1.0e-8)
        weight, poi = poipmf(qv*tau, right, left = 0)
        z = zeros(3)
        H = zero(Q)
        convunifstep!(:T, :N,
            P, poi, (0, right), weight, qv * weight, x, y, z, H)
        @test z ≈ z0
        @test X[1:3,4:6] ≈ H
    end

    @testset "conv trans trans" begin
        Q = [
            -3.0 2.0 0.0;
            1.0 -5.0 4.0;
            0.0 1.0 -2.0
        ]
        x = rand(3)
        y = rand(3)
        A = [
            Q' reshape(x, (3,1)) * reshape(y, (1,3));
            zeros(3,3) Q'
        ]

        tau = 0.5
        X = exp(A*tau)
        z0 = X[1:3,1:3] * x

        P, qv = unif(Q)
        right = rightbound(qv*tau, 1.0e-8)
        weight, poi = poipmf(qv*tau, right, left = 0)
        z = zeros(3)
        H = zero(Q)
        convunifstep!(:T, :T,
            P, poi, (0, right), weight, qv * weight, x, y, z, H)
        @test z ≈ z0
        @test X[1:3,4:6]' ≈ H
    end
end

@testset "tran 1" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x = Float64[1, 0, 0]
    r = Float64[1, 1, 0]

    ts = LinRange(0.0, 10.0, 10)
    @time irwd, crwd, y, cy = tran(Q, x, r, ts, forward=:T)
    
    res,cres = mexpc(Q, x, ts, transpose=:T)
    @test isapprox(irwd, [sum(x .* r) for x = res])
    @test isapprox(crwd, [sum(x .* r) for x = cres])
end

@testset "tran 2" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x = Float64[1, 0, 0]
    r = Float64[1, 1, 0]

    ts = LinRange(0.0, 10.0, 10)
    @time irwd, crwd, y, cy = tran(Q, x, r, ts, forward=:N)
    
    res,cres = mexpc(Q, x, ts, transpose=:T)
    @test isapprox(irwd, [sum(x .* r) for x = res])
    @test isapprox(crwd, [sum(x .* r) for x = cres])
end

@testset "tran 3" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x = Float64[1 0 0; 0 1 0]
    r = Float64[1, 1, 0]

    ts = LinRange(0.0, 10.0, 10)
    @time irwd, crwd, y, cy = tran(Q, x, r, ts, forward=:T)
    
    res,cres = mexpc(Q, r, ts, transpose=:N)
    @test isapprox(irwd, [x* r for r = res])
    @test isapprox(crwd, [x* r for r = cres])
end

@testset "tran 4" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x = Float64[1 0 0; 0 1 0]
    r = Float64[1, 1, 0]

    ts = LinRange(0.0, 10.0, 10)
    @time irwd, crwd, y, cy = tran(Q, x, r, ts, forward=:N)
    
    res,cres = mexpc(Q, r, ts, transpose=:N)
    @test isapprox(irwd, [x* r for r = res])
    @test isapprox(crwd, [x* r for r = cres])
end

@testset "tran 5" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x = Float64[1 0 0; 0 1 0]
    r = Float64[1 1 0; 0 1 0]

    ts = LinRange(0.0, 10.0, 10)
    @time irwd, crwd, y, cy = tran(Q, x, r', ts, forward=:N)
    
    res,cres = mexpc(Q, r', ts, transpose=:N)
    @test isapprox(irwd, [x * r for r = res])
    @test isapprox(crwd, [x * r for r = cres])
end

@testset "tran 6" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x = Float64[1 0 0; 0 1 0]
    r = Float64[1 1 0; 0 1 0]

    ts = LinRange(0.0, 10.0, 10)
    @time irwd, crwd, y, cy = tran(Q, x, r', ts, forward=:T)
    
    res,cres = mexpc(Q, r', ts, transpose=:N)
    @test isapprox(irwd, [x * r for r = res])
    @test isapprox(crwd, [x * r for r = cres])
end

@testset "mexp3" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x0 = [
        0.5 0.5 0.0;
        0.5 0.5 0.5
    ]
    t = 10.0
    res0 = exp(Q*t) * x0'
    res1 = mexp(Q, x0', t)
    res2 = mexp(SparseCSR(Q), x0', t)
    res3 = mexp(SparseCSC(Q), x0', t)
    res4 = mexp(SparseCOO(Q), x0', t)
    res5 = mexp(SparseMatrixCSC(Q), x0', t)
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    @test res0 ≈ res5
    res0 = exp(Q'*t) * x0'
    res1 = mexp(Q, x0', t, transpose=:T)
    res2 = mexp(SparseCSR(Q), x0', t, transpose=:T)
    res3 = mexp(SparseCSC(Q), x0', t, transpose=:T)
    res4 = mexp(SparseCOO(Q), x0', t, transpose=:T)
    res5 = mexp(SparseMatrixCSC(Q), x0', t, transpose=:T)
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    @test res0 ≈ res5
end

@testset "mexp4" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x0 = [
        0.5 0.5 0.0;
        0.5 0.5 0.5
    ]
    ts = LinRange(0.0, 10.0, 10)
    res0 = [exp(Q*t) * x0' for t = ts]
    res1 = mexp(Q, x0', ts)
    res2 = mexp(SparseCSR(Q), x0', ts)
    res3 = mexp(SparseCSC(Q), x0', ts)
    res4 = mexp(SparseCOO(Q), x0', ts)
    res5 = mexp(SparseMatrixCSC(Q), x0', ts)
    # println(typeof(res1))
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    @test res0 ≈ res5
    @time res0 = [exp(Q'*t) * x0' for t = ts]
    @time res0 = [exp(Q'*t) * x0' for t = ts]
    @time res1 = mexp(Q, x0', ts, transpose=:T)
    @time res1 = mexp(Q, x0', ts, transpose=:T)
    @time res2 = mexp(SparseCSR(Q), x0', ts, transpose=:T)
    @time res2 = mexp(SparseCSR(Q), x0', ts, transpose=:T)
    @time res3 = mexp(SparseCSC(Q), x0', ts, transpose=:T)
    @time res3 = mexp(SparseCSC(Q), x0', ts, transpose=:T)
    @time res4 = mexp(SparseCOO(Q), x0', ts, transpose=:T)
    @time res4 = mexp(SparseCOO(Q), x0', ts, transpose=:T)
    @time res5 = mexp(SparseMatrixCSC(Q), x0', ts, transpose=:T)
    @time res5 = mexp(SparseMatrixCSC(Q), x0', ts, transpose=:T)
    @test res0 ≈ res1
    @test res0 ≈ res2
    @test res0 ≈ res3
    @test res0 ≈ res4
    @test res0 ≈ res5
end

@testset "mexpc3" begin
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
    x1 = copy(x0')
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
    res1 = mexpc(Q, x1', t)
    res2 = mexpc(SparseCSR(Q), x1', t)
    res3 = mexpc(SparseCSC(Q), x1', t)
    res4 = mexpc(SparseCOO(Q), x1', t)
    res5 = mexpc(SparseMatrixCSC(Q), x1', t)
    @test res0[1:3,1:2] ≈ res1[1]
    @test res0[1:3,1:2] ≈ res2[1]
    @test res0[1:3,1:2] ≈ res3[1]
    @test res0[1:3,1:2] ≈ res4[1]
    @test res0[1:3,1:2] ≈ res5[1]
    @test res0[4:6,1:2] ≈ res1[2]
    @test res0[4:6,1:2] ≈ res2[2]
    @test res0[4:6,1:2] ≈ res3[2]
    @test res0[4:6,1:2] ≈ res4[2]
    @test res0[4:6,1:2] ≈ res5[2]

    Qdash = [
        Q' zeros(3,3);
        I zeros(3,3)
    ]
    x0dash = [
        x0;
        zeros(3,2)
    ]
    res0 = exp(Qdash*t) * x0dash
    res1 = mexpc(Q, x1', t, transpose=:T)
    res2 = mexpc(SparseCSR(Q), x1', t, transpose=:T)
    res3 = mexpc(SparseCSC(Q), x1', t, transpose=:T)
    res4 = mexpc(SparseCOO(Q), x1', t, transpose=:T)
    res5 = mexpc(SparseMatrixCSC(Q), x1', t, transpose=:T)
    @test res0[1:3,1:2] ≈ res1[1]
    @test res0[1:3,1:2] ≈ res2[1]
    @test res0[1:3,1:2] ≈ res3[1]
    @test res0[1:3,1:2] ≈ res4[1]
    @test res0[1:3,1:2] ≈ res5[1]
    @test res0[4:6,1:2] ≈ res1[2]
    @test res0[4:6,1:2] ≈ res2[2]
    @test res0[4:6,1:2] ≈ res3[2]
    @test res0[4:6,1:2] ≈ res4[2]
    @test res0[4:6,1:2] ≈ res5[2]
end

@testset "mexpc4" begin
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
    x1 = copy(x0')
    ts = LinRange(0.0, 10.0, 10)

    Qdash = [
        Q zeros(3,3);
        I zeros(3,3)
    ]
    x0dash = [
        x0;
        zeros(3,2)
    ]
    @time res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    @time res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    @time res1 = mexpc(Q, x1', ts)
    @time res1 = mexpc(Q, x1', ts)
    @time res2 = mexpc(SparseCSR(Q), x1', ts)
    @time res2 = mexpc(SparseCSR(Q), x1', ts)
    @time res3 = mexpc(SparseCSC(Q), x1', ts)
    @time res3 = mexpc(SparseCSC(Q), x1', ts)
    @time res4 = mexpc(SparseCOO(Q), x1', ts)
    @time res4 = mexpc(SparseCOO(Q), x1', ts)
    @time res5 = mexpc(SparseMatrixCSC(Q), x1', ts)
    @time res5 = mexpc(SparseMatrixCSC(Q), x1', ts)
    @test res0[1] ≈ res1[1]
    @test res0[1] ≈ res2[1]
    @test res0[1] ≈ res3[1]
    @test res0[1] ≈ res4[1]
    @test res0[1] ≈ res5[1]
    @test res0[2] ≈ res1[2]
    @test res0[2] ≈ res2[2]
    @test res0[2] ≈ res3[2]
    @test res0[2] ≈ res4[2]
    @test res0[2] ≈ res5[2]

    Qdash = [
        Q' zeros(3,3);
        I zeros(3,3)
    ]
    x0dash = [
        x0;
        zeros(3,2)
    ]
    @time res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    @time res0 = ([(exp(Qdash*t) * x0dash)[1:3,1:2] for t = ts], [(exp(Qdash*t) * x0dash)[4:6,1:2] for t = ts])
    @time res1 = mexpc(Q, x1', ts, transpose=:T)
    @time res1 = mexpc(Q, x1', ts, transpose=:T)
    @time res2 = mexpc(SparseCSR(Q), x1', ts, transpose=:T)
    @time res2 = mexpc(SparseCSR(Q), x1', ts, transpose=:T)
    @time res3 = mexpc(SparseCSC(Q), x1', ts, transpose=:T)
    @time res3 = mexpc(SparseCSC(Q), x1', ts, transpose=:T)
    @time res4 = mexpc(SparseCOO(Q), x1', ts, transpose=:T)
    @time res4 = mexpc(SparseCOO(Q), x1', ts, transpose=:T)
    @time res5 = mexpc(SparseMatrixCSC(Q), x1', ts, transpose=:T)
    @time res5 = mexpc(SparseMatrixCSC(Q), x1', ts, transpose=:T)
    @test res0[1] ≈ res1[1]
    @test res0[1] ≈ res2[1]
    @test res0[1] ≈ res3[1]
    @test res0[1] ≈ res4[1]
    @test res0[1] ≈ res5[1]
    @test res0[2] ≈ res1[2]
    @test res0[2] ≈ res2[2]
    @test res0[2] ≈ res3[2]
    @test res0[2] ≈ res4[2]
    @test res0[2] ≈ res5[2]
end

@testset "mixexp2" begin
    Q = [
        -3.0 2.0 0.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x0 = [
        1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0
    ]
    x1 = copy(x0')
    ans = [
        0.2510060736298584394000 0.1554210097708885740531 0.2424928563850387575052;
        0.1989569330779636657791 0.2168314920514892163439 0.4320884477342963880808;
        0.1989569330779636657791 0.1686453260298337586409 0.4802746137559519845617
    ]
    @time res1 = mexp(Q, x1', Weibull(2.0, 1.0))
    @time res1 = mexp(Q, x1', Weibull(2.0, 1.0))
    for i = eachindex(ans)
        @test abs(ans[i] - res1[i]) < 1.0e-8
    end
end

@testset "mixexpc2" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    x0 = [
        1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0
    ]
    x1 = copy(x0')
    ans_y = [
        0.3209844251001398029999 0.2157169520527697503809 0.4632986228470717948724;
        0.2263385249666138854252 0.2406732432647746833254 0.5329882317685930015472;
        0.2263385249666138854252 0.1924870824122626444819 0.5811743926211048183461
    ]
    ans_cy = [
        0.3913106241978086541344 0.1769700990060719680841 0.3179461986875124490659;
        0.1649720992311948519760 0.3017542963096322417016 0.4195005263505657833178;
        0.1649720992311948519760 0.1431186564517208936742 0.5781361662084771868564
    ]
    @time y, cy = mexpc(Q, x1', Weibull(2.0, 1.0))
    @time y, cy = mexpc(Q, x1', Weibull(2.0, 1.0))
    for i = eachindex(ans_y)
        @test abs(ans_y[i] - y[i]) < 1.0e-8
    end
    for i = eachindex(ans_cy)
        @test abs(ans_cy[i] - cy[i]) < 1.0e-8
    end
end

@testset "zeromat" begin
    Q = [0.0][:,:]
    x0 = eye(1)
    y = mexp(Q, x0, 1.0)
    @test y == [1.0][:,:]
    y, cy = mexpc(Q, x0, 10.0)
    @test y == [1.0][:,:]
    @test isapprox(cy, [10.0][:,:])
end

@testset "zeromat_mix" begin
    Q = [0.0][:,:]
    x0 = eye(1)
    y = mexpmix(Q, x0) do t
        exp(-t)
    end
    @test isapprox(y, [1.0][:,:])
    y, cy = mexpcmix(Q, x0, bounds=(0.0, 10.0)) do t
        exp(-t)
    end
    println(cy)
    @test isapprox(y[1,1], 1-exp(-10.0))
    @test isapprox(cy[1,1], (1-(1+10.0*1.0)*exp(-10.0*1.0))/1.0)
end