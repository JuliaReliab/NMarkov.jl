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

function makeQ()
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
end

function makeQ2()
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
    return Q, Qdash
end

@testset "GTH" begin
    Q = makeQ()
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
    Q = makeQ()
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
    Q = makeQ()
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
    Q, Qdash = makeQ2()
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
    Q, Qdash = makeQ2()
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
