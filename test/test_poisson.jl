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
