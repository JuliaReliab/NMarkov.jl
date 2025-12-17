using NMarkov

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
    @time irwd, crwd, y, cy = tran(Q, x, r', ts, forward=:T)
    
    res,cres = mexpc(Q, r', ts, transpose=:N)
    @test isapprox(irwd, [x * r for r = res])
    @test isapprox(crwd, [x * r for r = cres])
end
@testset "tran type flexibility" begin
    Q = [
        -3.0 2.0 1.0;
        1.0 -5.0 4.0;
        1.0 1.0 -2.0
    ]
    # Test with Int vectors and Int time series
    x_int = [1, 0, 0]  # Int vector
    r_int = [1, 2, 3]  # Int vector
    ts_int = 1:10  # Int range
    
    x_float = Float64[1, 0, 0]
    r_float = Float64[1, 2, 3]
    ts_float = Float64.(1:10)
    
    irwd_float, crwd_float, y_float, cy_float = tran(Q, x_float, r_float, ts_float, forward=:T)
    irwd_int, crwd_int, y_int, cy_int = tran(Q, x_int, r_int, ts_int, forward=:T)
    
    @test irwd_int ≈ irwd_float
    @test crwd_int ≈ crwd_float
    @test y_int ≈ y_float
    @test cy_int ≈ cy_float
end