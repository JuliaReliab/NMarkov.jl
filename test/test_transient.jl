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
