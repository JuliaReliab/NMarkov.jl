
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
