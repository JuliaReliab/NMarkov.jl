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
