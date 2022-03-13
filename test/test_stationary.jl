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

@testset "STSEN1" begin
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
    y0 = stsen(Q, pis, b)
    
    Q1 = copy(Q)
    Q1[:,1] .= 1
    b[1] = 0.0
    x0 = Q1' \ (-b)

    @test x0 ≈ y0
end

@testset "STSEN12" begin
    Q = [
        -2.0 2.0 0.0;
        1.0 -2.0 1.0;
        0.0 1.0 -1.0
    ]
    Qdash = [
        -1.0 1.0 0.0;
        0.0 0.0 0.0;
        0.0 0.0 0.0
    ]
    pis = gth(Q)
    b = Qdash' * pis
    y0 = stsen(Q, pis, b)
    
    Q1 = copy(Q)
    Q1[:,1] .= 1
    b[1] = 0.0
    x0 = Q1' \ (-b)

    @test x0 ≈ y0
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
