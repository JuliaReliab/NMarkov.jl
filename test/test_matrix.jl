@testset "unif1" begin
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
    P, qv = unif(Q)
    println(P)
end

@testset "unif2" begin
    Q = [
        -3.0 3.0 0.0;
        0.0 -5.0 5.0;
        0.0 0.0 -2.0
    ]
    P, qv = unif(Q)
    println(P)
end

@testset "unif3" begin
    Q = [
        -3.0 3.0 0.0;
        0.0 -5.0 5.0;
        0.0 0.0 0.0
    ]
    spQ = sparse(Q)
    P, qv = unif(spQ)
    println(P)
end
