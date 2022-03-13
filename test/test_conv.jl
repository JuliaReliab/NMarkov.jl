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
