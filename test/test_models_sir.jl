@testset "SIR model" begin
    model = SIRModel(2)

    @test model.population_size == 2
    @test length(states(model)) == 6
    @test states(model) == [(2, 0, 0), (1, 0, 1), (0, 0, 2), (1, 1, 0), (0, 1, 1), (0, 2, 0)]
    @test initial_distribution(model, (1, 1, 0)) == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    @test initial_distribution(model, zeros(6)) == zeros(6)

    Q = Matrix(generator(model, [0.5, 1.5]))
    @test size(Q) == (6, 6)
    @test Q[4, 1] == 0.0
    @test Q[6, 4] == 0.5
    @test Q[2, 4] == 1.5
    @test Q[6, 5] == 0.0
    @test Q[3, 5] == 1.5
    @test Q[5, 6] == 3.0
    @test Q[4, 5] == 0.0
    @test Q[1, 4] == 0.0
    @test Q[4, 4] == -2.0
    @test Q[5, 5] == -1.5
    @test Q[6, 6] == -3.0
    @test count(!iszero, Q) == 7
    @test all(Q[i, j] >= 0.0 for i in axes(Q, 1), j in axes(Q, 2) if i != j)
    @test all(Q[i, i] <= 0.0 for i in axes(Q, 1))
    @test vec(sum(Q; dims=1)) ≈ zeros(6)

    @test_throws ArgumentError SIRModel(0)
    @test_throws ArgumentError initial_distribution(model, (2, 1, 0))
    @test_throws ArgumentError generator(model, [0.1])
    @test_throws ArgumentError generator(model, [-0.1, 0.2])
end
