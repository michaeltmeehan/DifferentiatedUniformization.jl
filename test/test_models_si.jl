@testset "SI model" begin
    model = SIModel(3)

    @test model.population_size == 3
    @test length(states(model)) == 4
    @test states(model) == [(3, 0), (2, 1), (1, 2), (0, 3)]
    @test initial_distribution(model, (2, 1)) == [0.0, 1.0, 0.0, 0.0]
    @test initial_distribution(model, [0.0, 1.0, 0.0, 0.0]) == [0.0, 1.0, 0.0, 0.0]

    Q = Matrix(generator(model, [0.5]))
    @test size(Q) == (4, 4)
    @test Q[2, 1] == 0.0
    @test Q[3, 2] == 1.0
    @test Q[4, 3] == 1.0
    @test Q[1, 1] == 0.0
    @test Q[2, 2] == -1.0
    @test Q[3, 3] == -1.0
    @test Q[4, 4] == 0.0
    @test Q[1, 2] == 0.0
    @test Q[2, 3] == 0.0
    @test Q[3, 4] == 0.0
    @test count(!iszero, Q) == 4
    @test all(Q[i, j] >= 0.0 for i in axes(Q, 1), j in axes(Q, 2) if i != j)
    @test all(Q[i, i] <= 0.0 for i in axes(Q, 1))
    @test vec(sum(Q; dims=1)) ≈ zeros(4)

    @test_throws ArgumentError SIModel(0)
    @test_throws ArgumentError initial_distribution(model, (3, 1))
    @test_throws ArgumentError generator(model, Float64[])
    @test_throws ArgumentError generator(model, [-0.1])
end
