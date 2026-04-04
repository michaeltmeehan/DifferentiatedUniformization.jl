@testset "SIS model" begin
    model = SISModel(2)

    @test model.population_size == 2
    @test length(states(model)) == 3
    @test states(model) == [(2, 0), (1, 1), (0, 2)]
    @test initial_distribution(model, (1, 1)) == [0.0, 1.0, 0.0]
    @test initial_distribution(model, [1.0, 0.0, 0.0]) == [1.0, 0.0, 0.0]

    Q = Matrix(generator(model, [0.5, 1.5]))
    @test size(Q) == (3, 3)
    @test Q[2, 1] == 0.0
    @test Q[3, 2] == 0.5
    @test Q[1, 2] == 1.5
    @test Q[2, 3] == 3.0
    @test Q[1, 3] == 0.0
    @test Q[3, 1] == 0.0
    @test Q[1, 1] == 0.0
    @test Q[2, 2] == -2.0
    @test Q[3, 3] == -3.0
    @test count(!iszero, Q) == 5
    @test all(Q[i, j] >= 0.0 for i in axes(Q, 1), j in axes(Q, 2) if i != j)
    @test all(Q[i, i] <= 0.0 for i in axes(Q, 1))
    @test vec(sum(Q; dims=1)) ≈ zeros(3)

    @test_throws ArgumentError SISModel(-1)
    @test_throws ArgumentError initial_distribution(model, [1.0, 0.0])
    @test_throws ArgumentError generator(model, [0.1])
    @test_throws ArgumentError generator(model, [0.1, -0.2])
end
