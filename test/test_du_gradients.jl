@testset "Generator derivative structure" begin
    si_model = SIModel(2)
    si_dQ = generator_derivatives(si_model, [0.5])
    @test length(si_dQ) == 1
    @test Matrix(si_dQ[1]) == [0.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 1.0 0.0]

    sis_model = SISModel(2)
    sis_dQ = generator_derivatives(sis_model, [0.5, 1.5])
    @test length(sis_dQ) == 2
    @test Matrix(sis_dQ[1]) == [0.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 1.0 -0.0]
    @test Matrix(sis_dQ[2]) == [0.0 1.0 0.0; 0.0 -1.0 2.0; 0.0 0.0 -2.0]

    sir_model = SIRModel(2)
    sir_dQ = generator_derivatives(sir_model, [0.5, 1.5])
    @test length(sir_dQ) == 2
    @test size(sir_dQ[1]) == (6, 6)
    @test size(sir_dQ[2]) == (6, 6)
    @test Matrix(sir_dQ[1])[6, 4] == 1.0
    @test Matrix(sir_dQ[1])[4, 4] == -1.0
    @test Matrix(sir_dQ[2])[2, 4] == 1.0
    @test Matrix(sir_dQ[2])[4, 4] == -1.0
end

@testset "Differentiated uniformization core behavior" begin
    model = SISModel(2)
    θ = [0.5, 1.5]
    Q = generator(model, θ)
    dQ = generator_derivatives(model, θ)
    p0 = initial_distribution(model, (1, 1))

    result_zero = differentiate_uniformize(Q, dQ, 0.0, p0; gamma=4.0)
    @test result_zero isa DUGradientResult
    @test result_zero.p == p0
    @test result_zero.dp == zeros(length(p0), length(θ))
    @test result_zero.n_terms == 1
    @test result_zero.tail_mass_bound == 0.0

    result = differentiate_uniformize(Q, dQ, 0.5, p0; tol=1.0e-12, gamma=4.0)
    fd = DifferentiatedUniformization.finite_difference_gradient(
        θ_local -> propagate(model, θ_local, (1, 1), 0.5; tol=1.0e-12, gamma=4.0).p,
        θ;
        h=1.0e-6,
    )

    @test result.γ == 4.0
    @test size(result.dp) == (length(p0), length(θ))
    @test result.dp ≈ fd atol=1.0e-7 rtol=1.0e-6
    @test vec(sum(result.dp; dims=1)) ≈ zeros(length(θ)) atol=1.0e-10
end

@testset "Model-level gradient propagation" begin
    let
        model = SIModel(2)
        θ = [0.5]
        result = propagate_with_gradient(model, θ, (1, 1), 0.7; tol=1.0e-12, gamma=1.0)
        fd = DifferentiatedUniformization.finite_difference_gradient(
            θ_local -> propagate(model, θ_local, (1, 1), 0.7; tol=1.0e-12, gamma=1.0).p,
            θ;
            h=1.0e-6,
        )

        @test result.p ≈ propagate(model, θ, (1, 1), 0.7; tol=1.0e-12, gamma=1.0).p
        @test result.dp ≈ fd atol=1.0e-7 rtol=1.0e-6
    end

    let
        model = SISModel(2)
        θ = [0.5, 1.5]
        result = propagate_with_gradient(model, θ, (1, 1), 0.4; tol=1.0e-12, gamma=4.0)
        fd = DifferentiatedUniformization.finite_difference_gradient(
            θ_local -> propagate(model, θ_local, (1, 1), 0.4; tol=1.0e-12, gamma=4.0).p,
            θ;
            h=1.0e-6,
        )

        @test result.dp ≈ fd atol=1.0e-7 rtol=1.0e-6
    end

    let
        model = SIRModel(2)
        θ = [0.5, 1.5]
        result = propagate_with_gradient(model, θ, (1, 1, 0), 0.4; tol=1.0e-12, gamma=4.0)
        fd = DifferentiatedUniformization.finite_difference_gradient(
            θ_local -> propagate(model, θ_local, (1, 1, 0), 0.4; tol=1.0e-12, gamma=4.0).p,
            θ;
            h=1.0e-6,
        )

        @test result.dp ≈ fd atol=1.0e-7 rtol=1.0e-6
        @test vec(sum(result.dp; dims=1)) ≈ zeros(length(θ)) atol=1.0e-10
    end
end
