exact_propagation(Q, t, p0) = exp(Matrix(Q) * t) * p0

@testset "Uniformization helpers" begin
    Q = generator(SISModel(2), [0.5, 1.5])

    @test DifferentiatedUniformization.choose_uniformization_rate(Q) ≈ 3.0
    @test DifferentiatedUniformization.choose_uniformization_rate(Q; gamma=4.0) == 4.0
    @test_throws ArgumentError DifferentiatedUniformization.choose_uniformization_rate(Q; gamma=2.0, γ=3.0)

    n_terms_loose, tail_loose = DifferentiatedUniformization.choose_truncation_terms(2.0; tol=1.0e-3)
    n_terms_tight, tail_tight = DifferentiatedUniformization.choose_truncation_terms(2.0; tol=1.0e-10)
    n_terms_capped, tail_capped = DifferentiatedUniformization.choose_truncation_terms(2.0; tol=1.0e-12, max_terms=2)

    @test n_terms_loose <= n_terms_tight
    @test tail_tight <= tail_loose
    @test n_terms_capped == 2
    @test tail_capped > 1.0e-12
    @test tail_capped ≈ DifferentiatedUniformization.poisson_tail_bound(2.0, 1)
end

@testset "Uniformization core behavior" begin
    model = SISModel(2)
    Q = generator(model, [0.5, 1.5])
    p0 = initial_distribution(model, (1, 1))

    result_zero = uniformize(Q, 0.0, p0)
    @test result_zero isa DUResult
    @test result_zero.p == p0
    @test result_zero.n_terms == 1
    @test result_zero.tail_mass_bound == 0.0

    result = uniformize(Q, 0.5, p0; tol=1.0e-12)
    exact = exact_propagation(Q, 0.5, p0)
    @test result.p ≈ exact atol=1.0e-11 rtol=1.0e-10
    @test result.γ ≈ 3.0
    @test result.tail_mass_bound <= 1.0e-12
    @test sum(result.p) ≈ 1.0 atol=1.0e-12
    @test minimum(result.p) >= -1.0e-12

    small_t = 1.0e-6
    small_time = uniformize(Q, small_t, p0; tol=1.0e-14)
    @test small_time.p ≈ p0 + small_t * (Q * p0) atol=1.0e-10 rtol=1.0e-8

    capped = uniformize(Q, 0.5, p0; tol=1.0e-15, max_terms=2)
    @test capped.n_terms == 2
    @test capped.tail_mass_bound > 1.0e-15

    @test_throws ArgumentError uniformize(Q, -0.1, p0)
    @test_throws ArgumentError uniformize(Q, 0.5, p0; gamma=2.5)
    @test_throws ArgumentError uniformize(Q, 0.5, p0; max_terms=0)
end

@testset "Exact validation against matrix exponential" begin
    let
        model = SIModel(2)
        θ = [0.5]
        p0 = initial_distribution(model, (1, 1))
        Q = generator(model, θ)
        result = propagate(model, θ, (1, 1), 0.7; tol=1.0e-12)
        exact = exact_propagation(Q, 0.7, p0)

        @test result.p ≈ exact atol=1.0e-11 rtol=1.0e-10
        @test sum(result.p) ≈ 1.0 atol=1.0e-12
        @test minimum(result.p) >= -1.0e-12
    end

    let
        model = SISModel(2)
        θ = [0.5, 1.5]
        p0 = initial_distribution(model, (1, 1))
        Q = generator(model, θ)
        result = propagate(model, θ, (1, 1), 0.4; tol=1.0e-12)
        exact = exact_propagation(Q, 0.4, p0)

        @test result.p ≈ exact atol=1.0e-11 rtol=1.0e-10
        @test sum(result.p) ≈ 1.0 atol=1.0e-12
        @test minimum(result.p) >= -1.0e-12
    end

    let
        model = SIRModel(2)
        θ = [0.5, 1.5]
        p0 = initial_distribution(model, (1, 1, 0))
        Q = generator(model, θ)
        result = propagate(model, θ, (1, 1, 0), 0.4; tol=1.0e-12)
        exact = exact_propagation(Q, 0.4, p0)

        @test result.p ≈ exact atol=1.0e-11 rtol=1.0e-10
        @test sum(result.p) ≈ 1.0 atol=1.0e-12
        @test minimum(result.p) >= -1.0e-12
    end
end
