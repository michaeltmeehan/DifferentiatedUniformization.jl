@testset "Log-density wrapper" begin
    model = SISModel(2)
    data = ExactStatePath([(1, 1), (0, 2), (1, 1)], [0.0, 0.2, 0.5])
    θ = [0.5, 1.5]
    problem = ExactPathLogDensity(model, data; gamma=4.0, tol=1.0e-12)

    @test dimension(problem) == 2
    @test logdensity(problem, θ) ==
        loglikelihood(model, θ, data; gamma=4.0, tol=1.0e-12)

    ll_problem, grad_problem = logdensity_and_gradient(problem, θ)
    ll_direct, grad_direct = loglikelihood_and_gradient(model, θ, data; gamma=4.0, tol=1.0e-12)
    @test ll_problem == ll_direct
    @test grad_problem == grad_direct

    si_problem = ExactPathLogDensity(
        SIModel(2),
        ExactStatePath([(1, 1), (0, 2)], [0.0, 2.0]);
        gamma=1.0,
    )
    @test dimension(si_problem) == 1
    @test logdensity(si_problem, [0.5]) ≈ loglikelihood(SIModel(2), [0.5], ExactStatePath([(1, 1), (0, 2)], [0.0, 2.0]); gamma=1.0)

    err = capture_exception(() -> logdensity(problem, [0.5]))
    @test err isa ArgumentError
    @test occursin("expects 2 parameters", sprint(showerror, err))
end
