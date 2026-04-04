@testset "Likelihood and simulation layers" begin
    si_model = SIModel(2)
    θ_si = [0.5]
    gamma_si = 1.0

    same_state_path = ExactStatePath([(1, 1), (1, 1)], [0.0, 2.0])
    jump_path = ExactStatePath([(1, 1), (0, 2)], [0.0, 2.0])

    @test loglikelihood(si_model, θ_si, same_state_path; gamma=gamma_si) ≈ -1.0 atol=1.0e-12
    @test loglikelihood(si_model, θ_si, jump_path; gamma=gamma_si) ≈ log(1 - exp(-1.0)) atol=1.0e-11

    singleton_path = ExactStatePath([(1, 1)], [0.0])
    ll_singleton, grad_singleton = loglikelihood_and_gradient(si_model, θ_si, singleton_path; gamma=gamma_si)
    @test ll_singleton == 0.0
    @test grad_singleton == zeros(1)

    zero_time_same = ExactStatePath([(1, 1), (1, 1)], [0.0, 0.0])
    zero_time_jump = ExactStatePath([(1, 1), (0, 2)], [0.0, 0.0])
    @test loglikelihood(si_model, θ_si, zero_time_same; gamma=gamma_si) == 0.0
    @test loglikelihood(si_model, θ_si, zero_time_jump; gamma=gamma_si) == -Inf

    ll_jump, grad_jump = loglikelihood_and_gradient(si_model, θ_si, jump_path; gamma=gamma_si)
    fd_jump = DifferentiatedUniformization.finite_difference_gradient(
        θ_local -> [loglikelihood(si_model, θ_local, jump_path; gamma=gamma_si)],
        θ_si;
        h=1.0e-6,
    )
    @test ll_jump ≈ log(1 - exp(-1.0)) atol=1.0e-11
    @test grad_jump ≈ vec(fd_jump) atol=1.0e-7 rtol=1.0e-6

    sis_model = SISModel(2)
    θ_sis = [0.5, 1.5]
    gamma_sis = 4.0
    full_path = ExactStatePath([(1, 1), (0, 2), (1, 1)], [0.0, 0.2, 0.5])
    first_leg = ExactStatePath([(1, 1), (0, 2)], [0.0, 0.2])
    second_leg = ExactStatePath([(0, 2), (1, 1)], [0.2, 0.5])

    @test loglikelihood(sis_model, θ_sis, full_path; gamma=gamma_sis) ≈
        loglikelihood(sis_model, θ_sis, first_leg; gamma=gamma_sis) +
        loglikelihood(sis_model, θ_sis, second_leg; gamma=gamma_sis)

    ll_sis, grad_sis = loglikelihood_and_gradient(sis_model, θ_sis, full_path; gamma=gamma_sis)
    fd_sis = DifferentiatedUniformization.finite_difference_gradient(
        θ_local -> [loglikelihood(sis_model, θ_local, full_path; gamma=gamma_sis)],
        θ_sis;
        h=1.0e-6,
    )
    @test isfinite(ll_sis)
    @test grad_sis ≈ vec(fd_sis) atol=1.0e-7 rtol=1.0e-6

    @test_throws ArgumentError ExactStatePath([(1, 1), (0, 2)], [0.2])
    @test_throws ArgumentError ExactStatePath([(1, 1), (0, 2)], [0.2, 0.1])
    @test_throws ArgumentError loglikelihood(si_model, θ_si, ExactStatePath([(3, 0), (1, 1)], [0.0, 1.0]); gamma=gamma_si)

    impossible_ll, impossible_grad = loglikelihood_and_gradient(si_model, θ_si, zero_time_jump; gamma=gamma_si)
    @test impossible_ll == -Inf
    @test all(isnan, impossible_grad)

    err = capture_exception(() -> simulate_gillespie(sis_model, θ_sis, (1, 1), 1.0, Random.default_rng()))
    @test err isa UnimplementedDUError
    @test occursin("simulate_gillespie", sprint(showerror, err))
end
