@testset "Package bootstrap" begin
    @test DifferentiatedUniformization isa Module

    expected_exports = [
        :AbstractCTMCModel,
        :SIModel,
        :SISModel,
        :SIRModel,
        :DUResult,
        :DUGradientResult,
        :ExactStatePath,
        :CTMCTrajectory,
        :CTMCEnsemble,
        :ExactPathLogDensity,
        :UnimplementedDUError,
        :states,
        :generator,
        :generator_derivatives,
        :generator_operator,
        :generator_derivative_operators,
        :initial_distribution,
        :uniformize,
        :differentiate_uniformize,
        :propagate,
        :propagate_with_gradient,
        :loglikelihood,
        :loglikelihood_and_gradient,
        :dimension,
        :logdensity,
        :logdensity_and_gradient,
        :state_at_time,
        :states_on_grid,
        :simulate_gillespie,
        :simulate_ensemble,
        :empirical_terminal_distribution,
        :empirical_state_probabilities,
    ]

    for name in expected_exports
        @test name in names(DifferentiatedUniformization)
        @test isdefined(DifferentiatedUniformization, name)
    end

    result = DUResult([1.0, 0.0], 1, 2.0, 0.0)
    gradient_result = DUGradientResult([1.0, 0.0], zeros(2, 1), 1, 2.0, 0.0)

    @test result.p == [1.0, 0.0]
    @test size(gradient_result.dp) == (2, 1)
end
