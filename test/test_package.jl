@testset "Package bootstrap" begin
    @test DifferentiatedUniformization isa Module

    expected_exports = [
        :AbstractCTMCModel,
        :SIModel,
        :SISModel,
        :SIRModel,
        :DUResult,
        :DUGradientResult,
        :UnimplementedDUError,
        :states,
        :generator,
        :generator_derivatives,
        :initial_distribution,
        :uniformize,
        :differentiate_uniformize,
        :propagate,
        :propagate_with_gradient,
        :loglikelihood,
        :loglikelihood_and_gradient,
        :simulate_gillespie,
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
