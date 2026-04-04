@testset "Inference and simulation placeholders" begin
    model = SIRModel(2)

    err = capture_exception(() -> loglikelihood(model, [0.1, 0.2], nothing))
    @test err isa UnimplementedDUError
    @test occursin("loglikelihood", sprint(showerror, err))

    err = capture_exception(() -> loglikelihood_and_gradient(model, [0.1, 0.2], nothing))
    @test err isa UnimplementedDUError
    @test occursin("loglikelihood_and_gradient", sprint(showerror, err))

    err = capture_exception(() -> simulate_gillespie(model, [0.1, 0.2], (2, 0, 0), 1.0, Random.default_rng()))
    @test err isa UnimplementedDUError
    @test occursin("simulate_gillespie", sprint(showerror, err))
end
