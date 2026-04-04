@testset "Differentiated uniformization placeholders" begin
    model = SISModel(2)

    err = capture_exception(() -> generator_derivatives(model, [0.1, 0.2]))
    @test err isa UnimplementedDUError
    @test occursin("generator_derivatives", sprint(showerror, err))

    err = capture_exception(() -> differentiate_uniformize(nothing, nothing, 1.0))
    @test err isa UnimplementedDUError
    @test occursin("differentiate_uniformize", sprint(showerror, err))

    err = capture_exception(() -> propagate_with_gradient(model, [0.1, 0.2], (1, 1), 1.0))
    @test err isa UnimplementedDUError
    @test occursin("generator_derivatives", sprint(showerror, err))
end
