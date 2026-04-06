using DifferentiatedUniformization

println("Operator backend example")

for (model, theta, x0) in [
    (SIModel(8), [0.3], (7, 1)),
    (SISModel(8), [0.3, 0.8], (7, 1)),
]
    gamma = 1.05 * DifferentiatedUniformization.maximum_exit_rate(generator(model, theta))

    sparse_result = propagate(model, theta, x0, 1.0; gamma=gamma, backend=:sparse)
    structured_result = propagate(model, theta, x0, 1.0; gamma=gamma, backend=:structured)

    println("Model: $(typeof(model))")
    println("State count: $(length(states(model)))")
    println("Fixed gamma: $(gamma)")
    println("Structured agreement: $(structured_result.p ≈ sparse_result.p)")
    println()
end

let
    model = SIRModel(8)
    theta = [0.3, 0.8]
    x0 = (7, 1, 0)
    gamma = 1.05 * DifferentiatedUniformization.maximum_exit_rate(generator(model, theta))

    sparse_result = propagate(model, theta, x0, 1.0; gamma=gamma, backend=:sparse)
    structured_result = propagate(model, theta, x0, 1.0; gamma=gamma, backend=:structured)
    tensor_result = propagate(model, theta, x0, 1.0; gamma=gamma, backend=:tensor)

    println("Model: $(typeof(model))")
    println("State count: $(length(states(model)))")
    println("Fixed gamma: $(gamma)")
    println("Structured agreement: $(structured_result.p ≈ sparse_result.p)")
    println("Tensor agreement: $(tensor_result.p ≈ sparse_result.p)")
    println()
end
