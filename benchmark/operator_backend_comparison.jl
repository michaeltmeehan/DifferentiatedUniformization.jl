using DifferentiatedUniformization

for (model, theta, x0) in [
    (SIModel(40), [0.15], (39, 1)),
    (SISModel(40), [0.15, 0.9], (39, 1)),
    (SIRModel(20), [0.15, 0.9], (19, 1, 0)),
]
    q_sparse = generator(model, theta)
    gamma = 1.05 * DifferentiatedUniformization.maximum_exit_rate(q_sparse)

    sparse_time = @elapsed sparse_result = propagate(model, theta, x0, 1.0; gamma=gamma, backend=:sparse)
    structured_time = @elapsed structured_result = propagate(model, theta, x0, 1.0; gamma=gamma, backend=:structured)

    println("Operator backend comparison for $(typeof(model))")
    println("Sparse generator size: $(size(q_sparse))")
    println("Fixed gamma: $(gamma)")
    println("Structured agreement: $(structured_result.p ≈ sparse_result.p)")
    println("Sparse propagation time (s): $(sparse_time)")
    println("Structured propagation time (s): $(structured_time)")

    if model isa SIRModel
        tensor_time = @elapsed tensor_result = propagate(model, theta, x0, 1.0; gamma=gamma, backend=:tensor)
        println("Tensor agreement: $(tensor_result.p ≈ sparse_result.p)")
        println("Tensor propagation time (s): $(tensor_time)")
    end

    println()
end
