@testset "Generator operator backends" begin
    let
        model = SIModel(4)
        theta = [0.7]
        v = [0.1, 0.2, 0.3, 0.15, 0.25]

        Q_sparse = generator(model, theta)
        op_sparse = generator_operator(model, theta; backend=:sparse)
        op_structured = generator_operator(model, theta; backend=:structured)

        @test op_sparse isa SparseGeneratorOperator
        @test op_structured isa AbstractGeneratorOperator
        @test DifferentiatedUniformization.state_dimension(op_structured) == length(states(model))
        @test Matrix(DifferentiatedUniformization.materialize(op_structured)) == Matrix(Q_sparse)
        @test DifferentiatedUniformization.maximum_exit_rate(op_structured) ==
            DifferentiatedUniformization.maximum_exit_rate(Q_sparse)
        @test DifferentiatedUniformization.apply_operator(op_sparse, v) == Q_sparse * v
        @test DifferentiatedUniformization.apply_operator(op_structured, v) ≈ Q_sparse * v

        dQ_sparse = generator_derivatives(model, theta)[1]
        dQ_structured = generator_derivative_operators(model, theta; backend=:structured)[1]
        @test Matrix(DifferentiatedUniformization.materialize(dQ_structured)) == Matrix(dQ_sparse)
        @test DifferentiatedUniformization.apply_operator(dQ_structured, v) ≈ dQ_sparse * v

        err = capture_exception(() -> generator_operator(model, theta; backend=:tensor))
        @test err isa ArgumentError
        @test occursin("tensor generator backend is not implemented", sprint(showerror, err))
    end

    let
        model = SISModel(4)
        theta = [0.7, 0.9]
        v = [0.1, 0.2, 0.3, 0.15, 0.25]

        Q_sparse = generator(model, theta)
        op_structured = generator_operator(model, theta; backend=:structured)

        @test DifferentiatedUniformization.state_dimension(op_structured) == length(states(model))
        @test Matrix(DifferentiatedUniformization.materialize(op_structured)) == Matrix(Q_sparse)
        @test DifferentiatedUniformization.maximum_exit_rate(op_structured) ==
            DifferentiatedUniformization.maximum_exit_rate(Q_sparse)
        @test DifferentiatedUniformization.apply_operator(op_structured, v) ≈ Q_sparse * v

        dQ_sparse = generator_derivatives(model, theta)
        dQ_structured = generator_derivative_operators(model, theta; backend=:structured)
        @test length(dQ_structured) == 2
        @test Matrix(DifferentiatedUniformization.materialize(dQ_structured[1])) == Matrix(dQ_sparse[1])
        @test Matrix(DifferentiatedUniformization.materialize(dQ_structured[2])) == Matrix(dQ_sparse[2])
        @test DifferentiatedUniformization.apply_operator(dQ_structured[1], v) ≈ dQ_sparse[1] * v
        @test DifferentiatedUniformization.apply_operator(dQ_structured[2], v) ≈ dQ_sparse[2] * v

        err = capture_exception(() -> generator_operator(model, theta; backend=:tensor))
        @test err isa ArgumentError
        @test occursin("tensor generator backend is not implemented", sprint(showerror, err))
    end

    let
        model = SIRModel(3)
        theta = [0.6, 0.8]
        v = collect(range(0.05, 0.5; length=length(states(model))))

        Q_sparse = generator(model, theta)
        op_structured = generator_operator(model, theta; backend=:structured)
        op_tensor = generator_operator(model, theta; backend=:tensor)

        @test DifferentiatedUniformization.state_dimension(op_structured) == length(states(model))
        @test DifferentiatedUniformization.state_dimension(op_tensor) == length(states(model))
        @test Matrix(DifferentiatedUniformization.materialize(op_structured)) == Matrix(Q_sparse)
        @test Matrix(DifferentiatedUniformization.materialize(op_tensor)) == Matrix(Q_sparse)
        @test DifferentiatedUniformization.maximum_exit_rate(op_structured) ==
            DifferentiatedUniformization.maximum_exit_rate(Q_sparse)
        @test DifferentiatedUniformization.apply_operator(op_structured, v) ≈ Q_sparse * v
        @test DifferentiatedUniformization.apply_operator(op_tensor, v) ≈ Q_sparse * v

        dQ_sparse = generator_derivatives(model, theta)
        dQ_structured = generator_derivative_operators(model, theta; backend=:structured)
        dQ_tensor = generator_derivative_operators(model, theta; backend=:tensor)
        @test length(dQ_structured) == 2
        @test length(dQ_tensor) == 2
        @test Matrix(DifferentiatedUniformization.materialize(dQ_structured[1])) == Matrix(dQ_sparse[1])
        @test Matrix(DifferentiatedUniformization.materialize(dQ_structured[2])) == Matrix(dQ_sparse[2])
        @test Matrix(DifferentiatedUniformization.materialize(dQ_tensor[1])) == Matrix(dQ_sparse[1])
        @test Matrix(DifferentiatedUniformization.materialize(dQ_tensor[2])) == Matrix(dQ_sparse[2])
        @test DifferentiatedUniformization.apply_operator(dQ_structured[1], v) ≈ dQ_sparse[1] * v
        @test DifferentiatedUniformization.apply_operator(dQ_structured[2], v) ≈ dQ_sparse[2] * v
        @test DifferentiatedUniformization.apply_operator(dQ_tensor[1], v) ≈ dQ_sparse[1] * v
        @test DifferentiatedUniformization.apply_operator(dQ_tensor[2], v) ≈ dQ_sparse[2] * v

        basis = initial_distribution(model, (2, 1, 0))
        tensor_image = DifferentiatedUniformization.apply_operator(op_tensor, basis)
        sparse_image = Q_sparse * basis
        @test tensor_image ≈ sparse_image
        @test tensor_image[findfirst(isequal((1, 2, 0)), states(model))] == 1.2
        @test tensor_image[findfirst(isequal((2, 0, 1)), states(model))] == 0.8
        @test tensor_image[findfirst(isequal((2, 1, 0)), states(model))] == -2.0
    end
end

@testset "Propagation and likelihood backend agreement" begin
    let
        model = SIModel(4)
        theta = [0.4]
        gamma = 3.0
        t = 0.7
        data = ExactStatePath([(3, 1), (2, 2), (1, 3)], [0.0, 0.3, 0.8])

        sparse_result = propagate(model, theta, (3, 1), t; gamma=gamma, backend=:sparse)
        structured_result = propagate(model, theta, (3, 1), t; gamma=gamma, backend=:structured)
        @test structured_result.p ≈ sparse_result.p atol=1.0e-12 rtol=1.0e-12

        sparse_gradient = propagate_with_gradient(model, theta, (3, 1), t; gamma=gamma, backend=:sparse)
        structured_gradient = propagate_with_gradient(model, theta, (3, 1), t; gamma=gamma, backend=:structured)
        @test structured_gradient.p ≈ sparse_gradient.p atol=1.0e-12 rtol=1.0e-12
        @test structured_gradient.dp ≈ sparse_gradient.dp atol=1.0e-10 rtol=1.0e-10

        ll_sparse = loglikelihood(model, theta, data; gamma=gamma, backend=:sparse)
        ll_structured = loglikelihood(model, theta, data; gamma=gamma, backend=:structured)
        @test ll_structured ≈ ll_sparse atol=1.0e-12 rtol=1.0e-12
    end

    let
        model = SISModel(3)
        theta = [0.5, 1.1]
        gamma = 4.0
        t = 0.5
        data = ExactStatePath([(2, 1), (1, 2), (2, 1)], [0.0, 0.2, 0.5])

        sparse_result = propagate(model, theta, (2, 1), t; gamma=gamma, backend=:sparse)
        structured_result = propagate(model, theta, (2, 1), t; gamma=gamma, backend=:structured)
        @test structured_result.p ≈ sparse_result.p atol=1.0e-12 rtol=1.0e-12

        sparse_gradient = propagate_with_gradient(model, theta, (2, 1), t; gamma=gamma, backend=:sparse)
        structured_gradient = propagate_with_gradient(model, theta, (2, 1), t; gamma=gamma, backend=:structured)
        @test structured_gradient.p ≈ sparse_gradient.p atol=1.0e-12 rtol=1.0e-12
        @test structured_gradient.dp ≈ sparse_gradient.dp atol=1.0e-10 rtol=1.0e-10

        ll_sparse = loglikelihood(model, theta, data; gamma=gamma, backend=:sparse)
        ll_structured = loglikelihood(model, theta, data; gamma=gamma, backend=:structured)
        @test ll_structured ≈ ll_sparse atol=1.0e-12 rtol=1.0e-12
    end

    let
        model = SIRModel(3)
        theta = [0.5, 1.0]
        gamma = 4.0
        t = 0.4
        data = ExactStatePath([(2, 1, 0), (1, 2, 0), (1, 1, 1)], [0.0, 0.15, 0.4])

        sparse_result = propagate(model, theta, (2, 1, 0), t; gamma=gamma, backend=:sparse)
        structured_result = propagate(model, theta, (2, 1, 0), t; gamma=gamma, backend=:structured)
        tensor_result = propagate(model, theta, (2, 1, 0), t; gamma=gamma, backend=:tensor)
        @test structured_result.p ≈ sparse_result.p atol=1.0e-12 rtol=1.0e-12
        @test tensor_result.p ≈ sparse_result.p atol=1.0e-12 rtol=1.0e-12

        sparse_gradient = propagate_with_gradient(model, theta, (2, 1, 0), t; gamma=gamma, backend=:sparse)
        structured_gradient = propagate_with_gradient(model, theta, (2, 1, 0), t; gamma=gamma, backend=:structured)
        tensor_gradient = propagate_with_gradient(model, theta, (2, 1, 0), t; gamma=gamma, backend=:tensor)
        @test structured_gradient.p ≈ sparse_gradient.p atol=1.0e-12 rtol=1.0e-12
        @test tensor_gradient.p ≈ sparse_gradient.p atol=1.0e-12 rtol=1.0e-12
        @test structured_gradient.dp ≈ sparse_gradient.dp atol=1.0e-10 rtol=1.0e-10
        @test tensor_gradient.dp ≈ sparse_gradient.dp atol=1.0e-10 rtol=1.0e-10

        ll_sparse = loglikelihood(model, theta, data; gamma=gamma, backend=:sparse)
        ll_structured = loglikelihood(model, theta, data; gamma=gamma, backend=:structured)
        ll_tensor = loglikelihood(model, theta, data; gamma=gamma, backend=:tensor)
        @test ll_structured ≈ ll_sparse atol=1.0e-12 rtol=1.0e-12
        @test ll_tensor ≈ ll_sparse atol=1.0e-12 rtol=1.0e-12

        llg_sparse, grad_sparse = loglikelihood_and_gradient(model, theta, data; gamma=gamma, backend=:sparse)
        llg_structured, grad_structured = loglikelihood_and_gradient(model, theta, data; gamma=gamma, backend=:structured)
        llg_tensor, grad_tensor = loglikelihood_and_gradient(model, theta, data; gamma=gamma, backend=:tensor)
        @test llg_structured ≈ llg_sparse atol=1.0e-12 rtol=1.0e-12
        @test llg_tensor ≈ llg_sparse atol=1.0e-12 rtol=1.0e-12
        @test grad_structured ≈ grad_sparse atol=1.0e-10 rtol=1.0e-10
        @test grad_tensor ≈ grad_sparse atol=1.0e-10 rtol=1.0e-10

        problem = ExactPathLogDensity(model, data; gamma=gamma, backend=:tensor)
        ll_problem, grad_problem = logdensity_and_gradient(problem, theta)
        @test ll_problem ≈ ll_sparse atol=1.0e-12 rtol=1.0e-12
        @test grad_problem ≈ grad_sparse atol=1.0e-10 rtol=1.0e-10
    end
end
