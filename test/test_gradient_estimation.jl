@testset "Gradient-based estimation workflow" begin
    sigmoid(x) = inv(1 + exp(-x))
    logit(p) = log(p / (1 - p))
    beta_from_eta(eta, beta_upper) = beta_upper * sigmoid(eta)
    dbeta_deta(beta, beta_upper) = beta * (1 - beta / beta_upper)

    function objective_and_gradient(eta, model, data; gamma, beta_upper)
        beta = beta_from_eta(eta, beta_upper)
        loglik, gradient = loglikelihood_and_gradient(model, [beta], data; gamma=gamma)
        chain = dbeta_deta(beta, beta_upper)
        return -loglik, -gradient[1] * chain
    end

    function estimate_beta(initial_eta, model, data; gamma, beta_upper, max_iters=25, tol=1.0e-8)
        eta = initial_eta
        history = Float64[]

        for _ in 1:max_iters
            objective, gradient = objective_and_gradient(eta, model, data; gamma=gamma, beta_upper=beta_upper)
            push!(history, objective)

            abs(gradient) <= tol && break

            step = 1.0
            accepted = false
            while step >= 1.0e-8
                candidate_eta = eta - step * gradient
                candidate_objective, _ = objective_and_gradient(candidate_eta, model, data; gamma=gamma, beta_upper=beta_upper)

                if candidate_objective <= objective - 1.0e-4 * step * gradient^2
                    eta = candidate_eta
                    accepted = true
                    break
                end

                step *= 0.5
            end

            accepted || break
        end

        final_objective, final_gradient = objective_and_gradient(eta, model, data; gamma=gamma, beta_upper=beta_upper)
        return (
            eta=eta,
            beta=beta_from_eta(eta, beta_upper),
            objective=final_objective,
            gradient=final_gradient,
            history=history,
        )
    end

    model = SIModel(2)
    true_beta = 1.4
    gamma = 2.5
    beta_upper = 2.4
    data = ExactStatePath(
        [(1, 1), (1, 1), (0, 2), (0, 2)],
        [0.0, 0.5, 1.0, 1.5],
    )

    initial_beta = 0.7
    initial_eta = logit(initial_beta / beta_upper)
    initial_objective, initial_gradient = objective_and_gradient(initial_eta, model, data; gamma=gamma, beta_upper=beta_upper)
    fit = estimate_beta(initial_eta, model, data; gamma=gamma, beta_upper=beta_upper)

    @test isfinite(initial_objective)
    @test isfinite(initial_gradient)
    @test isfinite(fit.objective)
    @test isfinite(fit.gradient)
    @test fit.objective <= initial_objective
    @test abs(fit.beta - true_beta) < abs(initial_beta - true_beta)
    @test 0.9 <= fit.beta <= 1.8

    fd_gradient = DifferentiatedUniformization.finite_difference_gradient(
        eta_local -> [objective_and_gradient(eta_local[1], model, data; gamma=gamma, beta_upper=beta_upper)[1]],
        [initial_eta];
        h=1.0e-6,
    )
    @test initial_gradient ≈ fd_gradient[1, 1] atol=1.0e-7 rtol=1.0e-6

    problem = ExactPathLogDensity(model, data; gamma=gamma)
    @test logdensity(problem, [true_beta]) >= logdensity(problem, [initial_beta])
end
