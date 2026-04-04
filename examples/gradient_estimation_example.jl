using DifferentiatedUniformization

# Small gradient-based estimation workflow for an SI model on synthetic exact-
# state path data. We optimize a bounded transform of beta so positivity and the
# fixed-gamma uniformization constraint are both respected during the search.

sigmoid(x) = inv(1 + exp(-x))
logit(p) = log(p / (1 - p))

function beta_from_eta(eta, beta_upper)
    return beta_upper * sigmoid(eta)
end

function dbeta_deta(beta, beta_upper)
    return beta * (1 - beta / beta_upper)
end

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
true_theta = [1.4]
gamma = 2.5
beta_upper = 2.4
data = ExactStatePath(
    [(1, 1), (1, 1), (0, 2), (0, 2)],
    [0.0, 0.5, 1.0, 1.5],
)

initial_beta = 0.7
initial_eta = logit(initial_beta / beta_upper)
fit = estimate_beta(initial_eta, model, data; gamma=gamma, beta_upper=beta_upper)

println("Gradient-based estimation example")
println("True theta: $(true_theta)")
println("Initial beta guess: $(initial_beta)")
println("Fixed gamma: $(gamma)")
println("Estimated beta: $(fit.beta)")
println("Final negative log-likelihood: $(fit.objective)")
println("Final transformed-space gradient: $(fit.gradient)")
println("Objective history: $(fit.history)")
