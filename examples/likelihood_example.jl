using DifferentiatedUniformization

model = SISModel(2)
θ = [0.5, 1.5]
gamma = 4.0

data = ExactStatePath(
    [(1, 1), (2, 0), (1, 1)],
    [0.0, 0.2, 0.5],
)

loglik = loglikelihood(model, θ, data; gamma=gamma)
loglik_grad = loglikelihood_and_gradient(model, θ, data; gamma=gamma)

println("Likelihood example")
println("Log-likelihood: $(loglik)")
println("Log-likelihood and gradient: $(loglik_grad)")
