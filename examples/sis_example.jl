using DifferentiatedUniformization

model = SISModel(10)
θ = [0.2, 0.1]
Q = generator(model, θ)
result = propagate(model, θ, (9, 1), 0.5)

println("SIS example for population size $(model.population_size)")
println("Generator size: $(size(Q))")
println("Terms used: $(result.n_terms)")
