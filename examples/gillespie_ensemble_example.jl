using Random
using DifferentiatedUniformization

model = SIModel(2)
θ = [0.5]
gamma = 1.0
rng = MersenneTwister(2026)

ensemble = simulate_ensemble(model, θ, (1, 1), 1.0, 2000; rng=rng, save_paths=false)
empirical = empirical_terminal_distribution(model, ensemble)
du = propagate(model, θ, (1, 1), 1.0; gamma=gamma).p

println("Ensemble Gillespie example")
println("Empirical terminal distribution: $(empirical)")
println("DU terminal distribution: $(du)")
