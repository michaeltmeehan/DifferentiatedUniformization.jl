using Random
using DifferentiatedUniformization

model = SISModel(3)
θ = [0.5, 1.0]
rng = MersenneTwister(1234)

path = simulate_gillespie(model, θ, (2, 1), 2.0; rng=rng)

println("Single-trajectory Gillespie example")
println("Jump times: $(path.times)")
println("Visited states: $(path.states)")
println("State at t = 1.0: $(state_at_time(path, 1.0))")
