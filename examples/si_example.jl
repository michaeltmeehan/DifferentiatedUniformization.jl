using DifferentiatedUniformization

model = SIModel(10)
state_space = states(model)
initial_state = (9, 1)
p0 = initial_distribution(model, initial_state)
θ = [0.2]
Q = generator(model, θ)
result = propagate(model, θ, initial_state, 0.5)

println("SI example")
println("Number of states: $(length(state_space))")
println("Initial distribution length: $(length(p0))")
println("Generator size: $(size(Q))")
println("Terms used: $(result.n_terms)")
println("Tail mass bound: $(result.tail_mass_bound)")
