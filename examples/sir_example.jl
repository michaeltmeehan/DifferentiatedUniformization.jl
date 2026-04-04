using DifferentiatedUniformization

model = SIRModel(10)
θ = [0.2, 0.1]
Q = generator(model, θ)
result = propagate(model, θ, (9, 1, 0), 0.5)

println("SIR example with $(length(states(model))) states")
println("Generator size: $(size(Q))")
println("Terms used: $(result.n_terms)")
