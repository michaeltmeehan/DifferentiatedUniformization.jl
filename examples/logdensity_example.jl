using DifferentiatedUniformization

model = SISModel(2)
data = ExactStatePath(
    [(1, 1), (0, 2), (1, 1)],
    [0.0, 0.2, 0.5],
)

problem = ExactPathLogDensity(model, data; gamma=4.0)
θ = [0.5, 1.5]

println("Log-density example")
println("Parameter dimension: $(dimension(problem))")
println("Log density: $(logdensity(problem, θ))")
println("Log density and gradient: $(logdensity_and_gradient(problem, θ))")
