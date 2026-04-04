using DifferentiatedUniformization

# Small diagnostic calibration scan for an SI model on synthetic exact-state
# path data. This is a sanity check / visualization tool, not the main intended
# inference workflow.

model = SIModel(2)
true_theta = [1.4]
candidate_grid = collect(0.6:0.2:2.2)
gamma = 2.5

data = ExactStatePath(
    [(1, 1), (1, 1), (0, 2), (0, 2)],
    [0.0, 0.5, 1.0, 1.5],
)

problem = ExactPathLogDensity(model, data; gamma=gamma)
grid_logdensities = [logdensity(problem, [beta]) for beta in candidate_grid]
best_index = argmax(grid_logdensities)
best_theta = [candidate_grid[best_index]]
best_value, best_gradient = logdensity_and_gradient(problem, best_theta)

println("Calibration example")
println("True theta: $(true_theta)")
println("Fixed gamma used for the scan: $(gamma)")
println("Candidate grid: $(candidate_grid)")
println("Grid log densities: $(grid_logdensities)")
println("Best grid theta: $(best_theta)")
println("Best log density: $(best_value)")
println("Gradient at best grid theta: $(best_gradient)")
