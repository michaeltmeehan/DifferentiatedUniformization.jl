@testset "Calibration workflow" begin
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

    @test dimension(problem) == 1
    @test length(grid_logdensities) == length(candidate_grid)
    @test best_theta == true_theta
    @test best_value == maximum(grid_logdensities)
    @test isfinite(best_value)
    @test all(isfinite, best_gradient)
    @test gamma > maximum(candidate_grid)

    fd_gradient = DifferentiatedUniformization.finite_difference_gradient(
        theta_local -> [logdensity(problem, theta_local)],
        best_theta;
        h=1.0e-6,
    )
    @test best_gradient ≈ vec(fd_gradient) atol=1.0e-7 rtol=1.0e-6

    swapped_problem = ExactPathLogDensity(
        SISModel(2),
        ExactStatePath([(1, 1), (0, 2), (1, 1)], [0.0, 0.2, 0.5]);
        gamma=4.0,
    )
    ordered_value = logdensity(swapped_problem, [0.5, 1.5])
    swapped_value = logdensity(swapped_problem, [1.5, 0.5])
    @test ordered_value != swapped_value

    err = capture_exception(() -> logdensity(swapped_problem, [1.5]))
    @test err isa ArgumentError
end
