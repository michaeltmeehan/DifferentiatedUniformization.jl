@testset "Gillespie simulation" begin
    model = SIModel(2)
    θ = [0.5]

    path1 = simulate_gillespie(model, θ, (1, 1), 1.0; rng=MersenneTwister(1234))
    path2 = simulate_gillespie(model, θ, (1, 1), 1.0; rng=MersenneTwister(1234))

    @test path1.states == path2.states
    @test path1.times == path2.times
    @test path1.final_time == 1.0
    @test path1.times[1] == 0.0
    @test issorted(path1.times)
    @test all(state in states(model) for state in path1.states)
    @test state_at_time(path1, 0.0) == (1, 1)
    @test state_at_time(path1, 1.0) == path1.states[end]

    grid_states = states_on_grid(path1, [0.0, 0.25, 0.5, 1.0])
    @test length(grid_states) == 4
    @test grid_states[1] == (1, 1)

    absorbing_path = simulate_gillespie(model, θ, (0, 2), (0.0, 1.0); rng=MersenneTwister(7))
    @test absorbing_path.states == [(0, 2)]
    @test absorbing_path.times == [0.0]
    @test state_at_time(absorbing_path, 0.8) == (0, 2)

    ensemble = simulate_ensemble(model, θ, (1, 1), 1.0, 200; rng=MersenneTwister(55), save_paths=true)
    @test length(ensemble.terminal_states) == 200
    @test ensemble.trajectories !== nothing
    @test length(ensemble.trajectories) == 200
    @test ensemble.final_time == 1.0

    terminal_distribution = empirical_terminal_distribution(model, ensemble)
    @test length(terminal_distribution) == length(states(model))
    @test sum(terminal_distribution) ≈ 1.0 atol=1.0e-12
    @test all(x -> x >= 0.0, terminal_distribution)

    time_grid = [0.0, 0.5, 1.0]
    empirical_grid = empirical_state_probabilities(model, ensemble, time_grid)
    @test size(empirical_grid) == (length(states(model)), length(time_grid))
    @test all(sum(empirical_grid; dims=1) .≈ 1.0)

    ensemble_no_paths = simulate_ensemble(model, θ, (1, 1), 1.0, 10; rng=MersenneTwister(55), save_paths=false)
    @test ensemble_no_paths.trajectories === nothing
    @test_throws ArgumentError empirical_state_probabilities(model, ensemble_no_paths, time_grid)

    benchmark_ensemble = simulate_ensemble(model, θ, (1, 1), 1.0, 4000; rng=MersenneTwister(2026), save_paths=false)
    empirical_terminal = empirical_terminal_distribution(model, benchmark_ensemble)
    du_terminal = propagate(model, θ, (1, 1), 1.0; gamma=1.0).p
    @test empirical_terminal ≈ du_terminal atol=0.05
end
