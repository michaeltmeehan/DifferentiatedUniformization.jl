"""
    simulate_gillespie(model, θ, x0, tspan; rng=Random.default_rng(), max_events=10^6)

Simulate a single finite-state CTMC trajectory with the Gillespie direct method.

The simulation uses the package generator convention:

- probability vectors are columns
- `Q[to, from]` is the rate from state `from` to state `to`
- the total exit rate from a state is `-Q[from, from]`

`tspan` may be either:

- a final time `tfinal`, interpreted as `(0.0, tfinal)`
- an explicit tuple `(t0, tfinal)`

The returned `CTMCTrajectory` stores jump times and entered states. The path is
defined up to `final_time`, remaining constant after the last recorded jump.
"""
function simulate_gillespie(
    model::AbstractCTMCModel,
    θ,
    x0,
    tspan;
    rng=Random.default_rng(),
    max_events::Integer=1_000_000,
)
    max_events >= 0 || throw(ArgumentError("max_events must be nonnegative"))
    t0, tfinal = _normalize_tspan(tspan)

    state_space = states(model)
    state_to_index = Dict(state => idx for (idx, state) in enumerate(state_space))
    haskey(state_to_index, x0) || throw(ArgumentError("initial state $(x0) is not in the model state space"))

    Q = generator(model, θ)
    size(Q, 1) == length(state_space) == size(Q, 2) ||
        throw(ArgumentError("generator size must match the state space size"))

    current_idx = state_to_index[x0]
    current_time = t0
    visited_states = [state_space[current_idx]]
    event_times = [current_time]
    n_events = 0

    while current_time < tfinal
        exit_rate = -Float64(Q[current_idx, current_idx])
        exit_rate >= 0 || throw(ArgumentError("generator has negative exit rate in state index $(current_idx)"))

        if exit_rate == 0
            break
        end

        n_events < max_events || throw(ArgumentError("max_events reached before the time horizon"))

        waiting_time = randexp(rng) / exit_rate
        next_time = current_time + waiting_time
        next_time > tfinal && break

        next_idx = _sample_next_state(rng, Q, current_idx, exit_rate)
        current_time = next_time
        current_idx = next_idx
        push!(event_times, current_time)
        push!(visited_states, state_space[current_idx])
        n_events += 1
    end

    return CTMCTrajectory(visited_states, event_times, tfinal)
end

"""
    state_at_time(trajectory, t)

Return the CTMC state occupied by `trajectory` at time `t`.
"""
function state_at_time(trajectory::CTMCTrajectory, t::Real)
    t < trajectory.times[1] && throw(ArgumentError("query time is before the trajectory start time"))
    t > trajectory.final_time && throw(ArgumentError("query time is after the trajectory final_time"))
    idx = searchsortedlast(trajectory.times, t)
    return trajectory.states[idx]
end

"""
    states_on_grid(trajectory, time_grid)

Return the states occupied by `trajectory` at each time in `time_grid`.
"""
function states_on_grid(trajectory::CTMCTrajectory, time_grid::AbstractVector{<:Real})
    return [state_at_time(trajectory, t) for t in time_grid]
end

"""
    simulate_ensemble(model, θ, x0, tspan, n; rng=Random.default_rng(), save_paths=true, max_events=10^6)

Simulate `n` CTMC trajectories serially using a shared RNG stream.

If `save_paths=false`, only terminal states are retained. Reproducibility is
entirely determined by the supplied `rng` and the serial simulation order.
"""
function simulate_ensemble(
    model::AbstractCTMCModel,
    θ,
    x0,
    tspan,
    n::Integer;
    rng=Random.default_rng(),
    save_paths::Bool=true,
    max_events::Integer=1_000_000,
)
    n >= 1 || throw(ArgumentError("ensemble size n must be at least 1"))
    t0, tfinal = _normalize_tspan(tspan)
    trajectories = save_paths ? CTMCTrajectory[] : nothing
    terminal_states = Vector{typeof(x0)}(undef, n)

    for i in 1:n
        path = simulate_gillespie(model, θ, x0, (t0, tfinal); rng=rng, max_events=max_events)
        terminal_states[i] = path.states[end]
        if save_paths
            push!(trajectories, path)
        end
    end

    return CTMCEnsemble(terminal_states, tfinal, trajectories)
end

"""
    empirical_terminal_distribution(model, ensemble)

Return the empirical terminal-state distribution from a simulated ensemble,
aligned with `states(model)`.
"""
function empirical_terminal_distribution(model::AbstractCTMCModel, ensemble::CTMCEnsemble)
    state_space = states(model)
    state_to_index = Dict(state => idx for (idx, state) in enumerate(state_space))
    counts = zeros(Float64, length(state_space))

    for state in ensemble.terminal_states
        idx = get(state_to_index, state, nothing)
        idx === nothing && throw(ArgumentError("terminal state $(state) is not in the model state space"))
        counts[idx] += 1.0
    end

    return counts ./ length(ensemble.terminal_states)
end

"""
    empirical_state_probabilities(model, ensemble, time_grid)

Return empirical state probabilities on `time_grid`, aligned with `states(model)`.

This requires `ensemble` to have been simulated with `save_paths=true`.
"""
function empirical_state_probabilities(
    model::AbstractCTMCModel,
    ensemble::CTMCEnsemble,
    time_grid::AbstractVector{<:Real},
)
    ensemble.trajectories === nothing &&
        throw(ArgumentError("empirical_state_probabilities requires saved trajectories"))

    state_space = states(model)
    state_to_index = Dict(state => idx for (idx, state) in enumerate(state_space))
    probabilities = zeros(Float64, length(state_space), length(time_grid))

    for path in ensemble.trajectories
        sampled_states = states_on_grid(path, time_grid)
        for (k, state) in enumerate(sampled_states)
            idx = get(state_to_index, state, nothing)
            idx === nothing && throw(ArgumentError("trajectory state $(state) is not in the model state space"))
            probabilities[idx, k] += 1.0
        end
    end

    probabilities ./= length(ensemble.trajectories)
    return probabilities
end

function _normalize_tspan(tspan::Real)
    tspan >= 0 || throw(ArgumentError("final time must be nonnegative"))
    return 0.0, Float64(tspan)
end

function _normalize_tspan(tspan::Tuple{<:Real,<:Real})
    t0 = Float64(tspan[1])
    tfinal = Float64(tspan[2])
    tfinal >= t0 || throw(ArgumentError("tspan must satisfy tfinal >= t0"))
    return t0, tfinal
end

function _sample_next_state(rng, Q, current_idx::Integer, exit_rate::Real)
    threshold = rand(rng) * exit_rate
    cumulative = 0.0

    for next_idx in axes(Q, 1)
        next_idx == current_idx && continue
        rate = Float64(Q[next_idx, current_idx])
        rate <= 0 && continue
        cumulative += rate
        if threshold <= cumulative + sqrt(eps(Float64))
            return next_idx
        end
    end

    throw(ArgumentError("failed to sample a Gillespie transition from state index $(current_idx)"))
end
