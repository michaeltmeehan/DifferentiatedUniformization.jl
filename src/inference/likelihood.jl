"""
    loglikelihood(model, theta, data; tol=1e-12, gamma=nothing, γ=nothing, max_terms=nothing, backend=:sparse)

Evaluate the log-likelihood of a fully observed finite-state CTMC path.

The current data container is `ExactStatePath`, which factorizes the path
likelihood over consecutive exact state observations.

Gradient convention:

- if `gamma` is omitted, it is chosen from the generator and then treated as
  fixed within each differentiated-uniformization call
- derivatives do not include sensitivity of the automatic gamma-selection rule
- when smooth optimizer or finite-difference behavior matters, pass a fixed
  `gamma`

Backend convention:

- `backend=:sparse` uses the reference explicit sparse-matrix generator path
- `backend=:structured` uses a structured generator operator where available
- `backend=:tensor` uses a tensor/Kronecker backend where available
"""
function loglikelihood(
    model::AbstractCTMCModel,
    theta,
    data::ExactStatePath;
    tol::Real=default_tail_tolerance(),
    gamma=nothing,
    γ=nothing,
    max_terms=nothing,
    backend::Symbol=:sparse,
)
    _validate_observed_states(model, data)

    total_loglikelihood = 0.0
    for (from_state, to_state, delta_t) in _transition_intervals(data)
        transition_probability = _state_transition_probability(
            model,
            theta,
            from_state,
            to_state,
            delta_t;
            tol=tol,
            gamma=gamma,
            γ=γ,
            max_terms=max_terms,
            backend=backend,
        )

        if transition_probability <= 0.0
            return -Inf
        end

        total_loglikelihood += log(transition_probability)
    end

    return total_loglikelihood
end

"""
    loglikelihood_and_gradient(model, theta, data; tol=1e-12, gamma=nothing, γ=nothing, max_terms=nothing, backend=:sparse)

Evaluate the log-likelihood of a fully observed finite-state CTMC path together
with its gradient with respect to `theta`.

Gradient convention:

- if `gamma` is omitted, it is chosen from the generator and then treated as
  fixed within each differentiated-uniformization call
- derivatives do not include sensitivity of the automatic gamma-selection rule
- when smooth optimizer or finite-difference behavior matters, pass a fixed
  `gamma`

Backend convention:

- `backend=:sparse` uses the reference explicit sparse-matrix generator path
- `backend=:structured` uses a structured generator operator where available
- `backend=:tensor` uses a tensor/Kronecker backend where available
"""
function loglikelihood_and_gradient(
    model::AbstractCTMCModel,
    theta::AbstractVector{<:Real},
    data::ExactStatePath;
    tol::Real=default_tail_tolerance(),
    gamma=nothing,
    γ=nothing,
    max_terms=nothing,
    backend::Symbol=:sparse,
)
    _validate_observed_states(model, data)

    theta_vec = Float64.(collect(theta))
    total_loglikelihood = 0.0
    gradient = zeros(length(theta_vec))

    for (from_state, to_state, delta_t) in _transition_intervals(data)
        result = propagate_with_gradient(
            model,
            theta_vec,
            from_state,
            delta_t;
            tol=tol,
            gamma=gamma,
            γ=γ,
            max_terms=max_terms,
            backend=backend,
        )
        to_index = _state_index(model, to_state)
        transition_probability = result.p[to_index]

        if transition_probability <= 0.0
            return -Inf, fill(NaN, length(theta_vec))
        end

        total_loglikelihood += log(transition_probability)
        gradient .+= result.dp[to_index, :] ./ transition_probability
    end

    return total_loglikelihood, gradient
end

function _transition_intervals(data::ExactStatePath)
    intervals = Tuple[]
    for k in 2:length(data.states)
        push!(intervals, (data.states[k - 1], data.states[k], data.times[k] - data.times[k - 1]))
    end
    return intervals
end

function _validate_observed_states(model::AbstractCTMCModel, data::ExactStatePath)
    state_space = Set(states(model))
    for state in data.states
        state in state_space || throw(ArgumentError("observed state $(state) is not in the model state space"))
    end
    return true
end

function _state_index(model::AbstractCTMCModel, state)
    idx = findfirst(isequal(state), states(model))
    idx === nothing && throw(ArgumentError("state $(state) is not in the model state space"))
    return idx
end

function _state_transition_probability(
    model::AbstractCTMCModel,
    theta,
    from_state,
    to_state,
    delta_t;
    tol::Real=default_tail_tolerance(),
    gamma=nothing,
    γ=nothing,
    max_terms=nothing,
    backend::Symbol=:sparse,
)
    result = propagate(
        model,
        theta,
        from_state,
        delta_t;
        tol=tol,
        gamma=gamma,
        γ=γ,
        max_terms=max_terms,
        backend=backend,
    )
    return result.p[_state_index(model, to_state)]
end
