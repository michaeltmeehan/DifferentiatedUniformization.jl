"""
    uniformize(Q, t, p0; tol=1e-12, gamma=nothing, γ=nothing, max_terms=nothing)

Compute transient probabilities for a finite-state CTMC using explicit
uniformization under the column-vector convention `dp/dt = Q * p`.

If omitted, the uniformization rate is chosen as the maximum exit rate
`maximum(-diag(Q))`.

The result is returned as `DUResult`.
"""
function uniformize(
    Q,
    t,
    p0;
    tol::Real=default_tail_tolerance(),
    gamma=nothing,
    γ=nothing,
    max_terms=nothing,
)
    t >= 0 || throw(ArgumentError("time t must be nonnegative"))
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("generator matrix must be square"))

    p0_vec = Float64.(collect(p0))
    size(Q, 1) == length(p0_vec) ||
        throw(ArgumentError("generator dimension must match initial distribution length"))

    gamma_value = choose_uniformization_rate(Q; gamma=gamma, γ=γ)
    if gamma_value == 0.0 || t == 0
        return DUResult(copy(p0_vec), 1, gamma_value, 0.0)
    end

    max_exit_rate = maximum(-Float64.(diag(Q)))
    gamma_value + sqrt(eps(Float64)) >= max_exit_rate ||
        throw(ArgumentError("uniformization rate gamma must be at least the maximum exit rate"))

    lambda = gamma_value * Float64(t)
    n_terms, tail_mass_bound = choose_truncation_terms(lambda; tol=tol, max_terms=max_terms)

    n_states = size(Q, 1)
    P = sparse(1:n_states, 1:n_states, ones(Float64, n_states), n_states, n_states) + Q / gamma_value

    state_term = copy(p0_vec)
    weight = exp(-lambda)
    p = weight .* state_term

    for n in 1:(n_terms - 1)
        state_term = P * state_term
        weight *= lambda / n
        p .+= weight .* state_term
    end

    return DUResult(p, n_terms, gamma_value, tail_mass_bound)
end

"""
    uniformize((Q, p0), t; tol=1e-12, gamma=nothing, γ=nothing, max_terms=nothing)

Tuple-based convenience overload matching the call path used by `propagate`.
"""
function uniformize(
    Q_and_p0::Tuple,
    t;
    tol::Real=default_tail_tolerance(),
    gamma=nothing,
    γ=nothing,
    max_terms=nothing,
)
    length(Q_and_p0) == 2 || throw(ArgumentError("uniformize tuple input must be (Q, p0)"))
    Q, p0 = Q_and_p0
    return uniformize(Q, t, p0; tol=tol, gamma=gamma, γ=γ, max_terms=max_terms)
end

"""
    propagate(model, θ, p0, t; tol=1e-12, gamma=nothing, γ=nothing, max_terms=nothing)

High-level transient probability propagation wrapper for finite-state CTMC
models.

This is the main model-facing propagation entry point. It constructs the model
generator, converts `p0` to a probability vector via `initial_distribution`,
and then calls `uniformize(...)`.
"""
function propagate(
    model::AbstractCTMCModel,
    θ,
    p0,
    t;
    tol::Real=default_tail_tolerance(),
    gamma=nothing,
    γ=nothing,
    max_terms=nothing,
)
    Q = generator(model, θ)
    p0_vec = initial_distribution(model, p0)
    return uniformize(Q, t, p0_vec; tol=tol, gamma=gamma, γ=γ, max_terms=max_terms)
end
