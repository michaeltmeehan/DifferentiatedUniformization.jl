"""
    differentiate_uniformize(Q, dQ, t, p0; tol=1e-12, gamma=nothing, γ=nothing, max_terms=nothing)

Compute transient probabilities and first derivatives for a finite-state CTMC
using differentiated uniformization.

When `gamma` is omitted, it is chosen from `Q` and then held fixed during the
derivative propagation. This keeps the implementation explicit and matches the
piecewise-smooth behavior of the current finite-state validation workflow.
"""
function differentiate_uniformize(
    Q,
    dQ,
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

    dQ_ops = collect(dQ)
    n_params = length(dQ_ops)
    for dQ_j in dQ_ops
        size(dQ_j) == size(Q) || throw(ArgumentError("each generator derivative must match the size of Q"))
    end

    gamma_value = choose_uniformization_rate(Q; gamma=gamma, γ=γ)
    if gamma_value == 0.0 || t == 0
        return DUGradientResult(copy(p0_vec), zeros(length(p0_vec), n_params), 1, gamma_value, 0.0)
    end

    max_exit_rate = maximum(-Float64.(diag(Q)))
    gamma_value + sqrt(eps(Float64)) >= max_exit_rate ||
        throw(ArgumentError("uniformization rate gamma must be at least the maximum exit rate"))

    lambda = gamma_value * Float64(t)
    n_terms, tail_mass_bound = choose_truncation_terms(lambda; tol=tol, max_terms=max_terms)

    n_states = size(Q, 1)
    P = sparse(1:n_states, 1:n_states, ones(Float64, n_states), n_states, n_states) + Q / gamma_value
    dP = [dQ_j / gamma_value for dQ_j in dQ_ops]

    state_term = copy(p0_vec)
    gradient_terms = [zeros(length(p0_vec)) for _ in 1:n_params]
    weight = exp(-lambda)
    p = weight .* state_term
    dp = zeros(length(p0_vec), n_params)

    for j in 1:n_params
        dp[:, j] .= weight .* gradient_terms[j]
    end

    for n in 1:(n_terms - 1)
        previous_state = state_term
        previous_gradients = gradient_terms

        state_term = P * previous_state
        gradient_terms = [
            dP[j] * previous_state + P * previous_gradients[j] for j in 1:n_params
        ]

        weight *= lambda / n
        p .+= weight .* state_term
        for j in 1:n_params
            dp[:, j] .+= weight .* gradient_terms[j]
        end
    end

    return DUGradientResult(p, dp, n_terms, gamma_value, tail_mass_bound)
end

"""
    differentiate_uniformize((Q, p0), dQ, t; tol=1e-12, gamma=nothing, γ=nothing, max_terms=nothing)

Tuple-based convenience overload matching the call path used by
`propagate_with_gradient`.
"""
function differentiate_uniformize(
    Q_and_p0::Tuple,
    dQ,
    t;
    tol::Real=default_tail_tolerance(),
    gamma=nothing,
    γ=nothing,
    max_terms=nothing,
)
    length(Q_and_p0) == 2 || throw(ArgumentError("differentiate_uniformize tuple input must be (Q, p0)"))
    Q, p0 = Q_and_p0
    return differentiate_uniformize(Q, dQ, t, p0; tol=tol, gamma=gamma, γ=γ, max_terms=max_terms)
end

"""
    propagate_with_gradient(model, θ, p0, t; tol=1e-12, gamma=nothing, γ=nothing, max_terms=nothing)

High-level transient probability and gradient propagation wrapper for
finite-state CTMC models.
"""
function propagate_with_gradient(
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
    dQ = generator_derivatives(model, θ)
    p0_vec = initial_distribution(model, p0)
    return differentiate_uniformize(Q, dQ, t, p0_vec; tol=tol, gamma=gamma, γ=γ, max_terms=max_terms)
end
