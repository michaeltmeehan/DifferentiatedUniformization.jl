"""
    default_tail_tolerance()

Return the default truncation tolerance used by uniformization methods.
"""
default_tail_tolerance() = 1.0e-12

"""
    choose_uniformization_rate(Q; gamma=nothing, γ=nothing)

Choose an explicit uniformization rate for generator `Q`.

If no rate is supplied, this returns the maximum exit rate
`maximum(-diag(Q))`, which is sufficient under the package convention
`dp/dt = Q * p` with columns summing to zero.
"""
function choose_uniformization_rate(Q; gamma=nothing, γ=nothing)
    gamma_value = _resolve_gamma_argument(gamma, γ)
    if gamma_value !== nothing
        gamma_value >= 0 || throw(ArgumentError("uniformization rate gamma must be nonnegative"))
        return Float64(gamma_value)
    end

    size(Q, 1) == size(Q, 2) || throw(ArgumentError("generator matrix must be square"))
    exit_rates = -Float64.(diag(Q))
    any(!isfinite, exit_rates) && throw(ArgumentError("generator matrix must have finite diagonal entries"))
    return isempty(exit_rates) ? 0.0 : max(0.0, maximum(exit_rates))
end

function _resolve_gamma_argument(gamma, γ)
    if gamma !== nothing && γ !== nothing && gamma != γ
        throw(ArgumentError("received conflicting values for gamma and γ"))
    end
    return gamma === nothing ? γ : gamma
end

"""
    poisson_tail_bound(lambda, n)

Return the Poisson upper-tail mass `Pr(N > n)` for `N ~ Poisson(lambda)`.
"""
function poisson_tail_bound(lambda::Real, n::Integer)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
    n >= 0 || throw(ArgumentError("n must be nonnegative"))

    if lambda == 0
        return 0.0
    end

    lambda_big = big(float(lambda))
    weight = exp(-lambda_big)
    cumulative = weight

    for k in 1:n
        weight *= lambda_big / k
        cumulative += weight
    end

    return Float64(max(big(0.0), big(1.0) - cumulative))
end

"""
    choose_truncation_terms(lambda; tol=1e-12, max_terms=nothing)

Choose the number of Poisson terms to include in the uniformization sum.

Returns `(n_terms, tail_mass_bound)`, where `n_terms` is the number of terms
actually used in the finite sum.
"""
function choose_truncation_terms(
    lambda::Real;
    tol::Real=default_tail_tolerance(),
    max_terms=nothing,
)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
    tol >= 0 || throw(ArgumentError("tol must be nonnegative"))

    if max_terms !== nothing
        max_terms isa Integer || throw(ArgumentError("max_terms must be an integer or nothing"))
        max_terms >= 1 || throw(ArgumentError("max_terms must be at least 1"))
    end

    if lambda == 0
        return 1, 0.0
    end

    lambda_big = big(float(lambda))
    tol_big = big(float(tol))
    weight = exp(-lambda_big)
    cumulative = weight
    highest_power = 0

    while big(1.0) - cumulative > tol_big
        if max_terms !== nothing && highest_power + 1 >= max_terms
            break
        end
        highest_power += 1
        weight *= lambda_big / highest_power
        cumulative += weight
    end

    return highest_power + 1, Float64(max(big(0.0), big(1.0) - cumulative))
end
