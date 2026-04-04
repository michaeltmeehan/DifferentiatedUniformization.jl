"""
    SIRModel(population_size)

Finite-state susceptible-infectious-removed epidemic model with a fixed total
population size.

Parameter ordering for `generator(model, θ)`:

- `θ[1] = β`: infection rate for transitions `(S, I, R) -> (S - 1, I + 1, R)`
  with rate `β * S * I`
- `θ[2] = γ`: recovery rate for transitions `(S, I, R) -> (S, I - 1, R + 1)`
  with rate `γ * I`

Generator convention:

- probability vectors are treated as columns
- `dp/dt = Q * p`
- each column of `Q` sums to zero
- off-diagonal entry `Q[to, from]` is the transition rate from `from` to `to`
"""
struct SIRModel <: AbstractCTMCModel
    population_size::Int

    function SIRModel(population_size::Integer)
        population_size > 0 || throw(ArgumentError("population_size must be positive"))
        return new(population_size)
    end
end

"""
    states(model::SIRModel)

Return ordered `(S, I, R)` states with `S + I + R = N`.

Ordering is deterministic: infectious count `I` increases first, and within
each fixed `I`, removed count `R` increases; susceptible count is implied by
`S = N - I - R`.
"""
function states(model::SIRModel)
    N = model.population_size
    state_space = Tuple{Int,Int,Int}[]
    for i in 0:N
        for r in 0:(N - i)
            s = N - i - r
            push!(state_space, (s, i, r))
        end
    end
    return state_space
end

"""
    initial_distribution(model::SIRModel, u0)

Construct a point-mass initial distribution from an `(S, I, R)` state tuple.
"""
function initial_distribution(model::SIRModel, u0::Tuple{<:Integer,<:Integer,<:Integer})
    state = (Int(u0[1]), Int(u0[2]), Int(u0[3]))
    idx = findfirst(isequal(state), states(model))
    idx === nothing && throw(ArgumentError("initial state $(u0) is not in the SIR state space"))
    p0 = zeros(Float64, length(states(model)))
    p0[idx] = 1.0
    return p0
end

function initial_distribution(model::SIRModel, p0::AbstractVector{<:Real})
    length(p0) == length(states(model)) ||
        throw(ArgumentError("initial distribution length does not match state space"))
    return Float64.(collect(p0))
end

function generator(model::SIRModel, θ::AbstractVector{<:Real})
    length(θ) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [β, γ]"))
    β = Float64(θ[1])
    γ = Float64(θ[2])
    β >= 0.0 || throw(ArgumentError("SIRModel parameter β must be nonnegative"))
    γ >= 0.0 || throw(ArgumentError("SIRModel parameter γ must be nonnegative"))

    state_space = states(model)
    index_by_state = Dict(state => idx for (idx, state) in enumerate(state_space))
    n_states = length(state_space)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (from_idx, (s, i, r)) in enumerate(state_space)
        total_rate = 0.0

        if s > 0 && i > 0
            to_state = (s - 1, i + 1, r)
            rate = β * s * i
            to_idx = index_by_state[to_state]
            push!(rows, to_idx)
            push!(cols, from_idx)
            push!(vals, rate)
            total_rate += rate
        end

        if i > 0
            to_state = (s, i - 1, r + 1)
            rate = γ * i
            to_idx = index_by_state[to_state]
            push!(rows, to_idx)
            push!(cols, from_idx)
            push!(vals, rate)
            total_rate += rate
        end

        push!(rows, from_idx)
        push!(cols, from_idx)
        push!(vals, -total_rate)
    end

    return sparse(rows, cols, vals, n_states, n_states)
end
