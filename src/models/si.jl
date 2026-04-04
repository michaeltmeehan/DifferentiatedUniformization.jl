"""
    SIModel(population_size)

Finite-state susceptible-infectious epidemic model with a fixed total
population size.

Parameter ordering for `generator(model, θ)` and
`generator_derivatives(model, θ)`:

- `θ[1] = β`: infection rate for transitions `(S, I) -> (S - 1, I + 1)` with
  rate `β * S * I`

Generator convention:

- probability vectors are treated as columns
- `dp/dt = Q * p`
- each column of `Q` sums to zero
- off-diagonal entry `Q[to, from]` is the transition rate from `from` to `to`
"""
struct SIModel <: AbstractCTMCModel
    population_size::Int

    function SIModel(population_size::Integer)
        population_size > 0 || throw(ArgumentError("population_size must be positive"))
        return new(population_size)
    end
end

"""
    states(model::SIModel)

Return ordered `(S, I)` states with `S + I = N`.
"""
function states(model::SIModel)
    N = model.population_size
    return [(N - i, i) for i in 0:N]
end

"""
    initial_distribution(model::SIModel, u0)

Construct a point-mass initial distribution from an `(S, I)` state tuple.
"""
function initial_distribution(model::SIModel, u0::Tuple{<:Integer,<:Integer})
    idx = findfirst(isequal((Int(u0[1]), Int(u0[2]))), states(model))
    idx === nothing && throw(ArgumentError("initial state $(u0) is not in the SI state space"))
    p0 = zeros(Float64, length(states(model)))
    p0[idx] = 1.0
    return p0
end

function initial_distribution(model::SIModel, p0::AbstractVector{<:Real})
    length(p0) == length(states(model)) ||
        throw(ArgumentError("initial distribution length does not match state space"))
    return Float64.(collect(p0))
end

function generator(model::SIModel, θ::AbstractVector{<:Real})
    length(θ) == 1 || throw(ArgumentError("SIModel expects 1 parameter ordered as [β]"))
    β = Float64(θ[1])
    β >= 0.0 || throw(ArgumentError("SIModel parameter β must be nonnegative"))

    state_space = states(model)
    index_by_state = Dict(state => idx for (idx, state) in enumerate(state_space))
    n_states = length(state_space)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (from_idx, (s, i)) in enumerate(state_space)
        total_rate = 0.0

        if s > 0 && i > 0
            to_state = (s - 1, i + 1)
            rate = β * s * i
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

function generator_derivatives(model::SIModel, θ::AbstractVector{<:Real})
    length(θ) == 1 || throw(ArgumentError("SIModel expects 1 parameter ordered as [β]"))
    θ[1] >= 0 || throw(ArgumentError("SIModel parameter β must be nonnegative"))

    state_space = states(model)
    index_by_state = Dict(state => idx for (idx, state) in enumerate(state_space))
    n_states = length(state_space)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (from_idx, (s, i)) in enumerate(state_space)
        total_rate = 0.0

        if s > 0 && i > 0
            to_state = (s - 1, i + 1)
            rate = Float64(s * i)
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

    return [sparse(rows, cols, vals, n_states, n_states)]
end
