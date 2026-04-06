"""
    SIRModel(population_size)

Finite-state susceptible-infectious-removed epidemic model with a fixed total
population size.

Parameter ordering for `generator(model, θ)` and
`generator_derivatives(model, θ)`:

- `θ[1] = β`: infection rate for transitions `(S, I, R) -> (S - 1, I + 1, R)`
  with rate `β * S * I`
- `θ[2] = γ`: recovery rate for transitions `(S, I - 1, R + 1)`
  with rate `γ * I`

Generator convention:

- probability vectors are treated as columns
- `dp/dt = Q * p`
- each column of `Q` sums to zero
- off-diagonal entry `Q[to, from]` is the transition rate from `from` to `to`

This model also provides a structured operator backend for propagation and
gradient propagation without materializing the full generator matrix.
"""
struct SIRModel <: AbstractCTMCModel
    population_size::Int

    function SIRModel(population_size::Integer)
        population_size > 0 || throw(ArgumentError("population_size must be positive"))
        return new(population_size)
    end
end

struct SIRStructuredGenerator <: AbstractGeneratorOperator
    population_size::Int
    beta::Float64
    gamma::Float64
    state_space::Vector{Tuple{Int,Int,Int}}
    index_by_state::Dict{Tuple{Int,Int,Int},Int}
end

struct SIRStructuredDerivative <: AbstractGeneratorOperator
    population_size::Int
    parameter_index::Int
    state_space::Vector{Tuple{Int,Int,Int}}
    index_by_state::Dict{Tuple{Int,Int,Int},Int}
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

function generator_derivatives(model::SIRModel, θ::AbstractVector{<:Real})
    length(θ) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [β, γ]"))
    θ[1] >= 0 || throw(ArgumentError("SIRModel parameter β must be nonnegative"))
    θ[2] >= 0 || throw(ArgumentError("SIRModel parameter γ must be nonnegative"))

    state_space = states(model)
    index_by_state = Dict(state => idx for (idx, state) in enumerate(state_space))
    n_states = length(state_space)

    rows_beta = Int[]
    cols_beta = Int[]
    vals_beta = Float64[]
    rows_gamma = Int[]
    cols_gamma = Int[]
    vals_gamma = Float64[]

    for (from_idx, (s, i, r)) in enumerate(state_space)
        total_beta = 0.0
        total_gamma = 0.0

        if s > 0 && i > 0
            to_state = (s - 1, i + 1, r)
            rate = Float64(s * i)
            to_idx = index_by_state[to_state]
            push!(rows_beta, to_idx)
            push!(cols_beta, from_idx)
            push!(vals_beta, rate)
            total_beta += rate
        end

        if i > 0
            to_state = (s, i - 1, r + 1)
            rate = Float64(i)
            to_idx = index_by_state[to_state]
            push!(rows_gamma, to_idx)
            push!(cols_gamma, from_idx)
            push!(vals_gamma, rate)
            total_gamma += rate
        end

        push!(rows_beta, from_idx)
        push!(cols_beta, from_idx)
        push!(vals_beta, -total_beta)
        push!(rows_gamma, from_idx)
        push!(cols_gamma, from_idx)
        push!(vals_gamma, -total_gamma)
    end

    return [
        sparse(rows_beta, cols_beta, vals_beta, n_states, n_states),
        sparse(rows_gamma, cols_gamma, vals_gamma, n_states, n_states),
    ]
end

function structured_generator_operator(model::SIRModel, θ::AbstractVector{<:Real})
    length(θ) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [β, γ]"))
    β = Float64(θ[1])
    γ = Float64(θ[2])
    β >= 0.0 || throw(ArgumentError("SIRModel parameter β must be nonnegative"))
    γ >= 0.0 || throw(ArgumentError("SIRModel parameter γ must be nonnegative"))
    state_space = states(model)
    index_by_state = Dict(state => idx for (idx, state) in enumerate(state_space))
    return SIRStructuredGenerator(model.population_size, β, γ, state_space, index_by_state)
end

function structured_generator_derivative_operators(model::SIRModel, θ::AbstractVector{<:Real})
    length(θ) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [β, γ]"))
    θ[1] >= 0 || throw(ArgumentError("SIRModel parameter β must be nonnegative"))
    θ[2] >= 0 || throw(ArgumentError("SIRModel parameter γ must be nonnegative"))
    state_space = states(model)
    index_by_state = Dict(state => idx for (idx, state) in enumerate(state_space))
    return [
        SIRStructuredDerivative(model.population_size, 1, state_space, index_by_state),
        SIRStructuredDerivative(model.population_size, 2, state_space, index_by_state),
    ]
end

state_dimension(op::SIRStructuredGenerator) = length(op.state_space)
state_dimension(op::SIRStructuredDerivative) = length(op.state_space)

function maximum_exit_rate(op::SIRStructuredGenerator)
    max_rate = 0.0
    for (s, i, _) in op.state_space
        max_rate = max(max_rate, op.beta * s * i + op.gamma * i)
    end
    return max_rate
end

function maximum_exit_rate(op::SIRStructuredDerivative)
    max_rate = 0.0
    for (s, i, _) in op.state_space
        rate = op.parameter_index == 1 ? s * i : i
        max_rate = max(max_rate, float(rate))
    end
    return max_rate
end

function apply_operator(op::SIRStructuredGenerator, v::AbstractVector)
    length(v) == state_dimension(op) || throw(ArgumentError("vector length does not match SIR structured operator dimension"))
    out = zeros(Float64, length(v))

    for (from_idx, (s, i, r)) in enumerate(op.state_space)
        infection_rate = s > 0 && i > 0 ? op.beta * s * i : 0.0
        recovery_rate = i > 0 ? op.gamma * i : 0.0
        total_rate = infection_rate + recovery_rate

        out[from_idx] -= total_rate * v[from_idx]

        if infection_rate > 0.0
            to_idx = op.index_by_state[(s - 1, i + 1, r)]
            out[to_idx] += infection_rate * v[from_idx]
        end

        if recovery_rate > 0.0
            to_idx = op.index_by_state[(s, i - 1, r + 1)]
            out[to_idx] += recovery_rate * v[from_idx]
        end
    end

    return out
end

function apply_operator(op::SIRStructuredDerivative, v::AbstractVector)
    length(v) == state_dimension(op) || throw(ArgumentError("vector length does not match SIR structured derivative dimension"))
    out = zeros(Float64, length(v))

    for (from_idx, (s, i, r)) in enumerate(op.state_space)
        if op.parameter_index == 1
            rate = s > 0 && i > 0 ? float(s * i) : 0.0
            out[from_idx] -= rate * v[from_idx]
            if rate > 0.0
                to_idx = op.index_by_state[(s - 1, i + 1, r)]
                out[to_idx] += rate * v[from_idx]
            end
        else
            rate = i > 0 ? float(i) : 0.0
            out[from_idx] -= rate * v[from_idx]
            if rate > 0.0
                to_idx = op.index_by_state[(s, i - 1, r + 1)]
                out[to_idx] += rate * v[from_idx]
            end
        end
    end

    return out
end

function materialize(op::SIRStructuredGenerator)
    model = SIRModel(op.population_size)
    return generator(model, [op.beta, op.gamma])
end

function materialize(op::SIRStructuredDerivative)
    model = SIRModel(op.population_size)
    return generator_derivatives(model, [1.0, 1.0])[op.parameter_index]
end
