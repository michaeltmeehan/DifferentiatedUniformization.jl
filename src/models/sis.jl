"""
    SISModel(population_size)

Finite-state susceptible-infectious-susceptible epidemic model with a fixed
total population size.

Parameter ordering for `generator(model, θ)` and
`generator_derivatives(model, θ)`:

- `θ[1] = β`: infection rate for transitions `(S, I) -> (S - 1, I + 1)` with
  rate `β * S * I`
- `θ[2] = γ`: recovery rate for transitions `(S, I) -> (S + 1, I - 1)` with
  rate `γ * I`

Generator convention:

- probability vectors are treated as columns
- `dp/dt = Q * p`
- each column of `Q` sums to zero
- off-diagonal entry `Q[to, from]` is the transition rate from `from` to `to`

This model also provides a structured operator backend for propagation and
gradient propagation without materializing the full generator matrix.
"""
struct SISModel <: AbstractCTMCModel
    population_size::Int

    function SISModel(population_size::Integer)
        population_size > 0 || throw(ArgumentError("population_size must be positive"))
        return new(population_size)
    end
end

struct SISStructuredGenerator <: AbstractGeneratorOperator
    population_size::Int
    beta::Float64
    gamma::Float64
end

struct SISStructuredDerivative <: AbstractGeneratorOperator
    population_size::Int
    parameter_index::Int
end

"""
    states(model::SISModel)

Return ordered `(S, I)` states with `S + I = N`.
"""
function states(model::SISModel)
    N = model.population_size
    return [(N - i, i) for i in 0:N]
end

"""
    initial_distribution(model::SISModel, u0)

Construct a point-mass initial distribution from an `(S, I)` state tuple.
"""
function initial_distribution(model::SISModel, u0::Tuple{<:Integer,<:Integer})
    idx = findfirst(isequal((Int(u0[1]), Int(u0[2]))), states(model))
    idx === nothing && throw(ArgumentError("initial state $(u0) is not in the SIS state space"))
    p0 = zeros(Float64, length(states(model)))
    p0[idx] = 1.0
    return p0
end

function initial_distribution(model::SISModel, p0::AbstractVector{<:Real})
    length(p0) == length(states(model)) ||
        throw(ArgumentError("initial distribution length does not match state space"))
    return Float64.(collect(p0))
end

function generator(model::SISModel, θ::AbstractVector{<:Real})
    length(θ) == 2 || throw(ArgumentError("SISModel expects 2 parameters ordered as [β, γ]"))
    β = Float64(θ[1])
    γ = Float64(θ[2])
    β >= 0.0 || throw(ArgumentError("SISModel parameter β must be nonnegative"))
    γ >= 0.0 || throw(ArgumentError("SISModel parameter γ must be nonnegative"))

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

        if i > 0
            to_state = (s + 1, i - 1)
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

function generator_derivatives(model::SISModel, θ::AbstractVector{<:Real})
    length(θ) == 2 || throw(ArgumentError("SISModel expects 2 parameters ordered as [β, γ]"))
    θ[1] >= 0 || throw(ArgumentError("SISModel parameter β must be nonnegative"))
    θ[2] >= 0 || throw(ArgumentError("SISModel parameter γ must be nonnegative"))

    state_space = states(model)
    index_by_state = Dict(state => idx for (idx, state) in enumerate(state_space))
    n_states = length(state_space)

    rows_beta = Int[]
    cols_beta = Int[]
    vals_beta = Float64[]
    rows_gamma = Int[]
    cols_gamma = Int[]
    vals_gamma = Float64[]

    for (from_idx, (s, i)) in enumerate(state_space)
        total_beta = 0.0
        total_gamma = 0.0

        if s > 0 && i > 0
            to_state = (s - 1, i + 1)
            rate = Float64(s * i)
            to_idx = index_by_state[to_state]
            push!(rows_beta, to_idx)
            push!(cols_beta, from_idx)
            push!(vals_beta, rate)
            total_beta += rate
        end

        if i > 0
            to_state = (s + 1, i - 1)
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

function structured_generator_operator(model::SISModel, θ::AbstractVector{<:Real})
    length(θ) == 2 || throw(ArgumentError("SISModel expects 2 parameters ordered as [β, γ]"))
    β = Float64(θ[1])
    γ = Float64(θ[2])
    β >= 0.0 || throw(ArgumentError("SISModel parameter β must be nonnegative"))
    γ >= 0.0 || throw(ArgumentError("SISModel parameter γ must be nonnegative"))
    return SISStructuredGenerator(model.population_size, β, γ)
end

function structured_generator_derivative_operators(model::SISModel, θ::AbstractVector{<:Real})
    length(θ) == 2 || throw(ArgumentError("SISModel expects 2 parameters ordered as [β, γ]"))
    θ[1] >= 0 || throw(ArgumentError("SISModel parameter β must be nonnegative"))
    θ[2] >= 0 || throw(ArgumentError("SISModel parameter γ must be nonnegative"))
    return [
        SISStructuredDerivative(model.population_size, 1),
        SISStructuredDerivative(model.population_size, 2),
    ]
end

state_dimension(op::SISStructuredGenerator) = op.population_size + 1
state_dimension(op::SISStructuredDerivative) = op.population_size + 1

function maximum_exit_rate(op::SISStructuredGenerator)
    max_rate = 0.0
    N = op.population_size
    for i in 0:N
        max_rate = max(max_rate, op.beta * (N - i) * i + op.gamma * i)
    end
    return max_rate
end

function maximum_exit_rate(op::SISStructuredDerivative)
    max_rate = 0.0
    N = op.population_size
    for i in 0:N
        rate = op.parameter_index == 1 ? (N - i) * i : i
        max_rate = max(max_rate, float(rate))
    end
    return max_rate
end

function apply_operator(op::SISStructuredGenerator, v::AbstractVector)
    length(v) == state_dimension(op) || throw(ArgumentError("vector length does not match SIS structured operator dimension"))
    out = zeros(Float64, length(v))
    N = op.population_size

    for i in 0:N
        idx = i + 1
        infection_rate = op.beta * (N - i) * i
        recovery_rate = op.gamma * i
        total_rate = infection_rate + recovery_rate

        out[idx] -= total_rate * v[idx]
        if i < N
            out[idx + 1] += infection_rate * v[idx]
        end
        if i > 0
            out[idx - 1] += recovery_rate * v[idx]
        end
    end

    return out
end

function apply_operator(op::SISStructuredDerivative, v::AbstractVector)
    length(v) == state_dimension(op) || throw(ArgumentError("vector length does not match SIS structured derivative dimension"))
    out = zeros(Float64, length(v))
    N = op.population_size

    for i in 0:N
        idx = i + 1
        if op.parameter_index == 1
            rate = (N - i) * i
            out[idx] -= rate * v[idx]
            if i < N
                out[idx + 1] += rate * v[idx]
            end
        else
            rate = i
            out[idx] -= rate * v[idx]
            if i > 0
                out[idx - 1] += rate * v[idx]
            end
        end
    end

    return out
end

function materialize(op::SISStructuredGenerator)
    model = SISModel(op.population_size)
    return generator(model, [op.beta, op.gamma])
end

function materialize(op::SISStructuredDerivative)
    model = SISModel(op.population_size)
    return generator_derivatives(model, [1.0, 1.0])[op.parameter_index]
end
