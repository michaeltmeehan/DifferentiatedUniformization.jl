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

This model also provides a structured operator backend for propagation and
gradient propagation without materializing the full generator matrix.
"""
struct SIModel <: AbstractCTMCModel
    population_size::Int

    function SIModel(population_size::Integer)
        population_size > 0 || throw(ArgumentError("population_size must be positive"))
        return new(population_size)
    end
end

struct SIStructuredGenerator <: AbstractGeneratorOperator
    population_size::Int
    beta::Float64
end

struct SIStructuredDerivative <: AbstractGeneratorOperator
    population_size::Int
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

function structured_generator_operator(model::SIModel, θ::AbstractVector{<:Real})
    length(θ) == 1 || throw(ArgumentError("SIModel expects 1 parameter ordered as [β]"))
    β = Float64(θ[1])
    β >= 0.0 || throw(ArgumentError("SIModel parameter β must be nonnegative"))
    return SIStructuredGenerator(model.population_size, β)
end

function structured_generator_derivative_operators(model::SIModel, θ::AbstractVector{<:Real})
    length(θ) == 1 || throw(ArgumentError("SIModel expects 1 parameter ordered as [β]"))
    θ[1] >= 0 || throw(ArgumentError("SIModel parameter β must be nonnegative"))
    return [SIStructuredDerivative(model.population_size)]
end

state_dimension(op::SIStructuredGenerator) = op.population_size + 1
state_dimension(op::SIStructuredDerivative) = op.population_size + 1

maximum_exit_rate(op::SIStructuredGenerator) = maximum(Float64[op.beta * (op.population_size - i) * i for i in 0:op.population_size])
maximum_exit_rate(op::SIStructuredDerivative) = maximum(Float64[(op.population_size - i) * i for i in 0:op.population_size])

function apply_operator(op::SIStructuredGenerator, v::AbstractVector)
    length(v) == state_dimension(op) || throw(ArgumentError("vector length does not match SI structured operator dimension"))
    out = zeros(Float64, length(v))
    N = op.population_size
    β = op.beta

    for i in 0:N
        idx = i + 1
        rate = β * (N - i) * i
        out[idx] -= rate * v[idx]
        if i < N
            out[idx + 1] += rate * v[idx]
        end
    end

    return out
end

function apply_operator(op::SIStructuredDerivative, v::AbstractVector)
    length(v) == state_dimension(op) || throw(ArgumentError("vector length does not match SI structured derivative dimension"))
    out = zeros(Float64, length(v))
    N = op.population_size

    for i in 0:N
        idx = i + 1
        rate = (N - i) * i
        out[idx] -= rate * v[idx]
        if i < N
            out[idx + 1] += rate * v[idx]
        end
    end

    return out
end

function materialize(op::SIStructuredGenerator)
    model = SIModel(op.population_size)
    return generator(model, [op.beta])
end

function materialize(op::SIStructuredDerivative)
    model = SIModel(op.population_size)
    return generator_derivatives(model, [1.0])[1]
end
