"""
    SIRModel(population_size)

Finite-state susceptible-infectious-removed epidemic model with a fixed total
population size.

Parameter ordering for `generator(model, theta)` and
`generator_derivatives(model, theta)`:

- `theta[1] = beta`: infection rate for transitions `(S, I, R) -> (S - 1, I + 1, R)`
  with rate `beta * S * I`
- `theta[2] = gamma`: recovery rate for transitions `(S, I, R) -> (S, I - 1, R + 1)`
  with rate `gamma * I`

Generator convention:

- probability vectors are treated as columns
- `dp/dt = Q * p`
- each column of `Q` sums to zero
- off-diagonal entry `Q[to, from]` is the transition rate from `from` to `to`

This model provides both a matrix-free structured backend and a tensor backend.
The tensor backend follows the SIR tensor-product construction in the
differentiated-uniformization paper on the full Cartesian `(S, I)` grid.
To preserve the package's established raw-parameter convention
`beta * S * I` and `gamma * I`, the tensor backend uses the paper's operator
structure but scales the infection term to match the existing sparse reference
backend.
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

struct SIRTensorGenerator <: AbstractGeneratorOperator
    population_size::Int
    beta::Float64
    gamma::Float64
    s_plus_inf::SparseMatrixCSC{Float64,Int}
    s_minus_inf::SparseMatrixCSC{Float64,Int}
    s_plus_rec::SparseMatrixCSC{Float64,Int}
    s_minus_rec::SparseMatrixCSC{Float64,Int}
    i_plus_inf::SparseMatrixCSC{Float64,Int}
    i_minus_inf::SparseMatrixCSC{Float64,Int}
    i_plus_rec::SparseMatrixCSC{Float64,Int}
    i_minus_rec::SparseMatrixCSC{Float64,Int}
    grid_indices::Vector{Int}
    state_space::Vector{Tuple{Int,Int,Int}}
end

struct SIRTensorDerivative <: AbstractGeneratorOperator
    population_size::Int
    parameter_index::Int
    s_plus_inf::SparseMatrixCSC{Float64,Int}
    s_minus_inf::SparseMatrixCSC{Float64,Int}
    s_plus_rec::SparseMatrixCSC{Float64,Int}
    s_minus_rec::SparseMatrixCSC{Float64,Int}
    i_plus_inf::SparseMatrixCSC{Float64,Int}
    i_minus_inf::SparseMatrixCSC{Float64,Int}
    i_plus_rec::SparseMatrixCSC{Float64,Int}
    i_minus_rec::SparseMatrixCSC{Float64,Int}
    grid_indices::Vector{Int}
    state_space::Vector{Tuple{Int,Int,Int}}
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

function generator(model::SIRModel, theta::AbstractVector{<:Real})
    length(theta) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [beta, gamma]"))
    beta = Float64(theta[1])
    gamma = Float64(theta[2])
    beta >= 0.0 || throw(ArgumentError("SIRModel parameter beta must be nonnegative"))
    gamma >= 0.0 || throw(ArgumentError("SIRModel parameter gamma must be nonnegative"))

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
            rate = beta * s * i
            to_idx = index_by_state[to_state]
            push!(rows, to_idx)
            push!(cols, from_idx)
            push!(vals, rate)
            total_rate += rate
        end

        if i > 0
            to_state = (s, i - 1, r + 1)
            rate = gamma * i
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

function generator_derivatives(model::SIRModel, theta::AbstractVector{<:Real})
    length(theta) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [beta, gamma]"))
    theta[1] >= 0 || throw(ArgumentError("SIRModel parameter beta must be nonnegative"))
    theta[2] >= 0 || throw(ArgumentError("SIRModel parameter gamma must be nonnegative"))

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

function structured_generator_operator(model::SIRModel, theta::AbstractVector{<:Real})
    length(theta) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [beta, gamma]"))
    beta = Float64(theta[1])
    gamma = Float64(theta[2])
    beta >= 0.0 || throw(ArgumentError("SIRModel parameter beta must be nonnegative"))
    gamma >= 0.0 || throw(ArgumentError("SIRModel parameter gamma must be nonnegative"))
    state_space = states(model)
    index_by_state = Dict(state => idx for (idx, state) in enumerate(state_space))
    return SIRStructuredGenerator(model.population_size, beta, gamma, state_space, index_by_state)
end

function structured_generator_derivative_operators(model::SIRModel, theta::AbstractVector{<:Real})
    length(theta) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [beta, gamma]"))
    theta[1] >= 0 || throw(ArgumentError("SIRModel parameter beta must be nonnegative"))
    theta[2] >= 0 || throw(ArgumentError("SIRModel parameter gamma must be nonnegative"))
    state_space = states(model)
    index_by_state = Dict(state => idx for (idx, state) in enumerate(state_space))
    return [
        SIRStructuredDerivative(model.population_size, 1, state_space, index_by_state),
        SIRStructuredDerivative(model.population_size, 2, state_space, index_by_state),
    ]
end

function tensor_generator_operator(model::SIRModel, theta::AbstractVector{<:Real})
    length(theta) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [beta, gamma]"))
    beta = Float64(theta[1])
    gamma = Float64(theta[2])
    beta >= 0.0 || throw(ArgumentError("SIRModel parameter beta must be nonnegative"))
    gamma >= 0.0 || throw(ArgumentError("SIRModel parameter gamma must be nonnegative"))

    terms = _sir_tensor_terms(model.population_size)
    return SIRTensorGenerator(
        model.population_size,
        beta,
        gamma,
        terms.s_plus_inf,
        terms.s_minus_inf,
        terms.s_plus_rec,
        terms.s_minus_rec,
        terms.i_plus_inf,
        terms.i_minus_inf,
        terms.i_plus_rec,
        terms.i_minus_rec,
        terms.grid_indices,
        terms.state_space,
    )
end

function tensor_generator_derivative_operators(model::SIRModel, theta::AbstractVector{<:Real})
    length(theta) == 2 || throw(ArgumentError("SIRModel expects 2 parameters ordered as [beta, gamma]"))
    theta[1] >= 0 || throw(ArgumentError("SIRModel parameter beta must be nonnegative"))
    theta[2] >= 0 || throw(ArgumentError("SIRModel parameter gamma must be nonnegative"))

    terms = _sir_tensor_terms(model.population_size)
    return [
        SIRTensorDerivative(
            model.population_size,
            1,
            terms.s_plus_inf,
            terms.s_minus_inf,
            terms.s_plus_rec,
            terms.s_minus_rec,
            terms.i_plus_inf,
            terms.i_minus_inf,
            terms.i_plus_rec,
            terms.i_minus_rec,
            terms.grid_indices,
            terms.state_space,
        ),
        SIRTensorDerivative(
            model.population_size,
            2,
            terms.s_plus_inf,
            terms.s_minus_inf,
            terms.s_plus_rec,
            terms.s_minus_rec,
            terms.i_plus_inf,
            terms.i_minus_inf,
            terms.i_plus_rec,
            terms.i_minus_rec,
            terms.grid_indices,
            terms.state_space,
        ),
    ]
end

state_dimension(op::SIRStructuredGenerator) = length(op.state_space)
state_dimension(op::SIRStructuredDerivative) = length(op.state_space)
state_dimension(op::SIRTensorGenerator) = length(op.state_space)
state_dimension(op::SIRTensorDerivative) = length(op.state_space)

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

function maximum_exit_rate(op::SIRTensorGenerator)
    max_rate = 0.0
    for (s, i, _) in op.state_space
        max_rate = max(max_rate, op.beta * s * i + op.gamma * i)
    end
    return max_rate
end

function maximum_exit_rate(op::SIRTensorDerivative)
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

function apply_operator(op::SIRTensorGenerator, v::AbstractVector)
    length(v) == state_dimension(op) || throw(ArgumentError("vector length does not match SIR tensor operator dimension"))
    x = _sir_tensor_embed(v, op.population_size, op.grid_indices)
    y = zeros(Float64, size(x))

    y .+= op.beta .* _apply_tensor_term(op.s_plus_inf, op.i_plus_inf, x)
    y .+= op.gamma .* _apply_tensor_term(op.s_plus_rec, op.i_plus_rec, x)
    y .-= op.beta .* _apply_tensor_term(op.s_minus_inf, op.i_minus_inf, x)
    y .-= op.gamma .* _apply_tensor_term(op.s_minus_rec, op.i_minus_rec, x)

    return _sir_tensor_project(y, op.grid_indices)
end

function apply_operator(op::SIRTensorDerivative, v::AbstractVector)
    length(v) == state_dimension(op) || throw(ArgumentError("vector length does not match SIR tensor derivative dimension"))
    x = _sir_tensor_embed(v, op.population_size, op.grid_indices)
    y = zeros(Float64, size(x))

    if op.parameter_index == 1
        y .+= _apply_tensor_term(op.s_plus_inf, op.i_plus_inf, x)
        y .-= _apply_tensor_term(op.s_minus_inf, op.i_minus_inf, x)
    else
        y .+= _apply_tensor_term(op.s_plus_rec, op.i_plus_rec, x)
        y .-= _apply_tensor_term(op.s_minus_rec, op.i_minus_rec, x)
    end

    return _sir_tensor_project(y, op.grid_indices)
end

function materialize(op::SIRStructuredGenerator)
    model = SIRModel(op.population_size)
    return generator(model, [op.beta, op.gamma])
end

function materialize(op::SIRStructuredDerivative)
    model = SIRModel(op.population_size)
    return generator_derivatives(model, [1.0, 1.0])[op.parameter_index]
end

function materialize(op::SIRTensorGenerator)
    return _materialize_from_operator(op)
end

function materialize(op::SIRTensorDerivative)
    return _materialize_from_operator(op)
end

function _sir_tensor_terms(N::Int)
    n = N + 1
    state_space = states(SIRModel(N))
    grid_indices = [_sir_grid_linear_index(N, s, i) for (s, i, _) in state_space]

    s_plus_inf = sparse(1:(n - 1), 2:n, Float64.(1:N), n, n)
    s_minus_inf = sparse(1:n, 1:n, Float64.(0:N), n, n)
    s_plus_rec = sparse(1:n, 1:n, ones(Float64, n), n, n)
    s_minus_rec = sparse(1:n, 1:n, ones(Float64, n), n, n)

    i_plus_inf = sparse(2:n, 1:(n - 1), Float64.(0:(N - 1)), n, n)
    i_minus_inf = sparse(1:n, 1:n, Float64.(0:N), n, n)
    i_plus_rec = sparse(1:(n - 1), 2:n, Float64.(1:N), n, n)
    i_minus_rec = sparse(1:n, 1:n, Float64.(0:N), n, n)

    return (
        s_plus_inf=s_plus_inf,
        s_minus_inf=s_minus_inf,
        s_plus_rec=s_plus_rec,
        s_minus_rec=s_minus_rec,
        i_plus_inf=i_plus_inf,
        i_minus_inf=i_minus_inf,
        i_plus_rec=i_plus_rec,
        i_minus_rec=i_minus_rec,
        grid_indices=grid_indices,
        state_space=state_space,
    )
end

function _sir_grid_linear_index(N::Int, s::Int, i::Int)
    return LinearIndices((N + 1, N + 1))[i + 1, s + 1]
end

function _sir_tensor_embed(v::AbstractVector, N::Int, grid_indices::AbstractVector{Int})
    x = zeros(Float64, N + 1, N + 1)
    x[grid_indices] = Float64.(collect(v))
    return x
end

function _sir_tensor_project(x::AbstractMatrix, grid_indices::AbstractVector{Int})
    return Float64[x[idx] for idx in grid_indices]
end

function _apply_tensor_term(s_factor, i_factor, x::AbstractMatrix)
    return i_factor * x * transpose(s_factor)
end

function _materialize_from_operator(op::AbstractGeneratorOperator)
    n = state_dimension(op)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for j in 1:n
        basis = zeros(Float64, n)
        basis[j] = 1.0
        image = apply_operator(op, basis)
        for i in 1:n
            if !iszero(image[i])
                push!(rows, i)
                push!(cols, j)
                push!(vals, image[i])
            end
        end
    end

    return sparse(rows, cols, vals, n, n)
end
