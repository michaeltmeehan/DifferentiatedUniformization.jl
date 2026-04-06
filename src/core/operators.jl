"""
    AbstractGeneratorOperator

Abstract linear-operator interface for finite-state CTMC generators and their
parameter derivatives.

Operator backends must support:

- `state_dimension(op)`
- `apply_operator(op, v)`
- `maximum_exit_rate(op)`

Optional helper methods such as `materialize(op)` may be implemented for
testing, validation, or simulation-oriented workflows.
"""
abstract type AbstractGeneratorOperator end

"""
    SparseGeneratorOperator(matrix)

Thin operator wrapper around an explicit finite-state generator matrix.
"""
struct SparseGeneratorOperator{M<:AbstractMatrix} <: AbstractGeneratorOperator
    matrix::M
end

"""
    generator(model, θ)

Return the reference explicit sparse generator representation for `model` at
parameter vector `θ`.

Convention:

- probability vectors are columns
- `dp/dt = Q * p`
- `Q[to, from]` is the rate from state `from` to state `to`
- columns of `Q` sum to zero
"""
function generator(model::AbstractCTMCModel, θ)
    throw(UnimplementedDUError("generator(::$(typeof(model)), ::$(typeof(θ)))"))
end

"""
    generator_derivatives(model, θ)

Return derivative representations of the model generator with respect to the
ordered parameter vector `θ`.

Each returned derivative matrix uses the same convention as `generator(model, θ)`.
The returned collection order matches the model's documented parameter order.
"""
function generator_derivatives(model::AbstractCTMCModel, θ)
    throw(UnimplementedDUError("generator_derivatives(::$(typeof(model)), ::$(typeof(θ)))"))
end

"""
    generator_operator(model, θ; backend=:sparse)

Return a generator backend suitable for propagation.

Currently supported backends:

- `:sparse`: explicit sparse-matrix reference backend
- `:structured`: matrix-free structured operator backend where available
- `:tensor`: tensor/Kronecker backend where available
"""
function generator_operator(model::AbstractCTMCModel, θ; backend::Symbol=:sparse)
    if backend === :sparse
        return SparseGeneratorOperator(generator(model, θ))
    elseif backend === :structured
        return structured_generator_operator(model, θ)
    elseif backend === :tensor
        return tensor_generator_operator(model, θ)
    else
        throw(ArgumentError("unknown generator backend $(backend)"))
    end
end

"""
    generator_derivative_operators(model, θ; backend=:sparse)

Return derivative-operator backends matching `generator_operator(model, θ; backend=...)`.
"""
function generator_derivative_operators(model::AbstractCTMCModel, θ; backend::Symbol=:sparse)
    if backend === :sparse
        return [SparseGeneratorOperator(dQ) for dQ in generator_derivatives(model, θ)]
    elseif backend === :structured
        return structured_generator_derivative_operators(model, θ)
    elseif backend === :tensor
        return tensor_generator_derivative_operators(model, θ)
    else
        throw(ArgumentError("unknown generator backend $(backend)"))
    end
end

function structured_generator_operator(model::AbstractCTMCModel, θ)
    throw(ArgumentError("structured generator backend is not implemented for model type $(typeof(model))"))
end

function structured_generator_derivative_operators(model::AbstractCTMCModel, θ)
    throw(ArgumentError("structured derivative backend is not implemented for model type $(typeof(model))"))
end

function tensor_generator_operator(model::AbstractCTMCModel, θ)
    throw(ArgumentError("tensor generator backend is not implemented for model type $(typeof(model))"))
end

function tensor_generator_derivative_operators(model::AbstractCTMCModel, θ)
    throw(ArgumentError("tensor derivative backend is not implemented for model type $(typeof(model))"))
end

state_dimension(Q::AbstractMatrix) = size(Q, 1)
state_dimension(op::SparseGeneratorOperator) = size(op.matrix, 1)

apply_operator(Q::AbstractMatrix, v::AbstractVector) = Q * v
apply_operator(op::SparseGeneratorOperator, v::AbstractVector) = op.matrix * v

maximum_exit_rate(Q::AbstractMatrix) = max(0.0, maximum(-Float64.(diag(Q))))
maximum_exit_rate(op::SparseGeneratorOperator) = maximum_exit_rate(op.matrix)

materialize(Q::AbstractMatrix) = Q
materialize(op::SparseGeneratorOperator) = op.matrix
