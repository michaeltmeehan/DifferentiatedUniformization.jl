"""
    ExactPathLogDensity(model, data; tol=1e-12, gamma=nothing, max_terms=nothing)

Thin log-density wrapper around the exact-state path likelihood layer.

This object is intentionally minimal. It stores:

- a finite-state CTMC model
- an `ExactStatePath`
- fixed configuration for the likelihood evaluation

Parameter ordering is exactly the ordering used by `generator(model, θ)` and
`generator_derivatives(model, θ)` for the wrapped model.

For the canonical models currently implemented:

- `SIModel`: `θ = [β]`
- `SISModel`: `θ = [β, γ]`
- `SIRModel`: `θ = [β, γ]`

Gradient convention:

- if `gamma` is omitted, it is chosen from the generator and then treated as
  fixed within each differentiated-uniformization call
- derivatives do not include sensitivity of the automatic gamma-selection rule
- when smooth optimizer or finite-difference behavior matters, pass a fixed
  `gamma`

This wrapper is package-local for now. It intentionally mirrors the conceptual
role of a `LogDensityProblems.jl` object without adding a new dependency yet.
"""
struct ExactPathLogDensity{M<:AbstractCTMCModel,D<:ExactStatePath,T<:Real,G}
    model::M
    data::D
    tol::T
    gamma::G
    max_terms::Union{Nothing,Int}
end

function ExactPathLogDensity(
    model::AbstractCTMCModel,
    data::ExactStatePath;
    tol::Real=default_tail_tolerance(),
    gamma=nothing,
    max_terms=nothing,
)
    tol >= 0 || throw(ArgumentError("tol must be nonnegative"))
    if max_terms !== nothing
        max_terms isa Integer || throw(ArgumentError("max_terms must be an integer or nothing"))
        max_terms >= 1 || throw(ArgumentError("max_terms must be at least 1"))
    end
    return ExactPathLogDensity{typeof(model),typeof(data),typeof(float(tol)),typeof(gamma)}(
        model,
        data,
        float(tol),
        gamma,
        max_terms,
    )
end

"""
    dimension(problem::ExactPathLogDensity)

Return the expected length of the parameter vector for `problem`.

This is a thin convenience helper for optimization-style workflows.
"""
dimension(problem::ExactPathLogDensity) = _parameter_dimension(problem.model)

"""
    logdensity(problem::ExactPathLogDensity, θ)

Evaluate the wrapped exact-state path log-likelihood at parameter vector `θ`.

The parameter ordering is exactly the model ordering used by
`generator(model, θ)`.
"""
function logdensity(problem::ExactPathLogDensity, θ::AbstractVector{<:Real})
    return loglikelihood(
        problem.model,
        θ,
        problem.data;
        tol=problem.tol,
        gamma=problem.gamma,
        max_terms=problem.max_terms,
    )
end

"""
    logdensity_and_gradient(problem::ExactPathLogDensity, θ)

Evaluate the wrapped exact-state path log-likelihood and gradient at parameter
vector `θ`.

The returned gradient uses the same fixed-gamma convention as
`loglikelihood_and_gradient(...)`.
"""
function logdensity_and_gradient(problem::ExactPathLogDensity, θ::AbstractVector{<:Real})
    return loglikelihood_and_gradient(
        problem.model,
        θ,
        problem.data;
        tol=problem.tol,
        gamma=problem.gamma,
        max_terms=problem.max_terms,
    )
end

_parameter_dimension(::SIModel) = 1
_parameter_dimension(::SISModel) = 2
_parameter_dimension(::SIRModel) = 2
_parameter_dimension(model::AbstractCTMCModel) =
    throw(ArgumentError("parameter dimension is not defined for model type $(typeof(model))"))

const LOGDENSITY_TODO =
    "A direct LogDensityProblems.jl adapter is deferred; ExactPathLogDensity provides the same thin wrapper role without adding a dependency yet."
