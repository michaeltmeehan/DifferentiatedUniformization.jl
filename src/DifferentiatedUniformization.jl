module DifferentiatedUniformization

using Random
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: diag

export AbstractCTMCModel,
    SIModel,
    SISModel,
    SIRModel,
    DUResult,
    DUGradientResult,
    ExactStatePath,
    CTMCTrajectory,
    CTMCEnsemble,
    ExactPathLogDensity,
    UnimplementedDUError,
    states,
    generator,
    generator_derivatives,
    initial_distribution,
    uniformize,
    differentiate_uniformize,
    propagate,
    propagate_with_gradient,
    loglikelihood,
    loglikelihood_and_gradient,
    dimension,
    logdensity,
    logdensity_and_gradient,
    state_at_time,
    states_on_grid,
    simulate_gillespie,
    simulate_ensemble,
    empirical_terminal_distribution,
    empirical_state_probabilities

include("core/types.jl")
include("core/states.jl")
include("core/operators.jl")
include("core/truncation.jl")
include("core/uniformization.jl")
include("core/differentiated_uniformization.jl")

include("models/si.jl")
include("models/sis.jl")
include("models/sir.jl")

include("inference/likelihood.jl")
include("inference/logdensity.jl")

include("simulation/gillespie.jl")

include("utils/checks.jl")
include("utils/finite_difference.jl")

end # module DifferentiatedUniformization
