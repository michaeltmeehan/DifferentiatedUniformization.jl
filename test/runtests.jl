using Test
using LinearAlgebra
using Random
using DifferentiatedUniformization

function capture_exception(f::Function)
    try
        f()
        return nothing
    catch err
        return err
    end
end

include("test_package.jl")
include("test_models_si.jl")
include("test_models_sis.jl")
include("test_models_sir.jl")
include("test_operator_backends.jl")
include("test_uniformization.jl")
include("test_du_gradients.jl")
include("test_benchmarks.jl")
include("test_simulation.jl")
include("test_logdensity.jl")
include("test_calibration.jl")
include("test_gradient_estimation.jl")
