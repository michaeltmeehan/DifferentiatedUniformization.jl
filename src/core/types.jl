"""
    AbstractCTMCModel

Abstract supertype for finite-state, time-homogeneous CTMC models supported by
`DifferentiatedUniformization.jl`.
"""
abstract type AbstractCTMCModel end

"""
    UnimplementedDUError(feature)

Error thrown by scaffolded methods whose numerical implementation is deferred.
"""
struct UnimplementedDUError <: Exception
    feature::String
end

function Base.showerror(io::IO, err::UnimplementedDUError)
    print(
        io,
        "Unimplemented functionality: ",
        err.feature,
        ". This is a v0.1 scaffold placeholder; see TODOs in the source.",
    )
end

"""
    DUResult{T}

Result container for transient probability propagation via uniformization.
"""
struct DUResult{T}
    p::Vector{T}
    n_terms::Int
    γ::T
    tail_mass_bound::T
end

"""
    DUGradientResult{T}

Result container for transient probabilities and parameter gradients produced by
differentiate uniformization routines.
"""
struct DUGradientResult{T}
    p::Vector{T}
    dp::Matrix{T}
    n_terms::Int
    γ::T
    tail_mass_bound::T
end
