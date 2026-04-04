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
differentiated uniformization routines.
"""
struct DUGradientResult{T}
    p::Vector{T}
    dp::Matrix{T}
    n_terms::Int
    γ::T
    tail_mass_bound::T
end

"""
    ExactStatePath(states, times)

Container for a fully observed finite-state CTMC path observed at exact times.

- `states[k]` is the observed CTMC state at time `times[k]`
- `times` must be nondecreasing
- `states` and `times` must have the same length and contain at least one entry

This initial likelihood layer factorizes the path likelihood over consecutive
observation intervals.
"""
struct ExactStatePath{S,T<:Real}
    states::Vector{S}
    times::Vector{T}

    function ExactStatePath(states::AbstractVector, times::AbstractVector{<:Real})
        length(states) == length(times) ||
            throw(ArgumentError("states and times must have the same length"))
        length(states) >= 1 || throw(ArgumentError("ExactStatePath must contain at least one observation"))

        time_vec = collect(times)
        all(diff(time_vec) .>= zero(eltype(time_vec))) ||
            throw(ArgumentError("observation times must be nondecreasing"))

        return new{eltype(states),eltype(time_vec)}(collect(states), time_vec)
    end
end
