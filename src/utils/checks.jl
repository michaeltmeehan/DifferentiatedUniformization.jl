"""
    check_probability_vector(p; atol=1e-12)

Validate a probability vector shape and normalization for small deterministic
tests.
"""
function check_probability_vector(p::AbstractVector{<:Real}; atol::Real=1.0e-12)
    any(x -> x < -atol, p) && throw(ArgumentError("probability vectors must be nonnegative up to tolerance"))
    abs(sum(p) - 1.0) <= atol || throw(ArgumentError("probability vector must sum to one up to tolerance"))
    return true
end
