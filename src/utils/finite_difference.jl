"""
    finite_difference_gradient(f, θ; h=1e-6)

Compute a central-difference approximation to the Jacobian of `f(θ)`.

If `f(θ)` returns a vector of length `m` and `θ` has length `p`, the returned
matrix has shape `(m, p)` with one column per parameter.
"""
function finite_difference_gradient(f, θ::AbstractVector{<:Real}; h::Real=1.0e-6)
    h > 0 || throw(ArgumentError("finite-difference step size h must be positive"))

    θ_vec = Float64.(collect(θ))
    baseline = Float64.(collect(f(θ_vec)))
    jacobian = zeros(length(baseline), length(θ_vec))

    for j in eachindex(θ_vec)
        θ_plus = copy(θ_vec)
        θ_minus = copy(θ_vec)
        θ_plus[j] += h
        θ_minus[j] -= h
        f_plus = Float64.(collect(f(θ_plus)))
        f_minus = Float64.(collect(f(θ_minus)))
        length(f_plus) == length(baseline) == length(f_minus) ||
            throw(ArgumentError("f must return outputs of consistent length"))
        jacobian[:, j] .= (f_plus .- f_minus) ./ (2h)
    end

    return jacobian
end
