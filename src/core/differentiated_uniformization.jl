"""
    differentiate_uniformize(Q_or_op, dQ_or_ops, t; tol=1e-12, γ=nothing)

Compute transient probabilities and first derivatives using differentiated
uniformization.

This scaffold intentionally leaves the numerical routine unimplemented.
"""
function differentiate_uniformize(
    Q_or_op,
    dQ_or_ops,
    t;
    tol::Real=default_tail_tolerance(),
    γ=nothing,
)
    throw(UnimplementedDUError("differentiate_uniformize"))
end

"""
    propagate_with_gradient(model, θ, p0, t; tol=1e-12, γ=nothing)

High-level transient probability and gradient propagation wrapper for
finite-state CTMC models.
"""
function propagate_with_gradient(
    model::AbstractCTMCModel,
    θ,
    p0,
    t;
    tol::Real=default_tail_tolerance(),
    γ=nothing,
)
    Q = generator(model, θ)
    dQ = generator_derivatives(model, θ)
    p0_vec = initial_distribution(model, p0)
    return differentiate_uniformize((Q, p0_vec), dQ, t; tol=tol, γ=γ)
end
