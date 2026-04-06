"""
    generator(model, θ)

Return the finite-state generator representation for `model` at parameter vector
`θ`.

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
