"""
    generator(model, θ)

Return the finite-state generator representation for `model` at parameter vector
`θ`.
"""
function generator(model::AbstractCTMCModel, θ)
    throw(UnimplementedDUError("generator(::$(typeof(model)), ::$(typeof(θ)))"))
end

"""
    generator_derivatives(model, θ)

Return derivative representations of the model generator with respect to the
ordered parameter vector `θ`.
"""
function generator_derivatives(model::AbstractCTMCModel, θ)
    throw(UnimplementedDUError("generator_derivatives(::$(typeof(model)), ::$(typeof(θ)))"))
end
