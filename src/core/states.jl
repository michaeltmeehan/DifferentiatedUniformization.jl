"""
    states(model)

Return the ordered finite state space for `model`.
"""
function states(model::AbstractCTMCModel)
    throw(UnimplementedDUError("states(::$(typeof(model)))"))
end

"""
    initial_distribution(model, u0)

Convert a user-specified initial condition `u0` into a probability vector for
the ordered state space of `model`.
"""
function initial_distribution(model::AbstractCTMCModel, u0)
    throw(UnimplementedDUError("initial_distribution(::$(typeof(model)), ::$(typeof(u0)))"))
end
