"""
    states(model)

Return the ordered finite state space for `model`.

The ordering returned here is the canonical ordering used everywhere else in the
package:

- generator construction
- initial-distribution indexing
- transient propagation
- likelihood evaluation
"""
function states(model::AbstractCTMCModel)
    throw(UnimplementedDUError("states(::$(typeof(model)))"))
end

"""
    initial_distribution(model, u0)

Convert a user-specified initial condition `u0` into a probability vector for
the ordered state space of `model`.

Current model methods accept either:

- an exact state tuple, converted to a point mass
- an explicit probability vector already aligned with `states(model)`
"""
function initial_distribution(model::AbstractCTMCModel, u0)
    throw(UnimplementedDUError("initial_distribution(::$(typeof(model)), ::$(typeof(u0)))"))
end
