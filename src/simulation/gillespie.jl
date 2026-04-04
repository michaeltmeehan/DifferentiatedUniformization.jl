"""
    simulate_gillespie(model, θ, x0, t, rng)

Placeholder Gillespie-style simulation interface for validation and benchmarking.
"""
function simulate_gillespie(model::AbstractCTMCModel, θ, x0, t, rng)
    throw(UnimplementedDUError("simulate_gillespie"))
end
