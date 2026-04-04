"""
    loglikelihood(model, θ, data)

Placeholder log-likelihood interface for downstream calibration workflows.
"""
function loglikelihood(model::AbstractCTMCModel, θ, data)
    throw(UnimplementedDUError("loglikelihood"))
end

"""
    loglikelihood_and_gradient(model, θ, data)

Placeholder joint log-likelihood and gradient interface for downstream
calibration workflows.
"""
function loglikelihood_and_gradient(model::AbstractCTMCModel, θ, data)
    throw(UnimplementedDUError("loglikelihood_and_gradient"))
end
