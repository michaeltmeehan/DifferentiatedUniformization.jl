"""
Thin compatibility wrappers for `LogDensityProblems.jl` are intentionally
deferred.

The current likelihood layer is stable enough for direct function-based
calibration workflows, but adding an external wrapper now would either require a
new dependency or an additional optional-extension pattern. That is better done
once the parameter and data interfaces settle.
"""
const LOGDENSITY_TODO = "LogDensityProblems-style wrappers are deferred; use loglikelihood and loglikelihood_and_gradient directly for now."
