# DifferentiatedUniformization.jl

`DifferentiatedUniformization.jl` is a Julia package for finite-state,
time-homogeneous CTMCs with uniformization and differentiated uniformization as
its core numerical targets.

## Status

Implemented in the current repository:

- top-level package module and public API surface
- finite-state epidemic model types for SI, SIS, and SIR
- explicit sparse generator construction for SI, SIS, and SIR
- plain uniformization-based propagation for finite-state generators
- differentiated uniformization-based gradient propagation
- exact-state path likelihoods for fully observed discrete-time transitions
- a thin package-local log-density wrapper for calibration workflows
- exact small-state validation tests against matrix exponential propagation
- placeholder interface for Gillespie simulation

## Log-density bridge

`ExactPathLogDensity(model, data; ...)` is a small wrapper around the current
exact-state path likelihood layer.

- it stores a model, observed data, and fixed likelihood configuration
- it uses the same parameter ordering as `generator(model, θ)`
- `dimension(problem)` reports the expected parameter vector length
- `logdensity(problem, θ)` and `logdensity_and_gradient(problem, θ)` forward to
  the existing likelihood functions

The gradient convention is the same as the rest of the package:

- if `gamma` is omitted, it is chosen from the generator and then treated as
  fixed within each differentiated-uniformization call
- derivatives do not include sensitivity of the automatic gamma-selection rule
- when smooth optimizer or finite-difference behavior matters, pass a fixed
  `gamma`

A direct `LogDensityProblems.jl` adapter is deferred until the package
interfaces settle a bit more.

## Next steps

- add a direct `LogDensityProblems.jl` adapter once the package interface settles
