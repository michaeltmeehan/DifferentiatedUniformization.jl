# DifferentiatedUniformization.jl

`DifferentiatedUniformization.jl` is a Julia package scaffold for finite-state,
time-homogeneous CTMCs with uniformization and differentiated uniformization as
its core numerical targets.

## Status

The repository is currently in the v0.1 bootstrap phase.

Implemented in this scaffold:

- top-level package module and public API surface
- finite-state epidemic model types for SI, SIS, and SIR
- explicit sparse generator construction for SI, SIS, and SIR
- plain uniformization-based propagation for finite-state generators
- exact small-state validation tests against matrix exponential propagation
- placeholder interfaces for gradients, likelihoods, and Gillespie simulation
- baseline tests for loading, constructors, and deferred-method errors

## Next steps

- implement differentiated uniformization and finite-difference gradient checks
