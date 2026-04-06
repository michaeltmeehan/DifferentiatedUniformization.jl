# Architecture

The current codebase is intentionally organized around a small explicit core.

## `src/models/`

Model definitions and state-space-specific generator construction.

- `si.jl`
- `sis.jl`
- `sir.jl`

These files define:

- model structs
- deterministic state enumeration
- reference sparse generator matrices
- reference sparse generator derivative matrices
- initial-distribution helpers
- structured generator operators where implemented
- tensor/Kronecker operators where implemented

## `src/core/`

Core propagation and shared numerical infrastructure.

- `types.jl`: result structs and small shared containers
- `states.jl`: model-level state API
- `operators.jl`: generator matrix and operator abstractions
- `truncation.jl`: gamma selection and Poisson truncation helpers
- `uniformization.jl`: transient probability propagation
- `differentiated_uniformization.jl`: transient gradient propagation

The most important architectural split in the current package is:

- explicit sparse generators remain the reference backend
- DU algorithms are written against a small operator interface
- structured and tensor backends plug into that same operator interface

That interface currently needs only a few operations:

- `state_dimension(op)`
- `apply_operator(op, v)`
- `maximum_exit_rate(op)`

This keeps the propagation code backend-agnostic without turning the package
into a generic linear-algebra framework.

## `src/inference/`

Likelihood-facing layer.

- `likelihood.jl`: exact-state path likelihoods and gradients
- `logdensity.jl`: thin package-local log-density wrapper

These functions now accept a `backend` keyword so the same likelihood code can
run against sparse, structured, or tensor propagation paths where available.

## `src/simulation/`

Simulation and empirical-summary helpers.

- `gillespie.jl`: single-trajectory Gillespie simulation, ensembles, and
  empirical state-distribution helpers

The simulation layer still uses the explicit sparse generator representation.
That keeps the stochastic validation path simple and inspectable.

## `src/utils/`

Small utility helpers used mainly for validation.

- probability-vector checks
- finite-difference Jacobian helper

## `examples/`

Small runnable scripts demonstrating:

- basic propagation
- likelihood evaluation
- log-density wrapping
- diagnostic grid scans
- gradient-based estimation
- Gillespie trajectories and ensemble comparisons
- backend agreement examples, including the SIR tensor backend

## `test/`

Small explicit validation tests for:

- state spaces and generators
- sparse-versus-operator backend agreement
- uniformization
- differentiated uniformization
- likelihoods
- log-density wrapper
- calibration workflows
- Gillespie simulation and Monte Carlo agreement
