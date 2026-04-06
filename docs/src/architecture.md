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
- generator matrices
- generator derivative matrices
- initial-distribution helpers

## `src/core/`

Core propagation and supporting types.

- `types.jl`: result structs and small shared containers
- `states.jl`: model-level state API
- `operators.jl`: generator API
- `truncation.jl`: gamma selection and Poisson truncation helpers
- `uniformization.jl`: transient probability propagation
- `differentiated_uniformization.jl`: transient gradient propagation

## `src/inference/`

Likelihood-facing layer.

- `likelihood.jl`: exact-state path likelihoods and gradients
- `logdensity.jl`: thin package-local log-density wrapper

## `src/simulation/`

Simulation and empirical-summary helpers.

- `gillespie.jl`: single-trajectory Gillespie simulation, ensembles, and
  empirical state-distribution helpers

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

## `test/`

Small explicit validation tests for:

- state spaces and generators
- uniformization
- differentiated uniformization
- likelihoods
- log-density wrapper
- calibration workflows
- Gillespie simulation and Monte Carlo agreement
