# Overview

`DifferentiatedUniformization.jl` is aimed at a narrow but useful problem:
finite-state, time-homogeneous CTMCs where both transient probabilities and
parameter derivatives matter.

The current package is organized around a simple progression:

1. specify a finite-state model
2. build its generator
3. propagate probabilities by uniformization
4. propagate derivatives by differentiated uniformization
5. build exact-state path likelihoods for calibration

The motivating examples are small epidemic compartment models:

- SI
- SIS
- SIR

The package intentionally favors explicit matrices, explicit state ordering, and
small result structs over abstraction-heavy design. That makes the current code
easy to inspect and easy to validate against exact calculations on small state
spaces.

## Current Package Surface

Main public model and propagation API:

- `states(model)`
- `generator(model, θ)`
- `generator_derivatives(model, θ)`
- `initial_distribution(model, u0)`
- `uniformize(Q, t, p0; ...)`
- `differentiate_uniformize(Q, dQ, t, p0; ...)`
- `propagate(model, θ, p0, t; ...)`
- `propagate_with_gradient(model, θ, p0, t; ...)`

Current inference API:

- `ExactStatePath(states, times)`
- `loglikelihood(model, θ, data; ...)`
- `loglikelihood_and_gradient(model, θ, data; ...)`
- `ExactPathLogDensity(model, data; ...)`
- `logdensity(problem, θ)`
- `logdensity_and_gradient(problem, θ)`

Result types:

- `DUResult`
- `DUGradientResult`
