# DifferentiatedUniformization.jl

A Julia package for computing transient probabilities and parameter gradients for
finite-state continuous-time Markov chains (CTMCs) using uniformization and
differentiated uniformization.

## Motivation

Many stochastic compartmental epidemic models can be written as finite-state
CTMCs with generator matrix `Q(θ)`, where `θ` is a vector of model parameters.
For inference and calibration, one often needs not only transient probabilities

`p(t) = exp(tQ(θ)) p(0)`

but also derivatives of those probabilities, or of derived log-likelihoods,
with respect to the parameters.

`DifferentiatedUniformization.jl` aims to provide a Julia-native implementation
of:

- transient probability propagation via uniformization
- parameter gradient calculation via differentiated uniformization
- validation against exact matrix exponential methods
- benchmarking against Gillespie-style simulation
- clean interfaces for downstream optimization and Bayesian calibration workflows

## Initial scope

Version 0.1 is intentionally narrow.

| Included | Excluded |
|---|---|
| Finite-state CTMCs | Infinite-state models |
| Time-homogeneous generators | Time-inhomogeneous generators |
| SI, SIS, SIR, possibly SEIR models | Non-Markovian dwell times |
| Exact validation on small models | Full ecosystem integrations from day one |
| Gillespie benchmarking | GPU support |

## Current feature set

| Feature | Description |
|---|---|
| `uniformize` | Compute transient probabilities |
| `differentiate_uniformize` | Compute transient probabilities and gradients |
| `propagate` | High-level model propagation |
| `propagate_with_gradient` | High-level propagation plus parameter gradients |
| `loglikelihood` | Likelihood evaluation for fully observed exact-state paths |
| `loglikelihood_and_gradient` | Likelihood and gradient for calibration |
| `ExactPathLogDensity` | Thin log-density wrapper for calibration workflows |
| `simulate_gillespie` | Monte Carlo benchmarking and validation |

## Canonical models

| Model | Status |
|---|---|
| SI | Implemented for v0.1 |
| SIS | Implemented for v0.1 |
| SIR | Implemented for v0.1 |
| SEIR | Optional stretch goal |

## Package philosophy

The package is intended to be:

| Principle | Meaning |
|---|---|
| Core-first | Get the mathematics and numerics right before broad integration |
| Explicit | Keep model/state/generator structure visible |
| Validation-heavy | Check results against exact and simulation-based references |
| Interoperable | Make outputs easy to use with the Julia inference ecosystem |

## Repository structure

```text
src/
  core/
  models/
  inference/
  simulation/
  utils/

test/
examples/
docs/
benchmark/
```

## Interoperability status

The package is intended to plug naturally into Julia calibration workflows by
exposing stable likelihood and gradient functions. Initial downstream targets
include:

- `Optimization.jl`
- `LogDensityProblems.jl`
- `Turing.jl` / HMC workflows via thin wrappers or examples

Current inference bridge:

- `ExactPathLogDensity(model, data; ...)` is a thin package-local log-density wrapper
- it uses the same parameter ordering as `generator(model, θ)`
- a direct `LogDensityProblems.jl` adapter is deferred so the core package can
  stay dependency-light while the interface settles

## Calibration workflow

The intended current route for calibration-oriented use is:

1. construct a finite-state model such as `SIModel`, `SISModel`, or `SIRModel`
2. build an `ExactStatePath` from fully observed states at exact times
3. evaluate `loglikelihood(model, θ, data; ...)` or
   `loglikelihood_and_gradient(model, θ, data; ...)`
4. optionally wrap the problem as `ExactPathLogDensity(model, data; ...)`
5. use the differentiated-uniformization gradient in a gradient-based optimizer
   or optimization loop

The main intended inference workflow is gradient-based estimation, because
differentiated uniformization provides direct likelihood gradients without
relying on generic AD through the propagator.

See [examples/gradient_estimation_example.jl](/C:/Users/jc213439/Dropbox/dev/DifferentiatedUniformization/examples/gradient_estimation_example.jl)
for a small synthetic SI estimation example using a bounded parameter transform
and backtracking gradient descent on the negative log-likelihood.

See [examples/calibration_example.jl](/C:/Users/jc213439/Dropbox/dev/DifferentiatedUniformization/examples/calibration_example.jl)
for a small synthetic SI calibration workflow based on a deterministic grid
search. That scan is best viewed as a diagnostic or visualization tool rather
than the main inference route. When scanning or optimizing over a parameter
range with fixed `gamma`, choose a value that dominates the maximum exit rate
over the whole region.

## Current status

This repository now has a working finite-state core for:

- SI, SIS, and SIR state-space enumeration and sparse generators
- plain uniformization
- differentiated uniformization
- exact-state path likelihoods for fully observed transitions
- a thin log-density wrapper for calibration-oriented workflows
- a small synthetic calibration example on exact-state path data
- a small synthetic gradient-based estimation example using DU gradients
- Gillespie simulation, ensemble summaries, and empirical DU comparisons

Current gradient convention:

- automatic gamma selection is treated as fixed within a
  differentiated-uniformization call
- derivatives do not include sensitivity of the gamma-selection rule
- for smooth finite-difference or optimizer behavior, prefer passing a fixed `gamma`

See [`PLAN.md`](./PLAN.md) for the detailed project brief.
