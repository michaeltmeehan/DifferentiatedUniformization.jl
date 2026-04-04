# DifferentiatedUniformization.jl

A Julia package for computing transient probabilities and parameter gradients for **finite-state continuous-time Markov chains (CTMCs)** using **uniformization** and **differentiated uniformization**.

## Motivation

Many stochastic compartmental epidemic models can be written as finite-state CTMCs with generator matrix \(Q(\theta)\), where \(\theta\) is a vector of model parameters. For inference and calibration, one often needs not only transient probabilities
\[
p(t) = \exp(tQ(\theta))p(0),
\]
but also derivatives of those probabilities, or of derived log-likelihoods, with respect to the parameters.

`DifferentiatedUniformization.jl` is intended to provide a Julia-native implementation of:

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

## Planned features

| Feature | Description |
|---|---|
| `uniformize` | Compute transient probabilities |
| `differentiate_uniformize` | Compute transient probabilities and gradients |
| `propagate` | High-level model propagation |
| `propagate_with_gradient` | High-level propagation plus parameter gradients |
| `loglikelihood` | Likelihood evaluation for fully observed exact-state paths |
| `loglikelihood_and_gradient` | Likelihood and gradient for calibration |
| `simulate_gillespie` | Monte Carlo benchmarking and validation |

## Planned canonical models

| Model | Status |
|---|---|
| SI | Planned for v0.1 |
| SIS | Planned for v0.1 |
| SIR | Planned for v0.1 |
| SEIR | Optional stretch goal |

## Package philosophy

The package is intended to be:

| Principle | Meaning |
|---|---|
| Core-first | Get the mathematics and numerics right before broad integration |
| Explicit | Keep model/state/generator structure visible |
| Validation-heavy | Check results against exact and simulation-based references |
| Interoperable | Make outputs easy to use with the Julia inference ecosystem |

## Planned repository structure

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

## Validation strategy

The initial validation plan is:

| Validation type | Comparison |
|---|---|
| Exact transient probabilities | DU vs matrix exponential |
| Exact gradients | DU vs finite differences |
| Simulation validation | DU vs Gillespie Monte Carlo |

## Interoperability goals

The package is intended to plug naturally into Julia calibration workflows by exposing stable likelihood and gradient functions. Initial downstream targets include:

- `Optimization.jl`
- `LogDensityProblems.jl`
- `Turing.jl` / HMC workflows via thin wrappers or examples

The primary gradient engine will be differentiated uniformization itself, rather than relying exclusively on generic automatic differentiation.

## Development roadmap

| Stage | Goal |
|---|---|
| 1 | Package skeleton, CI, formatting, tests |
| 2 | SI/SIS/SIR state-space and generator builders |
| 3 | Uniformization implementation |
| 4 | Differentiated uniformization implementation |
| 5 | Exact validation tests |
| 6 | Gillespie benchmark module |
| 7 | Likelihood and gradient wrappers |
| 8 | HMC / `LogDensityProblems.jl` example |

## Current status

This repository now has a working finite-state core for:

- SI, SIS, and SIR state-space enumeration and sparse generators
- plain uniformization
- differentiated uniformization
- exact-state path likelihoods for fully observed transitions

Current gradient convention:

- automatic gamma selection is treated as fixed within a differentiated-uniformization call
- derivatives do not include sensitivity of the gamma-selection rule
- for smooth finite-difference or optimizer behavior, prefer passing a fixed `γ`

See [`PLAN.md`](./PLAN.md) for the detailed project brief.
