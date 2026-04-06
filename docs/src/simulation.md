# Simulation And Benchmarking

The simulation layer provides Gillespie-style sample paths for:

- validation against deterministic propagation
- empirical benchmark comparisons
- synthetic data generation

## Current API

Single trajectory:

```julia
path = simulate_gillespie(model, theta, x0, 2.0; rng=MersenneTwister(1234))
```

Useful helpers:

```julia
state_at_time(path, 1.0)
states_on_grid(path, [0.0, 0.5, 1.0, 2.0])
```

Ensemble simulation:

```julia
ensemble = simulate_ensemble(model, theta, x0, 2.0, 1000; rng=MersenneTwister(1234))
```

Empirical summaries:

```julia
empirical_terminal_distribution(model, ensemble)
empirical_state_probabilities(model, ensemble, [0.0, 1.0, 2.0])
```

## Reproducibility

All stochastic entry points take an explicit `rng` keyword. If you pass the
same RNG seed and call the simulator in the same order, you should get the same
sample paths and ensemble summaries.

`simulate_ensemble(...)` currently uses one shared RNG stream serially across
the ensemble. This keeps the first implementation explicit and reproducible.

## Comparison Against DU

For a modest benchmark comparison:

```julia
using Random
using DifferentiatedUniformization

model = SIModel(2)
theta = [0.5]

ensemble = simulate_ensemble(model, theta, (1, 1), 1.0, 2000; rng=MersenneTwister(2026), save_paths=false)
empirical = empirical_terminal_distribution(model, ensemble)
du = propagate(model, theta, (1, 1), 1.0; gamma=1.0).p
```

The empirical distribution should approach the DU result as the ensemble size
grows.

## Intended Role

The simulation layer is intended for:

- validation of deterministic propagation routines
- empirical benchmarking
- synthetic data generation

It is not intended to replace the deterministic DU workflow for likelihood and
gradient evaluation.
