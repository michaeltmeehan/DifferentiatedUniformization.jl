# DifferentiatedUniformization.jl

`DifferentiatedUniformization.jl` is a Julia package for finite-state,
time-homogeneous continuous-time Markov chains (CTMCs), with stochastic
epidemic models as the motivating application.

The package currently supports:

- explicit finite-state SI, SIS, and SIR models
- generator construction under a consistent column-vector convention
- transient propagation by uniformization
- first derivatives by differentiated uniformization
- sparse and structured generator backends for the canonical models
- exact-state path likelihoods for fully observed transitions
- a thin package-local log-density wrapper for calibration workflows
- Gillespie simulation, ensemble summaries, and empirical DU comparisons

## What To Read First

- [Getting Started](getting-started.md): the shortest path from model to propagation
- [Calibration Tutorial](calibration-tutorial.md): end-to-end gradient-based estimation on synthetic data
- [Simulation And Benchmarking](simulation.md): Gillespie trajectories, ensembles, and DU comparison
- [Conventions](conventions.md): the probability-vector, generator, and gamma conventions used everywhere

## Documentation Map

- [Overview](overview.md)
- [Scope And Models](scope-and-models.md)
- [Conventions](conventions.md)
- [Generator Backends](backends.md)
- [Uniformization](uniformization.md)
- [Differentiated Uniformization](differentiated-uniformization.md)
- [Exact-State Path Likelihoods](likelihoods.md)
- [Calibration Tutorial](calibration-tutorial.md)
- [Simulation And Benchmarking](simulation.md)
- [Getting Started](getting-started.md)
- [Architecture](architecture.md)
- [Limitations](limitations.md)

## Supported Workflows

Core supported workflows:

1. build a finite-state model such as `SIModel`, `SISModel`, or `SIRModel`
2. inspect the ordered state space with `states(model)`
3. construct `generator(model, θ)` and, when needed, `generator_derivatives(model, θ)`
4. propagate transient probabilities with `propagate(...)`
5. propagate probabilities and gradients with `propagate_with_gradient(...)`
6. evaluate path likelihoods with `loglikelihood(...)` and `loglikelihood_and_gradient(...)`
7. wrap the likelihood as `ExactPathLogDensity(...)` for optimization-style workflows
8. simulate trajectories and ensembles with Gillespie methods for validation and benchmarking

Diagnostic / sanity-check workflows:

- small grid scans over one parameter
- direct inspection of generators and derivative matrices
- sparse-versus-structured backend agreement checks
- exact comparison against matrix exponential calculations on very small systems
- Monte Carlo comparison of empirical state distributions against DU propagation

Deferred workflows:

- hidden-state or partial-observation models
- particle methods, filtering, or smoothing
- direct Turing.jl integration
- a direct external `LogDensityProblems.jl` adapter
- broader structured backends beyond the canonical models

## Important Conventions

- Probability vectors are columns.
- The forward equation is `dp/dt = Q * p`.
- `Q[to, from]` is the transition rate from state `from` to state `to`.
- Columns of `Q` sum to zero.
- If `gamma` is omitted in differentiated propagation, it is selected from `Q`
  and then treated as fixed within that call.
- Derivatives do not include sensitivity of the automatic `gamma` selection rule.

For calibration-oriented work, especially finite differences or gradient-based
optimization, prefer passing a fixed `gamma`.
