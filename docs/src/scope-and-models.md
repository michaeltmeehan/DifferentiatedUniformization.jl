# Scope And Models

## Current Scope

The current implementation is deliberately narrow.

Supported:

- finite-state CTMCs
- time-homogeneous generators
- explicit SI, SIS, and SIR models
- transient propagation
- first derivatives with respect to model parameters
- exact-state path likelihoods for fully observed transitions

Not yet supported:

- infinite-state models
- time-inhomogeneous CTMCs
- hidden-state or partially observed models
- non-Markovian dwell-time models
- particle methods
- Turing.jl-specific integrations

## Canonical Models

### SI

- State space: `(S, I)` with `S + I = N`
- Parameter ordering: `θ = [β]`
- Transition:
  - `(S, I) -> (S - 1, I + 1)` at rate `β S I`

### SIS

- State space: `(S, I)` with `S + I = N`
- Parameter ordering: `θ = [β, γ]`
- Transitions:
  - `(S, I) -> (S - 1, I + 1)` at rate `β S I`
  - `(S, I) -> (S + 1, I - 1)` at rate `γ I`

### SIR

- State space: `(S, I, R)` with `S + I + R = N`
- Parameter ordering: `θ = [β, γ]`
- Transitions:
  - `(S, I, R) -> (S - 1, I + 1, R)` at rate `β S I`
  - `(S, I, R) -> (S, I - 1, R + 1)` at rate `γ I`

## State Ordering

The package uses deterministic orderings so generators, propagated
probabilities, and likelihood calculations all refer to the same indexing.

- `SIModel` and `SISModel`: states are ordered by increasing infectious count
- `SIRModel`: infectious count increases first, and for fixed infectious count,
  removed count increases

Use `states(model)` whenever you need to inspect or verify the ordering.
