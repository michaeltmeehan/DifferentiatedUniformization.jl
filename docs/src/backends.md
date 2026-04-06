# Generator Backends

The package distinguishes between two generator backends:

- the reference sparse backend
- a structured operator backend

## Sparse Backend

The sparse backend is the default everywhere:

```julia
Q = generator(model, theta)
result = propagate(model, theta, x0, t; backend=:sparse)
```

This path is the current ground-truth implementation for:

- small-model validation
- exact matrix comparisons
- simulation support
- debugging and inspection

## Structured Backend

The structured backend exposes the generator as a linear operator instead of
requiring a materialized sparse matrix.

```julia
op = generator_operator(model, theta; backend=:structured)
result = propagate(model, theta, x0, t; backend=:structured)
```

The current structured implementation covers the canonical finite-state models:

- `SIModel`
- `SISModel`
- `SIRModel`

It supports:

- propagation
- differentiated propagation
- exact-state likelihood calls
- the package-local log-density wrapper

The main purpose of this layer is to let the core algorithms apply `Q * p` and
`dQ/dtheta * p` without forcing full generator materialization.

## Current Tradeoffs

Use `backend=:sparse` when:

- you want the most inspectable representation
- you need matrix-based validation or simulation
- you are checking small-model structure by hand

Use `backend=:structured` when:

- you want the same DU workflow without materializing the generator
- you are benchmarking operator application against the sparse reference path
- you are working in propagation- or likelihood-focused code where `Q * p` is
  the only operation needed

When you pass a fixed `gamma`, it must still dominate the model's maximum exit
rate. For reusable scripts or parameter scans, a safe pattern is to compute a
fixed value from the sparse reference generator:

```julia
gamma = 1.05 * DifferentiatedUniformization.maximum_exit_rate(generator(model, theta))
```

## Current Limitation

The structured backend is still intentionally simple. It is a thin
non-materializing operator implementation for the canonical models, not yet a
fully generic reaction-network compiler or a broad tensor/Kronecker framework.
