# Generator Backends

The package now distinguishes among three generator backends:

- the reference sparse backend
- a matrix-free structured backend
- a tensor/Kronecker backend

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

The structured backend exposes the generator as a linear operator without
requiring a materialized sparse matrix.

```julia
op = generator_operator(model, theta; backend=:structured)
result = propagate(model, theta, x0, t; backend=:structured)
```

The structured backend is currently available for the canonical finite-state
models:

- `SIModel`
- `SISModel`
- `SIRModel`

It supports:

- propagation
- differentiated propagation
- exact-state likelihood calls
- the package-local log-density wrapper

This backend is matrix-free, but it is not tensor/Kronecker-based.

## Tensor Backend

The tensor backend is a separate backend option:

```julia
op = generator_operator(model, theta; backend=:tensor)
result = propagate(model, theta, x0, t; backend=:tensor)
```

The current tensor backend is implemented for `SIRModel` only.

It follows the SIR tensor-product construction from the differentiated
uniformization paper by Rupp et al. The internal representation uses the full
Cartesian `(S, I)` grid `{0, ..., N} x {0, ..., N}` and applies the generator
through tensor terms rather than a materialized full CTMC matrix.

Specifically, it is built from the paper's decomposition

`Q = (beta / N) (S+_inf ⊗ I+_inf) + alpha (S+_rec ⊗ I+_rec) - (beta / N) (S-_inf ⊗ I-_inf) - alpha (S-_rec ⊗ I-_rec)`

with matrix-vector products implemented by reshaping to a 2D `(I, S)` grid and
using Kronecker identities.

The paper writes derivatives with respect to `log alpha` and `log beta`. The
package's public API still uses raw parameter vectors `[beta, gamma]`, so the
tensor backend converts the paper formulas back to raw-parameter derivatives by
the chain rule. The package also preserves its existing infection-rate
convention `beta * S * I`, so the tensor backend uses the paper's tensor
operator structure while matching the sparse reference backend's public
parameterization. This preserves agreement with the sparse and structured
backends.

## Current Tradeoffs

Use `backend=:sparse` when:

- you want the most inspectable representation
- you need matrix-based validation or simulation
- you are checking small-model structure by hand

Use `backend=:structured` when:

- you want a simple matrix-free backend for the canonical models
- you are benchmarking operator application against the sparse reference path
- you do not specifically need the paper's tensor/Kronecker construction

Use `backend=:tensor` when:

- you are working with `SIRModel`
- you want the paper-faithful tensor/Kronecker generator representation
- you want DU and differentiated DU to run without materializing the SIR generator

When you pass a fixed `gamma`, it must still dominate the relevant maximum exit
rate. For reusable scripts or parameter scans, a safe pattern is to compute a
fixed value from the sparse reference generator:

```julia
gamma = 1.05 * DifferentiatedUniformization.maximum_exit_rate(generator(model, theta))
```

## Current Limitation

The tensor backend is intentionally narrow in v0.1:

- it is implemented only for `SIRModel`
- it does not yet expose low-rank tensor formats
- it is not yet generalized to arbitrary CTMCs or reaction networks
