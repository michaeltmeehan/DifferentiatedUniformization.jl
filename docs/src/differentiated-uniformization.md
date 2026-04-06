# Differentiated Uniformization

Differentiated uniformization extends the propagation workflow from
probabilities to first derivatives with respect to model parameters.

## What It Produces

The current implementation computes:

- the propagated probability vector `p(t)`
- the matrix of derivatives `dp/dθ`

with one gradient column per parameter.

## Current API

Low-level usage:

```julia
result = differentiate_uniformize(Q, dQ, t, p0; gamma=..., tol=...)
```

High-level model usage:

```julia
result = propagate_with_gradient(model, theta, x0, t; gamma=..., tol=...)
```

The returned [`DUGradientResult`](/C:/Users/jc213439/Dropbox/dev/DifferentiatedUniformization/src/core/types.jl)
contains:

- `p`: propagated probability vector
- `dp`: derivative matrix with one column per parameter
- `n_terms`: number of Poisson terms used
- `γ`: chosen or supplied uniformization rate
- `tail_mass_bound`: truncation diagnostic

## Why It Matters

This is the main reason the package exists: it gives a direct route to
likelihood gradients for calibration without relying on generic AD through the
propagator.

## Gamma Convention

The most important current caveat is the gamma convention:

- if `gamma` is omitted, it is chosen automatically from `Q`
- that chosen value is then treated as fixed inside the differentiated call
- the returned derivative does not include sensitivity of the gamma-selection rule

For optimization, finite differences, or HMC-style workflows, it is usually
better to pass a fixed `gamma`.

## Example

```julia
using DifferentiatedUniformization

model = SISModel(2)
theta = [0.5, 1.5]

result = propagate_with_gradient(model, theta, (1, 1), 0.4; gamma=4.0)

result.p
result.dp
```
