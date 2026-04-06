# Uniformization

Uniformization is the core propagation method currently used for transient
probabilities.

## High-Level Idea

For a finite-state generator `Q`, choose `gamma` large enough that

`gamma >= max exit rate`

and define

`P = I + Q / gamma`.

Then the transient probability vector can be written as a Poisson-weighted sum
over repeated applications of `P`.

## Current Implementation

Use:

```julia
result = uniformize(Q, t, p0; tol=1e-12, gamma=nothing, max_terms=nothing)
```

The returned [`DUResult`](/C:/Users/jc213439/Dropbox/dev/DifferentiatedUniformization/src/core/types.jl)
contains:

- `p`: propagated probability vector
- `n_terms`: number of Poisson terms used
- `γ`: chosen or supplied uniformization rate
- `tail_mass_bound`: truncation diagnostic from the Poisson tail

## Generator Input

`uniformize(...)` can now work with either:

- an explicit generator matrix
- a generator operator backend that supports applying `Q * p`

The model-facing wrapper is:

```julia
result = propagate(model, theta, x0, t; backend=:sparse)
```

Current backends:

- `backend=:sparse`: reference explicit sparse generator path
- `backend=:structured`: structured operator path where implemented
- `backend=:tensor`: tensor/Kronecker operator path where implemented

## Automatic Gamma Selection

If `gamma` is omitted, the package chooses the maximum exit rate under the
package convention.

For explicit matrices this is `maximum(-diag(Q))`. For operator backends it is
the backend's reported `maximum_exit_rate`.

## Truncation

The infinite series is truncated explicitly using the Poisson tail. This makes
the stopping rule deterministic and testable.

## Example

```julia
using DifferentiatedUniformization

model = SIModel(2)
theta = [0.5]
gamma = 1.0

sparse_result = propagate(model, theta, (1, 1), 0.7; gamma=gamma, backend=:sparse)
structured_result = propagate(model, theta, (1, 1), 0.7; gamma=gamma, backend=:structured)

structured_result.p ≈ sparse_result.p
```
