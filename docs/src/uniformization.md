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

## Automatic Gamma Selection

If `gamma` is omitted, the package chooses

`maximum(-diag(Q))`

which is the maximum exit rate under the package convention.

## Truncation

The infinite series is truncated explicitly using the Poisson tail. This makes
the stopping rule deterministic and testable.

## Example

```julia
using DifferentiatedUniformization

model = SIModel(2)
theta = [0.5]
Q = generator(model, theta)
p0 = initial_distribution(model, (1, 1))

result = uniformize(Q, 0.7, p0; gamma=1.0)
result.p
```
