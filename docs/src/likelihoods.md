# Exact-State Path Likelihoods

The current likelihood layer is intentionally narrow: it supports fully
observed exact CTMC states at exact observation times.

## Data Container

Use:

```julia
data = ExactStatePath(states, times)
```

where:

- `states[k]` is the observed CTMC state at time `times[k]`
- `times` is nondecreasing
- each observed state must lie in `states(model)`

## Likelihood Factorization

The path likelihood factorizes over consecutive intervals:

- start from the observed state at the beginning of an interval
- propagate to the next observation time
- read off the probability of the next observed state
- accumulate log probabilities across intervals

This is a fully observed transition model, not a hidden-state observation model.

## Current API

```julia
ll = loglikelihood(model, theta, data; gamma=...)
ll, grad = loglikelihood_and_gradient(model, theta, data; gamma=...)
```

The gradient is obtained through the differentiated-uniformization machinery.
Both functions also accept `backend=:sparse` or `backend=:structured` so the
likelihood layer can follow the same generator backend used for propagation.

## Impossible Paths

If an observed transition has zero probability under the current parameter
value:

- `loglikelihood` returns `-Inf`
- `loglikelihood_and_gradient` returns `(-Inf, NaN-vector)`

## Example

```julia
using DifferentiatedUniformization

model = SIModel(2)
theta = [1.4]
data = ExactStatePath([(1, 1), (1, 1), (0, 2)], [0.0, 0.5, 1.0])

ll = loglikelihood(model, theta, data; gamma=2.5)
ll, grad = loglikelihood_and_gradient(model, theta, data; gamma=2.5)
```
