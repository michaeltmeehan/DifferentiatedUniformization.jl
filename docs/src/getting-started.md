# Getting Started

This tutorial shows the shortest path from model construction to transient
probabilities and gradients.

## 1. Build a Model

```julia
using DifferentiatedUniformization

model = SIModel(2)
theta = [0.8]
```

## 2. Inspect The State Space

```julia
states(model)
```

For `SIModel(2)`, this gives:

```julia
[(2, 0), (1, 1), (0, 2)]
```

## 3. Construct The Generator

```julia
Q = generator(model, theta)
Matrix(Q)
```

Remember the package convention:

- probabilities are columns
- `dp/dt = Q * p`
- `Q[to, from]` is the rate from `from` to `to`

## 4. Propagate Probabilities

```julia
result = propagate(model, theta, (1, 1), 0.7; gamma=1.0)
result.p
```

## 5. Propagate Gradients

```julia
grad_result = propagate_with_gradient(model, theta, (1, 1), 0.7; gamma=1.0)
grad_result.p
grad_result.dp
```

## 6. Build A Likelihood

```julia
data = ExactStatePath([(1, 1), (1, 1), (0, 2)], [0.0, 0.5, 1.0])

ll = loglikelihood(model, theta, data; gamma=2.0)
ll, grad = loglikelihood_and_gradient(model, theta, data; gamma=2.0)
```

## 7. Wrap As A Log-Density Problem

```julia
problem = ExactPathLogDensity(model, data; gamma=2.0)

dimension(problem)
logdensity(problem, theta)
logdensity_and_gradient(problem, theta)
```

## Practical Note

For gradient-based work, pass a fixed `gamma` whenever possible. The package can
choose `gamma` automatically, but that selection rule is treated as fixed inside
a differentiated propagation call.
