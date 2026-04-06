# Calibration Tutorial

This tutorial shows the main intended inference workflow: gradient-based
estimation using differentiated uniformization.

We use a small SI example with synthetic exact-state path data.

## 1. Define The Model And Data

```julia
using DifferentiatedUniformization

model = SIModel(2)
true_theta = [1.4]
gamma = 2.5
beta_upper = 2.4

data = ExactStatePath(
    [(1, 1), (1, 1), (0, 2), (0, 2)],
    [0.0, 0.5, 1.0, 1.5],
)
```

We use a fixed `gamma` because the optimization loop should see a smooth
objective.

## 2. Define The Objective

The package already provides the log-likelihood and its gradient with respect to
`beta`. In the example script we optimize a transformed parameter `eta` so that
`beta` stays positive and below a fixed upper bound:

```julia
sigmoid(x) = inv(1 + exp(-x))
logit(p) = log(p / (1 - p))

beta_from_eta(eta, beta_upper) = beta_upper * sigmoid(eta)
dbeta_deta(beta, beta_upper) = beta * (1 - beta / beta_upper)
```

Then:

```julia
function objective_and_gradient(eta, model, data; gamma, beta_upper)
    beta = beta_from_eta(eta, beta_upper)
    loglik, gradient = loglikelihood_and_gradient(model, [beta], data; gamma=gamma)
    chain = dbeta_deta(beta, beta_upper)
    return -loglik, -gradient[1] * chain
end
```

## 3. Run A Small Gradient-Based Optimizer

The example script
[examples/gradient_estimation_example.jl](/C:/Users/jc213439/Dropbox/dev/DifferentiatedUniformization/examples/gradient_estimation_example.jl)
uses a simple backtracking gradient descent loop on the negative log-likelihood.

This is intentionally lightweight:

- no new optimizer dependency
- explicit use of DU gradients
- easy to read and validate

## 4. Use The Log-Density Wrapper

You can also wrap the same data as:

```julia
problem = ExactPathLogDensity(model, data; gamma=gamma)
```

and evaluate:

```julia
logdensity(problem, [1.4])
logdensity_and_gradient(problem, [1.4])
```

## 5. Grid Scan Versus Gradient-Based Estimation

The repository includes both:

- a grid scan example: useful for sanity checks and visualization
- a gradient-based estimation example: the main intended inference workflow

The point of differentiated uniformization is the second one: accurate
likelihood derivatives for gradient-based calibration.
