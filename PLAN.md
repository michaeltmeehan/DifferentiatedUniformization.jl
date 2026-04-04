# DifferentiatedUniformization.jl — v0.1 Plan

## 1. Purpose

`DifferentiatedUniformization.jl` will be a Julia package for computing transient probabilities and parameter gradients for **finite-state continuous-time Markov chains (CTMCs)** using **uniformization** and **differentiated uniformization**. The motivating application is stochastic compartmental epidemic models, where the generator matrix \(Q(\theta)\) depends on model parameters and direct storage or differentiation of the full matrix may be impractical.

The package should aim to provide:

| Capability | Description |
|---|---|
| Transient probabilities | Compute \(p(t) = \exp(tQ(\theta)) p(0)\) |
| Parameter gradients | Compute \(\partial p(t) / \partial \theta_i\) |
| Likelihood support | Build log-likelihoods from observed state or partial observations |
| Calibration support | Expose stable likelihood/gradient functions suitable for use in optimization and Bayesian inference |
| Validation tools | Compare against exact matrix exponential methods and Gillespie-style simulation |

---

## 2. Initial scope for v0.1

Version 0.1 should be deliberately narrow.

### Included

| Included in v0.1 | Notes |
|---|---|
| Finite-state CTMCs | Explicit finite state space only |
| Time-homogeneous models | \(Q\) constant over each likelihood interval |
| Small to medium compartmental epidemic models | SI, SIS, SIR, possibly SEIR |
| Uniformization and differentiated uniformization | Core package functionality |
| Exact validation on small systems | Using dense/sparse matrix exponential where feasible |
| Gillespie benchmarking | For empirical validation and performance comparison |
| Julia-native inference hooks | Clean functions returning log densities and gradients |

### Excluded

| Excluded from v0.1 | Reason |
|---|---|
| Infinite-state or open population models | Requires truncation design beyond initial scope |
| Non-Markovian dwell times | Outside CTMC framework |
| Time-inhomogeneous generators | Can be added later as interval-wise extensions |
| Hidden Markov/state-space observation models | Better added after core CTMC engine is validated |
| GPU support | Not required initially |
| Full Turing.jl integration layer | Better after core API stabilizes |
| Arbitrary reaction-network compiler | Too large for first release |

---

## 3. Mathematical target

We consider a CTMC on a finite state space \(X\) with generator \(Q(\theta)\), where the probability vector satisfies the Kolmogorov forward equation and solution \(p(t)=\exp(tQ)p(0)\).

Uniformization introduces
\[
P = I + \frac{1}{\gamma}Q
\]
for \(\gamma \ge \max_x |Q_{x,x}|\), so that
\[
p(t) = \sum_{n=0}^{\infty} e^{-\gamma t}\frac{(\gamma t)^n}{n!} P^n p(0).
\]

### v0.1 numerical goals

| Goal | Requirement |
|---|---|
| Stable probability propagation | Accurate transient probabilities for finite-state models |
| Stable gradients | Accurate first derivatives with respect to each free parameter |
| Controlled truncation | Poisson-tail stopping rule with configurable tolerance |
| Reproducibility | Deterministic outputs for the DU routines |
| Validation | Agreement with exact methods and finite-difference gradients on small models |

---

## 4. Intended users

| User type | Need |
|---|---|
| Method developers | Test differentiated uniformization on new CTMCs |
| Epidemic modellers | Compute exact or near-exact transient probabilities and gradients |
| Bayesian users | Supply log densities and gradients to HMC-style workflows |
| Julia package developers | Build higher-level inference tooling on top of the core engine |

---

## 5. Package philosophy

The package should be **core-first** rather than ecosystem-first.

| Principle | Implication |
|---|---|
| Small, reliable core | Focus first on mathematically correct DU implementation |
| Explicit interfaces | Do not hide model structure behind too much abstraction |
| Interoperable outputs | Return values in a form usable by Optimization.jl, LogDensityProblems.jl, Turing.jl, etc. |
| Validation-heavy | Every major feature should have comparison tests |
| No unnecessary magic | User should be able to inspect states, generators, and derivative operators |

---

## 6. Proposed v0.1 public API

This is a provisional API sketch.

### Core model interface

A model should minimally provide or support:

| Function | Purpose |
|---|---|
| `states(model)` | Return ordered finite state space |
| `generator(model, θ)` | Return generator or generator action |
| `generator_derivatives(model, θ)` | Return derivative operators with respect to each parameter |
| `initial_distribution(model, u0)` | Convert user-specified initial condition to probability vector |

### Core DU functions

| Function | Purpose |
|---|---|
| `uniformize(Q_or_op, t; tol=1e-12, γ=nothing)` | Compute transient probability propagation |
| `differentiate_uniformize(Q_or_op, dQ_or_ops, t; tol=1e-12, γ=nothing)` | Compute transient probabilities and gradients |
| `propagate(model, θ, p0, t; tol=1e-12)` | High-level transient propagation |
| `propagate_with_gradient(model, θ, p0, t; tol=1e-12)` | High-level transient propagation plus gradients |

### Likelihood layer

| Function | Purpose |
|---|---|
| `loglikelihood(model, θ, data)` | Compute log-likelihood from observations |
| `loglikelihood_and_gradient(model, θ, data)` | Compute log-likelihood and gradient |
| `transition_probability(...)` | Optional convenience helper for direct transition probabilities |

### Benchmark / validation layer

| Function | Purpose |
|---|---|
| `simulate_gillespie(model, θ, x0, t, rng)` | Monte Carlo benchmark simulator |
| `estimate_transition_probabilities(...)` | Empirical benchmark from simulation |
| `finite_difference_gradient(f, θ)` | Validation helper |

---

## 7. Internal architecture

Suggested repository layout:

```text
DifferentiatedUniformization.jl/
├─ Project.toml
├─ src/
│  ├─ DifferentiatedUniformization.jl
│  ├─ core/
│  │  ├─ types.jl
│  │  ├─ states.jl
│  │  ├─ operators.jl
│  │  ├─ uniformization.jl
│  │  ├─ differentiated_uniformization.jl
│  │  └─ truncation.jl
│  ├─ models/
│  │  ├─ si.jl
│  │  ├─ sis.jl
│  │  ├─ sir.jl
│  │  └─ seir.jl
│  ├─ inference/
│  │  ├─ likelihood.jl
│  │  └─ logdensity.jl
│  ├─ simulation/
│  │  └─ gillespie.jl
│  └─ utils/
│     ├─ checks.jl
│     └─ finite_difference.jl
├─ test/
│  ├─ runtests.jl
│  ├─ test_uniformization.jl
│  ├─ test_du_gradients.jl
│  ├─ test_models_si.jl
│  ├─ test_models_sis.jl
│  ├─ test_models_sir.jl
│  └─ test_benchmarks.jl
├─ examples/
│  ├─ si_example.jl
│  ├─ sis_example.jl
│  ├─ sir_example.jl
│  └─ hmc_logdensity_example.jl
├─ docs/
│  └─ ...
└─ benchmark/
   ├─ exact_matrix_exp.jl
   └─ gillespie_comparison.jl
```

---

## 8. Data structures and representation choices

These choices should be fixed early so implementation remains consistent.

| Design question | v0.1 choice |
|---|---|
| State representation | Ordered vector of concrete state tuples or small immutable structs |
| Probability vectors | `Vector{Float64}` initially |
| Parameter vector | `AbstractVector{<:Real}` with explicit ordering documented per model |
| Generator representation | Prefer linear operator style; allow explicit sparse matrices for small models |
| Derivative representation | Same abstraction as generator |
| Tolerance handling | Explicit keyword arguments |
| Return style | Prefer small result structs rather than many positional outputs |

Suggested result structs:

```julia
struct DUResult{T}
    p::Vector{T}
    n_terms::Int
    γ::T
    tail_mass_bound::T
end

struct DUGradientResult{T}
    p::Vector{T}
    dp::Matrix{T}   # columns correspond to parameters
    n_terms::Int
    γ::T
    tail_mass_bound::T
end
```

---

## 9. Validation plan

This should be in place from day one.

### Validation category A: exact small-state comparisons

For small models where the generator can be built explicitly, compare DU outputs against dense or sparse matrix exponential calculations.

| Model | Comparison |
|---|---|
| SI | Exact matrix exponential vs DU |
| SIS | Exact matrix exponential vs DU |
| SIR | Exact matrix exponential vs DU for small \(N\) |
| SEIR | Optional exact comparison for very small \(N\) |

### Validation category B: gradient checks

| Check | Method |
|---|---|
| DU gradient vs finite difference | Central differences on small systems |
| DU gradient vs AD on explicit matrix exponential code | Optional secondary check |
| Likelihood gradient vs finite difference | For observation models used in examples |

### Validation category C: stochastic simulation comparison

| Check | Method |
|---|---|
| State marginals | DU probabilities vs empirical Gillespie frequencies |
| Mean trajectories | DU implied moments vs simulation averages |
| Limiting behaviour | Agreement improves with more Monte Carlo replicates |

---

## 10. Benchmark plan

Benchmarks should answer three questions:

| Question | Comparison |
|---|---|
| Is DU correct? | Compare to exact matrix exponential on small systems |
| Is DU practically useful? | Compare runtime to exact methods as state size grows |
| Does DU match simulation? | Compare to Gillespie Monte Carlo estimates |

### Initial benchmark suite

| Benchmark | Description |
|---|---|
| `benchmark_exact_si` | SI model, increasing population size |
| `benchmark_exact_sis` | SIS model, increasing population size |
| `benchmark_exact_sir` | SIR model, increasing population size |
| `benchmark_du_gradient` | Cost of gradient evaluation vs finite differences |
| `benchmark_gillespie_sir` | DU probabilities vs empirical simulation |

---

## 11. Interoperability strategy

For v0.1, interoperability should proceed in layers:

| Layer | Goal |
|---|---|
| Core numerical layer | Return stable `loglikelihood_and_gradient` |
| LogDensityProblems layer | Wrap package outputs as a Julia log density |
| Turing/HMC examples | Provide example use, not hard dependency |
| AD compatibility | Nice-to-have, but not the primary design driver |

Recommended stance:

- **Primary gradient source:** differentiated uniformization itself
- **Secondary support:** compatibility with ForwardDiff for outer parameter transforms
- **Deferred:** deep Zygote-first design

This keeps the package aligned with its mathematical contribution rather than rebuilding around generic AD.

---

## 12. Testing strategy

### Unit tests

| Area | Tests |
|---|---|
| Generator construction | Rates correct, rows sum appropriately, indexing consistent |
| Uniformization | Correct against exact matrix exponential |
| Truncation | Tail mass bound respected |
| DU gradients | Match finite differences |
| Model APIs | Consistent state indexing and parameter handling |

### Property-style tests

| Property | Example |
|---|---|
| Probability mass conserved | `sum(p) ≈ 1` |
| Non-negativity | `minimum(p) ≥ -tol` |
| Gradient sanity | Small perturbations agree with first-order approximation |

### Regression tests

Include fixed examples with frozen expected outputs for a few canonical small models.

---

## 13. Documentation goals

| Doc page | Purpose |
|---|---|
| Overview | What DU is and when to use it |
| Quick start | Minimal example |
| Mathematical notes | Uniformization and differentiated uniformization summary |
| Model interface | How to add a new CTMC |
| Validation examples | Exact and simulation comparisons |
| Inference example | HMC/log-density example |

---

## 14. Initial canonical models

These models should be implemented first and treated as reference examples.

| Model | Reason |
|---|---|
| SI | Simplest epidemic CTMC |
| SIS | Includes return transitions |
| SIR | Main motivating epidemic example |
| SEIR | Optional stretch goal for v0.1 |

---

## 15. Risks and technical challenges

| Risk | Mitigation |
|---|---|
| State-space explosion | Keep v0.1 to small/medium finite models |
| Confusion about state indexing | Standardize a single ordering convention |
| Numerical instability in gradients | Test aggressively against finite differences |
| Over-abstracting too early | Start with explicit model implementations |
| Premature ecosystem integration | Delay Turing-specific work until core validated |

---

## 16. Definition of done for v0.1

Version 0.1 is complete when all of the following are true:

| Requirement | Done criterion |
|---|---|
| Core DU implemented | Probabilities computed for finite-state CTMCs |
| Gradient DU implemented | First derivatives for all free parameters returned |
| Canonical models included | At least SI, SIS, SIR |
| Exact validation present | Tests against matrix exponential pass |
| Gillespie benchmark present | Simulation comparison examples included |
| Documentation present | Quick start and model interface documented |
| Calibration-ready API present | Stable log-likelihood and gradient functions exposed |

---

## 17. Recommended first work packets

These should become the first GitHub issues.

| Issue | Task |
|---|---|
| 1 | Create package skeleton with CI, formatting, docs, and tests |
| 2 | Implement state-space enumeration and generator builders for SI/SIS/SIR |
| 3 | Implement plain uniformization |
| 4 | Implement differentiated uniformization |
| 5 | Add exact matrix exponential validation tests |
| 6 | Add Gillespie simulation module for benchmarking |
| 7 | Add likelihood and gradient wrappers |
| 8 | Add `LogDensityProblems.jl` wrapper and one HMC example |

---

## 18. Recommended repository files to add immediately

| File | Purpose |
|---|---|
| `README.md` | Package overview and roadmap |
| `PLAN.md` | This design brief |
| `ROADMAP.md` | Issue ordering and future extensions |
| `CONTRIBUTING.md` | Development conventions |
| `docs/src/index.md` | Initial docs landing page |
