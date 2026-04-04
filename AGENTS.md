# AGENTS.md

## Purpose

This repository implements `DifferentiatedUniformization.jl`, a Julia package for **finite-state continuous-time Markov chains (CTMCs)** using:

- uniformization
- differentiated uniformization

The primary motivating applications are stochastic compartmental epidemic models such as SI, SIS, and SIR.

Read `PLAN.md` and `README.md` before making substantial changes.

---

## Project status

This repository is in the **bootstrap / early implementation** phase.

The current goal is to build a clean, reliable, well-tested core for v0.1. Do not broaden scope unless explicitly requested.

---

## Scope constraints

### In scope for v0.1

- finite-state CTMCs only
- time-homogeneous generators
- small to medium epidemic compartment models
- SI, SIS, SIR as primary reference models
- SEIR as an optional stretch goal
- transient probability propagation
- first derivatives with respect to model parameters
- exact validation on small state spaces
- Gillespie-style simulation for benchmarking
- compatibility-oriented likelihood/gradient interfaces

### Out of scope for v0.1

- infinite-state models
- open-ended population models
- non-Markovian dwell times
- time-inhomogeneous generators
- arbitrary reaction-network compilers
- GPU support
- broad ecosystem wrappers as a first priority
- premature optimization that obscures clarity

If a task would require expanding scope, stop and state that clearly.

---

## General development philosophy

Prioritize:

1. correctness
2. clarity
3. consistency
4. testability
5. modest, reviewable changes

Prefer explicit Julia code over clever abstraction.

Do not overengineer interfaces. Keep the initial design small and inspectable.

When uncertain, choose the simpler design that aligns with `PLAN.md`.

---

## Repository structure

Maintain this approximate structure unless explicitly instructed otherwise:

```text
src/
  DifferentiatedUniformization.jl
  core/
  models/
  inference/
  simulation/
  utils/

test/
examples/
docs/
benchmark/
```

Top-level module exports should remain intentional and minimal.

---

## Julia coding guidelines

### Style

- Prefer clear function and type names.
- Use multiple dispatch where it improves clarity.
- Add docstrings for public types and functions.
- Keep functions reasonably short and focused.
- Avoid unnecessary macros.
- Avoid type piracy.
- Avoid hidden global state.

### Types

- Prefer concrete structs for simple models.
- Keep state representations explicit.
- Use parametric types only when they meaningfully improve correctness or flexibility.
- Do not introduce deep type hierarchies unless there is a clear need.

### Errors

- Fail loudly and informatively.
- For placeholder functionality, throw clear and consistent errors.
- Do not silently return incorrect fallback values.

### Dependencies

- Keep dependencies minimal.
- Do not add heavy packages unless there is a strong justification.
- Prefer Julia standard library where appropriate.

---

## Numerical implementation guidance

### Core principles

- The core numerical routines must be deterministic.
- Truncation logic must be explicit and testable.
- Numerical tolerances should be user-configurable through keywords where appropriate.
- Validation against exact methods is required for small systems.

### Uniformization

When implementing uniformization:

- make the choice of uniformization rate explicit
- document assumptions about the generator representation
- expose truncation information where useful
- avoid hidden numerical heuristics

### Differentiated uniformization

When implementing differentiated uniformization:

- maintain a clean separation between probability propagation and derivative propagation
- document parameter ordering clearly
- ensure gradient output shape is explicit and consistent
- validate against finite differences on small models

---

## Model interface expectations

At minimum, model-related code should support or move toward supporting:

- `states(model)`
- `generator(model, θ)`
- `generator_derivatives(model, θ)`
- `initial_distribution(model, u0)`

Do not introduce a more elaborate interface unless needed by the current task.

Canonical initial models:

- SI
- SIS
- SIR

SEIR is optional and should not delay progress.

---

## Validation requirements

Every substantial numerical addition should come with tests.

Preferred validation ladder:

1. unit tests for structure and indexing
2. exact comparison against matrix exponential on small systems
3. gradient comparison against finite differences
4. stochastic comparison against Gillespie simulation where relevant

Do not merge or present numerical code as complete without validation.

---

## Testing guidance

When adding or modifying functionality:

- add or update tests
- keep tests small and readable
- use deterministic setups where possible
- avoid fragile tests tied to incidental formatting
- ensure failure messages are interpretable

Minimum baseline tests should check:

- package loading
- exported names
- model construction
- expected dimensions and indexing
- error behaviour for unimplemented stubs
- correctness of numerical routines once implemented

---

## Documentation guidance

For public-facing features:

- add concise docstrings
- keep README examples simple
- prefer small, concrete examples
- document assumptions and parameter conventions

Do not let documentation drift away from actual code behaviour.

---

## Benchmark guidance

Benchmarks are important, but they are not substitutes for correctness tests.

Use benchmarks to answer questions like:

- does the implementation agree with exact results?
- how does runtime scale with state-space size?
- how does DU compare with finite differences or simulation?

Avoid performance tuning before the basic implementation is correct and tested.

---

## Interoperability guidance

The package should eventually plug into tools such as:

- `Optimization.jl`
- `LogDensityProblems.jl`
- `Turing.jl`

However:

- do not let downstream wrapper design dominate the core implementation
- the primary gradient engine should be differentiated uniformization itself
- wrappers should be thin and should follow once the core is stable

---

## Change management rules

Before making changes:

1. inspect repository structure
2. read relevant files
3. identify assumptions
4. keep edits scoped to the requested task

After making changes:

1. run relevant tests
2. summarize files changed
3. summarize assumptions made
4. note any deferred issues or TODOs

Do not silently change project scope, naming conventions, or architecture.

---

## Preferred implementation sequence

Unless explicitly asked otherwise, work in roughly this order:

1. package scaffold
2. model and state-space enumeration
3. generator construction
4. plain uniformization
5. differentiated uniformization
6. exact validation tests
7. Gillespie benchmarking
8. likelihood wrappers
9. interoperability wrappers/examples

---

## What to avoid

- implementing features not requested
- inventing additional scientific scope
- introducing complicated generic frameworks too early
- writing untested numerical code
- broad refactors unrelated to the current task
- changing public APIs without necessity
- mixing benchmarking and production logic unnecessarily

---

## When in doubt

If a design decision is ambiguous:

- prefer the smallest design that satisfies the current task
- follow `PLAN.md`
- leave a clear TODO rather than inventing speculative machinery
- state assumptions explicitly in the summary
