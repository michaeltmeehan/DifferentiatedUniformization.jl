# Limitations And Deferred Features

This page describes what the package does not currently try to do.

## Current Limitations

- finite-state models only
- time-homogeneous generators only
- exact-state path likelihoods only
- first derivatives only
- canonical epidemic models only
- no built-in optimizer dependency
- no hidden-state observation model
- structured operator backend remains limited to the canonical SI, SIS, and SIR models
- tensor/Kronecker backend is currently implemented for SIR only

## Important Gradient Caveat

Automatic `gamma` selection is not differentiated through.

Current behavior:

- if `gamma` is omitted, the package chooses a valid value from `Q`
- that value is then treated as fixed inside the differentiated call

Implication:

- for optimization or finite-difference checks, fixed `gamma` is usually the
  safer choice

## Deferred Features

- SEIR implementation
- hidden-state / partial-observation models
- filtering and smoothing
- particle methods
- direct Turing.jl integration
- a direct `LogDensityProblems.jl` adapter
- broader interoperability layers
- parallel ensemble simulation
- broader structured backends beyond the canonical models
- tensor/Kronecker backends beyond the current SIR construction

## What The Current Package Is Best At

The current implementation is best viewed as:

- a validated finite-state CTMC core
- an explicit reference implementation for SI/SIS/SIR
- a testbed for gradient-based calibration using differentiated uniformization
- a benchmark target for Gillespie-versus-DU comparisons
