# Conventions

This package uses one generator convention everywhere. It is important to keep
that convention in mind when interpreting results or implementing extensions.

## Probability And Generator Convention

- Probability vectors are columns.
- The forward equation is `dp/dt = Q * p`.
- `Q[to, from]` is the rate from state `from` to state `to`.
- Columns of `Q` sum to zero.
- Diagonal entries are nonpositive.
- Off-diagonal entries are nonnegative.

This means the diagonal entry in each column is minus the total exit rate from
that state.

## Uniformization Convention

For a generator `Q`, uniformization uses

`P = I + Q / gamma`

with `gamma >= max exit rate`.

Under the package convention, `P` acts on probability columns exactly as `Q`
does. So the transient probability vector is approximated by a truncated
Poisson-weighted sum of `P^n p0`.

## Gamma Convention In Differentiation

If `gamma` is omitted:

- the package chooses it from the generator
- then treats that chosen value as fixed within the differentiated propagation
  call

As a result:

- derivatives do not include sensitivity of the automatic `gamma` selection rule
- the gradient is piecewise smooth with respect to parameters
- smooth finite-difference or optimizer behavior is best obtained by passing a
  fixed `gamma`

## Likelihood Convention

The current likelihood layer assumes fully observed exact-state paths:

- observed states are exact CTMC states
- observation times are exact and nondecreasing
- path likelihoods factorize over consecutive time intervals

This is intentionally simpler than a hidden-state observation model.
