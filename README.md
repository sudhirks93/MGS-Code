# Modified Graph State (MGS) Codes — Construction and Distance Analysis

Code accompanying the paper *Graph State Codes for Coded Quantum Storage
Networks*. It constructs the generalized MGS stabilizer code family

    [[ (2*alpha + beta) * p,  beta * p,  d ]]_2

and computes two analytical upper bounds on its minimum distance. **This
repository reproduces Table 2 of the paper.**

## Requirements

    python >= 3.8
    numpy
    matplotlib        # only needed for the plots in mgs_distance.py

    pip install numpy matplotlib

## Quick start

    python reproduce_table2.py            # reproduce Table 2 of the paper
    python reproduce_table2.py --latex    # same, as a LaTeX tabular
    python test_mgs.py                    # run the self-checks
    python mgs_distance.py                # full sweep + plots

## Parameters

| symbol  | meaning                                   |
|---------|-------------------------------------------|
| `alpha` | row-blocks of `U` (positive integer)      |
| `beta`  | column-blocks of `U` (positive integer)   |
| `p`     | odd prime                                 |
| `J`     | index set, a subset of `{1, ..., (p-1)/2}`|

Derived: `r = alpha*p`, `k = beta*p`, `n = (2*alpha+beta)*p`,
rate `R = beta/(2*alpha+beta)`, Singleton bound `d_S = alpha*p + 1`.

Not every `J` is admissible. The construction requires
`gcd(t(x)^2 + 1, x^p - 1) = 1`, otherwise `M = A_CC` is singular and no
choice of `U` can repair it. `build_HMGS` raises a `ValueError` naming the
condition when this fails.

## What the distance numbers mean

`d_A` and `d_C` are **upper bounds** on the true minimum distance, not the
distance itself: computing the exact distance of a non-CSS stabilizer code
is NP-hard. Each bound is *certified* here, meaning the code exhibits an
explicit logical operator of that weight and verifies that it lies in the
centralizer of the stabilizer group but not in the group itself:

* `d_A = 1 + |I|_min` — from a Case-1 (message-qubit) logical operator.
  Grows with `p`; unaffected by the choice of `J`.
  Certified via `meta['d_A_certified']` and `meta['d_A_witness_weight']`.
* `d_C = 2 + 2|J|` — from a Case-2 (check-qubit) logical operator.
  Constant in `alpha, beta, p`; increases with `|J|`.
  Certified column by column inside `build_HMGS` (use `verbose=True` to see it).

`test_mgs.py` additionally checks, for several parameter sets, that the
generators commute under the symplectic form and that `rank(H) = 2r`, i.e.
that the output really is an `[[n, k, d]]` stabilizer code.

## A note on the greedy construction

`U` is built greedily: powers `Q^i` are added one at a time and kept only
if the full-rank condition still holds. The scan runs `i = 0, 1, ..., p-1`.
A different scan order could in principle yield a different `U`; in the
cases we tested, `d_A` was unchanged, but the construction is a greedy one
and is not claimed to be canonical.

## Files

| file                   | purpose                                       |
|------------------------|-----------------------------------------------|
| `mgs_distance.py`      | construction, distance bounds, sweeps, plots  |
| `reproduce_table2.py`  | regenerates Table 2 of the paper              |
| `test_mgs.py`          | self-checks                                   |

## Citation

If you use this code, please cite the paper.
