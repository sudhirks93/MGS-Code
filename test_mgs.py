#!/usr/bin/env python3
"""Self-checks for the MGS construction.  Run:  python test_mgs.py"""
import io, contextlib, numpy as np
from mgs_distance import build_HMGS, f2_rref, symplectic_weight

CASES = [(1,1,5,[1]), (1,1,7,[1,2]), (1,2,7,[1]), (2,1,5,[1]), (1,2,11,[1,2,3])]

def symplectic_gram(H, n):
    L = np.block([[np.zeros((n,n),int), np.eye(n,dtype=int)],
                  [np.eye(n,dtype=int), np.zeros((n,n),int)]])
    return (H.astype(int) @ L @ H.astype(int).T) % 2

def main():
    fails = 0
    for (a,b,p,J) in CASES:
        with contextlib.redirect_stdout(io.StringIO()):
            H, m = build_HMGS(a,b,p,J)
        n, r = m['n'], m['r']
        tag = f"a={a},b={b},p={p},J={J}"

        # 1. the generators must all commute (symplectic self-orthogonality)
        if symplectic_gram(H, n).any():
            print(f"FAIL {tag}: stabilizer generators do not commute"); fails += 1

        # 2. H must have full rank 2r  (=> exactly k = n - 2r logical qubits)
        rank = len(f2_rref(H % 2)[1])
        if rank != 2*r:
            print(f"FAIL {tag}: rank(H)={rank}, expected {2*r}"); fails += 1

        # 3. both distance bounds must be certified, not merely asserted
        if not m['d_A_certified']:
            print(f"FAIL {tag}: d_A witness is not a genuine logical operator"); fails += 1
        if m['d_A_witness_weight'] != m['d_A']:
            print(f"FAIL {tag}: d_A witness weight {m['d_A_witness_weight']} != d_A {m['d_A']}")
            fails += 1

        # 4. the structural formula d_C = 2 + 2|J|
        if m['d_C'] != 2 + 2*len(J):
            print(f"FAIL {tag}: d_C={m['d_C']}, formula gives {2+2*len(J)}"); fails += 1

        # 5. the bounds must respect the quantum Singleton bound
        if min(m['d_A'], m['d_C']) > m['d_S']:
            print(f"FAIL {tag}: min(d_A,d_C) exceeds Singleton bound d_S"); fails += 1

        if not fails:
            print(f"ok   {tag}:  n={m['n']} k={m['k']} d_A={m['d_A']} d_C={m['d_C']} d_S={m['d_S']}")
    print()
    print("ALL TESTS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())
