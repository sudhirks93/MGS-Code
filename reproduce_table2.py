#!/usr/bin/env python3
"""
Regenerate Table 2 of the paper (the [[(2a+b)p, bp, d]]_2 MGS code family).

Usage:
    python reproduce_table2.py            # print the table
    python reproduce_table2.py --latex    # print it as a LaTeX tabular

Every row is rebuilt from scratch by mgs_distance.build_HMGS and each
distance bound is certified by exhibiting an explicit logical operator.
"""
import argparse, io, contextlib
from mgs_distance import build_HMGS

ROWS = [(1,1,5,[1]),(1,1,7,[1]),(1,1,7,[1,2]),(1,1,11,[1]),(1,1,11,[1,2]),
        (1,1,11,[1,2,3]),(1,1,13,[1]),(1,1,13,[1,2]),(1,1,13,[1,2,3]),
        (1,2,5,[1]),(1,2,7,[1]),(1,2,7,[1,2]),(1,2,11,[1]),(1,2,11,[1,2]),
        (1,2,11,[1,2,3]),(1,2,13,[1]),(1,2,13,[1,2]),(1,2,13,[1,2,3]),
        (2,1,5,[1]),(2,1,7,[1]),(2,1,7,[1,2]),(2,1,11,[1]),(2,1,11,[1,2]),
        (2,1,11,[1,2,3]),(2,1,13,[1]),(2,1,13,[1,2]),(2,1,13,[1,2,3])]

def jstr(J): return "\\{" + ",".join(map(str, J)) + "\\}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX tabular")
    args = ap.parse_args()

    out, all_ok = [], True
    for (a, b, p, J) in ROWS:
        with contextlib.redirect_stdout(io.StringIO()):
            _, m = build_HMGS(a, b, p, J)
        all_ok &= m["d_A_certified"]
        out.append((a, b, p, J, m["n"], m["k"], f"{b}/{2*a+b}", m["d_A"], m["d_C"]))

    if args.latex:
        print("\\begin{tabular}{ccccccccc}\n\\toprule")
        print("$\\alpha$ & $\\beta$ & $p$ & $J$ & $n$ & $k$ & $R$ & $d_A$ & $d_C$\\\\")
        print("\\midrule")
        for a, b, p, J, n, k, R, dA, dC in out:
            print(f"{a} & {b} & {p:>2} & ${jstr(J)}$ & {n:>2} & {k:>2} & ${R}$ & {dA:>2} & {dC}\\\\")
        print("\\bottomrule\n\\end{tabular}")
    else:
        hdr = f"{'alpha':>5}{'beta':>5}{'p':>4}  {'J':<10}{'n':>5}{'k':>5}{'R':>7}{'d_A':>6}{'d_C':>6}"
        print(hdr); print("-"*len(hdr))
        for a, b, p, J, n, k, R, dA, dC in out:
            print(f"{a:>5}{b:>5}{p:>4}  {str(J):<10}{n:>5}{k:>5}{R:>7}{dA:>6}{dC:>6}")
        print("-"*len(hdr))
        print(f"{len(out)} rows.  All d_A witnesses certified: {all_ok}")

if __name__ == "__main__":
    main()
