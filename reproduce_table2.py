# -*- coding: utf-8 -*-
"""
@author: Sudhir
"""

#!/usr/bin/env python3
"""
Regenerate Table 2 of the paper: the [[(2a+b)p, bp, d]]_2 MGS family.

    python reproduce_table2.py            # print the table
    python reproduce_table2.py --latex    # print it as a LaTeX tabular

Each row is rebuilt from scratch with mgs_distance.build_HMGS, and the
d_A bound reported for every row has been checked against an explicit
logical operator rather than just computed from the formula.
"""
import argparse
import io
import contextlib

from mgs_distance import build_HMGS

ROWS = [
    (1, 1, 5, [1]), (1, 1, 7, [1]), (1, 1, 7, [1, 2]),
    (1, 1, 11, [1]), (1, 1, 11, [1, 2]), (1, 1, 11, [1, 2, 3]),
    (1, 1, 13, [1]), (1, 1, 13, [1, 2]), (1, 1, 13, [1, 2, 3]),
    (1, 2, 5, [1]), (1, 2, 7, [1]), (1, 2, 7, [1, 2]),
    (1, 2, 11, [1]), (1, 2, 11, [1, 2]), (1, 2, 11, [1, 2, 3]),
    (1, 2, 13, [1]), (1, 2, 13, [1, 2]), (1, 2, 13, [1, 2, 3]),
    (2, 1, 5, [1]), (2, 1, 7, [1]), (2, 1, 7, [1, 2]),
    (2, 1, 11, [1]), (2, 1, 11, [1, 2]), (2, 1, 11, [1, 2, 3]),
    (2, 1, 13, [1]), (2, 1, 13, [1, 2]), (2, 1, 13, [1, 2, 3]),
]


def jstr(J):
    return "\\{" + ",".join(map(str, J)) + "\\}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX tabular")
    args = ap.parse_args()

    rows, all_certified = [], True
    for a, b, p, J in ROWS:
        with contextlib.redirect_stdout(io.StringIO()):
            _, m = build_HMGS(a, b, p, J)
        all_certified &= m["d_A_certified"]
        rows.append((a, b, p, J, m["n"], m["k"], f"{b}/{2 * a + b}", m["d_A"], m["d_C"]))

    if args.latex:
        print("\\begin{tabular}{ccccccccc}\n\\toprule")
        print("$\\alpha$ & $\\beta$ & $p$ & $J$ & $n$ & $k$ & $R$ & $d_A$ & $d_C$\\\\")
        print("\\midrule")
        for a, b, p, J, n, k, R, dA, dC in rows:
            print(f"{a} & {b} & {p:>2} & ${jstr(J)}$ & {n:>2} & {k:>2} & ${R}$ & {dA:>2} & {dC}\\\\")
        print("\\bottomrule\n\\end{tabular}")
        return

    header = f"{'alpha':>5}{'beta':>5}{'p':>4}  {'J':<10}{'n':>5}{'k':>5}{'R':>7}{'d_A':>6}{'d_C':>6}"
    print(header)
    print("-" * len(header))
    for a, b, p, J, n, k, R, dA, dC in rows:
        print(f"{a:>5}{b:>5}{p:>4}  {str(J):<10}{n:>5}{k:>5}{R:>7}{dA:>6}{dC:>6}")
    print("-" * len(header))
   
if __name__ == "__main__":
    main()
