# -*- coding: utf-8 -*-
"""
@author: Sudhir
"""

# -*- coding: utf-8 -*-
"""
Distance analysis for the [[(2a+b)p, bp, d]]_2 MGS code family.

Given (alpha, beta, p, J) this builds the parity-check matrix H_MGS and
returns two upper bounds on the true minimum distance d (exact distance
for a non-CSS stabilizer code is NP-hard to compute):

  d_A = 1 + |I|_min, the weight of the lightest message-qubit logical
        operator (Case 1). Grows with p, doesn't depend on J.

  d_C = 2 + 2|J|, the weight of the lightest check-qubit logical
        operator (Case 2). Constant in alpha, beta, p; grows with |J|.

Both bounds are certified rather than just asserted: for each one we
build the explicit logical operator, confirm it commutes with every
stabilizer generator, and confirm it is not itself in the stabilizer
group (see build_HMGS).
"""

import numpy as np

# ---------------------------------------------------------------------
# GF(2) polynomial arithmetic, working mod x^p - 1
# ---------------------------------------------------------------------

def poly_mul_mod(a, b, p):
    """Multiply two length-p polynomials mod x^p - 1 over GF(2)."""
    res = np.zeros(p, dtype=np.uint8)
    for i in range(p):
        if a[i]:
            for j in range(p):
                if b[j]:
                    res[(i + j) % p] ^= 1
    return res


def poly_degree(f):
    for i in range(len(f) - 1, -1, -1):
        if f[i]:
            return i
    return -1


def poly_add(a, b):
    n = max(len(a), len(b))
    r = np.zeros(n, dtype=np.uint8)
    r[:len(a)] ^= a
    r[:len(b)] ^= b
    return r


def poly_divmod(f, g):
    f = f.copy()
    dg = poly_degree(g)
    if dg < 0:
        raise ZeroDivisionError("division by the zero polynomial")
    q = np.zeros(max(poly_degree(f) - dg + 1, 1), dtype=np.uint8)
    while poly_degree(f) >= dg:
        d = poly_degree(f) - dg
        if d >= len(q):
            q = np.append(q, np.zeros(d - len(q) + 1, dtype=np.uint8))
        q[d] ^= 1
        shifted = np.zeros(d + len(g), dtype=np.uint8)
        shifted[d:d + len(g)] = g
        f = poly_add(f, shifted)
    return q, f


def poly_gcd(a, b):
    a, b = a.copy(), b.copy()
    while poly_degree(b) >= 0 and np.any(b):
        _, r = poly_divmod(a, b)
        a, b = b, r
    return a


def is_coprime_with_xp_minus1(poly, p):
    mod = np.zeros(p + 1, dtype=np.uint8)
    mod[0] = 1
    mod[p] = 1
    pe = np.zeros(p + 1, dtype=np.uint8)
    pe[:p] = poly
    return poly_degree(poly_gcd(pe, mod)) == 0


def check_J_valid(p, J, t_poly):
    """
    Necessary condition for det(M) != 0, checked before running the
    greedy construction: at U = 0, M reduces to A_CC and
    det(t(x)^2 I_alpha + K(x)) becomes (t(x)^2 + 1)^alpha, so the
    condition is just gcd(t(x)^2 + 1, x^p - 1) = 1. If this fails, no
    choice of U can make M nonsingular for this J.
    """
    t2 = poly_mul_mod(t_poly, t_poly, p)
    t2[0] ^= 1
    return is_coprime_with_xp_minus1(t2, p)


# ---------------------------------------------------------------------
# Circulant matrices / blocks
# ---------------------------------------------------------------------

def circulant_power(p, i):
    """The p x p circulant shift matrix Q^i."""
    P = np.zeros((p, p), dtype=np.uint8)
    for row in range(p):
        P[row, (row + i) % p] = 1
    return P


def build_T(alpha, p, J):
    """
    T is alpha x alpha block-diagonal, each diagonal block equal to
    sum_{j in J} (Q^j + Q^{p-j}), off-diagonal blocks zero.
    """
    r = alpha * p
    T_blk = np.zeros((p, p), dtype=np.uint8)
    for j in J:
        T_blk = (T_blk + circulant_power(p, j) + circulant_power(p, p - j)) % 2
    T = np.zeros((r, r), dtype=np.uint8)
    for s in range(alpha):
        T[s * p:(s + 1) * p, s * p:(s + 1) * p] = T_blk
    return T, T_blk


def t_polynomial(p, J):
    t = np.zeros(p, dtype=np.uint8)
    for j in J:
        t[j % p] ^= 1
        t[(p - j) % p] ^= 1
    return t


# ---------------------------------------------------------------------
# Full-rank condition on M (Lemma 2)
# ---------------------------------------------------------------------

def check_fullrank_condition(alpha, t_poly, u_matrix, p):
    """
    Check that det(t(x)^2 I_alpha + K(x)) is coprime to x^p - 1, where
    K_{s,s'}(x) = delta_{s,s'} + sum_l u_{s,l}(x) u_{s',l}(x^-1).

    u_matrix has shape (alpha, beta, p); u_matrix[s, l, :] is u_{s,l}(x).
    """
    K = np.zeros((alpha, alpha, p), dtype=np.uint8)
    for s in range(alpha):
        for sp in range(alpha):
            if s == sp:
                K[s, sp, 0] ^= 1
            for ell in range(u_matrix.shape[1]):
                u = u_matrix[s, ell, :]
                u_sp = u_matrix[sp, ell, :]
                u_sp_inv = np.zeros(p, dtype=np.uint8)
                u_sp_inv[0] = u_sp[0]
                for i in range(1, p):
                    u_sp_inv[p - i] = u_sp[i]
                K[s, sp] = (K[s, sp] + poly_mul_mod(u, u_sp_inv, p)) % 2

    t2 = poly_mul_mod(t_poly, t_poly, p)
    A = K.copy()
    for s in range(alpha):
        A[s, s] = (A[s, s] + t2) % 2

    if alpha == 1:
        det_poly = A[0, 0]
    else:
        # Bareiss-style elimination over GF(2)[x]/(x^p-1); no sign
        # tracking needed since -1 == +1 in GF(2).
        M = [[A[i, j].copy() for j in range(alpha)] for i in range(alpha)]
        det_poly = np.zeros(p, dtype=np.uint8)
        det_poly[0] = 1

        for col in range(alpha):
            pivot_row = None
            for row in range(col, alpha):
                if poly_degree(M[row][col]) >= 0 and np.any(M[row][col]):
                    pivot_row = row
                    break
            if pivot_row is None:
                return False
            if pivot_row != col:
                M[col], M[pivot_row] = M[pivot_row], M[col]

            det_poly = poly_mul_mod(det_poly, M[col][col], p) % 2

            for row in range(col + 1, alpha):
                for c2 in range(col + 1, alpha):
                    t1 = poly_mul_mod(M[col][col], M[row][c2], p)
                    t2_ = poly_mul_mod(M[row][col], M[col][c2], p)
                    M[row][c2] = (t1 + t2_) % 2
                M[row][col] = np.zeros(p, dtype=np.uint8)

    return is_coprime_with_xp_minus1(det_poly, p)


# ---------------------------------------------------------------------
# Greedy construction of U
# ---------------------------------------------------------------------

def greedy_build_U(alpha, beta, p, t_poly):
    """
    Build U block by block. For each block U_{s,l} we try adding powers
    Q^0, ..., Q^{p-1} one at a time, keeping a power only if the
    full-rank condition on M still holds afterwards; if a single power
    fails we try pairing it with a later one before giving up on it.

    Returns the accepted index sets and the corresponding polynomial
    representation u_matrix (shape alpha x beta x p).
    """
    u_matrix = np.zeros((alpha, beta, p), dtype=np.uint8)
    I_sets = [[[] for _ in range(beta)] for _ in range(alpha)]

    for s in range(alpha):
        for ell in range(beta):
            added = set()
            for i in range(p):
                if i in added:
                    continue
                u_matrix[s, ell, i] ^= 1
                if check_fullrank_condition(alpha, t_poly, u_matrix, p):
                    I_sets[s][ell].append(i)
                    added.add(i)
                    continue
                found = False
                for j in range(i + 1, p):
                    if j in added:
                        continue
                    u_matrix[s, ell, j] ^= 1
                    if check_fullrank_condition(alpha, t_poly, u_matrix, p):
                        I_sets[s][ell].extend([i, j])
                        added.add(i)
                        added.add(j)
                        found = True
                        break
                    u_matrix[s, ell, j] ^= 1
                if not found:
                    u_matrix[s, ell, i] ^= 1  # revert, this power stays out

    return I_sets, u_matrix


# ---------------------------------------------------------------------
# GF(2) linear algebra helpers
# ---------------------------------------------------------------------

def f2_rref(M):
    M = M.copy() % 2
    rows, cols = M.shape
    pivots = []
    row = 0
    for col in range(cols):
        found = next((r for r in range(row, rows) if M[r, col]), -1)
        if found < 0:
            continue
        M[[row, found]] = M[[found, row]]
        pivots.append(col)
        for r in range(rows):
            if r != row and M[r, col]:
                M[r] = (M[r] + M[row]) % 2
        row += 1
        if row == rows:
            break
    return M, pivots


def symplectic_weight(v, n):
    return int(np.any(np.stack([v[:n], v[n:]], axis=0), axis=0).sum())


# ---------------------------------------------------------------------
# Build H_MGS and its certified distance bounds
# ---------------------------------------------------------------------

def build_HMGS(alpha, beta, p, J=None):
    """
    Build H_MGS = [B_CM | I_2r | A_CM | M] for given (alpha, beta, p, J)
    and return (H, meta), where meta carries n, k, R, the Singleton
    bound d_S, and the two certified distance bounds d_A, d_C.
    """
    if J is None:
        J = [1]
    maxj = (p - 1) // 2
    for j in J:
        if j > maxj:
            raise ValueError(f"J contains j={j} > (p-1)/2={maxj} for p={p}")

    r = alpha * p
    k = beta * p
    n = (2 * alpha + beta) * p
    R = beta / (2 * alpha + beta)
    d_S = r + 1

    T, T_blk = build_T(alpha, p, J)
    t_poly = t_polynomial(p, J)

    if not check_J_valid(p, J, t_poly):
        raise ValueError(
            f"J={J} is not usable for p={p}: gcd(t(x)^2+1, x^p-1) != 1, "
            f"so M is singular for every choice of U. Pick J from "
            f"{{1,...,{(p - 1) // 2}}} such that this gcd condition holds."
        )

    I_sets, u_matrix = greedy_build_U(alpha, beta, p, t_poly)

    U = np.zeros((r, k), dtype=np.uint8)
    for s in range(alpha):
        for ell in range(beta):
            block = np.zeros((p, p), dtype=np.uint8)
            for i in I_sets[s][ell]:
                block = (block + circulant_power(p, i)) % 2
            U[s * p:(s + 1) * p, ell * p:(ell + 1) * p] = block

    I_r = np.eye(r, dtype=np.uint8)
    I_2r = np.eye(2 * r, dtype=np.uint8)

    ACM = np.vstack([U, np.zeros((r, k), dtype=np.uint8)])
    BCM = np.vstack([np.zeros((r, k), dtype=np.uint8), U])
    ACC = np.block([[T, I_r], [I_r, T]])
    M_mat = (ACC + ACM @ BCM.T) % 2
    H = np.hstack([np.hstack([BCM, I_2r]), np.hstack([ACM, M_mat])])

    rank_H = len(f2_rref(H % 2)[1])

    # d_A: lightest message-qubit logical operator (Case 1)
    col_weights = [sum(len(I_sets[s][ell]) for s in range(alpha)) for ell in range(beta)]
    I_min = min(col_weights)
    d_A = I_min + 1

    j_min = min(range(k), key=lambda j: int(ACM[:, j].sum()))
    v_A = np.zeros(2 * n, dtype=np.uint8)
    v_A[j_min] = 1
    v_A[n + k: n + k + 2 * r] = ACM[:, j_min]
    v_A_swap = np.concatenate([v_A[n:], v_A[:n]])
    dA_in_centralizer = bool(np.all((H @ v_A_swap) % 2 == 0))
    dA_is_stabilizer = len(f2_rref(np.vstack([H, v_A.reshape(1, -1)]) % 2)[1]) == rank_H
    dA_certified = dA_in_centralizer and not dA_is_stabilizer
    dA_witness_weight = symplectic_weight(v_A, n)

    # d_C: lightest check-qubit logical operator (Case 2)
    valid_cols = [i for i in range(2 * r) if np.any((ACM.T @ I_2r[i]) % 2 != 0)]
    genuine_weights = []
    for i in valid_cols:
        v_C = np.zeros(2 * n, dtype=np.uint8)
        v_C[k + i] = 1
        v_C[n + k: n + k + 2 * r] = M_mat[:, i]
        v_swap = np.concatenate([v_C[n:], v_C[:n]])
        in_centralizer = np.all((H @ v_swap) % 2 == 0)
        is_stabilizer = len(f2_rref(np.vstack([H, v_C.reshape(1, -1)]) % 2)[1]) == rank_H
        if in_centralizer and not is_stabilizer:
            genuine_weights.append(symplectic_weight(v_C, n))

    d_C = min(genuine_weights) if genuine_weights else d_S

    meta = dict(
        alpha=alpha, beta=beta, p=p, J=J,
        r=r, k=k, n=n, R=R, d_S=d_S,
        I_sets=I_sets,
        col_weights_U=col_weights,
        I_min=I_min,
        d_A=d_A,
        d_A_certified=dA_certified,
        d_A_witness_weight=dA_witness_weight,
        d_C=d_C,
        d_bound=min(d_A, d_C),
        n_genuine_C=len(genuine_weights),
    )
    return H, meta
