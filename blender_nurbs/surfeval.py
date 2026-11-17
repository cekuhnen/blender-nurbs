"""Vectorised evaluation of the kernel's B-spline surfaces in numpy.

The kernel evaluates one (u, v) per Python call (a few microseconds each, plus
the call). The analysis overlays need hundreds of thousands of exact points,
normals and curvatures; here they come from the control net directly
(ext.surface_cvs: degrees, knots, CVs, weights), whole arrays at once:

* basis_ders: the non-zero B-spline basis functions and their derivatives at
  many parameters (the textbook Cox-de Boor recursion, run across arrays).
* Surf.grid: S and its partial derivatives on a tensor grid u_i x v_j. A grid
  is separable, so it is two small matrix products per derivative -- the cost
  is that of writing the result.
* Surf.at: the same at scattered (u, v) (feet of points, refinement).
* Surf.feet: closest (u, v) of many points (grid seed + Newton, vectorised).

Rational surfaces are evaluated in homogeneous form and divided out with the
quotient rule (exact derivatives, not finite differences).
"""

import numpy as np

from . import store


def basis_ders(U, p, n, t, nd):
    """(span, N) of the n basis functions of degree p on knots U at the
    parameters t: span (m,) is the knot span of each t (U[span] <= t <
    U[span + 1], clamped to the valid range [p, n - 1]); N (m, nd + 1, p + 1)
    holds the p + 1 functions that are non-zero there (N_{span - p + r}) and
    their derivatives up to order nd."""
    U = np.asarray(U, float)
    t = np.asarray(t, float).ravel()
    m = len(t)
    span = np.clip(np.searchsorted(U, t, side="right") - 1, p, n - 1)
    ndu = np.zeros((m, p + 1, p + 1))
    ndu[:, 0, 0] = 1.0
    left = np.zeros((m, p + 1))
    right = np.zeros((m, p + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(1, p + 1):
            left[:, j] = t - U[span + 1 - j]
            right[:, j] = U[span + j] - t
            saved = np.zeros(m)
            for r in range(j):
                ndu[:, j, r] = right[:, r + 1] + left[:, j - r]          # knot differences
                temp = np.where(ndu[:, j, r] != 0.0, ndu[:, r, j - 1] / ndu[:, j, r], 0.0)
                ndu[:, r, j] = saved + right[:, r + 1] * temp
                saved = left[:, j - r] * temp
            ndu[:, j, j] = saved
        N = np.zeros((m, nd + 1, p + 1))
        N[:, 0, :] = ndu[:, :, p]
        nd_ = min(nd, p)
        for r in range(p + 1):
            a = np.zeros((2, m, p + 1))
            a[0, :, 0] = 1.0
            s1, s2 = 0, 1
            for k in range(1, nd_ + 1):
                d = np.zeros(m)
                rk, pk = r - k, p - k
                if r >= k:
                    den = ndu[:, pk + 1, rk]
                    a[s2, :, 0] = np.where(den != 0.0, a[s1, :, 0] / den, 0.0)
                    d = a[s2, :, 0] * ndu[:, rk, pk]
                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else p - r
                for j in range(j1, j2 + 1):
                    den = ndu[:, pk + 1, rk + j]
                    a[s2, :, j] = np.where(den != 0.0, (a[s1, :, j] - a[s1, :, j - 1]) / den, 0.0)
                    d = d + a[s2, :, j] * ndu[:, rk + j, pk]
                if r <= pk:
                    den = ndu[:, pk + 1, r]
                    a[s2, :, k] = np.where(den != 0.0, -a[s1, :, k - 1] / den, 0.0)
                    d = d + a[s2, :, k] * ndu[:, r, pk]
                N[:, k, r] = d
                s1, s2 = s2, s1
    f = float(p)
    for k in range(1, nd_ + 1):
        N[:, k, :] *= f
        f *= (p - k)
    return span, N


def _ratio(a, b):
    """a / b with 0 / 0 = 0 (the convention of the basis recursion)."""
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b != 0)


def _basis_matrix(U, p, n, t, nd):
    """(nd + 1, m, n) dense basis matrices: row i holds every function's k-th
    derivative at t[i] (for the separable grid products). The recursion runs
    over all functions at once -- a few whole-array steps per degree -- which
    for the short parameter lists of a grid beats the per-span algorithm."""
    U = np.asarray(U, float)
    t = np.asarray(t, float).ravel()
    m = len(t)
    span = np.clip(np.searchsorted(U, t, side="right") - 1, p, n - 1)
    N = np.zeros((m, len(U) - 1))
    N[np.arange(m), span] = 1.0                       # degree 0
    tc = t[:, None]
    low = {}
    for k in range(1, p + 1):
        if k >= p - nd:
            low[k - 1] = N
        i = np.arange(len(U) - 1 - k)
        a = _ratio(tc - U[i], U[i + k] - U[i])
        b = _ratio(U[i + k + 1] - tc, U[i + k + 1] - U[i + 1])
        N = a * N[:, :-1] + b * N[:, 1:]
    out = np.zeros((nd + 1, m, n))
    out[0] = N
    for d in range(1, min(nd, p) + 1):
        # d-th derivative of the degree-p functions from the degree-(p - d) ones
        D = low[p - d]
        for k in range(p - d + 1, p + 1):
            i = np.arange(D.shape[1] - 1)
            D = k * (_ratio(D[:, :-1], U[i + k] - U[i]) - _ratio(D[:, 1:], U[i + k + 1] - U[i + 1]))
        out[d] = D
    return out


class Surf:
    """A kernel B-spline surface in numpy form (raises for anything else)."""

    def __init__(self, surf):
        _, ext = store._k()
        d = ext.surface_cvs(surf)
        self.p, self.q = int(d["degree_u"]), int(d["degree_v"])
        self.nu, self.nv = int(d["count_u"]), int(d["count_v"])
        self.U = np.asarray(d["knots_u"], float)
        self.V = np.asarray(d["knots_v"], float)
        if len(self.U) != self.nu + self.p + 1 or len(self.V) != self.nv + self.q + 1:
            raise ValueError("unexpected knot vector length")
        P = np.asarray(d["points"], float).reshape(self.nu, self.nv, 3)
        w = d.get("weights")
        W = np.asarray(w, float).reshape(self.nu, self.nv) if w else np.ones((self.nu, self.nv))
        self.rational = bool(np.abs(W - 1.0).max() > 1e-12)
        self.Pw = np.concatenate([P * W[..., None], W[..., None]], axis=2)     # (nu, nv, 4)
        self.u_domain = (float(self.U[self.p]), float(self.U[self.nu]))
        self.v_domain = (float(self.V[self.q]), float(self.V[self.nv]))
        self.surf = surf

    # -- evaluation ----------------------------------------------------------

    def _divide(self, A, nd):
        """Homogeneous derivatives {(k, l): (..., 4)} -> Euclidean {(k, l): (..., 3)}."""
        if not self.rational:
            return {kl: a[..., :3] for kl, a in A.items()}
        from math import comb
        w = {kl: a[..., 3:4] for kl, a in A.items()}
        S = {}
        # S_kl = (A_kl - sum over (i, j) != (0, 0) of C(k, i) C(l, j) w_ij S_(k-i)(l-j)) / w_00
        # (Piegl & Tiller A4.4), in order of total degree so every S on the right is known
        for k, l in sorted(self._orders(nd), key=sum):
            v = A[k, l][..., :3].copy()
            for i in range(k + 1):
                for j in range(l + 1):
                    if i or j:
                        v -= comb(k, i) * comb(l, j) * w[i, j] * S[k - i, l - j]
            S[k, l] = v / w[0, 0]
        return S

    @staticmethod
    def _orders(nd):
        return [(k, l) for k in range(nd + 1) for l in range(nd + 1 - k)]

    def grid(self, us, vs, nd=1):
        """{(k, l): (len(us), len(vs), 3)} -- d^(k+l) S / du^k dv^l on the grid."""
        us, vs = np.asarray(us, float), np.asarray(vs, float)
        Bu = _basis_matrix(self.U, self.p, self.nu, us, nd)            # (nd+1, mu, nu)
        Bv = _basis_matrix(self.V, self.q, self.nv, vs, nd)            # (nd+1, mv, nv)
        Pw = self.Pw.reshape(self.nu, -1)
        A = {}
        rows = [(Bu[k] @ Pw).reshape(len(us), self.nv, 4).transpose(0, 2, 1) for k in range(nd + 1)]   # (mu, 4, nv)
        for k, l in self._orders(nd):
            A[k, l] = (rows[k] @ Bv[l].T).transpose(0, 2, 1)                # (mu, mv, 4)
        return self._divide(A, nd)

    def at(self, u, v, nd=1, chunk=20000):
        """{(k, l): (m, 3)} at scattered parameters. Few points: the control
        points of each point's span gathered and summed. Many: the points
        grouped by knot cell (span pair), whose control points are one block --
        a small matrix product per cell instead of a gather per point."""
        u = np.asarray(u, float).ravel()
        v = np.asarray(v, float).ravel()
        m = len(u)
        orders = self._orders(nd)
        if m <= 64:
            return self._at_gather(u, v, nd)
        su, Nu = basis_ders(self.U, self.p, self.nu, u, nd)
        sv, Nv = basis_ders(self.V, self.q, self.nv, v, nd)
        key = su * (self.nv + 1) + sv
        order = np.argsort(key, kind="stable")
        ks, starts = np.unique(key[order], return_index=True)
        ends = np.r_[starts[1:], m]
        A = {kl: np.empty((m, 4)) for kl in orders}
        p1, q1 = self.p + 1, self.q + 1
        for kk, a, b in zip(ks.tolist(), starts.tolist(), ends.tolist()):
            iu, iv = kk // (self.nv + 1), kk % (self.nv + 1)
            idx = order[a:b]
            C = self.Pw[iu - self.p:iu + 1, iv - self.q:iv + 1].reshape(p1, q1 * 4)
            for k in range(nd + 1):
                R = (Nu[idx, k] @ C).reshape(-1, q1, 4)                     # (mc, q+1, 4)
                for l in range(nd + 1 - k):
                    A[k, l][idx] = np.einsum("mb,mbc->mc", Nv[idx, l], R)
        return self._divide(A, nd)

    def _at_gather(self, u, v, nd):
        su, Nu = basis_ders(self.U, self.p, self.nu, u, nd)
        sv, Nv = basis_ders(self.V, self.q, self.nv, v, nd)
        iu = su[:, None] - self.p + np.arange(self.p + 1)[None, :]           # (m, p+1)
        iv = sv[:, None] - self.q + np.arange(self.q + 1)[None, :]           # (m, q+1)
        C = self.Pw[iu[:, :, None], iv[:, None, :]]                         # (m, p+1, q+1, 4)
        A = {}
        rows = [np.einsum("ma,mabc->mbc", Nu[:, k], C) for k in range(nd + 1)]   # (m, q+1, 4)
        for k, l in self._orders(nd):
            A[k, l] = np.einsum("mb,mbc->mc", Nv[:, l], rows[k])
        return self._divide(A, nd)

    def periodic(self):
        """(u, v): whether the surface wraps around in that direction."""
        try:
            return store._periodic(self.surf)
        except Exception:
            return False, False

    def feet(self, Q, g=33, iters=12, per=None, seed=None):
        """Closest (u, v) of every point of Q: from seed (m, 2), else from the
        nearest point of a g x g grid, then Newton on the normal equations, all
        points at once."""
        Q = np.asarray(Q, float).reshape(-1, 3)
        (u0, u1), (v0, v1) = self.u_domain, self.v_domain
        if seed is not None:
            u, v = np.array(seed[:, 0], float), np.array(seed[:, 1], float)
        else:
            gu, gv = np.linspace(u0, u1, g), np.linspace(v0, v1, g)
            flat = self.grid(gu, gv, 0)[0, 0].reshape(-1, 3)
            k = store.nearest_rows(flat, Q)
            u, v = gu[k // g].copy(), gv[k % g].copy()
        pu, pv = self.periodic() if per is None else per
        for _ in range(iters):
            D = self.at(u, v, 1)
            r = Q - D[0, 0]
            su, sv = D[1, 0], D[0, 1]
            a = np.einsum("ij,ij->i", su, su)
            b = np.einsum("ij,ij->i", su, sv)
            c = np.einsum("ij,ij->i", sv, sv)
            g1 = np.einsum("ij,ij->i", su, r)
            g2 = np.einsum("ij,ij->i", sv, r)
            det = a * c - b * b
            okd = det > 1e-12 * np.maximum(a * c, 1e-300)
            safe = np.where(okd, det, 1.0)
            du = np.where(okd, (c * g1 - b * g2) / safe, g1 / np.maximum(a, 1e-300))
            dv = np.where(okd, (a * g2 - b * g1) / safe, g2 / np.maximum(c, 1e-300))
            du = np.nan_to_num(du)
            dv = np.nan_to_num(dv)
            un = (u0 + (u + du - u0) % (u1 - u0)) if pu else np.clip(u + du, u0, u1)
            vn = (v0 + (v + dv - v0) % (v1 - v0)) if pv else np.clip(v + dv, v0, v1)
            moved = float(np.max(np.abs(un - u), initial=0.0)) / (u1 - u0) + \
                float(np.max(np.abs(vn - v), initial=0.0)) / (v1 - v0)
            u, v = un, vn
            if moved < 1e-12:                    # (a foot held at the border stops moving too)
                break
        return np.stack([u, v], axis=1)


def normals(D):
    """Unit su x sv of an evaluation dict (zero where it vanishes)."""
    n = np.cross(D[1, 0], D[0, 1])
    ln = np.linalg.norm(n, axis=-1)
    return n / np.where(ln > 0, ln, 1.0)[..., None], ln


def forms(D):
    """(n, Q, C) of an evaluation dict with third derivatives (nd >= 3): the
    unit su x sv normal, and the surface's height over its tangent plane,
    h(x) = 1/2 x.Q.x + 1/6 C(x, x, x) + ..., as world-space tensors acting on
    tangent vectors: Q (..., 3, 3) the second fundamental form, C (..., 3, 3, 3)
    the third-order one (symmetric). C(c, c, c) for a unit tangent c is the
    rate of change dk/ds of the normal curvature along the normal section in
    direction c -- what G3 compares across an edge.

    Along a parameter line c(s) through the point with c' = x (tangent):
        C(x, x, x) = n . c''' - 3 Q(x, P c'')       (P: the tangent part)
    which is the expansion below in the (u, v) coordinates of x."""
    su, sv = D[1, 0], D[0, 1]
    n, _ = normals(D)
    J = np.stack([su, sv], axis=-1)                                     # (..., 3, 2)
    JtJ = np.einsum("...ia,...ib->...ab", J, J)
    det = JtJ[..., 0, 0] * JtJ[..., 1, 1] - JtJ[..., 0, 1] ** 2
    det = np.where(np.abs(det) > 1e-300, det, 1e-300)
    inv = np.stack([np.stack([JtJ[..., 1, 1], -JtJ[..., 0, 1]], -1),
                    np.stack([-JtJ[..., 1, 0], JtJ[..., 0, 0]], -1)], -2) / det[..., None, None]
    Jp = np.einsum("...ab,...ib->...ai", inv, J)                        # (..., 2, 3): x -> (u, v) coords
    S2 = np.stack([np.stack([D[2, 0], D[1, 1]], -2), np.stack([D[1, 1], D[0, 2]], -2)], -3)   # (..., 2, 2, 3)
    d3 = {0: D[3, 0], 1: D[2, 1], 2: D[1, 2], 3: D[0, 3]}               # by the number of v's
    S3 = np.stack([np.stack([np.stack([d3[a + b + c] for c in (0, 1)], -2) for b in (0, 1)], -3)
                   for a in (0, 1)], -4)                                # (..., 2, 2, 2, 3)
    H = np.einsum("...abi,...i->...ab", S2, n)
    JpS2 = np.einsum("...di,...bci->...bcd", Jp, S2)
    raw = np.einsum("...abci,...i->...abc", S3, n) - 3.0 * np.einsum("...ad,...bcd->...abc", H, JpS2)
    T3 = sum(np.moveaxis(raw, (-3, -2, -1), perm) for perm in
             ((-3, -2, -1), (-3, -1, -2), (-2, -3, -1), (-2, -1, -3), (-1, -3, -2), (-1, -2, -3))) / 6.0
    Q = np.einsum("...ab,...ai,...bj->...ij", H, Jp, Jp)
    C = np.einsum("...abc,...ai,...bj,...ck->...ijk", T3, Jp, Jp, Jp)
    return n, Q, C


def cubic(C, x):
    """C(x, x, x) of third-order forms C (..., 3, 3, 3) and vectors x (..., 3)."""
    return np.einsum("...ijk,...i,...j,...k->...", C, x, x, x)


def curvatures(D):
    """(K, H, k1, k2) of an evaluation dict with second derivatives, relative
    to the su x sv normal (k1 >= k2)."""
    su, sv = D[1, 0], D[0, 1]
    n, ln = normals(D)
    E = np.einsum("...i,...i->...", su, su)
    F = np.einsum("...i,...i->...", su, sv)
    G = np.einsum("...i,...i->...", sv, sv)
    L = np.einsum("...i,...i->...", D[2, 0], n)
    M = np.einsum("...i,...i->...", D[1, 1], n)
    N = np.einsum("...i,...i->...", D[0, 2], n)
    den = E * G - F * F
    good = den > 1e-24 * np.maximum((E + G) ** 2, 1e-300)
    den = np.where(good, den, 1.0)
    K = np.where(good, (L * N - M * M) / den, 0.0)
    H = np.where(good, (E * N - 2.0 * F * M + G * L) / (2.0 * den), 0.0)
    d = np.sqrt(np.maximum(H * H - K, 0.0))
    return K, H, H + d, H - d, good
