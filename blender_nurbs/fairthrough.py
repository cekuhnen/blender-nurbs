"""CV Curve, Through Points with aligned ends: the fairest curve through the
clicks.

The plain way -- each aligned end fixes its CVs (align_start on the plain
interpolation's end derivatives), the CVs left over are solved so the curve
passes through the clicks at their chord-length parameters -- overshoots badly
with G2 / G3 ends: the clicks next to an end sit at parameters where the end's
fixed CVs rule the curve, so the next CV must swing far out to reach them
(curves 30x to 30000x longer than the clicks' polyline).

Here the curve still passes exactly through every click and meets the ends'
conditions exactly, but what those leave free is chosen for the least bending
energy, int |C''|^2:
  * one extra CV per G2 / G3 end, so the system is not square;
  * the free CVs: the least-energy solution of the interpolation conditions
    (a small KKT system);
  * the ends' free numbers (align.EndShape: tangent length, the curvature and
    rate parts the continuity leaves free) and the clicks' parameters (their
    spacing, kept in order through log-gaps): Levenberg-Marquardt on the
    energy, from the plain solution.
"""

import numpy as np

from . import align, geom

TAN_FLOOR = 0.3        # an end's tangent length stays at least this share of the plain solution's

# cached per (N, p): energy factor R (int |C''|^2 = |R Q|^2) and the knots
_CACHE = {}


def _energy_factor(N, p):
    key = (N, p)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    from .surfeval import basis_ders
    U = np.asarray(geom.clamped_uniform_knots(N, p), float)
    # Gauss-Legendre on every knot span: exact for the polynomial C''
    gx, gw = np.polynomial.legendre.leggauss(max(p, 2))
    ks = np.unique(U[p:N + 1])
    ts, ws = [], []
    for a, b in zip(ks[:-1], ks[1:]):
        ts.append(0.5 * (a + b) + 0.5 * (b - a) * gx)
        ws.append(0.5 * (b - a) * gw)
    ts, ws = np.concatenate(ts), np.concatenate(ws)
    span, Nd = basis_ders(U, p, N, ts, 2)
    B2 = np.zeros((len(ts), N))
    for i, s in enumerate(span):
        B2[i, s - p:s + 1] = Nd[i, 2]
    H = (B2 * ws[:, None]).T @ B2
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    R = (V * np.sqrt(np.maximum(w, 0.0))).T            # |R Q|^2 = Q^T H Q
    out = (R, H, U)
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def _basis_rows(U, p, N, t):
    from .surfeval import basis_ders
    span, Nd = basis_ders(U, p, N, np.asarray(t, float), 0)
    A = np.zeros((len(t), N))
    for i, s in enumerate(span):
        A[i, s - p:s + 1] = Nd[i, 0]
    return A


class _Problem:
    def __init__(self, S, N, p, ks, ke):
        self.S, self.N, self.p = S, N, p
        self.R, self.H, self.U = _energy_factor(N, p)
        self.fixed = list(range(ks + 1)) + list(range(N - 1 - ke, N))
        self.free = [i for i in range(N) if i not in self.fixed]
        Hf = self.H[np.ix_(self.free, self.free)]
        self.Hf = Hf + 1e-12 * (np.trace(Hf) or 1.0) * np.eye(len(self.free))
        self.HfF = self.H[np.ix_(self.free, self.fixed)]

    def solve(self, Q, t):
        """Q with its free CVs the least-energy ones through the inner clicks at t."""
        A = _basis_rows(self.U, self.p, self.N, t[1:-1])
        Af, AF = A[:, self.free], A[:, self.fixed]
        QF = Q[self.fixed]
        m, f = Af.shape[0], len(self.free)
        K = np.zeros((f + m, f + m))
        K[:f, :f] = self.Hf
        K[:f, f:] = Af.T
        K[f:, :f] = Af
        rhs = np.vstack([-self.HfF @ QF, self.S[1:-1] - AF @ QF])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        out = Q.copy()
        out[self.free] = sol[:f]
        return out


def _lm(res, x0, clamp, iters=60):
    """Levenberg-Marquardt (numeric Jacobian), stopping once the energy stops falling."""
    x = clamp(np.array(x0, float))
    r = res(x)
    f = float(r @ r)
    mu = 1e-3
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            h = 1e-6 * (1.0 + abs(x[j]))
            e = np.zeros(len(x))
            e[j] = h
            J[:, j] = (res(clamp(x + e)) - r) / h
        A, g = J.T @ J, J.T @ r
        while mu < 1e10:
            xn = clamp(x + np.linalg.solve(A + mu * np.diag(np.diag(A) + 1e-12), -g))
            rn = res(xn)
            fn = float(rn @ rn)
            if np.isfinite(fn) and fn < f:
                done = f - fn < 1e-7 * f
                x, r, f = xn, rn, fn
                mu = max(mu / 3.0, 1e-12)
                break
            mu *= 4.0
        else:
            break
        if done:
            break
    return x


def through_ends(S, deg, ends, extra=None):
    """CVs and degree of a uniform spline through the points S whose ends are
    aligned (ends as align.align_open: {"0" / "1": (spec, frame)}), the
    fairest such. Returns (X, p, specs)."""
    from .drawtools import interpolate, _chord
    S = np.asarray(S, float)
    n = len(S)
    specs = {e: dict(sp_) for e, (sp_, _) in ends.items()}
    ks = align.CONT[specs["0"]["cont"]] if "0" in specs else 0
    ke = align.CONT[specs["1"]["cont"]] if "1" in specs else 0
    if extra is None:
        extra = (ks >= 2) + (ke >= 2)
    N = n + ks + ke + extra
    X0, p0 = interpolate(S, deg)
    p = min(max(deg, align.degree_for([s_["cont"] for s_ in specs.values()])), N - 1)
    U0 = geom.clamped_uniform_knots(n, p0)
    U = geom.clamped_uniform_knots(N, p)
    size = float(np.linalg.norm(np.ptp(S, axis=0))) or 1.0
    Q = np.zeros((N, 3))
    I = np.eye(4)
    shapes = []                                  # (end, EndShape, knots from that end, k)
    for e, k in (("0", ks), ("1", ke)):
        A_, Ua = (X0, U0) if e == "0" else align._reverse(X0, U0)
        d1, d2, _, _ = align._derivs(A_, Ua, p0) if p0 >= 1 else (A_[1] - A_[0], np.zeros(3), 1, 1)
        d3 = align._d3(A_, Ua, p0) if k >= 3 else None
        seg, Ub = (Q, U) if e == "0" else align._reverse(Q, U)
        seg = align._set_start(seg, Ub, p, min(k, p), A_[0], d1, d2, d3)
        if e in ends:
            sp_ = specs[e]
            fr = ends[e][1]
            side = align.orient_side(sp_)
            if side == "curve":
                Se = S if e == "0" else S[::-1]
                fr = align.orient(sp_, fr, np.mean(Se[1:], axis=0) - Se[0])     # off on the curve's side
            elif side == "d1":
                fr = align.orient(sp_, fr, d1)
            seg = align.align_start(seg, Ub, p, sp_, fr, I)
            if k:
                shapes.append((e, align.EndShape(Ub, p, k, sp_.get("mode", "EDGE"), fr, np.eye(3), np.zeros(3),
                                                 size), k))
        Q = seg if e == "0" else align._reverse(seg, Ub)[0]
    t0 = _chord(S, False)
    if t0 is None:
        t0 = np.linspace(0.0, 1.0, n)
    prob = _Problem(S, N, p, ks, ke)
    if n <= 2 and not shapes:
        return Q, p, specs

    xs0 = [sh.numbers(Q if e == "0" else Q[::-1]) for e, sh, _ in shapes]
    sizes = [len(x) for x in xs0]
    # the end tangent may not shrink below TAN_FLOOR of where it starts: the
    # energy (in the parameter) would squeeze it, bunching the end CVs
    floors = [TAN_FLOOR * (abs(float(x[0])) if sh.mode != "SURFACE" else float(np.linalg.norm(x[:2])))
              for x, (_, sh, _) in zip(xs0, shapes)]
    nx = int(sum(sizes))
    g0 = np.log(np.maximum(np.diff(t0), 1e-9))
    v0 = np.concatenate(xs0 + [g0])

    def params(v):
        g = np.exp(np.clip(v[nx:], -30.0, 30.0))
        t = np.concatenate([[0.0], np.cumsum(g)]) / g.sum()
        t[-1] = 1.0
        return t

    def build(v):
        Qb = Q.copy()
        o = 0
        for (e, sh, k), sz in zip(shapes, sizes):
            seg = Qb if e == "0" else Qb[::-1].copy()
            rows = sh.rows(sh.clamp(v[o:o + sz]), seg)
            for i in range(k + 1):
                Qb[i if e == "0" else N - 1 - i] = rows[i]
            o += sz
        return prob.solve(Qb, params(v))

    def res(v):
        return (prob.R @ build(v)).ravel() / size

    def clamp(v):
        v = np.array(v, float)
        o = 0
        for (e, sh, k), sz, fl in zip(shapes, sizes, floors):
            x = sh.clamp(v[o:o + sz])
            if sh.mode == "SURFACE":
                L = float(np.linalg.norm(x[:2]))
                if L < fl:
                    x[:2] = x[:2] * (fl / L) if L > 1e-300 else np.array([fl, 0.0])
            else:
                x[0] = max(x[0], fl)
            v[o:o + sz] = x
            o += sz
        v[nx:] += g0.mean() - v[nx:].mean()          # the gaps' scale is free: keep it put
        return v

    v = _lm(res, v0, clamp)
    Qv = build(v)
    if not np.all(np.isfinite(Qv)):
        Qv = prob.solve(Q, t0)
    return Qv, p, specs
