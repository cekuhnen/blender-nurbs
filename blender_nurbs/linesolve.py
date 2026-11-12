"""Lines that meet curves at a right angle or touch them, for the Line tool.

  from_point(c, S, kind)   the points of curve c where a line from S meets it
                           perpendicular ("perp") or touches it ("tan")
  between(c1, k1, c2, k2)  lines from c1 to c2, perpendicular or tangent at
                           each end: tangent to two circles, the common normal
                           of two curves, tangent to one and square to the other
  nearest_t(c, P)          the parameter of c's point nearest P

A Curve is anything with a parameter range and an evaluator
ev(t) -> (point, first derivative, second derivative), world space: a kernel
curve (kernel_curve) or, in tests, an analytic one. Roots are bracketed on
samples and polished with Newton, so they are exact to rounding.

A line touches a curve only in the curve's plane: tangency is measured there,
and offered only for planar curves (not straight ones) with the line lying in
their plane. A right angle is measured in 3D and works on any curve or edge.
"""

import numpy as np

SAMPLES = 256          # samples along a curve to bracket the roots
TOL = 1e-10            # |cos| (perp) or |sin| (tan) of the angle that counts as solved
FLAT = 1e-7            # planar / straight: off the plane / line by at most this times the size


def kernel_curve(kc, sm, closed=None):
    """A Curve on a kernel curve (world space, as features.input_curve gives)."""
    t0, t1 = kc.parameter_range()

    def ev(t):
        d = sm.evaluate_curve(kc, float(t))
        return (np.array(d["point"], float), np.array(d["tangent"], float), np.array(d["curvature"], float))
    return Curve(ev, t0, t1, closed)


class Curve:
    def __init__(self, ev, t0, t1, closed=None, samples=SAMPLES):
        self.ev, self.t0, self.t1 = ev, float(t0), float(t1)
        self.ts = np.linspace(self.t0, self.t1, samples + 1)
        F = [ev(t) for t in self.ts]
        self.W = np.array([f[0] for f in F], float)
        self.D = np.array([f[1] for f in F], float)
        self.size = float(np.ptp(self.W, axis=0).max()) or 1.0
        if closed is None:
            closed = bool(np.linalg.norm(self.W[0] - self.W[-1]) <= 1e-9 * self.size)
        self.closed = closed
        self.center = self.W.mean(axis=0)
        X = self.W - self.center
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        off_line = X - np.outer(X @ Vt[0], Vt[0])
        self.straight = bool(np.linalg.norm(off_line, axis=1).max() <= FLAT * self.size)
        self.direction = Vt[0]
        planar = not self.straight and float(np.abs(X @ Vt[2]).max()) <= FLAT * self.size
        self.normal = Vt[2] if planar else None

    def wrap(self, t):
        if self.closed:
            return self.t0 + (t - self.t0) % (self.t1 - self.t0)
        return min(max(t, self.t0), self.t1)

    def at(self, t):
        """(t, point, first derivative, second derivative), t wrapped / clamped."""
        t = self.wrap(float(t))
        P, D, DD = self.ev(t)
        return t, np.asarray(P, float), np.asarray(D, float), np.asarray(DD, float)

    @property
    def span(self):
        return self.t1 - self.t0

    def in_plane(self, P, tol=1e-6):
        """P lies in this (planar) curve's plane."""
        return self.normal is not None and abs(float(self.normal @ (np.asarray(P, float) - self.center))) \
            <= tol * self.size


def _cos(D, V):
    return abs(float(D @ V)) / ((np.linalg.norm(D) * np.linalg.norm(V)) or 1.0)


def _sin(n, D, V):
    return abs(float(n @ np.cross(D, V))) / ((np.linalg.norm(D) * np.linalg.norm(V)) or 1.0)


def _brackets(c, f):
    """Parameters where the sampled residual f changes sign (linear guess)."""
    out = []
    for i in range(len(f) - 1):
        a, b = f[i], f[i + 1]
        if a == 0.0:
            out.append(c.ts[i])
        elif a * b < 0.0:
            out.append(c.ts[i] + (c.ts[i + 1] - c.ts[i]) * a / (a - b))
    if f[-1] == 0.0 and not c.closed:
        out.append(c.ts[-1])
    return out


def _newton1(c, S, kind, n, t):
    """Polish a root of the line S -> c(t) being perpendicular / tangent."""
    lim = 0.05 * c.span
    for _ in range(50):
        t, P, D, DD = c.at(t)
        V = P - S
        if kind == "perp":
            F, dF = float(D @ V), float(DD @ V + D @ D)
        else:
            F, dF = float(n @ np.cross(D, V)), float(n @ np.cross(DD, V))
        if dF == 0.0:
            break
        step = max(-lim, min(lim, F / dF))
        t -= step
        if abs(step) <= 1e-15 * (abs(t) + c.span):
            break
    t, P, D, DD = c.at(t)
    V = P - S
    if np.linalg.norm(V) <= 1e-9 * c.size:
        return None
    ok = _cos(D, V) <= TOL if kind == "perp" else _sin(n, D, V) <= TOL
    return (t, P) if ok else None


def _dedupe(sols, size, key=lambda s: s[1]):
    out = []
    for s in sols:
        if all(np.linalg.norm(key(s) - key(o)) > 1e-8 * size for o in out):
            out.append(s)
    return out


def from_point(c, S, kind):
    """[(t, Q)]: the points Q = c(t) where the line from S meets c perpendicular
    ("perp") or touches it ("tan"; only when S lies in c's plane)."""
    S = np.asarray(S, float)
    n = None
    if kind == "tan":
        if not c.in_plane(S):
            return []
        n = c.normal
    V = c.W - S
    f = (c.D * V).sum(axis=1) if kind == "perp" else np.cross(c.D, V) @ n
    sols = [r for r in (_newton1(c, S, kind, n, t) for t in _brackets(c, f)) if r is not None]
    return _dedupe(sols, c.size)


def nearest_t(c, P):
    """Parameter of c's point nearest P."""
    P = np.asarray(P, float)
    t = float(c.ts[int(np.argmin(np.linalg.norm(c.W - P, axis=1)))])
    lim = 0.05 * c.span
    for _ in range(50):
        t, Q, D, DD = c.at(t)
        F, dF = float(D @ (Q - P)), float(DD @ (Q - P) + D @ D)
        if dF <= 0.0:
            break
        step = max(-lim, min(lim, F / dF))
        t -= step
        if abs(step) <= 1e-15 * (abs(t) + c.span):
            break
    return c.wrap(t)


def _plane_for(c1, k1, c2, k2):
    """The plane normal tangency is measured in, None if no tangency is asked,
    False if it cannot hold (a curve is not planar, or the two planes differ)."""
    ns = [c.normal for c, k in ((c1, k1), (c2, k2)) if k == "tan"]
    if not ns:
        return None
    if any(n is None for n in ns):
        return False
    n = ns[0]
    other = c2 if k1 == "tan" else c1
    if other.normal is not None and np.linalg.norm(np.cross(n, other.normal)) > 1e-7:
        return False
    owner = c1 if k1 == "tan" else c2
    if not owner.in_plane(other.W[0]) or not owner.in_plane(other.W[len(other.W) // 2]):
        return False
    return n


def _newton2(c1, k1, c2, k2, n, a, b):
    lim1, lim2 = 0.05 * c1.span, 0.05 * c2.span
    for _ in range(80):
        a, P1, D1, DD1 = c1.at(a)
        b, P2, D2, DD2 = c2.at(b)
        V = P2 - P1
        if k1 == "perp":
            F1, J11, J12 = float(D1 @ V), float(DD1 @ V - D1 @ D1), float(D1 @ D2)
        else:
            F1, J11, J12 = float(n @ np.cross(D1, V)), float(n @ np.cross(DD1, V)), float(n @ np.cross(D1, D2))
        if k2 == "perp":
            F2, J21, J22 = float(D2 @ V), float(-D2 @ D1), float(DD2 @ V + D2 @ D2)
        else:
            F2, J21, J22 = float(n @ np.cross(D2, V)), float(n @ np.cross(D1, D2)), float(n @ np.cross(DD2, V))
        det = J11 * J22 - J12 * J21
        if det == 0.0:
            return None
        da = -(F1 * J22 - F2 * J12) / det
        db = -(J11 * F2 - J21 * F1) / det
        s = min(1.0, lim1 / (abs(da) or 1e-300), lim2 / (abs(db) or 1e-300))
        a, b = a + s * da, b + s * db
        if s == 1.0 and abs(da) <= 1e-15 * (abs(a) + c1.span) and abs(db) <= 1e-15 * (abs(b) + c2.span):
            break
    a, P1, D1, _ = c1.at(a)
    b, P2, D2, _ = c2.at(b)
    V = P2 - P1
    if np.linalg.norm(V) <= 1e-9 * max(c1.size, c2.size):
        return None
    for k, D in ((k1, D1), (k2, D2)):
        if (_cos(D, V) if k == "perp" else _sin(n, D, V)) > TOL:
            return None
    if n is not None and abs(float(n @ V)) > 1e-6 * np.linalg.norm(V):
        return None
    return a, b, P1, P2


def between(c1, k1, c2, k2, seeds):
    """[(t1, t2, Q1, Q2)]: lines from Q1 on c1 to Q2 on c2, perpendicular ("perp")
    or tangent ("tan") to each, polished from the (t1, t2) seeds."""
    n = _plane_for(c1, k1, c2, k2)
    if n is False:
        return []
    sols = [r for r in (_newton2(c1, k1, c2, k2, n, a, b) for a, b in seeds) if r is not None]
    return _dedupe(sols, max(c1.size, c2.size), key=lambda s: np.concatenate([s[2], s[3]]))


def seeds_from(c1, t1, c2, extra=(), count=24):
    """Seeds for between(): the start's parameter on c1 with spread-out ones on c2."""
    ts = np.linspace(c2.t0, c2.t1, count, endpoint=not c2.closed)
    return [(t1, float(t)) for t in ts] + [(t1, float(t)) for t in extra]


def side_direction(c, t, n=None):
    """Unit direction square to c at t: in c's plane when it is planar, else in
    the plane with normal n (the drawing plane). None if undefined."""
    _, _, D, _ = c.at(t)
    m = c.normal if c.normal is not None else n
    if m is None:
        return None
    d = np.cross(np.asarray(m, float), D)
    L = np.linalg.norm(d)
    return None if L <= 1e-12 * (np.linalg.norm(D) or 1.0) else d / L
