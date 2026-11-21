"""Surface fillets built from exact cross-sections (the worker's chord and
blend fillets).

Per section -- a plane square to the surfaces' intersection curve -- a ball
touches both surfaces: its centre C sits at signed distance r_A from A and r_B
from B (Newton on the two distances; their gradients are the unit normals at
the feet). The feet are the rails. Chord: one radius per section, found so
that the rails are exactly the chord apart (the angle between the surfaces
changes along the crossing, so the radius does). Between the rails the
section is a circular arc (exact, rational), a straight chamfer, or a G1
cubic / G2 quintic / G3 degree-7 blend: tangent to each surface square
across the rail and, G2, with its normal curvature in that direction; G3
also with the rate that curvature changes at (the surface's third-order
form, from the exact third derivatives of its CV net). SOFT2 / SOFT3: the
soft fillet (softfillet.py), a G2 quintic / G3 degree-7 span whose curvature
peaks at exactly 1 / R in the middle; each section's ball is sized from its
own angle so the rails sit where that span needs them.

The sections are skinned through, in homogeneous coordinates (rational arcs
stay exact), periodic round a closed crossing; a section is added halfway
wherever the surface misses that section's rails, middle or tangent planes
by more than the tolerance there.

Pure numpy + the kernel module (no bpy): imported by worker.py."""

import math

import numpy as np

SHAPES = ("CIRCULAR", "LINEAR", "G1", "G2", "G3", "SOFT2", "SOFT3")
MAX_SECTIONS = 400
ANGLE_TOL = math.radians(0.02)       # surface normal vs a surface's along a rail (labels: 0.1 deg)


# ---------------------------------------------------------------------------
# B-spline helpers (interpolation, periodic too)
# ---------------------------------------------------------------------------

def _basis(U, p, t):
    """All degree-p B-spline basis values at t on knot vector U (len(U) - p - 1)."""
    U = np.asarray(U, float)
    m = len(U) - 1
    N = np.zeros(m)
    if t >= U[-1]:
        k = max(i for i in range(m) if U[i] < U[i + 1])
    else:
        k = int(np.searchsorted(U, t, side="right") - 1)
    N[k] = 1.0
    for d in range(1, p + 1):
        M = np.zeros(m - d)
        for i in range(m - d):
            a = 0.0
            if N[i] and U[i + d] > U[i]:
                a = (t - U[i]) / (U[i + d] - U[i]) * N[i]
            b = 0.0
            if N[i + 1] and U[i + d + 1] > U[i + 1]:
                b = (U[i + d + 1] - t) / (U[i + d + 1] - U[i + 1]) * N[i + 1]
            M[i] = a + b
        N = M
    return N


def _basis_ders(U, p, t, nd):
    """[N, N', ..., N^(nd)]: the degree-p basis values and their derivatives
    at t (each len(U) - p - 1), from the lower degrees' values."""
    U = np.asarray(U, float)
    m = len(U) - 1
    N = np.zeros(m)
    if t >= U[-1]:
        k = max(i for i in range(m) if U[i] < U[i + 1])
    else:
        k = int(np.searchsorted(U, t, side="right") - 1)
    N[k] = 1.0
    levels = [N]
    for d in range(1, p + 1):
        i = np.arange(m - d)
        a, b = U[i + d] - U[i], U[i + d + 1] - U[i + 1]
        lo = np.divide((t - U[i]) * N[:-1], a, out=np.zeros(m - d), where=a > 0)
        hi = np.divide((U[i + d + 1] - t) * N[1:], b, out=np.zeros(m - d), where=b > 0)
        N = lo + hi
        levels.append(N)
    out = []
    for r in range(nd + 1):
        if r > p:
            out.append(np.zeros(m - p))
            continue
        D = levels[p - r]
        for d in range(p - r + 1, p + 1):
            i = np.arange(len(D) - 1)
            a, b = U[i + d] - U[i], U[i + d + 1] - U[i + 1]
            D = d * (np.divide(D[:-1], a, out=np.zeros(len(i)), where=a > 0)
                     - np.divide(D[1:], b, out=np.zeros(len(i)), where=b > 0))
        out.append(D)
    return out


class _Exact:
    """A kernel surface's partial derivatives up to the third, exact from
    its CV net (rational: the quotient rule)."""

    def __init__(self, ext, surf):
        d = ext.surface_cvs(surf)
        self.p, self.q = int(d["degree_u"]), int(d["degree_v"])
        nu, nv = int(d["count_u"]), int(d["count_v"])
        self.U, self.V = list(d["knots_u"]), list(d["knots_v"])
        P = np.asarray(d["points"], float).reshape(nu, nv, 3)
        w = d.get("weights")
        W = np.asarray(w, float).reshape(nu, nv) if w else np.ones((nu, nv))
        self.Pw = np.concatenate([P * W[..., None], W[..., None]], axis=2)

    def ders(self, u, v):
        """{(k, l): d^(k+l) S / du^k dv^l} for k + l <= 3."""
        Bu, Bv = _basis_ders(self.U, self.p, u, 3), _basis_ders(self.V, self.q, v, 3)
        A = {(k, l): np.einsum("i,j,ijc->c", Bu[k], Bv[l], self.Pw)
             for k in range(4) for l in range(4 - k)}
        S = {}
        for k, l in sorted(A, key=sum):
            x = A[k, l][:3].copy()
            for i in range(k + 1):
                for j in range(l + 1):
                    if i or j:
                        x -= math.comb(k, i) * math.comb(l, j) * A[i, j][3] * S[k - i, l - j]
            S[k, l] = x / A[0, 0][3]
        return S


def _insert(P, U, p, t):
    """Boehm: t inserted once. P (n, dim)."""
    U = list(U)
    k = int(np.searchsorted(U, t, side="right") - 1)
    k = min(k, len(U) - p - 2)
    Q = []
    for i in range(len(P) + 1):
        if i <= k - p:
            Q.append(P[i])
        elif i > k:
            Q.append(P[i - 1])
        else:
            den = U[i + p] - U[i]
            a = (t - U[i]) / den if den > 0 else 0.0
            Q.append((1 - a) * P[i - 1] + a * P[i])
    U.insert(k + 1, t)
    return np.array(Q), U


def _clamp_start(P, U, p):
    a = U[p]
    while sum(abs(k - a) <= 1e-14 for k in U) < p:
        P, U = _insert(P, U, p, a)
    s = next(i for i, k in enumerate(U) if abs(k - a) <= 1e-14)
    return P[s - 1:], [a] * (p + 1) + list(U[s + p:])


def _clamp(P, U, p):
    """An unclamped spline over [U[p], U[-p-1]] as a clamped one."""
    P, U = _clamp_start(P, U, p)
    a, b = U[0], U[-1]
    R, V = _clamp_start(P[::-1], [a + b - k for k in reversed(U)], p)
    return R[::-1], [a + b - k for k in reversed(V)]


def interp_open(H, t, q=3):
    """(knots, CVs) of the degree-q spline through rows H (n, ...) at t (0..1)."""
    n = len(H)
    q = max(1, min(q, n - 1))
    U = [0.0] * (q + 1) + [float(np.mean(t[j:j + q])) for j in range(1, n - q)] + [1.0] * (q + 1)
    A = np.array([_basis(U, q, x)[:n] for x in t])
    X = np.linalg.solve(A, H.reshape(n, -1)).reshape(H.shape)
    return q, U, X


def interp_closed(H, t):
    """(degree, knots, CVs) of the periodic cubic through rows H (n >= 4, ...)
    at t (t[0] = 0 < ... < 1, period 1), clamped at t = 0: C2 across the seam."""
    n, q = len(H), 3
    if n < 4:
        raise ValueError("a closed fillet needs 4 sections at least")
    kn = lambda k: float(t[k % n]) + math.floor(k / n)
    U = [kn(k) for k in range(-q, n + q + 1)]            # U[q] = 0, U[n + q] = 1
    ncv = n + q                                          # CV k is P[k mod n]
    A = np.zeros((n, n))
    for i, x in enumerate(t):
        N = _basis(U, q, float(x))[:ncv]
        for k in range(ncv):
            A[i, k % n] += N[k]
    X = np.linalg.solve(A, H.reshape(n, -1))
    Q = np.array([X[k % n] for k in range(ncv)])
    Pc, Uc = _clamp(Q, U, q)
    return q, [float(u) for u in Uc], Pc.reshape((len(Pc),) + H.shape[1:])


# ---------------------------------------------------------------------------
# Surfaces: feet, normals, curvature
# ---------------------------------------------------------------------------

class Base:
    """One of the two surfaces (all faces of a brep), normals turned to the
    ball's side (side +1: the faces' outward normals, -1: the other way)."""

    def __init__(self, sm, brep, side):
        self.sm = sm
        self.faces = list(brep.faces())
        self.surfs = [f.surface() for f in self.faces]
        self.orient = []
        for f, s in zip(self.faces, self.surfs):
            (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
            u, v = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            _, su, sv = s.derivatives(u, v)
            n = np.cross(su, sv)
            try:
                o = np.array(f.outward_normal_at_uv(u, v), float)
                self.orient.append(1.0 if float(n @ o) >= 0 else -1.0)
            except Exception:
                self.orient.append(1.0)
        self.side = float(side)
        self._exact = {}

    def curvature_rate(self, where, n, t):
        """(normal curvature, its rate dk/ds) in the unit tangent direction t
        (sign: along n), exact: k = II(t, t), dk/ds = C(t, t, t) -- along the
        parameter line through the point with c' = t, n.c''' - 3 II(t, P c'')."""
        fi, u, v = where
        if fi not in self._exact:
            self._exact[fi] = _Exact(self.sm.blender_ext, self.surfs[fi])
        D = self._exact[fi].ders(u, v)
        J = np.array([D[1, 0], D[0, 1]]).T
        a, c = np.linalg.lstsq(J, t, rcond=None)[0]
        H = np.array([[D[2, 0] @ n, D[1, 1] @ n], [D[1, 1] @ n, D[0, 2] @ n]])
        c2 = a * a * D[2, 0] + 2 * a * c * D[1, 1] + c * c * D[0, 2]
        c3 = a ** 3 * D[3, 0] + 3 * a * a * c * D[2, 1] + 3 * a * c * c * D[1, 2] + c ** 3 * D[0, 3]
        b = np.linalg.lstsq(J, c2 - (c2 @ n) * n, rcond=None)[0]
        ac = np.array([a, c])
        return float(ac @ H @ ac), float(c3 @ n - 3.0 * (ac @ H @ b))

    def face_of(self, q):
        if len(self.faces) == 1:
            return 0
        return min(range(len(self.faces)), key=lambda i: self.faces[i].closest_point(tuple(q))[1])

    def foot(self, q, fi=None):
        """(point, unit normal on the ball's side, (face, u, v)) nearest q."""
        fi = self.face_of(q) if fi is None else fi
        s = self.surfs[fi]
        _, uvs, _ = self.sm.surface_closest_point(s, tuple(float(x) for x in q))
        u, v = _uv(s, uvs[0])
        pt, su, sv = s.derivatives(u, v)
        n = np.cross(su, sv)
        n = n / (np.linalg.norm(n) or 1.0) * self.orient[fi] * self.side
        return np.array(pt, float), n, (fi, u, v)

    def normal_curvature(self, where, n, t):
        """Normal curvature (sign: along n) in the tangent direction t."""
        fi, u, v = where
        s = self.surfs[fi]
        su, sv, suu, suv, svv = _second(s, u, v)
        a, c = np.linalg.lstsq(np.array([su, sv]).T, t, rcond=None)[0]
        return float(a * a * (suu @ n) + 2 * a * c * (suv @ n) + c * c * (svv @ n)) / max(float(t @ t), 1e-30)


def _uv(surf, uv):
    """A closest point's (u, v), inside the surface's domain (the kernel can
    return one a rounding past its edge, which it then will not evaluate)."""
    u, v = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    return min(max(float(u), u0), u1), min(max(float(v), v0), v1)


def _second(surf, u, v):
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    hu, hv = 1e-5 * (u1 - u0), 1e-5 * (v1 - v0)
    _, su, sv = (np.array(x, float) for x in surf.derivatives(u, v))

    def at(uu, vv):
        _, a, b = surf.derivatives(uu, vv)
        return np.array(a, float), np.array(b, float)

    def d(t, lo, hi, h, f):
        if t - h >= lo and t + h <= hi:
            (a0, b0), (a1, b1) = f(t - h), f(t + h)
            return (a1 - a0) / (2 * h), (b1 - b0) / (2 * h)
        s_ = 1.0 if t - h < lo else -1.0
        (a1, b1), (a2, b2) = f(t + s_ * h), f(t + 2 * s_ * h)
        return (s_ * (-3 * su + 4 * a1 - a2) / (2 * h), s_ * (-3 * sv + 4 * b1 - b2) / (2 * h))

    suu, svu = d(u, u0, u1, hu, lambda t: at(t, v))
    suv, svv = d(v, v0, v1, hv, lambda t: at(u, t))
    return su, sv, suu, 0.5 * (suv + svu), svv


# ---------------------------------------------------------------------------
# The crossing
# ---------------------------------------------------------------------------

class Crossing:
    """An intersection chain -- the kernel's curves joined end to end (it
    breaks a crossing where it passes an edge or seam of either surface) --
    by arc length s in 0..1."""

    def __init__(self, chain, m=401):
        self.pieces = []                       # (curve, reversed, t samples, cumulative length from the chain's start)
        pts, start = [], 0.0
        for c, rev in chain:
            t0, t1 = c.parameter_range()
            T = np.linspace(t1, t0, m) if rev else np.linspace(t0, t1, m)
            P = np.array([c.evaluate(t) for t in T], float)
            d = start + np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            self.pieces.append((c, T, d))
            pts.append(P)
            start = float(d[-1])
        self.length = start
        P = np.vstack(pts)
        self.closed = len(P) > 2 and float(np.linalg.norm(P[0] - P[-1])) <= 1e-5 * max(self.length, 1e-12)
        self.ends = np.array([q[2][0] for q in self.pieces[1:]])
        self.off = 0.0          # closed: where s = 0 lies (see start_away)

    def start_away(self, polys):
        """A closed crossing restarted at its point farthest from the edges
        (polylines): the fillet's seam then lies away from the surfaces' own
        seams -- a rail cutter whose seam lies on a closed surface's seam
        does not cut it into regions."""
        if not self.closed or not polys:
            return
        ks = np.linspace(0.0, 1.0, 96, endpoint=False)
        Q = np.array([self._pos(k) for k in ks])
        E = np.vstack(polys)
        d = np.min(np.linalg.norm(Q[:, None, :] - E[None, :, :], axis=2), axis=1)
        self.off = float(ks[int(np.argmax(d))])

    def _pos(self, s):
        s = s % 1.0 if self.closed else min(max(s, 0.0), 1.0)
        L = s * self.length
        k = int(np.searchsorted(self.ends, L, side="right"))
        c, T, d = self.pieces[k]
        return np.array(c.evaluate(float(np.interp(L, d, T))), float)

    def at(self, s):
        if self.closed:
            s = (s + self.off) % 1.0
        h = 1e-5
        if not self.closed:
            lo, hi = max(s - h, 0.0), min(s + h, 1.0)
        else:
            lo, hi = s - h, s + h
        T = self._pos(hi) - self._pos(lo)
        return self._pos(s), T / (np.linalg.norm(T) or 1.0)


def chains(curves, tol):
    """The curves joined end to end where their ends meet: [[(curve, reversed)]]."""
    ends = []
    for c in curves:
        t0, t1 = c.parameter_range()
        ends.append((np.array(c.evaluate(t0), float), np.array(c.evaluate(t1), float)))
    left = list(range(len(curves)))
    out = []
    while left:
        i = left.pop(0)
        ch = [(i, False)]
        head, tail = ends[i][0], ends[i][1]
        grown = True
        while grown and left:
            grown = False
            for j in list(left):
                a, b = ends[j]
                if np.linalg.norm(a - tail) <= tol:
                    ch.append((j, False)); tail = b
                elif np.linalg.norm(b - tail) <= tol:
                    ch.append((j, True)); tail = a
                elif np.linalg.norm(b - head) <= tol:
                    ch.insert(0, (j, False)); head = a
                elif np.linalg.norm(a - head) <= tol:
                    ch.insert(0, (j, True)); head = b
                else:
                    continue
                left.remove(j)
                grown = True
        out.append([(curves[k], r) for k, r in ch])
    return out


# ---------------------------------------------------------------------------
# One section
# ---------------------------------------------------------------------------

class Section:
    __slots__ = ("s", "P0", "C", "pa", "pb", "na", "nb", "wa", "wb", "ra", "rb", "turn")


def _plane(T):
    a = np.array([1.0, 0.0, 0.0]) if abs(T[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(T, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(T, e1)


class Solver:
    def __init__(self, A, B, ra, rb, chord=None, scale=1.0, soft=None):
        self.A, self.B = A, B
        self.ra, self.rb, self.chord = ra, rb, chord
        self.scale = scale
        self.soft = soft            # (R, smoothing, cont): each section's ball from its own angle

    def ball(self, P0, T, ra, rb, C):
        """Centre in the plane (P0, T) at signed distances ra from A, rb from B."""
        e1, e2 = _plane(T)
        C = P0 + ((C - P0) @ e1) * e1 + ((C - P0) @ e2) * e2
        fa = fb = None
        for _ in range(60):
            pa, na, wa = self.A.foot(C, fa)
            pb, nb, wb = self.B.foot(C, fb)
            fa, fb = wa[0], wb[0]
            F = np.array([na @ (C - pa) - ra, nb @ (C - pb) - rb])
            J = np.array([[na @ e1, na @ e2], [nb @ e1, nb @ e2]])
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                raise RuntimeError("the surfaces are tangent here: no ball fits between them")
            lim = 0.5 * max(ra, rb)                     # damped: never jump past the ball's size
            nd = float(np.linalg.norm(dx))
            if nd > lim:
                dx *= lim / nd
            C = C + dx[0] * e1 + dx[1] * e2
            if nd < 1e-12 * self.scale:
                break
        pa, na, wa = self.A.foot(C, fa)
        pb, nb, wb = self.B.foot(C, fb)
        tol_a, tol_b = 1e-6 * max(ra, 1e-9) + 1e-9 * self.scale, 1e-6 * max(rb, 1e-9) + 1e-9 * self.scale
        # the whole offset, not only its normal part: a foot clamped to a
        # surface's edge (the ball past where the surface ends) is no contact
        if np.linalg.norm(C - pa - ra * na) > 10 * tol_a or np.linalg.norm(C - pb - rb * nb) > 10 * tol_b:
            raise RuntimeError("no ball of this size fits between the surfaces here")
        return C, pa, na, wa, pb, nb, wb

    def start(self, P0, ra, rb):
        """A first centre from the normals at the crossing point."""
        _, na, _ = self.A.foot(P0)
        _, nb, _ = self.B.foot(P0)
        g = float(na @ nb)
        det = max(1.0 - g * g, 1e-6)
        ca, cb = (na - g * nb) / det, (nb - g * na) / det
        return P0 + ra * ca + rb * cb, na, nb

    def section(self, cr, s, guess=None):
        P0, T = cr.at(s)
        if self.chord:
            C0, na0, nb0 = self.start(P0, 1.0, 1.0)
            turn = math.pi - math.acos(max(-1.0, min(1.0, float(-na0 @ nb0))))
            k = 2.0 * math.sin(0.5 * max(turn, math.radians(1.0)))
            r = guess if guess else self.chord / k
            C = P0 + r * (C0 - P0)
            r_prev = ch_prev = None
            for _ in range(40):
                C, pa, na, wa, pb, nb, wb = self.ball(P0, T, r, r, C)
                ch = float(np.linalg.norm(pa - pb))
                err = ch - self.chord
                if abs(err) <= 1e-10 * self.chord:
                    break
                if r_prev is not None and abs(ch - ch_prev) > 1e-15:
                    r_new = r - err * (r - r_prev) / (ch - ch_prev)       # secant
                else:
                    r_new = r * self.chord / (ch or self.chord)
                r_new = min(max(r_new, 0.5 * r), 2.0 * r)
                C = P0 + (C - P0) * (r_new / r)
                r_prev, ch_prev, r = r, ch, r_new
            ra = rb = r
        elif self.soft:
            import softfillet
            R, smooth, cont = self.soft
            C0, na0, nb0 = self.start(P0, 1.0, 1.0)
            turn = math.acos(max(-1.0, min(1.0, float(na0 @ nb0))))
            r = guess if guess else softfillet.ball_radius(R, turn, smooth, cont)
            C = P0 + r * (C0 - P0)
            for _ in range(40):
                C, pa, na, wa, pb, nb, wb = self.ball(P0, T, r, r, C)
                turn = math.acos(max(-1.0, min(1.0, float(na @ nb))))
                r_new = softfillet.ball_radius(R, turn, smooth, cont)
                if abs(r_new - r) <= 1e-12 * r:
                    break
                C = P0 + (C - P0) * (r_new / r)
                r = r_new
            # on curved faces the faces' curvature bends the span: size the ball
            # so this section's own sharpest point has radius R exactly (secant)
            shape = "SOFT3" if cont >= 3 else "SOFT2"
            r_prev = e_prev = None
            for _ in range(12):
                x = Section()
                x.s, x.P0, x.C, x.pa, x.pb, x.na, x.nb, x.wa, x.wb, x.ra, x.rb = s, P0, C, pa, pb, na, nb, wa, wb, r, r
                x.turn = math.acos(max(-1.0, min(1.0, float(na @ nb))))
                _, rows = section_rows(x, shape, self.A, self.B, self.soft)
                err = softfillet.peak_radius(rows[:, :3] / rows[:, 3:]) - R
                if abs(err) <= 1e-10 * R:
                    break
                if r_prev is not None and abs(err - e_prev) > 1e-300:
                    r_new = r - err * (r - r_prev) / (err - e_prev)
                else:
                    r_new = r * R / (err + R)
                r_new = min(max(r_new, 0.5 * r), 2.0 * r)
                C = P0 + (C - P0) * (r_new / r)
                r_prev, e_prev, r = r, err, r_new
                C, pa, na, wa, pb, nb, wb = self.ball(P0, T, r, r, C)
            ra = rb = r
        else:
            ra, rb = self.ra, self.rb
            C = guess if guess is not None else self.start(P0, ra, rb)[0]
            C, pa, na, wa, pb, nb, wb = self.ball(P0, T, ra, rb, C)
        x = Section()
        x.s, x.P0, x.C, x.pa, x.pb, x.na, x.nb, x.wa, x.wb, x.ra, x.rb = s, P0, C, pa, pb, na, nb, wa, wb, ra, rb
        x.turn = math.acos(max(-1.0, min(1.0, float(na @ nb))))
        return x


# ---------------------------------------------------------------------------
# Section curves (homogeneous rows) and the skin
# ---------------------------------------------------------------------------

def _across(x, base, p, n, other):
    """Unit direction tangent to the surface at p (normal n), toward the other rail."""
    d = other - p
    d = d - (d @ n) * n
    return d / (np.linalg.norm(d) or 1.0)


def section_rows(x, shape, A, B, soft=None):
    """(degree, rows (deg+1, 4) homogeneous) of the section's curve, rail A first."""
    pa, pb = x.pa, x.pb
    h1 = lambda p, w=1.0: np.append(np.asarray(p, float) * w, w)
    if shape in ("SOFT2", "SOFT3"):
        import softfillet
        _, smooth, cont = soft
        sh = softfillet.shape(x.turn, smooth, cont)
        ta = _across(x, A, pa, x.na, pb)
        tb = _across(x, B, pb, x.nb, pa)
        L = 0.5 * (x.ra + x.rb) * math.tan(0.5 * x.turn)
        if cont >= 3:
            ka, ka3 = A.curvature_rate(x.wa, x.na, ta)
            kb, kb3 = B.curvature_rate(x.wb, x.nb, tb)
        else:
            ka, kb = A.normal_curvature(x.wa, x.na, ta), B.normal_curvature(x.wb, x.nb, tb)
            ka3 = kb3 = 0.0
        P = softfillet.section_cvs(pa, ta, x.na, ka, pb, tb, x.nb, kb, L, sh["spacing"], sh["degree"], ka3, kb3)
        return sh["degree"], np.array([h1(q) for q in P])
    if shape == "LINEAR":
        return 1, np.array([h1(pa), h1(pb)])
    if shape == "CIRCULAR":
        a, b = pa - x.C, pb - x.C
        cosw = float(a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
        om = math.acos(max(-1.0, min(1.0, cosw)))
        w = math.cos(0.5 * om)
        r = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
        m = a + b
        m = m / (np.linalg.norm(m) or 1.0)
        W = x.C + m * r / max(w, 1e-9)
        return 2, np.array([h1(pa), h1(W, w), h1(pb)])
    ta = _across(x, A, pa, x.na, pb)
    tb = _across(x, B, pb, x.nb, pa)
    d = float(np.linalg.norm(pa - pb))
    if shape == "G1":
        h = d / 3.0
        return 3, np.array([h1(pa), h1(pa + h * ta), h1(pb + h * tb), h1(pb)])
    if shape == "G3":
        # degree 7, h = d / 7: C'(0) = 7 h ta, C''(0) = 42 (P2 - 2 P1 + P0) = (7 h)^2 k n,
        # C'''(0) = 210 (P3 - 3 P2 + 3 P1 - P0) = (7 h)^3 k' n (its part in the
        # tangent plane is a reparameterisation: 0) -- each side the same way
        h = d / 7.0
        rows = []
        for p, t, n, base, where in ((pa, ta, x.na, A, x.wa), (pb, tb, x.nb, B, x.wb)):
            k, k3 = base.curvature_rate(where, n, t)
            P1 = p + h * t
            P2 = p + 2 * h * t + (7.0 / 6.0) * h * h * k * n
            P3 = 3 * P2 - 3 * P1 + p + (343.0 / 210.0) * h ** 3 * k3 * n
            rows.append([p, P1, P2, P3])
        return 7, np.array([h1(q) for q in rows[0] + rows[1][::-1]])
    # G2 quintic: C'(0) = 5 h ta, C''(0) . n = 25 h^2 k  ->  P2 = P0 + 2 h ta + 1.25 h^2 k n
    h = d / 5.0
    ka = A.normal_curvature(x.wa, x.na, ta)
    kb = B.normal_curvature(x.wb, x.nb, tb)
    return 5, np.array([h1(pa), h1(pa + h * ta), h1(pa + 2 * h * ta + 1.25 * h * h * ka * x.na),
                        h1(pb + 2 * h * tb + 1.25 * h * h * kb * x.nb), h1(pb + h * tb), h1(pb)])


def skin(ext, secs, rows, deg, closed):
    """Kernel surface through the sections' rows (u along the crossing, v across)."""
    H = np.array(rows, float)                                  # (n, m, 4)
    P = H[..., :3] / H[..., 3:]
    n = len(H)
    if closed:
        d = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=2)
    else:
        d = np.linalg.norm(np.diff(P, axis=0), axis=2)
    tot = d.sum(axis=0)
    d = np.where(tot > 1e-14, d / np.where(tot > 1e-14, tot, 1.0), 1.0 / len(d))
    t = np.concatenate([[0.0], np.cumsum(d.mean(axis=1))])
    if closed:
        t = t[:n]
        q, U, X = interp_closed(H, t)
    else:
        t /= t[-1]
        q, U, X = interp_open(H, t, 3)
    if (X[..., 3] <= 1e-9).any():
        raise ValueError("the fillet's sections give non-positive weights")
    Uv = [0.0] * (deg + 1) + [1.0] * (deg + 1)
    pts = [tuple(float(c) for c in X[i, j, :3] / X[i, j, 3]) for i in range(len(X)) for j in range(X.shape[1])]
    wts = [float(X[i, j, 3]) for i in range(len(X)) for j in range(X.shape[1])]
    rational = any(abs(w - 1.0) > 1e-12 for w in wts)
    return ext.surface_from_cvs(q, deg, pts, [float(u) for u in U], Uv, wts if rational else None)


def _check(sm, surf, x, rows, deg, gap_tol, tangent=True):
    """True when the surface holds section x (rails, middle; tangent planes
    along the rails unless a chamfer's)."""
    w = np.linspace(0.0, 1.0, 5)
    H = np.asarray(rows, float)
    for k, ww in enumerate(w):
        N = _bern(deg, ww)
        q = (N[:, None] * H).sum(axis=0)
        q = q[:3] / q[3]
        cp, uvs, dist = sm.surface_closest_point(surf, tuple(float(c) for c in q))
        if dist > gap_tol:
            return False
        if tangent and k in (0, len(w) - 1):
            u, v = _uv(surf, uvs[0])
            _, su, sv = surf.derivatives(u, v)
            nn = np.cross(su, sv)
            nn /= np.linalg.norm(nn) or 1.0
            ref = x.na if k == 0 else x.nb
            if math.acos(min(1.0, abs(float(nn @ ref)))) > ANGLE_TOL:
                return False
    return True


def _bern(n, t):
    return np.array([math.comb(n, i) * t ** i * (1 - t) ** (n - i) for i in range(n + 1)])


# ---------------------------------------------------------------------------
# The fillet
# ---------------------------------------------------------------------------


def _sides_on(sm, brep_a, brep_b, cr, ra, rb, chord, size, faces, soft=None):
    """(side_a, side_b) whose ball's rails lie on the given faces (sampled at
    three places along the crossing)."""
    best = None
    for sa in (-1.0, 1.0):
        for sb in (-1.0, 1.0):
            solver = Solver(Base(sm, brep_a, sa), Base(sm, brep_b, sb), ra, rb, chord, size, soft)
            worst = 0.0
            try:
                for s in (0.25, 0.5, 0.75):
                    x = solver.section(cr, s)
                    worst = max(worst, faces[0].closest_point(tuple(x.pa))[1], faces[1].closest_point(tuple(x.pb))[1])
            except Exception:
                continue
            if best is None or worst < best[0]:
                best = (worst, sa, sb)
    if best is None or best[0] > 1e-4 * size:
        raise RuntimeError("no fillet of this size fits between the two faces")
    return best[1], best[2]

def build(sm, brep_a, brep_b, side_a, side_b, ra, rb, chord=None, shape="CIRCULAR", tolerance=1e-4,
          curves=None, on_faces=None, smoothing=0.6):
    """(fillet brep, info) between two world-space breps. side_a / side_b: +1
    for the fillet on that surface's outward-normal side, -1 the other.
    curves: the crossing to follow (an edge fillet's edges) instead of the
    breps' intersection. on_faces = (face on A, face on B) with side_a /
    side_b None: the sides are the ones whose rails land on those faces
    (an edge fillet: the one corner between the two faces that both have).
    SOFT2 / SOFT3: ra is the soft fillet's minimum radius R (rb is not used),
    smoothing its share of the turn in the ramps."""
    ext = sm.blender_ext
    if shape not in SHAPES:
        raise ValueError(f"unknown section {shape}")
    soft = None
    if shape in ("SOFT2", "SOFT3"):
        import softfillet
        cont = 3 if shape == "SOFT3" else 2
        soft = (float(ra), softfillet.clamp_smoothing(smoothing, cont), cont)
        chord = None
        ra = rb = softfillet.ball_radius(ra, 0.5 * math.pi, soft[1], cont)     # a first guess, for the sides
    if curves is None:
        try:
            curves, _ = sm.intersect_breps(brep_a, brep_b)
        except RuntimeError:
            raise RuntimeError("the surfaces do not cross")
    bb = [brep_a.bounding_box(), brep_b.bounding_box()]
    size = max(float(np.linalg.norm(np.array(b[1], float) - np.array(b[0], float))) for b in bb) or 1.0
    crossings = [Crossing(ch) for ch in chains(list(curves or []), 1e-5 * size)]
    crossings = [c for c in crossings if c.length > 1e-9]
    if not crossings:
        raise RuntimeError("the surfaces do not cross")
    cr = max(crossings, key=lambda c: c.length)
    if side_a is None or side_b is None:
        side_a, side_b = _sides_on(sm, brep_a, brep_b, cr, ra, rb, chord, size, on_faces, soft)
    A, B = Base(sm, brep_a, side_a), Base(sm, brep_b, side_b)
    solver = Solver(A, B, ra, rb, chord, size, soft)
    polys = []
    for br in (brep_a, brep_b):
        for e in br.edges():
            t0, t1 = e.parameter_range()
            polys.append(np.array([e.point_at(t0 + (t1 - t0) * k / 48.0) for k in range(49)], float))
    cr.start_away(polys)
    n0 = 12 if cr.closed else 9
    secs = []
    for s in np.linspace(0.0, 1.0, n0, endpoint=not cr.closed):
        secs.append(solver.section(cr, float(s), secs[-1].ra if ((chord or soft) and secs) else None))
    gap_tol = max(min(tolerance, 1e-5 * max(cr.length, 1e-9)), 1e-9 * size)
    surf = None
    for _round in range(8):
        rows = [section_rows(x, shape, A, B, soft) for x in secs]
        deg = rows[0][0]
        surf = skin(ext, secs, [r for _, r in rows], deg, cr.closed)
        add = []
        n = len(secs)
        for i in range(n if cr.closed else n - 1):
            s0, s1 = secs[i].s, (secs[i + 1].s if i + 1 < n else 1.0)
            x = solver.section(cr, 0.5 * (s0 + s1), secs[i].ra if (chord or soft) else None)
            dg, xr = section_rows(x, shape, A, B, soft)
            if not _check(sm, surf, x, xr, dg, gap_tol, shape != "LINEAR"):
                add.append((i + 1, x))
        if not add or len(secs) + len(add) > MAX_SECTIONS:
            break
        for i, x in reversed(add):
            secs.insert(i, x)
    radii = [x.ra for x in secs] + [x.rb for x in secs]
    chords = [float(np.linalg.norm(x.pa - x.pb)) for x in secs]
    f = ext.brep_from_surface(surf)
    info = {"radius": (float(min(radii)), float(np.mean(radii)), float(max(radii))),
            "chord": (float(min(chords)), float(np.mean(chords)), float(max(chords))),
            "sections": len(secs)}
    if soft:
        import softfillet
        pk = [softfillet.peak_radius(np.asarray(r)[:, :3] / np.asarray(r)[:, 3:]) for _, r in rows]
        info["min_radius"] = (float(min(pk)), float(np.mean(pk)), float(max(pk)))
        info["smoothing"] = soft[1]
    if len(crossings) > 1:
        info["note"] = f"the surfaces cross {len(crossings)} times: filleted the longest crossing"
    return f, info
