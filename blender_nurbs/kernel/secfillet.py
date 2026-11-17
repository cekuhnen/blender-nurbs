"""Surface fillets built from exact cross-sections (the worker's chord and
blend fillets).

Per section -- a plane square to the surfaces' intersection curve -- a ball
touches both surfaces: its centre C sits at signed distance r_A from A and r_B
from B (Newton on the two distances; their gradients are the unit normals at
the feet). The feet are the rails. Chord: one radius per section, found so
that the rails are exactly the chord apart (the angle between the surfaces
changes along the crossing, so the radius does). Between the rails the
section is a circular arc (exact, rational), a straight chamfer, or a G1
cubic / G2 quintic blend: tangent to each surface square across the rail
and, G2, with its normal curvature in that direction.

The sections are skinned through, in homogeneous coordinates (rational arcs
stay exact), periodic round a closed crossing; a section is added halfway
wherever the surface misses that section's rails, middle or tangent planes
by more than the tolerance there.

Pure numpy + the kernel module (no bpy): imported by worker.py."""

import math

import numpy as np

SHAPES = ("CIRCULAR", "LINEAR", "G1", "G2")
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

    def face_of(self, q):
        if len(self.faces) == 1:
            return 0
        return min(range(len(self.faces)), key=lambda i: self.faces[i].closest_point(tuple(q))[1])

    def foot(self, q, fi=None):
        """(point, unit normal on the ball's side, (face, u, v)) nearest q."""
        fi = self.face_of(q) if fi is None else fi
        s = self.surfs[fi]
        _, uvs, _ = self.sm.surface_closest_point(s, tuple(float(x) for x in q))
        uv = uvs[0]
        u, v = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
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
    """An intersection curve, by arc length s in 0..1."""

    def __init__(self, curve, m=801):
        t0, t1 = curve.parameter_range()
        self.c = curve
        T = np.linspace(t0, t1, m)
        P = np.array([curve.evaluate(t) for t in T], float)
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        self.length = float(d[-1])
        self.s, self.t, self.h = d / (d[-1] or 1.0), T, 1e-4 * (t1 - t0)
        self.t0, self.t1 = t0, t1
        self.closed = float(np.linalg.norm(P[0] - P[-1])) <= 1e-6 * max(self.length, 1e-12)
        self.P = P
        self.off = 0.0          # closed: where s = 0 lies (see start_away)

    def start_away(self, polys):
        """A closed crossing restarted at its point farthest from the edges
        (polylines): the fillet's seam then lies away from the surfaces' own
        seams -- a rail cutter whose seam lies on a closed surface's seam
        does not cut it into regions."""
        if not self.closed or not polys:
            return
        ks = np.linspace(0.0, 1.0, 96, endpoint=False)
        Q = np.array([self.at(k)[0] for k in ks])
        E = np.vstack(polys)
        d = np.min(np.linalg.norm(Q[:, None, :] - E[None, :, :], axis=2), axis=1)
        self.off = float(ks[int(np.argmax(d))])

    def _p(self, t):
        if self.closed:
            t = self.t0 + (t - self.t0) % (self.t1 - self.t0)
        else:
            t = min(max(t, self.t0), self.t1)
        return np.array(self.c.evaluate(t), float)

    def at(self, s):
        if self.closed:
            s = (s + self.off) % 1.0
        t = float(np.interp(s, self.s, self.t))
        p = np.array(self.c.evaluate(t), float)
        T = self._p(t + self.h) - self._p(t - self.h)
        return p, T / (np.linalg.norm(T) or 1.0)


# ---------------------------------------------------------------------------
# One section
# ---------------------------------------------------------------------------

class Section:
    __slots__ = ("s", "P0", "C", "pa", "pb", "na", "nb", "wa", "wb", "ra", "rb")


def _plane(T):
    a = np.array([1.0, 0.0, 0.0]) if abs(T[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(T, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(T, e1)


class Solver:
    def __init__(self, A, B, ra, rb, chord=None, scale=1.0):
        self.A, self.B = A, B
        self.ra, self.rb, self.chord = ra, rb, chord
        self.scale = scale

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
        if abs(na @ (C - pa) - ra) > 1e-6 * max(ra, 1e-9) + 1e-9 * self.scale or \
                abs(nb @ (C - pb) - rb) > 1e-6 * max(rb, 1e-9) + 1e-9 * self.scale:
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
        else:
            ra, rb = self.ra, self.rb
            C = guess if guess is not None else self.start(P0, ra, rb)[0]
            C, pa, na, wa, pb, nb, wb = self.ball(P0, T, ra, rb, C)
        x = Section()
        x.s, x.P0, x.C, x.pa, x.pb, x.na, x.nb, x.wa, x.wb, x.ra, x.rb = s, P0, C, pa, pb, na, nb, wa, wb, ra, rb
        return x


# ---------------------------------------------------------------------------
# Section curves (homogeneous rows) and the skin
# ---------------------------------------------------------------------------

def _across(x, base, p, n, other):
    """Unit direction tangent to the surface at p (normal n), toward the other rail."""
    d = other - p
    d = d - (d @ n) * n
    return d / (np.linalg.norm(d) or 1.0)


def section_rows(x, shape, A, B):
    """(degree, rows (deg+1, 4) homogeneous) of the section's curve, rail A first."""
    pa, pb = x.pa, x.pb
    h1 = lambda p, w=1.0: np.append(np.asarray(p, float) * w, w)
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
            uv = uvs[0]
            u, v = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
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

def build(sm, brep_a, brep_b, side_a, side_b, ra, rb, chord=None, shape="CIRCULAR", tolerance=1e-4):
    """(fillet brep, info) between two world-space breps. side_a / side_b: +1
    for the fillet on that surface's outward-normal side, -1 the other."""
    ext = sm.blender_ext
    if shape not in SHAPES:
        raise ValueError(f"unknown section {shape}")
    try:
        curves, _ = sm.intersect_breps(brep_a, brep_b)
    except RuntimeError:
        raise RuntimeError("the surfaces do not cross")
    crossings = [Crossing(c) for c in curves or []]
    crossings = [c for c in crossings if c.length > 1e-9]
    if not crossings:
        raise RuntimeError("the surfaces do not cross")
    A, B = Base(sm, brep_a, side_a), Base(sm, brep_b, side_b)
    bb = [brep_a.bounding_box(), brep_b.bounding_box()]
    size = max(float(np.linalg.norm(np.array(b[1], float) - np.array(b[0], float))) for b in bb) or 1.0
    solver = Solver(A, B, ra, rb, chord, size)
    cr = max(crossings, key=lambda c: c.length)
    polys = []
    for br in (brep_a, brep_b):
        for e in br.edges():
            t0, t1 = e.parameter_range()
            polys.append(np.array([e.point_at(t0 + (t1 - t0) * k / 48.0) for k in range(49)], float))
    cr.start_away(polys)
    n0 = 12 if cr.closed else 9
    secs = []
    for s in np.linspace(0.0, 1.0, n0, endpoint=not cr.closed):
        secs.append(solver.section(cr, float(s), secs[-1].ra if (chord and secs) else None))
    gap_tol = max(min(tolerance, 1e-5 * max(cr.length, 1e-9)), 1e-9 * size)
    surf = None
    for _round in range(8):
        rows = [section_rows(x, shape, A, B) for x in secs]
        deg = rows[0][0]
        surf = skin(ext, secs, [r for _, r in rows], deg, cr.closed)
        add = []
        n = len(secs)
        for i in range(n if cr.closed else n - 1):
            s0, s1 = secs[i].s, (secs[i + 1].s if i + 1 < n else 1.0)
            x = solver.section(cr, 0.5 * (s0 + s1), secs[i].ra if chord else None)
            dg, xr = section_rows(x, shape, A, B)
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
    if len(crossings) > 1:
        info["note"] = f"the surfaces cross {len(crossings)} times: filleted the longest crossing"
    return f, info
