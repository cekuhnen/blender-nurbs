"""STEP (ISO 10303-21, AP214 "automotive design") export of NURBS objects.

Exact geometry: every face's surface is written as a B-spline surface (rational
when it has weights), every edge as a B-spline curve cut exactly to the edge's
range -- or, with analytic on (the default), as what it truly is when it is one
within 1e-7 of the body's size: a PLANE, CYLINDRICAL_, CONICAL_, SPHERICAL_
or TOROIDAL_SURFACE, a CIRCLE, ELLIPSE or LINE, as CAD programs write them (an exact cylinder is then
a true cylinder in the file, its holes and bosses recognised by the reader) -- with full B-rep topology (vertices, edge curves shared by their two
faces, oriented edges, outer / inner face bounds, face sense). Closed bodies
become MANIFOLD_SOLID_BREPs, open ones SHELL_BASED_SURFACE_MODELs. Each object
is one body, in world space, written in the chosen length unit.
"""

import datetime
import os

import numpy as np

from . import geom, store

UNITS = {"MM": ("MILLI", 1000.0), "CM": ("CENTI", 100.0), "M": (None, 1.0)}


def _vpt(v):
    q = v.point()
    return (q.x, q.y, q.z) if hasattr(q, "x") else tuple(q)


def _f(x):
    """STEP real: always with a decimal point."""
    s = f"{float(x):.15G}"
    if "E" in s:
        m, e = s.split("E")
        if "." not in m:
            m += "."
        return f"{m}E{e}"
    if "." not in s:
        s += "."
    return s


def _str(s):
    return "'" + str(s).replace("'", "''").replace("\\", "\\\\") + "'"


def _knots(U):
    """Distinct knots + multiplicities."""
    ks, ms = [], []
    for k in U:
        if ks and abs(k - ks[-1]) <= 1e-12 * (1 + abs(k)):
            ms[-1] += 1
        else:
            ks.append(float(k))
            ms.append(1)
    return ks, ms


# --------------------------------------------------------------------------
# Analytic recognition: is this B-spline a line, circle, plane, cylinder, cone,
# sphere or torus?
# Measured on samples, within tol (absolute, export units); None if not.
# --------------------------------------------------------------------------

def fit_line(S, tol):
    """(start, unit direction) of points on a straight line, else None."""
    S = np.asarray(S, float)
    d = S[-1] - S[0]
    L = float(np.linalg.norm(d))
    if L <= tol:
        return None
    d /= L
    r = S - S[0]
    off = r - np.outer(r @ d, d)
    if float(np.linalg.norm(off, axis=1).max()) > tol:
        return None
    return S[0], d


def fit_plane(S, tol):
    """(centroid, unit normal) of points on a plane, else None."""
    S = np.asarray(S, float).reshape(-1, 3)
    c = S.mean(axis=0)
    _, sv, Vt = np.linalg.svd(S - c, full_matrices=False)
    if sv[1] <= tol:                      # all on a line (or a point): no plane
        return None
    n = Vt[2]
    if float(np.abs((S - c) @ n).max()) > tol:
        return None
    return c, n


def _circle_2d(Q):
    """Kasa fit of 2D points: (centre, radius)."""
    m = Q.mean(axis=0)
    q = Q - m
    A = np.column_stack([2 * q, np.ones(len(q))])
    sol, *_ = np.linalg.lstsq(A, (q * q).sum(axis=1), rcond=None)
    c = sol[:2]
    return c + m, float(np.sqrt(max(sol[2] + c @ c, 0.0)))


def fit_circle(S, tol, size):
    """(centre, unit normal, radius) of points on a circle or arc, the normal
    so that the points run counter-clockwise round it; else None. Straight
    runs are not circles (check fit_line first)."""
    S = np.asarray(S, float)
    pl = fit_plane(S, tol)
    if pl is None:
        return None
    c, n = pl
    x = S[0] - c
    x -= (x @ n) * n
    if np.linalg.norm(x) < 1e-12:
        x = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    Q = np.column_stack([(S - c) @ x, (S - c) @ y])
    c2, r = _circle_2d(Q)
    if not (tol < r < 1e3 * size):
        return None
    if float(np.abs(np.linalg.norm(Q - c2, axis=1) - r).max()) > tol:
        return None
    C = c + c2[0] * x + c2[1] * y
    turn = sum(np.cross(S[i] - C, S[i + 1] - C) for i in range(len(S) - 1))
    if float(turn @ n) < 0:
        n = -n
    return C, n, r


def fit_ellipse(S, tol, size):
    """(centre, unit normal, major axis direction, major radius, minor radius)
    of points on an ellipse or elliptic arc, the normal so that the points run
    counter-clockwise round it; else None. Lines and circles are not ellipses
    (check fit_line and fit_circle first)."""
    S = np.asarray(S, float)
    if len(S) < 6:
        return None
    pl = fit_plane(S, tol)
    if pl is None:
        return None
    c, n = pl
    x = S[0] - c
    x -= (x @ n) * n
    if np.linalg.norm(x) < 1e-12:
        x = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    Q = np.column_stack([(S - c) @ x, (S - c) @ y])
    s = float(np.abs(Q).max())
    if s <= tol:
        return None
    q = Q / s
    # the conic A x² + B xy + C y² + D x + E y + F = 0 through the points
    Dm = np.column_stack([q[:, 0] ** 2, q[:, 0] * q[:, 1], q[:, 1] ** 2, q[:, 0], q[:, 1], np.ones(len(q))])
    A, B, C, D, E, F = np.linalg.svd(Dm)[2][-1]
    if B * B - 4 * A * C >= 0:            # a parabola or hyperbola
        return None
    c2 = np.linalg.solve([[2 * A, B], [B, 2 * C]], [-D, -E])
    Fc = F + 0.5 * (D * c2[0] + E * c2[1])
    lam, vec = np.linalg.eigh([[A, B / 2], [B / 2, C]])
    ax2 = -Fc / lam
    if not (ax2 > 0).all():
        return None
    k = int(np.argmax(ax2))               # the major axis first
    r = np.sqrt(ax2[[k, 1 - k]]) * s
    if not (tol < r[1] and r[0] < 1e3 * size):
        return None
    u = vec[:, k]
    C2 = c2 * s
    # every point on it: the point at its own eccentric angle is no nearer than the foot
    d = Q - C2
    e1, e2 = d @ u, d @ np.array([-u[1], u[0]])
    t = np.arctan2(e2 / r[1], e1 / r[0])
    off = np.hypot(e1 - r[0] * np.cos(t), e2 - r[1] * np.sin(t))
    if float(off.max()) > tol:
        return None
    Cw = c + C2[0] * x + C2[1] * y
    U = u[0] * x + u[1] * y
    turn = sum(np.cross(S[i] - Cw, S[i + 1] - Cw) for i in range(len(S) - 1))
    if float(turn @ n) < 0:
        n = -n
    return Cw, n, U, float(r[0]), float(r[1])


def fit_cylinder(G, tol, size):
    """(point on the axis, unit axis, radius) of a grid of surface points
    G (mu, mv, 3) that lie on a circular cylinder (its rulings along u or v),
    else None. Planes are not cylinders (check fit_plane first)."""
    G = np.asarray(G, float)
    for rows in (G, G.transpose(1, 0, 2)):          # rulings along v, then along u
        a = None
        ok = True
        for R in rows:
            ln = fit_line(R, tol)
            if ln is None:
                ok = False
                break
            d = ln[1]
            if a is None:
                a = d
            elif np.linalg.norm(np.cross(a, d)) * float(np.ptp(R @ d)) > tol:
                ok = False
                break
        if not ok or a is None:
            continue
        P = G.reshape(-1, 3)
        x = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(a, x)
        Q = np.column_stack([P @ x, P @ y])
        c2, r = _circle_2d(Q)
        if not (tol < r < 1e3 * size):
            continue
        if float(np.abs(np.linalg.norm(Q - c2, axis=1) - r).max()) > tol:
            continue
        return c2[0] * x + c2[1] * y + float(P[0] @ a) * a, a, r
    return None


def fit_cone(G, tol, size):
    """(apex, unit axis pointing from the apex into the face, semi-angle) of a
    grid of surface points G (mu, mv, 3) on one nappe of a circular cone, its
    rulings along u or v, else None. Planes and cylinders are not cones (check
    fit_plane and fit_cylinder first)."""
    G = np.asarray(G, float)
    P = G.reshape(-1, 3)
    for rows in (G, G.transpose(1, 0, 2)):          # rulings along v, then along u
        lines = [fit_line(R, tol) for R in rows]
        if any(ln is None for ln in lines):
            continue
        D = np.array([d for _, d in lines])          # all in parameter order, so all one way
        if float(np.linalg.norm(np.cross(D, D[0]), axis=1).max()) < 1e-9:
            continue                                  # parallel rulings: a cylinder
        # the apex: nearest to all the rulings
        M = np.zeros((3, 3))
        b = np.zeros(3)
        for (o, d) in lines:
            Q = np.eye(3) - np.outer(d, d)
            M += Q
            b += Q @ o
        if np.linalg.svd(M, compute_uv=False)[-1] < 1e-9 * len(lines):
            continue
        apex = np.linalg.solve(M, b)
        if float(np.linalg.norm(apex - P.mean(axis=0))) > 1e3 * size:
            continue
        # the axis: the ruling directions' tips lie on a circle round it
        c = D.mean(axis=0)
        _, sv, Vt = np.linalg.svd(D - c, full_matrices=False)
        if sv[1] < 1e-9:                              # too few distinct rulings to tell
            continue
        a = Vt[2]
        r = P - apex
        if float((r @ a).mean()) < 0:
            a = -a
        h = r @ a
        rho = np.linalg.norm(r - np.outer(h, a), axis=1)
        if float(h.min()) < -tol:                     # both nappes
            continue
        far = rho + h > 1e-3 * (float(np.ptp(h)) + float(rho.max()))
        al = float(np.arctan2(rho[far], h[far]).mean())
        if not (1e-6 < al < 0.5 * np.pi - 1e-6):
            continue
        if float(np.abs(rho * np.cos(al) - h * np.sin(al)).max()) > tol:
            continue
        return apex, a, al
    return None


def revolution_axis(G, tol, size):
    """(point, unit axis) of a grid of surface points G (mu, mv, 3) whose rows
    along u or v are circles round one axis (the parallels of a surface of
    revolution), else None. Rows shrunk to a point (a pole) are skipped."""
    G = np.asarray(G, float)
    for rows in (G.transpose(1, 0, 2), G):          # parallels along u, then along v
        circles = []
        ok = True
        for R in rows:
            if float(np.linalg.norm(R - R[0], axis=1).max()) <= tol:
                continue                              # a pole
            ci = fit_circle(R, tol, size)
            if ci is None:
                ok = False
                break
            circles.append(ci)
        if not ok or len(circles) < 2:
            continue
        C0, a, _ = circles[0]
        for C, n, r in circles[1:]:
            d = C - C0
            if (np.linalg.norm(np.cross(n, a)) * r > tol or
                    np.linalg.norm(d - (d @ a) * a) > tol):
                ok = False
                break
        if ok:
            return C0, a
    return None


def fit_sphere(G, tol, size):
    """(centre, radius) of surface points on a sphere, else None. Planes are
    not spheres (check fit_plane first)."""
    P = np.asarray(G, float).reshape(-1, 3)
    m = P.mean(axis=0)
    q = P - m
    A = np.column_stack([2 * q, np.ones(len(q))])
    sol, *_ = np.linalg.lstsq(A, (q * q).sum(axis=1), rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + c @ c, 0.0)))
    if not (tol < r < 1e3 * size):
        return None
    if float(np.abs(np.linalg.norm(q - c, axis=1) - r).max()) > tol:
        return None
    return c + m, r


def fit_torus(G, tol, size):
    """(centre, unit axis, major radius, minor radius) of a grid of surface
    points G (mu, mv, 3) on a ring torus (major radius larger than the
    minor), its parallels along u or v, else None. Planes, cylinders, cones
    and spheres are surfaces of revolution too: check them first."""
    ax = revolution_axis(G, tol, size)
    if ax is None:
        return None
    p, a = ax
    r = np.asarray(G, float).reshape(-1, 3) - p
    z = r @ a
    rho = np.linalg.norm(r - np.outer(z, a), axis=1)
    (R, z0), rm = _circle_2d(np.column_stack([rho, z]))
    if not (tol < rm < 1e3 * size and R > rm + tol):
        return None
    if float(np.abs(np.hypot(rho - R, z - z0) - rm).max()) > tol:
        return None
    return p + z0 * a, a, float(R), rm


def _regular(su, sv, rel):
    """Do su, sv span a tangent plane (not a pole: one of them ~0, or parallel)?"""
    m = max(float(np.linalg.norm(su)), float(np.linalg.norm(sv)))
    return m > 0 and float(np.linalg.norm(np.cross(su, sv))) > rel * m * m


def _off_degenerate(surf, uv, rel=1e-6):
    """uv, or (where su x sv vanishes there, e.g. a sphere's pole) a point
    stepped from it towards the middle of the surface's domain, where the
    normal is defined; its direction is the pole's by continuity."""
    u, v = float(uv[0]), float(uv[1])
    _, su, sv = surf.derivatives(u, v)
    su, sv = np.asarray(su, float), np.asarray(sv, float)
    if _regular(su, sv, rel):
        return u, v
    try:
        (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    except Exception:
        from .surfeval import Surf
        E = Surf(surf)
        (u0, u1), (v0, v1) = E.u_domain, E.v_domain
    for s in (1e-4, 1e-3, 1e-2):
        uu, vv = u + s * (0.5 * (u0 + u1) - u), v + s * (0.5 * (v0 + v1) - v)
        _, a, b = surf.derivatives(uu, vv)
        a, b = np.asarray(a, float), np.asarray(b, float)
        if _regular(a, b, rel):
            return uu, vv
    return u, v


def _perp(n, hint=None):
    """A unit vector square to n (hint's part square to n, when it has one)."""
    if hint is not None:
        h = np.asarray(hint, float) - (np.asarray(hint, float) @ n) * n
        if np.linalg.norm(h) > 1e-9:
            return h / np.linalg.norm(h)
    h = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    return h / np.linalg.norm(h)


class _Writer:
    def __init__(self):
        self.lines = []
        self.n = 0

    def add(self, text):
        self.n += 1
        self.lines.append(f"#{self.n}={text};")
        return f"#{self.n}"


class StepExporter:
    REL_TOL = 1e-7          # analytic when within this much of the body's size

    def __init__(self, scale=1.0, unit="MM", analytic=True):
        self.w = _Writer()
        self.unit_prefix, unit_factor = UNITS[unit]
        self.scale = scale * unit_factor
        self.analytic = analytic
        self.stats = {"bodies": 0, "faces": 0, "edges": 0, "solids": 0, "sheets": 0,
                      "analytic": {"CYLINDRICAL_SURFACE": 0, "CONICAL_SURFACE": 0, "SPHERICAL_SURFACE": 0,
                                   "TOROIDAL_SURFACE": 0, "PLANE": 0, "CIRCLE": 0, "ELLIPSE": 0,
                                   "LINE": 0}}
        self._dir_cache = {}

    # -- small geometry ----------------------------------------------------
    def point(self, p):
        return self.w.add(f"CARTESIAN_POINT('',({','.join(_f(c) for c in p)}))")

    def direction(self, d):
        key = tuple(round(float(c), 12) for c in d)
        if key not in self._dir_cache:
            self._dir_cache[key] = self.w.add(f"DIRECTION('',({','.join(_f(c) for c in d)}))")
        return self._dir_cache[key]

    def _pts(self, P):
        return "(" + ",".join(self.point(p) for p in P) + ")"

    def placement(self, o, z, x):
        return self.w.add(f"AXIS2_PLACEMENT_3D('',{self.point(o)},{self.direction(z)},{self.direction(x)})")

    # -- analytic ----------------------------------------------------------
    def analytic_curve(self, S, tol, size):
        """LINE, CIRCLE or ELLIPSE through the edge's samples S (in its direction), else None."""
        ln = fit_line(S, tol)
        if ln is not None:
            o, d = ln
            vec = self.w.add(f"VECTOR('',{self.direction(d)},1.)")
            self.stats["analytic"]["LINE"] += 1
            return self.w.add(f"LINE('',{self.point(o)},{vec})")
        ci = fit_circle(S, tol, size)
        if ci is not None:
            C, n, r = ci
            self.stats["analytic"]["CIRCLE"] += 1
            return self.w.add(f"CIRCLE('',{self.placement(C, n, _perp(n, S[0] - C))},{_f(r)})")
        el = fit_ellipse(S, tol, size)
        if el is not None:
            C, n, U, ra, rb = el
            self.stats["analytic"]["ELLIPSE"] += 1
            return self.w.add(f"ELLIPSE('',{self.placement(C, n, _perp(n, U))},{_f(ra)},{_f(rb)})")
        return None

    def analytic_surface(self, G, tol, size):
        """(entity, normal at a point: P -> unit normal, entity type) of a PLANE,
        CYLINDRICAL_, CONICAL_, SPHERICAL_ or TOROIDAL_SURFACE through the
        surface samples G, else None."""
        pl = fit_plane(G, tol)
        if pl is not None:
            c, n = pl
            self.stats["analytic"]["PLANE"] += 1
            G2 = np.asarray(G, float)
            ent = self.w.add(f"PLANE('',{self.placement(c, n, _perp(n, G2[-1, 0] - G2[0, 0]))})")
            return ent, (lambda P: n), "PLANE"
        cy = fit_cylinder(G, tol, size)
        if cy is not None:
            o, a, r = cy
            self.stats["analytic"]["CYLINDRICAL_SURFACE"] += 1
            ent = self.w.add(f"CYLINDRICAL_SURFACE('',{self.placement(o, a, _perp(a, G[0][0] - o))},{_f(r)})")

            def normal(P):
                d = np.asarray(P, float) - o
                d -= (d @ a) * a
                return d / (np.linalg.norm(d) or 1.0)
            return ent, normal, "CYLINDRICAL_SURFACE"
        co = fit_cone(G, tol, size)
        if co is not None:
            apex, a, al = co
            G2 = np.asarray(G, float).reshape(-1, 3)
            hm = float(((G2 - apex) @ a).mean())           # placed at the face's mean height
            o = apex + hm * a
            self.stats["analytic"]["CONICAL_SURFACE"] += 1
            ent = self.w.add(f"CONICAL_SURFACE('',{self.placement(o, a, _perp(a, G2[0] - o))},"
                             f"{_f(hm * np.tan(al))},{_f(al)})")

            def normal(P):           # STEP's: away from the axis, tilted back towards the apex
                d = np.asarray(P, float) - apex
                e = d - (d @ a) * a
                e /= (np.linalg.norm(e) or 1.0)
                return e * np.cos(al) - a * np.sin(al)
            return ent, normal, "CONICAL_SURFACE"
        G2 = np.asarray(G, float).reshape(-1, 3)
        sp = fit_sphere(G, tol, size)
        if sp is not None:
            c, r = sp
            ax = revolution_axis(G, tol, size)            # its poles on the surface's own axis, when it has one
            z = ax[1] if ax is not None else np.array([0.0, 0.0, 1.0])
            self.stats["analytic"]["SPHERICAL_SURFACE"] += 1
            ent = self.w.add(f"SPHERICAL_SURFACE('',{self.placement(c, z, _perp(z, G2[0] - c))},{_f(r)})")
            return ent, (lambda P: (np.asarray(P, float) - c) / r), "SPHERICAL_SURFACE"
        to = fit_torus(G, tol, size)
        if to is not None:
            c, a, R, rm = to
            self.stats["analytic"]["TOROIDAL_SURFACE"] += 1
            ent = self.w.add(f"TOROIDAL_SURFACE('',{self.placement(c, a, _perp(a, G2[0] - c))},{_f(R)},{_f(rm)})")

            def normal(P):           # STEP's: away from the tube's centre circle
                d = np.asarray(P, float) - c
                e = d - (d @ a) * a
                e = c + R * e / (np.linalg.norm(e) or 1.0)
                n = np.asarray(P, float) - e
                return n / (np.linalg.norm(n) or 1.0)
            return ent, normal, "TOROIDAL_SURFACE"
        return None

    # -- curves / surfaces -------------------------------------------------
    def bspline_curve(self, deg, P, U, W=None):
        ks, ms = _knots(U)
        pts = self._pts(P)
        mult = "(" + ",".join(str(m) for m in ms) + ")"
        kn = "(" + ",".join(_f(k) for k in ks) + ")"
        if W is None or max(abs(w - 1.0) for w in W) < 1e-12:
            return self.w.add(f"B_SPLINE_CURVE_WITH_KNOTS('',{deg},{pts},.UNSPECIFIED.,.F.,.F.,{mult},{kn},.UNSPECIFIED.)")
        ws = "(" + ",".join(_f(w) for w in W) + ")"
        return self.w.add(f"(BOUNDED_CURVE() B_SPLINE_CURVE({deg},{pts},.UNSPECIFIED.,.F.,.F.) "
                          f"B_SPLINE_CURVE_WITH_KNOTS({mult},{kn},.UNSPECIFIED.) CURVE() "
                          f"GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE({ws}) REPRESENTATION_ITEM(''))")

    def bspline_surface(self, du, dv, P, nu, nv, Uu, Uv, W=None):
        rows = []
        for i in range(nu):
            rows.append("(" + ",".join(self.point(P[i * nv + j]) for j in range(nv)) + ")")
        grid = "(" + ",".join(rows) + ")"
        ku, mu = _knots(Uu)
        kv, mv = _knots(Uv)
        mults = f"({','.join(map(str, mu))}),({','.join(map(str, mv))})"
        knots = f"({','.join(_f(k) for k in ku)}),({','.join(_f(k) for k in kv)})"
        if W is None or max(abs(w - 1.0) for w in W) < 1e-12:
            return self.w.add(f"B_SPLINE_SURFACE_WITH_KNOTS('',{du},{dv},{grid},.UNSPECIFIED.,.F.,.F.,.F.,"
                              f"{mults},{knots},.UNSPECIFIED.)")
        wrows = []
        for i in range(nu):
            wrows.append("(" + ",".join(_f(W[i * nv + j]) for j in range(nv)) + ")")
        ws = "(" + ",".join(wrows) + ")"
        return self.w.add(f"(BOUNDED_SURFACE() B_SPLINE_SURFACE({du},{dv},{grid},.UNSPECIFIED.,.F.,.F.,.F.) "
                          f"B_SPLINE_SURFACE_WITH_KNOTS({mults},{knots},.UNSPECIFIED.) "
                          f"GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE({ws}) "
                          f"REPRESENTATION_ITEM('') SURFACE())")

    # -- one body ----------------------------------------------------------
    def body(self, obj):
        """Write one NURBS object; returns the representation item (solid / shell model)."""
        from .kernel import load
        from .square import _edge_curve
        ext = load().blender_ext
        brep = store.get_brep(obj)
        M = np.array(geom.world_matrix(obj), dtype=float)
        A, t = M[:3, :3] * self.scale, M[:3, 3] * self.scale
        X = lambda p: A @ np.asarray(p, float) + t
        size = float(np.linalg.norm(np.ptp(np.array(brep.bounding_box(), float), axis=0))) or 1.0
        xsize = size * float(np.linalg.norm(A, 2))           # the body's size in the file's units
        tol = self.REL_TOL * xsize

        verts, edges = {}, {}
        # topology, not positions: a closed edge (a trim circle) whose curve
        # closes to only ~1e-6 must still start and end at ONE vertex, or its
        # loop is open and readers drop the face (Plasticity lost a fillet)
        all_v = list(brep.vertices())
        all_e = list(brep.edges())

        def index_of(items, x):
            for i, y in enumerate(items):
                if y == x:
                    return i
            return None

        def vkey(p):
            return ("p",) + tuple(np.round(np.asarray(p, float) / size, 9))

        def vertex(kv, p):
            i = index_of(all_v, kv) if kv is not None else None
            k = ("v", i) if i is not None else vkey(p)
            if k not in verts:
                q = p
                if i is not None:
                    try:
                        q = all_v[i].point()
                        q = (q.x, q.y, q.z) if hasattr(q, "x") else tuple(q)
                    except Exception:
                        q = p
                verts[k] = self.w.add(f"VERTEX_POINT('',{self.point(X(q))})")
            return verts[k]

        def edge_key(e):
            i = index_of(all_e, e)
            if i is not None:
                return ("e", i)
            t0, t1 = e.parameter_range()
            return (vkey(e.point_at(t0)), vkey(e.point_at(t1)), vkey(e.point_at(0.5 * (t0 + t1))))

        def end_vertices(e):
            try:
                return e.start_vertex(), e.end_vertex()
            except Exception:
                return None, None

        def edge_curve(e):
            k = edge_key(e)
            if k in edges:
                return edges[k]
            t0, t1 = e.parameter_range()
            cur = None
            if self.analytic:
                S = np.array([X(e.point_at(t)) for t in np.linspace(t0, t1, 49)])
                cur = self.analytic_curve(S, tol, xsize)
            if cur is None:
                c = _edge_curve(e)
                d = ext.curve_cvs(c)
                P = [X(p) for p in d["points"]]
                W = list(d["weights"]) if d.get("weights") else None
                cur = self.bspline_curve(d["degree"], P, list(d["knots"]), W)
            sv, ev = end_vertices(e)
            p0, p1 = e.point_at(t0), e.point_at(t1)
            if sv is not None and ev is not None:
                # the kernel's start vertex is where the curve starts (else swap)
                a = np.asarray(_vpt(sv), float)
                if np.linalg.norm(a - np.asarray(p0, float)) > np.linalg.norm(a - np.asarray(p1, float)) + 1e-12:
                    sv, ev = ev, sv
            v0, v1 = vertex(sv, p0), vertex(ev, p1)
            ec = self.w.add(f"EDGE_CURVE('',{v0},{v1},{cur},.T.)")
            edges[k] = ec
            self.stats["edges"] += 1
            return ec

        faces = []
        for f in brep.faces():
            surf = f.surface()
            d = ext.surface_cvs(surf)
            an = None
            if self.analytic:
                from .surfeval import Surf
                E = Surf(surf)
                nu, nv = (min(4 * (n - 1) + 1, 61) for n in (E.nu, E.nv))
                us, vs = np.linspace(*E.u_domain, max(nu, 9)), np.linspace(*E.v_domain, max(nv, 9))
                G = E.grid(us, vs, 0)[0, 0] @ A.T + t
                an = self.analytic_surface(G, tol, xsize)
            an_kind = None
            if an is not None:
                srf, an_normal, an_kind = an
            else:
                P = [X(p) for p in d["points"]]
                W = list(d["weights"]) if d.get("weights") else None
                srf = self.bspline_surface(d["degree_u"], d["degree_v"], P, d["count_u"], d["count_v"],
                                           list(d["knots_u"]), list(d["knots_v"]), W)
            # face sense: the face's outward normal vs the surface's own normal,
            # probed at a point inside the (trimmed) face
            from .square import _foot
            sense = ".T."
            uv = None
            try:
                uv = _foot(surf, np.array(f.interior_point(), float))
                n_out = np.array(f.outward_normal_at_uv(*uv), float)
                uv = _off_degenerate(surf, uv)        # a pole has no su x sv: probe beside it
                _, su, sv = surf.derivatives(*uv)
                sense = ".T." if float(np.cross(su, sv) @ n_out) >= 0 else ".F."
            except Exception:
                pass
            # the kernel's loops run round the B-spline's own normal; STEP's run round the face's normal
            # (the surface's, reversed when the sense is .F.). An analytic surface has its own normal, so
            # where the B-spline's pointed the other way from the face its bounds are written reversed
            bound_sense = ".F." if an is not None and sense == ".F." else ".T."
            if an is not None:
                # the analytic surface's own normal against the B-spline's (as written): flip the sense if opposed
                try:
                    if uv is None:
                        uv = (0.5 * sum(E.u_domain), 0.5 * sum(E.v_domain))
                    D = E.at([uv[0]], [uv[1]], 1)
                    Nb = np.cross(A @ D[1, 0][0], A @ D[0, 1][0])
                    if float(Nb @ an_normal(X(D[0, 0][0]))) < 0:
                        sense = ".F." if sense == ".T." else ".T."
                except Exception:
                    pass
            outer = f.boundary_outer_loop()
            outer_ids = {edge_key(e) for e, _ in outer}
            bounds = []
            loops = list(f.boundary_loops())
            if an_kind == "SPHERICAL_SURFACE":
                # a whole sphere: bounded only by its seam (run there and back) and collapsed pole edges.
                # Written as CAD programs do, with no seam: one VERTEX_LOOP at a pole
                used = {}
                for loop in loops:
                    for e, _ in loop:
                        if e.length() > 1e-9 * size:
                            used.setdefault(edge_key(e), []).append(e)
                if used and all(len(v) == 2 for v in used.values()):
                    e = next(iter(used.values()))[0]
                    t0, _ = e.parameter_range()
                    vl = self.w.add(f"VERTEX_LOOP('',{vertex(end_vertices(e)[0], e.point_at(t0))})")
                    bounds.append(self.w.add(f"FACE_BOUND('',{vl},.T.)"))
                    loops = []
            for loop in loops:
                oes = []
                for e, fwd in loop:
                    if e.length() <= 1e-9 * size:       # degenerate (collapsed) edge: readers re-create it
                        continue
                    oes.append(self.w.add(f"ORIENTED_EDGE('',*,*,{edge_curve(e)},{'.T.' if fwd else '.F.'})"))
                if not oes:
                    continue
                lp = self.w.add(f"EDGE_LOOP('',({','.join(oes)}))")
                is_outer = {edge_key(e) for e, _ in loop} == outer_ids
                bounds.append(self.w.add(f"{'FACE_OUTER_BOUND' if is_outer else 'FACE_BOUND'}('',{lp},{bound_sense})"))
            faces.append(self.w.add(f"ADVANCED_FACE('',({','.join(bounds)}),{srf},{sense})"))
            self.stats["faces"] += 1
        self.stats["bodies"] += 1
        name = _str(obj.name)
        try:
            solid = bool(brep.is_manifold_solid())
        except Exception:
            solid = False
        if solid:
            shell = self.w.add(f"CLOSED_SHELL('',({','.join(faces)}))")
            self.stats["solids"] += 1
            return "solid", self.w.add(f"MANIFOLD_SOLID_BREP({name},{shell})")
        shell = self.w.add(f"OPEN_SHELL('',({','.join(faces)}))")
        self.stats["sheets"] += 1
        return "sheet", self.w.add(f"SHELL_BASED_SURFACE_MODEL({name},({shell}))")

    # -- file --------------------------------------------------------------
    def write(self, path, objects, product_name=None):
        w = self.w
        name = product_name or os.path.splitext(os.path.basename(path))[0]
        app = w.add("APPLICATION_CONTEXT('automotive design')")
        w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{app})")
        pctx = w.add(f"PRODUCT_CONTEXT('',{app},'mechanical')")
        prod = w.add(f"PRODUCT({_str(name)},{_str(name)},'',({pctx}))")
        w.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,({prod}))")
        pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',{prod})")
        pdc = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{app},'design')")
        pd = w.add(f"PRODUCT_DEFINITION('design','',{pdf},{pdc})")
        pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',{pd})")
        prefix = f".{self.unit_prefix}." if self.unit_prefix else "$"
        lu = w.add(f"(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT({prefix},.METRE.))")
        au = w.add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))")
        su = w.add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())")
        unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),{lu},'distance_accuracy_value',"
                    "'confusion accuracy')")
        ctx = w.add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({unc})) "
                    f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({lu},{au},{su})) REPRESENTATION_CONTEXT('Context #1',"
                    "'3D Context with UNIT and UNCERTAINTY'))")
        ax = w.add(f"AXIS2_PLACEMENT_3D('',{self.point((0, 0, 0))},{self.direction((0, 0, 1))},"
                   f"{self.direction((1, 0, 0))})")
        bodies = []
        failed = []
        for obj in objects:
            try:
                bodies.append((obj, *self.body(obj)))
            except Exception as exc:
                failed.append((obj.name, str(exc)))
        main = w.add(f"SHAPE_REPRESENTATION({_str(name)},({ax}),{ctx})")
        w.add(f"SHAPE_DEFINITION_REPRESENTATION({pds},{main})")
        for obj, kind, item in bodies:
            if kind == "solid":
                rep = w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION({_str(obj.name)},({item},{ax}),{ctx})")
            else:
                rep = w.add(f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION({_str(obj.name)},({item},{ax}),{ctx})")
            w.add(f"SHAPE_REPRESENTATION_RELATIONSHIP('','',{main},{rep})")
        stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        header = ["ISO-10303-21;", "HEADER;",
                  "FILE_DESCRIPTION(('NURBS Prototype export'),'2;1');",
                  f"FILE_NAME({_str(os.path.basename(path))},'{stamp}',(''),(''),'NURBS Prototype (Blender)',"
                  "'USD BRep kernel','');",
                  "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(header + w.lines + ["ENDSEC;", "END-ISO-10303-21;", ""]))
        self.stats["failed"] = failed
        return self.stats


def export_step(path, objects, unit="MM", scale=1.0, analytic=True):
    ex = StepExporter(scale=scale, unit=unit, analytic=analytic)
    stats = ex.write(path, [o for o in objects if store.is_nurbs(o)])
    try:
        with open(path, encoding="utf-8") as fh:
            stats["open_loops"] = loop_problems(fh.read())
    except Exception:
        pass
    return stats


def loop_problems(text):
    """Edge loops of a written STEP file that don't close (an edge ending at a
    vertex the next one doesn't start from): readers drop such faces. A list
    of loop ids, empty when all are sound."""
    import re
    ent = {}
    for m in re.finditer(r"^#(\d+)=([A-Z_0-9]+)\((.*)\);\s*$", text, re.M):
        ent[int(m.group(1))] = (m.group(2), m.group(3))
    refs = lambda a: [int(x) for x in re.findall(r"#(\d+)", a)]
    bad = []
    for i, (kind, args) in ent.items():
        if kind != "EDGE_LOOP":
            continue
        chain = []
        for oe in refs(args):
            k2, a2 = ent.get(oe, ("", ""))
            if k2 != "ORIENTED_EDGE":
                continue
            ec = refs(a2)[0]
            fwd = a2.rstrip().endswith(".T.")
            k3, a3 = ent.get(ec, ("", ""))
            v0, v1 = refs(a3)[:2]
            if not a3.rstrip().endswith(".T."):
                v0, v1 = v1, v0
            chain.append((v0, v1) if fwd else (v1, v0))
        if chain and any(chain[j][1] != chain[(j + 1) % len(chain)][0] for j in range(len(chain))):
            bad.append(i)
    return bad


# --------------------------------------------------------------------------
# Blender operator: File > Export > NURBS (.step)
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty  # noqa: E402
from bpy_extras.io_utils import ExportHelper  # noqa: E402


class NPROTO_OT_export_step(bpy.types.Operator, ExportHelper):
    """Export NURBS objects as STEP (AP214): exact B-spline surfaces and curves
    (true cylinders, planes, circles, ellipses and lines as STEP's own) with full B-rep
    topology; closed bodies as solids"""
    bl_idname = "nproto.export_step"
    bl_label = "Export STEP"
    bl_options = {'PRESET'}

    filename_ext = ".step"
    filter_glob: StringProperty(default="*.step;*.stp", options={'HIDDEN'})
    use_selection: BoolProperty(name="Selection Only", default=True)
    unit: EnumProperty(name="Unit", items=[("MM", "Millimetres", ""), ("CM", "Centimetres", ""),
                                           ("M", "Metres", "")], default="MM",
                       description="Length unit of the file; Blender units are taken as metres")
    global_scale: FloatProperty(name="Scale", default=1.0, min=1e-6, soft_max=1000.0)
    analytic: BoolProperty(name="Analytic Surfaces", default=True,
                           description="Write true planes, cylinders, cones, spheres, tori, circles, ellipses and lines as "
                                       "STEP's own analytic entities, as CAD programs do (holes, bosses, countersinks "
                                       "and fillets are then recognised); off: everything as B-splines")

    def execute(self, context):
        sc = context.scene.unit_settings.scale_length or 1.0      # Blender units -> metres
        objs = context.selected_objects if self.use_selection else context.scene.objects
        objs = [o for o in objs if store.is_nurbs(o) and o.visible_get()]
        if not objs:
            self.report({'WARNING'}, "No NURBS objects to export")
            return {'CANCELLED'}
        stats = export_step(self.filepath, objs, unit=self.unit, scale=self.global_scale * sc,
                            analytic=self.analytic)
        msg = (f"STEP: {stats['bodies']} bodies ({stats['solids']} solid, {stats['sheets']} sheet), "
               f"{stats['faces']} faces, {stats['edges']} edges")
        an = stats["analytic"]
        parts = [f"{an[k]} {w}" for k, w in (("CYLINDRICAL_SURFACE", "cylinder"), ("CONICAL_SURFACE", "cone"),
                                              ("SPHERICAL_SURFACE", "sphere"), ("TOROIDAL_SURFACE", "torus"))
                 if an[k]]
        if self.analytic and parts:
            msg += f"; true {', '.join(parts)} faces"
        if self.analytic and an["CIRCLE"]:
            msg += f"; {an['CIRCLE']} circles"
        if self.analytic and an["ELLIPSE"]:
            msg += f"; {an['ELLIPSE']} ellipses"
        if stats.get("open_loops"):
            msg += f"; {len(stats['open_loops'])} edge loop(s) don't close -- other programs may drop those faces"
        if stats.get("failed"):
            msg += f"; skipped: {', '.join(n for n, _ in stats['failed'])}"
            self.report({'WARNING'}, msg)
        elif stats.get("open_loops"):
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def menu_export(self, context):
    self.layout.operator(NPROTO_OT_export_step.bl_idname, text="NURBS (.step)")


classes = (NPROTO_OT_export_step,)


def register():
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
