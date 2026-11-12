"""Square: a four-sided patch from 4 boundaries with G0 / G1 / G2 to the
neighbouring surfaces along each side (Alias "Square").

Construction (CV level, one light net):
  * every side gives CV rows: the boundary (G0), + a tangent row along the
    neighbour's outward cross direction (G1), + a curvature row matching the
    neighbour's normal curvature across the edge (G2). Rows are fitted on the
    patch's own knot vector (default: degree 5, one span for G2 = 6x6 CVs).
  * Coons boolean sum on the CV grids: S = L_u + L_v - T, where L_v is built
    from the bottom/top rows, L_u from the left/right rows and T is L_u's
    construction applied to L_v (the corner correction). Corner CVs claimed by
    two sides are averaged.
"""

import numpy as np

from . import features, geom, store

CONT = {"FREE": 0, "G0": 0, "G1": 1, "G2": 2}


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


# --------------------------------------------------------------------------
# Side sampling: exact boundary points + neighbour cross data (world space)
# --------------------------------------------------------------------------

class Side:
    """P (m,3) boundary points by arc length; B outward cross direction (unit,
    tangent to the neighbour, pointing away from it); N neighbour normal;
    K normal curvature of the neighbour in direction B. has_surface False for curves."""

    def __init__(self, P, B=None, N=None, K=None, t=None, D2=None, knots=None, degree=None):
        self.P = np.asarray(P, float)
        # along-edge knot structure of the neighbour (interior knots on [0,1], in
        # this side's parameter t) when the edge is an iso-curve of it
        self.knots, self.degree = knots, degree
        self.src = None             # (object name, ref) of the input this side came from
        self.Nout = None            # the neighbour FACE's outward normal (orientation of the shading)
        self.Q = None               # neighbour's second fundamental form per sample (3x3)
        self.fade1 = self.fade2 = None  # per-sample weight of the G1 / G2 rows (eased at conflicting corners)
        self.free = False           # Loft with rails: a side with no condition (P is only the initial guess)
        self.B, self.N, self.K = B, N, K
        # second derivative into the patch belonging to the (non-unit) speed B
        self.D2 = D2 if D2 is not None or B is None else K[:, None] * N
        self.has_surface = B is not None
        # the source's own parameter at the samples, normalised to [0, 1]:
        # fitting on it reproduces polynomial boundaries exactly
        self.t = np.linspace(0, 1, len(self.P)) if t is None else np.asarray(t, float)

    def reversed(self):
        r = lambda a: None if a is None else a[::-1].copy()
        kn = None if self.knots is None else sorted(1.0 - k for k in self.knots)
        out = Side(self.P[::-1].copy(), r(self.B), r(self.N), r(self.K), 1.0 - self.t[::-1], r(self.D2), kn, self.degree)
        out.src = self.src
        out.Nout = r(self.Nout)
        out.Q = r(self.Q)
        out.fade1, out.fade2 = r(self.fade1), r(self.fade2)
        out.free = self.free
        return out


def _by_arclength(f, t0, t1, m):
    """Samples uniform in the source's own parameter (see Side.t)."""
    return np.linspace(t0, t1, m)


def grid_starts(surf, Q):
    """_foot's own start for every point of Q -- the nearest point of its
    17 x 17 grid -- with the grid evaluated once for all of them."""
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    uv = [(u0 + (u1 - u0) * i / 16, v0 + (v1 - v0) * j / 16) for i in range(17) for j in range(17)]
    G = np.array([surf.evaluate(u, v) for u, v in uv], float)
    Q = np.asarray(Q, float).reshape(-1, 3)
    return [uv[k] for k in np.argmin(np.linalg.norm(G[None, :, :] - Q[:, None, :], axis=2), axis=1)]


def _foot(surf, q, uv0=None):
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    if uv0 is None:
        best = None
        for i in range(17):
            for j in range(17):
                u, v = u0 + (u1 - u0) * i / 16, v0 + (v1 - v0) * j / 16
                d = np.linalg.norm(np.array(surf.evaluate(u, v)) - q)
                if best is None or d < best[0]:
                    best = (d, u, v)
        _, u, v = best
    else:
        u, v = uv0
    for _ in range(40):
        pt, su, sv = surf.derivatives(u, v)
        r = np.array(pt) - q
        J = np.array([su, sv]).T
        try:
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        u = min(max(u + step[0], u0), u1)
        v = min(max(v + step[1], v0), v1)
        if np.linalg.norm(step) < 1e-13:
            break
    return u, v


def _second(surf, u, v):
    """Su, Sv, Suu, Suv, Svv by finite differences of the first derivatives
    (one-sided at the domain border)."""
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    hu, hv = 1e-5 * (u1 - u0), 1e-5 * (v1 - v0)
    _, su, sv = (np.array(x, float) for x in surf.derivatives(u, v))

    def d(axis):
        if axis == 0:
            a, b = max(u - hu, u0), min(u + hu, u1)
            _, sua, sva = surf.derivatives(a, v)
            _, sub, svb = surf.derivatives(b, v)
            return (np.array(sub) - np.array(sua)) / (b - a), (np.array(svb) - np.array(sva)) / (b - a)
        a, b = max(v - hv, v0), min(v + hv, v1)
        _, sua, _ = surf.derivatives(u, a)
        _, sub, _ = surf.derivatives(u, b)
        _, _, sva = surf.derivatives(u, a)
        _, _, svb = surf.derivatives(u, b)
        return (np.array(sub) - np.array(sua)) / (b - a), (np.array(svb) - np.array(sva)) / (b - a)

    suu, svu = d(0)
    suv, svv = d(1)
    return su, sv, suu, 0.5 * (suv + svu), svv


def sample_side(obj, ref, m):
    """Side data for a curve (G0 only) or a NURBS surface edge."""
    sm, _ = _k()
    M = geom.world_matrix(obj)
    if not features.is_edge_ref(ref) and not features.is_brep_edge_ref(ref):
        kc = features.input_curve(obj, ref)
        t0, t1 = kc.parameter_range()
        tt = _by_arclength(kc.evaluate, t0, t1, m)
        return Side([kc.evaluate(t) for t in tt], t=(tt - tt[0]) / ((tt[-1] - tt[0]) or 1.0))
    brep = store.get_brep(obj)
    R = np.array(M.to_3x3(), dtype=float)
    if features.is_brep_edge_ref(ref):
        edge = brep.edges()[int(ref[1:])]
        face = edge.faces()[0]
        surf = face.surface()
        t0, t1 = edge.parameter_range()
        tt = _by_arclength(lambda t: np.array(edge.point_at(t)), t0, t1, m)
        Pl = np.array([edge.point_at(t) for t in tt], float)
        uv, prev = [], None
        for q in Pl:
            prev = _foot(surf, q, prev)
            uv.append(prev)
        # kernel trim curves can have an irregular parameter: work on the
        # face's surface instead. An iso-curve edge (typical for blend ends)
        # becomes exactly like a natural edge: the surface's own parameter
        us, vs = np.array([a for a, _ in uv]), np.array([b for _, b in uv])
        (du0, du1), (dv0, dv1) = surf.u_domain(), surf.v_domain()
        iso = None
        if np.ptp(us) < 1e-6 * (du1 - du0) and np.ptp(vs) > 1e-3 * (dv1 - dv0):
            iso = "u"
        elif np.ptp(vs) < 1e-6 * (dv1 - dv0) and np.ptp(us) > 1e-3 * (du1 - du0):
            iso = "v"
        if iso == "u":
            uc = min(max(float(np.median(us)), du0), du1)
            tt = np.linspace(vs[0], vs[-1], m)
            uv = [(uc, x) for x in tt]
        elif iso == "v":
            vc = min(max(float(np.median(vs)), dv0), dv1)
            tt = np.linspace(us[0], us[-1], m)
            uv = [(x, vc) for x in tt]
        Pl = np.array([surf.evaluate(a, b) for a, b in uv], float)
        own = None
        if iso is None:
            own = _light_edge_curve(edge)
        if own is not None:
            # the edge's own 3D curve is a light B-spline (e.g. an edge made by
            # Square's re-trim): sample it in its own parameter and adopt its layout
            tt = np.linspace(t0, t1, m)
            Pl = np.array([edge.point_at(t) for t in tt], float)
            uv, prev = [], None
            for q in Pl:
                prev = _foot(surf, q, prev)
                uv.append(prev)
        elif iso is None:
            # irregular kernel parameter: resample evenly by arc length (so chord
            # parameters are uniform and simple edges fit with few CVs)
            d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pl, axis=0), axis=1))])
            target = np.linspace(0, d[-1], m)
            uva = np.array(uv)
            guess = np.array([np.interp(target, d, uva[:, k]) for k in range(2)]).T
            q3 = np.array([np.interp(target, d, Pl[:, k]) for k in range(3)]).T
            uv = [_foot(surf, q, tuple(g)) for q, g in zip(q3, guess)]
            Pl = np.array([surf.evaluate(a, b) for a, b in uv], float)
            tt = target
        brep_iso = iso
        brep_own = own
    else:
        face = brep.faces()[0]
        surf = face.surface()
        (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
        if ref in ("U0", "U1"):
            uc = u0 if ref == "U0" else u1
            f = lambda s: np.array(surf.evaluate(uc, s))
            tt = _by_arclength(f, v0, v1, m)
            uv = [(uc, s) for s in tt]
        else:
            vc = v0 if ref == "V0" else v1
            f = lambda s: np.array(surf.evaluate(s, vc))
            tt = _by_arclength(f, u0, u1, m)
            uv = [(s, vc) for s in tt]
        Pl = np.array([surf.evaluate(u, v) for u, v in uv], float)
    Md = np.array(M, dtype=float)          # double precision (mathutils is float32)
    Pw = Pl @ Md[:3, :3].T + Md[:3, 3]
    size = float(np.linalg.norm(np.ptp(Pl, axis=0))) or 1.0
    eps = 1e-3 * size
    B, N, K, NB, ND2, QF = [], [], [], [], [], []
    own_layout = None
    if features.is_brep_edge_ref(ref):
        natural_dir = brep_iso
        own_layout = brep_own
    else:
        natural_dir = "u" if ref in ("U0", "U1") else "v"
    t_norm = (np.asarray(tt) - tt[0]) / ((tt[-1] - tt[0]) or 1.0)
    for i, ((u, v), pl) in enumerate(zip(uv, Pl)):
        su, sv, suu, suv, svv = _second(surf, u, v)
        Su, Sv, Suu, Suv, Svv = (R @ x for x in (su, sv, suu, suv, svv))
        n = np.cross(Su, Sv)
        n /= np.linalg.norm(n) or 1.0
        T = Pw[min(i + 1, m - 1)] - Pw[max(i - 1, 0)]
        T /= np.linalg.norm(T) or 1.0
        b = np.cross(n, T)
        b /= np.linalg.norm(b) or 1.0
        bl = np.linalg.solve(Md[:3, :3], b)
        bl /= np.linalg.norm(bl) or 1.0
        dp = face.closest_point(tuple(pl + eps * bl))[1]
        dm = face.closest_point(tuple(pl - eps * bl))[1]
        if dm > dp:          # +b goes back onto the face: outward is -b
            b = -b
        # normal curvature of the neighbour in direction b
        a_b = np.linalg.lstsq(np.array([Su, Sv]).T, b, rcond=None)[0]
        a, c = a_b
        k = (a * a * (Suu @ n) + 2 * a * c * (Suv @ n) + c * c * (Svv @ n)) / max(b @ b, 1e-30)
        # second fundamental form as a 3x3 form on the tangent plane: II(w, w) = w Q w
        J = np.array([Su, Sv]).T
        Jp = np.linalg.pinv(J)
        H = np.array([[Suu @ n, Suv @ n], [Suv @ n, Svv @ n]])
        QF.append(Jp.T @ H @ Jp)
        B.append(b)
        N.append(n)
        K.append(k)
        if natural_dir is not None:     # the neighbour's own transverse derivatives
            d1 = Su if natural_dir == "u" else Sv
            d2 = Suu if natural_dir == "u" else Svv
            sgn = 1.0 if d1 @ b > 0 else -1.0
            NB.append(sgn * d1)
            ND2.append(d2)
    try:
        Nout = []
        for (u, v) in uv:
            no = np.array(face.outward_normal_at_uv(u, v), float)
            no = R @ no
            Nout.append(no / (np.linalg.norm(no) or 1.0))
        Nout = np.array(Nout)
    except Exception:
        Nout = None
    if natural_dir is not None:
        # constant reparameterisation lambda = 1/mean|N_t|: S_v = lambda*N_t,
        # S_vv = lambda^2 N_tt -- polynomial for polynomial neighbours, exact G2
        lam = 1.0 / (float(np.mean(np.linalg.norm(NB, axis=1))) or 1.0)
        # the neighbour's knots / degree along this edge, in the side's parameter
        _, ext_ = _k()
        dd = ext_.surface_cvs(surf)
        along = "v" if natural_dir == "u" else "u"
        kv = dd["knots_" + along]
        deg = dd["degree_" + along]
        a0, a1 = float(tt[0]), float(tt[-1])
        kn = sorted((k - a0) / (a1 - a0) for k in kv[deg + 1:len(kv) - deg - 1]) if a1 != a0 else []
        kn = [k for k in kn if 1e-9 < k < 1 - 1e-9]
        # only a side covering the neighbour's WHOLE edge has a meaningful layout;
        # a short piece of it (left over from a trim) is treated like a trim curve
        full = float(kv[-1] - kv[0]) or 1.0
        if abs(a1 - a0) < 0.95 * full:
            kn, deg = None, None
        sd_ = Side(Pw, np.array(NB) * lam, np.array(N), np.array(K), t_norm, np.array(ND2) * lam * lam, kn, deg)
        sd_.Nout = Nout
        sd_.Q = np.array(QF)
        return sd_
    if own_layout is not None:
        sd_ = Side(Pw, np.array(B), np.array(N), np.array(K), t_norm, None, own_layout[1], own_layout[0])
    else:
        sd_ = Side(Pw, np.array(B), np.array(N), np.array(K), t_norm)
    sd_.Nout = Nout
    sd_.Q = np.array(QF)
    return sd_


def _light_edge_curve(edge, max_interior=6):
    """(degree, interior knots normalised to the edge's range) when the edge's 3D
    curve is a B-spline with few knots whose parameter is the edge parameter."""
    _, ext = _k()
    try:
        d = ext.curve_cvs(edge.curve())
    except Exception:
        return None
    t0, t1 = edge.parameter_range()
    if t1 <= t0:
        return None
    p, U = d["degree"], list(d["knots"])
    interior = [(k - t0) / (t1 - t0) for k in U[p + 1:len(U) - p - 1]]
    interior = [k for k in interior if 1e-9 < k < 1 - 1e-9]
    if len(interior) > max_interior or p < 4:      # cubic trim curves: no layout worth copying
        return None
    try:     # the edge parameter must be the curve parameter
        q = np.array(edge.point_at(0.5 * (t0 + t1)))
        c = np.array(edge.curve().evaluate(0.5 * (t0 + t1)))
    except Exception:
        return None
    if np.linalg.norm(q - c) > 1e-9 * (1 + np.linalg.norm(q)):
        return None
    return p, interior


# --------------------------------------------------------------------------
# Patch construction
# --------------------------------------------------------------------------

def _chord_params(rows):
    N = len(rows[0])
    acc, cnt = np.zeros(N), 0
    for R in rows:
        d = np.linalg.norm(np.diff(R, axis=0), axis=1)
        if d.sum() > 1e-12:
            acc[1:] += np.cumsum(d) / d.sum()
            cnt += 1
    t = acc / max(cnt, 1) if cnt else np.linspace(0, 1, N)
    t[0], t[-1] = 0.0, 1.0
    return t


def _fit(rows, U, p, n_cvs, t=None):
    """Least-squares fit of several sample rows on ONE knot vector (ends exact),
    at parameters t (default: averaged chord length).
    Returns (CV rows, max residual per row)."""
    t = _chord_params(rows) if t is None else t
    B = np.hstack(rows)
    X = geom._ls_fit(t, B, U, p, n_cvs)
    A = geom.interp_matrix(t, U, p, n_cvs)
    res = np.linalg.norm((A @ X - B).reshape(len(t), len(rows), 3), axis=2).max(axis=0)
    return [X[:, 3 * i:3 * i + 3] for i in range(len(rows))], res


ANG_TOL = np.radians(0.1)    # tangent rows: direction error
CUR_TOL = 1e-2               # curvature rows: relative error


def _quality_ok(groups, res, size):
    """groups: list of (rows) per side in fit order; res: residual per row."""
    k = 0
    for rows in groups:
        for j in range(len(rows)):
            r = res[k]
            k += 1
            if j == 0:
                if r > 1e-4 * size:
                    return False
            else:
                step = float(np.mean(np.linalg.norm(rows[j] - rows[j - 1], axis=1))) or 1.0
                if r / step > (ANG_TOL if j == 1 else CUR_TOL):
                    return False
    return True


def _fit_dir(side_rows_fn, ta, tb, p, spans, size, max_spans=12):
    """Fit both sides of one direction (each on its own parameters, one shared
    knot vector); spans 0 = auto (fewest spans that meet the tolerances).
    side_rows_fn(U) -> (rows_a, rows_b)."""
    tries = range(1, max_spans + 1) if not spans else [spans]
    for s_ in tries:
        n = p + s_
        U = geom.clamped_uniform_knots(n, p)
        ra, rb = side_rows_fn(U)
        fa, res_a = _fit(ra, U, p, n, ta)
        fb, res_b = _fit(rb, U, p, n, tb)
        res = np.concatenate([res_a, res_b])
        if spans or _quality_ok([ra, rb], res, size):
            break
    return U, n, fa, fb, res


def _end_deriv(side, end):
    """d P / d t of a side at t=0 (end 0) or t=1 (end 1), 2nd-order one-sided."""
    P, t = side.P, side.t
    if end == 0:
        t0, t1, t2, p0, p1, p2 = t[0], t[1], t[2], P[0], P[1], P[2]
    else:
        t0, t1, t2, p0, p1, p2 = t[-1], t[-2], t[-3], P[-1], P[-2], P[-3]
    # derivative of the quadratic through 3 (possibly unevenly spaced) samples, at t0
    a, b = (t1 - t0) or 1e-12, (t2 - t0) or 1e-12
    return (p1 - p0) * (b / (a * (b - a))) - (p2 - p0) * (a / (b * (b - a))) if b != a else (p1 - p0) / a


def _corner_scale(side, i, target, L):
    """Scale a so that a*B[i] matches the adjacent boundary's derivative `target`."""
    if float(np.linalg.norm(target)) < 1e-9 * L:
        return 0.0          # the adjacent side is collapsed to a point
    b = side.B[i]
    a = float(target @ b) / max(float(b @ b), 1e-30)
    return a if a > 1e-9 * L else L


def _side_rows(side, k, w, L, U, p):
    """Sample rows (boundary first) of one side for continuity order k.
    Clamped start with knots U: C'(0) = p/u1 (P1-P0),
    C''(0) = p(p-1)/u1 [ (P2-P1)/u2 - (P1-P0)/u1 ]."""
    rows = [side.P]
    if k == 0 or not side.has_surface:
        return rows
    u1 = U[p + 1] - U[0]
    u2 = U[p + 2] - U[0] if p + 2 < len(U) else u1
    # L: a scalar, or (a0, a1) = scales matching the adjacent boundaries at the corners
    if isinstance(L, tuple):
        a = (1 - side.t) * L[0] + side.t * L[1]
    else:
        a = np.full(len(side.P), float(L))
    a = (w * a)[:, None]
    D1 = a * side.B
    P1 = side.P + (u1 / p) * D1
    rows.append(P1)
    if k >= 2:
        D2 = (a ** 2) * side.D2
        P2 = P1 + u2 * (D2 * u1 / (p * (p - 1)) + (P1 - side.P) / u1)
        rows.append(P2)
    return rows


def _greville(U, p, n_cvs):
    return np.array([sum(U[i + 1:i + p + 1]) / p for i in range(n_cvs)])


def _blend_grid(first_rows, last_rows, n_cvs, gre):
    """CV rows 0..len(first)-1 from `first`, the last ones from `last` (reversed
    order: last_rows[0] is the boundary), interior rows linear in Greville."""
    G = [None] * n_cvs
    for j, r in enumerate(first_rows):
        G[j] = r
    for j, r in enumerate(last_rows):
        G[n_cvs - 1 - j] = r
    a, b = len(first_rows) - 1, n_cvs - len(last_rows)
    for j in range(a + 1, b):
        s = (gre[j] - gre[a]) / ((gre[b] - gre[a]) or 1.0)
        G[j] = (1 - s) * G[a] + s * G[b]
    return np.array(G)


CORNER_FADE = 0.5          # fraction of a side over which a corner-limited condition comes back
CORNER_G1_DEG = 1.0        # neighbour normals further apart than this: no common tangent plane
CORNER_G2_REL = 0.05       # curvature mismatch (vs the largest curvature involved) that G2 can't bridge


def _end_curvature(P, end):
    """Curvature vector of the polyline P at its start (end=0) or end (-1):
    central differences at the 2nd..4th samples, extrapolated to the end."""
    Q = P if end == 0 else P[::-1]
    if len(Q) < 5:
        return np.zeros(3)
    ks = []
    for i in (1, 2, 3):
        a, b, c = Q[i - 1], Q[i], Q[i + 1]
        h1, h2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
        if h1 < 1e-12 or h2 < 1e-12:
            return np.zeros(3)
        ks.append(2.0 * ((c - b) / h2 - (b - a) / h1) / (h1 + h2))
    return 3.0 * ks[0] - 3.0 * ks[1] + ks[2]

LAST_NOTES = []


def corner_notes(sides, conflicts):
    """One readable line per eased corner."""
    name = lambda sd: f"{sd.src[0]} {sd.src[1]}" if sd.src else "curve"
    seen, out = set(), []
    for i, e, kind, j in conflicts:
        key = (min(i, j), max(i, j), kind)
        if key in seen:
            continue
        seen.add(key)
        why = "tangent planes differ" if kind == "G1" else "curvatures differ"
        out.append(f"{name(sides[i])} / {name(sides[j])}: {why}, {kind} eased at the corner")
    return out


def corner_fades(sides, ks):
    """A side's G1 / G2 can be impossible at a corner: the two neighbours'
    tangent planes differ there (no patch is tangent to both), or the adjacent
    boundary curve's normal curvature differs from what this side's neighbour
    needs. Forcing those rows throws the CVs out, so they are eased to zero
    toward such corners (Alias / ICEM do the same) and come back to full weight
    over CORNER_FADE of the side. Sets sd.fade1 / sd.fade2; returns the conflicts."""
    bottom, right, top, left = sides
    allp = np.vstack([sd.P for sd in sides])
    size = float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0
    kref = max([1.0 / size] + [float(np.abs(sd.K).max()) for sd, k in zip(sides, ks) if k >= 2 and sd.has_surface])
    for sd in sides:
        sd.fade1 = np.ones(len(sd.t))
        sd.fade2 = np.ones(len(sd.t))
    # (side a, its end, side b, its end) for the four corners
    corners = [(0, 0, 3, 0), (0, -1, 1, 0), (2, 0, 3, -1), (2, -1, 1, -1)]
    limit = {}            # (side index, end) -> highest continuity possible at that corner

    def lim(i, e, c):
        limit[(i, e)] = min(limit.get((i, e), 3), c)

    conflicts = []
    for ia, ea, ib, eb in corners:
        A, B = sides[ia], sides[ib]
        for (i, e, S, j, f, T) in ((ia, ea, A, ib, eb, B), (ib, eb, B, ia, ea, A)):
            k = ks[i]
            if k == 0 or not S.has_surface or (T.free and not getattr(T, "shaped", False)):
                continue
            n = S.N[e] / (np.linalg.norm(S.N[e]) or 1.0)
            tb = T.P[1] - T.P[0] if f == 0 else T.P[-2] - T.P[-1]
            tb = tb / (np.linalg.norm(tb) or 1.0)
            g1_bad = abs(float(tb @ n)) > np.sin(np.radians(CORNER_G1_DEG))
            if ks[j] >= 1 and T.has_surface:
                m = T.N[f] / (np.linalg.norm(T.N[f]) or 1.0)
                g1_bad = g1_bad or np.degrees(np.arccos(min(1.0, abs(float(n @ m))))) > CORNER_G1_DEG
            if g1_bad:
                lim(i, e, 0)
                conflicts.append((i, e, "G1", j))
                continue
            if k >= 2 and S.Q is not None:
                kc = float(_end_curvature(T.P, f) @ n) * np.sign(float(S.N[e] @ n) or 1.0)
                kn = float(tb @ S.Q[e] @ tb)
                if abs(kc - kn) > CORNER_G2_REL * max(kref, abs(kc), abs(kn)):
                    lim(i, e, 1)
                    conflicts.append((i, e, "G2", j))
    for (i, e), c in limit.items():
        sd = sides[i]
        t = sd.t if e == 0 else 1.0 - sd.t
        x = np.clip(t / CORNER_FADE, 0.0, 1.0)
        ramp = x * x * (3.0 - 2.0 * x)
        if c < 1:
            sd.fade1 = np.minimum(sd.fade1, ramp)
        if c < 2:
            sd.fade2 = np.minimum(sd.fade2, ramp)
    return conflicts


def square_surface(sides, conts, weights, degree=0, spans_u=0, spans_v=0, max_auto=6, match=True, retrim=False,
                   max_move=None):
    """sides: [bottom, right, top, left] Side objects, oriented so that bottom and
    top run along +u, left and right along +v. conts/weights per side.
    spans 0 = auto: the fewest spans whose net keeps every side's boundary,
    tangent and curvature rows within tolerance (checked on the finished net).
    Returns (kernel surface, info dict)."""
    _, ext = _k()
    bottom, right, top, left = sides
    # Free: the side has no condition at all (not even position): the patch's
    # edge there follows from the other sides and the net's fairness
    for c, sd in zip(conts, sides):
        sd.free = (c == "FREE")
    ks = [CONT[c] if sd.has_surface else 0 for c, sd in zip(conts, sides)]
    du, dv = (degree, degree) if not isinstance(degree, (tuple, list)) else (int(degree[0]), int(degree[1]))
    du, dv = int(du or 0), int(dv or 0)
    conflicts = corner_fades(sides, ks)
    LAST_NOTES[:] = corner_notes(sides, conflicts)
    kb, kr, kt, kl = ks
    def make(r):
        S, (Uu, Uv), _, info = r
        nu, nv = S.shape[:2]
        pts = [tuple(map(float, S[i, j])) for i in range(nu) for j in range(nv)]
        info["folds"] = net_folds(S)
        return ext.surface_from_cvs(info["degree"][0], info["degree"][1], pts, Uu, Uv), info

    # a kernel trim curve as a side can only be followed to ~1e-3 by a light
    # net; don't chase that with spans (Class-A: light patch first)
    trim_side = any(sd.has_surface and sd.knots is None for sd in sides)
    gap_rel = 1e-3 if trim_side else 1e-4

    def score_of(q_):
        sc = max(q_["gap"] / (gap_rel * q_["size"]), q_["angle"] / 0.1, q_["curv"] / 0.10)
        # a crossed CV net is never acceptable, however close the edges are
        return sc + 100.0 * q_.get("folds", 0) if q_.get("folds") else sc

    if match and not du and not dv and not spans_u and not spans_v and any(sd.knots is not None for sd in sides):
        m_surf, m_info = _square_matched(sides, ks, weights, make, score_of, retrim, max_move)
        if retrim and m_info.get("clean_edges"):
            return m_surf, m_info
        # the neighbours' layout isn't always the light one (a degree-3 neighbour
        # next to a G2 side needs extra spans): compare with the Class-A layouts
        m_score = score_of(m_info)
        cands = [(m_score, m_surf, m_info)] + _light_candidates(sides, ks, weights, make, score_of,
                                                                 max_cvs=m_info["cvs"][0] * m_info["cvs"][1])
        return _pick_lightest(cands)
    if not du and not dv and not spans_u and not spans_v:
        cands = _light_candidates(sides, ks, weights, make, score_of)
        if cands:
            return _pick_lightest(cands)
    if not du and not dv:
        du = dv = max(3, 1 + kl + kr, 1 + kb + kt)
    # one direction set: the other is the lowest the continuity needs in one span
    # (u runs along bottom / top, so its degree carries the left / right conditions)
    p = int(du or max(3, 1 + kl + kr))
    q_ = int(dv or max(3, 1 + kb + kt))

    if not spans_u or not spans_v:
        best, tried_f = None, []
        for s_ in range(1, max_auto + 1):
            try:
                r = _build(sides, ks, weights, p, spans_u or s_, spans_v or s_, q_)
            except RuntimeError:
                continue
            surf, info = make(r)
            q = measure(surf, sides, ks)
            info.update(q)
            score = score_of(info)
            tried_f.append((score, surf, info))
            if best is None or score < best[0]:
                best = (score, surf, info)
            if score <= 1.0:
                break
            if len(tried_f) >= 3 and min(t[0] for t in tried_f[-2:]) > 0.75 * min(t[0] for t in tried_f[:-2]):
                break
        if best is None:
            raise RuntimeError(f"degree {p} x {q_}: too few CVs for the chosen continuity")
        if best[0] > 1.0:
            best = min((t for t in tried_f if t[0] <= 1.5 * best[0]), key=lambda t: t[2]["cvs"][0] * t[2]["cvs"][1])
        _, surf, info = best
        return surf, info
    surf, info = make(_build(sides, ks, weights, p, spans_u, spans_v, q_))
    info.update(measure(surf, sides, ks))
    return surf, info


def _light_candidates(sides, ks, weights, make, score_of, max_cvs=None, limit=10):
    """Class-A layouts lightest first: degree 3 / 5 with the fewest spans (at
    equal CV count the higher degree, i.e. fewer spans). Stops at the first one
    within tolerance. Returns [(score, surf, info)]."""
    order = sorted(((p_, s_) for p_ in (3, 5) for s_ in range(1, 7)), key=lambda t: (t[0] + t[1], -t[0]))
    out = []
    for p_, s_ in order:
        n = p_ + s_
        if max_cvs is not None and n * n > max_cvs:
            break
        try:
            r = _build(sides, ks, weights, p_, s_, s_)
        except RuntimeError:
            continue
        surf, info = make(r)
        info.update(measure(surf, sides, ks))
        sc = score_of(info)
        out.append((sc, surf, info))
        if sc <= 1.0 or len(out) >= limit:
            break
    return out


def _pick_lightest(cands):
    """The lightest net within tolerance; if none is, the lightest within 1.5x
    of the best score (a heavier net that is only marginally better isn't worth it)."""
    ok = [c for c in cands if c[0] <= 1.0]
    if not ok:
        best = min(c[0] for c in cands)
        ok = [c for c in cands if c[0] <= 1.5 * best]
        for c in ok:
            c[2]["tolerance_reached"] = False
    sc, surf, info = min(ok, key=lambda c: c[2]["cvs"][0] * c[2]["cvs"][1])
    return surf, info



def _merge_knots(lists, tol=1e-6):
    out = []
    for k in sorted(k for kn in lists for k in kn):
        if not out or abs(k - out[-1]) > tol:
            out.append(k)
    return out


def _merge_knots_mult(groups, tol=1e-6):
    """Union of knot lists, each knot with its required multiplicity (max over groups)."""
    need = []
    for kn, mult in groups:
        for k in _merge_knots([kn]):
            for e in need:
                if abs(e[0] - k) <= tol:
                    e[1] = max(e[1], mult)
                    break
            else:
                need.append([k, mult])
    return sorted(k for k, mlt in need for _ in range(mlt))


MAX_EDGE_MOVE = 5e-2    # re-trim: how far (relative to the size) a clean edge may move the face's edge
CLEAN_HERMITE = True    # True: pin the clean edge's end tangent / curvature to the original's


def _clean_side(sd, deg, interior):
    """Replace a trim-curve side by ONE clean curve in the matched layout (ends
    kept), so the square stays light; the target face is re-trimmed to it."""
    U = [0.0] * (deg + 1) + list(interior) + [1.0] * (deg + 1)
    n = len(U) - deg - 1
    # end rows from the original edge's end tangent and curvature (so the
    # corners stay compatible with the neighbouring sides); the rest by least squares
    def end_derivs(P, t):
        c = np.polyfit(t[:7] - t[0], P[:7], 3)          # local cubic, per coordinate
        return c[2], 2 * c[1]
    d1a, d2a = end_derivs(sd.P, sd.t)
    d1b, d2b = end_derivs(sd.P[::-1], 1.0 - sd.t[::-1])
    k = min(2, (n - 2) // 2) if CLEAN_HERMITE else 0
    X = np.zeros((n, 3))
    X[0], X[-1] = sd.P[0], sd.P[-1]
    for (i0, sgn, P0, d1, d2) in ((0, 1, sd.P[0], d1a, d2a), (n - 1, -1, sd.P[-1], d1b, d2b)):
        Ux = U if sgn == 1 else [1.0 - x for x in reversed(U)]
        u1, u2 = Ux[deg + 1], Ux[deg + 2] if deg + 2 < len(Ux) else Ux[deg + 1]
        if k >= 1:
            X[i0 + sgn] = P0 + (u1 / deg) * d1
        if k >= 2:
            X[i0 + 2 * sgn] = X[i0 + sgn] + u2 * (d2 * u1 / (deg * (deg - 1)) + (X[i0 + sgn] - P0) / u1)
    free = list(range(k + 1, n - k - 1))
    A = geom.interp_matrix(sd.t, U, deg, n)
    if free:
        fixed = [i for i in range(n) if i not in free]
        R = sd.P - A[:, fixed] @ X[fixed]
        X[free] = np.linalg.lstsq(A[:, free], R, rcond=None)[0]
    P = A @ X
    T = np.gradient(P, axis=0)
    B = np.cross(sd.N, T)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-30
    B *= np.sign(np.sum(B * sd.B, axis=1, keepdims=True) + 1e-30)
    dev = float(np.linalg.norm(P - sd.P, axis=1).max())
    sd.P, sd.B = P, B
    sd.D2 = sd.K[:, None] * sd.N
    sd.knots, sd.degree = list(interior), deg
    return {"src": sd.src, "degree": deg, "knots": U, "cvs": X, "deviation": dev}


def _square_matched(sides, ks, weights, make, score_of, retrim=False, max_move=None):
    """Degree and knots taken from the neighbours along their iso edges, so the
    patch's boundary rows ARE the neighbours' edge CV rows (Alias / Plasticity
    style CV-aligned square). Directions without an iso neighbour get the
    continuity-based degree and one span. Knots are refined (span midpoints)
    only if a non-matching side (e.g. a trim curve) needs it."""
    bottom, right, top, left = sides
    kb, kr, kt, kl = ks
    dirs = []
    # u direction: knots from bottom/top (they run along u), but the number of
    # CV columns must hold the LEFT/RIGHT rows; v direction the other way round
    for pair, (ka, kb_) in (((bottom, top), (kl, kr)), ((left, right), (kb, kt))):
        iso = [sd for sd in pair if sd.knots is not None]
        if iso:
            # at least the degree one span needs for the continuity; a neighbour of
            # lower degree stays exact: its knots are repeated (deg - its degree + 1)x
            deg = max(max(sd.degree for sd in iso), min(5, 1 + ka + kb_), 1)
            interior = _merge_knots_mult([(sd.knots, deg - sd.degree + 1) for sd in iso])
        else:
            # no neighbour layout to adopt: the Class-A default, one span of degree
            # 5 when any side is G2 (3 for G1 / G0)
            deg, interior = (5 if max(ks) >= 2 else max(3, 1 + ka + kb_)), []
        while deg + 1 + len(interior) < ka + kb_ + 2:     # enough CV rows for the continuity
            pts = [0.0] + interior + [1.0]
            j = int(np.argmax(np.diff(pts)))
            interior = sorted(interior + [0.5 * (pts[j] + pts[j + 1])])
        dirs.append([deg, interior])
    base = [list(d[1]) for d in dirs]
    # sides to replace by a clean curve (trim curves on planar faces, re-trimmed later)
    targets = []
    if retrim:
        for di, pair in enumerate(((bottom, top), (left, right))):
            for sd in pair:
                if sd.has_surface and sd.knots is None and sd.src is not None \
                        and features.is_brep_edge_ref(sd.src[1]) and is_planar_target(sd.src[0]):
                    targets.append((di, sd, (sd.P.copy(), sd.B.copy(), sd.D2.copy())))
    allp = np.vstack([sd.P for sd in sides])
    size = float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0
    move_tol = (MAX_EDGE_MOVE if max_move is None else max_move) * size
    steps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]       # uniform spans merged in, for non-matching sides
    best, tried = None, []
    for it in range(len(steps)):
        clean = []
        for di, sd, (P0, B0, D20) in targets:        # re-clean from the original edge each round
            sd.P, sd.B, sd.D2, sd.knots, sd.degree = P0.copy(), B0.copy(), D20.copy(), None, None
            clean.append(_clean_side(sd, dirs[di][0], dirs[di][1]))
        (p, iu), (q, iv) = dirs
        Uu = [0.0] * (p + 1) + iu + [1.0] * (p + 1)
        Uv = [0.0] * (q + 1) + iv + [1.0] * (q + 1)
        try:
            r = _build_knots(sides, ks, weights, Uu, p, Uv, q, const_alpha=True)
        except RuntimeError:
            break
        surf, info = make(r)
        qm = measure(surf, sides, ks)
        info.update(qm)
        info["matched"] = True
        info["clean_edges"] = clean
        info["edge_moved"] = max([c["deviation"] for c in clean], default=0.0)
        sc = max(score_of(info), info["edge_moved"] / move_tol if clean else 0.0)
        snap_ = [(sd.P.copy(), sd.B.copy(), sd.D2.copy(), sd.knots, sd.degree) for _, sd, _ in targets]
        tried.append((sc, surf, info, snap_))
        if best is None or sc < best[0]:
            best = (sc, surf, info, snap_)
        if sc <= 1.0:
            break
        # diminishing returns: two refinements in a row that do not improve the
        # best by 25% -> stop (an irregular trim edge can't be met exactly anyway)
        if len(tried) >= 3 and min(t[0] for t in tried[-2:]) > 0.75 * min(t[0] for t in tried[:-2]):
            break
        # refine the directions that have a side NOT matching a neighbour's knots
        if it + 1 >= len(steps):
            break
        n_ = steps[it + 1]
        for di, (d, b_, pair) in enumerate(zip(dirs, base, ((bottom, top), (left, right)))):
            needs = any(sd.knots is None for sd in pair) or any(t[0] == di for t in targets)
            if needs:
                extra = [k / n_ for k in range(1, n_) if all(abs(k / n_ - b) > 0.25 / n_ for b in b_)]
                d[1] = sorted(b_ + extra)
    if tried and best[0] > 1.0:
        # tolerance not reachable: the LIGHTEST net within 1.5x of the best score
        best = min((t for t in tried if t[0] <= 1.5 * best[0]), key=lambda t: t[2]["cvs"][0] * t[2]["cvs"][1])
        best[2]["tolerance_reached"] = False
    if best is not None:     # leave the sides as in the chosen net (measure / re-trim use them)
        for (di, sd, _), st in zip(targets, best[3]):
            sd.P, sd.B, sd.D2, sd.knots, sd.degree = st
    if best is None:
        raise RuntimeError("could not build a matched square")
    return best[1], best[2]


def side_is_uv(at):
    """'v' if the side lies on a v = const line (bottom / top): its cross derivative is S_v."""
    u_a, v_a = at(0.25)
    u_b, v_b = at(0.75)
    return "v" if abs(v_a - v_b) < 1e-12 else "u"


def measure(surf, sides, ks, n=15):
    """Achieved continuity along the 4 sides (patch params known from the
    construction): max gap, max normal angle (deg, G1/G2 sides) and max
    curvature mismatch across the edge (G2 sides; relative to the largest
    neighbour curvature)."""
    bottom, right, top, left = sides
    allp = np.vstack([sd.P for sd in sides])
    size = float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0
    gap = ang = curv = 0.0
    kref = max([1.0 / size] + [float(np.abs(sd.K).max()) for sd, k in zip(sides, ks) if k >= 2 and sd.has_surface])
    for sd, k, at in ((bottom, ks[0], lambda t: (t, 0.0)), (right, ks[1], lambda t: (1.0, t)),
                      (top, ks[2], lambda t: (t, 1.0)), (left, ks[3], lambda t: (0.0, t))):
        if sd.free:
            continue
        idx = np.linspace(1, len(sd.P) - 2, n).astype(int)
        for i in idx:
            u, v = at(sd.t[i])
            gap = max(gap, float(np.linalg.norm(np.array(surf.evaluate(u, v)) - sd.P[i])))
            if k == 0 or not sd.has_surface:
                continue
            su, sv, suu, suv, svv = _second(surf, u, v)
            nrm = np.cross(su, sv)
            nrm /= np.linalg.norm(nrm) or 1.0
            f1 = 1.0 if sd.fade1 is None else float(sd.fade1[i])
            f2 = 1.0 if sd.fade2 is None else float(sd.fade2[i])
            ang = max(ang, f1 * float(np.degrees(np.arccos(min(1.0, abs(nrm @ sd.N[i]))))))
            if k >= 2:
                if sd.Q is not None:
                    # the patch's cross direction at this side
                    cdir = sv if at is not None and side_is_uv(at) == "v" else su
                    c = cdir / (np.linalg.norm(cdir) or 1.0)
                    a_, c_ = np.linalg.lstsq(np.array([su, sv]).T, c, rcond=None)[0]
                    kp = (a_ * a_ * (suu @ nrm) + 2 * a_ * c_ * (suv @ nrm) + c_ * c_ * (svv @ nrm)) * np.sign(nrm @ sd.N[i])
                    kt = float(c @ sd.Q[i] @ c)
                else:
                    b = sd.B[i] / (np.linalg.norm(sd.B[i]) or 1.0)
                    a_, c_ = np.linalg.lstsq(np.array([su, sv]).T, b, rcond=None)[0]
                    kp = (a_ * a_ * (suu @ nrm) + 2 * a_ * c_ * (suv @ nrm) + c_ * c_ * (svv @ nrm)) * np.sign(nrm @ sd.N[i])
                    kt = sd.K[i]
                curv = max(curv, f2 * abs(kp - kt) / kref)
    if EXTRA_G0:      # Loft with rails: inner profiles / guide rail points
        step = max(1, len(EXTRA_G0) // 200)
        for u, v, P in EXTRA_G0[::step]:
            gap = max(gap, float(np.linalg.norm(np.array(surf.evaluate(u, v)) - P)))
    eased = sum(1 for sd in sides for f in (sd.fade1, sd.fade2) if f is not None for e in (0, -1) if f[e] < 1.0)
    return {"gap": gap, "angle": ang, "curv": curv, "size": size, "eased_corners": eased}


def _build(sides, ks, weights, p, spans_u, spans_v, q=None):
    """One net with fixed spans (uniform knots); degree p along u, q along v."""
    q = p if q is None else q
    nu, nv = p + spans_u, q + spans_v
    return _build_knots(sides, ks, weights, geom.clamped_uniform_knots(nu, p), p,
                        geom.clamped_uniform_knots(nv, q), q)


EXTRA_G0 = []       # Loft with rails: extra position conditions (u, v, point) inside the patch
SOLVER = "lsq"      # "lsq": least squares over the whole net; "rules": boolean sum + ownership only
G2_PASSES = 12
PREF_NATURAL = True
W_G0, W_G1, W_G2, W_PREF1, W_PREF2, W_FILL = 1e3, 1e3, 1e2, 0.1, 0.03, 1e-3
W_FAIR, FAIR_REF = 1.0, "x0"   # net fairness vs the boolean-sum net (stops inner rows crossing; exact cases stay exact)
FAIR_STEPS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)   # fairness raised until the net is clean
NET_EXT_MAX = 1.5                         # CV net extent vs the boundary's, above = blown up


def net_folds(X):
    """Number of quad corners of the CV net that turn the wrong way (a crossed /
    bow-tie quad has at least one)."""
    nu, nv = X.shape[:2]
    tiny = 1e-9 * (float(np.ptp(X.reshape(-1, 3), axis=0).max()) or 1.0)
    bad = 0
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b, c, d = X[i, j], X[i + 1, j], X[i + 1, j + 1], X[i, j + 1]
            if min(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(d - c), np.linalg.norm(a - d)) < tiny:
                continue        # a quad on a collapsed side (3-sided patch) is a triangle
            cr = (np.cross(b - a, d - a), np.cross(c - b, a - b), np.cross(d - c, b - c), np.cross(a - d, c - d))
            nq = cr[0] + cr[1] + cr[2] + cr[3]
            ln = np.linalg.norm(nq)
            if ln < 1e-14:
                continue
            # a corner that turns back by more than 5% of the quad's mean corner area
            # (a nearly straight corner of a smooth net is not a fold)
            bad += sum(1 for v in cr if float(v @ nq) / ln < -0.05 * ln / 4.0)
    return bad


def _end_stencil(U, p, at_end):
    """Weights on the boundary CV line and the next two (inward) for the value,
    the inward first derivative and the second derivative at a clamped end."""
    if at_end:
        U = [U[-1] - x for x in reversed(U)]          # mirror: inward = +parameter
    u1 = U[p + 1] - U[0]
    u2 = (U[p + 2] - U[0]) if p + 2 < len(U) else u1
    d1 = (-p / u1, p / u1, 0.0)
    if p >= 2:
        c = p * (p - 1) / u1
        d2 = (c / u1, -c * (1 / u1 + 1 / u2), c / u2)
    else:
        d2 = (0.0, 0.0, 0.0)
    return d1, d2


def _ls_solve(X0, sides, ks, weights, prefs, Uu, p, Uv, q, size):
    """Least-squares CV net: G0 (position), G1 (cross tangent in the neighbour's
    tangent plane) and G2 (normal curvature across the edge) as weighted
    equations along every side, soft preferences for the cross derivatives'
    tangential parts, and a weak pull to the boolean-sum net X0."""
    nu, nv = X0.shape[:2]
    nvar = nu * nv * 3
    var = lambda i, j, c: (i * nv + j) * 3 + c
    rows, rhs = [], []
    g0_rows, g0_rhs = [], []
    g2_meta = []      # (row index, weight, normal curvature target, cross-derivative stencil)

    def add(coefs, value, w, g0=False):
        r = np.zeros(nvar)
        for k, a in coefs:
            r[k] += a
        (g0_rows if g0 else rows).append(w * r)
        (g0_rhs if g0 else rhs).append(w * value)

    # lines: (boundary line index, step inward, along-knots, along-degree, at_end, transposed)
    geo = [(0, 1, Uu, p, Uv, q, False, False),        # bottom: v = 0, lines are rows j
           (nu - 1, -1, Uv, q, Uu, p, True, True),     # right:  u = 1, lines are columns i
           (nv - 1, -1, Uu, p, Uv, q, True, False),    # top:    v = 1
           (0, 1, Uv, q, Uu, p, False, True)]          # left:   u = 0
    for sd, k, w, pref, (line0, step, Ua, pa, Uc, pc, at_end, transp) in zip(sides, ks, weights, prefs, geo):
        if sd.free:
            continue
        n_al = len(Ua) - pa - 1
        A = geom.interp_matrix(sd.t, Ua, pa, n_al)                 # (m, n_along)
        d1, d2 = _end_stencil(Uc, pc, at_end)
        cv = (lambda a, l: (l, a)) if transp else (lambda a, l: (a, l))
        lines = [line0 + step * r for r in range(3)]
        lines = [l for l in lines if 0 <= l < (nu if transp else nv)]
        if isinstance(pref, tuple):
            al = (1 - sd.t) * pref[0] + sd.t * pref[1]
        else:
            al = np.full(len(sd.t), float(pref))
        al = (w * al)[:, None]
        has = sd.has_surface
        D1 = al * sd.B if has else None
        D2 = (al ** 2) * sd.D2 if has else None
        if has and PREF_NATURAL:
            # tangential preference = the patch's natural cross direction (the
            # boolean-sum net's cross derivative) projected into the neighbour's
            # tangent plane; the neighbour's own cross direction can be oblique
            # to the patch (short trimmed pieces) and shears the CV net
            d1x = np.array([sum(A[m_][a] * sum(d1[r] * X0[cv(a, lines[r])] for r in range(2))
                                for a in np.nonzero(np.abs(A[m_]) > 1e-14)[0]) for m_ in range(len(sd.t))])
            nn = sd.N / (np.linalg.norm(sd.N, axis=1, keepdims=True) + 1e-30)
            d1x = d1x - np.sum(d1x * nn, axis=1, keepdims=True) * nn
            D1 = d1x
            # second derivative: keep the normal target; tangential part from the net
            d2x = np.array([sum(A[m_][a] * sum(d2[r] * X0[cv(a, lines[r])] for r in range(3))
                                for a in np.nonzero(np.abs(A[m_]) > 1e-14)[0]) for m_ in range(len(sd.t))]) \
                if len(lines) >= 3 else np.zeros_like(d1x)
            d2n = np.sum(D2 * nn, axis=1, keepdims=True) * nn
            D2 = d2x - np.sum(d2x * nn, axis=1, keepdims=True) * nn + d2n
        s1 = float(np.mean(np.linalg.norm(D1, axis=1))) if has else 1.0
        s1 = s1 or 1.0
        for m_ in range(len(sd.t)):
            Arow = A[m_]
            nz = np.nonzero(np.abs(Arow) > 1e-14)[0]
            for c in range(3):                                      # G0 (solved first)
                add([(var(*cv(a, lines[0]), c), Arow[a]) for a in nz], sd.P[m_, c], 1.0, g0=True)
            if not has or k == 0:
                continue
            n = sd.N[m_]
            f1 = 1.0 if sd.fade1 is None else float(sd.fade1[m_])
            f2 = 1.0 if sd.fade2 is None else float(sd.fade2[m_])
            # G1: n . S_cross = 0
            add([(var(*cv(a, lines[r]), c), Arow[a] * d1[r] * n[c]) for a in nz for r in range(2) for c in range(3)],
                0.0, f1 * W_G1 / s1)
            for c in range(3):                                      # tangential preference
                add([(var(*cv(a, lines[r]), c), Arow[a] * d1[r]) for a in nz for r in range(2)],
                    D1[m_, c], f1 * W_PREF1 / s1)
            if k >= 2 and len(lines) >= 3 and pc >= 2 and f2 > 0:
                s2 = s1 * s1 / size
                # normal curvature target, kept for the update passes below
                b1 = float(D1[m_] @ D1[m_]) or 1e-30
                g2_meta.append((len(rows), f2 * W_G2 / s2, float(n @ D2[m_]) / b1,
                                [(cv(a, lines[r]), Arow[a] * d1[r]) for a in nz for r in range(2)],
                                None if sd.Q is None else sd.Q[m_]))
                add([(var(*cv(a, lines[r]), c), Arow[a] * d2[r] * n[c]) for a in nz for r in range(3) for c in range(3)],
                    float(n @ D2[m_]), f2 * W_G2 / s2)
                for c in range(3):
                    add([(var(*cv(a, lines[r]), c), Arow[a] * d2[r]) for a in nz for r in range(3)],
                        D2[m_, c], f2 * W_PREF2 / s2)
    # extra position conditions inside the patch (Loft with rails), solved with G0
    for u, v, P in EXTRA_G0:
        su_ = geom.find_span(nu - 1, p, u, Uu)
        Nu = geom.basis_funs(su_, u, p, Uu)
        sv_ = geom.find_span(nv - 1, q, v, Uv)
        Nv = geom.basis_funs(sv_, v, q, Uv)
        for c in range(3):
            add([(var(su_ - p + a, sv_ - q + b, c), Nu[a] * Nv[b]) for a in range(p + 1) for b in range(q + 1)],
                float(P[c]), 1.0, g0=True)
    # net fairness: even CV spacing along every row and column (second
    # differences of the control net). The cross-derivative rows only fix the
    # normal parts; this sets the tangential parts so the inner rows spread
    # evenly instead of shooting across each other.
    fair_rows = []
    if W_FAIR > 0:
        wf = W_FAIR / size
        for i in range(nu):
            for j in range(nv):
                for di, dj in ((1, 0), (0, 1)):
                    if i - di >= 0 and i + di < nu and j - dj >= 0 and j + dj < nv:
                        ref = X0[i - di, j - dj] - 2 * X0[i, j] + X0[i + di, j + dj] if FAIR_REF == "x0" else np.zeros(3)
                        for c in range(3):
                            fair_rows.append(len(rows))
                            add([(var(i - di, j - dj, c), 1.0), (var(i, j, c), -2.0), (var(i + di, j + dj, c), 1.0)],
                                ref[c], wf)
    # weak pull to the boolean-sum net (exact on polynomial data, fills free CVs)
    for i in range(nu):
        for j in range(nv):
            for c in range(3):
                add([(var(i, j, c), 1.0)], X0[i, j, c], W_FILL / size)
    # lexicographic least squares: the boundaries (G0) first, as well as they can
    # be met; G1 / G2 and the shape preferences only use the remaining freedom
    A0, b0 = np.array(g0_rows), np.array(g0_rhs)
    A1, b1 = np.array(rows), np.array(rhs)
    U_, sv, Vt = np.linalg.svd(A0, full_matrices=True)
    r = int(np.sum(sv > 1e-10 * sv[0]))
    x0 = Vt[:r].T @ ((U_[:, :r].T @ b0) / sv[:r])            # min-norm G0 solution
    Z = Vt[r:].T                                             # free directions (keep G0)
    if not Z.shape[1]:
        return x0.reshape(nu, nv, 3)
    def solve_free(A, rhs_, AZ=None):
        AZ = A @ Z if AZ is None else AZ
        return x0 + Z @ np.linalg.lstsq(AZ, rhs_ - A @ x0, rcond=None)[0]

    # the G2 rows with a neighbour curvature form, vectorised: row indices,
    # weights, stencils as a (rows, nu*nv) matrix on the CVs, and the 3x3 forms
    nl = [m for m in g2_meta if m[4] is not None]
    if nl:
        nl_ri = np.array([m[0] for m in nl])
        nl_w = np.array([m[1] for m in nl])
        nl_S = np.zeros((len(nl), nu * nv))
        for k_, m in enumerate(nl):
            for (i, j), c in m[3]:
                nl_S[k_, i * nv + j] += c
        nl_Q = np.array([m[4] for m in nl])

    def solve_all(A1, b1):
        x = solve_free(A1, b1)
        # G2 means n.S'' = II_neighbour(S', S') with S' the patch's ACTUAL cross
        # derivative: quadratic in the CVs. Gauss-Newton: linearise the right side
        # around the current solution, II(S',S') ~ 2 II(S'k, S') - II(S'k, S'k),
        # move it into the row and re-solve until it settles.
        # Each step is damped (halved until the true, nonlinear error drops): an
        # undamped step can run away when a neighbour's curvature in the patch's
        # oblique cross direction feeds back into the cross derivative.
        if nl:
            A1Z = A1 @ Z
            base_rows = A1[nl_ri]

            def objective(x_):
                r = A1 @ x_ - b1
                d1v = nl_S @ x_.reshape(-1, 3)                          # (rows, 3)
                quad = np.einsum("ki,kij,kj->k", d1v, nl_Q, d1v)
                r[nl_ri] = base_rows @ x_ - nl_w * quad
                return float(r @ r)
            f = objective(x)
            for _ in range(G2_PASSES):
                d1v = nl_S @ x.reshape(-1, 3)
                g = np.einsum("kij,kj->ki", nl_Q, d1v)                  # (rows, 3)
                A = A1.copy()
                A[nl_ri] = base_rows - (2.0 * nl_w)[:, None] * (nl_S[:, :, None] * g[:, None, :]).reshape(len(nl), -1)
                rhs_ = b1.copy()
                rhs_[nl_ri] = -nl_w * np.einsum("ki,ki->k", d1v, g)
                AZ = A1Z.copy()
                AZ[nl_ri] = A[nl_ri] @ Z
                dx = solve_free(A, rhs_, AZ) - x
                alpha, accepted = 1.0, False
                while alpha > 1e-3:
                    f_new = objective(x + alpha * dx)
                    if f_new < f:
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    break
                x = x + alpha * dx
                converged = f - f_new < 1e-10 * max(f, 1e-30)
                f = f_new
                if converged or np.abs(alpha * dx).max() < 1e-11 * (np.abs(x).max() or 1.0):
                    break
        return x

    # Conditions that the layout can't hold all at once (a G2 side whose two
    # ends are pinned by conflicting boundaries, say) are met in a single span
    # only by zigzagging CV rows. If the net comes out crossed or blown up,
    # raise the fairness until it's clean: a fair patch with a little less G2
    # beats an exact one with a broken net (Auto then adds spans if it helps).
    ext0 = float(np.ptp(np.vstack([sd.P for sd in sides]), axis=0).max()) or 1.0
    x = None
    for scale in FAIR_STEPS:
        A_, b_ = A1.copy(), b1.copy()
        if fair_rows and scale != 1.0:
            A_[fair_rows] *= scale
            b_[fair_rows] *= scale
        x = solve_all(A_, b_)
        X = x.reshape(nu, nv, 3)
        if net_folds(X) == 0 and float(np.ptp(x.reshape(-1, 3), axis=0).max()) <= NET_EXT_MAX * ext0:
            break
    return x.reshape(nu, nv, 3)


def _build_knots(sides, ks, weights, Uu, p, Uv, q, const_alpha=False):
    """One net on the given knot vectors. Returns (S grid, (Uu, Uv), score, info);
    score <= 1 means every side's rows survived within tolerance.
    const_alpha: constant cross scale per side, so the tangent / curvature rows
    stay in the neighbour's own spline space (CV rows aligned with it)."""
    bottom, right, top, left = sides
    kb, kr, kt, kl = ks
    wb, wr, wt, wl = weights
    nu, nv = len(Uu) - p - 1, len(Uv) - q - 1
    if nu < kl + kr + 2 or nv < kb + kt + 2:
        raise RuntimeError("too few CVs")
    Lv = float(np.mean(np.linalg.norm(bottom.P - top.P, axis=1))) or 1.0
    Lu = float(np.mean(np.linalg.norm(left.P - right.P, axis=1))) or 1.0

    # corner-compatible cross scales (exact reparameterisations of each neighbour):
    # the patch's cross derivative at a corner equals the adjacent boundary's derivative
    def scales(side, start_target, end_target, L):
        if not side.has_surface:
            return L
        a0, a1 = _corner_scale(side, 0, start_target, L), _corner_scale(side, -1, end_target, L)
        if const_alpha:
            nz = [a for a in (a0, a1) if a > 0]
            return float(np.mean(nz)) if nz else L
        return (a0, a1)
    Lb = scales(bottom, _end_deriv(left, 0), _end_deriv(right, 0), Lv)
    Lt = scales(top, -_end_deriv(left, 1), -_end_deriv(right, 1), Lv)
    Ll = scales(left, _end_deriv(bottom, 0), _end_deriv(top, 0), Lu)
    Lr = scales(right, -_end_deriv(bottom, 1), -_end_deriv(top, 1), Lu)
    allp = np.vstack([sd.P for sd in sides])
    size = float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0
    rb, rt = _side_rows(bottom, kb, wb, Lb, Uv, q), _side_rows(top, kt, wt, Lt, Uv, q)
    rl, rr = _side_rows(left, kl, wl, Ll, Uu, p), _side_rows(right, kr, wr, Lr, Uu, p)
    fb, res_b = _fit(rb, Uu, p, nu, bottom.t)
    ft, res_t = _fit(rt, Uu, p, nu, top.t)
    fl, res_l = _fit(rl, Uv, q, nv, left.t)
    fr, res_r = _fit(rr, Uv, q, nv, right.t)
    gu, gv = _greville(Uu, p, nu), _greville(Uv, q, nv)
    LV = _blend_grid(fb, ft, nv, gv).transpose(1, 0, 2)      # (nu, nv, 3)
    LU = _blend_grid(fl, fr, nu, gu)                         # (nu, nv, 3)
    T = _blend_grid([LV[i] for i in range(len(fl))], [LV[nu - 1 - i] for i in range(len(fr))], nu, gu)
    S = LU + LV - T
    # CVs inside a side's rows belong to that side (so boundaries and their
    # tangent / curvature rows are kept exactly); where two sides claim a CV the
    # closer side wins, true ties (twist CVs) are averaged
    nl, nr_, nb_, nt = len(fl), len(fr), len(fb), len(ft)
    for i in range(nu):
        di = i if i < nl else (nu - 1 - i if i >= nu - nr_ else None)
        for j in range(nv):
            dj = j if j < nb_ else (nv - 1 - j if j >= nv - nt else None)
            if di is None and dj is None:
                continue
            if dj is None or (di is not None and di < dj):
                S[i, j] = LU[i, j]
            elif di is None or dj < di:
                S[i, j] = LV[i, j]
            else:
                S[i, j] = 0.5 * (LU[i, j] + LV[i, j])

    if SOLVER == "lsq":
        # corner-exact (linearly varying) cross scales for the soft preferences
        lin = lambda side, a, b, L: (L if not side.has_surface else
                                     (_corner_scale(side, 0, a, L), _corner_scale(side, -1, b, L)))
        prefs = [lin(bottom, _end_deriv(left, 0), _end_deriv(right, 0), Lv),
                 lin(right, -_end_deriv(bottom, 1), -_end_deriv(top, 1), Lu),
                 lin(top, -_end_deriv(left, 1), -_end_deriv(right, 1), Lv),
                 lin(left, _end_deriv(bottom, 0), _end_deriv(top, 0), Lu)]
        S = _ls_solve(S, sides, ks, weights, prefs, Uu, p, Uv, q, size)

    # score: how well each side's rows survived (fit + ownership), vs tolerance
    def row_tol(rows, j):
        if j == 0:
            return 1e-4 * size
        step = float(np.mean(np.linalg.norm(rows[j] - rows[j - 1], axis=1))) or 1.0
        return (ANG_TOL if j == 1 else CUR_TOL) * step
    score = 0.0
    for rows, fitted, res, grid_rows in (
            (rb, fb, res_b, [S[:, j] for j in range(nb_)]), (rt, ft, res_t, [S[:, nv - 1 - j] for j in range(nt)]),
            (rl, fl, res_l, [S[i, :] for i in range(nl)]), (rr, fr, res_r, [S[nu - 1 - i, :] for i in range(nr_)])):
        for j in range(len(rows)):
            dev = res[j] + float(np.linalg.norm(grid_rows[j] - fitted[j], axis=1).max())
            score = max(score, dev / row_tol(rows, j))
    info = {"degree": (p, q), "cvs": (nu, nv), "spans": (len(set(Uu)) - 1, len(set(Uv)) - 1),
            "boundary_dev": float(max(res_b[0], res_t[0], res_l[0], res_r[0]))}
    return S, (Uu, Uv), score, info


def order_sides(raw):
    """raw: 4 Side objects in any order / direction -> (bottom, right, top, left,
    index map) with bottom/top along +u and left/right along +v."""
    idx = [0]
    chain = [raw[0]]
    rest = [1, 2, 3]
    while rest:
        end = chain[-1].P[-1]
        k = min(rest, key=lambda i: min(np.linalg.norm(raw[i].P[0] - end), np.linalg.norm(raw[i].P[-1] - end)))
        rest.remove(k)
        s = raw[k]
        chain.append(s if np.linalg.norm(s.P[0] - end) <= np.linalg.norm(s.P[-1] - end) else s.reversed())
        idx.append(k)
    c0, c1, c2, c3 = chain                    # loop: bottom, right, top (reversed), left (reversed)
    return [c0, c1, c2.reversed(), c3.reversed()], idx


@features.builder("square")
def build_square(inputs, p):
    if len(inputs) not in (3, 4):
        raise RuntimeError("Square needs 4 boundaries (or 3: one corner collapses to a point)")
    m = int(p.get("samples", 48))
    raw = [sample_side(o, r, m) for o, r in inputs]
    for sd, (o, r) in zip(raw, inputs):
        sd.src = (o.name, r)
    if len(raw) == 3:
        # chain the three, then a degenerate 4th side (a point) closes the loop
        chain, idx3 = [raw[0]], [0]
        rest = [1, 2]
        while rest:
            end = chain[-1].P[-1]
            k = min(rest, key=lambda i: min(np.linalg.norm(raw[i].P[0] - end), np.linalg.norm(raw[i].P[-1] - end)))
            rest.remove(k)
            sd = raw[k]
            chain.append(sd if np.linalg.norm(sd.P[0] - end) <= np.linalg.norm(sd.P[-1] - end) else sd.reversed())
            idx3.append(k)
        pt = chain[0].P[0]
        raw = chain + [Side(np.repeat(pt[None, :], m, axis=0))]
        p = dict(p)
        p["e3"], p["w3"] = "G0", 1.0
        p = {**p, **{f"e{j}": p.get(f"e{i}", "G0") for j, i in enumerate(idx3)},
             **{f"w{j}": p.get(f"w{i}", 1.0) for j, i in enumerate(idx3)}}
    sides, idx = order_sides(raw)
    conts = [p.get(f"e{i}", "G0") for i in idx]
    weights = [float(p.get(f"w{i}", 1.0)) for i in idx]
    d_old = int(p.get("square_degree", 0) or 0)          # older files: one degree for both
    deg = (int(p.get("sq_degree_u", d_old) or 0), int(p.get("sq_degree_v", d_old) or 0))
    surf, info = square_surface(sides, conts, weights, deg,
                                int(p.get("sq_spans_u", 0)), int(p.get("sq_spans_v", 0)),
                                match=bool(p.get("sq_match", True)), retrim=bool(p.get("sq_retrim", False)),
                                max_move=float(p.get("sq_move", 5.0)) / 100.0)
    surf = _orient_like_neighbours(surf, sides)
    info["corner_notes"] = list(LAST_NOTES)
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return surf


def _orient_like_neighbours(surf, sides):
    """Reverse u if the patch's normal disagrees with the neighbouring faces'
    outward normals (so shading / matcaps match across the edges)."""
    _, ext = _k()
    vote = 0.0
    for sd, at in zip(sides, (lambda t: (t, 0.0), lambda t: (1.0, t), lambda t: (t, 1.0), lambda t: (0.0, t))):
        if sd.Nout is None:
            continue
        for i in np.linspace(1, len(sd.P) - 2, 7).astype(int):
            u, v = at(sd.t[i])
            _, su, sv = surf.derivatives(u, v)
            n = np.cross(su, sv)
            vote += float(n @ sd.Nout[i]) / (np.linalg.norm(n) or 1.0)
    if vote >= 0:
        return surf
    return reverse_u(surf)


def reverse_u(surf):
    """The same surface with u reversed: the normal turns over; U0 and U1 swap."""
    _, ext = _k()
    d = ext.surface_cvs(surf)
    nu, nv = d["count_u"], d["count_v"]
    P = np.array(d["points"], float).reshape(nu, nv, 3)[::-1]
    W = np.array(d["weights"], float).reshape(nu, nv)[::-1] if d.get("weights") else None
    K = list(d["knots_u"])
    Ku = [K[0] + K[-1] - k for k in reversed(K)]
    pts = [tuple(map(float, P[i, j])) for i in range(nu) for j in range(nv)]
    return ext.surface_from_cvs(d["degree_u"], d["degree_v"], pts, Ku, list(d["knots_v"]),
                                [float(x) for x in W.reshape(-1)] if W is not None else None)


LAST_INFO = {}


def store_info(obj):
    """Keep the last build's quality report on the object (Feature panel)."""
    import json
    def conv(v):
        if isinstance(v, (tuple, list)):
            return [conv(x) for x in v]
        if isinstance(v, str):
            return v
        return float(v)
    obj["nproto_square_info"] = json.dumps({k: conv(v) for k, v in LAST_INFO.items() if k != "clean_edges"})


# --------------------------------------------------------------------------
# Re-trim a planar target face to a clean square edge
# --------------------------------------------------------------------------

def is_planar_target(name):
    import bpy
    obj = bpy.data.objects.get(name)
    if obj is None or not store.is_nurbs(obj):
        return False
    brep = store.get_brep(obj)
    if brep.face_count() != 1:
        return False
    _, ext = _k()
    try:
        d = ext.surface_cvs(brep.faces()[0].surface())
    except Exception:
        return False
    P = np.array(d["points"], float)
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    return np.linalg.svd(P - P.mean(axis=0), compute_uv=False)[-1] < 1e-7 * size


def _edge_curve(edge):
    """Exact kernel curve of a B-rep edge over its own parameter range."""
    from . import extend
    sm, ext = _k()
    t0, t1 = edge.parameter_range()
    try:
        d = ext.curve_cvs(edge.curve())
    except Exception:
        d = None
    if d is not None:
        p, U = d["degree"], list(d["knots"])
        P = np.array(d["points"], float)
        w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
        Pw = np.hstack([P * w[:, None], w[:, None]])
        q0 = np.array(edge.point_at(t0))
        if U[-1] - t1 > 1e-12:
            Q, U = extend.end_curve(list(Pw), U, p, t1)
            Pw = np.array(Q)
        if t0 - U[0] > 1e-12:
            a, b = U[0], U[-1]
            Q, Ur = extend.end_curve(list(Pw[::-1]), [a + b - x for x in reversed(U)], p, a + b - t0)
            Pw, U = np.array(Q)[::-1], [a + b - x for x in reversed(Ur)]
        pts = [tuple(map(float, x[:3] / x[3])) for x in Pw]
        wts = [float(x[3]) for x in Pw]
        rational = max(abs(x - 1.0) for x in wts) > 1e-12
        c = ext.curve_from_cvs(p, pts, U, wts if rational else None)
        if np.linalg.norm(np.array(c.evaluate(U[0])) - q0) < 1e-6 * (1 + np.linalg.norm(q0)):
            return c
    pts = [tuple(edge.point_at(t0 + (t1 - t0) * i / 40)) for i in range(41)]
    return sm.create_curve_interp_points(pts)


def retrim_planar(obj, ref, cvs_world, knots, degree):
    """Rebuild the planar face of `obj` from its own boundary edges (exact
    sub-curves) with edge `ref` replaced by the clean curve. Returns the new ref."""
    sm, ext = _k()
    brep = store.get_brep(obj)
    face = brep.faces()[0]
    Md = np.array(geom.world_matrix(obj), dtype=float)
    Mi = np.linalg.inv(Md)
    loc = [tuple(map(float, Mi[:3, :3] @ np.asarray(x) + Mi[:3, 3])) for x in cvs_world]
    clean = ext.curve_from_cvs(int(degree), loc, list(map(float, knots)), None)
    target = brep.edges()[int(ref[1:])]
    tq = np.array(target.point_at(sum(target.parameter_range()) / 2))
    curves = []
    for e in face.edges():
        em = np.array(e.point_at(sum(e.parameter_range()) / 2))
        if np.linalg.norm(em - tq) < 1e-9 * (1 + np.linalg.norm(tq)):
            curves.append(clean)
        else:
            curves.append(_edge_curve(e))
    new = sm.create_planar_faces(curves)
    store.set_geometry(obj, new)
    # the new edge that is the clean curve
    cm = np.array(clean.evaluate(0.5 * (knots[0] + knots[-1])))
    b2 = store.get_brep(obj)
    best = min(range(b2.edge_count()), key=lambda i: np.linalg.norm(
        np.array(b2.edges()[i].point_at(sum(b2.edges()[i].parameter_range()) / 2)) - cm))
    return f"E{best}"
