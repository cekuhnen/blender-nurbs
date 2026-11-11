"""Sweeps, pipes, boundary surfaces and curve-extraction features.

Sweep construction: the profile's CVs are moved RIGIDLY with a frame along
the rail, so every cross-section is an exact copy of the profile (rational
profiles included); only the rail direction is fitted (shared-knot least
squares, like the blend loft). Surface = u: along the rail, v: profile.
"""

import math

import numpy as np
from mathutils import Vector

from . import features, geom, store
from .kernel import load as _load_kernel


def _k():
    sm = _load_kernel()
    return sm, sm.blender_ext


# --------------------------------------------------------------------------
# Rails
# --------------------------------------------------------------------------

def _arc_samples(kcurve, n):
    """n points equally spaced by arc length, with unit tangents."""
    sm, _ = _k()
    t0, t1 = kcurve.parameter_range()
    ts = np.linspace(t0, t1, 400)
    P = np.array([kcurve.evaluate(t) for t in ts])
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    targets = np.linspace(0, L[-1], n)
    tt = np.interp(targets, L, ts)
    pts, tan = [], []
    for t in tt:
        d = sm.evaluate_curve(kcurve, float(t))
        pts.append(d["point"])
        v = np.array(d["tangent"], float)
        tan.append(v / (np.linalg.norm(v) or 1.0))
    return np.array(pts, float), np.array(tan, float)


def _rmf(P, T, r0):
    """Rotation-minimizing frames (double reflection, Wang et al. 2008)."""
    R = [r0 / np.linalg.norm(r0)]
    for i in range(len(P) - 1):
        v1 = P[i + 1] - P[i]
        c1 = v1 @ v1
        if c1 < 1e-24:
            R.append(R[-1]); continue
        rL = R[-1] - (2 / c1) * (v1 @ R[-1]) * v1
        tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        r = rL if c2 < 1e-24 else rL - (2 / c2) * (v2 @ rL) * v2
        r = r - (r @ T[i + 1]) * T[i + 1]
        R.append(r / (np.linalg.norm(r) or 1.0))
    return np.array(R)


def _perp(t, hint):
    v = hint - (hint @ t) * t
    if np.linalg.norm(v) < 1e-9:
        a = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
        v = a - (a @ t) * t
    return v / np.linalg.norm(v)


def _surface_edge_frames(obj, ref, n):
    """Rail along a NURBS surface edge: points, tangents, outward in-plane
    direction B and surface normal N, by arc length."""
    sm, _ = _k()
    kc = features.input_curve(obj, ref)
    P, T = _arc_samples(kc, n)
    M = geom.world_matrix(obj)
    Mi = M.inverted()
    R3 = np.array(M.to_3x3())
    brep = store.get_brep(obj)
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    eps = 2e-3 * size
    B, N = [], []
    for p, t in zip(P, T):
        pl = np.array(Mi @ Vector(p))
        face = min(brep.faces(), key=lambda f: f.closest_point(tuple(pl))[1])
        surf = face.surface()
        # surface normal at the edge point: Newton-free via two nearby surface points
        q, _ = face.closest_point(tuple(pl))
        # find uv by sampling the surface domain near q (coarse grid + refine)
        nrm = _normal_near(surf, np.array(q))
        nw = R3 @ nrm
        nw /= np.linalg.norm(nw) or 1.0
        b = np.cross(nw, t)
        b /= np.linalg.norm(b) or 1.0
        # outward = the side where we leave the face (closest distance grows)
        dp = face.closest_point(tuple(np.array(Mi @ Vector(p + eps * b))))[1]
        dm = face.closest_point(tuple(np.array(Mi @ Vector(p - eps * b))))[1]
        if dm > dp:
            b = -b
        B.append(b)
        N.append(np.cross(t, b))
    return P, T, np.array(B), np.array(N)


def _normal_near(surf, q):
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    best = None
    for i in range(17):
        for j in range(17):
            u, v = u0 + (u1 - u0) * i / 16, v0 + (v1 - v0) * j / 16
            d = np.linalg.norm(np.array(surf.evaluate(u, v)) - q)
            if best is None or d < best[0]:
                best = (d, u, v)
    _, u, v = best
    du, dv = (u1 - u0) / 16, (v1 - v0) / 16
    for _ in range(30):   # Gauss-Newton refine of the foot point
        pt, su, sv = surf.derivatives(u, v)
        r = np.array(pt) - q
        J = np.array([su, sv]).T
        try:
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
        except Exception:
            break
        u = min(max(u + step[0], u0), u1)
        v = min(max(v + step[1], v0), v1)
        if np.linalg.norm(step) < 1e-12:
            break
    n = np.array(surf.normal(u, v), float)
    return n / (np.linalg.norm(n) or 1.0)


# --------------------------------------------------------------------------
# Surface from cross-section rows
# --------------------------------------------------------------------------

def _surface_from_rows(rows, prof, tol_rel=1e-4):
    """rows: K arrays (nv,3) of profile CVs at each rail sample. Fit along the
    rail per CV (shared knots), keep the profile's v knots / weights."""
    _, ext = _k()
    rows = np.array(rows)                 # (K, nv, 3)
    K, nv, _ = rows.shape
    cols = [rows[:, j, :] for j in range(nv)]
    size = float(np.linalg.norm(np.ptp(rows.reshape(-1, 3), axis=0))) or 1.0
    cv_cols, U, deg_u = geom.fit_rows(cols, 3, tol=tol_rel * size)
    nu = len(cv_cols[0])
    pts, wts = [], []
    w = prof["weights"] or [1.0] * nv
    for iu in range(nu):
        for j in range(nv):
            pts.append(tuple(map(float, cv_cols[j][iu])))
            wts.append(float(w[j]))
    rational = any(abs(x - 1.0) > 1e-12 for x in wts)
    return ext.surface_from_cvs(deg_u, prof["degree"], pts, [float(x) for x in U], prof["knots"],
                                wts if rational else None)


def _profile(kc):
    _, ext = _k()
    d = ext.curve_cvs(kc)
    return {"degree": d["degree"], "points": np.array(d["points"], float), "weights": list(d["weights"]),
            "knots": list(d["knots"])}


# --------------------------------------------------------------------------
# Sweep features
# --------------------------------------------------------------------------

@features.builder("sweep")
def build_sweep(inputs, p):
    """inputs: profile, rail [, rail2]. Rails may be curves or surface edges."""
    n = int(p.get("samples", 32))
    prof_obj, prof_ref = inputs[0]
    prof_k = features.input_curve(prof_obj, prof_ref)
    prof = _profile(prof_k)
    Q = prof["points"]
    if len(inputs) >= 3:
        return _sweep_two_rails(inputs, p, prof, n)
    rail_obj, rail_ref = inputs[1]
    tangent = bool(p.get("tangent", True)) and (features.is_edge_ref(rail_ref) or features.is_brep_edge_ref(rail_ref))
    psize = float(np.linalg.norm(np.ptp(Q, axis=0))) or 1.0
    ends = [np.array(prof_k.start_point()), np.array(prof_k.end_point())]
    if tangent:
        P, T, B, N = _surface_edge_frames(rail_obj, rail_ref, n)
    else:
        P, T = _arc_samples(features.input_curve(rail_obj, rail_ref), n)
    # a profile standing on the rail's END sweeps from there (the rail is run backwards)
    d0 = min(np.linalg.norm(x - P[0]) for x in ends)
    d1 = min(np.linalg.norm(x - P[-1]) for x in ends)
    if d1 < d0 and d1 <= 0.1 * psize:
        P, T = P[::-1].copy(), -T[::-1]
        if tangent:
            B = B[::-1].copy()
            N = np.cross(T, B)
    if not tangent:
        hint = (Q.mean(axis=0) - P[0])
        N0 = _perp(T[0], hint if np.linalg.norm(hint) > 1e-9 else np.array([0, 0, 1.0]))
        N = _rmf(P, T, N0)
        B = np.cross(T, N)
    # an OPEN profile touching the rail is attached there by its nearest end
    # (exact G0); a closed profile keeps its position relative to the rail
    e = 0 if np.linalg.norm(ends[0] - P[0]) <= np.linalg.norm(ends[1] - P[0]) else 1
    closed = np.linalg.norm(ends[0] - ends[1]) <= 1e-6 * psize
    attach = tangent or (not closed and np.linalg.norm(ends[e] - P[0]) <= 0.1 * psize)
    F0 = np.array([T[0], N[0], B[0]])      # rows = axes
    q = (Q - (ends[e] if attach else P[0])) @ F0.T
    if tangent:
        # rotate the profile about T so its start tangent lies in the surface's
        # tangent plane, pointing off the edge -> G1 to the surface along the rail
        sm, _ = _k()
        t0, t1 = prof_k.parameter_range()
        tp = np.array(sm.evaluate_curve(prof_k, t0 if e == 0 else t1)["tangent"], float) * (1 if e == 0 else -1)
        lt = F0 @ tp                        # (t, n, b) components
        phi = math.atan2(lt[1], lt[2])      # angle of the tangent in the (n, b) plane, from +b
        if p.get("flip_tangent"):
            phi += math.pi
        c, s = math.cos(phi), math.sin(phi)     # (sin phi, cos phi) -> (0, 1): tangent onto +b
        qn, qb = q[:, 1].copy(), q[:, 2].copy()
        q[:, 1] = c * qn - s * qb
        q[:, 2] = s * qn + c * qb
    scale_end = float(p.get("scale_end", 1.0))
    twist = math.radians(float(p.get("twist", 0.0))) if not tangent else 0.0
    rows = []
    for k in range(n):
        f = k / (n - 1)
        sc = 1.0 + (scale_end - 1.0) * f
        th = twist * f
        c, s = math.cos(th), math.sin(th)
        nk = c * N[k] + s * B[k]
        bk = -s * N[k] + c * B[k]
        rows.append(P[k] + sc * (np.outer(q[:, 0], T[k]) + np.outer(q[:, 1], nk) + np.outer(q[:, 2], bk)))
    return _surface_from_rows(rows, prof)


def _sweep_two_rails(inputs, p, prof, n):
    (r1o, r1r), (r2o, r2r) = inputs[1], inputs[2]
    P1, T1 = _arc_samples(features.input_curve(r1o, r1r), n)
    P2, T2 = _arc_samples(features.input_curve(r2o, r2r), n)
    if np.linalg.norm(P1[0] - P2[0]) + np.linalg.norm(P1[-1] - P2[-1]) > \
            np.linalg.norm(P1[0] - P2[-1]) + np.linalg.norm(P1[-1] - P2[0]):
        P2, T2 = P2[::-1].copy(), -T2[::-1]       # rails run the same way
    Q = prof["points"]

    def frame(k):
        x = P2[k] - P1[k]
        L = float(np.linalg.norm(x)) or 1.0
        x = x / L
        z = _perp(x, (T1[k] + T2[k]) / 2)
        y = np.cross(z, x)
        return x, y, z, L

    x0, y0, z0, L0 = frame(0)
    q = (Q - P1[0]) @ np.array([x0, y0, z0]).T / L0
    keep_height = bool(p.get("keep_height", False))
    rows = []
    for k in range(n):
        x, y, z, L = frame(k)
        h = (L0 / L) if keep_height else 1.0
        rows.append(P1[k] + L * (np.outer(q[:, 0], x) + h * np.outer(q[:, 1], y) + h * np.outer(q[:, 2], z)))
    return _surface_from_rows(rows, prof)


@features.builder("pipe")
def build_pipe(inputs, p):
    from .primitives import exact_circle_spec, spec_to_kernel_curve
    sm, ext = _k()
    rail_k = features.input_curve(*inputs[0])
    n = int(p.get("samples", 32))
    P, T = _arc_samples(rail_k, n)
    N = _rmf(P, T, _perp(T[0], np.array([0, 0, 1.0])))
    B = np.cross(T, N)
    r = float(p.get("radius", 0.2))
    prof = _profile(spec_to_kernel_curve(exact_circle_spec(r)))
    q = prof["points"]                       # circle in its local XY plane, centre origin
    rows = [P[k] + np.outer(q[:, 0], N[k]) + np.outer(q[:, 1], B[k]) for k in range(n)]
    return _surface_from_rows(rows, prof)


# --------------------------------------------------------------------------
# Boundary (4-sided Coons) surface
# --------------------------------------------------------------------------

boundary_info = {}      # how the last Boundary was built (operator / panel report)


def _boundary_chain(inputs):
    """The 4 inputs as bottom, right, top, left: [(index, reversed)] so that
    bottom and top run in +u, left and right in +v (head to tail around the loop)."""
    ends = []
    for o, r in inputs:
        kc = features.input_curve(o, r)
        t0, t1 = kc.parameter_range()
        ends.append((np.array(kc.evaluate(t0)), np.array(kc.evaluate(t1))))
    chain = [(0, False)]
    rest = [1, 2, 3]
    while rest:
        k0, rv = chain[-1]
        end = ends[k0][0] if rv else ends[k0][1]
        k = min(rest, key=lambda i: min(np.linalg.norm(ends[i][0] - end), np.linalg.norm(ends[i][1] - end)))
        rest.remove(k)
        chain.append((k, np.linalg.norm(ends[k][1] - end) < np.linalg.norm(ends[k][0] - end)))
    (b, rb), (r, rr), (t, rt), (l, rl) = chain
    return [(b, rb), (r, rr), (t, not rt), (l, not rl)]


def _knot_mults(U, p):
    """{interior knot: multiplicity} of a clamped knot vector."""
    out = {}
    for k in U[p + 1:len(U) - p - 1]:
        k = round(float(k), 12)
        out[k] = out.get(k, 0) + 1
    return out


def _common_space(sides, p_req):
    """Degree and knot vector holding both curves exactly (degree raised to the
    larger / requested one, knots merged). None if p_req is below their degree."""
    p = max(sd["deg"] for sd in sides)
    if p_req:
        if p_req < p:
            return None
        p = p_req
    mult = {}
    for sd in sides:
        for k, m in _knot_mults(sd["knots"], sd["deg"]).items():
            mult[k] = max(mult.get(k, 0), min(p, m + p - sd["deg"]))
    U = [0.0] * (p + 1)
    for k in sorted(mult):
        U += [k] * mult[k]
    return p, U + [1.0] * (p + 1)


def _refit_exact(sd, p, U):
    """CVs of the curve sd in the (larger) space (p, U); None if not exact."""
    from .salign import basis_matrix
    n = len(U) - p - 1
    ts = np.linspace(0.0, 1.0, max(60, 12 * n))
    Q = np.array([geom.eval_bspline(sd["P"], None, sd["knots"], sd["deg"], t) for t in ts])
    A = basis_matrix(U, p, n, ts)
    X = np.linalg.lstsq(A, Q, rcond=None)[0]
    size = float(np.linalg.norm(np.ptp(Q, axis=0))) or 1.0
    if np.abs(A @ X - Q).max() > 1e-9 * size:
        return None
    X[0], X[-1] = Q[0], Q[-1]
    return X


def _greville(U, p):
    n = len(U) - p - 1
    return np.array([sum(U[i + 1:i + p + 1]) / p for i in range(n)])


def _boundary_exact(inputs, chain, p):
    """Exact bilinear Coons patch in CV form, or the reason it is not possible
    (rational input, trimmed edge, degree / spans asked below what the curves need)."""
    sides = []
    for k, rv in chain:
        o, ref = inputs[k]
        sd = features.exact_side(o, ref)
        if sd is None:
            why = "is a trimmed edge" if features.is_brep_edge_ref(ref) else "is rational (arc / circle)"
            return f"{o.name} {ref if o.type != 'CURVE' else ''}".strip() + " " + why
        if rv:
            sd = dict(sd, P=np.asarray(sd["P"])[::-1].copy(), knots=[1.0 - x for x in reversed(sd["knots"])])
        sides.append(sd)
    bot, right, top, left = sides
    out = []
    for pair, deg_req, spans_req, d in (((bot, top), p["du"], p["su"], "U"), ((left, right), p["dv"], p["sv"], "V")):
        sp = _common_space(pair, deg_req)
        if sp is None:
            return f"Degree {d} {deg_req} is below the curves' degree {max(sd['deg'] for sd in pair)}"
        deg, U = sp
        if spans_req and len(set(U)) - 1 != spans_req:
            return f"Spans {d} {spans_req}: the curves have {len(set(U)) - 1}"
        rows = [_refit_exact(sd, deg, U) for sd in pair]
        if any(r is None for r in rows):
            return "the curves could not be matched exactly"
        out.append((deg, U, rows))
    (du, Uu, (B, T)), (dv, Uv, (L, R)) = out
    gu, gv = _greville(Uu, du)[:, None, None], _greville(Uv, dv)[None, :, None]
    c00, c10, c01, c11 = B[0], B[-1], T[0], T[-1]
    S = ((1 - gv) * B[:, None] + gv * T[:, None] + (1 - gu) * L[None] + gu * R[None]
         - ((1 - gu) * (1 - gv) * c00 + gu * (1 - gv) * c10 + (1 - gu) * gv * c01 + gu * gv * c11))
    return du, dv, Uu, Uv, S


def _fit_rows_n(rows, degree, n_cvs):
    """geom.fit_rows with a fixed CV count (shared chord parameters)."""
    N = len(rows[0])
    p = min(degree, N - 1)
    acc = np.zeros(N)
    for Rw in rows:
        d = np.linalg.norm(np.diff(Rw, axis=0), axis=1)
        if d.sum() > 0:
            acc[1:] += np.cumsum(d) / d.sum()
    params = acc / len(rows)
    params[0], params[-1] = 0.0, 1.0
    n_cvs = max(p + 1, min(n_cvs, N))
    U = geom._ls_knots(params, n_cvs, p)
    X = geom._ls_fit(params, np.hstack(rows), U, p, n_cvs)
    return [X[:, 3 * i:3 * i + 3] for i in range(len(rows))], U, p


@features.builder("boundary")
def build_boundary(inputs, p):
    """Four curves / edges forming a loop -> a bilinearly blended Coons patch.
    Exact (the edges ARE the inputs) when the inputs are polynomial B-splines and
    the asked degree / spans allow it; otherwise sampled and fitted."""
    _, ext = _k()
    m = int(p.get("samples", 24))
    req = {"du": int(p.get("sq_degree_u", 0)), "dv": int(p.get("sq_degree_v", 0)),
           "su": int(p.get("sq_spans_u", 0)), "sv": int(p.get("sq_spans_v", 0))}
    chain = _boundary_chain(inputs)
    ex, why = None, None
    if p.get("bd_exact", False):          # older Boundaries (no setting) stay fitted
        try:
            ex = _boundary_exact(inputs, chain, req)
        except Exception as exc:
            ex = f"exact build failed ({exc})"
        if isinstance(ex, str):
            ex, why = None, ex
    if ex is not None:
        du, dv, Uu, Uv, S = ex
        nu, nv = S.shape[:2]
        boundary_info.clear()
        boundary_info.update(exact=True, cvs=(nu, nv), degree=(du, dv))
        pts = [tuple(map(float, S[i, j])) for i in range(nu) for j in range(nv)]
        return ext.surface_from_cvs(du, dv, pts, [float(x) for x in Uu], [float(x) for x in Uv])

    curves = []
    for k, rv in chain:
        kc = features.input_curve(*inputs[k])
        t0, t1 = kc.parameter_range()
        c = np.array([kc.evaluate(t0 + (t1 - t0) * i / (m - 1)) for i in range(m)])
        curves.append(c[::-1] if rv else c)
    bottom, right, top, left = curves
    S = np.zeros((m, m, 3))
    for i in range(m):
        u = i / (m - 1)
        for j in range(m):
            v = j / (m - 1)
            S[i, j] = ((1 - v) * bottom[i] + v * top[i] + (1 - u) * left[j] + u * right[j]
                       - ((1 - u) * (1 - v) * bottom[0] + u * (1 - v) * bottom[-1]
                          + (1 - u) * v * top[0] + u * v * top[-1]))
    size = float(np.linalg.norm(np.ptp(S.reshape(-1, 3), axis=0))) or 1.0
    tol = 1e-4 * size
    deg_u, deg_v = req["du"] or 3, req["dv"] or 3

    def fit(rows, deg, spans):
        if spans:
            return _fit_rows_n(rows, deg, spans + deg)
        return geom.fit_rows(rows, deg, tol=tol)
    # fit along v for each u row, then along u for each resulting CV column
    vcv, Uv, dv = fit([S[i] for i in range(m)], deg_v, req["sv"])
    nv = len(vcv[0])
    ucv, Uu, du = fit([np.array([vcv[i][j] for i in range(m)]) for j in range(nv)], deg_u, req["su"])
    nu = len(ucv[0])
    boundary_info.clear()
    boundary_info.update(exact=False, cvs=(nu, nv), degree=(du, dv), why=why)
    pts = [tuple(map(float, ucv[j][i])) for i in range(nu) for j in range(nv)]
    return ext.surface_from_cvs(du, dv, pts, [float(x) for x in Uu], [float(x) for x in Uv])


# --------------------------------------------------------------------------
# Curves from surfaces: iso curve, duplicate edge, project, intersect
# --------------------------------------------------------------------------

def _poly_to_spec(P, deg=3, closed=None):
    from .primitives import _fit_open_spec
    P = np.array(P, float)
    span = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    n = max(deg + 1, min(40, int(len(P) / 3)))
    best = None
    for cvs in range(deg + 1, n + 1):       # fewest CVs within tolerance
        sp = _fit_open_spec([tuple(x) for x in P], deg, cvs)
        d, Pc, W, U = geom.blender_nurbs_to_clamped(sp["points"], sp["weights"], sp["order"], False, True, False)
        Q = np.array([geom.eval_bspline(Pc, W, U, d, t) for t in np.linspace(0, 1, 120)])
        err = max(np.min(np.linalg.norm(Q - x, axis=1)) for x in P[:: max(1, len(P) // 60)])
        best = sp
        if err <= 1e-4 * span:
            break
    return best


def _kcurve_poly(kc, n=200):
    t0, t1 = kc.parameter_range()
    return [kc.evaluate(t0 + (t1 - t0) * i / (n - 1)) for i in range(n)]


@features.builder("iso_curve")
def build_iso_curve(inputs, p):
    obj = inputs[0][0]
    surf = store.get_surfaces(obj)[0]
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    t = float(p.get("param", 0.5))
    M = geom.world_matrix(obj)
    if p.get("iso_dir", "U") == "U":
        u = u0 + (u1 - u0) * t
        pts = [surf.evaluate(u, v0 + (v1 - v0) * i / 199) for i in range(200)]
    else:
        v = v0 + (v1 - v0) * t
        pts = [surf.evaluate(u0 + (u1 - u0) * i / 199, v) for i in range(200)]
    return ("splines", [_poly_to_spec([tuple(M @ Vector(x)) for x in pts])])


@features.builder("dup_edge")
def build_dup_edge(inputs, p):
    return ("splines", [_poly_to_spec(_kcurve_poly(features.input_curve(o, r))) for o, r in inputs])


@features.builder("project_curve")
def build_project_curve(inputs, p):
    sm, _ = _k()
    *curves, (surf_obj, _) = inputs
    d = p.get("dir", [0, 0, -1])
    brep_w = _world_brep(surf_obj)
    out = []
    for o, r in curves:
        for pc in sm.project_curve(brep_w, features.input_curve(o, r), tuple(d)):
            out.append(_poly_to_spec(_kcurve_poly(pc)))
    if not out:
        raise RuntimeError("the curve does not project onto the surface in that direction")
    return ("splines", out)


@features.builder("intersect")
def build_intersect(inputs, p):
    sm, _ = _k()
    (a, _), (b, _) = inputs[0], inputs[1]
    curves, _ = sm.intersect_breps(_world_brep(a), _world_brep(b))
    if not curves:
        raise RuntimeError("the surfaces do not intersect")
    return ("splines", [_poly_to_spec(_kcurve_poly(c)) for c in curves])


def _world_brep(obj):
    """A world-space copy of an object's brep (in-process: read-only kernel ops)."""
    sm, _ = _k()
    b = store.brep_from_text(obj[store.PROP_BREP])
    M = geom.world_matrix(obj)
    loc, rot, scale = M.decompose()
    if any(abs(s - 1.0) > 1e-12 for s in scale):
        b = sm.scale(b, tuple(scale))
    axis, ang = rot.to_axis_angle()
    if abs(ang) > 1e-12:
        b = sm.rotate(b, (0, 0, 0), tuple(axis), math.degrees(ang))
    if loc.length > 1e-12:
        b = sm.translate(b, tuple(loc))
    return b


# --------------------------------------------------------------------------
# Rebuild surface (history step, in-process)
# --------------------------------------------------------------------------

def _avg_chord(G, axis):
    """Normalized cumulative chord length along `axis` of the sample grid,
    averaged over all rows (degenerate rows such as a sphere pole are skipped)."""
    R = G if axis == 0 else G.transpose(1, 0, 2)
    n = R.shape[0]
    acc, cnt = np.zeros(n), 0
    for j in range(R.shape[1]):
        d = np.linalg.norm(np.diff(R[:, j], axis=0), axis=1)
        tot = d.sum()
        if tot > 1e-12:
            acc[1:] += np.cumsum(d) / tot
            cnt += 1
    if cnt == 0:
        return np.linspace(0, 1, n)
    t = acc / cnt
    t[0], t[-1] = 0.0, 1.0
    return t


def rebuild_surface_text(text, du, dv, su, sv):
    """Refit an untrimmed surface with the given degrees / spans (uniform knots)."""
    _, ext = _k()
    surf = store.brep_from_text(text).faces()[0].surface()
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    nu, nv = du + su, dv + sv
    mu, mv = max(4 * nu, 24), max(4 * nv, 24)
    mu, mv = max(mu, 40), max(mv, 40)
    us, vs = np.linspace(0, 1, mu), np.linspace(0, 1, mv)
    G = np.array([[surf.evaluate(u0 + (u1 - u0) * a, v0 + (v1 - v0) * b) for b in vs] for a in us])
    # fit against averaged chord-length parameters, not the old surface's own
    # parameters: a surface whose parameter speed varies (e.g. a plane with evenly
    # spaced cubic CVs) would otherwise bunch the new CVs together
    tu, tv = _avg_chord(G, 0), _avg_chord(G, 1)
    Uu, Uv = geom.clamped_uniform_knots(nu, du), geom.clamped_uniform_knots(nv, dv)
    Av, Au = geom.interp_matrix(tv, Uv, dv, nv), geom.interp_matrix(tu, Uu, du, nu)
    # least squares in v for every u row, then in u for every CV column
    Xv = np.array([np.linalg.lstsq(Av, G[i], rcond=None)[0] for i in range(mu)])      # (mu, nv, 3)
    X = np.array([np.linalg.lstsq(Au, Xv[:, j, :], rcond=None)[0] for j in range(nv)])  # (nv, nu, 3)
    pts = [tuple(map(float, X[j, i])) for i in range(nu) for j in range(nv)]
    s = ext.surface_from_cvs(du, dv, pts, Uu, Uv)
    return store.brep_to_text(ext.brep_from_surface(s))


def apply_rebuild_surface(obj, text, inputs, p):
    return rebuild_surface_text(text, int(p["degree_u"]), int(p["degree_v"]), int(p["spans_u"]), int(p["spans_v"]))


def _register_step():
    from . import steps
    steps.APPLY["rebuild_surface"] = apply_rebuild_surface
    steps.LABEL["rebuild_surface"] = lambda p: (f"Rebuild · degree {p['degree_u']}x{p['degree_v']}, "
                                                f"spans {p['spans_u']}x{p['spans_v']}")


_register_step()
