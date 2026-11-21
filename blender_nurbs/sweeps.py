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
    # the kernel's global foot: a seeded Newton started on the wrong side of a
    # closed surface's seam (v = 0 for a point at v = 0.97) stops at the
    # domain's edge, centimetres away -- the Continuity Check then showed a
    # 9.8 deg kink on a fillet that met a cylinder within 0.003 deg
    try:
        sm, _ = _k()
        _, uvs, _ = sm.surface_closest_point(surf, tuple(float(x) for x in q))
        uv = uvs[0]
        u, v = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
        n = np.array(surf.normal(u, v), float)
        return n / (np.linalg.norm(n) or 1.0)
    except Exception:
        pass
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

def _surface_from_rows(rows, prof, tol_rel=1e-4, end_dirs=None):
    """rows: K arrays (nv,3) of profile CVs at each rail sample. Fit along the
    rail per CV (shared knots), keep the profile's v knots / weights.
    end_dirs: (d0, d1) unit directions the surface leaves its first / last row
    along (the second CV of every column moved onto that line): what a fit
    alone only comes close to, so a part tangent to them there is G1. Either
    may be None (that end as fitted), or (nv, 3): a direction per column."""
    _, ext = _k()
    rows = np.array(rows)                 # (K, nv, 3)
    K, nv, _ = rows.shape
    cols = [rows[:, j, :] for j in range(nv)]
    size = float(np.linalg.norm(np.ptp(rows.reshape(-1, 3), axis=0))) or 1.0
    cv_cols, U, deg_u = geom.fit_rows(cols, 3, tol=tol_rel * size)
    nu = len(cv_cols[0])
    if end_dirs is not None and nu >= 4:
        for jc, c in enumerate(cv_cols):
            j_col = [jc]
            for a, b, d in ((0, 1, end_dirs[0]), (nu - 1, nu - 2, end_dirs[1])):
                if d is None:
                    continue
                d = np.asarray(d, float)
                if d.ndim == 2:                  # one direction per column (a tapering tube's end)
                    d = d[j_col[0]]
                    d = d / (np.linalg.norm(d) or 1.0)
                c[b] = c[a] + float((c[b] - c[a]) @ d) * d
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


PIPE_CAPS = ("NONE", "FLAT", "ROUND")


def pipe_cap_style(p):
    """The ends a Pipe gets: NONE (an open tube), FLAT or ROUND. A solid
    (caps=True, Body: Solid / Join / Cut / Intersect) always has caps: flat
    unless round ones are asked for."""
    style = str(p.get("cap_style", "NONE")).upper()
    style = style if style in PIPE_CAPS else "NONE"
    if p.get("caps") and style == "NONE":
        return "FLAT"
    return style


PIPE_CORNER_DEG = 5.0     # a rail turning more than this at one point has a corner: the pipe mitres it
PIPE_MITRE_SHARE = 0.25   # a mitre may reach back at most this share of the stretch it ends


def pipe_frames(rail_k, n, closed=False):
    """(P, T, N, B): the rail's samples and the pipe's rotation-minimising
    frames there (what build_pipe moves the circle along). closed: the frames
    are turned a little more along the way so the last one is the first
    again (rotation-minimising frames round a loop that is not flat come back
    turned), so a closed rail's tube meets itself at its seam."""
    P, T = _arc_samples(rail_k, n)
    N = _rmf(P, T, _perp(T[0], np.array([0, 0, 1.0])))
    if closed and len(P) > 2:
        t = T[0]
        a = N[-1] - (N[-1] @ t) * t
        phi = math.atan2(float(np.cross(a, N[0]) @ t), float(a @ N[0]))   # N[-1] turned by phi about t is N[0]
        s = np.linspace(0.0, 1.0, len(P))
        for k in range(1, len(P)):
            c, si = math.cos(phi * s[k]), math.sin(phi * s[k])
            N[k] = c * N[k] + si * np.cross(T[k], N[k])
        N[-1] = N[0]
    B = np.cross(T, N)
    return P, T, N, B


def rail_corners(kc, max_deg=PIPE_CORNER_DEG):
    """Where the curve turns sharper than max_deg at one point -- the joints
    of a chain of edges, a polyline's corners, a vector handle; a closed
    curve's seam too -- as dicts: u (parameter), point, deg, t_in / t_out
    (unit tangents arriving / leaving), in parameter order."""
    sm, ext = _k()
    d = ext.curve_cvs(kc)
    p, U = int(d["degree"]), [float(u) for u in d["knots"]]
    t0, t1 = kc.parameter_range()
    eps = 1e-7 * ((t1 - t0) or 1.0)

    def tan(t):
        v = np.array(sm.evaluate_curve(kc, float(min(t1, max(t0, t))))["tangent"], float)
        return v / (np.linalg.norm(v) or 1.0)
    knots = sorted({round(u, 12) for u in U[p + 1:len(U) - p - 1] if t0 + eps < u < t1 - eps})
    tests = [(u, tan(u - 4 * eps), tan(u + 4 * eps)) for u in knots if sum(abs(x - u) < 1e-12 for x in U) >= p]
    if features.is_closed_curve(kc):
        tests.insert(0, (t0, tan(t1 - 4 * eps), tan(t0 + 4 * eps)))
    out = []
    for u, a, b in tests:
        deg = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))
        if deg > max_deg:
            out.append({"u": float(u), "point": np.array(kc.evaluate(float(u)), float), "deg": deg,
                        "t_in": a, "t_out": b})
    return out


def pipe_rail(inputs):
    """The rail of one pipe: inputs that touch end to end make one exact curve."""
    return features.profile_curve(inputs)


def _track(kc, ranges, m):
    """A stretch of the rail between corners, resampled evenly by arc length:
    (s, P, T) with m samples. ranges: [(ta, tb)] run in order (a closed
    rail's stretch over its seam takes two). Tangents one-sided at the ends."""
    sm, _ = _k()
    t0, t1 = kc.parameter_range()
    eps = 1e-9 * ((t1 - t0) or 1.0)
    ts, rid = [], []
    for i, (ta, tb) in enumerate(ranges):
        x = np.linspace(ta, tb, 300)
        x[0], x[-1] = ta + eps, tb - eps
        ts += list(x)
        rid += [i] * len(x)
    ts, rid = np.array(ts), np.array(rid)
    Q = np.array([kc.evaluate(float(t)) for t in ts], float)
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    s = np.linspace(0.0, L[-1], m)
    idx = np.clip(np.searchsorted(L, s, side="right") - 1, 0, len(L) - 2)
    tt = []
    for k, i in enumerate(idx):
        if rid[i] != rid[i + 1] or L[i + 1] <= L[i]:
            tt.append(ts[i + 1] if s[k] > L[i] else ts[i])
        else:
            f = (s[k] - L[i]) / (L[i + 1] - L[i])
            tt.append(ts[i] + f * (ts[i + 1] - ts[i]))
    P, T = [], []
    for t in tt:
        dd = sm.evaluate_curve(kc, float(t))
        P.append(dd["point"])
        v = np.array(dd["tangent"], float)
        T.append(v / (np.linalg.norm(v) or 1.0))
    P, T = np.array(P, float), np.array(T, float)
    P[0], P[-1] = Q[0], Q[-1]
    return s, P, T


def _mirror(v, nrm):
    return v - 2.0 * float(v @ nrm) * nrm


def _corner_info(c):
    return {"C": c["point"], "n": c["n"], "deg": c["deg"], "t_in": c["t_in"], "t_out": c["t_out"]}


def _rot(v, axis, ang):
    """v turned by ang (radians) about the unit axis (right-handed)."""
    v = np.asarray(v, float)
    c, s_ = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s_ + axis * float(axis @ v) * (1.0 - c)


def _elbow_axis(c):
    a = np.cross(c["t_in"], c["t_out"])
    return a / (np.linalg.norm(a) or 1.0)


def _turned(v, c):
    """v turned as the rail turns at corner c (t_in onto t_out)."""
    return _rot(v, _elbow_axis(c), math.radians(c["deg"]))


def elbow_reach(c, bend):
    """How far before (and after) corner c an elbow of centreline radius bend starts."""
    return float(bend) * math.tan(math.radians(c["deg"]) / 2.0)


def elbow_geometry(c, bend):
    """(start point, centre, axis, turn in radians) of the elbow at corner c:
    its centreline is the arc of radius bend tangent to both legs."""
    th = math.radians(c["deg"])
    d = elbow_reach(c, bend)
    p0 = c["C"] - d * c["t_in"]
    m = c["t_out"] - float(c["t_out"] @ c["t_in"]) * c["t_in"]
    m /= np.linalg.norm(m)
    return p0, p0 + bend * m, _elbow_axis(c), th


def elbow_cvs(X, w, c, bend):
    """CVs (nu, nv, 3), weights (nu, nv) and u knots of the exact elbow: the
    pipe's end circle (CVs X, weights w, in the plane square to t_in where
    the elbow starts) turned round the elbow's axis through the turn, a
    rational quadratic arc per quarter turn or less."""
    _, O, ax, th = elbow_geometry(c, bend)
    nseg = max(1, int(math.ceil(th / (math.pi / 2) - 1e-9)))
    dth = th / nseg
    rel = np.asarray(X, float) - O
    along = np.outer(rel @ ax, ax)
    rho = rel - along
    rows, wts = [], []
    for i in range(nseg):
        a0 = i * dth
        if i == 0:
            rows.append(O + along + _rot_rows(rho, ax, a0))
            wts.append(np.asarray(w, float))
        rows.append(O + along + _rot_rows(rho, ax, a0 + dth / 2) / math.cos(dth / 2))
        wts.append(np.asarray(w, float) * math.cos(dth / 2))
        rows.append(O + along + _rot_rows(rho, ax, a0 + dth))
        wts.append(np.asarray(w, float))
    U = [0.0, 0.0, 0.0] + [x for i in range(1, nseg) for x in (i / nseg, i / nseg)] + [1.0, 1.0, 1.0]
    return np.array(rows), np.array(wts), U


def _rot_rows(V, axis, ang):
    c, s_ = math.cos(ang), math.sin(ang)
    return V * c + np.cross(axis, V) * s_ + np.outer(V @ axis, axis) * (1.0 - c)


def elbow_limit(tracks):
    """The largest elbow centreline radius every stretch has room for: the
    elbows at its two ends may take it all, no more."""
    best = math.inf
    for tr in tracks:
        L = float(tr["s"][-1])
        k = sum(math.tan(math.radians(tr[key]["deg"]) / 2.0) for key in ("start", "end")
                if tr[key] is not None and tr[key].get("kind") != "split")
        if k > 0:
            best = min(best, L / k)
    return best


def pipe_tracks(rail_k, closed, corner="MITRE"):
    """The rail split at its corners into stretches, each a dict: s, P, T, N
    (frames), h (B = h T x N), start / end (the mitre there: {"C", "n"}, None
    at an open end). A stretch's first frame is the one before it mirrored in
    the mitre plane (the mirror turns the rail's direction round too, so the
    frame keeps its handedness), so the two tubes' end CVs are the same points
    and the tubes face the same way. corner ROUND: the stretches end short of
    each corner for an elbow (a torus patch, elbow_surface) and a stretch's
    first frame is the last one turned round the elbow's axis. None when the
    rail has no corner (a pipe of one piece, as pipe_frames makes it)."""
    corners = rail_corners(rail_k)
    if not corners:
        return None
    for c in corners:
        if c["deg"] > 175.0:
            raise RuntimeError(f"the rail turns back on itself ({c['deg']:.0f}°): a pipe can't take that corner")
        # the mitre plane mirrors the tube before the corner (running back, -t_in) onto the one after (t_out)
        c["n"] = (c["t_in"] + c["t_out"]) / np.linalg.norm(c["t_in"] + c["t_out"])
    t0, t1 = rail_k.parameter_range()
    spans = []
    if not closed:
        cuts = [t0] + [c["u"] for c in corners] + [t1]
        cs = [None] + corners + [None]
        for i in range(len(cuts) - 1):
            spans.append(([(cuts[i], cuts[i + 1])], cs[i], cs[i + 1]))
    else:
        seam = corners[0] if abs(corners[0]["u"] - t0) < 1e-12 else None
        inner = [c for c in corners if c is not seam]
        if seam is not None:
            cuts = [t0] + [c["u"] for c in inner] + [t1]
            cs = [seam] + inner + [seam]
            for i in range(len(cuts) - 1):
                spans.append(([(cuts[i], cuts[i + 1])], cs[i], cs[i + 1]))
        else:               # the seam is smooth: the stretch over it runs from the last corner round to the first
            for i in range(len(inner) - 1):
                spans.append(([(inner[i]["u"], inner[i + 1]["u"])], inner[i], inner[i + 1]))
            spans.append(([(inner[-1]["u"], t1), (t0, inner[0]["u"])], inner[-1], inner[0]))
    tracks, prev = [], None
    for ranges, a, b in spans:
        s, P, T = _track(rail_k, ranges, 241)
        if prev is None:
            r0, h = _perp(T[0], np.array([0, 0, 1.0])), 1.0
        else:
            r0 = _turned(prev["N"][-1], a) if corner == "ROUND" else _mirror(prev["N"][-1], a["n"])
            h = prev["h"]
        N = _rmf(P, T, r0)
        kind = "elbow" if corner == "ROUND" else "mitre"
        tracks.append({"s": s, "P": P, "T": T, "N": N, "h": h,
                       "start": None if a is None else dict(_corner_info(a), kind=kind),
                       "end": None if b is None else dict(_corner_info(b), kind=kind)})
        prev = tracks[-1]
    return tracks


def _ramp(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _frame_end(tr, k):
    T, N = tr["T"][k], tr["N"][k]
    return tr["P"][k], T, N, tr["h"] * np.cross(T, N)


PIPE_TAPERS = ("LINEAR", "SMOOTH")


class RadiusLaw:
    """The pipe's radius along its rail (g: rail length from its start, of
    total L): constant, or through radius points (f, r) -- f the share of the
    rail's length, 0 and 1 its ends -- Linear (straight between the points),
    or Smooth: a monotone cubic (PCHIP) through them, level at both ends, at
    every highest and lowest point and nowhere overshooting the points, so a
    round cap on an end is a half sphere. Two points, Smooth: the smoothstep."""

    def __init__(self, r0, r1=None, taper="LINEAR", L=1.0, points=()):
        r1 = float(r0 if r1 is None else r1)
        pts = sorted((min(1.0, max(0.0, float(f))), float(r)) for f, r in points if 0.0 < float(f) < 1.0)
        self.f = np.array([0.0] + [f for f, _ in pts] + [1.0])
        self.r = np.array([float(r0)] + [r for _, r in pts] + [r1])
        keep = np.concatenate([[True], np.diff(self.f) > 1e-9])      # two points at one place: the later one
        self.f, self.r = self.f[keep], self.r[keep]
        self.f[-1] = 1.0
        self.r0, self.r1 = float(self.r[0]), float(self.r[-1])
        self.taper = taper if taper in PIPE_TAPERS else "LINEAR"
        self.L = float(L) or 1.0
        h = np.diff(self.f)
        d = np.diff(self.r) / h
        m = np.zeros(len(self.f))                 # slopes dr/df at the points: level at both ends
        for k in range(1, len(self.f) - 1):
            if d[k - 1] * d[k] > 0.0:             # Fritsch-Carlson: weighted harmonic mean; 0 at an extreme
                w1, w2 = 2.0 * h[k] + h[k - 1], h[k] + 2.0 * h[k - 1]
                m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        self._m = m

    @property
    def constant(self):
        return float(np.ptp(self.r)) <= 1e-12 * max(abs(self.r0), 1e-12)

    @property
    def rmax(self):
        return float(self.r.max())

    @property
    def rmin(self):
        return float(self.r.min())

    def _eval(self, f, deriv=False):
        f = np.clip(np.asarray(f, float), 0.0, 1.0)
        k = np.clip(np.searchsorted(self.f, f, side="right") - 1, 0, len(self.f) - 2)
        f0, h = self.f[k], self.f[k + 1] - self.f[k]
        t = (f - f0) / h
        ra, rb = self.r[k], self.r[k + 1]
        if self.taper == "LINEAR":
            return (rb - ra) / h if deriv else ra + (rb - ra) * t
        ma, mb = self._m[k], self._m[k + 1]
        if deriv:
            return ((6 * t * t - 6 * t) * ra + (-6 * t * t + 6 * t) * rb) / h + \
                (3 * t * t - 4 * t + 1) * ma + (3 * t * t - 2 * t) * mb
        t2, t3 = t * t, t * t * t
        return (2 * t3 - 3 * t2 + 1) * ra + (t3 - 2 * t2 + t) * h * ma + (-2 * t3 + 3 * t2) * rb + (t3 - t2) * h * mb

    def __call__(self, g):
        if self.constant:
            return np.full(np.shape(g), self.r0) if np.ndim(g) else self.r0
        out = self._eval(np.asarray(g, float) / self.L)
        return out if np.ndim(g) else float(out)

    def slope(self, g):
        """dr / dg."""
        if self.constant:
            return 0.0
        return float(self._eval(float(g) / self.L, deriv=True)) / self.L


def pipe_radius_law(p, L):
    r0 = float(p.get("radius", 0.2))
    if not p.get("variable"):
        return RadiusLaw(r0, r0, "LINEAR", L)
    return RadiusLaw(r0, float(p.get("radius_end", r0)), str(p.get("taper", "LINEAR")).upper(), L,
                     p.get("radius_points", ()))


def mitre_reach(tr, q, bend=None, scale=(1.0, 1.0)):
    """(t0, t1): how far each profile point q (n, 2) lies from the stretch's
    start / end mitre plane along the rail (0 at an open end). At an elbow
    (bend: its centreline radius) every column stops bend tan(turn / 2) short
    of the corner, where the elbow starts. scale: the pipe's radius at the
    start / end corner when q is the unit circle."""
    out = []
    for (k, key, sgn), sc in zip(((0, "start", 1.0), (-1, "end", -1.0)), scale):
        m = tr[key]
        if m is None:
            out.append(np.zeros(len(q)))
            continue
        if m["kind"] == "split":                  # a Linear radius point: the tube just ends there, square
            out.append(np.zeros(len(q)))
            continue
        if m["kind"] == "elbow":
            out.append(np.full(len(q), sgn * elbow_reach(m, bend)))
            continue
        P, T, N, B = _frame_end(tr, k)
        X = P + sc * (np.outer(q[:, 0], N) + np.outer(q[:, 1], B))
        out.append(((m["C"] - X) @ m["n"]) / float(T @ m["n"]))
    return out


def mitre_limit(tracks):
    """The largest radius every mitre has room for: a mitre reaches back
    r tan(turn / 2) along its stretch, at most PIPE_MITRE_SHARE of it."""
    best = math.inf
    unit = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 73)])
    for tr in tracks:
        L = float(tr["s"][-1])
        for t in mitre_reach(tr, unit):
            e = float(np.abs(t).max())
            if e > 0:
                best = min(best, PIPE_MITRE_SHARE * L / e)
    return best


def mitred_rows(tr, q, n, bend=None, law=None, s0=0.0):
    """Rows (n, len(q), 3) of the tube along one stretch, each column the
    profile point q_j carried along the rail; at a mitred end the columns are
    cut off on the mitre plane (an end row on it: the circle projected along
    the rail, an exact ellipse, the same points as the next tube's). Each
    column runs its own length of the rail (from its start mitre to its end
    mitre) at an even pace, so on a straight stretch every column is a
    straight line at constant speed: the exact cylinder, and a light fit.
    A column reaching past the corner runs on straight along the end tangent;
    one ending short of it on a curved stretch is eased onto the ellipse over
    the mitre's reach, so the tube leaves its true shape there by about the
    rail's sag over that reach (no mitre between curved tubes is exact: their
    true cuts by the mitre plane are not the same curve).
    law (RadiusLaw, s0: the rail length where this stretch starts): q is then
    the unit circle and the radius follows the law, held at the corner's
    value over the mitre's / elbow's reach and eased back after it, so the
    mitre / elbow stays exact (two tapering tubes have no common cut)."""
    s, P, T, N, h = tr["s"], tr["P"], tr["T"], tr["N"], tr["h"]
    L = float(s[-1])
    if law is None:
        law = RadiusLaw(1.0)
    rc0, rc1 = law(s0), law(s0 + L)
    t0, t1 = mitre_reach(tr, q, bend, (rc0, rc1))
    e0, e1 = float(np.abs(t0).max()), float(np.abs(t1).max())
    qx, qy = q[:, 0], q[:, 1]

    def radius(S):
        r_ = law(s0 + np.clip(S, 0.0, L))
        if law.constant:
            return r_
        w0 = _ramp((2.0 * e0 - S) / e0) if tr["start"] is not None and e0 > 0 else 0.0 * S
        w1 = _ramp((S - (L - 2.0 * e1)) / e1) if tr["end"] is not None and e1 > 0 else 0.0 * S
        tot = np.maximum(w0 + w1, 1.0)
        w0, w1 = w0 / tot, w1 / tot
        return (1.0 - w0 - w1) * r_ + w0 * rc0 + w1 * rc1

    def sweep(S):
        """Column points at rail lengths S (any shape; the last axis runs over q), straight past the ends."""
        Sc = np.clip(S, 0.0, L)
        A = np.stack([np.stack([np.interp(Sc, s, M[:, i]) for i in range(3)], axis=-1) for M in (P, T, N)])
        Pi, Ti, Ni = A[0], A[1], A[2]
        Ti /= np.linalg.norm(Ti, axis=-1, keepdims=True)
        Ni -= np.sum(Ni * Ti, axis=-1, keepdims=True) * Ti
        Ni /= np.linalg.norm(Ni, axis=-1, keepdims=True)
        rS = np.asarray(radius(S), float)[..., None]
        X = Pi + rS * (qx[..., None] * Ni + qy[..., None] * (h * np.cross(Ti, Ni)))
        return X + np.minimum(S, 0.0)[..., None] * T[0] + np.maximum(S - L, 0.0)[..., None] * T[-1]

    sig = np.linspace(0.0, 1.0, n)[:, None]
    chord = P[-1] - P[0]
    chord = chord / (np.linalg.norm(chord) or 1.0)
    rel = P - P[0]
    straight = float(np.linalg.norm(rel - np.outer(rel @ chord, chord), axis=1).max()) <= 1e-9 * L
    if straight:            # every column a straight line at an even pace: the exact cylinder, a light fit
        S = t0[None, :] + sig * (L + t1 - t0)[None, :]
    else:                   # the columns' offsets only near the corners: the rows stay circles in between
        S = sig * L
        if e0 > 0:
            lam = min(0.45, max(0.12, 1.6 * e0 / L))
            S = S + _ramp(1.0 - sig / lam) * t0[None, :]
        if e1 > 0:
            lam = min(0.45, max(0.12, 1.6 * e1 / L))
            S = S + _ramp((sig - (1.0 - lam)) / lam) * t1[None, :]
    X = sweep(S)
    for t, e, k in ((t0, e0, 0), (t1, e1, -1)):
        if e <= 0:
            continue
        Pk, Tk, Nk, Bk = _frame_end(tr, k)
        rc = rc0 if k == 0 else rc1
        E = Pk + rc * (np.outer(qx, Nk) + np.outer(qy, Bk)) + t[:, None] * Tk      # the ellipse's points
        end = t if k == 0 else L + t
        corr = E - sweep(end)                                                # 0 where the column runs straight
        far = (S - end[None, :]) if k == 0 else (end[None, :] - S)          # rail length from the column's end
        X = X + _ramp(1.0 - far / e)[..., None] * corr[None, :, :]
    return X


def corner_message(inputs, corners):
    names = sorted({o.name for o, _ in inputs})
    where = names[0] if len(names) == 1 else f"the chain of {len(inputs)}"
    worst = max(c["deg"] for c in corners)
    n = len(corners)
    return f"{where} turns {worst:.0f}° at {'a corner' if n == 1 else f'{n} corners'}"


def mitre_message(inputs, tracks, r):
    lim = mitre_limit(tracks)
    return (f"the radius is too large for the mitred corners: at most {lim:.4g} here (a mitre may reach back "
            f"a quarter of the stretch it ends) -- a smaller radius, or pipe the pieces separately")


def cap_meridian(r, slope):
    """(offset, R, theta0) of the round cap on a pipe end of radius r whose
    radius grows outward at slope dr/ds: the sphere tangent to the tube along
    the end circle has its centre offset r * slope along the outward axis and
    radius R = r sqrt(1 + slope^2); seen from its centre the end circle lies
    at theta0 = -atan(slope) and the pole at 90 degrees (slope 0: the half
    sphere)."""
    return r * slope, r * math.sqrt(1.0 + slope * slope), -math.atan(slope)


def round_cap_cvs(C, N, B, T, q, w, r, slope=0.0):
    """CVs (nu, nv, 3), weights (nu, nv) and u knots of the exact round cap
    closing a pipe end: the end circle (centre C, CVs C + r (q_x N + q_y B)
    for the unit circle q, weights w) carried over a sphere tangent to the
    tube (cap_meridian) to its pole on the outward axis T. A rational
    quadratic meridian per quarter turn or less (one for a half sphere)."""
    off, R, th0 = cap_meridian(r, slope)
    O = C + off * T
    span = 0.5 * math.pi - th0
    nseg = max(1, int(math.ceil(span / (0.5 * math.pi) - 1e-9)))
    d = span / nseg
    mer = []                                  # (rho, z, weight) of the meridian's CVs
    for i in range(nseg):
        a0 = th0 + i * d
        if i == 0:
            mer.append((R * math.cos(a0), R * math.sin(a0), 1.0))
        am = a0 + d / 2
        mer.append((R * math.cos(am) / math.cos(d / 2), R * math.sin(am) / math.cos(d / 2), math.cos(d / 2)))
        mer.append((R * math.cos(a0 + d), R * math.sin(a0 + d), 1.0))
    ring = np.outer(q[:, 0], N) + np.outer(q[:, 1], B)
    P = np.array([O + z * T + rho * ring for rho, z, _ in mer])
    P[-1] = O + R * T                         # the pole, exactly
    W = np.array([[wm * wi for wi in w] for _, _, wm in mer])
    U = [0.0, 0.0, 0.0] + [x for i in range(1, nseg) for x in (i / nseg, i / nseg)] + [1.0, 1.0, 1.0]
    return P, W, U


def _round_cap(C, N, B, T, prof, r, slope=0.0):
    """prof: the unit circle's profile."""
    _, ext = _k()
    q = prof["points"]
    w = prof["weights"] or [1.0] * len(q)
    P, W, U = round_cap_cvs(C, N, B, T, q, w, r, slope)
    pts = [tuple(map(float, P[j, i])) for j in range(len(P)) for i in range(len(q))]
    wts = [float(W[j, i]) for j in range(len(P)) for i in range(len(q))]
    return ext.surface_from_cvs(2, prof["degree"], pts, U, prof["knots"], wts)


def _end_dirs(T, N, B, q, slope):
    """The directions each column of a tapering tube leaves its end along
    (T + dr/ds times the profile point's direction): G1 to a round cap."""
    if abs(slope) <= 1e-15:
        return T
    return T[None, :] + slope * (np.outer(q[:, 0], N) + np.outer(q[:, 1], B))


@features.builder("pipe")
def build_pipe(inputs, p):
    """The tube along the rail (inputs that touch end to end make one); with
    caps (Cap Flat / Round, or a solid) a closed body: flat discs or exact
    round caps on the ends. A closed rail has no ends: its tube is closed
    already (a solid of it is the ring). A rail with sharp corners gets a
    tube per stretch, mitred where they meet (pipe_tracks / mitred_rows), or
    with Corners Round joined by exact torus elbows of centreline radius
    Bend x the pipe's radius (elbow_cvs), G1 to straight legs. Variable
    Radius: the radius runs from Radius to End Radius along the rail
    (RadiusLaw); a round cap is then the sphere tangent to the tapering tube,
    and the radius holds its corner's value through each mitre / elbow."""
    from .primitives import exact_circle_spec, spec_to_kernel_curve
    rail_k = pipe_rail(inputs)
    closed = features.is_closed_curve(rail_k)
    prof = _profile(spec_to_kernel_curve(exact_circle_spec(1.0)))
    q = prof["points"]                       # the unit circle in its local XY plane, centre origin
    samples = int(p.get("samples", 32))
    style = pipe_cap_style(p)
    corner = "ROUND" if str(p.get("corner_style", "MITRE")).upper() == "ROUND" else "MITRE"
    tracks = pipe_tracks(rail_k, closed, corner)
    creased = bool(p.get("variable")) and str(p.get("taper", "LINEAR")).upper() == "LINEAR" and \
        any(0.0 < float(f) < 1.0 for f, _ in p.get("radius_points", ()))
    if tracks is None and creased:
        tracks = [_whole_track(rail_k)]         # built in pieces between the radius points
    if tracks is None:
        n = min(600, samples * len(inputs))      # samples per curve / edge of the rail
        P, T, N, B = pipe_frames(rail_k, n, closed)
        g = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        law = pipe_radius_law(p, float(g[-1]))
        _check_law(law, closed)
        rk = law(g)
        rows = [P[k] + rk[k] * (np.outer(q[:, 0], N[k]) + np.outer(q[:, 1], B[k])) for k in range(n)]
        round_ = style == "ROUND" and not closed
        s_a, s_b = law.slope(0.0), law.slope(float(g[-1]))
        # round caps: the tube leaves its ends along the cap's sphere, so the caps are G1 to it
        ends = (_end_dirs(T[0], N[0], B[0], q, s_a), _end_dirs(T[-1], N[-1], B[-1], q, s_b)) if round_ else None
        tube = _surface_from_rows(rows, prof, end_dirs=ends)
        if style == "NONE":
            return tube
        walls = [tube]
        if round_:
            walls += [_round_cap(P[0], N[0], B[0], -T[0], prof, law.r0, -s_a),
                      _round_cap(P[-1], N[-1], B[-1], T[-1], prof, law.r1, s_b)]
        return features.cap_walls(walls)
    total = sum(float(tr["s"][-1]) for tr in tracks)
    law = pipe_radius_law(p, total)
    _check_law(law, closed)
    if creased:
        tracks = _split_tracks(tracks, law)
    rmax = law.rmax
    bend = float(p.get("bend", 2.0)) * rmax
    if corner == "MITRE" and rmax > mitre_limit(tracks) * (1.0 + 1e-9):
        raise RuntimeError(mitre_message(inputs, tracks, rmax))
    if corner == "ROUND":
        if bend < rmax * (1.0 - 1e-9):
            raise RuntimeError("the elbows' bend radius is smaller than the pipe's: Bend must be at least 1")
        if bend > elbow_limit(tracks) * (1.0 + 1e-6):     # elbows that meet fill their stretch exactly
            raise RuntimeError(elbow_message(tracks, rmax))
    walls, caps = [], []
    s0 = 0.0
    for tr in tracks:
        L = float(tr["s"][-1])
        n = int(min(300, max(samples, round(samples * len(inputs) * L / total))))
        used = sum(elbow_reach(tr[key], bend) for key in ("start", "end")
                   if tr[key] is not None and tr[key]["kind"] == "elbow")
        if L - used > 1e-7 * L:                  # elbows that meet leave no straight run between them
            rows = _affine_rows(mitred_rows(tr, q, n, bend, law, s0), q)
            ends = []
            for k, key, out in ((0, "start", -1.0), (-1, "end", 1.0)):
                Pk, Tk, Nk, Bk = _frame_end(tr, k)
                if tr[key] is not None:
                    # a mitre: a crease, the end row exact; an elbow: square to the leg, G1 to it
                    ends.append(Tk if tr[key]["kind"] == "elbow" else None)
                elif style == "ROUND":
                    gk = s0 + (0.0 if k == 0 else L)
                    sl = law.slope(gk)
                    ends.append(_end_dirs(Tk, Nk, Bk, q, sl))
                    caps.append(_round_cap(Pk, Nk, Bk, out * Tk, prof, law(gk), out * sl))
                else:
                    ends.append(None)
            walls.append(_surface_from_rows(rows, prof, end_dirs=tuple(ends)))
        elif style == "ROUND":                    # no tube, but an open end still gets its cap
            for k, key, out in ((0, "start", -1.0), (-1, "end", 1.0)):
                if tr[key] is None:
                    gk = s0 + (0.0 if k == 0 else L)
                    Pk, Tk, Nk, Bk = _frame_end(tr, k)
                    caps.append(_round_cap(Pk, Nk, Bk, out * Tk, prof, law(gk), out * law.slope(gk)))
        if tr["end"] is not None and tr["end"]["kind"] == "elbow":
            walls.append(_elbow_surface(tr, prof, bend, law(s0 + L)))
        s0 += L
    walls += caps
    if style == "NONE":
        return _joined(walls)
    return features.cap_walls(walls)


def _whole_track(rail_k):
    """The rail as one stretch (no corners), sampled densely: for building it
    in pieces (Linear radius points)."""
    t0, t1 = rail_k.parameter_range()
    s, P, T = _track(rail_k, [(t0, t1)], 401)
    N = _rmf(P, T, _perp(T[0], np.array([0, 0, 1.0])))
    return {"s": s, "P": P, "T": T, "N": N, "h": 1.0, "start": None, "end": None}


def _split_tracks(tracks, law):
    """The stretches cut at the law's inner points (Linear: the radius turns
    there, a crease between two cones, which one fit along the rail would round
    off): the pieces share the point's sample, so their end circles are the same
    CVs. Ends made by a cut are kind "split" (no cap, no mitre)."""
    cuts = [float(f) * law.L for f in law.f[1:-1]]
    out, s0 = [], 0.0
    for tr in tracks:
        L = float(tr["s"][-1])
        pieces, cur, a = [], tr, s0
        for g in cuts:
            Lc = float(cur["s"][-1])
            x = g - a
            if not 1e-9 * L < x < Lc - 1e-9 * L:
                continue
            s, P, T, N = cur["s"], cur["P"], cur["T"], cur["N"]
            j = int(np.searchsorted(s, x))
            f = (x - s[j - 1]) / (s[j] - s[j - 1])
            Px = P[j - 1] + f * (P[j] - P[j - 1])
            Tx = T[j - 1] + f * (T[j] - T[j - 1])
            Tx /= np.linalg.norm(Tx)
            Nx = N[j - 1] + f * (N[j] - N[j - 1])
            Nx -= float(Nx @ Tx) * Tx
            Nx /= np.linalg.norm(Nx)
            left = {"s": np.append(s[:j], x), "P": np.vstack([P[:j], Px]), "T": np.vstack([T[:j], Tx]),
                    "N": np.vstack([N[:j], Nx]), "h": cur["h"], "start": cur["start"], "end": {"kind": "split"}}
            right = {"s": np.concatenate([[0.0], s[j:] - x]), "P": np.vstack([Px, P[j:]]),
                     "T": np.vstack([Tx, T[j:]]), "N": np.vstack([Nx, N[j:]]), "h": cur["h"],
                     "start": {"kind": "split"}, "end": cur["end"]}
            pieces.append(left)
            cur, a = right, a + x
        pieces.append(cur)
        out += pieces
        s0 += L
    return out


def _check_law(law, closed):
    if closed and not law.constant:
        raise RuntimeError("a closed rail has no start and end for a varying radius: use one radius")
    if law.rmin <= 0.0:
        raise RuntimeError("the radius must stay above 0")


def _affine_rows(rows, q):
    """Rows of profile CVs, each replaced by the affine image of the profile
    (CVs q, in its plane) closest to it. A row that is not a rigid copy of
    the circle (the columns of a mitred stretch run different lengths of a
    curved rail) would bend at the circle's joints (its quarter points: knots
    of full multiplicity, G1 only while the three CVs there stay in line), and
    the normal would jump along those lines; an affine image keeps them in
    line, so every row is an exact conic. The fit along the rail is linear in
    the rows and affine images of q make a linear space, so the surface keeps
    it: G1 across those lines. Circles and the ellipses they project to on a
    mitre are affine images already, and are kept exactly."""
    rows = np.asarray(rows, float)
    A = np.hstack([np.asarray(q, float)[:, :2], np.ones((len(q), 1))])        # (nv, 3)
    pinv = np.linalg.pinv(A)
    return np.einsum("ij,kjl->kil", A, np.einsum("ij,kjl->kil", pinv, rows))


def _elbow_surface(tr, prof, bend, r=1.0):
    """The exact elbow at the end of stretch tr: its end frame's circle of
    radius r (prof: the unit circle), where the elbow starts, turned round
    the corner."""
    _, ext = _k()
    c = tr["end"]
    Pk, Tk, Nk, Bk = _frame_end(tr, -1)
    p0 = elbow_geometry(c, bend)[0]
    q = prof["points"]
    w = prof["weights"] or [1.0] * len(q)
    X = p0 + r * (np.outer(q[:, 0], Nk) + np.outer(q[:, 1], Bk))
    R, W, U = elbow_cvs(X, w, c, bend)
    pts = [tuple(map(float, R[i, j])) for i in range(len(R)) for j in range(len(q))]
    wts = [float(W[i, j]) for i in range(len(R)) for j in range(len(q))]
    return ext.surface_from_cvs(2, prof["degree"], pts, U, prof["knots"], wts)


def elbow_message(tracks, r):
    lim = elbow_limit(tracks) / r
    return (f"the elbows don't fit: Bend at most {lim:.3g} here (two elbows may take a whole stretch "
            f"between them) -- a smaller Bend or radius, or Corners: Mitre")


def _joined(walls):
    """Surfaces (world) stitched along their shared edges into one open body."""
    from . import jobs
    from .steps import IDENT
    objs = {f"s{i}": (store.brep_to_text(store.to_brep(w)), IDENT) for i, w in enumerate(walls)}
    lo = np.min([np.array(store.to_brep(w).bounding_box()[0], float) for w in walls], axis=0)
    hi = np.max([np.array(store.to_brep(w).bounding_box()[1], float) for w in walls], axis=0)
    tol = 1e-6 * (float(np.linalg.norm(hi - lo)) or 1.0)
    _, breps = jobs.run("join", objs, ["joined"], order=list(objs), into="s0", tolerance=tol)
    return ("brep_text", breps["joined"])


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


def _spec_samples(sp, n=97):
    deg, P, W, U = geom.blender_nurbs_to_clamped(sp["points"], sp["weights"], sp["order"], sp["cyclic"],
                                                 sp["endpoint"], sp["bezier"])
    return np.array([geom.eval_bspline(P, W, U, deg, t) for t in np.linspace(U[deg], U[-deg - 1], n)])


def _cyclic_if_closed(sp):
    """A closed endpoint + Bezier spec as a cyclic one (one CV fewer, no seam
    CV pair), when that holds exactly the same curve."""
    if sp["cyclic"] or not (sp["endpoint"] and sp["bezier"]):
        return sp
    P, W, k = np.array(sp["points"], float), sp["weights"], sp["order"]
    size = float(np.ptp(P, axis=0).max()) or 1.0
    n = len(P) - 1
    if (n < 2 * (k - 1) or n % (k - 1) or np.linalg.norm(P[0] - P[-1]) > 1e-9 * size
            or abs(W[0] - W[-1]) > 1e-12 * abs(W[0])):
        return sp
    cyc = dict(sp, points=sp["points"][:-1], weights=W[:-1], cyclic=True)
    try:
        A, B = _spec_samples(sp), _spec_samples(cyc)
    except Exception:
        return sp
    return cyc if np.abs(A - B).max() <= 1e-9 * size else sp


def _kcurve_to_spec(kc):
    """The kernel curve's own CVs, weights and knots as a Blender spline
    (exact: Bezier spans, or uniform when that is exact; cyclic when closed).
    Only a curve Blender cannot hold (degree above 5) is refitted."""
    from .curvejoin import kernel_curve_spans, spans_to_spec
    try:
        return _cyclic_if_closed(spans_to_spec(kernel_curve_spans(kc), simplify=True))
    except ValueError:
        return _poly_to_spec(_kcurve_poly(kc))


def _iso_kcurve(obj, iso_dir, t):
    """Exact world-space isoparm of an object's surface at fraction t of the
    fixed direction's domain: the CV columns blended by that direction's basis
    (in homogeneous space), with the other direction's degree and knots."""
    _, ext = _k()
    d = ext.surface_cvs(store.get_surfaces(obj)[0])
    nu, nv = d["count_u"], d["count_v"]
    Pg = np.array(d["points"], float).reshape(nu, nv, 3)
    Wg = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    H = np.concatenate([Pg * Wg[..., None], Wg[..., None]], axis=2)
    if iso_dir == "U":                       # u fixed, the curve runs along v
        U, deg, cnt = list(d["knots_u"]), d["degree_u"], nu
    else:
        U, deg, cnt = list(d["knots_v"]), d["degree_v"], nv
        H = H.transpose(1, 0, 2)
    s = U[deg] + (U[-deg - 1] - U[deg]) * min(max(t, 0.0), 1.0)
    N = geom.interp_matrix([s], U, deg, cnt)[0]
    Q = np.tensordot(N, H, axes=(0, 0))      # n_out x 4
    W = Q[:, 3]
    P = Q[:, :3] / W[:, None]
    M = np.array(geom.world_matrix(obj), dtype=float)
    Pw = [tuple(map(float, M[:3, :3] @ p + M[:3, 3])) for p in P]
    knots = d["knots_v"] if iso_dir == "U" else d["knots_u"]
    deg_c = d["degree_v"] if iso_dir == "U" else d["degree_u"]
    Wl = None if np.abs(W - 1.0).max() < 1e-12 else [float(w) for w in W]
    return ext.curve_from_cvs(int(deg_c), Pw, [float(k) for k in knots], Wl)


@features.builder("iso_curve")
def build_iso_curve(inputs, p):
    obj = inputs[0][0]
    kc = _iso_kcurve(obj, p.get("iso_dir", "U"), float(p.get("param", 0.5)))
    return ("splines", [_kcurve_to_spec(kc)])


@features.builder("dup_edge")
def build_dup_edge(inputs, p):
    return ("splines", [_kcurve_to_spec(features.input_curve(o, r)) for o, r in inputs])


class ProjectMiss(RuntimeError):
    """No curve lands on the surface along the projection direction."""


# SmApiProjectCurve returns the bare SM_ERR (not SM_ERR_INVALID_INPUT) when
# nothing lands on the Brep: the direction is parallel to the surface, or the
# curve's shadow falls past the surface's edges.
_PROJECT_MISS = "SM_ERR (1001)"


def _brep_planes(brep):
    """(point, unit normal, face) of every planar face of a brep (CVs coplanar)."""
    _, ext = _k()
    out = []
    for f in brep.faces():
        P = np.array(ext.surface_cvs(f.surface())["points"], float)
        c = P.mean(axis=0)
        size = float(np.ptp(P, axis=0).max()) or 1.0
        n = np.linalg.svd(P - c)[2][-1]
        if np.abs((P - c) @ n).max() <= 1e-9 * size:
            out.append((c, n, f))
    return out


def _closest_g(spans, x, per=64):
    """Global span parameter of the point of a span list closest to x."""
    from .curvejoin import eval_span
    from .curvetrim import sample_spans
    Q, G = sample_spans(spans, per)
    g = float(G[int(np.argmin(np.linalg.norm(Q - x, axis=1)))])
    n, h = len(spans), 1.0 / per
    lo, hi = max(0.0, g - h), min(float(n), g + h)

    def dist(g_):
        i = min(int(g_), n - 1)
        return float(np.linalg.norm(eval_span(spans[i], [g_ - i])[0] - x))
    r = (math.sqrt(5) - 1) / 2                 # golden section on the bracket
    a, b = hi - r * (hi - lo), lo + r * (hi - lo)
    for _ in range(60):
        if dist(a) < dist(b):
            hi, b = b, a
            a = hi - r * (hi - lo)
        else:
            lo, a = a, b
            b = lo + r * (hi - lo)
    return 0.5 * (lo + hi)


def _exact_planar_piece(kc, poly, planes, d):
    """The projection of curve kc onto a plane is an affine image of it: the
    same degree, knots and weights, its CVs moved onto the plane along d. The
    kernel's projected piece `poly` (sampled) says which plane and which part
    of the curve lands on the face; returns that part as an exact spline spec,
    or None when the piece is not on a planar face or the exact part does not
    follow it."""
    from .curvejoin import kernel_curve_spans, transform_spans, spans_to_spec, _size
    from .curvetrim import clean_cuts, is_closed, piece_at, split_spans, sample_spans
    P = np.asarray(poly, float)
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    d = np.asarray(d, float)
    plane = next(((c, n, f) for c, n, f in planes if np.abs((P - c) @ n).max() <= 1e-6 * size
                  and abs(d @ n) > 1e-6 * np.linalg.norm(d)), None)
    if plane is None:
        return None
    c, n, face = plane
    M = np.eye(4)                              # x -> x - d (n.(x - c)) / (d.n)
    M[:3, :3] -= np.outer(d, n) / (d @ n)
    M[:3, 3] = d * (n @ c) / (d @ n)
    try:
        spans = transform_spans(kernel_curve_spans(kc), M)
    except ValueError:
        return None
    closed = is_closed(spans)
    tol = 1e-6 * _size(spans)
    a, b = P[0], P[-1]
    ends = [spans[0][0, :3] / spans[0][0, 3], spans[-1][-1, :3] / spans[-1][-1, 3]]
    whole = (np.linalg.norm(a - b) <= tol) if closed else (
        min(np.linalg.norm(a - ends[0]) + np.linalg.norm(b - ends[1]),
            np.linalg.norm(a - ends[1]) + np.linalg.norm(b - ends[0])) <= 2 * tol)
    if whole:
        part = spans
    else:
        # the kernel's piece ends are only near the face's edges (about 1e-3 of
        # the size): cut where the exact curve crosses an edge, nearest each end
        from .curvetrim import intersect
        from .square import _edge_curve
        hits = []
        for e in face.edges():
            try:
                hits += [(g, X) for g, _, X in intersect(spans, kernel_curve_spans(_edge_curve(e)), 1e-7 * size)]
            except Exception:
                pass

        def cut_at(x):
            if hits:
                g, X = min(hits, key=lambda h: np.linalg.norm(h[1] - x))
                if np.linalg.norm(X - x) <= 1e-2 * size:
                    return g, X
            return _closest_g(spans, x), x
        cuts = clean_cuts(spans, [cut_at(x) for x in (a, b)], closed)
        pieces = split_spans(spans, cuts, closed)
        part = pieces[piece_at(pieces, _closest_g(spans, P[len(P) // 2]), len(spans))][0]
    spec = spans_to_spec(part, simplify=True)
    if whole and closed:
        spec = _cyclic_if_closed(spec)
    Q = sample_spans(part, max(32, 400 // len(part)))[0]   # it must lie where the kernel's piece does
    dd = np.linalg.norm(Q[:, None] - P[None], axis=2)
    if max(dd.min(axis=1).max(), dd.min(axis=0).max()) > 5e-3 * size:
        return None
    return spec


@features.builder("project_curve")
def build_project_curve(inputs, p):
    sm, _ = _k()
    *curves, (surf_obj, _) = inputs
    d = p.get("dir", [0, 0, -1])
    brep_w = _world_brep(surf_obj)
    planes = _brep_planes(brep_w)
    out = []
    for o, r in curves:
        kc = features.input_curve(o, r)
        try:
            pcs = sm.project_curve(brep_w, kc, tuple(d))
        except RuntimeError as exc:
            if _PROJECT_MISS not in str(exc):
                raise
            continue                    # this curve misses; keep the others
        for pc in pcs:
            poly = _kcurve_poly(pc, 400)
            # onto a plane: exact (the curve's own CVs moved); else a fit
            out.append((planes and _exact_planar_piece(kc, poly, planes, d)) or _poly_to_spec(poly))
    if not out:
        raise ProjectMiss("the curve misses the surface in that direction")
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
