"""Dome: close an open end of a surface (a rim) with a dome, G1 or G2 to it.

A round rim -- the end of a cylinder, a cone or any surface of revolution --
gets an exact surface of revolution: a Bezier profile from the rim to the
axis that leaves the rim along the surface (G1: tangent; G2: tangent and the
same curvature, degree 5) and reaches the axis level, so the top is smooth.
It is revolved with the rational circle (9 CVs round, starting at the rim's
own seam), so the dome meets the rim exactly. Height (x the rim's radius) and
Fullness shape the profile.

Any other planar rim (an ellipse, a rounded rectangle ...) gets a dome on
the rim's own curve: its first row IS the rim, the next rows are fitted to
leave it along the surface (G1 / G2), the top rows are the rim scaled towards
the top point. Both keep history: the dome follows the surface it closes.
"""

import json

import numpy as np

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty, StringProperty

from . import features, store
from .ops import AskValue, PickMixin, _decode_inputs, _encode_inputs, _kernel_ok, _make_result, _pick_candidates, _select_only

K_CIRCLE = 0.5522847498            # cubic Bezier quarter circle: tangent length / radius
S2 = np.sqrt(0.5)


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


# --------------------------------------------------------------------------
# the rim: is it round, and how does the surface run into it
# --------------------------------------------------------------------------

def _vec(q):
    return np.array([q.x, q.y, q.z], float) if hasattr(q, "x") else np.array(q, float)


def rim_data(obj, ref, n=64):
    """What a dome needs to know about the rim `ref` of obj (world space):
    {"round", "c", "a", "x0", "r", "T2", "kappa", "why"}. T2 = the surface's
    direction across the rim, outward, in the rim's (radial, axial) plane;
    kappa = its curvature there (> 0: bending towards the axis)."""
    sm, _ = _k()
    kc = features.input_curve(obj, ref)
    t0, t1 = kc.parameter_range()
    ends = _vec(kc.evaluate(t0)), _vec(kc.evaluate(t1))
    ts = np.linspace(t0, t1, n, endpoint=False)
    P = np.array([_vec(kc.evaluate(t)) for t in ts])
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    out = {"round": False, "why": ""}
    if np.linalg.norm(ends[0] - ends[1]) > 1e-5 * size:
        out["why"] = "the edge is not a closed rim"
        return out
    m0 = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - m0)
    a = vt[2]
    e1, e2 = vt[0], vt[1]
    xy = np.c_[(P - m0) @ e1, (P - m0) @ e2]
    cx, cy, _ = np.linalg.lstsq(np.c_[2.0 * xy, np.ones(len(xy))], (xy ** 2).sum(axis=1), rcond=None)[0]
    c = m0 + cx * e1 + cy * e2
    rad = P - c - np.outer((P - c) @ a, a)
    rr = np.linalg.norm(rad, axis=1)
    r = float(rr.mean())
    planar = float(np.abs((P - m0) @ a).max()) <= 1e-4 * r
    circle = float(np.ptp(rr)) <= 1e-4 * r
    # the surface across the rim, at a few points round it
    from . import geom
    M = np.array(geom.world_matrix(obj), float)
    R, Tm = M[:3, :3], M[:3, 3]
    Ri = np.linalg.inv(R)
    faces = store.get_brep(obj).faces()
    h = 1e-4 * (t1 - t0)
    eps, dlt = 1e-3 * size, 0.03 * r
    T2s, ks, tang = [], [], []
    for i in range(0, n, max(1, n // 8)):
        p = P[i]
        te = _vec(kc.evaluate(min(ts[i] + h, t1))) - _vec(kc.evaluate(max(ts[i] - h, t0)))
        te /= np.linalg.norm(te) or 1.0
        q = Ri @ (p - Tm)
        face = min(faces, key=lambda f: f.closest_point(tuple(q))[1])
        surf = face.surface()
        _, uvs, _ = sm.surface_closest_point(surf, tuple(float(x) for x in q))
        uv = (uvs[0].x, uvs[0].y) if hasattr(uvs[0], "x") else (uvs[0][0], uvs[0][1])
        nw = Ri.T @ np.array(surf.normal(*uv), float)
        nw /= np.linalg.norm(nw) or 1.0
        to = np.cross(te, nw)
        to /= np.linalg.norm(to) or 1.0
        tl = Ri @ to
        if face.closest_point(tuple(q - eps * tl))[1] > face.closest_point(tuple(q + eps * tl))[1]:
            to, tl = -to, -tl                       # outward: the side away from the face
        if i == 0:
            if to @ a < 0:
                a = -a                              # the axis points the way the surface leaves the rim
        er = rad[i] / (rr[i] or 1.0)
        u2 = np.array([to @ er, to @ a])
        tang.append(abs(to @ np.cross(a, er)))
        # curvature across the rim: a point a little way back on the surface
        fq = _vec(sm.surface_closest_point(surf, tuple(float(x) for x in q - dlt * tl))[0])
        d3 = R @ (fq - q)
        n2 = -u2[1] * er + u2[0] * a
        ks.append(2.0 * float(d3 @ n2) / float(d3 @ d3))
        T2s.append(u2 / (np.linalg.norm(u2) or 1.0))
    T2s = np.array(T2s)
    T2 = T2s.mean(axis=0)
    T2 /= np.linalg.norm(T2) or 1.0
    spread = float(np.degrees(np.arccos(np.clip(T2s @ T2, -1.0, 1.0))).max())
    kap = float(np.mean(ks))
    out.update(c=c, a=a, x0=rad[0] / (rr[0] or 1.0), r=r, T2=T2, kappa=kap)
    if not (planar and circle):
        out["why"] = "the rim is not a circle"
    elif spread > 0.3 or max(tang) > 5e-3 or np.ptp(ks) > max(0.02 * abs(kap), 1e-3 / r):
        out["why"] = "the surface does not run into the rim the same way all round"
    elif T2[1] < 0.05:
        out["why"] = "the surface does not run up into the rim (it ends flat or turned back)"
    else:
        out["round"] = True
    return out


# --------------------------------------------------------------------------
# the dome of a round rim: an exact surface of revolution
# --------------------------------------------------------------------------

def round_height(r, u):
    """Height of the round dome on a rim of radius r that the surface runs into
    along u = (radial, axial): the sphere tangent to it there, centred on the
    axis (a vertical wall: a hemisphere, r; leaning in, lower)."""
    return r * (1.0 + u[0]) / u[1]


def arc_k(u):
    """Cubic handle length / tangent length for the turn from u to the top's
    level direction: (4/3) tan(theta/4) / tan(theta/2), 0.5523 for a quarter."""
    th = float(np.arccos(np.clip(-u[0], -1.0, 1.0)))
    th = min(max(th, 0.05), np.pi - 0.05)
    return (4.0 / 3.0) * np.tan(th / 4.0) / np.tan(th / 2.0)


def profile(rim, cont="G2", height=1.0, fullness=1.0):
    """The dome's profile: Bezier CVs (rows, 2) as (radius, height above the
    rim). height 1 = the round dome (round_height); less: flatter."""
    r = rim["r"]
    u = np.array(rim["T2"], float)
    H = max(height, 0.02) * round_height(r, u)
    P0, A, e = np.array([r, 0.0]), np.array([0.0, H]), np.array([1.0, 0.0])
    X = P0 + (H / u[1]) * u                       # where the rim's tangent reaches the top's height
    L0 = float(np.linalg.norm(X - P0))
    Xr = max(float(X[0]), 0.25 * r)               # (a wall leaning in steeply: keep the top round)
    k = arc_k(u) * fullness
    C = [P0, P0 + k * L0 * u, A + k * Xr * e, A]       # the round arc (cubic)
    if cont == "G1":
        cv = C
    else:
        cv = _g2_rows(C, u, rim["kappa"])
    cv = np.array(cv, float)
    cv[:, 0] = np.maximum(cv[:, 0], 0.0)          # never past the axis
    return cv


def _elevate(C):
    """Cubic Bezier CVs -> the same curve as degree 5."""
    C = [np.asarray(c, float) for c in C]
    Q = [C[0], (C[0] + 3.0 * C[1]) / 4.0, (C[1] + C[2]) / 2.0, (3.0 * C[2] + C[3]) / 4.0, C[3]]
    return [Q[0], (Q[0] + 4.0 * Q[1]) / 5.0, (2.0 * Q[1] + 3.0 * Q[2]) / 5.0, (3.0 * Q[2] + 2.0 * Q[3]) / 5.0,
            (4.0 * Q[3] + Q[4]) / 5.0, Q[4]]


def _g2_rows(C, u, kappa, m=5):
    """The round arc C (cubic) as degree 5 -- its top keeps the arc's own
    curvature, so the dome stays round there -- with only the rim end made G2:
    the third CV on the rim's tangent, offset by the surface's curvature."""
    R = _elevate(C)
    al = float(np.linalg.norm(R[1] - R[0]))
    n2 = np.array([-u[1], u[0]])
    R[2] = R[0] + 2.0 * al * np.asarray(u, float) + (kappa * m * al * al / (m - 1)) * n2
    return R


def revolved(rim, cv):
    """The profile turned round the rim's axis: u = the profile (rim -> top),
    v = round (rational circle, 9 CVs, from the rim's seam)."""
    _, ext = _k()
    c, a, x0 = (np.array(rim[k], float) for k in ("c", "a", "x0"))
    y0 = np.cross(a, x0)
    pts, wts = [], []
    for rho, z in cv:
        for k in range(9):
            ang = k * np.pi / 4.0
            rr = rho if k % 2 == 0 else rho / S2
            q = c + z * a + rr * (np.cos(ang) * x0 + np.sin(ang) * y0)
            pts.append(tuple(float(x) for x in q))
            wts.append(1.0 if k % 2 == 0 else float(S2))
    m = len(cv) - 1
    return ext.surface_from_cvs(m, 2, pts, [0.0] * (m + 1) + [1.0] * (m + 1),
                                [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0], wts)


# --------------------------------------------------------------------------
# the dome of any planar rim: rows on the rim's own basis
# --------------------------------------------------------------------------

def _across(obj, kc, ts, size, r_ref):
    """Outward tangent T (unit, world) across the rim and the surface's normal
    curvature along it, at the curve parameters ts."""
    from . import geom
    sm, _ = _k()
    M = np.array(geom.world_matrix(obj), float)
    R, Tm = M[:3, :3], M[:3, 3]
    Ri = np.linalg.inv(R)
    faces = store.get_brep(obj).faces()
    t0, t1 = kc.parameter_range()
    h = 1e-4 * (t1 - t0)
    eps, dlt = 1e-3 * size, 0.03 * r_ref
    P, T, K, NS = [], [], [], []
    for t in ts:
        p = _vec(kc.evaluate(t))
        te = _vec(kc.evaluate(min(t + h, t1))) - _vec(kc.evaluate(max(t - h, t0)))
        te /= np.linalg.norm(te) or 1.0
        q = Ri @ (p - Tm)
        face = min(faces, key=lambda f: f.closest_point(tuple(q))[1])
        surf = face.surface()
        _, uvs, _ = sm.surface_closest_point(surf, tuple(float(x) for x in q))
        uv = (uvs[0].x, uvs[0].y) if hasattr(uvs[0], "x") else (uvs[0][0], uvs[0][1])
        nw = Ri.T @ np.array(surf.normal(*uv), float)
        nw /= np.linalg.norm(nw) or 1.0
        to = np.cross(te, nw)
        to /= np.linalg.norm(to) or 1.0
        tl = Ri @ to
        if face.closest_point(tuple(q - eps * tl))[1] > face.closest_point(tuple(q + eps * tl))[1]:
            to, tl = -to, -tl
        fq = _vec(sm.surface_closest_point(surf, tuple(float(x) for x in q - dlt * tl))[0])
        P.append(p)
        T.append(to)
        NS.append((R @ (fq - q), nw))
    return np.array(P), np.array(T), NS


def free_dome(obj, ref, cont="G2", height=1.0, fullness=1.0):
    """A dome over any planar rim: u = the profile (rim -> top), v = the rim's
    own curve (its CVs, knots and weights). Row 0 is the rim itself; the next
    one (G1) or two (G2) rows are fitted on the rim's basis to leave it along
    the surface (and with its curvature); the top rows are the rim scaled
    towards the top point, so the top is smooth."""
    from .kernel.classa import basis, insert_knot
    sm, ext = _k()
    kc = features.input_curve(obj, ref)
    cv = ext.curve_cvs(kc)
    p = int(cv["degree"])
    U = [float(k) for k in cv["knots"]]
    Q = np.array(cv["points"], float)
    w = np.array(cv["weights"], float) if cv.get("weights") else np.ones(len(Q))
    if any(abs(U[i] - U[0]) > 1e-12 or abs(U[-1 - i] - U[-1]) > 1e-12 for i in range(p + 1)):
        raise RuntimeError("the rim's curve is not clamped")
    size = float(np.linalg.norm(np.ptp(Q, axis=0))) or 1.0
    if np.linalg.norm(Q[0] - Q[-1]) > 1e-6 * size:
        raise RuntimeError("the rim is not closed")
    S = np.array([_vec(kc.evaluate(t)) for t in np.linspace(U[0], U[-1], 200, endpoint=False)])
    m0 = S.mean(axis=0)
    _, _, vt = np.linalg.svd(S - m0)
    a = vt[2]
    if float(np.abs((S - m0) @ a).max()) > 1e-3 * size:
        raise RuntimeError("the rim is not planar: use Patch for this opening")
    c = m0
    rho = np.linalg.norm((S - c) - np.outer((S - c) @ a, a), axis=1)
    rbar = float(rho.mean())
    Pw = np.c_[Q * w[:, None], w]
    m = 3 if cont == "G1" else 5
    tol = 1e-5 * size
    last = None
    for _ in range(12):
        br = sorted(set(U))
        ts = np.concatenate([np.linspace(x, y, 12, endpoint=False) for x, y in zip(br[:-1], br[1:])])
        P, T, NS = _across(obj, kc, ts, size, rbar)
        if _ == 0 and float(np.mean(T @ a)) < 0:
            a = -a
        tz = T @ a
        if tz.min() < 0.05:
            raise RuntimeError("the surface does not run up into the rim all round")
        inpl = (P - c) - np.outer((P - c) @ a, a)
        rr = np.linalg.norm(inpl, axis=1)
        ur = np.einsum("ij,ij->i", T, inpl / rr[:, None])
        H = max(height, 0.02) * float(np.mean([round_height(r_, (x_, z_)) for r_, x_, z_ in zip(rr, ur, tz)]))
        A = c + H * a
        X = P + (H - (P - c) @ a)[:, None] / tz[:, None] * T
        L0 = np.linalg.norm(X - P, axis=1)
        Xr = np.maximum(np.einsum("ij,ij->i", X - c, inpl / rr[:, None]), 0.25 * rr)
        kt = np.array([arc_k((x_, z_)) for x_, z_ in zip(ur, tz)]) * fullness
        lam = float(np.mean(kt * Xr / rr))
        k = kt
        targets = []
        if m == 3:
            targets.append(P + (k * L0)[:, None] * T)
            top = [lam]
        else:
            # the round arc as degree 5 (see _g2_rows), the rim end made G2
            al = 0.6 * k * L0
            er = inpl / rr[:, None]
            n2 = -tz[:, None] * er + np.einsum("ij,ij->i", T, er)[:, None] * a
            n2 /= np.linalg.norm(n2, axis=1)[:, None]
            kap = np.array([2.0 * float(d @ nn) / float(d @ d) for (d, _), nn in zip(NS, n2)])
            C1 = P + (k * L0)[:, None] * T
            C2 = A + (k * Xr)[:, None] * er
            targets.append(P + al[:, None] * T)
            targets.append(P + 2.0 * al[:, None] * T + (kap * m * al * al / (m - 1))[:, None] * n2)
            targets.append(0.3 * C1 + 0.6 * C2 + 0.1 * A)
            top = [0.6 * lam]
        B = basis(U, p, ts)
        wv = Pw[:, 3]
        Rb = B * wv[None, :] / (B @ wv)[:, None]
        Rf = Rb[:, :-1].copy()
        Rf[:, 0] += Rb[:, -1]
        rows, worst = [], np.zeros(len(ts))
        for tg in targets:
            Xc = np.linalg.lstsq(Rf, tg, rcond=None)[0]
            Xc = np.vstack([Xc, Xc[:1]])
            rows.append(Xc)
            worst = np.maximum(worst, np.linalg.norm(Rb @ Xc - tg, axis=1))
        err = float(worst.max())
        # done: within tolerance, or no longer getting better (the surface itself
        # turns a little at its own seam: more knots only chase that), or out of
        # room -- the rows always match the knots they were fitted on
        if err <= tol or (last is not None and err > 0.9 * last) or _ == 11 or len(Pw) > 400:
            break
        last = err
        spans = sorted({max(i for i, b in enumerate(br) if b <= t and i < len(br) - 1)
                        for t, e_ in zip(ts, worst) if e_ > tol})
        for i in spans:
            Pw, U = insert_knot(Pw, U, p, 0.5 * (br[i] + br[i + 1]))
        Pw = np.asarray(Pw, float)
    Qc = Pw[:, :3] / Pw[:, 3:]
    inq = (Qc - c) - np.outer((Qc - c) @ a, a)
    allrows = [Qc] + rows + [A + s_ * inq for s_ in top] + [np.repeat(A[None, :], len(Qc), axis=0)]
    pts = [tuple(float(x) for x in q) for row in allrows for q in row]
    wts = [float(x) for _ in allrows for x in Pw[:, 3]]
    rational = max(abs(x - 1.0) for x in wts) > 1e-12
    return ext.surface_from_cvs(m, p, pts, [0.0] * (m + 1) + [1.0] * (m + 1), [float(x) for x in U],
                                wts if rational else None)


# --------------------------------------------------------------------------
# draped: one even four-sided surface over the dome and a little past the
# rim, trimmed back to the rim (no pole; the CVs spread evenly)
# --------------------------------------------------------------------------

def draped_dome(obj, ref, polar, n=12, p=5, margin=1.12):
    """The polar dome (`polar`: u = rim -> top, v = round) as one even n x n
    degree-p surface laid over it and on past the rim, then trimmed to the rim.

    Layout: a rectangle in the rim's plane along its long and short axes,
    `margin` x the rim's extent. A point of it at the fraction s of the way
    from the centre out to the rim (in its direction) stands for the dome
    point s of the way along that direction's profile, measured by arc length
    -- not by height over the plane, which a steep wall would stretch to
    nothing. Past the rim (s > 1, the rectangle's outer band and corners) it
    stands for the wall carried on, with the dome's curvature there (what the
    trim then takes away). The net is fitted to that with the rim weighted
    and held to the wall's tangent plane."""
    from . import patch
    from .kernel.classa import basis
    sm, ext = _k()
    (pu0, pu1), (pv0, pv1) = polar.u_domain(), polar.v_domain()
    NU, NV = 41, 160
    us = np.linspace(pu0, pu1, NU)
    vs = np.linspace(pv0, pv1, NV, endpoint=False)
    D = np.array([[_vec(polar.evaluate(u, v)) for u in us] for v in vs])        # (NV, NU, 3), rim at u0
    rimP = D[:, 0]
    c = rimP.mean(axis=0)
    _, _, vt = np.linalg.svd(rimP - c)
    e1, a = vt[0], vt[2]
    if (D[:, -1].mean(axis=0) - c) @ a < 0:
        a = -a
    e2 = np.cross(a, e1)
    rx, ry = (rimP - c) @ e1, (rimP - c) @ e2
    phi = np.unwrap(np.arctan2(ry, rx))
    rho = np.hypot(rx, ry)
    if phi[-1] < phi[0]:                                                        # run the table counter-clockwise
        order = np.arange(NV)[::-1]
        phi, rho, D, rimP, vs = phi[order], rho[order], D[order], rimP[order], vs[order]
        rx, ry = rx[order], ry[order]
    phi0 = phi[0]
    seg = np.linalg.norm(np.diff(D, axis=1), axis=2)
    arc = np.concatenate([np.zeros((NV, 1)), np.cumsum(seg, axis=1)], axis=1)     # from the rim
    L = arc[:, -1:]
    s_of_u = 1.0 - arc / L                                                      # 1 at the rim, 0 at the top
    # the dome's way out of the rim and its curvature there, for the band past it
    der = [polar.derivatives(pu0, v) for v in vs]
    Du = np.array([_vec(d[1]) for d in der])
    sp = np.linalg.norm(Du, axis=1)
    tin = Du / sp[:, None]
    h = 1e-3 * (pu1 - pu0)
    Duu = (np.array([_vec(polar.derivatives(pu0 + h, v)[1]) for v in vs]) - Du) / h
    kv = (Duu - np.einsum("ij,ij->i", Duu, tin)[:, None] * tin) / (sp ** 2)[:, None]
    Nw = np.cross(tin, np.array([_vec(d[2]) for d in der]))
    Nw /= np.linalg.norm(Nw, axis=1)[:, None]
    X, Y = margin * np.abs(rx).max(), margin * np.abs(ry).max()

    def target(px, py):
        ang = np.arctan2(py, px)
        ang = phi0 + np.mod(ang - phi0, 2.0 * np.pi)
        j = np.searchsorted(phi, ang) - 1
        j = np.clip(j, 0, NV - 1)
        j2 = (j + 1) % NV
        span = np.where(j2 == 0, phi[0] + 2.0 * np.pi - phi[j], phi[j2] - phi[j])
        f = np.clip((ang - phi[j]) / np.maximum(span, 1e-12), 0.0, 1.0)
        rr = (1.0 - f) * rho[j] + f * rho[j2]
        sv = np.hypot(px, py) / np.maximum(rr, 1e-12)
        out = np.zeros((len(px), 3))
        for k in range(len(px)):
            pts_k = []
            for jj in (j[k], j2[k]):
                if sv[k] <= 1.0:
                    uu = np.interp(-sv[k], -s_of_u[jj], us)                    # s falls along the profile
                    pts_k.append(_vec(polar.evaluate(float(uu), float(vs[jj]))))
                else:
                    d = (sv[k] - 1.0) * L[jj, 0]
                    pts_k.append(rimP[jj] - d * tin[jj] + 0.5 * d * d * kv[jj])
            out[k] = (1.0 - f[k]) * pts_k[0] + f[k] * pts_k[1]
        return out

    G = 64
    gu = np.linspace(0.0, 1.0, G)
    UU, VV = np.meshgrid(gu, gu, indexing="ij")
    uu, vv = UU.reshape(-1), VV.reshape(-1)
    # and a dense band either side of the rim (its curvature there is the G2)
    band = np.concatenate([np.linspace(0.85, 0.995, 8), np.linspace(1.005, 1.12, 6)])
    bx = np.concatenate([f_ * rx for f_ in band])
    by = np.concatenate([f_ * ry for f_ in band])
    uu = np.concatenate([uu, 0.5 * (bx / X + 1.0)])
    vv = np.concatenate([vv, 0.5 * (by / Y + 1.0)])
    pts = target((2.0 * uu - 1.0) * X, (2.0 * vv - 1.0) * Y)
    nb = len(band) * NV
    # the rim itself, weighted, and the wall's tangent plane there
    ru, rv = 0.5 * (rx / X + 1.0), 0.5 * (ry / Y + 1.0)
    uu = np.concatenate([uu, ru])
    vv = np.concatenate([vv, rv])
    pts = np.vstack([pts, rimP])
    w = np.r_[np.ones(G * G), np.full(nb, 4.0), np.full(NV, 60.0 * G * G / NV / 20.0)]
    ku = list(np.r_[[0.0] * (p + 1), np.linspace(0, 1, n - p + 1)[1:-1], [1.0] * (p + 1)])
    A = (basis(ku, p, uu)[:, :, None] * basis(ku, p, vv)[:, None, :]).reshape(len(uu), n * n)
    nn = n * n
    Mx = A.T @ (A * w[:, None])
    Mbig = np.zeros((3 * nn, 3 * nn))
    rhs = np.zeros(3 * nn)
    for d in range(3):
        Mbig[d * nn:(d + 1) * nn, d * nn:(d + 1) * nn] = Mx
        rhs[d * nn:(d + 1) * nn] = A.T @ (w * pts[:, d])
    Gu = (basis(ku, p, ru, 1)[:, :, None] * basis(ku, p, rv)[:, None, :]).reshape(NV, nn)
    Gv = (basis(ku, p, ru)[:, :, None] * basis(ku, p, rv, 1)[:, None, :]).reshape(NV, nn)
    w1 = 20.0 * G * G / NV / 20.0
    for Gm in (Gu, Gv):
        for k in range(NV):
            g = np.concatenate([Nw[k, d] * Gm[k] for d in range(3)])
            Mbig += w1 * np.outer(g, g)
    # fairing: second differences of the net both ways (keeps the corners, which
    # only the trim takes away, from swinging out)
    rows = []
    for i in range(n):
        for j in range(n - 2):
            r1 = np.zeros(nn); r1[[i * n + j, i * n + j + 1, i * n + j + 2]] = (1.0, -2.0, 1.0); rows.append(r1)
            r2 = np.zeros(nn); r2[[j * n + i, (j + 1) * n + i, (j + 2) * n + i]] = (1.0, -2.0, 1.0); rows.append(r2)
    Dm = np.array(rows)
    lam = 1e-4 * float(np.trace(Mx)) / nn
    for d in range(3):
        Mbig[d * nn:(d + 1) * nn, d * nn:(d + 1) * nn] += lam * (Dm.T @ Dm)
    Xs = np.linalg.solve(Mbig, rhs)
    P = np.stack([Xs[d * nn:(d + 1) * nn] for d in range(3)], axis=1).reshape(n, n, 3)
    make = lambda Pg: ext.surface_from_cvs(p, p, [tuple(float(x) for x in q) for q in Pg.reshape(-1, 3)], ku, ku)
    surf = make(P)
    if features.edge_vote(surf, [(obj, ref)]) < 0:                              # face the way the wall does
        P = P[::-1].copy()
        surf = make(P)
    size = float(np.linalg.norm(np.ptp(rimP, axis=0))) or 1.0
    kc = features.input_curve(obj, ref)
    # trim with the rim itself (the net passes through it within the fit): a
    # copy re-interpolated through points pulled onto the net lost its region.
    # Which side a loop keeps follows its direction -- try both, keep the dome
    want = ext.brep_from_surface(polar).faces()[0].area()
    best, last = None, None
    tol = 1e-4 * size
    for rev in (False, True):
        cs = [patch._reverse(ext, kc)] if rev else [patch._copy_curve(ext, kc)]
        try:
            b = ext.trim_surface_with_3d_curves_tol(make(P), [1], cs, tol, 0.01)
            ar = sum(f.area() for f in b.faces())
        except Exception as exc:
            last = exc
            continue
        if ar > 0 and (best is None or abs(ar - want) < abs(best[0] - want)):
            best = (ar, b)
    if best is None or abs(best[0] - want) > 0.25 * want:
        raise RuntimeError(f"trimming the draped dome to the rim failed ({last or 'wrong region'})")
    return best[1]


@features.builder("dome")
def build_dome(inputs, p):
    obj, ref = inputs[0]
    cont = p.get("dome_cont", "G2")
    h, f = float(p.get("dome_height", 1.0)), float(p.get("dome_fullness", 1.0))
    rim = rim_data(obj, ref)
    polar = revolved(rim, profile(rim, cont, h, f)) if rim["round"] else free_dome(obj, ref, cont, h, f)
    if p.get("dome_layout", "POLAR") == "DRAPED":
        return draped_dome(obj, ref, polar, n=int(p.get("dome_cvs", 20)))
    return polar


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------

def _open_rims(context, objs=None):
    """Pick candidates that are open, closed edges (rims) of NURBS surfaces."""
    from .contcheck import edge_kind
    out = []
    for c in _pick_candidates(context, True):
        obj, ref, pts = c
        if ref.startswith("S") or not store.is_nurbs(obj) or (objs is not None and obj not in objs):
            continue
        if len(pts) < 3:
            continue
        L = sum((pts[i + 1] - pts[i]).length for i in range(len(pts) - 1)) or 1.0
        if (pts[0] - pts[-1]).length > 1e-4 * L:
            continue                               # not closed
        if ref.startswith("E") and edge_kind(obj, ref) != "open":
            continue
        if not ref.startswith("E"):
            # a natural surface's closed boundary: not the seam (its two ends lie on each other)
            P = np.array([tuple(p) for p in pts])
            if np.linalg.norm(np.ptp(P, axis=0)) < 1e-9:
                continue
        out.append(c)
    return out


class NPROTO_OT_dome(AskValue, PickMixin, bpy.types.Operator):
    """Close an open end (rim) of a surface with a dome, G1 or G2 to it. A round rim gets an
    exact surface of revolution; any other planar rim a dome on the rim's own curve. With one open
    rim on the selection it domes that; otherwise pick the rim (or pick its edge in Edge mode first)"""
    bl_idname = "nproto.dome"
    bl_label = "Dome"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 1
    pick_max = 1

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    dome_layout: EnumProperty(name="Layout", default="POLAR",
                              items=[("POLAR", "Polar", "Rows from the rim to one top point (a round rim: an exact revolution)"),
                                       ("DRAPED", "Draped", "One even four-sided surface over the opening, trimmed "
                                                            "to the rim (no pole: easy to edit)")])
    dome_cvs: IntProperty(name="CVs", default=20, min=8, max=24,
                          description="Draped: CVs in each direction of the surface")
    dome_cont: EnumProperty(name="Continuity", default="G2",
                            items=[("G1", "G1", "Tangent to the surface"),
                                   ("G2", "G2", "Tangent, with the surface's curvature carried on")])
    dome_height: FloatProperty(name="Height", default=1.0, min=0.05, soft_max=3.0,
                               description="1: round (the sphere that carries the surface on over the rim); "
                                           "less: flatter, more: taller")
    dome_fullness: FloatProperty(name="Fullness", default=1.0, min=0.3, max=2.0,
                                 description="How far the rim's direction carries into the dome (fuller shoulder "
                                             "and flatter top above 1)")

    def pick_candidates(self, context):
        return self._rims

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if self.inputs:                     # picked already: the settings, then the dome
            return self.ask(context)
        from . import subsel
        sel = [o for o in context.selected_objects if store.is_nurbs(o)]
        picked = subsel.selected_edges(context)
        rims = _open_rims(context, sel or None)
        if picked:
            hit = [c for c in rims if (c[0], c[1]) in picked]
            if hit:
                self.inputs = _encode_inputs([(hit[0][0], hit[0][1])])
                return self.ask(context)
        if sel and len(rims) == 1 and not event.alt:
            self.inputs = _encode_inputs([(rims[0][0], rims[0][1])])
            return self.ask(context)
        self._rims = rims or _open_rims(context)
        if not self._rims:
            self.report({'WARNING'}, "No open rim (a closed open edge) to dome")
            return {'CANCELLED'}
        return self.start_picking(context, event)

    def draw(self, context):
        lay = self.layout
        lay.row().prop(self, "dome_layout", expand=True)
        lay.row().prop(self, "dome_cont", expand=True)
        lay.prop(self, "dome_height")
        lay.prop(self, "dome_fullness")
        if self.dome_layout == "DRAPED":
            lay.prop(self, "dome_cvs")

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            ins = _decode_inputs(self.inputs)
        except Exception as exc:
            self.report({'ERROR'}, f"Dome: {exc}")
            return {'CANCELLED'}
        if len(ins) != 1:
            self.report({'WARNING'}, "Pick one open rim")
            return {'CANCELLED'}
        params = {"dome_layout": self.dome_layout, "dome_cont": self.dome_cont, "dome_height": float(self.dome_height),
                  "dome_fullness": float(self.dome_fullness), "dome_cvs": int(self.dome_cvs)}
        try:
            rim = rim_data(*ins[0])
            obj = _make_result(context, "Dome", "dome", ins, params)
        except Exception as exc:
            self.report({'ERROR'}, f"Dome failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        kind = "draped, trimmed" if self.dome_layout == "DRAPED" else (
            "exact revolution" if rim['round'] else "on the rim curve")
        self.report({'INFO'}, f"Dome: {kind}, {self.dome_cont}")
        return {'FINISHED'}


classes = (NPROTO_OT_dome,)
