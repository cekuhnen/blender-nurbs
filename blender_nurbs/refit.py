"""Refit to Trim: turn a trimmed, four-sided face into an untrimmed NURBS
surface whose four edges ARE the trim boundary and whose shape follows the
original surface (Alias "untrim to boundary" / Rhino-style rebuild-to-edges).

The outer trim loop is split into 4 sides at its corners. Their UV images on
the old surface span a Coons patch in parameter space; the old surface is
evaluated through it, giving points exactly on the original shape, which are
then fitted with a clean tensor-product net (degree / spans, auto spans to a
tolerance). Only the four-sided outer loop is used (no holes).
"""

import numpy as np

from . import geom, store


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


CORNER_DEG = 25.0


def _edge_polyline(edge, n=64):
    t0, t1 = edge.parameter_range()
    return np.array([_point_at(edge, t, t0, t1) for t in np.linspace(t0, t1, n)], float)


def _point_at(edge, t, t0, t1):
    """edge.point_at(t), stepping a hair inside the range if the kernel calls
    an end parameter out of range (its float rounding differs between the
    Windows / Linux / macOS builds)."""
    try:
        return edge.point_at(t)
    except Exception:
        h = (t1 - t0) or 1.0
        for k in (1e-12, 1e-10, 1e-8):
            try:
                return edge.point_at(min(max(t, t0 + k * h), t1 - k * h))
            except Exception:
                continue
        raise


def _chain(polys, tol):
    """Chain edge polylines into one closed loop (orienting them)."""
    loop = [polys[0]]
    rest = list(range(1, len(polys)))
    while rest:
        end = loop[-1][-1]
        k = min(rest, key=lambda i: min(np.linalg.norm(polys[i][0] - end), np.linalg.norm(polys[i][-1] - end)))
        p = polys[k]
        if np.linalg.norm(p[0] - end) > np.linalg.norm(p[-1] - end):
            p = p[::-1]
        if np.linalg.norm(p[0] - end) > tol:
            raise RuntimeError("the trim boundary is not one closed loop")
        loop.append(p)
        rest.remove(k)
    return loop


def _four_sides(loop):
    """Split a chained loop of edge polylines into 4 sides at its corners."""
    n = len(loop)
    turn = []
    for i in range(n):
        a = loop[i - 1][-1] - loop[i - 1][-4]
        b = loop[i][3] - loop[i][0]
        c = float(a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
        turn.append(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))
    if n == 4:
        starts = [0, 1, 2, 3]
    else:
        starts = sorted(range(n), key=lambda i: -turn[i])[:4]
        if len(starts) < 4 or min(turn[i] for i in starts) < CORNER_DEG:
            raise RuntimeError(f"the trim boundary has {sum(t >= CORNER_DEG for t in turn)} corners -- "
                               "Refit to Trim needs a four-sided boundary")
        starts.sort()
    sides = []
    for j in range(4):
        a, b = starts[j], starts[(j + 1) % 4]
        idx = list(range(a, b)) if a < b else list(range(a, n)) + list(range(0, b))
        sides.append(np.vstack([loop[i] if k == 0 else loop[i][1:] for k, i in enumerate(idx)]))
    return sides


def _resample(P, m):
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0, d[-1], m)
    return np.array([np.interp(t, d, P[:, k]) for k in range(3)]).T


def boundary_uv(surf, sides, m):
    """3D samples of the 4 sides (bottom, right, top, left; bottom/top along +s,
    left/right along +t) and their UV images on the surface."""
    from .square import _foot
    s0, s1, s2, s3 = sides
    oriented = [s0, s1, s2[::-1], s3[::-1]]
    P3, UV = [], []
    prev = None
    for side in oriented:
        Q = _resample(side, m)
        uv = []
        for q in Q:
            prev = _foot(surf, q, prev)
            uv.append(prev)
        P3.append(Q)
        UV.append(np.array(uv))
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    for uv in UV:
        d = np.abs(np.diff(uv, axis=0))
        if (d[:, 0] > 0.5 * (u1 - u0)).any() or (d[:, 1] > 0.5 * (v1 - v0)).any():
            raise RuntimeError("the trim boundary crosses the surface seam")
    return P3, UV


def refit_points(surf, UV, m):
    """Points on the old surface over a Coons patch in its UV space (m x m)."""
    B, R, T, L = UV
    s = np.linspace(0, 1, m)
    G = np.zeros((m, m, 3))
    c00, c10, c01, c11 = B[0], B[-1], T[0], T[-1]
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    for i, si in enumerate(s):
        for j, tj in enumerate(s):
            uv = ((1 - tj) * B[i] + tj * T[i] + (1 - si) * L[j] + si * R[j]
                  - ((1 - si) * (1 - tj) * c00 + si * (1 - tj) * c10 + (1 - si) * tj * c01 + si * tj * c11))
            u = min(max(uv[0], u0), u1)
            v = min(max(uv[1], v0), v1)
            G[i, j] = surf.evaluate(u, v)
    return G


def fit_grid(G, p, su, sv):
    """Least squares tensor-product fit of an (m x m) grid at uniform params."""
    m = G.shape[0]
    t = np.linspace(0, 1, m)
    nu, nv = p + su, p + sv
    Uu, Uv = geom.clamped_uniform_knots(nu, p), geom.clamped_uniform_knots(nv, p)
    Au, Av = geom.interp_matrix(t, Uu, p, nu), geom.interp_matrix(t, Uv, p, nv)
    Xv = np.array([np.linalg.lstsq(Av, G[i], rcond=None)[0] for i in range(m)])          # (m, nv, 3)
    X = np.array([np.linalg.lstsq(Au, Xv[:, j, :], rcond=None)[0] for j in range(nv)])    # (nv, nu, 3)
    X = X.transpose(1, 0, 2)                                                             # (nu, nv, 3)
    F = np.einsum("ia,abk,jb->ijk", Au, X, Av)
    dev = float(np.linalg.norm(F - G, axis=2).max())
    return X, Uu, Uv, dev


def refit_text(text, degree=3, spans_u=0, spans_v=0, tol_rel=1e-4, samples=81):
    """Brep text of the untrimmed refit. Returns (text, info)."""
    _, ext = _k()
    brep = store.brep_from_text(text)
    if brep.face_count() != 1:
        raise RuntimeError("Refit to Trim works on single trimmed surfaces")
    face = brep.faces()[0]
    surf = face.surface()
    polys = [_edge_polyline(e) for e in face.edges()]
    allp = np.vstack(polys)
    size = float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0
    loop = _chain(polys, 1e-3 * size)
    sides = _four_sides(loop)
    m = int(samples)
    _, UV = boundary_uv(surf, sides, m)
    G = refit_points(surf, UV, m)
    tol = tol_rel * size
    p = int(degree)
    t = np.linspace(0, 1, m)

    def fit1d(rows, spans):
        n = p + spans
        U = geom.clamped_uniform_knots(n, p)
        A = geom.interp_matrix(t, U, p, n)
        B = np.hstack(rows)
        X = np.linalg.lstsq(A, B, rcond=None)[0]
        return float(np.linalg.norm((A @ X - B).reshape(m, -1, 3), axis=2).max())

    # each direction on its own: the fewest spans whose 1D fits stay within tol/2
    def pick(rows, fixed):
        if fixed:
            return fixed
        for s_ in range(1, max(2, m // 2 - p)):
            if fit1d(rows, s_) <= 0.5 * tol:
                return s_
        return max(1, m // 2 - p)
    sv = pick([G[i] for i in range(m)], spans_v)                  # along v, for every u row
    su = pick([G[:, j] for j in range(m)], spans_u)               # along u, for every v column
    X, Uu, Uv, dev = fit_grid(G, p, su, sv)
    nu, nv = X.shape[:2]
    pts = [tuple(map(float, X[i, j])) for i in range(nu) for j in range(nv)]
    s = ext.surface_from_cvs(p, p, pts, Uu, Uv)
    info = {"cvs": (nu, nv), "spans": (su, sv), "degree": p, "deviation": dev, "size": size}
    return store.brep_to_text(ext.brep_from_surface(s)), info


LAST_INFO = {}


def apply_refit_trim(obj, text, inputs, p):
    out, info = refit_text(text, int(p.get("refit_degree", 3)), int(p.get("refit_spans_u", 0)),
                           int(p.get("refit_spans_v", 0)))
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return out


def _register_step():
    from . import steps
    steps.APPLY["refit_trim"] = apply_refit_trim
    steps.LABEL["refit_trim"] = lambda p: (f"Refit to Trim · degree {p.get('refit_degree', 3)}, spans "
                                           f"{p.get('refit_spans_u', 0) or 'auto'} x {p.get('refit_spans_v', 0) or 'auto'}")


_register_step()
