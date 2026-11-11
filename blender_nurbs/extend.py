"""Extend / shrink a surface at one boundary, keeping its flow (Alias "Natural").

Exact: the surface's last span is continued as the SAME polynomial (rational:
the same conic / rational patch). With knots U, degree p and new end
parameter b', the new knot vector is U[0..k] + [b']*(p+1) (k = span holding b';
the last span when extending) and every CV whose knot window touches b' becomes
the blossom of span k's piece at that window. The old part of the surface stays
identical; extending past the end is the analytic continuation of the last
span, a negative distance cuts the surface back exactly.
"""

import numpy as np

from . import store
from .kernel import load as _load

EDGES = ("U0", "U1", "V0", "V1")


def _deboor(Pw, U, p, k, args):
    """Blossom of span k's polynomial piece at `args` (len p); with all args
    equal to t this is the de Boor point of that piece at t (also outside the span)."""
    d = [np.array(Pw[k - p + j], dtype=float) for j in range(p + 1)]
    for r in range(1, p + 1):
        t = args[r - 1]
        for j in range(p, r - 1, -1):
            i = k - p + j
            den = U[i + p + 1 - r] - U[i]
            a = (t - U[i]) / den if den != 0 else 0.0
            d[j] = (1.0 - a) * d[j - 1] + a * d[j]
    return d[p]


def _span(U, p, n, t):
    """Index k (p <= k <= n) with U[k] <= t < U[k+1]; the last span for t >= end."""
    if t >= U[n]:
        return n
    k = p
    while k < n and U[k + 1] <= t:
        k += 1
    return k


def end_curve(Pw, U, p, b_new):
    """Move the END of a B-spline (homogeneous CVs Pw, clamped knots U) to b_new."""
    n = len(Pw) - 1
    if b_new <= U[p] + 1e-12:
        raise ValueError("cannot shrink past the other end")
    k = _span(U, p, n, b_new)
    while k > p and U[k] >= b_new:      # b_new on a knot: end the previous span there
        k -= 1
    Un = list(U[:k + 1]) + [b_new] * (p + 1)
    Q = []
    for i in range(k + 1):
        if i + p <= k:
            Q.append(np.array(Pw[i], dtype=float))
        else:
            Q.append(_deboor(Pw, U, p, k, Un[i + 1:i + p + 1]))
    return Q, Un


def curve_point(Pw, U, p, t):
    k = _span(U, p, len(Pw) - 1, t)
    q = _deboor(Pw, U, p, k, [t] * p)
    return q[:3] / q[3]


def _grid(d):
    """(nu, nv, 4) homogeneous CV grid from surface_cvs data."""
    nu, nv = d["count_u"], d["count_v"]
    P = np.array(d["points"], dtype=float).reshape(nu, nv, 3)
    w = np.array(d["weights"], dtype=float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    return np.concatenate([P * w[..., None], w[..., None]], axis=2)


def _extend_u_end(G, Uu, du, Uv, dv, distance):
    """Extend / shrink the u = end boundary of grid G by `distance` (world units
    along the surface, averaged over a few v positions)."""
    nu = G.shape[0]
    b = Uu[nu]
    vs = [Uv[dv] + (Uv[len(Uv) - dv - 1] - Uv[dv]) * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    # u-direction curves at a few v: collapse the grid along v (homogeneous)
    rows = []
    for v in vs:
        kv = _span(Uv, dv, G.shape[1] - 1, v)
        rows.append([_deboor(G[i], Uv, dv, kv, [v] * dv) for i in range(nu)])

    def length(t):   # signed arc length from b to t along the rows (averaged)
        tot = 0.0
        ts = np.linspace(b, t, 65)
        for R in rows:
            pts = np.array([curve_point(R, Uu, du, x) for x in ts])
            tot += np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        return np.sign(t - b) * tot / len(rows)

    span0 = Uu[nu] - Uu[du]
    if abs(distance) < 1e-12:
        return G, list(Uu)
    if distance > 0:
        lo, hi = b, b + span0
        while length(hi) < distance and hi - b < 1e3 * span0:
            hi = b + 2 * (hi - b)
    else:
        if -distance >= abs(length(Uu[du] + 1e-9 * span0)):
            raise ValueError("shrink distance is longer than the surface")
        lo, hi = Uu[du] + 1e-9 * span0, b
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if length(mid) < distance:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    cols = [end_curve([G[i, j] for i in range(nu)], Uu, du, t) for j in range(G.shape[1])]
    Un = cols[0][1]
    Gn = np.array([[cols[j][0][i] for j in range(G.shape[1])] for i in range(len(cols[0][0]))])
    return Gn, Un


def _flip_u(G, Uu):
    a, b = Uu[0], Uu[-1]
    return G[::-1].copy(), [a + b - x for x in reversed(Uu)]


def extend_surface_text(text, edge, distance):
    """Brep text of the single untrimmed surface extended at `edge` (U0/U1/V0/V1)."""
    ext = _load().blender_ext
    brep = store.brep_from_text(text)
    if brep.face_count() != 1:
        raise RuntimeError("only single surfaces can be extended (Unjoin first)")
    d = ext.surface_cvs(brep.faces()[0].surface())
    G = _grid(d)
    Uu, Uv = list(d["knots_u"]), list(d["knots_v"])
    du, dv = d["degree_u"], d["degree_v"]
    if edge in ("V0", "V1"):              # work in u: transpose
        G, Uu, Uv, du, dv = G.transpose(1, 0, 2).copy(), Uv, Uu, dv, du
    start = edge in ("U0", "V0")
    if start:
        G, Uu = _flip_u(G, Uu)
    G, Uu = _extend_u_end(G, Uu, du, Uv, dv, distance)
    if start:
        G, Uu = _flip_u(G, Uu)
    if edge in ("V0", "V1"):
        G, Uu, Uv, du, dv = G.transpose(1, 0, 2).copy(), Uv, Uu, dv, du
    # normalise knots to [0, 1]
    norm = lambda K: [(x - K[0]) / (K[-1] - K[0]) for x in K]
    w = G[..., 3]
    pts = [tuple(map(float, G[i, j, :3] / w[i, j])) for i in range(G.shape[0]) for j in range(G.shape[1])]
    rational = np.abs(w - 1.0).max() > 1e-12
    weights = [float(x) for x in w.reshape(-1)] if rational else None
    s = ext.surface_from_cvs(du, dv, pts, norm(Uu), norm(Uv), weights)
    return store.brep_to_text(ext.brep_from_surface(s))


def apply_extend_surface(obj, text, inputs, p):
    return extend_surface_text(text, p["edge"], float(p["distance"]))


def _register_step():
    from . import steps
    steps.APPLY["extend_surface"] = apply_extend_surface
    steps.LABEL["extend_surface"] = lambda p: f"Extend · {p['edge']} {p['distance']:+.3g}"


_register_step()
