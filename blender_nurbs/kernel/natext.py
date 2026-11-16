"""Natural (exact) extension of B-spline curves and surfaces, numpy only.

Shared by Extend Surface (extend.py, in Blender) and the worker (cutters that
must reach across the surface they split). The last span is continued as the
SAME polynomial (rational: the same conic / rational patch): with knots U,
degree p and new end parameter b', the knot vector becomes U[0..k] + [b']*(p+1)
(k = span holding b') and every CV whose knot window touches b' becomes the
blossom of span k's piece at that window. The old part stays identical.
"""

import numpy as np


def deboor(Pw, U, p, k, args):
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


def span(U, p, n, t):
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
    k = span(U, p, n, b_new)
    while k > p and U[k] >= b_new:      # b_new on a knot: end the previous span there
        k -= 1
    Un = list(U[:k + 1]) + [b_new] * (p + 1)
    Q = []
    for i in range(k + 1):
        if i + p <= k:
            Q.append(np.array(Pw[i], dtype=float))
        else:
            Q.append(deboor(Pw, U, p, k, Un[i + 1:i + p + 1]))
    return Q, Un


def curve_point(Pw, U, p, t):
    k = span(U, p, len(Pw) - 1, t)
    q = deboor(Pw, U, p, k, [t] * p)
    return q[:3] / q[3]


def end_parameter(rows, U, p, distance):
    """End parameter that moves the end of the curves `rows` (homogeneous CV
    lists sharing knots U) by `distance` along them (arc length, averaged;
    negative: shrink)."""
    n = len(rows[0])
    b = U[n]

    def length(t):   # signed arc length from b to t along the rows (averaged)
        tot = 0.0
        ts = np.linspace(b, t, 65)
        for R in rows:
            pts = np.array([curve_point(R, U, p, x) for x in ts])
            tot += np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        return np.sign(t - b) * tot / len(rows)

    span0 = U[n] - U[p]
    if distance > 0:
        lo, hi = b, b + span0
        while length(hi) < distance and hi - b < 1e3 * span0:
            hi = b + 2 * (hi - b)
    else:
        if -distance >= abs(length(U[p] + 1e-9 * span0)):
            raise ValueError("shrink distance is longer than the surface")
        lo, hi = U[p] + 1e-9 * span0, b
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if length(mid) < distance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def insert_knot(Pw, U, p, t):
    """Boehm: knot t inserted once into the curve (homogeneous CVs Pw). Exact."""
    k = max(i for i in range(len(U) - 1) if U[i] <= t < U[i + 1])
    Q = []
    for i in range(len(Pw) + 1):
        if i <= k - p:
            Q.append(np.array(Pw[i], dtype=float))
        elif i > k:
            Q.append(np.array(Pw[i - 1], dtype=float))
        else:
            a = (t - U[i]) / (U[i + p] - U[i]) if U[i + p] != U[i] else 0.0
            Q.append((1 - a) * np.array(Pw[i - 1], dtype=float) + a * np.array(Pw[i], dtype=float))
    return Q, list(U[:k + 1]) + [t] + list(U[k + 1:])


def _last_span(U, p, n):
    """(start, end) of the last non-empty span of a curve with n CVs."""
    b = U[n]
    a = max(x for x in U[p:n + 1] if x < b)
    return a, b


def _arc(rows, U, p, a, b):
    tot = 0.0
    for R in rows:
        pts = np.array([curve_point(R, U, p, x) for x in np.linspace(a, b, 33)])
        tot += np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    return tot / len(rows)


def extend_rows(rows, U, p, distance, chunks=False):
    """The curves `rows` (homogeneous CV lists sharing knots U) extended at
    their end by `distance`. chunks: grow at most one last-span length at a
    time, a knot inserted in the middle of the last span before each step, so
    no span gets long: a rational arc keeps each span well under 180 deg (one
    rational span cannot reach that) and a polynomial is not continued far
    from where it was fitted. Exact either way; chunks adds knots."""
    rows = [list(R) for R in rows]
    U = list(U)
    left = float(distance)
    for _ in range(64):
        if left <= 1e-12:
            break
        step = left
        if chunks and distance > 0:       # halve the last span, grow it back to its old length at most
            a, b = _last_span(U, p, len(rows[0]))
            new = [insert_knot(R, U, p, 0.5 * (a + b)) for R in rows]
            rows, U = [q for q, _ in new], new[0][1]
            step = min(left, max(_arc(rows, U, p, 0.5 * (a + b), b), 0.02 * distance))
        t = end_parameter(rows, U, p, step)
        new = [end_curve(R, U, p, t) for R in rows]
        rows, U = [q for q, _ in new], new[0][1]
        left -= step
        if not chunks or distance <= 0:
            break
    return rows, U


def flip(G, U):
    """Reverse the first direction of a CV array (and its knots)."""
    a, b = U[0], U[-1]
    return G[::-1].copy(), [a + b - x for x in reversed(U)]


def extend_u_end(G, Uu, du, Uv, dv, distance, chunks=False):
    """Extend / shrink the u = end boundary of the homogeneous grid G (nu, nv, 4)
    by `distance` (world units along the surface, averaged over a few v).
    chunks: see extend_rows."""
    if abs(distance) < 1e-12:
        return G, list(Uu)
    nu, nv = G.shape[0], G.shape[1]
    vs = [Uv[dv] + (Uv[len(Uv) - dv - 1] - Uv[dv]) * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    rows = []        # u-direction curves at a few v: the grid collapsed along v
    for v in vs:
        kv = span(Uv, dv, nv - 1, v)
        rows.append([deboor(G[i], Uv, dv, kv, [v] * dv) for i in range(nu)])
    if not chunks:
        t = end_parameter(rows, Uu, du, distance)
        cols = [end_curve([G[i, j] for i in range(nu)], Uu, du, t) for j in range(nv)]
        Un = cols[0][1]
        Gn = np.array([[cols[j][0][i] for j in range(nv)] for i in range(len(cols[0][0]))])
        return Gn, Un
    # the sample rows steer the steps; the CV columns follow them exactly
    allrows = rows + [[G[i, j] for i in range(nu)] for j in range(nv)]
    U = list(Uu)
    left = float(distance)
    for _ in range(64):
        if left <= 1e-12:
            break
        step = left
        if distance > 0:                  # halve the last span, grow it back to its old length at most
            a, b = _last_span(U, du, len(allrows[0]))
            new = [insert_knot(R, U, du, 0.5 * (a + b)) for R in allrows]
            allrows, U = [q for q, _ in new], new[0][1]
            step = min(left, max(_arc(allrows[:len(rows)], U, du, 0.5 * (a + b), b), 0.02 * distance))
        t = end_parameter(allrows[:len(rows)], U, du, step)
        new = [end_curve(R, U, du, t) for R in allrows]
        allrows, U = [q for q, _ in new], new[0][1]
        left -= step
        if distance <= 0:
            break
    cols = allrows[len(rows):]
    Gn = np.array([[cols[j][i] for j in range(nv)] for i in range(len(cols[0]))])
    return Gn, U


def _norm(K):
    return [(x - K[0]) / (K[-1] - K[0]) for x in K]


def extend_grid(d, edge, distance, chunks=False):
    """surface_cvs data `d` extended by `distance` at `edge` (U0/U1/V0/V1).
    Returns (degree_u, degree_v, points, knots_u, knots_v, weights or None),
    the arguments of surface_from_cvs, knots normalised to [0, 1]."""
    nu, nv = d["count_u"], d["count_v"]
    P = np.array(d["points"], dtype=float).reshape(nu, nv, 3)
    w = np.array(d["weights"], dtype=float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    G = np.concatenate([P * w[..., None], w[..., None]], axis=2)
    Uu, Uv = list(d["knots_u"]), list(d["knots_v"])
    du, dv = d["degree_u"], d["degree_v"]
    if edge in ("V0", "V1"):              # work in u: transpose
        G, Uu, Uv, du, dv = G.transpose(1, 0, 2).copy(), Uv, Uu, dv, du
    start = edge in ("U0", "V0")
    if start:
        G, Uu = flip(G, Uu)
    G, Uu = extend_u_end(G, Uu, du, Uv, dv, distance, chunks)
    if start:
        G, Uu = flip(G, Uu)
    if edge in ("V0", "V1"):
        G, Uu, Uv, du, dv = G.transpose(1, 0, 2).copy(), Uv, Uu, dv, du
    w = G[..., 3]
    pts = [tuple(map(float, G[i, j, :3] / w[i, j])) for i in range(G.shape[0]) for j in range(G.shape[1])]
    rational = np.abs(w - 1.0).max() > 1e-12
    return du, dv, pts, _norm(Uu), _norm(Uv), ([float(x) for x in w.reshape(-1)] if rational else None)


def extend_curve(d, at_start, distance, chunks=False):
    """curve_cvs-style data {degree, points, knots, weights} extended by
    `distance` (arc length) at its start or end (chunks: see extend_rows).
    Knots normalised to [0, 1]."""
    p = int(d["degree"])
    P = np.array(d["points"], dtype=float)
    w = np.array(d["weights"], dtype=float) if d.get("weights") else np.ones(len(P))
    G = np.concatenate([P * w[:, None], w[:, None]], axis=1)
    U = [float(k) for k in d["knots"]]
    if at_start:
        G, U = flip(G, U)
    rows, U = extend_rows([list(G)], U, p, distance, chunks)
    G = np.array(rows[0])
    if at_start:
        G, U = flip(G, U)
    w = G[:, 3]
    rational = np.abs(w - 1.0).max() > 1e-12
    return {"degree": p, "points": [list(map(float, g[:3] / g[3])) for g in G], "knots": _norm(U),
            "weights": [float(x) for x in w] if rational else None}
