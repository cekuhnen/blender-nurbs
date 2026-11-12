"""Patch (XNURBS-style N-sided fill): one trimmed NURBS surface filling a
closed loop of any number of curves / surface edges, G0 or G1 per edge.

  1. the inputs are sampled (positions; neighbour normals for surface edges)
     and chained head to tail into one loop
  2. four corners split the loop into four chains (the sharpest corners whose
     Coons patch bends least); each chain is laid out in (u, v) between the
     square's corners with its turning within the surface (layout_quad), so
     a corner inside a chain bulges instead of creasing the surface -- the
     net runs evenly along the opening (solve_quad)
  3. one least-squares solve for an even n x n net:
        G0  the position at every edge sample
        G1  the neighbour's tangent plane (N . S_u = N . S_v = 0)
        fairness: thin-plate energy over the whole net
     then twice more with the fairness measured on the surface just solved
     (_geo_gram), so the layout's bends don't show in the shape
  4. the inputs, projected onto the surface, trim it to the loop.
The normal is turned to agree with the neighbours' faces. Older patches keep
their method: "DEVELOP" (loop laid out flat, solve3d) or "PROJECT" (height
field over the opening's plane, solve).
"""

import numpy as np

from . import features, geom

CONT = {"G0": 0, "G1": 1, "G2": 2}


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


# --------------------------------------------------------------------------
# B-spline basis with derivatives (Piegl & Tiller A2.3)
# --------------------------------------------------------------------------

def ders_basis(span, t, p, U, nd=2):
    ndu = np.zeros((p + 1, p + 1))
    ndu[0, 0] = 1.0
    left, right = np.zeros(p + 1), np.zeros(p + 1)
    for j in range(1, p + 1):
        left[j] = t - U[span + 1 - j]
        right[j] = U[span + j] - t
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            tmp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        ndu[j, j] = saved
    ders = np.zeros((nd + 1, p + 1))
    ders[0] = ndu[:, p]
    a = np.zeros((2, p + 1))
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, nd + 1):
            d = 0.0
            rk, pk = r - k, p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1
    f = p
    for k in range(1, nd + 1):
        ders[k] *= f
        f *= p - k
    return ders


def find_spans(n, p, t, U):
    """geom.find_span for many parameters t at once (n = last CV index)."""
    return np.clip(np.searchsorted(U, t, side="right") - 1, p, n)


def ders_basis_many(span, t, p, U, nd=2):
    """ders_basis at many parameters at once: span, t (m,) -> (m, nd + 1,
    p + 1). The same arithmetic in the same order, on arrays: equal to the
    last bit."""
    t = np.asarray(t, float)
    span = np.asarray(span, int)
    U = np.asarray(U, float)
    m = len(t)
    ndu = np.zeros((p + 1, p + 1, m))
    ndu[0, 0] = 1.0
    left, right = np.zeros((p + 1, m)), np.zeros((p + 1, m))
    for j in range(1, p + 1):
        left[j] = t - U[span + 1 - j]
        right[j] = U[span + j] - t
        saved = np.zeros(m)
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            tmp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        ndu[j, j] = saved
    ders = np.zeros((nd + 1, p + 1, m))
    ders[0] = ndu[:, p]
    a = np.zeros((2, p + 1, m))
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, nd + 1):
            d = np.zeros(m)
            rk, pk = r - k, p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d = d + a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d = d + a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1
    f = p
    for k in range(1, nd + 1):
        ders[k] *= f
        f *= p - k
    return ders.transpose(2, 0, 1)


PAIRS2 = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
PAIRS3 = ((3, 0), (2, 1), (1, 2), (0, 3))


def _dims(n, U):
    """(nu, nv, Uu, Uv) from a count n (square net, U both ways) or a pair
    (nu, nv) with knot vectors (Uu, Uv)."""
    if isinstance(n, (tuple, list)):
        return int(n[0]), int(n[1]), U[0], U[1]
    return int(n), int(n), U, U


def surface_rows(u, v, n, p, U, pairs=PAIRS2):
    """Basis rows (len nu*nv, CV (i, j) at i*nv + j) of the derivatives `pairs`
    ((du, dv) orders; default S, S_u, S_v, S_uu, S_uv, S_vv) at (u, v) for a
    net of degree p: n x n on knots U, or n = (nu, nv) on U = (Uu, Uv)."""
    nu, nv, Uu, Uv = _dims(n, U)
    su = geom.find_span(nu - 1, p, u, Uu)
    sv = geom.find_span(nv - 1, p, v, Uv)
    nd = max(max(a, b) for a, b in pairs)
    Du, Dv = ders_basis(su, u, p, Uu, nd), ders_basis(sv, v, p, Uv, nd)
    out = []
    for a, b in pairs:
        row = np.zeros(nu * nv)
        blk = np.outer(Du[a], Dv[b])
        for i in range(p + 1):
            row[(su - p + i) * nv + (sv - p):(su - p + i) * nv + sv + 1] = blk[i]
        out.append(row)
    return out


def surface_rows_many(uv, n, p, U, pairs=PAIRS2):
    """surface_rows at many (u, v) at once: (m, len(pairs), nu * nv), the
    same values."""
    nu, nv, Uu, Uv = _dims(n, U)
    uv = np.asarray(uv, float).reshape(-1, 2)
    m = len(uv)
    u, v = uv[:, 0], uv[:, 1]
    su, sv = find_spans(nu - 1, p, u, Uu), find_spans(nv - 1, p, v, Uv)
    nd = max(max(a, b) for a, b in pairs)
    Du, Dv = ders_basis_many(su, u, p, Uu, nd), ders_basis_many(sv, v, p, Uv, nd)
    k = np.arange(p + 1)
    cols = (su - p)[:, None, None] * nv + k[None, :, None] * nv + (sv - p)[:, None, None] + k[None, None, :]
    out = np.zeros((m, len(pairs), nu * nv))
    at = np.arange(m)[:, None, None]
    for q, (a, b) in enumerate(pairs):
        out[at, q, cols] = Du[:, a, :, None] * Dv[:, b, None, :]
    return out


def lstsq2(Su, Sv, r):
    """Per row k: the least-squares (a, c) of a Su[k] + c Sv[k] = r[k]
    (3-vectors), as np.linalg.lstsq (the same cut-off for a degenerate
    pair), for all rows at once: (m, 2)."""
    if not len(r):
        return np.zeros((0, 2))
    J = np.stack([Su, Sv], axis=-1)
    return (np.linalg.pinv(J, rtol=None) @ np.asarray(r, float)[..., None])[..., 0]


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def chain(sides):
    """[(index, reversed)] chaining the sides head to tail, and the closing gap."""
    order = [(0, False)]
    rest = list(range(1, len(sides)))
    while rest:
        k0, rv = order[-1]
        end = sides[k0].P[0] if rv else sides[k0].P[-1]
        k = min(rest, key=lambda i: min(np.linalg.norm(sides[i].P[0] - end), np.linalg.norm(sides[i].P[-1] - end)))
        rest.remove(k)
        order.append((k, bool(np.linalg.norm(sides[k].P[-1] - end) < np.linalg.norm(sides[k].P[0] - end))))
    k0, rv = order[-1]
    end = sides[k0].P[0] if rv else sides[k0].P[-1]
    start = sides[0].P[0]
    return order, float(np.linalg.norm(end - start))


def _self_intersects(Q):
    """Does the closed 2D polygon Q cross itself (non-adjacent segments)?"""
    A, B = Q, np.roll(Q, -1, axis=0)
    n = len(Q)
    d = B - A
    for i in range(n):
        j = np.arange(i + 2, n if i > 0 else n - 1)
        if not len(j):
            continue
        p, r = A[i], d[i]
        q, s = A[j], d[j]
        rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
        qp = q - p
        ok = np.abs(rxs) > 1e-15
        t = np.where(ok, (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / np.where(ok, rxs, 1), -1)
        w = np.where(ok, (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / np.where(ok, rxs, 1), -1)
        if np.any(ok & (t > 1e-9) & (t < 1 - 1e-9) & (w > 1e-9) & (w < 1 - 1e-9)):
            return True
    return False


def inside(Q, pts):
    """Even-odd test: which 2D points lie inside the closed polygon Q."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    a, b = Q, np.roll(Q, -1, axis=0)
    cond = (a[:, 1] > y) != (b[:, 1] > y)
    xi = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / np.where(b[:, 1] != a[:, 1], b[:, 1] - a[:, 1], 1e-30)
    return (np.sum(cond & (x < xi), axis=1) % 2) == 1


def frame(P):
    """Centre, unit normal (Newell) and in-plane axes of the closed loop P."""
    c = P.mean(axis=0)
    Q = P - c
    nrm = np.cross(Q, np.roll(Q, -1, axis=0)).sum(axis=0)
    if np.linalg.norm(nrm) < 1e-12:
        nrm = np.linalg.svd(Q, full_matrices=False)[2][2]
    nrm = nrm / np.linalg.norm(nrm)
    flat = Q - np.outer(Q @ nrm, nrm)
    e1 = np.linalg.svd(flat, full_matrices=False)[2][0]
    e1 = e1 - (e1 @ nrm) * nrm
    e1 /= np.linalg.norm(e1)
    return c, nrm, e1, np.cross(nrm, e1)


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------

def solve(sides, conts, n=20, p=3, fair=1.0, margin=0.08, energy=2, g1w=10.0, g0w=10.0, outside_w=0.01):
    """CV net (n, n, 3) and knots of the fill; sides are oriented Side objects
    (world space) in loop order, conts 0 / 1 per side.

    The patch is a height field over the plane the opening faces:
        S(x, y) = c + x e1 + y e2 + h(x, y) nrm
    (the in-plane part is exact with CVs at the Greville points), so
        G0  h = (P - c) . nrm
        G1  dh/dx = -(N . e1) / (N . nrm),  dh/dy = -(N . e2) / (N . nrm)
            (the neighbour's tangent plane as a slope)
        fairness  curvature variation of h (energy 3) or thin plate (2)
    -- all linear in the n*n height CVs and well conditioned."""
    loop = np.vstack([sd.P[:-1] for sd in sides])
    allP = np.vstack([sd.P for sd in sides])
    c, nrm, e1, e2 = frame(loop)
    xy = np.c_[(allP - c) @ e1, (allP - c) @ e2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    ext_ = float((hi - lo).max())
    lo = (lo + hi) / 2 - ext_ * (0.5 + margin)
    span = ext_ * (1 + 2 * margin)
    to_uv = lambda P: np.clip(np.c_[(P - c) @ e1 - lo[0], (P - c) @ e2 - lo[1]] / span, 0.0, 1.0)
    loop2d = to_uv(loop)
    if _self_intersects(loop2d[:: max(1, len(loop2d) // 200)]):
        raise RuntimeError("the opening folds over when seen from its average direction "
                           "(too curved for a projected patch)")
    U = geom.clamped_uniform_knots(n, p)
    N = n * n
    size = float(np.linalg.norm(np.ptp(allP, axis=0))) or 1.0
    n_g0 = sum(len(sd.P) for sd in sides)
    rows, rhs = [], []
    steep = 0.0
    for sd, cont in zip(sides, conts):
        uv = to_uv(sd.P)
        B = [surface_rows(a, b, n, p, U) for a, b in uv]
        w = 1.0 / np.sqrt(n_g0)
        rows.append(g0w * w * np.array([b[0] for b in B]))
        rhs.append(g0w * w * ((sd.P - c) @ nrm) / size)
        if cont >= 1 and sd.has_surface:
            Nn = np.asarray(sd.N, float)
            dn = Nn @ nrm
            dn = np.where(np.abs(dn) < 1e-3, np.sign(dn + 1e-30) * 1e-3, dn)
            steep = max(steep, float(np.degrees(np.arccos(np.clip(np.abs(Nn @ nrm), 0, 1))).max()))
            w1 = g1w * w
            rows.append(w1 * np.array([b[1] for b in B]) * size / span)   # dh/dx (slope; h is solved / size)
            rhs.append(w1 * -(Nn @ e1) / dn)
            rows.append(w1 * np.array([b[2] for b in B]) * size / span)
            rhs.append(w1 * -(Nn @ e2) / dn)
    if steep > 80.0:
        raise RuntimeError(f"a neighbouring surface stands almost edge-on to the opening ({steep:.0f}°): "
                           "the projected patch can't follow it")
    # fairness over the whole (square) domain, Gauss quadrature per knot span
    gq, gw = np.polynomial.legendre.leggauss(p + 1)
    knots = sorted(set(U))
    order = 3 if (energy == 3 and p >= 3) else 2
    F = []
    qp = []
    for a0, a1 in zip(knots[:-1], knots[1:]):
        for b0, b1 in zip(knots[:-1], knots[1:]):
            for x, wx in zip(gq, gw):
                for y, wy in zip(gq, gw):
                    qp.append((a0 + (x + 1) * (a1 - a0) / 2, b0 + (y + 1) * (b1 - b0) / 2,
                               wx * wy * (a1 - a0) * (b1 - b0) / 4))
    qp = np.array(qp)
    # the energy counts inside the opening; outside (trimmed away) only lightly
    wq = np.where(inside(loop2d, qp[:, :2]), 1.0, outside_w)
    for (u, v, aw), wi in zip(qp, wq):
        s = np.sqrt(aw * wi)
        if order == 3:
            r = surface_rows(u, v, n, p, U, PAIRS3)
            F += [s * r[0], np.sqrt(3) * s * r[1], np.sqrt(3) * s * r[2], s * r[3]]
        else:
            r = surface_rows(u, v, n, p, U)
            F += [s * r[3], np.sqrt(2) * s * r[4], s * r[5]]
    F = np.array(F) * (size / span) ** order / size        # dimensionless
    rows.append(1e-3 * fair * F)
    rhs.append(np.zeros(len(F)))
    rows.append(1e-7 * np.eye(N))
    rhs.append(np.zeros(N))
    A, Y = np.vstack(rows), np.concatenate(rhs)
    h = np.linalg.lstsq(A, Y, rcond=None)[0] * size
    gre = np.array([sum(U[i + 1:i + p + 1]) / p for i in range(n)])
    G = np.array([c + (lo[0] + gu * span) * e1 + (lo[1] + gv * span) * e2 for gu in gre for gv in gre])
    G = G + np.outer(h, nrm)
    # report: largest normal angle at the G1 samples
    worst = 0.0
    X = G.reshape(N, 3)
    for sd, cont in zip(sides, conts):
        if cont < 1 or not sd.has_surface:
            continue
        uv = to_uv(sd.P)[1:-1]
        B = [surface_rows(a, b, n, p, U) for a, b in uv]
        Su = np.array([b[1] for b in B]) @ X
        Sv = np.array([b[2] for b in B]) @ X
        n_ = np.cross(Su, Sv)
        n_ /= np.linalg.norm(n_, axis=1)[:, None]
        worst = max(worst, float(np.degrees(np.arccos(np.clip(np.abs(np.sum(n_ * np.asarray(sd.N)[1:-1], axis=1)), 0, 1))).max()))
    solve.worst_angle = worst
    return G.reshape(n, n, 3), U, nrm


# --------------------------------------------------------------------------
# General openings: developed loop, full 3D solve
# --------------------------------------------------------------------------

def develop(sides):
    """2D positions of every side sample: the loop 'unrolled' with its own
    lengths, the turning at corners and along curved edges kept and scaled so
    the loop closes once (a planar polygon stays itself, a circle a circle).
    Returns [(m_i, 2)] per side."""
    pts = [sd.P for sd in sides]
    loop = np.vstack([P[:-1] for P in pts])
    c0, nrm, _, _ = frame(loop)
    # concave corners (negative turning) only make sense for a flat loop; a
    # loop in space is laid out as convex (unsigned turning)
    flat = float(np.abs((loop - c0) @ nrm).max()) < 0.02 * (float(np.linalg.norm(np.ptp(loop, axis=0))) or 1.0)
    seg, turn = [], []
    prevT = None
    for P in pts:
        d = np.diff(P, axis=0)
        L = np.linalg.norm(d, axis=1)
        T = d / np.maximum(L, 1e-30)[:, None]
        for k in range(len(T)):
            if prevT is None:
                turn.append(0.0)
            else:
                cr = np.cross(prevT, T[k])
                ang = float(np.arctan2(np.linalg.norm(cr), np.clip(prevT @ T[k], -1, 1)))
                sgn = -1.0 if (flat and (cr @ nrm) < 0) else 1.0
                turn.append(sgn * ang)
            seg.append(L[k])
            prevT = T[k]
    first = np.diff(pts[0], axis=0)[0]
    first = first / (np.linalg.norm(first) or 1.0)
    cr = np.cross(prevT, first)
    close_turn = float(np.arctan2(np.linalg.norm(cr), np.clip(prevT @ first, -1, 1)))
    close_turn *= -1.0 if (flat and (cr @ nrm) < 0) else 1.0
    turn[0] = close_turn
    tot = sum(turn)
    if abs(tot) < 1e-6:
        raise RuntimeError("the loop does not turn: can't lay it out")
    sc = 2 * np.pi / tot
    h, xy = 0.0, [np.zeros(2)]
    for L, t in zip(seg, turn):
        h += t * sc
        xy.append(xy[-1] + L * np.array([np.cos(h), np.sin(h)]))
    xy = np.array(xy)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    xy = xy - np.outer(s / s[-1], xy[-1] - xy[0])         # close it
    out, k = [], 0
    for P in pts:
        out.append(xy[k:k + len(P)])
        k += len(P) - 1
    return out


def _fair_rows(n, p, U, loop2d, outside_w, order=2):
    gq, gw = np.polynomial.legendre.leggauss(p + 1)
    knots = sorted(set(U))
    qp = []
    for a0, a1 in zip(knots[:-1], knots[1:]):
        for b0, b1 in zip(knots[:-1], knots[1:]):
            for x, wx in zip(gq, gw):
                for y, wy in zip(gq, gw):
                    qp.append((a0 + (x + 1) * (a1 - a0) / 2, b0 + (y + 1) * (b1 - b0) / 2,
                               wx * wy * (a1 - a0) * (b1 - b0) / 4))
    qp = np.array(qp)
    wq = np.ones(len(qp)) if loop2d is None else np.where(inside(loop2d, qp[:, :2]), 1.0, outside_w)
    F = []
    for (u, v, aw), wi in zip(qp, wq):
        s_ = np.sqrt(aw * wi)
        if order == 3 and p >= 3:
            r = surface_rows(u, v, n, p, U, PAIRS3)
            F += [s_ * r[0], np.sqrt(3) * s_ * r[1], np.sqrt(3) * s_ * r[2], s_ * r[3]]
        else:
            r = surface_rows(u, v, n, p, U)
            F += [s_ * r[3], np.sqrt(2) * s_ * r[4], s_ * r[5]]
    return np.array(F)


def solve3d(sides, conts, n=20, p=3, fair=1.0, margin=0.08, g0w=100.0, g1w=30.0, soft=0.0, outside_w=0.01):
    """Any opening: the loop developed into 2D (develop) is the trim boundary in
    (u, v); all three coordinates of the net are solved:
        stage A  G0 + fairness                   (the shape)
        stage B  + G1: N . S_u = N . S_v = 0 (the neighbour's tangent plane);
                 the cross tangent's direction and length follow from the shape
                 (soft > 0 adds a pull towards A's length -- off by default)"""
    loop = np.vstack([sd.P[:-1] for sd in sides])
    allP = np.vstack([sd.P for sd in sides])
    size = float(np.linalg.norm(np.ptp(allP, axis=0))) or 1.0
    c = loop.mean(axis=0)
    xy = develop(sides)
    allxy = np.vstack(xy)
    lo, hi = allxy.min(axis=0), allxy.max(axis=0)
    ext_ = float((hi - lo).max())
    lo = (lo + hi) / 2 - ext_ * (0.5 + margin)
    span = ext_ * (1 + 2 * margin)
    uvs = [np.clip((q - lo) / span, 0, 1) for q in xy]
    loop2d = np.vstack([q[:-1] for q in uvs])
    if _self_intersects(loop2d[:: max(1, len(loop2d) // 200)]):
        raise RuntimeError("the opening's outline crosses itself when laid out flat")
    area = 0.5 * np.sum(loop2d[:, 0] * np.roll(loop2d[:, 1], -1) - np.roll(loop2d[:, 0], -1) * loop2d[:, 1])
    U = geom.clamped_uniform_knots(n, p)
    N = n * n
    n_g0 = sum(len(sd.P) for sd in sides)
    Fm = _fair_rows(n, p, U, loop2d, outside_w)

    def vec(R):                      # rows R (m, N) applied to each coordinate
        Z = np.zeros((3 * len(R), 3 * N))
        for d in range(3):
            Z[d::3, d::3] = R
        return Z

    rowsA, rhsA = [], []
    Bs = []
    for sd, uv in zip(sides, uvs):
        B = [surface_rows(a, b, n, p, U) for a, b in uv]
        Bs.append(B)
        w = g0w / np.sqrt(n_g0)
        rowsA.append(w * vec(np.array([b[0] for b in B])))
        rhsA.append(w * ((sd.P - c) / size).reshape(-1))
    rowsA.append(1e-3 * fair * vec(Fm))
    rhsA.append(np.zeros(3 * len(Fm)))
    rowsA.append(1e-6 * np.eye(3 * N))
    rhsA.append(np.zeros(3 * N))
    xA = np.linalg.lstsq(np.vstack(rowsA), np.concatenate(rhsA), rcond=None)[0]
    X = xA.reshape(N, 3)
    if any(cn >= 1 and sd.has_surface for sd, cn in zip(sides, conts)):
        rows, rhs = list(rowsA), list(rhsA)
        n_g1 = sum(len(sd.P) for sd, cn in zip(sides, conts) if cn >= 1 and sd.has_surface)
        for sd, cn, uv, B, q in zip(sides, conts, uvs, Bs, xy):
            if cn < 1 or not sd.has_surface:
                continue
            Nn = np.asarray(sd.N, float)
            Ru = np.array([b[1] for b in B])
            Rv = np.array([b[2] for b in B])
            w = g1w / np.sqrt(n_g1)
            for R in (Ru, Rv):            # N . S_u = 0, N . S_v = 0
                Z = np.zeros((len(R), 3 * N))
                for d in range(3):
                    Z[:, d::3] = R * Nn[:, d:d + 1]
                rows.append(w * Z)
                rhs.append(np.zeros(len(R)))
            # inward direction in (u, v) and the cross tangent it should have
            t2 = np.gradient(uv, axis=0)
            t2 /= np.maximum(np.linalg.norm(t2, axis=1), 1e-30)[:, None]
            m2 = np.c_[-t2[:, 1], t2[:, 0]] * (1.0 if area > 0 else -1.0)
            Rm = Ru * m2[:, :1] + Rv * m2[:, 1:]
            Sm = Rm @ X
            T3 = np.gradient(sd.P, axis=0)
            T3 /= np.maximum(np.linalg.norm(T3, axis=1), 1e-30)[:, None]
            b = np.cross(Nn, T3)
            b /= np.maximum(np.linalg.norm(b, axis=1), 1e-30)[:, None]
            b *= np.sign(np.sum(b * np.asarray(sd.B, float), axis=1))[:, None]
            if soft > 0:
                V = np.linalg.norm(Sm, axis=1)[:, None] * b
                ws = soft / np.sqrt(n_g1)
                rows.append(ws * vec(Rm))
                rhs.append(ws * V.reshape(-1))
        x = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        X = x.reshape(N, 3)
    G = X * size + c
    return G.reshape(n, n, 3), U


# --------------------------------------------------------------------------
# Four-sided base (default): the opening split into four chains at its
# sharpest corners, a smooth carrier spanned between them, extended a little
# and trimmed -- the net runs evenly along the opening (like Shapr3D / XNURBS)
# --------------------------------------------------------------------------

def _flat_loop(sides):
    """The loop as one closed polyline (junctions once) and the arc length at
    every point (M + 1 values, the last closing the loop)."""
    P = np.vstack([sd.P[:-1] for sd in sides])
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    return P, np.concatenate([[0.0], np.cumsum(seg)])


def _turning(P):
    a = P - np.roll(P, 1, axis=0)
    b = np.roll(P, -1, axis=0) - P
    a /= np.maximum(np.linalg.norm(a, axis=1), 1e-30)[:, None]
    b /= np.maximum(np.linalg.norm(b, axis=1), 1e-30)[:, None]
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def _chain_pts(P, cum, i0, i1, t):
    """Points at the fractions t of the loop's arc from point i0 to i1 (wraps)."""
    M = len(P)
    L = cum[-1]
    s0 = cum[i0]
    s1 = cum[i1] if i1 > i0 else cum[i1] + L
    s = s0 + np.asarray(t) * (s1 - s0)
    Pc = np.vstack([P, P, P[:1]])
    cc = np.concatenate([cum[:-1], cum[:-1] + L, [2 * L]])
    return np.stack([np.interp(s, cc, Pc[:, d]) for d in range(3)], 1)


def _coons(P, cum, idx, g):
    """Discrete Coons patch (g x g) of the four chains between corners idx."""
    t = np.linspace(0, 1, g)
    C0 = _chain_pts(P, cum, idx[0], idx[1], t)
    C1 = _chain_pts(P, cum, idx[1], idx[2], t)
    C2 = _chain_pts(P, cum, idx[2], idx[3], t)[::-1]
    C3 = _chain_pts(P, cum, idx[3], idx[0], t)[::-1]
    u, v = t[:, None, None], t[None, :, None]
    S = ((1 - v) * C0[:, None] + v * C2[:, None] + (1 - u) * C3[None] + u * C1[None]
         - ((1 - u) * (1 - v) * C0[0] + u * (1 - v) * C0[-1] + (1 - u) * v * C2[0] + u * v * C2[-1]))
    return S


def _coons_energy(S, size):
    h = 1.0 / (len(S) - 1)
    duu = (S[2:] - 2 * S[1:-1] + S[:-2]) / h ** 2
    dvv = (S[:, 2:] - 2 * S[:, 1:-1] + S[:, :-2]) / h ** 2
    duv = (S[1:, 1:] - S[1:, :-1] - S[:-1, 1:] + S[:-1, :-1]) / h ** 2
    return float((np.sum(duu ** 2) + 2 * np.sum(duv ** 2) + np.sum(dvv ** 2)) * h * h) / size ** 2


def corners(sides, max_cand=8, min_turn=25.0, fixed=None):
    """The four loop points (indices into the flat loop) the patch's base is
    spanned between: among the sharpest corners, the four whose Coons patch
    bends least; a loop with fewer corners gets the rest at quarter lengths.
    fixed: loop indices that must be among them (picked by hand); four of
    them are the answer as they are."""
    from itertools import combinations
    P, cum = _flat_loop(sides)
    M = len(P)
    fixed = sorted({int(i) % M for i in (fixed or [])})
    if len(fixed) >= 4:
        return fixed[:4], P, cum
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    # signed: an inward (reflex) corner can't be a corner of the base square
    # (a 90 degree corner of the domain can't open up to 270 degrees)
    tr = np.degrees(_signed_turning(P, _probe_normals(P, cum)))
    near = 0.03 * cum[-1]
    cand = list(fixed)
    for i in np.argsort(-tr):
        if tr[i] < min_turn or len(cand) >= max_cand + len(fixed):
            break
        if all(min(abs(cum[i] - cum[j]), cum[-1] - abs(cum[i] - cum[j])) > near for j in cand):
            cand.append(int(i))
    if len(cand) < 4:
        s0 = cum[cand[0]] if cand else 0.0
        extra = []
        for q in range(4):
            s = (s0 + q * cum[-1] / 4) % cum[-1]
            i = int(np.argmin(np.abs(cum[:-1] - s)))
            d = min([min(abs(cum[i] - cum[j]), cum[-1] - abs(cum[i] - cum[j])) for j in cand] or [np.inf])
            extra.append((d, i))
        extra.sort(reverse=True)
        cand += [i for _, i in extra[:4 - len(cand)]]
    best = None
    for combo in combinations(sorted(set(cand)), 4):
        if len(set(combo)) < 4 or not set(fixed) <= set(combo):
            continue
        e = _coons_energy(_coons(P, cum, combo, 14), size)
        lens = [((cum[combo[(k + 1) % 4]] if k < 3 else cum[combo[0]] + cum[-1]) - cum[combo[k]]) for k in range(4)]
        bal = max(lens[0], lens[2]) / max(min(lens[0], lens[2]), 1e-30) + \
            max(lens[1], lens[3]) / max(min(lens[1], lens[3]), 1e-30)
        sc = e * (1.0 + 0.25 * (bal - 2.0))
        if best is None or sc < best[0]:
            best = (sc, combo)
    return list(best[1]), P, cum


def _eval_rows(uv, n, p, U, pairs=PAIRS2):
    """surface_rows at every (u, v) of uv: (m, len(pairs), nu * nv)."""
    return surface_rows_many(uv, n, p, U, pairs)


def _project(X, uv, P, n, p, U, it=6):
    """Closest (u, v) on the net X (N, 3) for the points P, from uv (every
    point by its own Newton steps, all at once)."""
    uv = np.array(uv, float)
    P = np.asarray(P, float)
    for _ in range(it):
        R = surface_rows_many(uv, n, p, U, PAIRS2[:3])
        S, Su, Sv = R[:, 0] @ X, R[:, 1] @ X, R[:, 2] @ X
        uv = np.clip(uv + np.clip(lstsq2(Su, Sv, P - S), -0.1, 0.1), 0.0, 1.0)
    return uv


def _loop_normals(P, cum, idx, g=48):
    """Unit normals of the four chains' Coons patch at every loop point (the
    surface the loop bounds, roughly), oriented so u x v is CCW."""
    S = _coons(P, cum, idx, g)
    Su = np.gradient(S, axis=0)
    Sv = np.gradient(S, axis=1)
    Nn = np.cross(Su, Sv)
    Nn /= np.maximum(np.linalg.norm(Nn, axis=2), 1e-30)[..., None]
    out = np.zeros_like(P)
    M = len(P)
    L = cum[-1]
    for q in range(4):
        a, b = idx[q], idx[(q + 1) % 4]
        sa, sb = cum[a], (cum[b] if b > a else cum[b] + L)
        k = a
        while True:
            i = k % M
            s_i = cum[i] if cum[i] >= sa - 1e-12 else cum[i] + L
            f = min(max((s_i - sa) / max(sb - sa, 1e-30), 0.0), 1.0)
            r = int(round(f * (g - 1)))
            out[i] = [Nn[r, 0], Nn[g - 1, r], Nn[g - 1 - r, g - 1], Nn[0, g - 1 - r]][q]
            if i == b % M:
                break
            k += 1
    return _consistent(P, out)


def _signed_turning(P, nrm):
    """Turning (radians) at every loop point within the plane normal to nrm:
    positive toward the inside of the loop (convex), negative at an inward
    (reflex) corner."""
    ti = P - np.roll(P, 1, axis=0)
    to = np.roll(P, -1, axis=0) - P
    ti = ti - np.sum(ti * nrm, axis=1)[:, None] * nrm
    to = to - np.sum(to * nrm, axis=1)[:, None] * nrm
    return np.arctan2(np.sum(np.cross(ti, to) * nrm, axis=1), np.sum(ti * to, axis=1))


def _consistent(P, Nn):
    """Loop normals made one-sided: the Coons patch of a concave (L-shaped,
    star) opening folds over near the inward corners, so its normal flips
    there and the turning there reads the wrong way round. Walk the loop and
    flip every normal that turns against its predecessor, then orient them
    all so the loop turns positively overall (its inside to the left)."""
    Nn = np.array(Nn, float)
    for i in range(1, len(Nn)):
        if float(Nn[i] @ Nn[i - 1]) < 0:
            Nn[i] = -Nn[i]
    if float(np.sum(_signed_turning(P, Nn))) < 0:
        Nn = -Nn
    return Nn


def _probe_normals(P, cum):
    """Loop normals independent of the corner choice: the Coons patch spanned
    between the quarter-length points."""
    L = cum[-1]
    idx = [int(np.argmin(np.abs(cum[:-1] - q * L / 4))) for q in range(4)]
    if len(set(idx)) < 4:
        idx = [q * len(P) // 4 for q in range(4)]
    return _loop_normals(P, cum, idx)


def corner_kappa(t):
    """Share of a corner's turning (radians, within the surface) kept in the
    (u, v) layout of a chain: half for ordinary corners (a full turn makes
    the chain bulge too far and the net wiggles, e.g. the elbow's 90 degree
    floor-wall corners), all of it at inward corners and sharp spikes (over
    ~100 degrees of turning), where half a turn leaves the map folded."""
    d = np.degrees(t)
    if d < -10.0:
        return 1.0
    if d <= 95.0:
        return 0.5
    return min(1.0, 0.5 + 0.5 * (d - 95.0) / 25.0)


def layout_quad(sides, idx, kappa=0.5, kfun=None):
    """(u, v) of every side sample: the four corners at the unit square's
    corners, each chain laid out with its own lengths and its turning WITHIN
    the surface (geodesic: a corner where a floor edge meets a wall edge
    turns, an arch around a tube does not), scaled onto its chord. A chain
    with a corner inside thus bulges past the square instead of being forced
    straight (which would crease the surface across). Returns [(m_i, 2)]."""
    P, cum = _flat_loop(sides)
    M = len(P)
    nrm = _loop_normals(P, cum, idx)
    ti = P - np.roll(P, 1, axis=0)
    to = np.roll(P, -1, axis=0) - P
    ti -= np.sum(ti * nrm, axis=1)[:, None] * nrm
    to -= np.sum(to * nrm, axis=1)[:, None] * nrm
    turn = np.arctan2(np.sum(np.cross(ti, to) * nrm, axis=1), np.sum(ti * to, axis=1))
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    kfun = corner_kappa if kfun is None else kfun
    kmul = np.array([kfun(float(t)) for t in turn]) / 0.5
    corners2 = np.array([0, 1, 1 + 1j, 1j])
    chains = []
    for q in range(4):
        a, b = idx[q], idx[(q + 1) % 4]
        ks = [k % M for k in range(a, (b if b > a else b + M) + 1)]
        f = np.concatenate([[0.0], np.cumsum([seg[k] for k in ks[:-1]])])
        f = f / (f[-1] or 1.0)
        w_s = np.clip((0.4 - f) / 0.4, 0.0, 1.0)
        w_e = np.clip((f - 0.6) / 0.4, 0.0, 1.0)
        w_s[0] = w_s[-1] = w_e[0] = w_e[-1] = 0.0
        w_s = w_s / (w_s.sum() or 1.0)
        w_e = w_e / (w_e.sum() or 1.0)
        chains.append((ks, w_s, w_e))

    def build(kap, b_s, b_e):
        uvc = np.zeros(M, complex)
        for q, (ks, w_s, w_e) in enumerate(chains):
            z, h, pts = 0j, 0.0, [0j]
            for n_, k in enumerate(ks[:-1]):
                if n_ > 0:
                    h += kap * kmul[k] * turn[k] + b_s[q] * w_s[n_] + b_e[q] * w_e[n_]
                z = z + seg[k] * np.exp(1j * h)
                pts.append(z)
            pts = np.array(pts)
            chord = pts[-1] - pts[0]
            if abs(chord) < 0.25 * float(np.sum(np.abs(np.diff(pts)))):
                return None
            mapped = corners2[q] + (pts - pts[0]) * (corners2[(q + 1) % 4] - corners2[q]) / chord
            for k, w in zip(ks[:-1], mapped[:-1]):
                uvc[k] = w
        return uvc

    def corner_turn(uvc, c):
        d_in, d_out = uvc[c] - uvc[(c - 1) % M], uvc[(c + 1) % M] - uvc[c]
        return float(np.angle(d_out / d_in)) if abs(d_in) > 0 and abs(d_out) > 0 else 0.0

    for kap in (kappa, 0.5 * kappa, 0.25 * kappa, 0.0):
        # a base-square corner where the loop hardly turns (the made-up 4th
        # corner of a three-sided opening, a corner on a smooth round hole)
        # would turn 90 degrees in (u, v) but not in space: the surface pinches
        # there. The chains on both sides bend toward the inside near it until
        # the (u, v) corner turns about as much as the loop does in space.
        b_s, b_e = np.zeros(4), np.zeros(4)
        uvc = build(kap, b_s, b_e)
        for _ in range(6 if uvc is not None else 0):
            done = True
            for q in range(4):
                d = corner_turn(uvc, idx[q]) - max(float(turn[idx[q]]), 0.0)
                if d > np.radians(15.0):
                    b_e[(q - 1) % 4] += 0.35 * d
                    b_s[q] += 0.35 * d
                    done = False
            if done:
                break
            nxt = build(kap, b_s, b_e)
            if nxt is None:
                break
            uvc = nxt
        if uvc is None:
            continue
        loop2d = np.c_[uvc.real, uvc.imag]
        if kap == 0.0 or not _self_intersects(loop2d[:: max(1, M // 200)]):
            break
    out, k = [], 0
    for sd in sides:
        ii = [(k + j) % M for j in range(len(sd.P))]
        out.append(loop2d[ii].copy())
        k += len(sd.P) - 1
    return out


def _end_curvature(Q, nrm):
    """Normal curvature (on nrm) at Q[0] of the polyline Q: a quadratic in arc
    length through its first samples. Also its unit tangent there."""
    m = min(len(Q), 7)
    sq = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q[:m], axis=0), axis=1))])
    coef = np.linalg.lstsq(np.c_[np.ones(m), sq, sq * sq], Q[:m], rcond=None)[0]
    d1n = np.linalg.norm(coef[1]) or 1.0
    return float((2 * coef[2]) @ nrm) / d1n ** 2, coef[1] / d1n


def _g2_ease(sides, si, bb, size, reach=0.5):
    """Weights (0..1) of side si's G2 rows. At a corner the next side lies on
    the patch too, so its normal curvature there must equal what the
    neighbour's second fundamental form gives in its direction (Euler:
    c^2 k_bb + 2cs k_bT + s^2 k_TT across b / along T). Where it doesn't -- a
    straight floor edge leaving a curved wall -- no surface can have both:
    G2 is eased to zero toward that corner over `reach` of the side."""
    sd = sides[si]
    P = np.asarray(sd.P, float)
    Nn = np.asarray(sd.N, float)
    s_ = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    L = s_[-1] or 1.0
    w = np.ones(len(P))
    nb = len(sides)
    for end in (0, -1):
        other = sides[(si - 1) % nb] if end == 0 else sides[(si + 1) % nb]
        Q = np.asarray(other.P, float)
        Q = Q[::-1] if end == 0 else Q                  # Q[0] at the shared corner
        own = P if end == 0 else P[::-1]                 # own[0] at the corner too
        Nown = Nn if end == 0 else Nn[::-1]
        if len(Q) < 3 or len(own) < 3:
            continue
        nrm = Nown[0]
        b = bb[end]
        k_adj, t_adj = _end_curvature(Q, nrm)
        k_tt, T = _end_curvature(own, nrm)
        ds = np.linalg.norm(own[1] - own[0]) or 1.0
        k_bt = -float((Nown[1] - Nown[0]) @ b) / ds
        k_bb = float(np.asarray(sd.K, float)[end])
        c, s = float(t_adj @ b), float(t_adj @ T)
        h = np.hypot(c, s) or 1.0
        c, s = c / h, s / h
        k_pred = c * c * k_bb + 2 * c * s * k_bt + s * s * k_tt
        if abs(k_adj - k_pred) <= 0.3 * max(abs(k_pred), abs(k_adj), abs(k_bb), 1.0 / L):
            continue
        dist = s_ if end == 0 else (L - s_)
        t = np.clip(dist / (reach * L), 0.0, 1.0)
        w = np.minimum(w, t * t * (3 - 2 * t))
    return w


CORNER_G1_DEG = 1.0         # (as Square) smaller mismatches (approximate trim curves) are bridged by the solve


def _end_tangent(Q, tq=None):
    """Unit tangent at Q[0], pointing along Q: a cubic fit over the first
    samples in the source's own parameter tq (exact for polynomial edges,
    O(h^4) otherwise; a chord or a quadratic in arc length is off by tenths
    of a degree on a sampled arc, enough to fake a corner conflict)."""
    m = min(len(Q), 8)
    t = np.asarray(tq[:m], float) - float(tq[0]) if tq is not None else \
        np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q[:m], axis=0), axis=1))])
    sc = float(np.max(np.abs(t))) or 1.0
    t = t / sc
    deg = min(3, m - 1)
    coef = np.linalg.lstsq(np.vander(t, deg + 1, increasing=True), Q[:m], rcond=None)[0]
    d = coef[1]
    if float(d @ (Q[1] - Q[0])) < 0:
        d = -d
    return d / (np.linalg.norm(d) or 1.0)


def _exact_tangent(sd, at, toward):
    """Unit tangent of the side's source curve at its end nearest `at`,
    pointing toward `toward` -- from the curve itself (None if the side has no
    source): sampled points can't give a corner's tangent to 0.1 degrees."""
    src = getattr(sd, "src", None)
    if not src:
        return None
    try:
        import bpy
        obj = bpy.data.objects.get(src[0]) if isinstance(src[0], str) else src[0]
        if obj is None:
            return None
        kc = features.input_curve(obj, src[1])
        t0, t1 = kc.parameter_range()
        p0, p1 = np.array(kc.evaluate(t0), float), np.array(kc.evaluate(t1), float)
        t = t0 if np.linalg.norm(p0 - at) <= np.linalg.norm(p1 - at) else t1
        e = (t1 - t0) * 1e-6 * (1.0 if t == t0 else -1.0)
        d = np.array(kc.evaluate(t + e), float) - np.array(kc.evaluate(t), float)
        if np.linalg.norm(d) < 1e-30:
            return None
        d /= np.linalg.norm(d)
        return d if float(d @ (np.asarray(toward) - np.asarray(at))) >= 0 else -d
    except Exception:
        return None


def corner_eases(sides, conts, reach=0.5):
    """Per side: (w1, notes) -- weights (0..1) of its G1 rows. At a corner the
    patch's tangent plane is fixed by the two boundary edges meeting there
    (it contains both). A G1 side whose neighbour's tangent plane does not
    contain the other edge (a drafted tube wall meeting a vertical skirt,
    say) can't be G1 at that corner: forcing it squashes the patch flat there
    (a fold), so its G1 rows ease to zero toward that corner over `reach` of
    the side. Corners where the two edges run on in nearly the same direction
    compare the two neighbours' normals instead. notes: [(side, other side,
    degrees, end)] for every eased corner."""
    nb = len(sides)
    W = [np.ones(len(sd.P)) for sd in sides]
    notes = []
    for si, (sd, cn) in enumerate(zip(sides, conts)):
        if cn < 1 or not sd.has_surface:
            continue
        P = np.asarray(sd.P, float)
        s_ = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        L = s_[-1] or 1.0
        for end in (0, -1):
            j = (si - 1) % nb if end == 0 else (si + 1) % nb
            other = sides[j]
            Q = np.asarray(other.P, float)
            tq = np.asarray(other.t, float)
            Q, tq = (Q[::-1], tq[::-1]) if end == 0 else (Q, tq)      # Q[0] at the corner
            own, to = (P, np.asarray(sd.t, float)) if end == 0 else (P[::-1], np.asarray(sd.t, float)[::-1])
            if len(Q) < 3 or len(own) < 3:
                continue
            n = np.asarray(sd.N[end], float)
            n = n / (np.linalg.norm(n) or 1.0)
            t_adj = _exact_tangent(other, Q[0], Q[1])
            t_adj = _end_tangent(Q, tq) if t_adj is None else t_adj
            t_own = _exact_tangent(sd, own[0], own[1])
            t_own = _end_tangent(own, to) if t_own is None else t_own
            if abs(float(t_adj @ t_own)) < 0.9:         # a real corner: the other edge must lie in the plane
                miss = np.degrees(np.arcsin(min(1.0, abs(float(t_adj @ n)))))
            elif conts[j] >= 1 and other.has_surface:   # edges run on: the two neighbours must agree
                m = np.asarray(other.N[0 if end == -1 else -1], float)
                m = m / (np.linalg.norm(m) or 1.0)
                miss = np.degrees(np.arccos(min(1.0, abs(float(n @ m)))))
            else:
                continue
            if miss <= CORNER_G1_DEG:
                continue
            dist = s_ if end == 0 else (L - s_)
            t = np.clip(dist / (reach * L), 0.0, 1.0)
            W[si] = np.minimum(W[si], t * t * (3 - 2 * t))
            notes.append((si, j, float(miss), end))
    return W, notes


def _net_rows(n, p, U):
    """Second divided differences of every CV row and column against the
    Greville abscissae (zero for a net that is linear in its parameters): the
    control net's own fairness, so its rows run evenly (a degree-5 surface
    hides wobbly CVs, the net shows them). n, U as surface_rows."""
    nu, nv, Uu, Uv = _dims(n, U)
    D = []
    for ax, (m, Ua, other) in enumerate(((nu, Uu, nv), (nv, Uv, nu))):
        g = np.array([sum(Ua[i + 1:i + p + 1]) / p for i in range(m)])
        h = np.diff(g)
        for i in range(1, m - 1):
            c0, c2 = 1.0 / h[i - 1], 1.0 / h[i]
            sc = 1.0 / (0.5 * (h[i - 1] + h[i])) / (m - 1) ** 2
            coef = (c0 * sc, -(c0 + c2) * sc, c2 * sc)
            for j in range(other):
                row = np.zeros(nu * nv)
                for d_, cf in zip((-1, 0, 1), coef):
                    a, b = (i + d_, j) if ax == 0 else (j, i + d_)
                    row[a * nv + b] = cf
                D.append(row)
    return np.array(D)


_GRAM = {}
_GRAM1 = {}


def _gram1(n, p):
    """1D Gram matrices (4, n, n) of the clamped uniform basis and its first
    three derivatives over [0, 1]."""
    key = (n, p)
    if key in _GRAM1:
        return _GRAM1[key]
    U = geom.clamped_uniform_knots(n, p)
    nd = min(3, p)
    M = np.zeros((4, n, n))
    gq, gw = np.polynomial.legendre.leggauss(p + 1)
    knots = sorted(set(U))
    for a0, a1 in zip(knots[:-1], knots[1:]):
        for x, wx in zip(gq, gw):
            t = a0 + (x + 1) * (a1 - a0) / 2
            span = geom.find_span(n - 1, p, t, U)
            D = ders_basis(span, t, p, U, nd)
            sl = slice(span - p, span + 1)
            for k in range(nd + 1):
                M[k, sl, sl] += wx * (a1 - a0) / 2 * np.outer(D[k], D[k])
    _GRAM1[key] = M
    return M


def _gram(n, p, forder=2):
    """(F2, F3, D): for an n x n (or n = (nu, nv)) net on clamped uniform
    knots, the thin-plate energy (second derivatives) and the curvature-
    variation energy (third derivatives) as quadratic forms -- exact: the net
    is a tensor product, so both split into 1D Gram matrices of the basis and
    its derivatives -- and the net fairness D^T D. Cached per size."""
    nu, nv = (int(n[0]), int(n[1])) if isinstance(n, (tuple, list)) else (int(n), int(n))
    key = (nu, nv, p)
    if key in _GRAM:
        return _GRAM[key]
    A, B = _gram1(nu, p), _gram1(nv, p)
    F2 = np.kron(A[2], B[0]) + 2 * np.kron(A[1], B[1]) + np.kron(A[0], B[2])
    F3 = (np.kron(A[3], B[0]) + 3 * np.kron(A[2], B[1]) + 3 * np.kron(A[1], B[2]) + np.kron(A[0], B[3])) \
        if p >= 3 else np.zeros_like(F2)
    # scale the third-order form to the second's size (1/n^2 per derivative
    # pair), so f3 is a share rather than a knot-dependent magnitude
    F3 = F3 * (np.trace(F2) / max(np.trace(F3), 1e-30))
    Uu, Uv = geom.clamped_uniform_knots(nu, p), geom.clamped_uniform_knots(nv, p)
    Dn = _net_rows((nu, nv), p, (Uu, Uv))
    _GRAM[key] = (F2, F3, Dn.T @ Dn)
    return _GRAM[key]


def _gauss1(n, p):
    """Gauss points of the clamped uniform basis on [0, 1]: (t, weights,
    [(span, basis derivatives 0..2)])."""
    U = geom.clamped_uniform_knots(n, p)
    gq, gw = np.polynomial.legendre.leggauss(p + 1)
    knots = sorted(set(U))
    T, W, D = [], [], []
    for a0, a1 in zip(knots[:-1], knots[1:]):
        for x, wx in zip(gq, gw):
            t = a0 + (x + 1) * (a1 - a0) / 2
            span = geom.find_span(n - 1, p, t, U)
            T.append(t)
            W.append(wx * (a1 - a0) / 2)
            D.append((span, ders_basis(span, t, p, U, 2)))
    return np.array(T), np.array(W), D


def _gram_weighted(n2, p, wfun):
    """The thin-plate energy with a stiffness w(u, v) (> 0) as a quadratic
    form on the nu x nv net: sum over Gauss points of w (S_uu^2 + 2 S_uv^2 +
    S_vv^2). w = 1 gives _gram's F2. Assembled per u Gauss point as Kronecker
    blocks (the net is a tensor product)."""
    nu, nv = n2
    Tu, Wu, Du = _gauss1(nu, p)
    Tv, Wv, Dv = _gauss1(nv, p)
    wg = wfun(Tu[:, None] * np.ones((1, len(Tv))), np.ones((len(Tu), 1)) * Tv[None, :])
    Bv = np.zeros((len(Tv), 3, nv, nv))
    for g, (span, Dd) in enumerate(Dv):
        sl = slice(span - p, span + 1)
        for l in range(3):
            Bv[g, l, sl, sl] = Wv[g] * np.outer(Dd[l], Dd[l])
    F = np.zeros((nu * nv, nu * nv))
    for g, (span, Dd) in enumerate(Du):
        C = np.einsum("g,glab->lab", wg[g], Bv)
        r0 = (span - p) * nv
        r1 = r0 + (p + 1) * nv
        for k, l, cf in ((2, 0, 1.0), (1, 1, 2.0), (0, 2, 1.0)):
            F[r0:r1, r0:r1] += cf * np.kron(Wu[g] * np.outer(Dd[k], Dd[k]), C[l])
    return F


# G2 Gauss-Newton stops once a step lowers the energy by less than this share:
# after the first step the rest only move CVs ~1e-5 of the size (a Patch's
# edges measured the same to the last digit), and each step is a full solve
GN_RTOL = 1e-4
GEO_PASSES = 2         # solves with the fairness measured on the surface itself (0: in (u, v) only)
GEO_THIRD = 0.02       # curvature variation next to bending, per the loop's size squared
GEO_COND = 20.0        # the metric's stretch is clamped at this ratio (sides of a cell)
GEO_GAUSS = 4          # Gauss points per knot span and direction
_GEO1 = {}


def _geo1(n, p):
    """Per knot span of the clamped uniform basis on [0, 1]: the Gauss points
    (spans, g), their weights (spans, g), the basis derivatives 0..3 there
    (spans, g, 4, p + 1; zero past the degree) and the span's first basis
    index (spans,). Cached per size."""
    key = (n, p)
    if key in _GEO1:
        return _GEO1[key]
    U = geom.clamped_uniform_knots(n, p)
    x, w = np.polynomial.legendre.leggauss(GEO_GAUSS)
    knots = sorted(set(U))
    nd = min(3, p)
    T, W, D, first = [], [], [], []
    for a0, a1 in zip(knots[:-1], knots[1:]):
        h = a1 - a0
        ts = a0 + (x + 1) * h / 2
        span = geom.find_span(n - 1, p, float(ts[0]), U)
        Dd = np.zeros((GEO_GAUSS, 4, p + 1))
        for g, t in enumerate(ts):
            Dd[g, :nd + 1] = ders_basis(span, float(t), p, U, nd)
        T.append(ts)
        W.append(w * h / 2)
        D.append(Dd)
        first.append(span - p)
    _GEO1[key] = (np.array(T), np.array(W), np.array(D), np.array(first))
    return _GEO1[key]


def _geo_gram(X, n2, p, wfun=None):
    """(F2, F3): bending and curvature variation measured on the surface of
    the net X (N, 3), as quadratic forms on the net --
        F2  |S_xx|^2 + 2 |S_xy|^2 + |S_yy|^2
        F3  |S_xxx|^2 + 3 |S_xxy|^2 + 3 |S_xyy|^2 + |S_yyy|^2
    over its area, in local orthonormal coordinates (x, y) of X's surface:
    d/dx, d/dy = g^-1/2 (d/du, d/dv), g its first fundamental form, held
    fixed. _gram measures the same in (u, v), so wherever the layout bends
    the parameter lines (a corner inside a chain) the surface bends with
    them: a ridge in the zebra, kinked CV rows. Measured on the surface the
    layout no longer shows. wfun(u, v): stiffness (tension) or None.
    Vectorised per knot cell (the (p + 1)^2 CVs every Gauss point of it
    sees), then scattered into the net."""
    nu, nv = n2
    Tu, Wu, Du, fu = _geo1(nu, p)
    Tv, Wv, Dv, fv = _geo1(nv, p)
    su, sv, g, q = len(fu), len(fv), GEO_GAUSS, p + 1
    N = nu * nv
    iu = fu[:, None] + np.arange(q)[None, :]
    iv = fv[:, None] + np.arange(q)[None, :]
    C = X.reshape(nu, nv, 3)[iu[:, None, :, None], iv[None, :, None, :]]        # (su, sv, q, q, 3)

    def rows(k, l):
        # the basis' (k, l) derivative at every Gauss point: (su, sv, g, g, q, q)
        return Du[:, None, :, None, k, :, None] * Dv[None, :, None, :, l, None, :]

    Su = np.einsum("Aai,Bbj,ABijc->ABabc", Du[:, :, 1], Dv[:, :, 0], C)
    Sv = np.einsum("Aai,Bbj,ABijc->ABabc", Du[:, :, 0], Dv[:, :, 1], C)
    g11, g12, g22 = (Su * Su).sum(-1), (Su * Sv).sum(-1), (Sv * Sv).sum(-1)
    gm = np.stack([np.stack([g11, g12], -1), np.stack([g12, g22], -1)], -2)
    lam, V = np.linalg.eigh(gm)                        # ascending: (stretch across, along)
    ref = float(np.median(lam.sum(-1))) / 2 or 1.0
    lam = np.maximum(lam, 1e-6 * ref)                  # degenerate spots (outside the trim)
    lam[..., 1] = np.minimum(lam[..., 1], GEO_COND ** 2 * lam[..., 0])
    K = np.einsum("...ij,...j,...kj->...ik", V, 1.0 / np.sqrt(lam), V)
    dA = np.sqrt(np.maximum(g11 * g22 - g12 * g12, 1e-12 * ref * ref))
    W = Wu[:, None, :, None] * Wv[None, :, None, :] * dA
    if wfun is not None:
        W = W * wfun(np.broadcast_to(Tu[:, None, :, None], W.shape), np.broadcast_to(Tv[None, :, None, :], W.shape))
    sw = np.sqrt(W)[..., None, None]
    a1, a2 = K[..., 0, 0, None, None], K[..., 0, 1, None, None]      # d/dx = a1 d/du + a2 d/dv
    b1, b2 = K[..., 1, 0, None, None], K[..., 1, 1, None, None]      # d/dy = b1 d/du + b2 d/dv
    R = {kl: rows(*kl) for kl in ((2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3))}

    def d2(x1, x2, y1, y2):
        return x1 * y1 * R[2, 0] + (x1 * y2 + x2 * y1) * R[1, 1] + x2 * y2 * R[0, 2]

    def d3(x1, x2, y1, y2, z1, z2):
        return (x1 * y1 * z1 * R[3, 0] + (x1 * y1 * z2 + x1 * y2 * z1 + x2 * y1 * z1) * R[2, 1]
                + (x1 * y2 * z2 + x2 * y1 * z2 + x2 * y2 * z1) * R[1, 2] + x2 * y2 * z2 * R[0, 3])

    r2 = np.sqrt(2.0)
    r3 = np.sqrt(3.0)
    two = [d2(a1, a2, a1, a2), r2 * d2(a1, a2, b1, b2), d2(b1, b2, b1, b2)]
    three = [d3(a1, a2, a1, a2, a1, a2), r3 * d3(a1, a2, a1, a2, b1, b2),
             r3 * d3(a1, a2, b1, b2, b1, b2), d3(b1, b2, b1, b2, b1, b2)]
    # global index of every cell's CVs, for the scatter
    ids = (iu[:, None, :, None] * nv + iv[None, :, None, :]).reshape(su * sv, q * q)
    lin = (ids[:, :, None] * N + ids[:, None, :]).ravel()
    out = []
    for parts in (two, three):
        Rm = np.stack([sw * r for r in parts], 2)                 # (su, sv, k, g, g, q, q)
        Rm = Rm.reshape(su * sv, -1, q * q)
        L = np.matmul(Rm.transpose(0, 2, 1), Rm)                  # (cells, q*q, q*q)
        out.append(np.bincount(lin, weights=L.ravel(), minlength=N * N).reshape(N, N))
    return out[0], out[1]


def _stiffness(sides, uv, tens, reach):
    """w(u, v) for tension: a side with tension t makes the patch t^TENSION_EXP
    times as stiff against bending next to it, fading out over `reach` in
    (u, v) (smoothly; several sides multiply)."""
    segs = []
    o = 0
    for si, sd in enumerate(sides):
        q = uv[o:o + len(sd.P)]
        o += len(sd.P)
        if abs(tens[si] - 1.0) > 1e-6:
            segs.append((q, TENSION_EXP * np.log(max(tens[si], 1e-3))))

    def w(U_, V_):
        X = np.stack([U_.ravel(), V_.ravel()], 1)
        e = np.zeros(len(X))
        for q, lg in segs:
            # distance to the side's polyline in (u, v)
            a, b = q[:-1], q[1:]
            ab = b - a
            t = np.clip(np.einsum("mkd,kd->mk", X[:, None, :] - a[None], ab) / np.maximum(np.sum(ab * ab, 1), 1e-30), 0, 1)
            d = np.min(np.linalg.norm(X[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=2), axis=1)
            e += lg * _smooth(1.0 - d / reach)
        return np.exp(e).reshape(U_.shape)
    return w


TENSION_EXP = 2.0          # tension t: t^2 times as stiff against bending next to that edge
TENSION_REACH = 0.35       # fading out over this distance into the patch (in (u, v))
GUIDE_FADE = 0.12          # guides give way to the edges within this share of the loop's size
GUIDE_MIN = 0.02
GUIDE_PASSES = 1


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ata_add(Q, A):
    """Q += A^T A, over only the columns A uses (a row of basis values
    touches (p + 1)^2 of the net's CVs): the same sum, a fraction of the
    work of the dense product."""
    nz = np.flatnonzero(np.any(A != 0.0, axis=0))
    if not len(nz):
        return
    Ac = A[:, nz]
    M = Ac.T @ Ac
    if Q.flags.c_contiguous:
        # through flat indices: a third of the time of Q[np.ix_(nz, nz)]
        Q.reshape(-1)[(nz[:, None] * Q.shape[1] + nz[None, :]).ravel()] += M.ravel()
    else:
        Q[np.ix_(nz, nz)] += M


def band_solve(Q, b, rows, band):
    """Q x = b for a symmetric positive definite Q whose unknowns come in
    `rows` equal groups in order (a net's CV rows) with nothing between
    groups more than `band` apart -- every term of a Patch's system is a
    product of B-spline basis functions, which reach p CV rows -- so Q is
    block tridiagonal in blocks of `band` groups. Eliminated block by
    block: about (band / rows)^2 of the dense solve's work (a 24 x 24
    degree-5 net: 4 ms instead of 13). LinAlgError as np.linalg.solve."""
    n = len(Q)
    bs = band * (n // rows)
    if n % rows or 3 * bs > n:
        return np.linalg.solve(Q, b)
    e = list(range(0, n, bs)) + [n]
    sl = [slice(a, c) for a, c in zip(e[:-1], e[1:])]
    K = len(sl)
    B = np.asarray(b, float).reshape(n, -1)
    m = B.shape[1]
    W, Z = [None] * K, [None] * K
    for k in range(K):
        S, y = Q[sl[k], sl[k]], B[sl[k]]
        if k:
            L = Q[sl[k], sl[k - 1]]
            S = S - L @ W[k - 1]
            y = y - L @ Z[k - 1]
        if k + 1 < K:
            G = np.linalg.solve(S, np.hstack([Q[sl[k], sl[k + 1]], y]))
            W[k], Z[k] = G[:, :-m], G[:, -m:]
        else:
            Z[k] = np.linalg.solve(S, y)
    x = np.empty_like(B)
    x[sl[-1]] = Z[-1]
    for k in range(K - 2, -1, -1):
        x[sl[k]] = Z[k] - W[k] @ x[sl[k + 1]]
    return x.reshape(np.shape(b))


def net_solve(Q, b, nu, p):
    """Q x = b for a Patch's system on a net of nu CV rows (unknowns: CV
    i * nv + j, then x, y, z) by band_solve; lstsq when Q is singular.
    (Reordering a net with shorter v rows first narrows the blocks, but the
    reordered copy of Q costs more than that saves.)"""
    try:
        return band_solve(Q, b, nu, max(p, 2))     # _net_rows' second differences reach 2 rows
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Q, b, rcond=None)[0]


def _grid_uv(X, n, p, U, P, g=41):
    """(u, v) of the nearest point of a g x g grid on the net X for every
    point of P (start values for _project)."""
    nu, nv, Uu, Uv = _dims(n, U)
    t = np.linspace(0.0, 1.0, g)
    Bu = geom.interp_matrix(t, Uu, p, nu)
    Bv = geom.interp_matrix(t, Uv, p, nv)
    S = np.einsum("ai,bj,ijd->abd", Bu, Bv, X.reshape(nu, nv, 3)).reshape(-1, 3)
    k = np.argmin(((P[:, None, :] - S[None, :, :]) ** 2).sum(axis=2), axis=1)
    return np.c_[t[k // g], t[k % g]]


def solve_quad(sides, conts, n=24, p=3, fair=1.0, margin=0.12, g0w=100.0, g1w=30.0, idx=None, kappa=0.5,
               reproject=True, forder=2, fw=4.5e-3, g2w=3.0, gn_iters=8, cw=0.02, f3=0.0, g0ease=10.0, kfun=None,
               tension=None, guides=None, gw=1.0, report=None, geo=GEO_PASSES, geo3=GEO_THIRD):
    """The patch's net (nu x nv; n = count or (nu, nv)): the four chains
    between the corners idx laid out in (u, v) by layout_quad, the layout's
    box placed in [m, 1 - m] (m = margin share) so the net runs a little past
    every edge for the trim, then one least-squares solve:
        G0  the position at every edge sample (and on every guide curve)
        G1  the neighbour's tangent plane (N . S_u = N . S_v = 0)
        fairness: thin-plate energy over the whole net (no loose CVs outside)
    and once more with every edge sample's (u, v) moved to its closest point;
    G2 after that by damped Gauss-Newton. Then `geo` times the same with the
    fairness measured on the surface just solved (_geo_gram: bending plus
    geo3 times the curvature variation) rather than in (u, v).
      tension  per side (loop order): the patch next to it is t^2 times as
               stiff against bending (1 = as solved; > 1 the surface keeps the
               edge's direction further in, < 1 it turns away sooner) -- the
               edge rows are untouched, so continuity holds as without
      guides   [(m, 3) points]: curves inside the opening the surface passes
               through (their (u, v) from the plain solve's closest points)
      report   a dict to fill: guide deviation, per-side fit errors.
    Returns (G (nu, nv, 3), (Uu, Uv), idx)."""
    if idx is None:
        idx, P, cum = corners(sides)
    else:
        P, cum = _flat_loop(sides)
    nu, nv = (int(n[0]), int(n[1])) if isinstance(n, (tuple, list)) else (int(n), int(n))
    n2 = (nu, nv)
    allP = np.vstack([sd.P for sd in sides])
    size = float(np.linalg.norm(np.ptp(allP, axis=0))) or 1.0
    c = P.mean(axis=0)
    Uu, Uv = geom.clamped_uniform_knots(nu, p), geom.clamped_uniform_knots(nv, p)
    U2 = (Uu, Uv)
    N = nu * nv
    m = margin / (1 + 2 * margin)
    lay = layout_quad(sides, idx, kappa, kfun)
    allq = np.vstack(lay)
    lo, hi = allq.min(axis=0), allq.max(axis=0)
    uv = np.vstack([m + (1 - 2 * m) * (q - lo) / np.maximum(hi - lo, 1e-12) for q in lay])
    loopP = (allP - c) / size
    tens = [1.0 if tension is None or si >= len(tension) or tension[si] is None else float(tension[si])
            for si in range(len(sides))]
    tens_on = [si for si, t in enumerate(tens) if abs(t - 1.0) > 1e-6]
    F2, F3, Dg = _gram(n2, p)
    wfun = _stiffness(sides, uv, tens, TENSION_REACH) if tens_on else None
    if wfun is not None:
        # tension: the bending energy weighted up / down next to those edges
        F2 = _gram_weighted(n2, p, wfun)

    def per_coord(M, Z=None):
        # the (N, N) form M acting on each coordinate alike: kron(M, I3) (added to Z)
        Z = np.zeros((3 * N, 3 * N)) if Z is None else Z
        for d in range(3):
            Z[d::3, d::3] += M
        return Z

    def fairness(F):
        # fairness, net fairness and a tiny pull to zero act on each coordinate alike
        return per_coord((fw * fair) ** 2 * ((1.0 - f3) * F + f3 * F3) + (cw * fair) ** 2 * Dg + 1e-12 * np.eye(N))

    n_g0 = len(loopP)
    n_g1 = sum(len(sd.P) for sd, cn in zip(sides, conts) if cn >= 1 and sd.has_surface) or 1
    n_g2 = sum(len(sd.P) for sd, cn in zip(sides, conts) if cn >= 2 and sd.has_surface) or 1
    has_g2 = any(cn >= 2 and sd.has_surface for sd, cn in zip(sides, conts))
    gP = [(np.asarray(gq, float) - c) / size for gq in (guides or []) if len(gq) >= 2]
    # a guide holds less right at the loop (where it meets an edge, the edge's
    # own G0 / G1 decide: a guide arriving at a slope can't override them)
    gfade = [_smooth(np.min(np.linalg.norm(gq[:, None, :] - loopP[None, ::2, :], axis=2), axis=1) / GUIDE_FADE)
             * (1.0 - GUIDE_MIN) + GUIDE_MIN for gq in gP]
    # G2 targets: the neighbour's normal curvature straight across the edge
    # (b = N x T, the direction contlabels.measure checks), scaled to the
    # normalised coordinates
    W1, _ = corner_eases(sides, conts)
    g2 = {}
    for si, (sd, cn) in enumerate(zip(sides, conts)):
        if cn >= 2 and sd.has_surface:
            Nt = np.asarray(sd.N, float)
            bb = _cross_dirs(sd)
            T3 = np.gradient(np.asarray(sd.P, float), axis=0)
            T3 /= np.maximum(np.linalg.norm(T3, axis=1), 1e-30)[:, None]
            g2[si] = (Nt, bb, np.asarray(sd.K, float) * size,
                      np.minimum(_g2_ease(sides, si, bb, size), W1[si]), T3)

    def side_slices():
        o = 0
        for si, sd in enumerate(sides):
            yield si, sd, slice(o, o + len(sd.P))
            o += len(sd.P)

    def constraints(uv, guide=None):
        """G0 and G1 (linear in the net) as normal equations (A^T A, A^T r,
        r.r), plus each G2 side's basis rows at its samples. guide: [(uv,
        points)]. Summed block by block over the CVs each block touches;
        positions (G0, guides) act on each coordinate alike, so theirs is one
        (N, N) form."""
        AtA, M0, R0r = np.zeros((3 * N, 3 * N)), np.zeros((N, N)), np.zeros((N, 3))
        rr, g2B = 0.0, {}

        def position(R0, w, Pn):
            nonlocal rr
            ata_add(M0, R0 * w[:, None])
            R0r[:] += (R0 * (w * w)[:, None]).T @ Pn
            rr += float(np.sum((w[:, None] * Pn) ** 2))

        for si, sd, sl in side_slices():
            cn = conts[si]
            B = _eval_rows(uv[sl], n2, p, U2, PAIRS2)
            # position holds harder where an eased corner lets the shape bend
            position(B[:, 0], g0w / np.sqrt(n_g0) * (1.0 + g0ease * (1.0 - W1[si])), (sd.P - c) / size)
            if cn >= 1 and sd.has_surface:
                Nn = np.asarray(sd.N, float)
                w1 = g1w / np.sqrt(n_g1) * W1[si]
                for Rr in (B[:, 1], B[:, 2]):
                    ata_add(AtA, (w1[:, None, None] * Rr[:, :, None] * Nn[:, None, :]).reshape(len(Rr), 3 * N))
            if si in g2:
                ks = [k for k in range(1, len(B) - 1) if g2[si][3][k] > 0.0]
                g2B[si] = (ks,) + tuple(B[ks, q] for q in range(1, 6))
        for (guv, gq), gfw in zip(guide or [], gfade):
            position(_eval_rows(guv, n2, p, U2, ((0, 0),))[:, 0], gw * g0w / np.sqrt(n_g0) * gfw, gq)
        per_coord(M0, AtA)
        return AtA, R0r.reshape(-1), rr, g2B

    def base_system(S3, con):
        """The constraints' normal equations plus the fairness forms S3."""
        AtA, Atr, rr, g2B = con
        return S3 + AtA, Atr, rr, g2B

    w2 = g2w / np.sqrt(n_g2)

    def g2_terms(X, g2B, rows=True):
        """Per G2 side: (the residual n . S_ww / |S_w|^2 - kappa, weighted;
        the rows of its linearisation at X: n . S_ww with w fixed so that
        S_w = b, = kappa |S_w|^2). All of a side's samples at once: (res,
        rows (3 per sample), right-hand sides), rows None without any (or
        when not asked for: rows=False, the residuals alone)."""
        res, Zs, rs = [], [], []
        for si, (ks, Ru, Rv, Ruu, Ruv, Rvv) in g2B.items():
            if not len(ks):
                continue
            Nt, bb, kap, ease, T3 = (np.asarray(a, float)[ks] for a in g2[si])
            Su, Sv = Ru @ X, Rv @ X
            ac = lstsq2(Su, Sv, bb)
            a_, c_ = ac[:, :1], ac[:, 1:]
            Sw = a_ * Su + c_ * Sv
            q = np.sum(Sw * Sw, axis=1)
            q[q == 0.0] = 1e-30
            ew = w2 * ease
            Rww = a_ * a_ * Ruu + 2 * a_ * c_ * Ruv + c_ * c_ * Rvv
            res.append(ew * (np.sum(Nt * (Rww @ X), axis=1) / q - kap))
            if not rows:
                continue
            # keep w pointing straight across: S_w has no part along the
            # edge and keeps its length (else the linear model can meet
            # the curvature along a direction that has turned away from b)
            Rw = a_ * Ru + c_ * Rv
            Z = np.stack([Rww[:, :, None] * Nt[:, None, :], Rw[:, :, None] * T3[:, None, :],
                          Rw[:, :, None] * bb[:, None, :]], axis=1)
            Zs.append((ew[:, None, None, None] * Z).reshape(-1, 3 * N))
            rs.append((ew[:, None] * np.stack([kap * q, np.zeros_like(q), np.sqrt(q)], axis=1)).reshape(-1))
        res = np.concatenate(res) if res else np.zeros(0)
        if not Zs:
            return res, None, None
        return res, np.vstack(Zs), np.concatenate(rs)

    # position and tangency first (once more after moving the edge samples to
    # their closest points), then G2 by damped Gauss-Newton: a step is only
    # taken (shortened as needed) if the true objective -- the linear terms
    # plus the exact curvature mismatch -- drops
    def _solve(Q, b):
        return net_solve(Q, b, nu, p)

    def reproj(X, uv):
        return np.clip(_project(X, uv, loopP, n2, p, U2, it=3), 0.02, 0.98)

    def gauss_newton(X, Q0, b0, c0, g2B):
        """G2 by damped Gauss-Newton on the system (Q0, b0, c0, g2B)."""
        if not has_g2:
            return X
        def energy(X):
            x = X.reshape(-1)
            return float(x @ (Q0 @ x)) - 2 * float(b0 @ x) + c0 + float(np.sum(g2_terms(X, g2B, rows=False)[0] ** 2))
        E = energy(X)
        for _ in range(gn_iters):
            _, Zm, rs = g2_terms(X, g2B)
            if Zm is None:
                break
            Q = Q0.copy()
            ata_add(Q, Zm)
            Xl = _solve(Q, b0 + Zm.T @ rs).reshape(N, 3)
            step = Xl - X
            E_old = E
            for t in (1.0, 0.5, 0.25, 0.125, 0.0625):
                Et = energy(X + t * step)
                if Et < E:
                    X, E = X + t * step, Et
                    break
            else:
                break
            if t == 1.0 and float(np.linalg.norm(step)) < 1e-9:
                break
            if E_old - E < GN_RTOL * E_old:
                break           # converged: the remaining steps gain too little to show
        return X

    uv0 = uv
    con0 = constraints(uv0)

    def run(S3):
        """One solve with the fairness S3: the plain solve, the edge samples
        moved to their closest points (and the guides placed), solved again,
        then G2. Returns (X, the samples' (u, v), the guides)."""
        sys0 = base_system(S3, con0)
        X = _solve(*sys0[:2]).reshape(N, 3)
        u_, guide = uv0, None
        if reproject or gP:
            if reproject:
                u_ = reproj(X, u_)
            if gP:
                # guides: (u, v) where the plain surface comes closest, then
                # once with them (and everything moved to its closest point)
                guide = [(np.clip(_project(X, _grid_uv(X, n2, p, U2, gq), gq, n2, p, U2, it=4), 0.0, 1.0), gq)
                         for gq in gP]
                for _ in range(GUIDE_PASSES):
                    Q, b, _, _ = base_system(S3, constraints(u_, guide))
                    X = _solve(Q, b).reshape(N, 3)
                    u_ = reproj(X, u_)
                    guide = [(np.clip(_project(X, guv, gq, n2, p, U2, it=3), 0.0, 1.0), gq) for guv, gq in guide]
            sys0 = base_system(S3, constraints(u_, guide))
            X = _solve(*sys0[:2]).reshape(N, 3)
        return gauss_newton(X, *sys0), u_, guide

    X, uv, guide = run(fairness(F2))
    # then with the fairness measured on that surface (the layout's bends no
    # longer imprint), and once more on the result: the metric settles
    done = 0
    for _ in range(geo):
        Q2, Q3 = _geo_gram(X, n2, p, wfun)
        t2 = float(np.trace(Q2))
        if not t2 > 1e-30:
            break
        X, uv, guide = run(fairness(float(np.trace(F2)) / t2 * (Q2 + geo3 * Q3)))
        done += 1
    if report is not None:
        report.update(cvs=n2, idx=list(map(int, idx)), geo=done)
        if guide:
            # away from the edges (near them the edges' own continuity wins)
            dev = []
            for (guv, gq), gfw in zip(guide, gfade):
                d = np.linalg.norm(_eval_rows(guv, n2, p, U2, ((0, 0),))[:, 0] @ X - gq, axis=1)
                far = gfw >= 0.999
                dev.append(float(np.max(d[far] if far.any() else d)) * size)
            report["guide_dev"] = max(dev)
        if report.get("check"):
            report["errors"] = _fit_errors(X, reproj(X, uv), sides, conts, n2, p, U2, c, size, W1, g2)
    G = X * size + c
    return G.reshape(nu, nv, 3), U2, idx


def _fit_errors(X, uv, sides, conts, n2, p, U2, c, size, W1, g2):
    """Per side (loop order): (gap relative to the side's length, largest
    angle to the neighbour's normal in degrees, largest curvature error
    relative to the neighbour's largest curvature), outside eased zones --
    what contlabels.measure checks, on the untrimmed net."""
    out, o = [], 0
    for si, sd in enumerate(sides):
        q = uv[o:o + len(sd.P)]
        o += len(sd.P)
        B = _eval_rows(q, n2, p, U2, PAIRS2)
        S = B[:, 0] @ X * size + c
        L = float(np.sum(np.linalg.norm(np.diff(sd.P, axis=0), axis=1))) or 1.0
        gap = float(np.max(np.linalg.norm(S - sd.P, axis=1))) / L
        ang = cur = 0.0
        if conts[si] >= 1 and sd.has_surface:
            Su, Sv = B[:, 1] @ X, B[:, 2] @ X
            nn = np.cross(Su, Sv)
            nn /= np.maximum(np.linalg.norm(nn, axis=1), 1e-30)[:, None]
            a = np.degrees(np.arccos(np.clip(np.abs(np.sum(nn * np.asarray(sd.N, float), axis=1)), 0, 1)))
            keep = W1[si] >= 0.999
            keep[[0, -1]] = False
            ang = float(np.max(a[keep])) if keep.any() else 0.0
            if conts[si] >= 2 and si in g2:
                Nt, bb, kap, ease, _ = g2[si]
                ks = [k for k in range(1, len(B) - 1) if ease[k] >= 0.999]
                ac = lstsq2(Su[ks], Sv[ks], np.asarray(bb, float)[ks])
                a_, c_ = ac[:, :1], ac[:, 1:]
                Sw = a_ * Su[ks] + c_ * Sv[ks]
                Sww = a_ * a_ * (B[ks, 3] @ X) + 2 * a_ * c_ * (B[ks, 4] @ X) + c_ * c_ * (B[ks, 5] @ X)
                q2 = np.sum(Sw * Sw, axis=1)
                q2[q2 == 0.0] = 1e-30
                errs = np.abs(np.sum(np.asarray(Nt, float)[ks] * Sww, axis=1) / q2 - np.asarray(kap, float)[ks]).tolist()
                # as contlabels.measure: relative to the neighbour's largest
                # curvature, at least 0.05 / the edge's length
                ref = max(float(np.max(np.abs(kap))), 0.05 * size / L)
                cur = (max(errs) / ref) if errs else 0.0
        out.append((gap, ang, cur))
    return out


def steepness(sides, conts, nrm):
    """Largest angle (degrees) between the projection direction and the G1
    neighbours' tangent planes' normals."""
    a = 0.0
    for sd, cn in zip(sides, conts):
        if cn >= 1 and sd.has_surface:
            a = max(a, float(np.degrees(np.arccos(np.clip(np.abs(np.asarray(sd.N) @ nrm), 0, 1))).max()))
    return a


# --------------------------------------------------------------------------
# Feature
# --------------------------------------------------------------------------

def _orient(G, U, sides):
    """Flip the net (reverse u) when its normal disagrees with the neighbours'
    faces at the G1 samples. U: knots, or (Uu, Uv)."""
    _, ext = _k()
    Uu, Uv = U if isinstance(U, tuple) else (U, U)
    p = len(Uu) - G.shape[0] - 1
    q = len(Uv) - G.shape[1] - 1
    s = ext.surface_from_cvs(p, q, [tuple(map(float, x)) for x in G.reshape(-1, 3)], Uu, Uv)
    from .square import _foot, grid_starts
    picks = [(sd, k) for sd in sides if sd.Nout is not None for k in range(1, len(sd.P) - 1, max(1, len(sd.P) // 6))]
    starts = grid_starts(s, [sd.P[k] for sd, k in picks]) if picks else []
    votes = 0.0
    for (sd, k), uv0 in zip(picks, starts):
        _, su, sv = s.derivatives(*_foot(s, sd.P[k], uv0))
        votes += float(np.cross(su, sv) @ sd.Nout[k])
    return (G[::-1].copy(), True) if votes < 0 else (G, False)


def _on_surface(sm, ext, surf, kc, rev=False, n=40):
    """The input curve pulled onto the patch surface (trim curve), running
    along the loop."""
    from .drawtools import interpolate
    t0, t1 = kc.parameter_range()
    Q = []
    ts = np.linspace(t0, t1, n)
    for t in (ts[::-1] if rev else ts):
        q = sm.surface_closest_point(surf, tuple(map(float, kc.evaluate(t))))
        q = q[0] if isinstance(q, tuple) else q
        Q.append([q.x, q.y, q.z])
    X, deg = interpolate(np.array(Q), 3)
    return ext.curve_from_cvs(deg, [tuple(map(float, x)) for x in X], geom.clamped_uniform_knots(len(X), deg), None)


def _reverse(ext, kc):
    d = ext.curve_cvs(kc)
    U = list(d["knots"])
    a, b = U[0], U[-1]
    W = d.get("weights")
    return ext.curve_from_cvs(d["degree"], list(d["points"])[::-1], [a + b - k for k in reversed(U)],
                              list(W)[::-1] if W else None)


def guide_indices(p):
    """Inputs of a Patch that are guide curves (inside the opening), not edges
    of the loop."""
    return sorted({int(i) for i in (p or {}).get("pt_guides", [])})


def loop_sides(inputs, p=None, m=None):
    """A Patch's loop: (sides in loop order, order [(input index, reversed)],
    guide point arrays [(k, (m, 3))], closing gap, size). m: samples per
    edge (None: about 160 around the loop, by length -- what the build uses)."""
    from .square import sample_side
    gi = set(guide_indices(p))
    loop = [k for k in range(len(inputs)) if k not in gi]
    if m is None:
        lengths = []
        for k in loop:
            kc = features.input_curve(*inputs[k])
            t0, t1 = kc.parameter_range()
            pts = np.array([kc.evaluate(t) for t in np.linspace(t0, t1, 16)])
            lengths.append(float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        total = sum(lengths) or 1.0
        counts = [int(np.clip(160 * L / total, 10, 80)) for L in lengths]
    else:
        counts = [m] * len(loop)
    raw = []
    for k, cnt in zip(loop, counts):
        o, r = inputs[k]
        raw.append(sample_side(o, r, cnt))
        raw[-1].src = (o.name, r)
    order, gap = chain(raw)
    sides = [raw[j].reversed() if rv else raw[j] for j, rv in order]
    order = [(loop[j], rv) for j, rv in order]
    size = float(np.linalg.norm(np.ptp(np.vstack([sd.P for sd in raw]), axis=0))) or 1.0
    guides = []
    for k in sorted(gi):
        if k >= len(inputs):
            continue
        kc = features.input_curve(*inputs[k])
        t0, t1 = kc.parameter_range()
        guides.append((k, np.array([kc.evaluate(t) for t in np.linspace(t0, t1, 48)], float)))
    return sides, order, guides, gap, size


def _side_arc(sd):
    s_ = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(sd.P, axis=0), axis=1))])
    return s_ / (s_[-1] or 1.0)


def corner_loop_indices(sides, order, kf):
    """Flat-loop indices of corners given as [input index, fraction along the
    input in its own direction] (as stored in pt_corners)."""
    M = sum(len(sd.P) - 1 for sd in sides)
    out = []
    for k, f in kf:
        off = 0
        for (kk, rv), sd in zip(order, sides):
            if kk == int(k):
                fl = 1.0 - float(f) if rv else float(f)
                out.append((off + int(np.argmin(np.abs(_side_arc(sd) - fl)))) % M)
                break
            off += len(sd.P) - 1
    return out


def loop_points(sides, order, idx):
    """[[input index, fraction along it, (x, y, z), fraction of the loop's
    length]] of flat-loop indices."""
    _, cum = _flat_loop(sides)
    out = []
    for i in idx:
        off = 0
        for (k, rv), sd in zip(order, sides):
            m = len(sd.P)
            if off <= i < off + m - 1:
                fl = float(_side_arc(sd)[i - off])
                out.append([int(k), round(1.0 - fl if rv else fl, 6), [float(x) for x in sd.P[i - off]],
                            float(cum[i] / (cum[-1] or 1.0))])
                break
            off += m - 1
    return out


def junctions(sides, order):
    """Flat-loop indices where one input meets the next."""
    out, off = [], 0
    for sd in sides:
        out.append(off)
        off += len(sd.P) - 1
    return out


def auto_conts(inputs, n=20, deg=3, fair=1.0, limit=45.0, g2=True, p=None):
    """Per input: "G1" where a position-only fill already meets the neighbour at
    less than `limit` degrees (tangency is what's wanted there), "G0" where the
    neighbour stands across the fill (a floor under a wall, a lid) or the input
    is a curve (and on guide curves)."""
    from .square import _foot
    _, ext = _k()
    sides, order, _, _, _ = loop_sides(inputs, p, 24)
    idx = None
    if p and p.get("pt_corners"):
        idx = corners(sides, fixed=corner_loop_indices(sides, order, p["pt_corners"]))[0]
    G, (Uu, Uv), _ = solve_quad(sides, [0] * len(sides), n=n, p=deg, fair=fair, idx=idx, geo=0)
    s = ext.surface_from_cvs(deg, deg, [tuple(map(float, x)) for x in G.reshape(-1, 3)], Uu, Uv)
    out = ["G0"] * len(inputs)
    for (k, _), sd in zip(order, sides):
        o, r = inputs[k]
        if not sd.has_surface or o.type == 'CURVE':
            continue
        angs, prev = [], None
        for i in range(2, len(sd.P) - 2, 3):
            prev = _foot(s, sd.P[i], prev)
            _, su, sv = s.derivatives(*prev)
            nn = np.cross(su, sv)
            nn /= np.linalg.norm(nn) or 1.0
            angs.append(np.degrees(np.arccos(min(1.0, abs(float(nn @ sd.N[i]))))))
        if angs and float(np.median(angs)) < limit:
            out[k] = "G1"
    # G2 where it can hold: not at an edge with a corner where the next side
    # leaves across it with another curvature (eased there, see _g2_ease)
    if g2:
        for si, ((k, _), sd) in enumerate(zip(order, sides)):
            if out[k] == "G1" and float(np.min(_g2_ease(sides, si, _cross_dirs(sd), 1.0))) >= 1.0:
                out[k] = "G2"
    return out


def _cross_dirs(sd):
    """Unit direction straight across the edge into the patch (N x T, on the
    side of the neighbour's cross direction B)."""
    Nt = np.asarray(sd.N, float)
    T3 = np.gradient(np.asarray(sd.P, float), axis=0)
    bb = np.cross(Nt, T3)
    bb /= np.maximum(np.linalg.norm(bb, axis=1), 1e-30)[:, None]
    return bb * np.where(np.sum(bb * np.asarray(sd.B, float), axis=1) < 0, -1.0, 1.0)[:, None]


def input_eases(inputs, conts, reach=0.5, p=None):
    """Corners where an input's continuity is eased, in the input's own
    direction: {input index: [(end (0 start / 1 end), reach, other input
    index, "G1" or "G2", degrees or None, the corner's position)]}.
    conts: "G0"/"G1"/"G2" per input."""
    sides, order, _, _, _ = loop_sides(inputs, p, 24)
    sc = [CONT.get(conts[k], 0) if sd.has_surface else 0 for (k, _), sd in zip(order, sides)]
    nb = len(sides)
    out = {}

    def put(si, side_end, j, kind, deg):
        k, rv = order[si]
        end = (1 if side_end == 0 else 0) if rv else (0 if side_end == 0 else 1)
        out.setdefault(k, []).append((end, reach, order[j][0], kind, deg,
                                      np.asarray(sides[si].P[side_end], float).copy()))

    _, notes = corner_eases(sides, sc, reach)
    g1 = set()
    for si, j, deg, e in notes:
        put(si, e, j, "G1", deg)
        g1.add((si, e))
    for si, sd in enumerate(sides):
        if sc[si] < 2:
            continue
        w = _g2_ease(sides, si, _cross_dirs(sd), 1.0, reach)
        for e, j in ((0, (si - 1) % nb), (-1, (si + 1) % nb)):
            if w[e] < 1.0 and (si, e) not in g1:
                put(si, e, j, "G2", None)
    return out


def eased_edges(inputs, conts=None, p=None):
    """[(input index, the other input's index)] for every edge whose G1 / G2
    is eased toward a corner."""
    conts = conts or ["G2"] * len(inputs)
    return [(k, j) for k, lst in input_eases(inputs, conts, p=p).items() for _, _, j, _, _, _ in lst]


info = {}
AUTO_COUNTS = (8, 10, 12, 14, 16, 20, 24, 28, 32)


def lightest(sides, sc, deg, fair, idx, cap, **kw):
    """The fewest CVs per direction (nu, nv) at which every edge holds its
    continuity within the checks' tolerances (contlabels: gap, angle,
    curvature), at most cap = (nu, nv). Each direction grows on its own:
    the chains along u (the 1st and 3rd) tell nu, the others nv."""
    from .contlabels import TOL_ANGLE, TOL_CURV, TOL_GAP_REL
    P, cum = _flat_loop(sides)
    # which chain (0..3) every side lies on: by its first sample's loop index
    chain_of, off = [], 0
    for sd in sides:
        mid = (off + (len(sd.P) - 1) // 2) % len(P)
        q = 3
        for j in range(4):
            a, b = idx[j], idx[(j + 1) % 4]
            if (a <= mid < b) if a < b else (mid >= a or mid < b):
                q = j
                break
        chain_of.append(q)
        off += len(sd.P) - 1
    lo = max(deg + 3, AUTO_COUNTS[0])
    steps = [c for c in AUTO_COUNTS if c >= lo]
    geo = kw.pop("geo", GEO_PASSES)

    def at(iu, iv, g):
        """Solve at the iu-th / iv-th count: ((G, U, report), the worst error
        over its limit, whether the u / v chains miss and can grow)."""
        nu, nv = min(steps[iu], cap[0]), min(steps[iv], cap[1])
        rep = {"check": True}
        G, U, _ = solve_quad(sides, sc, n=(nu, nv), p=deg, fair=fair, idx=idx, report=rep, geo=g, **kw)
        worst, bad_u, bad_v = 0.0, False, False
        for (gap, ang, cur), q, cn in zip(rep["errors"], chain_of, sc):
            # a little inside the checks' limits: the trim adds its own error
            r = max(gap / (0.5 * TOL_GAP_REL), ang / (0.7 * TOL_ANGLE) if cn >= 1 else 0.0,
                    cur / (0.7 * TOL_CURV) if cn >= 2 else 0.0)
            worst = max(worst, r)
            if r > 1.0:
                bad_u |= q in (0, 2)
                bad_v |= q in (1, 3)
        return (G, U, rep), worst, bad_u and nu < cap[0] and iu + 1 < len(steps), \
            bad_v and nv < cap[1] and iv + 1 < len(steps)

    # the counts by the quick solve (fairness in (u, v) only)
    iu = iv = 0
    for _ in range(2 * len(steps)):
        best, worst, gu, gv = at(iu, iv, 0)
        if not (gu or gv):
            break
        iu += gu
        iv += gv
    if not geo:
        return best
    # then in full at those counts, growing on only while more CVs help (an
    # edge right at its limit at every count would otherwise grow to the cap)
    best, worst, gu, gv = at(iu, iv, geo)
    while gu or gv:
        iu += gu
        iv += gv
        nxt = at(iu, iv, geo)
        if nxt[1] > 0.9 * worst:
            break
        best, worst, gu, gv = nxt
    return best


@features.builder("patch")
def build_patch(inputs, p):
    sm, ext = _k()
    conts = [CONT.get(c, 1) for c in p.get("cont", ["G1"] * len(inputs))]
    while len(conts) < len(inputs):
        conts.append(1)
    tension = [float(x) for x in p.get("pt_tension", [])]
    deg, fair = int(p.get("pt_degree", 3)), float(p.get("pt_fair", 1.0))
    nu = int(p.get("pt_cvs", 20))
    nv = int(p.get("pt_cvs_v", 0)) or nu
    sides, order, guides, gap, size = loop_sides(inputs, p)
    if gap > 1e-3 * size:
        raise RuntimeError(f"the picked edges don't form a closed loop (gap {gap:.3g})")
    sc = [conts[k] for k, _ in order]
    st = [tension[k] if k < len(tension) else 1.0 for k, _ in order]
    # the loop laid out flat (works for any opening); "PROJECT" = height field
    # over the plane the opening faces (older patches / flat-ish openings)
    method = p.get("pt_method", "QUAD")
    rep = {}
    idx = None
    if method == "PROJECT":
        G, U, _ = solve(sides, sc, n=nu, p=deg, fair=fair)
        used = "projected"
    elif method == "DEVELOP":
        G, U = solve3d(sides, sc, n=nu, p=deg, fair=fair)
        used = "developed"
    else:
        fixed = corner_loop_indices(sides, order, p.get("pt_corners") or [])
        idx = corners(sides, fixed=fixed)[0]
        # fairness measured on the surface itself (default; older patches too)
        kw = dict(tension=st, guides=[g for _, g in guides], geo=GEO_PASSES if p.get("pt_surface_fair", True) else 0)
        if p.get("pt_auto_cvs"):
            G, U, rep = lightest(sides, sc, deg, fair, idx, (nu, nv), **kw)
        else:
            rep = {}
            G, U, idx = solve_quad(sides, sc, n=(nu, nv), p=deg, fair=fair, idx=idx, report=rep, **kw)
        used = "four-sided"
    Uu, Uv = U if isinstance(U, tuple) else (U, U)
    G, flipped = _orient(G, (Uu, Uv), sides)
    pts = [tuple(map(float, x)) for x in G.reshape(-1, 3)]
    surf = ext.surface_from_cvs(deg, deg, pts, Uu, Uv)
    curves = [_on_surface(sm, ext, surf, features.input_curve(*inputs[k]), rv) for k, rv in order]
    info.clear()
    info.update(cvs=(int(G.shape[0]), int(G.shape[1])), degree=deg, sides=len(order), guides=len(guides),
                method=used, guide_dev=float(rep.get("guide_dev", 0.0)))
    if idx is not None:
        info["corners"] = loop_points(sides, order, idx)
        info["junctions"] = loop_points(sides, order, junctions(sides, order))
        info["manual"] = bool(p.get("pt_corners"))
    return trim_to_loop(sm, ext, lambda: ext.surface_from_cvs(deg, deg, pts, Uu, Uv), curves,
                        float(np.ptp(G.reshape(-1, 3), axis=0).max()))


TRIM_FIT_REL = 1e-5     # UV trim curves within this fraction of the patch's size of its 3D edges


def _copy_curve(ext, kc):
    d = ext.curve_cvs(kc)
    return ext.curve_from_cvs(d["degree"], list(d["points"]), list(d["knots"]), d.get("weights") or None)


def trim_to_loop(sm, ext, make_surface, curves, size):
    """The surface trimmed to the loop of 3D curves (on the surface, in loop
    order; tried reversed too). The UV trim curves are fitted within
    TRIM_FIT_REL of the size: the kernel's own trim fits them within 0.01
    model units, so on a small part a UV curve strayed up to 0.5 mm from its
    edge, and Edit CVs (which re-lifts edges from the UV curves) pulled the
    edge off its neighbour. Falls back to the kernel's trim (an older kernel
    module without the call, or a failure)."""
    tol = max(TRIM_FIT_REL * (size or 1.0), 1e-9)
    attempts = []
    if hasattr(ext, "trim_surface_with_3d_curves_tol"):
        attempts += [(True, False), (True, True)]
    attempts += [(False, False), (False, True)]
    last = None
    for tight, reverse in attempts:
        # the call consumes its curves (even when it fails): fresh copies each time
        cs = [_reverse(ext, c) for c in curves[::-1]] if reverse else [_copy_curve(ext, c) for c in curves]
        try:
            if tight:
                return ext.trim_surface_with_3d_curves_tol(make_surface(), [len(cs)], cs, tol, 0.01)
            return sm.trim_surface_with_3d_curves(make_surface(), [len(cs)], cs)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"trimming the patch to the loop failed ({last})")


def store_info(obj):
    """Keep the last build's corners / net size on the object (viewport
    corner markers, Feature panel)."""
    import json
    obj["nproto_patch_info"] = json.dumps(info)


def split_loop(inputs, tol_rel=1e-3):
    """Which picked inputs close the opening and which are guide curves inside
    it: inputs whose ends don't meet another input's end are guides (repeated
    until none is left dangling); if what remains still doesn't chain into
    one loop (a guide running corner to corner), the first picks that close a
    loop are the loop. Returns the guides' indices."""
    ends = []
    for o, r in inputs:
        kc = features.input_curve(o, r)
        t0, t1 = kc.parameter_range()
        ends.append((np.array(kc.evaluate(t0), float), np.array(kc.evaluate(t1), float)))
    allp = np.array([e for pr in ends for e in pr])
    size = float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0
    tol = tol_rel * size

    def simple_cycle(keep):
        if len(keep) < 2:
            return False
        for k in keep:
            for e in ends[k]:
                hits = sum(1 for j in keep for f in ends[j] if np.linalg.norm(e - f) <= tol) - 1
                closed = np.linalg.norm(ends[k][0] - ends[k][1]) <= tol
                if hits != (2 if closed else 1):
                    return False
        return True

    def prune(keep):
        keep = list(keep)
        while True:
            drop = [k for k in keep if any(
                not any(np.linalg.norm(e - f) <= tol for j in keep if j != k for f in ends[j])
                for e in ends[k]) and np.linalg.norm(ends[k][0] - ends[k][1]) > tol]
            if not drop:
                return keep
            keep = [k for k in keep if k not in drop]

    keep = prune(range(len(inputs)))
    if not simple_cycle(keep):
        for n_ in range(2, len(inputs) + 1):
            pre = prune(range(n_))
            if simple_cycle(pre):
                keep = pre
                break
    return [k for k in range(len(inputs)) if k not in keep]


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty  # noqa: E402

from bpy.props import CollectionProperty  # noqa: E402

from .feature_ui import PATCH_ALL, SURFACE_FAIR_DESC, NprotoPatchEdge  # noqa: E402
from .ops import PickMixin, _decode_inputs, _kernel_ok, _select_only, polylines_closed  # noqa: E402


def draw_settings(lay, data, edges, inf, all_prop, obj=None, h=None):
    """A Patch's settings, laid out the same in the Feature panel (data: the
    object's feature settings; obj, h given) and in the Patch operator's redo
    panel (data: the operator)."""
    col = lay.column(align=True)
    row = col.row(align=True)
    row.prop(data, "pt_cvs", text="CVs U")
    row.prop(data, "pt_cvs_v", text="CVs V")
    row = col.row(align=True)
    row.prop(data, "pt_auto_cvs", toggle=True)
    row.prop(data, "pt_degree")
    row = col.row(align=True)
    row.prop(data, "pt_fair")
    row.prop(data, "pt_surface_fair", text="On Surface", toggle=True)
    waiting = obj is not None and features.pending(obj)
    if inf.get("cvs"):
        txt = f"{inf['cvs'][0]} x {inf['cvs'][1]} CVs · degree {inf.get('degree', '?')}"
        if waiting:
            txt += " · waiting for Rebuild"
        elif data.pt_auto_cvs:
            txt += " (lightest that holds)"
        lay.label(text=txt, icon='MESH_GRID')

    def cells(layout):
        sp = layout.split(factor=0.4, align=True)
        a = sp.row(align=True)
        sp2 = sp.split(factor=0.7, align=True)
        return a, sp2.row(align=True), sp2.row(align=True)
    tab = lay.column(align=True)
    a, b, c = cells(tab)
    a.label(text="Edge")
    b.label(text="Continuity")
    c.label(text="Tension")
    a, b, c = cells(tab)
    a.label(text="All edges")
    for v in ("AUTO", "G0", "G1", "G2"):
        b.prop_enum(data, all_prop, v)
    c.active = len({round(float(e.tension), 6) for e in edges}) <= 1      # grey: edges differ
    c.prop(data, "pt_tension_all", text="")
    for e in edges:
        a, b, c = cells(tab)
        a.label(text=e.label)
        if e.curve:
            b.label(text="G0 (curve)")
        else:
            b.prop(e, "cont", expand=True)
        c.prop(e, "tension", text="")

    p = h["params"] if h else {}
    r_ = lay.row(align=True)
    manual = bool(p.get("pt_corners"))
    r_.label(text="Corners: set by hand" if manual else "Corners: automatic", icon='PIVOT_BOUNDBOX')
    if obj is not None and manual:
        r_.operator("nproto.patch_auto_corners", text="Automatic")
    if obj is not None:
        guides = guide_indices(p)
        gb = lay.column(align=True)
        for k in guides:
            if k < len(h["inputs"]):
                src, ref = h["inputs"][k]
                r_ = gb.row(align=True)
                r_.label(text=f"Guide: {src.name}" + (f" {ref}" if ref not in ("S0", "") else ""), icon='CURVE_DATA')
                r_.operator("nproto.patch_remove_guide", text="", icon='X').index = k
        if guides and inf.get("guide_dev") is not None and not waiting:
            gb.label(text=f"guides held within {inf['guide_dev']:.2g}")
        lay.operator("nproto.patch_add_guides", icon='ADD')
    elif inf.get("guides"):
        g = inf["guides"]
        lay.label(text=f"{g} guide curve{'s' if g > 1 else ''}, held within {inf.get('guide_dev', 0.0):.2g}",
                  icon='CURVE_DATA')
    hint = lay.column(align=True)
    hint.label(text="Labels: click = next, Ctrl+click = back", icon='MOUSE_LMB')
    hint.label(text="Dots: click = corner, Ctrl+click = auto", icon='MOUSE_LMB')


class NPROTO_OT_patch(PickMixin, bpy.types.Operator):
    """Fill a closed opening of any number of edges / curves with one trimmed surface,
    G0, G1 or G2 to each neighbouring surface (XNURBS-style). Pick the edges around the
    opening; it fills as soon as the loop closes (Enter fills what is picked). Shift-click
    the closing edge to keep picking: curves picked then (or ones that don't touch the loop's
    ends) are guides the surface passes through"""
    bl_idname = "nproto.patch"
    bl_label = "Patch"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 2
    pick_max = 256
    finish_on_min = False
    toggle_picks = True
    shift_extends = True
    extra_help = "   (click every edge around the opening; it fills when the loop closes; " \
                 "Shift-click the last edge to add guide curves, then Enter)"
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    default_cont: EnumProperty(name="All Edges", default="AUTO", items=PATCH_ALL,
                               description="Continuity of every edge at once (single edges: their rows below, "
                                           "or their labels in the viewport)")
    edges: CollectionProperty(type=NprotoPatchEdge, options={'SKIP_SAVE'})
    last_all: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    last_conts: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    last_tension: FloatProperty(default=-1.0, options={'HIDDEN', 'SKIP_SAVE'})
    pt_cvs: IntProperty(name="CVs U", default=24, min=6, max=40,
                        description="CVs in the surface's u direction (with Lightest: at most this many)")
    pt_cvs_v: IntProperty(name="CVs V", default=24, min=6, max=40,
                          description="CVs in the surface's v direction (with Lightest: at most this many)")
    pt_auto_cvs: BoolProperty(name="Lightest", default=False,
                              description="Use the fewest CVs (per direction) at which every edge still holds its "
                                          "continuity; CVs U / V are then the most it may use")
    pt_degree: IntProperty(name="Degree", default=5, min=2, max=5,
                           description="Degree of the patch: 5 (default) keeps the zebra smooth across the knot lines, 3 is lighter")
    pt_fair: FloatProperty(name="Fairness", default=1.0, min=0.0, soft_max=20.0,
                           description="Weight of the bending energy: higher = calmer, fuller surface; "
                                       "lower = follows the edges' tangents more tightly")
    pt_surface_fair: BoolProperty(name="Fair on Surface", default=True, description=SURFACE_FAIR_DESC)
    pt_tension_all: FloatProperty(name="Tension", default=1.0, min=0.25, max=4.0, soft_min=0.5, soft_max=2.0,
                                  description="Tension of every edge at once: how far each edge's direction carries "
                                              "into the patch (stiffer = more, softer = less against bending next to "
                                              "it); 1 = as solved")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if self.default_cont == "MIXED":
            self.default_cont = "AUTO"
        return self.start_picking(context, event)

    def closes_loop(self):
        byref = {(c[0].name, c[1]): c[2] for c in self._cands}
        polys = [byref[(o.name, r)] for o, r in self._picks if (o.name, r) in byref]
        if len(polys) < 2:
            return False
        allp = [q for pl in polys for q in pl]
        size = max((a - b).length for a in allp[:1] for b in allp) or 1.0
        tol = 1e-3 * size
        # guide curves picked before the loop closes dangle: leave them out
        while True:
            ends = [(pl[0], pl[-1]) for pl in polys]
            keep = [pl for k, pl in enumerate(polys)
                    if (pl[0] - pl[-1]).length <= tol or all(
                        any((e - f).length <= tol for j, pr in enumerate(ends) if j != k for f in pr)
                        for e in ends[k])]
            if len(keep) == len(polys):
                break
            polys = keep
        return len(polys) >= 2 and polylines_closed(polys, tol)

    def draw(self, context):
        draw_settings(self.layout, self, self.edges, info, "default_cont")

    def _edge_settings(self, ins, guides):
        """Per input (continuity, tension) from the redo panel's rows; the "All
        edges" row overrides every row when it changed (or on the first run)."""
        n = len(ins)
        loop = [k for k in range(n) if k not in guides]
        surf = [features.is_surface_edge(r) and o.type != 'CURVE' for o, r in ins]
        if [e.index for e in self.edges] != loop:            # first run (or other inputs)
            self.edges.clear()
            for k in loop:
                e = self.edges.add()
                e.index, e.label, e.curve = k, f"{ins[k][0].name} {ins[k][1]}", not surf[k]
            if self.default_cont == "MIXED":
                self.default_cont = "AUTO"
            redo_all = redo_t = True
        else:
            if self.default_cont == self.last_all and ",".join(e.cont for e in self.edges) != self.last_conts:
                self.default_cont = "MIXED"                  # an edge was set by itself
            redo_all = self.default_cont not in (self.last_all, "MIXED")
            redo_t = abs(self.pt_tension_all - self.last_tension) > 1e-9
        if redo_all:
            if self.default_cont == "AUTO":
                conts = auto_conts(ins, min(self.pt_cvs, 24), self.pt_degree, self.pt_fair, p={"pt_guides": guides})
            else:
                conts = [self.default_cont if surf[k] else "G0" for k in range(n)]
            for e in self.edges:
                e.cont = "G0" if e.curve or conts[e.index] not in ("G0", "G1", "G2") else conts[e.index]
        if redo_t:
            for e in self.edges:
                e.tension = self.pt_tension_all
        conts, tens = ["G0"] * n, [1.0] * n
        for e in self.edges:
            conts[e.index] = "G0" if e.curve else e.cont
            tens[e.index] = float(e.tension)
        self.last_all, self.last_tension = self.default_cont, self.pt_tension_all
        self.last_conts = ",".join(e.cont for e in self.edges)
        return conts, tens

    def execute(self, context):
        from .ops_sweep import _new_surface
        from .contlabels import measure
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            ins = _decode_inputs(self.inputs)
            if len(ins) < 2:
                raise ValueError("pick the edges around the opening")
            guides = split_loop(ins)
            conts, tens = self._edge_settings(ins, guides)
            params = {"cont": conts, "pt_cvs": self.pt_cvs, "pt_cvs_v": self.pt_cvs_v, "pt_degree": self.pt_degree,
                      "pt_fair": self.pt_fair, "pt_method": "QUAD", "pt_guides": guides}
            if self.default_cont != "MIXED":
                params["pt_all"] = self.default_cont
            if self.pt_auto_cvs:
                params["pt_auto_cvs"] = True
            if not self.pt_surface_fair:
                params["pt_surface_fair"] = False
            if any(abs(t - 1.0) > 1e-6 for t in tens):
                params["pt_tension"] = tens
            obj = _new_surface(context, "Patch", "patch", ins, params)
            store_info(obj)
            self._conts = conts
        except Exception as exc:
            self.report({'ERROR'}, f"Patch failed: {exc}")
            return {'CANCELLED'}
        eased, zones = [], {}
        try:
            for k, lst in sorted(input_eases(ins, conts, p=params).items()):
                for end, f, j, kind, deg, _ in lst:
                    why = f"tangent planes differ {deg:.2g}°" if kind == "G1" else "curvatures differ"
                    eased.append(f"{ins[k][0].name} {ins[k][1]} {kind} toward {ins[j][0].name} {ins[j][1]} ({why})")
                    zones.setdefault(k, []).append((end, f, why))
        except Exception as exc:
            print("NURBS patch ease check:", exc)
        bad = []
        from .contlabels import guide_skip
        skip = guide_skip(obj) if guides else None
        for k, ((o, r), c) in enumerate(zip(ins, conts)):
            try:
                chk = measure(obj, o, r, CONT[c], ease=zones.get(k), skip_near=skip if k in guides else None)
                if not chk["ok"] and not chk.get("eased"):
                    bad.append(f"{o.name} {r} {'guide' if k in guides else c} {chk['text']}")
            except Exception as exc:
                print("NURBS patch check:", exc)
        note = ("  Eased at corners where no surface can follow both neighbours: " + "; ".join(eased)) if eased else ""
        gnote = f" + {len(guides)} guide curve{'s' if len(guides) > 1 else ''}" if guides else ""
        if bad:
            self.report({'WARNING'}, "Patch: " + "; ".join(bad) + note)
        else:
            cnt = {k: sum(1 for i, c in enumerate(conts) if c == k and i not in guides) for k in ("G2", "G1", "G0")}
            self.report({'INFO'}, f"Patch: {len(ins) - len(guides)} edges hold their continuity ("
                        + ", ".join(f"{v} {k}" for k, v in cnt.items() if v) + ")" + gnote
                        + f", {info['cvs'][0]} x {info['cvs'][1]} CVs" + note)
        _select_only(context, [obj])
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Editing a Patch afterwards: corners, guides
# --------------------------------------------------------------------------

def patch_info(obj):
    import json
    try:
        return json.loads(obj.get("nproto_patch_info", "{}"))
    except ValueError:
        return {}


def _edit_params(obj, context, change):
    """Apply change(params, inputs) -> new inputs (or None: same) to a Patch's
    history and rebuild (while Pause Rebuild is on: only note that it waits);
    puts everything back if the rebuild fails."""
    import json
    h = features.history(obj)
    if h is None or h["op"] != "patch":
        raise ValueError("not a Patch with history")
    old_raw = obj[features.PROP_HIST]
    old_ins = list(h["inputs"])
    params = json.loads(json.dumps(h["params"]))
    ins = change(params, list(old_ins))

    def write(ins_, params_):
        if ins_ is not None:
            features.record(obj, "patch", ins_, params_)
        else:
            raw = json.loads(old_raw)
            raw["params"] = params_
            obj[features.PROP_HIST] = json.dumps(raw)

    write(ins, params)
    if features.paused(context):
        obj[features.PROP_PENDING] = 1
    else:
        try:
            features.rebuild(obj, context, force=True)
            store_info(obj)
        except Exception:
            write(old_ins if ins is not None else None, json.loads(old_raw)["params"])
            obj[features.PROP_HIST] = old_raw
            try:
                features.rebuild(obj, context, force=True)
                store_info(obj)
            except Exception:
                pass
            raise
    try:
        from .feature_ui import sync_from_history
        sync_from_history(obj)
    except Exception:
        pass
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _drop_input(params, k):
    """Remove input k from a Patch's per-input settings (indices above shift)."""
    for key in ("cont", "pt_tension"):
        if key in params and k < len(params[key]):
            params[key] = params[key][:k] + params[key][k + 1:]
    params["pt_guides"] = [g - (g > k) for g in params.get("pt_guides", []) if g != k]
    if params.get("pt_corners"):
        params["pt_corners"] = [[c - (c > k), f] for c, f in params["pt_corners"] if c != k]
        if len(params["pt_corners"]) < 4:
            params.pop("pt_corners")


class NPROTO_OT_patch_corner(bpy.types.Operator):
    """Make this junction a corner of the patch's four-sided base: the nearest corner moves here
    (Ctrl+click a dot: back to automatic corners)"""
    bl_idname = "nproto.patch_corner"
    bl_label = "Patch Corner"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    obj_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    index: IntProperty(default=-1, options={'HIDDEN', 'SKIP_SAVE'})
    reset: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def description(cls, context, props):
        obj = bpy.data.objects.get(props.obj_name)
        if obj is not None:
            pts = corner_points(obj)
            if 0 <= props.index < len(pts):
                c = pts[props.index]
                if c["corner"]:
                    return ("A corner of the patch's base (" + ("set by hand" if patch_info(obj).get("manual") else
                            "automatic") + "). Click another dot to move the nearest corner there; "
                            "Ctrl+click: automatic corners")
                return ("Click: the nearest corner of the patch's base moves to this junction "
                        "(Ctrl+click: automatic corners)")
        return cls.__doc__

    def invoke(self, context, event):
        self.reset = event.ctrl or event.alt
        return self.execute(context)

    def execute(self, context):
        obj = bpy.data.objects.get(self.obj_name)
        if obj is None:
            return {'CANCELLED'}
        if self.reset:
            try:
                _edit_params(obj, context, lambda p, ins: p.pop("pt_corners", None) and None)
            except Exception as exc:
                self.report({'ERROR'}, f"Corners: {exc}")
                return {'CANCELLED'}
            self.report({'INFO'}, f"{obj.name}: automatic corners")
            return {'FINISHED'}
        pts = corner_points(obj)
        if not 0 <= self.index < len(pts):
            return {'CANCELLED'}
        hit = pts[self.index]
        if hit["corner"]:
            self.report({'INFO'}, "Already a corner: click another dot to move the nearest corner there "
                                  "(Ctrl+click: automatic corners)")
            return {'CANCELLED'}
        cs = [c for c in pts if c["corner"]]
        if len(cs) != 4:
            return {'CANCELLED'}
        d = [min(abs(c["s"] - hit["s"]), 1.0 - abs(c["s"] - hit["s"])) for c in cs]
        j = int(np.argmin(d))
        new = [[c["k"], c["f"]] for i, c in enumerate(cs) if i != j] + [[hit["k"], hit["f"]]]

        def change(p, ins):
            p["pt_corners"] = new
        try:
            _edit_params(obj, context, change)
        except Exception as exc:
            self.report({'ERROR'}, f"Corners: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"{obj.name}: corner moved (Ctrl+click a dot: automatic corners)")
        return {'FINISHED'}


_corner_cache = {}


def _corners_now(obj):
    """Corners / junctions of a Patch built before they were stored: from its
    loop (the build's sampling and corner choice, no solve). Cached."""
    import json
    h = features.history(obj)
    if h is None or h["op"] != "patch" or h["params"].get("pt_method", "QUAD") != "QUAD":
        return {}
    key = (obj.session_uid, obj.get(features.PROP_HIST), obj.get(features.PROP_KEY))
    if key in _corner_cache:
        return _corner_cache[key]
    try:
        sides, order, _, _, _ = loop_sides(h["inputs"], h["params"])
        fixed = corner_loop_indices(sides, order, h["params"].get("pt_corners") or [])
        idx = corners(sides, fixed=fixed)[0]
        inf = {"corners": loop_points(sides, order, idx), "junctions": loop_points(sides, order, junctions(sides, order)),
               "manual": bool(h["params"].get("pt_corners"))}
    except Exception as exc:
        print("NURBS patch corners:", exc)
        inf = {}
    _corner_cache.clear()
    _corner_cache[key] = inf
    return inf


def corner_points(obj):
    """[{pos, corner, k, f, s}]: the junctions between a Patch's inputs and its
    base's four corners (some may lie inside an input), from the last build."""
    from mathutils import Vector
    inf = patch_info(obj)
    if "corners" not in inf or features.pending(obj):     # (paused: from the settings, not the last build)
        inf = _corners_now(obj)
    if "corners" not in inf:
        return []
    out = []
    for c in inf["corners"]:
        out.append({"pos": Vector(c[2]), "corner": True, "k": c[0], "f": c[1], "s": c[3] if len(c) > 3 else 0.0})
    for jn in inf.get("junctions", []):
        if any((Vector(jn[2]) - o["pos"]).length < 1e-9 + 1e-6 * (o["pos"].length or 1.0) for o in out):
            continue
        out.append({"pos": Vector(jn[2]), "corner": False, "k": jn[0], "f": jn[1], "s": jn[3] if len(jn) > 3 else 0.0})
    return out


class NPROTO_OT_patch_add_guides(PickMixin, bpy.types.Operator):
    """Pick curves inside the opening for the active Patch to pass through (Enter when done)"""
    bl_idname = "nproto.patch_add_guides"
    bl_label = "Add Guide Curves"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 1
    pick_max = 64
    finish_on_min = False
    toggle_picks = True
    extra_help = "   (curves the patch passes through; Enter adds them)"
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    target: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def pick_candidates(self, context):
        return [c for c in super().pick_candidates(context) if c[0].name != self.target]

    def invoke(self, context, event):
        obj = context.view_layer.objects.active
        h = features.history(obj) if obj is not None else None
        if h is None or h["op"] != "patch":
            self.report({'WARNING'}, "Select a Patch with history first")
            return {'CANCELLED'}
        self.target = obj.name
        return self.start_picking(context, event)

    def execute(self, context):
        obj = bpy.data.objects.get(self.target)
        if obj is None:
            return {'CANCELLED'}
        try:
            new = _decode_inputs(self.inputs)

            def change(p, ins):
                have = {(o.name, r) for o, r in ins}
                add = [(o, r) for o, r in new if (o.name, r) not in have]
                n0 = len(ins)
                p["cont"] = (list(p.get("cont", [])) + ["G0"] * n0)[:n0] + ["G0"] * len(add)
                if "pt_tension" in p:
                    p["pt_tension"] = (list(p["pt_tension"]) + [1.0] * n0)[:n0] + [1.0] * len(add)
                p["pt_guides"] = sorted(set(p.get("pt_guides", [])) | set(range(n0, n0 + len(add))))
                return ins + add
            _edit_params(obj, context, change)
        except Exception as exc:
            self.report({'ERROR'}, f"Guides: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"{obj.name}: {len(guide_indices(features.history(obj)['params']))} guide curve(s); "
                              f"held within {patch_info(obj).get('guide_dev', 0.0):.2g}")
        _select_only(context, [obj])
        return {'FINISHED'}


class NPROTO_OT_patch_remove_guide(bpy.types.Operator):
    """Stop using this guide curve (the curve itself stays)"""
    bl_idname = "nproto.patch_remove_guide"
    bl_label = "Remove Guide"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    index: IntProperty(default=-1, options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        obj = context.view_layer.objects.active
        try:
            def change(p, ins):
                if self.index not in p.get("pt_guides", []):
                    raise ValueError("not a guide curve")
                _drop_input(p, self.index)
                return ins[:self.index] + ins[self.index + 1:]
            _edit_params(obj, context, change)
        except Exception as exc:
            self.report({'ERROR'}, f"Guides: {exc}")
            return {'CANCELLED'}
        return {'FINISHED'}


class NPROTO_OT_patch_auto_corners(bpy.types.Operator):
    """Choose the patch's four base corners automatically again"""
    bl_idname = "nproto.patch_auto_corners"
    bl_label = "Automatic Corners"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        obj = context.view_layer.objects.active
        try:
            _edit_params(obj, context, lambda p, ins: p.pop("pt_corners", None) and None)
        except Exception as exc:
            self.report({'ERROR'}, f"Corners: {exc}")
            return {'CANCELLED'}
        return {'FINISHED'}


classes = (NPROTO_OT_patch, NPROTO_OT_patch_corner, NPROTO_OT_patch_add_guides, NPROTO_OT_patch_remove_guide,
           NPROTO_OT_patch_auto_corners)
