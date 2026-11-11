"""Class-A rebuild of kernel edge fillets (runs in the worker, numpy + SMLib).

The kernel builds a fillet along a curved edge as one dense face (degree 3,
adaptive knots that cluster where the edge's curvature changes, a 4-span cubic
approximation of the arc across). This module replaces every such face with
the fewest single-span (Bezier) patches that stay within a tolerance:

* Cross-sections are sampled from the kernel face and turned into exact
  section curves: a rational quadratic arc for Circular (end points on the
  rails, middle CV where the two face tangents meet, weight sin(phi / 2)), a
  line for Chamfer, a Bezier fit with tangent-fixed ends for Blend.
* The section CVs (homogeneous) are fitted along the edge with one span of
  degree 3..max_degree; only if that misses the tolerance are 2, 3 ... spans
  tried, and the result is split into that many Bezier patches.
* At an end that meets another fillet face (e.g. the exact straight fillet of
  the next edge) the patch interpolates that face's end section and matches
  its section derivatives: G1 (first derivative up to a common scale) or G2
  (second derivative up to the reparametrisation terms).
* Each candidate is measured against the kernel face (deviation), the rails
  (gap) and the neighbouring faces (normal angle); the lightest one inside
  the tolerances wins.

The patches replace the dense faces and are stitched back into the body.
"""

import numpy as np


# ---------------------------------------------------------------------------
# B-spline basics
# ---------------------------------------------------------------------------

def clamped_knots(p, spans, inner=None):
    if inner is None:
        inner = list(np.linspace(0.0, 1.0, spans + 1)[1:-1])
    return np.array([0.0] * (p + 1) + list(inner) + [1.0] * (p + 1))


def equidistribute(s, err, U, p, spans):
    """New interior knots from a fit's per-sample error: each current span's
    error is ~ c h^(p+1) |f^(p+1)|, so (e / h^(p+1))^(1/(p+1)) estimates the
    local complexity; the knots then share its integral evenly."""
    br = sorted(set(float(u) for u in U))
    dens = []
    for a, b in zip(br[:-1], br[1:]):
        m = (s >= a) & (s <= b)
        e = float(err[m].max()) if m.any() else 0.0
        dens.append((max(e, 1e-300) / (b - a) ** (p + 1)) ** (1.0 / (p + 1)))
    dens = np.array(dens) + 1e-3 * max(dens)
    cdf = np.r_[0.0, np.cumsum(dens * np.diff(br))]
    cdf /= cdf[-1]
    return list(np.interp(np.linspace(0, 1, spans + 1)[1:-1], cdf, br))


def basis(U, p, ts, der=0):
    """Basis functions (or their der-th derivative) of degree p on knots U at
    parameters ts: array (len(ts), ncv)."""
    U = np.asarray(U, float)
    ts = np.atleast_1d(np.asarray(ts, float))
    m = len(U) - 1
    last = max(i for i in range(m) if U[i] < U[i + 1])     # last non-empty span
    T0 = np.zeros((len(ts), m))
    for k, t in enumerate(ts):
        if t >= U[last + 1]:
            T0[k, last] = 1.0
        else:
            i = np.searchsorted(U, t, side="right") - 1
            T0[k, min(max(i, 0), last)] = 1.0
    tables = [T0]
    for q in range(1, p + 1):
        prev = tables[-1]
        n = m - q
        T = np.zeros((len(ts), n))
        for i in range(n):
            d1 = U[i + q] - U[i]
            d2 = U[i + q + 1] - U[i + 1]
            if d1 > 0:
                T[:, i] += (ts - U[i]) / d1 * prev[:, i]
            if d2 > 0:
                T[:, i] += (U[i + q + 1] - ts) / d2 * prev[:, i + 1]
        tables.append(T)

    def deriv(q, d):            # d-th derivative of the degree-q table
        if d == 0:
            return tables[q]
        lower = deriv(q - 1, d - 1)
        n = m - q
        T = np.zeros((len(ts), n))
        for i in range(n):
            d1 = U[i + q] - U[i]
            d2 = U[i + q + 1] - U[i + 1]
            if d1 > 0:
                T[:, i] += q / d1 * lower[:, i]
            if d2 > 0:
                T[:, i] -= q / d2 * lower[:, i + 1]
        return T

    return deriv(p, der)


def insert_knot(P, U, p, t):
    """Boehm insertion of t once. P: (ncv, dim) homogeneous."""
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


def bezier_pieces(P, U, p):
    """Split a clamped B-spline (homogeneous CVs) into Bezier segments."""
    U = [float(u) for u in U]
    for t in sorted(set(U[p + 1:-p - 1])):
        while U.count(t) < p:
            P, U = insert_knot(P, U, p, t)
    return [P[j * p:j * p + p + 1] for j in range((len(P) - 1) // p)]


def bernstein(n, u):
    from math import comb
    u = np.atleast_1d(np.asarray(u, float))
    return np.stack([comb(n, i) * u ** i * (1 - u) ** (n - i) for i in range(n + 1)], axis=1)


# ---------------------------------------------------------------------------
# Fillet faces: which way is across, sections
# ---------------------------------------------------------------------------

def _uv(surf, sm, p):
    _, uvs, _ = sm.surface_closest_point(surf, tuple(float(x) for x in p))
    uv = uvs[0]
    return (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])


def _edge_mid(e):
    t0, t1 = e.parameter_range()
    return np.array(e.point_at(0.5 * (t0 + t1)))


def _same_face(f, g):
    return f.area() == g.area() and np.allclose(f.bounding_box(), g.bounding_box())


def face_layout(face, sm, is_fillet):
    """Rails / ends of a 4-sided fillet face. is_fillet(face) tells fillet
    faces from the original ones. Returns a dict or a reason string."""
    surf = face.surface()
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    sides = {}
    for e in face.edges():
        uv = _uv(surf, sm, _edge_mid(e))
        d = {"u0": abs(uv[0] - u0) / (u1 - u0), "u1": abs(uv[0] - u1) / (u1 - u0),
             "v0": abs(uv[1] - v0) / (v1 - v0), "v1": abs(uv[1] - v1) / (v1 - v0)}
        key = min(d, key=d.get)
        if d[key] > 1e-3:
            return "an edge does not lie on the fillet surface's boundary (trimmed fillet face)"
        others = [g for g in e.faces() if not _same_face(g, face)]
        if key in sides:
            return "a side of the fillet face is split into several edges"
        sides[key] = (e, others[0] if others else None)
    if len(sides) != 4:
        return f"the fillet face has {len(sides)} sides, not 4 (corner blend)"
    rail_u = all(sides[k][1] is not None and not is_fillet(sides[k][1]) for k in ("u0", "u1"))
    rail_v = all(sides[k][1] is not None and not is_fillet(sides[k][1]) for k in ("v0", "v1"))
    if rail_u == rail_v:
        return "cannot tell the fillet's rails from its ends"
    across = "u" if rail_u else "v"
    a, b = ("u0", "u1") if across == "u" else ("v0", "v1")
    e0, e1 = ("v0", "v1") if across == "u" else ("u0", "u1")
    return {"surf": surf, "across": across, "dom_a": (u0, u1) if across == "u" else (v0, v1),
            "dom_t": (v0, v1) if across == "u" else (u0, u1),
            "rails": (sides[a], sides[b]), "ends": (sides[e0], sides[e1])}


def _eval(lay, a, t):
    """Point and across-derivative at across param a, along param t."""
    s = lay["surf"]
    if lay["across"] == "u":
        p, du, _ = s.derivatives(a, t)
        return np.array(p), np.array(du)
    p, _, dv = s.derivatives(t, a)
    return np.array(p), np.array(dv)


def _vec(q):
    return np.array([q.x, q.y, q.z]) if hasattr(q, "x") else np.array(q, float)


def _foot(surf, sm, p):
    q, uvs, _ = sm.surface_closest_point(surf, tuple(float(x) for x in p))
    uv = uvs[0]
    uv = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
    return _vec(q), np.array(surf.normal(*uv)), uv


def ball(lay, sm, t, radius, A, B, tA):
    """Exact rolling-ball contact points and centre for the section at t:
    the ball of the given radius touching both rail faces, centre in the
    plane through the kernel section normal to the fillet's direction."""
    s = lay["surf"]
    am = 0.5 * sum(lay["dom_a"])
    if lay["across"] == "u":
        M, _, T = s.derivatives(am, t)
    else:
        M, T, _ = s.derivatives(t, am)
    M, T = np.array(M), np.array(T)
    T /= np.linalg.norm(T) or 1.0
    n = (B - A) - ((B - A) @ tA) / (tA @ tA) * tA
    C = A + radius * n / (np.linalg.norm(n) or 1.0)
    S1, S2 = lay["rails"][0][1].surface(), lay["rails"][1][1].surface()
    for _ in range(12):
        P1, n1, _ = _foot(S1, sm, C)
        P2, n2, _ = _foot(S2, sm, C)
        n1 = n1 if n1 @ (C - P1) >= 0 else -n1
        n2 = n2 if n2 @ (C - P2) >= 0 else -n2
        Cn = np.linalg.solve(np.array([n1, n2, T]), np.array([radius + n1 @ P1, radius + n2 @ P2, T @ M]))
        done = np.linalg.norm(Cn - C) < 1e-13 * (1.0 + np.linalg.norm(C))
        C = Cn
        if done:
            break
    P1, _, _ = _foot(S1, sm, C)
    P2, _, _ = _foot(S2, sm, C)
    if abs(np.linalg.norm(P1 - C) - radius) > 1e-6 * radius or abs(np.linalg.norm(P2 - C) - radius) > 1e-6 * radius:
        return None
    return P1, P2, C


def _normal_curvature(surf, sm, A, t, n, L):
    """Normal curvature of a face at A in direction t, positive when it bends
    towards n (from the height of a nearby face point above the tangent)."""
    eps = 0.05 * L
    Q, _, _ = _foot(surf, sm, A + eps * t)
    d = Q - A
    return 2.0 * float(d @ n) / float(d @ d)


def blend_section(lay, sm, t, radius, A, B, tA, m, k):
    """Exact Blend section: degree-m Bezier between the rolling-ball contact
    points, the first c + 1 CVs (G_c, c = (m - 1) // 2) along each face's
    tangent at spacing k x tangent length (the kernel's own fullness, see
    blend_fullness), with the face's normal curvature on CV 2 for G2."""
    exact = ball(lay, sm, t, radius, A, B, tA)
    if exact is None:
        return None
    A, B, C = exact
    ua, ub = A - C, B - C
    th = np.arccos(np.clip(ua @ ub / (np.linalg.norm(ua) * np.linalg.norm(ub)), -1.0, 1.0))
    bis = ua + ub
    P1 = C + radius / np.cos(0.5 * th) * bis / np.linalg.norm(bis)
    c = (m - 1) // 2
    sides = []
    for X, face in ((A, lay["rails"][0][1]), (B, lay["rails"][1][1])):
        L = float(np.linalg.norm(P1 - X))
        tX = (P1 - X) / L
        nX = (C - X) / radius
        al = k * L
        cv = [X + j * al * tX for j in range(c + 1)]
        if c >= 2:
            kap = _normal_curvature(face.surface(), sm, X, tX, nX, L)
            cv[2] = cv[2] + kap * m * al * al / (m - 1) * nX
        sides.append(cv)
    cvs = sides[0] + sides[1][::-1]
    return np.array([np.r_[q, 1.0] for q in cvs])


def blend_fullness(lay, sm, m, n=9):
    """The kernel Blend's tangent spacing: |CV1 - contact| / tangent length,
    read from its own sections (both ends of each)."""
    t0, t1 = lay["dom_t"]
    out = []
    for t in np.linspace(t0, t1, n + 2)[1:-1]:
        Q = section(lay, t, "BLEND", m)
        X = Q[:, :3]
        a0, a1 = lay["dom_a"]
        A, tA = _eval(lay, a0, t)
        B, tB = _eval(lay, a1, t)
        d = B - A
        tB = -tB
        M = np.array([[tA @ tA, -tA @ tB], [tA @ tB, -tB @ tB]])
        x, y = np.linalg.lstsq(M, np.array([d @ tA, d @ tB]), rcond=None)[0]
        P1 = 0.5 * ((A + x * tA) + (B + y * tB))
        for e0, e1 in ((X[0], X[1]), (X[-1], X[-2])):
            L = np.linalg.norm(P1 - e0)
            if L > 0:
                out.append(float(np.linalg.norm(e1 - e0) / L))
    return out


def section(lay, t, kind, m=3, sm=None, radius=None, blend_k=None):
    """Homogeneous section CVs (rows x 4) at along-param t."""
    a0, a1 = lay["dom_a"]
    A, tA = _eval(lay, a0, t)
    B, tB = _eval(lay, a1, t)
    tB = -tB
    if kind == "BLEND" and radius and sm is not None and blend_k:
        Q = blend_section(lay, sm, t, radius, A, B, tA, m, blend_k)
        if Q is not None:
            return Q
    if kind == "CHAMFER":
        return np.array([np.r_[A, 1.0], np.r_[B, 1.0]])
    if kind == "CIRCULAR":
        exact = ball(lay, sm, t, radius, A, B, tA) if (radius and sm is not None) else None
        if exact is not None:
            A, B, C = exact
            ua, ub = A - C, B - C
            th = np.arccos(np.clip(ua @ ub / (np.linalg.norm(ua) * np.linalg.norm(ub)), -1.0, 1.0))
            w = float(np.cos(0.5 * th))
            bis = ua + ub
            P1 = C + radius / w * bis / np.linalg.norm(bis)
            return np.array([np.r_[A, 1.0], np.r_[w * P1, w], np.r_[B, 1.0]])
        # where the two tangent lines meet (closest point between them)
        d = B - A
        M = np.array([[tA @ tA, -tA @ tB], [tA @ tB, -tB @ tB]])
        x, y = np.linalg.lstsq(M, np.array([d @ tA, d @ tB]), rcond=None)[0]
        P1 = 0.5 * ((A + x * tA) + (B + y * tB))
        ua, ub = A - P1, B - P1
        c = ua @ ub / (np.linalg.norm(ua) * np.linalg.norm(ub) + 1e-300)
        phi = np.arccos(np.clip(c, -1.0, 1.0))
        w = float(np.sin(0.5 * phi))
        return np.array([np.r_[A, 1.0], np.r_[w * P1, w], np.r_[B, 1.0]])
    # BLEND: Bezier of degree m through the iso curve, end tangents kept
    us = np.linspace(a0, a1, 4 * m + 9)
    pts = np.array([_eval(lay, u, t)[0] for u in us])
    # the kernel's own across parameter (smooth along the edge): fitting every
    # section by its chord length instead makes the CVs jitter from section to section
    s = (us - a0) / (a1 - a0)
    Bm = bernstein(m, s)
    na, nb = tA / (np.linalg.norm(tA) or 1.0), tB / (np.linalg.norm(tB) or 1.0)
    # unknowns: alpha, beta, inner CVs 2..m-2 (3 each)
    ninner = max(m - 3, 0)
    rows, rhs = [], []
    for k in range(len(s)):
        base = Bm[k, 0] * A + Bm[k, m] * B + Bm[k, 1] * A + Bm[k, m - 1] * B
        for c in range(3):
            r = np.zeros(2 + 3 * ninner)
            r[0] = Bm[k, 1] * na[c]
            r[1] = Bm[k, m - 1] * nb[c]
            for j in range(ninner):
                r[2 + 3 * j + c] = Bm[k, 2 + j]
            rows.append(r)
            rhs.append(pts[k, c] - base[c])
    z = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    C = [A, A + z[0] * na] + [z[2 + 3 * j:5 + 3 * j] for j in range(ninner)] + [B + z[1] * nb, B]
    return np.array([np.r_[c, 1.0] for c in C])


def _orient_rows(Q, A_ref):
    """Reverse the rows if the section starts at the other rail."""
    if np.linalg.norm(Q[-1, :3] / Q[-1, 3] - A_ref) < np.linalg.norm(Q[0, :3] / Q[0, 3] - A_ref):
        return Q[::-1].copy()
    return Q


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _end_data(nlay, shared_edge, kind, m, A_ref, inward_dir, sm, radius, blend_k=None):
    """Section, first and second derivative of a neighbouring fillet face at
    the end it shares with ours, oriented along inward_dir (our first rail
    moving away from the shared end)."""
    t0, t1 = nlay["dom_t"]
    mid = _edge_mid(shared_edge)
    am = 0.5 * sum(nlay["dom_a"])
    q0 = _eval(nlay, am, t0)[0]
    q1 = _eval(nlay, am, t1)[0]
    te, sgn = (t0, 1.0) if np.linalg.norm(q0 - mid) < np.linalg.norm(q1 - mid) else (t1, -1.0)
    h = 1e-3 * (t1 - t0)
    Qs = [_orient_rows(section(nlay, te + sgn * k * h, kind, m, sm, radius, blend_k), A_ref) for k in range(3)]
    D = (-3 * Qs[0] + 4 * Qs[1] - Qs[2]) / (2 * h)     # d/d(tau), tau running into the neighbour
    D2 = (Qs[0] - 2 * Qs[1] + Qs[2]) / h ** 2
    if D[0, :3] @ inward_dir < 0:
        D = -D
    return Qs[0], D, D2


def _project_params(s, Q, C, U, p, iters=3):
    """Move every sample to the parameter of its closest curve point (in the
    homogeneous section space): parameter correction."""
    s = s.copy()
    for _ in range(iters):
        P = basis(U, p, s) @ C
        d1 = basis(U, p, s, 1) @ C
        d2 = basis(U, p, s, 2) @ C
        e = P - Q
        num = (e * d1).sum(1)
        den = (d1 * d1).sum(1) + (e * d2).sum(1)
        den = np.where(den > 1e-300, den, (d1 * d1).sum(1) + 1e-300)
        s = np.clip(s - num / den, 0.0, 1.0)
    s[0], s[-1] = 0.0, 1.0
    return np.maximum.accumulate(s)


def _mirror(x):
    """Make increasing parameters / knots symmetric about 0.5."""
    x = np.asarray(x, float)
    return 0.5 * (x + 1.0 - x[::-1])


def fit_rows(s, Q, p, spans, ends, fair=1e-7, iters=4, correct=0, inner=None, sym=False):
    """Fit homogeneous section rows Q (N, R) along s in [0, 1].
    ends: [end0, end1], each None or dict(Q, D, D2, cont) with cont 0/1/2.
    sym: the sections are mirror-symmetric (sample k <-> N-1-k): parameters
    and knots are kept symmetric too, so the patch layout is.
    Returns (CVs (ncv, R), knots, corrected s)."""
    if sym:
        s = _mirror(s)
        if inner is not None:
            inner = list(_mirror(inner))
    U = clamped_knots(p, spans, inner)
    ncv = p + spans
    for rnd in range(correct + 1):
        C = _fit_once(s, Q, p, U, ncv, ends, fair, iters, Q.shape[1] // 4)
        if rnd < correct:
            s = _project_params(s, Q, C, U, p)
            if sym:
                s = _mirror(s)
    return C, U, s


def is_symmetric(secs, ends, rel=1e-4):
    """Mirror symmetry of the sampled fillet (sections at uniform arc length):
    the distances within and between neighbouring sections read the same
    from both ends (rigid invariants, so any mirror plane works)."""
    if (ends[0] is None) != (ends[1] is None) or (ends[0] and ends[0]["cont"] != ends[1]["cont"]):
        return False
    X = secs[..., :3] / secs[..., 3:4]
    N = len(X)
    size = float(np.linalg.norm(np.ptp(X.reshape(-1, 3), axis=0))) or 1.0
    def sig(k, k2):
        return np.linalg.norm(X[k][:, None, :] - X[k2][None, :, :], axis=2)
    worst = 0.0
    for k in range(N - 1):
        worst = max(worst, float(np.abs(sig(k, k + 1) - sig(N - 1 - k, N - 2 - k)).max()))
        worst = max(worst, float(np.abs(sig(k, k) - sig(N - 1 - k, N - 1 - k)).max()))
    return worst < rel * size


def _fit_once(s, Q, p, U, ncv, ends, fair, iters, nrows):
    """One constrained least-squares solve. At a joined end every section row
    i gets C1_i = C0_i + lam_i D_i (G1) and the second derivative
    lam_i^2 D2_i + mu_i D_i (G2): each row may slide along the neighbour's own
    direction at its own rate (for the extruded / straight fillets this is
    tangential, so G1 / G2 hold exactly; rails moving at different speeds,
    as on any curved fillet, are then possible)."""
    N, R = Q.shape
    Bk = basis(U, p, s)
    d1 = [basis(U, p, [0.0], 1)[0], basis(U, p, [1.0], 1)[0]]
    d2 = [basis(U, p, [0.0], 2)[0], basis(U, p, [1.0], 2)[0]]
    lam_fix = np.zeros((2, nrows))
    C = None
    for it in range(iters):
        fixed, scal = set(), []
        for side, e in enumerate(ends):
            if e is None:
                continue
            idx = [0, 1, 2] if side == 0 else [ncv - 1, ncv - 2, ncv - 3]
            fixed.add(idx[0])
            for kind, need in (("lam", 1), ("mu", 2)):
                if e["cont"] >= need:
                    fixed.add(idx[need])
                    scal += [(side, kind, r) for r in range(nrows)]
        free = [j for j in range(ncv) if j not in fixed]
        nz = len(free) * R + len(scal)
        G = np.zeros((ncv, R, nz))
        h = np.zeros((ncv, R))
        for fi, j in enumerate(free):
            G[j, np.arange(R), fi * R + np.arange(R)] = 1.0
        si = {k: len(free) * R + i for i, k in enumerate(scal)}
        for side, e in enumerate(ends):
            if e is None:
                continue
            idx = [0, 1, 2] if side == 0 else [ncv - 1, ncv - 2, ncv - 3]
            h[idx[0]] = e["Q"]
            if e["cont"] >= 1:
                a0, a1 = d1[side][idx[0]], d1[side][idx[1]]
            if e["cont"] >= 2:
                b0, b1, b2 = d2[side][idx[0]], d2[side][idx[1]], d2[side][idx[2]]
            for r in range(nrows):
                sl = slice(4 * r, 4 * r + 4)
                if e["cont"] >= 1:
                    # a0 C0 + a1 C1 = lam D
                    G[idx[1], sl, si[(side, "lam", r)]] = e["D"][sl] / a1
                    h[idx[1], sl] = -a0 * e["Q"][sl] / a1
                if e["cont"] >= 2:
                    # b0 C0 + b1 C1 + b2 C2 = lam^2 D2 + mu D
                    G[idx[2], sl, si[(side, "mu", r)]] = e["D"][sl] / b2
                    G[idx[2], sl, si[(side, "lam", r)]] = -b1 * G[idx[1], sl, si[(side, "lam", r)]] / b2
                    h[idx[2], sl] = (lam_fix[side, r] ** 2 * e["D2"][sl] - b0 * e["Q"][sl]
                                     - b1 * h[idx[1], sl]) / b2
        A = np.einsum("kj,jrz->krz", Bk, G).reshape(N * R, nz)
        b = (Q - Bk @ h).reshape(N * R)
        rows, rhs = [A], [b]
        if fair > 0 and ncv > 2:
            w = np.sqrt(fair * N)
            for j in range(1, ncv - 1):
                rows.append(w * (G[j - 1] - 2 * G[j] + G[j + 1]))
                rhs.append(-w * (h[j - 1] - 2 * h[j] + h[j + 1]))
        z = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        C = np.einsum("jrz,z->jr", G, z) + h
        new = np.array([[z[si[(sd, "lam", r)]] if (sd, "lam", r) in si else 0.0 for r in range(nrows)]
                        for sd in (0, 1)])
        if not any(e is not None and e["cont"] >= 2 for e in ends):
            break
        if np.allclose(new, lam_fix, rtol=1e-9, atol=1e-15):
            break
        lam_fix = new
    return C


def eval_rows(C, U, p, nrows, vs):
    return (basis(U, p, vs) @ C).reshape(len(vs), nrows, 4)


def eval_sections(rows, us):
    """rows (nv, nrows, 4) homogeneous sections -> points (nu, nv, 3)."""
    H = np.einsum("ui,vic->uvc", bernstein(rows.shape[1] - 1, us), rows)
    return H[..., :3] / H[..., 3:4]


# ---------------------------------------------------------------------------
# One face
# ---------------------------------------------------------------------------

def _sample_along(lay, kind, m, sm, radius, n=121, blend_k=None):
    """Sections at uniform arc length of the fillet's middle curve."""
    t0, t1 = lay["dom_t"]
    am = 0.5 * sum(lay["dom_a"])
    tt = np.linspace(t0, t1, 401)
    mid = np.array([_eval(lay, am, t)[0] for t in tt])
    ch = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(mid, axis=0), axis=1))]
    ts = np.interp(np.linspace(0.0, 1.0, n) * ch[-1], ch, tt)
    Q0 = section(lay, ts[0], kind, m, sm, radius, blend_k)
    secs = np.array([_orient_rows(section(lay, t, kind, m, sm, radius, blend_k), Q0[0, :3]) for t in ts])
    if radius:
        # space the sections evenly along the exact ones (the kernel face's
        # middle line is only approximate): a symmetric fillet then samples
        # symmetrically
        for _ in range(2):
            X = (secs[..., :3] / secs[..., 3:4]).mean(axis=1)
            ce = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
            ts = np.interp(np.linspace(0.0, 1.0, n) * ce[-1], ce, ts)
            secs = np.array([_orient_rows(section(lay, t, kind, m, sm, radius, blend_k), Q0[0, :3]) for t in ts])
    return secs, ch[-1]


def _param_candidates(secs, lay, sm):
    """Initial parameters for the sections: chord length of the middle row,
    and every rail-face surface coordinate that runs monotonically along the
    fillet (a face's own parametrisation is often the most polynomial one)."""
    X = secs[:, secs.shape[1] // 2, :3] / secs[:, secs.shape[1] // 2, 3:4]
    ch = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    out = [ch / ch[-1]]
    for row, (edge, face) in ((0, lay["rails"][0]), (secs.shape[1] - 1, lay["rails"][1])):
        surf = face.surface()
        uv = np.array([_foot(surf, sm, secs[k, row, :3] / secs[k, row, 3])[2] for k in range(len(secs))])
        for c in range(2):
            x = uv[:, c]
            if abs(x[-1] - x[0]) < 1e-12:
                continue
            x = (x - x[0]) / (x[-1] - x[0])
            if np.all(np.diff(x) > 0):
                out.append(x)
    return out


def _patch_end(built, A_end):
    """Section, d/dv pointing away from an already rebuilt neighbour, and
    d2/dv2 at the end of its patch that touches the point A_end."""
    C, U, p, nrows = built
    best = None
    for v, sgn in ((0.0, -1.0), (1.0, 1.0)):
        Q = eval_rows(C, U, p, nrows, [v])[0]
        X = Q[:, :3] / Q[:, 3:4]
        for rev in (False, True):
            d = np.linalg.norm((X[-1] if rev else X[0]) - A_end)
            if best is None or d < best[0]:
                best = (d, v, sgn, rev)
    _, v, sgn, rev = best
    Q = eval_rows(C, U, p, nrows, [v])[0]
    D = (basis(U, p, [v], 1) @ C).reshape(nrows, 4) * sgn
    D2 = (basis(U, p, [v], 2) @ C).reshape(nrows, 4)
    if rev:
        Q, D, D2 = Q[::-1], D[::-1], D2[::-1]
    return Q, D, D2, best[0]


def rebuild_face(face, sm, is_fillet, kind="CIRCULAR", cont=2, tol=1e-4, ang_tol=0.1,
                 max_degree=7, max_spans=6, layout_of=None, radius=None, blend_k=None,
                 built_of=None):
    """Lightest Bezier patch set for one dense fillet face, or a reason string.
    Returns dict(pieces=[(along x rows x 4 homogeneous)], degree_u, degree_v, report)."""
    lay = layout_of(face) if layout_of else face_layout(face, sm, is_fillet)
    if isinstance(lay, str):
        return lay
    m = 3
    if kind == "BLEND":
        d = sm.blender_ext.surface_cvs(lay["surf"])
        m = int(np.clip(d["degree_u"] if lay["across"] == "u" else d["degree_v"], 3, 7))
    secs, length = _sample_along(lay, kind, m, sm, radius, blend_k=blend_k)
    N, nrows = secs.shape[0], secs.shape[1]
    Q = secs.reshape(N, nrows * 4)
    # ends: a neighbour already rebuilt -> G1 / G2 to its patch (exact
    # derivatives); a neighbour still to be rebuilt -> only its section (it
    # will take the G1 / G2 to this patch); a kept fillet face (e.g. the exact
    # straight Circular fillet) -> G1 / G2 to its own sections
    ends, end_info = [], []
    for side, (edge, other) in enumerate(lay["ends"]):
        k = 0 if side == 0 else N - 1
        kin = 3 if side == 0 else N - 4
        A_end = secs[k, 0, :3]
        if other is None or not is_fillet(other):
            ends.append(None)
            end_info.append("free")
            continue
        built = built_of(other) if built_of else None
        if built == "pending":
            ends.append({"Q": secs[k].reshape(-1), "D": None, "D2": None, "cont": 0})
            end_info.append(f"G{cont}")
            continue
        if built is not None:
            Qe, D, D2, dist = _patch_end(built, A_end)
            if side == 1:
                D = -D
        else:
            nl = layout_of(other) if layout_of else face_layout(other, sm, is_fillet)
            if isinstance(nl, str):
                ends.append(None)
                end_info.append("free")
                continue
            Qe, D, D2 = _end_data(nl, edge, kind, m, A_end, secs[kin, 0, :3] - A_end, sm, radius, blend_k)
            if side == 1:
                D = -D      # at v = 1 the derivative points out of the patch, into the neighbour
            dist = np.linalg.norm(Qe[0, :3] / Qe[0, 3] - A_end)
        if dist > 0.02 * length:
            ends.append(None)
            end_info.append("free")
            continue
        ends.append({"Q": Qe.reshape(-1), "D": D.reshape(-1), "D2": D2.reshape(-1), "cont": cont})
        end_info.append(f"G{cont}")
    rail_faces = [lay["rails"][0][1], lay["rails"][1][1]]
    rail_edges = [lay["rails"][0][0], lay["rails"][1][0]]

    sym = is_symmetric(secs, ends)
    exact = bool(radius and (kind == "CIRCULAR" or (kind == "BLEND" and blend_k)))

    # starting parameters: the candidate a plain single-span fit likes best
    p0 = min(7, max_degree)
    cands = _param_candidates(secs, lay, sm)
    scores = []
    for s0 in cands:
        C, U, s1 = fit_rows(s0, Q, p0, 1, ends, sym=sym)
        scores.append(np.abs(basis(U, p0, s1) @ C - Q).max())
    s_init = cands[int(np.argmin(scores))]

    need = sum(0 if e is None else 1 + e["cont"] for e in ends)
    max_c = max([0] + [e["cont"] for e in ends if e is not None])
    best = None
    for spans in range(1, max_spans + 1):
        for p in range(max(1, max_c), max_degree + 1):
            if p + spans < max(need + 1, 2):
                continue
            # even patches first (uniform knots); move the knots to where the
            # error is only if even ones miss the tolerance
            C, U, s = fit_rows(s_init, Q, p, spans, ends, correct=6, sym=sym)
            rep = measure(C, U, p, nrows, secs, s, lay, rail_edges, rail_faces, sm, exact)
            ok = rep["dev"] <= tol and rep["angle"] <= ang_tol and rep["min_w"] > 0
            if spans > 1 and not ok:
                e = _sample_err(C, U, p, nrows, secs, s)
                Uc, sc, ec, e0 = U, s, e, e.max()
                for _ in range(5):
                    C2, U2, s2 = fit_rows(sc, Q, p, spans, ends, correct=3, sym=sym,
                                          inner=equidistribute(sc, ec, Uc, p, spans))
                    e2 = _sample_err(C2, U2, p, nrows, secs, s2)
                    Uc, sc, ec = U2, s2, e2
                    if e2.max() < e0:
                        rep2 = measure(C2, U2, p, nrows, secs, s2, lay, rail_edges, rail_faces, sm, exact)
                        ok2 = rep2["dev"] <= tol and rep2["angle"] <= ang_tol and rep2["min_w"] > 0
                        if ok2 or rep2["dev"] < rep["dev"]:
                            C, U, s, rep, ok, e0 = C2, U2, s2, rep2, ok2, e2.max()
                        if ok:
                            break
            rep.update(spans=spans, degree=p, symmetric=sym)
            score = (not ok, spans * p if ok else rep["dev"])
            if best is None or score < best[0]:
                best = (score, C, U, p, spans, rep)
            if ok:
                break
        if best is not None and not best[0][0]:
            break
    _, C, U, p, spans, rep = best
    pieces = bezier_pieces(C, U, p)
    rep["ends"] = end_info
    rep["rows"] = nrows
    rep["within"] = not best[0][0]
    rep["kernel_cvs"] = list(_kernel_counts(lay, sm))
    rep["exact_sections"] = exact
    return {"pieces": [P.reshape(p + 1, nrows, 4) for P in pieces], "degree_u": nrows - 1,
            "degree_v": p, "report": rep, "spline": (C, U, p, nrows)}


def _kernel_counts(lay, sm):
    d = sm.blender_ext.surface_cvs(lay["surf"])
    a, b = d["count_u"], d["count_v"]
    return (a, b) if lay["across"] == "u" else (b, a)


def patch_derivs(C, U, p, nrows, u, vs):
    """Point, d/du and d/dv of the rational patch at across-param u, along vs."""
    rows = eval_rows(C, U, p, nrows, vs)                           # (nv, nrows, 4)
    rows_v = (basis(U, p, vs, 1) @ C).reshape(len(vs), nrows, 4)
    n = nrows - 1
    bu = bernstein(n, [u])[0]
    bdu = n * (np.r_[0.0, bernstein(n - 1, [u])[0]] - np.r_[bernstein(n - 1, [u])[0], 0.0]) if n > 0 else np.zeros(1)
    H, Hu, Hv = rows.transpose(0, 2, 1) @ bu, rows.transpose(0, 2, 1) @ bdu, rows_v.transpose(0, 2, 1) @ bu
    P = H[:, :3] / H[:, 3:]
    Su = (Hu[:, :3] - P * Hu[:, 3:]) / H[:, 3:]
    Sv = (Hv[:, :3] - P * Hv[:, 3:]) / H[:, 3:]
    return P, Su, Sv


def _sample_err(C, U, p, nrows, secs, s, nu=5):
    us = np.linspace(0, 1, nu)
    X = eval_sections(eval_rows(C, U, p, nrows, s), us)
    return np.linalg.norm(X - eval_sections(secs, us), axis=2).max(0)


def measure(C, U, p, nrows, secs, s, lay, rail_edges, rail_faces, sm, exact=True, nu=9):
    """dev: largest distance between the patch and the sampled sections
    (exact rolling-ball arcs for Circular), at the samples' parameters.
    edge_gap: patch boundary to the kernel's rail edges (what Stitch closes).
    angle: patch normal vs the rail face's normal along both rails."""
    us = np.linspace(0, 1, nu)
    X = eval_sections(eval_rows(C, U, p, nrows, s), us)
    Y = eval_sections(secs, us)
    dev = float(np.linalg.norm(X - Y, axis=2).max())
    if not exact:
        # sections taken from the kernel face: the distance to that face (the
        # parametric comparison above also counts where along the section a point sits)
        surf = lay["surf"]
        G = eval_sections(eval_rows(C, U, p, nrows, np.linspace(0, 1, 41)), us)
        dev = max(sm.surface_closest_point(surf, tuple(float(c) for c in q))[2] for q in G.reshape(-1, 3))
    vs = np.linspace(0, 1, 41)
    gap, ang = 0.0, 0.0
    for edge, face, u in zip(rail_edges, rail_faces, (0.0, 1.0)):
        fs = face.surface()
        P, Su, Sv = patch_derivs(C, U, p, nrows, u, vs)
        for j in range(len(vs)):
            gap = max(gap, edge.closest_point(tuple(P[j]))[2])
            n1 = np.cross(Su[j], Sv[j])
            _, n2, _ = _foot(fs, sm, P[j])
            c = abs(n1 @ n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-300)
            ang = max(ang, float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    w = eval_rows(C, U, p, nrows, vs)[..., 3]
    return {"dev": dev, "edge_gap": float(gap), "angle": ang, "min_w": float(w.min())}


# ---------------------------------------------------------------------------
# Whole body
# ---------------------------------------------------------------------------

def is_light(surf, ext):
    d = ext.surface_cvs(surf)
    return d["count_u"] == d["degree_u"] + 1 and d["count_v"] == d["degree_v"] + 1


def patch_surface(P, ext):
    """Bezier piece (along x rows x 4, homogeneous) -> kernel surface with
    u = across (rows), v = along."""
    nv, nr = P.shape[0], P.shape[1]
    pts, wts = [], []
    for i in range(nr):
        for j in range(nv):
            h = P[j, i]
            pts.append(tuple(float(x) for x in h[:3] / h[3]))
            wts.append(float(h[3]))
    rational = any(abs(w - 1.0) > 1e-12 for w in wts)
    return ext.surface_from_cvs(nr - 1, nv - 1, pts, [0.0] * nr + [1.0] * nr, [0.0] * nv + [1.0] * nv,
                                wts if rational else None)


def rebuild_body(b, sm, fillet_ids, kind="CIRCULAR", cont=2, tol=1e-4, ang_tol=0.1,
                 max_degree=7, max_spans=6, radius=None):
    """Replace the dense fillet faces (indices fillet_ids into b.faces()) of
    brep b by Bezier patches. Returns (new brep, report)."""
    ext = sm.blender_ext
    faces = b.faces()
    fil = [faces[i] for i in fillet_ids]

    def is_fillet(f):
        return any(_same_face(f, g) for g in fil)

    cache = {}

    def layout_of(f):
        k = next(i for i, g in enumerate(fil) if _same_face(f, g))
        if k not in cache:
            cache[k] = face_layout(f, sm, is_fillet)
        return cache[k]

    lo, hi = (np.array(v) for v in b.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    blend_k = None
    if kind == "BLEND" and radius:
        ratios = []
        for f in fil:
            lay = layout_of(f)
            if not isinstance(lay, str):
                d = ext.surface_cvs(lay["surf"])
                m = d["degree_u"] if lay["across"] == "u" else d["degree_v"]
                if 3 <= m <= 7 and m % 2 == 1:
                    ratios += blend_fullness(lay, sm, m)
        blend_k = float(np.median(ratios)) if ratios else None
    todo = [i for i, f in enumerate(fil) if not (is_light(f.surface(), ext) and kind != "BLEND")]
    # simplest faces first (e.g. straight runs before the bend): later faces
    # take their G1 / G2 from the patches already built
    todo.sort(key=lambda i: (int(np.prod(_kernel_counts(layout_of(fil[i]), sm)))
                             if not isinstance(layout_of(fil[i]), str) else 0, i))
    built = {}

    def built_of(g):
        k = next(i for i, h in enumerate(fil) if _same_face(g, h))
        if k in built:
            return built[k]
        return "pending" if k in todo else None

    results, report = [], []
    for i in todo:
        f = fil[i]
        res = rebuild_face(f, sm, is_fillet, kind, cont, tol, ang_tol, max_degree, max_spans, layout_of, radius,
                           blend_k, built_of)
        if isinstance(res, str):
            report.append({"skipped": res})
            built[i] = None
            continue
        built[i] = res["spline"]
        results.append((f, res))
        report.append(res["report"])
    if not results:
        return b, report
    gap = max(r["report"]["edge_gap"] for _, r in results)
    before_lamina = sum(1 for e in b.edges() if len(e.faces()) == 1)
    ext.delete_faces(b, [f for f, _ in results])
    patches = [ext.brep_from_surface(patch_surface(P, ext)) for _, r in results for P in r["pieces"]]
    if hasattr(ext, "append_brep"):       # plain copy: no imprint where a patch touches a face
        merged = b
        for q in patches:
            ext.append_brep(merged, q)
    else:
        merged = sm.merge_breps([b] + patches, operation=sm.BooleanOp.MERGE)
    st_tol = max(2.5 * gap, 1e-6 * size)
    res = sm.stitch_advanced(merged, st_tol, squeeze_small_edges=False, making_manifold_solid=False)
    out = res[0]
    lamina = sum(1 for e in out.edges() if len(e.faces()) == 1)
    try:
        start = max(out.faces(), key=lambda f: f.area())
        sm.unify_normals(out, start, 100)
    except Exception:
        pass
    return out, {"faces": report, "lamina_before": before_lamina, "lamina_after": lamina,
                 "stitch_tol": st_tol, "patches": len(patches)}
