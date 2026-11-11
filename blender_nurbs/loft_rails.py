"""Loft with rails (Fusion / SolidWorks style), on Square's least-squares solver.

Profiles P0 .. Pn (2 or more) run along u, rails (0, 1 or 2) along v:

* 2 rails  -> the four sides of a Square: first profile, rail B, last profile,
              rail A. Inner profiles become exact position conditions inside the
              net. End profiles and rails that are surface edges take G0/G1/G2.
* 1 rail   -> a guide curve: the net passes through it (position only). The
              sides between the profiles' ends are free.
* 0 rails  -> the profiles alone, with G0/G1/G2 at the end profiles; the sides
              are free.

Rails are re-parameterised (piecewise linearly) so every profile meets both
rails at the same v. A free side has no condition: its curve through the
profiles' ends is only the starting guess of the net.
"""

import numpy as np

from . import features, geom, square
from .square import CONT, Side, sample_side

LAST_INFO = {}
TOUCH_REL = 1e-3          # profile / rail contact tolerance (of the size)


def _touch_dist(a, b):
    """Smallest distance between two sampled polylines (a's points to b's segments)."""
    d = np.linalg.norm(a.P[:, None, :] - b.P[None, :, :], axis=2)
    i = int(np.argmin(d.min(axis=1)))
    lo, hi = max(0, i - 2), min(len(a.P), i + 3)
    return min(_nearest_t(b, x)[1] for x in a.P[lo:hi])


def classify(sides, tol):
    """Profiles first: input 0 is a profile; rails are the inputs that touch it
    and every other profile. Returns (profile indices, rail indices)."""
    n = len(sides)
    touch = [[i != j and _touch_dist(sides[i], sides[j]) <= tol for j in range(n)] for i in range(n)]
    rails = [j for j in range(1, n) if touch[0][j]]
    profiles = [i for i in range(n) if i not in rails]
    rails = [r for r in rails if all(touch[r][p_] for p_ in profiles)]
    profiles = [i for i in range(n) if i not in rails]
    if len(rails) > 2:
        raise RuntimeError("more than two inputs touch the first profile: pick the profiles first, then at most two rails")
    return profiles, rails


def _nearest_t(side, x):
    """Parameter (side.t) of the point on the sampled side nearest to x (linear in the segment)."""
    P, t = side.P, side.t
    best = (np.inf, 0.0)
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        ab = b - a
        L = float(ab @ ab)
        s = 0.0 if L < 1e-30 else min(1.0, max(0.0, float((x - a) @ ab) / L))
        d = float(np.linalg.norm(a + s * ab - x))
        if d < best[0]:
            best = (d, t[i] + s * (t[i + 1] - t[i]))
    return best[1], best[0]


def _at(side, t):
    return np.array([np.interp(t, side.t, side.P[:, c]) for c in range(3)])


def _contact(pr, rl):
    """Where a profile and a rail touch: (profile parameter, rail parameter, distance)."""
    best = min(((_nearest_t(rl, x), i) for i, x in enumerate(pr.P)), key=lambda h: h[0][1])
    (trl, d), i = best
    tpr = pr.t[i]
    for _ in range(3):                       # alternate projections to the contact
        tpr, _ = _nearest_t(pr, _at(rl, trl))
        trl, d = _nearest_t(rl, _at(pr, tpr))
    return float(tpr), float(trl), float(d)


def _interp_side(points, v, m):
    """A free side: interpolating curve through `points` at parameters v."""
    pts = np.asarray(points, float)
    t = np.linspace(0.0, 1.0, m)
    if len(pts) == 2:
        P = (1 - t)[:, None] * pts[0] + t[:, None] * pts[1]
    else:
        deg = min(3, len(pts) - 1)
        U = [0.0] * (deg + 1)
        for j in range(1, len(pts) - deg):
            U.append(sum(v[j:j + deg]) / deg)
        U += [1.0] * (deg + 1)
        A = geom.interp_matrix(np.asarray(v), U, deg, len(pts))
        X = np.linalg.solve(A, pts)
        P = geom.interp_matrix(t, U, deg, len(pts)) @ X
    s = Side(P, t=t)
    s.free = True
    return s


def _remap(side, hits, v):
    """Re-parameterise a rail so that its parameter `hits[k]` becomes v[k]."""
    order = np.argsort(hits)
    hx, vx = np.asarray(hits)[order], np.asarray(v)[order]
    side.t = np.interp(side.t, hx, vx)
    if side.knots is not None:
        side.knots = [float(np.interp(k, hx, vx)) for k in side.knots]


def _curve_layout(kc):
    """(degree, interior knots on [0, 1]) of a curve input, in its own parameter
    (= the Side's t). Rational curves: None (their layout needs the weights too)."""
    _, ext = square._k()
    d = ext.curve_cvs(kc)
    if d.get("weights") and max(abs(w - 1.0) for w in d["weights"]) > 1e-12:
        return None, None
    U, p = list(d["knots"]), int(d["degree"])
    lo, hi = U[0], U[-1]
    inner = [(u - lo) / ((hi - lo) or 1.0) for u in U[p + 1:len(U) - p - 1]]
    return p, [k for k in inner if 1e-9 < k < 1 - 1e-9]


def build(inputs, p):
    m = int(p.get("samples", 48))
    raw = [sample_side(o, r, m) for o, r in inputs]
    for sd, (o, r) in zip(raw, inputs):
        sd.src = (o.name, r)
        if sd.knots is None and not features.is_surface_edge(r):
            sd.degree, sd.knots = _curve_layout(features.input_curve(o, r))
    size = float(np.linalg.norm(np.ptp(np.vstack([s.P for s in raw]), axis=0))) or 1.0
    tol = TOUCH_REL * size * 10
    prof_idx, rail_idx = classify(raw, tol)
    if len(prof_idx) < 2:
        raise RuntimeError("Loft with rails needs at least two profiles")
    profs = [raw[i] for i in prof_idx]
    rails = [raw[i] for i in rail_idx]

    # order the profiles along rail A; every rail runs from the first profile to the last
    if rails:
        hit_a = [_contact(pr, rails[0])[1] for pr in profs]
        order = list(np.argsort(hit_a))
        profs = [profs[i] for i in order]
        prof_idx = [prof_idx[i] for i in order]
        for j, rl in enumerate(rails):
            if _contact(profs[0], rl)[1] > _contact(profs[-1], rl)[1]:
                rails[j] = rl.reversed()
    # every profile starts on rail A (2 rails) or runs like the previous one
    for k, pr in enumerate(profs):
        if len(rails) == 2:
            flip = _nearest_t(rails[0], pr.P[-1])[1] < _nearest_t(rails[0], pr.P[0])[1]
        else:
            flip = k > 0 and geom.should_flip(profs[k - 1].P, pr.P)
        if flip:
            profs[k] = pr.reversed()
    # the rails must end on the end profiles
    for j, rl in enumerate(rails if len(rails) == 2 else []):
        for pr, end in ((profs[0], 0), (profs[-1], -1)):
            d = _nearest_t(pr, rl.P[end])[1]
            if d > tol:
                raise RuntimeError(f"rail {j + 1} ({rl.src[0]} {rl.src[1]}) does not end on the "
                                   f"{'first' if end == 0 else 'last'} profile (gap {d:.3g})")

    # v of every profile: where it meets the rails (mean of both), else chord length
    if len(rails) == 2:
        ha = [_nearest_t(rails[0], pr.P[0])[0] for pr in profs]
        hb = [_nearest_t(rails[1], pr.P[-1])[0] for pr in profs]
        v = [0.5 * (a + b) for a, b in zip(ha, hb)]
    elif len(rails) == 1:
        hits = [_contact(pr, rails[0]) for pr in profs]
        for k, (_, h, d) in enumerate(hits):
            if d > tol:
                raise RuntimeError(f"the rail does not touch profile {k + 1} (gap {d:.3g})")
        v = [h for _, h, _ in hits]
    else:
        dist = [float(np.mean(np.linalg.norm(profs[k + 1].P - profs[k].P, axis=1))) for k in range(len(profs) - 1)]
        c = np.concatenate([[0.0], np.cumsum(dist)])
        v = list(c / (c[-1] or 1.0))
    v = [(x - v[0]) / ((v[-1] - v[0]) or 1.0) for x in v]
    v[0], v[-1] = 0.0, 1.0
    if any(b - a < 1e-3 for a, b in zip(v[:-1], v[1:])):
        raise RuntimeError("two profiles meet the rails at the same place")

    # sides: bottom = first profile, top = last profile (both along +u)
    square.EXTRA_G0[:] = []
    if len(rails) == 2:
        _remap(rails[0], ha, v)
        _remap(rails[1], hb, v)
        left, right = rails[0], rails[1]
    else:
        left = _interp_side([pr.P[0] for pr in profs], v, m)
        right = _interp_side([pr.P[-1] for pr in profs], v, m)
        if len(rails) == 1:
            rl = rails[0]
            # the rail crosses profile k at its parameter a_k; in between, u is
            # interpolated linearly in v
            a = [h[0] for h in hits]
            rhit = [h[1] for h in hits]
            order = np.argsort(rhit)
            # the free sides start as the guide carried along by the profiles'
            # ends (offset from the contact point, interpolated in v), not as a
            # straight line between the profiles: the free edge then follows the
            # rail's shape instead of being pulled flat
            rv = np.interp(rl.t, np.asarray(rhit)[order], np.asarray(v)[order])
            tt = np.linspace(0.0, 1.0, m)
            Rv = np.array([np.interp(tt, rv, rl.P[:, c]) for c in range(3)]).T
            for side, end in ((left, 0), (right, -1)):
                off = np.array([pr.P[end] - _at(pr, ak) for pr, ak in zip(profs, a)])
                O = np.array([np.interp(tt, v, off[:, c]) for c in range(3)]).T
                side.P, side.t = Rv + O, tt
                side.shaped = True     # carries the guide's shape: corner checks may use it
                if rl.knots is not None and len(rhit) == 2:
                    side.degree = rl.degree
                    side.knots = [float(np.interp(k, np.asarray(rhit)[order], np.asarray(v)[order])) for k in rl.knots]
            for t_, P_ in zip(rl.t, rl.P):
                vv = float(np.interp(t_, np.asarray(rhit)[order], np.asarray(v)[order]))
                uu = float(np.interp(vv, v, a))
                square.EXTRA_G0.append((min(max(uu, 0.0), 1.0), min(max(vv, 0.0), 1.0), P_.copy()))
    for k, pr in enumerate(profs[1:-1], 1):
        for t_, P_ in zip(pr.t, pr.P):
            square.EXTRA_G0.append((float(t_), float(v[k]), P_.copy()))

    sides = [profs[0], right, profs[-1], left]
    names = ["start", "rb", "end", "ra"]
    conts = [p.get(f"lr_{n}", "G0") for n in names]
    if len(rails) < 2:
        conts[1] = conts[3] = "G0"
    weights = [float(p.get(f"lw_{n}", 1.0)) for n in names]
    ks = [CONT[c] if sd.has_surface and not sd.free else 0 for c, sd in zip(conts, sides)]
    try:
        surf, info = _solve(sides, ks, weights, int(p.get("lr_degree", 0) or 0),
                            int(p.get("lr_spans_u", 0)), int(p.get("lr_spans_v", 0)), profs[1:-1],
                            guide=rails[0] if len(rails) == 1 else None)
        surf = square._orient_like_neighbours(surf, sides)
    finally:
        square.EXTRA_G0[:] = []
    info["edge_sides"] = {n: bool(sd.has_surface and not sd.free) for n, sd in zip(names, sides)}
    info["profiles"] = len(profs)
    info["rails"] = len(rails)
    info["corner_notes"] = list(square.LAST_NOTES)
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return surf


def _min_spans(rows_t, p, tol, max_spans=24):
    """Fewest uniform spans of degree p that fit every (t, points) row within tol."""
    for s in range(1, max_spans + 1):
        n = p + s
        U = geom.clamped_uniform_knots(n, p)
        ok = True
        for t, P in rows_t:
            X = geom._ls_fit(np.asarray(t), P, U, p, n)
            if np.linalg.norm(geom.interp_matrix(np.asarray(t), U, p, n) @ X - P, axis=1).max() > tol:
                ok = False
                break
        if ok:
            return s
    return max_spans


def _matched_layout(sides, ks, inner, guide):
    """The inputs' own layout (Plasticity / Alias style): along u the profiles'
    degree and knots, along v the rails' (or the guide's); the degree is raised
    only as far as the continuity across that direction needs, a lower-degree
    input keeping its knots with multiplicity. None when an input has no
    layout (a trim edge, a rational curve)."""
    bottom, right, top, left = sides
    out = []
    for group, (ka, kb) in (([bottom, top] + list(inner), (ks[3], ks[1])),
                            ([sd for sd in (left, right) if not sd.free] or ([guide] if guide is not None else []),
                             (ks[0], ks[2]))):
        need = 1 + ka + kb
        if not group:                    # no rails, no guide: the lightest that holds the profiles
            deg = max(1, need) if not inner else max(3, need)
            interior = []
            n_rows = len(inner) + 2
            while deg + 1 + len(interior) < n_rows:
                pts = [0.0] + interior + [1.0]
                j = int(np.argmax(np.diff(pts)))
                interior = sorted(interior + [0.5 * (pts[j] + pts[j + 1])])
        else:
            if any(sd is None or sd.knots is None or sd.degree is None for sd in group):
                return None
            deg = max(max(sd.degree for sd in group), need)
            if deg > 7:
                return None
            interior = square._merge_knots_mult([(sd.knots, deg - sd.degree + 1) for sd in group])
        while deg + 1 + len(interior) < ka + kb + 2:
            pts = [0.0] + interior + [1.0]
            j = int(np.argmax(np.diff(pts)))
            interior = sorted(interior + [0.5 * (pts[j] + pts[j + 1])])
        out.append((deg, [0.0] * (deg + 1) + interior + [1.0] * (deg + 1)))
    return out


def _solve(sides, ks, weights, degree, spans_u, spans_v, inner, guide=None):
    _, ext = square._k()
    conflicts = square.corner_fades(sides, ks)
    square.LAST_NOTES[:] = square.corner_notes(sides, conflicts)
    size = float(np.linalg.norm(np.ptp(np.vstack([sd.P for sd in sides]), axis=0))) or 1.0

    def make(r):
        S, (Uu, Uv), _, info = r
        nu, nv = S.shape[:2]
        pts = [tuple(map(float, S[i, j])) for i in range(nu) for j in range(nv)]
        info["folds"] = square.net_folds(S)
        return ext.surface_from_cvs(info["degree"][0], info["degree"][1], pts, Uu, Uv), info

    trim_side = any(sd.has_surface and sd.knots is None and not sd.free for sd in sides)
    gap_rel = 1e-3 if trim_side else 1e-4

    def score_of(q_):
        sc = max(q_["gap"] / (gap_rel * q_["size"]), q_["angle"] / 0.1, q_["curv"] / 0.10)
        return sc + 100.0 * q_["folds"] if q_.get("folds") else sc

    degrees = [degree] if degree else ([5, 3] if max(ks) >= 2 else [3, 5])
    tol = gap_rel * size
    cands = []
    lay = None if (degree or spans_u or spans_v) else _matched_layout(sides, ks, inner, guide)
    if lay is not None:
        (p_, Uu), (q_, Uv) = lay
        try:
            r = square._build_knots(sides, ks, weights, Uu, p_, Uv, q_, const_alpha=True)
            surf, info = make(r)
            info.update(square.measure(surf, sides, ks))
            info["matched"] = True
            sc = score_of(info)
            if sc <= 1.0:
                return surf, info          # the inputs' own layout holds everything
            cands.append((sc, surf, info))
        except RuntimeError:
            pass
    for p_ in degrees:
        # spans the boundaries alone need: profiles (and inner profiles) along u,
        # rails / free sides along v (inner profiles need rows to sit on)
        prof_rows = [(sd.t, sd.P) for sd in [sides[0], sides[2]] + list(inner)]
        rail_rows = [(sd.t, sd.P) for sd in (sides[1], sides[3]) if not sd.free]
        su0 = spans_u or _min_spans(prof_rows, p_, tol)
        n_inner = len(inner)
        sv0 = spans_v or max(1, _min_spans(rail_rows, p_, tol) if rail_rows else 1, n_inner + 1 - p_ + 1)
        while p_ + su0 < ks[1] + ks[3] + 2:
            su0 += 1
        while p_ + sv0 < ks[0] + ks[2] + 2:
            sv0 += 1
        tries = [(su0, sv0)] if (spans_u and spans_v) else \
            sorted({(su0 + a, sv0 + b) for a in range(0, 4) for b in range(0, 4)
                    if (not spans_u or a == 0) and (not spans_v or b == 0)},
                   key=lambda s: ((p_ + s[0]) * (p_ + s[1]), s))
        best_here = None
        for su, sv in tries[:10]:
            try:
                r = square._build(sides, ks, weights, p_, su, sv)
            except RuntimeError:
                continue
            surf, info = make(r)
            info.update(square.measure(surf, sides, ks))
            sc = score_of(info)
            cands.append((sc, surf, info))
            if best_here is None or sc < best_here:
                best_here = sc
            if sc <= 1.0:
                break
    if not cands:
        raise RuntimeError("too few CVs for the chosen continuity")
    return square._pick_lightest(cands)


# --------------------------------------------------------------------------
# Centreline loft: one rail used as a spine
# --------------------------------------------------------------------------

def _frames(kc, n=400):
    """Dense spine samples by arc length: points, rotation-minimising frames
    (T, N, B) and the arc-length fraction s."""
    from .sweeps import _arc_samples, _rmf, _perp
    P, T = _arc_samples(kc, n)
    N = _rmf(P, T, _perp(T[0], np.array([0.0, 0.0, 1.0])))
    B = np.cross(T, N)
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return P, T, N, B, L / (L[-1] or 1.0)


def _frame_at(fr, s):
    P, T, N, B, S = fr
    pt = np.array([np.interp(s, S, P[:, c]) for c in range(3)])
    t = np.array([np.interp(s, S, T[:, c]) for c in range(3)])
    n = np.array([np.interp(s, S, N[:, c]) for c in range(3)])
    t /= np.linalg.norm(t) or 1.0
    n -= (n @ t) * t
    n /= np.linalg.norm(n) or 1.0
    return pt, np.array([t, n, np.cross(t, n)])


def _compatible_profiles(kcs):
    """CV rows of all profiles on ONE layout: their own when they share degree,
    knots and weights (exact), else a shared-knot fit of their samples."""
    _, ext = square._k()
    ds = [ext.curve_cvs(k) for k in kcs]
    norm = lambda U: [(u - U[0]) / ((U[-1] - U[0]) or 1.0) for u in U]
    d0 = ds[0]
    same = all(d["degree"] == d0["degree"] and len(d["knots"]) == len(d0["knots"]) and
               np.allclose(norm(d["knots"]), norm(d0["knots"]), atol=1e-9) and
               np.allclose(d["weights"] or [1.0] * len(d["points"]), d0["weights"] or [1.0] * len(d0["points"]), atol=1e-12)
               for d in ds)
    if same:
        prof = {"degree": d0["degree"], "knots": list(norm(d0["knots"])), "weights": list(d0["weights"] or [])}
        return [np.array(d["points"], float) for d in ds], prof, True
    # different layouts: each profile fitted on the SAME uniform cubic knots, on
    # its own chord-length parameters from dense samples (a shared parameter
    # would put one profile's corners where another is smooth, and a near-
    # interpolation of few samples wiggles between them)
    rows, ts = [], []
    for k in kcs:
        t0, t1 = k.parameter_range()
        S = np.array([k.evaluate(t) for t in np.linspace(t0, t1, 240)])
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))])
        rows.append(S)
        ts.append(d / (d[-1] or 1.0))
    size = float(np.linalg.norm(np.ptp(np.vstack(rows), axis=0))) or 1.0
    deg, best = 3, None
    for n in range(deg + 1, 61):
        U = geom.clamped_uniform_knots(n, deg)
        cvs, err = [], 0.0
        for S, t in zip(rows, ts):
            X = geom._ls_fit(t, S, U, deg, n)
            err = max(err, float(np.linalg.norm(geom.interp_matrix(t, U, deg, n) @ X - S, axis=1).max()))
            cvs.append(X)
        best = (cvs, U)
        if err <= 1e-4 * size:
            break
    cvs, U = best
    return [np.asarray(c, float) for c in cvs], {"degree": deg, "knots": [float(u) for u in U], "weights": []}, False


def _interp_over(s_k, values, s):
    """Interpolate per-profile arrays (K, ...) at spine positions s: linear for
    2 profiles, an interpolating cubic (or lower) B-spline for more."""
    V = np.asarray(values, float)
    K = len(V)
    flat = V.reshape(K, -1)
    if K == 2:
        f = (np.asarray(s) - s_k[0]) / ((s_k[1] - s_k[0]) or 1.0)
        out = (1 - f)[:, None] * flat[0] + f[:, None] * flat[1]
    else:
        deg = min(3, K - 1)
        x = (np.asarray(s_k) - s_k[0]) / ((s_k[-1] - s_k[0]) or 1.0)
        U = [0.0] * (deg + 1) + [sum(x[j:j + deg]) / deg for j in range(1, K - deg)] + [1.0] * (deg + 1)
        X = np.linalg.solve(geom.interp_matrix(x, U, deg, K), flat)
        xs = (np.asarray(s) - s_k[0]) / ((s_k[-1] - s_k[0]) or 1.0)
        out = geom.interp_matrix(np.clip(xs, 0, 1), U, deg, K) @ X
    return out.reshape((len(s),) + V.shape[1:])


def surface_gap(surf, point_sets, nu=60, nv=120):
    """Largest distance from the points to the surface: foot points refined from
    a fine grid (the 17x17 start of square._foot can land on the wrong side of a
    long or closed surface)."""
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    us, vs = np.linspace(u0, u1, nu), np.linspace(v0, v1, nv)
    G = np.array([[surf.evaluate(u, v) for v in vs] for u in us]).reshape(-1, 3)
    gap = 0.0
    for X in point_sets:
        for x in X:
            k = int(np.argmin(np.linalg.norm(G - x, axis=1)))
            u, v = square._foot(surf, x, (us[k // nv], vs[k % nv]))
            gap = max(gap, float(np.linalg.norm(np.array(surf.evaluate(u, v)) - x)))
    return gap


def build_centreline(inputs, p):
    """Profiles (2+, in order) and a spine (the last input): every profile is
    expressed in the spine's rotation-minimising frame where the spine passes
    it, and the sections in between are interpolated in those frames, so the
    loft follows the spine (Fusion's centreline rail). The spine need not touch
    the profiles."""
    from .sweeps import _surface_from_rows
    *prof_in, spine_in = inputs
    if len(prof_in) < 2:
        raise RuntimeError("Centreline loft needs at least two profiles and a spine (picked last)")
    kcs = [features.input_curve(o, r) for o, r in prof_in]
    fr = _frames(features.input_curve(*spine_in))
    SP = fr[0]
    # where the spine passes each profile: nearest spine point to the profile's centre
    s_k = []
    for kc in kcs:
        t0, t1 = kc.parameter_range()
        Q = np.array([kc.evaluate(t) for t in np.linspace(t0, t1, 60)])
        c = Q.mean(axis=0)
        s_k.append(float(fr[4][int(np.argmin(np.linalg.norm(SP - c, axis=1)))]))
    order = np.argsort(s_k)
    kcs = [kcs[i] for i in order]
    s_k = [s_k[i] for i in order]
    if any(b - a < 1e-4 for a, b in zip(s_k[:-1], s_k[1:])):
        raise RuntimeError("two profiles sit at the same place on the spine")
    # profiles run the same way (no twist)
    for k in range(1, len(kcs)):
        a = geom.sample_curve(kcs[k - 1], 16)
        b = geom.sample_curve(kcs[k], 16)
        pa, Fa = _frame_at(fr, s_k[k - 1])
        pb, Fb = _frame_at(fr, s_k[k])
        if geom.should_flip((a - pa) @ Fa.T, (b - pb) @ Fb.T):
            kcs[k] = geom.reverse_curve(kcs[k])
    cvs, prof, exact = _compatible_profiles(kcs)
    local = []
    for C, s in zip(cvs, s_k):
        pt, F = _frame_at(fr, s)
        local.append((C - pt) @ F.T)
    n = int(p.get("samples", 32))
    ss = np.linspace(s_k[0], s_k[-1], n)
    for s in s_k:                      # sample exactly at every profile
        ss = np.append(ss, s)
    ss = np.unique(np.round(ss, 12))
    # rotation about the spine between consecutive profiles (a closed profile's
    # seam, or a turned profile) is interpolated as an angle, not CV by CV:
    # interpolating the CVs of two turned circles would shrink the section
    rot = lambda q, a: np.column_stack([q[:, 0], np.cos(a) * q[:, 1] - np.sin(a) * q[:, 2],
                                        np.sin(a) * q[:, 1] + np.cos(a) * q[:, 2]])
    aligned, phi = [local[0]], [0.0]
    for q in local[1:]:
        a2, b2 = q[:, 1:3], aligned[-1][:, 1:3]
        psi = float(np.arctan2(np.sum(a2[:, 0] * b2[:, 1] - a2[:, 1] * b2[:, 0]), np.sum(a2 * b2)))
        aligned.append(rot(q, psi))              # turned onto the previous profile
        phi.append(phi[-1] - psi)                # q = rot(aligned, phi)
    shapes = _interp_over(s_k, aligned, ss)
    angles = _interp_over(s_k, np.array(phi)[:, None], ss)[:, 0]
    secs = [rot(q, a) for q, a in zip(shapes, angles)]
    rows = []
    for s, q in zip(ss, secs):
        pt, F = _frame_at(fr, s)
        rows.append(pt + q @ F)
    surf = _surface_from_rows(rows, prof)
    info = {"profiles": len(kcs), "rails": 1, "mode": "centreline", "exact_profiles": exact}
    d = square._k()[1].surface_cvs(surf)
    info["cvs"] = (d["count_u"], d["count_v"])
    info["degree"] = (d["degree_u"], d["degree_v"])
    gap = surface_gap(surf, [geom.sample_curve(kc, 21) for kc in kcs])
    info.update({"gap": gap, "angle": 0.0, "curv": 0.0, "corner_notes": [], "edge_sides": {}})
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return surf


@features.builder("loft_rails")
def build_loft_rails(inputs, p):
    if p.get("lr_mode", "RAILS") == "CENTRELINE":
        return build_centreline(inputs, p)
    return build(inputs, p)


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import EnumProperty, FloatProperty, IntProperty, StringProperty  # noqa: E402

from .ops import PickMixin, _decode_inputs, _encode_inputs, _kernel_ok, _select_only  # noqa: E402

LR_CONT = [("G0", "G0", "Position"), ("G1", "G1", "Tangent"), ("G2", "G2", "Curvature")]
LR_DEG = [("0", "Auto", "Degree 5 with G2, else 3 (fewest spans within tolerance)"), ("3", "3", ""), ("5", "5", "")]
SIDES = (("start", "Start Profile"), ("end", "End Profile"), ("ra", "Rail A"), ("rb", "Rail B"))


class NPROTO_OT_loft_rails(PickMixin, bpy.types.Operator):
    """Loft through profiles along 0, 1 or 2 rails, with G0 / G1 / G2 at the end
    profiles and the rails where they are surface edges (Fusion / SolidWorks
    loft). Pick the profiles in order, then the rails; Enter to finish"""
    bl_idname = "nproto.loft_rails"
    bl_label = "Loft Rails"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 2
    pick_max = 32
    finish_on_min = False
    extra_help = "   (profiles first, then up to 2 rails; Centreline mode: the spine last)"
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    lr_start: EnumProperty(name="Start Profile", items=LR_CONT, default="G0")
    lr_end: EnumProperty(name="End Profile", items=LR_CONT, default="G0")
    lr_ra: EnumProperty(name="Rail A", items=LR_CONT, default="G0")
    lr_rb: EnumProperty(name="Rail B", items=LR_CONT, default="G0")
    lw_start: FloatProperty(name="Weight", default=1.0, min=0.05, soft_max=3.0)
    lw_end: FloatProperty(name="Weight", default=1.0, min=0.05, soft_max=3.0)
    lw_ra: FloatProperty(name="Weight", default=1.0, min=0.05, soft_max=3.0)
    lw_rb: FloatProperty(name="Weight", default=1.0, min=0.05, soft_max=3.0)
    lr_mode: EnumProperty(name="Mode", items=[
        ("RAILS", "Rails", "Rails touch the profiles: 2 = side rails, 1 = guide curve, 0 = profiles only"),
        ("CENTRELINE", "Centreline", "The last pick is a spine: sections follow it in its frames (it needn't touch the profiles)")],
        default="RAILS")
    lr_degree: EnumProperty(name="Degree", items=LR_DEG, default="0")
    lr_spans_u: IntProperty(name="Spans U", default=0, min=0, max=32, description="Along the profiles (0 = auto)")
    lr_spans_v: IntProperty(name="Spans V", default=0, min=0, max=32, description="Along the rails (0 = auto)")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        return self.start_picking(context)

    def _params(self):
        p = {f"lr_{n}": getattr(self, f"lr_{n}") for n, _ in SIDES}
        p.update({f"lw_{n}": getattr(self, f"lw_{n}") for n, _ in SIDES})
        p.update({"lr_mode": self.lr_mode, "lr_degree": int(self.lr_degree), "lr_spans_u": self.lr_spans_u, "lr_spans_v": self.lr_spans_v})
        return p

    def draw(self, context):
        lay = self.layout
        lay.row().prop(self, "lr_mode", expand=True)
        inf = LAST_INFO
        if self.lr_mode == "CENTRELINE":
            if inf and inf.get("mode") == "centreline":
                lay.label(text=f"{inf['profiles']} profiles along the spine · {inf['cvs'][0]} x {inf['cvs'][1]} CVs · "
                               f"profiles {'exact' if inf.get('exact_profiles') else 'refitted to one layout'}", icon='MESH_GRID')
                lay.label(text=f"gap to the profiles {inf.get('gap', 0):.1e}", icon='DRIVER_DISTANCE')
            return
        rails = inf.get("rails", 2)
        edges = inf.get("edge_sides", {})
        for n, label in SIDES:
            if n in ("ra", "rb") and rails < 2:
                continue
            row = lay.row(align=True)
            row.label(text=label)
            sub = row.row(align=True)
            sub.enabled = edges.get(n, True)
            sub.prop(self, f"lr_{n}", expand=True)
            sub.prop(self, f"lw_{n}", text="")
        lay.prop(self, "lr_degree")
        row = lay.row(align=True)
        row.prop(self, "lr_spans_u")
        row.prop(self, "lr_spans_v")
        if inf:
            lay.label(text=f"{inf['profiles']} profiles · {inf['rails']} rail(s) · {inf['cvs'][0]} x {inf['cvs'][1]} CVs · "
                           f"degree {inf['degree'][0]}", icon='MESH_GRID')
            lay.label(text=f"gap {inf.get('gap', 0):.1e} · max angle {inf.get('angle', 0):.3f}° · "
                           f"curvature err {100 * inf.get('curv', 0):.1f}%", icon='DRIVER_DISTANCE')
            for note in inf.get("corner_notes", []):
                lay.label(text=note, icon='ERROR')

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            inputs = _decode_inputs(self.inputs)
            if len(inputs) < 2:
                raise ValueError("pick at least two profiles")
            params = self._params()
            g = features.run("loft_rails", inputs, params)
            from . import store
            obj = store.new_object(context, "LoftRails", g)
            if features.history_enabled(context):
                features.record(obj, "loft_rails", inputs, params)
        except Exception as exc:
            self.report({'ERROR'}, f"Loft Rails failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        inf = LAST_INFO
        self.report({'INFO'}, f"Loft Rails: {inf['profiles']} profiles, {inf['rails']} rail(s), "
                              f"{inf['cvs'][0]}x{inf['cvs'][1]} CVs, angle {inf.get('angle', 0):.3f}°, "
                              f"curvature {100 * inf.get('curv', 0):.1f}%")
        return {'FINISHED'}


classes = (NPROTO_OT_loft_rails,)
