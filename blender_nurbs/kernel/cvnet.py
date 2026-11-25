"""Control nets for Face CVs (sculpting one face of a solid), pure numpy.

A net is a dict in the kernel's surface_cvs layout: degree_u, degree_v,
count_u, count_v, knots_u, knots_v, points (u-major, count_u * count_v),
weights ([] when not rational).

- refine(net, nu, nv, du, dv): the same surface on a finer net (higher degree
  and / or more CVs). Exact: the target spline space holds the source, so a
  least-squares solve on samples gives it back to float precision.
- extend(net, dist): the same polynomial continued past each side by about
  dist (parameter steps from the speed at the end), CV count unchanged.
  Exact over the old domain; used so a sculpted face's new surface reaches
  past the walls it is re-trimmed against.
"""

import numpy as np


def basis(U, p, ts):
    """(len(ts), n) B-spline basis values on the clamped knots U, degree p;
    ts inside [U[p], U[n]] (the right end belongs to the last span)."""
    U = np.asarray(U, float)
    ts = np.asarray(ts, float)
    m = len(U) - 1
    spans = [i for i in range(m) if U[i] < U[i + 1]]
    N = np.zeros((len(ts), m))
    for i in spans:
        N[:, i] = (ts >= U[i]) & (ts < U[i + 1])
    last = spans[-1]
    N[ts >= U[last + 1], last] = 1.0
    for k in range(1, p + 1):
        M = np.zeros((len(ts), m - k))
        for i in range(m - k):
            d1, d2 = U[i + k] - U[i], U[i + k + 1] - U[i + 1]
            if d1 > 0:
                M[:, i] += (ts - U[i]) / d1 * N[:, i]
            if d2 > 0:
                M[:, i] += (U[i + k + 1] - ts) / d2 * N[:, i + 1]
        N = M
    return N


def _homogeneous(net):
    nu, nv = net["count_u"], net["count_v"]
    P = np.asarray(net["points"], float).reshape(nu, nv, 3)
    w = np.asarray(net.get("weights") or [1.0] * (nu * nv), float).reshape(nu, nv, 1)
    return np.concatenate([P * w, w], axis=2)


def _from_homogeneous(H, net):
    w = H[..., 3]
    P = H[..., :3] / w[..., None]
    rational = float(np.abs(w - 1.0).max()) > 1e-12
    out = dict(net)
    out["count_u"], out["count_v"] = int(H.shape[0]), int(H.shape[1])
    out["points"] = [list(map(float, q)) for q in P.reshape(-1, 3)]
    out["weights"] = [float(x) for x in w.reshape(-1)] if rational else []
    return out


def _samples(U, p, per=None):
    """Parameters spread over every non-empty span of U (enough to pin a
    spline of degree p down)."""
    U = np.asarray(U, float)
    k = per or p + 3
    ts = []
    for a, b in zip(U[:-1], U[1:]):
        if b > a:
            ts.extend(np.linspace(a, b, k)[:-1])
    ts.append(U[-1])
    return np.array(ts)


def _refit(H, Us, ps, Ut, pt, axis):
    """H's control rows along axis re-expressed on the knots Ut, degree pt
    (exact when the target space holds the source)."""
    ts = np.union1d(_samples(Us, ps), _samples(Ut, pt))
    Bs, Bt = basis(Us, ps, ts), basis(Ut, pt, ts)
    Hm = np.moveaxis(H, axis, 0)
    S = np.tensordot(Bs, Hm, axes=(1, 0))
    X = np.linalg.lstsq(Bt, S.reshape(len(ts), -1), rcond=None)[0].reshape((Bt.shape[1],) + Hm.shape[1:])
    return np.moveaxis(X, 0, axis)


def elevated_knots(U, p, q):
    """U raised from degree p to q (each distinct knot's multiplicity + q - p)."""
    U = [float(x) for x in U]
    out = []
    for x in sorted(set(U)):
        out += [x] * (U.count(x) + (q - p))
    return out


def refined_knots(U, p, q, n):
    """Knots of degree q holding the degree-p spline on U, with at least n
    CVs: elevated, then the knots still needed spread over the spans (more in
    longer ones, evenly inside each), so the layout stays balanced."""
    K = elevated_knots(U, p, q)
    need = n - (len(K) - q - 1)
    if need <= 0:
        return K
    D = sorted(set(K))
    spans = list(zip(D[:-1], D[1:]))
    k = [0] * len(spans)
    for _ in range(need):
        i = max(range(len(spans)), key=lambda j: ((spans[j][1] - spans[j][0]) / (k[j] + 1), -j))
        k[i] += 1
    new = [a + (b - a) * j / (m + 1) for (a, b), m in zip(spans, k) for j in range(1, m + 1)]
    return sorted(K + new)


def uniform_knots(n, p):
    """Clamped uniform knots on [0, 1] for n CVs of degree p."""
    return [0.0] * (p + 1) + [k / (n - p) for k in range(1, n - p)] + [1.0] * (p + 1)


def even_plane_net(origin, a, b, la, lb, nu, nv, du, dv):
    """A flat net over the rectangle origin + [0, la] a + [0, lb] b with its
    CVs evenly spaced (uniform knots). Exactly that flat face whatever the
    spacing: CVs in a plane make the plane, CVs on a straight edge keep it
    straight -- only the parameter speed varies a little toward the sides."""
    o, a, b = (np.asarray(x, float) for x in (origin, a, b))
    pts = [o + la * i / (nu - 1) * a + lb * j / (nv - 1) * b for i in range(nu) for j in range(nv)]
    return {"degree_u": int(du), "degree_v": int(dv), "count_u": int(nu), "count_v": int(nv),
            "knots_u": uniform_knots(nu, du), "knots_v": uniform_knots(nv, dv),
            "points": [list(map(float, q)) for q in pts], "weights": []}


def min_counts(net, du, dv):
    """The fewest CVs a refine to degrees du, dv can give."""
    return (len(elevated_knots(net["knots_u"], net["degree_u"], du)) - du - 1,
            len(elevated_knots(net["knots_v"], net["degree_v"], dv)) - dv - 1)


def refine(net, nu, nv, du, dv):
    """The same surface on knots of degrees du, dv with (at least) nu x nv CVs."""
    du, dv = max(int(du), net["degree_u"]), max(int(dv), net["degree_v"])
    Ku = refined_knots(net["knots_u"], net["degree_u"], du, int(nu))
    Kv = refined_knots(net["knots_v"], net["degree_v"], dv, int(nv))
    H = _homogeneous(net)
    H = _refit(H, net["knots_u"], net["degree_u"], Ku, du, 0)
    H = _refit(H, net["knots_v"], net["degree_v"], Kv, dv, 1)
    out = _from_homogeneous(H, net)
    out.update(degree_u=du, degree_v=dv, knots_u=Ku, knots_v=Kv)
    return out


def evaluate(net, us, vs):
    """(len(us), len(vs), 3) points of the net's surface on the grid."""
    H = _homogeneous(net)
    S = np.einsum("ai,ijk,bj->abk", basis(net["knots_u"], net["degree_u"], us), H,
                  basis(net["knots_v"], net["degree_v"], vs))
    return S[..., :3] / S[..., 3:]


def evaluate_at(net, us, vs):
    """(n, 3) points of the net's surface at the parameter pairs (us[i], vs[i])."""
    H = _homogeneous(net)
    S = np.einsum("ni,ijk,nj->nk", basis(net["knots_u"], net["degree_u"], us), H,
                  basis(net["knots_v"], net["degree_v"], vs))
    return S[:, :3] / S[:, 3:]


def project(net, Q, us, vs, steps=8):
    """(us, vs) polished so each S(u, v) is the foot of Q[i] (Gauss-Newton
    from the given start; parameters stay in the domain)."""
    Q = np.asarray(Q, float)
    us, vs = np.array(us, float), np.array(vs, float)
    Ku, Kv = net["knots_u"], net["knots_v"]
    lo_u, hi_u = Ku[net["degree_u"]], Ku[-net["degree_u"] - 1]
    lo_v, hi_v = Kv[net["degree_v"]], Kv[-net["degree_v"] - 1]
    h = 1e-7 * max(hi_u - lo_u, hi_v - lo_v)
    for _ in range(steps):
        S = evaluate_at(net, us, vs)
        du = np.where(us + h <= hi_u, h, -h)
        dv = np.where(vs + h <= hi_v, h, -h)
        Su = (evaluate_at(net, us + du, vs) - S) / du[:, None]
        Sv = (evaluate_at(net, us, vs + dv) - S) / dv[:, None]
        r = Q - S
        a, b, c = (Su * Su).sum(1), (Su * Sv).sum(1), (Sv * Sv).sum(1)
        g1, g2 = (Su * r).sum(1), (Sv * r).sum(1)
        det = a * c - b * b
        ok = np.abs(det) > 1e-30
        d1 = np.where(ok, (c * g1 - b * g2) / np.where(ok, det, 1.0), 0.0)
        d2 = np.where(ok, (a * g2 - b * g1) / np.where(ok, det, 1.0), 0.0)
        us = np.clip(us + d1, lo_u, hi_u)
        vs = np.clip(vs + d2, lo_v, hi_v)
    return us, vs


def _end_steps(net, axis, dist):
    """Parameter steps (start, end) along axis that reach about dist past each
    end (0 at a collapsed end, a pole)."""
    K = np.asarray(net["knots_u" if axis == 0 else "knots_v"], float)
    Ko = np.asarray(net["knots_v" if axis == 0 else "knots_u"], float)
    p = net["degree_u" if axis == 0 else "degree_v"]
    a, b = K[p], K[-p - 1]
    h = 1e-4 * (b - a)
    across = np.linspace(Ko[0], Ko[-1], 9)
    out = []
    for t0, t1 in ((a, a + h), (b - h, b)):
        if axis == 0:
            P = evaluate(net, [t0, t1], across)
            sp = np.linalg.norm(P[1] - P[0], axis=-1) / h
        else:
            P = evaluate(net, across, [t0, t1])
            sp = np.linalg.norm(P[:, 1] - P[:, 0], axis=-1) / h
        s = float(sp.min())
        out.append(0.0 if s <= 1e-12 else min(dist / s, 2.0 * (b - a)))
    return out


def extend(net, dist, tol):
    """The net's surface continued as the same polynomial past each side
    (about dist further in space). Raises when the continuation can't be
    trusted (a weight would reach 0, or the old part moved more than tol)."""
    H = _homogeneous(net)
    cur = dict(net)
    for axis in (0, 1):
        key, p = ("knots_u", net["degree_u"]) if axis == 0 else ("knots_v", net["degree_v"])
        K = list(map(float, cur[key]))
        s0, s1 = _end_steps(cur, axis, dist)
        if s0 <= 0.0 and s1 <= 0.0:
            continue
        Kn = [K[0] - s0] * (p + 1) + K[p + 1:-p - 1] + [K[-1] + s1] * (p + 1)
        ts = _samples(K, p, 2 * p + 4)
        Bs, Bt = basis(K, p, ts), basis(Kn, p, ts)
        Hm = np.moveaxis(H, axis, 0)
        S = np.tensordot(Bs, Hm, axes=(1, 0))
        X = np.linalg.lstsq(Bt, S.reshape(len(ts), -1), rcond=None)[0].reshape(Hm.shape)
        if float(np.abs(Bt @ X.reshape(len(X), -1) - S.reshape(len(ts), -1)).max()) > tol:
            raise RuntimeError("the sculpted surface could not be continued past its sides")
        if X[..., 3].min() <= 1e-9:
            raise RuntimeError("a weight of the continued surface reached 0")
        # (no knots at the old ends: a wall cutting the face along a knot line
        # came out 5e-3 off in the kernel's intersection)
        H = np.moveaxis(X, 0, axis)
        cur = _from_homogeneous(H, cur)
        cur[key] = Kn
    return cur


def subpatch(net, u0, u1, v0, v1):
    """The part of the net's surface over [u0, u1] x [v0, v1] as a net of
    its own (exact; the knots inside are kept, the ends clamped there)."""
    H = _homogeneous(net)
    cur = dict(net)
    for axis, (a, b) in ((0, (u0, u1)), (1, (v0, v1))):
        key, p = ("knots_u", net["degree_u"]) if axis == 0 else ("knots_v", net["degree_v"])
        K = list(map(float, cur[key]))
        a, b = max(float(a), K[p]), min(float(b), K[-p - 1])
        if (a <= K[p] and b >= K[-p - 1]) or b <= a:
            continue
        Kn = [a] * (p + 1) + [k for k in K[p + 1:-p - 1] if a < k < b] + [b] * (p + 1)
        ts = _samples(Kn, p, 2 * p + 4)
        Bs, Bt = basis(K, p, ts), basis(Kn, p, ts)
        Hm = np.moveaxis(H, axis, 0)
        S = np.tensordot(Bs, Hm, axes=(1, 0))
        X = np.linalg.lstsq(Bt, S.reshape(len(ts), -1), rcond=None)[0]
        H = np.moveaxis(X.reshape((Bt.shape[1],) + Hm.shape[1:]), 0, axis)
        cur = _from_homogeneous(H, cur)
        cur[key] = Kn
    return cur


def plane_net(origin, a, b, la, lb):
    """A flat 2 x 2 degree-1 net over the rectangle origin + [0, la] a + [0, lb] b."""
    o, a, b = (np.asarray(x, float) for x in (origin, a, b))
    pts = [o, o + lb * b, o + la * a, o + la * a + lb * b]
    return {"degree_u": 1, "degree_v": 1, "count_u": 2, "count_v": 2, "knots_u": [0.0, 0.0, 1.0, 1.0],
            "knots_v": [0.0, 0.0, 1.0, 1.0], "points": [list(map(float, q)) for q in pts], "weights": []}


# --------------------------------------------------------------------------
# Rebuild: a sculpted net refitted to a new layout (not exact)
# --------------------------------------------------------------------------

def _rect_frame(net, S, tol):
    """(origin, a, b, n, la, lb) when the surface S (a grid of its points)
    stands over a rectangle -- its corners and sides on one, the map to it
    one-to-one (a flat face sculpted mostly along its normal) -- else None."""
    o = S[0, 0]
    ea, eb = S[-1, 0] - o, S[0, -1] - o
    n = np.cross(ea, eb)
    if np.linalg.norm(n) <= 1e-30 or np.linalg.norm(ea) <= tol:
        return None
    n /= np.linalg.norm(n)
    a = ea - (ea @ n) * n
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    X, Y = (S - o) @ a, (S - o) @ b
    la, lb = float(X[-1, 0]), float(Y[0, -1])
    if la <= tol or lb <= tol:
        return None
    sides = (np.abs(X[0]).max(), np.abs(X[-1] - la).max(), np.abs(Y[:, 0]).max(), np.abs(Y[:, -1] - lb).max())
    if max(sides) > tol:
        return None
    J = np.diff(X, axis=0)[:, :-1] * np.diff(Y, axis=1)[:-1] - np.diff(X, axis=1)[:-1] * np.diff(Y, axis=0)[:, :-1]
    if J.min() <= 0:
        return None
    return o, a, b, n, la, lb


def _invert_xy(net, frame, x, y, us0, vs0, steps=12):
    """Source parameters (u, v) whose points stand over (x, y) in the frame."""
    o, a, b = frame[0], frame[1], frame[2]
    Ku, Kv = net["knots_u"], net["knots_v"]
    lo_u, hi_u = Ku[net["degree_u"]], Ku[-net["degree_u"] - 1]
    lo_v, hi_v = Kv[net["degree_v"]], Kv[-net["degree_v"] - 1]
    us, vs = np.array(us0, float), np.array(vs0, float)
    h = 1e-7 * max(hi_u - lo_u, hi_v - lo_v)
    XY = lambda u, v: np.stack([(evaluate_at(net, u, v) - o) @ a, (evaluate_at(net, u, v) - o) @ b], axis=1)
    for _ in range(steps):
        F = XY(us, vs)
        du = np.where(us + h <= hi_u, h, -h)
        dv = np.where(vs + h <= hi_v, h, -h)
        Fu = (XY(us + du, vs) - F) / du[:, None]
        Fv = (XY(us, vs + dv) - F) / dv[:, None]
        r = np.stack([x, y], axis=1) - F
        det = Fu[:, 0] * Fv[:, 1] - Fu[:, 1] * Fv[:, 0]
        det = np.where(np.abs(det) > 1e-30, det, 1e-30)
        us = np.clip(us + (r[:, 0] * Fv[:, 1] - r[:, 1] * Fv[:, 0]) / det, lo_u, hi_u)
        vs = np.clip(vs + (Fu[:, 0] * r[:, 1] - Fu[:, 1] * r[:, 0]) / det, lo_v, hi_v)
    return us, vs


def rebuild(net, nu, nv, deg, tol):
    """(net, max deviation): the net's surface refitted on nu x nv CVs of
    degree deg with uniform knots. Over a rectangle (a flat face sculpted
    along its normal) the new CVs form an even grid on it and only their
    heights are fitted; otherwise the fit keeps the surface's parameters."""
    deg = int(deg)
    nu, nv = max(int(nu), deg + 1), max(int(nv), deg + 1)
    Ku, Kv = net["knots_u"], net["knots_v"]
    a_u, b_u = Ku[net["degree_u"]], Ku[-net["degree_u"] - 1]
    a_v, b_v = Kv[net["degree_v"]], Kv[-net["degree_v"] - 1]
    ku = max(4 * nu, 4 * net["count_u"]) + 1
    kv = max(4 * nv, 4 * net["count_v"]) + 1
    us, vs = np.linspace(a_u, b_u, ku), np.linspace(a_v, b_v, kv)
    S = evaluate(net, us, vs)
    Tu, Tv = uniform_knots(nu, deg), uniform_knots(nv, deg)
    ss, ts = np.linspace(0, 1, 4 * nu + 1), np.linspace(0, 1, 4 * nv + 1)
    Bs, Bt = basis(Tu, deg, ss), basis(Tv, deg, ts)
    frame = _rect_frame(net, S, tol)
    if frame is not None:
        o, a, b, n, la, lb = frame
        xs = Bs @ (la * np.arange(nu) / (nu - 1))            # where the even grid's iso lines stand
        ys = Bt @ (lb * np.arange(nv) / (nv - 1))
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
        u0 = a_u + (b_u - a_u) * Xg.ravel() / la
        v0 = a_v + (b_v - a_v) * Yg.ravel() / lb
        uu, vv = _invert_xy(net, frame, Xg.ravel(), Yg.ravel(), u0, v0)
        Q = evaluate_at(net, uu, vv)
        H = ((Q - o) @ n).reshape(len(ss), len(ts))
        Hc = np.linalg.lstsq(Bs, np.linalg.lstsq(Bt, H.T, rcond=None)[0].T, rcond=None)[0]
        I, J = np.meshgrid(la * np.arange(nu) / (nu - 1), lb * np.arange(nv) / (nv - 1), indexing="ij")
        P = o + I[..., None] * a + J[..., None] * b + Hc[..., None] * n
        dev_pts = Q.reshape(len(ss), len(ts), 3)
        T = {"degree_u": deg, "degree_v": deg, "count_u": nu, "count_v": nv, "knots_u": Tu, "knots_v": Tv,
             "points": [list(map(float, q)) for q in P.reshape(-1, 3)], "weights": []}
        dev = float(np.linalg.norm(evaluate(T, ss, ts) - dev_pts, axis=-1).max())
    else:
        Sg = evaluate(net, a_u + (b_u - a_u) * ss, a_v + (b_v - a_v) * ts)
        C = np.linalg.lstsq(Bs, np.linalg.lstsq(Bt, Sg.transpose(1, 0, 2).reshape(len(ts), -1), rcond=None)[0]
                            .reshape(nv, len(ss), 3).transpose(1, 0, 2).reshape(len(ss), -1), rcond=None)[0]
        P = C.reshape(nu, nv, 3)
        T = {"degree_u": deg, "degree_v": deg, "count_u": nu, "count_v": nv,
             "knots_u": [a_u + (b_u - a_u) * k for k in Tu], "knots_v": [a_v + (b_v - a_v) * k for k in Tv],
             "points": [list(map(float, q)) for q in P.reshape(-1, 3)], "weights": []}
        dev = float(np.linalg.norm(evaluate(T, a_u + (b_u - a_u) * ss, a_v + (b_v - a_v) * ts) - Sg, axis=-1).max())
    # (also between the fitted samples)
    mid_u, mid_v = 0.5 * (us[1:] + us[:-1]), 0.5 * (vs[1:] + vs[:-1])
    if frame is not None:
        Sm = evaluate(net, mid_u, mid_v)
        Xm, Ym = (Sm - frame[0]) @ frame[1], (Sm - frame[0]) @ frame[2]
        Hm = (Sm - frame[0]) @ frame[3]
        sm_, tm_ = _invert_param(Tu, deg, la * np.arange(nu) / (nu - 1), Xm.ravel()), \
            _invert_param(Tv, deg, lb * np.arange(nv) / (nv - 1), Ym.ravel())
        Hfit = np.einsum("ki,ij,kj->k", basis(Tu, deg, sm_), Hc, basis(Tv, deg, tm_))
        dev = max(dev, float(np.abs(Hfit - Hm.ravel()).max()))
    else:
        dev = max(dev, float(np.linalg.norm(evaluate(T, mid_u, mid_v) - evaluate(net, mid_u, mid_v), axis=-1).max()))
    return T, dev


def _invert_param(K, p, coeffs, x):
    """s in [0, 1] with sum N_i(s) coeffs_i = x (a monotone 1-D spline)."""
    s = np.linspace(0, 1, 4001)
    f = basis(K, p, s) @ coeffs
    return np.clip(np.interp(x, f, s), 0.0, 1.0)


def relayout(net, op, tol):
    """The net after a layout step op = [kind, nu, nv, deg]: "add" (exact
    refine, the shape kept) or "rebuild" (refit). -> (net, deviation)."""
    kind, nu, nv, deg = op
    if kind == "add":
        return refine(net, nu, nv, deg, deg), 0.0
    return rebuild(net, nu, nv, deg, tol)


def extend_curve(c, at_start, dist, tol):
    """A curve {degree, points, knots, weights} continued as the same
    polynomial about dist past its start (at_start) or end; the old part kept
    to tol. Raises when a weight would reach 0 (an arc run too far)."""
    p, K = int(c["degree"]), [float(k) for k in c["knots"]]
    P = np.asarray(c["points"], float)
    w = np.asarray(c.get("weights") or [1.0] * len(P), float)
    H = np.c_[P * w[:, None], w]
    a, b = K[p], K[-p - 1]
    h = 1e-4 * (b - a)
    ev = lambda t: (lambda S: S[:, :3] / S[:, 3:])(basis(K, p, t) @ H)
    t0, t1 = (a, a + h) if at_start else (b - h, b)
    sp = float(np.linalg.norm(np.diff(ev([t0, t1]), axis=0))) / h
    if sp <= 1e-12:
        raise RuntimeError("the curve stands still at that end")
    s = min(dist / sp, 2.0 * (b - a))
    Kn = ([K[0] - s] * (p + 1) + K[p + 1:-p - 1] + [K[-1]] * (p + 1)) if at_start else \
        ([K[0]] * (p + 1) + K[p + 1:-p - 1] + [K[-1] + s] * (p + 1))
    ts = _samples(K, p, 2 * p + 4)
    Bs, Bt = basis(K, p, ts), basis(Kn, p, ts)
    S = Bs @ H
    X = np.linalg.lstsq(Bt, S, rcond=None)[0]
    if float(np.abs(Bt @ X - S).max()) > tol:
        raise RuntimeError("the curve could not be continued")
    if X[:, 3].min() <= 1e-9:
        raise RuntimeError("a weight of the continued curve reached 0")
    rational = float(np.abs(X[:, 3] - 1.0).max()) > 1e-12
    return {"degree": p, "points": [list(map(float, q)) for q in X[:, :3] / X[:, 3:]], "knots": Kn,
            "weights": [float(x) for x in X[:, 3]] if rational else []}
