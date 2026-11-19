"""Offset on Surface.

Offsets curves that lie on a surface by a distance measured ON the surface:
every point of the curve walks along the surface, square to the curve, for
the distance (a geodesic, the straightest path on the surface). The result
lies on the surface: a border strip, a panel gap or a split line next to a
trim edge stays the same width across the surface's curvature, where Offset
Curve (in a plane) would move off it.

* Input: curves on the surface (drawn on it, projected, intersections) or
  surface edges. Curves near but not on the surface are pulled onto it first
  (the gap is reported). Touching inputs are chained and offset as one.
* Target: one or more NURBS objects; joined faces are crossed (the walk goes
  on into the neighbouring face), trimmed faces stop it at their trim.
  Where the walk runs off the surface (a free edge or a trim), the offset
  ends there: the result is cut, as the surface allows.
* Corners of the input (tangent breaks): on the outside of the turn filled
  Round (a circle on the surface about the corner: geodesic rays at every
  angle between the two sides), Sharp (both offsets continued along the
  surface until they meet) or Chamfer (a line on the surface between them);
  on the inside both pieces are trimmed where they cross.
* Folds: where the distance is larger than the curve's radius of curvature
  on the surface, the offset would loop back: the loop is cut out at the
  crossing that closes it (in a tangent-plane chart there: locally exact).
* The walk: a predictor step x + h t projected onto the surface, the
  direction carried along by the rotation that takes one surface normal to
  the next (discrete parallel transport); second order, a last short step
  lands on the exact distance. A sphere's latitude circle offsets onto the
  latitude d / R further within 1e-6 of R at the default steps.
* Positive distances: to the left of the curve seen from the surface's
  outward normal on open curves; outward (the side where the curve grows) on
  closed ones. Flip turns that round; Both Sides makes the two.

The offset is fitted (cubic, within Tolerance) between its corners, which
stay corners: the spline then lies on the surface within Tolerance.
"""

import numpy as np

from . import curvejoin, curveoffset, curvetrim, features, store, surfeval

LAST_INFO = {}
PER_SPAN = 8                # samples per input span (at least)
MAX_SAMPLES = 2400          # per segment
_FAST = [False]             # the live preview: fewer samples, polylines, no fitting


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-300 else v * 0.0


def _unit_rows(V):
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    return V / np.where(n > 1e-300, n, 1.0)


def _dot(A, B):
    return np.einsum("ij,ij->i", A, B)


def _tangent_part(V, N):
    return V - _dot(V, N)[:, None] * N


def _transport(T, N0, N1):
    """T (tangent at normals N0) carried to the tangent planes of N1 by the
    smallest rotation taking N0 to N1."""
    v = np.cross(N0, N1)
    c = _dot(N0, N1)
    k = 1.0 / np.maximum(1.0 + c, 1e-12)
    out = T * c[:, None] + np.cross(v, T) + v * (_dot(v, T) * k)[:, None]
    return _unit_rows(_tangent_part(out, N1))


def _inside_polys(uv, polys):
    """Even-odd test of the points uv (m, 2) against the closed polygons."""
    if not len(uv):
        return np.zeros(0, bool)
    x, y = uv[:, 0][:, None], uv[:, 1][:, None]
    c = np.zeros(len(uv), bool)
    for P in polys:
        x1, y1 = P[:, 0][None, :], P[:, 1][None, :]
        x2, y2 = np.roll(P[:, 0], -1)[None, :], np.roll(P[:, 1], -1)[None, :]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = (x2 - x1) * (y - y1) / np.where(y2 != y1, y2 - y1, 1.0) + x1
        c ^= (np.count_nonzero(cond & (x < xs), axis=1) % 2) == 1
    return c


# --------------------------------------------------------------------------
# The surface the offset walks on
# --------------------------------------------------------------------------

class _Face:
    def __init__(self, face, size):
        from . import anamesh
        self.E = surfeval.Surf(face.surface())
        self.sgn = anamesh._orient(face, self.E)
        self.per = self.E.periodic()
        self.face = face
        self.polys = None
        if not anamesh.is_untrimmed(face, self.E, size):
            self.polys = []
            sag = 0.0
            for loop in face.boundary_loops():
                pts = []
                for e, fw in loop:
                    t0, t1 = e.parameter_range()
                    ts = np.linspace(t0, t1, 65) if fw else np.linspace(t1, t0, 65)
                    pts += [e.point_at(float(t)) for t in ts[:-1]]
                if len(pts) >= 6:
                    uv = self.E.feet(np.array(pts, float))
                    # the polygon keeps every other point; the rest measure its chord error
                    even, odd = uv[0::2], uv[1::2]
                    mid = 0.5 * (even + np.roll(even, -1, axis=0))[:len(odd)]
                    sag = max(sag, float(np.linalg.norm(odd - mid, axis=1).max()))
                    self.polys.append(even)
            if not self.polys:
                self.polys = None
            self.uv_tol = 3.0 * sag
        (u0, u1), (v0, v1) = self.E.u_domain, self.E.v_domain
        self.span = (u1 - u0, v1 - v0)
        self.uv_tol = getattr(self, "uv_tol", 0.0) + 1e-9 * max(self.span)

    def near_trim(self, uv):
        """Points within the polygon's chord error of the trim (on the trim edge)."""
        near = np.zeros(len(uv), bool)
        for P in self.polys:
            A, B = P, np.roll(P, -1, axis=0)
            AB = B - A
            L = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-300)
            for s in range(0, len(uv), 2048):
                X = uv[s:s + 2048, None, :]
                t = np.clip(np.einsum("mkj,kj->mk", X - A[None], AB) / L[None], 0.0, 1.0)
                d = np.linalg.norm(A[None] + t[..., None] * AB[None] - X, axis=2).min(axis=1)
                near[s:s + 2048] |= d <= self.uv_tol
        return near

    def frame(self, uv):
        D = self.E.at(uv[:, 0], uv[:, 1], 1)
        n, ln = surfeval.normals(D)
        return D[0, 0], n * self.sgn, ln

    def inside(self, uv, X, S, N, eps, on_trim=False, parts=False):
        """Whether the feet uv of X lie on the face: not held at the surface's
        border (the rest of X - S not along the normal) and inside its trim
        (on_trim: or on it, within the trim polygon's chord error). parts:
        (on the surface, and inside the trim too)."""
        r = X - S
        on = np.linalg.norm(_tangent_part(r, N), axis=1) <= eps
        ok = on.copy()
        if self.polys is not None and ok.any():
            w = uv[ok]
            ins = _inside_polys(w, self.polys)
            near = self.near_trim(w)
            if near.any():
                # the polygon is a chord of the trim: near it, the kernel's exact test
                exact = np.array([self.face.classify_uv(float(a), float(b)) != "outside" for a, b in w[near]])
                ins[near] = exact | on_trim
            ok[ok] = ins
        return (on, ok) if parts else ok


class Target:
    """The faces of the target objects (world space), and feet on them."""

    def __init__(self, breps):
        bbs = [b.bounding_box() for b in breps]
        lo = np.min([np.array(b[0], float) for b in bbs], axis=0)
        hi = np.max([np.array(b[1], float) for b in bbs], axis=0)
        self.size = float(np.linalg.norm(hi - lo)) or 1.0
        self.faces = []
        for b in breps:
            for f in b.faces():
                try:
                    self.faces.append(_Face(f, self.size))
                except Exception as exc:
                    print("NURBS offset on surface: skipping a face:", exc)
        if not self.faces:
            raise ValueError("the target has no B-spline faces")
        self.eps = 1e-7 * self.size

    def feet_all(self, X):
        """(face, uv, point, normal, distance) of the nearest point on any face."""
        m = len(X)
        F = np.full(m, -1)
        UV = np.zeros((m, 2))
        S = np.array(X, float)
        N = np.zeros((m, 3))
        best = np.full(m, np.inf)
        for fi, fc in enumerate(self.faces):
            uv = fc.E.feet(X, g=33, iters=16)
            P, n, _ = fc.frame(uv)
            d = np.linalg.norm(X - P, axis=1)
            ok = fc.inside(uv, X, P, n, max(self.eps, 1e-5 * self.size), on_trim=True)
            better = ok & (d < best)
            F[better], UV[better], S[better], N[better], best[better] = fi, uv[better], P[better], n[better], \
                d[better]
        return F, UV, S, N, best

    def project(self, Y, F, UV, gap, ghosts=False):
        """Feet of Y, each first on its own face (from its uv), else on another
        face within gap: (face, uv, point, normal, ok). ghosts: a foot on its
        own surface but outside the face's trim, with no face to go on to, is
        kept (ok) and marked: (..., ok, ghost)."""
        m = len(Y)
        F2, UV2 = F.copy(), UV.copy()
        S, N = np.zeros((m, 3)), np.zeros((m, 3))
        ok = np.zeros(m, bool)
        on = np.zeros(m, bool)
        for fi in np.unique(F):
            if fi < 0:
                continue
            idx = np.nonzero(F == fi)[0]
            fc = self.faces[fi]
            uv = fc.E.feet(Y[idx], seed=UV[idx], iters=12)
            P, n, _ = fc.frame(uv)
            UV2[idx], S[idx], N[idx] = uv, P, n
            on[idx], ok[idx] = fc.inside(uv, Y[idx], P, n, self.eps, parts=True)
        own = (F2.copy(), UV2.copy(), S.copy(), N.copy())
        bad = np.nonzero(~ok)[0]
        if len(bad) and len(self.faces) > 1:
            best = np.full(len(bad), np.inf)
            for fi, fc in enumerate(self.faces):
                Yb = Y[bad]
                uv = fc.E.feet(Yb, g=17, iters=16)
                P, n, _ = fc.frame(uv)
                d = np.linalg.norm(Yb - P, axis=1)
                take = fc.inside(uv, Yb, P, n, self.eps) & (d <= gap) & (d < best) & (F[bad] != fi)
                if take.any():
                    k = bad[take]
                    F2[k], UV2[k], S[k], N[k], ok[k] = fi, uv[take], P[take], n[take], True
                    best[take] = d[take]
        if not ghosts:
            return F2, UV2, S, N, ok
        ghost = ~ok & on & (F >= 0)
        if ghost.any():
            F2[ghost], UV2[ghost], S[ghost], N[ghost] = (a[ghost] for a in own)
            ok = ok | ghost
        return F2, UV2, S, N, ok, ghost


def _on_face(tg, fi, Y, seed):
    fc = tg.faces[fi]
    uv = fc.E.feet(Y, seed=seed, iters=12)
    P, n, _ = fc.frame(uv)
    return uv, P, n, fc.inside(uv, Y, P, n, tg.eps)


def _cross_edge(tg, X, F, UV, N, D, hk, F2, UV2, iters=30):
    """Steps from X (face F) along D that land on face F2: (the point where
    the step leaves F, the normal of F2 there, D carried over the edge)."""
    lo, hi = np.zeros(len(X)), np.array(hk, float)
    Slo, Nlo, UVlo = X.copy(), N.copy(), UV.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Y = X + mid[:, None] * D
        for fi in np.unique(F):
            k = np.nonzero(F == fi)[0]
            uv, P, n, ok = _on_face(tg, fi, Y[k], UVlo[k])
            good, badk = k[ok], k[~ok]
            lo[good], Slo[good], Nlo[good], UVlo[good] = mid[good], P[ok], n[ok], uv[ok]
            hi[badk] = mid[badk]
    Dl = _transport(D, N, Nlo)
    N2 = np.zeros_like(N)
    for fi in np.unique(F2):
        k = np.nonzero(F2 == fi)[0]
        uv, P, n, _ = _on_face(tg, fi, Slo[k], UV2[k])
        N2[k] = n
    return Slo, N2, _transport(Dl, Nlo, N2)


def _uv_on(tg, F, X, seed):
    out = np.array(seed, float)
    for fi in np.unique(F):
        k = np.nonzero(F == fi)[0]
        out[k] = tg.faces[fi].E.feet(X[k], seed=seed[k], iters=12)
    return out


Target.cross_edge = _cross_edge
Target.uv_on = _uv_on


def march(tg, X, F, UV, N, D, dist, steps):
    """Walk the points X (on faces F at UV, normals N) along the surface in
    the tangent directions D for the length dist. Returns (X, F, UV, N, D,
    alive): alive is False for a walk that ran off the surface (its arrays
    hold the last point on it)."""
    X, F, UV, N, D = X.copy(), F.copy(), UV.copy(), N.copy(), D.copy()
    alive = F >= 0
    ghost = np.zeros(len(X), bool)      # walking over a free trim's cut-away part (must end on the face)
    if dist <= 0.0:
        return X, F, UV, N, D, alive
    steps = max(1, int(steps))
    h = dist / steps
    walked = np.zeros(len(X))
    for k in range(steps + 1):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        hk = np.full(len(idx), h) if k < steps else dist - walked[idx]
        go = np.abs(hk) > 1e-14 * max(dist, 1e-300)
        idx, hk = idx[go], hk[go]
        if not len(idx):
            break
        Y = X[idx] + hk[:, None] * D[idx]
        F2, UV2, S, N2, ok, gh = tg.project(Y, F[idx], UV[idx], gap=max(abs(h), tg.eps), ghosts=True)
        D2 = _transport(D[idx], N[idx], N2)
        cross = np.nonzero(ok & (F2 != F[idx]))[0]
        if len(cross):
            # onto another face: the step ends where it leaves this one (exact at a crease)
            Sx, Nx, Dx = tg.cross_edge(X[idx[cross]], F[idx[cross]], UV[idx[cross]], N[idx[cross]],
                                       D[idx[cross]], hk[cross], F2[cross], UV2[cross])
            S[cross], N2[cross], D2[cross] = Sx, Nx, Dx
            UV2[cross] = tg.uv_on(F2[cross], Sx, UV2[cross])
        walked_new = walked[idx] + np.linalg.norm(S - X[idx], axis=1) * np.sign(hk)
        g = idx[ok]
        X[g], F[g], UV[g], N[g], D[g], walked[g] = S[ok], F2[ok], UV2[ok], N2[ok], D2[ok], walked_new[ok]
        ghost[g] = gh[ok]
        alive[idx[~ok]] = False
    alive &= ~ghost
    return X, F, UV, N, D, alive


def _steps(dist, size):
    n = int(np.ceil(abs(dist) / (0.004 * size)))
    lo, hi = (6, 40) if _FAST[0] else (24, 400)
    return min(hi, max(lo, n))


# --------------------------------------------------------------------------
# One chain
# --------------------------------------------------------------------------

class _Base:
    """A segment of input spans (smooth between corners) evaluated at global
    parameters g (span index + fraction)."""

    def __init__(self, spans):
        self.spans = spans

    def eval(self, g):
        P, T = [], []
        n = len(self.spans)
        for x in np.atleast_1d(g):
            i = min(int(np.floor(x)), n - 1)
            p, dp = curvetrim.eval_d(self.spans[i], float(min(1.0, max(0.0, x - i))))
            P.append(p)
            T.append(_unit(dp))
        return np.array(P, float), np.array(T, float)

    def params(self, total):
        """Sample parameters: at least PER_SPAN per span, more on long spans."""
        n = len(self.spans)
        L = []
        for s in self.spans:
            Q = curvejoin.eval_span(s, np.linspace(0, 1, 9))
            L.append(float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()))
        Ltot = sum(L) or 1.0
        per = 4 if _FAST[0] else PER_SPAN
        out = []
        for i, Li in enumerate(L):
            k = max(per, int(np.ceil(total * Li / Ltot)))
            out.append(i + np.linspace(0.0, 1.0, k + 1)[(1 if i else 0):])
        g = np.concatenate(out)
        if len(g) > MAX_SAMPLES:
            g = np.interp(np.linspace(0, len(g) - 1, MAX_SAMPLES), np.arange(len(g)), g)
        return g


class _Run:
    """Offset points with what goes along: Q (on the surface), F/UV/N there,
    D (the walk's direction at the end), C the base points, T the offset's
    tangent, g the base parameters."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def take(self, sel):
        return _Run(**{k: (v[sel] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(self.Q) else v)
                       for k, v in self.__dict__.items()})


def _offset_at(tg, base, g, s, dist, info, seed=None):
    """Walk from the base points at parameters g: (run, alive)."""
    C0, T0 = base.eval(g)
    F, UV, C, N, gap = tg.feet_all(C0)
    on = F >= 0
    if on.any():
        info["gap"] = max(info["gap"], float(gap[on].max()))
    t = _unit_rows(_tangent_part(T0, N))
    b = np.cross(N, t) * s
    Q, F2, UV2, N2, D2, alive = march(tg, C, F, UV, N, b, abs(dist), _steps(dist, tg.size))
    if dist < 0:
        pass
    alive &= on
    Tq = _unit_rows(np.cross(D2, N2) * s)       # along the base's direction (Gauss: square to the walk)
    run = _Run(Q=Q, F=F2, UV=UV2, N=N2, D=D2, C=C, T=Tq, g=np.asarray(g, float), Cn=N, Ct=t,
               K=np.zeros(len(Q), int))
    return run, alive


def _refine_cut(tg, base, ga, gb, s, dist, info, iters=16):
    """The last point of a walk that stays on the surface between a base
    parameter ga (walk alive) and gb (walk dead): bisection."""
    a, b = float(ga), float(gb)
    best = None
    for _ in range(iters):
        m = 0.5 * (a + b)
        run, alive = _offset_at(tg, base, np.array([m]), s, dist, info)
        if alive[0]:
            a, best = m, run
        else:
            b = m
    return best


KINK_COS = np.cos(np.radians(3.0))     # neighbouring offset points on two faces this far apart: a crease


def _kinks(tg, base, run, alive, s, dist, info, iters=14):
    """The run with the points where the offset crosses a crease between two
    faces put in (found by bisection on the base, marked as corners)."""
    if _FAST[0] or len(tg.faces) < 2:
        return run, alive
    i_s = [i for i in range(len(run.Q) - 1) if alive[i] and alive[i + 1] and run.F[i] != run.F[i + 1]
           and float(run.N[i] @ run.N[i + 1]) < KINK_COS]
    if not i_s:
        return run, alive
    parts, alv, last = [], [], 0
    for i in i_s:
        a, b, fa = float(run.g[i]), float(run.g[i + 1]), run.F[i]
        ra = rb = None
        for _ in range(iters):
            m = 0.5 * (a + b)
            r, al = _offset_at(tg, base, np.array([m]), s, dist, info)
            if not al[0]:
                break
            if r.F[0] == fa:
                a, ra = m, r
            else:
                b, rb = m, r
        if ra is None or rb is None:
            continue
        k = ra.take(np.array([0]))
        k.Q = 0.5 * (ra.Q + rb.Q)
        k.T = _unit_rows(rb.Q - ra.Q) if np.linalg.norm(rb.Q - ra.Q) > 0 else ra.T
        k.K = np.array([2])
        parts += [run.take(np.arange(last, i + 1)), k]
        alv += [alive[last:i + 1], np.array([True])]
        last = i + 1
        info["creases"] += 1
    parts.append(run.take(np.arange(last, len(run.Q))))
    alv.append(alive[last:])
    return _concat(parts), np.concatenate(alv)


def _segment_runs(tg, base, s, dist, info):
    """The offset of one smooth segment: [(run, starts_cut, ends_cut)] (a
    walk running off the surface cuts it into runs)."""
    g = base.params(120 if _FAST[0] else 600)
    run, alive = _offset_at(tg, base, g, s, dist, info)
    run, alive = _kinks(tg, base, run, alive, s, dist, info)
    g = run.g
    if alive.all():
        return [(run, False, False)]
    out = []
    idx = np.nonzero(alive)[0]
    if not len(idx):
        info["cuts"] += 1
        return []
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    for a, b in zip(starts, ends):
        sel = np.arange(a, b + 1)
        piece = run.take(sel)
        cut0, cut1 = a > 0, b < len(g) - 1
        if not _FAST[0]:
            parts = []
            if cut0:
                r0 = _refine_cut(tg, base, g[a], g[a - 1], s, dist, info)
                if r0 is not None:
                    parts.append(r0)
            parts.append(piece)
            if cut1:
                r1 = _refine_cut(tg, base, g[b], g[b + 1], s, dist, info)
                if r1 is not None:
                    parts.append(r1)
            if len(parts) > 1:
                piece = _concat(parts)
        if len(piece.Q) >= 2:
            out.append((piece, cut0, cut1))
        info["cuts"] += int(cut0) + int(cut1)
    return out


def _concat(runs):
    keys = [k for k, v in runs[0].__dict__.items() if isinstance(v, np.ndarray) and v.ndim]
    return _Run(**{k: np.concatenate([getattr(r, k) for r in runs]) for k in keys})


# -- corners ---------------------------------------------------------------

def _point_frame(tg, P):
    F, UV, S, N, _ = tg.feet_all(np.atleast_2d(P))
    return F, UV, S, N


def _fan(tg, pa, pb, s, dist, info):
    """Round: the geodesic circle about the corner between the end of piece
    a and the start of piece b (their walk directions at the corner)."""
    F, UV, C, N = _point_frame(tg, pa.C[-1])
    if F[0] < 0:
        return None
    da = _unit(_tangent_part(np.atleast_2d(np.cross(pa.Cn[-1], pa.Ct[-1]) * s), N)[0])
    db = _unit(_tangent_part(np.atleast_2d(np.cross(pb.Cn[0], pb.Ct[0]) * s), N)[0])
    ang = float(np.arctan2(np.cross(da, db) @ N[0], da @ db))
    if abs(ang) < 1e-6:
        return None
    if _FAST[0]:
        k = max(2, int(np.ceil(abs(ang) / np.radians(15))))
    else:
        tol = max(1e-9, info["tol"])
        step = 2.0 * np.arccos(max(0.0, 1.0 - 0.1 * tol / max(abs(dist), 1e-12)))
        k = int(min(180, max(4, np.ceil(abs(ang) / max(step, np.radians(0.5))))))
    angs = np.linspace(0.0, ang, k + 1)[1:-1]
    if not len(angs):
        return None
    n0 = N[0]
    e2 = np.cross(n0, da)
    Dirs = np.cos(angs)[:, None] * da + np.sin(angs)[:, None] * e2
    m = len(angs)
    Q, F2, UV2, N2, D2, alive = march(tg, np.repeat(C, m, 0), np.repeat(F, m), np.repeat(UV, m, 0),
                                      np.repeat(N, m, 0), Dirs, abs(dist), _steps(dist, tg.size))
    if not alive.all():
        return None
    Tq = _unit_rows(np.cross(D2, N2) * s)
    return _Run(Q=Q, F=F2, UV=UV2, N=N2, D=D2, C=np.repeat(C, m, 0), T=Tq, g=np.full(m, np.nan),
                Cn=np.repeat(N, m, 0), Ct=np.repeat(np.atleast_2d(pa.Ct[-1]), m, 0), K=np.zeros(m, int))


def _ray(tg, r, i, sign, length):
    """Polyline of a walk from offset point i of run r along its tangent (sign: +1
    forward, -1 backward) for the length."""
    n = 64 if not _FAST[0] else 16
    X = r.Q[i:i + 1]
    F, UV, N, D = r.F[i:i + 1], r.UV[i:i + 1], r.N[i:i + 1], r.T[i:i + 1] * sign
    pts = [X[0]]
    h = length / n
    for _ in range(n):
        X, F, UV, N, D, alive = march(tg, X, F, UV, N, D, h, 1)
        if not alive[0]:
            break
        pts.append(X[0])
    return np.array(pts)


def _chart(N):
    a = np.array([1.0, 0, 0]) if abs(N[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(np.cross(N, a))
    return e1, np.cross(N, e1)


def _first_crossing(A, B, e1, e2):
    """First crossing of polylines A and B (2D in the chart): (i, sa, j, sb) or None."""
    a2 = np.column_stack([A @ e1, A @ e2])
    b2 = np.column_stack([B @ e1, B @ e2])
    for i in range(len(a2) - 1):
        hit, sb = curveoffset._seg_cross_2d(a2[i], a2[i + 1], b2[:-1], b2[1:])
        if hit.any():
            j = int(np.argmax(hit))
            # where along a: solve with the roles swapped
            hit2, sa = curveoffset._seg_cross_2d(b2[j], b2[j + 1], a2[i:i + 1], a2[i + 1:i + 2])
            return i, float(sa[0]) if hit2.any() else 0.5, j, float(sb[j])
    return None


def _sharp(tg, pa, pb, s, dist, theta):
    """Sharp: the two offsets continued along the surface until they meet:
    (points to insert between them) or None."""
    if theta > np.radians(165):
        return None
    L = abs(dist) * np.tan(0.5 * theta) * 1.5 + 1e-6 * tg.size
    A = _ray(tg, pa, len(pa.Q) - 1, 1.0, L)
    B = _ray(tg, pb, 0, -1.0, L)
    if len(A) < 2 or len(B) < 2:
        return None
    e1, e2 = _chart(pa.N[-1])
    hit = _first_crossing(A, B, e1, e2)
    if hit is None:
        return None
    i, sa, j, sb = hit
    X = A[i] + sa * (A[i + 1] - A[i])
    return np.vstack([A[1:i + 1], X[None], B[1:j + 1][::-1]]), i         # (points, index of X in them)


def _chamfer(tg, pa, pb):
    """Chamfer: the chord between the two ends, pulled onto the surface."""
    a, b = pa.Q[-1], pb.Q[0]
    ts = np.linspace(0, 1, 9 if not _FAST[0] else 4)[1:-1]
    Y = a[None] + ts[:, None] * (b - a)[None]
    F, UV, S, N, ok = tg.project(Y, np.repeat(pa.F[-1:], len(Y)), np.repeat(pa.UV[-1:], len(Y), 0),
                                 gap=float(np.linalg.norm(b - a)))
    return S if ok.all() else Y


# --------------------------------------------------------------------------
# A polyline with corners and folds
# --------------------------------------------------------------------------

class _Poly:
    """Points Q with normals N, tangents T and base points C; brk per point
    (0 smooth, 1 a G1 joint, 2 a corner) and force per segment (0 judge by
    the base, 1 never a fold, 2 always one). pending: (force, brk) for the
    segment that joins the next block, and that block's first point."""

    def __init__(self):
        self.Q, self.N, self.T, self.C, self.brk, self.force = [], [], [], [], [], []
        self.pending = (0, 0)

    def add(self, Q, N, T, C, brk_first=0, brk_last=0, force=0, K=None):
        Q = np.atleast_2d(Q)
        m = len(Q)
        if not m:
            return
        jf, jb = self.pending
        self.pending = (0, 0)
        if self.Q:
            self.force.append(jf)
            brk_first = max(brk_first, jb)
        b = [0] * m if K is None else [int(x) for x in K]
        b[0] = max(b[0], brk_first)
        b[-1] = max(b[-1], brk_last)
        self.Q += list(Q)
        self.N += list(np.atleast_2d(N))
        self.T += list(np.atleast_2d(T))
        self.C += list(np.atleast_2d(C))
        self.brk += b
        self.force += [force] * (m - 1)

    def extend(self, other):
        """other appended (joined by pending)."""
        if not other.Q:
            return
        jf, jb = self.pending
        self.pending = (0, 0)
        if self.Q:
            self.force.append(jf)
        first = len(self.Q)
        self.Q += other.Q
        self.N += other.N
        self.T += other.T
        self.C += other.C
        self.brk += other.brk
        self.force += other.force
        if first:
            self.brk[first] = max(self.brk[first], jb)

    def arrays(self):
        return (np.array(self.Q, float), np.array(self.N, float), np.array(self.T, float),
                np.array(self.C, float), np.array(self.brk, int), np.array(self.force, int))


def remove_loops(Q, N, T, C, brk, force, reach):
    """Cut out the folds: each run of segments going back against the base is
    the inside of a loop, cut at the crossing that closes it (looked for in
    the tangent-plane chart at the fold, within reach of it). Returns the
    arrays with the crossings as corners, and the counts (folds, unresolved)."""
    folds = bad = 0
    for _ in range(400):
        if len(Q) < 3:
            break
        dq, dc = np.diff(Q, axis=0), np.diff(C, axis=0)
        rev = (_dot(dq, dc) <= 0.0) & (np.linalg.norm(dq, axis=1) > 1e-12 * reach)
        rev[force == 1] = False
        rev[force == 2] = True
        if not rev.any():
            break
        r0 = int(np.argmax(rev))
        r1 = r0
        while r1 + 1 < len(rev) and rev[r1 + 1]:
            r1 += 1
        e1, e2 = _chart(N[r0])
        q2 = np.column_stack([Q @ e1, Q @ e2])
        near = np.linalg.norm(Q - Q[r0], axis=1) <= reach + float(np.linalg.norm(Q[r1 + 1] - Q[r0]))
        nseg = len(Q) - 1
        best = None
        js = np.array([j for j in range(r1 + 1, nseg) if near[j] or near[j + 1]], int)
        if len(js):
            for i in range(r0 - 1, -1, -1):
                if not (near[i] or near[i + 1]):
                    if best is not None or i < r0 - 4 and not near[i + 1]:
                        break
                    continue
                hit, s = curveoffset._seg_cross_2d(q2[i], q2[i + 1], q2[js], q2[js + 1])
                if hit.any():
                    k = int(np.argmax(hit))
                    j = int(js[k])
                    if best is None or j - i < best[0]:
                        best = (j - i, i, j, float(s[k]))
        if best is None:
            bad += 1
            force = force.copy()
            force[r0:r1 + 1] = 1             # bridged: no longer looked at
            continue
        _, i, j, s = best
        X = Q[i] + s * (Q[i + 1] - Q[i])
        Nx = _unit(N[i] + s * (N[i + 1] - N[i]))
        Tx = _unit(Q[j + 1] - X)
        Q = np.vstack([Q[:i + 1], X, Q[j + 1:]])
        N = np.vstack([N[:i + 1], Nx, N[j + 1:]])
        T = np.vstack([T[:i + 1], Tx, T[j + 1:]])
        C = np.vstack([C[:i + 1], C[j][None], C[j + 1:]])
        brk = np.concatenate([brk[:i + 1], [2], brk[j + 1:]])
        force = np.concatenate([force[:i], [force[i]], [force[j]], force[j + 1:]])
        folds += 1
    return (Q, N, T, C, brk, force), folds, bad


def _rotate(arrs, k, closed_seg_force=1):
    """Closed polyline arrays started at point k (the closing segment gets
    closed_seg_force; the last point repeats the first)."""
    Q, N, T, C, brk, force = arrs
    brk = brk.copy()
    if np.linalg.norm(Q[-1] - Q[0]) <= 1e-12 * (np.ptp(Q, axis=0).max() or 1.0):
        # the last point repeats the first: its segment is the closing one, its break the first point's
        brk[0] = max(brk[0], brk[-1])
        Q, N, T, C, brk = Q[:-1], N[:-1], T[:-1], C[:-1], brk[:-1]
    else:
        force = np.concatenate([force, [closed_seg_force]])     # the segment from the last point to the first
    m = len(Q)
    idx = (np.arange(m + 1) + k) % m
    sidx = (np.arange(m) + k) % m
    return Q[idx], N[idx], T[idx], C[idx], brk[idx], force[sidx]


def _fit(Q, T, brk, closed, tol):
    """Spans of the polyline fitted between its breaks."""
    if _FAST[0]:
        return None
    cuts = [0] + [i for i in range(1, len(Q) - 1) if brk[i]] + [len(Q) - 1]
    if closed and len(cuts) == 2:
        pc = curveoffset.fit_polyline(Q, None, None, 3, tol, closed=True)
        return pc["spans"], pc["err"]
    spans, err = [], 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        S = Q[a:b + 1]
        if len(S) < 2:
            continue
        t0 = T[a] if brk[a] == 1 or (a == 0 and not closed) else S[1] - S[0]
        t1 = T[b] if brk[b] == 1 or (b == len(Q) - 1 and not closed) else S[-1] - S[-2]
        if float(np.dot(t0, S[-1] - S[0])) < 0 and len(S) > 2:
            t0 = S[1] - S[0]
        if float(np.dot(t1, S[-1] - S[0])) < 0 and len(S) > 2:
            t1 = S[-1] - S[-2]
        pc = curveoffset.fit_polyline(S, t0, t1, 3, tol)
        spans += pc["spans"]
        err = max(err, pc["err"])
    return spans, err


def _turn(pa, pb, N):
    a = _unit(_tangent_part(np.atleast_2d(pa.Ct[-1]), N)[0])
    b = _unit(_tangent_part(np.atleast_2d(pb.Ct[0]), N)[0])
    return float(np.arctan2(np.cross(a, b) @ N[0], a @ b))


def _closed_side(tg, segs, closed):
    """+1 when the left of the chain (seen from the outward normal) is its
    outside (it turns right in all), else -1; open chains: +1."""
    if not closed:
        return 1.0
    total = 0.0
    prev_t = None
    for base, _ in segs:
        g = base.params(64)
        P, T = base.eval(g)
        F, UV, S, N, _ = tg.feet_all(P)
        t = _unit_rows(_tangent_part(T, N))
        if prev_t is not None:
            total += float(np.arctan2(np.cross(prev_t, t[0]) @ N[0], prev_t @ t[0]))
        a, b = t[:-1], t[1:]
        total += float(np.arctan2(_dot(np.cross(a, b), N[:-1]), _dot(a, b)).sum())
        prev_t = t[-1]
    if prev_t is not None:
        base0 = segs[0][0]
        P, T = base0.eval(np.array([0.0]))
        F, UV, S, N, _ = tg.feet_all(P)
        t0 = _unit(_tangent_part(T, N)[0])
        total += float(np.arctan2(np.cross(prev_t, t0) @ N[0], prev_t @ t0))
    return -1.0 if total > 0 else 1.0


def _fill_corner(tg, poly, pa, pb, s, dist, corner, info):
    """Add what goes between the end of piece a and the start of piece b (at a
    corner of the input) to poly, and set how b joins."""
    Fp, UVp, Cp, Np = _point_frame(tg, pa.C[-1])
    if Fp[0] < 0:
        poly.pending = (1, 2)
        return
    turn = _turn(pa, pb, Np)
    theta = abs(turn)
    if theta < 1e-6:
        return
    if s * turn > 0:                      # the inside of the turn: both overlap, trimmed at their crossing
        info["inside"] += 1
        poly.pending = (2, 0)
        return
    if corner == "SHARP":
        sh = _sharp(tg, pa, pb, s, dist, theta)
        if sh is not None:
            P, ix = sh
            _, _, _, Np2, _ = tg.feet_all(P)
            Tm = _unit_rows(np.vstack([np.diff(P, axis=0), (pb.Q[0] - P[-1])[None]]))
            poly.brk[-1] = max(poly.brk[-1], 1)
            poly.pending = (1, 0)
            poly.add(P, Np2, Tm, np.repeat(np.atleast_2d(pa.C[-1]), len(P), 0), force=1)
            poly.brk[len(poly.Q) - len(P) + ix] = 2
            poly.pending = (1, 1)
            return
    if corner == "CHAMFER":
        P = _chamfer(tg, pa, pb)
        _, _, _, Np2, _ = tg.feet_all(P)
        Tm = _unit_rows(np.repeat((pb.Q[0] - pa.Q[-1])[None], len(P), 0))
        poly.brk[-1] = 2
        poly.pending = (1, 0)
        poly.add(P, Np2, Tm, np.repeat(np.atleast_2d(pa.C[-1]), len(P), 0), force=1)
        poly.pending = (1, 2)
        return
    fill = _fan(tg, pa, pb, s, dist, info)
    poly.brk[-1] = max(poly.brk[-1], 1)
    if fill is not None:
        poly.pending = (1, 0)
        poly.add(fill.Q, fill.N, fill.T, fill.C, force=1)
    poly.pending = (1, 1)


SMOOTH_DEG = 10.0          # a run of straight spans turning less than this at each joint: a sampled curve
SMOOTH_MIN = 6             # spans in such a run, at least


def smooth_polylines(spans, closed, tol):
    """Runs of short straight spans that turn a little at every joint (a
    projected or intersected curve: a polyline through points on the surface)
    replaced by a cubic spline through their points within tol: offsetting
    each straight piece would leave a gap or an overlap at every joint."""
    from . import curvetools
    n = len(spans)
    straight = [curvetools.is_straight(s) for s in spans]
    if sum(straight) < SMOOTH_MIN:
        return spans, 0
    c = np.cos(np.radians(SMOOTH_DEG))

    def soft(i):            # the joint before span i (i - 1 -> i) is a small turn between straight spans
        a, b = curvejoin._tangent(spans[i - 1], 1), curvejoin._tangent(spans[i], 0)
        return straight[i - 1] and straight[i] and float(_unit(a) @ _unit(b)) >= c
    joints = [soft(i) for i in range(n)]            # joints[0]: the closing joint (closed chains)
    if closed and all(straight) and all(joints):
        V = np.array([curvejoin._cart(s)[0] for s in spans])
        pc = curveoffset.fit_polyline(V, None, None, 3, tol, closed=True)
        return pc["spans"], 1
    out, i, made = [], 0, 0
    while i < n:
        j = i
        while straight[i] and j + 1 < n and joints[j + 1]:
            j += 1
        if straight[i] and j - i + 1 >= SMOOTH_MIN:
            run = spans[i:j + 1]
            V = np.array([curvejoin._cart(s)[0] for s in run] + [curvejoin._cart(run[-1])[-1]])
            pc = curveoffset.fit_polyline(V, V[1] - V[0], V[-1] - V[-2], 3, tol)
            out += pc["spans"]
            made += 1
        else:
            out += spans[i:j + 1]
        i = j + 1
    return out, made


def offset_chain(tg, spans, closed, side, dist, corner, info):
    """[(Q polyline, closed, (spans, err) or None)] of one chain offset to one
    side (side: +1 the positive side)."""
    raw, cornered = curveoffset._segments(spans, closed)
    segs = [(_Base(sp), c) for sp, c in raw]
    s = side * _closed_side(tg, segs, closed)
    pieces = [_segment_runs(tg, base, s, dist, info) for base, _ in segs]
    n = len(segs)
    out = []                 # finished polylines (not closed)
    poly = _Poly()
    first_poly = None        # a closed chain cut somewhere: what ran from its start, joined at the end
    started_clean = bool(pieces[0]) and not pieces[0][0][1]

    def flush():
        nonlocal poly, first_poly
        if poly.Q:
            if closed and first_poly is None and started_clean and not out:
                first_poly = poly
            else:
                out.append(poly)
        poly = _Poly()

    for k in range(n):
        runs = pieces[k]
        if not runs:
            flush()
        for run, cut0, cut1 in runs:
            if cut0:
                flush()
            poly.add(run.Q, run.N, run.T, run.C, brk_first=2 if cut0 else 0, brk_last=2 if cut1 else 0,
                     K=run.K)
            if cut1:
                flush()
        if k == n - 1 and not closed:
            break
        nxt = pieces[(k + 1) % n]
        if not runs or not nxt or runs[-1][2] or nxt[0][1] or not poly.Q:
            flush()
            continue
        if cornered:
            _fill_corner(tg, poly, runs[-1][0], nxt[0][0], s, dist, corner, info)
    whole = closed and first_poly is None and not out and bool(poly.Q)
    result = []
    if whole:
        # closed and never cut: close the loop on the first point (a smooth closed curve's
        # samples hold its seam twice: the last one goes, its offset may differ a little)
        r0 = pieces[0][0][0]
        if not cornered and len(poly.Q) > 2 and np.linalg.norm(poly.C[-1] - poly.C[0]) <= 1e-9 * tg.size:
            for a in (poly.Q, poly.N, poly.T, poly.C, poly.brk, poly.force):
                a.pop()
        poly.add(r0.Q[:1], r0.N[:1], r0.T[:1], r0.C[:1])
        result += _finish(poly, True, dist, info)
    else:
        if first_poly is not None:
            if poly.Q:
                if len(poly.Q) > 1 and np.linalg.norm(poly.C[-1] - first_poly.C[0]) <= 1e-9 * tg.size:
                    for a in (poly.Q, poly.N, poly.T, poly.C, poly.brk, poly.force):
                        a.pop()                  # the seam, sampled twice
                poly.extend(first_poly)          # across the start of the closed chain
            else:
                poly = first_poly
        if poly.Q:
            out.append(poly)
        for pl in out:
            result += _finish(pl, False, dist, info)
    return result


def _finish(poly, closed, dist, info):
    if not poly.Q:
        return []
    arrs = poly.arrays()
    Q = arrs[0]
    if len(Q) < 2:
        return []
    reach = 4.0 * abs(dist)
    if closed:
        # start the loop search where nothing folds: in the middle of the longest good run
        Qa, Na, Ta, Ca, brka, forcea = arrs
        dqa = np.diff(Qa, axis=0)
        rev = (_dot(dqa, np.diff(Ca, axis=0)) <= 0) & (np.linalg.norm(dqa, axis=1) > 1e-12 * reach)
        rev[forcea == 1] = False
        rev[forcea == 2] = True
        k = 0
        if rev.any():
            good = ~rev
            best, cur, start = (0, 0), 0, 0
            for i, gd in enumerate(good):
                if gd:
                    if cur == 0:
                        start = i
                    cur += 1
                    if cur > best[0]:
                        best = (cur, start)
                else:
                    cur = 0
            k = best[1] + best[0] // 2
        elif brka.any():
            k = int(np.argmax(brka > 0))
        arrs = _rotate(arrs, k)
    (Q, N, T, C, brk, force), folds, bad = remove_loops(*arrs, reach)
    info["folds"] += folds
    info["unresolved"] += bad
    if closed and brk.any() and not brk[0]:
        k = int(np.argmax(brk > 0))
        arrs = _rotate((Q, N, T, C, brk, force), k)
        Q, N, T, C, brk, force = arrs
    if closed:
        brk[0] = brk[-1] = max(brk[0], brk[-1])
    fit = _fit(Q, T, brk, closed, info["tol"])
    if fit is not None:
        info["err"] = max(info["err"], fit[1])
    return [(Q, closed, fit)]


# --------------------------------------------------------------------------
# Inputs, results
# --------------------------------------------------------------------------

def _new_info(tol=1e-4):
    return {"chains": 0, "folds": 0, "unresolved": 0, "cuts": 0, "inside": 0, "gap": 0.0, "err": 0.0,
            "tol": tol, "nothing": 0, "creases": 0, "smoothed": 0}


def prepare(tg, pieces, tol):
    """[(spans, closed, polylines smoothed)]: the inputs chained where their
    ends touch, sampled polylines made smooth splines."""
    out = []
    for spans, _, closed, _ in curvejoin.chain_pieces(pieces, max(1e-5 * tg.size, 1e-6)):
        spans, made = smooth_polylines(spans, closed, 0.25 * tol)
        out.append((spans, closed, made))
    return out


def offset_polylines(tg, pieces, p, info, chains=None):
    """[(Q, closed, fit)] of every chain and side."""
    d0 = float(p.get("distance", 0.1))
    if abs(d0) < 1e-12:
        raise ValueError("Distance is 0")
    sign = -1.0 if p.get("flip") else 1.0
    sides = (1.0, -1.0) if p.get("both_sides") else (1.0,)
    corner = p.get("os_corner", "SHARP")
    if chains is None:
        chains = prepare(tg, pieces, info["tol"])
    out = []
    for spans, closed, made in chains:
        info["smoothed"] += made
        info["chains"] += 1
        for sd in sides:
            side = sign * sd * (1.0 if d0 > 0 else -1.0)
            res = offset_chain(tg, spans, closed, side, abs(d0), corner, info)
            if not res:
                info["nothing"] += 1
            out += res
    return out


def offset_on_surface(tg, pieces, p):
    """Spline specs (world) of the offset curves."""
    tol = float(p.get("tolerance", 1e-4))
    info = _new_info(tol)
    specs = []
    for Q, closed, fit in offset_polylines(tg, pieces, p, info):
        if fit is None:
            continue
        spans, _ = fit
        if spans:
            specs.append(curvejoin.spans_to_spec(spans, simplify=True))
    LAST_INFO.clear()
    LAST_INFO.update(info)
    if not specs:
        raise ValueError("the offset runs off the surface everywhere: a smaller distance, or the other side")
    return specs


def preview_polylines(tg, chains, p):
    """World polylines of the offset, quick enough to follow the mouse
    (chains: prepare's)."""
    if abs(float(p.get("distance", 0.1))) < 1e-12:
        return []
    _FAST[0] = True
    try:
        return [Q for Q, _, _ in offset_polylines(tg, None, p, _new_info(), chains=chains)]
    finally:
        _FAST[0] = False


def arrow(tg, chains, p, at):
    """(base point, unit direction on the surface a positive distance walks,
    walk polyline to the current distance) at the input point nearest at."""
    best = None
    for spans, closed, _ in chains:
        raw, _ = curveoffset._segments(spans, closed)
        segs = [(_Base(sp), c) for sp, c in raw]
        s = _closed_side(tg, segs, closed) * (-1.0 if p.get("flip") else 1.0)
        for base, _ in segs:
            g = base.params(48)
            P, _ = base.eval(g)
            k = int(np.argmin(np.linalg.norm(P - np.asarray(at, float), axis=1)))
            d = float(np.linalg.norm(P[k] - at))
            if best is None or d < best[0]:
                best = (d, base, g[k], s)
    if best is None:
        return None
    _, base, gk, s = best
    C0, T0 = base.eval(np.array([gk]))
    F, UV, C, N, _ = tg.feet_all(C0)
    if F[0] < 0:
        return None
    t = _unit_rows(_tangent_part(T0, N))
    b = np.cross(N, t) * s
    dist = float(p.get("distance", 0.0))
    D = b * (1.0 if dist >= 0 else -1.0)
    pts = [C[0]]
    X, Fx, UVx, Nx, Dx = C, F, UV, N, D
    n = 24
    for _ in range(n if abs(dist) > 0 else 0):
        X, Fx, UVx, Nx, Dx, alive = march(tg, X, Fx, UVx, Nx, Dx, abs(dist) / n, 1)
        if not alive[0]:
            break
        pts.append(X[0])
    return C[0], b[0], np.array(pts)


def target_of(objs):
    from .sweeps import _world_brep
    return Target([_world_brep(o) for o in objs])


@features.builder("offset_on_surface")
def build_offset_on_surface(inputs, p):
    curves = [(o, r) for o, r in inputs if r != "SURF"]
    surfs = [o for o, r in inputs if r == "SURF"]
    if not curves or not surfs:
        raise ValueError("needs curves and the surface they lie on")
    tg = target_of(surfs)
    pieces = [curvejoin.input_spans(o, r) for o, r in curves]
    return ("splines", offset_on_surface(tg, pieces, p))


def report_text(info):
    msg = f"Offset {info['chains']} curve(s) on the surface"
    notes = []
    if info["err"] > 0:
        notes.append(f"fit within {info['err']:.2g}")
    if info["cuts"]:
        notes.append(f"cut where it runs off the surface ({info['cuts']}x)")
    if info["folds"]:
        notes.append(f"{info['folds']} fold(s) cut out")
    if info["gap"] > 1e-6:
        notes.append(f"input pulled onto the surface (up to {info['gap']:.2g} off)")
    return msg + (" · " + ", ".join(notes) if notes else "")


def warnings(info):
    out = []
    if info["unresolved"]:
        out.append(f"{info['unresolved']} fold(s) could not be cut cleanly")
    if info["nothing"]:
        out.append(f"{info['nothing']} side(s) ran off the surface completely")
    return out


# --------------------------------------------------------------------------
# Which surfaces the curves lie on
# --------------------------------------------------------------------------

def _samples(pieces, per=8):
    return np.vstack([curvetrim.sample_spans(sp, per)[0] for sp in pieces])


def built_from(obj, sources):
    """Whether obj's feature takes one of the source objects as an input (a
    flange grown from a curve lies against it, but is no surface under it)."""
    try:
        h = features.history(obj)
    except Exception:
        return False
    names = {s.name for s in sources}
    return bool(h) and any(getattr(o, "name", o) in names for o, _ in h.get("inputs", []))


def find_targets(pieces, candidates, rel=2e-3, sources=()):
    """The candidate NURBS objects the curves lie on: each within rel of its
    size of some of the curves' points (the nearest one if none is). Objects
    built from the sources (the curves' objects) are not candidates."""
    candidates = [o for o in candidates if not built_from(o, sources)]
    P = _samples(pieces)
    lo, hi = P.min(axis=0), P.max(axis=0)
    hits, nearest = [], None
    for o in candidates:
        try:
            tg = target_of([o])
        except Exception:
            continue
        bb = np.array([o.matrix_world @ v.co for v in o.data.vertices]) if len(o.data.vertices) else None
        if bb is not None:
            pad = 0.05 * tg.size
            if (bb.min(axis=0) - pad > hi).any() or (bb.max(axis=0) + pad < lo).any():
                continue
        _, _, _, _, d = tg.feet_all(P)
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        near = d <= rel * tg.size
        if near.mean() >= 0.25:
            hits.append(o)
        if nearest is None or float(np.median(d)) < nearest[0]:
            nearest = (float(np.median(d)), o)
    if not hits and nearest is not None and nearest[0] <= 0.05 * max(float(np.ptp(P, axis=0).max()), 1e-9):
        hits = [nearest[1]]
    return hits


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import json  # noqa: E402

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty  # noqa: E402

from . import hints  # noqa: E402
from .ops import AskValue, PickMixin, _decode_inputs, _encode_inputs, _kernel_ok, _select_only  # noqa: E402
from .valuedrag import BASE_PX, ValueDrag  # noqa: E402

CORNER_ITEMS = [("SHARP", "Sharp", "Continue both offsets along the surface until they meet"),
                ("ROUND", "Round", "Fill the corner with a circle on the surface about the original corner"),
                ("CHAMFER", "Chamfer", "Connect the two offsets with a line on the surface")]
CORNERS = ("SHARP", "ROUND", "CHAMFER")


class NPROTO_OT_offset_on_surface(ValueDrag, AskValue, PickMixin, bpy.types.Operator):
    """Offset curves on a surface, by a distance measured along it: each point
    walks square to the curve on the surface, across joined faces, and the
    offset ends where it runs off the surface. Curves on the surface (drawn,
    projected, intersections) or surface edges. Select the curves (and the
    surface, or it is found), or click curves / edges and Enter; then drag
    the arrow, type a value, click away or Enter"""
    bl_idname = "nproto.offset_on_surface"
    bl_label = "Offset on Surface"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 1
    pick_max = 256
    finish_on_min = False
    start_value = 0.0           # the arrow on the curve: nothing until it is dragged (L: the last distance)
    fit_view = "shrink"

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    surfaces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    distance: FloatProperty(name="Distance", default=0.1, unit='LENGTH', soft_min=-10.0, soft_max=10.0,
                            description="Measured along the surface. Positive: outward on closed curves, to the "
                                        "left (seen from the surface's outside) on open ones")
    os_corner: EnumProperty(name="Corners", items=CORNER_ITEMS, default="SHARP")
    flip: BoolProperty(name="Flip", default=False, description="Offset to the other side")
    both_sides: BoolProperty(name="Both Sides", default=False, description="Offset to both sides")
    tolerance: FloatProperty(name="Tolerance", default=1e-4, min=1e-7, soft_max=0.01, precision=6, unit='LENGTH',
                             description="Largest deviation of the fitted offset from the surface walk")

    # -- inputs ---------------------------------------------------------------
    def pick_state(self):
        n = len(getattr(self, "_picks", []))
        return "click curves on a surface, or its edges" + (f" · {n} picked, Enter to offset" if n else "")

    def _curve_inputs(self):
        return _decode_inputs(self.inputs)

    def _targets(self, context, curves):
        if self.surfaces:
            objs = [bpy.data.objects[n] for n in json.loads(self.surfaces) if n in bpy.data.objects]
            if objs:
                return objs
        own = []
        for o, r in curves:
            if o.type != 'CURVE' and store.is_nurbs(o) and o not in own:
                own.append(o)
        sel = [o for o in context.selected_objects if o.type != 'CURVE' and store.is_nurbs(o) and o not in own]
        objs = own + sel
        if not objs or any(o.type == 'CURVE' for o, _ in curves) and not sel:
            pieces = [curvejoin.input_spans(o, r) for o, r in curves if o.type == 'CURVE']
            if pieces:
                cands = [o for o in context.visible_objects if o.type != 'CURVE' and store.is_nurbs(o)
                         and o not in objs]
                objs += find_targets(pieces, cands, sources=[o for o, r in curves if o.type == 'CURVE'])
        self.surfaces = json.dumps([o.name for o in objs])
        return objs

    def invoke(self, context, event):
        from . import curvetrim as _ct
        if not _kernel_ok(self):
            return {'CANCELLED'}
        _ct._object_mode(context)
        self.surfaces = ""
        if not self.inputs:
            crv = [o for o in context.selected_objects if o.type == 'CURVE' and len(o.data.splines)]
            if crv and not event.alt:
                self.inputs = _encode_inputs([(o, f"S{i}") for o in crv for i in range(len(o.data.splines))])
            else:
                return self.start_picking(context, event)
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        if not self._targets(context, self._curve_inputs()):
            self.report({'WARNING'}, "No surface under the curves: select the surface they lie on too")
            return {'CANCELLED'}
        return self.ask(context)

    # -- the live value stage (valuedrag.py) ----------------------------------
    def drag_setup(self, context, mouse):
        curves = self._curve_inputs()
        objs = self._targets(context, curves)
        if not curves or not objs:
            return False
        self._tg = target_of(objs)
        self._pieces = [curvejoin.input_spans(o, r) for o, r in curves]
        self._chains = prepare(self._tg, self._pieces, self.tolerance)
        P = _samples([sp for sp, _, _ in self._chains], 16)
        self._pts = P
        self._anchor = P[len(P) // 2]
        self._base_at_mouse = False
        if mouse is not None:
            best = None
            for q in P:
                sq = self._screen(context, q)
                if sq is not None and (sq - mouse).length <= BASE_PX and (best is None or (sq - mouse).length < best[0]):
                    best = ((sq - mouse).length, q)
            if best is not None:
                self._anchor, self._base_at_mouse = best[1], True
        self._cache = None
        return True

    def drag_on_input(self, context, mouse):
        return self._near_screen(context, self._pts, mouse)

    def _p(self):
        return {"distance": self.distance, "os_corner": self.os_corner, "flip": self.flip,
                "both_sides": self.both_sides, "tolerance": self.tolerance}

    def drag_handles(self):
        a = arrow(self._tg, self._chains, self._p(), self._anchor)
        if a is None:
            B, v, path = self._anchor, np.array([0.0, 0.0, 1.0]), None
        else:
            B, v, path = a
        h = {"field": "distance", "kind": "arrow", "origin": B, "dir": v, "param": self.distance, "base": B,
             "to_value": lambda t: t}
        if path is not None and len(path) >= 2:
            h["pos"], h["path"], h["head_from"] = path[-1], path, path[-2]
        return [h]

    def drag_preview(self):
        p = self._p()
        key = tuple(sorted(p.items()))
        if self._cache is None or self._cache[0] != key:
            try:
                lines = preview_polylines(self._tg, self._chains, p)
            except Exception as exc:        # a distance the shape can't take: nothing to show
                print("NURBS offset on surface preview:", exc)
                lines = []
            self._cache = (key, lines)
        return {"lines": self._cache[1], "tris": np.zeros((0, 3))}

    def drag_check(self):
        return "" if abs(self.distance) > 1e-12 else "Distance 0: drag the arrow out, or type a value"

    def drag_key(self, context, event, ch):
        if ch == "f":
            self.flip = not self.flip
        elif ch == "c":
            self.os_corner = CORNERS[(CORNERS.index(self.os_corner) + 1) % len(CORNERS)]
        elif ch == "b":
            self.both_sides = not self.both_sides
        else:
            return False
        return True

    def drag_rows(self):
        return [("Side", [hints.key("F", "the other side"), hints.toggle("B", "both sides", self.both_sides),
                          hints.note("or drag the arrow across the curve")]),
                ("Corners", [hints.cycle("C", ["sharp", "round", "chamfer"], self.os_corner.lower())])]

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "distance")
        lay.row().prop(self, "os_corner", expand=True)
        lay.prop(self, "flip")
        lay.prop(self, "both_sides")
        lay.prop(self, "tolerance")

    def execute(self, context):
        from .primitives import write_spline_object
        try:
            curves = self._curve_inputs()
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if not curves:
            self.report({'WARNING'}, "Select curves on a surface, or pick its edges")
            return {'CANCELLED'}
        objs = self._targets(context, curves)
        if not objs:
            self.report({'WARNING'}, "No surface under the curves: select the surface they lie on too")
            return {'CANCELLED'}
        inputs = list(curves) + [(o, "SURF") for o in objs]
        params = self._p()
        try:
            g = features.run("offset_on_surface", inputs, params)
        except Exception as exc:
            self.report({'ERROR'}, f"Offset on Surface: {exc}")
            return {'CANCELLED'}
        first = curves[0][0]
        name = first.name + "_srf_offset"
        cu = bpy.data.curves.new(name, 'CURVE')
        out = bpy.data.objects.new(name, cu)
        (context.collection or context.scene.collection).objects.link(out)
        write_spline_object(out, g[1])
        if features.history_enabled(context):
            features.record(out, "offset_on_surface", inputs, params)
        _select_only(context, [out])
        info = dict(LAST_INFO)
        warn = warnings(info)
        for w in warn:
            self.report({'WARNING'}, w)
        if not warn:
            self.report({'INFO'}, report_text(info))
        return {'FINISHED'}


classes = (NPROTO_OT_offset_on_surface,)
