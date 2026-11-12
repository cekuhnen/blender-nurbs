"""Rebuild Curve / Extend Curve.

Rebuild Curve refits a curve as ONE clean uniform spline: chosen degree,
evenly spread CVs (non-rational, no internal knot stacks), the shape held
within Tolerance. Open curves keep their end points exactly and, with Keep
End Tangents, their end directions (so a rebuilt piece stays G1 to whatever
it met). Closed curves become periodic splines (smooth across the seam).
The fit is least squares on dense samples of the exact curve with parameter
correction (samples projected onto the fit, then refit), plus a tiny fairing
term that keeps the CV spacing even. CVs 0 picks the fewest CVs within
Tolerance.

Extend Curve lengthens an open curve at one or both ends, exactly keeping the
original part:
  Natural -- the end span continued as the same polynomial (a rational arc
             continues as the same circle);
  Line    -- a straight tangent segment (G1);
  Arc     -- a circular arc with the end's tangent and curvature (G2).
By Distance (arc length of the added piece) or To Active Curve (up to where
the extension first meets the active curve).
"""

import json

import numpy as np

from . import curvejoin, curvetrim, geom

MAX_DEGREE = curvejoin.MAX_DEGREE


# --------------------------------------------------------------------------
# Sampling the exact curve
# --------------------------------------------------------------------------

def dense_samples(spans, per=64):
    pts, _ = curvetrim.sample_spans(spans, per)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12])
    return pts[keep]


def _chord_params(S):
    d = np.linalg.norm(np.diff(S, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)]) / d.sum()


# --------------------------------------------------------------------------
# Basis matrices (open clamped uniform / periodic uniform)
# --------------------------------------------------------------------------

class Basis:
    def __init__(self, n, p, closed):
        self.n, self.p, self.closed = n, p, closed
        if closed:
            self.U = [float(k) / n for k in range(-p, n + p + 1)]
            self.ncv = n + p
        else:
            self.U = geom.clamped_uniform_knots(n, p)
            self.ncv = n

    def matrix(self, ts):
        ts = np.asarray(ts, float)
        if self.closed:
            ts = np.mod(ts, 1.0)
        else:
            ts = np.clip(ts, 0.0, 1.0)
        A = geom.interp_matrix(ts, self.U, self.p, self.ncv)
        if self.closed:
            F = np.zeros((len(ts), self.n))
            for j in range(self.ncv):
                F[:, j % self.n] += A[:, j]
            return F
        return A


def _eval(B, X, ts, h=1e-6):
    """Points, first and second derivatives (finite differences in t)."""
    lo = ts - h
    hi = ts + h
    if not B.closed:   # stay inside [0, 1]
        lo = np.where(ts - h < 0, ts, lo)
        hi = np.where(ts + h > 1, ts, hi)
    C0 = B.matrix(ts) @ X
    Cl, Ch = B.matrix(lo) @ X, B.matrix(hi) @ X
    dt = (hi - lo)[:, None]
    d1 = (Ch - Cl) / np.where(dt == 0, 1, dt)
    d2 = (Ch - 2 * C0 + Cl) / np.where(dt == 0, 1, (dt / 2) ** 2)
    return C0, d1, d2


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def _solve(B, S, ts, tangents, fair):
    """Least squares CVs (n, 3) with the end constraints."""
    n = B.n
    A = B.matrix(ts)
    # second-difference fairing on the CVs (even spacing)
    if B.closed:
        D = np.zeros((n, n))
        for i in range(n):
            D[i, (i - 1) % n], D[i, i], D[i, (i + 1) % n] = 1, -2, 1
    else:
        D = np.zeros((max(n - 2, 0), n))
        for i in range(n - 2):
            D[i, i:i + 3] = (1, -2, 1)
    w = np.sqrt(fair * len(S) / max(n, 1))
    if B.closed:
        M = np.vstack([A, w * D])
        R = np.vstack([S, np.zeros((len(D), 3))])
        return np.linalg.lstsq(M, R, rcond=None)[0]
    # open: X0, X(n-1) fixed; with tangents X1 = X0 + a T0, X(n-2) = Xe - b T1
    X = np.zeros((n, 3))
    X[0], X[-1] = S[0], S[-1]
    use_t = tangents is not None and n >= 4 and B.p >= 2
    free = list(range(2, n - 2)) if use_t else list(range(1, n - 1))
    # columns: per free CV 3 coordinates, then a, b
    ncol = 3 * len(free) + (2 if use_t else 0)
    def expand(Mrows):
        """Mrows (k, n) acting on X -> (3k, ncol) on z and the constant part."""
        k = Mrows.shape[0]
        G = np.zeros((3 * k, ncol))
        c = np.zeros((k, 3))
        for col, i in enumerate(free):
            for d in range(3):
                G[d::3, 3 * col + d] = Mrows[:, i]
        c += np.outer(Mrows[:, 0], X[0]) + np.outer(Mrows[:, n - 1], X[-1])
        if use_t:
            T0, T1 = tangents
            c += np.outer(Mrows[:, 1], X[0]) + np.outer(Mrows[:, n - 2], X[-1])
            for d in range(3):
                G[d::3, -2] = Mrows[:, 1] * T0[d]
                G[d::3, -1] = -Mrows[:, n - 2] * T1[d]
        return G, c.reshape(-1)
    Ga, ca = expand(A)
    Gd, cd = expand(D) if len(D) else (np.zeros((0, ncol)), np.zeros(0))
    G = np.vstack([Ga, w * Gd])
    rhs = np.concatenate([S.reshape(-1) - ca, -w * cd])
    z = np.linalg.lstsq(G, rhs, rcond=None)[0]
    for col, i in enumerate(free):
        X[i] = z[3 * col:3 * col + 3]
    if use_t:
        a, b = z[-2], z[-1]
        L = np.linalg.norm(S[-1] - S[0]) + 1e-12
        # a tangent CV must point forward (a flipped one would put a cusp at the end)
        a, b = max(a, 1e-3 * L / n), max(b, 1e-3 * L / n)
        X[1] = X[0] + a * tangents[0]
        X[-2] = X[-1] - b * tangents[1]
    return X


def _project(B, X, S, ts, iters=3):
    for _ in range(iters):
        C, d1, d2 = _eval(B, X, ts)
        r = C - S
        num = np.einsum("ij,ij->i", r, d1)
        den = np.einsum("ij,ij->i", d1, d1) + np.einsum("ij,ij->i", r, d2)
        den = np.where(np.abs(den) < 1e-18, 1e-18, den)
        step = np.clip(-num / den, -0.5 / B.n, 0.5 / B.n)
        ts = ts + step
        ts = np.mod(ts, 1.0) if B.closed else np.clip(ts, 0.0, 1.0)
        if not B.closed:
            ts[0], ts[-1] = 0.0, 1.0
    return ts


def _seg_dist(P, S):
    """Distance from each point of P to the polyline S."""
    a, b = S[:-1], S[1:]
    ab = b - a
    L = np.einsum("ij,ij->i", ab, ab)
    L = np.where(L == 0, 1, L)
    out = np.empty(len(P))
    for k, p in enumerate(P):
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / L, 0, 1)
        out[k] = np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1))
    return out


def fit(S, deg, n, closed, tangents=None, fair=1e-6, corrections=5):
    """Uniform spline (n CVs, degree deg) through samples S. Returns (X, error)."""
    B = Basis(n, deg, closed)
    if closed:
        S = S[:-1] if np.linalg.norm(S[0] - S[-1]) < 1e-12 else S
        d = np.linalg.norm(np.diff(np.vstack([S, S[:1]]), axis=0), axis=1)
        ts = np.concatenate([[0.0], np.cumsum(d)[:-1]]) / d.sum()
    else:
        ts = _chord_params(S)
    X = _solve(B, S, ts, tangents, fair)
    for _ in range(corrections):
        ts = _project(B, X, S, ts)
        X = _solve(B, S, ts, tangents, fair)
    ts = _project(B, X, S, ts)
    err = float(np.linalg.norm(B.matrix(ts) @ X - S, axis=1).max())
    # the other way round: the fit must not bulge between the samples
    tt = np.linspace(0, 1, 12 * n + 1)[:-1] if closed else np.linspace(0, 1, 12 * n + 1)
    ring = np.vstack([S, S[:1]]) if closed else S
    err = max(err, float(_seg_dist(B.matrix(tt) @ X, ring).max()))
    return X, err


def end_tangents(spans):
    return curvejoin._tangent(spans[0], 0), curvejoin._tangent(spans[-1], 1)


def rebuild(spans, deg, n_cvs, closed, tol, keep_tangents=True, max_cvs=120):
    """(CVs, degree, error) of the rebuilt curve; n_cvs 0 = fewest within tol."""
    S = dense_samples(spans, 64)
    deg = max(1, min(MAX_DEGREE, int(deg)))
    tangents = None if closed or not keep_tangents else end_tangents(spans)
    lo = deg + 1
    if n_cvs:
        X, e = fit(S, deg, max(int(n_cvs), lo), closed, tangents)
        return X, deg, e
    cache = {}
    def f(n):
        if n not in cache:
            cache[n] = fit(S, deg, n, closed, tangents)
        return cache[n]
    # grow until within tolerance, then bisect for the fewest
    n = lo
    while f(n)[1] > tol and n < max_cvs:
        n = min(max_cvs, n * 2)
    good, bad = n, lo - 1
    if f(lo)[1] <= tol:
        good = lo
    while good - bad > 1:
        m = (good + bad) // 2
        if f(m)[1] <= tol:
            good = m
        else:
            bad = m
    if f(good)[1] > tol:      # not monotone near the limit: take the best tried
        good = min(cache, key=lambda k: cache[k][1])
    X, e = f(good)
    return X, deg, e


def rebuilt_spec(X, deg, closed):
    from .primitives import spec
    if closed:
        return spec([tuple(x) for x in X], order=deg + 1, cyclic=True, endpoint=False)
    return spec([tuple(x) for x in X], order=deg + 1)


# --------------------------------------------------------------------------
# Extend
# --------------------------------------------------------------------------

def _span_len(seg, a=0.0, b=1.0, n=64):
    P = curvejoin.eval_span(seg, np.linspace(a, b, n))
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _natural_piece(seg, e):
    """The span's polynomial continued from t = 1 to 1 + e, as a Bezier span."""
    longer, _ = curvetrim.split_bezier(seg, 1.0 + e)       # covers [0, 1 + e]
    _, piece = curvetrim.split_bezier(longer, 1.0 / (1.0 + e))
    return piece


def _weights_ok(seg, e):
    return (np.all(_natural_piece(seg, e)[:, 3] > 1e-9) and
            np.all(curvetrim.split_bezier(seg, 1.0 + e)[0][:, 3] > 1e-9))


def natural_e(seg, length=None, e_max=None):
    """How far (in the span's own parameter, past t = 1) the natural
    continuation goes: by arc length, or as far as e_max allows."""
    if length is None:
        hi = e_max
        while not _weights_ok(seg, hi) and hi > 1e-6:
            hi *= 0.5
        return hi
    hi = 1.0
    while _span_len(_natural_piece(seg, hi)) < length and _weights_ok(seg, hi * 2) and hi < 1e4:
        hi *= 2
    if not _weights_ok(seg, hi) or _span_len(_natural_piece(seg, hi)) < length:
        raise ValueError("Natural extension: the end span cannot be continued that far (weights turn negative)")
    lo = 0.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if _span_len(_natural_piece(seg, m)) < length:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def natural_extension(seg, length=None, e_max=None):
    return _natural_piece(seg, natural_e(seg, length, e_max))


def is_straight(seg, rel=1e-9):
    P = seg[:, :3] / seg[:, 3:4]
    d = P[-1] - P[0]
    L = np.linalg.norm(d)
    if L < 1e-15:
        return False
    d /= L
    off = (P - P[0]) - np.outer((P - P[0]) @ d, d)
    return float(np.linalg.norm(off, axis=1).max()) <= rel * max(L, 1.0) * 1e3


def is_circular(seg):
    """A rational quadratic span that is an exact circular arc."""
    if len(seg) != 3:
        return False
    w = seg[:, 3]
    P = seg[:, :3] / seg[:, 3:4]
    a, b = np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[2] - P[1])
    if a < 1e-15 or abs(a - b) > 1e-6 * a:      # Blender stores CVs in single precision
        return False
    cos_half = float((P[1] - P[0]) @ (P[2] - P[1])) / (a * b)   # cos of the turning angle
    wn = w[1] / np.sqrt(w[0] * w[2])
    return abs(wn - np.sqrt((1 + cos_half) / 2)) < 1e-6


def line_extension(P, T, length):
    return np.array([[*P, 1.0], [*(P + T * length), 1.0]])


def end_frame(seg):
    """End point, unit tangent and curvature vector of a rational Bezier span
    at t = 1 (exact derivatives of the homogeneous curve)."""
    Q = np.array(seg, float)
    p = len(Q) - 1
    H = Q[-1]
    D1 = p * (Q[1:] - Q[:-1])
    dH = D1[-1]
    ddH = (p - 1) * (D1[-1] - D1[-2]) if p >= 2 else np.zeros(4)
    w, dw, ddw = H[3], dH[3], ddH[3]
    C = H[:3] / w
    d1 = (dH[:3] - dw * C) / w
    d2 = (ddH[:3] - 2 * dw * d1 - ddw * C) / w
    s2 = float(d1 @ d1)
    T = d1 / np.sqrt(s2)
    k = (d2 - (d2 @ T) * T) / s2
    return C, T, k


def arc_extension(P, T, kvec, length=None, angle=None):
    """Circular arc spans leaving P along T with curvature vector kvec."""
    kap = float(np.linalg.norm(kvec))
    if kap < 1e-9:
        return [line_extension(P, T, length)]
    R = 1.0 / kap
    N = kvec / kap
    O = P + R * N
    total = angle if angle is not None else length / R
    total = min(total, 2 * np.pi * 0.999)
    nseg = max(1, int(np.ceil(total / (np.pi / 2))))
    out = []
    pt = lambda phi: O + R * (-np.cos(phi) * N + np.sin(phi) * T)
    for k in range(nseg):
        a, b = total * k / nseg, total * (k + 1) / nseg
        half = 0.5 * (b - a)
        w = np.cos(half)
        mid = O + (R / w) * (-np.cos(a + half) * N + np.sin(a + half) * T)
        out.append(np.array([[*pt(a), 1.0], [*(mid * w), w], [*pt(b), 1.0]]))
    return out


def extend_end(spans, kind, length=None, target=None, reach=None, tol=curvetrim.TOL):
    """Spans with the END extended. target: spans of a curve to extend up to
    (length is then ignored; reach = the longest extension tried).

    Where the extension continues the end span's own shape (Natural; Line on a
    straight end; Arc on an exact arc) the last span itself is lengthened:
    no span is added, so a line just gets its end CV moved and a Bezier-knot
    or single-span curve keeps its CV count."""
    last = spans[-1]
    P, T, kv = end_frame(last)
    natural = (kind == "NATURAL" or (kind == "LINE" and is_straight(last))
               or (kind == "ARC" and is_circular(last)))
    if natural:
        if target is None:
            e = natural_e(last, length)
        else:
            E = natural_e(last, e_max=64.0)
            piece = _natural_piece(last, E)
            hits = [h for h in curvetrim.intersect([piece], target, tol) if np.linalg.norm(h[2] - P) > tol]
            if not hits:
                raise ValueError("the extension never reaches the boundary curve")
            e = min(h[0] for h in hits) * E
        longer, _ = curvetrim.split_bezier(last, 1.0 + e)
        return [s.copy() for s in spans[:-1]] + [longer]
    if target is None:
        ext = [line_extension(P, T, length)] if kind == "LINE" else arc_extension(P, T, kv, length)
    else:
        ext = [line_extension(P, T, reach)] if kind == "LINE" else arc_extension(P, T, kv, length=reach)
        hits = [h for h in curvetrim.intersect(ext, target, tol)
                if np.linalg.norm(h[2] - P) > tol]
        if not hits:
            raise ValueError("the extension never reaches the boundary curve")
        g = min(h[0] for h in hits)
        ext = curvetrim.split_spans(ext, [g], False)[0][0]
    return [s.copy() for s in spans] + ext


def extend(spans, ends, kind, length=None, target=None, reach=None, tol=curvetrim.TOL):
    out = [s.copy() for s in spans]
    if "END" in ends:
        out = extend_end(out, kind, length, target, reach, tol)
    if "START" in ends:
        out = curvejoin._reverse(extend_end(curvejoin._reverse(out), kind, length, target, reach, tol))
    return out


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty  # noqa: E402

DEGREE_ITEMS = [(str(d), str(d), "") for d in range(1, MAX_DEGREE + 1)]


def _selected_curves(context):
    return [o for o in context.selected_objects if o.type == 'CURVE' and len(o.data.splines)]


def _targets(context):
    """[(object, [spline indices])]: in Edit Mode the selected splines (all of
    an object's splines if none of its points is selected), else every spline
    of the selected curve objects."""
    if curvetrim.in_edit(context):
        return [(o, curvetrim.selected_splines(o)) for o in curvetrim.edit_objects(context)]
    return [(o, list(range(len(o.data.splines)))) for o in _selected_curves(context)]


def write_bezier_inplace(sp, spans_local):
    """Write cubic (or lower) non-rational spans back into an open BEZIER spline
    with the same number of segments, keeping it a Bezier spline. False if it
    doesn't fit (then the caller writes NURBS)."""
    if sp.type != 'BEZIER' or sp.use_cyclic_u or len(spans_local) != len(sp.bezier_points) - 1:
        return False
    if any(len(s) > 4 or np.abs(s[:, 3] - s[0, 3]).max() > 1e-12 for s in spans_local):
        return False
    segs = [curvejoin.elevate(s / s[0, 3], 3)[:, :3] for s in spans_local]
    bp = sp.bezier_points
    for k, seg in enumerate(segs):
        a, b = bp[k], bp[k + 1]
        old = [tuple(a.co), tuple(a.handle_right), tuple(b.handle_left), tuple(b.co)]
        new = [tuple(map(float, x)) for x in seg]
        if np.abs(np.array(old) - np.array(new)).max() < 1e-7:
            continue
        for q, typ in ((a, "handle_right_type"), (b, "handle_left_type")):
            if getattr(q, typ) in ('AUTO', 'VECTOR'):
                setattr(q, typ, 'ALIGNED' if q not in (bp[0], bp[-1]) else 'FREE')
        a.co, a.handle_right, b.handle_left, b.co = new
        for q in (a, b):   # the other handle of an end point follows it (free)
            if q in (bp[0], bp[-1]):
                q.handle_left_type = q.handle_right_type = 'FREE'
    return True


def _write_world(op, obj, index, world_spans=None, spec_world=None):
    """Put one spline back from world-space spans (or a world-space spec).
    In Edit Mode this edits the edit data in place and selects the result."""
    Minv = np.linalg.inv(np.array(geom.world_matrix(obj), float))
    curvetrim._drop_history(op, obj)
    if world_spans is not None:
        loc = curvejoin.transform_spans(world_spans, Minv)
        sp = obj.data.splines[index]
        if write_bezier_inplace(sp, loc):
            curvetrim.touch(obj)
            return
        curvetrim.put_back(obj, index, [loc], same_object=True)
        return
    pts = np.array(spec_world["points"], float)
    loc = pts @ Minv[:3, :3].T + Minv[:3, 3]
    _put_spec(obj, index, dict(spec_world, points=[tuple(p) for p in loc]))


def _put_spec(obj, index, sp):
    from .primitives import _add_spline
    cu = obj.data
    after = [curvetrim._spline_spec(s) for s in list(cu.splines)[index + 1:]]
    for s in list(cu.splines)[index:]:
        cu.splines.remove(s)
    _add_spline(cu, sp)
    if cu.is_editmode:
        curvetrim.select_spline(cu.splines[-1])
    for d in after:
        curvetrim._readd(cu, d)
    curvetrim.touch(obj)


class NPROTO_OT_rebuild_curve(bpy.types.Operator):
    """Refit the selected curves as clean uniform splines: chosen degree, evenly
    spread CVs, same shape within Tolerance (end points kept exactly)"""
    bl_idname = "nproto.rebuild_curve"
    bl_label = "Rebuild Curve"
    bl_options = {'REGISTER', 'UNDO'}

    rc_degree: EnumProperty(name="Degree", items=DEGREE_ITEMS, default="3")
    rc_cvs: IntProperty(name="CVs (0 auto)", default=0, min=0, max=120,
                        description="Number of CVs; 0 = the fewest that keep the shape within Tolerance")
    tolerance: FloatProperty(name="Tolerance", default=1e-4, min=1e-7, soft_max=0.1, precision=6, unit='LENGTH',
                             description="Largest allowed deviation from the original curve (CVs 0)")
    keep_tangents: BoolProperty(name="Keep End Tangents", default=True,
                                description="Open curves keep their end directions (stay G1 to what they meet)")

    def invoke(self, context, event):
        curvetrim._object_mode(context)
        return self.execute(context)

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "rc_degree")
        lay.prop(self, "rc_cvs")
        lay.prop(self, "tolerance")
        lay.prop(self, "keep_tangents")

    def execute(self, context):
        targets = _targets(context)
        if not targets:
            self.report({'WARNING'}, "Select curves to rebuild")
            return {'CANCELLED'}
        msgs, worst = [], 0.0
        for obj, idx in targets:
            M = geom.world_matrix(obj)
            if curvetrim.in_edit(context):
                for sp in obj.data.splines:
                    curvetrim.select_spline(sp, False)
            for i in sorted(idx, reverse=True):
                sp = obj.data.splines[i]
                try:
                    spans = curvejoin.transform_spans(curvejoin.spline_spans(sp), M)
                    before = len(sp.points) if sp.type != 'BEZIER' else 3 * len(sp.bezier_points)
                    closed = curvetrim.is_closed(spans, sp.use_cyclic_u)
                    X, deg, e = rebuild(spans, int(self.rc_degree), self.rc_cvs, closed, self.tolerance,
                                        self.keep_tangents)
                except Exception as exc:
                    self.report({'ERROR'}, f"{obj.name}: {exc}")
                    continue
                _write_world(self, obj, i, spec_world=rebuilt_spec(X, deg, closed))
                worst = max(worst, e)
                msgs.append(f"{obj.name}: {before} -> {len(X)} CVs")
        if not msgs:
            return {'CANCELLED'}
        level = 'WARNING' if (self.rc_cvs == 0 and worst > self.tolerance) else 'INFO'
        self.report({level}, f"Rebuilt degree {self.rc_degree} · " + ", ".join(msgs)
                    + f" · max deviation {worst:.3g}"
                    + (" (above Tolerance: raise CVs or the degree)" if level == 'WARNING' else ""))
        return {'FINISHED'}


class NPROTO_OT_extend_curve(bpy.types.Operator):
    """Extend the selected open curves at their ends (the original part stays
    exact): Natural continuation, a tangent Line or a curvature-matching Arc,
    by a distance or up to another curve. A line just gets its end CV moved"""
    bl_idname = "nproto.extend_curve"
    bl_label = "Extend Curve"
    bl_options = {'REGISTER', 'UNDO'}

    ec_type: EnumProperty(name="Type", default="NATURAL", items=[
        ("NATURAL", "Natural", "Continue the end span's own shape (a line stays a line, an arc the same circle)"),
        ("LINE", "Line", "Straight tangent extension (G1)"),
        ("ARC", "Arc", "Circular arc matching the end's tangent and curvature (G2)")])
    ec_to: EnumProperty(name="Extend", default="DISTANCE", items=[
        ("DISTANCE", "By Distance", "Add this much length"),
        ("ANY", "To Next Curve", "Extend up to the first visible curve the extension meets"),
        ("ACTIVE", "To Active Curve", "Extend up to where the extension meets the active curve "
                                      "(Object Mode; select the boundary curve last)")])
    ec_end: EnumProperty(name="End", default="NEAREST", items=[
        ("NEAREST", "Auto", "Edit Mode: the selected end point(s). Otherwise the end nearest the boundary "
                            "curve (To ... Curve) or the 3D cursor"),
        ("START", "Start", ""), ("END", "End", ""), ("BOTH", "Both", "")])
    distance: FloatProperty(name="Distance", default=0.5, min=0.0, soft_max=10.0, unit='LENGTH')
    keep_cvs: BoolProperty(name="Keep CV Count", default=False,
                           description="A curve that can't be lengthened exactly in place (a uniform spline with "
                                       "several spans gains CVs) is refit with its own degree and CV count instead; "
                                       "the deviation is reported. Lines, arcs, Bezier and single-span curves "
                                       "never gain CVs")
    tolerance: FloatProperty(name="Tolerance", default=curvetrim.TOL, min=1e-7, precision=6, unit='LENGTH',
                             options={'HIDDEN'})
    names: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    active: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        curvetrim._object_mode(context)
        self.names = json.dumps([o.name for o in _selected_curves(context)])
        self.active = context.active_object.name if context.active_object else ""
        return self.execute(context)

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.row().prop(self, "ec_type", expand=True)
        lay.prop(self, "ec_to")
        lay.prop(self, "ec_end")
        if self.ec_to == "DISTANCE":
            lay.prop(self, "distance")
        lay.prop(self, "keep_cvs")

    def _boundary(self, context, obj, i):
        """World spans the extension of spline (obj, i) runs up to, or None."""
        if self.ec_to == "DISTANCE":
            return None
        if self.ec_to == "ACTIVE" and not curvetrim.in_edit(context):
            tobj = bpy.data.objects.get(self.active)
            pool = [tobj] if tobj is not None and tobj.type == 'CURVE' else []
        else:
            pool = [o for o in context.visible_objects if o.type == 'CURVE']
        out = []
        for o in pool:
            M = geom.world_matrix(o)
            for k, sp in enumerate(o.data.splines):
                if o == obj and k == i:
                    continue
                try:
                    out += curvejoin.transform_spans(curvejoin.spline_spans(sp), M)
                except Exception:
                    pass
        return out

    def execute(self, context):
        edit = curvetrim.in_edit(context)
        if edit:
            targets = _targets(context)
        else:
            if not self.names:
                self.names = json.dumps([o.name for o in _selected_curves(context)])
                self.active = context.active_object.name if context.active_object else ""
            objs = [bpy.data.objects[n] for n in json.loads(self.names) if n in bpy.data.objects]
            if self.ec_to == "ACTIVE":
                tobj = bpy.data.objects.get(self.active)
                if tobj is None or tobj.type != 'CURVE':
                    self.report({'WARNING'}, "To Active Curve: make the boundary curve active (select it last)")
                    return {'CANCELLED'}
                objs = [o for o in objs if o.name != tobj.name]
            targets = [(o, list(range(len(o.data.splines)))) for o in objs if o.type == 'CURVE']
        if not targets:
            self.report({'WARNING'}, "Select the curves to extend" +
                        (" and, last, the boundary curve" if self.ec_to == "ACTIVE" else ""))
            return {'CANCELLED'}
        cursor = np.array(context.scene.cursor.location, float)
        done, errors, notes = 0, [], []
        for obj, idx in targets:
            M = geom.world_matrix(obj)
            for i in sorted(idx, reverse=True):
                sp = obj.data.splines[i]
                try:
                    spans = curvejoin.transform_spans(curvejoin.spline_spans(sp), M)
                    if curvetrim.is_closed(spans, sp.use_cyclic_u):
                        raise ValueError("closed curves have no ends to extend")
                    target = self._boundary(context, obj, i)
                    if target is not None and not target:
                        raise ValueError("no boundary curve to extend to")
                    ends = self._ends(sp, spans, target, cursor, edit)
                    reach = None
                    if target is not None:
                        allp = np.vstack([curvejoin._cart(s) for s in spans + target])
                        reach = 3.0 * float(np.linalg.norm(np.ptp(allp, axis=0)))
                    new = extend(spans, ends, self.ec_type, self.distance, target, reach, self.tolerance)
                    before = len(sp.bezier_points) if sp.type == 'BEZIER' else len(sp.points)
                    refit = None
                    if self.keep_cvs and sp.type != 'BEZIER' and len(curvejoin.spans_to_spec(new)["points"]) > before:
                        X, deg, e = rebuild(new, sp.order_u - 1, before, False, self.tolerance, True)
                        refit = rebuilt_spec(X, deg, False)
                        notes.append(f"{obj.name} refit to {before} CVs (deviation {e:.2g})")
                except Exception as exc:
                    errors.append(f"{obj.name}: {exc}")
                    continue
                if refit is not None:
                    _write_world(self, obj, i, spec_world=refit)
                else:
                    _write_world(self, obj, i, world_spans=new)
                done += 1
        for e in errors:
            self.report({'WARNING'}, e)
        if not done:
            return {'CANCELLED'}
        self.report({'INFO'}, f"Extended {done} curve(s) ({self.ec_type.title()})" +
                    (" · " + ", ".join(notes) if notes else ""))
        return {'FINISHED'}

    def _ends(self, sp, spans, target, cursor, edit):
        if self.ec_end == "BOTH":
            return ("START", "END")
        if self.ec_end in ("START", "END"):
            return (self.ec_end,)
        if edit:
            a_sel, b_sel = curvetrim.end_selection(sp)
            if a_sel or b_sel:
                return tuple(e for e, on in (("START", a_sel), ("END", b_sel)) if on)
        a, b = curvejoin._ends(spans)
        if target is not None:
            ring = np.vstack([curvetrim.sample_spans([s], 16)[0] for s in target])
            da, db = _seg_dist(np.array([a, b]), ring)
        else:
            da, db = np.linalg.norm(a - cursor), np.linalg.norm(b - cursor)
        return ("START",) if da < db else ("END",)


# --------------------------------------------------------------------------
# Degree: what the Active Object panel shows for a curve (Blender's spline
# panel says Order U = degree + 1)
# --------------------------------------------------------------------------

def spline_info(sp):
    """{"degree", "cvs", "spans", "kind", "rational"} of a Blender spline."""
    if sp.type == 'BEZIER':
        n = len(sp.bezier_points)
        return {"degree": 3, "cvs": n, "spans": max(n - 1 + (1 if sp.use_cyclic_u else 0), 0),
                "kind": "Bezier", "rational": False}
    n = len(sp.points)
    if sp.type == 'POLY':
        return {"degree": 1, "cvs": n, "spans": max(n - 1 + (1 if sp.use_cyclic_u else 0), 0),
                "kind": "polyline", "rational": False}
    p = max(1, min(sp.order_u, n) - 1)
    if sp.use_cyclic_u:
        spans, kind = n, "closed"
    elif sp.use_bezier_u:
        spans, kind = max((n - 1) // p, 1), "piecewise Bezier"
    else:
        spans, kind = max(n - p, 1), ("open" if sp.use_endpoint_u else "unclamped ends")
    rational = any(abs(pt.co[3] - 1.0) > 1e-9 for pt in sp.points)
    return {"degree": p, "cvs": n, "spans": spans, "kind": kind, "rational": rational}


def degree_note(info):
    """What the degree means for the curvature comb, or None."""
    if info["rational"] or info["spans"] < 2:
        return None
    if info["degree"] == 2:
        return "Degree 2: the curvature jumps or kinks at every knot, the comb steps there. Degree 3+ flows (G2)."
    if info["kind"] == "piecewise Bezier" and info["degree"] >= 3:
        return "Piecewise Bezier: the spans meet at stacked knots, where the comb can step."
    return None


class NPROTO_OT_curve_degree(bpy.types.Operator):
    """Set the curve's degree, keeping its CVs (the shape follows the new
    degree). Blender's spline panel calls this Order U, which is degree + 1.
    Rebuild Curve keeps the shape instead"""
    bl_idname = "nproto.curve_degree"
    bl_label = "Curve Degree"
    bl_options = {'REGISTER', 'UNDO'}

    degree: IntProperty(name="Degree", default=3, min=1, max=MAX_DEGREE)
    spline: IntProperty(name="Spline", default=-1, min=-1, description="-1: every NURBS spline")

    def execute(self, context):
        obj = context.view_layer.objects.active
        if obj is None or obj.type != 'CURVE':
            self.report({'WARNING'}, "Make a curve active")
            return {'CANCELLED'}
        splines = list(obj.data.splines)
        targets = splines if self.spline < 0 else splines[self.spline:self.spline + 1]
        done, short = 0, 0
        for sp in targets:
            if sp.type != 'NURBS':
                continue
            if len(sp.points) < self.degree + 1:
                short += 1
                continue
            sp.order_u = self.degree + 1
            done += 1
        obj.data.update_tag()
        if hasattr(obj, "update_tag"):
            obj.update_tag()
        if short and not done:
            self.report({'WARNING'}, f"Degree {self.degree} needs at least {self.degree + 1} CVs")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Degree {self.degree} (Blender: Order U {self.degree + 1})"
                    + (f" · {short} spline(s) have too few CVs" if short else ""))
        return {'FINISHED'}


def draw_curve_info(layout, context, obj):
    """The Curve box of the Active Object panel: per spline its degree, CVs,
    spans and kind; Degree 1-5 buttons (same CVs); a note when the degree
    makes the comb step; Rebuild as degree 3 keeping the shape."""
    from . import features
    splines = list(obj.data.splines)
    if not splines:
        return
    box = layout.box()
    box.label(text=f"Curve: {len(splines)} spline(s)" if len(splines) > 1 else "Curve", icon='CURVE_NCURVE')
    editable = not features.has_history(obj)
    notes = set()
    for i, sp in enumerate(splines[:3]):
        info = spline_info(sp)
        col = box.column(align=True)
        head = f"Spline {i + 1}: " if len(splines) > 1 else ""
        extra = " · rational" if info["rational"] else ""
        col.label(text=f"{head}degree {info['degree']} · {info['cvs']} CVs · {info['spans']} span(s) · "
                       f"{info['kind']}{extra}")
        if sp.type == 'NURBS' and editable:
            row = col.row(align=True)
            row.label(text="Degree")
            for d in range(1, MAX_DEGREE + 1):
                sub = row.row(align=True)
                sub.enabled = info["cvs"] >= d + 1
                op = sub.operator("nproto.curve_degree", text=str(d), depress=(d == info["degree"]))
                op.degree, op.spline = d, i
            col.label(text=f"Blender: Order U {info['degree'] + 1} (= degree + 1)")
        note = degree_note(info)
        if note:
            notes.add(note)
    if len(splines) > 3:
        box.label(text=f"... and {len(splines) - 3} more spline(s)")
    for note in sorted(notes):
        box.label(text=note, icon='INFO')
    if notes and editable:
        op = box.operator("nproto.rebuild_curve", text="Rebuild as Degree 3 (same shape, smooth comb)",
                          icon='FILE_REFRESH')
        op.rc_degree, op.rc_cvs = "3", 0
    elif not editable:
        box.label(text="Built by its history: set the degree in its Feature box", icon='MODIFIER')


classes = (NPROTO_OT_rebuild_curve, NPROTO_OT_extend_curve, NPROTO_OT_curve_degree)
