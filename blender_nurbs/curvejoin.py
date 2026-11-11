"""Join Curves / Unjoin Curves.

Join chains the selected curves end to end (flipping pieces as needed) and
turns each chain into ONE Blender NURBS spline:

* Exact   -- every piece is split into its Bezier spans, raised to the highest
             degree in the chain and the spans are strung together. Blender
             stores that as an endpoint + Bezier-knot spline (every interior
             knot of multiplicity = degree), so the shape is unchanged to the
             last digit, joints included. When the result happens to be a
             plain uniform spline (e.g. two halves of one curve) it is stored
             as that instead.
* Rebuild -- one clean uniform spline (chosen degree, CV count or auto)
             fitted through the chain: the Class-A choice, no internal knots.

Blender's legacy splines cannot hold arbitrary knot vectors, which is why the
exact result keeps a knot at every span boundary rather than removing the
knots at smooth joints.

Unjoin splits a curve back into pieces: at the joints a Join recorded, else at
every corner (tangent break). Multi-spline objects come apart into one object
per spline. Pieces are written in the lowest degree that holds them exactly.
"""

import json

import numpy as np

from . import features, geom

LAST_INFO = {}
MAX_DEGREE = 5          # Blender NURBS splines: order <= 6
CORNER_DEG = 1.0        # Unjoin: a tangent break above this is a corner
PROP_JOINTS = "nproto_joints"


# --------------------------------------------------------------------------
# Bezier spans (homogeneous: rows x*w, y*w, z*w, w)
# --------------------------------------------------------------------------

def _clamped_to_spans(deg, P, W, U):
    """Bezier spans of a clamped NURBS curve (knot insertion to full multiplicity)."""
    P = np.asarray(P, float)
    W = np.ones(len(P)) if W is None else np.asarray(W, float)
    Pw = [np.array([*(p * w), w]) for p, w in zip(P, W)]
    U = [float(u) for u in U]
    if deg < 1:
        raise ValueError("degree 0 curve")
    lo, hi = U[deg], U[len(U) - deg - 1]
    tol = 1e-12 * max(1.0, abs(hi - lo))
    interior = sorted({u for u in U if lo + tol < u < hi - tol})
    for t in interior:
        while sum(1 for u in U if abs(u - t) <= 1e-12) < deg:
            Pw, U = geom._insert_knot(Pw, U, deg, t)
    Pw = np.array(Pw)
    nseg = (len(Pw) - 1) // deg
    return [Pw[i * deg:i * deg + deg + 1].copy() for i in range(nseg)]


def spline_spans(spline):
    """Bezier spans of a Blender spline in its object's local space."""
    if spline.type == 'BEZIER':
        bp = list(spline.bezier_points)
        idx = list(range(len(bp) - 1)) + ([len(bp) - 1] if spline.use_cyclic_u and len(bp) > 1 else [])
        out = []
        for i in idx:
            a, b = bp[i], bp[(i + 1) % len(bp)]
            pts = [a.co, a.handle_right, b.handle_left, b.co]
            out.append(np.array([[*p, 1.0] for p in pts], float))
        if not out:
            raise ValueError("Bezier spline needs two points")
        return out
    co = [tuple(p.co) for p in spline.points]
    pts = [c[:3] for c in co]
    if spline.type == 'POLY':
        if spline.use_cyclic_u:
            pts = pts + [pts[0]]
        return [np.array([[*pts[i], 1.0], [*pts[i + 1], 1.0]]) for i in range(len(pts) - 1)]
    order = max(2, min(spline.order_u, len(pts)))
    deg, P, W, U = geom.blender_nurbs_to_clamped(pts, [c[3] for c in co], order, spline.use_cyclic_u,
                                                 spline.use_endpoint_u, spline.use_bezier_u)
    return _clamped_to_spans(deg, P, W, U)


def kernel_curve_spans(kc):
    """Bezier spans of a kernel curve (e.g. an exact surface edge)."""
    _, ext = features._k()
    d = ext.curve_cvs(kc)
    return _clamped_to_spans(int(d["degree"]), d["points"], d["weights"] or None, d["knots"])


def transform_spans(spans, M):
    M = np.array(M, float)
    out = []
    for s in spans:
        w = s[:, 3:4]
        p = s[:, :3] / w @ M[:3, :3].T + M[:3, 3]
        out.append(np.hstack([p * w, w]))
    return out


def elevate(seg, q):
    """Raise a homogeneous Bezier span to degree q (exact)."""
    s = seg
    while len(s) - 1 < q:
        p = len(s) - 1
        n = np.zeros((p + 2, s.shape[1]))
        n[0], n[-1] = s[0], s[-1]
        for i in range(1, p + 1):
            a = i / (p + 1)
            n[i] = a * s[i - 1] + (1 - a) * s[i]
        s = n
    return s


def _elev_matrix(r, q):
    """(q+1) x (r+1) matrix taking degree-r Bezier CVs to degree q."""
    E = np.eye(r + 1)
    for p in range(r, q):
        n = np.zeros((p + 2, p + 1))
        n[0, 0] = n[-1, -1] = 1.0
        for i in range(1, p + 1):
            a = i / (p + 1)
            n[i, i - 1], n[i, i] = a, 1 - a
        E = n @ E
    return E


def lowest_degree(seg, tol):
    """Lowest degree that holds this span exactly, and its CVs."""
    q = len(seg) - 1
    for r in range(1, q):
        E = _elev_matrix(r, q)
        X = np.linalg.lstsq(E, seg, rcond=None)[0]
        if np.abs(E @ X - seg).max() <= tol:
            return r, X
    return q, seg


def _cart(s):
    return s[:, :3] / s[:, 3:4]


def eval_span(seg, ts):
    """Points on a rational Bezier span at parameters ts in [0, 1]."""
    from math import comb
    p = len(seg) - 1
    ts = np.asarray(ts, float)
    B = np.array([[comb(p, i) * t ** i * (1 - t) ** (p - i) for i in range(p + 1)] for t in ts])
    H = B @ seg
    return H[:, :3] / H[:, 3:4]


def _tangent(seg, end):
    """Unit tangent (direction of travel) of a span at its start (0) or end (1)."""
    P = _cart(seg)
    if end == 0:
        for k in range(1, len(P)):
            d = P[k] - P[0]
            if np.linalg.norm(d) > 1e-12:
                return d / np.linalg.norm(d)
    else:
        for k in range(len(P) - 2, -1, -1):
            d = P[-1] - P[k]
            if np.linalg.norm(d) > 1e-12:
                return d / np.linalg.norm(d)
    return np.zeros(3)


def _size(spans):
    P = np.vstack([_cart(s) for s in spans])
    return float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0


# --------------------------------------------------------------------------
# Spans -> Blender spline spec
# --------------------------------------------------------------------------

def _uniform_fit(spans, deg, size):
    """The spans as ONE uniform endpoint spline, if that is exact (non-rational only)."""
    n = len(spans)
    if n < 2 or deg < 2 or any(np.abs(s[:, 3] - 1.0).max() > 1e-12 for s in spans):
        return None
    ncv = n + deg
    U = geom.clamped_uniform_knots(ncv, deg)
    ts, pts = [], []
    k = 3 * deg + 2
    for i, s in enumerate(spans):
        loc = np.linspace(0, 1, k)
        ts += list((i + loc) / n)
        pts.append(eval_span(s, loc))
    A = geom.interp_matrix(np.clip(ts, 0, 1), U, deg, ncv)
    S = np.vstack(pts)
    X = np.linalg.lstsq(A, S, rcond=None)[0]
    if np.abs(A @ X - S).max() > 1e-6 * size:
        return None
    from .primitives import spec
    return spec([tuple(x) for x in X], order=deg + 1)


def spans_to_spec(spans, simplify=True):
    """Spline spec holding these spans exactly (endpoint + Bezier knots, or a
    plain uniform spline / single span when that is exact). Degree = max span
    degree, or the lowest exact degree when simplify."""
    from .primitives import spec
    size = _size(spans)
    tol = 1e-6 * size        # Blender stores CVs in single precision
    q = max(len(s) - 1 for s in spans)
    spans = [elevate(s, q) for s in spans]
    # a joint stores ONE homogeneous CV: scale each span (all its weights, which
    # leaves its shape unchanged) so its first weight equals the previous span's last
    for k in range(1, len(spans)):
        a, b = spans[k - 1][-1, 3], spans[k][0, 3]
        if b != 0 and abs(a - b) > 1e-15 * abs(a):
            spans[k] = spans[k] * (a / b)
    if simplify:
        r = max(lowest_degree(s, tol)[0] for s in spans)
        if r < q:
            E = _elev_matrix(r, q)
            spans = [np.linalg.lstsq(E, s, rcond=None)[0] for s in spans]
            q = r
    if q > MAX_DEGREE:
        raise ValueError(f"degree {q} is above Blender's limit of {MAX_DEGREE}: use Rebuild")
    u = _uniform_fit(spans, q, size) if simplify else None
    if u is not None:
        return u
    rows = [spans[0][0]] + [r for s in spans for r in s[1:]]
    rows = np.array(rows)
    W = rows[:, 3]
    P = rows[:, :3] / W[:, None]
    if len(spans) == 1:
        return spec([tuple(p) for p in P], list(W), order=q + 1)
    return spec([tuple(p) for p in P], list(W), order=q + 1, endpoint=True, bezier=True)


def spec_span_count(sp):
    deg, P, W, U = geom.blender_nurbs_to_clamped(sp["points"], sp["weights"], sp["order"], sp["cyclic"],
                                                 sp["endpoint"], sp["bezier"])
    return len(_clamped_to_spans(deg, P, W, U))


# --------------------------------------------------------------------------
# Chaining
# --------------------------------------------------------------------------

def _reverse(piece):
    return [s[::-1].copy() for s in reversed(piece)]


def _ends(piece):
    return _cart(piece[0])[0], _cart(piece[-1])[-1]


def chain_pieces(pieces, tol):
    """Order pieces (lists of spans) into chains, flipping as needed.
    Returns [(spans, joints, closed, max_gap)] -- joints = span indices where
    a new input piece starts."""
    left = list(range(len(pieces)))
    chains = []

    def free_end(i):
        a, b = _ends(pieces[i])
        others = [_ends(pieces[j]) for j in left if j != i]
        near = lambda x: any(min(np.linalg.norm(x - e[0]), np.linalg.norm(x - e[1])) <= tol for e in others)
        return (not near(a)), (not near(b))

    while left:
        # start at a piece with a free end (open chain), free end first
        start, flip = left[0], False
        for i in left:
            fa, fb = free_end(i)
            if fa or fb:
                start, flip = i, (not fa)
                break
        left.remove(start)
        cur = _reverse(pieces[start]) if flip else [s.copy() for s in pieces[start]]
        parts = [cur]
        gap = 0.0
        for direction in (1, 0):           # grow forward, then backward
            while left:
                end = _ends(parts[-1])[1] if direction else _ends(parts[0])[0]
                best = None
                for j in left:
                    a, b = _ends(pieces[j])
                    for d, rev in ((np.linalg.norm(end - a), False), (np.linalg.norm(end - b), True)):
                        if d <= tol and (best is None or d < best[0]):
                            best = (d, j, rev)
                if best is None:
                    break
                d, j, rev = best
                left.remove(j)
                pc = _reverse(pieces[j]) if rev else [s.copy() for s in pieces[j]]
                if not direction:
                    pc = _reverse(pc)      # backward growth: the new piece must END at our start
                gap = max(gap, d)
                if direction:
                    parts.append(pc)
                else:
                    parts.insert(0, pc)
        # snap the joints (midpoint of the two ends, weights kept)
        for k in range(len(parts) - 1):
            _snap(parts[k][-1], -1, parts[k + 1][0], 0)
        closed = False
        if sum(len(p) for p in parts) > 1:
            a, b = _ends(parts[0])[0], _ends(parts[-1])[1]
            d = np.linalg.norm(a - b)
            if d <= tol:
                closed = True
                gap = max(gap, d)
                _snap(parts[-1][-1], -1, parts[0][0], 0)
        joints, n = [], 0
        for p in parts[:-1]:
            n += len(p)
            joints.append(n)
        chains.append(([s for p in parts for s in p], joints, closed, gap))
    return chains


def _snap(seg_a, ia, seg_b, ib):
    pa = seg_a[ia, :3] / seg_a[ia, 3]
    pb = seg_b[ib, :3] / seg_b[ib, 3]
    m = 0.5 * (pa + pb)
    seg_a[ia, :3] = m * seg_a[ia, 3]
    seg_b[ib, :3] = m * seg_b[ib, 3]


# --------------------------------------------------------------------------
# Rebuild (fit one uniform spline through the chain)
# --------------------------------------------------------------------------

def rebuild_spec(spans, deg, n_cvs, closed):
    from .primitives import _fit_open_spec
    S = [eval_span(spans[0], [0.0])]
    for s in spans:
        S.append(eval_span(s, np.linspace(0, 1, 41)[1:]))
    S = np.vstack(S)
    size = _size(spans)
    deg = max(1, min(MAX_DEGREE, int(deg)))

    dch = np.linalg.norm(np.diff(S, axis=0), axis=1)
    tch = np.concatenate([[0.0], np.cumsum(dch)]) / dch.sum()   # the fit's own parameters

    def err(sp):
        # residual at the fit's sample parameters (an upper bound of the distance to the curve)
        d, P, W, U = geom.blender_nurbs_to_clamped(sp["points"], sp["weights"], sp["order"], False, True, False)
        Q = np.array([geom.eval_bspline(P, W, U, d, t) for t in tch])
        return float(np.linalg.norm(Q - S, axis=1).max())

    if n_cvs and n_cvs > deg:
        sp = _fit_open_spec(S, deg, int(n_cvs))
        return sp, err(sp)
    best = None
    for n in range(deg + 1, 81):
        sp = _fit_open_spec(S, deg, n)
        e = err(sp)
        best = (sp, e)
        if e <= 1e-4 * size:
            break
    return best


# --------------------------------------------------------------------------
# Feature builder
# --------------------------------------------------------------------------

def input_spans(obj, ref):
    if ref.startswith("S") and obj.type == 'CURVE':
        return transform_spans(spline_spans(obj.data.splines[int(ref[1:])]), geom.world_matrix(obj))
    return kernel_curve_spans(features.input_curve(obj, ref))


@features.builder("join_curves")
def build_join_curves(inputs, p):
    tol = float(p.get("tolerance", 1e-3))
    pieces = [input_spans(o, r) for o, r in inputs]
    chains = chain_pieces(pieces, tol)
    mode = p.get("jc_mode", "EXACT")
    specs, info = [], {"chains": len(chains), "joints": 0, "gap": 0.0, "fit_error": 0.0, "pieces": len(pieces)}
    for spans, joints, closed, gap in chains:
        info["joints"] += len(joints) + (1 if closed and joints else 0)
        info["gap"] = max(info["gap"], gap)
        if mode == "REBUILD":
            sp, e = rebuild_spec(spans, int(p.get("jc_degree", 3)), int(p.get("jc_cvs", 0)), closed)
            info["fit_error"] = max(info["fit_error"], float(e))
        else:
            sp = spans_to_spec(spans, simplify=True)
            sp["joints"] = {"nseg": len(spans), "joints": joints}
        specs.append(sp)
    info["cvs"] = sum(len(s["points"]) for s in specs)
    info["degree"] = max(s["order"] - 1 for s in specs)
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return ("splines", specs)


# --------------------------------------------------------------------------
# Unjoin
# --------------------------------------------------------------------------

def split_points(spans, joints=None):
    """Span indices where a new piece starts: the recorded joints, plus every
    corner (tangent break above CORNER_DEG)."""
    cut = set(j for j in (joints or []) if 0 < j < len(spans))
    c = np.cos(np.radians(CORNER_DEG))
    for i in range(1, len(spans)):
        a, b = _tangent(spans[i - 1], 1), _tangent(spans[i], 0)
        if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0 and float(a @ b) < c:
            cut.add(i)
    return sorted(cut)


def unjoin_spline(spline, record=None):
    """Spec list (local space) of the pieces of one spline."""
    spans = spline_spans(spline)
    joints = None
    if record and record.get("nseg") == len(spans):
        joints = record.get("joints")
    cuts = [0] + split_points(spans, joints) + [len(spans)]
    return [spans_to_spec(spans[a:b], simplify=True) for a, b in zip(cuts[:-1], cuts[1:])]


def joint_records(obj):
    try:
        recs = json.loads(obj.get(PROP_JOINTS, "[]"))
    except (TypeError, ValueError):
        return []
    return recs if isinstance(recs, list) else []


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import EnumProperty, FloatProperty, IntProperty, StringProperty  # noqa: E402

MODE_ITEMS = [("EXACT", "Exact", "Keep every piece's shape exactly (a knot at every span boundary)"),
              ("REBUILD", "Rebuild", "Fit one clean uniform curve through the chain (no internal knots)")]
DEGREE_ITEMS = [(str(d), str(d), "") for d in range(1, MAX_DEGREE + 1)]


def _curve_inputs(objs):
    return [(o, f"S{i}") for o in objs for i in range(len(o.data.splines))]


def _report_text(info):
    msg = f"Joined {info['pieces']} curves at {info['joints']} joint(s)"
    if info["chains"] > 1:
        msg += f" into {info['chains']} separate curves (ends further apart than Tolerance stay open)"
    msg += f" -- degree {info['degree']}, {info['cvs']} CVs"
    if info["gap"] > 0:
        msg += f", largest gap closed {info['gap']:.3g}"
    if info.get("fit_error"):
        msg += f", fit error {info['fit_error']:.3g}"
    return msg


class NPROTO_OT_join_curves(bpy.types.Operator):
    """Join the selected curves end to end into one curve (pieces are flipped
    as needed). With construction history the source curves stay hidden and
    drive the result"""
    bl_idname = "nproto.join_curves"
    bl_label = "Join Curves"
    bl_options = {'REGISTER', 'UNDO'}

    jc_mode: EnumProperty(name="Mode", items=MODE_ITEMS, default="EXACT")
    tolerance: FloatProperty(name="Tolerance", default=0.001, min=1e-6, soft_max=0.1, precision=5, unit='LENGTH',
                             description="Largest gap between curve ends that is still joined (the ends meet in the middle)")
    jc_degree: EnumProperty(name="Degree", items=DEGREE_ITEMS, default="3", description="Rebuild: degree of the fitted curve")
    jc_cvs: IntProperty(name="CVs (0 auto)", default=0, min=0, max=200,
                        description="Rebuild: number of CVs (0 = fewest within 0.01% of the size)")
    names: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        self.names = json.dumps([o.name for o in context.selected_objects if o.type == 'CURVE'])
        return self.execute(context)

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.row().prop(self, "jc_mode", expand=True)
        lay.prop(self, "tolerance")
        if self.jc_mode == "REBUILD":
            lay.prop(self, "jc_degree")
            lay.prop(self, "jc_cvs")

    def execute(self, context):
        from .ops import _select_only
        if not self.names:
            self.names = json.dumps([o.name for o in context.selected_objects if o.type == 'CURVE'])
        objs = [bpy.data.objects[n] for n in json.loads(self.names) if n in bpy.data.objects]
        inputs = _curve_inputs(objs)
        if len(inputs) < 2:
            self.report({'WARNING'}, "Select two or more curves")
            return {'CANCELLED'}
        params = {"jc_mode": self.jc_mode, "tolerance": self.tolerance,
                  "jc_degree": int(self.jc_degree), "jc_cvs": self.jc_cvs}
        try:
            g = features.run("join_curves", inputs, params)
        except Exception as exc:
            self.report({'ERROR'}, f"Join Curves failed: {exc}")
            return {'CANCELLED'}
        info = dict(LAST_INFO)
        if info["joints"] == 0:
            self.report({'WARNING'}, "No curve ends lie within Tolerance of each other -- nothing joined")
            return {'CANCELLED'}
        from .primitives import write_spline_object
        cu = bpy.data.curves.new("JoinedCurve", 'CURVE')
        out = bpy.data.objects.new("JoinedCurve", cu)
        (context.collection or context.scene.collection).objects.link(out)
        write_spline_object(out, g[1])
        if features.history_enabled(context):
            features.record(out, "join_curves", inputs, params)
            for o in objs:
                o.hide_set(True)
                o.hide_render = True
            tail = " (sources kept hidden: history; Restore Inputs brings them back)"
        else:
            for o in objs:
                data = o.data
                bpy.data.objects.remove(o)
                if data is not None and data.users == 0:
                    bpy.data.curves.remove(data)
            tail = ""
        _select_only(context, [out])
        self.report({'INFO'}, _report_text(info) + tail)
        return {'FINISHED'}


class NPROTO_OT_unjoin_curves(bpy.types.Operator):
    """Split the selected curves into pieces: at the joints of a Join Curves
    result and at every corner. A curve object holding several splines comes
    apart into one object per spline"""
    bl_idname = "nproto.unjoin_curves"
    bl_label = "Unjoin Curves"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .ops import _select_only
        from .primitives import write_spline_object
        objs = [o for o in context.selected_objects if o.type == 'CURVE']
        if not objs:
            self.report({'WARNING'}, "Select curves")
            return {'CANCELLED'}
        made, untouched = [], []
        for obj in objs:
            recs = joint_records(obj)
            try:
                pieces = []
                for i, sp in enumerate(obj.data.splines):
                    pieces += unjoin_spline(sp, recs[i] if i < len(recs) and isinstance(recs[i], dict) else None)
            except Exception as exc:
                self.report({'ERROR'}, f"{obj.name}: {exc}")
                continue
            if len(pieces) < 2:
                untouched.append(obj.name)
                continue
            if features.has_history(obj):
                h = features.history(obj)
                if h and h["op"] == "join_curves":
                    hidden = [src.name for src, _ in h["inputs"] if src.hide_get()]
                    if hidden:
                        self.report({'INFO'}, "The Join's source curves stay hidden: " + ", ".join(dict.fromkeys(hidden)))
                features.clear_history(obj)
                self.report({'INFO'}, f"{obj.name}: construction history removed")
            write_spline_object(obj, pieces[0])
            made.append(obj)
            for k, sp in enumerate(pieces[1:], 1):
                cu = obj.data.copy()
                o = bpy.data.objects.new(f"{obj.name}_part{k}", cu)
                for coll in obj.users_collection:
                    coll.objects.link(o)
                o.matrix_world = obj.matrix_world.copy()
                write_spline_object(o, sp)
                made.append(o)
        if not made:
            self.report({'WARNING'}, "Nothing to unjoin: no joints or corners in " + ", ".join(untouched))
            return {'CANCELLED'}
        _select_only(context, made)
        self.report({'INFO'}, f"Unjoined into {len(made)} curves")
        return {'FINISHED'}


classes = (NPROTO_OT_join_curves, NPROTO_OT_unjoin_curves)
