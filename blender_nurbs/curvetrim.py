"""Split Curves / Trim Curves: cut curves where they cross other curves.

Both work on the exact Bezier spans of each spline (rational ones included),
so the pieces lie exactly on the original curve:

* Intersections -- every pair of spans is subdivided (convex-hull boxes)
  until the candidates are small, then refined by Newton on C1(s) = C2(t).
  A pair is an intersection when the curves come within Tolerance. With
  "Apparent" the curves are first flattened along the view direction, so
  curves that only cross on screen (at different depths) count too.
* Cutting -- de Casteljau subdivision of the span holding each cut; the
  pieces go back to Blender as endpoint + Bezier-knot NURBS (exact).
  A closed curve cut at k points gives k pieces (the piece across the seam
  stays one piece). Crossings at an open curve's own end point don't cut.

Split Curves cuts every selected curve at its crossings with the other
selected curves (or only with the active one, which stays whole) and makes
one object per piece. Trim Curves is interactive: click the piece of a
curve to delete; the cutting curves are the selected ones (all visible
curves if nothing is selected).
"""

import json

import numpy as np

from . import curvejoin, features, geom

TOL = 1e-4
MAX_CANDIDATES = 4000


# --------------------------------------------------------------------------
# Bezier span helpers (homogeneous rows x*w, y*w, z*w, w)
# --------------------------------------------------------------------------

def split_bezier(seg, t):
    """de Casteljau split of a homogeneous Bezier span at t -> (left, right)."""
    Q = np.array(seg, float)
    L, R = [Q[0]], [Q[-1]]
    for _ in range(len(Q) - 1):
        Q = (1.0 - t) * Q[:-1] + t * Q[1:]
        L.append(Q[0])
        R.append(Q[-1])
    return np.array(L), np.array(R[::-1])


def eval_d(seg, t):
    """Point and first derivative of a rational Bezier span at t."""
    Q = np.array(seg, float)
    p = len(Q) - 1
    for _ in range(p - 1):
        Q = (1.0 - t) * Q[:-1] + t * Q[1:]
    H = (1.0 - t) * Q[0] + t * Q[1]
    dH = p * (Q[1] - Q[0])
    C = H[:3] / H[3]
    return C, (dH[:3] - dH[3] * C) / H[3]


def _box(seg):
    P = seg[:, :3] / seg[:, 3:4]
    return P.min(0), P.max(0)


def flatten(spans, direction):
    """Spans projected along `direction` onto the plane through the origin (exact:
    the projection is linear, so it acts on the homogeneous CVs directly)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    L = np.eye(3) - np.outer(d, d)
    out = []
    for s in spans:
        n = s.copy()
        n[:, :3] = s[:, :3] @ L.T
        out.append(n)
    return out


def spans_size(spans):
    P = np.vstack([s[:, :3] / s[:, 3:4] for s in spans])
    return float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0


# --------------------------------------------------------------------------
# Intersections
# --------------------------------------------------------------------------

def _refine(a, b, s, t, iters=40):
    for _ in range(iters):
        P, dP = eval_d(a, s)
        Q, dQ = eval_d(b, t)
        J = np.stack([dP, -dQ], axis=1)
        step = np.linalg.lstsq(J, Q - P, rcond=None)[0]
        s2, t2 = float(np.clip(s + step[0], 0.0, 1.0)), float(np.clip(t + step[1], 0.0, 1.0))
        done = abs(s2 - s) + abs(t2 - t) < 1e-15
        s, t = s2, t2
        if done:
            break
    P, _ = eval_d(a, s)
    Q, _ = eval_d(b, t)
    return s, t, float(np.linalg.norm(P - Q)), 0.5 * (P + Q)


def _span_pair(a, b, tol, small, out):
    """Seeds (s, t) for intersections of spans a and b (recursive subdivision)."""
    stack = [(a, 0.0, 1.0, b, 0.0, 1.0, 0)]
    while stack:
        if len(out) > MAX_CANDIDATES:
            return
        A, a0, a1, B, b0, b1, depth = stack.pop()
        amin, amax = _box(A)
        bmin, bmax = _box(B)
        if np.any(amin > bmax + tol) or np.any(bmin > amax + tol):
            continue
        da, db = np.linalg.norm(amax - amin), np.linalg.norm(bmax - bmin)
        if depth >= 48 or (da <= small and db <= small):
            out.append((0.5 * (a0 + a1), 0.5 * (b0 + b1)))
            continue
        if da >= db:
            L, R = split_bezier(A, 0.5)
            m = 0.5 * (a0 + a1)
            stack += [(L, a0, m, B, b0, b1, depth + 1), (R, m, a1, B, b0, b1, depth + 1)]
        else:
            L, R = split_bezier(B, 0.5)
            m = 0.5 * (b0 + b1)
            stack += [(A, a0, a1, L, b0, m, depth + 1), (A, a0, a1, R, m, b1, depth + 1)]


def intersect(spans_a, spans_b, tol=TOL, same=False):
    """Intersections of two span lists: [(gA, gB, point)], g = span index + local t.
    same=True: the two lists are the same curve (self-intersections; the
    trivial hits of a span with itself / its neighbours are skipped)."""
    size = max(spans_size(spans_a), spans_size(spans_b))
    small = max(10 * tol, 2e-3 * size)
    boxes_b = [_box(s) for s in spans_b]
    hits = []
    for i, a in enumerate(spans_a):
        amin, amax = _box(a)
        for j, b in enumerate(spans_b):
            if same and j <= i + 1 and not (i == 0 and j == len(spans_b) - 1 and j > 1):
                continue
            bmin, bmax = boxes_b[j]
            if np.any(amin > bmax + tol) or np.any(bmin > amax + tol):
                continue
            seeds = []
            _span_pair(a, b, tol, small, seeds)
            for s, t in seeds:
                s, t, d, P = _refine(a, b, s, t)
                if d <= tol:
                    hits.append((i + s, j + t, P))
    # merge duplicates (several seeds per crossing, crossings on span ends)
    out = []
    for gA, gB, P in sorted(hits, key=lambda h: h[0]):
        if any(np.linalg.norm(P - q[2]) <= tol and abs(gA - q[0]) < 1e-3 for q in out[-4:]):
            continue
        out.append((gA, gB, P))
    return out


# --------------------------------------------------------------------------
# Cutting
# --------------------------------------------------------------------------

def is_closed(spans, cyclic=False):
    if cyclic:
        return True
    a = spans[0][0, :3] / spans[0][0, 3]
    b = spans[-1][-1, :3] / spans[-1][-1, 3]
    return len(spans) > 1 and np.linalg.norm(a - b) <= 1e-6 * spans_size(spans)


def clean_cuts(spans, cuts, closed, tol=TOL):
    """Sorted, de-duplicated cut parameters; crossings at an open curve's own
    end points are dropped (they would make zero-length pieces)."""
    n = len(spans)
    ends = [spans[0][0, :3] / spans[0][0, 3], spans[-1][-1, :3] / spans[-1][-1, 3]]
    out = []
    for g, P in sorted(cuts, key=lambda c: c[0]):
        g = float(g)
        if closed:
            g = g % n
            if n - g < 1e-9:
                g = 0.0
        elif g < 1e-9 or g > n - 1e-9 or any(np.linalg.norm(P - e) <= tol for e in ends):
            continue
        if out and (abs(g - out[-1][0]) < 1e-9 or np.linalg.norm(P - out[-1][1]) <= tol):
            continue
        out.append((g, P))
    if closed and len(out) > 1 and (n - out[-1][0] + out[0][0] < 1e-9 or
                                    np.linalg.norm(out[-1][1] - out[0][1]) <= tol):
        out.pop()
    return [g for g, _ in out]


def split_spans(spans, cuts, closed):
    """Cut a span list at sorted global parameters. Returns
    [(spans, g_start, g_end)]; on a closed curve the piece across the seam has
    g_start > g_end."""
    n = len(spans)
    eps = 1e-9
    if not cuts:
        return [([s.copy() for s in spans], 0.0, float(n))]
    pieces, cur, start = [], [], 0.0
    for i, seg in enumerate(spans):
        if cur and any(abs(g - i) < eps for g in cuts):
            pieces.append((cur, start, float(i)))
            cur, start = [], float(i)
        prev = 0.0
        seg = seg.copy()
        for g in cuts:
            t = g - i
            if eps < t < 1.0 - eps:
                L, seg = split_bezier(seg, (t - prev) / (1.0 - prev))
                cur.append(L)
                pieces.append((cur, start, g))
                cur, start, prev = [], g, t
        cur.append(seg)
    pieces.append((cur, start, float(n)))
    if closed:
        if abs(cuts[0]) < eps:     # a cut on the seam: the pieces already end there
            pass
        else:                      # join the last piece and the first across the seam
            last = pieces.pop()
            first = pieces[0]
            pieces[0] = (last[0] + first[0], last[1], first[2])
    return [p for p in pieces if p[0]]


def remove_piece(pieces, k, closed):
    """Span lists left after deleting piece k: the neighbours on each side stay
    joined (an open curve gives the part before and the part after k; a closed
    one the single arc from the end of k round to its start)."""
    lists = [p[0] for p in pieces]
    if closed:
        rest = lists[k + 1:] + lists[:k]
        return [[s for p in rest for s in p]] if rest else []
    out = [[s for p in lists[:k] for s in p], [s for p in lists[k + 1:] for s in p]]
    return [x for x in out if x]


def piece_at(pieces, g, n):
    """Index of the piece holding global parameter g."""
    for k, (_, a, b) in enumerate(pieces):
        if (a <= b and a - 1e-12 <= g <= b + 1e-12) or (a > b and (g >= a or g <= b)):
            return k
    return min(range(len(pieces)), key=lambda k: min(abs(g - pieces[k][1]), abs(g - pieces[k][2])))


def sample_spans(spans, per=24):
    """World points and their global parameters along a span list."""
    pts, gs = [], []
    for i, s in enumerate(spans):
        ts = np.linspace(0.0, 1.0, per + 1)
        if i:
            ts = ts[1:]
        pts.append(curvejoin.eval_span(s, ts))
        gs += list(i + ts)
    return np.vstack(pts), np.array(gs)


def piece_polyline(piece_spans, per=24):
    return sample_spans(piece_spans, per)[0]


# --------------------------------------------------------------------------
# Blender side
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, StringProperty  # noqa: E402
from mathutils import Vector  # noqa: E402

from . import draw  # noqa: E402


class CurveSet:
    """World-space spans of curve splines, keyed by (object name, spline index)."""

    def __init__(self, objs, direction=None, tol=TOL):
        self.tol = tol
        self.direction = direction
        self.local, self.world, self.flat, self.closed = {}, {}, {}, {}
        for o in objs:
            self.add(o)
        self._cuts = {}

    def add(self, obj):
        M = geom.world_matrix(obj)
        for i, sp in enumerate(obj.data.splines):
            try:
                loc = curvejoin.spline_spans(sp)
            except Exception as exc:  # unusable spline (single point...)
                print("NURBS curves: skipping", obj.name, i, exc)
                continue
            key = (obj.name, i)
            self.local[key] = loc
            self.world[key] = curvejoin.transform_spans(loc, M)
            self.flat[key] = flatten(self.world[key], self.direction) if self.direction is not None else self.world[key]
            self.closed[key] = is_closed(loc, sp.use_cyclic_u)
        self._cuts = {}

    def cuts(self, key, cutters):
        """Clean cut parameters of spline `key` against the cutter splines."""
        ck = (key, tuple(sorted(cutters)))
        if ck not in self._cuts:
            raw = []
            for c in cutters:
                if c == key or c not in self.flat:
                    continue
                for gA, _, _ in intersect(self.flat[key], self.flat[c], self.tol):
                    k = min(int(gA), len(self.world[key]) - 1)
                    raw.append((gA, eval_d(self.world[key][k], gA - k)[0]))
            self._cuts[ck] = clean_cuts(self.world[key], raw, self.closed[key], self.tol)
        return self._cuts[ck]


def _curve_objects(objs):
    return [o for o in objs if o.type == 'CURVE' and len(o.data.splines)]


def _view_direction(context):
    rv3d = getattr(context, "region_data", None)
    if rv3d is None:
        return None
    return tuple(rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)))


def _object_mode(context):
    """Leave a non-curve edit mode (curve Edit Mode is handled in place)."""
    if context.mode not in ('OBJECT', 'EDIT_CURVE') and context.active_object is not None:
        bpy.ops.object.mode_set(mode='OBJECT')


def in_edit(context):
    """Curve Edit Mode: the tools then edit the splines in place (writes through
    curve.splines go to the edit data, and edit-mode undo covers them); pieces
    stay splines of the same object instead of new objects."""
    return context.mode == 'EDIT_CURVE'


def edit_objects(context):
    return [o for o in getattr(context, "objects_in_mode", []) or [] if o.type == 'CURVE' and len(o.data.splines)]


def spline_selected(sp):
    if sp.type == 'BEZIER':
        return any(b.select_control_point or b.select_left_handle or b.select_right_handle for b in sp.bezier_points)
    return any(p.select for p in sp.points)


def end_selection(sp):
    """(start selected, end selected) of a spline in Edit Mode."""
    if sp.type == 'BEZIER':
        b = sp.bezier_points
        return bool(b[0].select_control_point), bool(b[-1].select_control_point)
    return bool(sp.points[0].select), bool(sp.points[-1].select)


def selected_splines(obj):
    """Indices of the splines with a selected point (all of them if none is)."""
    idx = [i for i, sp in enumerate(obj.data.splines) if spline_selected(sp)]
    return idx or list(range(len(obj.data.splines)))


def select_spline(sp, state=True):
    if sp.type == 'BEZIER':
        for b in sp.bezier_points:
            b.select_control_point = b.select_left_handle = b.select_right_handle = state
    else:
        for p in sp.points:
            p.select = state


def touch(obj):
    obj.data.update_tag()
    obj.update_tag()


def _drop_history(op, obj):
    if features.has_history(obj):
        features.clear_history(obj)
        op.report({'INFO'}, f"{obj.name}: construction history removed")
    if "nproto_joints" in obj:
        del obj["nproto_joints"]


def put_back(obj, index, pieces_local, same_object=False):
    """Put the pieces of spline `index` back: the first replaces the spline in
    `obj` (the splines after it are re-added so their order is kept: Blender has
    no spline insert). Every further piece becomes a new object, or with
    same_object (Edit Mode) another spline of `obj`. Returns the new objects."""
    from .primitives import _add_spline
    cu = obj.data
    after = []
    for sp in list(cu.splines)[index + 1:]:
        after.append(_spline_spec(sp))
    for sp in list(cu.splines)[index:]:
        cu.splines.remove(sp)
    specs = [curvejoin.spans_to_spec(p, simplify=True) for p in pieces_local]
    for sp in (specs if same_object else specs[:1]):
        _add_spline(cu, sp)
        if cu.is_editmode:
            select_spline(cu.splines[-1])
    for sp in after:
        _readd(cu, sp)
    touch(obj)
    if same_object:
        return []
    made = []
    for k, sp in enumerate(specs[1:], 1):
        name = f"{obj.name}_part{k}"
        nc = bpy.data.curves.new(name, 'CURVE')
        for attr in ("dimensions", "bevel_depth", "bevel_resolution", "resolution_u"):
            setattr(nc, attr, getattr(cu, attr))
        o = bpy.data.objects.new(name, nc)
        for coll in obj.users_collection:
            coll.objects.link(o)
        o.parent = obj.parent
        o.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
        o.matrix_basis = obj.matrix_basis.copy()
        _add_spline(nc, sp)
        made.append(o)
    return made


def snapshot(obj):
    return ([_spline_spec(sp) for sp in obj.data.splines],
            {k: obj[k] for k in obj.keys() if k.startswith("nproto")})


def restore(obj, snap):
    specs, props = snap
    cu = obj.data
    for sp in list(cu.splines):
        cu.splines.remove(sp)
    for d in specs:
        _readd(cu, d)
    for k in [k for k in obj.keys() if k.startswith("nproto")]:
        del obj[k]
    for k, v in props.items():
        obj[k] = v
    touch(obj)


def _spline_spec(sp):
    """Raw copy of a Blender spline (any type) so it can be re-added unchanged."""
    d = {"type": sp.type, "cyclic": sp.use_cyclic_u, "order": sp.order_u, "resolution": sp.resolution_u,
         "endpoint": sp.use_endpoint_u, "bezier": sp.use_bezier_u, "smooth": sp.use_smooth}
    if sp.type == 'BEZIER':
        d["bp"] = [(tuple(b.co), tuple(b.handle_left), tuple(b.handle_right), b.handle_left_type,
                    b.handle_right_type, b.tilt, b.radius) for b in sp.bezier_points]
    else:
        d["pts"] = [(tuple(p.co), p.tilt, p.radius) for p in sp.points]
    return d


def _readd(cu, d):
    s = cu.splines.new(d["type"])
    if d["type"] == 'BEZIER':
        s.bezier_points.add(len(d["bp"]) - 1)
        for b, (co, hl, hr, tl, tr, tilt, rad) in zip(s.bezier_points, d["bp"]):
            b.co, b.handle_left, b.handle_right = co, hl, hr
            b.handle_left_type, b.handle_right_type = tl, tr
            b.tilt, b.radius = tilt, rad
    else:
        s.points.add(len(d["pts"]) - 1)
        for p, (co, tilt, rad) in zip(s.points, d["pts"]):
            p.co, p.tilt, p.radius = co, tilt, rad
        s.order_u = d["order"]
        s.use_endpoint_u, s.use_bezier_u = d["endpoint"], d["bezier"]
    s.use_cyclic_u, s.resolution_u, s.use_smooth = d["cyclic"], d["resolution"], d["smooth"]


# --------------------------------------------------------------------------
# Split Curves
# --------------------------------------------------------------------------

class NPROTO_OT_split_curves(bpy.types.Operator):
    """Split the selected curves where they cross each other (exact: the pieces
    lie on the original curves). Each piece becomes its own curve object"""
    bl_idname = "nproto.split_curves"
    bl_label = "Split Curves"
    bl_options = {'REGISTER', 'UNDO'}

    cutter: EnumProperty(name="Cut By", default="ALL", items=[
        ("ALL", "Each Other", "Every selected curve is split where it crosses any other selected curve"),
        ("ACTIVE", "Active Curve", "Only the other curves are split, where they cross the active curve; "
                                   "the active curve stays whole")])
    tolerance: FloatProperty(name="Tolerance", default=TOL, min=1e-7, soft_max=0.01, precision=6, unit='LENGTH',
                             description="Curves closer than this count as crossing")
    apparent: BoolProperty(name="Apparent (View)", default=False,
                           description="Also cut where curves only cross as seen in the view (at different depths)")
    view_dir: FloatVectorProperty(size=3, options={'HIDDEN', 'SKIP_SAVE'})
    names: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    active: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        _object_mode(context)
        self.names = json.dumps([o.name for o in _curve_objects(context.selected_objects)])
        self.active = context.active_object.name if context.active_object else ""
        d = _view_direction(context)
        if d:
            self.view_dir = d
        return self.execute(context)

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        if in_edit(context):
            lay.label(text="Edit Mode: selected splines cut by every visible curve", icon='INFO')
        else:
            lay.row().prop(self, "cutter", expand=True)
        lay.prop(self, "tolerance")
        lay.prop(self, "apparent")

    def execute(self, context):
        from .ops import _select_only
        if in_edit(context):
            return self._execute_edit(context)
        if not self.names:
            self.names = json.dumps([o.name for o in _curve_objects(context.selected_objects)])
            self.active = context.active_object.name if context.active_object else ""
        objs = _curve_objects([bpy.data.objects[n] for n in json.loads(self.names) if n in bpy.data.objects])
        if self.cutter == "ACTIVE" and self.active not in [o.name for o in objs]:
            self.report({'WARNING'}, "Cut By Active: make one of the selected curves active")
            return {'CANCELLED'}
        nsplines = sum(len(o.data.splines) for o in objs)
        if nsplines < 2:
            self.report({'WARNING'}, "Select two or more curves (Split Curves cuts them where they cross)")
            return {'CANCELLED'}
        direction = tuple(self.view_dir) if self.apparent and Vector(self.view_dir).length > 0.5 else None
        cs = CurveSet(objs, direction, self.tolerance)
        keys = list(cs.world)
        if self.cutter == "ACTIVE":
            cutters = [k for k in keys if k[0] == self.active]
            targets = [k for k in keys if k[0] != self.active]
        else:
            cutters, targets = keys, keys
        plan = {}
        for key in targets:
            cuts = cs.cuts(key, [c for c in cutters if c != key])
            if cuts:
                plan[key] = split_spans(cs.local[key], cuts, cs.closed[key])
        if not plan:
            self.report({'WARNING'}, "The curves don't cross (within Tolerance) -- nothing split"
                        + ("" if self.apparent else "; try Apparent (View) for curves at different depths"))
            return {'CANCELLED'}
        made, npieces = [], 0
        # highest spline index first, so the indices still to do stay valid
        for (name, idx) in sorted(plan, key=lambda k: (k[0], -k[1])):
            obj = bpy.data.objects[name]
            pieces = plan[(name, idx)]
            _drop_history(self, obj)
            made += put_back(obj, idx, [p[0] for p in pieces])
            npieces += len(pieces)
            if obj not in made:
                made.append(obj)
        _select_only(context, made)
        self.report({'INFO'}, f"Split {len(plan)} curve(s) into {npieces} pieces")
        return {'FINISHED'}

    def _execute_edit(self, context):
        """Edit Mode: the selected splines (all, if none is selected) of the
        objects being edited are cut by every visible curve; the pieces stay
        splines of the same object."""
        objs = edit_objects(context)
        if not objs:
            self.report({'WARNING'}, "No curve in Edit Mode")
            return {'CANCELLED'}
        direction = tuple(self.view_dir) if self.apparent and Vector(self.view_dir).length > 0.5 else None
        names = {o.name for o in objs}
        others = [o for o in _curve_objects(context.visible_objects) if o.name not in names]
        cs = CurveSet(objs + others, direction, self.tolerance)
        cutters = list(cs.world)
        targets = [(o.name, i) for o in objs for i in selected_splines(o) if (o.name, i) in cs.world]
        plan = {}
        for key in targets:
            cuts = cs.cuts(key, [c for c in cutters if c != key])
            if cuts:
                plan[key] = split_spans(cs.local[key], cuts, cs.closed[key])
        if not plan:
            self.report({'WARNING'}, "The selected splines don't cross another curve (within Tolerance) -- nothing split")
            return {'CANCELLED'}
        npieces = 0
        for obj in objs:
            for sp in obj.data.splines:
                select_spline(sp, False)
        for (name, idx) in sorted(plan, key=lambda k: (k[0], -k[1])):
            obj = bpy.data.objects[name]
            _drop_history(self, obj)
            put_back(obj, idx, [p[0] for p in plan[(name, idx)]], same_object=True)
            npieces += len(plan[(name, idx)])
        self.report({'INFO'}, f"Split {len(plan)} spline(s) into {npieces} (selected)")
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Trim Curves (interactive)
# --------------------------------------------------------------------------

class NPROTO_OT_trim_curves(bpy.types.Operator):
    """Click the parts of curves to delete; curves are cut where they cross the
    cutting curves (the selected curves, or all visible curves if none are
    selected). The rest of each curve stays exact"""
    bl_idname = "nproto.trim_curves"
    bl_label = "Trim Curves"
    bl_options = {'UNDO'}

    tolerance: FloatProperty(name="Tolerance", default=TOL, min=1e-7, precision=6, unit='LENGTH',
                             options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def execute(self, context):
        self.report({'WARNING'}, "Trim Curves is interactive: start it from the panel or menu")
        return {'CANCELLED'}

    def invoke(self, context, event):
        _object_mode(context)
        vis = _curve_objects(context.visible_objects)
        if not vis:
            self.report({'WARNING'}, "No visible curves to trim")
            return {'CANCELLED'}
        self._edit = in_edit(context)
        self._edit_names = {o.name for o in edit_objects(context)} if self._edit else None
        sel = [] if self._edit else _curve_objects(context.selected_objects)
        self._cutter_names = {o.name for o in sel} if sel else None   # None: every visible curve cuts
        self._apparent = False
        self._undo = []          # [(obj name, curve copy, saved props, made names)]
        self._hover = None
        self._load(context)
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["hover"] = None
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {'RUNNING_MODAL'}

    # -- state ------------------------------------------------------------
    def _load(self, context):
        vis = _curve_objects(context.visible_objects)
        direction = _view_direction(context) if self._apparent else None
        self._cs = CurveSet(vis, direction, self.tolerance)
        self._samples = {k: sample_spans(w, 16) for k, w in self._cs.world.items()
                         if self._edit_names is None or k[0] in self._edit_names}   # what can be trimmed
        self._cutters = [k for k in self._cs.world
                         if self._cutter_names is None or k[0] in self._cutter_names]
        # the cutting curves themselves are drawn highlighted
        draw.pick_overlay["picked"] = ([] if self._cutter_names is None else
                                       [[Vector(p) for p in self._samples[k][0]] for k in self._cutters])

    def _status(self, context, extra=""):
        from . import hints
        who = "all visible curves" if self._cutter_names is None else "the selected curves"
        rows = [("Trim", [hints.toggle("V", "apparent (as seen in this view)", self._apparent),
                          hints.key("Backspace", "undo"), hints.key("Enter", "done")])]
        if self._edit:
            rows.append(("", [hints.note("Edit Mode: only the curves being edited are trimmed.")]))
        hints.show(context, "Trim Curves", [f"click the part to delete (cut by {who})", extra.strip(" ·")], rows,
                   [("MOUSE_LMB", "Trim"), ("MOUSE_RMB", "Done"), ("EVENT_ESC", "Undo all")])

    def _hit(self, context, event, max_px=18.0):
        from bpy_extras import view3d_utils
        area = context.area
        region = next((r for r in area.regions if r.type == 'WINDOW'), context.region)
        rv3d = getattr(area.spaces.active, "region_3d", None) or context.region_data
        m = np.array([event.mouse_x - region.x, event.mouse_y - region.y], float)
        best = None
        for key, (pts, gs) in self._samples.items():
            q = [view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(p)) for p in pts]
            for k in range(len(q) - 1):
                if q[k] is None or q[k + 1] is None:
                    continue
                a, b = np.array(q[k]), np.array(q[k + 1])
                seg = b - a
                L = seg @ seg
                t = 0.0 if L == 0 else float(np.clip((m - a) @ seg / L, 0, 1))
                d = float(np.linalg.norm(a + t * seg - m))
                if d <= max_px and (best is None or d < best[0]):
                    best = (d, key, gs[k] + t * (gs[k + 1] - gs[k]))
        if best is None:
            return None
        _, key, g = best
        cuts = self._cs.cuts(key, [c for c in self._cutters if c != key])
        if not cuts:
            return {"key": key, "pieces": None, "index": None,
                    "poly": [Vector(p) for p in self._samples[key][0]]}
        pieces = split_spans(self._cs.world[key], cuts, self._cs.closed[key])
        k = piece_at(pieces, g, len(self._cs.world[key]))
        return {"key": key, "g": g, "cuts": cuts, "index": k, "pieces": pieces,
                "poly": [Vector(p) for p in piece_polyline(pieces[k][0], 16)]}

    def _hover_update(self, context, event):
        self._hover = self._hit(context, event)
        h = self._hover
        draw.pick_overlay["hover"] = h["poly"] if h else None
        extra = ""
        if h and h["pieces"] is None:
            extra = f"   [{h['key'][0]}: no crossing with a cutting curve]"
        self._status(context, extra)
        context.area.tag_redraw()

    def _trim(self, context, event):
        h = self._hit(context, event)
        if h is None:
            return
        if h["pieces"] is None:
            self.report({'WARNING'}, f"{h['key'][0]} doesn't cross a cutting curve: nothing to trim")
            return
        name, idx = h["key"]
        obj = bpy.data.objects[name]
        local = split_spans(self._cs.local[h["key"]], h["cuts"], self._cs.closed[h["key"]])
        keep = remove_piece(local, h["index"], self._cs.closed[h["key"]])
        self._undo.append([name, snapshot(obj), None, []])
        _drop_history(self, obj)
        if keep:
            made = put_back(obj, idx, keep, same_object=self._edit)
        else:
            from_spline = obj.data.splines[idx]
            obj.data.splines.remove(from_spline)
            touch(obj)
            made = []
        self._undo[-1][3] = [o.name for o in made]
        if self._cutter_names is not None and name in self._cutter_names:
            self._cutter_names |= {o.name for o in made}
        self._load(context)
        self._hover_update(context, event)

    def _undo_last(self, context):
        if not self._undo:
            return
        name, snap, _, made = self._undo.pop()
        for n in made:
            o = bpy.data.objects.get(n)
            if o is not None:
                d = o.data
                bpy.data.objects.remove(o)
                if d is not None and d.users == 0:
                    bpy.data.curves.remove(d)
        obj = bpy.data.objects.get(name)
        if obj is not None:
            restore(obj, snap)
        if self._cutter_names is not None:
            self._cutter_names -= set(made)
        self._load(context)

    def _end(self, context):
        from . import hints
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["hover"] = None
        hints.clear(context)

    def modal(self, context, event):
        from .ops import PickMixin
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        from . import hints
        if hints.handle(event):
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event):
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            self._hover_update(context, event)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._trim(context, event)
        elif event.type == 'V' and event.value == 'PRESS':
            self._apparent = not self._apparent
            self._load(context)
            self._hover_update(context, event)
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS':
            self._undo_last(context)
            self._hover_update(context, event)
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE', 'RIGHTMOUSE'} and event.value == 'PRESS':
            n = len(self._undo)
            self._end(context)
            if not n:
                return {'CANCELLED'}
            self.report({'INFO'}, f"Trimmed {n} piece(s)")
            return {'FINISHED'}
        elif event.type == 'ESC' and event.value == 'PRESS':
            while self._undo:
                self._undo_last(context)
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}


classes = (NPROTO_OT_split_curves, NPROTO_OT_trim_curves)
