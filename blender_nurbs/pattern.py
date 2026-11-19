"""Pattern: linear and circular copies of NURBS objects and curves.

Blender's Array modifier copies only the display mesh: the copies have no
B-rep (no STEP export, no CV net, nothing to pick, trim or join). A pattern
makes each copy an object of its own carrying the source's exact geometry.

A B-rep lives in its object's local space and the object's matrix places it,
so a copy is the source's B-rep text under another matrix: copy k sits at
T_k @ M_src (T_k: the pattern's k-th move or turn, world). Nothing is
re-tessellated when the copies are made: they take the source's mesh as is.

With construction history each copy keeps a "pattern" feature (input: the
source; params: the pattern and the copy's index). Its builder returns the
source's geometry at T_k @ M_src in world space, and the copy's placement
(features.PROP_PLACE = inv(the matrix it was made at)) brings that into its
own local space. So when the source changes shape or moves, the copies follow;
a copy moved by hand keeps its offset. All copies of one run share a group id:
the Feature panel's Edit Pattern changes them together (more or fewer copies,
another spacing) and Select Pattern selects them.

Index 0 is the source itself; a linear grid numbers copy (i, j) i + j * count.
"""

import json
import math
import uuid

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty
from mathutils import Matrix, Vector

from . import draw, features, geom, hints, store
from .valuedrag import ValueDrag, angle_pick, rotate_points

OP = "pattern"
MAX_COPIES = 2000           # one run makes at most this many objects
PREVIEW_LINES = 40000       # more preview polylines than this: each copy drawn as its box
PREVIEW_TRIS = 60000        # more preview triangles than this: lines only

MODE_ITEMS = [("LINEAR", "Linear", "Copies in a row (or a grid) along a direction"),
              ("CIRCULAR", "Circular", "Copies turned around an axis")]
LIN_AXIS_ITEMS = [("X", "World X", ""), ("Y", "World Y", ""), ("Z", "World Z", ""),
                  ("LOCAL_X", "Local X", ""), ("LOCAL_Y", "Local Y", ""), ("LOCAL_Z", "Local Z", ""),
                  ("PICKED", "Picked", "Along a line or edge picked for it (A in the tool): start to end; "
                                       "a closed curve: along its normal")]
ROWS_AXIS_ITEMS = LIN_AXIS_ITEMS[:6]
ROT_AXIS_ITEMS = LIN_AXIS_ITEMS[:6] + [
    ("PICKED", "Picked", "A line, straight edge or circle picked for it (A in the tool): a line is the axis "
                         "from its start to its end, a circle's axis runs through its centre")]
CENTER_ITEMS = [("CURSOR", "3D Cursor", "The axis runs through the 3D cursor (where it is when the pattern is made)"),
                ("WORLD", "World Origin", ""),
                ("OBJECT", "Object Origin", "Each object turns about its own origin (copies follow it when it moves)")]
SPACE_ITEMS = [("SPACING", "Spacing", "The distance from one copy to the next"),
               ("EXTENT", "Length", "The distance from the first copy to the last")]


# --------------------------------------------------------------------------
# The pattern's transforms (pure: params + source -> world 4x4 matrices)
# --------------------------------------------------------------------------

def _translation(v):
    T = np.eye(4)
    T[:3, 3] = np.asarray(v, float)
    return T


def _rotation(z, deg):
    """3x3: deg degrees about unit z, right-handed (as rotate_points)."""
    z = np.asarray(z, float)
    z = z / (np.linalg.norm(z) or 1.0)
    a = math.radians(deg)
    K = np.array([[0.0, -z[2], z[1]], [z[2], 0.0, -z[0]], [-z[1], z[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def _unit(v, fallback=(0.0, 0.0, 1.0)):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.asarray(fallback, float)


def world_matrix(obj):
    return np.array(geom.world_matrix(obj), float)


def linear_dir(p, src, which="axis"):
    """Unit direction (world) of the row (which="axis") or of the second row (axis_b)."""
    ax = p.get(which, "X" if which == "axis" else "Y")
    if ax == "PICKED":
        return _unit(p.get("axis_dir", (1.0, 0.0, 0.0)), (1.0, 0.0, 0.0))
    return _unit(geom.axis_vector(ax, src))


def circ_axis(p, src):
    """(point, unit direction) of a circular pattern's axis (world)."""
    ax = p.get("rot_axis", "Z")
    if ax == "PICKED":
        return np.asarray(p.get("axis_point", (0.0, 0.0, 0.0)), float), _unit(p.get("axis_dir", (0.0, 0.0, 1.0)))
    z = _unit(geom.axis_vector(ax, src))
    if p.get("center", "CURSOR") == "OBJECT":
        return world_matrix(src)[:3, 3].copy(), z
    return np.asarray(p.get("center_point", (0.0, 0.0, 0.0)), float), z


def step(distance, n, space):
    """The move from one copy to the next."""
    if space == "EXTENT":
        return distance / max(n - 1, 1)
    return distance


def angle_step(angle, n):
    """The turn from one copy to the next: a full turn spreads n copies evenly
    (the last is not on the first), a part of one has copies at both ends."""
    if abs(abs(angle) - 360.0) < 1e-6:
        return angle / n
    return angle / max(n - 1, 1)


def total_count(p):
    n = max(int(p.get("count", 2)), 1)
    if p.get("mode", "LINEAR") == "LINEAR":
        return n * max(int(p.get("count_b", 1)), 1)
    return n


def anchor(obj):
    """Centre of obj's box (world): what a copy that is not turned keeps its place by."""
    M = world_matrix(obj)
    P = _local_points(obj)
    if P is None or not len(P):
        return M[:3, 3].copy()
    c = 0.5 * (P.min(axis=0) + P.max(axis=0))
    return M[:3, :3] @ c + M[:3, 3]


def _local_points(obj):
    if obj.type == 'MESH':
        me = obj.data
        n = len(me.vertices)
        if not n:
            return None
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        return co.reshape(-1, 3).astype(float)
    if obj.type == 'CURVE':
        pts = []
        for sp in obj.data.splines:
            if sp.type == 'BEZIER':
                pts += [tuple(b.co) for b in sp.bezier_points]
            else:
                pts += [tuple(q.co[:3]) for q in sp.points]
        return np.array(pts, float) if pts else None
    return None


def copy_transform(p, src, index, _cache=None):
    """World 4x4 that moves the source to copy `index` (None: no such copy)."""
    n = max(int(p.get("count", 2)), 1)
    if p.get("mode", "LINEAR") == "LINEAR":
        nb = max(int(p.get("count_b", 1)), 1)
        if not 0 < index < n * nb:
            return None
        i, j = index % n, index // n
        space = p.get("space", "SPACING")
        s = step(float(p.get("distance", 0.0)), n, space)
        v = i * s * linear_dir(p, src)
        if j:
            v = v + j * step(float(p.get("distance_b", 0.0)), nb, space) * linear_dir(p, src, "axis_b")
        return _translation(v)
    if not 0 < index < n:
        return None
    O, z = circ_axis(p, src)
    R = _rotation(z, index * angle_step(float(p.get("angle", 360.0)), n))
    if p.get("rotate", True):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = O - R @ O
        return T
    a = anchor(src) if _cache is None else _cache.setdefault("anchor", anchor(src))
    return _translation(R @ (a - O) + O - a)          # moved round the axis, not turned


def copy_transforms(p, src):
    """[(index, world 4x4)] of every copy (the source, index 0, left out)."""
    cache = {}
    out = []
    for k in range(1, total_count(p)):
        T = copy_transform(p, src, k, cache)
        if T is not None:
            out.append((k, T))
    return out


def circle_or_line(pts):
    """(point, unit direction, closed) of a picked polyline: an open one runs
    from its start to its end; a closed one (a circle, a hole's edge) gives its
    centre (circle fit in its plane) and its normal (right-handed with its run)."""
    P = np.asarray([tuple(q) for q in pts], float)
    size = float(np.ptp(P, axis=0).max()) if len(P) else 0.0
    if len(P) < 2 or size < 1e-12:
        return None
    closed = float(np.linalg.norm(P[-1] - P[0])) <= 1e-6 * size
    if not closed:
        return P[0], _unit(P[-1] - P[0]), False
    Q = P[:-1] if len(P) > 3 else P
    c = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - c)
    n = vt[2]
    x, y = vt[0], vt[1]
    u, v = (Q - c) @ x, (Q - c) @ y
    A = np.column_stack([u, v, np.ones_like(u)])          # u^2 + v^2 = 2a u + 2b v + k
    try:
        sol = np.linalg.lstsq(A, u * u + v * v, rcond=None)[0]
        c = c + 0.5 * sol[0] * x + 0.5 * sol[1] * y
    except np.linalg.LinAlgError:
        pass
    turn = sum(float(np.cross(a - c, b - c) @ n) for a, b in zip(P[:-1], P[1:]))
    return c, (n if turn >= 0.0 else -n), True


# --------------------------------------------------------------------------
# History: a copy follows its source
# --------------------------------------------------------------------------

@features.builder(OP)
def build_pattern(inputs, p):
    """The source's geometry at copy p["index"], in world space (the copy's
    placement takes it into its own local space). Curves: rewritten in place."""
    src, _ = inputs[0]
    T = copy_transform(p, src, int(p.get("index", 1)))
    if T is None:
        raise RuntimeError("the pattern has no copy %s" % p.get("index"))
    W = T @ world_matrix(src)
    if src.type == 'CURVE':
        return ("inplace", lambda obj: _write_curve(obj, src, features.placement(obj), W))
    if not store.is_nurbs(src):
        raise RuntimeError(f"{src.name} is not a NURBS object")
    if store.is_natural(src) and store.get_brep(src).face_count() == 1:
        return _moved_surface(store.get_surfaces(src)[0], W)   # an untrimmed surface stays one (Edge mode, CVs)
    b = store.brep_from_text(src[store.PROP_BREP])
    store.transform(b, W)
    return ("brep_text", store.brep_to_text(b))


def _moved_surface(surf, W):
    """A free surface: surf's CVs moved by the affine W (exact, rational ones
    too: an affine move commutes with the weights). A surface of a Brep's face
    can't be moved apart from its Brep."""
    _, ext = features._k()
    d = ext.surface_cvs(surf)
    P = np.array(d["points"], float).reshape(-1, 3) @ W[:3, :3].T + W[:3, 3]
    return ext.surface_from_cvs(d["degree_u"], d["degree_v"], [tuple(map(float, q)) for q in P],
                                list(d["knots_u"]), list(d["knots_v"]),
                                [float(x) for x in d["weights"]] if d.get("weights") else None)


def _write_curve(obj, src, A, W):
    """obj's curve data: src's, moved by A @ W (its local space when obj's
    placement is A). As made: A @ W is the identity, the data a plain copy."""
    L = W if A is None else np.asarray(A, float) @ W
    new = src.data.copy()
    if np.abs(L - np.eye(4)).max() > 1e-12:
        new.transform(Matrix(L.tolist()))
    old = obj.data
    obj.data = new
    if old is not None and old != new:
        name = old.name
        if old.users == 0:
            bpy.data.curves.remove(old)
            new.name = name


def pattern_of(obj):
    """(history dict, params) of a pattern copy, or None."""
    if obj is None or features.PROP_HIST not in obj:
        return None
    try:
        raw = json.loads(obj[features.PROP_HIST])
    except ValueError:
        return None
    if raw.get("op") != OP:
        return None
    return raw, raw.get("params", {})


def group_copies(group, objects=None):
    """Every copy of a pattern group: [(copy, source, index)]."""
    out = []
    if not group:
        return out
    for o in (objects if objects is not None else bpy.data.objects):
        pg = pattern_of(o)
        if pg is None or pg[1].get("group") != group:
            continue
        src = o.get("nproto_in_0")
        if src is not None:
            out.append((o, src, int(pg[1].get("index", 0))))
    return out


def group_sources(group):
    seen = []
    for _, src, _ in group_copies(group):
        if src not in seen:
            seen.append(src)
    return seen


# --------------------------------------------------------------------------
# Making the copies
# --------------------------------------------------------------------------

def usable(obj):
    return obj is not None and (store.is_nurbs(obj) or obj.type == 'CURVE')


def _link_like(obj, src, context):
    colls = [c for c in src.users_collection if c.name != features.HISTORY_COLL]
    if not colls:
        colls = [context.collection or context.scene.collection]
    for c in colls:
        c.objects.link(obj)


def _new_copy(context, src, F, index):
    """A new object holding src's geometry at world matrix F (its mesh is
    src's, copied: the same B-rep in the same local space looks the same)."""
    name = f"{src.name}_p{index}"
    if src.type == 'CURVE':
        obj = bpy.data.objects.new(name, src.data.copy())
    else:
        obj = bpy.data.objects.new(name, src.data.copy())
        for k in (store.PROP_BREP, store.PROP_DISP, store.PROP_NATURAL, store.PROP_UNTRIM,
                  store.PROP_UNTRIM + "_natural"):
            if k in src:
                obj[k] = src[k]
        if store.PROP_NATURAL not in obj:
            obj[store.PROP_NATURAL] = 1 if store.is_natural(src) else 0
        obj[store.PROP_REV] = store._new_rev(obj)
    _link_like(obj, src, context)
    obj.matrix_world = Matrix(F.tolist())
    obj.color = src.color
    if obj.type == 'MESH':
        from .pivot import remember
        remember(obj)
    return obj


def _ensure_matrix(obj, F):
    """obj's matrix set to F (unparented, as a copy is made)."""
    if obj.parent is not None:
        obj.parent = None
    obj.matrix_basis = Matrix(F.tolist())


def make_pattern(context, sources, p, group=None):
    """Copies of each source by the pattern p. group: an existing pattern's
    group id -- its copies are moved and rebuilt where an index still exists,
    made where it is new, deleted where it is gone. -> (copies, errors)."""
    hist = features.history_enabled(context)
    group = group or uuid.uuid4().hex[:12]
    p = dict(p, group=group)
    existing = {}
    for o, src, k in group_copies(group):
        existing[(src.name, k)] = o
    made, errors = [], []
    for src in sources:
        M = world_matrix(src)
        for k, T in copy_transforms(p, src):
            F = T @ M
            obj = existing.pop((src.name, k), None)
            try:
                if obj is None:
                    obj = _new_copy(context, src, F, k)
                    fresh = True
                else:
                    _ensure_matrix(obj, F)
                    fresh = False
                if hist:
                    features.record(obj, OP, [(src, "OBJ")], dict(p, index=k))
                    features.set_placement(obj, np.linalg.inv(F))
                    if fresh:
                        features.mark_built(obj)     # its geometry is what the feature gives
                    else:
                        features.rebuild(obj, context, force=True)
                elif not fresh:
                    _rewrite_plain(obj, src, context)
                made.append(obj)
            except Exception as exc:
                errors.append(f"{src.name} copy {k}: {exc}")
    for obj in existing.values():           # copies the pattern no longer has
        _remove(obj)
    return made, errors


def _rewrite_plain(obj, src, context):
    """A copy without history given src's geometry again (in its local space)."""
    if src.type == 'CURVE':
        _write_curve(obj, src, None, np.eye(4))
    else:
        store.set_geometry_text(obj, src[store.PROP_BREP], context, keep_untrimmed=False)
        obj[store.PROP_NATURAL] = 1 if store.is_natural(src) else 0


def _remove(obj):
    data = obj.data
    bpy.data.objects.remove(obj)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
        elif isinstance(data, bpy.types.Curve):
            bpy.data.curves.remove(data)


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------

_DIGITS = {f"NUMPAD_{i}": str(i) for i in range(10)} | {
    k: str(i) for i, k in enumerate(("ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"))}


class NPROTO_OT_pattern(ValueDrag, bpy.types.Operator):
    """Copies of the selected NURBS objects and curves in a row, a grid or around
    an axis -- real NURBS objects (Blender's Array copies only the mesh). From
    the UI the copies show as a preview with a handle: drag it, type a value,
    arrow keys for the count; click away or Enter to make them"""
    bl_idname = "nproto.pattern"
    bl_label = "Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Pattern", items=MODE_ITEMS, default="LINEAR")
    count: IntProperty(name="Count", default=4, min=2, soft_max=50, max=1000,
                       description="How many there are, the original included")
    # linear
    distance: FloatProperty(name="Spacing", default=0.0, unit='LENGTH',
                            description="Spacing: from one copy to the next. Length: from the first to the last "
                                        "(negative: the other way)")
    space: EnumProperty(name="Distance", items=SPACE_ITEMS, default="SPACING")
    axis: EnumProperty(name="Direction", items=LIN_AXIS_ITEMS, default="X")
    count_b: IntProperty(name="Rows", default=1, min=1, soft_max=50, max=1000,
                         description="A grid: this many rows of the row, along the second direction")
    distance_b: FloatProperty(name="Row Spacing", default=0.0, unit='LENGTH',
                              description="Between the rows (Length: from the first row to the last)")
    axis_b: EnumProperty(name="Row Direction", items=ROWS_AXIS_ITEMS, default="Y")
    # circular
    angle: FloatProperty(name="Angle", default=360.0, min=-360.0, max=360.0,
                         description="The turn the copies spread over: 360 spreads them evenly round, "
                                     "less has copies at both ends")
    rot_axis: EnumProperty(name="Axis", items=ROT_AXIS_ITEMS, default="Z")
    center: EnumProperty(name="Center", items=CENTER_ITEMS, default="CURSOR",
                         description="Where a world or local axis runs through (a picked one has its own place)")
    center_point: FloatVectorProperty(name="Center Point", size=3, subtype='XYZ', unit='LENGTH',
                                      options={'HIDDEN'})
    rotate: BoolProperty(name="Rotate Copies", default=True,
                         description="Turn each copy with the pattern (off: moved round the axis, kept upright)")
    # a picked line / circle (both modes)
    axis_point: FloatVectorProperty(name="Axis Point", size=3, subtype='XYZ', unit='LENGTH',
                                    description="Picked: a point the axis runs through (world)")
    axis_dir: FloatVectorProperty(name="Axis Direction", size=3, subtype='XYZ', default=(0.0, 0.0, 1.0),
                                  description="Picked: the direction / axis (world)")
    objects: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                            description="JSON list of the object names to copy (empty: the selection)")
    group: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="An existing pattern to change (its group id)")

    # -- the live value stage (valuedrag.py) ----------------------------------
    angle_fields = ("angle",)
    start_value = 0.0           # linear: the copies on the original until the arrow is dragged (L: the last spacing)
    AXES = {"x": "X", "y": "Y", "z": "Z"}

    @property
    def drag_fields(self):
        if self.mode == "CIRCULAR":
            return ("angle",)
        return ("distance", "distance_b") if self.count_b > 1 else ("distance",)

    def params(self):
        p = {"mode": self.mode, "count": self.count}
        if self.mode == "LINEAR":
            p.update(distance=self.distance, space=self.space, axis=self.axis, count_b=self.count_b,
                     distance_b=self.distance_b, axis_b=self.axis_b)
            if self.axis == "PICKED":
                p["axis_dir"] = list(self.axis_dir)
        else:
            p.update(angle=self.angle, rot_axis=self.rot_axis, center=self.center, rotate=self.rotate,
                     center_point=list(self.center_point))
            if self.rot_axis == "PICKED":
                p.update(axis_point=list(self.axis_point), axis_dir=list(self.axis_dir))
        return p

    def _load(self, p):
        """The operator's settings from a pattern's params (Edit Pattern)."""
        for k in ("mode", "count", "distance", "space", "axis", "count_b", "distance_b", "axis_b", "angle",
                  "rot_axis", "center", "rotate"):
            if k in p:
                setattr(self, k, p[k])
        for k in ("center_point", "axis_point", "axis_dir"):
            if k in p:
                setattr(self, k, tuple(float(x) for x in p[k]))

    def _sources(self, context):
        if self.objects:
            names = json.loads(self.objects)
            return [o for o in (bpy.data.objects.get(n) for n in names) if usable(o)]
        if self.group:
            return [o for o in group_sources(self.group) if usable(o)]
        return [o for o in context.selected_objects if usable(o)]

    def _resolve_center(self, context):
        if self.mode == "CIRCULAR" and self.rot_axis != "PICKED":
            if self.center == "CURSOR":
                self.center_point = tuple(context.scene.cursor.location)
            elif self.center == "WORLD":
                self.center_point = (0.0, 0.0, 0.0)

    # -- start ---------------------------------------------------------------
    def invoke(self, context, event):
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        if self.group:
            copies = group_copies(self.group)
            if not copies:
                self.report({'WARNING'}, "That pattern has no copies left")
                return {'CANCELLED'}
            act = context.view_layer.objects.active
            ref = next((c for c in copies if c[0] == act), copies[0])
            self._load(pattern_of(ref[0])[1])
        srcs = self._sources(context)
        if not srcs:
            self.report({'WARNING'}, "Select NURBS objects or curves to copy "
                                     "(for plain meshes Blender's Array modifier does it)")
            return {'CANCELLED'}
        self.objects = json.dumps([o.name for o in srcs])
        self._resolve_center(context)
        if self.group:
            # an edit starts from the pattern as it is, not from nothing
            self._keep = {f: getattr(self, f) for f in ("distance", "distance_b")}
        return self.ask(context)

    def ask(self, context):
        r = super().ask(context)
        keep = getattr(self, "_keep", None)
        if keep and getattr(self, "_dragging", False):
            for f, v in keep.items():
                setattr(self, f, v)
            self._drag_refresh(context)
        return r

    def _fallback(self, context):
        return context.window_manager.invoke_props_dialog(self, width=340, title=self.bl_label,
                                                          confirm_text="Make Pattern")

    def prepare_value(self, context):
        pass

    # -- make it --------------------------------------------------------------
    def execute(self, context):
        from .ops import _kernel_ok
        if not _kernel_ok(self):
            return {'CANCELLED'}
        srcs = self._sources(context)
        if not srcs:
            self.report({'WARNING'}, "Select NURBS objects or curves to copy")
            return {'CANCELLED'}
        if not self.properties.is_property_set("center_point"):
            self._resolve_center(context)            # (a script: the cursor as it is now)
        p = self.params()
        n = total_count(p) - 1
        if n * len(srcs) > MAX_COPIES:
            self.report({'ERROR'}, f"{n * len(srcs)} copies: at most {MAX_COPIES} in one pattern")
            return {'CANCELLED'}
        if self.mode == "LINEAR" and self.count_b > 1:
            d0, d1 = linear_dir(p, srcs[0]), linear_dir(p, srcs[0], "axis_b")
            if abs(float(d0 @ d1)) > 1.0 - 1e-9:
                self.report({'WARNING'}, "The rows run along the row itself: pick another row direction")
        act = context.view_layer.objects.active
        made, errors = make_pattern(context, srcs, p, self.group or None)
        for e in errors:
            self.report({'ERROR'}, "Pattern: " + e)
        if not made and errors:
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for o in srcs + made:
            try:
                o.select_set(True)
            except RuntimeError:            # not in this view layer
                pass
        keep = act if (self.group and act in made) else srcs[0]
        context.view_layer.objects.active = keep
        self.report({'INFO'}, f"Pattern: {len(made)} cop{'y' if len(made) == 1 else 'ies'}")
        return {'FINISHED'}

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.row().prop(self, "mode", expand=True)
        lay.prop(self, "count")
        if self.mode == "LINEAR":
            lay.row().prop(self, "space", expand=True)
            lay.prop(self, "distance", text="Spacing" if self.space == "SPACING" else "Length")
            lay.prop(self, "axis")
            if self.axis == "PICKED":
                lay.prop(self, "axis_dir")
            col = lay.column()
            col.prop(self, "count_b")
            sub = col.column()
            sub.active = self.count_b > 1
            sub.prop(self, "distance_b", text="Row Spacing" if self.space == "SPACING" else "Row Length")
            sub.prop(self, "axis_b")
        else:
            lay.prop(self, "angle")
            lay.prop(self, "rot_axis")
            if self.rot_axis == "PICKED":
                lay.prop(self, "axis_point")
                lay.prop(self, "axis_dir")
            else:
                lay.prop(self, "center")
                if self.center == "CURSOR":
                    lay.prop(self, "center_point", text="At")
            lay.prop(self, "rotate")

    # -- the value stage --------------------------------------------------------
    def _label(self, name):
        if name == "distance":
            return "Spacing" if self.space == "SPACING" else "Length"
        if name == "distance_b":
            return "Row Spacing" if self.space == "SPACING" else "Row Length"
        return super()._label(name)

    def drag_setup(self, context, mouse):
        srcs = self._sources(context)
        if not srcs:
            return False
        self._pv = []
        for src in srcs:
            M = world_matrix(src)
            self._pv.append((src, _preview_lines(src, M), _preview_tris(src, M), _box(src, M)))
        self._ctx = context
        self._ctyping = None
        self._picking_axis = False
        self._base_at_mouse = False
        return True

    def _anchor(self):
        """The handles' foot: the middle of all the sources' boxes (world)."""
        B = np.vstack([b for _, _, _, b in self._pv])
        return 0.5 * (B.min(axis=0) + B.max(axis=0))

    def drag_handles(self):
        p = self.params()
        src = self._pv[0][0]
        a = self._anchor()
        if self.mode == "LINEAR":
            out = []
            for field, cnt, which in (("distance", self.count, "axis"), ("distance_b", self.count_b, "axis_b")):
                if field == "distance_b" and cnt < 2:
                    continue
                d = linear_dir(p, src, which)
                k = (cnt - 1) if self.space == "SPACING" else 1
                out.append({"field": field, "kind": "arrow", "origin": a, "dir": d,
                            "param": getattr(self, field) * k, "base": a,
                            "to_value": (lambda t, k=k: t / k)})
            return out
        O, z = circ_axis(p, src)
        P = self._circle_foot(a, O, z)
        if P is None:
            return []
        C = O + z * float((P - O) @ z)
        x = _unit(P - C)
        y = np.cross(z, x)
        ang = self.angle
        m = max(2, int(abs(ang) / 3.0) + 1)
        path = np.vstack([rotate_points(P[None], O, z, ang * i / (m - 1)) for i in range(m)])
        back = ang - math.copysign(min(6.0, abs(ang) * 0.5), ang) if abs(ang) > 1e-9 else 0.0
        return [{"field": "angle", "kind": "arrow", "pos": rotate_points(P[None], O, z, ang)[0], "param": ang,
                 "origin": C, "dir": None, "path": path, "head_from": rotate_points(P[None], O, z, back)[0],
                 "pick": lambda o, r, prev: angle_pick(C, x, y, z, o, r, prev),
                 "to_value": lambda t: t, "snap": (5.0, 1.0)}]

    def _circle_foot(self, a, O, z):
        """Where the angle handle starts: the sources' middle, or (when that is
        on the axis) the box corner farthest from it."""
        B = np.vstack([b for _, _, _, b in self._pv])
        size = float(np.ptp(B, axis=0).max()) or 1.0
        rel = a - O
        if np.linalg.norm(rel - (rel @ z) * z) > 1e-3 * size:
            return a
        rel = B - O
        rad = np.linalg.norm(rel - np.outer(rel @ z, z), axis=1)
        if float(rad.max()) <= 1e-9 * size:
            return None
        return B[int(np.argmax(rad))]

    def drag_preview(self):
        p = self.params()
        lines, tris, guides = [], [], []
        n_lines = sum(len(L) for _, L, _, _ in self._pv) * max(total_count(p) - 1, 0)
        n_tris = sum(len(T) for _, _, T, _ in self._pv) * max(total_count(p) - 1, 0)
        for src, L, Tw, box in self._pv:
            for k, T in copy_transforms(p, src):
                R, t = T[:3, :3], T[:3, 3]
                if n_lines <= PREVIEW_LINES:
                    lines += [P @ R.T + t for P in L]
                else:
                    lines += _box_lines(box @ R.T + t)
                if n_tris <= PREVIEW_TRIS and len(Tw):
                    tris.append(Tw @ R.T + t)
        a = self._anchor()
        src = self._pv[0][0]
        if self.mode == "CIRCULAR":
            O, z = circ_axis(p, src)
            B = np.vstack([b for _, _, _, b in self._pv])
            h = (B - O) @ z
            size = float(np.ptp(B, axis=0).max()) or 1.0
            pad = max(0.15 * float(np.ptp(h)), 0.5 * size)
            guides.append((np.array([O + z * (h.min() - pad), O + z * (h.max() + pad)]), (0.55, 0.85, 1.0, 0.9)))
            P = self._circle_foot(a, O, z)
            if P is not None:
                ring = np.vstack([rotate_points(P[None], O, z, 360.0 * i / 96) for i in range(97)])
                guides.append((ring, (0.55, 0.85, 1.0, 0.35)))
        else:
            ends = [a]
            for _, T in copy_transforms(p, src):
                ends.append(T[:3, :3] @ a + T[:3, 3])
            E = np.array(ends)
            d = linear_dir(p, src)
            s = (E - a) @ d
            if float(np.ptp(s)) > 0:
                guides.append((np.array([a + d * s.min(), a + d * s.max()]), (0.55, 0.85, 1.0, 0.6)))
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3)), "guides": guides}

    def drag_check(self):
        if self.mode == "LINEAR":
            if abs(self.distance) <= 0.0:
                return f"{self._label('distance')} 0: drag the arrow out, or type a value"
            if self.count_b > 1 and abs(self.distance_b) <= 0.0:
                return f"{self._label('distance_b')} 0: Tab to it and drag its arrow, or type a value"
        elif abs(self.angle) < 0.05:
            return "Angle 0: drag the handle around the axis, or type an angle"
        n = (total_count(self.params()) - 1) * len(self._pv)
        if n > MAX_COPIES:
            return f"{n} copies: at most {MAX_COPIES}"
        return ""

    def drag_key(self, context, event, ch):
        if ch in self.AXES:         # X Y Z: that world axis; again: the object's local one
            ax = self.AXES[ch]
            if self.mode == "LINEAR":
                self.axis = ("LOCAL_" + ax) if self.axis == ax else ax
            else:
                self.rot_axis = ("LOCAL_" + ax) if self.rot_axis == ax else ax
        elif ch == "a":
            self._start_axis_pick(context)
        elif ch == "f":
            if self.mode == "LINEAR":
                self.distance = -self.distance
            else:
                self.angle = -self.angle
        elif ch == "m":
            self.mode = "CIRCULAR" if self.mode == "LINEAR" else "LINEAR"
            self._field = 0
            self._resolve_center(context)
        elif ch == "o" and self.mode == "CIRCULAR":
            order = ["CURSOR", "WORLD", "OBJECT"]
            self.center = order[(order.index(self.center) + 1) % 3]
            if self.rot_axis == "PICKED":
                self.rot_axis = "Z"
            self._resolve_center(context)
        elif ch == "r" and self.mode == "CIRCULAR":
            self.rotate = not self.rotate
        elif ch == "e" and self.mode == "LINEAR":
            # the copies stay where they are: the value changes with its meaning
            k = max(self.count - 1, 1)
            kb = max(self.count_b - 1, 1)
            if self.space == "SPACING":
                self.space, self.distance, self.distance_b = "EXTENT", self.distance * k, self.distance_b * kb
            else:
                self.space, self.distance, self.distance_b = "SPACING", self.distance / k, self.distance_b / kb
        else:
            return False
        return True

    def drag_handle_hints(self):
        if self.mode == "CIRCULAR":
            return [hints.key("LMB drag", "the arrow around the axis: angle")]
        return [hints.key("LMB drag", "the arrow: " + ("the last copy" if self.count_b < 2 else "a row's end"))]

    def drag_rows(self):
        count = [hints.key("Up Down", f"count {self.count}"),
                 hints.key("N", "type the count", on=True if self._ctyping is not None else None)]
        if self.mode == "LINEAR":
            count.append(hints.key("Left Right", f"rows {self.count_b}"))
            picked = self.axis == "PICKED"
            cur = None if picked else self.axis.replace("LOCAL_", "").lower()
            return [("Copies", count),
                    ("Direction", [hints.cycle("X Y Z", ["x", "y", "z"], cur),
                                   hints.key("A", "pick a line / edge", on=True if picked else None),
                                   hints.key("F", "other way")]),
                    ("Distance", [hints.cycle("E", ["spacing", "first to last"],
                                              "spacing" if self.space == "SPACING" else "first to last")]),
                    ("", [hints.key("M", "circular pattern")])]
        picked = self.rot_axis == "PICKED"
        cur = None if picked else self.rot_axis.replace("LOCAL_", "").lower()
        return [("Copies", count + [hints.toggle("R", "rotate copies", self.rotate)]),
                ("Axis", [hints.cycle("X Y Z", ["x", "y", "z"], cur),
                          hints.key("A", "pick a line / circle", on=True if picked else None),
                          hints.cycle("O", ["cursor", "world", "object"], None if picked else self.center.lower(),
                                      "center"),
                          hints.key("F", "other way"), hints.key("Ctrl", "5° steps")]),
                ("", [hints.key("M", "linear pattern")])]

    def drag_state(self):
        typed = self._ctyping
        out = [f"Count {typed}|" if typed is not None else f"Count {self.count}"]
        if self.mode == "LINEAR":
            if self.count_b > 1:
                out.append(f"Rows {self.count_b}")
            out.append("Direction " + _axis_word(self.axis))
        else:
            out.append("Axis " + _axis_word(self.rot_axis))
            if self.rot_axis != "PICKED":
                out.append("Center " + dict(CENTER_ITEMS_SHORT)[self.center])
            if not self.rotate:
                out.append("Not rotated")
        return out

    # -- the count, the axis pick ----------------------------------------------
    def drag_modal_hook(self, context, event):
        et = event.type
        if getattr(self, "_picking_axis", False):
            return self._axis_pick_modal(context, event)
        if self._ctyping is not None:
            if event.value != 'PRESS' or et in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
                return None
            ch = getattr(event, "unicode", "") or _DIGITS.get(et, "")
            if ch.isdigit():
                self._ctyping += ch
                self._set_count_typed()
            elif et == 'BACK_SPACE':
                self._ctyping = self._ctyping[:-1]
                self._set_count_typed()
            elif et == 'ESC':
                self.count = self._count_before
                self._ctyping = None
            elif et in {'RET', 'NUMPAD_ENTER', 'TAB', 'N', 'SPACE'}:
                self._ctyping = None
            elif et in {'LEFTMOUSE', 'RIGHTMOUSE'}:
                self._ctyping = None
                return None
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        if event.value != 'PRESS' or self._grab is not None:
            return None
        if et in {'UP_ARROW', 'DOWN_ARROW'}:
            self.count = max(2, self.count + (1 if et == 'UP_ARROW' else -1))
        elif et in {'RIGHT_ARROW', 'LEFT_ARROW'} and self.mode == "LINEAR":
            self.count_b = max(1, self.count_b + (1 if et == 'RIGHT_ARROW' else -1))
        elif et == 'N' and not self._typed and not event.ctrl and not event.oskey:
            self._ctyping = ""
            self._count_before = self.count
        else:
            return None
        self._drag_refresh(context)
        return {'RUNNING_MODAL'}

    def _set_count_typed(self):
        if self._ctyping:
            self.count = max(2, min(1000, int(self._ctyping)))
        else:
            self.count = self._count_before

    def _start_axis_pick(self, context):
        from .ops import _pick_candidates
        self._axis_cands = [c for c in _pick_candidates(context)
                            if float(np.ptp(np.array([tuple(q) for q in c[2]]), axis=0).max()) > 1e-9]
        self._axis_hover = None
        self._picking_axis = True
        draw.pick_overlay["hover"] = None
        what = "a line, edge or circle for the axis" if self.mode == "CIRCULAR" else \
            "a line or edge for the direction (start to end)"
        hints.show(context, self.bl_label, ["click " + what],
                   [("Pick", [hints.key("LMB", "use it"), hints.key("Esc", "back, unchanged")])],
                   [("MOUSE_LMB", "Pick"), ("EVENT_ESC", "Back")])

    def _axis_pick_modal(self, context, event):
        from .ops import PickMixin, _RegionMouse, _near_candidates
        et = event.type
        if et in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event):
            return {'PASS_THROUGH'}
        if et == 'MOUSEMOVE':
            ties = _near_candidates(context, _RegionMouse(self._mouse(context, event)), self._axis_cands)
            self._axis_hover = ties[0] if ties else None
            draw.pick_overlay["hover"] = self._axis_hover[2] if self._axis_hover else None
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.value != 'PRESS':
            return {'RUNNING_MODAL'}
        if et in {'ESC', 'RIGHTMOUSE'}:
            self._end_axis_pick()
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        if et == 'LEFTMOUSE' and self._axis_hover is not None:
            got = circle_or_line(self._axis_hover[2])
            if got is not None:
                O, d, closed = got
                self.axis_point, self.axis_dir = tuple(O), tuple(d)
                if self.mode == "LINEAR":
                    self.axis = "PICKED"
                else:
                    self.rot_axis = "PICKED"
            self._end_axis_pick()
            self._drag_refresh(context)
        return {'RUNNING_MODAL'}

    def _end_axis_pick(self):
        self._picking_axis = False
        draw.pick_overlay["hover"] = None


CENTER_ITEMS_SHORT = [("CURSOR", "3D Cursor"), ("WORLD", "World Origin"), ("OBJECT", "Object Origin")]


def _axis_word(ax):
    if ax == "PICKED":
        return "picked"
    return ax.replace("LOCAL_", "Local ") if ax.startswith("LOCAL_") else "World " + ax


# preview geometry (world) of a source
def _preview_lines(src, M):
    R, t = M[:3, :3], M[:3, 3]
    if src.type == 'CURVE':
        out = []
        for i in range(len(src.data.splines)):
            try:
                out.append(geom.sample_curve(features.input_curve(src, f"S{i}"), 64))
            except Exception:
                pass
        return out
    return [np.asarray(pl, float) @ R.T + t for pl in store.display(src).get("edges", []) if len(pl) > 1]


def _preview_tris(src, M):
    if src.type != 'MESH':
        return np.zeros((0, 3))
    me = src.data
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    if not nt or nt > PREVIEW_TRIS:
        return np.zeros((0, 3))
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    tri = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    P = co.reshape(-1, 3).astype(float)[tri]
    return P @ M[:3, :3].T + M[:3, 3]


def _box(src, M):
    P = _local_points(src)
    if P is None or not len(P):
        P = np.zeros((1, 3))
    lo, hi = P.min(axis=0), P.max(axis=0)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return C @ M[:3, :3].T + M[:3, 3]


def _box_lines(C):
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [np.array([C[a], C[b]]) for a, b in pairs]


# --------------------------------------------------------------------------
# Feature panel: a copy's pattern
# --------------------------------------------------------------------------

class NPROTO_OT_pattern_select(bpy.types.Operator):
    """Select every copy of this pattern and its originals"""
    bl_idname = "nproto.pattern_select"
    bl_label = "Select Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    group: StringProperty(options={'HIDDEN'})

    def execute(self, context):
        copies = group_copies(self.group, context.scene.objects)
        if not copies:
            self.report({'WARNING'}, "No copies of that pattern")
            return {'CANCELLED'}
        act = context.view_layer.objects.active
        for o in context.selected_objects:
            o.select_set(False)
        for o in dict.fromkeys([s for _, s, _ in copies] + [c for c, _, _ in copies]):
            try:
                o.select_set(True)
            except RuntimeError:
                pass
        if act is not None and act.select_get():
            context.view_layer.objects.active = act
        return {'FINISHED'}


def summary(p):
    """'Linear · 4 × 20 mm' style words for a pattern's params."""
    from .valuedrag import fmt_length
    ctx = bpy.context
    n = int(p.get("count", 2))
    if p.get("mode") == "CIRCULAR":
        ang = float(p.get("angle", 360.0))
        txt = f"Circular · {n} over {ang:g}° · " + _axis_word(p.get("rot_axis", "Z"))
        return txt + ("" if p.get("rotate", True) else " · not rotated")
    word = "spacing" if p.get("space", "SPACING") == "SPACING" else "length"
    txt = f"Linear · {n} · {word} {fmt_length(ctx, float(p.get('distance', 0.0)))} · " + _axis_word(p.get("axis", "X"))
    if int(p.get("count_b", 1)) > 1:
        txt += f" · {int(p['count_b'])} rows {fmt_length(ctx, float(p.get('distance_b', 0.0)))}"
    return txt


def draw_feature(box, context, obj, h):
    """The Feature panel's part for a pattern copy (feature_ui.draw_feature)."""
    p = h["params"]
    col = box.column(align=True)
    col.label(text=summary(p), icon='MOD_ARRAY')
    copies = group_copies(p.get("group", ""), context.scene.objects)
    col.label(text=f"Copy {int(p.get('index', 0))} · {len(copies)} copies in the pattern", icon='LINKED')
    row = box.row(align=True)
    op = row.operator("nproto.pattern", text="Edit Pattern", icon='MODIFIER')
    op.group = p.get("group", "")
    row.operator("nproto.pattern_select", text="Select All", icon='RESTRICT_SELECT_OFF').group = p.get("group", "")


classes = (NPROTO_OT_pattern, NPROTO_OT_pattern_select)
