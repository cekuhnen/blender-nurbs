"""Move, Rotate and Scale by points, and copies along the way (hold G, R, S).

Tap G or R and Blender's own Move or Rotate runs, as it always did. Hold the
key (longer than the preference's Hold Time, 300 ms by default) and the
tool here runs instead. Its points are picked with the sketch tools'
snapping (curve and edge ends, middles, face centres, CVs, mesh vertices and
edges; Ctrl the grid, Alt nothing); C and W put the next point on the 3D
cursor and the world origin.

Move: a base point, then the target. X / Y / Z lock the target along that
axis from the base, Shift+X / Y / Z into the plane of the other two. A typed
number is the distance along the direction the mouse shows (or the axis).

Rotate: the centre, a start point (the reference direction), an end point;
the turn is the angle from the start to the end about the axis. The axis is
the drawing plane's normal (Top view: Z), or X / Y / Z (the work plane's
when one is active), or V the view direction. Shift snaps the angle to 15°;
a typed number is the angle in degrees, from the start point (or straight
after the centre, from the plane's X).

Scale: the centre, a first reference point, a second: the first's distance
from the centre becomes the second's (factor = second / first). Uniform by
default; X / Y / Z scale along that axis only (1D), Shift+X / Y / Z in the
plane square to it (2D), M cycles uniform / 1D along the first reference /
2D in the drawing plane. Shift steps the factor by 0.1. A typed number is
the factor before the first reference and the second's length after it
(2x: a factor). Scale makes no copies.

Copies (Move, Rotate): typed with a leading * or /, as SketchUp and 3ds Max take them.
  *n   n copies, each one step further on (the original stays)
  /n   n copies spread evenly up to the target / end (the last on it)
Move's D (Copy) leaves the original: each click puts a copy on the target
and the tool goes on from the same base (Rhino's Copy) until Enter or the
right button. S (Move, Rotate) scales what lands on the target / end, a
reference pair or a typed factor after that click: Move's about the base
point carried there, Rotate's about the start point carried round or (A)
the centre of the turn (a spiral). The copies on the way blend from the
original's size to it, *n's go on changing as much each step; a scaled
copy a full turn round is kept.
A copy that would land on the original (a full turn) is left out. NURBS
objects and curves are copied as a linear or circular Pattern (pattern.py):
each copy has the source's exact B-rep, follows it through construction
history, and Edit Pattern changes the count later. Other objects are
duplicated. A * or / anywhere else in a number is arithmetic (10*2).

P: put the objects' origins (pivots) at the base point / centre first, as
Move Pivot does.

In Edit Mode the selected vertices / CVs / points move (a CV cage's surface
follows); copies are an Object Mode thing there.

Nothing changes until the last point is clicked (or Enter / Space): the tool
then works in one undo step, and the Adjust Last Operation panel changes the
offset or angle, the copies and their count afterwards.
"""

import math
import time

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty
from mathutils import Matrix, Vector

from . import draw, hints, pattern, snapping
from .drawtools import AXIS_COL, REFRESH_KEYS, SketchMixin, parse_length

EPS = 1e-12
HOLD_MS = 300                   # default hold time: a press longer than this starts the NURBS tool
MAX_PREVIEW_PTS = 20000         # Edit Mode: more moved points than this are not drawn one by one
ANGLE_SNAP = 15.0               # Shift: the turn in steps of this many degrees
COPY_ITEMS = [("NONE", "None", "Move or turn the selection itself"),
              ("ARRAY", "Copies (*)", "Copies one step apart: the first on the target, the next one step on"),
              ("DIVIDE", "Copies Between (/)", "Copies spread evenly from the start to the target, "
                                               "the last on the target")]
COL_LINE = (1.0, 1.0, 1.0, 0.85)
COL_BASE = (0.6, 0.6, 0.6, 0.9)
COL_COPY = (1.0, 0.62, 0.12, 0.9)
COL_AXIS = (1.0, 1.0, 1.0, 0.6)
EDIT_MODES = {'EDIT_MESH', 'EDIT_CURVE', 'EDIT_SURFACE', 'EDIT_LATTICE'}


# --------------------------------------------------------------------------
# Pure helpers (tested headless)
# --------------------------------------------------------------------------

def parse_copies(text):
    """'*5' -> ('ARRAY', 5), '/4' -> ('DIVIDE', 4); anything else (an unfinished
    '*', a number, 0 copies) -> None."""
    t = (text or "").strip()
    if len(t) < 2 or t[0] not in "*/":
        return None
    try:
        n = int(t[1:].strip())
    except ValueError:
        return None
    if n < 1:
        return None
    return ("ARRAY" if t[0] == "*" else "DIVIDE"), n


def parse_angle(text):
    """A typed angle in degrees (a number or arithmetic: 90/4), or None."""
    t = (text or "").strip().rstrip("°")
    if not t:
        return None
    try:
        v = float(eval(t, {"__builtins__": {}}, {}))
    except Exception:
        return None
    return v if math.isfinite(v) else None


def copy_offsets(d, mode, n):
    """The world offsets of the copies (the original's left out)."""
    d = np.asarray(d, float)
    if mode == "ARRAY":
        return [k * d for k in range(1, n + 1)]
    if mode == "DIVIDE":
        return [d * (k / n) for k in range(1, n + 1)]
    return []


def copy_scales(f, mode, n):
    """The scales of the copies (copy_offsets' order): the one on the target
    is f, the ones between blend from 1; *n's go on changing as much a step."""
    if mode == "ARRAY":
        return [1.0 + k * (f - 1.0) for k in range(1, n + 1)]
    if mode == "DIVIDE":
        return [1.0 + (f - 1.0) * k / n for k in range(1, n + 1)]
    return []


def move_mats(base, d, f=1.0, mode="NONE", n=1):
    """World 4x4s: the move itself (NONE), or each copy's -- moved, and
    scaled about the base point carried with it."""
    base = np.asarray(base, float)
    d = np.asarray(d, float)
    if mode == "NONE":
        return [translation(d) @ scale_matrix(base, f)]
    return [translation(o) @ scale_matrix(base, s) for o, s in zip(copy_offsets(d, mode, n),
                                                                   copy_scales(f, mode, n))]


def _full_turn(a):
    return abs(abs(a) - 360.0) < 1e-4


def copy_angles(deg, mode, n):
    """The turns (degrees) of the copies, the original's left out; a copy a
    full turn round (back on the original) is left out too."""
    if mode == "ARRAY":
        out = [deg * k for k in range(1, n + 1)]
        return out[:-1] if _full_turn(deg * n) else out
    if mode == "DIVIDE":
        out = [deg * k / n for k in range(1, n + 1)]
        return out[:-1] if _full_turn(deg) else out
    return []


def translation(d):
    T = np.eye(4)
    T[:3, 3] = np.asarray(d, float)
    return T


def rotation_matrix(C, axis, deg):
    """World 4x4: deg degrees about the axis through C (right-handed)."""
    R = pattern._rotation(axis, deg)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(C, float) - R @ np.asarray(C, float)
    return T


def signed_angle(a, b, n):
    """Degrees from a to b about the unit axis n (both seen in the plane square
    to n), in (-180, 180]; None when either lies along n."""
    n = np.asarray(n, float) / (np.linalg.norm(n) or 1.0)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - (a @ n) * n
    b = b - (b @ n) * n
    if np.linalg.norm(a) <= 1e-12 or np.linalg.norm(b) <= 1e-12:
        return None
    return math.degrees(math.atan2(float(n @ np.cross(a, b)), float(a @ b)))


def pattern_params(base, d, mode, n, scale=1.0):
    """The linear Pattern (pattern.py params) that makes the move's copies
    (scale: copy_scales' f, about the base point)."""
    d = np.asarray(d, float)
    L = float(np.linalg.norm(d))
    x = d / L if L > EPS else np.array([1.0, 0.0, 0.0])
    p = {"v": 2, "mode": "LINEAR", "space": "SPACING" if mode == "ARRAY" else "EXTENT", "turn": 0.0,
            "center_point": [float(c) for c in base], "frame_x": [float(c) for c in x],
            "frame_z": [float(c) for c in pattern._perp(x)],
            "count_x": int(n) + 1, "distance_x": L, "count_y": 1, "distance_y": 0.0, "count_z": 1, "distance_z": 0.0}
    if abs(float(scale) - 1.0) > 1e-12:
        p["scale"] = float(scale)
        p["scale_center"] = [float(c) for c in base]
    return p


def turn_steps(deg, f, mode, n):
    """[(degrees, scale)] of the copies: copy_angles' turns, copy_scales'
    sizes. A copy a full turn round is left out only when it is not scaled
    (then it would lie on the original)."""
    if mode == "ARRAY":
        angles = [deg * k for k in range(1, n + 1)]
    elif mode == "DIVIDE":
        angles = [deg * k / n for k in range(1, n + 1)]
    else:
        return []
    out = list(zip(angles, copy_scales(f, mode, n)))
    if out and _full_turn(out[-1][0]) and abs(out[-1][1] - 1.0) <= 1e-9:
        out.pop()
    return out


def turn_mats(C, axis, deg, f=1.0, Q=None, mode="NONE", n=1):
    """World 4x4s: the turn itself (NONE), or each copy's -- scaled about Q
    (a point carried round, or the centre), then turned about the axis
    through C."""
    C = np.asarray(C, float)
    Q = C if Q is None else np.asarray(Q, float)
    if mode == "NONE":
        return [rotation_matrix(C, axis, deg) @ scale_matrix(Q, f)]
    return [rotation_matrix(C, axis, a) @ scale_matrix(Q, s) for a, s in turn_steps(deg, f, mode, n)]


def circular_params(C, axis, deg, mode, n, scale=1.0, Q=None):
    """The circular Pattern that makes the turn's copies (turn_steps')."""
    total = deg * n if mode == "ARRAY" else deg
    count = int(n) + 1
    scaled = abs(float(scale) - 1.0) > 1e-12
    if _full_turn(total) and not scaled:
        count -= 1              # a full turn spreads `count` evenly round: none on the original
    z = np.asarray(axis, float) / (np.linalg.norm(axis) or 1.0)
    p = {"v": 2, "mode": "CIRCULAR", "space": "SPACING", "turn": 0.0,
         "center_point": [float(c) for c in C], "frame_x": [float(c) for c in pattern._perp(z)],
         "frame_z": [float(c) for c in z], "count": count, "angle": float(total), "rotate": True}
    if scaled:
        p["scale"] = copy_scales(float(scale), mode, int(n))[-1]      # the Pattern's: the last copy's
        p["scale_center"] = [float(c) for c in (C if Q is None else Q)]
        if _full_turn(total):
            p["ends"] = True    # the last copy (scaled) at the full turn
    return p


SCALE_MODES = [("UNIFORM", "Uniform", "The same in every direction"),
               ("ONE", "1D", "Along one direction only (the reference's, or X / Y / Z)"),
               ("TWO", "2D", "In a plane only (the drawing plane, or square to X / Y / Z); "
                             "unchanged along its normal")]
FACTOR_SNAP = 0.1               # Shift: the factor in steps of this


def scale_matrix(C, f, mode="UNIFORM", d=(0.0, 0.0, 1.0)):
    """World 4x4: scale by f about C -- uniform, along the unit direction d
    (ONE), or in the plane square to d (TWO)."""
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) or 1.0)
    if mode == "ONE":
        A = np.eye(3) + (f - 1.0) * np.outer(d, d)
    elif mode == "TWO":
        A = f * np.eye(3) + (1.0 - f) * np.outer(d, d)
    else:
        A = f * np.eye(3)
    C = np.asarray(C, float)
    T = np.eye(4)
    T[:3, :3] = A
    T[:3, 3] = C - A @ C
    return T


def ref_length(C, P, mode="UNIFORM", d=(0.0, 0.0, 1.0)):
    """How far P is from C as the scale measures it: the distance (UNIFORM),
    along d (ONE, signed), or in the plane square to d (TWO)."""
    v = np.asarray(P, float) - np.asarray(C, float)
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) or 1.0)
    if mode == "ONE":
        return float(v @ d)
    if mode == "TWO":
        return float(np.linalg.norm(v - (v @ d) * d))
    return float(np.linalg.norm(v))


def parse_factor(text):
    """A typed factor: '2', '1/3', 'x2', '2x' -> the number, or None."""
    t = (text or "").strip().lower().strip("x×").strip()
    return parse_angle(t)


def roots(objs):
    """The objects none of whose parents is among them (a child moves with its parent)."""
    names = {o.name for o in objs}
    out = []
    for o in objs:
        p, inside = o.parent, False
        while p is not None:
            if p.name in names:
                inside = True
                break
            p = p.parent
        if not inside:
            out.append(o)
    return out


def transform_objects(objs, T):
    """Each of objs (roots) moved by the world 4x4 T. -> the names of those
    whose matrix can't hold the result exactly (a scale slanted to a turned
    object would shear it: Blender keeps the nearest location / rotation /
    scale)."""
    T = Matrix(np.asarray(T, float).tolist())
    off = []
    for o in objs:
        want = T @ o.matrix_world
        o.matrix_world = want
        size = max(max(abs(x) for x in row) for row in want)
        got = o.matrix_basis            # (built from location / rotation / scale; matrix_world reads back
        if o.parent is not None:        #  the matrix just set until the next update)
            got = o.parent.matrix_world @ o.matrix_parent_inverse @ got
        if max(abs(got[i][j] - want[i][j]) for i in range(4) for j in range(4)) > 1e-5 * max(size, 1.0):
            off.append(o.name)
    return off


def move_objects(objs, d):
    transform_objects(objs, translation(d))


def _duplicate(context, src, T):
    obj = src.copy()
    if src.data is not None:
        obj.data = src.data.copy()
    colls = list(src.users_collection) or [context.collection or context.scene.collection]
    for c in colls:
        c.objects.link(obj)
    obj.matrix_world = Matrix(np.asarray(T, float).tolist()) @ src.matrix_world
    return obj


def make_copies(context, objs, params, mats):
    """Copies of objs (roots): NURBS objects and curves as one Pattern (params),
    other objects duplicated under each world 4x4 of mats. -> (copies, errors)."""
    made, errors = [], []
    nurbs = [o for o in objs if pattern.usable(o)]
    plain = [o for o in objs if not pattern.usable(o)]
    if nurbs and mats:
        m, e = pattern.make_pattern(context, nurbs, params)
        made += m
        errors += e
    for o in plain:
        for T in mats:
            try:
                made.append(_duplicate(context, o, T))
            except Exception as exc:
                errors.append(f"{o.name}: {exc}")
    return made, errors


def edit_points(context):
    """World positions of the selected points of the objects in Edit Mode."""
    pts = []
    for o in context.objects_in_mode:
        M = o.matrix_world
        if o.type == 'MESH':
            bm = bmesh.from_edit_mesh(o.data)
            pts += [M @ v.co for v in bm.verts if v.select]
        elif o.type in {'CURVE', 'SURFACE'}:
            for sp in o.data.splines:
                for b in sp.bezier_points:
                    if b.select_control_point:
                        pts.append(M @ b.co)
                pts += [M @ p.co.xyz for p in sp.points if p.select]
        elif o.type == 'LATTICE':
            pts += [M @ p.co_deform for p in o.data.points if p.select]
    return np.array([tuple(p) for p in pts], float).reshape(-1, 3)


def transform_edit_points(context, T):
    """The selected points of the objects in Edit Mode moved by the world 4x4
    T (a Bezier point's handles with it). -> how many moved."""
    T = Matrix(np.asarray(T, float).tolist())
    n = 0
    for o in context.objects_in_mode:
        L = o.matrix_world.inverted() @ T @ o.matrix_world       # the same move in o's local space
        if o.type == 'MESH':
            bm = bmesh.from_edit_mesh(o.data)
            vs = [v for v in bm.verts if v.select]
            for v in vs:
                v.co = L @ v.co
            if vs:
                bmesh.update_edit_mesh(o.data, loop_triangles=True, destructive=False)
            n += len(vs)
        elif o.type in {'CURVE', 'SURFACE'}:
            for sp in o.data.splines:
                for b in sp.bezier_points:
                    if b.select_control_point:
                        b.co, b.handle_left, b.handle_right = L @ b.co, L @ b.handle_left, L @ b.handle_right
                        n += 1
                    else:
                        if b.select_left_handle:
                            b.handle_left = L @ b.handle_left
                            n += 1
                        if b.select_right_handle:
                            b.handle_right = L @ b.handle_right
                            n += 1
                for p in sp.points:
                    if p.select:
                        p.co = (*(L @ p.co.xyz), p.co[3])
                        n += 1
            o.data.update_tag()
        elif o.type == 'LATTICE':
            for p in o.data.points:
                if p.select:
                    p.co_deform = L @ p.co_deform
                    n += 1
            o.data.update_tag()
    return n


def move_edit_points(context, d):
    return transform_edit_points(context, translation(d))


def hold_ms():
    p = hints._prefs()
    return int(getattr(p, "hold_ms", HOLD_MS)) if p is not None else HOLD_MS


def hold_enabled():
    p = hints._prefs()
    return bool(getattr(p, "hold_transform", True)) if p is not None else True


def _len_text(context, x):
    from .valuedrag import fmt_length
    try:
        return fmt_length(context, float(x))
    except Exception:
        return f"{float(x):.4g}"


def _deg_text(a):
    return f"{a:.4g}°"


# --------------------------------------------------------------------------
# What the tools share: the selection, its ghosts, the typed copies, apply
# --------------------------------------------------------------------------

class _PointsTool(SketchMixin):
    """Move / Rotate by points: one base, two operators."""
    bl_options = {'REGISTER', 'UNDO'}

    face_planes = False         # a free point lies on the drawing plane through the last point
    held_key = 'G'              # the key that started the tool (its release and key repeat are ignored)
    title = "Move"
    allow_copies = True         # *n / /n typed

    copies: EnumProperty(name="Copies", items=COPY_ITEMS, default="NONE")
    count: IntProperty(name="Count", default=1, min=1, soft_max=50, max=pattern.MAX_COPIES,
                       description="How many copies (the original not counted)")

    @classmethod
    def poll(cls, context):
        return context.mode in {"OBJECT"} | EDIT_MODES

    # -- doing it ---------------------------------------------------------------
    def _transform(self):
        """World 4x4 of the move / turn itself."""
        raise NotImplementedError

    def _copy_mats(self):
        raise NotImplementedError

    def _copy_params(self):
        raise NotImplementedError

    def _pivot_point(self):
        raise NotImplementedError

    def _amount_text(self, context):
        raise NotImplementedError

    def _is_identity(self):
        raise NotImplementedError

    def _copying(self):
        return self.copies != "NONE"

    def _copy_jobs(self):
        """[(Pattern params, world 4x4s)]: the copies to make, one Pattern each."""
        return [(self._copy_params(), self._copy_mats())]

    def execute(self, context):
        name = self.title
        if context.mode in EDIT_MODES:
            if self._copying():
                self.report({'WARNING'}, f"{name}: copies are made in Object Mode (Edit Mode moves the points)")
            if self._is_identity():
                return {'CANCELLED'}
            n = transform_edit_points(context, self._transform())
            if not n:
                self.report({'WARNING'}, f"{name}: nothing selected")
                return {'CANCELLED'}
            self.report({'INFO'}, f"{name}: {n} point(s), {self._amount_text(context)}")
            return {'FINISHED'}
        objs = roots(list(context.selected_objects))
        if not objs:
            self.report({'WARNING'}, f"{name}: select the objects")
            return {'CANCELLED'}
        if self._is_identity():
            return {'CANCELLED'}
        if self.set_pivot:
            from . import pivottool
            for o in objs:
                why = pivottool.refusal(o)
                if why:
                    self.report({'WARNING'}, f"{name}: pivot not moved, " + why)
                    continue
                try:
                    pivottool.move_origin(o, self._pivot_point(), context.view_layer)
                except Exception as exc:
                    self.report({'WARNING'}, f"{name}: pivot of {o.name} not moved: {exc}")
        if not self._copying():
            off = transform_objects(objs, self._transform())
            if off:
                self.report({'WARNING'}, f"{name}: {', '.join(off[:3])}{' ...' if len(off) > 3 else ''}: turned "
                                         "against the scale's direction, it can't be sheared; scaled as near as "
                                         "an object allows (apply its rotation first, or scale in Edit Mode)")
            self.report({'INFO'}, f"{name}: {len(objs)} object(s), {self._amount_text(context)}")
            return {'FINISHED'}
        jobs = [(p, m) for p, m in self._copy_jobs() if m]
        k = sum(len(m) for _, m in jobs)
        if not k:
            self.report({'WARNING'}, f"{name}: no copies (they would all lie on the original)")
            return {'CANCELLED'}
        if k * len(objs) > pattern.MAX_COPIES:
            self.report({'ERROR'}, f"{name}: {k * len(objs)} copies, at most {pattern.MAX_COPIES} at once")
            return {'CANCELLED'}
        act = context.view_layer.objects.active
        made, errors = [], []
        for params, mats in jobs:
            m, e = make_copies(context, objs, params, mats)
            made += m
            errors += e
        for e in errors:
            self.report({'ERROR'}, f"{name}: " + e)
        if not made:
            return {'CANCELLED'}
        for o in made:
            try:
                o.select_set(True)
            except RuntimeError:
                pass
        if act is not None:
            context.view_layer.objects.active = act
        self.report({'INFO'}, f"{name}: {len(made)} cop{'y' if len(made) == 1 else 'ies'}")
        return {'FINISHED'}

    # -- starting -------------------------------------------------------------
    def _gather(self, context):
        """The selection (objects, or Edit Mode's points) and its ghost
        geometry. -> False (reported) when there is nothing to work on."""
        self._edit = context.mode in EDIT_MODES
        if self._edit:
            for o in context.objects_in_mode:
                try:
                    o.update_from_editmode()
                except Exception:
                    pass
            self._pts = edit_points(context)
            if not len(self._pts):
                self.report({'WARNING'}, f"{self.title}: nothing selected")
                return False
            self._objs = []
            self._center = self._pts.mean(axis=0)
            self._pv = []
            return True
        self._objs = roots(list(context.selected_objects))
        if not self._objs:
            self.report({'WARNING'}, f"{self.title}: select the objects")
            return False
        self._pts = np.zeros((0, 3))
        self._center = np.mean([np.array(o.matrix_world.translation, float) for o in self._objs], axis=0)
        try:
            self._pv = pattern._preview_sources(self._objs)
        except Exception as exc:
            print(f"NURBS {self.title} preview:", exc)
            self._pv = []
        return True

    def _snap_targets(self, context):
        """As the sketch tools', and the CVs of the NURBS objects being moved
        snap whether or not their net is shown (a base point on a CV)."""
        moving = {o.name for o in getattr(self, "_objs", ())}
        return snapping.collect(context.visible_objects,
                                lambda o: o.name in moving or draw.cv_net_shown(context, o),
                                snapping.sketch_mesh_elements)

    # -- typing: a value, or (a leading * or /) the copies -------------------------
    def _live_copies(self):
        if self._edit or not self.allow_copies:
            return None
        if self._buf and self._buf[0] in "*/":
            return parse_copies(self._buf) or self._copy
        return self._copy

    def _typed_key(self, context, event):
        """A character typed into the buffer? (True: taken)"""
        ch = self.typed_char(event)
        if not (self.typing_ready() and ch) or event.ctrl or event.alt or event.oskey:
            return False
        if self._in_stage() and not self._buf and ch in "*/":
            return False            # the scale stage: the copies were set before it
        copies_typed = self._buf[:1] in ("*", "/")
        if (not self._buf and ch in "0123456789.-+*/") \
                or (self._buf and not copies_typed and (ch in "0123456789.-+*/'\" ()" or ch.isalpha())) \
                or (copies_typed and ch.isdigit()):
            if ch in "*/" and not self._buf and not self.allow_copies:
                return False
            if ch in "*/" and not self._buf and self._edit:
                self.report({'INFO'}, f"{self.title}: copies are made in Object Mode")
                return True
            self._buf += ch
            return True
        return False

    def _fix_buffer(self, context):
        """Tab / Enter: what is typed becomes the value or the copies."""
        if not self._buf:
            return False
        if self._buf[0] in "*/":
            c = parse_copies(self._buf)
            if c is not None and not self._edit:
                self._copy = c
        else:
            v = self._live_value(context)
            if v is not None:
                self._value = v
        self._buf = ""
        return True

    def _live_value(self, context):
        """The typed value (distance / angle), the buffer included."""
        raise NotImplementedError

    # -- S: the scale stage (Move, Rotate) -----------------------------------------
    def _in_stage(self):
        return getattr(self, "_sc", None) is not None

    def _stage_center(self):
        """The point the stage scales about (world)."""
        raise NotImplementedError

    def _parse_scale(self, context, text):
        """("F", factor) or ("L", length), as the Scale tool reads them: a
        factor before the first reference or with an x (2x), else a length."""
        t = (text or "").strip()
        if not t:
            return None
        if self._r1 is None or "x" in t.lower() or "×" in t:
            v = parse_factor(t)
            return None if v is None else ("F", v)
        try:
            return "L", parse_length(context, t)
        except Exception:
            return None

    def _stage_factor(self, context, event, M):
        """The stage's factor now (or None): typed, or the second reference's
        distance from the stage centre over the first's."""
        typed = self._live_value(context)
        if typed is not None and typed[0] == "F":
            return typed[1]
        if self._r1 is None:
            return None
        C = self._stage_center()
        L1 = float(np.linalg.norm(self._r1 - C))
        if L1 <= EPS:
            return None
        if typed is not None:
            return typed[1] / L1
        if M is None:
            return None
        f = float(np.linalg.norm(M - C)) / L1
        if event is not None and event.shift:
            f = round(f / FACTOR_SNAP) * FACTOR_SNAP
        return f

    def _stage_refs(self, context, f, P):
        """The stage's marks: its centre, the first reference and where it
        goes. -> (refs, label point)."""
        V = lambda p: Vector(tuple(map(float, p)))
        C = self._stage_center()
        refs = self._cross(context, C, COL_LINE)
        at = P if P is not None else C
        if self._r1 is not None:
            refs.append(([V(C), V(self._r1)], COL_BASE, 1.5))
            refs += self._cross(context, self._r1, COL_BASE, 0.6)
            at = C + f * (self._r1 - C)                 # where the first reference ends up
            refs.append(([V(C), V(at)], COL_LINE, 2.0))
        elif P is not None:
            refs.append(([V(C), V(P)], COL_BASE, 1.0))
        return refs, at

    def _scale_hints(self, about=None):
        out = [hints.toggle("S", "scale at the " + ("target" if self.title == "Move" else "end")
                            + ", copies blend", self._scale_on)]
        if about is not None and self._scale_on:
            out.append(about)
        if self._in_stage():
            out.append(hints.key("Shift", f"steps of {FACTOR_SNAP:g}"))
        return out

    def _stage_type_hints(self):
        return [hints.key("0-9", "factor, then a length (2x: a factor)"),
                hints.key("Tab", "fix the typed value", on=None if self._buf else False)]

    # -- display ----------------------------------------------------------------
    def _ghosts(self, mats):
        """The objects drawn under each world 4x4 of mats."""
        lines, tris = [], []
        k = len(mats)
        n_lines = sum(len(L) for _, L, _, _ in self._pv) * k
        n_tris = sum(len(T) // 3 for _, _, T, _ in self._pv) * k
        for _src, L, Tw, box in self._pv:
            for T in mats:
                R, t = T[:3, :3], T[:3, 3]
                if n_lines <= pattern.PREVIEW_LINES:
                    lines += [Q @ R.T + t for Q in L]
                else:
                    lines += pattern._box_lines(box @ R.T + t)
                if n_tris <= pattern.PREVIEW_TRIS and len(Tw):
                    tris.append(Tw @ R.T + t)
                elif not len(Tw):
                    if not L:
                        lines += pattern._box_lines(box @ R.T + t)
                else:
                    tris.append(pattern._box_tris(box @ R.T + t))
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3))}

    def _show_result(self, mats, label, at, refs):
        """Ghosts (objects) or moved points (Edit Mode) under mats, the label at `at`."""
        V = lambda p: Vector(tuple(map(float, p)))
        cvs = []
        if self._edit:
            pv = {"lines": [], "tris": None}
            if len(self._pts) <= MAX_PREVIEW_PTS:
                T = mats[-1]
                cvs = [V(q) for q in self._pts @ T[:3, :3].T + T[:3, 3]]
        else:
            pv = self._ghosts(mats)
        pv["labels"] = [(V(at), label, True)]
        draw.sketch_overlay["refs"] = refs
        draw.sketch_overlay["cvs"] = cvs
        draw.pick_overlay["preview"] = pv

    def _copies_label(self):
        cp = self._live_copies()
        if not cp:
            return ""
        return f"  ×{cp[1]}" if cp[0] == "ARRAY" else f"  /{cp[1]}"

    def _cross(self, context, P, col, scale=1.0):
        from .workplanes import marker_size
        V = lambda p: Vector(tuple(map(float, p)))
        c = 0.1 * scale * marker_size(self._win(context)[1], P)
        return [([V(P - c * e), V(P + c * e)], col, 1.5) for e in np.eye(3)]

    def _type_hints(self, ready, value_text):
        c = self._live_copies()
        if not self.allow_copies:
            return [hints.key("0-9", value_text, on=None if ready else False),
                    hints.key("Tab", "fix the typed value", on=None if self._buf else False)]
        return [hints.key("0-9", value_text, on=None if ready else False),
                hints.key("*n", "n copies", on=(True if c and c[0] == "ARRAY" else None)
                          if ready and not self._edit else False),
                hints.key("/n", "n copies between", on=(True if c and c[0] == "DIVIDE" else None)
                          if ready and not self._edit else False),
                hints.key("Tab", "fix the typed value", on=None if self._buf else False)]

    def _state_common(self, context):
        what = (f"{len(self._pts)} point(s)" if self._edit else
                self._objs[0].name + (f" +{len(self._objs) - 1}" if len(self._objs) > 1 else ""))
        out = []
        if self._buf:
            out.append(f"typed {self._buf}_")
        c = self._live_copies()
        if c:
            made = getattr(self, "_n_made", None)
            out.append(f"{c[1]} cop{'y' if c[1] == 1 else 'ies'}" + (" between" if c[0] == "DIVIDE" else "")
                       + (f" ({made} made: one would lie on the original)" if made is not None and made < c[1]
                          else ""))
        if self._pivot:
            out.append("pivot to " + ("base" if self.title == "Move" else "centre"))
        return what, out

    def _end(self, context):
        draw.pick_overlay["preview"] = None
        self.sketch_end(context)
        context.area.tag_redraw()

    def _start_common(self, context):
        self._lock = None
        self._buf = ""              # being typed: a value, or (a leading * or /) the copies
        self._value = None          # a value typed and fixed (Tab / Enter)
        self._copy = None           # (mode, n) fixed
        self._pivot = False
        self.sketch_start(context)


# --------------------------------------------------------------------------
# Move
# --------------------------------------------------------------------------

class NPROTO_PG_move_place(bpy.types.PropertyGroup):
    """One earlier placement of Move's Copy mode (D): its base, move, scale, copies."""
    base: FloatVectorProperty(name="Base Point", size=3, subtype='XYZ', unit='LENGTH')
    offset: FloatVectorProperty(name="Move", size=3, subtype='TRANSLATION', unit='LENGTH')
    scale: FloatProperty(name="Scale", default=1.0, min=0.0001, precision=4)
    copies: EnumProperty(name="Copies", items=COPY_ITEMS, default="ARRAY")
    count: IntProperty(name="Count", default=1, min=1, max=pattern.MAX_COPIES)


def _scale_refusal(f, mode, n):
    """Why copies scaled so can't be made (one would shrink to nothing), or ''."""
    ss = [f] + copy_scales(f, mode, n)
    if min(ss) > 1e-9:
        return ""
    k = next(i for i, x in enumerate(ss[1:], 1) if x <= 1e-9)
    return f"copy {k} would scale to nothing (×{f:.3g} at the target): fewer copies, or a scale nearer 1"


class NPROTO_OT_move(_PointsTool, bpy.types.Operator):
    """Move from a base point to a target, both snapped (ends, middles, CVs,
    vertices, the grid); type a distance, *n for n copies, /n for copies in
    between; D leaves the original (copies click after click), S scales what
    lands on the target. Hold G to start it"""
    bl_idname = "nproto.move"
    bl_label = "Move by Points"
    bl_options = {'REGISTER', 'UNDO'}

    held_key = 'G'
    title = "Move"

    base: FloatVectorProperty(name="Base Point", size=3, subtype='XYZ', unit='LENGTH',
                              description="Where the move starts (world)")
    offset: FloatVectorProperty(name="Move", size=3, subtype='TRANSLATION', unit='LENGTH',
                                description="From the base point to the target (world)")
    scale: FloatProperty(name="Scale at Target", default=1.0, min=0.0001, soft_max=10.0, precision=4,
                         options={'SKIP_SAVE'},
                         description="What lands on the target is scaled by this about the base point "
                                     "carried there; the copies before it blend from the original's size "
                                     "(*n's go on changing as much each step)")
    keep: BoolProperty(name="Copy", default=False, options={'SKIP_SAVE'},
                       description="Leave the original where it is: a copy goes to the target (D in the tool)")
    more: CollectionProperty(type=NPROTO_PG_move_place, options={'SKIP_SAVE'},
                             description="Copy mode: the copies placed before the last one")
    set_pivot: BoolProperty(name="Pivot to Base Point", default=False,
                            description="Put the objects' origins at the base point first (Move Pivot), "
                                        "so they end up on the target")

    def draw(self, context):
        lay = self.layout
        lay.prop(self, "offset")
        lay.prop(self, "scale")
        if context.mode == 'OBJECT':
            lay.prop(self, "keep")
            lay.prop(self, "copies")
            if self.copies != "NONE":
                lay.prop(self, "count")
            if self.keep and len(self.more):
                lay.label(text=f"{len(self.more)} more cop{'y' if len(self.more) == 1 else 'ies'} placed "
                               "before this one", icon='DUPLICATE')
            lay.prop(self, "set_pivot")

    # -- what it does -------------------------------------------------------------
    def _placements(self):
        """[(base, offset, scale, copies, count)] of every placement, the
        operator's own last; Copy mode makes a placement with no copies one copy."""
        own = (np.array(self.base, float), np.array(self.offset, float), float(self.scale),
               self.copies, int(self.count))
        if not self.keep:
            return [own]
        out = [(np.array(m.base, float), np.array(m.offset, float), float(m.scale), m.copies, int(m.count))
               for m in self.more] + [own]
        return [(b, d, f, c, n) if c != "NONE" else (b, d, f, "ARRAY", 1) for b, d, f, c, n in out]

    def _copying(self):
        return self.keep or self.copies != "NONE"

    def _copy_jobs(self):
        return [(pattern_params(b, d, c, n, f), move_mats(b, d, f, c, n)) for b, d, f, c, n in self._placements()]

    def _transform(self):
        return move_mats(self.base, self.offset, float(self.scale))[0]

    def _copy_mats(self):
        return move_mats(self.base, self.offset, float(self.scale), self.copies, int(self.count))

    def _copy_params(self):
        return pattern_params(np.array(self.base, float), np.array(self.offset, float), self.copies,
                              int(self.count), float(self.scale))

    def _pivot_point(self):
        return np.array(self.base, float)

    def _amount_text(self, context):
        f = float(self.scale)
        return "moved " + _len_text(context, np.linalg.norm(self.offset)) + \
            ("" if abs(f - 1.0) <= 1e-9 else f", scaled ×{f:.4g}")

    def _is_identity(self):
        return float(np.linalg.norm(self.offset)) <= EPS and abs(float(self.scale) - 1.0) <= 1e-9 \
            and not (self.keep and len(self.more))

    def execute(self, context):
        for _b, _d, f, c, n in self._placements():
            why = _scale_refusal(f, c, n)
            if why:
                self.report({'WARNING'}, "Move: " + why)
                return {'CANCELLED'}
        return _PointsTool.execute(self, context)

    # -- starting -------------------------------------------------------------
    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Move: run it in a 3D View")
            return {'CANCELLED'}
        if not self._gather(context):
            return {'CANCELLED'}
        self._base_pt = None        # the base point, once picked
        self._sc = None             # S: the target picked, its scale being set (the centre)
        self._r1 = None             # ... and the scale's first reference
        self._scale_on = False      # S: a scale stage after each target
        self._keep = False          # D: the original stays, copies click after click
        self._placed = []           # D: [(base, offset, scale, copies, count)] placed so far
        self._start_common(context)
        self._plane = self._view_plane(context, self._center)
        context.window_manager.modal_handler_add(self)
        self._refresh(context, event)
        return {'RUNNING_MODAL'}

    # -- typing ---------------------------------------------------------------
    def typing_ready(self):
        return getattr(self, "_base_pt", None) is not None

    def _stage_center(self):
        return self._sc

    def _live_value(self, context):
        if self._sc is not None:
            if self._buf:
                v = self._parse_scale(context, self._buf)
                return self._value if v is None else v
            return self._value
        if self._buf and self._buf[0] not in "*/":
            try:
                return parse_length(context, self._buf)
            except Exception:
                return self._value
        return self._value

    def _mode_n(self):
        """(copies, count) of the next placement: what is typed, or one copy in Copy mode."""
        c = self._live_copies()
        if c:
            return c
        return ("ARRAY", 1) if self._keep else ("NONE", 1)

    # -- picking --------------------------------------------------------------
    def _target(self, context, event):
        """(point, snap) of the point under the mouse, the typed distance applied."""
        through = self._base_pt if self._base_pt is not None else self._center
        self._plane = self._view_plane(context, through)
        P, snap = self.pick_point(context, event, prev=self._base_pt)
        P = None if P is None else np.asarray(P, float)
        L = self._live_value(context) if self._base_pt is not None else None
        if L is None:
            return P, snap
        B = self._base_pt
        lk = self._lock
        if lk in ("X", "Y", "Z"):
            e = self._axes(context)[lk]
            e = e / (np.linalg.norm(e) or 1.0)
            if P is not None and float((P - B) @ e) < 0:
                e = -e
        elif P is not None and np.linalg.norm(P - B) > EPS:
            e = (P - B) / np.linalg.norm(P - B)
        else:
            e = self._plane[2]
        return B + L * e, None

    def _scale_point(self, context, event):
        """The scale stage: (point, snap) under the mouse, about the target."""
        self._plane = self._view_plane(context, self._sc)
        P, snap = self.pick_point(context, event, prev=self._sc)
        return (None if P is None else np.asarray(P, float)), snap

    def _pick_now(self, context, event):
        return self._scale_point(context, event) if self._sc is not None else self._target(context, event)

    def _placed_mats(self):
        out = []
        for b, d, f, c, n in self._placed:
            out += move_mats(b, d, f, c, n)
        return out

    # -- display ----------------------------------------------------------------
    def _status(self, context, f=None):
        what, extra = self._state_common(context)
        sc = self._sc is not None
        if self._base_pt is None:
            step = "click the base point"
        elif sc:
            step = ("click a reference for the scale (or type a factor, Enter: unscaled)" if self._r1 is None
                    else "click the second reference (or type its length)")
        else:
            step = "click the target" + (f" (copy {len(self._placed) + 1})" if self._keep else "")
        state = [what, step]
        L = None if sc else self._live_value(context)
        if L is not None and not self._buf:
            state.append("distance " + _len_text(context, L))
        if sc and f is not None and not self._buf:
            state.append(f"×{f:.4g}")
        state += extra
        if self._placed:
            state.append(f"{len(self._placed)} placed")
        if not sc:
            state.append(self.lock_text().strip(" ·"))
        ready = self._base_pt is not None
        put = [hints.key("C", "3D cursor"), hints.key("W", "world origin")]
        if ready:
            put.append(hints.key("B", "pick the base again"))
        rows = [("Point", put)]
        if not sc:
            rows.append(("Lock", self.direction_hints()))
        rows.append(("Snap", self.snap_hints()))
        rows.append(("Type", self._type_hints(ready, "distance") if not sc else self._stage_type_hints()))
        rows.append(("Scale", self._scale_hints()))
        if not self._edit:
            rows.append(("Copy", [hints.toggle("D", "keep the original, copy click after click", self._keep)]))
            rows.append(("Pivot", [hints.toggle("P", "origins to the base point", self._pivot)]))
        rows.append(("Move", [hints.key("Backspace", "step back"), hints.key("Enter", "done")]))
        done = bool(self._keep and self._placed)
        hints.show(context, "Move", state, rows,
                   [("MOUSE_LMB", "Base / target" + (" / references" if self._scale_on else "")),
                    ("MOUSE_RMB", "Done" if done else "Cancel"), ("EVENT_ESC", "Cancel")])

    def _preview(self, context, event, hover):
        ov = draw.sketch_overlay
        P, snap = hover
        ov["snap"] = Vector(tuple(P)) if P is not None and snap is not None else None
        ov["snap_label"] = snap[3] if P is not None and snap is not None else None
        ov["cursor"] = Vector(tuple(P)) if P is not None else None
        V = lambda p: Vector(tuple(map(float, p)))
        B = self._base_pt
        refs = [] if B is None else self._cross(context, B, COL_BASE)
        mode, n = self._mode_n()
        f = None
        if self._sc is not None:                    # the scale stage: the target fixed, its scale shown
            C = self._sc
            f = self._stage_factor(context, event, P)
            fs = f if f is not None and f > 0 else 1.0
            d = C - B
            offs = copy_offsets(d, mode, n) if mode != "NONE" else [d]
            if np.linalg.norm(d) > EPS:
                refs.append(([V(B), V(B + offs[-1])], COL_LINE, 2.0))
            for off in offs[:-1]:
                refs += self._cross(context, B + off, COL_COPY, 0.6)
            more, at = self._stage_refs(context, fs, P)
            self._show_result(self._placed_mats() + move_mats(B, d, fs, mode, n), f"×{fs:.4g}"
                              + self._copies_label(), at, refs + more)
        elif B is not None and P is not None and np.linalg.norm(P - B) > EPS:
            d = P - B
            offs = copy_offsets(d, mode, n) if mode != "NONE" else [d]
            refs.append(([V(B), V(B + offs[-1])], COL_LINE, 2.0))
            for off in offs[:-1]:
                refs += self._cross(context, B + off, COL_COPY, 0.6)
            self._show_result(self._placed_mats() + move_mats(B, d, 1.0, mode, n),
                              _len_text(context, np.linalg.norm(d)) + self._copies_label(), P, refs)
        elif self._placed:
            self._show_result(self._placed_mats(), f"{len(self._placed)} placed",
                              P if P is not None else (B if B is not None else self._center), refs)
        else:
            ov["refs"], ov["cvs"] = refs, []
            draw.pick_overlay["preview"] = None
        if self._sc is None:
            self.axis_guides(context, B)
        else:
            ov["axes"] = []
        self._status(context, f)
        context.area.tag_redraw()

    def _refresh(self, context, event):
        h = self._pick_now(context, event)
        self._preview(context, event, h)
        return h

    # -- modal ------------------------------------------------------------------
    def _place(self, context, event, P):
        """A point picked (a click, C, W): the base, the target, or a scale
        reference. -> the operator's result when it is done, else None."""
        if P is None:
            return None
        P = np.asarray(P, float)
        if self._base_pt is None:
            self._base_pt = P
            if self._lock in ("X", "Y", "Z", "-X", "-Y", "-Z"):
                self._lock = None           # an axis lock held for the base: the target picks afresh
            self._refresh(context, event)
            return None
        if self._sc is None and self._scale_on:     # the target: its scale next
            if self._buf:
                self._fix_buffer(context)
            self._sc, self._r1, self._value = P, None, None
            self._refresh(context, event)
            return None
        if self._sc is not None:
            typed = self._live_value(context)
            if self._r1 is None and not (typed is not None and typed[0] == "F"):
                if np.linalg.norm(P - self._sc) <= EPS:
                    self.report({'INFO'}, "Move: the scale's reference must lie away from the target")
                    return None
                self._r1 = P
                self._refresh(context, event)
                return None
            return self._put(context, event, self._sc, self._stage_factor(context, event, P))
        return self._put(context, event, P, 1.0)

    def _put(self, context, event, P, f):
        """The target P scaled by f: done, or (Copy mode) placed and on to the next."""
        if f is None:
            return None
        if f <= 0:
            self.report({'WARNING'}, f"Move: the second reference lies past the target (×{f:.3g}); not placed")
            return None
        d = np.asarray(P, float) - self._base_pt
        mode, n = self._mode_n()
        if abs(f - 1.0) <= 1e-9:
            f = 1.0
        why = _scale_refusal(f, mode, n)
        if why:
            self.report({'WARNING'}, "Move: " + why)
            return None
        if not self._keep:
            return self._finish(context, d, f)
        if np.linalg.norm(d) <= EPS and f == 1.0:
            self.report({'INFO'}, "Move: a copy there would lie on the original")
            return None
        self._placed.append((self._base_pt.copy(), d, f, mode, n))
        self._sc, self._r1, self._buf, self._value = None, None, "", None
        self._refresh(context, event)
        return None

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        et, ev = event.type, event.value
        if et == self.held_key:         # the held key that started it (release, key repeat)
            return {'RUNNING_MODAL'}
        if self._typed_key(context, event):
            self._refresh(context, event)
            return {'RUNNING_MODAL'}
        mods = event.ctrl or event.alt or event.oskey
        if et in REFRESH_KEYS or self.snap_key(context, event):
            self._refresh(context, event)
        elif et == 'LEFTMOUSE' and ev == 'PRESS':
            P, _ = self._pick_now(context, event)
            r = self._place(context, event, P)
            if r is not None:
                return r
        elif ev == 'PRESS' and et in {'X', 'Y', 'Z'} and not event.ctrl:
            if self._sc is None:
                self.axis_key(event)
                self._refresh(context, event)
        elif ev == 'PRESS' and et in {'C', 'W'} and not mods:
            P = np.array(context.scene.cursor.location, float) if et == 'C' else np.zeros(3)
            r = self._place(context, event, P)
            if r is not None:
                return r
        elif ev == 'PRESS' and et == 'B' and self._base_pt is not None:
            self._base_pt, self._sc, self._r1 = None, None, None
            self._buf, self._value = "", None
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'S' and not mods:
            self._scale_on = not self._scale_on
            if not self._scale_on and self._sc is not None:
                self._sc, self._r1, self._buf, self._value = None, None, "", None
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'D' and not mods:
            if self._edit:
                self.report({'INFO'}, "Move: copies are made in Object Mode")
            elif self._keep and self._placed:
                self.report({'INFO'}, f"Move: {len(self._placed)} cop{'y' if len(self._placed) == 1 else 'ies'} "
                                      "placed (Backspace takes them back first)")
            else:
                self._keep = not self._keep
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'P' and not self._edit:
            self._pivot = not self._pivot
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'TAB':
            if self._fix_buffer(context):
                self._refresh(context, event)
        elif ev == 'PRESS' and et == 'BACK_SPACE':
            if self._buf:
                self._buf = self._buf[:-1]
            elif self._sc is not None:
                if self._value is not None:
                    self._value = None
                elif self._r1 is not None:
                    self._r1 = None
                else:
                    self._sc = None
            elif self._copy is not None:
                self._copy = None
            elif self._value is not None:
                self._value = None
            elif self._placed:
                self._placed.pop()
            elif self._base_pt is not None:
                self._base_pt = None
            self._refresh(context, event)
        elif ev == 'PRESS' and et in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
            r = self._enter(context, event, et)
            if r is not None:
                return r
        elif ev == 'PRESS' and et == 'RIGHTMOUSE' and self._keep and self._placed:
            return self._done(context)
        elif ev == 'PRESS' and et in {'ESC', 'RIGHTMOUSE'}:
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _enter(self, context, event, et):
        if self._buf and et != 'SPACE' and self._buf[0] in "*/":
            self._fix_buffer(context)           # the copies typed: then click (or Enter) for the target
        elif self._sc is not None:              # the scale: typed, the mouse as the second reference, or none
            P, _ = self._scale_point(context, event)
            unscaled = self._r1 is None and self._live_value(context) is None
            r = self._put(context, event, self._sc, 1.0 if unscaled else self._stage_factor(context, event, P))
            if r is not None:
                return r
        elif self._base_pt is not None:
            idle = not self._buf and self._value is None
            if self._keep and self._placed and idle:
                return self._done(context)      # Copy mode: Enter ends it
            n0 = len(self._placed)
            P, _ = self._target(context, event)
            r = self._place(context, event, P)
            if r is not None:
                return r
            if self._keep and idle and len(self._placed) > n0:
                return self._done(context)      # nothing placed before: the one under the mouse, and done
        self._refresh(context, event)
        return None

    def _set_props(self, placements):
        *rest, (b, d, f, c, n) = placements
        self.more.clear()
        for rb, rd, rf, rc, rn in rest:
            it = self.more.add()
            it.base, it.offset, it.scale, it.copies, it.count = (tuple(map(float, rb)), tuple(map(float, rd)),
                                                                 float(rf), rc, int(rn))
        self.base = tuple(map(float, b))
        self.offset = tuple(map(float, d))
        self.scale = float(f)
        self.copies, self.count = c, int(n)
        self.set_pivot = bool(self._pivot)

    def _done(self, context):
        """Copy mode: the copies placed are made."""
        self._end(context)
        if not self._placed:
            return {'CANCELLED'}
        self._set_props(self._placed)
        self.keep = True
        return self.execute(context)

    def _finish(self, context, d, f=1.0):
        c = self._live_copies()
        self._end(context)
        if np.linalg.norm(d) <= EPS and abs(f - 1.0) <= 1e-9:
            return {'CANCELLED'}
        self._set_props([(self._base_pt, d, f) + ((c[0], c[1]) if c else ("NONE", 1))])
        self.keep = False
        return self.execute(context)


# --------------------------------------------------------------------------
# Rotate
# --------------------------------------------------------------------------

AXIS_MODES = {"PLANE": "drawing plane", "X": "X", "Y": "Y", "Z": "Z", "VIEW": "view"}


class NPROTO_OT_rotate(_PointsTool, bpy.types.Operator):
    """Turn about a centre from a start point to an end point, all snapped
    (ends, middles, CVs, vertices, the grid); X / Y / Z / V pick the axis,
    type an angle, *n for n copies, /n for copies in between; S scales what
    lands on the end (copies blend), about the point carried round or the
    centre (A). Hold R to start it"""
    bl_idname = "nproto.rotate"
    bl_label = "Rotate by Points"
    bl_options = {'REGISTER', 'UNDO'}

    held_key = 'R'
    title = "Rotate"

    center: FloatVectorProperty(name="Centre", size=3, subtype='XYZ', unit='LENGTH',
                                description="The point the turn is about (world)")
    axis: FloatVectorProperty(name="Axis", size=3, subtype='DIRECTION', default=(0.0, 0.0, 1.0),
                              description="The axis through the centre (world)")
    angle: FloatProperty(name="Angle", subtype='ANGLE', default=0.0,
                         description="The turn about the axis (right-handed)")
    scale: FloatProperty(name="Scale at End", default=1.0, min=0.0001, soft_max=10.0, precision=4,
                         options={'SKIP_SAVE'},
                         description="What lands on the end is scaled by this; the copies before it blend from "
                                     "the original's size (*n's go on changing as much each step)")
    scale_about: EnumProperty(name="Scale About", default="POINT", options={'SKIP_SAVE'}, items=[
        ("POINT", "Carried Point", "Each copy is scaled about the start point turned with it: the copies stay "
                                   "on the circle"),
        ("CENTER", "Centre", "Each copy is scaled about the centre of the turn: the copies come in or go out "
                             "as they shrink or grow (a spiral)")])
    scale_point: FloatVectorProperty(name="Carried Point", size=3, subtype='XYZ', unit='LENGTH',
                                     options={'SKIP_SAVE'},
                                     description="The point turned with the objects that they are scaled "
                                                 "about (the start point; world)")
    set_pivot: BoolProperty(name="Pivot to Centre", default=False,
                            description="Put the objects' origins at the centre first (Move Pivot)")

    def draw(self, context):
        lay = self.layout
        lay.prop(self, "angle")
        lay.prop(self, "center")
        lay.prop(self, "axis")
        lay.prop(self, "scale")
        if abs(self.scale - 1.0) > 1e-9:
            lay.prop(self, "scale_about")
        if context.mode == 'OBJECT':
            lay.prop(self, "copies")
            if self.copies != "NONE":
                lay.prop(self, "count")
            lay.prop(self, "set_pivot")

    # -- what it does -------------------------------------------------------------
    def _deg(self):
        return round(math.degrees(self.angle), 4)     # (the property is float32 radians: 90° reads 90.0000025°)

    def _axis(self):
        a = np.array(self.axis, float)
        return a / (np.linalg.norm(a) or 1.0)

    def _q(self):
        """The point the copies are scaled about (before the turn)."""
        return np.array(self.center if self.scale_about == "CENTER" else self.scale_point, float)

    def _transform(self):
        return turn_mats(self.center, self._axis(), self._deg(), float(self.scale), self._q())[0]

    def _copy_mats(self):
        return turn_mats(self.center, self._axis(), self._deg(), float(self.scale), self._q(), self.copies,
                         int(self.count))

    def _copy_params(self):
        return circular_params(np.array(self.center, float), self._axis(), self._deg(), self.copies, int(self.count),
                               float(self.scale), self._q())

    def _pivot_point(self):
        return np.array(self.center, float)

    def _amount_text(self, context):
        f = float(self.scale)
        return "turned " + _deg_text(self._deg()) + ("" if abs(f - 1.0) <= 1e-9 else f", scaled ×{f:.4g}")

    def _is_identity(self):
        return (abs(self._deg()) <= 1e-9 and abs(float(self.scale) - 1.0) <= 1e-9) \
            or float(np.linalg.norm(self.axis)) <= EPS

    def execute(self, context):
        why = _scale_refusal(float(self.scale), self.copies, int(self.count))
        if why:
            self.report({'WARNING'}, "Rotate: " + why)
            return {'CANCELLED'}
        return _PointsTool.execute(self, context)

    # -- starting -------------------------------------------------------------
    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Rotate: run it in a 3D View")
            return {'CANCELLED'}
        if not self._gather(context):
            return {'CANCELLED'}
        self._c = None              # the centre, once picked
        self._s = None              # the start point (the reference direction)
        self._axis_mode = "PLANE"
        self._frame = None          # (n, u, v) fixed at the centre for PLANE / VIEW
        self._sc = None             # S: the turn fixed (degrees), its scale being set
        self._r1 = None             # ... and the scale's first reference
        self._scale_on = False      # S: a scale stage after the end
        self._about = "POINT"       # A: scale about the carried start point / the centre
        self._start_common(context)
        self._plane = self._view_plane(context, self._center)
        context.window_manager.modal_handler_add(self)
        self._refresh(context, event)
        return {'RUNNING_MODAL'}

    # -- the axis -------------------------------------------------------------
    def _view_frame(self, context):
        rv3d = self._win(context)[1]
        q = rv3d.view_rotation
        n = np.array(q @ Vector((0.0, 0.0, 1.0)), float)      # towards the viewer: counter-clockwise is +
        u = np.array(q @ Vector((1.0, 0.0, 0.0)), float)
        return n, u, np.cross(n, u)

    def _rot_frame(self, context):
        """(n, u, v): the axis and two directions square to it."""
        m = self._axis_mode
        if m in ("X", "Y", "Z"):
            ax = self._axes(context)
            k = "XYZ".index(m)
            n = ax[m]
            u, v = ax["XYZ"[(k + 1) % 3]], ax["XYZ"[(k + 2) % 3]]
            return n, u, v
        if self._frame is not None:
            return self._frame
        if m == "VIEW":
            return self._view_frame(context)
        _O, n, u, v = self._view_plane(context, self._c if self._c is not None else self._center)
        return np.asarray(n, float), np.asarray(u, float), np.asarray(v, float)

    def _set_axis(self, context, m):
        self._axis_mode = "PLANE" if self._axis_mode == m else m
        self._frame = None
        if self._c is not None and self._axis_mode in ("PLANE", "VIEW"):
            self._frame = self._rot_frame(context)

    # -- typing ---------------------------------------------------------------
    def typing_ready(self):
        return getattr(self, "_c", None) is not None

    def _live_value(self, context):
        if self._buf and self._buf[0] not in "*/":
            v = self._parse_scale(context, self._buf) if self._sc is not None else parse_angle(self._buf)
            return self._value if v is None else v
        return self._value

    # -- S: the scale ------------------------------------------------------------
    def _carried(self):
        """The point turned with the objects (the start; else the selection's middle)."""
        return self._s if self._s is not None else self._center

    def _q_now(self):
        return self._c if self._about == "CENTER" else self._carried()

    def _stage_center(self):
        """Where the end copy is scaled about: the carried point turned, or the centre."""
        if self._about == "CENTER":
            return self._c
        n = self._frame_n
        return (rotation_matrix(self._c, n, self._sc) @ np.append(self._carried(), 1.0))[:3]

    def _enter_stage(self, context, a):
        c = self._live_copies()
        if c:
            self._copy = c
        n, _u, _v = self._rot_frame(context)
        self._frame_n = np.asarray(n, float)
        self._sc, self._r1, self._buf, self._value = a, None, "", None

    # -- picking --------------------------------------------------------------
    def _hover_point(self, context, event):
        """(point, snap) under the mouse; after the centre it lies on the plane of the turn."""
        if self._sc is not None:
            n, u, v = self._rot_frame(context)
            C = self._stage_center()
            self._plane = (C, n, u, v)
            P, snap = self.pick_point(context, event, prev=C)
        elif self._c is None:
            self._plane = self._view_plane(context, self._center)
            P, snap = self.pick_point(context, event)
        else:
            n, u, v = self._rot_frame(context)
            self._plane = (self._c, n, u, v)
            P, snap = self.pick_point(context, event, prev=self._c)
        return (None if P is None else np.asarray(P, float)), snap

    def _turn(self, context, event, P):
        """The angle (degrees) the tool would turn by now, or None."""
        if self._sc is not None:
            return self._sc
        typed = self._live_value(context)
        if typed is not None:
            return typed
        if self._c is None or self._s is None or P is None:
            return None
        n, _u, _v = self._rot_frame(context)
        a = signed_angle(self._s - self._c, P - self._c, n)
        if a is not None and event is not None and event.shift:
            a = round(a / ANGLE_SNAP) * ANGLE_SNAP
        return a

    def _ref_dir(self, context):
        """Unit reference direction in the plane of the turn (towards the start)."""
        n, u, _v = self._rot_frame(context)
        if self._s is not None:
            d = self._s - self._c
            d = d - (d @ n) * n
            if np.linalg.norm(d) > EPS:
                return d / np.linalg.norm(d), float(np.linalg.norm(d))
        return u / (np.linalg.norm(u) or 1.0), None

    # -- display ----------------------------------------------------------------
    def _status(self, context, a=None):
        what, extra = self._state_common(context)
        sc = self._sc is not None
        if sc:
            step = ("click a reference for the scale (or type a factor, Enter: unscaled)" if self._r1 is None
                    else "click the second reference (or type its length)")
        else:
            step = ("click the centre" if self._c is None else
                    "click the start (or type an angle)" if self._s is None else "click the end")
        state = [what, step]
        if a is not None and (not self._buf or sc):
            state.append(_deg_text(a))
        f = getattr(self, "_f_now", None)
        if sc and f is not None and not self._buf:
            state.append(f"×{f:.4g}")
        state += extra
        state.append("axis " + AXIS_MODES[self._axis_mode])
        ready = self._c is not None
        put = [hints.key("C", "3D cursor"), hints.key("W", "world origin")]
        if ready:
            put.append(hints.key("B", "pick the centre again"))
        m = self._axis_mode
        rows = [("Point", put),
                ("Axis", [hints.key("X Y Z", "axis", on=True if m in ("X", "Y", "Z") else None),
                          hints.key("V", "view", on=True if m == "VIEW" else None),
                          hints.key("Shift", f"{ANGLE_SNAP:g}°")]),
                ("Snap", self.snap_hints()),
                ("Type", self._stage_type_hints() if sc else self._type_hints(ready, "angle")),
                ("Scale", self._scale_hints(hints.cycle("A", ["carried point", "centre"],
                                                         "centre" if self._about == "CENTER" else "carried point",
                                                         "scale about")))]
        if not self._edit:
            rows.append(("Pivot", [hints.toggle("P", "origins to the centre", self._pivot)]))
        rows.append(("Rotate", [hints.key("Backspace", "step back"), hints.key("Enter", "done")]))
        hints.show(context, "Rotate", state, rows,
                   [("MOUSE_LMB", "Centre / start / end" + (" / references" if self._scale_on else "")),
                    ("MOUSE_RMB", "Cancel"), ("EVENT_ESC", "Cancel")])

    def _preview(self, context, event, hover):
        ov = draw.sketch_overlay
        P, snap = hover
        ov["snap"] = Vector(tuple(P)) if P is not None and snap is not None else None
        ov["snap_label"] = snap[3] if P is not None and snap is not None else None
        ov["cursor"] = Vector(tuple(P)) if P is not None else None
        ov["axes"] = []
        V = lambda p: Vector(tuple(map(float, p)))
        C = self._c
        a = self._turn(context, event, P)
        self._n_made = None
        self._f_now = None
        if C is None:
            ov["refs"], ov["cvs"] = [], []
            draw.pick_overlay["preview"] = None
            self._status(context)
            context.area.tag_redraw()
            return
        from .workplanes import marker_size
        L = marker_size(self._win(context)[1], C)
        n, _u, _v = self._rot_frame(context)
        col = AXIS_COL.get(self._axis_mode, COL_AXIS) if self._axis_mode in ("X", "Y", "Z") else COL_AXIS
        refs = self._cross(context, C, COL_BASE)
        refs.append(([V(C - 0.6 * L * n), V(C + 0.6 * L * n)], col, 2.0))
        e, r = self._ref_dir(context)
        if r is None:
            r = L if P is None else max(float(np.linalg.norm((P - C) - ((P - C) @ n) * n)), 1e-6 * L) or L
        if self._s is not None:
            refs.append(([V(C), V(self._s)], COL_BASE, 1.5))
        elif P is not None and a is None:
            refs.append(([V(C), V(P)], COL_BASE, 1.0))         # where the start would set the reference
        if a is None or abs(a) <= 1e-9:
            ov["refs"], ov["cvs"] = refs, []
            draw.pick_overlay["preview"] = None
            self._status(context, a)
            context.area.tag_redraw()
            return
        R = pattern._rotation(n, a)
        E = C + r * (R @ e)
        refs.append(([V(C), V(E)], COL_LINE, 2.0))
        arc = [V(C + r * (pattern._rotation(n, a * k / 48) @ e)) for k in range(49)]
        refs.append((arc, COL_LINE, 1.5))
        cp = self._live_copies()
        f = 1.0
        if self._sc is not None:
            self._f_now = self._stage_factor(context, event, P)
            f = self._f_now if self._f_now is not None and self._f_now > 0 else 1.0
        steps = turn_steps(a, f, *cp) if cp else [(a, f)]
        angles = [b for b, _s in steps]
        self._n_made = len(angles) if cp else None
        if cp and angles:
            span = max(angles, key=abs)
            if abs(span) > abs(a) + 1e-9:          # *n: the copies go on round past the end
                refs.append(([V(C + r * (pattern._rotation(n, a + (span - a) * k / 48) @ e)) for k in range(49)],
                             COL_COPY, 1.0))
        for b in angles if cp else []:
            if abs(b - a) <= 1e-9:
                continue                            # (the end line marks that one)
            q = C + r * (pattern._rotation(n, b) @ e)
            refs.append(([V(C + 0.85 * (q - C)), V(C + 1.15 * (q - C))], COL_COPY, 2.0))
        Q = self._q_now()
        mats = [rotation_matrix(C, n, b) @ scale_matrix(Q, sb) for b, sb in steps] or [np.eye(4)]
        label, at = _deg_text(a) + self._copies_label(), E
        if self._sc is not None:
            more, at = self._stage_refs(context, f, P)
            refs += more
            label = f"×{f:.4g}  " + label
        self._show_result(mats, label, at, refs)
        self._status(context, a)
        context.area.tag_redraw()

    def _refresh(self, context, event):
        h = self._hover_point(context, event)
        self._preview(context, event, h)
        return h

    # -- modal ------------------------------------------------------------------
    def _place(self, context, event, P):
        """A point picked (a click, C, W): the centre, the start, or the end (done)."""
        if P is None:
            return None
        P = np.asarray(P, float)
        if self._c is None:
            self._c = P
            self._frame = None
            if self._axis_mode in ("PLANE", "VIEW"):
                self._frame = self._rot_frame(context)      # fixed now: orbiting does not turn the axis
            self._refresh(context, event)
            return None
        if self._s is None and self._live_value(context) is None:
            n, _u, _v = self._rot_frame(context)
            d = P - self._c
            if np.linalg.norm(d - (d @ n) * n) <= EPS:
                self.report({'INFO'}, "Rotate: the start must lie off the axis")
                return None
            self._s = P
            self._refresh(context, event)
            return None
        if self._sc is not None:                    # the scale's references
            typed = self._live_value(context)
            if self._r1 is None and not (typed is not None and typed[0] == "F"):
                if np.linalg.norm(P - self._stage_center()) <= EPS:
                    self.report({'INFO'}, "Rotate: the scale's reference must lie away from its centre")
                    return None
                self._r1 = P
                self._refresh(context, event)
                return None
            return self._finish(context, event, P, self._stage_factor(context, event, P))
        return self._end_point(context, event, P)

    def _end_point(self, context, event, P):
        """The end picked (or the typed angle): the scale next (S), or done."""
        a = self._turn(context, event, P)
        if a is None:
            return None
        if self._scale_on:
            self._enter_stage(context, a)
            self._refresh(context, event)
            return None
        return self._finish(context, event, P)

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        et, ev = event.type, event.value
        if et == self.held_key:
            return {'RUNNING_MODAL'}
        if self._typed_key(context, event):
            self._refresh(context, event)
            return {'RUNNING_MODAL'}
        if et in REFRESH_KEYS or self.snap_key(context, event):
            self._refresh(context, event)
        elif et == 'LEFTMOUSE' and ev == 'PRESS':
            P, _ = self._hover_point(context, event)
            r = self._place(context, event, P)
            if r is not None:
                return r
        elif ev == 'PRESS' and et in {'X', 'Y', 'Z', 'V'} and not (event.ctrl or event.alt or event.oskey):
            if self._sc is None:
                self._set_axis(context, "VIEW" if et == 'V' else et)
                self._refresh(context, event)
        elif ev == 'PRESS' and et == 'S' and not (event.ctrl or event.alt or event.oskey):
            self._scale_on = not self._scale_on
            if not self._scale_on and self._sc is not None:
                self._sc, self._r1, self._buf, self._value = None, None, "", None
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'A' and not (event.ctrl or event.alt or event.oskey):
            self._about = "POINT" if self._about == "CENTER" else "CENTER"
            if self._sc is not None:
                self._r1 = None         # the reference was measured from the other centre
            self._refresh(context, event)
        elif ev == 'PRESS' and et in {'C', 'W'} and not (event.ctrl or event.alt or event.oskey):
            P = np.array(context.scene.cursor.location, float) if et == 'C' else np.zeros(3)
            r = self._place(context, event, P)
            if r is not None:
                return r
        elif ev == 'PRESS' and et == 'B' and self._c is not None:
            self._c, self._s, self._frame, self._sc, self._r1 = None, None, None, None, None
            self._buf, self._value = "", None
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'P' and not self._edit:
            self._pivot = not self._pivot
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'TAB':
            if self._fix_buffer(context):
                self._refresh(context, event)
        elif ev == 'PRESS' and et == 'BACK_SPACE':
            if self._buf:
                self._buf = self._buf[:-1]
            elif self._sc is not None:
                if self._value is not None:
                    self._value = None
                elif self._r1 is not None:
                    self._r1 = None
                else:
                    self._sc = None
            elif self._copy is not None:
                self._copy = None
            elif self._value is not None:
                self._value = None
            elif self._s is not None:
                self._s = None
            elif self._c is not None:
                self._c, self._frame = None, None
            self._refresh(context, event)
        elif ev == 'PRESS' and et in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
            if self._buf and et != 'SPACE' and self._buf[0] in "*/":
                self._fix_buffer(context)
                self._refresh(context, event)
            elif self._sc is not None:          # the scale: typed, the mouse as the second reference, or none
                P, _ = self._hover_point(context, event)
                unscaled = self._r1 is None and self._live_value(context) is None
                r = self._finish(context, event, P, 1.0 if unscaled else self._stage_factor(context, event, P))
                if r is not None:
                    return r
            elif self._c is not None:
                P, _ = self._hover_point(context, event)
                r = self._end_point(context, event, P)
                if r is not None:
                    return r
        elif ev == 'PRESS' and et in {'ESC', 'RIGHTMOUSE'}:
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context, event, P, f=1.0):
        if f is None:
            return None
        if f <= 0:
            self.report({'WARNING'}, f"Rotate: the second reference lies past the centre (×{f:.3g}); not scaled")
            return None
        a = self._turn(context, event, P)
        c = self._live_copies()
        why = _scale_refusal(f, *(c or ("NONE", 1)))
        if why:
            self.report({'WARNING'}, "Rotate: " + why)
            return None
        n = self._frame_n if self._sc is not None else np.asarray(self._rot_frame(context)[0], float)
        C, Q = self._c, self._carried()
        self._end(context)
        if a is None or (abs(a) <= 1e-9 and abs(f - 1.0) <= 1e-9):
            return {'CANCELLED'}
        self.center = tuple(map(float, C))
        self.axis = tuple(map(float, n))
        self.angle = math.radians(a)
        self.scale = 1.0 if abs(f - 1.0) <= 1e-9 else float(f)
        self.scale_about = self._about
        self.scale_point = tuple(map(float, Q))
        self.copies, self.count = (c[0], c[1]) if c else ("NONE", 1)
        self.set_pivot = bool(self._pivot)
        return self.execute(context)


# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------

MODE_TEXT = {"UNIFORM": "uniform", "ONE": "1D", "TWO": "2D"}
MODE_NEXT = {"UNIFORM": "ONE", "ONE": "TWO", "TWO": "UNIFORM"}


class NPROTO_OT_scale(_PointsTool, bpy.types.Operator):
    """Scale about a centre by reference: the first reference point's
    distance becomes the second's (or a typed length / factor); uniform, 1D
    or 2D (X / Y / Z, Shift+X / Y / Z). Hold S to start it"""
    bl_idname = "nproto.scale"
    bl_label = "Scale by Points"
    bl_options = {'REGISTER', 'UNDO'}

    held_key = 'S'
    title = "Scale"
    allow_copies = False

    center: FloatVectorProperty(name="Centre", size=3, subtype='XYZ', unit='LENGTH',
                                description="The point that stays (world)")
    factor: FloatProperty(name="Factor", default=1.0, soft_min=0.01, soft_max=100.0, precision=4,
                          description="Lengths become this many times as long")
    mode: EnumProperty(name="Scale", items=SCALE_MODES, default="UNIFORM")
    direction: FloatVectorProperty(name="Direction", size=3, subtype='DIRECTION', default=(0.0, 0.0, 1.0),
                                   description="1D: the direction scaled; 2D: the normal of the plane scaled")
    set_pivot: BoolProperty(name="Pivot to Centre", default=False,
                            description="Put the objects' origins at the centre first (Move Pivot)")

    def draw(self, context):
        lay = self.layout
        lay.prop(self, "factor")
        lay.prop(self, "mode")
        lay.prop(self, "center")
        if self.mode != "UNIFORM":
            lay.prop(self, "direction")
        if context.mode == 'OBJECT':
            lay.prop(self, "set_pivot")

    # -- what it does -------------------------------------------------------------
    def _transform(self):
        return scale_matrix(np.array(self.center, float), float(self.factor), self.mode,
                            np.array(self.direction, float))

    def _copy_mats(self):
        return []

    def _copy_params(self):
        return {}

    def _pivot_point(self):
        return np.array(self.center, float)

    def _amount_text(self, context):
        return f"scaled ×{self.factor:.4g}" + ("" if self.mode == "UNIFORM" else f" ({MODE_TEXT[self.mode]})")

    def _is_identity(self):
        return abs(float(self.factor) - 1.0) <= 1e-9

    def execute(self, context):
        if float(self.factor) <= 1e-9:
            self.report({'WARNING'}, "Scale: the factor must be above 0")
            return {'CANCELLED'}
        self.copies = "NONE"
        return _PointsTool.execute(self, context)

    # -- starting -------------------------------------------------------------
    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Scale: run it in a 3D View")
            return {'CANCELLED'}
        if not self._gather(context):
            return {'CANCELLED'}
        self._c = None              # the centre
        self._r1 = None             # the first reference point
        self._smode = "UNIFORM"
        self._axis = None           # 'X' / 'Y' / 'Z' chosen with the key, else None
        self._normal = None         # the drawing plane's normal, fixed at the centre (2D)
        self._start_common(context)
        self._plane = self._view_plane(context, self._center)
        context.window_manager.modal_handler_add(self)
        self._refresh(context, event)
        return {'RUNNING_MODAL'}

    # -- the direction ------------------------------------------------------------
    def _dir(self, context, P=None):
        """1D: the direction scaled; 2D: the plane's normal; None (1D) until a reference shows it."""
        if self._axis is not None:
            a = self._axes(context)[self._axis]
            return a / (np.linalg.norm(a) or 1.0)
        if self._smode == "TWO":
            if self._normal is not None:
                return self._normal
            n = self._view_plane(context, self._c if self._c is not None else self._center)[1]
            return np.asarray(n, float)
        if self._smode == "ONE":
            Q = self._r1 if self._r1 is not None else P
            if Q is not None and self._c is not None and np.linalg.norm(Q - self._c) > EPS:
                return (Q - self._c) / np.linalg.norm(Q - self._c)
            return None
        return np.array([0.0, 0.0, 1.0])

    def _axis_key(self, context, event):
        """X / Y / Z: 1D along it; Shift: 2D square to it; again: uniform."""
        mode = "TWO" if event.shift else "ONE"
        if self._axis == event.type and self._smode == mode:
            self._axis, self._smode = None, "UNIFORM"
        else:
            self._axis, self._smode = event.type, mode

    # -- typing ---------------------------------------------------------------
    def typing_ready(self):
        return getattr(self, "_c", None) is not None

    def _parse(self, context, text):
        """("F", factor) or ("L", length): a factor before the first reference
        or with an x (2x); after it a length."""
        t = (text or "").strip()
        if not t:
            return None
        if self._r1 is None or "x" in t.lower() or "×" in t:
            v = parse_factor(t)
            return None if v is None else ("F", v)
        try:
            return "L", parse_length(context, t)
        except Exception:
            return None

    def _live_value(self, context):
        if self._buf:
            v = self._parse(context, self._buf)
            return self._value if v is None else v
        return self._value

    # -- picking --------------------------------------------------------------
    def _hover_point(self, context, event):
        through = self._c if self._c is not None else self._center
        self._plane = self._view_plane(context, through)
        P, snap = self.pick_point(context, event, prev=self._c)
        return (None if P is None else np.asarray(P, float)), snap

    def _factor(self, context, event, P):
        """The factor the tool would scale by now (or None), and the second
        reference's length."""
        typed = self._live_value(context)
        d = self._dir(context, P)
        if typed is not None and typed[0] == "F":
            return typed[1], None
        if self._c is None or self._r1 is None or (d is None and self._smode == "ONE"):
            return None, None
        L1 = ref_length(self._c, self._r1, self._smode, d if d is not None else (0, 0, 1))
        if abs(L1) <= EPS:
            return None, None
        if typed is not None:
            return typed[1] / L1, typed[1]
        if P is None:
            return None, None
        L2 = ref_length(self._c, P, self._smode, d if d is not None else (0, 0, 1))
        f = L2 / L1
        if event is not None and event.shift and not self._axis_shift_event(event):
            f = round(f / FACTOR_SNAP) * FACTOR_SNAP
        return f, L2

    @staticmethod
    def _axis_shift_event(event):
        return event.type in {'X', 'Y', 'Z'}

    # -- display ----------------------------------------------------------------
    def _status(self, context, f=None):
        what, extra = self._state_common(context)
        step = ("click the centre" if self._c is None else
                "click the first reference (or type a factor)" if self._r1 is None else
                "click the second reference (or type its length)")
        state = [what, step]
        if f is not None and not self._buf:
            state.append(f"×{f:.4g}")
        state += extra
        state.append(MODE_TEXT[self._smode] + (f" {self._axis}" if self._axis else ""))
        ready = self._c is not None
        put = [hints.key("C", "3D cursor"), hints.key("W", "world origin")]
        if ready:
            put.append(hints.key("B", "pick the centre again"))
        ax = self._axis
        rows = [("Point", put),
                ("Scale", [hints.cycle("M", ["uniform", "1D", "2D"], MODE_TEXT[self._smode]),
                           hints.key("X Y Z", "1D along", on=True if ax and self._smode == "ONE" else None),
                           hints.key("Shift+X/Y/Z", "2D square to", on=True if ax and self._smode == "TWO" else None),
                           hints.key("Shift", f"steps of {FACTOR_SNAP:g}")]),
                ("Snap", self.snap_hints()),
                ("Type", self._type_hints(ready, "factor, then a length (2x: a factor)"))]
        if not self._edit:
            rows.append(("Pivot", [hints.toggle("P", "origins to the centre", self._pivot)]))
        rows.append(("Scale", [hints.key("Backspace", "step back"), hints.key("Enter", "done")]))
        hints.show(context, "Scale", state, rows,
                   [("MOUSE_LMB", "Centre / reference / reference"), ("MOUSE_RMB", "Cancel"),
                    ("EVENT_ESC", "Cancel")])

    def _preview(self, context, event, hover):
        ov = draw.sketch_overlay
        P, snap = hover
        ov["snap"] = Vector(tuple(P)) if P is not None and snap is not None else None
        ov["snap_label"] = snap[3] if P is not None and snap is not None else None
        ov["cursor"] = Vector(tuple(P)) if P is not None else None
        ov["axes"] = []
        V = lambda p: Vector(tuple(map(float, p)))
        C = self._c
        if C is None:
            ov["refs"], ov["cvs"] = [], []
            draw.pick_overlay["preview"] = None
            self._status(context)
            context.area.tag_redraw()
            return
        f, _L2 = self._factor(context, event, P)
        from .workplanes import marker_size
        L = marker_size(self._win(context)[1], C)
        refs = self._cross(context, C, COL_BASE)
        d = self._dir(context, P)
        col = AXIS_COL.get(self._axis, COL_AXIS) if self._axis else COL_AXIS
        if d is not None and self._smode == "ONE":
            refs.append(([V(C - 3 * L * d), V(C + 3 * L * d)], col, 1.5))     # the direction scaled
        elif d is not None and self._smode == "TWO":
            refs.append(([V(C - 0.6 * L * d), V(C + 0.6 * L * d)], col, 2.0))  # the normal that stays
        if self._r1 is not None:
            refs.append(([V(C), V(self._r1)], COL_BASE, 1.5))
            refs += self._cross(context, self._r1, COL_BASE, 0.6)
        if f is None or abs(f - 1.0) <= 1e-12 or f <= 0:
            if self._r1 is None and P is not None:
                refs.append(([V(C), V(P)], COL_BASE, 1.0))
            ov["refs"], ov["cvs"] = refs, []
            draw.pick_overlay["preview"] = None
            self._status(context, None if f is None or f <= 0 else f)
            context.area.tag_redraw()
            return
        T = scale_matrix(C, f, self._smode, d if d is not None else (0, 0, 1))
        if self._r1 is not None:
            E = (T @ np.append(self._r1, 1.0))[:3]          # where the first reference ends up
            refs.append(([V(C), V(E)], COL_LINE, 2.0))
            at = E
        else:
            at = P if P is not None else C
        self._show_result([T], f"×{f:.4g}", at, refs)
        self._status(context, f)
        context.area.tag_redraw()

    def _refresh(self, context, event):
        h = self._hover_point(context, event)
        self._preview(context, event, h)
        return h

    # -- modal ------------------------------------------------------------------
    def _place(self, context, event, P):
        """A point picked (a click, C, W): the centre, the first reference, or the second (done)."""
        if P is None:
            return None
        P = np.asarray(P, float)
        if self._c is None:
            self._c = P
            self._normal = np.asarray(self._view_plane(context, P)[1], float)
            self._refresh(context, event)
            return None
        typed = self._live_value(context)
        if self._r1 is None and not (typed is not None and typed[0] == "F"):
            d = self._dir(context, P)
            if abs(ref_length(self._c, P, self._smode, d if d is not None else (0, 0, 1))) <= EPS:
                self.report({'INFO'}, "Scale: the reference must lie away from the centre")
                return None
            self._r1 = P
            self._refresh(context, event)
            return None
        return self._finish(context, event, P)

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        et, ev = event.type, event.value
        if et == self.held_key:
            return {'RUNNING_MODAL'}
        if self._typed_key(context, event):
            self._refresh(context, event)
            return {'RUNNING_MODAL'}
        if et in REFRESH_KEYS or self.snap_key(context, event):
            self._refresh(context, event)
        elif et == 'LEFTMOUSE' and ev == 'PRESS':
            P, _ = self._hover_point(context, event)
            r = self._place(context, event, P)
            if r is not None:
                return r
        elif ev == 'PRESS' and et in {'X', 'Y', 'Z'} and not (event.ctrl or event.alt or event.oskey):
            self._axis_key(context, event)
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'M' and not (event.ctrl or event.alt or event.oskey):
            self._smode, self._axis = MODE_NEXT[self._smode], None
            self._refresh(context, event)
        elif ev == 'PRESS' and et in {'C', 'W'} and not (event.ctrl or event.alt or event.oskey):
            P = np.array(context.scene.cursor.location, float) if et == 'C' else np.zeros(3)
            r = self._place(context, event, P)
            if r is not None:
                return r
        elif ev == 'PRESS' and et == 'B' and self._c is not None:
            self._c, self._r1, self._normal = None, None, None
            self._buf, self._value = "", None
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'P' and not self._edit:
            self._pivot = not self._pivot
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'TAB':
            if self._fix_buffer(context):
                self._refresh(context, event)
        elif ev == 'PRESS' and et == 'BACK_SPACE':
            if self._buf:
                self._buf = self._buf[:-1]
            elif self._value is not None:
                self._value = None
            elif self._r1 is not None:
                self._r1 = None
            elif self._c is not None:
                self._c, self._normal = None, None
            self._refresh(context, event)
        elif ev == 'PRESS' and et in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
            if self._c is not None:
                P, _ = self._hover_point(context, event)
                f, _ = self._factor(context, event, P)
                if f is not None:
                    return self._finish(context, event, P)
        elif ev == 'PRESS' and et in {'ESC', 'RIGHTMOUSE'}:
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context, event, P):
        f, _ = self._factor(context, event, P)
        d = self._dir(context, P)
        C = self._c
        self._end(context)
        if f is None or abs(f - 1.0) <= 1e-12:
            return {'CANCELLED'}
        if f <= 0:
            self.report({'WARNING'}, "Scale: the second reference lies past the centre (the factor would be "
                                     f"{f:.3g}); nothing scaled")
            return {'CANCELLED'}
        self.center = tuple(map(float, C))
        self.factor = float(f)
        self.mode = self._smode
        self.direction = tuple(map(float, d if d is not None else (0.0, 0.0, 1.0)))
        self.set_pivot = bool(self._pivot)
        return self.execute(context)


# --------------------------------------------------------------------------
# Hold the key: Blender's tool on a tap, this one on a hold
# --------------------------------------------------------------------------

# kind: (key, Blender's operator on a tap, ours on a hold); an operator is
# its id or (id, {property: value}). "." is Blender's Pivot Point pie.
HOLD_TOOLS = {"MOVE": ('G', "transform.translate", NPROTO_OT_move.bl_idname),
              "ROTATE": ('R', "transform.rotate", NPROTO_OT_rotate.bl_idname),
              "SCALE": ('S', "transform.resize", NPROTO_OT_scale.bl_idname),
              "PIVOT": ('PERIOD', ("wm.call_menu_pie", {"name": "VIEW3D_MT_pivot_pie"}), "nproto.move_pivot")}


def _op(idname):
    idname, props = (idname, {}) if isinstance(idname, str) else idname
    mod, name = idname.split(".")
    return getattr(getattr(bpy.ops, mod), name), props


def _run(idname):
    op, props = _op(idname)
    return op('INVOKE_DEFAULT', **props)


class NPROTO_OT_transform_hold(bpy.types.Operator):
    """Tap: Blender's own Move / Rotate / Scale (or the Pivot Point pie on ".").
    Hold: Move / Rotate / Scale by Points (picked points, copies), or Move Pivot
    on "." -- the add-on's Hold Time preference sets how long"""
    bl_idname = "nproto.transform_hold"
    bl_label = "Blender's Tool (tap) / By Points (hold)"
    bl_options = {'INTERNAL'}

    kind: EnumProperty(items=[("MOVE", "Move", ""), ("ROTATE", "Rotate", ""), ("SCALE", "Scale", ""),
                              ("PIVOT", "Move Pivot", "")], default="MOVE",
                       options={'HIDDEN'})

    def _native(self):
        try:
            _run(HOLD_TOOLS[self.kind][1])
        except RuntimeError as exc:
            print("NURBS hold key:", exc)
        return {'FINISHED'}

    def _stop(self, context):
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def invoke(self, context, event):
        if event.is_repeat:
            return {'CANCELLED'}
        if not hold_enabled() or not _op(HOLD_TOOLS[self.kind][2])[0].poll():
            return self._native()
        self._key = event.type
        self._t0 = time.monotonic()
        self._timer = context.window_manager.event_timer_add(0.02, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == self._key:
            if event.value == 'RELEASE':
                self._stop(context)
                return self._native()
            return {'RUNNING_MODAL'}                # key repeat while held
        if event.type == 'TIMER':
            if (time.monotonic() - self._t0) * 1000.0 >= hold_ms():
                self._stop(context)
                try:
                    _run(HOLD_TOOLS[self.kind][2])
                except RuntimeError as exc:
                    print("NURBS hold key:", exc)
                return {'FINISHED'}
            return {'RUNNING_MODAL'}
        if event.type == 'ESC':
            self._stop(context)
            return {'CANCELLED'}
        if event.value == 'PRESS' and event.type not in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._stop(context)                     # another key while it is down (G X typed fast): a tap
            self._native()
            return {'FINISHED', 'PASS_THROUGH'}
        return {'PASS_THROUGH'}


def menu_transform(self, context):
    self.layout.operator(NPROTO_OT_move.bl_idname, icon='EMPTY_ARROWS')
    self.layout.operator(NPROTO_OT_rotate.bl_idname, icon='DRIVER_ROTATIONAL_DIFFERENCE')
    self.layout.operator(NPROTO_OT_scale.bl_idname, icon='FULLSCREEN_ENTER')


MENUS = ("VIEW3D_MT_transform_object", "VIEW3D_MT_transform")
KEYMAPS = ("Object Mode", "Mesh", "Curve", "Lattice")       # where Blender binds G / R / S (before "3D View"'s ".")

classes = (NPROTO_PG_move_place, NPROTO_OT_move, NPROTO_OT_rotate, NPROTO_OT_scale, NPROTO_OT_transform_hold)
_keymaps = []


def register():
    for m in MENUS:
        getattr(bpy.types, m).append(menu_transform)
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is not None:
        for name in KEYMAPS:
            km = kc.keymaps.new(name=name, space_type='EMPTY')
            for kind, (key, _native, _ours) in HOLD_TOOLS.items():
                kmi = km.keymap_items.new(NPROTO_OT_transform_hold.bl_idname, key, 'PRESS')
                kmi.properties.kind = kind
                _keymaps.append((km, kmi))


def unregister():
    for km, kmi in _keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _keymaps.clear()
    for m in MENUS:
        try:
            getattr(bpy.types, m).remove(menu_transform)
        except Exception:
            pass
