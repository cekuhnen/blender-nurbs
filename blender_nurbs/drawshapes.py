"""Circle and Arc drawing tools (Create Curves).

Interactive, with the sketch tools' snapping, smart guides and typed values
(drawtools.SketchMixin). Each ends as an exact NURBS circle / arc feature
(prim_circle / prim_arc) placed by its object matrix, so its radius and angle
stay editable in the Feature panel.

Circle (M cycles the mode):
  Centre    click the centre, then the radius (mouse, or type it)
  2 Points  two points across it: the diameter (type: the diameter)
  3 Points  through three points
  Tangent   tangent to two straight lines / edges: click both, then the mouse
            picks the corner and the size (or type the radius)
Arc (M):
  3 Points  start, end, then a point it passes through
  Centre    centre, start (the radius), then the end: the mouse sweeps either
            way round (type: the angle)
  Tangent   start on a curve, line or mesh edge, leaving along it, then the end
            (type: the radius, Tab the angle -- 90 is a half circle)
  Fillet    tangent to two straight lines / edges: click both, then the corner
            and the radius (mouse, or type it): the arc between the tangent points
"""

import json
import math

import numpy as np

# --------------------------------------------------------------------------
# Geometry (pure numpy)
# --------------------------------------------------------------------------


def _unit(v):
    v = np.asarray(v, float)
    return v / (float(np.linalg.norm(v)) or 1.0)


def circle_3(A, B, C):
    """(centre, radius, unit normal) of the circle through A, B, C; None when
    they lie on one line."""
    A, B, C = (np.asarray(x, float) for x in (A, B, C))
    ab, ac = B - A, C - A
    n = np.cross(ab, ac)
    nn = float(n @ n)
    if nn <= 1e-24 * max(float(ab @ ab) * float(ac @ ac), 1e-300):
        return None
    O = A + (float(ac @ ac) * np.cross(n, ab) + float(ab @ ab) * np.cross(ac, n)) / (2.0 * nn)
    return O, float(np.linalg.norm(A - O)), n / math.sqrt(nn)


def corner_circle(L1, L2, P, R=None):
    """The circle tangent to two lines (point, direction) in the corner P lies
    in: {centre, radius, t1, t2 (tangent points), corner, bisector, normal};
    radius R, or (R None) the one whose centre is P's foot on the bisector.
    None for parallel or skew lines."""
    p1, d1 = np.asarray(L1[0], float), _unit(L1[1])
    p2, d2 = np.asarray(L2[0], float), _unit(L2[1])
    n = np.cross(d1, d2)
    if float(np.linalg.norm(n)) < 1e-9:
        return None
    n = _unit(n)
    ab, *_ = np.linalg.lstsq(np.stack([d1, -d2], axis=1), p2 - p1, rcond=None)
    X = p1 + ab[0] * d1
    if float(np.linalg.norm(X - (p2 + ab[1] * d2))) > 1e-7 * (1.0 + float(np.linalg.norm(X))):
        return None                                   # skew: no common plane
    w = np.asarray(P, float) - X
    w -= float(w @ n) * n
    st, *_ = np.linalg.lstsq(np.stack([d1, d2], axis=1), w, rcond=None)
    e1 = d1 if st[0] >= 0 else -d1
    e2 = d2 if st[1] >= 0 else -d2
    half = 0.5 * math.acos(min(max(float(e1 @ e2), -1.0), 1.0))
    if half < 1e-9:
        return None
    u = _unit(e1 + e2)
    if R is None:
        R = max(float(w @ u), 1e-12) * math.sin(half)
    R = float(R)
    C = X + (R / math.sin(half)) * u
    k = R / math.tan(half)
    return {"centre": C, "radius": R, "t1": X + k * e1, "t2": X + k * e2, "corner": X, "bisector": u, "normal": n}


def centre_arc(C, S, sweep, n):
    """(start, middle, end) of the arc about C from S, sweeping `sweep` radians
    about the unit normal n (negative: the other way)."""
    C, S = np.asarray(C, float), np.asarray(S, float)
    r = float(np.linalg.norm(S - C))
    x = (S - C) / r
    y = np.cross(np.asarray(n, float), x)
    at = lambda a: C + r * (math.cos(a) * x + math.sin(a) * y)
    return S, at(0.5 * sweep), at(sweep)


def arc_through(S, P, E):
    """Middle point of the arc from S to E through P; None when they lie on one line."""
    c3 = circle_3(S, P, E)
    if c3 is None:
        return None
    C, r, _ = c3
    S, P, E = (np.asarray(x, float) for x in (S, P, E))
    n = _unit(np.cross(P - S, E - P))          # S -> P -> E runs counter-clockwise about n
    x = (S - C) / r
    y = np.cross(n, x)
    thE = math.atan2(float((E - C) @ y), float((E - C) @ x)) % (2.0 * math.pi)
    return C + r * (math.cos(thE / 2) * x + math.sin(thE / 2) * y)


def signed_angle(a, b, n):
    """Angle from a to b about n, in (-pi, pi]."""
    return math.atan2(float(np.cross(a, b) @ n), float(a @ b))


def arc_frame(S, M, E):
    """(4x4 world matrix, radius, angle in degrees) of the arc S -> M -> E as a
    prim_arc: centred, starting on local +X and running counter-clockwise."""
    from .drawtools import arc_circle
    C, r, x, y, th = arc_circle(S, M, E)
    z = np.cross(x, y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, C
    return T, r, math.degrees(th)


def circle_frame(C, n, x=None):
    """4x4 world matrix of a circle centred at C in the plane with normal n."""
    n = _unit(n)
    if x is None or float(np.linalg.norm(np.asarray(x, float) - float(np.asarray(x, float) @ n) * n)) < 1e-12:
        x = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _unit(np.asarray(x, float) - float(np.asarray(x, float) @ n) * n)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(n, x), n, np.asarray(C, float)
    return T


def frame(C, x, n):
    """4x4 world matrix: origin C, local X along x (in the plane), Z the plane's normal n."""
    n = _unit(n)
    x = _unit(np.asarray(x, float) - float(np.asarray(x, float) @ n) * n)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(n, x), n, np.asarray(C, float)
    return T


def to_world(T, pts):
    return [T[:3, :3] @ np.asarray(p, float) + T[:3, 3] for p in pts]


def rounded_outline(hw, hh, r, k=8):
    """A (rounded) rectangle's outline, half sizes hw, hh, corner radius r (local XY)."""
    r = max(0.0, min(float(r), hw, hh))
    if r <= 0.0:
        pts = [(hw, -hh, 0.0), (hw, hh, 0.0), (-hw, hh, 0.0), (-hw, -hh, 0.0)]
        return pts + [pts[0]]
    pts = []
    for cx, cy, a0 in ((hw - r, -hh + r, -90.0), (hw - r, hh - r, 0.0), (-hw + r, hh - r, 90.0), (-hw + r, -hh + r, 180.0)):
        for j in range(k + 1):
            a = math.radians(a0 + 90.0 * j / k)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a), 0.0))
    return pts + [pts[0]]


def ellipse_outline(a, b, k=64):
    return [(a * math.cos(t), b * math.sin(t), 0.0) for t in np.linspace(0.0, 2.0 * math.pi, k + 1)]


def polygon_outline(R, n):
    return [(R * math.cos(2 * math.pi * k / n), R * math.sin(2 * math.pi * k / n), 0.0) for k in range(n + 1)]


def circle_points(C, r, n, x=None, k=64):
    T = circle_frame(C, n, x)
    return [T[:3, 3] + r * (math.cos(a) * T[:3, 0] + math.sin(a) * T[:3, 1])
            for a in np.linspace(0.0, 2.0 * math.pi, k + 1)]


# --------------------------------------------------------------------------
# Blender side
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import EnumProperty, FloatProperty, StringProperty  # noqa: E402
from bpy_extras import view3d_utils  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

from . import draw, hints, linesolve  # noqa: E402
from .drawtools import (MOUSE_SHAPE, REFRESH_KEYS, SNAP_PX, SketchMixin, _select_new, arc_points,  # noqa: E402
                        parse_length, tangent_arc)

COL_PICKED = (1.0, 1.0, 1.0, 1.0)
COL_HOVER_LINE = (1.0, 0.85, 0.2, 1.0)


class _ShapeTool(SketchMixin):
    """Clicks, typed values, picked lines; subclasses say what a mode needs
    (MODES, _shape) and what it makes (_make)."""
    MODES = ()
    TITLE = ""
    PARAMS = ()             # operator properties a generic shape's parameters go through (the redo panel edits them)
    FIELD_NAMES = {"R": "Radius", "D": "Diameter", "A": "Angle", "R2": "Minor", "X": "Width", "Y": "Height",
                   "C": "Corner", "L": "Length", "W": "Width"}
    FIELD_HINTS = {"R": "the radius", "D": "the diameter", "A": "the angle", "R2": "the minor radius",
                   "X": "the width", "Y": "the height", "C": "the corner radius", "L": "the length", "W": "the width"}
    NUMBERS = ("A", "T")    # typed as plain numbers (an angle, a count), not lengths

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, f"{self.bl_label}: run it in a 3D View")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self._reset()
        self.sketch_start(context)
        context.window_manager.modal_handler_add(self)
        self._preview(context, event)
        return {'RUNNING_MODAL'}

    # -- state ---------------------------------------------------------------
    def _reset(self):
        self._pts, self._lines_picked = [], []
        self._plane = None
        self._sweep, self._last_ang = 0.0, None
        self._tangent, self._tangent_fixed = None, False
        self._reset_typing()

    def _reset_typing(self):
        self._buf, self._field, self._vals = "", self._fields()[0] if self._fields() else "", {}

    def _fields(self):
        """The values this mode takes typed, now: e.g. ["R"], ["R", "A"]."""
        return []

    def typing_ready(self):
        return bool(self._fields()) and (bool(self._pts) or bool(self._lines_picked))

    def _typed(self, context):
        """{field: value} typed so far, the one being typed included."""
        out = dict(self._vals)
        if self._buf and self._field:
            try:
                out[self._field] = (float(eval(self._buf, {"__builtins__": {}}, {})) if self._field in self.NUMBERS
                                    else parse_length(context, self._buf))
            except Exception:
                pass
        return out

    def guide_points(self):
        return list(self._pts)

    def guide_axes(self):
        """The legs between the clicked points, newest first: guides run along
        and square to them through every point -- an ellipse's major and minor
        axes, a rotated rectangle's sides, a slot's axis."""
        P = [np.asarray(p, float) for p in self._pts]
        return [_unit(P[k + 1] - P[k]) for k in range(len(P) - 2, -1, -1) if np.linalg.norm(P[k + 1] - P[k]) > 1e-12]

    # -- lines for the tangent modes ------------------------------------------
    def _curve_of(self, key):
        if getattr(self, "_curves", None) is None:
            self._curves = {}
        if key not in self._curves:
            from . import features, store
            c = None
            obj = bpy.data.objects.get(key[0])
            if obj is not None:
                try:
                    c = linesolve.kernel_curve(features.input_curve(obj, key[1]), store._k()[0])
                except Exception as exc:
                    print(f"{self.TITLE}: cannot evaluate", key, exc)
            self._curves[key] = c
        return self._curves[key]

    def _line_under(self, context, event):
        """(point, direction, polyline) of the straight line / edge under the mouse, or None."""
        near = self._snaps.near_lines(self._view(context), self._mouse(context, event), SNAP_PX, limit=1)
        if not near:
            return None
        W, src = self._snaps.line_info(near[0][0])
        c = self._curve_of(tuple(src[:2])) if src is not None else None
        if c is None or not c.straight:
            return None
        return (np.asarray(c.center, float), _unit(c.direction), W)

    def _mouse_on(self, context, event, O, n):
        return self._ray_point(context, Vector(self._mouse(context, event)), (np.asarray(O, float), _unit(n)))

    # -- modal -------------------------------------------------------------------
    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        ch = self.typed_char(event)
        if self.typing_ready() and ch and (ch in "0123456789.-+*/'\"" or (self._buf and (ch.isalpha() or ch == " "))) \
                and not (event.ctrl or event.alt or event.oskey):
            self._buf += ch
            self._preview(context, event)
            return {'RUNNING_MODAL'}
        if event.type in REFRESH_KEYS or self.snap_key(context, event):
            self._preview(context, event)
        elif self._extra_key(context, event):
            self._preview(context, event)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            done = self._click(context, event)
            if done is not None:
                return done
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type == 'TAB' and self.typing_ready() and len(self._fields()) > 1:
            self._vals = self._typed(context)
            fs = self._fields()
            self._buf, self._field = "", fs[(fs.index(self._field) + 1) % len(fs)] if self._field in fs else fs[0]
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and self.typing_ready():
            if self._typed(context):
                done = self._enter(context, event)
                if done is not None:
                    return done
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type == 'BACK_SPACE':
            if self._buf:
                self._buf = self._buf[:-1]
            elif self._vals:
                self._reset_typing()
            elif getattr(self, "_picks", None):
                self._picks.pop()                       # a picked curve (Fillet)
            elif self._lines_picked and len(self._lines_picked) >= len(self._pts):
                self._lines_picked.pop()
            elif self._pts:
                self._pts.pop()
                self._sweep, self._last_ang, self._tangent = 0.0, None, None
            if not self._pts and not self._lines_picked:
                self._plane = None
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type == 'M' and not self._buf:
            ids = [m[0] for m in self.MODES]
            self.mode = ids[(ids.index(self.mode) + 1) % len(ids)]
            self._reset()
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type in {'ESC', 'RIGHTMOUSE'}:
            self.sketch_end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _extra_key(self, context, event):
        """A tool's own key (True when it was one)."""
        return False

    def _extra_rows(self):
        return []

    def _enter(self, context, event):
        """Enter with a typed value: place the shape (None: not yet)."""
        sh = self._shape(context, event, final=True)
        return None if sh is None else self._finish(context, sh)

    def _point(self, context, event):
        """(snapped point, snap) under the mouse, on the drawing plane when free."""
        prev = self._pts[-1] if self._pts else None
        return self.pick_point(context, event, prev=prev)

    def _add_point(self, context, P):
        self.fix_plane(context, P)
        self._pts.append(np.asarray(P, float))
        self._reset_typing()

    def _pick_line(self, context, event):
        ln = self._line_under(context, event)
        if ln is None:
            self.report({'WARNING'}, f"{self.TITLE}: click a straight line or edge")
            return
        if self._plane is None:
            W = np.asarray(ln[2], float)
            f = self._face_under(context, self._mouse(context, event), getattr(event, "alt", False))
            if f is not None and self.on_face(f, W[0]) and self.on_face(f, W[-1]):
                self.fix_plane(context, ln[0], on=(W[0], W[-1]))     # a line drawn on a flat face: its plane
            else:
                self._face_hover = None
                P = self._mouse_on(context, event, ln[0], self._view_plane(context, ln[0])[1])
                self.fix_plane(context, P if P is not None else ln[0])
        self._lines_picked.append(ln)
        self._reset_typing()

    # -- preview ------------------------------------------------------------------
    def _preview(self, context, event):
        ov = draw.sketch_overlay
        ov["curves"], ov["refs"] = [], []
        ov["cvs"] = [Vector(p) for p in self._pts]
        hover = None
        if self._needs_line():
            ln = self._line_under(context, event)
            if ln is not None:
                ov["refs"].append(([Vector(p) for p in ln[2]], COL_HOVER_LINE, 3.0))
            ov["snap"], ov["snap_label"] = None, None
            ov["cursor"] = None
            draw.sketch_overlay["guides"] = []
        else:
            hover = self._point(context, event)
            ov["snap"] = Vector(hover[0]) if hover[1] is not None else None
            ov["snap_label"] = hover[1][3] if hover[1] is not None else None
            ov["cursor"] = Vector(hover[0]) if hover[0] is not None else None
        for ln in self._lines_picked:
            ov["refs"].append(([Vector(p) for p in ln[2]], COL_PICKED, 2.0))
        sh = self._shape(context, event)
        if sh is not None:
            ov["curves"] = [[Vector(p) for p in self._polyline(sh)]]
        self._status(context, sh)
        context.area.tag_redraw()

    @staticmethod
    def _polyline(sh):
        if sh["kind"] == "circle":
            return circle_points(sh["centre"], sh["radius"], sh["normal"], sh.get("x"))
        if sh["kind"] == "arc":
            return arc_points(sh["S"], sh["M"], sh["E"], 48)
        return sh["poly"]

    # -- status and key hints ------------------------------------------------------
    def _status(self, context, sh):
        us = context.scene.unit_settings
        fmt = lambda x: bpy.utils.units.to_string(us.system, 'LENGTH', x * (us.scale_length or 1.0), precision=4)
        mode = dict((m[0], m[1]) for m in self.MODES)[self.mode]
        state = [mode, self._step()]
        typed = self._typed(context)
        names = self.FIELD_NAMES
        for f in self._fields():
            if self._field == f and self._buf:
                txt = f"[{self._buf}_]"
            elif f in typed:
                txt = f"{typed[f]:g}°" if f == "A" else (f"{typed[f]:g}" if f in self.NUMBERS else fmt(typed[f]))
            elif sh is not None and f in (sh.get("vals") or {}):
                v_ = sh["vals"][f]
                txt = f"{v_:.1f}°" if f == "A" else (f"{v_:.3g}" if f in self.NUMBERS else fmt(v_))
            elif sh is not None and f in ("R", "D") and "radius" in sh:
                txt = fmt(sh["radius"] * (2.0 if f == "D" else 1.0))
            elif sh is not None and f == "A" and sh.get("angle") is not None:
                txt = f"{sh['angle']:.1f}°"
            else:
                txt = "-"
            if self._field == f and not self._buf and len(self._fields()) > 1:
                txt = f"[{txt}]"
            state.append(f"{names[f]} {txt}")
        if sh is not None and sh.get("note"):
            state.append(sh["note"])
        if sh is not None and sh["kind"] == "arc" and "A" not in self._fields():
            state.append(f"Radius {fmt(sh['radius'])}  ·  {sh['angle']:.1f}°")
        ready = self.typing_ready()
        fs = self._fields()
        rows = [("Mode", [hints.cycle("M", [m[1].lower() for m in self.MODES], mode.lower())])]
        if fs:
            tab = ("switch to the angle (90: half circle)" if self.mode == "TANGENT" else "switch to the angle") \
                if fs == ["R", "A"] else "next value (" + ", ".join(self.FIELD_NAMES[f].lower() for f in fs) + ")"
            rows.append(("Type", [hints.key("123", self.FIELD_HINTS[fs[0]] + ("" if ready else ", after the first click"),
                                            on=(True if self._buf else None) if ready else False)]
                         + ([hints.key("Tab", tab)] if len(fs) > 1 else [])
                         + [hints.key("Enter", "place it"), hints.key("Backspace", "undo")]))
        else:
            rows.append(("Place", [hints.key("Backspace", "undo")]))
        rows += self._extra_rows()
        rows.append(("Snap", self.snap_hints()))
        hints.show(context, self.TITLE, state, rows, MOUSE_SHAPE)

    # -- finish ---------------------------------------------------------------------
    def _finish(self, context, sh):
        if sh is None or not (sh.get("ok", True) and sh.get("radius", 1.0) > 1e-9):
            self.report({'WARNING'}, f"{self.TITLE}: give it a size")
            return None
        if "op" in sh:                                      # a generic shape: op, params, matrix
            for k in self.PARAMS:
                setattr(self, k, sh["params"][k])
            self.data = json.dumps({"matrix": sh["T"].tolist(), "op": sh["op"], "params": sh["params"],
                                    "name": sh["name"]})
            self.sketch_end(context)
            return self.execute(context)
        if sh["kind"] == "circle":
            T = circle_frame(sh["centre"], sh["normal"], sh.get("x"))
            self.radius = sh["radius"]
            self.data = json.dumps({"matrix": T.tolist()})
        else:
            T, r, ang = arc_frame(sh["S"], sh["M"], sh["E"])
            self.radius, self.angle = r, ang
            self.data = json.dumps({"matrix": T.tolist()})
        self.sketch_end(context)
        return self.execute(context)

    def _make_generic(self, context):
        from .feature_ui import place_primitive
        try:
            d = json.loads(self.data or "{}")
        except ValueError:
            d = {}
        if not d.get("matrix"):
            self.report({'WARNING'}, f"{self.TITLE} is interactive: start it from the panel or menu")
            return {'CANCELLED'}
        params = dict(d["params"])
        for k in self.PARAMS:                               # the redo panel's values
            params[k] = getattr(self, k)
        obj = place_primitive(context, d["name"], d["op"], params, Matrix(d["matrix"]))
        _select_new(context, [obj])
        return {'FINISHED'}

    def _make(self, context, kind):
        from .feature_ui import place_primitive, primitive_params
        try:
            d = json.loads(self.data or "{}")
        except ValueError:
            d = {}
        if not d.get("matrix"):
            self.report({'WARNING'}, f"{self.TITLE} is interactive: start it from the panel or menu")
            return {'CANCELLED'}
        if kind == "arc":
            params = primitive_params("prim_arc", radius=float(self.radius), angle=float(self.angle), degree=0)
            obj = place_primitive(context, "NURBS Arc", "prim_arc", params, Matrix(d["matrix"]))
        else:
            params = primitive_params("prim_circle", radius=float(self.radius), degree=0, elliptic=False)
            obj = place_primitive(context, "NURBS Circle", "prim_circle", params, Matrix(d["matrix"]))
        _select_new(context, [obj])
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Circle
# --------------------------------------------------------------------------

CIRCLE_MODES = [("CENTER", "Centre", "Click the centre, then the radius"),
                ("TWO", "2 Points", "Two points across the circle: its diameter"),
                ("THREE", "3 Points", "Through three points"),
                ("TANGENT", "Tangent", "Tangent to two straight lines or edges, in the corner the mouse is in")]


class NPROTO_OT_draw_circle(_ShapeTool, bpy.types.Operator):
    """Draw an exact NURBS circle: by its centre and radius, by 2 points across
    it, through 3 points, or tangent to two lines (M switches). Snaps like the
    other drawing tools; type the radius. It stays editable in the Feature panel"""
    bl_idname = "nproto.draw_circle"
    bl_label = "Circle"
    bl_options = {'REGISTER', 'UNDO'}
    MODES = CIRCLE_MODES
    TITLE = "Circle"

    mode: EnumProperty(name="Mode", items=CIRCLE_MODES, default="CENTER")
    radius: FloatProperty(name="Radius", default=0.5, min=1e-6, unit='LENGTH')
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.prop(self, "radius")

    def _needs_line(self):
        return self.mode == "TANGENT" and len(self._lines_picked) < 2

    def _fields(self):
        if self.mode == "CENTER":
            return ["R"]
        if self.mode == "TWO":
            return ["D"]
        if self.mode == "TANGENT":
            return ["R"]
        return []

    def _step(self):
        if self.mode == "TANGENT":
            return ["click the first line", "click the second line", "pick the corner and the size"][
                min(len(self._lines_picked), 2)]
        need = {"CENTER": ["click the centre", "click the radius"], "TWO": ["click a point", "click across"],
                "THREE": ["click the first point", "click the second point", "click the third point"]}[self.mode]
        return need[min(len(self._pts), len(need) - 1)]

    def _shape(self, context, event, final=False):
        typed = self._typed(context)
        if self.mode == "TANGENT":
            if len(self._lines_picked) < 2:
                return None
            L1, L2 = self._lines_picked
            cc = corner_circle(L1[:2], L2[:2], L1[0])
            if cc is None:
                return None
            P = self._mouse_on(context, event, cc["corner"], cc["normal"])
            cc = corner_circle(L1[:2], L2[:2], P if P is not None else cc["corner"], typed.get("R"))
            return None if cc is None else {"kind": "circle", "centre": cc["centre"], "radius": cc["radius"],
                                            "normal": cc["normal"]}
        if not self._pts:
            return None
        O, n, u, v = self._plane
        P, _ = self._point(context, event)
        if self.mode == "CENTER":
            C = self._pts[0]
            if "R" in typed:
                return {"kind": "circle", "centre": C, "radius": typed["R"], "normal": n, "x": u}
            if P is None:
                return None
            w = P - C
            w -= float(w @ n) * n
            return {"kind": "circle", "centre": C, "radius": float(np.linalg.norm(w)), "normal": n, "x": _unit(w)}
        if self.mode == "TWO":
            A = self._pts[0]
            if P is None:
                return None
            w = P - A
            w -= float(w @ n) * n
            if "D" in typed:
                w = _unit(w if np.linalg.norm(w) > 1e-12 else u) * typed["D"]
            return {"kind": "circle", "centre": A + 0.5 * w, "radius": 0.5 * float(np.linalg.norm(w)), "normal": n,
                    "x": -_unit(w)}
        if self.mode == "THREE":
            if len(self._pts) < 2 or P is None:
                return None
            c3 = circle_3(self._pts[0], self._pts[1], P)
            if c3 is None:
                return None
            C, r, nn = c3
            if float(nn @ n) < 0:
                nn = -nn
            return {"kind": "circle", "centre": C, "radius": r, "normal": nn, "x": _unit(self._pts[0] - C)}
        return None

    def _click(self, context, event):
        if self.mode == "TANGENT":
            if len(self._lines_picked) < 2:
                self._pick_line(context, event)
                return None
            return self._finish(context, self._shape(context, event, final=True))
        P, _ = self._point(context, event)
        if P is None:
            return None
        need = {"CENTER": 1, "TWO": 1, "THREE": 2}[self.mode]
        if len(self._pts) < need:
            self._add_point(context, P)
            return None
        return self._finish(context, self._shape(context, event, final=True))

    def execute(self, context):
        out = self._make(context, "circle")
        if out == {'FINISHED'}:
            self.report({'INFO'}, f"Circle, radius {self.radius:.4g}")
        return out


# --------------------------------------------------------------------------
# Arc
# --------------------------------------------------------------------------

ARC_MODES = [("THREE", "3 Points", "Start, end, then a point the arc passes through"),
             ("CENTER", "Centre", "Centre, start (the radius), then the end: sweep either way round"),
             ("TANGENT", "Tangent", "Start on a curve, line or mesh edge, leaving along it; then the end"),
             ("FILLET", "Fillet", "Tangent to two straight lines or edges: the arc between the tangent points")]


class NPROTO_OT_draw_arc(_ShapeTool, bpy.types.Operator):
    """Draw an exact NURBS arc: through 3 points, by its centre, tangent from a
    curve or line, or tangent to two lines (a fillet; M switches). Snaps like the
    other drawing tools; type the radius and the angle. It stays editable in the
    Feature panel"""
    bl_idname = "nproto.draw_arc"
    bl_label = "Arc"
    bl_options = {'REGISTER', 'UNDO'}
    MODES = ARC_MODES
    TITLE = "Arc"

    mode: EnumProperty(name="Mode", items=ARC_MODES, default="THREE")
    radius: FloatProperty(name="Radius", default=0.5, min=1e-6, unit='LENGTH')
    angle: FloatProperty(name="Angle", default=90.0, min=0.01, max=360.0)
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        self.layout.use_property_split = True
        col = self.layout.column(align=True)
        col.prop(self, "radius")
        col.prop(self, "angle")

    def _needs_line(self):
        return self.mode == "FILLET" and len(self._lines_picked) < 2

    def _fields(self):
        if self.mode == "CENTER":
            return ["R"] if len(self._pts) < 2 else ["A"]
        if self.mode == "TANGENT":
            return ["R", "A"]
        if self.mode == "FILLET":
            return ["R"]
        return []

    def _step(self):
        if self.mode == "FILLET":
            return ["click the first line", "click the second line", "pick the corner and the radius"][
                min(len(self._lines_picked), 2)]
        need = {"THREE": ["click the start", "click the end", "click a point on the arc"],
                "CENTER": ["click the centre", "click the start", "click the end"],
                "TANGENT": ["click on a curve, line or edge", "click the end"]}[self.mode]
        return need[min(len(self._pts), len(need) - 1)]

    def _start_tangent(self, P, snap):
        """(unit direction, fixed) a Tangent arc leaves P with: along the curve,
        edge or mesh edge it was snapped onto. At a curve's end it is fixed: it
        leaves outward, carrying the curve on; elsewhere the mouse picks the
        way. None if P is on nothing."""
        if snap is None:
            return None
        if snap[2] is not None:
            c = self._curve_of(tuple(snap[2][:2]))
            if c is not None:
                t = linesolve.nearest_t(c, P)
                D = _unit(c.at(t)[2])
                if not c.closed:
                    tol = 1e-6 * c.span
                    if abs(t - c.t0) <= tol:
                        return -D, True                   # the curve's start: leave backwards from it
                    if abs(t - c.t1) <= tol:
                        return D, True
                return D, False
        fr = snap[1]
        if isinstance(fr, np.ndarray) and fr.shape == (2, 3):      # a mesh edge: its direction
            return _unit(fr[0]), False
        return None

    def _shape(self, context, event, final=False):
        typed = self._typed(context)
        if self.mode == "FILLET":
            if len(self._lines_picked) < 2:
                return None
            L1, L2 = self._lines_picked
            cc = corner_circle(L1[:2], L2[:2], L1[0])
            if cc is None:
                return None
            P = self._mouse_on(context, event, cc["corner"], cc["normal"])
            cc = corner_circle(L1[:2], L2[:2], P if P is not None else cc["corner"], typed.get("R"))
            if cc is None:
                return None
            M = cc["centre"] - cc["radius"] * cc["bisector"]
            return self._arc(cc["t1"], M, cc["t2"])
        if not self._pts:
            return None
        O, n, u, v = self._plane
        P, _ = self._point(context, event)
        if self.mode == "THREE":
            if len(self._pts) < 2 or P is None:
                return None
            S, E = self._pts
            Mm = arc_through(S, P, E)
            return None if Mm is None else self._arc(S, Mm, E)
        if self.mode == "CENTER":
            C = self._pts[0]
            if len(self._pts) == 1:
                if P is None and "R" not in typed:
                    return None
                w = (P if P is not None else C + u) - C
                w -= float(w @ n) * n
                r = typed.get("R", float(np.linalg.norm(w)))
                return {"kind": "circle", "centre": C, "radius": r, "normal": n, "x": _unit(w)}
            S = self._pts[1]
            if P is not None:
                w = P - C
                w -= float(w @ n) * n
                a = signed_angle(_unit(S - C), _unit(w), n) if np.linalg.norm(w) > 1e-12 else 0.0
                if self._last_ang is not None:              # follow the mouse round, either way, past 180
                    da = a - self._last_ang
                    da = (da + math.pi) % (2 * math.pi) - math.pi
                    self._sweep += da
                else:
                    self._sweep = a
                self._last_ang = a
            sweep = self._sweep
            if "A" in typed:
                sweep = math.copysign(math.radians(min(abs(typed["A"]), 360.0)), sweep or 1.0)
            if abs(sweep) < 1e-9:
                return None
            return self._arc(*centre_arc(C, S, sweep, n))
        if self.mode == "TANGENT":
            if self._tangent is None:
                return None
            S, t = self._pts[0], self._tangent
            nn = _unit(n - float(n @ t) * t) if np.linalg.norm(n - float(n @ t) * t) > 1e-9 else _unit(np.cross(t, u))
            ref = P if P is not None else S + t
            d = ref - S
            if not getattr(self, "_tangent_fixed", False) and float(np.linalg.norm(d)) > 1e-12 \
                    and float(d @ t) < -0.3 * float(np.linalg.norm(d)):
                t = self._tangent = -t           # mid-curve: the way the mouse clearly goes (it holds near 90)
            if "R" in typed or "A" in typed:
                m = _unit(np.cross(nn, t))
                if float((ref - S) @ m) < 0:
                    m = -m
                if "A" in typed:
                    phi = math.radians(min(max(abs(typed["A"]), 0.01), 179.99))
                    d = math.cos(phi) * t + math.sin(phi) * m
                    E = S + (2.0 * typed["R"] * math.sin(phi) if "R" in typed else max(float((ref - S) @ d), 1e-9)) * d
                else:
                    Cc = S + typed["R"] * m
                    w = ref - Cc
                    w -= float(w @ nn) * nn
                    E = Cc + typed["R"] * _unit(w if np.linalg.norm(w) > 1e-12 else -m)
            else:
                if P is None:
                    return None
                E = P
            Mm = tangent_arc(S, t, E)
            return None if Mm is None else self._arc(S, Mm, E)
        return None

    @staticmethod
    def _arc(S, M, E):
        from .drawtools import arc_circle
        C, r, x, y, th = arc_circle(S, M, E)
        return {"kind": "arc", "S": np.asarray(S, float), "M": np.asarray(M, float), "E": np.asarray(E, float),
                "radius": r, "angle": math.degrees(th)}

    def _enter(self, context, event):
        if self.mode == "CENTER" and len(self._pts) == 1:
            return self._click(context, event)          # the typed radius places the start
        return _ShapeTool._enter(self, context, event)

    def _click(self, context, event):
        if self.mode == "FILLET":
            if len(self._lines_picked) < 2:
                self._pick_line(context, event)
                return None
            return self._finish(context, self._shape(context, event, final=True))
        P, snap = self._point(context, event)
        if P is None:
            return None
        if self.mode == "TANGENT" and not self._pts:
            st = self._start_tangent(P, snap)
            if st is None:
                self.report({'WARNING'}, "Arc: start on a curve, line or edge (it leaves along it)")
                return None
            self._add_point(context, P)
            self._tangent, self._tangent_fixed = st
            return None
        need = {"THREE": 2, "CENTER": 2, "TANGENT": 1}[self.mode]
        if len(self._pts) < need:
            if self.mode == "CENTER" and len(self._pts) == 1:
                sh = self._shape(context, event)              # the start: on the radius circle
                if sh is None:
                    return None
                P = sh["centre"] + sh["radius"] * sh["x"]
                self._pts.append(np.asarray(P, float))
                self._sweep, self._last_ang = 0.0, None
                self._reset_typing()
                return None
            self._add_point(context, P)
            return None
        return self._finish(context, self._shape(context, event, final=True))

    def execute(self, context):
        out = self._make(context, "arc")
        if out == {'FINISHED'}:
            self.report({'INFO'}, f"Arc, radius {self.radius:.4g}, {self.angle:.4g}°")
        return out


# --------------------------------------------------------------------------
# Ellipse, Rectangle, Slot, Polygon
# --------------------------------------------------------------------------

def _flat(w, n):
    w = np.asarray(w, float)
    return w - float(w @ n) * n


class _TwoStep(_ShapeTool):
    """Shapes placed with two or three clicks; Enter with a typed value places
    the next point, or the shape on its last step."""
    STEPS = 2

    def _enter(self, context, event):
        if len(self._pts) < self.STEPS - 1:
            return self._click(context, event)
        return _ShapeTool._enter(self, context, event)

    def _click(self, context, event):
        P, _ = self._point(context, event)
        if P is None:
            return None
        if not self._pts:
            self._add_point(context, P)
            return None
        sh = self._shape(context, event, final=True)
        if len(self._pts) < self.STEPS - 1:
            if sh is None or sh.get("next") is None:
                return None
            self._pts.append(np.asarray(sh["next"], float))       # the point the typed values give
            self._reset_typing()
            return None
        return self._finish(context, sh)

    def execute(self, context):
        out = self._make_generic(context)
        if out == {'FINISHED'}:
            self.report({'INFO'}, self.TITLE + ": " + ", ".join(f"{k} {getattr(self, k):.4g}" for k in self.PARAMS))
        return out


ELLIPSE_MODES = [("CENTER", "Centre", "The centre, the end of the major axis, then the minor size"),
                 ("AXIS", "Axis", "Both ends of the major axis, then the minor size")]


class NPROTO_OT_draw_ellipse(_TwoStep, bpy.types.Operator):
    """Draw an exact NURBS ellipse: its centre (or one end of its major axis),
    the major axis' end, then the minor size (M switches). Type the sizes. It
    stays editable in the Feature panel"""
    bl_idname = "nproto.draw_ellipse"
    bl_label = "Ellipse"
    bl_options = {'REGISTER', 'UNDO'}
    MODES, TITLE, STEPS, PARAMS = ELLIPSE_MODES, "Ellipse", 3, ("radius", "radius2")

    mode: EnumProperty(name="Mode", items=ELLIPSE_MODES, default="CENTER")
    radius: FloatProperty(name="Major Radius", default=0.5, min=1e-6, unit='LENGTH')
    radius2: FloatProperty(name="Minor Radius", default=0.25, min=1e-6, unit='LENGTH')
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        self.layout.use_property_split = True
        col = self.layout.column(align=True)
        col.prop(self, "radius")
        col.prop(self, "radius2")

    def _needs_line(self):
        return False

    def _fields(self):
        if len(self._pts) == 1:
            return ["R"] if self.mode == "CENTER" else ["D"]
        return ["R2"] if len(self._pts) == 2 else []

    def _step(self):
        need = (["click the centre", "click the end of the major axis", "click the minor size"]
                if self.mode == "CENTER" else
                ["click one end of the major axis", "click the other end", "click the minor size"])
        return need[min(len(self._pts), 2)]

    def _shape(self, context, event, final=False):
        if not self._pts:
            return None
        O, n, u, v = self._plane
        P, _ = self._point(context, event)
        typed = self._typed(context)
        if len(self._pts) == 1:
            base = self._pts[0]
            w = _flat((P if P is not None else base + u) - base, n)
            if np.linalg.norm(w) < 1e-12:
                w = u * 1e-9
            if self.mode == "CENTER":
                a_vec = _unit(w) * typed["R"] if "R" in typed else w
                C, nxt = base, base + (_unit(w) * typed["R"] if "R" in typed else w)
            else:
                w = _unit(w) * typed["D"] if "D" in typed else w
                C, a_vec, nxt = base + 0.5 * w, 0.5 * w, base + w
            a = float(np.linalg.norm(a_vec))
            if a < 1e-12:
                return None
            T = frame(C, a_vec, n)
            return {"kind": "ellipse", "poly": to_world(T, ellipse_outline(a, 0.5 * a)), "next": nxt, "ok": False,
                    "vals": {"R": a, "D": 2 * a}}
        A, B = self._pts[0], self._pts[1]
        C, a_vec = (A, B - A) if self.mode == "CENTER" else (0.5 * (A + B), 0.5 * (B - A))
        a = float(np.linalg.norm(a_vec))
        T = frame(C, a_vec, n)
        b = typed["R2"] if "R2" in typed else (abs(float((P - C) @ T[:3, 1])) if P is not None else 0.0)
        if a < 1e-12 or b < 1e-12:
            return None
        params = {"radius": a, "radius2": b, "elliptic": True, "degree": 0}
        from .feature_ui import primitive_params
        return {"kind": "ellipse", "op": "prim_circle", "name": "NURBS Ellipse", "T": T,
                "params": primitive_params("prim_circle", **params), "poly": to_world(T, ellipse_outline(a, b)),
                "vals": {"R2": b}}


RECT_MODES = [("CORNERS", "2 Corners", "Two opposite corners"),
              ("CENTER", "Centre", "The centre, then a corner"),
              ("THREE", "3 Points", "Two corners along one side (its direction), then the other side's length")]


class NPROTO_OT_draw_rectangle(_TwoStep, bpy.types.Operator):
    """Draw a NURBS rectangle: two corners, its centre and a corner, or three
    points (rotated; M switches). Type the width, height and a corner radius
    (exact rounded corners). It stays editable in the Feature panel"""
    bl_idname = "nproto.draw_rectangle"
    bl_label = "Rectangle"
    bl_options = {'REGISTER', 'UNDO'}
    MODES, TITLE, PARAMS = RECT_MODES, "Rectangle", ("width", "depth", "corner")

    mode: EnumProperty(name="Mode", items=RECT_MODES, default="CORNERS")
    width: FloatProperty(name="Width", default=1.0, min=1e-6, unit='LENGTH')
    depth: FloatProperty(name="Height", default=0.5, min=1e-6, unit='LENGTH')
    corner: FloatProperty(name="Corner Radius", default=0.0, min=0.0, unit='LENGTH',
                          description="Round the corners (exact quarter circles); kept for the next rectangle")
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    @property
    def STEPS(self):
        return 3 if self.mode == "THREE" else 2

    def draw(self, context):
        self.layout.use_property_split = True
        col = self.layout.column(align=True)
        col.prop(self, "width")
        col.prop(self, "depth")
        col.prop(self, "corner")

    def _needs_line(self):
        return False

    def _fields(self):
        if self.mode == "THREE":
            return ["X"] if len(self._pts) == 1 else (["Y", "C"] if len(self._pts) == 2 else [])
        return ["X", "Y", "C"] if len(self._pts) == 1 else []

    def _step(self):
        if self.mode == "THREE":
            return ["click a corner", "click the next corner (the side's direction)", "click the other side"][
                min(len(self._pts), 2)]
        return ["click a corner" if self.mode == "CORNERS" else "click the centre",
                "click the opposite corner" if self.mode == "CORNERS" else "click a corner"][min(len(self._pts), 1)]

    def _shape(self, context, event, final=False):
        if not self._pts:
            return None
        O, n, u, v = self._plane
        P, _ = self._point(context, event)
        typed = self._typed(context)
        A = self._pts[0]
        if self.mode == "THREE" and len(self._pts) == 1:
            w = _flat((P if P is not None else A + u) - A, n)
            L = typed.get("X", float(np.linalg.norm(w)))
            B = A + _unit(w if np.linalg.norm(w) > 1e-12 else u) * L
            return {"kind": "rect", "poly": [A, B], "next": B, "ok": False, "vals": {"X": L}}
        if self.mode == "THREE":
            B = self._pts[1]
            x = _unit(B - A)
            X = float(np.linalg.norm(B - A))
            y = np.cross(_unit(n), x)
            h = float((P - A) @ y) if P is not None else 0.0
            Y = typed.get("Y", abs(h))
            C = 0.5 * (A + B) + 0.5 * math.copysign(Y, h or 1.0) * y
            xdir = x
        else:
            w = _flat((P if P is not None else A + u + v) - A, n)
            dx, dy = float(w @ u), float(w @ v)
            k = 1.0 if self.mode == "CORNERS" else 2.0
            X, Y = typed.get("X", k * abs(dx)), typed.get("Y", k * abs(dy))
            C = A + 0.5 * (math.copysign(X, dx or 1.0) * u + math.copysign(Y, dy or 1.0) * v) \
                if self.mode == "CORNERS" else A
            xdir = u
        if X < 1e-12 or Y < 1e-12:
            return None
        r = min(typed.get("C", float(self.corner)), 0.5 * min(X, Y))
        T = frame(C, xdir, n)
        from .feature_ui import primitive_params
        return {"kind": "rect", "op": "prim_rectangle", "name": "NURBS Rectangle", "T": T,
                "params": primitive_params("prim_rectangle", width=X, depth=Y, corner=r),
                "poly": to_world(T, rounded_outline(0.5 * X, 0.5 * Y, r)), "vals": {"X": X, "Y": Y, "C": r}}


SLOT_MODES = [("CENTERS", "Centre to Centre", "Click both arc centres, then the width")]


class NPROTO_OT_draw_slot(_TwoStep, bpy.types.Operator):
    """Draw a slot: click both arc centres, then the width (type the length and
    the width). An exact NURBS curve: two lines and two half circles. It stays
    editable in the Feature panel (length, width)"""
    bl_idname = "nproto.draw_slot"
    bl_label = "Slot"
    bl_options = {'REGISTER', 'UNDO'}
    MODES, TITLE, STEPS, PARAMS = SLOT_MODES, "Slot", 3, ("length", "width")

    mode: EnumProperty(name="Mode", items=SLOT_MODES, default="CENTERS")
    length: FloatProperty(name="Length", default=1.0, min=0.0, unit='LENGTH', description="From one arc centre to the other")
    width: FloatProperty(name="Width", default=0.4, min=1e-6, unit='LENGTH')
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        self.layout.use_property_split = True
        col = self.layout.column(align=True)
        col.prop(self, "length")
        col.prop(self, "width")

    def _needs_line(self):
        return False

    def _fields(self):
        return ["L"] if len(self._pts) == 1 else (["W"] if len(self._pts) == 2 else [])

    def _step(self):
        return ["click one arc centre", "click the other arc centre", "click the width"][min(len(self._pts), 2)]

    def _shape(self, context, event, final=False):
        if not self._pts:
            return None
        O, n, u, v = self._plane
        P, _ = self._point(context, event)
        typed = self._typed(context)
        A = self._pts[0]
        if len(self._pts) == 1:
            w = _flat((P if P is not None else A + u) - A, n)
            L = typed.get("L", float(np.linalg.norm(w)))
            B = A + _unit(w if np.linalg.norm(w) > 1e-12 else u) * L
            W = float(self.width)
            T = frame(0.5 * (A + B), B - A if L > 1e-12 else u, n)
            return {"kind": "slot", "poly": to_world(T, rounded_outline(0.5 * L + 0.5 * W, 0.5 * W, 0.5 * W)),
                    "next": B, "ok": False, "vals": {"L": L}}
        B = self._pts[1]
        L = float(np.linalg.norm(B - A))
        T = frame(0.5 * (A + B), B - A if L > 1e-12 else u, n)
        W = typed.get("W", 2.0 * abs(float((P - A) @ T[:3, 1])) if P is not None else float(self.width))
        if W < 1e-12:
            return None
        from .feature_ui import primitive_params
        return {"kind": "slot", "op": "prim_slot", "name": "NURBS Slot", "T": T,
                "params": primitive_params("prim_slot", length=L, width=W),
                "poly": to_world(T, rounded_outline(0.5 * L + 0.5 * W, 0.5 * W, 0.5 * W)), "vals": {"W": W}}


POLY_MODES = [("CORNER", "To a Corner", "The centre, then a corner"),
              ("SIDE", "To a Side", "The centre, then the middle of a side")]


class NPROTO_OT_draw_polygon(_TwoStep, bpy.types.Operator):
    """Draw a regular polygon: its centre, then a corner or the middle of a
    side (M switches); Up / Down change the number of sides. Type the radius.
    It stays editable in the Feature panel"""
    bl_idname = "nproto.draw_polygon"
    bl_label = "Polygon"
    bl_options = {'REGISTER', 'UNDO'}
    MODES, TITLE, STEPS, PARAMS = POLY_MODES, "Polygon", 2, ("radius", "sides")

    mode: EnumProperty(name="Mode", items=POLY_MODES, default="CORNER")
    radius: FloatProperty(name="Radius", default=0.5, min=1e-6, unit='LENGTH', description="To a corner")
    sides: bpy.props.IntProperty(name="Sides", default=6, min=3, max=64)
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        self.layout.use_property_split = True
        col = self.layout.column(align=True)
        col.prop(self, "radius")
        col.prop(self, "sides")

    def _needs_line(self):
        return False

    def _fields(self):
        return ["R"] if len(self._pts) == 1 else []

    def _step(self):
        return ["click the centre", "click a corner" if self.mode == "CORNER" else "click a side's middle"][
            min(len(self._pts), 1)]

    def _extra_key(self, context, event):
        if event.value == 'PRESS' and event.type in {'UP_ARROW', 'DOWN_ARROW'}:
            self.sides = max(3, min(64, int(self.sides) + (1 if event.type == 'UP_ARROW' else -1)))
            return True
        return False

    def _extra_rows(self):
        return [("Sides", [hints.key("↑ ↓", f"sides ({int(self.sides)})")])]

    def _shape(self, context, event, final=False):
        if not self._pts:
            return None
        O, n, u, v = self._plane
        P, _ = self._point(context, event)
        typed = self._typed(context)
        C = self._pts[0]
        w = _flat((P if P is not None else C + u) - C, n)
        d = typed.get("R", float(np.linalg.norm(w)))
        if d < 1e-12:
            return None
        k = max(3, int(self.sides))
        x = _unit(w if np.linalg.norm(w) > 1e-12 else u)
        if self.mode == "SIDE":                          # the mouse at a side's middle: the corners turn by half a side
            R = d / math.cos(math.pi / k)
            a = math.pi / k
            x = math.cos(a) * x + math.sin(a) * np.cross(_unit(n), x)
        else:
            R = d
        T = frame(C, x, n)
        from .feature_ui import primitive_params
        return {"kind": "polygon", "op": "prim_polygon", "name": "NURBS Polygon", "T": T,
                "params": primitive_params("prim_polygon", radius=R, sides=k),
                "poly": to_world(T, polygon_outline(R, k)), "vals": {"R": d}}


def axis_height(C, a, o, d):
    """Signed distance along the axis (C, unit a) of the axis point nearest the
    view ray (o, unit d); None when the ray runs (nearly) along the axis."""
    C, a, o, d = (np.asarray(x, float) for x in (C, a, o, d))
    w = C - o
    b = float(a @ d)
    den = 1.0 - b * b
    if den < 1e-4:
        return None
    return (b * float(d @ w) - float(a @ w)) / den


def helix_outline(r, h, turns, taper=0.0, left=False, k=240):
    lean = math.tan(math.radians(max(-80.0, min(80.0, taper))))
    hand = -1.0 if left else 1.0
    out = []
    for j in range(k + 1):
        z = h * j / k
        rr = max(0.0, r - z * lean)
        a = 2.0 * math.pi * turns * j / k
        out.append((rr * math.cos(a), hand * rr * math.sin(a), z))
    return out


HELIX_MODES = [("CENTER", "Centre", "The centre, the start (its radius), then the height"),
               ("AROUND", "Around", "Click a circle or circular edge: its centre, radius and axis; then the height")]
REAL_SNAPS = {"first", "end", "mid", "center", "cv", "vertex", "edge", "curve", "line", "face"}


class NPROTO_OT_draw_helix(_TwoStep, bpy.types.Operator):
    """Draw a helix in 3D: its centre, its start (the radius), then its height
    along the axis -- or around a circle or circular edge (M switches). Type the
    height, then (Tab) turns, pitch and taper; L makes it left-handed. It stays
    editable in the Feature panel"""
    bl_idname = "nproto.draw_helix"
    bl_label = "Helix"
    bl_options = {'REGISTER', 'UNDO'}
    MODES, TITLE, PARAMS = HELIX_MODES, "Helix", ("radius", "height", "turns", "taper", "left")
    FIELD_NAMES = dict(_ShapeTool.FIELD_NAMES, H="Height", T="Turns", P="Pitch", A="Taper")
    FIELD_HINTS = dict(_ShapeTool.FIELD_HINTS, H="the height", T="the turns", P="the pitch", A="the taper angle")

    mode: EnumProperty(name="Mode", items=HELIX_MODES, default="CENTER")
    radius: FloatProperty(name="Radius", default=0.5, min=1e-6, unit='LENGTH')
    height: FloatProperty(name="Height", default=1.0, min=1e-6, unit='LENGTH')
    turns: FloatProperty(name="Turns", default=3.0, min=0.1, soft_max=50.0)
    taper: FloatProperty(name="Taper °", default=0.0, min=-80.0, max=80.0, soft_min=-45.0, soft_max=45.0,
                         description="Lean in (positive: narrower on top) or out, like a thread on a cone")
    left: bpy.props.BoolProperty(name="Left-Handed", default=False, description="Turn the other way as it rises")
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    @property
    def STEPS(self):
        return 3 if self.mode == "CENTER" else 1

    def draw(self, context):
        self.layout.use_property_split = True
        col = self.layout.column(align=True)
        for k in self.PARAMS:
            col.prop(self, k)

    def _reset(self):
        _ShapeTool._reset(self)
        self._axis, self._circle = None, None

    def _needs_line(self):
        return self.mode == "AROUND" and not self._pts

    def _height_step(self):
        return len(self._pts) == 2

    def _fields(self):
        if self._height_step():
            return ["H", "T", "P", "A"]
        return ["R"] if self.mode == "CENTER" and len(self._pts) == 1 else []

    def _step(self):
        if self._height_step():
            return "the height: move along the axis (or type it)"
        if self.mode == "AROUND":
            return ["click a circle or circular edge", "click where it starts on the circle"][min(len(self._pts), 1)]
        return ["click the centre", "click the start (the radius)"][min(len(self._pts), 1)]

    def _extra_key(self, context, event):
        if event.value == 'PRESS' and event.type == 'L' and not self._buf:
            self.left = not self.left
            return True
        if event.value == 'PRESS' and event.type == 'BACK_SPACE' and self.mode == "AROUND" and not self._buf \
                and not self._vals and len(self._pts) == 1:
            self._pts, self._axis, self._circle = [], None, None      # back to picking a circle
            return True
        return False

    def _extra_rows(self):
        return [("Thread", [hints.toggle("L", "left-handed", self.left)])]

    def _line_under(self, context, event):
        """(point, None, polyline, (centre, radius, normal)) of the circle / circular edge under the mouse."""
        view, mouse = self._view(context), self._mouse(context, event)
        for k, _ in self._snaps.near_lines(view, mouse, SNAP_PX, limit=4):
            W, src = self._snaps.line_info(k)
            c = self._curve_of(tuple(src[:2])) if src is not None else None
            if c is None or c.straight:
                continue
            S = c.W
            c3 = circle_3(S[0], S[len(S) // 3], S[(2 * len(S)) // 3])
            if c3 is None:
                continue
            C, r, nn = c3
            if float(np.abs(np.linalg.norm(S - C, axis=1) - r).max()) > 1e-6 * r \
                    or float(np.abs((S - C) @ nn).max()) > 1e-6 * r:
                continue                                 # not a circle (or an arc of one)
            P, _, _ = self._snaps.slide(view, mouse, k)
            return (np.asarray(P, float), None, W, (C, r, nn))
        return None

    def _coaxial_rim(self, P, snap, C, a):
        """(height, radius) a cone gives the helix through the snapped point P:
        a circle on the helix's axis (a rim: its level and radius), a straight
        edge in a plane with the axis (a cone's seam line: P's height and its
        distance from the axis), or P on the axis itself (a cone's tip: radius
        0). None for anything else."""
        P = np.asarray(P, float)
        h_P, rho_P = float((P - C) @ a), float(np.linalg.norm(_flat(P - C, a)))
        srcs = [snap[2]] if snap is not None and snap[2] is not None else []
        try:
            srcs += [s3 for s3, _, _ in self._snaps.through(P)]
        except Exception:
            pass
        r0 = max(float(np.linalg.norm(_flat(self._pts[1] - C, a))), 1e-9) if len(self._pts) > 1 else 1.0
        if rho_P <= 1e-6 * r0:
            return h_P, 0.0                              # on the axis: a cone's tip
        for src in srcs:
            c = self._curve_of(tuple(src[:2]))
            if c is None:
                continue
            S = c.W
            if c.straight:                               # a seam line: in a plane with the axis
                d = _unit(c.direction)
                pn = np.cross(a, d)
                if float(np.linalg.norm(pn)) < 1e-9:
                    return h_P, rho_P                    # parallel to the axis: a cylinder's seam
                if abs(float((np.asarray(c.center, float) - C) @ _unit(pn))) <= 1e-6 * max(r0, c.size):
                    return h_P, rho_P
                continue
            c3 = circle_3(S[0], S[len(S) // 3], S[(2 * len(S)) // 3])
            if c3 is None:
                continue
            Cc, rc, nc = c3
            if float(np.abs(np.linalg.norm(S - Cc, axis=1) - rc).max()) > 1e-6 * rc:
                continue                                 # not a circle
            off = _flat(Cc - C, a)
            if abs(float(nc @ a)) < 1 - 1e-6 or float(np.linalg.norm(off)) > 1e-6 * max(rc, 1e-9):
                continue                                 # not on the helix's axis
            return float((Cc - C) @ a), rc
        return None

    def _click(self, context, event):
        if self.mode == "AROUND" and not self._pts:
            ln = self._line_under(context, event)
            if ln is None:
                self.report({'WARNING'}, "Helix: click a circle or a circular edge")
                return None
            C, r, nn = ln[3]
            self.fix_plane(context, C)
            if float(nn @ self._plane[1]) < 0:
                nn = -nn                                 # the axis faces the viewer (up in a top view)
            self._pts = [np.asarray(C, float)]
            self._axis, self._circle = _unit(nn), (np.asarray(C, float), float(r), _unit(nn), ln[2])
            self._reset_typing()
            return None
        if self.mode == "CENTER" and not self._pts:
            P, _ = self._point(context, event)
            if P is None:
                return None
            self._add_point(context, P)
            self._axis = _unit(self._plane[1])
            return None
        sh = self._shape(context, event, final=True)
        if not self._height_step():
            if sh is None or sh.get("next") is None:
                return None
            self._pts.append(np.asarray(sh["next"], float))
            self._reset_typing()
            return None
        return self._finish(context, sh)

    def _enter(self, context, event):
        if not self._height_step():
            return self._click(context, event)
        return _ShapeTool._enter(self, context, event)

    def _shape(self, context, event, final=False):
        if not self._pts:
            return None
        O, n, u, v = self._plane
        typed = self._typed(context)
        P, snap = self._point(context, event)
        C = self._pts[0]
        if self.mode == "AROUND" and not self._height_step() and self._circle is not None:
            Cc, rc, nc, W = self._circle                 # the start: on that circle, where the mouse points
            w = _flat((P if P is not None else Cc + u) - Cc, nc)
            S = Cc + rc * _unit(w if np.linalg.norm(w) > 1e-12 else np.cross(nc, [1.0, 0.0, 0.0]))
            return {"kind": "helix", "poly": [np.asarray(q, float) for q in W] + [S, Cc], "next": S, "ok": False,
                    "vals": {"R": rc}}
        if not self._height_step():                      # the start: radius and direction on the plane
            w = _flat((P if P is not None else C + u) - C, n)
            r = typed.get("R", float(np.linalg.norm(w)))
            if r < 1e-12:
                return None
            S = C + _unit(w if np.linalg.norm(w) > 1e-12 else u) * r
            return {"kind": "helix", "poly": circle_points(C, r, n, S - C), "next": S, "ok": False, "vals": {"R": r}}
        S, a = self._pts[1], self._axis
        r = float(np.linalg.norm(_flat(S - C, a)))
        rim = None
        if "H" in typed:                                 # a negative height goes the other way
            h = typed["H"] if typed["H"] < 0 else math.copysign(typed["H"], getattr(self, "_h_sign", 1.0))
        else:
            h = None
            if snap is not None and P is not None and (snap[0] in REAL_SNAPS):
                rim = self._coaxial_rim(np.asarray(P, float), snap, C, a)
                # level with what it snapped to; on the other rim of a cone: exactly that rim's level
                h = rim[0] if rim is not None else float((np.asarray(P, float) - C) @ a)
            if h is None:
                region, rv3d = self._win(context)
                m = Vector(self._mouse(context, event))
                o = np.array(view3d_utils.region_2d_to_origin_3d(region, rv3d, m), float)
                d = np.array(view3d_utils.region_2d_to_vector_3d(region, rv3d, m), float)
                h = axis_height(C, a, o, _unit(d))
                if h is None:                                    # looking down the axis: the distance on the plane
                    Pm = self._ray_point(context, m, (C, a))
                    h = float(np.linalg.norm(_flat(Pm - C, a))) if Pm is not None else 0.0
                if event.ctrl:
                    st = self._grid_step(context, C)
                    h = round(h / st) * st
            self._h_sign = 1.0 if h >= 0 else -1.0
        H = abs(h)
        if H < 1e-12:
            return None
        turns = typed.get("T", float(self.turns))
        if "P" in typed and typed["P"] > 1e-12:
            turns = H / typed["P"]
        turns = max(0.1, turns)
        taper = typed.get("A", float(self.taper))
        note = None
        if rim is not None and "A" not in typed:         # the cone between the two rims: its taper
            taper = math.degrees(math.atan2(r - rim[1], H))
            note = f"taper {taper:.2f}° from the cone"
        T = frame(C, S - C, a if h >= 0 else -a)
        from .feature_ui import primitive_params
        params = primitive_params("prim_helix", radius=r, height=H, turns=turns, taper=taper, left=bool(self.left),
                                  degree=3, spans_u=8)
        return {"kind": "helix", "op": "prim_helix", "name": "NURBS Helix", "T": T, "params": params,
                "poly": to_world(T, helix_outline(r, H, turns, taper, self.left)),
                "vals": {"H": H, "T": turns, "P": H / turns, "A": taper}, "note": note}


classes = (NPROTO_OT_draw_circle, NPROTO_OT_draw_arc, NPROTO_OT_draw_ellipse, NPROTO_OT_draw_rectangle,
           NPROTO_OT_draw_slot, NPROTO_OT_draw_polygon, NPROTO_OT_draw_helix)
