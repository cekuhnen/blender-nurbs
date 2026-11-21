"""Slide handles of a two-sided Loft: a dot where the loft meets each end of
its two sides (From / To of side A and B). Press a dot and drag: it slides
along the side's whole edge, the new stretch drawn as you go; release writes
it into the loft's history (one undo step, its value in the redo panel).
Shown with the continuity labels (Display > Continuity) on selected lofts."""
import json

import bpy
import numpy as np
from bpy.props import FloatProperty, StringProperty
from bpy_extras import view3d_utils
from mathutils import Vector

from . import draw, features, hints

KEYS = features.SLIDE_KEYS          # a_from, a_to, b_from, b_to
DOT_PX = 9.0                        # a dot can be pressed this close to it
hover = {"key": None}               # (object name, key) of the dot under the mouse
DRAG = {"key": None}                # (object name, key) being dragged: its old dot is not drawn

_cache = {}


def _eligible(obj):
    h = features.history(obj)
    if h is None or h.get("op") != "loft":
        return None
    try:
        if not features.can_slide(h["inputs"], h["params"]):
            return None
    except Exception:
        return None
    return h


def tracks(obj):
    """[(points, runs_back)] of the loft's two sides (features.slide_tracks), cached per build."""
    h = _eligible(obj)
    if h is None:
        return None
    key = (obj.session_uid, obj.get(features.PROP_HIST), obj.get("nproto_build_key"))
    hit = _cache.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    try:
        tr = features.slide_tracks(h["inputs"], h["params"])
    except Exception as exc:
        print("NURBS loft slide handles:", obj.name, exc)
        tr = None
    _cache[obj.session_uid] = (key, tr)
    return tr


def value_of(obj, key):
    h = features.history(obj)
    return float(h["params"].get(key, 0.0 if key.endswith("from") else 1.0)) if h else 0.0


def at(track, value):
    """(point, unit tangent) on a side track at share `value` of its From / To."""
    P, back = track
    f = 1.0 - value if back else value
    x = min(max(f, 0.0), 1.0) * (len(P) - 1)
    i = min(int(x), len(P) - 2)
    t = x - i
    d = P[i + 1] - P[i]
    return P[i] + t * d, d / (np.linalg.norm(d) or 1.0)


def stretch(track, lo, hi):
    """The points of a side track between shares lo and hi of its From / To."""
    P, back = track
    a, b = sorted((1.0 - lo, 1.0 - hi) if back else (lo, hi))
    n = len(P) - 1
    i0, i1 = int(np.ceil(a * n)), int(np.floor(b * n))
    mid = [P[i] for i in range(i0, i1 + 1)]
    return np.array([at((P, False), a)[0]] + mid + [at((P, False), b)[0]])


def dots(obj):
    """[{pos, key}] of the loft's slide dots (none for a loft that cannot slide)."""
    tr = tracks(obj)
    if not tr:
        return []
    out = []
    for k in KEYS:
        out.append({"pos": Vector(at(tr[0 if k[0] == "a" else 1], value_of(obj, k))[0]), "key": k})
    return out


def _to_px(context, p):
    return view3d_utils.location_3d_to_region_2d(context.region, context.region_data, Vector(p))


# --------------------------------------------------------------------------
# The drag
# --------------------------------------------------------------------------

class NPROTO_OT_loft_slide(bpy.types.Operator):
    """Slide this end of the loft along its edge (drag; Ctrl: steps of 0.05, Shift+Ctrl 0.01).
    The loft spans only From..To of the side and keeps its continuity there"""
    bl_idname = "nproto.loft_slide"
    bl_label = "Slide Loft End"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    obj_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    key: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    value: FloatProperty(name="Position", min=0.0, max=1.0, subtype='FACTOR',
                         description="Where along the side this end of the loft sits (share of its length)")

    @classmethod
    def description(cls, context, props):
        obj = bpy.data.objects.get(props.obj_name)
        if obj is None or props.key not in KEYS:
            return cls.__doc__
        return (f"{_label(props.key)} {value_of(obj, props.key):.3f}: drag along the edge to slide this end "
                "of the loft (Ctrl: steps of 0.05, Shift+Ctrl: 0.01; Esc: back)")

    def draw(self, context):
        self.layout.prop(self, "value", text=_label(self.key), slider=True)

    # -- values ---------------------------------------------------------------
    def _limits(self, obj):
        """The range this end may take: it stays on its side of the other end."""
        other = self.key[0] + ("_to" if self.key.endswith("from") else "_from")
        o = value_of(obj, other)
        gap = features.SLIDE_MIN
        return (0.0, max(o - gap, 0.0)) if self.key.endswith("from") else (min(o + gap, 1.0), 1.0)

    def execute(self, context):
        obj = bpy.data.objects.get(self.obj_name)
        if obj is None or self.key not in KEYS or _eligible(obj) is None:
            return {'CANCELLED'}
        lo, hi = self._limits(obj)
        v = min(max(float(self.value), lo), hi)
        old = obj[features.PROP_HIST]
        raw = json.loads(old)
        for k in KEYS:                                 # all four, so the loft has them as a set
            raw["params"].setdefault(k, 0.0 if k.endswith("from") else 1.0)
        raw["params"][self.key] = v
        obj[features.PROP_HIST] = json.dumps(raw)
        try:
            features.edit_rebuild(obj, context, force=True)
        except Exception as exc:
            obj[features.PROP_HIST] = old
            self.report({'ERROR'}, f"Slide: {exc}")
            return {'CANCELLED'}
        try:
            from . import feature_ui
            feature_ui.sync_from_history(obj)         # the Feature box's From / To follow
        except Exception as exc:
            print("NURBS loft slide panel sync:", exc)
        self.report({'INFO'}, f"{obj.name} · {_label(self.key)} {v:.3f}"
                    + ("  (paused: Rebuild builds it)" if features.paused(context) else ""))
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return {'FINISHED'}

    # -- interactive ------------------------------------------------------------
    def invoke(self, context, event):
        obj = bpy.data.objects.get(self.obj_name)
        tr = tracks(obj) if obj is not None else None
        if not tr or self.key not in KEYS or context.region_data is None:
            return {'CANCELLED'}
        self._obj = obj
        self._track = tr[0 if self.key[0] == "a" else 1]
        self._other = tr[1 if self.key[0] == "a" else 0]
        self._start = value_of(obj, self.key)
        self.value = self._start
        self._moved = False
        self._press = (event.mouse_x, event.mouse_y)
        self._grab = (event.value == 'PRESS')         # a press: drag until release; a click: until the next click
        DRAG["key"] = (obj.name, self.key)
        context.window_manager.modal_handler_add(self)
        self._refresh(context)
        return {'RUNNING_MODAL'}

    def _value_at(self, context, event):
        """The share of the side under the mouse: the nearest point of its
        whole edge on screen."""
        P, back = self._track
        region, rv3d = context.region, context.region_data
        m = Vector((event.mouse_region_x, event.mouse_region_y))
        best = None
        prev = None
        for i, p in enumerate(P):
            q = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(p))
            if q is not None and prev is not None:
                seg = q - prev
                L = seg.length_squared
                t = 0.0 if L == 0 else max(0.0, min(1.0, (m - prev).dot(seg) / L))
                d = (prev + seg * t - m).length
                if best is None or d < best[0]:
                    best = (d, (i - 1 + t) / (len(P) - 1))
            prev = q
        if best is None:
            return None
        f = best[1]
        v = 1.0 - f if back else f
        if event.ctrl:
            step = 0.01 if event.shift else 0.05
            v = round(v / step) * step
        lo, hi = self._limits(self._obj)
        return min(max(v, lo), hi)

    def modal(self, context, event):
        et = event.type
        if et == 'INBETWEEN_MOUSEMOVE':
            return {'PASS_THROUGH'}
        if et in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'}:
            return {'PASS_THROUGH'}
        if et in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._end(context)
            return {'CANCELLED'}
        if et == 'MOUSEMOVE' or et in {'LEFT_CTRL', 'RIGHT_CTRL', 'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            if et == 'MOUSEMOVE' and abs(event.mouse_x - self._press[0]) + abs(event.mouse_y - self._press[1]) > 2:
                self._moved = True
            if self._moved:
                v = self._value_at(context, event)
                if v is not None:
                    self.value = v
            self._refresh(context)
            return {'RUNNING_MODAL'}
        if et == 'LEFTMOUSE':
            done = (event.value == 'RELEASE' and self._grab) or (event.value == 'PRESS' and not self._grab)
            if not done:
                return {'RUNNING_MODAL'}
            self._end(context)
            if not self._moved or abs(self.value - self._start) < 1e-9:
                return {'CANCELLED'}                   # a click on a dot without a drag changes nothing
            return self.execute(context)
        if et in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            self._end(context)
            return self.execute(context) if abs(self.value - self._start) > 1e-9 else {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _refresh(self, context):
        lo_k, hi_k = self.key[0] + "_from", self.key[0] + "_to"
        lo = self.value if self.key == lo_k else value_of(self._obj, lo_k)
        hi = self.value if self.key == hi_k else value_of(self._obj, hi_k)
        o = "b" if self.key[0] == "a" else "a"
        olo, ohi = value_of(self._obj, o + "_from"), value_of(self._obj, o + "_to")
        mine = stretch(self._track, lo, hi)
        theirs = stretch(self._other, olo, ohi)
        pos, tan = at(self._track, self.value)
        # the loft's new outline: the two stretches and straight lines between their ends
        a_s, b_s = (mine, theirs) if self.key[0] == "a" else (theirs, mine)
        b0, b1 = b_s[0], b_s[-1]
        if np.linalg.norm(a_s[0] - b0) + np.linalg.norm(a_s[-1] - b1) > \
                np.linalg.norm(a_s[0] - b1) + np.linalg.norm(a_s[-1] - b0):
            b0, b1 = b1, b0
        pv = {"lines": [mine, theirs, np.array([a_s[0], b0]), np.array([a_s[-1], b1])], "tris": None,
              "guides": [(self._track[0], (1.0, 1.0, 1.0, 0.25))],
              "handles": [{"kind": "slide", "pos": pos, "dir": tan, "hot": True, "active": True}],
              "labels": [(pos, f"{_label(self.key)} {self.value:.3f}", True)]}
        draw.pick_overlay["preview"] = pv
        hints.show(context, self.bl_label, [f"{_label(self.key)} {self.value:.3f}"],
                   [("Slide", [hints.key("Ctrl", "steps of 0.05"), hints.key("Shift+Ctrl", "steps of 0.01")])],
                   [("MOUSE_LMB_DRAG", "Slide along the edge"), ("EVENT_ESC", "Cancel")])
        _redraw(context)

    def _end(self, context):
        DRAG["key"] = None
        draw.pick_overlay["preview"] = None
        hints.clear(context)
        _redraw(context)


def _redraw(context):
    """Every 3D view of the window (a gizmo's operator may not run in the view's own area)."""
    win = context.window or (context.window_manager.windows[0] if context.window_manager.windows else None)
    for a in (win.screen.areas if win else []):
        if a.type == 'VIEW_3D':
            a.tag_redraw()


def _label(key):
    return f"{'From' if key.endswith('from') else 'To'} {key[0].upper()}"


# --------------------------------------------------------------------------
# The dots as gizmos (hit-tested in 2D, drawn by draw._draw_px)
# --------------------------------------------------------------------------

class NPROTO_GT_loft_slide(bpy.types.Gizmo):
    bl_idname = "NPROTO_GT_loft_slide"

    __slots__ = ("obj_name", "key")

    def setup(self):
        self.obj_name, self.key = "", ""

    def draw(self, context):
        pass

    def draw_select(self, context, select_id):
        pass

    def test_select(self, context, location):
        obj = bpy.data.objects.get(self.obj_name)
        hit = False
        if obj is not None and DRAG["key"] is None:
            for d in dots(obj):
                if d["key"] == self.key:
                    q = _to_px(context, d["pos"])
                    hit = q is not None and (Vector(location) - q).length <= DOT_PX
        key = (self.obj_name, self.key)
        if hit:
            if hover["key"] != key:
                hover["key"] = key
                context.area.tag_redraw()
            return 0
        if hover["key"] == key:
            hover["key"] = None
            context.area.tag_redraw()
        return -1


def visible_objects(context):
    from . import contlabels
    return [o for o in contlabels.visible_label_objects(context) if tracks(o)]


class NPROTO_GGT_loft_slides(bpy.types.GizmoGroup):
    bl_idname = "NPROTO_GGT_loft_slides"
    bl_label = "NURBS loft slide handles"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'PERSISTENT', 'SHOW_MODAL_ALL'}

    @classmethod
    def poll(cls, context):
        return bool(visible_objects(context))

    def _sync(self, context):
        want = [(o.name, k) for o in visible_objects(context) for k in KEYS]
        have = [(g.obj_name, g.key) for g in self.gizmos]
        if want == have:
            return
        self.gizmos.clear()
        for name, k in want:
            g = self.gizmos.new(NPROTO_GT_loft_slide.bl_idname)
            props = g.target_set_operator("nproto.loft_slide")
            g.obj_name, g.key = name, k
            props.obj_name, props.key = name, k
            g.use_draw_modal = False

    def setup(self, context):
        self._sync(context)

    def refresh(self, context):
        self._sync(context)

    def draw_prepare(self, context):
        self._sync(context)


def draw_dots(context, region, rv3d, disc):
    """The slide dots of the visible lofts: white rings, yellow under the mouse."""
    for obj in visible_objects(context):
        for d in dots(obj):
            if DRAG["key"] == (obj.name, d["key"]):
                continue
            q = view3d_utils.location_3d_to_region_2d(region, rv3d, d["pos"])
            if q is None:
                continue
            hot = hover["key"] == (obj.name, d["key"])
            disc(q, 7.5 if hot else 6.0, (0.05, 0.05, 0.05, 0.9))
            disc(q, 5.5 if hot else 4.2, (1.0, 0.85, 0.2, 1.0) if hot else (1.0, 1.0, 1.0, 1.0))


classes = (NPROTO_OT_loft_slide, NPROTO_GT_loft_slide, NPROTO_GGT_loft_slides)
