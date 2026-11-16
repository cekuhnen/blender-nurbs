"""Live value stage with handles: once a command knows its inputs, it shows
its result as a preview with handles on it (Extrude: an arrow standing on the
curve for the distance, a diamond on the top edge for the draft), instead of
asking for the values in a dialog before anything can be seen.

    drag a handle       its value follows; nothing changes while the mouse
                        only moves (a handle under it lights up)
    press on the input  and pull (when the tool was started by clicking the
                        curve): the arrow comes out of the curve under the
                        cursor, in one motion
    type                an exact value for the active field (unit letters
                        work: 25mm, 2ft); dragging a handle makes its field
                        the active one
    Ctrl                snap to the grid step you can see (1 degree for angles)
    Shift               precise (a tenth of the motion)
    Tab                 the next field (Extrude: Distance, then Draft)
    LMB away from the handles, Enter, Space
                        apply -- the Adjust Last Operation panel still works
    Esc / RMB           cancel

The value starts at the last one used, unless that would be tiny or run off
the screen at this zoom: then a round length a quarter of the view is used.
So pick + Enter is as quick as before.
The add-on preference Value Input = Dialog turns the stage off: every tool
then asks in its dialog, as before.
A tool mixes ValueDrag in before AskValue (whose dialog stays the fallback for
when the preview can't be set up) and implements the drag_* methods. Scripts
(EXEC) and background runs never enter the stage.
"""

import math

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

from . import draw, hints

# the view still turns, pans and zooms (the number pad types numbers, as in Blender's own Move)
NAV = {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE',
       'NDOF_MOTION', 'ACCENT_GRAVE'}
NUMPAD_CHARS = {f"NUMPAD_{i}": str(i) for i in range(10)} | {
    "NUMPAD_PERIOD": ".", "NUMPAD_MINUS": "-", "NUMPAD_PLUS": "+", "NUMPAD_SLASH": "/", "NUMPAD_ASTERIX": "*"}
NUMBER_START = set("0123456789.-+")
PULL_PX = 6.0           # a press on the input becomes a pull once the mouse moved this far
HANDLE_PX = 16.0        # a handle can be grabbed this close to it
SHAFT_PX = 7.0          # ... or this close to the arrow's shaft
BASE_PX = 40.0          # the arrow stands where the curve was clicked, if it was within this
MOUSE = [("MOUSE_LMB_DRAG", "Drag a handle"), ("MOUSE_LMB", "Apply"), ("EVENT_ESC", "Cancel")]


def axis_param(ray_o, ray_d, B, d):
    """Parameter t of the point B + t d on the line closest to the mouse ray,
    or None when the line runs (almost) along the ray: seen end-on it has no
    direction on screen to drag along."""
    ray_o, ray_d = np.asarray(ray_o, float), np.asarray(ray_d, float)
    B, d = np.asarray(B, float), np.asarray(d, float)
    ray_d = ray_d / (np.linalg.norm(ray_d) or 1.0)
    d = d / (np.linalg.norm(d) or 1.0)
    b = float(d @ ray_d)
    den = 1.0 - b * b
    if den < 1e-4:          # within ~0.6 degrees of the view direction
        return None
    w = B - ray_o
    return float((b * (ray_d @ w) - d @ w) / den)


def snap(value, step):
    return round(value / step) * step if step > 0 else value


def fmt_length(context, x):
    us = context.scene.unit_settings
    if us.system == 'NONE':
        return f"{x:.4g}"
    try:
        return bpy.utils.units.to_string(us.system, 'LENGTH', x * (us.scale_length or 1.0), precision=4)
    except Exception:
        return f"{x:.4g}"


def ruled_preview(rows, ribs=8):
    """Lines and see-through fill of a surface given by rows of points (the
    preview of a ruled / extruded surface): every row, a few ribs across them
    (the ends always), and triangles between neighbouring rows."""
    rows = [np.asarray(r, float) for r in rows]
    lines = [r for r in rows]
    n = len(rows[0])
    for k in sorted({0, n - 1} | {int(round(i * (n - 1) / ribs)) for i in range(ribs + 1)}):
        lines.append(np.array([r[k] for r in rows]))
    tris = []
    for a, b in zip(rows[:-1], rows[1:]):
        q = np.stack([a[:-1], a[1:], b[1:], a[:-1], b[1:], b[:-1]], axis=1)    # two triangles a quad
        tris.append(q.reshape(-1, 3))
    return lines, (np.vstack(tris) if tris else np.zeros((0, 3)))


def _seg_dist(p, a, b):
    ab = b - a
    L = ab.length_squared
    t = 0.0 if L == 0 else max(0.0, min(1.0, (p - a).dot(ab) / L))
    return (a + ab * t - p).length


class ValueDrag:
    """Mixin for AskValue commands: the live value stage (see the module doc).

    A tool sets drag_fields (the properties Tab steps through, the first
    active; angle_fields are among them) and implements:
      drag_setup(context, mouse) -> bool
            gather what the preview needs; mouse: where the command was started
            (region pixels, or None outside the view), to stand the arrow
            there. False: the dialog instead
      drag_handles() -> [handle]
            each {"field", "kind" ("arrow" | "draft"), "origin", "dir", "param",
            "to_value"(t) -> value, "base" (arrow: the shaft's foot)}: the
            handle sits at origin + param * dir and is dragged along that line
      drag_preview() -> {"lines": [polylines], "tris": (n,3)}
      drag_check() -> str           why it can't be applied yet ('' when it can)
      drag_key(context, event, ch)  a key of the tool's own: True when handled
      drag_rows() / drag_state()    its hint-panel rows / header words
    The first field's first handle is the one a pull drags.
    """

    drag_fields = ("distance",)
    angle_fields = ()

    def invoke(self, context, event):
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        return super().invoke(context, event)

    def ask(self, context):
        if bpy.app.background or hints.value_input() == 'DIALOG':
            return super().ask(context)      # the dialog (the preference, or no view to drag in)
        self.prepare_value(context)
        region, rv3d = self._win(context)
        se = getattr(self, "_start_event", None)
        mouse = None
        if se is not None and region is not None:
            from .ops import PickMixin
            probe = _Probe(se[0], se[1])
            if PickMixin._in_view(context, probe):
                mouse = Vector((se[0] - region.x, se[1] - region.y))
        try:
            ready = self.drag_setup(context, mouse)
        except Exception as exc:
            print("NURBS value preview:", exc)
            ready = False
        if not ready:
            return super().ask(context)
        self._dragging = True
        self._field = 0
        self._typed = ""
        self._grab = None           # the handle being dragged: {"h", "t0", "p0"}
        self._hot = None            # the handle under the mouse (its index)
        self._msg = ""
        self._edited = False        # a value was dragged or typed
        # started by a press on the input (picking it): pulling out of it extrudes
        self._pull = (se is not None and se[2] == 'LEFTMOUSE' and se[3] == 'PRESS'
                      and mouse is not None and getattr(self, "_base_at_mouse", False))
        self._pull_from = Vector((se[0], se[1])) if self._pull else None
        try:
            self._fit_start(context)
        except Exception as exc:
            print("NURBS value preview:", exc)
        context.window_manager.modal_handler_add(self)
        self._drag_refresh(context)
        return {'RUNNING_MODAL'}

    # -- the view ------------------------------------------------------------
    @staticmethod
    def _win(context):
        """The 3D view's main region and its view, whichever region the tool
        was started from (a sidebar button runs the modal in the sidebar)."""
        area = context.area
        region = next((r for r in area.regions if r.type == 'WINDOW'), context.region) if area else context.region
        rv3d = getattr(area.spaces.active, "region_3d", None) if area else None
        return region, rv3d or context.region_data

    def _mouse(self, context, event):
        region, _ = self._win(context)
        return Vector((event.mouse_x - region.x, event.mouse_y - region.y))

    def _screen(self, context, p):
        region, rv3d = self._win(context)
        return view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(p))

    def _ray_param(self, context, m, origin, direction):
        region, rv3d = self._win(context)
        o = view3d_utils.region_2d_to_origin_3d(region, rv3d, m)
        r = view3d_utils.region_2d_to_vector_3d(region, rv3d, m)
        return axis_param(o, r, origin, direction)

    def _ppu_along(self, context, B, d):
        """(pixels per unit along d at B, along the screen's up there)."""
        from .drawtools import pixels_per_unit
        region, rv3d = self._win(context)
        up = np.array(rv3d.view_matrix.inverted().to_3x3() @ Vector((0.0, 1.0, 0.0)))
        return pixels_per_unit(region, rv3d, B, [d]), pixels_per_unit(region, rv3d, B, [up])

    # -- values --------------------------------------------------------------
    def _field_name(self):
        return self.drag_fields[self._field % len(self.drag_fields)]

    def _is_angle(self, name=None):
        return (name or self._field_name()) in self.angle_fields

    def _label(self, name):
        try:
            return self.bl_rna.properties[name].name.replace(" °", "")
        except Exception:
            return name.title()

    def _fmt(self, context, name):
        v = getattr(self, name)
        return f"{v:.4g}°" if self._is_angle(name) else fmt_length(context, v)

    def _parse(self, context, text):
        text = text.strip().replace("°", "")
        if text in ("", "-", "+", "."):
            return None
        try:
            if self._is_angle():
                return float(eval(text, {"__builtins__": {}}, {}))      # 3*2, 45/2
            from .drawtools import parse_length
            return parse_length(context, text)
        except Exception:
            return None

    def _grid_step(self, context, at, along):
        from .drawtools import grid_steps, pixels_per_unit, visible_grid_step
        region, rv3d = self._win(context)
        ppu = pixels_per_unit(region, rv3d, at, [along])
        if ppu is None:
            return 0.0
        us = context.scene.unit_settings
        ov = getattr(context.area.spaces.active, "overlay", None) if context.area else None
        steps = grid_steps(us.system, us.scale_length or 1.0, getattr(ov, "grid_scale", 1.0) or 1.0,
                           getattr(ov, "grid_subdivisions", 10) or 10)
        return visible_grid_step(steps, ppu)

    def _main_index(self):
        """Index of the first field's handle (the one a pull drags), or None."""
        return next((i for i, h in enumerate(self.drag_handles()) if h["field"] == self.drag_fields[0]), None)

    def _main_handle(self):
        i = self._main_index()
        return None if i is None else self.drag_handles()[i]

    def _fit_start(self, context):
        """A starting length that fits this view: the last value used unless
        its arrow would be shorter than 40 px or longer than about half the
        view; then a round length about a quarter of the view, same sign."""
        if self._is_angle(self.drag_fields[0]):
            return
        region, rv3d = self._win(context)
        h = self._main_handle()
        if rv3d is None or region is None or h is None:
            return
        B, d = np.asarray(h["origin"], float), np.asarray(h["dir"], float)
        d = d / (np.linalg.norm(d) or 1.0)
        ppu, ppu_up = self._ppu_along(context, B, d)
        if not ppu or not ppu_up or ppu < 0.2 * ppu_up:
            return                      # seen (almost) end-on: its length can't be judged on screen
        name = self.drag_fields[0]
        v = getattr(self, name)
        view = min(region.width, region.height)
        if 40.0 <= abs(v) * ppu <= 0.45 * view:
            return
        from .ops_trim import nice_length
        setattr(self, name, (-1.0 if v < 0 else 1.0) * nice_length(0.25 * view / ppu))

    # -- handles ----------------------------------------------------------------
    @staticmethod
    def _pos(h):
        return np.asarray(h["origin"], float) + h["param"] * np.asarray(h["dir"], float)

    def _handle_at(self, context, m):
        """Index of the handle under region point m, or None."""
        best = None
        for i, h in enumerate(self.drag_handles()):
            q = self._screen(context, self._pos(h))
            if q is None:
                continue
            dist = (q - m).length
            if dist > HANDLE_PX and h["kind"] == "arrow" and h.get("base") is not None:
                b = self._screen(context, h["base"])
                if b is not None and _seg_dist(m, b, q) <= SHAFT_PX:
                    dist = HANDLE_PX        # on the shaft: grabbable, after a handle proper
            if dist <= HANDLE_PX and (best is None or dist < best[0]):
                best = (dist, i)
        return None if best is None else best[1]

    def _start_grab(self, context, i, m, absolute=False):
        """Grab handle i at region point m. absolute: the handle jumps to the
        cursor (a pull out of the curve); else it keeps its offset to it."""
        h = self.drag_handles()[i]
        t = self._ray_param(context, m, h["origin"], h["dir"])
        if t is None:
            return False
        self._grab = {"i": i, "t0": t, "p0": 0.0 if absolute else h["param"], "fine": False,
                      "abs": absolute}
        if absolute:
            self._grab["t0"] = 0.0
        self._field = self.drag_fields.index(h["field"])
        self._typed = ""
        return True

    def _drag_to(self, context, event):
        g = self._grab
        h = self.drag_handles()[g["i"]]
        m = self._mouse(context, event)
        t = self._ray_param(context, m, h["origin"], h["dir"])
        if t is None:
            return
        fine = bool(event.shift)
        if fine != g["fine"]:           # Shift went down / up: carry on from here
            g["p0"], g["t0"], g["fine"] = h["param"], t, fine
        p = g["p0"] + (t - g["t0"]) * (0.1 if fine else 1.0)
        name = h["field"]
        v = h["to_value"](p)
        if v is None:
            return
        if event.ctrl:
            if self._is_angle(name):
                v = snap(v, 0.1 if fine else 1.0)
            else:
                d = np.asarray(h["dir"], float)
                v = snap(v, self._grid_step(context, np.asarray(h["origin"], float), d) * (0.1 if fine else 1.0))
        setattr(self, name, v)
        self._edited = True

    def _apply_typed(self, context):
        v = self._parse(context, self._typed)
        if v is not None:
            setattr(self, self._field_name(), v)
            self._edited = True

    def _too_short(self, context):
        """A length dragged to (about) nothing -- the arrow back on the input:
        under 3 px on screen. A typed value is taken as it is."""
        name = self.drag_fields[0]
        if self._typed or self._is_angle(name):
            return ""
        h = self._main_handle()
        if h is None:
            return ""
        d = np.asarray(h["dir"], float)
        ppu = self._ppu_along(context, np.asarray(h["origin"], float), d / (np.linalg.norm(d) or 1.0))[0]
        if ppu and abs(getattr(self, name)) * ppu < 3.0:
            return f"{self._label(name)} 0: drag the arrow out, or type a value"
        return ""

    # -- the modal stage -----------------------------------------------------
    def modal(self, context, event):
        if not getattr(self, "_dragging", False):
            return super().modal(context, event)        # still picking the inputs
        region, _ = self._win(context)
        if region is not None and region != context.region:
            with context.temp_override(region=region):
                return self._drag_modal(bpy.context, event)
        return self._drag_modal(context, event)

    def _apply(self, context):
        if self._typed:
            self._apply_typed(context)
        msg = self._too_short(context) or self.drag_check()
        if msg:                         # e.g. a zero distance: say so and keep going
            self._msg = msg
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        self._drag_end(context)
        return self.execute(context)

    def _drag_modal(self, context, event):
        from .ops import PickMixin
        et = event.type
        if et == 'INBETWEEN_MOUSEMOVE':
            return {'PASS_THROUGH'}       # a gizmo drag needs them unhandled
        if et in NAV:
            return {'PASS_THROUGH'}
        if hints.handle(event):
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        dragging = self._grab is not None
        if (et in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not dragging and not self._pull
                and not PickMixin._in_view(context, event)):
            return {'PASS_THROUGH'}       # the sidebar, a header, the navigation gizmo
        if et == 'MOUSEMOVE':
            m = self._mouse(context, event)
            if self._pull and self._grab is None:
                if (Vector((event.mouse_x, event.mouse_y)) - self._pull_from).length >= PULL_PX:
                    i = self._main_index()
                    if i is None or not self._start_grab(context, i, m, absolute=True):
                        self._pull = False
            if self._grab is not None:
                self._drag_to(context, event)
                self._drag_refresh(context)
            else:
                hot = self._handle_at(context, m)
                if hot != self._hot:
                    self._hot = hot
                    self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        if et == 'LEFTMOUSE' and event.value == 'RELEASE':
            self._pull = False
            if self._grab is not None:
                self._grab = None
                self._hot = self._handle_at(context, self._mouse(context, event))
                self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        if et in {'LEFT_SHIFT', 'RIGHT_SHIFT', 'LEFT_CTRL', 'RIGHT_CTRL', 'OSKEY'}:
            return {'RUNNING_MODAL'}      # takes effect on the next move
        if event.value != 'PRESS':
            return {'RUNNING_MODAL'}
        if et == 'LEFTMOUSE':
            i = self._handle_at(context, self._mouse(context, event))
            if i is not None and self._start_grab(context, i, self._mouse(context, event)):
                self._drag_refresh(context)
                return {'RUNNING_MODAL'}
            return self._apply(context)   # a click away from the handles applies
        if et in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
            return self._apply(context)
        if et in {'ESC', 'RIGHTMOUSE'}:
            self._drag_end(context)
            return {'CANCELLED'}
        if self._grab is not None:
            return {'RUNNING_MODAL'}      # no typing while dragging
        if et == 'BACK_SPACE':
            self._typed = self._typed[:-1]
            if self._typed:
                self._apply_typed(context)
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        if et == 'TAB':
            if self._typed:
                self._apply_typed(context)
            self._typed = ""
            self._field = (self._field + 1) % len(self.drag_fields)
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        ch = getattr(event, "unicode", "") or NUMPAD_CHARS.get(et, "")
        if ch and not event.ctrl and not event.oskey:
            if self._typed or ch in NUMBER_START:
                self._typed += ch        # once typing, letters are units (mm, ft)
                self._apply_typed(context)
                self._drag_refresh(context)
                return {'RUNNING_MODAL'}
            if self.drag_key(context, event, ch.lower()):
                self._drag_refresh(context)
                return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}

    def _drag_refresh(self, context):
        name = self._field_name()
        try:
            pv = self.drag_preview()
            handles = self.drag_handles()
        except Exception as exc:
            print("NURBS value preview:", exc)
            pv, handles = None, []
        if pv is not None:
            active_i = self._grab["i"] if self._grab is not None else self._hot
            pv["handles"] = []
            pv["labels"] = []
            for i, h in enumerate(handles):
                f = h["field"]
                hot = i == active_i
                pos = self._pos(h)
                pv["handles"].append({"kind": h["kind"], "pos": pos, "base": h.get("base"),
                                      "dir": h["dir"], "hot": hot, "active": f == name})
                val = f"{self._typed}|" if (f == name and self._typed) else self._fmt(context, f)
                if f == name or hot or self._is_angle(f) and abs(getattr(self, f)) > 1e-9:
                    pv["labels"].append((pos, f"{self._label(f)} {val}", f == name))
        draw.pick_overlay["preview"] = pv
        state = []
        for f in self.drag_fields:
            val = (self._typed + "|") if (f == name and self._typed) else self._fmt(context, f)
            state.append(f"[{self._label(f)} {val}]" if f == name and len(self.drag_fields) > 1
                         else f"{self._label(f)} {val}")
        state += list(self.drag_state())
        if self._msg:
            state.append(self._msg)
            self._msg = ""
        rows = [("Handles", [hints.key("LMB drag", "arrow: distance"),
                             hints.key("LMB drag", "diamond: draft")] if "draft" in self.drag_fields
                 else [hints.key("LMB drag", "a handle")]),
                ("Value", [hints.key("0-9", "type it", on=True if self._typed else None),
                           hints.key("Ctrl", "snap to grid"), hints.key("Shift", "precise"),
                           hints.key("Backspace", "erase")])]
        if len(self.drag_fields) > 1:
            rows[1][1].append(hints.cycle("Tab", [self._label(f) for f in self.drag_fields], self._label(name)))
        rows += list(self.drag_rows())
        rows.append(("", [hints.note("click away from the handles or Enter: make it")]))
        hints.show(context, self.bl_label, state, rows, MOUSE)
        context.area.tag_redraw()

    def _drag_end(self, context):
        self._dragging = False
        self._grab = None
        draw.pick_overlay["preview"] = None
        hints.clear(context)

    # -- defaults --------------------------------------------------------------
    def drag_key(self, context, event, ch):
        return False

    def drag_check(self):
        return ""

    def drag_rows(self):
        return []

    def drag_state(self):
        return []


class _Probe:
    """A mouse position in window pixels, for PickMixin._in_view."""
    type = 'NONE'
    value = 'NOTHING'

    def __init__(self, x, y):
        self.mouse_x, self.mouse_y = x, y


def draft_angle(off, h):
    """Draft (degrees) that moves the top edge out by off at height h."""
    if abs(h) < 1e-12:
        return None
    return math.degrees(math.atan(off / h))
