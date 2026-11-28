"""Pinned measurements: readings kept in the file that follow the model.

P in Measure (or the pin in the Analyze panel's Selection box) pins what is
measured: one thing's own reading (a circle's radius, a point's X Y Z, a
face's radii at a point) or what two things measure against each other.
A pin stores what it measures, not the numbers:

* a point: its position, in the local space of the object it was snapped to
  (it moves with that object), else in world space;
* a surface edge or a face: the object, its index and a point on it (local
  space), found again after a rebuild renumbers them (subsel's anchors);
* a curve spline: the object and the spline.

So a pin re-measures whenever its objects change or move, and is drawn
in the 3D view with its value (Analyze > Measurements lists them, the
header's ruler shows / hides them all). When what it measured is gone (the
object deleted, the face no longer there) it says *stale* and keeps its last
value. Pins live in the scene (Scene.nproto_pins), are saved with the file
and take part in undo. Nothing in the model changes.
"""

import json
import uuid

import numpy as np

import bpy
from bpy.props import BoolProperty, CollectionProperty, IntProperty, StringProperty

from . import store

COL_PIN = (0.55, 0.85, 1.0, 1.0)
COL_STALE = (1.0, 0.45, 0.4, 1.0)
_cache = {}                 # pin id -> (key, reading or None)
_last = {}                  # pin id -> (reading, label anchor): the last good reading, kept while stale


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

class NPROTO_PG_measure_pin(bpy.types.PropertyGroup):
    """A pinned measurement (what it measures, as JSON in data)."""
    label: StringProperty(name="Name", default="")
    visible: BoolProperty(name="Show", default=True, description="Show this measurement in the 3D view",
                          update=lambda s, c: _redraw())
    data: StringProperty(default="{}")


def _redraw():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


def pins(scene):
    return getattr(scene, "nproto_pins", None) or []


def _M(obj):
    from . import geom
    return np.array(geom.world_matrix(obj), float)


def _to_local(obj, P):
    return (np.linalg.inv(_M(obj)) @ np.r_[np.asarray(P, float), 1.0])[:3]


def _to_world(obj, q):
    return (_M(obj) @ np.r_[np.asarray(q, float), 1.0])[:3]


def point_ref(P, obj=None):
    """A point, kept in obj's local space (it follows obj) or in world space."""
    if obj is not None:
        return {"t": "point", "obj": obj.name, "co": [float(x) for x in _to_local(obj, P)]}
    return {"t": "point", "obj": "", "co": [float(x) for x in P]}


def element_ref(key, at=None):
    """A ref for a measurepair element key ("L", object, ref) / ("F", object, face)
    or the readout's ("E", object, edge) / ("S", object, spline); at: the world
    point a one-thing reading was taken at (a face's radii, a curve's curvature)."""
    from . import subsel
    t, name, i = key[0], key[1], key[2]
    obj = bpy.data.objects.get(name)
    if obj is None:
        return None
    if t == "E":
        t, i = "L", f"E{i}"
    elif t == "S":
        t = "L"
    ref = {"t": t, "obj": name}
    if t == "L":
        ref["ref"] = i
        if str(i).startswith("E") and store.is_nurbs(obj):
            ref["anchor"] = subsel._anchor(obj, "EDGE", int(str(i)[1:]))
    else:
        ref["i"] = int(i)
        ref["anchor"] = subsel._anchor(obj, "FACE", int(i))
    if at is not None:
        ref["at"] = [float(x) for x in _to_local(obj, at)]
    return ref


def add(scene, a, b=None, diameter=False, label=""):
    """Pin a reading of ref a (and b): the new pin."""
    p = scene.nproto_pins.add()
    p.data = json.dumps({"id": uuid.uuid4().hex[:12], "a": a, "b": b, "dia": bool(diameter)})
    n = 1 + max([int(q.label[1:]) for q in scene.nproto_pins if q.label[:1] == "M" and q.label[1:].isdigit()] or [0])
    p.label = label or f"M{n}"
    _redraw()
    return p


def _data(p):
    try:
        return json.loads(p.data or "{}")
    except ValueError:
        return {}


# --------------------------------------------------------------------------
# Measuring a pin again
# --------------------------------------------------------------------------

def _okey(obj):
    """What a pin's reading depends on in obj: its geometry and transform."""
    from . import measureinfo
    if obj is None:
        return None
    if store.is_nurbs(obj):
        return ("N", obj.get(store.PROP_REV, 0), tuple(np.round(_M(obj), 12).ravel()))
    return measureinfo._key(obj)


def _objects(d):
    return [r.get("obj") for r in (d.get("a"), d.get("b")) if r and r.get("obj")]


def resolve(ref):
    """(the measurepair element ref means now, the world point a one-thing reading
    is at) or None when it is gone."""
    from . import measurepair, subsel
    if ref is None:
        return None
    name = ref.get("obj") or ""
    obj = bpy.data.objects.get(name) if name else None
    if name and obj is None:
        return None
    if ref["t"] == "point":
        P = _to_world(obj, ref["co"]) if obj is not None else np.array(ref["co"], float)
        return measurepair.point_el(P), P
    at = _to_world(obj, ref["at"]) if ref.get("at") is not None else None
    if ref["t"] == "L":
        r = str(ref["ref"])
        if r.startswith("E") and store.is_nurbs(obj):
            i = subsel._locate(obj, "EDGE", int(r[1:]), ref.get("anchor"), subsel._size(obj)) \
                if ref.get("anchor") is not None else int(r[1:])
            if i is None:
                return None
            return measurepair.edge_el(obj, i), at
        if obj.type != 'CURVE' or int(r[1:] or 0) >= len(obj.data.splines):
            return None
        return measurepair.spline_el(obj, r), at
    if not store.is_nurbs(obj):
        return None
    i = subsel._locate(obj, "FACE", int(ref["i"]), ref.get("anchor"), subsel._size(obj))
    if i is None:
        return None
    return measurepair.face_el(obj, i), at


def _single(el, at, ref):
    """One thing's own reading, as Measure shows it."""
    from . import features, linesolve
    from .measure import curve_radius, surface_radius
    t = el["type"]
    if t == "point":
        return {"kind": "point", "point": el["P"]}
    if el["what"] in ("edge", "curve"):
        obj = bpy.data.objects.get(ref["obj"])
        c = linesolve.kernel_curve(features.input_curve(obj, el["key"][2] if el["key"][0] == "S" else
                                                        f"E{el['key'][2]}"), store._k()[0])
        m = curve_radius(c, at if at is not None else el["P"])
        if m is not None and el.get("length") is not None:
            m["length"] = el["length"]
        elif m is not None and t == "line":
            m["length"] = el["length"]
        return m
    obj = bpy.data.objects.get(ref["obj"])
    return surface_radius(obj, at if at is not None else el["P"])


def reading(p):
    """The pin's reading now: {"kind": ...} as Measure's, or None when stale."""
    from . import measurepair
    d = _data(p)
    pid = d.get("id", p.data)
    key = (p.data,) + tuple(_okey(bpy.data.objects.get(n)) for n in _objects(d))
    hit = _cache.get(pid)
    if hit is not None and hit[0] == key:
        return hit[1]
    m = None
    try:
        A = resolve(d.get("a"))
        B = resolve(d.get("b")) if d.get("b") else None
        if A is not None and (B is not None or not d.get("b")):
            if B is None:
                m = _single(A[0], A[1], d["a"])
            else:
                m = {"kind": "pair", "r": measurepair.relate(A[0], B[0])}
    except Exception as exc:
        print("NURBS pinned measurement:", p.label, exc)
        m = None
    _cache[pid] = (key, m)
    if m is not None:
        from .measure import geometry
        _last[pid] = (m, geometry(m)[2])
    return m


def text(context, p):
    """(the pin's value as text, stale)."""
    from .measure import label
    d = _data(p)
    m = reading(p)
    if m is not None:
        return label(context, m, d.get("dia", False)), False
    old = _last.get(d.get("id", p.data))
    return (label(context, old[0], d.get("dia", False)) if old else ""), True


# --------------------------------------------------------------------------
# Drawing (every 3D view, Object and Edit Mode)
# --------------------------------------------------------------------------

def draw_px(ctx, region, rv3d, box):
    import blf
    import gpu
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from gpu_extras.batch import batch_for_shader
    from . import hints
    from .measure import geometry, label
    sc = ctx.scene
    if not pins(sc) or not getattr(sc, "nproto_pins_show", True):
        return
    sp = ctx.space_data
    if sp is None or not getattr(sp.overlay, "show_overlays", True):
        return
    s = getattr(ctx.preferences.system, "ui_scale", 1.0) or 1.0
    font = hints.FONT
    blf.size(font, hints.SIZE * s)
    line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    flat = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')

    def q2(P):
        return location_3d_to_region_2d(region, rv3d, tuple(float(x) for x in P))

    for p in pins(sc):
        if not p.visible:
            continue
        d = _data(p)
        pid = d.get("id", p.data)
        m = reading(p)
        stale = m is None
        if not stale:
            lines, pts, at = geometry(m)
        elif pid in _last:
            m, at = _last[pid]
            lines, pts, _ = geometry(m)
        else:
            continue
        col = COL_STALE if stale else COL_PIN
        segs = []
        for pl in lines:
            qs = [q2(P) for P in pl]
            for a, b in zip(qs[:-1], qs[1:]):
                if a is not None and b is not None:
                    segs += [(a.x, a.y), (b.x, b.y)]
        if segs:
            for c, lw in (((0.0, 0.0, 0.0, 0.55), 3.5 * s), (col, 1.5 * s)):
                bt = batch_for_shader(line, 'LINES', {"pos": segs})
                line.bind()
                line.uniform_float("color", c)
                line.uniform_float("lineWidth", lw)
                line.uniform_float("viewportSize", (region.width, region.height))
                bt.draw(line)
        for P in pts:
            q = q2(P)
            if q is None:
                continue
            r = 2.5 * s
            bt = batch_for_shader(flat, 'TRIS', {"pos": [(q.x - r, q.y - r), (q.x + r, q.y - r), (q.x + r, q.y + r),
                                                          (q.x - r, q.y - r), (q.x + r, q.y + r), (q.x - r, q.y + r)]})
            flat.bind()
            flat.uniform_float("color", col)
            bt.draw(flat)
        if at is None:
            continue
        q = q2(at)
        if q is None:
            continue
        txt = f"{p.label}  {label(ctx, m, d.get('dia', False))}" + ("  (stale)" if stale else "")
        w, h = blf.dimensions(font, txt)
        x, y = q.x + 10 * s, q.y + 8 * s
        box(x - 5 * s, y - 5 * s, w + 10 * s, h + 10 * s, hints.COL_BG)
        blf.color(font, *col)
        blf.position(font, x, y, 0)
        blf.draw(font, txt)
    gpu.state.blend_set('NONE')


# --------------------------------------------------------------------------
# Panel and operators
# --------------------------------------------------------------------------

def draw_panel(col, context):
    """Analyze > Measurements: the pins, each with its value, eye and remove."""
    sc = context.scene
    if not hasattr(sc, "nproto_pins"):
        return
    box = col.box()
    head = box.row(align=True)
    n = len(sc.nproto_pins)
    head.label(text=f"Measurements ({n})" if n else "Measurements", icon='PINNED')
    if n:
        head.prop(sc, "nproto_pins_show", text="", emboss=False,
                  icon='HIDE_OFF' if sc.nproto_pins_show else 'HIDE_ON')
        head.operator("nproto.measure_pin_copy", text="", icon='COPYDOWN', emboss=False)
        head.operator("nproto.measure_pin_clear", text="", icon='TRASH', emboss=False)
    else:
        box.label(text="P in Measure pins a reading here")
        return
    for k, p in enumerate(sc.nproto_pins):
        r = box.row(align=True)
        r.prop(p, "visible", text="", emboss=False, icon='HIDE_OFF' if p.visible else 'HIDE_ON')
        txt, stale = text(context, p)
        sub = r.row(align=True)
        sub.alert = stale
        sub.label(text=f"{p.label}  {txt}" + ("  (stale)" if stale else ""))
        op = r.operator("nproto.measure_pin_remove", text="", icon='X', emboss=False)
        op.index = k


def header(layout, context):
    """The viewport header's ruler: shows / hides the pins, with their count."""
    sc = context.scene
    n = len(pins(sc))
    if n:
        layout.prop(sc, "nproto_pins_show", text=str(n), icon='DRIVER_DISTANCE', toggle=True)


def copy_lines(context):
    """One line a pin: its name, value, and (stale)."""
    out = []
    for p in pins(context.scene):
        txt, stale = text(context, p)
        out.append(f"{p.label}: {txt}" + (" (stale)" if stale else ""))
    return out


class NPROTO_OT_measure_pin_remove(bpy.types.Operator):
    """Remove this pinned measurement"""
    bl_idname = "nproto.measure_pin_remove"
    bl_label = "Remove Measurement"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    index: IntProperty(default=-1)

    def execute(self, context):
        sc = context.scene
        if not 0 <= self.index < len(sc.nproto_pins):
            return {'CANCELLED'}
        sc.nproto_pins.remove(self.index)
        _redraw()
        return {'FINISHED'}


class NPROTO_OT_measure_pin_clear(bpy.types.Operator):
    """Remove every pinned measurement of this scene"""
    bl_idname = "nproto.measure_pin_clear"
    bl_label = "Clear Measurements"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        context.scene.nproto_pins.clear()
        _redraw()
        return {'FINISHED'}


class NPROTO_OT_measure_pin_copy(bpy.types.Operator):
    """Copy every pinned measurement to the clipboard, one a line"""
    bl_idname = "nproto.measure_pin_copy"
    bl_label = "Copy Measurements"
    bl_options = {'REGISTER'}

    def execute(self, context):
        lines = copy_lines(context)
        if not lines:
            return {'CANCELLED'}
        context.window_manager.clipboard = "\n".join(lines)
        self.report({'INFO'}, f"Copied {len(lines)} measurement" + ("s" if len(lines) > 1 else ""))
        return {'FINISHED'}


class NPROTO_OT_measure_pin_selection(bpy.types.Operator):
    """Pin what the two selected things measure against each other (the
    selection readout's distance, angle, centres): it stays in the view and
    follows the model"""
    bl_idname = "nproto.measure_pin_selection"
    bl_label = "Pin Measurement"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import measureinfo
        els = measureinfo.pair_refs(context)
        if not els:
            self.report({'WARNING'}, "Pick two things to pin what they measure")
            return {'CANCELLED'}
        a, b = (element_ref((t, o.name, i)) for t, o, i in els)
        p = add(context.scene, a, b)
        self.report({'INFO'}, f"Pinned {p.label}")
        return {'FINISHED'}


classes = (NPROTO_PG_measure_pin, NPROTO_OT_measure_pin_remove, NPROTO_OT_measure_pin_clear,
           NPROTO_OT_measure_pin_copy, NPROTO_OT_measure_pin_selection)


def register():
    bpy.types.Scene.nproto_pins = CollectionProperty(type=NPROTO_PG_measure_pin)
    bpy.types.Scene.nproto_pins_show = BoolProperty(
        name="Show Measurements", default=True, description="Show the pinned measurements in the 3D view",
        update=lambda s, c: _redraw())


def unregister():
    for k in ("nproto_pins", "nproto_pins_show"):
        if hasattr(bpy.types.Scene, k):
            delattr(bpy.types.Scene, k)
