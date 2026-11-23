"""Align Parts: mate a snap element of one part with one of another.

Click an element of the part to move (a ring, a cylinder's axis, a face,
its centre, an edge, a vertex, an edge's middle: snapframes.element), then
one of the part it goes to. Hovering the second shows the result as a
ghost; the click fixes it. The part turns the least that lines the two up
(faces flush and facing each other, rings and axes coaxial, edges parallel)
and moves the least that brings them together (alignsolve.mate).

The part that moves: the selection when the first element lies on it, else
the object the element belongs to (no need to select first).

After the second click, before it is applied:
  F       flip (faces back to back, the other way along an axis)
  R       spin a quarter turn about the mated direction, Shift+R 15°
  0-9     a gap off the target (Tab fixes it; negative sinks in)
  D       keep the original: a copy goes there
  click   on the moving part: a second pair (its element, then one of the
          target) that sets only the spin; on another part: that element
          becomes the target
  Enter / Space / right click / a click on nothing: done; Esc cancels;
  Backspace steps back.
F, R, D and typing work while hovering the target too. It is one undo step,
and the Adjust Last Operation panel changes the flip, spin, gap and copy
afterwards. Object Mode only.

Started by holding A (a tap is still Select All; movetool.HOLD_TOOLS), from
the toolbar (after Move Pivot: its first click is the first element, and the
tool before it comes back when it ends, as Move Pivot's does), Object >
Transform, the right-click menu, the NURBS Proto menu or F3.
"""

import os

import math

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, PointerProperty, StringProperty
from mathutils import Vector

from . import alignsolve, draw, hints, pattern, pivottool, snapframes, snapping
from .drawtools import REFRESH_KEYS, parse_length
from .movetool import _PointsTool, _duplicate, _len_text, roots, transform_objects

SPIN_STEP = 90.0
SPIN_FINE = 15.0
COL_SRC = (0.75, 0.75, 0.75, 0.9)       # the moving part's element, where it was picked
COL_DST = (0.25, 1.0, 0.45, 1.0)        # the target's element (snap green)
COL_SECOND = (1.0, 0.62, 0.12, 1.0)     # a second pair's elements
ELEMENT_TYPES = [(t, t.title(), "") for t in ("POINT", "LINE", "PLANE", "RING", "AXIS")]
RELATION_TEXT = {"opposed": "facing", "parallel": "parallel", "in plane": "in the face", "none": ""}


class NPROTO_PG_align_element(bpy.types.PropertyGroup):
    """A snap element as the operator keeps it (world space, before the move)."""
    type: EnumProperty(name="Type", items=ELEMENT_TYPES, default="POINT")
    label: StringProperty(name="Label")
    o: FloatVectorProperty(name="Origin", size=3, subtype='XYZ', unit='LENGTH')
    d: FloatVectorProperty(name="Direction", size=3, subtype='DIRECTION')
    hint: FloatVectorProperty(name="Hint", size=3, subtype='DIRECTION')
    pin: BoolProperty(name="Pinned")

    def put(self, el):
        self.type, self.label = el["type"], el.get("label", "")
        self.o = tuple(map(float, el["o"]))
        self.d = tuple(map(float, el["d"])) if el.get("d") is not None else (0.0, 0.0, 0.0)
        self.hint = tuple(map(float, el["hint"])) if el.get("hint") is not None else (0.0, 0.0, 0.0)
        self.pin = bool(el.get("pin"))

    def get(self):
        v = lambda x: None if float(np.linalg.norm(x)) < 1e-9 else np.array(x, float)
        e = alignsolve.make(self.type, np.array(self.o, float), v(self.d), pin=self.pin, hint=v(self.hint))
        e["label"] = self.label
        return e


def descendants(objs):
    """objs and every child under them (they move with their parents)."""
    out, todo = {}, list(objs)
    while todo:
        o = todo.pop()
        if o.name in out:
            continue
        out[o.name] = o
        todo += list(o.children)
    return list(out.values())


def moving_set(selected, owner):
    """The roots that move when the first element lies on `owner`: the
    selection if owner is in it (or under it), else owner alone."""
    sel = roots(list(selected))
    names = {o.name for o in descendants(sel)}
    if owner is not None and owner.name in names:
        return sel
    return [owner] if owner is not None else []


def copy_object(context, src, M):
    """A copy of src under the world 4x4 M: a NURBS object as a fresh object
    with the same B-rep (no history), anything else duplicated."""
    if pattern.usable(src) and src.type != 'CURVE':
        return pattern._new_copy(context, src, np.asarray(M, float) @ np.array(src.matrix_world, float), "a")
    return _duplicate(context, src, M)


class NPROTO_OT_align_parts(_PointsTool, bpy.types.Operator):
    """Mate an element of one part with one of another: click a ring, axis,
    face, centre, edge or point of the part to move, then one of the target.
    F flips, R spins (Shift+R 15°), type a gap, D leaves a copy, a second
    pair sets the spin"""
    bl_idname = "nproto.align_parts"
    bl_label = "Align Parts"
    bl_options = {'REGISTER', 'UNDO'}

    title = "Align Parts"
    held_key = 'NONE'
    allow_copies = False
    face_planes = False

    a: PointerProperty(type=NPROTO_PG_align_element, options={'SKIP_SAVE'})
    b: PointerProperty(type=NPROTO_PG_align_element, options={'SKIP_SAVE'})
    flip: BoolProperty(name="Flip", default=False, options={'SKIP_SAVE'},
                       description="The other way round: faces back to back, the other sense along an axis")
    spin: FloatProperty(name="Spin", subtype='ANGLE', default=0.0, options={'SKIP_SAVE'},
                        description="Turn about the mated direction")
    gap: FloatProperty(name="Gap", subtype='DISTANCE', default=0.0, options={'SKIP_SAVE'},
                       description="Distance off the target (negative: into it)")
    keep: BoolProperty(name="Copy", default=False, options={'SKIP_SAVE'},
                       description="Leave the original where it is: a copy goes there (D in the tool)")
    from_tool: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})    # the toolbar tool's click started it
    parts: StringProperty(name="Parts", options={'SKIP_SAVE', 'HIDDEN'},
                          description="The objects that move, one name a line (empty: the selection)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def draw(self, context):
        lay = self.layout
        lay.label(text=f"{self.a.label or self.a.type.title()} → {self.b.label or self.b.type.title()}")
        lay.prop(self, "flip")
        lay.prop(self, "spin")
        lay.prop(self, "gap")
        lay.prop(self, "keep")

    # -- doing it -------------------------------------------------------------
    def _solve(self, a, b, flip=None, spin=None, gap=None, second=None):
        return alignsolve.mate(a, b, self.flip if flip is None else flip,
                               float(self.spin) if spin is None else spin,
                               float(self.gap) if gap is None else gap, second)

    def execute(self, context):
        named = [bpy.data.objects.get(n) for n in self.parts.split("\n") if n]
        objs = roots([o for o in named if o is not None]) if self.parts else roots(list(context.selected_objects))
        if objs and self.parts:             # (a redo starts from the selection before the tool)
            for o in context.selected_objects:
                o.select_set(False)
            for o in objs:
                o.select_set(True)
            context.view_layer.objects.active = objs[0]
        if not objs:
            self.report({'WARNING'}, "Align Parts: nothing to move")
            return {'CANCELLED'}
        r = self._solve(self.a.get(), self.b.get())
        M = r["M"]
        if np.allclose(M, np.eye(4), atol=1e-12) and not self.keep:
            self.report({'INFO'}, "Align Parts: already in place")
            return {'FINISHED'}
        what = f"{self.a.label or self.a.type.title()} → {self.b.label or self.b.type.title()}"
        amount = f"turned {r['angle']:.4g}°, moved {_len_text(context, r['dist'])}"
        if not self.keep:
            off = transform_objects(objs, M)
            if off:
                self.report({'WARNING'}, f"Align Parts: {', '.join(off[:3])}: its scale is slanted to the turn; "
                                         "placed as near as the object allows (apply its scale first)")
            self.report({'INFO'}, f"Align Parts: {what}, {amount}")
            return {'FINISHED'}
        made = []
        for o in objs:
            try:
                made.append(copy_object(context, o, M))
            except Exception as exc:
                self.report({'ERROR'}, f"Align Parts: {o.name} not copied: {exc}")
        if not made:
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for o in made:
            o.select_set(True)
        context.view_layer.objects.active = made[0]
        self.report({'INFO'}, f"Align Parts: {len(made)} cop{'y' if len(made) == 1 else 'ies'}, {what}, {amount}")
        return {'FINISHED'}

    # -- starting -------------------------------------------------------------
    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Align Parts: run it in a 3D View")
            return {'CANCELLED'}
        self._edit = False
        self._objs, self._pv, self._pts = [], [], np.zeros((0, 3))
        self._center = np.zeros(3)
        self._stage = "SRC"         # SRC: the moving part's element; DST: the target's; REFINE; SECOND
        self._a = self._b = None    # the elements fixed
        self._a2 = None             # a second pair's first element
        self._el = None             # the element under the mouse
        self._flip, self._spin, self._keep = False, 0.0, False
        self._others = None         # snap targets of everything but the moving parts
        self._start_common(context)
        context.window_manager.modal_handler_add(self)
        if self.from_tool and event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._click(context, event)         # the toolbar tool's click picks the first element
        if self._stage == "SRC":
            self._refresh(context, event)
        return {'RUNNING_MODAL'}

    def _snap_targets(self, context):
        """Every visible part, its CVs only where its net is shown; all the
        mesh elements (the faces too: they give planes, axes, spheres)."""
        return snapping.collect(context.visible_objects, lambda o: draw.cv_net_shown(context, o))

    def _set_moving(self, context, owner):
        self._objs = moving_set(context.selected_objects, owner)
        if not self._objs:
            return False
        moving = {o.name for o in descendants(self._objs)}
        self._others = snapping.collect([o for o in context.visible_objects if o.name not in moving],
                                        lambda o: draw.cv_net_shown(context, o))
        try:
            self._pv = pattern._preview_sources(descendants(self._objs))
        except Exception as exc:
            print("NURBS Align Parts preview:", exc)
            self._pv = []
        return True

    def _moving_names(self):
        return {o.name for o in descendants(self._objs)}

    # -- typing: the gap ------------------------------------------------------
    def typing_ready(self):
        return getattr(self, "_stage", "SRC") != "SRC"

    def _live_value(self, context):
        if self._buf:
            try:
                return parse_length(context, self._buf)
            except Exception:
                return self._value
        return self._value

    def _gap(self, context):
        g = self._live_value(context)
        return 0.0 if g is None else float(g)

    # -- picking --------------------------------------------------------------
    def _element(self, context, event, targets):
        try:
            return snapframes.element(targets, self._view(context), self._mouse(context, event))
        except Exception as exc:
            print("NURBS Align Parts snap:", exc)
            return None

    def _hover_el(self, context, event):
        """The element under the mouse for this stage, and whether it lies on
        the moving parts (a second pair's first) -> (element, on_moving)."""
        if event is None:
            return None, False
        if self._stage == "SRC":
            return self._element(context, event, self._snaps), False
        if self._stage == "REFINE":
            el = self._element(context, event, self._snaps)
            if el is not None and el.get("obj") in self._moving_names():
                return el, True
            return self._element(context, event, self._others), False
        return self._element(context, event, self._others), False

    # -- display ----------------------------------------------------------------
    def _result(self, context, b=None, second=None):
        """mate() of the fixed source with b (or the fixed target) as the tool stands."""
        b = b if b is not None else self._b
        if self._a is None or b is None:
            return None
        if second is not None:
            return self._solve(self._a, b, self._flip, 0.0, self._gap(context), second)
        return self._solve(self._a, b, self._flip, self._spin, self._gap(context))

    def _status(self, context, r=None, el=None, on_moving=False):
        what = (self._objs[0].name + (f" +{len(self._objs) - 1}" if len(self._objs) > 1 else "")) \
            if self._objs else "a part"
        st = self._stage
        if st == "SRC":
            step = "click an element of the part to move"
        elif st == "DST":
            step = "click the element to mate it with"
        elif st == "SECOND":
            step = "second pair: click the target's element to line it up with"
        else:
            step = ("second pair: click to start it with this element" if on_moving else
                    "Enter: done · click the moving part: a second pair · another part: a new target")
        state = [what, step]
        if el is not None:
            state.append(el.get("label", el["type"].title()))
        if r is not None:
            rel = RELATION_TEXT.get(r["relation"], "")
            state.append((rel + " · " if rel else "") + f"turn {r['angle']:.4g}°, move {_len_text(context, r['dist'])}")
        if self._buf:
            state.append(f"gap {self._buf}_")
        elif self._value is not None:
            state.append("gap " + _len_text(context, self._value))
        if self._flip:
            state.append("flipped")
        if abs(self._spin) > 1e-9:
            state.append(f"spin {math.degrees(self._spin):.4g}°")
        if self._keep:
            state.append("copy")
        ready = st != "SRC"
        rows = [("Mate", [hints.toggle("F", "flip", self._flip),
                          hints.key("R", f"spin {SPIN_STEP:g}°", on=None if ready else False),
                          hints.key("Shift+R", f"spin {SPIN_FINE:g}°", on=None if ready else False)]),
                ("Type", [hints.key("0-9", "gap off the target", on=None if ready else False),
                          hints.key("Tab", "fix the typed value", on=None if self._buf else False)]),
                ("Copy", [hints.toggle("D", "keep the original", self._keep)]),
                ("Done", [hints.key("Backspace", "step back"), hints.key("Enter", "done",
                                                                          on=None if st == "REFINE" else False)])]
        hints.show(context, "Align Parts", state, rows,
                   [("MOUSE_LMB", "Element"), ("MOUSE_RMB", "Done" if st == "REFINE" else "Cancel"),
                    ("EVENT_ESC", "Cancel")])

    def _show(self, context, r, els, label_at, label):
        ov = draw.sketch_overlay
        ov["elements"] = [g for el, col in els if el is not None for g in snapframes.overlay(el, col)]
        if r is not None and self._pv:
            pv = self._ghosts([r["M"]])
            pv["labels"] = [(Vector(tuple(map(float, label_at))), label, True)]
            draw.pick_overlay["preview"] = pv
        else:
            draw.pick_overlay["preview"] = None

    def _preview(self, context, event, hover):
        el, on_moving = hover
        self._el = el
        ov = draw.sketch_overlay
        ov["snap"] = Vector(tuple(map(float, el["P"]))) if el is not None else None
        ov["snap_label"] = el.get("label") if el is not None else None
        st = self._stage
        r = None
        els = []
        if st == "SRC":
            els = [(el, COL_DST)]
        elif st == "DST":
            r = self._result(context, el) if el is not None else None
            els = [(self._a, COL_SRC), (el, COL_DST)]
        elif st == "REFINE":
            r = self._result(context)
            els = [(self._a, COL_SRC), (self._b, COL_DST), (el, COL_SECOND if on_moving else COL_DST)]
        else:   # SECOND
            r = self._result(context, second=(self._a2, el)) if el is not None else self._result(context)
            els = [(self._a, COL_SRC), (self._b, COL_DST), (self._a2, COL_SECOND), (el, COL_SECOND)]
        self._r = r
        label = ""
        if r is not None:
            label = f"{self._a.get('label', '')} → {(self._b if st != 'DST' else el).get('label', '')}"
        self._show(context, r, els, r["p"] if r is not None else np.zeros(3), label)
        self._status(context, r, el, on_moving)
        context.area.tag_redraw()

    def _refresh(self, context, event):
        h = self._hover_el(context, event)
        self._last_event = event
        self._preview(context, event, h)
        return h

    # -- modal ------------------------------------------------------------------
    def _click(self, context, event):
        el, on_moving = self._hover_el(context, event)
        st = self._stage
        if st == "SRC":
            if el is None:
                return None
            owner = bpy.data.objects.get(el.get("obj") or "")
            if owner is None or not self._set_moving(context, owner):
                self.report({'INFO'}, "Align Parts: that element belongs to no part")
                return None
            self._a, self._stage = el, "DST"
        elif st == "DST":
            if el is None:
                return None
            self._b, self._stage = el, "REFINE"
        elif st == "REFINE":
            if el is None:
                return self._finish(context)
            if on_moving:
                self._a2, self._stage = el, "SECOND"
            else:
                self._b = el            # a new target
        else:   # SECOND
            if el is None:
                return None
            r = self._result(context, second=(self._a2, el))
            self._spin = r["spin"]
            self._a2, self._stage = None, "REFINE"
        self._refresh(context, event)
        return None

    def _back(self, context, event):
        if self._buf:
            self._buf = self._buf[:-1]
        elif self._stage == "SECOND":
            self._a2, self._stage = None, "REFINE"
        elif self._stage == "REFINE":
            self._b, self._stage = None, "DST"
        elif self._stage == "DST":
            if self._value is not None:
                self._value = None
            else:
                self._a, self._objs, self._pv, self._stage = None, [], [], "SRC"
        self._refresh(context, event)

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        et, ev = event.type, event.value
        if self._typed_key(context, event):
            self._refresh(context, event)
            return {'RUNNING_MODAL'}
        mods = event.ctrl or event.alt or event.oskey
        if et in REFRESH_KEYS:
            self._refresh(context, event)
        elif et == 'LEFTMOUSE' and ev == 'PRESS':
            r = self._click(context, event)
            if r is not None:
                return r
        elif ev == 'PRESS' and et == 'F' and not mods:
            self._flip = not self._flip
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'R' and not mods and self._stage != "SRC":
            self._spin += math.radians(SPIN_FINE if event.shift else SPIN_STEP)
            self._spin = math.remainder(self._spin, 2 * math.pi)
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'D' and not mods:
            self._keep = not self._keep
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'TAB':
            if self._fix_buffer(context):
                self._refresh(context, event)
        elif ev == 'PRESS' and et == 'BACK_SPACE':
            self._back(context, event)
        elif ev == 'PRESS' and et in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
            if self._buf:
                self._fix_buffer(context)
            if self._stage in ("REFINE", "SECOND"):
                return self._finish(context)
            if self._stage == "DST" and self._el is not None:
                self._b = self._el
                return self._finish(context)
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'RIGHTMOUSE' and self._stage in ("REFINE", "SECOND"):
            return self._finish(context)
        elif ev == 'PRESS' and et in {'ESC', 'RIGHTMOUSE'}:
            if et == 'ESC' and self._stage == "SECOND":
                self._a2, self._stage = None, "REFINE"
                self._refresh(context, event)
                return {'RUNNING_MODAL'}
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _end(self, context):
        draw.sketch_overlay["elements"] = []
        _PointsTool._end(self, context)
        if self.from_tool:                      # a one-off: the toolbar's tool before it comes back
            pivottool.restore_tool(context, TOOL_ID)

    def _finish(self, context):
        gap = self._gap(context)
        self._end(context)
        if self._a is None or self._b is None:
            return {'CANCELLED'}
        self.a.put(self._a)
        self.b.put(self._b)
        self.flip, self.spin, self.gap, self.keep = self._flip, float(self._spin), gap, self._keep
        self.parts = "\n".join(o.name for o in self._objs)     # these move (and end up selected)
        return self.execute(context)


def menu_transform(self, context):
    self.layout.operator(NPROTO_OT_align_parts.bl_idname, icon='SNAP_FACE')


# -- the toolbar tool (Object Mode, after Move Pivot) ---------------------------------

TOOL_ID = "nproto.align_parts_tool"
ICON = os.path.join(os.path.dirname(__file__), "icons", "ops.nproto.align_parts")     # tools/make_align_icon.py


class NPROTO_TL_align_parts(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'OBJECT'
    bl_idname = TOOL_ID
    bl_label = "Align Parts"
    bl_description = ("Click an element of the part to move (ring, axis, face, centre, edge, point), then one of "
                      "the part it goes to: it turns and moves to mate them. F flips, R spins, type a gap, D "
                      "leaves a copy, Enter applies. Also: hold A")
    bl_icon = ICON
    bl_keymap = ((NPROTO_OT_align_parts.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'},
                  {"properties": [("from_tool", True)]}),)

    def draw_settings(context, layout, tool):
        layout.label(text="Click an element of the part to move, then the target's · Enter applies · or hold A",
                     icon='SNAP_FACE')


def _after_pivot(mode):
    """The tool from the toolbar's end to just after Move Pivot's (see
    pivottool._after_transform for why register_tool's after= can't)."""
    from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
    tools = ToolSelectPanelHelper._tool_class_from_space_type('VIEW_3D')._tools[mode]
    ids = [getattr(it, "idname", None) for it in tools]
    if TOOL_ID not in ids or pivottool.TOOL_ID not in ids:
        return
    mine = ids.index(TOOL_ID)
    item = tools[mine]
    del tools[mine - 1:mine + 1]            # it and the separator register_tool put before it
    ids = [getattr(it, "idname", None) for it in tools]
    tools.insert(ids.index(pivottool.TOOL_ID) + 1, item)


classes = (NPROTO_PG_align_element, NPROTO_OT_align_parts)
MENUS = ("VIEW3D_MT_transform_object", "VIEW3D_MT_object_context_menu")


def register():
    for m in MENUS:
        getattr(bpy.types, m).append(menu_transform)
    pivottool.ONE_SHOT.add(TOOL_ID)
    try:
        bpy.utils.register_tool(NPROTO_TL_align_parts, separator=True)
        _after_pivot('OBJECT')
    except Exception as exc:
        print("NURBS Align Parts tool not in the toolbar:", exc)


def unregister():
    try:
        bpy.utils.unregister_tool(NPROTO_TL_align_parts)
    except Exception:
        pass
    pivottool.ONE_SHOT.discard(TOOL_ID)
    for m in MENUS:
        try:
            getattr(bpy.types, m).remove(menu_transform)
        except Exception:
            pass
