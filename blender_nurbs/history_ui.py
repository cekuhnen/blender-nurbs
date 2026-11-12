"""Construction history made visible and editable.

* Steps: every in-place history step (Trim, Edge Fillet, Rebuild...) can be
  clicked open to edit its settings live, muted (skipped) or removed.
* Tree: the chain of objects the active object is built from (hidden inputs
  included), each clickable to make it active (= query-edit it) and with an
  eye toggle; plus the objects built from the active one ("Used by").
"""

import json

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import features, store

_syncing = [False]

# settings each step type exposes (in this order)
STEP_FIELDS = {
    "trim": ["action", "scope", "projection"],
    "edge_fillet": ["shape", "rail_type", "radius", "variable", "end_radius", "continuity", "thumbweight",
                    "class_a", "ca_tolerance", "ca_angle", "ca_degree", "ca_continuity"],
    "rebuild_surface": ["degree_u", "degree_v", "spans_u", "spans_v"],
    "extend_surface": ["distance"],
    "refit_trim": ["refit_degree", "refit_spans_u", "refit_spans_v"],
}
CA_DEFAULTS = {"ca_tolerance": 1e-4, "ca_angle": 0.1, "ca_degree": 7}
ROLE = {"trim": "trim cutter", "fillet_trim": "fillet partner"}
STEP_NOTE = {"fillet_trim": "Follows its Surface Fillet (edit the fillet object)"}


def _raw_stack(obj):
    return json.loads(obj[features.PROP_STACK]) if features.PROP_STACK in obj else []


def _rebuild(obj, context):
    try:
        features.edit_rebuild(obj, context)   # dependents follow through the history handler
    except Exception as exc:
        print(f"NURBS history step edit ({obj.name}): {exc}")
        return str(exc)
    return None


def _on_step_change(self, context):
    if _syncing[0]:
        return
    obj = self.id_data
    raw = _raw_stack(obj)
    k = self.index
    if not 0 <= k < len(raw):
        return
    p = raw[k]["params"]
    op = raw[k]["op"]
    for name in STEP_FIELDS.get(op, []):
        v = getattr(self, name)
        if name == "variable":
            continue
        if name == "rail_type":                 # chord: the size is the distance between the rails
            p["chord"] = self.radius if v == "CHORD" else None
            continue
        if name == "end_radius":
            p["end_radius"] = v if self.variable and self.rail_type != "CHORD" else None
        elif name in ("continuity", "ca_continuity", "ca_degree"):
            p[name] = int(v)
        else:
            p[name] = v
    obj[features.PROP_STACK] = json.dumps(raw)
    err = _rebuild(obj, context)
    self.error = err or ""


def _S(**kw):
    kw.setdefault("update", _on_step_change)
    return kw


class NprotoStep(bpy.types.PropertyGroup):
    index: IntProperty(default=-1)          # the step open for editing (-1: none)
    focus: BoolProperty(default=False)      # opened from the Construction Tree: the panel shows only this step
    error: StringProperty()
    action: EnumProperty(name="Action", items=[("KEEP", "Keep", "Keep the picked side"),
                                               ("DELETE", "Discard", "Remove the picked side")], **_S())
    scope: EnumProperty(name="Scope", items=[("THROUGH", "Through", "Every region on the picked side"),
                                             ("PICKED", "Picked Region", "Only the clicked region")], **_S())
    projection: EnumProperty(name="Project", items=[
        ("CURVE", "Curve Normal", ""), ("VIEW", "View", ""), ("NORMAL", "Surface Normal", ""),
        ("X", "World X", ""), ("Y", "World Y", ""), ("Z", "World Z", "")], **_S())
    shape: EnumProperty(name="Shape", items=[("CIRCULAR", "Circular", ""), ("CHAMFER", "Chamfer", ""),
                                             ("BLEND", "Blend", "")], **_S())
    rail_type: EnumProperty(name="Rail", items=[
        ("RADIUS", "Rolling Ball", "The size is the radius of a ball rolling between the faces"),
        ("CHORD", "Chord", "The size is the distance between the two rails")], **_S())
    radius: FloatProperty(name="Radius", min=1e-4, soft_max=5.0, unit='LENGTH', **_S())
    variable: BoolProperty(name="Variable Radius", **_S())
    end_radius: FloatProperty(name="End Radius", min=1e-4, soft_max=5.0, unit='LENGTH', **_S())
    continuity: EnumProperty(name="Continuity", items=[("1", "G1", ""), ("2", "G2", ""), ("3", "G3", "")], **_S())
    thumbweight: FloatProperty(name="Fullness", min=0.1, soft_max=3.0, **_S())
    class_a: BoolProperty(name="Class-A Patches", **_S())
    ca_tolerance: FloatProperty(name="Tolerance", min=1e-6, soft_max=0.01, precision=5, unit='LENGTH', **_S())
    ca_angle: FloatProperty(name="Angle Tolerance", min=0.001, soft_max=1.0, precision=3, **_S())
    ca_degree: IntProperty(name="Max Degree", min=3, max=9, **_S())
    ca_continuity: EnumProperty(name="Ends", items=[("1", "G1", ""), ("2", "G2", "")], **_S())
    degree_u: IntProperty(name="Degree U", min=1, max=7, **_S())
    degree_v: IntProperty(name="Degree V", min=1, max=7, **_S())
    spans_u: IntProperty(name="Spans U", min=1, max=64, **_S())
    spans_v: IntProperty(name="Spans V", min=1, max=64, **_S())
    distance: FloatProperty(name="Distance", soft_min=-10.0, soft_max=10.0, unit='LENGTH', **_S())
    refit_degree: IntProperty(name="Degree", min=1, max=7, **_S())
    refit_spans_u: IntProperty(name="Spans U (0 auto)", min=0, max=32, **_S())
    refit_spans_v: IntProperty(name="Spans V (0 auto)", min=0, max=32, **_S())


def sync_step(obj, k):
    """Load stack step k's params into the step editor."""
    pg = obj.nproto_step
    raw = _raw_stack(obj)
    _syncing[0] = True
    try:
        pg.index = k
        pg.error = ""
        if not 0 <= k < len(raw):
            return
        p = raw[k]["params"]
        for name in STEP_FIELDS.get(raw[k]["op"], []):
            try:
                if name == "rail_type":
                    pg.rail_type = "CHORD" if p.get("chord") else "RADIUS"
                elif name == "variable":
                    pg.variable = p.get("end_radius") is not None
                elif name == "end_radius":
                    pg.end_radius = p["end_radius"] if p.get("end_radius") is not None else 2 * p.get("radius", 0.1)
                elif name == "continuity":
                    pg.continuity = str(int(p.get("continuity", 2)))
                elif name == "ca_continuity":
                    pg.ca_continuity = str(int(p.get("ca_continuity", 2)))
                elif name == "class_a":
                    pg.class_a = bool(p.get("class_a", False))
                elif name in CA_DEFAULTS and name not in p:
                    setattr(pg, name, CA_DEFAULTS[name])
                elif name in p:
                    setattr(pg, name, p[name])
            except (TypeError, ValueError):
                pass
    finally:
        _syncing[0] = False


class NPROTO_OT_open_step(bpy.types.Operator):
    """Show / hide this history step's settings (edit them live)"""
    bl_idname = "nproto.open_step"
    bl_label = "Edit History Step"
    bl_options = {'INTERNAL'}

    index: IntProperty()

    def execute(self, context):
        obj = context.view_layer.objects.active
        if obj is None:
            return {'CANCELLED'}
        obj.nproto_step.focus = False
        if obj.nproto_step.index == self.index:
            obj.nproto_step.index = -1
        else:
            sync_step(obj, self.index)
        return {'FINISHED'}


class NPROTO_OT_step_unfocus(bpy.types.Operator):
    """Back to the whole feature: its settings and all its history steps"""
    bl_idname = "nproto.step_unfocus"
    bl_label = "Show Feature"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        obj = context.view_layer.objects.active
        if obj is not None:
            obj.nproto_step.focus = False
        return {'FINISHED'}


class NPROTO_OT_mute_step(bpy.types.Operator):
    """Skip / re-apply this history step (the step is kept)"""
    bl_idname = "nproto.mute_step"
    bl_label = "Mute History Step"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty()
    name: StringProperty(options={'SKIP_SAVE'})    # the step's object (default: the active one)

    def execute(self, context):
        obj = bpy.data.objects.get(self.name) if self.name else context.view_layer.objects.active
        raw = _raw_stack(obj) if obj else []
        if not 0 <= self.index < len(raw):
            return {'CANCELLED'}
        p = raw[self.index]["params"]
        p["_off"] = not p.get("_off", False)
        obj[features.PROP_STACK] = json.dumps(raw)
        err = _rebuild(obj, context)
        if err:
            self.report({'ERROR'}, f"Rebuild failed: {err}")
        return {'FINISHED'}


class NPROTO_OT_history_focus(bpy.types.Operator):
    """Make this object active to see and edit its own feature and steps
    (a hidden input stays hidden; its edges are highlighted). Shift: also unhide it"""
    bl_idname = "nproto.history_focus"
    bl_label = "Go to Object"
    bl_options = {'UNDO'}

    name: StringProperty()
    root: StringProperty()      # the tree this was clicked in stays shown
    reveal: BoolProperty(default=False)

    def invoke(self, context, event):
        self.reveal = event.shift
        return self.execute(context)

    def execute(self, context):
        obj = bpy.data.objects.get(self.name)
        if obj is None:
            return {'CANCELLED'}
        _focus(context, obj, self.root, self.reveal)
        obj.nproto_step.focus = False          # the object itself: its feature and all steps
        return {'FINISHED'}


def _focus(context, obj, root, reveal=False):
    """Make obj active (a hidden input stays hidden: its edges are drawn as a
    ghost) and keep root's tree shown."""
    if reveal:
        features.set_shown(context, obj, True)
    for o in context.selected_objects:
        o.select_set(False)
    try:
        obj.select_set(True)
    except RuntimeError:
        pass
    context.view_layer.objects.active = obj
    from . import draw
    draw.ghost["name"] = obj.name if not features.is_shown(obj) else None
    context.scene["nproto_tree_root"] = root or obj.name
    for area in context.screen.areas if context.screen else []:
        area.tag_redraw()


class NPROTO_OT_tree_step(bpy.types.Operator):
    """Open this history step: its object becomes active with the step's
    settings shown for editing"""
    bl_idname = "nproto.tree_step"
    bl_label = "Edit History Step"
    bl_options = {'UNDO'}

    name: StringProperty()
    root: StringProperty()
    index: IntProperty()

    def execute(self, context):
        obj = bpy.data.objects.get(self.name)
        if obj is None:
            return {'CANCELLED'}
        _focus(context, obj, self.root)
        sync_step(obj, self.index)
        obj.nproto_step.focus = True           # the panel shows just this step
        return {'FINISHED'}


class NPROTO_OT_history_toggle_hide(bpy.types.Operator):
    """Show / hide this object in the viewport"""
    bl_idname = "nproto.history_toggle_hide"
    bl_label = "Toggle Visibility"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty()

    def execute(self, context):
        obj = bpy.data.objects.get(self.name)
        if obj is None:
            return {'CANCELLED'}
        features.set_shown(context, obj, not features.is_shown(obj))
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def step_label(e):
    from . import steps
    return steps.LABEL.get(e["op"], lambda p: e["op"].replace("_", " ").title())(e["params"])


def draw_steps(box, obj, only=None, bare=False):
    """The object's history steps; only=k: just step k, open (step focus);
    bare: only its settings, no row of its own (the step view)."""
    try:
        entries = features.stack(obj)
    except RuntimeError as exc:
        box.label(text=str(exc), icon='ERROR')
        return
    if not entries:
        return
    pg = obj.nproto_step
    col = box.column(align=True)
    if only is None:
        col.label(text="History steps (applied in order)", icon='SORTSIZE')
    for k, e in enumerate(entries):
        if only is not None and k != only:
            continue
        off = e["params"].get("_off", False)
        is_open = pg.index == k or only == k
        if not bare:
            row = col.row(align=True)
            row.operator("nproto.mute_step", text="", icon='CHECKBOX_DEHLT' if off else 'CHECKBOX_HLT',
                         emboss=False).index = k
            sub = row.row(align=True)
            sub.active = not off
            ins = ", ".join(o.name for o, _ in e["inputs"])
            sub.operator("nproto.open_step", text=f"{k + 1}. {step_label(e)}" + (f"  ({ins})" if ins else ""),
                         icon='DISCLOSURE_TRI_DOWN' if is_open else 'DISCLOSURE_TRI_RIGHT', emboss=False).index = k
            row.operator("nproto.remove_step", text="", icon='X').index = k
        if is_open:
            inner = col.column(align=True) if bare else col.box().column(align=True)
            fields = STEP_FIELDS.get(e["op"])
            rims = None
            if e["op"] == "edge_fillet":
                from . import steps as _steps
                rims = _steps.rims_of(obj, e["params"].get("mids_local", []))
            if fields:
                for name in fields:
                    if name == "end_radius" and (not pg.variable or pg.rail_type == "CHORD"):
                        continue
                    if name == "variable" and pg.rail_type == "CHORD":
                        continue
                    if name == "radius":
                        inner.prop(pg, name, text="Chord" if pg.rail_type == "CHORD" else
                                   ("Distance" if pg.shape == "CHAMFER" else "Radius"))
                        continue
                    if name == "thumbweight" and pg.shape != 'BLEND':
                        continue
                    if name == "continuity" and pg.shape != 'BLEND':
                        continue
                    if name.startswith("ca_") and not pg.class_a:
                        continue
                    if name.startswith("ca_") and rims and rims.get("only"):
                        continue                  # (see the note below: nothing for them to do)
                    if name in ("action", "continuity", "ca_continuity", "rail_type"):
                        inner.row(align=True).prop(pg, name, expand=True)
                    else:
                        inner.prop(pg, name)
            else:
                inner.label(text=STEP_NOTE.get(e["op"], "No settings"), icon='INFO')
            if rims and rims.get("rings"):
                from .ops_trim import ring_summary
                note = inner.column(align=True)
                note.label(text=ring_summary(rims["rings"]).replace("rim ", "Rim: ", 1), icon='MESH_CIRCLE')
                if pg.class_a and rims.get("only"):
                    note.label(text="Class-A settings don't apply to it", icon='INFO')
            if e["op"] == "trim":
                inner.label(text="Keep / Discard flips the trimmed side", icon='INFO')
            if e["op"] == "rebuild_surface":
                inner.label(text=f"Result: {pg.degree_u + pg.spans_u} x {pg.degree_v + pg.spans_v} CVs  "
                                 "(viewport: amber = spans, purple = CVs)", icon='MESH_GRID')
            if pg.error:
                inner.label(text=pg.error, icon='ERROR')


STEP_ICON = {"edge_fillet": 'MOD_BEVEL', "fillet_trim": 'MOD_BEVEL', "trim": 'MOD_BOOLEAN',
             "flip_normal": 'NORMALS_FACE', "surface_align": 'SNAP_EDGE', "extend_surface": 'FULLSCREEN_ENTER',
             "rebuild_surface": 'MOD_REMESH', "refit_trim": 'MOD_LATTICE'}


def _summary(obj, steps=True):
    from .feature_ui import PRETTY
    parts = []
    h = features.history(obj)
    if h is not None:
        parts.append(PRETTY.get(h["op"], h["op"].replace("prim_", "").replace("_", " ").title()))
    elif features.PROP_HIST in obj:
        parts.append("broken")
    if not steps:            # the tree lists the steps on their own rows
        if not parts and features.PROP_BASE in obj:
            parts.append("snapshot")
        return parts[0] if parts else ("frozen" if store.is_nurbs(obj) else obj.type.title())
    try:
        parts += [step_label(e).split(" ·")[0] + (" (off)" if e["params"].get("_off") else "")
                  for e in features.stack(obj)]
    except RuntimeError:
        parts.append("step lost an input")
    return " → ".join(parts) if parts else ("frozen" if store.is_nurbs(obj) else obj.type.title())


def _children(obj):
    """(object, role) pairs this object is built from, in order, unique."""
    out, seen = [], set()
    h = features.history(obj)
    if h is not None:
        for src, _ in h["inputs"]:
            if src.name not in seen:
                seen.add(src.name)
                out.append((src, "input"))
    try:
        for e in features.stack(obj):
            for src, _ in e["inputs"]:
                if src.name not in seen:
                    seen.add(src.name)
                    out.append((src, ROLE.get(e["op"], step_label(e).split(" ·")[0].lower())))
    except RuntimeError:
        pass
    return out


def _node(col, obj, depth, role, active, seen, root):
    """One tree line: name (click = make active) | role · summary | eye.
    Each object is expanded once; later mentions (e.g. two surfaces that trim
    each other) say so instead of repeating the subtree."""
    split = col.split(factor=0.5, align=True)
    left = split.row(align=True)
    if depth:
        left.separator(factor=1.2 * depth)
    icon = 'OUTLINER_OB_SURFACE' if store.is_nurbs(obj) else ('OUTLINER_OB_CURVE' if obj.type == 'CURVE' else 'OBJECT_DATA')
    op = left.operator("nproto.history_focus", text=obj.name, icon=icon, emboss=False, depress=obj == active)
    op.name, op.root = obj.name, root.name
    right = split.row(align=True)
    repeat = obj.name in seen
    info = "see above" if repeat else _summary(obj, steps=False)
    if role not in ("self", "input"):
        info = f"{role} · {info}"
    right.label(text=info)
    right.operator("nproto.history_toggle_hide", text="", icon='HIDE_OFF' if features.is_shown(obj) else 'HIDE_ON',
                   emboss=False).name = obj.name
    if repeat or depth > 6:
        return
    seen.add(obj.name)
    # the steps applied on top of the feature (fillets, trims, flips ...), in order
    try:
        entries = features.stack(obj)
    except RuntimeError:
        entries = []
        col.label(text="a history step lost an input", icon='ERROR')
    for k, e in enumerate(entries):
        sp = col.split(factor=0.5, align=True)
        lft = sp.row(align=True)
        lft.separator(factor=1.2 * (depth + 1))
        label = step_label(e)
        head, _, rest = label.partition(" · ")
        off = e["params"].get("_off")
        op = lft.operator("nproto.tree_step", text=f"{k + 1}. {head}", icon=STEP_ICON.get(e["op"], 'MODIFIER'),
                          emboss=False, depress=obj == active and obj.nproto_step.index == k)
        op.name, op.root, op.index = obj.name, root.name, k
        r = sp.row(align=True)
        r.active = not off
        r.label(text=(rest or "step") + (" (off)" if off else ""))
        mo = r.operator("nproto.mute_step", text="", emboss=False,       # on / off, where the eyes are
                        icon='CHECKBOX_DEHLT' if off else 'CHECKBOX_HLT')
        mo.index, mo.name = k, obj.name
    for src, r in _children(obj):
        _node(col, src, depth + 1, r, active, seen, root)


def dependents(obj, scene):
    from . import history
    names = history._dependents_map(scene).get(obj.name, set())
    return [bpy.data.objects[n] for n in sorted(names) if n in bpy.data.objects]


def upstream(obj, limit=200):
    """Every object this one is (directly or indirectly) built from."""
    out, todo = set(), [obj]
    while todo and len(out) < limit:
        o = todo.pop()
        for src, _ in _children(o):
            if src.name not in out:
                out.add(src.name)
                todo.append(src)
    return out


def tree_root(context, obj):
    """The object whose tree is shown: the one last navigated from, as long as
    the active object is part of its tree; otherwise the active object."""
    root = bpy.data.objects.get(context.scene.get("nproto_tree_root", ""))
    if root is not None and root != obj and obj.name in upstream(root):
        return root
    return obj


def draw_tree(layout, context, obj):
    s = context.scene.nproto_settings
    root = tree_root(context, obj)
    box = layout.box()
    row = box.row()
    row.prop(s, "show_tree", text="Construction Tree", emboss=False,
             icon='DISCLOSURE_TRI_DOWN' if s.show_tree else 'DISCLOSURE_TRI_RIGHT')
    if root != obj:
        op = row.operator("nproto.history_focus", text=f"Back to {root.name}", icon='BACK')
        op.name, op.root = root.name, root.name
    if not s.show_tree:
        return
    col = box.column(align=True)
    _node(col, root, 0, "self", obj, set(), root)
    used = dependents(obj, context.scene)
    if used:
        col.separator()
        col.label(text="Used by (rebuild when this changes):", icon='LINKED')
        for d in used:
            sp = col.split(factor=0.5, align=True)
            op = sp.row().operator("nproto.history_focus", text=d.name, icon='OUTLINER_OB_SURFACE', emboss=False)
            op.name, op.root = d.name, d.name
            r = sp.row(align=True)
            r.label(text=_summary(d))
            r.operator("nproto.history_toggle_hide", text="", icon='HIDE_OFF' if features.is_shown(d) else 'HIDE_ON',
                       emboss=False).name = d.name
    box.label(text="Click a name to edit it (it stays hidden; Shift+click shows it)", icon='INFO')


classes = (NprotoStep, NPROTO_OT_open_step, NPROTO_OT_mute_step, NPROTO_OT_history_focus,
           NPROTO_OT_history_toggle_hide, NPROTO_OT_tree_step, NPROTO_OT_step_unfocus)


def register():
    bpy.types.Object.nproto_step = PointerProperty(type=NprotoStep)


def unregister():
    del bpy.types.Object.nproto_step
