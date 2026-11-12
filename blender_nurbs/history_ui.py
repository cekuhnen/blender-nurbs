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
        return parts[0] if parts else ("baked" if store.is_nurbs(obj) else obj.type.title())
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


def build_order(root, limit=400):
    """root and everything it is built from, each once, every object after
    what it is built from (oldest first): the order the model was built in."""
    out, seen = [], set()

    def visit(o, depth):
        if o.name in seen or depth > 60 or len(out) >= limit:
            return
        seen.add(o.name)
        for src, _ in _children(o):
            visit(src, depth + 1)
        out.append(o)
    visit(root, 0)
    return out


def _row_object(col, obj, active, root):
    """One feature that makes an object: name (click = edit it) | what it is | eye."""
    split = col.split(factor=0.5, align=True)
    icon = 'OUTLINER_OB_SURFACE' if store.is_nurbs(obj) else ('OUTLINER_OB_CURVE' if obj.type == 'CURVE' else 'OBJECT_DATA')
    op = split.row(align=True).operator("nproto.history_focus", text=obj.name, icon=icon, emboss=False,
                                        depress=obj == active and not obj.nproto_step.focus)
    op.name, op.root = obj.name, root.name
    right = split.row(align=True)
    right.label(text=_summary(obj, steps=False))
    right.operator("nproto.history_toggle_hide", text="", icon='HIDE_OFF' if features.is_shown(obj) else 'HIDE_ON',
                   emboss=False).name = obj.name


def _row_step(col, obj, k, e, active, root):
    """One step applied to that object (a fillet, trim ...): name | settings | eye
    (on / off: the object without it)."""
    sp = col.split(factor=0.5, align=True)
    lft = sp.row(align=True)
    lft.separator(factor=1.2)
    head, _, rest = step_label(e).partition(" · ")
    off = e["params"].get("_off")
    op = lft.operator("nproto.tree_step", text=f"{k + 1}. {head}", icon=STEP_ICON.get(e["op"], 'MODIFIER'),
                      emboss=False, depress=obj == active and obj.nproto_step.focus and obj.nproto_step.index == k)
    op.name, op.root, op.index = obj.name, root.name, k
    r = sp.row(align=True)
    r.active = not off
    r.label(text=(rest or "step") + (" (off)" if off else ""))
    mo = r.operator("nproto.mute_step", text="", emboss=False, icon='HIDE_ON' if off else 'HIDE_OFF')
    mo.index, mo.name = k, obj.name


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
    """The body's history in the order it was built (oldest first): every
    feature once, each followed by the steps applied to it. The eye shows what
    a feature made: an object (shown / hidden) or a step (on / off)."""
    s = context.scene.nproto_settings
    root = tree_root(context, obj)
    box = layout.box()
    row = box.row()
    row.prop(s, "show_tree", text="History (oldest first)", emboss=False,
             icon='DISCLOSURE_TRI_DOWN' if s.show_tree else 'DISCLOSURE_TRI_RIGHT')
    if root != obj:
        op = row.operator("nproto.history_focus", text=f"Back to {root.name}", icon='BACK')
        op.name, op.root = root.name, root.name
    if not s.show_tree:
        return
    col = box.column(align=True)
    for o in build_order(root):
        _row_object(col, o, obj, root)
        try:
            entries = features.stack(o)
        except RuntimeError:
            entries = []
            col.label(text="a history step lost an input", icon='ERROR')
        for k, e in enumerate(entries):
            _row_step(col, o, k, e, obj, root)
    box.label(text="Click a feature to edit it · Shift+click shows it", icon='INFO')


def from_feeds(obj, scene):
    """'From: a, b · Feeds: c' for a feature's panel ('' when neither)."""
    src = list(dict.fromkeys(o.name for o in features.history_inputs(obj)))
    dst = [d.name for d in dependents(obj, scene)]
    parts = []
    if src:
        parts.append("From: " + ", ".join(src[:4]) + (" ..." if len(src) > 4 else ""))
    if dst:
        parts.append("Feeds: " + ", ".join(dst[:4]) + (" ..." if len(dst) > 4 else ""))
    return "  ·  ".join(parts)


class NPROTO_OT_bake_feature(bpy.types.Operator):
    """Bake: keep the result as it is and drop the history under it (Alias' Delete
    History). A feature stops following its inputs; a step is baked into the body
    together with what comes before it (later steps stay live). Edit CVs bakes too"""
    bl_idname = "nproto.bake_feature"
    bl_label = "Bake"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty()
    step: IntProperty(default=-1)

    def execute(self, context):
        obj = bpy.data.objects.get(self.name)
        if obj is None or not features.has_history(obj):
            return {'CANCELLED'}
        try:
            freed = features.bake(obj, self.step)
        except Exception as exc:
            self.report({'ERROR'}, f"Bake failed: {exc}")
            return {'CANCELLED'}
        obj.nproto_step.focus = False
        obj.nproto_step.index = -1
        gone = features.purge_inputs(context, freed)
        what = obj.name if self.step < 0 else f"{obj.name} up to step {self.step + 1}"
        self.report({'INFO'}, f"Baked {what}" + (f"; deleted {len(gone)} hidden input(s) nothing uses now" if gone else ""))
        return {'FINISHED'}


class NPROTO_OT_delete_feature(bpy.types.Operator):
    """Remove this feature: its object goes and the objects it was built from
    come back (shown, selected) -- one step back, like undoing the command.
    Shift+click: delete the hidden objects it was built from too. When other
    features are built from it, choose: bake them first (they keep their
    shape) or delete them too"""
    bl_idname = "nproto.delete_feature"
    bl_label = "Delete Feature"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty()
    mode: EnumProperty(name="Built from it", default="BAKE",
                       items=[("BAKE", "Bake them (they keep their shape)", ""),
                              ("ALL", "Delete them too", "")])
    keep_inputs: BoolProperty(name="Bring Inputs Back", default=True,
                              description="Show and select the objects the feature was built from (off: delete the "
                                          "hidden ones nothing else uses)")

    def _users(self, context, obj):
        out, todo = [], [obj]
        while todo:
            o = todo.pop()
            for d in features.users_of(o):
                if d not in out and d is not obj:
                    out.append(d)
                    todo.append(d)
        return out

    def invoke(self, context, event):
        obj = bpy.data.objects.get(self.name)
        if obj is None:
            return {'CANCELLED'}
        self.keep_inputs = not event.shift
        if self._users(context, obj):
            return context.window_manager.invoke_props_dialog(self, width=360)
        return self.execute(context)

    def draw(self, context):
        obj = bpy.data.objects.get(self.name)
        users = self._users(context, obj) if obj else []
        self.layout.label(text=f"Built from {self.name}: " + ", ".join(u.name for u in users[:5])
                          + (" ..." if len(users) > 5 else ""))
        self.layout.prop(self, "mode", expand=True)

    def execute(self, context):
        obj = bpy.data.objects.get(self.name)
        if obj is None:
            return {'CANCELLED'}
        users = self._users(context, obj)
        baked = []                                # inputs freed by baking what is built from obj
        doomed = [obj]
        if users and self.mode == "ALL":
            doomed += users
        elif users:
            for d in features.users_of(obj):      # their history stops at this object
                st = features.stack(d)
                uses = [k for k, e in enumerate(st) if any(src is obj for src, _ in e["inputs"])]
                h = features.history(d)
                if uses:
                    baked += features.bake(d, max(uses))
                elif h is not None and any(src is obj for src, _ in h["inputs"]):
                    baked += features.bake(d, -1)
        names = [o.name for o in doomed]
        own = []                                  # what the removed features were built from
        for o in doomed:
            own += features.history_inputs(o)
        from .ops_trim import _remove_object
        for o in doomed:
            _remove_object(o)
        back, shown = [], []
        if self.keep_inputs:
            # one step back: the inputs the removed features kept hidden come back, unless
            # another Join / Thicken / ... still keeps them (a Loft or Boundary built from
            # one leaves it shown, so it doesn't count)
            alive = [o for o in dict.fromkeys(own) if _alive(o)]
            back = [o for o in alive if features.is_kept(context, o) and not features.keepers_of(o)]
            features.release_inputs(context, back)
            shown = [o for o in alive if features.is_shown(o)]
            own = []
        gone = features.purge_inputs(context, [o for o in dict.fromkeys(own + baked) if _alive(o)])
        if shown:
            from .ops import _select_only
            _select_only(context, shown)
        msg = "Deleted " + ", ".join(names)
        if back:
            msg += "; back: " + ", ".join(o.name for o in back[:6]) + (" ..." if len(back) > 6 else "")
        if gone:
            msg += f"; and {len(gone)} hidden input(s) nothing uses now"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


def _alive(o):
    try:
        return o.name in bpy.data.objects
    except ReferenceError:
        return False


classes = (NprotoStep, NPROTO_OT_open_step, NPROTO_OT_mute_step, NPROTO_OT_history_focus,
           NPROTO_OT_history_toggle_hide, NPROTO_OT_tree_step, NPROTO_OT_step_unfocus,
           NPROTO_OT_bake_feature, NPROTO_OT_delete_feature)


def register():
    bpy.types.Object.nproto_step = PointerProperty(type=NprotoStep)


def unregister():
    del bpy.types.Object.nproto_step
