"""CV editing through a temporary "cage" mesh.

Edit CVs builds a mesh whose vertices are the surface's control points (u-major,
the hull as edges) and puts the user into Blender's own Edit Mode on it, so
G / R / S, proportional editing, snapping etc. all work. A depsgraph handler
watches the cage and rebuilds the exact NURBS surface from it (same degrees,
knots and weights) on a short timer. Done keeps the result, Cancel restores
the original B-rep; both remove the cage.

Leaving Edit Mode on the cage (Tab) is the same as Done.

Construction history stays until a CV actually moves: Done after a real edit
removes it (the edit can't be replayed when the inputs change), while looking
at the CVs, Cancel, or Done without a change keep it -- and keep the B-rep
exactly as it was (no re-lift of the trim edges).
"""

import json

import bmesh
import bpy
from bpy.app.handlers import persistent
from mathutils import Vector

from . import features, geom, store
from .kernel import load as load_kernel

CAGE_OF = "nproto_cage_of"      # on the cage: name of the surface object
CAGE_META = "nproto_cage_meta"  # on the cage: degrees, knots, weights, counts
CAGE_ORIG = "nproto_cage_orig"  # on the cage: surface brep text before editing
CAGE_CVS0 = "nproto_cage_cvs0"  # on the cage: the CVs before editing (surface local space)

_dirty = set()
_busy = False
_DELAY = 0.04
_modes = {}       # cage name -> mode seen last (to catch Tab out of Edit Mode)
_finish = set()   # cages to finish (Tab out = Done)


def is_cage(obj):
    return obj is not None and obj.type == 'MESH' and CAGE_OF in obj


def cage_for(surface_obj):
    for o in bpy.data.objects:
        if is_cage(o) and o[CAGE_OF] == surface_obj.name:
            return o
    return None


def editable_reason(obj):
    """None if the object's CVs can be edited, else a message."""
    if not store.is_nurbs(obj):
        return "not a NURBS object"
    brep = store.get_brep(obj)
    if brep.face_count() != 1 and not _same_surface(brep):
        return "joined objects have several different surfaces -- Unjoin first"
    try:
        load_kernel().blender_ext.surface_cvs(brep.faces()[0].surface())
    except Exception:
        return "this surface has no control points (e.g. an exact offset surface)"
    return None


def _same_surface(brep):
    """True when all faces lie on one underlying surface (e.g. a trimmed
    revolve stored as two faces split along its seam)."""
    ext = load_kernel().blender_ext
    try:
        nets = [ext.surface_cvs(f.surface()) for f in brep.faces()]
    except Exception:
        return False
    a = nets[0]
    for b in nets[1:]:
        if (b["count_u"], b["count_v"], b["degree_u"], b["degree_v"]) != (a["count_u"], a["count_v"], a["degree_u"], a["degree_v"]):
            return False
        if any(abs(x - y) > 1e-9 for p, q in zip(a["points"], b["points"]) for x, y in zip(p, q)):
            return False
    return True


def make_cage(context, obj):
    ext = load_kernel().blender_ext
    d = ext.surface_cvs(store.get_brep(obj).faces()[0].surface())
    nu, nv = d["count_u"], d["count_v"]
    verts = [tuple(p) for p in d["points"]]
    edges = []
    for iu in range(nu):
        for iv in range(nv - 1):
            edges.append((iu * nv + iv, iu * nv + iv + 1))
    for iv in range(nv):
        for iu in range(nu - 1):
            edges.append((iu * nv + iv, (iu + 1) * nv + iv))
    me = bpy.data.meshes.new(obj.name + "_CVs")
    me.from_pydata(verts, edges, [])
    me.update()
    cage = bpy.data.objects.new(obj.name + "_CVs", me)
    for coll in obj.users_collection:
        coll.objects.link(cage)
    cage.matrix_world = geom.world_matrix(obj)
    cage.parent = None
    # no Wire / In Front: the add-on overlay draws the hull + CVs on top itself
    # (Show CVs style); Blender's orange selected-object wire would cover it
    cage.display_type = 'SOLID'
    cage.show_in_front = False
    cage.hide_render = True
    cage[CAGE_OF] = obj.name
    cage[CAGE_ORIG] = obj[store.PROP_BREP]
    cage["nproto_cage_trimmed"] = 0 if store.is_natural(obj) else 1
    if store.PROP_UNTRIM in obj:
        cage["nproto_cage_orig_untrim"] = obj[store.PROP_UNTRIM]
    if store.PROP_UNTRIM + "_natural" in obj:
        cage["nproto_cage_orig_untrim_nat"] = obj[store.PROP_UNTRIM + "_natural"]
    cage[CAGE_META] = json.dumps({"degree_u": d["degree_u"], "degree_v": d["degree_v"], "nu": nu, "nv": nv,
                                  "knots_u": d["knots_u"], "knots_v": d["knots_v"], "weights": d["weights"]})
    cage[CAGE_CVS0] = json.dumps([list(map(float, p)) for p in d["points"]])
    return cage


def _snap(cage, pts):
    """(CVs, moved): the cage's CVs (surface local space) with every CV that
    wasn't moved put back to its exact original value -- the cage mesh keeps
    32-bit floats, so even untouched CVs come back rounded -- and whether any
    CV moved."""
    raw = cage.get(CAGE_CVS0)
    if raw is None:
        return pts, True
    a = json.loads(raw)
    if len(a) != len(pts):
        return pts, True
    size = max((abs(c) for p in a for c in p), default=1.0) or 1.0
    tol = 1e-6 * size          # float32 rounding is ~6e-8 of the coordinates
    out, moved = [], False
    for p, q in zip(a, pts):
        if all(abs(x - y) <= tol for x, y in zip(p, q)):
            out.append(tuple(p))
        else:
            out.append(tuple(q))
            moved = True
    return out, moved


def _restore(surf_obj, cage, context=None):
    """Put the surface back exactly as it was before the cage (B-rep text and
    untrim data)."""
    trimmed = bool(cage.get("nproto_cage_trimmed"))
    if surf_obj.get(store.PROP_BREP) != cage[CAGE_ORIG]:
        store.set_geometry_text(surf_obj, cage[CAGE_ORIG], context, keep_untrimmed=False)
    surf_obj[store.PROP_NATURAL] = 0 if trimmed else 1
    if "nproto_cage_orig_untrim" in cage:
        surf_obj[store.PROP_UNTRIM] = cage["nproto_cage_orig_untrim"]
    elif store.PROP_UNTRIM in surf_obj:
        del surf_obj[store.PROP_UNTRIM]
    nat = store.PROP_UNTRIM + "_natural"
    if "nproto_cage_orig_untrim_nat" in cage:
        surf_obj[nat] = cage["nproto_cage_orig_untrim_nat"]
    elif nat in surf_obj:
        del surf_obj[nat]


def cage_points(cage):
    """CV positions in the cage's local space (edit-mode aware)."""
    if cage.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(cage.data)
        bm.verts.ensure_lookup_table()
        return [tuple(v.co) for v in bm.verts]
    return [tuple(v.co) for v in cage.data.vertices]


def rebuild_from_cage(cage, context=None):
    """Rebuild the surface from the cage's CVs. Returns True when the CVs
    moved; when they didn't, the surface is put back exactly as it was."""
    surf_obj = bpy.data.objects.get(cage[CAGE_OF])
    if surf_obj is None:
        return False
    meta = json.loads(cage[CAGE_META])
    pts = cage_points(cage)
    if len(pts) != meta["nu"] * meta["nv"]:
        raise RuntimeError("the CV cage must keep its vertex count (don't add or delete vertices)")
    # cage and surface may have different transforms: bring CVs into the surface's local space
    M = geom.world_matrix(surf_obj).inverted() @ geom.world_matrix(cage)
    pts, moved = _snap(cage, [tuple(M @ Vector(p)) for p in pts])
    if not moved:
        _restore(surf_obj, cage, context)
        return False
    if cage.get("nproto_cage_trimmed"):
        # trims stay fixed in UV: rebuild from the ORIGINAL trimmed brep each time,
        # in the worker process (deep kernel edit; a crash must not take Blender down)
        from . import jobs
        with _OriginalBrep(surf_obj, cage):
            _, breps = jobs.run("retarget", {"target": surf_obj}, ["result", "untrimmed"],
                                meta=meta, cvs=[list(p) for p in pts])
        store.set_geometry_text(surf_obj, breps["result"], context, keep_untrimmed=False)
        surf_obj[store.PROP_UNTRIM] = breps["untrimmed"]
        surf_obj[store.PROP_UNTRIM + "_natural"] = 1
        return True
    ext = load_kernel().blender_ext
    s = ext.surface_from_cvs(meta["degree_u"], meta["degree_v"], pts, meta["knots_u"], meta["knots_v"],
                             meta["weights"] or None)
    store.set_geometry(surf_obj, s, context)
    return True


class _OriginalBrep:
    """Temporarily present the pre-edit brep to jobs.run (which reads obj[PROP_BREP])."""

    def __init__(self, obj, cage):
        self.obj, self.cage = obj, cage

    def __enter__(self):
        self.current = self.obj[store.PROP_BREP]
        self.obj[store.PROP_BREP] = self.cage[CAGE_ORIG]

    def __exit__(self, *exc):
        self.obj[store.PROP_BREP] = self.current
        return False


@persistent
def _on_depsgraph(scene, depsgraph):
    if _busy:
        return
    for u in depsgraph.updates:
        idb = u.id
        obj = None
        if isinstance(idb, bpy.types.Object):
            obj = idb.original
        elif isinstance(idb, bpy.types.Mesh):
            for o in scene.objects:
                if o.data == idb.original and is_cage(o):
                    obj = o
                    break
        if obj is not None and is_cage(obj) and (u.is_updated_geometry or u.is_updated_transform):
            _dirty.add(obj.name)
    # leaving Edit Mode on a cage (Tab) works like Done: keep the edit, remove the cage
    for o in scene.objects:
        if CAGE_OF in o and o.type == 'MESH':
            prev = _modes.get(o.name)
            _modes[o.name] = o.mode
            if prev == 'EDIT' and o.mode == 'OBJECT':
                _finish.add(o.name)
    if _dirty and not bpy.app.timers.is_registered(_process):
        bpy.app.timers.register(_process, first_interval=_DELAY)
    if _finish and not bpy.app.timers.is_registered(_finish_cages):
        bpy.app.timers.register(_finish_cages, first_interval=_DELAY)


def finish_cage(cage, view_layer=None):
    """Keep the CV edit and remove the cage (no operator context needed)."""
    surf_obj = bpy.data.objects.get(cage[CAGE_OF])
    if surf_obj is not None:
        if rebuild_from_cage(cage) and features.has_history(surf_obj):
            features.clear_history(surf_obj)
            print(f"NURBS: {surf_obj.name}: construction history removed (CVs moved)")
    _modes.pop(cage.name, None)
    me = cage.data
    bpy.data.objects.remove(cage)
    if me.users == 0:
        bpy.data.meshes.remove(me)
    if surf_obj is not None:
        vl = view_layer or bpy.context.view_layer
        for o in vl.objects:
            if o.select_get():
                o.select_set(False)
        try:
            surf_obj.select_set(True)
        except RuntimeError:
            pass
        vl.objects.active = surf_obj
    return surf_obj


def _finish_cages():
    global _busy
    names = list(_finish)
    _finish.clear()
    done = False
    _busy = True
    try:
        for n in names:
            cage = bpy.data.objects.get(n)
            if cage is None or not is_cage(cage) or cage.mode != 'OBJECT':
                continue
            try:
                finish_cage(cage)
                done = True
            except Exception as exc:
                print("NURBS CV edit (finish):", exc)
    finally:
        _busy = False
    if done:
        try:
            bpy.ops.ed.undo_push(message="Edit CVs Done")
        except Exception:
            pass
    return None


def _process():
    global _busy
    names = list(_dirty)
    _dirty.clear()
    _busy = True
    try:
        for n in names:
            cage = bpy.data.objects.get(n)
            if cage is None or not is_cage(cage):
                continue
            try:
                rebuild_from_cage(cage)
            except Exception as exc:
                print("NURBS CV edit:", exc)
    finally:
        _busy = False
    return None


def _end(context, cage, keep):
    """Remove the cage, keeping the edit (keep) or restoring the surface.
    Returns (surface object, True when its history was removed)."""
    surf_obj = bpy.data.objects.get(cage[CAGE_OF])
    if context.object is not None and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    dropped = False
    if surf_obj is not None:
        if keep and rebuild_from_cage(cage, context):
            if features.has_history(surf_obj):
                features.clear_history(surf_obj)
                dropped = True
        elif not keep:
            _restore(surf_obj, cage, context)
    _modes.pop(cage.name, None)
    _finish.discard(cage.name)
    me = cage.data
    bpy.data.objects.remove(cage)
    if me.users == 0:
        bpy.data.meshes.remove(me)
    if surf_obj is not None:
        for o in context.selected_objects:
            o.select_set(False)
        surf_obj.select_set(True)
        context.view_layer.objects.active = surf_obj
    return surf_obj, dropped


def _active_cage(context):
    obj = context.active_object
    if is_cage(obj):
        return obj
    if obj is not None:
        return cage_for(obj)
    return None


class NPROTO_OT_edit_cvs(bpy.types.Operator):
    """Edit the surface's control points with Blender's own Edit Mode tools
    (a temporary CV cage; the surface updates live)"""
    bl_idname = "nproto.edit_cvs"
    bl_label = "Edit CVs"
    bl_options = {'UNDO'}   # no redo panel: re-running a mode switch via redo is unsafe

    def execute(self, context):
        if load_kernel() is None:
            self.report({'ERROR'}, "USD BRep kernel not available")
            return {'CANCELLED'}
        obj = context.active_object
        if is_cage(obj):
            cage = obj
        else:
            if obj is None:
                self.report({'WARNING'}, "Select a NURBS surface")
                return {'CANCELLED'}
            cage = cage_for(obj)
            if cage is None:
                why = editable_reason(obj)
                if why:
                    self.report({'WARNING'}, f"{obj.name}: {why}")
                    return {'CANCELLED'}
                if features.has_history(obj):
                    self.report({'INFO'}, f"{obj.name}: moving a CV removes its construction history "
                                          "(Cancel, or leaving without a change, keeps it)")
                if context.object is not None and context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                cage = make_cage(context, obj)
        for o in context.selected_objects:
            o.select_set(False)
        cage.hide_set(False)
        cage.select_set(True)
        context.view_layer.objects.active = cage
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            context.tool_settings.mesh_select_mode = (True, False, False)
        except Exception:
            pass
        return {'FINISHED'}


class NPROTO_OT_cvs_done(bpy.types.Operator):
    """Keep the CV edits and remove the CV cage"""
    bl_idname = "nproto.cvs_done"
    bl_label = "Done"
    bl_options = {'UNDO'}   # no redo panel: re-running a mode switch via redo is unsafe

    def execute(self, context):
        cage = _active_cage(context)
        if cage is None:
            return {'CANCELLED'}
        surf_obj, dropped = _end(context, cage, keep=True)
        if dropped:
            self.report({'INFO'}, f"{surf_obj.name}: construction history removed (CVs moved)")
        return {'FINISHED'}


class NPROTO_OT_cvs_cancel(bpy.types.Operator):
    """Discard the CV edits (restore the surface) and remove the CV cage"""
    bl_idname = "nproto.cvs_cancel"
    bl_label = "Cancel"
    bl_options = {'UNDO'}   # no redo panel: re-running a mode switch via redo is unsafe

    def execute(self, context):
        cage = _active_cage(context)
        if cage is None:
            return {'CANCELLED'}
        _end(context, cage, keep=False)
        return {'FINISHED'}


class NPROTO_OT_tab_edit(bpy.types.Operator):
    """Tab on a NURBS surface: edit its CVs (Tab again = Done).
    Other objects keep Blender's normal Tab"""
    bl_idname = "nproto.tab_edit"
    bl_label = "Edit CVs (Tab)"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        obj = context.view_layer.objects.active
        return context.mode == 'OBJECT' and obj is not None and store.is_nurbs(obj) and not is_cage(obj)

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        obj = context.view_layer.objects.active
        if load_kernel() is None:
            return {'PASS_THROUGH'}
        if cage_for(obj) is None:
            why = editable_reason(obj)
            if why:
                # never drop into Edit Mode on the tessellation mesh: it is display only
                self.report({'WARNING'}, f"{obj.name}: {why}")
                return {'CANCELLED'}
        return bpy.ops.nproto.edit_cvs('EXEC_DEFAULT')


classes = (NPROTO_OT_edit_cvs, NPROTO_OT_cvs_done, NPROTO_OT_cvs_cancel, NPROTO_OT_tab_edit)
_keymaps = []


def register():
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is not None:
        km = kc.keymaps.new(name="Object Mode", space_type='EMPTY')
        _keymaps.append((km, km.keymap_items.new(NPROTO_OT_tab_edit.bl_idname, 'TAB', 'PRESS')))
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    for km, kmi in _keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _keymaps.clear()
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    if bpy.app.timers.is_registered(_process):
        bpy.app.timers.unregister(_process)
    if bpy.app.timers.is_registered(_finish_cages):
        bpy.app.timers.unregister(_finish_cages)
