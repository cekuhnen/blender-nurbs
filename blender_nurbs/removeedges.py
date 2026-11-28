"""Remove Edges: edges of a body taken out where the faces either side lie on
one surface -- the faces merge into one and the body stays closed, its shape
unchanged (Rhino's MergeFace / MergeAllCoplanarFaces, Plasticity's Delete
Redundant Topology).

Edge mode: pick edges, X / Delete (or Remove Edges). An edge whose faces are
not on one surface (a crease, two faces that only touch tangentially, a flat
face beside a curved one) stays picked and the report says why. Flat
Tolerance (F9): flat faces within that angle of one plane merge too, refitted
onto one plane through them (kernel/planerefit.py: their corners move a
little, the faces round them follow). Fit Tolerance (F9): faces that meet
tangentially on different surfaces merge onto one NURBS surface fitted
through them within that distance (kernel/surfjoin.py). Remove Redundant
Edges: every edge of the selected bodies that can go, and the
vertices left where two edges meet in a straight or smooth line.

Remove Faces (Face mode, Shift+X / Shift+Delete): the picked faces go and the
faces round each gap are extended until they meet, so the body stays closed
(a hole, a boss, a pocket, a chamfer, a sliver). A picked fillet goes by
Remove Fillets' rules. X / Delete (Delete Faces) still opens the body.

With history each is a step on the body (merge_faces: all, or the edges found
again by a point on each; remove_faces: a point on each face); without it the
body changes in place. Kernel side: kernel/mergefaces.py and
kernel/healfaces.py, worker jobs merge_faces and remove_faces."""

import json

import bpy
import math

from bpy.props import BoolProperty, FloatProperty, StringProperty
from mathutils import Vector

from . import features, geom, jobs, store, subsel
from .ops import _kernel_ok, _select_only


def edge_point(obj, i):
    """A point on display edge i of obj (local space): the middle sample of its
    polyline, which lies on the edge."""
    P = subsel._edge_polys(obj)[i]
    if P is None:
        return None
    return [float(x) for x in P[len(P) // 2]]


def _merge(target, pts_world, flat_deg=0.0, fit_tol=0.0):
    return jobs.run("merge_faces", {"target": target}, ["result"], edge_pts=pts_world, flat_deg=flat_deg,
                    fit_tol=fit_tol, into="target")


class _RemoveEdges:
    """Remove the picked edges (Edge mode) where the faces either side lie on
    one surface: they become one face, the body stays closed and keeps its
    shape. All Edges: every edge of the selected bodies that is not needed
    (coplanar faces, pieces of one surface) and the vertices it leaves"""
    edges: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="The edges to remove: [[body name, a point on the edge (local)], ...]")
    all_edges: BoolProperty(name="All Edges", default=False, options={'SKIP_SAVE'},
                            description="Every edge of the selected bodies whose faces lie on one surface")
    flat_tolerance: FloatProperty(name="Flat Tolerance", default=0.0, min=0.0, max=math.radians(10.0),
                                  subtype='ANGLE', precision=3,
                                  description="Flat faces within this angle of one plane merge too: they are "
                                              "refitted onto one plane fitted through them, so their corners move "
                                              "a little (0: exact merges only, the shape never changes)")
    fit_tolerance: FloatProperty(name="Fit Tolerance", default=0.0, min=0.0, soft_max=1.0, unit='LENGTH',
                                 precision=4, step=0.1,
                                 description="Faces that meet tangentially but lie on different surfaces (split "
                                             "patches, a flat side running into a round end) merge too: refitted "
                                             "onto one new NURBS surface within this distance of them (0: exact "
                                             "merges only)")

    @classmethod
    def description(cls, context, props):
        if props and props.all_edges:
            return ("Remove every edge of the selected bodies that is not needed: faces on one plane or "
                    "pieces of one surface become one face, and vertices left on straight or smooth edges go")
        return cls.__doc__

    def _items(self, context):
        """{body name: (body, [local points] or None for all)}"""
        if self.all_edges:
            return {o.name: (o, None) for o in context.selected_objects if store.is_nurbs(o)}
        out = {}
        if self.edges:
            for name, p in json.loads(self.edges):
                o = bpy.data.objects.get(name)
                if o is None:
                    raise ValueError(f"{name} not found")
                out.setdefault(name, (o, []))[1].append(list(map(float, p)))
            return out
        for o in context.selected_objects:
            if not store.is_nurbs(o):
                continue
            pts = [p for p in (edge_point(o, i) for i in subsel.picks(o, "EDGE")) if p is not None]
            if pts:
                out[o.name] = (o, pts)
        return out

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            items = self._items(context)
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Remove Edges: {exc}")
            return {'CANCELLED'}
        if not items:
            self.report({'WARNING'}, "Remove Edges: select a body" if self.all_edges else
                        "Remove Edges: pick edges first (Edge mode), or use Remove Redundant Edges")
            return {'CANCELLED'}
        hist = features.history_enabled(context)
        done, notes, warn, adjustable = [], [], False, False
        for o, pts in items.values():
            M = geom.world_matrix(o)
            world = None if pts is None else [list(M @ Vector(p)) for p in pts]
            try:
                info, breps = _merge(o, world, math.degrees(self.flat_tolerance), self.fit_tolerance)
            except jobs.JobError as exc:
                notes.append(f"{o.name}: {exc}")
                warn = True
                adjustable |= "Flat Tolerance" in str(exc) or "Fit Tolerance" in str(exc)
                continue
            pre = o[store.PROP_BREP]
            if not hist and features.has_history(o):
                features.clear_history(o)
                notes.append(f"{o.name}: construction history removed")
            store.set_geometry_text(o, breps["result"], context, keep_untrimmed=False)
            if hist:
                params = {"mode": "ALL"} if pts is None else {"mode": "PICKED", "edges_local": pts}
                if self.flat_tolerance > 0:
                    params["flat_tolerance"] = float(self.flat_tolerance)
                if self.fit_tolerance > 0:
                    params["fit_tolerance"] = float(self.fit_tolerance)
                features.push_step(o, "merge_faces", [], params, base_text=pre)
            refused = info.get("refused", [])
            if pts is not None:
                # the edges that stayed stay picked; the removed ones are gone
                M_inv = M.inverted()
                keep = [list(M_inv @ Vector(r["point"])) for r in refused if r.get("point")]
                subsel.set_picks(o, "EDGE", [])
                if keep:
                    idx = [subsel._locate(o, "EDGE", -1, p, subsel._size(o)) for p in keep]
                    subsel.set_picks(o, "EDGE", [i for i in idx if i is not None])
            done.append(o)
            n, v = int(info.get("removed", 0)), int(info.get("vertices", 0))
            msg = f"{o.name}: {n} edge{'s' if n != 1 else ''} removed"
            if v:
                msg += f", {v} {'vertices' if v != 1 else 'vertex'}"
            if info.get("refitted"):
                msg += f"; {info['refitted']} by a plane refit, corners moved up to {info.get('moved', 0.0):.3g}"
            if info.get("fitted"):
                msg += (f"; {info['fitted']} by a surface refit, within {info.get('deviation', 0.0):.3g} "
                        f"of the faces")
            msg += f" ({info.get('faces_before', '?')} → {info.get('faces', '?')} faces)"
            if refused:
                whys = list(dict.fromkeys(r["why"] for r in refused))
                msg += f"; {len(refused)} kept: " + "; ".join(whys)
                warn = True
            if info.get("failed"):
                msg += "; " + "; ".join(dict.fromkeys(f["why"] for f in info["failed"]))
                warn = True
            notes.append(msg)
        for m in notes:
            print("NURBS remove edges:", m)
        if notes:
            self.report({'WARNING'} if warn else {'INFO'}, " · ".join(notes))
        if not done:
            if adjustable:
                # nothing changed, but a Flat Tolerance would merge them: finish, so F9 can set it
                which = " / ".join(k for k in ("Flat Tolerance", "Fit Tolerance") if k in " ".join(notes))
                self.report({'WARNING'}, " · ".join(notes) + f" -- F9: raise {which}")
                return {'FINISHED'}
            return {'CANCELLED'}
        if self.all_edges:
            _select_only(context, done)
        return {'FINISHED'}


def _heal(target, pts_world):
    return jobs.run("remove_faces", {"target": target}, ["result"], face_pts=pts_world, into="target")


class _RemoveFaces:
    """Remove the picked faces (Face mode) and heal the gap: the faces round it
    are extended until they meet, so the body stays closed (a hole's wall:
    the plate is whole again; a boss; a chamfer or a sliver: a sharp edge).
    Shift+X / Shift+Delete in Face mode. Delete Faces (X) opens the body
    instead"""
    faces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="The faces to remove: [[body name, a point on the face (local)], ...]")

    def _items(self, context):
        """{body name: (body, [local points])}"""
        out = {}
        if self.faces:
            for name, p in json.loads(self.faces):
                o = bpy.data.objects.get(name)
                if o is None:
                    raise ValueError(f"{name} not found")
                out.setdefault(name, (o, []))[1].append(list(map(float, p)))
            return out
        from .pushpull import face_frame
        for o in context.selected_objects:
            if not store.is_nurbs(o):
                continue
            pts = [[float(x) for x in face_frame(o, i)[0]] for i in subsel.picks(o, "FACE")]
            if pts:
                out[o.name] = (o, pts)
        return out

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            items = self._items(context)
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Remove Faces: {exc}")
            return {'CANCELLED'}
        if not items:
            self.report({'WARNING'}, "Remove Faces: pick faces first (Face mode)")
            return {'CANCELLED'}
        hist = features.history_enabled(context)
        done, notes, warn = [], [], False
        for o, pts in items.values():
            M = geom.world_matrix(o)
            try:
                info, breps = _heal(o, [list(M @ Vector(p)) for p in pts])
            except jobs.JobError as exc:
                notes.append(f"{o.name}: {exc} -- Delete Faces (X) opens the body there instead")
                warn = True
                continue
            pre = o[store.PROP_BREP]
            if not hist and features.has_history(o):
                features.clear_history(o)
                notes.append(f"{o.name}: construction history removed")
            store.set_geometry_text(o, breps["result"], context, keep_untrimmed=False)
            if hist:
                features.push_step(o, "remove_faces", [], {"faces_local": pts}, base_text=pre)
            subsel.set_picks(o, "FACE", [])
            done.append(o)
            n = int(info.get("removed", 0))
            msg = f"{o.name}: {n} face{'s' if n != 1 else ''} removed and healed"
            msg += f" ({info.get('faces_before', '?')} → {info.get('faces', '?')} faces)"
            if info.get("added"):
                msg += f", {info['added']} enclosed face{'s' if info['added'] != 1 else ''} taken along"
            if info.get("failed"):
                msg += "; kept: " + "; ".join(dict.fromkeys(f["why"] for f in info["failed"]))
                warn = True
            notes.append(msg)
        for m in notes:
            print("NURBS remove faces:", m)
        if notes:
            self.report({'WARNING'} if warn else {'INFO'}, " · ".join(notes))
        return {'FINISHED'} if done else {'CANCELLED'}


def apply_remove_faces(obj, text, inputs, p):
    """Remove Faces on rebuild: the faces at the stored points again."""
    M = geom.world_matrix(obj)
    pts = [list(M @ Vector(q)) for q in p.get("faces_local", [])]
    _, breps = _heal((text, jobs.xform(obj)), pts)
    return breps["result"]


def apply_merge_faces(obj, text, inputs, p):
    """Remove Edges on rebuild: every edge that can go (mode ALL), or the
    edges at the stored points (those that can't any more stay)."""
    M = geom.world_matrix(obj)
    pts = None
    if p.get("mode") != "ALL":
        pts = [list(M @ Vector(q)) for q in p.get("edges_local", [])]
    try:
        _, breps = _merge((text, jobs.xform(obj)), pts, math.degrees(float(p.get("flat_tolerance", 0.0))),
                          float(p.get("fit_tolerance", 0.0)))
    except jobs.JobError as exc:
        if "nothing removed" in str(exc):      # the edges are needed now (an earlier step changed): as it is
            return text
        raise
    return breps["result"]


def _label(p):
    from . import steps
    flat, fit = float(p.get("flat_tolerance", 0.0)), float(p.get("fit_tolerance", 0.0))
    tail = f" · flat {math.degrees(flat):.3g}°" if flat > 0 else ""
    tail += f" · fit {steps.length_text(fit)}" if fit > 0 else ""
    if p.get("mode") == "ALL":
        return "Remove Edges · all redundant" + tail
    n = len(p.get("edges_local", []))
    return f"Remove Edges · {n} picked" + tail


def _register_step():
    from . import steps
    steps.APPLY["merge_faces"] = apply_merge_faces
    steps.LABEL["merge_faces"] = _label
    steps.APPLY["remove_faces"] = apply_remove_faces
    steps.LABEL["remove_faces"] = lambda p: f"Remove Faces · {len(p.get('faces_local', []))} healed"


_register_step()

# -- X / Delete with picks: keymap items (not the picker's modal handler), so the
#    operator is registered and F9 adjusts it (Flat Tolerance). Each runs only with
#    the picks it takes; otherwise Blender's own X / Delete (delete the object) runs.

def _picked(context, kind):
    if not subsel.kinds(context) or context.mode != 'OBJECT':
        return False
    return any(store.is_nurbs(o) and subsel.picks(o, kind) for o in context.selected_objects)


# (operators share code through mixins: registering a subclass of a registered
#  operator class breaks the parent's execute)
class NPROTO_OT_remove_edges(_RemoveEdges, bpy.types.Operator):
    __doc__ = _RemoveEdges.__doc__
    bl_idname = "nproto.remove_edges"
    bl_label = "Remove Edges"
    bl_options = {'REGISTER', 'UNDO'}


class NPROTO_OT_remove_faces(_RemoveFaces, bpy.types.Operator):
    __doc__ = _RemoveFaces.__doc__
    bl_idname = "nproto.remove_faces"
    bl_label = "Remove Faces"
    bl_options = {'REGISTER', 'UNDO'}


class NPROTO_OT_remove_edges_key(_RemoveEdges, bpy.types.Operator):
    """Remove the picked edges where their faces merge into one (X / Delete in Edge mode)"""
    bl_idname = "nproto.remove_edges_key"
    bl_label = "Remove Edges"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return _picked(context, "EDGE") and not _picked(context, "FACE")


class NPROTO_OT_remove_faces_key(_RemoveFaces, bpy.types.Operator):
    """Remove the picked faces and heal the gap (Shift+X / Shift+Delete in Face mode)"""
    bl_idname = "nproto.remove_faces_key"
    bl_label = "Remove Faces"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return _picked(context, "FACE")


def _delete_faces_key():
    from .ops_trim import NPROTO_OT_delete_faces

    class NPROTO_OT_delete_faces_key(bpy.types.Operator):
        """Delete the picked faces; the body opens (X / Delete in Face mode)"""
        bl_idname = "nproto.delete_faces_key"
        bl_label = "Delete Faces"
        bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}
        execute = NPROTO_OT_delete_faces.__dict__["execute"]     # (not a subclass: see above)

        @classmethod
        def poll(cls, context):
            return _picked(context, "FACE")
    return NPROTO_OT_delete_faces_key


classes = (NPROTO_OT_remove_edges, NPROTO_OT_remove_faces, NPROTO_OT_remove_edges_key,
           NPROTO_OT_remove_faces_key, _delete_faces_key())
_keymaps = []


def register():
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is None:
        return
    km = kc.keymaps.new(name="Object Mode", space_type='EMPTY')
    for key in ('X', 'DEL'):
        _keymaps.append((km, km.keymap_items.new("nproto.delete_faces_key", key, 'PRESS')))
        _keymaps.append((km, km.keymap_items.new("nproto.remove_edges_key", key, 'PRESS')))
        _keymaps.append((km, km.keymap_items.new("nproto.remove_faces_key", key, 'PRESS', shift=True)))


def unregister():
    for km, kmi in _keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _keymaps.clear()
