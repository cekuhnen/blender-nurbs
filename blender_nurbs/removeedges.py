"""Remove Edges: edges of a body taken out where the faces either side lie on
one surface -- the faces merge into one and the body stays closed, its shape
unchanged (Rhino's MergeFace / MergeAllCoplanarFaces, Plasticity's Delete
Redundant Topology).

Edge mode: pick edges, X / Delete (or Remove Edges). An edge whose faces are
not on one surface (a crease, two faces that only touch tangentially, a flat
face beside a curved one) stays picked and the report says why. Remove
Redundant Edges: every edge of the selected bodies that can go, and the
vertices left where two edges meet in a straight or smooth line.

With history the removal is a step on the body (merge_faces: all, or the edges
found again by a point on each); without it the body changes in place.
Kernel side: kernel/mergefaces.py, worker job merge_faces."""

import json

import bpy
from bpy.props import BoolProperty, StringProperty
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


def _merge(target, pts_world):
    return jobs.run("merge_faces", {"target": target}, ["result"], edge_pts=pts_world, into="target")


class NPROTO_OT_remove_edges(bpy.types.Operator):
    """Remove the picked edges (Edge mode) where the faces either side lie on
    one surface: they become one face, the body stays closed and keeps its
    shape. All Edges: every edge of the selected bodies that is not needed
    (coplanar faces, pieces of one surface) and the vertices it leaves"""
    bl_idname = "nproto.remove_edges"
    bl_label = "Remove Edges"
    bl_options = {'REGISTER', 'UNDO'}

    edges: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="The edges to remove: [[body name, a point on the edge (local)], ...]")
    all_edges: BoolProperty(name="All Edges", default=False, options={'SKIP_SAVE'},
                            description="Every edge of the selected bodies whose faces lie on one surface")

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
        done, notes, warn = [], [], False
        for o, pts in items.values():
            M = geom.world_matrix(o)
            world = None if pts is None else [list(M @ Vector(p)) for p in pts]
            try:
                info, breps = _merge(o, world)
            except jobs.JobError as exc:
                notes.append(f"{o.name}: {exc}")
                warn = True
                continue
            pre = o[store.PROP_BREP]
            if not hist and features.has_history(o):
                features.clear_history(o)
                notes.append(f"{o.name}: construction history removed")
            store.set_geometry_text(o, breps["result"], context, keep_untrimmed=False)
            if hist:
                params = {"mode": "ALL"} if pts is None else {"mode": "PICKED", "edges_local": pts}
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
                msg += f", {v} vertex{'es' if v != 1 else ''}"
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
            return {'CANCELLED'}
        if self.all_edges:
            _select_only(context, done)
        return {'FINISHED'}


def apply_merge_faces(obj, text, inputs, p):
    """Remove Edges on rebuild: every edge that can go (mode ALL), or the
    edges at the stored points (those that can't any more stay)."""
    M = geom.world_matrix(obj)
    pts = None
    if p.get("mode") != "ALL":
        pts = [list(M @ Vector(q)) for q in p.get("edges_local", [])]
    try:
        _, breps = _merge((text, jobs.xform(obj)), pts)
    except jobs.JobError as exc:
        if "nothing removed" in str(exc):      # the edges are needed now (an earlier step changed): as it is
            return text
        raise
    return breps["result"]


def _label(p):
    if p.get("mode") == "ALL":
        return "Remove Edges · all redundant"
    n = len(p.get("edges_local", []))
    return f"Remove Edges · {n} picked"


def _register_step():
    from . import steps
    steps.APPLY["merge_faces"] = apply_merge_faces
    steps.LABEL["merge_faces"] = _label


_register_step()

classes = (NPROTO_OT_remove_edges,)
