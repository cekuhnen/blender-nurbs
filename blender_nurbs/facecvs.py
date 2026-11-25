"""Face CVs: sculpt one face of a solid by its control points.

Pick a face (Face mode) and Tab (or Edit CVs): the face gets a CV cage
(cvs.py) of its own net -- a flat face a fresh cubic rectangle over its
outline (its edge CVs sit on its edges), a curved face its own surface --
refined exactly (CVs U / V, degree), so nothing changes before a CV moves.

Moving CVs rebuilds the solid live (worker job face_cvs, in the kept
worker): the face takes the new surface and every wall keeps its own
surface, extended or cut back to meet it. Edge CVs moved along a wall move
that edge, a corner CV slides the corner along its side edge, and moving
edge CVs out of a wall's plane only changes how the face leans there (the
wall still cuts it off).

Phase 1 limits: sharp edges only (a fillet along the face: Remove Fillets
first), three faces at each corner, no closed (seamed) faces. A shape that
can't be built keeps the last one that could, with the reason in the panel.

Done adds a Face CVs step (history on): the face's point and normal, the
net's layout and each moved CV's offset in its own frame (CV Tweak's
offsets), replayed on the face's fresh net when earlier steps change.
"""

import json
import time

import bpy
import numpy as np
from mathutils import Vector

from . import cvs, cvtweak, features, geom, jobs, store, subsel
from .kernel import cvnet

OP = "face_cvs"
CAGE_FACE = "nproto_cage_face"       # on a face cage: json {pt_local, n_local, flat, min_counts, plane}
CAGE_GOOD = "nproto_cage_good"       # json: the last CVs (local) the solid was built from
CAGE_ERROR = "nproto_cage_error"     # why the last CVs could not be built


def is_face_cage(cage):
    return cage is not None and CAGE_FACE in cage


def _multi_surface(obj):
    if not store.is_nurbs(obj):
        return False
    b = store.get_brep(obj)
    return b.face_count() > 1 and not cvs._same_surface(b)


def face_to_edit(obj):
    """The face of a joined body to sculpt: the one face picked on it (Face
    mode), else None."""
    if not _multi_surface(obj):
        return None
    try:
        idx = subsel.picks(obj, "FACE")
    except Exception:
        return None
    return idx[0] if len(idx) == 1 else None


# --------------------------------------------------------------------------
# Worker calls
# --------------------------------------------------------------------------

def _face_world(obj, face):
    M = geom.world_matrix(obj)
    n = (M.to_3x3().inverted().transposed() @ Vector(face["n_local"])).normalized()
    return {"face_pt": list(M @ Vector(face["pt_local"])), "face_normal": list(n), "face_flat": bool(face["flat"])}


def net_for(obj, text, face, counts=None, degree=None):
    """The face's sculpting net in obj's local space (worker job face_cvs)."""
    res, _ = jobs.run(OP, {"target": (text, jobs.xform(obj))}, [], persistent=True,
                      counts=list(counts) if counts else None, degree=degree, **_face_world(obj, face))
    net = res["net"]
    Mi = geom.world_matrix(obj).inverted()
    net["points"] = [list(Mi @ Vector(q)) for q in net["points"]]
    return net


def built_text(obj, text, face, meta, X0, X1):
    """The body text with the face on the net meta with CVs X1 (local; X0
    the unmoved ones)."""
    M = geom.world_matrix(obj)
    W = lambda X: [list(M @ Vector(q)) for q in X]
    net = {"degree_u": meta["degree_u"], "degree_v": meta["degree_v"], "count_u": meta["nu"],
           "count_v": meta["nv"], "knots_u": meta["knots_u"], "knots_v": meta["knots_v"],
           "weights": meta.get("weights") or []}
    _, breps = jobs.run(OP, {"target": (text, jobs.xform(obj))}, ["result"], persistent=True,
                        net=net, cvs=W(X1), cvs0=W(X0), into="target", **_face_world(obj, face))
    return breps["result"]


# --------------------------------------------------------------------------
# The cage
# --------------------------------------------------------------------------

def _settings(context=None):
    return getattr((context or bpy.context).window_manager, "nproto_face_cvs", None)


_syncing = [False]


def _sync_settings(cage, context=None):
    s = _settings(context)
    if s is None:
        return
    meta = json.loads(cage[cvs.CAGE_META])
    _syncing[0] = True
    try:
        s.nu, s.nv, s.degree = meta["nu"], meta["nv"], max(meta["degree_u"], meta["degree_v"])
    finally:
        _syncing[0] = False


def make_face_cage(context, obj, i, counts=None, degree=None):
    """A CV cage on face i of obj (refined to counts / degree)."""
    from . import pushpull
    p_local, _, n_world, flat = pushpull.face_frame(obj, i)
    M = geom.world_matrix(obj)
    n_local = (M.to_3x3().transposed() @ Vector(n_world)).normalized()
    face = {"pt_local": [float(x) for x in p_local], "n_local": list(n_local), "flat": bool(flat)}
    net = net_for(obj, obj[store.PROP_BREP], face, counts, degree)
    face["min_counts"], face["plane"] = net["min_counts"], net["plane"]
    face["open"] = [net["count_u"], net["count_v"], max(net["degree_u"], net["degree_v"])]
    face["layout"] = []           # Add CVs / Rebuild after moves, replayed in order on the step's rebuild
    cage = cvs.make_cage(context, obj, net)
    cage[CAGE_FACE] = json.dumps(face)
    _sync_settings(cage, context)
    return cage


def _moved(cage):
    return CAGE_GOOD in cage or CAGE_ERROR in cage


def _tol(net):
    P = np.asarray(net["points"], float)
    return 1e-6 * (float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0)


def _nets(cage, obj):
    """(unmoved net, current net) of a cage, in the surface's local space."""
    meta = json.loads(cage[cvs.CAGE_META])
    M = geom.world_matrix(obj).inverted() @ geom.world_matrix(cage)
    pts, _ = cvs._snap(cage, [tuple(M @ Vector(q)) for q in cvs.cage_points(cage)])
    base = {"degree_u": meta["degree_u"], "degree_v": meta["degree_v"], "count_u": meta["nu"], "count_v": meta["nv"],
            "knots_u": meta["knots_u"], "knots_v": meta["knots_v"], "weights": meta.get("weights") or [],
            "points": json.loads(cage[cvs.CAGE_CVS0])}
    return base, dict(base, points=[list(map(float, q)) for q in pts])


def _set_cage_net(context, cage, net, base_points):
    """Give the cage net's CVs (mesh, layout) with base_points as its unmoved CVs."""
    was_edit = cage.mode == 'EDIT'
    if was_edit:
        bpy.ops.object.mode_set(mode='OBJECT')
    old = cage.data
    cage.data = _cage_mesh(old.name, net)
    for e in list(cage.data.vertices) + list(cage.data.edges):
        e.select = False
    if old.users == 0:
        bpy.data.meshes.remove(old)
    cage[cvs.CAGE_META] = json.dumps({"degree_u": net["degree_u"], "degree_v": net["degree_v"],
                                      "nu": net["count_u"], "nv": net["count_v"], "knots_u": list(net["knots_u"]),
                                      "knots_v": list(net["knots_v"]), "weights": list(net.get("weights") or [])})
    cage[cvs.CAGE_CVS0] = json.dumps([list(map(float, q)) for q in base_points])
    if was_edit:
        bpy.ops.object.mode_set(mode='EDIT')


def rebuild(surf_obj, cage, pts, moved, context=None):
    """cvs.rebuild_from_cage for a face cage: the solid rebuilt from the
    original with the face on the moved net. A shape that can't be built
    keeps the last good one (the reason is shown in the panel)."""
    if not moved:
        cvs._restore(surf_obj, cage, context)
        for k in (CAGE_GOOD, CAGE_ERROR):
            if k in cage:
                del cage[k]
        return False
    meta = json.loads(cage[cvs.CAGE_META])
    X0 = json.loads(cage[cvs.CAGE_CVS0])
    try:
        text = built_text(surf_obj, cage[cvs.CAGE_ORIG], json.loads(cage[CAGE_FACE]), meta, X0, pts)
    except jobs.JobError as exc:
        cage[CAGE_ERROR] = str(exc)
        _redraw()
        return True
    store.DRAFT[0] = True                   # a live drag: the quick shading, the exact one once it settles
    try:
        store.set_geometry_text(surf_obj, text, context, keep_untrimmed=False)
    finally:
        store.DRAFT[0] = False
    _drafted.add(surf_obj.name)
    _last[0] = time.perf_counter()
    if not bpy.app.timers.is_registered(_settle):
        bpy.app.timers.register(_settle, first_interval=SETTLE)
    cage[CAGE_GOOD] = json.dumps([list(map(float, p)) for p in pts])
    if CAGE_ERROR in cage:
        del cage[CAGE_ERROR]
    return True


SETTLE = 0.3            # s without a live rebuild: the drag paused, write the exact shading
_drafted = set()        # objects drawn with draft shading
_last = [0.0]


def settle_now(names=None):
    """Rewrite the drafted objects' meshes with the exact crease normals."""
    for n in list(names if names is not None else _drafted):
        _drafted.discard(n)
        o = bpy.data.objects.get(n)
        if o is not None and store.is_nurbs(o):
            try:
                store.write_mesh(o, store.get_brep(o))
            except Exception as exc:
                print("NURBS Face CVs (shading):", exc)
    _redraw()


def _settle():
    wait = SETTLE - (time.perf_counter() - _last[0])
    if wait > 0.01:
        return wait
    settle_now()
    return None


def _redraw():
    try:
        for w in bpy.context.window_manager.windows:
            for a in w.screen.areas:
                if a.type in {'VIEW_3D'}:
                    a.tag_redraw()
    except Exception:
        pass


def keep_edit(surf, cage, context=None):
    """Done on a face cage: record the edit as a Face CVs step (history on).
    -> "face_step", "cleared" (history off: the old history removed) or None."""
    settle_now([surf.name])
    good = cage.get(CAGE_GOOD)
    if not good:
        return None
    if not features.history_enabled(context):
        if features.has_history(surf):
            features.clear_history(surf)
            return "cleared"
        return None
    meta = json.loads(cage[cvs.CAGE_META])
    shape = (meta["nu"], meta["nv"], 3)
    X0 = np.asarray(json.loads(cage[cvs.CAGE_CVS0]), float)
    X1 = np.asarray(json.loads(good), float)
    moves = cvtweak.offsets(X0, X1, shape)
    if not moves:
        return None
    face = json.loads(cage[CAGE_FACE])
    features.push_step(surf, OP, [], {
        "face_local": face["pt_local"], "normal_local": face["n_local"], "flat": face["flat"],
        "nu": meta["nu"], "nv": meta["nv"], "deg": [meta["degree_u"], meta["degree_v"]], "moves": moves,
        "open": face.get("open"), "layout": face.get("layout") or []},
        base_text=cage[cvs.CAGE_ORIG])
    return "face_step"


# --------------------------------------------------------------------------
# History step
# --------------------------------------------------------------------------

def apply_face_cvs(obj, text, inputs, p):
    """Face CVs on rebuild: the face's net made again from the body as it is
    now, the recorded offsets applied in its CVs' own frames."""
    face = {"pt_local": p["face_local"], "n_local": p["normal_local"], "flat": p.get("flat", False)}
    if p.get("open"):
        net = net_for(obj, text, face, counts=p["open"][:2], degree=p["open"][2])
        for op in p.get("layout") or []:
            net = cvnet.relayout(net, op, _tol(net))[0]
    else:
        net = net_for(obj, text, face, counts=[p["nu"], p["nv"]], degree=max(p["deg"]))
    if (net["count_u"], net["count_v"], net["degree_u"], net["degree_v"]) != (p["nu"], p["nv"], *p["deg"]):
        raise RuntimeError(f"Face CVs: the face's CV layout changed ({p['nu']} x {p['nv']} -> "
                           f"{net['count_u']} x {net['count_v']}): mute or delete the step")
    shape = (p["nu"], p["nv"], 3)
    X0 = np.asarray(net["points"], float)
    X1 = cvtweak.moved(X0, p["moves"], shape)
    meta = {"degree_u": net["degree_u"], "degree_v": net["degree_v"], "nu": net["count_u"], "nv": net["count_v"],
            "knots_u": net["knots_u"], "knots_v": net["knots_v"], "weights": net["weights"]}
    return built_text(obj, text, face, meta, X0.tolist(), X1.tolist())


def label(p):
    n = len(p.get("moves", []))
    return f"Face CVs · {p['nu']} x {p['nv']} · {n} CV{'s' if n != 1 else ''} moved"


def _register_step():
    from . import steps
    steps.APPLY[OP] = apply_face_cvs
    steps.LABEL[OP] = label
    features.KEEP_STEPS.add(OP)      # trimmed / natural as its input


_register_step()


# --------------------------------------------------------------------------
# CV counts (Add CVs) -- only while no CV has moved
# --------------------------------------------------------------------------

def _active_face_cage(context):
    obj = context.active_object
    cage = obj if cvs.is_cage(obj) else (cvs.cage_for(obj) if obj is not None else None)
    return cage if is_face_cage(cage) else None


def _counts_changed(self, context):
    if _syncing[0]:
        return
    cage = _active_face_cage(context)
    if cage is None or _moved(cage):
        return                  # after moves the fields wait for Add CVs / Rebuild
    if not bpy.app.timers.is_registered(_refine_later):
        bpy.app.timers.register(_refine_later, first_interval=0.05)


def _refine_later():
    """A fresh net for the new counts (from a timer: outside the field's
    update, where switching modes is not allowed)."""
    try:
        win = bpy.context.window_manager.windows[0]
        with bpy.context.temp_override(window=win):
            bpy.ops.nproto.face_cvs_refine('EXEC_DEFAULT', mode="AUTO")
    except Exception as exc:
        print("NURBS Face CVs:", exc)
    return None


def _cage_mesh(name, net):
    nu, nv = net["count_u"], net["count_v"]
    edges = [(iu * nv + iv, iu * nv + iv + 1) for iu in range(nu) for iv in range(nv - 1)]
    edges += [(iu * nv + iv, (iu + 1) * nv + iv) for iv in range(nv) for iu in range(nu - 1)]
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in net["points"]], edges, [])
    me.update()
    return me


class NPROTO_OT_face_cvs_refine(bpy.types.Operator):
    """Change the sculpted face's CV layout. Before a CV moves: a fresh net
    (a flat face keeps its even grid). After: Add CVs refines exactly (the
    shape is kept, the new CVs land where the knots go); Rebuild refits the
    shape on an even layout (it may move a little: the deviation is shown)"""
    bl_idname = "nproto.face_cvs_refine"
    bl_label = "Face CVs: Layout"
    bl_options = {'INTERNAL'}

    mode: bpy.props.EnumProperty(items=[("AUTO", "Auto", "A fresh net (only before a CV moves)"),
                                        ("ADD", "Add CVs", "More CVs, the shape kept exactly"),
                                        ("REBUILD", "Rebuild", "Refit on an even layout (may move a little)")],
                                 default="AUTO")

    def execute(self, context):
        cage = _active_face_cage(context)
        s = _settings(context)
        if cage is None or s is None:
            return {'CANCELLED'}
        obj = bpy.data.objects.get(cage[cvs.CAGE_OF])
        face = json.loads(cage[CAGE_FACE])
        if self.mode == "AUTO":
            if _moved(cage):
                return {'CANCELLED'}          # (after moves the fields wait for Add CVs / Rebuild)
            try:
                net = net_for(obj, cage[cvs.CAGE_ORIG], face, counts=[s.nu, s.nv], degree=s.degree)
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"Face CVs: {exc}")
                return {'CANCELLED'}
            _set_cage_net(context, cage, net, net["points"])
            face["open"] = [net["count_u"], net["count_v"], max(net["degree_u"], net["degree_v"])]
            face["layout"] = []
            cage[CAGE_FACE] = json.dumps(face)
            _sync_settings(cage, context)
            return {'FINISHED'}
        base, cur = _nets(cage, obj)
        op = ["add" if self.mode == "ADD" else "rebuild", int(s.nu), int(s.nv), int(s.degree)]
        if op[0] == "add" and (s.nu < cur["count_u"] or s.nv < cur["count_v"]
                               or s.degree < max(cur["degree_u"], cur["degree_v"])):
            self.report({'WARNING'}, "Add CVs only adds (U, V and Degree at least as they are) -- "
                                     "use Rebuild for fewer CVs or a lower degree")
            _sync_settings(cage, context)
            return {'CANCELLED'}
        tol = _tol(cur)
        new_base = cvnet.relayout(base, op, tol)[0]
        new_cur, dev = cvnet.relayout(cur, op, tol)
        keep = {k: cage[k] for k in (cvs.CAGE_META, cvs.CAGE_CVS0, CAGE_FACE, CAGE_GOOD) if k in cage}
        old_pts = [list(map(float, q)) for q in cur["points"]]
        _set_cage_net(context, cage, new_cur, new_base["points"])
        face["layout"] = list(face.get("layout") or []) + [op]
        cage[CAGE_FACE] = json.dumps(face)
        for k in (CAGE_GOOD, CAGE_ERROR):
            if k in cage:
                del cage[k]
        cvs.rebuild_from_cage(cage, context)
        if cage.get(CAGE_ERROR) or CAGE_GOOD not in cage:
            why = cage.get(CAGE_ERROR, "")
            _set_cage_net(context, cage, dict(cur, points=old_pts), base["points"])
            for k, v in keep.items():
                cage[k] = v
            if CAGE_ERROR in cage:
                del cage[CAGE_ERROR]
            cvs.rebuild_from_cage(cage, context)
            self.report({'WARNING'}, f"Face CVs: that layout can't be built ({why}) -- kept as it was")
            _sync_settings(cage, context)
            return {'CANCELLED'}
        cage["nproto_face_dev"] = float(dev)
        _sync_settings(cage, context)
        self.report({'INFO'}, f"Face CVs: {new_cur['count_u']} x {new_cur['count_v']} CVs" +
                    (" (exact)" if op[0] == "add" else f", the shape moved up to {dev:.3g}"))
        return {'FINISHED'}


class NProtoFaceCVsSettings(bpy.types.PropertyGroup):
    nu: bpy.props.IntProperty(name="CVs U", min=2, max=40, default=6, update=_counts_changed,
                              description="Control points along the face's U direction")
    nv: bpy.props.IntProperty(name="CVs V", min=2, max=40, default=5, update=_counts_changed,
                              description="Control points along the face's V direction")
    degree: bpy.props.IntProperty(name="Degree", min=1, max=7, default=3, update=_counts_changed,
                                  description="Degree of the sculpted face (never lower than the face's own)")


def draw_panel(lay, context, cage):
    """The Face CVs settings under the cage's line in the Active Object panel."""
    s = _settings(context)
    col = lay.column(align=True)
    col.label(text="Sculpting a face: the walls re-trim live", icon='MESH_GRID')
    if s is not None:
        row = col.row(align=True)
        row.prop(s, "nu", text="U")
        row.prop(s, "nv", text="V")
        row.prop(s, "degree")
        if _moved(cage):
            row = col.row(align=True)
            row.operator(NPROTO_OT_face_cvs_refine.bl_idname, text="Add CVs (keep shape)",
                         icon='ADD').mode = "ADD"
            row.operator(NPROTO_OT_face_cvs_refine.bl_idname, text="Rebuild (refit)",
                         icon='FILE_REFRESH').mode = "REBUILD"
            if "nproto_face_dev" in cage:
                col.label(text=f"Last rebuild moved the shape up to {cage['nproto_face_dev']:.3g}", icon='INFO')
    err = cage.get(CAGE_ERROR)
    if err:
        box = lay.box()
        box.alert = True
        box.label(text="Last move can't be built:", icon='ERROR')
        for k in range(0, min(len(err), 240), 48):
            box.label(text=err[k:k + 48])


classes = (NPROTO_OT_face_cvs_refine, NProtoFaceCVsSettings)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.WindowManager.nproto_face_cvs = bpy.props.PointerProperty(type=NProtoFaceCVsSettings)
    from . import history_ui
    history_ui.STEP_ICON[OP] = 'MESH_GRID'
    history_ui.STEP_NOTE[OP] = "Pick the face (Face mode) and Tab to sculpt it again"


def unregister():
    if hasattr(bpy.types.WindowManager, "nproto_face_cvs"):
        del bpy.types.WindowManager.nproto_face_cvs
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass
