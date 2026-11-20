"""Push / Pull: a face of a closed solid moved along its normal, as in
Shapr3D and Plasticity. Out (a positive distance) adds material, in removes
it. A flat face moves: the prism it sweeps is joined to or cut from the body
(the worker's push_pull job, which falls back to extruding the face and
stitching it when the boolean can't). A curved face is offset (Offset Face):
the slab between it and its offset is joined or cut, so a hole or boss
changes radius. The face comes from Face mode (the picked faces of the
selected bodies) or a click on a face; then an arrow stands on it, from 0:
drag it out or through the face, type a value, click away or Enter.

With history the move is a step on the body (push_pull, replayed on rebuilds,
the faces found again by a point on each); without it the body changes in
place."""

import json

import bpy
import numpy as np
from bpy.props import FloatProperty, StringProperty
from mathutils import Vector

from . import draw, features, geom, hints, jobs, store, subsel
from .ops import AskValue, _is_solid, _kernel_ok, _select_only
from .valuedrag import NAV, ValueDrag, ruled_preview


def _uvxy(uv):
    return (uv.x, uv.y) if hasattr(uv, "x") else (float(uv[0]), float(uv[1]))


def _face_index(obj, p_local):
    faces = store.get_brep(obj).faces()
    return min(range(len(faces)), key=lambda k: faces[k].closest_point(tuple(map(float, p_local)))[1])


def face_frame(obj, i):
    """(point inside face i (local), world point, world unit outward normal,
    flat?) of a NURBS body's face."""
    sm, _ = store._k()
    f = store.get_brep(obj).faces()[i]
    ip = f.interior_point()
    p = np.array([ip.x, ip.y, ip.z] if hasattr(ip, "x") else ip, float)
    _, uvs, _ = sm.surface_closest_point(f.surface(), tuple(map(float, p)))
    n = np.array(f.outward_normal_at_uv(*_uvxy(uvs[0])), float)
    M = geom.world_matrix(obj)
    nw = np.array(M.to_3x3().inverted().transposed() @ Vector(n), float)
    nw /= np.linalg.norm(nw) or 1.0
    return p, np.array(M @ Vector(p), float), nw, bool(f.surface().is_planar())


def normals_at(obj, i, P):
    """World unit outward normals of face i at the local points P (on it or near)."""
    sm, _ = store._k()
    f = store.get_brep(obj).faces()[i]
    srf = f.surface()
    N = []
    for q in np.asarray(P, float):
        _, uvs, _ = sm.surface_closest_point(srf, tuple(map(float, q)))
        N.append(f.outward_normal_at_uv(*_uvxy(uvs[0])))
    N = np.asarray(N, float).reshape(-1, 3)
    Nw = N @ np.array(geom.world_matrix(obj).to_3x3().inverted().transposed()).T
    return Nw / np.maximum(np.linalg.norm(Nw, axis=1), 1e-300)[:, None]


def _world(M, P):
    return (np.c_[P, np.ones(len(P))] @ M.T)[:, :3]


def _faces_json(items):
    return json.dumps([[o.name, [float(x) for x in p]] for o, p in items])


class _FacePicking:
    """Clicking faces in the view (Push / Pull, Replace Face): what is under
    the mouse, lit up."""
    MAX_FILL = 3000             # a curved face with more tessellation points than this: outlines only

    def _hit(self, context, event):
        region, _ = ValueDrag._win(context)
        if region is None:
            return None
        xy = (event.mouse_x - region.x, event.mouse_y - region.y)
        if not (0 <= xy[0] < region.width and 0 <= xy[1] < region.height):
            return None
        with context.temp_override(region=region):
            try:
                return subsel.face_under(bpy.context, xy)
            except Exception as exc:
                print("NURBS push/pull pick:", exc)
                return None

    def _show_face(self, hit, solids=True):
        """Outline and fill the face under the mouse (solids: of solids only;
        False: of open surfaces only)."""
        if hit == getattr(self, "_hover", None):
            return
        self._hover = hit
        pv = None
        if hit is not None:
            o, i = hit
            try:
                if _is_solid(o) == solids:
                    fr = face_frame(o, i)
                    pts, tris, polys = subsel.face_tris(o, i)
                    M = np.array(geom.world_matrix(o))
                    lift = 2e-3 * float(np.linalg.norm(np.ptp(np.vstack(polys), axis=0)) or 1.0)
                    if fr[3]:                   # off the face: no z-fight
                        W = lambda P: _world(M, P) + lift * fr[2]
                    else:
                        W = lambda P: _world(M, P) + lift * normals_at(o, i, P)
                    pv = {"lines": [W(P) for P in polys],
                          "tris": W(pts)[tris].reshape(-1, 3) if len(tris) and len(pts) <= self.MAX_FILL else
                          np.zeros((0, 3))}
            except Exception as exc:
                print("NURBS push/pull hover:", exc)
        draw.pick_overlay["preview"] = pv

    def _end_pick(self, context):
        draw.pick_overlay["preview"] = None
        hints.clear(context)
        context.area.tag_redraw()



class NPROTO_OT_push_pull(_FacePicking, ValueDrag, AskValue, bpy.types.Operator):
    """Push or pull faces of a solid along their normals: out adds material,
    in removes it; a curved face is offset (a hole or boss changes radius).
    Faces picked in Face mode, or click a face; then drag the arrow (through
    the face to push), type a value, click away or Enter"""
    bl_idname = "nproto.push_pull"
    bl_label = "Push / Pull"
    bl_options = {'REGISTER', 'UNDO'}

    faces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="The faces: [[body name, a point on the face (local)], ...]")
    distance: FloatProperty(name="Distance", default=0.0, unit='LENGTH',
                            description="Along each face's outward normal: out (positive) adds material, "
                                        "in (negative) removes it")

    fit_view = "shrink"
    start_value = 0.0           # the arrow on the face: nothing moves until it is dragged (L: the last distance)

    # -- the faces ------------------------------------------------------------
    def _items(self):
        out = []
        for name, p in json.loads(self.faces or "[]"):
            o = bpy.data.objects.get(name)
            if o is None:
                raise ValueError(f"{name} not found")
            out.append((o, np.array(p, float)))
        return out

    def _check(self, items):
        """Why these faces can't be pushed / pulled ('' when they can)."""
        for o, p in items:
            if not store.is_nurbs(o) or not _is_solid(o):
                return f"{o.name} is not a closed solid (Join its faces into a closed body first)"
        return ""

    def _take_picks(self, context):
        """No faces given: the faces picked in Face mode on the selected bodies."""
        if self.faces:
            return
        picked = []
        for o in context.selected_objects:
            if store.is_nurbs(o):
                picked += [(o, face_frame(o, i)[0]) for i in subsel.picks(o, "FACE")]
        if picked:
            self.faces = _faces_json(picked)

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        self._take_picks(context)
        if self.faces:
            try:
                why = self._check(self._items())
            except Exception as exc:
                why = str(exc)
            if why:
                self.report({'WARNING'}, why)
                return {'CANCELLED'}
            return self.ask(context)
        if bpy.app.background:
            self.report({'WARNING'}, "Pick a face of a solid (Face mode)")
            return {'CANCELLED'}
        # pick a face: the next click on a face of a solid starts the value stage
        self._dragging = False
        self._hover = None
        context.window_manager.modal_handler_add(self)
        self._pick_hints(context)
        return {'RUNNING_MODAL'}

    # -- picking a face ---------------------------------------------------------
    def _pick_hints(self, context, msg=""):
        state = ["click a face of a solid (curved: offset)"] + ([msg] if msg else [])
        hints.show(context, self.bl_label, state,
                   [("", [hints.note("or pick faces in Face mode first: they all move together")])],
                   [("MOUSE_LMB", "Pick the face"), ("EVENT_ESC", "Cancel")])

    def modal(self, context, event):
        if getattr(self, "_dragging", False):
            return ValueDrag.modal(self, context, event)
        if event.type in NAV:
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            self._show_face(self._hit(context, event))
            context.area.tag_redraw()
            return {'PASS_THROUGH'}
        if event.type in ('ESC', 'RIGHTMOUSE') and event.value == 'PRESS':
            self._end_pick(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = self._hit(context, event)
            if hit is None:
                self._pick_hints(context, "no face there")
                return {'RUNNING_MODAL'}
            o, i = hit
            p = face_frame(o, i)[0]
            why = self._check([(o, p)])
            if why:
                self._pick_hints(context, why)
                return {'RUNNING_MODAL'}
            self._end_pick(context)
            for x in context.selected_objects:
                x.select_set(False)
            o.select_set(True)
            context.view_layer.objects.active = o
            # a value stage can't start inside this loop: start the command again
            # with the face (the press goes on: pulling drags the arrow out)
            bpy.ops.nproto.push_pull('INVOKE_DEFAULT', faces=_faces_json([(o, p)]))
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    # -- the value stage (valuedrag.py) -----------------------------------------
    def drag_setup(self, context, mouse):
        """Each face's outline and fill (world) with the normal at every point
        (one normal for a flat face): the preview moves them by distance x normal."""
        self._frames = []
        for o, p in self._items():
            i = _face_index(o, p)
            _, pw, nw, flat = face_frame(o, i)
            pts, tris, polys = subsel.face_tris(o, i)
            M = np.array(geom.world_matrix(o))
            if not len(tris) or len(pts) > self.MAX_FILL:
                pts, tris = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
            if flat:
                Np = [np.broadcast_to(nw, P.shape) for P in polys]
                Nt = np.broadcast_to(nw, pts.shape)
            else:
                Np = [normals_at(o, i, P) for P in polys]
                Nt = normals_at(o, i, pts) if len(pts) else pts
            self._frames.append({"P": pw, "n": nw, "flat": flat, "polys": [_world(M, P) for P in polys], "Np": Np,
                                 "pts": _world(M, pts) if len(pts) else pts, "Nt": Nt, "tris": tris})
        if not self._frames:
            return False
        f0 = self._frames[0]
        self._B, self._n = f0["P"], f0["n"]
        self._base_at_mouse = False
        if mouse is not None:           # the mouse on the (first) face: the arrow stands there
            region, rv3d = self._win(context)
            o0, p0 = self._items()[0]
            i0 = _face_index(o0, p0)
            from bpy_extras import view3d_utils
            ro = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
            rd = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
            hit, loc, _, _, ob, _ = context.scene.ray_cast(context.evaluated_depsgraph_get(), ro, rd)
            if hit and ob is not None and ob.original == o0:
                q = np.array(geom.world_matrix(o0).inverted() @ loc, float)
                if _face_index(o0, q) == i0:
                    self._B = np.array(loc, float)
                    self._n = f0["n"] if f0["flat"] else normals_at(o0, i0, [q])[0]
                    self._base_at_mouse = True
        return True

    def drag_handles(self):
        return [{"field": "distance", "kind": "arrow", "origin": self._B, "dir": self._n, "param": self.distance,
                 "base": self._B, "to_value": lambda t: t}]

    def drag_preview(self):
        d = self.distance
        lines, tris = [], []
        for fr in self._frames:
            for P, N in zip(fr["polys"], fr["Np"]):
                Q = P + d * N
                lines.append(Q)
                L, T = ruled_preview([P, Q], ribs=4)
                lines += L[2:]
                tris.append(T)
            if len(fr["pts"]) and not fr["flat"]:       # the offset face itself (a flat one: its outline says it)
                tris.append((fr["pts"] + d * fr["Nt"])[fr["tris"]].reshape(-1, 3))
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3))}

    def drag_check(self):
        if abs(self.distance) <= 0.0:
            return "Distance 0: drag the arrow out (pull) or into the body (push), or type a value"
        return ""

    def drag_key(self, context, event, ch):
        if ch == "f":
            self.distance = -self.distance
            return True
        return False

    def drag_rows(self):
        return [("Direction", [hints.key("F", "the other way"),
                               hints.note("drag through the face to push")])]

    def drag_state(self):
        out = ["curved: offset"] if any(not f["flat"] for f in getattr(self, "_frames", [])) else []
        if abs(self.distance) > 0.0:
            out.append("pull: adds material" if self.distance > 0 else "push: removes material")
        return out

    def drag_handle_hints(self):
        return [hints.key("LMB drag", "the arrow (out: pull, in: push)")]

    # -- making it --------------------------------------------------------------
    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._take_picks(context)
        try:
            items = self._items()
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Push / Pull: {exc}")
            return {'CANCELLED'}
        if not items:
            self.report({'WARNING'}, "Pick a face of a solid (Face mode, or click it)")
            return {'CANCELLED'}
        if abs(self.distance) <= 0.0:
            self.report({'WARNING'}, "Push / Pull: the distance is 0")
            return {'CANCELLED'}
        by_obj = {}
        for o, p in items:
            by_obj.setdefault(o.name, (o, []))[1].append(p)
        hist = features.history_enabled(context)
        done = []
        for o, pts in by_obj.values():
            M = geom.world_matrix(o)
            Mi = M.inverted()
            frames = [face_frame(o, _face_index(o, p)) for p in pts]
            had_picks = bool(subsel.picks(o, "FACE")) or "FACE" in subsel.kinds(context)
            try:
                info, breps = jobs.run("push_pull", {"target": o}, ["result"],
                                       face_pts=[[float(x) for x in f[1]] for f in frames],
                                       face_normals=[[float(x) for x in f[2]] for f in frames],
                                       face_flat=[bool(f[3]) for f in frames],
                                       distance=float(self.distance), into="target")
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"Push / Pull on {o.name} failed: {exc}")
                continue
            pre = o[store.PROP_BREP]
            if not hist and features.has_history(o):
                features.clear_history(o)
                self.report({'INFO'}, f"{o.name}: construction history removed")
            store.set_geometry_text(o, breps["result"], context, keep_untrimmed=False)
            if hist:
                features.push_step(o, "push_pull", [], {
                    "faces_local": [[float(x) for x in f[0]] for f in frames],
                    "normals_local": [[float(x) for x in (M.to_3x3().transposed() @ Vector(f[2])).normalized()]
                                      for f in frames],
                    "flat": [bool(f[3]) for f in frames],
                    "distance": float(self.distance)}, base_text=pre)
            # the moved faces stay picked (Face mode): push / pull them again
            if had_picks:
                moved = [Mi @ Vector(f[1] + self.distance * f[2]) for f in frames]
                try:
                    subsel.set_picks(o, "FACE", [_face_index(o, q) for q in moved])
                except Exception as exc:
                    print("NURBS push/pull picks:", exc)
            done.append(o)
            how = ("offset out" if self.distance > 0 else "offset in") if not any(f[3] for f in frames) else \
                ("pulled" if self.distance > 0 else "pushed")
            self.report({'INFO'}, f"{o.name}: {len(pts)} face{'s' if len(pts) > 1 else ''} {how} "
                                  f"{abs(self.distance):.4g} · volume {info.get('volume', 0):.6g}")
            lean = float(info.get("lean", 0.0))
            if lean > 0.5:
                step = abs(self.distance) * np.sin(np.radians(lean))
                self.report({'WARNING'}, f"{o.name}: a neighbour meets the moved face {lean:.0f}° from square; "
                                         f"the walls go along the face's normal, so a step of about {step:.3g} "
                                         "is left along that edge")
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}


class NPROTO_OT_replace_face(_FacePicking, bpy.types.Operator):
    """Replace faces of a solid by a surface: the face's sides are extended or
    cut back along its normal to meet it (the surface above, below or across
    the face). Faces picked in Face mode with the surface selected, or click
    the face, then the surface"""
    bl_idname = "nproto.replace_face"
    bl_label = "Replace Face"
    bl_options = {'REGISTER', 'UNDO'}

    faces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="The faces: [[body name, a point on the face (local)], ...]")
    surface: StringProperty(name="Surface", options={'SKIP_SAVE'},
                            description="The open NURBS surface that takes the faces' place")

    def _items(self):
        return NPROTO_OT_push_pull._items(self)

    @staticmethod
    def _is_sheet(o):
        return store.is_nurbs(o) and not _is_solid(o)

    def _take_selection(self, context):
        """Face mode picks on the selected solids; the selected open surface."""
        if not self.faces:
            picked = []
            for o in context.selected_objects:
                if store.is_nurbs(o) and _is_solid(o):
                    picked += [(o, face_frame(o, i)[0]) for i in subsel.picks(o, "FACE")]
            if picked:
                self.faces = _faces_json(picked)
        if not self.surface:
            bodies = {n for n, _ in json.loads(self.faces or "[]")}
            sheets = [o for o in context.selected_objects if self._is_sheet(o) and o.name not in bodies]
            if len(sheets) == 1:
                self.surface = sheets[0].name

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._take_selection(context)
        if self.faces and self.surface:
            return self.execute(context)
        if bpy.app.background:
            return self.execute(context)
        self._stage = "SURFACE" if self.faces else "FACE"
        self._hover = None
        context.window_manager.modal_handler_add(self)
        self._hints(context)
        return {'RUNNING_MODAL'}

    def _hints(self, context, msg=""):
        step = ("click the face to replace (a face of a solid)" if self._stage == "FACE"
                else "click the surface that replaces it (an open surface)")
        hints.show(context, self.bl_label, [step] + ([msg] if msg else []),
                   [("", [hints.note("or pick faces in Face mode and select the surface: no clicks")])],
                   [("MOUSE_LMB", "Pick"), ("EVENT_ESC", "Cancel")])

    def modal(self, context, event):
        if event.type in NAV:
            return {'PASS_THROUGH'}
        face_stage = self._stage == "FACE"
        if event.type == 'MOUSEMOVE':
            self._show_face(self._hit(context, event), solids=face_stage)
            context.area.tag_redraw()
            return {'PASS_THROUGH'}
        if event.type in ('ESC', 'RIGHTMOUSE') and event.value == 'PRESS':
            self._end_pick(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = self._hit(context, event)
            if hit is None:
                self._hints(context, "nothing there")
                return {'RUNNING_MODAL'}
            o, i = hit
            if face_stage:
                if not _is_solid(o):
                    self._hints(context, f"{o.name} is not a closed solid")
                    return {'RUNNING_MODAL'}
                self.faces = _faces_json([(o, face_frame(o, i)[0])])
                self._stage = "SURFACE"
                self._hover = None
                self._show_face(None)
                self._hints(context)
                return {'RUNNING_MODAL'}
            bodies = {n for n, _ in json.loads(self.faces)}
            if not self._is_sheet(o) or o.name in bodies:
                self._hints(context, f"{o.name} is not an open surface")
                return {'RUNNING_MODAL'}
            self.surface = o.name
            self._end_pick(context)
            return self.execute(context)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._take_selection(context)
        try:
            items = self._items()
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Replace Face: {exc}")
            return {'CANCELLED'}
        surf = bpy.data.objects.get(self.surface)
        if not items or surf is None:
            self.report({'WARNING'}, "Replace Face: pick the face of a solid (Face mode) and the surface "
                                     "that replaces it")
            return {'CANCELLED'}
        if not self._is_sheet(surf):
            self.report({'WARNING'}, f"Replace Face: {surf.name} is not an open NURBS surface")
            return {'CANCELLED'}
        by_obj = {}
        for o, p in items:
            if o == surf:
                continue
            by_obj.setdefault(o.name, (o, []))[1].append(p)
        hist = features.history_enabled(context)
        done = []
        for o, pts in by_obj.values():
            if not _is_solid(o):
                self.report({'ERROR'}, f"Replace Face: {o.name} is not a closed solid")
                continue
            M = geom.world_matrix(o)
            frames = [face_frame(o, _face_index(o, p)) for p in pts]
            try:
                info, breps = jobs.run("replace_face", {"target": o, "surface": surf}, ["result"],
                                       face_pts=[[float(x) for x in f[1]] for f in frames],
                                       face_normals=[[float(x) for x in f[2]] for f in frames],
                                       face_flat=[bool(f[3]) for f in frames], into="target")
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"Replace Face on {o.name} failed: {exc}")
                continue
            pre = o[store.PROP_BREP]
            if not hist and features.has_history(o):
                features.clear_history(o)
                self.report({'INFO'}, f"{o.name}: construction history removed")
            store.set_geometry_text(o, breps["result"], context, keep_untrimmed=False)
            if hist:
                features.push_step(o, "replace_face", [(surf, "SURF")], {
                    "faces_local": [[float(x) for x in f[0]] for f in frames],
                    "normals_local": [[float(x) for x in (M.to_3x3().transposed() @ Vector(f[2])).normalized()]
                                      for f in frames],
                    "flat": [bool(f[3]) for f in frames]}, base_text=pre)
            subsel.set_picks(o, "FACE", [])        # the faces are gone: no pick moves onto a neighbour
            done.append(o)
            self.report({'INFO'}, f"{o.name}: {len(pts)} face{'s' if len(pts) > 1 else ''} replaced by "
                                  f"{surf.name} · volume {info.get('volume', 0):.6g}")
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}


classes = (NPROTO_OT_push_pull, NPROTO_OT_replace_face)
