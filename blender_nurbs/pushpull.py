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
place.

Rotate Face turns faces about a hinge (a straight edge, usually one of the
face's own): the face keeps its shape, its neighbours keep their surfaces
(extended or cut back to meet it), as a draft on a single face. The worker's
rotate_face job; history step rotate_face.

Move Face slides a feature (a hole, pocket, boss or slot: the faces inside an
inner loop of the face they sit in, kernel/featregion.py) along that face:
the feature moves rigidly and the face is trimmed again round it. On a flat
face it slides two ways in it; on a cylinder it slides along the axis and
turns about it; on another revolved face (a cone, a sphere) it turns about
the axis; on an extruded face it slides along the extrusion (seat_motion).
The worker's move_face job; history step move_face."""

import json

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty, StringProperty
from mathutils import Vector

from . import draw, features, geom, hints, jobs, store, subsel
from .ops import AskValue, _is_solid, _kernel_ok, _select_only
from .valuedrag import NAV, ValueDrag, angle_pick, rotate_points, ruled_preview


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


def _local_normal(face, q):
    sm, _ = store._k()
    _, uvs, _ = sm.surface_closest_point(face.surface(), tuple(map(float, q)))
    n = np.array(face.outward_normal_at_uv(*_uvxy(uvs[0])), float)
    return n / (np.linalg.norm(n) or 1.0)


def rim_moves(obj, i, polys):
    """For each outline polyline of face i (local), the world move of each of
    its points per unit of distance as the tool makes it: off the face along
    its normal while staying on the neighbour's surface (a leaning neighbour:
    the point slides in its plane), the corners along their side edges."""
    b = store.get_brep(obj)
    F = b.faces()[i]
    Fe = list(F.edges())
    L = np.array(geom.world_matrix(obj).to_3x3())
    out = []
    for P in polys:
        mid = P[len(P) // 2]
        e = min(Fe, key=lambda x: x.closest_point(tuple(map(float, mid)))[-1])     # (point, t, distance)
        others = [f for f in e.faces() if f != F]
        D = []
        for k, q in enumerate(P):
            n = _local_normal(F, q)
            if not others:                       # a seam: just along the normal
                D.append(n)
                continue
            m = _local_normal(others[0], q)
            T = P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)]
            u = np.cross(T, n)
            mu = float(m @ u)
            D.append(n - (float(m @ n) / mu) * u if abs(mu) > 1e-9 * (np.linalg.norm(u) or 1.0) else n)
        # the corners slide along the side edge where two neighbours meet
        for k in (0, len(P) - 1):
            q = P[k]
            v = min(e.vertices(), key=lambda x: float(np.linalg.norm(np.array(x.point(), float) - q)))
            if np.linalg.norm(np.array(v.point(), float) - q) > 1e-6 * (np.linalg.norm(np.ptp(P, axis=0)) or 1.0):
                continue
            sides = [x for x in v.edges() if x not in Fe]
            if len(sides) != 1 or not others:
                continue
            fs = [f for f in sides[0].faces()]
            if len(fs) != 2:
                continue
            w = np.cross(_local_normal(fs[0], q), _local_normal(fs[1], q))
            n = _local_normal(F, q)
            if abs(float(n @ w)) > 1e-6 * (np.linalg.norm(w) or 1.0):
                D[k] = w / float(n @ w)
        Dw = np.array(D) @ L.T
        out.append(Dw)
    return out


def _world(M, P):
    return (np.c_[P, np.ones(len(P))] @ M.T)[:, :3]


def _faces_json(items):
    return json.dumps([[o.name, [float(x) for x in p]] for o, p in items])


def face_highlight(o, i, max_fill=3000):
    """The overlay preview lighting face i of o: its outlines and fill (world),
    lifted a hair off the face so they don't z-fight with it. A curved face
    with more tessellation points than max_fill: outlines only."""
    fr = face_frame(o, i)
    pts, tris, polys = subsel.face_tris(o, i)
    M = np.array(geom.world_matrix(o))
    lift = 2e-3 * float(np.linalg.norm(np.ptp(np.vstack(polys), axis=0)) or 1.0)
    if fr[3]:
        W = lambda P: _world(M, P) + lift * fr[2]
    else:
        W = lambda P: _world(M, P) + lift * normals_at(o, i, P)
    return {"lines": [W(P) for P in polys],
            "tris": W(pts)[tris].reshape(-1, 3) if len(tris) and len(pts) <= max_fill else np.zeros((0, 3))}


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
                    pv = face_highlight(o, i, self.MAX_FILL)
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
                Nt = np.broadcast_to(nw, pts.shape)
            else:
                Nt = normals_at(o, i, pts) if len(pts) else pts
            try:
                Np = rim_moves(o, i, polys)
            except Exception as exc:              # the plain normals: walls along them
                print("NURBS push/pull preview:", exc)
                Np = [np.broadcast_to(nw, P.shape) if flat else normals_at(o, i, P) for P in polys]
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
            if "exact" in info.get("how", []):
                self.report({'INFO'}, f"{o.name}: the leaning neighbours were extended on their own surfaces")
            lean = float(info.get("lean", 0.0))
            if lean > 0.5:
                step = abs(self.distance) * np.sin(np.radians(lean))
                self.report({'WARNING'}, f"{o.name}: a neighbour meets the moved face {lean:.0f}° from square and "
                                         "could not be extended on its own surface; the walls go along the "
                                         f"face's normal, so a step of about {step:.3g} is left along that edge")
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
                                  f"{surf.name} · volume {info.get('volume', 0):.6g}"
                                  + (" · leaning neighbours extended on their own surfaces"
                                     if "exact" in info.get("how", []) else ""))
            lean = float(info.get("lean", 0.0))
            if lean > 0.5:
                self.report({'WARNING'}, f"{o.name}: a neighbour meets the face {lean:.0f}° from square and "
                                         "could not be extended on its own surface (the sides go along the "
                                         "face's normal: a small step is left)")
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}


HINGE_RGBA = (0.55, 0.85, 1.0, 0.95)        # the hinge (axis) line
NOT_HINGE_RGBA = (1.0, 0.45, 0.35, 0.9)      # an edge under the mouse that can't be one (curved)


def _rot3(z, deg):
    """3x3: deg degrees about unit z, right-handed (as valuedrag.rotate_points)."""
    z = np.asarray(z, float)
    a = np.radians(deg)
    K = np.array([[0.0, -z[2], z[1]], [z[2], 0.0, -z[0]], [-z[1], z[0], 0.0]])
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def straight_edge(ob, i):
    """(A, B) world end points of edge i of body ob when it is straight, else None."""
    P = subsel._edge_polys(ob)[i]
    if P is None or len(P) < 2:
        return None
    W = _world(np.array(geom.world_matrix(ob)), P)
    A, B = W[0], W[-1]
    L = float(np.linalg.norm(B - A))
    if L <= 1e-12:
        return None
    t = (B - A) / L
    rel = W - A
    off = rel - np.outer(rel @ t, t)
    if float(np.linalg.norm(off, axis=1).max()) > 1e-4 * L:
        return None
    return A, B


def hinge_axis(face_world, normal_world, A, B):
    """(point, unit direction) of the hinge through A, B, turned so that a
    positive angle swings the face out (its middle along its outward normal)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    z = (B - A) / (np.linalg.norm(B - A) or 1.0)
    r = np.asarray(face_world, float) - A
    r = r - (r @ z) * z
    if float(np.cross(z, r) @ np.asarray(normal_world, float)) < 0.0:
        z = -z
    return A, z


def hinge_on_face(o, i, O, z):
    """Is the hinge (world line through O along z) one of face i's own edges?
    Then a rebuild that moves the face takes its hinge edge along."""
    f = store.get_brep(o).faces()[i]
    Mi = geom.world_matrix(o).inverted()
    size = subsel._size(o)
    s = 1e-2 * size
    d = lambda q: f.closest_point(tuple(map(float, Mi @ Vector(q))))[1]
    tol = 1e-6 * size
    return bool(d(O) <= tol and (d(O + s * z) <= tol or d(O - s * z) <= tol))


class NPROTO_OT_rotate_face(_FacePicking, ValueDrag, AskValue, bpy.types.Operator):
    """Turn faces of a solid about a hinge: click the face, then a straight
    edge (one of its own tilts it like a draft); drag the angle or type it.
    The neighbours keep their own surfaces, extended or cut back to meet the
    turned face"""
    bl_idname = "nproto.rotate_face"
    bl_label = "Rotate Face"
    bl_options = {'REGISTER', 'UNDO'}

    faces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="The faces: [[body name, a point on the face (local)], ...]")
    axis_point: FloatVectorProperty(size=3, options={'HIDDEN', 'SKIP_SAVE'},
                                    description="A point of the hinge (world)")
    axis_dir: FloatVectorProperty(size=3, options={'HIDDEN', 'SKIP_SAVE'},
                                  description="The hinge's direction (world): a positive angle turns right-handed "
                                              "about it")
    angle: FloatProperty(name="Angle", default=0.0, min=-179.0, max=179.0,
                         description="Degrees about the hinge: positive swings the face out (adds material when "
                                     "it is hinged on its own edge), negative in")

    drag_fields = ("angle",)
    angle_fields = ("angle",)
    fit_view = "none"

    # -- the faces and the hinge ------------------------------------------------
    def _items(self):
        return NPROTO_OT_push_pull._items(self)

    def _check(self, items):
        return NPROTO_OT_push_pull._check(self, items)

    def _has_axis(self):
        return float(np.linalg.norm(np.array(self.axis_dir, float))) > 1e-12

    def _set_axis(self, A, B):
        o, p = self._items()[0]
        _, pw, nw, _ = face_frame(o, _face_index(o, p))
        O, z = hinge_axis(pw, nw, A, B)
        self.axis_point, self.axis_dir = [float(x) for x in O], [float(x) for x in z]

    def _take_picks(self, context):
        """Faces picked in Face mode on the selected bodies; one picked straight
        edge (Edge mode) as the hinge."""
        NPROTO_OT_push_pull._take_picks(self, context)
        if not self.faces or self._has_axis():
            return
        edges = [(o, i) for o in context.selected_objects if store.is_nurbs(o) for i in subsel.picks(o, "EDGE")]
        if len(edges) == 1:
            ab = straight_edge(*edges[0])
            if ab is not None:
                self._set_axis(*ab)

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
            if self._has_axis():
                return self.ask(context)
        if bpy.app.background:
            self.report({'WARNING'}, "Pick a face of a solid (Face mode) and a straight edge as the hinge")
            return {'CANCELLED'}
        self._stage = "HINGE" if self.faces else "FACE"
        self._dragging = False
        self._hover = None
        self._hinge_hover = None
        context.window_manager.modal_handler_add(self)
        self._stage_hints(context)
        if self._stage == "HINGE":
            self._show_hinge(None)
            context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    # -- picking ------------------------------------------------------------------
    def _stage_hints(self, context, msg=""):
        if self._stage == "FACE":
            step = ["click the face to turn (a face of a solid)"]
            more = [hints.note("or pick faces in Face mode first: they all turn about one hinge")]
        else:
            step = ["click the hinge: a straight edge (the face's own edge tilts it)"]
            more = [hints.note("or pick the edge in Edge mode with the face")]
        hints.show(context, self.bl_label, step + ([msg] if msg else []), [("", more)],
                   [("MOUSE_LMB", "Pick"), ("EVENT_ESC", "Cancel")])

    def _edge_hit(self, context, event):
        region, _ = ValueDrag._win(context)
        if region is None:
            return None
        xy = (event.mouse_x - region.x, event.mouse_y - region.y)
        if not (0 <= xy[0] < region.width and 0 <= xy[1] < region.height):
            return None
        with context.temp_override(region=region):
            try:
                return subsel.edge_under(bpy.context, xy)
            except Exception as exc:
                print("NURBS rotate face pick:", exc)
                return None

    def _show_hinge(self, hit):
        """The picked faces lit, and the edge under the mouse: blue when it can
        be the hinge, red when it is curved."""
        pv = {"lines": [], "tris": np.zeros((0, 3)), "guides": []}
        tris = []
        for o, p in self._items():
            try:
                h = face_highlight(o, _face_index(o, p), self.MAX_FILL)
                pv["lines"] += h["lines"]
                tris.append(h["tris"])
            except Exception as exc:
                print("NURBS rotate face hover:", exc)
        if tris:
            pv["tris"] = np.vstack(tris)
        if hit is not None:
            ob, i = hit
            P = subsel._edge_polys(ob)[i]
            if P is not None:
                ab = straight_edge(ob, i)
                if ab is not None:                 # the hinge axis: past both ends, as in the value stage
                    A, B = ab
                    pv["guides"].append((np.array([A - 0.25 * (B - A), B + 0.25 * (B - A)]), HINGE_RGBA))
                else:
                    pv["guides"].append((_world(np.array(geom.world_matrix(ob)), P), NOT_HINGE_RGBA))
        draw.pick_overlay["preview"] = pv

    def modal(self, context, event):
        if getattr(self, "_dragging", False):
            return ValueDrag.modal(self, context, event)
        if event.type in NAV or event.type == 'INBETWEEN_MOUSEMOVE':     # (gizmo drags need the in-between moves)
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            if self._stage == "FACE":
                self._show_face(self._hit(context, event))
            else:
                hit = self._edge_hit(context, event)
                if hit != self._hinge_hover:
                    self._hinge_hover = hit
                    self._show_hinge(hit)
            context.area.tag_redraw()
            return {'PASS_THROUGH'}
        if event.type in ('ESC', 'RIGHTMOUSE') and event.value == 'PRESS':
            self._end_pick(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._stage == "FACE":
                hit = self._hit(context, event)
                if hit is None:
                    self._stage_hints(context, "no face there")
                    return {'RUNNING_MODAL'}
                o, i = hit
                p = face_frame(o, i)[0]
                why = self._check([(o, p)])
                if why:
                    self._stage_hints(context, why)
                    return {'RUNNING_MODAL'}
                self.faces = _faces_json([(o, p)])
                self._stage = "HINGE"
                self._hinge_hover = None
                self._show_hinge(None)
                self._stage_hints(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            hit = self._edge_hit(context, event)
            if hit is None:
                self._stage_hints(context, "no edge there")
                return {'RUNNING_MODAL'}
            ab = straight_edge(*hit)
            if ab is None:
                self._stage_hints(context, "that edge is curved: the hinge must be straight")
                return {'RUNNING_MODAL'}
            self._set_axis(*ab)
            self._end_pick(context)
            bodies = {o for o, _ in self._items()}
            for x in context.selected_objects:
                x.select_set(False)
            for o in bodies:
                o.select_set(True)
            context.view_layer.objects.active = next(iter(bodies))
            # a value stage can't start inside this loop: start the command again
            bpy.ops.nproto.rotate_face('INVOKE_DEFAULT', faces=self.faces, axis_point=self.axis_point,
                                       axis_dir=self.axis_dir)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    # -- the value stage (valuedrag.py) -----------------------------------------
    def _axis(self):
        z = np.array(self.axis_dir, float)
        return np.array(self.axis_point, float), z / (np.linalg.norm(z) or 1.0)

    def drag_setup(self, context, mouse):
        """Each face's outline and fill (world), with the way each outline
        point slides on its neighbour (rim_moves): the preview meets the
        turned plane along those."""
        if not self._has_axis():
            return False
        O, z = self._axis()
        self._frames = []
        for o, p in self._items():
            i = _face_index(o, p)
            _, pw, nw, flat = face_frame(o, i)
            pts, tris, polys = subsel.face_tris(o, i)
            M = np.array(geom.world_matrix(o))
            if not len(tris) or len(pts) > self.MAX_FILL:
                pts, tris = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
            try:
                Np = rim_moves(o, i, polys)
            except Exception as exc:
                print("NURBS rotate face preview:", exc)
                Np = [np.broadcast_to(nw, P.shape) for P in polys]
            W = [_world(M, P) for P in polys]
            self._frames.append({"P": pw, "n": nw, "flat": flat, "polys": W, "Np": Np,
                                 "pts": _world(M, pts) if len(pts) else pts, "tris": tris})
        if not self._frames:
            return False
        # the handle: the middle of the outline's part farthest from the hinge
        # (a straight edge across from it: its middle; it lies on the face's surface)
        Q = np.vstack([q for fr in self._frames for q in fr["polys"]])
        rel = Q - O
        r = np.linalg.norm(rel - np.outer(rel @ z, z), axis=1)
        if r.max() <= 1e-12:
            return False
        self._far = Q[r >= r.max() * (1.0 - 1e-3)].mean(axis=0)
        self._lift = 2e-3 * float(np.linalg.norm(np.ptp(Q, axis=0)) or 1.0)
        h = (Q - O) @ z
        pad = 0.15 * float(np.ptp(h) or r.max())
        self._axis_line = np.array([O + z * (h.min() - pad), O + z * (h.max() + pad)])
        self.angle = 0.0
        self._base_at_mouse = False
        return True

    def drag_handles(self):
        O, z = self._axis()
        P = self._far
        C = O + z * float((P - O) @ z)
        x = P - C
        rho = float(np.linalg.norm(x))
        if rho < 1e-12:
            return []
        x /= rho
        y = np.cross(z, x)
        a = self.angle
        n = max(2, int(abs(a) / 3.0) + 1)
        path = np.vstack([rotate_points(P[None], O, z, a * i / (n - 1)) for i in range(n)])
        back = a - np.copysign(min(6.0, abs(a) * 0.5), a) if abs(a) > 1e-9 else 0.0
        return [{"field": "angle", "kind": "arrow", "pos": rotate_points(P[None], O, z, a)[0], "param": a,
                 "origin": C, "dir": None, "path": path, "head_from": rotate_points(P[None], O, z, back)[0],
                 "pick": lambda o, r, prev: angle_pick(C, x, y, z, o, r, prev),
                 "to_value": lambda t: max(-179.0, min(179.0, t)), "snap": (5.0, 1.0)}]

    def drag_preview(self):
        O, z = self._axis()
        a = self.angle
        R = _rot3(z, a)
        lines, tris = [], []
        for fr in self._frames:
            g0, gn = O + R @ (fr["P"] - O), R @ fr["n"]
            for P, N in zip(fr["polys"], fr["Np"]):
                if fr["flat"]:
                    # each outline point slides on its neighbour until it meets the turned plane
                    den = np.asarray(N, float) @ gn
                    ok = np.abs(den) > 1e-6
                    s = np.where(ok, ((g0 - P) @ gn) / np.where(ok, den, 1.0), 0.0)
                    Q = np.where(ok[:, None], P + s[:, None] * N, rotate_points(P, O, z, a))
                else:
                    Q = rotate_points(P, O, z, a)
                lines.append(Q)
                L, T = ruled_preview([P, Q], ribs=4)
                lines += L[2:]
                tris.append(T)
            if len(fr["pts"]):
                X = fr["pts"]
                if fr["flat"]:                 # the fill dropped onto the turned plane along the normal
                    den = float(fr["n"] @ gn)
                    X = X + np.outer(((g0 - X) @ gn) / den, fr["n"]) if abs(den) > 1e-6 else \
                        rotate_points(X, O, z, a)
                    X = X + self._lift * gn         # a hair off the face: no z-fighting at small angles
                else:
                    X = rotate_points(X, O, z, a)
                tris.append(X[fr["tris"]].reshape(-1, 3))
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3)),
                "guides": [(self._axis_line, HINGE_RGBA)]}

    def drag_check(self):
        if abs(self.angle) <= 0.0:
            return "Angle 0: drag the arrow round the hinge, or type a value"
        return ""

    def drag_key(self, context, event, ch):
        if ch == "f":
            self.angle = -self.angle
            return True
        return False

    def drag_rows(self):
        return [("Direction", [hints.key("F", "the other way")])]

    def drag_state(self):
        if abs(self.angle) <= 0.0:
            return []
        return ["out" if self.angle > 0 else "in"]

    def drag_handle_hints(self):
        return [hints.key("LMB drag", "the arrow round the hinge")]

    # -- making it --------------------------------------------------------------
    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._take_picks(context)
        try:
            items = self._items()
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Rotate Face: {exc}")
            return {'CANCELLED'}
        if not items:
            self.report({'WARNING'}, "Pick a face of a solid (Face mode, or click it)")
            return {'CANCELLED'}
        if not self._has_axis():
            self.report({'WARNING'}, "Rotate Face: pick a straight edge as the hinge")
            return {'CANCELLED'}
        if abs(self.angle) <= 0.0:
            self.report({'WARNING'}, "Rotate Face: the angle is 0")
            return {'CANCELLED'}
        why = self._check(items)
        if why:
            self.report({'WARNING'}, why)
            return {'CANCELLED'}
        O, z = self._axis()
        R = _rot3(z, self.angle)
        by_obj = {}
        for o, p in items:
            by_obj.setdefault(o.name, (o, []))[1].append(p)
        hist = features.history_enabled(context)
        done = []
        for o, pts in by_obj.values():
            M = geom.world_matrix(o)
            Mi = M.inverted()
            frames = [face_frame(o, _face_index(o, p)) for p in pts]
            own = [hinge_on_face(o, _face_index(o, p), O, z) for p in pts]
            had_picks = bool(subsel.picks(o, "FACE")) or "FACE" in subsel.kinds(context)
            try:
                info, breps = jobs.run("rotate_face", {"target": o}, ["result"],
                                       face_pts=[[float(x) for x in f[1]] for f in frames],
                                       face_normals=[[float(x) for x in f[2]] for f in frames],
                                       face_flat=[bool(f[3]) for f in frames],
                                       axis_point=[float(x) for x in O], axis_dir=[float(x) for x in z],
                                       hinge_on_face=own, angle=float(self.angle), into="target")
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"Rotate Face on {o.name} failed: {exc}")
                continue
            pre = o[store.PROP_BREP]
            if not hist and features.has_history(o):
                features.clear_history(o)
                self.report({'INFO'}, f"{o.name}: construction history removed")
            store.set_geometry_text(o, breps["result"], context, keep_untrimmed=False)
            if hist:
                L3 = M.to_3x3()
                features.push_step(o, "rotate_face", [], {
                    "faces_local": [[float(x) for x in f[0]] for f in frames],
                    "normals_local": [[float(x) for x in (L3.transposed() @ Vector(f[2])).normalized()]
                                      for f in frames],
                    "flat": [bool(f[3]) for f in frames],
                    "axis_point_local": [float(x) for x in Mi @ Vector(O)],
                    "axis_dir_local": [float(x) for x in (L3.inverted() @ Vector(z)).normalized()],
                    "hinge_on_face": own,
                    "angle": float(self.angle)}, base_text=pre)
            # the turned faces stay picked (Face mode): a point of each, turned with it
            if had_picks:
                moved = [Mi @ Vector(O + R @ (f[1] - O)) for f in frames]
                try:
                    subsel.set_picks(o, "FACE", [_face_index(o, q) for q in moved])
                except Exception as exc:
                    print("NURBS rotate face picks:", exc)
            done.append(o)
            self.report({'INFO'}, f"{o.name}: {len(pts)} face{'s' if len(pts) > 1 else ''} turned "
                                  f"{self.angle:.4g}° · volume {info.get('volume', 0):.6g}")
            lean = float(info.get("lean", 0.0))
            if any(h != "exact" for h in info.get("how", [])) and lean > 0.5:
                self.report({'WARNING'}, f"{o.name}: a neighbour meets the face {lean:.0f}° from square and "
                                         "could not be extended on its own surface; the walls go along the "
                                         "face's normal, so a small step is left along that edge")
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}


_features = {}      # (object, rev, picked face indices) -> feature_of's answer


def feature_of(o, idxs):
    """(feature face indices, seat face indices, {seat index: the feature's
    centre on its rim in that seat (local)}, {seat index: the centre of each
    opening in it (local): a cross hole has two in a cylinder's side}) of the
    faces idxs of body o, or None when they are no feature
    (kernel/featregion.py: the faces inside an inner loop of the face they
    sit in)."""
    key = (o.session_uid, o.get(store.PROP_REV, 0), tuple(sorted(idxs)))
    if key in _features:
        return _features[key]
    if len(_features) > 64:
        _features.clear()
    from .kernel.featregion import face_cycles, feature_region
    b = store.get_brep(o)
    faces = b.faces()
    reg = feature_region(b, [faces[i] for i in idxs])
    out = None
    if reg is not None:
        region, seats, rim = reg
        centres, openings = {}, {}
        for k, S in enumerate(faces):
            if S not in seats:
                continue
            pts, ops = [], []
            for c, kind in face_cycles(S)[0]:
                if kind != "inner" or not all(e in rim for e, _ in c):
                    continue
                P = []
                for e, _ in c:
                    t0, t1 = e.parameter_range()
                    P += [np.array(e.point_at(float(t)), float) for t in np.linspace(t0, t1, 9)[:-1]]
                pts += P
                ops.append(np.mean(P, axis=0))
            centres[k] = np.mean(pts, axis=0)
            openings[k] = ops
        out = (sorted(i for i, f in enumerate(faces) if f in region), sorted(centres), centres, openings)
    _features[key] = out
    return out


def _plane_axes(n):
    """Two unit axes in the plane across n: the world axes most across it,
    flattened into it (a plate on XY: X and Y)."""
    n = np.asarray(n, float)
    W = np.eye(3)
    order = np.argsort(np.abs(W @ n))
    a = W[order[0]] - (W[order[0]] @ n) * n
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    if float(b @ W[order[1]]) < 0.0:          # keep the second near its world axis's sign
        b = -b
    return a, b


def _tilt_axis(rel, z):
    """The axis a feature at rel (from a sphere's centre) tilts about: across
    its meridian, so + tilts it towards z's end (at the pole: across world X,
    else Y)."""
    rel, z = np.asarray(rel, float), np.asarray(z, float)
    t = np.cross(rel, z)
    if np.linalg.norm(t) <= 1e-6 * (np.linalg.norm(rel) or 1.0):
        w = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        t = np.cross(z, w)
    return t / np.linalg.norm(t)


MOTIONS = {"PLANE": "slides in the flat face", "AXIS": "slides along the cylinder's axis and turns about it",
           "TURN": "turns about the revolved face's axis", "SLIDE": "slides along the extruded face",
           "SPHERE": "turns about the sphere's axis and tilts along its meridian"}


def seat_motion(o, s):
    """(mode, world point on the axis, world unit axis) of how a feature can
    move along seat face s of body o -- 'PLANE' (the axis: its normal),
    'AXIS' (a cylinder: along and about its axis), 'SPHERE' (about its
    centre: the point is the centre, the axis its own, else world Z), 'TURN'
    (a surface of revolution: about its axis), 'SLIDE' (a linear extrusion:
    along it) -- or None. Read off the seat's surface (step_export's fits on
    a grid of it)."""
    from . import step_export as sx
    f = store.get_brep(o).faces()[s]
    srf = f.surface()
    M = np.array(geom.world_matrix(o))
    if srf.is_planar():
        p, w, n, _ = face_frame(o, s)
        return "PLANE", w, n
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    G = np.array([[srf.evaluate(float(u), float(v)) for v in np.linspace(v0, v1, 13)]
                  for u in np.linspace(u0, u1, 13)], float)
    G = _world(M, G.reshape(-1, 3)).reshape(G.shape)
    size = float(np.linalg.norm(np.ptp(G.reshape(-1, 3), axis=0))) or 1.0
    tol = 1e-7 * size                         # exact (rational) only: a smooth circle's wall is no cylinder
    cyl = sx.fit_cylinder(G, tol, size)
    if cyl is not None:
        return "AXIS", np.asarray(cyl[0], float), np.asarray(cyl[1], float)
    rev = sx.fit_revolution(G, tol, size)
    sph = sx.fit_sphere(G, tol, size)
    if sph is not None:
        return "SPHERE", np.asarray(sph[0], float), \
            (np.asarray(rev[1], float) if rev is not None else np.array([0.0, 0.0, 1.0]))
    if rev is not None:
        return "TURN", np.asarray(rev[0], float), np.asarray(rev[1], float)
    ex = sx.fit_extrusion(G, tol, size)
    if ex is not None:
        d = np.asarray(ex[1], float)
        return "SLIDE", G[0, 0], d / (np.linalg.norm(d) or 1.0)
    return None


class NPROTO_OT_move_face(_FacePicking, ValueDrag, AskValue, bpy.types.Operator):
    """Move a feature along the face it sits in: click a face of a hole,
    pocket, boss or slot (the whole feature lights up), drag the arrows or
    type the values. On a flat face it slides two ways; on a cylinder it slides
    along the axis and turns about it; on a sphere it turns about its axis
    and tilts along its meridian; on a cone or another revolved face it
    turns. The feature moves rigidly; the face around it is trimmed again"""
    bl_idname = "nproto.move_face"
    bl_label = "Move Face"
    bl_options = {'REGISTER', 'UNDO'}

    faces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="The faces: [[body name, a point on the face (local)], ...]")
    axis_x: FloatVectorProperty(size=3, options={'HIDDEN', 'SKIP_SAVE'},
                                description="The first move direction (world, in the face the feature sits in)")
    axis_y: FloatVectorProperty(size=3, options={'HIDDEN', 'SKIP_SAVE'},
                                description="The second move direction (world)")
    mode: EnumProperty(items=[(k, k.title(), v) for k, v in MOTIONS.items()], default="PLANE",
                       options={'HIDDEN', 'SKIP_SAVE'}, description="How the face it sits in lets it move")
    axis_point: FloatVectorProperty(size=3, options={'HIDDEN', 'SKIP_SAVE'},
                                    description="A point on the axis (world) of a curved face it sits in")
    move_x: FloatProperty(name="X", default=0.0, unit='LENGTH', description="Along the first direction")
    move_y: FloatProperty(name="Y", default=0.0, unit='LENGTH', description="Along the second direction")
    slide: FloatProperty(name="Along", default=0.0, unit='LENGTH',
                         description="Along the axis of the curved face the feature sits in")
    turn: FloatProperty(name="Turn °", default=0.0, min=-360.0, max=360.0,
                        description="About the axis of the curved face the feature sits in (degrees)")
    tilt: FloatProperty(name="Tilt °", default=0.0, min=-360.0, max=360.0,
                        description="On a sphere: along the meridian through the feature, + towards the "
                                    "axis's top (degrees)")

    FIELDS = {"PLANE": ("move_x", "move_y"), "AXIS": ("slide", "turn"), "TURN": ("turn",), "SLIDE": ("slide",),
              "SPHERE": ("turn", "tilt")}
    drag_fields = ("move_x", "move_y")
    angle_fields = ("turn", "tilt")
    fit_view = "none"
    start_value = 0.0

    # -- the faces ------------------------------------------------------------
    def _items(self):
        return NPROTO_OT_push_pull._items(self)

    def _check(self, items):
        why = NPROTO_OT_push_pull._check(self, items)
        if why:
            return why
        for o, idxs in self._by_obj(items).values():
            feat = feature_of(o, idxs)
            if feat is None:
                return (f"{o.name}: not part of a feature sitting inside a face (a hole, pocket, boss or "
                        "slot); Push / Pull or Rotate Face move a face of the body itself")
            if all(seat_motion(o, k) is None for k in feat[1]):
                return (f"{o.name}: the feature sits in a curved face with no axis to slide or turn it "
                        "along (a flat face, a cylinder, a revolved or an extruded face can take it)")
        return ""

    @staticmethod
    def _by_obj(items):
        out = {}
        for o, p in items:
            out.setdefault(o.name, (o, []))[1].append(_face_index(o, p))
        return out

    def _take_picks(self, context):
        NPROTO_OT_push_pull._take_picks(self, context)

    def _has_axes(self):
        return float(np.linalg.norm(np.array(self.axis_x, float))) > 1e-12

    def _seat(self, context, o, feat):
        """(seat index, its world normal there, the opening's centre (local)):
        the feature's opening facing the view (a through hole has two: in a
        plate's top and bottom, or both in a rod's side), else the one facing
        up."""
        view = None
        if context is not None:
            _, rv3d = self._win(context)
            if rv3d is not None:
                view = np.array(rv3d.view_rotation @ Vector((0.0, 0.0, 1.0)), float)    # toward the viewer
        best = None
        faces = store.get_brep(o).faces()
        N3 = geom.world_matrix(o).to_3x3().inverted().transposed()
        for s in feat[1]:
            if seat_motion(o, s) is None:
                continue
            for c in feat[3][s] or [feat[2][s]]:
                nw = np.array(N3 @ Vector(_local_normal(faces[s], c)), float)   # at the opening
                nw /= np.linalg.norm(nw) or 1.0
                score = float(nw @ view) if view is not None else float(nw[2])
                if best is None or score > best[0]:
                    best = (score, s, nw, c)
        return best[1], best[2], best[3]

    def _set_axes(self, context=None):
        """How it moves (seat_motion of the seat facing the view): on a flat
        face two directions in it, on a curved one its axis."""
        o, idxs = next(iter(self._by_obj(self._items()).values()))
        feat = feature_of(o, idxs)
        s, n, _ = self._seat(context, o, feat)
        mode, P, z = seat_motion(o, s)
        self.mode = mode
        if mode == "PLANE":
            a, b = _plane_axes(n)
            self.axis_x, self.axis_y = [float(x) for x in a], [float(x) for x in b]
            return
        if float(np.asarray(z) @ np.array([0.0, 0.0, 1.0])) < -1e-9 or (abs(z[2]) <= 1e-9 and z[0] + z[1] < 0):
            z = -np.asarray(z)                  # the axis upwards (or towards +X/+Y): + is up the axis
        self.axis_x = [float(x) for x in z]
        self.axis_y = [0.0, 0.0, 0.0]
        self.axis_point = [float(x) for x in P]
        if mode == "SPHERE":                    # the tilt axis: across the meridian through the opening
            C = np.array(geom.world_matrix(o) @ Vector(self._seat(context, o, feat)[2]), float)
            self.axis_y = [float(x) for x in _tilt_axis(C - P, z)]

    def _fields(self):
        self.drag_fields = self.FIELDS[self.mode]
        return self.drag_fields

    def _vector(self):
        if self.mode == "PLANE":
            return self.move_x * np.array(self.axis_x, float) + self.move_y * np.array(self.axis_y, float)
        return self.slide * np.array(self.axis_x, float)

    def _angle(self):
        return float(self.turn) if self.mode in ("AXIS", "TURN", "SPHERE") else 0.0

    def _turns(self):
        """[(axis point, unit axis, degrees)] in the order they apply: on a
        sphere the tilt (about axis_y), then the turn (about axis_x)."""
        O, z = np.array(self.axis_point, float), np.array(self.axis_x, float)
        out = []
        if self.mode == "SPHERE" and self.tilt:
            out.append((O, np.array(self.axis_y, float), float(self.tilt)))
        if self._angle():
            out.append((O, z, self._angle()))
        return out

    def _moved(self, P, upto=None):
        """World points P carried by the move: turned (tilted), then slid."""
        P = np.asarray(P, float)
        for O, z, a in self._turns()[:upto]:
            P = rotate_points(P, O, z, a)
        return P + self._vector()

    def draw(self, context):
        for f in self.FIELDS.get(self.mode, ()):
            self.layout.prop(self, f)

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
            if not self._has_axes():
                self._set_axes(context)
            return self.ask(context)
        if bpy.app.background:
            self.report({'WARNING'}, "Pick a face of a hole, pocket or boss (Face mode)")
            return {'CANCELLED'}
        self._dragging = False
        self._hover = None
        context.window_manager.modal_handler_add(self)
        self._pick_hints(context)
        return {'RUNNING_MODAL'}

    # -- picking ------------------------------------------------------------------
    def _pick_hints(self, context, msg=""):
        hints.show(context, self.bl_label, ["click a face of a hole, pocket, boss or slot"] + ([msg] if msg else []),
                   [("", [hints.note("the whole feature lights up; or pick its faces in Face mode first")])],
                   [("MOUSE_LMB", "Pick"), ("EVENT_ESC", "Cancel")])

    def _show_feature(self, hit):
        """The feature under the mouse lit (all its faces); a face that is no
        feature: only its outline, and the hint says so."""
        if hit == self._hover:
            return
        self._hover = hit
        pv = None
        if hit is not None:
            o, i = hit
            try:
                if _is_solid(o):
                    feat = feature_of(o, [i])
                    idxs = feat[0] if feat is not None else [i]
                    pv = {"lines": [], "tris": []}
                    for k in idxs:
                        h = face_highlight(o, k, self.MAX_FILL)
                        pv["lines"] += h["lines"]
                        if feat is not None:
                            pv["tris"].append(h["tris"])
                    pv["tris"] = np.vstack(pv["tris"]) if pv["tris"] else np.zeros((0, 3))
            except Exception as exc:
                print("NURBS move face hover:", exc)
        draw.pick_overlay["preview"] = pv

    def modal(self, context, event):
        if getattr(self, "_dragging", False):
            return ValueDrag.modal(self, context, event)
        if event.type in NAV or event.type == 'INBETWEEN_MOUSEMOVE':     # (gizmo drags need the in-between moves)
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            self._show_feature(self._hit(context, event))
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
                self._pick_hints(context, why.split(": ", 1)[-1].split(";")[0])
                return {'RUNNING_MODAL'}
            self._end_pick(context)
            for x in context.selected_objects:
                x.select_set(False)
            o.select_set(True)
            context.view_layer.objects.active = o
            bpy.ops.nproto.move_face('INVOKE_DEFAULT', faces=_faces_json([(o, p)]))
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    # -- the value stage (valuedrag.py) -----------------------------------------
    def drag_setup(self, context, mouse):
        """The feature's faces (world outlines and fill) and its centre: the
        preview moves them by the vector."""
        if not self._has_axes():
            self._set_axes(context)
        self._frames = []
        C, ext_ = [], []
        for o, idxs in self._by_obj(self._items()).values():
            feat = feature_of(o, idxs)
            if feat is None:
                return False
            M = np.array(geom.world_matrix(o))
            for k in feat[0]:
                pts, tris, polys = subsel.face_tris(o, k)
                if not len(tris) or len(pts) > self.MAX_FILL:
                    pts, tris = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
                W = [_world(M, P) for P in polys]
                self._frames.append({"polys": W, "pts": _world(M, pts) if len(pts) else pts, "tris": tris})
                ext_ += W
            C.append(_world(M, self._seat(context, o, feat)[2][None])[0])
        if not self._frames:
            return False
        self._C = np.mean(C, axis=0)
        Q = np.vstack(ext_)
        span = float(np.linalg.norm(np.ptp(Q, axis=0))) or 1.0
        self._L0 = 0.6 * span                     # the arrows' length past the moved centre
        self._lift = 2e-3 * span
        self.move_x = self.move_y = self.slide = self.turn = self.tilt = 0.0
        self._fields()
        self._span = span
        if self.mode == "SPHERE":
            O = np.array(self.axis_point, float)
            self._r = float(np.linalg.norm(self._C - O))
            if self._r <= 1e-9 * span:
                return False
            self._t0 = np.array(self.axis_y, float)
        if self.mode in ("AXIS", "TURN", "SPHERE"):
            O, z = np.array(self.axis_point, float), np.array(self.axis_x, float)
            rel = self._C - O
            rad = rel - (rel @ z) * z
            if np.linalg.norm(rad) <= 1e-3 * span:
                if self.mode != "SPHERE":
                    return False
                rad = np.cross(z, self._t0)          # at the sphere's pole: along the tilt
            self._x = rad / np.linalg.norm(rad)
            r = Q - O
            rho = float(np.linalg.norm(r - np.outer(r @ z, z), axis=1).max())
            self._R = rho + 0.35 * span           # the turn handle: on a circle just outside the feature
            h = (Q - O) @ z
            pad = 0.25 * span
            self._axis_line = np.array([O + z * (h.min() - pad), O + z * (h.max() + pad)])
        self._base_at_mouse = False
        return True

    def _turn_handle(self):
        O, z = np.array(self.axis_point, float), np.array(self.axis_x, float)
        C = self._moved(self._C[None], upto=1)[0] - self._vector() if self.mode == "SPHERE" and self.tilt \
            else self._C                          # (on a sphere: where the tilt took it)
        Cax = O + z * float((C - O) @ z) + self._vector()
        rad = C - O - float((C - O) @ z) * z
        x = rad / np.linalg.norm(rad) if np.linalg.norm(rad) > 1e-3 * self._span else self._x
        y = np.cross(z, x)
        P = Cax + self._R * x
        a = self.turn
        n = max(2, int(abs(a) / 3.0) + 1)
        path = np.vstack([rotate_points(P[None], Cax, z, a * i / (n - 1)) for i in range(n)])
        back = a - np.copysign(min(6.0, abs(a) * 0.5), a) if abs(a) > 1e-9 else 0.0
        return {"field": "turn", "kind": "arrow", "pos": rotate_points(P[None], Cax, z, a)[0], "param": a,
                "origin": Cax, "dir": None, "path": path, "head_from": rotate_points(P[None], Cax, z, back)[0],
                "pick": lambda o, r, prev: angle_pick(Cax, x, y, z, o, r, prev),
                "to_value": lambda t: max(-360.0, min(360.0, t)), "snap": (5.0, 1.0)}

    def drag_handles(self):
        ax, ay = np.array(self.axis_x, float), np.array(self.axis_y, float)
        moved = self._moved(self._C[None])[0]
        L0 = self._L0
        if self.mode == "PLANE":
            return [{"field": "move_x", "kind": "arrow", "origin": self._C + self.move_y * ay, "dir": ax,
                     "param": self.move_x + L0, "base": moved, "to_value": lambda t: t - L0},
                    {"field": "move_y", "kind": "arrow", "origin": self._C + self.move_x * ax, "dir": ay,
                     "param": self.move_y + L0, "base": moved, "to_value": lambda t: t - L0}]
        out = []
        if self.mode in ("AXIS", "SLIDE"):
            turned = moved - self._vector()
            out.append({"field": "slide", "kind": "arrow", "origin": turned, "dir": ax,
                        "param": self.slide + L0, "base": moved, "to_value": lambda t: t - L0})
        if self.mode in ("AXIS", "TURN", "SPHERE"):
            out.append(self._turn_handle())
        if self.mode == "SPHERE":
            out.append(self._tilt_handle())
        return out

    def _tilt_handle(self):
        """The tilt: an arc in the feature's meridian (turned with the
        turn), about the tilt axis through the sphere's centre."""
        O, z = np.array(self.axis_point, float), np.array(self.axis_x, float)
        t = rotate_points(self._t0[None], np.zeros(3), z, self.turn)[0]
        x = rotate_points(((self._C - O) / self._r)[None], np.zeros(3), z, self.turn)[0]
        y = np.cross(t, x)
        P = O + (self._r + 1.1 * self._span) * x          # further out than the turn handle
        a = self.tilt
        n = max(2, int(abs(a) / 3.0) + 1)
        path = np.vstack([rotate_points(P[None], O, t, a * i / (n - 1)) for i in range(n)])
        back = a - np.copysign(min(6.0, abs(a) * 0.5), a) if abs(a) > 1e-9 else 0.0
        return {"field": "tilt", "kind": "arrow", "pos": rotate_points(P[None], O, t, a)[0], "param": a,
                "origin": O, "dir": None, "path": path, "head_from": rotate_points(P[None], O, t, back)[0],
                "pick": lambda o, r, prev: angle_pick(O, x, y, t, o, r, prev),
                "to_value": lambda v: max(-360.0, min(360.0, v)), "snap": (5.0, 1.0)}

    def _lifted(self, P):
        """Preview fill points lifted off the face (the plane's normal; out
        from a curved face's axis)."""
        if self.mode == "PLANE":
            return P + self._lift * np.cross(np.array(self.axis_x, float), np.array(self.axis_y, float))
        if self.mode == "SLIDE":
            return P
        O, z = np.array(self.axis_point, float), np.array(self.axis_x, float)
        if self.mode == "SPHERE":
            r = P - O
            return P + self._lift * r / np.maximum(np.linalg.norm(r, axis=1), 1e-300)[:, None]
        r = P - O
        r = r - np.outer(r @ z, z)
        return P + self._lift * r / np.maximum(np.linalg.norm(r, axis=1), 1e-300)[:, None]

    def drag_preview(self):
        lines, tris = [], []
        for fr in self._frames:
            lines += [self._moved(P) for P in fr["polys"]]
            if len(fr["pts"]):
                tris.append(self._lifted(self._moved(fr["pts"]))[fr["tris"]].reshape(-1, 3))
        guides = []
        if self.mode in ("AXIS", "TURN", "SPHERE"):
            guides.append((self._axis_line + self._vector(), HINGE_RGBA))
            C = self._C
            for O, z, a in self._turns():          # the path of the feature's centre, turn by turn
                n = max(2, int(abs(a) / 3.0) + 1)
                guides.append((np.vstack([rotate_points(C[None], O, z, a * i / (n - 1)) for i in range(n)])
                               + self._vector(), HINGE_RGBA))
                C = rotate_points(C[None], O, z, a)[0]
        d = self._vector()
        if np.linalg.norm(d) > 0:
            start = self._moved(self._C[None])[0] - d
            guides.append((np.array([start, start + d]), HINGE_RGBA))
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3)), "guides": guides}

    def drag_check(self):
        if np.linalg.norm(self._vector()) <= 0.0 and not self._turns():
            if self.mode == "PLANE":
                return "No move yet: drag an arrow along the face, or type a distance (Tab: the other direction)"
            if self.mode == "AXIS":
                return "No move yet: drag the arrow along the axis or the handle round it (Tab: the other)"
            if self.mode == "TURN":
                return "No turn yet: drag the handle round the axis, or type an angle"
            if self.mode == "SPHERE":
                return "No move yet: drag the handle round the axis or the one along the meridian (Tab: the other)"
            return "No move yet: drag the arrow along the face, or type a distance"
        return ""

    def drag_rows(self):
        return [("", [hints.note(f"the feature {MOTIONS[self.mode]}")])]

    def drag_handle_hints(self):
        if self.mode == "PLANE":
            return [hints.key("LMB drag", "an arrow (the two directions in the face)")]
        if self.mode == "AXIS":
            return [hints.key("LMB drag", "the arrow (along the axis) or the handle (round it)")]
        if self.mode == "TURN":
            return [hints.key("LMB drag", "the handle round the axis")]
        if self.mode == "SPHERE":
            return [hints.key("LMB drag", "a handle (round the axis, or along the meridian)")]
        return [hints.key("LMB drag", "the arrow along the face")]

    # -- making it --------------------------------------------------------------
    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._take_picks(context)
        try:
            items = self._items()
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Move Face: {exc}")
            return {'CANCELLED'}
        if not items:
            self.report({'WARNING'}, "Pick a face of a hole, pocket or boss (Face mode, or click it)")
            return {'CANCELLED'}
        why = self._check(items)
        if why:
            self.report({'WARNING'}, why)
            return {'CANCELLED'}
        if not self._has_axes():
            self._set_axes(context)
        d = self._vector()
        ang = self._angle()
        turns = self._turns()
        if np.linalg.norm(d) <= 0.0 and not turns:
            self.report({'WARNING'}, "Move Face: the move is 0")
            return {'CANCELLED'}
        turn = {"turns": [[[float(x) for x in O], [float(x) for x in z], a] for O, z, a in turns]} if turns else {}
        hist = features.history_enabled(context)
        done = []
        for o, pts in [(o, [p for oo, p in items if oo == o]) for o in {o for o, _ in items}]:
            M = geom.world_matrix(o)
            Mi = M.inverted()
            frames = [face_frame(o, _face_index(o, p)) for p in pts]
            feat = feature_of(o, [_face_index(o, p) for p in pts])
            had_picks = bool(subsel.picks(o, "FACE")) or "FACE" in subsel.kinds(context)
            try:
                info, breps = jobs.run("move_face", {"target": o}, ["result"],
                                       face_pts=[[float(x) for x in f[1]] for f in frames],
                                       vector=[float(x) for x in d], into="target", **turn)
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"Move Face on {o.name} failed: {exc}")
                continue
            pre = o[store.PROP_BREP]
            if not hist and features.has_history(o):
                features.clear_history(o)
                self.report({'INFO'}, f"{o.name}: construction history removed")
            # where the feature's faces will be (local), for the picks
            moved = [Mi @ Vector(self._moved(np.array(M @ Vector(face_frame(o, k)[0]), float)[None])[0])
                     for k in feat[0]] if had_picks and feat is not None else []
            store.set_geometry_text(o, breps["result"], context, keep_untrimmed=False)
            if hist:
                L3 = M.to_3x3().inverted()
                curved = self.mode != "PLANE"
                features.push_step(o, "move_face", [], {
                    "faces_local": [[float(x) for x in f[0]] for f in frames], "mode": self.mode,
                    "axis_x_local": [float(x) for x in L3 @ Vector(self.axis_x)],
                    "axis_y_local": [float(x) for x in L3 @ Vector(self.axis_y)],
                    "axis_point_local": [float(x) for x in Mi @ Vector(self.axis_point)],
                    "move_x": 0.0 if curved else float(self.move_x), "move_y": 0.0 if curved else float(self.move_y),
                    "slide": float(self.slide) if curved else 0.0, "turn": ang,
                    "tilt": float(self.tilt) if self.mode == "SPHERE" else 0.0}, base_text=pre)
            if moved:
                try:
                    subsel.set_picks(o, "FACE", sorted({_face_index(o, q) for q in moved}))
                except Exception as exc:
                    print("NURBS move face picks:", exc)
            done.append(o)
            how = " ".join(x for x in (f"turned {ang:.4g}°" if ang else "",
                                       f"tilted {self.tilt:.4g}°" if self.mode == "SPHERE" and self.tilt else "",
                                       f"moved {float(np.linalg.norm(d)):.4g}" if np.linalg.norm(d) > 0 else "") if x)
            self.report({'INFO'}, f"{o.name}: a feature of {info.get('feature', 0)} face"
                                  f"{'s' if info.get('feature', 0) != 1 else ''} {how} · "
                                  f"volume {info.get('volume', 0):.6g}")
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}


classes = (NPROTO_OT_push_pull, NPROTO_OT_replace_face, NPROTO_OT_rotate_face, NPROTO_OT_move_face)
