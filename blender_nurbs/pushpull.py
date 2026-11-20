"""Push / Pull: a flat face of a closed solid moved along its normal, as in
Shapr3D and Plasticity. Out (a positive distance) adds material, in removes
it: the prism the face sweeps is joined to or cut from the body (the worker's
push_pull job, which falls back to extruding the face and stitching it when
the boolean can't). The face comes from Face mode (the picked faces of the
selected bodies) or a click on a flat face; then an arrow stands on it, from
0: drag it out or through the face, type a value, click away or Enter.

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


def _faces_json(items):
    return json.dumps([[o.name, [float(x) for x in p]] for o, p in items])


class NPROTO_OT_push_pull(ValueDrag, AskValue, bpy.types.Operator):
    """Push or pull flat faces of a solid along their normals: out adds
    material, in removes it. Faces picked in Face mode, or click a flat face;
    then drag the arrow (through the face to push), type a value, click away
    or Enter"""
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
            i = _face_index(o, p)
            if not face_frame(o, i)[3]:
                return f"Push / Pull moves flat faces: the picked face of {o.name} is curved"
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
            self.report({'WARNING'}, "Pick a flat face of a solid (Face mode)")
            return {'CANCELLED'}
        # pick a face: the next click on a flat face starts the value stage
        self._dragging = False
        self._hover = None
        context.window_manager.modal_handler_add(self)
        self._pick_hints(context)
        return {'RUNNING_MODAL'}

    # -- picking a face ---------------------------------------------------------
    def _pick_hints(self, context, msg=""):
        state = ["click a flat face of a solid"] + ([msg] if msg else [])
        hints.show(context, self.bl_label, state,
                   [("", [hints.note("or pick faces in Face mode first: they all move together")])],
                   [("MOUSE_LMB", "Pick the face"), ("EVENT_ESC", "Cancel")])

    def _hit(self, context, event):
        region, _ = self._win(context)
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

    def _show_face(self, hit):
        """Outline and fill the face under the mouse (flat faces of solids only)."""
        if hit == self._hover:
            return
        self._hover = hit
        pv = None
        if hit is not None:
            o, i = hit
            try:
                fr = face_frame(o, i)
                if fr[3] and _is_solid(o):
                    pts, tris, polys = subsel.face_tris(o, i)
                    M = np.array(geom.world_matrix(o))
                    lift = 2e-3 * float(np.linalg.norm(np.ptp(np.vstack(polys), axis=0)) or 1.0) * fr[2]
                    W = lambda P: (np.c_[P, np.ones(len(P))] @ M.T)[:, :3] + lift    # off the face: no z-fight
                    pv = {"lines": [W(P) for P in polys], "tris": W(pts)[tris].reshape(-1, 3) if len(tris) else
                          np.zeros((0, 3))}
            except Exception as exc:
                print("NURBS push/pull hover:", exc)
        draw.pick_overlay["preview"] = pv

    def _end_pick(self, context):
        draw.pick_overlay["preview"] = None
        hints.clear(context)
        context.area.tag_redraw()

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
        self._frames = []
        for o, p in self._items():
            i = _face_index(o, p)
            _, pw, nw, _ = face_frame(o, i)
            M = np.array(geom.world_matrix(o))
            polys = [(np.c_[P, np.ones(len(P))] @ M.T)[:, :3] for P in subsel.face_tris(o, i)[2]]
            self._frames.append((pw, nw, polys))
        if not self._frames:
            return False
        B, n, _ = self._frames[0]
        self._B = B
        self._base_at_mouse = False
        if mouse is not None:           # the mouse on the (first) face: the arrow stands there
            region, rv3d = self._win(context)
            first = self._items()[0]
            with context.temp_override(region=region):
                hit = subsel.face_under(bpy.context, (mouse[0], mouse[1]))
            if hit is not None and hit[0] == first[0] and hit[1] == _face_index(*first):
                from bpy_extras import view3d_utils
                o = np.array(view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse), float)
                d = np.array(view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse), float)
                den = float(d @ n)
                if abs(den) > 1e-6:
                    self._B = o + d * float((B - o) @ n) / den
                    self._base_at_mouse = True
        return True

    def drag_handles(self):
        _, n, _ = self._frames[0]
        return [{"field": "distance", "kind": "arrow", "origin": self._B, "dir": n, "param": self.distance,
                 "base": self._B, "to_value": lambda t: t}]

    def drag_preview(self):
        d = self.distance
        lines, tris = [], []
        for _, n, polys in self._frames:
            for P in polys:
                Q = P + d * n
                lines.append(Q)
                L, T = ruled_preview([P, Q], ribs=4)
                lines += L[2:]
                tris.append(T)
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
        if abs(self.distance) <= 0.0:
            return []
        return ["pull: adds material" if self.distance > 0 else "push: removes material"]

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
            self.report({'WARNING'}, "Pick a flat face of a solid (Face mode, or click it)")
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
                    "distance": float(self.distance)}, base_text=pre)
            # the moved faces stay picked (Face mode): push / pull them again
            if had_picks:
                moved = [Mi @ Vector(f[1] + self.distance * f[2]) for f in frames]
                try:
                    subsel.set_picks(o, "FACE", [_face_index(o, q) for q in moved])
                except Exception as exc:
                    print("NURBS push/pull picks:", exc)
            done.append(o)
            how = "pulled" if self.distance > 0 else "pushed"
            self.report({'INFO'}, f"{o.name}: {len(pts)} face{'s' if len(pts) > 1 else ''} {how} "
                                  f"{abs(self.distance):.4g} · volume {info.get('volume', 0):.6g}")
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}


classes = (NPROTO_OT_push_pull,)
