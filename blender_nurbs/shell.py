"""Shell: a closed solid hollowed into walls of one thickness, the picked
faces taken away (open), as in Fusion, Shapr3D and Plasticity. A block
with its top picked becomes a box; a bar with both ends picked becomes a
tube (a welded frame's members). No face picked: a closed hollow (a void
inside).

Pick the faces in Face mode first, or start the tool and click them (a
click on a lit face takes it back), Enter; then an arrow stands on the
first face pointing into the body: drag it, type a thickness, click away or
Enter. F: the walls go outside instead (the body becomes the cavity).

The preview draws the walls' inner edges where the kernel's shell puts
them (offset faces extended until they meet), so the shape of the cavity
shows before anything is computed.

With history the shell is a step on the body (shell, replayed on rebuilds,
the faces found again by a point on each); without it the body changes in
place. The worker's shell job (the kernel's own shell_brep)."""

import json

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, StringProperty
from mathutils import Vector

from . import draw, features, geom, hints, jobs, store, subsel
from .ops import AskValue, _is_solid, _kernel_ok, _select_only
from .pushpull import _FacePicking, _face_index, _faces_json, _local_normal, face_frame, face_highlight, normals_at
from .valuedrag import ValueDrag, ruled_preview

EDGE_SAMPLES = 16               # points along a curved edge in the preview


def _segments(polys):
    """Polylines as one (m, 2, 3) array of segments (the region overlay's border)."""
    segs = [np.stack([P[:-1], P[1:]], axis=1) for P in polys if len(P) > 1]
    return np.concatenate(segs) if segs else np.zeros((0, 2, 3))


def _edge_points(e, flat):
    """Local points along an edge: its ends for a straight one (both faces flat), else EDGE_SAMPLES."""
    t0, t1 = e.parameter_range()
    n = 2 if flat else EDGE_SAMPLES
    return np.array([e.point_at(t) for t in np.linspace(t0, t1, n)], float).reshape(-1, 3)


def face_centre(o, i):
    """(world point, world unit outward normal) in the middle of face i of o:
    the area centroid of its tessellation brought onto the face (the arrow
    stands there)."""
    pts, tris, _ = subsel.face_tris(o, i)
    f = store.get_brep(o).faces()[i]
    if len(tris):
        T = pts[np.asarray(tris)]
        a = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        c = (T.mean(axis=1) * a[:, None]).sum(axis=0) / (a.sum() or 1.0)
        q = np.array(f.closest_point(tuple(map(float, c)))[0], float)
    else:
        q = face_frame(o, i)[0]
    return np.array(geom.world_matrix(o) @ Vector(q), float), normals_at(o, i, [q])[0]


def wall_edges(o, removed):
    """Where the shell's inner edges go, per unit of thickness (world): for
    each edge of body o, (points (k,3), move (k,3)) -- the inner edge of an
    inside wall is points - thickness x move, of an outside wall points +
    thickness x move. An edge between two kept faces goes to where their
    offsets meet (w with w.nA = w.nB = 1); an edge of a removed face slides in
    that face until it is the thickness off the kept neighbour (the open
    rim, flagged True). Edges of removed faces only are left out."""
    b = store.get_brep(o)
    faces = list(b.faces())
    rem = set(removed)
    M = geom.world_matrix(o)
    A = np.array(M)
    L = np.array(M.to_3x3().inverted().transposed())
    flat = [bool(f.surface().is_planar()) for f in faces]

    def fidx(f):
        for k, g in enumerate(faces):
            if g == f:
                return k
        return -1

    corners = {}

    def corner(v):
        """The move of vertex v per unit thickness: D.n = 1 for each kept face
        round it, D.n = 0 for each removed one (the rim stays in that face);
        least squares, the shortest D when the faces leave it free."""
        key = tuple(np.round(np.array(v.point(), float), 9))
        if key in corners:
            return corners[key]
        q = np.array(v.point(), float)
        rows, rhs = [], []
        for x in v.edges():
            for f in x.faces():
                k = fidx(f)
                if k < 0:
                    continue
                n = np.array(_local_normal(faces[k], q)) @ L.T
                n /= np.linalg.norm(n) or 1.0
                want = 0.0 if k in rem else 1.0
                if any(abs(float(n @ r)) > 1.0 - 1e-6 and w == want for r, w in zip(rows, rhs)):
                    continue                     # the same plane again (or the same face from another edge)
                rows.append(n)
                rhs.append(want)
        Dv = None
        if rows and any(w > 0 for w in rhs):
            Dv = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=1e-6)[0]
        corners[key] = Dv
        return Dv

    out = []
    for e in b.edges():
        fs = [fidx(f) for f in e.faces()]
        fs = [k for k in fs if k >= 0]
        if not fs:
            continue
        if len(fs) == 1:
            fs = fs * 2                          # a seam: the face meets itself
        a, c = fs[0], fs[1]
        if a in rem and c in rem:
            continue
        P = _edge_points(e, flat[a] and flat[c])
        if len(P) < 2:
            continue
        Pw = (np.c_[P, np.ones(len(P))] @ A.T)[:, :3]

        def nrm(k):
            N = np.array([_local_normal(faces[k], q) for q in P]) @ L.T
            return N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
        Na, Nc = nrm(a), nrm(c)
        rim = (a in rem) != (c in rem)
        if rim:
            n, m = (Na, Nc) if a in rem else (Nc, Na)      # n: the removed face, m: the kept one
            U = m - np.sum(m * n, axis=1)[:, None] * n
            uu = np.sum(U * U, axis=1)
            D = np.where(uu[:, None] > 1e-6, U / np.maximum(uu, 1e-300)[:, None], m)
        else:
            den = 1.0 + np.sum(Na * Nc, axis=1)
            D = np.where(den[:, None] > 1e-3, (Na + Nc) / np.maximum(den, 1e-300)[:, None], Na)
        # the ends: where every face round the vertex says (a box corner meets its third face)
        for k in (0, len(P) - 1):
            v = min(e.vertices(), key=lambda x: float(np.linalg.norm(np.array(x.point(), float) - P[k])))
            Dv = corner(v)
            if Dv is not None:
                D[k] = Dv
        out.append((Pw, D, rim))
    return out


class NPROTO_OT_shell(_FacePicking, ValueDrag, AskValue, bpy.types.Operator):
    """Hollow a solid into walls of one thickness, the picked faces taken away
    (a block with its top picked becomes a box, a bar with both ends picked a
    tube). Faces picked in Face mode, or click them and Enter (none: a closed
    hollow); then drag the arrow into the body, type a thickness, click away
    or Enter"""
    bl_idname = "nproto.shell"
    bl_label = "Shell"
    bl_options = {'REGISTER', 'UNDO'}

    faces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="The faces taken away: [[body name, a point on the face (local)], ...]")
    bodies: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                           description="Bodies hollowed with no face taken away (a closed void): [name, ...]")
    thickness: FloatProperty(name="Thickness", default=0.0, min=0.0, unit='LENGTH',
                             description="The walls' thickness")
    side: EnumProperty(name="Side", default='INSIDE',
                       items=[('INSIDE', "Inside", "The walls inside the body: its outside stays as it is"),
                              ('OUTSIDE', "Outside", "The walls around the body: it becomes the cavity")],
                       description="Which side of the faces the walls go")

    drag_fields = ("thickness",)
    fit_view = "shrink"
    start_value = 0.0           # the arrow on the face: nothing until it is dragged (L: the last thickness)

    # -- the inputs -----------------------------------------------------------
    def _items(self):
        """{body name: (body, [local points of its removed faces])}, bodies
        with no face (a closed hollow) included with an empty list."""
        out = {}
        for name, p in json.loads(self.faces or "[]"):
            o = bpy.data.objects.get(name)
            if o is None:
                raise ValueError(f"{name} not found")
            out.setdefault(name, (o, []))[1].append(np.array(p, float))
        for name in json.loads(self.bodies or "[]"):
            o = bpy.data.objects.get(name)
            if o is None:
                raise ValueError(f"{name} not found")
            out.setdefault(name, (o, []))
        return out

    @staticmethod
    def _check(o):
        if not store.is_nurbs(o) or not _is_solid(o):
            return f"{o.name} is not a closed solid (Join its faces into a closed body first)"
        return ""

    def _take_picks(self, context):
        """No faces given: the faces picked in Face mode on the selected bodies."""
        if self.faces or self.bodies:
            return
        picked = []
        for o in context.selected_objects:
            if store.is_nurbs(o):
                picked += [(o, face_frame(o, i)[0]) for i in subsel.picks(o, "FACE")]
        if picked:
            self.faces = _faces_json(picked)

    def _whole_bodies(self, context):
        """The selected solids, for a closed hollow."""
        return [o for o in context.selected_objects if store.is_nurbs(o) and _is_solid(o)]

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        self._take_picks(context)
        if self.faces or self.bodies:
            try:
                why = next((w for w in (self._check(o) for o, _ in self._items().values()) if w), "")
            except Exception as exc:
                why = str(exc)
            if why:
                self.report({'WARNING'}, why)
                return {'CANCELLED'}
            return self.ask(context)
        if bpy.app.background:
            return self.execute(context)
        self._dragging = False
        self._hover = None
        self._picked = []               # [(body, face index)]
        context.window_manager.modal_handler_add(self)
        self._pick_hints(context)
        return {'RUNNING_MODAL'}

    # -- picking the faces ------------------------------------------------------
    def _pick_hints(self, context, msg=""):
        n = len(getattr(self, "_picked", []))
        state = [f"{n} face{'s' if n != 1 else ''} to take away" if n else "click the faces to take away"]
        hints.show(context, self.bl_label, state + ([msg] if msg else []),
                   [("", [hints.note("a click on a picked face takes it back"),
                          hints.note("Enter with none: a closed hollow of the selected solids")])],
                   [("MOUSE_LMB", "Pick / unpick a face"), ("EVENT_RETURN", "Thickness"), ("EVENT_ESC", "Cancel")])

    def _show_picked(self):
        regs = []
        for o, i in self._picked:
            try:
                h = face_highlight(o, i, self.MAX_FILL)
            except Exception as exc:
                print("NURBS shell pick:", exc)
                continue
            regs.append((np.asarray(h["tris"], float).reshape(-1, 3, 3), _segments(h["lines"]), False))
        draw.pick_overlay["regions"] = regs

    def _end_pick(self, context):
        draw.pick_overlay["regions"] = []
        super()._end_pick(context)

    def modal(self, context, event):
        if getattr(self, "_dragging", False):
            return ValueDrag.modal(self, context, event)
        if self._view_event(context, event):        # the view turns / pans / zooms, the gizmo and sidebar work
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
            why = self._check(hit[0])
            if why:
                self._pick_hints(context, why)
                return {'RUNNING_MODAL'}
            if hit in self._picked:
                self._picked.remove(hit)
            else:
                self._picked.append(hit)
            self._show_picked()
            self._pick_hints(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in ('RET', 'NUMPAD_ENTER', 'SPACE') and event.value == 'PRESS':
            picked = [(o, face_frame(o, i)[0]) for o, i in self._picked]
            bodies = [] if picked else [o.name for o in self._whole_bodies(context)]
            if not picked and not bodies:
                self._pick_hints(context, "pick a face (or select a solid for a closed hollow)")
                return {'RUNNING_MODAL'}
            self._end_pick(context)
            objs = list(dict.fromkeys(o for o, _ in picked)) or [bpy.data.objects[n] for n in bodies]
            _select_only(context, objs)
            # the value stage can't start inside this loop: start the command again with the faces
            bpy.ops.nproto.shell('INVOKE_DEFAULT', faces=_faces_json(picked) if picked else "",
                                 bodies=json.dumps(bodies) if bodies else "", side=self.side)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    # -- the value stage (valuedrag.py) -----------------------------------------
    def drag_setup(self, context, mouse):
        """The inner edges' moves per unit thickness (wall_edges) and the arrow
        on the first removed face (no face: on the body's first face)."""
        self._walls = []
        regs = []
        first = None
        for o, pts in self._items().values():
            idx = [_face_index(o, p) for p in pts]
            self._walls += wall_edges(o, idx)
            for i in idx:
                h = face_highlight(o, i, self.MAX_FILL)
                regs.append((np.asarray(h["tris"], float).reshape(-1, 3, 3), _segments(h["lines"]), False))
            if first is None:
                first = (o, idx[0] if idx else 0)
        if first is None:
            return False
        self._B, self._n = face_centre(*first)
        draw.pick_overlay["regions"] = regs
        self._base_at_mouse = False
        return True

    def _sign(self):
        return 1.0 if self.side == 'OUTSIDE' else -1.0

    def drag_handles(self):
        return [{"field": "thickness", "kind": "arrow", "origin": self._B, "dir": self._sign() * self._n,
                 "param": self.thickness, "base": self._B, "to_value": lambda t: max(0.0, t)}]

    def drag_preview(self):
        s = self._sign() * self.thickness
        lines, tris = [], []
        for P, D, rim in self._walls:
            Q = P + s * D
            lines.append(Q)
            if rim:                     # the open rim: the wall's cut face
                L, T = ruled_preview([P, Q], ribs=2)
                lines += L[2:]
                tris.append(T)
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3))}

    def drag_check(self):
        if self.thickness <= 0.0:
            return "Thickness 0: drag the arrow into the body, or type a thickness"
        return ""

    def drag_key(self, context, event, ch):
        if ch == "f":
            self.side = 'INSIDE' if self.side == 'OUTSIDE' else 'OUTSIDE'
            return True
        return False

    def drag_rows(self):
        return [("Side", [hints.key("F", "walls outside" if self.side == 'INSIDE' else "walls inside")])]

    def drag_state(self):
        out = ["walls outside: the body becomes the cavity" if self.side == 'OUTSIDE' else "walls inside"]
        if not self.faces:
            out.append("closed hollow")
        return out

    def drag_handle_hints(self):
        return [hints.key("LMB drag", "the arrow (the wall's thickness)")]

    def _drag_end(self, context):
        draw.pick_overlay["regions"] = []
        super()._drag_end(context)

    # -- making it --------------------------------------------------------------
    def execute(self, context):
        draw.pick_overlay["regions"] = []
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._take_picks(context)
        if not self.faces and not self.bodies:          # a script with a solid selected: a closed hollow
            self.bodies = json.dumps([o.name for o in self._whole_bodies(context)])
        try:
            items = self._items()
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Shell: {exc}")
            return {'CANCELLED'}
        if not items:
            self.report({'WARNING'}, "Pick the faces to take away from a solid (Face mode, or click them)")
            return {'CANCELLED'}
        if self.thickness <= 0.0:
            self.report({'WARNING'}, "Shell: the thickness is 0")
            return {'CANCELLED'}
        hist = features.history_enabled(context)
        done = []
        for o, pts in items.values():
            why = self._check(o)
            if why:
                self.report({'ERROR'}, f"Shell: {why}")
                continue
            M = geom.world_matrix(o)
            frames = [face_frame(o, _face_index(o, p)) for p in pts]
            try:
                info, breps = jobs.run("shell", {"target": o}, ["result"],
                                       face_pts=[[float(x) for x in f[1]] for f in frames],
                                       thickness=float(self.thickness), side=self.side, into="target")
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"Shell on {o.name} failed: {exc}")
                continue
            pre = o[store.PROP_BREP]
            if not hist and features.has_history(o):
                features.clear_history(o)
                self.report({'INFO'}, f"{o.name}: construction history removed")
            store.set_geometry_text(o, breps["result"], context, keep_untrimmed=False)
            if hist:
                features.push_step(o, "shell", [], {
                    "faces_local": [[float(x) for x in f[0]] for f in frames],
                    "thickness": float(self.thickness), "shell_side": self.side}, base_text=pre)
            if subsel.picks(o, "FACE"):
                subsel.set_picks(o, "FACE", [])           # the picked faces are gone
            done.append(o)
            what = (f"{len(frames)} face{'s' if len(frames) != 1 else ''} open" if frames else "closed hollow")
            self.report({'INFO'}, f"{o.name}: shelled {self.thickness:.4g} "
                                  f"{'outside' if self.side == 'OUTSIDE' else 'inside'}, {what} · "
                                  f"volume {info.get('volume', 0):.6g}")
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}


def apply_shell(obj, text, inputs, p):
    """Shell on rebuild: the faces at the stored points taken away again and
    the rest made walls of the stored thickness."""
    M = geom.world_matrix(obj)
    pts = [list(M @ Vector(q)) for q in p.get("faces_local", [])]
    _, breps = jobs.run("shell", {"target": (text, jobs.xform(obj))}, ["result"], face_pts=pts,
                        thickness=float(p["thickness"]), side=p.get("shell_side", "INSIDE"), into="target")
    return breps["result"]


def _shell_label(p):
    n = len(p.get("faces_local", []))
    return (f"Shell · {float(p.get('thickness', 0)):.3g}" + (" out" if p.get("shell_side") == "OUTSIDE" else "")
            + (f" · {n} face{'s' if n != 1 else ''} open" if n else " · closed"))


def _register_step():
    from . import steps
    steps.APPLY["shell"] = apply_shell
    steps.LABEL["shell"] = _shell_label


_register_step()

classes = (NPROTO_OT_shell,)
