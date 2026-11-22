"""Move by two points, and copies along the move (hold G).

Tap G and Blender's own Move runs, as it always did. Hold G (longer than the
preference's Hold Time, 300 ms by default) and this tool runs instead: a
move from a base point to a target, both picked with the sketch tools'
snapping (curve and edge ends, middles, face centres, CVs, mesh vertices and
edges; Ctrl the grid, Alt nothing, X / Y / Z along that axis from the base,
Shift+X / Y / Z in the plane of the other two). A typed number is the
distance along the direction the mouse shows (or the locked axis).

Copies: typed with a leading * or /, as SketchUp and 3ds Max take them.
  *n   n copies, each one move further on (the original stays)
  /n   n copies spaced evenly from the base to the target (the last on it)
NURBS objects and curves are copied as a linear Pattern (pattern.py): each
copy has the source's exact B-rep, follows it through construction history,
and Edit Pattern changes the count later. Other objects are duplicated.
A * or / anywhere else in a number is arithmetic (10*2).

P: put the objects' origins (pivots) at the base point too, as Move Pivot
does, so they end up on the target.

In Edit Mode the selected vertices / CVs / points move (a CV cage's surface
follows); copies are an Object Mode thing there.

Nothing changes until the target is clicked (or Enter / Space): the tool
then moves (or copies) in one undo step, and the Adjust Last Operation
panel changes the offset, the copies and their count afterwards.
"""

import time

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatVectorProperty, IntProperty
from mathutils import Vector

from . import draw, hints, pattern, snapping
from .drawtools import REFRESH_KEYS, SketchMixin, parse_length

EPS = 1e-12
HOLD_MS = 300                   # default hold time: a press longer than this starts the NURBS tool
MAX_PREVIEW_PTS = 20000         # Edit Mode: more moved points than this are not drawn one by one
COPY_ITEMS = [("NONE", "Move", "Move the selection"),
              ("ARRAY", "Copies (*)", "Copies one move apart: the first on the target, the next one move on"),
              ("DIVIDE", "Copies Between (/)", "Copies spread evenly from the base point to the target, "
                                               "the last on the target")]
COL_LINE = (1.0, 1.0, 1.0, 0.85)
COL_BASE = (0.6, 0.6, 0.6, 0.9)
COL_COPY = (1.0, 0.62, 0.12, 0.9)
EDIT_MODES = {'EDIT_MESH', 'EDIT_CURVE', 'EDIT_SURFACE', 'EDIT_LATTICE'}


# --------------------------------------------------------------------------
# Pure helpers (tested headless)
# --------------------------------------------------------------------------

def parse_copies(text):
    """'*5' -> ('ARRAY', 5), '/4' -> ('DIVIDE', 4); anything else (an unfinished
    '*', a number, 0 copies) -> None."""
    t = (text or "").strip()
    if len(t) < 2 or t[0] not in "*/":
        return None
    try:
        n = int(t[1:].strip())
    except ValueError:
        return None
    if n < 1:
        return None
    return ("ARRAY" if t[0] == "*" else "DIVIDE"), n


def copy_offsets(d, mode, n):
    """The world offsets of the copies (the original's left out)."""
    d = np.asarray(d, float)
    if mode == "ARRAY":
        return [k * d for k in range(1, n + 1)]
    if mode == "DIVIDE":
        return [d * (k / n) for k in range(1, n + 1)]
    return []


def pattern_params(base, d, mode, n):
    """The linear Pattern (pattern.py params) that makes those copies."""
    d = np.asarray(d, float)
    L = float(np.linalg.norm(d))
    x = d / L
    return {"v": 2, "mode": "LINEAR", "space": "SPACING" if mode == "ARRAY" else "EXTENT", "turn": 0.0,
            "center_point": [float(c) for c in base], "frame_x": [float(c) for c in x],
            "frame_z": [float(c) for c in pattern._perp(x)],
            "count_x": int(n) + 1, "distance_x": L, "count_y": 1, "distance_y": 0.0, "count_z": 1, "distance_z": 0.0}


def roots(objs):
    """The objects none of whose parents is among them (a child moves with its parent)."""
    names = {o.name for o in objs}
    out = []
    for o in objs:
        p, inside = o.parent, False
        while p is not None:
            if p.name in names:
                inside = True
                break
            p = p.parent
        if not inside:
            out.append(o)
    return out


def move_objects(objs, d):
    """Each of objs (roots) moved by the world vector d."""
    d = Vector(tuple(map(float, d)))
    for o in objs:
        M = o.matrix_world.copy()
        M.translation = M.translation + d
        o.matrix_world = M


def _duplicate(context, src, off):
    obj = src.copy()
    if src.data is not None:
        obj.data = src.data.copy()
    colls = list(src.users_collection) or [context.collection or context.scene.collection]
    for c in colls:
        c.objects.link(obj)
    M = src.matrix_world.copy()
    M.translation = M.translation + Vector(tuple(map(float, off)))
    obj.matrix_world = M
    return obj


def make_copies(context, objs, base, d, mode, n):
    """Copies of objs (roots) along d: NURBS objects and curves as one linear
    Pattern, other objects duplicated. -> (copies, errors)."""
    made, errors = [], []
    nurbs = [o for o in objs if pattern.usable(o)]
    plain = [o for o in objs if not pattern.usable(o)]
    if nurbs:
        m, e = pattern.make_pattern(context, nurbs, pattern_params(base, d, mode, n))
        made += m
        errors += e
    for o in plain:
        for off in copy_offsets(d, mode, n):
            try:
                made.append(_duplicate(context, o, off))
            except Exception as exc:
                errors.append(f"{o.name}: {exc}")
    return made, errors


def edit_points(context):
    """World positions of the selected points of the objects in Edit Mode."""
    pts = []
    for o in context.objects_in_mode:
        M = o.matrix_world
        if o.type == 'MESH':
            bm = bmesh.from_edit_mesh(o.data)
            pts += [M @ v.co for v in bm.verts if v.select]
        elif o.type in {'CURVE', 'SURFACE'}:
            for sp in o.data.splines:
                for b in sp.bezier_points:
                    if b.select_control_point:
                        pts.append(M @ b.co)
                pts += [M @ p.co.xyz for p in sp.points if p.select]
        elif o.type == 'LATTICE':
            pts += [M @ p.co_deform for p in o.data.points if p.select]
    return np.array([tuple(p) for p in pts], float).reshape(-1, 3)


def move_edit_points(context, d):
    """The selected points of the objects in Edit Mode moved by the world
    vector d (a Bezier point's handles with it). -> how many moved."""
    d = Vector(tuple(map(float, d)))
    n = 0
    for o in context.objects_in_mode:
        dl = o.matrix_world.inverted().to_3x3() @ d       # (an affine map: the offset's local form)
        if o.type == 'MESH':
            bm = bmesh.from_edit_mesh(o.data)
            vs = [v for v in bm.verts if v.select]
            for v in vs:
                v.co += dl
            if vs:
                bmesh.update_edit_mesh(o.data, loop_triangles=True, destructive=False)
            n += len(vs)
        elif o.type in {'CURVE', 'SURFACE'}:
            for sp in o.data.splines:
                for b in sp.bezier_points:
                    if b.select_control_point:
                        b.co += dl
                        b.handle_left += dl
                        b.handle_right += dl
                        n += 1
                    else:
                        if b.select_left_handle:
                            b.handle_left += dl
                            n += 1
                        if b.select_right_handle:
                            b.handle_right += dl
                            n += 1
                for p in sp.points:
                    if p.select:
                        p.co = (*(p.co.xyz + dl), p.co[3])
                        n += 1
            o.data.update_tag()
        elif o.type == 'LATTICE':
            for p in o.data.points:
                if p.select:
                    p.co_deform = p.co_deform + dl
                    n += 1
            o.data.update_tag()
    return n


def hold_ms():
    p = hints._prefs()
    return int(getattr(p, "hold_ms", HOLD_MS)) if p is not None else HOLD_MS


def hold_enabled():
    p = hints._prefs()
    return bool(getattr(p, "hold_transform", True)) if p is not None else True


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------

class NPROTO_OT_move(SketchMixin, bpy.types.Operator):
    """Move from a base point to a target, both snapped (ends, middles, CVs,
    vertices, the grid); type a distance, *n for n copies, /n for copies in
    between. Hold G to start it"""
    bl_idname = "nproto.move"
    bl_label = "Move by Points"
    bl_options = {'REGISTER', 'UNDO'}

    face_planes = False         # a free point lies on the view's drawing plane through the base

    base: FloatVectorProperty(name="Base Point", size=3, subtype='XYZ', unit='LENGTH',
                              description="Where the move starts (world)")
    offset: FloatVectorProperty(name="Move", size=3, subtype='TRANSLATION', unit='LENGTH',
                                description="From the base point to the target (world)")
    copies: EnumProperty(name="Copies", items=COPY_ITEMS, default="NONE")
    count: IntProperty(name="Count", default=1, min=1, soft_max=50, max=pattern.MAX_COPIES,
                       description="How many copies (the original not counted)")
    set_pivot: BoolProperty(name="Pivot to Base Point", default=False,
                            description="Put the objects' origins at the base point first (Move Pivot), "
                                        "so they end up on the target")

    @classmethod
    def poll(cls, context):
        return context.mode in {"OBJECT"} | EDIT_MODES

    def draw(self, context):
        lay = self.layout
        lay.prop(self, "offset")
        if context.mode == 'OBJECT':
            lay.prop(self, "copies")
            if self.copies != "NONE":
                lay.prop(self, "count")
            lay.prop(self, "set_pivot")

    # -- doing it ---------------------------------------------------------------
    def execute(self, context):
        d = np.array(self.offset, float)
        if context.mode in EDIT_MODES:
            if self.copies != "NONE":
                self.report({'WARNING'}, "Move: copies are made in Object Mode (Edit Mode moves the points)")
            if np.linalg.norm(d) <= EPS:
                return {'CANCELLED'}
            n = move_edit_points(context, d)
            if not n:
                self.report({'WARNING'}, "Move: nothing selected")
                return {'CANCELLED'}
            self.report({'INFO'}, f"Move: {n} point(s) moved {self._len_text(context, d)}")
            return {'FINISHED'}
        objs = roots(list(context.selected_objects))
        if not objs:
            self.report({'WARNING'}, "Move: select the objects to move")
            return {'CANCELLED'}
        if np.linalg.norm(d) <= EPS:
            return {'CANCELLED'}
        if self.set_pivot:
            from . import pivottool
            for o in objs:
                why = pivottool.refusal(o)
                if why:
                    self.report({'WARNING'}, "Move: pivot not moved, " + why)
                    continue
                try:
                    pivottool.move_origin(o, np.array(self.base, float), context.view_layer)
                except Exception as exc:
                    self.report({'WARNING'}, f"Move: pivot of {o.name} not moved: {exc}")
        if self.copies == "NONE":
            move_objects(objs, d)
            self.report({'INFO'}, f"Move: {len(objs)} object(s) moved {self._len_text(context, d)}")
            return {'FINISHED'}
        n = int(self.count)
        if n * len(objs) > pattern.MAX_COPIES:
            self.report({'ERROR'}, f"Move: {n * len(objs)} copies, at most {pattern.MAX_COPIES} at once")
            return {'CANCELLED'}
        act = context.view_layer.objects.active
        made, errors = make_copies(context, objs, np.array(self.base, float), d, self.copies, n)
        for e in errors:
            self.report({'ERROR'}, "Move: " + e)
        if not made:
            return {'CANCELLED'}
        for o in made:
            try:
                o.select_set(True)
            except RuntimeError:
                pass
        if act is not None:
            context.view_layer.objects.active = act
        self.report({'INFO'}, f"Move: {len(made)} cop{'y' if len(made) == 1 else 'ies'}")
        return {'FINISHED'}

    @staticmethod
    def _len_text(context, d):
        from .valuedrag import fmt_length
        try:
            return fmt_length(context, float(np.linalg.norm(d)))
        except Exception:
            return f"{float(np.linalg.norm(d)):.4g}"

    # -- starting -------------------------------------------------------------
    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Move: run it in a 3D View")
            return {'CANCELLED'}
        self._edit = context.mode in EDIT_MODES
        if self._edit:
            for o in context.objects_in_mode:
                try:
                    o.update_from_editmode()
                except Exception:
                    pass
            self._pts = edit_points(context)
            if not len(self._pts):
                self.report({'WARNING'}, "Move: nothing selected")
                return {'CANCELLED'}
            self._objs = []
            self._center = self._pts.mean(axis=0)
            self._pv = []
        else:
            self._objs = roots(list(context.selected_objects))
            if not self._objs:
                self.report({'WARNING'}, "Move: select the objects to move")
                return {'CANCELLED'}
            self._pts = np.zeros((0, 3))
            self._center = np.mean([np.array(o.matrix_world.translation, float) for o in self._objs], axis=0)
            try:
                self._pv = pattern._preview_sources(self._objs)
            except Exception as exc:
                print("NURBS Move preview:", exc)
                self._pv = []
        self._base_pt = None        # the base point, once picked
        self._target_pt = None
        self._lock = None
        self._buf = ""              # being typed: a distance, or (a leading * or /) the copies
        self._dist = None           # a distance typed and fixed (Tab / Enter)
        self._copy = None           # (mode, n) fixed
        self._pivot = False
        self.sketch_start(context)
        self._plane = self._view_plane(context, self._center)
        context.window_manager.modal_handler_add(self)
        self._refresh(context, event)
        return {'RUNNING_MODAL'}

    def _snap_targets(self, context):
        """As the sketch tools', and the CVs of the NURBS objects being moved
        snap whether or not their net is shown (a base point on a CV)."""
        moving = {o.name for o in getattr(self, "_objs", ())}
        return snapping.collect(context.visible_objects,
                                lambda o: o.name in moving or draw.cv_net_shown(context, o),
                                snapping.sketch_mesh_elements)

    # -- typing ---------------------------------------------------------------
    def typing_ready(self):
        return getattr(self, "_base_pt", None) is not None

    def _live_dist(self, context):
        if self._buf and self._buf[0] not in "*/":
            try:
                return parse_length(context, self._buf)
            except Exception:
                return self._dist
        return self._dist

    def _live_copies(self):
        if self._buf and self._buf[0] in "*/":
            return parse_copies(self._buf) or self._copy
        return self._copy

    def _fix_buffer(self, context):
        """Tab / Enter: what is typed becomes the distance or the copies."""
        if not self._buf:
            return False
        if self._buf[0] in "*/":
            c = parse_copies(self._buf)
            if c is not None and not self._edit:
                self._copy = c
        else:
            L = self._live_dist(context)
            if L is not None:
                self._dist = L
        self._buf = ""
        return True

    # -- picking --------------------------------------------------------------
    def _target(self, context, event):
        """(point, snap) of the point under the mouse, the typed distance applied."""
        through = self._base_pt if self._base_pt is not None else self._center
        self._plane = self._view_plane(context, through)
        P, snap = self.pick_point(context, event, prev=self._base_pt)
        P = None if P is None else np.asarray(P, float)
        L = self._live_dist(context) if self._base_pt is not None else None
        if L is None:
            return P, snap
        B = self._base_pt
        lk = self._lock
        if lk in ("X", "Y", "Z"):
            e = self._axes(context)[lk]
            e = e / (np.linalg.norm(e) or 1.0)
            if P is not None and float((P - B) @ e) < 0:
                e = -e
        elif P is not None and np.linalg.norm(P - B) > EPS:
            e = (P - B) / np.linalg.norm(P - B)
        else:
            e = self._plane[2]
        return B + L * e, None

    # -- display ----------------------------------------------------------------
    def _offsets(self, d):
        c = None if self._edit else self._live_copies()
        return copy_offsets(d, *c) if c else [d]

    def _status(self, context):
        what = (f"{len(self._pts)} point(s)" if self._edit else
                self._objs[0].name + (f" +{len(self._objs) - 1}" if len(self._objs) > 1 else ""))
        c = None if self._edit else self._live_copies()
        L = self._live_dist(context)
        if self._base_pt is None:
            step = "click the base point"
        else:
            step = "click the target"
        state = [what, step]
        if L is not None:
            state.append("distance " + self._len_text(context, np.array([L, 0.0, 0.0])))
        if self._buf:
            state.append(f"typed {self._buf}_")
        if c:
            state.append(f"{c[1]} cop{'y' if c[1] == 1 else 'ies'}" + (" between" if c[0] == "DIVIDE" else ""))
        if self._pivot:
            state.append("pivot to base")
        state.append(self.lock_text().strip(" ·"))
        ready = self._base_pt is not None
        put = [hints.key("C", "3D cursor"), hints.key("W", "world origin")]
        if ready:
            put.append(hints.key("B", "pick the base again"))
        rows = [("Point", put),
                ("Lock", self.direction_hints()),
                ("Snap", self.snap_hints()),
                ("Type", [hints.key("0-9", "distance", on=None if ready else False),
                          hints.key("*n", "n copies", on=(True if c and c[0] == "ARRAY" else None)
                                    if ready and not self._edit else False),
                          hints.key("/n", "n copies between", on=(True if c and c[0] == "DIVIDE" else None)
                                    if ready and not self._edit else False),
                          hints.key("Tab", "fix the typed value", on=None if self._buf else False)])]
        if not self._edit:
            rows.append(("Pivot", [hints.toggle("P", "origins to the base point", self._pivot)]))
        rows.append(("Move", [hints.key("Backspace", "step back"), hints.key("Enter", "done")]))
        hints.show(context, "Move", state, rows,
                   [("MOUSE_LMB", "Base / target"), ("MOUSE_RMB", "Cancel"), ("EVENT_ESC", "Cancel")])

    def _preview(self, context, hover):
        ov = draw.sketch_overlay
        P, snap = hover
        ov["snap"] = Vector(tuple(P)) if P is not None and snap is not None else None
        ov["snap_label"] = snap[3] if P is not None and snap is not None else None
        ov["cursor"] = Vector(tuple(P)) if P is not None else None
        V = lambda p: Vector(tuple(map(float, p)))
        refs, cvs = [], []
        pv = None
        B = self._base_pt
        if B is not None and P is not None and np.linalg.norm(P - B) > EPS:
            d = P - B
            offs = self._offsets(d)
            refs.append(([V(B), V(B + offs[-1] if len(offs) > 1 else P)], COL_LINE, 2.0))
            from .workplanes import marker_size
            c = 0.1 * marker_size(self._win(context)[1], B)
            refs += [([V(B - c * e), V(B + c * e)], COL_BASE, 1.5) for e in np.eye(3)]
            for off in offs[:-1] if len(offs) > 1 else []:
                Q = B + off
                refs += [([V(Q - 0.6 * c * e), V(Q + 0.6 * c * e)], COL_COPY, 1.5) for e in np.eye(3)]
            if self._edit:
                if len(self._pts) <= MAX_PREVIEW_PTS:
                    cvs = [V(q) for q in self._pts + d]
            else:
                pv = self._ghosts(offs)
            label = self._len_text(context, d)
            cp = None if self._edit else self._live_copies()
            if cp:
                label += f"  ×{cp[1]}" if cp[0] == "ARRAY" else f"  /{cp[1]}"
            if pv is not None:
                pv["labels"] = [(V(P), label, True)]
            else:
                pv = {"lines": [], "tris": None, "labels": [(V(P), label, True)]}
        elif B is not None:
            from .workplanes import marker_size
            c = 0.1 * marker_size(self._win(context)[1], B)
            refs += [([V(B - c * e), V(B + c * e)], COL_BASE, 1.5) for e in np.eye(3)]
        ov["refs"] = refs
        ov["cvs"] = cvs
        draw.pick_overlay["preview"] = pv
        self.axis_guides(context, B)
        self._status(context)
        context.area.tag_redraw()

    def _ghosts(self, offs):
        """The objects drawn where the move (or each copy) puts them."""
        lines, tris = [], []
        k = len(offs)
        n_lines = sum(len(L) for _, L, _, _ in self._pv) * k
        n_tris = sum(len(T) // 3 for _, _, T, _ in self._pv) * k
        for _src, L, Tw, box in self._pv:
            for off in offs:
                if n_lines <= pattern.PREVIEW_LINES:
                    lines += [Q + off for Q in L]
                else:
                    lines += pattern._box_lines(box + off)
                if n_tris <= pattern.PREVIEW_TRIS and len(Tw):
                    tris.append(Tw + off)
                elif not len(Tw):
                    if not L:
                        lines += pattern._box_lines(box + off)
                else:
                    tris.append(pattern._box_tris(box + off))
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3))}

    def _refresh(self, context, event):
        h = self._target(context, event)
        self._target_pt = h[0]
        self._preview(context, h)
        return h

    def _end(self, context):
        draw.pick_overlay["preview"] = None
        self.sketch_end(context)
        context.area.tag_redraw()

    # -- modal ------------------------------------------------------------------
    def _place(self, context, event, P):
        """A point picked (a click, C, W): the base, or the target (done)."""
        if P is None:
            return None
        P = np.asarray(P, float)
        if self._base_pt is None:
            self._base_pt = P
            if self._lock in ("X", "Y", "Z", "-X", "-Y", "-Z"):
                self._lock = None           # an axis lock held for the base: the target picks afresh
            self._refresh(context, event)
            return None
        return self._finish(context, P)

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        et, ev = event.type, event.value
        if et == 'G':                   # the held key that started it (release, key repeat)
            return {'RUNNING_MODAL'}
        ch = self.typed_char(event)
        if self.typing_ready() and ch and not (event.ctrl or event.alt or event.oskey):
            copies_typed = self._buf[:1] in ("*", "/")
            if (not self._buf and ch in "0123456789.-+*/") \
                    or (self._buf and not copies_typed and (ch in "0123456789.-+*/'\" " or ch.isalpha())) \
                    or (copies_typed and ch.isdigit()):
                if ch in "*/" and not self._buf and self._edit:
                    self.report({'INFO'}, "Move: copies are made in Object Mode")
                    return {'RUNNING_MODAL'}
                self._buf += ch
                self._refresh(context, event)
                return {'RUNNING_MODAL'}
        if et in REFRESH_KEYS or self.snap_key(context, event):
            self._refresh(context, event)
        elif et == 'LEFTMOUSE' and ev == 'PRESS':
            P, _ = self._target(context, event)
            r = self._place(context, event, P)
            if r is not None:
                return r
        elif ev == 'PRESS' and et in {'X', 'Y', 'Z'} and not event.ctrl:
            self.axis_key(event)
            self._refresh(context, event)
        elif ev == 'PRESS' and et in {'C', 'W'} and not (event.ctrl or event.alt or event.oskey):
            P = np.array(context.scene.cursor.location, float) if et == 'C' else np.zeros(3)
            r = self._place(context, event, P)
            if r is not None:
                return r
        elif ev == 'PRESS' and et == 'B' and self._base_pt is not None:
            self._base_pt = None
            self._buf, self._dist = "", None
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'P' and not self._edit:
            self._pivot = not self._pivot
            self._refresh(context, event)
        elif ev == 'PRESS' and et == 'TAB':
            if self._fix_buffer(context):
                self._refresh(context, event)
        elif ev == 'PRESS' and et == 'BACK_SPACE':
            if self._buf:
                self._buf = self._buf[:-1]
            elif self._copy is not None:
                self._copy = None
            elif self._dist is not None:
                self._dist = None
            elif self._base_pt is not None:
                self._base_pt = None
            self._refresh(context, event)
        elif ev == 'PRESS' and et in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
            if self._buf and et != 'SPACE' and self._buf[0] in "*/":
                self._fix_buffer(context)           # the copies typed: then click (or Enter) for the target
                self._refresh(context, event)
            elif self._base_pt is not None:
                P, _ = self._target(context, event)
                if P is not None:
                    return self._finish(context, P)
        elif ev == 'PRESS' and et in {'ESC', 'RIGHTMOUSE'}:
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context, P):
        c = None if self._edit else self._live_copies()
        d = np.asarray(P, float) - self._base_pt
        self._end(context)
        if np.linalg.norm(d) <= EPS:
            return {'CANCELLED'}
        self.base = tuple(map(float, self._base_pt))
        self.offset = tuple(map(float, d))
        self.copies, self.count = (c[0], c[1]) if c else ("NONE", 1)
        self.set_pivot = bool(self._pivot)
        return self.execute(context)


# --------------------------------------------------------------------------
# Hold G: Blender's Move on a tap, this one on a hold
# --------------------------------------------------------------------------

HOLD_TOOLS = {"MOVE": ("transform.translate", NPROTO_OT_move.bl_idname)}


def _run(idname):
    mod, name = idname.split(".")
    return getattr(getattr(bpy.ops, mod), name)('INVOKE_DEFAULT')


class NPROTO_OT_transform_hold(bpy.types.Operator):
    """Tap: Blender's own Move. Hold: Move by Points (base point, target,
    copies) -- the add-on's Hold Time preference sets how long"""
    bl_idname = "nproto.transform_hold"
    bl_label = "Move (tap) / Move by Points (hold)"
    bl_options = {'INTERNAL'}

    kind: EnumProperty(items=[("MOVE", "Move", "")], default="MOVE", options={'HIDDEN'})

    def _native(self):
        try:
            _run(HOLD_TOOLS[self.kind][0])
        except RuntimeError as exc:
            print("NURBS hold key:", exc)
        return {'FINISHED'}

    def _stop(self, context):
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def invoke(self, context, event):
        if event.is_repeat:
            return {'CANCELLED'}
        ours = HOLD_TOOLS[self.kind][1]
        mod, name = ours.split(".")
        if not hold_enabled() or not getattr(getattr(bpy.ops, mod), name).poll():
            return self._native()
        self._key = event.type
        self._t0 = time.monotonic()
        self._timer = context.window_manager.event_timer_add(0.02, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == self._key:
            if event.value == 'RELEASE':
                self._stop(context)
                return self._native()
            return {'RUNNING_MODAL'}                # key repeat while held
        if event.type == 'TIMER':
            if (time.monotonic() - self._t0) * 1000.0 >= hold_ms():
                self._stop(context)
                try:
                    _run(HOLD_TOOLS[self.kind][1])
                except RuntimeError as exc:
                    print("NURBS hold key:", exc)
                return {'FINISHED'}
            return {'RUNNING_MODAL'}
        if event.type == 'ESC':
            self._stop(context)
            return {'CANCELLED'}
        if event.value == 'PRESS' and event.type not in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._stop(context)                     # another key while G is down (G X typed fast): a tap
            self._native()
            return {'FINISHED', 'PASS_THROUGH'}
        return {'PASS_THROUGH'}


def menu_transform(self, context):
    self.layout.operator(NPROTO_OT_move.bl_idname, icon='EMPTY_ARROWS')


MENUS = ("VIEW3D_MT_transform_object", "VIEW3D_MT_transform")
KEYMAPS = ("Object Mode", "Mesh", "Curve", "Lattice")       # where Blender binds G to Move

classes = (NPROTO_OT_move, NPROTO_OT_transform_hold)
_keymaps = []


def register():
    for m in MENUS:
        getattr(bpy.types, m).append(menu_transform)
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is not None:
        for name in KEYMAPS:
            km = kc.keymaps.new(name=name, space_type='EMPTY')
            kmi = km.keymap_items.new(NPROTO_OT_transform_hold.bl_idname, 'G', 'PRESS')
            kmi.properties.kind = "MOVE"
            _keymaps.append((km, kmi))


def unregister():
    for km, kmi in _keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _keymaps.clear()
    for m in MENUS:
        try:
            getattr(bpy.types, m).remove(menu_transform)
        except Exception:
            pass
