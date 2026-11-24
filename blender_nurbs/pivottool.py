"""Move Pivot: put an object's origin where you click, as Alias's and Maya's
Move Pivot do, in Object Mode and in Edit Mode.

The tool shows the pivot (the active object's origin) and moves it to the
point under the mouse, snapped like the sketch tools (drawtools.SketchMixin):
curve and edge ends, middles, face centres, CVs (always those of the
surfaces whose pivot moves: their net is drawn), mesh vertices and edges,
Ctrl the grid, Alt no snapping, X / Y / Z along that axis through the
pivot (the work plane's when one is active; Shift: in the plane of the other
two). A free point lies on the drawing plane through the pivot. Click to
put it there, or press and drag it along.
C, W and B put it on the 3D cursor, the world origin or the middle of the
object's bounds; in Edit Mode S puts it on the selection's middle.

Only the origin moves: the geometry stays where it is in the world (the data
moves the other way, as Blender's Set Origin does). Children stay in place.
A NURBS object's exact B-rep, its feature placement and everything built
from it follow through pivot.apply, as they do after Set Origin. In Edit
Mode on a NURBS surface's CV cage the surface's origin moves; the cage keeps
its own (cvs.py maps between the two).

Nothing changes until Enter (or Space, right-click): the tool then moves
every origin at once, one undo step. Backspace takes the last placement
back, Esc leaves everything as it was. With several objects, T switches
between moving every origin to the pivot and keeping their offsets.

Where to find it: the toolbar (after Blender's Transform, in Object and
Edit Mode: its click is the first placement, and when it is done the tool
before it comes back), hold "." (tap is Blender's Pivot Point pie,
movetool.HOLD_TOOLS), the right-click menus, Object > Transform and F3
("pivot" or "origin").
"""

import os

import bmesh
import bpy
from bpy.props import BoolProperty
import numpy as np
from mathutils import Matrix, Vector

from . import draw, hints, pivot, snapping, store
from .drawtools import REFRESH_KEYS, SketchMixin

KINDS = {'MESH', 'CURVE', 'SURFACE', 'LATTICE'}     # objects whose data moves under the origin
COL_START = (0.6, 0.6, 0.6, 0.8)
AX_COL = ((1.0, 0.25, 0.3, 1.0), (0.45, 0.9, 0.2, 1.0), (0.25, 0.55, 1.0, 1.0))
EPS = 1e-12


def owner(obj):
    """The object whose origin a Move Pivot on obj moves: a CV cage's surface, else obj."""
    from .cvs import CAGE_OF, is_cage
    if is_cage(obj):
        return bpy.data.objects.get(obj[CAGE_OF])
    return obj


def refusal(obj):
    """Why obj's origin can't be moved, or None."""
    if obj is None:
        return "nothing to move"
    if obj.type not in KINDS or obj.data is None:
        return f"{obj.name}: a {obj.type.lower()} has no geometry to keep in place"
    if obj.library is not None or obj.data.library is not None or obj.override_library is not None:
        return f"{obj.name}: linked from a library"
    if obj.data.users > 1:
        return f"{obj.name}: its {obj.type.lower()} data is shared by {obj.data.users} objects"
    return None


def _np(M):
    return np.array([list(r) for r in M], float)


def _transform_cage(surf, A):
    """The CV cage open on surf keeps what it restores (Cancel) and compares
    against in surf's local space: move those by A as well."""
    import json
    from . import cvs
    cage = cvs.cage_for(surf)
    if cage is None:
        return
    for k in (cvs.CAGE_ORIG, "nproto_cage_orig_untrim"):
        if isinstance(cage.get(k), str):
            cage[k] = pivot._transform_text(cage[k], A)
    raw = cage.get(cvs.CAGE_CVS0)
    if raw:
        P = np.array(json.loads(raw), float).reshape(-1, 3)
        cage[cvs.CAGE_CVS0] = json.dumps((P @ A[:3, :3].T + A[:3, 3]).tolist())


def move_origin(obj, P, view_layer=None):
    """Put obj's origin at the world point P; its geometry, its children and
    (NURBS) its exact B-rep stay where they are in the world. obj must be in
    Object Mode (a CV cage's surface always is). -> True when it moved."""
    vl = view_layer or bpy.context.view_layer
    nurbs = store.is_nurbs(obj)
    if nurbs:
        pivot.check(obj)                # its B-rep is in step with the mesh and the matrix now
    M = obj.matrix_world
    d = M.inverted() @ Vector(tuple(map(float, P)))       # the new origin in its local space
    size = max(max(abs(c) for c in obj.dimensions), 1.0)
    if d.length <= 1e-12 * size:
        return False
    A = Matrix.Translation(-d)          # local change: the data moves back by d
    rev = obj.get(store.PROP_REV)
    obj.data.transform(A, shape_keys=True)
    obj.data.update_tag()
    obj.location = obj.location + obj.matrix_basis.to_3x3() @ d     # basis @ T(d): only the location changes
    for c in obj.children:
        if c.parent_type == 'OBJECT':   # world = parent @ parent_inverse @ basis: keep it
            c.matrix_parent_inverse = A @ c.matrix_parent_inverse
    vl.update()
    if nurbs:
        if obj.get(store.PROP_REV) == rev:      # (else pivot's depsgraph handler followed in the update)
            pivot.apply(obj, _np(A))
        _transform_cage(obj, _np(A))
    return True


class NPROTO_OT_move_pivot(SketchMixin, bpy.types.Operator):
    """Move the pivot (object origin) without moving the geometry, as Alias's
    and Maya's Move Pivot: click or drag it onto a point, snapping to CVs,
    vertices, ends, middles and edges. Works in Object and Edit Mode"""
    bl_idname = "nproto.move_pivot"
    bl_label = "Move Pivot (Origin)"
    bl_options = {'REGISTER'}     # its own undo step (apply_targets): Edit Mode's would hold only the mesh

    from_tool: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})    # the toolbar tool's click started it

    face_planes = False         # a free point lies on the drawing plane through the pivot

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return context.area is not None and context.area.type == 'VIEW_3D' and obj is not None \
            and context.mode in {'OBJECT', 'EDIT_MESH', 'EDIT_CURVE', 'EDIT_SURFACE', 'EDIT_LATTICE'}

    # -- what moves ------------------------------------------------------------
    def _collect(self, context):
        """(objects whose origins move, the active one first; refusals)."""
        if context.mode == 'OBJECT':
            objs = [context.active_object] + [o for o in context.selected_objects if o != context.active_object]
        else:
            objs = [context.active_object] + [o for o in context.objects_in_mode if o != context.active_object]
        out, why, seen = [], [], set()
        for o in objs:
            if o is None:
                continue
            w = owner(o)
            if w is None or w.name in seen:
                continue
            seen.add(w.name)
            r = refusal(w)
            if r:
                why.append(r)
            else:
                out.append(w)
        return out, why

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Move Pivot: run it in a 3D View")
            return {'CANCELLED'}
        self._objs, why = self._collect(context)
        if not self._objs:
            self.report({'WARNING'}, "Move Pivot: " + (why[0] if why else "select an object"))
            return {'CANCELLED'}
        for r in why:
            self.report({'WARNING'}, f"Move Pivot skips {r}")
        for o in context.objects_in_mode:       # what is being edited snaps as it is now
            try:
                o.update_from_editmode()
            except Exception:
                pass
        self._start = [np.array(o.matrix_world.translation, float) for o in self._objs]
        self._pivot = self._start[0].copy()
        self._history = []          # pivots placed before (Backspace)
        self._drag = False
        self._together = True       # several objects: every origin to the pivot (T: keep the offsets)
        self._lock = None
        self.sketch_start(context)
        self._net_lines, self._net_pts = [], []
        for nu, nv, P in self._nets(context):          # the CVs that snap, drawn as CVs / Hulls draws them
            G = P.reshape(nu, nv, 3)
            V = lambda p: Vector(tuple(map(float, p)))
            self._net_lines += [([V(p) for p in G[i]], draw.COL_HULL, 1.5) for i in range(nu)]
            self._net_lines += [([V(p) for p in G[:, j]], draw.COL_HULL, 1.5) for j in range(nv)]
            self._net_pts += [V(p) for p in P]
        self._plane = self._view_plane(context, self._pivot)
        context.window_manager.modal_handler_add(self)
        hover = self._hover_pick(context, event)
        if self.from_tool and event.type == 'LEFTMOUSE' and event.value == 'PRESS' and hover[0] is not None:
            self._place(hover[0])           # the toolbar tool's click places the pivot (and may drag it)
            self._drag = True
        self._preview(context, hover)
        return {'RUNNING_MODAL'}

    # -- picking ----------------------------------------------------------------
    def _snap_targets(self, context):
        """As the sketch tools', and the CVs of the surfaces whose pivot moves
        snap whether or not their CV net is shown (a CV cage open on one too)."""
        moving = {o.name for o in getattr(self, "_objs", ())}
        return snapping.collect(context.visible_objects,
                                lambda o: o.name in moving or draw.cv_net_shown(context, o),
                                snapping.sketch_mesh_elements)

    def _nets(self, context):
        """World CV nets [(nu, nv, points)] of the moved NURBS objects whose net
        is not drawn already (CVs / Hulls, a CV cage): the tool draws them."""
        from .cvs import cage_for
        out = []
        for o in self._objs:
            if not store.is_nurbs(o) or draw.cv_net_shown(context, o) or cage_for(o) is not None:
                continue
            M = np.array(o.matrix_world, float)
            for net in store.display(o).get("nets", []):
                P = np.asarray(net["p"], float).reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
                out.append((net["nu"], net["nv"], P))
        return out

    def _hover_pick(self, context, event):
        self._plane = self._view_plane(context, self._pivot)    # (the view may have turned)
        P, snap = self.pick_point(context, event, prev=self._pivot)
        return (None if P is None else np.asarray(P, float)), snap

    def _selection_middle(self, context):
        """The middle of the selected vertices / points in Edit Mode (world), or None."""
        pts = []
        for o in context.objects_in_mode:
            M = o.matrix_world
            if o.type == 'MESH':
                bm = bmesh.from_edit_mesh(o.data)
                pts += [M @ v.co for v in bm.verts if v.select]
            elif o.type in {'CURVE', 'SURFACE'}:
                for sp in o.data.splines:
                    pts += [M @ p.co for p in sp.bezier_points if p.select_control_point]
                    pts += [M @ p.co.xyz for p in sp.points if p.select]
            elif o.type == 'LATTICE':
                pts += [M @ p.co_deform for p in o.data.points if p.select]
        if not pts:
            return None
        return np.mean(np.array([tuple(p) for p in pts], float), axis=0)

    def _bounds_middle(self):
        o = self._objs[0]
        C = np.array([tuple(o.matrix_world @ Vector(c)) for c in o.bound_box], float)
        return 0.5 * (C.min(axis=0) + C.max(axis=0))

    def _place(self, P):
        if P is None:
            return
        P = np.asarray(P, float)
        if np.linalg.norm(P - self._pivot) <= EPS:
            return
        self._history.append(self._pivot.copy())
        self._pivot = P

    def _targets(self):
        """[(object, new world origin)]."""
        off = self._pivot - self._start[0]
        return [(o, self._pivot if (self._together or i == 0) else s + off)
                for i, (o, s) in enumerate(zip(self._objs, self._start))]

    # -- display ----------------------------------------------------------------
    def _status(self, context):
        n = len(self._objs)
        moved = float(np.linalg.norm(self._pivot - self._start[0]))
        state = [self._objs[0].name + (f" +{n - 1}" if n > 1 else ""),
                 "drag the pivot" if self._drag else "click where the pivot goes",
                 f"moved {moved:.4g}" if moved > EPS else "", self.lock_text().strip(" ·")]
        put = [hints.key("C", "3D cursor"), hints.key("W", "world origin"), hints.key("B", "bounds middle")]
        if context.mode != 'OBJECT':
            put.append(hints.key("S", "selection middle"))
        rows = [("Put", put),
                ("Lock", [hints.key("X Y Z", "axis", on=True if self._lock in ("X", "Y", "Z") else None),
                          hints.key("Shift+X/Y/Z", "plane", on=True if self._lock in ("-X", "-Y", "-Z") else None)]),
                ("Snap", self.snap_hints()),
                ("Pivot", [hints.key("Backspace", "undo a placement", on=None if self._history else False),
                           hints.key("Enter", "done")]
                 + ([hints.cycle("T", ["together", "keep offsets"],
                                 "together" if self._together else "keep offsets", "several objects")]
                    if n > 1 else []))]
        hints.show(context, "Move Pivot", state, rows,
                   [("MOUSE_LMB", "Place / drag"), ("MOUSE_RMB", "Done"), ("EVENT_ESC", "Cancel")])

    def _preview(self, context, hover=None):
        ov = draw.sketch_overlay
        P, snap = hover if hover is not None else (None, None)
        ov["snap"] = Vector(tuple(P)) if P is not None and snap is not None else None
        ov["snap_label"] = snap[3] if P is not None and snap is not None else None
        ov["cursor"] = Vector(tuple(self._pivot))
        from .workplanes import marker_size
        L = marker_size(self._win(context)[1], self._pivot)
        R = self._objs[0].matrix_world.to_3x3().normalized()
        V = lambda p: Vector(tuple(map(float, p)))
        refs = list(self._net_lines)
        for k in range(3):
            a = np.array(R.col[k], float)
            refs.append(([V(self._pivot), V(self._pivot + L * a)], AX_COL[k], 3.0))
        targets = self._targets()
        c = 0.12 * L
        cross = lambda X: [([V(X - c * e), V(X + c * e)], COL_START, 1.5) for e in np.eye(3)]
        for i, ((o, Q), S) in enumerate(zip(targets, self._start)):
            if np.linalg.norm(Q - S) > EPS:
                refs.append(([V(S), V(Q)], COL_START, 1.0))     # from where it was
                refs += cross(S)
            if i:
                refs += cross(Q)                                # the other objects' new origins
        ov["refs"] = refs
        ov["cvs"] = list(self._net_pts)
        self.axis_guides(context, self._pivot)
        self._status(context)
        context.area.tag_redraw()

    # -- modal ------------------------------------------------------------------
    def modal(self, context, event):
        if self._drag and event.type in {'MOUSEMOVE', 'LEFTMOUSE'}:
            pass                        # a drag keeps its moves and its release, over the sidebar too
        elif self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        hover = lambda: self._hover_pick(context, event)
        et, ev = event.type, event.value
        if et in REFRESH_KEYS or self.snap_key(context, event):
            h = hover()
            if self._drag and et == 'MOUSEMOVE' and h[0] is not None:
                self._pivot = h[0]
            self._preview(context, h)
        elif et == 'LEFTMOUSE' and ev == 'PRESS':
            P, snap = hover()
            if P is not None:
                self._place(P)
                self._drag = True
            self._preview(context, (P, snap))
        elif et == 'LEFTMOUSE' and ev == 'RELEASE':
            self._drag = False
            self._preview(context, hover())
        elif ev == 'PRESS' and et in {'X', 'Y', 'Z'} and not event.ctrl:
            self.axis_key(event)
            self._preview(context, hover())
        elif ev == 'PRESS' and et in {'C', 'W', 'B', 'S'} and not (event.ctrl or event.alt):
            P = {"C": lambda: np.array(context.scene.cursor.location, float),
                 "W": lambda: np.zeros(3),
                 "B": self._bounds_middle,
                 "S": lambda: self._selection_middle(context) if context.mode != 'OBJECT' else None}[et]()
            if et == 'S' and P is None and context.mode != 'OBJECT':
                self.report({'INFO'}, "Move Pivot: nothing selected")
            self._place(P)
            self._preview(context, hover())
        elif ev == 'PRESS' and et == 'T' and len(self._objs) > 1:
            self._together = not self._together
            self._preview(context, hover())
        elif ev == 'PRESS' and et == 'BACK_SPACE' and self._history:
            self._pivot = self._history.pop()
            self._preview(context, hover())
        elif ev == 'PRESS' and et in {'RET', 'NUMPAD_ENTER', 'SPACE', 'RIGHTMOUSE'}:
            return self._finish(context)
        elif ev == 'PRESS' and et == 'ESC':
            self.sketch_end(context)
            self._tool_back(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _tool_back(self, context):
        """Started from the toolbar: the tool that was active before comes back
        (a pivot move is a one-off; the next click should select again)."""
        if self.from_tool:
            restore_tool(context)

    def _finish(self, context):
        self.sketch_end(context)
        self._tool_back(context)
        targets = [(o, Q) for (o, Q), S in zip(self._targets(), self._start) if np.linalg.norm(Q - S) > EPS]
        if not targets:
            return {'CANCELLED'}
        moved = apply_targets(context, targets, push=True)
        self.report({'INFO'}, f"Move Pivot: {len(moved)} origin(s) moved" if moved else "Move Pivot: nothing moved")
        return {'FINISHED'} if moved else {'CANCELLED'}


def apply_targets(context, targets, push=False):
    """Move each (object, world point) origin. -> names moved.
    In Edit Mode the objects are taken out of it for the move and put back
    in after; leaving a CV cage's Edit Mode here is not its Done (cvs._busy).
    With push the move is one undo step, made in Object Mode: an Edit Mode
    step holds only the mesh, so undoing it would put the mesh back under the
    moved origin and the geometry would jump; a CV cage's step would not
    hold its surface at all. One Ctrl+Z then takes the move back, in Edit
    Mode too."""
    from . import cvs
    edit = context.mode != 'OBJECT'
    busy = cvs._busy
    if edit:
        cvs._busy = True
        bpy.ops.object.mode_set(mode='OBJECT')
    moved = []
    try:
        for o, Q in targets:
            try:
                if move_origin(o, Q, context.view_layer):
                    moved.append(o.name)
            except Exception as exc:
                print(f"NURBS Move Pivot: {o.name}: {exc}")
    finally:
        try:
            if push and moved:
                bpy.ops.ed.undo_push(message="Move Pivot")
            if edit:
                bpy.ops.object.mode_set(mode='EDIT')
        finally:
            cvs._busy = busy
    return moved


def menu_transform(self, context):
    self.layout.separator()
    self.layout.operator(NPROTO_OT_move_pivot.bl_idname, icon='OBJECT_ORIGIN')


# Object > Transform and Edit Mode's Transform, and the right-click menus
# (Object Mode's has Blender's Set Origin; a surface's Edit Mode uses the curve's)
MENUS = ("VIEW3D_MT_transform_object", "VIEW3D_MT_transform", "VIEW3D_MT_object_context_menu",
         "VIEW3D_MT_edit_mesh_context_menu", "VIEW3D_MT_edit_curve_context_menu",
         "VIEW3D_MT_edit_lattice_context_menu")


# -- the toolbar tool -------------------------------------------------------------
# Blender's tools stay on until another is picked; this one hands the toolbar
# back when its Move Pivot ends. The tool before it is the last other one a
# 3D View drew with, per mode (_note_tool).

TOOL_ID = "nproto.move_pivot_tool"
TOOL_MODES = ('OBJECT', 'EDIT_MESH', 'EDIT_CURVE', 'EDIT_SURFACE', 'EDIT_LATTICE')
ICON = os.path.join(os.path.dirname(__file__), "icons", "ops.nproto.move_pivot")    # tools/make_pivot_icon.py
_last_tool = {}         # mode -> the tool id that was active before this one
ONE_SHOT = {TOOL_ID}    # toolbar tools that hand the toolbar back (aligntool adds Align Parts')
_draw_handle = None


def _active_tool(context):
    try:
        t = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
    except Exception:
        return None
    return t.idname if t is not None else None


def _note_tool():
    ctx = bpy.context
    t = _active_tool(ctx)
    if t and t not in ONE_SHOT:
        _last_tool[ctx.mode] = t


def restore_tool(context, tool_id=TOOL_ID):
    """Put back the tool that was active before the one-shot tool tool_id (if it still is)."""
    if _active_tool(context) != tool_id:
        return
    prev = _last_tool.get(context.mode, "builtin.select_box")
    try:
        bpy.ops.wm.tool_set_by_id(name=prev, space_type='VIEW_3D')
    except Exception as exc:
        print("NURBS Move Pivot: tool not put back:", exc)


def _tool_class(mode):
    class Tool(bpy.types.WorkSpaceTool):
        bl_space_type = 'VIEW_3D'
        bl_context_mode = mode
        bl_idname = TOOL_ID
        bl_label = "Move Pivot (Origin)"
        bl_description = ("Click where the object's pivot (origin) goes: it snaps to CVs, vertices, ends, middles "
                          "and edges; the geometry stays. Enter applies, Esc cancels. Also: hold .")
        bl_icon = ICON
        bl_keymap = ((NPROTO_OT_move_pivot.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'},
                      {"properties": [("from_tool", True)]}),)

        def draw_settings(context, layout, tool):
            layout.label(text="Click where the pivot goes · Enter applies · or hold the . key, any tool",
                         icon='OBJECT_ORIGIN')

    Tool.__name__ = "NPROTO_TL_move_pivot_" + mode.lower()
    return Tool


TOOLS = tuple(_tool_class(m) for m in TOOL_MODES)


def _after_transform(mode):
    """Move the tool from the toolbar's end to just after Blender's Transform.
    (register_tool's after= can't: for a tool outside a group it skips along
    the tool's own fields and lands after the separator that follows.)"""
    from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
    tools = ToolSelectPanelHelper._tool_class_from_space_type('VIEW_3D')._tools[mode]
    mine = next((i for i, it in enumerate(tools) if getattr(it, "idname", None) == TOOL_ID), None)
    at = next((i for i, it in enumerate(tools) if getattr(it, "idname", None) == "builtin.transform"), None)
    if mine is None or at is None:
        return                              # (stays at the end, after its separator)
    item = tools[mine]
    del tools[mine - 1:mine + 1]            # it and the separator register_tool put before it
    tools.insert(at + 1, item)


classes = (NPROTO_OT_move_pivot,)


def register():
    global _draw_handle
    for m in MENUS:
        menu = getattr(bpy.types, m, None)
        if menu is not None:
            menu.append(menu_transform)
    for t in TOOLS:
        try:
            bpy.utils.register_tool(t, separator=True)
            _after_transform(t.bl_context_mode)
        except Exception as exc:
            print("NURBS Move Pivot tool not in the toolbar:", t.bl_context_mode, exc)
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_note_tool, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    for t in TOOLS:
        try:
            bpy.utils.unregister_tool(t)
        except Exception:
            pass
    for m in MENUS:
        menu = getattr(bpy.types, m, None)
        if menu is not None:
            menu.remove(menu_transform)
