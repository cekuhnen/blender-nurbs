"""Work planes (CPlanes): named construction planes saved with the scene.

A work plane is a frame -- an origin and X / Y axes (its normal Z = X x Y) --
kept in scene.nproto_workplanes, as many as you like, each with a name
(Rhino's / Alias's named construction planes). One of them, or the World, is
active. Later steps draw the active plane's grid and sketch and snap on it;
active_frame() is what they read.

Placing one:
  * World XY / XZ / YZ (Top / Front / Right)
  * View: the view's rotation through the 3D cursor
  * 3D Cursor: the cursor's location and rotation
  * Selection: the active object's axes (Object Mode); a selected face's
    centre, normal and longest edge, or three selected vertices (Edit Mode)
  * Pick: click on a flat face (its plane, axes along its edges), or click an
    origin, an X direction and a point on the Y side (three points); Enter
    takes the plane so far

The frame math at the top is plain numpy (tested without Blender's UI)."""

import math

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------
# frame math
# --------------------------------------------------------------------------

def frame_from_axes(O, x, ydir):
    """(O, x, y, n) orthonormal: x along `x`, y the part of `ydir` normal to
    x, n = x cross y. None when x is zero or ydir is parallel to x."""
    O, x, ydir = (np.asarray(a, float) for a in (O, x, ydir))
    lx = float(np.linalg.norm(x))
    if lx < EPS:
        return None
    x = x / lx
    y = ydir - float(ydir @ x) * x
    ly = float(np.linalg.norm(y))
    if ly < EPS * max(1.0, float(np.linalg.norm(ydir))):
        return None
    y = y / ly
    return O, x, y, np.cross(x, y)


def frame_from_points(A, B, C):
    """Three points: origin A, X towards B, Y on C's side."""
    A, B, C = (np.asarray(p, float) for p in (A, B, C))
    return frame_from_axes(A, B - A, C - A)


def frame_from_normal(O, n, xhint=None):
    """A frame through O with normal n; X along xhint's part in the plane, else
    a stable world-based direction (world X, or world Y for a plane normal to X)."""
    n = np.asarray(n, float)
    ln = float(np.linalg.norm(n))
    if ln < EPS:
        return None
    n = n / ln
    cands = ([np.asarray(xhint, float)] if xhint is not None else []) + [np.eye(3)[0], np.eye(3)[1]]
    for h in cands:
        x = h - float(h @ n) * n
        if float(np.linalg.norm(x)) > 1e-6 * max(1.0, float(np.linalg.norm(h))):
            return frame_from_axes(O, x, np.cross(n, x))
    return None


def euler_xyz_matrix(a, b, c):
    """Rotation matrix of an XYZ Euler (Blender's: X first, then Y, then Z), double precision."""
    a, b, c = float(a), float(b), float(c)
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def plan_view(fr, at, side="TOP"):
    """(3x3 view rotation, view centre) looking square onto frame fr: screen
    right = the plane's X, up = its Y, looking down its normal (TOP); from
    behind (BOTTOM): right = X, up = -Y. The centre is `at` moved onto the
    plane along its normal (what you were looking at stays in view)."""
    O, x, y, n = (np.asarray(a, float) for a in fr)
    R = np.column_stack([x, y, n]) if side == "TOP" else np.column_stack([x, -y, -n])
    at = np.asarray(at, float)
    return R, at - float((at - O) @ n) * n


def frame_matrix(fr):
    """4x4 matrix of a frame: columns X, Y, Z (normal), origin."""
    O, x, y, n = fr
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, n, O
    return M


def frame_from_matrix(M):
    """Frame of a 4x4 (or 3x4) matrix, scale and shear removed (X kept, Y
    made normal to it in the XY plane)."""
    M = np.asarray(M, float)
    return frame_from_axes(M[:3, 3], M[:3, 0], M[:3, 1])


def face_frame(center, normal, verts):
    """A mesh face's frame: origin at its centre, X along its longest edge."""
    V = [np.asarray(p, float) for p in verts]
    best, xb = -1.0, None
    for i in range(len(V)):
        e = V[(i + 1) % len(V)] - V[i]
        le = float(e @ e)
        if le > best:
            best, xb = le, e
    return frame_from_normal(center, normal, xb)


WORLD = {"XY": (np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]),
         "XZ": (np.zeros(3), np.eye(3)[0], np.eye(3)[2], -np.eye(3)[1]),     # Front: seen from -Y
         "YZ": (np.zeros(3), np.eye(3)[1], np.eye(3)[2], np.eye(3)[0])}      # Right: seen from +X


def patch_size(extent):
    """A round patch size a little larger than `extent` (1, 2 or 5 times a power of ten)."""
    e = max(float(extent), 1e-6) * 1.25
    k = 10.0 ** math.floor(math.log10(e))
    return next(m * k for m in (1.0, 2.0, 5.0, 10.0) if m * k >= e * (1 - 1e-9))


def covering_half(fr, pts):
    """Half the width of the square about fr's origin (sides along its X / Y)
    that covers the points."""
    O, x, y, _ = fr
    d = np.asarray(pts, float) - np.asarray(O, float)
    return float(max(np.abs(d @ x).max(), np.abs(d @ y).max())) if len(d) else 1.0


def unique_name(base, taken):
    """base, or base.001, base.002 ... not in `taken`."""
    if base not in taken:
        return base
    i = 1
    while f"{base}.{i:03d}" in taken:
        i += 1
    return f"{base}.{i:03d}"


def fixed_levels(spacing, major=10, n=8):
    """Grid levels for a fixed spacing: spacing, then every `major` lines
    a heavier one, and so on (n levels, finest first)."""
    major = max(2, int(major))
    return [float(spacing) * major ** i for i in range(n)]


def endless_square(eye, vdir, persp, half_w, half_h, f_px, coarsest, fade_px, clip_end):
    """(cx, cy, R) in plane coordinates (the plane is z = 0): the square an
    endless grid is drawn on, or None. eye: the eye point (perspective) or
    the view centre (orthographic), vdir the view direction, both in plane
    coordinates. Perspective: around the eye's foot point, out to where the
    coarsest level gets too dense (f_px pixels per unit at unit distance),
    at most clip_end. Orthographic: what the view sees of the plane, from
    half_w / half_h (the view's half sizes in units)."""
    eye, vdir = np.asarray(eye, float), np.asarray(vdir, float)
    if persp:
        R = min(coarsest * f_px / fade_px, clip_end)
        if abs(eye[2]) >= R:
            return None
        return float(eye[0]), float(eye[1]), float(R)
    if abs(vdir[2]) < 1e-6:
        return None
    c = eye - vdir * (eye[2] / vdir[2])
    R = 1.2 * math.hypot(half_w, half_h / abs(vdir[2]))
    return float(c[0]), float(c[1]), float(min(R, clip_end))


# --------------------------------------------------------------------------
# Blender side
# --------------------------------------------------------------------------

import bpy  # noqa: E402
import bmesh  # noqa: E402
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,  # noqa: E402
                       IntProperty, PointerProperty, StringProperty)
from mathutils import Euler, Matrix, Vector  # noqa: E402


def _redraw(self=None, context=None):
    wm = getattr(bpy.context, "window_manager", None)
    for win in (wm.windows if wm else ()):
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# An endless work-plane grid replaces Blender's floor: while one is active the
# floor, its axes and the Top / Front / Side grid are switched off in every 3D
# view, their settings kept in the scene (FLOOR_STASH) and put back when the
# World or a plane without an endless grid becomes active.
FLOOR_PROPS = ("show_floor", "show_axis_x", "show_axis_y", "show_axis_z", "show_ortho_grid")
FLOOR_STASH = "nproto_floor_stash"


def _views():
    """(key, overlay) of every 3D view in every screen."""
    for scr in bpy.data.screens:
        for ai, area in enumerate(scr.areas):
            for si, sp in enumerate(area.spaces):
                if sp.type == 'VIEW_3D':
                    yield f"{scr.name}|{ai}|{si}", sp.overlay


def floor_hidden_wanted(scene):
    wp = active_plane(scene)
    return wp is not None and wp.grid == "ENDLESS"


def sync_floor(scene):
    """Hide Blender's floor grid for an endless work-plane grid, or put it back."""
    import json
    hide = floor_hidden_wanted(scene)
    stash = scene.get(FLOOR_STASH)
    if hide:
        kept = json.loads(stash) if stash else {}
        for key, ov in _views():
            if key not in kept:
                kept[key] = [bool(getattr(ov, p)) for p in FLOOR_PROPS]
            for p in FLOOR_PROPS:
                setattr(ov, p, False)
        scene[FLOOR_STASH] = json.dumps(kept)
    elif stash:
        kept = json.loads(stash)
        for key, ov in _views():
            for p, val in zip(FLOOR_PROPS, kept.get(key, [True, True, True, False, True])):
                setattr(ov, p, val)
        del scene[FLOOR_STASH]
    _redraw()


def _on_grid(self, context):
    sync_floor(context.scene if context and context.scene else bpy.context.scene)


class NprotoWorkPlane(bpy.types.PropertyGroup):
    """A named work plane: origin and rotation (its local X / Y span the plane)."""
    # name: PropertyGroup's own
    origin: FloatVectorProperty(name="Origin", size=3, subtype='TRANSLATION', unit='LENGTH', update=_redraw,
                                description="Where the plane's X and Y axes cross")
    rotation: FloatVectorProperty(name="Rotation", size=3, subtype='EULER', unit='ROTATION', update=_redraw,
                                  description="The plane's axes as a rotation of the world's (XYZ Euler): "
                                              "its X and Y span the plane, Z is its normal")
    grid: EnumProperty(name="Grid", default="PATCH", update=_on_grid,
                       items=[("OFF", "Off", "No grid: only the plane's axes at its origin", 'CHECKBOX_DEHLT', 0),
                              ("PATCH", "Patch", "A square grid of the given size around the origin", 'MESH_PLANE', 1),
                              ("ENDLESS", "Endless", "The grid fills the view, like the floor -- Blender's floor "
                                                     "grid and axes are hidden while this plane is active",
                               'MESH_GRID', 2)])
    size: FloatProperty(name="Size", default=2.0, min=1e-6, soft_max=1000.0, unit='LENGTH', update=_redraw,
                        description="Width of the patch grid")
    auto_spacing: BoolProperty(name="Auto Spacing", default=True, update=_redraw,
                               description="Grid lines like the viewport's (the scene's units, Grid Scale and "
                                           "Subdivisions), finer as you zoom in")
    spacing: FloatProperty(name="Spacing", default=0.1, min=1e-6, soft_max=100.0, unit='LENGTH', update=_redraw,
                           description="Distance between the finest grid lines")
    major: IntProperty(name="Major Every", default=10, min=2, max=100, update=_redraw,
                       description="Every this many lines a heavier one")


class NprotoWorkPlanes(bpy.types.PropertyGroup):
    planes: CollectionProperty(type=NprotoWorkPlane)
    index: IntProperty(name="Work Plane", default=0, min=0, description="The work plane shown for editing")
    active: IntProperty(name="Active Work Plane", default=-1, min=-1, update=_on_grid,
                        description="The work plane sketches and the grid use (-1: the World)")


def set_frame(wp, fr):
    """Store frame fr in work plane wp."""
    O, x, y, n = fr
    R = Matrix([list(map(float, x)), list(map(float, y)), list(map(float, n))]).transposed()
    wp.origin = tuple(map(float, O))
    wp.rotation = tuple(R.to_euler('XYZ'))


def frame_of(wp):
    """(O, x, y, n) of work plane wp, numpy, in double precision: mathutils
    builds rotations in single precision, whose axes are square to each other
    only to about 1e-8 -- steps along X / Y (Ctrl's grid) then drifted off
    the plane by that much."""
    R = euler_xyz_matrix(*wp.rotation)
    return np.array(wp.origin, float), R[:, 0], R[:, 1], R[:, 2]


def active_plane(scene):
    """The active work plane (NprotoWorkPlane), or None for the World."""
    st = getattr(scene, "nproto_workplanes", None)
    if st is None or not (0 <= st.active < len(st.planes)):
        return None
    return st.planes[st.active]


def active_frame(scene):
    """(O, x, y, n) of the active work plane; None when the World is active."""
    wp = active_plane(scene)
    return None if wp is None else frame_of(wp)


def add_plane(scene, name, fr, activate=True):
    """A new work plane `name` (made unique) at frame fr; selected, and active
    with activate. Returns it."""
    st = scene.nproto_workplanes
    wp = st.planes.add()
    wp.name = unique_name(name, {p.name for p in st.planes if p != wp})
    set_frame(wp, fr)
    st.index = len(st.planes) - 1
    if activate:
        st.active = st.index
    return wp


# -- frames from the scene ----------------------------------------------------

def _view_rv3d(context):
    sp = context.space_data if context.area and context.area.type == 'VIEW_3D' else None
    if sp is None:
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    sp = area.spaces.active
                    break
            if sp is not None:
                break
    return getattr(sp, "region_3d", None)


def view_frame(context):
    """The view's axes (X right, Y up, normal towards the viewer) through the 3D cursor."""
    rv3d = _view_rv3d(context)
    if rv3d is None:
        return None
    R = np.array(rv3d.view_rotation.to_matrix(), float)
    return np.array(context.scene.cursor.location, float), R[:, 0], R[:, 1], R[:, 2]


def cursor_frame(context):
    return frame_from_matrix(np.array(context.scene.cursor.matrix, float))


def selection_frame(context):
    """(frame, label) from the selection, or (None, why)."""
    obj = context.active_object
    if context.mode == 'EDIT_MESH' and obj is not None and obj.type == 'MESH':
        M = obj.matrix_world
        bm = bmesh.from_edit_mesh(obj.data)
        faces = [f for f in bm.faces if f.select]
        act = bm.faces.active if bm.faces.active is not None and bm.faces.active.select else None
        if act is not None or len(faces) == 1:
            f = act or faces[0]
            W = [M @ v.co for v in f.verts]
            nrm = (M.to_3x3().inverted_safe().transposed() @ f.normal).normalized()
            fr = face_frame(np.array(M @ f.calc_center_median(), float), np.array(nrm, float), W)
            return (fr, "Face") if fr is not None else (None, "the face is degenerate")
        hist = [e for e in bm.select_history if isinstance(e, bmesh.types.BMVert) and e.select]
        verts = hist + [v for v in bm.verts if v.select and v not in hist]
        if len(verts) >= 3:
            A, B, C = (np.array(M @ v.co, float) for v in verts[:3])
            fr = frame_from_points(A, B, C)
            return (fr, "3 Points") if fr is not None else (None, "the three vertices lie on a line")
        return None, "select a face, or three vertices (origin, X, Y side)"
    if obj is not None:
        fr = frame_from_matrix(np.array(obj.matrix_world, float))
        return (fr, obj.name) if fr is not None else (None, "the object's axes are degenerate")
    return None, "no active object"


SOURCES = [("WORLD_XY", "Top (World XY)", "The floor: world X and Y, normal +Z", 'AXIS_TOP', 0),
           ("WORLD_XZ", "Front (World XZ)", "World X and Z, normal -Y (seen from the front)", 'AXIS_FRONT', 1),
           ("WORLD_YZ", "Right (World YZ)", "World Y and Z, normal +X (seen from the right)", 'AXIS_SIDE', 2),
           ("VIEW", "View", "Parallel to the screen, through the 3D cursor", 'HIDE_OFF', 3),
           ("CURSOR", "3D Cursor", "The 3D cursor's location and rotation", 'PIVOT_CURSOR', 4),
           ("SELECTION", "Selection", "Object Mode: the active object's axes. Edit Mode: the selected face "
                                      "(centre, normal, X along its longest edge) or three selected vertices "
                                      "(origin, X, Y side)", 'RESTRICT_SELECT_OFF', 5)]
SOURCE_NAMES = {"WORLD_XY": "Top", "WORLD_XZ": "Front", "WORLD_YZ": "Right", "VIEW": "View", "CURSOR": "Cursor"}


def source_frame(context, source):
    """(frame, name) for a source, or (None, why)."""
    if source.startswith("WORLD_"):
        return WORLD[source[6:]], SOURCE_NAMES[source]
    if source == "VIEW":
        fr = view_frame(context)
        return (fr, "View") if fr is not None else (None, "no 3D View")
    if source == "CURSOR":
        return cursor_frame(context), "Cursor"
    return selection_frame(context)


# -- operators ---------------------------------------------------------------

class NPROTO_OT_workplane_add(bpy.types.Operator):
    """Add a named work plane"""
    bl_idname = "nproto.workplane_add"
    bl_label = "Add Work Plane"
    bl_options = {'REGISTER', 'UNDO'}

    source: EnumProperty(name="From", items=SOURCES, default="WORLD_XY")
    activate: BoolProperty(name="Make Active", default=True)

    @classmethod
    def description(cls, context, props):
        return dict((s[0], s[2]) for s in SOURCES).get(props.source, cls.__doc__)

    def execute(self, context):
        fr, name = source_frame(context, self.source)
        if fr is None:
            self.report({'WARNING'}, f"Work plane: {name}")
            return {'CANCELLED'}
        wp = add_plane(context.scene, name, fr, self.activate)
        self.report({'INFO'}, f"Work plane '{wp.name}'")
        return {'FINISHED'}


class NPROTO_OT_workplane_redefine(bpy.types.Operator):
    """Move the selected work plane (keeping its name)"""
    bl_idname = "nproto.workplane_redefine"
    bl_label = "Redefine Work Plane"
    bl_options = {'REGISTER', 'UNDO'}

    source: EnumProperty(name="From", items=SOURCES, default="CURSOR")

    @classmethod
    def poll(cls, context):
        st = context.scene.nproto_workplanes
        return 0 <= st.index < len(st.planes)

    @classmethod
    def description(cls, context, props):
        return "Move the selected work plane to: " + dict((s[0], s[2]) for s in SOURCES).get(props.source, "")

    def execute(self, context):
        fr, why = source_frame(context, self.source)
        if fr is None:
            self.report({'WARNING'}, f"Work plane: {why}")
            return {'CANCELLED'}
        st = context.scene.nproto_workplanes
        set_frame(st.planes[st.index], fr)
        return {'FINISHED'}


class NPROTO_OT_workplane_remove(bpy.types.Operator):
    """Delete the selected work plane (the World becomes active if it was)"""
    bl_idname = "nproto.workplane_remove"
    bl_label = "Delete Work Plane"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        st = context.scene.nproto_workplanes
        return 0 <= st.index < len(st.planes)

    def execute(self, context):
        st = context.scene.nproto_workplanes
        i = st.index
        st.planes.remove(i)
        if st.active == i:
            st.active = -1
        elif st.active > i:
            st.active -= 1
        st.index = min(i, len(st.planes) - 1) if st.planes else 0
        return {'FINISHED'}


class NPROTO_OT_workplane_move(bpy.types.Operator):
    """Move the selected work plane up or down the list"""
    bl_idname = "nproto.workplane_move"
    bl_label = "Move Work Plane"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(items=[("UP", "Up", ""), ("DOWN", "Down", "")])

    @classmethod
    def poll(cls, context):
        st = context.scene.nproto_workplanes
        return 0 <= st.index < len(st.planes)

    def execute(self, context):
        st = context.scene.nproto_workplanes
        i = st.index
        j = i - 1 if self.direction == "UP" else i + 1
        if not 0 <= j < len(st.planes):
            return {'CANCELLED'}
        st.planes.move(i, j)
        if st.active == i:
            st.active = j
        elif st.active == j:
            st.active = i
        st.index = j
        return {'FINISHED'}


class NPROTO_OT_workplane_activate(bpy.types.Operator):
    """Make this the active work plane (-1: the World)"""
    bl_idname = "nproto.workplane_activate"
    bl_label = "Activate Work Plane"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(default=-1, min=-1)

    def execute(self, context):
        st = context.scene.nproto_workplanes
        st.active = self.index if self.index < len(st.planes) else -1
        if self.index >= 0:
            st.index = self.index
        return {'FINISHED'}


_anims = {}     # region_3d pointer -> running view animation


def _animate(rv3d, area, q1, loc1, persp, ms):
    """Turn the view to rotation q1 / centre loc1 over ms milliseconds (Blender's
    Smooth View), then set the projection; at once with ms 0."""
    key = rv3d.as_pointer()
    q0, loc0 = rv3d.view_rotation.copy(), rv3d.view_location.copy()
    if q0.dot(q1) < 0:
        q1 = -q1
    if ms <= 0:
        rv3d.view_rotation, rv3d.view_location = q1, loc1
        if persp:
            rv3d.view_perspective = persp
        area.tag_redraw()
        return
    import time
    t0 = time.perf_counter()
    _anims[key] = t0

    def step():
        if _anims.get(key) != t0:
            return None                         # a newer turn took over
        try:
            t = min(1.0, (time.perf_counter() - t0) * 1000.0 / ms)
            e = t * t * (3.0 - 2.0 * t)
            rv3d.view_rotation = q0.slerp(q1, e)
            rv3d.view_location = loc0.lerp(loc1, e)
            area.tag_redraw()
        except ReferenceError:                  # the view closed
            _anims.pop(key, None)
            return None
        if t >= 1.0:
            if persp:
                rv3d.view_perspective = persp
            _anims.pop(key, None)
            return None
        return 0.01
    bpy.app.timers.register(step)


class NPROTO_OT_workplane_view(bpy.types.Operator):
    """Look square onto a work plane, its X to the right and Y up (the World: Top).
    Orthographic with Auto Perspective on (Preferences > Navigation)"""
    bl_idname = "nproto.workplane_view"
    bl_label = "Look at Work Plane"
    bl_options = {'REGISTER'}

    index: IntProperty(name="Work Plane", default=-2, min=-2, options={'SKIP_SAVE'},
                       description="-2: the active one, -1: the World")
    side: EnumProperty(name="Side", items=[("TOP", "Top", "Looking down its normal"),
                                           ("BOTTOM", "Bottom", "From behind: looking up its normal")],
                       default="TOP", options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def execute(self, context):
        st = context.scene.nproto_workplanes
        i = st.active if self.index == -2 else self.index
        fr = frame_of(st.planes[i]) if 0 <= i < len(st.planes) else WORLD["XY"]
        sp = context.space_data
        rv3d = sp.region_3d
        R, loc = plan_view(fr, np.array(rv3d.view_location, float), self.side)
        q = Matrix(R.tolist()).to_quaternion()
        prefs = context.preferences
        persp = 'ORTHO' if prefs.inputs.use_auto_perspective and rv3d.view_perspective == 'PERSP' else None
        if rv3d.view_perspective == 'CAMERA':
            persp = 'ORTHO' if prefs.inputs.use_auto_perspective else 'PERSP'
        _animate(rv3d, context.area, q, Vector(tuple(map(float, loc))), persp, prefs.view.smooth_view)
        return {'FINISHED'}


TURNS = [("FLIP", "Flip", "Turn it over: the normal points the other way (about its X)"),
         ("ROT90", "Rotate 90°", "Rotate it 90° about its normal"),
         ("SWAP", "Swap X / Y", "Its X becomes Y and Y becomes X (the normal flips)")]


class NPROTO_OT_workplane_turn(bpy.types.Operator):
    """Turn the selected work plane about its own axes"""
    bl_idname = "nproto.workplane_turn"
    bl_label = "Turn Work Plane"
    bl_options = {'REGISTER', 'UNDO'}

    action: EnumProperty(items=TURNS)

    @classmethod
    def poll(cls, context):
        st = context.scene.nproto_workplanes
        return 0 <= st.index < len(st.planes)

    @classmethod
    def description(cls, context, props):
        return {i[0]: i[2] for i in TURNS}.get(props.action, cls.__doc__)

    def execute(self, context):
        st = context.scene.nproto_workplanes
        wp = st.planes[st.index]
        O, x, y, n = frame_of(wp)
        if self.action == "FLIP":
            fr = (O, x, -y, -n)
        elif self.action == "ROT90":
            fr = (O, y, -x, n)
        else:
            fr = (O, y, x, -n)
        set_frame(wp, fr)
        return {'FINISHED'}


# -- Pick: a flat face, or three points ----------------------------------------

from . import draw, hints  # noqa: E402
from .drawtools import REFRESH_KEYS, SketchMixin  # noqa: E402

COL_AX = {"x": (1.0, 0.25, 0.3, 1.0), "y": (0.45, 0.9, 0.2, 1.0), "n": (0.3, 0.55, 1.0, 1.0)}
COL_OUTLINE = (0.6, 0.85, 1.0, 0.9)


def marker_size(rv3d, at):
    """World length of a marker about 60 px tall at point `at`."""
    try:
        if rv3d.is_perspective:
            eye = np.array(rv3d.view_matrix.inverted().translation, float)
            d = float(np.linalg.norm(np.asarray(at, float) - eye))
            return 0.12 * max(d, 1e-4)
        return 0.12 * max(float(rv3d.view_distance), 1e-4)
    except Exception:
        return 1.0


def marker_lines(fr, L):
    """[(polyline, colour, width)]: the plane's X / Y / normal and a square around its origin."""
    O, x, y, n = (np.asarray(a, float) for a in fr)
    V = lambda p: Vector(tuple(map(float, p)))
    sq = [O + L * (a * x + b * y) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
    return [([V(p) for p in sq], COL_OUTLINE, 1.5),
            ([V(O), V(O + L * x)], COL_AX["x"], 3.0),
            ([V(O), V(O + L * y)], COL_AX["y"], 3.0),
            ([V(O), V(O + 0.5 * L * n)], COL_AX["n"], 2.0)]


class NPROTO_OT_workplane_pick(SketchMixin, bpy.types.Operator):
    """Place a work plane by clicking: on a flat face its plane (Enter takes
    it, X along the face's edges), or an origin, an X direction and a point
    on the Y side. Snaps like the drawing tools (Ctrl: grid, Alt: none)"""
    bl_idname = "nproto.workplane_pick"
    bl_label = "Pick Work Plane"
    bl_options = {'REGISTER', 'UNDO'}

    activate: BoolProperty(name="Make Active", default=True)
    redefine: BoolProperty(name="Redefine Selected", default=False, options={'SKIP_SAVE'},
                           description="Move the selected work plane instead of adding one")

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, f"{self.bl_label}: run it in a 3D View")
            return {'CANCELLED'}
        self._pts, self._lock = [], None
        self.sketch_start(context)
        context.window_manager.modal_handler_add(self)
        self._preview(context, self.pick_point(context, event))
        return {'RUNNING_MODAL'}

    def _frame(self, hover=None):
        """The frame so far (with the hovered point as the next click)."""
        pts = list(self._pts)
        if hover is not None and hover[0] is not None and len(pts) < 3:
            pts.append(np.asarray(hover[0], float))
        if not pts:
            return None
        if len(pts) == 1:
            if self._plane is None:
                f = getattr(self, "_face_hover", None)
                if f is not None:
                    return pts[0], f["u"], f["v"], f["n"]
                return None
            O, n, u, v = self._plane
            return pts[0], u, v, n
        O, n, u, v = self._plane
        A, B = pts[0], pts[1]
        if len(pts) == 2:
            x = B - A
            return frame_from_axes(A, x, np.cross(n, x)) if np.linalg.norm(x) > EPS else (A, u, v, n)
        return frame_from_points(*pts[:3]) or frame_from_axes(A, B - A, np.cross(n, B - A))

    def _status(self, context):
        k = len(self._pts)
        state = ["click a flat face or the origin", "click the X direction (Enter: take this plane)",
                 "click the Y side (Enter: keep the plane's normal)"][min(k, 2)]
        rows = [("Place", [hints.key("Enter", "take the plane so far" if k else "after the first click",
                                     on=None if k else False),
                           hints.key("Backspace", "undo a click")]),
                ("Snap", self.snap_hints())]
        hints.show(context, "Work Plane", [state], rows,
                   [("MOUSE_LMB", "Place"), ("EVENT_ESC", "Cancel")])

    def _preview(self, context, hover=None):
        ov = draw.sketch_overlay
        ov["snap"] = Vector(hover[0]) if hover and hover[1] is not None else None
        ov["snap_label"] = hover[1][3] if hover and hover[1] is not None else None
        ov["cursor"] = Vector(hover[0]) if hover and hover[0] is not None else None
        ov["cvs"] = [Vector(tuple(map(float, p))) for p in self._pts]
        fr = self._frame(hover)
        ov["refs"] = []
        if fr is not None:
            ov["refs"] = marker_lines(fr, marker_size(self._win(context)[1], fr[0]))
        self._status(context)
        context.area.tag_redraw()

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        hover = lambda: self.pick_point(context, event)
        if event.type in REFRESH_KEYS or self.snap_key(context, event):
            self._preview(context, hover())
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            P, _ = hover()
            if P is None:
                return {'RUNNING_MODAL'}
            P = np.asarray(P, float)
            if not self._pts:
                self.fix_plane(context, P)
                self._from_face = getattr(self, "_plane_face", None) is not None
            elif any(np.linalg.norm(P - q) < 1e-9 for q in self._pts):
                return {'RUNNING_MODAL'}
            self._pts.append(P)
            if len(self._pts) == 3:
                return self._finish(context)
            self._preview(context, (P, None))
        elif event.value == 'PRESS' and event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and self._pts:
            return self._finish(context)
        elif event.value == 'PRESS' and event.type == 'BACK_SPACE' and self._pts:
            self._pts.pop()
            if not self._pts:
                self._plane, self._plane_face = None, None
            self._preview(context, hover())
        elif event.value == 'PRESS' and event.type in {'ESC', 'RIGHTMOUSE'}:
            self.sketch_end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        fr = self._frame()
        self.sketch_end(context)
        if fr is None:
            self.report({'WARNING'}, "Work plane: the points lie on a line")
            return {'CANCELLED'}
        st = context.scene.nproto_workplanes
        if self.redefine and 0 <= st.index < len(st.planes):
            set_frame(st.planes[st.index], fr)
            name = st.planes[st.index].name
        else:
            face = getattr(self, "_plane_face", None) if getattr(self, "_from_face", False) else None
            name = "Face" if face is not None and len(self._pts) == 1 else \
                "3 Points" if len(self._pts) == 3 else "Plane"
            wp = add_plane(context.scene, name, fr, self.activate)
            if face is not None and len(face.get("seg", ())):
                wp.size = patch_size(2.0 * covering_half(fr, np.asarray(face["seg"], float).reshape(-1, 3)))
            name = wp.name
        self.report({'INFO'}, f"Work plane '{name}'")
        _redraw()
        return {'FINISHED'}


# -- UI ----------------------------------------------------------------------

class NPROTO_MT_workplane_add(bpy.types.Menu):
    bl_label = "Add Work Plane"
    bl_idname = "NPROTO_MT_workplane_add"

    def draw(self, context):
        lay = self.layout
        lay.operator("nproto.workplane_pick", text="Pick (Face / 3 Points)", icon='RESTRICT_SELECT_OFF')
        lay.separator()
        for key, label, _, icon, _ in SOURCES:
            lay.operator("nproto.workplane_add", text=label, icon=icon).source = key


class NPROTO_MT_workplane_redefine(bpy.types.Menu):
    bl_label = "Redefine Work Plane"
    bl_idname = "NPROTO_MT_workplane_redefine"

    def draw(self, context):
        lay = self.layout
        op = lay.operator("nproto.workplane_pick", text="Pick (Face / 3 Points)", icon='RESTRICT_SELECT_OFF')
        op.redefine = True
        lay.separator()
        for key, label, _, icon, _ in SOURCES:
            lay.operator("nproto.workplane_redefine", text=label, icon=icon).source = key


class NPROTO_UL_workplanes(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        on = data.active == index
        row.operator("nproto.workplane_activate", text="", emboss=False,
                     icon='RADIOBUT_ON' if on else 'RADIOBUT_OFF').index = index
        row.prop(item, "name", text="", emboss=False)


class VIEW3D_PT_nproto_workplanes(bpy.types.Panel):
    """Named work planes: the active one is where the grid and sketches go"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "NURBS Proto"
    bl_label = "Work Planes"
    bl_order = 7
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        wp = active_plane(context.scene)
        self.layout.label(text="", icon='MESH_GRID' if wp is not None else 'WORLD')

    def draw(self, context):
        lay = self.layout
        st = context.scene.nproto_workplanes
        row = lay.row(align=True)
        row.operator("nproto.workplane_activate", text="World", depress=st.active < 0,
                     icon='RADIOBUT_ON' if st.active < 0 else 'RADIOBUT_OFF').index = -1
        row.operator("nproto.workplane_view", text="", icon='VIEW_ORTHO')     # the active plane (World: Top)
        row = lay.row()
        row.template_list("NPROTO_UL_workplanes", "", st, "planes", st, "index", rows=4)
        col = row.column(align=True)
        col.menu("NPROTO_MT_workplane_add", text="", icon='ADD')
        col.operator("nproto.workplane_remove", text="", icon='REMOVE')
        col.separator()
        col.operator("nproto.workplane_move", text="", icon='TRIA_UP').direction = "UP"
        col.operator("nproto.workplane_move", text="", icon='TRIA_DOWN').direction = "DOWN"
        if not (0 <= st.index < len(st.planes)):
            return
        wp = st.planes[st.index]
        col = lay.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(wp, "origin")
        col.prop(wp, "rotation")
        lay.row(align=True).prop(wp, "grid", expand=True)
        if wp.grid != "OFF":
            col = lay.column()
            col.use_property_split = True
            col.use_property_decorate = False
            if wp.grid == "PATCH":
                col.prop(wp, "size")
            col.prop(wp, "auto_spacing")
            sub = col.column(align=True)
            sub.enabled = not wp.auto_spacing
            sub.prop(wp, "spacing")
            sub.prop(wp, "major")
        row = lay.row(align=True)
        row.operator("nproto.workplane_turn", text="Flip").action = "FLIP"
        row.operator("nproto.workplane_turn", text="Rotate 90°").action = "ROT90"
        row.operator("nproto.workplane_turn", text="Swap X/Y").action = "SWAP"
        lay.menu("NPROTO_MT_workplane_redefine", text="Redefine", icon='FILE_REFRESH')
        row = lay.row(align=True)
        row.operator("nproto.workplane_view", text="Look at", icon='VIEW_ORTHO').index = st.index
        op = row.operator("nproto.workplane_view", text="From Behind")
        op.index, op.side = st.index, "BOTTOM"


# -- the active plane's grid ---------------------------------------------------
# Drawn in the plane's own coordinates (x, y; the plane is z = 0 there) with
# the fine floor grid's rules (draw.py): eight levels, finest first; each
# fades in from GRID_FADE_PX apart, the next two up are heavier; the whole
# grid fades as the plane turns edge-on, an endless one also towards its rim.
# Its X and Y axes are drawn in the theme's axis colours.

import gpu  # noqa: E402
from gpu_extras.batch import batch_for_shader  # noqa: E402

_shader = None      # compiled on first use; False when it can't be
DEPTH_BIAS = 2e-5   # normalised depth the grid is pulled towards the viewer (ortho depth is linear: less)

_VERT = """
void main()
{
  co = pos;
  wpos = (ModelMatrix * vec4(pos, 0.0, 1.0)).xyz;
  gl_Position = ViewProjectionMatrix * vec4(wpos, 1.0);
  gl_Position.z -= rim.z * gl_Position.w;     /* a hair towards the viewer: shows on faces lying in the plane */
}
"""

# stepsA / stepsB: the eight levels; params: (fade px, extra half width,
# emphasis shade, rim fade radius -- 0: none); rim: (centre x, y) of the
# fade, depth bias); eye: (eye point or view direction, perspective?)
_FRAG = """
float level(int i)
{
  return (i < 4) ? stepsA[i] : stepsB[min(i, 7) - 4];
}

float grid_line(vec2 c, vec2 fw, float s)
{
  vec2 h = vec2(s * 0.5);
  vec2 d = abs(mod(c + h, vec2(s)) - h) / max(fw, vec2(1e-12));
  return 1.0 - smoothstep(-0.0924, 1.0924, min(d.x, d.y) - params.y);
}

float axis_line(float c, float fw)
{
  return 1.0 - smoothstep(-0.0924, 1.0924, abs(c) / max(fw, 1e-12) - params.y - 0.5);
}

void main()
{
  vec2 dx = dFdx(co);
  vec2 dy = dFdy(co);
  float det = abs(dx.x * dy.y - dy.x * dx.y);
  if (det < 1e-30) {
    discard;
  }
  float ppu = min(length(vec2(dx.y, dy.y)), length(vec2(dx.x, dy.x))) / det;
  float fpx = params.x;
  int i = 0;
  while (i < 7 && level(i) * ppu < fpx) {
    i++;
  }
  float s0 = (i > 0) ? level(i - 1) : 0.0;
  float s1 = level(i);
  float s2 = level(i + 1);
  float s3 = level(i + 2);
  float t = clamp((fpx - s0 * ppu) / max((s1 - s0) * ppu, 1e-12), 0.0, 1.0);
  float blend = 1.0 - t;
  blend = blend * blend * blend;
  vec2 fw = abs(dx) + abs(dy);
  vec4 emph = vec4(clamp(gridColor.rgb + vec3(params.z), 0.0, 1.0), gridColor.a);
  vec4 c = gridColor;
  c.a *= grid_line(co, fw, s1) * blend;
  c = mix(c, mix(gridColor, emph, blend), grid_line(co, fw, s2));
  c = mix(c, emph, grid_line(co, fw, s3));
  c = mix(c, axisX, axis_line(co.y, fw.y));
  c = mix(c, axisY, axis_line(co.x, fw.x));
  vec3 n = normalize(ModelMatrix[2].xyz);
  vec3 vv = (eye.w > 0.5) ? normalize(eye.xyz - wpos) : eye.xyz;
  float a = 1.0 - abs(dot(vv, n));
  a *= a;
  c.a *= 1.0 - a * a;
  if (params.w > 0.0) {
    c.a *= 1.0 - smoothstep(0.5 * params.w, params.w, length(co - rim.xy));
  }
  fragColor = c;
}
"""


def _grid_shader():
    global _shader
    if _shader is None:
        try:
            iface = gpu.types.GPUStageInterfaceInfo("nproto_workplane_iface")
            iface.smooth('VEC2', "co")
            iface.smooth('VEC3', "wpos")
            info = gpu.types.GPUShaderCreateInfo()
            for name in ("ViewProjectionMatrix", "ModelMatrix"):
                info.push_constant('MAT4', name)
            for name in ("gridColor", "axisX", "axisY", "stepsA", "stepsB", "params", "rim", "eye"):
                info.push_constant('VEC4', name)
            info.vertex_in(0, 'VEC2', "pos")
            info.vertex_out(iface)
            info.fragment_out(0, 'VEC4', "fragColor")
            info.vertex_source(_VERT)
            info.fragment_source(_FRAG)
            _shader = gpu.shader.create_from_info(info)
        except Exception as exc:
            print("NURBS Proto work plane grid:", exc)
            _shader = False
    return _shader or None


def plane_levels(context, wp, overlay=None):
    """The eight grid levels of work plane wp, finest first: the viewport's
    (units, Grid Scale, Subdivisions) with Auto Spacing, else its own."""
    from .drawtools import grid_steps
    if not wp.auto_spacing:
        return fixed_levels(wp.spacing, wp.major)
    us = context.scene.unit_settings
    ov = overlay
    return grid_steps(us.system, us.scale_length or 1.0, getattr(ov, "grid_scale", 1.0) or 1.0,
                      getattr(ov, "grid_subdivisions", 10) or 10)


def grid_square(context, wp, levels):
    """(cx, cy, R, rim) in plane coordinates for wp's grid in this view; rim:
    fade towards the edge (endless). None: nothing to draw."""
    if wp.grid == "PATCH":
        return 0.0, 0.0, 0.5 * float(wp.size), False
    from .drawtools import GRID_FADE_PX
    rv3d, region, sp = context.region_data, context.region, context.space_data
    Minv = np.linalg.inv(frame_matrix(frame_of(wp)))
    W = rv3d.window_matrix
    vdir = Minv[:3, :3] @ np.array(rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)), float)
    if rv3d.is_perspective:
        eye = Minv @ np.append(np.array(rv3d.view_matrix.inverted().translation, float), 1.0)
    else:
        eye = Minv @ np.append(np.array(rv3d.view_location, float), 1.0)
    sq = endless_square(eye[:3], vdir, rv3d.is_perspective,
                        1.0 / max(abs(W[0][0]), 1e-12), 1.0 / max(abs(W[1][1]), 1e-12),
                        abs(W[1][1]) * region.height * 0.5, levels[-1], GRID_FADE_PX, sp.clip_end)
    return None if sq is None else sq + (True,)


def draw_grid(ctx):
    """The active work plane's grid (under everything); the plane's axes marker
    with its grid off."""
    wp = active_plane(ctx.scene)
    rv3d, region, sp = getattr(ctx, "region_data", None), ctx.region, ctx.space_data
    if wp is None or rv3d is None or region is None:
        return
    ov = getattr(sp, "overlay", None)
    if ov is not None and not ov.show_overlays:
        return
    fr = frame_of(wp)
    if wp.grid == "OFF":
        return
    sh = _grid_shader()
    if sh is None:
        return
    levels = plane_levels(ctx, wp, ov)
    sq = grid_square(ctx, wp, levels)
    if sq is None:
        return
    cx, cy, R, rim = sq
    from .drawtools import GRID_FADE_PX
    quad = [(cx - R, cy - R), (cx + R, cy - R), (cx + R, cy + R), (cx - R, cy + R)]
    batch = batch_for_shader(sh, 'TRIS', {"pos": quad}, indices=((0, 1, 2), (0, 2, 3)))
    theme = ctx.preferences.themes[0]
    col = tuple(theme.view_3d.grid)
    col = col if len(col) == 4 else col + (1.0,)
    try:
        bg = tuple(theme.view_3d.space.gradients.high_gradient)
    except Exception:
        bg = (0.24, 0.24, 0.24)
    shade = (20.0 if sum(col[:3]) + 0.12 > sum(bg[:3]) else -10.0) / 255.0
    ui = theme.user_interface
    ax = lambda c: tuple(c)[:3] + (1.0,)
    if rv3d.is_perspective:
        e, persp = rv3d.view_matrix.inverted().translation, 1.0
    else:
        e, persp = rv3d.view_rotation @ Vector((0.0, 0.0, 1.0)), 0.0
    pix = ctx.preferences.system.pixel_size
    lv = [float(v) for v in levels] + [float(levels[-1])] * (8 - len(levels))
    mask = gpu.state.depth_mask_get()
    gpu.state.depth_mask_set(False)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('ALPHA')
    sh.bind()
    sh.uniform_float("ViewProjectionMatrix", rv3d.perspective_matrix)
    sh.uniform_float("ModelMatrix", Matrix(frame_matrix(fr).tolist()))
    sh.uniform_float("gridColor", col)
    sh.uniform_float("axisX", ax(ui.axis_x))
    sh.uniform_float("axisY", ax(ui.axis_y))
    sh.uniform_float("stepsA", tuple(lv[:4]))
    sh.uniform_float("stepsB", tuple(lv[4:8]))
    sh.uniform_float("params", (float(GRID_FADE_PX), max(0.0, pix - 1.0) * 0.5, shade, float(R) if rim else 0.0))
    sh.uniform_float("rim", (float(cx), float(cy), DEPTH_BIAS * (1.0 if rv3d.is_perspective else 0.1), 0.0))
    sh.uniform_float("eye", (float(e.x), float(e.y), float(e.z), persp))
    batch.draw(sh)
    gpu.state.depth_mask_set(mask)
    if not rim:                      # the patch's outline
        O, x, y, n = fr
        corners = [O + R * (a * x + b * y) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
        V = [Vector(tuple(map(float, p))) for p in corners]
        draw._draw_lines([p for k in range(4) for p in (V[k], V[k + 1])], col[:3] + (0.6,), 1.0)


def draw_grid_safe(ctx):
    """draw_grid, switched off for the session on its first error."""
    global _shader
    try:
        draw_grid(ctx)
    except Exception as exc:
        print("NURBS Proto work plane grid off:", exc)
        _shader = False


def draw_active_marker(ctx):
    """The active work plane's axes and normal at its origin (its grid is drawn
    first, under everything): the square too while the grid is off."""
    wp = active_plane(ctx.scene)
    rv3d = getattr(ctx, "region_data", None)
    if wp is None or rv3d is None:
        return
    fr = frame_of(wp)
    lines = marker_lines(fr, marker_size(rv3d, fr[0]))
    if wp.grid != "OFF":
        lines = lines[1:]
    for pl, col, w in lines:
        draw._draw_lines([p for i in range(len(pl) - 1) for p in (pl[i], pl[i + 1])], col, w)


classes = (NprotoWorkPlane, NprotoWorkPlanes, NPROTO_OT_workplane_add, NPROTO_OT_workplane_redefine,
           NPROTO_OT_workplane_view,
           NPROTO_OT_workplane_remove, NPROTO_OT_workplane_move, NPROTO_OT_workplane_activate,
           NPROTO_OT_workplane_turn, NPROTO_OT_workplane_pick, NPROTO_MT_workplane_add,
           NPROTO_MT_workplane_redefine, NPROTO_UL_workplanes, VIEW3D_PT_nproto_workplanes)


def register():
    bpy.types.Scene.nproto_workplanes = PointerProperty(type=NprotoWorkPlanes)


def unregister():
    del bpy.types.Scene.nproto_workplanes
