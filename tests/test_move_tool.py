"""Move by Points (hold G):  blender -b --factory-startup --python tests/test_move_tool.py"""
import os, sys
import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, geom, movetool, pattern, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def W(o):
    return np.array(geom.world_matrix(o), float)

def select(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0]

# -- typed copies ---------------------------------------------------------------
check("parse *5", movetool.parse_copies("*5") == ("ARRAY", 5))
check("parse /4", movetool.parse_copies("/ 4") == ("DIVIDE", 4))
for t in ("*", "/", "*0", "5", "*x", "", "10*2"):
    check(f"parse {t!r} is none", movetool.parse_copies(t) is None)
d = np.array([3.0, 0.0, 0.0])
check("array offsets", np.allclose(movetool.copy_offsets(d, "ARRAY", 3), [d, 2 * d, 3 * d]))
check("divide offsets", np.allclose(movetool.copy_offsets(d, "DIVIDE", 3), [d / 3, 2 * d / 3, d]))
check("no offsets", movetool.copy_offsets(d, "NONE", 3) == [])

# -- plain move: objects, parents -----------------------------------------------
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1, 2, 3))
cube = ctx.active_object
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(4, 0, 0))
ball = ctx.active_object
ball.parent = cube
ball.matrix_parent_inverse = cube.matrix_world.inverted()
ctx.view_layer.update()
b0 = W(ball)[:3, 3].copy()
select(cube, ball)
r = bpy.ops.nproto.move(base=(1, 2, 3), offset=(0.5, -1.0, 2.0))
ctx.view_layer.update()
check("move finished", r == {'FINISHED'}, r)
check("cube moved", np.allclose(W(cube)[:3, 3], (1.5, 1.0, 5.0)), W(cube)[:3, 3])
check("child moved once, with its parent", np.allclose(W(ball)[:3, 3], b0 + (0.5, -1.0, 2.0)), W(ball)[:3, 3])
check("roots skip the child", movetool.roots([cube, ball]) == [cube])
check("zero move cancels", bpy.ops.nproto.move(offset=(0, 0, 0)) == {'CANCELLED'})

# -- pivot to the base point ------------------------------------------------------
bpy.data.objects.remove(ball)
select(cube)
P0 = np.array([tuple(cube.matrix_world @ v.co) for v in cube.data.vertices])
r = bpy.ops.nproto.move(base=(2.0, 1.5, 5.5), offset=(1.0, 0.0, 0.0), set_pivot=True)
ctx.view_layer.update()
P1 = np.array([tuple(cube.matrix_world @ v.co) for v in cube.data.vertices])
check("pivot: origin ends on the target", np.allclose(W(cube)[:3, 3], (3.0, 1.5, 5.5)), W(cube)[:3, 3])
check("pivot: geometry moved by the offset only", np.allclose(P1, P0 + (1.0, 0.0, 0.0)))

# -- plain mesh copies --------------------------------------------------------------
select(cube)
n0 = len(bpy.data.objects)
M0 = W(cube)
r = bpy.ops.nproto.move(base=(0, 0, 0), offset=(0, 2, 0), copies="ARRAY", count=3)
new = [o for o in bpy.data.objects if o.name.startswith("Cube") and o is not cube]
check("mesh copies made", r == {'FINISHED'} and len(bpy.data.objects) == n0 + 3, len(bpy.data.objects) - n0)
ys = sorted(round(W(o)[1, 3] - M0[1, 3], 6) for o in new)
check("mesh copies one move apart", np.allclose(ys, [2, 4, 6]), ys)
check("original stays", np.allclose(W(cube), M0))
check("copies own their mesh", all(o.data is not cube.data for o in new))
for o in new:
    bpy.data.objects.remove(o)

# -- NURBS copies: a linear Pattern ----------------------------------------------------
bpy.ops.nproto.add_primitive(kind="BOX", width=1.0, depth=0.5, height=0.4)
box = ctx.active_object
check("box is nurbs", store.is_nurbs(box))
select(box)
Mb = W(box)
r = bpy.ops.nproto.move(base=tuple(Mb[:3, 3]), offset=(3.0, 1.0, 0.0), copies="DIVIDE", count=4)
cps = [o for o in bpy.data.objects if o.get("nproto_in_0") is box and pattern.pattern_of(o) is not None]
check("divide: 4 pattern copies", r == {'FINISHED'} and len(cps) == 4, len(cps))
offs = sorted([tuple(np.round(W(o)[:3, 3] - Mb[:3, 3], 6)) for o in cps])
want = sorted([tuple(np.round(np.array([3.0, 1.0, 0.0]) * k / 4, 6)) for k in (1, 2, 3, 4)])
check("divide: copies spread to the target", np.allclose(offs, want), offs)
p = pattern.pattern_of(cps[0])[1]
check("divide: pattern by length", p["space"] == "EXTENT" and p["count_x"] == 5
      and abs(p["distance_x"] - np.hypot(3, 1)) < 1e-9, p)
check("copies selected with the source", box.select_get() and all(o.select_get() for o in cps))
check("copies carry the b-rep", all(store.is_nurbs(o) for o in cps))
grp = p["group"]
check("one group", all(pattern.pattern_of(o)[1]["group"] == grp for o in cps))
for o in cps:
    bpy.data.objects.remove(o)

select(box)
r = bpy.ops.nproto.move(base=(0, 0, 0), offset=(0.0, 0.0, 1.5), copies="ARRAY", count=2)
cps = [o for o in bpy.data.objects if o.get("nproto_in_0") is box and pattern.pattern_of(o) is not None]
zs = sorted(round(W(o)[2, 3] - Mb[2, 3], 6) for o in cps)
check("array: copies one move apart", np.allclose(zs, [1.5, 3.0]), zs)
check("array: pattern by spacing", pattern.pattern_of(cps[0])[1]["space"] == "SPACING")
# the source changes: the copies follow (history)
if features.history_enabled(ctx):
    box.location.x += 0.0          # (placement only: the copies keep their offset to the source's b-rep)
check("array: source not moved", np.allclose(W(box), Mb))

# -- Edit Mode: the selected points move ---------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
pl = ctx.active_object
pl.scale = (2.0, 1.0, 1.0)
ctx.view_layer.update()
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
r = bpy.ops.nproto.move(offset=(1.0, 0.0, 0.5))
bpy.ops.object.mode_set(mode='OBJECT')
Pw = np.array([tuple(pl.matrix_world @ v.co) for v in pl.data.vertices])
check("edit mesh: points moved in world", r == {'FINISHED'} and np.allclose(Pw.mean(axis=0), (1.0, 0.0, 0.5)),
      Pw.mean(axis=0))
check("edit mesh: size kept", np.allclose(np.ptp(Pw, axis=0), (4.0, 2.0, 0.0)), np.ptp(Pw, axis=0))
check("edit mesh: object stays", np.allclose(W(pl)[:3, 3], 0))

bpy.ops.curve.primitive_bezier_curve_add(location=(0, 0, 0))
cv = ctx.active_object
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.curve.select_all(action='DESELECT')
bp = cv.data.splines[0].bezier_points[0]
bp.select_control_point = True
bp.handle_left_type = bp.handle_right_type = 'FREE'     # (auto handles are recomputed on leaving Edit Mode)
h0, c0 = Vector(bp.handle_left), Vector(bp.co)
other = Vector(cv.data.splines[0].bezier_points[1].co)
r = bpy.ops.nproto.move(offset=(0.0, 0.0, 1.0), copies="ARRAY", count=2)
bpy.ops.object.mode_set(mode='OBJECT')
bp = cv.data.splines[0].bezier_points[0]
check("edit curve: point and handles moved", (bp.co - c0 - Vector((0, 0, 1))).length < 1e-6
      and (bp.handle_left - h0 - Vector((0, 0, 1))).length < 1e-6)
check("edit curve: others stay", (cv.data.splines[0].bezier_points[1].co - other).length < 1e-6)
check("edit: no copies made", len([o for o in bpy.data.objects if o.type == 'CURVE']) == 1)

# -- Rotate: pure helpers -------------------------------------------------------------
import math
check("angle 90", movetool.parse_angle("90") == 90.0)
check("angle 90/4", abs(movetool.parse_angle("90/4") - 22.5) < 1e-12)
check("angle -30°", movetool.parse_angle("-30°") == -30.0)
check("angle junk", movetool.parse_angle("abc") is None and movetool.parse_angle("") is None)
z = np.array([0.0, 0.0, 1.0])
check("signed +90", abs(movetool.signed_angle((1, 0, 0), (0, 1, 0), z) - 90) < 1e-9)
check("signed -90", abs(movetool.signed_angle((0, 1, 0), (1, 0, 0), z) + 90) < 1e-9)
check("signed seen in the plane", abs(movetool.signed_angle((1, 0, 5), (0, 2, -3), z) - 90) < 1e-9)
check("signed along the axis", movetool.signed_angle((0, 0, 1), (1, 0, 0), z) is None)
check("copies *3 @90", np.allclose(movetool.copy_angles(90, "ARRAY", 3), [90, 180, 270]))
check("copies *4 @90 skip the full turn", np.allclose(movetool.copy_angles(90, "ARRAY", 4), [90, 180, 270]))
check("copies /3 @90", np.allclose(movetool.copy_angles(90, "DIVIDE", 3), [30, 60, 90]))
check("copies /4 @360", np.allclose(movetool.copy_angles(360, "DIVIDE", 4), [90, 180, 270]))
T = movetool.rotation_matrix((1, 0, 0), z, 90)
check("rotation about a point", np.allclose(T @ [2, 0, 0, 1], [1, 1, 0, 1]))
# the circular pattern makes exactly copy_angles' turns
for deg, mode, n in ((90, "ARRAY", 3), (90, "ARRAY", 4), (-40, "DIVIDE", 4), (360, "DIVIDE", 6), (30, "ARRAY", 2)):
    p = movetool.circular_params((1, 2, 0), z, deg, mode, n)
    got = [pattern.copy_transform(p, None, k) for k in range(1, pattern.total_count(p))]
    want = [movetool.rotation_matrix((1, 2, 0), z, a) for a in movetool.copy_angles(deg, mode, n)]
    check(f"pattern = copy angles {deg} {mode} {n}", len(got) == len(want)
          and all(np.allclose(g, w) for g, w in zip(got, want)), (len(got), len(want)))

# -- Rotate: objects ------------------------------------------------------------------
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(3, 0, 0))
cube = ctx.active_object
select(cube)
M0 = W(cube)
r = bpy.ops.nproto.rotate(center=(1, 0, 0), axis=(0, 0, 1), angle=math.radians(90))
ctx.view_layer.update()
check("rotate finished", r == {'FINISHED'}, r)
check("rotate: about the centre", np.allclose(W(cube), movetool.rotation_matrix((1, 0, 0), z, 90) @ M0, atol=1e-6), W(cube)[:3, 3])
check("rotate: zero turn cancels", bpy.ops.nproto.rotate(angle=0.0) == {'CANCELLED'})

select(cube)
M0 = W(cube)
r = bpy.ops.nproto.rotate(center=(0, 0, 0), axis=(1, 0, 0), angle=math.radians(45), copies="ARRAY", count=2)
new = [o for o in bpy.data.objects if o is not cube]
check("rotate: mesh copies", r == {'FINISHED'} and len(new) == 2, len(new))
want = [movetool.rotation_matrix((0, 0, 0), (1, 0, 0), a) @ M0 for a in (45, 90)]
check("rotate: mesh copies turned", all(any(np.allclose(W(o), w, atol=1e-6) for o in new) for w in want))
check("rotate: original stays", np.allclose(W(cube), M0))
for o in new:
    bpy.data.objects.remove(o)

select(cube)
P0 = np.array([tuple(cube.matrix_world @ v.co) for v in cube.data.vertices])
r = bpy.ops.nproto.rotate(center=(5, 0, 0), axis=(0, 0, 1), angle=math.radians(180), set_pivot=True)
ctx.view_layer.update()
P1 = np.array([tuple(cube.matrix_world @ v.co) for v in cube.data.vertices])
R180 = movetool.rotation_matrix((5, 0, 0), z, 180)
check("rotate pivot: origin on the centre", np.allclose(W(cube)[:3, 3], (5, 0, 0)), W(cube)[:3, 3])
check("rotate pivot: geometry turned about the centre", np.allclose(P1, P0 @ R180[:3, :3].T + R180[:3, 3]))

bpy.ops.nproto.add_primitive(kind="BOX", width=1.0, depth=0.5, height=0.4)
box = ctx.active_object
box.location = (2.0, 0.0, 0.0)
ctx.view_layer.update()
select(box)
Mb = W(box)
r = bpy.ops.nproto.rotate(center=(0, 0, 0), axis=(0, 0, 1), angle=math.radians(90), copies="ARRAY", count=4)
cps = [o for o in bpy.data.objects if o.get("nproto_in_0") is box and pattern.pattern_of(o) is not None]
check("rotate *4 @90: three nurbs copies (none on the original)", r == {'FINISHED'} and len(cps) == 3, len(cps))
want = [movetool.rotation_matrix((0, 0, 0), z, a) @ Mb for a in (90, 180, 270)]
check("rotate: nurbs copies turned", all(any(np.allclose(W(o), w, atol=1e-6) for o in cps) for w in want))
p = pattern.pattern_of(cps[0])[1]
check("rotate: a circular pattern", p["mode"] == "CIRCULAR" and p["count"] == 4 and abs(p["angle"] - 360) < 1e-9, p)
for o in cps:
    bpy.data.objects.remove(o)
select(box)
r = bpy.ops.nproto.rotate(center=(0, 0, 0), axis=(0, 0, 1), angle=math.radians(-120), copies="DIVIDE", count=2)
cps = [o for o in bpy.data.objects if o.get("nproto_in_0") is box and pattern.pattern_of(o) is not None]
want = [movetool.rotation_matrix((0, 0, 0), z, a) @ Mb for a in (-60, -120)]
check("rotate /2 @-120: copies at -60 and -120", len(cps) == 2
      and all(any(np.allclose(W(o), w, atol=1e-6) for o in cps) for w in want), len(cps))
for o in cps:
    bpy.data.objects.remove(o)

# -- Rotate: Edit Mode --------------------------------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(1, 0, 0))
pl = ctx.active_object
pl.scale = (2.0, 1.0, 1.0)
ctx.view_layer.update()
Pw0 = np.array([tuple(pl.matrix_world @ v.co) for v in pl.data.vertices])
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
r = bpy.ops.nproto.rotate(center=(0, 0, 0), axis=(0, 0, 1), angle=math.radians(90))
bpy.ops.object.mode_set(mode='OBJECT')
Pw = np.array([tuple(pl.matrix_world @ v.co) for v in pl.data.vertices])
R90 = movetool.rotation_matrix((0, 0, 0), z, 90)
check("rotate edit mesh: points turned in world", r == {'FINISHED'}
      and np.allclose(Pw, Pw0 @ R90[:3, :3].T + R90[:3, 3], atol=1e-6))
check("rotate edit mesh: object stays", np.allclose(W(pl)[:3, 3], (1, 0, 0)))

# -- Scale: pure helpers ------------------------------------------------------------------
pt = lambda T, p: (T @ np.append(np.asarray(p, float), 1.0))[:3]
check("scale uniform about C", np.allclose(pt(movetool.scale_matrix((1, 0, 0), 2.0), (2, 1, 1)), (3, 2, 2)))
check("scale 1D along X", np.allclose(pt(movetool.scale_matrix((1, 0, 0), 2.0, "ONE", (1, 0, 0)), (2, 1, 1)), (3, 1, 1)))
check("scale 1D along a slant", np.allclose(pt(movetool.scale_matrix((0, 0, 0), 3.0, "ONE", (1, 1, 0)), (1, -1, 0)),
                                           (1, -1, 0)))
check("scale 2D square to Z", np.allclose(pt(movetool.scale_matrix((0, 0, 0), 2.0, "TWO", (0, 0, 1)), (2, 1, 1)), (4, 2, 1)))
check("ref length uniform", abs(movetool.ref_length((0, 0, 0), (3, 4, 0)) - 5) < 1e-12)
check("ref length 1D signed", abs(movetool.ref_length((1, 0, 0), (-1, 5, 0), "ONE", (1, 0, 0)) + 2) < 1e-12)
check("ref length 2D", abs(movetool.ref_length((0, 0, 0), (3, 4, 9), "TWO", (0, 0, 1)) - 5) < 1e-12)
for t, v in (("2", 2.0), ("x2", 2.0), ("2x", 2.0), ("1/4", 0.25), ("×1.5", 1.5)):
    check(f"factor {t!r}", movetool.parse_factor(t) == v)
check("factor junk", movetool.parse_factor("x") is None)

# -- Scale: objects ------------------------------------------------------------------------
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(2, 0, 0))
cube = ctx.active_object
select(cube)
M0 = W(cube)
r = bpy.ops.nproto.scale(center=(1, 0, 0), factor=3.0)
ctx.view_layer.update()
check("scale finished", r == {'FINISHED'}, r)
check("scale: about the centre", np.allclose(W(cube), movetool.scale_matrix((1, 0, 0), 3.0) @ M0, atol=1e-6),
      W(cube)[:3, 3])
check("scale: factor 1 cancels", bpy.ops.nproto.scale(factor=1.0) == {'CANCELLED'})
check("scale: factor 0 cancels", bpy.ops.nproto.scale(factor=0.0) == {'CANCELLED'})

select(cube)
M0 = W(cube)
r = bpy.ops.nproto.scale(center=(0, 0, 0), factor=0.5, mode="ONE", direction=(0, 1, 0))
ctx.view_layer.update()
check("scale 1D: along Y only", np.allclose(W(cube), movetool.scale_matrix((0, 0, 0), 0.5, "ONE", (0, 1, 0)) @ M0,
                                           atol=1e-6), np.array(cube.scale))

select(cube)
P0 = np.array([tuple(cube.matrix_world @ v.co) for v in cube.data.vertices])
r = bpy.ops.nproto.scale(center=(4, 0, 0), factor=2.0, set_pivot=True)
ctx.view_layer.update()
P1 = np.array([tuple(cube.matrix_world @ v.co) for v in cube.data.vertices])
S2 = movetool.scale_matrix((4, 0, 0), 2.0)
check("scale pivot: origin on the centre", np.allclose(W(cube)[:3, 3], (4, 0, 0)), W(cube)[:3, 3])
check("scale pivot: geometry scaled about the centre", np.allclose(P1, P0 @ S2[:3, :3].T + S2[:3, 3], atol=1e-5))

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0), rotation=(0, 0, math.radians(30)))
turned = ctx.active_object
off = movetool.transform_objects([turned], movetool.scale_matrix((0, 0, 0), 2.0, "ONE", (1, 0, 0)))
check("scale 1D on a turned object: reported (no shear)", off == [turned.name], off)
ctx.view_layer.update()
off = movetool.transform_objects([turned], movetool.scale_matrix((0, 0, 0), 2.0))
check("uniform on a turned object: exact", off == [], off)

bpy.ops.nproto.add_primitive(kind="BOX", width=1.0, depth=0.5, height=0.4)
box = ctx.active_object
box.location = (2.0, 1.0, 0.0)
ctx.view_layer.update()
select(box)
Mb = W(box)
r = bpy.ops.nproto.scale(center=(0, 0, 0), factor=1.5)
ctx.view_layer.update()
check("scale nurbs: matrix", r == {'FINISHED'} and np.allclose(W(box), movetool.scale_matrix((0, 0, 0), 1.5) @ Mb,
                                                                atol=1e-6))
check("scale nurbs: still nurbs, no copies", store.is_nurbs(box)
      and not [o for o in bpy.data.objects if o.get("nproto_in_0") is box])

# -- Scale: Edit Mode ----------------------------------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(1, 0, 0))
pl = ctx.active_object
pl.rotation_euler = (0, 0, math.radians(30))
ctx.view_layer.update()
Pw0 = np.array([tuple(pl.matrix_world @ v.co) for v in pl.data.vertices])
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
r = bpy.ops.nproto.scale(center=(0, 0, 0), factor=2.0, mode="ONE", direction=(1, 0, 0))
bpy.ops.object.mode_set(mode='OBJECT')
Pw = np.array([tuple(pl.matrix_world @ v.co) for v in pl.data.vertices])
S1 = movetool.scale_matrix((0, 0, 0), 2.0, "ONE", (1, 0, 0))
check("scale edit mesh: 1D in world on a turned object (sheared points)", r == {'FINISHED'}
      and np.allclose(Pw, Pw0 @ S1[:3, :3].T + S1[:3, 3], atol=1e-5))

# -- the hold key ------------------------------------------------------------------
kc = ctx.window_manager.keyconfigs.addon
items = [(km.name, k.idname) for km, k in movetool._keymaps]
check("hold G / R bound in object / edit keymaps", {n for n, _ in items} == set(movetool.KEYMAPS)
      and all(i == "nproto.transform_hold" for _, i in items)
      and sorted((km.name, k.type, k.properties.kind) for km, k in movetool._keymaps)
      == sorted((n, key, kind) for n in movetool.KEYMAPS for kind, (key, _a, _b) in movetool.HOLD_TOOLS.items()),
      items)
check("hold default", movetool.hold_ms() == 300 and movetool.hold_enabled())
check("hold . : tap is Blender's Pivot Point pie, hold is Move Pivot",
      movetool.HOLD_TOOLS["PIVOT"] == ('PERIOD', ("wm.call_menu_pie", {"name": "VIEW3D_MT_pivot_pie"}),
                                       "nproto.move_pivot"))

# -- Move Pivot: toolbar tool and right-click menus ------------------------------
from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
from blender_nurbs import pivottool
_tb = ToolSelectPanelHelper._tool_class_from_space_type('VIEW_3D')
_ids = lambda m: [getattr(i, "idname", None) for i in _tb.tools_from_context(ctx, mode=m)]
_places = {m: (_ids(m).index(pivottool.TOOL_ID) - _ids(m).index("builtin.transform")
               if pivottool.TOOL_ID in _ids(m) else None) for m in pivottool.TOOL_MODES}
check("Move Pivot tool in the toolbar right after Transform, Object and Edit Modes",
      all(v == 1 for v in _places.values()), _places)
check("Move Pivot in the right-click menus",
      all(pivottool.menu_transform in getattr(bpy.types, m).draw._draw_funcs
          for m in ("VIEW3D_MT_object_context_menu", "VIEW3D_MT_edit_mesh_context_menu",
                    "VIEW3D_MT_edit_curve_context_menu")))
check("F3 finds Move Pivot by 'origin' too", "Origin" in bpy.types.NPROTO_OT_move_pivot.bl_label)
check("Move Pivot icon file exists", os.path.isfile(pivottool.ICON + ".dat"))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
