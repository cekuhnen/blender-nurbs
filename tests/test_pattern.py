"""Pattern (linear / circular / path copies):  blender -b --factory-startup --python tests/test_pattern.py"""
import json, math, os, sys
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, geom, pattern, store

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

def surf_pts(o, n=5):
    """World points on each face's surface (a grid over its domain)."""
    out = []
    M = W(o)
    for s in store.get_surfaces(o):
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        for i in range(n):
            for j in range(n):
                p = np.array(s.evaluate(u0 + (u1 - u0) * i / (n - 1), v0 + (v1 - v0) * j / (n - 1)), float)
                out.append(M[:3, :3] @ p + M[:3, 3])
    return np.array(out)

def curve_pts(o):
    M = W(o)
    return np.array([M[:3, :3] @ np.array(q.co[:3]) + M[:3, 3] for sp in o.data.splines for q in sp.points])

def moved(P, T):
    return P @ T[:3, :3].T + T[:3, 3]

def rotz(deg):
    T = np.eye(4); T[:3, :3] = pattern._rotation((0, 0, 1), deg); return T

def select(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0]

def run(**kw):
    r = bpy.ops.nproto.pattern(**kw)
    return r, list(ctx.selected_objects)

def copies_of(src, group=None):
    out = [o for o in bpy.data.objects if o.get("nproto_in_0") is src and pattern.pattern_of(o) is not None
           and (group is None or pattern.pattern_of(o)[1]["group"] == group)]
    return sorted(out, key=lambda o: pattern.pattern_of(o)[1]["index"])

def idx(c):
    return pattern.pattern_of(c)[1]["index"]

def follow(objs):
    for o in objs:
        features.rebuild(o)


# --- linear: an untrimmed plane, turned and moved --------------------------
bpy.ops.nproto.add_primitive(kind="PLANE", width=1.0, depth=0.6)
PL = ctx.active_object
PL.location = (1.0, 2.0, 0.5)
PL.rotation_euler = (0.3, 0.0, 0.7)
ctx.view_layer.update()
select(PL)
P0 = surf_pts(PL)
r, sel = run(mode="LINEAR", count_x=4, distance_x=2.0)
cps = copies_of(PL)
check("linear: 3 copies made, the source and copies selected", r == {'FINISHED'} and len(cps) == 3
      and set(sel) == set(cps) | {PL}, f"{len(cps)}")
check("linear: copies are NURBS objects, untrimmed like their source",
      all(store.is_nurbs(c) and store.is_natural(c) for c in cps))
err = max(np.abs(surf_pts(c) - (P0 + [2.0 * (k + 1), 0, 0])).max() for k, c in enumerate(cps))
check("linear: copy k is the source moved k * spacing along X", err < 1e-6, f"{err:.2e}")
check("linear: copy matrix = move @ source matrix",
      max(np.abs(W(c) - pattern._translation((2.0 * (k + 1), 0, 0)) @ W(PL)).max() for k, c in enumerate(cps)) < 1e-6)
check("linear: copies take the source's mesh (no new tessellation)",
      all(len(c.data.polygons) == len(PL.data.polygons) and c.data is not PL.data for c in cps))
check("linear: copies have the pattern feature", all(features.history(c)["op"] == "pattern" for c in cps))
grp = pattern.pattern_of(cps[0])[1]["group"]
check("linear: one group for the run", all(pattern.pattern_of(c)[1]["group"] == grp for c in cps))
check("linear: a copy built as made needs no rebuild",
      all(c.get(features.PROP_KEY) == features.build_key(c) for c in cps))

# the source moves: the copies follow
PL.location = (1.0, 3.0, 1.5)
ctx.view_layer.update()
follow(cps)
P1 = surf_pts(PL)
err = max(np.abs(surf_pts(c) - (P1 + [2.0 * (k + 1), 0, 0])).max() for k, c in enumerate(cps))
check("history: copies follow the source when it moves", err < 1e-6, f"{err:.2e}")
check("history: a rebuilt copy is still untrimmed", all(store.is_natural(c) for c in cps))

# the source changes shape: the copies follow
store.set_geometry(PL, pattern._moved_surface(store.get_surfaces(PL)[0], np.diag([1.5, 1.0, 1.0, 1.0])))
follow(cps)
P2 = surf_pts(PL)
err = max(np.abs(surf_pts(c) - (P2 + [2.0 * (k + 1), 0, 0])).max() for k, c in enumerate(cps))
check("history: copies follow the source's new shape", err < 1e-6, f"{err:.2e}")

# a copy moved by hand keeps its offset when the source changes again
cps[0].location = Vector(cps[0].location) + Vector((0, 0, 3.0))
ctx.view_layer.update()
PL.location = (1.0, 3.0, 2.5)
ctx.view_layer.update()
follow(cps)
P3 = surf_pts(PL)
err0 = np.abs(surf_pts(cps[0]) - (P3 + [2.0, 0, 3.0])).max()
err1 = np.abs(surf_pts(cps[1]) - (P3 + [4.0, 0, 0])).max()
check("history: a copy moved by hand keeps its offset", err0 < 1e-6 and err1 < 1e-6, f"{err0:.2e} {err1:.2e}")

# Edit Pattern: more copies (the old ones kept), then fewer
names = [c.name for c in cps]
r = bpy.ops.nproto.pattern(group=grp, objects=json.dumps([PL.name]), mode="LINEAR", count_x=6, distance_x=1.0)
cps2 = copies_of(PL)
check("edit: 5 copies, the 3 there were kept (same objects)",
      r == {'FINISHED'} and len(cps2) == 5 and [c.name for c in cps2[:3]] == names, f"{len(cps2)}")
P4 = surf_pts(PL)
err = max(np.abs(surf_pts(c) - (P4 + [1.0 * (k + 1), 0, 0])).max() for k, c in enumerate(cps2))
check("edit: the copies at the new spacing (the hand move dropped)", err < 1e-6, f"{err:.2e}")
r = bpy.ops.nproto.pattern(group=grp, objects=json.dumps([PL.name]), mode="LINEAR", count_x=2, distance_x=1.0)
cps3 = copies_of(PL)
check("edit: fewer copies remove the rest", r == {'FINISHED'} and len(cps3) == 1
      and all(bpy.data.objects.get(n) is None for n in names[1:]), f"{len(cps3)}")

# Select Pattern
select(cps3[0])
bpy.ops.nproto.pattern_select(group=grp)
check("select: the copies and the original", set(ctx.selected_objects) == {PL, cps3[0]})

# Bake: a plain object, out of the group
features.bake(cps3[0])
check("bake: no history, geometry kept", not features.has_history(cps3[0]) and store.is_nurbs(cps3[0])
      and pattern.group_copies(grp) == [])


# --- grid, block, a turned widget, a widget lined up with a face ---------------
bpy.ops.nproto.add_primitive(kind="PLANE", width=0.5, depth=0.5)
GP = ctx.active_object
GP.location = (0.0, -10.0, 0.0)
ctx.view_layer.update()
select(GP)
G0 = surf_pts(GP)
r, _ = run(mode="LINEAR", space="EXTENT", count_x=4, distance_x=6.0, count_y=3, distance_y=1.0)
g1 = copies_of(GP)
want = {i + 4 * j: (2.0 * i, 0.5 * j, 0.0) for j in range(3) for i in range(4) if i + 4 * j}
err = max(np.abs(surf_pts(c) - (G0 + want[idx(c)])).max() for c in g1)
check("grid: 4 x 3 (11 copies), Length 6 = spacing 2, rows 1 over 2 = 0.5", len(g1) == 11 and err < 1e-6,
      f"{len(g1)} {err:.2e}")
select(GP)
r, _ = run(mode="LINEAR", count_x=2, distance_x=1.0, count_y=2, distance_y=2.0, count_z=3, distance_z=0.5)
g2 = [c for c in copies_of(GP) if c not in g1]
want = {i + 2 * j + 4 * k: (1.0 * i, 2.0 * j, 0.5 * k) for k in range(3) for j in range(2) for i in range(2)}
err = max(np.abs(surf_pts(c) - (G0 + want[idx(c)])).max() for c in g2)
check("block: 2 x 2 x 3 along X, Y and Z (11 copies)", len(g2) == 11 and err < 1e-6, f"{len(g2)} {err:.2e}")
select(GP)
r, _ = run(mode="LINEAR", count_x=3, distance_x=1.0, turn=30.0)
g3 = [c for c in copies_of(GP) if c not in g1 + g2]
d = np.array([math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0])
err = max(np.abs(surf_pts(c) - (G0 + idx(c) * d)).max() for c in g3)
check("turn: the widget turned 30 degrees: the row at 30 degrees", len(g3) == 2 and err < 1e-6, f"{err:.2e}")
select(GP)
r, _ = run(mode="LINEAR", count_y=3, count_x=1, distance_y=1.0, frame_x=(1, 0, 0), frame_z=(0, 1, 0))
g4 = [c for c in copies_of(GP) if c not in g1 + g2 + g3]
err = max(np.abs(surf_pts(c) - (G0 + idx(c) * np.array([0.0, 0.0, -1.0]))).max() for c in g4) if g4 else 1.0
check("face frame: Z along world Y: the widget's Y runs along -Z (Z x X)", len(g4) == 2 and err < 1e-6, f"{err:.2e}")


# --- circular: a box (a solid, several faces) about a Z axis through the centre ----
bpy.ops.nproto.add_primitive(kind="BOX", width=1.0, depth=0.5, height=0.4)
BX = ctx.active_object
BX.location = (5.0, 0.0, 0.0)
ctx.view_layer.update()
select(BX)
B0 = surf_pts(BX)
nf = store.get_brep(BX).face_count()
r, _ = run(mode="CIRCULAR", count=6, angle=360.0, center_point=(0, 0, 0), frame_z=(0, 0, 1))
bc = copies_of(BX)
err = max(np.abs(surf_pts(c) - moved(B0, rotz(60.0 * idx(c)))).max() for c in bc)
check("circular: 5 copies, 60 degrees apart (a full turn: none on the source)",
      len(bc) == 5 and err < 1e-6, f"{len(bc)} {err:.2e}")
check("circular: copies keep every face, trimmed", all(store.get_brep(c).face_count() == nf
                                                        and not store.is_natural(c) for c in bc))
BX.location = (6.0, 0.0, 0.0)
ctx.view_layer.update()
follow(bc)
B1 = surf_pts(BX)
err = max(np.abs(surf_pts(c) - moved(B1, rotz(60.0 * idx(c)))).max() for c in bc)
check("circular: the copies follow the source round the axis", err < 1e-6, f"{err:.2e}")

# a part of a turn, not rotated, about a point of its own
select(BX)
O = np.array([6.0, 2.0, 0.0])
r, _ = run(mode="CIRCULAR", count=3, angle=90.0, center_point=tuple(O), frame_z=(0, 0, 1), rotate=False)
pc = [c for c in copies_of(BX) if c not in bc]
a = pattern.anchor(BX)
err = 0.0
for c in pc:
    R = pattern._rotation((0, 0, 1), 45.0 * idx(c))
    err = max(err, np.abs(surf_pts(c) - (B1 + (R @ (a - O) + O - a))).max())
check("circular 90 degrees, 3: copies at 45 and 90, kept upright, about the widget's centre",
      len(pc) == 2 and err < 1e-6
      and all(np.abs(W(c)[:3, :3] - W(BX)[:3, :3]).max() < 1e-6 for c in pc), f"{len(pc)} {err:.2e}")
# a script without a centre: the 3D cursor
ctx.scene.cursor.location = (6.0, 0.0, 3.0)
select(BX)
r, _ = run(mode="CIRCULAR", count=4, angle=360.0, frame_z=(1, 0, 0))
xc = [c for c in copies_of(BX) if c not in bc + pc]
Tx = np.eye(4); Tx[:3, :3] = pattern._rotation((1, 0, 0), 90.0); Tx[:3, 3] = (0, 3.0, 3.0)    # about X through (6,0,3)
check("circular: no centre given: the 3D cursor; axis X", len(xc) == 3
      and np.abs(surf_pts(xc[0]) - moved(B1, Tx)).max() < 1e-6)
ctx.scene.cursor.location = (0.0, 0.0, 0.0)

# a picked axis: a circle's centre and normal
pts = [Vector((3.0 + 2.0 * math.cos(t), -1.0 + 2.0 * math.sin(t), 4.0))
       for t in (2 * math.pi * (i / 47) ** 1.3 for i in range(48))]
pts[-1] = pts[0]
c0, n0, closed = pattern.circle_or_line(pts)
check("pick: a closed curve gives its circle's centre and normal (uneven samples)",
      closed and np.abs(c0 - (3.0, -1.0, 4.0)).max() < 1e-6 and np.abs(n0 - (0, 0, 1)).max() < 1e-9,
      f"{c0} {n0}")
l0, d0, closed = pattern.circle_or_line([Vector((1, 1, 1)), Vector((1, 1, 2)), Vector((1, 1, 4))])
check("pick: an open curve runs from its start to its end", not closed and np.allclose(d0, (0, 0, 1)))


# --- path: along a circle (closed) and a line (open) --------------------------------
bpy.ops.curve.primitive_nurbs_circle_add(radius=5.0, location=(0, 40, 0))
CIR = ctx.active_object
CIR.name = "PathCircle"
bpy.ops.nproto.add_primitive(kind="BOX", width=0.4, depth=0.8, height=0.3)
PB = ctx.active_object
PB.location = (5.0, 40.0, 0.0)
ctx.view_layer.update()
Q0 = surf_pts(PB)
select(PB)
r, _ = run(mode="PATH", count=4, path_obj=CIR.name, path_ref="S0")
pp = copies_of(PB)
def about_center(deg, C=(0, 40, 0)):
    T = rotz(deg); C = np.array(C, float); T[:3, 3] = C - T[:3, :3] @ C; return T
# (Blender's circle runs clockwise seen from above: the copies go round that way)
err = min(max(np.abs(surf_pts(c) - moved(Q0, about_center(sg * 90.0 * idx(c)))).max() for c in pp)
          for sg in (1, -1)) if pp else 1.0
check("path: a closed circle, 4: spread all the way round, turned with the path",
      r == {'FINISHED'} and len(pp) == 3 and err < 2e-3, f"{len(pp)} {err:.2e}")
check("path: the path is the copies' second input", all(c.get("nproto_in_1") is CIR for c in pp))
# the path changes shape (an ellipse): the copies follow
before = [surf_pts(c) for c in pp]
CIR.scale = (1.6, 1.0, 1.0)
ctx.view_layer.update()
follow(pp)
prm = pattern.pattern_of(pp[0])[1]
err = max(np.abs(surf_pts(c) - moved(Q0, pattern.copy_transform(prm, PB, idx(c), (CIR, "S0")))).max() for c in pp)
gone = max(np.abs(surf_pts(c) - b0).max() for c, b0 in zip(pp, before))
check("path: copies follow the path when it changes", err < 1e-6 and gone > 0.1, f"{err:.2e} moved {gone:.2f}")
CIR.scale = (1, 1, 1)
ctx.view_layer.update()
follow(pp)

cu = bpy.data.curves.new("Rail", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('POLY'); sp.points.add(1)
sp.points[0].co = (0, -20, 0, 1); sp.points[1].co = (10, -20, 0, 1)
RAIL = bpy.data.objects.new("Rail", cu); ctx.collection.objects.link(RAIL)
bpy.ops.nproto.add_primitive(kind="PLANE", width=0.5, depth=0.5)
RP = ctx.active_object
RP.location = (0.0, -20.0, 1.0)
ctx.view_layer.update()
R0 = surf_pts(RP)
select(RP)
r, _ = run(mode="PATH", count=3, path_obj=RAIL.name, path_ref="S0")
rp = copies_of(RP)
err = max(np.abs(surf_pts(c) - (R0 + [5.0 * idx(c), 0, 0])).max() for c in rp)
check("path: an open line, 3, distance 0: from the source to the end", len(rp) == 2 and err < 1e-6, f"{err:.2e}")
select(RP)
r, _ = run(mode="PATH", count=4, path_obj=RAIL.name, path_ref="S0", space="SPACING", distance=2.0)
rp2 = [c for c in copies_of(RP) if c not in rp]
err = max(np.abs(surf_pts(c) - (R0 + [2.0 * idx(c), 0, 0])).max() for c in rp2)
check("path: spacing 2 along the line", len(rp2) == 3 and err < 1e-6, f"{err:.2e}")
# the path is selected with the source, active: it is the path, not copied
select(RAIL, RP)
r = bpy.ops.nproto.pattern('EXEC_DEFAULT', mode="PATH", count=2, path_obj=RAIL.name, path_ref="S0")
check("path: the path itself is not copied", not copies_of(RAIL))
select(PB)
r, _ = run(mode="PATH", count=4, path_obj=CIR.name, path_ref="S0", rotate=False)
pn = [c for c in copies_of(PB) if c not in pp]
check("path: not turned: only moved", len(pn) == 3
      and all(np.abs(W(c)[:3, :3] - W(PB)[:3, :3]).max() < 1e-6 for c in pn))
try:
    r = bpy.ops.nproto.pattern(mode="PATH", count=3)
except RuntimeError:
    r = {'CANCELLED'}
check("path: no path given is refused", r == {'CANCELLED'})


# --- the first Pattern's params still build --------------------------------------
old = {"mode": "LINEAR", "count": 3, "distance": 2.0, "axis": "Y", "count_b": 2, "distance_b": 1.0,
       "axis_b": "Z", "space": "SPACING"}
T = pattern.copy_transform(old, PL, 1 + 3 * 1)          # (1, 1): one along Y, one row up Z
check("old params: a row along Y with rows along Z", np.allclose(T[:3, 3], (0, 2.0, 1.0)), f"{T[:3, 3]}")
oldc = {"mode": "CIRCULAR", "count": 4, "angle": 360.0, "rot_axis": "Z", "center": "WORLD",
        "center_point": [0, 0, 0], "rotate": True}
T = pattern.copy_transform(oldc, PL, 1)
check("old params: circular about World Z", np.allclose(T[:3, :3], pattern._rotation((0, 0, 1), 90.0)))


# --- curves -------------------------------------------------------------------
cu = bpy.data.curves.new("Prof", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('NURBS'); sp.points.add(3)
for q, c in zip(sp.points, [(0, 0, 0), (1, 0.5, 0), (2, -0.5, 0), (3, 0, 0)]):
    q.co = (*c, 1.0)
CV = bpy.data.objects.new("Prof", cu); ctx.collection.objects.link(CV)
CV.location = (0.0, 20.0, 0.0)
ctx.view_layer.update()
select(CV)
r, _ = run(mode="LINEAR", count_x=1, count_z=3, distance_z=1.5)
cc = copies_of(CV)
C0 = curve_pts(CV)
err = max(np.abs(curve_pts(c) - (C0 + [0, 0, 1.5 * (k + 1)])).max() for k, c in enumerate(cc))
check("curves: copies are curves of their own, moved along Z", len(cc) == 2 and err < 1e-6
      and all(c.type == 'CURVE' and c.data is not CV.data for c in cc), f"{err:.2e}")
CV.data.splines[0].points[1].co = (1.0, 2.0, 0.0, 1.0)
follow(cc)
C1 = curve_pts(CV)
err = max(np.abs(curve_pts(c) - (C1 + [0, 0, 1.5 * (k + 1)])).max() for k, c in enumerate(cc))
check("curves: copies follow a CV edit of the source", err < 1e-6, f"{err:.2e}")
check("curves: no curve data left behind by rebuilds",
      len([d for d in bpy.data.curves if d.users == 0]) == 0)


# --- without construction history: plain copies --------------------------------
ctx.scene.nproto_settings.construction_history = False
bpy.ops.nproto.add_primitive(kind="PLANE")
NP = ctx.active_object
NP.location = (0.0, 30.0, 0.0)
ctx.view_layer.update()
select(NP)
r, _ = run(mode="LINEAR", count_x=3, distance_x=1.5)
pl = [o for o in ctx.selected_objects if o is not NP]
check("no history: plain NURBS copies", r == {'FINISHED'} and len(pl) == 2
      and all(store.is_nurbs(o) and not features.has_history(o) for o in pl))
ctx.scene.nproto_settings.construction_history = True

# --- nothing usable selected ---------------------------------------------------
bpy.ops.mesh.primitive_cube_add()
select(ctx.active_object)
try:
    r = bpy.ops.nproto.pattern(mode="LINEAR", count_x=3, distance_x=1.0)
except RuntimeError:
    r = {'CANCELLED'}
check("a plain mesh is not patterned (Blender's Array is for those)", r == {'CANCELLED'})

# --- too many copies ------------------------------------------------------------
select(GP)
try:
    r = bpy.ops.nproto.pattern(mode="LINEAR", count_x=100, distance_x=1.0, count_y=30, distance_y=1.0)
except RuntimeError:
    r = {'CANCELLED'}
check("too many copies are refused", r == {'CANCELLED'})

# --- the tool table lists it ---------------------------------------------------
from blender_nurbs import toolgroups
ids = [(t[0], (t[3] or {}).get("mode")) for t in toolgroups.tools()]
check("toolbar: Linear, Circular and Path Pattern",
      all(("nproto.pattern", m) in ids for m in ("LINEAR", "CIRCULAR", "PATH")))
icons = bpy.types.UILayout.bl_rna.functions["prop"].parameters["icon"].enum_items.keys()
check("toolbar icons exist", all(t[2] in icons for t in toolgroups.tools() if t[0] == "nproto.pattern"))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
