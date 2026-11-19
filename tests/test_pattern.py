"""Pattern (linear / circular copies):  blender -b --factory-startup --python tests/test_pattern.py"""
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

def select(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0]

def run(**kw):
    r = bpy.ops.nproto.pattern(**kw)
    sel = [o for o in ctx.selected_objects]
    return r, sel

def copies_of(src):
    return sorted([o for o in bpy.data.objects if o.get("nproto_in_0") is src
                   and pattern.pattern_of(o) is not None], key=lambda o: pattern.pattern_of(o)[1]["index"])

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
r, sel = run(mode="LINEAR", count=4, distance=2.0, axis="X")
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
r = bpy.ops.nproto.pattern(group=grp, objects=json.dumps([PL.name]), mode="LINEAR", count=6, distance=1.0, axis="X")
cps2 = copies_of(PL)
check("edit: 5 copies, the 3 there were kept (same objects)",
      r == {'FINISHED'} and len(cps2) == 5 and [c.name for c in cps2[:3]] == names, f"{len(cps2)}")
P4 = surf_pts(PL)
err = max(np.abs(surf_pts(c) - (P4 + [1.0 * (k + 1), 0, 0])).max() for k, c in enumerate(cps2))
check("edit: the copies at the new spacing (the hand move dropped)", err < 1e-6, f"{err:.2e}")
r = bpy.ops.nproto.pattern(group=grp, objects=json.dumps([PL.name]), mode="LINEAR", count=2, distance=1.0, axis="X")
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


# --- extent and a grid ------------------------------------------------------
bpy.ops.nproto.add_primitive(kind="PLANE", width=0.5, depth=0.5)
GP = ctx.active_object
GP.location = (0.0, -10.0, 0.0)
ctx.view_layer.update()
select(GP)
G0 = surf_pts(GP)
r, _ = run(mode="LINEAR", count=4, distance=6.0, space="EXTENT", axis="X", count_b=3, distance_b=1.0,
           axis_b="Y")
gc = copies_of(GP)
want = {i + 4 * j: (2.0 * i, 0.5 * j, 0.0) for j in range(3) for i in range(4) if i + 4 * j}
err = max(np.abs(surf_pts(c) - (G0 + want[pattern.pattern_of(c)[1]["index"]])).max() for c in gc)
check("grid: 4 x 3 (11 copies), Length 6 = spacing 2, rows 1 over 2 = 0.5", len(gc) == 11 and err < 1e-6,
      f"{len(gc)} {err:.2e}")


# --- circular: a box (a solid, several faces) about the world Z axis ----------
bpy.ops.nproto.add_primitive(kind="BOX", width=1.0, depth=0.5, height=0.4)
BX = ctx.active_object
BX.location = (5.0, 0.0, 0.0)
ctx.view_layer.update()
select(BX)
B0 = surf_pts(BX)
nf = store.get_brep(BX).face_count()
r, _ = run(mode="CIRCULAR", count=6, angle=360.0, rot_axis="Z", center="WORLD")
bc = copies_of(BX)
err = 0.0
for c in bc:
    k = pattern.pattern_of(c)[1]["index"]
    T = np.eye(4); T[:3, :3] = pattern._rotation((0, 0, 1), 60.0 * k)
    err = max(err, np.abs(surf_pts(c) - moved(B0, T)).max())
check("circular: 5 copies, 60 degrees apart (a full turn: none on the source)",
      len(bc) == 5 and err < 1e-6, f"{len(bc)} {err:.2e}")
check("circular: copies keep every face, trimmed", all(store.get_brep(c).face_count() == nf
                                                        and not store.is_natural(c) for c in bc))
BX.location = (6.0, 0.0, 0.0)
ctx.view_layer.update()
follow(bc)
B1 = surf_pts(BX)
err = 0.0
for c in bc:
    k = pattern.pattern_of(c)[1]["index"]
    T = np.eye(4); T[:3, :3] = pattern._rotation((0, 0, 1), 60.0 * k)
    err = max(err, np.abs(surf_pts(c) - moved(B1, T)).max())
check("circular: the copies follow the source round the axis", err < 1e-6, f"{err:.2e}")

# a part of a turn, not rotated, about the 3D cursor
ctx.scene.cursor.location = (6.0, 2.0, 0.0)
select(BX)
r, _ = run(mode="CIRCULAR", count=3, angle=90.0, rot_axis="Z", center="CURSOR", rotate=False)
pc = [c for c in copies_of(BX) if c not in bc]
a = pattern.anchor(BX)
O = np.array([6.0, 2.0, 0.0])
err = 0.0
for c in pc:
    k = pattern.pattern_of(c)[1]["index"]
    R = pattern._rotation((0, 0, 1), 45.0 * k)
    err = max(err, np.abs(surf_pts(c) - (B1 + (R @ (a - O) + O - a))).max())
check("circular 90 degrees, 3: copies at 45 and 90, kept upright, about the cursor",
      len(pc) == 2 and err < 1e-6
      and all(np.abs(W(c)[:3, :3] - W(BX)[:3, :3]).max() < 1e-6 for c in pc), f"{len(pc)} {err:.2e}")
ctx.scene.cursor.location = (0.0, 0.0, 0.0)
follow(pc)
err = 0.0
for c in pc:
    k = pattern.pattern_of(c)[1]["index"]
    R = pattern._rotation((0, 0, 1), 45.0 * k)
    err = max(err, np.abs(surf_pts(c) - (B1 + (R @ (a - O) + O - a))).max())
check("circular: moving the cursor later leaves the pattern where it was", err < 1e-6, f"{err:.2e}")

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


# --- curves -------------------------------------------------------------------
cu = bpy.data.curves.new("Prof", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('NURBS'); sp.points.add(3)
for q, c in zip(sp.points, [(0, 0, 0), (1, 0.5, 0), (2, -0.5, 0), (3, 0, 0)]):
    q.co = (*c, 1.0)
CV = bpy.data.objects.new("Prof", cu); ctx.collection.objects.link(CV)
CV.location = (0.0, 20.0, 0.0)
ctx.view_layer.update()
select(CV)
r, _ = run(mode="LINEAR", count=3, distance=1.5, axis="Z")
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
r, _ = run(mode="LINEAR", count=3, distance=1.5, axis="X")
pl = [o for o in ctx.selected_objects if o is not NP]
check("no history: plain NURBS copies", r == {'FINISHED'} and len(pl) == 2
      and all(store.is_nurbs(o) and not features.has_history(o) for o in pl))
ctx.scene.nproto_settings.construction_history = True

# --- nothing usable selected ---------------------------------------------------
bpy.ops.mesh.primitive_cube_add()
select(ctx.active_object)
try:
    r = bpy.ops.nproto.pattern(mode="LINEAR", count=3, distance=1.0)
except RuntimeError:
    r = {'CANCELLED'}
check("a plain mesh is not patterned (Blender's Array is for those)", r == {'CANCELLED'})

# --- too many copies ------------------------------------------------------------
select(GP)
try:
    r = bpy.ops.nproto.pattern(mode="LINEAR", count=100, distance=1.0, count_b=30, distance_b=1.0)
except RuntimeError:
    r = {'CANCELLED'}
check("too many copies are refused", r == {'CANCELLED'})

# --- the tool table lists it ---------------------------------------------------
from blender_nurbs import toolgroups
ids = [(t[0], (t[3] or {}).get("mode")) for t in toolgroups.tools()]
check("toolbar: Linear and Circular Pattern", ("nproto.pattern", "LINEAR") in ids and ("nproto.pattern", "CIRCULAR") in ids)
icons = bpy.types.UILayout.bl_rna.functions["prop"].parameters["icon"].enum_items.keys()
check("toolbar icons exist", all(t[2] in icons for t in toolgroups.tools() if t[0] == "nproto.pattern"))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
