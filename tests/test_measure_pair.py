"""Two-element measurements:  blender -b --factory-startup --python tests/test_measure_pair.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import measure as ms, measureinfo as mi, measurepair as mp, store, subsel

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def only(*objs):
    for o in ctx.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0] if objs else None
    ctx.view_layer.update()

def world(o, q):
    M = np.array(o.matrix_world, float)
    return M[:3, :3] @ np.asarray(q, float) + M[:3, 3]

def fnear(o, q):
    """Face of o nearest the local point q."""
    fs = store.get_brep(o).faces()
    return min(range(len(fs)), key=lambda k: fs[k].closest_point(tuple(map(float, q)))[1])

def enear(o, q):
    es = store.get_brep(o).edges()
    return min(range(len(es)), key=lambda k: es[k].closest_point(tuple(map(float, q)))[2])

def F(o, q):
    return mp.face_el(o, fnear(o, q))

def E(o, q):
    return mp.edge_el(o, enear(o, q))

def R(a, b):
    return mp.relate(a, b)

def near(x, y, tol=1e-9):
    return abs(x - y) <= tol * max(1.0, abs(y))

# --- a box: faces and edges ---------------------------------------------------------------
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=1.5, height=1); B = ctx.active_object
bb = np.array([list(v) for v in B.bound_box]); lo, hi = bb.min(0), bb.max(0)
top, bot = F(B, [0, 0, hi[2]]), F(B, [0, 0, lo[2]])
front, side = F(B, [0, lo[1], 0.5 * (lo[2] + hi[2])]), F(B, [hi[0], 0, 0.5 * (lo[2] + hi[2])])
check("box faces read as planes", all(e["type"] == "plane" for e in (top, bot, front, side)),
      [e["type"] for e in (top, bot, front, side)])
r = R(top, bot)
check("top / bottom: parallel faces 1 apart, no angle, shortest not repeated",
      r["parallel"] and near(r["parallel"][0], 1.0) and r["angle"] is None
      and [a for a, _ in mp.rows(ctx, r)] == ["Parallel faces"], mp.text(ctx, r))
r = R(top, front)
check("top / front: 90° between the faces, touching", r["angle"] and near(r["angle"][0], 90.0) and r["angle"][1] == "faces"
      and r["touching"], mp.text(ctx, r))
e1, e2 = E(B, [0, lo[1], hi[2]]), E(B, [hi[0], 0, hi[2]])
check("box edges read as lines", e1["type"] == "line" and e2["type"] == "line")
r = R(e1, e2)
check("two edges at a corner: 90° (corner)", r["angle"] and near(r["angle"][0], 90.0) and r["angle"][1] == "corner",
      mp.text(ctx, r))
e3 = E(B, [0, hi[1], lo[2]])
r = R(e1, e3)
check("opposite edges: parallel lines, the diagonal apart", r["parallel"] and near(r["parallel"][0], math.hypot(1.5, 1.0))
      and r["parallel"][1] == "lines", mp.text(ctx, r))
r = R(e1, bot)
check("an edge and a face it is parallel to: line to face 1", r["parallel"] and near(r["parallel"][0], 1.0)
      and r["parallel"][1] == "line to face", mp.text(ctx, r))

# --- the box moved, turned and scaled non-uniformly: world values -------------------------------
B.location = (3, -2, 1); B.rotation_euler = (0.4, 0.2, 0.9); ctx.view_layer.update()
r = R(F(B, [0, 0, hi[2]]), F(B, [0, 0, lo[2]]))
check("turned and moved: still 1 apart", near(r["parallel"][0], 1.0, 1e-6), mp.text(ctx, r))
B.scale = (1, 1, 3); ctx.view_layer.update()
r = R(F(B, [0, 0, hi[2]]), F(B, [0, 0, lo[2]]))
check("scaled x3 along its height: 3 apart (a world copy)", near(r["parallel"][0], 3.0, 1e-6), mp.text(ctx, r))
check("the Brep itself is untouched", np.allclose(np.array([list(v) for v in B.bound_box]), bb))
B.location = (0, 0, 0); B.rotation_euler = (0, 0, 0); B.scale = (1, 1, 1); ctx.view_layer.update()

# --- a 45° chamfer: 135° to the faces beside it ---------------------------------------------------
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[B.name, f"E{enear(B, [0, lo[1], hi[2]])}"]]), shape="CHAMFER", radius=0.2)
B = ctx.active_object
ch = F(B, [0, lo[1] + 0.1, hi[2] - 0.1])
r = R(ch, F(B, [0, 0, hi[2]]))
check("45° chamfer and the top: 135° between the faces", ch["type"] == "plane" and r["angle"]
      and near(r["angle"][0], 135.0, 1e-6) and r["angle"][1] == "faces", mp.text(ctx, r))
r = R(ch, F(B, [0, 0, lo[2]]))
check("chamfer and the bottom (apart): the acute angle 45°", near(r["angle"][0], 45.0, 1e-6) and not r["touching"],
      mp.text(ctx, r))

# --- cylinders and circles: axes, centres -------------------------------------------------------
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.5, height=2.0, caps=True); C1 = ctx.active_object
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.3, height=2.0, caps=True); C2 = ctx.active_object
C1.location = (10, 0, 0); C2.location = (13, 4, 0); ctx.view_layer.update()
bb1 = np.array([list(v) for v in C1.bound_box]); z0, z1 = bb1[:, 2].min(), bb1[:, 2].max()
c1, c2 = F(C1, [0.5, 0, 0.5 * (z0 + z1)]), F(C2, [0.3, 0, 0.5 * (z0 + z1)])
check("cylinder faces: radius and the axis", c1["type"] == "cylinder" and near(c1["r"], 0.5, 1e-9)
      and abs(abs(c1["n"][2]) - 1) < 1e-9 and np.hypot(c1["C"][0] - 10, c1["C"][1]) < 1e-9, c1["type"])
r = R(c1, c2)
check("two holes: axes 5 apart, shortest 4.2", r["parallel"] and near(r["parallel"][0], 5.0) and r["parallel"][1] == "axes"
      and near(r["min"][2], 4.2, 1e-7), mp.text(ctx, r))
rim1, rim2 = E(C1, [0.5, 0, z1]), E(C2, [0.3, 0, z1])
check("rims read as circles", rim1["type"] == "circle" and near(rim1["r"], 0.5, 1e-7))
r = R(rim1, rim2)
check("two rims: centres 5 apart (axes not repeated)", r["centres"] and near(r["centres"][2], 5.0, 1e-7)
      and [a for a, _ in mp.rows(ctx, r)][0] == "Centres" and "Axes" not in mp.text(ctx, r), mp.text(ctx, r))
cap = F(C1, [0.1, 0.1, z0])
r = R(rim1, cap)
check("a rim and the other cap: the centre 2 from it, no angle (parallel)", r["centres"] and near(r["centres"][2], 2.0, 1e-7)
      and r["angle"] is None, mp.text(ctx, r))
r = R(c1, side)
check("a hole's axis and a face it runs along: axis to face", r["parallel"] and r["parallel"][1] == "axis to face"
      and near(r["parallel"][0], 10 - hi[0], 1e-7), mp.text(ctx, r))
r = R(c1, F(B, [0, 0, lo[2]]))
check("a hole's axis square to a face: 90°", r["angle"] and near(r["angle"][0], 90.0, 1e-6), mp.text(ctx, r))

# --- sphere, points ---------------------------------------------------------------------------------
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0); S = ctx.active_object
S.location = (0, 10, 0); ctx.view_layer.update()
s = F(S, [0.3, 0.3, 0.9])
r = R(s, mp.point_el([0, 10, 5]))
check("sphere: from the centre 5, shortest 4", s["type"] == "sphere" and near(r["centres"][2], 5.0, 1e-7)
      and near(r["min"][2], 4.0, 1e-6), mp.text(ctx, r))
r = R(mp.point_el([1, 2, 3]), mp.point_el([4, 6, 3]))
check("two points: 5, ΔX 3 ΔY 4", near(r["min"][2], 5.0) and list(r["delta"]) == [3, 4, 0]
      and "ΔX 3" in mp.text(ctx, r), mp.text(ctx, r))
check("X alone: ΔX 3", mp.rows(ctx, r, "X") == [("ΔX", mi.fmt(ctx, 3.0))], mp.rows(ctx, r, "X"))
r = R(mp.point_el([0, 0, hi[2] + 2]), top)
check("a point above a face: 2", near(r["min"][2], 2.0, 1e-9), mp.text(ctx, r))

# --- curves: skew lines exact, a free curve refined --------------------------------------------------
def poly(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for k, p in enumerate(pts):
        sp.points[k].co = (*p, 1)
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    return o
L1 = poly("L1", [(-3, 0, 20), (3, 0, 20)]); L2 = poly("L2", [(0, -3, 22), (0, 3, 22)])
ctx.view_layer.update()
l1, l2 = mp.spline_el(L1, "S0"), mp.spline_el(L2, "S0")
r = R(l1, l2)
check("skew lines: 2 apart, 90°", l1["type"] == "line" and near(r["min"][2], 2.0) and near(r["angle"][0], 90.0)
      and r["angle"][1] == "lines", mp.text(ctx, r))
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0, elliptic=True, radius2=0.6); EL = ctx.active_object
EL.location = (0, 0, 30); ctx.view_layer.update()
L3 = poly("L3", [(-3, 2, 30), (3, 2, 30)]); ctx.view_layer.update()
el = mp.spline_el(EL, "S0")
r = R(el, mp.spline_el(L3, "S0"))
# (to Blender's single-precision curve CVs: radius2 0.6 is 0.60000002384)
check("ellipse and a line beside it: 1.4 (refined)", el["type"] == "curve" and near(r["min"][2], 1.4, 1e-7),
      f"{r['min'][2]:.12f}")
r = R(s, F(B, [0, 0, lo[2]]))
check("sphere and a face's plane: the centre 0.5 from it", near(r["centres"][2], 0.0 - lo[2], 1e-7), mp.text(ctx, r))

# --- the selection readout: two picks, two bodies, two curves -------------------------------------
B2 = B
only(B2)
subsel.set_picks(B2, "FACE", [fnear(B2, [0, 0, hi[2]]), fnear(B2, [0, 0, lo[2]])])
info = mi.gather(ctx)
labs = [a for a, _ in mi.rows(ctx, info)]
check("readout, two faces picked: their areas and parallel faces 1", "Parallel faces" in labs and "2 faces" in labs
      and near(info["pair"]["parallel"][0], 1.0), mi.text(ctx, info))
subsel.set_picks(B2, "FACE", [])
subsel.set_picks(B2, "EDGE", [enear(B2, [hi[0], 0, hi[2]]), enear(B2, [hi[0], hi[1], 0.5 * (lo[2] + hi[2])])])
info = mi.gather(ctx)
check("readout, two edges at a corner: 90°", info["pair"] is not None and near(info["pair"]["angle"][0], 90.0, 1e-6),
      mi.text(ctx, info))
subsel.set_picks(B2, "EDGE", [enear(B2, [hi[0], 0, hi[2]])])
info = mi.gather(ctx)
check("one edge: no pair", info["pair"] is None, mi.text(ctx, info))
subsel.set_picks(B2, "EDGE", [])
only(C1, C2)
info = mi.gather(ctx)
g = info["clearance"]
check("two bodies: the clearance 4.2", g is not None and near(g[0], 4.2, 1e-4), f"{g} | {mi.text(ctx, info)}")
only(L1, L2)
info = mi.gather(ctx)
check("two curves: 2 apart, 90°", info["pair"] is not None and near(info["pair"]["min"][2], 2.0), mi.text(ctx, info))
only(L1, L2, L3)
check("three curves: no pair", mi.gather(ctx)["pair"] is None)

# --- Measure's labels -------------------------------------------------------------------------------
check("Measure: a point reads X Y Z", ms.label(ctx, {"kind": "point", "point": np.array([1.0, 2.0, 3.0])}).startswith("X 1"))
pr = {"kind": "pair", "r": R(l1, l2)}
check("Measure: a pair's label and drawing (the shortest line)", "Angle 90°" in ms.label(ctx, pr) and len(ms.geometry(pr)[0]) == 1)
pr = {"kind": "pair", "r": R(rim1, rim2)}
check("Measure: centres drawn too", len(ms.geometry(pr)[0]) == 2)
check("outline of a face: its boundary", len(mp.outline(top)) >= 4 and len(mp.outline(rim1)) == 1)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
