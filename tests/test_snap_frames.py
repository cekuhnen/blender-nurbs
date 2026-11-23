"""Snap elements with a frame (Align):  blender -b --factory-startup --python tests/test_snap_frames.py"""
import math, os, sys, time
import bpy
import numpy as np
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import snapframes as sf, snapping as sn, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    sn.forget_meshes()

def view(kind="top", s=2.0, W=1000):
    """Orthographic view, camera 10 away; 1 unit = W/(2s) px. top: looking down -Z,
    front: along +Y (screen x = X, y = Z)."""
    VM = np.eye(4)
    if kind == "front":
        VM[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    VM[2, 3] = -10.0
    PM = np.diag([1 / s, 1 / s, -0.01, 1.0]) @ VM
    return sn.View(PM, VM, W, W)
px = lambda x, y, s=2.0, W=1000: np.array([W / 2 + W / 2 * x / s, W / 2 + W / 2 * y / s])
top, front = view(), view("front")
Z, Y, X = np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
par = lambda a, b: abs(abs(float(np.dot(a, b))) - 1.0) < 1e-6
def el_at(T, v, xy):
    return sf.element(T, v, xy)
def desc(e):
    return None if e is None else (e["type"], e["label"], np.round(e["o"], 5).tolist(),
                                   None if e["d"] is None else np.round(e["d"], 5).tolist(), e["r"], e["obj"])

# -- NURBS cylinder: rim ring, centre ring, side axis, cap plane -----------------
clear()
ctx.scene.cursor.location = (0, 0, 0)
bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.5, height=1.2, caps=True)
CY = ctx.active_object
T = sn.collect([CY])
zt = max(f.centroid()[2] for f in store.get_brep(CY).faces())
check("rings: both rims of the NURBS cylinder are rings", len(T.rings) == 2 and all(abs(g["r"] - 0.5) < 1e-6 for g in T.rings),
      f"{[(g['r'], g['ref']) for g in T.rings]}")
a = math.radians(40)
e = el_at(T, top, px(0.5 * math.cos(a), 0.5 * math.sin(a)) + (3, 2))
check("NURBS rim: a Ring, centre on the axis at the top, axis +Z (towards the viewer), r 0.5",
      e is not None and e["type"] == "RING" and e["pin"] and np.allclose(e["o"], (0, 0, zt), atol=1e-7)
      and np.allclose(e["d"], Z, atol=1e-9) and abs(e["r"] - 0.5) < 1e-7 and e["obj"] == CY.name, desc(e))
check("NURBS rim: glyph is a full circle and the axis arrow", e is not None and len(e["glyph"]) == 2
      and e["glyph"][0][1] == "main" and e["glyph"][1][1] == "axis", f"{[g[1] for g in e['glyph']] if e else None}")
e = el_at(T, top, px(0, 0) + (3, -2))
check("NURBS cap centre: the top Ring (nearest the viewer)", e is not None and e["type"] == "RING"
      and np.allclose(e["o"], (0, 0, zt), atol=1e-7), desc(e))
e = el_at(T, top, px(0.22, 0.13))
check("NURBS cap away from rim and centre: a Face, normal +Z, not pinned", e is not None and e["type"] == "PLANE"
      and not e["pin"] and np.allclose(e["d"], Z, atol=1e-9) and abs(e["o"][2] - zt) < 1e-6, desc(e))
e = el_at(T, front, px(0.18, 0.2))
check("NURBS side (front view): an Axis along Z through the origin, r 0.5", e is not None and e["type"] == "AXIS"
      and par(e["d"], Z) and np.linalg.norm(e["o"][:2]) < 1e-7 and abs(e["r"] - 0.5) < 1e-6
      and any(g[1] == "dash" for g in e["glyph"]), desc(e))
t0 = time.perf_counter()
for k in range(50):
    sf.element(T, front, px(0.18 + 0.001 * k, 0.2))
dt = (time.perf_counter() - t0) / 50
check("NURBS side: an element costs under 5 ms once cached", dt < 5e-3, f"{dt * 1e3:.2f} ms")

# -- turned object: rings and axes in world space --------------------------------
CY.matrix_world = Matrix.Rotation(math.radians(90), 4, 'X') @ Matrix.Translation((0, 0, 0))
bpy.context.view_layer.update()
T = sn.collect([CY])
check("turned cylinder (90 about X): ring axes along Y", len(T.rings) == 2 and all(par(g["n"], Y) for g in T.rings),
      f"{[np.round(g['n'], 4).tolist() for g in T.rings]}")
e = el_at(T, top, px(0.1, 0.25))
check("turned cylinder, side from the top: Axis along Y", e is not None and e["type"] == "AXIS" and par(e["d"], Y)
      and abs(e["o"][0]) < 1e-7 and abs(e["o"][2]) < 1e-7, desc(e))

# -- NURBS box: edge line, vertex point, edge middle with its direction, face centre
clear()
bpy.ops.nproto.add_primitive(kind='BOX', width=2.0, depth=1.0, height=1.0)
BX = ctx.active_object
T = sn.collect([BX])
lo, hi = np.array([-1.0, -0.5, -0.5]), np.array([1.0, 0.5, 0.5])       # (face bounding boxes are padded)
_bb = np.array(BX.bound_box, float)
check("NURBS box: 2 x 1 x 1 round the origin", np.allclose(_bb.min(axis=0), lo, atol=1e-5)
      and np.allclose(_bb.max(axis=0), hi, atol=1e-5), f"{_bb.min(axis=0)} {_bb.max(axis=0)}")
xm, ym = hi[0], 0.5 * (lo[1] + hi[1])
e = el_at(T, top, px(xm - 0.3, hi[1]) + (0, 3))
check("NURBS box top edge (top view): a Line along X, not pinned", e is not None and e["type"] == "LINE"
      and par(e["d"], X) and not e["pin"] and e["src"] is not None, desc(e))
e = el_at(T, top, px(hi[0], hi[1]) + (2, 2))
check("NURBS box corner: a Vertex point", e is not None and e["type"] == "POINT" and e["label"] == "End"
      and np.allclose(e["o"][:2], hi[:2], atol=1e-6), desc(e))
e = el_at(T, top, px(0.5 * (lo[0] + hi[0]), hi[1]) + (1, 2))
check("NURBS box edge middle: a Mid point with the edge direction as its hint", e is not None and e["type"] == "POINT"
      and e["label"] == "Mid" and e["hint"] is not None and par(e["hint"], X), desc(e))
c = 0.5 * (lo + hi)
e = el_at(T, top, px(c[0], c[1]) + (2, 2))
check("NURBS box top centre: a pinned Center plane, normal +Z, hint along an edge", e is not None
      and e["type"] == "PLANE" and e["pin"] and e["label"] == "Center" and np.allclose(e["d"], Z, atol=1e-9)
      and np.allclose(e["o"], (c[0], c[1], hi[2]), atol=1e-6) and e["hint"] is not None
      and (par(e["hint"], X) or par(e["hint"], Y)), desc(e))
e = el_at(T, front, px(c[0] + 0.37, c[2] + 0.21))
check("NURBS box front face: a Face, normal -Y (towards the viewer)", e is not None and e["type"] == "PLANE"
      and np.allclose(e["d"], -Y, atol=1e-9), desc(e))

# -- NURBS sphere: its centre; cone: its axis -------------------------------------
clear()
bpy.ops.nproto.add_primitive(kind='SPHERE', radius=0.8)
SP = ctx.active_object
T = sn.collect([SP])
e = el_at(T, top, px(0.21, 0.33))
check("NURBS sphere: a Sphere point at its centre", e is not None and e["type"] == "POINT" and e["label"] == "Sphere"
      and np.linalg.norm(e["o"] - np.array(SP.matrix_world.translation)) < 1e-6, desc(e))
clear()
bpy.ops.nproto.add_primitive(kind='CONE', radius=0.6, height=1.0)
CO = ctx.active_object
T = sn.collect([CO])
e = el_at(T, front, px(0.12, 0.3))
check("NURBS cone side: an Axis along Z (no radius)", e is not None and e["type"] == "AXIS" and par(e["d"], Z)
      and np.linalg.norm(e["o"][:2]) < 1e-6 and e["r"] is None, desc(e))

# -- plain meshes ------------------------------------------------------------------
clear()
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1.0, depth=2.0, location=(3, 0, 0))
MC = ctx.active_object
T = sn.collect([MC])
a = math.radians(50)
e = el_at(T, top, px(3 + math.cos(a), math.sin(a)) + (2, 1), )
check("mesh cylinder rim: a Ring (beats the vertices and edge middles on it), centre (3,0,1), r 1",
      e is not None and e["type"] == "RING" and np.allclose(e["o"], (3, 0, 1), atol=1e-5)
      and par(e["d"], Z) and abs(e["r"] - 1) < 1e-5 and e["obj"] == MC.name, desc(e))
e = el_at(T, top, px(3, 0) + (2, 2))
check("mesh cylinder cap centre: the top Ring", e is not None and e["type"] == "RING"
      and np.allclose(e["o"], (3, 0, 1), atol=1e-5), desc(e))
e = el_at(T, top, px(3.35, 0.42))
check("mesh cylinder cap: a Face, normal +Z", e is not None and e["type"] == "PLANE" and np.allclose(e["d"], Z, atol=1e-6),
      desc(e))
e = el_at(T, front, px(3.27, 0.31))
check("mesh cylinder side: an Axis along Z through (3,0), r 1", e is not None and e["type"] == "AXIS"
      and par(e["d"], Z) and np.allclose(e["o"][:2], (3, 0), atol=1e-5) and abs(e["r"] - 1) < 1e-4, desc(e))

clear()
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
MB = ctx.active_object
T = sn.collect([MB])
e = el_at(T, top, px(0.3, 1) + (0, 3))
check("mesh cube edge: a Line along X", e is not None and e["type"] == "LINE" and par(e["d"], X)
      and e["obj"] == MB.name, desc(e))
e = el_at(T, top, px(1, 1) + (2, 2))
check("mesh cube corner: a Vertex", e is not None and e["type"] == "POINT" and e["label"] == "Vertex"
      and np.allclose(e["o"], (1, 1, 1), atol=1e-6), desc(e))
e = el_at(T, top, px(0, 1) + (2, 2))
check("mesh cube edge middle: Mid, hint along X", e is not None and e["label"] == "Mid" and e["hint"] is not None
      and par(e["hint"], X), desc(e))
e = el_at(T, top, px(0, 0) + (2, 2))
check("mesh cube top centre: a pinned Center plane", e is not None and e["type"] == "PLANE" and e["pin"]
      and np.allclose(e["o"], (0, 0, 1), atol=1e-6) and np.allclose(e["d"], Z, atol=1e-6), desc(e))
e = el_at(T, top, px(0.41, -0.33))
check("mesh cube top: a Face", e is not None and e["type"] == "PLANE" and not e["pin"], desc(e))

clear()
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.2, location=(0, 0, 0.5))
MS = ctx.active_object
T = sn.collect([MS])
e = el_at(T, top, px(0.37, 0.29))
check("mesh UV sphere: a Sphere point at (0,0,0.5)", e is not None and e["label"] == "Sphere"
      and np.allclose(e["o"], (0, 0, 0.5), atol=1e-4), desc(e))

# -- sketch snapping unchanged: pick still returns its 4-tuple ----------------------
clear()
bpy.ops.mesh.primitive_cube_add(size=2.0)
T = sn.collect([ctx.active_object])
h = T.pick(top, px(1, 1) + (2, 2))
check("pick: still (point, kind, tangent, src)", h is not None and len(h) == 4 and h[1] == "vertex", f"{h}")

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
