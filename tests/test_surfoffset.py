"""Offset on Surface:  blender -b --factory-startup --python tests/test_surfoffset.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, geom, history, store, surfoffset as so

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context

def clear():
    """Each case in a scene of its own (the surface under a curve is found among the visible ones)."""
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
clear()

def poly(name, pts, cyclic=False, loc=(0, 0, 0)):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1); sp.use_cyclic_u = cyclic
    for q, c in zip(sp.points, pts): q.co = (*c, 1.0)
    o = bpy.data.objects.new(name, cu); o.location = loc; ctx.collection.objects.link(o); return o

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

def offset(inputs, select=(), **kw):
    """Run the operator (EXEC); the result's sample points (world) per spline."""
    bpy.ops.object.select_all(action='DESELECT')
    for o in select:
        o.select_set(True)
    t = time.time()
    try:
        r = bpy.ops.nproto.offset_on_surface(inputs=ins(*inputs), **kw)
    except RuntimeError as exc:           # an ERROR report (refused)
        print("   refused:", str(exc).strip())
        r = {'CANCELLED'}
    dt = time.time() - t
    if r != {'FINISHED'}:
        return r, [], dt
    out = ctx.active_object
    pts = [geom.sample_curve(features.input_curve(out, f"S{i}"), 400) for i in range(len(out.data.splines))]
    return r, pts, dt

def on_surface(obj, P):
    tg = so.target_of([obj])
    return float(tg.feet_all(np.asarray(P))[4].max())

# --- sphere: latitude circle -> latitude (d / R further), any distance -----
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0); SPH = ctx.active_object
lat = math.radians(30)
bpy.ops.curve.primitive_nurbs_circle_add(radius=math.cos(lat), location=(0, 0, math.sin(lat))); LAT = ctx.active_object
for d in (0.2, -0.3, 0.8):
    r, pts, dt = offset([(LAT, "S0")], distance=d)        # the sphere is found under the curve
    P = np.vstack(pts) if pts else np.zeros((1, 3))
    lat_new = np.arcsin(np.clip(P[:, 2] / np.linalg.norm(P, axis=1), -1, 1))
    err = float(np.abs(lat_new - (lat - d)).max())       # outward from a northern circle: south
    check(f"sphere latitude d={d}", r == {'FINISHED'} and len(pts) == 1 and err < 2e-4
          and abs(np.linalg.norm(P, axis=1) - 1).max() < 2e-4, f"err {err:.2e} {dt:.2f}s")
r, pts, _ = offset([(LAT, "S0")], distance=0.2, both_sides=True)
check("sphere both sides", r == {'FINISHED'} and len(pts) == 2)
out = ctx.active_object
check("history recorded", features.history(out) is not None and features.history(out)["op"] == "offset_on_surface",
      str((features.history(out) or {}).get("op")))

# --- plane, square: corners --------------------------------------------------
clear()
bpy.ops.nproto.add_primitive(kind="PLANE", width=4, depth=4); PL = ctx.active_object
SQ = poly("Sq", [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)], cyclic=True)

def sq_dist(P, h=0.5):
    dx = np.maximum(np.abs(P[:, 0]) - h, 0); dy = np.maximum(np.abs(P[:, 1]) - h, 0)
    return np.hypot(dx, dy)

r, pts, _ = offset([(SQ, "S0")], select=[PL], distance=0.2, os_corner="ROUND")
P = np.vstack(pts)
check("square round: 0.2 from the square", r == {'FINISHED'} and abs(sq_dist(P) - 0.2).max() < 2e-4)
r, pts, _ = offset([(SQ, "S0")], select=[PL], distance=0.2, os_corner="SHARP")
P = np.vstack(pts); m = np.maximum(np.abs(P[:, 0]), np.abs(P[:, 1]))
check("square sharp: a 1.4 square", abs(m - 0.7).max() < 1e-5 and np.abs(P[:, :2]).max() > 0.6999)
r, pts, _ = offset([(SQ, "S0")], select=[PL], distance=0.2, os_corner="CHAMFER")
P = np.vstack(pts)
check("square chamfer: cut corners", np.abs(P[:, :2]).max() < 0.7 + 1e-5 and sq_dist(P).min() > 0.14)
r, pts, _ = offset([(SQ, "S0")], select=[PL], distance=0.2, flip=True)
P = np.vstack(pts); m = np.maximum(np.abs(P[:, 0]), np.abs(P[:, 1]))
check("square inside: a 0.6 square, overlaps trimmed", abs(m - 0.3).max() < 1e-5)

# a sampled curve (a dense polyline, as Project Curve makes) runs off the plane: cut at its edge
ARC = poly("Arc", [(1.5 * math.cos(a), 1.5 * math.sin(a), 0) for a in np.linspace(-1.2, 1.2, 60)])
r, pts, _ = offset([(ARC, "S0")], select=[PL], distance=-0.8)
P = np.vstack(pts) if pts else np.zeros((1, 3))
rr = np.hypot(P[:, 0], P[:, 1])
check("polyline arc: smooth offset, cut at the plane's edge",
      r == {'FINISHED'} and len(pts) == 2 and abs(np.abs(P[:, :2]).max() - 2.0) < 1e-4 and abs(rr - 2.3).max() < 2e-4,
      f"pieces {len(pts)} r {rr.min():.5f}..{rr.max():.5f}")
LN = poly("Ln", [(1.0, -1.0, 0), (1.0, 1.0, 0)])
r, pts, _ = offset([(LN, "S0")], select=[PL], distance=-1.5)
check("all off the surface: refused", r == {'CANCELLED'})

# --- box: over the crease onto the side faces -------------------------------
clear()
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=1.5, height=1); BX = ctx.active_object
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 0, 0.5)); C2 = ctx.active_object
r, pts, _ = offset([(C2, "S0")], distance=0.5)
P = np.vstack(pts)
top = P[:, 2] > 0.5 - 1e-6
w = 0.5 - P[:, 2]
unf = np.where(top, np.hypot(P[:, 0], P[:, 1]), np.hypot(P[:, 0], np.abs(P[:, 1]) + w))   # the side unfolded
check("box: across the crease, exact unfolded", r == {'FINISHED'} and (~top).any() and abs(unf - 1).max() < 2e-4,
      f"{unf.min():.5f}..{unf.max():.5f}")

# --- cylinder: lines along it -> lines d / R round (one across the seam) ------
clear()
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0); CY = ctx.active_object
zz = np.array([CY.matrix_world @ v.co for v in CY.data.vertices])[:, 2]
z0, z1 = zz.min() + 0.2, zz.max() - 0.2
for ang in (0.0, -math.pi / 2 + 0.1):
    L = poly(f"CyLn{ang}", [(math.cos(ang), math.sin(ang), z0), (math.cos(ang), math.sin(ang), z1)])
    r, pts, _ = offset([(L, "S0")], distance=0.3, both_sides=True)
    P = np.vstack(pts) if pts else np.zeros((1, 3))
    a = np.angle(np.exp(1j * (np.arctan2(P[:, 1], P[:, 0]) - ang)))
    check(f"cylinder line at {ang:.2f} rad: +-0.3 round it", r == {'FINISHED'} and len(pts) == 2
          and abs(np.abs(a) - 0.3).max() < 2e-4 and abs(np.hypot(P[:, 0], P[:, 1]) - 1).max() < 2e-4,
          f"angles {np.abs(a).min():.5f}..{np.abs(a).max():.5f}")

# --- trimmed: a hole's edge, offset away from the hole ----------------------
clear()
def make_floor(name):
    ln = poly(name + "_ln", [(0, -2, 0), (0, 2, 0)])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=4.0, direction="X")
    o = ctx.active_object; o.name = name; return o
FL = make_floor("Floor")
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 0, 1)); CC = ctx.active_object
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.trim(target="Floor", cutters=ins((CC, "S0")), ref_point=(1, 0, 0), normal=(0, 0, 1),
                        view_dir=(0, 0, -1), action="DELETE")
check("floor with a hole", r == {'FINISHED'} and not store.is_natural(FL))
edges = store.display(FL)["edges"]
hole = [i for i, e in enumerate(edges) if np.hypot(*(np.array(e)[:, :2] - [1, 0]).T).max() < 0.51]
check("hole edge found", len(hole) == 1, str(len(edges)))
for d, want in ((0.2, 0.7), (-0.2, None)):
    r, pts, _ = offset([(FL, f"E{hole[0]}")], distance=d)
    if want is None:
        check("hole edge into the hole: refused", r == {'CANCELLED'})
        continue
    P = np.vstack(pts) if pts else np.zeros((1, 3))
    rr = np.hypot(P[:, 0] - 1, P[:, 1])
    check("hole edge offset onto the floor", r == {'FINISHED'} and abs(rr - want).max() < 2e-4, f"{rr.min():.5f}..{rr.max():.5f}")
# a curve on the floor next to the hole: the offset (a line) is cut exactly at the hole's trim
CR = poly("Cross", [(1.0 + x, -0.2, 0) for x in np.linspace(-1.5, 1.5, 2)])
r, pts, _ = offset([(CR, "S0")], select=[FL], distance=0.3)          # left of +x: +y, to y = 0.1
P = np.vstack(pts) if pts else np.zeros((1, 3))
d_hole = np.hypot(P[:, 0] - 1, P[:, 1])
check("cut at the hole's trim", r == {'FINISHED'} and len(pts) == 2 and d_hole.min() > 0.5 - 2e-4
      and abs(d_hole.min() - 0.5) < 2e-4 and abs(P[:, 1] - 0.1).max() < 1e-5, f"pieces {len(pts)} {d_hole.min():.5f}")

# --- history: the offset follows its curve and surface ------------------------
clear()
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0); SPH = ctx.active_object
bpy.ops.curve.primitive_nurbs_circle_add(radius=math.cos(lat), location=(0, 0, math.sin(lat))); LAT = ctx.active_object
r, pts, _ = offset([(LAT, "S0")], distance=0.2)
res = ctx.active_object
h = features.history(res)
check("history inputs: the curve and the surface", [r_ for _, r_ in h["inputs"]] == ["S0", "SURF"],
      str(h["inputs"]))
LAT.location.z -= 0.1           # the circle slides down the sphere (now off it a little: pulled onto it)
ctx.view_layer.update()
features.rebuild(res, ctx, force=True)
P = geom.sample_curve(features.input_curve(res, "S0"), 200)
lat2 = math.asin(math.sin(lat) - 0.1)            # (its feet: the circle's points pulled square onto the sphere)
want = math.atan2(math.sin(lat) - 0.1, math.cos(lat)) - 0.2
got = np.arcsin(np.clip(P[:, 2] / np.linalg.norm(P, axis=1), -1, 1))
check("rebuilt after the curve moved", abs(got - want).max() < 3e-4, f"{got.min():.5f} want {want:.5f}")

# --- the live preview is quick --------------------------------------------------
LAT.location.z += 0.1
ctx.view_layer.update()
tg = so.target_of([SPH])
chains = so.prepare(tg, [so.curvejoin.input_spans(LAT, "S0")], 1e-4)
t = time.time()
for d in np.linspace(0.05, 0.5, 10):
    so.preview_polylines(tg, chains, {"distance": float(d)})
pv = (time.time() - t) / 10
check("preview under 0.1 s", pv < 0.1, f"{pv * 1000:.0f} ms")
a = so.arrow(tg, chains, {"distance": 0.3}, np.array([0.0, -0.87, 0.5]))
check("arrow: on the curve, pointing outward (south), path 0.3 long", a is not None and a[1][2] < -0.3
      and abs(np.linalg.norm(np.diff(a[2], axis=0), axis=1).sum() - 0.3) < 1e-3)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
