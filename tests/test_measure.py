"""Measure:  blender -b --factory-startup --python tests/test_measure.py"""
import math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, linesolve, measure as ms, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sm = store._k()[0]

def curve(o, ref="S0"):
    return linesolve.kernel_curve(features.input_curve(o, ref), sm)

# --- distance -------------------------------------------------------------------
m = ms.distance((0, 0, 0), (3, 4, 12))
check("distance and its deltas", abs(m["length"] - 13) < 1e-12 and list(m["delta"]) == [3, 4, 12])
txt = ms.label(ctx, m)
check("distance label: length and ΔX ΔY ΔZ", txt.startswith("13") and "ΔX 3" in txt and "ΔZ 12" in txt, txt)

# --- circles: exact radius and centre -------------------------------------------------
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.7); C = ctx.active_object
C.location = (1, 2, 0.5); ctx.view_layer.update()
r = ms.curve_radius(curve(C), np.array([1.7, 2.3, 0.5]))
# (to Blender's single-precision curve CVs)
check("exact circle: 'circle', radius and centre exact", r["source"] == "circle" and abs(r["radius"] - 0.7) < 1e-6
      and np.linalg.norm(r["centre"] - [1, 2, 0.5]) < 1e-6 and abs(np.linalg.norm(r["point"] - r["centre"]) - r["radius"]) < 1e-12,
      f"{r['source']} {r['radius']:.10f}")
check("diameter label doubles it", ms.label(ctx, r, diameter=True).startswith("Ø 1.4"), ms.label(ctx, r, True))
bpy.ops.nproto.add_primitive(kind="ARC", radius=0.4, angle=120.0); A = ctx.active_object
r = ms.curve_radius(curve(A), np.array([0.4, 0.05, 0]))
check("exact arc: 'circle', radius exact", r["source"] == "circle" and abs(r["radius"] - 0.4) < 1e-6 and not r["full"],
      f"{r['source']} {r['radius']:.10f}")

# --- curves: radius of curvature -------------------------------------------------------
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.7, degree="3", spans_u=8); S = ctx.active_object
cs = curve(S)
rr = [ms.curve_radius(cs, 0.7 * np.array([math.cos(a), math.sin(a), 0]))["radius"] for a in np.linspace(0, 2 * math.pi, 33)]
# a degree-3 circle is within 0.1 % in position, but its curvature varies round it: Measure shows that
check("smooth circle: 'curvature', varying a few % round 0.7", ms.curve_radius(cs, np.array([0.7, 0, 0]))["source"] == "curvature"
      and 0.6 < min(rr) < max(rr) < 0.8 and max(rr) - min(rr) > 1e-3, f"{min(rr):.4f}..{max(rr):.4f}")
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0, elliptic=True, radius2=0.6); E = ctx.active_object
ce = curve(E)
src = ms.curve_radius(ce, np.array([1.0, 0, 0]))
r1 = ms.curve_radius(ce, np.array([1.0, 0.0, 0]))
r2 = ms.curve_radius(ce, np.array([0.0, 0.6, 0]))
check("ellipse: radius of curvature b²/a at the major vertex, a²/b at the minor",
      r1["source"] == "curvature" and abs(r1["radius"] - 0.36) < 1e-6 and abs(r2["radius"] - 1.0 / 0.6) < 1e-6,
      f"{r1['radius']:.6f} / {r2['radius']:.6f}")
check("ellipse: the centre of curvature on the inside", np.linalg.norm(r1["centre"] - [1 - 0.36, 0, 0]) < 1e-6)
bpy.ops.nproto.add_primitive(kind="LINE", length=2.0); L = ctx.active_object
r = ms.curve_radius(curve(L), np.array([0.3, 0, 0]))
check("line: straight", r["source"] == "straight" and math.isinf(r["radius"]) and ms.label(ctx, r) == "straight")

# --- surfaces: principal radii ------------------------------------------------------------
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.5, height=2.0); CY = ctx.active_object
s = ms.surface_radius(CY, np.array([0.5, 0, 0.3]))
(ra, da, ca), (rb, db, cb) = s["principal"]
check("cylinder face: radius 0.5 round it, flat along it", abs(ra - 0.5) < 1e-9 and math.isinf(rb)
      and abs(abs(db[2]) - 1) < 1e-6 and abs(da[2]) < 1e-6, f"{ra:.10f} / {rb}")
check("cylinder: centre of curvature on the axis", ca is not None and np.hypot(ca[0], ca[1]) < 1e-9)
check("cylinder label", ms.label(ctx, s) == f"R {ms._fmt(ctx, ra)}  ·  flat  (principal)", ms.label(ctx, s))
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.3); SP = ctx.active_object
SP.location = (5, 0, 0); ctx.view_layer.update()
s = ms.surface_radius(SP, np.array([5 + 1.3 * math.cos(0.4), 1.3 * math.sin(0.4) * 0.6, 1.3 * math.sin(0.4) * 0.8]))
(ra, _, ca), (rb, _, _) = s["principal"]
check("sphere: the same radius both ways", abs(ra - 1.3) < 1e-6 and abs(rb - 1.3) < 1e-6 and "both ways" in ms.label(ctx, s)
      and np.linalg.norm(ca - [5, 0, 0]) < 1e-6, f"{ra:.8f} {rb:.8f}")
bpy.ops.nproto.add_primitive(kind="PLANE"); PL = ctx.active_object
PL.location = (0, 5, 0); ctx.view_layer.update()
s = ms.surface_radius(PL, np.array([0.2, 5.1, 0]))
check("plane: flat both ways", all(math.isinf(r_) for r_, _, _ in s["principal"]), ms.label(ctx, s))

# --- drawing geometry -------------------------------------------------------------------------
lines, pts, at = ms.geometry(ms.curve_radius(curve(C), np.array([1.7, 2, 0.5])))
check("radius drawing: centre line and the circle", len(lines) == 2 and len(pts) == 2)
check("the operator is registered", hasattr(bpy.ops.nproto, "measure"))

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
