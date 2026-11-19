"""Circle type and Draw Cylinder:  blender -b --factory-startup --python tests/test_drawcylinder.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import drawshapes as ds, features, store, surfeval
from blender_nurbs.feature_ui import primitive_params

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

T = ds.frame(np.array([1.0, 2.0, 0.5]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

# --- Circle: Exact and Smooth ------------------------------------------------------
r = bpy.ops.nproto.draw_circle(data=json.dumps({"matrix": T.tolist()}), radius=0.7, ctype="EXACT")
c = ctx.active_object
sp = c.data.splines[0]
W = [p.co[3] for p in sp.points]
P = np.array([(c.matrix_world @ p.co.to_3d()) for p in sp.points])
check("exact circle: rational, order 3", r == {'FINISHED'} and sp.order_u == 3 and min(W) < 0.9)
from blender_nurbs import geom
S = geom.sample_curve(features.input_curve(c, "S0"), 360)
rad = np.linalg.norm(S[:, :2] - [1, 2], axis=1)
check("exact circle: radius exact", abs(rad - 0.7).max() < 1e-6 and abs(S[:, 2] - 0.5).max() < 1e-9, f"{abs(rad - 0.7).max():.1e}")
check("exact circle: history degree 0", features.history(c)["params"]["degree"] == 0)

r = bpy.ops.nproto.draw_circle(data=json.dumps({"matrix": T.tolist()}), radius=0.7, ctype="SMOOTH", spans_u=8)
c = ctx.active_object
sp = c.data.splines[0]
W = [p.co[3] for p in sp.points]
S = geom.sample_curve(features.input_curve(c, "S0"), 720)
rad = np.linalg.norm(S[:, :2] - [1, 2], axis=1)
h = features.history(c)["params"]
check("smooth circle: degree 3, not rational, 8 CVs", r == {'FINISHED'} and sp.order_u == 4 and max(W) == min(W) == 1.0
      and len(sp.points) == 8 and h["degree"] == 3 and h["spans_u"] == 8, f"order {sp.order_u}, {len(sp.points)} CVs")
check("smooth circle: close to the radius", abs(rad - 0.7).max() < 0.01 * 0.7, f"{abs(rad - 0.7).max() / 0.7:.2%}")
r = bpy.ops.nproto.draw_circle(data=json.dumps({"matrix": T.tolist()}), radius=0.7, ctype="SMOOTH", spans_u=12)
check("smooth circle: spans", len(ctx.active_object.data.splines[0].points) == 12)

# --- Cylinder: what the tool places (its execute: matrix + parameters) -------------
def cyl(ctype, caps, spans=8, r_=0.5, h_=1.2):
    Tc = ds.frame(np.array([0.0, 0.0, 0.0]) + np.array([0, 0, 1.0]) * (0.5 * h_), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    params = primitive_params("prim_cylinder", radius=r_, height=h_, spans_v=1, caps=caps, **ds.circle_degree(ctype, spans))
    data = {"matrix": Tc.tolist(), "op": "prim_cylinder", "params": params, "name": "NURBS Cylinder"}
    out = bpy.ops.nproto.draw_cylinder(data=json.dumps(data), radius=r_, height=h_, ctype=ctype, caps=caps, spans_u=spans)
    return out, ctx.active_object

def side_face(obj):
    for f in store.get_brep(obj).faces():
        E = surfeval.Surf(f.surface())
        P = E.grid(np.linspace(*E.u_domain, 9), np.linspace(*E.v_domain, 9), 0)[0, 0].reshape(-1, 3)
        if np.ptp(P[:, 2]) > 1e-6:
            return E, P
    return None, None

r, o = cyl("EXACT", True)
faces = store.get_brep(o).faces()
E, P = side_face(o)
M = np.array(o.matrix_world, float)
Pw = P @ M[:3, :3].T + M[:3, 3]
check("exact solid cylinder: 3 faces (side, caps)", r == {'FINISHED'} and len(faces) == 3, f"{len(faces)} faces")
check("exact cylinder: radius exact, base on the clicked plane, height", abs(np.hypot(Pw[:, 0], Pw[:, 1]) - 0.5).max() < 1e-6
      and abs(Pw[:, 2].min()) < 1e-6 and abs(Pw[:, 2].max() - 1.2) < 1e-6, f"r err {abs(np.hypot(Pw[:, 0], Pw[:, 1]) - 0.5).max():.1e} z {Pw[:, 2].min():.4f}..{Pw[:, 2].max():.4f}")
check("exact cylinder: rational", E.rational)
h = features.history(o)
check("cylinder history: prim_cylinder, editable", h["op"] == "prim_cylinder" and h["params"]["caps"] is True
      and h["params"]["degree"] == 0)

r, o = cyl("SMOOTH", False, spans=10)
faces = store.get_brep(o).faces()
E, P = side_face(o)
Pw = P @ np.array(o.matrix_world, float)[:3, :3].T + np.array(o.matrix_world, float)[:3, 3]
check("smooth tube: one face, degree 3 round, not rational", r == {'FINISHED'} and len(faces) == 1 and not E.rational
      and 3 in (E.p, E.q), f"{len(faces)} faces, degree ({E.p}, {E.q}), rational {E.rational}")
check("smooth tube: close to the radius", abs(np.hypot(Pw[:, 0], Pw[:, 1]) - 0.5).max() < 0.01 * 0.5)
check("smooth: spans round it", max(E.nu, E.nv) == 10 + 3 or max(E.nu, E.nv) >= 10, f"CVs {E.nu} x {E.nv}")

# the tool's preview outline and the type helpers
pl = ds.cylinder_outline(ds.frame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])), 0.5, 2.0)
check("preview: two rims and four rulings", len(pl) == 6 and abs(np.array(pl[1])[:, 2] - 1.0).max() < 1e-12)
check("type parameters", ds.circle_degree("EXACT", 8) == {"degree": 0} and ds.circle_degree("SMOOTH", 6) == {"degree": 3, "spans_u": 6})

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
