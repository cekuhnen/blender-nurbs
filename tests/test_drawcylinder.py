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

# --- Arc: Exact and Smooth ---------------------------------------------------------
def arc_check(ctype, spans=4):
    r = bpy.ops.nproto.draw_arc(data=json.dumps({"matrix": T.tolist()}), radius=0.6, angle=120.0, ctype=ctype,
                                spans_u=spans)
    a = ctx.active_object
    sp = a.data.splines[0]
    S = geom.sample_curve(features.input_curve(a, "S0"), 360)
    rad = np.linalg.norm(S[:, :2] - [1, 2], axis=1)
    ang = np.degrees(np.arctan2(S[:, 1] - 2, S[:, 0] - 1))
    return r, a, sp, rad, ang

r, a, sp, rad, ang = arc_check("EXACT")
check("exact arc: rational, order 3, radius exact", r == {'FINISHED'} and sp.order_u == 3
      and min(p.co[3] for p in sp.points) < 0.9 and abs(rad - 0.6).max() < 1e-6, f"{abs(rad - 0.6).max():.1e}")
check("exact arc: history degree 0, 120°", features.history(a)["params"]["degree"] == 0
      and abs(np.ptp(ang) - 120) < 1e-3, f"{np.ptp(ang):.4f}°")
r, a, sp, rad, ang = arc_check("SMOOTH", 5)
h = features.history(a)["params"]
check("smooth arc: degree 3, not rational, spans + 3 CVs", r == {'FINISHED'} and sp.order_u == 4
      and all(p.co[3] == 1.0 for p in sp.points) and len(sp.points) == 5 + 3 and h["degree"] == 3 and h["spans_u"] == 5,
      f"order {sp.order_u}, {len(sp.points)} CVs")
check("smooth arc: close to the radius, ends on the arc's ends", abs(rad - 0.6).max() < 0.002 * 0.6
      and abs(np.ptp(ang) - 120) < 0.05, f"{abs(rad - 0.6).max() / 0.6:.3%}, {np.ptp(ang):.4f}°")

# --- Ellipse: Exact and Smooth (a generic shape: matrix + parameters) --------------
def ellipse(ctype, spans=8):
    params = primitive_params("prim_circle", radius=0.8, radius2=0.3, elliptic=True, degree=0)
    data = {"matrix": T.tolist(), "op": "prim_circle", "params": params, "name": "NURBS Ellipse"}
    r = bpy.ops.nproto.draw_ellipse(data=json.dumps(data), radius=0.8, radius2=0.3, ctype=ctype, spans_u=spans)
    e = ctx.active_object
    S = geom.sample_curve(features.input_curve(e, "S0"), 720)
    q = (S[:, 0] - 1) ** 2 / 0.8 ** 2 + (S[:, 1] - 2) ** 2 / 0.3 ** 2     # 1 on the ellipse
    return r, e, e.data.splines[0], q

r, e, sp, q = ellipse("EXACT")
check("exact ellipse: rational, on the ellipse", r == {'FINISHED'} and sp.order_u == 3
      and min(p.co[3] for p in sp.points) < 0.9 and abs(q - 1).max() < 1e-5, f"{abs(q - 1).max():.1e}")
check("exact ellipse: history degree 0", features.history(e)["params"]["degree"] == 0)
r, e, sp, q = ellipse("SMOOTH", 10)
h = features.history(e)["params"]
check("smooth ellipse: degree 3, periodic, not rational, 10 CVs", r == {'FINISHED'} and sp.order_u == 4
      and sp.use_cyclic_u and all(p.co[3] == 1.0 for p in sp.points) and len(sp.points) == 10
      and h["degree"] == 3 and h["spans_u"] == 10 and h["elliptic"], f"order {sp.order_u}, {len(sp.points)} CVs")
check("smooth ellipse: close to the ellipse", abs(np.sqrt(q) - 1).max() < 0.02, f"{abs(np.sqrt(q) - 1).max():.2%}")

# the interactive key: D flips the type on all three tools
class _Ev:
    value, type, ctrl, alt, oskey = 'PRESS', 'D', False, False, False
class _Op:
    _buf, ctype = "", "EXACT"
o_ = _Op()
check("D key: exact -> smooth -> exact", ds.type_key(o_, _Ev()) and o_.ctype == "SMOOTH"
      and ds.type_key(o_, _Ev()) and o_.ctype == "EXACT")
check("hints name the shape", "a true arc" in str(ds.type_rows(o_, "arc")) and "a true ellipse" in str(ds.type_rows(o_, "ellipse")))

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
