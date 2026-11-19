"""Rail Revolve:  blender -b --factory-startup --python tests/test_railrevolve.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, geom, railrevolve as rr, store, surfeval

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

def nurbs(name, pts, order=4, cyclic=False, weights=None):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for k, (q, c) in enumerate(zip(sp.points, pts)):
        q.co = (*c, 1.0 if weights is None else weights[k])
    sp.order_u = order; sp.use_endpoint_u = True; sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ellipse(name, a, b, z, tilt=0.0):
    bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)); o = ctx.active_object; o.name = name
    o.scale = (a, b, 1.0); o.location = (0, 0, z); o.rotation_euler = (tilt, 0, 0)
    ctx.view_layer.update()
    return o

def run(pairs, **kw):
    try:
        r = bpy.ops.nproto.rail_revolve(inputs=ins(*pairs), **kw)
    except RuntimeError as exc:
        print("   refused:", str(exc).strip())
        return {'CANCELLED'}
    return r

def grid(obj, n=21):
    E = surfeval.Surf(store.get_brep(obj).faces()[0].surface())
    D = E.grid(np.linspace(*E.u_domain, n), np.linspace(*E.v_domain, 2 * n), 1)
    return D[0, 0], surfeval.normals(D)[0]

# the profile: an S-curve in the XZ half-plane, from near the axis up to (1, 0, 1)
PROF = nurbs("Prof", [(0.3, 0, 0), (0.9, 0, 0.1), (0.4, 0, 0.6), (0.7, 0, 0.8), (1.0, 0, 1.0)])
Pp = geom.sample_curve(features.input_curve(PROF, "S0"), 4000)
prof_rz = np.c_[Pp[:, 0], Pp[:, 2]]

def poly_dist(Q, L):
    """Distance of every point of Q to the polyline L (segments, not just its points)."""
    A, B = L[:-1], L[1:]
    AB = B - A
    den = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-300)
    out = np.empty(len(Q))
    for i, q in enumerate(Q):
        t = np.clip(((q - A) * AB).sum(axis=1) / den, 0, 1)
        out[i] = np.linalg.norm(A + t[:, None] * AB - q, axis=1).min()
    return out

def on_profile_rz(P, scale_r=None):
    """Distance of each point's (r, z) from the profile's (r, z) curve (r scaled by scale_r(point))."""
    r = np.hypot(P[:, 0], P[:, 1])
    if scale_r is not None:
        r = r / scale_r(P)
    return poly_dist(np.c_[r, P[:, 2]], prof_rz)

# --- a circle through the profile's end: a plain revolve ------------------------
C = ellipse("Circ", 1.0, 1.0, 1.0)
r = run([(C, "S0"), (PROF, "S0")])                      # rail first: told apart by the sweep
O = ctx.active_object
P, N = grid(O)
d = on_profile_rz(P.reshape(-1, 3))
check("circle rail = revolve: every section is the profile", r == {'FINISHED'} and d.max() < 1e-6,
      f"{d.max():.1e} · {rr.LAST_INFO}")
check("profile / rail told apart", rr.LAST_INFO["profile"].startswith("Prof") and rr.LAST_INFO["rail"].startswith("Circ"))
mid = P[P.shape[0] // 2]
outward = np.einsum("ij,ij->i", N[N.shape[0] // 2][:, :2], mid[:, :2])
check("faces away from the axis", (outward > 0).mean() > 0.9)
h = features.history(O)
check("history: rail_revolve with profile and rail", h["op"] == "rail_revolve" and len(h["inputs"]) == 2)

# --- an ellipse: every section is the profile stretched radially -----------------
EL = ellipse("Ell", 1.0, 0.6, 1.0)
r = run([(PROF, "S0"), (EL, "S0")])
O = ctx.active_object
P, N = grid(O)
Pf = P.reshape(-1, 3)
def r_rail(Q):
    th = np.arctan2(Q[:, 1], Q[:, 0])
    return 1.0 / np.sqrt((np.cos(th) / 1.0) ** 2 + (np.sin(th) / 0.6) ** 2)
d = on_profile_rz(Pf, r_rail)
end_row = P[-1] if abs(np.linalg.norm(P[-1][:, :2], axis=1) - r_rail(P[-1])).max() < 1e-6 else P[0]
Re = geom.sample_curve(features.input_curve(EL, "S0"), 6000)
end_row = P[-1]
de = poly_dist(end_row, Re)
# each section: the profile scaled by the rail's radius at that angle (the rail measured, not assumed an ellipse)
Rr = np.hypot(Re[:, 0], Re[:, 1]); Rth = np.arctan2(Re[:, 1], Re[:, 0]); o_ = np.argsort(Rth)
r_at = lambda Q: np.interp(np.arctan2(Q[:, 1], Q[:, 0]), Rth[o_], Rr[o_], period=2 * np.pi)
d = on_profile_rz(Pf, r_at)
check("ellipse rail: sections are the profile scaled to the rail", r == {'FINISHED'} and d.max() < 1e-4, f"{d.max():.1e}")
check("the profile's end runs along the rail", de.max() < 1e-6 and abs(end_row[:, 2] - 1.0).max() < 1e-9, f"{de.max():.1e}")

# --- a tilted rail about world Z: Keep ignores its height, Shift and Scale follow it --
TL = ellipse("Tilt", 1.0, 1.0, 1.0, tilt=math.radians(15))
Rs = geom.sample_curve(features.input_curve(TL, "S0"), 6000)
for mode in ("KEEP", "SHIFT", "SCALE"):
    r = run([(PROF, "S0"), (TL, "S0")], rr_height=mode, axis="Z", pivot="WORLD")
    O = ctx.active_object
    P, _ = grid(O, 31)
    end = P[-1]                         # (u: along the profile; its last row is the end on the rail)
    dr = poly_dist(end, Rs).max()
    if mode == "KEEP":
        check("Keep: the end stays at its height", abs(end[:, 2] - 1.0).max() < 1e-9, f"z {end[:, 2].min():.4f}..{end[:, 2].max():.4f}")
    else:
        check(f"{mode.title()}: the end runs on the tilted rail", dr < 1e-6, f"{dr:.1e}")
        if mode == "SCALE":
            check("Scale: the profile's start stays on the axis point's plane", abs(P[0][:, 2]).max() < 1e-9)

# --- Rail Centre (default): profile and rail away from the origin --------------------
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(4, 2, 1)); C2 = ctx.active_object
PR2 = nurbs("Prof2", [(4.3, 2, 0), (4.9, 2, 0.1), (4.4, 2, 0.6), (4.7, 2, 0.8), (5.0, 2, 1.0)])
r = run([(PR2, "S0"), (C2, "S0")])
P, _ = grid(ctx.active_object)
Q = P.reshape(-1, 3) - [4, 2, 0]
d = on_profile_rz(Q + [0, 0, 0], None) if False else poly_dist(np.c_[np.hypot(Q[:, 0], Q[:, 1]), Q[:, 2]],
                                                                 np.c_[prof_rz[:, 0], prof_rz[:, 1]])
check("Rail Centre: revolves about the rail's centre", r == {'FINISHED'} and d.max() < 1e-6, f"{d.max():.1e}")
TL.location = (0, 0, 1); ctx.view_layer.update()
r = run([(PROF, "S0"), (TL, "S0")])
check("Rail Centre on a tilted rail: the axis tilts with it (Keep follows the rail)",
      r == {'FINISHED'} and poly_dist(grid(ctx.active_object, 31)[0][-1], Rs).max() < 1e-6)

# --- a picked axis line, a rational rail (exact circle), the selection -----------
AX = nurbs("Axis", [(0, 0, -1), (0, 0, 2)], order=2)
r = run([(PROF, "S0"), (C, "S0"), (AX, "S0")], axis="PICKED")
check("axis from a picked line", r == {'FINISHED'})
bpy.ops.object.select_all(action='DESELECT')
PROF.select_set(True); C.select_set(True)
r = bpy.ops.nproto.rail_revolve('EXEC_DEFAULT', inputs=ins((PROF, "S0"), (C, "S0")))
check("operator", r == {'FINISHED'})

# --- off the axis plane: still built, and said ---------------------------------------
OFF = nurbs("Off", [(0.3, 0.0, 0), (0.6, 0.15, 0.4), (1.0, 0, 1.0)], order=3)
r = run([(OFF, "S0"), (C, "S0")])
check("a profile off a plane through the axis: built, reported", r == {'FINISHED'} and rr.LAST_INFO["off_plane"] > 0.1,
      f"{rr.LAST_INFO['off_plane']:.3f}")

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
