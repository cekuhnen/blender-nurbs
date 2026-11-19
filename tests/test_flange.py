"""Flange / Draft Surface:  blender -b --factory-startup --python tests/test_flange.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, flange as fl, geom, store, surfeval

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

def run(inputs, select=(), **kw):
    """The operator (EXEC): (result, [(points grid, normals grid)] per face, world)."""
    bpy.ops.object.select_all(action='DESELECT')
    for o in select:
        o.select_set(True)
    try:
        r = bpy.ops.nproto.flange(inputs=ins(*inputs), **kw)
    except RuntimeError as exc:
        print("   refused:", str(exc).strip())
        return {'CANCELLED'}, []
    if r != {'FINISHED'}:
        return r, []
    out = ctx.active_object
    M = np.array(out.matrix_world, float)
    faces = []
    for f in store.get_brep(out).faces():
        E = surfeval.Surf(f.surface())
        D = E.grid(np.linspace(*E.u_domain, 5), np.linspace(*E.v_domain, 25), 1)
        P = D[0, 0] @ M[:3, :3].T + M[:3, 3]
        n = surfeval.normals(D)[0] @ M[:3, :3].T
        faces.append((P, n))
    return r, faces

def outward(obj):
    f = store.get_brep(obj).faces()[0]
    (u0, v0), (u1, v1) = f.uv_domain()
    return np.array(f.outward_normal_at_uv(0.5 * (u0 + u1), 0.5 * (v0 + v1)), float)

def base_far(P, on_base):
    """(base row, far row) of a face grid: the base is the row on the edge."""
    return (P[0], P[-1]) if on_base(P[0]) else (P[-1], P[0])

# --- plane: every side, flange angles ----------------------------------------
clear()
bpy.ops.nproto.add_primitive(kind="PLANE", width=4, depth=4); PL = ctx.active_object
nz = float(outward(PL)[2])
on_border = lambda R: abs(R[:, 2]).max() < 1e-9 and abs(np.abs(R[:, :2]).max(axis=1) - 2).max() < 1e-9
for e in ("U0", "U1", "V0", "V1"):
    for ang in (90.0, 0.0, 45.0, -30.0):
        r, faces = run([(PL, e)], length=0.5, angle=ang)
        (P, N), = faces
        base, far = base_far(P, on_border)
        dz = far[:, 2] - base[:, 2]
        dout = np.abs(far[:, :2]).max(axis=1) - 2.0
        good = (abs(dz - nz * 0.5 * math.sin(math.radians(ang))).max() < 1e-6
                and abs(dout - 0.5 * math.cos(math.radians(ang))).max() < 1e-6)
        if ang == 0.0:          # continues the plane, facing the same way
            good &= float(N[2, 12] @ outward(PL)) > 0.999999
        check(f"plane {e} flange {ang:+.0f}", r == {'FINISHED'} and good,
              f"dz {dz.min():.4f}..{dz.max():.4f} out {dout.min():.4f}..{dout.max():.4f}")
out = ctx.active_object
h = features.history(out)
check("history: flange feature on the edge", h is not None and h["op"] == "flange" and h["inputs"][0][1] == "V1")

# --- draft: constant angle to the pull, also where edges run slanted ---------
r, faces = run([(PL, "U1")], fl_mode="DRAFT", length=1.0, draft=5.0)
(P, N), = faces
ang = np.degrees(np.arcsin(np.abs(N.reshape(-1, 3)[:, 2])))
base, far = base_far(P, on_border)
check("draft 5 deg, opening away from the plane", abs(ang - 5).max() < 1e-6 and far[:, 0].min() > 2.08,
      f"{ang.min():.5f}..{ang.max():.5f}")
r, faces = run([(PL, "U1")], fl_mode="DRAFT", length=1.0, draft=5.0, measure="HEIGHT")
(P, N), = faces
check("draft by height: flat top at z = 1", abs(np.abs(P[-1][:, 2]).max() - 1) < 1e-9 or abs(np.abs(P[0][:, 2]).max() - 1) < 1e-9)
r, faces = run([(PL, "U1")], fl_mode="DRAFT", length=1.0, draft=5.0, pull_flip=True)
(P, N), = faces
check("flip pull: goes down", min(P[0][:, 2].min(), P[-1][:, 2].min()) < -0.99)
PL.rotation_euler = (0, math.radians(35), math.radians(20)); ctx.view_layer.update()
for e in ("U0", "V1"):
    r, faces = run([(PL, e)], fl_mode="DRAFT", length=1.0, draft=7.0)
    (P, N), = faces
    ang = np.degrees(np.arcsin(np.abs(N.reshape(-1, 3)[:, 2])))
    check(f"tilted plane {e}: the surface keeps 7 deg draft", abs(ang - 7).max() < 1e-4, f"{ang.min():.5f}..{ang.max():.5f}")

# --- cylinder rim: 0 continues it (G1), +-90 a flat ring -----------------------
clear()
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0); CY = ctx.active_object
rims = []
for e in ("U0", "U1", "V0", "V1"):
    Q = geom.sample_curve(features.input_curve(CY, e), 9)
    if np.ptp(Q[:, 2]) < 1e-9 and np.ptp(np.hypot(Q[:, 0], Q[:, 1])) < 1e-9:
        rims.append((Q[0, 2], e))
ztop, top = max(rims)
f = store.get_brep(CY).faces()[0]; s = f.surface()
q = np.array(s.evaluate(0.3, 0.5)); n0 = np.array(f.outward_normal_at_uv(0.3, 0.5))
faces_out = float(n0 @ np.array([q[0], q[1], 0.0])) > 0          # (this primitive faces its axis)
r, faces = run([(CY, top)], length=0.3, angle=0.0)
(P, N), = faces
R = np.hypot(P[..., 0], P[..., 1])
on_rim = lambda X: abs(X[:, 2] - ztop).max() < 1e-9
base, _ = base_far(P, on_rim)
k = 0 if on_rim(P[0]) else len(P) - 1
radial = np.c_[P[k][:, :2], np.zeros(len(P[k]))]; radial /= np.linalg.norm(radial, axis=1, keepdims=True)
check("cylinder rim flange 0: the tube continues, G1", abs(R - 1).max() < 1e-6 and P[..., 2].max() > ztop + 0.299
      and abs(np.abs(np.einsum("ij,ij->i", N[k], radial)) - 1).max() < 1e-6)
r, faces = run([(CY, top)], length=0.3, angle=90.0)
(P, N), = faces
R = np.hypot(P[..., 0], P[..., 1])
want = 1.3 if faces_out else 0.7
check("cylinder rim flange 90: a flat ring along the face's outside", abs(P[..., 2] - ztop).max() < 1e-6
      and abs((R.max() if faces_out else R.min()) - want) < 1e-6, f"R {R.min():.5f}..{R.max():.5f}")
zbot, bot = min(rims)
r, faces = run([(CY, bot)], fl_mode="DRAFT", length=0.5, draft=10.0, pull_flip=True)
(P, N), = faces
ang = np.degrees(np.arcsin(np.abs(N.reshape(-1, 3)[:, 2])))
R = np.hypot(P[..., 0], P[..., 1])
check("draft from a circle (pull -Z): a 10 deg cone opening out", abs(ang - 10).max() < 1e-4 and R.max() > 1.08,
      f"{ang.min():.5f}..{ang.max():.5f}")

# --- trimmed: a hole's edge; four free outer edges ------------------------------
clear()
cu = bpy.data.curves.new("ln", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('POLY'); sp.points.add(1); sp.points[0].co = (0, -2, 0, 1); sp.points[1].co = (0, 2, 0, 1)
ln = bpy.data.objects.new("ln", cu); ctx.collection.objects.link(ln)
bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=4.0, direction="X"); FL = ctx.active_object; FL.name = "Floor"
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(1, 0, 1)); CC = ctx.active_object
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(target="Floor", cutters=ins((CC, "S0")), ref_point=(1, 0, 0), normal=(0, 0, 1),
                    view_dir=(0, 0, -1), action="DELETE")
edges = store.display(FL)["edges"]
hole = [i for i, e in enumerate(edges) if np.hypot(*(np.array(e)[:, :2] - [1, 0]).T).max() < 0.51][0]
fz = float(np.sign(outward(FL)[2]))
r, faces = run([(FL, f"E{hole}")], length=0.2, angle=0.0)
P = np.vstack([F.reshape(-1, 3) for F, _ in faces]) if faces else np.zeros((1, 3))
Rh = np.hypot(P[:, 0] - 1, P[:, 1])
check("hole edge flange 0: into the hole, in the floor's plane", r == {'FINISHED'} and abs(P[:, 2]).max() < 1e-6
      and abs(Rh.min() - 0.3) < 2e-4 and abs(Rh.max() - 0.5) < 1e-4, f"{Rh.min():.5f}..{Rh.max():.5f}")
r, faces = run([(FL, f"E{hole}")], length=0.2, angle=90.0)
P = np.vstack([F.reshape(-1, 3) for F, _ in faces])
Rh = np.hypot(P[:, 0] - 1, P[:, 1])
check("hole edge flange 90: a collar on the floor's outside", abs(Rh - 0.5).max() < 1e-4 and abs((P[:, 2] * fz).max() - 0.2) < 1e-6)
r, faces = run([(FL, f"E{hole}")], length=0.2, angle=0.0, flip=True)
P = np.vstack([F.reshape(-1, 3) for F, _ in faces])
Rh = np.hypot(P[:, 0] - 1, P[:, 1])
check("flip side: over the floor", abs(Rh.max() - 0.7) < 2e-4)
outer = [i for i in range(len(edges)) if i != hole]
r, faces = run([(FL, f"E{i}") for i in outer], length=0.3, angle=90.0)
check("four free edges: four walls", len(faces) == len(outer) and fl.LAST_INFO["faces"] == len(outer))

# --- curve on a surface: a wall on the sphere along a latitude ------------------
clear()
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0); SPH = ctx.active_object
lat = math.radians(30)
bpy.ops.curve.primitive_nurbs_circle_add(radius=math.cos(lat), location=(0, 0, math.sin(lat))); LAT = ctx.active_object
sph_out = float(outward(SPH) @ np.array(store.get_brep(SPH).faces()[0].surface().evaluate(0.3, 0.4))) > 0
r, faces = run([(LAT, "S0")], length=0.2, angle=90.0)          # the sphere is found under the curve
(P, N), = faces
rr = np.linalg.norm(P, axis=-1)
check("curve on sphere, 90: a radial wall", r == {'FINISHED'} and abs(rr.min() - (1 if sph_out else 0.8)) < 1e-4
      and abs(rr.max() - (1.2 if sph_out else 1.0)) < 1e-4, f"{rr.min():.5f}..{rr.max():.5f}")
r, faces = run([(LAT, "S0")], length=0.2, angle=0.0, both_sides=True)
check("curve on sphere, both sides: two faces", len(faces) == 2)
_h = features.history(ctx.active_object)["inputs"]
check("history: curve and surface inputs", [r_ for _, r_ in _h] == ["S0", "SURF"], str(_h))

# --- refusals -------------------------------------------------------------------
clear()
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=1.5, height=1); BX = ctx.active_object
joined = [i for i, e in enumerate(store.get_brep(BX).edges()) if len(e.faces()) == 2]
r, faces = run([(BX, f"E{joined[0]}")], length=0.3)
check("a joined edge is refused", r == {'CANCELLED'})

# --- the live preview is quick ---------------------------------------------------
clear()
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0); CY = ctx.active_object
prep = fl.prepare([(CY, top)])
t = time.time()
for a in np.linspace(0, 90, 10):
    fl.faces_rows(prep[0], {"fl_length": 0.3, "angle": float(a)})
pv = (time.time() - t) / 10
check("preview under 0.05 s", pv < 0.05, f"{pv * 1000:.1f} ms")
a = fl.anchor(prep[0], {"fl_length": 0.3, "angle": 90.0}, np.array([1.0, 0.0, ztop]))
check("handle frame at the rim", a is not None and abs(np.linalg.norm(a["C"] - [1, 0, ztop])) < 0.2)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
