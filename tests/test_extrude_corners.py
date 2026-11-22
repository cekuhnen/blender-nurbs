"""Extrude profiles (straight-line normal, chained curves and edges, draft side)
and Flange corners (Open / Mitre / Blend):
    blender -b --factory-startup --python tests/test_extrude_corners.py"""
import json, math, os, sys
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
_, ext = features._k()

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

def nurbs(name, pts, order=2):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.order_u = order; sp.use_endpoint_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def arc(radius, loc, rot=0.0):
    bpy.ops.nproto.add_primitive(kind="ARC", radius=radius, angle=180.0, degree="0")
    o = ctx.active_object
    o.location = loc; o.rotation_euler = (0, 0, rot)
    ctx.view_layer.update()
    return o

def cvs(o, k=0):
    d = ext.surface_cvs(store.get_surfaces(o)[k])
    return np.array(d["points"], float).reshape(d["count_u"], d["count_v"], 3), d

def grid(f, nu=5, nv=41):
    E = surfeval.Surf(f.surface())
    D = E.grid(np.linspace(*E.u_domain, nu), np.linspace(*E.v_domain, nv), 1)
    return D[0, 0], surfeval.normals(D)[0]

def extrude(*pairs, **kw):
    bpy.ops.object.select_all(action='DESELECT')
    r = bpy.ops.nproto.extrude(inputs=ins(*pairs), **kw)
    return r, sorted(ctx.selected_objects, key=lambda o: o.name)

# --- 1. Curve Normal of a straight line: the work plane's normal, not chance -------
clear()
dirs = []
for x in (0.0508, 0.254, -0.37, 1.0 / 3.0, 12.7):
    L = nurbs("L", [(x, 0.1778, 0.0), (x, -0.127, 0.0)])
    dirs.append(np.array(features.extrude_direction(L, features.input_curve(L, "S0"), {"direction": "NORMAL"})))
    bpy.data.objects.remove(L)
check("a straight line extrudes along world Z wherever it lies (rounding no longer picks)",
      all(abs(d - [0, 0, 1]).max() < 1e-12 for d in dirs), str([np.round(d, 6).tolist() for d in dirs]))
L = nurbs("L", [(0.254, 0.1778, 0.0), (0.254, -0.127, 0.0)])
c = features.input_curve(L, "S0")
d1 = np.array(features.extrude_direction(L, c, {"direction": "NORMAL", "normal_hint": [1.0, 0.0, 0.0]}))
d2 = np.array(features.extrude_direction(L, c, {"direction": "NORMAL", "normal_hint": [0.0, 0.3, 1.0]}))
check("a straight line takes the work plane's normal, made square to it",
      abs(d1 - [1, 0, 0]).max() < 1e-12 and abs(d2 - [0, 0, 1]).max() < 1e-12, f"{d1} {d2}")
V = nurbs("V", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
dv = np.array(features.extrude_direction(V, features.input_curve(V, "S0"), {"direction": "NORMAL"}))
check("a line along the hint: a world axis square to it", abs(dv @ [0, 0, 1]) < 1e-12 and abs(np.linalg.norm(dv) - 1) < 1e-12)
r, (E,) = extrude((L, "S0"), distance=0.12)
P, _ = cvs(E)
check("Extrude of the line from the file: a wall standing up (Blender keeps curve points in single precision)",
      abs(P[1, :, 2] - 0.12).max() < 1e-6 and abs(P[:, :, 0] - 0.254).max() < 1e-6)
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); CI = ctx.active_object
cc = features.input_curve(CI, "S0")
dn = np.array(features.extrude_direction(CI, cc, {"direction": "NORMAL", "normal_hint": [0, 0, -1.0]}))
d_old = np.array(features.extrude_direction(CI, cc, {"direction": "NORMAL"}))
check("a planar curve's normal turns to the hint's side; without one (older extrudes) it is the plane's own",
      abs(dn - [0, 0, -1]).max() < 1e-9 and abs(d_old - np.array(geom.curve_plane_normal(cc)[0])).max() < 1e-15)

# --- 2. a slot outline of two lines and two arcs: ONE drafted surface ---------------
clear()
A1 = arc(1.0, (0.0, 1.0, 0.0))                 # (1,1) over (0,2) to (-1,1)
A2 = arc(1.0, (0.0, -1.0, 0.0), math.pi)       # (-1,-1) under (0,-2) to (1,-1)
L1 = nurbs("L1", [(-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)])
L2 = nurbs("L2", [(1.0, -1.0, 0.0), (1.0, 1.0, 0.0)])   # drawn the other way round
h, a = 0.5, 10.0
off = h * math.tan(math.radians(a))
r, objs = extrude((L1, "S0"), (A1, "S0"), (L2, "S0"), (A2, "S0"), distance=h, draft=a)
f = store.get_brep(objs[0]).faces() if objs else []
G, N = grid(f[0], 2, 401) if f else (None, None)
far = G[-1] if G is not None and G[-1][:, 2].max() > 0.4 else (G[0] if G is not None else None)
def slot_dist(q):
    """In-plane distance of q outside the slot |x| <= 1 (|y| <= 1) with round ends."""
    x, y = q[0], q[1]
    if abs(y) <= 1.0:
        return abs(x) - 1.0
    return math.hypot(x, abs(y) - 1.0) - 1.0
dd = np.array([slot_dist(q) for q in far]) if far is not None else np.array([9.0])
check("two lines + two arcs: one surface, every side drafted outward by h tan(draft)",
      r == {'FINISHED'} and len(objs) == 1 and len(f) == 1 and abs(dd - off).max() < 1e-5
      and abs(far[:, 2] - h).max() < 1e-9, f"objects {len(objs)} faces {len(f)} off {dd.min():.6f}..{dd.max():.6f} want {off:.6f}")
hist = features.history(objs[0]) if objs else None
check("history: the chain's four inputs, rebuilt the same",
      hist is not None and len(hist["inputs"]) == 4 and features.rebuild(objs[0], ctx, force=True) is not False)
r, objs = extrude((L1, "S0"), (A1, "S0"), (L2, "S0"), (A2, "S0"), distance=h, draft=a, chain=False)
check("Chain off: one surface each, as before", len(objs) == 4)

# --- 3. separate open curves drafted together lean out of their middle --------------
r, objs = extrude((L1, "S0"), (L2, "S0"), distance=h, draft=a)
xs = sorted(float(cvs(o)[0][1, :, 0].mean()) for o in objs)
check("two parallel lines with draft: both lean away from the middle",
      len(objs) == 2 and abs(xs[0] - (-1 - off)) < 1e-9 and abs(xs[1] - (1 + off)) < 1e-9, str(xs))
from blender_nurbs.ops import extrude_rows
rows = extrude_rows(np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]), np.array([0, 0, 1.0]), h, a, smooth=False,
                    centre=[0.0, 0.0, 0.0])
check("the preview drafts the same side", abs(rows[1][:, 0] - (1 + off)).max() < 1e-12)

# --- 4. several surface edges: one profile, mitred corners --------------------------
clear()
bpy.ops.nproto.add_primitive(kind="PLANE", width=2, depth=2); PL = ctx.active_object
sides = [s for s in ("U0", "V1", "U1")]
r, objs = extrude(*[(PL, s) for s in sides], distance=h, draft=a)
P, _ = cvs(objs[0]) if objs else (np.zeros((2, 1, 3)), None)
top = P[1]
corner_ok = sum(1 for q in top if abs(abs(q[0]) - (1 + off)) < 1e-9 and abs(abs(q[1]) - (1 + off)) < 1e-9)
check("three edges of a plane: one surface, the far edge mitred at both corners",
      r == {'FINISHED'} and len(objs) == 1 and corner_ok == 2 and abs(top[:, 2] - h).max() < 1e-9,
      f"objects {len(objs)} corners {corner_ok}")
r, objs = extrude((PL, "U0"), distance=h)
check("one edge still works", r == {'FINISHED'} and len(objs) == 1)
# branching: three lines from one point stay three surfaces
B1 = nurbs("B1", [(0, 0, 0), (1, 0, 0)]); B2 = nurbs("B2", [(0, 0, 0), (0, 1, 0)]); B3 = nurbs("B3", [(0, 0, 0), (-1, -1, 0)])
r, objs = extrude((B1, "S0"), (B2, "S0"), (B3, "S0"), distance=h)
check("curves that branch: one surface each", r == {'FINISHED'} and len(objs) == 3)

# --- 5. Flange corners on three plane edges ---------------------------------------
def flange(inputs, **kw):
    bpy.ops.object.select_all(action='DESELECT')
    r = bpy.ops.nproto.flange(inputs=ins(*inputs), **kw)
    return r, ctx.active_object

def shared(o):
    return sum(1 for e in store.get_brep(o).edges() if len(e.faces()) == 2)

def planar(P):
    c = P.reshape(-1, 3).mean(axis=0)
    _, S, _ = np.linalg.svd(P.reshape(-1, 3) - c)
    return S[2] < 1e-9

clear()
bpy.ops.nproto.add_primitive(kind="PLANE", width=2, depth=2); PL = ctx.active_object
E3 = [(PL, s) for s in ("U0", "V1", "U1")]
Lf, ang = 0.3, 70.0
reach = Lf * math.cos(math.radians(ang))
r, o = flange(E3, length=Lf, angle=ang, fl_corner="OPEN")
check("Open: three faces, nothing shared (as before)", r == {'FINISHED'} and fl.LAST_INFO["faces"] == 3
      and shared(o) == 0 and fl.LAST_INFO["open"] == 2)
r, o = flange(E3, length=Lf, angle=ang, fl_corner="MITRE")
faces = store.get_brep(o).faces()
nets = [cvs(o, k)[0] for k in range(len(faces))]
farpts = np.vstack([n.reshape(-1, 3) for n in nets])
corners = [q for q in farpts if abs(abs(q[0]) - (1 + reach)) < 1e-6 and abs(abs(q[1]) - (1 + reach)) < 1e-6]
check("Mitre: three flat faces stitched at two corners, far edges meeting at the mitre point",
      r == {'FINISHED'} and len(faces) == 3 and shared(o) == 2 and fl.LAST_INFO["mitre"] == 2
      and all(planar(n) for n in nets) and len(corners) == 4, f"shared {shared(o)} corners {len(corners)}")
# the flange angle in the middle of each side is still 70 degrees
okang = True
for fc in faces:
    G, Nn = grid(fc, 3, 41)
    nrm = Nn[1, 20]
    okang &= abs(abs(nrm[2]) - math.cos(math.radians(ang))) < 1e-6      # the face stands at 70 deg to the plane
check("Mitre keeps the flange angle along the sides", okang)
r, o = flange(E3, length=Lf, angle=ang, fl_corner="BLEND")
faces = store.get_brep(o).faces()
check("Blend: two round corners added, all stitched", r == {'FINISHED'} and len(faces) == 5 and shared(o) == 4
      and fl.LAST_INFO["blend"] == 2, f"faces {len(faces)} shared {shared(o)}")
# the corner is a cone: its far edge a quarter circle about the corner's vertical, tangent to both flanges
fans = [fc for fc in faces if np.ptp(cvs(o, faces.index(fc))[0][0], axis=0).max() < 1e-12]
g1, round_ok, same_way = 0.0, True, True
for fc in fans:
    G, Nn = grid(fc, 5, 41)
    apex = G[0, 0]
    R = np.linalg.norm(G[-1][:, :2] - apex[:2], axis=1)
    round_ok &= abs(R - reach).max() < 1e-6 and abs(G[-1][:, 2] - Lf * math.sin(math.radians(ang))).max() < 1e-6
    for other in faces:
        if other is fc:
            continue
        Go, No = grid(other, 5, 201)
        for k in (0, -1):
            for j in (0, -1):            # the fan's rulings to its two far ends
                if np.linalg.norm(Go[-1][k] - G[-1][j]) < 1e-6 or np.linalg.norm(Go[0][k] - G[-1][j]) < 1e-6:
                    dot = float(No[2, k] @ Nn[2, j])
                    same_way &= dot > 0
                    g1 = max(g1, math.degrees(math.acos(min(1.0, abs(dot)))))
check("Blend corners: round (a cone), tangent to the flanges they join", len(fans) == 2 and round_ok and g1 < 1e-5,
      f"fans {len(fans)} G1 {g1:.2e} deg")
check("Blend corners face the way their flanges do", same_way)

# inside corners (flanges over the face): cut back, Blend mitres them too
r, o = flange(E3, length=Lf, angle=ang, flip=True, fl_corner="BLEND")
farpts = np.vstack([cvs(o, k)[0].reshape(-1, 3) for k in range(len(store.get_brep(o).faces()))])
inner = [q for q in farpts if abs(abs(q[0]) - (1 - reach)) < 1e-6 and abs(abs(q[1]) - (1 - reach)) < 1e-6]
check("inside corners: mitred (cut back) even with Blend", r == {'FINISHED'} and fl.LAST_INFO["mitre"] == 2
      and fl.LAST_INFO["blend"] == 0 and shared(o) == 2 and len(inner) == 4)

# an older flange (no Corners in its history) still builds open
h_ = json.loads(o[features.PROP_HIST]); h_["params"].pop("fl_corner", None); o[features.PROP_HIST] = json.dumps(h_)
features.rebuild(o, ctx, force=True)
check("a flange made before Corners existed stays open", len(store.get_brep(o).faces()) == 3 and shared(o) == 0)

# --- 6. a curved edge meeting a straight one -----------------------------------------
clear()
A = arc(1.0, (0.0, 0.0, 0.0))
r, (W,) = extrude((A, "S0"), distance=1.0, direction="Z")
B = store.get_brep(W)
edges = B.edges()
def mid(e):
    t0, t1 = e.curve().parameter_range() if hasattr(e, "curve") else (0, 1)
    return np.array(e.curve().evaluate(0.5 * (t0 + t1)), float)
picks = []
for i, e in enumerate(edges):
    m = mid(e)
    if abs(m[2]) < 1e-9 or (abs(m[2] - 0.5) < 1e-6 and m[0] > 0.5):     # the bottom arc and the side at x = 1
        picks.append((W, f"E{i}"))
for mode in ("MITRE", "BLEND"):
    r, o = flange(picks, length=0.2, angle=45.0, fl_corner=mode)
    info = dict(fl.LAST_INFO)
    nf = len(store.get_brep(o).faces())
    # away from the corner the arc's flange is untouched: its far edge a circle at one height
    spread = None
    for fc in store.get_brep(o).faces():
        G, _ = grid(fc, 3, 181)
        for base, far_ in ((G[0], G[-1]), (G[-1], G[0])):
            if abs(base[:, 2]).max() < 1e-9 and abs(np.linalg.norm(base[:, :2], axis=1) - 1).max() < 1e-6 \
                    and np.ptp(base, axis=0).max() > 0.1:          # (not a Blend corner's point)
                ang_ = np.degrees(np.arctan2(base[:, 1], base[:, 0]))
                keep = (ang_ > 40.0) & (ang_ < 179.0)          # the corner is at (1, 0, 0)
                Rr = np.linalg.norm(far_[keep, :2], axis=1)
                spread = float(max(np.ptp(Rr), np.ptp(far_[keep, 2])))
    check(f"curved + straight edge, {mode}: stitched at the corner, the arc's flange unchanged away from it",
          r == {'FINISHED'} and shared(o) >= 1 and (info["mitre"] if mode == "MITRE" else info["blend"]) == 1
          and spread is not None and spread < 1e-4, f"faces {nf} shared {shared(o)} {info} spread {spread}")

# --- 4. A joined profile extrudes with a face per piece: a trimmed rectangle's
#        three sides joined to a half circle that meets them tangentially. The
#        arc gets its own face, its rims are circular edges (their centre snaps).
from blender_nurbs import snapping
clear()
U = nurbs("U", [(-0.15, 0.28, 0), (-0.15, -0.28, 0), (0.15, -0.28, 0), (0.15, 0.28, 0)])
A = arc(0.15, (0.0, 0.28, 0.0))
bpy.ops.object.select_all(action='DESELECT')
for o in (U, A):
    o.select_set(True)
ctx.view_layer.objects.active = U
r = bpy.ops.nproto.join_curves()
J = ctx.active_object
bpy.ops.object.select_all(action='DESELECT')
r2 = bpy.ops.nproto.extrude(inputs=ins((J, "S0")), distance=0.6, direction="Z", body="NEW")
S = ctx.active_object
B = store.get_brep(S)
cens = []
for e in B.edges():
    t0, t1 = e.parameter_range()
    c = snapping.circle_centre([e.point_at(t0 + (t1 - t0) * k / 12) for k in range(13)])
    if c is not None:
        cens.append(np.round(c, 6))
vol = abs(B.volume(1e-6)) if B.is_manifold_solid() else 0.0
want = (0.3 * 0.56 + 0.5 * math.pi * 0.15 ** 2) * 0.6
check("joined line + tangent arc: one wall face per piece, the arc's rims circular (centre snaps)",
      r == {'FINISHED'} and r2 == {'FINISHED'} and B.face_count() == 6 and len(cens) == 2
      and all(abs(c[0]) < 1e-6 and abs(c[1] - 0.28) < 1e-6 for c in cens) and abs(vol - want) < 1e-6 * want,
      f"faces {B.face_count()} centres {[c.tolist() for c in cens]} vol {vol} want {want}")

# a slot (one spline: lines into half circles, no join record) splits there too
clear()
from blender_nurbs import primitives
cu = bpy.data.curves.new("Slot", 'CURVE')
K = bpy.data.objects.new("Slot", cu); ctx.collection.objects.link(K)
primitives.write_spline_object(K, features.run("prim_slot", [], {"length": 1.0, "width": 0.4})[1])
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.extrude(inputs=ins((K, "S0")), distance=0.5, direction="Z", body="NEW")
B = store.get_brep(ctx.active_object)
ncirc = sum(1 for e in B.edges() if snapping.circle_centre(
    [e.point_at(e.parameter_range()[0] + (e.parameter_range()[1] - e.parameter_range()[0]) * k / 12)
     for k in range(13)]) is not None)
check("slot: two straight walls, two half-round walls, four circular rims",
      r == {'FINISHED'} and B.face_count() == 6 and ncirc == 4 and B.is_manifold_solid(),
      f"faces {B.face_count()} circular edges {ncirc}")

print("ALL OK" if ok else "SOME FAILED")
