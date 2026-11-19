"""Pattern along a path on a surface (stitches along a seam):
blender -b --factory-startup --python tests/test_pattern_surface.py"""
import json, math, os, sys, time
import bpy
import numpy as np
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, pattern, store, surfoffset

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def select(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0]

def copies_of(src, group=None):
    out = [o for o in bpy.data.objects if o.get("nproto_in_0") is src and pattern.pattern_of(o) is not None
           and (group is None or pattern.pattern_of(o)[1]["group"] == group)]
    return sorted(out, key=lambda o: pattern.pattern_of(o)[1]["index"])

def W(o):
    return pattern.world_matrix(o)

def stitch(name, S, N, T, h):
    """A small box (a stitch) lying on the surface at S: Z the normal N, X along T, its centre h above."""
    bpy.ops.nproto.add_primitive(kind="BOX", width=0.4, depth=0.1, height=0.05)
    b = ctx.active_object
    b.name = name
    ctx.view_layer.update()
    c_loc = pattern.anchor(b)               # (identity matrix: the box's centre in its own space)
    Z = N / np.linalg.norm(N)
    X = T - (T @ Z) * Z
    X /= np.linalg.norm(X)
    R = np.column_stack([X, np.cross(Z, X), Z])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = S + h * Z - R @ c_loc
    b.matrix_world = Matrix(M.tolist())
    ctx.view_layer.update()
    return b

def on_surface_errors(cps, tg, h):
    """Per copy: (angle of its Z to the surface normal under its centre, |height - h|, |X . normal|)."""
    A = np.array([pattern.anchor(c) for c in cps])
    _, _, S, N, _ = tg.feet_all(A)
    ang, dh, xn = [], [], []
    for c, a, s, n in zip(cps, A, S, N):
        M = W(c)
        z = M[:3, 2] / np.linalg.norm(M[:3, 2])
        x = M[:3, 0] / np.linalg.norm(M[:3, 0])
        ang.append(math.degrees(math.acos(min(1.0, abs(float(z @ n))))))
        dh.append(abs(float(np.linalg.norm(a - s)) - h))
        xn.append(abs(float(x @ n)))
    return max(ang), max(dh), max(xn)


# --- a wavy curve on a sphere (not a circle: the path frame twists away from the normal) ---------
R, C = 3.0, np.array([0.0, 80.0, 0.0])
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=R)
SPH = ctx.active_object
SPH.name = "Seat"
SPH.location = tuple(C)
ctx.view_layer.update()

def on_sphere(th):
    ph = 0.45 * math.sin(3.0 * th)
    return C + R * np.array([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph)])

ths = np.linspace(0.3, 2.4, 400)
cu = bpy.data.curves.new("Seam", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('POLY'); sp.points.add(len(ths) - 1)
for q, th in zip(sp.points, ths):
    q.co = (*on_sphere(th), 1.0)
SEAM = bpy.data.objects.new("Seam", cu); ctx.collection.objects.link(SEAM)
ctx.view_layer.update()

tg = surfoffset.target_of([SPH])
P0 = on_sphere(0.3)
_, _, S0, N0, _ = tg.feet_all(P0[None])
h = 0.03
ST = stitch("Stitch", S0[0], N0[0], on_sphere(0.31) - on_sphere(0.3), h)

found = pattern.path_surfaces((SEAM, "S0"), [o for o in ctx.view_layer.objects], {ST.name})
check("surface: the curve is found lying on the sphere (and only on it)", found == [SPH], str([o.name for o in found]))

select(ST)
t0 = time.time()
r = bpy.ops.nproto.pattern(mode="PATH", count=12, path_obj=SEAM.name, path_ref="S0", orient="SURFACE")
dt = time.time() - t0
cps = copies_of(ST)
check("surface: 11 copies made", r == {'FINISHED'} and len(cps) == 11, f"{len(cps)} {dt:.2f}s")
check("surface: the sphere is the copies' third input",
      all(c.get("nproto_in_2") is SPH and features.history(c)["refs"][2] == "SURF" for c in cps))
ang, dh, xn = on_surface_errors(cps, tg, h)
check("surface: every copy's Z is the surface normal under it", ang < 0.05, f"{ang:.4f} deg")
check("surface: every copy sits as high above the surface as the source", dh < 1e-4, f"{dh:.2e}")
check("surface: every copy's X lies in the tangent plane", xn < 1e-3, f"{xn:.2e}")
pdS = pattern.path_data((SEAM, "S0"))
L = np.array([pattern.anchor(c) for c in [ST] + cps])
_, _, LS, _, _ = tg.feet_all(L)
gaps = np.diff([pdS.nearest(q) for q in LS])
check("surface: copies evenly spaced along the path (arc length)", float(np.ptp(gaps)) < 1e-3 * float(gaps.mean()),
      f"{gaps.min():.5f}..{gaps.max():.5f}")
check("surface: summary says so", "on the surface" in pattern.summary(pattern.pattern_of(cps[0])[1]))

# (on a sphere the normal is rotation-minimising along any curve: the path frame nearly agrees.
#  The panel below is where the two differ.)
select(ST)
r = bpy.ops.nproto.pattern(mode="PATH", count=12, path_obj=SEAM.name, path_ref="S0", orient="PATH")
cpp = [c for c in copies_of(ST) if c not in cps]
check("path frame: no surface input", all(c.get("nproto_in_2") is None for c in cpp))

# history: the surface grows, the copies follow it (the path is now inside it: the feet move out)
SPH.scale = (1.1, 1.1, 1.1)
ctx.view_layer.update()
for c in cps:
    features.rebuild(c)
tg2 = surfoffset.target_of([SPH])
A = np.array([pattern.anchor(c) for c in cps])
_, _, S2, N2, _ = tg2.feet_all(A)
z = np.array([W(c)[:3, 2] / np.linalg.norm(W(c)[:3, 2]) for c in cps])
ang2 = max(math.degrees(math.acos(min(1.0, abs(float(a @ b))))) for a, b in zip(z, N2))
hs = np.linalg.norm(A - S2, axis=1)
h_src = float(np.linalg.norm(pattern.anchor(ST) - tg2.feet_all(pattern.anchor(ST)[None])[2][0]))
check("history: copies follow the surface when it changes", ang2 < 0.05 and float(np.abs(hs - h_src).max()) < 1e-4,
      f"{ang2:.4f} deg {np.abs(hs - h_src).max():.2e}")
SPH.scale = (1, 1, 1)
ctx.view_layer.update()
for c in cps:
    features.rebuild(c)

# Edit Pattern keeps the surface
prm = pattern.pattern_of(cps[0])[1]
check("params keep orient", prm.get("orient") == "SURFACE")
old = {k: v for k, v in prm.items() if k != "orient"}
check("patterns made before the option: turned with the path",
      np.abs(pattern.copy_transform(old, ST, 3, (SEAM, "S0"), None, [SPH]) -
             pattern.copy_transform(dict(old, orient="PATH"), ST, 3, (SEAM, "S0"))).max() < 1e-12)

# --- a surface edge (twisted: not in a plane) ---------------------------------------------------
_, ext = features._k()
pts = []
for i in range(4):
    for j in range(4):
        x, y = 2.0 * i, 2.0 * j
        pts.append((x, 40.0 + y + 0.6 * math.sin(1.3 * x), 0.8 * math.sin(0.9 * x + 0.5 * y) + 0.15 * x * y))
kn = [0.0] * 4 + [1.0] * 4
srf = ext.surface_from_cvs(3, 3, pts, kn, kn, None)
bpy.ops.nproto.add_primitive(kind="PLANE", width=1.0, depth=1.0)
PAN = ctx.active_object
PAN.name = "Panel"
store.set_geometry(PAN, srf)
ctx.view_layer.update()
edges = store.display(PAN)["edges"]
tgp = surfoffset.target_of([PAN])
ref = "V0"
kc = features.input_curve(PAN, ref)
t0_, t1_ = kc.parameter_range()
E0 = np.array(kc.evaluate(t0_ + 0.05 * (t1_ - t0_)), float)
E1 = np.array(kc.evaluate(t0_ + 0.06 * (t1_ - t0_)), float)
_, _, Se, Ne, _ = tgp.feet_all(E0[None])
# the stitch a little inside the panel (the face it sits on)
ST2 = stitch("Stitch2", Se[0], Ne[0], E1 - E0, h)
select(ST2)
r = bpy.ops.nproto.pattern(mode="PATH", count=8, path_obj=PAN.name, path_ref=ref, orient="SURFACE")
ce = copies_of(ST2)
ang, dh, xn = on_surface_errors(ce, tgp, h)
check("edge: copies along a surface edge follow its surface", r == {'FINISHED'} and len(ce) == 7 and ang < 0.05
      and dh < 1e-4 and xn < 1e-3, f"{len(ce)} {ang:.4f} deg {dh:.2e} {xn:.2e}")
check("edge: the edge's own object is not added again as an input", all(c.get("nproto_in_2") is None for c in ce))
select(ST2)
r = bpy.ops.nproto.pattern(mode="PATH", count=8, path_obj=PAN.name, path_ref=ref, orient="PATH")
ce2 = [c for c in copies_of(ST2) if c not in ce]
angp, _, _ = on_surface_errors(ce2, tgp, h)
check("edge: the path frame tilts off it", angp > 1.0, f"{angp:.2f} deg")

# a wavy curve inside the panel (a stitch line): found on it, copies on it
uv = [(0.15 + 0.7 * t, 0.5 + 0.25 * math.sin(2.5 * math.pi * t)) for t in np.linspace(0, 1, 400)]
cu = bpy.data.curves.new("Line", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('POLY'); sp.points.add(len(uv) - 1)
for q, (u, v) in zip(sp.points, uv):
    q.co = (*srf.evaluate(u, v), 1.0)
LINE = bpy.data.objects.new("StitchLine", cu); ctx.collection.objects.link(LINE)
ctx.view_layer.update()
Q0, Q1 = np.array(srf.evaluate(*uv[0]), float), np.array(srf.evaluate(*uv[1]), float)
_, _, Sq, Nq, _ = tgp.feet_all(Q0[None])
ST3 = stitch("Stitch3", Sq[0], Nq[0], Q1 - Q0, h)
select(ST3)
r = bpy.ops.nproto.pattern(mode="PATH", count=15, path_obj=LINE.name, path_ref="S0")     # (Surface: the default)
cl = copies_of(ST3)
ang, dh, xn = on_surface_errors(cl, tgp, h)
check("curve on a panel: found, copies on it by default", len(cl) == 14 and all(c.get("nproto_in_2") is PAN for c in cl)
      and ang < 0.05 and dh < 1e-4 and xn < 1e-3, f"{len(cl)} {ang:.4f} deg {dh:.2e} {xn:.2e}")
select(ST3)
r = bpy.ops.nproto.pattern(mode="PATH", count=15, path_obj=LINE.name, path_ref="S0", orient="PATH")
cl2 = [c for c in copies_of(ST3) if c not in cl]
angp, _, _ = on_surface_errors(cl2, tgp, h)
check("curve on a panel: the path frame tilts off it (what Surface fixes)", angp > 2.0, f"{angp:.2f} deg")

# --- a seam: the edge between two joined panels (folded 60 degrees) -------------------------------
def zA(x, y):
    return 0.3 * math.sin(0.7 * y) + 0.05 * x
ptsA, ptsB = [], []
for i in range(4):
    for j in range(4):
        x, y = 2.0 * i, 2.0 * j
        ptsA.append((x, 20.0 + y, zA(x, y)))
for i in range(4):
    for j in range(4):
        y = 2.0 * j
        ptsB.append((6.0 + 2.0 * i * math.cos(math.radians(60)), 20.0 + y,
                     zA(6.0, y) - 2.0 * i * math.sin(math.radians(60)) + 0.1 * i * math.sin(y)))
sA = ext.surface_from_cvs(3, 3, ptsA, kn, kn, None)
sB = ext.surface_from_cvs(3, 3, ptsB, kn, kn, None)
objs = []
for nm, sf in (("PanelA", sA), ("PanelB", sB)):
    bpy.ops.nproto.add_primitive(kind="PLANE", width=1.0, depth=1.0)
    o = ctx.active_object
    o.name = nm
    store.set_geometry(o, sf)
    objs.append(o)
select(*objs)
r = bpy.ops.nproto.join()
J = ctx.active_object
jb = store.get_brep(J)
seam = [i for i, e in enumerate(jb.edges()) if len(e.faces()) == 2]
check("seam: two panels joined along one edge", r == {'FINISHED'} and len(seam) == 1, f"{seam}")
tgj = surfoffset.target_of([J])
tA, tB = surfoffset.target_of([objs[0]]), surfoffset.target_of([objs[1]])
for side, sf, (u, v), (u2, v2), tside, other in (("B", sB, (0.06, 0.1), (0.06, 0.11), tB, tA),
                                                  ("A", sA, (0.94, 0.1), (0.94, 0.11), tA, tB)):
    Pq, Pq2 = np.array(sf.evaluate(u, v), float), np.array(sf.evaluate(u2, v2), float)
    _, _, Sq, Nq, _ = tside.feet_all(Pq[None])
    stj = stitch("Seam" + side, Sq[0], Nq[0], Pq2 - Pq, h)
    select(stj)
    r = bpy.ops.nproto.pattern(mode="PATH", count=8, path_obj=J.name, path_ref=f"E{seam[0]}")
    cj = copies_of(stj)
    # (the stitch sits 0.36 off the seam: its copies keep that offset from the frame at the seam,
    #  so the normal under them agrees to the panel's bend across that width, not exactly)
    ang, dh, xn = on_surface_errors(cj, tside, h)
    pdj = pattern.path_data((J, f"E{seam[0]}"))
    on_seam = np.array([pdj.at(pdj.nearest(pattern.anchor(c)))[0] for c in cj])
    No = other.feet_all(on_seam)[3]
    z = np.array([W(c)[:3, 2] / np.linalg.norm(W(c)[:3, 2]) for c in cj])
    ang_o = min(math.degrees(math.acos(min(1.0, abs(float(a @ b))))) for a, b in zip(z, No))
    check(f"seam: a stitch on panel {side}: its copies follow panel {side}, not the other",
          len(cj) == 7 and ang < 0.5 and dh < 1e-3 and ang_o > 50.0,
          f"{len(cj)} {ang:.4f} deg {dh:.2e}; to the other panel {ang_o:.1f} deg")

# --- a curve on no surface: turned with the path ----------------------------------------------
bpy.ops.curve.primitive_nurbs_circle_add(radius=5.0, location=(0, -60, 0))
CIR = ctx.active_object
bpy.ops.nproto.add_primitive(kind="BOX", width=0.4, depth=0.8, height=0.3)
PB = ctx.active_object
PB.location = (5.0, -60.0, 0.0)
ctx.view_layer.update()
select(PB)
r = bpy.ops.nproto.pattern(mode="PATH", count=4, path_obj=CIR.name, path_ref="S0", orient="SURFACE")
cc = copies_of(PB)
prm = pattern.pattern_of(cc[0])[1]
err = max(np.abs(W(c) - pattern.copy_transform(dict(prm, orient="PATH"), PB, pattern.pattern_of(c)[1]["index"],
                                                (CIR, "S0")) @ W(PB)).max() for c in cc)
check("no surface: the path frame, and no surface input", len(cc) == 3 and err < 1e-5
      and all(c.get("nproto_in_2") is None for c in cc), f"{err:.2e}")

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
