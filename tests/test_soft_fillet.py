"""Soft fillet (Surface Fillet and Edge Fillet, Shape Soft): one Bezier span
across, G2 / G3 to both faces, minimum radius exactly the Radius:
    blender -b --factory-startup --python tests/test_soft_fillet.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, history_ui
from blender_nurbs.kernel import softfillet as sf
from blender_nurbs.ops_trim import soft_sections

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def poly(name, pts, cyclic=False):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o
def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])
def patch(name, p0, p1, d, dist):
    ln = poly(name + "_ln", [p0, p1])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=dist, direction=d)
    o = ctx.active_object; o.name = name; return o
def A(o): return store.get_brep(o).area(1e-7)

# --- the section itself (pure numpy) ---------------------------------------
phi = math.pi / 2
for cont in (2, 3):
    worst, bumps = 0.0, 0
    for deg in (20, 45, 90, 135, 160):
        for s in (0.1, 0.3, 0.6, 1.0):
            d = sf.shape(math.radians(deg), sf.clamp_smoothing(s, cont), cont)
            P = sf.planar_cvs(d["spacing"], math.radians(deg))
            k = np.abs(sf.curvature(P, np.linspace(0.0, 0.5, 401)))
            worst = max(worst, abs(k.max() - d["g"]) / d["g"])
            bumps += not d["monotone"] or not d["ok"]
            worst = max(worst, abs(k[0]))                       # zero curvature at the face (a plane)
    check(f"soft G{cont} section: peak curvature g / L, zero at the faces, rising steadily (20-160 deg, "
          f"smoothing 0.1-1)", worst < 1e-6 and bumps == 0, f"worst {worst:.1e}, bumps {bumps}")
d = sf.shape(phi, 0.6, 2)
P = sf.planar_cvs(d["spacing"], phi)
d1 = sf._ders(P, 5, np.array([0.0]), 1)[0]
check("soft G2: degree 5, 3 CVs on each face's tangent line", d["degree"] == 5 and len(P) == 6
      and abs(P[2][1]) < 1e-15 and abs(P[3][0] - 1.0) < 1e-15 and d1[1] == 0.0)
d3 = sf.shape(phi, 0.6, 3)
check("soft G3: degree 7, 4 CVs on each tangent line", d3["degree"] == 7 and len(d3["spacing"]) == 3)
# smoother -> longer setback for the same minimum radius
gs = [sf.shape(phi, s, 2)["g"] for s in (0.1, 0.4, 0.7, 1.0)]
check("soft: more smoothing reaches further along the faces", all(a < b for a, b in zip(gs, gs[1:])),
      " ".join(f"{g:.3f}" for g in gs))
# G3 on a curved face: the span takes the face's curvature and its rate at the rail
pa, ta, na = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
pb, tb, nb = np.array([3.0, 3.0, 0]), np.array([0, -1.0, 0]), np.array([-1.0, 0, 0])
for cont, k3 in ((2, 0.0), (3, 0.7)):
    sh = sf.shape(phi, 0.6, cont)
    cv = sf.section_cvs(pa, ta, na, 0.4, pb, tb, nb, 0.0, 3.0, sh["spacing"], sh["degree"], k3, 0.0)
    Q = cv[:, :2]
    k0 = sf.curvature(Q, np.array([0.0]))[0]
    h = 1e-6
    rate = (sf.curvature(Q, np.array([h]))[0] - k0) / (h * np.linalg.norm(sf._ders(Q, len(Q) - 1, np.array([0.0]), 1)))
    good = abs(k0 - 0.4) < 1e-9 and (cont == 2 or abs(rate - k3) < 1e-3)
    check(f"soft G{cont} on a face curving 0.4 across" + (" at rate 0.7" if cont == 3 else ""), good,
          f"curvature {k0:.6f}" + (f", rate {rate:.4f}" if cont == 3 else ""))

# --- Surface Fillet, Soft ----------------------------------------------------
fl = patch("Floor", (0, -2, 0), (0, 2, 0), "X", 4.0)                 # z = 0, x 0..4
wl = patch("Wall", (2, -2, -1), (2, 2, -1), "Z", 3.0)                # x = 2, z -1..2
R = 0.3
for cont in ("2", "3"):
    bpy.ops.object.select_all(action='DESELECT')
    r = bpy.ops.nproto.fillet(surf_a="Floor", surf_b="Wall", radius=R, shape="SOFT", continuity=cont,
                              smoothing=0.6, trim_bases=False)
    fil = ctx.active_object
    b = store.get_brep(fil)
    s = b.faces()[0].surface()
    (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
    # the section through the middle (v across: the skin's u runs along the crossing)
    vm = 0.5 * (u0 + u1)
    X = np.array([s.evaluate(vm, v) for v in np.linspace(v0, v1, 4001)], float)
    d1 = np.gradient(X, axis=0); d2 = np.gradient(d1, axis=0)
    kk = np.linalg.norm(np.cross(d1, d2), axis=1) / np.linalg.norm(d1, axis=1) ** 3
    rmin = 1.0 / kk[50:-50].max()
    L = sf.setback(R, phi, sf.clamp_smoothing(0.6, int(cont)), int(cont))
    ends = sorted([X[0], X[-1]], key=lambda p: p[2])
    on_rails = abs(ends[0][2]) < 1e-6 and abs(abs(ends[0][0] - 2.0) - L) < 1e-5 and \
        abs(ends[1][0] - 2.0) < 1e-6 and abs(ends[1][2] - L) < 1e-5
    check(f"surface fillet soft G{cont}: min radius {R}, rails at the soft setback {L:.4f}",
          r == {'FINISHED'} and abs(rmin - R) < 2e-4 and on_rails,
          f"min radius {rmin:.5f}, ends {np.round(ends, 5).tolist()}")
    if features.history_enabled(ctx):
        p = features.history(fil)["params"] if hasattr(features, "history") else None
    bpy.data.objects.remove(fil)

# preview sections lie on what the worker builds (planes: exact)
P0 = np.array([[2.0, 0.0, 0.0]])
X = soft_sections(P0, np.array([[-1.0, 0, 0]]), np.array([[0, 0, 1.0]]), R, 0.6, 2, k=5)[0]
L2 = sf.setback(R, phi, 0.6, 2)
check("soft preview: rails at the setback, through the span's middle",
      np.allclose(X[0], [2 - L2, 0, 0]) and np.allclose(X[-1], [2, 0, L2]) and abs(X[2][0] - (2 - X[2][2])) < 1e-12,
      np.round(X, 4).tolist())

# --- Edge Fillet, Soft: an L of two sheets (exact sections) ----------------
def make_L(tag):
    a = patch(tag + "a", (0, 140, 0), (0, 144, 0), "X", 2.0)
    b = patch(tag + "b", (2, 140, 0), (2, 144, 0), "Z", 2.0)
    bpy.ops.object.select_all(action='DESELECT'); a.select_set(True); b.select_set(True)
    ctx.view_layer.objects.active = a
    bpy.ops.nproto.join()
    o = ctx.active_object; o.name = tag; return o
def fillet_all(o, **kw):
    return bpy.ops.nproto.edge_fillet(inputs=json.dumps([[o.name, f"E{i}"] for i in store.shared_edges(o)]), **kw)
def span_length(R, smooth, cont):
    sh = sf.shape(phi, sf.clamp_smoothing(smooth, cont), cont)
    L = sh["g"] * R
    Q = sf._bern(sh["degree"], np.linspace(0, 1, 20001)) @ sf.planar_cvs(sh["spacing"], phi, L)
    return float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()), L
for cont in ("2", "3"):
    o = make_L("SL" + cont)
    r = fillet_all(o, shape="SOFT", radius=0.3, smoothing=0.7, continuity=cont)
    ln, L = span_length(0.3, 0.7, int(cont))
    want = 16.0 - 2 * 4 * L + 4 * ln
    check(f"edge fillet soft G{cont} on an L: area = faces cut back by the setback + the span",
          r == {'FINISHED'} and abs(A(o) - want) < 1e-4 and store.get_brep(o).face_count() == 3,
          f"{A(o):.5f} vs {want:.5f}")

# history: Smoothing edited in the step re-runs the fillet
ctx.scene.nproto_settings.construction_history = True
o = make_L("SLH")
r = fillet_all(o, shape="SOFT", radius=0.3, smoothing=0.4, continuity="2")
st = features.stack(o)
k = len(st) - 1
check("edge fillet soft: a history step with its smoothing", r == {'FINISHED'} and st[k]["op"] == "edge_fillet"
      and st[k]["params"].get("shape") == "SOFT" and abs(st[k]["params"].get("smoothing", 0) - 0.4) < 1e-6)
a0 = A(o)
history_ui.sync_step(o, k)
o.nproto_step.smoothing = 0.9
ln, L = span_length(0.3, 0.9, 2)
want = 16.0 - 2 * 4 * L + 4 * ln
check("edge fillet soft: editing the step's Smoothing rebuilds it", abs(A(o) - want) < 1e-4 and abs(A(o) - a0) > 1e-3,
      f"{a0:.5f} -> {A(o):.5f} (want {want:.5f}) {o.nproto_step.error}")

# --- Edge Fillet, Soft: a box (kernel rails, strips rebuilt) ---------------
ctx.scene.nproto_settings.construction_history = False
rc = poly("BoxR", [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], cyclic=True)
bpy.ops.nproto.extrude(inputs=ins((rc, "S0")), distance=2.0, direction="Z", body="NEW")
box = ctx.active_object
v_box = abs(store.get_brep(box).volume(1e-6))
r = fillet_all(box, shape="SOFT", radius=0.2, smoothing=0.6, continuity="2")
b = store.get_brep(box)
n_open = sum(1 for e in b.edges() if len(e.faces()) == 1)
solid = bool(b.is_manifold_solid())
v = abs(b.volume(1e-6)) if solid else 0.0
# volume removed: the section's cut-off area along the 36 m of edges, less at
# the 8 corners, where each strip stops up to L short of the vertex
sh = sf.shape(phi, 0.6, 2)
Lb = sh["g"] * 0.2
Q = sf._bern(5, np.linspace(0, 1, 4001)) @ sf.planar_cvs(sh["spacing"], phi, Lb)
poly_ = np.vstack([Q, [[Lb, 0.0]]])
cut = 0.5 * abs(np.sum(poly_[:, 0] * np.roll(poly_[:, 1], -1) - np.roll(poly_[:, 0], -1) * poly_[:, 1]))
hi_, lo_ = 36 * cut, (36 - 8 * 3 * 2 * Lb) * cut
check("edge fillet soft on a box: 12 soft strips, the 8 corners filled with G2 Squares, a closed solid",
      r == {'FINISHED'} and b.face_count() == 26 and n_open == 0 and solid and lo_ < v_box - v < hi_,
      f"faces {b.face_count()}, open edges {n_open}, solid {solid}, removed {v_box - v:.4f} "
      f"(between {lo_:.4f} and {hi_:.4f})")
# G3 strips: the corner Squares must not fold (Match Neighbours at degree 7 does; the
# fill then takes the Square's own degree-5 layout)
from blender_nurbs import steps as _steps
rc3 = poly("BoxR3", [(20, 0, 0), (24, 0, 0), (24, 3, 0), (20, 3, 0)], cyclic=True)
bpy.ops.nproto.extrude(inputs=ins((rc3, "S0")), distance=2.0, direction="Z", body="NEW")
box3 = ctx.active_object
r = fillet_all(box3, shape="SOFT", radius=0.25, smoothing=0.8, continuity="3")
b3 = store.get_brep(box3)
_ext = store._k()[1]
corners = [f for f in b3.faces() if min(_ext.surface_cvs(f.surface())["degree_u"],
                                        _ext.surface_cvs(f.surface())["degree_v"]) > 1]
check("edge fillet soft G3 on a box: 8 corner patches, none folded, a closed solid",
      r == {'FINISHED'} and len(corners) == 8 and not any(_steps._folded(f.surface()) for f in corners)
      and bool(b3.is_manifold_solid()), f"corners {len(corners)}, folded "
      f"{sum(_steps._folded(f.surface()) for f in corners)}")
# without Fill Corners the openings stay
rc2 = poly("BoxR2", [(10, 0, 0), (14, 0, 0), (14, 3, 0), (10, 3, 0)], cyclic=True)
bpy.ops.nproto.extrude(inputs=ins((rc2, "S0")), distance=2.0, direction="Z", body="NEW")
box2 = ctx.active_object
r = fillet_all(box2, shape="SOFT", radius=0.2, smoothing=0.6, continuity="2", fill_corners=False)
b2 = store.get_brep(box2)
check("edge fillet soft, Fill Corners off: 8 three-sided openings left",
      r == {'FINISHED'} and b2.face_count() == 18 and sum(1 for e in b2.edges() if len(e.faces()) == 1) == 24)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
