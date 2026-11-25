"""Tangent to Surface ends with a set direction (align.py aims): Square to the
edge, an Angle from it, or Along a guide curve on the surface -- G1 - G3, no
sideways bend (the guide's own for Along Curve), in CV Curve, Align Curve,
Blend Curve, history rebuilds and Edit Mode CV moves.
    blender -b --factory-startup --python tests/test_align_aim.py"""
import json, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import align, blendedit, drawtools, features, geom, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
from blender_nurbs.kernel import load as load_kernel
ext = load_kernel().blender_ext

def nurbs(name, pts, order=4):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.order_u = order; sp.use_endpoint_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

NET = []
for i in range(4):
    for j in range(4):
        x, y = i * 1.0, j * 1.0
        NET.append((x, y, 0.5 * np.sin(1.3 * x + 0.4) * np.cos(0.9 * y) + 0.15 * x * y + 0.1 * x * x))
NET = np.array(NET)
S = store.new_object(ctx, "Bumpy", ext.surface_from_cvs(3, 3, [tuple(p) for p in NET], geom.clamped_uniform_knots(4, 3),
                                                         geom.clamped_uniform_knots(4, 3), None))

def arc_jet(d1, d2, d3, n):
    """(unit t, normal curvature, sideways kg, sideways j) of a curve's end
    derivatives (arc length; n the surface normal, s = n x t)."""
    a = np.linalg.norm(d1)
    a1 = d1 @ d2 / a
    a2 = (d2 @ d2 + d1 @ d3 - a1 * a1) / a
    X2 = (d2 - (a1 / a) * d1) / a ** 2
    X3 = (d3 - 3 * (a1 / a) * d2 + (3 * a1 * a1 / a ** 2 - a2 / a) * d1) / a ** 3
    t = d1 / a
    s = np.cross(n, t)
    return t, X2 @ n, X2 @ s, X3 @ s

def end_derivs(Q, p, e):
    """(P, C', C'', C''') at end e of the endpoint-uniform (Q, p), leading into the curve."""
    Q = np.asarray(Q, float)
    seg = Q if e == "0" else Q[::-1]
    U = geom.clamped_uniform_knots(len(seg), p)
    d1, d2, _, _ = align._derivs(seg, U, p)
    return seg[0], d1, d2, align._d3(seg, U, p)

def deg_from(axes, t):
    return float(np.degrees(np.arctan2(t @ axes[2], t @ axes[1]))) % 180.0

def contact_gap(P, d1, d2, d3, k, ref="V0"):
    _, n, Qf, Cf, _, _ = align.surface_jet(S, ref, P)
    L = np.linalg.norm(d1)
    g = [abs(n @ d1) / L]
    if k >= 2:
        g.append(abs(n @ d2 - d1 @ Qf @ d1) / L ** 2)
    if k >= 3:
        d2t = d2 - (d2 @ n) * n
        g.append(abs(n @ d3 - 3 * d1 @ Qf @ d2t - np.einsum("ijk,i,j,k->", Cf, d1, d1, d1)) / L ** 3)
    return max(g)

# a guide lying exactly on the surface: an iso line, the CVs of the net blended across
s0 = 0.37
B = np.array([(1 - s0) ** 3, 3 * s0 * (1 - s0) ** 2, 3 * s0 * s0 * (1 - s0), s0 ** 3])
best = None
for lay in ("i", "j"):
    G = np.array([sum(B[i] * NET[i * 4 + j] for i in range(4)) for j in range(4)]) if lay == "i" else \
        np.array([sum(B[j] * NET[i * 4 + j] for j in range(4)) for i in range(4)])
    gap = max(np.linalg.norm(align.surface_jet(S, "F0", q)[0] - q) for q in
              [G[0], G[-1], 0.125 * G[0] + 0.375 * G[1] + 0.375 * G[2] + 0.125 * G[3]])
    if best is None or gap < best[0]:
        best = (gap, G)
GUIDE = nurbs("Guide", [tuple(p) for p in best[1]])
GREF = min(("U0", "U1", "V0", "V1"),
           key=lambda r: np.linalg.norm(align.edge_frame(S, r, best[1][0], "NEAREST")[0] - best[1][0]))
GP = best[1][0]
check("test setup: the guide lies on the surface, from an edge", best[0] < 1e-6 and
      np.linalg.norm(align.edge_frame(S, GREF, GP, "NEAREST")[0] - GP) < 1e-6, (best[0], GREF))

# ------------------------------------------------------------- solver: align_open, exact (float64)
EDGE = "V0"
P0 = align.edge_frame(S, EDGE, np.array([1.4, -0.2, 0.0]), "NEAREST")[0]
click = [P0, P0 + np.array([0.3, -0.8, 0.4]), P0 + np.array([1.0, -1.6, 0.5]), P0 + np.array([1.6, -2.4, 0.9])]
for d, ang in (("SQUARE", 90.0), ("ANGLE", 35.0), ("ANGLE", 140.0)):
    for cont, k in (("G1", 1), ("G2", 2), ("G3", 3)):
        spec = {"cont": cont, "mode": "SURFACE", "snap": "NEAREST", "flip": False, "dir": d, "angle": ang}
        fr = align.end_frame(S, EDGE, P0, "NEAREST", "SURFACE", aim=align.spec_aim(spec))
        Q, p, specs = align.align_open(np.array(click), 3, {"0": (spec, fr)})
        P, d1, d2, d3 = end_derivs(Q, p, "0")
        axes = align.aim_axes(S, EDGE, P)
        t, kn, kg, j = arc_jet(d1, d2, d3, axes[0])
        a = deg_from(axes, t)
        side = t @ (np.mean(Q[1:], axis=0) - P) > 0
        g = contact_gap(P, d1, d2, d3, k, EDGE)
        sw = max(abs(kg) if k >= 2 else 0.0, abs(j) if k >= 3 else 0.0)
        check(f"solver {d} {ang:.0f} {cont}: on the edge, {ang:.0f}° from it, toward the curve, contact, no sideways bend",
              np.linalg.norm(P - P0) < 1e-12 and abs(a - ang) < 1e-7 and side and g < 1e-9 and sw < 1e-8,
              (a, g, sw, specs["0"].get("flip")))

# Along Curve: the guide's direction, its sideways bend (kg) and rate (j)
_, n, _, _, _, _ = align.surface_jet(S, GREF, GP)
gt, gkg, gj = align.guide_jet(GUIDE, "S0", GP, n)
click = [GP, GP + 0.8 * gt + np.array([0, 0, 0.3]), GP + 1.6 * gt + np.array([0.4, 0.4, 0.5]), GP + 2.2 * gt]
for cont, k in (("G2", 2), ("G3", 3)):
    spec = {"cont": cont, "mode": "SURFACE", "snap": "NEAREST", "flip": False, "dir": "CURVE", "guide": 1}
    fr = align.end_frame(S, GREF, GP, "NEAREST", "SURFACE", aim=align.spec_aim(spec, [None, (GUIDE, "S0")]))
    Q, p, specs = align.align_open(np.array(click), 3, {"0": (spec, fr)})
    P, d1, d2, d3 = end_derivs(Q, p, "0")
    t, kn, kg, j = arc_jet(d1, d2, d3, n)
    sg = 1.0 if t @ gt > 0 else -1.0
    g = contact_gap(P, d1, d2, d3, k, GREF)
    check(f"solver Along Curve {cont}: the guide's direction and sideways bend" + (" and rate" if k >= 3 else ""),
          abs(abs(t @ gt) - 1) < 1e-12 and g < 1e-9 and abs(kg - gkg) < 1e-8 * (1 + abs(gkg))
          and (k < 3 or abs(j - sg * gj) < 1e-7 * (1 + abs(gj))), (kg, gkg, j, sg * gj, g))
    # the guide itself is a curve on the surface: Along Curve G2 is G2 to the guide (one curve)
    if k == 2:
        kv = lambda a1, a2: np.cross(np.cross(a1, a2), a1) / (a1 @ a1) ** 2
        G = geom.curve_derivs(features.input_curve(GUIDE, "S0"), features.input_curve(GUIDE, "S0").parameter_range()[0], 2)
        check("solver Along Curve G2: the same curvature as the guide (G2 to it)",
              np.linalg.norm(kv(d1, d2) - kv(G[1], G[2])) < 1e-8 * (1 + np.linalg.norm(kv(G[1], G[2]))))

# legacy: no "dir" is Free, as before (the curve's own direction projected)
spec = {"cont": "G2", "mode": "SURFACE", "snap": "NEAREST", "flip": False}
fr = align.end_frame(S, EDGE, P0, "NEAREST", "SURFACE", aim=align.spec_aim(spec))
check("legacy spec (no dir): a Free surface frame", fr[3] is not None and align.spec_aim(spec) is None)
spec_sq_face = {"mode": "SURFACE", "dir": "SQUARE"}
check("Square on a face: Free (no edge to be square to)",
      align.end_frame(S, "F0", P0 + np.array([0.5, 0.6, 0]), "END", "SURFACE", aim=align.spec_aim(spec_sq_face))[3] is not None)

# ------------------------------------------------------------- CV Curve: execute, history, rebuild, edit
def cv_curve(**kw):
    pts = [list(map(float, P0)), [2.0, -1.0, 0.5], [2.6, -2.0, 0.7], [3.4, -2.6, 0.9]]
    data = {"pts": pts, "start": [S.name, EDGE, "NEAREST", 0], "end": None, "edit": "", "normal": None}
    data.update(kw.pop("data", {}))
    r = bpy.ops.nproto.draw_curve(data=json.dumps(data), mode="CV", degree="5", keep_aligned=True, edge_mode="SURFACE",
                                  **kw)
    return r, ctx.active_object

def start_state(o, ref=EDGE):
    kc = features.input_curve(o, "S0")
    D = geom.curve_derivs(kc, kc.parameter_range()[0], 3)
    axes = align.aim_axes(S, ref, D[0])
    t, kn, kg, j = arc_jet(D[1], D[2], D[3], axes[0])
    return D, axes, deg_from(axes, t), kg, j

for d, ang, cont in (("SQUARE", 90.0, "G2"), ("ANGLE", 60.0, "G3")):
    r, o = cv_curve(start_cont=cont, start_dir=d, start_angle=ang)
    h = features.history(o)
    e0 = h["params"]["ends"]["0"] if h else {}
    D, axes, a, kg, j = start_state(o)
    check(f"CV Curve {d} {cont}: Align history with the direction", r == {'FINISHED'} and e0.get("dir") == d and
          (d != "ANGLE" or abs(e0.get("angle", 0) - ang) < 1e-9), e0)
    check(f"CV Curve {d} {cont}: {ang:.0f}° from the edge, no sideways bend (float32 CVs)",
          abs(a - ang) < 1e-3 and abs(kg) < 1e-3 * (1 + abs(D[2] @ D[2]) ** 0.5), (a, kg, j))
    before = np.array([tuple(q.co)[:3] for q in o.data.splines[0].points])
    features.rebuild(o, ctx, force=True)
    after = np.array([tuple(q.co)[:3] for q in o.data.splines[0].points])
    check(f"CV Curve {d} {cont}: a rebuild solves to itself", np.abs(after - before).max() < 1e-5, np.abs(after - before).max())
    # Edit Mode CV move off the aim: constrain_align puts the end back on it
    sp = o.data.splines[0]
    q = sp.points[1]
    q.co = (q.co[0] + 0.15, q.co[1] + 0.2, q.co[2] - 0.1, 1.0)
    q.select = True
    blendedit.constrain_align(o)
    D, axes, a2, kg2, _ = start_state(o)
    check(f"CV Curve {d} {cont}: an Edit Mode CV move keeps the direction", abs(a2 - ang) < 1e-3 and abs(kg2) < 1e-2,
          (a2, kg2))
    # the surface moves: the end follows, still at the angle
    S.location = (0.0, 0.0, 0.3); ctx.view_layer.update()
    features.rebuild(o, ctx, force=True)
    D, axes, a3, kg3, _ = start_state(o)
    gap = np.linalg.norm(D[0] - align.edge_frame(S, EDGE, D[0], "NEAREST")[0])
    check(f"CV Curve {d} {cont}: follows the moved surface", gap < 1e-5 and abs(a3 - ang) < 1e-3, (gap, a3))
    S.location = (0.0, 0.0, 0.0); ctx.view_layer.update()
    bpy.data.objects.remove(o)

r, o = cv_curve(start_cont="G3", start_dir="CURVE",
                data={"pts": [list(map(float, GP)), list(map(float, GP + 0.8 * gt + np.array([0, 0, 0.3]))),
                              list(map(float, GP + 1.6 * gt)), list(map(float, GP + 2.2 * gt))],
                      "start": [S.name, GREF, "NEAREST", 0], "start_guide": [GUIDE.name, "S0"]})
h = features.history(o)
e0 = h["params"]["ends"]["0"]
check("CV Curve Along Curve: the guide is a history input", e0.get("dir") == "CURVE" and
      h["inputs"][e0["guide"]][0] == GUIDE, [(a.name, b) for a, b in h["inputs"]])
D, axes, a, kg, j = start_state(o, GREF)
check("CV Curve Along Curve G3: the guide's direction and sideways bend (float32 CVs)",
      abs(abs((D[1] / np.linalg.norm(D[1])) @ gt) - 1) < 1e-6 and abs(kg - gkg) < 2e-3 * (1 + abs(gkg)), (kg, gkg))
bpy.data.objects.remove(o)
r, o = cv_curve(start_cont="G2", start_dir="CURVE", data={"start_guide": ["Nope", "S0"]})
check("CV Curve Along Curve with a missing guide: Square", features.history(o)["params"]["ends"]["0"].get("dir") == "SQUARE")
bpy.data.objects.remove(o)

# ------------------------------------------------------------- Align Curve operator
C = nurbs("AlignMe", [(1.5, -0.3, 0.2), (1.9, -0.9, 0.5), (2.2, -1.6, 0.6), (2.8, -2.2, 0.9), (3.3, -2.9, 1.0)])
for dirn, ang in (("ANGLE", 50.0), ("SQUARE", 90.0)):
    r = bpy.ops.nproto.align_curve('EXEC_DEFAULT', inputs=json.dumps([[C.name, "S0"], [S.name, EDGE]]), end=0,
                                   cont="G2", mode="SURFACE", snap="NEAREST", direction=dirn, angle=ang)
    D, axes, a, kg, _ = start_state(C)
    e0 = features.history(C)["params"]["ends"]["0"]
    check(f"Align Curve {dirn}: {ang:.0f}° from the edge, toward the curve, no sideways bend",
          r == {'FINISHED'} and e0.get("dir") == dirn and abs(a - ang) < 1e-3 and abs(kg) < 1e-3
          and D[1] @ (np.mean(np.array([tuple(q.co)[:3] for q in C.data.splines[0].points]), axis=0) - D[0]) > 0, (a, kg, e0))
r = bpy.ops.nproto.align_curve('EXEC_DEFAULT', inputs=json.dumps([[C.name, "S0"], [S.name, EDGE]]), end=0,
                               cont="G2", mode="SURFACE", snap="NEAREST", direction="FREE")
check("Align Curve Free: no direction in the history (as before)", "dir" not in features.history(C)["params"]["ends"]["0"])
bpy.data.objects.remove(C)

# ------------------------------------------------------------- Blend Curve: angle on an edge
E2 = nurbs("Other", [(1.0, -4.0, 1.5), (2.0, -4.2, 1.6), (3.0, -4.0, 1.4), (4.0, -4.5, 1.5)])
base = {"end_a": 0, "end_b": 0, "cont_a": "G2", "cont_b": "G2", "w_a": 1.0, "w_b": 1.0, "c_a": 0.0, "c_b": 0.0,
        "d_a": 0.0, "d_b": 0.0, "flip_a": False, "flip_b": False, "bc_mode_a": "SURFACE", "bc_on_a": True, "bc_at_a": 0.45}
for ang in (None, 60.0):
    prm = dict(base, **({} if ang is None else {"bc_ang_a": ang}))
    fa, fb = features.blend_frames([(S, EDGE), (E2, "S0")], prm)
    axes = align.aim_axes(S, EDGE, fa[0])
    a = deg_from(axes, fa[1] / np.linalg.norm(fa[1]))
    want = 90.0 if ang is None else ang
    a_line = abs(a - want)                  # Blend and the aims measure alike: from the edge toward off the face
    check(f"Blend Curve side on an edge: {want:.0f}° from it", a_line < 1e-7 and abs(fa[1] @ axes[0]) < 1e-12, a)

print("ALL PASS" if ok else "SOME FAILED")
sys.stdout.flush()
os._exit(0 if ok else 1)
