"""Blend Curve and Align Curve: CV edits in Edit Mode keep G1 / G2 / G3 at the ends:
    blender -b --factory-startup --python tests/test_blend_edit.py"""
import json, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import blendedit, features, geom

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context

def nurbs(name, pts, loc=(0, 0, 0)):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.order_u = 4; sp.use_endpoint_u = True
    o = bpy.data.objects.new(name, cu); o.location = loc; ctx.collection.objects.link(o); return o

def select(objs, active):
    for o in ctx.view_layer.objects: o.select_set(False)
    for o in objs: o.select_set(True)
    ctx.view_layer.objects.active = active

def frame(o, end, flip=False):
    c = features.input_curve(o, "S0")
    t0, t1 = c.parameter_range()
    D = geom.curve_derivs(c, t1 if end else t0, 3)
    s = 1.0 if end else -1.0
    return [D[0], s * D[1], D[2], s * D[3]]

def blend_derivs(o, at_end):
    c = geom.spline_to_kernel_curve(o.data.splines[0], geom.world_matrix(o))
    t0, t1 = c.parameter_range()
    D = geom.curve_derivs(c, t1 if at_end else t0, 3)
    s = -1.0 if at_end else 1.0          # into the blend
    return [D[0], s * D[1], D[2], s * D[3]]

def gaps(C, B):
    """G0..G3 defects between input frame C (outward) and blend frame B (into the blend):
    B' = l C', B'' = l^2 C'' + m C', B''' = l^3 C''' + 3 l m C'' + v C' for some l > 0, m, v."""
    T = C[1] / np.linalg.norm(C[1])
    perp = lambda v: v - (v @ T) * T
    l = (B[1] @ T) / np.linalg.norm(C[1])
    g1 = np.linalg.norm(perp(B[1])) / np.linalg.norm(B[1])
    R2 = B[2] - l * l * C[2]
    m = (R2 @ T) / np.linalg.norm(C[1])
    g2 = np.linalg.norm(perp(R2)) / (np.linalg.norm(B[2]) + 1e-12)
    R3 = B[3] - l ** 3 * C[3] - 3 * l * m * C[2]
    g3 = np.linalg.norm(perp(R3)) / (np.linalg.norm(B[3]) + 1e-12)
    return np.linalg.norm(B[0] - C[0]), (g1 if l > 0 else 1.0), g2, g3

def world_cvs(o):
    return blendedit._world_cvs(o, o.data.splines[0])

def set_cv(o, i, P):
    M = np.linalg.inv(np.array(geom.world_matrix(o), float))
    q = M[:3, :3] @ np.asarray(P, float) + M[:3, 3]
    o.data.splines[0].points[i].co = (*q, 1.0)
    ctx.view_layer.update()              # depsgraph_update_post: blendedit puts the CVs back on the continuity

def check_ends(name, blend, a, b, ka, kb, tol=2e-5):   # CVs are float32: ~1e-5 at G3
    ga = gaps(frame(a, 1), blend_derivs(blend, False))
    gb = gaps(frame(b, 0), blend_derivs(blend, True))
    good = ga[0] < tol and gb[0] < tol and all(g < tol for g in ga[1:1 + ka]) and all(g < tol for g in gb[1:1 + kb])
    check(name, good, f"A {['%.1e' % g for g in ga[:ka + 1]]} B {['%.1e' % g for g in gb[:kb + 1]]}")

def params(o):
    return features.history(o)["params"]

# two curved, non-planar input curves
A = nurbs("A", [(-6, 0, 0), (-4.5, 1.2, 0.3), (-3, 0.8, -0.2), (-2, 2.0, 0.4), (-1, 1.0, 0.1)])
B = nurbs("B", [(1.5, 1.5, 0.0), (2.5, 0.2, 0.5), (3.5, 1.4, -0.3), (5, 0.4, 0.2), (6, 1.0, 0.0)])

def make_blend(ca, cb):
    select([A, B], B)
    r = bpy.ops.nproto.blend_curve('EXEC_DEFAULT', cont_a=ca, cont_b=cb)
    assert r == {'FINISHED'}, r
    return ctx.view_layer.objects.active

for ca, cb in (("G3", "G3"), ("G2", "G1"), ("G1", "G3"), ("G0", "G2")):
    ka, kb = geom.CONT_ORDER[ca], geom.CONT_ORDER[cb]
    n = 1 + ka + kb
    bl = make_blend(ca, cb)
    tag = f"{ca}/{cb}"
    check_ends(f"{tag} built", bl, A, B, ka, kb)
    select([bl], bl)
    bpy.ops.object.mode_set(mode='EDIT')
    W0 = world_cvs(bl)
    L = np.linalg.norm(W0[-1] - W0[0])
    rng = np.random.default_rng(7)
    for i in range(1, n):                 # every inner CV, pulled somewhere off its constraint
        before = world_cvs(bl)
        target = before[i] + rng.normal(size=3) * 0.25 * L / n
        set_cv(bl, i, target)
        after = world_cvs(bl)
        check_ends(f"{tag} CV {i} moved", bl, A, B, ka, kb)
        check(f"{tag} CV {i}: end points stay", np.abs(after[[0, -1]] - W0[[0, -1]]).max() < 1e-9)
        # the moved CV went toward the target (as far as the continuity lets it)
        check(f"{tag} CV {i}: nearer the target", np.linalg.norm(after[i] - target) <= np.linalg.norm(before[i] - target) + 1e-9)
        # the CVs equal what the stored settings build: the edit is part of the feature
        g = features.run("blend_curve", features.history(bl)["inputs"], params(bl))
        check(f"{tag} CV {i}: settings rebuild it", np.abs(np.array(g[2]) - after).max() < 2e-6)
    # an end point dragged snaps back onto its input
    if n >= 2:
        set_cv(bl, 0, W0[0] + np.array([0.3, 0.2, -0.1]))
        check(f"{tag} end point snaps back", np.abs(world_cvs(bl)[0] - W0[0]).max() < 1e-9)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects.remove(bl)

# G1 side: P1 moved sideways lands on the tangent at the target's tangent distance
bl = make_blend("G1", "G1")
select([bl], bl); bpy.ops.object.mode_set(mode='EDIT')
W = world_cvs(bl)
Ca = frame(A, 1)
T = Ca[1] / np.linalg.norm(Ca[1])
side = np.cross(T, [0, 0, 1.0]); side /= np.linalg.norm(side)
set_cv(bl, 1, W[1] + 0.7 * T + 0.5 * side)
W1 = world_cvs(bl)
check("G1 P1: on the tangent, tangent part kept", np.linalg.norm(W1[1] - (W[1] + 0.7 * T)) < 1e-7,
      f"{np.linalg.norm(W1[1] - (W[1] + 0.7 * T)):.2e}")
bpy.ops.object.mode_set(mode='OBJECT'); bpy.data.objects.remove(bl)

# G2 side: P2 lifted along its own height -> the weight grows (P1 slides out), G2 kept
bl = make_blend("G2", "G2")
w0 = params(bl)["w_a"]
select([bl], bl); bpy.ops.object.mode_set(mode='EDIT')
W = world_cvs(bl)
h = (W[2] - W[0]) - ((W[2] - W[0]) @ T) * T          # P2's height off A's tangent line
set_cv(bl, 2, W[2] + 0.8 * h)
W1 = world_cvs(bl)
check("G2 P2 lifted: weight grows", params(bl)["w_a"] > w0 * 1.2, f"{w0:.3f} -> {params(bl)['w_a']:.3f}")
check("G2 P2 lifted: reaches the target", np.linalg.norm(W1[2] - (W[2] + 0.8 * h)) < 1e-5 * np.linalg.norm(h),
      f"{np.linalg.norm(W1[2] - (W[2] + 0.8 * h)):.2e}")
check_ends("G2 P2 lifted", bl, A, B, 2, 2)
# a Feature-panel edit while in Edit Mode moves the CVs in place (point selection kept)
sp = bl.data.splines[0]
for q in sp.points: q.select = False
sp.points[1].select = True
bl.nproto_feature.w_b = 0.6
check("panel edit in Edit Mode", abs(params(bl)["w_b"] - 0.6) < 1e-6 and sp.points[1].select
      and len(sp.points) == 6, (params(bl)["w_b"], sp.points[1].select, len(sp.points)))
check("panel edit moved the CVs", np.linalg.norm(world_cvs(bl)[4] - W1[4]) > 1e-3)
check_ends("panel edit in Edit Mode", bl, A, B, 2, 2)
# continuity changed in Edit Mode: a new spline (more CVs) and still exact
bl.nproto_feature.cont_b = "G3"
ctx.view_layer.update()
check("cont change in Edit Mode", len(bl.data.splines[0].points) == 7, len(bl.data.splines[0].points))
check_ends("cont change in Edit Mode", bl, A, B, 2, 3)
W = world_cvs(bl)
set_cv(bl, 4, W[4] + np.array([0.1, -0.3, 0.2]))
check_ends("cont change, then CV moved", bl, A, B, 2, 3)
bpy.ops.object.mode_set(mode='OBJECT')
edited = dict(params(bl))

# the edit survives an input change: the rebuild uses the fitted settings
A.location.y += 0.5
ctx.view_layer.update()
features.rebuild(bl, ctx)
p2 = params(bl)
check("input moved: settings kept", all(abs(p2[k] - edited[k]) < 1e-12 for k in ("w_a", "c_a", "w_b", "c_b", "d_b")))
check_ends("input moved: rebuilt exact", bl, A, B, 2, 3)

# CVs added in Edit Mode: no longer a blend, left alone
select([bl], bl); bpy.ops.object.mode_set(mode='EDIT')
bl.data.splines[0].points.add(1)
bl.data.splines[0].points[-1].co = (9, 9, 9, 1)
ctx.view_layer.update()
check("points added: left as edited", tuple(bl.data.splines[0].points[-1].co)[:3] == (9, 9, 9))
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------ Align Curve
from blender_nurbs import align

def edge_frame_out(o, end):
    c = features.input_curve(o, "S0")
    t0, t1 = c.parameter_range()
    D = geom.curve_derivs(c, t1 if end else t0, 3)
    s = 1.0 if end else -1.0
    return [D[0], s * D[1], D[2], s * D[3]]

def local_cvs(o):
    return np.array([tuple(q.co)[:3] for q in o.data.splines[0].points])

def select_only(o, i):
    for j, q in enumerate(o.data.splines[0].points): q.select = (j == i)

E0 = nurbs("E0", [(-6, 0, 0), (-4.5, 1.2, 0.3), (-3, 0.8, -0.2), (-2, 2.0, 0.4), (-1, 1.0, 0.1)])   # its end at x=-1
E1 = nurbs("E1", [(5, 1.0, 0.0), (6, 2.0, 0.3), (7, 0.5, 0.0), (8, 1.5, -0.2)])                    # its start at x=5
def drawn(name):
    return nurbs(name, [(-0.7, 1.4, 0), (0, 2.2, 0.2), (1, 1, 0), (2, 2.5, -0.1), (3, 1.4, 0), (4.6, 1.2, 0.1)])

# Weight applies once: a rebuild no longer scales the tangent again
C = drawn("AW")
r = bpy.ops.nproto.align_curve('EXEC_DEFAULT', inputs=json.dumps([[C.name, "S0"], [E0.name, "S0"]]), end=0, cont="G2", weight=1.5)
lens = []
for _ in range(4):
    Q = local_cvs(C); lens.append(np.linalg.norm(Q[1] - Q[0]))
    E0.location.z += 0.01; ctx.view_layer.update(); features.rebuild(C, ctx)
check("align weight: rebuilds keep the tangent length", r == {'FINISHED'} and max(lens) - min(lens) < 1e-3 * lens[0],
      ["%.4f" % x for x in lens])
E0.location.z = 0.0; ctx.view_layer.update(); features.rebuild(C, ctx)
Q = local_cvs(C); features.rebuild(C, ctx, force=True)
check("align: rebuild of an aligned curve gives it back", np.abs(local_cvs(C) - Q).max() < 1e-5, np.abs(local_cvs(C) - Q).max())
bpy.data.objects.remove(C)

def align_gaps(o):
    ga = gaps(edge_frame_out(E0, 1), blend_derivs(o, False))
    gb = gaps(edge_frame_out(E1, 0), blend_derivs(o, True))
    return ga, gb

# EDGE: G2 at the start, G1 at the end; every CV dragged (selected) somewhere
C = drawn("AE")
bpy.ops.nproto.align_curve('EXEC_DEFAULT', inputs=json.dumps([[C.name, "S0"], [E0.name, "S0"]]), end=0, cont="G2")
bpy.ops.nproto.align_curve('EXEC_DEFAULT', inputs=json.dumps([[C.name, "S0"], [E1.name, "S0"]]), end=1, cont="G1")
ga, gb = align_gaps(C)
check("align EDGE built", max(ga[:3]) < 2e-5 and max(gb[:2]) < 2e-5, (ga[:3], gb[:2]))
select([C], C); bpy.ops.object.mode_set(mode='EDIT')
N = len(C.data.splines[0].points)
rng = np.random.default_rng(3)
for i in range(N):
    before = world_cvs(C)
    target = before[i] + rng.normal(size=3) * 0.3
    select_only(C, i)
    set_cv(C, i, target)
    after = world_cvs(C)
    ga, gb = align_gaps(C)
    check(f"align EDGE CV {i} moved: G2 / G1 hold", max(ga[:3]) < 2e-5 and max(gb[:2]) < 2e-5,
          f"A {['%.1e' % g for g in ga[:3]]} B {['%.1e' % g for g in gb[:2]]}")
    free = 3 <= i <= N - 3
    if free:
        check(f"align EDGE CV {i} (inner): free", np.linalg.norm(after[i] - target) < 1e-5)
    elif i not in (0, N - 1):
        check(f"align EDGE CV {i}: nearer the target", np.linalg.norm(after[i] - target) < np.linalg.norm(before[i] - target) + 1e-9)
    others = [j for j in range(N) if j != i and not ({i, j} <= {0, 1, 2} or {i, j} <= {N - 1, N - 2})]
    check(f"align EDGE CV {i}: CVs of the other end and the inside stay", np.abs(after[others] - before[others]).max() < 1e-5)
# lifting Q2 off the tangent slides Q1 out (its height comes from the tangent length)
W = world_cvs(C)
Ca = edge_frame_out(E0, 1); T = Ca[1] / np.linalg.norm(Ca[1])
h = (W[2] - W[0]) - ((W[2] - W[0]) @ T) * T
select_only(C, 2); set_cv(C, 2, W[2] + 0.5 * h)
W1 = world_cvs(C)
check("align EDGE Q2 lifted: Q1 slides out, Q2 reaches", np.linalg.norm(W1[1] - W1[0]) > np.linalg.norm(W[1] - W[0]) * 1.05
      and np.linalg.norm(W1[2] - (W[2] + 0.5 * h)) < 1e-4 * np.linalg.norm(h), np.linalg.norm(W1[2] - (W[2] + 0.5 * h)))
Wedit = world_cvs(C)
bpy.ops.object.mode_set(mode='OBJECT')
features.rebuild(C, ctx, force=True)
check("align EDGE: leaving Edit Mode keeps the edit", np.abs(world_cvs(C) - Wedit).max() < 1e-5, np.abs(world_cvs(C) - Wedit).max())
bpy.data.objects.remove(C)

# NEAREST: the end CV slides along its edge
C = drawn("AN")
bpy.ops.nproto.align_curve('EXEC_DEFAULT', inputs=json.dumps([[C.name, "S0"], [E0.name, "S0"]]), end=0, cont="G2", snap="NEAREST")
select([C], C); bpy.ops.object.mode_set(mode='EDIT')
kc = features.input_curve(E0, "S0"); t0, t1 = kc.parameter_range()
on_edge = np.array(kc.evaluate(t0 + 0.8 * (t1 - t0)))
select_only(C, 0); set_cv(C, 0, on_edge + np.array([0.0, 0.0, 0.3]))
W = world_cvs(C)
c = features.input_curve(C, "S0")
ts = np.linspace(t0, t1, 4001); dist = min(np.linalg.norm(np.array(kc.evaluate(t)) - W[0]) for t in ts)
check("align NEAREST: end slides along the edge", dist < 1e-3 and np.linalg.norm(W[0] - on_edge) < 0.2, (dist, np.linalg.norm(W[0] - on_edge)))
m = align.measure(C, 0, E0, "S0", "NEAREST")
check("align NEAREST: still G2 there", m[1] < 2e-3 and m[2] < 1e-3 * max(1.0, 1.0), m)
bpy.ops.object.mode_set(mode='OBJECT')
bpy.data.objects.remove(C)

# SURFACE: tangent to an extruded arc's surface, G2 (normal curvature)
bpy.ops.nproto.add_primitive(kind="ARC", radius=2.0, angle=90.0, degree="0"); arc = ctx.active_object; arc.location = (0, 0, -30)
bpy.ops.nproto.extrude(inputs=json.dumps([[arc.name, "S0"]]), distance=1.5, direction="Z"); surf = ctx.active_object
from blender_nurbs import store
ss = store.get_surfaces(surf)[0]
lo = min((w for w in ("U0", "U1", "V0", "V1") if np.ptp([p[2] for p in geom.boundary_points(ss, w, 8)]) < 1e-6),
         key=lambda w: geom.boundary_points(ss, w, 2)[0][2])
C = nurbs("AS", [(0.5, 2.5, -30.2), (0.3, 2.0, -29.5), (0.6, 1.5, -29.0), (1.2, 1.0, -28.6), (1.6, 0.8, -28.2), (2.0, 0.5, -28.0)])
r = bpy.ops.nproto.align_curve('EXEC_DEFAULT', inputs=json.dumps([[C.name, "S0"], [surf.name, lo]]), end=0, cont="G2", mode="SURFACE")
def surf_gaps(o):
    c = features.input_curve(o, "S0"); t0_, _ = c.parameter_range()
    D = geom.curve_derivs(c, t0_, 2)
    n, k_n = align.surface_frame(surf, lo, D[0])
    t = D[1] / np.linalg.norm(D[1])
    kvec = (D[2] - (D[2] @ t) * t) / (D[1] @ D[1])
    return abs(float(n @ t)), abs(float(kvec @ n) - k_n(t))
g = surf_gaps(C)
check("align SURFACE built", r == {'FINISHED'} and g[0] < 1e-5 and g[1] < 1e-4, g)
select([C], C); bpy.ops.object.mode_set(mode='EDIT')
for i in (1, 2):
    W = world_cvs(C)
    select_only(C, i); set_cv(C, i, W[i] + rng.normal(size=3) * 0.3)
    g = surf_gaps(C)
    check(f"align SURFACE CV {i} moved: tangent plane and normal curvature hold", g[0] < 1e-5 and g[1] < 1e-3, g)
bpy.ops.object.mode_set(mode='OBJECT')

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
