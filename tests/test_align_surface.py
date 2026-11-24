"""Align Curve, Blend Curve and CV Curve on sketches, surface edges and surfaces:
G0 - G3 continuing an edge, tangent to a surface and perpendicular to it.
    blender -b --factory-startup --python tests/test_align_surface.py [-- "user file.blend"]"""
import json, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import align, drawtools, features, geom, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
sm = blender_nurbs.kernel.load() if hasattr(blender_nurbs, "kernel") else None
from blender_nurbs.kernel import load as load_kernel
ext = load_kernel().blender_ext

def nurbs(name, pts, loc=(0, 0, 0), order=4, cyclic=False, splines=None):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    for P in (splines or [pts]):
        sp = cu.splines.new('NURBS'); sp.points.add(len(P) - 1)
        for p, c in zip(sp.points, P): p.co = (*c, 1.0)
        sp.order_u = order; sp.use_endpoint_u = not cyclic; sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); o.location = loc; ctx.collection.objects.link(o); return o

def bumpy(name, off):
    """A bicubic patch, curved both ways (its third-order form is not zero)."""
    pts = []
    for i in range(4):
        for j in range(4):
            x, y = i * 1.0, j * 1.0
            pts.append((x + off[0], y + off[1], off[2] + 0.5 * np.sin(1.3 * x + 0.4) * np.cos(0.9 * y) + 0.15 * x * y))
    s = ext.surface_from_cvs(3, 3, pts, geom.clamped_uniform_knots(4, 3), geom.clamped_uniform_knots(4, 3), None)
    return store.new_object(ctx, name, s)

def start_jet(o, end, ref="S0"):
    """(C, C', C'', C''') at the curve's end `end`, derivatives leading INTO the curve."""
    c = features.input_curve(o, ref)
    t0, t1 = c.parameter_range()
    D = geom.curve_derivs(c, t1 if end else t0, 3)
    s = -1.0 if end else 1.0
    return D[0], s * D[1], D[2], s * D[3]

def reparam_gap(C1, C2, C3, E1, E2, E3, k):
    """How far (C', C'', C''') is from a reparameterisation of (E', E'', E''')
    up to order k (relative, per order): the G<k> gap."""
    e = E1 / np.linalg.norm(E1)
    lam = (C1 @ E1) / (E1 @ E1)
    out = [np.linalg.norm(C1 - lam * E1) / np.linalg.norm(C1), float(lam > 0) - 1.0]
    if k >= 2:
        r2 = C2 - lam * lam * E2
        mu = (r2 @ E1) / (E1 @ E1)
        out.append(np.linalg.norm(r2 - mu * E1) / (np.linalg.norm(C1) ** 2 + 1e-30))
        if k >= 3:
            r3 = C3 - lam ** 3 * E3 - 3 * lam * mu * E2
            nu = (r3 @ E1) / (E1 @ E1)
            out.append(np.linalg.norm(r3 - nu * E1) / (np.linalg.norm(C1) ** 3 + 1e-30))
    return max(abs(x) for x in out)

def surface_gap(C1, C2, C3, P, obj, ref, k, face=0):
    """Tangent-to-surface gaps: n.C' (G1), n.C'' - Q(C',C') (G2), n.C''' - 3Q(C',C''_t) - C(C',C',C') (G3)."""
    _, n, Q, Cf, _, _ = align.surface_jet(obj, ref, P, face)
    L = np.linalg.norm(C1)
    g = [abs(n @ C1) / L]
    if k >= 2:
        g.append(abs(n @ C2 - C1 @ Q @ C1) / L ** 2)
    if k >= 3:
        C2t = C2 - (C2 @ n) * n
        g.append(abs(n @ C3 - 3 * C1 @ Q @ C2t - np.einsum("ijk,i,j,k->", Cf, C1, C1, C1)) / L ** 3)
    return max(g)

def run_align(curve, target, ref, end, cont, mode="EDGE", snap="END", sref="S0"):
    return bpy.ops.nproto.align_curve('EXEC_DEFAULT', inputs=json.dumps([[curve.name, sref], [target.name, ref]]),
                                      end=end, cont=cont, mode=mode, snap=snap)

S = bumpy("Bumpy", (0, 0, 0))
z = np.zeros(3)

# ------------------------------------------------------------- Align: perpendicular to a face
for cont, k in (("G1", 1), ("G2", 2), ("G3", 3)):
    C = nurbs("Perp" + cont, [(1.4, 1.6, 0.9), (1.6, 1.7, 1.6), (2.0, 1.3, 2.2), (2.6, 1.2, 2.6), (3.0, 1.6, 3.2), (3.4, 1.4, 3.6)])
    r = run_align(C, S, "F0", 0, cont, "NORMAL")
    P, C1, C2, C3 = start_jet(C, 0)
    _, n, _, _, _, _ = align.surface_jet(S, "F0", P)
    foot = align.surface_jet(S, "F0", P)[0]
    side = n if n @ (np.mean(np.array([tuple(q.co)[:3] for q in C.data.splines[0].points]), axis=0) - P) > 0 else -n
    gap = reparam_gap(C1, C2, C3, side, z, z, k)
    check(f"align {cont} perpendicular to a face", r == {'FINISHED'} and np.linalg.norm(P - foot) < 1e-6 and gap < 1e-5,
          (np.linalg.norm(P - foot), gap))
    if k == 3:
        check("align G3: degree raised to 4", C.data.splines[0].order_u - 1 >= 4, C.data.splines[0].order_u - 1)
        # history: the surface moves, the end follows (still on it, still square to it)
        S.location = (0.0, 0.0, 0.4); ctx.view_layer.update()
        features.rebuild(C, ctx, force=True)
        P, C1, C2, C3 = start_jet(C, 0)
        foot = align.surface_jet(S, "F0", P)[0]
        _, n, _, _, _, _ = align.surface_jet(S, "F0", P)
        gap = reparam_gap(C1, C2, C3, n if n @ C1 > 0 else -n, z, z, 3)
        check("align perpendicular: rebuild follows the moved surface", np.linalg.norm(P - foot) < 1e-6 and gap < 1e-5,
              (np.linalg.norm(P - foot), gap))
        before = np.array([tuple(q.co)[:3] for q in C.data.splines[0].points])
        features.rebuild(C, ctx, force=True)
        after = np.array([tuple(q.co)[:3] for q in C.data.splines[0].points])
        check("align perpendicular: a rebuild solves to itself", np.abs(after - before).max() < 1e-6, np.abs(after - before).max())
        S.location = (0.0, 0.0, 0.0); ctx.view_layer.update()
    bpy.data.objects.remove(C)

# ------------------------------------------------------------- Align: perpendicular on an edge, flip
C = nurbs("PerpEdge", [(-0.5, 1.5, 0.3), (-0.9, 1.6, 0.8), (-1.3, 1.4, 1.5), (-1.8, 1.2, 1.9), (-2.4, 1.0, 2.2)])
r = run_align(C, S, "U0", 0, "G2", "NORMAL", "NEAREST")
P, C1, C2, C3 = start_jet(C, 0)
_, n, _, _, _, _ = align.surface_jet(S, "U0", P)
d_edge = np.linalg.norm(P - np.array(align.edge_frame(S, "U0", P, "NEAREST")[0]))
check("align G2 perpendicular on a surface edge (nearest point)", r == {'FINISHED'} and d_edge < 1e-6 and
      reparam_gap(C1, C2, C3, n if n @ C1 > 0 else -n, z, z, 2) < 1e-5, (d_edge, reparam_gap(C1, C2, C3, n if n @ C1 > 0 else -n, z, z, 2)))
s0 = np.sign(n @ C1)
bpy.ops.nproto.align_curve('EXEC_DEFAULT', inputs=json.dumps([[C.name, "S0"], [S.name, "U0"]]), end=0, cont="G2",
                           mode="NORMAL", snap="NEAREST", flip=True)
P, C1, C2, C3 = start_jet(C, 0)
check("align perpendicular: Flip leaves on the other side", np.sign(n @ C1) == -s0)
bpy.data.objects.remove(C)

# ------------------------------------------------------------- Align: tangent to a surface, G3
C = nurbs("Tan3", [(1.5, 0.2, 0.0), (1.6, -0.6, 0.2), (1.4, -1.3, 0.5), (1.9, -2.0, 0.6), (2.2, -2.6, 0.9), (2.4, -3.2, 1.0)])
r = run_align(C, S, "V0", 0, "G3", "SURFACE", "NEAREST")
P, C1, C2, C3 = start_jet(C, 0)
g = surface_gap(C1, C2, C3, P, S, "V0", 3)
check("align G3 tangent to a doubly curved surface (order-3 contact)", r == {'FINISHED'} and g < 1e-5, g)

# ------------------------------------------------------------- Align: continue a sketch line / curve, G3
E = nurbs("E", [(-6, 0, 0), (-4.5, 1.2, 0.3), (-3, 0.8, -0.2), (-2, 2.0, 0.4), (-1, 1.0, 0.1)], loc=(0, 0, 10))
C = nurbs("Cont3", [(-0.7, 1.4, 10), (0, 2.2, 10.2), (1, 1, 10), (2, 2.5, 9.9), (3, 1.4, 10), (4.6, 1.2, 10.1)])
r = run_align(C, E, "S0", 0, "G3")
P, C1, C2, C3 = start_jet(C, 0)
Pe, De, DDe, *_ = align.end_frame(E, "S0", P)
D3e = align.frame_d3(align.end_frame(E, "S0", P))
g = reparam_gap(C1, C2, C3, De, DDe, D3e, 3)
# (single-precision CVs: the third derivative magnifies their rounding)
check("align G3 continuing a curve", r == {'FINISHED'} and np.linalg.norm(P - Pe) < 1e-9 and g < 1e-4, g)
before = np.array([tuple(q.co)[:3] for q in C.data.splines[0].points])
features.rebuild(C, ctx, force=True)
after = np.array([tuple(q.co)[:3] for q in C.data.splines[0].points])
check("align G3: a rebuild solves to itself", np.abs(after - before).max() < 1e-6, np.abs(after - before).max())

# a Bezier curve asked for G3 is refitted (a whole cubic span would be fixed)
cu = bpy.data.curves.new("BZ", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('BEZIER'); sp.bezier_points.add(1)
for b, co in zip(sp.bezier_points, [(-0.8, 1.2, 10.0), (3.0, 2.0, 10.5)]):
    b.co = co; b.handle_left_type = b.handle_right_type = 'AUTO'
BZ = bpy.data.objects.new("BZ", cu); ctx.collection.objects.link(BZ)
r = run_align(BZ, E, "S0", 0, "G3")
P, C1, C2, C3 = start_jet(BZ, 0)
check("align G3 on a Bezier curve: refitted, G3", r == {'FINISHED'} and BZ.data.splines[0].type == 'NURBS' and
      reparam_gap(C1, C2, C3, De, DDe, D3e, 3) < 1e-5, reparam_gap(C1, C2, C3, De, DDe, D3e, 3))

# ------------------------------------------------------------- Align: a spline of a multi-spline sketch
other = [(10, 0, 0), (11, 1, 0), (12, 0, 0), (13, 1, 0)]
SK = nurbs("Sketch", None, splines=[other, [(-0.7, 1.4, 10), (0.5, 2.2, 10.2), (1.5, 1.0, 10), (3.0, 2.0, 10)]])
r = run_align(SK, E, "S0", 0, "G2", sref="S1")
P, C1, C2, C3 = start_jet(SK, 0, "S1")
S0_now = np.array([tuple(q.co)[:3] for q in SK.data.splines[0].points])
check("align spline S1 of a sketch: G2, the other spline kept", r == {'FINISHED'} and len(SK.data.splines) == 2
      and np.abs(S0_now - np.array(other)).max() < 1e-6 and reparam_gap(C1, C2, C3, De, DDe, D3e, 2) < 1e-5,
      reparam_gap(C1, C2, C3, De, DDe, D3e, 2))
features.rebuild(SK, ctx, force=True)
check("align sketch spline: rebuild keeps both splines", len(SK.data.splines) == 2)

# ------------------------------------------------------------- Blend Curve
def blend(ia, ib, **kw):
    r = bpy.ops.nproto.blend_curve('EXEC_DEFAULT', inputs=json.dumps([list(ia), list(ib)]), **kw)
    return r, ctx.active_object

def blend_jets(o):
    """(side A jet into the blend, side B jet into the blend)."""
    return start_jet(o, 0), start_jet(o, 1)

# face point, perpendicular (G3), to a curve end (G2)
G = nurbs("G", [(5.0, 5.0, 3.0), (6.0, 5.5, 3.5), (7.0, 5.0, 3.2), (8.0, 4.0, 3.0)])
for ca in ("G1", "G2", "G3"):
    r, B = blend((S.name, "F0"), (G.name, "S0"), uv_a=(0.4, 0.6), mode_a="NORMAL", cont_a=ca, cont_b="G2", end_b=0)
    (Pa, A1, A2, A3), (Pb, B1, B2, B3) = blend_jets(B)
    Ps, n, _, _, _, _ = align.surface_jet(S, "F0", None, uv=(0.4, 0.6))
    ka = int(ca[1])
    ga = reparam_gap(A1, A2, A3, n if n @ A1 > 0 else -n, z, z, ka)
    Pg, Dg, DDg = geom.curve_end_frame(features.input_curve(G, "S0"), 0)
    gb = reparam_gap(B1, B2, B3, Dg, DDg, z, 2)
    h = features.history(B)
    check(f"blend {ca} perpendicular from a face point, G2 to a curve", r == {'FINISHED'} and np.linalg.norm(Pa - Ps) < 1e-5
          and ga < 1e-5 and gb < 1e-5 and h["params"].get("bc_mode_a") == "NORMAL", (ga, gb))
    if ca == "G3":
        before = np.array([tuple(q.co)[:3] for q in B.data.splines[0].points])
        features.rebuild(B, ctx, force=True)
        after = np.array([tuple(q.co)[:3] for q in B.data.splines[0].points])
        check("blend from a face: rebuild gives it back", np.abs(after - before).max() < 1e-6, np.abs(after - before).max())

# surface edge, tangent to the surface across the edge, G3
r, B = blend((S.name, "U1"), (G.name, "S0"), mode_a="SURFACE", cont_a="G3", cont_b="G1", on_a=True, at_a=0.3)
(Pa, A1, A2, A3), _ = blend_jets(B)
kc = features.input_curve(S, "U1"); t0, t1 = kc.parameter_range()
on_edge = np.array(kc.evaluate(t0 + 0.3 * (t1 - t0)), float)
e = np.array(geom.curve_derivs(kc, t0 + 0.3 * (t1 - t0), 1)[1])
g = surface_gap(A1, A2, A3, Pa, S, "U1", 3)
check("blend G3 tangent to the surface, square to the edge, at a point along it",
      r == {'FINISHED'} and np.linalg.norm(Pa - on_edge) < 1e-5 and abs(e @ A1) / np.linalg.norm(e) / np.linalg.norm(A1) < 1e-5
      and g < 1e-5, g)
uvS = align.surface_jet(S, "U1", Pa)[4]
check("blend across the edge: leaves the surface (off the face)", uvS[0] > 0.99 and (A1 @ np.array(geom.curve_derivs(
    features.input_curve(S, "V0"), features.input_curve(S, "V0").parameter_range()[1], 1)[1])) > 0)

# old blends (no new settings) rebuild as before: the edge itself continued at its end
r, B = blend((S.name, "U1"), (G.name, "S0"), mode_a="EDGE", cont_a="G2", end_a=1)
(Pa, A1, A2, A3), _ = blend_jets(B)
fa = geom.curve_end_frame(features.input_curve(S, "U1"), 1)
check("blend Continue Edge on a surface edge: the edge continued", reparam_gap(A1, A2, A3, fa[1], fa[2], z, 2) < 1e-5)

# a point along a closed sketch (circle-like), toward the other side
R = 1.5
ring = [(R * np.cos(a) + 20, R * np.sin(a), 0.0) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
O = nurbs("Ring", ring, cyclic=True)
H = nurbs("H", [(24, 0, 2.0), (25, 0, 2.5), (26, 0.5, 2.5)])
r, B = blend((O.name, "S0"), (H.name, "S0"), on_a=True, at_a=0.1, cont_a="G2", cont_b="G2", end_b=0)
(Pa, A1, A2, A3), _ = blend_jets(B)
kc = features.input_curve(O, "S0"); t0, t1 = kc.parameter_range()
Dc = geom.curve_derivs(kc, t0 + 0.1 * (t1 - t0), 2)
Pb = np.array(features.input_curve(H, "S0").evaluate(features.input_curve(H, "S0").parameter_range()[0]))
toward = Dc[1] if Dc[1] @ (Pb - Dc[0]) > 0 else -Dc[1]
check("blend from a point along a closed sketch, G2, toward the other side",
      r == {'FINISHED'} and np.linalg.norm(Pa - Dc[0]) < 1e-5 and reparam_gap(A1, A2, A3, toward, Dc[2], z, 2) < 1e-5,
      reparam_gap(A1, A2, A3, toward, Dc[2], z, 2))

# ------------------------------------------------------------- CV Curve: ends on a face (perpendicular) and an edge
clicks = [(1.6, 2.2, 0.5), (1.8, 2.4, 1.3), (2.6, 2.0, 2.0), (3.5, 2.6, 2.4), (5.0, 5.0, 3.0)]   # ends on G's start
P0 = align.surface_jet(S, "F0", np.array(clicks[0]))[0]
clicks[0] = tuple(P0)
for mode in ("CV", "THROUGH"):
    ends = {"0": ({"cont": "G3", "weight": 1.0, "mode": "NORMAL", "snap": "NEAREST", "flip": False},
                  align.end_frame(S, "F0", P0, "NEAREST", "NORMAL")),
            "1": ({"cont": "G2", "weight": 1.0, "mode": "EDGE", "snap": "END", "flip": False},
                  align.end_frame(G, "S0", np.array(clicks[-1]), "END", "EDGE"))}
    sp_, specs = drawtools.build_spec_ends(clicks, mode, 3, ends)
    obj = drawtools.add_curve(ctx, "CVC", [sp_])
    P, C1, C2, C3 = start_jet(obj, 0)
    _, n, _, _, _, _ = align.surface_jet(S, "F0", P)
    ga = reparam_gap(C1, C2, C3, n if n @ C1 > 0 else -n, z, z, 3)
    upward = (n @ C1) * (n @ (np.array(clicks[2]) - P)) > 0
    inputs = [(obj, "S0"), (S, "F0"), (G, "S0")]
    ends_h = {"0": dict({k: specs["0"][k] for k in ("cont", "weight", "mode", "snap", "flip")}, input=1),
              "1": dict({k: specs["1"][k] for k in ("cont", "weight", "mode", "snap", "flip")}, input=2)}
    before = np.array([tuple(q.co)[:3] for q in obj.data.splines[0].points])
    kind, deg_, Q, _ = align.solve(obj, ends_h, inputs)
    # (single-precision CVs: a G3 end's third derivative magnifies their rounding)
    check(f"CV Curve ({mode}) G3 perpendicular from a face, on the clicks' side", ga < 1e-3 and upward, ga)
    check(f"CV Curve ({mode}): its Align history solves to itself", np.abs(Q - before).max() < 1e-6, np.abs(Q - before).max())
    if mode == "THROUGH":
        kc = features.input_curve(obj, "S0")
        ds = [float(np.linalg.norm(np.array(kc.evaluate(align.curve_param_near(kc, np.array(c)))) - c)) for c in clicks]
        check("CV Curve through points: still through every click", max(ds) < 1e-5, ds)

# ------------------------------------------------------------- Symmetry Plane
def flat(name, off):
    pts = [(i * 1.0 + off[0], j * 1.0 + off[1], off[2]) for i in range(4) for j in range(4)]
    s = ext.surface_from_cvs(3, 3, pts, geom.clamped_uniform_knots(4, 3), geom.clamped_uniform_knots(4, 3), None)
    return store.new_object(ctx, name, s)

def sym_gaps(C1, C2, C3, n, k):
    """(square to the plane, n.C'' (G2: the mirror image joins C2), C''' in the plane (G3)), relative."""
    L = np.linalg.norm(C1)
    g = [np.linalg.norm(np.cross(C1 / L, n))]
    if k >= 2:
        g.append(abs(n @ C2) / L ** 2)
    if k >= 3:
        g.append(np.linalg.norm(C3 - (C3 @ n) * n) / L ** 3)
    return max(g)

def bend(C1, C2):
    """Curvature at the end."""
    return np.linalg.norm(np.cross(C1, C2)) / np.linalg.norm(C1) ** 3

PL = flat("Mirror", (20.0, 0.0, 0.0))
nz = np.array([0.0, 0.0, 1.0])
for cont, k in (("G1", 1), ("G2", 2), ("G3", 3)):
    C = nurbs("Sym" + cont, [(21.0, 1.5, 0.05), (21.6, 1.6, 0.4), (22.5, 1.4, 0.9), (23.0, 2.0, 1.6), (23.8, 2.6, 2.0), (24.5, 2.4, 2.6)])
    r = run_align(C, PL, "F0", 0, cont, "SYMMETRY")
    P, C1, C2, C3 = start_jet(C, 0)
    g = sym_gaps(C1, C2, C3, nz, k)
    check(f"align {cont} symmetry plane: square to it, mirror joins C{k}", r == {'FINISHED'} and abs(P[2]) < 1e-6 and g < 1e-5
          and C1[2] > 0, g)
    if k >= 2:
        check(f"align {cont} symmetry plane: the bend at the plane is kept (not flattened)", bend(C1, C2) > 0.05, bend(C1, C2))
    if k == 3:
        before = np.array([tuple(q.co)[:3] for q in C.data.splines[0].points])
        features.rebuild(C, ctx, force=True)
        after = np.array([tuple(q.co)[:3] for q in C.data.splines[0].points])
        check("align symmetry: a rebuild solves to itself (single precision at x ~ 24)",
              np.abs(after - before).max() < 4e-7 * (1 + np.abs(before).max()), np.abs(after - before).max())
        # constrained CV editing (blendedit): move CV 2 and 3, the end stays on the mirror rules
        from blender_nurbs import blendedit
        for o in ctx.view_layer.objects: o.select_set(False)
        C.select_set(True); ctx.view_layer.objects.active = C
        bpy.ops.object.mode_set(mode='EDIT')
        rng = np.random.default_rng(3)
        for i in (2, 3):
            M = np.array(geom.world_matrix(C), float)
            W = before[i] @ M[:3, :3].T + M[:3, 3]
            for j, q in enumerate(C.data.splines[0].points): q.select = (j == i)
            Wt = W + rng.normal(size=3) * 0.25
            q = np.linalg.inv(M)[:3, :3] @ Wt + np.linalg.inv(M)[:3, 3]
            C.data.splines[0].points[i].co = (*q, 1.0)
            ctx.view_layer.update()
            P, C1, C2, C3 = start_jet(C, 0)
            check(f"align symmetry CV {i} moved: the mirror rules hold", sym_gaps(C1, C2, C3, nz, 3) < 1e-4 and abs(P[2]) < 1e-5,
                  sym_gaps(C1, C2, C3, nz, 3))
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects.remove(C)

# Blend from the mirror plane: Curvature bends it at the plane, still C3 with its mirror image
H2 = nurbs("H2", [(25.0, 3.0, 2.0), (26.0, 3.5, 2.5), (27.0, 3.0, 2.2)])
for c_a, k in ((0.0, 2), (0.8, 2), (0.8, 3)):
    r, B = blend((PL.name, "F0"), (H2.name, "S0"), uv_a=(0.3, 0.4), mode_a="SYMMETRY", cont_a=f"G{k}", cont_b="G2",
                 end_b=0, c_a=c_a)
    (Pa, A1, A2, A3), _ = blend_jets(B)
    g = sym_gaps(A1, A2, A3, nz, k)
    bent = bend(A1, A2)
    check(f"blend G{k} symmetry plane, Curvature {c_a}: mirror joins C{k}, bend {'kept' if c_a else 'none'}",
          r == {'FINISHED'} and g < 1e-5 and ((bent > 1e-3) if c_a else (bent < 1e-6)), (g, bent))
    if k == 3:
        before = np.array([tuple(q.co)[:3] for q in B.data.splines[0].points])
        features.rebuild(B, ctx, force=True)
        after = np.array([tuple(q.co)[:3] for q in B.data.splines[0].points])
        check("blend symmetry: rebuild gives it back", np.abs(after - before).max() < 1e-6, np.abs(after - before).max())

# CV Curve starting on the mirror plane
clicks = [(21.5, 1.0, 0.0), (21.8, 1.4, 0.8), (22.4, 1.2, 1.6), (23.2, 1.8, 2.0), (24.0, 2.4, 2.4)]
ends = {"0": ({"cont": "G3", "weight": 1.0, "mode": "SYMMETRY", "snap": "NEAREST", "flip": False},
              align.end_frame(PL, "F0", np.array(clicks[0]), "NEAREST", "SYMMETRY"))}
sp_, specs = drawtools.build_spec_ends(clicks, "CV", 3, ends)
obj = drawtools.add_curve(ctx, "CVSym", [sp_])
P, C1, C2, C3 = start_jet(obj, 0)
check("CV Curve G3 on a symmetry plane: square to it, mirror joins C3, on the clicks' side",
      sym_gaps(C1, C2, C3, nz, 3) < 1e-5 and C1[2] > 0, sym_gaps(C1, C2, C3, nz, 3))

# ------------------------------------------------------------- Through Points: no overshoot next to an aligned end
# (close clicks turning away from a G2 / G3 end used to give curves 15x to 35000x the clicks' path)
Ed = nurbs("Ed", [(-2.0, 2.0, 0.5), (-1.0, 2.5, 0.5), (0.0, 2.0, 0.5), (1.6, 2.2, 0.5)], loc=(0, 0, 0))
Gd = nurbs("Gd", [(5.0, 5.0, 3.0), (6.0, 5.5, 3.5), (7.0, 5.0, 3.2), (8.0, 4.0, 3.0)])
P0e = np.array([1.6, 2.2, 0.5])
P0s = align.surface_jet(S, "F0", P0e)[0]
layouts = {"close turn": [(1.8, 2.4, 1.3), (2.6, 2.0, 2.0), (3.5, 2.6, 2.4), (5.0, 5.0, 3.0)],
           "very close": [(1.65, 2.25, 0.7), (2.4, 2.8, 1.4), (3.6, 3.8, 2.4), (5.0, 5.0, 3.0)],
           "many": [(1.8, 2.3, 0.9), (2.2, 2.5, 1.5), (2.8, 3.0, 2.0), (3.5, 3.6, 2.5), (4.3, 4.4, 2.8), (5.0, 5.0, 3.0)]}
import time
worst, slowest, gaps, cont_gaps = 0.0, 0.0, 0.0, 0.0
for lname, rest in layouts.items():
    for mode, tgt, ref, P0 in (("EDGE", Ed, "S0", P0e), ("NORMAL", S, "F0", P0s), ("SURFACE", S, "F0", P0s),
                               ("SYMMETRY", S, "F0", P0s)):
        for c0 in ("G1", "G2", "G3"):
            Sx = np.array([P0] + rest, float)
            snap = "END" if ref == "S0" else "NEAREST"
            ends = {"0": ({"cont": c0, "weight": 1.0, "mode": mode, "snap": snap, "flip": False},
                          align.end_frame(tgt, ref, P0, snap, mode)),
                    "1": ({"cont": "G2", "weight": 1.0, "mode": "EDGE", "snap": "END", "flip": False},
                          align.end_frame(Gd, "S0", Sx[-1], "END", "EDGE"))}
            t_ = time.perf_counter()
            sp_, specs = drawtools.build_spec_ends(Sx, "THROUGH", 3, ends)
            slowest = max(slowest, time.perf_counter() - t_)
            o = drawtools.add_curve(ctx, "Thr", [sp_])
            kc = features.input_curve(o, "S0")
            t0, t1 = kc.parameter_range()
            W = np.array([kc.evaluate(t) for t in np.linspace(t0, t1, 3000)])
            ratio = np.sum(np.linalg.norm(np.diff(W, axis=0), axis=1)) / np.sum(np.linalg.norm(np.diff(Sx, axis=0), axis=1))
            worst = max(worst, ratio)
            gaps = max(gaps, max(float(np.linalg.norm(np.array(kc.evaluate(align.curve_param_near(kc, c))) - c)) for c in Sx))
            k = int(c0[1])
            # measured on the solved CVs (double): Blender keeps CVs in single precision,
            # and a G3 end's third derivative magnifies that rounding to ~1e-4
            X = np.array(sp_["points"], float); pp = sp_["order"] - 1
            U_ = geom.clamped_uniform_knots(len(X), pp)
            C1, C2, _, _ = align._derivs(X, U_, pp)
            C3, P = align._d3(X, U_, pp), X[0]
            if mode == "EDGE":
                fr = align.end_frame(tgt, ref, P0, snap, mode)
                g = reparam_gap(C1, C2, C3, fr[1], fr[2], align.frame_d3(fr), k)
            elif mode == "SURFACE":
                g = surface_gap(C1, C2, C3, P, tgt, ref, k)
            else:
                n_ = align.surface_jet(tgt, ref, P)[1]
                g = reparam_gap(C1, C2, C3, n_ if n_ @ C1 > 0 else -n_, z, z, k) if mode == "NORMAL" else \
                    sym_gaps(C1, C2, C3, n_, k)
            cont_gaps = max(cont_gaps, g)
            bpy.data.objects.remove(o)
check("Through Points with G1-G3 ends: no overshoot (curve <= 1.1x the clicks' path)", worst < 1.1, worst)
check("Through Points: still through every click", gaps < 1e-5, gaps)
check("Through Points: the start's G1-G3 condition holds exactly", cont_gaps < 1e-9, cont_gaps)
check("Through Points: fast enough for the live preview (< 0.25 s)", slowest < 0.25, slowest)

# ------------------------------------------------------------- G3 keeps the shape (CVs added)
def poly_dev(A, B):
    a, b = B[:-1], B[1:]; ab = b - a; L2 = np.maximum((ab * ab).sum(1), 1e-30); out = 0.0
    for x in A:
        t = np.clip(((x - a) * ab).sum(1) / L2, 0, 1)
        out = max(out, float(np.min(np.linalg.norm(a + t[:, None] * ab - x, axis=1))))
    return out

def samples(o):
    c = features.input_curve(o, "S0"); t0, t1 = c.parameter_range()
    return np.array([c.evaluate(t) for t in np.linspace(t0, t1, 300)])

# Align Curve: a 5-CV cubic aligned G3 is refitted to degree 4 with the 3 CVs its end takes added
C5 = nurbs("Short G3", [(20.5, 1.0, 0.05), (20.7, 1.0, 1.2), (21.7, 1.0, 2.2), (23.1, 1.0, 2.8), (23.9, 1.0, 4.4)])   # upright on the plane
before = samples(C5)
size = float(np.linalg.norm(np.ptp(before, axis=0)))
r = run_align(C5, PL, "F0", 0, "G3", "SYMMETRY")
after = samples(C5)
d = max(poly_dev(before[45:255], after), poly_dev(after[45:255], before)) / size
check("align G3 on a 5-CV curve: 3 CVs added (8, degree 4), its shape kept",
      r == {'FINISHED'} and len(C5.data.splines[0].points) == 8 and C5.data.splines[0].order_u == 5 and d < 0.07,
      (len(C5.data.splines[0].points), d))
nb = len(C5.data.splines[0].points)
features.rebuild(C5, ctx, force=True)
check("align G3 on a 5-CV curve: rebuilds add nothing more", len(C5.data.splines[0].points) == nb)

# CV Curve, CV mode: a G3 end adds one CV and moves the same clicks as G2 (the 2nd and 3rd)
clicks = np.array([(21.0, 1.5, 0.0), (21.05, 1.5, 0.7), (21.45, 1.5, 1.3), (22.2, 1.5, 1.75), (23.1, 1.5, 2.0),
                   (24.0, 1.5, 2.4), (24.6, 1.5, 3.1)])
ends = {"0": ({"cont": "G3", "weight": 1.0, "mode": "SYMMETRY", "snap": "NEAREST", "flip": False},
              align.end_frame(PL, "F0", clicks[0], "NEAREST", "SYMMETRY"))}
sp_, specs = drawtools.build_spec_ends(clicks, "CV", 3, ends)
X = np.array(sp_["points"], float)
kept = [i for i in range(1, len(clicks)) if np.min(np.linalg.norm(X - clicks[i], axis=1)) < 1e-9]
check("CV Curve CV mode G3: one CV added, clicks 3 on stay where clicked",
      len(X) == len(clicks) + 1 and kept == list(range(3, len(clicks))), (len(X), kept))

# ------------------------------------------------------------- CV Curve keeps the degree asked for (2 clicks, G2 ends)
# The degree was clamped to the clicks (2 -> 1) before the ends added CVs, so
# the ends' minimum (cubic) won whatever degree was picked.
LA, LB = nurbs("DegA", [(-6, -6, 0), (-2, -2, 0)], order=2), nurbs("DegB", [(6, -6, 0), (2.5, -2.5, 0)], order=2)
clicks = [(-2.0, -2.0, 0.0), (2.5, -2.5, 0.0)]
for want, exp in ((5, 5), (4, 4), (1, 3)):
    ends = {"0": ({"cont": "G2", "weight": 1.0, "mode": "EDGE", "snap": "END", "flip": False},
                  align.end_frame(LA, "S0", np.array(clicks[0]), "END", "EDGE")),
            "1": ({"cont": "G2", "weight": 1.0, "mode": "EDGE", "snap": "END", "flip": False},
                  align.end_frame(LB, "S0", np.array(clicks[1]), "END", "EDGE"))}
    sp_, specs = drawtools.build_spec_ends(clicks, "CV", want, ends)
    X = np.array([tuple(q)[:3] for q in sp_["points"]])
    D0, D1 = np.array([1.0, 1.0, 0.0]) / 2 ** .5, np.array([-1.0, 1.0, 0.0]) / 2 ** .5
    off = max(max(np.linalg.norm(np.cross(X[j] - X[0], D0)) for j in (1, 2)),
              max(np.linalg.norm(np.cross(X[-1 - j] - X[-1], D1)) for j in (1, 2)))
    check(f"CV Curve degree {want}, G2 ends on lines: degree {exp}, 6 CVs, first legs on the lines",
          sp_["order"] - 1 == exp and len(X) == 6 and off < 1e-6, (sp_["order"] - 1, len(X), off))
# G3 ends: the refit used to count the legs' inserted CVs twice, leaving two on
# the clicks' chord: the curve dipped to it (two inflections). Now 2 + 6 CVs,
# all on the ends, and the CV polygon turns one way only (no inflection).
for want in (4, 5):
    ends = {e: ({"cont": "G3", "weight": 1.0, "mode": "EDGE", "snap": "END", "flip": False},
                align.end_frame(L, "S0", np.array(c), "END", "EDGE")) for e, L, c in (("0", LA, clicks[0]), ("1", LB, clicks[1]))}
    sp_, _ = drawtools.build_spec_ends(clicks, "CV", want, ends)
    X = np.array([tuple(q)[:3] for q in sp_["points"]])
    legs = np.diff(X, axis=0)
    turns = [float(np.cross(a, b)[2]) for a, b in zip(legs[:-1], legs[1:])]
    sc = 1e-9 * max(abs(t) for t in turns)
    one_way = all(t <= sc for t in turns) or all(t >= -sc for t in turns)
    check(f"CV Curve degree {want}, G3 ends on lines: 8 CVs, no dip between the ends",
          sp_["order"] - 1 == want and len(X) == 8 and one_way, (sp_["order"] - 1, len(X), np.round(turns, 4)))
# A middle click: it stays a CV (a refit put it on the old curve, below the
# click, while the long end legs rose above it: a dip at degree 4), and the
# ends' legs stop short enough that the CV polygon turns one way only.
mid = (0.2, -0.5, 0.0)          # inside the lines' wedge: a curve with no inflection exists
for cont in ("G1", "G2", "G3"):
    for want in (3, 4, 5):
        ends = {e: ({"cont": cont, "weight": 1.0, "mode": "EDGE", "snap": "END", "flip": False},
                    align.end_frame(L, "S0", np.array(c), "END", "EDGE")) for e, L, c in (("0", LA, clicks[0]), ("1", LB, clicks[1]))}
        sp_, _ = drawtools.build_spec_ends([clicks[0], mid, clicks[1]], "CV", want, ends)
        X = np.array([tuple(q)[:3] for q in sp_["points"]])
        k = align.CONT[cont]
        kept = np.linalg.norm(X - np.array(mid), axis=1).min()
        check(f"CV Curve {cont} degree {want}, a middle click: still a CV, 3 + {2 * k} CVs, no dip",
              kept < 1e-9 and len(X) == 3 + 2 * k and align.one_way(X), (kept, len(X)))

# ------------------------------------------------------------- the user's file: aligned curves rebuild stably
user = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(ROOT, "Blend Files", "blend cv align curve  spline surface.blend")
if os.path.exists(user):
    bpy.ops.wm.open_mainfile(filepath=user)
    for o in [o for o in bpy.data.objects if o.type == 'CURVE' and (features.history(o) or {}).get("op") == "align_curve"]:
        features.rebuild(o, bpy.context, force=True)        # (a file saved out of date with its targets catches up)
        before = np.array([tuple(q.co)[:3] for q in o.data.splines[0].points])
        features.rebuild(o, bpy.context, force=True)
        after = np.array([tuple(q.co)[:3] for q in o.data.splines[0].points])
        d = np.abs(after - before).max() if after.shape == before.shape else np.inf
        check(f"user file: {o.name} rebuilds to itself", d < 1e-5, d)

print("ALL PASS" if ok else "SOME FAILED")
