"""Curve Network:  blender -b --factory-startup --python tests/test_network.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, geom, network as nw, store, surfeval
from blender_nurbs.primitives import write_spline_object

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

def nurbs(name, pts, order=4):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for q, c in zip(sp.points, pts): q.co = (*c, 1.0)
    sp.order_u = order; sp.use_endpoint_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def loft(name, ys, xs, f):
    secs = [nurbs(f"{name}_s{k}", [(x, y, f(x, y)) for x in xs]) for k, y in enumerate(ys)]
    bpy.ops.nproto.loft(inputs=ins(*[(o, "S0") for o in secs]))
    o = ctx.active_object; o.name = name
    for s in secs:
        bpy.data.objects.remove(s)
    return o

def iso(src, d, t, name):
    g = features.run("iso_curve", [(src, "SURF")], {"iso_dir": d, "param": t})
    cu = bpy.data.curves.new(name, 'CURVE'); o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    write_spline_object(o, g[1]); return o

def surf_of(obj):
    return surfeval.Surf(store.get_brep(obj).faces()[0].surface())

def dev_to(E_src, obj, n=30):
    E = surf_of(obj)
    G = E.grid(np.linspace(*E.u_domain, n), np.linspace(*E.v_domain, n), 0)[0, 0].reshape(-1, 3)
    M = np.array(obj.matrix_world, float); G = G @ M[:3, :3].T + M[:3, 3]
    uv = E_src.feet(G)
    return float(np.linalg.norm(E_src.at(uv[:, 0], uv[:, 1], 0)[0, 0] - G, axis=1).max())

def run(pairs, **kw):
    try:
        r = bpy.ops.nproto.network(inputs=ins(*pairs), **kw)
    except RuntimeError as exc:
        print("   refused:", str(exc).strip())
        return {'CANCELLED'}
    return r

f = lambda x, y: 0.4 * math.sin(1.3 * x + 0.8 * y)

# --- iso curves of a surface: the network gives the surface back -------------
clear()
L = loft("Src", (-2, -0.7, 0.7, 2), np.linspace(-2, 2, 7), f)
S = surf_of(L)
(u0, u1), (v0, v1) = S.u_domain, S.v_domain
A = [iso(L, "U", u0 + (u1 - u0) * t, f"A{i}") for i, t in enumerate((0.0, 0.35, 0.7, 1.0))]
B = [iso(L, "V", v0 + (v1 - v0) * t, f"B{i}") for i, t in enumerate((0.0, 0.5, 1.0))]
L.hide_set(True)
shuffled = [B[1], A[1], B[2], A[0], A[3], B[0], A[2]]
t = time.time()
r = run([(o, "S0") for o in shuffled])
dt = time.time() - t
N = ctx.active_object
check("network from iso curves (any order) gives the surface back", r == {'FINISHED'} and dev_to(S, N) < 5e-5,
      f"dev {dev_to(S, N):.1e} · {nw.LAST_INFO['curves']} curves, {nw.LAST_INFO['cvs']} CVs, {dt:.2f}s")
check("curves on the network", max(nw.LAST_INFO["gap"], nw.LAST_INFO["curve_gap"]) < 1e-4,
      f"{nw.LAST_INFO['curve_gap']:.1e}")
h = features.history(N)
check("history: network feature with all curves", h["op"] == "network" and len(h["inputs"]) == 7)
# a reversed curve (its points turned round) changes nothing
sp = A[1].data.splines[0]
co = [tuple(q.co) for q in sp.points]
for q, c in zip(sp.points, reversed(co)):
    q.co = c
r = run([(o, "S0") for o in shuffled])
check("a reversed curve is turned back", r == {'FINISHED'} and dev_to(S, ctx.active_object) < 5e-5)

# selection start: select the curves, invoke
bpy.ops.object.select_all(action='DESELECT')
for o in A + B:
    o.select_set(True)
ctx.view_layer.objects.active = A[0]
op = bpy.ops.nproto.network
r = op('EXEC_DEFAULT', inputs=ins(*[(o, "S0") for o in A + B]))
check("operator from the selection", r == {'FINISHED'})

# --- refusals ------------------------------------------------------------------
r = run([(o, "S0") for o in A[:2] + B[:1]])
check("too few curves: refused", r == {'CANCELLED'})
far = nurbs("Far", [(5, 5, 0), (6, 5, 0), (7, 5, 0), (8, 5, 0)])
r = run([(o, "S0") for o in A + B] + [(far, "S0")])
check("a curve that touches nothing: refused", r == {'CANCELLED'})

# --- G1 / G2 to two surfaces across a gap ----------------------------------------
# Left: a flat plate up to x = -1. Right: a twisted ruled surface from x = 1 whose
# slope at its edge is 0.3 y^2 (1.2 at the ends, 0 in the middle). Only the two
# outer curves cross the gap, leaving tangent at both ends (curvature 0, as the
# rulings): a G0 net between them creases mid-edge, G1 / G2 must not.
clear()
bpy.ops.nproto.add_primitive(kind="PLANE", width=1.0, depth=4.0); S1 = ctx.active_object
S1.location = (-1.5, 0, 0)
ctx.view_layer.update()
L1 = nurbs("R1", [(1, -2, 0.5), (1, 2, 0.5)], order=2)
L2 = nurbs("R2", [(2, -2, 1.7), (2, 0, -0.7), (2, 2, 1.7)], order=3)
bpy.ops.nproto.loft(inputs=ins((L1, "S0"), (L2, "S0"))); S2 = ctx.active_object
L1.hide_set(True); L2.hide_set(True)

def edge_at(obj, x):
    for e in ("U0", "U1", "V0", "V1"):
        P = geom.sample_curve(features.input_curve(obj, e), 7)
        if abs(P[:, 0] - x).max() < 1e-6:
            return e
le, re = edge_at(S1, -1.0), edge_at(S2, 1.0)
check("edges found", le is not None and re is not None, f"{le} {re}")
cross = []
for k, y in enumerate((-2.0, 2.0)):
    d = np.array([1.0, 0.0, 0.3 * y * y]); d /= np.linalg.norm(d)
    P0, P5 = np.array([-1.0, y, 0.0]), np.array([1.0, y, 0.5])
    pts = [P0, P0 + [0.35, 0, 0], P0 + [0.7, 0, 0], P5 - 0.7 * d, P5 - 0.35 * d, P5]
    cross.append(nurbs(f"C{k}", [tuple(q) for q in pts], order=6))

def edge_angle(net, src, e):
    E, Es = surf_of(net), surf_of(src)
    M, Ms = np.array(net.matrix_world, float), np.array(src.matrix_world, float)
    P = geom.sample_curve(features.input_curve(src, e), 25)[1:-1]
    a = E.feet((P - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T)
    b = Es.feet((P - Ms[:3, 3]) @ np.linalg.inv(Ms[:3, :3]).T)
    n1 = surfeval.normals(E.at(a[:, 0], a[:, 1], 1))[0] @ M[:3, :3].T
    n2 = surfeval.normals(Es.at(b[:, 0], b[:, 1], 1))[0] @ Ms[:3, :3].T
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True); n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", n1, n2)), 0, 1))).max())

pairs = [(S1, le), (S2, re)] + [(o, "S0") for o in cross]
for cont in ("G0", "G1", "G2"):
    r = run(pairs, nw_cont=cont)
    inf = dict(nw.LAST_INFO)
    NET = ctx.active_object
    ang = edge_angle(NET, S2, re) if r == {'FINISHED'} else 99
    if cont == "G0":
        check("G0: creases at the twisted surface", r == {'FINISHED'} and ang > 5.0, f"normals {ang:.2f}°")
    else:
        angl = edge_angle(NET, S1, le)
        good = r == {'FINISHED'} and sum(inf["edge_sides"].values()) == 2 and max(ang, angl) < 0.1 and \
            max(inf["gap"], inf["curve_gap"]) < 1e-3
        if cont == "G2":
            good &= inf["curv"] < 0.1
        check(f"{cont} to both surfaces", good,
              f"normals {angl:.4f}° / {ang:.4f}° · curvature {100 * inf.get('curv', 0):.2f}% · "
              f"gap {max(inf.get('gap', 0), inf.get('curve_gap', 0)):.1e} · {inf.get('cvs')} CVs degree {inf.get('degree')}")

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
