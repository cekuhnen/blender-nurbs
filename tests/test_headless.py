"""Headless integration test:  blender -b --factory-startup --python tests/test_headless.py"""
import json, math, os, sys, tempfile
import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, geom, history, jobs, steps
from blender_nurbs.kernel import load as _kl
ext = _kl().blender_ext

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def bezier(name, pts, z=0.0):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('BEZIER'); sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = (p[0], p[1], z); bp.handle_left_type = bp.handle_right_type = 'AUTO'
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def nurbs(name, pts, order=4, endpoint=True):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.order_u = order; sp.use_endpoint_u = endpoint
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

A = bezier("A", [(0,0),(1,0.4),(2,0),(3,0.4)], 0.0)
B = nurbs("B", [(0,0,2),(1,-0.4,2),(2,0,2),(3,-0.4,2)])
C = nurbs("C", [(0,0,4),(1,0.3,4.3),(2,0,4),(3,0.3,4)], endpoint=False)

# --- Loft G0 (kernel skin), 3 curves
r = bpy.ops.nproto.loft(inputs=ins((A,"S0"),(B,"S0"),(C,"S0")))
L = ctx.active_object
check("loft skin 3 curves", r == {'FINISHED'} and store.is_nurbs(L) and len(L.data.polygons) > 0,
      f"{len(L.data.polygons)} tris")
check("loft has history", features.history(L) is not None)

# --- Extrude + Revolve + Planar
r = bpy.ops.nproto.extrude(inputs=ins((A,"S0")), distance=1.5, direction="Y")
E1 = ctx.active_object; check("extrude", r == {'FINISHED'} and len(E1.data.polygons) > 0)
prof = nurbs("Prof", [(1,0,0),(1.3,0,0.5),(0.8,0,1),(1,0,1.5)])
r = bpy.ops.nproto.revolve(inputs=ins((prof,"S0")), axis="Z", pivot="WORLD", angle=360)
R = ctx.active_object; check("revolve", r == {'FINISHED'} and len(R.data.polygons) > 0)
bpy.ops.curve.primitive_bezier_circle_add(radius=1.0, location=(5,0,0))
circ = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); circ.select_set(True)
r = bpy.ops.nproto.planar(); P = ctx.active_object
check("planar", r == {'FINISHED'} and store.is_nurbs(P) and len(P.data.polygons) > 0,
      f"area={store.get_brep(P).area():.4f} (pi={math.pi:.4f})")
bpy.ops.object.select_all(action='DESELECT'); R.select_set(True)
r = bpy.ops.nproto.offset(distance=0.1); O = ctx.active_object
check("offset", r == {'FINISHED'} and len(O.data.polygons) > 0)

# --- Two surfaces facing each other, G1/G2 blend between their edges
s1c = nurbs("S1c", [(0,0,0),(1,0.3,0),(2,-0.2,0),(3,0,0)])
s2c = nurbs("S2c", [(0,0,3),(1,-0.3,3),(2,0.2,3),(3,0,3)])
bpy.ops.nproto.extrude(inputs=ins((s1c,"S0")), distance=1.0, direction="Z"); S1 = ctx.active_object
# extrude downward from z=0: flip direction by negative distance
bpy.ops.nproto.extrude(inputs=ins((s2c,"S0")), distance=-1.0, direction="Z"); S2 = ctx.active_object
# S1 spans z 0..1 (u along Z); S2 spans z 3..2; find edges facing each other
def edge_z(obj, which):
    s = store.get_surfaces(obj)[0]
    return np.mean([p[2] for p in geom.boundary_points(s, which, 8)])
ea = max(("U0","U1"), key=lambda w: edge_z(S1, w))
eb = min(("U0","U1"), key=lambda w: edge_z(S2, w))
check("facing edges found", abs(edge_z(S1, ea) - 1) < 1e-6 and abs(edge_z(S2, eb) - 2) < 1e-6, f"{ea} {eb}")

def cross_check(blend, src, src_edge, cont):
    """Compare normals (G1) across the shared edge at a few points."""
    sb = store.get_surfaces(blend)[0]; ss = store.get_surfaces(src)[0]
    worst = 0.0
    for t in (0.2, 0.5, 0.8):
        es = geom.sample_surface_edge(ss, src_edge, src.matrix_world, 11)
        p = es.P[int(t*10)]
        # nearest point on blend boundary rows u=0 / u=1
        (u0,u1),(v0,v1) = sb.u_domain(), sb.v_domain()
        best = None
        for u in (u0, u1):
            for k in range(201):
                v = v0 + (v1-v0)*k/200
                q = np.array(sb.evaluate(u, v)); d = np.linalg.norm(q - p)
                if best is None or d < best[0]: best = (d, u, v)
        d, u, v = best
        nb = np.array(sb.normal(u, v)); 
        # source normal at that point
        (a0,a1),(b0,b1) = ss.u_domain(), ss.v_domain()
        uu = a0 if src_edge=="U0" else a1 if src_edge=="U1" else None
        bestv = min(((np.linalg.norm(np.array(ss.evaluate(uu, b0+(b1-b0)*k/200)) - p), b0+(b1-b0)*k/200) for k in range(201)))
        ns = np.array(ss.normal(uu, bestv[1]))
        ang = math.degrees(math.acos(min(1, abs(float(np.dot(nb, ns)) / (np.linalg.norm(nb)*np.linalg.norm(ns))))))
        worst = max(worst, ang)
    return d, worst

for cont in ("G0", "G1", "G2"):
    r = bpy.ops.nproto.loft(inputs=ins((S1,ea),(S2,eb)), cont_a=cont, cont_b=cont, tan_a=1.0, tan_b=1.0)
    BL = ctx.active_object
    gap, ang = cross_check(BL, S1, ea, cont)
    d = store.display(BL)["nets"][0]
    if cont == "G0":
        check(f"blend {cont}", r == {'FINISHED'} and gap < 1e-3, f"gap={gap:.2e} normal angle={ang:.2f}deg nu={d['nu']}")
    else:
        check(f"blend {cont}", r == {'FINISHED'} and gap < 1e-3 and ang < 0.5,
              f"gap={gap:.2e} normal angle={ang:.3f}deg rows={d['nu']}")
G2 = BL

def cross_curvature(surf, u, v, axis, M=None):
    """Normal curvature along the u (axis=0) or v (axis=1) iso direction via finite differences."""
    h = 1e-4
    def P(uu, vv): return np.array(surf.evaluate(uu, vv))
    if axis == 0:
        d1 = (P(u + h, v) - P(u - h, v)) / (2*h); d2 = (P(u + h, v) - 2*P(u, v) + P(u - h, v)) / (h*h)
    else:
        d1 = (P(u, v + h) - P(u, v - h)) / (2*h); d2 = (P(u, v + h) - 2*P(u, v) + P(u, v - h)) / (h*h)
    n = np.array(surf.normal(u, v)); n /= np.linalg.norm(n)
    return float(np.dot(d2, n) / np.dot(d1, d1)), n
sb = store.get_surfaces(G2)[0]; ss = store.get_surfaces(S1)[0]
# source edge ea is U1 or U0 of S1 (u = extrusion direction); blend's u runs across.
us = ss.u_domain()[1] - 1e-3 if ea == "U1" else ss.u_domain()[0] + 1e-3
k_src, n_src = cross_curvature(ss, us, 0.5, 0)
# blend side touching S1: u = 0 (a side), probe just inside
k_bl, n_bl = cross_curvature(sb, 1e-3, 0.5, 0)
k_bl *= np.sign(np.dot(n_src, n_bl))
check("blend G2 curvature match", abs(k_src - k_bl) < 0.05 + 0.05 * abs(k_src), f"k_src={k_src:.4f} k_blend={k_bl:.4f}")

# --- G2 against a source that is curved across the edge (the 3-curve skin L)
top = nurbs("Top", [(0,0,6.5),(1,0.2,6.5),(2,0,6.5),(3,0.2,6.5)])
sL = store.get_surfaces(L)[0]
eL = max(("U0","U1"), key=lambda w: np.mean([p[2] for p in geom.boundary_points(sL, w, 8)]))
uL = sL.u_domain()[1] - 2e-3 if eL == "U1" else sL.u_domain()[0] + 2e-3
res = {}
for cont in ("G1", "G2"):
    bpy.ops.nproto.loft(inputs=ins((L, eL), (top, "S0")), cont_a=cont, tan_a=1.0)
    blend = ctx.active_object; sbl = store.get_surfaces(blend)[0]
    ks = []; kb = []
    for v in (0.3, 0.5, 0.7):
        k1, n1 = cross_curvature(sL, uL, v, 0)
        k2, n2 = cross_curvature(sbl, 2e-3, v, 0)
        ks.append(k1); kb.append(k2 * np.sign(np.dot(n1, n2)))
    res[cont] = (ks, kb)
    bpy.data.objects.remove(blend)
ks, kb2 = res["G2"]; _, kb1 = res["G1"]
err2 = max(abs(a - b) for a, b in zip(ks, kb2)); err1 = max(abs(a - b) for a, b in zip(ks, kb1))
check("G2 matches cross curvature of curved source", err2 < 0.1 * max(abs(k) for k in ks) + 0.02 and err2 < err1,
      f"k_src={[round(k,3) for k in ks]} G2={[round(k,3) for k in kb2]} G1={[round(k,3) for k in kb1]}")

# --- History: move input curve B, process the timer, loft must change
before = sorted(tuple(round(c, 5) for c in v.co) for v in L.data.vertices)
B.location.x += 0.5
ctx.view_layer.update()
history._process()  # in UI the timer does this
after = sorted(tuple(round(c, 5) for c in v.co) for v in L.data.vertices)
check("history follows input move", before != after)
# history chain: move S1 -> G2 blend rebuilds
rev = G2.get("nproto_rev")
S1.location.y += 0.2; ctx.view_layer.update(); history._process()
check("history rebuilds blend from moved surface", G2.get("nproto_rev") != rev)

# --- Delete history
bpy.ops.object.select_all(action='DESELECT'); L.select_set(True)
bpy.ops.nproto.delete_history()
check("delete history", features.history(L) is None)

# --- Trim / Split / Untrim / Fillet / Chamfer (worker process) -----------
def poly(name, pts, loc=(0, 0, 0)):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    o = bpy.data.objects.new(name, cu); o.location = loc; ctx.collection.objects.link(o); return o
def area(o): return store.get_brep(o).area()
def make_floor(name, loc=(0, 0, 0)):
    ln = poly(name + "_ln", [(0, -2, 0), (0, 2, 0)], loc)
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=4.0, direction="X")
    o = ctx.active_object; o.name = name; return o
floor = make_floor("Floor")
check("floor area 16", abs(area(floor) - 16) < 1e-6, f"{area(floor):.4f}")
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 0, 1)); cc = ctx.active_object
tr = dict(target="Floor", cutters=ins((cc, "S0")), ref_point=(1, 0, 0), normal=(0, 0, 1), view_dir=(0, 0, -1))
r = bpy.ops.nproto.trim(action="KEEP", **tr)
check("trim keep circle", r == {'FINISHED'} and abs(area(floor) - math.pi / 4) < 2e-3, f"{area(floor):.4f} vs {math.pi/4:.4f}")
check("trimmed edges pickable (E refs)", not store.is_natural(floor) and len(store.display(floor)["edges"]) >= 1)
r = bpy.ops.nproto.untrim()
check("untrim", r == {'FINISHED'} and abs(area(floor) - 16) < 1e-6 and store.is_natural(floor))
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.trim(action="DELETE", **tr)
check("trim discard circle", abs(area(floor) - (16 - math.pi / 4)) < 2e-3, f"{area(floor):.4f}")
floor.select_set(True); ctx.view_layer.objects.active = floor; bpy.ops.nproto.untrim()
r = bpy.ops.nproto.trim(action="SPLIT", **tr)
parts = [o for o in bpy.data.objects if o.name.startswith("Floor") and store.is_nurbs(o)]
check("split -> 2 objects", len(parts) == 2 and abs(sum(area(o) for o in parts) - 16) < 2e-3,
      ", ".join(f"{o.name}={area(o):.3f}" for o in parts))
for o in parts:
    if o.name != "Floor": bpy.data.objects.remove(o)
floor.select_set(True); ctx.view_layer.objects.active = floor; bpy.ops.nproto.untrim()

# trim on a moved object (local/world handling)
f2 = make_floor("FloorMoved", loc=(10, 0, 0))
f2.location.z += 0.5; ctx.view_layer.update()
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(11, 0, 2)); cc2 = ctx.active_object
r = bpy.ops.nproto.trim(target="FloorMoved", cutters=ins((cc2, "S0")), ref_point=(11, 0, 0.5), normal=(0, 0, 1), action="KEEP")
ws = [f2.matrix_world @ v.co for v in f2.data.vertices]
cx = sum(v.x for v in ws) / len(ws); cz = sum(v.z for v in ws) / len(ws)
check("trim moved object", r == {'FINISHED'} and abs(area(f2) - math.pi / 4) < 2e-3 and abs(cx - 11) < 0.05 and abs(cz - 0.5) < 1e-4,
      f"area={area(f2):.4f} centre=({cx:.3f}, {cz:.3f})")

# trim by an intersecting surface
wl = poly("Wall_ln", [(2, -2, -1), (2, 2, -1)])
bpy.ops.nproto.extrude(inputs=ins((wl, "S0")), distance=3.0, direction="Z"); wall = ctx.active_object; wall.name = "Wall"
r = bpy.ops.nproto.trim(target="Floor", cutters=json.dumps([["Wall", "SURF"]]), ref_point=(1, 0, 0), normal=(0, 0, 1), action="KEEP")
check("trim by surface intersection", r == {'FINISHED'} and abs(area(floor) - 8) < 2e-3, f"{area(floor):.4f}")
floor.select_set(True); ctx.view_layer.objects.active = floor; bpy.ops.object.select_all(action='DESELECT')
floor.select_set(True); bpy.ops.nproto.untrim()

# fillet floor/wall, trimming both
wall_area0 = area(wall)
r = bpy.ops.nproto.fillet(surf_a="Floor", surf_b="Wall", radius=0.3, shape="CIRCULAR", trim_bases=True)
fil = ctx.active_object
check("fillet circular", r == {'FINISHED'} and abs(area(fil) - 0.5 * math.pi * 0.3 * 4) < 2e-3,
      f"fillet={area(fil):.4f} (expect {0.5*math.pi*0.3*4:.4f})")
fa, wa = area(floor), area(wall)
check("fillet trimmed both bases", abs(fa - 1.7 * 4) < 2e-3 and abs(wa - (wall_area0 - 0.3 * 4)) < 2e-3 or abs(wa - (wall_area0 - 1.3*4)) < 2e-3,
      f"floor={fa:.4f} wall={wa:.4f} (wall before {wall_area0:.2f})")
bpy.ops.ed.undo_push(message="x")
for o in (floor, wall):
    bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); bpy.ops.nproto.untrim()
r = bpy.ops.nproto.fillet(surf_a="Floor", surf_b="Wall", radius=0.3, shape="CHAMFER", trim_bases=False)
ch = ctx.active_object
check("chamfer", r == {'FINISHED'} and abs(area(ch) - 0.3 * math.sqrt(2) * 4) < 2e-3 and abs(area(floor) - 16) < 1e-6,
      f"chamfer={area(ch):.4f} (expect {0.3*math.sqrt(2)*4:.4f}), floor untouched={area(floor):.1f}")

# no solution / crash safety: parallel far-apart surfaces
far = make_floor("FarFloor", loc=(0, 0, 50))
try:
    r = bpy.ops.nproto.fillet(surf_a="Floor", surf_b="FarFloor", radius=0.3)
except RuntimeError as exc:
    r = str(exc)
check("fillet with no solution fails cleanly", r != {'FINISHED'}, str(r)[:90])

# loft from a trimmed B-rep edge (G0)
bpy.ops.nproto.trim(target="FloorMoved", cutters=ins((cc2, "S0")), ref_point=(11, 0, 0.5), normal=(0, 0, 1), action="KEEP")
eidx = 0
r = bpy.ops.nproto.loft(inputs=ins((f2, f"E{eidx}"), (cc2, "S0")))
check("loft from trimmed edge", r == {'FINISHED'} and len(ctx.active_object.data.polygons) > 0)

# --- Planar from picked edges / loops ------------------------------------
from blender_nurbs import ops as nops
from mathutils import Vector as V
pf = make_floor("PlanarSrc", loc=(0, 20, 0))
r = bpy.ops.nproto.planar(inputs=ins((pf, "U0"), (pf, "U1"), (pf, "V0"), (pf, "V1")))
po = ctx.active_object
check("planar from 4 surface edges", r == {'FINISHED'} and abs(area(po) - 16) < 1e-3, f"{area(po):.4f}")
check("chain natural surface edge -> 4 edges", nops.edge_loop_refs(pf, "V1") == ["U0", "U1", "V0", "V1"])
# trimmed hole: pick ONE edge of the hole, chain gives the hole loop
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 20, 1)); hc = ctx.active_object
bpy.ops.nproto.trim(target="PlanarSrc", cutters=ins((hc, "S0")), ref_point=(1, 20, 0), normal=(0, 0, 1), action="DELETE")
edges = store.display(pf)["edges"]
def centroid(pl): return V([sum(p[i] for p in pl) / len(pl) for i in range(3)])
hole_i = min(range(len(edges)), key=lambda i: (pf.matrix_world @ centroid(edges[i]) - V((1, 20, 0))).length if edges[i] else 1e9)
loop = nops.edge_loop_refs(pf, f"E{hole_i}")
outer_i = next(i for i in range(len(edges)) if f"E{i}" not in loop and edges[i])
outer = nops.edge_loop_refs(pf, f"E{outer_i}")
check("chain trimmed edges -> hole loop / outer loop", len(loop) >= 1 and len(outer) == 4 and not set(loop) & set(outer),
      f"hole={loop} outer={outer}")
r = bpy.ops.nproto.planar(inputs=ins(*[(pf, e) for e in loop]))
check("planar fills the hole", r == {'FINISHED'} and abs(area(ctx.active_object) - math.pi / 4) < 3e-3,
      f"{area(ctx.active_object):.4f}")
r = bpy.ops.nproto.planar(inputs=ins(*[(pf, e) for e in outer + loop]))
check("planar outer + hole loops", r == {'FINISHED'} and abs(area(ctx.active_object) - (16 - math.pi / 4)) < 3e-3,
      f"{area(ctx.active_object):.4f}")
# closure detection on world polylines
M = pf.matrix_world
polys = [[M @ V(p) for p in edges[int(e[1:])]] for e in outer]
check("closure: 4 outer edges closed, 3 open", nops.polylines_closed(polys, 1e-4) and not nops.polylines_closed(polys[:3], 1e-4))
# mixed: 3 surface edges + one curve closing the 4th side (curve on the U0? side)
src = make_floor("PlanarSrc2", loc=(0, 30, 0))
bb = [store.display(src)["nets"][0]["p"][i] for i in (0, -1)]
side = {}
for w in ("U0", "U1", "V0", "V1"):
    pl = [src.matrix_world @ V(p) for p in geom.boundary_points(store.get_surfaces(src)[0], w, 2)]
    side[w] = pl
w_last = "V1"
a_, b_ = side[w_last]
cl = poly("CloseLine", [tuple(a_), tuple(b_)])
r = bpy.ops.nproto.planar(inputs=ins((src, "U0"), (src, "U1"), (src, "V0"), (cl, "S0")))
check("planar from 3 edges + 1 curve", r == {'FINISHED'} and abs(area(ctx.active_object) - 16) < 1e-3, f"{area(ctx.active_object):.4f}")
# non-planar: lift the closing curve
cl.location.z += 1.0; ctx.view_layer.update()
try:
    r = bpy.ops.nproto.planar(inputs=ins((src, "U0"), (src, "U1"), (src, "V0"), (cl, "S0")))
except RuntimeError as exc:
    r = str(exc)
check("non-planar profile fails cleanly", r != {'FINISHED'}, str(r)[:80])

# --- Planar cap on a revolve rim (closed edge = its own loop) -------------
rp = nurbs("CapProf", [(31, 0, 0), (31.4, 0, 0.6), (30.7, 0, 1.4), (31.2, 0, 2.0)])
rp.location = (0, 0, 0)
bpy.ops.nproto.revolve(inputs=ins((rp, "S0")), axis="Z", pivot="WORLD", angle=360)
rv = ctx.active_object
s_rv = store.get_surfaces(rv)[0]
rims = {w: geom.boundary_points(s_rv, w, 8) for w in ("U0", "U1", "V0", "V1")}
top = max(rims, key=lambda w: sum(p[2] for p in rims[w]) / 8 if abs(rims[w][0][2] - rims[w][-1][2]) < 1e-9 else -1e9)
check("revolve rim chains to itself only", nops.edge_loop_refs(rv, top) == [top], f"{top} -> {nops.edge_loop_refs(rv, top)}")
r = bpy.ops.nproto.planar(inputs=ins((rv, top)))
check("planar cap on revolve rim", r == {'FINISHED'} and abs(area(ctx.active_object) - math.pi * 31.2 ** 2) < 0.5,
      f"{area(ctx.active_object):.2f} vs {math.pi*31.2**2:.2f}")

# --- Blender NURBS circle primitive: exact conversion + stale matrices ---
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.8, location=(40, 0, 1.5), rotation=(math.pi / 2, 0, 0))
nc = ctx.active_object
sp = nc.data.splines[0]
kc = features.input_curve(nc, "S0")
P = geom.sample_curve(kc, 200)
c = np.array([40, 0, 1.5])
rad = np.linalg.norm(P - c, axis=1)
check("NURBS circle primitive converts to an exact circle", rad.max() - rad.min() < 1e-5 and abs(rad.mean() - 0.8) < 1e-5,
      f"order={sp.order_u} cyclic={sp.use_cyclic_u} endpoint={sp.use_endpoint_u} bezier={sp.use_bezier_u} radius {rad.min():.6f}..{rad.max():.6f}")
r = bpy.ops.nproto.extrude(inputs=ins((nc, "S0")), distance=2.0, direction="NORMAL", symmetric=True)
ex = ctx.active_object
ex_area = store.get_brep(ex).area(1e-7)
check("extruded NURBS circle = cylinder, area 2*pi*r*h", abs(ex_area - 2 * math.pi * 0.8 * 4.0) < 1e-4,
      f"{ex_area:.6f} vs {2*math.pi*0.8*4:.6f}")
# stale runtime matrix (what F9 redo sees right after its undo): move without updating
nc.location = (45, 3, 2.5)
kc2 = features.input_curve(nc, "S0")
_s = geom.sample_curve(kc2, 400); ctr = 0.5 * (_s.min(axis=0) + _s.max(axis=0))
check("inputs use stored transform, not stale matrix_world", np.linalg.norm(ctr - np.array([45, 3, 2.5])) < 1e-4,
      f"centre={np.round(ctr, 4)} matrix_world.t={tuple(round(x, 2) for x in nc.matrix_world.translation)}")
ctx.view_layer.update()
# a Blender spline that Blender itself cannot evaluate gives a clear error
bad = nurbs("BadBez", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0), (2, 0, 0)])
bs = bad.data.splines[0]; bs.order_u = 4; bs.use_cyclic_u = True; bs.use_bezier_u = True
try:
    features.input_curve(bad, "S0"); msg = "no error"
except ValueError as exc:
    msg = str(exc)
check("unevaluable Blender spline -> clear error", "cannot evaluate" in msg, msg[:80])

# --- Trim on a closed revolve (vase): region picking + curve-normal projection
vp = nurbs("VaseProf", [(1.2, 0, 0), (1.5, 0, 0.8), (0.8, 0, 1.8), (1.1, 0, 3.0)])
vp.location = (60, 0, 0); ctx.view_layer.update()
bpy.ops.nproto.revolve(inputs=ins((vp, "S0")), axis="Z", pivot="OBJECT", angle=360)
vase = ctx.active_object; vase.name = "VaseT"
vase_area = store.get_brep(vase).area(1e-6)
def vcircle(name, x, r):
    bpy.ops.curve.primitive_nurbs_circle_add(radius=r, location=(x, 0, 1.5), rotation=(math.pi / 2, 0, 0))
    o = ctx.active_object; o.name = name; return o
small = vcircle("VaseSmall", 60.0, 0.3)          # inside the outline: hits front AND back wall
rad_front = 1.25  # vase radius near z=1.5 is ~1.1-1.3; pick point in front (y>0) on the wall
front_pt = (60.0, rad_front, 1.5)
def vtrim(action, cutter, pt, scope="PICKED"):
    vase.select_set(True); ctx.view_layer.objects.active = vase
    return bpy.ops.nproto.trim(target="VaseT", cutters=ins((cutter, "S0")), ref_point=pt,
                               normal=(0, 1, 0), view_dir=(0, -1, 0), action=action, scope=scope)
r = vtrim("DELETE", small, front_pt)
a1 = store.get_brep(vase).area(1e-6)
disc = vase_area - a1
check("vase: discard removes only the picked (front) region", r == {'FINISHED'} and 0.2 < disc < 0.4,
      f"removed {disc:.4f} (one wall disc ~0.29, both walls would be ~0.58)")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
r = vtrim("KEEP", small, front_pt)
check("vase: keep leaves only the picked disc", r == {'FINISHED'} and abs(store.get_brep(vase).area(1e-6) - disc) < 1e-3,
      f"kept {store.get_brep(vase).area(1e-6):.4f}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
side = vcircle("VaseSide", 61.25, 0.5)           # straddles the outline: one notch through the side
on_wall, _ = store.get_brep(vase).faces()[0].closest_point((61.25, 0.0, 1.5))
r = vtrim("DELETE", side, on_wall)
notch = vase_area - store.get_brep(vase).area(1e-6)
check("vase: circle across the outline cuts one notch (both seam halves)", r == {'FINISHED'} and 1.2 < notch < 2.0,
      f"removed {notch:.4f}, faces={store.get_brep(vase).face_count()}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# curve-normal is the default: same result as forcing world Y for these circles
r = bpy.ops.nproto.trim(target="VaseT", cutters=ins((small, "S0")), ref_point=front_pt, normal=(1, 0, 0),
                        view_dir=(1, 0, 0), action="DELETE", scope="PICKED")
check("projection defaults to the curve normal (ignores surface normal/view)",
      r == {'FINISHED'} and abs((vase_area - store.get_brep(vase).area(1e-6)) - disc) < 1e-3)
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()

# --- Trim: THROUGH scope, revolve seam, surface cutters -------------------
r = vtrim("DELETE", small, front_pt, scope="THROUGH")
both = vase_area - store.get_brep(vase).area(1e-6)
check("through: circle cuts both walls in one trim", r == {'FINISHED'} and abs(both - 2 * disc) < 2e-3,
      f"removed {both:.4f} (2 x {disc:.4f})")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# click OUTSIDE the circle -> the complement: keep only the two discs... with DELETE: remove everything outside
r = vtrim("KEEP", small, front_pt, scope="THROUGH")
check("through keep: keeps both discs", r == {'FINISHED'} and abs(store.get_brep(vase).area(1e-6) - 2 * disc) < 2e-3,
      f"kept {store.get_brep(vase).area(1e-6):.4f}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# seam: circle centred on the revolve seam (the profile plane y=0, x=61.x side faces +x) -> the
# projected hole is split by the seam into two faces; it must still be removed as ONE region
seam_c = vcircle("SeamCircle", 61.0, 0.3)
seam_c.rotation_euler = (0, math.pi / 2, 0); ctx.view_layer.update()   # normal along X: projects onto the seam side
on_seam, _ = store.get_brep(vase).faces()[0].closest_point((61.5, 0.0, 1.5))
r = vtrim("DELETE", seam_c, on_seam, scope="PICKED")
seam_hole = vase_area - store.get_brep(vase).area(1e-6)
check("seam: hole across the revolve seam removed completely", r == {'FINISHED'} and 0.2 < seam_hole < 0.4,
      f"removed {seam_hole:.4f} (full disc ~{math.pi*0.3**2:.3f} on a curved wall)")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# surface cutter: a tube along Y through the vase (exact imprint)
tc = vcircle("TubeC", 60.0, 0.4); tc.location.y = -3; ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((tc, "S0")), distance=6.0, direction="Y"); tube = ctx.active_object; tube.name = "TubeT"
tube_area0 = store.get_brep(tube).area(1e-6)
r = bpy.ops.nproto.trim(target="VaseT", cutters=json.dumps([["TubeT", "SURF"]]), ref_point=front_pt,
                        normal=(0, 1, 0), view_dir=(0, -1, 0), action="DELETE", scope="THROUGH")
holes = vase_area - store.get_brep(vase).area(1e-6)
check("surface cutter: tube punches both walls exactly", r == {'FINISHED'} and 0.6 < holes < 1.6,
      f"removed {holes:.4f}, faces={store.get_brep(vase).face_count()}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# trim the tube by the vase: discard the part inside the vase
inside_pt = (60.0, 0.4, 1.5)   # on the tube, inside the vase
r = bpy.ops.nproto.trim(target="TubeT", cutters=json.dumps([["VaseT", "SURF"]]), ref_point=inside_pt,
                        normal=(0, 0, 1), view_dir=(0, -1, 0), action="DELETE", scope="THROUGH")
t_after = store.get_brep(tube).area(1e-6)
check("surface cutter: tube trimmed back to the vase", r == {'FINISHED'} and 0.5 < tube_area0 - t_after < tube_area0 * 0.6,
      f"tube {tube_area0:.3f} -> {t_after:.3f}, faces={store.get_brep(tube).face_count()}")

# --- Split ---------------------------------------------------------------
def pieces(prefix):     # (with history the split surface stays as a hidden input: not a piece)
    return sorted((o for o in bpy.data.objects if o.name.startswith(prefix) and store.is_nurbs(o) and not o.hide_get()),
                  key=lambda o: o.name)
sf = make_floor("SpF", loc=(0, 80, 0))
swl = poly("SpW_ln", [(2, 78, -1), (2, 82, -1)])
bpy.ops.nproto.extrude(inputs=ins((swl, "S0")), distance=3.0, direction="Z"); sw = ctx.active_object; sw.name = "SpW"
bpy.ops.object.select_all(action='DESELECT'); sf.select_set(True); sw.select_set(True); ctx.view_layer.objects.active = sw
r = bpy.ops.nproto.split()
fl, wl = pieces("SpF"), pieces("SpW")
check("split: two surfaces split each other", r == {'FINISHED'} and len(fl) == 2 and len(wl) == 2,
      f"floor={[round(area(o), 3) for o in fl]} wall={[round(area(o), 3) for o in wl]}")
check("split: areas preserved", abs(sum(area(o) for o in fl) - 16) < 1e-3 and abs(sum(area(o) for o in wl) - 12) < 1e-3
      and sorted(round(area(o), 3) for o in wl) == [4.0, 8.0])
bpy.ops.object.select_all(action='DESELECT'); fl[-1].select_set(True); ctx.view_layer.objects.active = fl[-1]
bpy.ops.nproto.untrim()
check("split: untrim a piece restores the whole surface", abs(area(fl[-1]) - 16) < 1e-6)
# curve splits a surface
cf = make_floor("SpC", loc=(0, 90, 0))
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 90, 1)); scc = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); cf.select_set(True); scc.select_set(True)
r = bpy.ops.nproto.split()
cp = pieces("SpC")
check("split: curve splits a surface", r == {'FINISHED'} and len(cp) == 2 and abs(min(area(o) for o in cp) - math.pi / 4) < 2e-3,
      f"{[round(area(o), 4) for o in cp]}")
# vase + tube split each other (seam aware): vase -> remainder + 2 holes, tube -> inside + outside parts
for o in (vase, tube):
    bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); bpy.ops.nproto.untrim()
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); tube.select_set(True)
r = bpy.ops.nproto.split()
vp_, tp_ = pieces("VaseT"), pieces("TubeT")
check("split: vase and tube split each other", r == {'FINISHED'} and len(vp_) == 3 and len(tp_) >= 2
      and abs(sum(store.get_brep(o).area(1e-7) for o in vp_) - vase_area) < 2e-3,
      f"vase pieces={[round(area(o), 3) for o in vp_]} tube pieces={[round(area(o), 3) for o in tp_]}")

# split keeps history: the regions follow the split surface and what cut it
_sl = poly("SpH_ln", [(0, 120, 0), (0, 124, 0)])
bpy.ops.nproto.extrude(inputs=ins((_sl, "S0")), distance=4.0, direction="X"); _shf = ctx.active_object; _shf.name = "SpH"
_scl = poly("SpHW_ln", [(2, 119, -1), (2, 125, -1)])
bpy.ops.nproto.extrude(inputs=ins((_scl, "S0")), distance=3.0, direction="Z"); _shw = ctx.active_object; _shw.name = "SpHW"
bpy.ops.object.select_all(action='DESELECT'); _shf.select_set(True); _shw.select_set(True); ctx.view_layer.objects.active = _shf
r = bpy.ops.nproto.split()
_hp = pieces("SpH_")
_hp_hist = [features.history(o) for o in _hp]
_kept = _shf.hide_get() and not _shw.hide_get()       # (the wall is not split by the floor: it stays, as the cutter)
_a0 = sorted(round(area(o), 4) for o in _hp)
_shw.location.x += 1.0; ctx.view_layer.update()          # move the cutting wall: the regions follow
for o in _hp:
    features.rebuild(o, ctx, force=True)
_a1 = sorted(round(area(o), 4) for o in _hp)
check("split with history: the split surface is kept as a hidden input, every region is its own object with a Split "
      "feature, and moving the cutter re-splits them (8 + 8 -> 12 + 4)",
      r == {'FINISHED'} and len(_hp) == 2 and all(h is not None and h["op"] == "split_region" for h in _hp_hist)
      and _kept and _a0 == [8.0, 8.0] and _a1 == [4.0, 12.0], f"{[o.name for o in _hp]} {_a0} -> {_a1} kept {_kept}")

# --- Join / Unjoin ---------------------------------------------------------
def patch(name, p0, p1, d, dist):
    ln = poly(name + "_ln", [p0, p1])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=dist, direction=d)
    o = ctx.active_object; o.name = name; return o
JA = patch("JA", (0, 100, 0), (0, 104, 0), "X", 2.0)
JB = patch("JB", (2, 100, 0), (2, 104, 0), "X", 2.0)
JC = patch("JC", (2, 104, 0), (4, 104, 0), "Z", 2.0)
bpy.ops.object.select_all(action='DESELECT')
for o in (JA, JB, JC): o.select_set(True)
ctx.view_layer.objects.active = JA
r = bpy.ops.nproto.join()
J = ctx.active_object
jb = store.get_brep(J)
shared = sum(1 for e in jb.edges() if len(e.faces()) == 2)
check("join: 3 touching surfaces -> 1 joined object, 2 stitched edges, sources hidden", r == {'FINISHED'} and jb.face_count() == 3
      and shared == 2 and all(o.hide_get() for o in (JA, JB, JC)) and features.history(J)["op"] == "join",
      f"faces={jb.face_count()} shared={shared} area={jb.area(1e-6):.3f}")
jname = J.name
r = bpy.ops.nproto.restore_inputs('EXEC_DEFAULT')
check("restore inputs (history join): the 3 sources come back", r == {'FINISHED'} and jname not in bpy.data.objects
      and not any(o.hide_get() for o in (JA, JB, JC)))
# history off: the old one-shot behaviour (merge into active, sources removed; unjoin per face)
ctx.scene.nproto_settings.construction_history = False
bpy.ops.object.select_all(action='DESELECT')
for o in (JA, JB, JC): o.select_set(True)
ctx.view_layer.objects.active = JA
r = bpy.ops.nproto.join()
check("join without history merges into the active object", r == {'FINISHED'} and ctx.active_object.name == "JA"
      and "JB" not in bpy.data.objects and store.get_brep(JA).face_count() == 3)
r = bpy.ops.nproto.unjoin()
parts = [o for o in bpy.data.objects if o.name.startswith("JA") and store.is_nurbs(o)]
check("unjoin without history: one object per face", r == {'FINISHED'} and len(parts) == 3
      and abs(sum(store.get_brep(o).area(1e-6) for o in parts) - 20) < 1e-4)
ctx.scene.nproto_settings.construction_history = True
# gaps and tolerance
G1 = patch("G1", (0, 110, 0), (0, 114, 0), "X", 2.0)
G2 = patch("G2", (2.0005, 110, 0), (2.0005, 114, 0), "X", 2.0)
bpy.ops.object.select_all(action='DESELECT'); G1.select_set(True); G2.select_set(True); ctx.view_layer.objects.active = G1
r = bpy.ops.nproto.join()
check("join: 0.0005 gap stitched at default tolerance", r == {'FINISHED'} and store.get_brep(ctx.active_object).face_count() == 2)
G3 = patch("G3", (0, 120, 0), (0, 124, 0), "X", 2.0)
G4 = patch("G4", (2.01, 120, 0), (2.01, 124, 0), "X", 2.0)
bpy.ops.object.select_all(action='DESELECT'); G3.select_set(True); G4.select_set(True); ctx.view_layer.objects.active = G3
try:
    r = bpy.ops.nproto.join()
except RuntimeError as exc:
    r = str(exc)
check("join: 0.01 gap refused at default tolerance", r != {'FINISHED'} and "G4" in bpy.data.objects, str(r)[:80])
r = bpy.ops.nproto.join(tolerance=0.02)
check("join: 0.01 gap stitched with tolerance 0.02", r == {'FINISHED'} and store.get_brep(ctx.active_object).face_count() == 2)
# closed solid: tube + two planar caps
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 130, 0)); cb = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((cb, "S0")), distance=2.0, direction="Z"); tube2 = ctx.active_object
bpy.ops.nproto.planar(inputs=ins((cb, "S0"))); cap1 = ctx.active_object
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 130, 2)); ct = ctx.active_object
bpy.ops.nproto.planar(inputs=ins((ct, "S0"))); cap2 = ctx.active_object
bpy.ops.object.select_all(action='DESELECT')
for o in (cap1, cap2, tube2): o.select_set(True)
ctx.view_layer.objects.active = tube2
r = bpy.ops.nproto.join()
tube2 = ctx.active_object
sb = store.get_brep(tube2)
vol = sb.volume(1e-6) if sb.is_manifold_solid() else 0.0
check("join: tube + 2 caps = closed solid", r == {'FINISHED'} and sb.is_manifold_solid() and abs(abs(vol) - math.pi * 0.25 * 2) < 1e-3,
      f"solid={sb.is_manifold_solid()} volume={vol:.5f} (pi*r^2*h={math.pi*0.25*2:.5f})")

# --- Edge fillet -----------------------------------------------------------
def make_L(tag, loc=(0, 0, 0)):
    a = patch(tag + "a", (0, 140, 0), (0, 144, 0), "X", 2.0)
    b = patch(tag + "b", (2, 140, 0), (2, 144, 0), "Z", 2.0)
    bpy.ops.object.select_all(action='DESELECT'); a.select_set(True); b.select_set(True); ctx.view_layer.objects.active = a
    bpy.ops.nproto.join()
    o = ctx.active_object; o.name = tag
    o.location = loc; ctx.view_layer.update()
    return o
def fillet_all(o, **kw):
    return bpy.ops.nproto.edge_fillet(inputs=json.dumps([[o.name, f"E{i}"] for i in store.shared_edges(o)]), **kw)
def A(o): return store.get_brep(o).area(1e-7)
L1 = make_L("EF1")
check("edge fillet: L has one shared edge", len(store.shared_edges(L1)) == 1)
r = fillet_all(L1, shape="CIRCULAR", radius=0.3)
check("edge fillet circular", r == {'FINISHED'} and abs(A(L1) - (16 - 2.4 + 4 * math.pi * 0.3 / 2)) < 1e-3,
      f"{A(L1):.4f} vs {16 - 2.4 + 4*math.pi*0.3/2:.4f}")
L2 = make_L("EF2")
r = fillet_all(L2, shape="CHAMFER", radius=0.3)
check("edge chamfer", r == {'FINISHED'} and abs(A(L2) - (16 - 2.4 + 4 * 0.3 * math.sqrt(2))) < 1e-3, f"{A(L2):.4f}")
L3 = make_L("EF3")
r = fillet_all(L3, shape="BLEND", radius=0.3, continuity="2")
check("edge blend G2", r == {'FINISHED'} and A(L3) > 16 - 2.4 + 4 * math.pi * 0.3 / 2, f"{A(L3):.4f} (> circular)")
L4 = make_L("EF4")
r = fillet_all(L4, shape="CIRCULAR", radius=0.1, variable=True, end_radius=0.4)
check("edge fillet variable radius", r == {'FINISHED'} and store.get_brep(L4).face_count() == 3, f"area {A(L4):.4f}")
L5 = make_L("EF5", loc=(5, -3, 2))
r = fillet_all(L5, shape="CIRCULAR", radius=0.3)
check("edge fillet on a moved object", r == {'FINISHED'} and abs(A(L5) - (16 - 2.4 + 4 * math.pi * 0.3 / 2)) < 1e-3)
# solid: fillet only the top cap edge of the healed cylinder (tube2 from the join tests)
sb = store.get_brep(tube2)
v0 = abs(sb.volume(1e-7))
M = geom.world_matrix(tube2)
top_i = [i for i in store.shared_edges(tube2) if abs((M @ Vector(store.display(tube2)["edges"][i][20]))[2] - (M @ Vector((0, 0, 2)))[2]) < 1e-4]
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[tube2.name, f"E{i}"] for i in top_i]), shape="CIRCULAR", radius=0.1)
sb = store.get_brep(tube2)
check("edge fillet on a solid keeps it solid", r == {'FINISHED'} and len(top_i) == 1 and sb.is_manifold_solid()
      and 0 < v0 - abs(sb.volume(1e-7)) < 0.02, f"volume {v0:.5f} -> {abs(sb.volume(1e-7)):.5f}")
# a non-shared edge is refused cleanly
open_i = [i for i in range(len(store.display(L1)["edges"])) if i not in store.shared_edges(L1)][0]
try:
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[L1.name, f"E{open_i}"]]), radius=0.1)
except RuntimeError as exc:
    r = str(exc)
check("edge fillet refuses a boundary edge", r != {'FINISHED'}, str(r)[:80])

# --- CV editing -------------------------------------------------------------
import bmesh
from blender_nurbs import cvs as ncvs
from blender_nurbs import history as nhist
cvc = nurbs("CVcurve", [(0, 150, 0), (1, 150.5, 0), (2, 149.5, 0), (3, 150, 0)])
bpy.ops.nproto.extrude(inputs=ins((cvc, "S0")), distance=2.0, direction="Z"); cs = ctx.active_object; cs.name = "CVSurf"
# a G0 loft from the surface's top edge to a curve (has history -> must follow the CV edit)
topc = nurbs("CVtop", [(0, 151, 4), (1, 151, 4), (2, 151, 4), (3, 151, 4)])
s_cs = store.get_surfaces(cs)[0]
top_edge = max(("U0", "U1"), key=lambda w: geom.boundary_points(s_cs, w, 4)[0][2])
bpy.ops.nproto.loft(inputs=ins((cs, top_edge), (topc, "S0")))
lofted = ctx.active_object
area_before = store.get_brep(cs).area(1e-7)
bpy.ops.object.select_all(action='DESELECT'); cs.select_set(True); ctx.view_layer.objects.active = cs
r = bpy.ops.nproto.edit_cvs()
cage = ctx.active_object
d0 = ext.surface_cvs(store.get_surfaces(cs)[0])
check("edit CVs opens a cage in edit mode", r == {'FINISHED'} and ncvs.is_cage(cage) and cage.mode == 'EDIT'
      and len(cage.data.vertices) == d0["count_u"] * d0["count_v"], f"{d0['count_u']}x{d0['count_v']} CVs, mode={cage.mode}")
# move the last CV (a surface corner on the top edge) up by 0.5 like a G-grab would
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table()
moved_idx = d0["count_u"] * d0["count_v"] - 1
bm.verts[moved_idx].co.z += 0.5
bmesh.update_edit_mesh(cage.data)
ctx.view_layer.update(); ncvs._process()
new_corner = ext.surface_cvs(store.get_surfaces(cs)[0])["points"][moved_idx]
s_new = store.get_surfaces(cs)[0]
(u0, u1), (v0, v1) = s_new.u_domain(), s_new.v_domain()
corner = s_new.evaluate(u1, v1)
check("moving a cage vertex moves the CV and the surface corner live",
      abs(new_corner[2] - (d0["points"][moved_idx][2] + 0.5)) < 1e-9 and np.linalg.norm(np.array(corner) - np.array(new_corner)) < 1e-9,
      f"CV z {d0['points'][moved_idx][2]:.3f} -> {new_corner[2]:.3f}, corner {np.round(corner, 3)}")
loft_rev = lofted.get("nproto_rev")
ctx.view_layer.update(); nhist._process()
check("loft built from the surface follows the CV edit (history)", lofted.get("nproto_rev") != loft_rev)
r = bpy.ops.nproto.cvs_done()
check("Done keeps the edit and removes the cage", r == {'FINISHED'} and ncvs.cage_for(cs) is None
      and abs(store.get_brep(cs).area(1e-7) - area_before) > 1e-4 and store.is_natural(cs))
# Tab out of Edit Mode on the cage = Done
bpy.ops.object.select_all(action='DESELECT'); cs.select_set(True); ctx.view_layer.objects.active = cs
bpy.ops.nproto.edit_cvs(); cage = ctx.active_object; cname = cage.name
ctx.view_layer.update()
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table()
bm.verts[0].co.z -= 0.25; bmesh.update_edit_mesh(cage.data)
ctx.view_layer.update(); ncvs._process()
a_tab = store.get_brep(cs).area(1e-7)
bpy.ops.object.mode_set(mode='OBJECT')        # Tab
ctx.view_layer.update()
pending = cname in ncvs._finish
ncvs._finish_cages()
check("Tab out of the CV cage keeps the edit and removes the cage", pending and cname not in bpy.data.objects
      and ncvs.cage_for(cs) is None and ctx.view_layer.objects.active == cs
      and abs(store.get_brep(cs).area(1e-7) - a_tab) < 1e-9, f"pending={pending}")
# Cancel restores
bpy.ops.object.select_all(action='DESELECT'); cs.select_set(True); ctx.view_layer.objects.active = cs
area_mid = store.get_brep(cs).area(1e-7)
bpy.ops.nproto.edit_cvs(); cage = ctx.active_object
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table(); bm.verts[0].co.x -= 1.0; bmesh.update_edit_mesh(cage.data)
ctx.view_layer.update(); ncvs._process()
changed = abs(store.get_brep(cs).area(1e-7) - area_mid) > 1e-4
r = bpy.ops.nproto.cvs_cancel()
check("Cancel restores the surface", r == {'FINISHED'} and changed and abs(store.get_brep(cs).area(1e-7) - area_mid) < 1e-7
      and ncvs.cage_for(cs) is None)
# trimmed surfaces: trims stay fixed in UV while the CVs move
def edge_gap(o):
    b = store.get_brep(o); worst = 0.0
    for e in b.edges():
        t0, t1 = e.parameter_range()
        for k in range(9):
            p = tuple(e.point_at(t0 + (t1 - t0) * k / 8))
            worst = max(worst, min(f.closest_point(p)[1] for f in b.faces()))
    return worst
bpy.ops.object.select_all(action='DESELECT'); floor.select_set(True); ctx.view_layer.objects.active = floor
bpy.ops.nproto.trim(target="Floor", cutters=ins((cc, "S0")), ref_point=(1, 0, 0), normal=(0, 0, 1), action="DELETE")
holed_area = store.get_brep(floor).area(1e-7)
r = bpy.ops.nproto.edit_cvs(); cage = ctx.active_object
ok_open = r == {'FINISHED'} and ncvs.is_cage(cage)
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table()
bm.verts[len(bm.verts) - 1].co.z += 1.0          # lift one corner: the floor tilts/warps
bmesh.update_edit_mesh(cage.data); ctx.view_layer.update(); ncvs._process()
fb = store.get_brep(floor)
hole_mid = [np.array(e.point_at(sum(e.parameter_range()) / 2)) for e in fb.edges()]
hole_z = max(abs(p[2]) for p in hole_mid if abs(p[0] - 1) < 0.6 and abs(p[1]) < 0.6) if hole_mid else 0
check("trimmed CV edit: hole stays, edges ride on the new surface", ok_open and fb.face_count() == 1 and edge_gap(floor) < 1e-6
      and hole_z > 0.01 and not store.is_natural(floor),
      f"faces={fb.face_count()} gap={edge_gap(floor):.1e} hole edge lifted to z={hole_z:.3f}")
r = bpy.ops.nproto.cvs_done()
edited_untrimmed_ok = store.PROP_UNTRIM in floor
bpy.ops.nproto.untrim()
plain = store.get_brep(floor).area(1e-7)
check("trimmed CV edit: Done, then Untrim gives the EDITED plain surface", r == {'FINISHED'} and edited_untrimmed_ok
      and plain > 16 + 1e-3, f"untrimmed area {plain:.4f} (flat was 16)")
# cancel on a trimmed surface restores it exactly
bpy.ops.nproto.trim(target="Floor", cutters=ins((cc, "S0")), ref_point=(1, 0, 0), normal=(0, 0, 1), action="DELETE")
before = floor[store.PROP_BREP]; before_untrim = floor.get(store.PROP_UNTRIM)
bpy.ops.nproto.edit_cvs(); cage = ctx.active_object
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table(); bm.verts[0].co.z -= 2.0
bmesh.update_edit_mesh(cage.data); ctx.view_layer.update(); ncvs._process()
r = bpy.ops.nproto.cvs_cancel()
check("trimmed CV edit: Cancel restores brep and untrim data", r == {'FINISHED'} and floor[store.PROP_BREP] == before
      and floor.get(store.PROP_UNTRIM) == before_untrim and not store.is_natural(floor))
bpy.ops.nproto.untrim()
# vase with a hole across the revolve seam: scale the whole CV net about the axis
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); ctx.view_layer.objects.active = vase
if store.PROP_UNTRIM in vase: bpy.ops.nproto.untrim()
on_seam, _ = store.get_brep(vase).faces()[0].closest_point((61.5, 0.0, 1.5))
r = vtrim("DELETE", seam_c, on_seam, scope="PICKED")
a_before = store.get_brep(vase).area(1e-7)
faces_before = store.get_brep(vase).face_count()
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); ctx.view_layer.objects.active = vase
r2 = bpy.ops.nproto.edit_cvs(); cage = ctx.active_object
M = geom.world_matrix(cage); Mi = M.inverted()
bm = bmesh.from_edit_mesh(cage.data)
for v in bm.verts:
    w = M @ v.co
    w.x = 60 + (w.x - 60) * 1.15; w.y = w.y * 1.15
    v.co = Mi @ w
bmesh.update_edit_mesh(cage.data); ctx.view_layer.update(); ncvs._process()
bpy.ops.nproto.cvs_done()
vb = store.get_brep(vase)
check("trimmed CV edit on a revolve (hole across the seam)", r == {'FINISHED'} and r2 == {'FINISHED'} and vb.face_count() == faces_before
      and edge_gap(vase) < 1e-6 and 1.10 < vb.area(1e-7) / a_before < 1.20,
      f"faces {faces_before}, area {a_before:.4f} -> {vb.area(1e-7):.4f} (x{vb.area(1e-7)/a_before:.3f}), gap {edge_gap(vase):.1e}")
bpy.ops.nproto.untrim()

# --- Blend curve ---------------------------------------------------------
def kappa(d):
    c1, c2 = np.array(d["tangent"]), np.array(d["curvature"])
    return np.cross(c1, c2) / (np.linalg.norm(c1) ** 3)
BA = nurbs("BlendA", [(0, 160, 0), (1, 160.8, 0.2), (2, 159.4, 0.4), (3, 160, 0)])
BB = nurbs("BlendB", [(5, 161, 1.0), (6, 162, 0.6), (7, 160.5, 0.9), (8, 161.5, 1.5)])
smk = _kl()
for cont in ("G0", "G1", "G2"):
    bpy.ops.object.select_all(action='DESELECT')
    r = bpy.ops.nproto.blend_curve(inputs=ins((BA, "S0"), (BB, "S0")), end_a=1, end_b=0, cont_a=cont, cont_b=cont)
    bc = ctx.active_object
    kb = features.input_curve(bc, "S0")
    ka_ = features.input_curve(BA, "S0"); kbb = features.input_curve(BB, "S0")
    t0, t1 = kb.parameter_range()
    s0, s1 = smk.evaluate_curve(kb, t0), smk.evaluate_curve(kb, t1)
    ea = smk.evaluate_curve(ka_, ka_.parameter_range()[1]); eb = smk.evaluate_curve(kbb, kbb.parameter_range()[0])
    g0 = np.linalg.norm(np.array(s0["point"]) - ea["point"]) + np.linalg.norm(np.array(s1["point"]) - eb["point"])
    u = lambda v: np.array(v) / np.linalg.norm(v)
    ang_a = math.degrees(math.acos(np.clip(np.dot(u(s0["tangent"]), u(ea["tangent"])), -1, 1)))
    ang_b = math.degrees(math.acos(np.clip(np.dot(u(s1["tangent"]), u(eb["tangent"])), -1, 1)))
    dk = np.linalg.norm(kappa(s0) - kappa(ea)) + np.linalg.norm(kappa(s1) - kappa(eb))
    sp = bc.data.splines[0]
    if cont == "G0":
        okc = g0 < 1e-5 and sp.order_u == 2
    elif cont == "G1":
        okc = g0 < 1e-5 and ang_a < 1e-3 and ang_b < 1e-3 and sp.order_u == 4
    else:
        okc = g0 < 1e-5 and ang_a < 1e-3 and ang_b < 1e-3 and dk < 1e-3 and sp.order_u == 6
    check(f"blend curve {cont}", r == {'FINISHED'} and okc,
          f"order={sp.order_u} gap={g0:.1e} tangent angles=({ang_a:.4f},{ang_b:.4f})deg curvature diff={dk:.1e}")
# weight = tangent length (|B'(0)| = w * L)
bpy.ops.nproto.blend_curve(inputs=ins((BA, "S0"), (BB, "S0")), end_a=1, end_b=0, cont_a="G1", cont_b="G1", w_a=2.0)
bc = ctx.active_object; kb = features.input_curve(bc, "S0")
L = np.linalg.norm(np.array(ka_.end_point()) - np.array(kbb.start_point()))
d0 = np.linalg.norm(smk.evaluate_curve(kb, kb.parameter_range()[0])["tangent"]) / (kb.parameter_range()[1] - kb.parameter_range()[0])
check("blend curve weight sets the tangent length", abs(d0 - 2.0 * L) < 1e-4, f"|B'(0)|={d0:.4f} vs 2*L={2*L:.4f}")
# history: move curve B -> the blend follows
start_before = tuple(bc.data.splines[0].points[-1].co)
BB.location.z += 0.5; ctx.view_layer.update(); history._process()
check("blend curve follows its inputs (history)", tuple(bc.data.splines[0].points[-1].co) != start_before)
# selection-based: the closest pair of ends is used
bpy.ops.object.select_all(action='DESELECT'); BA.select_set(True); BB.select_set(True); ctx.view_layer.objects.active = BB
r = bpy.ops.nproto.blend_curve('INVOKE_DEFAULT')
bc2 = ctx.active_object
p_first = np.array(bc2.matrix_world @ Vector(bc2.data.splines[0].points[0].co[:3]))
check("blend curve from selection picks the nearest ends", r == {'FINISHED'} and np.linalg.norm(p_first - np.array(ka_.end_point())) < 1e-5)
# a blend curve is a normal curve input: loft from it
r = bpy.ops.nproto.extrude(inputs=ins((bc2, "S0")), distance=1.0, direction="Z")
check("blend curve usable as a tool input (extrude)", r == {'FINISHED'})

# --- Primitives + live Feature editor -------------------------------------
ctx.scene.cursor.location = (0, 200, 0)
def prim(kind, **kw):
    r = bpy.ops.nproto.add_primitive(kind=kind, **kw)
    return r, ctx.active_object
def Ar(o): return store.get_brep(o).area(1e-7)
def Vol(o):
    b = store.get_brep(o); return abs(b.volume(1e-7)) if b.is_manifold_solid() else 0.0
def radii(o, c=(0, 200, 0)):
    P = geom.sample_curve(features.input_curve(o, "S0"), 400)
    d = np.linalg.norm(P - np.array(c), axis=1); return d.min(), d.max()
r, o = prim("CIRCLE", radius=1.5, degree="0"); lo, hi = radii(o)
check("prim circle exact", r == {'FINISHED'} and hi - lo < 1e-6 and abs(lo - 1.5) < 1e-6, f"{lo:.7f}..{hi:.7f}")
r, o = prim("CIRCLE", radius=1.5, degree="3", spans_u=8); lo, hi = radii(o)
check("prim circle degree 3 / 8 spans", r == {'FINISHED'} and hi - lo < 0.02 and o.data.splines[0].order_u == 4, f"{lo:.4f}..{hi:.4f}")
r, o = prim("ARC", radius=2.0, angle=135.0, degree="0"); lo, hi = radii(o)
kc = features.input_curve(o, "S0")
check("prim arc exact 135deg", r == {"FINISHED"} and hi - lo < 1e-5 and abs(kc.length(1e-7) - 2.0 * math.radians(135)) < 1e-5,
      f"length {kc.length(1e-7):.6f} vs {2*math.radians(135):.6f}")
r, o = prim("SPHERE", radius=1.0, degree="0")
check("prim sphere exact", r == {'FINISHED'} and abs(Ar(o) - 4 * math.pi) < 1e-4, f"area {Ar(o):.6f} vs {4*math.pi:.6f}")
sph = o
r, o = prim("SPHERE", radius=1.0, degree="3", spans_u=8, spans_v=4)
check("prim sphere degree 3", r == {'FINISHED'} and abs(Ar(o) - 4 * math.pi) / (4 * math.pi) < 0.01, f"area {Ar(o):.4f}")
r, o = prim("CYLINDER", radius=0.5, height=2.0, degree="0", caps=True)
check("prim cylinder exact + caps = solid", r == {'FINISHED'} and abs(Vol(o) - math.pi * 0.25 * 2) < 1e-4, f"volume {Vol(o):.6f}")
r, o = prim("CONE", radius=1.0, radius2=0.0, height=3.0, degree="0", caps=True)
check("prim cone exact + cap = solid", r == {'FINISHED'} and abs(Vol(o) - math.pi * 3 / 3) < 1e-3, f"volume {Vol(o):.5f} vs {math.pi:.5f}")
r, o = prim("TORUS", radius=2.0, radius2=0.5, degree="0")
check("prim torus exact", r == {'FINISHED'} and abs(Ar(o) - 4 * math.pi ** 2 * 2 * 0.5) < 1e-3, f"area {Ar(o):.5f}")
r, o = prim("BOX", width=2, depth=3, height=4)
check("prim box = solid", r == {'FINISHED'} and abs(Vol(o) - 24) < 1e-6 and store.get_brep(o).face_count() == 6, f"volume {Vol(o):.6f}")
r, o = prim("PLANE", width=3, depth=2, degree="3", spans_u=2, spans_v=2)
check("prim plane degree 3", r == {'FINISHED'} and abs(Ar(o) - 6) < 1e-6)
r, o = prim("DISK", radius=1.0, degree="0")
check("prim disk exact", r == {'FINISHED'} and abs(Ar(o) - math.pi) < 1e-5)
for kind in ("LINE", "RECTANGLE", "POLYGON", "HELIX", "ELLIPSE"):
    r, o = prim(kind)
    check(f"prim {kind.lower()}", r == {'FINISHED'} and o.type == 'CURVE' and features.input_curve(o, "S0") is not None)
# feature editor: change the exact sphere's radius / degree live
sph.nproto_feature.radius = 2.0
check("feature edit: sphere radius 1 -> 2", abs(Ar(sph) - 16 * math.pi) < 1e-3, f"area {Ar(sph):.5f}")
sph.nproto_feature.degree = "3"
ext_ = _kl().blender_ext
check("feature edit: sphere degree exact -> 3", ext_.surface_cvs(store.get_surfaces(sph)[0])["degree_u"] == 3)
# feature editor on a blend curve: weight change rebuilds it and its dependent extrusion
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.blend_curve(inputs=ins((BA, "S0"), (BB, "S0")), end_a=1, end_b=0, cont_a="G2", cont_b="G2")
bcf = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((bcf, "S0")), distance=1.0, direction="Z"); dep = ctx.active_object
pts0 = [tuple(p.co) for p in bcf.data.splines[0].points]; rev0 = dep.get("nproto_rev")
check("feature panel synced for blend curve", bcf.nproto_feature.cont_a == "G2" and abs(bcf.nproto_feature.w_a - 1.0) < 1e-9)
bcf.nproto_feature.w_a = 2.5
ctx.view_layer.update(); history._process()
check("feature edit: blend weight rebuilds curve and its dependents",
      [tuple(p.co) for p in bcf.data.splines[0].points] != pts0 and dep.get("nproto_rev") != rev0)
bcf.nproto_feature.cont_b = "G1"
check("feature edit: blend continuity G2 -> G1 (degree 5 -> 4)", bcf.data.splines[0].order_u == 5)

# --- History chains: Trim / Split / Surface Fillet / Join / Edge Fillet ---
def settle():
    for _ in range(4):
        ctx.view_layer.update(); history._process()
def A7(o): return store.get_brep(o).area(1e-7)
def hole_center(o):
    b = store.get_brep(o)
    best = None
    for e in b.edges():
        P = np.array([e.point_at(t) for t in np.linspace(*e.parameter_range(), 24)])
        if np.linalg.norm(P[0] - P[-1]) < 1e-6:   # the closed hole edge
            best = P.mean(axis=0)
    return best
# 1) extrude (history) -> trim with a circle (history step)
hl = poly("HTline", [(0, 250, 0), (0, 254, 0)])
bpy.ops.nproto.extrude(inputs=ins((hl, "S0")), distance=4.0, direction="X"); hs = ctx.active_object; hs.name = "HTsurf"
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 251, 1)); hc = ctx.active_object; hc.name = "HTcut"
r = bpy.ops.nproto.trim(target="HTsurf", cutters=ins((hc, "S0")), ref_point=(1, 251, 0), normal=(0, 0, 1), action="DELETE")
st = features.stack(hs)
check("trim records a history step on a history surface", r == {'FINISHED'} and len(st) == 1 and st[0]["op"] == "trim"
      and features.history(hs)["op"] == "extrude" and abs(A7(hs) - (16 - math.pi / 4)) < 2e-3)
hc.location.x += 1.5; settle()
c = hole_center(hs)
check("moving the cutting curve moves the hole", c is not None and abs(c[0] - 2.5) < 0.05 and abs(A7(hs) - (16 - math.pi / 4)) < 2e-3,
      f"hole centre x={c[0] if c is not None else None}")
hl.location.y += 0.5; settle()
bb = store.get_brep(hs).bounding_box()
check("moving the extrude's curve rebuilds surface + trim", abs(bb[0][1] - 250.5) < 1e-3 and abs(A7(hs) - (16 - math.pi / 4)) < 2e-3,
      f"ymin={bb[0][1]:.3f}")
# remove the trim step -> plain extrusion again
bpy.ops.object.select_all(action='DESELECT'); hs.select_set(True); ctx.view_layer.objects.active = hs
r = bpy.ops.nproto.remove_step(index=0)
check("remove history step", r == {'FINISHED'} and features.PROP_STACK not in hs and abs(A7(hs) - 16) < 1e-6
      and store.is_natural(hs))
# 2) split action keeps history on both pieces
r = bpy.ops.nproto.trim(target="HTsurf", cutters=ins((hc, "S0")), ref_point=(2.5, 251.5, 0), normal=(0, 0, 1), action="SPLIT")
piece = bpy.data.objects.get("HTsurf_split")
check("split action: both pieces carry history", r == {'FINISHED'} and piece is not None and len(features.stack(piece)) == 1
      and features.stack(piece)[0]["params"]["action"] == "DELETE")
hc.location.x -= 1.0; settle()
check("split pieces both follow the cutter", abs(A7(hs) + A7(piece) - 16) < 2e-3 and abs(A7(hs) - math.pi / 4) < 2e-3
      and abs(0.5 * (store.get_brep(hs).bounding_box()[0][0] + store.get_brep(hs).bounding_box()[1][0]) - 1.5) < 0.05,
      f"{A7(hs):.4f} + {A7(piece):.4f}")
bpy.data.objects.remove(piece)
bpy.ops.object.select_all(action='DESELECT'); hs.select_set(True); ctx.view_layer.objects.active = hs
r = bpy.ops.nproto.untrim()
check("untrim removes trim steps", r == {'FINISHED'} and features.PROP_STACK not in hs and abs(A7(hs) - 16) < 1e-6)
# 3) surface fillet with history: fillet surface + base trims follow the wall's curve
fl_ = poly("HFfloor_ln", [(0, 260, 0), (0, 264, 0)])
bpy.ops.nproto.extrude(inputs=ins((fl_, "S0")), distance=4.0, direction="X"); hfl = ctx.active_object; hfl.name = "HFfloor"
wl_ = poly("HFwall_ln", [(2, 260, -1), (2, 264, -1)])
bpy.ops.nproto.extrude(inputs=ins((wl_, "S0")), distance=3.0, direction="Z"); hwl = ctx.active_object; hwl.name = "HFwall"
r = bpy.ops.nproto.fillet(surf_a="HFfloor", surf_b="HFwall", radius=0.3, trim_bases=True)
hf = ctx.active_object
check("surface fillet with history", r == {'FINISHED'} and features.history(hf)["op"] == "surface_fillet"
      and features.stack(hfl)[0]["op"] == "fillet_trim" and features.stack(hwl)[0]["op"] == "fillet_trim"
      and abs(A7(hfl) - 6.8) < 2e-3, f"floor {A7(hfl):.4f}")
wl_.location.x += 0.5; settle()
fb_ = store.get_brep(hf).bounding_box()
check("moving the wall moves the fillet and re-trims the floor", abs(A7(hfl) - (2.2 * 4)) < 2e-3
      and abs(fb_[0][0] - 2.2) < 0.02, f"floor {A7(hfl):.4f} (expect 8.8), fillet xmin {fb_[0][0]:.3f}")
# 4) chain: curves -> extrudes -> join (history) -> edge fillet (step)
ja_ = poly("CHa_ln", [(0, 270, 0), (0, 274, 0)])
bpy.ops.nproto.extrude(inputs=ins((ja_, "S0")), distance=2.0, direction="X"); cha = ctx.active_object
jb_ = poly("CHb_ln", [(2, 270, 0), (2, 274, 0)])
bpy.ops.nproto.extrude(inputs=ins((jb_, "S0")), distance=2.0, direction="Z"); chb = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); cha.select_set(True); chb.select_set(True); ctx.view_layer.objects.active = cha
bpy.ops.nproto.join(); chj = ctx.active_object
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[chj.name, f"E{i}"] for i in store.shared_edges(chj)]), radius=0.3)
a_fil = A7(chj)
check("edge fillet on a history join records a step", r == {'FINISHED'} and features.stack(chj)[0]["op"] == "edge_fillet"
      and abs(a_fil - (16 - 2.4 + 4 * math.pi * 0.3 / 2)) < 1e-3)
# move both source curves together (the joint stays closed): the whole chain follows
ja_.location.y += 1.0; jb_.location.y += 1.0; settle()
bbj = store.get_brep(chj).bounding_box()
check("curve move ripples through extrude -> join -> edge fillet", abs(bbj[0][1] - 271.0) < 1e-3
      and abs(A7(chj) - a_fil) < 1e-3 and store.get_brep(chj).face_count() == 3, f"ymin {bbj[0][1]:.3f}")
chb_rev = chj.get("nproto_rev")
bpy.ops.object.select_all(action='DESELECT')
cha.hide_set(False); chb.hide_set(False)
# lengthen the wall via its feature panel: height 2 -> 3, the joined + filleted result follows
chb.nproto_feature.distance = 3.0; settle()
check("feature edit on a hidden join source updates the filleted join", chj.get("nproto_rev") != chb_rev
      and abs(A7(chj) - (8 + 12 - 2.4 + 4 * math.pi * 0.3 / 2)) < 1e-3, f"area {A7(chj):.4f}")

# 5) editable history steps: radius of the edge fillet step, mute, tree
from blender_nurbs import history_ui
area_r3 = A7(chj)
ctx.view_layer.objects.active = chj
history_ui.sync_step(chj, 0)
chj.nproto_step.radius = 0.5
check("editing a step's radius rebuilds the fillet", abs(A7(chj) - (8 + 12 - 4.0 + 4 * math.pi * 0.5 / 2)) < 1e-3
      and features.stack(chj)[0]["params"]["radius"] == 0.5, f"area {A7(chj):.4f}")
r = bpy.ops.nproto.mute_step(index=0)
check("muting the step removes the fillet (step kept)", r == {'FINISHED'} and store.get_brep(chj).face_count() == 2
      and len(features.stack(chj)) == 1 and abs(A7(chj) - 20) < 1e-3, f"area {A7(chj):.4f}")
bpy.ops.nproto.mute_step(index=0)
check("un-muting re-applies it", store.get_brep(chj).face_count() == 3)
kids = [o.name for o, _ in history_ui._children(chj)]
check("tree: the join lists both sources; the source is used by the join",
      sorted(kids) == sorted([cha.name, chb.name]) and chj in history_ui.dependents(cha, ctx.scene)
      and "Join" in history_ui._summary(chj) and "Edge" in history_ui._summary(chj), f"{kids} {[cha.name, chb.name]} {[d.name for d in history_ui.dependents(cha, ctx.scene)]}")
import importlib
importlib.reload(features)
check("feature builders survive an add-on module reload", "join" in features.BUILDERS and "prim_cylinder" in features.BUILDERS)
cha.hide_set(True)
bpy.ops.nproto.history_focus(name=cha.name, root=chj.name)
check("tree: clicking a hidden input keeps the tree on the join and makes the input editable",
      ctx.view_layer.objects.active == cha and history_ui.tree_root(ctx, cha) == chj and cha.hide_get())
bpy.ops.nproto.history_focus(name=chj.name, root=chj.name)
# Unjoin keeps the current (filleted) shape; Restore Inputs is the separate undo-the-join
a_now = A7(chj)
bpy.ops.object.select_all(action='DESELECT'); chj.select_set(True); ctx.view_layer.objects.active = chj
jn = chj.name
r = bpy.ops.nproto.unjoin()
pieces = [o for o in bpy.data.objects if (o.name == jn or o.name.startswith(jn + "_face")) and store.is_nurbs(o)]
check("unjoin keeps the fillet: one object per face, same total area", r == {'FINISHED'} and len(pieces) == 3
      and abs(sum(A7(o) for o in pieces) - a_now) < 1e-3 and not features.has_history(chj),
      f"{len(pieces)} pieces, {sum(A7(o) for o in pieces):.4f} vs {a_now:.4f}")

# --- Sweep / Pipe / Boundary / curves / Rebuild / Continuity ---------------
from blender_nurbs import sweeps as nsw
def circle_at(name, r, loc, rot=(0, 0, 0)):
    bpy.ops.curve.primitive_nurbs_circle_add(radius=r, location=loc, rotation=rot)
    o = ctx.active_object; o.name = name; return o
def SA(o): return store.get_brep(o).area(1e-7)
Y0 = 300
# 1-rail straight: circle r=0.3 swept 5 along Z -> cylinder
pr = circle_at("SWprof", 0.3, (0, Y0, 0))
rl = poly("SWrail", [(0, Y0, 0), (0, Y0, 5)])
r = bpy.ops.nproto.sweep(inputs=ins((pr, "S0"), (rl, "S0")))
sw = ctx.active_object
check("sweep 1 rail (straight)", r == {'FINISHED'} and abs(SA(sw) - 2 * math.pi * 0.3 * 5) < 2e-3, f"{SA(sw):.5f} vs {2*math.pi*0.3*5:.5f}")
# curved rail: quarter arc radius 3 in the XZ plane, starting at the profile
rc = nurbs("SWarc", [(0, Y0 + 5, 0), (0, Y0 + 5, 1.6568542), (1.3431458, Y0 + 5, 3.0), (3, Y0 + 5, 3)])
rc.data.splines[0].order_u = 3
sp0 = rc.data.splines[0]; sp0.points.add(0)
# exact quarter circle as rational quadratic
sp0.points[0].co = (0, Y0 + 5, 0, 1); sp0.points[1].co = (0, Y0 + 5, 3, 0.70710678); sp0.points[2].co = (3, Y0 + 5, 3, 1)
for i in range(len(sp0.points) - 1, 2, -1):
    pass
rc2 = nurbs("SWarc2", [(0, Y0 + 5, 0), (0, Y0 + 5, 3), (3, Y0 + 5, 3)])
rc2.data.splines[0].order_u = 3
rc2.data.splines[0].points[1].co = (0, Y0 + 5, 3, 0.70710678)
pr2 = circle_at("SWprof2", 0.3, (0, Y0 + 5, 0))
r = bpy.ops.nproto.sweep(inputs=ins((pr2, "S0"), (rc2, "S0")))
sw2 = ctx.active_object
expect = 2 * math.pi * 0.3 * (3 * math.pi / 2)
check("sweep 1 rail (curved, Pappus)", r == {'FINISHED'} and abs(SA(sw2) - expect) / expect < 2e-3, f"{SA(sw2):.4f} vs {expect:.4f}")
# end scale: frustum r 0.3 -> 0.6 over 5
r = bpy.ops.nproto.sweep(inputs=ins((pr, "S0"), (rl, "S0")), scale_end=2.0)
sl = math.sqrt(5 ** 2 + 0.3 ** 2)
check("sweep end scale (frustum)", r == {'FINISHED'} and abs(SA(ctx.active_object) - math.pi * (0.3 + 0.6) * sl) < 3e-3,
      f"{SA(ctx.active_object):.5f} vs {math.pi*(0.3+0.6)*sl:.5f}")
# twist: rectangle profile, 90 deg twist -> the end section is rotated
rect = poly("SWrect", [(-0.5, Y0 + 10 - 0.2, 0), (0.5, Y0 + 10 - 0.2, 0), (0.5, Y0 + 10 + 0.2, 0), (-0.5, Y0 + 10 + 0.2, 0), (-0.5, Y0 + 10 - 0.2, 0)])
rl3 = poly("SWrail3", [(0, Y0 + 10, 0), (0, Y0 + 10, 4)])
r = bpy.ops.nproto.sweep(inputs=ins((rect, "S0"), (rl3, "S0")), twist=90.0)
bt = store.get_brep(ctx.active_object).bounding_box()
check("sweep twist 90", r == {'FINISHED'} and bt[1][1] - bt[0][1] > 0.9, f"y extent {bt[1][1]-bt[0][1]:.3f} (untwisted 0.4)")
# two rails: diverging lines, an arc profile spanning them
r1 = poly("SW2r1", [(0, Y0 + 15, 0), (0, Y0 + 15 + 6, 0)])
r2 = poly("SW2r2", [(2, Y0 + 15, 0), (4, Y0 + 15 + 6, 0)])
arcp = nurbs("SW2prof", [(0, Y0 + 15, 0), (1, Y0 + 15, 1.5), (2, Y0 + 15, 0)]); arcp.data.splines[0].order_u = 3
r = bpy.ops.nproto.sweep(inputs=ins((arcp, "S0"), (r1, "S0"), (r2, "S0")))
s2r = store.get_surfaces(ctx.active_object)[0]
(u0, u1), (v0, v1) = s2r.u_domain(), s2r.v_domain()
end_gap = max(np.linalg.norm(np.array(s2r.evaluate(u1, v)) - np.array(p)) for v, p in ((v0, (0, Y0 + 21, 0)), (v1, (4, Y0 + 21, 0))))
check("sweep 2 rails: edges ride on both rails", r == {'FINISHED'} and end_gap < 1e-4, f"end gap {end_gap:.2e}")
# tangent sweep along the top edge of a curved wall (G1 to the wall)
wc = nurbs("TWcurve", [(0, Y0 + 25, 0), (1.5, Y0 + 26, 0), (3, Y0 + 24.5, 0), (4.5, Y0 + 25.5, 0)])
bpy.ops.nproto.extrude(inputs=ins((wc, "S0")), distance=2.0, direction="Z"); wall2 = ctx.active_object; wall2.name = "TWall"
s_w = store.get_surfaces(wall2)[0]
top = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([q[2] for q in geom.boundary_points(s_w, w, 8)]))
p0 = np.array(geom.boundary_points(s_w, top, 2)[0])
prof3 = nurbs("TWprof", [tuple(p0), tuple(p0 + np.array([0.0, 0.6, 0.6])), tuple(p0 + np.array([0.0, 1.4, 0.4]))])
prof3.data.splines[0].order_u = 3
r = bpy.ops.nproto.sweep(inputs=ins((prof3, "S0"), (wall2, top)), tangent=True)
tsw = ctx.active_object
ss_ = store.get_surfaces(tsw)[0]
worst = 0.0
for qpt in geom.boundary_points(s_w, top, 9)[1:-1]:
    na = nsw._normal_near(s_w, np.array(qpt)); nb = nsw._normal_near(ss_, np.array(qpt))
    worst = max(worst, math.degrees(math.acos(min(1.0, abs(float(na @ nb))))))
check("tangent sweep along a surface edge is G1", r == {'FINISHED'} and worst < 0.5, f"max normal angle {worst:.3f} deg")
bpy.ops.object.select_all(action='DESELECT'); wall2.select_set(True); tsw.select_set(True)
r = bpy.ops.nproto.continuity()
rep = json.loads(ctx.scene["nproto_last_continuity"])
check("continuity check on the tangent sweep", r == {'FINISHED'} and rep["gap"] < 1e-3 and rep["angle"] < 1.5, str(rep))
# pipe along the quarter arc
bpy.ops.object.select_all(action='DESELECT'); rc2.select_set(True); ctx.view_layer.objects.active = rc2
r = bpy.ops.nproto.pipe(inputs=ins((rc2, "S0")), radius=0.2)
check("pipe (Pappus)", r == {'FINISHED'} and abs(SA(ctx.active_object) - 2 * math.pi * 0.2 * 3 * math.pi / 2) < 5e-3)
# boundary: 4 lines of a 2x3 rectangle (in any order / direction) -> planar patch
b1 = poly("BD1", [(0, Y0 + 30, 0), (2, Y0 + 30, 0)]); b2 = poly("BD2", [(2, Y0 + 33, 0), (2, Y0 + 30, 0)])
b3 = poly("BD3", [(0, Y0 + 33, 0), (2, Y0 + 33, 0)]); b4 = poly("BD4", [(0, Y0 + 30, 0), (0, Y0 + 33, 0)])
r = bpy.ops.nproto.boundary(inputs=ins((b1, "S0"), (b3, "S0"), (b2, "S0"), (b4, "S0")))
check("boundary surface from 4 curves", r == {'FINISHED'} and abs(SA(ctx.active_object) - 6) < 1e-4, f"{SA(ctx.active_object):.6f}")
# curves from surfaces
ctx.scene.cursor.location = (0, Y0 + 40, 0)
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0, degree="0"); sp_ = ctx.active_object
bpy.ops.nproto.iso_curve(iso_dir="V", param=0.5); iso = ctx.active_object
L_iso = features.input_curve(iso, "S0").length(1e-6)
check("iso curve (sphere equator / meridian)", abs(L_iso - 2 * math.pi) < 0.01 or abs(L_iso - math.pi) < 0.01, f"length {L_iso:.4f}")
ctx.scene.cursor.location = (0, Y0 + 45, 0)
bpy.ops.nproto.add_primitive(kind="PLANE", width=4, depth=4, pl_degree_u=3, pl_degree_v=3, spans_u=1, spans_v=1); pl_ = ctx.active_object
bpy.ops.curve.primitive_bezier_circle_add(radius=1.0, location=(0, Y0 + 45, 3)); pc = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); pc.select_set(True); pl_.select_set(True); ctx.view_layer.objects.active = pl_
r = bpy.ops.nproto.project_curve(direction="Z")
pcs = geom.sample_curve(features.input_curve(ctx.active_object, "S0"), 64)
check("project curve onto a surface", r == {'FINISHED'} and np.abs(pcs[:, 2]).max() < 1e-4
      and abs(np.linalg.norm(pcs[:, :2] - [0, Y0 + 45], axis=1).mean() - 1.0) < 1e-3)
bpy.ops.object.select_all(action='DESELECT'); sp_.select_set(True)
ctx.scene.cursor.location = (0, Y0 + 40, 0.5)
bpy.ops.nproto.add_primitive(kind="PLANE", width=4, depth=4, pl_degree_u=1, pl_degree_v=1, spans_u=1, spans_v=1); cutp = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); sp_.select_set(True); cutp.select_set(True)
r = bpy.ops.nproto.intersect()
ic = geom.sample_curve(features.input_curve(ctx.active_object, "S0"), 64)
rr = np.linalg.norm(ic[:, :2] - [0, Y0 + 40], axis=1)
check("intersect sphere x plane = circle r=sqrt(0.75)", r == {'FINISHED'} and abs(rr.mean() - math.sqrt(0.75)) < 2e-3
      and np.abs(ic[:, 2] - 0.5).max() < 1e-3, f"r={rr.mean():.5f}")
# rebuild surface: exact sphere -> degree 3, 6x6 spans
bpy.ops.object.select_all(action='DESELECT'); sp_.select_set(True); ctx.view_layer.objects.active = sp_
r = bpy.ops.nproto.rebuild_surface(degree_u=3, degree_v=3, spans_u=8, spans_v=6)
d_ = _kl().blender_ext.surface_cvs(store.get_surfaces(sp_)[0])
check("rebuild surface degree/spans", r == {'FINISHED'} and (d_["degree_u"], d_["count_u"], d_["count_v"]) == (3, 11, 9)
      and abs(SA(sp_) - 4 * math.pi) / (4 * math.pi) < 0.01, f"{d_['count_u']}x{d_['count_v']} CVs, area {SA(sp_):.4f}")
# rebuild a plane with evenly spaced cubic CVs (5x5 spans) to 1 span: CVs evenly at thirds
bpy.ops.nproto.add_primitive(kind="PLANE", width=3, depth=3, pl_degree_u=3, pl_degree_v=3, spans_u=5, spans_v=5); rp_ = ctx.active_object
rp_.location = (0, 0, -40)
bpy.ops.nproto.rebuild_surface(degree_u=3, degree_v=3, spans_u=1, spans_v=1)
d_ = _kl().blender_ext.surface_cvs(store.get_surfaces(rp_)[0])
xs = sorted({round(p[0], 4) for p in d_["points"]}); ys = sorted({round(p[1], 4) for p in d_["points"]})
check("rebuild 5x5-span plane to 1 span: CVs evenly spaced, flat, same size",
      d_["count_u"] == 4 and np.allclose(xs, [-1.5, -0.5, 0.5, 1.5], atol=2e-3) and np.allclose(ys, [-1.5, -0.5, 0.5, 1.5], atol=2e-3)
      and max(abs(p[2]) for p in d_["points"]) < 1e-9 and abs(SA(rp_) - 9) < 1e-6, f"x {xs} y {ys} n {d_['count_u']}x{d_['count_v']} deg {d_['degree_u']} area {SA(rp_)}")
from blender_nurbs import draw as ndraw, cvs as ncvs2
tp = ndraw.topology(rp_)
check("rebuild preview topology: 1x1 spans, 4x4 CVs, 4 isoparms", tp is not None and tp["spans"] == (1, 1)
      and tp["count"] == (4, 4) and len(tp["iso"]) == 4 and len(tp["hull"]) == 8, str(tp and (tp["spans"], tp["count"])))
bpy.ops.object.select_all(action='DESELECT'); rp_.select_set(True); ctx.view_layer.objects.active = rp_
r = bpy.ops.nproto.tab_edit()
cg = ctx.active_object
check("Tab on a NURBS surface opens its CV cage", r == {"FINISHED"} and ncvs2.is_cage(cg) and cg.mode == "EDIT", f"{r} {cg.name} {cg.mode} {ctx.mode}")
bpy.ops.nproto.cvs_cancel()
# --- Extend surface (exact natural continuation) ---------------------------
bpy.ops.nproto.add_primitive(kind="PLANE", width=3, depth=3, pl_degree_u=3, pl_degree_v=3, spans_u=5, spans_v=5); ex_ = ctx.active_object
ex_.location = (0, 0, -60)
bb0 = store.get_brep(ex_).bounding_box()
xedge = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([p[0] for p in geom.boundary_points(store.get_surfaces(ex_)[0], w, 8)]))
r = bpy.ops.nproto.extend(inputs=json.dumps([[ex_.name, xedge]]), distance=1.0)
bb1 = store.get_brep(ex_).bounding_box()
check("extend plane edge by 1: +1 in x, flat, area 12, history step", r == {'FINISHED'} and abs(bb1[1][0] - bb0[1][0] - 1.0) < 1e-4
      and abs(bb1[0][0] - bb0[0][0]) < 1e-9 and abs(bb1[1][2] - bb0[1][2]) < 1e-9 and abs(SA(ex_) - 12) < 1e-4
      and features.stack(ex_)[-1]["op"] == "extend_surface", f"dx {bb1[1][0] - bb0[1][0]:.6f} area {SA(ex_):.5f} {bb0} {bb1} {[e['op'] for e in features.stack(ex_)]}")
from blender_nurbs import history_ui as _hu
_hu.sync_step(ex_, len(features.stack(ex_)) - 1)
ex_.nproto_step.distance = -0.5
bb2 = store.get_brep(ex_).bounding_box()
check("history: distance -0.5 shrinks the edge exactly", abs(bb2[1][0] - bb0[1][0] + 0.5) < 1e-4 and abs(SA(ex_) - 7.5) < 1e-4,
      f"dx {bb2[1][0] - bb0[1][0]:.6f} area {SA(ex_):.5f}")
# exact cylinder: extending across its seam keeps it a perfect circle (rational continuation)
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0", spans_u=1, spans_v=1, caps=False); ecy = ctx.active_object
ecy.location = (10, 0, -60)
s0 = store.get_surfaces(ecy)[0]
seam = [w for w in ("U0", "U1", "V0", "V1") if np.ptp([p[2] for p in geom.boundary_points(s0, w, 8)]) > 1.0][0]
a0 = SA(ecy)
r = bpy.ops.nproto.extend(inputs=json.dumps([[ecy.name, seam]]), distance=0.5)
vv = np.array([v.co for v in ecy.data.vertices])
rad = np.linalg.norm(vv[:, :2], axis=1)
check("extend exact cylinder across the seam: stays on radius 1, area grows by 0.5 x 2", r == {'FINISHED'}
      and np.abs(rad - 1).max() < 2e-3 and abs(SA(ecy) - a0 - 1.0) < 1e-3, f"r {rad.min():.5f}..{rad.max():.5f} dA {SA(ecy) - a0:.5f}")

# freeform: the original surface is untouched, the extension continues it smoothly
from blender_nurbs import extend as _ext
_pts = [(i * 0.6, j * 0.6, 0.35 * math.sin(i * 1.1) * math.cos(j * 0.8) + 0.1 * i) for i in range(5) for j in range(5)]
_s = ext.surface_from_cvs(3, 3, _pts, geom.clamped_uniform_knots(5, 3), geom.clamped_uniform_knots(5, 3))
_t0 = store.brep_to_text(ext.brep_from_surface(_s))
_s0 = store.brep_from_text(_t0).faces()[0].surface()
_s1 = store.brep_from_text(_ext.extend_surface_text(_t0, "U1", 0.8)).faces()[0].surface()
# old u in [0,1] maps to new [0, c]; find c from the knots (the interior knots are unchanged in old units)
_d1 = ext.surface_cvs(_s1); _d0 = ext.surface_cvs(_s0)
c = _d1["knots_u"][len(_d1["knots_u"]) - 4 - 1]   # last interior knot of new = 2/3 of old (old units /(b'))
c = c / _d0["knots_u"][len(_d0["knots_u"]) - 4 - 1]
dev = max(np.linalg.norm(np.array(_s0.evaluate(u, v)) - np.array(_s1.evaluate(u * c, v)))
          for u in np.linspace(0, 1, 9) for v in np.linspace(0, 1, 9))
# smooth across the old end: compare second derivatives just inside / outside via finite differences
def _p(u, v): return np.array(_s1.evaluate(u, v))
h = 1e-3; kink = max(np.linalg.norm((_p(c + h, v) - 2 * _p(c, v) + _p(c - h, v))) / h**2 -
                     np.linalg.norm((_p(c, v) - 2 * _p(c - h, v) + _p(c - 2 * h, v))) / h**2 for v in (0.2, 0.5, 0.8))
check("extend freeform: original part unchanged, curvature continuous across the old edge",
      dev < 1e-9 and abs(kink) < 1e-1, f"dev {dev:.2e} d2 jump {kink:.3e} c {c:.4f}")

# --- Square (G0/G1/G2 four-sided patch) -------------------------------------
from blender_nurbs import square as _sq
def _qF(s1, s2, t1, t2):
    x = (s1 + s2) / 2; y = (t1 + t2) / 2
    return (x, y + 300, 0.2 * s1 * s2 - 0.15 * t1 * t2 + 0.05 * x * y)
def _qpatch(name, x0, x1, y0, y1):
    A = [(x0, x0), (x0, x1), (x1, x1)]; B_ = [(y0, y0), (y0, y1), (y1, y1)]
    srf = ext.surface_from_cvs(2, 2, [_qF(a[0], a[1], b[0], b[1]) for a in A for b in B_], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    return store.new_object(ctx, name, srf)
qL, qR = _qpatch("SQ_L", -2, -1, -1, 1), _qpatch("SQ_R", 1, 2, -1, 1)
qB, qT = _qpatch("SQ_B", -1, 1, -2, -1), _qpatch("SQ_T", -1, 1, 1, 2)
def _near_edge(o, pt):
    sf = store.get_surfaces(o)[0]
    return min(("U0", "U1", "V0", "V1"), key=lambda w: np.linalg.norm(np.mean(geom.boundary_points(sf, w, 9), axis=0) - np.array(pt)))
sq_in = [(qL, _near_edge(qL, (-1, 300, 0))), (qB, _near_edge(qB, (0, 299, 0))), (qR, _near_edge(qR, (1, 300, 0))), (qT, _near_edge(qT, (0, 301, 0)))]
for cont, deg in (("G0", 2), ("G1", 3), ("G2", 5)):   # neighbours' degree 2, raised to what one span needs
    r = bpy.ops.nproto.square(inputs=json.dumps([[o.name, rf] for o, rf in sq_in]), e0=cont, e1=cont, e2=cont, e3=cont)
    sqo = ctx.active_object
    ds = ext.surface_cvs(store.get_surfaces(sqo)[0])
    srf = store.get_surfaces(sqo)[0]
    dev = max(abs(srf.evaluate(u, v)[2] - (lambda x, y: 0.2 * x * x - 0.15 * y * y + 0.05 * x * y)(srf.evaluate(u, v)[0], srf.evaluate(u, v)[1] - 300))
              for u in np.linspace(0, 1, 7) for v in np.linspace(0, 1, 7))
    qi = json.loads(sqo["nproto_square_info"])
    check(f"square {cont} in a curved quadratic hole: exact surface, degree matched to the neighbours", r == {'FINISHED'}
          and ds["degree_u"] == deg and (ds["count_u"], ds["count_v"]) == (deg + 1, deg + 1) and dev < 1e-7 and qi["angle"] < 1e-3,
          f"deg {ds['degree_u']} {ds['count_u']}x{ds['count_v']} dev {dev:.1e} angle {qi['angle']:.2e}")
for cont, deg in (("G1", 3), ("G2", 5)):
    r = bpy.ops.nproto.square(inputs=json.dumps([[o.name, rf] for o, rf in sq_in]), e0=cont, e1=cont, e2=cont, e3=cont,
                              sq_degree_u=str(deg), sq_degree_v=str(deg))
    ds = ext.surface_cvs(store.get_surfaces(ctx.active_object)[0])
    check(f"square {cont} with degree {deg} set: one span, {deg + 1}x{deg + 1}", r == {'FINISHED'} and ds["degree_u"] == deg
          and (ds["count_u"], ds["count_v"]) == (deg + 1, deg + 1), f"{ds['count_u']}x{ds['count_v']}")
# separate U / V degrees, the G0 default, Free sides, older files' single degree
r = bpy.ops.nproto.square(inputs=json.dumps([[o.name, rf] for o, rf in sq_in]), e0="G1", e1="G1", e2="G1", e3="G1",
                          sq_degree_u="3", sq_degree_v="5", sq_spans_u=1, sq_spans_v=1)
ds = ext.surface_cvs(store.get_surfaces(ctx.active_object)[0])
check("square: degree U 3 x V 5", r == {'FINISHED'} and (ds["degree_u"], ds["degree_v"]) == (3, 5)
      and (ds["count_u"], ds["count_v"]) == (4, 6), f"{ds['degree_u']}x{ds['degree_v']} {ds['count_u']}x{ds['count_v']}")
from blender_nurbs import ops_sweep as _ops_sw
check("square: edges start at G0", all(_ops_sw.NPROTO_OT_square.__annotations__[f"e{i}"].keywords["default"] == "G0"
                                       for i in range(4)))
_g = features.run("square", [(o, rf) for o, rf in sq_in], {"e0": "G1", "e1": "G1", "e2": "G1", "e3": "G1",
                                                         "square_degree": 5, "sq_spans_u": 1, "sq_spans_v": 1})
check("square: an older file's single Degree still applies to both directions", tuple(_sq.LAST_INFO["degree"]) == (5, 5))
# Free: the side isn't held to its boundary -- here a wavy curve on top of the hole
from blender_nurbs import curvetrim as _ct, curvejoin as _cj, curvetools as _cvt
wav = nurbs("SQ_Wave", [(-1.0, 301.0, 0.0), (-0.5, 301.0, 0.3), (0.0, 301.0, -0.3), (0.5, 301.0, 0.3), (1.0, 301.0, 0.1)])
_fin = [(qL, sq_in[0][1]), (qB, sq_in[1][1]), (qR, sq_in[2][1]), (wav, "S0")]
gaps = {}
for c3 in ("G0", "FREE"):
    r = bpy.ops.nproto.square(inputs=json.dumps([[o.name, rf] for o, rf in _fin]), e0="G0", e1="G0", e2="G0", e3=c3)
    srf = store.get_surfaces(ctx.active_object)[0]
    _ts = np.linspace(0, 1, 401)
    top = np.array([srf.evaluate(u, 1.0) for u in _ts] + [srf.evaluate(u, 0.0) for u in _ts]
                   + [srf.evaluate(0.0, v) for v in _ts] + [srf.evaluate(1.0, v) for v in _ts])
    W = _ct.sample_spans(_cj.transform_spans(_cj.spline_spans(wav.data.splines[0]), geom.world_matrix(wav)), 200)[0]
    gaps[c3 + "_maxdev"] = float(max(_cvt._seg_dist(W, top).max(), 0)) if r == {'FINISHED'} else None
check("square: a Free side doesn't follow its curve (G0 does)", gaps["G0_maxdev"] is not None and gaps["G0_maxdev"] < 1e-3
      and gaps["FREE_maxdev"] is not None and gaps["FREE_maxdev"] > 0.05, f"{gaps}")
# history: change one side to G0 through the Feature panel
from blender_nurbs import feature_ui as _fui
_fui.sync_from_history(sqo)
sqo.nproto_feature.e0 = "G0"
check("square feature edit rebuilds (edge 1 -> G0)", features.history(sqo)["params"]["e0"] == "G0"
      and store.get_brep(sqo).face_count() == 1)
# clickable continuity labels: one per side; a click steps the side's setting and rebuilds
from blender_nurbs import contlabels as _cl
_lb = _cl.labels(sqo)
check("continuity labels: 4 on a square, surface edges offer Free/G0/G1/G2", len(_lb) == 4
      and all(L["values"] == ["FREE", "G0", "G1", "G2"] for L in _lb), f"{[(L['param'], L['value']) for L in _lb]}")
_before = features.history(sqo)["params"]["e1"]
r = bpy.ops.nproto.cycle_continuity(obj_name=sqo.name, index=1)
_after = features.history(sqo)["params"]["e1"]
r2 = bpy.ops.nproto.cycle_continuity(obj_name=sqo.name, index=1, step=-1)
check("continuity labels: click steps the side and the square rebuilds; step back restores",
      r == {'FINISHED'} and _after == _cl.next_value(_cl.ORDER, _before, 1) and r2 == {'FINISHED'}
      and features.history(sqo)["params"]["e1"] == _before and store.get_brep(sqo).face_count() == 1,
      f"{_before} -> {_after} -> {features.history(sqo)['params']['e1']}")
_c1 = nurbs("CL_A", [(0, 500, 0), (1, 500, 0.3), (2, 500, 0)]); _c2 = nurbs("CL_B", [(0, 502, 0), (1, 502, -0.3), (2, 502, 0)])
bpy.ops.nproto.loft(inputs=ins((_c1, "S0"), (_c2, "S0"))); _lo = ctx.active_object
_lf = _cl.labels(_lo)
check("continuity labels: a loft of curves has 2 labels that stay G0", len(_lf) == 2 and all(x["values"] == ["G0"] for x in _lf))
# three-sided: kernel fillet corner replaced by a G2 square
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0"); sbx = ctx.active_object
sbx.location = (0, 320, 0)
_b = store.get_brep(sbx)
def _emid(b, i):
    e = b.edges()[i]; t0, t1 = e.parameter_range(); return np.array(e.point_at(0.5 * (t0 + t1)))
_pick = [i for i in store.shared_edges(sbx) if sum(abs(_emid(_b, i) - 1) < 1e-6) >= 2]
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[sbx.name, f"E{i}"] for i in _pick]), shape="BLEND", radius=0.4, continuity="2")
bpy.ops.object.select_all(action='DESELECT'); sbx.select_set(True); ctx.view_layer.objects.active = sbx
bpy.ops.nproto.unjoin()
_parts = [o for o in bpy.data.objects if o.name.startswith(sbx.name) and store.is_nurbs(o)]
_corner = [o for o in _parts if store.get_brep(o).edge_count() == 3][0]
_cb = store.get_brep(_corner); _cm = [_emid(_cb, i) for i in range(3)]
_ins = []
for o in _parts:
    if o is _corner:
        continue
    bo = store.get_brep(o)
    for i in range(bo.edge_count()):
        if min(np.linalg.norm(_emid(bo, i) - c) for c in _cm) < 1e-6:
            _ins.append([o.name, f"E{i}"])
bpy.data.objects.remove(_corner)
r = bpy.ops.nproto.square(inputs=json.dumps(_ins), e0="G2", e1="G2", e2="G2")
sq3 = ctx.active_object; qi3 = json.loads(sq3.get("nproto_square_info", "{}"))
check("square, 3 sides (fillet corner): degree 5 single span like the blends, G2 within 0.01 deg", r == {'FINISHED'}
      and len(_ins) == 3 and sq3 is not sqo and tuple(qi3.get("cvs", ())) == (6, 6)
      and qi3.get("gap", 1) < 1e-5 and qi3.get("angle", 9) <= 0.01 and qi3.get("curv", 9) <= 0.01,
      f"{r} {len(_ins)} " + str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in qi3.items()}))

# --- Refit to Trim: trimmed 4-sided wall -> untrimmed surface in the boundary --
bpy.ops.nproto.add_primitive(kind="ARC", radius=1.0, angle=120.0, degree="0"); rarc = ctx.active_object
rarc.location = (0, 0, -120)
bpy.ops.nproto.extrude(inputs=ins((rarc, "S0")), distance=4.0, direction="Z"); rwall = ctx.active_object
rwall.name = "RF_Wall"
_wpts = [(x, -3.0, -120 + 3.0 + 0.3 * math.sin(3 * x)) for x in np.linspace(-1.6, 1.6, 41)]
_cu = bpy.data.curves.new("RF_Cut", 'CURVE'); _cu.dimensions = '3D'
_sp = _cu.splines.new('NURBS'); _sp.points.add(len(_wpts[::4]) - 1)
for _p, _c in zip(_sp.points, _wpts[::4]): _p.co = (*_c, 1.0)
_sp.order_u = 4; _sp.use_endpoint_u = True
rcut = bpy.data.objects.new("RF_Cut", _cu); ctx.collection.objects.link(rcut)
_ws = store.get_surfaces(rwall)[0]
_ref = geom.world_matrix(rwall) @ Vector(_ws.evaluate(sum(_ws.u_domain()) / 2, sum(_ws.v_domain()) / 2))
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.trim(action="KEEP", target="RF_Wall", cutters=ins((rcut, "S0")), ref_point=tuple(_ref), projection="Y",
                        normal=(0, 0, 1), view_dir=(0, 1, 0))
a_trim = store.get_brep(rwall).area(1e-7)
check("refit setup: wall trimmed along a wavy curve", r == {'FINISHED'} and not store.is_natural(rwall), f"area {a_trim:.4f}")
bpy.ops.object.select_all(action='DESELECT'); rwall.select_set(True); ctx.view_layer.objects.active = rwall
r = bpy.ops.nproto.refit_trim()
from blender_nurbs import refit as _rf
rs = store.get_surfaces(rwall)[0]
# every refit point lies on the original cylinder wall (radius 1 about the arc centre)
_M = geom.world_matrix(rwall)
_pts = np.array([np.array(_M @ Vector(rs.evaluate(u, v))) for u in np.linspace(0, 1, 9) for v in np.linspace(0, 1, 9)])
_rad = np.linalg.norm(_pts[:, :2] - np.array(rarc.location[:2]), axis=1)
check("refit to trim: untrimmed, area kept, on the original wall, light net", r == {'FINISHED'} and store.is_natural(rwall)
      and store.get_brep(rwall).face_count() == 1 and abs(store.get_brep(rwall).area(1e-7) - a_trim) / a_trim < 1e-3
      and np.abs(_rad - 1).max() < 2e-3 and max(_rf.LAST_INFO["cvs"]) <= 24
      and features.stack(rwall)[-1]["op"] == "refit_trim",
      f"area {store.get_brep(rwall).area(1e-7):.4f} vs {a_trim:.4f}, r {_rad.min():.5f}..{_rad.max():.5f}, {_rf.LAST_INFO}")

# --- Loft G1/G2 from TRIMMED edges, light net ------------------------------
lf = make_floor("LT_Floor", (0, 400, 0))
_c1 = poly("LT_c1", [(1.5, 395, 0), (1.5, 405, 0)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="LT_Floor", cutters=ins((_c1, "S0")), ref_point=(0.5, 400, 0), normal=(0, 0, 1),
                    view_dir=(0, 0, -1), projection="Z")
_wl = poly("LT_wl", [(3, 398, 1), (3, 402, 1)])
bpy.ops.nproto.extrude(inputs=ins((_wl, "S0")), distance=2.0, direction="Z"); lw = ctx.active_object; lw.name = "LT_Wall"
_c2 = poly("LT_c2", [(2.0, 395, 1.5), (2.0, 405, 1.5)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="LT_Wall", cutters=ins((_c2, "S0")), ref_point=(3, 400, 2.5), normal=(1, 0, 0),
                    view_dir=(1, 0, 0), projection="X")
def _trim_edge(o, test):
    b = store.get_brep(o); M = geom.world_matrix(o)
    for i, e in enumerate(b.edges()):
        t0, t1 = e.parameter_range(); m = M @ Vector(e.point_at(0.5 * (t0 + t1)))
        if test(m):
            return f"E{i}"
ea = _trim_edge(lf, lambda m: abs(m.x - 1.5) < 1e-4)
eb = _trim_edge(lw, lambda m: abs(m.z - 1.5) < 1e-4)
for cont in ("G1", "G2"):
    r = bpy.ops.nproto.loft(inputs=json.dumps([[lf.name, ea], [lw.name, eb]]), cont_a=cont, cont_b=cont)
    lo = ctx.active_object
    dl = ext.surface_cvs(store.get_surfaces(lo)[0])
    sl = store.get_surfaces(lo)[0]
    # tangent to the floor (normal = z) along the floor edge, tangent to the wall (normal = x) along the wall edge
    Ml = np.array(geom.world_matrix(lo))
    def _nrm(u, v):
        _, su, sv = sl.derivatives(u, v); n = np.cross(su, sv); n = Ml[:3, :3] @ n; return n / np.linalg.norm(n)
    (u0, u1), (v0, v1) = sl.u_domain(), sl.v_domain()
    if (Ml @ np.append(sl.evaluate(u0, 0.5 * (v0 + v1)), 1))[2] > 0.5:   # u may run either way (orientation)
        u0, u1 = u1, u0
    angA = max(np.degrees(np.arccos(min(1, abs(_nrm(u0, v) @ np.array([0, 0, 1]))))) for v in np.linspace(v0, v1, 9))
    angB = max(np.degrees(np.arccos(min(1, abs(_nrm(u1, v) @ np.array([1, 0, 0]))))) for v in np.linspace(v0, v1, 9))
    check(f"loft {cont} between two trimmed edges: tangent to both faces, light net", r == {'FINISHED'} and ea and eb
          and angA < 0.01 and angB < 0.01 and dl["count_v"] <= 4,
          f"{ea} {eb} angles {angA:.4f}/{angB:.4f} CVs {dl['count_u']}x{dl['count_v']}")

# a straight (degree-1) edge lofted G2 to a CURVED trimmed edge (a spoon's handle to its bowl):
# cubic along the edge, one smooth face -- not a polyline through the samples split into strips
_sb = [nurbs(f"SB{k}", [(x, -w, 600.0), (x, 0, 600.5 + 0.3 * (k == 1)), (x, w, 600.0)], order=3)
       for k, (x, w) in enumerate(((-2.0, 1.0), (0.0, 1.5), (2.0, 1.0)))]
bpy.ops.nproto.loft(inputs=ins(*[(c_, "S0") for c_ in _sb])); _bowl = ctx.active_object; _bowl.name = "SB_Bowl"
_cut = poly("SB_cut", [(1.2, -3, 605), (1.2, 3, 605)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="SB_Bowl", cutters=ins((_cut, "S0")), ref_point=(0, 0, 600.6),
                    normal=(0, 0, 1), view_dir=(0, 0, -1), projection="Z")
_eb = _trim_edge(_bowl, lambda m: abs(m.x - 1.2) < 1e-3)
_hl = poly("SB_hl", [(3, -0.4, 600.4), (3, 0.4, 600.4)])
bpy.ops.nproto.extrude(inputs=ins((_hl, "S0")), distance=3.0, direction="X"); _hd = ctx.active_object
_er = min(("U0", "U1", "V0", "V1"), key=lambda e_: abs(np.mean(np.array(geom.boundary_points(store.get_surfaces(_hd)[0], e_, 5))[:, 0]
                                                               + geom.world_matrix(_hd).translation.x) - 3))
for _smp in (5, 7, 8, 9, 12, 24):
    r = bpy.ops.nproto.loft(inputs=json.dumps([[_hd.name, _er], [_bowl.name, _eb]]), cont_a="G2", cont_b="G2", samples=_smp)
    _sp = ctx.active_object
    _d = ext.surface_cvs(store.get_surfaces(_sp)[0])
    _labs = __import__("blender_nurbs.contlabels", fromlist=["x"]).labels(_sp)
    check(f"loft G2 straight edge -> curved trimmed edge ({_smp} samples): cubic along it, one smooth face, holds",
          r == {'FINISHED'} and _eb and store.get_brep(_sp).face_count() == 1 and _d["degree_v"] >= 3
          and store._crease_pieces(store.get_surfaces(_sp)[0], 1.0) is None
          and all(L_["check"] and L_["check"]["ok"] for L_ in _labs),
          f"{_eb} {_er} faces {store.get_brep(_sp).face_count()} degree {_d['degree_u']}x{_d['degree_v']} "
          f"CVs {_d['count_u']}x{_d['count_v']} {[L_['check'] and L_['check']['text'] for L_ in _labs]}")

_ok_or = True
_s3 = store.get_surfaces(sq3)[0]; _M3 = np.array(geom.world_matrix(sq3))
for _nm, _rf in _ins:
    _sd = _sq.sample_side(bpy.data.objects[_nm], _rf, 9)
    for _i in (2, 4, 6):
        _u, _v = _sq._foot(_s3, np.linalg.solve(_M3[:3, :3], _sd.P[_i] - _M3[:3, 3]))
        _, _su, _sv = _s3.derivatives(_u, _v)
        _n = _M3[:3, :3] @ np.cross(_su, _sv)
        _ok_or &= float(_n @ _sd.Nout[_i]) > 0
check("square normal agrees with the neighbouring faces' outward normals", _ok_or)
# display meshes of neighbouring objects meet without visible slits (fine edge sampling)
import bmesh as _bmesh
from mathutils.bvhtree import BVHTree as _BVH
def _bnd_world(o):
    bm = _bmesh.new(); bm.from_mesh(o.data)
    out = [o.matrix_world @ v.co for v in bm.verts if any(e.is_boundary for e in v.link_edges)]; bm.free(); return out
def _bvh_world(o):
    o.data.calc_loop_triangles()
    return _BVH.FromPolygons([o.matrix_world @ v.co for v in o.data.vertices], [tuple(t.vertices) for t in o.data.loop_triangles])
_nbs = [bpy.data.objects[n] for n, _ in _ins]
_tr = {o.name: _bvh_world(o) for o in _nbs}
_gap = 0.0
for _p in _bnd_world(sq3):
    _gap = max(_gap, min(_tr[o.name].find_nearest(_p)[3] for o in _nbs))
check("display meshes of the square and its neighbours meet (no visible slits)", _gap < 1e-4 * 10, f"max gap {_gap:.2e}")
# re-trim a planar face to a clean curve (exact other edges kept)
rtf = make_floor("RT_Floor", (0, 420, 0))
_c3 = poly("RT_c", [(1.5, 415, 0), (1.5, 425, 0)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="RT_Floor", cutters=ins((_c3, "S0")), ref_point=(0.5, 420, 0), normal=(0, 0, 1),
                    view_dir=(0, 0, -1), projection="Z")
_e = _trim_edge(rtf, lambda m: abs(m.x - 1.5) < 1e-4)
_a0 = store.get_brep(rtf).area(1e-7)
# clean curve: degree 5 Bezier bulging 0.1 outwards in +x (end points on the old trim line)
_ys = np.linspace(418, 422, 6)
_cv = [(1.5 + (0.1 * 16 / 5 * (i * (5 - i)) / 6.25 / 4 if 0 < i < 5 else 0.0), y, 0.0) for i, y in enumerate(_ys)]
_kn = [0.0] * 6 + [1.0] * 6
_new = _sq.retrim_planar(rtf, _e, _cv, _kn, 5)
_cc = ext.curve_from_cvs(5, _cv, _kn, None)
_bulge = np.trapezoid([_cc.evaluate(t)[0] - 1.5 for t in np.linspace(0, 1, 401)], [(_cc.evaluate(t)[1]) for t in np.linspace(0, 1, 401)])
check("re-trim planar face to a clean curve: new edge, area grows by the bulge, other edges kept",
      store.get_brep(rtf).face_count() == 1 and _new.startswith("E")
      and abs(store.get_brep(rtf).area(1e-7) - (_a0 + _bulge)) < 1e-4 and store.get_brep(rtf).edge_count() == 4,
      f"{_a0:.4f} + {_bulge:.4f} -> {store.get_brep(rtf).area(1e-7):.4f}, edges {store.get_brep(rtf).edge_count()}")

# --- Diagnostics: curvature comb ---------------------------------------------
from blender_nurbs import diagnostics as _dg
bpy.ops.curve.primitive_nurbs_circle_add(radius=2.0, location=(0, 0, -150)); dc = ctx.active_object
_d = _dg.comb_data(dc, 64)
_P, _K = _d[0]
_kap = np.linalg.norm(_K, axis=1)
_toc = np.array([0, 0, -150]) - _P
_cosang = np.sum(_K * _toc, axis=1) / (_kap * np.linalg.norm(_toc, axis=1))
check("curvature comb: NURBS circle r=2 -> kappa 0.5 everywhere, pointing to the centre",
      np.abs(_kap - 0.5).max() < 1e-4 and _cosang.min() > 0.9999, f"kappa {_kap.min():.6f}..{_kap.max():.6f}")
# an S-curve: the comb changes side at the inflection (curvature vector flips)
_sc = poly("DG_S", [(0, 0, -160), (1, 0, -160)])
_sc.data.splines.clear(); _sp = _sc.data.splines.new('BEZIER'); _sp.bezier_points.add(1)
_sp.bezier_points[0].co = (0, 0, -160); _sp.bezier_points[0].handle_right = (1, 1, -160); _sp.bezier_points[0].handle_left = (-1, -1, -160)
_sp.bezier_points[1].co = (3, 0, -160); _sp.bezier_points[1].handle_left = (2, -1, -160); _sp.bezier_points[1].handle_right = (4, 1, -160)
_P2, _K2 = _dg.comb_data(_sc, 64)[0]
_side = np.sign(np.cross(np.gradient(_P2, axis=0), _K2)[:, 2])
check("curvature comb: S-curve flips side once (inflection)", int(np.sum(np.abs(np.diff(_side[2:-2])) > 0)) == 1)

# zebra uses its own fine tessellation with exact normals
_zp, _zn, _zt = _dg.zebra_mesh(sq3, 2.0, 0.0005)
_ss = store.get_surfaces(sq3)[0]
_errs = []
# the 3-sided patch has one corner collapsed to a point: its normal is undefined there
_corners = [np.array(_ss.evaluate(a, b)) for a in _ss.u_domain() for b in _ss.v_domain()]
_pole = [c for i, c in enumerate(_corners) if any(np.linalg.norm(c - d) < 1e-9 for d in _corners[i + 1:])]
_rad = 0.15 * float(np.linalg.norm(np.ptp(_zp, axis=0)))
for _k_ in range(0, len(_zp), max(1, len(_zp) // 40)):
    if any(np.linalg.norm(_zp[_k_] - c) < _rad for c in _pole):
        continue
    _u, _v = _sq._foot(_ss, _zp[_k_].astype(float))
    _, _su, _sv = _ss.derivatives(_u, _v); _ne = np.cross(_su, _sv); _ne /= np.linalg.norm(_ne)
    _errs.append(np.degrees(np.arccos(min(1, abs(float(_ne @ (_zn[_k_] / np.linalg.norm(_zn[_k_]))))))))
check("zebra tessellation: fine and with exact normals", len(_zt) > 200 and max(_errs) < 0.05,
      f"{len(_zt)} tris, max normal error {max(_errs):.4f} deg")

# --- STEP export -------------------------------------------------------------
_stp = os.path.join(tempfile.gettempdir(), "nproto_test.step")
bpy.ops.object.select_all(action='DESELECT')
for _o in (sbx_parts if False else []): pass
_cyl = [o for o in bpy.data.objects if o.name.startswith("NURBS Cylinder") and store.is_nurbs(o)]
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0", caps=True); _sc_ = ctx.active_object
_sc_.location = (0, 0, -200)
bpy.ops.object.select_all(action='DESELECT'); _sc_.select_set(True); lf.select_set(True)
r = bpy.ops.nproto.export_step(filepath=_stp, use_selection=True, unit="MM")
_txt = open(_stp).read()
check("STEP export: valid Part 21 file, one solid + one sheet, rational surfaces, mm",
      r == {'FINISHED'} and _txt.startswith("ISO-10303-21;") and _txt.rstrip().endswith("END-ISO-10303-21;")
      and _txt.count("MANIFOLD_SOLID_BREP(") == 1 and _txt.count("SHELL_BASED_SURFACE_MODEL(") == 1
      and "RATIONAL_B_SPLINE_SURFACE" in _txt and "SI_UNIT(.MILLI.,.METRE.)" in _txt
      and _txt.count("ADVANCED_FACE(") == 3 + store.get_brep(lf).face_count(),
      f"faces {_txt.count('ADVANCED_FACE(')}")
# every #ref used is defined
import re as _re
_defs = set(_re.findall(r"^#(\d+)=", _txt, _re.M)); _uses = set(_re.findall(r"#(\d+)", _txt))
check("STEP export: every referenced entity is defined", _uses <= _defs, str(sorted(_uses - _defs)[:5]))

# --- Diagnostics: curvature map + draft angle --------------------------------
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=2.0, degree="0"); _sph = ctx.active_object; _sph.location = (0, 0, -170)
_P, _N, _T, _V = _dg.face_curvature_mesh(_sph, 8.0, 0.004)
check("curvature map: sphere r=2 -> Gaussian 0.25, mean +0.5 (convex, outward), k1 = k2 at every vertex",
      len(_T) > 50 and np.abs(_V["GAUSS"] - 0.25).max() < 1e-3 and np.abs(_V["MEAN"] - 0.5).max() < 1e-3
      and np.abs(_V["MAX"] - _V["MIN"]).max() < 1e-3,
      f"K {_V['GAUSS'].min():.5f}..{_V['GAUSS'].max():.5f} H {_V['MEAN'].min():.5f}..{_V['MEAN'].max():.5f}")
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0", caps=False); _cy2 = ctx.active_object
_cy2.location = (6, 0, -170)
_P, _N, _T, _V = _dg.face_curvature_mesh(_cy2, 8.0, 0.004)
_sg = np.sign(_V["MEAN"].mean())   # open sheet: its face normal decides convex / concave
check("curvature map: cylinder r=1 -> Gaussian 0, mean +-0.5, principal 1 and 0 (consistent sign)",
      np.abs(_V["GAUSS"]).max() < 1e-3 and np.abs(_V["MEAN"] - 0.5 * _sg).max() < 1e-3
      and np.abs((_V["MAX"] if _sg > 0 else -_V["MIN"]) - 1).max() < 1e-3
      and np.abs(_V["MIN"] if _sg > 0 else _V["MAX"]).max() < 1e-3,
      f"H {_V['MEAN'].min():.4f}..{_V['MEAN'].max():.4f} k1 {_V['MAX'].min():.4f}..{_V['MAX'].max():.4f} "
      f"k2 {_V['MIN'].min():.4f}..{_V['MIN'].max():.4f}")
check("draft angle: vertical wall 0, 5 deg tilt towards pull +5, undercut -5",
      abs(_dg.draft_angle_deg((1, 0, 0), (0, 0, 1))) < 1e-9
      and abs(_dg.draft_angle_deg((math.cos(math.radians(5)), 0, math.sin(math.radians(5))), (0, 0, 1)) - 5) < 1e-9
      and abs(_dg.draft_angle_deg((math.cos(math.radians(5)), 0, -math.sin(math.radians(5))), (0, 0, 1)) + 5) < 1e-9)

# --- Extrude: tilt, vector, draft (Multi-Draft) -------------------------------
_ln = poly("EX_ln", [(0, 0, -190), (2, 0, -190)])
r = bpy.ops.nproto.extrude(inputs=ins((_ln, "S0")), distance=1.0, direction="Z", tilt=30.0, tilt_azimuth=90.0)
_bb = store.get_brep(ctx.active_object).bounding_box()
check("extrude tilted 30 deg (towards +Y): far edge at (0, sin30, cos30)", r == {'FINISHED'}
      and abs(_bb[1][1] - 0.5) < 1e-4 and abs(_bb[1][2] - (-190 + math.cos(math.radians(30)))) < 1e-4, str(_bb))
r = bpy.ops.nproto.extrude(inputs=ins((_ln, "S0")), distance=2.0, direction="VECTOR", dir_vec=(0, 1, 1))
_bb = store.get_brep(ctx.active_object).bounding_box()
check("extrude along a vector (0,1,1): length 2", abs(_bb[1][1] - math.sqrt(2)) < 1e-4)
# draft: closed square 2x2, height 1, 10 deg -> exact mitred trapezoids
_sq_c = bpy.data.curves.new("EX_sq", 'CURVE'); _sq_c.dimensions = '3D'
_sp = _sq_c.splines.new('POLY'); _sp.points.add(3)
for _pt, _c in zip(_sp.points, [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]): _pt.co = (*_c, 1.0)
_sp.use_cyclic_u = True
_sqo = bpy.data.objects.new("EX_sq", _sq_c); ctx.collection.objects.link(_sqo); _sqo.location = (10, 0, -190)
r = bpy.ops.nproto.extrude(inputs=ins((_sqo, "S0")), distance=1.0, direction="Z", draft=10.0)
_dr = ctx.active_object
_t = math.tan(math.radians(10))
_area = 4 * ((2 + (2 + 2 * _t)) / 2) * math.sqrt(1 + _t * _t)
_bb = store.get_brep(_dr).bounding_box()
check("extrude square with 10 deg draft: outward, exact trapezoids (mitred corners)", r == {'FINISHED'}
      and abs(store.get_brep(_dr).area(1e-7) - _area) < 1e-6 and abs((_bb[1][0] - _bb[0][0]) - (2 + 2 * _t)) < 1e-4,
      f"area {store.get_brep(_dr).area(1e-7):.6f} vs {_area:.6f}")
r = bpy.ops.nproto.extrude(inputs=ins((_sqo, "S0")), distance=1.0, direction="Z", draft=10.0, draft_flip=True)
_bb = store.get_brep(ctx.active_object).bounding_box()
check("flip draft: top shrinks inwards", abs((_bb[1][0] - _bb[0][0]) - 2.0) < 1e-4
      and abs(store.get_brep(ctx.active_object).area(1e-7) - 4 * ((2 + (2 - 2 * _t)) / 2) * math.sqrt(1 + _t * _t)) < 1e-6)
# draft on a circle: cone frustum
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(20, 0, -190)); _ci = ctx.active_object
r = bpy.ops.nproto.extrude(inputs=ins((_ci, "S0")), distance=1.0, direction="Z", draft=10.0)
_r2 = 1 + _t
_fr = math.pi * (1 + _r2) * math.sqrt(1 + _t * _t)
check("extrude circle with 10 deg draft: cone frustum area", r == {'FINISHED'}
      and abs(store.get_brep(ctx.active_object).area(1e-7) - _fr) / _fr < 1e-4,
      f"{store.get_brep(ctx.active_object).area(1e-7):.5f} vs {_fr:.5f}")
r = bpy.ops.nproto.extrude(inputs=ins((_ci, "S0")), distance=1.0, direction="Z", draft=10.0, symmetric=True)
_sy = store.get_surfaces(ctx.active_object)[0]
_M = np.array(geom.world_matrix(ctx.active_object))
_pts = np.array([_M[:3, :3] @ np.array(_sy.evaluate(u, v)) + _M[:3, 3] for u in (0.0, 0.5, 1.0) for v in np.linspace(0, 1, 13)])
_rad = np.linalg.norm(_pts[:, :2] - np.array([20, 0]), axis=1).reshape(3, -1)
check("symmetric draft: both ends open away from the parting circle (r 1 -> 1+tan10 at z -1 and +1)", r == {'FINISHED'}
      and np.abs(_rad[1] - 1).max() < 1e-4 and np.abs(_rad[0] - _r2).max() < 1e-4 and np.abs(_rad[2] - _r2).max() < 1e-4
      and abs(_pts[:, 2].max() - _pts[:, 2].min() - 2) < 1e-6, f"{_rad.min(axis=1)} {_rad.max(axis=1)}")
# a drafted exact circle stays exact: the wall is a true cone on the circle's own 9 rational CVs
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _ec = ctx.active_object
_ec.location = (30, 0, -190); ctx.view_layer.update()
r = bpy.ops.nproto.extrude(inputs=ins((_ec, "S0")), distance=1.0, direction="Z", draft=10.0)
for _sym in (False, True):
    if _sym:
        r = bpy.ops.nproto.extrude(inputs=ins((_ec, "S0")), distance=1.0, direction="Z", draft=10.0, symmetric=True)
    _cs_ = store.get_surfaces(ctx.active_object)[0]
    _cd = ext.surface_cvs(_cs_)
    _M = np.array(geom.world_matrix(ctx.active_object))
    (_u0, _u1), (_v0, _v1) = _cs_.u_domain(), _cs_.v_domain()
    _rows = [np.array([_M[:3, :3] @ np.array(_cs_.evaluate(_u, _v)) + _M[:3, 3] for _v in np.linspace(_v0, _v1, 721)])
             for _u in (_u0, _u1)]
    _rr = [np.linalg.norm(_q[:, :2] - np.array([30, 0]), axis=1) for _q in _rows]
    check(f"drafted circle{' (symmetric)' if _sym else ''}: an exact cone on the circle's own rational CVs "
          "(9 round, radius constant to 1e-6 at both ends, 1 -> 1 + tan 10)", r == {'FINISHED'} and bool(_cd["weights"])
          and _cd["count_v"] == 9 and _cd["count_u"] == (3 if _sym else 2) and max(np.ptp(_x) for _x in _rr) < 1e-6
          and abs(_rr[0].mean() - (1 + _t if _sym else 1.0)) < 1e-6 and abs(_rr[1].mean() - (1 + _t)) < 1e-6,
          f"cvs {_cd['count_u']}x{_cd['count_v']} rational {bool(_cd['weights'])} radii "
          f"{[(round(float(_x.mean()), 7), float(np.ptp(_x))) for _x in _rr]}")
# a smooth profile with a sharp corner still drafts (sampled, mitred path)
_bzc = bezier("EX_bz", [(40, 0), (42, 0), (42, 2), (40, 2)], -190.0)
_bzc.data.splines[0].use_cyclic_u = True
for _bp in _bzc.data.splines[0].bezier_points:
    _bp.handle_left_type = _bp.handle_right_type = 'VECTOR'
r = bpy.ops.nproto.extrude(inputs=ins((_bzc, "S0")), distance=1.0, direction="Z", draft=10.0)
_bb = store.get_brep(ctx.active_object).bounding_box()
check("drafted smooth profile with sharp corners: still drafts with mitred corners",
      r == {'FINISHED'} and abs((_bb[1][0] - _bb[0][0]) - (2 + 2 * _t)) < 1e-3, str(_bb))

# sharp shading at B-rep creases (cap rim of a capped cylinder)
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0", caps=True); _cc = ctx.active_object
_cc.location = (0, 0, -230)
_me = _cc.data; _me.calc_loop_triangles()
_worst = 0.0
for _t in _me.loop_triangles:
    _tn = _t.normal
    for _l in _t.loops:
        _worst = max(_worst, math.degrees(_me.corner_normals[_l].vector.angle(_tn)))
check("capped cylinder: corners at the rim shade with their own face's normal (sharp crease)", _worst < 20, f"max corner-vs-triangle angle {_worst:.1f}")

# no long sliver fans on ruled / revolved sides (they shade as dark bands)
_me = _cc.data; _me.calc_loop_triangles()
_asp = []
for _t in _me.loop_triangles:
    _p = [Vector(_me.vertices[v].co) for v in _t.vertices]
    _L = max((_p[k] - _p[(k + 1) % 3]).length for k in range(3))
    _asp.append(_L * _L / (2 * _t.area + 1e-30))
check("display mesh of a capped cylinder: no sliver fans (95% of triangles aspect < 12)",
      float(np.percentile(_asp, 95)) < 12, f"p95 aspect {np.percentile(_asp, 95):.1f}")

# revolve seam angle: same surface, seam moved around the axis
_prof = poly("RV_prof", [(1.0, 0, 0), (1.3, 0, 1.0), (0.9, 0, 2.0)], loc=(0, 0, -260))
bpy.ops.nproto.revolve(inputs=ins((_prof, "S0")), angle=360.0, axis="Z", pivot="OBJECT", seam_angle=0.0); _r0 = ctx.active_object
bpy.ops.nproto.revolve(inputs=ins((_prof, "S0")), angle=360.0, axis="Z", pivot="OBJECT", seam_angle=120.0); _r1 = ctx.active_object
_s1 = store.get_surfaces(_r1)[0]
_seam = np.array(_s1.evaluate(_s1.u_domain()[0], sum(_s1.v_domain()) / 2)) if True else None
_ang = math.degrees(math.atan2(*(_seam[[1, 0]])))
_d = store.get_surfaces(_r0)[0]
_ar0, _ar1 = store.get_brep(_r0).area(1e-7), store.get_brep(_r1).area(1e-7)
check("revolve seam angle: same area, seam turned to 120 deg", abs(_ar0 - _ar1) < 1e-5 * max(_ar0, 1.0)
      and (abs(_ang - 120) < 0.5 or abs(_ang - 120 + 360) < 0.5 or abs(_ang + 240) < 0.5 or abs(np.degrees(np.arctan2(_seam[1], _seam[0])) - 120) < 0.5),
      f"seam at {np.degrees(np.arctan2(_seam[1], _seam[0])):.1f} deg, areas {_ar0:.9f} / {_ar1:.9f}")

# revolve of a MULTI-SPAN profile covers the whole profile (kernel revolve only did span 1)
_cu = bpy.data.curves.new("RV_ms", 'CURVE'); _cu.dimensions = '3D'
_sp = _cu.splines.new('NURBS'); _pp = [(1.0, 0, 0), (1.4, 0, 0.8), (0.9, 0, 1.6), (1.0, 0, 3.0)]; _sp.points.add(3)
for _q, _c in zip(_sp.points, _pp): _q.co = (*_c, 1.0)
_sp.order_u = 3; _sp.use_endpoint_u = True
_ms = bpy.data.objects.new("RV_ms", _cu); ctx.collection.objects.link(_ms); _ms.location = (0, 0, -300)
bpy.ops.nproto.revolve(inputs=ins((_ms, "S0")), angle=360.0, axis="Z", pivot="OBJECT"); _rv = ctx.active_object
_kc = features.input_curve(_ms, "S0"); _t0, _t1 = _kc.parameter_range()
_Q = np.array([_kc.evaluate(t) for t in np.linspace(_t0, _t1, 4001)]); _Q[:, :2] -= np.array([0, 0])
_r = np.hypot(_Q[:, 0], _Q[:, 1]); _ds = np.linalg.norm(np.diff(_Q, axis=0), axis=1)
_pappus = float(np.sum(2 * np.pi * 0.5 * (_r[1:] + _r[:-1]) * _ds))
_bbr = store.get_brep(_rv).bounding_box()
check("revolve of a 2-span profile: full height and Pappus area", abs(_bbr[1][2] - _bbr[0][2] - 3.0) < 1e-3
      and abs(store.get_brep(_rv).area(1e-7) - _pappus) / _pappus < 1e-4,
      f"height {_bbr[1][2] - _bbr[0][2]:.4f} area {store.get_brep(_rv).area(1e-7):.4f} vs {_pappus:.4f}")
bpy.ops.nproto.revolve(inputs=ins((_ms, "S0")), angle=90.0, axis="Z", pivot="OBJECT"); _rq = ctx.active_object
check("revolve 90 deg: a quarter of the area", abs(store.get_brep(_rq).area(1e-7) - _pappus / 4) / _pappus < 1e-4)

# vase with a cylinder opening across its seam: trim, join, edge fillet the opening
bpy.ops.nproto.revolve(inputs=ins((_ms, "S0")), angle=360.0, axis="Z", pivot="OBJECT"); _va = ctx.active_object; _va.name = "VO_Vase"
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.5, height=3.0, degree="0", caps=False); _cy = ctx.active_object; _cy.name = "VO_Cyl"
_cy.rotation_euler = (0, math.radians(90), 0); _cy.location = (0, 0, -298.5); ctx.view_layer.update()
_bbc = store.get_brep(_cy).bounding_box(); _Mc = np.array(_cy.matrix_world)
_xs = [(_Mc[:3, :3] @ np.array([x, y, z]) + _Mc[:3, 3])[0] for x in (_bbc[0][0], _bbc[1][0]) for y in (_bbc[0][1], _bbc[1][1]) for z in (_bbc[0][2], _bbc[1][2])]
_cy.location.x += 0.3 - min(_xs); ctx.view_layer.update()
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="DELETE", scope="PICKED", target="VO_Vase", cutters=ins((_cy, "SURF")), ref_point=(0.97, 0, -298.5),
                    normal=(1, 0, 0), view_dir=(-1, 0, 0))
bpy.ops.nproto.trim(action="KEEP", scope="PICKED", target="VO_Cyl", cutters=ins((_va, "SURF")), ref_point=(0.5, 0, -298.0),
                    normal=(0, 0, 1), view_dir=(0, 0, -1))
bpy.ops.object.select_all(action='DESELECT'); _va.select_set(True); _cy.select_set(True); ctx.view_layer.objects.active = _va
bpy.ops.nproto.join(tolerance=1e-3); _vj = ctx.active_object
_a_before = store.get_brep(_vj).area(1e-6); _f_before = store.get_brep(_vj).face_count()
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_vj.name, f"E{i}"] for i in store.shared_edges(_vj)]), shape="CIRCULAR", radius=0.08)
check("vase opening across the seam: trim + join + edge fillet succeed", r == {'FINISHED'} and features.stack(_vj)
      and features.stack(_vj)[-1]["op"] == "edge_fillet" and store.get_brep(_vj).face_count() > _f_before
      and abs(store.get_brep(_vj).area(1e-6) - _a_before) > 1e-3,
      f"faces {_f_before} -> {store.get_brep(_vj).face_count()}")

# tangent chain picking: a rim split into 3 arcs is one chain; a 90 deg edge is not part of it
from blender_nurbs.ops_trim import tangent_chain as _tch
def _arc(a0, a1): return [Vector((math.cos(t), math.sin(t), 0)) for t in np.linspace(a0, a1, 20)]
_cands = [("O", "E0", _arc(0, 2)), ("O", "E1", _arc(2, 4)), ("O", "E2", _arc(4, 2 * math.pi)),
          ("O", "E3", [Vector((1, 0, 0)), Vector((1, 0, 1))])]
_ch = sorted(c[1] for c in _tch(_cands[1], _cands))
check("tangent chain: a rim in 3 pieces is picked together, the 90 deg edge is not", _ch == ["E0", "E1", "E2"], str(_ch))

# revolve + planar cap from its rim: exact rim -> clean join (no slivers / chunks); solid edge fillet;
# surface fillet trims only the strip under the fillet on the closed revolve
_rim = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([p[2] for p in geom.boundary_points(store.get_surfaces(_rv)[0], w, 8)]))
r = bpy.ops.nproto.planar(inputs=ins((_rv, _rim))); _cap = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _rv.select_set(True); _cap.select_set(True); ctx.view_layer.objects.active = _rv
bpy.ops.nproto.join(tolerance=1e-3); _rj = ctx.active_object
_bj = store.get_brep(_rj)
check("join revolve + planar cap: 2 faces, one shared rim edge, no slivers",
      _bj.face_count() == 2 and len(store.shared_edges(_rj)) == 1 and min(f.area() for f in _bj.faces()) > 1e-3,
      f"faces {[round(f.area(), 4) for f in _bj.faces()]} shared {len(store.shared_edges(_rj))}")
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_rj.name, f"E{i}"] for i in store.shared_edges(_rj)]), radius=0.1)
check("edge fillet on the revolve + cap rim succeeds", r == {'FINISHED'} and store.get_brep(_rj).face_count() == 3)
_ident = {"loc": [0, 0, 0], "axis": [0, 0, 1], "angle": 0.0, "scale": [1, 1, 1]}
_a0 = store.get_brep(_rv).area(1e-6)
_info, _br = jobs.run("fillet", {"a": _rv, "b": _cap}, ["fillet", "a", "b"], radius_a=-0.1, radius_b=-0.1, xsect=1,
                      tolerance=1e-4, trim_bases=True)
_ta = store.brep_from_text(_br["a"]).area(1e-6)
check("surface fillet on a closed revolve trims only the strip under the fillet", 0.9 * _a0 < _ta < _a0,
      f"{_a0:.4f} -> {_ta:.4f}")

# surface fillet operator with Auto Side finds the working side even if other sides raise kernel errors
bpy.ops.object.select_all(action='DESELECT')
_rv2 = [o for o in bpy.data.objects if o.name.startswith("Revolve") and store.is_nurbs(o) and features.history(o)
        and features.history(o)["op"] == "revolve"][-1]
_rv2.hide_set(False); _cap.hide_set(False)
_rv2.select_set(True); _cap.select_set(True); ctx.view_layer.objects.active = _cap
r = bpy.ops.nproto.fillet(surf_a=_rv2.name, surf_b=_cap.name, radius=0.1, flip_a=False, flip_b=True, auto_flip=True,
                          trim_bases=True)
check("surface fillet (revolve + cap) with Auto Side succeeds from a failing start side", r == {'FINISHED'})

# surface fillet where the intersection runs across a revolve's seam (seam moved internally)
bpy.ops.nproto.revolve(inputs=ins((_ms, "S0")), angle=360.0, axis="Z", pivot="OBJECT"); _sv = ctx.active_object
_seam_pt = np.array(geom.world_matrix(_sv) @ Vector(store.get_surfaces(_sv)[0].evaluate(0.0, 0.5)))
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.35, height=3.0, degree="0", caps=False); _sc2 = ctx.active_object
_d = _seam_pt[:2] / np.linalg.norm(_seam_pt[:2] - np.array([0, 0])) if np.linalg.norm(_seam_pt[:2]) > 0 else np.array([1, 0])
_sc2.rotation_euler = (0, math.radians(90), math.atan2(_d[1], _d[0])); _sc2.location = (0, 0, _seam_pt[2]); ctx.view_layer.update()
_ok = False
for _sa in (1, -1):
    for _sb in (1, -1):
        try:
            jobs.run("fillet", {"a": _sc2, "b": _sv}, ["fillet"], radius_a=_sa * 0.08, radius_b=_sb * 0.08, xsect=2,
                     tolerance=1e-4, trim_bases=False)
            _ok = True
        except jobs.JobError:
            pass
check("surface fillet across a revolve's seam is found (seam moved internally)", _ok)

# --- Align Curve ---------------------------------------------------------------
from blender_nurbs import align as _al
bpy.ops.nproto.add_primitive(kind="ARC", radius=2.0, angle=90.0, degree="0"); _aa = ctx.active_object; _aa.location = (0, 0, -330)
bpy.ops.nproto.extrude(inputs=ins((_aa, "S0")), distance=1.5, direction="Z"); _asf = ctx.active_object; _asf.name = "AL_Surf"
_ss = store.get_surfaces(_asf)[0]
_edges = [w for w in ("U0", "U1", "V0", "V1")
          if np.ptp([p[2] for p in geom.boundary_points(_ss, w, 8)]) < 1e-6]       # the two arc edges (z const)
_lo, _hi = sorted(_edges, key=lambda w: geom.boundary_points(_ss, w, 2)[0][2])
# a drawn NURBS curve near the arc's end at (0, 2): from below the low arc end to the high arc end
_cu = bpy.data.curves.new("AL_c", 'CURVE'); _cu.dimensions = '3D'; _sp = _cu.splines.new('NURBS')
_pp = [(-0.2, 2.3, -330.0), (-0.8, 2.6, -330.0), (-1.4, 2.6, -329.6), (-1.4, 2.6, -328.9), (-0.8, 2.6, -328.5), (-0.2, 2.3, -328.5)]
_sp.points.add(len(_pp) - 1)
for _q, _c in zip(_sp.points, _pp): _q.co = (*_c, 1.0)
_sp.order_u = 4; _sp.use_endpoint_u = True
_ac = bpy.data.objects.new("AL_c", _cu); ctx.collection.objects.link(_ac)
r1 = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ac.name, "S0"], [_asf.name, _lo]]), end=0, cont="G2", mode="EDGE")
r2 = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ac.name, "S0"], [_asf.name, _hi]]), end=1, cont="G2", mode="EDGE")
_m0 = _al.measure(_ac, 0, _asf, _lo); _m1 = _al.measure(_ac, 1, _asf, _hi)
check("align curve: both ends G2 to the surface's arc edges (gap, angle, curvature)", r1 == r2 == {'FINISHED'}
      and _m0[0] < 1e-6 and _m1[0] < 1e-6 and _m0[1] < 1e-4 and _m1[1] < 1e-4 and _m0[2] < 1e-4 and _m1[2] < 1e-4,
      f"{_m0} {_m1}")
check("align curve: history holds both ends", features.history(_ac)["op"] == "align_curve"
      and set(features.history(_ac)["params"]["ends"]) == {"0", "1"})
_mid_before = np.array(_ac.data.splines[0].points[len(_ac.data.splines[0].points) // 2].co[:3])
_asf.location.x += 0.5; settle()
_m0 = _al.measure(_ac, 0, _asf, _lo); _m1 = _al.measure(_ac, 1, _asf, _hi)
check("align curve follows the surface (history), same 6 CVs", _m0[0] < 1e-6 and _m1[0] < 1e-6 and _m0[1] < 1e-4
      and _m0[2] < 1e-4 and len(_ac.data.splines[0].points) == 6, f"{_m0} {_m1}")
# tangent to surface: a curve that leaves the low arc edge end; its start tangent lies in the surface's tangent plane
_cu2 = bpy.data.curves.new("AL_d", 'CURVE'); _cu2.dimensions = '3D'; _sp2 = _cu2.splines.new('NURBS')
_pp2 = [(0.5, 2.5, -330.2), (0.3, 2.0, -329.5), (0.6, 1.5, -329.0), (1.2, 1.0, -328.6), (1.6, 0.8, -328.2)]
_sp2.points.add(len(_pp2) - 1)
for _q, _c in zip(_sp2.points, _pp2): _q.co = (*_c, 1.0)
_sp2.order_u = 4; _sp2.use_endpoint_u = True
_ad = bpy.data.objects.new("AL_d", _cu2); ctx.collection.objects.link(_ad)
r = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ad.name, "S0"], [_asf.name, _lo]]), end=0, cont="G1", mode="SURFACE")
_kc = features.input_curve(_ad, "S0"); _P, _D, _ = geom.curve_end_frame(_kc, 0)
_n, _ = _al.surface_frame(_asf, _lo, _P)
check("align curve, Tangent to Surface: start on the edge end, tangent in the surface's tangent plane", r == {'FINISHED'}
      and abs(float(_n @ (_D / np.linalg.norm(_D)))) < 1e-6, f"n.t = {float(_n @ (_D / np.linalg.norm(_D))):.2e}")

_cu3 = bpy.data.curves.new("AL_e", 'CURVE'); _cu3.dimensions = '3D'; _sp3 = _cu3.splines.new('NURBS')
_pp3 = [(-0.2, 2.3, -330.0), (-1.4, 2.6, -329.6), (-1.4, 2.6, -328.9), (-0.2, 2.3, -328.5)]
_sp3.points.add(3)
for _q, _c in zip(_sp3.points, _pp3): _q.co = (*_c, 1.0)
_sp3.order_u = 4; _sp3.use_endpoint_u = True
_ae = bpy.data.objects.new("AL_e", _cu3); ctx.collection.objects.link(_ae)
bpy.ops.nproto.align_curve(inputs=json.dumps([[_ae.name, "S0"], [_asf.name, _lo]]), end=0, cont="G2")
r = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ae.name, "S0"], [_asf.name, _hi]]), end=1, cont="G2")
_m0 = _al.measure(_ae, 0, _asf, _lo); _m1 = _al.measure(_ae, 1, _asf, _hi)
check("align a 4-CV curve G2 at both ends: refit to enough CVs, both ends G2", r == {'FINISHED'}
      and len(_ae.data.splines[0].points) >= 6 and max(_m0[0], _m1[0]) < 1e-6 and max(_m0[1], _m1[1]) < 1e-4
      and max(_m0[2], _m1[2]) < 1e-4, f"{len(_ae.data.splines[0].points)} CVs {_m0} {_m1}")
bpy.ops.curve.primitive_bezier_curve_add(location=(0, 0, 0)); _ab = ctx.active_object; _ab.name = "AL_b"
_bpts = _ab.data.splines[0].bezier_points; _ab.data.splines[0].bezier_points.add(1)
for _b, (_co, _hl, _hr) in zip(_ab.data.splines[0].bezier_points, [((-0.2, 2.3, -330), (0, 2, -330), (-0.8, 2.6, -330)),
                                                                  ((-1.4, 2.6, -329.2), (-1.4, 2.6, -329.8), (-1.4, 2.6, -328.7)),
                                                                  ((-0.2, 2.3, -328.5), (-0.8, 2.6, -328.5), (0, 2, -328.5))]):
    _b.co, _b.handle_left, _b.handle_right = _co, _hl, _hr
bpy.ops.nproto.align_curve(inputs=json.dumps([[_ab.name, "S0"], [_asf.name, _lo]]), end=0, cont="G2")
r = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ab.name, "S0"], [_asf.name, _hi]]), end=1, cont="G2")
_m0 = _al.measure(_ab, 0, _asf, _lo); _m1 = _al.measure(_ab, 1, _asf, _hi)
check("align a Bezier curve G2 at both ends (handles), stays Bezier", r == {'FINISHED'} and _ab.data.splines[0].type == 'BEZIER'
      and max(_m0[0], _m1[0]) < 1e-5 and max(_m0[1], _m1[1]) < 1e-3 and max(_m0[2], _m1[2]) < 1e-3, f"{_m0} {_m1}")

# continuity check catches a 90 degree corner
cA = patch("CCa", (0, Y0 + 50, 0), (0, Y0 + 54, 0), "X", 2.0)      # floor 0..2 in x
cB = patch("CCb", (2, Y0 + 50, 0), (2, Y0 + 54, 0), "Z", 2.0)      # wall standing on its edge x=2
bpy.ops.object.select_all(action='DESELECT'); cA.select_set(True); cB.select_set(True)
if "nproto_last_continuity" in ctx.scene: del ctx.scene["nproto_last_continuity"]
r = bpy.ops.nproto.continuity()
rep2 = json.loads(ctx.scene.get("nproto_last_continuity", "{}"))
check("continuity check reports a 90 deg crease", r == {'FINISHED'} and rep2.get("angle", 0) > 80, str(rep2))
# history: move the sweep's rail -> the sweep follows
a_before = SA(sw)
rl.data.splines[0].points[1].co = (0, Y0, 6, 1); ctx.view_layer.update(); history._process()
check("sweep follows its rail (history)", abs(SA(sw) - 2 * math.pi * 0.3 * 6) < 2e-3, f"{a_before:.4f} -> {SA(sw):.4f}")

# --- display/brep caches survive undo-like revision reuse
cc = [o for o in bpy.data.objects if store.is_nurbs(o)][0]
txt_a = cc[store.PROP_BREP]; rev_a = cc.get(store.PROP_REV)
store.display(cc); store.get_brep(cc)          # caches hold state A
store.set_geometry_text(cc, sw[store.PROP_BREP], ctx, keep_untrimmed=False)   # state B
store._disp_cache[cc.session_uid] = (cc[store.PROP_REV], {"edges": []})      # stale entry with B's rev
cc[store.PROP_REV] = rev_a                      # "undo" back to A's revision
store._disp_cache.pop(cc.session_uid, None); store.display(cc)
store.set_geometry_text(cc, txt_a, ctx, keep_untrimmed=False)                 # new edit after undo
check("revision tokens never repeat after undo", cc[store.PROP_REV] != rev_a)
check("display cache follows the geometry", len(store.display(cc).get("edges", [])) == store.get_brep(cc).edges().__len__())



# --- Loft keeps the inputs' layout along the edge (planes from lines: no refit to degree 3)
la = nurbs("LyA", [(-1, 90, 0), (1, 90, 0)], order=2); lb = nurbs("LyB", [(-1, 93, 0), (1, 93, 0)], order=2)
EA = store.new_object(ctx, "LyEA", features.run("extrude", [(la, "S0")], {"distance": 1.0, "direction": "Z", "dir_vec": [0, 0, 1]}))
EB = store.new_object(ctx, "LyEB", features.run("extrude", [(lb, "S0")], {"distance": 1.0, "direction": "Z", "dir_vec": [0, 0, 1]}))
top = max(("U0", "U1", "V0", "V1"), key=lambda r: geom.sample_curve(features.input_curve(EA, r), 3)[:, 2].min())
lay = {}
for cont in ("G0", "G1", "G2"):
    d = ext.surface_cvs(features.run("loft", [(EA, top), (EB, top)], {"cont_a": cont, "cont_b": cont}))
    lay[cont] = (d["degree_v"], d["count_v"], d["count_u"])
check("loft between degree-1 edges stays degree 1 along the edge", lay == {"G0": (1, 2, 2), "G1": (1, 2, 4), "G2": (1, 2, 6)}, str(lay))


# --- Sweep: a profile standing on the rail's END sweeps from there
srail = nurbs("SwRail", [(0, 100, 0), (0, 101.5, 0), (2, 101.5, 0), (2, 100, 0)], order=4)
sp_end = nurbs("SwProfEnd", [(2, 100, 0), (2, 100, 1)], order=2)
sp_start = nurbs("SwProfStart", [(0, 100, 0), (0, 100, 1)], order=2)
bb = {}
for pr in (sp_end, sp_start):
    d = ext.surface_cvs(features.run("sweep", [(pr, "S0"), (srail, "S0")], {"samples": 32}))
    X = np.array(d["points"]); bb[pr.name] = (X.min(0), X.max(0))
check("sweep from the rail's end matches the sweep from its start",
      all(np.allclose(bb["SwProfEnd"][i], bb["SwProfStart"][i], atol=1e-3) for i in (0, 1)) and bb["SwProfEnd"][1][1] < 101.6,
      str({k: (np.round(v[0], 3).tolist(), np.round(v[1], 3).tolist()) for k, v in bb.items()}))


# --- Loft Rails: profiles + 0 / 1 / 2 rails, G0 / G1 / G2
from blender_nurbs import loft_rails, square as _sqm
def _lr_dev(surf, objs):
    out = []
    for o, r in objs:
        for x in geom.sample_curve(features.input_curve(o, r), 21):
            u, v = _sqm._foot(surf, x)
            out.append(float(np.linalg.norm(np.array(surf.evaluate(u, v)) - x)))
    return max(out)
lp0 = nurbs("LRp0", [(0, 120, 0), (4, 120, 0)], order=2); lp1 = nurbs("LRp1", [(0, 124, 0), (4, 124, 0)], order=2)
lr0 = nurbs("LRr0", [(0, 120, 0), (0, 121, 0.6), (0, 122, 0.9), (0, 123, 0.6), (0, 124, 0)])
lr1 = nurbs("LRr1", [(4, 120, 0), (4, 121, 0.6), (4, 122, 0.9), (4, 123, 0.6), (4, 124, 0)])
zm = geom.sample_curve(features.input_curve(lr0, "S0"), 201)[100][2]
lpm = nurbs("LRpm", [(0, 122, zm), (1.3, 122, zm + 1.0), (2.7, 122, zm + 1.0), (4, 122, zm)])
ins5 = [(lp0, "S0"), (lr1, "S0"), (lpm, "S0"), (lr0, "S0"), (lp1, "S0")]      # any pick order after the 1st profile
g = features.run("loft_rails", ins5, {})
inf = dict(loft_rails.LAST_INFO)
check("loft rails: 3 profiles + 2 rails, through every input", inf["profiles"] == 3 and inf["rails"] == 2 and
      _lr_dev(g, ins5) < 1e-4 and inf["folds"] == 0, f"{inf['cvs']} dev {_lr_dev(g, ins5):.1e}")
lg = nurbs("LRguide", [(2, 120, 0), (2, 121.3, 1.0), (2, 122.7, 1.0), (2, 124, 0)])
g = features.run("loft_rails", [(lp0, "S0"), (lp1, "S0"), (lg, "S0")], {})
check("loft rails: 1 guide rail through the middle", loft_rails.LAST_INFO["rails"] == 1 and
      _lr_dev(g, [(lp0, "S0"), (lp1, "S0"), (lg, "S0")]) < 1e-4, f"{_lr_dev(g, [(lp0, 'S0'), (lp1, 'S0'), (lg, 'S0')]):.1e}")
g = features.run("loft_rails", [(lp0, "S0"), (lpm, "S0"), (lp1, "S0")], {})
check("loft rails: 3 profiles, no rails", loft_rails.LAST_INFO["rails"] == 0 and
      _lr_dev(g, [(lp0, "S0"), (lpm, "S0"), (lp1, "S0")]) < 1e-4)
gs = features.run("square", [(lp0, "S0"), (lr1, "S0"), (lp1, "S0"), (lr0, "S0")], {})
g = features.run("loft_rails", [(lp0, "S0"), (lp1, "S0"), (lr0, "S0"), (lr1, "S0")], {})
pts = np.array([gs.evaluate(*(np.array(gs.u_domain()) @ [1 - a, a], np.array(gs.v_domain()) @ [1 - b, b]))
                for a in np.linspace(0, 1, 7) for b in np.linspace(0, 1, 7)])
dmax = max(float(np.linalg.norm(np.array(g.evaluate(*_sqm._foot(g, x))) - x)) for x in pts)
check("loft rails: 2 profiles + 2 rails = Square", dmax < 1e-3, f"{dmax:.1e}")
def _lr_wall(name, a, b):
    c = nurbs(name + "_c", [a, b], order=2)
    o = store.new_object(ctx, name, features.run("extrude", [(c, "S0")], {"distance": 1.0, "direction": "Z", "dir_vec": [0, 0, 1]}))
    return o, max(("U0", "U1", "V0", "V1"), key=lambda r: geom.sample_curve(features.input_curve(o, r), 3)[:, 2].min())
wa = _lr_wall("LRwa", (0, 130, 0), (4, 130, 0)); wb = _lr_wall("LRwb", (0, 134, 0), (4, 134, 0))
wg = nurbs("LRwg", [(2, 130, 1), (2, 130, 1.8), (2, 134, 1.8), (2, 134, 1)])
for cont, lim in (("G1", 0.1), ("G2", 0.1)):
    g = features.run("loft_rails", [wa, wb], {"lr_start": cont, "lr_end": cont})
    inf = dict(loft_rails.LAST_INFO)
    check(f"loft rails: {cont} to both walls", inf["angle"] < 0.1 and inf["curv"] <= lim and inf["folds"] == 0,
          f"{inf['cvs']} angle {inf['angle']:.4f} curv {inf['curv']:.3f}")
g = features.run("loft_rails", [wa, wb, (wg, "S0")], {"lr_start": "G1", "lr_end": "G1"})
inf = dict(loft_rails.LAST_INFO)
check("loft rails: G1 to both walls + guide rail", inf["rails"] == 1 and inf["angle"] < 0.1 and
      _lr_dev(g, [wa, wb, (wg, "S0")]) < 1e-3, f"{inf['cvs']} angle {inf['angle']:.4f}")
r = bpy.ops.nproto.loft_rails(inputs=ins(*ins5))
check("loft rails operator + history", r == {'FINISHED'} and features.history(ctx.active_object)["op"] == "loft_rails")


# the inputs' own layout: degree-1 profiles x the rails' layout, no densifying
d = ext.surface_cvs(features.run("loft_rails", [(lp0, "S0"), (lp1, "S0"), (lr0, "S0"), (lr1, "S0")], {}))
check("loft rails keeps the inputs' layout (degree-1 profiles, rails' CVs)",
      (d["degree_u"], d["count_u"], d["degree_v"], d["count_v"]) == (1, 2, 3, 5), str((d["degree_u"], d["count_u"], d["degree_v"], d["count_v"])))
# one rail: the free edge follows the rail; G1 to two walls keeps the net even
wv1 = _lr_wall("LRwv1", (-2, 150, 0), (0, 150, 0)); wv2 = _lr_wall("LRwv2", (-2, 152, 0), (0, 152, 0))
def _vedge(o):
    return min(("U0", "U1", "V0", "V1"), key=lambda r: np.linalg.norm(geom.sample_curve(features.input_curve(o, r), 2)[:, :2].mean(axis=0) - (0, 151)))
bc = nurbs("LRbend", [(0, 150, 0), (0.6, 150, 0), (1.2, 150, 0), (1.2, 152, 0), (0.6, 152, 0), (0, 152, 0)], order=6)   # G2 to the walls, like a Blend Curve
for cont in ("G0", "G1", "G2"):
    g = features.run("loft_rails", [(wv1[0], _vedge(wv1[0])), (wv2[0], _vedge(wv2[0])), (bc, "S0")], {"lr_start": cont, "lr_end": cont})
    d = ext.surface_cvs(g); X = np.array(d["points"]).reshape(d["count_u"], d["count_v"], 3)
    zs = np.ptp(X[:, :, 2], axis=1)            # every CV row along the rail stays level
    inf = dict(loft_rails.LAST_INFO)
    check(f"loft rails, 1 rail at the edge, {cont}: light even net, free edge follows the rail",
          d["count_u"] * d["count_v"] <= 12 and zs.max() < 1e-6 and inf["angle"] < 0.1,
          f"{(d['count_u'], d['count_v'])} level {zs.max():.1e} angle {inf['angle']:.3f}")

# Planar from a curved surface's edge + a closing edge: a picked edge of a curved
# surface no longer drags its whole (non-planar) boundary loop along
from blender_nurbs import ops as _ops
bend_s = store.new_object(ctx, "PLbend", features.run("extrude", [(bc, "S0")], {"distance": 1.0, "direction": "Z", "dir_vec": [0, 0, 1]}))
b_ = store.get_brep(bend_s)
top_ref = max((f"E{i}" for i in range(len(b_.edges()))),
              key=lambda r: min(e[2] for e in [b_.edges()[int(r[1:])].point_at(t) for t in b_.edges()[int(r[1:])].parameter_range()]))
loop = _ops.edge_loop_refs(bend_s, top_ref)
polys = [[Vector(b_.edges()[int(r[1:])].point_at(t)) for t in np.linspace(*b_.edges()[int(r[1:])].parameter_range(), 9)] for r in loop]
check("planar picking: a curved surface's boundary loop is not planar (single edge is taken)", len(loop) > 1 and not _ops._polylines_planar(polys))
close = nurbs("PLclose", [(0, 150, 1), (0, 152, 1)], order=2)
g = features.run("planar", [(bend_s, top_ref), (close, "S0")], {})
check("planar from a curved surface's edge + a closing curve", g is not None)

# picking: an edge that continues the profile is taken alone, even on a planar face
import types as _types
cap = store.new_object(ctx, "PLcap", g)
def _cands_of(o):
    b = store.get_brep(o); out = []
    for i, e in enumerate(b.edges()):
        out.append((o, f"E{i}", [Vector(e.point_at(t)) for t in np.linspace(*e.parameter_range(), 9)]))
    return out
cands = _cands_of(bend_s) + _cands_of(cap)
fake = _types.SimpleNamespace(chain=True, _cands=cands, _picks=[])
fake._touches_picked = lambda c: _ops.NPROTO_OT_planar._touches_picked(fake, c)
cap_edges = [c for c in cands if c[0] is cap]
straight = min(cap_edges, key=lambda c: (c[2][0] - c[2][-1]).length - sum((a - b).length for a, b in zip(c[2][:-1], c[2][1:])) * -1)
first = _ops.NPROTO_OT_planar.expand_pick(fake, straight)
fake._picks = [(bend_s, top_ref)]
second = _ops.NPROTO_OT_planar.expand_pick(fake, straight)
check("planar picking: a planar face's loop on the first pick, a single edge when continuing a profile",
      len(first) == len(cap_edges) and len(second) == 1, f"{len(first)} / {len(second)}")

# --- Loft Rails, Centreline mode: profiles follow a spine they don't touch
import math as _m
cl_spine = nurbs("CLspine", [(0, 140, 0), (0, 143, 0), (3, 146, 0), (6, 146, 0)])
def _circle_at(name, c, nrm, r):
    from blender_nurbs import primitives
    o = bpy.data.objects.new(name, bpy.data.curves.new(name, 'CURVE')); ctx.collection.objects.link(o)
    primitives.write_spline_object(o, primitives.exact_circle_spec(r))
    o.location = c
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(nrm))
    return o
c0 = _circle_at("CLc0", (0, 140, 0), (0, 1, 0), 0.5)
c1 = _circle_at("CLc1", (6, 146, 0), (1, 0, 0), 0.3)
bpy.context.view_layer.update()
cin = [(c0, "S0"), (c1, "S0"), (cl_spine, "S0")]
g = features.run("loft_rails", cin, {"lr_mode": "CENTRELINE"})
inf = dict(loft_rails.LAST_INFO)
# the sections follow the spine: the spine's middle point is (nearly) at the centre of the section there
mid = geom.sample_curve(features.input_curve(cl_spine, "S0"), 3)[1]
(u0, u1), (v0, v1) = g.u_domain(), g.v_domain()
best = min(range(41), key=lambda i: np.linalg.norm(np.mean([g.evaluate(u0 + (u1 - u0) * i / 40, v0 + (v1 - v0) * j / 24)
                                                              for j in range(24)], axis=0) - mid))
ring = np.array([g.evaluate(u0 + (u1 - u0) * best / 40, v0 + (v1 - v0) * j / 24) for j in range(24)])
rad = np.linalg.norm(ring - mid, axis=1)
check("centreline loft: through both profiles, exact rational profiles", inf["gap"] < 1e-4 and inf["exact_profiles"],
      f"gap {inf['gap']:.1e} {inf['cvs']}")
check("centreline loft follows the spine (round section around it)", np.ptp(rad) < 0.05 and 0.29 < rad.mean() < 0.51,
      f"radius {rad.mean():.3f} +- {np.ptp(rad):.3f}")
rads = []
for i in range(41):
    rg = np.array([g.evaluate(u0 + (u1 - u0) * i / 40, v0 + (v1 - v0) * j / 24) for j in range(24)])
    rads.append(np.linalg.norm(rg - rg.mean(axis=0), axis=1).mean())
check("centreline loft: the section radius goes evenly from 0.5 to 0.3 (no shrink from turned seams)",
      min(rads) > 0.29 and max(rads) < 0.51 and all(b <= a + 1e-3 for a, b in zip(rads[:-1], rads[1:])),
      f"{min(rads):.3f}..{max(rads):.3f}")

sp_pts = geom.sample_curve(features.input_curve(cl_spine, "S0"), 201)
q = sp_pts[100]
tq = sp_pts[101] - sp_pts[99]; tq /= np.linalg.norm(tq)
e1 = np.cross(tq, (0, 0, 1)); e1 /= np.linalg.norm(e1); e2 = np.cross(tq, e1)
sq_pts = [tuple(q + 0.4 * (a * e1 + b * e2)) for a, b in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 0))]
lsq = nurbs("CLdiamond", sq_pts, order=2)
g = features.run("loft_rails", [(c0, "S0"), (lsq, "S0"), (c1, "S0"), (cl_spine, "S0")], {"lr_mode": "CENTRELINE"})
inf = dict(loft_rails.LAST_INFO)
check("centreline loft: 3 mixed profiles (circle, diamond, circle) refitted to one layout",
      inf["profiles"] == 3 and not inf["exact_profiles"] and inf["gap"] < 1e-2, f"gap {inf['gap']:.1e} {inf['cvs']}")

r = bpy.ops.nproto.loft_rails(inputs=ins(*cin), lr_mode="CENTRELINE")
check("centreline loft operator + history", r == {'FINISHED'} and
      features.history(ctx.active_object)["params"]["lr_mode"] == "CENTRELINE")

# --- Join Curves / Unjoin Curves
from blender_nurbs import curvejoin, primitives
ctx.scene.nproto_settings.construction_history = True
def _samples(o, n=300):
    return np.vstack([geom.sample_curve(features.input_curve(o, f"S{i}"), n) for i in range(len(o.data.splines))])
def _dev(src, dst):
    D = _samples(dst, 20000)
    return max(np.min(np.linalg.norm(D - x, axis=1)) for x in _samples(src, 120))
jl1 = nurbs("JL1", [(-4, 60, 0), (0, 60, 0)], order=2)
jb = nurbs("JB", [(0, 60, 0), (1.5, 60, 0), (2.2, 61, 0), (2.2, 62, 0), (1.5, 63, 0), (0, 63, 0)], order=4)
jl2 = nurbs("JL2", [(-4, 63, 0), (0.0005, 63, 0)], order=2)       # points the wrong way, small gap
srcs = [jl1, jb, jl2]
ref_pts = {o.name: _samples(o) for o in srcs}
for o in ctx.selected_objects: o.select_set(False)
for o in srcs: o.select_set(True)
r = bpy.ops.nproto.join_curves(names=json.dumps([o.name for o in srcs]), tolerance=0.001)
J = ctx.active_object
info = dict(curvejoin.LAST_INFO)
check("join curves: one spline, 2 joints", r == {'FINISHED'} and len(J.data.splines) == 1 and info["joints"] == 2,
      str(info))
dev = max(_dev(o, J) for o in srcs)
check("join curves exact: shape unchanged", dev < 1e-3, f"{dev:.2e}")
check("join curves: history + sources hidden", features.history(J) is not None and all(o.hide_get() for o in srcs))
jb.data.splines[0].points[2].co = (2.6, 61, 0, 1); jb.data.splines[0].points[3].co = (2.6, 62, 0, 1)
features.rebuild(J, ctx)
check("join curves follows its inputs", _dev(jb, J) < 1e-3, f"{_dev(jb, J):.2e}")
r = bpy.ops.nproto.join_curves(names=json.dumps([o.name for o in srcs]), tolerance=0.001, jc_mode="REBUILD")
JR = ctx.active_object
sp = JR.data.splines[0]
check("join curves rebuild: one uniform spline", r == {'FINISHED'} and not sp.use_bezier_u and
      curvejoin.LAST_INFO["fit_error"] < 1e-3 * 7, str(dict(curvejoin.LAST_INFO)))
for o in ctx.selected_objects: o.select_set(False)
J.select_set(True); ctx.view_layer.objects.active = J
r = bpy.ops.nproto.unjoin_curves()
parts = [o for o in ctx.selected_objects]
degs = sorted(o.data.splines[0].order_u - 1 for o in parts)
check("unjoin curves: back to 3 pieces", r == {'FINISHED'} and len(parts) == 3 and degs == [1, 1, 3], str(degs))
devs = [min(max(np.min(np.linalg.norm(_samples(p, 20000) - x, axis=1)) for x in _samples(o, 60)) for p in parts) for o in srcs]
check("unjoin curves: pieces match the sources", max(devs) < 1e-3, str([f"{d:.1e}" for d in devs]))
# rational arc + line: exact join keeps the arc
arc = bpy.data.objects.new("JArc", bpy.data.curves.new("JArc", 'CURVE')); ctx.collection.objects.link(arc)
primitives.write_spline_object(arc, primitives.exact_arc_spec(1.0, 90.0))
arc.location = (0, 70, 0)
ln = nurbs("JLn", [(1, 70, 0), (1, 67, 0)], order=2)
bpy.context.view_layer.update()
r = bpy.ops.nproto.join_curves(names=json.dumps([arc.name, ln.name]), tolerance=0.001)
JA = ctx.active_object
P = _samples(JA, 400)
rad = np.linalg.norm(P - np.array([0, 70, 0]), axis=1)
on_arc = P[:, 1] >= 70 + 1e-6
check("join curves keeps a rational arc exact", r == {'FINISHED'} and np.abs(rad[on_arc] - 1).max() < 1e-6,
      f"{np.abs(rad[on_arc] - 1).max():.1e}")
# smooth single curve: nothing to unjoin; two-spline object comes apart
for o in ctx.selected_objects: o.select_set(False)
sm1 = nurbs("JSmooth", [(0, 80, 0), (1, 81, 0), (2, 80, 0), (3, 81, 0), (4, 80, 0)], order=4)
sm1.select_set(True); ctx.view_layer.objects.active = sm1
r = bpy.ops.nproto.unjoin_curves()
check("unjoin curves leaves a smooth curve alone", r == {'CANCELLED'})
sp2 = sm1.data.splines.new('NURBS'); sp2.points.add(1)
sp2.points[0].co = (0, 85, 0, 1); sp2.points[1].co = (3, 85, 0, 1); sp2.order_u = 2
r = bpy.ops.nproto.unjoin_curves()
check("unjoin curves separates splines", r == {'FINISHED'} and len(ctx.selected_objects) == 2)
# Join button routes curves to Join Curves
for o in ctx.selected_objects: o.select_set(False)
for o in parts[:2]: o.select_set(True)
ctx.view_layer.objects.active = parts[0]
with ctx.temp_override(selected_objects=parts[:2], active_object=parts[0]):
    r = bpy.ops.nproto.join('INVOKE_DEFAULT')
check("Join with curves selected joins curves", r == {'FINISHED'} and ctx.active_object.type == 'CURVE')

# --- Save / reload
path = os.path.join(tempfile.gettempdir(), "nurbs_test.blend")
bpy.ops.wm.save_as_mainfile(filepath=path)
bpy.ops.wm.open_mainfile(filepath=path)
store._cache.clear()
L2 = bpy.data.objects["Loft"]
b = store.get_brep(L2)
check("reload brep", b.face_count() == 1)
blend2 = bpy.data.objects[G2.name] if False else [o for o in bpy.data.objects if o.name.startswith("Loft") and features.history(o)][-1]
check("reload history intact", features.history(blend2) is not None)

# --- Class-A edge fillet: a U-bend wall (straight / degree-5 G2 bend / straight) with a flat top
_Y = -0.9144
_l1 = nurbs("CA_L1", [(0, 0, 0), (1, 0, 0)], order=2)
_bd = nurbs("CA_Bend", [(1, 0, 0), (1.6, 0, 0), (2.2, 0, 0), (2.2, _Y, 0), (1.6, _Y, 0), (1, _Y, 0)], order=6)
_l2 = nurbs("CA_L2", [(1, _Y, 0), (0, _Y, 0)], order=2)
_cl = nurbs("CA_Close", [(0, _Y, 1), (0, 0, 1)], order=2)
_walls = []
for _c in (_l1, _bd, _l2):
    bpy.ops.nproto.extrude(inputs=ins((_c, "S0")), distance=1.0, direction="Z"); _walls.append(ctx.active_object)
def _top(o):
    return max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([p[2] for p in geom.boundary_points(store.get_surfaces(o)[0], w, 8)]))
r = bpy.ops.nproto.planar(inputs=ins(*[(w, _top(w)) for w in _walls], (_cl, "S0"))); _capU = ctx.active_object
check("class-a scene: planar top from the wall rims", r == {'FINISHED'} and store.is_nurbs(_capU))
bpy.ops.object.select_all(action='DESELECT')
for o in _walls + [_capU]:
    o.select_set(True)
ctx.view_layer.objects.active = _walls[0]
bpy.ops.nproto.join(tolerance=1e-3); _uj = ctx.active_object
_bju = store.get_brep(_uj)
from blender_nurbs.ops_trim import _edge_mid_world as _emw
_top_edges = [i for i in store.shared_edges(_uj) if abs(_emw(_uj, i)[2] - 1.0) < 1e-3]
check("class-a scene: join gives 3 top edges to fillet", len(_top_edges) == 3, str(_top_edges))
_ujk = _uj.copy(); _ujk.data = _uj.data.copy(); ctx.collection.objects.link(_ujk)   # kernel-only reference
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_ujk.name, f"E{i}"] for i in _top_edges]), radius=0.1)
def _dense(o):
    out = []
    for f in store.get_brep(o).faces():
        d = ext.surface_cvs(f.surface())
        if d["count_u"] > d["degree_u"] + 1 or d["count_v"] > d["degree_v"] + 1:
            out.append((d["count_u"], d["count_v"]))
    return out
check("kernel fillet makes a dense face along the bend", r == {'FINISHED'} and any(max(c) > 12 for c in _dense(_ujk)),
      str(_dense(_ujk)))
def _lamina(o):
    b = store.get_brep(o)
    return sum(1 for e in b.edges() if len(e.faces()) == 1), sum(1 for e in b.edges() if len(e.faces()) > 2)
_cnt = {}
for _tol in (1e-3, 1e-4):
    _o = _uj.copy(); _o.data = _uj.data.copy(); ctx.collection.objects.link(_o)
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_o.name, f"E{i}"] for i in _top_edges]), radius=0.1,
                                   class_a=True, ca_tolerance=_tol, ca_continuity="2")
    _rep = json.loads(_o.get("nproto_class_a", "{}"))
    _fr = [f for f in _rep.get("faces", []) if "skipped" not in f]
    _cnt[_tol] = sum(f["spans"] for f in _fr)
    check(f"class-a fillet (tol {_tol}): every fillet face single-span, within tolerance, G2 ends",
          r == {'FINISHED'} and _fr and not _dense(_o) and all(f["within"] and f["dev"] <= _tol and f["angle"] <= 0.1
                                                                  and f["ends"] == ["G2", "G2"] for f in _fr),
          f"{_rep.get('error', '')} {[(f['spans'], f['degree'], round(f['dev'], 6)) for f in _fr]}")
    check(f"class-a fillet (tol {_tol}): stitched like the kernel fillet (same open edges, no non-manifold)",
          _lamina(_o) == (_lamina(_ujk)[0], 0), f"{_lamina(_o)} vs kernel {_lamina(_ujk)}")
    if _tol == 1e-4:
        _st = features.stack(_o)[-1]
        _nf = store.get_brep(_o).face_count()
        features.rebuild(_o, ctx)
        check("class-a fillet is a history step and replays", _st["params"].get("class_a") is True
              and store.get_brep(_o).face_count() == _nf)
check("class-a: a tighter tolerance uses more patches", _cnt[1e-4] >= _cnt[1e-3] >= 1, str(_cnt))

# Class-A on a Blend fillet of the symmetric U: straights exact (degree 1 along), bend symmetric
_ob = _uj.copy(); _ob.data = _uj.data.copy(); ctx.collection.objects.link(_ob)
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_ob.name, f"E{i}"] for i in _top_edges]), shape="BLEND", radius=0.1,
                               continuity="2", class_a=True, ca_tolerance=1e-4)
_rb = [f for f in json.loads(_ob.get("nproto_class_a", "{}")).get("faces", []) if "skipped" not in f]
_bend = [f for f in _rb if f["spans"] > 1 or f["degree"] > 1]
_strt = [f for f in _rb if f["spans"] == 1 and f["degree"] == 1]
check("class-a blend: straight runs are single exact patches, the bend is symmetric and G2 to them",
      r == {'FINISHED'} and len(_strt) == 2 and len(_bend) == 1 and _bend[0]["symmetric"] and _bend[0]["within"]
      and _bend[0]["exact_sections"] and _bend[0]["ends"] == ["G2", "G2"],
      str([(f["spans"], f["degree"], f.get("symmetric"), f["ends"]) for f in _rb]))
_ar = sorted((round(f.area(), 6) for f in store.get_brep(_ob).faces() if f.area() < 0.2), reverse=True)
_pairs = [abs(a - b) for a, b in zip(_ar[0::2], _ar[1::2])]
check("class-a blend: patch areas come in mirror pairs", _pairs and max(_pairs) < 1e-4, str(_ar))
check("display toggles live on the window manager (no undo step)", hasattr(bpy.context.window_manager, "nproto_view")
      and not hasattr(ctx.scene.nproto_settings, "show_cvs"))

# Class-A unit pieces: Bezier split is exact, G2 end constraint holds
from blender_nurbs.kernel import classa as _ca
_P = np.random.RandomState(3).rand(9, 4) + 0.5
_U = _ca.clamped_knots(5, 4, [0.2, 0.5, 0.7])
_pcs = _ca.bezier_pieces(_P, _U, 5)
_tt = np.linspace(0, 1, 50)
_ref = _ca.basis(_U, 5, _tt) @ _P
_br = [0.0, 0.2, 0.5, 0.7, 1.0]
_err = 0.0
for _k, _t in enumerate(_tt):
    _j = min(int(np.searchsorted(_br, _t, side="right")) - 1, 3)
    _u = (_t - _br[_j]) / (_br[_j + 1] - _br[_j])
    _err = max(_err, float(np.abs(_ca.bernstein(5, [_u])[0] @ _pcs[_j] - _ref[_k]).max()))
check("class-a: Bezier split of a B-spline is exact", len(_pcs) == 4 and _err < 1e-12, f"{_err:.1e}")

# --- Split / Trim Curves (line across a circle, as in the user's sketch)
from blender_nurbs import curvetrim as _ct, curvejoin as _cj
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
def _line(name, a, b):
    o = nurbs(name, [a, b], order=2); return o
def _on_curve(piece_obj, ref_spans):
    """max distance of points of piece_obj to the reference curve (dense samples)."""
    sp = _cj.transform_spans(_cj.spline_spans(piece_obj.data.splines[0]), geom.world_matrix(piece_obj))
    pts, _ = _ct.sample_spans(sp, 20)
    if ref_spans == "circle":
        return float(np.abs(np.linalg.norm(pts[:, :2], axis=1) - 1.0).max() + np.abs(pts[:, 2]).max())
    return float(np.abs(pts[:, 1] - 0.35).max() + np.abs(pts[:, 2]).max())
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0))
CIR = ctx.active_object; CIR.name = "TCircle"
LN = _line("TLine", (-2.0, 0.35, 0.0), (2.0, 0.35, 0.0))
cs = _ct.CurveSet([CIR, LN])
hits = _ct.intersect(cs.world[("TCircle", 0)], cs.world[("TLine", 0)])
xs = sorted(float(h[2][0]) for h in hits)
xe = math.sqrt(1 - 0.35 ** 2)
check("curves: line x circle -> 2 exact crossings", len(hits) == 2 and abs(xs[0] + xe) < 1e-9 and abs(xs[1] - xe) < 1e-9,
      f"{xs}")
circ_ref = cs.world[("TCircle", 0)]; line_ref = cs.world[("TLine", 0)]
bpy.ops.object.select_all(action='DESELECT'); CIR.select_set(True); LN.select_set(True)
ctx.view_layer.objects.active = CIR
r = bpy.ops.nproto.split_curves()
parts = [o for o in bpy.data.objects if o.type == 'CURVE']
lines = [o for o in parts if o.name.startswith("TLine")]; circs = [o for o in parts if o.name.startswith("TCircle")]
check("split curves: line -> 3, circle -> 2", r == {'FINISHED'} and len(lines) == 3 and len(circs) == 2,
      f"{[o.name for o in parts]}")
dev = max([_on_curve(o, "circle") for o in circs] + [_on_curve(o, "line") for o in lines])
check("split curves: pieces lie on the originals", dev < 1e-6, f"{dev:.1e}")
_len = lambda o: float(np.linalg.norm(np.diff(_ct.sample_spans(_cj.spline_spans(o.data.splines[0]), 200)[0], axis=0), axis=1).sum())
tot = sum(_len(o) for o in circs)
check("split curves: circle pieces add up to 2*pi", abs(tot - 2 * math.pi) < 1e-4, f"{tot:.6f}")
arcs = sorted(_len(o) for o in circs)
check("split curves: arc lengths match the crossing angles", abs(arcs[0] - (math.pi - 2 * math.asin(0.35))) < 1e-4, f"{arcs}")
# Trim logic: cut the line by the circle and drop the outside piece on the right
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)); CIR = ctx.active_object; CIR.name = "TCircle"
LN = _line("TLine", (-2.0, 0.35, 0.0), (2.0, 0.35, 0.0))
cs = _ct.CurveSet([CIR, LN]); key = ("TLine", 0)
cuts = cs.cuts(key, [("TCircle", 0)])
pcs = _ct.split_spans(cs.local[key], cuts, cs.closed[key])
k = _ct.piece_at(pcs, 0.95, 1)
check("trim curves: click near the right end picks the outer right piece", len(pcs) == 3 and k == 2)
made = _ct.put_back(LN, 0, _ct.remove_piece(pcs, k, False))
xs = sorted(float(c) for o in [LN] + made for p in o.data.splines[0].points for c in [p.co[0]])
check("trim curves: line now ends on the circle", len(made) == 0 and abs(max(xs) - xe) < 1e-6, f"{max(xs):.6f}")
# closed curve: the piece across the seam stays one piece
key = ("TCircle", 0)
cuts = cs.cuts(key, [("TLine", 0)])
pcs = _ct.split_spans(cs.local[key], cuts, True)
check("trim curves: circle cut twice -> 2 pieces", len(pcs) == 2)
check("trim curves: deleting the middle of the line leaves 2 parts, of a circle 1 arc",
      len(_ct.remove_piece(_ct.split_spans(cs.local[("TLine", 0)], [0.25, 0.75], False), 1, False)) == 2
      and len(_ct.remove_piece(pcs, 0, True)) == 1)
# apparent intersection: line above the circle, seen from the top
LN2 = _line("TLineHigh", (-2.0, -0.2, 1.0), (2.0, -0.2, 1.0))
cs3 = _ct.CurveSet([CIR, LN2]); cs2 = _ct.CurveSet([CIR, LN2], (0, 0, -1))
check("split curves: apparent crossings only with a view direction",
      not cs3.cuts(("TLineHigh", 0), [("TCircle", 0)]) and len(cs2.cuts(("TLineHigh", 0), [("TCircle", 0)])) == 2)
# Bezier curves crossing twice
B1 = bezier("TB1", [(0, 0), (1, 1), (2, 0), (3, 1)]); B2 = bezier("TB2", [(0, 0.5), (1.5, 0.5), (3, 0.5)])
cs4 = _ct.CurveSet([B1, B2]); h4 = _ct.intersect(cs4.world[("TB1", 0)], cs4.world[("TB2", 0)])
err = max(abs(float(h[2][1]) - 0.5) for h in h4) if h4 else 1
check("curves: Bezier x Bezier crossings on y=0.5", len(h4) == 3 and err < 1e-9, f"{len(h4)} {err:.1e}")

# --- Rebuild / Extend Curve
from blender_nurbs import curvetools as _cv
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)); CIR = ctx.active_object; CIR.name = "RCircle"
LN = _line("RLine", (-2.0, 0.35, 0.0), (2.0, 0.35, 0.0))
bpy.ops.object.select_all(action='DESELECT'); CIR.select_set(True); LN.select_set(True); ctx.view_layer.objects.active = CIR
bpy.ops.nproto.split_curves(cutter="ALL")
arcs = [o for o in bpy.data.objects if o.name.startswith("RCircle")]
low = max(arcs, key=lambda o: _len(o))
def _wsp(o): return _cj.transform_spans(_cj.spline_spans(o.data.splines[0]), geom.world_matrix(o))
ends0 = _cj._ends(_wsp(low)); tan0 = _cv.end_tangents(_wsp(low))
bpy.ops.object.select_all(action='DESELECT'); low.select_set(True); ctx.view_layer.objects.active = low
r = bpy.ops.nproto.rebuild_curve(rc_degree="3", rc_cvs=0, tolerance=1e-4)
sp = low.data.splines[0]
pts = _ct.sample_spans(_wsp(low), 40)[0]
dev = float(np.abs(np.linalg.norm(pts[:, :2], axis=1) - 1).max())
ends1 = _cj._ends(_wsp(low)); tan1 = _cv.end_tangents(_wsp(low))
check("rebuild curve: split arc -> one uniform non-rational cubic within tolerance",
      r == {'FINISHED'} and sp.order_u == 4 and not sp.use_bezier_u and all(abs(p.co[3] - 1) < 1e-9 for p in sp.points)
      and dev <= 1.2e-4, f"{len(sp.points)} CVs dev {dev:.1e}")
check("rebuild curve: ends and end tangents kept",
      max(np.linalg.norm(ends0[0] - ends1[0]), np.linalg.norm(ends0[1] - ends1[1])) < 1e-6
      and min(float(tan0[0] @ tan1[0]), float(tan0[1] @ tan1[1])) > 1 - 1e-9)
r = bpy.ops.nproto.rebuild_curve(rc_degree="5", rc_cvs=12)
check("rebuild curve: fixed degree / CV count", len(low.data.splines[0].points) == 12 and low.data.splines[0].order_u == 6)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(4, 0, 0)); FC = ctx.active_object
r = bpy.ops.nproto.rebuild_curve(rc_degree="3", rc_cvs=0, tolerance=1e-4)
pts = _ct.sample_spans(_wsp(FC), 40)[0] - np.array([4, 0, 0])
dev = float(np.abs(np.linalg.norm(pts[:, :2], axis=1) - 1).max())
check("rebuild curve: closed circle -> periodic cubic", FC.data.splines[0].use_cyclic_u and dev <= 1.2e-4,
      f"{len(FC.data.splines[0].points)} CVs dev {dev:.1e}")
# Extend: a short arc naturally / as an arc stays on its circle; a line grows by the distance
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)); CIR = ctx.active_object; CIR.name = "ECircle"
LN = _line("ELine", (-2.0, 0.35, 0.0), (2.0, 0.35, 0.0))
bpy.ops.object.select_all(action='DESELECT'); CIR.select_set(True); LN.select_set(True); ctx.view_layer.objects.active = LN
bpy.ops.nproto.split_curves(cutter="ACTIVE")
arc = min([o for o in bpy.data.objects if o.name.startswith("ECircle")], key=_len)
L0 = _len(arc)
n_arc0 = len(arc.data.splines[0].points)
for kind in ("NATURAL", "ARC"):
    bpy.ops.object.select_all(action='DESELECT'); arc.select_set(True); ctx.view_layer.objects.active = arc
    r = bpy.ops.nproto.extend_curve(ec_type=kind, ec_to="DISTANCE", ec_end="END", distance=0.3)
    pts = _ct.sample_spans(_wsp(arc), 30)[0]
    dev = float(np.abs(np.linalg.norm(pts[:, :2], axis=1) - 1).max())
    L1 = _len(arc)
    check(f"extend curve ({kind.lower()}): arc stays on its circle, +0.3 long", r == {'FINISHED'} and dev < 1e-6
          and abs(L1 - L0 - 0.3) < 1e-3, f"dev {dev:.1e} dL {L1 - L0:.4f}")
    L0 = L1
S2 = _line("EShort", (0.0, -0.2, 0.0), (0.3, -0.2, 0.0))
bpy.ops.object.select_all(action='DESELECT'); S2.select_set(True); ctx.view_layer.objects.active = S2
r = bpy.ops.nproto.extend_curve(ec_type="LINE", ec_to="DISTANCE", ec_end="BOTH", distance=0.25)
xs = [p.co[0] for p in S2.data.splines[0].points]
check("extend curve (line, both ends): end CVs moved, still 2 CVs", r == {'FINISHED'} and len(xs) == 2
      and abs(min(xs) + 0.25) < 1e-6 and abs(max(xs) - 0.55) < 1e-6, f"{xs}")
S3 = _line("EToCircle", (0.0, -0.5, 0.0), (0.4, -0.5, 0.0))
bpy.ops.object.select_all(action='DESELECT'); S3.select_set(True); CIR2 = [o for o in bpy.data.objects if o.name.startswith("ECircle") and o != arc][0]
CIR2.select_set(True); ctx.view_layer.objects.active = CIR2
r = bpy.ops.nproto.extend_curve(ec_type="LINE", ec_to="ACTIVE", ec_end="NEAREST")
xs = [p.co[0] for p in S3.data.splines[0].points]
check("extend curve to the active curve (nearest end)", r == {'FINISHED'} and abs(max(xs) - math.sqrt(0.75)) < 1e-5
      and abs(min(xs)) < 1e-9, f"{xs}")

check("extend curve: an exact arc keeps its CV count", len(arc.data.splines[0].points) == n_arc0,
      f"{n_arc0} -> {len(arc.data.splines[0].points)}")
BZ = bezier("EBez", [(0, 2), (1, 2.5), (2, 2)])
bpy.ops.object.select_all(action='DESELECT'); BZ.select_set(True); ctx.view_layer.objects.active = BZ
r = bpy.ops.nproto.extend_curve(ec_type="NATURAL", ec_to="DISTANCE", ec_end="END", distance=0.4)
check("extend curve: a Bezier spline stays Bezier with its point count",
      r == {'FINISHED'} and BZ.data.splines[0].type == 'BEZIER' and len(BZ.data.splines[0].bezier_points) == 3)
UC = nurbs("EUni", [(0, 3, 0), (1, 3.5, 0), (2, 3, 0), (3, 3.5, 0), (4, 3, 0), (5, 3.5, 0)])
bpy.ops.object.select_all(action='DESELECT'); UC.select_set(True); ctx.view_layer.objects.active = UC
r = bpy.ops.nproto.extend_curve(ec_type="NATURAL", ec_to="DISTANCE", ec_end="END", distance=0.3, keep_cvs=True)
check("extend curve: Keep CV Count refits a uniform spline", r == {'FINISHED'} and len(UC.data.splines[0].points) == 6)
# To Next Curve: a line inside the circle reaches it
S4 = _line("ENext", (0.0, 0.1, 0.0), (0.2, 0.1, 0.0))
bpy.ops.object.select_all(action='DESELECT'); S4.select_set(True); ctx.view_layer.objects.active = S4
bpy.context.scene.cursor.location = (5, 0.1, 0)
r = bpy.ops.nproto.extend_curve(ec_type="LINE", ec_to="ANY", ec_end="END")
xs = [p.co[0] for p in S4.data.splines[0].points]
check("extend curve to the next curve", r == {'FINISHED'} and len(xs) == 2 and abs(max(xs) - math.sqrt(1 - 0.01)) < 1e-5, f"{xs}")
bpy.context.scene.cursor.location = (0, 0, 0)

# --- Edit Mode: Rebuild / Extend / Split edit the splines in place
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)); CIR = ctx.active_object; CIR.name = "MCircle"
TWO = nurbs("MTwo", [(-2.0, 0.35, 0.0), (2.0, 0.35, 0.0)], order=2)
s2 = TWO.data.splines.new('NURBS'); s2.points.add(1)
s2.points[0].co = (-2.0, -0.5, 0.0, 1.0); s2.points[1].co = (-1.5, -0.5, 0.0, 1.0); s2.order_u = 2; s2.use_endpoint_u = True
bpy.ops.object.select_all(action='DESELECT'); TWO.select_set(True); ctx.view_layer.objects.active = TWO
bpy.ops.object.mode_set(mode='EDIT')
for sp in TWO.data.splines:
    for p in sp.points:
        p.select = False
TWO.data.splines[1].points[1].select = True        # the short line's end point
r = bpy.ops.nproto.extend_curve(ec_type="LINE", ec_to="ANY", ec_end="NEAREST")
check("edit mode: extend the selected end up to the circle, still in Edit Mode",
      r == {'FINISHED'} and ctx.mode == 'EDIT_CURVE')
for sp in TWO.data.splines:
    for p in sp.points:
        p.select = False
TWO.data.splines[0].points[0].select = True        # the long line
r = bpy.ops.nproto.split_curves()
nspl = len(TWO.data.splines)
bpy.ops.object.mode_set(mode='OBJECT')
short = [sp for sp in TWO.data.splines if abs(sp.points[0].co[1] + 0.5) < 1e-6]
xe2 = -math.sqrt(1 - 0.25)
check("edit mode: extended line kept after leaving Edit Mode (2 CVs, ends on the circle)",
      len(short) == 1 and len(short[0].points) == 2 and abs(short[0].points[1].co[0] - xe2) < 1e-5,
      f"{[tuple(p.co) for p in short[0].points] if short else None}")
check("edit mode: split keeps the pieces as splines of the same object", r == {'FINISHED'} and nspl == 4
      and len([o for o in bpy.data.objects if o.name.startswith("MTwo")]) == 1, f"{nspl} splines")
bpy.ops.object.mode_set(mode='EDIT')
for sp in TWO.data.splines:
    for p in sp.points:
        p.select = False
CIRsel = None
bpy.ops.object.mode_set(mode='OBJECT')
CIR.select_set(True); TWO.select_set(False); ctx.view_layer.objects.active = CIR
bpy.ops.object.mode_set(mode='EDIT')
r = bpy.ops.nproto.rebuild_curve(rc_degree="3", rc_cvs=10)
ok_edit = ctx.mode == 'EDIT_CURVE'
bpy.ops.object.mode_set(mode='OBJECT')
check("edit mode: rebuild in place (circle -> 10 periodic CVs)", r == {'FINISHED'} and ok_edit
      and len(CIR.data.splines[0].points) == 10 and CIR.data.splines[0].use_cyclic_u)

# --- Sketch tools: CV Curve (CVs / Through Points) and Line
from blender_nurbs import drawtools as _dt
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
if ctx.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
clicks = [(0, 0, 0), (1, 0.8, 0), (2.5, 0.2, 0), (3, 1.5, 0), (4.2, 1.0, 0)]
def _curve_pts(o, per=4000):
    sp = _cj.transform_spans(_cj.spline_spans(o.data.splines[0]), geom.world_matrix(o))
    return _ct.sample_spans(sp, per)[0]
def _passes(o, pts):
    from blender_nurbs import curvetools as _cvt
    C = _curve_pts(o, 400)
    return float(_cvt._seg_dist(np.array(pts, float), C).max())
data = json.dumps({"pts": clicks, "start_out": None, "end_out": None, "edit": ""})
r = bpy.ops.nproto.draw_curve(data=data, mode="THROUGH", degree="3", closed=False)
o = ctx.active_object; sp = o.data.splines[0]
d = _passes(o, clicks)
check("cv curve (through points): passes through every click, uniform cubic",
      r == {'FINISHED'} and sp.order_u == 4 and sp.use_endpoint_u and not sp.use_bezier_u and len(sp.points) == 5
      and d < 5e-5, f"{d:.1e}")
r = bpy.ops.nproto.draw_curve(data=data, mode="CV", degree="3", closed=False)
o = ctx.active_object; M = np.array(geom.world_matrix(o))
cvs = [M[:3, :3] @ np.array(p.co[:3]) + M[:3, 3] for p in o.data.splines[0].points]
check("cv curve (CVs): the clicks are the CVs", r == {'FINISHED'} and
      max(np.linalg.norm(np.array(a) - b) for a, b in zip(clicks, cvs)) < 1e-6)
r = bpy.ops.nproto.draw_curve(data=data, mode="THROUGH", degree="3", closed=True)
o = ctx.active_object
d = _passes(o, clicks)
check("cv curve (through points, closed): periodic through every click",
      r == {'FINISHED'} and o.data.splines[0].use_cyclic_u and d < 5e-5, f"{d:.1e}")
d2 = json.dumps({"pts": [(0, 0, 0), (1, 1, 0), (1.5, 0.5, 0), (2, 0, 0)], "start_out": (1, 0, 0), "end_out": (0, -1, 0), "edit": ""})
r = bpy.ops.nproto.draw_curve(data=d2, mode="CV", degree="2", closed=False, start_cont="G1", end_cont="G1")
o = ctx.active_object; M = np.array(geom.world_matrix(o))
cv = [M[:3, :3] @ np.array(p.co[:3]) + M[:3, 3] for p in o.data.splines[0].points]
check("cv curve: tangent match puts the 2nd CV on the snapped end's tangent", r == {'FINISHED'}
      and abs(cv[1][1]) < 1e-6 and cv[1][0] > 0 and abs(cv[2][0] - 2) < 1e-6 and cv[2][1] < 0, f"{cv}")
X, p_ = _dt.interpolate(np.random.RandomState(1).rand(7, 3), 5)
check("cv curve: interpolation degree 5 keeps one CV per point", len(X) == 7 and p_ == 5)
lines = [[(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)], [(3, 0, 0), (4, 0, 0)]]
n0 = len(bpy.data.objects)
r = bpy.ops.nproto.draw_line(data=json.dumps({"lines": lines, "edit": ""}), polyline=False)
check("line: one 2-CV curve per line", r == {'FINISHED'} and len(bpy.data.objects) - n0 == 3
      and all(len(o.data.splines[0].points) == 2 for o in ctx.selected_objects))
n0 = len(bpy.data.objects)
r = bpy.ops.nproto.draw_line(data=json.dumps({"lines": lines, "edit": ""}), polyline=True)
sizes = sorted(len(o.data.splines[0].points) for o in ctx.selected_objects)
check("line (polyline): connected lines become one degree-1 curve", r == {'FINISHED'} and sizes == [2, 3], f"{sizes}")
host = ctx.selected_objects[0]
bpy.ops.object.select_all(action='DESELECT'); host.select_set(True); ctx.view_layer.objects.active = host
bpy.ops.object.mode_set(mode='EDIT')
ns = len(host.data.splines)
r = bpy.ops.nproto.draw_curve(data=json.dumps({"pts": clicks, "start_out": None, "end_out": None, "edit": host.name}),
                              mode="CV", degree="3", closed=False)
ok_mode = ctx.mode == 'EDIT_CURVE'
bpy.ops.object.mode_set(mode='OBJECT')
check("cv curve in Edit Mode: added as a spline of the edited object", r == {'FINISHED'} and ok_mode
      and len(host.data.splines) == ns + 1)

# line tool: typed length / angle
P = _dt.polar_point((1, 1, 0), (1, 0, 0), (0, 1, 0), 2.0, 90.0)
check("line: typed 2 at 90 deg goes straight up", np.allclose(P, (1, 3, 0)))
P = _dt.polar_point((0, 0, 0), (1, 0, 0), (0, 0, 1), 1.0, 30.0)
check("line: typed angle in a front-view plane", np.allclose(P, (math.cos(math.radians(30)), 0, 0.5)))
us = ctx.scene.unit_settings
us.system, us.length_unit = 'IMPERIAL', 'INCHES'
v1, v2, v3 = _dt.parse_length(ctx, "3"), _dt.parse_length(ctx, "2ft 6in"), _dt.parse_length(ctx, "10mm")
check("line: typed lengths use the display unit (inches) and honour unit letters",
      abs(v1 - 0.0762) < 1e-6 and abs(v2 - 0.762) < 1e-6 and abs(v3 - 0.01) < 1e-6, f"{v1} {v2} {v3}")
us.system = 'METRIC'; us.length_unit = 'METERS'

# --- Sketch snapping (CV Curve / Line): curve ends, middles, points on curves,
#     surface edges, face centres, shown surface CVs; nearest the viewer wins
from blender_nurbs import snapping as _sn, draw as _drw
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.cursor.location = (0, 0, 0)
bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.5, height=1.2, caps=True)
SCY = ctx.active_object
SLN = nurbs("SnapLine", [(1.0, -1.0, 0.0), (3.0, 1.0, 0.0)], order=2)
def _snap_view(front=False, s=2.0, W=1000):
    """Orthographic top (or front) view, camera 10 away; 1 unit = W/(2s) px."""
    VM = np.eye(4)
    if front:                                  # looking along +Y: screen x = X, y = Z
        VM[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    VM[2, 3] = -10.0
    PM = np.diag([1 / s, 1 / s, -0.01, 1.0]) @ VM
    return _sn.View(PM, VM, W, W)
_px = lambda x, y, s=2.0, W=1000: np.array([W / 2 + W / 2 * x / s, W / 2 + W / 2 * y / s])
top, front = _snap_view(), _snap_view(True)
T = _sn.collect([SCY, SLN])
h = T.pick(top, _px(1, -1) + (3, 2))
check("snap: curve end point (exact)", h is not None and h[1] == "end" and np.allclose(h[0], (1, -1, 0), atol=1e-12)
      and h[2] is not None, f"{h}")
h = T.pick(top, _px(2, 0) + (2, -3))
check("snap: middle of a line", h is not None and h[1] == "mid" and np.allclose(h[0], (2, 0, 0), atol=1e-9), f"{h}")
q = np.array([1.6, -0.4, 0.0])
h = T.pick(top, _px(*q[:2]) + 5 * np.array([-1, 1]) / math.sqrt(2))
dl = abs(np.cross(h[0] - (1, -1, 0), np.array([1, 1, 0]) / math.sqrt(2))[2]) if h else 1
check("snap: a point on a curve lies on it", h is not None and h[1] == "curve" and dl < 1e-12
      and np.linalg.norm(h[0] - q) < 0.02, f"{h} off {dl:.1e}")
a = math.radians(30)
c30 = 0.5 * np.array([math.cos(a), math.sin(a)])
h = T.pick(top, _px(*c30) + 4 * c30 / 0.5)
_ztop = max(f_.centroid()[2] for f_ in store.get_brep(SCY).faces())      # the cap plane (stored in float32)
_de = min(e_.closest_point(tuple(h[0]))[2] for e_ in store.get_brep(SCY).edges()) if h else 1
check("snap: a point on a surface edge is exactly on it, the top rim (nearest the viewer)",
      h is not None and h[1] == "edge" and _de < 1e-9 and abs(h[0][2] - _ztop) < 1e-7
      and np.linalg.norm(h[0][:2] - c30) < 0.02, f"{h} off {_de:.1e}")
h2 = T.pick(top, _px(*c30) + 4 * c30 / 0.5, edges=False)
check("snap: edges off (Ctrl / Shift / lock) leaves only points", h2 is None, f"{h2}")
h = T.pick(top, _px(0, 0) + (3, 3))
check("snap: face centre of a disk is its exact centre, the top cap over the bottom one",
      h is not None and h[1] == "center" and np.linalg.norm(h[0][:2]) < 1e-8 and abs(h[0][2] - _ztop) < 1e-9, f"{h}")
_cs = [_sn.face_center(SCY, i_) for i_ in range(3)]
check("snap: a cylinder side wraps its centroid: its centre is the axis point",
      sum(1 for c_ in _cs if c_ is not None and np.linalg.norm(c_) < 1e-5) == 1, f"{_cs}")
h = T.pick(front, _px(0, 0) + (2, -2))
check("snap: in front of the axis point, the seam edge's middle is nearer the viewer and wins",
      h is not None and h[1] == "mid" and np.linalg.norm(h[0] - (0, -0.5, 0)) < 1e-7, f"{h}")
h = T.pick(top, _px(0, -0.5) + (2, 2))
check("snap: surface edge end point (the seam vertex)", h is not None and h[1] == "end"
      and abs(np.linalg.norm(h[0][:2]) - 0.5) < 1e-9, f"{h}")
arcA = nurbs("ArcA", [(0, 0, 0), (1, 0, 0.4), (2, 0, 0)], order=3)
arcB = nurbs("ArcB", [(0, 2, 0), (1, 2, 0.4), (2, 2, 0)], order=3)
bpy.ops.nproto.loft(inputs=ins((arcA, "S0"), (arcB, "S0")))
SLF = ctx.active_object
T = _sn.collect([SLF])
h = T.pick(top, _px(1, 1) + (3, -2))
fc = store.get_brep(SLF).faces()[0]
dfc = fc.closest_point(tuple(h[0]))[1] if h else 1
check("snap: a curved face's centre lies on the face, in its middle", h is not None and h[1] == "center"
      and dfc < 1e-9 and np.linalg.norm(h[0][:2] - (1, 1)) < 1e-3 and 0.15 < h[0][2] < 0.25, f"{h} off {dfc:.1e}")
ncv = T.kinds.count("cv")
Ts = _sn.collect([SLF], lambda o: True)
nnet = sum(len(n_["p"]) for n_ in store.display(SLF)["nets"])
h = Ts.pick(front, _px(1, 0.4) + (2, 1))
check("snap: surface CVs snap only when the CV net is shown", ncv == 0 and Ts.kinds.count("cv") == nnet > 0
      and h is not None and h[1] == "cv" and np.allclose(h[0], (1, 0, 0.4), atol=1e-6), f"{ncv} {nnet} {h}")
v_ = ctx.window_manager.nproto_view
old_cv = v_.show_cvs
bpy.ops.object.select_all(action='DESELECT'); SLF.select_set(True)
v_.show_cvs = True
on = _drw.cv_net_shown(ctx, SLF)
SLF.select_set(False)
off = _drw.cv_net_shown(ctx, SLF)
v_.show_cvs = old_cv
check("snap: a surface's CV net counts as shown when CVs / Hulls is on and it is selected", on and not off)
import time as _time
T = _sn.collect([SCY, SLN, SLF], lambda o: True)
t0 = _time.time()
for k in range(50):
    T.pick(top, _px(0.01 * k, 0.3))
dt = (_time.time() - t0) / 50
check("snap: picking stays quick", dt < 0.02, f"{dt * 1e3:.2f} ms per mouse move")

# --- Line with Polyline on: every click adds a segment; clicking the first point closes it
_a, _b, _c = np.array((0.0, 0, 0)), np.array((1.0, 0, 0)), np.array((1.0, 1, 0))
_ls = [(_a, _b)]
_s1 = _dt.next_start(_ls, _b, True)
_ls.append((_b, _c))
_s2 = _dt.next_start(_ls, _c, True)
_ls.append((_c, _a))
_s3 = _dt.next_start(_ls, _a, True)
_s4 = _dt.next_start([(_a, _b)], _b, False)
check("polyline: each click starts the next segment, the first point closes it, off: no chaining",
      np.allclose(_s1, _b) and np.allclose(_s2, _c) and _s3 is None and _s4 is None, f"{_s1} {_s2} {_s3} {_s4}")
r = bpy.ops.nproto.draw_line(data=json.dumps({"lines": [[list(a_), list(b_)] for a_, b_ in _ls], "edit": ""}),
                             polyline=True)
_o = ctx.active_object
check("polyline: a closed chain of clicks becomes one closed degree-1 curve", r == {'FINISHED'}
      and len(ctx.selected_objects) == 1 and _o.data.splines[0].use_cyclic_u and len(_o.data.splines[0].points) == 3)

# --- Sketch axis locks (X / Y / Z, Shift: the plane of the other two)
P = _dt.axis_lock_point("X", (1, 2, 3), ((5, -3, 10), (0, 0, -1)))
check("axis lock X: slides along X through the previous point", np.allclose(P, (5, 2, 3)), f"{P}")
P = _dt.axis_lock_point("Z", (1, 2, 3), ((0, -10, 7), (0, 1, 0)))
check("axis lock Z: height from the mouse ray in a front view", np.allclose(P, (1, 2, 7)), f"{P}")
P = _dt.axis_lock_point("Z", (1, 2, 3), ((4, 4, 10), (0, 0, -1)))
check("axis lock Z seen end-on stays at the previous point", np.allclose(P, (1, 2, 3)), f"{P}")
P = _dt.axis_lock_point("-Z", (1, 2, 3), ((0.5, -10, 7), (0.1, 1, -0.2)))
check("plane lock Shift+Z: stays at the previous point's height", abs(P[2] - 3) < 1e-12, f"{P}")
P = _dt.axis_lock_point("-X", (1, 2, 3), ((4, 4, 10), (0, 0, -1)), plane_point=(4.0, 4.0, 0.0))
check("plane lock Shift+X seen edge-on: the drawing-plane point projected into it",
      np.allclose(P, (1, 4, 0)), f"{P}")
P = _dt.axis_lock_point("X", (1, 2, 3), ((0, 0, 10), (0, 0, -1)), snapped=(3.3, 7, -1))
check("axis lock X: a snapped point is projected onto the lock line", np.allclose(P, (3.3, 2, 3)), f"{P}")
P = _dt.axis_lock_point("-Y", (0, 5, 0), ((0.26, 0, 0.74), (0, 1, 0)), step=0.25)
check("plane lock with Ctrl: grid steps from the previous point", np.allclose(P, (0.25, 5, 0.75)), f"{P}")

# --- CV Curve start / end conditions (G0 / G1 / G2 to the snapped curve or
#     surface edge) and the Align history step they add
from blender_nurbs import align as _al
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.nproto_settings.construction_history = True
TGT = nurbs("CondTarget", [(-3, 0, 0), (-2, 0.5, 0), (-1, 1.5, 0.3), (0, 1, 0)])
def _draw(pts, start=None, end=None, **kw):
    r_ = bpy.ops.nproto.draw_curve(data=json.dumps({"pts": pts, "start": start, "end": end, "edit": ""}), **kw)
    return r_, ctx.active_object
def _cvs(o):
    M_ = np.array(geom.world_matrix(o))
    return np.array([M_[:3, :3] @ np.array(p_.co[:3]) + M_[:3, 3] for p_ in o.data.splines[0].points])
clk = [(0, 1, 0), (1, 0.4, 0.2), (2, 1.2, 0.0), (3, 0.5, -0.2)]
r, CG2 = _draw(clk, [TGT.name, "S0", "END"], None, mode="CV", degree="3", start_cont="G2", end_cont="G1")
gap, ang, dk = _al.measure(CG2, 0, TGT, "S0", "END")
h_ = features.history(CG2)
check("cv curve G2 start on a curve end: continues it (gap, angle, curvature)", r == {'FINISHED'}
      and gap < 1e-6 and ang < 1e-4 and dk < 1e-6, f"{gap:.1e} {ang:.1e}° {dk:.1e}")
check("cv curve G2: an Align history step is added at once", h_ is not None and h_["op"] == "align_curve"
      and h_["params"]["ends"]["0"]["cont"] == "G2" and "1" not in h_["params"]["ends"], f"{h_}")
X0 = _cvs(CG2)
check("cv curve G2: only the end CVs moved, the others are the clicks",
      np.allclose(X0[3], clk[3], atol=1e-6) and np.linalg.norm(X0[1] - clk[1]) > 1e-3, f"{X0}")
TGT.data.splines[0].points[2].co = (-1, 2.2, 0.6, 1.0)
features.rebuild(CG2, ctx)
gap, ang, dk = _al.measure(CG2, 0, TGT, "S0", "END")
X1 = _cvs(CG2)
check("cv curve G2: moving the target keeps the start continuous (history)", gap < 1e-6 and ang < 1e-4 and dk < 1e-3
      and np.allclose(X1[3], X0[3], atol=1e-6) and np.linalg.norm(X1[2] - X0[2]) > 1e-3,
      f"{gap:.1e} {ang:.1e}° {dk:.1e}")
r, C0 = _draw(clk, [TGT.name, "S0", "END"], None, mode="CV", degree="3", start_cont="G0", end_cont="G0")
check("cv curve G0 on a curve end: no history, the clicks are the CVs", r == {'FINISHED'}
      and features.history(C0) is None and np.allclose(_cvs(C0), clk, atol=1e-6))
r, C2 = _draw([(0, 1, 0), (1.5, 2, 0.5)], [TGT.name, "S0", "END"], None, mode="CV", degree="3", start_cont="G2")
gap, ang, dk = _al.measure(C2, 0, TGT, "S0", "END")
check("cv curve G2 with only 2 clicks: more CVs (and degree 3) so it can hold", r == {'FINISHED'}
      and len(C2.data.splines[0].points) >= 3 and C2.data.splines[0].order_u == 4 and gap < 1e-6 and ang < 1e-4,
      f"{len(C2.data.splines[0].points)} CVs {gap:.1e} {ang:.1e}°")
# along a curve (NEAREST): the new curve leaves tangent to it, the way it was drawn
kc_ = features.input_curve(TGT, "S0")
t0_, t1_ = kc_.parameter_range()
Pm = np.array(kc_.evaluate(0.5 * (t0_ + t1_)), float)
Tm = np.array(_al._k()[0].evaluate_curve(kc_, 0.5 * (t0_ + t1_))["tangent"], float)
side = Pm + np.cross(Tm, (0, 0, 1)) / np.linalg.norm(Tm)
r, CN = _draw([tuple(Pm), tuple(side - 0.3 * Tm / np.linalg.norm(Tm)), tuple(side - Tm)],
              [TGT.name, "S0", "NEAREST"], None, mode="CV", degree="3", start_cont="G1")
X = _cvs(CN)
d1 = (X[1] - X[0]) / np.linalg.norm(X[1] - X[0])
tm = Tm / np.linalg.norm(Tm)
check("cv curve G1 along a curve: leaves along its tangent, in the drawn direction (flipped)",
      r == {'FINISHED'} and np.linalg.norm(np.cross(d1, tm)) < 1e-6 and d1 @ tm < 0
      and features.history(CN)["params"]["ends"]["0"]["flip"], f"{d1} {tm}")
# a curved surface edge, Through Points: tangent to the surface, its normal
# curvature matched (G2), still through every click
arcA = nurbs("ArcA", [(0, 0, 0), (1, 0, 0.4), (2, 0, 0)], order=3)
arcB = nurbs("ArcB", [(0, 2, 0), (1, 2, 0.4), (2, 2, 0)], order=3)
bpy.ops.nproto.loft(inputs=ins((arcA, "S0"), (arcB, "S0")))
ARC = ctx.active_object
dE = store.display(ARC)["edges"]
ei = min(range(len(dE)), key=lambda i_: abs(np.mean(np.array(dE[i_])[:, 1])) + 10 * (np.ptp(np.array(dE[i_])[:, 2]) < 1e-6))
clk = [(0.5, 0.0, 0.15), (1.1, -0.6, 0.35), (1.6, -1.4, 0.3), (2.0, -2.0, 0.0)]
r, CS = _draw(clk, [ARC.name, f"E{ei}", "NEAREST"], None, mode="THROUGH", degree="3", start_cont="G2",
              edge_mode="SURFACE")
kcs = features.input_curve(CS, "S0")
ds = _al._k()[0].evaluate_curve(kcs, kcs.parameter_range()[0])
c1, c2 = np.array(ds["tangent"], float), np.array(ds["curvature"], float)
nrm, k_n = _al.surface_frame(ARC, f"E{ei}", np.array(ds["point"], float))
tilt = math.degrees(math.asin(min(1.0, abs(c1 @ nrm) / np.linalg.norm(c1))))
kc_curve = (c2 @ nrm) / (c1 @ c1)
kc_surf = k_n(c1 / np.linalg.norm(c1))
check("cv curve through points, G2 on a surface edge: tangent to the surface, normal curvature matched",
      r == {'FINISHED'} and tilt < 1e-5 and abs(kc_surf) > 0.02 and abs(kc_curve - kc_surf) < 1e-5 * abs(kc_surf)
      and np.linalg.norm(np.array(ds["point"]) - clk[0]) < 1e-6, f"tilt {tilt:.1e}° k {kc_curve:.5f} vs {kc_surf:.5f}")
d = _passes(CS, clk)
check("cv curve through points with G2: still passes through every click", d < 5e-5, f"{d:.1e}")
Xb = _cvs(CS)
kind_, deg_, Qs, _ = _al.solve(CS, features.history(CS)["params"]["ends"],
                               [(CS, "S0"), (ARC, f"E{ei}")])
M_ = np.array(geom.world_matrix(CS))
dq = np.abs(Qs @ M_[:3, :3].T + M_[:3, 3] - Xb).max()
check("cv curve: the recorded Align step rebuilds the drawn curve unchanged", dq < 1e-6, f"{dq:.1e}")
ctx.scene.nproto_settings.construction_history = False
r, CH = _draw(clk, [ARC.name, f"E{ei}", "NEAREST"], None, mode="THROUGH", degree="3", start_cont="G1")
ctx.scene.nproto_settings.construction_history = True
check("cv curve with history off: aligned once, no history", r == {'FINISHED'} and features.history(CH) is None)

# --- The sketch tools driven through their modal code (fake top view, mouse
#     events): snap -> reference (edge / face, T cycles) -> G1 end / tangent line
import types as _types
from mathutils import Matrix as _Mx, Quaternion as _Qt
class _FArea:
    def __init__(self, region, rv3d):
        self.regions = [region]
        self.spaces = _types.SimpleNamespace(active=_types.SimpleNamespace(region_3d=rv3d))
        self.type = 'VIEW_3D'
        self.text = None
    def header_text_set(self, t):
        self.text = t
    def tag_redraw(self):
        pass
class _FCtx:
    """bpy.context with a top orthographic 3D view of 1000 px = 4 units."""
    def __init__(self):
        V = _Mx.Translation((0, 0, -10))
        Wm = _Mx(((0.5, 0, 0, 0), (0, 0.5, 0, 0), (0, 0, -0.01, 0), (0, 0, 0, 1)))
        self.region = _types.SimpleNamespace(x=0, y=0, width=1000, height=1000, type='WINDOW')
        self.region_data = _types.SimpleNamespace(view_matrix=V, perspective_matrix=Wm @ V, window_matrix=Wm,
                                                  view_rotation=_Qt(), is_perspective=False, view_distance=10.0,
                                                  view_perspective='ORTHO')
        self.area = _FArea(self.region, self.region_data)
    def __getattr__(self, k):
        return getattr(bpy.context, k)
def _ev(type_, xy=(0, 0), value='PRESS', shift=False, ctrl=False):
    return _types.SimpleNamespace(type=type_, value=value, mouse_x=float(xy[0]), mouse_y=float(xy[1]), shift=shift,
                                  ctrl=ctrl, alt=False, oskey=False, unicode="")
_wpx = lambda x, y: (500 + 250 * x, 500 + 250 * y)
def _fake(op_cls, **props):
    cls = type("Fake" + op_cls.__name__, (_dt.SketchMixin,), {})
    for k_, v_ in op_cls.__dict__.items():
        if callable(v_) or isinstance(v_, staticmethod):
            if k_ not in ("passthrough", "invoke", "__init__") and not k_.startswith("bl_"):
                setattr(cls, k_, v_)
    cls.passthrough = lambda self, c, e: False
    cls.report = lambda self, t, m: self.msgs.append(m)
    f = cls()
    f.msgs, f.data = [], ""
    for k_, v_ in props.items():
        setattr(f, k_, v_)
    return f
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.cursor.location = (0, 0, 0)
ctx.scene.nproto_settings.construction_history = True
bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.5, height=1.2, caps=True)
MCY = ctx.active_object
fc = _FCtx()
rim = 0.5 * np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
cv = _fake(_dt.NPROTO_OT_draw_curve, mode="CV", degree="3", closed=False, start_cont="G1", end_cont="G1",
           edge_mode="SURFACE")
cv._edit, cv._pts, cv._snapinfo, cv._frames, cv._start_refs = None, [], [], {}, None
cv.sketch_start(fc)
cv.modal(fc, _ev('MOUSEMOVE', _wpx(*rim), 'NOTHING'))
lab1 = _drw.sketch_overlay["snap_label"]
nref = len(_drw.sketch_overlay["refs"])
src1 = cv._refs["cands"][cv._refs["i"]]["src"] if cv._refs else None
cv.modal(fc, _ev('T', _wpx(*rim)))
lab2 = _drw.sketch_overlay["snap_label"]
src2 = cv._refs["cands"][cv._refs["i"]]["src"] if cv._refs else None
check("sketch modal: a rim point offers both faces along the edge, T switches face (highlighted)",
      lab1 == "Edge · Face 1/2 (T)" and lab2 == "Edge · Face 2/2 (T)" and nref >= 2 and src1 and src2
      and src1[3] != src2[3] and src1[:2] == src2[:2], f"{lab1} / {lab2} {src1} {src2}")
cv.modal(fc, _ev('LEFTMOUSE', _wpx(*rim)))
cv.modal(fc, _ev('LEFTMOUSE', _wpx(0.95, 0.55)))
cv.modal(fc, _ev('LEFTMOUSE', _wpx(1.4, 0.3)))
cv.modal(fc, _ev('LEFTMOUSE', _wpx(1.9, 0.5)))
r = cv.modal(fc, _ev('RET'))
MO = ctx.active_object
h_ = features.history(MO) if MO is not None and MO.type == 'CURVE' else None
e0 = h_["params"]["ends"]["0"] if h_ else {}
kcs = features.input_curve(MO, "S0") if h_ else None
if kcs is not None:
    ds = _al._k()[0].evaluate_curve(kcs, kcs.parameter_range()[0])
    nrm, _kn = _al.surface_frame(MCY, src2[1], np.array(ds["point"], float), src2[3])
    tilt = math.degrees(math.asin(min(1.0, abs(np.array(ds["tangent"]) @ nrm) / np.linalg.norm(ds["tangent"]))))
else:
    tilt = 99
check("sketch modal: clicked CV curve is G1 to the face chosen with T, with an Align history step",
      r == {'FINISHED'} and h_ is not None and e0.get("face") == src2[3] and e0.get("mode") == "SURFACE"
      and e0.get("cont") == "G1" and tilt < 1e-5, f"{r} {e0} tilt {tilt:.1e}° {cv.msgs}")
# Line: G1 start on the rim = tangent to the edge there
ln = _fake(_dt.NPROTO_OT_draw_line, polyline=False, start_mode="TANGENT", line_mode="EDGE")
ln._edit, ln._lines, ln._start, ln._lock, ln._start_refs = None, [], None, None, None
ln._reset_typing()
ln.sketch_start(fc)
ln.modal(fc, _ev('MOUSEMOVE', _wpx(*rim), 'NOTHING'))
ln.modal(fc, _ev('LEFTMOUSE', _wpx(*rim)))
lock1 = ln._lock
ln.modal(fc, _ev('MOUSEMOVE', _wpx(0.9, 1.2), 'NOTHING'))
ln.modal(fc, _ev('LEFTMOUSE', _wpx(0.9, 1.2)))
a_, b_ = ln._lines[0] if ln._lines else (np.zeros(3), np.zeros(3))
dirn = (b_ - a_) / (np.linalg.norm(b_ - a_) or 1)
check("sketch modal: a Line started on the rim with S (G1) runs tangent to the rim",
      lock1 == "T" and abs(dirn @ (a_ * (1, 1, 0)) / np.linalg.norm(a_[:2])) < 1e-6 and abs(dirn[2]) < 1e-9
      and np.linalg.norm(b_ - a_) > 0.1, f"{lock1} {dirn}")
r = ln.modal(fc, _ev('RIGHTMOUSE'))
check("sketch modal: the tangent line is made", r == {'FINISHED'} and ctx.active_object.type == 'CURVE')
# Line at the seam corner: two edges meet (the rim, the seam); T switches which one the tangent follows
ln = _fake(_dt.NPROTO_OT_draw_line, polyline=False, start_mode="TANGENT", line_mode="EDGE")
ln._edit, ln._lines, ln._start, ln._lock, ln._start_refs = None, [], None, None, None
ln._reset_typing()
ln.sketch_start(fc)
seam = None
for e_ in store.get_brep(MCY).edges():
    p0_ = np.array(e_.point_at(e_.parameter_range()[0]))
    if p0_[2] > 0.5:
        seam = p0_
        break
ln.modal(fc, _ev('MOUSEMOVE', _wpx(seam[0], seam[1]), 'NOTHING'))
labc = _drw.sketch_overlay["snap_label"]
ln.modal(fc, _ev('LEFTMOUSE', _wpx(seam[0], seam[1])))
t1 = np.array(ln._tvec) / np.linalg.norm(ln._tvec) if getattr(ln, "_tvec", None) is not None else None
ln.modal(fc, _ev('T', _wpx(seam[0], seam[1])))
t2 = np.array(ln._tvec) / np.linalg.norm(ln._tvec) if getattr(ln, "_tvec", None) is not None else None
vert = lambda t: t is not None and abs(abs(t[2]) - 1) < 1e-6
horiz = lambda t: t is not None and abs(t[2]) < 1e-6
check("sketch modal: at a corner T switches between the edges meeting there (rim tangent / seam)",
      labc is not None and labc.startswith("End · Edge 1/") and ((vert(t1) and horiz(t2)) or (horiz(t1) and vert(t2))),
      f"{labc} {t1} {t2}")
ln.modal(fc, _ev('ESC'))
# Polyline: click, click, click, click the first point -> closed, then right-click
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
pl_ = _fake(_dt.NPROTO_OT_draw_line, polyline=True, start_mode="FREE", line_mode="EDGE")
pl_._edit, pl_._lines, pl_._start, pl_._lock, pl_._start_refs = None, [], None, None, None
pl_._reset_typing()
pl_.sketch_start(fc)
for q_ in ((0, 0), (1, 0), (1, 1), (0, 0)):
    pl_.modal(fc, _ev('MOUSEMOVE', _wpx(*q_), 'NOTHING'))
    pl_.modal(fc, _ev('LEFTMOUSE', _wpx(*q_)))
n_l, st_ = len(pl_._lines), pl_._start
r = pl_.modal(fc, _ev('RIGHTMOUSE'))
o_ = ctx.active_object
check("sketch modal: polyline clicks chain, clicking the first point closes it into one curve",
      n_l == 3 and st_ is None and r == {'FINISHED'} and o_ is not None and o_.data.splines[0].use_cyclic_u
      and len(o_.data.splines[0].points) == 3, f"{n_l} {st_} {r}")

# --- At a corner shared by two edges: the edge slid along stays the reference;
#     L locks the point onto one edge
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.nproto.add_primitive(kind="PLANE", width=2, depth=2, pl_degree_u=1, pl_degree_v=1, spans_u=1, spans_v=1)
def _new_ln():
    f = _fake(_dt.NPROTO_OT_draw_line, polyline=False, start_mode="TANGENT", line_mode="EDGE")
    f._edit, f._lines, f._start, f._lock, f._start_refs = None, [], None, None, None
    f._reset_typing()
    f.sketch_start(fc)
    return f
def _corner_from(path):
    f = _new_ln()
    for q_ in path:
        f.modal(fc, _ev('MOUSEMOVE', _wpx(*q_), 'NOTHING'))
    lab_ = _drw.sketch_overlay["snap_label"]
    f.modal(fc, _ev('LEFTMOUSE', _wpx(*path[-1])))
    t_ = np.array(f._tvec) / np.linalg.norm(f._tvec) if f._lock == "T" else np.zeros(3)
    f.modal(fc, _ev('ESC'))
    return lab_, t_
lab_a, ta = _corner_from([(0.2, 1), (0.6, 1), (0.9, 1), (1, 1)])
lab_b, tb = _corner_from([(1, 0.2), (1, 0.6), (1, 0.9), (1, 1)])
check("corner: the edge slid along into a shared end stays the reference (top edge -> X, right edge -> Y)",
      abs(abs(ta[0]) - 1) < 1e-9 and abs(abs(tb[1]) - 1) < 1e-9 and lab_a == lab_b == "End · Edge 1/2 (T)",
      f"{lab_a} {ta} / {lab_b} {tb}")
f = _new_ln()
f.modal(fc, _ev('MOUSEMOVE', _wpx(0.3, 1), 'NOTHING'))
lab0 = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('L', _wpx(0.3, 1)))
f.modal(fc, _ev('MOUSEMOVE', _wpx(0.3, 0.2), 'NOTHING'))
P_l, sn_l = f.pick_point(fc, _ev('MOUSEMOVE', _wpx(0.3, 0.2), 'NOTHING'))
check("lock onto an edge (L): the point stays on the edge wherever the mouse goes",
      lab0 == "Edge" and np.allclose(P_l, (0.3, 1, 0), atol=1e-6) and sn_l is not None
      and sn_l[3] == "Edge · locked (L)", f"{lab0} {P_l} {sn_l}")
f.modal(fc, _ev('MOUSEMOVE', _wpx(1.0, 0.97), 'NOTHING'))
lab1 = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('LEFTMOUSE', _wpx(1.0, 0.97)))
t1 = np.array(f._tvec) / np.linalg.norm(f._tvec) if f._lock == "T" else np.zeros(3)
check("lock onto an edge: at the shared corner only that edge counts; the click releases the lock",
      lab1 == "End · locked (L)" and np.allclose(f._start, (1, 1, 0), atol=1e-9) and abs(abs(t1[0]) - 1) < 1e-9
      and f._edge_lock is None, f"{lab1} {f._start} {t1}")
f.modal(fc, _ev('ESC'))

# --- Profile: circle / ellipse / square / rectangle perpendicular to a curve or
#     surface edge, following it (history)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
PPATH = nurbs("PPath", [(-1, 0, 0), (1, 0, 0)], order=2)
def _type(f, ch, type_, xy):
    return f.modal(fc, _types.SimpleNamespace(type=type_, value='PRESS', mouse_x=float(xy[0]), mouse_y=float(xy[1]),
                                              shift=False, ctrl=False, alt=False, oskey=False, unicode=ch))
def _new_pf(**kw):
    props = dict(shape="CIRCLE", radius=0.25, radius2=0.15, width=0.5, depth=0.3, at=0.0, rotation=0.0,
                 degree="0", spans_u=8)
    props.update(kw)
    f = _fake(_dt.NPROTO_OT_draw_profile, **props)
    f._center = f._src = f._frame = None
    f._lock = None
    f._buf, f._field, f._s1, f._s2 = "", 1, None, None
    f.sketch_start(fc)
    return f
def _samples(o, n=24):
    kc_ = features.input_curve(o, "S0")
    a_, b_ = kc_.parameter_range()
    return np.array([kc_.evaluate(a_ + (b_ - a_) * k / n) for k in range(n)], float)
M_ = _pt_frame = __import__("blender_nurbs.primitives", fromlist=["x"]).profile_frame(PPATH, "S0", 0.5)
check("profile frame on a line: Z along the line, Y up, at the middle",
      abs(abs(M_[0, 2]) - 1) < 1e-9 and np.allclose(M_[:3, 1], (0, 0, 1)) and np.allclose(M_[:3, 3], (0, 0, 0)),
      f"{M_}")
pf = _new_pf()
pf.modal(fc, _ev('MOUSEMOVE', _wpx(0.3, 0), 'NOTHING'))
prev_ok = bool(_drw.sketch_overlay["curves"])
pf.modal(fc, _ev('LEFTMOUSE', _wpx(0.3, 0)))
at_ = pf.at
pf.modal(fc, _ev('MOUSEMOVE', _wpx(0.3, 0.4), 'NOTHING'))
r = pf.modal(fc, _ev('LEFTMOUSE', _wpx(0.3, 0.4)))
PCI = ctx.active_object
S_ = _samples(PCI) if r == {'FINISHED'} else np.zeros((3, 3))
h_ = features.history(PCI) if r == {'FINISHED'} else None
check("profile (circle) on a curve: centred on it, perpendicular, radius dragged out, history to the curve",
      prev_ok and r == {'FINISHED'} and abs(at_ - 0.65) < 1e-6 and np.allclose(S_[:, 0], 0.3, atol=1e-6)
      and np.allclose(np.linalg.norm(S_ - (0.3, 0, 0), axis=1), 0.4, atol=1e-5) and h_ is not None
      and h_["op"] == "perp_profile" and h_["inputs"][0][0] == PPATH, f"{r} {at_} {S_[:2]}")
PPATH.location = (0, 0.5, 0.2)
features.rebuild(PCI, ctx)
S_ = _samples(PCI)
check("profile: moving the curve moves the profile with it (still perpendicular, same size)",
      np.allclose(S_.mean(axis=0), (0.3, 0.5, 0.2), atol=1e-6) and np.allclose(S_[:, 0], 0.3, atol=1e-6)
      and np.allclose(np.linalg.norm(S_ - (0.3, 0.5, 0.2), axis=1), 0.4, atol=1e-5), f"{S_.mean(axis=0)}")
PCI.nproto_feature.radius = 0.2
S_ = _samples(PCI)
r2_ = np.linalg.norm(S_ - (0.3, 0.5, 0.2), axis=1)
PCI.nproto_feature.pp_shape = "SQUARE"
PCI.nproto_feature.width = 0.3
cv_ = np.array([(PCI.matrix_world @ Vector(p_.co[:3]))[:] for p_ in PCI.data.splines[0].points])
check("profile: Radius / Shape / Size edited in the Feature panel rebuild it in place",
      np.allclose(r2_, 0.2, atol=1e-5) and len(cv_) == 4 and np.allclose(cv_[:, 0], 0.3, atol=1e-6)
      and np.allclose(np.abs(cv_[:, 1] - 0.5), 0.15, atol=1e-6) and np.allclose(np.abs(cv_[:, 2] - 0.2), 0.15, atol=1e-6),
      f"{r2_[:2]} {cv_}")
PPATH.location = (0, 0, 0)
bpy.ops.nproto.add_primitive(kind="PLANE", width=2, depth=2, pl_degree_u=1, pl_degree_v=1, spans_u=1, spans_v=1)
pf = _new_pf()
pf.modal(fc, _ev('MOUSEMOVE', _wpx(0.2, 1), 'NOTHING'))
lab_ = _drw.sketch_overlay["snap_label"]
pf.modal(fc, _ev('S', _wpx(0.2, 1)))
pf.modal(fc, _ev('LEFTMOUSE', _wpx(0.2, 1)))
_type(pf, "0", 'ZERO', _wpx(0.2, 1.1))
_type(pf, ".", 'PERIOD', _wpx(0.2, 1.1))
_type(pf, "5", 'FIVE', _wpx(0.2, 1.1))
r = _type(pf, "", 'RET', _wpx(0.2, 1.1))
PSQ = ctx.active_object
cv_ = np.array([(PSQ.matrix_world @ Vector(p_.co[:3]))[:] for p_ in PSQ.data.splines[0].points]) if r == {'FINISHED'} \
    else np.zeros((4, 3))
check("profile (square, typed 0.5) on a surface edge: square to the surface (sides along its normal)",
      r == {'FINISHED'} and len(cv_) == 4 and np.allclose(cv_[:, 0], 0.2, atol=1e-6)
      and np.allclose(np.abs(cv_[:, 1] - 1), 0.25, atol=1e-6) and np.allclose(np.abs(cv_[:, 2]), 0.25, atol=1e-6)
      and features.history(PSQ)["params"]["pp_shape"] == "SQUARE", f"{lab_} {r} {cv_}")

# --- Draw Plane: two corners or typed X / Y; degree and spans per direction
C_, X_, Y_, sg_ = _dt.plane_rect((0, 0, 0), (1.5, -0.8, 0), (1, 0, 0), (0, 1, 0))
check("draw plane: two corners give the centre and the X / Y sizes",
      np.allclose(C_, (0.75, -0.4, 0)) and abs(X_ - 1.5) < 1e-12 and abs(Y_ - 0.8) < 1e-12 and sg_ == (1.0, -1.0))
C_, X_, Y_, _ = _dt.plane_rect((1, 1, 0), (2, 1.5, 0), (1, 0, 0), (0, 1, 0), from_center=True)
C2_, X2_, Y2_, _ = _dt.plane_rect((0, 0, 0), (1, 0.4, 0), (1, 0, 0), (0, 1, 0), square=True)
C3_, X3_, Y3_, _ = _dt.plane_rect((0, 0, 0), (-0.3, 0.2, 0), (1, 0, 0), (0, 1, 0), size=(3.0, None))
check("draw plane: from the centre, square, typed X (the mouse gives the side)",
      np.allclose(C_, (1, 1, 0)) and np.allclose((X_, Y_), (2, 1)) and np.allclose(C2_, (0.5, 0.5, 0))
      and np.allclose((X2_, Y2_), (1, 1)) and np.allclose(C3_, (-1.5, 0.1, 0)) and np.allclose((X3_, Y3_), (3, 0.2)))
Mf_ = _dt.plane_matrix((0, 0, 0), (1, 0, 0), (0, 0, 1))
check("draw plane: drawn in a front view it stands up, its normal towards the viewer (-Y)",
      np.allclose(Mf_[:3, 2], (0, -1, 0)) and abs(np.linalg.det(Mf_[:3, :3]) - 1) < 1e-12)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
def _new_dp(**kw):
    props = dict(size_x=2.0, size_y=2.0, from_center=False, degree_u=3, degree_v=2, spans_u=2, spans_v=1)
    props.update(kw)
    f = _fake(_dt.NPROTO_OT_draw_plane, **props)
    f._start, f._lock = None, None
    f._reset_typing()
    f.sketch_start(fc)
    return f
def _surf_info(o):
    d_ = ext.surface_cvs(store.get_surfaces(o)[0])
    M_ = np.array(geom.world_matrix(o))
    W_ = np.array(d_["points"], float) @ M_[:3, :3].T + M_[:3, 3]
    return d_["count_u"], d_["count_v"], d_["degree_u"], d_["degree_v"], W_.min(axis=0), W_.max(axis=0)
dp = _new_dp()
dp.modal(fc, _ev('MOUSEMOVE', _wpx(0.2, 0.1), 'NOTHING'))
dp.modal(fc, _ev('LEFTMOUSE', _wpx(0.2, 0.1)))
dp.modal(fc, _ev('MOUSEMOVE', _wpx(1.4, 0.9), 'NOTHING'))
r = dp.modal(fc, _ev('LEFTMOUSE', _wpx(1.4, 0.9)))
PL = ctx.active_object
nu_, nv_, du_, dv_, lo_, hi_ = _surf_info(PL) if r == {'FINISHED'} else (0, 0, 0, 0, np.zeros(3), np.zeros(3))
hp_ = features.history(PL)["params"] if r == {'FINISHED'} else {}
check("draw plane (clicks): corner to corner, degree 3 x 2 with 2 x 1 spans -> 5 x 3 CVs, in the drawing plane",
      r == {'FINISHED'} and (nu_, nv_, du_, dv_) == (5, 3, 3, 2) and np.allclose(lo_, (0.2, 0.1, 0), atol=1e-6)
      and np.allclose(hi_, (1.4, 0.9, 0), atol=1e-6) and abs(hp_.get("width", 0) - 1.2) < 1e-6
      and abs(hp_.get("depth", 0) - 0.8) < 1e-6, f"{r} {(nu_, nv_, du_, dv_)} {lo_} {hi_} {hp_}")
def _type(f, ch, type_, xy):
    return f.modal(fc, _types.SimpleNamespace(type=type_, value='PRESS', mouse_x=float(xy[0]), mouse_y=float(xy[1]),
                                              shift=False, ctrl=False, alt=False, oskey=False, unicode=ch))
dp = _new_dp(degree_u=3, degree_v=3, spans_u=1, spans_v=1)
q0 = _wpx(0, 0)
dp.modal(fc, _ev('LEFTMOUSE', q0))
qm = _wpx(-0.5, 0.5)
_type(dp, "2", 'TWO', qm)
_type(dp, "", 'TAB', qm)
_type(dp, "1", 'ONE', qm)
r = _type(dp, "", 'RET', qm)
PT = ctx.active_object
nu_, nv_, du_, dv_, lo_, hi_ = _surf_info(PT) if r == {'FINISHED'} else (0, 0, 0, 0, np.zeros(3), np.zeros(3))
check("draw plane (typed): X 2, Tab, Y 1, Enter -> 2 x 1 on the side of the mouse, 4 x 4 CVs",
      r == {'FINISHED'} and (nu_, nv_) == (4, 4) and np.allclose(lo_, (-2, 0, 0), atol=1e-6)
      and np.allclose(hi_, (0, 1, 0), atol=1e-6), f"{r} {lo_} {hi_}")
PL.nproto_feature.pl_degree_u = 5
nu_, nv_, du_, dv_, lo_, hi_ = _surf_info(PL)
check("plane: Degree U changed in the Feature panel rebuilds it, same size", (nu_, du_, nv_, dv_) == (7, 5, 3, 2)
      and np.allclose(lo_, (0.2, 0.1, 0), atol=1e-6) and np.allclose(hi_, (1.4, 0.9, 0), atol=1e-6), f"{nu_} {du_}")
_old = ext.surface_cvs(features.run("prim_plane", [], {"width": 1.0, "depth": 1.0, "degree": 1, "spans_u": 1,
                                                       "spans_v": 1}))
check("plane: one made before Degree U / V (one degree) still builds the same", _old["degree_u"] == 1
      and _old["degree_v"] == 1 and _old["count_u"] == 2)

# --- Loft: Closed (periodic through the sections and back to the first)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
def _ring_square(name, a_deg, R=2.0, h=0.2):
    a = math.radians(a_deg)
    c = np.array([R * math.cos(a), R * math.sin(a), 0.0])
    X = np.array([-math.cos(a), -math.sin(a), 0.0])
    Z = np.array([0.0, 0.0, 1.0])
    pts = [c - h * X - h * Z, c + h * X - h * Z, c + h * X + h * Z, c - h * X + h * Z]
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(3)
    for p_, q_ in zip(sp.points, pts):
        p_.co = (*q_, 1.0)
    sp.order_u = 2; sp.use_cyclic_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    return o, pts
sq = [_ring_square(f"RS{k}", 90 * k) for k in range(4)]
order = [0, 2, 1, 3]                       # picked across the ring: the loft must go round it
r = bpy.ops.nproto.loft(inputs=ins(*[(sq[k][0], "S0") for k in order]), closed=True)
RL = ctx.active_object
def _raw(o):
    """The loft's surface before it is split at its folds."""
    h_ = features.history(o)
    return features._build_loft(h_["inputs"], h_["params"])
def _holds(o, sections, n=12):
    """Largest distance between each section curve and the surface's iso curve
    through it (U runs along the loft, V along the sections)."""
    srf = _raw(o)
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    worst = 0.0
    for so in sections:
        kc_ = features.input_curve(so, "S0")
        a_, b_ = kc_.parameter_range()
        C = [np.array(kc_.evaluate(a_ + (b_ - a_) * j / n), float) for j in range(n + 1)]
        f = lambda u: np.linalg.norm(np.array(srf.evaluate(u, v0), float) - C[0])
        us = np.linspace(u0, u1, 801)
        k = int(np.argmin([f(u) for u in us]))
        lo, hi = us[max(k - 1, 0)], us[min(k + 1, 800)]
        for _ in range(80):                          # golden section on the start point
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            lo, hi = (lo, m2) if f(m1) < f(m2) else (m1, hi)
        u = 0.5 * (lo + hi)
        worst = max(worst, max(np.linalg.norm(np.array(srf.evaluate(u, v0 + (v1 - v0) * j / n), float) - C[j])
                               for j in range(n + 1)))
    return worst
def _seam(o):
    """(gap, tangent jump, curvature jump) where the loft direction (U) closes."""
    srf = _raw(o)
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    hh = 1e-4 * (u1 - u0)
    gap = d1 = d2 = 0.0
    for k in range(7):
        v = v0 + (v1 - v0) * (k + 0.5) / 7
        S = lambda u: np.array(srf.evaluate(u, v), float)
        a0, a1, a2 = S(u0), S(u0 + hh), S(u0 + 2 * hh)
        b0, b1, b2 = S(u1), S(u1 - hh), S(u1 - 2 * hh)
        gap = max(gap, np.linalg.norm(a0 - b0))
        t0, t1 = (a1 - a0) / hh, (b0 - b1) / hh
        d1 = max(d1, np.linalg.norm(t0 - t1) / (np.linalg.norm(t0) or 1))
        c0, c1 = (a2 - 2 * a1 + a0) / hh ** 2, (b2 - 2 * b1 + b0) / hh ** 2
        d2 = max(d2, np.linalg.norm(c0 - c1) / (np.linalg.norm(c0) + 1e-9))
    return gap, d1, d2
dmax = _holds(RL, [o for o, _ in sq]) if r == {'FINISHED'} else 1
gap, d1, d2 = _seam(RL) if r == {'FINISHED'} else (1, 1, 1)
check("loft closed: through all four sections (given out of order) and round to the first",
      r == {'FINISHED'} and dmax < 1e-6 and gap < 1e-9, f"{r} {dmax:.1e} gap {gap:.1e}")
check("loft closed: smooth across the seam (tangent and curvature continuous)", d1 < 1e-3 and d2 < 1e-2,
      f"{d1:.1e} {d2:.1e}")
RL.nproto_feature.closed = False
gap_o, _, _ = _seam(RL)
check("loft: switching Closed off in the Feature panel gives the open loft again", gap_o > 0.5, f"{gap_o:.2f}")
circ = []
for k in range(4):
    a = math.radians(90 * k)
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.3, degree="0")
    o = ctx.active_object
    o.location = (2 * math.cos(a), 2 * math.sin(a), 0.0)
    o.rotation_euler = (math.pi / 2, 0.0, a)
    circ.append(o)
ctx.view_layer.update()
r = bpy.ops.nproto.loft(inputs=ins(*[(o, "S0") for o in circ]), closed=True)
RC = ctx.active_object
dmax = _holds(RC, circ) if r == {'FINISHED'} else 1
gap, d1, d2 = _seam(RC) if r == {'FINISHED'} else (1, 1, 1)
check("loft closed through exact circles: holds every circle (rational), closed and smooth at the seam",
      r == {'FINISHED'} and dmax < 1e-6 and gap < 1e-9 and d1 < 1e-3, f"{dmax:.1e} {gap:.1e} {d1:.1e}")
# a loft through rectangles has four folds (C0 lines): tessellated piece by piece
def _tilted(P, N, T):
    P, N, T = np.asarray(P, float), np.asarray(N, float), store._tri_array(T)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]
    Nu = N / np.maximum(np.linalg.norm(N, axis=1), 1e-30)[:, None]
    return int((np.min(np.stack([np.abs(np.sum(fn * Nu[T[:, k]], axis=1)) for k in range(3)], 1), axis=1) < 0.7).sum())
RL.nproto_feature.closed = True
srf_ = _raw(RL)
pcs_ = store._crease_pieces(srf_, 5.0) or []
dev_ = 0.0
for pc_ in pcs_:
    (a0, a1), (b0, b1) = pc_.u_domain(), pc_.v_domain()
    for uu in np.linspace(a0, a1, 5):
        for vv in np.linspace(b0, b1, 5):
            dev_ = max(dev_, np.linalg.norm(np.array(pc_.evaluate(uu, vv)) - np.array(srf_.evaluate(uu, vv))))
P1_, N1_, T1_ = store._tessellate(store.to_brep(srf_), 0.0005, 5.0, 2.0, 1 / 30, natural=True)
P0_, N0_, T0_ = store._tessellate(store.to_brep(srf_), 0.0005, 5.0, 2.0, 1 / 30, natural=False)
check("a surface with 4 folds splits exactly into smooth pieces, tessellated with sharp creases",
      len(pcs_) == 4 and dev_ < 1e-12 and _tilted(P1_, N1_, T1_) == 0 and _tilted(P0_, N0_, T0_) > 100,
      f"{len(pcs_)} pieces, dev {dev_:.1e}, smeared {_tilted(P1_, N1_, T1_)} (was {_tilted(P0_, N0_, T0_)}), "
      f"{len(P1_)} verts (was {len(P0_)})")
from blender_nurbs import contlabels as _clb
RL.nproto_feature.closed = False
_labs = _clb.labels(RL)
check("loft through rectangles: its G0 labels hold (the check finds the right side of each fold)",
      len(_labs) == 2 and all(L_["check"] and L_["check"]["ok"] for L_ in _labs),
      f"{[L_['check'] and L_['check']['text'] for L_ in _labs]}")
RL.nproto_feature.closed = True
_brl = store.get_brep(RL)
_sh = store.shared_edges(RL)
check("loft through rectangles: one face per side, joined along the folds (a closed ring: a solid)",
      _brl.face_count() == 4 and len(_sh) >= 4 and bool(_brl.is_manifold_solid()),
      f"{_brl.face_count()} faces, {len(_sh)} shared edges, solid {_brl.is_manifold_solid()}")
_n_before = _brl.face_count()
_fold_edges = [i for i in _sh if np.ptp(np.array(store.display(RL)["edges"][i])[:, 2]) < 1e-6
               and abs(np.array(store.display(RL)["edges"][i])[0, 2]) > 0.1]       # the 4 ring corners (z = +-0.2)
try:
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[RL.name, f"E{i}"] for i in _fold_edges]), radius=0.05)
except RuntimeError as exc:
    r = str(exc)
check("Edge Fillet rounds the folds of a loft through rectangles", r == {'FINISHED'} and len(_fold_edges) == 4
      and store.get_brep(RL).face_count() > _n_before, f"{r} {len(_fold_edges)} edges -> {store.get_brep(RL).face_count()} faces")
_labs = _clb.labels(RL)
check("after Edge Fillet the loft's G0 labels still hold (checked on the loft, not the rounded result)",
      len(_labs) == 2 and all(L_["check"] and L_["check"]["ok"] for L_ in _labs),
      f"{[L_['check'] and L_['check']['text'] for L_ in _labs]}")
_cl_old = _clb.measure(RL, *_labs[0]["src"], 0)
check("(the rounded result itself is off the sections at the corners)", not _cl_old["ok"], _cl_old["text"])
check("a smooth closed loft (exact circles: G1 at its double knots) stays one face",
      store._crease_pieces(store.get_surfaces(RC)[0], 1.0) is None and store.get_brep(RC).face_count() == 1)
try:
    r = bpy.ops.nproto.loft(inputs=ins((circ[0], "S0"), (circ[1], "S0")), closed=True)
except RuntimeError as exc:
    r = str(exc)
check("loft closed with only two sections is refused", "at least 3 sections" in str(r), f"{r}")

# --- STEP import: our own export comes back with the same shape
from blender_nurbs import step_import as _si, step_export as _se
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
if ctx.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
ctx.scene.unit_settings.scale_length = 1.0
bpy.context.scene.cursor.location = (0, 0, 0)
r = bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.5, height=1.2, caps=True)
CYL = ctx.active_object
tmp_step = os.path.join(tempfile.mkdtemp(), "cyl.step")
_se.export_step(tmp_step, [CYL], unit="MM")
res, rep = _si.read_step(tmp_step, 1.0)
check("step import: our exported capped cylinder comes back as one closed solid",
      rep["bodies"] == 1 and rep["failed"] == 0 and rep["solids"] == 1, f"{rep}")
sm_ = _kl()
bb = np.array(sm_.brep_bounding_box(res[0][1], True))
check("step import: millimetre file -> Blender units, same size", np.allclose(bb[1] - bb[0], (1.0, 1.0, 1.2), atol=2e-3),
      f"{bb.tolist()}")
r = bpy.ops.nproto.import_step(filepath=tmp_step)
imp_objs = [o for o in ctx.selected_objects]
check("step import operator: one NURBS object in a new collection", r == {'FINISHED'} and len(imp_objs) == 1
      and store.is_nurbs(imp_objs[0]) and imp_objs[0].users_collection[0].name.startswith("cyl"))
# analytic surfaces: a hand-written STEP cylinder (CYLINDRICAL_SURFACE + 2 PLANEs + circles + seam line), in inches
step_txt = """ISO-10303-21;
HEADER;FILE_DESCRIPTION((''),'2;1');FILE_NAME('c','',(''),(''),'','','');FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));ENDSEC;
DATA;
#1=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));
#2=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#1);
#3=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);
#4=(CONVERSION_BASED_UNIT('INCH',#2) LENGTH_UNIT() NAMED_UNIT(#3));
#5=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));
#6=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#4,#5)) REPRESENTATION_CONTEXT('',''));
#10=CARTESIAN_POINT('',(0.,0.,0.));
#11=DIRECTION('',(0.,0.,1.));
#12=DIRECTION('',(1.,0.,0.));
#13=AXIS2_PLACEMENT_3D('',#10,#11,#12);
#14=CARTESIAN_POINT('',(0.,0.,2.));
#15=AXIS2_PLACEMENT_3D('',#14,#11,#12);
#20=CYLINDRICAL_SURFACE('',#13,1.);
#21=PLANE('',#13);
#22=PLANE('',#15);
#30=CIRCLE('',#13,1.);
#31=CIRCLE('',#15,1.);
#32=CARTESIAN_POINT('',(1.,0.,0.));
#33=CARTESIAN_POINT('',(1.,0.,2.));
#34=VERTEX_POINT('',#32);
#35=VERTEX_POINT('',#33);
#36=VECTOR('',#11,1.);
#37=LINE('',#32,#36);
#40=EDGE_CURVE('',#34,#34,#30,.T.);
#41=EDGE_CURVE('',#35,#35,#31,.T.);
#42=EDGE_CURVE('',#34,#35,#37,.T.);
#50=ORIENTED_EDGE('',*,*,#40,.T.);
#51=ORIENTED_EDGE('',*,*,#42,.T.);
#52=ORIENTED_EDGE('',*,*,#41,.F.);
#53=ORIENTED_EDGE('',*,*,#42,.F.);
#54=EDGE_LOOP('',(#50,#51,#52,#53));
#55=FACE_OUTER_BOUND('',#54,.T.);
#56=ADVANCED_FACE('',(#55),#20,.T.);
#60=ORIENTED_EDGE('',*,*,#40,.F.);
#61=EDGE_LOOP('',(#60));
#62=FACE_OUTER_BOUND('',#61,.T.);
#63=ADVANCED_FACE('',(#62),#21,.F.);
#64=ORIENTED_EDGE('',*,*,#41,.T.);
#65=EDGE_LOOP('',(#64));
#66=FACE_OUTER_BOUND('',#65,.T.);
#67=ADVANCED_FACE('',(#66),#22,.T.);
#70=CLOSED_SHELL('',(#56,#63,#67));
#71=MANIFOLD_SOLID_BREP('Can',#70);
#72=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#71,#13),#6);
ENDSEC;
END-ISO-10303-21;
"""
tmp2 = os.path.join(tempfile.mkdtemp(), "can.step")
open(tmp2, "w").write(step_txt)
res, rep = _si.read_step(tmp2, 1.0)
bb = np.array(sm_.brep_bounding_box(res[0][1], True)) if res else None
check("step import: analytic cylinder + planes (inch file) -> closed exact solid",
      rep["faces"] == 3 and rep["solids"] == 1 and bb is not None
      and np.allclose(bb[1] - bb[0], (0.0508, 0.0508, 0.0508), atol=1e-5), f"{rep} {None if bb is None else bb.tolist()}")
vol = sm_.compute_volume(res[0][1]) if res else 0
vol = vol[0] if isinstance(vol, tuple) else vol
check("step import: cylinder volume matches pi r^2 h (kernel volume is numeric)", abs(abs(vol) / (math.pi * 0.0254 ** 2 * 0.0508) - 1) < 1e-4,
      f"{vol} vs {math.pi * 0.0254 ** 2 * 0.0508}")


# --- Surface Align (edges G0 / G1 / G2 to the touching surfaces) ------------
from math import comb as _comb
from blender_nurbs import salign as _sa
ctx.scene.nproto_settings.construction_history = True
_F = lambda x, y: 0.05 * x**3 - 0.1 * x * y**2 + 0.08 * y**3 + 0.2 * x * y - 0.1 * y**2
_Bi = np.linalg.inv(np.array([[_comb(3, j) * s**j * (1 - s)**(3 - j) for j in range(4)] for s in np.linspace(0, 1, 4)]))
def _quad(name, x0, x1, y0, y1):
    X, Y = np.meshgrid(np.linspace(x0, x1, 4), np.linspace(y0, y1, 4), indexing="ij")
    C = [_Bi @ A @ _Bi.T for A in (X, Y, _F(X, Y))]
    pts = [(C[0][i, j] + 20, C[1][i, j], C[2][i, j]) for i in range(4) for j in range(4)]
    return store.new_object(ctx, name, ext.surface_from_cvs(3, 3, pts, [0]*4 + [1]*4, [0]*4 + [1]*4))
saN, saN2, saNR = _quad("SA_N", 0, 2, 0, 2), _quad("SA_N2", 2, 4, -2, 0), _quad("SA_NR", 4, 6, 0, 2)
_xs, _ys = np.linspace(2, 4, 6), np.linspace(0, 2, 6)
_pts = [(x + 20, y, _F(x, y) + 0.4 * (x - 2) * (4 - x) * y * (2 - y) + 0.01 * math.sin(3 * y)) for x in _xs for y in _ys]
saS = store.new_object(ctx, "SA_S", ext.surface_from_cvs(3, 3, _pts, geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(6, 3)))
def _sa_rep(o, jobs_):
    rep = []
    _sa.align_text(o, o[store.PROP_BREP], jobs_, report=rep, measure_only=True)
    return rep
def _sa_nums(rep):
    import re
    return [[float(x) for x in re.findall(r"(?:gap|angle|curvature) ([-0-9.e+]+)", r)] for r in rep]
_before = _sa_nums(_sa_rep(saS, [("U0", (saN, "U1"), "G2")]))[0]
r = bpy.ops.nproto.surface_align(inputs=ins((saS, "U0"), (saS, "V0"), (saS, "U1")), cont_U0="G2", cont_V0="G2", cont_U1="G2")
_jobs = [("U0", (saN, "U1"), "G2"), ("V0", (saN2, "V1"), "G2"), ("U1", (saNR, "U0"), "G2")]
_after = _sa_nums(_sa_rep(saS, _jobs))
check("surface align: 3 edges G2 at once, neighbours found automatically (gap, angle, curvature %)",
      r == {'FINISHED'} and _before[1] > 5 and all(g < 1e-4 and a < 0.02 and k < 5 for g, a, k in _after)
      and [e["op"] for e in features.stack(saS)] == ["surface_align"] and store.is_natural(saS), f"{_before} -> {_after}")
features.remove_step(saS, 0); features.rebuild(saS, ctx, force=True)
r = bpy.ops.nproto.surface_align(inputs=ins((saS, "U0")), cont_U0="G1")
saN.location.z += 0.2; ctx.view_layer.update(); features.rebuild(saS, ctx)
_f = _sa_nums(_sa_rep(saS, [("U0", (saN, "U1"), "G1")]))[0]
check("surface align: history -- moving the neighbour realigns the edge (G1)", r == {'FINISHED'} and _f[0] < 1e-4 and _f[1] < 0.02, f"{_f}")
saN.location.z -= 0.2; ctx.view_layer.update()
from blender_nurbs import contlabels as _cl
_cl._cache.clear()
_L = _cl.labels(saS)
_r = bpy.ops.nproto.cycle_continuity(obj_name=saS.name, index=0) if _L else None
_cl._cache.clear()
_L2 = _cl.labels(saS)
_f = _sa_nums(_sa_rep(saS, [("U0", (saN, "U1"), "G2")]))[0]
check("continuity labels: Surface Align edge shows a label, click steps G1 -> G2 and realigns",
      len(_L) == 1 and _L[0]["value"] == "G1" and _r == {'FINISHED'} and _L2[0]["value"] == "G2"
      and features.stack(saS)[0]["params"]["cont"] == ["G2"] and len(_f) == 3 and _f[2] < 5, f"{_L} {_L2} {_f}")
# too few rows: a 4-row cubic aligned G2 at both ends gets knots inserted
_pts4 = [(x + 20, y, 0.0) for x in np.linspace(2, 4, 4) for y in _ys]
saS4 = store.new_object(ctx, "SA_S4", ext.surface_from_cvs(3, 3, _pts4, geom.clamped_uniform_knots(4, 3), geom.clamped_uniform_knots(6, 3)))
r = bpy.ops.nproto.surface_align(inputs=ins((saS4, "U0"), (saN, "U1"), (saS4, "U1"), (saNR, "U0")), cont_U0="G2", cont_U1="G2")
_f = _sa_nums(_sa_rep(saS4, [("U0", (saN, "U1"), "G2"), ("U1", (saNR, "U0"), "G2")]))
check("surface align: picked targets, rows added for G2 at both ends",
      r == {'FINISHED'} and ext.surface_cvs(store.get_surfaces(saS4)[0])["count_u"] >= 6
      and all(g < 1e-4 and a < 0.02 and k < 5 for g, a, k in _f), f"{_f}")



# --- continuity check (red ring) and overlapping neighbours -----------------
_m_bad = _cl.measure(saS4, saN, "U1", 2) if False else None
_ov = _quad("SA_OV", 2, 3, 0, 2)        # a second fill on S's own side, sharing S's U0 edge
_fresh = store.new_object(ctx, "SA_S5", ext.surface_from_cvs(3, 3, _pts, geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(6, 3)))
_c = [c for c in __import__("blender_nurbs.ops", fromlist=["x"])._pick_candidates(ctx) if c[0] != _fresh]
_nb = _sa.find_neighbour(_fresh, "U0", _c)
check("surface align: an overlapping fill of the same opening is not taken as the neighbour",
      _nb is not None and _nb[0] == saN and _sa.overlaps(_fresh, "U0", _ov, "U0") and not _sa.overlaps(_fresh, "U0", saN, "U1"),
      f"{_nb and (_nb[0].name, _nb[1])}")
_before_chk = _cl.measure(_fresh, saN, "U1", 2)
r = bpy.ops.nproto.surface_align(inputs=ins((_fresh, "U0")), cont_U0="G2")
_cl._cache.clear()
_lab = _cl.labels(_fresh)
check("continuity check: unaligned edge FAILS G2, aligned label holds (no red ring)",
      not _before_chk["ok"] and r == {'FINISHED'} and _lab and _lab[0]["check"] and _lab[0]["check"]["ok"],
      f"{_before_chk['text']} | {_lab and _lab[0]['check']}")
_lab_l = _cl.labels(bpy.data.objects["SA_S"]) if "SA_S" in bpy.data.objects else []

# --- Boundary: exact Coons edges, degree / spans ------------------------------
_bx = 40.0
bdB = nurbs("BD_B", [(_bx + 0, 0, 0), (_bx + 1, -0.2, 0.3), (_bx + 2, 0.3, -0.2), (_bx + 2.5, -0.1, 0.4), (_bx + 3, 0, 0)])
bdR = nurbs("BD_R", [(_bx + 3, 0, 0), (_bx + 3.4, 1, 0.5), (_bx + 3, 2, 0)], order=3)
bdT = nurbs("BD_T", [(_bx + 0, 2, 0.1), (_bx + 1.5, 2.3, 0.6), (_bx + 3, 2, 0)], order=3)      # runs backwards
bdL = nurbs("BD_L", [(_bx + 0, 2, 0.1), (_bx - 0.3, 1.2, -0.3), (_bx + 0.2, 0.6, 0.2), (_bx + 0, 0, 0)])
_bd_in = ins((bdB, "S0"), (bdR, "S0"), (bdT, "S0"), (bdL, "S0"))
def _bd_err(o):
    face = store.get_brep(o).faces()[0]
    err = 0.0
    for c in (bdB, bdR, bdT, bdL):
        kc = features.input_curve(c, "S0"); t0, t1 = kc.parameter_range()
        err = max(err, max(face.closest_point(tuple(kc.evaluate(t0 + (t1 - t0) * k / 30)))[1] for k in range(31)))
    return err
from blender_nurbs import sweeps as _sw
def _bd(**kw):
    r = bpy.ops.nproto.boundary(inputs=_bd_in, **kw)
    o = ctx.active_object
    d = ext.surface_cvs(store.get_surfaces(o)[0])
    return r, o, (d["degree_u"], d["degree_v"]), (d["count_u"], d["count_v"]), _bd_err(o), dict(_sw.boundary_info)
r, bdE, deg, cnt, err, inf = _bd()
check("boundary: exact Coons -- edges are the input curves, their own degree / CVs",
      r == {'FINISHED'} and inf["exact"] and err < 1e-7 and deg == (3, 3) and cnt == (5, 4), f"{deg} {cnt} {err:.2e} {inf}")
r, bdF, deg, cnt, errF, inf = _bd(bd_method="FIT", samples=12)
check("boundary: Exact Edges off -> sampled fit, edges only approximate", r == {'FINISHED'} and not inf["exact"] and errF > 1e-6,
      f"{deg} {cnt} {errF:.2e}")
r, bd5, deg, cnt, err, inf = _bd(sq_degree_u="5", sq_degree_v="4")
check("boundary: higher degree asked -> still exact (degree raised)", r == {'FINISHED'} and inf["exact"] and deg == (5, 4) and err < 1e-7,
      f"{deg} {cnt} {err:.2e}")
r, bdS, deg, cnt, err, inf = _bd(sq_spans_u=4, sq_spans_v=3)
check("boundary: spans that the curves don't have -> fitted with exactly that layout",
      r == {'FINISHED'} and not inf["exact"] and cnt == (4 + 3, 3 + 3) and err < 2e-2, f"{deg} {cnt} {err:.2e}")

r, bdW, deg, cnt, err, inf = _bd(sq_degree_u="2")
check("boundary: exact not possible -> fitted, and the panel says why", r == {'FINISHED'} and not inf["exact"]
      and "below the curves' degree 3" in (inf.get("why") or "") and deg[0] == 2, f"{inf}")


# --- Patch: N-sided trimmed fill, G0 / G1 per edge ---------------------------
from blender_nurbs import patch as _pt, square as _sqm
from blender_nurbs.drawtools import interpolate as _interp
_smk = _kl()
_PF = lambda x, y: 0.25 * math.sin(0.9 * x) * math.cos(0.7 * y) + 0.05 * x * y
_pxs = np.linspace(-3, 3, 10)
_pbase = ext.surface_from_cvs(3, 3, [(x + 60, y, _PF(x, y)) for x in _pxs for y in _pxs],
                              geom.clamped_uniform_knots(10, 3), geom.clamped_uniform_knots(10, 3))
def _pcurve(srf, a, b, z, n=12):
    Q = []
    for t in np.linspace(0, 1, n):
        x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        r_ = _smk.surface_closest_point(srf, (x, y, z(x, y)))
        r_ = r_[0] if isinstance(r_, tuple) else r_
        Q.append((r_.x, r_.y, r_.z))
    X, d_ = _interp(np.array(Q), 3)
    return ext.curve_from_cvs(d_, [tuple(map(float, x)) for x in X], geom.clamped_uniform_knots(len(X), d_), None)
_pent = [(60 + 1.3 * math.cos(a), 1.1 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 6)[:-1] + 0.3]
_zf = lambda x, y: _PF(x - 60, y)
_hole = [_pcurve(_pbase, _pent[i], _pent[(i + 1) % 5], _zf) for i in range(5)]
_cor = [(57.1, -2.9), (62.9, -2.9), (62.9, 2.9), (57.1, 2.9)]
_outer = [_pcurve(_pbase, _cor[i], _cor[(i + 1) % 4], _zf) for i in range(4)]
ptH = store.new_object(ctx, "PT_Holed", _smk.trim_surface_with_3d_curves(_pbase, [4, 5], _outer + _hole))
_hrefs = [f"E{i}" for i, pl in enumerate(store.display(ptH)["edges"])
          if np.linalg.norm(np.array(pl[len(pl) // 2])[:2] - (60, 0)) < 1.6]
r = bpy.ops.nproto.patch(inputs=ins(*[(ptH, x) for x in _hrefs]), default_cont="G1")
ptP = ctx.active_object
_pm = [_cl.measure(ptP, ptH, x, 1) for x in _hrefs]
check("patch: 5-sided hole in a curved surface -> one trimmed face, G1 on every edge",
      r == {'FINISHED'} and len(_hrefs) == 5 and store.get_brep(ptP).face_count() == 1 and not store.is_natural(ptP)
      and all(m_["ok"] and m_["angle"] < 0.05 for m_ in _pm), str([m_["text"] for m_ in _pm]))
_cl._cache.clear()
_pl = _cl.labels(ptP)
_r2 = bpy.ops.nproto.cycle_continuity(obj_name=ptP.name, index=0, step=-1)
_cl._cache.clear()
_pl2 = _cl.labels(ptP)
check("patch: one label per edge; clicking steps that edge G1 -> G0 and rebuilds",
      len(_pl) == 5 and all(L["value"] == "G1" for L in _pl) and _r2 == {'FINISHED'}
      and _pl2[0]["value"] == "G0" and features.history(ptP)["params"]["cont"][0] == "G0"
      and all(L["check"] and L["check"]["ok"] for L in _pl2), f"{[(L['value'], L['check'] and L['check']['text']) for L in _pl2]}")
# a window in a cylinder wall (about 100 degrees around)
_cr = 1.0
_cyl = ext.surface_from_cvs(3, 3, [(80 + _cr * math.cos(a), _cr * math.sin(a), z) for a in np.linspace(-1.6, 1.6, 9)
                                  for z in np.linspace(-2, 2, 6)], geom.clamped_uniform_knots(9, 3), geom.clamped_uniform_knots(6, 3))
def _ccurve(a0, z0, a1, z1, n=14):
    Q = []
    for t in np.linspace(0, 1, n):
        a, z = a0 + t * (a1 - a0), z0 + t * (z1 - z0)
        r_ = _smk.surface_closest_point(_cyl, (80 + _cr * math.cos(a), _cr * math.sin(a), z))
        r_ = r_[0] if isinstance(r_, tuple) else r_
        Q.append((r_.x, r_.y, r_.z))
    X, d_ = _interp(np.array(Q), 3)
    return ext.curve_from_cvs(d_, [tuple(map(float, x)) for x in X], geom.clamped_uniform_knots(len(X), d_), None)
_win = [(-0.85, -1), (0.85, -1.2), (0.95, 0.4), (0.0, 1.2), (-0.9, 0.6)]
_wc = [_ccurve(*_win[i], *_win[(i + 1) % 5]) for i in range(5)]
_bnd = [(-1.55, -1.9), (1.55, -1.9), (1.55, 1.9), (-1.55, 1.9)]
_bc = [_ccurve(*_bnd[i], *_bnd[(i + 1) % 4]) for i in range(4)]
ptC = store.new_object(ctx, "PT_Cyl", _smk.trim_surface_with_3d_curves(_cyl, [4, 5], _bc + _wc))
_crefs = [f"E{i}" for i, pl in enumerate(store.display(ptC)["edges"]) if abs(np.array(pl[len(pl) // 2])[2]) < 1.3
          and abs(math.atan2(pl[len(pl) // 2][1], pl[len(pl) // 2][0] - 80)) < 1.2]
r = bpy.ops.nproto.patch(inputs=ins(*[(ptC, x) for x in _crefs]), default_cont="G1")
ptPC = ctx.active_object
_pc = [_cl.measure(ptPC, ptC, x, 1) for x in _crefs]
_s = store.get_brep(ptPC).faces()[0].surface()
_dev = []
for a in (-0.4, 0.0, 0.4):
    for z in (-0.5, 0.0, 0.5):
        q_ = _smk.surface_closest_point(_cyl, (80 + math.cos(a), math.sin(a), z))
        q_ = q_[0] if isinstance(q_, tuple) else q_
        q_ = np.array([q_.x, q_.y, q_.z])
        _dev.append(float(np.linalg.norm(np.array(_s.evaluate(*_sqm._foot(_s, q_))) - q_)))
check("patch: window in a cylinder wall -> G1 all round, close to the wall inside",
      r == {'FINISHED'} and len(_crefs) == 5 and all(m_["ok"] for m_ in _pc) and max(_dev) < 0.06,
      f"{[m_['text'] for m_ in _pc]} dev {max(_dev):.4f}")

_cvP = np.array(ext.surface_cvs(_s)["points"]) if hasattr(ext, "surface_cvs") else None
_loopP = np.vstack([np.array([e.evaluate(t) for t in np.linspace(*e.parameter_range(), 20)]) for e in _wc])
_grow = (np.ptp(_cvP, axis=0).max() / np.ptp(_loopP, axis=0).max()) if _cvP is not None else 1.0
check("patch: the net stays close around the opening (four-sided base, no loose CVs)",
      _grow < 1.6 and _pt.info.get("method") == "four-sided", f"net / opening size {_grow:.2f}, {_pt.info}")

# a corner inside one of the four chains (the pentagon has 5) is laid out off
# the straight border, so the surface doesn't crease across from it
_raw = [_sqm.sample_side(ptH, x, 24) for x in _hrefs]
_ord, _ = _pt.chain(_raw)
_sds = [_raw[k].reversed() if rv else _raw[k] for k, rv in _ord]
_idx, _P5, _ = _pt.corners(_sds)
_lay = _pt.layout_quad(_sds, _idx, 0.5)
_starts = np.cumsum([0] + [len(sd.P) - 1 for sd in _sds[:-1]])
_free = [k for k, st in enumerate(_starts) if min(abs(int(st) - i) for i in _idx) > 2]
_off = [float(min(abs(_lay[k][0][0]), abs(1 - _lay[k][0][0]), abs(_lay[k][0][1]), abs(1 - _lay[k][0][1]))) for k in _free]
check("patch: a corner inside a side bulges off the border in (u, v) (no crease across the patch)",
      len(_free) == 1 and _off[0] > 0.03, f"corners {_idx}, free {_free}, off {_off}")

# a trimmed patch's display mesh keeps to Max Edge (the kernel leaves long
# fans where a trim crosses its cells; they shaded as facets)
_bb = store.get_brep(ptPC).bounding_box()
_diag = float(np.linalg.norm(np.array(_bb[1]) - np.array(_bb[0])))
_V = np.array([v.co for v in ptPC.data.vertices])
_me = max(max(np.linalg.norm(_V[pl.vertices[i]] - _V[pl.vertices[(i + 1) % len(pl.vertices)]]) for i in range(len(pl.vertices)))
          for pl in ptPC.data.polygons)
check("patch: display mesh refined to Max Edge on the trimmed face", _me <= 1.01 * store.max_edge_setting(ctx) * _diag,
      f"longest edge {_me:.4f}, limit {store.max_edge_setting(ctx) * _diag:.4f}, {len(ptPC.data.polygons)} tris")

# G2: the 5-sided hole in the freeform surface, curvature-continuous all round
r = bpy.ops.nproto.patch(inputs=ins(*[(ptH, x) for x in _hrefs]), default_cont="G2")
ptG2 = ctx.active_object
_g2m = [_cl.measure(ptG2, ptH, x, 2) for x in _hrefs]
check("patch: G2 on every edge of the 5-sided hole holds (gap, angle, curvature)",
      r == {'FINISHED'} and all(m_["ok"] for m_ in _g2m), str([m_["text"] for m_ in _g2m]))
_cl._cache.clear()
_l2 = _cl.labels(ptG2)
check("patch: labels offer G2", all(L["values"] == ["G0", "G1", "G2"] and L["value"] == "G2" for L in _l2),
      str([(L["value"], L["values"]) for L in _l2]))

# G2 on a strongly curved neighbour (the cylinder window): the cross
# direction stays pinned in the Gauss-Newton model, so curvature follows
r = bpy.ops.nproto.patch(inputs=ins(*[(ptC, x) for x in _crefs]), default_cont="G2")
ptCG2 = ctx.active_object
_cg2 = [_cl.measure(ptCG2, ptC, x, 2) for x in _crefs]
check("patch: G2 on the cylinder window: every edge within 10% curvature and 0.15 deg",
      r == {'FINISHED'} and all(m_["curv"] < 0.10 and m_["angle"] < 0.15 for m_ in _cg2)
      and sum(m_["ok"] for m_ in _cg2) >= 4, str([m_["text"] for m_ in _cg2]))

# the CV net itself runs evenly (net fairness on the Greville differences)
def _net_rough(obj):
    d_ = ext.surface_cvs(store.get_brep(obj).faces()[0].surface())
    G_ = np.array(d_["points"]).reshape(d_["count_u"], d_["count_v"], 3)
    gr = lambda U_, p_, n_: np.array([sum(U_[i + 1:i + p_ + 1]) / p_ for i in range(n_)])
    out_ = []
    for ax, g_ in ((0, gr(d_["knots_u"], d_["degree_u"], d_["count_u"])), (1, gr(d_["knots_v"], d_["degree_v"], d_["count_v"]))):
        X_ = np.moveaxis(G_, ax, 0)
        h_ = np.diff(g_)
        s_ = np.diff(X_, axis=0) / h_[:, None, None]
        dd_ = np.diff(s_, axis=0) / ((h_[1:] + h_[:-1]) / 2)[:, None, None]
        out_.append(np.linalg.norm(np.diff(dd_, axis=0), axis=2).ravel())
    return float(np.median(np.concatenate(out_))) / float(np.linalg.norm(np.ptp(G_.reshape(-1, 3), axis=0)))
_orig_sq = _pt.solve_quad
_rough = []
for _kw in ({"cw": 0.0, "geo": 0}, {"geo": 0}):
    _pt.solve_quad = lambda *a, _k=_kw, **k: _orig_sq(*a, **dict(k, **_k))
    r = bpy.ops.nproto.patch(inputs=ins(*[(ptC, x) for x in _crefs]), default_cont="G1")
    _rough.append(_net_rough(ctx.active_object))
_pt.solve_quad = _orig_sq
r = bpy.ops.nproto.patch(inputs=ins(*[(ptC, x) for x in _crefs]), default_cont="G1")
_rough.append(_net_rough(ctx.active_object))
_pcn = [_cl.measure(ctx.active_object, ptC, x, 1) for x in _crefs]
check("patch: the CV net runs evenly: net fairness evens the (u, v)-faired net, the surface-faired one "
      "is as even again (edges hold)",
      _rough[1] < 0.8 * _rough[0] and _rough[2] <= _rough[1] and all(m_["ok"] for m_ in _pcn),
      "rough plain / + net fairness / + surface passes " + " / ".join(f"{x:.2f}" for x in _rough))

# zebra mesh of a trimmed patch: no folded triangles, and normals bend at most
# a little more than the zebra's angle tolerance across any edge
from blender_nurbs import diagnostics as _dg
_zP, _zN, _zT = _dg.zebra_mesh(ptPC, 2.0, 0.0005)
_zP = np.asarray(_zP, float); _zN = np.asarray(_zN, float)
_zN /= np.linalg.norm(_zN, axis=1)[:, None]
_fn = np.cross(_zP[_zT[:, 1]] - _zP[_zT[:, 0]], _zP[_zT[:, 2]] - _zP[_zT[:, 0]])
_ar = np.linalg.norm(_fn, axis=1)
_nv = _zN[_zT[:, 0]] + _zN[_zT[:, 1]] + _zN[_zT[:, 2]]
_ok = _ar > 1e-2 * np.median(_ar)          # slivers along the trim edge are invisible
_cos = np.sum(_fn * _nv, axis=1)[_ok] / (_ar[_ok] * np.linalg.norm(_nv, axis=1)[_ok])
_eang = np.degrees(np.arccos(np.clip(np.concatenate([np.sum(_zN[_zT[:, i]] * _zN[_zT[:, j]], axis=1)
                                                     for i, j in ((0, 1), (1, 2), (2, 0))]), -1, 1)))
check("zebra mesh: no folded triangles, normals within the angle tolerance across edges",
      int(np.sum(_cos < 0.3)) == 0 and float(np.percentile(_eang, 99.9)) < 2.5,
      f"folded {int(np.sum(_cos < 0.3))}, edge normal angle p99.9 {np.percentile(_eang, 99.9):.2f} deg, {len(_zT)} tris")

# crack repair: open edges of the fine zebra mesh lie on the B-rep's edges
# only, and a real trimmed hole stays open
def _open_edges(T_):
    c_ = {}
    for t in T_:
        t = [int(x) for x in t]
        for k in range(len(t)):
            a_, b_ = t[k], t[(k + 1) % len(t)]
            key = (min(a_, b_), max(a_, b_))
            c_[key] = c_.get(key, 0) + 1
    return [k for k, v in c_.items() if v == 1]
_zP, _zN, _zT = _dg.zebra_mesh(ptCG2, 1.0, 0.0005)
_zP = np.asarray(_zP, float)
_E = store._edge_samples(store.get_brep(ptCG2), 400)
_op = _open_edges(_zT)
_far = [e for e in _op if np.min(np.linalg.norm(_E - (_zP[e[0]] + _zP[e[1]]) / 2, axis=1)) > 0.01]
_hl = _open_edges(store._tessellate(store.get_brep(ptH), *store.tess_settings(), store.edge_angle_setting(), store.max_edge_setting())[2])
_hP = np.asarray(store._tessellate(store.get_brep(ptH), *store.tess_settings(), store.edge_angle_setting(), store.max_edge_setting())[0], float)
_near_hole = [e for e in _hl if np.linalg.norm(((_hP[e[0]] + _hP[e[1]]) / 2)[:2] - (60, 0)) < 1.6]
check("tessellation: no cracks inside a face, a real trimmed hole stays open",
      len(_far) <= 0.01 * len(_op) and len(_near_hole) >= 10,
      f"open edges off the B-rep edges {len(_far)} of {len(_op)}; hole rim edges {len(_near_hole)}")

# corner conflict: a square opening between four flat strips, one strip
# tilted 3 degrees about its own edge -- its tangent plane can't contain the
# neighbouring edges, so G1 there is eased toward both its corners (orange
# label) instead of squashing the patch flat at the corner (a fold)
def _strip(name, a, b, out, k=1):
    a, b = np.array(a, float), np.array(b, float)
    o_ = np.array(out, float)
    pts = [tuple(a), tuple(a + o_), tuple(b), tuple(b + o_)]
    return store.new_object(ctx, name, ext.surface_from_cvs(1, 1, pts, [0, 0, 1, 1], [0, 0, 1, 1]))
_th = math.radians(3.0)
_X0 = 200.0
_sq = [(_X0, 0, 0), (_X0 + 4, 0, 0), (_X0 + 4, 4, 0), (_X0, 4, 0)]
_strips = [_strip("CC_A", _sq[0], _sq[1], (0, -2 * math.cos(_th), 2 * math.sin(_th))),
           _strip("CC_B", _sq[1], _sq[2], (2, 0, 0)),
           _strip("CC_C", _sq[2], _sq[3], (0, 2, 0)),
           _strip("CC_D", _sq[3], _sq[0], (-2, 0, 0))]
def _inner_edge(o_, a, b):
    m_ = (np.array(a) + np.array(b)) / 2
    eds = store.display(o_)["edges"]
    return f"E{min(range(len(eds)), key=lambda i: np.linalg.norm(np.array(eds[i][len(eds[i]) // 2]) - m_))}"
_cin = [(o_, _inner_edge(o_, _sq[i], _sq[(i + 1) % 4])) for i, o_ in enumerate(_strips)]
_cc_ref = {}
for o_, r_ in _cin:
    _cc_ref[o_.name] = r_
_ccn = [_strips[i].name for i in range(4)]
r = bpy.ops.nproto.patch(inputs=ins(*_cin), default_cont="G1")
ptCC = ctx.active_object
_cl._cache.clear()
_ccl = _cl.labels(ptCC)
_s = store.get_brep(ptCC).faces()[0].surface()
(_u0, _u1), (_v0, _v1) = _s.u_domain(), _s.v_domain()
_J = np.array([np.linalg.norm(np.cross(*_s.derivatives(u_, v_)[1:])) for u_ in np.linspace(_u0, _u1, 40)
               for v_ in np.linspace(_v0, _v1, 40)])
_byname = {L["name"].split()[0]: L for L in _ccl}
check("patch: a corner no surface can be G1 to is eased (orange), the other edges hold, no fold",
      r == {'FINISHED'} and _byname["CC_A"]["check"].get("eased") and not _byname["CC_A"]["check"]["ok"]
      and all(_byname[n_]["check"]["ok"] for n_ in ("CC_B", "CC_C", "CC_D")) and _J.min() / np.median(_J) > 0.1,
      f"{[(n_, _byname[n_]['check']['text'][:70]) for n_ in _ccn]} jmin {_J.min() / np.median(_J):.3f}")
_mk = _cl.corner_marks(ptCC)
_mkp = sorted([tuple(np.round(np.array(m_["pos"]), 3)) for m_ in _mk])
check("patch: a square marker on each eased corner, with the angle",
      len(_mk) == 2 and _mkp == sorted([tuple(map(float, _sq[0])), tuple(map(float, _sq[1]))])
      and all(abs(m_["deg"] - 3.0) < 0.1 for m_ in _mk), str([(p_, round(m_["deg"], 2)) for p_, m_ in zip(_mkp, _mk)]))

# --- Patch robustness: openings of many kinds ------------------------------
from mathutils import Matrix as _Mx
_RF = {}
def _rb_base(kind, X0):
    xs = np.linspace(-3, 3, 10)
    f = {"free": lambda x, y: 0.25 * math.sin(0.9 * x) * math.cos(0.7 * y) + 0.05 * x * y,
         "flat": lambda x, y: 0.0,
         "dome": lambda x, y: 0.6 - 0.07 * (x * x + y * y)}[kind]
    U_ = geom.clamped_uniform_knots(10, 3)
    srf = ext.surface_from_cvs(3, 3, [(x + X0, y, f(x, y)) for x in xs for y in xs], U_, U_)
    _RF[id(srf)] = (f, X0)
    return srf
def _rb_on(srf, xy):
    f, X0 = _RF[id(srf)]
    Q = []
    for x, y in xy:
        r_ = _smk.surface_closest_point(srf, (x, y, f(x - X0, y)))
        r_ = r_[0] if isinstance(r_, tuple) else r_
        Q.append((r_.x, r_.y, r_.z))
    X_, d_ = _interp(np.array(Q), 3)
    return ext.curve_from_cvs(d_, [tuple(map(float, x)) for x in X_], geom.clamped_uniform_knots(len(X_), d_), None)
_rb_seg = lambda a, b, n=14: [(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for t in np.linspace(0, 1, n)]
_rb_arc = lambda c, r, a0, a1, n=16: [(c[0] + r * math.cos(a), c[1] + r * math.sin(a)) for a in np.linspace(a0, a1, n)]
_rb_poly = lambda pts: [_rb_seg(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
def _rb_holed(name, kind, X0, pieces, M=None):
    srf = _rb_base(kind, X0)
    cor = [(X0 - 2.9, -2.9), (X0 + 2.9, -2.9), (X0 + 2.9, 2.9), (X0 - 2.9, 2.9)]
    outer = [_rb_on(srf, _rb_seg(cor[i], cor[(i + 1) % 4])) for i in range(4)]
    hole = [_rb_on(srf, [(x + X0, y) for x, y in pc]) for pc in pieces]
    o_ = store.new_object(ctx, name, _smk.trim_surface_with_3d_curves(srf, [4, len(hole)], outer + hole))
    eds = store.display(o_)["edges"]
    refs = [f"E{i}" for i, pl in enumerate(eds) if pl and max(abs(pl[len(pl) // 2][0] - X0), abs(pl[len(pl) // 2][1])) < 2.4]
    if M is not None:
        o_.matrix_world = M
    return o_, refs, srf
def _rb_jmin(o_):
    s_ = store.get_brep(o_).faces()[0].surface()
    (a0, a1), (b0, b1) = s_.u_domain(), s_.v_domain()
    J_ = np.array([np.linalg.norm(np.cross(*s_.derivatives(u_, v_)[1:])) for u_ in np.linspace(a0, a1, 30)
                   for v_ in np.linspace(b0, b1, 30)])
    return J_.min() / np.median(J_)
def _rb_dev(o_, srf, X0, pieces):
    pts = np.vstack([np.array(pc) for pc in pieces]); c_ = pts.mean(axis=0)
    s_ = store.get_brep(o_).faces()[0].surface(); f, _ = _RF[id(srf)]; d_ = 0.0
    for fr in (0.0, 0.3, 0.6):
        for p_ in pts[::3]:
            q_ = c_ + fr * (p_ - c_)
            r_ = _smk.surface_closest_point(srf, (q_[0] + X0, q_[1], f(q_[0], q_[1])))
            r_ = r_[0] if isinstance(r_, tuple) else r_
            P_ = np.array([r_.x, r_.y, r_.z])
            d_ = max(d_, float(np.linalg.norm(np.array(s_.evaluate(*_sqm._foot(s_, P_))) - P_)))
    return d_
_Lsh = [(-1.2, -1.2), (1.2, -1.2), (1.2, 0.0), (0.0, 0.0), (0.0, 1.2), (-1.2, 1.2)]
_star = [((1.4 if i % 2 == 0 else 0.7) * math.cos(a), (1.4 if i % 2 == 0 else 0.7) * math.sin(a))
         for i, a in enumerate(np.linspace(0, 2 * math.pi, 11)[:-1])]
_RB = [  # name, kind, pieces, continuity, max deviation from the surface the hole was cut from, matrix
    ("triangle", "free", _rb_poly([(-1.3, -1.0), (1.4, -0.8), (0.1, 1.3)]), "G2", 0.02, None),
    ("L_flat", "flat", _rb_poly(_Lsh), "G1", 1e-4, None),
    ("L_free", "free", _rb_poly(_Lsh), "G2", 0.02, None),
    ("star", "dome", _rb_poly(_star), "G1", 0.02, None),
    ("circle2", "free", [_rb_arc((0, 0), 1.2, 0, math.pi), _rb_arc((0, 0), 1.2, math.pi, 2 * math.pi)], "G2", 0.02, None),
    ("L_xform", "free", _rb_poly(_Lsh), "G2", None,
     _Mx.Translation((5, -3, 2)) @ _Mx.Rotation(0.5, 4, 'X') @ _Mx.Rotation(0.4, 4, 'Z') @ _Mx.Scale(2.0, 4)),
]
_rb_out = []
for _i, (_nm, _kd, _pc, _cn, _dmax, _M) in enumerate(_RB):
    _X0 = 300.0 + 10.0 * _i
    _o, _rf, _sf = _rb_holed("RB_" + _nm, _kd, _X0, _pc, _M)
    try:
        r = bpy.ops.nproto.patch(inputs=ins(*[(_o, x) for x in _rf]), default_cont=_cn)
    except Exception as exc:
        _rb_out.append((_nm, "EXC " + str(exc)[:80])); continue
    _P = ctx.active_object
    _cl._cache.clear()
    _bad = [L["check"]["text"][:40] for L in _cl.labels(_P) if L["check"] and not L["check"]["ok"]]
    _jm = _rb_jmin(_P)
    _dv = _rb_dev(_P, _sf, _X0, _pc) if _dmax is not None else 0.0
    _rb_out.append((_nm, r == {'FINISHED'} and not _bad and _jm > 0.1 and (_dmax is None or _dv < _dmax),
                    len(_rf), _bad[:2], round(_jm, 3), round(_dv, 4)))
check("patch robustness: triangle, concave L (flat / curved / transformed), 5-point star, two-arc circle "
      "-- every edge holds, no fold, close to the surface the hole was cut from",
      all(x[1] is True for x in _rb_out), str(_rb_out))
_cv5 = []
for _i, (_a, _b) in enumerate(zip([(400 + 1.3 * math.cos(t), 1.3 * math.sin(t), 0.3 * math.sin(2 * t))
                                   for t in np.linspace(0, 2 * math.pi, 6)[:-1]], [None] * 5)):
    _cv5.append(_a)
_cvo = [nurbs(f"RB_cv{i}", [_cv5[i], _cv5[(i + 1) % 5]], order=2) for i in range(5)]
r = bpy.ops.nproto.patch(inputs=ins(*[(o_, "S0") for o_ in _cvo]))
check("patch robustness: five plain curves in space (G0)", r == {'FINISHED'} and _rb_jmin(ctx.active_object) > 0.1,
      f"jmin {_rb_jmin(ctx.active_object):.3f}")

# fairness measured on the surface (not in (u, v)): where the layout bends
# the parameter lines, the (u, v) thin plate left ridges -- curvature varies
# far less inside with the surface passes
def _pt_hvar(G, U, p, loop, m=160):
    """int |grad H|^2 dA (times the loop's size^2) inside the loop, a little
    off it."""
    nu, nv = G.shape[:2]
    X = G.reshape(-1, 3)
    uv = _pt._project(X, _pt._grid_uv(X, (nu, nv), p, U, loop), loop, (nu, nv), p, U, it=4)
    t = np.linspace(0.0, 1.0, m)
    def B(n_, U_):
        out = np.zeros((3, m, n_))
        for a, s in enumerate(t):
            sp = geom.find_span(n_ - 1, p, float(s), U_)
            out[:, a, sp - p:sp + 1] = _pt.ders_basis(sp, float(s), p, U_, 2)
        return out
    Bu, Bv = B(nu, U[0]), B(nv, U[1])
    S = lambda a, b: np.einsum("ai,ijc,bj->abc", Bu[a], G, Bv[b])
    Su, Sv, Suu, Suv, Svv = S(1, 0), S(0, 1), S(2, 0), S(1, 1), S(0, 2)
    N = np.cross(Su, Sv)
    N /= np.linalg.norm(N, axis=2, keepdims=True)
    E, F, Gg = (Su * Su).sum(2), (Su * Sv).sum(2), (Sv * Sv).sum(2)
    det = E * Gg - F * F
    H = (E * (Svv * N).sum(2) - 2 * F * (Suv * N).sum(2) + Gg * (Suu * N).sum(2)) / (2 * det)
    Hu, Hv = np.gradient(H, t, t)
    g2 = (Gg * Hu * Hu - 2 * F * Hu * Hv + E * Hv * Hv) / det
    UU, VV = np.meshgrid(t, t, indexing="ij")
    x, y = UU.ravel(), VV.ravel()
    msk = np.zeros(len(x), bool)
    px, py = uv[:, 0], uv[:, 1]
    for a, b, c, d in zip(px, py, np.roll(px, -1), np.roll(py, -1)):
        msk ^= ((b > y) != (d > y)) & (x < a + (y - b) * (c - a) / ((d - b) if d != b else 1e-30))
    msk = msk.reshape(m, m)
    for _ in range(int(m * 0.06)):
        msk[1:-1, 1:-1] &= msk[:-2, 1:-1] & msk[2:, 1:-1] & msk[1:-1, :-2] & msk[1:-1, 2:]
    return float((g2 * np.sqrt(det) * (t[1] - t[0]) ** 2)[msk].sum()) * float(np.linalg.norm(np.ptp(loop, axis=0))) ** 2
_hv = {}
_hv_objs = [("pentagon G2", ptG2), ("cylinder window G2", ptCG2)] + \
    [(nm, ob) for nm in ("L_free", "star") for ob in bpy.data.objects
     if (features.history(ob) or {}).get("params", {}).get("pt_method") == "QUAD"
     and features.history(ob)["inputs"] and features.history(ob)["inputs"][0][0].name == "RB_" + nm]
for _nm, _ob in _hv_objs:
    _h = features.history(_ob)
    _sds, _ord, _, _, _ = _pt.loop_sides(_h["inputs"], _h["params"])
    _sc = [_pt.CONT.get(_h["params"]["cont"][k], 1) for k, _ in _ord]
    _ix = _pt.corners(_sds, fixed=_pt.corner_loop_indices(_sds, _ord, []))[0]
    _lp = np.vstack([sd.P for sd in _sds])
    _hv[_nm] = []
    for _g in (0, _pt.GEO_PASSES):
        _rp = {"check": True}
        _G, _U, _ = _pt.solve_quad(_sds, _sc, n=(24, 24), p=5, fair=1.0, idx=_ix, geo=_g, report=_rp)
        _hv[_nm].append((_pt_hvar(_G, _U, 5, _lp), max(e[0] for e in _rp["errors"])))
check("patch: fairness on the surface -- curvature varies far less inside than with the (u, v) thin plate "
      "(pentagon, cylinder window, L, star; under 0.6x), edges as close",
      len(_hv) == 4 and all(v[1][0] < 0.6 * v[0][0] and v[1][1] < max(2 * v[0][1], 5e-5) for v in _hv.values()),
      str({k: [(round(a, 3), f"{b:.1e}") for a, b in v] for k, v in _hv.items()}))

# --- Patch solver speed-ups: the vectorised basis is the scalar one to the
# last bit, and the block solve's premise -- nothing couples CV rows more
# than the degree apart -- holds for every system a Patch solves
_rng = np.random.default_rng(7)
_bd = 0.0
for _p in (1, 2, 3, 5):
    for _n in (_p + 1, _p + 5, 24):
        _Uk = geom.clamped_uniform_knots(_n, _p)
        _ts = np.concatenate([_rng.random(60), np.unique(_Uk)])
        _sp = _pt.find_spans(_n - 1, _p, _ts, _Uk)
        _bd = max(_bd, float(np.abs(_sp - [geom.find_span(_n - 1, _p, float(t), _Uk) for t in _ts]).max()))
        _D = _pt.ders_basis_many(_sp, _ts, _p, _Uk, min(3, _p))
        _bd = max(_bd, max(float(np.abs(_D[k] - _pt.ders_basis(int(s), float(t), _p, _Uk, min(3, _p))).max())
                           for k, (s, t) in enumerate(zip(_sp, _ts))))
_U2 = (geom.clamped_uniform_knots(9, 3), geom.clamped_uniform_knots(12, 3))
_uvs = _rng.random((30, 2))
_bd = max(_bd, float(np.abs(_pt.surface_rows_many(_uvs, (9, 12), 3, _U2) -
                            np.array([_pt.surface_rows(float(a), float(b), (9, 12), 3, _U2) for a, b in _uvs])).max()))
_Su, _Sv, _r = _rng.normal(size=(20, 3)), _rng.normal(size=(20, 3)), _rng.normal(size=(20, 3))
_Sv[0] = 2 * _Su[0]                                   # a degenerate pair: lstsq's minimum-norm answer
_l2 = float(np.abs(_pt.lstsq2(_Su, _Sv, _r) - np.array([np.linalg.lstsq(np.c_[a, b], c, rcond=None)[0]
                                                        for a, b, c in zip(_Su, _Sv, _r)])).max())
check("patch solver: vectorised spans / basis / rows equal the scalar ones exactly, lstsq2 = lstsq",
      _bd == 0.0 and _l2 < 1e-12, f"basis {_bd}, lstsq2 {_l2:.1e}")
_h = features.history(ptG2)
_sds, _ord, _, _, _ = _pt.loop_sides(_h["inputs"], _h["params"])
_sc = [_pt.CONT.get(_h["params"]["cont"][k], 1) for k, _ in _ord]
_ix = _pt.corners(_sds, fixed=_pt.corner_loop_indices(_sds, _ord, []))[0]
_real_ns = _pt.net_solve
_seen = []
def _ns_spy(Q, b, nu, p):
    nv = len(Q) // (3 * nu)
    R = (Q.reshape(nu, 3 * nv, nu, 3 * nv) != 0).any(axis=(1, 3))
    _i, _j = np.nonzero(R)
    x = _real_ns(Q, b, nu, p)
    _seen.append((int(np.abs(_i - _j).max()), max(p, 2),
                  float(np.abs(x - np.linalg.solve(Q, b)).max() / np.abs(np.linalg.solve(Q, b)).max())))
    return x
_pt.net_solve = _ns_spy
try:
    for _n2, _pp, _kw in (((24, 24), 5, {}), ((20, 14), 3, {"tension": [1.8] + [1.0] * (len(_sds) - 1)})):
        _pt.solve_quad(_sds, _sc, n=_n2, p=_pp, fair=1.0, idx=_ix, **_kw)
finally:
    _pt.net_solve = _real_ns
check("patch solver: every system is zero between CV rows more than the degree apart (degree 5 and 3, "
      "tension), and the block solve gives the dense solve's answer",
      len(_seen) >= 8 and all(bw <= lim and d < 1e-7 for bw, lim, d in _seen),
      str([(bw, lim, f"{d:.0e}") for bw, lim, d in _seen]))

# --- Patch: separate U / V counts, lightest net, tension, corners by hand, guide curves
from blender_nurbs import feature_ui as _fui
def _pt_net(o_):
    d_ = _smk.blender_ext.surface_cvs(store.get_brep(o_).faces()[0].surface())
    return d_["count_u"], d_["count_v"], np.array(d_["points"], float)
def _pt_bad(o_):
    _cl._cache.clear()
    return [L["name"] + " " + L["check"]["text"][:40] for L in _cl.labels(o_)
            if L["check"] and not L["check"]["ok"] and not L["check"].get("eased")]
_o, _rf, _sf = _rb_holed("PF_L", "free", 400.0, _rb_poly(_Lsh))
_loop = [(_o, x) for x in _rf]
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1", pt_cvs=16, pt_cvs_v=10)
_P = ctx.active_object
check("patch: separate CV counts U / V (16 x 10), every edge holds",
      r == {'FINISHED'} and _pt_net(_P)[:2] == (16, 10) and not _pt_bad(_P), str((_pt_net(_P)[:2], _pt_bad(_P))))
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G2", pt_auto_cvs=True)
_P = ctx.active_object
_nu, _nv, _ = _pt_net(_P)
check("patch: Lightest -- far fewer CVs than 24 x 24, every edge still holds G2, no fold",
      r == {'FINISHED'} and _nu * _nv <= 16 * 16 and not _pt_bad(_P) and _rb_jmin(_P) > 0.1,
      str((_nu, _nv, _pt_bad(_P), round(_rb_jmin(_P), 3))))
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1")
_P = ctx.active_object
_, _, _G0 = _pt_net(_P)
_size = float(np.linalg.norm(np.ptp(_G0, axis=0)))
_fui.sync_from_history(_P)
_pg = _P.nproto_feature
_moves = []
for _t in (1.8, 0.6):
    _pg.pt_edges[0].tension = _t
    _moves.append((_t, round(float(np.abs(_pt_net(_P)[2] - _G0).max()) / _size, 4), _pt_bad(_P)))
_pg.pt_edges[0].tension = 1.0
_back = float(np.abs(_pt_net(_P)[2] - _G0).max())
check("patch: tension per edge (panel) reshapes the patch, every edge keeps G1; back to 1 = the plain patch",
      all(m_ > 0.005 and not b_ for _, m_, b_ in _moves) and _back < 1e-9
      and "pt_tension" not in features.history(_P)["params"], str((_moves, _back)))
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G2", pt_tension_all=1.6)
check("patch: tension on every edge (operator) with G2, every edge holds",
      r == {'FINISHED'} and not _pt_bad(ctx.active_object), str(_pt_bad(ctx.active_object)))
# fairness on the surface: on by default (not stored); off in the operator =
# the (u, v) fairness (stored); the Feature panel switches it back on
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G2", pt_surface_fair=False)
_Ps = ctx.active_object
_, _, _Goff = _pt_net(_Ps)
_fui.sync_from_history(_Ps)
_pgs = _Ps.nproto_feature
_off_ok = (r == {'FINISHED'} and features.history(_Ps)["params"].get("pt_surface_fair") is False
           and _pgs.pt_surface_fair is False and not _pt_bad(_Ps))
_pgs.pt_surface_fair = True
_, _, _Gon = _pt_net(_Ps)
_on_ok = features.history(_Ps)["params"].get("pt_surface_fair") is True and not _pt_bad(_Ps)
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G2")
_Pd = ctx.active_object
_, _, _Gdef = _pt_net(_Pd)
_fui.sync_from_history(_Pd)
_sw = (float(np.abs(_Gon - _Goff).max()) / _size, float(np.abs(_Gon - _Gdef).max()))
check("patch: 'On Surface' fairness -- on by default (not stored), off in the operator = (u, v) fairness (stored), "
      "the Feature panel switches it back on (= the default patch), edges hold",
      _off_ok and _on_ok and "pt_surface_fair" not in features.history(_Pd)["params"] and _Pd.nproto_feature.pt_surface_fair
      and _sw[0] > 1e-3 and _sw[1] < 1e-9, str((_off_ok, _on_ok, _sw)))
# corners by hand: click a junction that isn't a corner -> the nearest corner moves there
_cps = _pt.corner_points(_P)
_j = next(i for i, c_ in enumerate(_cps) if not c_["corner"])
_jk = (_cps[_j]["k"], _cps[_j]["f"])
r = bpy.ops.nproto.patch_corner(obj_name=_P.name, index=_j)
_pc_ = features.history(_P)["params"].get("pt_corners")
_now = [(c_["k"], c_["f"]) for c_ in _pt.corner_points(_P) if c_["corner"]]
_moved = float(np.abs(_pt_net(_P)[2] - _G0).max())
r2 = bpy.ops.nproto.patch_corner(obj_name=_P.name, index=0, reset=True)
check("patch: a click on a junction makes it a corner (stored, edges hold); Alt+click back to automatic",
      r == {'FINISHED'} and _pc_ and len(_pc_) == 4 and _jk in _now and _moved > 1e-4 and not _pt_bad(_P)
      and r2 == {'FINISHED'} and "pt_corners" not in features.history(_P)["params"]
      and float(np.abs(_pt_net(_P)[2] - _G0).max()) < 1e-9, str((_pc_, _now, _jk, _moved)))
# guide curves: one inside the opening, lifted 0.12 above the surface the hole was cut from
_f, _ = _RF[id(_sf)]
_gp = []
for _k in range(13):
    _x, _y = -0.9 + 0.6 * _k / 12, -0.6 + 1.2 * _k / 12
    _q = _smk.surface_closest_point(_sf, (_x + 400.0, _y, _f(_x, _y)))
    _q = _q[0] if isinstance(_q, tuple) else _q
    _gp.append((_q.x, _q.y, _q.z + 0.12 * math.sin(math.pi * _k / 12) ** 2))
_go = nurbs("PF_guide", _gp)
ctx.view_layer.update()
r = bpy.ops.nproto.patch(inputs=ins(*([(_go, "S0")] + _loop)), default_cont="G1")   # guide picked first
_P = ctx.active_object
_h = features.history(_P)
_gl = [L for L in _cl.labels(_P) if L.get("guide")]
check("patch: a curve inside the opening is a guide -- the patch passes through it, every edge holds",
      r == {'FINISHED'} and _h["params"].get("pt_guides") == [0] and _gl and _gl[0]["check"]["ok"]
      and not _pt_bad(_P) and _pt.patch_info(_P).get("guide_dev", 1.0) < 1e-3,
      str((_h["params"].get("pt_guides"), _pt.patch_info(_P).get("guide_dev"), _pt_bad(_P))))
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1")
_P = ctx.active_object
ctx.view_layer.objects.active = _P
r = bpy.ops.nproto.patch_add_guides(target=_P.name, inputs=ins((_go, "S0")))
_ok1 = r == {'FINISHED'} and features.history(_P)["params"].get("pt_guides") == [6] and not _pt_bad(_P)
r = bpy.ops.nproto.patch_remove_guide(index=6)
check("patch: guides added afterwards and removed again (back to the plain patch)",
      _ok1 and r == {'FINISHED'} and not features.history(_P)["params"].get("pt_guides")
      and len(features.history(_P)["inputs"]) == 6 and float(np.abs(_pt_net(_P)[2] - _G0).max()) < 1e-9,
      str(features.history(_P)["params"]))
# which picks are guides: dangling ones; a guide from corner to corner by pick order
_diag = nurbs("PF_diag", [_cps[0]["pos"][:], _cps[2]["pos"][:]], order=2)
check("patch: guides told from the loop (dangling, or picked after the loop closed)",
      _pt.split_loop([(_go, "S0")] + _loop) == [0] and _pt.split_loop(_loop) == []
      and _pt.split_loop(_loop + [(_diag, "S0")]) == [6],
      str((_pt.split_loop([(_go, "S0")] + _loop), _pt.split_loop(_loop + [(_diag, "S0")]))))

# --- Pause Rebuild: settings wait, Rebuild builds them; per-edge rows; "All edges" row; operator's edge rows
from blender_nurbs import history as _hist
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1")
_P = ctx.active_object
_G1n = _pt_net(_P)[2]
ctx.scene.nproto_settings.pause_rebuild = True
_cl._cache.clear()
_li = next(i for i, L in enumerate(_cl.labels(_P)) if L["param"] == "pc:0")
r1 = bpy.ops.nproto.cycle_continuity(obj_name=_P.name, index=_li)          # G1 -> G2, held back
_pg = _P.nproto_feature
_pg.pt_edges[1].cont = "G0"                                                 # panel row, held back
_pg.pt_edges[2].tension = 1.5
_held_same = float(np.abs(_pt_net(_P)[2] - _G1n).max()) < 1e-12
_cl._cache.clear()
_unchk = all(L["check"] is None and L["pending"] for L in _cl.labels(_P))
_wt = sorted(_hist.waiting(ctx.scene))
_hp = features.history(_P)["params"]
r2 = bpy.ops.nproto.rebuild_waiting()
_moved = float(np.abs(_pt_net(_P)[2] - _G1n).max())
check("pause rebuild: label click and panel edits only change the settings (geometry unchanged, labels "
      "unchecked); Rebuild builds them and every edge holds",
      r1 == {'FINISHED'} and _held_same and _unchk and _wt == [_P.name]
      and _hp["cont"][:3] == ["G2", "G0", "G1"] and abs(_hp["pt_tension"][2] - 1.5) < 1e-9
      and r2 == {'FINISHED'} and not features.pending(_P) and _moved > 1e-4 and not _pt_bad(_P)
      and not _hist.waiting(ctx.scene), str((_held_same, _unchk, _wt, _hp.get("cont"), _moved, _pt_bad(_P))))
_pg.pt_edges[1].cont = "G1"
_pg.pt_edges[2].tension = 1.0
_cl._cache.clear()
_li = next(i for i, L in enumerate(_cl.labels(_P)) if L["param"] == "pc:0")
bpy.ops.nproto.cycle_continuity(obj_name=_P.name, index=_li, step=-1)      # G2 -> G1 (Ctrl+click)
_w2 = features.pending(_P)
ctx.scene.nproto_settings.pause_rebuild = False                              # switching off builds
check("pause rebuild: Ctrl+click steps back; switching the pause off builds what waited (= the plain G1 patch)",
      _w2 and not features.pending(_P) and float(np.abs(_pt_net(_P)[2] - _G1n).max()) < 1e-9
      and features.history(_P)["params"]["cont"][:3] == ["G1", "G1", "G1"], str(features.history(_P)["params"]))
# "All edges" row in the Feature panel
_fui.sync_from_history(_P)
_all0 = _pg.pt_cont_all
_pg.pt_cont_all = "G2"
_hp = features.history(_P)["params"]
_ok_all = _hp.get("pt_all") == "G2" and set(_hp["cont"]) == {"G2"} and all(e.cont == "G2" for e in _pg.pt_edges) \
    and not _pt_bad(_P)
_pg.pt_edges[3].cont = "G1"
_mixed = _pg.pt_cont_all == "MIXED" and "pt_all" not in features.history(_P)["params"]
_pg.pt_tension_all = 1.3
_tall = np.allclose(features.history(_P)["params"].get("pt_tension", [0]), [1.3] * 6, atol=1e-6) and all(abs(e.tension - 1.3) < 1e-6 for e in _pg.pt_edges)
_pg.pt_tension_all = 1.0
check("patch panel: 'All edges' sets every edge (stored), an edge set by itself makes it Mixed; all tensions at once",
      _all0 == "G1" and _ok_all and _mixed and _tall and "pt_tension" not in features.history(_P)["params"],
      str((_all0, _ok_all, _mixed, _tall, features.history(_P)["params"])))
# the operator's redo panel: edge rows (a redo keeps them; an edge set by itself -> Mixed)
_rows = [{"name": "", "index": k, "label": "e", "curve": False, "cont": "G2" if k == 0 else "G1", "tension": 1.4 if k == 1 else 1.0}
         for k in range(6)]
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1", last_all="G1", last_conts=",".join(["G1"] * 6),
                         pt_tension_all=1.0, last_tension=1.0, edges=_rows)
_hp = features.history(ctx.active_object)["params"]
check("patch operator: per-edge continuity and tension rows (redo panel) are built as set",
      r == {'FINISHED'} and _hp["cont"] == ["G2"] + ["G1"] * 5 and abs(_hp.get("pt_tension", [0, 0])[1] - 1.4) < 1e-6
      and "pt_all" not in _hp and not _pt_bad(ctx.active_object), str(_hp))

# --- tessellation: a periodic surface's seam meshes without folds (u wraps instead of clamping at the seam)
from blender_nurbs import store as _st
_fold_counts = []
for _op, _p in (("prim_cylinder", {"radius": 1.0, "height": 2.0, "degree": 3, "spans_u": 8, "spans_v": 1, "caps": False}),
                ("prim_torus", {"radius": 1.0, "radius2": 0.3, "degree": 0, "spans_u": 8, "spans_v": 8})):
    _b = _st.to_brep(features.run(_op, [], _p))
    _P, _N, _T = _st._tessellate(_b, 0.002, 15.0, 2.0, 1.0 / 30.0)
    _fold_counts.append((_op, _st._periodic(_b.faces()[0].surface()), len(_st._folded(np.asarray(_P, float), np.asarray(_N, float), _st._tri_array(_T)))))
check("tessellation: tube and torus (periodic) display meshes have no folded triangles at the seam",
      all(per_[0] and n_ == 0 for _, per_, n_ in _fold_counts), str(_fold_counts))

# --- UI: the four tool groups list every tool once; panels and menus draw without errors
from types import SimpleNamespace as _NS
from blender_nurbs import toolgroups as _tg, ui as _ui, primitives as _prims


class _FakeProps:
    def __init__(self, d): object.__setattr__(self, "_d", d)
    def __setattr__(self, k, v): self._d[k] = v
    def __getattr__(self, k): return self._d.get(k)


class _FakeLay:
    def __init__(self, rec): object.__setattr__(self, "_rec", rec)
    def __setattr__(self, k, v): pass
    def __getattr__(self, name):
        rec = self._rec
        def f(*a, **k):
            if name == "operator":
                e = {"op": a[0], "props": {}}
                rec.append(e)
                return _FakeProps(e["props"])
            if name == "menu":
                rec.append({"menu": a[0]})
            if name == "prop":
                rec.append({"prop": a[1]})
            return _FakeLay(rec)
        return f


def _draw(cls, **extra):
    rec = []
    cls.draw(_NS(layout=_FakeLay(rec), **extra), ctx)
    return rec


_ui_err, _drawn = [], {}
for _cls in _tg.classes:
    try:
        if issubclass(_cls, bpy.types.Panel):
            _drawn[_cls.__name__] = _draw(_cls, group=_cls.group, hint=_cls.hint)
        else:
            _drawn[_cls.__name__] = _draw(_cls, group=getattr(_cls, "group", ""))
    except Exception as _e:
        _ui_err.append(f"{_cls.__name__}: {_e!r}")
_in_panels = [e["op"] for n, r in _drawn.items() if n.startswith("VIEW3D_PT") for e in r if "op" in e]
_missing = []
for _op in set(_in_panels):
    try:
        getattr(bpy.ops.nproto, _op.split(".")[1]).get_rna_type()
    except Exception:
        _missing.append(_op)
_not_tools = {"add_primitive", "cvs_cancel", "cvs_done", "cycle_continuity", "delete_history", "collect_inputs", "export_step",
              "import_step", "open_step", "history_focus", "history_toggle_hide", "tree_step", "step_unfocus", "mute_step", "patch_add_guides",
              "patch_auto_corners", "patch_corner", "patch_remove_guide", "rebuild", "rebuild_waiting", "remove_step",
              "restore_inputs", "sync_feature", "tab_edit", "continuity", "continuity_clear", "section_add", "section_orient",
              "section_targets", "section_curves", "section_remove", "subsel_picker", "bake_feature", "delete_feature",
              "curve_degree", "fix_clip_start", "delete_faces"}
_all_tools = {"nproto." + n for n in dir(bpy.ops.nproto) if n not in _not_tools}
_tool_ops = [o_ for o_ in _in_panels if o_ not in ("nproto.add_primitive", "nproto.boolean")] + ["nproto.boolean"]
_bool_ops = sorted(e["props"].get("operation") for n, r in _drawn.items() if n.startswith("VIEW3D_PT") for e in r
                   if e.get("op") == "nproto.boolean")
_kinds = [e["props"].get("kind") for n, r in _drawn.items() if n.startswith("VIEW3D_PT") for e in r
          if e.get("op") == "nproto.add_primitive"]
check("UI: every tool is in exactly one group panel and all operators exist",
      not _ui_err and not _missing and sorted(_tool_ops) == sorted(_all_tools)
      and sorted(_kinds) == sorted(list(_prims.CURVE_KINDS) + list(_prims.SURFACE_KINDS))
      and _bool_ops == ["DIFFERENCE", "INTERSECTION", "UNION"],
      f"bool={_bool_ops} err={_ui_err} missing={_missing} extra={sorted(set(_tool_ops) - _all_tools)} "
      f"absent={sorted(_all_tools - set(_tool_ops))} dup={sorted({o for o in _tool_ops if _tool_ops.count(o) > 1})}")
_menu_ops = sorted(e["op"] for n, r in _drawn.items() if n.startswith("VIEW3D_MT") and n != "VIEW3D_MT_nproto_tools"
                   for e in r if "op" in e)
_subs = [e["menu"] for e in _drawn["VIEW3D_MT_nproto_tools"] if "menu" in e]
check("UI: header menu holds the four groups, the same tools as the panels",
      _menu_ops == sorted(_in_panels) and _subs == ["VIEW3D_MT_nproto_curves", "VIEW3D_MT_nproto_surfaces",
                                                    "VIEW3D_MT_nproto_curve_edit", "VIEW3D_MT_nproto_surface_edit"],
      str(_subs))
_act = ctx.view_layer.objects.active
try:
    _shown = _ui.VIEW3D_PT_nproto_active.poll(ctx)
    _top = _draw(_ui.VIEW3D_PT_nproto_active)
    _rest = [_draw(_ui.VIEW3D_PT_nproto_analyze_side), _draw(_ui.VIEW3D_PT_nproto_display),
             _draw(_ui.VIEW3D_PT_nproto_tessellation), _draw(_ui.VIEW3D_PT_nproto_history)]
    _hrec = []
    _ui.header_analyze(_NS(layout=_FakeLay(_hrec)), ctx)
    ctx.view_layer.objects.active = None
    _hidden = not _ui.VIEW3D_PT_nproto_active.poll(ctx)
    _side_err = None
except Exception as _e:
    _shown = _hidden = False
    _side_err = repr(_e)
finally:
    ctx.view_layer.objects.active = _act
check("UI: Active Object panel shows for a NURBS object, hides with nothing active; Analyze and Settings draw",
      _side_err is None and _shown and _hidden and store.is_nurbs(_act), f"{_side_err} active={_act and _act.name}")
_tog = ["show_edges", "show_cvs", "show_cont_labels"]
check("UI: Settings split -- Display (the three toggles), Tessellation (closed, its four values), History; the "
      "toggles also sit as icons in the 3D view header",
      _side_err is None and [e["prop"] for e in _rest[1] if "prop" in e] == _tog
      and len([e for e in _rest[2] if "prop" in e]) == 4
      and "construction_history" in [e.get("prop") for e in _rest[3]]
      and [e["prop"] for e in _hrec if e.get("prop") in _tog] == _tog
      and 'DEFAULT_CLOSED' in _ui.VIEW3D_PT_nproto_tessellation.bl_options, str(_hrec))

# --- Edit CVs on a Patch: trims fitted tightly (re-lifting the edges keeps them in place),
#     history and the exact B-rep kept unless a CV moves
from blender_nurbs import jobs as _jobs, patch as _pmod
_sm_ext = _smk.blender_ext


def _edge_pts(b_, n_=40):
    out_ = []
    for e_ in b_.edges():
        t0_, t1_ = e_.parameter_range()
        out_.append(np.array([list(e_.point_at(t_)) for t_ in np.linspace(t0_, t1_, n_)]))
    return out_


def _relift_move(o_):
    """Worst edge move, relative to the edge's length, when the edges are
    re-lifted from their UV trim curves onto the unchanged surface (what a CV
    edit does to edges it doesn't touch)."""
    cg_ = ncvs.make_cage(ctx, o_)
    meta_, pts_ = json.loads(cg_[ncvs.CAGE_META]), json.loads(cg_[ncvs.CAGE_CVS0])
    me_ = cg_.data
    bpy.data.objects.remove(cg_)
    bpy.data.meshes.remove(me_)
    _, br_ = _jobs.run("retarget", {"target": o_}, ["result", "untrimmed"], meta=meta_, cvs=pts_)
    a_, b_ = _edge_pts(store.get_brep(o_)), _edge_pts(store.brep_from_text(br_["result"]))
    return max(float(np.linalg.norm(x - y, axis=1).max() / (np.linalg.norm(np.diff(x, axis=0), axis=1).sum() or 1.0))
               for x, y in zip(a_, b_))


_PP = ctx.active_object
_tight = getattr(_sm_ext, "trim_surface_with_3d_curves_tol", None)
_mv_old = float("nan")
if _tight is not None:
    delattr(_sm_ext, "trim_surface_with_3d_curves_tol")      # the kernel's own trim (0.01 fit) for comparison
    features.rebuild(_PP, ctx, force=True)
    _mv_old = _relift_move(_PP)
    setattr(_sm_ext, "trim_surface_with_3d_curves_tol", _tight)
    features.rebuild(_PP, ctx, force=True)
_mv_new = _relift_move(_PP)
check("patch trims: re-lifting the edges from their UV curves keeps them within the labels' gap tolerance",
      _tight is not None and _mv_new < 0.5 * _cl.TOL_GAP_REL, f"old trim {_mv_old:.1e}, new {_mv_new:.1e} (x edge length)")

_txt0 = _PP[store.PROP_BREP]
bpy.ops.object.select_all(action='DESELECT'); _PP.select_set(True); ctx.view_layer.objects.active = _PP
r1 = bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object
_hist_open = features.has_history(_PP)
ctx.view_layer.update(); ncvs._dirty.add(_cg.name); ncvs._process()      # cage update without a move
_same_open = _PP[store.PROP_BREP] == _txt0
r2 = bpy.ops.nproto.cvs_done()
check("Edit CVs on a patch, Done without moving a CV: history and the exact B-rep are kept",
      r1 == {'FINISHED'} and r2 == {'FINISHED'} and _hist_open and _same_open and features.has_history(_PP)
      and _PP[store.PROP_BREP] == _txt0 and ncvs.cage_for(_PP) is None)
bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object; _cgn = _cg.name
bpy.ops.object.mode_set(mode='OBJECT'); ctx.view_layer.update(); ncvs._finish_cages()       # Tab out
check("Edit CVs on a patch, Tab out without a change: history and B-rep kept",
      _cgn not in bpy.data.objects and features.has_history(_PP) and _PP[store.PROP_BREP] == _txt0)
bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object
_bm = bmesh.from_edit_mesh(_cg.data); _bm.verts.ensure_lookup_table()
_mid = len(_bm.verts) // 2 + int(math.sqrt(len(_bm.verts))) // 2
_bm.verts[_mid].co.z += 0.2; bmesh.update_edit_mesh(_cg.data)
ctx.view_layer.update(); ncvs._process()
_changed = _PP[store.PROP_BREP] != _txt0
r3 = bpy.ops.nproto.cvs_cancel()
check("Edit CVs on a patch, move a CV then Cancel: history and B-rep kept",
      r3 == {'FINISHED'} and _changed and features.has_history(_PP) and _PP[store.PROP_BREP] == _txt0)
bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object
_bm = bmesh.from_edit_mesh(_cg.data); _bm.verts.ensure_lookup_table()
_bm.verts[_mid].co.z += 0.2; bmesh.update_edit_mesh(_cg.data)
ctx.view_layer.update(); ncvs._process()
r4 = bpy.ops.nproto.cvs_done()
check("Edit CVs on a patch, move a CV then Done: the edit stays and the history is removed",
      r4 == {'FINISHED'} and not features.has_history(_PP) and _PP[store.PROP_BREP] != _txt0)

# --- Edit CVs on a trimmed face whose edges have no UV trim curves (some STEP faces):
#     they are dropped onto the surface first, then the edit works as usual
_nuv_pts = [(x, y, 0.3 * math.sin(x) * math.cos(y)) for x in np.linspace(500, 503, 4) for y in np.linspace(0, 3, 4)]
_nuv_srf = _sm_ext.surface_from_cvs(3, 3, _nuv_pts, [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], None)
_NU = store.new_object(ctx, "NoUVFace", _sm_ext.brep_from_surface(_nuv_srf))      # a Brep: handled as trimmed
_nuv_missing = _sm_ext.count_missing_uv_trims(store.get_brep(_NU))
_nuv_a0 = store.get_brep(_NU).area(1e-9)
bpy.ops.object.select_all(action='DESELECT'); _NU.select_set(True); ctx.view_layer.objects.active = _NU
r1 = bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object
_bm = bmesh.from_edit_mesh(_cg.data); _bm.verts.ensure_lookup_table()
_bm.verts[5].co.z += 0.5; _bm.verts[10].co.z -= 0.4; bmesh.update_edit_mesh(_cg.data)
ctx.view_layer.update(); ncvs._process()
r2 = bpy.ops.nproto.cvs_done()
_nb = store.get_brep(_NU)
check("Edit CVs on a trimmed face without UV trim curves: the edit works, edges stay on the surface",
      _nuv_missing == 4 and not store.is_natural(_NU) and r1 == {'FINISHED'} and r2 == {'FINISHED'}
      and _nb.face_count() == 1 and abs(_nb.area(1e-9) - _nuv_a0) > 1e-3 and edge_gap(_NU) < 1e-6
      and _sm_ext.count_missing_uv_trims(_nb) == 0,
      f"missing before {_nuv_missing}, area {_nuv_a0:.4f} -> {_nb.area(1e-9):.4f}, gap {edge_gap(_NU):.1e}")

# --- display mesh: flat faces stay a few triangles, curved faces follow the angle tolerance
_ch, _an = store.tess_settings(ctx)
_flatb = store.to_brep(features.run("prim_plane", [], {"size": 2.0, "degree": 1, "spans_u": 1, "spans_v": 1}))
_tubeb = store.to_brep(features.run("prim_cylinder", [], {"radius": 1.0, "height": 2.0, "degree": 3,
                                                                        "spans_u": 8, "spans_v": 1, "caps": False}))
_fP, _fN, _fT = store._tessellate(_flatb, _ch, _an, 2.0, 1.0 / 30.0)
_cP, _cN, _cT = store._tessellate(_tubeb, _ch, _an, 2.0, 1.0 / 30.0)
_cT = store._tri_array(_cT)
_cn = np.asarray(_cN, float); _cn /= np.linalg.norm(_cn, axis=1, keepdims=True)
_spread = np.degrees(np.arccos(np.clip(np.min(np.vstack([np.einsum("ij,ij->i", _cn[_cT[:, a]], _cn[_cT[:, b]])
                                                         for a, b in ((0, 1), (1, 2), (2, 0))]), axis=0), -1, 1)))
check("display mesh: a flat face is a few triangles, a curved one stays within the angle tolerance",
      abs(_ch - 0.0005) < 1e-7 and abs(_an - 5.0) < 1e-6 and len(store._tri_array(_fT)) <= 8 and float(np.percentile(_spread, 99)) <= _an + 0.5,
      f"flat {len(store._tri_array(_fT))} tris, tube {len(_cT)} tris (at 0.002 / 15 deg: "
      f"{len(store._tri_array(store._tessellate(_tubeb, 0.002, 15.0, 2.0, 1.0 / 30.0)[2]))}), "
      f"99th pct normal spread {np.percentile(_spread, 99):.2f} deg")

# --- Environment matcap: Red is gone, Environment reflects a tunnel, a horizon or Blender's HDRIs
from blender_nurbs import environments as _env
_vw = ctx.window_manager.nproto_view
_styles = {i.identifier: i.value for i in _vw.bl_rna.properties["matcap_style"].enum_items}
_srcs = [i[0] for i in _ui._env_source_items(_vw, ctx)]      # a dynamic enum: bl_rna lists no items
_hn = _env.hdri_names()
check("matcap: Highlight / Gray Lines / Environment (Red removed, stored numbers kept); sources are "
      "Light Tunnel, Horizon and Blender's HDRIs",
      _styles == {"HIGHLIGHT": 0, "LINES": 2, "ENVIRONMENT": 3} and _srcs[:2] == ["TUNNEL", "HORIZON"]
      and _srcs[2:] == _hn and {"city", "studio", "sunset"} <= set(_hn), f"{_styles} {_srcs}")
_D, _dA = _env.equirect_dirs(64, 32)
_E1 = _env.irradiance(np.ones((32, 64, 3)))
_uu, _vv = _env.equirect_uv(_D)
_ui_x = ((np.arange(64) + 0.5) / 64)[None, :].repeat(32, 0).ravel()
_vi_x = ((np.arange(32) + 0.5) / 32)[:, None].repeat(64, 1).ravel()
check("environment: equirect directions and (u, v) agree, solid angles sum to 4 pi, "
      "a uniform sky gives irradiance 1 everywhere",
      np.allclose(np.mod(_uu, 1.0), _ui_x, atol=1e-9) and np.allclose(_vv, _vi_x, atol=1e-9)
      and abs(_dA.sum() - 4 * np.pi) < 0.01 and abs(_E1.min() - 1) < 0.02 and abs(_E1.max() - 1) < 0.02,
      f"sum dA {_dA.sum():.4f}, E {_E1.min():.4f}..{_E1.max():.4f}")
_lum = np.array([0.2126, 0.7152, 0.0722])
_hd = {n: _env.hdri(n) for n in ("city", "forest", "sunset", "studio")}
_up = {n: (float(d["irr"][-1].mean(axis=0) @ _lum), float(d["irr"][0].mean(axis=0) @ _lum)) for n, d in _hd.items()}
check("environment: Blender's HDRIs load at 1024 / 256 / 64 wide, sky up (outdoor ones light from above), "
      "brightest irradiance x gain = 1",
      all(d is not None and [l.shape[1] for l in d["levels"]] == [1024, 256, 64]
          and abs(float((d["irr"] @ _lum).max()) * d["gain"] - 1) < 1e-6 for d in _hd.values())
      and all(_up[n][0] > 2 * _up[n][1] for n in ("city", "forest", "sunset")), str(_up))
_ring = np.stack([np.zeros(720), np.cos(np.linspace(0, 2 * np.pi, 720, endpoint=False)),
                  np.sin(np.linspace(0, 2 * np.pi, 720, endpoint=False))], axis=1)
_on = _env.tunnel(_ring, 8)[:, 0] > 0.3
_hz = _env.horizon(np.array([[1, 0, 0.3], [1, 0, -0.3], [1, 0, 0.1]]), 0.2)[:, 0]
_Et, _gt = _env.procedural("TUNNEL", 8)
check("environment: the tunnel has 8 strips around its axis (one straight up), the horizon splits at its height",
      int(np.sum(_on & ~np.roll(_on, 1))) == 8 and _on[180] and _hz[0] > 0.5 and _hz[1] < 0.1 and _hz[2] < 0.1
      and abs(float((_Et @ _lum).max()) * _gt - 1) < 1e-6, f"strips {int(np.sum(_on & ~np.roll(_on, 1)))}, horizon {_hz}")
_vw.show_matcap, _vw.matcap_style = True, 'ENVIRONMENT'
try:
    _pa = []
    for _sv in ("TUNNEL", "HORIZON", "studio"):
        _vw.env_source = _sv
        _pa.append([e["prop"] for e in _draw(_ui.VIEW3D_PT_nproto_analyze_side) if "prop" in e])
    _env_err = None
except Exception as _e:
    _env_err = repr(_e)
finally:
    _vw.show_matcap = False
    _vw.property_unset("env_source")
check("UI: Environment shows its source, Rotation + Lights (tunnel), Height (horizon), Rotation (HDRI), "
      "paint, gloss, roughness and metallic, and no Clear Coat toggle (it is the coat)",
      _env_err is None and all({"env_source", "env_paint", "env_gloss", "env_roughness", "env_metallic"} <= set(p_)
                               and "env_coat" not in p_ for p_ in _pa)
      and "env_lights" in _pa[0] and "env_height" in _pa[1] and "env_rotation" in _pa[2]
      and "env_lights" not in _pa[2] and "zebra_direction" not in _pa[0], f"{_env_err} {_pa}")

# --- Clear Coat: offered over Draft / Curvature / Highlight / Gray Lines, not over Zebra; it has the
#     environment's controls but no paint (the overlay is the base colour)
_coat = {}
try:
    for _nm, _set in (("draft", {"show_draft": True}), ("curv", {"show_curvature": True}),
                      ("hl", {"show_matcap": True, "matcap_style": 'HIGHLIGHT'}),
                      ("lines", {"show_matcap": True, "matcap_style": 'LINES'}), ("zebra", {"show_zebra": True})):
        for _k, _val in _set.items():
            setattr(_vw, _k, _val)
        _vw.env_coat = False
        _off = [e["prop"] for e in _draw(_ui.VIEW3D_PT_nproto_analyze_side) if "prop" in e]
        _vw.env_coat = True
        _on = [e["prop"] for e in _draw(_ui.VIEW3D_PT_nproto_analyze_side) if "prop" in e]
        _hdr = []
        _ui.header_analyze(_NS(layout=_NS(popover=lambda *a, **k: _hdr.append(k.get("text")),
                                         row=lambda *a, **k: _NS(prop=lambda *a, **k: None))), ctx)
        _coat[_nm] = ("env_coat" in _off, "env_source" in _off, "env_source" in _on, "env_roughness" in _on,
                      "env_paint" in _on or "env_metallic" in _on, _hdr[0])
        for _k in ("show_draft", "show_curvature", "show_matcap", "show_zebra"):
            setattr(_vw, _k, False)
    _coat_err = None
except Exception as _e:
    _coat_err = repr(_e)
finally:
    for _k in ("show_draft", "show_curvature", "show_matcap", "show_zebra", "env_coat", "matcap_style"):
        _vw.property_unset(_k)
check("UI: Clear Coat toggle over Draft / Curvature / Highlight / Gray Lines (not Zebra); on, it shows the "
      "environment and roughness, no paint; the header names it",
      _coat_err is None
      and all(_coat[n_][:5] == (True, False, True, True, False) and _coat[n_][5].endswith("Coat")
              for n_ in ("draft", "curv", "hl", "lines"))
      and _coat["zebra"][:3] == (False, False, False) and "Coat" not in _coat["zebra"][5], f"{_coat_err} {_coat}")

# a radius left over from the last use (or the default) can be far bigger than the part:
# from the UI (fit_pending, set by invoke) it is scaled to the part; a script's radius is kept
from blender_nurbs import draw as _dr, ops_trim as _otr
_sf = poly("SmallFloor_ln", [(29.98, 0, 0), (30.02, 0, 0)])
bpy.ops.nproto.extrude(inputs=ins((_sf, "S0")), distance=0.05, direction="Y"); _sfl = ctx.active_object; _sfl.name = "SmallFloor"
_sw = poly("SmallWall_ln", [(30, 0, -0.02), (30, 0.05, -0.02)])
bpy.ops.nproto.extrude(inputs=ins((_sw, "S0")), distance=0.04, direction="Z"); _swl = ctx.active_object; _swl.name = "SmallWall"
try:
    _r_script = bpy.ops.nproto.fillet(surf_a="SmallFloor", surf_b="SmallWall", radius=0.25, trim_bases=False)
except RuntimeError as _e:
    _r_script = {'CANCELLED'} if "may be too big" in str(_e) else str(_e)
_r_ui = bpy.ops.nproto.fillet(surf_a="SmallFloor", surf_b="SmallWall", radius=0.25, trim_bases=False, fit_pending=True)
_sfil = ctx.active_object
check("fillet: a radius too big for the part is scaled to it from the UI (script radius kept: no fillet), "
      "nice_length rounds to 1/2/5",
      _r_script == {'CANCELLED'} and _r_ui == {'FINISHED'} and _sfil.name.startswith("Fillet")
      and max(_sfil.dimensions[0], _sfil.dimensions[2]) < 0.004
      and [_otr.nice_length(x) for x in (0.0027, 0.0099, 0.6, 1.0)] == [0.002, 0.005, 0.5, 1.0],
      f"{_r_script} {_r_ui} {_sfil.name} {tuple(round(d, 4) for d in _sfil.dimensions)}")

# the edges of a hidden active object are drawn as a ghost only when it was picked
# in the Construction Tree, not after hiding it with H
_swl.hide_set(False); ctx.view_layer.objects.active = _swl; _swl.hide_set(True)
_g_h = _dr.ghost["name"] == _swl.name
bpy.ops.nproto.history_focus(name=_swl.name)
_g_tree = _dr.ghost["name"] == _swl.name
_swl.hide_set(False); bpy.ops.nproto.history_focus(name=_swl.name)
check("ghost edges: only for a hidden object picked in the Construction Tree",
      not _g_h and _g_tree and _dr.ghost["name"] is None, f"{_g_h} {_g_tree} {_dr.ghost}")

# a fillet ring around a cap (closed in u, clamped knots): its mesh must not be pulled across the
# seam (mesh area = surface area), and it faces the way the cap and the wall face
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.5, degree="0", caps=False); _rw = ctx.active_object
_rw.location = (60, 0, 0)
bpy.ops.nproto.add_primitive(kind="DISK", radius=1.3, degree="0"); _rc = ctx.active_object
_rc.location = (60, 0, 1.0)
ctx.view_layer.update()
_rr = bpy.ops.nproto.fillet(surf_a=_rw.name, surf_b=_rc.name, radius=0.2, trim_bases=True)
_rf = ctx.active_object
_rme = _rf.data
_rar = np.zeros(len(_rme.polygons)); _rme.polygons.foreach_get("area", _rar)
_rfn = np.zeros(len(_rme.polygons) * 3); _rme.polygons.foreach_get("normal", _rfn); _rfn = _rfn.reshape(-1, 3)
_rce = np.zeros(len(_rme.polygons) * 3); _rme.polygons.foreach_get("center", _rce); _rce = _rce.reshape(-1, 3)
_rrad = _rce[:, :2] - (np.array(_rc.location[:2]) - np.array(_rf.location[:2])); _rrad = _rrad / np.maximum(np.linalg.norm(_rrad, axis=1), 1e-12)[:, None]
# at the rails the fillet is tangent to the bases: there it must face the way they face
def _mesh_nc(o):
    m_ = o.data; n_ = np.zeros(len(m_.polygons) * 3); c_ = np.zeros(len(m_.polygons) * 3)
    m_.polygons.foreach_get("normal", n_); m_.polygons.foreach_get("center", c_)
    W_ = np.array(o.matrix_world); c_ = c_.reshape(-1, 3) @ W_[:3, :3].T + W_[:3, 3]
    return n_.reshape(-1, 3) @ np.linalg.inv(W_[:3, :3]), c_      # (normals: rotation / translation only here)
_axis = np.array(_rw.location[:2])
def _radial(c_):
    d_ = c_[:, :2] - _axis
    return d_ / np.maximum(np.linalg.norm(d_, axis=1), 1e-12)[:, None]
_wn, _wc = _mesh_nc(_rw)
_cn, _ = _mesh_nc(_rc)
_wall_sign = np.sign(np.mean(np.sum(_wn[:, :2] * _radial(_wc), axis=1)))
_cap_sign = np.sign(np.mean(_cn[:, 2]))
_fn, _fc = _mesh_nc(_rf)
_on_cap = np.abs(_fc[:, 2] - _rc.location.z) < 0.01
_on_wall = np.abs(np.linalg.norm(_fc[:, :2] - _axis, axis=1) - 1.0) < 0.01
_rbrep = store.get_brep(_rf).area()
_rdot = min(float(np.mean(_fn[_on_cap, 2] * _cap_sign > 0)) if _on_cap.any() else 0.0,
            float(np.mean(np.sum(_fn[_on_wall, :2] * _radial(_fc[_on_wall]), axis=1) * _wall_sign > 0))
            if _on_wall.any() else 0.0)
_rar = np.zeros(len(_rf.data.polygons)); _rf.data.polygons.foreach_get("area", _rar)
check("fillet ring round a cap: mesh area = surface area (no faces across the seam), faces the way the cap and wall face at the rails",
      _rr == {'FINISHED'} and abs(float(_rar.sum()) - _rbrep) < 0.01 * _rbrep and _rdot > 0.99
      and store._periodic(store.get_brep(_rf).faces()[0].surface()) in ((True, False), (False, True)),
      f"{_rr} mesh {float(_rar.sum()):.4f} brep {_rbrep:.4f} outward {_rdot:.3f}")

# Boolean of closed solids: union / difference / intersection, with history (inputs hidden, the
# operation editable) and without (the active object receives the result)
def _box_at(x):
    bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0")
    o = ctx.active_object; o.location = (x, 0, 0); ctx.view_layer.update(); return o
_bv = {}
for _op in ("UNION", "DIFFERENCE", "INTERSECTION"):
    _ba, _bb = _box_at(70), _box_at(71)
    _r = bpy.ops.nproto.boolean(operation=_op, names=json.dumps([_ba.name, _bb.name]))
    _res = ctx.active_object
    _bv[_op] = (_r, round(abs(store.get_brep(_res).volume(1e-6)), 4), bool(store.get_brep(_res).is_manifold_solid()),
                features.history(_res)["op"], _ba.hide_get() and _bb.hide_get())
_hres = _res
_hres.nproto_feature.bool_op = "UNION"        # the Feature panel: edit the operation
_hv = round(abs(store.get_brep(_hres).volume(1e-6)), 4)
bpy.ops.object.select_all(action='DESELECT'); _hres.select_set(True); ctx.view_layer.objects.active = _hres
_hname = _hres.name
_rst = bpy.ops.nproto.restore_inputs()
_rst_ok = _rst == {'FINISHED'} and _hname not in [o.name for o in bpy.data.objects] and not _ba.hide_get()
_prev = ctx.scene.nproto_settings.construction_history
ctx.scene.nproto_settings.construction_history = False
_ba2, _bb2 = _box_at(80), _box_at(81)
_bb2n = _bb2.name
try:
    _r2 = bpy.ops.nproto.boolean(operation="DIFFERENCE", names=json.dumps([_ba2.name, _bb2.name]))
finally:
    ctx.scene.nproto_settings.construction_history = _prev
_nohist = (_r2, ctx.active_object.name == _ba2.name, _bb2n not in bpy.data.objects,
           round(abs(store.get_brep(_ba2).volume(1e-6)), 4))
try:
    _bad = bpy.ops.nproto.boolean(operation="UNION", names=json.dumps([_ba.name, _rw.name]))
except RuntimeError as _e:
    _bad = str(_e)
check("boolean: union 12 / difference 4 / intersection 4 of two overlapping 2-cubes, closed solids, "
      "history (inputs hidden, operation editable, Restore Inputs), no-history replaces the target, open sheet refused",
      [_bv[o][:3] for o in ("UNION", "DIFFERENCE", "INTERSECTION")]
      == [({'FINISHED'}, 12.0, True), ({'FINISHED'}, 4.0, True), ({'FINISHED'}, 4.0, True)]
      and all(v[3] == "boolean" and v[4] for v in _bv.values()) and _hv == 12.0 and _rst_ok
      and _nohist == ({'FINISHED'}, True, True, 4.0)
      and "not a closed solid" in str(_bad),
      f"{_bv} edit={_hv} restore={_rst_ok} nohist={_nohist} bad={str(_bad)[:120]}")

# --- surfaces built from other surfaces' edges face the same way; Flip Normal ---
from blender_nurbs.square import sample_side as _ss, grid_starts as _gs, _foot as _ft
def _seam_dot(obj, src, ref):
    sd = _ss(src, ref, 10)
    s_ = store.get_surfaces(obj)[0]
    M_ = np.array(geom.world_matrix(obj), float); Mi_ = np.linalg.inv(M_)
    d_ = []
    for k in range(1, 9):
        q = (Mi_ @ np.append(sd.P[k], 1))[:3]
        _, su, sv = s_.derivatives(*_ft(s_, q, _gs(s_, [q])[0]))
        n_ = M_[:3, :3] @ np.cross(su, sv)
        d_.append(float(n_ @ sd.Nout[k]) / np.linalg.norm(n_))
    return round(float(np.mean(d_)), 3)
_o1 = nurbs("OR_c1", [(90, 0, 0), (90, 0.7, 0), (90, 1.4, 0), (90, 2, 0)])
_o2 = nurbs("OR_c2", [(92, 0, 0), (92, 0.7, 0), (92, 1.4, 0), (92, 2, 0)])
_o3 = nurbs("OR_c3", [(94, 2, 0.5), (94, 1.4, 0.5), (94, 0.7, 0.5), (94, 0, 0.5)])
bpy.ops.nproto.loft(inputs=ins((_o1, "S0"), (_o2, "S0"))); _oA = ctx.active_object
_sA = store.get_surfaces(_oA)[0]
_eA = min(("U0", "U1", "V0", "V1"), key=lambda r: abs(np.mean([p[0] for p in geom.boundary_points(_sA, r, 5)]) - 92))
_votes = {}
for _flipB in (False, True):        # whichever way the curve runs, B comes out facing like A
    bpy.ops.nproto.loft(inputs=ins((_oA, _eA), (_o3, "S0")), flip_b=_flipB); _oB = ctx.active_object
    _votes[_flipB] = _seam_dot(_oB, _oA, _eA)
_fl_before = (_oA.get(features.PROP_FLIPPED), json.loads(_oB[features.PROP_HIST])["refs"][0])
_oA.nproto_feature.flip_normal = True           # Feature panel: Flip Normal on A
features.rebuild(_oA, ctx, force=True)
_ref_after = json.loads(_oB[features.PROP_HIST])["refs"][0]
features.rebuild(_oB, ctx)
_sB = store.get_surfaces(_oB)[0]
_gapB = max(float(np.linalg.norm(np.array(_sB.evaluate(*_ft(_sB, np.array(p), _gs(_sB, [p])[0]))) - np.array(p)))
            for p in geom.boundary_points(store.get_surfaces(_oA)[0], _ref_after, 9))
check("orientation: a loft from a surface edge faces like that surface (either curve direction); Flip Normal on the "
      "source turns it over, renames the edge the loft uses, and the loft follows",
      min(_votes.values()) > 0.9 and _fl_before[0] == 0 and _oA.get(features.PROP_FLIPPED) == 1
      and _ref_after == {"U0": "U1", "U1": "U0"}.get(_fl_before[1], _fl_before[1])
      and _seam_dot(_oB, _oA, _ref_after) > 0.9 and _gapB < 1e-6,
      f"votes={_votes} before={_fl_before} after={_ref_after} flipped={_oA.get(features.PROP_FLIPPED)} "
      f"dot={_seam_dot(_oB, _oA, _ref_after)} gap={_gapB:.1e}")

# --- feature batch: CV selection, Flip Normal, G2 check, Offset / Thicken, chord + blend fillets ---
from blender_nurbs import cvs as _cvs
from blender_nurbs.ops_sweep import continuity_report as _cr
def _floor(name, y, x0=0.0):
    ln = poly(name + "_ln", [(x0, y - 2, 0), (x0, y + 2, 0)])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=4.0, direction="X")
    o = ctx.active_object; o.name = name; return o
def _wall(name, y, x=2.0):
    ln = poly(name + "_ln", [(x, y - 2, -1), (x, y + 2, -1)])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=3.0, direction="Z")
    o = ctx.active_object; o.name = name; return o
def _sel(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    ctx.view_layer.objects.active = objs[0]
def _face_normals(o):
    b = store.get_brep(o); R = np.array(geom.world_matrix(o).to_3x3())
    out = []
    for f in b.faces():
        c = f.centroid()
        _, uvs, _ = _kl().surface_closest_point(f.surface(), tuple(c))
        uv = uvs[0]; n = R @ np.array(f.outward_normal_at_uv(uv.x, uv.y)); out.append(n / np.linalg.norm(n))
    return out
def _bbz(o):
    b = store.get_brep(o); M = geom.world_matrix(o)
    lo, hi = b.bounding_box(); return round((M @ Vector(lo)).z, 4), round((M @ Vector(hi)).z, 4)

# 1. a new CV cage starts with nothing selected
_fc = _floor("FB_cage", 900)
_cage = _cvs.make_cage(ctx, _fc)
_nsel = sum(1 for v in _cage.data.vertices if v.select)
_me = _cage.data; bpy.data.objects.remove(_cage); bpy.data.meshes.remove(_me)
check("CV cage: no CV selected when it opens", _nsel == 0, f"{_nsel} selected")

# 2. Flip Normal: plain surface with history -> step, flip again -> step gone
_f1 = _floor("FB_flip", 910)
_n0 = _face_normals(_f1)[0]
_sel(_f1); _r1 = bpy.ops.nproto.flip_normal()
_n1 = _face_normals(_f1)[0]; _st1 = [e["op"] for e in features.stack(_f1)]
_sel(_f1); _r2 = bpy.ops.nproto.flip_normal()
_n2 = _face_normals(_f1)[0]; _st2 = [e["op"] for e in features.stack(_f1)]
check("flip normal: a plain surface turns over (history step), flipping again removes the step",
      _r1 == _r2 == {'FINISHED'} and _n0 @ _n1 < -0.999 and _n0 @ _n2 > 0.999 and _st1 == ["flip_normal"]
      and _st2 == [] and store.is_natural(_f1), f"{_n0} {_n1} {_n2} {_st1} {_st2}")
# trimmed: trims kept, still trimmed, history edit (input moved) keeps it flipped
_f2 = _floor("FB_flipT", 920)
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 920, 1)); _cc = ctx.active_object
bpy.ops.nproto.trim(action="KEEP", target=_f2.name, cutters=ins((_cc, "S0")), ref_point=(1, 920, 0), normal=(0, 0, 1),
                    view_dir=(0, 0, -1))
_a0, _m0 = area(_f2), _face_normals(_f2)[0]
_sel(_f2); _r3 = bpy.ops.nproto.flip_normal()
_a1, _m1 = area(_f2), _face_normals(_f2)[0]
check("flip normal: a trimmed surface turns over, trim kept", _r3 == {'FINISHED'} and abs(_a1 - _a0) < 1e-4 * _a0
      and _m0 @ _m1 < -0.999 and not store.is_natural(_f2), f"area {_a0:.5f} -> {_a1:.5f} n {_m0} -> {_m1}")
# joined L: both faces turn over, still joined
_L = make_L("FB_L", loc=(0, 790, 0))
_ln0 = _face_normals(_L); _sh0 = len(store.shared_edges(_L))
_sel(_L); _r4 = bpy.ops.nproto.flip_normal()
_ln1 = _face_normals(_L)
check("flip normal: a joined sheet turns over, still joined",
      _r4 == {'FINISHED'} and len(_ln1) == 2 and all(a @ b < -0.999 for a, b in zip(_ln0, _ln1))
      and len(store.shared_edges(_L)) == _sh0 == 1, f"{_ln0} -> {_ln1}")
# without construction history: geometry replaced, no stack
_prev = ctx.scene.nproto_settings.construction_history
ctx.scene.nproto_settings.construction_history = False
try:
    _f3 = _floor("FB_flipN", 930)
    _k0 = _face_normals(_f3)[0]
    _sel(_f3); _r5 = bpy.ops.nproto.flip_normal()
    _k1 = _face_normals(_f3)[0]
finally:
    ctx.scene.nproto_settings.construction_history = _prev
check("flip normal without history", _r5 == {'FINISHED'} and _k0 @ _k1 < -0.999 and not features.stack(_f3))

# 3. Continuity Check reports G2: a G2 loft to a flat floor matches its curvature, a G1 loft does not
_gf = _floor("G2_floor", 980)
_gwl = poly("G2_wall_ln", [(6, 978, 1), (6, 982, 1)])
bpy.ops.nproto.extrude(inputs=ins((_gwl, "S0")), distance=2.0, direction="Z"); _gw = ctx.active_object
def _iso_at(o, key, val):
    s_ = store.get_surfaces(o)[0]
    return min(("U0", "U1", "V0", "V1"), key=lambda r: abs(np.mean([p[key] for p in geom.boundary_points(s_, r, 5)]) - val))
_ge_f, _ge_w = _iso_at(_gf, 0, 4.0), _iso_at(_gw, 2, 1.0)
_rep = {}
for _cont in ("G1", "G2"):
    bpy.ops.nproto.loft(inputs=json.dumps([[_gf.name, _ge_f], [_gw.name, _ge_w]]), cont_a=_cont, cont_b=_cont)
    _lo = ctx.active_object
    _rep[_cont] = _cr(_lo, _gf) or _cr(_gf, _lo)
check("continuity check: G2 measured (curvature across the edge), G2 loft passes, G1 loft does not",
      _rep["G2"] and _rep["G1"] and _rep["G2"]["curv_diff"] < 0.02 and _rep["G1"]["curv_diff"] > 0.1,
      f"G1 {_rep['G1'] and round(_rep['G1']['curv_diff'], 4)} G2 {_rep['G2'] and round(_rep['G2']['curv_diff'], 5)}")

# 4. Offset of a joined sheet and of a trimmed surface; Thicken single / joined / symmetric
_L2 = make_L("OF_L", loc=(0, 810, 0))
_sel(_L2); _r6 = bpy.ops.nproto.offset(distance=0.1); _of = ctx.active_object
_sel(_f2); _r7 = bpy.ops.nproto.offset(distance=0.1); _ot = ctx.active_object
check("offset: joined sheet (2 faces, corner closed) and trimmed surface (trim kept, moved 0.1)",
      _r6 == _r7 == {'FINISHED'} and store.get_brep(_of).face_count() == 2 and len(store.shared_edges(_of)) == 1
      and abs(area(_ot) - area(_f2)) < 1e-3 * area(_f2) and abs(abs(_bbz(_ot)[0]) - 0.1) < 1e-6
      and features.history(_of)["op"] == "offset",
      f"faces {store.get_brep(_of).face_count()} shared {len(store.shared_edges(_of))} area {area(_ot):.5f}/{area(_f2):.5f} z {_bbz(_ot)}")
_t1 = _floor("TK_single", 940)
_sel(_t1); _r8 = bpy.ops.nproto.thicken(distance=0.1); _tk = ctx.active_object
_v1 = Vol(_tk)
_tk.nproto_feature.distance = 0.2; features.rebuild(_tk, ctx)
_v2 = Vol(_tk)
_t2 = _floor("TK_sym", 950)
_sel(_t2); bpy.ops.nproto.thicken(distance=0.1, symmetric=True); _ts = ctx.active_object
_L3 = make_L("TK_L", loc=(0, 830, 0))
_sel(_L3); bpy.ops.nproto.thicken(distance=0.1); _tl = ctx.active_object
check("thicken: 4x4 sheet -> closed solid 1.6, feature edit 0.2 -> 3.2, symmetric centred, sheet hidden, joined L -> solid",
      _r8 == {'FINISHED'} and abs(_v1 - 1.6) < 1e-3 and abs(_v2 - 3.2) < 1e-3 and _t1.hide_get()
      and _bbz(_ts) == (-0.05, 0.05) and abs(Vol(_ts) - 1.6) < 1e-3 and Vol(_tl) > 0,
      f"v {_v1:.4f} {_v2:.4f} sym {_bbz(_ts)} {Vol(_ts):.4f} L {Vol(_tl):.4f}")

# 5. Fillets: chord (distance between the rails) and the G2 blend between two surfaces
def _chord_of(o):
    s_ = store.get_surfaces(o)[0]; (u0, u1), (v0, v1) = s_.u_domain(), s_.v_domain()
    du = np.mean([np.linalg.norm(np.array(s_.evaluate(u0, v)) - np.array(s_.evaluate(u1, v))) for v in np.linspace(v0, v1, 9)])
    dv = np.mean([np.linalg.norm(np.array(s_.evaluate(u, v0)) - np.array(s_.evaluate(u, v1))) for u in np.linspace(u0, u1, 9)])
    return float(min(du, dv))
_cf, _cw = _floor("CH_floor", 960), _wall("CH_wall", 960)
_r9 = bpy.ops.nproto.fillet(surf_a=_cf.name, surf_b=_cw.name, radius=0.4, rail_type="CHORD", shape="CIRCULAR",
                            trim_bases=True)
_cfil = ctx.active_object
_rr = 0.4 / math.sqrt(2.0)
check("surface fillet by chord: rails 0.4 apart, ball radius 0.4 / sqrt 2 at 90 deg",
      _r9 == {'FINISHED'} and abs(_chord_of(_cfil) - 0.4) < 1e-4 and abs(area(_cfil) - 0.5 * math.pi * _rr * 4) < 2e-3,
      f"chord {_chord_of(_cfil):.6f} area {area(_cfil):.4f} vs {0.5 * math.pi * _rr * 4:.4f}")
_bf, _bw = _floor("BL_floor", 970), _wall("BL_wall", 970)
_r10 = bpy.ops.nproto.fillet(surf_a=_bf.name, surf_b=_bw.name, radius=0.3, shape="BLEND", continuity="2",
                             trim_bases=True)
_bfil = ctx.active_object
_c1, _c2 = _cr(_bfil, _bf) or _cr(_bf, _bfil), _cr(_bfil, _bw) or _cr(_bw, _bfil)
_rb = features.rebuild(_bfil, ctx, force=True)
_c3 = _cr(_bfil, _bf) or _cr(_bf, _bfil)
check("surface fillet G2 blend: G2 to both surfaces, history rebuild keeps it",
      _r10 == {'FINISHED'} and _c1 and _c2 and _c1["gap"] < 1e-5 and _c2["gap"] < 1e-5 and _c1["angle"] < 0.05
      and _c2["angle"] < 0.05 and _c1["curv_diff"] < 0.02 and _c2["curv_diff"] < 0.02 and _c3 and _c3["curv_diff"] < 0.02
      and features.history(_bfil)["params"].get("blend") == 2,
      f"{_c1} {_c2} after rebuild {_c3}")
_L4 = make_L("CH_L", loc=(0, 850, 0))
_ea0 = A(_L4)
_r11 = fillet_all(_L4, shape="CIRCULAR", radius=0.3, rail_type="CHORD")
_re = 0.3 / math.sqrt(2.0)
check("edge fillet by chord: ball radius chord / sqrt 2 on a 90 deg edge (area change r L (2 - pi/2))",
      _r11 == {'FINISHED'} and abs((_ea0 - A(_L4)) - _re * 4 * (2 - math.pi / 2)) < 2e-3,
      f"dA {_ea0 - A(_L4):.5f} vs {_re * 4 * (2 - math.pi / 2):.5f}")

from blender_nurbs import history_ui as _hui
_hui.sync_step(_L4, 0)
_pg = _L4.nproto_step
_shown = (_pg.rail_type, round(_pg.radius, 6), _hui.STEP_FIELDS["edge_fillet"][1])
_lbl = steps.LABEL["edge_fillet"](features.stack(_L4)[0]["params"])
_pg.rail_type = "RADIUS"                      # step editor: back to a rolling ball of 0.3
_dA_r = _ea0 - A(_L4)
check("edge fillet step shows Rail = Chord (label 'chord'); switching it to Rolling Ball rebuilds with radius 0.3",
      _shown == ("CHORD", 0.3, "rail_type") and "chord 0.3" in _lbl
      and abs(_dA_r - 0.3 * 4 * (2 - math.pi / 2)) < 2e-3 and not features.stack(_L4)[0]["params"].get("chord"),
      f"{_shown} {_lbl} dA {_dA_r:.5f}")

# --- Boundary Fill: the solid a wall and two sheets enclose ---
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _bc = ctx.active_object
_bc.location = (0, 1000, -1); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_bc, "S0")), distance=2.0, direction="Z"); _bw = ctx.active_object
_bl = poly("BF_lo_ln", [(-2, 998, -0.5), (-2, 1002, -0.5)])
bpy.ops.nproto.extrude(inputs=ins((_bl, "S0")), distance=4.0, direction="X"); _blo = ctx.active_object
_bh = poly("BF_hi_ln", [(-2, 998, 0.5), (-2, 1002, 0.5)])
bpy.ops.nproto.extrude(inputs=ins((_bh, "S0")), distance=4.0, direction="X"); _bhi = ctx.active_object
_sel(_bw, _blo, _bhi)
_rf = bpy.ops.nproto.boundary_fill(); _sol = ctx.active_object
_v0 = Vol(_sol)
_bh.location.z += 0.5; ctx.view_layer.update()
features.rebuild(_bhi, ctx); features.rebuild(_sol, ctx)
_v1 = Vol(_sol)
features.release_inputs(ctx, [_blo, _bhi])
_sel(_blo, _bhi)
try:
    _bad = bpy.ops.nproto.boundary_fill()
except RuntimeError as _e:
    _bad = str(_e)
check("boundary fill: wall + two sheets -> closed solid pi r^2 h, inputs hidden, follows a moved sheet; "
      "surfaces that enclose nothing are refused",
      _rf == {'FINISHED'} and abs(_v0 - math.pi) < 0.01 * math.pi and abs(_v1 - 1.5 * math.pi) < 0.015 * math.pi
      and _bw.hide_get() and features.history(_sol)["op"] == "boundary_fill"
      and ("enclose" in str(_bad) or "does not cross" in str(_bad)),
      f"v {_v0:.4f} -> {_v1:.4f} bad={str(_bad)[:100]}")

# --- kept inputs live in the hidden NURBS History collection; Delete History removes freed ones ---
_ka = patch("KI_a", (0, 1100, 0), (0, 1104, 0), "X", 2.0)
_kb = patch("KI_b", (2, 1100, 0), (2, 1104, 0), "Z", 2.0)
_home = _ka.users_collection[0].name
_sel(_ka, _kb); bpy.ops.nproto.join(); _kj = ctx.active_object
_hc = features.history_collection(ctx, create=False)
_in_coll = _hc is not None and all(o.name in _hc.objects and len(o.users_collection) == 1 for o in (_ka, _kb))
for _o in ctx.view_layer.objects: _o.hide_set(False)  # what Unhide All does
_still_hidden = not _ka.visible_get() and not _kb.visible_get() and _kj.visible_get()
_sel(_kj); bpy.ops.nproto.restore_inputs()
_back = (_ka.users_collection[0].name == _home and _ka.visible_get() and _kb.visible_get()
         and _ka.name not in _hc.objects and "KI_a_joined" not in [o.name for o in bpy.data.objects])
_sel(_ka, _kb); bpy.ops.nproto.join(); _kj = ctx.active_object
_names = (_ka.name, _kb.name)
_sel(_kj); bpy.ops.nproto.delete_history(scope='SELECTED')
_purged = all(n not in bpy.data.objects for n in _names) and not features.has_history(_kj) and _kj.name in bpy.data.objects
# an older scene: the inputs sit hidden in the main collection -> Collect Inputs tidies them
_kc = patch("KI_c", (0, 1110, 0), (0, 1114, 0), "X", 2.0)
_kd = patch("KI_d", (2, 1110, 0), (2, 1114, 0), "Z", 2.0)
_sel(_kc, _kd); bpy.ops.nproto.join(); _kj2 = ctx.active_object
features.release_inputs(ctx, [_kc, _kd])              # as if made before the collection existed
_rc = bpy.ops.nproto.collect_inputs()
_collected = _kc.name in _hc.objects and _kd.name in _hc.objects and not _kc.visible_get()
check("kept inputs: Join puts them in the hidden NURBS History collection (stay hidden after Unhide All), "
      "Restore Inputs puts them back, Delete History deletes the ones nothing uses, Collect Inputs tidies old scenes",
      _in_coll and _still_hidden and _back and _purged and _rc == {'FINISHED'} and _collected,
      f"coll={_in_coll} hidden={_still_hidden} back={_back} purged={_purged} collected={_collected}")

# --- Construction Tree lists history steps (fillets ...) on their own rows; eye works on kept inputs ---
import types as _types
class _FL:
    def __init__(self, log): self.log, self.active = log, True
    def split(self, **k): return _FL(self.log)
    def row(self, **k): return _FL(self.log)
    def column(self, **k): return _FL(self.log)
    def box(self): return _FL(self.log)
    def separator(self, **k): pass
    def prop(self, *a, **k): pass
    def label(self, text="", **k): self.log.append(("label", text))
    def operator(self, idname, text="", **k):
        o = _types.SimpleNamespace(); self.log.append(("op", idname, text, o)); return o
_ta = patch("TR_a", (0, 1120, 0), (0, 1124, 0), "X", 2.0)
_tb = patch("TR_b", (2, 1120, 0), (2, 1124, 0), "Z", 2.0)
_sel(_ta, _tb); bpy.ops.nproto.join(); _tj = ctx.active_object
fillet_all(_tj, shape="CIRCULAR", radius=0.3, rail_type="CHORD")
_log = []
_hui.draw_tree(_FL(_log), ctx, _tj)
_steps = [e for e in _log if e[0] == "op" and e[1] == "nproto.tree_step"]
_labels = [e[1] for e in _log if e[0] == "label"]
_objrows = [e[2] for e in _log if e[0] == "op" and e[1] == "nproto.history_focus"]
_ok_rows = len(_steps) == 1 and _steps[0][2].startswith("1. Edge Circular") and any("chord 0.3" in t for t in _labels)
_order = [o.name for o in _hui.build_order(_tj)]
_ok_rows = _ok_rows and _objrows[-1] == _tj.name and _order[-1] == _tj.name and len(_objrows) == len(set(_objrows)) \
    and _order.index("TR_a") < _order.index(_tj.name) and _order.index("TR_b") < _order.index(_tj.name)
_sel(_ta)
bpy.ops.nproto.tree_step(name=_tj.name, root=_tj.name, index=0)
_opened = ctx.view_layer.objects.active == _tj and _tj.nproto_step.index == 0 and _tj.nproto_step.rail_type == "CHORD"
bpy.ops.nproto.history_toggle_hide(name=_ta.name)
_vis_on = _ta.visible_get()
bpy.ops.nproto.history_toggle_hide(name=_ta.name)
_vis_off = not _ta.visible_get() and _ta.name not in ctx.scene.collection.objects
check("history list: built order (inputs first, the body last, each once), a fillet step has its own row "
      "(click opens it); the eye shows / hides a kept input",
      _ok_rows and _opened and _vis_on and _vis_off,
      f"rows={[e[2] for e in _steps]} labels={_labels} opened={_opened} eye={_vis_on}/{_vis_off}")

# --- step focus: a step opened from the tree shows only that step; the object row shows the feature again ---
_log = []
features.mark_built(_tj)
_ctx_feat = _types.SimpleNamespace(view_layer=ctx.view_layer, scene=ctx.scene, window_manager=ctx.window_manager)
from blender_nurbs import feature_ui as _fui
bpy.ops.nproto.tree_step(name=_tj.name, root=_tj.name, index=0)
_fui.draw_feature(_FL(_log), ctx)
_lab1 = [e[1] for e in _log if e[0] == "label"]
from blender_nurbs import subsel as _ssf
_sf = _ssf.step_faces(_tj)
_tjf = store.get_brep(_tj).faces()
_sf_ok = len(_sf) == 1 and _ssf.faces_shown(_tj) == set(_sf) and _tjf[_sf[0]].area() < min(f.area() for k, f in enumerate(_tjf) if k != _sf[0])
_mute_ops = [e for e in _log if e[0] == "op" and e[1] == "nproto.mute_step"]
bpy.ops.nproto.history_focus(name=_tj.name, root=_tj.name)
_log = []
_fui.draw_feature(_FL(_log), ctx)
_lab2 = [e[1] for e in _log if e[0] == "label"]
check("step focus: a step picked in the tree reads like a feature (Feature: <step>, its own settings); "
      "the object shows only its own feature (its steps are tree rows, each with its on / off switch)",
      any(t.startswith("Feature: Edge") for t in _lab1) and not any(t.startswith("Feature: Join") for t in _lab1)
      and not any(t.startswith("Step 1 of") for t in _lab1)
      and any(t.startswith("Feature: Join") for t in _lab2) and not any("History steps" in t for t in _lab2)
      and not _tj.nproto_step.focus and len(_mute_ops) == 1 and _mute_ops[0][3].name == _tj.name,
      f"{_lab1[:4]} | {_lab2[:4]} | mute {len(_mute_ops)}")
check("step focus: the fillet step knows its face (highlighted while open; CVs / Hulls shows just its net)",
      _sf_ok and _ssf.step_faces(_tj) == [], f"{_sf} areas {[round(f.area(), 3) for f in _tjf]}")

# --- two surfaces trimmed by each other follow their inputs (the rebuild order does not matter) ---
from blender_nurbs import history as _hist
from blender_nurbs.contcheck import edge_kind as _ek_kind
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _mca = ctx.active_object
_mca.location = (0, 2000, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_mca, "S0")), distance=3.0, direction="Z"); _mA = ctx.active_object; _mA.name = "MT_A"
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.4); _mcb = ctx.active_object
_mcb.location = (0, 2000, 1.5); _mcb.rotation_euler = (0, math.radians(90), 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_mcb, "S0")), distance=2.0, direction="X"); _mB = ctx.active_object; _mB.name = "MT_B"
r1 = bpy.ops.nproto.trim(target="MT_A", cutters=json.dumps([["MT_B", "SURF"]]), ref_point=(1, 2000, 1.5),
                         normal=(0, 0, 1), action="DELETE", scope="THROUGH")
r2 = bpy.ops.nproto.trim(target="MT_B", cutters=json.dumps([["MT_A", "SURF"]]), ref_point=(0.5, 2000, 1.9),
                         normal=(0, 0, 1), action="DELETE", scope="THROUGH")
_sel(_mA, _mB); bpy.ops.nproto.join(tolerance=1e-3); _mj = ctx.active_object
def _open_n(o):
    return sum(1 for i in range(len(store.get_brep(o).edges())) if _ek_kind(o, f"E{i}") == "open")
_mj_area0, _mj_open0, _mA_area0 = area(_mj), _open_n(_mj), area(_mA)
_hp = json.loads(_mcb[features.PROP_HIST]); _hp["params"]["radius"] = 0.55
_mcb[features.PROP_HIST] = json.dumps(_hp)
features.rebuild(_mcb, ctx, force=True)
for _o in (_mA, _mB, _mj):     # the order that broke: A first, while B still has its old shape
    features.rebuild(_o, ctx)
_mj_area1, _mj_open1, _mA_area1 = area(_mj), _open_n(_mj), area(_mA)
# the hole in A is now the radius-0.55 crossing: A lost more area, the body still closes up along it
check("two surfaces trimmed by each other, joined: changing one circle's radius re-trims BOTH (whichever rebuilds "
      "first) and the joined body follows, still closed along the crossing",
      r1 == {'FINISHED'} and r2 == {'FINISHED'} and _mj_open0 == 3 and _mj_open1 == 3
      and _mA_area1 < _mA_area0 - 0.2 and abs(_mj_area1 - _mj_area0) > 0.05,
      f"open {_mj_open0}->{_mj_open1} A {_mA_area0:.3f}->{_mA_area1:.3f} body {_mj_area0:.3f}->{_mj_area1:.3f}")

# --- Bake / Delete (Alias' Delete History): a feature stops following its inputs; a step bakes with what is under it ---
_bk_a = patch("BK_a", (0, 1140, 0), (0, 1144, 0), "X", 2.0)
_bk_b = patch("BK_b", (2, 1140, 0), (2, 1144, 0), "Z", 2.0)
_sel(_bk_a, _bk_b); bpy.ops.nproto.join(); _bk = ctx.active_object
fillet_all(_bk, shape="CIRCULAR", radius=0.3)
_area_bk = area(_bk)
_bk2 = _bk.copy(); _bk2.data = _bk.data.copy(); ctx.collection.objects.link(_bk2)
r1 = bpy.ops.nproto.bake_feature(name=_bk.name, step=-1)          # the Join only: the fillet stays live
_st1 = features.stack(_bk)
_live_fillet = features.history(_bk) is None and len(_st1) == 1 and features.PROP_BASE in _bk
features.rebuild(_bk, ctx, force=True)
_same1 = abs(area(_bk) - _area_bk) < 1e-6
r2 = bpy.ops.nproto.bake_feature(name=_bk2.name, step=0)          # up to the fillet: all of it
_none = not features.has_history(_bk2) and abs(area(_bk2) - _area_bk) < 1e-6
check("bake: the feature alone (its steps stay live, the result unchanged); up to a step (nothing left, the shape kept)",
      r1 == {'FINISHED'} and r2 == {'FINISHED'} and _live_fillet and _same1 and _none,
      f"stack {len(_st1)} same {_same1} none {_none}")
_dk_a = patch("DK_a", (0, 1150, 0), (0, 1154, 0), "X", 2.0)
_dk_b = patch("DK_b", (2, 1150, 0), (2, 1154, 0), "Z", 2.0)
_sel(_dk_a, _dk_b); bpy.ops.nproto.join(); _dk = ctx.active_object
_dk_name = _dk.name
bpy.ops.nproto.delete_feature(name=_dk_name, mode="ALL", keep_inputs=False)     # Shift+click: inputs go too
_gone_all = _dk_name not in bpy.data.objects and "DK_a" not in bpy.data.objects and "DK_b" not in bpy.data.objects
_ek_a = patch("EK_a", (0, 1160, 0), (0, 1164, 0), "X", 2.0)
_ek_b = patch("EK_b", (2, 1160, 0), (2, 1164, 0), "Z", 2.0)
_sel(_ek_a, _ek_b); bpy.ops.nproto.join(); _ek = ctx.active_object
_ek_area = area(_ek)
_feeds = _hui.from_feeds(_ek_a, ctx.scene)
bpy.ops.nproto.delete_feature(name="EK_a", mode="BAKE")          # an input: what is built from it is baked
_kept_shape = _ek.name in bpy.data.objects and not features.has_history(_ek) and abs(area(_ek) - _ek_area) < 1e-6
check("delete (Shift+click): a body takes its unused hidden inputs with it; deleting an input bakes what was built "
      "from it (it keeps its shape) and tidies the inputs nothing uses now",
      _gone_all and _kept_shape and "EK_a" not in bpy.data.objects and "EK_b" not in bpy.data.objects
      and "Feeds: " in _feeds,
      f"gone_all {_gone_all} kept {_kept_shape} feeds '{_feeds}' left {[n for n in ('EK_a', 'EK_b') if n in bpy.data.objects]}")

_rk_a = patch("RK_a", (0, 1170, 0), (0, 1174, 0), "X", 2.0)
_rk_b = patch("RK_b", (2, 1170, 0), (2, 1174, 0), "Z", 2.0)
_sel(_rk_a, _rk_b); bpy.ops.nproto.join(); _rk = ctx.active_object
_rk_name = _rk.name
_sel(_rk); bpy.ops.nproto.thicken(distance=0.05); _rt = ctx.active_object
_rt_name = _rt.name
_hidden0 = _rk.hide_get()
bpy.ops.nproto.delete_feature(name=_rt_name)
_step1 = (_rt_name not in bpy.data.objects and _rk_name in bpy.data.objects and not _rk.hide_get()
          and _rk.select_get() and ctx.view_layer.objects.active == _rk and features.has_history(_rk))
bpy.ops.nproto.delete_feature(name=_rk_name)
_step2 = (_rk_name not in bpy.data.objects and all(n_ in bpy.data.objects for n_ in ("RK_a", "RK_b"))
          and not bpy.data.objects["RK_a"].hide_get() and not bpy.data.objects["RK_b"].hide_get()
          and bpy.data.objects["RK_a"].select_get() and bpy.data.objects["RK_b"].select_get())
check("delete (the trash, a plain click): one step back -- removing a Thicken brings back the joined body it was made "
      "from (shown, selected, its history kept); removing that Join brings back its surfaces",
      _hidden0 and _step1 and _step2, f"hidden {_hidden0} step1 {_step1} step2 {_step2}")

# a Join of two surfaces and the loft between them: removing the Join brings all three back -- the loft still
# uses the two, but a loft doesn't keep its inputs hidden (only Join, Thicken, ... do)
_ja = patch("JL_a", (0, 1180, 0), (0, 1184, 0), "X", 2.0)
_jb = patch("JL_b", (4, 1180, 0), (4, 1184, 0), "X", 2.0)
def _edge_at_x(o, x):
    s_ = store.get_surfaces(o)[0]; M_ = geom.world_matrix(o)
    return min(("U0", "U1", "V0", "V1"), key=lambda e_: abs(np.mean([(M_ @ Vector(p_)).x for p_ in geom.boundary_points(s_, e_, 5)]) - x))
bpy.ops.nproto.loft(inputs=json.dumps([[_ja.name, _edge_at_x(_ja, 2.0)], [_jb.name, _edge_at_x(_jb, 4.0)]]),
                    cont_a="G1", cont_b="G1")
_jc = ctx.active_object
_sel(_ja, _jb, _jc); bpy.ops.nproto.join(); _jj = ctx.active_object
_jj_name, _parts3 = _jj.name, [_ja.name, _jb.name, _jc.name]
_hid3 = all(bpy.data.objects[n_].hide_get() for n_ in _parts3)
bpy.ops.nproto.delete_feature(name=_jj_name)
_hc = features.history_collection(ctx, create=False)
_back3 = all(not bpy.data.objects[n_].hide_get() and (_hc is None or n_ not in _hc.objects)
             and bpy.data.objects[n_].select_get() for n_ in _parts3)
# an input stranded in the History collection (nothing keeps it any more): its eye brings it home, not a peek
_orph = patch("JL_orphan", (0, 1190, 0), (0, 1194, 0), "X", 2.0)
features.keep_inputs(ctx, [_orph])
features.set_shown(ctx, _orph, True)
_hc = features.history_collection(ctx, create=False)
_home = (not _orph.hide_get() and _orph.name not in _hc.objects and features.PROP_HOME not in _orph
         and features.PROP_PEEK not in _orph)
check("delete: removing a Join brings back every part it hid, also ones a loft is built from (a loft doesn't hide "
      "them); a hidden input nothing keeps any more goes home when its eye is clicked",
      _hid3 and _back3 and _home, f"hid {_hid3} back {_back3} home {_home}")

# --- Class-A on a closed rim of two edges (a tilted cap on a round wall): light patches, no kernel fallback ---
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _cc2 = ctx.active_object
_cc2.location = (0, 1200, -1); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_cc2, "S0")), distance=2.0, direction="Z"); _cw2 = ctx.active_object
_cl = poly("CA_lo_ln", [(-2, 1198, -0.5), (-2, 1202, -0.5)])
bpy.ops.nproto.extrude(inputs=ins((_cl, "S0")), distance=4.0, direction="X"); _clo = ctx.active_object
_ch = poly("CA_hi_ln", [(-2.5, 1198, 0.2), (-2.5, 1202, 0.2)])
bpy.ops.nproto.extrude(inputs=ins((_ch, "S0")), distance=5.0, direction="VECTOR", dir_vec=(1.0, 0.0, 0.15)); _chi = ctx.active_object
_sel(_cw2, _clo, _chi); bpy.ops.nproto.boundary_fill(); _cs = ctx.active_object
_bb = store.get_brep(_cs)
_top = [i for i in store.shared_edges(_cs)
        if (lambda e: (lambda t: e.point_at(0.5 * (t[0] + t[1]))[2])(e.parameter_range()))(_bb.edges()[i]) > -0.2
        and _bb.edges()[i].length() > 0.5]
_cs_plain = _cs.copy(); _cs_plain.data = _cs.data.copy(); ctx.collection.objects.link(_cs_plain)
_rca = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_cs.name, f"E{i}"] for i in _top]), shape="BLEND",
                                  continuity="3", radius=0.15, rail_type="CHORD", class_a=True, ca_tolerance=1e-3)
_rep = json.loads(_cs.get("nproto_class_a", "{}"))
_bf2 = store.get_brep(_cs)
_dense = [f for f in _bf2.faces() if max(ext.surface_cvs(f.surface())["count_u"], ext.surface_cvs(f.surface())["count_v"]) > 69]
from blender_nurbs.ops_trim import class_a_summary as _cas
_rings_ca = _rep.get("rings") or []
check("class-A on a closed (not round) rim, blend G3, chord: the rim is ONE even periodic ring (as without Class-A), "
      "no patches, no kernel fallback, and the report says so",
      _rca == {'FINISHED'} and len(_top) >= 1 and "error" not in _rep and _rep.get("patches", 1) == 0
      and len(_rings_ca) == 1 and _rings_ca[0]["ring"] == "periodic" and _rings_ca[0]["round"] <= 69 and not _dense
      and "one even ring" in _cas(_rep)[0] and _cas(_rep)[1],
      f"edges {_top} report {str(_rep)[:300]} dense {len(_dense)} · {_cas(_rep)[0]}")

# --- Blend / chamfer on a closed rim (cap on a round wall): the kernel's face has no seam and
#     covers a sliver; it must come back as the full ring, joined to cap and wall ---
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _rc = ctx.active_object
_rc.location = (0, 1300, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_rc, "S0")), distance=2.0, direction="Z"); _rw = ctx.active_object
_rtop = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([q[2] for q in geom.boundary_points(store.get_surfaces(_rw)[0], w, 8)]))
bpy.ops.nproto.planar(inputs=ins((_rw, _rtop))); _rcap = ctx.active_object
_sel(_rw, _rcap); bpy.ops.nproto.join(tolerance=1e-3); _rj = ctx.active_object
_ring_area = {}
for _shape, _cont in (("CIRCULAR", "1"), ("BLEND", "2"), ("BLEND", "3"), ("CHAMFER", "1")):
    _o = _rj.copy(); _o.data = _rj.data.copy(); ctx.collection.objects.link(_o)
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_o.name, f"E{i}"] for i in store.shared_edges(_o)]),
                                   shape=_shape, continuity=_cont, radius=0.2)
    _b = store.get_brep(_o)
    _ring_area[_shape + _cont] = (r == {'FINISHED'}, _b.face_count(), round(min(f.area() for f in _b.faces()), 4),
                                  sum(1 for e in _b.edges() if len(e.faces()) == 1))
_ca_ref = _ring_area["CIRCULAR1"][2]
check("blend / chamfer of a closed rim: full ring face (not a sliver), joined like the circular fillet",
      all(v[0] and v[1] == 3 and v[2] > 0.7 * _ca_ref and v[3] == _ring_area["CIRCULAR1"][3] for v in _ring_area.values()),
      str(_ring_area))

# --- A fillet round a closed rim is ONE smooth face: on a round part (a drafted wall) the exact surface of
#     revolution of its section (9 rational CVs round, like the wall); otherwise an even periodic degree-5 ring ---
def _skey(s_):
    d_ = ext.surface_cvs(s_)
    return (d_["count_u"], d_["count_v"], tuple(round(c_, 7) for p_ in d_["points"] for c_ in p_))
def _new_faces(o_, before_keys):
    return [f for f in store.get_brep(o_).faces() if _skey(f.surface()) not in before_keys]
def _ring_iso(s_, along, at, n=721):
    (u0_, u1_), (v0_, v1_) = s_.u_domain(), s_.v_domain()
    ts_ = np.linspace(0, 1, n)
    if along == "u":
        return np.array([s_.evaluate(u0_ + (u1_ - u0_) * t_, v0_ + (v1_ - v0_) * at) for t_ in ts_])
    return np.array([s_.evaluate(u0_ + (u1_ - u0_) * at, v0_ + (v1_ - v0_) * t_) for t_ in ts_])
def _round_dir(s_):
    """The parameter direction that runs round the ring (its iso closes on itself)."""
    for along in ("u", "v"):
        q_ = _ring_iso(s_, along, 0.5, 41)
        if np.linalg.norm(q_[0] - q_[-1]) < 1e-6 and np.ptp(q_, axis=0).max() > 1e-3:
            return along
    return None
from blender_nurbs.contcheck import edge_kind as _cc_kind
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _dc = ctx.active_object
_dc.location = (0, 1350, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_dc, "S0")), distance=2.0, direction="Z", draft=-2.0); _dw = ctx.active_object
_dtop = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([q[2] for q in geom.boundary_points(store.get_surfaces(_dw)[0], w, 8)]))
bpy.ops.nproto.planar(inputs=ins((_dw, _dtop))); _dcap = ctx.active_object
_sel(_dw, _dcap); bpy.ops.nproto.join(tolerance=1e-3); _dj = ctx.active_object
_dkeys = {_skey(f.surface()) for f in store.get_brep(_dj).faces()}
_revs = {}
for _shape, _cont in (("BLEND", "2"), ("BLEND", "3"), ("CIRCULAR", "1"), ("CHAMFER", "1")):
    _o = _dj.copy(); _o.data = _dj.data.copy(); ctx.collection.objects.link(_o)
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_o.name, f"E{i}"] for i in store.shared_edges(_o)]),
                                   shape=_shape, continuity=_cont, radius=0.3)
    _nf = _new_faces(_o, _dkeys)
    _info = {"ok": r == {'FINISHED'}, "new": len(_nf)}
    if len(_nf) == 1:
        _s = _nf[0].surface(); _d = ext.surface_cvs(_s); _rd = _round_dir(_s)
        _info["round"] = _rd
        if _rd:
            _n_round = _d["count_u"] if _rd == "u" else _d["count_v"]
            _deg_round = _d["degree_u"] if _rd == "u" else _d["degree_v"]
            _sp = []
            for _at in (0.0, 0.25, 0.5, 0.75, 1.0):
                _q = _ring_iso(_s, _rd, _at)
                _sp.append(max(float(np.ptp(np.linalg.norm(_q[:, :2] - np.array([0, 1350]), axis=1))), float(np.ptp(_q[:, 2]))))
            _info.update(cvs=_n_round, deg=_deg_round, rational=bool(_d["weights"]), spread=max(_sp))
    _info["open"] = sum(1 for i in range(len(store.get_brep(_o).edges())) if _cc_kind(_o, f"E{i}") == "open")
    _revs[_shape + _cont] = (_info, _o)
check("drafted rim, every fillet shape: ONE ring face, an exact surface of revolution (9 rational CVs round, "
      "degree 2, every iso round it a true circle to 1e-6), stitched in with no new open edge",
      all(v[0]["ok"] and v[0]["new"] == 1 and v[0].get("cvs") == 9 and v[0].get("deg") == 2 and v[0].get("rational")
          and v[0].get("spread", 1) < 1e-6 and v[0]["open"] == 1 for v in _revs.values()),
      str({k: v[0] for k, v in _revs.items()}))
_oca = _dj.copy(); _oca.data = _dj.data.copy(); ctx.collection.objects.link(_oca)
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_oca.name, f"E{i}"] for i in store.shared_edges(_oca)]),
                               shape="BLEND", continuity="2", radius=0.3, class_a=True)
_nfa = _new_faces(_oca, _dkeys)
_repa = json.loads(_oca.get("nproto_class_a", "{}"))
_da = ext.surface_cvs(_nfa[0].surface()) if len(_nfa) == 1 else None
check("drafted rim with Class-A Patches on: the same exact revolution (9 CVs round), reported as such",
      r == {'FINISHED'} and len(_nfa) == 1 and _da is not None and 9 in (_da["count_u"], _da["count_v"])
      and [g["ring"] for g in _repa.get("rings", [])] == ["revolution"] and "exact revolution" in _cas(_repa)[0],
      f"{len(_nfa)} new faces, {_da and (_da['count_u'], _da['count_v'])} · {_cas(_repa)[0]}")
_st_ca = features.stack(_oca)
_rims_ca = steps.rims_of(_oca, _st_ca[-1]["params"]["mids_local"]) if _st_ca else None
check("the fillet step records what its rim became (its panel shows it, and hides the Class-A settings: "
      "every fillet face is the ring)", _rims_ca is not None and _rims_ca["only"]
      and [g["ring"] for g in _rims_ca["rings"]] == ["revolution"], str(_rims_ca))
from blender_nurbs import contcheck as _cc0
_cc0.clear()
_bo = _revs["BLEND2"][1]
for i in store.shared_edges(_bo):
    _cc0.add(_bo, f"E{i}")
_dj2 = [_cc0.judge(_cc0.result(p_), ctx.window_manager.nproto_view) for p_ in _cc0.PAIRS]
_cc0.clear()
check("drafted rim, G2 blend ring: G2 to wall and cap all round (gap < 1e-6, < 0.01 deg, curvature within 0.5%)",
      len(_dj2) == 2 and all(isinstance(j, dict) and j["G0"][0] < 1e-6 and j["G1"][0] < 0.01 and j["G2"][0] < 0.005
                             for j in _dj2), str(_dj2))
# the tilted rim of the Class-A test without Class-A: not round, so one even periodic degree-5 ring
_pkeys = {_skey(f.surface()) for f in store.get_brep(_cs_plain).faces()}
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_cs_plain.name, f"E{i}"] for i in _top]), shape="BLEND",
                               continuity="2", radius=0.15)
_pf = _new_faces(_cs_plain, _pkeys)
_pinfo = {"ok": r == {'FINISHED'}, "new": len(_pf)}
if len(_pf) == 1:
    _s = _pf[0].surface(); _d = ext.surface_cvs(_s); _rd = _round_dir(_s)
    _kr = np.array(_d["knots_u"] if _rd == "u" else _d["knots_v"]) if _rd else np.zeros(0)
    _inner = _kr[(_kr > _kr[0]) & (_kr < _kr[-1])] if len(_kr) else _kr
    _pinfo.update(round=_rd, deg=(_d["degree_u"] if _rd == "u" else _d["degree_v"]) if _rd else None,
                  cvs=(_d["count_u"] if _rd == "u" else _d["count_v"]) if _rd else None,
                  even=bool(len(_inner) and len(np.unique(np.round(_inner, 12))) == len(_inner)
                            and np.ptp(np.diff(np.r_[_kr[0], _inner, _kr[-1]])) < 1e-9))
_cc0.clear()
_pk = _skey(_pf[0].surface()) if len(_pf) == 1 else None
for i, e_ in enumerate(store.get_brep(_cs_plain).edges()):      # the ring's rails and its seam
    if any(_skey(f_.surface()) == _pk for f_ in e_.faces()) and _cc_kind(_cs_plain, f"E{i}") in ("shared", "seam"):
        _cc0.add(_cs_plain, f"E{i}")
_pj = [_cc0.judge(_cc0.result(p_), ctx.window_manager.nproto_view) for p_ in _cc0.PAIRS]
_cc0.clear()
_pinfo["cont"] = [{k_: round(v_[0], 6) for k_, v_ in j_.items()} if isinstance(j_, dict) else j_ for j_ in _pj]
check("tilted (not round) rim, G2 blend: ONE even periodic ring (degree 5, uniform simple knots, <= 69 CVs round), "
      "G2 across its seam and to both faces",
      _pinfo["ok"] and _pinfo["new"] == 1 and _pinfo.get("deg") == 5 and _pinfo.get("even") and _pinfo.get("cvs", 999) <= 69
      and len(_pj) >= 3 and all(isinstance(j_, dict) and j_["G0"][0] < 1e-5 and j_["G1"][0] < 0.05 and j_["G2"][0] < 0.01
                                for j_ in _pj),
      str(_pinfo))

# --- Continuity Check on picked edges: one stitched edge of a joined body, or two surfaces' edges ---
from blender_nurbs import contcheck as _cc
from blender_nurbs.ops import _pick_candidates as _pc
_vw = ctx.window_manager.nproto_view
_ccres = {}
for _shape, _cont in (("CIRCULAR", "1"), ("BLEND", "2")):
    _o = _rj.copy(); _o.data = _rj.data.copy(); ctx.collection.objects.link(_o)
    bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_o.name, f"E{i}"] for i in store.shared_edges(_o)]),
                               shape=_shape, continuity=_cont, radius=0.2)
    _cc.clear()
    for i in store.shared_edges(_o):
        _cc.add(_o, f"E{i}")
    _rs = [_cc.result(p_) for p_ in _cc.PAIRS]
    _ccres[_shape] = [r_ if isinstance(r_, str) else {lv: (round(v_[0], 6), v_[1]) for lv, v_ in _cc.judge(r_, _vw).items()}
                      for r_ in _rs]
_rim = [_cc.result(p_) for p_ in _cc.PAIRS]
from blender_nurbs.kernel import load as _kl2
_cvmax = max(max(_kl2().blender_ext.surface_cvs(f.surface())["count_u"], _kl2().blender_ext.surface_cvs(f.surface())["count_v"])
             for f in store.get_brep(_o).faces())
check("continuity check: a closed rim is sampled all the way round (seam included), with exactly the Samples set, "
      "and the samples a true comb needs are recommended (4 per CV)",
      all(not isinstance(r_, str) and r_["closed"] and len(r_["ang"]) == _vw.cont_samples
          and r_["recommended"] == min(4 * _cvmax, 1200) for r_ in _rim),
      f"{[(r_['closed'], len(r_['ang']), r_['recommended']) for r_ in _rim if not isinstance(r_, str)]} cvs {_cvmax}")
_s0 = _vw.cont_samples
_vw.cont_samples = 40
_rim40 = [_cc.result(p_) for p_ in _cc.PAIRS]
_vw.cont_samples = _s0
check("continuity check: changing Samples changes the measurement (and so the comb)",
      all(not isinstance(r_, str) and len(r_["ang"]) == 40 for r_ in _rim40))
_g2 = [j for j in _ccres["BLEND"] if isinstance(j, dict)]
check("continuity check, stitched edges: circular fillet holds G1 but not G2; the G2 blend's curvature matches "
      "(within 1%) and it sits within 0.1 mm / 0.3 deg of wall and cap",
      all(isinstance(j, dict) and j["G1"][1] and not j["G2"][1] for j in _ccres["CIRCULAR"]) and len(_g2) == 2
      and all(j["G2"][0] < 0.01 and j["G0"][0] < 1e-4 and j["G1"][0] < 0.3 for j in _g2), str(_ccres))
_cc.clear()
_k1 = patch("CK_floor", (0, 1400, 0), (0, 1404, 0), "X", 2.0)
_k2 = patch("CK_wall", (2, 1400, 0), (2, 1404, 0), "Z", 2.0)
_cands = [c for c in _pc(ctx, True) if not c[1].startswith("S") and store.is_nurbs(c[0])]
_ea = [c for c in _cands if c[0] == _k1 and all(abs(p_[0] - 2.0) < 1e-4 for p_ in c[2])]
_pa = _cc.find_partner(_k1, _ea[0][1], _ea[0][2], _cands) if _ea else None
if _pa:
    _cc.add(_k1, _ea[0][1], _pa)
_r90 = _cc.result(_cc.PAIRS[-1]) if _cc.PAIRS else "no pair"
_j90 = _cc.judge(_r90, _vw) if not isinstance(_r90, str) else {}
check("continuity check, two surfaces: the partner edge is found; G0 holds, G1 fails at the 90 deg crease",
      _pa is not None and _pa[0] == _k2 and _j90 and _j90["G0"][1] and not _j90["G1"][1] and abs(_j90["G1"][0] - 90) < 1,
      f"{_pa} {_j90 or _r90}")
_seams = [f"E{i}" for i in range(len(store.get_brep(_o).edges())) if _cc.edge_kind(_o, f"E{i}") == "seam"]
_cc.clear()
for _r in _seams:
    _cc.add(_o, _r)
_sr = [_cc.result(p_) for p_ in _cc.PAIRS]
_sj = [_cc.judge(r_, _vw) if not isinstance(r_, str) else r_ for r_ in _sr]
check("continuity check, seams: a joined body's seam edges (a closed face meeting itself) are measured with one click, "
      "between the face's two ends", len(_seams) >= 2 and all(isinstance(j, dict) and j["G0"][0] < 1e-4 and j["G1"][0] < 0.5
                                                            for j in _sj),
      f"{_seams} {[j if isinstance(j, str) else {k: round(x[0], 6) for k, x in j.items()} for j in _sj]}")
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _sc = ctx.active_object
_sc.location = (0, 1500, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_sc, "S0")), distance=2.0, direction="Z"); _sw = ctx.active_object
_cands = [c for c in _pc(ctx, True) if not c[1].startswith("S") and store.is_nurbs(c[0])]
_own = [c for c in _cands if c[0] == _sw]
_hit = None
for c in _own:
    _pp = _cc.find_partner(_sw, c[1], c[2], _cands)
    if _pp is not None and _pp[0] == _sw:
        _hit = (c[1], _pp[1]); break
_cc.clear()
if _hit:
    _cc.add(_sw, _hit[0], (_sw, _hit[1]))
_ns = _cc.result(_cc.PAIRS[-1]) if _cc.PAIRS else "no pair"
_nj = _cc.judge(_ns, _vw) if not isinstance(_ns, str) else _ns
check("continuity check, untrimmed closed surface: its two ends are paired and measured end against end",
      _hit is not None and isinstance(_nj, dict) and _nj["G0"][0] < 1e-6 and _nj["G1"][0] < 1.0, f"{_hit} {_nj}")
# STEP export of that blended rim: every edge loop must close (a closed trim curve that ends ~1e-6 from
# its start is still ONE vertex), or other programs drop the face -- Plasticity lost the fillet
from blender_nurbs import step_export as _se2, step_import as _si2
_rim_step = os.path.join(tempfile.mkdtemp(), "rim.step")
_rst = _se2.export_step(_rim_step, [_o], unit="MM")
with open(_rim_step) as _fh:
    _rtxt = _fh.read()
_rres, _rrep = _si2.read_step(_rim_step, 1.0)
_bo = store.get_brep(_o)
check("step export: the blended rim's loops all close, one vertex per kernel vertex, and it reads back with every face",
      _rst["open_loops"] == [] and _rtxt.count("VERTEX_POINT") == len(list(_bo.vertices()))
      and _rrep["failed"] == 0 and _rres and _rres[0][1].face_count() == _bo.face_count(),
      f"open {_rst.get('open_loops')} verts {_rtxt.count('VERTEX_POINT')}/{len(list(_bo.vertices()))} {_rrep}")
_cc.clear(); _cc.add(_k1, _ea[0][1], _pa) if _ea and _pa else None
_vw.cont_show_checks = True
_a1 = _cc.active(ctx)
_cc.stop(ctx)
_a2 = _cc.active(ctx)
check("continuity check: its button toggles it like Zebra -- off hides the checks (kept for when it is on again)",
      _a1 and not _a2 and len(_cc.PAIRS) == 1 and not _vw.cont_show_checks)
_vw.cont_show_checks = True
_n_before = len(_cc.PAIRS)
bpy.ops.nproto.continuity_clear(index=-1)
check("continuity check: Clear All empties the list", _n_before == 1 and not _cc.PAIRS)

# --- Dynamic Section: exact plane sections with exact curvature combs ---
from blender_nurbs import sections as _sx
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0, degree="0"); _sph2 = ctx.active_object
_sph2.location = (0, 1600, 0); ctx.view_layer.update()
bpy.ops.object.select_all(action='DESELECT'); _sph2.select_set(True); ctx.view_layer.objects.active = _sph2
r = bpy.ops.nproto.section_add()
_sec = ctx.active_object
_sec.nproto_section.count = 3; _sec.nproto_section.step = 0.4
_sec.location = (0, 1600, 0)
def _sect_err(sec, ctr):
    errs, n_c = [], 0
    for sd in _sx.compute(sec):
        p0, nn = sd["plane"]
        d = float((p0 - ctr) @ nn)
        rho = math.sqrt(max(1.0 - d * d, 1e-12))
        cc = ctr + d * nn                       # the circle's centre
        for c in sd["curves"]:
            n_c += 1
            if c["K"] is None:
                return None, n_c
            for P_, K_ in zip(c["P"], c["K"]):
                k = np.linalg.norm(K_)
                inward = (cc - P_) / (np.linalg.norm(cc - P_) or 1.0)
                errs.append(max(abs(k * rho - 1.0), 1.0 - float(K_ @ inward) / (k or 1.0)))
    return (max(errs) if errs else None), n_c
_ctr = np.array([0.0, 1600.0, 0.0])
for _ax, _rot in (("Z", (0, 0, 0)), ("tilted 35 deg", (math.radians(35), math.radians(20), 0))):
    _sec.rotation_euler = _rot; ctx.view_layer.update()
    _e, _nc = _sect_err(_sec, _ctr)
    check(f"dynamic section ({_ax}): 3 planes cut a sphere in 3 circles; every comb hair has the circle's exact curvature "
          "1/sqrt(R^2-d^2) and points to its centre", r == {'FINISHED'} and _sx.is_section(_sec) and _nc == 3
          and _e is not None and _e < 1e-3, f"circles {_nc} worst rel. error {_e}")
_sec.rotation_euler = (0, 0, 0); ctx.view_layer.update()
_k0 = _sx._cache.get(_sec.name, (None,))[0]
_f0 = _sx.comb_factor(_sec.nproto_section, _sx.targets(_sec), _sx._size(_sx.targets(_sec))[1])
_sec.location.z += 0.1; ctx.view_layer.update(); _sx.compute(_sec)
_f1 = _sx.comb_factor(_sec.nproto_section, _sx.targets(_sec), _sx._size(_sx.targets(_sec))[1])
check("dynamic section: Scale works like the other combs (a sphere's curvature 1/R at Scale 1 makes a hair a fifth of "
      "its size) and stays the same while the section moves", abs(_f0 - _f1) < 1e-12
      and abs(_f0 * 1.0 - 0.2 * _sx._size(_sx.targets(_sec))[1]) < 1e-3 * _sx._size(_sx.targets(_sec))[1], f"{_f0} {_f1}")
_mid = _sx.planes(_sec)[1][0]
check("dynamic section: moving the section moves its planes (centred on it) and recomputes the sections",
      _sx._cache[_sec.name][0] != _k0 and abs(_mid.z - 0.1) < 1e-6)
_sec.location.z = 0.0; ctx.view_layer.update()
_n_sec = len([o for o in ctx.scene.objects if _sx.is_section(o)])
bpy.ops.object.select_all(action='DESELECT'); _sph2.select_set(True); ctx.view_layer.objects.active = _sph2
_on0 = _sx.section_active(ctx)
r2 = bpy.ops.nproto.section_add()
_off1 = not _sx.section_active(ctx) and _sec.hide_get()
r3 = bpy.ops.nproto.section_add()
check("dynamic section: the button toggles it like Zebra -- off hides the section, on brings the same one back "
      "selected (one per scene, never a second)",
      r2 == r3 == {'FINISHED'} and _on0 and _off1 and _sx.section_active(ctx) and not _sec.hide_get()
      and len([o for o in ctx.scene.objects if _sx.is_section(o)]) == _n_sec == 1
      and ctx.view_layer.objects.active == _sec and _sec.select_get())
r = bpy.ops.nproto.section_curves()
_scv = ctx.scene.objects.get("Section Curves")
check("dynamic section: Create Curves makes the sections real curves", r == {'FINISHED'} and _scv is not None
      and _scv.type == 'CURVE' and len(_scv.data.splines) == 3, str(_scv and len(_scv.data.splines)))

# --- Visual Clip: a Geometry Nodes modifier drops the faces past the middle plane ---
_sec.rotation_euler = (0, 0, 0); _sec.location = (0, 1600, 0); _sec.nproto_section.offset = 0.0
ctx.view_layer.update()
def _kept(o):
    dg = ctx.evaluated_depsgraph_get()
    me = o.evaluated_get(dg).to_mesh()
    M = o.matrix_world
    zs = [(M @ pl.center).z for pl in me.polygons]
    vz = [(M @ vv.co).z for vv in me.vertices]
    n = len(me.polygons)
    o.evaluated_get(dg).to_mesh_clear()
    _kept.verts = (min(vz) if vz else None, max(vz) if vz else None)
    return n, (min(zs) if zs else None), (max(zs) if zs else None)
_n_all = len(_sph2.data.polygons)
_sec.nproto_section.clip = True; ctx.view_layer.update()
_c1 = _kept(_sph2)
_sec.nproto_section.flip = True; ctx.view_layer.update()
_c2 = _kept(_sph2)
_sec.nproto_section.flip = False; _sec.nproto_section.slab = True; _sec.nproto_section.slab_width = 0.3
ctx.view_layer.update()
_c3 = _kept(_sph2)
_v3 = _kept.verts
_planes = _sx.clip_planes(_sph2)
_sec.nproto_section.slab = False; ctx.view_layer.update()
_kept(_sph2); _v1 = _kept.verts
_sec.nproto_section.slab = True; ctx.view_layer.update()
check("visual clip cuts cleanly: every vertex left is on the kept side and the cut reaches the plane exactly "
      "(no sawtooth of whole triangles); Clip Offset the same at its second plane",
      abs(_v1[1]) < 1e-5 and abs(_v3[1]) < 1e-5 and abs(_v3[0] + 0.3) < 1e-5, f"behind {_v1} slab {_v3}")
from mathutils import Vector as _V
_seg = _sx.keep_segments([_V((0, 0, -1)), _V((0, 0, 1)), _V((1, 0, 2)), _V((1, 0, 3))],
                         [(_V((0, 0, 0)), _V((0, 0, 1)), None)])
check("visual clip: B-rep edge lines are cut exactly at the plane", len(_seg) == 2 and abs(_seg[1].z) < 1e-12
      and abs(_seg[0].z + 1) < 1e-12, str(_seg))
check("visual clip: only the faces behind the middle plane stay (Flip: the other side; Clip Offset: a slab)",
      _sph2.modifiers.get(_sx.CLIP) is not None and 0 < _c1[0] < _n_all and _c1[2] <= 1e-6
      and _c2[1] >= -1e-6 and 0 < _c3[0] < _c1[0] and _c3[1] >= -0.3 - 1e-6 and _planes,
      f"all {_n_all} behind {_c1} flip {_c2} slab {_c3}")
_sec.nproto_section.slab = False; ctx.view_layer.update()
_c4 = _kept(_sph2)
bpy.ops.nproto.section_add(); ctx.view_layer.update()          # switched off: the clip pauses
_c5 = _kept(_sph2)
bpy.ops.nproto.section_add(); ctx.view_layer.update()          # on again: it clips as before
_c6 = _kept(_sph2)
check("visual clip pauses while the section is switched off and comes back with it",
      _c5[0] == _n_all and _c6[0] == _c4[0] < _n_all, f"{_c4[0]} -> off {_c5[0]} -> on {_c6[0]}")
_sec.nproto_section.clip = False; ctx.view_layer.update()
_off = _sph2.modifiers.get(_sx.CLIP) is None
_sec.nproto_section.clip = True; ctx.view_layer.update()
ctx.view_layer.objects.active = _sph2
bpy.ops.nproto.section_remove(name=_sec.name); ctx.view_layer.update(); _sx._clip_cleanup(ctx.scene)
check("visual clip: turning it off, or removing the section (Remove), removes the modifier and the section",
      _off and _sph2.modifiers.get(_sx.CLIP) is None and not any(_sx.is_section(o) for o in ctx.scene.objects) and len(_sph2.evaluated_get(ctx.evaluated_depsgraph_get()).data.polygons) == _n_all)

# --- Surface Combs: exact curvature combs on iso-curves ---
from blender_nurbs import isocombs as _ic
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.5, height=1.0, caps=False); _icy = ctx.active_object
_icy.location = (0, 1700, 0); ctx.view_layer.update()
_vw.iso_count_u = 3; _vw.iso_count_v = 3; _vw.iso_samples = 40
_runs, _k95, _isz = _ic.compute(_icy, _vw)
_round, _flat, _bad = 0, 0, []
for _P, _K in _runs:
    _k = np.linalg.norm(_K, axis=1)
    if _k.max() < 1e-6:
        _flat += 1
        continue
    _round += 1
    _ax = np.array([0.0, 1700.0])
    for _p, _kv in zip(_P, _K):
        _in = np.array([_ax[0] - _p[0], _ax[1] - _p[1], 0.0]); _in /= np.linalg.norm(_in)
        _bad.append(max(abs(np.linalg.norm(_kv) * 0.5 - 1.0), 1.0 - float(_kv @ _in) / np.linalg.norm(_kv)))
check("surface combs: on a cylinder the iso-curves around it have curvature exactly 1/r pointing to the axis, "
      "those along it none", _round == 3 and _flat == 3 and max(_bad) < 1e-4, f"round {_round} flat {_flat} err {max(_bad) if _bad else None}")

# --- Visual Clip on a straight wall (long triangles) cut by a tilted plane: the cut stays ON the surface ---
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _tc = ctx.active_object
_tc.location = (0, 1800, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_tc, "S0")), distance=2.0, direction="Z"); _tw = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _tw.select_set(True); ctx.view_layer.objects.active = _tw
bpy.ops.nproto.section_add(); _ts = ctx.active_object
_ts.location = (0, 1800, 1.0); _ts.rotation_euler = (math.radians(35), 0, 0)
_ts.nproto_section.clip = True; ctx.view_layer.update()
_dg = ctx.evaluated_depsgraph_get()
_me = _tw.evaluated_get(_dg).to_mesh()
_Mw = _tw.matrix_world
_vs = [(_Mw @ vv.co) for vv in _me.vertices]
_tw.evaluated_get(_dg).to_mesh_clear()
_nrm_t = (_ts.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
_rad = max(abs(math.hypot(p_.x, p_.y - 1800) - 1.0) for p_ in _vs)
_sd = max((p_ - _ts.matrix_world.translation).dot(_nrm_t) for p_ in _vs)
_edge = max((_Mw @ _tw.data.vertices[e.vertices[0]].co - _Mw @ _tw.data.vertices[e.vertices[1]].co).length
            for e in _tw.data.edges)
check("visual clip, tilted plane through a straight wall: the cut edge lies on the plane AND on the surface "
      "(corners move along the surface, not off it; float32 mesh at y = 1800)", _sd < 2e-4 and _rad < 2e-3,
      f"off the cylinder {_rad:.2e}, past the plane {_sd:.2e}, longest mesh edge {_edge:.3f}")
bpy.ops.nproto.section_remove(name=_ts.name)

# --- Face / Edge selection modes (picks on a body; CVs / Hulls and Continuity Check follow them) ---
from blender_nurbs import subsel as _ssel
from blender_nurbs import contcheck as _ccs
_vw = ctx.window_manager.nproto_view
_sb = _revs["BLEND2"][1]
bpy.ops.object.select_all(action='DESELECT'); _sb.select_set(True); ctx.view_layer.objects.active = _sb
_sbf = store.get_brep(_sb).faces()
_ri = next(k for k, f in enumerate(_sbf) if _skey(f.surface()) not in _dkeys)        # the ring face
_sb[_ssel.PROP] = "x"
_vw.sel_mode = {"FACE"}
_cleared = _ssel.PROP not in _sb
_ssel.set_picks(_sb, "FACE", [_ri])
_nets = store.display(_sb).get("nets", [])
_shown = [n for k, n in enumerate(_nets) if n.get("f", k) in (_ssel.faces_shown(_sb) or set())]
_pts, _tris, _polys = _ssel.face_tris(_sb, _ri)
_tri_area = float(sum(np.linalg.norm(np.cross(_pts[b] - _pts[a], _pts[c] - _pts[a])) / 2 for a, b, c in _tris))
check("Face mode: switching mode clears old picks; a picked face is kept on the body, CVs / Hulls shows only its net, "
      "and its own tessellation covers it (area within 2%)",
      _cleared and _ssel.get(_sb) == ("FACE", [_ri]) and len(_nets) == 3 and len(_shown) == 1
      and _shown[0]["f"] == _ri and abs(_tri_area - _sbf[_ri].area()) < 0.02 * _sbf[_ri].area() and len(_polys) >= 2,
      f"get {_ssel.get(_sb)} nets {len(_nets)} shown {len(_shown)} area {_tri_area:.4f} vs {_sbf[_ri].area():.4f}")
# a stored pick whose index no longer matches (a rebuild renumbered the faces) is found again by its point
_d0 = json.loads(_sb[_ssel.PROP]); _wrong = (_ri + 1) % len(_sbf)
_sb[_ssel.PROP] = json.dumps({"kind": "FACE", "items": [[_wrong, _d0["FACE"][0][1]]]})   # older files' format
_remap = _ssel.get(_sb)
_sb.select_set(False)
_desel = _ssel.get(_sb)
_sb.select_set(True)
check("Face mode: a pick survives renumbering (found again by its point) and only counts while its body is selected",
      _remap == ("FACE", [_ri]) and _desel == (None, []), f"{_remap} {_desel}")
_vw.sel_mode = {"EDGE"}
_sh = store.shared_edges(_sb)
_ssel.set_picks(_sb, "EDGE", _sh)
_cleared_f = _ssel.faces_shown(_sb) is None
_ccs.clear()
_nadd = _ccs.add_selected(ctx)
_jj = [_ccs.judge(_ccs.result(p_), _vw) for p_ in _ccs.PAIRS]
_ccs.clear()
check("Edge mode: picked edges are what Edge Fillet / Continuity Check take; the check measures them at once "
      "(the drafted G2 rim: G2 both sides)",
      _cleared_f and _ssel.selected_edges(ctx) == [(_sb, f"E{i}") for i in _sh] and _nadd == len(_sh) == 2
      and all(isinstance(j, dict) and j["G2"][1] for j in _jj), f"{_ssel.selected_edges(ctx)} {_nadd} {_jj}")
_vw.sel_mode = {"BODY"}
check("Body mode: the picks go", _ssel.PROP not in _sb and _ssel.get(_sb) == (None, []))

# --- Dome: close an open rim, G1 / G2 (exact revolution on a round rim, the rim's own curve otherwise) ---
from blender_nurbs import dome as _dome
from blender_nurbs import contcheck as _ccd
from blender_nurbs.ops import _pick_candidates as _pcd
def _top_rim(o):
    return max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([q[2] for q in geom.boundary_points(store.get_surfaces(o)[0], w, 8)]))
def _dome_cont(base, ref):
    _ccd.clear()
    cands = [c for c in _pcd(ctx, True) if not c[1].startswith("S")]
    own = next(c for c in cands if c[0] == base and c[1] == ref)
    partner = _ccd.find_partner(base, ref, own[2], cands)
    if partner is None:
        return None
    _ccd.add(base, ref, partner)
    j = _ccd.judge(_ccd.result(_ccd.PAIRS[0]), ctx.window_manager.nproto_view)
    _ccd.clear()
    return {k: float(v[0]) for k, v in j.items()}
_dres = {}
for _i, (_shape, _cont) in enumerate((("CIRCLE", "G1"), ("CIRCLE", "G2"), ("ELLIPSE", "G2"))):
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _dc = ctx.active_object
    if _shape == "ELLIPSE":
        _h = json.loads(_dc[features.PROP_HIST]); _h["params"]["elliptic"] = True; _h["params"]["radius2"] = 0.6
        _dc[features.PROP_HIST] = json.dumps(_h); features.rebuild(_dc, ctx, force=True)
    _dc.location = (4.0 * _i, 2100, 0); ctx.view_layer.update()
    bpy.ops.nproto.extrude(inputs=ins((_dc, "S0")), distance=2.0, direction="Z", draft=-8.0); _dbase = ctx.active_object
    _dref = _top_rim(_dbase)
    r = bpy.ops.nproto.dome(inputs=ins((_dbase, _dref)), dome_cont=_cont, dome_height=0.75)
    _dd = ctx.active_object
    _dcv = ext.surface_cvs(store.get_surfaces(_dd)[0])
    _dres[_shape + _cont] = {"ok": r == {'FINISHED'} and _dd.name.startswith("Dome"), "deg": (_dcv["degree_u"], _dcv["degree_v"]),
                             "n": (_dcv["count_u"], _dcv["count_v"]), "j": _dome_cont(_dbase, _dref),
                             "round": _dome.rim_data(_dbase, _dref)["round"], "obj": _dd, "base": _dbase, "ref": _dref}
_g1, _g2, _el = _dres["CIRCLEG1"], _dres["CIRCLEG2"], _dres["ELLIPSEG2"]
check("dome on a round rim (a drafted cone): an exact surface of revolution, 9 rational CVs round; G1 tangent, "
      "G2 also with the wall's curvature",
      _g1["ok"] and _g2["ok"] and _g1["round"] and _g1["deg"] == (3, 2) and _g1["n"] == (4, 9) and _g2["deg"] == (5, 2)
      and _g1["j"] and _g1["j"]["G0"] < 1e-6 and _g1["j"]["G1"] < 0.01
      and _g2["j"] and _g2["j"]["G0"] < 1e-6 and _g2["j"]["G1"] < 0.01 and _g2["j"]["G2"] < 0.005,
      str({k: (v["deg"], v["n"], v["j"]) for k, v in _dres.items()}))
check("dome on a planar rim that is not round (a drafted ellipse): on the rim's own curve, G2 to it",
      _el["ok"] and not _el["round"] and _el["j"] and _el["j"]["G0"] < 1e-5 and _el["j"]["G1"] < 0.05
      and _el["j"]["G2"] < 0.02, str(_el["j"]))
_h = json.loads(_g2["base"][features.PROP_HIST]); _h["params"]["draft"] = -3.0
_g2["base"][features.PROP_HIST] = json.dumps(_h)
features.rebuild(_g2["base"], ctx, force=True); features.rebuild(_g2["obj"], ctx)
_jj = _dome_cont(_g2["base"], _g2["ref"])
check("dome history: change the surface's draft and the dome follows (still G2)",
      _jj and _jj["G0"] < 1e-6 and _jj["G1"] < 0.01 and _jj["G2"] < 0.005 and features.history(_g2["obj"])["op"] == "dome",
      str(_jj))

# a barrel loft through three ellipses, its top rim leaning in (the fit used to stall at the loft's seam)
_bl = []
for _z, _r in ((0.0, 1.0), (1.0, 1.25), (2.0, 0.8)):
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=_r); _bc = ctx.active_object
    _h = json.loads(_bc[features.PROP_HIST]); _h["params"]["elliptic"] = True; _h["params"]["radius2"] = 0.7 * _r
    _bc[features.PROP_HIST] = json.dumps(_h); features.rebuild(_bc, ctx, force=True)
    _bc.location = (0, 2120, _z); ctx.view_layer.update()
    _bl.append(_bc)
bpy.ops.nproto.loft(inputs=ins(*[(c_, "S0") for c_ in _bl])); _bar = ctx.active_object
_bref = _top_rim(_bar)
r = bpy.ops.nproto.dome(inputs=ins((_bar, _bref)), dome_cont="G2")
_bd = ctx.active_object
_bj = _dome_cont(_bar, _bref) if r == {'FINISHED'} else None
check("dome on a barrel loft's top rim (an ellipse leaning in): built, G2 to it",
      r == {'FINISHED'} and _bd.name.startswith("Dome") and _bj and _bj["G0"] < 1e-5 and _bj["G1"] < 0.05 and _bj["G2"] < 0.02,
      str(_bj))

# draped: one even four-sided surface over the opening, trimmed to the rim (round, a flat spatula, an ellipse)
_drp = {}
for _i, (_nm, _r2, _dr) in enumerate((("round", None, -8.0), ("spatula", 0.25, 0.0), ("ellipse", 0.6, -5.0))):
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _dc = ctx.active_object
    if _r2:
        _h = json.loads(_dc[features.PROP_HIST]); _h["params"]["elliptic"] = True; _h["params"]["radius2"] = _r2
        _dc[features.PROP_HIST] = json.dumps(_h); features.rebuild(_dc, ctx, force=True)
    _dc.location = (4.0 * _i, 2140, 0); ctx.view_layer.update()
    bpy.ops.nproto.extrude(inputs=ins((_dc, "S0")), distance=2.0, direction="Z", draft=_dr); _db = ctx.active_object
    _rf = _top_rim(_db)
    r = bpy.ops.nproto.dome(inputs=ins((_db, _rf)), dome_cont="G2", dome_layout="DRAPED")
    _do = ctx.active_object
    _f = store.get_brep(_do).faces()
    _sd = ext.surface_cvs(_f[0].surface()) if _f else None
    _drp[_nm] = {"ok": r == {'FINISHED'} and _do.name.startswith("Dome"), "faces": len(_f),
                 "net": _sd and (_sd["count_u"], _sd["count_v"], _sd["degree_u"]), "j": _dome_cont(_db, _rf)}
check("draped dome: one even 20 x 20 degree-5 surface trimmed to the rim (no pole), G2 to it -- round, a flat "
      "spatula (4:1) and a drafted ellipse",
      all(v["ok"] and v["faces"] == 1 and v["net"] == (20, 20, 5) and v["j"] and v["j"]["G0"] < 1e-5
          and v["j"]["G1"] < 0.1 and v["j"]["G2"] < 0.05 for v in _drp.values()),
      str({k: (v["net"], v["j"]) for k, v in _drp.items()}))

# the G2 dome is round at its top (it used to go flat there: its top three CV rows at one height)
def _apex_k(obj, frac=0.04):
    s = store.get_surfaces(obj)[0]
    (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
    vs = np.linspace(v0, v1, 16, endpoint=False)
    def spread(u):
        return float(np.ptp(np.array([s.evaluate(u, v) for v in vs]), axis=0).max())
    ue, ui = (u1, u1 - frac * (u1 - u0)) if spread(u1) < spread(u0) else (u0, u0 + frac * (u1 - u0))
    top = np.array(s.evaluate(ue, v0))
    ks = []
    for v in vs:                                   # circles through the top, centred on the axis
        q = np.array(s.evaluate(ui, v))
        d, dz = float(np.hypot(*(q - top)[:2])), float(top[2] - q[2])
        ks.append(2.0 * dz / (d * d + dz * dz))
    return min(ks), max(ks)
r = bpy.ops.nproto.dome(inputs=ins((_g2["base"], _g2["ref"])), dome_cont="G1", dome_height=0.75)   # (its draft changed above)
_ka1, _ka2 = _apex_k(ctx.active_object), _apex_k(_g2["obj"])
r = bpy.ops.nproto.dome(inputs=ins((_el["base"], _el["ref"])), dome_cont="G1", dome_height=0.75)
_ke1, _ke2 = _apex_k(ctx.active_object), _apex_k(_el["obj"])
check("G2 dome: as round at its top as the G1 one (no flat tip) -- round rim and ellipse",
      _ka2[0] > 0.9 * _ka1[0] and _ka2[0] > 0.5 and _ke2[0] > 0.85 * _ke1[0] and _ke2[0] > 0.3,
      f"round G1 {_ka1} G2 {_ka2}; ellipse G1 {_ke1} G2 {_ke2}")

# a loft of selected curves goes along them (a selection has no order: bottom, top, middle folded back)
from blender_nurbs.ops import loft_selection_order as _lso
_ls = []
for _z, _r in ((0.0, 1.0), (2.0, 1.0), (1.0, 1.3)):
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=_r); _lc = ctx.active_object
    _lc.location = (0, 2160, _z); ctx.view_layer.update(); _ls.append(_lc)
_lo = _lso(_ls, _ls[2])
bpy.ops.nproto.loft(inputs=ins(*[(c_, "S0") for c_ in _lo])); _lf = ctx.active_object
_Ml = geom.world_matrix(_lf)
_lz = sorted(float(np.mean([(_Ml @ Vector(q))[2] for q in geom.boundary_points(store.get_surfaces(_lf)[0], w, 16)]))
             for w in ("U0", "U1", "V0", "V1"))
check("loft of a selection: along the sections (bottom, middle, top), not in the selection's order; picks keep "
      "theirs; two sections keep the active one last",
      [o.name for o in _lo] == [_ls[0].name, _ls[2].name, _ls[1].name] and abs(_lz[0]) < 1e-3 and abs(_lz[-1] - 2.0) < 1e-3
      and [o.name for o in _lso(_ls[:2], _ls[0])] == [_ls[1].name, _ls[0].name]
      and [o.name for o in _lso(_ls, _ls[2], closed=True)][-1] == _ls[2].name,
      f"{[o.name for o in _lo]} ends {_lz}")

# loft through sections of different layouts (two circles with Bezier knots, a reshaped one with uniform knots
# that starts elsewhere): the kernel's skin rose past the top section and folded back down to it, and twisted
def _cyc(name, pts, z, endpoint, s=1.0):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p_, c_ in zip(sp.points, pts): p_.co = (s * c_[0], s * c_[1], 0.0, c_[2])
    sp.order_u = 3; sp.use_cyclic_u = True; sp.use_endpoint_u = endpoint
    o_ = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o_); o_.location = (0, 2180, z); return o_
_ring = [(0, -.161, 1), (-.181, -.161, .707), (-.181, 0, 1), (-.181, .161, .707), (0, .161, 1), (.181, .161, .707), (.181, 0, 1), (.181, -.161, .707)]
_bent = [(0, -.174, 1), (-.16, -.135, .707), (-.206, 0, 1), (-.081, .135, .707), (0, .167, 1), (.081, .135, .707), (.206, 0, 1), (.16, -.135, .707)]
_secs = [_cyc("LsA", _ring, 0.0, True), _cyc("LsB", _ring, 0.087, True), _cyc("LsC", _bent, 0.188, False, 0.749)]
ctx.view_layer.update()
_lres = {}
for _aa in (True, False):
    _g = features._build_loft([(o_, "S0") for o_ in _secs], {"auto_align": _aa, "closed": False})
    (_u0, _u1), (_v0, _v1) = _g.u_domain(), _g.v_domain()
    _Z = np.array([[_g.evaluate(u, v)[2] for u in np.linspace(_u0, _u1, 41)] for v in np.linspace(_v0, _v1, 24, endpoint=False)])
    _st = []
    for u in np.linspace(_u0, _u1, 9):
        _P = np.array([_g.evaluate(u, v) for v in np.linspace(_v0, _v1, 12, endpoint=False)]); _c = _P.mean(axis=0)
        _st.append(float(np.degrees(np.arctan2(_P[0, 1] - _c[1], _P[0, 0] - _c[0]))))
    _cp = lambda q: features._k()[0].surface_closest_point(_g, tuple(map(float, q)))[0]
    _sec_err = max(float(np.linalg.norm(np.array([_cp(q).x, _cp(q).y, _cp(q).z]) - q))
                   for o_ in _secs for q in geom.sample_curve(features.input_curve(o_, "S0"), 30))
    _lres[_aa] = {"rise": float(np.diff(_Z, axis=1).min()), "top": float(_Z.max()), "twist": max(_st) - min(_st), "err": _sec_err}
check("loft through sections of different layouts: rises steadily (no overshoot past the top and back), holds "
      "every section; Auto Align starts the closed sections alike (no twist)",
      all(v["rise"] > 0 and v["top"] < 0.188 + 1e-6 and v["err"] < 1e-6 for v in _lres.values()) and _lres[True]["twist"] < 1.0
      and _lres[False]["twist"] > 15.0, str(_lres))
_whole = features.run("loft", [(o_, "S0") for o_ in _secs], {"auto_align": True, "closed": False})
check("...and stays one face: between sections of different layouts it turns a degree at the circle's quarter "
      "knots, which is no fold (at the sections it is smooth)",
      isinstance(_whole, features._k()[0].Surface), type(_whole).__name__)
# a closed section reseated: the same curve, starting elsewhere
_kc = features.input_curve(_secs[2], "S0")
_t0, _t1 = _kc.parameter_range()
_rs = geom.reseat_closed(_kc, _t0 + 0.37 * (_t1 - _t0))
_Pa, _Pb = geom.sample_curve(_kc, 300), geom.sample_curve(_rs, 300)
check("reseat a closed curve: the same curve, starting at the new point",
      np.linalg.norm(_Pb[0] - np.array(_kc.evaluate(_t0 + 0.37 * (_t1 - _t0)))) < 1e-9
      and max(np.min(np.linalg.norm(_Pa - q, axis=1)) for q in _Pb) < 0.01 * 0.2
      and abs(np.linalg.norm(np.diff(_Pa, axis=0), axis=1).sum() - np.linalg.norm(np.diff(_Pb, axis=0), axis=1).sum()) < 1e-4)
# curve evaluation a hair past the end of the range works on every kernel build (Linux / Windows refuse
# the end point by a rounding step where macOS takes it); really out of range still raises
from blender_nurbs.kernel import load as _klh
_smh = _klh()
_ce = _smh.create_curve_interp_points([(float(x), float(np.sin(x)), 0.0) for x in np.linspace(0.0, 3.0, 8)])
_ct0, _ct1 = _ce.parameter_range()
_hair_ok = np.linalg.norm(np.array(_ce.evaluate(_ct1 + 1e-13 * (_ct1 - _ct0))) - np.array(_ce.evaluate(_ct1))) < 1e-9
try:
    _ce.evaluate(_ct1 + 0.5 * (_ct1 - _ct0)); _far_raises = False
except Exception:
    _far_raises = True
check("kernel guard: Edge.point_at / Curve.evaluate are wrapped; a hair past the end evaluates the end",
      getattr(type(_ce).evaluate, "_nproto_safe", False) and getattr(_smh.Edge.point_at, "_nproto_safe", False) and _hair_ok,
      f"far past the end raises: {_far_raises}")

# --- Line: Perpendicular / Tangent end snaps, lines tangent to two circles, square / parallel directions
from blender_nurbs import linesolve as _lsv


def _circ(c, r):
    c = np.asarray(c, float)
    return _lsv.Curve(lambda t: (c + r * np.array([math.cos(t), math.sin(t), 0.0]),
                                 r * np.array([-math.sin(t), math.cos(t), 0.0]),
                                 -r * np.array([math.cos(t), math.sin(t), 0.0])), 0.0, 2 * math.pi)


_C1, _C2 = _circ((0, 0, 0), 1.0), _circ((4, 0, 0), 0.5)
_tp = _lsv.from_point(_C1, (3, 0, 0), "tan")
check("line solve: the two tangent points from a point to a circle (x = r^2 / d)", len(_tp) == 2 and all(
    abs(Q[0] - 1 / 3) < 1e-12 and abs(abs(Q[1]) - math.sqrt(8 / 9)) < 1e-12 for _, Q in _tp), str(_tp))
_seeds = [s_ for a_ in (0.0, 1.5, 3.1, 4.7) for s_ in _lsv.seeds_from(_C1, a_, _C2)]
_tt = _lsv.between(_C1, "tan", _C2, "tan", _seeds)
_exp = []                       # outer tangents: cos = (r1 - r2) / d; inner: (r1 + r2) / d
for _cth, _sg in ((0.125, 1), (0.375, -1)):
    for _sy in (1, -1):
        _sth = _sy * math.sqrt(1 - _cth * _cth)
        _exp.append((np.array([_cth, _sth, 0]), np.array([4 + _sg * 0.5 * _cth, _sg * 0.5 * _sth, 0])))
check("line solve: two circles have four lines tangent to both, all found exactly", len(_tt) == 4 and all(
    any(np.allclose(Q1, a_, atol=1e-12) and np.allclose(Q2, b_, atol=1e-12) for _, _, Q1, Q2 in _tt)
    for a_, b_ in _exp), str([(s_[2], s_[3]) for s_ in _tt]))
_pp = _lsv.between(_C1, "perp", _C2, "perp", _seeds)
_tq = _lsv.between(_C1, "tan", _C2, "perp", _seeds)
check("line solve: square to both circles runs along the line of centres; tangent to one, square to the other",
      len(_pp) == 4 and all(abs(Q1[1]) < 1e-12 and abs(Q2[1]) < 1e-12 for _, _, Q1, Q2 in _pp)
      and len(_tq) == 4 and all(abs(np.linalg.norm(Q1) - 1) < 1e-12 and abs((Q2 - Q1) @ Q1) < 1e-12
                                and abs(np.cross(Q2 - Q1, Q2 - (4, 0, 0))[2]) < 1e-12 for _, _, Q1, Q2 in _tq))
_ln = _lsv.Curve(lambda t: (np.array([-2.0, -3, 0]) + t * np.array([7.0, 4, 0]), np.array([7.0, 4, 0]), np.zeros(3)),
                 0.0, 1.0)
_ft = _lsv.from_point(_ln, (0, 0, 0), "perp")
check("line solve: the foot of the perpendicular on a line; a straight line has no tangent points",
      _ln.straight and len(_ft) == 1 and np.allclose(_ft[0][1], (0.8, -1.4, 0), atol=1e-12)
      and _lsv.from_point(_ln, (0, 0, 0), "tan") == [], str(_ft))
check("line solve: no tangent from a point off the circle's plane (a right angle still works)",
      _lsv.from_point(_C1, (3, 0, 0.5), "tan") == [] and len(_lsv.from_point(_C1, (3, 0, 0.5), "perp")) == 2)
check("line solve: square to a circle lies in its plane, square to a line in the drawing plane",
      np.allclose(np.abs(_lsv.side_direction(_C1, 0.0)), (1, 0, 0))
      and np.allclose(np.abs(_lsv.side_direction(_ln, 0.3, (0, 0, 1))), np.abs(np.array([-4.0, 7, 0]) / math.sqrt(65))))
# the add-on's own circles, through the kernel
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.6)
_KA = ctx.active_object
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.3)
_KB = ctx.active_object
_KA.location, _KB.location = (-1, 0, 0), (1, 0, 0)
ctx.view_layer.update()
_smk = features._k()[0]
_kA = _lsv.kernel_curve(features.input_curve(_KA, "S0"), _smk)
_kB = _lsv.kernel_curve(features.input_curve(_KB, "S0"), _smk)
_ktt = _lsv.between(_kA, "tan", _kB, "tan", _lsv.seeds_from(_kA, _lsv.nearest_t(_kA, (-0.9, 0.6, 0)), _kB))


def _ksin(c, t, V):
    D = c.at(t)[2]
    return float(np.linalg.norm(np.cross(D, V)) / (np.linalg.norm(D) * np.linalg.norm(V)))


check("line solve on kernel circles: both lines tangent to both on the side clicked, exact to rounding",
      len(_ktt) == 2 and all(Q1[1] > 0 and _ksin(_kA, a_, Q2 - Q1) < 1e-9 and _ksin(_kB, b_, Q2 - Q1) < 1e-9
                             for a_, b_, Q1, Q2 in _ktt), str([(s_[2], s_[3]) for s_ in _ktt]))
_Tk = _sn.collect([_KA, _KB])
_nl = _Tk.near_lines(_snap_view(), _px(-0.4, 0.0) + (5, 0), 2 * _sn.SNAP_PX)
check("snap: the curves passing near the mouse, nearest first",
      len(_nl) >= 1 and _Tk.line_info(_nl[0][0])[1][:2] == (_KA.name, "S0") and abs(_nl[0][1] - 5.0) < 0.1, str(_nl))


# the Line tool's own logic, on a stand-in (a modal operator cannot run headless)
class _FakeLine(_dt.SketchMixin):
    start_mode, line_mode = "FREE", "EDGE"


for _nm in ("_curve", "_start_curve", "_solutions", "_apply_start"):
    setattr(_FakeLine, _nm, _dt.NPROTO_OT_draw_line.__dict__[_nm])
_fl = _FakeLine()
_fl._curves, _fl._solved, _fl._lock = {}, {}, None
_fl._plane = (np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
_fl._start, _fl._start_refs = np.array([0.8, 0.0, 0.0]), None
_sol = _fl._solutions((_KA.name, "S0"), _kA)
_lab = sorted(s_[0] for s_ in _sol)
check("Line: from a free start the end snaps Perpendicular (2) and Tangent (2) to a circle",
      _lab == ["Perpendicular"] * 2 + ["Tangent"] * 2
      and all(abs(np.linalg.norm(Q - (-1, 0, 0)) - 0.6) < 1e-6 for _, _, Q in _sol), str(_lab))
_ta = _lsv.nearest_t(_kA, (-1.0, 0.6, 0))
_fl._start = _kA.at(_ta)[1]
_fl._start_refs = {"cands": [{"src": (_KA.name, "S0", "NEAREST", None)}], "i": 0}
_fl.start_mode = "TANGENT"
_sol2 = _fl._solutions((_KB.name, "S0"), _kB)
_both = [s_ for s_ in _sol2 if s_[0] == "Tangent to both"]
check("Line: a Tangent start slides along its circle to a line tangent to both circles",
      len(_both) == 2 and all(np.linalg.norm(A_ - _fl._start) > 1e-3 and A_[1] > 0 for _, A_, _ in _both)
      and any(s_[0] == "Tangent · Perpendicular" for s_ in _sol2), str([s_[0] for s_ in _sol2]))
_fl._apply_start()
_tan_ok = _fl._lock == "T" and abs(_fl._tvec[1]) < 1e-6 * np.linalg.norm(_fl._tvec)
_fl.start_mode = "PERP"
_fl._apply_start()
check("Line: a Tangent start leaves along the circle, a Perpendicular one square to it (in its plane)",
      _tan_ok and _fl._lock == "T" and _fl._tkind == "square" and np.allclose(np.abs(_fl._tvec), (0, 1, 0), atol=1e-6),
      f"{_fl._tvec}")
_fl._lock, _fl._akind = "A", "square"
check("Line: the A lock reads 'square to the line'", "square to the line" in _fl.lock_text())

# ...and through the modal code: mouse moves and clicks in the fake top view (1 unit = 250 px)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.6)
_MA = ctx.active_object
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.3)
_MB = ctx.active_object
_MA.location, _MB.location = (-1, 0, 0), (1, 0, 0)
ctx.view_layer.update()
bpy.ops.nproto.draw_line(data=json.dumps({"lines": [[[-1.5, -1.3, 0.0], [1.5, -0.7, 0.0]]], "edit": ""}))
_ML = ctx.active_object


def _ln_tool(**props):
    f = _fake(_dt.NPROTO_OT_draw_line, **dict(dict(polyline=False, start_mode="FREE", line_mode="EDGE"), **props))
    f._edit, f._lines, f._start, f._lock, f._start_refs = None, [], None, None, None
    f._reset_typing()
    f.sketch_start(fc)
    return f


def _at(f, q, px=(0, 0), **kw):
    xy = np.array(_wpx(q[0], q[1])) + px
    f.modal(fc, _ev('MOUSEMOVE', xy, 'NOTHING', **kw))
    return xy


def _click(f, q, px=(0, 0)):
    xy = _at(f, q, px)
    f.modal(fc, _ev('LEFTMOUSE', xy))


_c1, _r1, _c2, _r2 = np.array([-1.0, 0, 0]), 0.6, np.array([1.0, 0, 0]), 0.3
# a free start, the end Tangent to a circle
_S = np.array([0.0, 1.5, 0])
_d = _S - _c1
_dh, _ca = _d / np.linalg.norm(_d), _r1 / np.linalg.norm(_d)
_Qt = _c1 + _r1 * (_ca * _dh + math.sqrt(1 - _ca * _ca) * np.array([-_dh[1], _dh[0], 0]))
f = _ln_tool()
_click(f, _S)
_at(f, _Qt, (3, 2))
_lab = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('LEFTMOUSE', np.array(_wpx(*_Qt[:2])) + (3, 2)))
_L0 = f._lines[0] if f._lines else (np.zeros(3), np.zeros(3))
check("Line modal: from a free start the end snaps Tangent to a circle (exact tangent point)",
      _lab == "Tangent" and np.linalg.norm(_L0[0] - _S) < 1e-9 and np.linalg.norm(_L0[1] - _Qt) < 1e-6,
      f"{_lab} {_L0} want {_Qt}")
f.modal(fc, _ev('ESC'))
# a Tangent start on the big circle's top (a CV on the curve): it slides to the line tangent to both circles
_ct = (_r1 - _r2) / np.linalg.norm(_c2 - _c1)
_nt = np.array([_ct, math.sqrt(1 - _ct * _ct), 0])
_Q1, _Q2 = _c1 + _r1 * _nt, _c2 + _r2 * _nt
f = _ln_tool(start_mode="TANGENT")
_click(f, (-1.0, 0.6))
_lk = f._lock
_at(f, _Q2, (2, 3))
_lab = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('LEFTMOUSE', np.array(_wpx(*_Q2[:2])) + (2, 3)))
_L0 = f._lines[0] if f._lines else (np.zeros(3), np.zeros(3))
check("Line modal: a Tangent start on a circle's quarter point (a CV) leaves tangent, then slides to the line "
      "tangent to both circles", _lk == "T" and _lab == "Tangent to both" and np.linalg.norm(_L0[0] - _Q1) < 1e-6
      and np.linalg.norm(_L0[1] - _Q2) < 1e-6, f"{_lk} {_lab} {_L0} want {_Q1} {_Q2}")
f.modal(fc, _ev('ESC'))
# a free start, the end Perpendicular to a line
_S = np.array([-0.2, 1.2, 0])
_LA, _LD = np.array([-1.5, -1.3, 0]), np.array([3.0, 0.6, 0])
_foot = _LA + _LD * ((_S - _LA) @ _LD) / (_LD @ _LD)
f = _ln_tool()
_click(f, _S)
_at(f, _foot, (3, -2))
_lab = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('LEFTMOUSE', np.array(_wpx(*_foot[:2])) + (3, -2)))
_L0 = f._lines[0] if f._lines else (np.zeros(3), np.zeros(3))
check("Line modal: the end snaps Perpendicular to a line (its foot)",
      _lab == "Perpendicular" and np.linalg.norm(_L0[1] - _foot) < 1e-6, f"{_lab} {_L0} want {_foot}")   # curves are float32
f.modal(fc, _ev('ESC'))
# A / Shift+A over the line: the next line runs parallel / square to it
_dirs = {}
for _sh in (False, True):
    f = _ln_tool()
    _at(f, (-1.0, -1.2))
    f.modal(fc, _ev('A', _wpx(-1.0, -1.2), shift=_sh))
    _kind = (f._lock, getattr(f, "_akind", None))
    _click(f, (-1.5, 1.6))
    _click(f, (0.8, 1.95) if not _sh else (-1.2, 0.2))
    a_, b_ = f._lines[0] if f._lines else (np.zeros(3), np.ones(3))
    _dirs[_sh] = (_kind, (b_ - a_) / np.linalg.norm(b_ - a_))
    f.modal(fc, _ev('ESC'))
_ud = _LD / np.linalg.norm(_LD)
check("Line modal: A draws parallel to the line under the mouse, Shift+A square to it",
      _dirs[False][0] == ("A", "parallel") and abs(abs(_dirs[False][1] @ _ud) - 1) < 1e-9
      and _dirs[True][0] == ("A", "square") and abs(_dirs[True][1] @ _ud) < 1e-6, str(_dirs))

# --- Key hints: a short header, a grouped panel that wraps to the width, the mouse basics in the status bar
from blender_nurbs import hints as _hn
_meas = lambda t: (7.0 * len(t), 12.0)
_rows = [("Place", [_hn.key("0-9", "length"), _hn.key("Tab", "angle"), _hn.key("Enter", "place")]),
         ("Direction", [_hn.key("H V", "horizontal / vertical", on=True), _hn.key("X Y Z", "axis"),
                        _hn.key("Shift+X/Y/Z", "plane"), _hn.key("Shift", "15°"), _hn.key("Ctrl", "grid")]),
         ("Start", [_hn.cycle("S", ["free", "tangent", "perpendicular"], "tangent")]),
         ("Polyline", [_hn.toggle("P", "", False)]),
         ("", [_hn.note("Tangent chains are picked together; click a picked edge again to un-pick it.")])]
_wW, _hW, _opsW = _hn.layout(_rows, _meas, 2000)
_wN, _hN, _opsN = _hn.layout(_rows, _meas, 380)
_lines = lambda ops: len({round(o[2]) for o in ops if o[0] == "text"})
_right = lambda ops: max((o[1] + o[3]) if o[0] == "cap" else (o[1] + _meas(o[3])[0]) for o in ops)
check("hints: one line per group on a wide view; on a small one the rows wrap and stay inside the width",
      _lines(_opsW) == 5 and _lines(_opsN) > 5 and _right(_opsN) <= 380 - 8 + 1e-9 and _wN <= 380 and _hN > _hW,
      f"{_lines(_opsW)} / {_lines(_opsN)} lines, right edge {_right(_opsN):.0f}, {_wN:.0f} x {_hN:.0f}")
_sty = {o[3]: o[4] for o in _opsW if o[0] == "text"}
_cap = {o[5]: o[6] for o in _opsW if o[0] == "cap"}
check("hints: a cycling key shows every option with the current one lit, a toggle shows off dimmed, an active "
      "lock is lit", _sty.get("tangent") == "on" and _sty.get("free") == "off" and _sty.get("perpendicular") == "off"
      and _sty.get("off") == "off" and _sty.get("horizontal / vertical") == "on" and _cap.get("H") == "on"
      and _cap.get("X") == "cap" and _sty.get("Place") == "label", f"{_sty} {_cap}")


class _HArea:
    def __init__(self):
        self.text, self.regions = "unset", []
    def header_text_set(self, t):
        self.text = t
    def tag_redraw(self):
        pass
    def as_pointer(self):
        return 4242


_hc = _types.SimpleNamespace(area=_HArea(), workspace=None)
_head = _hn.show(_hc, "Line", ["place the end", "", "Length 2 ft"], _rows, [("MOUSE_LMB", "Place")])
_spec_ok = _hn.HUD["spec"] is not None and _hn.HUD["area"] == 4242
_hn.clear(_hc)
check("hints: the header holds only the live state (no key list); clear() drops header and panel",
      _head == "Line  ·  place the end  ·  Length 2 ft" and _spec_ok and _hc.area.text is None
      and _hn.HUD["spec"] is None, _head)
_was = _hn.collapsed()
_hn.handle(_types.SimpleNamespace(type='F1', value='PRESS', unicode=""))
_c1_ = _hn.collapsed()
_hn.handle(_types.SimpleNamespace(type='SLASH', value='PRESS', unicode="?"))
check("hints: F1 or ? collapses the panel and brings it back", _c1_ != _was and _hn.collapsed() == _was)
_reg = lambda t, x, y, w, h: _types.SimpleNamespace(type=t, x=x, y=y, width=w, height=h)
_win = _reg('WINDOW', 0, 21, 1200, 800)
_side = [_reg('TOOLS', 0, 21, 44, 800), _reg('UI', 900, 21, 300, 800)]
_ins_hidden = _hn._insets(_types.SimpleNamespace(regions=[_win] + _side + [_reg('HUD', 0, 0, 280, 287)]), _win)
_ins = _hn._insets(_types.SimpleNamespace(regions=[_win] + _side + [_reg('HUD', 54, 31, 260, 120)]), _win)
check("hints: the free width excludes the toolbar and the sidebar; a hidden redo panel (parked below the view, "
      "still sized) is ignored, a shown one is found", _ins_hidden == (44, 300, None) and _ins == (44, 300, (54, 314, 130)),
      f"{_ins_hidden} {_ins}")
_p0 = _hn.place(400, 90, 1200, _ins_hidden)
_p1 = _hn.place(400, 90, 1200, _ins)
_p2 = _hn.place(700, 90, 1200, _ins)
check("hints: the panel sits centred at the bottom of the free width; beside a shown redo panel when there is room, "
      "else above it", _p0 == (272, 10) and _p1 == (407, 10) and _p2 == (122, 140), f"{_p0} {_p1} {_p2}")
# every converted tool fills the panel: the sketch tools through their modal code
f = _ln_tool(start_mode="TANGENT")
_at(f, (1.6, 1.7))
_sp = _hn.HUD["spec"]
_grp = [r[0] for r in _sp["rows"]] if _sp else []
_start = [i for r in _sp["rows"] if r[0] == "Start" for i in r[1]] if _sp else []
check("hints: Draw Line shows its groups, the start mode lit, a short header",
      _grp == ["Type", "Direction", "Start", "Reference", "Polyline"] and _start and _start[0]["current"] == "tangent"
      and fc.area.text.startswith("Line  ·  click the start") and "Backspace" not in fc.area.text, f"{_grp} {fc.area.text}")
f.modal(fc, _ev('ESC'))
check("hints: leaving the tool clears the header and the panel", fc.area.text is None and _hn.HUD["spec"] is None)
_cvh = _fake(_dt.NPROTO_OT_draw_curve, mode="CV", degree="3", closed=False, start_cont="G1", end_cont="G2",
             edge_mode="EDGE")
_cvh._edit, _cvh._pts, _cvh._snapinfo, _cvh._frames, _cvh._start_refs = None, [], [], {}, None
_cvh.sketch_start(fc)
_cvh._status(fc)
_ends = [i for r in _hn.HUD["spec"]["rows"] if r[0] == "Ends" for i in r[1]]
check("hints: CV Curve shows start / end continuity as choices with the current ones lit",
      [e["current"] for e in _ends] == ["G1", "G2"] and fc.area.text.startswith("CV Curve"), str(_ends))
_cvh.sketch_end(fc)
from blender_nurbs import ops_trim as _otm
_efr = _otm.NPROTO_OT_edge_fillet.pick_rows(_types.SimpleNamespace(
    _picks=[], _ties=[], pick_min=1, extra_keys=_otm.NPROTO_OT_edge_fillet.extra_keys,
    extra_help=_otm.NPROTO_OT_edge_fillet.extra_help))
check("hints: a picking tool lists its own keys and its advice (Edge Fillet: Alt+click, A)",
      [r[0] for r in _efr] == ["Pick", "Also", ""] and [i["keys"] for i in _efr[1][1]] == [["Alt+LMB"], ["A"]]
      and "un-pick" in _efr[2][1][0]["note"], str(_efr))

# number-pad digits (and the top-row ones under Emulate Numpad, which arrive as number-pad keys): they change the
# view until the first click, then they type the value
f = _ln_tool()
_pass0 = _dt.SketchMixin.passthrough(f, fc, _ev('NUMPAD_3'))
_click(f, (-1.5, 1.6))
_pass1 = _dt.SketchMixin.passthrough(f, fc, _ev('NUMPAD_3'))
for _t in ('NUMPAD_3', 'NUMPAD_PERIOD', 'NUMPAD_5'):
    f.modal(fc, _ev(_t, _wpx(0.5, 1.6)))
_typed = f._buf
f.modal(fc, _ev('RET', _wpx(0.5, 1.6)))
_L0 = f._lines[0] if f._lines else (np.zeros(3), np.zeros(3))
check("Line: number-pad digits change the view before the first click, then type the length (3.5 placed)",
      _pass0 and not _pass1 and _typed == "3.5" and abs(np.linalg.norm(_L0[1] - _L0[0]) - _dt.parse_length(ctx, "3.5")) < 1e-9,
      f"{_pass0} {_pass1} '{_typed}' {_L0}")
f.modal(fc, _ev('ESC'))

# --- G2 loft to the untrimmed side of a trimmed surface (a spoon's bowl to its handle): that side is exact,
#     whole or cut short by the trim, so the loft is exact at any Samples; two exact layouts are matched exactly
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
from blender_nurbs import contlabels as _clb
_tb = [nurbs(f"TB{k}", [(x, -w, 0.0), (x, 0, 0.5 + 0.3 * (k == 1)), (x, w, 0.0)], order=3)
       for k, (x, w) in enumerate(((-2.0, 1.0), (0.0, 1.5), (2.0, 1.0)))]
bpy.ops.nproto.loft(inputs=ins(*[(c_, "S0") for c_ in _tb])); _tbw = ctx.active_object; _tbw.name = "TB_Bowl"
_tcut = poly("TB_cut", [(1.2, -3, 5), (1.2, 3, 5)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="TB_Bowl", cutters=ins((_tcut, "S0")), ref_point=(0, 0, 0.6),
                    normal=(0, 0, 1), view_dir=(0, 0, -1), projection="Z")
_e_end = _trim_edge(_tbw, lambda m: abs(m.x + 2.0) < 1e-4)                 # the first section: a whole side
_e_long = _trim_edge(_tbw, lambda m: m.y < -0.9 and m.x < 1.0)             # a long side, cut short at x = 1.2
_e_cut = _trim_edge(_tbw, lambda m: abs(m.x - 1.2) < 1e-3)                 # the trim curve itself
_iso_end, _iso_long = features.brep_edge_iso(_tbw, _e_end), features.brep_edge_iso(_tbw, _e_long)
check("a trimmed surface's untouched side is found as that side of its surface (whole, or the stretch the trim "
      "left), the trim curve is not", not store.is_natural(_tbw) and _iso_end is not None and _iso_end[2:] == (0.0, 1.0)
      and _iso_long is not None and 0.0 <= _iso_long[2] < _iso_long[3] < 1.0 and (_iso_long[3] - _iso_long[2]) > 0.7
      and features.brep_edge_iso(_tbw, _e_cut) is None, f"{_iso_end and _iso_end[1:]} {_iso_long and _iso_long[1:]}")


def _plate(name, a, b, d, direction):
    pl_ = poly(name, [a, b])
    bpy.ops.nproto.extrude(inputs=ins((pl_, "S0")), distance=d, direction=direction)
    return ctx.active_object


def _near_side(o, test):
    s_ = store.get_surfaces(o)[0]; M_ = geom.world_matrix(o)
    for e_ in ("U0", "U1", "V0", "V1"):
        if test(M_ @ Vector(geom.boundary_points(s_, e_, 3)[1])):
            return e_


_pe = _plate("TB_pe", (-5, -0.8, 0.3), (-5, 0.8, 0.3), 2.0, "X")             # up to x = -3, facing the first section
_pl = _plate("TB_pl", (-1.5, -4.5, 0.2), (0.5, -4.5, 0.2), 1.5, "Y")         # up to y = -3, facing the long side
_res = {}
for _nm, _p_, _side, _e in (("whole side", _pe, _near_side(_pe, lambda m: abs(m.x + 3) < 1e-6), _e_end),
                            ("side cut short", _pl, _near_side(_pl, lambda m: abs(m.y + 3) < 1e-6), _e_long)):
    _lay = set()
    for _smp in (5, 9, 24):
        r = bpy.ops.nproto.loft(inputs=json.dumps([[_p_.name, _side], [_tbw.name, _e]]), cont_a="G2", cont_b="G2",
                                samples=_smp)
        _lo = ctx.active_object
        _d = ext.surface_cvs(store.get_surfaces(_lo)[0])
        _lay.add((_d["degree_v"], _d["count_v"], tuple(round(k_, 9) for k_ in _d["knots_v"])))
        _m = [_clb.measure(_lo, _p_, _side, 2), _clb.measure(_lo, _tbw, _e, 2)]
        _res.setdefault(_nm, []).append((r, max(m_["angle"] for m_ in _m), max(m_["curv"] for m_ in _m),
                                         max(m_["gap"] for m_ in _m)))
    _res[_nm + " layouts"] = _lay
_Gl = np.array(ext.surface_cvs(store.get_surfaces(_lo)[0])["points"]).reshape(_d["count_u"], _d["count_v"], 3)
_gap = lambda i: float(np.mean(np.linalg.norm(_Gl[i + 1] - _Gl[i], axis=1)))
_ratio = _gap(0) / _gap(_d["count_u"] - 2)
check("G2 loft to a trimmed surface's side: its tangent rows as long as the plate's (both tangent 1) -- they were "
      "11x too short, the loft hugged the bowl and turned sharply away", 0.5 < _ratio < 2.0,
      f"first gaps {_gap(0):.4f} / {_gap(_d['count_u'] - 2):.4f}")
check("G2 loft to a trimmed bowl's untouched side (whole, and cut short by the trim): exact at any Samples -- one "
      "layout (the bowl's, the plate's line raised to it), angle and curvature at rounding",
      all(len(_res[k_ + " layouts"]) == 1 and next(iter(_res[k_ + " layouts"]))[0] == 2
          and all(r_ == {'FINISHED'} and a_ < 1e-4 and c_ < 1e-5 and g_ < 1e-9 for r_, a_, c_, g_ in _res[k_])
          for k_ in ("whole side", "side cut short")), str(_res))
# exactness of the tools: a side cut to a stretch, two sides put on one layout -- the same curves
_rng = np.random.default_rng(7)
_side3 = {"deg": 3, "knots": [0, 0, 0, 0, 0.3, 0.6, 1, 1, 1, 1], "P": _rng.normal(size=(6, 3)),
          "D": _rng.normal(size=(6, 3)), "DD": _rng.normal(size=(6, 3))}
_side2 = {"deg": 2, "knots": [0, 0, 0, 0.5, 1, 1, 1], "P": _rng.normal(size=(4, 3)), "D": _rng.normal(size=(4, 3)),
          "DD": None}
def _ev_side(sd_, key, t):
    c_ = ext.curve_from_cvs(int(sd_["deg"]), [tuple(map(float, p_)) for p_ in sd_[key]], [float(k_) for k_ in sd_["knots"]], None)
    return np.array(c_.evaluate(float(t)), float)
_sub = geom.sub_side(_side3, 0.2, 0.75)
_err_sub = max(np.linalg.norm(_ev_side(_sub, k_, t_) - _ev_side(_side3, k_, 0.2 + 0.55 * t_))
               for k_ in ("P", "D", "DD") for t_ in np.linspace(0, 1, 11))
_c2, _c3 = geom.compatible_sides(_side2, _side3)
_err_cmp = max(max(np.linalg.norm(_ev_side(_c2, k_, t_) - _ev_side(_side2, k_, t_)) for k_ in ("P", "D") for t_ in np.linspace(0, 1, 11)),
               max(np.linalg.norm(_ev_side(_c3, k_, t_) - _ev_side(_side3, k_, t_)) for k_ in ("P", "D", "DD") for t_ in np.linspace(0, 1, 11)))
check("exact sides: cut to a stretch, and two layouts (degree 2 and 3) put on one, the curves don't move",
      _err_sub < 1e-12 and _err_cmp < 1e-12 and _c2["deg"] == _c3["deg"] == 3 and _c2["knots"] == _c3["knots"]
      and _c2["DD"] is None, f"{_err_sub:.1e} {_err_cmp:.1e}")
# second derivatives at a surface's border: z = -(1 - u)^3 has z_uu = 0 at u = 1, z_uuu = 6 (the old one-sided
# difference gave the value half a step inside: -3e-5)
from blender_nurbs import square as _sqr
_cub = ext.surface_from_cvs(3, 1, [(x_, y_, z_) for x_, z_ in ((0, -1), (1 / 3, 0), (2 / 3, 0), (1, 0)) for y_ in (0, 1)],
                            [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1])
_d2 = _sqr._second(_cub, 1.0, 0.5)[2]
_d2i = _sqr._second(_cub, 0.5, 0.5)[2]
check("second derivatives at a surface's border are second-order accurate (z_uu = 0 at u = 1, -3 at u = 0.5)",
      abs(_d2[2]) < 1e-7 and abs(_d2i[2] + 3.0) < 1e-6, f"{_d2} {_d2i}")

# --- Degree, not Order: the Curve box (degree, CVs, spans; Degree buttons keep the CVs) and knot ticks on the comb
from blender_nurbs import curvetools as _ctl, diagnostics as _dgn
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
_q2 = nurbs("DG_Q2", [(0, 0, 0), (1, 1.5, 0), (3, 1.2, 0), (4.5, 0, 0)], order=3)   # degree 2, two spans
_q3 = nurbs("DG_Q3", [(0, 0, 0), (1, 1.5, 0), (2, 2.0, 0), (3, 1.5, 0), (4, 0, 0)], order=4)
_pl = poly("DG_PL", [(0, 0, 0), (2, 0, 0), (3, 1, 0)])
_i2, _i3 = _ctl.spline_info(_q2.data.splines[0]), _ctl.spline_info(_q3.data.splines[0])
check("Curve box: an Order U 3 spline reads as degree 2 with 2 spans and a note on its comb; degree 3 has none",
      (_i2["degree"], _i2["cvs"], _i2["spans"], _i2["kind"]) == (2, 4, 2, "open") and _ctl.degree_note(_i2)
      and (_i3["degree"], _i3["spans"]) == (3, 2) and _ctl.degree_note(_i3) is None, f"{_i2} {_i3}")
_mk = lambda o: _dgn.comb_knots(o, 64)[0]
_k2, _k3, _kp = _mk(_q2), _mk(_q3), _mk(_pl)
_P2, _K2 = _dgn.comb_data(_q2, 64)[0]
_kap2 = np.linalg.norm(_K2, axis=1)
_step = float(np.max(np.abs(np.diff(_kap2))))
check("comb: a tick at every knot -- orange where a degree-2 curve's curvature jumps, grey on degree 3, red at a "
      "polyline's bend; the jump is drawn as a step", [k_[2] for k_ in _k2] == ["jump"]
      and [k_[2] for k_ in _k3] == ["smooth"] and [k_[2] for k_ in _kp] == ["corner"]
      and _step > 0.05 * float(_kap2.max()), f"{[k_[2] for k_ in _k2]} {[k_[2] for k_ in _k3]} {[k_[2] for k_ in _kp]} step {_step:.3f}")
ctx.view_layer.objects.active = _q2
r = bpy.ops.nproto.curve_degree(degree=3, spline=0)
_ord = _q2.data.splines[0].order_u
r2 = bpy.ops.nproto.curve_degree(degree=4, spline=0)
check("Degree buttons: degree 3 sets Order U 4 on the same 4 CVs; degree 4 needs 5 CVs and is refused",
      r == {'FINISHED'} and _ord == 4 and len(_q2.data.splines[0].points) == 4 and r2 == {'CANCELLED'}
      and _q2.data.splines[0].order_u == 4)
_q2.data.splines[0].order_u = 3


class _RecLay:
    def __init__(self, rec): object.__setattr__(self, "_rec", rec)
    def __setattr__(self, k, v): pass
    def __getattr__(self, name):
        rec = self._rec
        def f(*a, **k):
            if name == "operator":
                e = {"op": a[0], "text": k.get("text"), "depress": k.get("depress"), "props": {}}
                rec.append(e)
                return _FakeProps(e["props"])
            if name == "label":
                rec.append({"label": k.get("text", a[0] if a else "")})
            return _RecLay(rec)
        return f


_rec = []
_ctl.draw_curve_info(_RecLay(_rec), ctx, _q2)
_labels = [e["label"] for e in _rec if "label" in e]
_degs = [e for e in _rec if e.get("op") == "nproto.curve_degree"]
_reb = [e for e in _rec if e.get("op") == "nproto.rebuild_curve"]
check("Curve box draws: degree / CVs / spans, Degree 1-5 with 2 pressed, Blender's Order U beside it, the note and "
      "Rebuild as degree 3", any("degree 2 · 4 CVs · 2 span(s)" in l_ for l_ in _labels)
      and [e["text"] for e in _degs] == ["1", "2", "3", "4", "5"] and [e["depress"] for e in _degs].index(True) == 1
      and any("Order U 3" in l_ for l_ in _labels) and any(l_.startswith("Degree 2: the curvature jumps or kinks") for l_ in _labels)
      and len(_reb) == 1 and _reb[0]["props"] == {"rc_degree": "3", "rc_cvs": 0}, str(_rec))
ctx.view_layer.objects.active = _q2
for o in ctx.selected_objects:
    o.select_set(False)
_q2.select_set(True)
r = bpy.ops.nproto.rebuild_curve(rc_degree="3", rc_cvs=0)
_kr = _dgn.comb_knots(_q2, 64)[0]
check("Rebuild as degree 3 (the box's button) leaves no curvature jump on the comb",
      r == {'FINISHED'} and _q2.data.splines[0].order_u == 4 and all(k_[2] == "smooth" for k_ in _kr),
      f"{r} order {_q2.data.splines[0].order_u} {[k_[2] for k_ in _kr]}")

# --- While drawing: the comb on the CV Curve preview (K), the sidebar usable (mouse moves pass through),
#     Emulate Numpad's top-row digits set the degree; Body / Face / Edge at the right of the tool header;
#     a G-label click while the feature's redo panel is up changes the value there
_v = ctx.window_manager.nproto_view
_comb0 = _v.show_comb
_v.show_comb = False
_cvk = _fake(_dt.NPROTO_OT_draw_curve, mode="CV", degree="2", closed=False, start_cont="G0", end_cont="G0",
             edge_mode="EDGE")
_cvk._edit, _cvk._pts, _cvk._snapinfo, _cvk._frames, _cvk._start_refs = None, [], [], {}, None
_cvk.sketch_start(fc)
for _q in ((-1.5, -1.5), (-0.5, 1.2), (0.7, 1.0), (1.6, -1.2)):
    _cvk.modal(fc, _ev('MOUSEMOVE', _wpx(*_q), 'NOTHING'))
    _cvk.modal(fc, _ev('LEFTMOUSE', _wpx(*_q)))
_cvk.modal(fc, _ev('MOUSEMOVE', _wpx(1.8, -1.8), 'NOTHING'))
_no_comb = _drw.sketch_overlay.get("comb")
_cvk.modal(fc, _ev('K', _wpx(1.8, -1.8)))
_cg = _drw.sketch_overlay.get("comb")
_show = _v.show_comb
_khint = [i for r in _hn.HUD["spec"]["rows"] if r[0] == "Show" for i in r[1]] if _hn.HUD["spec"] else []
check("CV Curve: K turns the curvature comb on the curve being drawn on (the Analyze switch), with its knot ticks",
      _no_comb is None and _show and _cg is not None and len(_cg[0]) > 20 and len(_cg[1]) > 20
      and len(_cg[2]["jump"]) >= 2 and _khint and _khint[0]["current"] == "on",
      f"{_no_comb is None} {_show} {None if _cg is None else (len(_cg[0]), {k: len(v_) for k, v_ in _cg[2].items()})}")
_out = _types.SimpleNamespace(type='MOUSEMOVE', value='NOTHING', mouse_x=1500.0, mouse_y=500.0, shift=False, ctrl=False,
                              alt=False, oskey=False, unicode="")
_inn = _types.SimpleNamespace(type='MOUSEMOVE', value='NOTHING', mouse_x=500.0, mouse_y=500.0, shift=False, ctrl=False,
                              alt=False, oskey=False, unicode="")
check("sketch tools: a mouse move over the sidebar goes to the UI (its buttons respond), one in the view does not",
      _dt.SketchMixin.passthrough(_cvk, fc, _out) and not _dt.SketchMixin.passthrough(_cvk, fc, _inn))
_emu = ctx.preferences.inputs.use_emulate_numpad
ctx.preferences.inputs.use_emulate_numpad = True
_np_mine = _dt.SketchMixin.passthrough(_cvk, fc, _ev('NUMPAD_4'))
_cvk.modal(fc, _ev('NUMPAD_4', _wpx(1.8, -1.8)))
_deg4 = _cvk.degree
ctx.preferences.inputs.use_emulate_numpad = False
_np_view = _dt.SketchMixin.passthrough(_cvk, fc, _ev('NUMPAD_4'))
ctx.preferences.inputs.use_emulate_numpad = _emu
check("CV Curve: with Emulate Numpad the top-row digits (sent as number-pad keys) set the degree; without it the "
      "number pad still changes the view", not _np_mine and _deg4 == "4" and _np_view, f"{_np_mine} {_deg4} {_np_view}")
_cvk.modal(fc, _ev('ESC'))
_v.show_comb = _comb0
check("leaving CV Curve clears the comb preview", _drw.sketch_overlay.get("comb") is None)
from blender_nurbs import subsel as _ssl
_hdr = bpy.types.VIEW3D_HT_tool_header._dyn_ui_initialize()
_dts = bpy.types.VIEW3D_HT_tool_header.draw_tool_settings
_calls = []
_orig_dts = getattr(_dts, "_nproto_orig", None)
if _orig_dts is not None:
    _dts._nproto_orig = lambda self, c: _calls.append("blender")
    _sv = _ssl.draw_tool_header
    _ssl.draw_tool_header = lambda self, c: _calls.append("nurbs")
    try:
        _dts(None, ctx)
    finally:
        _dts._nproto_orig, _ssl.draw_tool_header = _orig_dts, _sv
check("Body / Face / Edge come right after Blender's tool settings (before the spacer and Options), not at either "
      "end of the tool header", getattr(_dts, "_nproto_wrapped", False) and _calls == ["blender", "nurbs"]
      and _ssl.draw_tool_header not in _hdr, f"{_calls} {[f.__name__ for f in _hdr]}")
# the redo panel's operator: the last one run, when it made the active feature
_lft = bpy.data.objects.new("RD_loft", None)
ctx.collection.objects.link(_lft)
_lft[features.PROP_HIST] = json.dumps({"op": "loft", "refs": [], "params": {"cont_a": "G0"}})
ctx.view_layer.objects.active = _lft


class _FOpProps:
    def __init__(self):
        self.cont_a = "G0"
        self.bl_rna = _types.SimpleNamespace(properties={"cont_a": None})


_fop = _types.SimpleNamespace(bl_idname="NPROTO_OT_loft", name="Loft", properties=_FOpProps())
_fctx = _types.SimpleNamespace(window_manager=_types.SimpleNamespace(operators=[_fop]), view_layer=ctx.view_layer)
_got = _clb.redo_operator(_fctx, _lft)
_fctx2 = _types.SimpleNamespace(window_manager=_types.SimpleNamespace(
    operators=[_types.SimpleNamespace(bl_idname="VIEW3D_OT_select")]), view_layer=ctx.view_layer)
check("label click: the redo panel's operator is found only while it is the last one run and made the active "
      "feature", _got is _fop and _clb.redo_operator(_fctx2, _lft) is None)
bpy.data.objects.remove(_lft)
# _in_view: a hidden redo panel (parked below the view, still sized) does not hide the view's corner
from blender_nurbs.ops import PickMixin as _PM
_reg2 = lambda t, x, y, w, h: _types.SimpleNamespace(type=t, x=x, y=y, width=w, height=h)
_ar = _types.SimpleNamespace(regions=[_reg2('WINDOW', 0, 21, 1200, 800), _reg2('HUD', 0, 0, 280, 287)])
_ar2 = _types.SimpleNamespace(regions=[_reg2('WINDOW', 0, 21, 1200, 800), _reg2('HUD', 10, 31, 280, 287)])
_evc = _types.SimpleNamespace(mouse_x=100.0, mouse_y=100.0)
check("a click in the view's bottom-left corner reaches the tool when the redo panel is hidden, not when it shows",
      _PM._in_view(_types.SimpleNamespace(area=_ar), _evc) and not _PM._in_view(_types.SimpleNamespace(area=_ar2), _evc))

# --- Clip Start: too small for the view -> a warning with a one-click fix (Display panel, header icon)
_spc = lambda n: _types.SimpleNamespace(clip_start=n, type='VIEW_3D')
_rvp = lambda z, p='PERSP': _types.SimpleNamespace(view_distance=z, view_perspective=p)
_c_bad = _ui.clip_check(_spc(1e-6), _rvp(0.37))
check("Clip Start: 1 um at 0.37 m warns (depth steps ~8 mm) and suggests 0.2 mm; 1 mm, the default and an "
      "orthographic view don't", _c_bad is not None and abs(_c_bad["step"] - 6e-8 * 0.37 ** 2 / 1e-6) < 1e-12
      and abs(_c_bad["suggest"] - 2e-4) < 1e-12 and _ui.clip_check(_spc(1e-3), _rvp(0.37)) is None
      and _ui.clip_check(_spc(0.01), _rvp(0.37)) is None and _ui.clip_check(_spc(1e-6), _rvp(0.37, 'ORTHO')) is None,
      str(_c_bad))
_wrec = []
_wctx = _types.SimpleNamespace(space_data=_types.SimpleNamespace(type='VIEW_3D', clip_start=1e-6,
                                                                 region_3d=_rvp(0.37)), scene=ctx.scene)
_shown_w = _ui.draw_clip_warning(_FakeLay(_wrec), _wctx)
_fix = [e for e in _wrec if e.get("op") == "nproto.fix_clip_start"]
check("Clip Start warning: the Display panel shows it with a button that sets the suggested value",
      _shown_w and len(_fix) == 1 and abs(_fix[0]["props"]["value"] - 2e-4) < 1e-12, str(_wrec))

# --- Face + Edge together, picks feeding tools, gizmo clicks pass through, picked faces split off / deleted
from blender_nurbs import subsel as _sx2, ops as _opx
_st = _sx2.settle_modes
check("selection modes: Face and Edge combine (Shift+click), Body stands alone, none left is Body",
      _st({"BODY", "FACE"}, {"BODY"}) == {"FACE"} and _st({"FACE", "BODY"}, {"FACE"}) == {"BODY"}
      and _st({"FACE", "EDGE"}, {"FACE"}) == {"FACE", "EDGE"} and _st(set(), {"EDGE"}) == {"BODY"})
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
_vw.sel_mode = {"BODY"}
bpy.ops.nproto.add_primitive(kind="BOX")
_bx = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_vw.sel_mode = {"FACE"}
_sx2.set_picks(_bx, "FACE", [0])
_vw.sel_mode = {"FACE", "EDGE"}
_kept = _sx2.picks(_bx, "FACE") == [0]
_sx2.set_picks(_bx, "EDGE", [3, 5])
_all = _sx2.get_all(_bx)
_sum = _sx2.summary(_bx)
_vw.sel_mode = {"EDGE"}
check("Face + Edge: switching Edge on keeps the face picks; both kinds are kept per body (summary names both); "
      "switching Face off drops only the face picks", _kept and _all == {"FACE": [0], "EDGE": [3, 5]}
      and _sum.startswith("1 face and 2 edges picked") and _sx2.kinds(ctx) == {"EDGE"}
      and _sx2.get_all(_bx) == {"FACE": [], "EDGE": [3, 5]}, f"{_all} '{_sum}'")
# the navigation gizmo's corner (and the continuity labels) take their own clicks
_rgn = lambda t_, x, y, w, h: _types.SimpleNamespace(type=t_, x=x, y=y, width=w, height=h)
_gw = _rgn('WINDOW', 0, 21, 1200, 800)
_gar = _types.SimpleNamespace(regions=[_gw, _rgn('HEADER', 0, 795, 1200, 26), _rgn('UI', 900, 21, 300, 800)],
                              spaces=_types.SimpleNamespace(active=_types.SimpleNamespace(show_gizmo=True,
                                                                                          show_gizmo_navigate=True)))
_gctx = _types.SimpleNamespace(preferences=_types.SimpleNamespace(
    view=_types.SimpleNamespace(gizmo_size_navigate_v3d=80, mini_axis_type='GIZMO', show_navigate_ui=True),
    system=_types.SimpleNamespace(ui_scale=1.0)))
from blender_nurbs import contlabels as _clh
_h0 = _clh.hover.get("key")
_clh.hover["key"] = None
_on_ball, _on_view = _sx2._over_gizmo(_gctx, _gar, _gw, 860, 760), _sx2._over_gizmo(_gctx, _gar, _gw, 500, 400)
_clh.hover["key"] = ("x", 0)
_on_label = _sx2._over_gizmo(_gctx, _gar, _gw, 500, 400)
_clh.hover["key"] = _h0
check("Face / Edge mode: clicks on the navigation gizmo (beside the sidebar, under the header) and on continuity "
      "labels go to them; the rest of the view picks", _on_ball and not _on_view and _on_label)
# tools take the picked edges -- through their real invoke (background Blender has no window: bpy.ops would run
# execute straight away), on a stand-in like the sketch tools'
def _fake_pick(op_cls, **props):
    body = {k: v for k, v in op_cls.__dict__.items() if callable(v) and not k.startswith("bl_")
            and k not in ("__init__", "pick_candidates")}      # (its super() needs the real class)
    for k in ("accepts_picks", "pick_min", "pick_max", "finish_on_min", "bl_label", "bl_idname"):
        if hasattr(op_cls, k):
            body[k] = getattr(op_cls, k)
    bases = tuple(b for b in op_cls.__mro__[1:] if b not in (bpy.types.Operator, object) and not b.__name__.startswith("bpy"))
    f = type("FakePick" + op_cls.__name__, bases or (_opx.PickMixin,), body)()
    f.msgs, f.inputs = [], ""
    f.report = lambda t_, m: f.msgs.append(m)
    for k, v in props.items():
        setattr(f, k, v)
    return f


_ev0 = _types.SimpleNamespace(type='LEFTMOUSE', value='PRESS', alt=False, shift=False, ctrl=False, oskey=False)
_sx2.set_picks(_bx, "EDGE", [3, 5])
_n0 = len(bpy.data.objects)
_fd = _fake_pick(_opx.NPROTO_OT_dup_edge if hasattr(_opx, "NPROTO_OT_dup_edge") else
                 __import__("blender_nurbs.ops_sweep", fromlist=["x"]).NPROTO_OT_dup_edge)
r = _fd.invoke(ctx, _ev0)
_dup_made = len(bpy.data.objects) - _n0
_dup_in = json.loads(_fd.inputs or "[]")
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_sx2.set_picks(_bx, "EDGE", [3])
_fp = _fake_pick(_opx.NPROTO_OT_planar, chain=True)
r2 = _fp.invoke(ctx, _ev0)
_pl = ctx.active_object
_pl_in = json.loads(_fp.inputs or "[]")
check("tools take picked edges: Duplicate Edge makes their two curves at once, Planar fills the picked edge's "
      "planar loop (4 edges)", r == {'FINISHED'} and [x[1] for x in _dup_in] == ["E3", "E5"] and _dup_made >= 1
      and r2 == {'FINISHED'} and len(_pl_in) == 4 and _pl is not _bx and store.is_nurbs(_pl)
      and store.get_brep(_pl).face_count() == 1, f"{r} {_dup_in} {_dup_made} {r2} {_pl_in} {_fp.msgs} {_fd.msgs}")
_fake_self = _types.SimpleNamespace(pick_min=2, pick_max=32, expand_pick=lambda c: [c],
                                    _cands=_opx._pick_candidates(ctx, True))
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_sx2.set_picks(_bx, "EDGE", [3])
_one = _opx.PickMixin.picked_inputs(_fake_self, ctx)
_sx2.set_picks(_bx, "EDGE", [3, 5])
_two = _opx.PickMixin.picked_inputs(_fake_self, ctx)
check("picked edges that don't fit a tool (too few for a Loft) leave it to pick by hand",
      _one is None and _two == [(_bx, "E3"), (_bx, "E5")], f"{_one} {_two}")
# faces: Unjoin splits the picked ones off, X / Delete removes them (all of them: the body goes)
_vw.sel_mode = {"FACE"}
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_nf = store.get_brep(_bx).face_count()
_sx2.set_picks(_bx, "FACE", [0, 2])
r = bpy.ops.nproto.unjoin()
_after = store.get_brep(_bx).face_count()
_parts = [o for o in ctx.selected_objects if o is not _bx]
check("Unjoin with picked faces: just those come off (one object each), the rest stays one body",
      r == {'FINISHED'} and _nf == 6 and _after == 4 and len(_parts) == 2
      and all(store.get_brep(o).face_count() == 1 for o in _parts), f"{_nf} -> {_after}, {len(_parts)} parts")
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_sx2.set_picks(_bx, "FACE", [1])
r = bpy.ops.nproto.delete_faces()
_left = store.get_brep(_bx).face_count()
_p0 = _parts[0]
bpy.ops.object.select_all(action='DESELECT'); _p0.select_set(True); ctx.view_layer.objects.active = _p0
_sx2.set_picks(_p0, "FACE", [0])
_pname = _p0.name
r2 = bpy.ops.nproto.delete_faces()
check("Delete Faces (X / Delete in Face mode): picked faces go; a body whose faces are all picked is deleted",
      r == {'FINISHED'} and _left == 3 and r2 == {'FINISHED'} and _pname not in bpy.data.objects, f"{_left}")
_vw.sel_mode = {"BODY"}

print("ALL OK" if ok else "SOME FAILED")
