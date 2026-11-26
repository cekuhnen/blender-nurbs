"""Loft G1 / G2 onto a slot's extruded edge: the slot's line-to-arc joints
(curvature jumps, also inside one joined edge) get span knots, so G1 / G2
hold; the closest-point search finds points just before a closed loft's seam:
    blender -b --factory-startup --python tests/test_loft_bends.py"""
import json, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, geom, contlabels
from blender_nurbs.square import _foot

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

W = 0.7071067811865476

def slot_cyclic(name, half, r, z, bezier=False):
    """A slot as Blender's one cyclic order-3 spline (rational; with Bezier
    knots its arcs are exact quarter circles, without them a C1 curve whose
    curvature jumps at most knots)."""
    a = half
    pts = [(-a, -r, 1), (0, -r, 1), (a, -r, 1), (a + r, -r, W), (a + r, 0, 1), (a + r, r, W),
           (a, r, 1), (0, r, 1), (-a, r, 1), (-a - r, r, W), (-a - r, 0, 1), (-a - r, -r, W)]
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p, (x, y, w) in zip(sp.points, pts):
        p.co = (x, y, z, w)
    sp.order_u = 3; sp.use_cyclic_u = True; sp.use_endpoint_u = True; sp.use_bezier_u = bezier
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    return o

def slot_pieces(name, half, r, z):
    """The same slot as four splines: two lines, two half-circle arcs."""
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    def spline(pts, order):
        sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
        for p, (x, y, w) in zip(sp.points, pts):
            p.co = (x, y, z, w)
        sp.order_u = order; sp.use_endpoint_u = True
    a = half
    spline([(-a, r, 1), (-a - r, r, W), (-a - r, 0, 1), (-a - r, -r, W), (-a, -r, 1)], 3)
    spline([(a, -r, 1), (a + r, -r, W), (a + r, 0, 1), (a + r, r, W), (a, r, 1)], 3)
    spline([(-a, r, 1), (a, r, 1)], 2)
    spline([(-a, -r, 1), (a, -r, 1)], 2)
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    return o

def extrude(prof, refs, h):
    ctx.view_layer.objects.active = prof
    r = bpy.ops.nproto.extrude(inputs=json.dumps([[prof.name, x] for x in refs]), distance=h, direction="Z")
    return ctx.active_object if r == {'FINISHED'} else None

def inner_kink(surf):
    """Largest normal jump (degrees) across the loft's own knot lines along
    the side (a zebra stripe's kink inside the loft)."""
    from blender_nurbs.surfeval import Surf
    V = np.array(Surf(surf).V)
    worst = 0.0
    for t in sorted(set(np.round(V, 12))):
        if not 0.0 < t < 1.0:
            continue
        for u in np.linspace(0.0, 1.0, 11):
            n1, n2 = np.array(surf.normal(u, t - 1e-9)), np.array(surf.normal(u, t + 1e-9))
            c = abs(float(n1 @ n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2))
            worst = max(worst, float(np.degrees(np.arccos(min(1.0, c)))))
    return worst

def low_closed_edge(E):
    s = store.get_surfaces(E)[0]
    M = np.array(E.matrix_world)
    best = None
    for e in ("U0", "U1", "V0", "V1"):
        P = [M[:3, :3] @ np.array(p) + M[:3, 3] for p in geom.boundary_points(s, e, 9)]
        if np.linalg.norm(P[0] - P[-1]) < 1e-6 and (best is None or P[0][2] < best[1]):
            best = (e, P[0][2])
    return best[0]

# 1. one cyclic slot to the extrude of a bigger one (the reported file, its
# slots with Bezier knots; and with uniform knots): G1 / G2 hold, at the
# labels and between their samples, and the loft has no kink inside
for bez, n_bends, max_spans in ((True, 3, 16), (False, 7, 32)):
    kind = "Bezier knots" if bez else "uniform knots"
    A = slot_cyclic("SlotA", 0.05, 0.05, 0.0, bez)
    S = slot_cyclic("SlotB", 0.076, 0.05, 0.06, bez)
    E = extrude(S, ["S0"], 0.12)
    check(f"extrude of the cyclic slot ({kind})", E is not None)
    eb = low_closed_edge(E)
    bends = features.side_bend_points([(E, eb)])
    check(f"bends ({kind}): the curvature jumps found inside the one edge", len(bends) == n_bends, f"{len(bends)}")
    for cont in ("G1", "G2"):
        features.LOFT_NOTES.clear()
        ctx.view_layer.objects.active = A
        r = bpy.ops.nproto.loft(inputs=json.dumps([[A.name, "S0"], [E.name, eb]]), cont_a="G0", cont_b=cont, fit="SPANS")
        lo = ctx.active_object
        labs = contlabels.labels(lo) if r == {'FINISHED'} else []
        held = bool(labs) and all(L.get("check") and L["check"]["ok"] for L in labs)
        fine = contlabels.measure(lo, E, eb, 2, n=401) if labs else {"angle": 9.0}
        kink = inner_kink(list(store.get_brep(lo).faces())[0].surface()) if labs else 9.0
        check(f"cyclic slot loft ({kind}) {cont}: labels hold, 0.01° at 401 points, no kink inside",
              held and not features.LOFT_NOTES and (features.LAST_SPANS[0] or 99) <= max_spans
              and fine["angle"] < 0.01 and kink < 0.02,
              "; ".join(L["check"]["text"] for L in labs if L.get("check")) +
              f" | 401: {fine['angle']:.4f}°, inner kink {kink:.4f}°, spans {features.LAST_SPANS[0]} {features.LOFT_NOTES}")

# 2. the slot as four pieces, extruded into one joined edge; the loft's side A
# is the four pieces: G1 / G2 hold, and no gap is read at the closed chain's seam
P4 = slot_pieces("Slot4", 0.076, 0.05, 0.0)
P4b = slot_pieces("Slot4b", 0.076, 0.05, 0.06)
E4 = extrude(P4b, ["S0", "S1", "S2", "S3"], 0.12)
check("extrude of the four pieces: one face", E4 is not None and len(list(store.get_brep(E4).faces())) == 1)
e4 = low_closed_edge(E4)
side_a = [(P4, r) for r in ("S0", "S2", "S1", "S3")]
base = dict(cont_a="G0", tan_a=1.0, tan_b=1.0, cur_a=0.0, cur_b=0.0, auto_align=True, flip_b=False,
            ruled=False, closed=False, samples=24, edges_a="EDGES", edges_b="EDGES", fit="SPANS",
            span_degree=5, span_faces=False, sides=[4, 1])
check("four-piece side and the joined edge pair their joints",
      features.paired_samples([side_a, [(E4, e4)]], 120, False) is not None)
for cont in ("G1", "G2"):
    features.LOFT_NOTES.clear()
    g = features.build_loft(side_a + [(E4, e4)], dict(base, cont_b=cont))
    worst, what = features._verify_sides(g, [side_a, [(E4, e4)]], ("G0", cont))
    kink = inner_kink(list(store.to_brep(g).faces())[0].surface())
    check(f"four-piece slot loft {cont}: holds (gap at the seam read right), no kink inside",
          worst <= 1.0 and not features.LOFT_NOTES and kink < 0.02,
          f"{worst:.2f} {what}, inner kink {kink:.4f}° {features.LOFT_NOTES}")

# 3. the closest point just before a closed loft's seam (v = 1 lies on v = 0)
surf = list(store.to_brep(g).faces())[0].surface()
q = np.array(surf.evaluate(0.0, 0.985))
uv = _foot(surf, q, None)
gap = float(np.linalg.norm(np.array(surf.evaluate(*uv)) - q))
check("closest point just before the seam: on the far side of it", gap < 1e-9, f"gap {gap:.2e}, uv {uv}")

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
