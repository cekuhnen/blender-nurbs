"""Loft with slid ends (From / To along each side, G0 - G2 onto the surfaces):
    blender -b --factory-startup --python tests/test_loft_slide.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, geom, contlabels
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

def nurbs(name, pts, order=4):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.order_u = order; sp.use_endpoint_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

def edge_z(obj, which):
    s = store.get_surfaces(obj)[0]
    return np.mean([p[2] for p in geom.boundary_points(s, which, 8)])

def arc_point(obj, ref, f):
    """The point at share f of the edge's length (independent of the loft code)."""
    c = features._input_curve_full(obj, ref)
    t0, t1 = c.parameter_range()
    tt = np.linspace(t0, t1, 4001)
    P = np.array([c.evaluate(float(t)) for t in tt])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return np.array(c.evaluate(float(np.interp(f * s[-1], s, tt))))

def boundary_ends(lo):
    """End points of the loft's four sides (world)."""
    s = store.get_surfaces(lo)[0]
    M = np.array(geom.world_matrix(lo))
    out = {}
    for w in ("U0", "U1", "V0", "V1"):
        P = np.array(geom.boundary_points(s, w, 3))
        out[w] = [M[:3, :3] @ p + M[:3, 3] for p in (P[0], P[-1])]
    return out

def normal_angle_along(lo, src, ref, lo_f, hi_f, n=9):
    """Worst angle between the loft's and the source's normals at points of the
    source edge inside lo_f..hi_f (closest points, no loft code)."""
    sl = store.get_surfaces(lo)[0]; ss = store.get_surfaces(src)[0]
    worst = 0.0
    for f in np.linspace(lo_f + 0.05 * (hi_f - lo_f), hi_f - 0.05 * (hi_f - lo_f), n):
        p = arc_point(src, ref, f)
        res = []
        for s_ in (sl, ss):
            from blender_nurbs.features import _k
            sm, _ = _k()
            w = sm.surface_closest_point(s_, tuple(map(float, p)))[1][0]
            _, su, sv = s_.derivatives(float(w.x), float(w.y))
            nn = np.cross(su, sv); res.append(nn / np.linalg.norm(nn))
        worst = max(worst, math.degrees(math.acos(min(1.0, abs(float(res[0] @ res[1]))))))
    return worst

# two surfaces of different widths facing each other (as the blend example: the
# loft spans only a stretch of each edge)
s1c = nurbs("S1c", [(0, 0, 0), (1, 0.3, 0), (2, -0.2, 0), (3, 0.1, 0), (4, 0, 0)])
s2c = nurbs("S2c", [(1, 0, 3), (2, -0.3, 3), (3, 0.2, 3), (4.5, 0, 3)])
bpy.ops.nproto.extrude(inputs=ins((s1c, "S0")), distance=1.0, direction="Z"); S1 = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((s2c, "S0")), distance=-1.0, direction="Z"); S2 = ctx.active_object
ea = max(("U0", "U1"), key=lambda w: edge_z(S1, w))
eb = min(("U0", "U1"), key=lambda w: edge_z(S2, w))

SL = dict(a_from=0.2, a_to=0.85, b_from=0.1, b_to=0.6)
for cont in ("G0", "G1", "G2"):
    for edges in ("EDGES", "FREE"):
        r = bpy.ops.nproto.loft(inputs=ins((S1, ea), (S2, eb)), cont_a=cont, cont_b=cont,
                                edges_a=edges, edges_b=edges, **SL)
        lo = ctx.active_object
        if r != {'FINISHED'}:
            check(f"slid loft {cont} {edges}", False, "did not build"); continue
        # its corners sit at the slid points of each edge
        want = [arc_point(S1, ea, SL["a_from"]), arc_point(S1, ea, SL["a_to"])]
        # B counted from the same end as A (auto align): find which way B runs
        pa0, pa1 = arc_point(S1, ea, 0), arc_point(S1, ea, 1)
        pb0, pb1 = arc_point(S2, eb, 0), arc_point(S2, eb, 1)
        same = np.linalg.norm(pa0 - pb0) + np.linalg.norm(pa1 - pb1) <= np.linalg.norm(pa0 - pb1) + np.linalg.norm(pa1 - pb0)
        fb = (SL["b_from"], SL["b_to"]) if same else (1 - SL["b_to"], 1 - SL["b_from"])
        want += [arc_point(S2, eb, fb[0]), arc_point(S2, eb, fb[1])]
        corners = [p for ends in boundary_ends(lo).values() for p in ends]
        miss = max(min(np.linalg.norm(c - w) for c in corners) for w in want)
        labs = contlabels.labels(lo)
        lab_ok = len(labs) == 2 and all(L.get("check") and L["check"]["ok"] for L in labs)
        ang = max(normal_angle_along(lo, S1, ea, SL["a_from"], SL["a_to"]),
                  normal_angle_along(lo, S2, eb, *fb)) if cont != "G0" else 0.0
        check(f"slid loft {cont} {edges}: corners at the slid points, labels hold",
              miss < 1e-6 and lab_ok and ang < 0.1,
              f"corner miss {miss:.2e}, normal angle {ang:.3f}°, labels " +
              "; ".join(L["check"]["text"] if L.get("check") else "none" for L in labs))

# the slide is history: edited, the loft rebuilds over the new stretch
h = json.loads(lo[features.PROP_HIST])
h["params"]["a_from"] = 0.0
lo[features.PROP_HIST] = json.dumps(h)
features.edit_rebuild(lo, ctx)
corners = [p for ends in boundary_ends(lo).values() for p in ends]
miss = min(np.linalg.norm(c - arc_point(S1, ea, 0.0)) for c in corners)
labs = contlabels.labels(lo)
check("slid loft history edit: From A = 0 reaches the edge's end, labels hold",
      miss < 1e-6 and all(L["check"]["ok"] for L in labs), f"miss {miss:.2e}")

# slide handles: a dot at each end of the stretch on both sides, on the loft's corners
from blender_nurbs import loftslide
ds = loftslide.dots(lo)
corners = [p for ends in boundary_ends(lo).values() for p in ends]
dmiss = max(min(np.linalg.norm(c - np.array(d["pos"])) for c in corners) for d in ds) if ds else 1e9
check("slide handles: four dots on the loft's corners", len(ds) == 4 and dmiss < 1e-4, f"{len(ds)} dots, miss {dmiss:.2e}")
# dragging one (the operator as its redo panel runs it): the history and the loft follow
r = bpy.ops.nproto.loft_slide(obj_name=lo.name, key="b_to", value=0.45)
hp = features.history(lo)["params"]
ds = loftslide.dots(lo)
corners = [p for ends in boundary_ends(lo).values() for p in ends]
dmiss = max(min(np.linalg.norm(c - np.array(d["pos"])) for c in corners) for d in ds)
labs = contlabels.labels(lo)
check("slide handle drag: To B 0.45 in the history, the loft rebuilt onto the dots, labels hold",
      r == {'FINISHED'} and abs(hp["b_to"] - 0.45) < 1e-6 and dmiss < 1e-4 and all(L["check"]["ok"] for L in labs),
      f"{hp.get('b_to')} miss {dmiss:.2e}")
r = bpy.ops.nproto.loft_slide(obj_name=lo.name, key="b_from", value=0.9)       # past To B: stops short of it
hp = features.history(lo)["params"]
check("slide handle stays on its side of the other end", r == {'FINISHED'} and hp["b_from"] < hp["b_to"] - 0.005,
      f"from {hp['b_from']:.3f} to {hp['b_to']:.3f}")
# a loft with rails / three sections has no handles
_pc = [nurbs(f"Sec{i}", [(0, 0, 4 + i), (1, 0.2, 4 + i), (2, 0, 4 + i)], order=3) for i in range(3)]
bpy.ops.nproto.loft(inputs=ins(*[(c, "S0") for c in _pc]))
check("no slide handles on a loft through three sections", loftslide.dots(ctx.active_object) == [])

# not slid: as before (whole edges)
r = bpy.ops.nproto.loft(inputs=ins((S1, ea), (S2, eb)), cont_a="G2", cont_b="G2")
lo = ctx.active_object
corners = [p for ends in boundary_ends(lo).values() for p in ends]
miss = min(np.linalg.norm(c - arc_point(S1, ea, 0.0)) for c in corners)
check("loft not slid spans the whole edges", r == {'FINISHED'} and miss < 1e-6
      and "a_from" not in features.history(lo)["params"], f"miss {miss:.2e}")

# natural edges, one slid end at the edge's corner
def poly(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o
floor_c = nurbs("Fc", [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
bpy.ops.nproto.extrude(inputs=ins((floor_c, "S0")), distance=2.0, direction="Y"); F = ctx.active_object
wall_c = nurbs("Wc", [(0, 3.5, 1), (1, 3.5, 1.2), (2, 3.5, 0.9), (3, 3.5, 1)])
bpy.ops.nproto.extrude(inputs=ins((wall_c, "S0")), distance=1.0, direction="Z"); W = ctx.active_object
eF = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([p[1] for p in geom.boundary_points(store.get_surfaces(F)[0], w, 8)]))
eW = min(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([p[2] for p in geom.boundary_points(store.get_surfaces(W)[0], w, 8)]))
for cont in ("G1", "G2"):
    r = bpy.ops.nproto.loft(inputs=ins((F, eF), (W, eW)), cont_a=cont, cont_b=cont,
                            a_from=0.3, a_to=1.0, b_from=0.0, b_to=0.7)
    lo = ctx.active_object
    labs = contlabels.labels(lo) if r == {'FINISHED'} else []
    check(f"slid loft {cont}, one end at the edge's corner: labels hold",
          r == {'FINISHED'} and len(labs) == 2 and all(L["check"]["ok"] for L in labs),
          "; ".join(L["check"]["text"] for L in labs))

# trimmed (B-rep) edges: a straight trim along the face's lines, and a wavy one
from mathutils import Vector
def _trim_edge(o, test):
    b = store.get_brep(o); M = geom.world_matrix(o)
    for i, e in enumerate(b.edges()):
        t0, t1 = e.parameter_range(); m = M @ Vector(e.point_at(0.5 * (t0 + t1)))
        if test(m):
            return f"E{i}"
def floor(name, y0):
    ln = poly(name + "_ln", [(0, y0 - 2, 0), (0, y0 + 2, 0)])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=4.0, direction="X")
    o = ctx.active_object; o.name = name; return o
for kind, cut in (("straight", [(1.5, 15, 0), (1.5, 25, 0)]),
                  ("wavy", [(1.5, 35, 0), (1.3, 38, 0), (1.8, 41, 0), (1.5, 45, 0)])):
    y0 = cut[0][1] + 5
    lf = floor(f"TF_{kind}", y0)
    c1 = nurbs(f"TC_{kind}", cut) if kind == "wavy" else poly(f"TC_{kind}", cut)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.nproto.trim(action="KEEP", target=lf.name, cutters=ins((c1, "S0")), ref_point=(0.5, y0, 0),
                        normal=(0, 0, 1), view_dir=(0, 0, -1), projection="Z")
    wl = poly(f"TW_{kind}", [(3, y0 - 2, 1), (3, y0 + 2, 1)])
    bpy.ops.nproto.extrude(inputs=ins((wl, "S0")), distance=2.0, direction="Z"); lw = ctx.active_object
    c2 = poly(f"TC2_{kind}", [(2.0, y0 - 5, 1.5), (2.0, y0 + 5, 1.5)])
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.nproto.trim(action="KEEP", target=lw.name, cutters=ins((c2, "S0")), ref_point=(3, y0, 2.5),
                        normal=(1, 0, 0), view_dir=(1, 0, 0), projection="X")
    ea_ = _trim_edge(lf, lambda m: 1.2 < m.x < 1.9 and abs(m.z) < 1e-6)
    eb_ = _trim_edge(lw, lambda m: abs(m.z - 1.5) < 1e-4)
    for cont in ("G1", "G2"):
        try:
            r = bpy.ops.nproto.loft(inputs=json.dumps([[lf.name, ea_], [lw.name, eb_]]), cont_a=cont, cont_b=cont,
                                    a_from=0.2, a_to=0.7, b_from=0.35, b_to=0.9)
        except RuntimeError as exc:
            r = str(exc)
        lo = ctx.active_object
        labs = contlabels.labels(lo) if r == {'FINISHED'} else []
        span = None
        if labs:
            ys = [p[1] for ends in boundary_ends(lo).values() for p in ends if abs(p[2]) < 1e-6]
            span = max(ys) - min(ys)
        check(f"slid loft {cont} between {kind} trimmed edges: labels hold, spans a stretch",
              r == {'FINISHED'} and len(labs) == 2 and all(L["check"]["ok"] for L in labs) and span is not None
              and span < 3.0, f"{ea_} {eb_} span {span}; " + "; ".join(L["check"]["text"] for L in labs if L.get("check")))

# a side of two edges end to end, slid across the joint
c1 = nurbs("C1", [(0, 0, 0), (1, 0.2, 0), (2, 0, 0)], order=3)
c2 = nurbs("C2", [(2, 0, 0), (3, -0.2, 0), (4, 0, 0)], order=3)
bpy.ops.nproto.extrude(inputs=ins((c1, "S0")), distance=1.0, direction="Z"); E1 = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((c2, "S0")), distance=1.0, direction="Z"); E2 = ctx.active_object
e1 = max(("U0", "U1"), key=lambda w: edge_z(E1, w))
e2 = max(("U0", "U1"), key=lambda w: edge_z(E2, w))
top = nurbs("Top", [(0.5, 1.0, 2.5), (2, 1.2, 2.6), (3.5, 1.0, 2.5)])
r = bpy.ops.nproto.loft(inputs=ins((E1, e1), (E2, e2), (top, "S0")), sides=json.dumps([2, 1]),
                        cont_a="G1", a_from=0.3, a_to=0.8)
lo = ctx.active_object
labs = contlabels.labels(lo) if r == {'FINISHED'} else []
xs = sorted(p[0] for ends in boundary_ends(lo).values() for p in ends if abs(p[2] - 1.0) < 1e-6)
check("slid loft over a chained side (across its joint): G1 holds, ends at 0.3 / 0.8 of it",
      r == {'FINISHED'} and labs and labs[0]["check"]["ok"] and xs and xs[0] > 1.0 and xs[-1] < 3.4,
      f"x {xs[0] if xs else None:.3f}..{xs[-1] if xs else None:.3f}; " + "; ".join(L["check"]["text"] for L in labs if L.get("check")))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
