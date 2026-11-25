"""Chamfer by one distance or two (Surface Fillet and Edge Fillet): the rails
sit those distances from the crossing / edge, on each surface or face:
    blender -b --factory-startup --python tests/test_chamfer_distances.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, history_ui, store
from blender_nurbs.sweeps import _world_brep

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def poly(name, pts, cyclic=False):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def sheet(name, p0, p1, length=4.0):
    ln = poly(name + "_ln", [p0, p1])
    bpy.ops.nproto.extrude(inputs=json.dumps([[ln.name, "S0"]]), distance=length, direction="Y")
    o = ctx.active_object; o.name = name
    bpy.data.objects.remove(ln)
    return o

def prism(profile, h=2.0):
    R = poly("R", [(x, y, 0) for x, y in profile], cyclic=True)
    bpy.ops.nproto.extrude(inputs=json.dumps([[R.name, "S0"]]), distance=h, direction="Z", body="NEW")
    return ctx.active_object

def brep(o): return store.get_brep(o)

def edge_at(o, p):
    es = brep(o).edges()
    def d(e):
        t0, t1 = e.parameter_range()
        return np.linalg.norm(np.array(e.point_at(0.5 * (t0 + t1)), float) - np.asarray(p, float))
    return min(range(len(es)), key=lambda k: d(es[k]))

def face_at(o, p):
    fs = brep(o).faces()
    return fs[min(range(len(fs)), key=lambda k: fs[k].closest_point(tuple(map(float, p)))[1])]

def rail_dists(fil, on_a, on_b):
    """Distances from the crossing (the Y axis) of the fillet's boundary points
    on surface A (on_a(p)) and on B (the end lines across between: neither)."""
    A, B = [], []
    for e in _world_brep(fil).edges():
        t0, t1 = e.parameter_range()
        for k in range(1, 8):
            p = np.array(e.point_at(t0 + (t1 - t0) * k / 8.0), float)
            r = math.hypot(p[0], p[2])
            if r < 1e-6:
                continue
            if on_a(p):
                A.append(r)
            elif on_b(p):
                B.append(r)
    return A, B

def spread(v):
    return (min(v), max(v)) if v else (None, None)

# --- Surface Fillet, chamfer: one distance, then two, at 90 and 60 deg -----
for deg in (90.0, 60.0):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    for da, db in ((0.3, 0.3), (0.2, 0.5)):
        clear()
        a = sheet("Floor", (-2, 0, 0), (2, 0, 0))
        b = sheet("Wall", (-1.5 * c, 0, -1.5 * s), (2 * c, 0, 2 * s))
        r = bpy.ops.nproto.fillet(surf_a=a.name, surf_b=b.name, shape="CHAMFER", radius=da, radius_b=db,
                                  same_radius=da == db, trim_bases=True)
        fil = ctx.active_object
        nw = np.array([-s, 0.0, c])
        A, B = rail_dists(fil, lambda p: abs(p[2]) < 1e-6, lambda p: abs(p @ nw) < 1e-6) \
            if r == {'FINISHED'} else ([], [])
        (a0, a1), (b0, b1) = spread(A), spread(B)
        good = r == {'FINISHED'} and A and B and abs(a0 - da) < 1e-5 and abs(a1 - da) < 1e-5 \
            and abs(b0 - db) < 1e-5 and abs(b1 - db) < 1e-5
        hist = fil.get("nproto_hist", "")
        check(f"Surface chamfer {deg:.0f} deg, distances {da} / {db}: rails at them on A / B", good,
              f"A {a0}..{a1}, B {b0}..{b1}")
        if da != db:
            check("... recorded as distances in its history", '"setback": true' in hist)

# --- Edge Fillet on a box: one distance, two, flipped, four top edges ----------
def box():
    return prism([(0, 0), (4, 0), (4, 3), (0, 3)])

def efil(B, pts, **kw):
    seen = {}
    from blender_nurbs import jobs
    run = jobs.run
    def spy(op, *a, **k):
        res = run(op, *a, **k)
        if op == "edge_fillet":
            seen.update(res[0])
        return res
    jobs.run = spy
    try:
        r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[B.name, f"E{edge_at(B, p)}"] for p in pts]),
                                       shape="CHAMFER", **kw)
    finally:
        jobs.run = run
    return r, seen

def face_area(o, p): return face_at(o, p).area(1e-7)

clear(); B = box()
r, info = efil(B, [(2, 0, 2)], radius=0.3)
check("Edge chamfer, one distance 0.3 on a box edge: volume - 0.18", r == {'FINISHED'} and
      abs(brep(B).volume() - (24 - 0.18)) < 1e-4, f"{brep(B).volume():.5f}")

# a slanted side (the faces 116.6 deg apart): one distance is the setback on both faces
clear()
R = poly("R", [(0, 0, 0), (4, 0, 0), (3, 0, 2), (0, 0, 2)], cyclic=True)
bpy.ops.nproto.extrude(inputs=json.dumps([[R.name, "S0"]]), distance=3.0, direction="Y", body="NEW")
W = ctx.active_object
top, slant = (1.5, 1.5, 2), (3.5, 1.5, 1)
a0, s0 = face_area(W, top), face_area(W, slant)
r, info = efil(W, [(3, 1.5, 2)], radius=0.25)
check("... an edge at another angle: the distance is the setback on both faces (each loses 0.25 x 3)",
      r == {'FINISHED'} and abs((a0 - face_area(W, top)) - 0.75) < 1e-3
      and abs((s0 - face_area(W, slant)) - 0.75) < 1e-3,
      f"top lost {a0 - face_area(W, top):.5f}, slant {s0 - face_area(W, slant):.5f}")

for flip in (False, True):
    clear(); B = box()
    r, info = efil(B, [(2, 0, 2)], radius=0.2, distance_b=0.5, same_distance=False, flip_distances=flip)
    top_loss = 12 - face_area(B, (2, 1.5, 2))
    side_loss = 8 - face_area(B, (2, 0, 1))
    pair = (round(top_loss / 4, 4), round(side_loss / 4, 4))
    check(f"Edge chamfer, two distances 0.2 / 0.5 (flip {flip}): one face cut back 0.2, the other 0.5, "
          "volume - 0.2", r == {'FINISHED'} and sorted(pair) == [0.2, 0.5]
          and abs(brep(B).volume() - (24 - 0.2)) < 1e-4, f"setbacks {pair}, {brep(B).volume():.5f}")
    if not flip:
        first = pair
    else:
        check("... Flip swaps the faces", pair == first[::-1], f"{first} -> {pair}")

clear(); B = box()
TOP4 = [(2, 0, 2), (2, 3, 2), (0, 1.5, 2), (4, 1.5, 2)]
r, info = efil(B, TOP4, radius=0.2, distance_b=0.5, same_distance=False)
ta, sa = face_area(B, (2, 1.5, 2)), face_area(B, (2, 0, 1))
check("four top edges, two distances: A on the top (inset 0.2), 0.5 down the sides, mitred corners, solid",
      r == {'FINISHED'} and brep(B).is_manifold_solid() and brep(B).face_count() == 10
      and abs(ta - 3.6 * 2.6) < 1e-4 and abs(sa - 6.0) < 1e-4, f"top {ta:.5f}, side {sa:.5f}, "
      f"{brep(B).face_count()} faces")

# a concave edge (an L's inner corner): material added
clear(); Lb = prism([(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)])
v0 = brep(Lb).volume()
r, info = efil(Lb, [(1, 1, 1)], radius=0.2, distance_b=0.5, same_distance=False)
check("a concave edge, two distances: material added (+0.1), solid", r == {'FINISHED'}
      and brep(Lb).is_manifold_solid() and abs(brep(Lb).volume() - v0 - 0.1) < 1e-4,
      f"{brep(Lb).volume() - v0:.5f}")

# a joined sheet's edge (the chamfer ends at the sheets' boundary): exact sections
clear()
a = sheet("F", (-2, 0, 0), (0, 0, 0)); b = sheet("W", (0, 0, 0), (0, 0, 2))
for o in ctx.selected_objects: o.select_set(False)
a.select_set(True); b.select_set(True); ctx.view_layer.objects.active = a
bpy.ops.nproto.join()
J = ctx.active_object
r, info = efil(J, [(0, 2, 0)], radius=0.2, distance_b=0.5, same_distance=False)
fl, wl = 8 - face_area(J, (-1, 2, 0)), 8 - face_area(J, (0, 2, 1))
check("a joined sheet's edge, two distances: exact sections, faces cut back 0.2 and 0.5",
      r == {'FINISHED'} and sorted([round(fl / 4, 4), round(wl / 4, 4)]) == [0.2, 0.5],
      f"floor {fl / 4:.5f}, wall {wl / 4:.5f}, {info.get('setback')}")

# a lone box edge: Distance A on the larger face (the top)
clear(); B = box()
r, info = efil(B, [(2, 0, 2)], radius=0.2, distance_b=0.5, same_distance=False)
check("one box edge, two distances: A goes on the larger face (the top)",
      abs((12 - face_area(B, (2, 1.5, 2))) / 4 - 0.2) < 1e-4, f"top lost {(12 - face_area(B, (2, 1.5, 2))) / 4:.5f}")

# --- rims: a cylinder's, and a revolved puck's three ---------------------------
def kernel_ok(o):
    """The kernel can measure the solid (mass properties: its volume() reads 0 when not)."""
    try:
        store.get_brep(o).mass_properties()
        return True
    except Exception:
        return False

def vol(o): return store.get_brep(o).volume(1e-8)

def rim_edge(o, z, rad):
    """The shared closed edge at height z and radius rad (world = local here)."""
    es = store.get_brep(o).edges()
    def d(i):
        e = es[i]
        t0, t1 = e.parameter_range()
        q = np.array(e.point_at(0.5 * (t0 + t1)), float)
        return abs(q[2] - z) + abs(math.hypot(q[0], q[1]) - rad) + (0 if len(e.faces()) == 2 else 1e3)
    return min(range(len(es)), key=d)

def rim_chamfer(o, i, **kw):
    try:
        return bpy.ops.nproto.edge_fillet(inputs=json.dumps([[o.name, f"E{i}"]]), shape="CHAMFER", **kw)
    except RuntimeError as exc:
        return str(exc)

clear()
bpy.ops.nproto.add_primitive(kind="CYLINDER", caps=True)
C = ctx.active_object
bb = np.array([list(v) for v in C.bound_box])
R, H = float(bb[:, 0].max()), float(bb[:, 2].max())
z0 = float(bb[:, 2].min())
side0, cap0 = face_area(C, (R, 0, 0.5 * (z0 + H))), face_area(C, (0.3 * R, 0.2 * R, H))
v0 = vol(C)
r = rim_chamfer(C, rim_edge(C, H, R), radius=0.15, distance_b=0.3, same_distance=False)

def chamfer_profile_gap(o, P, Q):
    """The chamfer face's outline (its two rails and its seam) in (r, z): the
    largest distance of its points from the segment P-Q (the exact chamfer's
    profile), and how close they come to P and to Q (its rails)."""
    P, Q = np.array(P, float), np.array(Q, float)
    f = face_at(o, tuple(np.r_[0.5 * (P[0] + Q[0]), 0.0, 0.5 * (P[1] + Q[1])]))
    gap, near_p, near_q = 0.0, 1.0, 1.0
    for e in f.edges():
        t0, t1 = e.parameter_range()
        for k in range(17):
            q = np.array(e.point_at(t0 + (t1 - t0) * k / 16.0), float)
            x = np.array([math.hypot(q[0], q[1]), q[2]])
            t = float(np.clip((x - P) @ (Q - P) / ((Q - P) @ (Q - P)), 0.0, 1.0))
            gap = max(gap, float(np.linalg.norm(P + t * (Q - P) - x)))
            near_p, near_q = min(near_p, float(np.linalg.norm(x - P))), min(near_q, float(np.linalg.norm(x - Q)))
    return gap, near_p, near_q

gap = chamfer_profile_gap(C, (R, H - 0.15), (R - 0.3, H))
want_v = 0.5 * 0.15 * 0.3 * 2 * math.pi * (R - 0.3 / 3)                             # Pappus: the cut-off ring
check("cylinder rim, two distances 0.15 / 0.3: A down the side (the larger face), 0.3 off the cap -- the "
      "chamfer face is exactly that cone", r == {'FINISHED'} and max(gap) < 1e-7 and kernel_ok(C)
      and abs((v0 - vol(C)) - want_v) < 1e-5 * v0,          # (the kernel measures this cylinder to ~2e-6)
      f"profile gap / rails {gap}, volume {v0 - vol(C):.6f} (want {want_v:.6f})")

clear()
bpy.ops.nproto.add_primitive(kind="CYLINDER", caps=True)
C = ctx.active_object
v0 = vol(C)
r = rim_chamfer(C, rim_edge(C, H, R), radius=0.2, flip_distances=True)
want_v = 0.5 * 0.2 * 0.2 * 2 * math.pi * (R - 0.2 / 3)
check("cylinder rim, one distance 0.2: the exact ring", r == {'FINISHED'} and abs((v0 - vol(C)) - want_v) < 1e-5
      and kernel_ok(C), f"volume {v0 - vol(C):.6f} (want {want_v:.6f})")

# the puck (top, side, a 26.6 deg bevel, bottom): two distances on each rim made
# a body the kernel could not measure (volume 0) -- the chamfer ring's seam lies
# away from the faces' seams and splits their rails there
PUCK = [(0, 0, 0.508), (0.2032, 0, 0.508), (0.2032, 0, 0.4953), (0.1778, 0, 0.4826), (0, 0, 0.4826)]
# per rim: (r, z) of the edge, the unit (r, z) directions along face A and face B
# away from it (A: the larger face), its name
BV = np.array([0.0254, 0.0127]) / math.hypot(0.0254, 0.0127)          # up the bevel, 26.6 deg
RIMS = (((0.2032, 0.508), (-1.0, 0.0), (0.0, -1.0), "top | side"),     # A the top
        ((0.2032, 0.4953), -BV, (0.0, 1.0), "side | bevel"),           # A the bevel
        ((0.1778, 0.4826), (-1.0, 0.0), BV, "bevel | bottom"))         # A the bottom

def ring_volume(P, ua, ub, da, db):
    """Pappus: the triangle (edge, rail on A, rail on B) in (r, z) turned round the axis."""
    P, ua, ub = np.array(P), np.array(ua), np.array(ub)
    T = np.array([P, P + da * ua, P + db * ub])
    area = 0.5 * abs((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0])
    return area * 2 * math.pi * T[:, 0].mean()

for (rad, z), ua, ub, name in RIMS:
    for flip in (False, True):
        clear()
        pr = poly("Puck", PUCK)
        bpy.ops.nproto.revolve(inputs=json.dumps([[pr.name, "S0"]]), angle=360.0, axis="Z", pivot="WORLD",
                               body="NEW")
        V = ctx.active_object
        v0, n0 = vol(V), store.get_brep(V).face_count()
        r = rim_chamfer(V, rim_edge(V, z, rad), radius=0.003, distance_b=0.005, same_distance=False,
                        flip_distances=flip)
        lost = v0 - vol(V) if r == {'FINISHED'} else float("nan")
        da, db = (0.005, 0.003) if flip else (0.003, 0.005)
        cap = ring_volume((rad, z), ua, ub, da, db)
        good = (n0 == 4 and r == {'FINISHED'} and store.get_brep(V).is_manifold_solid() and kernel_ok(V)
                and store.get_brep(V).face_count() == 5 and abs(lost - cap) < 1e-3 * cap)
        extra = ""
        if name == "top | side" and good:       # A on the top (the larger face): its radius 0.003 less
            top_r = math.sqrt(face_area(V, (0.05, 0.03, 0.508)) / math.pi)
            want_top = 0.005 if flip else 0.003
            good = good and abs((rad - top_r) - want_top) < 1e-6
            lost_want = 0.5 * 0.003 * 0.005 * 2 * math.pi * (rad - want_top / 3)
            good = good and abs(lost - lost_want) < 1e-8
            extra = f", top cut back {rad - top_r:.6f}, volume {lost:.4e} (want {lost_want:.4e})"
        check(f"puck rim {name}, two distances (flip {flip}): a solid the kernel measures, its volume cut "
              "exactly by the chamfer ring", good, f"{n0} -> {store.get_brep(V).face_count()} faces, {r}, "
              f"lost {lost:.5e} (want {cap:.5e}){extra}")

# --- history: the step keeps the distances; editing them re-runs it --------------
ctx.scene.nproto_settings.construction_history = True
clear(); B = box()
r, info = efil(B, TOP4, radius=0.2, distance_b=0.5, same_distance=False)
st = features.stack(B)
k = len(st) - 1
p_ = st[k]["params"]
check("history: an edge chamfer step with its two distances", r == {'FINISHED'} and st[k]["op"] == "edge_fillet"
      and p_.get("setback") and abs(p_.get("distance_b", 0) - 0.5) < 1e-9, str({x: p_.get(x) for x in
                                                                              ("setback", "radius", "distance_b", "flip")}))
history_ui.sync_step(B, k)
check("... the step editor shows two distances", not B.nproto_step.same_distance
      and abs(B.nproto_step.distance_b - 0.5) < 1e-6)
B.nproto_step.distance_b = 0.4
sa = face_area(B, (2, 0, 1))
check("... Distance B edited in the step: the sides are cut back 0.4", abs(sa - 4 * 1.6) < 1e-4,
      f"side {sa:.5f} {B.nproto_step.error}")
B.nproto_step.flip_distances = True
ta = face_area(B, (2, 1.5, 2))
check("... Flip in the step: the top is inset 0.4, the sides cut back 0.2", abs(ta - 3.2 * 2.2) < 1e-4
      and abs(face_area(B, (2, 0, 1)) - 4 * 1.8) < 1e-4, f"top {ta:.5f} {B.nproto_step.error}")
B.nproto_step.same_distance = True
ta = face_area(B, (2, 1.5, 2))
check("... one distance again: 0.2 on both", abs(ta - 3.6 * 2.6) < 1e-4 and abs(face_area(B, (2, 0, 1)) - 4 * 1.8) < 1e-4,
      f"top {ta:.5f} {B.nproto_step.error}")

clear()
a = sheet("Floor", (-2, 0, 0), (2, 0, 0)); b = sheet("Wall", (0, 0, -1.5), (0, 0, 2))
r = bpy.ops.nproto.fillet(surf_a=a.name, surf_b=b.name, shape="CHAMFER", radius=0.2, radius_b=0.5,
                          same_radius=False, trim_bases=True)
fil = ctx.active_object
h = features.history(fil)
check("history: a surface chamfer feature with its distances", bool(h and h["params"].get("setback")
      and abs(abs(h["params"]["radius_a"]) - 0.2) < 1e-6 and abs(abs(h["params"]["radius_b"]) - 0.5) < 1e-6))
from blender_nurbs import feature_ui
wa0 = store.get_brep(b).area(1e-7)
feature_ui.sync_from_history(fil)
fil.nproto_feature.radius_b = 0.4 if fil.nproto_feature.radius_b > 0 else -0.4
A, B_ = rail_dists(fil, lambda p: abs(p[2]) < 1e-6, lambda p: abs(p[0]) < 1e-6)
wa1 = store.get_brep(b).area(1e-7)
check("... Distance B edited in the feature panel: the chamfer and the wall's trim follow (0.4)",
      B_ and abs(min(B_) - 0.4) < 1e-5 and abs(max(B_) - 0.4) < 1e-5 and abs((wa1 - wa0) - 0.1 * 4) < 1e-4,
      f"B {spread(B_)}, wall area {wa0:.5f} -> {wa1:.5f}")
ctx.scene.nproto_settings.construction_history = False

# --- the redo panels draw (one and two distances) ----------------------------
class _L:
    def __getattr__(self, n): return lambda *a, **k: self
    enabled = active = True
from blender_nurbs.ops_trim import NPROTO_OT_edge_fillet, NPROTO_OT_fillet
for cls, kw in ((NPROTO_OT_edge_fillet, {"same_distance": False}), (NPROTO_OT_edge_fillet, {}),
                (NPROTO_OT_fillet, {"same_radius": False}), (NPROTO_OT_fillet, {"variable": True})):
    class _Op:
        layout = _L()
        shape, rail_type, variable, same_distance, same_radius = "CHAMFER", "RADIUS", False, True, True
        continuity, class_a = "2", False
    o_ = _Op()
    for k_, v_ in kw.items():
        setattr(o_, k_, v_)
    for m in ("_setback", "_two", "_variable"):
        if hasattr(cls, m):
            setattr(_Op, m, getattr(cls, m))
    try:
        cls.draw(o_, ctx); drew = True
    except Exception as exc:
        drew = exc
    check(f"{cls.__name__} draws {kw or 'one distance'}", drew is True, str(drew))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
