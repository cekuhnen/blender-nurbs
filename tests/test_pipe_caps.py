"""Pipe: Cap None / Flat / Round, Radius or Diameter, history:
    blender -b --factory-startup --python tests/test_pipe_caps.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store

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
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ins(*pairs):
    return json.dumps([[o.name, r] for o, r in pairs])

def solid(o):
    try:
        return store.get_brep(o).is_manifold_solid()
    except Exception:
        return False

def vol(o):
    return abs(float(store.get_brep(o).volume(1e-6)))

def pipe(rail, **kw):
    r = bpy.ops.nproto.pipe(inputs=ins((rail, "S0")), **kw)
    return r, ctx.active_object

ctx.scene.nproto_settings.construction_history = True
clear()
r0, L = 0.5, 3.0
rod = math.pi * r0 * r0 * L
ball = 4 / 3 * math.pi * r0 ** 3
line = poly("L", [(0, 0, 0), (0, 0, L)])

r, o = pipe(line, radius=r0)
check("cap none: open tube", r == {'FINISHED'} and not solid(o))
r, o = pipe(line, radius=r0, cap_style="FLAT")
check("cap flat: closed body", r == {'FINISHED'} and solid(o) and abs(vol(o) - rod) < 1e-4, f"{vol(o):.5f} want {rod:.5f}")
r, o = pipe(line, radius=r0, cap_style="ROUND")
b = store.get_brep(o)
zs = [p[2] for f in b.faces() for p in [f.interior_point()]]
check("cap round: closed body, rod + ball", r == {'FINISHED'} and solid(o) and abs(vol(o) - (rod + ball)) < 1e-4
      and b.face_count() == 3, f"{vol(o):.5f} want {rod + ball:.5f}, {b.face_count()} faces")
lo, hi = b.bounding_box()
check("cap round: ends reach r past the rail", abs(lo[2] + r0) < 1e-4 and abs(hi[2] - (L + r0)) < 1e-4, f"{lo} {hi}")
r, o = pipe(line, diameter=2 * r0, use_diameter=True, cap_style="ROUND", body="NEW")
check("diameter + round, new solid", r == {'FINISHED'} and solid(o) and abs(vol(o) - (rod + ball)) < 1e-4,
      f"{vol(o):.5f}")
check("history keeps radius and cap", features.history(o)["params"].get("radius") == r0
      and features.history(o)["params"].get("cap_style") == "ROUND")
r, o = pipe(line, radius=r0, cap_style="NONE", body="NEW")
check("solid with cap none: flat", r == {'FINISHED'} and solid(o) and abs(vol(o) - rod) < 1e-4, f"{vol(o):.5f}")

# a bent rail: the half spheres sit tangent to the tube (normals agree across the seam)
bz = bpy.data.curves.new("BZ", 'CURVE'); bz.dimensions = '3D'
sp = bz.splines.new('BEZIER'); sp.bezier_points.add(1)
sp.bezier_points[0].co = (0, 0, 0); sp.bezier_points[0].handle_right = (2, 0, 0); sp.bezier_points[0].handle_left = (-2, 0, 0)
sp.bezier_points[1].co = (4, 4, 1); sp.bezier_points[1].handle_left = (4, 2, 1); sp.bezier_points[1].handle_right = (4, 6, 1)
bent = bpy.data.objects.new("BZ", bz); ctx.collection.objects.link(bent)
r, o = pipe(bent, radius=0.3, cap_style="ROUND")
b = store.get_brep(o)
worst = 0.0
for e in b.edges():
    fs = e.faces()
    if len(fs) != 2:
        continue
    t0, t1 = e.parameter_range() if hasattr(e, "parameter_range") else (0.0, 1.0)
    for t in np.linspace(0.1, 0.9, 5):
        try:
            p = e.evaluate(float(t0 + (t1 - t0) * t))
        except Exception:
            continue
        ns = []
        for f in fs:
            cp = f.closest_point(tuple(p))
            ns.append(np.array(f.normal_at(*cp[0]) if hasattr(f, "normal_at") else (0, 0, 0), float))
        if all(np.linalg.norm(n) > 0 for n in ns):
            c = abs(float(ns[0] @ ns[1]) / (np.linalg.norm(ns[0]) * np.linalg.norm(ns[1])))
            worst = max(worst, math.degrees(math.acos(min(1.0, c))))
check("bent rail, round caps: closed", r == {'FINISHED'} and solid(o) and b.face_count() == 3,
      f"{b.face_count()} faces")
print("  seam normal angle (deg):", round(worst, 4))

# a closed rail has no ends: round caps leave the ring as it is
bpy.ops.curve.primitive_nurbs_circle_add(radius=3.0, location=(10, 0, 0))
ring = ctx.active_object
r, o = pipe(ring, radius=0.5, cap_style="ROUND")
torus = 2 * math.pi * 3.0 * math.pi * 0.25
check("closed rail, round: the ring", r == {'FINISHED'} and solid(o) and abs(vol(o) - torus) < 5e-3,
      f"{vol(o):.5f} want {torus:.5f}")

# history: the Feature panel's Cap rebuilds the pipe
r, o = pipe(line, radius=r0, cap_style="FLAT")
h = json.loads(o[features.PROP_HIST]); h["params"]["cap_style"] = "ROUND"; o[features.PROP_HIST] = json.dumps(h)
features.rebuild(o, ctx)
check("history edit flat -> round", solid(o) and abs(vol(o) - (rod + ball)) < 1e-4, f"{vol(o):.5f}")
h["params"]["cap_style"] = "NONE"; o[features.PROP_HIST] = json.dumps(h)
features.rebuild(o, ctx)
check("history edit round -> none: open tube", not solid(o))


# --- rails of several pieces ---------------------------------------------------
def pipes(pairs, **kw):
    before = {o.name for o in bpy.data.objects}
    try:
        r = bpy.ops.nproto.pipe(inputs=ins(*pairs), **kw)
    except RuntimeError as exc:
        r = str(exc)
    return r, [o for o in bpy.data.objects if o.name not in before]

clear()
a1 = poly("A1", [(0, 0, 0), (0, 0, 2)]); a2 = poly("A2", [(0, 0, 2), (0, 0, 5)])
r, made = pipes([(a1, "S0"), (a2, "S0")], radius=0.5, cap_style="FLAT")
want = math.pi * 0.25 * 5
check("two lines end to end: one pipe", r == {'FINISHED'} and len(made) == 1 and solid(made[0])
      and abs(vol(made[0]) - want) < 1e-4, f"{len(made)} made, {[round(vol(o), 5) for o in made]} want {want:.5f}")
h = features.history(made[0])
check("history keeps both pieces", h is not None and len(h["inputs"]) == 2)

b1 = poly("B1", [(10, 0, 0), (12, 0, 0)]); b2 = poly("B2", [(12, 0, 0), (12, 2, 0)])
rod = lambda r, L: math.pi * r * r * L
r, made = pipes([(b1, "S0"), (b2, "S0")], radius=0.2, cap_style="FLAT")
check("90 degree corner: mitred, one closed body", r == {'FINISHED'} and len(made) == 1 and solid(made[0])
      and abs(vol(made[0]) - rod(0.2, 4)) < 1e-4, f"{r} {[round(vol(o), 5) for o in made]} want {rod(0.2, 4):.5f}")
if made:
    b = store.get_brep(made[0])
    check("  two tubes and two caps", b.face_count() == 4, f"{b.face_count()} faces")
r, made = pipes([(b1, "S0"), (b2, "S0")], radius=0.2)
check("cap none: an open mitred pipe in one piece", r == {'FINISHED'} and len(made) == 1 and not solid(made[0])
      and store.get_brep(made[0]).face_count() == 2)
if made:
    b = store.get_brep(made[0])
    shared = sum(1 for e in b.edges() if len(e.faces()) == 2)
    check("  the tubes share the mitre edge", shared >= 1, f"{shared} shared edges")
r, made = pipes([(b1, "S0"), (b2, "S0")], radius=0.2, cap_style="ROUND", body="NEW")
want = rod(0.2, 4) + 4 / 3 * math.pi * 0.008
check("mitre with round caps, new solid", r == {'FINISHED'} and len(made) == 1 and solid(made[0])
      and abs(vol(made[0]) - want) < 1e-4, f"{[round(vol(o), 5) for o in made]} want {want:.5f}")
r, made = pipes([(b1, "S0"), (b2, "S0")], radius=0.2, join=False)
check("join off: a pipe each", r == {'FINISHED'} and len(made) == 2)
L3 = poly("L3", [(20, 0, 0), (22, 0, 0), (23, 1.5, 0.5), (23, 3, 0)])
r, made = pipes([(L3, "S0")], radius=0.15, cap_style="FLAT")
segs = [np.linalg.norm(np.subtract(b_, a_)) for a_, b_ in zip([(20, 0, 0), (22, 0, 0), (23, 1.5, 0.5)],
                                                             [(22, 0, 0), (23, 1.5, 0.5), (23, 3, 0)])]
want = rod(0.15, sum(segs))
check("3D polyline, two mitres", r == {'FINISHED'} and len(made) == 1 and solid(made[0])
      and abs(vol(made[0]) - want) < 1e-4, f"{[round(vol(o), 5) for o in made]} want {want:.5f}")
sq = poly("Sq", [(30, 0, 0), (32, 0, 0), (32, 2, 0), (30, 2, 0)], cyclic=True)
r, made = pipes([(sq, "S0")], radius=0.2, body="NEW")
check("closed square: a mitred solid frame", r == {'FINISHED'} and len(made) == 1 and solid(made[0])
      and abs(vol(made[0]) - rod(0.2, 8)) < 1e-4, f"{r} {[round(vol(o), 5) for o in made]} want {rod(0.2, 8):.5f}")
r, made = pipes([(sq, "S0")], radius=0.2)
check("closed square, cap none: closed too (no ends)", r == {'FINISHED'} and len(made) == 1
      and store.get_brep(made[0]).face_count() == 4)
r, made = pipes([(b1, "S0"), (b2, "S0")], radius=0.6)
check("a radius the mitre has no room for is refused", "too large" in str(r) and not made, str(r)[:150])
# a curved stretch into a corner: the tube near the mitre runs straight; check it stays close to the radius
bz = bpy.data.curves.new("CQ", 'CURVE'); bz.dimensions = '3D'
sp = bz.splines.new('BEZIER'); sp.bezier_points.add(1)
p0, p1 = sp.bezier_points
p0.co = (40, 0, 0); p0.handle_right = (41, 0, 0); p0.handle_left = (39, 0, 0)
p1.co = (42, 1, 0); p1.handle_left = (42, 0, 0); p1.handle_right = (42, 2, 0)
cq = bpy.data.objects.new("CQ", bz); ctx.collection.objects.link(cq)
up = poly("UP", [(42, 1, 0), (42, 1, 2)])
r, made = pipes([(cq, "S0"), (up, "S0")], radius=0.1, cap_style="FLAT")
check("curve into a corner: mitred solid", r == {'FINISHED'} and len(made) == 1 and solid(made[0]), str(r)[:150])
if made:
    rail = features.profile_curve([(cq, "S0"), (up, "S0")])
    t0_, t1_ = rail.parameter_range()
    # each tube is the rail on its side of the corner, its axis run on past the corner (the mitre's tips lie there)
    from blender_nurbs import sweeps
    c = sweeps.rail_corners(rail)[0]
    ext_ = np.linspace(0.0, 0.3, 300)[:, None]
    CA = np.vstack([np.array([rail.evaluate(t) for t in np.linspace(t0_, c["u"], 3000)]), c["point"] + ext_ * c["t_in"]])
    CB = np.vstack([np.array([rail.evaluate(t) for t in np.linspace(c["u"], t1_, 3000)]), c["point"] - ext_ * c["t_out"]])
    worst = 0.0
    for f in store.get_brep(made[0]).faces():
        sf = f.surface()
        (u0_, u1_), (v0_, v1_) = sf.u_domain(), sf.v_domain()
        if sf.is_planar():
            continue                    # the flat caps
        for u in np.linspace(u0_, u1_, 41):
            for v in np.linspace(v0_, v1_, 17):
                x = np.array(sf.evaluate(u, v))
                if f.classify_uv(u, v) == "outside":
                    continue
                err = min(abs(np.min(np.linalg.norm(CA - x, axis=1)) - 0.1),
                          abs(np.min(np.linalg.norm(CB - x, axis=1)) - 0.1))
                worst = max(worst, err)
    print("  curve into corner: worst radius error", round(worst, 6))
    check("  stays within 3% of the radius", worst < 0.003, f"{worst:.6f}")

# an arc and a line tangent to it: one smooth rail
bpy.ops.curve.primitive_nurbs_path_add()
cu = bpy.data.curves.new("Arc", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('BEZIER'); sp.bezier_points.add(1)
k = 0.5522847
p0, p1 = sp.bezier_points
p0.co = (30, 0, 0); p0.handle_right = (30, k, 0); p0.handle_left = (30, -k, 0)
p1.co = (29, 1, 0); p1.handle_left = (30 - k + 0, 1, 0); p1.handle_left = (29 + k, 1, 0); p1.handle_right = (29 - k, 1, 0)
arc = bpy.data.objects.new("Arc", cu); ctx.collection.objects.link(arc)
ln = poly("Ln", [(29, 1, 0), (26, 1, 0)])
for o in list(bpy.data.objects):
    if o.name.startswith("NurbsPath"):
        bpy.data.objects.remove(o)
r, made = pipes([(arc, "S0"), (ln, "S0")], radius=0.2, cap_style="ROUND")
want = math.pi * 0.04 * (math.pi / 2 + 3) + 4 / 3 * math.pi * 0.008
check("tangent arc + line, round caps: one closed pipe", r == {'FINISHED'} and len(made) == 1 and solid(made[0])
      and abs(vol(made[0]) - want) < 2e-3, f"{[round(vol(o), 5) for o in made]} want {want:.5f}")

# a curve object of two splines: every spline is a rail (the selection path)
cu = bpy.data.curves.new("Two", 'CURVE'); cu.dimensions = '3D'
for x in (40, 42):
    sp = cu.splines.new('POLY'); sp.points.add(1)
    sp.points[0].co = (x, 0, 0, 1); sp.points[1].co = (x, 0, 2, 1)
two = bpy.data.objects.new("Two", cu); ctx.collection.objects.link(two)
from blender_nurbs.ops import _curve_inputs
for o in ctx.selected_objects:
    o.select_set(False)
two.select_set(True)
sel = _curve_inputs(ctx)
r, made = pipes(sel, radius=0.2, body="NEW")
check("both splines of one curve object", len(sel) == 2 and r == {'FINISHED'} and len(made) == 2
      and all(solid(o) for o in made))

# a closed loop that is not flat: the tube meets itself at the seam (a solid ring)
cu = bpy.data.curves.new("Saddle", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('NURBS'); n = 12
sp.points.add(n - 1)
for i, pt in enumerate(sp.points):
    t = 2 * math.pi * i / n
    pt.co = (50 + 3 * math.cos(t), 3 * math.sin(t), 1.2 * math.cos(2 * t), 1.0)
sp.use_cyclic_u = True; sp.order_u = 4
sad = bpy.data.objects.new("Saddle", cu); ctx.collection.objects.link(sad)
r, made = pipes([(sad, "S0")], radius=0.3, body="NEW")
check("closed saddle loop: a solid ring", r == {'FINISHED'} and len(made) == 1 and solid(made[0]),
      str(r)[:140])
if made:
    from blender_nurbs import sweeps
    P, T, N, B = sweeps.pipe_frames(features.input_curve(sad, "S0"), 64, True)
    check("closed rail frames come back to the start", float(np.linalg.norm(N[-1] - N[0])) < 1e-9)

# surface edges that touch: two extrusions of a tangent arc and line, their bottom edges as one rail
def bottom_edge(o):
    for ref in ("U0", "U1", "V0", "V1"):
        try:
            kc = features.input_curve(o, ref)
        except Exception:
            continue
        t0, t1 = kc.parameter_range()
        P = np.array([kc.evaluate(t) for t in np.linspace(t0, t1, 9)])
        if np.abs(P[:, 2]).max() < 1e-6 and np.ptp(P, axis=0).max() > 1e-3:
            return ref
r1 = bpy.ops.nproto.extrude(inputs=ins((arc, "S0")), distance=1.0, direction="Z"); s1 = ctx.active_object
r2 = bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=1.0, direction="Z"); s2 = ctx.active_object
e1, e2 = bottom_edge(s1), bottom_edge(s2)
r, made = pipes([(s1, e1), (s2, e2)], radius=0.1, cap_style="FLAT")
want = math.pi * 0.01 * (math.pi / 2 + 3)
check("two touching surface edges: one pipe", e1 and e2 and r == {'FINISHED'} and len(made) == 1
      and solid(made[0]) and abs(vol(made[0]) - want) < 1e-3, f"{e1} {e2} {r} {[round(vol(o), 5) for o in made]}")

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
