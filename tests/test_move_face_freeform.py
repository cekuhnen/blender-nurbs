"""Move Face on free-form faces: features carried over a face with no symmetry,
their rims cut anew where they land:
    blender -b --factory-startup --python tests/test_move_face_freeform.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import drawshapes as ds, features, pushpull, step_export as sx, steps, store
from blender_nurbs.valuedrag import rotate_points

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def ins(*objs): return json.dumps([[o.name, "S0"] for o in objs])
def vol(o): return abs(store.get_brep(o).volume(1e-6))
def nfaces(o): return store.get_brep(o).face_count()

def step_hl(o, k):
    """Faces the history panel outlines in blue with step k of o picked."""
    from blender_nurbs import subsel as _ss
    ctx.view_layer.objects.active = o
    bpy.ops.nproto.tree_step(name=o.name, root=o.name, index=k)
    out = sorted(_ss.step_faces(o))
    o.nproto_step.focus = False
    return out

def near_face(o, q):
    fs = store.get_brep(o).faces()
    return min(range(len(fs)), key=lambda j: fs[j].closest_point(tuple(map(float, q)))[1])
def solid(o): return bool(store.get_brep(o).is_manifold_solid())
def faces(*items): return json.dumps([[o.name, list(p)] for o, p in items])

def run(op, *a, **kw):
    try:
        return op(*a, **kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

def circle(r, at, normal=(0, 0, 1)):
    n = np.array(normal, float)
    x = np.array((1, 0, 0) if abs(n[0]) < 0.9 else (0, 1, 0), float)
    bpy.ops.nproto.draw_circle(data=json.dumps({"matrix": ds.frame(np.array(at, float), x, n).tolist()}),
                               radius=r, ctype="EXACT")
    return ctx.active_object

def poly(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def extrude(curves, h, body="NEW", direction="Z", **kw):
    bpy.ops.nproto.extrude(inputs=ins(*curves), distance=h, direction=direction, body=body, **kw)
    return ctx.active_object

def torus(R=2.0, r=0.6):
    bpy.ops.nproto.add_primitive(kind="TORUS", radius=R, radius2=r)
    return ctx.active_object

def vertical_hole(at, r=0.2):
    return extrude([circle(r, (at[0], at[1], -1))], 2.0, body="CUT", direction="Z")

def hole_axis(o, limit=20.0):
    """(point at z = 0, unit direction) of the one small cylindrical face's axis."""
    for f in store.get_brep(o).faces():
        srf = f.surface()
        if srf.is_planar() or f.area(1e-6) >= limit:
            continue
        (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
        G = np.array([[srf.evaluate(float(u), float(v)) for v in np.linspace(v0, v1, 9)]
                      for u in np.linspace(u0, u1, 9)], float)
        fit = sx.fit_cylinder(G, 1e-7, 4.0)
        if fit is not None:
            p, a = np.asarray(fit[0], float), np.asarray(fit[1], float)
            return (p - p[2] / a[2] * a if abs(a[2]) > 1e-9 else p), a
    return None

def floor_centre(o, near, area=math.pi * 0.0225):
    """Centre of the blind hole floor (a flat disc of that area) nearest near."""
    cs = []
    for f in store.get_brep(o).faces():
        if f.surface().is_planar() and abs(f.area(1e-6) - area) < 1e-4:
            e = f.edges()[0]
            t0, t1 = e.parameter_range()
            P = np.array([e.point_at(float(t)) for t in np.linspace(t0, t1, 25)[:-1]], float)
            cs.append(sx.fit_circle(P, 1e-7, 4.0)[0])
    return min(cs, key=lambda c: np.linalg.norm(c - np.asarray(near, float))) if cs else None

def rz(P, deg):
    return rotate_points(np.asarray(P, float)[None], np.zeros(3), np.array([0, 0, 1.0]), deg)[0]

def move(o, p, **kw):
    return run(bpy.ops.nproto.move_face, faces=faces((o, p)), **kw)


sm, ext = store._k()

def bump_box():
    """A 4 x 3 x 2 box whose top is a bicubic bump (a free-form face)."""
    R = poly("R", [(-2, -1.5, 0), (2, -1.5, 0), (2, 1.5, 0), (-2, 1.5, 0)])
    B = extrude([R], 2.0)
    pts = []
    for i, x in enumerate(np.linspace(-3, 3, 4)):
        for j, y in enumerate(np.linspace(-2.5, 2.5, 4)):
            pts.append((float(x), float(y), 2.0 + (0.7 if 0 < i < 3 and 0 < j < 3 else 0.0) + 0.15 * x))
    U = [0.0] * 4 + [1.0] * 4
    S = store.new_object(ctx, "Bump", ext.surface_from_cvs(3, 3, pts, U, U, None))
    run(bpy.ops.nproto.replace_face, faces=faces((B, (0, 0, 2))), surface=S.name)
    bpy.data.objects.remove(S)
    return B

def top_index(o):
    fb = store.get_brep(o).faces()
    return min(range(len(fb)), key=lambda k: fb[k].closest_point((0, 0, 4))[1])

def expected(o, idx, vec_xy, follow=True):
    """(R, A, A2) the move should use: the anchor (the feature's opening on the top) moved by
    vec_xy along the top's tangent axes there."""
    feat = pushpull.feature_of(o, idx)
    fb = store.get_brep(o).faces()
    s = max(feat[1], key=lambda k: fb[k].closest_point((0, 0, 4))[1] * -1)     # the seat up top
    C = np.array(feat[3][s][0], float) if feat[3].get(s) else np.array(feat[2][s], float)
    _, A, _, n0 = pushpull.free_frame(o, s, C, np.zeros(3))
    ax, ay = pushpull._plane_axes(n0)
    R, A, A2, _ = pushpull.free_frame(o, s, C, vec_xy[0] * ax + vec_xy[1] * ay, follow)
    return R, A, A2

def line_dist(p, a, q, b):
    """Distance between the lines p + t a and q + s b when parallel (else inf)."""
    if abs(abs(float(np.dot(a, b))) - 1) > 1e-7:
        return np.inf
    w = q - p
    return float(np.linalg.norm(w - (w @ a) * a))

def face_ids(o, near):
    return [pushpull._face_index(o, near)]

ctx.scene.nproto_settings.construction_history = True

# --- a hole through the bump and the flat bottom ----------------------------------------
clear()
B = bump_box()
B = extrude([circle(0.25, (-0.8, 0, -1))], 5.0, body="CUT")
n0 = nfaces(B)
s = top_index(B)
mode = pushpull.seat_motion(B, s)
check("a free-form seat moves by frames", mode is not None and mode[0] == "FREE", mode and mode[0])
idx = face_ids(B, (-0.8, 0.25, 1.0))
R, A, A2 = expected(B, idx, (0.6, 0.3))
t0 = time.perf_counter()
r = move(B, (-0.8, 0.25, 1.0), move_x=0.6, move_y=0.3)
dt = time.perf_counter() - t0
p, a = hole_axis(B)
want_p, want_a = R @ (np.array([-0.8, 0, 0.0]) - A) + A2, R @ np.array([0, 0, 1.0])
check("the hole slid over the bump, square to it where it lands", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and line_dist(p, a, want_p, want_a) < 1e-6 and abs(float(want_a[2])) < 0.9999,
      f"axis {a.round(5)} vs {want_a.round(5)} off {line_dist(p, a, want_p, want_a):.2e} {dt * 1000:.0f} ms")
check("move time", dt < 4.0, f"{dt * 1000:.0f} ms")
st = features.stack(B)
check("the move is a step", st[-1]["params"]["mode"] == "FREE"
      and steps.LABEL["move_face"](st[-1]["params"]) == "Move Face · 0.6708 m", steps.LABEL["move_face"](st[-1]["params"]))
features.rebuild(B, ctx)
p2, a2 = hole_axis(B)
check("it replays on rebuild", line_dist(p2, a2, p, a) < 1e-6 and solid(B))
hl = step_hl(B, len(st) - 1)
fs = store.get_brep(B).faces()
def _off_axis(j):
    lo, hi = (np.array(v, float) for v in fs[j].bounding_box())
    c = (lo + hi) / 2 - p2
    return float(np.linalg.norm(c - (c @ a2) * a2))
check("picked free-form Move Face step highlights the moved hole", hl and top_index(B) not in hl
      and all(not fs[j].surface().is_planar() and _off_axis(j) < 0.3 for j in hl), f"{hl} top {top_index(B)}")
st[-1]["params"]["follow"] = False
features._write_stack(B, st)
features.rebuild(B, ctx)
p2, a2 = hole_axis(B)
check("edited to keep its direction (not square)", abs(abs(float(a2[2])) - 1) < 1e-9 and solid(B)
      and nfaces(B) == n0, f"axis {a2.round(6)}")

# --- a pocket by its floor, a boss by its top; off the face refused -------------------------
ctx.scene.nproto_settings.construction_history = False
clear()
B = bump_box()
B = extrude([circle(0.3, (0.8, 0.3, 4))], -2.8, body="CUT")
n0 = nfaces(B)
idx = face_ids(B, (0.8, 0.3, 1.2))
R, A, A2 = expected(B, idx, (-0.9, -0.4))
r = move(B, (0.8, 0.3, 1.2), move_x=-0.9, move_y=-0.4)
fl = min(store.get_brep(B).faces(), key=lambda f: f.closest_point(tuple(R @ (np.array([0.8, 0.3, 1.2]) - A) + A2))[1])
want = R @ (np.array([0.8, 0.3, 1.2]) - A) + A2
check("a pocket slid by its floor (the floor goes with it)", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and fl.closest_point(tuple(want))[1] < 1e-6 and fl.surface().is_planar())
clear()
B = bump_box()
B = extrude([circle(0.3, (0.0, -0.6, 1.5))], 1.8, body="JOIN")
n0, v0 = nfaces(B), vol(B)
idx = face_ids(B, (0.0, -0.6, 3.3))
R, A, A2 = expected(B, idx, (0.5, 0.9))
r = move(B, (0.0, -0.6, 3.3), move_x=0.5, move_y=0.9)
want = R @ (np.array([0.0, -0.6, 3.3]) - A) + A2
top = min(store.get_brep(B).faces(), key=lambda f: f.closest_point(tuple(want))[1])
check("a boss slid by its top", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and top.closest_point(tuple(want))[1] < 1e-6 and abs(vol(B) - v0) < 0.05 * v0, f"vol {vol(B):.5f}/{v0:.5f}")
v = vol(B)
r = move(B, tuple(want), move_x=3.0)
check("a slide off the face refused", r == {'CANCELLED'} and abs(vol(B) - v) < 1e-9)

# --- two pockets: one slid into the other refused; a drafted pocket refused --------------------
clear()
B = bump_box()
extrude([circle(0.3, (-1.0, 0.0, 4))], -2.8, body="CUT")
B = extrude([circle(0.3, (0.6, 0.0, 4))], -2.8, body="CUT")
v0 = vol(B)
r = move(B, (0.6, 0.0, 1.2), move_x=-1.5)
check("a pocket slid into another refused", r == {'CANCELLED'} and abs(vol(B) - v0) < 1e-9)
# --- drafted walls: continued along their own rulings to a far plane ------------------------
def disc_centre(o, near, r_):
    """Centre of the flat disc of radius r_ nearest near (a drafted pocket's floor, a boss's top)."""
    cs = []
    for f in store.get_brep(o).faces():
        if f.surface().is_planar() and abs(f.area(1e-6) - math.pi * r_ * r_) < 1e-4:
            e = f.edges()[0]
            t0, t1 = e.parameter_range()
            P = np.array([e.point_at(float(t)) for t in np.linspace(t0, t1, 25)[:-1]], float)
            cs.append(sx.fit_circle(P, 1e-7, 4.0)[0])
    return min(cs, key=lambda c: np.linalg.norm(c - np.asarray(near, float))) if cs else None

r_floor = 0.3 - 2.8 * math.tan(math.radians(5.0))
clear()
B = bump_box()
B = extrude([circle(0.3, (0.6, 0.0, 4))], -2.8, body="CUT", draft=5.0)
n0 = nfaces(B)
c0 = disc_centre(B, (0.6, 0, 1.2), r_floor)
idx = face_ids(B, (0.6, 0.0, 1.2))
R, A, A2 = expected(B, idx, (-0.7, 0.3))
r = move(B, (0.6, 0.0, 1.2), move_x=-0.7, move_y=0.3)
c1 = disc_centre(B, R @ (c0 - A) + A2, r_floor)
check("a drafted (conical) pocket slid by its floor", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and c1 is not None and np.allclose(c1, R @ (c0 - A) + A2, atol=1e-6), f"{c1} vs {R @ (c0 - A) + A2}")
clear()
B = bump_box()
B = extrude([poly("Sq", [(0.3, -0.3, 4), (0.9, -0.3, 4), (0.9, 0.3, 4), (0.3, 0.3, 4)])], -2.8, body="CUT", draft=6.0)
n0 = nfaces(B)
fl0 = min(store.get_brep(B).faces(), key=lambda f: f.closest_point((0.6, 0.0, 1.2))[1])
c0 = np.mean([np.array(v, float) for v in fl0.bounding_box()], axis=0)
idx = face_ids(B, (0.6, 0.0, 1.2))
R, A, A2 = expected(B, idx, (-0.8, 0.0))
r = move(B, (0.6, 0.0, 1.2), move_x=-0.8)
want = R @ (c0 - A) + A2
fl = min(store.get_brep(B).faces(), key=lambda f: f.closest_point(tuple(want))[1])
check("a drafted square pocket slid (flat walls leaning in)", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and fl.closest_point(tuple(want))[1] < 1e-6 and fl.surface().is_planar())
clear()
B = bump_box()
B = extrude([circle(0.25, (-0.8, 0, -1))], 5.0, body="CUT", draft=3.0)
n0 = nfaces(B)
r = move(B, (-0.8, 0.0, 1.0), move_x=0.6)
check("a drafted hole through the bump and the bottom slid", r == {'FINISHED'} and solid(B) and nfaces(B) == n0)
clear()
B = bump_box()
B = extrude([circle(0.3, (0.0, -0.6, 1.5))], 1.8, body="JOIN", draft=-5.0)
r_top = 0.3 - 1.8 * math.tan(math.radians(5.0))
n0 = nfaces(B)
c0 = disc_centre(B, (0.0, -0.6, 3.3), r_top)
idx = face_ids(B, (0.0, -0.6, 3.3))
R, A, A2 = expected(B, idx, (0.5, 0.6))
r = move(B, (0.0, -0.6, 3.3), move_x=0.5, move_y=0.6)
c1 = disc_centre(B, R @ (c0 - A) + A2, r_top)
check("a drafted boss slid by its top", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and c1 is not None and np.allclose(c1, R @ (c0 - A) + A2, atol=1e-6), f"{c1}")

# --- rounded walls: a ball's dimple and bump (the ball moved and cut again), a filleted rim ----
def ball_op(B, at, r_, op):
    ctx.scene.cursor.location = at
    bpy.ops.nproto.add_primitive(kind="SPHERE", radius=r_)
    ctx.scene.cursor.location = (0, 0, 0)
    Sb = ctx.active_object
    bpy.ops.nproto.boolean(operation=op, names=json.dumps([B.name, Sb.name]))
    return ctx.active_object

def sphere_centre(o, limit=3.0):
    for f in store.get_brep(o).faces():
        srf = f.surface()
        if srf.is_planar() or f.area(1e-6) >= limit:
            continue
        (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
        G = np.array([[srf.evaluate(float(u), float(v)) for v in np.linspace(v0, v1, 9)]
                      for u in np.linspace(u0, u1, 9)], float)
        fit = sx.fit_sphere(G, 1e-7, 4.0)
        if fit is not None:
            return np.asarray(fit[0], float)
    return None

for kind, at, r_, op, pick, mv in (("dimple", (0.6, 0.0, 2.75), 0.4, "DIFFERENCE", (0.6, 0.0, 2.36), (-0.8, 0.3)),
                                   ("bump", (-0.6, 0.2, 2.5), 0.45, "UNION", (-0.6, 0.2, 2.95), (0.9, -0.2))):
    clear()
    B = ball_op(bump_box(), at, r_, op)
    n0 = nfaces(B)
    idx = face_ids(B, pick)
    R, A, A2 = expected(B, idx, mv)
    r = move(B, pick, move_x=mv[0], move_y=mv[1])
    c1 = sphere_centre(B)
    want = R @ (np.array(at, float) - A) + A2
    check(f"a ball's {kind} slid over the bump (the ball moved, cut again)", r == {'FINISHED'} and solid(B)
          and nfaces(B) == n0 and c1 is not None and np.allclose(c1, want, atol=1e-6), f"{c1} vs {want}")

clear()
B = bump_box()
B = extrude([circle(0.3, (0.6, 0.0, 4))], -2.8, body="CUT")
es = store.get_brep(B).edges()
rim = min(range(len(es)), key=lambda i: abs(es[i].length() - 2 * math.pi * 0.3)
          + abs(np.array(es[i].point_at(sum(es[i].parameter_range()) / 2))[2] - 2.5))
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[B.name, f"E{rim}"]]), radius=0.08, shape="CIRCULAR")
B = ctx.active_object
n0 = nfaces(B)
fil0 = sorted(f.area(1e-6) for f in store.get_brep(B).faces() if not f.surface().is_planar())[0]
idx = face_ids(B, (0.6, 0.0, 1.2))
R, A, A2 = expected(B, idx, (-0.5, 0.0))
r = move(B, (0.6, 0.0, 1.2), move_x=-0.5)
want = R @ (np.array([0.6, 0.0, 1.2]) - A) + A2
fl = min(store.get_brep(B).faces(), key=lambda f: f.closest_point(tuple(want))[1])
fil1 = sorted(f.area(1e-6) for f in store.get_brep(B).faces() if not f.surface().is_planar())[0]
check("a pocket with a filleted rim slid (the rim rounded again where it lands)", r == {'FINISHED'} and solid(B)
      and nfaces(B) == n0 and fl.closest_point(tuple(want))[1] < 1e-6 and abs(fil1 - fil0) < 0.1 * fil0,
      f"fillet area {fil1:.4f} / {fil0:.4f}")

clear()
B = bump_box()
B = extrude([circle(0.3, (0.6, 0.0, 4))], -2.8, body="CUT")
v0 = vol(B)
r = move(B, (0.6, 0.0, 1.2), move_x=1.9)
check("rounded or not, off the face is refused", r == {'CANCELLED'} and abs(vol(B) - v0) < 1e-9)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
