"""Move Face on spheres: features turned about a sphere's axis and tilted
along its meridian, over its poles; holes at a dome's top:
    blender -b --factory-startup --python tests/test_move_face_sphere.py"""
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

def extrude(curves, h, body="NEW", direction="Z", **kw):
    bpy.ops.nproto.extrude(inputs=ins(*curves), distance=h, direction=direction, body=body, **kw)
    return ctx.active_object

def ball(r=1.0, at=(0, 0, 0)):
    ctx.scene.cursor.location = at
    bpy.ops.nproto.add_primitive(kind="SPHERE", radius=r)
    ctx.scene.cursor.location = (0, 0, 0)
    return ctx.active_object

def dome():
    """A ball of radius 1.2 at z = 3 on a rod of radius 1, 3 high (the rod is no feature of it)."""
    R = extrude([circle(1.0, (0, 0, 0))], 3.0)
    S = ball(1.2, (0, 0, 3))
    bpy.ops.nproto.boolean(operation="UNION", names=json.dumps([R.name, S.name]))
    return ctx.active_object

def floor_centre(o, area=math.pi * 0.04):
    """Centre of the flat disc (a blind hole's floor) of the given area."""
    for f in store.get_brep(o).faces():
        if f.surface().is_planar() and abs(f.area(1e-6) - area) < 1e-4:
            e = f.edges()[0]
            t0, t1 = e.parameter_range()
            P = np.array([e.point_at(float(t)) for t in np.linspace(t0, t1, 25)[:-1]], float)
            return sx.fit_circle(P, 1e-7, 4.0)[0]
    return None

def hole_axis(o, limit=4.0):
    """(point, unit direction) of the one small cylindrical face's axis."""
    for f in store.get_brep(o).faces():
        srf = f.surface()
        if srf.is_planar() or f.area(1e-6) >= limit:
            continue
        (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
        G = np.array([[srf.evaluate(float(u), float(v)) for v in np.linspace(v0, v1, 9)]
                      for u in np.linspace(u0, u1, 9)], float)
        fit = sx.fit_cylinder(G, 1e-7, 4.0)
        if fit is not None:
            return np.asarray(fit[0], float), np.asarray(fit[1], float)
    return None

def same_line(a, b):
    return abs(abs(float(np.dot(a, b))) - 1.0) < 1e-9

def move(o, p, **kw):
    return run(bpy.ops.nproto.move_face, faces=faces((o, p)), **kw)

def rz(P, deg, O=(0, 0, 0)):
    return rotate_points(np.asarray(P, float)[None], np.asarray(O, float), np.array([0, 0, 1.0]), deg)[0]

ctx.scene.nproto_settings.construction_history = True

# --- a blind hole in a ball: turned, tilted, tilted over the pole -------------------
clear()
B = ball()
B = extrude([circle(0.2, (0.3, -2, 0.4), normal=(0, 1, 0))], 1.5, body="CUT", direction="Y")
v0, n0 = vol(B), nfaces(B)
fb = store.get_brep(B).faces()
seat = max(range(len(fb)), key=lambda k: fb[k].area(1e-6))
mode = pushpull.seat_motion(B, seat)
check("a sphere seat turns and tilts about its centre", mode is not None and mode[0] == "SPHERE"
      and np.linalg.norm(mode[1]) < 1e-6 and abs(abs(float(mode[2][2])) - 1) < 1e-9, mode and mode[0])
c0 = floor_centre(B)
t0 = time.perf_counter()
r = move(B, (0.3, -0.5, 0.4), turn=90.0)
dt = time.perf_counter() - t0
c1 = floor_centre(B)
check("turned 90° about the sphere's axis", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and abs(vol(B) - v0) < 1e-5 and np.allclose(c1, rz(c0, 90), atol=1e-6),
      f"{c1} vs {rz(c0, 90)} vol {vol(B):.6f}/{v0:.6f} {dt * 1000:.0f} ms")
check("move time", dt < 3.0, f"{dt * 1000:.0f} ms")
r = move(B, tuple(c1), tilt=25.0)
st = features.stack(B)
t = np.array(st[-1]["params"]["axis_y_local"], float)
c2 = floor_centre(B)
want = rotate_points(c1[None], np.zeros(3), t, 25.0)[0]
check("tilted 25° along its meridian (towards the top)", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and abs(vol(B) - v0) < 1e-5 and np.allclose(c2, want, atol=1e-6) and c2[2] > c1[2]
      and abs(t[2]) < 1e-9, f"{c2} vs {want}")
check("the step is a sphere move", st[-1]["params"]["mode"] == "SPHERE"
      and steps.LABEL["move_face"](st[-1]["params"]) == "Move Face · tilt 25°", steps.LABEL["move_face"](st[-1]["params"]))
r = move(B, tuple(c2), tilt=70.0)
t = np.array(features.stack(B)[-1]["params"]["axis_y_local"], float)
c3 = floor_centre(B)
want = rotate_points(c2[None], np.zeros(3), t, 70.0)[0]
check("tilted on over the pole", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and abs(vol(B) - v0) < 1e-5 and np.allclose(c3, want, atol=1e-6)
      and float(np.dot(c3[:2], c2[:2])) < 0, f"{c3} vs {want}")
features.rebuild(B, ctx)
check("all three replay on rebuild", np.allclose(floor_centre(B), c3, atol=1e-6) and solid(B))
st = features.stack(B)
st[-1]["params"]["tilt"] = 40.0
features._write_stack(B, st)
features.rebuild(B, ctx)
want = rotate_points(c2[None], np.zeros(3), t, 40.0)[0]
check("the last tilt edited to 40°", np.allclose(floor_centre(B), want, atol=1e-6) and solid(B)
      and abs(vol(B) - v0) < 1e-5, f"{floor_centre(B)} vs {want}")

# --- a hole through both poles, tilted off them ----------------------------------------
ctx.scene.nproto_settings.construction_history = False
clear()
B = ball()
B = extrude([circle(0.2, (0, 0, -2))], 4.0, body="CUT", direction="Z")
v0, n0 = vol(B), nfaces(B)
r = move(B, (0, 0, 0), mode="SPHERE", axis_point=(0, 0, 0), axis_x=(0, 0, 1), axis_y=(1, 0, 0), tilt=45.0)
p, a = hole_axis(B)
check("a hole through the poles tilted 45°", r == {'FINISHED'} and solid(B) and nfaces(B) == n0
      and abs(vol(B) - v0) < 1e-5 and same_line(a, [0, -math.sqrt(0.5), math.sqrt(0.5)])
      and np.linalg.norm(np.cross(p, a)) < 1e-6, f"axis {a} vol {vol(B):.6f}/{v0:.6f}")

# --- two holes in a ball: one tilted into the other is refused --------------------------
clear()
B = ball()
extrude([circle(0.2, (0, -2, 0), normal=(0, 1, 0))], 1.5, body="CUT", direction="Y")
B = extrude([circle(0.2, (2, 0, 0), normal=(1, 0, 0))], -1.5, body="CUT", direction="X")
v0 = vol(B)
r = move(B, (0, -0.75, 0), turn=90.0)
check("a hole turned onto another refused", r == {'CANCELLED'} and abs(vol(B) - v0) < 1e-9)
r = move(B, (0, -0.75, 0), turn=45.0)
check("turned half way: clear of it", r == {'FINISHED'} and solid(B) and abs(vol(B) - v0) < 1e-5)

# --- a dome: the hole drilled at its top, tilted and turned; too far is refused ---------
clear()
D = dome()
D = extrude([circle(0.2, (0, 0, 5))], -2.0, body="CUT", direction="Z")
v0, n0 = vol(D), nfaces(D)
fb = store.get_brep(D).faces()
mode = pushpull.seat_motion(D, max(range(len(fb)), key=lambda k: fb[k].area(1e-6) if not fb[k].surface().is_planar()
                                   and abs(fb[k].area(1e-6) - 14.68) > 0.1 else 0))
check("a dome is a sphere seat", mode is not None and mode[0] == "SPHERE" and np.allclose(mode[1], [0, 0, 3], atol=1e-6),
      mode and (mode[0], mode[1]))
r = move(D, (0, 0, 3.6), mode="SPHERE", axis_point=(0, 0, 3), axis_x=(0, 0, 1), axis_y=(1, 0, 0), tilt=30.0)
p, a = hole_axis(D)
check("the dome's top hole tilted 30° off the pole", r == {'FINISHED'} and solid(D) and nfaces(D) == n0
      and abs(vol(D) - v0) < 1e-5 and same_line(a, [0, -0.5, math.sqrt(0.75)])
      and np.linalg.norm(np.cross(p - np.array([0, 0, 3.0]), a)) < 1e-6, f"axis {a} vol {vol(D):.6f}/{v0:.6f}")
q = np.array([0, 0, 3.0]) + 0.6 * np.array([0, -0.5, math.sqrt(0.75)])
r = move(D, tuple(q), turn=120.0)
p, a = hole_axis(D)
check("then turned 120° about the dome's axis", r == {'FINISHED'} and solid(D) and abs(vol(D) - v0) < 1e-5
      and same_line(a, rz([0, -0.5, math.sqrt(0.75)], 120)), f"axis {a}")
v = vol(D)
q = np.array([0, 0, 3.0]) + 0.6 * a * np.sign(a[2])
r = move(D, tuple(q), tilt=-100.0)
check("a tilt past the dome's base refused", r == {'CANCELLED'} and abs(vol(D) - v) < 1e-9)

# --- a dome's side hole tilted over its top ----------------------------------------------
clear()
D = dome()
D = extrude([circle(0.2, (0.4, -2, 3.6), normal=(0, 1, 0))], 1.5, body="CUT", direction="Y")
v0, n0 = vol(D), nfaces(D)
c0 = floor_centre(D)
r = move(D, tuple(c0), mode="SPHERE", axis_point=(0, 0, 3), axis_x=(0, 0, 1), axis_y=(1, 0, 0), tilt=-70.0)
c1 = floor_centre(D)
want = rotate_points(c0[None], np.array([0, 0, 3.0]), np.array([1.0, 0, 0]), -70.0)[0]
check("a dome's side hole tilted over its top", r == {'FINISHED'} and solid(D) and nfaces(D) == n0
      and abs(vol(D) - v0) < 1e-5 and c1 is not None and np.allclose(c1, want, atol=1e-6) and c1[1] > 0,
      f"{c1} vs {want} vol {vol(D):.6f}/{v0:.6f}")
r = move(D, (0, 0, 1.5), turn=30.0)
check("the rod a dome sits on is no feature", r == {'CANCELLED'})

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
