"""Move Face on tori: features turned about a torus's axis, across both of its
seams; a band of a torus (an elbow):
    blender -b --factory-startup --python tests/test_move_face_torus.py"""
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

def hole_axis(o, limit=3.0):
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

ctx.scene.nproto_settings.construction_history = True

# --- a hole through a whole torus's tube (its bottom opening across the tube's seam) -------
clear()
T = torus()
T = vertical_hole((2, 0))
v0, n0 = vol(T), nfaces(T)
fb = store.get_brep(T).faces()
seat = max(range(len(fb)), key=lambda k: fb[k].area(1e-6))
mode = pushpull.seat_motion(T, seat)
check("a torus seat turns about its axis", mode is not None and mode[0] == "TURN"
      and abs(abs(float(mode[2][2])) - 1) < 1e-9 and np.linalg.norm(np.asarray(mode[1])[:2]) < 1e-6, mode and mode[0])
t0 = time.perf_counter()
r = move(T, (2, 0, 0), turn=90.0)
dt = time.perf_counter() - t0
p, a = hole_axis(T)
check("the hole turned 90° round the torus", r == {'FINISHED'} and solid(T) and nfaces(T) == n0
      and abs(vol(T) - v0) < 1e-5 and np.allclose(p[:2], [0, 2], atol=1e-6) and abs(abs(a[2]) - 1) < 1e-9,
      f"at {p.round(6)} vol {vol(T):.6f}/{v0:.6f} {dt * 1000:.0f} ms")
check("move time", dt < 4.0, f"{dt * 1000:.0f} ms")
r = move(T, (0, 2, 0), turn=180.0)
p, a = hole_axis(T)
check("turned on across the torus's seam", r == {'FINISHED'} and solid(T) and nfaces(T) == n0
      and abs(vol(T) - v0) < 1e-5 and np.allclose(p[:2], [0, -2], atol=1e-6), f"at {p.round(6)}")
st = features.stack(T)
check("the turns are steps", [e["params"]["mode"] for e in st[-2:]] == ["TURN", "TURN"]
      and steps.LABEL["move_face"](st[-1]["params"]) == "Move Face · 180°", steps.LABEL["move_face"](st[-1]["params"]))
features.rebuild(T, ctx)
p, a = hole_axis(T)
check("both replay on rebuild", np.allclose(p[:2], [0, -2], atol=1e-6) and solid(T))
st[-1]["params"]["turn"] = 45.0
features._write_stack(T, st)
features.rebuild(T, ctx)
p, a = hole_axis(T)
check("the last turn edited to 45°", np.allclose(p[:2], rz([0, 2, 0], 45)[:2], atol=1e-6) and solid(T)
      and abs(vol(T) - v0) < 1e-5, f"at {p.round(6)}")
v = vol(T)
q = rz([0, 2, 0], 45)
r = move(T, tuple(q), mode="AXIS", axis_point=(0, 0, 0), axis_x=(0, 0, 1), slide=0.3)
check("a slide along a torus's axis refused", r == {'CANCELLED'} and abs(vol(T) - v) < 1e-9)

# --- a blind hole in the outer equator, onto the seam; into another hole refused ---------
ctx.scene.nproto_settings.construction_history = False
clear()
T = torus()
extrude([circle(0.15, (3, 0, 0), normal=(1, 0, 0))], -0.7, body="CUT", direction="X")
T = extrude([circle(0.15, (0, 3, 0), normal=(0, 1, 0))], -0.7, body="CUT", direction="Y")
v0, n0 = vol(T), nfaces(T)
c0 = floor_centre(T, (2.3, 0, 0))
r = move(T, tuple(c0), turn=-90.0)
c1 = floor_centre(T, (0, -2.3, 0))
check("a blind hole turned onto the seam", r == {'FINISHED'} and solid(T) and nfaces(T) == n0
      and abs(vol(T) - v0) < 1e-5 and c1 is not None and np.allclose(c1, rz(c0, -90), atol=1e-6), f"{c1}")
r = move(T, tuple(c1), turn=-180.0)
check("a blind hole turned into another refused", r == {'CANCELLED'} and abs(vol(T) - v0) < 1e-5)

# --- a half torus (an elbow): turned along it; past its end refused ------------------------
clear()
T = torus()
B = poly("Cut", [(-3, -3, -1), (3, -3, -1), (3, 0, -1), (-3, 0, -1)])
T = extrude([B], 2.0, body="CUT")
c = 2 * np.array([math.cos(math.radians(45)), math.sin(math.radians(45))])
T = vertical_hole(c)
v0, n0 = vol(T), nfaces(T)
r = move(T, (c[0], c[1], 0), turn=60.0)
p, a = hole_axis(T)
check("a hole in an elbow turned along it", r == {'FINISHED'} and solid(T) and nfaces(T) == n0
      and abs(vol(T) - v0) < 1e-5 and np.allclose(p[:2], rz([c[0], c[1], 0], 60)[:2], atol=1e-6),
      f"at {p.round(6)} vol {vol(T):.6f}/{v0:.6f}")
q = rz([c[0], c[1], 0], 60)
r = move(T, tuple(q), turn=80.0)
check("turned past the elbow's end refused", r == {'CANCELLED'} and abs(vol(T) - v0) < 1e-5)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
