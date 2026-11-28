"""Move Face on curved faces: features turned about and slid along a cylinder's
axis, turned about a cone's; holes across the wall's seam:
    blender -b --factory-startup --python tests/test_move_face_curved.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import drawshapes as ds, features, pushpull, steps, store

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
    """An exact (rational) circle."""
    n = np.array(normal, float)
    x = np.array((1, 0, 0) if abs(n[0]) < 0.9 else (0, 1, 0), float)
    bpy.ops.nproto.draw_circle(data=json.dumps({"matrix": ds.frame(np.array(at, float), x, n).tolist()}),
                               radius=r, ctype="EXACT")
    return ctx.active_object

def extrude(curves, h, body="NEW", direction="Z", **kw):
    bpy.ops.nproto.extrude(inputs=ins(*curves), distance=h, direction=direction, body=body, **kw)
    return ctx.active_object

def rod(r=1.0, h=4.0, **kw):
    return extrude([circle(r, (0, 0, 0))], h, **kw)

def cross_hole(z, r=0.3, depth=4.0, body="CUT"):
    """A hole along Y at height z from y = -2 (through the rod for depth 4)."""
    return extrude([circle(r, (0, -2, z), normal=(0, 1, 0))], depth, body=body, direction="Y")

def hole_axes(o, limit=5.0):
    """(point nearest the Z axis, unit direction) of each small cylindrical
    face's axis (the holes' walls), from an exact fit of its surface."""
    from blender_nurbs import step_export as sx
    out = []
    for f in store.get_brep(o).faces():
        srf = f.surface()
        if srf.is_planar() or f.area(1e-6) >= limit:
            continue
        (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
        G = np.array([[srf.evaluate(float(u), float(v)) for v in np.linspace(v0, v1, 9)]
                      for u in np.linspace(u0, u1, 9)], float)
        fit = sx.fit_cylinder(G, 1e-7, 4.0)
        if fit is None:
            continue
        p, a = np.asarray(fit[0], float), np.asarray(fit[1], float)
        # the axis point nearest the Z axis (a ⊥ Z): drop it onto the plane through Z across a
        q = p - (p[:2] @ a[:2]) / max(a[:2] @ a[:2], 1e-300) * a
        out.append((q, a))
    return out

def angle_of(a):
    """The direction's angle about Z (degrees, 0..180: a line either way)."""
    return math.degrees(math.atan2(a[1], a[0])) % 180.0

def move(o, p, **kw):
    return run(bpy.ops.nproto.move_face, faces=faces((o, p)), **kw)

ctx.scene.nproto_settings.construction_history = True

# --- a cross hole in a rod: one wall face, one opening across its seam ---------------
clear()
R = rod()
R = cross_hole(2.0)
v0, n0 = vol(R), nfaces(R)
fb = store.get_brep(R).faces()
seat = max(range(len(fb)), key=lambda k: fb[k].area(1e-6))
mode = pushpull.seat_motion(R, seat)
check("a cylinder seat moves along and about its axis", mode is not None and mode[0] == "AXIS"
      and abs(abs(float(mode[2][2])) - 1) < 1e-6 and np.linalg.norm(np.asarray(mode[1])[:2]) < 1e-6, mode)
t0 = time.perf_counter()
r = move(R, (0, 0, 2.0), slide=0.75)
dt = time.perf_counter() - t0
(q, a), = hole_axes(R)
check("the cross hole slid up the axis", r == {'FINISHED'} and solid(R) and nfaces(R) == n0
      and abs(vol(R) - v0) < 1e-5 and abs(q[2] - 2.75) < 1e-6 and abs(angle_of(a) - 90) < 1e-6,
      f"z {q[2]:.7f} at {angle_of(a):.4f}° vol {vol(R):.6f}/{v0:.6f} {dt * 1000:.0f} ms")
t0 = time.perf_counter()
r = move(R, (0, 0, 2.75), turn=90.0)
dt = time.perf_counter() - t0
(q, a), = hole_axes(R)
check("then turned 90° (now along X)", r == {'FINISHED'} and solid(R) and nfaces(R) == n0
      and abs(vol(R) - v0) < 1e-5 and abs(q[2] - 2.75) < 1e-6 and abs(angle_of(a) - 0) < 1e-6,
      f"z {q[2]:.7f} at {angle_of(a):.4f}° vol {vol(R):.6f}/{v0:.6f} {dt * 1000:.0f} ms")
check("turn time", dt < 3.0, f"{dt * 1000:.0f} ms")
st = features.stack(R)
check("the turn is a step on a cylinder", st[-1]["op"] == "move_face" and st[-1]["params"]["mode"] == "AXIS"
      and abs(st[-1]["params"]["turn"] - 90) < 1e-9, st[-1]["params"].get("mode"))
check("the labels", [steps.LABEL["move_face"](e["params"]) for e in st[-2:]] == ["Move Face · 0.75 m", "Move Face · 90°"],
      [steps.LABEL["move_face"](e["params"]) for e in st[-2:]])
features.rebuild(R, ctx)
(q, a), = hole_axes(R)
check("both replay on rebuild", abs(q[2] - 2.75) < 1e-6 and abs(angle_of(a)) < 1e-6 and solid(R))
st[-1]["params"]["turn"] = 45.0
features._write_stack(R, st)
features.rebuild(R, ctx)
(q, a), = hole_axes(R)
check("the step's turn edited to 45°", abs(angle_of(a) - 135) < 1e-6 and abs(q[2] - 2.75) < 1e-6
      and solid(R) and abs(vol(R) - v0) < 1e-5, f"at {angle_of(a):.4f}°")
v = vol(R)
r = move(R, (0, 0, 2.75), slide=1.2)
check("a slide off the end of the rod refused", r == {'CANCELLED'} and abs(vol(R) - v) < 1e-9)
r = move(R, (0, 0, 2.75), mode="PLANE", axis_x=(1, 0, 0), axis_y=(0, 1, 0), move_x=0.3)
check("a move off the cylinder refused", r == {'CANCELLED'} and abs(vol(R) - v) < 1e-9)

# --- two holes: one slid into the other is refused; turned past it is fine -----------
ctx.scene.nproto_settings.construction_history = False
clear()
R = rod()
cross_hole(1.0)
R = cross_hole(3.0)
v0 = vol(R)
r = move(R, (0, 0, 1.0), slide=1.8)
check("a hole slid into another refused", r == {'CANCELLED'} and abs(vol(R) - v0) < 1e-9)
r = move(R, (0, 0, 1.0), turn=-30.0)
r2 = move(R, (0, 0, 1.0), slide=1.6)
zs = sorted(round(float(q[2]), 6) for q, _ in hole_axes(R))
check("a hole turned, then slid past the other", r == r2 == {'FINISHED'} and solid(R)
      and abs(vol(R) - v0) < 1e-5 and zs == [2.6, 3.0], zs)

# --- a blind hole (a pocket) across the seam, turned half round -----------------------
clear()
R = rod()
R = cross_hole(2.0, depth=1.5)               # y from -2 to -0.5: a floor inside the rod
v0, n0 = vol(R), nfaces(R)
r = move(R, (0, -0.75, 2.0), turn=180.0)     # picked by its floor
fl = [f for f in store.get_brep(R).faces() if f.surface().is_planar() and abs(f.area(1e-6) - math.pi * 0.09) < 1e-4
      and abs(np.mean([np.array(v, float) for v in f.bounding_box()], axis=0)[2] - 2.0) < 1e-6]
check("a blind hole turned to the other side", r == {'FINISHED'} and solid(R) and nfaces(R) == n0
      and abs(vol(R) - v0) < 1e-5 and len(fl) == 1
      and abs(float(np.mean([np.array(v, float) for v in fl[0].bounding_box()], axis=0)[1]) - 0.5) < 1e-6,
      f"vol {vol(R):.6f}/{v0:.6f}")

# --- a drafted blind hole in a rod, turned (rigid: the rim stays on the cylinder)
clear()
R = rod()
R = cross_hole(2.0, depth=1.5)
v_plain = vol(R)
clear()
R = rod()
R = extrude([circle(0.3, (0, -2, 2.0), normal=(0, 1, 0))], 1.5, body="CUT", direction="Y", draft=6.0)
v0, n0 = vol(R), nfaces(R)
r = move(R, (0, -0.75, 2.0), turn=60.0)
check("a drafted blind hole turned in a rod", r == {'FINISHED'} and solid(R) and nfaces(R) == n0
      and abs(vol(R) - v0) < 1e-5 and abs(vol(R) - v_plain) > 1e-3, f"vol {vol(R):.6f}/{v0:.6f}")

# --- a boss on a rod, turned by its top ------------------------------------------------
clear()
R = rod()
R = extrude([circle(0.25, (0, -0.5, 2), normal=(0, -1, 0))], -1.0, body="JOIN", direction="Y")
v0, n0 = vol(R), nfaces(R)
r = move(R, (0, -1.5, 2.0), turn=90.0, slide=-0.5)
top = min(store.get_brep(R).faces(), key=lambda f: f.closest_point((1.5, 0, 1.5))[1])
check("a boss turned and slid by its top", r == {'FINISHED'} and solid(R) and nfaces(R) == n0
      and abs(vol(R) - v0) < 1e-5 and top.closest_point((1.5, 0, 1.5))[1] < 1e-6, f"vol {vol(R):.6f}/{v0:.6f}")

# --- a tube: a hole through one wall has two seats (outside and inside) ---------------
clear()
T = extrude([circle(1.0, (0, 0, 0)), circle(0.6, (0, 0, 0))], 4.0)
T = cross_hole(2.0, depth=1.5)
v0, n0 = vol(T), nfaces(T)
r = move(T, (0, -0.8, 2.0), turn=60.0)
(q, a), = hole_axes(T, 3.0)
check("a hole through a tube's wall turned (both seats)", r == {'FINISHED'} and solid(T) and nfaces(T) == n0
      and abs(vol(T) - v0) < 1e-5 and abs(angle_of(a) - 150.0) < 1e-6,
      f"at {angle_of(a):.4f}° vol {vol(T):.6f}/{v0:.6f}")

# --- a cone: turned about its axis only ------------------------------------------------
clear()
C = rod(draft=10.0)
C = cross_hole(2.0)
v0 = vol(C)
fb = store.get_brep(C).faces()
seat = max(range(len(fb)), key=lambda k: fb[k].area(1e-6))
mode = pushpull.seat_motion(C, seat)
check("a cone seat turns about its axis", mode is not None and mode[0] == "TURN", mode and mode[0])
r = move(C, (0, 0, 2.0), turn=45.0)
check("the hole through a cone turned 45°", r == {'FINISHED'} and solid(C) and abs(vol(C) - v0) < 1e-5,
      f"vol {vol(C):.6f}/{v0:.6f}")
r = move(C, (0, 0, 2.0), mode="AXIS", axis_point=(0, 0, 0), axis_x=(0, 0, 1), slide=0.5)
check("a slide along a cone refused", r == {'CANCELLED'} and abs(vol(C) - v0) < 1e-5)

# --- an extruded wall that is no cylinder (a smooth, not exact, circle): slid only -----
clear()
n = np.array((0, 0, 1.0))
bpy.ops.nproto.draw_circle(data=json.dumps({"matrix": ds.frame(np.zeros(3), np.array((1.0, 0, 0)), n).tolist()}),
                           radius=1.0, ctype="SMOOTH", spans_u=8)
E = extrude([ctx.active_object], 4.0)
E = cross_hole(2.0)
v0, n0 = vol(E), nfaces(E)
fb = store.get_brep(E).faces()
seat = max(range(len(fb)), key=lambda k: fb[k].area(1e-6))
mode = pushpull.seat_motion(E, seat)
check("an extruded seat slides along the extrusion", mode is not None and mode[0] == "SLIDE"
      and abs(abs(float(mode[2][2])) - 1) < 1e-6, mode and mode[0])
r = move(E, (0, 0, 2.0), slide=-0.6)
(q, a), = hole_axes(E)
check("the hole slid down the extruded wall", r == {'FINISHED'} and solid(E) and nfaces(E) == n0
      and abs(vol(E) - v0) < 1e-5 and abs(q[2] - 1.4) < 1e-6, f"z {q[2]:.7f} vol {vol(E):.6f}/{v0:.6f}")
r = move(E, (0, 0, 1.4), mode="AXIS", axis_point=(0, 0, 0), axis_x=(0, 0, 1), turn=30.0)
check("a turn on a wall that is no cylinder refused", r == {'CANCELLED'} and abs(vol(E) - v0) < 1e-5)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
