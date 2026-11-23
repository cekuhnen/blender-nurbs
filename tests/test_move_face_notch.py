"""Move Face on notches: features cut into the edge of the faces they sit in,
slid along that edge:
    blender -b --factory-startup --python tests/test_move_face_notch.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import drawshapes as ds, features, pushpull, steps, store, subsel

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def poly(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

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

def extrude(curves, h, body="NEW"):
    bpy.ops.nproto.extrude(inputs=ins(*curves), distance=h, direction="Z", body=body)
    return ctx.active_object

def circle(r, at):
    bpy.ops.curve.primitive_nurbs_circle_add(radius=r, location=at)
    return ctx.active_object

def move(o, pts, dx, dy):
    return run(bpy.ops.nproto.move_face, faces=faces(*[(o, p) for p in pts]), axis_x=(1, 0, 0), axis_y=(0, 1, 0),
               move_x=dx, move_y=dy)

def curved_centre(o):
    """Centre (x, y) of the body's curved faces' bounding boxes."""
    cs = []
    for f in store.get_brep(o).faces():
        if not f.surface().is_planar():
            lo, hi = (np.array(v, float) for v in f.bounding_box())
            cs.append((lo + hi) / 2)
    return np.mean(cs, axis=0)[:2] if cs else None


def plate():
    return extrude([poly("R", [(0, 0, 0), (6, 0, 0), (6, 4, 0), (0, 4, 0)])], 1.0)

def wall_at(o, p, n):
    """Is there a flat face through p with normal +-n?"""
    for f in store.get_brep(o).faces():
        if f.surface().is_planar() and f.closest_point(tuple(p))[1] < 1e-6:
            q = np.array(f.interior_point(), float)
            if abs(abs(float(np.dot(q - np.asarray(p, float), n))) - 0) < 1e-6:
                return True
    return False

def slide(o, p, d):
    return run(bpy.ops.nproto.move_face, faces=faces((o, p)), slide=d)

ctx.scene.nproto_settings.construction_history = True

# --- a rectangular notch through a plate's front edge ------------------------------------------
clear()
P = plate()
P = extrude([poly("N", [(2, -1, -1), (3, -1, -1), (3, 1, -1), (2, 1, -1)])], 3.0, body="CUT")
v0, n0 = vol(P), nfaces(P)
fb = store.get_brep(P).faces()
k = pushpull._face_index(P, (2.5, 1.0, 0.5))
feat = pushpull.feature_of(P, [k])
check("a notch is found from one of its faces", feat is not None and feat[4] and len(feat[0]) == 3
      and len(feat[1]) == 4, feat and (feat[0], feat[1]))
check("it slides along the edge it is cut into", np.allclose(np.abs(pushpull.notch_direction(P, feat)), [1, 0, 0]))
t0 = time.perf_counter()
r = slide(P, (2.5, 1.0, 0.5), 1.5)
dt = time.perf_counter() - t0
check("the notch slid 1.5 along the edge", r == {'FINISHED'} and solid(P) and nfaces(P) == n0
      and abs(vol(P) - v0) < 1e-6 and wall_at(P, (3.5, 0.5, 0.5), [1, 0, 0]) and wall_at(P, (4.5, 0.5, 0.5), [1, 0, 0])
      and wall_at(P, (4.0, 1.0, 0.5), [0, 1, 0]), f"vol {vol(P):.6f}/{v0:.6f} {dt * 1000:.0f} ms")
check("move time", dt < 2.0, f"{dt * 1000:.0f} ms")
st = features.stack(P)
check("the slide is a step", st[-1]["op"] == "move_face" and st[-1]["params"]["mode"] == "SLIDE"
      and steps.LABEL["move_face"](st[-1]["params"]) == "Move Face · 1.5", steps.LABEL["move_face"](st[-1]["params"]))
features.rebuild(P, ctx)
check("it replays on rebuild", wall_at(P, (3.5, 0.5, 0.5), [1, 0, 0]) and solid(P))
st[-1]["params"]["slide"] = -1.0
features._write_stack(P, st)
features.rebuild(P, ctx)
check("the step edited to -1", wall_at(P, (1.0, 0.5, 0.5), [1, 0, 0]) and wall_at(P, (2.0, 0.5, 0.5), [1, 0, 0])
      and solid(P) and abs(vol(P) - v0) < 1e-6)
v = vol(P)
r = slide(P, (1.5, 1.0, 0.5), -1.0)
check("a slide onto the plate's end refused", r == {'CANCELLED'} and abs(vol(P) - v) < 1e-9)
r = run(bpy.ops.nproto.move_face, faces=faces((P, (1.5, 1.0, 0.5))), mode="PLANE", axis_x=(0, 1, 0),
        axis_y=(1, 0, 0), move_x=0.5)
check("a move off the edge refused", r == {'CANCELLED'} and abs(vol(P) - v) < 1e-9)
r = run(bpy.ops.nproto.move_face, faces=faces((P, (5.0, 3.0, 1.0))), slide=1.0)
check("the plate's top is still no feature", r == {'CANCELLED'})

# --- a U notch (two walls and a round end), with a hole further along --------------------------
ctx.scene.nproto_settings.construction_history = False
clear()
P = plate()
bpy.ops.nproto.draw_circle(data=json.dumps({"matrix": ds.frame(np.array([4.0, 1.0, -1]), np.array([1.0, 0, 0]),
                                                                np.array([0, 0, 1.0])).tolist()}), radius=0.4, ctype="EXACT")
C = ctx.active_object
extrude([C], 3.0, body="CUT")
P = extrude([poly("S", [(3.6, -1, -1), (4.4, -1, -1), (4.4, 1, -1), (3.6, 1, -1)])], 3.0, body="CUT")
bpy.ops.nproto.draw_circle(data=json.dumps({"matrix": ds.frame(np.array([1.5, 0.8, 2]), np.array([1.0, 0, 0]),
                                                                np.array([0, 0, 1.0])).tolist()}), radius=0.3, ctype="EXACT")
P = extrude([ctx.active_object], -3.0, body="CUT")
v0, n0 = vol(P), nfaces(P)
r = slide(P, (4.0, 1.4, 0.5), -1.2)
check("a U notch slid along the edge", r == {'FINISHED'} and solid(P) and nfaces(P) == n0 and abs(vol(P) - v0) < 1e-5
      and wall_at(P, (2.4, 0.5, 0.5), [1, 0, 0]), f"vol {vol(P):.6f}/{v0:.6f}")
r = slide(P, (2.8, 1.4, 0.5), -1.0)
check("a U notch slid into a hole refused", r == {'CANCELLED'} and abs(vol(P) - v0) < 1e-5)

# --- a blind notch (a step down from the top at the front edge) --------------------------------
clear()
P = plate()
P = extrude([poly("B", [(2, -1, 2), (3.2, -1, 2), (3.2, 0.8, 2), (2, 0.8, 2)])], -1.6, body="CUT")
v0, n0 = vol(P), nfaces(P)
r = slide(P, (2.6, 0.4, 0.4), 2.0)
fl = min(store.get_brep(P).faces(), key=lambda f: f.closest_point((4.6, 0.4, 0.4))[1])
check("a blind notch (a step) slid by its floor", r == {'FINISHED'} and solid(P) and nfaces(P) == n0
      and abs(vol(P) - v0) < 1e-6 and fl.closest_point((4.6, 0.4, 0.4))[1] < 1e-6, f"vol {vol(P):.6f}/{v0:.6f}")

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
