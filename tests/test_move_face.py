"""Move Face: features (holes, pockets, bosses) slid along the face they sit in:
    blender -b --factory-startup --python tests/test_move_face.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, subsel

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

def plate_with_hole():
    extrude([poly("R", [(0, 0, 0), (6, 0, 0), (6, 4, 0), (0, 4, 0)])], 1.0)
    return extrude([circle(0.5, (2, 2, 1))], -1.0, body="CUT")

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"
    clear()
    P = plate_with_hole()
    v0, c0 = vol(P), curved_centre(P)
    t0 = time.perf_counter()
    r = move(P, [(2.5, 2, 0.5)], 1.5, 0.0)
    dt = time.perf_counter() - t0
    c1 = curved_centre(P)
    check(f"[{tag}] hole slid along the plate", r == {'FINISHED'} and solid(P) and nfaces(P) == 7
          and abs(vol(P) - v0) < 1e-5 and np.allclose(c1 - c0, [1.5, 0], atol=1e-6),
          f"vol {vol(P):.6f} / {v0:.6f} centre {c1} {dt * 1000:.0f} ms")
    check(f"[{tag}] move time", dt < 2.0, f"{dt * 1000:.0f} ms")
    if hist:
        st = features.stack(P)
        check("[history] the move is a step", [e["op"] for e in st][-1:] == ["move_face"], [e["op"] for e in st])
        features.rebuild(P, ctx)
        check("[history] the move replays on rebuild", np.allclose(curved_centre(P), [3.5, 2], atol=1e-6),
              curved_centre(P))
        st[-1]["params"]["move_y"] = 0.75
        features._write_stack(P, st)
        features.rebuild(P, ctx)
        check("[history] the step's Y edited", np.allclose(curved_centre(P), [3.5, 2.75], atol=1e-6),
              curved_centre(P))
    v = vol(P)
    r = move(P, [(4.0, 2.0, 0.5)] if not hist else [(4.0, 2.75, 0.5)], 0.0, 2.0)
    check(f"[{tag}] a move off the plate refused", r == {'CANCELLED'} and abs(vol(P) - v) < 1e-9)
    r = run(bpy.ops.nproto.move_face, faces=faces((P, (5.5, 3.5, 1))), axis_x=(1, 0, 0), axis_y=(0, 1, 0),
            move_x=1.0)
    check(f"[{tag}] the plate's own top is no feature", r == {'CANCELLED'} and abs(vol(P) - v) < 1e-9)
    r = run(bpy.ops.nproto.move_face, faces=faces((P, tuple(curved_centre(P)) + (0.5,))), axis_x=(1, 0, 0),
            axis_y=(0, 0, 1), move_y=0.5)
    # (the point is the hole's axis: the wall is the nearest face)
    check(f"[{tag}] a move out of the plate's face refused", r == {'CANCELLED'} and abs(vol(P) - v) < 1e-9)

    # a pocket (picked by its floor) and a boss (picked by its top)
    clear()
    P = plate_with_hole()
    K = poly("K", [(3.5, 0.5, 1), (5.5, 0.5, 1), (5.5, 1.5, 1), (3.5, 1.5, 1)])
    P = extrude([K], -0.5, body="CUT")
    v0 = vol(P)
    r = move(P, [(4.5, 1, 0.5)], 0.0, 1.0)
    check(f"[{tag}] pocket slid by its floor", r == {'FINISHED'} and solid(P) and nfaces(P) == 12
          and abs(vol(P) - v0) < 1e-5, f"vol {vol(P):.6f} / {v0:.6f}")
    floor = min(store.get_brep(P).faces(), key=lambda f: f.closest_point((4.5, 2.0, 0.5))[1])
    check(f"[{tag}] the floor is where it went", floor.closest_point((4.5, 2.0, 0.5))[1] < 1e-6)
    r = move(P, [(4.5, 2, 0.5)], -2.0, 0.0)
    check(f"[{tag}] pocket into the hole refused", r == {'CANCELLED'} and abs(vol(P) - v0) < 1e-5)
    P = extrude([circle(0.4, (1, 1, 1))], 0.8, body="JOIN")
    v0 = vol(P)
    r = move(P, [(1, 1, 1.8)], 0.5, 0.0)
    top = min(store.get_brep(P).faces(), key=lambda f: f.closest_point((1.5, 1, 1.8))[1])
    check(f"[{tag}] boss slid by its top", r == {'FINISHED'} and solid(P) and abs(vol(P) - v0) < 1e-5
          and top.closest_point((1.5, 1, 1.8))[1] < 1e-6, f"vol {vol(P):.6f} / {v0:.6f}")

# --- a hole in a boss's top goes with the boss
ctx.scene.nproto_settings.construction_history = True
clear()
extrude([poly("R", [(0, 0, 0), (6, 0, 0), (6, 4, 0), (0, 4, 0)])], 1.0)
P = extrude([circle(1.0, (2, 2, 1))], 1.0, body="JOIN")
P = extrude([circle(0.3, (2, 2, 2))], -0.5, body="CUT")
v0, n0 = vol(P), nfaces(P)
r = move(P, [(2, 1, 1.5)], 1.5, 0.0)            # the boss's wall
wall = [f for f in store.get_brep(P).faces() if not f.surface().is_planar()]
cs = sorted(float(np.mean([np.array(v, float) for v in f.bounding_box()], axis=0)[0]) for f in wall)
check("a hole in the boss moves with it", r == {'FINISHED'} and solid(P) and nfaces(P) == n0
      and abs(vol(P) - v0) < 1e-5 and np.allclose(cs, [3.5, 3.5], atol=1e-6), f"x {cs} vol {vol(P):.6f}")

# --- a drafted pocket slides on a flat face like any other (the rim stays on it)
clear()
extrude([poly("R", [(0, 0, 0), (6, 0, 0), (6, 4, 0), (0, 4, 0)])], 1.0)
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(2, 2, 1))
bpy.ops.nproto.extrude(inputs=ins(ctx.active_object), distance=-0.6, direction="Z", body="CUT", draft=8.0)
P = ctx.active_object
v0, n0 = vol(P), nfaces(P)
r = move(P, [(2, 2, 0.4)], 1.5, 0.5)
fl = min(store.get_brep(P).faces(), key=lambda f: f.closest_point((3.5, 2.5, 0.4))[1])
check("a drafted pocket slid on a flat face", r == {'FINISHED'} and solid(P) and nfaces(P) == n0
      and abs(vol(P) - v0) < 1e-5 and fl.closest_point((3.5, 2.5, 0.4))[1] < 1e-6, f"vol {vol(P):.6f}/{v0:.6f}")

# --- Face mode picks; the directions come from the face it sits in
clear()
P = plate_with_hole()
for o in ctx.selected_objects: o.select_set(False)
P.select_set(True); ctx.view_layer.objects.active = P
br = store.get_brep(P)
wall = next(k for k in range(br.face_count()) if not br.faces()[k].surface().is_planar())
subsel.set_picks(P, "FACE", [wall])
r = run(bpy.ops.nproto.move_face, 'INVOKE_DEFAULT', move_x=1.0)
check("Face mode pick taken (X along the plate)", r == {'FINISHED'} and np.allclose(curved_centre(P), [3, 2], atol=1e-6),
      curved_centre(P))
pk = subsel.picks(P, "FACE")
check("the moved feature stays picked", len(pk) == 1 and not store.get_brep(P).faces()[pk[0]].surface().is_planar(), pk)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
