"""Rotate Face: faces of solids turned about a hinge, neighbours kept on their surfaces:
    blender -b --factory-startup --python tests/test_rotate_face.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, subsel
from blender_nurbs.pushpull import hinge_axis

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

def extrude(curves, h, body="NEW"):
    bpy.ops.nproto.extrude(inputs=ins(*curves), distance=h, direction="Z", body=body)
    return ctx.active_object

def box(x0, y0, x1, y1, h):
    return extrude([poly("R", [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)])], h)

def rot(o, pts, A, B, angle):
    """Rotate Face with the hinge through A, B (turned so + swings out)."""
    from blender_nurbs.pushpull import face_frame, _face_index
    _, pw, nw, _ = face_frame(o, _face_index(o, np.array(pts[0], float)))
    O, z = hinge_axis(pw, nw, A, B)
    return run(bpy.ops.nproto.rotate_face, faces=faces(*[(o, p) for p in pts]), axis_point=list(O),
               axis_dir=list(z), angle=angle)

def wedge(L, W, deg):
    return 0.5 * L * W * W * math.tan(math.radians(deg))

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"
    # a box's top hinged on its front edge: out adds the wedge, in removes it
    clear()
    B = box(0, 0, 4, 3, 2)
    t0 = time.perf_counter()
    r = rot(B, [(2, 1.5, 2)], (0, 0, 2), (4, 0, 2), 10.0)
    dt = time.perf_counter() - t0
    check(f"[{tag}] box top tilted out", r == {'FINISHED'} and solid(B) and nfaces(B) == 6
          and abs(vol(B) - (24 + wedge(4, 3, 10))) < 1e-6, f"vol {vol(B):.6f} want {24 + wedge(4, 3, 10):.6f}")
    check(f"[{tag}] tilt time", dt < 2.0, f"{dt * 1000:.0f} ms")
    if hist:
        st = features.stack(B)
        check("[history] the turn is a step", [e["op"] for e in st] == ["rotate_face"], [e["op"] for e in st])
        features.rebuild(B, ctx)
        check("[history] the turn replays on rebuild", abs(vol(B) - (24 + wedge(4, 3, 10))) < 1e-6,
              f"vol {vol(B):.6f}")
        a10 = math.radians(10.0)
        hl = step_hl(B, 0)
        k = near_face(B, (2, 1.5 * math.cos(a10), 2 + 1.5 * math.sin(a10)))
        check("[history] picked Rotate Face step highlights the tilted top",
              hl == [k] and abs(store.get_brep(B).faces()[k].area(1e-6) - 12 / math.cos(a10)) < 1e-4,
              f"{hl} want [{k}] area {store.get_brep(B).faces()[k].area(1e-6):.6f}")
        st[0]["params"]["angle"] = -20.0
        features._write_stack(B, st)
        features.rebuild(B, ctx)
        check("[history] the step's angle edited", abs(vol(B) - (24 - wedge(4, 3, 20))) < 1e-6,
              f"vol {vol(B):.6f} want {24 - wedge(4, 3, 20):.6f}")
        raw = json.loads(B[features.PROP_HIST])
        raw["params"]["distance"] = 3.0                  # the extrude 3 high instead of 2: the hinge stays put
        B[features.PROP_HIST] = json.dumps(raw)
        features.rebuild(B, ctx)
        check("[history] a taller box: the hinge follows the top's edge", solid(B) and nfaces(B) == 6
              and abs(vol(B) - (36 - wedge(4, 3, 20))) < 1e-6, f"vol {vol(B):.6f} want {36 - wedge(4, 3, 20):.6f}")
    # in, and too far in (the front... the back wall would vanish)
    clear()
    B = box(0, 0, 4, 3, 2)
    r = rot(B, [(2, 1.5, 2)], (0, 0, 2), (4, 0, 2), -30.0)
    check(f"[{tag}] box top tilted in", r == {'FINISHED'} and abs(vol(B) - (24 - wedge(4, 3, 30))) < 1e-6,
          f"vol {vol(B):.6f}")
    v = vol(B)
    r = rot(B, [(2, 1.5, 1.2)], (0, 0, 2), (4, 0, 2), -40.0)
    check(f"[{tag}] turning past a wall's far end refused", r == {'CANCELLED'} and abs(vol(B) - v) < 1e-9)
    # a side hinged on its bottom edge (a draft on one face)
    clear()
    B = box(0, 0, 4, 3, 2)
    r = rot(B, [(4, 1.5, 1)], (4, 0, 0), (4, 3, 0), -15.0)
    check(f"[{tag}] side drafted in about its bottom edge", r == {'FINISHED'} and nfaces(B) == 6
          and abs(vol(B) - (24 - 0.5 * 3 * 4 * math.tan(math.radians(15)))) < 1e-6, f"vol {vol(B):.6f}")
    # a top with a pocket: the pocket's walls are extended to the tilted plane
    clear()
    B = box(0, 0, 4, 3, 2)
    K = poly("K", [(1, 1, 2), (3, 1, 2), (3, 2, 2), (1, 2, 2)])
    bpy.ops.nproto.extrude(inputs=ins(K), distance=-1.0, direction="Z", body="CUT")
    P = ctx.active_object
    r = rot(P, [(0.5, 0.5, 2)], (0, 0, 2), (4, 0, 2), 10.0)
    t = math.tan(math.radians(10))
    want = 22 + wedge(4, 3, 10) - 2 * t * 1.5          # less the wedge over the pocket (y 1..2, x 1..3)
    check(f"[{tag}] holed top tilted, the pocket walls follow", r == {'FINISHED'} and solid(P)
          and nfaces(P) == 11 and abs(vol(P) - want) < 1e-6, f"vol {vol(P):.6f} want {want:.6f}")
    # a cylinder's top about a diameter: the walls stay one face, the volume stays
    clear()
    bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0))
    Cy = extrude([ctx.active_object], 2.0)
    v0 = vol(Cy)
    r = rot(Cy, [(0.2, 0.2, 2)], (-1, 0, 2), (1, 0, 2), 20.0)
    check(f"[{tag}] cylinder top tilted about a diameter", r == {'FINISHED'} and solid(Cy) and nfaces(Cy) == 3
          and abs(vol(Cy) - v0) < 1e-5, f"vol {vol(Cy):.6f} want {v0:.6f} faces {nfaces(Cy)}")
    # a tube's end (a cap with a hole: two walls)
    clear()
    bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0))
    a = ctx.active_object
    bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 0, 0))
    b = ctx.active_object
    T = extrude([a, b], 3.0)
    v0 = vol(T)
    r = rot(T, [(0.75, 0, 3)], (-1, 0, 3), (1, 0, 3), 15.0)
    check(f"[{tag}] tube end tilted", r == {'FINISHED'} and solid(T) and nfaces(T) == 4 and abs(vol(T) - v0) < 1e-5,
          f"vol {vol(T):.6f} want {v0:.6f} faces {nfaces(T)}")
    # two faces at once about one hinge
    clear()
    B = box(0, 0, 4, 3, 2)
    r = rot(B, [(2, 1.5, 2), (2, 3, 1)], (0, 0, 2), (4, 0, 2), 10.0)
    check(f"[{tag}] two faces about one hinge", r == {'FINISHED'} and solid(B) and nfaces(B) == 6,
          f"vol {vol(B):.6f}")

# --- the command takes Face mode + Edge mode picks: positive swings the face out
ctx.scene.nproto_settings.construction_history = True
clear()
B = box(0, 0, 4, 3, 2)
for o in ctx.selected_objects: o.select_set(False)
B.select_set(True); ctx.view_layer.objects.active = B
br = store.get_brep(B)
top = min(range(br.face_count()), key=lambda k: br.faces()[k].closest_point((2, 1.5, 2))[1])
polys = subsel._edge_polys(B)
front = min(range(len(polys)), key=lambda k: np.inf if polys[k] is None else
            float(np.linalg.norm(polys[k].mean(axis=0) - np.array([2, 0, 2]))))
subsel.set_picks(B, "FACE", [top])
subsel.set_picks(B, "EDGE", [front])
r = run(bpy.ops.nproto.rotate_face, 'INVOKE_DEFAULT', angle=10.0)
check("Face and Edge mode picks taken (+ is out)", r == {'FINISHED'} and abs(vol(B) - (24 + wedge(4, 3, 10))) < 1e-6,
      f"vol {vol(B):.6f}")
pk = subsel.picks(B, "FACE")
n = None
if pk:
    from blender_nurbs.pushpull import face_frame
    n = face_frame(B, pk[0])[2]
want_n = np.array([0, -math.sin(math.radians(10)), math.cos(math.radians(10))])
check("the turned face stays picked", n is not None and float(n @ want_n) > 0.9999, n)
# a curved edge can't be a hinge
clear()
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0))
Cy = extrude([ctx.active_object], 2.0)
for o in ctx.selected_objects: o.select_set(False)
Cy.select_set(True); ctx.view_layer.objects.active = Cy
br = store.get_brep(Cy)
topc = min(range(br.face_count()), key=lambda k: br.faces()[k].closest_point((0.2, 0.2, 2))[1])
polys = subsel._edge_polys(Cy)
rim = min(range(len(polys)), key=lambda k: np.inf if polys[k] is None else abs(float(polys[k][:, 2].mean()) - 2))
subsel.set_picks(Cy, "FACE", [topc])
subsel.set_picks(Cy, "EDGE", [rim])
v0 = vol(Cy)
r = run(bpy.ops.nproto.rotate_face, 'INVOKE_DEFAULT', angle=10.0)
check("a curved edge is no hinge", r == {'CANCELLED'} and abs(vol(Cy) - v0) < 1e-9)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
