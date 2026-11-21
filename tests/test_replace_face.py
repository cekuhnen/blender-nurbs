"""Replace Face: a face of a solid replaced by a surface:
    blender -b --factory-startup --python tests/test_replace_face.py"""
import json, math, os, sys, time
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, subsel
from blender_nurbs.kernel import load as load_kernel

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
sm = load_kernel()
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def poly(name, pts, cyclic=True):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])
def vol(o): return abs(store.get_brep(o).volume(1e-6))
def solid(o): return bool(store.get_brep(o).is_manifold_solid())
def faces(*items): return json.dumps([[o.name, list(p)] for o, p in items])

def run(op, *a, **kw):
    try:
        return op(*a, **kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

def box(x0, y0, x1, y1, h):
    R = poly("R", [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)])
    bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=h, direction="Z", body="NEW")
    return ctx.active_object

def sheet(name, z, x0=-3, x1=3, y0=-2.5, y1=2.5):
    """A bilinear surface over [x0, x1] x [y0, y1] at height z(x, y) (exact for planes)."""
    c = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    s = sm.create_surface_from_corner_points(*[(x, y, z(x, y)) for x, y in c])
    return store.new_object(ctx, name, s)

def hist_on(on):
    ctx.scene.nproto_settings.construction_history = on

for hist in (True, False):
    hist_on(hist)
    tag = "history" if hist else "no history"
    clear()
    B = box(-2, -1.5, 2, 1.5, 2)
    S = sheet("Above", lambda x, y: 2.5 + 0.25 * x)
    t0 = time.perf_counter()
    r = run(bpy.ops.nproto.replace_face, faces=faces((B, (0, 0, 2))), surface=S.name)
    dt = time.perf_counter() - t0
    check(f"[{tag}] top replaced by a surface above (sides extended)", r == {'FINISHED'} and solid(B)
          and abs(vol(B) - 30) < 1e-6 and store.get_brep(B).face_count() == 6,
          f"vol {vol(B):.5f} faces {store.get_brep(B).face_count()}")
    check(f"[{tag}] time", dt < 3.0, f"{dt * 1000:.0f} ms")
    clear()
    B = box(-2, -1.5, 2, 1.5, 2)
    S = sheet("Across", lambda x, y: 2 + 0.5 * x)
    r = run(bpy.ops.nproto.replace_face, faces=faces((B, (0, 0, 2))), surface=S.name)
    check(f"[{tag}] top replaced by a surface across it", r == {'FINISHED'} and solid(B) and abs(vol(B) - 24) < 1e-6,
          f"vol {vol(B):.5f}")
    clear()
    B = box(-2, -1.5, 2, 1.5, 2)
    S = sheet("Below", lambda x, y: 1 + 0.1 * y)
    r = run(bpy.ops.nproto.replace_face, faces=faces((B, (0, 0, 2))), surface=S.name)
    check(f"[{tag}] top replaced by a surface below (sides cut back)", r == {'FINISHED'} and solid(B)
          and abs(vol(B) - 12) < 1e-6, f"vol {vol(B):.5f}")
    # a side face, under a leaning surface
    clear()
    B = box(-2, -1.5, 2, 1.5, 2)
    c = [(2.5, -3, -1), (3.5, -3, 3), (2.5, 3, -1), (3.5, 3, 3)]
    S = store.new_object(ctx, "Side", sm.create_surface_from_corner_points(*c))
    r = run(bpy.ops.nproto.replace_face, faces=faces((B, (2, 0, 1))), surface=S.name)
    check(f"[{tag}] side face replaced", r == {'FINISHED'} and solid(B) and abs(vol(B) - 30) < 1e-6, f"vol {vol(B):.5f}")
    # a cylinder's round top (its side lies on a curved neighbour)
    clear()
    bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0))
    Cc = ctx.active_object
    bpy.ops.nproto.extrude(inputs=ins((Cc, "S0")), distance=2.0, direction="Z", body="NEW")
    Cy = ctx.active_object
    v0 = vol(Cy)
    S = sheet("Dome", lambda x, y: 2.5 + 0.25 * x, -2, 2, -2, 2)
    r = run(bpy.ops.nproto.replace_face, faces=faces((Cy, (0.2, 0.2, 2))), surface=S.name)
    check(f"[{tag}] cylinder top replaced", r == {'FINISHED'} and solid(Cy) and abs(vol(Cy) / v0 - 1.25) < 1e-5,
          f"ratio {vol(Cy) / v0:.6f} want 1.25")
    # refused: a surface that doesn't cover the face, a solid as the surface
    clear()
    B = box(-2, -1.5, 2, 1.5, 2)
    S = sheet("Small", lambda x, y: 2.5, -1, 1, -1, 1)
    r = run(bpy.ops.nproto.replace_face, faces=faces((B, (0, 0, 2))), surface=S.name)
    check(f"[{tag}] a surface too small refused", r == {'CANCELLED'} and abs(vol(B) - 24) < 1e-9)
    B2 = box(5, 0, 6, 1, 1)
    r = run(bpy.ops.nproto.replace_face, faces=faces((B, (0, 0, 2))), surface=B2.name)
    check(f"[{tag}] a solid as the surface refused", r == {'CANCELLED'} and abs(vol(B) - 24) < 1e-9)

# --- a drafted box's top replaced by a tilted sheet: its sides keep their draft up to it
import numpy as np
def drafted_under(t, z, w=4.0, h=3.0, n=1200):
    """Volume of the w x h rectangle drafted outward (tan t) cut by the surface z(x, y): midpoint grid."""
    X = np.linspace(-4, 4, n, endpoint=False) + 4.0 / n
    Y = np.linspace(-3, 3, n, endpoint=False) + 3.0 / n
    x, y = np.meshgrid(X, Y, indexing="ij")
    zmin = np.maximum(0.0, np.maximum((np.abs(x) - w / 2) / t, (np.abs(y) - h / 2) / t))
    return float(np.clip(z(x, y) - zmin, 0, None).sum() * (8.0 / n) * (6.0 / n))
for hist in (True, False):
    hist_on(hist)
    tag = "history" if hist else "no history"
    clear()
    R = poly("R", [(-2, -1.5, 0), (2, -1.5, 0), (2, 1.5, 0), (-2, 1.5, 0)])
    bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=2.0, direction="Z", draft=10.0, body="NEW")
    D = ctx.active_object
    S = sheet("Tilt", lambda x, y: 2.5 + 0.1 * x, -4, 4, -3, 3)
    r = run(bpy.ops.nproto.replace_face, faces=faces((D, (0, 0, 2))), surface=S.name)
    want = drafted_under(math.tan(math.radians(10.0)), lambda x, y: 2.5 + 0.1 * x)
    check(f"[{tag}] drafted box top replaced: the draft carries on to the surface", r == {'FINISHED'} and solid(D)
          and abs(vol(D) - want) < 2e-3 and store.get_brep(D).face_count() == 6,
          f"vol {vol(D):.5f} want ~{want:.5f} faces {store.get_brep(D).face_count()}")

# --- Face mode + selection: the picked face, the selected surface
hist_on(True)
clear()
B = box(-2, -1.5, 2, 1.5, 2)
S = sheet("Above", lambda x, y: 2.5 + 0.25 * x)
for o in ctx.selected_objects: o.select_set(False)
B.select_set(True); S.select_set(True); ctx.view_layer.objects.active = B
top = min(range(store.get_brep(B).face_count()),
          key=lambda k: store.get_brep(B).faces()[k].closest_point((0, 0, 2))[1])
subsel.set_picks(B, "FACE", [top])
r = run(bpy.ops.nproto.replace_face, 'INVOKE_DEFAULT')
check("Face mode pick + selected surface", r == {'FINISHED'} and abs(vol(B) - 30) < 1e-6, f"vol {vol(B):.5f}")
st = features.stack(B)
check("a Replace Face step with the surface as its input", [e["op"] for e in st] == ["replace_face"]
      and st[0]["inputs"][0][0] == S, [e["op"] for e in st])
# --- history: the surface moved up, the body follows
S.location.z += 0.5
ctx.view_layer.update()
features.rebuild(B, ctx)
check("history: the surface moved, the body rebuilt", abs(vol(B) - 36) < 1e-6, f"vol {vol(B):.5f}")
# ... and the extrude under it made taller: its top is still the face replaced
raw = json.loads(B[features.PROP_HIST]); raw["params"]["distance"] = 2.5
B[features.PROP_HIST] = json.dumps(raw)
features.rebuild(B, ctx)
check("history: a taller box keeps its top replaced", abs(vol(B) - 36) < 1e-6, f"vol {vol(B):.5f}")

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
