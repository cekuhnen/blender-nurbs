"""Push / Pull of flat faces of solids:
    blender -b --factory-startup --python tests/test_push_pull.py"""
import json, math, os, sys, time
import bpy

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

def box(x0, y0, x1, y1, h, name="R"):
    R = poly(name, [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)])
    bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=h, direction="Z", body="NEW")
    return ctx.active_object, R

def hist_on(on):
    ctx.scene.nproto_settings.construction_history = on

for hist in (True, False):
    hist_on(hist)
    tag = "history" if hist else "no history"
    clear()
    B, _ = box(0, 0, 4, 3, 2)
    t0 = time.perf_counter()
    r = run(bpy.ops.nproto.push_pull, faces=faces((B, (2, 1.5, 2))), distance=1.0)
    dt = time.perf_counter() - t0
    check(f"[{tag}] pull the top", r == {'FINISHED'} and solid(B) and abs(vol(B) - 36) < 1e-6
          and store.get_brep(B).face_count() == 6, f"vol {vol(B):.5f} faces {store.get_brep(B).face_count()}")
    check(f"[{tag}] pull time", dt < 2.0, f"{dt * 1000:.0f} ms")
    r = run(bpy.ops.nproto.push_pull, faces=faces((B, (2, 1.5, 3))), distance=-0.5)
    check(f"[{tag}] push the top", r == {'FINISHED'} and abs(vol(B) - 30) < 1e-6, f"vol {vol(B):.5f}")
    # two faces at once: the top and the +x side
    r = run(bpy.ops.nproto.push_pull, faces=faces((B, (2, 1.5, 2.5)), (B, (4, 1.5, 1))), distance=1.0)
    check(f"[{tag}] two faces at once", r == {'FINISHED'} and abs(vol(B) - (4 * 3 * 3.5 + 3 * 3.5 * 1)) < 1e-6,
          f"vol {vol(B):.5f}")
    if hist:
        st = features.stack(B)
        check("[history] each push / pull is a step", [e["op"] for e in st] == ["push_pull"] * 3,
              [e["op"] for e in st])
    # a pocket floor pushed deeper, and a face with a hole pulled
    clear()
    B, _ = box(0, 0, 4, 3, 2)
    K = poly("K", [(1, 1, 2), (3, 1, 2), (3, 2, 2), (1, 2, 2)])
    bpy.ops.nproto.extrude(inputs=ins((K, "S0")), distance=-1.0, direction="Z", body="CUT")
    P = ctx.active_object
    r = run(bpy.ops.nproto.push_pull, faces=faces((P, (2, 1.5, 1))), distance=-0.5)
    check(f"[{tag}] pocket floor pushed deeper", r == {'FINISHED'} and abs(vol(P) - (24 - 3)) < 1e-6, f"vol {vol(P):.5f}")
    r = run(bpy.ops.nproto.push_pull, faces=faces((P, (0.5, 0.5, 2))), distance=1.0)
    check(f"[{tag}] holed top pulled (the pocket goes up with it)", r == {'FINISHED'}
          and abs(vol(P) - (24 - 3 + 12 - 2)) < 1e-6, f"vol {vol(P):.5f}")
    # a cylinder's flat top (the boolean can't: its side lies on the cylinder; extruded instead)
    clear()
    bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0))
    Cc = ctx.active_object
    bpy.ops.nproto.extrude(inputs=ins((Cc, "S0")), distance=2.0, direction="Z", body="NEW")
    Cy = ctx.active_object
    v0 = vol(Cy)
    r = run(bpy.ops.nproto.push_pull, faces=faces((Cy, (0.2, 0.2, 2))), distance=1.0)
    check(f"[{tag}] cylinder top pulled", r == {'FINISHED'} and solid(Cy) and abs(vol(Cy) - v0 * 1.5) < 1e-4,
          f"vol {vol(Cy):.5f} want {v0 * 1.5:.5f}")
    r = run(bpy.ops.nproto.push_pull, faces=faces((Cy, (0.2, 0.2, 3))), distance=-2.0)
    check(f"[{tag}] cylinder top pushed", r == {'FINISHED'} and solid(Cy) and abs(vol(Cy) - v0 * 0.5) < 1e-4,
          f"vol {vol(Cy):.5f} want {v0 * 0.5:.5f}")
    # refused: a curved face, a push through the whole body, an open sheet
    v1 = vol(Cy)
    r = run(bpy.ops.nproto.push_pull, faces=faces((Cy, (1, 0, 0.5))), distance=0.2)
    check(f"[{tag}] curved face refused", r == {'CANCELLED'} and abs(vol(Cy) - v1) < 1e-9)
    r = run(bpy.ops.nproto.push_pull, faces=faces((Cy, (0.2, 0.2, 1))), distance=-3.0)
    check(f"[{tag}] push through everything refused", r == {'CANCELLED'} and abs(vol(Cy) - v1) < 1e-9)

# --- history: the step follows the body's rebuild (a taller box: its top found by its normal)
hist_on(True)
clear()
B, R = box(0, 0, 4, 3, 2)
bpy.ops.nproto.push_pull(faces=faces((B, (2, 1.5, 2))), distance=1.0)
raw = json.loads(B[features.PROP_HIST])
raw["params"]["distance"] = 3.0                  # the extrude 3 high instead of 2
B[features.PROP_HIST] = json.dumps(raw)
features.rebuild(B, ctx)
check("history: a taller box keeps its top pulled", abs(vol(B) - 4 * 3 * 4) < 1e-6, f"vol {vol(B):.5f}")
# the step's distance edited
st = features.stack(B)
st[0]["params"]["distance"] = 2.0
features._write_stack(B, st)
features.rebuild(B, ctx)
check("history: the step's distance edited", abs(vol(B) - 4 * 3 * 5) < 1e-6, f"vol {vol(B):.5f}")

# --- Face mode picks: the command takes them, and the moved face stays picked
clear()
B, _ = box(0, 0, 4, 3, 2)
for o in ctx.selected_objects: o.select_set(False)
B.select_set(True); ctx.view_layer.objects.active = B
top = min(range(store.get_brep(B).face_count()),
          key=lambda k: store.get_brep(B).faces()[k].closest_point((2, 1.5, 2))[1])
subsel.set_picks(B, "FACE", [top])
r = run(bpy.ops.nproto.push_pull, 'INVOKE_DEFAULT', distance=0.5)
check("Face mode picks taken", r == {'FINISHED'} and abs(vol(B) - 30) < 1e-6, f"vol {vol(B):.5f}")
pk = subsel.picks(B, "FACE")
c = store.get_brep(B).faces()[pk[0]].interior_point() if pk else None
check("the moved face stays picked", len(pk) == 1 and abs((c.z if hasattr(c, "z") else c[2]) - 2.5) < 1e-6, pk)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
