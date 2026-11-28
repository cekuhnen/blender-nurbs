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
        # the row label in the scene's units, as the feature panel shows the distance
        from blender_nurbs import steps as _steps, subsel as _subsel
        us = ctx.scene.unit_settings
        lab_m = _steps.LABEL["push_pull"](st[1]["params"])
        us.system, us.length_unit = 'IMPERIAL', 'INCHES'
        lab_in = _steps.LABEL["push_pull"](st[1]["params"])
        us.system, us.length_unit = 'METRIC', 'METERS'
        check("[history] Push / Pull label in scene units", lab_m == "Push · 0.5 m" and lab_in == 'Push · 19.69"',
              f"{lab_m!r} {lab_in!r}")
        # the picked step highlights the faces it moved, where they are now;
        # the first step's top was moved again by the second: nothing for it
        hl = []
        for k in range(3):
            bpy.ops.nproto.tree_step(name=B.name, root=B.name, index=k)
            hl.append(sorted(_subsel.step_faces(B)))
        bf = store.get_brep(B).faces()
        at = lambda q: min(range(len(bf)), key=lambda j: bf[j].closest_point(q)[1])
        want2 = sorted({at((2, 1.5, 3.5)), at((5, 1.5, 1))})
        check("[history] picked Push / Pull step highlights its moved faces",
              hl[0] == [] and hl[1] == [] and hl[2] == want2, f"{hl} want {want2}")
        B.nproto_step.focus = False
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
    # a curved face is offset: the cylinder's side, out then in (the radius changes)
    v1 = vol(Cy)                                   # radius 1, height 1
    r = run(bpy.ops.nproto.push_pull, faces=faces((Cy, (1, 0, 0.5))), distance=0.2)
    check(f"[{tag}] cylinder side offset out", r == {'FINISHED'} and solid(Cy)
          and abs(vol(Cy) / v1 - 1.44) < 1e-4, f"ratio {vol(Cy) / v1:.6f} want 1.44")
    r = run(bpy.ops.nproto.push_pull, faces=faces((Cy, (1.2, 0, 0.5))), distance=-0.4)
    check(f"[{tag}] cylinder side offset in", r == {'FINISHED'} and solid(Cy)
          and abs(vol(Cy) / v1 - 0.64) < 1e-4, f"ratio {vol(Cy) / v1:.6f} want 0.64")
    if hist:
        e = features.stack(Cy)[-1]
        check("[history] a curved face's step says so", e["params"].get("flat") == [False], e["params"].get("flat"))
        v2 = vol(Cy)
        features.rebuild(Cy, ctx)
        check("[history] the offset replays on rebuild", abs(vol(Cy) - v2) < 1e-6, f"{vol(Cy):.6f} vs {v2:.6f}")
    v1 = vol(Cy)
    r = run(bpy.ops.nproto.push_pull, faces=faces((Cy, (0.8, 0, 0.5))), distance=-0.9)
    check(f"[{tag}] offset past the axis refused", r == {'CANCELLED'} and abs(vol(Cy) - v1) < 1e-9)
    r = run(bpy.ops.nproto.push_pull, faces=faces((Cy, (0.2, 0.2, 1))), distance=-3.0)
    check(f"[{tag}] push through everything refused", r == {'CANCELLED'} and abs(vol(Cy) - v1) < 1e-9)

# --- a hole through a box: offset smaller and bigger
for hist in (True, False):
    hist_on(hist)
    tag = "history" if hist else "no history"
    clear()
    B, _ = box(-2, -1.5, 2, 1.5, 2)
    bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 0, 2))
    Hc = ctx.active_object
    bpy.ops.nproto.extrude(inputs=ins((Hc, "S0")), distance=-2.0, direction="Z", body="CUT")
    X = ctx.active_object
    hole = (24 - vol(X)) / 2                       # the hole's cross-section (pi r^2 of this circle)
    r = run(bpy.ops.nproto.push_pull, faces=faces((X, (0.5, 0, 1))), distance=0.1)
    want = 24 - 2 * hole * 0.64
    check(f"[{tag}] hole offset smaller", r == {'FINISHED'} and solid(X) and abs(vol(X) - want) < 1e-4,
          f"vol {vol(X):.5f} want {want:.5f}")
    r = run(bpy.ops.nproto.push_pull, faces=faces((X, (0.4, 0, 1))), distance=-0.3)
    want = 24 - 2 * hole * 1.96
    check(f"[{tag}] hole offset bigger", r == {'FINISHED'} and solid(X) and abs(vol(X) - want) < 1e-4,
          f"vol {vol(X):.5f} want {want:.5f}")

# --- leaning neighbours: extended on their own surfaces (exact, no step)
def frustum(R1, R2, h):
    return math.pi * h / 3 * (R1 * R1 + R1 * R2 + R2 * R2)

def drafted(H, sgn, ang=10.0, w0=4.0, h0=3.0):
    """Volume of a w0 x h0 rectangle extruded H with draft ang (sgn +1 wider going up)."""
    t = sgn * math.tan(math.radians(ang))
    a, b = w0 + 2 * H * t, h0 + 2 * H * t      # top sides; the area is quadratic in z: Simpson is exact
    m = (w0 + H * t) * (h0 + H * t)
    return H / 6 * (w0 * h0 + 4 * m + a * b)

for hist in (True, False):
    hist_on(hist)
    tag = "history" if hist else "no history"
    clear()
    R = poly("R", [(-2, -1.5, 0), (2, -1.5, 0), (2, 1.5, 0), (-2, 1.5, 0)])
    bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=2.0, direction="Z", draft=10.0, body="NEW")
    D = ctx.active_object
    t0 = time.perf_counter()
    r = run(bpy.ops.nproto.push_pull, faces=faces((D, (0, 0, 2))), distance=1.0)
    dt = time.perf_counter() - t0
    check(f"[{tag}] drafted box: top pulled, the draft carries on", r == {'FINISHED'} and solid(D)
          and abs(vol(D) - drafted(3, 1)) < 1e-6 and store.get_brep(D).face_count() == 6,
          f"vol {vol(D):.6f} want {drafted(3, 1):.6f} faces {store.get_brep(D).face_count()} {dt * 1000:.0f} ms")
    r = run(bpy.ops.nproto.push_pull, faces=faces((D, (0, 0, 3))), distance=-1.5)
    check(f"[{tag}] drafted box: top pushed", r == {'FINISHED'} and abs(vol(D) - drafted(1.5, 1)) < 1e-6,
          f"vol {vol(D):.6f} want {drafted(1.5, 1):.6f}")
    if hist:
        features.rebuild(D, ctx)
        check("[history] the exact steps replay", abs(vol(D) - drafted(1.5, 1)) < 1e-6, f"vol {vol(D):.6f}")
    clear()
    bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0))
    a = ctx.active_object
    bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 0, 2))
    b = ctx.active_object
    bpy.ops.nproto.loft(inputs=ins((a, "S0"), (b, "S0")), body="NEW")
    Fr = ctx.active_object
    v0 = vol(Fr)
    ex = frustum(1, 0.5, 2) / v0                  # the loft's circles are exact: ex ~ 1
    r = run(bpy.ops.nproto.push_pull, faces=faces((Fr, (0.1, 0.1, 2))), distance=1.0)
    want = frustum(1, 0.25, 3) / ex
    check(f"[{tag}] frustum: top pulled, the cone carries on", r == {'FINISHED'} and solid(Fr)
          and abs(vol(Fr) - want) < 1e-5 and store.get_brep(Fr).face_count() == 3,
          f"vol {vol(Fr):.6f} want {want:.6f} faces {store.get_brep(Fr).face_count()}")
    clear()
    a = bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)) and ctx.active_object
    b = bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 0, 2)) and ctx.active_object
    bpy.ops.nproto.loft(inputs=ins((a, "S0"), (b, "S0")), body="NEW")
    Fr = ctx.active_object
    k = math.sqrt(1 + 0.25 ** 2)                  # the side's slope 1/4: its radius grows by d k
    r = run(bpy.ops.nproto.push_pull, faces=faces((Fr, (0.75, 0, 1))), distance=0.1)
    want = frustum(1 + 0.1 * k, 0.5 + 0.1 * k, 2) / ex
    check(f"[{tag}] frustum: side offset, the caps follow", r == {'FINISHED'} and solid(Fr)
          and abs(vol(Fr) - want) < 1e-5, f"vol {vol(Fr):.6f} want {want:.6f}")

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
