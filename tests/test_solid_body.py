"""Body option (Surface, New Solid, Join, Cut, Intersect) of Extrude, Revolve,
Sweep, Pipe and Loft:
    blender -b --factory-startup --python tests/test_solid_body.py"""
import json, math, os, sys
import bpy
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store

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

def ncircle(name, r, loc):
    bpy.ops.curve.primitive_nurbs_circle_add(radius=r, location=loc)
    o = ctx.active_object; o.name = name; return o

def circle(name, r, loc):
    bpy.ops.curve.primitive_bezier_circle_add(radius=r, location=loc)
    o = ctx.active_object; o.name = name; return o

def rect(name, x0, y0, x1, y1, z=0.0):
    return poly(name, [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

def vol(o):
    return abs(store.get_brep(o).volume(1e-6))

def solid(o):
    return bool(store.get_brep(o).is_manifold_solid())

def run(op, **kw):
    try:
        return op(**kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

def hist_on(on):
    ctx.scene.nproto_settings.construction_history = on

for hist in (True, False):
    hist_on(hist)
    tag = "history" if hist else "no history"
    clear()
    # --- New Solid: a box from a rectangle
    R = rect("R", 0, 0, 4, 3)
    r = bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=2.0, direction="Z", body="NEW")
    B = ctx.active_object
    check(f"[{tag}] extrude new solid", r == {'FINISHED'} and solid(B) and abs(vol(B) - 24.0) < 1e-4,
          f"vol {vol(B):.5f}")
    # --- New Solid with a hole: a rectangle with a circle inside
    R2 = rect("R2", 10, 0, 14, 3)
    C2 = circle("C2", 1.0, (12, 1.5, 0))
    r = bpy.ops.nproto.extrude(inputs=ins((R2, "S0"), (C2, "S0")), distance=2.0, direction="Z", body="NEW")
    H = ctx.active_object
    want = 24.0 - math.pi * 2.0
    check(f"[{tag}] extrude outline with hole: one solid", r == {'FINISHED'} and len(ctx.selected_objects) == 1
          and solid(H) and abs(vol(H) - want) < 3e-3, f"vol {vol(H):.5f} want {want:.5f}")
    # --- Join: a circle on the box's top face, extruded up
    C = circle("C", 1.0, (2, 1.5, 2.0))
    r = bpy.ops.nproto.extrude(inputs=ins((C, "S0")), distance=1.0, direction="Z", body="JOIN")
    J = ctx.active_object
    want = 24.0 + math.pi
    bname = "Extrude"                  # the box's name (B itself is the hidden base with history)
    check(f"[{tag}] extrude join keeps the body's name", r == {'FINISHED'} and J.name == bname, J.name)
    check(f"[{tag}] join volume", solid(J) and abs(vol(J) - want) < 3e-3, f"vol {vol(J):.5f} want {want:.5f}")
    # --- Cut: a rectangle on the top, cut 1 down
    K = rect("K", 0.5, 0.5, 1.5, 2.5, 2.0)
    r = bpy.ops.nproto.extrude(inputs=ins((K, "S0")), distance=-1.0, direction="Z", body="CUT")
    X = ctx.active_object
    want -= 2.0
    check(f"[{tag}] extrude cut", r == {'FINISHED'} and solid(X) and abs(vol(X) - want) < 3e-3,
          f"vol {vol(X):.5f} want {want:.5f}")
    check(f"[{tag}] cut keeps the body's name", X.name == bname, X.name)
    if hist:
        check("[history] the bodies before are kept hidden", not B.visible_get() and not J.visible_get()
              and features.history(X)["op"] == "boolean")
    else:
        check("[no history] the body is changed in place", X is B and len([o for o in bpy.data.objects
              if o.type == 'MESH']) == 2, [o.name for o in bpy.data.objects if o.type == 'MESH'])
    # --- Intersect: a tall cylinder through the box, intersected
    T = circle("T", 1.0, (3.0, 1.5, -1.0))
    r = bpy.ops.nproto.extrude(inputs=ins((T, "S0")), distance=5.0, direction="Z", body="INTERSECT")
    I = ctx.active_object
    check(f"[{tag}] extrude intersect", r == {'FINISHED'} and solid(I) and vol(I) > 0 and vol(I) < want,
          f"vol {vol(I):.5f}")
    # --- open profile: refused as a solid
    O = poly("O", [(20, 0, 0), (21, 0, 0), (21, 1, 0)], cyclic=False)
    try:
        r = bpy.ops.nproto.extrude(inputs=ins((O, "S0")), distance=1.0, direction="Z", body="NEW")
    except RuntimeError as exc:
        r = str(exc)
    check(f"[{tag}] open profile refused", "closed profiles" in str(r), str(r)[:80])

    # --- Revolve: closed profile, full turn (a ring), part turn (caps), open profile on the axis
    clear()
    P = poly("P", [(2, 0, 0), (3, 0, 0), (3, 0, 1), (2, 0, 1)])      # in XZ, off the Z axis
    bpy.ops.object.select_all(action='DESELECT')
    r = bpy.ops.nproto.revolve(inputs=ins((P, "S0")), angle=360.0, axis="Z", pivot="WORLD", body="NEW")
    V = ctx.active_object
    want = math.pi * (9 - 4) * 1.0
    check(f"[{tag}] revolve ring solid", r == {'FINISHED'} and solid(V) and abs(vol(V) - want) < 1e-3,
          f"vol {vol(V):.5f} want {want:.5f}")
    P2 = poly("P2", [(2, 0, 0), (3, 0, 0), (3, 0, 1), (2, 0, 1)])
    r = bpy.ops.nproto.revolve(inputs=ins((P2, "S0")), angle=90.0, axis="Z", pivot="WORLD", body="NEW")
    V2 = ctx.active_object
    check(f"[{tag}] revolve 90 deg capped", r == {'FINISHED'} and solid(V2) and abs(vol(V2) - want / 4) < 1e-3,
          f"vol {vol(V2):.5f} want {want / 4:.5f}")
    # half disc profile on the axis: a quarter of a sphere-like wedge (profile x>=0 in XZ, ends on Z)
    S = poly("S", [(0, 0, -1), (1, 0, -1), (1, 0, 1), (0, 0, 1)], cyclic=False)
    r = run(bpy.ops.nproto.revolve, inputs=ins((S, "S0")), angle=360.0, axis="Z", pivot="WORLD", body="NEW")
    V3 = ctx.active_object
    check(f"[{tag}] revolve open profile ends on axis, full turn", r == {'FINISHED'} and solid(V3)
          and abs(vol(V3) - math.pi * 2) < 1e-3, f"vol {vol(V3) if r == {'FINISHED'} else 0:.5f}")
    S2 = poly("S2", [(0, 5, -1), (1, 5, -1), (1, 5, 1), (0, 5, 1)], cyclic=False)
    r = run(bpy.ops.nproto.revolve, inputs=ins((S2, "S0")), angle=90.0, axis_point=(0, 5, 0), axis_dir=(0, 0, 1),
                               axis="PICKED", body="NEW")
    V4 = ctx.active_object
    check(f"[{tag}] revolve open profile on axis, 90 deg (half caps)", r == {'FINISHED'} and solid(V4)
          and abs(vol(V4) - math.pi / 2) < 1e-3, f"vol {vol(V4) if r == {'FINISHED'} else 0:.5f}")
    # closed rectangle with a side ON the axis (the kernel refused its degenerate face): that side
    # is left out, the rest revolved -- a disc, as Surface and as a solid (full and part turn)
    for body, ang in (("SURFACE", 360.0), ("NEW", 360.0), ("NEW", 90.0)):
        Q = poly("Q", [(-8, 9, 0), (0, 9, 0), (0, 9, 1), (-8, 9, 1)])   # the side x=0 lies on the axis
        r = run(bpy.ops.nproto.revolve, inputs=ins((Q, "S0")), angle=ang, axis_point=(0, 9, 0),
                axis_dir=(0, 0, 1), axis="PICKED", body=body)
        V5 = ctx.active_object
        want5 = math.pi * 64 * ang / 360.0
        good = r == {'FINISHED'} and solid(V5) and abs(vol(V5) - want5) < 1e-3 * want5
        check(f"[{tag}] revolve rectangle with a side on the axis, {body} {ang:.0f} deg", good,
              f"vol {vol(V5) if r == {'FINISHED'} else 0:.4f} want {want5:.4f}")

# --- draft: an outline with a hole, 5 deg; walls lean the same way (the hole narrows as the outline grows)
import time
hist_on(True)
clear()
R = rect("R", 0, 0, 4, 4)
Q = rect("Q", 1, 1, 3, 3)
t0 = time.perf_counter()
r = run(bpy.ops.nproto.extrude, inputs=ins((R, "S0"), (Q, "S0")), distance=1.0, direction="Z", draft=5.0, body="NEW")
dt = time.perf_counter() - t0
D = ctx.active_object
t = math.tan(math.radians(5.0))
# frustum of the outline minus the hole's inverted frustum: material grows by 2t per side per unit height
def frustum(a, b, h):        # square sides a (bottom) -> b (top)
    return h / 3.0 * (a * a + a * b + b * b)
want = frustum(4, 4 + 2 * t, 1) - frustum(2, 2 - 2 * t, 1)
check("drafted outline with hole", r == {'FINISHED'} and solid(D) and abs(vol(D) - want) < 1e-3,
      f"vol {vol(D) if r == {'FINISHED'} else 0:.5f} want {want:.5f}")
check("solid extrude time", dt < 3.0, f"{dt * 1000:.0f} ms")
# --- symmetric
S = rect("S", 10, 0, 12, 2)
r = run(bpy.ops.nproto.extrude, inputs=ins((S, "S0")), distance=1.0, direction="Z", symmetric=True, body="NEW")
check("symmetric solid", r == {'FINISHED'} and solid(ctx.active_object) and abs(vol(ctx.active_object) - 8.0) < 1e-4)
# --- revolve cut: a groove round a cylinder
clear()
bpy.ops.curve.primitive_nurbs_circle_add(radius=2.0, location=(0, 0, 0))
Cy = ctx.active_object
r = run(bpy.ops.nproto.extrude, inputs=ins((Cy, "S0")), distance=4.0, direction="Z", body="NEW")
Body = ctx.active_object
v0 = vol(Body)
G = poly("G", [(1.5, 0, 1.5), (2.5, 0, 1.5), (2.5, 0, 2.5), (1.5, 0, 2.5)])
t0 = time.perf_counter()
r = run(bpy.ops.nproto.revolve, inputs=ins((G, "S0")), angle=360.0, axis="Z", pivot="WORLD", body="CUT")
dt = time.perf_counter() - t0
X = ctx.active_object
groove = math.pi * (4.0 - 2.25) * 1.0          # the part of the ring inside the cylinder (r 1.5 .. 2)
check("revolve cut groove", r == {'FINISHED'} and solid(X) and abs((v0 - vol(X)) - groove) < 2e-3,
      f"removed {v0 - vol(X):.5f} want {groove:.5f}")
check("revolve cut time", dt < 4.0, f"{dt * 1000:.0f} ms")
# --- Surface stays the default and unchanged
L = poly("L", [(5, 0, 0), (6, 0, 0), (6, 1, 0)], cyclic=False)
r = run(bpy.ops.nproto.extrude, inputs=ins((L, "S0")), distance=1.0, direction="Z")
check("surface extrude (open profile) still works", r == {'FINISHED'} and not solid(ctx.active_object))

# --- history: moving the profile rebuilds the joined body
hist_on(True)
clear()
R = rect("R", 0, 0, 4, 3)
bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=2.0, direction="Z", body="NEW")
C = circle("C", 1.0, (2, 1.5, 2.0))
bpy.ops.nproto.extrude(inputs=ins((C, "S0")), distance=1.0, direction="Z", body="JOIN")
J = ctx.active_object
v0 = vol(J)
C.scale = (0.5, 0.5, 1.0)
tool = next(o for o in bpy.data.objects if features.history(o) and features.history(o)["op"] == "extrude"
            and o.get("nproto_in_0") is C)
features.rebuild(tool, ctx)
features.rebuild(J, ctx)
check("history: a smaller circle rebuilds the joined body", abs(vol(J) - (24.0 + math.pi * 0.25)) < 1e-3,
      f"vol {vol(J):.5f} (was {v0:.5f})")

# --- Sweep, Pipe, Loft ------------------------------------------------------
for hist in (True, False):
    hist_on(hist)
    tag = "history" if hist else "no history"
    clear()
    rod = math.pi * 0.25 * 3.0                   # r 0.5, length 3
    # Sweep: a circle along a line
    P = ncircle("P", 0.5, (0, 0, 0)); L = poly("L", [(0, 0, 0), (0, 0, 3)], cyclic=False)
    r = run(bpy.ops.nproto.sweep, inputs=ins((P, "S0"), (L, "S0")), body="NEW")
    S = ctx.active_object
    check(f"[{tag}] sweep new solid", r == {'FINISHED'} and solid(S) and abs(vol(S) - rod) < 1e-4,
          f"vol {vol(S):.5f} want {rod:.5f}")
    # Sweep along a bent rail: a solid, the volume by Pappus (rail length x area)
    P2 = ncircle("P2", 0.5, (10, 0, 0)); A = poly("A", [(10, 0, 0), (10, 0, 2), (12, 0, 4)], cyclic=False)
    r = run(bpy.ops.nproto.sweep, inputs=ins((P2, "S0"), (A, "S0")), body="NEW")
    want = math.pi * 0.25 * (2 + 2 * math.sqrt(2))
    check(f"[{tag}] sweep bent rail solid", r == {'FINISHED'} and solid(ctx.active_object)
          and abs(vol(ctx.active_object) - want) / want < 0.02, f"vol {vol(ctx.active_object):.5f} ~ {want:.5f}")
    # open profile refused
    O = poly("O", [(20, 0, 0), (21, 0, 0)], cyclic=False); L3 = poly("L3", [(20, 0, 0), (20, 0, 1)], cyclic=False)
    try:
        r = bpy.ops.nproto.sweep(inputs=ins((O, "S0"), (L3, "S0")), body="NEW")
    except RuntimeError as exc:
        r = str(exc)
    check(f"[{tag}] sweep open profile refused", "closed profile" in str(r), str(r)[:90])
    # Pipe: two lines, a rod each
    clear()
    L1 = poly("L1", [(0, 0, 0), (0, 0, 3)], cyclic=False); L2 = poly("L2", [(5, 0, 0), (5, 0, 3)], cyclic=False)
    r = run(bpy.ops.nproto.pipe, inputs=ins((L1, "S0"), (L2, "S0")), radius=0.5, body="NEW")
    made = list(ctx.selected_objects)
    check(f"[{tag}] pipe new solids, one per curve", r == {'FINISHED'} and len(made) == 2
          and all(solid(o) and abs(vol(o) - rod) < 1e-4 for o in made), [f"{vol(o):.5f}" for o in made])
    # Pipe round a closed circle: a torus (one face closed both ways), halved by Intersect with a box
    C = ncircle("C", 3.0, (20, 0, 0))
    Rb = rect("Rb", 20, -5, 25, 5, -1.0)
    run(bpy.ops.nproto.extrude, inputs=ins((Rb, "S0")), distance=2.0, direction="Z", body="NEW")
    box = ctx.active_object
    torus = 2 * math.pi * 3.0 * math.pi * 0.25
    r = run(bpy.ops.nproto.pipe, inputs=ins((C, "S0")), radius=0.5, body="INTERSECT", target=box.name)
    I = ctx.active_object
    check(f"[{tag}] pipe torus intersected with a box: half the torus", r == {'FINISHED'} and solid(I)
          and abs(vol(I) - torus / 2) < 2e-3, f"vol {vol(I):.5f} want {torus / 2:.5f}")
    # Pipe Cut: a hole through a box
    clear()
    R = rect("R", 0, 0, 4, 4)
    run(bpy.ops.nproto.extrude, inputs=ins((R, "S0")), distance=4.0, direction="Z", body="NEW")
    H = poly("H", [(2, 2, -1), (2, 2, 5)], cyclic=False)
    r = run(bpy.ops.nproto.pipe, inputs=ins((H, "S0")), radius=0.5, body="CUT")
    X = ctx.active_object
    want = 64.0 - math.pi * 0.25 * 4.0
    check(f"[{tag}] pipe cut a hole", r == {'FINISHED'} and solid(X) and abs(vol(X) - want) < 1e-3
          and X.name == "Extrude", f"vol {vol(X):.5f} want {want:.5f} {X.name}")
    # Loft: two circles (a frustum) joined on top of the box
    a = ncircle("a", 1.0, (2, 2, 4)); b = ncircle("b", 0.5, (2, 2, 6))
    r = run(bpy.ops.nproto.loft, inputs=ins((a, "S0"), (b, "S0")), body="JOIN")
    J = ctx.active_object
    fr = math.pi * 2 / 3 * (1 + 0.5 + 0.25)
    check(f"[{tag}] loft join a frustum", r == {'FINISHED'} and solid(J) and abs(vol(J) - (want + fr)) < 1e-3
          and J.name == "Extrude", f"vol {vol(J):.5f} want {want + fr:.5f} {J.name}")
    # Loft through three squares (folds: a body of faces), new solid
    sq = [rect(f"q{i}", 10 - s, -s, 10 + s, s, z) for i, (s, z) in enumerate([(1, 0), (0.5, 1), (1, 2)])]
    r = run(bpy.ops.nproto.loft, inputs=ins(*[(o, "S0") for o in sq]), body="NEW")
    want = 2 / 5 + 4 / 3 + 2                   # side z^2 - 2z + 2 (through 2, 1, 2): its square integrated
    check(f"[{tag}] loft through squares solid", r == {'FINISHED'} and solid(ctx.active_object)
          and abs(vol(ctx.active_object) - want) < 1e-3, f"vol {vol(ctx.active_object):.5f} want {want:.5f}")
    # open sections refused
    o1 = poly("o1", [(20, 0, 0), (21, 0, 0)], cyclic=False); o2 = poly("o2", [(20, 0, 1), (21, 0, 1)], cyclic=False)
    try:
        r = bpy.ops.nproto.loft(inputs=ins((o1, "S0"), (o2, "S0")), body="NEW")
    except RuntimeError as exc:
        r = str(exc)
    check(f"[{tag}] loft open sections refused", "closed sections" in str(r), str(r)[:90])
    # Surface stays the default
    r = run(bpy.ops.nproto.loft, inputs=ins((o1, "S0"), (o2, "S0")))
    r2 = run(bpy.ops.nproto.pipe, inputs=ins((o1, "S0"),), radius=0.1)
    check(f"[{tag}] surface loft / pipe still work", r == {'FINISHED'} and r2 == {'FINISHED'}
          and not solid(ctx.active_object))

# --- history: a smaller sweep profile rebuilds the cut body
hist_on(True)
clear()
R = rect("R", 0, 0, 4, 4)
bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=4.0, direction="Z", body="NEW")
P = ncircle("P", 0.5, (2, 2, -1)); L = poly("L", [(2, 2, -1), (2, 2, 5)], cyclic=False)
bpy.ops.nproto.sweep(inputs=ins((P, "S0"), (L, "S0")), body="CUT")
X = ctx.active_object
P.scale = (0.5, 0.5, 1.0)
tool = next(o for o in bpy.data.objects if features.history(o) and features.history(o)["op"] == "sweep")
features.rebuild(tool, ctx)
features.rebuild(X, ctx)
want = 64.0 - math.pi * 0.0625 * 4.0
check("history: a smaller sweep profile rebuilds the cut body", abs(vol(X) - want) < 1e-3,
      f"vol {vol(X):.5f} want {want:.5f}")
check("history: the sweep tool keeps caps", features.history(tool)["params"].get("caps") is True)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
