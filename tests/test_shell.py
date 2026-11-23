"""Shell: a closed solid hollowed into walls, the picked faces taken away:
    blender -b --factory-startup --python tests/test_shell.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, subsel, shell, steps
from blender_nurbs.pushpull import face_frame

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
def nfaces(o): return store.get_brep(o).face_count()

def run(op, *a, **kw):
    try:
        return op(*a, **kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

def extruded(pts, h, name="R"):
    R = poly(name, pts)
    bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=h, direction="Z", body="NEW")
    return ctx.active_object

def select(*objs):
    for o in ctx.selected_objects: o.select_set(False)
    for o in objs: o.select_set(True)
    ctx.view_layer.objects.active = objs[-1]

def preview_on_result(o, removed_idx, before_text, t, side="INSIDE"):
    """How far the preview's inner edges (wall_edges on the body before) lie
    from the shelled body: the preview shows the cavity where it will be."""
    after_text = o[store.PROP_BREP]
    store.set_geometry_text(o, before_text, ctx, keep_untrimmed=False)
    walls = shell.wall_edges(o, removed_idx)
    store.set_geometry_text(o, after_text, ctx, keep_untrimmed=False)
    b = store.get_brep(o)
    Mi = np.array(o.matrix_world.inverted())
    s = t if side == "OUTSIDE" else -t
    V = np.array([v.point() for v in b.vertices()], float)
    worst = 0.0
    for P, D, rim in walls:
        Q = P + s * D
        for q in Q:
            ql = (Mi @ np.r_[q, 1.0])[:3]
            worst = max(worst, min(f.closest_point(tuple(map(float, ql)))[1] for f in b.faces()))
        for q in (Q[0], Q[-1]):        # the ends are corners of the result
            ql = (Mi @ np.r_[q, 1.0])[:3]
            worst = max(worst, float(np.min(np.linalg.norm(V - ql, axis=1))))
    return worst

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"

    # a block, its top taken away: an open box
    clear()
    B = extruded([(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], 2)
    before = B[store.PROP_BREP]
    top = [i for i in range(nfaces(B)) if face_frame(B, i)[1][2] > 1.999]
    t0 = time.perf_counter()
    r = run(bpy.ops.nproto.shell, faces=faces((B, (2, 1.5, 2))), thickness=0.2)
    dt = time.perf_counter() - t0
    want = 24 - 3.6 * 2.6 * 1.8
    check(f"[{tag}] box: top open, walls 0.2 inside", r == {'FINISHED'} and solid(B) and abs(vol(B) - want) < 1e-5
          and nfaces(B) == 11, f"vol {vol(B):.5f} want {want:.5f} faces {nfaces(B)} {dt * 1000:.0f} ms")
    if r == {'FINISHED'}:
        d = preview_on_result(B, top, before, 0.2)
        check(f"[{tag}] box: the preview's inner edges and corners lie on the result", d < 1e-6, f"{d:.2e}")
    if hist:
        st = features.stack(B)
        check("history: one shell step", len(st) == 1 and st[0]["op"] == "shell"
              and steps.LABEL["shell"](st[0]["params"]) == "Shell · 0.2 · 1 face open", str(st))
        # the step's thickness edited: the walls follow
        raw = json.loads(B[features.PROP_STACK]); raw[0]["params"]["thickness"] = 0.5
        B[features.PROP_STACK] = json.dumps(raw)
        features.rebuild(B, ctx, force=True)
        want = 24 - 3.0 * 2.0 * 1.5
        check("history: thickness 0.5 on rebuild", abs(vol(B) - want) < 1e-5, f"vol {vol(B):.5f} want {want:.5f}")
        # the body before the step changed (moved): the step follows
        B.location.x = 10; ctx.view_layer.update()
        features.rebuild(B, ctx, force=True)
        check("history: moved body rebuilds the same shell", abs(vol(B) - want) < 1e-5, f"vol {vol(B):.5f}")

    # an L of a welded frame (one body), both ends taken away: a mitred tube
    clear()
    s, L = 0.5, 5.0
    Lb = extruded([(0, 0, 0), (L, 0, 0), (L, s, 0), (s, s, 0), (s, L, 0), (0, L, 0)], s, "L")
    v0 = vol(Lb)
    before = Lb[store.PROP_BREP]
    ends = [(L, s / 2, s / 2), (s / 2, L, s / 2)]
    idx = [min(range(nfaces(Lb)), key=lambda i: np.linalg.norm(face_frame(Lb, i)[1] - np.array(e))) for e in ends]
    r = run(bpy.ops.nproto.shell, faces=faces(*[(Lb, e) for e in ends]), thickness=0.05)
    cav = (L - 0.05) * 0.4 * 0.4 + (L - 0.45) * 0.4 * 0.4
    check(f"[{tag}] L frame: both ends open, a tube", r == {'FINISHED'} and solid(Lb)
          and abs(vol(Lb) - (v0 - cav)) < 1e-5, f"vol {vol(Lb):.5f} want {v0 - cav:.5f} faces {nfaces(Lb)}")
    if r == {'FINISHED'}:
        d = preview_on_result(Lb, idx, before, 0.05)
        check(f"[{tag}] L frame: the preview's inner edges and corners lie on the result", d < 1e-6, f"{d:.2e}")

    # Face mode picks on two bodies at once (the welded frame's two halves)
    clear()
    A = extruded([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)], 1, "A")
    C = extruded([(5, 0, 0), (7, 0, 0), (7, 1, 0), (5, 1, 0)], 1, "C")
    for o, x in ((A, 2), (C, 5)):
        i = min(range(nfaces(o)), key=lambda k: abs(face_frame(o, k)[1][0] - x) + abs(face_frame(o, k)[1][2] - 0.5))
        subsel.set_picks(o, "FACE", [i])
    select(A, C)
    r = run(bpy.ops.nproto.shell, thickness=0.1)
    want = 2 - 1.9 * 0.8 * 0.8
    check(f"[{tag}] Face mode picks on two bodies", r == {'FINISHED'} and abs(vol(A) - want) < 1e-5
          and abs(vol(C) - want) < 1e-5 and not subsel.picks(A, "FACE"), f"{vol(A):.5f} {vol(C):.5f} want {want:.5f}")

    # walls outside: the body becomes the cavity
    clear()
    B = extruded([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], 2)
    before = B[store.PROP_BREP]
    top = [i for i in range(nfaces(B)) if face_frame(B, i)[1][2] > 1.999]
    r = run(bpy.ops.nproto.shell, faces=faces((B, (1, 1, 2))), thickness=0.1, side="OUTSIDE")
    want = 2.2 * 2.2 * 2.1 - 8
    check(f"[{tag}] walls outside", r == {'FINISHED'} and abs(vol(B) - want) < 1e-5, f"vol {vol(B):.5f} want {want:.5f}")
    if r == {'FINISHED'}:
        d = preview_on_result(B, top, before, 0.1, "OUTSIDE")
        check(f"[{tag}] outside: the preview's outer edges lie on the result", d < 1e-6, f"{d:.2e}")

    # no face: a closed hollow of the selected solid
    clear()
    B = extruded([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], 2)
    select(B)
    r = run(bpy.ops.nproto.shell, thickness=0.1)
    check(f"[{tag}] closed hollow", r == {'FINISHED'} and solid(B) and abs(vol(B) - (8 - 1.8 ** 3)) < 1e-5
          and nfaces(B) == 12, f"vol {vol(B):.5f} faces {nfaces(B)}")

    # a round bar, both caps taken away: a pipe
    clear()
    bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.5, height=3.0, caps=True)
    P = ctx.active_object
    caps = [face_frame(P, i)[0] for i in range(nfaces(P)) if face_frame(P, i)[3]]
    v0 = vol(P)
    r = run(bpy.ops.nproto.shell, faces=faces(*[(P, c) for c in caps]), thickness=0.05)
    h = v0 / (math.pi * 0.25)
    want = math.pi * (0.25 - 0.2025) * h
    check(f"[{tag}] cylinder: both caps open, a pipe", r == {'FINISHED'} and len(caps) == 2
          and abs(vol(P) - want) < 1e-4 * want, f"vol {vol(P):.5f} want {want:.5f} caps {len(caps)}")

    # too thick for the part: refused, the body untouched
    clear()
    B = extruded([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], 2)
    text = B[store.PROP_BREP]
    r = run(bpy.ops.nproto.shell, faces=faces((B, (1, 1, 2))), thickness=1.5)
    check(f"[{tag}] wall thicker than the part: refused, body kept", r == {'CANCELLED'}
          and B[store.PROP_BREP] == text and not features.stack(B), str(r))

    # an open sheet is refused
    clear()
    R = poly("S", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    bpy.ops.nproto.planar(inputs=ins((R, "S0")))
    S = ctx.active_object
    r = run(bpy.ops.nproto.shell, faces=faces((S, (0.5, 0.5, 0))), thickness=0.1)
    check(f"[{tag}] an open sheet is refused", r == {'CANCELLED'}, str(r))

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
