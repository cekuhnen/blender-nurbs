"""Split on closed solids: every piece a closed solid, capped where it was cut
(Solid Pieces, on by default):
    blender -b --factory-startup --python tests/test_split_solid.py"""
import json, math, os, sys
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, steps, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def poly(name, pts, cyclic=False):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])
def vol(o): return abs(store.get_brep(o).volume(1e-6))
def solid(o): return bool(store.get_brep(o).is_manifold_solid())

def box(name="Box"):
    R = poly(name + "_R", [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], cyclic=True)
    bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=2.0, direction="Z", body="NEW")
    o = ctx.active_object; o.name = name
    return o

def split(tgt, *cutters, surfaces=(), **kw):
    kw.setdefault("projection", "Z")
    try:
        r = bpy.ops.nproto.split(targets=json.dumps([tgt.name] + [s.name for s in surfaces]),
                                 curves=json.dumps([[c.name, "S0"] for c in cutters]), **kw)
    except RuntimeError as exc:
        print("   ", exc)
        r = {'CANCELLED'}
    return r, sorted(ctx.selected_objects, key=vol)

def all_solid(pieces): return all(solid(o) for o in pieces)
def close(a, b, tol=2e-3): return abs(a - b) <= tol

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"

    # --- a box cut across by a line: two closed boxes
    clear()
    B = box()
    L = poly("L", [(1.5, -1, 1), (1.5, 4, 1)])
    r, P = split(B, L)
    v = [vol(o) for o in P]
    check(f"[{tag}] line across a box: two solids", r == {'FINISHED'} and len(P) == 2 and all_solid(P)
          and close(v[0], 9.0) and close(v[1], 15.0), f"{v}")
    if hist:
        piece = P[0]
        L.data.splines[0].points[0].co = (1.0, -1, 1, 1)       # the cut moves: the piece follows, still closed
        L.data.splines[0].points[1].co = (1.0, 4, 1, 1)
        features.rebuild(piece, ctx, force=True)
        check(f"[{tag}] moved cutter: piece rebuilt closed", solid(piece) and close(vol(piece), 6.0),
              f"{vol(piece):.5f}")

    # --- the cut stops inside the box: extended across by default; with Extend off nothing splits
    clear()
    B = box()
    L = poly("L", [(1.5, -1, 1), (1.5, 1.5, 1)])
    r, P = split(B, L)
    check(f"[{tag}] short line extended: two solids", r == {'FINISHED'} and len(P) == 2 and all_solid(P),
          f"{[round(vol(o), 4) for o in P]}")
    clear()
    B = box()
    L = poly("L", [(1.5, -1, 1), (1.5, 1.5, 1)])
    r, P = split(B, L, extend=False)
    check(f"[{tag}] short line, Extend off: not split", r == {'CANCELLED'})

    # --- a circle punches a plug out
    clear()
    B = box()
    bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(2, 1.5, 1))
    C = ctx.active_object
    r, P = split(B, C)
    plug = math.pi * 0.25 * 2.0
    v = [vol(o) for o in P]
    check(f"[{tag}] circle: plug + box with a hole", r == {'FINISHED'} and len(P) == 2 and all_solid(P)
          and close(v[0], plug) and close(v[1], 24.0 - plug), f"{v}")

    # --- two crossing lines: four pieces; two of them joined into one
    clear()
    B = box()
    L1 = poly("L1", [(1.5, -1, 1), (1.5, 4, 1)])
    L2 = poly("L2", [(-1, 1.0, 1), (5, 1.0, 1)])
    r, P = split(B, L1, L2)
    v = [vol(o) for o in P]
    check(f"[{tag}] crossing lines: four solids", r == {'FINISHED'} and len(P) == 4 and all_solid(P)
          and close(sum(v), 24.0) and close(v[0], 3.0) and close(v[-1], 10.0), f"{v}")
    clear()
    B = box()
    L1 = poly("L1", [(1.5, -1, 1), (1.5, 4, 1)])
    L2 = poly("L2", [(-1, 1.0, 1), (5, 1.0, 1)])
    p = {"surfaces_cut": True, "projection": "Z", "view_dir": [0, 0, -1], "extend": True, "reach": "BOTH",
         "through": False, "solids": True}
    info = steps.split_info([(B, "SURF"), (L1, "S0"), (L2, "S0")], p)
    # join the two pieces with y < 1 (3 + 5 = 8): the cap between them goes
    marks = [[pt, pt[1] < 1.0, sd] for pt, sd in zip(info["points"], info["sides"])]
    r, P = split(B, L1, L2, marks=json.dumps({B.name: marks}))
    v = [vol(o) for o in P]
    joined = [o for o in P if close(vol(o), 8.0)]
    check(f"[{tag}] two pieces joined: one solid, no inner wall", r == {'FINISHED'} and len(P) == 3
          and all_solid(P) and len(joined) == 1 and store.get_brep(joined[0]).face_count() == 6, f"{v}")

    # --- a box with a hole cut through the hole (the user's case: four face regions, two solids)
    clear()
    B = box("Base")
    bpy.ops.curve.primitive_nurbs_circle_add(radius=0.8, location=(2, 2, 3))
    H = ctx.active_object
    bpy.ops.nproto.extrude(inputs=ins((H, "S0")), distance=-4.0, direction="Z", body="CUT", target=B.name)
    B = ctx.active_object
    hole = math.pi * 0.64 * 2.0
    L = poly("L", [(-1, 2.0, 1), (5, 2.0, 1)])
    r, P = split(B, L)
    v = [vol(o) for o in P]
    check(f"[{tag}] cut through a hole: two solids", r == {'FINISHED'} and len(P) == 2 and all_solid(P)
          and close(v[0], 8.0 - hole / 2) and close(v[1], 16.0 - hole / 2), f"{v}")

    # --- a cutting surface (a sheet through the box)
    clear()
    B = box()
    S = poly("S", [(2.5, -1, -1), (2.5, 4, -1)])
    bpy.ops.nproto.extrude(inputs=ins((S, "S0")), distance=4.0, direction="Z", body="SURFACE")
    W = ctx.active_object
    try:
        r = bpy.ops.nproto.split(targets=json.dumps([B.name]), curves="[]", cutter_surfs=json.dumps([W.name]))
    except RuntimeError as exc:
        print("   ", exc)
        r = {'CANCELLED'}
    P = sorted([o for o in ctx.selected_objects if o != W], key=vol)
    v = [vol(o) for o in P]
    check(f"[{tag}] surface cutter: two solids", r == {'FINISHED'} and len(P) == 2 and all_solid(P)
          and close(v[0], 9.0) and close(v[1], 15.0), f"{v}")

    # --- Solid Pieces off: the faces are only split, as before
    clear()
    B = box()
    L = poly("L", [(1.5, -1, 1), (1.5, 4, 1)])
    r, P = split(B, L, solids=False)
    check(f"[{tag}] Solid Pieces off: open pieces", r == {'FINISHED'} and len(P) == 2
          and not any(solid(o) for o in P))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
