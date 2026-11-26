"""Builds a scene to try Remove Edges' Fit Tolerance in, headless:
    blender -b --factory-startup --python tools/fit_tolerance_scene.py -- "<out.blend>"

Back row (y = 0): bodies to try it on. Edge mode, pick the edge(s) named on
the label, X: the edge is kept (the faces are tangent but on different
surfaces) and the report names the Fit Tolerance; F9, set Fit Tolerance (e.g.
1 mm) and the faces become one refitted face. Remove Redundant Edges with a
Fit Tolerance refits only curvature-continuous joints (the two-patch top, the
grid), never a fillet's or the slot's G1 edges.

Front row (y = -6): the same bodies already done (history on: open the
Feature box and set Fit Tolerance to 0 to get the edges back). The bodies are
the ones tests/test_fit_tolerance.py checks."""
import math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import store, subsel
from blender_nurbs.kernel import load
from blender_nurbs.kernel import mergefaces as mf
from blender_nurbs.kernel.classa import basis

out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(ROOT, "Blend Files",
                                                                               "Fit Tolerance Test.blend")
sm = load()
ext = sm.blender_ext
ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.nproto_settings.construction_history = True
ctx.scene.unit_settings.system = 'METRIC'

# the test's body builders (one source for the scene and the checks)
src = open(os.path.join(ROOT, "tests", "test_fit_tolerance.py")).read()
ns = {"sm": sm, "ext": ext, "np": np, "math": math, "mf": mf, "basis": basis}
exec(src[src.index("def bump("):src.index("# --- kernel")], ns)
patch, joined, topped_block, ring, bump = ns["patch"], ns["joined"], ns["topped_block"], ns["ring"], ns["bump"]

coll_try = bpy.data.collections.new("Try it")
coll_done = bpy.data.collections.new("Done (history)")
coll_lbl = bpy.data.collections.new("Labels")
for c in (coll_try, coll_done, coll_lbl):
    ctx.scene.collection.children.link(c)
TMP = os.path.join(os.path.dirname(os.path.abspath(out)), ".fit_scene_tmp.sm")
TOL = 0.001


def move_to(o, coll):
    for c in list(o.users_collection):
        c.objects.unlink(o)
    coll.objects.link(o)


def body(name, b, at, coll):
    from mathutils import Matrix
    b.write_to_file(TMP, ascii=True)
    with open(TMP) as f:
        o = store.new_object_text(ctx, name, f.read(), Matrix.Translation(at))
    move_to(o, coll)
    return o


def label(text, loc, size=0.27):
    cu = bpy.data.curves.new("lbl_" + text[:20], 'FONT')
    cu.body = text
    cu.size = size
    o = bpy.data.objects.new("Label " + text.split("\n")[0][:28], cu)
    o.location = loc
    coll_lbl.objects.link(o)
    o.hide_select = True
    return o


def pick(o, pts):
    polys = subsel._edge_polys(o)
    subsel.set_picks(o, "EDGE", [int(np.argmin([subsel._dist_poly(P, np.array(p, float)) if P is not None else np.inf
                                                 for P in polys])) for p in pts])


def done(o, pts=None, **kw):
    """Run Remove Edges on o (the picked edges at pts, or all) with history."""
    for x in ctx.selected_objects:
        x.select_set(False)
    o.select_set(True)
    ctx.view_layer.objects.active = o
    if pts is not None:
        pick(o, pts)
    r = bpy.ops.nproto.remove_edges(all_edges=pts is None, **kw)
    print(f"  {o.name}: {r} -> {store.get_brep(o).face_count()} faces")


# --- the bodies --------------------------------------------------------------------------
A = patch(0, 1, 0, 1, 6, 6)
Bw = patch(1, 2, 0, 1, 7, 8, warp=lambda t: t + 0.2 * t * (1 - t))
MID = [1.0, 0.5, float(bump(1.0, 0.5))]
Q = [patch(0, 1, 0, 0.5, 5, 5), patch(1, 2, 0, 0.5, 6, 6, lambda t: t + 0.15 * t * (1 - t)),
     patch(0, 1, 0.5, 1, 6, 6, lambda t: t - 0.1 * t * (1 - t)), patch(1, 2, 0.5, 1, 5, 5)]
GRID_PTS = [[1.0, 0.25, float(bump(1, 0.25))], [1.0, 0.75, float(bump(1, 0.75))],
            [0.5, 0.5, float(bump(0.5, 0.5))], [1.5, 0.5, float(bump(1.5, 0.5))]]


def slot():
    return sm.boolean_union(sm.create_box((0, -0.5, 0), 1, 1, 0.4), sm.create_cylinder((1, 0, 0), 0.5, 0.4))


def crease():
    Pc, dc = ns["cvs"](patch(1, 2, 0, 1, 6, 6))
    Pc[..., 2] += 0.035 * (Pc[..., 0] - 1)                    # tilted 2°: a crease at x = 1
    B3 = ext.surface_from_cvs(3, 3, [tuple(map(float, c)) for c in Pc.reshape(-1, 3)], dc["knots_u"],
                              dc["knots_v"], None)
    return joined([patch(0, 1, 0, 1, 6, 6), B3])


def filleted():
    b = sm.create_box((0, 0, 0), 2, 1, 0.6)
    e = min(b.edges(), key=lambda g: g.closest_point((1, 0, 0.6))[2])
    sm.fillet_edges(b, [e], radius=0.2)
    return b


CASES = [
    # name, builder, try-label, picks for the done copy (None: Remove Redundant Edges), done label
    ("Two-Patch Top", lambda: topped_block(A, Bw),
     "Two-patch top (block)\nsmooth patches, other knots,\nparameterized differently\npick the top seam (x = 1), X\nthen F9: Fit Tolerance 1 mm",
     [MID], "one top face"),
    ("Patch Grid", lambda: joined(Q),
     "2 x 2 patch grid (sheet)\neach patch its own way\npick the 4 inner edges, X\nthen F9: Fit Tolerance 1 mm",
     GRID_PTS, "one face"),
    ("Slot", slot,
     "Slot (solid)\nflat side runs into the\nround end (tangent, G1)\npick the side edges at\nthe round end, X, F9",
     [(1, 0.5, 0.2), (1, -0.5, 0.2)], "sides + round end\nrefitted (2 edges)"),
    ("Flared Ring", lambda: joined([ring(0, 1, [1, 1, 1, 1]), ring(1, 2, [1, 1, 1.05, 1.1, 1.12], rot=2)]),
     "Flared ring (open tube)\ncylinder + tangent flare,\nseams a quarter turn apart\npick the middle circle, X, F9",
     [(0.0, 1.0, 1.0)], "one closed face"),
    ("Crease 2 deg", crease,
     "2° crease (sheet)\npick the middle edge, X:\nkept at any Fit Tolerance\n(over 0.5° is a crease)",
     None, None),
    ("Filleted Box", filleted,
     "Filleted box\nRemove Redundant Edges +\nFit Tolerance: fillet kept\n(G1 joints only when picked)",
     None, "Remove Redundant Edges\n+ Fit Tolerance:\nnothing to do (fillet kept)"),
]

x = 0.0
for name, make, try_text, picks, done_text in CASES:
    at_try, at_done = (x, 0.0, 0.0), (x, -6.0, 0.0)
    body("Try " + name, make(), at_try, coll_try)
    label(try_text, (x, -1.6, 0))
    if done_text is not None:
        o = body("Done " + name, make(), at_done, coll_done)
        done(o, picks, fit_tolerance=TOL)
        label(done_text + "\n(Feature box: Fit Tolerance)", (x, -7.6, 0))
    x += 3.5 if name not in ("Flared Ring",) else 4.0

label("Remove Edges · Fit Tolerance\nTry it: Edge mode, pick, X, F9    Done: history on", (0, 2.6, 0), size=0.35)
for o in ctx.selected_objects:
    o.select_set(False)
if os.path.exists(TMP):
    os.remove(TMP)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("saved", out)
