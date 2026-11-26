"""Remove Edges: edges whose faces lie on one surface taken out, the faces
merged into one, the body closed and its shape unchanged:
    blender -b --factory-startup --python tests/test_remove_edges.py"""
import json, os, sys, tempfile, time
import bpy
import numpy as np
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, removeedges, steps, store, subsel
from blender_nurbs.kernel import load
from blender_nurbs.kernel import mergefaces as mf

sm = load()
ext = sm.blender_ext
ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
TMP = tempfile.mkdtemp(prefix="nproto_re_")
UNION = sm.BooleanOp.UNION

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def counts(b): return (b.face_count(), b.edge_count(), b.vertex_count())
def vol_b(b): return abs(b.volume(1e-6))
def vol(o): return vol_b(store.get_brep(o))
def nfaces(o): return store.get_brep(o).face_count()
def solid(o): return bool(store.get_brep(o).is_manifold_solid())

def body(name, b, matrix=None):
    p = os.path.join(TMP, name + ".sm")
    b.write_to_file(p, ascii=True)
    with open(p) as f:
        return store.new_object_text(ctx, name, f.read(), matrix)

def select(*objs):
    for o in ctx.selected_objects: o.select_set(False)
    for o in objs: o.select_set(True)
    ctx.view_layer.objects.active = objs[-1]

def run(**kw):
    try:
        return bpy.ops.nproto.remove_edges(**kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

def edge_at(b, p): return min(b.edges(), key=lambda e: e.closest_point(p)[2])

def raw_union(a, c):          # the kernel's union without its clean-up: the seams stay
    return sm.boolean_with_options(a, c, operation=UNION, post_process=False)

def imprint(a, c):
    return sm.boolean_with_options(a, c, operation=UNION, imprinting=True)

def tribox():
    """A 2 x 1.5 x 1 box made of 12 flat triangles (a triangulated mesh rebuilt
    face for face): every diagonal is between two faces on one plane."""
    def line(a, b): return ext.curve_from_cvs(1, [tuple(a), tuple(b)], [0, 0, 1, 1], None)
    def tri(pts):
        P = np.array(pts, float); n = np.cross(P[1] - P[0], P[2] - P[0]); n /= np.linalg.norm(n)
        u = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0]); v = np.cross(n, u); c = P.mean(0)
        C = [c - 3 * u - 3 * v, c - 3 * u + 3 * v, c + 3 * u - 3 * v, c + 3 * u + 3 * v]
        s = ext.surface_from_cvs(1, 1, [tuple(map(float, q)) for q in C], [0, 0, 1, 1], [0, 0, 1, 1], None)
        return ext.trim_surface_with_3d_curves_tol(s, [3], [line(pts[i], pts[(i + 1) % 3]) for i in range(3)],
                                                   1e-6, 1e-5)
    def v(x, y, z): return (2 * x, 1.5 * y, z)
    quads = [[v(0,0,0), v(0,1,0), v(1,1,0), v(1,0,0)], [v(0,0,1), v(1,0,1), v(1,1,1), v(0,1,1)],
             [v(0,0,0), v(1,0,0), v(1,0,1), v(0,0,1)], [v(0,1,0), v(0,1,1), v(1,1,1), v(1,1,0)],
             [v(0,0,0), v(0,0,1), v(0,1,1), v(0,1,0)], [v(1,0,0), v(1,1,0), v(1,1,1), v(1,0,1)]]
    out = None
    for q in quads:
        for t in ([q[0], q[1], q[2]], [q[0], q[2], q[3]]):
            f = tri(t)
            if out is None: out = f
            else: ext.append_brep(out, f)
    return sm.stitch_advanced(out, 1e-6, squeeze_small_edges=False, making_manifold_solid=True)[0]

# --- kernel (mergefaces) ------------------------------------------------------------------
check("kernel has what Remove Edges needs", mf.has_kernel(sm))

def merged(name, b, want, picked=None, vol_tol=1e-6):
    v0 = vol_b(b)
    t0 = time.perf_counter()
    out, rep = mf.merge(sm, b, picked=picked)
    dt = (time.perf_counter() - t0) * 1000
    got = None if out is None else counts(out)
    good = out is not None and got == want and out.is_manifold_solid() and abs(vol_b(out) - v0) <= vol_tol * v0
    check(f"{name}: {counts(b)} -> {want}", good, f"got {got} {dt:.0f} ms {rep}")
    return out, rep

tb = tribox()
check("tribox: 12 triangles, closed", counts(tb) == (12, 18, 8) and tb.is_manifold_solid())
f = mf.find(sm, tb)
check("find: the 6 diagonals can go, the 12 box edges can't (creases)",
      sum(x["ok"] for x in f["edges"]) == 6 and {x["why"] for x in f["edges"] if not x["ok"]} == {"a crease of 90°"},
      str({x["why"] for x in f["edges"]}))
merged("triangulated box, all", tb, (6, 12, 8))
merged("triangulated box, the top diagonal picked", tb, (11, 17, 8), picked=[(1, 0.75, 1)])
out, rep = mf.merge(sm, tb, picked=[(1, 0, 1)])
check("a box edge picked: refused as a crease, nothing changes", out is None and rep["refused"]
      and rep["refused"][0]["why"] == "a crease of 90°", str(rep))
out, rep = mf.merge(sm, tb, picked=[(5, 5, 5)])
check("a pick far from every edge: missed", out is None and rep["missed"] == [0], str(rep))

two = raw_union(sm.create_box((0, 0, 0), 1, 1.5, 1), sm.create_box((1, 0, 0), 1, 1.5, 1))
merged("two boxes side by side (seam ring, its 4 vertices go too)", two, (6, 12, 8))
merged("two boxes: the top seam only (its ends still have the side seams)", two, (9, 19, 12),
       picked=[(1, 0.75, 1)])
three = raw_union(raw_union(sm.create_box((0, 0, 0), 1, 1, 1), sm.create_box((1, 0, 0), 1, 1, 1)),
                  sm.create_box((2, 0, 0), 1, 1, 1))
merged("three boxes: one seam ring picked (its vertices go)", three, (10, 20, 12),
       picked=[(1, 0.5, 1), (1, 0, 0.5), (1, 0.5, 0), (1, 1, 0.5)])
step = raw_union(sm.create_box((0, 0, 0), 1, 1.5, 1), sm.create_box((1, 0.3, 0), 1, 0.9, 1))
merged("box + narrower box: only top and bottom are coplanar", step, (10, 24, 16))
merged("a circle imprinted on a face (a closed edge)", imprint(sm.create_box((0, 0, 0), 2, 1.5, 1),
                                                               sm.create_cylinder((1, 0.75, 0.5), 0.3, 1)), (6, 12, 8))
merged("stacked cylinders (two surfaces joined into one)",
       raw_union(sm.create_cylinder((0, 0, 0), 0.5, 1), sm.create_cylinder((0, 0, 1), 0.5, 0.7)), (3, 3, 2))
merged("a sphere imprinted in half (one whole surface again)",
       imprint(sm.create_sphere((0, 0, 0), 1), sm.create_box((0, -2, -2), 2, 4, 4)), (1, 1, 2))
# (the kernel's NURBS form of an analytic cylinder measures ~1e-5 apart: tolerance for that)
merged("a cylinder imprinted by a box", imprint(sm.create_cylinder((0, 0, 0), 0.5, 1),
                                                sm.create_box((0, -1, 0.3), 1, 2, 0.4)), (3, 3, 2), vol_tol=2e-5)
out, rep = mf.merge(sm, sm.create_box((0, 0, 0), 2, 1.5, 1))
check("a plain box: nothing to remove", out is None and not rep["refused"], str(rep))
slot = sm.boolean_union(sm.create_box((0, -0.5, 0), 1, 1, 0.4), sm.create_cylinder((1, 0, 0), 0.5, 0.4))
out, rep = mf.merge(sm, slot, picked=[(1, 0.5, 0.2), (1, -0.5, 0.2)])
check("slot: the flat sides' edges onto the round end (tangent) are kept, and say why",
      out is None and len(rep["refused"]) == 2 and all(r["why"].startswith("a flat face meets a curved one")
                                                       and "Fit Tolerance" in r["why"]
                                                       for r in rep["refused"]), f"{counts(slot)} {rep}")

# --- Flat Tolerance: near-coplanar faces refitted onto one plane (planerefit) --------------
import math
def polyhedron(V, F):
    """A closed body of flat polygons (vertex list, faces as index loops)."""
    def line(a, b): return ext.curve_from_cvs(1, [tuple(map(float, a)), tuple(map(float, b))], [0, 0, 1, 1], None)
    out = None
    for f in F:
        P = np.array([V[i] for i in f], float); c = P.mean(0); n = np.zeros(3)
        for i in range(len(P)): n += np.cross(P[i] - c, P[(i + 1) % len(P)] - c)
        n /= np.linalg.norm(n); u = P[1] - P[0]; u -= (u @ n) * n; u /= np.linalg.norm(u); v = np.cross(n, u)
        C = [c - 4 * u - 4 * v, c - 4 * u + 4 * v, c + 4 * u - 4 * v, c + 4 * u + 4 * v]
        srf = ext.surface_from_cvs(1, 1, [tuple(map(float, q)) for q in C], [0, 0, 1, 1], [0, 0, 1, 1], None)
        g = ext.trim_surface_with_3d_curves_tol(srf, [len(P)], [line(P[i], P[(i + 1) % len(P)]) for i in range(len(P))],
                                                1e-7, 1e-5)
        if out is None: out = g
        else: ext.append_brep(out, g)
    return sm.stitch_advanced(out, 1e-6, squeeze_small_edges=False, making_manifold_solid=True)[0]

def roof(deg=0.3):
    """A 2 x 1.5 x 1 box whose top is two faces meeting at a shallow ridge (deg each)."""
    h = 0.75 * math.tan(math.radians(deg))
    V = [(0, 0, 0), (2, 0, 0), (2, 1.5, 0), (0, 1.5, 0), (0, 0, 1), (2, 0, 1), (2, 1.5, 1), (0, 1.5, 1),
         (0, 0.75, 1 + h), (2, 0.75, 1 + h)]
    F = [[0, 3, 2, 1], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 8, 7, 3], [1, 2, 6, 9, 5], [4, 5, 9, 8], [8, 9, 6, 7]]
    return polyhedron(V, F), h

def raised_tribox(dz=0.004):
    """The triangulated box with its corner (2, 1.5, 1) raised by dz: three quads slightly folded."""
    vi = lambda x, y, z: x * 4 + y * 2 + z
    V = [[2 * x, 1.5 * y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    V[vi(1, 1, 1)][2] += dz
    quads = [[vi(0,0,0), vi(0,1,0), vi(1,1,0), vi(1,0,0)], [vi(0,0,1), vi(1,0,1), vi(1,1,1), vi(0,1,1)],
             [vi(0,0,0), vi(1,0,0), vi(1,0,1), vi(0,0,1)], [vi(0,1,0), vi(0,1,1), vi(1,1,1), vi(1,1,0)],
             [vi(0,0,0), vi(0,0,1), vi(0,1,1), vi(0,1,0)], [vi(1,0,0), vi(1,1,0), vi(1,1,1), vi(1,0,1)]]
    F = []
    for q in quads: F += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return polyhedron(V, F)

rf, h = roof(0.3)
out, rep = mf.merge(sm, rf, flat_deg=0.2)
check("roof (0.3° ridge), Flat Tolerance 0.2°: nothing (above it)", out is None, str(rep))
out, rep = mf.merge(sm, rf, flat_deg=1.0)
zs = sorted({round(float(v.point()[2]), 6) for v in out.vertices()}) if out is not None else None
check("roof, Flat Tolerance 1°: one flat top, the plain box shape, corners moved h/2",
      out is not None and counts(out) == (6, 12, 8) and out.is_manifold_solid()
      and abs(vol_b(out) - vol_b(rf)) < 1e-6 and zs == [0.0, round(1 + h / 2, 6)]
      and abs(rep["moved"] - h / 2) < 1e-6 and rep["refitted"] == 1, f"{out and counts(out)} z {zs} {rep}")
out, rep = mf.merge(sm, rf, picked=[(1, 0.75, 1 + h)], flat_deg=1.0)
check("roof, the ridge picked: the same box (its ends' vertices joined too)",
      out is not None and counts(out) == (6, 12, 8) and rep["vertices"] == 2, f"{out and counts(out)} {rep}")
out, rep = mf.merge(sm, rf, picked=[(1, 0.75, 1 + h)])
check("roof, the ridge picked, no tolerance: kept, says its angle (0.6°) and the Flat Tolerance that merges it",
      out is None and rep["refused"] and "0.6°" in rep["refused"][0]["why"]
      and "Flat Tolerance of 0.6° or more" in rep["refused"][0]["why"], str(rep))
tb4 = raised_tribox(0.004)
out, rep = mf.merge(sm, tb4)
check("raised corner, exact only: the unfolded quads merge, the folded ones stay", out is not None
      and counts(out)[0] == 7, f"{out and counts(out)} {rep}")
out, rep = mf.merge(sm, tb4, flat_deg=1.0)
check("raised corner, Flat Tolerance 1°: a plain 6-face box (three planes meet at the corner)",
      out is not None and counts(out) == (6, 12, 8) and out.is_manifold_solid() and 0 < rep["moved"] < 0.004,
      f"{out and counts(out)} {rep}")
rr, h = roof(0.3)
sm.fillet_edges(rr, [edge_at(rr, (0, 0, 0.5))], radius=0.2)
out, rep = mf.merge(sm, rr, picked=[(1, 0.75, 1 + h)], flat_deg=1.0)
check("a fillet at a corner of the near-flat top: refused (a curved face would have to change)",
      out is None and rep["refused"] and "curved face" in rep["refused"][0]["why"], str(rep))

# --- the operator ------------------------------------------------------------------------
def pick_edges(o, pts):
    """Pick the display edges of o nearest these local points (Edge mode)."""
    polys = subsel._edge_polys(o)
    idx = []
    for p in pts:
        q = np.array(p, float)
        d = [subsel._dist_poly(P, q) if P is not None else np.inf for P in polys]
        idx.append(int(np.argmin(d)))
    subsel.set_picks(o, "EDGE", idx)

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"

    clear()
    B = body("Tri", tribox(), Matrix.Translation((3, -1, 0.5)))
    select(B)
    pick_edges(B, [(1, 0.75, 1), (1, 0, 1)])           # the top diagonal and a box edge
    r = run()
    check(f"[{tag}] two picks: the diagonal goes, the box edge stays picked",
          r == {'FINISHED'} and nfaces(B) == 11 and solid(B) and abs(vol(B) - 3.0) < 1e-7
          and len(subsel.picks(B, "EDGE")) == 1, f"faces {nfaces(B)} picks {subsel.picks(B, 'EDGE')}")
    V = np.array([v.point() for v in store.get_brep(B).vertices()], float)
    check(f"[{tag}] the body is where it was (local space kept)", np.allclose(V.min(0), (0, 0, 0), atol=1e-7)
          and np.allclose(V.max(0), (2, 1.5, 1), atol=1e-7), f"{V.min(0)} {V.max(0)}")
    r = run(all_edges=True)
    check(f"[{tag}] Remove Redundant Edges: the plain box", r == {'FINISHED'} and nfaces(B) == 6 and solid(B)
          and abs(vol(B) - 3.0) < 1e-7, f"faces {nfaces(B)}")
    if hist:
        st = features.stack(B)
        check("history: a picked step then an all step", [e["op"] for e in st] == ["merge_faces", "merge_faces"]
              and steps.LABEL["merge_faces"](st[0]["params"]) == "Remove Edges · 2 picked"
              and steps.LABEL["merge_faces"](st[1]["params"]) == "Remove Edges · all redundant", str(st)[:300])
        B.location.x = 10; ctx.view_layer.update()
        features.rebuild(B, ctx, force=True)
        check("history: moved body rebuilds to the plain box", nfaces(B) == 6 and abs(vol(B) - 3.0) < 1e-7,
              f"faces {nfaces(B)}")
        raw = json.loads(B[features.PROP_STACK]); raw.pop()
        B[features.PROP_STACK] = json.dumps(raw)
        features.rebuild(B, ctx, force=True)
        check("history: without the all step, 11 faces again", nfaces(B) == 11, f"faces {nfaces(B)}")
    else:
        check("no history: no steps recorded", not features.stack(B))

    clear()
    B = body("Raised", raised_tribox(0.004))
    select(B)
    r = run(all_edges=True, flat_tolerance=math.radians(1.0))
    check(f"[{tag}] Remove Redundant Edges, Flat Tolerance 1°: the folded box flat", r == {'FINISHED'}
          and nfaces(B) == 6 and solid(B), f"faces {nfaces(B)}")
    if hist:
        st = features.stack(B)
        check("history: label shows the tolerance", steps.LABEL["merge_faces"](st[-1]["params"])
              == "Remove Edges · all redundant · flat 1°", steps.LABEL["merge_faces"](st[-1]["params"]))
        raw = json.loads(B[features.PROP_STACK]); raw[-1]["params"]["flat_tolerance"] = 0.0
        B[features.PROP_STACK] = json.dumps(raw)
        features.rebuild(B, ctx, force=True)
        check("history: tolerance set to 0, the folded quads stay (exact merges only)", nfaces(B) == 7,
              f"faces {nfaces(B)}")

    clear()
    B = body("Box", sm.create_box((0, 0, 0), 2, 1.5, 1))
    select(B)
    pick_edges(B, [(1, 0, 1)])
    r = run()
    check(f"[{tag}] only a needed edge picked: cancelled, the body unchanged, the pick kept",
          r == {'CANCELLED'} and nfaces(B) == 6 and len(subsel.picks(B, "EDGE")) == 1, f"{r} {nfaces(B)}")
    subsel.set_picks(B, "EDGE", [])
    r = run()
    check(f"[{tag}] nothing picked: cancelled", r == {'CANCELLED'})

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
