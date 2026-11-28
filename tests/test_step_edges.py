"""Picked Extend / Surface Align / Match Surface steps outline the edge they acted on:
    blender -b --factory-startup --python tests/test_step_edges.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, geom, store, subsel

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
ctx.scene.nproto_settings.construction_history = True
ext = store._k()[1]
def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

def step_edge_hl(o, k):
    """(edge indices, their world polylines) outlined in blue with step k of o picked."""
    ctx.view_layer.objects.active = o
    bpy.ops.nproto.tree_step(name=o.name, root=o.name, index=k)
    es = sorted(subsel.step_edges(o))
    o.nproto_step.focus = False
    M = np.array(o.matrix_world)
    return es, [np.asarray(subsel._edge_polys(o)[i]) @ M[:3, :3].T + M[:3, 3] for i in es]

def grid_surface(name, xs, ys, z):
    """A bicubic surface through the grid xs x ys (u along x, v along y), heights z(x, y)."""
    pts = [(x, y, z(x, y)) for x in xs for y in ys]
    return store.new_object(ctx, name, ext.surface_from_cvs(3, 3, pts, geom.clamped_uniform_knots(len(xs), 3),
                                                            geom.clamped_uniform_knots(len(ys), 3)))

# --- Extend: the grown side --------------------------------------------------------------
bpy.ops.nproto.add_primitive(kind="PLANE", width=3, depth=3, pl_degree_u=3, pl_degree_v=3, spans_u=5, spans_v=5)
E = ctx.active_object
xedge = max(("U0", "U1", "V0", "V1"),
            key=lambda w: np.mean([p[0] for p in geom.boundary_points(store.get_surfaces(E)[0], w, 8)]))
r = bpy.ops.nproto.extend(inputs=ins((E, xedge)), distance=1.0)
xmax = 1.5 + 1.0                                     # the 3-wide plane about the origin, grown by 1
es, P = step_edge_hl(E, len(features.stack(E)) - 1)
check("picked Extend step outlines the grown side (the far x edge)", r == {'FINISHED'} and len(es) == 1
      and np.abs(P[0][:, 0] - xmax).max() < 1e-6 and abs(np.ptp(P[0][:, 1]) - 3.0) < 1e-6, f"{es} xmax {xmax}")
nope = subsel.step_faces(E)
check("... and no face fill for it", nope == [], str(nope))

# --- Extend on a trimmed surface: only the picked boundary piece -----------------------------
# a 4 x 4 plane with a round notch in its +x edge: two boundary pieces there (y -2..-1 and 1..2)
from blender_nurbs import extend as _ext
bpy.ops.nproto.add_primitive(kind="PLANE", width=4, depth=4, pl_degree_u=3, pl_degree_v=3, spans_u=4, spans_v=4)
T = ctx.active_object; ctx.view_layer.update()
xe = max(("U0", "U1", "V0", "V1"),
         key=lambda w: np.mean([p[0] for p in geom.boundary_points(store.get_surfaces(T)[0], w, 8)]))
bpy.ops.curve.primitive_bezier_circle_add(radius=1.0, location=(2, 0, 1)); C = ctx.active_object
r0 = bpy.ops.nproto.trim(target=T.name, cutters=ins((C, "S0")), ref_point=(2, 0, 0), normal=(0, 0, 1),
                         view_dir=(0, 0, -1), action="DELETE")
eb = store.get_brep(T).edges()
mid = lambda i: np.array(eb[i].point_at(sum(eb[i].parameter_range()) / 2))
runs = sorted([i for i, v in _ext.boundary_edges(T).items() if v[0] == xe], key=lambda i: mid(i)[1])
up = f"E{runs[-1]}" if len(runs) == 2 else None
r = bpy.ops.nproto.extend(inputs=ins((T, up)), distance=1.0) if up else None
st = features.stack(T)
es, P = step_edge_hl(T, len(st) - 1) if st and st[-1]["op"] == "extend_surface" else ([], [])
check("picked Extend step on a trimmed surface outlines only the grown piece (x = 3, y 1..2)",
      r0 == {'FINISHED'} and r == {'FINISHED'} and not store.is_natural(T)
      and st[-1]["params"].get("at") is not None and len(es) == 1
      and np.abs(P[0][:, 0] - 3.0).max() < 1e-6 and abs(P[0][:, 1].min() - 1.0) < 1e-6
      and abs(P[0][:, 1].max() - 2.0) < 1e-6,
      f"{r0} {r} runs {runs} at {st[-1]['params'].get('at') if st else None} {es} "
      f"{[(P_[:, 0].min().round(4), P_[:, 0].max().round(4), P_[:, 1].min().round(4), P_[:, 1].max().round(4)) for P_ in P]}")

# --- Surface Align / Match Surface: the matched side -------------------------------------
F = lambda x, y: 0.05 * x**3 - 0.1 * x * y**2 + 0.08 * y**3 + 0.2 * x * y - 0.1 * y**2
bump = lambda x, y: F(x, y) + 0.4 * (x - 2) * (4 - x) * y * (2 - y) + 0.01 * math.sin(3 * y)
for op, kw in (("surface_align", {"cont_U0": "G1"}), ("match_surface", {"cont_U0": "G1"})):
    N = grid_surface(f"{op}_N", np.linspace(0, 2, 4), np.linspace(0, 2, 4), F)        # the neighbour, across x = 2
    S = grid_surface(f"{op}_S", np.linspace(2, 4, 6), np.linspace(0, 2, 6), bump)     # its U0 lies on x = 2
    for o in ctx.selected_objects: o.select_set(False)
    S.select_set(True); ctx.view_layer.objects.active = S
    r = getattr(bpy.ops.nproto, op)(inputs=ins((S, "U0")), **kw)
    st = features.stack(S)
    es, P = step_edge_hl(S, len(st) - 1) if st else ([], [])
    check(f"picked {op} step outlines the matched side (U0, on x = 2)",
          r == {'FINISHED'} and [e["op"] for e in st] == [op] and len(es) == 1
          and np.abs(P[0][:, 0] - 2.0).max() < 1e-6 and abs(np.ptp(P[0][:, 1]) - 2.0) < 1e-6,
          f"{r} {[e['op'] for e in st]} {es}")
    bpy.data.objects.remove(N); bpy.data.objects.remove(S)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
