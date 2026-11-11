"""Where does the time go? Times the add-on's main work on a real scene.

    blender -b --factory-startup scene.blend --python tools/bench.py
    blender -b --factory-startup scene.blend --python tools/bench.py -- --profile Patch
    blender -b --factory-startup scene.blend --python tools/bench.py -- --json out.json

For every NURBS object (and every curve feature): a full rebuild from its
history (builder, kernel jobs, mesh), the display tessellation alone, the
continuity labels, the overlay's per-redraw edge work, and the snap targets.
--profile NAME runs a cProfile of that object's rebuild and prints the top
functions. Nothing is saved: the scene is only read.
"""

import cProfile
import io
import json
import os
import pstats
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import blender_nurbs  # noqa: E402

blender_nurbs.register()
from blender_nurbs import contlabels, cvs, diagnostics, draw, features, snapping, store  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
PROFILE = argv[argv.index("--profile") + 1] if "--profile" in argv else None
JSON_OUT = argv[argv.index("--json") + 1] if "--json" in argv else None
ctx = bpy.context


def timed(fn, *a, **kw):
    t = time.perf_counter()
    try:
        fn(*a, **kw)
        err = None
    except Exception as exc:              # a scene may hold objects that fail: note it, go on
        err = f"{type(exc).__name__}: {exc}"[:80]
    return time.perf_counter() - t, err


def overlay_edges(obj):
    """What _draw does with an object's edges on every redraw."""
    d = store.display(obj)
    draw._lines_from_polylines(d.get("edges", []), obj.matrix_world, None)


def cv_edit_step(obj):
    """One live update of Edit CVs: a CV moved, the surface rebuilt from the cage."""
    _, ext = store._k()
    d = ext.surface_cvs(store.get_brep(obj).faces()[0].surface())
    pts = [list(p) for p in d["points"]]
    pts[len(pts) // 2][2] += 1e-3
    s = ext.surface_from_cvs(d["degree_u"], d["degree_v"], [tuple(p) for p in pts], d["knots_u"], d["knots_v"],
                             d["weights"] or None)
    store.set_geometry(obj, s, ctx)


rows = []
objs = [o for o in bpy.data.objects if store.is_nurbs(o) or (o.type == 'CURVE' and features.history(o))]
print(f"BENCH {bpy.data.filepath}: {len(objs)} objects")
for o in objs:
    r = {"name": o.name, "op": (features.history(o) or {}).get("op") or ("steps" if features.stack(o) else "-")}
    if store.is_nurbs(o):
        b = store.get_brep(o)
        r["faces"] = b.face_count()
        chord, angle = store.tess_settings(ctx)
        r["tess"], _ = timed(store._tessellate, b, chord, angle, store.edge_angle_setting(ctx),
                             store.max_edge_setting(ctx), natural=store.is_natural(o))
        r["verts"] = len(o.data.vertices)
        contlabels._cache.pop(o.session_uid, None)
        r["labels"], _ = timed(contlabels.labels, o)
        r["draw"], _ = timed(overlay_edges, o)
        v = ctx.window_manager.nproto_view
        r["zebra"], _ = timed(diagnostics.zebra_mesh, o, float(v.zebra_angle), float(v.zebra_chord))
        r["curv"], _ = timed(diagnostics.face_curvature_mesh, o, float(v.curv_angle), float(v.curv_chord))
        if b.face_count() == 1 and store.is_natural(o):
            r["cv_edit"], _ = timed(cv_edit_step, o)
    if features.has_history(o):
        r["rebuild"], r["error"] = timed(features.rebuild, o, ctx, True)
    rows.append(r)

snap_t, _ = timed(snapping.collect, [o for o in bpy.data.objects if o.visible_get()])

cols = ("rebuild", "tess", "labels", "zebra", "curv", "cv_edit")
print(f"\n{'object':28s} {'feature':14s} {'faces':>5s} {'verts':>7s} " + " ".join(f"{c:>9s}" for c in cols))
for r in sorted(rows, key=lambda r: -max(r.get(c, 0) or 0 for c in cols)):
    vals = " ".join(f"{1e3 * r[c]:8.0f}ms" if c in r else f"{'-':>9s}" for c in cols)
    print(f"{r['name'][:28]:28s} {r['op'][:14]:14s} {r.get('faces', ''):>5} {r.get('verts', ''):>7} {vals}"
          + (f"   ERROR {r['error']}" if r.get("error") else ""))
tot = {c: sum(r.get(c, 0) or 0 for r in rows) for c in cols}
print(f"{'TOTAL':28s} {'':14s} {'':5s} {'':7s} " + " ".join(f"{1e3 * tot[c]:8.0f}ms" for c in cols))
print(f"snap targets for the whole scene: {1e3 * snap_t:.0f} ms")

if JSON_OUT:
    with open(JSON_OUT, "w") as f:
        json.dump({"file": bpy.data.filepath, "rows": rows, "snap": snap_t}, f, indent=1)

if PROFILE:
    o = bpy.data.objects[PROFILE]
    pr = cProfile.Profile()
    pr.enable()
    features.rebuild(o, ctx, force=True)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print(s.getvalue())
