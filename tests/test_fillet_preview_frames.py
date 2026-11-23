"""Edge Fillet preview frames on a part whose edges end at corners (a block
with a U-shaped hole and a notch, from a boolean): at an edge's ends the
step that tells which side of the edge a face lies on leaves the trims both
ways, and the end sections came out flipped, sticking out of the preview as
little flags. Every edge's directions into its faces must run continuously:
    blender -b --factory-startup --python tests/test_fillet_preview_frames.py"""
import gzip, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import store
from blender_nurbs.ops_trim import fillet_frames, fillet_sections

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.nproto_settings.construction_history = False

with gzip.open(os.path.join(ROOT, "tests", "data", "fillet_preview_corners.sm.gz"), "rt") as f:
    o = store.new_object_text(ctx, "Part", f.read())

shared = sorted(store.shared_edges(o))
check("the part's edges are shared", len(shared) >= 30, len(shared))

flips, reach = [], []
for i in shared:
    fr = fillet_frames(o, i)
    if fr is None:
        continue
    P, D1, D2 = fr
    for D in (D1, D2):
        c = np.sum(D[1:] * D[:-1], axis=1)
        if c.min() < 0.5:
            flips.append((i, round(float(c.min()), 3)))
    cosa = np.sum(D1 * D2, axis=1)
    if cosa.max() < 0.95 and cosa.min() > -0.95:        # a real wedge (not a smooth or a folded edge)
        X = fillet_sections(P, D1, D2, 0.01)
        d = np.linalg.norm(X - P[:, None], axis=2).max(axis=1)
        reach.append((i, round(float(d.max() / np.median(d)), 3)))
check("every edge's directions into its faces run on without a flip (ends included)", not flips, flips)
check("no section reaches far beyond its neighbours (no flags at the ends)",
      reach and max(r for _, r in reach) < 1.5, max(reach, key=lambda x: x[1]) if reach else None)

print("ALL OK" if ok else "SOME FAILED")
if not ok:
    sys.exit(1)
