"""Edge fillet on the corner edges of a thickened, bent plywood shell (the
41.6 mm edges where its top strip meets its side strip). The patches round
those corners stop exactly at the edges' vertices, and the kernel failed
(SM_ERR 1001 / 1007) at one radius and not the next, mirror-image corners
disagreeing. The worker retries with those faces on surfaces continued past
their sides, then puts their own surfaces back:
    blender -b --factory-startup --python tests/test_fillet_continued.py"""
import gzip, json, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.nproto_settings.construction_history = False

with gzip.open(os.path.join(ROOT, "tests", "data", "fillet_ply_corner.sm.gz"), "rt") as f:
    text = f.read()
src = store.new_object_text(ctx, "Ply", text)
b0 = store.get_brep(src)
_, ext = store._k()
inputs = [np.array(ext.surface_cvs(f.surface())["points"], float) for f in b0.faces()]
em = []
for e in b0.edges():
    t0, t1 = e.parameter_range()
    em.append(e.point_at(0.5 * (t0 + t1)))
em = np.array(em, float)
# the two top corners (left, right): edges through the thickness
corners = {"left": [-0.3007, 0.3113, 0.6042], "right": [0.3007, 0.3113, 0.6042]}
pick = {k: int(np.argmin(np.linalg.norm(em - np.array(m), axis=1))) for k, m in corners.items()}
check("the corner edges are found", all(np.linalg.norm(em[pick[k]] - np.array(m)) < 1e-3 for k, m in corners.items()),
      pick)

def on_inputs(b):
    """Faces not on one of the input surfaces (the fillets, and any face
    left on a continued surface)."""
    n = 0
    for f in b.faces():
        P = np.array(ext.surface_cvs(f.surface())["points"], float)
        if not any(K.shape == P.shape and float(np.abs(K - P).max()) < 1e-12 for K in inputs):
            n += 1
    return n

# both corners at 60 mm (the kernel alone: SM_ERR 1001), the left one at 20 mm
# (SM_ERR 1001 alone), the right one at 60 mm (the kernel's own first try)
for names, radius in ((("left", "right"), 0.06), (("left",), 0.02), (("right",), 0.06)):
    o = store.new_object_text(ctx, "Ply_" + "_".join(names), text)
    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True)
    ctx.view_layer.objects.active = o
    try:
        res = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[o.name, f"E{pick[k]}"] for k in names]),
                                         shape="CIRCULAR", radius=radius)
    except RuntimeError as exc:
        res = str(exc)
    b = store.get_brep(o)
    tag = f"{'+'.join(names)} r {radius * 1000:.0f} mm"
    check(f"{tag}: filleted", res == {'FINISHED'} and b.face_count() == b0.face_count() + len(names),
          f"{res} {b.face_count()}")
    check(f"{tag}: a closed solid", bool(b.is_manifold_solid()))
    check(f"{tag}: every other face on its own surface", on_inputs(b) == len(names), on_inputs(b))
    lo, hi = (np.array(v) for v in b.bounding_box())
    lo0, hi0 = (np.array(v) for v in b0.bounding_box())
    check(f"{tag}: the body stays where it was", float(np.abs(hi - hi0).max()) < 1e-6
          and float(np.abs(lo - lo0).max()) < 1e-6)

print("ALL OK" if ok else "SOME FAILED")
if not ok:
    sys.exit(1)
