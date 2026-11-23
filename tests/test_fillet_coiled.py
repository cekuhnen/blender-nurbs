"""Edge fillet whose run-out spills onto a half cylinder: the kernel carries
that face on over a surface coiled one and a half times round, its trim
covering the face more than once, and the face failed to mesh (a hole in the
solid). It must come back re-trimmed on the cylinder, closed and meshable:
    blender -b --factory-startup --python tests/test_fillet_coiled.py"""
import bmesh, gzip, json, os, sys
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

# a box with a half-cylinder bump at the back, a slot down into a hole through
# the bump; the two short edges where the back wall meets the bump's top
with gzip.open(os.path.join(ROOT, "tests", "data", "fillet_coiled_base.sm.gz"), "rt") as f:
    o = store.new_object_text(ctx, "Bump", f.read())
b0 = store.get_brep(o)
mids = np.array([[0.1313, 0.2794, 0.23904], [-0.1313, 0.2794, 0.23904]])
em = []
for e in b0.edges():
    t0, t1 = e.parameter_range()
    em.append(e.point_at(0.5 * (t0 + t1)))
em = np.array(em, float)
picks = [int(np.argmin(np.linalg.norm(em - m, axis=1))) for m in mids]
check("the two edges are found", all(np.linalg.norm(em[i] - m) < 1e-3 for i, m in zip(picks, mids)), picks)

for cont in ("1", "2"):
    o2 = store.new_object_text(ctx, "Bump_G" + cont, store.brep_to_text(b0))
    bpy.ops.object.select_all(action='DESELECT')
    o2.select_set(True); ctx.view_layer.objects.active = o2
    res = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[o2.name, f"E{i}"] for i in picks]), shape="CIRCULAR",
                                     radius=0.2147, continuity=cont)
    b = store.get_brep(o2)
    lo, hi = (np.array(v) for v in b.bounding_box())
    size = float(np.linalg.norm(hi - lo))
    _, fails = b.tessellate(chord_height_tolerance=5e-4 * size, curve_angle_tolerance_deg=2.0,
                            surface_angle_tolerance_deg=5.0, max_edge_length=size / 30.0,
                            allow_partial=True, return_failures=True)
    bm = bmesh.new(); bm.from_mesh(o2.data)
    holes = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()
    check(f"G{cont}: the fillet is made, 3 new faces", res == {'FINISHED'} and b.face_count() == b0.face_count() + 3,
          b.face_count())
    check(f"G{cont}: still a closed solid", bool(b.is_manifold_solid()))
    check(f"G{cont}: every face meshes (no hole)", not fails and holes == 0, f"{fails} {holes} open mesh edges")
    # the bump's cylinder (r 0.1524 about x = 0, y = 0.2794): one face, its
    # area no more than the half cylinder up to the fillet could have
    cyl = [f for f in b.faces() if abs(np.hypot(f.interior_point()[0], f.interior_point()[1] - 0.2794) - 0.1524) < 1e-4
           and f.interior_point()[1] > 0.3]
    check(f"G{cont}: the cylinder face is covered once", len(cyl) == 1 and cyl[0].area() < np.pi * 0.1524 * 0.34,
          [round(f.area(), 4) for f in cyl])
    # the left run-out: the cylinder's sliver meets the back wall at 90 degrees;
    # its corners must shade with their own face's normal, not one averaged
    # across that crease (spikes)
    sm, _ = store._k()
    me, faces = o2.data, b.faces()
    CN = np.array([c.vector[:] for c in me.corner_normals])
    worst = 0.0
    for poly in me.polygons:
        c = np.array(poly.center[:])
        if not (c[0] < -0.10 and 0.2 < c[1] < 0.4 and 0.15 < c[2] < 0.45):
            continue
        f = faces[int(np.argmin([fc.closest_point(tuple(map(float, c)))[1] for fc in faces]))]
        for li, vi in zip(poly.loop_indices, poly.vertices):
            w = sm.surface_closest_point(f.surface(), tuple(map(float, me.vertices[vi].co)))[1][0]
            n = np.array(f.outward_normal_at_uv(*((w.x, w.y) if hasattr(w, "x") else (w[0], w[1]))), float)
            worst = max(worst, float(np.degrees(np.arccos(np.clip(CN[li] @ n / np.linalg.norm(n), -1, 1)))))
    check(f"G{cont}: run-out corners shade with their face's normal", worst < 5.0, f"worst {worst:.1f} deg")

print("ALL OK" if ok else "SOME FAILED")
if not ok:
    sys.exit(1)
