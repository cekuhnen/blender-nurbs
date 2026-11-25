"""A second fillet on a revolved puck after a first one (1/8" on the two outer
corners, then the bottom edge). The coiled-face repair compared the kernel's
default area() -- ~0.1 % off on a curved face -- with a precise one, swapped
the sound bottom disc for a re-trim, and the kernel's next fillet crashed:
    blender -b --factory-startup --python tests/test_fillet_after_fillet.py"""
import json, os, sys
import bpy

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
R, H, Z0, IN = 0.2032, 0.0254, 0.4826, 0.003175      # an 8" x 1" puck, its lower rim bevelled

def line(pts):
    cu = bpy.data.curves.new("L", 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1.0)
    sp.order_u = 2; sp.use_endpoint_u = True
    o = bpy.data.objects.new("L", cu); ctx.collection.objects.link(o)
    return o

L = line([(0, 0, Z0 + H), (-R, 0, Z0 + H), (-R, 0, Z0 + H / 2), (-0.1778, 0, Z0), (0, 0, Z0)])
bpy.ops.nproto.revolve(inputs=json.dumps([[L.name, "S0"]]), angle=360.0, axis="Z", pivot="WORLD", body="NEW")
V = ctx.active_object

def edge_at(rad, z):
    for i, e in enumerate(store.get_brep(V).edges()):
        t0, t1 = e.parameter_range()
        p = e.point_at(0.5 * (t0 + t1))
        if abs((p[0] ** 2 + p[1] ** 2) ** 0.5 - rad) < 1e-4 and abs(p[2] - z) < 1e-4 and len(e.faces()) == 2:
            return i

def fillet(idxs, r):
    try:
        return bpy.ops.nproto.edge_fillet(inputs=json.dumps([[V.name, f"E{k}"] for k in idxs]), radius=r)
    except RuntimeError as exc:
        return str(exc).strip().splitlines()[-1]

r1 = fillet([edge_at(R, Z0 + H), edge_at(R, Z0 + H / 2)], IN)
b = store.get_brep(V)
check("first fillet: the two outer corners", r1 == {'FINISHED'} and b.face_count() == 6 and b.is_manifold_solid(),
      f"{r1} faces {b.face_count()}")
for r in (IN, 2 * IN):
    base = V[store.PROP_BREP]
    r2 = fillet([edge_at(0.1778, Z0)], r)
    b = store.get_brep(V)
    check(f"second fillet: the bottom edge at {r * 1000:.3f} mm", r2 == {'FINISHED'} and b.face_count() == 7
          and b.is_manifold_solid(), f"{r2} faces {b.face_count()}")
    store.set_geometry_text(V, base, ctx)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
