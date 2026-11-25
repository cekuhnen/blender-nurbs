"""Revolve of a profile whose end lies on the axis only to float32 precision
(a rectangle turned by -90 degrees in Blender, cut where it leaves the axis:
its corner 2e-9 off)
makes a pole, not a ring nanometres wide shading as a bright spot:
    blender -b --factory-startup --python tests/test_revolve_pole.py"""
import json, os, sys
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
W, D, Z = 0.2032, 0.0254, 0.508          # 8 x 1 inch, one side on the Z axis

def rect():
    cu = bpy.data.curves.new("Rect", 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(3)
    for p, c in zip(sp.points, [(-W, 0, 0), (0, 0, 0), (0, D, 0), (-W, D, 0)]):
        p.co = (*c, 1.0)
    sp.order_u = 2; sp.use_cyclic_u = True
    o = bpy.data.objects.new("Rect", cu); ctx.collection.objects.link(o)
    o.location = (0, 0, Z); o.rotation_euler = (-1.5707963, 0, 0)       # as Blender stores -90 degrees
    ctx.view_layer.update()
    return o

for body in ("SURFACE", "NEW"):
    for seam in (0.0, 37.0):
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o)
        R = rect()
        r = bpy.ops.nproto.revolve(inputs=json.dumps([[R.name, "S0"]]), angle=360.0, axis="Z",
                                   pivot="WORLD", body=body, seam_angle=seam)
        V = ctx.active_object
        me = V.data
        P = np.array([v.co[:] for v in me.vertices])
        cn = np.zeros(len(me.loops) * 3)
        me.corner_normals.foreach_get("vector", cn)
        cn = cn.reshape(-1, 3)
        lv = np.array([l.vertex_index for l in me.loops])
        rad = np.linalg.norm(P[lv][:, :2], axis=1)
        worst = 0.0
        for z in (Z - D, Z):                 # both flat discs, inside their rims: every corner normal is +-Z
            m = (np.abs(P[lv][:, 2] - z) < 1e-6) & (rad < W - 1e-5)
            if m.any():
                worst = max(worst, float(np.degrees(np.arccos(np.clip(np.abs(cn[m][:, 2]), 0, 1))).max()))
        near = int(((np.linalg.norm(P[:, :2], axis=1) < 1e-3) & (np.linalg.norm(P[:, :2], axis=1) > 1e-9)).sum())
        check(f"[{body} seam {seam:g}] disc normals along the axis", r == {'FINISHED'} and worst < 0.01,
              f"worst {worst:.3f} deg")
        check(f"[{body} seam {seam:g}] no ring of vertices round the axis", near == 0, f"{near} within 1 mm")

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
