"""Revolve of a profile whose end lies on the axis only to float32 precision
(a rectangle turned by -90 degrees in Blender, cut where it leaves the axis:
its corner 2e-9 off)
makes a pole, not a ring nanometres wide shading as a bright spot:
    blender -b --factory-startup --python tests/test_revolve_pole.py"""
import json, math, os, sys
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

# --- a puck: top, side, a 26.6 deg bevel and the bottom, one polyline -------
# as a surface it is one face per stretch (its corners are edges a chamfer can
# take), and the shallow bevel | bottom crease shades sharp on both bodies
PUCK = [(0, 0, 0.508), (0.2032, 0, 0.508), (0.2032, 0, 0.4953), (0.1778, 0, 0.4826), (0, 0, 0.4826)]

def puck_profile():
    cu = bpy.data.curves.new("Puck", 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(PUCK) - 1)
    for p, c in zip(sp.points, PUCK):
        p.co = (*c, 1.0)
    o = bpy.data.objects.new("Puck", cu); ctx.collection.objects.link(o)
    return o

def crease_normals_z(V, r, z):
    me = V.data
    P = np.array([v.co[:] for v in me.vertices])
    cn = np.zeros(len(me.loops) * 3)
    me.corner_normals.foreach_get("vector", cn)
    cn = cn.reshape(-1, 3)
    lv = np.array([l.vertex_index for l in me.loops])
    m = (np.abs(np.linalg.norm(P[lv][:, :2], axis=1) - r) < 1e-5) & (np.abs(P[lv][:, 2] - z) < 1e-5)
    return cn[m][:, 2]

for body in ("SURFACE", "NEW"):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    pr = puck_profile()
    r = bpy.ops.nproto.revolve(inputs=json.dumps([[pr.name, "S0"]]), angle=360.0, axis="Z", pivot="WORLD", body=body)
    V = ctx.active_object
    b = store.get_brep(V)
    check(f"[puck {body}] one face per stretch of the profile, its corners shared edges",
          r == {'FINISHED'} and b.face_count() == 4 and len(store.shared_edges(V)) == 3,
          f"{b.face_count()} faces, shared {store.shared_edges(V)}")
    nz = crease_normals_z(V, 0.1778, 0.4826)
    bevel = -2.0 / np.sqrt(5.0)
    check(f"[puck {body}] the 26.6 deg bevel | bottom crease shades sharp (each side its own normal)",
          len(nz) and np.abs(nz + 1.0).min() < 1e-3 and np.abs(nz - bevel).min() < 1e-3
          and min(np.abs(nz + 1.0).max(), np.abs(nz - bevel).max()) > 0.05,
          f"normal z at the crease {np.unique(np.round(nz, 3))[:6]}")
    es = b.edges()
    k = min(store.shared_edges(V), key=lambda i: abs(es[i].point_at(es[i].parameter_range()[0])[2] - 0.4826))
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[V.name, f"E{k}"]]), shape="CHAMFER", radius=0.003175)
    check(f"[puck {body}] a 0.125 in chamfer on the bottom edge", r == {'FINISHED'}
          and store.get_brep(V).face_count() == 5, f"{store.get_brep(V).face_count()} faces")

# --- the puck's rims rounded, then: a fillet's seam picked for a chamfer is
# refused by name; the round step turned into a chamfer sets each rim back the
# distance, though the two rims meet at different angles ----------------------
from blender_nurbs import features, history_ui
ctx.scene.nproto_settings.construction_history = True
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
pr = puck_profile()
bpy.ops.nproto.revolve(inputs=json.dumps([[pr.name, "S0"]]), angle=360.0, axis="Z", pivot="WORLD", body="NEW")
V = ctx.active_object

def rim_edge(r, z, want_shared=True):
    es = store.get_brep(V).edges()
    def d(i):
        p = es[i].point_at(es[i].parameter_range()[0])
        return abs(math.hypot(p[0], p[1]) - r) + abs(p[2] - z)
    return min(store.shared_edges(V), key=d)

D = 0.003175
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[V.name, f"E{rim_edge(0.2032, 0.508)}"],
                                              [V.name, f"E{rim_edge(0.2032, 0.4953)}"]]),
                           shape="CIRCULAR", radius=D)
es = store.get_brep(V).edges()
seam = rim_edge(0.2032, 0.4953 + D)                  # where the bottom round meets the side: tangent
err = ""
try:
    bpy.ops.nproto.edge_fillet(inputs=json.dumps([[V.name, f"E{seam}"]]), shape="CHAMFER", radius=D)
except RuntimeError as exc:
    err = str(exc)
check("a fillet's tangent seam picked for a chamfer: refused, saying why", "meet tangent" in err, err.strip()[:120])
bottom = rim_edge(0.1778, 0.4826)
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[V.name, f"E{seam}"], [V.name, f"E{bottom}"]]),
                               shape="CHAMFER", radius=D)
check("... picked with a sharp edge: the sharp one is chamfered, the seam left out", r == {'FINISHED'}
      and len(features.stack(V)) == 2, f"{len(features.stack(V))} steps")
features.remove_step(V, 1)
features.rebuild(V, ctx, force=True)
history_ui.sync_step(V, 0)
V.nproto_step.shape = "CHAMFER"
p0 = features.stack(V)[0]["params"]
b = store.get_brep(V)
rz = []
for e in b.edges():
    t0, t1 = e.parameter_range()
    for t in (t0, t1):                  # (the rims' circles start and end on the seam line)
        q = e.point_at(t)
        rz.append((math.hypot(q[0], q[1]), q[2]))
rz = np.array(rz)
top_in = 0.2032 - rz[(np.abs(rz[:, 1] - 0.508) < 1e-6) & (rz[:, 0] > 0.1)][:, 0].min()   # the top's rim pulled in
side = rz[np.abs(rz[:, 0] - 0.2032) < 1e-6][:, 1]
top_down, bot_up = 0.508 - side[side > 0.5].min(), side[side < 0.5].max() - 0.4953
bev = rz[(rz[:, 0] < 0.2032 - 1e-6) & (rz[:, 1] > 0.4826 + 1e-6) & (rz[:, 1] < 0.4953)]
bev_d = float(np.hypot(0.2032 - bev[:, 0], 0.4953 - bev[:, 1]).min()) if len(bev) else 0.0
check("the round step switched to Chamfer: sized by distance, 0.125 in on all four faces of both rims",
      not V.nproto_step.error and p0.get("setback") and b.is_manifold_solid() and b.volume() > 0
      and max(abs(x - D) for x in (top_in, top_down, bot_up, bev_d)) < 2e-5,
      f"top {top_in:.5f} / {top_down:.5f}, side {bot_up:.5f}, bevel {bev_d:.5f}, {V.nproto_step.error}")
ctx.scene.nproto_settings.construction_history = False

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
