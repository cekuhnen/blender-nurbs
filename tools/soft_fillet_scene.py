"""Builds a scene to try the Soft fillet in (Surface Fillet and Edge Fillet,
Shape Soft), headless:
    blender -b --factory-startup --python tools/soft_fillet_scene.py -- "<out.blend>"

Back row (y = 0): inputs to try the tools on. Front row (y = -9): the same
parts already filleted (history on: open the Feature box to change Radius,
Smoothing, G2 / G3), with a circular box beside the soft ones to compare."""
import json, math, os, sys
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import store

out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(ROOT, "Blend Files",
                                                                               "Soft Fillet Test.blend")
ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.nproto_settings.construction_history = True
ctx.scene.unit_settings.system = 'METRIC'

coll_try = bpy.data.collections.new("Try it")
coll_done = bpy.data.collections.new("Filleted")
coll_lbl = bpy.data.collections.new("Labels")
coll_in = bpy.data.collections.new("Profiles")
for c in (coll_try, coll_done, coll_lbl, coll_in):
    ctx.scene.collection.children.link(c)


def move_to(o, coll):
    for c in list(o.users_collection):
        c.objects.unlink(o)
    coll.objects.link(o)


def poly(name, pts, cyclic=False):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    return o


def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])


def label(text, loc, size=0.45):
    cu = bpy.data.curves.new("lbl_" + text[:20], 'FONT')
    cu.body = text
    cu.size = size
    cu.align_x = 'LEFT'
    o = bpy.data.objects.new("Label " + text[:24], cu)
    o.location = loc
    coll_lbl.objects.link(o)
    o.hide_select = True
    return o


def patch(name, p0, p1, d, dist):
    ln = poly(name + " line", [p0, p1])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=dist, direction=d)
    o = ctx.active_object; o.name = name
    move_to(ln, coll_in)
    return o


def floor_wall(tag, x, y):
    f = patch(tag + " Floor", (x, y - 2, 0), (x, y + 2, 0), "X", 4.0)       # z = 0, x .. x + 4
    w = patch(tag + " Wall", (x + 2, y - 2, -1), (x + 2, y + 2, -1), "Z", 3.0)  # x = x + 2
    return f, w


def l_sheet(tag, x, y):
    a = patch(tag + " a", (x, y - 2, 0), (x, y + 2, 0), "X", 2.0)
    b = patch(tag + " b", (x + 2, y - 2, 0), (x + 2, y + 2, 0), "Z", 2.0)
    bpy.ops.object.select_all(action='DESELECT'); a.select_set(True); b.select_set(True)
    ctx.view_layer.objects.active = a
    bpy.ops.nproto.join()
    o = ctx.active_object; o.name = tag
    return o


def box(tag, x, y, w=4.0, d=3.0, h=2.0):
    r = poly(tag + " profile", [(x, y, 0), (x + w, y, 0), (x + w, y + d, 0), (x, y + d, 0)], cyclic=True)
    bpy.ops.nproto.extrude(inputs=ins((r, "S0")), distance=h, direction="Z", body="NEW")
    o = ctx.active_object; o.name = tag
    move_to(r, coll_in)
    return o


def puck(tag, x, y, r=1.5, h=1.0):
    bpy.ops.curve.primitive_nurbs_circle_add(radius=r, location=(x, y, 0))
    c = ctx.active_object; c.name = tag + " profile"
    bpy.ops.nproto.extrude(inputs=ins((c, "S0")), distance=h, direction="Z", body="NEW")
    o = ctx.active_object; o.name = tag
    move_to(c, coll_in)
    return o


def fillet_all(o, **kw):
    return bpy.ops.nproto.edge_fillet(inputs=json.dumps([[o.name, f"E{i}"] for i in store.shared_edges(o)]), **kw)


made = []

# --- back row: try the tools on these -------------------------------------
f, w = floor_wall("Try SF", 0, 0)
label("Surface Fillet\nselect Floor + Wall", (0, -3.4, 0))
lt = l_sheet("Try EF L", 8, 0)
label("Edge Fillet\nthe shared edge", (8, -3.4, 0))
bt = box("Try EF Box", 15, -1.5)
label("Edge Fillet\nAlt+click: all edges", (15, -3.4, 0))
pt = puck("Try EF Puck", 24.5, 0)
label("Edge Fillet\nthe two rims", (23, -3.4, 0))
for o in (f, w, lt, bt, pt):
    move_to(o, coll_try)
    made.append(o.name)

# --- front row: already filleted ------------------------------------------
Y = -9.0
f2, w2 = floor_wall("Done SF", 0, Y)
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.fillet(surf_a=f2.name, surf_b=w2.name, radius=0.3, shape="SOFT", continuity="2",
                          smoothing=0.6, trim_bases=True)
sf_obj = ctx.active_object
sf_obj.name = "Done SF Soft G2 R0.3"
for o in (f2, w2, sf_obj):
    move_to(o, coll_done)
label("Surface Fillet Soft G2\nR 0.3, Smoothing 0.6", (0, Y - 3.4, 0))
print("surface fillet:", r)

l2 = l_sheet("Done EF L Soft G3", 8, Y)
r = fillet_all(l2, shape="SOFT", radius=0.3, smoothing=0.7, continuity="3")
move_to(l2, coll_done)
label("Edge Fillet Soft G3\nR 0.3, Smoothing 0.7", (8, Y - 3.4, 0))
print("L:", r)

for k, (name, kw, txt) in enumerate((
        ("Done Box Circular R0.25", dict(shape="CIRCULAR", radius=0.25), "Circular R 0.25\n(to compare)"),
        ("Done Box Soft G2 R0.25", dict(shape="SOFT", radius=0.25, smoothing=0.6, continuity="2"),
         "Soft G2 R 0.25\nSmoothing 0.6"),
        ("Done Box Soft G3 R0.25", dict(shape="SOFT", radius=0.25, smoothing=0.8, continuity="3"),
         "Soft G3 R 0.25\nSmoothing 0.8"))):
    x = 15 + 6.5 * k
    b = box(name, x, Y - 1.5)
    r = fillet_all(b, **kw)
    move_to(b, coll_done)
    label(txt, (x, Y - 3.4, 0))
    print(name, r, store.get_brep(b).face_count(), "faces, solid", store.get_brep(b).is_manifold_solid())

p2 = puck("Done Puck Soft G2 R0.2", 36.5, Y)
r = fillet_all(p2, shape="SOFT", radius=0.2, smoothing=0.6, continuity="2")
move_to(p2, coll_done)
label("Puck Soft G2\nR 0.2", (35, Y - 3.4, 0))
print("puck:", r)

coll_in.hide_viewport = True
coll_in.hide_render = True

# a camera-free view: frame everything from the front, slightly above
for scr in bpy.data.screens:
    for area in scr.areas:
        if area.type == 'VIEW_3D':
            for sp in area.spaces:
                if sp.type == 'VIEW_3D':
                    r3 = sp.region_3d
                    r3.view_location = (19.0, -4.5, 0.5)
                    r3.view_distance = 38.0
                    r3.view_rotation = (0.82, 0.45, 0.17, 0.31)
                    sp.shading.type = 'SOLID'
bpy.ops.object.select_all(action='DESELECT')
os.makedirs(os.path.dirname(out), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("saved", out)
