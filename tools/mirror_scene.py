"""Builds a scene to try the Mirror tool in (Mirror Copy and Make Symmetric), headless:
    blender -b --factory-startup --python tools/mirror_scene.py -- "<out.blend>"

One mirror plane, y = 0, drawn as a wire outline. Left: Mirror Copy stations
(a surface, a solid box, a trimmed disk, a curve), right: Make Symmetric
stations (NURBS curve, Bezier, closed curve, surface). A 'try' object is left
for you to run the tool on; a 'done' one has the result made, to edit and
check. Every try object's default plane is y = 0. Prints a REPORT line of checks."""
import os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(ROOT, "Blend Files",
                                                                               "Mirror tool test.blend")
import blender_nurbs; blender_nurbs.register()
from blender_nurbs import geom, pattern, store, symmetry
from blender_nurbs.kernel import load
ext = load().blender_ext
ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.nproto_settings.construction_history = True
YZ = dict(center_point=(0, 0, 0), frame_z=(0, 1, 0), frame_x=(1, 0, 0))     # the plane y = 0
def R(p):
    """The stations are written along -Y with the plane x = 0; this turns them 90 deg about Z
    so they run along X with the plane y = 0 (a wide strip for a wide viewport)."""
    return (-p[1], p[0], p[2])
report = {}


def select(*objs):
    if ctx.object is not None and ctx.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in ctx.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0]


def curve(name, pts, cyclic=False, kind='NURBS', handles=None):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new(kind)
    if kind == 'BEZIER':
        sp.bezier_points.add(len(pts) - 1)
        for b, c, (hl, hr) in zip(sp.bezier_points, pts, handles):
            b.co = R(c)
            b.handle_left_type = b.handle_right_type = 'FREE'
            b.handle_left, b.handle_right = R(hl), R(hr)
    else:
        sp.points.add(len(pts) - 1)
        for p, c in zip(sp.points, pts):
            p.co = (*R(c), 1.0)
        sp.order_u = 4; sp.use_endpoint_u = not cyclic
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    return o


def surface(name, G):
    nu, nv = G.shape[:2]
    G = np.array([[R(p) for p in row] for row in G], float)
    s = ext.surface_from_cvs(3, 3, [tuple(p) for p in G.reshape(-1, 3)], geom.clamped_uniform_knots(nu, 3),
                             geom.clamped_uniform_knots(nv, 3), None)
    return store.new_object(ctx, name, s)


def label(text, at, size=0.27):
    cu = bpy.data.curves.new("Label", 'FONT'); cu.body = text; cu.size = size; cu.align_x = 'CENTER'
    o = bpy.data.objects.new("Label: " + text.split("\n")[0], cu); o.location = R(at)
    o.hide_select = True
    ctx.collection.objects.link(o)


def mirror_copy(src, name):
    select(src)
    r = bpy.ops.nproto.mirror(mirror_output='COPY', **YZ)
    assert r == {'FINISHED'}, (src.name, r)
    cp = [o for o in bpy.data.objects if o.get("nproto_in_0") is src][0]
    cp.name = name
    return cp


def make_symmetric(o, **kw):
    select(o)
    r = bpy.ops.nproto.mirror(mirror_output='SYMMETRIC', **YZ, **kw)
    assert r == {'FINISHED'}, (o.name, r)
    assert symmetry.sym_of(o) is not None, o.name


def fender(y0):
    """A 4 x 4 patch from the plane outwards: one edge on x = 0, a bulge, a dip at the far edge."""
    G = np.array([[(x, y0 + y, 0.55 * np.cos(0.6 * y) * np.sin(0.9 * x + 0.5) + 0.12 * x * y)
                   for y in np.linspace(-1.5, 1.5, 5)] for x in np.linspace(0.0, 3.0, 5)], float)
    return G


LX = 4.4       # labels above the stations (the copies land below, y < 0)

# ---------------- Mirror Copy (y > 0) ----------------
label("MIRROR COPY", (6.2, 12.5, 0), 0.6)
f1 = surface("Fender (try Mirror Copy)", fender(20.0))
label("Try: Mirror (Pattern group),\nOutput Copy, plane y = 0, Enter.\nThe seam meets, normals agree", (LX, 20.6, 0))

f2 = surface("Fender", fender(15.0))
c2 = mirror_copy(f2, "Fender (mirror copy)")
label("Done: a live copy. Move the\nsource or its CVs: the copy\nfollows. Edit Mirror: offset", (LX, 15.6, 0))

bpy.ops.nproto.add_primitive(kind="BOX")
bx = ctx.active_object; bx.name = "Box"
bx.location = R((1.4, 10.0, 0.5)); bx.rotation_euler = (0.2, 0.15, 0.5 + np.pi / 2)
ctx.view_layer.update()
c3 = mirror_copy(bx, "Box (mirror copy)")
label("Done: a solid. Face Orientation:\nboth blue = outward,\nno negative scale", (LX, 10.6, 0))

bpy.ops.nproto.add_primitive(kind="DISK")
dk = ctx.active_object; dk.name = "Disk"
dk.location = R((1.6, 5.5, 0.3)); dk.rotation_euler = (0.5, 0.3, np.pi / 2)
ctx.view_layer.update()
c4 = mirror_copy(dk, "Disk (mirror copy)")
label("Done: a trimmed sheet,\nturned over to face\nthe way its source does", (LX, 5.9, 0))

hc = curve("Half Profile (try Mirror Copy)", [(0.0, 1.0, 0.4), (0.0, 1.9, 1.3), (1.2, 2.4, 1.8), (2.4, 2.1, 1.0), (2.9, 1.2, 0.4)])
label("Try: a curve half. Mirror Copy,\ndrag the arrow: the plane\nmoves (offset)", (LX, 1.8, 0))

# ---------------- Make Symmetric (y < 0) ----------------
label("MAKE SYMMETRIC", (6.2, -16.0, 0), 0.6)
lop = [(-2.6, -4.6, 0.0), (-1.9, -3.4, 0.8), (-0.6, -3.0, 1.2), (0.4, -3.6, 1.0), (1.4, -2.9, 0.4), (2.2, -3.9, 0.1), (3.1, -4.8, 0.0)]
curve("Lopsided Curve (try Make Symmetric)", lop)
label("Try: Mirror, S = Make Symmetric.\nThe first half is kept, F flips.\nThen Tab and move a CV", (LX, -3.4, 0))

sc = curve("Symmetric Curve", [(x, y - 5.0, z) for x, y, z in lop])
make_symmetric(sc)
label("Done: Tab, G a CV: its twin\n(blue dot) follows, the middle\nstays on the plane. Ctrl+Z ok", (LX, -8.4, 0))

bz_pts = [(-2.4, -13.6, 0.0), (-0.9, -12.6, 0.6), (0.7, -12.9, 0.4), (2.3, -13.9, 0.1)]
bz_h = [((-2.8, -13.0, 0.0), (-2.0, -14.2, 0.0)), ((-1.5, -12.3, 0.7), (-0.4, -12.8, 0.5)),
        ((0.3, -12.5, 0.5), (1.2, -13.4, 0.2)), ((2.0, -14.4, 0.2), (2.6, -13.2, 0.0))]
bz = curve("Symmetric Bezier", bz_pts, kind='BEZIER', handles=bz_h)
make_symmetric(bz)
label("Done: Bezier. A left handle\nmirrors its twin's right one", (LX, -13.4, 0))

ring = [(np.cos(a) * 2.2 + 0.4 * np.sin(3 * a), np.sin(a) * 1.4 - 18.0, 0.3 * np.cos(2 * a))
        for a in np.linspace(0, 2 * np.pi, 9)[:-1] + 0.35]
rg = curve("Symmetric Ring", ring, cyclic=True)
make_symmetric(rg)
label("Done: closed curve. Pairs\nwrap round the start", (LX, -18.0, 0))

rng = np.random.default_rng(7)
G = np.array([[(-2.5 + iu + 0.25, -25.0 + 0.8 * iv, 0.8 + 0.35 * np.sin(iu + 0.7 * iv) + 0.12 * rng.normal())
               for iv in range(5)] for iu in range(6)], float)
surface("Lopsided Surface (try Make Symmetric)", G)
label("Try: a surface. Make Symmetric\npairs CV rows across the plane", (LX, -23.6, 0))

G2 = G.copy(); G2[:, :, 1] -= 5.5
ss = surface("Symmetric Surface", G2)
make_symmetric(ss)
label("Done: Edit CVs (cage), move\na CV: its twin follows", (LX, -29.1, 0))

label("MIRROR PLANE  y = 0", (-0.25, -34.5, 0), 0.4)

# the plane as a wire reference (not selectable)
me = bpy.data.meshes.new("Mirror Plane y=0")
me.from_pydata([R(p) for p in ((0, -33, -0.6), (0, 24, -0.6), (0, 24, 2.0), (0, -33, 2.0))], [], [(0, 1, 2, 3)])
pl = bpy.data.objects.new("Mirror Plane y=0", me); ctx.collection.objects.link(pl)
pl.display_type = 'WIRE'; pl.hide_select = True; pl.hide_render = True

# ---------------- checks ----------------
ctx.view_layer.update()
def W(o): return np.array(geom.world_matrix(o), float)
for src, cp in ((f2, c2), (bx, c3), (dk, c4)):
    report[cp.name] = {"det": round(float(np.linalg.det(W(cp)[:3, :3])), 6),
                       "history": (pattern.pattern_of(cp) or [None, {}])[1].get("mode")}
for o in (sc, rg):
    P = np.array([tuple(q.co[:3]) for q in o.data.splines[0].points])
    s = symmetry.sym_of(o)
    tw = symmetry.chain_twins(len(P), o.data.splines[0].use_cyclic_u, (s.get("shift") or {}).get("0", 0))
    report[o.name] = float(np.abs(P[tw] - symmetry.reflect(s, P)).max())
report[ss.name] = symmetry.sym_of(ss).get("dir")
report["objects"] = len(bpy.data.objects)
class _Stub: pass
for name, outp in (("Fender (try Mirror Copy)", "COPY"), ("Half Profile (try Mirror Copy)", "COPY"),
                   ("Lopsided Curve (try Make Symmetric)", "SYMMETRIC"), ("Lopsided Surface (try Make Symmetric)", "SYMMETRIC")):
    st = _Stub(); st.mirror_output = outp
    pattern._PatternTool._mirror_widget(st, None, [bpy.data.objects[name]])
    report["default plane " + name] = st.frame_z

# view: top, ortho, the whole strip
for scr in bpy.data.screens:
    for area in scr.areas:
        if area.type == 'VIEW_3D':
            r3 = area.spaces.active.region_3d
            r3.view_rotation = (1.0, 0.0, 0.0, 0.0)
            r3.view_perspective = 'ORTHO'
            r3.view_location = (4.5, 1.6, 0.0)
            r3.view_distance = 45.0
select(f1)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("REPORT", report)
print("SAVED", out)
