"""Selection readout:  blender -b --factory-startup --python tests/test_measure_readout.py"""
import math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, linesolve, measure as ms, measureinfo as mi, store, subsel

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sm = store._k()[0]

def only(*objs):
    for o in ctx.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0] if objs else None
    ctx.view_layer.update()

def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)

# --- a solid box: volume, area, centroid; moved, turned, scaled -----------------------
bpy.ops.nproto.add_primitive(kind="BOX"); B = ctx.active_object
only(B)
info = mi.gather(ctx)
check("box: one solid, volume 8, area 24, centroid at the origin",
      len(info["solids"]) == 1 and rel(info["solids"][0][0], 8) < 1e-9 and rel(info["solids"][0][1], 24) < 1e-9
      and np.linalg.norm(mi.centroid(info)) < 1e-9, str(info["solids"]))
B.location = (1, 2, 3); B.rotation_euler = (0.3, 0.5, 0.7); B.scale = (2, 2, 2); ctx.view_layer.update()
info = mi.gather(ctx)
v, a, c = info["solids"][0]
check("box moved, turned, scaled x2: volume 64, area 96, centroid at its location",
      rel(v, 64) < 1e-6 and rel(a, 96) < 1e-6 and np.linalg.norm(c - [1, 2, 3]) < 1e-5, f"{v} {a} {c}")
B.rotation_euler = (0, 0, 0); B.scale = (1, 2, 3); ctx.view_layer.update()
info = mi.gather(ctx)
v, a, c = info["solids"][0]
check("box scaled (1, 2, 3): volume 48, area 88 (a moved copy)", rel(v, 48) < 1e-6 and rel(a, 88) < 1e-6
      and np.linalg.norm(c - [1, 2, 3]) < 1e-5, f"{v} {a}")
check("the body's own Brep is untouched by that copy", abs(store.get_brep(B).volume(1e-6) - 8) < 1e-6)
B.scale = (1, 1, 1); B.location = (0, 0, 0); ctx.view_layer.update()

# --- cached: no kernel call while nothing changes ---------------------------------------
calls = []
orig = mi.body_reading
mi.body_reading = lambda *a, **k: calls.append(1) or orig(*a, **k)
mi.gather(ctx); mi.gather(ctx)
n0 = len(calls)
B.location = (0.5, 0, 0); ctx.view_layer.update(); mi.gather(ctx)
mi.body_reading = orig
check("cached until the body moves (then measured once)", n0 == 1 and len(calls) == 2, f"{n0} {len(calls)}")
B.location = (0, 0, 0); ctx.view_layer.update()

# --- picked faces and edges ---------------------------------------------------------------
subsel.set_picks(B, "FACE", [0, 3])
info = mi.gather(ctx)
check("two picked faces: their areas (4 each), no body reading",
      len(info["faces"]) == 2 and all(rel(x, 4) < 1e-9 for x in info["faces"]) and not info["solids"], str(info["faces"]))
subsel.set_picks(B, "EDGE", [0, 1, 2])
info = mi.gather(ctx)
check("faces and edges picked: both (three edges of 2)",
      len(info["faces"]) == 2 and len(info["edges"]) == 3 and all(rel(x, 2) < 1e-9 for x in info["edges"]))
rows = mi.rows(ctx, info)
check("rows: total length and total area", any(a == "3 edges" and "Total length" in b for a, b in rows)
      and any(a == "2 faces" and "Total area" in b for a, b in rows), str(rows))
subsel.set_picks(B, "FACE", []); subsel.set_picks(B, "EDGE", [])

# --- sphere, open sheet --------------------------------------------------------------------
bpy.ops.nproto.add_primitive(kind="SPHERE"); S = ctx.active_object
S.location = (5, 0, 0); ctx.view_layer.update()
only(S)
v, a, c = mi.gather(ctx)["solids"][0]
check("sphere r=1: volume 4/3 pi, area 4 pi (1e-5)", rel(v, 4 / 3 * math.pi) < 1e-5 and rel(a, 4 * math.pi) < 1e-5
      and np.linalg.norm(c - [5, 0, 0]) < 1e-5, f"{v} {a}")
bpy.ops.nproto.add_primitive(kind="CYLINDER"); C = ctx.active_object
only(C)
info = mi.gather(ctx)
check("open cylinder (r 1, height 2): an open surface, area 4 pi, no volume",
      not info["solids"] and len(info["sheets"]) == 1 and rel(info["sheets"][0], 4 * math.pi) < 1e-5, str(info["sheets"]))
check("open: the row says no volume and gives no centroid",
      any("no volume" in b for _, b in mi.rows(ctx, info)) and mi.centroid(info) is None)
only(B, S)
info = mi.gather(ctx)
c = mi.centroid(info)
check("box + sphere: totals and the volume-weighted centroid",
      len(info["solids"]) == 2 and np.linalg.norm(c - [5 * (4 / 3 * math.pi) / (8 + 4 / 3 * math.pi), 0, 0]) < 1e-5,
      str(c))

# --- curves --------------------------------------------------------------------------------
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.7); K = ctx.active_object
K.scale = (3, 3, 3); ctx.view_layer.update()
only(K)
info = mi.gather(ctx)
check("exact circle r 0.7 scaled x3: length 2 pi 2.1", len(info["curves"]) == 1
      and rel(info["curves"][0], 2 * math.pi * 2.1) < 1e-6, f"{info['curves']}")
bpy.ops.curve.primitive_bezier_curve_add(); Z = ctx.active_object
kz = features.input_curve(Z, "S0")
Wz = linesolve.kernel_curve(kz, sm, closed=False).W
check("Bezier curve: length matches a fine polyline (1e-4)",
      rel(mi.gather(ctx)["curves"][0], float(np.linalg.norm(np.diff(Wz, axis=0), axis=1).sum())) < 1e-4)
bpy.ops.curve.primitive_nurbs_path_add(); P = ctx.active_object
only(P, Z)
info = mi.gather(ctx)
check("two curves: two lengths", len(info["curves"]) == 2 and any(a == "2 curves" and "Total length" in b
                                                              for a, b in mi.rows(ctx, info)))
# an edge's length via the kernel curve = the kernel's edge length
kc = features.input_curve(S, "E0")
check("curve_length of an edge curve = Edge.length", rel(mi.curve_length(kc), store.get_brep(S).edges()[0].length()) < 1e-6)

# --- text, Copy -----------------------------------------------------------------------------
only(B)
txt = mi.text(ctx)
check("text: volume, area and centroid lines", "Volume" in txt and "Centroid" in txt, txt)
check("the centroid's cross in the view is off at first (the origin stays the one point marked)",
      not mi.show_centroid(ctx))
check("the centroid marker is the cross unless the Centre of Mass symbol is picked",
      mi.centroid_style(ctx) == "CROSS")
check("Copy runs", bpy.ops.nproto.measure_copy() == {'FINISHED'})
only()
check("nothing selected: no readout", mi.gather(ctx) is None and mi.rows(ctx, None) == [])

# --- Measure tool label carries the length ------------------------------------------------
m = ms.curve_radius(linesolve.kernel_curve(features.input_curve(K, "S0"), sm), np.array([2.1, 0, 0]))
m["length"] = 1.5
check("radius label: '· L'", "L " in ms.label(ctx, m), ms.label(ctx, m))

# --- units: the scene's Length unit, adaptive otherwise -----------------------------------
us = ctx.scene.unit_settings
us.length_unit, us.scale_length = 'MILLIMETERS', 0.001
check("mm scene: 158.4 mm, 10816 mm², 64 mm³", mi.fmt(ctx, 158.4) == "158.4 mm" and mi.fmt(ctx, 10816.0, 2) == "10816 mm²"
      and mi.fmt(ctx, 64.0000001, 3) == "64 mm³", f"{mi.fmt(ctx, 158.4)} {mi.fmt(ctx, 10816.0, 2)} {mi.fmt(ctx, 64.0000001, 3)}")
check("no -0, six digits", mi.fmt(ctx, -2.7e-17) == "0 mm" and mi.fmt(ctx, 1 / 3) == "0.333333 mm", mi.fmt(ctx, 1 / 3))
us.system = 'IMPERIAL'; us.length_unit = 'INCHES'; us.scale_length = 1.0
check("inch scene: 1 in", mi.fmt(ctx, 0.0254) == "1 in" and mi.fmt(ctx, 0.0254 ** 2, 2) == "1 in²", mi.fmt(ctx, 0.0254))
us.system = 'METRIC'; us.length_unit = 'ADAPTIVE'
check("adaptive: Blender's units", mi.fmt(ctx, 0.0123456) == "1.2346 cm", mi.fmt(ctx, 0.0123456))
B.location = (1, 2, 3); B.rotation_euler = (0.3, 0.5, 0.7); B.scale = (2, 2, 2); ctx.view_layer.update()
only(B)
v = mi.gather(ctx)["solids"][0][0]
check("turned and scaled x2: the scale exact (64 to 1e-12)", abs(v - 64) < 1e-12 * 64, f"{v!r}")

# --- slow bodies settle before they are measured again ----------------------------------
B.rotation_euler = (0, 0, 0); B.scale = (1, 1, 1); B.location = (0, 0, 0); ctx.view_layer.update()
only(B)
mi.gather(ctx)                                  # measured: 8
mi._cost[B.name] = 1.0                          # as if it took a second
B.scale = (2, 2, 2); ctx.view_layer.update()
i1 = mi.gather(ctx, settle=True)
check("slow body changed: the old volume, marked stale", i1["stale"] and abs(i1["solids"][0][0] - 8) < 1e-9)
k, t = mi._seen[B.name]
mi._seen[B.name] = (k, t - 1.0)                 # it stayed unchanged long enough
i2 = mi.gather(ctx, settle=True)
check("then measured: 64, not stale", not i2["stale"] and abs(i2["solids"][0][0] - 64) < 1e-9)
mi._cost[B.name] = 1.0
B.scale = (3, 3, 3); ctx.view_layer.update()
check("Copy / gather without settle measures at once", abs(mi.gather(ctx)["solids"][0][0] - 216) < 1e-9)
mi._cache.clear(); mi._cost[B.name] = 1.0
B.scale = (1, 1, 1); ctx.view_layer.update()
check("nothing measured yet: 'Measuring…'", mi.rows(ctx, mi.gather(ctx, settle=True)) == [("", "Measuring…")])

# --- radii: circle / arc edges, cylinder / sphere / fillet faces ------------------------
import json
mi._cache.clear(); mi._cost.clear()
us.length_unit, us.scale_length = 'ADAPTIVE', 1.0
def edge_near(o, q):
    es = store.get_brep(o).edges()
    def d(e):
        t0, t1 = e.parameter_range()
        return min(np.linalg.norm(np.array(e.point_at(t0 + (t1 - t0) * k / 8), float) - q) for k in range(9))
    return min(range(len(es)), key=lambda k: d(es[k]))
def face_near(o, q):
    fs = store.get_brep(o).faces()
    return min(range(len(fs)), key=lambda k: fs[k].closest_point(tuple(map(float, q)))[1])

bpy.ops.nproto.add_primitive(kind="CYLINDER"); Cy = ctx.active_object
Cy.location = (0, -6, 0); Cy.scale = (2, 2, 2); ctx.view_layer.update()
only(Cy)
info = mi.gather(ctx)
check("open cylinder object (r 1, x2): its face reads R 2 cylinder", info["sheet_radii"] and info["sheet_radii"][0][:2] == (
    info["sheet_radii"][0][0], "cylinder") and abs(info["sheet_radii"][0][0] - 2) < 1e-9, str(info["sheet_radii"]))
i_top = edge_near(Cy, np.array([1.0, 0, 1.0]))
subsel.set_picks(Cy, "EDGE", [i_top])
info = mi.gather(ctx)
a = info["edge_arcs"][0]
check("its rim circle: R 2, full, Ø in the row", a is not None and abs(a[0] - 2) < 1e-7 and a[2]
      and "Ø 4" in mi.rows(ctx, info)[0][1], f"{a} {mi.rows(ctx, info)}")
subsel.set_picks(Cy, "EDGE", [])

only(S)
subsel.set_picks(S, "FACE", [0])
info = mi.gather(ctx)
check("sphere face: R 1 sphere", info["face_radii"][0] is not None and info["face_radii"][0][1] == "sphere"
      and abs(info["face_radii"][0][0] - 1) < 1e-6, str(info["face_radii"]))
subsel.set_picks(S, "FACE", [])
subsel.set_picks(B, "FACE", [0]); only(B)
check("a box's flat face: no radius", mi.gather(ctx)["face_radii"] == [None]
      and "R " not in mi.rows(ctx, mi.gather(ctx))[0][1])
subsel.set_picks(B, "FACE", [])

# Edge Fillet r 0.5 on a box's top edge: the fillet face R 0.5 (cylinder), its rims quarter arcs
R = bpy.data.curves.new("Rp", 'CURVE'); R.dimensions = '3D'
sp = R.splines.new('POLY'); sp.points.add(3)
for k, (x, y) in enumerate([(0, 0), (4, 0), (4, 3), (0, 3)]):
    sp.points[k].co = (x, y + 10, 0, 1)
sp.use_cyclic_u = True
Ro = bpy.data.objects.new("Rp", R); ctx.collection.objects.link(Ro)
bpy.ops.nproto.extrude(inputs=json.dumps([[Ro.name, "S0"]]), distance=2.0, direction="Z", body="NEW")
X = ctx.active_object
e = edge_near(X, np.array([2.0, 10.0, 2.0]))
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[X.name, f"E{e}"]]), radius=0.5)
X = ctx.active_object
only(X)
f = face_near(X, np.array([2.0, 10.0 + 0.5 - 0.5 * 0.7071, 2.0 - 0.5 + 0.5 * 0.7071]))
subsel.set_picks(X, "FACE", [f])
info = mi.gather(ctx)
fr = info["face_radii"][0]
check("edge fillet r 0.5: its face reads R 0.5 (cylinder)", r == {'FINISHED'} and fr is not None
      and abs(fr[0] - 0.5) < 1e-4 and fr[1] == "cylinder", f"{r} {fr} {mi.rows(ctx, info)}")
subsel.set_picks(X, "FACE", [])
e1 = edge_near(X, np.array([0.0, 10.0 + 0.5 - 0.5 * 0.7071, 2.0 - 0.5 + 0.5 * 0.7071]))
subsel.set_picks(X, "EDGE", [e1])
info = mi.gather(ctx)
a = info["edge_arcs"][0]
check("its end: a quarter arc, R 0.5, 90°", a is not None and abs(a[0] - 0.5) < 1e-4 and abs(a[1] - 90) < 0.05
      and not a[2] and "90° arc" in mi.rows(ctx, info)[0][1], f"{a} {mi.rows(ctx, info)}")
e2 = edge_near(X, np.array([4.0, 10.0 + 0.5 - 0.5 * 0.7071, 2.0 - 0.5 + 0.5 * 0.7071]))
subsel.set_picks(X, "EDGE", [e1, e2])
check("both ends: 'R 0.5 m each'", "each" in mi.rows(ctx, mi.gather(ctx))[0][1], str(mi.rows(ctx, mi.gather(ctx))))
subsel.set_picks(X, "EDGE", [])

# Surface Fillet between two sheets: the fillet object on its own reads its radius
def sheet(name, p0, p1, length=4.0):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(1)
    sp.points[0].co = (*p0, 1); sp.points[1].co = (*p1, 1)
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    bpy.ops.nproto.extrude(inputs=json.dumps([[o.name, "S0"]]), distance=length, direction="Y")
    return ctx.active_object

fa = sheet("Floor", (-2, 30, 0), (2, 30, 0))
fb = sheet("Wall", (0, 30, -1.5), (0, 30, 2))
r = bpy.ops.nproto.fillet(surf_a=fa.name, surf_b=fb.name, radius=0.3, trim_bases=True)
fil = ctx.active_object
only(fil)
info = mi.gather(ctx)
sr = info["sheet_radii"]
check("Surface Fillet r 0.3, selected on its own: R 0.3", r == {'FINISHED'} and sr and all(x is not None for x in sr)
      and all(abs(x[0] - 0.3) < 3e-4 for x in sr), f"{r} {sr} {mi.rows(ctx, info)}")

# an edge fillet round a cylinder's rim: a torus, constant one way: 'fillet'
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); Ci = ctx.active_object
Ci.location = (0, 40, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=json.dumps([[Ci.name, "S0"]]), distance=1.0, direction="Z", body="NEW")
T = ctx.active_object
e = edge_near(T, np.array([1.0, 40.0, 1.0]))
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[T.name, f"E{e}"]]), radius=0.2)
T = ctx.active_object
only(T)
c45 = 0.2 * 0.7071
f = face_near(T, np.array([0.8 + c45, 40.0, 0.8 + c45]))
subsel.set_picks(T, "FACE", [f])
info = mi.gather(ctx)
fr = info["face_radii"][0]
check("rim fillet r 0.2 on a cylinder: R 0.2 fillet (a torus)", r == {'FINISHED'} and fr is not None
      and abs(fr[0] - 0.2) < 2e-4 and fr[1] == "fillet", f"{r} {fr} {mi.rows(ctx, info)}")
subsel.set_picks(T, "FACE", [])

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
