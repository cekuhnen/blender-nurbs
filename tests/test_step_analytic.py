"""STEP export of true cylinders, cones, spheres, tori, planes, circles, ellipses and lines:
blender -b --factory-startup --python tests/test_step_analytic.py"""
import json, math, os, re, sys, tempfile
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import step_export as se, step_import as si, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
TMP = tempfile.mkdtemp()

# --- recognition on samples ------------------------------------------------------------
t = np.linspace(0, 1, 49)
L = np.outer(1 - t, [1, 2, 3]) + np.outer(t, [4, 6, 3])
o, d = se.fit_line(L, 1e-9)
check("line: start and direction", np.allclose(o, [1, 2, 3]) and np.allclose(d, [0.6, 0.8, 0]))
check("bent run is no line", se.fit_line(L + np.outer(np.sin(np.pi * t), [0, 0, 1e-6]), 1e-9) is None)
a = np.linspace(0.3, 0.3 + 4.0, 49)                     # an arc of 4 rad, clockwise round +Z
Ccw = np.column_stack([2 + 0.7 * np.cos(-a), -1 + 0.7 * np.sin(-a), np.full_like(a, 5)])
C, n, r = se.fit_circle(Ccw, 1e-9, 10)
check("circle: centre, radius; the normal makes the points run counter-clockwise (here -Z)",
      np.allclose(C, [2, -1, 5]) and abs(r - 0.7) < 1e-12 and np.allclose(n, [0, 0, -1]), f"{C} {r} {n}")
check("a line is no circle", se.fit_circle(L, 1e-9, 10) is None)
W = np.column_stack([np.cos(a), np.sin(a), 0.3 * np.sin(a)])        # a tilted ellipse
check("an ellipse is no circle", se.fit_circle(W, 1e-9, 10) is None)
el = se.fit_ellipse(W + [1, 2, 3], 1e-9, 10)
mj = np.array([0, 1, 0.3]) / math.sqrt(1.09)
check("ellipse: centre, major axis and radius, minor radius, normal counter-clockwise",
      el is not None and np.allclose(el[0], [1, 2, 3]) and abs(abs(el[2] @ mj) - 1) < 1e-12
      and abs(el[3] - math.sqrt(1.09)) < 1e-12 and abs(el[4] - 1) < 1e-12
      and np.allclose(el[1], np.cross([1, 0, 0], mj)), str(el))
check("ellipse, running the other way: the normal flips", se.fit_ellipse(W[::-1], 1e-9, 10) is not None
      and np.allclose(se.fit_ellipse(W[::-1], 1e-9, 10)[1], -np.cross([1, 0, 0], mj)))
ae = np.linspace(0, 2 * np.pi, 49)
Wf = np.column_stack([0.8 * np.cos(ae), 0.3 * np.sin(ae), np.zeros_like(ae)])
el = se.fit_ellipse(Wf, 1e-9, 10)
check("whole ellipse: radii 0.8, 0.3", el is not None and abs(el[3] - 0.8) < 1e-12 and abs(el[4] - 0.3) < 1e-12)
Hy = np.column_stack([np.cosh(t - 0.5), np.sinh(t - 0.5), np.zeros_like(t)])
Ov = np.column_stack([np.cos(a) * (1 + 1e-6 * np.cos(3 * a)), 0.5 * np.sin(a), np.zeros_like(a)])
check("a hyperbola, a non-planar run, an egg are no ellipses", se.fit_ellipse(Hy, 1e-9, 10) is None
      and se.fit_ellipse(np.column_stack([np.cos(a), np.sin(a), 0.1 * a]), 1e-9, 10) is None
      and se.fit_ellipse(Ov, 1e-9, 10) is None)
uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(-1, 2, 9), indexing="ij")
ax = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
e1 = np.array([0, 0, 1.0]); e2 = np.cross(ax, e1)
G = np.array([3, 4, 5]) + 0.25 * (np.cos(uu)[..., None] * e1 + np.sin(uu)[..., None] * e2) + vv[..., None] * ax
o, a_, r = se.fit_cylinder(G, 1e-9, 10)
d0 = (o - [3, 4, 5]) - ((o - [3, 4, 5]) @ ax) * ax
check("cylinder: radius, axis, a point on the axis", abs(r - 0.25) < 1e-12 and abs(abs(a_ @ ax) - 1) < 1e-12
      and np.linalg.norm(d0) < 1e-12, f"{r} {a_}")
check("cylinder, rulings along u (transposed grid)", se.fit_cylinder(G.transpose(1, 0, 2), 1e-9, 10) is not None)
Gcone = np.array([3, 4, 5]) + (0.25 + 0.1 * vv)[..., None] * (np.cos(uu)[..., None] * e1 + np.sin(uu)[..., None] * e2) \
    + vv[..., None] * ax
check("a cone is no cylinder", se.fit_cylinder(Gcone, 1e-9, 10) is None)
P = np.array([1, 2, 3]) + uu[..., None] * e1 + vv[..., None] * ax
check("plane: found; a cylinder is no plane", se.fit_plane(P, 1e-9) is not None and se.fit_plane(G, 1e-9) is None)
# a cone, apex (3,4,5) - 2.5 ax, radius 0.25 + 0.1 v: semi-angle atan(0.1)
co = se.fit_cone(Gcone, 1e-9, 10)
check("cone: apex, axis into the face, semi-angle", co is not None
      and np.linalg.norm(co[0] - (np.array([3, 4, 5]) - 2.5 * ax)) < 1e-9 and abs(co[1] @ ax - 1) < 1e-12
      and abs(co[2] - math.atan(0.1)) < 1e-12, f"{co}")
co = se.fit_cone(Gcone[:, ::-1].transpose(1, 0, 2), 1e-9, 10)
check("cone, rulings along u and running to the apex", co is not None and abs(co[1] @ ax - 1) < 1e-12)
uu2, ww = np.meshgrid(np.linspace(0.2, 1.4, 9), np.linspace(0, 1, 7), indexing="ij")     # a partial, pointed one
Gtip = (ww * 0.7)[..., None] * (np.cos(uu2)[..., None] * e1 + np.sin(uu2)[..., None] * e2) + (ww * 1.2)[..., None] * ax
co = se.fit_cone(Gtip, 1e-9, 10)
check("partial cone with its tip in the face", co is not None and np.linalg.norm(co[0]) < 1e-9
      and abs(co[2] - math.atan2(0.7, 1.2)) < 1e-12, f"{co}")
check("a cylinder is no cone; a bent cone is no cone", se.fit_cone(G, 1e-9, 10) is None
      and se.fit_cone(Gcone + 1e-6 * np.sin(np.pi * (vv + 1) / 3)[..., None] * e1, 1e-9, 10) is None)
Gboth = np.array([3, 4, 5]) + (0.1 * vv)[..., None] * (np.cos(uu)[..., None] * e1 + np.sin(uu)[..., None] * e2) \
    + vv[..., None] * ax                                    # v from -1 to 2: through the apex, both nappes
check("a double cone (both nappes) is not written as one", se.fit_cone(Gboth, 1e-9, 10) is None)
# a sphere and a torus round the tilted axis, centre (3,4,5)
ph = np.linspace(-1.2, 0.9, 9)[None, :]
Gsph = np.array([3, 4, 5]) + 0.8 * (np.cos(ph)[..., None] * (np.cos(uu[:, :1])[..., None] * e1
                                                            + np.sin(uu[:, :1])[..., None] * e2) + np.sin(ph)[..., None] * ax)
sp = se.fit_sphere(Gsph, 1e-9, 10)
check("sphere: centre and radius", sp is not None and np.allclose(sp[0], [3, 4, 5]) and abs(sp[1] - 0.8) < 1e-12, f"{sp}")
ra = se.revolution_axis(Gsph, 1e-9, 10)
check("sphere: its parallels give its axis", ra is not None and abs(abs(ra[1] @ ax) - 1) < 1e-12)
check("an ellipsoid is no sphere", se.fit_sphere(Gsph + 0.1 * ((Gsph - [3, 4, 5]) @ ax)[..., None] * ax, 1e-9, 10) is None)
ps = np.linspace(0, 2 * np.pi, 17)[None, :]
Gtor = np.array([3, 4, 5]) + (1.0 + 0.3 * np.cos(ps))[..., None] * (np.cos(uu[:, :1])[..., None] * e1
                                                                   + np.sin(uu[:, :1])[..., None] * e2) \
    + (0.3 * np.sin(ps))[..., None] * ax
to = se.fit_torus(Gtor, 1e-9, 10)
check("torus: centre, axis, major and minor radius", to is not None and np.allclose(to[0], [3, 4, 5])
      and abs(abs(to[1] @ ax) - 1) < 1e-12 and abs(to[2] - 1.0) < 1e-12 and abs(to[3] - 0.3) < 1e-12, f"{to}")
check("torus, parallels along v (transposed grid)", se.fit_torus(Gtor.transpose(1, 0, 2), 1e-9, 10) is not None)
check("a sphere is no torus; a cylinder is no torus", se.fit_torus(Gsph, 1e-9, 10) is None
      and se.fit_torus(G, 1e-9, 10) is None)
Gspin = np.array([3, 4, 5]) + (0.2 + 0.3 * np.cos(ps))[..., None] * (np.cos(uu[:, :1])[..., None] * e1
                                                                    + np.sin(uu[:, :1])[..., None] * e2) \
    + (0.3 * np.sin(ps))[..., None] * ax
check("a spindle torus (tube wider than the ring) is not written as one", se.fit_torus(Gspin, 1e-9, 10) is None)

# --- export: what each shape is written as ------------------------------------------------
def add(kind, **kw):
    bpy.ops.nproto.add_primitive(kind=kind, **kw)
    return ctx.active_object

def kinds(txt):
    return {k: len(re.findall(r"=%s\(" % k, txt)) for k in ("CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "SPHERICAL_SURFACE",
                                                             "TOROIDAL_SURFACE", "PLANE", "CIRCLE", "ELLIPSE", "LINE")} | {
        "BSPLINE_SURF": txt.count("B_SPLINE_SURFACE"), "BSPLINE_CURVE": txt.count("B_SPLINE_CURVE")}

def export(o, name, **kw):
    p = os.path.join(TMP, name + ".step")
    st = se.export_step(p, [o], unit="MM", **kw)
    txt = open(p).read()
    res, rep = si.read_step(p, 1.0)
    return st, txt, res, rep

def vol(b):
    v = b.volume(1e-6)
    return abs(v[0] if isinstance(v, tuple) else v)

cyl = add("CYLINDER", radius=0.5, height=1.2, caps=True, degree="0")
cyl.location, cyl.rotation_euler = (3, 1, 2), (0.4, 0.2, 0.3)
ctx.view_layer.update()
st, txt, res, rep = export(cyl, "cyl")
k = kinds(txt)
check("exact solid cylinder: a CYLINDRICAL_SURFACE, two PLANEs, two CIRCLEs, the seam a LINE; no B-splines",
      k == {"CYLINDRICAL_SURFACE": 1, "CONICAL_SURFACE": 0, "SPHERICAL_SURFACE": 0, "TOROIDAL_SURFACE": 0, "PLANE": 2, "CIRCLE": 2, "ELLIPSE": 0, "LINE": 1, "BSPLINE_SURF": 0, "BSPLINE_CURVE": 0}, str(k))
r_ = float(re.search(r"=CYLINDRICAL_SURFACE\('',#\d+,([^)]+)\)", txt).group(1))
# (within Blender's single-precision object matrix: rotated, it is a hair from rigid)
check("its radius in millimetres", abs(r_ - 500) < 1e-4, f"{r_}")
check("it is still one closed solid and every loop closes", st["solids"] == 1 and st["open_loops"] == [])
check("read back: one solid, all three faces", rep["solids"] == 1 and rep["failed"] == 0 and res
      and res[0][1].face_count() == 3, f"{rep}")
check("read back: the volume is pi r^2 h", res and abs(vol(res[0][1]) / (math.pi * 0.25 * 1.2) - 1) < 1e-4,
      f"{vol(res[0][1]) if res else None}")
_defs = set(re.findall(r"^#(\d+)=", txt, re.M))
check("every referenced entity is defined", set(re.findall(r"#(\d+)", txt)) <= _defs)

st, txt, res, rep = export(cyl, "cyl_bspline", analytic=False)
k = kinds(txt)
check("analytic off: B-splines only, as before", k["CYLINDRICAL_SURFACE"] == k["CONICAL_SURFACE"] == k["SPHERICAL_SURFACE"] == k["TOROIDAL_SURFACE"] == k["PLANE"] == k["CIRCLE"] == k["LINE"] == 0
      and k["BSPLINE_SURF"] and rep["solids"] == 1, str(k))

smooth = add("CYLINDER", radius=0.5, height=1.2, caps=True, degree="3")
st, txt, res, rep = export(smooth, "smooth")
k = kinds(txt)
check("smooth cylinder: its side and rims stay B-splines (not a true cylinder); the caps are flat",
      k["CYLINDRICAL_SURFACE"] == 0 and k["CIRCLE"] == 0 and k["PLANE"] == 2 and rep["solids"] == 1, str(k))

tube = add("CYLINDER", radius=0.5, height=1.2, caps=False, degree="0")
st, txt, res, rep = export(tube, "tube")
check("open tube: a CYLINDRICAL_SURFACE sheet, read back", kinds(txt)["CYLINDRICAL_SURFACE"] == 1 and st["sheets"] == 1
      and rep["failed"] == 0 and res and res[0][1].face_count() == 1, f"{rep}")

# a drilled hole: the cylinder face points into the axis (its bounds written reversed)
bx = add("BOX", width=2, depth=2, height=1)
dr = add("CYLINDER", radius=0.4, height=3, caps=True, degree="0")
r = bpy.ops.nproto.boolean(operation="DIFFERENCE", names=json.dumps([bx.name, dr.name]))
hole = ctx.active_object
st, txt, res, rep = export(hole, "hole")
k = kinds(txt)
check("box with a hole: the hole a CYLINDRICAL_SURFACE, its rims CIRCLEs, the box PLANEs and LINEs",
      r == {'FINISHED'} and k["CYLINDRICAL_SURFACE"] == 1 and k["CIRCLE"] == 2 and k["PLANE"] == 6
      and k["BSPLINE_SURF"] == 0, str(k))
check("box with a hole reads back: one solid, every face, the volume", rep["solids"] == 1 and rep["failed"] == 0
      and res and res[0][1].face_count() == 7 and abs(vol(res[0][1]) / (4.0 - math.pi * 0.16) - 1) < 1e-4,
      f"{rep} {vol(res[0][1]) if res else None}")

# cones: a frustum, a pointed one, a smooth one, a countersink
fr = add("CONE", radius=0.6, radius2=0.25, height=0.9, caps=True, degree="0")
fr.location, fr.rotation_euler = (-2, 1, 0.5), (0.3, -0.5, 0.8)
ctx.view_layer.update()
st, txt, res, rep = export(fr, "frustum")
k = kinds(txt)
check("exact frustum: a CONICAL_SURFACE, two PLANEs, two CIRCLEs, the seam a LINE; no B-splines",
      k == {"CYLINDRICAL_SURFACE": 0, "CONICAL_SURFACE": 1, "SPHERICAL_SURFACE": 0, "TOROIDAL_SURFACE": 0, "PLANE": 2, "CIRCLE": 2, "ELLIPSE": 0, "LINE": 1, "BSPLINE_SURF": 0,
            "BSPLINE_CURVE": 0}, str(k))
m = re.search(r"=CONICAL_SURFACE\('',(#\d+),([^,]+),([^)]+)\)", txt)
check("its semi-angle atan(0.35 / 0.9)", m and abs(float(m.group(3)) - math.atan(0.35 / 0.9)) < 1e-6, m and m.group(3))
check("frustum reads back: one solid, three faces, the volume", rep["solids"] == 1 and rep["failed"] == 0 and res
      and res[0][1].face_count() == 3
      and abs(vol(res[0][1]) / (math.pi * 0.9 / 3 * (0.36 + 0.15 + 0.0625)) - 1) < 1e-4,
      f"{rep} {vol(res[0][1]) if res else None}")
check("frustum: every referenced entity is defined",
      set(re.findall(r"#(\d+)", txt)) <= set(re.findall(r"^#(\d+)=", txt, re.M)))
tip = add("CONE", radius=0.5, radius2=0.0, height=1.0, caps=True, degree="0")
st, txt, res, rep = export(tip, "pointed")
k = kinds(txt)
check("exact pointed cone: a CONICAL_SURFACE and its base a PLANE", k["CONICAL_SURFACE"] == 1 and k["PLANE"] == 1
      and k["BSPLINE_SURF"] == 0, str(k))
check("pointed cone reads back: one solid, the volume", rep["solids"] == 1 and rep["failed"] == 0 and res
      and abs(vol(res[0][1]) / (math.pi * 0.25 / 3) - 1) < 1e-4, f"{rep} {vol(res[0][1]) if res else None}")
sc_ = add("CONE", radius=0.6, radius2=0.25, height=0.9, caps=True, degree="3")
st, txt, res, rep = export(sc_, "smooth_cone")
k = kinds(txt)
check("smooth cone: stays a B-spline", k["CONICAL_SURFACE"] == 0 and k["BSPLINE_SURF"] == 1 and rep["solids"] == 1,
      str(k))
bx = add("BOX", width=2, depth=2, height=1)
cs = add("CONE", radius=0.1, radius2=0.7, height=1.2, caps=True, degree="0")
cs.location = (0, 0, -0.6)          # a tapered hole through the box (z -0.5..0.5): radius 0.15 at the bottom, 0.65 at the top
ctx.view_layer.update()
r = bpy.ops.nproto.boolean(operation="DIFFERENCE", names=json.dumps([bx.name, cs.name]))
sink = ctx.active_object
st, txt, res, rep = export(sink, "countersink")
k = kinds(txt)
check("box with a tapered hole: the sink a CONICAL_SURFACE, its rims CIRCLEs, no B-spline surfaces",
      r == {'FINISHED'} and k["CONICAL_SURFACE"] == 1 and k["CIRCLE"] == 2 and k["BSPLINE_SURF"] == 0, str(k))
check("tapered hole reads back: one solid, every face", rep["solids"] == 1 and rep["failed"] == 0 and res
      and res[0][1].face_count() == store.get_brep(sink).face_count()
      and abs(vol(res[0][1]) / (4 - math.pi / 3 * (0.65 ** 2 + 0.65 * 0.15 + 0.15 ** 2)) - 1) < 1e-4,
      f"{rep} {vol(res[0][1]) if res else None}")

# spheres and tori: whole ones, a spherical dimple, a toroidal groove, a smooth sphere
sph = add("SPHERE", radius=0.7, degree="0")
sph.location, sph.rotation_euler = (1, 2, 3), (0.3, 0.5, 0.7)
ctx.view_layer.update()
st, txt, res, rep = export(sph, "sphere")
k = kinds(txt)
check("exact sphere: a SPHERICAL_SURFACE bounded by one VERTEX_LOOP (no seam), no B-splines",
      k["SPHERICAL_SURFACE"] == 1 and txt.count("VERTEX_LOOP(") == 1 and "EDGE_LOOP(" not in txt
      and k["BSPLINE_SURF"] == k["BSPLINE_CURVE"] == 0 and st["solids"] == 1, str(k))
m = re.search(r"=SPHERICAL_SURFACE\('',#\d+,([^)]+)\)", txt)
check("its radius in millimetres", m and abs(float(m.group(1)) - 700) < 1e-4, m and m.group(1))
check("whole sphere reads back (a VERTEX_LOOP face is the whole sphere): one face, the volume",
      rep["failed"] == 0 and res and res[0][1].face_count() == 1
      and abs(vol(res[0][1]) / (4 / 3 * math.pi * 0.343) - 1) < 1e-4, f"{rep} {vol(res[0][1]) if res else None}")
tor = add("TORUS", radius=1.0, radius2=0.3, degree="0")
tor.location, tor.rotation_euler = (-1, 0, 2), (0.6, 0.1, -0.4)
ctx.view_layer.update()
st, txt, res, rep = export(tor, "torus")
k = kinds(txt)
check("exact torus: a TOROIDAL_SURFACE, its two seams CIRCLEs, no B-splines", k["TOROIDAL_SURFACE"] == 1
      and k["CIRCLE"] == 2 and k["BSPLINE_SURF"] == k["BSPLINE_CURVE"] == 0 and st["solids"] == 1, str(k))
m = re.search(r"=TOROIDAL_SURFACE\('',#\d+,([^,]+),([^)]+)\)", txt)
check("its radii in millimetres", m and abs(float(m.group(1)) - 1000) < 1e-3 and abs(float(m.group(2)) - 300) < 1e-3,
      m and m.groups())
# (our importer splits a whole torus in two faces)
check("torus reads back: every face, the volume 2 pi^2 R r^2", rep["failed"] == 0 and res
      and abs(vol(res[0][1]) / (2 * math.pi ** 2 * 0.09) - 1) < 1e-4,
      f"{rep} {vol(res[0][1]) if res else None}")
bx = add("BOX", width=2, depth=2, height=1)
s2 = add("SPHERE", radius=0.6, degree="0")
s2.location = (0, 0, 0.5)           # centred on the box top: a half-sphere dimple
ctx.view_layer.update()
r = bpy.ops.nproto.boolean(operation="DIFFERENCE", names=json.dumps([bx.name, s2.name]))
dim = ctx.active_object
st, txt, res, rep = export(dim, "dimple")
k = kinds(txt)
check("box with a dimple: the dimple a SPHERICAL_SURFACE, the box PLANEs, no B-spline surfaces",
      r == {'FINISHED'} and k["SPHERICAL_SURFACE"] == 1 and k["PLANE"] == 6 and k["BSPLINE_SURF"] == 0, str(k))
check("dimple reads back: one solid, every face, the volume", rep["solids"] == 1 and rep["failed"] == 0 and res
      and res[0][1].face_count() == 7 and abs(vol(res[0][1]) / (4 - 2 / 3 * math.pi * 0.216) - 1) < 1e-4,
      f"{rep} {vol(res[0][1]) if res else None}")
bx = add("BOX", width=3, depth=3, height=1)
t2 = add("TORUS", radius=0.9, radius2=0.25, degree="0")
t2.location = (0, 0, 0.5)           # centred on the box top: a half-round groove
ctx.view_layer.update()
r = bpy.ops.nproto.boolean(operation="DIFFERENCE", names=json.dumps([bx.name, t2.name]))
gr = ctx.active_object
st, txt, res, rep = export(gr, "groove")
k = kinds(txt)
check("box with a groove: TOROIDAL_SURFACEs, the box PLANEs, no B-spline surfaces",
      r == {'FINISHED'} and k["TOROIDAL_SURFACE"] >= 1 and k["PLANE"] == 7 and k["BSPLINE_SURF"] == 0, str(k))
check("groove reads back: one solid, every face, the volume", rep["solids"] == 1 and rep["failed"] == 0 and res
      and res[0][1].face_count() == store.get_brep(gr).face_count()
      and abs(vol(res[0][1]) / (9 - math.pi ** 2 * 0.9 * 0.0625) - 1) < 1e-4,
      f"{rep} {vol(res[0][1]) if res else None}")
ss = add("SPHERE", radius=0.7, degree="3")
st, txt, res, rep = export(ss, "smooth_sphere")
check("smooth sphere: stays a B-spline", kinds(txt)["SPHERICAL_SURFACE"] == 0 and kinds(txt)["BSPLINE_SURF"] == 1)

# an extruded exact circle and an extruded smooth one
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.6, degree="0"); c = ctx.active_object
bpy.ops.nproto.extrude(inputs=json.dumps([[c.name, "S0"]]), distance=1.5, direction="Z")
st, txt, res, rep = export(ctx.active_object, "ext")
check("extruded exact circle: a CYLINDRICAL_SURFACE", kinds(txt)["CYLINDRICAL_SURFACE"] == 1 and rep["failed"] == 0)
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.6, degree="3"); c = ctx.active_object
bpy.ops.nproto.extrude(inputs=json.dumps([[c.name, "S0"]]), distance=1.5, direction="Z")
st, txt, res, rep = export(ctx.active_object, "ext_smooth")
check("extruded smooth circle: stays a B-spline", kinds(txt)["CYLINDRICAL_SURFACE"] == 0 and kinds(txt)["BSPLINE_SURF"] == 1)

# an elliptic cylinder (a cylinder scaled 1.6 x 0.6, tilted): its rims ELLIPSEs, the caps PLANEs, the side
# a B-spline (no cylinder); a smooth one's rims stay B-splines
ee = add("CYLINDER", radius=0.5, height=1.2, caps=True, degree="0")
ee.location, ee.rotation_euler, ee.scale = (1, -2, 0.5), (0.3, -0.5, 0.8), (1.6, 0.6, 1.0)
ctx.view_layer.update()
st, txt, res, rep = export(ee, "ellipse")
k = kinds(txt)
check("elliptic cylinder: two ELLIPSEs, two PLANEs, the seam a LINE; no cylinder, the side a B-spline",
      k["ELLIPSE"] == 2 and k["PLANE"] == 2 and k["CIRCLE"] == 0 and k["CYLINDRICAL_SURFACE"] == 0
      and k["BSPLINE_SURF"] >= 1 and k["LINE"] == 1 and k["BSPLINE_CURVE"] == 0, str(k))
ax_ = [tuple(map(float, m)) for m in re.findall(r"=ELLIPSE\('',#\d+,([^,]+),([^)]+)\)", txt)]
# (within Blender's single-precision object matrix)
check("ellipse: semi-axes 800 and 300 mm, the major one first", len(ax_) == 2
      and all(abs(a1 / 800 - 1) < 1e-6 and abs(a2 / 300 - 1) < 1e-6 for a1, a2 in ax_), str(ax_))
check("elliptic cylinder reads back: one solid, the volume", rep["solids"] == 1 and rep["failed"] == 0 and res
      and abs(vol(res[0][1]) / (math.pi * 0.8 * 0.3 * 1.2) - 1) < 1e-5, f"{rep} {vol(res[0][1]) if res else None}")
check("elliptic cylinder: no open loops", not se.loop_problems(txt), str(se.loop_problems(txt)))
es = add("CYLINDER", radius=0.5, height=1.2, caps=True, degree="3")
es.scale = (1.6, 0.6, 1.0)
ctx.view_layer.update()
st, txt, res, rep = export(es, "ellipse_smooth")
check("smooth elliptic cylinder: its rims stay B-splines", kinds(txt)["ELLIPSE"] == 0 and kinds(txt)["BSPLINE_CURVE"] >= 2,
      str(kinds(txt)))

# the operator: the option, on by default, and the report
p = os.path.join(TMP, "op.step")
bpy.ops.object.select_all(action='DESELECT'); cyl.select_set(True)
r = bpy.ops.nproto.export_step(filepath=p, use_selection=True)
check("operator: Analytic Surfaces on by default", r == {'FINISHED'} and kinds(open(p).read())["CYLINDRICAL_SURFACE"] == 1)
r = bpy.ops.nproto.export_step(filepath=p, use_selection=True, analytic=False)
check("operator: Analytic Surfaces off", r == {'FINISHED'} and kinds(open(p).read())["CYLINDRICAL_SURFACE"] == 0)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
