"""STEP export of true cylinders, planes, circles and lines:
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

# --- export: what each shape is written as ------------------------------------------------
def add(kind, **kw):
    bpy.ops.nproto.add_primitive(kind=kind, **kw)
    return ctx.active_object

def kinds(txt):
    return {k: len(re.findall(r"=%s\(" % k, txt)) for k in ("CYLINDRICAL_SURFACE", "PLANE", "CIRCLE", "LINE")} | {
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
      k == {"CYLINDRICAL_SURFACE": 1, "PLANE": 2, "CIRCLE": 2, "LINE": 1, "BSPLINE_SURF": 0, "BSPLINE_CURVE": 0}, str(k))
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
check("analytic off: B-splines only, as before", k["CYLINDRICAL_SURFACE"] == k["PLANE"] == k["CIRCLE"] == k["LINE"] == 0
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

# an extruded exact circle and an extruded smooth one
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.6, degree="0"); c = ctx.active_object
bpy.ops.nproto.extrude(inputs=json.dumps([[c.name, "S0"]]), distance=1.5, direction="Z")
st, txt, res, rep = export(ctx.active_object, "ext")
check("extruded exact circle: a CYLINDRICAL_SURFACE", kinds(txt)["CYLINDRICAL_SURFACE"] == 1 and rep["failed"] == 0)
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.6, degree="3"); c = ctx.active_object
bpy.ops.nproto.extrude(inputs=json.dumps([[c.name, "S0"]]), distance=1.5, direction="Z")
st, txt, res, rep = export(ctx.active_object, "ext_smooth")
check("extruded smooth circle: stays a B-spline", kinds(txt)["CYLINDRICAL_SURFACE"] == 0 and kinds(txt)["BSPLINE_SURF"] == 1)

# the operator: the option, on by default, and the report
p = os.path.join(TMP, "op.step")
bpy.ops.object.select_all(action='DESELECT'); cyl.select_set(True)
r = bpy.ops.nproto.export_step(filepath=p, use_selection=True)
check("operator: Analytic Surfaces on by default", r == {'FINISHED'} and kinds(open(p).read())["CYLINDRICAL_SURFACE"] == 1)
r = bpy.ops.nproto.export_step(filepath=p, use_selection=True, analytic=False)
check("operator: Analytic Surfaces off", r == {'FINISHED'} and kinds(open(p).read())["CYLINDRICAL_SURFACE"] == 0)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
