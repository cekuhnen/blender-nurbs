"""Draw Box:  blender -b --factory-startup --python tests/test_drawbox.py"""
import json, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import drawshapes as ds, features, store, surfeval, toolgroups
from blender_nurbs.feature_ui import primitive_params

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

n, u = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
A = np.array([1.0, 2.0, 0.5])

# --- the base rectangle in each mode ------------------------------------------------
C, X, Y, x = ds.base_rect("CORNERS", [A], A + [2.0, -1.0, 0.0], n, u, {})
check("corner to corner: centre between the corners", np.allclose(C, A + [1.0, -0.5, 0]) and (X, Y) == (2.0, 1.0))
C, X, Y, x = ds.base_rect("CORNERS", [A], A + [-2.0, -1.0, 0.0], n, u, {"X": 3.0})
check("corner to corner: typed width, mouse gives the side", np.allclose(C, A + [-1.5, -0.5, 0]) and (X, Y) == (3.0, 1.0))
C, X, Y, x = ds.base_rect("CENTER", [A], A + [0.5, 0.25, 0.0], n, u, {})
check("centre to corner: centred, twice the offset", np.allclose(C, A) and (X, Y) == (1.0, 0.5))
leg = ds.base_rect("THREE", [A], A + [3.0, 4.0, 0.0], n, u, {})
check("3 points, first leg: its end and length", isinstance(leg[0], str) and np.allclose(leg[1], A + [3, 4, 0]) and abs(leg[2] - 5) < 1e-12)
B = A + [3.0, 4.0, 0.0]
C, X, Y, x = ds.base_rect("THREE", [A, B], A + [-4.0, 3.0, 0.0] + [3, 4, 0], n, u, {})
check("3 points: rotated, the side's direction", np.allclose(x, [0.6, 0.8, 0]) and abs(X - 5) < 1e-12 and abs(Y - 5) < 1e-12
      and np.allclose(C, A + 0.5 * np.array([3, 4, 0]) + 2.5 * np.array([-0.8, 0.6, 0])))

# --- the tool's steps, without a 3D view ---------------------------------------------
attrs = {}
for cls in reversed(ds.NPROTO_OT_draw_box.__mro__):
    if cls.__module__.startswith("blender_nurbs"):
        attrs.update({k: v for k, v in vars(cls).items() if not k.startswith("__")})
Fake = type("Fake", (), attrs)

class Ev:
    value, type, ctrl, alt, oskey, shift = 'PRESS', 'LEFTMOUSE', False, False, False, False

def tool(mode):
    f = Fake()
    f.mode, f.degree, f.spans_u = mode, 1, 1
    f._reset()
    f._plane = (A, n, u, np.cross(n, u))
    f.fix_plane = lambda context, P, **k: None
    f.done = None
    f._finish = lambda context, sh: setattr(f, "done", sh) or {'FINISHED'}
    return f

def click(f, P, typed=None, snap=None):
    f._point = lambda context, event: (np.asarray(P, float), snap)
    f._vals = dict(typed or {})
    return f._click(ctx, Ev())

for mode, pts, base_c in (("CORNERS", [A, A + [2.0, 1.0, 0]], A + [1.0, 0.5, 0]),
                          ("CENTER", [A, A + [1.0, 0.5, 0]], A),
                          ("THREE", [A, A + [2.0, 0, 0], A + [1.0, 1.0, 0]], A + [1.0, 0.5, 0])):
    f = tool(mode)
    for P in pts:
        click(f, P)
    check(f"{mode.lower()}: base done, height next", f._height_step() and f._fields() == ["H"]
          and np.allclose(f._base[0], base_c), f"{f._base[0] if f._base else None}")
    out = click(f, A, {"H": 0.8})
    sh = f.done
    check(f"{mode.lower()}: a 2 x 1 x 0.8 box on the base", out == {'FINISHED'} and sh is not None
          and sh["op"] == "prim_box" and abs(sh["params"]["width"] - 2) < 1e-12 and abs(sh["params"]["depth"] - 1) < 1e-12
          and abs(sh["params"]["height"] - 0.8) < 1e-12 and np.allclose(sh["T"][:3, 3], base_c + [0, 0, 0.4]),
          f"{sh and sh['params']}")
    check(f"{mode.lower()}: preview, 12 edges", sh is not None and len(sh["polys"]) == 6
          and sum(len(p) - 1 for p in sh["polys"]) == 12)

f = tool("CORNERS")
click(f, A)
click(f, A + [2.0, 1.0, 0])
f._h_sign = -1.0
click(f, A, {"H": -0.6})
check("negative height: below the base", f.done is not None and np.allclose(f.done["T"][:3, 3], A + [1.0, 0.5, -0.3]))

# Diagonal: a snapped corner off the plane closes the box; on the plane, the height comes next
VERT = ("vertex", None, None, "vertex")
f = tool("DIAGONAL")
click(f, A)
out = click(f, A + [2.0, -1.0, 0.8], snap=VERT)
sh = f.done
check("diagonal: a corner in 3D closes the box", out == {'FINISHED'} and sh is not None
      and np.allclose([sh["params"][k] for k in ("width", "depth", "height")], [2, 1, 0.8])
      and np.allclose(sh["T"][:3, 3], A + [1.0, -0.5, 0.4]), f"{sh and sh['params']}")
f = tool("DIAGONAL")
click(f, A)
click(f, A + [-2.0, 1.0, -0.6], snap=VERT)
check("diagonal: below the plane", f.done is not None and abs(f.done["params"]["height"] - 0.6) < 1e-12
      and np.allclose(f.done["T"][:3, 3], A + [-1.0, 0.5, -0.3]))
f = tool("DIAGONAL")
click(f, A)
out = click(f, A + [2.0, 1.0, 0.0])
check("diagonal, corner on the plane: on to the height", out is None and f.done is None and f._height_step())
click(f, A, {"H": 0.5})
check("diagonal: then the height", f.done is not None and np.allclose(f.done["T"][:3, 3], A + [1.0, 0.5, 0.25]))
f = tool("DIAGONAL")
click(f, A)
check("diagonal: width, depth and height typed", f._fields() == ["X", "Y", "H"])
click(f, A + [1.0, 1.0, 0.0], {"X": 3.0, "Y": 2.0, "H": 1.0})
check("diagonal: typed sizes close it", f.done is not None
      and np.allclose([f.done["params"][k] for k in ("width", "depth", "height")], [3, 2, 1]))

f = tool("THREE")
for P in (A, A + [2.0, 0, 0], A + [1.0, 1.0, 0]):
    click(f, P)
f._buf, f._vals = "", {}
r = f._extra_key(ctx, type("E", (), {"value": 'PRESS', "type": 'BACK_SPACE'})())
check("backspace in the height step: back to the base", r and not f._height_step() and len(f._pts) == 2
      and f._fields() == ["Y"])

# --- what it places (execute: matrix + parameters) -----------------------------------
T = ds.frame(A + [1.0, 0.5, 0.4], np.array([0.6, 0.8, 0.0]), n)
params = primitive_params("prim_box", width=2.0, depth=1.0, height=0.8, degree=1, spans_u=1)
data = {"matrix": T.tolist(), "op": "prim_box", "params": params, "name": "NURBS Box"}
r = bpy.ops.nproto.draw_box(data=json.dumps(data), width=2.0, depth=1.0, height=0.8, degree=1, spans_u=1)
o = ctx.active_object
faces = store.get_brep(o).faces()
M = np.array(o.matrix_world, float)
P = []
for fc in faces:
    E = surfeval.Surf(fc.surface())
    G = E.grid(np.linspace(*E.u_domain, 5), np.linspace(*E.v_domain, 5), 0)[0, 0].reshape(-1, 3)
    P.append(G @ M[:3, :3].T + M[:3, 3])
P = np.vstack(P)
loc = (P - (A + [1.0, 0.5, 0.4])) @ np.stack([[0.6, 0.8, 0], [-0.8, 0.6, 0], n]).T
check("placed: a solid of 6 faces", r == {'FINISHED'} and len(faces) == 6, f"{len(faces)} faces")
check("placed: sizes and rotation", np.allclose(loc.min(0), [-1, -0.5, -0.4], atol=1e-6)
      and np.allclose(loc.max(0), [1, 0.5, 0.4], atol=1e-6), f"{loc.min(0)} {loc.max(0)}")
h = features.history(o)
check("history: prim_box, editable", h["op"] == "prim_box" and abs(h["params"]["height"] - 0.8) < 1e-6, str(h["params"]))
r = bpy.ops.nproto.draw_box(data=json.dumps(data), width=2.0, depth=1.0, height=0.8, degree=3, spans_u=2)
E = surfeval.Surf(store.get_brep(ctx.active_object).faces()[0].surface())
check("redo panel: degree 3, 2 spans", r == {'FINISHED'} and (E.p, E.q) == (3, 3) and (E.nu, E.nv) == (5, 5),
      f"degree ({E.p}, {E.q}), CVs {E.nu} x {E.nv}")

# --- in the palette, next to Plane and Cylinder ------------------------------------
row = [t[0] for t in toolgroups.GROUPS[1][2][0][1][0]]
check("palette: Plane, Box, Cylinder", row == ["nproto.draw_plane", "nproto.draw_box", "nproto.draw_cylinder"], row)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
