"""Align Parts (aligntool.py): elements picked as the tool picks them, then the
operator's execute (move, flip, gap, spin, copy, the moving parts).
    blender -b --factory-startup --python tests/test_align_tool.py
The modal part (hover, clicks, keys) needs a window: see the GUI test."""
import math, os, sys
import bpy
import numpy as np
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import aligntool, snapframes as sf, snapping as sn, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
X, Y, Z = np.eye(3)
close = lambda a, b, tol=1e-6: bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    sn.forget_meshes()

def view(s=6.0, W=1000):
    """Top view, orthographic, centred on the origin; 1 unit = W/(2s) px."""
    VM = np.eye(4)
    VM[2, 3] = -20.0
    PM = np.diag([1 / s, 1 / s, -0.01, 1.0]) @ VM
    return sn.View(PM, VM, W, W)
top = view()
px = lambda x, y, s=6.0, W=1000: np.array([W / 2 + W / 2 * x / s, W / 2 + W / 2 * y / s])

def as_prop(e):
    z = (0.0, 0.0, 0.0)
    return {"type": e["type"], "label": e.get("label", ""), "o": tuple(map(float, e["o"])),
            "d": z if e.get("d") is None else tuple(map(float, e["d"])),
            "hint": z if e.get("hint") is None else tuple(map(float, e["hint"])), "pin": bool(e.get("pin"))}

def M(o):
    return np.array(o.matrix_world, float)

def run(a, b, parts, **kw):
    return bpy.ops.nproto.align_parts(a=as_prop(a), b=as_prop(b), parts="\n".join(p.name for p in parts), **kw)

# -- a NURBS peg's top ring onto a NURBS box's top centre --------------------------
clear()
ctx.scene.cursor.location = (0, 0, 0)
bpy.ops.nproto.add_primitive(kind='BOX', width=2.0, depth=2.0, height=1.0)
BOX = ctx.active_object
BOX.name = "Box"
bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.5, height=1.2, caps=True)
PEG = ctx.active_object
PEG.name = "Peg"
PEG.matrix_world = Matrix.Translation((4.0, 3.0, 0.0))
ctx.view_layer.update()
M0 = M(PEG)
zt = max(f.centroid()[2] for f in store.get_brep(PEG).faces())       # the peg's top, local
zb = max(f.centroid()[2] for f in store.get_brep(BOX).faces())       # the box's top
T = sn.collect([BOX, PEG])
a = sf.element(T, top, px(4.0, 3.0) + (2, 2))                      # the peg's top: its ring centre
b = sf.element(T, top, px(0.0, 0.0) + (2, 2))                      # the box top's centre
check("picked: the peg's ring, the box's centre", a is not None and a["type"] == "RING" and a["obj"] == "Peg"
      and b is not None and b["type"] == "PLANE" and b["pin"] and b["obj"] == "Box",
      (a and (a["type"], a["obj"]), b and (b["type"], b["label"], b["obj"])))

# the moving parts: the element's object unless it lies on the selection
for o in ctx.selected_objects:
    o.select_set(False)
BOX.select_set(True)
check("moving set: an unselected part moves alone", [o.name for o in aligntool.moving_set(ctx.selected_objects, PEG)]
      == ["Peg"])
PEG.select_set(True)
check("moving set: the selection when the element is on it",
      sorted(o.name for o in aligntool.moving_set(ctx.selected_objects, PEG)) == ["Box", "Peg"])
for o in ctx.selected_objects:
    o.select_set(False)
BOX.select_set(True)            # the box selected: the redo / execute still moves the parts named

r = run(a, b, [PEG])
ctx.view_layer.update()
top_w = M(PEG) @ np.array([0, 0, zt, 1.0])
check("ring onto centre: finished", r == {'FINISHED'}, r)
check("ring onto centre: the peg's top on the box's top centre", close(top_w[:3], (0, 0, zb)), top_w[:3])
check("ring onto centre: upside down (the ends face each other)", close(M(PEG)[:3, 2], -Z))
check("ring onto centre: the peg selected, the box not", PEG.select_get() and not BOX.select_get())
check("the box did not move", close(M(BOX), np.eye(4)))

PEG.matrix_world = Matrix(M0.tolist())
r = run(a, b, [PEG], flip=True)
ctx.view_layer.update()
top_w = M(PEG) @ np.array([0, 0, zt, 1.0])
check("flip: upright, its top on the centre (back to back)", close(M(PEG)[:3, 2], Z) and close(top_w[:3], (0, 0, zb)))

PEG.matrix_world = Matrix(M0.tolist())
r = run(a, b, [PEG], gap=0.25)
ctx.view_layer.update()
top_w = M(PEG) @ np.array([0, 0, zt, 1.0])
check("gap: 0.25 off the box", close(top_w[:3], (0, 0, zb + 0.25)), top_w[:3])

PEG.matrix_world = Matrix(M0.tolist())
n0 = len(bpy.data.objects)
r = run(a, b, [PEG], keep=True)
ctx.view_layer.update()
new = [o for o in bpy.data.objects if o.name not in ("Box", "Peg")]
check("copy: the original stays", close(M(PEG), M0))
check("copy: one NURBS copy on the box", len(bpy.data.objects) == n0 + 1 and len(new) == 1 and store.is_nurbs(new[0])
      and close((M(new[0]) @ np.array([0, 0, zt, 1.0]))[:3], (0, 0, zb)) and close(M(new[0])[:3, 2], -Z),
      [o.name for o in new])
check("copy: the copy selected", new and new[0].select_get() and not PEG.select_get())
for o in new:
    bpy.data.objects.remove(o)

# -- a turned mesh cube's top face onto another's: flush, edges in line ---------------
clear()
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
B = ctx.active_object
B.name = "Base"
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(4.0, 0.0, 0.0), rotation=(0, 0, math.radians(30)))
A = ctx.active_object
A.name = "Block"
ctx.view_layer.update()
T = sn.collect([A, B])
a = sf.element(T, top, px(4.3, 0.2))                               # A's top face, off its centre
b = sf.element(T, top, px(-0.4, 0.3))                              # B's top face
c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
hint_ok = a is not None and a["hint"] is not None and min(abs(abs(float(a["hint"] @ X)) - c30),
                                                          abs(abs(float(a["hint"] @ X)) - s30)) < 1e-6
check("picked: two mesh faces, A's hint along its turned edges", a is not None and a["type"] == "PLANE"
      and a["obj"] == "Block" and b is not None and b["type"] == "PLANE" and b["obj"] == "Base" and hint_ok,
      (a and (a["type"], a["label"], a["hint"]), b and (b["type"], b["label"])))
MA0 = M(A)
r = run(a, b, [A])
ctx.view_layer.update()
R = M(A)[:3, :3]
check("face onto face: upside down", close(R[:, 2], -Z))
check("face onto face: edges in line with the base's", all(abs(abs(float(R[:, k] @ X)) - 1) < 1e-6
                                                           or abs(abs(float(R[:, k] @ Y)) - 1) < 1e-6 for k in (0, 1)),
      np.round(R, 4).tolist())
q = M(A) @ np.linalg.inv(MA0) @ np.append(a["o"], 1.0)        # the picked point, carried
check("face onto face: the picked point stays put, now on the base's top", close(q[:3], (a["o"][0], a["o"][1], 1.0)),
      (q[:3], a["o"]))
check("face onto face: the block's centre on top of the base", abs(M(A)[2, 3] - 2.0) < 1e-6)

A.matrix_world = Matrix.Translation((4.0, 0.0, 0.0)) @ Matrix.Rotation(math.radians(30), 4, 'Z')
r = run(a, b, [A], spin=math.radians(90))
ctx.view_layer.update()
check("spin 90: still flush on the base", abs(M(A)[2, 3] - 2.0) < 1e-6 and close(M(A)[:3, 2], -Z))

# -- execute with nothing to do, and without picks ----------------------------------
check("no parts, nothing selected: cancelled",
      (lambda: ([o.select_set(False) for o in ctx.selected_objects],
                bpy.ops.nproto.align_parts())[1])() == {'CANCELLED'})

print("ALL PASS" if ok else "SOME FAILED")
sys.stdout.flush()
os._exit(0 if ok else 1)
