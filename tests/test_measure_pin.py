"""Pinned measurements:  blender -b --factory-startup --python tests/test_measure_pin.py"""
import json, os, sys, tempfile
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import measurepin as pn, store, subsel

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def only(*objs):
    for o in ctx.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0] if objs else None
    ctx.view_layer.update()

def fnear(o, q):
    fs = store.get_brep(o).faces()
    return min(range(len(fs)), key=lambda k: fs[k].closest_point(tuple(map(float, q)))[1])

def enear(o, q):
    es = store.get_brep(o).edges()
    return min(range(len(es)), key=lambda k: es[k].closest_point(tuple(map(float, q)))[2])

def T(p):
    """(text, stale) of a pin (or its label: a Python reference to a pin goes
    bad once more pins are added, the collection moves in memory)."""
    if isinstance(p, str):
        p = next(q for q in ctx.scene.nproto_pins if q.label == p)
    return pn.text(ctx, p)

# --- two faces of a box: follows moves, scales and a rebuild that renumbers faces ---------------
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=1.5, height=1); B = ctx.active_object
bb = np.array([list(v) for v in B.bound_box]); lo, hi = bb.min(0), bb.max(0)
top, bot = fnear(B, [0.3, 0.2, hi[2]]), fnear(B, [0.3, 0.2, lo[2]])
p = pn.add(ctx.scene, pn.element_ref(("F", B.name, top)), pn.element_ref(("F", B.name, bot)))
p = p.label
check("pin two faces: named M1, parallel faces 1", p == "M1" and T(p) == ("Parallel faces 1 m", False), T(p))
B.location = (5, 1, 2); B.rotation_euler = (0.3, 0.1, 0.6); ctx.view_layer.update()
check("moved and turned: still 1", T(p)[0].startswith("Parallel faces 1 m"), T(p))
B.scale = (1, 1, 2); ctx.view_layer.update()
check("scaled x2 along its height: 2", T(p)[0].startswith("Parallel faces 2 m"), T(p))
B.location = (0, 0, 0); B.rotation_euler = (0, 0, 0); B.scale = (1, 1, 1); ctx.view_layer.update()
nf0 = len(store.get_brep(B).faces())
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[B.name, f"E{enear(B, [hi[0], hi[1], 0.5 * (lo[2] + hi[2])])}"]]), radius=0.2)
B2 = ctx.active_object
check("a fillet on a side edge renumbers the faces", len(store.get_brep(B2).faces()) == nf0 + 1)
check("the pin finds its faces again: still 1", T(p) == ("Parallel faces 1 m", False), T(p))

# --- points: on an object (follows it) and in world space --------------------------------------
p2 = pn.add(ctx.scene, pn.point_ref([0, 0, 5]), pn.point_ref([0, 0, hi[2]], B2)).label
check("two points: 5 - top", T(p2)[0].startswith("Distance 4.5 m"), T(p2))
B2.location = (0, 0, 1); ctx.view_layer.update()
check("the object's point moves with it, the world point stays", T(p2)[0].startswith("Distance 3.5 m"), T(p2))
B2.location = (0, 0, 0); ctx.view_layer.update()

# --- one thing's own reading: a rim's radius, a cylinder face's radii, a curve -----------------
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.5, height=2.0, caps=True); C = ctx.active_object
C.location = (10, 0, 0); ctx.view_layer.update()
cb = np.array([list(v) for v in C.bound_box]); z1 = cb[:, 2].max()
p3 = pn.add(ctx.scene, pn.element_ref(("E", C.name, enear(C, [0.5, 0, z1]))), diameter=True).label
check("a rim, diameter: Ø 1 circle", T(p3)[0].startswith("Ø 1 m") and "circle" in T(p3)[0], T(p3))
side = fnear(C, [0.5, 0, 0])
p4 = pn.add(ctx.scene, pn.element_ref(("F", C.name, side), at=np.array([10.5, 0, 0.0]))).label
check("a cylinder face at a point: R 0.5 and flat", T(p4)[0].startswith("R 50 cm") or T(p4)[0].startswith("R 0.5"), T(p4))
cu = bpy.data.curves.new("L", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('POLY'); sp.points.add(1)
sp.points[0].co = (0, 5, 0, 1); sp.points[1].co = (3, 9, 0, 1)
L = bpy.data.objects.new("L", cu); ctx.collection.objects.link(L); ctx.view_layer.update()
p5 = pn.add(ctx.scene, pn.element_ref(("S", L.name, "S0"))).label
check("a straight curve: its length 5", "L 5 m" in T(p5)[0], T(p5))
sp.points[1].co = (6, 13, 0, 1); ctx.view_layer.update()
check("the curve edited: 10", "L 10 m" in T(p5)[0], T(p5))

# --- gone: stale, keeping the last value ---------------------------------------------------------
bpy.data.objects.remove(C); ctx.view_layer.update()
check("the cylinder deleted: stale, last value kept", T(p3)[1] and T(p3)[0].startswith("Ø 1 m"), T(p3))

# --- operators: pin the readout's pair, copy, remove, clear --------------------------------------
only(B2)
subsel.set_picks(B2, "FACE", [fnear(B2, [0.3, 0.2, hi[2]]), fnear(B2, [0.3, lo[1], 0.0])])
n = len(ctx.scene.nproto_pins)
r = bpy.ops.nproto.measure_pin_selection()
pp = ctx.scene.nproto_pins[-1]
check("Pin Measurement from the readout: 90° between the faces", r == {'FINISHED'} and len(ctx.scene.nproto_pins) == n + 1
      and "Angle 90°" in T(pp)[0], T(pp))
subsel.set_picks(B2, "FACE", [])
check("nothing picked: nothing to pin", bpy.ops.nproto.measure_pin_selection() == {'CANCELLED'})
clip = "\n".join(pn.copy_lines(ctx))          # (no clipboard in background mode)
check("Copy: one line a pin, stale ones marked", bpy.ops.nproto.measure_pin_copy() == {'FINISHED'}
      and clip.count("\n") == len(ctx.scene.nproto_pins) - 1 and "M1: Parallel faces 1 m" in clip and "(stale)" in clip, clip)
bpy.ops.nproto.measure_pin_remove(index=len(ctx.scene.nproto_pins) - 1)
check("Remove: one fewer", len(ctx.scene.nproto_pins) == n)

# --- saved with the file -----------------------------------------------------------------------------
path = os.path.join(tempfile.mkdtemp(), "pins.blend")
bpy.ops.wm.save_as_mainfile(filepath=path)
labels = [q.label for q in ctx.scene.nproto_pins]
bpy.ops.wm.open_mainfile(filepath=path)
ctx = bpy.context
check("saved and opened: the pins are back", [q.label for q in ctx.scene.nproto_pins] == labels, labels)
check("caches dropped on load, values measured again", T(ctx.scene.nproto_pins[0]) == ("Parallel faces 1 m", False),
      T(ctx.scene.nproto_pins[0]))
bpy.ops.nproto.measure_pin_clear()
check("Clear: none left", len(ctx.scene.nproto_pins) == 0)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
