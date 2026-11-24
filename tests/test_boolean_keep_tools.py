"""Boolean Keep Tools: the tool bodies stay shown (history: live inputs the
result follows; no history: not deleted), Restore Inputs leaves them put:
    blender -b --factory-startup --python tests/test_boolean_keep_tools.py"""
import json, os, sys
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
def vol(o): return round(abs(store.get_brep(o).volume(1e-6)), 4)
def box_at(x, size=2):
    bpy.ops.nproto.add_primitive(kind="BOX", width=size, depth=size, height=size, degree="0")
    o = ctx.active_object; o.location = (x, 0, 0); ctx.view_layer.update(); return o
def homes(o): return sorted(c.name for c in o.users_collection)

# history on: target hidden, tool shown and live
ctx.scene.nproto_settings.construction_history = True
a, b = box_at(0), box_at(1)
tools = bpy.data.collections.new("Tools"); ctx.scene.collection.children.link(tools)
tools.objects.link(b)
for c_ in list(b.users_collection):
    if c_ is not tools: c_.objects.unlink(b)
b_homes = homes(b)
r = bpy.ops.nproto.boolean(operation="DIFFERENCE", keep_tools=True, names=json.dumps([a.name, b.name]))
res = ctx.active_object
check("history: difference 4 with the tool kept shown, target hidden",
      r == {'FINISHED'} and vol(res) == 4.0 and a.hide_get() and not b.hide_get()
      and homes(b) == b_homes and not features.is_kept(ctx, b),
      f"{r} vol {vol(res)} a hidden {a.hide_get()} b hidden {b.hide_get()} {homes(b)}")
b.location.x = 1.5; ctx.view_layer.update()
features.rebuild(res, ctx, force=True)
check("history: moving the kept tool rebuilds the result", vol(res) == 6.0, f"vol {vol(res)}")
bpy.ops.object.select_all(action='DESELECT'); res.select_set(True); ctx.view_layer.objects.active = res
r = bpy.ops.nproto.restore_inputs()
check("Restore Inputs: target back, kept tool stays in its own collection",
      r == {'FINISHED'} and not a.hide_get() and not b.hide_get() and homes(b) == b_homes,
      f"{r} {homes(b)}")

# the default still hides both
c, d = box_at(10), box_at(11)
bpy.ops.nproto.boolean(operation="UNION", keep_tools=False, names=json.dumps([c.name, d.name]))
check("default: both inputs hidden", c.hide_get() and d.hide_get() and vol(ctx.active_object) == 12.0)

# history off: target changed in place, tool kept
ctx.scene.nproto_settings.construction_history = False
e, f = box_at(20), box_at(21)
fn = f.name
r = bpy.ops.nproto.boolean(operation="DIFFERENCE", keep_tools=True, names=json.dumps([e.name, f.name]))
check("no history: target cut in place, tool not deleted",
      r == {'FINISHED'} and ctx.active_object == e and vol(e) == 4.0 and fn in bpy.data.objects
      and vol(bpy.data.objects[fn]) == 8.0 and not bpy.data.objects[fn].hide_get())
g, h = box_at(30), box_at(31)
hn = h.name
bpy.ops.nproto.boolean(operation="DIFFERENCE", keep_tools=False, names=json.dumps([g.name, h.name]))
check("no history default: tool deleted", hn not in bpy.data.objects and vol(g) == 4.0)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
