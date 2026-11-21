"""Tool palette layout and dimming:  blender -b --factory-startup --python tests/test_toolgroups.py"""
import os, sys
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import store, toolgroups as tg

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


class _Props:
    def __setattr__(self, k, v):
        pass


class Lay:
    """Records the `active` each operator is drawn under (nested rows inherit it)."""
    def __init__(self, out, active=True):
        object.__setattr__(self, "out", out)
        object.__setattr__(self, "_active", active)
    def __setattr__(self, k, v):
        if k == "active":
            object.__setattr__(self, "_active", self._active and v)
    def row(self, **k):
        return Lay(self.out, self._active)
    def column(self, **k):
        return Lay(self.out, self._active)
    def separator(self, **k):
        pass
    def operator(self, op, **k):
        self.out[k.get("text") or op] = self._active
        return _Props()
    def menu(self, m, **k):
        self.out[k.get("text") or m] = self._active


def lit(key):
    out = {}
    tg.draw_panel(Lay(out), key, ctx)
    return out


def select(obj):
    for o in ctx.view_layer.objects:
        o.select_set(False)
    ctx.view_layer.objects.active = obj
    if obj is not None:
        obj.select_set(True)


# --- the layout ------------------------------------------------------------------------
rows = [[t[0].split(".")[1] for t in row] for sec in tg.GROUP["SURFACES"][2][1:] for row in sec[1]]
check("Create Surfaces rows", rows == [
    ["loft", "loft_rails", "network"], ["revolve", "rail_revolve"], ["sweep", "pipe"],
    ["extrude", "flange"], ["planar", "dome"], ["boundary", "patch"],
    ["offset", "thicken"], ["pattern"], ["unroll", "flow"], ["from_blender_surface"]], rows)

# --- dimming -----------------------------------------------------------------------------
cd = bpy.data.curves.new("C", 'CURVE')
cd.splines.new('NURBS').points.add(3)
curve = bpy.data.objects.new("C", cd)
ctx.collection.objects.link(curve)
bpy.ops.nproto.add_primitive(kind="PLANE")
plane = ctx.active_object
sd = bpy.data.curves.new("S", 'SURFACE')
bsurf = bpy.data.objects.new("S", sd)
ctx.collection.objects.link(bsurf)
check("objects are what the tools see", tg.kinds(curve) == {"CURVE"} and store.is_nurbs(plane)
      and tg.kinds(bsurf) == {"SURFACE"}, f"{tg.kinds(curve)} {tg.kinds(bsurf)}")

EDGE_TOOLS = ["Loft", "Loft Rails", "Network", "Revolve", "Rail Revolve", "Sweep", "Pipe",
              "Extrude", "Flange", "Planar", "Dome", "Boundary", "Patch", "Pattern", "Flow"]
SURF_ONLY = ["Offset", "Thicken", "Unroll"]

select(None)
s = lit("SURFACES")
check("nothing selected: nothing dimmed", s and all(s.values()), str([k for k, v in s.items() if not v]))

select(plane)
s = lit("SURFACES")
c = lit("CURVES")
check("a NURBS surface: every tool that takes its edges stays lit",
      all(s.get(n, s.get("nproto." + n.lower().replace(" ", "_"))) for n in EDGE_TOOLS + SURF_ONLY)
      and not s["nproto.from_blender_surface"], str({k: v for k, v in s.items()}))
check("a NURBS surface: Blend Curve and Offset on Surface lit, the other curve tools dim",
      c["nproto.blend_curve"] and c["nproto.offset_on_surface"] and not c["Fillet"] and not c["nproto.offset_curve"]
      and c["CV Curve"], str(c))

select(curve)
s = lit("SURFACES")
lit_names = {k for k, v in s.items() if v}
check("a curve: the surface-only tools dim, Pattern and Flow stay lit beside them",
      not s["nproto.offset"] and not s["nproto.thicken"] and not s["Unroll"] and s["Flow"] and s["Pattern"]
      and s["nproto.loft"] and s["Network"] and s["nproto.extrude"] and s["nproto.flange"]
      and not s["nproto.from_blender_surface"], str(sorted(lit_names)))

select(bsurf)
s = lit("SURFACES")
check("a Blender surface: Convert Blender Surface lit, the curve and edge tools dim",
      s["nproto.from_blender_surface"] and not s["nproto.loft"] and not s["nproto.offset"]
      and s["Plane"] and s["Primitive"], str({k: v for k, v in s.items() if v}))

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
