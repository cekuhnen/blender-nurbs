"""Curve selection in Edit Curve mode (curvesel.py): what a click picks in
CV / Hull / Curve mode, Shift toggles, whole curves after box / lasso.
    blender -b --factory-startup --python tests/test_curve_select.py
Runs headless against a fixed top view (no window): the picker's rules, not
the event routing (tests/test_curve_select_gui.py covers that)."""
import os, sys, types

import bpy
import numpy as np
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import curvesel

ok = True


def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else extra)
    ok &= bool(cond)


for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

# a stadium like the report: two arcs (3 CVs, order 3) and two straight sides
cu = bpy.data.curves.new("Stadium", 'CURVE')
cu.dimensions = '3D'


def nurbs(pts, order):
    s = cu.splines.new('NURBS')
    s.points.add(len(pts) - 1)
    for p, c in zip(s.points, pts):
        p.co = (*c, 1.0)
    s.order_u = order
    s.use_endpoint_u = True
    return s


nurbs([(-1, 3, 0), (-1, 4, 0), (1, 4, 0)], 3)          # top arc (a half of one: enough here)
nurbs([(1, 3, 0), (1, -3, 0)], 2)                      # right side
nurbs([(1, -3, 0), (1, -4, 0), (-1, -4, 0)], 3)        # bottom arc
nurbs([(-1, -3, 0), (-1, 0, 0), (-1, 3, 0)], 2)        # left side: 3 CVs on a line
ob = bpy.data.objects.new("Stadium", cu)
bpy.context.scene.collection.objects.link(ob)
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')

# top view: world (x, y) -> pixels (400 + 50 x, 300 + 50 y)
W, H, S = 800, 600, 50.0
PM = Matrix(((2 * S / W, 0, 0, 0), (0, 2 * S / H, 0, 0), (0, 0, -0.01, 0), (0, 0, 0, 1)))


def px(x, y):
    return (400 + S * x, 300 + S * y)


ctx = types.SimpleNamespace(region=types.SimpleNamespace(width=W, height=H),
                            region_data=types.SimpleNamespace(perspective_matrix=PM),
                            objects_in_mode=[ob], window_manager=bpy.context.window_manager)
v = bpy.context.window_manager.nproto_view


def sel():
    return [tuple(p.select for p in s.points) for s in ob.data.splines]


def kind(h):
    return None if h is None else (h[1], h[2], tuple(h[3]))


# --- CV mode
v.curve_sel_mode = "CV"
check("CV: near a CV is Blender's", kind(curvesel.hit_at(ctx, px(1, 3.05)))[2] == (0,) and
      curvesel.hit_at(ctx, px(1, 3.05))[2] == "CV")
h = curvesel.hit_at(ctx, px(1.02, 0.0))
check("CV: middle of a straight side picks the curve", kind(h) == (1, "CURVE", (0, 1)), kind(h))
h = curvesel.hit_at(ctx, px(-1.02, 1.5))
check("CV: a 3-CV straight curve on its hull line counts as the curve", kind(h) == (3, "CURVE", (0, 1, 2)), kind(h))
h = curvesel.hit_at(ctx, px(-1.0, 3.5))
check("CV: the arc's hull line off the curve picks the span", kind(h) == (0, "SPAN", (0, 1)), kind(h))
check("CV: empty space picks nothing", curvesel.hit_at(ctx, px(0, 0)) is None)

curvesel.apply(ctx, curvesel.hit_at(ctx, px(1.02, 0.0))[:4], False)
check("click selects the whole side", sel() == [(False,) * 3, (True, True), (False,) * 3, (False,) * 3], sel())
before = {ob.name: curvesel.flags(ob)}
curvesel.apply(ctx, curvesel.hit_at(ctx, px(-1.0, 3.5))[:4], True, before)
check("Shift+click adds the span", sel()[0] == (True, True, False) and sel()[1] == (True, True), sel())
before = {ob.name: curvesel.flags(ob)}
curvesel.apply(ctx, curvesel.hit_at(ctx, px(1.02, 0.0))[:4], True, before)
check("Shift+click on a selected curve removes it", sel()[1] == (False, False) and sel()[0] == (True, True, False),
      sel())
# Blender's press-select changed the flags first: the click puts them back
before = {ob.name: curvesel.flags(ob)}
ob.data.splines[2].points[0].select = True
curvesel.apply(ctx, curvesel.hit_at(ctx, px(1.02, 0.0))[:4], True, before)
check("press-select undone before the toggle", sel()[2] == (False,) * 3 and sel()[1] == (True, True), sel())

# --- Hull mode
v.curve_sel_mode = "HULL"
h = curvesel.hit_at(ctx, px(1.0, 3.08))
check("Hull: near a CV picks the nearest span", h is not None and h[2] == "SPAN", kind(h))
h = curvesel.hit_at(ctx, px(-1.01, 1.5))
check("Hull: a span of the straight left side", kind(h) == (3, "SPAN", (1, 2)), kind(h))
curvesel.apply(ctx, None, False)
check("Hull: click on nothing deselects", not any(any(f) for f in sel()), sel())

# --- Curve mode
v.curve_sel_mode = "CURVE"
h = curvesel.hit_at(ctx, px(-1.0, 3.05))
check("Curve: near a CV picks its whole curve", h is not None and h[2] == "CURVE" and len(h[3]) in (2, 3), kind(h))
# switching to Curve turns a partial pick into whole curves
v.curve_sel_mode = "CV"
ob.data.splines[0].points[2].select = True
v.curve_sel_mode = "CURVE"
check("switch to Curve: partial picks grow to whole curves", sel()[0] == (True,) * 3, sel())

# box / lasso after Blender: whole curves in and out
bpy.ops.curve.select_all(action='DESELECT')
curvesel.SNAP[ob.name] = curvesel.flags(ob)
ob.data.splines[3].points[1].select = True                 # Blender's box took the middle CV


class Op:
    bl_idname = "VIEW3D_OT_select_box"

    def __init__(self, **kw):
        self.properties = types.SimpleNamespace(**kw)

    def as_pointer(self):
        return id(self)


curvesel.RUN["where"] = (None, None)
curvesel._whole_curves(ctx, Op(xmin=0, xmax=1, ymin=0, ymax=1, mode="SET"))     # (box far off the other curves)
check("box over one CV: its whole curve", sel()[3] == (True,) * 3 and not any(sel()[1]), sel())

# a box across the right side's line, no CV inside: the side comes in
curvesel.SNAP[ob.name] = curvesel.flags(ob)


class _Ctx(types.SimpleNamespace):
    def temp_override(self, **kw):
        import contextlib
        return contextlib.nullcontext()


ctx2 = _Ctx(**vars(ctx))
orig_project = curvesel._project
curvesel._project = lambda c, Wp: orig_project(ctx, Wp)
try:
    x0, y0 = px(0.8, -0.5)
    x1, y1 = px(1.2, 0.5)
    curvesel._whole_curves(ctx2, Op(xmin=x0, xmax=x1, ymin=y0, ymax=y1, mode="ADD"))
    check("box across a curve's line (no CV in it) adds the curve", sel()[1] == (True, True) and sel()[3] == (True,) * 3,
          sel())
    curvesel.SNAP[ob.name] = curvesel.flags(ob)
    ob.data.splines[3].points[0].select = False             # Blender's deselect box took one CV off
    curvesel._whole_curves(ctx2, Op(xmin=0, xmax=1, ymin=0, ymax=1, mode="SUB"))
    check("deselect box takes the whole curve out", sel()[3] == (False,) * 3 and sel()[1] == (True, True), sel())
    curvesel.SNAP[ob.name] = curvesel.flags(ob)
    L = types.SimpleNamespace(bl_idname="VIEW3D_OT_select_lasso", as_pointer=lambda: 1,
                              properties=types.SimpleNamespace(
                                  mode="ADD", path=[types.SimpleNamespace(loc=px(*q)) for q in
                                                    ((-0.5, 3.3), (0.5, 3.3), (0.5, 4.2), (-0.5, 4.2))]))
    curvesel._whole_curves(ctx2, L)
    check("lasso over the top arc's line adds it", sel()[0] == (True,) * 3, sel())
finally:
    curvesel._project = orig_project

bpy.ops.object.mode_set(mode='OBJECT')
print("ALL OK" if ok else "SOME FAILED")
sys.stdout.flush()
os._exit(0 if ok else 1)
