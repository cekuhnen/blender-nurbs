"""Surface Fillet with a variable radius (Variable Radius, start / end):
    blender -b --factory-startup --python tests/test_surface_fillet_variable.py

Two square sheets meeting edge to edge at 90 deg (a floor z = 0, x 0..L and a
wall x = 0, z 0..L, both y 0..W): the crossing is the y axis. A ball radius
r(y) linear in y has its rails at x = r(y) on the floor and z = r(y) on the
wall (the setback at 90 deg is the radius), so each sheet loses exactly
W (r0 + r1) / 2, and the fillet's area is the integral over its sections."""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store
from blender_nurbs.sweeps import _world_brep

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
L, W = 0.5, 0.4
WF = float(np.float32(W))           # the sheets' width as stored (float properties)


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)


def poly(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1.0)
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o


def sheet(name, p0, p1):
    ln = poly(name + "_ln", [p0, p1])
    bpy.ops.nproto.extrude(inputs=json.dumps([[ln.name, "S0"]]), distance=W, direction="Y")
    o = ctx.active_object; o.name = name
    bpy.data.objects.remove(ln)
    return o


def corner():
    clear()
    return sheet("Floor", (0, 0, 0), (L, 0, 0)), sheet("Wall", (0, 0, 0), (0, 0, L))


def A(o): return store.get_brep(o).area(1e-7)
def on_edges(o, p): return min(e.closest_point(tuple(p))[2] for e in _world_brep(o).edges())


def edge_pts(o, n=32):
    return np.array([e.point_at(t0 + (t1 - t0) * k / n) for e in _world_brep(o).edges()
                     for t0, t1 in [e.parameter_range()] for k in range(n + 1)])


def law(r0, r1, y0):
    """r(y): r0 at y = y0 (the start end, 0 or W), r1 at the other."""
    return lambda y: r0 + (r1 - r0) * (abs(y - y0) / W)


def section_area(r, sec, n=400, m=200):
    """Area of S(y, t) = sec(r(y), t) + (0, y, 0) over y 0..W, t 0..1 (midpoint rule)."""
    ys = (np.arange(n) + 0.5) / n * W
    ts = (np.arange(m) + 0.5) / m
    h, tot = 1e-6, 0.0
    for y in ys:
        for t in ts:
            Sy = (sec(r(y + h), t) - sec(r(y - h), t)) / (2 * h) + np.array([0, 1.0, 0])
            St = (sec(r(y), t + h) - sec(r(y), t - h)) / (2 * h)
            tot += np.linalg.norm(np.cross(Sy, St))
    return tot * (W / n) / m


def arc(r, t):          # centre (r, ., r), from the floor's foot (r, ., 0) to the wall's (0, ., r)
    th = 0.5 * math.pi * t
    return np.array([r - r * math.sin(th), 0.0, r - r * math.cos(th)])


def chamfer(r, t):
    return np.array([r * (1 - t), 0.0, r * t])


def fillet(a, b, **kw):
    r = bpy.ops.nproto.fillet(surf_a=a.name, surf_b=b.name, variable=True, **kw)
    return r, (ctx.active_object if r == {'FINISHED'} else None)


def start_y(fil):
    """The crossing's end the start radius went to (the step's start point)."""
    p = features.history(fil)["params"]
    return float(p["start_point"][1])


ctx.scene.nproto_settings.construction_history = True

for shape, r0, r1, sec in (("CIRCULAR", 0.10, 0.03, arc), ("CHAMFER", 0.08, 0.02, chamfer),
                           ("CIRCULAR", 0.10, 0.0, arc), ("CIRCULAR", 0.0, 0.06, arc)):
    tag = f"{shape.lower()} {r0:g} -> {r1:g}"
    a, b = corner()
    a0, b0 = A(a), A(b)
    r, fil = fillet(a, b, shape=shape, radius=r0, end_radius=r1)
    check(f"{tag}: made", r == {'FINISHED'} and fil is not None, r)
    if fil is None:
        continue
    y0 = start_y(fil)
    check(f"{tag}: the start radius at one end of the crossing", min(abs(y0), abs(y0 - W)) < 1e-6, y0)
    rr = law(r0, r1, y0)
    ys = np.linspace(0.0, W, 9)
    dev = max(max(on_edges(a, (rr(y), y, 0)), on_edges(b, (0, y, rr(y))), on_edges(fil, (rr(y), y, 0)),
                  on_edges(fil, (0, y, rr(y)))) for y in ys)
    check(f"{tag}: rails at the setback r(y) on both sheets and the fillet, at both ends and in between",
          dev < 1e-7, dev)
    lost = W * (r0 + r1) / 2
    check(f"{tag}: each sheet loses exactly the tapering strip, one face each",
          abs(A(a) - (a0 - lost)) < 1e-6 and abs(A(b) - (b0 - lost)) < 1e-6
          and store.get_brep(a).face_count() == 1 and store.get_brep(b).face_count() == 1,
          (A(a), a0 - lost, A(b), b0 - lost))
    want = section_area(rr, sec)
    check(f"{tag}: fillet area as the sections' integral", abs(A(fil) - want) < 2e-5 * max(want, 1e-3),
          (A(fil), want))
    for y_end, r_end in ((y0, r0), (WF if y0 < 0.5 * W else 0.0, r1)):
        if r_end == 0.0:
            P = edge_pts(fil)
            near = P[np.abs(P[:, 1] - y_end) < 1e-8]
            tip = float(np.abs(near[:, [0, 2]]).max()) if len(near) else np.inf
            check(f"{tag}: the 0 end closes exactly on the crossing (its rails meet at the sheets' corner)",
                  tip < 1e-9 and on_edges(a, (0, y_end, 0)) < 1e-9 and on_edges(b, (0, y_end, 0)) < 1e-9
                  and abs(float(P[:, 1].min())) < 1e-9 and abs(float(P[:, 1].max()) - WF) < 1e-9, tip)
    before = [A(o) for o in (a, b, fil)]
    for o in (fil, a, b):
        features.rebuild(o, ctx, force=True)
    after = [A(o) for o in (a, b, fil)]
    dev2 = max(max(on_edges(a, (rr(y), y, 0)), on_edges(fil, (0, y, rr(y)))) for y in ys)
    st = features.stack(a)
    check(f"{tag}: history carries the radii, a rebuild gives the same three surfaces",
          st and st[-1]["op"] == "fillet_trim" and np.allclose(st[-1]["params"].get("radii") or [9, 9], [r0, r1], atol=1e-7)
          and max(abs(x - y) for x, y in zip(before, after)) < 1e-9 and dev2 < 1e-7, (before, after, dev2))

# --- G2 blend: the rails follow the law too -------------------------------------
a, b = corner()
r, fil = fillet(a, b, shape="BLEND", continuity="2", radius=0.09, end_radius=0.0)
rr = law(0.09, 0.0, start_y(fil)) if fil else None
dev = max(max(on_edges(a, (rr(y), y, 0)), on_edges(b, (0, y, rr(y)))) for y in np.linspace(0, W, 9)) if fil else 1
check("blend G2 0.09 -> 0: rails at r(y), tapering to the crossing", r == {'FINISHED'} and dev < 1e-7, dev)

# --- the Feature panel: Radius A is the start radius; both trims follow ----------
from blender_nurbs import feature_ui


class Stub:                 # a layout that records the props and labels drawn
    drawn = []
    def __getattr__(self, n):
        def f(*a_, **k_):
            if n == "prop" and len(a_) > 1:
                Stub.drawn.append(a_[1])
            if n == "label":
                Stub.drawn.append(k_.get("text", ""))
            return Stub()
        return f
    def __setattr__(self, n, v): pass


def panel(obj):
    Stub.drawn = []
    for o in ctx.view_layer.objects:
        o.select_set(False)
    obj.select_set(True); ctx.view_layer.objects.active = obj
    feature_ui.draw_feature(Stub(), ctx)
    return Stub.drawn


for r_new in (0.06, 0.0):
    a, b = corner()
    a0 = A(a)
    r, fil = fillet(a, b, shape="CIRCULAR", radius=0.1, end_radius=0.03)
    feature_ui.sync_from_history(fil)
    drawn = panel(fil)
    pg = fil.nproto_feature
    shown = ("sf_start" in drawn and "radius_a" not in drawn and "radius_b" not in drawn
             and abs(pg.sf_start - 0.1) < 1e-6 and any(t.startswith("Variable Radius:") for t in drawn))
    pg.sf_start = r_new                     # the update rebuilds the fillet and both trimmed sheets
    rr = law(r_new, 0.03, start_y(fil))
    ys = np.linspace(0.0, W, 9)
    dev = max(max(on_edges(a, (rr(y), y, 0)), on_edges(b, (0, y, rr(y))), on_edges(fil, (rr(y), y, 0)))
              for y in ys)
    lost = W * (r_new + 0.03) / 2
    p_ = features.history(fil)["params"]
    check(f"panel: Start Radius (not Radius A / B) for a variable fillet; set to {r_new:g}, the fillet and both "
          "trimmed sheets follow the new law",
          shown and dev < 1e-7 and abs(A(a) - (a0 - lost)) < 1e-6 and abs(A(b) - (a0 - lost)) < 1e-6
          and abs(p_["radii"][0] - r_new) < 1e-7 and abs(abs(p_["radius_a"]) - max(r_new, 0.03)) < 1e-7
          and np.allclose(features.stack(a)[-1]["params"]["radii"], p_["radii"]),
          (shown, dev, A(a), a0 - lost, p_["radii"]))
ctx.scene.nproto_settings.construction_history = False

# --- the preview: two arrows, rails as the worker's law -------------------------
from blender_nurbs.ops_trim import surface_fillet_frames, surface_fillet_chain
a, b = corner()
sf, frac = surface_fillet_chain(surface_fillet_frames(a, b))
check("preview: the crossing as one run, its place along it 0 -> 1",
      len(sf) >= 1 and abs(frac[0][0]) < 1e-12 and abs(frac[-1][-1] - 1) < 1e-12
      and abs(float(np.linalg.norm(sf[0][0][0] - sf[-1][0][-1])) - WF) < 1e-9, len(sf))

from blender_nurbs import valuedrag as vd
from blender_nurbs.ops_trim import NPROTO_OT_fillet, surface_fillet_sections
body = {k: v for k, v in NPROTO_OT_fillet.__dict__.items() if not k.startswith("__") and not k.startswith("bl_")}
op = type("PlainFillet", (vd.ValueDrag,), body)()
for k, v in dict(surf_a=a.name, surf_b=b.name, flip_a=False, flip_b=False, rail_type="RADIUS", same_radius=True,
                 radius=0.1, radius_b=0.1, shape="CIRCULAR", continuity="2", ends="NOTCH", variable=True,
                 end_radius=0.0, _msg="").items():
    setattr(op, k, v)
op.drag_setup(ctx, None)
hs = op.drag_handles()
y0 = float(op._sf[0][0][0][1])
rr = law(0.1, 0.0, y0)
errs = []
for h in hs:            # each arrow on the floor's rail at its sample, giving back its own value
    p_ = vd.ValueDrag._pos(h)
    errs.append(max(abs(p_[0] - rr(p_[1])), abs(p_[2]), abs(h["to_value"](h["param"]) - getattr(op, h["field"]))))
prev = op.drag_preview()
X = surface_fillet_sections(*op._sf[0], *op._radii(*op._sf[0], 0), *op._sides())
rail = max(abs(q[0] - rr(q[1])) for q in X[:, 0])
check("preview: Variable gives a start and an end arrow on the rails r(y), the sections follow the same law",
      op.drag_fields == ("radius", "end_radius") and [h["field"] for h in hs] == ["radius", "end_radius"]
      and max(errs) < 1e-8 and rail < 1e-8 and len(prev["tris"]), (errs, rail))
op.drag_key(ctx, None, "v")
check("preview: V turns it off (one arrow)", op.drag_fields == ("radius",) and len(op.drag_handles()) == 1)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
