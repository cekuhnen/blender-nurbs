"""Sweep: Scale Points along the rail, Scale Taper Linear / Smooth:
    blender -b --factory-startup --python tests/test_sweep_scale.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, sweeps

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
ctx.scene.nproto_settings.construction_history = True

def poly(name, pts, cyclic=False):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ins(*pairs):
    return json.dumps([[o.name, r] for o, r in pairs])

def solid(o):
    try:
        return store.get_brep(o).is_manifold_solid()
    except Exception:
        return False

def vol(o):
    return abs(float(store.get_brep(o).volume(1e-6)))

def sweep(prof, rail, **kw):
    before = {o.name for o in bpy.data.objects}
    try:
        r = bpy.ops.nproto.sweep(inputs=ins((prof, "S0"), (rail, "S0")), **kw)
    except RuntimeError as exc:
        r = str(exc)
    return r, [o for o in bpy.data.objects if o.name not in before]

def pts(*pairs):
    return [{"name": "", "at": a, "scale": s, "remove": False} for a, s in pairs]

bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 0, 0))
circ = ctx.active_object
rail = poly("Rail", [(0, 0, 0), (0, 0, 3)])
fr = lambda a, b, L: math.pi * L * (a * a + a * b + b * b) / 3

r, made = sweep(circ, rail, body="NEW")
check("no points: the cylinder as before", r == {'FINISHED'} and len(made) == 1 and solid(made[0])
      and abs(vol(made[0]) - math.pi * 0.25 * 3) < 1e-4, f"{[round(vol(o), 5) for o in made]}")
r, made = sweep(circ, rail, body="NEW", scale_end=0.5)
check("End Scale alone: the frustum as before", r == {'FINISHED'} and abs(vol(made[0]) - fr(0.5, 0.25, 3)) < 1e-4,
      f"{[round(vol(o), 5) for o in made]} want {fr(0.5, 0.25, 3):.5f}")
r, made = sweep(circ, rail, body="NEW", scale_points=pts((0.5, 2.0)))
want = 2 * fr(0.5, 1.0, 1.5)
check("a scale point (linear): two frustums", r == {'FINISHED'} and len(made) == 1 and solid(made[0])
      and abs(vol(made[0]) - want) < 1e-4, f"{[round(vol(o), 5) for o in made]} want {want:.5f}")
if made:
    b = store.get_brep(made[0])
    lo, hi = b.bounding_box()
    check("  the bulge is exactly scale x the profile", abs(hi[0] - 1.0) < 1e-4, f"{hi}")
    check("  a face per piece, and the caps", b.face_count() == 4, f"{b.face_count()} faces")
    hp = features.history(made[0])["params"]
    check("  history keeps the point", hp.get("scale_points") == [[0.5, 2.0]] and hp.get("scale_taper") == "LINEAR",
          f"{hp.get('scale_points')}")
law = sweeps.RadiusLaw(1.0, 0.6, "SMOOTH", 1.0, [(0.3, 1.8), (0.7, 1.2)])
f = np.linspace(0, 1, 30001)
trap = getattr(np, "trapezoid", None) or np.trapz
want = math.pi * float(trap((0.5 * law(f)) ** 2, 3 * f))
r, made = sweep(circ, rail, body="NEW", scale_end=0.6, scale_taper="SMOOTH", scale_points=pts((0.3, 1.8), (0.7, 1.2)))
check("smooth through two points: volume", r == {'FINISHED'} and len(made) == 1 and solid(made[0])
      and abs(vol(made[0]) - want) < 2e-4, f"{[round(vol(o), 5) for o in made]} want {want:.5f}")
if made:
    check("  one smooth wall (no crease): 1 wall + 2 caps", store.get_brep(made[0]).face_count() == 3,
          f"{store.get_brep(made[0]).face_count()} faces")
line = poly("Ln", [(2, 0, 0), (2.6, 0, 0)])
rail2 = poly("Rail2", [(2, 0, 0), (2, 0, 3)])
r, made = sweep(line, rail2, scale_points=pts((0.4, 2.0), (0.8, 0.5)))
check("open profile, two points: one joined sheet of 3 faces", r == {'FINISHED'} and len(made) == 1
      and store.get_brep(made[0]).face_count() == 3 and not solid(made[0]),
      f"{r} {[store.get_brep(o).face_count() for o in made]}")
# the redo panel's Add button: a point in the middle of the longest stretch, at the scale there
r, made = sweep(circ, rail, body="NEW", scale_end=0.5, add_scale_point=True)
hp = features.history(made[0])["params"] if made else {}
check("F9 Add Scale Point: at 0.5, scale 0.75 (on the straight run, so the volume is unchanged)",
      r == {'FINISHED'} and hp.get("scale_points") == [[0.5, 0.75]] and abs(vol(made[0]) - fr(0.5, 0.25, 3)) < 1e-4,
      f"{hp.get('scale_points')}")
# the redo panel's X on a row: that point goes
r2, made2 = sweep(circ, rail, body="NEW", scale_points=[{"name": "", "at": 0.3, "scale": 2.0, "remove": True},
                                                        {"name": "", "at": 0.6, "scale": 1.5, "remove": False}])
hp2 = features.history(made2[0])["params"] if made2 else {}
sp2 = hp2.get("scale_points") or []
check("F9 X on a row removes that point", r2 == {'FINISHED'} and len(sp2) == 1
      and abs(sp2[0][0] - 0.6) < 1e-6 and abs(sp2[0][1] - 1.5) < 1e-6,
      f"{hp2.get('scale_points')}")
# the Feature panel: Add and Remove
if made:
    o = made[0]
    from blender_nurbs import feature_ui
    feature_ui.sync_from_history(o)
    bpy.ops.nproto.pipe_point_add(name=o.name)
    sp_ = features.history(o)["params"]["scale_points"]
    check("Feature panel: Add Scale Point", len(sp_) == 2, f"{sp_}")
    feature_ui.sync_from_history(o)
    bpy.ops.nproto.pipe_point_remove(name=o.name, index=0)
    sp_ = features.history(o)["params"]["scale_points"]
    check("Feature panel: Remove Scale Point", len(sp_) == 1, f"{sp_}")

# --- the live value stage: its preview and arrows lie on the sweep it makes ---------------------
from blender_nurbs import valuedrag as vd
from blender_nurbs.ops_sweep import NPROTO_OT_sweep, RailPoints

class _Pt:
    def __init__(self, at, scale):
        self.at, self.scale, self.remove = at, scale, False

class _Coll(list):
    def add(self):
        self.append(_Pt(0.5, 1.0)); return self[-1]
    def remove(self, i):
        del self[i]

def stage(prof, rail, ref="S0", **kw):
    """The stage of a Sweep as a plain object (no Blender operator): its own methods."""
    body = {k: v for k, v in NPROTO_OT_sweep.__dict__.items() if not k.startswith("__") and not k.startswith("bl_")}
    o = type("PlainSweep", (RailPoints, vd.ValueDrag), body)()
    attrs = dict(inputs=ins((prof, "S0"), (rail, ref)), samples=32, tangent=True, flip_tangent=False, scale_end=1.0,
                 twist=0.0, keep_height=False, scale_taper="LINEAR", scale_points=_Coll(), body="SURFACE")
    attrs.update(kw)
    for k, v in attrs.items():
        setattr(o, k, v)
    return o

def gap(obj, P):
    b = store.get_brep(obj)
    Mi = np.array(obj.matrix_world.inverted())
    out = 0.0
    for q in P:
        ql = (Mi @ np.append(q, 1.0))[:3]
        out = max(out, min(f.closest_point(tuple(map(float, ql)))[1] for f in b.faces()))
    return out

cu = bpy.data.curves.new("BendRail", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('NURBS'); sp.points.add(3)
for p_, c_ in zip(sp.points, [(0, 0, 0), (0, 0, 1.2), (0.3, 0, 2.2), (1.2, 0, 3)]): p_.co = (*c_, 1.0)
sp.order_u, sp.use_endpoint_u = 4, True
bend = bpy.data.objects.new("BendRail", cu); ctx.collection.objects.link(bend)
oprof = poly("OpenProf", [(0, 0, 0), (0.4, 0, 0.1), (0.7, 0, -0.1)])     # open, standing on the rail's start
for name, prof, rl, kw in (
        ("circle, straight rail, end scale and a linear point", circ, rail,
         dict(scale_end=0.5, scale_points=_Coll([_Pt(0.4, 1.6)]))),
        ("circle, smooth through two points", circ, rail,
         dict(scale_end=0.7, scale_taper="SMOOTH", scale_points=_Coll([_Pt(0.3, 1.5), _Pt(0.7, 0.8)]))),
        ("open profile, curved rail, twist", oprof, bend,
         dict(scale_end=1.4, twist=40.0, scale_points=_Coll([_Pt(0.5, 0.6)])))):
    st = stage(prof, rl, **kw)
    ready = st.drag_setup(ctx, None)
    pv = st.drag_preview() if ready else {"tris": np.zeros((0, 3)), "lines": []}
    hs = st.drag_handles() if ready else []
    r, made = sweep(prof, rl, **{k: (pts(*[(p.at, p.scale) for p in v]) if k == "scale_points" else v)
                                 for k, v in kw.items()})
    if not (ready and made):
        check("stage: " + name, False, f"{ready} {r}")
        continue
    P = np.vstack([pv["tris"][::7]] + [ln[::4] for ln in pv["lines"]])
    tips = [vd.ValueDrag._pos(h) for h in hs if h["kind"] == "arrow"]
    g_pv, g_tip = gap(made[0], P), gap(made[0], tips)
    size = float(np.linalg.norm(np.ptp(P, axis=0)))
    check("stage: " + name + ": preview and arrow tips on the sweep", g_pv < 2e-3 * size and g_tip < 2e-3 * size,
          f"preview {g_pv:.2e}, tips {g_tip:.2e} (size {size:.2f}), {len(tips)} arrows")
    h0 = hs[0]
    back = h0["to_value"](h0["param"])
    check("  the end arrow's length gives End Scale back", abs(back - st.scale_end) < 1e-9, f"{back}")

# adding a point: it starts at the scale already there (the shape does not change until it is dragged)
st = stage(circ, rail, scale_end=0.5)
st.drag_setup(ctx, None)
v = st._point_value(0.25)
check("stage: a new point starts at the scale there", abs(v - 0.875) < 1e-9, f"{v}")
st.scale_points.add().at = 0.25
st.scale_points[0].scale = v
check("  its fields: Scale 1 and Point 1 At", st.drag_fields == ("scale_end", "ss_0", "sa_0")
      and st._label("ss_0") == "Scale 1" and st._label("sa_0") == "Point 1 At", f"{st.drag_fields}")
st.ss_0, st.sa_0 = 1.3, 2.0
check("  dragging its fields sets the point (where kept inside the rail)",
      abs(st.scale_points[0].scale - 1.3) < 1e-9 and abs(st.scale_points[0].at - 0.995) < 1e-9)
h_at = [h for h in st.drag_handles() if h["field"] == "sa_0"][0]
f_ = h_at["pick"](np.array([5.0, 0.0, 1.5]), np.array([-1.0, 0.0, 0.0]), None)
check("  its diamond follows the mouse along the rail", abs(f_ - 0.5) < 0.02, f"{f_:.3f}")
# two rails: no stage (the rails set the scale), made at once
rail2 = poly("Rail2", [(1, 0, 0), (1, 0, 3)])
st2 = stage(circ, rail)
st2.inputs = ins((circ, "S0"), (rail, "S0"), (rail2, "S0"))
check("stage: two rails have none (made at once)", st2.drag_setup(ctx, None) is False)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
