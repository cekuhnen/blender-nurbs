"""Twist / Bend / Taper (deform.py):
    blender -b --factory-startup --python tests/test_deform.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import deform, features, store, surfeval
from blender_nurbs.unroll import inside_uvs

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def curve(name, loops, cyclic=True):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    for pts in loops:
        sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
        for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
        sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def select(*objs):
    for x in list(ctx.selected_objects): x.select_set(False)
    for o in objs: o.select_set(True)
    ctx.view_layer.objects.active = objs[-1]

def history(on):
    ctx.scene.nproto_settings.construction_history = on

L = 4.0
def bar(name="Bar", z0=-0.1, h=0.2, w=0.3):
    R = curve(name + "_p", [[(0, -w, z0), (L, -w, z0), (L, w, z0), (0, w, z0)]])
    old = set(bpy.data.objects)
    bpy.ops.nproto.extrude(inputs=json.dumps([[R.name, "S0"]]), distance=h, direction="Z", body="NEW")
    o = next(x for x in bpy.data.objects if x not in old and store.is_nurbs(x))
    if features.has_history(o):
        features.bake(o)
    if R.name in bpy.data.objects:
        bpy.data.objects.remove(R)
    if features.has_history(o):
        features.bake(o)
    return o

def vol(o): return abs(store.get_brep(o).volume(1e-6))
def solid(o): return bool(store.get_brep(o).is_manifold_solid())

def samples(b, n=9):
    out = []
    for f in b.faces():
        S = surfeval.Surf(f.surface())
        uv = inside_uvs(f, S, n)
        if len(uv):
            out.append(S.at(uv[:, 0], uv[:, 1], 0)[0, 0])
    return np.vstack(out)

def miss(o, P):
    """Largest distance of local points P from o's face surfaces."""
    best = np.full(len(P), np.inf)
    for f in store.get_brep(o).faces():
        S = surfeval.Surf(f.surface())
        uv = S.feet(P)
        best = np.minimum(best, np.linalg.norm(S.at(uv[:, 0], uv[:, 1], 0)[0, 0] - P, axis=1))
    return float(best.max())

def run(**kw):
    try:
        return bpy.ops.nproto.deform(**kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

def expect_map(o, kw, before_text):
    """The deformed object o against the map applied to samples of its old geometry."""
    b0 = store.brep_from_text(before_text)
    p = {"deform_kind": kw.get("kind", "TWIST"), "deform_angle": kw.get("angle", 90.0),
         "deform_factor": kw.get("factor", 0.5), "deform_axis": kw.get("axis", "AUTO"),
         "deform_direction": kw.get("direction", 0.0), "deform_smooth": kw.get("smooth", False),
         "deform_infinite": kw.get("infinite", False), "deform_flat": kw.get("flat", False)}
    fm = deform.map_for(deform.points_of_brep(b0), p)
    return miss(o, fm.map(samples(b0))) / fm.size


# --- the frame: Auto runs along the longest extent, bends across the thinnest
clear()
o = bar()
fr = deform.frame_for(deform.points_of_brep(store.get_brep(o)), {"deform_axis": "AUTO"})
check("auto frame: along X from x=0, bends along Z (thinnest), through the middle",
      np.allclose(fr.z, [1, 0, 0]) and abs(abs(fr.x[2]) - 1) < 1e-9 and abs(fr.L - L) < 1e-9
      and np.allclose(fr.O, [0, 0, 0], atol=1e-9), (fr.O.round(4), fr.z, fr.x, fr.L))

# --- the crop: a trim box much smaller than its surface, cut exactly
from blender_nurbs import surfmap
_, ext = store._k()
S0 = surfeval.Surf(ext.surface_from_cvs(3, 3, [(i, j, math.sin(i) * math.cos(j)) for i in range(6) for j in range(6)],
                                        [0, 0, 0, 0, 1, 2, 3, 3, 3, 3], [0, 0, 0, 0, 1, 2, 3, 3, 3, 3], None))
Sc = deform.crop_to_trim(None, S0, pad=0.0, box=((0.6, 1.2), (1.7, 2.5)))
uu, vv = np.meshgrid(np.linspace(0.6, 1.7, 7), np.linspace(1.2, 2.5, 7))
d = np.abs(Sc.at(uu.ravel(), vv.ravel(), 0)[0, 0] - S0.at(uu.ravel(), vv.ravel(), 0)[0, 0]).max()
check("crop: the cut-down surface is the same surface over the same (u, v)",
      d < 1e-12 and Sc.u_domain == (0.6, 1.7) and Sc.v_domain == (1.2, 2.5), (d, Sc.u_domain, Sc.v_domain, Sc.nu, Sc.nv))

# --- Twist (no history): exact volume, closed, on the true map
history(False)
for kw, vexp in ((dict(kind="TWIST", angle=180.0), 1.0), (dict(kind="TWIST", angle=720.0), 1.0),
                 (dict(kind="BEND", angle=90.0), 1.0), (dict(kind="BEND", angle=300.0), 1.0),
                 (dict(kind="TAPER", factor=0.5), 1 - 0.5 + 0.25 / 3),
                 (dict(kind="TAPER", factor=0.5, flat=True), 0.75),
                 (dict(kind="TWIST", angle=180.0, smooth=True), 1.0),
                 (dict(kind="BEND", angle=120.0, smooth=True), 1.0)):
    clear()
    o = bar()
    v0, t0 = vol(o), o[store.PROP_BREP]
    select(o)
    tt = time.time()
    r = run(**kw)
    dt = time.time() - tt
    m = expect_map(o, kw, t0) if r == {'FINISHED'} else None
    check(f"{kw}: closed, volume x{vexp:.4f}, on the map",
          r == {'FINISHED'} and solid(o) and abs(vol(o) / v0 - vexp) < 1e-4 and m < 1e-5,
          f"vol x{vol(o) / v0:.5f} miss {m} {dt:.2f}s" if r == {'FINISHED'} else r)

# a bent bar's end turns by the angle; its far end face's centre lands on the arc
clear()
o = bar()
select(o)
run(kind="BEND", angle=90.0)
P = samples(store.get_brep(o), 5)
R = 2 * L / math.pi
check("bend 90: a quarter circle of radius 2L/pi about (0, 2L/pi): the far end stands at z = R",
      abs(P[:, 2].max() - R) < 1e-3 and R - 0.1 < P[:, 0].max() < R + 0.1
      and np.abs(np.hypot(P[:, 0], P[:, 2] - R) - R).max() < 0.1 + 1e-4,
      (P[:, 0].max(), P[:, 2].max(), R))

# --- a part with holes (boolean): trimmed faces cropped and refitted, still one solid
clear()
o = bar("Slot")
for i, x0 in enumerate((0.6, 1.6, 2.6)):
    c = curve(f"h{i}", [[(x0 + 0.15 * math.cos(t), 0.12 * math.sin(t), -0.3)
                         for t in np.linspace(0, 2 * math.pi, 25)[:-1]]])
    old = set(bpy.data.objects)
    bpy.ops.nproto.extrude(inputs=json.dumps([[c.name, "S0"]]), distance=0.6, direction="Z", body="NEW")
    tool = next(x for x in bpy.data.objects if x not in old and store.is_nurbs(x))
    features.bake(tool)
    if c.name in bpy.data.objects:
        bpy.data.objects.remove(c)
    bpy.ops.nproto.boolean(operation="DIFFERENCE", keep_tools=False, names=json.dumps([o.name, tool.name]))
    o = ctx.active_object
if features.has_history(o):
    features.bake(o)
v0, nf, t0 = vol(o), store.get_brep(o).face_count(), o[store.PROP_BREP]
select(o)
tt = time.time()
r = run(kind="TWIST", angle=180.0)
dt = time.time() - tt
check("twisted part with 3 holes: one closed solid, same faces, volume kept, on the map",
      r == {'FINISHED'} and solid(o) and store.get_brep(o).face_count() == nf and abs(vol(o) / v0 - 1) < 1e-4
      and expect_map(o, dict(kind="TWIST", angle=180.0), t0) < 1e-5,
      f"{nf} faces, vol x{vol(o) / v0:.5f}, {dt:.2f}s")
cvs = sum(surfeval.Surf(f.surface()).nu * surfeval.Surf(f.surface()).nv for f in store.get_brep(o).faces())
check("cropped faces keep the CV count moderate", cvs < 60 * nf, cvs)

# --- the fold check: a bend tighter than the part is thick is refused
clear()
o = bar(h=2.4, z0=-1.2, w=1.0)       # thinnest across Y: 2 wide, bent at radius 4 / 5.9 < 1
t0 = o[store.PROP_BREP]
select(o)
r = run(kind="BEND", angle=340.0)
check("bend tighter than the part: refused, the part unchanged", r == {'CANCELLED'} and o[store.PROP_BREP] == t0)
r = run(kind="TAPER", factor=-0.2)
check("taper through zero: refused", r == {'CANCELLED'} and o[store.PROP_BREP] == t0)

# --- history: an in-place step, editable, and a live axis line
history(True)
clear()
o = bar()
v0, t_plain = vol(o), o[store.PROP_BREP]
select(o)
r = run(kind="BEND", angle=90.0)
st = features.stack(o)
check("history: a deform step on the stack, labelled", r == {'FINISHED'} and st and st[-1]["op"] == "deform",
      [deform.step_label(e["params"]) for e in st])
raw = json.loads(o[features.PROP_STACK])
raw[-1]["params"]["deform_angle"] = 180.0
o[features.PROP_STACK] = json.dumps(raw)
features.rebuild(o, ctx, force=True)
P = samples(store.get_brep(o), 5)
check("history: the angle edited to 180 -- the bar is bent into a U (its far end back over x < L/pi)",
      solid(o) and abs(vol(o) / v0 - 1) < 1e-4 and P[:, 0].max() < L / math.pi + 0.2 and P[:, 2].max() > 2 * L / math.pi,
      (P[:, 0].max(), P[:, 2].max()))
from blender_nurbs import history_ui
history_ui.sync_step(o, len(features.stack(o)) - 1)
pg = o.nproto_step
t_before = o[store.PROP_BREP]
pg.deform_angle = 60.0                       # the History panel's field: edits the step and rebuilds
check("History panel: the step's fields show it, an edit rebuilds",
      pg.deform_kind == "BEND" and features.stack(o)[-1]["params"]["deform_angle"] == 60.0
      and o[store.PROP_BREP] != t_before and solid(o) and not pg.error, (pg.deform_kind, pg.error))
raw = json.loads(o[features.PROP_STACK])
raw[-1]["params"]["deform_angle"] = 90.0
raw[-1]["params"]["deform_kind"] = "TWIST"
o[features.PROP_STACK] = json.dumps(raw)
features.rebuild(o, ctx, force=True)
check("history: switched to Twist", solid(o) and abs(vol(o) / v0 - 1) < 1e-4 and "Twist" in deform.step_label(raw[-1]["params"]))
select(o)
bpy.ops.nproto.remove_step(index=len(features.stack(o)) - 1)
check("history: the step removed -- the plain bar again", o[store.PROP_BREP] == t_plain
      and not features.has_history(o), (vol(o) / v0, features.has_history(o)))

clear()
o = bar()
line = curve("Axis", [[(1.0, 0, 0), (3.0, 0, 0)]], cyclic=False)
select(o, line)
r = run(kind="TWIST", angle=90.0)
st = features.stack(o)
check("line axis: the line selected last is the axis (a live step input)",
      r == {'FINISHED'} and st and st[-1]["params"]["deform_axis"] == "LINE" and st[-1]["inputs"]
      and st[-1]["inputs"][0][0] == line and line.name in bpy.data.objects)
b = store.get_brep(o)
P = samples(b, 7)
end = P[P[:, 0] > 3.2]
check("line axis: twisted only between x=1 and x=3 (the end beyond is turned 90 degrees as a whole)",
      len(end) and np.abs(end[:, 1]).max() < 0.1 + 1e-4 and np.abs(end[:, 2]).max() > 0.29, (np.abs(end).max(axis=0)))
t1 = o[store.PROP_BREP]
line.location.x += 0.5
ctx.view_layer.update()
features.rebuild(o, ctx)
check("line axis: moving the line re-twists the part", o[store.PROP_BREP] != t1 and solid(o))

# --- a single untrimmed surface stays untrimmed
history(False)
clear()
bpy.ops.nproto.add_primitive(kind="PLANE")
pl = ctx.active_object
pl.scale = (3.0, 0.5, 1.0)
bpy.ops.object.transform_apply(scale=True)
nat = store.is_natural(pl)
select(pl)
r = run(kind="BEND", angle=90.0, axis="X")
check("plane bent: still one untrimmed surface", r == {'FINISHED'} and store.get_brep(pl).face_count() == 1
      and store.is_natural(pl) == nat, (r, nat, store.is_natural(pl)))

# --- curves: a straight line off the axis twists into a helix-like curve
clear()
c = curve("Wire", [[(x, 0.3, 0.0) for x in np.linspace(0, L, 9)]], cyclic=False)
ref = curve("Ref", [[(0, 0, 0), (L, 0, 0)]], cyclic=False)           # the axis: a line selected last
select(c, ref)
r = run(kind="TWIST", angle=360.0)
from blender_nurbs import curvejoin, curvetrim
Pc, _ = curvetrim.sample_spans(curvejoin.spline_spans(c.data.splines[0]), 32)
Pc = np.asarray(Pc)
rad = np.hypot(Pc[:, 1], Pc[:, 2])
check("curve twisted: stays at its distance from the axis and turns round it",
      r == {'FINISHED'} and np.abs(rad - 0.3).max() < 1e-4 and Pc[:, 2].min() < -0.29 and Pc[:, 2].max() > 0.29,
      (np.abs(rad - 0.3).max(), Pc[:, 2].min(), Pc[:, 2].max()))

print("ALL OK" if ok else "SOME FAILED")
