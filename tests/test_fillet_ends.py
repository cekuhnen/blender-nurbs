"""Surface Fillet between surfaces that meet along only part of an edge
(Ends: Notch / Extend, kernel/filletends.py):
    blender -b --factory-startup --python tests/test_fillet_ends.py

A tent with one short side: a floor sheet 0.4 x 0.37 and a sloped sheet
meeting it edge to edge along its first 0.23 only."""
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
R = 0.03
LA, LB = 0.37, 0.23                       # floor and sloped sheet lengths along the crossing (Y)
SLOPE = np.array([0.15, 0.0, -0.16])
D1, D2 = np.array([-1.0, 0, 0]), SLOPE / np.linalg.norm(SLOPE)
ALPHA = math.acos(float(D1 @ D2))         # the angle inside the tent
SETBACK = R / math.tan(0.5 * ALPHA)       # the ball's rails from the crossing


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)


def poly(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1.0)
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o


def sheet(name, p0, p1, length):
    ln = poly(name + "_ln", [p0, p1])
    bpy.ops.nproto.extrude(inputs=json.dumps([[ln.name, "S0"]]), distance=length, direction="Y")
    o = ctx.active_object; o.name = name
    bpy.data.objects.remove(ln)
    return o


def tent():
    clear()
    a = sheet("Floor", (-0.4, 0, 0), (0, 0, 0), LA)
    b = sheet("Slope", (0, 0, 0), tuple(SLOPE), LB)
    return a, b


def A(o): return store.get_brep(o).area(1e-7)


def edge_pts(o):
    return np.array([e.point_at(t0 + (t1 - t0) * k / 16.0) for e in _world_brep(o).edges()
                     for t0, t1 in [e.parameter_range()] for k in range(17)])


def on_face(o, p): return min(f.closest_point(tuple(p))[1] for f in _world_brep(o).faces())
def on_edges(o, p): return min(e.closest_point(tuple(p))[2] for e in _world_brep(o).edges())


def fillet(a, b, **kw):
    kw.setdefault("radius", R)
    r = bpy.ops.nproto.fillet(surf_a=a.name, surf_b=b.name, **kw)
    return r, (ctx.active_object if r == {'FINISHED'} else None)


# --- Notch: the fillet stops where the slope ends, the floor is notched ----------
for shape in ("CIRCULAR", "CHAMFER", "BLEND"):
    a, b = tent()
    a0, b0 = A(a), A(b)
    r, fil = fillet(a, b, shape=shape)
    Y = edge_pts(fil)[:, 1] if fil else np.array([np.nan])
    fb = store.get_brep(a)
    check(f"notch {shape}: the fillet runs exactly along the shared 0.23 of the edge",
          r == {'FINISHED'} and abs(Y.min()) < 1e-6 and abs(Y.max() - LB) < 1e-6, (r, Y.min(), Y.max()))
    check(f"notch {shape}: the floor loses only the strip under the fillet, one face with the step (6 edges)",
          fb.face_count() == 1 and len(fb.edges()) == 6 and abs(A(a) - (a0 - SETBACK * LB)) < 1e-6,
          (fb.face_count(), len(fb.edges()), A(a), a0 - SETBACK * LB))
    check(f"notch {shape}: past the slope's end the floor keeps its own edge; the rail and the step are edges",
          on_edges(a, (0, 0.30, 0)) < 1e-7 and on_edges(a, (-SETBACK, 0.1, 0)) < 1e-7
          and on_edges(a, (-0.5 * SETBACK, LB, 0)) < 1e-7 and on_edges(a, (-0.5 * SETBACK, 0.1, 0)) > 6e-3)
    check(f"notch {shape}: the slope is cut back along its whole length",
          abs(A(b) - (b0 - SETBACK * LB)) < 1e-6 and store.get_brep(b).face_count() == 1, (A(b), b0 - SETBACK * LB))

# --- Extend: the slope grows to the floor's end first ---------------------------
a, b = tent()
a0, b0 = A(a), A(b)
r, fil = fillet(a, b, shape="CIRCULAR", ends="EXTEND")
Y = edge_pts(fil)[:, 1] if fil else np.array([np.nan])
check("extend: the fillet runs the floor's whole edge",
      r == {'FINISHED'} and abs(Y.min()) < 1e-6 and abs(Y.max() - LA) < 1e-6, (r, Y.min(), Y.max()))
check("extend: the slope is grown exactly to the floor's end (one natural face) and trimmed",
      store.get_brep(b).face_count() == 1 and abs(A(b) - (b0 / LB - SETBACK) * LA) < 1e-6
      and on_face(b, 0.5 * SLOPE + np.array([0, LA - 1e-4, 0])) < 1e-7
      and on_face(b, 0.5 * SLOPE + np.array([0, LA + 1e-3, 0])) > 5e-4, (A(b), (b0 / LB - SETBACK) * LA))
check("extend: the floor is cut back straight along its whole edge (4 edges)",
      len(store.get_brep(a).edges()) == 4 and abs(A(a) - (a0 - SETBACK * LA)) < 1e-6, (A(a), a0 - SETBACK * LA))

a, b = tent()
a0, b0 = A(a), A(b)
r, fil = fillet(a, b, shape="CIRCULAR", ends="EXTEND", trim_bases=False)
Y = edge_pts(fil)[:, 1] if fil else np.array([np.nan])
check("extend without trimming: the slope is grown, the floor untouched, the fillet runs the whole edge",
      r == {'FINISHED'} and abs(A(a) - a0) < 1e-9 and abs(A(b) - b0 / LB * LA) < 1e-6
      and abs(Y.max() - LA) < 1e-6, (A(a), A(b), Y.max()))

# --- Set Back: both edges moved back whole, a light Loft edge blend between ------
from blender_nurbs import contlabels
from blender_nurbs.kernel import load as _load


def cv_count(o):
    d = _load().blender_ext.surface_cvs(store.get_brep(o).faces()[0].surface())
    return d["count_u"] * d["count_v"]


for shape, cont, want in (("CIRCULAR", "2", "G2"), ("BLEND", "3", "G3"), ("BLEND", "1", "G1")):
    a, b = tent()
    a0, b0 = A(a), A(b)
    r, lo = fillet(a, b, shape=shape, continuity=cont, ends="SETBACK")
    labs = contlabels.labels(lo) if lo else []
    check(f"set back {shape} {want}: floor and slope each cut back by the setback along their WHOLE edges",
          r == {'FINISHED'} and abs(A(a) - (a0 - SETBACK * LA)) < 1e-6 and abs(A(b) - (b0 - SETBACK * LB)) < 1e-6
          and store.get_brep(a).face_count() == 1 and len(store.get_brep(a).edges()) == 4
          and store.get_brep(b).face_count() == 1, (r, A(a), a0 - SETBACK * LA, A(b), b0 - SETBACK * LB))
    E = edge_pts(lo) if lo else np.full((1, 3), np.nan)
    check(f"set back {shape} {want}: one light patch from the floor's whole edge to the slope's, labels hold {want}",
          lo is not None and store.get_brep(lo).face_count() == 1 and cv_count(lo) <= 48
          and abs(E[:, 1].max() - LA) < 1e-6 and len(labs) == 2 and all(L["check"]["ok"] for L in labs)
          and all(want in L["check"]["text"] or L.get("kind", want) == want for L in labs),
          (cv_count(lo) if lo else None, [L["check"]["text"] for L in labs]))

a, b = tent()
r, lo = fillet(a, b, shape="CHAMFER", ends="SETBACK")
check("set back chamfer: a straight blend", r == {'FINISHED'} and lo is not None
      and _load().blender_ext.surface_cvs(store.get_brep(lo).faces()[0].surface())["degree_v"] in (1,)
      or _load().blender_ext.surface_cvs(store.get_brep(lo).faces()[0].surface())["degree_u"] in (1,), r)

clear()
a = sheet("Plate", (-0.4, 0, 0), (0.4, 0, 0), 0.6)
fin_ln = poly("Fin_ln", [(0, 0.1, -0.1), (0, 0.1, 0.2)])
bpy.ops.nproto.extrude(inputs=json.dumps([[fin_ln.name, "S0"]]), distance=0.3, direction="Y")
fin = ctx.active_object
try:
    r = bpy.ops.nproto.fillet(surf_a=a.name, surf_b=fin.name, radius=0.02, ends="SETBACK")
except RuntimeError as exc:
    r = str(exc)
check("set back: surfaces that cross through each other are refused (no edges to move back)",
      r != {'FINISHED'} and "edge to edge" in str(r), str(r)[:120])

from blender_nurbs import filletsetback as _fsb
a, b = tent()
_smp = _fsb.preview_samples(a, b)
_X = _fsb.preview_sections(_smp, SETBACK, SETBACK, 5) if _smp else None
check("set back preview: sections from the floor's cut-back edge (whole length) to the slope's",
      _X is not None and np.allclose(_X[0, 0], [-SETBACK, 0, 0], atol=1e-6)
      and np.allclose(_X[-1, 0], [-SETBACK, LA, 0], atol=1e-6) and abs(_X[-1, -1][1] - LB) < 1e-6,
      None if _X is None else (_X[0, 0], _X[-1, 0], _X[-1, -1]))

# --- the preview follows Ends ---------------------------------------------------
from blender_nurbs.ops_trim import surface_fillet_frames, surface_fillet_uneven
a, b = tent()
fr_n = surface_fillet_frames(a, b, "NOTCH")
fr_e = surface_fillet_frames(a, b, "EXTEND")
yn = max(float(P[:, 1].max()) for P, _, _ in fr_n)
ye = max(float(P[:, 1].max()) for P, _, _ in fr_e)
check("preview: Notch along the shared part, Extend along the whole edge; E is offered (uneven ends)",
      abs(yn - LB) < 1e-6 and abs(ye - LA) < 1e-6 and surface_fillet_uneven(a, b), (yn, ye))
clear()
a = sheet("Floor", (-0.4, 0, 0), (0, 0, 0), LA)
b = sheet("Slope", (0, 0, 0), tuple(SLOPE), LA)
check("preview: equal sheets have no uneven ends (no E)", not surface_fillet_uneven(a, b))

# --- history: the fillet and both trims replay with the grown slope --------------
ctx.scene.nproto_settings.construction_history = True
a, b = tent()
r, fil = fillet(a, b, shape="CIRCULAR", ends="EXTEND")
before = [A(o) for o in (a, b, fil)]
st = features.stack(b)
for o in (fil, a, b):
    features.rebuild(o, ctx, force=True)
after = [A(o) for o in (a, b, fil)]
check("history: steps carry Ends, a rebuild gives the same three surfaces",
      st and st[-1]["op"] == "fillet_trim" and st[-1]["params"].get("ends") == "EXTEND"
      and max(abs(x - y) for x, y in zip(before, after)) < 1e-7, (before, after))

a, b = tent()
r, fil = fillet(a, b, shape="CIRCULAR")
before = [A(o) for o in (a, b, fil)]
for o in (fil, a, b):
    features.rebuild(o, ctx, force=True)
after = [A(o) for o in (a, b, fil)]
check("history: the notch replays", max(abs(x - y) for x, y in zip(before, after)) < 1e-7, (before, after))

a, b = tent()
r, lo = fillet(a, b, shape="CIRCULAR", ends="SETBACK")
before = [A(o) for o in (a, b, lo)]
for o in (a, b, lo):
    features.rebuild(o, ctx, force=True)
after = [A(o) for o in (a, b, lo)]
hidden = [o for o in bpy.data.objects if o.name.endswith("_setback")]
check("history: set back gives trim steps and a Loft that rebuild the same; the cut-back curves are kept hidden",
      features.stack(a)[-1]["op"] == "trim" and features.stack(b)[-1]["op"] == "trim"
      and features.history(lo)["op"] == "loft" and max(abs(x - y) for x, y in zip(before, after)) < 1e-7
      and len(hidden) == 2 and all(o.hide_get() for o in hidden), (before, after, [o.name for o in hidden]))
ctx.scene.nproto_settings.construction_history = False

# --- the crossing ending inside both: refused, not slotted ----------------------
clear()
a = sheet("Plate", (-0.4, 0, 0), (0.4, 0, 0), 0.6)               # y 0 .. 0.6
fin_ln = poly("Fin_ln", [(0, 0.1, -0.1), (0, 0.1, 0.2)])
bpy.ops.nproto.extrude(inputs=json.dumps([[fin_ln.name, "S0"]]), distance=0.3, direction="Y")
fin = ctx.active_object                                            # y 0.1 .. 0.4: ends inside the plate
pa0 = A(a)
r, fil = fillet(a, fin, shape="CIRCULAR", radius=0.02)
check("a fin ending inside the plate: the plate is not cut (no edge to notch back to)",
      abs(A(a) - pa0) < 1e-9, (r, A(a), pa0))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
