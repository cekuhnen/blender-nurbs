"""Surface Fillet with Radius A != Radius B (circular):
    blender -b --factory-startup --python tests/test_surface_fillet_two_radii.py

The kernel's circular fillet takes one radius (two fail in it); the worker
builds A != B from exact sections (secfillet.conic): the ball's centre sits
r_A from A and r_B from B, and between its feet the conic tangent to both.
At 90 deg that is the quarter ellipse about the centre with semi-axes the two
radii: a floor z = 0 and a wall x = 0 (the crossing the y axis), A = floor
0.1, B = wall 0.05 -> centre (0.05, y, 0.1), rails x = 0.05 on the floor and
z = 0.1 on the wall, every section on (x - 0.05)^2 / 0.05^2 + (z - 0.1)^2 /
0.1^2 = 1, and the fillet's area W times the quarter ellipse's length."""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, feature_ui
from blender_nurbs.sweeps import _world_brep

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
L, W = 0.5, 0.4


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


def A(o): return store.get_brep(o).area(1e-7)
def on_edges(o, p): return min(e.closest_point(tuple(p))[2] for e in _world_brep(o).edges())


def quarter_ellipse(a, b, n=200000):
    t = (np.arange(n) + 0.5) / n * 0.5 * math.pi
    return float(np.sum(np.hypot(a * np.sin(t), b * np.cos(t))) * 0.5 * math.pi / n)


def off_ellipse(fil, cx, cz, ax, az, k=13):
    """Largest |(x - cx)^2 / ax^2 + (z - cz)^2 / az^2 - 1| over a k x k grid of the fillet."""
    s = _world_brep(fil).faces()[0].surface()
    (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
    worst = 0.0
    for u in np.linspace(u0, u1, k):
        for v in np.linspace(v0, v1, k):
            x, y, z = s.evaluate(float(u), float(v))
            worst = max(worst, abs(((x - cx) / ax) ** 2 + ((z - cz) / az) ** 2 - 1.0))
    return worst


def fillet(a, b, ra, rb, **kw):
    try:
        r = bpy.ops.nproto.fillet(surf_a=a.name, surf_b=b.name, shape="CIRCULAR", radius=ra,
                                  same_radius=False, radius_b=rb, **kw)
    except RuntimeError as exc:
        return str(exc), None
    return r, (ctx.active_object if r == {'FINISHED'} else None)


ctx.scene.nproto_settings.construction_history = True
ys = np.linspace(0.0, W, 9)
RA, RB = 0.1, 0.05

for tag, fa, fw in (("sheets meeting edge to edge", ((0, 0, 0), (L, 0, 0)), ((0, 0, 0), (0, 0, L))),
                    ("sheets crossing", ((-L, 0, 0), (L, 0, 0)), ((0, 0, -L), (0, 0, L)))):
    clear()
    a, b = sheet("Floor", *fa), sheet("Wall", *fw)
    a0, b0 = A(a), A(b)
    r, fil = fillet(a, b, RA, RB)
    check(f"{tag}: Radius A 0.1, B 0.05 circular is made", r == {'FINISHED'} and fil is not None, r)
    if fil is None:
        continue
    # the side the fillet went to: its rails' signs
    pts = np.array([e.point_at(sum(e.parameter_range()) / 2) for e in _world_brep(fil).edges()])
    sx = float(np.sign(pts[np.argmax(np.abs(pts[:, 0])), 0]))
    sz = float(np.sign(pts[np.argmax(np.abs(pts[:, 2])), 2]))
    dev = max(max(on_edges(fil, (sx * RB, y, 0)), on_edges(fil, (0, y, sz * RA)),
                  on_edges(a, (sx * RB, y, 0)), on_edges(b, (0, y, sz * RA))) for y in ys)
    check(f"{tag}: rails at the ball's feet (floor x = B, wall z = A) on the fillet and both trimmed sheets",
          dev < 1e-7, dev)
    oe = off_ellipse(fil, sx * RB, sz * RA, RB, RA)
    check(f"{tag}: every section on the quarter ellipse about the centre, semi-axes B and A", oe < 1e-7, oe)
    q = quarter_ellipse(RB, RA)
    check(f"{tag}: fillet area W x the quarter ellipse's length", abs(A(fil) - W * q) < 1e-7, (A(fil), W * q))
    if tag.startswith("sheets meeting"):
        check(f"{tag}: the floor loses W x B, the wall W x A",
              abs(A(a) - (a0 - W * RB)) < 1e-6 and abs(A(b) - (b0 - W * RA)) < 1e-6, (A(a), a0 - W * RB, A(b)))
        before = (A(a), A(b), A(fil))
        for o in (fil, a, b):
            features.rebuild(o, ctx, force=True)
        check(f"{tag}: history replays it unchanged", np.allclose((A(a), A(b), A(fil)), before, atol=1e-9),
              ((A(a), A(b), A(fil)), before))

# --- the Feature panel: Radius A edited alone (A != B), the fillet and both trims follow
clear()
a, b = sheet("Floor", (0, 0, 0), (L, 0, 0)), sheet("Wall", (0, 0, 0), (0, 0, L))
a0 = A(a)
r = bpy.ops.nproto.fillet(surf_a=a.name, surf_b=b.name, shape="CIRCULAR", radius=0.1)
fil = ctx.active_object
feature_ui.sync_from_history(fil)
pg = fil.nproto_feature
pg.radius_a = math.copysign(0.06, pg.radius_a)          # B stays 0.1: centre (0.1, y, 0.06)
dev = max(max(on_edges(fil, (0.1, y, 0)), on_edges(fil, (0, y, 0.06)),
              on_edges(a, (0.1, y, 0)), on_edges(b, (0, y, 0.06))) for y in ys)
oe = off_ellipse(fil, 0.1, 0.06, 0.1, 0.06)
check("panel: Radius A alone set to 0.06 (B 0.1): the fillet and both trimmed sheets rebuild on the conic",
      r == {'FINISHED'} and dev < 1e-7 and oe < 1e-7 and abs(A(a) - (a0 - W * 0.1)) < 1e-6
      and abs(A(b) - (a0 - W * 0.06)) < 1e-6, (dev, oe, A(a), A(b)))

# --- the preview's sections: the same conic -------------------------------------
from blender_nurbs.ops_trim import surface_fillet_sections
P = np.array([[0.0, 0.2, 0.0]]); NA = np.array([[0.0, 0.0, 1.0]]); NB = np.array([[1.0, 0.0, 0.0]])
S = surface_fillet_sections(P, NA, NB, RA, RB, 1.0, 1.0, k=17)[0]
T1, T2 = S[0], S[-1]
cx, cz = (T1[0], T2[2]) if abs(T1[2]) < 1e-12 else (T2[0], T1[2])
ax, az = abs(cx), abs(cz)
e = max(abs(((p[0] - cx) / ax) ** 2 + ((p[2] - cz) / az) ** 2 - 1.0) for p in S)
check("preview: the sections are the same quarter ellipse (semi-axes A and B)",
      e < 1e-12 and sorted((ax, az)) == sorted((RA, RB)), (e, ax, az))
S = surface_fillet_sections(P, NA, NB, RA, RA, 1.0, 1.0, k=17)[0]
cx, cz = (S[0][0], S[-1][2]) if abs(S[0][2]) < 1e-12 else (S[-1][0], S[0][2])
e = max(abs(math.hypot(p[0] - cx, p[2] - cz) - RA) for p in S)
check("preview: equal radii stay the ball's arc", e < 1e-12, e)

# --- a chamfer's live preview: two arrows labelled Radius A / Radius B ----------
# (its A / B are the ball's radii, as for a circle: floor A 0.1, wall B 0.05 puts
# the floor's rail at x = 0.05 and the wall's at z = 0.1, the kernel's chamfer too)
from blender_nurbs import valuedrag as vd
from blender_nurbs.ops_trim import NPROTO_OT_fillet
clear()
a, b = sheet("Floor", (0, 0, 0), (L, 0, 0)), sheet("Wall", (0, 0, 0), (0, 0, L))
body = {k: v for k, v in NPROTO_OT_fillet.__dict__.items() if not k.startswith("__") and not k.startswith("bl_")}
op = type("PlainFillet", (vd.ValueDrag,), body)()
for k, v in dict(surf_a=a.name, surf_b=b.name, flip_a=False, flip_b=False, rail_type="RADIUS", same_radius=False,
                 radius=RA, radius_b=RB, shape="CHAMFER", continuity="2", ends="NOTCH", variable=False,
                 end_radius=0.0, _msg="").items():
    setattr(op, k, v)
op.drag_setup(ctx, None)
hs = op.drag_handles()
pos = {h["field"]: vd.ValueDrag._pos(h) for h in hs}
back = max(abs(h["to_value"](h["param"]) - getattr(op, h["field"])) for h in hs)
X = op.drag_preview()["lines"]
# each arrow on the rail its value moves: Radius A (the centre's height over the floor)
# the wall's rail z = A, Radius B the floor's rail x = B; dragging one 0.02 further
# out makes its value 0.02 bigger (at 90 deg), the other stays
on = (abs(pos["radius"][0]) + abs(pos["radius"][2] - RA) + abs(pos["radius_b"][0] - RB) + abs(pos["radius_b"][2])
      if set(pos) == {"radius", "radius_b"} else 1)
drag = max(abs(h["to_value"](h["param"] + 0.02) - (getattr(op, h["field"]) + 0.02)) for h in hs)
check("chamfer preview: two arrows labelled Radius A / Radius B, each on the rail its value moves",
      [op._label(f) for f in op.drag_fields] == ["Radius A", "Radius B"] and on < 1e-7 and back < 1e-9
      and drag < 1e-9 and len(X), ([op._label(f) for f in op.drag_fields], pos, back, drag))
X = surface_fillet_sections(*op._sf[0], *op._radii(*op._sf[0], 0), *op._sides(), "CHAMFER")
ends = max(max(abs(q[0] - RB) + abs(q[2]) for q in X[:, 0]), max(abs(q[0]) + abs(q[2] - RA) for q in X[:, -1]))
check("chamfer preview: the straight sections run from the floor's rail x = B to the wall's z = A "
      "(where the kernel's chamfer puts them)", ends < 1e-7, ends)
op.same_radius = True
check("chamfer preview: Same Radius, one arrow labelled Radius",
      [op._label(f) for f in op.drag_fields] == ["Radius"] and len(op.drag_handles()) == 1)
op.variable = True
check("chamfer preview: Variable, Start / End Radius",
      [op._label(f) for f in op.drag_fields] == ["Start Radius", "End Radius"])

ctx.scene.nproto_settings.construction_history = False
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
