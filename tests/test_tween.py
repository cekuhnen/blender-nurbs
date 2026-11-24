"""Tween:  blender -b --factory-startup --python tests/test_tween.py"""
import json, math, os, sys
import bpy
import numpy as np
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, geom, pattern, store, tween

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def W(o):
    return np.array(geom.world_matrix(o), float)

def sim(axis, deg, s, t):
    M = np.eye(4)
    M[:3, :3] = s * tween._rodrigues(np.asarray(axis, float) / np.linalg.norm(axis), math.radians(deg))
    M[:3, 3] = t
    return M

# -- the maths ---------------------------------------------------------------------
check("complex expm1", abs(np.expm1(complex(0.0, math.pi)) - (-2.0)) < 1e-12)
Wa = sim((0.3, -0.2, 1.0), 25.0, 1.3, (1.0, 2.0, 3.0))
for name, Wb in (("move", sim((0, 0, 1), 0.0, 1.0, (4, 0, 0)) @ Wa),
                 ("turn about an offset axis", sim((0, 0, 1), 90.0, 1.0, (5, -5, 0)) @ Wa),
                 ("screw", sim((1, 1, 0), 120.0, 1.0, (2, 2, 3)) @ Wa),
                 ("spiral", sim((0, 1, 0), 60.0, 0.6, (3, 0, 1)) @ Wa),
                 ("half turn", sim((0, 0, 1), 180.0, 1.0, (2, 0, 0)) @ Wa),
                 ("grow only", sim((0, 0, 1), 0.0, 2.0, (0, 0, 0)) @ Wa)):
    c = np.array([0.5, -1.0, 2.0])
    T1 = tween.step_transform(Wa, Wb, 1.0, c)
    check(f"{name}: one step lands on B", np.allclose(T1 @ Wa, Wb, atol=1e-9))
    Th = tween.step_transform(Wa, Wb, 0.5, c)
    check(f"{name}: two half steps make one", np.allclose(Th @ Th, T1, atol=1e-9))
    T3 = tween.step_transform(Wa, Wb, 1.0 / 3.0, c)
    check(f"{name}: three thirds make one, steps equal", np.allclose(T3 @ T3 @ T3, T1, atol=1e-9))
    T2 = tween.step_transform(Wa, Wb, 2.0, c)
    check(f"{name}: continue = the step again", np.allclose(T2, T1 @ T1, atol=1e-9))
    check(f"{name}: the centre point does not matter", np.allclose(
        tween.step_transform(Wa, Wb, 0.37, np.zeros(3)), tween.step_transform(Wa, Wb, 0.37, c), atol=1e-9))
# the turn about an offset axis: a midway copy lies on the arc, not on the chord
Wb = sim((0, 0, 1), 90.0, 1.0, (0, 0, 0))
Wa0 = sim((0, 0, 1), 0.0, 1.0, (2, 0, 0))
Wb0 = Wb @ Wa0
Th = tween.step_transform(Wa0, Wb0, 0.5, (2, 0, 0)) @ Wa0
check("half of a quarter turn about the origin: on the arc", np.allclose(Th[:3, 3], [math.sqrt(2), math.sqrt(2), 0]),
      Th[:3, 3])
Tl = tween.step_transform(Wa0, Wb0, 0.5, (2, 0, 0), long_way=True) @ Wa0
check("long way: the other side of the circle", np.allclose(Tl[:3, 3], [-math.sqrt(2), -math.sqrt(2), 0]),
      Tl[:3, 3])
check("long way still lands on B", np.allclose(tween.step_transform(Wa0, Wb0, 1.0, (2, 0, 0), True) @ Wa0, Wb0))
# a stretch: lands on B, no step consistency promised
St = np.diag([2.0, 1.0, 1.0, 1.0])
check("stretch lands on B", np.allclose(tween.step_transform(np.eye(4), St, 1.0, np.zeros(3)), St))
check("stretch half way: √2 along X", np.allclose(tween.step_transform(np.eye(4), St, 0.5, np.zeros(3))[:3, :3],
                                                   np.diag([math.sqrt(2), 1, 1])))
try:
    tween.step_transform(np.eye(4), np.diag([-1.0, 1, 1, 1]), 0.5, np.zeros(3))
    check("mirror refused", False)
except ValueError:
    check("mirror refused", True)
check("alphas", np.allclose(tween.alphas(3, 2), [0.25, 0.5, 0.75, 1.25, 1.5]))
check("alphas: none between, continue only", np.allclose(tween.alphas(0, 3), [2, 3, 4]))

def select(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[-1]

# -- meshes: plain duplicates --------------------------------------------------------
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(2, 0, 0))
ma = ctx.active_object
mb = ma.copy()
mb.data = ma.data.copy()
ctx.collection.objects.link(mb)
mb.matrix_world = Matrix(sim((0, 0, 1), 90.0, 0.5, (0, 0, 0)).tolist()) @ ma.matrix_world
ctx.view_layer.update()
n0 = len(bpy.data.objects)
select(ma, mb)
r = bpy.ops.nproto.tween(count=3)
ctx.view_layer.update()
new = [o for o in bpy.data.objects if o not in (ma, mb)]
check("mesh tween: 3 copies", r == {'FINISHED'} and len(new) == 3, len(new))
rad = sorted(round(float(np.linalg.norm(W(o)[:2, 3])), 5) for o in new)
want = sorted(round(2.0 * 0.5 ** (k / 4), 5) for k in (1, 2, 3))
check("mesh tween: spiral in (radius ×0.5^(k/4))", np.allclose(rad, want), (rad, want))
sc = sorted(round(float(np.linalg.norm(W(o)[:3, 0])), 5) for o in new)
check("mesh tween: sizes step evenly (×0.5^(k/4))", np.allclose(sc, sorted(0.5 ** (k / 4) for k in (1, 2, 3)),
                                                                 atol=1e-5), sc)
check("mesh tween: active stays B, all selected", ctx.view_layer.objects.active is mb
      and all(o.select_get() for o in new + [ma, mb]))
for o in new:
    bpy.data.objects.remove(o)
select(ma, mb)
check("one object: refused", (lambda: (ma.select_set(False), bpy.ops.nproto.tween())[1])() == {'CANCELLED'})

# -- NURBS: a Pattern that follows B ---------------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.add_primitive(kind="BOX", width=1.0, depth=0.5, height=0.4)
na = ctx.active_object
na.location = (3.0, 0.0, 0.0)
ctx.view_layer.update()
select(na)
bpy.ops.nproto.move(base=(0, 0, 0), offset=(0, 0, 0.001), copies="ARRAY", count=1)    # B: a copy of A
nb = next(o for o in bpy.data.objects if o.get("nproto_in_0") is na)
nb.matrix_world = Matrix(sim((0, 0, 1), 120.0, 1.0, (0, 0, 2.0)).tolist()) @ na.matrix_world
ctx.view_layer.update()
select(na, nb)
r = bpy.ops.nproto.tween(count=2, extra=1)
cps = sorted([(o, k) for o, _s, k in pattern.group_copies(
    next(pattern.pattern_of(o)[1]["group"] for o in bpy.data.objects
         if pattern.pattern_of(o) and pattern.pattern_of(o)[1].get("mode") == "TWEEN"))], key=lambda c: c[1])
check("nurbs tween: 3 pattern copies", r == {'FINISHED'} and len(cps) == 3, len(cps))
p = pattern.pattern_of(cps[0][0])[1]
check("nurbs tween: B is the history's second input", cps[0][0].get("nproto_in_1") is nb and p["mode"] == "TWEEN")
check("nurbs tween summary", pattern.summary(p) == "Tween · 2 between · 1 past", pattern.summary(p))
Wa_, Wb_ = W(na), W(nb)
for (o, k), x in zip(cps, (1 / 3, 2 / 3, 4 / 3)):
    want = tween.step_transform(Wa_, Wb_, x, pattern.anchor(na)) @ Wa_
    check(f"nurbs copy {k} placed at α={x:.3f}", np.allclose(W(o), want, atol=1e-6))
check("nurbs copies carry the b-rep", all(store.is_nurbs(o) for o, _k in cps))
if features.history_enabled(ctx):
    nb.matrix_world = Matrix(sim((0, 0, 1), 60.0, 1.0, (0, 0, 1.0)).tolist()) @ na.matrix_world
    ctx.view_layer.update()
    for o, _k in cps:
        features.rebuild(o, ctx)
    ctx.view_layer.update()
    o1 = cps[0][0]
    want = tween.step_transform(W(na), W(nb), 1 / 3, pattern.anchor(na)) @ W(na)
    P = np.array([tuple(o1.matrix_world @ v.co) for v in o1.data.vertices])
    Pa = np.array([tuple(na.matrix_world @ v.co) for v in na.data.vertices])
    Pw = Pa @ (want @ np.linalg.inv(W(na)))[:3, :3].T + (want @ np.linalg.inv(W(na)))[:3, 3]
    check("B moved: the copies re-tween (history)", np.allclose(P.min(axis=0), Pw.min(axis=0), atol=1e-5)
          and np.allclose(P.max(axis=0), Pw.max(axis=0), atol=1e-5), (P.min(axis=0), Pw.min(axis=0)))
# Edit Pattern: the tween operator on the group
grp = p["group"]
check("edit pattern opens Tween", pattern._edit_op(p) == "nproto.tween")
r = bpy.ops.nproto.tween(group=grp, count=4, extra=0)
check("edit tween: 4 copies now", r == {'FINISHED'} and len(pattern.group_copies(grp)) == 4,
      len(pattern.group_copies(grp)))
select(nb, na)
r = bpy.ops.nproto.tween(count=1, swap=True)
check("swap: from the active one... back to the other", r == {'FINISHED'})

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
