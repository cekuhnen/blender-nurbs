"""Extrude from a surface face (features.build_extrude_face):
    blender -b --factory-startup --python tests/test_extrude_face.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, ops, store, subsel
from blender_nurbs.kernel import load
from blender_nurbs.pushpull import face_frame

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
ext = load().blender_ext

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def ins(*pairs):
    return json.dumps([[o.name, r] for o, r in pairs])

def vol(o):
    return abs(store.get_brep(o).volume(1e-6))

def solid(o):
    return bool(store.get_brep(o).is_manifold_solid())

def select(*objs):
    for o in ctx.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[-1] if objs else None

def prim(kind):
    bpy.ops.nproto.add_primitive(kind=kind)
    return ctx.active_object

def face_facing(o, n):
    """The face of o whose outward normal is n."""
    return next(i for i in range(store.get_brep(o).face_count()) if np.dot(face_frame(o, i)[2], n) > 0.999)

def extrude(*pairs, **kw):
    kw.setdefault("direction", "NORMAL")
    try:
        return bpy.ops.nproto.extrude(inputs=ins(*pairs), **kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

def hist_on(on):
    ctx.scene.nproto_settings.construction_history = on


# --- an open surface: a New Solid, the sheet kept as its input ------------------------
clear(); hist_on(True)
P = prim("PLANE")
r = extrude((P, "F0"), distance=0.5)
S = ctx.active_object
check("sheet: a closed solid of the face's area x distance", r == {'FINISHED'} and S is not P and solid(S)
      and abs(vol(S) - 2.0) < 1e-4, f"{vol(S) if S is not P else None}")
check("sheet: kept in NURBS History (hidden), the solid's input", features.is_kept(ctx, P)
      and not P.visible_get() and features.history(S)["op"] == "extrude_face")
h = features.history(S)
raw = json.loads(S[features.PROP_HIST]); raw["params"]["distance"] = 1.5; S[features.PROP_HIST] = json.dumps(raw)
features.rebuild(S, ctx, force=True)
check("history: a new distance rebuilds it", abs(vol(S) - 6.0) < 1e-4, f"{vol(S):.6f}")

clear(); hist_on(False)
P = prim("PLANE")
r = extrude((P, "F0"), distance=-0.5)
S = ctx.active_object
check("sheet, no history: the solid replaces the sheet; a negative distance goes the other way",
      r == {'FINISHED'} and "Plane" not in [o.name for o in bpy.data.objects] and abs(vol(S) - 2.0) < 1e-4
      and store.get_brep(S).bounding_box()[0][2] < -0.49, f"{[o.name for o in bpy.data.objects]}")

clear(); hist_on(False)
P = prim("PLANE")
r = extrude((P, "F0"), distance=0.5, direction="VECTOR", dir_vec=(1.0, 0.0, 1.0))
S = ctx.active_object
check("sheet along a slanted vector: area x distance x cos(lean)", r == {'FINISHED'}
      and abs(vol(S) - 4 * 0.5 * math.cos(math.radians(45))) < 1e-4, f"{vol(S):.6f}")

# --- a face of a solid: joined to its body, or cut from it ----------------------------
clear(); hist_on(True)
B = prim("BOX")
B_NAME = B.name
top = face_facing(B, (0, 0, 1))
r = extrude((B, f"F{top}"), distance=0.5)
R = bpy.data.objects.get(B_NAME)
check("solid's face: Body Surface means Join to its own body (8 + 4 x 0.5)", r == {'FINISHED'} and R is not None
      and solid(R) and abs(vol(R) - 10.0) < 1e-3, f"{vol(R) if R else None}")

clear(); hist_on(False)
B = prim("BOX")
B_NAME = B.name
top = face_facing(B, (0, 0, 1))
r = extrude((B, f"F{top}"), distance=-0.5, body="CUT")
R = bpy.data.objects.get(B_NAME)
check("solid's face, Cut: the slab taken off (8 - 4 x 0.5)", r == {'FINISHED'} and R is not None and solid(R)
      and abs(vol(R) - 6.0) < 1e-3, f"{vol(R) if R else None}")

clear(); hist_on(False)
B = prim("BOX")
B_NAME = B.name
top = face_facing(B, (0, 0, 1))
r = extrude((B, f"F{top}"), distance=0.5, draft=10.0, body="JOIN")
R = bpy.data.objects.get(B_NAME)
t = 1.0 + 0.5 * math.tan(math.radians(10.0))
frustum = 0.5 / 3.0 * (4.0 + (2 * t) ** 2 + 2.0 * 2 * t)
check("solid's flat face with Draft 10°: a frustum joined on", r == {'FINISHED'} and R is not None and solid(R)
      and abs(vol(R) - (8.0 + frustum)) < 1e-3, f"{vol(R) if R else None} vs {8 + frustum:.5f}")

clear(); hist_on(False)
B = prim("BOX")
side = face_facing(B, (1, 0, 0))
r = extrude((B, f"F{side}"), distance=1.0, body="NEW")
N = ctx.active_object
check("solid's face, New Solid: a separate body beside it, the box untouched", r == {'FINISHED'} and N is not B
      and abs(vol(N) - 4.0) < 1e-4 and abs(vol(B) - 8.0) < 1e-4
      and store.get_brep(N).bounding_box()[0][0] > 0.99 - 1e-9)

# --- a curved face ------------------------------------------------------------------
clear(); hist_on(False)
pts = [(x, y, 0.4 if (x == 0 and y == 0) else 0.0) for x in (-1, 0, 1) for y in (-1, 0, 1)]
bump = store.new_object(ctx, "Bump", ext.surface_from_cvs(2, 2, pts, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], None))
check("curved face: Face Normal is its average (straight up for a bump)",
      np.allclose(features.face_normal(bump, 0), (0, 0, 1), atol=1e-3), str(features.face_normal(bump, 0)))
r = extrude((bump, "F0"), distance=0.5, body="NEW")
S = ctx.active_object
check("curved face: the face, its moved copy and walls close (2 x 2 x 0.5)", r == {'FINISHED'} and solid(S)
      and abs(vol(S) - 2.0) < 1e-4 and store.get_brep(S).face_count() == 6, f"{vol(S):.6f}")
bump = store.new_object(ctx, "Bump2", ext.surface_from_cvs(2, 2, pts, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], None))
n = len(bpy.data.objects)
r = extrude((bump, "F0"), distance=0.5, draft=5.0, body="NEW")
check("curved face with a draft: refused, nothing made", r == {'CANCELLED'} and len(bpy.data.objects) == n)
r = extrude((bump, "F0"), distance=0.5, direction="X", body="NEW")
check("curved face swept along itself: refused", r == {'CANCELLED'} and len(bpy.data.objects) == n)
T = prim("CYLINDER")
n = len(bpy.data.objects)
r = extrude((T, "F0"), distance=0.5, body="NEW")
check("a tube's side (it wraps round): refused", r == {'CANCELLED'} and len(bpy.data.objects) == n)

# --- what a selection gives ------------------------------------------------------------
clear(); hist_on(False)
P = prim("PLANE")
B = prim("BOX")
cu = bpy.data.curves.new("C", 'CURVE'); cu.splines.new('POLY').points.add(1)
C = bpy.data.objects.new("C", cu); ctx.collection.objects.link(C)
select(P)
check("selection: an open surface gives its faces", ops._face_inputs(ctx) == [(P, "F0")])
select(B)
check("selection: a solid alone gives nothing (its face is picked)", ops._face_inputs(ctx) == [])
subsel.set_picks(B, "FACE", [2])
check("selection: faces picked in Face mode", ops._face_inputs(ctx) == [(B, "F2")])
select(P, C)
op = bpy.types.NPROTO_OT_extrude
check("selection: curves come first", ops._curve_inputs(ctx) == [(C, "S0")])

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
