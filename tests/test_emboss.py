"""Emboss / Deboss on solids that are not flat or cylindrical:
    blender -b --factory-startup --python tests/test_emboss.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import emboss, features, flow, store, unroll

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
sm, ext = store._k()
H, D = 0.08, 0.02

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)

def select(objs, active):
    for o in list(ctx.view_layer.objects):
        if o is not None:
            o.select_set(False)
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = active

def poly(name, pts, cyclic=True):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def circle(name, r, loc, rot=(0, 0, 0)):
    bpy.ops.curve.primitive_nurbs_circle_add(radius=r, location=loc, rotation=rot)
    o = ctx.active_object; o.name = name; return o

def run(op, *a, **kw):
    try:
        return op(*a, **kw)
    except RuntimeError as exc:
        print("   ", str(exc).strip()[:300])
        return {'CANCELLED'}

def vol(o): return abs(flow.world_brep(o).volume(1e-7))
def solid(o): return bool(flow.world_brep(o).is_manifold_solid())

def L_shape(name, cx, cy, z, s, axes=(0, 1, 2)):
    """An L (a concave corner) of area 0.2 s^2 in the plane of axes[0], axes[1] at axes[2] = z."""
    pts2 = [(-0.6, -0.5), (-0.1, -0.5), (-0.1, -0.35), (-0.45, -0.35), (-0.45, 0.5), (-0.6, 0.5)]
    out = []
    for x, y in pts2:
        p = [0.0, 0.0, 0.0]
        p[axes[0]], p[axes[1]], p[axes[2]] = cx + s * x, cy + s * y, z
        out.append(tuple(p))
    return poly(name, out)

L_AREA = 0.5 * 0.15 + 0.85 * 0.15            # in units of s^2

def ring(name, cx, cy, z, s, n=48):
    """An O of n-gon outline and hole (radii 0.25 s and 0.13 s): area in s^2."""
    def pts(r):
        return [(cx + 0.35 * s + r * s * math.cos(a), cy + r * s * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, n + 1)[:-1]]
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    for r in (0.25, 0.13):
        sp = cu.splines.new('POLY'); P = pts(r); sp.points.add(len(P) - 1)
        for p, c in zip(sp.points, P): p.co = (*c, 1.0)
        sp.use_cyclic_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    area = 0.5 * n * math.sin(2 * math.pi / n) * (0.25 ** 2 - 0.13 ** 2)
    return o, area

def dome_body(name="Dome"):
    """A doubly curved sheet z = 0.5 (1 - (x/2)^2) + 0.4 (1 - y^2) over 4 x 2,
    thickened 0.3 down into a solid."""
    secs = [sm.create_curve_interp_points([(float(x), float(y), float(0.5 * (1 - (x / 2) ** 2) + 0.4 * (1 - y * y)))
                                           for x in np.linspace(-2, 2, 9)]) for y in np.linspace(-1, 1, 5)]
    sheet = store.new_object(ctx, name + "Sheet", sm.create_skin_surface(secs))
    select([sheet], sheet)
    bpy.ops.nproto.thicken(distance=0.3)
    b = ctx.active_object
    b.name = name
    return b

def emboss_op(objs, body, **kw):
    select(objs + [body], body)
    t0 = time.perf_counter()
    kw = {"height": H, "overlap": D, **kw}
    r = run(bpy.ops.nproto.emboss, 'INVOKE_DEFAULT', **kw)
    return r, ctx.active_object, time.perf_counter() - t0

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"
    clear()

    # -- Project onto a doubly curved dome: walls along Z, top = surface + H ---------------
    body = dome_body()
    V0 = vol(body)
    s = 0.6
    L = L_shape("L", 0.6, 0.3, 3.0, s)
    O, oa = ring("O", 0.6, 0.3, 3.0, s)
    A = (L_AREA + oa) * s * s
    r, res, dt = emboss_op([L, O], body)
    good = r == {'FINISHED'} and store.is_nurbs(res)
    check(f"[{tag}] project emboss on the dome", good and solid(res), f"{dt * 1000:.0f} ms")
    if good:
        # the region between the surface and the surface moved up by H: exactly A H
        check(f"[{tag}] project emboss: volume + area x height", abs(vol(res) - V0 - A * H) < 2e-3 * A * H,
              f"{vol(res) - V0:.6f} vs {A * H:.6f}")
        check(f"[{tag}] project emboss: result kept apart", (res is not body) == hist)
    if hist and good:
        # the Feature panel's Height: the result follows
        h = features.history(res)
        prm = dict(h["params"]); prm["em_height"] = 2 * H
        raw = json.loads(res[features.PROP_HIST]); raw["params"] = prm
        res[features.PROP_HIST] = json.dumps(raw)
        features.rebuild(res, ctx, force=True)
        check(f"[{tag}] height edit rebuilds", abs(vol(res) - V0 - 2 * A * H) < 2e-3 * A * H,
              f"{vol(res) - V0:.6f} vs {2 * A * H:.6f}")
        check(f"[{tag}] inputs kept hidden", features.HISTORY_COLL in [c.name[:len(features.HISTORY_COLL)]
                                                                         for c in body.users_collection])
    # deboss: the same region cut away
    clear()
    body = dome_body()
    V0 = vol(body)
    L = L_shape("L", 0.6, 0.3, 3.0, s)
    O, oa = ring("O", 0.6, 0.3, 3.0, s)
    r, res, dt = emboss_op([L, O], body, operation="DEBOSS")
    good = r == {'FINISHED'} and store.is_nurbs(res)
    check(f"[{tag}] project deboss on the dome", good and solid(res)
          and abs(V0 - vol(res) - A * H) < 2e-3 * A * H, f"{V0 - vol(res) if good else 0:.6f} vs {A * H:.6f}")

    # -- Project onto the side of a box: trimmed flat faces, sketch beside it --------------
    clear()
    bpy.ops.nproto.add_primitive(kind="BOX", width=2.0, depth=1.0, height=1.0)
    box = ctx.active_object
    V0 = vol(box)
    lo, hi = unroll._box(flow.world_brep(box))
    L = L_shape("L", 0.5 * (lo[0] + hi[0]), 0.5 * (lo[2] + hi[2]), hi[1] + 2.0, 0.5, axes=(0, 2, 1))
    r, res, dt = emboss_op([L], box)
    good = r == {'FINISHED'} and store.is_nurbs(res)
    check(f"[{tag}] project onto the box's side face", good and solid(res)
          and abs(vol(res) - V0 - L_AREA * 0.25 * H) < 1e-3 * L_AREA * 0.25 * H,
          f"{vol(res) - V0 if good else 0:.6f} vs {L_AREA * 0.25 * H:.6f}")
    if good:
        rlo, rhi = unroll._box(flow.world_brep(res))
        check(f"[{tag}] box: stands out of the face it was thrown at", abs(rhi[1] - (hi[1] + H)) < 1e-6
              and abs(rlo[1] - lo[1]) < 1e-9, f"y {rlo[1]:.4f}..{rhi[1]:.4f}")

    # a word (Blender text as curves) onto the front face: its letters' caps share one plane that
    # runs past the face's top and bottom edges -- the box's top and bottom, along the lines,
    # must not catch it
    cu = bpy.data.curves.new("Word", 'FONT'); cu.body = "LOGO"; cu.size = 0.55
    cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
    wo = bpy.data.objects.new("Word", cu); ctx.collection.objects.link(wo)
    wo.location = (0.5 * (lo[0] + hi[0]), lo[1] - 1.5, 0.5 * (lo[2] + hi[2])); wo.rotation_euler = (math.pi / 2, 0, 0)
    select([wo], wo)
    bpy.ops.object.convert(target='CURVE')
    wo = ctx.active_object; wo.data.dimensions = '3D'
    prof = emboss.profile_inputs([wo])
    A = abs(emboss._prism(prof, np.array([0.0, 1.0, 0.0]), 0.0, 1.0).volume(1e-7))
    box2 = res if (good and not hist) else box
    V0 = vol(box2)
    r, res2, dt = emboss_op([wo], box2, operation="DEBOSS")
    good2 = r == {'FINISHED'} and store.is_nurbs(res2)
    check(f"[{tag}] a word debossed into the box's face", good2 and solid(res2)
          and abs(V0 - vol(res2) - A * H) < 2e-3 * A * H, f"{V0 - vol(res2) if good2 else 0:.6f} vs {A * H:.6f}")

    # -- Wrap: fold back from the body's unrolled sheet -------------------------------------
    clear()
    body = dome_body()
    V0 = vol(body)
    select([body], body)
    run(bpy.ops.nproto.unroll, place="BESIDE")
    bf = flow.world_brep(body).faces()
    top = max(range(len(bf)), key=lambda k: bf[k].centroid()[2])
    sheets = [o for o in ctx.selected_objects if unroll.PROP_UNROLL in o]
    flat, k = next((o, i) for o in sheets for i, (src, _) in enumerate(json.loads(o[unroll.PROP_UNROLL])["faces"])
                   if src == top)
    print(f"    {len(sheets)} sheet(s); the top face on {flat.name}")
    c = np.array(flow.world_brep(flat).faces()[k].centroid())
    L = L_shape("L", c[0], c[1], c[2], s)
    O = circle("Oo", 0.25 * s, (c[0] + 0.35 * s, c[1], c[2]))
    Oi = circle("Oi", 0.13 * s, (c[0] + 0.35 * s, c[1], c[2]))
    A = (L_AREA + math.pi * (0.25 ** 2 - 0.13 ** 2)) * s * s
    r, res, dt = emboss_op([L, O, Oi, flat], body)
    good = r == {'FINISHED'} and store.is_nurbs(res)
    check(f"[{tag}] wrap: folded back from the unrolled body", good and solid(res)
          and emboss.LAST_INFO.get("wrap") == "fold", f"{dt * 1000:.0f} ms {emboss.LAST_INFO}")
    if good:
        dv = vol(res) - V0
        check(f"[{tag}] wrap: outward, about area x height", 0.85 * A * H < dv < 1.15 * A * H,
              f"{dv:.6f} vs {A * H:.6f}")
    # Place: a small disc moved as a whole onto the sphere, outward
    clear()
    bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0)
    sph = ctx.active_object
    V0 = vol(sph)
    disc = circle("Disc", 0.1, (0.2, 0.1, 3.0))
    r, res, dt = emboss_op([disc], sph, mode="PLACE", overlap=0.01)
    good = r == {'FINISHED'} and store.is_nurbs(res)
    A = math.pi * 0.01
    check(f"[{tag}] place a disc on the sphere", good and solid(res)
          and 0.9 * A * H < vol(res) - V0 < 1.1 * A * H, f"{vol(res) - V0 if good else 0:.6f} vs {A * H:.6f}")
    # Project near the sphere's pole (a seed there must not hold Newton): exactly A H
    clear()
    bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0)
    sph = ctx.active_object
    V0 = vol(sph)
    disc = circle("Disc", 0.1, (0.2, 0.1, 3.0))
    r, res, dt = emboss_op([disc], sph)
    good = r == {'FINISHED'} and store.is_nurbs(res)
    check(f"[{tag}] project a disc near the sphere's pole", good and solid(res)
          and abs(vol(res) - V0 - A * H) < 2e-3 * A * H, f"{vol(res) - V0 if good else 0:.6f} vs {A * H:.6f}")
    # outlines that cross (kerned letters) are refused with a reason, the body left as it was
    clear()
    bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0)
    sph = ctx.active_object
    c1, c2 = circle("C1", 0.2, (-0.12, 0.0, 3.0)), circle("C2", 0.2, (0.12, 0.0, 3.0))
    r, res, dt = emboss_op([c1, c2], sph)
    check(f"[{tag}] crossing outlines refused", r == {'CANCELLED'} and store.get_brep(sph).face_count() == 1
          and emboss.crossing_outlines([features.input_curve(c1, "S0"), features.input_curve(c2, "S0")],
                                       np.array([0.0, 0.0, 1.0]), 0.6) == (0, 1))
    # an open profile is refused with a reason
    clear()
    bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0)
    sph = ctx.active_object
    ln = poly("Open", [(0, 0, 3), (0.2, 0, 3), (0.2, 0.2, 3)], cyclic=False)
    r, res, dt = emboss_op([ln], sph)
    check(f"[{tag}] open profile refused", r == {'CANCELLED'} and store.get_brep(sph).face_count() == 1)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
