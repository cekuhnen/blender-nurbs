"""Faces with edges shorter than ~0.02 model units join (stitch) with their
neighbours: each tool that trims with ext.trim_surface_with_3d_curves_tol gives
the face a tolerance that follows the part's size. With the kernel's default of
0.01, Join merged a face's vertices closer than about twice that and paired the
wrong edges (or none). blender -b --factory-startup --python tests/test_small_feature_join.py"""
import json, math, os, sys
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import geom, store

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
sm, ext = store._k()
ctx.scene.nproto_settings.construction_history = False
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def poly(name, pts, cyclic=True):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])
def line(a, b): return ext.curve_from_cvs(1, [tuple(map(float, a)), tuple(map(float, b))], [0, 0, 1, 1], None)
def loop(P): return [line(P[i], P[(i + 1) % len(P)]) for i in range(len(P))]

def ngon(c, r, n=48, rev=False):
    """A 48-gon of radius 0.08: edges 0.0105 long."""
    P = [(c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2]) for a in np.linspace(0, 2 * math.pi, n + 1)[:-1]]
    return P[::-1] if rev else P

def plane(x0, y0, x1, y1, z=0.0):
    return ext.surface_from_cvs(1, 1, [(x, y, z) for x in (x0, x1) for y in (y0, y1)], [0, 0, 1, 1], [0, 0, 1, 1])

def square(cx, h):
    return [(cx - h, -h, 0), (cx + h, -h, 0), (cx + h, h, 0), (cx - h, h, 0)]

def holed(name, cx):
    """A 0.4 square plane at x = cx with a 48-gon hole, trimmed tightly (the input is not what is tested)."""
    return store.new_object(ctx, name, ext.trim_surface_with_3d_curves_tol(
        plane(cx - 0.2, -0.2, cx + 0.2, 0.2), [4, 48], loop(square(cx, 0.2)) + loop(ngon((cx, 0, 0), 0.08, rev=True)),
        1e-6, 1e-5))

def plug(name, cx):
    return store.new_object(ctx, name, ext.trim_surface_with_3d_curves_tol(
        plane(cx - 0.2, -0.2, cx + 0.2, 0.2), [48], loop(ngon((cx, 0, 0), 0.08)), 1e-6, 1e-5))

def stats(b):
    """(shared edges, open edges, open length)."""
    op = [e for e in b.edges() if len(e.faces()) == 1]
    return sum(1 for e in b.edges() if len(e.faces()) == 2), len(op), sum(e.length() for e in op)

def join(objs):
    """Join with the last object active; (ok, joined brep)."""
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    ctx.view_layer.objects.active = objs[-1]
    try:
        r = bpy.ops.nproto.join()
    except RuntimeError as exc:
        print("  join:", str(exc).strip().splitlines()[-1])
        return False, None
    return r == {'FINISHED'}, store.get_brep(ctx.active_object)

def edge_refs(o, key):
    E = store.display(o)["edges"]
    return [f"E{i}" for i, pl in enumerate(E) if key(np.array(pl[len(pl) // 2], float))]

# --- Patch (patch.trim_to_loop): a pentagon opening with a 0.014 side, joined to its five walls
P = [(0, 0, 0), (0.3, 0, 0), (0.3, 0.2, 0), (0.01, 0.2, 0), (0, 0.19, 0)]
walls = []
for i in range(5):
    ln = poly(f"PatchSide{i}", [P[i], P[(i + 1) % 5]], cyclic=False)
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=0.1, direction="Z"); walls.append(ctx.active_object)
top = [(w, edge_refs(w, lambda m: m[2] > 0.09)[0]) for w in walls]
r = bpy.ops.nproto.patch(inputs=ins(*top), default_cont="G0")
rj, jb = join(walls + [ctx.active_object])
s = stats(jb) if rj else None
check("patch: a pentagon with a 0.014 side joins to its walls (5 seams + 5 rim edges stitched)",
      r == {'FINISHED'} and rj and s[:2] == (10, 5), f"(shared, open) {s and s[:2]} (10, 5)")

# --- Dome, draped (dome.draped_dome): safe with 0.01 -- one closed rim edge, one vertex -- kept as a guard
out = []
for k, first in enumerate(("wall", "dome")):
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.1); dc = ctx.active_object
    dc.location = (1, 0.5 * k, 0); ctx.view_layer.update()
    bpy.ops.nproto.extrude(inputs=ins((dc, "S0")), distance=0.01, direction="Z"); dw = ctx.active_object
    rim = max(("U0", "U1", "V0", "V1"),
              key=lambda w: np.mean([q[2] for q in geom.boundary_points(store.get_surfaces(dw)[0], w, 8)]))
    r = bpy.ops.nproto.dome(inputs=ins((dw, rim)), dome_cont="G1", dome_layout="DRAPED"); dm = ctx.active_object
    rj, jb = join([dw, dm] if first == "wall" else [dm, dw])
    s = stats(jb) if rj else (0, 0, 0)
    # open: the wall's bottom rim and its 0.01 seam (an edge the wall's own loop uses twice)
    out.append((r == {'FINISHED'} and rj and s[0] == 1 and abs(s[2] - (2 * math.pi * 0.1 + 0.01)) < 1e-3, s[0]))
check("dome (draped): on a wall 0.01 tall joins to it, either one active", all(o for o, _ in out), str(out))

# --- Detach at Isoparm (isodetach._trim): a 0.01 wide strip cut off a holed plane joins back
hp = holed("IsoHoled", 2.0)
per0 = stats(store.get_brep(hp))[2]
r = bpy.ops.nproto.detach_iso(target=hp.name, cuts_u="[0.1, 0.125]")
made = list(ctx.selected_objects)
rj, jb = join(made)
s = stats(jb) if rj else None
check("detach at isoparm: a holed plane with a 0.01 wide strip cut off joins back (both cut lines stitched)",
      r == {'FINISHED'} and len(made) == 3 and rj and s[0] == 2 and abs(s[2] - per0) < 1e-6,
      f"{len(made)} pieces, shared {s and s[0]} (2), open length {s and round(s[2], 5)} ({per0:.5f})")

# --- Extend / cut back a trimmed surface (trimext._grow / _shrink): the 48-gon hole joins its plug
for dist, tag in ((0.1, "extended"), (-0.05, "cut back")):
    out = []
    for first in ("face", "plug"):
        eh = holed("ExtHoled", 3.0)
        side = edge_refs(eh, lambda m: m[0] < 2.81)
        r = bpy.ops.nproto.extend(inputs=json.dumps([[eh.name, side[0]]]), distance=dist); eh = ctx.active_object
        pg = plug("ExtPlug", 3.0)
        rj, jb = join([eh, pg] if first == "face" else [pg, eh])
        out.append(r == {'FINISHED'} and rj and stats(jb)[0] == 48)
    check(f"extend trimmed ({tag}): the holed plane joins the hole's plug, either one active (48 edges stitched)",
          all(out), str(out))

# --- Trim (worker heal_cut_seams): a circle projected through a 48-gon-holed plate, the top disc
# removed; the bottom face the circle only imprinted is one face again (stitched back to its walls)
ro = poly("PlateOut", square(5.0, 0.2))
ri = poly("PlateIn", ngon((5.0, 0, 0), 0.08))
bpy.ops.nproto.extrude(inputs=ins((ro, "S0"), (ri, "S0")), distance=0.1, direction="Z", body="NEW"); plate = ctx.active_object
n0 = store.get_brep(plate).face_count()
bpy.ops.curve.primitive_bezier_circle_add(radius=0.04, location=(5.12, 0.12, 0.3)); cc = ctx.active_object
r = bpy.ops.nproto.trim(target=plate.name, cutters=ins((cc, "S0")), ref_point=(5.12, 0.12, 0.1), normal=(0, 0, 1),
                        view_dir=(0, 0, -1), action="DELETE", scope="PICKED")
b = store.get_brep(plate)
bottom = [f for f in b.faces() if abs(f.interior_point()[2]) < 1e-6]
s = stats(b)
check("trim: the bottom of a 48-gon-holed plate keeps no seam from the cut; only the new hole is open",
      r == {'FINISHED'} and len(bottom) == 1 and b.face_count() == n0 and s[1] == 1,
      f"faces {b.face_count()} ({n0}), bottom faces {len(bottom)} (1), open {s[1]} (1)")

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
