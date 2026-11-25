"""Plywood: veneer on both sides of a bent shell, ply layers along its rim:
    blender -b --factory-startup --python tests/test_plywood.py"""
import json, math, os, sys
import bmesh
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import facetex, plywood, store, subsel

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def sel(*objs):
    for o in ctx.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0]

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

def arrays(o):
    me = o.data
    n, nl = len(me.polygons), len(me.loops)
    mi = np.empty(n, np.int64); me.polygons.foreach_get("material_index", mi)
    lt = np.empty(n, np.int64); me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(nl, np.int64); me.loops.foreach_get("vertex_index", lv)
    uv = np.empty(nl * 2); me.uv_layers["UVMap"].data.foreach_get("uv", uv)
    return mi, np.repeat(mi, lt), lv, uv.reshape(-1, 2)

def side_area(o, c):
    me = o.data
    return sum(p.area for p in me.polygons if p.material_index == c)

def uv_area(o, c):
    me = o.data
    uv = me.uv_layers["UVMap"].data
    a = 0.0
    for p in me.polygons:
        if p.material_index != c:
            continue
        q = [uv[i].uv for i in p.loop_indices]
        a += 0.5 * abs(sum(q[i].x * q[i - 1].y - q[i - 1].x * q[i].y for i in range(len(q))))
    return a

def continuous(o, c):
    """Every vertex of side c has one UV (the side is one unbroken sheet)."""
    mi, lc, lv, UV = arrays(o)
    worst = 0.0
    seen = {}
    for k in np.flatnonzero(lc == c):
        v = int(lv[k])
        if v in seen:
            worst = max(worst, float(np.linalg.norm(UV[k] - seen[v])))
        else:
            seen[v] = UV[k]
    return worst

def mean_normal(o, c):
    me = o.data
    n = sum((p.normal * p.area for p in me.polygons if p.material_index == c), me.polygons[0].normal * 0)
    return (o.matrix_world.to_3x3() @ n).normalized()

# --- 1. a bent sheet (seat, bend, back) of two faces, thickened 12 mm
cu = bpy.data.curves.new("prof", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('BEZIER'); sp.bezier_points.add(2)
for bp, p in zip(sp.bezier_points, [(0, -0.45, 0), (0, 0.0, 0.02), (0, 0.12, 0.5)]):
    bp.co = p; bp.handle_left_type = bp.handle_right_type = 'AUTO'
prof = bpy.data.objects.new("prof", cu); ctx.collection.objects.link(prof)
r = bpy.ops.nproto.extrude(inputs=ins((prof, "S0")), distance=0.5, direction="X")
sheet = ctx.active_object
sel(sheet)
bpy.ops.nproto.thicken(distance=0.012)
shell = ctx.active_object
check("setup: bent sheet thickened", store.is_nurbs(shell) and shell.name != sheet.name,
      f"{store.get_brep(shell).face_count()} faces")

sel(shell)
r = bpy.ops.nproto.plywood(target="MESH", front="WALNUT", back="SAME", core="BIRCH")
ply = ctx.active_object
mi, lc, lv, UV = arrays(ply)
nf, nb, ne = (int((mi == c).sum()) for c in range(3))
check("plywood: a new mesh, three classes, source hidden",
      r == {'FINISHED'} and ply != shell and ply.type == 'MESH' and not store.is_nurbs(ply)
      and nf > 0 and nb > 0 and ne > 0 and shell.hide_get() and shell.hide_render, f"{nf}/{nb}/{ne}")
check("plywood: 12 mm -> 9 plies (one per ~1.5 mm, odd)", ply["ply_count"] == 9, ply["ply_count"])
check("plywood: front faces up", mean_normal(ply, 0).z > 0.3 and mean_normal(ply, 1).z < -0.3,
      f"{mean_normal(ply, 0)} {mean_normal(ply, 1)}")
check("plywood: back uses the front's material (Same as Front), edge its own",
      ply.data.materials[0] == ply.data.materials[1] and ply.data.materials[2] != ply.data.materials[0]
      and ply.data.materials[2].get("nproto_ply") == "edge")
for c, name in ((0, "front"), (1, "back")):
    a3, a2 = side_area(ply, c), uv_area(ply, c)
    check(f"plywood: {name} veneer in metres (UV area {a2:.4f} = 3D area {a3:.4f}) and one unbroken sheet",
          abs(a2 - a3) < 0.03 * a3 and continuous(ply, c) < 1e-5, f"jump {continuous(ply, c):.2e}")
ev = UV[lc == 2]
check("plywood: edge band depth 0 (front) .. 1 (back)", abs(ev[:, 1].min()) < 1e-4 and abs(ev[:, 1].max() - 1) < 1e-4,
      f"{ev[:, 1].min():.4f} {ev[:, 1].max():.4f}")
perim = 2 * (0.5 + 1.0)    # 0.5 wide, the profile about 1 m long
check("plywood: edge band u runs along the rim (about its length)", 0.8 * perim < np.ptp(ev[:, 0]) < 1.25 * perim,
      f"{np.ptp(ev[:, 0]):.3f}")
# depth: corners on the front veneer's boundary are at 0, on the back's at 1
P = np.array([ply.matrix_world @ v.co for v in ply.data.vertices])
fv = set(lv[lc == 0].tolist()); bv = set(lv[lc == 1].tolist())
ek = np.flatnonzero(lc == 2)
on_f = [k for k in ek if lv[k] in fv]; on_b = [k for k in ek if lv[k] in bv]
check("plywood: the band meets the front at depth 0 and the back at depth 1",
      on_f and on_b and max(abs(UV[k, 1]) for k in on_f) < 1e-4 and min(UV[k, 1] for k in on_b) > 1 - 1e-4)
# the veneer grain along the sheet's length: the front sheet longer in u than in v
fu = UV[lc == 0]
check("plywood: grain (u) along the sheet's length", np.ptp(fu[:, 0]) > 1.5 * np.ptp(fu[:, 1]),
      f"{np.ptp(fu[:, 0]):.3f} x {np.ptp(fu[:, 1]):.3f}")

# --- 2. run again on the result: the same object and materials, from its source
front_mat = ply.data.materials[0]
front_mat.node_tree.nodes["Veneer BSDF"].inputs["Roughness"].default_value = 0.2   # a user's edit
sel(ply)
r = bpy.ops.nproto.plywood('INVOKE_DEFAULT')
check("update: same object, same materials (the user's edit kept), settings remembered",
      r == {'FINISHED'} and ctx.active_object == ply and ply.data.materials[0] == front_mat
      and abs(front_mat.node_tree.nodes["Veneer BSDF"].inputs["Roughness"].default_value - 0.2) < 1e-6
      and json.loads(ply[plywood.PROP])["front"] == "WALNUT"
      and sum(1 for o in bpy.data.objects if plywood.PROP in o) == 1)

# --- 3. swap sides, own back veneer, more plies, grain turned
sel(ply)
r = bpy.ops.nproto.plywood(front="WALNUT", back="BIRCH", core="BIRCH", swap=True, plies=11, grain_angle=90)
fu = arrays(ply)[3][arrays(ply)[1] == 0]
check("swap: the front faces down now, back its own material, 11 plies, grain across",
      r == {'FINISHED'} and mean_normal(ply, 0).z < -0.3 and ply.data.materials[1] != ply.data.materials[0]
      and ply["ply_count"] == 11 and np.ptp(fu[:, 1]) > 1.5 * np.ptp(fu[:, 0]),
      f"{r} {mean_normal(ply, 0).z:.2f} {ply.data.materials[1] != ply.data.materials[0]} {ply['ply_count']} "
      f"{np.ptp(fu[:, 0]):.3f} x {np.ptp(fu[:, 1]):.3f}")

# --- 4. a plain mesh: a flat 20 mm panel with a hole, textured in place
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=8, y_segments=8, size=0.5)
bm.faces.ensure_lookup_table()
mid = [f for f in bm.faces if abs(f.calc_center_median().x) < 0.13 and abs(f.calc_center_median().y) < 0.13]
bmesh.ops.delete(bm, geom=mid, context='FACES')
ext = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
bmesh.ops.translate(bm, vec=(0, 0, 0.02), verts=[g for g in ext["geom"] if isinstance(g, bmesh.types.BMVert)])
bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
me = bpy.data.meshes.new("panel"); bm.to_mesh(me); bm.free()
panel = bpy.data.objects.new("panel", me); ctx.collection.objects.link(panel)
sel(panel)
r = bpy.ops.nproto.plywood(front="OAK")
mi, lc, lv, UV = arrays(panel)
ev = UV[lc == 2]
check("mesh: textured in place (no new object), 20 mm -> 13 plies, both sides and the rims",
      r == {'FINISHED'} and ctx.active_object == panel and panel["ply_count"] == 13
      and (mi == 0).sum() > 0 and (mi == 1).sum() > 0 and (mi == 2).sum() == 4 * 8 + 8,
      f"{[(mi == c).sum() for c in range(3)]} plies {panel['ply_count']}")
check("mesh: the outer rim and the hole's rim each along its own length, no jumps inside a face",
      abs(ev[:, 1].min()) < 1e-6 and abs(ev[:, 1].max() - 1) < 1e-6
      and all(np.ptp(UV[[k for k in range(p.loop_start, p.loop_start + p.loop_total)], 0]) < 0.2
              for p in panel.data.polygons if p.material_index == 2),
      f"u {ev[:, 0].min():.3f}..{ev[:, 0].max():.3f}")

# --- 5. on the NURBS object itself: its own faces sorted, materials and UVs kept through rebuilds
shell.hide_set(False)
sel(shell)
mats_before = len(bpy.data.materials)
r = bpy.ops.nproto.plywood(target="OBJECT", front="OAK", back="SAME", core="BIRCH")
me = shell.data
mi, lc, lv, UV = arrays(shell)
fid = np.empty(len(me.polygons), np.int64); me.attributes[facetex.ATTR].data.foreach_get("value", fid)
fc = facetex.plywood_classes(shell)
check("brep: textured in place (no new object), every triangle on its face's class",
      r == {'FINISHED'} and ctx.active_object == shell and store.is_nurbs(shell) and not shell.hide_get()
      and fc is not None and np.array_equal(mi, fc[fid]) and set(np.unique(mi)) == {0, 1, 2}
      and len(me.materials) == 3 and me.materials[2].get("nproto_ply") == "edge",
      f"classes {fc} counts {np.bincount(mi)}")
labels_full = np.array([facetex._Faces(store.get_brep(shell)).nearest(np.array(p.center))[0] for p in me.polygons])
check("brep: the face labels match the nearest face of every triangle", np.array_equal(fid, labels_full),
      f"{int((fid != labels_full).sum())} differ")
for c, name in ((0, "front"), (1, "back")):
    a3, a2 = side_area(shell, c), uv_area(shell, c)
    check(f"brep: {name} veneer one sheet in metres (UV {a2:.4f} = 3D {a3:.4f})",
          abs(a2 - a3) < 0.03 * a3 and continuous(shell, c) < 1e-5, f"jump {continuous(shell, c):.2e}")
ev = UV[lc == 2]
check("brep: edge band depth 0..1, thickness and plies on the object",
      abs(ev[:, 1].min()) < 1e-4 and abs(ev[:, 1].max() - 1) < 1e-4 and abs(shell["ply_thickness"] - 0.012) < 0.0015
      and shell["ply_count"] == 9, f"t {shell['ply_thickness']:.5f}")
# a rebuild (new display detail) and a new B-rep (a new revision) keep it all
store.retessellate(shell, ctx)
store.set_geometry(shell, store.get_brep(shell).copy(), ctx)
mi2 = arrays(shell)[0]
check("brep: kept through a re-mesh and a new B-rep revision",
      facetex.LAST.get(shell.name) is None and set(np.unique(mi2)) == {0, 1, 2} and len(shell.data.materials) == 3
      and "UVMap" in shell.data.uv_layers and abs(uv_area(shell, 0) - side_area(shell, 0)) < 0.03 * side_area(shell, 0))
# your own materials: one named for the edge, one put in the back slot by hand, both kept on a re-run
mine_e = bpy.data.materials.new("My Ply Edge")
mine_b = bpy.data.materials.new("My Back")
shell.data.materials[1] = mine_b
sel(shell)
r = bpy.ops.nproto.plywood('INVOKE_DEFAULT', edge_material="My Ply Edge")
check("brep: own edge material used, hand-placed back material kept, front still the tool's",
      r == {'FINISHED'} and shell.data.materials[2] == mine_e and shell.data.materials[1] == mine_b
      and shell.data.materials[0].get("nproto_ply") == "front")
# Assign Material to picked faces: a fourth slot on one front face, kept through a rebuild
mine_f = bpy.data.materials.new("Inlay")
shell.data.materials.append(mine_f)
shell.active_material_index = 3
k_face = int(np.flatnonzero(fc == 0)[0])
subsel.set_picks(shell, "FACE", [k_face])
r = bpy.ops.nproto.face_material(action="ASSIGN")
store.set_geometry(shell, store.get_brep(shell).copy(), ctx)
mi3 = arrays(shell)[0]
fid3 = np.empty(len(shell.data.polygons), np.int64); shell.data.attributes[facetex.ATTR].data.foreach_get("value", fid3)
check("face material: the picked face takes slot 3, the others keep their plywood slots",
      r == {'FINISHED'} and (mi3[fid3 == k_face] == 3).all()
      and (mi3[fid3 != k_face] == facetex.plywood_classes(shell)[fid3[fid3 != k_face]]).all())
subsel.set_picks(shell, "FACE", [])
r = bpy.ops.nproto.face_material(action="REMOVE")
check("face material: Remove Texture leaves a plain NURBS mesh (no labels, no UVs, slot 0)",
      r == {'FINISHED'} and facetex.PROP not in shell and "UVMap" not in shell.data.uv_layers
      and max(p.material_index for p in shell.data.polygons) == 0)

# --- 6. any NURBS object: all faces one material, each face its own island in metres
sel(sheet)
sheet.hide_set(False)
m_sheet = bpy.data.materials.new("Sheet")
sheet.data.materials.append(m_sheet)
sheet.active_material_index = 0
r = bpy.ops.nproto.face_material(action="ASSIGN")
me = sheet.data
fid = np.empty(len(me.polygons), np.int64); me.attributes[facetex.ATTR].data.foreach_get("value", fid)
uv = np.empty(len(me.loops) * 2); me.uv_layers["UVMap"].data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
a3 = sum(p.area for p in me.polygons)
a2 = sum(0.5 * abs(sum(uv[i][0] * uv[j][1] - uv[j][0] * uv[i][1]
                       for i, j in zip(p.loop_indices, list(p.loop_indices[1:]) + [p.loop_indices[0]])))
         for p in me.polygons)
check("face material: all faces, UVs in metres (area kept), one island per face",
      r == {'FINISHED'} and abs(a2 - a3) < 0.03 * a3 and len(np.unique(fid)) == store.get_brep(sheet).face_count(),
      f"UV {a2:.4f} 3D {a3:.4f}")

# --- 7. not a shell: a lone sheet is refused with a message, nothing made
bpy.ops.mesh.primitive_plane_add(size=1.0)
pl = ctx.active_object
r = bpy.ops.nproto.plywood()
check("refuses a sheet with no thickness", r == {'CANCELLED'} and "nproto_plywood" not in pl)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
