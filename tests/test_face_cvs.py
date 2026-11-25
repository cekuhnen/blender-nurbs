"""Face CVs: sculpt one face of a solid by its CVs, the walls re-trimmed:
    blender -b --factory-startup --python tests/test_face_cvs.py"""
import json, os, sys, time
import bmesh
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import cvs, facecvs, features, store, subsel
from blender_nurbs.kernel import cvnet

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def poly(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def box(h=2.0):
    R = poly("R", [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)])
    bpy.ops.nproto.extrude(inputs=json.dumps([[R.name, "S0"]]), distance=h, direction="Z", body="NEW")
    return ctx.active_object

def brep(o): return store.get_brep(o)
def vol(o): return abs(brep(o).volume(1e-6))
def solid(o): return bool(brep(o).is_manifold_solid())

def face_at(o, p):
    fs = brep(o).faces()
    return min(range(len(fs)), key=lambda k: fs[k].closest_point(tuple(map(float, p)))[1])

def edge_at(o, p):
    es = brep(o).edges()
    def d(e):
        t0, t1 = e.parameter_range()
        return np.linalg.norm(np.array(e.point_at(0.5 * (t0 + t1)), float) - np.asarray(p, float))
    return min(range(len(es)), key=lambda k: d(es[k]))

def open_face(o, p):
    for x in ctx.selected_objects: x.select_set(False)
    o.select_set(True); ctx.view_layer.objects.active = o
    subsel.set_picks(o, "FACE", [face_at(o, p)])
    return bpy.ops.nproto.edit_cvs()

def cage_net(cage):
    meta = json.loads(cage[cvs.CAGE_META])
    return meta, np.array(cvs.cage_points(cage), float).reshape(meta["nu"], meta["nv"], 3)

def move(cage, fn):
    """fn(P (nu, nv, 3)) changes P in place; the cage's vertices follow, then the live rebuild runs."""
    meta, P = cage_net(cage)
    fn(P)
    bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table()
    for v, q in zip(bm.verts, P.reshape(-1, 3)):
        v.co = q
    bmesh.update_edit_mesh(cage.data)
    ctx.view_layer.update()
    t0 = time.perf_counter(); cvs._process(); dt = time.perf_counter() - t0
    return meta, P, dt

def on_net(o, meta, P, top_pt):
    """(max distance of the net's surface to the body's top face, max
    distance of the top face's edges from the box's sides): the top is the
    sculpted surface, trimmed exactly by the walls."""
    net = {"degree_u": meta["degree_u"], "degree_v": meta["degree_v"], "count_u": meta["nu"], "count_v": meta["nv"],
           "knots_u": meta["knots_u"], "knots_v": meta["knots_v"], "points": P.reshape(-1, 3).tolist(),
           "weights": meta["weights"]}
    G = cvnet.evaluate(net, np.linspace(0, 1, 9), np.linspace(0, 1, 7)).reshape(-1, 3)
    top = brep(o).faces()[face_at(o, top_pt)]
    d1 = max(top.closest_point(tuple(map(float, q)))[1] for q in G)
    E = np.array([e.point_at(t) for e in top.edges() for t in np.linspace(*e.parameter_range(), 15)], float)
    d2 = float(np.minimum.reduce([np.abs(E[:, 0]), np.abs(E[:, 0] - 4), np.abs(E[:, 1]), np.abs(E[:, 1] - 3)]).max())
    return d1, d2

def expected_volume(meta, P, base=0.0):
    """Box volume under the sculpted top: the integral of z dx dy over its
    rectangle (x and y run with the even CV grid, so with the Jacobian)."""
    net = {"degree_u": meta["degree_u"], "degree_v": meta["degree_v"], "count_u": meta["nu"], "count_v": meta["nv"],
           "knots_u": meta["knots_u"], "knots_v": meta["knots_v"], "points": P.reshape(-1, 3).tolist(),
           "weights": meta["weights"]}
    us, vs = np.linspace(0, 1, 401), np.linspace(0, 1, 321)
    S = cvnet.evaluate(net, us, vs)
    xu, xv = np.gradient(S[..., 0], us, axis=0), np.gradient(S[..., 0], vs, axis=1)
    yu, yv = np.gradient(S[..., 1], us, axis=0), np.gradient(S[..., 1], vs, axis=1)
    f = (S[..., 2] - base) * np.abs(xu * yv - xv * yu)
    w = lambda n: np.r_[0.5, np.ones(n - 2), 0.5]
    return float(w(len(us)) @ f @ w(len(vs))) / ((len(us) - 1) * (len(vs) - 1))

ctx.scene.nproto_settings.construction_history = True

# --- open a face cage on the box top: a fresh cubic rectangle, nothing changes yet
clear()
B = box()
text0 = B[store.PROP_BREP]
r = open_face(B, (2, 1.5, 2))
cage = ctx.active_object
check("Tab on a picked face opens a face cage in Edit Mode",
      r == {'FINISHED'} and facecvs.is_face_cage(cage) and cage.mode == 'EDIT', f"{r} {cage.name}")
meta, P = cage_net(cage)
check("the flat top gets a cubic net with its edge CVs on its edges",
      meta["degree_u"] == 3 and meta["degree_v"] == 3 and meta["nu"] >= 4 and meta["nv"] >= 4
      and np.allclose(P[..., 2], 2.0) and np.isclose(P[..., 0].min(), 0) and np.isclose(P[..., 0].max(), 4)
      and np.isclose(P[..., 1].min(), 0) and np.isclose(P[..., 1].max(), 3), f"{meta['nu']} x {meta['nv']}")
check("opening the cage changes nothing", B[store.PROP_BREP] == text0)
gaps = [np.diff(P[:, 0, 0]), np.diff(P[0, :, 1])]
check("the CVs are evenly spaced (an even grid, not bunched at the sides)",
      all(np.allclose(g, g[0]) for g in gaps), [np.round(g, 4).tolist() for g in gaps])

# --- Add CVs before a move (exact: the face keeps its shape)
s = ctx.window_manager.nproto_face_cvs
facecvs._syncing[0] = True; s.nu, s.nv, s.degree = 7, 5, 3; facecvs._syncing[0] = False
r = bpy.ops.nproto.face_cvs_refine()
meta, P = cage_net(cage)
check("Add CVs: 7 x 5 CVs, still flat on the top", r == {'FINISHED'} and (meta["nu"], meta["nv"]) == (7, 5)
      and np.allclose(P[..., 2], 2.0) and len(cage.data.vertices) == 35, f"{meta['nu']} x {meta['nv']}")

# --- sculpt inside: a dome; the walls stay, the solid is exact
def dome(P):
    P[2:5, 1:4, 2] += 1.0
meta, P, dt = move(cage, dome)
V, Ve = vol(B), expected_volume(meta, P)
check("a dome inside the face: closed 6-face solid, exact volume",
      solid(B) and brep(B).face_count() == 6 and abs(V - Ve) < 1e-4 * Ve, f"vol {V:.5f} expected {Ve:.5f}")
check("the live rebuild is quick", dt < 0.5, f"{dt * 1000:.0f} ms")

# --- edge CVs down along the front wall: the wall is re-trimmed
def front_down(P):
    P[2:5, 0, 2] -= 0.8
meta, P, dt = move(cage, front_down)
V, Ve = vol(B), expected_volume(meta, P)
y0 = float(P[0, 0, 1])                                   # the wall along the net's first row
wall = brep(B).faces()[face_at(B, (2, y0, 0.5))]
W = np.array([e.point_at(t) for e in wall.edges() for t in np.linspace(*e.parameter_range(), 9)], float)
zs = [q[2] for q in W if 0.5 < q[0] < 3.5 and q[2] > 0.5]     # its top edge, away from the corners
d1, d2 = on_net(B, meta, P, (2, 1.5, 2))
check("edge CVs down: that wall's top edge dips; the top is the sculpted surface, trimmed exactly",
      solid(B) and brep(B).face_count() == 6 and min(zs) < 1.5 and d1 < 1e-6 and d2 < 1e-9
      and abs(V - Ve) < 1e-4 * Ve,
      f"lowest wall top {min(zs):.3f}, top off the net {d1:.1e}, edges off the sides {d2:.1e}, "
      f"vol {V:.5f} expected {Ve:.5f}")

# --- a corner CV down: the corner slides down its side edge
def corner_down(P):
    P[-1, -1, 2] -= 0.6
meta, P, dt = move(cage, corner_down)
V, Ve = vol(B), expected_volume(meta, P)
cxy = P[-1, -1, :2]
vz = min((np.array(v.point(), float) for e in brep(B).edges() for v in e.vertices()),
         key=lambda q: np.linalg.norm(q[:2] - cxy) + abs(q[2] - 2))
d1, d2 = on_net(B, meta, P, (2, 1.5, 2))
check("a corner CV down: that corner of the solid goes down with it",
      solid(B) and abs(vz[2] - 1.4) < 1e-7 and d1 < 1e-6 and d2 < 1e-9 and abs(V - Ve) < 1e-4 * Ve,
      f"corner z {vz[2]:.10f}, top off the net {d1:.1e}, vol {V:.5f} / {Ve:.5f}")
good_text = B[store.PROP_BREP]

# --- after moves: Add CVs keeps the shape exactly, Rebuild refits on an even grid
text_before = B[store.PROP_BREP]
meta0, P0 = cage_net(cage)
facecvs._syncing[0] = True; s.nu, s.nv = meta0["nu"] + 2, meta0["nv"] + 1; facecvs._syncing[0] = False
r = bpy.ops.nproto.face_cvs_refine(mode="ADD")
meta1, P1 = cage_net(cage)
d1, d2 = on_net(B, meta1, P1, (2, 1.5, 2))
check("Add CVs after moves: more CVs, the same shape (exact)",
      r == {'FINISHED'} and (meta1["nu"], meta1["nv"]) == (meta0["nu"] + 2, meta0["nv"] + 1) and d1 < 1e-6
      and solid(B) and not cage.get(facecvs.CAGE_ERROR), f"{meta1['nu']} x {meta1['nv']}, top off the new net {d1:.1e}")
facecvs._syncing[0] = True; s.nu, s.nv = 6, 5; facecvs._syncing[0] = False
r = bpy.ops.nproto.face_cvs_refine(mode="REBUILD")
meta2, P2 = cage_net(cage)
d1, d2 = on_net(B, meta2, P2, (2, 1.5, 2))
gx = np.diff(P2[:, 0, 0]); gy = np.diff(P2[0, :, 1])
dev = cage.get("nproto_face_dev", -1)
check("Rebuild after moves: an even 6 x 5 grid again, the shape within the shown deviation",
      r == {'FINISHED'} and (meta2["nu"], meta2["nv"]) == (6, 5) and np.allclose(gx, gx[0]) and np.allclose(gy, gy[0])
      and d1 < 1e-6 and 0 <= dev < 0.2 and solid(B), f"dev {dev:.3g}, top off the rebuilt net {d1:.1e}")
r = bpy.ops.nproto.face_cvs_refine(mode="ADD")      # (6 x 5 again: nothing to add)
facecvs._syncing[0] = True; s.nu, s.nv = 4, 4; facecvs._syncing[0] = False
r = bpy.ops.nproto.face_cvs_refine(mode="ADD")
check("Add CVs never takes CVs away", r == {'CANCELLED'} and cage_net(cage)[0]["nu"] == 6)
P_layout = cage_net(cage)
good_text = B[store.PROP_BREP]

# --- a shape that can't be built keeps the last good one
def through_floor(P):
    P[3, 2, 2] -= 12.0
meta_l, P_l = P_layout
meta, P, dt = move(cage, through_floor)
check("a CV pushed through the floor: refused, the last shape kept",
      cage.get(facecvs.CAGE_ERROR) and B[store.PROP_BREP] == good_text, cage.get(facecvs.CAGE_ERROR, "")[:90])
def back_up(P):
    P[3, 2, 2] += 12.0
meta, P, dt = move(cage, back_up)
check("moved back: built again, the error gone", not cage.get(facecvs.CAGE_ERROR) and solid(B))
V_done = vol(B)

# --- Done: a Face CVs step; the rebuild replays it on a taller box
r = bpy.ops.nproto.cvs_done()
st = features.stack(B)
check("Done adds a Face CVs step", r == {'FINISHED'} and cvs.cage_for(B) is None and [e["op"] for e in st] == ["face_cvs"]
      and abs(vol(B) - V_done) < 1e-9, [e["op"] for e in st])
# Tab again on the sculpted top: the same net comes back (not the surface
# continued far past the walls for the re-trim)
P_done = P.copy()
fs = brep(B).faces()
itop = max(range(len(fs)), key=lambda k: fs[k].area(1e-6) if fs[k].closest_point((2, 1.5, 2.5))[1] < 1.5 else 0)
for x in ctx.selected_objects: x.select_set(False)
B.select_set(True); ctx.view_layer.objects.active = B
subsel.set_picks(B, "FACE", [itop])
r = bpy.ops.nproto.edit_cvs()
cage = ctx.active_object
meta2, P2 = cage_net(cage) if facecvs.is_face_cage(cage) else (None, None)
# (the kernel may hand the face back with u and v swapped or run the other way: the same grid)
cands = [P_done, P_done.transpose(1, 0, 2)]
cands += [c[::-1] for c in cands] + [c[:, ::-1] for c in cands] + [c[::-1, ::-1] for c in cands]
diffs = [float(np.abs(P2 - c).max()) for c in cands if P2 is not None and c.shape == P2.shape]
check("re-opening the sculpted face gives its sculpted net back", r == {'FINISHED'} and diffs and min(diffs) < 1e-5,
      f"{None if P2 is None else P2.shape} vs {P_done.shape}, diff {min(diffs) if diffs else '-'}")
if facecvs.is_face_cage(cage):
    bpy.ops.nproto.cvs_cancel()
check("Cancel on the re-opened face leaves one step and the shape", len(features.stack(B)) == 1
      and abs(vol(B) - V_done) < 1e-9)
raw = json.loads(B[features.PROP_HIST])
raw["params"]["distance"] = 3.0
B[features.PROP_HIST] = json.dumps(raw)
features.rebuild(B, ctx)
d1, d2 = on_net(B, meta, P + np.array([0.0, 0.0, 1.0]), (2, 1.5, 3))
check("history: a taller box keeps its sculpted top (one unit higher)", solid(B) and d1 < 1e-6 and d2 < 1e-9,
      f"top off the raised net {d1:.1e}, edges off the sides {d2:.1e}")

# --- Cancel puts the body back exactly
clear()
B = box()
text0 = B[store.PROP_BREP]
open_face(B, (2, 1.5, 2)); cage = ctx.active_object
move(cage, dome)
changed = B[store.PROP_BREP] != text0
r = bpy.ops.nproto.cvs_cancel()
check("Cancel restores the body exactly", changed and r == {'FINISHED'} and B[store.PROP_BREP] == text0
      and features.stack(B) == [])

# --- a face beside a sculpted face (the user's case: the top's front edge dipped, then the
# front wall sculpted): live, and quick -- its curved neighbour used to take ~1 s to extend
clear()
B = box()
open_face(B, (2, 1.5, 2)); cage = ctx.active_object
def top_dip(P):
    edge = np.isclose(P[..., 1], 0.0) & (P[..., 0] > 0.1) & (P[..., 0] < 3.9)   # the front edge's inner CVs
    P[..., 2] -= 0.8 * edge
move(cage, top_dip)
bpy.ops.nproto.cvs_done()
r = open_face(B, (2, 0, 0.5)); cage = ctx.active_object
check("the front wall beside the sculpted top opens", r == {'FINISHED'} and facecvs.is_face_cage(cage))
if facecvs.is_face_cage(cage):
    def bulge(P):
        P[1:-1, 1:-1, 1] -= 0.3                         # out of the front wall (-y)
    times = []
    for k in range(3):
        meta, P, dt = move(cage, bulge)
        times.append(dt)
    check("sculpting it: a closed 6-face solid, each live rebuild quick", solid(B) and brep(B).face_count() == 6
          and not cage.get(facecvs.CAGE_ERROR) and max(times) < 0.4,
          f"{[round(t * 1000) for t in times]} ms, err {cage.get(facecvs.CAGE_ERROR)}")
    bpy.ops.nproto.cvs_done()

# --- a filleted vertical edge (the video's box): the fillet is a wall and re-trims too
clear()
B = box()
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[B.name, f"E{edge_at(B, (4, 0, 1))}"]]), shape="CIRCULAR", radius=0.6)
nf = brep(B).face_count()
r = open_face(B, (2, 1.5, 2)); cage = ctx.active_object
check("a box with a filleted vertical edge: its top opens", r == {'FINISHED'} and facecvs.is_face_cage(cage), f"{nf} faces")
if facecvs.is_face_cage(cage):
    def dip(P):
        P[1:-1, 1:-1, 2] -= 0.9
    meta, P, dt = move(cage, dip)
    check("the top dips; the fillet and walls follow", solid(B) and brep(B).face_count() == nf
          and not cage.get(facecvs.CAGE_ERROR) and vol(B) < 24, f"vol {vol(B):.4f} faces {brep(B).face_count()} "
          f"{cage.get(facecvs.CAGE_ERROR, '')[:80]}")
    bpy.ops.nproto.cvs_done()

# --- a filleted top edge (tangent to the top) is refused
clear()
B = box()
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[B.name, f"E{edge_at(B, (2, 0, 2))}"]]), shape="CIRCULAR", radius=0.4)
r = open_face(B, (2, 1.5, 2))
check("a top with a fillet along it is refused (Remove Fillets first)",
      r == {'CANCELLED'} and not cvs.is_cage(ctx.active_object))

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
