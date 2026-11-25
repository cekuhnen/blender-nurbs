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

# --- fillets along the face: taken off to sculpt the sharp face, made again with their radius
from blender_nurbs.kernel import defeature
_sm = store._k()[0]

def fillet_edges(B, pts, r, shape="CIRCULAR"):
    bpy.ops.nproto.edge_fillet(inputs=json.dumps([[B.name, f"E{edge_at(B, p)}"] for p in pts]), shape=shape,
                               radius=r)

def _uv(srf, q):
    w = _sm.surface_closest_point(srf, tuple(map(float, q)))[1][0]
    return (float(w.x), float(w.y)) if hasattr(w, "x") else (float(w[0]), float(w[1]))

def _kmax(srf, u, v):
    """The larger principal curvature of srf at (u, v) (finite differences)."""
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    h = 1e-4 * max(u1 - u0, v1 - v0)
    u, v = min(max(u, u0 + h), u1 - h), min(max(v, v0 + h), v1 - h)
    S = lambda a, b: np.array(srf.evaluate(a, b), float)
    Su, Sv = (S(u + h, v) - S(u - h, v)) / (2 * h), (S(u, v + h) - S(u, v - h)) / (2 * h)
    Suu = (S(u + h, v) - 2 * S(u, v) + S(u - h, v)) / h ** 2
    Svv = (S(u, v + h) - 2 * S(u, v) + S(u, v - h)) / h ** 2
    Suv = (S(u + h, v + h) - S(u + h, v - h) - S(u - h, v + h) + S(u - h, v - h)) / (4 * h * h)
    n = np.cross(Su, Sv); n /= np.linalg.norm(n)
    I = np.array([[Su @ Su, Su @ Sv], [Su @ Sv, Sv @ Sv]])
    II = np.array([[Suu @ n, Suv @ n], [Suv @ n, Svv @ n]])
    return float(np.abs(np.linalg.eigvals(np.linalg.solve(I, II))).max())

def rounds(B, r, top_pt, ktol=0.05):
    """(how many faces round the top at top_pt are rounds of radius r -- one
    principal curvature 1/r at points inside them, within ktol --, the
    largest kink in degrees along the top's edges with them: 0 when they are tangent).
    The kernel's circular fillet is exact between planes but only fitted on
    a free-form face: ~0.5-1 deg on a gentle dome (its Fillet tool alike)."""
    b = brep(B)
    fs = b.faces()
    top = fs[face_at(B, top_pt)]
    n_ok, kink = 0, 0.0
    for e in top.edges():
        g = [f for f in e.faces() if f != top][0]
        if g.surface().is_planar():
            continue                                   # a sharp edge (no fillet there)
        t0, t1 = e.parameter_range()
        for t in np.linspace(0.05, 0.95, 10):
            q = np.array(e.point_at(t0 + (t1 - t0) * t), float)
            na = np.array(top.outward_normal_at_uv(*_uv(top.surface(), q)), float)
            nb = np.array(g.outward_normal_at_uv(*_uv(g.surface(), q)), float)
            c = na @ nb / (np.linalg.norm(na) * np.linalg.norm(nb))
            kink = max(kink, float(np.degrees(np.arccos(min(1.0, c)))))
    for g in {f for e in top.edges() for f in e.faces() if f != top}:
        if g.surface().is_planar():
            continue
        ip = np.array(g.interior_point(), float)
        pts = [ip] + [0.5 * (ip + np.array(e.point_at(sum(e.parameter_range()) / 2), float)) for e in g.edges()
                      if e.length() > 0.1]
        ks = [_kmax(g.surface(), *_uv(g.surface(), g.closest_point(tuple(map(float, q)))[0])) for q in pts]
        n_ok += all(abs(k * r - 1.0) < ktol for k in ks)
    return n_ok, kink

def round_faces(B, top_pt):
    """How many faces along the top's edges are curved (its fillets)."""
    top = brep(B).faces()[face_at(B, top_pt)]
    return len({f for e in top.edges() for f in e.faces() if f != top and not f.surface().is_planar()})

def fit(name, B, r, top_pt):
    """The kernel fillet's fit on a sculpted face, printed (not checked: its
    circular fillet is exact between planes, only fitted on a free-form
    face -- 0.5 to 10 deg off tangent here, the Fillet tool alike)."""
    rk = rounds(B, r, top_pt, 0.05)
    print(f"INFO kernel fillet fit, {name}: {rk[0]} of 4 within 5 % of radius {r}, worst kink {rk[1]:.2f} deg")

def exact_ok(B, r, top_pt, n):
    """(the n fillets round the top are exact: one principal curvature 1/r
    within 1 % and no kink above 0.05 deg, the measures)."""
    rk = rounds(B, r, top_pt, 0.01)
    return rk[0] == n and rk[1] < 0.05, rk

TOP4 = [(2, 0, 2), (2, 3, 2), (0, 1.5, 2), (4, 1.5, 2)]
def gentle(P):
    P[1:-1, 1:-1, 2] += 0.3
def front_row_down(P):
    P[:, 0, 2] -= 0.4

def net_mid(meta, P):
    net = {"degree_u": meta["degree_u"], "degree_v": meta["degree_v"], "count_u": meta["nu"], "count_v": meta["nv"],
           "knots_u": meta["knots_u"], "knots_v": meta["knots_v"], "points": P.reshape(-1, 3).tolist(),
           "weights": meta.get("weights") or []}
    return cvnet.evaluate(net, [0.5 * (meta["knots_u"][0] + meta["knots_u"][-1])],
                          [0.5 * (meta["knots_v"][0] + meta["knots_v"][-1])])[0, 0]

clear()
B = box()
fillet_edges(B, TOP4, 0.4)
n0 = brep(B).face_count()
r = open_face(B, (2, 1.5, 2)); cage = ctx.active_object
check("a top filleted all round opens", r == {'FINISHED'} and facecvs.is_face_cage(cage), f"{n0} faces")
if facecvs.is_face_cage(cage):
    meta, P = cage_net(cage)
    check("its net spans the sharp top (the fillets off: 4 x 3)",
          np.allclose([P[..., 0].min(), P[..., 0].max(), P[..., 1].min(), P[..., 1].max()], [0, 4, 0, 3], atol=1e-6)
          and np.allclose(P[..., 2], 2.0, atol=1e-9))
    meta, P, dt = move(cage, gentle)
    mid = net_mid(meta, P)
    top = brep(B).faces()[face_at(B, mid)]
    rk = rounds(B, 0.4, mid)
    fl = json.loads(cage.get(facecvs.CAGE_FILLETS, "null"))
    check("domed: a closed solid, its four 0.4 fillets made again (kernel fit: within 1 deg of tangent)",
          solid(B) and brep(B).face_count() == n0 and rk[0] == 4 and rk[1] < 1.0
          and top.closest_point(tuple(map(float, mid)))[1] < 1e-6 and not cage.get(facecvs.CAGE_ERROR),
          f"{brep(B).face_count()} faces, rounds / kink {rk}, {dt * 1000:.0f} ms")
    check("the panel reports the four fillets and their kink (live: the kernel's)", fl is not None
          and fl[0] == 4 and abs(fl[1] - rk[1]) < 0.3 and not fl[2], f"{fl} vs {rk[1]:.2f}")
    t0 = time.perf_counter(); facecvs.settle_now(); dt = time.perf_counter() - t0
    ok_x, rk = exact_ok(B, 0.4, mid, 4)
    fl = json.loads(cage.get(facecvs.CAGE_FILLETS, "null"))
    check("the drag pauses: the four fillets made exactly (rolling ball, tangent), the panel says so",
          ok_x and solid(B) and brep(B).face_count() == n0 and fl and fl[2],
          f"rounds / kink {rk}, {dt * 1000:.0f} ms, panel {fl}")
    meta, P, dt = move(cage, front_row_down)
    check("its front edge down: the front fillet follows", solid(B) and brep(B).face_count() == n0
          and round_faces(B, mid) == 4 and not cage.get(facecvs.CAGE_ERROR), cage.get(facecvs.CAGE_ERROR, "")[:80])
    fit("front edge down (live)", B, 0.4, mid)
    facecvs.settle_now()
    ok_x, rk = exact_ok(B, 0.4, mid, 4)
    check("... exact once it pauses", ok_x and solid(B) and brep(B).face_count() == n0, f"rounds / kink {rk}")
    V_done = vol(B)
    r = bpy.ops.nproto.cvs_done()
    check("Done: a Face CVs step after the fillet", r == {'FINISHED'} and abs(vol(B) - V_done) < 1e-5 * V_done
          and [e["op"] for e in features.stack(B)][-1] == "face_cvs", [e["op"] for e in features.stack(B)])
    # Tab on the sculpted top again: it opens (it reads as a fillet itself, tangent to its fillets all
    # round, and the kernel can't take fillets off a free-form face: worker._sharpen_region does)
    fs = brep(B).faces()
    itop = max(range(len(fs)), key=lambda k: fs[k].area(1e-6) if not fs[k].surface().is_planar() else 0)
    for x in ctx.selected_objects: x.select_set(False)
    B.select_set(True); ctx.view_layer.objects.active = B
    subsel.set_picks(B, "FACE", [itop])
    r = bpy.ops.nproto.edit_cvs(); cage = ctx.active_object
    meta2, P2 = cage_net(cage) if facecvs.is_face_cage(cage) else (None, None)
    cands = [P, P.transpose(1, 0, 2)]
    cands += [c[::-1] for c in cands] + [c[:, ::-1] for c in cands] + [c[::-1, ::-1] for c in cands]
    diffs = [float(np.abs(P2 - c).max()) for c in cands if P2 is not None and c.shape == P2.shape]
    check("the sculpted, filleted top opens again on its sculpted net", r == {'FINISHED'}
          and facecvs.is_face_cage(cage) and diffs and min(diffs) < 1e-4, f"diff {min(diffs) if diffs else '-'}")
    if facecvs.is_face_cage(cage):
        def lift(P):
            P[2:4, 2, 2] += 0.1
        move(cage, lift)
        check("... and sculpts again, its fillets made again", solid(B) and brep(B).face_count() == n0
              and round_faces(B, mid) == 4 and not cage.get(facecvs.CAGE_ERROR),
              f"{brep(B).face_count()} faces {cage.get(facecvs.CAGE_ERROR, '')[:80]}")
        fit("re-opened and sculpted again (live)", B, 0.4, mid)
        facecvs.settle_now()
        ok_x, rk = exact_ok(B, 0.4, mid, 4)
        check("... exact once it pauses", ok_x and solid(B) and brep(B).face_count() == n0, f"rounds / kink {rk}")
        bpy.ops.nproto.cvs_cancel()
    raw = json.loads(B[features.PROP_STACK])
    k = next((i for i, e in enumerate(raw) if e["op"] == "edge_fillet"), None)
    if k is not None:
        raw[k]["params"]["radius"] = 0.3
        B[features.PROP_STACK] = json.dumps(raw)
        features.rebuild(B, ctx)
    ok_x, rk = exact_ok(B, 0.3, mid, 4)
    check("history: the fillet made smaller, the sculpted top keeps exact 0.3 fillets", k is not None
          and solid(B) and brep(B).face_count() == n0 and ok_x, f"rounds / kink {rk}")

# --- one filleted top edge: the fillet ends against the side walls, exactly
clear()
B = box()
fillet_edges(B, [(2, 0, 2)], 0.4)
n0 = brep(B).face_count()
r = open_face(B, (2, 1.5, 2)); cage = ctx.active_object
if facecvs.is_face_cage(cage):
    meta, P, dt = move(cage, gentle)
    mid = net_mid(meta, P)
    facecvs.settle_now()
    ok_x, rk = exact_ok(B, 0.4, mid, 1)
    check("one filleted top edge, domed: its fillet made exactly, cut off by the side walls", ok_x and solid(B)
          and brep(B).face_count() == n0 and not cage.get(facecvs.CAGE_ERROR), f"{brep(B).face_count()}/{n0} "
          f"faces, rounds / kink {rk}")
    bpy.ops.nproto.cvs_cancel()
else:
    check("one filleted top edge opens", False)

clear()
B = box()
fillet_edges(B, TOP4 + [(2, 0, 0), (2, 3, 0), (0, 1.5, 0), (4, 1.5, 0), (0, 0, 1), (4, 0, 1), (4, 3, 1), (0, 3, 1)], 0.4)
n0 = brep(B).face_count()
r = open_face(B, (2, 1.5, 2)); cage = ctx.active_object
dt = 0.0
if facecvs.is_face_cage(cage):
    meta, P, dt = move(cage, front_row_down)
check("every edge filleted: the top sculpted, all twelve fillets (and corners) made again",
      facecvs.is_face_cage(cage) and solid(B) and brep(B).face_count() == n0 and not cage.get(facecvs.CAGE_ERROR)
      and round_faces(B, (2, 1.5, 2)) == 4, f"{brep(B).face_count()}/{n0} faces, {dt * 1000:.0f} ms")
fit("every edge filleted", B, 0.4, (2, 1.5, 2))
if facecvs.is_face_cage(cage):
    facecvs.settle_now()
    fl = json.loads(cage.get(facecvs.CAGE_FILLETS, "null"))
    check("... they meet three at a corner: still the kernel's when it pauses (tried, said in the panel)",
          solid(B) and brep(B).face_count() == n0 and fl and not fl[2] and fl[3], f"panel {fl}")
    bpy.ops.nproto.cvs_cancel()

# --- a soft (blend) fillet along the top is still refused
clear()
B = box()
fillet_edges(B, [(2, 0, 2)], 0.4, "BLEND")
r = open_face(B, (2, 1.5, 2))
check("a top with a soft fillet along it is refused (Remove Fillets first)",
      r == {'CANCELLED'} and not cvs.is_cage(ctx.active_object))

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
