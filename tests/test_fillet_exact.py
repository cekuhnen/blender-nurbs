"""Fillet tool on a curved face's edges: exact rolling-ball fillets
(worker._exact_tool_fillets) instead of the kernel's fitted ones:
    blender -b --factory-startup --python tests/test_fillet_exact.py"""
import json, os, sys, time
import bmesh
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import cvs, facecvs, store, subsel
from blender_nurbs.kernel import defeature

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
_sm = store._k()[0]

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def box(h=2.0):
    cu = bpy.data.curves.new("R", 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(3)
    for p, c in zip(sp.points, [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)]): p.co = (*c, 1.0)
    sp.use_cyclic_u = True
    R = bpy.data.objects.new("R", cu); ctx.collection.objects.link(R)
    bpy.ops.nproto.extrude(inputs=json.dumps([[R.name, "S0"]]), distance=h, direction="Z", body="NEW")
    return ctx.active_object

def brep(o): return store.get_brep(o)
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

def sculpt_top(B, fn):
    """The box top sculpted with Face CVs (fn moves the cage's CVs), Done."""
    for x in ctx.selected_objects: x.select_set(False)
    B.select_set(True); ctx.view_layer.objects.active = B
    subsel.set_picks(B, "FACE", [face_at(B, (2, 1.5, 2))])
    bpy.ops.nproto.edit_cvs()
    cage = ctx.active_object
    meta = json.loads(cage[cvs.CAGE_META])
    P = np.array(cvs.cage_points(cage), float).reshape(meta["nu"], meta["nv"], 3)
    fn(P)
    bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table()
    for v, q in zip(bm.verts, P.reshape(-1, 3)):
        v.co = q
    bmesh.update_edit_mesh(cage.data)
    ctx.view_layer.update()
    cvs._process()
    bpy.ops.nproto.cvs_done()

def fillet(B, pts, r):
    """Fillet tool on the edges nearest pts: (seconds, the job said exact)."""
    from blender_nurbs import jobs
    seen = {}
    run = jobs.run
    def spy(op, *a, **k):
        res = run(op, *a, **k)
        if op == "edge_fillet":
            seen.update(res[0])
        return res
    jobs.run = spy
    try:
        t0 = time.perf_counter()
        bpy.ops.nproto.edge_fillet(inputs=json.dumps([[B.name, f"E{edge_at(B, p)}"] for p in pts]),
                                   shape="CIRCULAR", radius=r)
        return time.perf_counter() - t0, bool(seen.get("exact"))
    finally:
        jobs.run = run

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
TOP = (2, 1.5, 2.3)
TOP4 = [(2, 0, 2), (2, 3, 2), (0, 1.5, 2), (4, 1.5, 2)]
def gentle(P): P[1:-1, 1:-1, 2] += 0.3
def steep(P): P[2:5, 1:4, 2] += 1.0

# --- the four edges of a gently domed top: exact
clear(); B = box(); sculpt_top(B, gentle)
n0 = brep(B).face_count()
dt, ex = fillet(B, TOP4, 0.4)
rk = rounds(B, 0.4, TOP, 0.01)
check("Fillet on a domed top's four edges: exact rolling-ball fillets, a closed solid", ex and solid(B)
      and brep(B).face_count() == n0 + 4 and rk[0] == 4 and rk[1] < 0.05, f"rounds / kink {rk}, {dt * 1000:.0f} ms")

# --- a steep dome (the kernel's came out open there): exact and closed
clear(); B = box(); sculpt_top(B, steep)
n0 = brep(B).face_count()
dt, ex = fillet(B, TOP4, 0.4)
rk = rounds(B, 0.4, TOP, 0.01)
check("... a steep dome too (the kernel's was open there)", ex and solid(B) and brep(B).face_count() == n0 + 4
      and rk[0] == 4 and rk[1] < 0.05, f"rounds / kink {rk}, {dt * 1000:.0f} ms")

# --- one edge: the fillet ends against the side walls
clear(); B = box(); sculpt_top(B, gentle)
n0 = brep(B).face_count()
dt, ex = fillet(B, [(2, 0, 2)], 0.4)
rk = rounds(B, 0.4, TOP, 0.01)
check("one edge of a domed top: exact, cut off by the side walls", ex and solid(B) and brep(B).face_count() == n0 + 1
      and rk[0] == 1 and rk[1] < 0.05, f"rounds / kink {rk}, {dt * 1000:.0f} ms")

# --- two edges meeting at a corner: one mitre
clear(); B = box(); sculpt_top(B, gentle)
n0 = brep(B).face_count()
dt, ex = fillet(B, [(2, 0, 2), (0, 1.5, 2)], 0.4)
rk = rounds(B, 0.4, TOP, 0.01)
check("two edges meeting at a corner: exact, one mitre", ex and solid(B) and brep(B).face_count() == n0 + 2
      and rk[0] == 2 and rk[1] < 0.05, f"rounds / kink {rk}, {dt * 1000:.0f} ms")

# --- a flat box: the kernel's (exact between planes already)
clear(); B = box()
dt, ex = fillet(B, TOP4, 0.4)
rk = rounds(B, 0.4, (2, 1.5, 2), 0.01)
check("a flat box top: the kernel's fillets (exact between planes)", not ex and solid(B) and brep(B).face_count() == 10
      and rk[0] == 4 and rk[1] < 0.05, f"rounds / kink {rk}")

# --- a vertical edge of a domed box (no curved face shared): the kernel's, unchanged
clear(); B = box(); sculpt_top(B, gentle)
n0 = brep(B).face_count()
dt, ex = fillet(B, [(4, 0, 1)], 0.4)
check("a vertical edge (between two flat walls): the kernel's", not ex and solid(B) and brep(B).face_count() == n0 + 1)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
