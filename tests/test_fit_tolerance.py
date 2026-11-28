"""Remove Edges with a Fit Tolerance: tangent faces on different surfaces merged
onto one NURBS surface fitted through them (kernel/surfjoin.py):
    blender -b --factory-startup --python tests/test_fit_tolerance.py"""
import json, math, os, sys, tempfile, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, steps, store, subsel
from blender_nurbs.kernel import load
from blender_nurbs.kernel import mergefaces as mf
from blender_nurbs.kernel import surfjoin as sj
from blender_nurbs.kernel.classa import basis

sm = load()
ext = sm.blender_ext
ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
TMP = tempfile.mkdtemp(prefix="nproto_fit_")
TOL = 1e-3

# --- bodies ------------------------------------------------------------------------------
def bump(x, y): return 0.3 * np.sin(1.3 * x + 0.4) * np.cos(1.7 * y) + 0.1 * x * y

def patch(x0, x1, y0, y1, nu, nv, warp=lambda t: t):
    """A bicubic least-squares patch of the height field over [x0, x1] x [y0, y1],
    its v parameter warped (the same shape, parameterized another way)."""
    p = 3
    Ku = np.r_[[0] * 4, np.linspace(0, 1, nu - 2)[1:-1], [1] * 4]
    Kv = np.r_[[0] * 4, np.linspace(0, 1, nv - 2)[1:-1], [1] * 4]
    s = np.linspace(0, 1, 60)
    G = np.stack(np.meshgrid(x0 + (x1 - x0) * s, y0 + (y1 - y0) * warp(s), indexing="ij"), -1)
    P = np.concatenate([G, bump(G[..., 0], G[..., 1])[..., None]], -1)
    C = np.linalg.lstsq(basis(Ku, p, s), P.reshape(60, -1), rcond=None)[0].reshape(nu, 60, 3).transpose(1, 0, 2)
    C = np.linalg.lstsq(basis(Kv, p, s), C.reshape(60, -1), rcond=None)[0].reshape(nv, nu, 3).transpose(1, 0, 2)
    return ext.surface_from_cvs(p, p, [tuple(map(float, c)) for c in C.reshape(-1, 3)], [float(k) for k in Ku],
                                [float(k) for k in Kv], None)

def own(s):
    """A copy of surface s: a brep made from a surface owns it, so each brep gets its own."""
    return mf._copy_surface(ext, s)

def joined(faces, solid=False):
    b = ext.brep_from_surface(own(faces[0]))
    for s in faces[1:]:
        ext.append_brep(b, ext.brep_from_surface(own(s)))
    return sm.stitch_advanced(b, 2e-4, squeeze_small_edges=False, making_manifold_solid=solid)[0]

def cvs(s):
    d = ext.surface_cvs(s)
    return np.array(d["points"], float).reshape(d["count_u"], d["count_v"], 3), d

def wall(P, K, z=-1.0):
    """The flat wall under a top boundary (a row of cubic CVs P on knots K)."""
    return ext.surface_from_cvs(3, 1, [q for p in P for q in (tuple(map(float, p)), (float(p[0]), float(p[1]), z))],
                                [float(k) for k in K], [0, 0, 1, 1], None)

def quad(*c):
    return ext.surface_from_cvs(1, 1, [tuple(map(float, q)) for q in c], [0, 0, 1, 1], [0, 0, 1, 1], None)

def topped_block(SA, SB, z=-1.0):
    """A block over [0, 2] x [0, 1] whose top is the two patches SA (x < 1), SB (x > 1)."""
    PA, dA = cvs(SA); PB, dB = cvs(SB)
    F = [SA, SB, wall(PA[:, 0], dA["knots_u"]), wall(PB[:, 0], dB["knots_u"]),
         wall(PA[:, -1], dA["knots_u"]), wall(PB[:, -1], dB["knots_u"]),
         wall(PA[0, :], dA["knots_v"]), wall(PB[-1, :], dB["knots_v"]),
         quad((0, 0, z), (0, 1, z), (1, 0, z), (1, 1, z)), quad((1, 0, z), (1, 1, z), (2, 0, z), (2, 1, z))]
    return joined(F, solid=True)

def ring(z0, z1, rs, rot=0):
    """A ring along z (cubic) closed round (rational quadratic circle), radius rs over z;
    rot: its seam a quarter turn further per 2."""
    w = math.cos(math.pi / 4)
    circ = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    W = [1, w, 1, w, 1, w, 1, w]
    circ, W = circ[rot:] + circ[:rot], W[rot:] + W[:rot]
    circ, W = circ + circ[:1], W + W[:1]
    pts, wts = [], []
    for z, r in zip(np.linspace(z0, z1, len(rs)), rs):
        for (x, y), ww in zip(circ, W):
            pts.append((r * x, r * y, float(z))); wts.append(ww)
    Ku = [0] * 4 + list(np.linspace(0, 1, len(rs) - 2)[1:-1]) + [1] * 4
    return ext.surface_from_cvs(3, 2, pts, [float(k) for k in Ku],
                                [0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1], wts)

def deviation(srcs, out, n=15):
    """Largest distance from samples of the source surfaces to the result's faces."""
    worst, fs = 0.0, out.faces()
    for s in srcs:
        g = ext.brep_from_surface(own(s)).faces()[0]
        d = ext.surface_cvs(s)
        for u in np.linspace(d["knots_u"][0], d["knots_u"][-1], n):
            for v in np.linspace(d["knots_v"][0], d["knots_v"][-1], n):
                q = tuple(map(float, g.point_at_uv(u, v)))
                worst = max(worst, min(float(f.closest_point(q)[1]) for f in fs))
    return worst

def vol_b(b): return abs(b.volume(1e-6))

A = patch(0, 1, 0, 1, 6, 6)
B = patch(1, 2, 0, 1, 7, 8)                                    # other knots, other CV count
Bw = patch(1, 2, 0, 1, 7, 8, warp=lambda t: t + 0.2 * t * (1 - t))   # and its v parameter warped
MID = [1.0, 0.5, float(bump(1.0, 0.5))]

# --- kernel ------------------------------------------------------------------------------
sh = joined([A, B])
out, rep = mf.merge(sm, sh, [MID])
check("split sheet, other knots, no tolerance: kept, the reason names the Fit Tolerance",
      out is None and rep["refused"] and "Fit Tolerance" in rep["refused"][0]["why"], str(rep))
out, rep = mf.merge(sm, sh, [MID], fit_tol=TOL)
check("split sheet, Fit Tolerance: one face within it", out is not None and out.face_count() == 1
      and rep["fitted"] == 1 and 0 < rep["deviation"] <= TOL and deviation([A, B], out) <= TOL,
      f"{rep} dev {out and deviation([A, B], out)}")

sh = joined([A, Bw])
out, rep = mf.merge(sm, sh, [MID], fit_tol=TOL)
check("parameterized differently along the seam: one face within the tolerance",
      out is not None and out.face_count() == 1 and deviation([A, Bw], out) <= TOL, str(rep))

blk = topped_block(A, Bw)
v0 = vol_b(blk)
t = time.time()
out, rep = mf.merge(sm, blk, [MID], fit_tol=TOL)
dt = time.time() - t
check("solid with a two-patch top: one top face, still a solid, volume within area x tolerance",
      out is not None and out.face_count() == 9 and out.is_manifold_solid()
      and abs(vol_b(out) - v0) <= 2 * TOL and rep["vertices"] == 0, f"{rep} {out and vol_b(out)} {v0}")
check("  quick (under 2 s)", dt < 2.0, f"{dt:.2f}s")
out, rep = mf.merge(sm, blk, None, fit_tol=TOL)
check("Remove Redundant Edges + Fit Tolerance: coplanar walls merged and the smooth top refitted",
      out is not None and out.face_count() == 6 and out.is_manifold_solid() and rep["fitted"] == 1, str(rep))
out, rep = mf.merge(sm, blk, None)
check("Remove Redundant Edges, no tolerance: walls only, the top stays two faces",
      out is not None and out.face_count() == 7 and rep["fitted"] == 0, str(rep))

Q = [patch(0, 1, 0, 0.5, 5, 5), patch(1, 2, 0, 0.5, 6, 6, lambda t: t + 0.15 * t * (1 - t)),
     patch(0, 1, 0.5, 1, 6, 6, lambda t: t - 0.1 * t * (1 - t)), patch(1, 2, 0.5, 1, 5, 5)]
grid = joined(Q)
picks = [[1.0, 0.25, float(bump(1, 0.25))], [1.0, 0.75, float(bump(1, 0.75))],
         [0.5, 0.5, float(bump(0.5, 0.5))], [1.5, 0.5, float(bump(1.5, 0.5))]]
out, rep = mf.merge(sm, grid, picks, fit_tol=TOL)
check("2 x 2 grid of patches, each its own way: one face within the tolerance",
      out is not None and out.face_count() == 1 and rep["fitted"] == 4 and deviation(Q, out) <= TOL, str(rep))

slot = sm.boolean_union(sm.create_box((0, -0.5, 0), 1, 1, 0.4), sm.create_cylinder((1, 0, 0), 0.5, 0.4))
out, rep = mf.merge(sm, slot, [(1, 0.5, 0.2), (1, -0.5, 0.2)], fit_tol=TOL)
check("slot: flat sides onto the round end (tangent) refitted, a solid of the same volume",
      out is not None and out.face_count() == slot.face_count() - 2 and out.is_manifold_solid()
      and abs(vol_b(out) - vol_b(slot)) < 1e-4 and rep["fitted"] == 2, f"{rep} {out and vol_b(out)}")
out, rep = mf.merge(sm, slot, None, fit_tol=TOL)
check("slot, Remove Redundant Edges + Fit Tolerance: a G1 joint (curvature jumps) is left alone",
      out is None or rep["fitted"] == 0, str(rep))
t = time.time()
out, rep = mf.merge(sm, slot, [(1, 0.5, 0.2)], fit_tol=1e-9)
check("slot, a tolerance no fit reaches: kept, says why", out is None and rep["refused"]
      and rep["refused"][0]["why"].startswith("surface refit: the fit"), f"{rep} {time.time() - t:.1f}s")

R1, R2 = ring(0, 1, [1, 1, 1, 1]), ring(1, 2, [1, 1, 1.05, 1.1, 1.12], rot=2)
rings = joined([R1, R2])
out, rep = mf.merge(sm, rings, [(0.0, 1.0, 1.0)], fit_tol=TOL)
check("two rings, one flaring out tangentially, seams a quarter turn apart: one closed face",
      out is not None and out.face_count() == 1 and deviation([R1, R2], out) <= TOL, str(rep))

A3 = patch(0, 1, 0, 1, 6, 6)
Pc, dc = cvs(patch(1, 2, 0, 1, 6, 6))
Pc[..., 2] += 0.035 * (Pc[..., 0] - 1)                         # tilted by 2°: a crease at x = 1
B3 = ext.surface_from_cvs(3, 3, [tuple(map(float, c)) for c in Pc.reshape(-1, 3)], dc["knots_u"], dc["knots_v"], None)
out, rep = mf.merge(sm, joined([A3, B3]), [MID], fit_tol=1e-2)
check("a 2° crease: never refitted, says crease", out is None and rep["refused"]
      and "crease" in rep["refused"][0]["why"], str(rep))

# the angle and curvature checks themselves
body = mf._Body(sm, slot)
e = min(slot.edges(), key=lambda g: g.closest_point((1, 0.5, 0.2))[2])
check("slot edge: tangent (angle ~0), curvature jumps (G1)", sj.tangent_angle(body, e) is not None
      and sj.curvature_jump(body, e) > 0.5, f"{sj.tangent_angle(body, e)} {sj.curvature_jump(body, e)}")
sh = joined([A, B])
body = mf._Body(sm, sh)
e = min(sh.edges(), key=lambda g: g.closest_point(tuple(MID))[2])
check("split smooth sheet: curvature continuous", sj.curvature_jump(body, e) < sj.G2_REL,
      f"{sj.curvature_jump(body, e)}")

# exact reseam of a closed direction
net, _ = sj._net(ext, R1)
rs = sj.reseam(net, 1, 0.3)
t_ = np.linspace(0, 1, 7)
x = np.linspace(0, 1, 50)
d = max(float(np.linalg.norm(sj._grid(rs, [u], x)[0] - sj._grid(net, [u], (x + 0.3) % 1.0)[0], axis=1).max())
        for u in t_) if rs is not None else 1.0
check("reseam: the same ring, started 0.3 further round (exact)", d < 1e-9, f"{d}")

# --- the operator ------------------------------------------------------------------------
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def new_body(name, b):
    p = os.path.join(TMP, name + ".sm")
    b.write_to_file(p, ascii=True)
    with open(p) as f:
        return store.new_object_text(ctx, name, f.read(), None)

def select(o):
    for x in ctx.selected_objects: x.select_set(False)
    o.select_set(True)
    ctx.view_layer.objects.active = o

def pick(o, pts):
    polys = subsel._edge_polys(o)
    subsel.set_picks(o, "EDGE", [int(np.argmin([subsel._dist_poly(P, np.array(p, float)) if P is not None else np.inf
                                                 for P in polys])) for p in pts])

def run(**kw):
    try:
        return bpy.ops.nproto.remove_edges(**kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

def nfaces(o): return store.get_brep(o).face_count()

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"
    clear()
    o = new_body("Top", blk)
    select(o)
    pick(o, [MID])
    r = run()
    check(f"[{tag}] no Fit Tolerance: finished for F9, the body unchanged, the pick kept",
          r == {'FINISHED'} and nfaces(o) == 10 and len(subsel.picks(o, "EDGE")) == 1, f"{r} {nfaces(o)}")
    r = run(fit_tolerance=TOL)
    check(f"[{tag}] Fit Tolerance: one top face, a solid", r == {'FINISHED'} and nfaces(o) == 9
          and store.get_brep(o).is_manifold_solid() and not subsel.picks(o, "EDGE"), f"{r} {nfaces(o)}")
    if hist:
        st = features.stack(o)
        lab = steps.LABEL["merge_faces"](st[-1]["params"])
        check("history: one step, the label shows the tolerance", [x["op"] for x in st] == ["merge_faces"]
              and lab == "Remove Edges · 1 picked · fit 0.001 m", lab)
        features.rebuild(o, ctx, force=True)
        check("history: rebuilds to one top face", nfaces(o) == 9, f"{nfaces(o)}")
        raw = json.loads(o[features.PROP_STACK]); raw[-1]["params"]["fit_tolerance"] = 0.0
        o[features.PROP_STACK] = json.dumps(raw)
        features.rebuild(o, ctx, force=True)
        check("history: tolerance set to 0, the two top faces again", nfaces(o) == 10, f"{nfaces(o)}")
    else:
        check("no history: no steps recorded", not features.stack(o))

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
