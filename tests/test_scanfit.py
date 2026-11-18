"""Fit to Scan, headless:  blender -b --factory-startup --python tests/test_scanfit.py

A synthetic scan (an elliptic capsule, open at the top with a lid n-gon over
the cut and a small hole at its tip) stands in for a residual-limb scan."""
import math
import os
import sys

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs  # noqa: E402

blender_nurbs.register()
from blender_nurbs import scanfit as sf, store, subd_brep as sb  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)


ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
MM = 1000.0

# -- 1. the sparse Catmull-Clark matrices are SubD to BREP's rules --------------------------
rng = np.random.default_rng(1)
S, R = 12, 5
V = np.array([(math.cos(2 * math.pi * j / S), math.sin(2 * math.pi * j / S), 0.5 * i)
              for i in range(R) for j in range(S)]) + rng.normal(0, 0.05, (S * R, 3))
F = [(i * S + j, i * S + (j + 1) % S, (i + 1) * S + (j + 1) % S, (i + 1) * S + j) for i in range(R - 1) for j in range(S)]
Fa = np.array(F)
sharp = sf.with_borders(F, {(0, 1)} | set())
keys = np.sort(np.array([a * len(V) + b for a, b in sharp], np.int64))
V1, F1, k1, _, p1 = sb.cc_step(V, Fa, keys, np.zeros((len(Fa), 4, 2)))
V2, F2, k2, _, p2 = sb.cc_step(V1, F1, k1, np.zeros((len(F1), 4, 2)))
ref = sb.cc_limit(V2, F2, k2)
got = sf.Sampler(len(V), F, {(0, 1)}, False, levels=2).points(V)
check("limit matrix = subd_brep cc_step x2 + cc_limit (quads, a crease, open borders)",
      np.abs(got - ref).max() < 1e-12, np.abs(got - ref).max())
Vp = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0.2), (0, 1, 0), (1, 1, 0.3), (2, 1, 0), (0.5, 2, 0.1), (1.5, 2, 0)], float)
Fp = [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 6), (4, 5, 7, 6)]
Vr, Fr, sr = sb.refine_any(Vp, Fp, sf.with_borders(Fp, set()), keep_corners=True)
Fr = np.array(Fr)
nf = np.bincount(np.array([a for f in Fp for a in f]), minlength=len(Vp))
pin = np.r_[nf == 1, np.zeros(len(Vr) - len(Vp), bool)]
ref = sb.cc_limit(Vr, Fr, np.sort(np.array([a * len(Vr) + b for a, b in sr], np.int64)), pin)
got = sf.Sampler(len(Vp), Fp, (), True, levels=1).points(Vp)
check("limit matrix with a triangle and Keep Corners = refine_any + cc_limit", np.abs(got - ref).max() < 1e-12,
      np.abs(got - ref).max())
A = sf.Sp(rng.integers(0, 7, 30), rng.integers(0, 5, 30), rng.normal(size=30), (7, 5))
B = sf.Sp(rng.integers(0, 5, 20), rng.integers(0, 6, 20), rng.normal(size=20), (5, 6))
check("sparse product", np.abs(sf.spmm(A, B).dense() - A.dense() @ B.dense()).max() < 1e-12)


# -- 2. a synthetic scan: elliptic capsule, open top closed by a lid n-gon, a hole at the tip -----
def capsule(rx=0.065, ry=0.05, h=0.26, r_tip=0.055, nu=96, nv=60, z0=0.2, ridge=0.0):
    """Triangles of an elliptic tube (bulging a little at mid height) on a half-ellipsoid end;
    ridge: the height of a narrow ridge running along the tube's side (at angle 0)."""
    pts, idx = [], {}
    lat = np.linspace(-0.5 * math.pi, 0.0, nv // 3)[1:]          # the end (pole left out: a hole)
    zs = np.linspace(0.0, h, nv)[1:]
    rows = [(math.cos(a), r_tip * math.sin(a)) for a in lat] + [(1.0 + 0.12 * math.sin(math.pi * z / h), z) for z in zs]
    for i, (sc, z) in enumerate(rows):
        for j in range(nu):
            t = 2 * math.pi * j / nu
            idx[(i, j)] = len(pts)
            dt = math.atan2(math.sin(t), math.cos(t))
            k = 1.0 + (ridge / rx) * math.exp(-(dt / 0.18) ** 2) * (z > 0)
            pts.append((rx * sc * k * math.cos(t), ry * sc * k * math.sin(t), z0 + z))
    tris = []
    for i in range(len(rows) - 1):
        for j in range(nu):
            a, b, c, d = idx[(i, j)], idx[(i, (j + 1) % nu)], idx[(i + 1, (j + 1) % nu)], idx[(i + 1, j)]
            tris += [(a, b, c), (a, c, d)]
    top = [idx[(len(rows) - 1, j)] for j in range(nu)]
    return np.array(pts), tris, top


P, tris, top = capsule()
me = bpy.data.meshes.new("SynthScan")
me.from_pydata([tuple(p) for p in P], [], tris + [tuple(top)])     # + the lid over the cut
me.update()
scan_obj = bpy.data.objects.new("SynthScan", me)
ctx.collection.objects.link(scan_obj)
scan = sf.scan_for(scan_obj)
loops = sf.border_loops(scan.V, scan.T)
check("the lid n-gon is left out: the cut is the longest open border, the tip hole a short one",
      len(loops) == 2 and len(loops[0][1]) == 96 and loops[0][0] > 10 * loops[1][0], [(round(l, 3), len(i)) for l, i in loops])

for o in ctx.scene.objects:
    o.select_set(False)
scan_obj.select_set(True)
ctx.view_layer.objects.active = scan_obj
res = bpy.ops.nproto.scan_tube_cage(rings=8, segments=12)
cage = ctx.active_object
info = dict(sf.LAST_INFO)
zc = np.array([v.co.z for v in cage.data.vertices])
check("Tube Cage from Scan: a SubD cage of 8 rings x 12 + a 2 x 2 grid cap, active, Subdivision on",
      res == {'FINISHED'} and len(cage.data.vertices) == 8 * 12 + 4 and len(cage.data.polygons) == 7 * 12 + 9
      and sf.is_subd(cage), (len(cage.data.vertices), len(cage.data.polygons)))
check("the cage reaches from the closed end up to just below the cut (not into the tip hole)",
      zc.min() < 0.2 + 0.01 and 0.2 + 0.26 * 0.9 < zc.max() < 0.2 + 0.26, (zc.min(), zc.max()))
check("the fitted surface lies on the scan (RMS < 0.3 mm, 95 % within 0.6 mm)",
      info["dev_rms"] < 0.3e-3 and info["dev_p95"] < 0.6e-3, f"{info['dev_rms'] * MM:.3f} / {info['dev_p95'] * MM:.3f} mm")
V0, F0, sh0, kc0 = sf.cage_of(cage)
sv = sf.Sampler(len(V0), F0, sh0, kc0, 3)
d_vertices_on = sf.deviation_stats(sf.Sampler(len(V0), F0, sh0, kc0, 3).points(
    sf.tube_cage(scan, rings=8, segments=12)[0]), scan, Fs=sv.Fs)
check("fitting beats a cage with its vertices on the scan (the SubD shrinks inside it) by 5x",
      d_vertices_on["dev_rms"] > 5 * info["dev_rms"], f"{d_vertices_on['dev_rms'] * MM:.2f} mm unfitted")

# deviation overlay data on the SubD cage
Pd, Nd, Td, Vd = sf.deviation_mesh(cage, scan_obj)
st = sf.STATS[cage.name]
check("Scan Deviation on the SubD model: values per vertex, stats as the fit's",
      len(Vd) == len(Pd) and abs(st["dev_rms"] - info["dev_rms"]) < 0.05e-3, f"{st['dev_rms'] * MM:.3f} mm")
check("the overlay arrays are C-contiguous (GPU buffers refuse anything else)",
      all(a.flags.c_contiguous for a in (Pd, Nd, Td, Vd)))
check("the overlay was turned on, with the scan as the scene's Scan",
      ctx.window_manager.nproto_view.show_scan_dev and ctx.scene.nproto_settings.scan_object == scan_obj)

# -- 3. SubD to BREP of the fitted cage measures the same ----------------------------------
for o in ctx.scene.objects:
    o.select_set(False)
cage.select_set(True)
ctx.view_layer.objects.active = cage
res = bpy.ops.nproto.subd_to_brep()
body = ctx.active_object
sf.deviation_mesh(body, scan_obj)
sb_ = sf.STATS[body.name]
check("the NURBS body from the fitted cage is as close to the scan as the SubD (area weighted)",
      res == {'FINISHED'} and store.is_nurbs(body) and abs(sb_["dev_rms"] - st["dev_rms"]) < 0.1 * st["dev_rms"] + 2e-5,
      f"{sb_['dev_rms'] * MM:.3f} vs {st['dev_rms'] * MM:.3f} mm")

# -- 4. Drape with a clearance: never inside the offset scan by much ------------------------
for o in ctx.scene.objects:
    o.select_set(False)
scan_obj.select_set(True)
ctx.view_layer.objects.active = scan_obj
bpy.ops.nproto.scan_tube_cage(rings=8, segments=12, drape=True, offset=0.003)
h = ctx.active_object
dinfo = dict(sf.LAST_INFO)
check("Drape + 3 mm: the surface stays outside the scan (at least 2.5 mm off) and about 3 mm off on average",
      dinfo["dev_min"] > -0.5e-3 and abs(dinfo["dev_mean"]) < 0.5e-3,
      f"min {dinfo['dev_min'] * MM + 3:.2f} mm, mean {dinfo['dev_mean'] * MM + 3:.2f} mm")

# -- 5. Fit Cage to Scan in Edit Mode moves only the selected vertices -----------------------
for o in ctx.scene.objects:
    o.select_set(False)
h.select_set(True)
ctx.view_layer.objects.active = h
Vb = np.array([v.co[:] for v in h.data.vertices])
sel = Vb[:, 2] < np.median(Vb[:, 2])
h.data.polygons.foreach_set("select", np.zeros(len(h.data.polygons), bool))    # (selected faces and edges
h.data.edges.foreach_set("select", np.zeros(len(h.data.edges), bool))          # would select their vertices)
h.data.vertices.foreach_set("select", sel)
bpy.ops.object.mode_set(mode='EDIT')
res = bpy.ops.nproto.scan_fit_cage(offset=0.0, drape=False)
bpy.ops.object.mode_set(mode='OBJECT')
Va = np.array([v.co[:] for v in h.data.vertices])
moved = np.linalg.norm(Va - Vb, axis=1)
check("Edit Mode: the unselected vertices stay, the selected ones move (scan from the cage's own link)",
      res == {'FINISHED'} and moved[~sel].max() < 1e-9 and moved[sel].max() > 1e-4,
      f"{moved[~sel].max():.1e} / {moved[sel].max() * MM:.2f} mm")

# -- 6. a modifier above Subdivision: refused, nothing moved ---------------------------------
h.modifiers.new("Disp", 'DISPLACE')
h.modifiers.move(len(h.modifiers) - 1, 0)
Vb = np.array([v.co[:] for v in h.data.vertices])
res = bpy.ops.nproto.scan_fit_cage()
Va = np.array([v.co[:] for v in h.data.vertices])
check("a modifier above Subdivision: the fit refuses and leaves the cage", res == {'CANCELLED'} and np.allclose(Va, Vb))

# -- 7. Refine Cage to Scan: loops where the fit misses, across the way the miss varies --------
uv_par, uv = sf.face_grid([(0, 1, 2, 3), (1, 4, 5, 2), (0, 3, 6)], 3)
inner = uv_par == 0
cells = set(map(tuple, np.round(uv[inner] * 16).astype(int)))
check("face_grid: a quad's 64 level-3 sub-quads tile its unit square once each, non-quads have no (u, v)",
      inner.sum() == 64 and cells == {(2 * i + 1, 2 * j + 1) for i in range(8) for j in range(8)}
      and np.isnan(uv[uv_par == 2]).all())
# on a flat regular grid the limit surface is the grid itself: (u, v) must match the sub-quads' places
G = 5
Vg = np.array([(i, j, 0.0) for j in range(G) for i in range(G)], float)
Fg = [(j * G + i, j * G + i + 1, (j + 1) * G + i + 1, (j + 1) * G + i) for j in range(G - 1) for i in range(G - 1)]
sg = sf.Sampler(len(Vg), Fg, (), False, 3)
Pg = sg.points(Vg)
par, uvg = sf.face_grid(Fg, 3)
mid = 1 * (G - 1) + 1                               # an inner face: corner 0 at (1, 1)
sel_ = par == mid
cen = Pg[sg.Fs[sel_]].mean(1)[:, :2]
check("face_grid's (u, v) are where the sub-quads lie (flat grid, inner face)",
      np.abs(cen - (np.array([1.0, 1.0]) + uvg[sel_])).max() < 1e-9, np.abs(cen - (1 + uvg[sel_])).max())

Pr, tr, topr = capsule(ridge=0.004, nu=144, nv=80)
mr = bpy.data.meshes.new("RidgeScan")
mr.from_pydata([tuple(p) for p in Pr], [], tr)
mr.update()
ridge_obj = bpy.data.objects.new("RidgeScan", mr)
ctx.collection.objects.link(ridge_obj)
for o in ctx.scene.objects:
    o.select_set(False)
ridge_obj.select_set(True)
ctx.view_layer.objects.active = ridge_obj
bpy.ops.nproto.scan_tube_cage(rings=6, segments=8)
rc = ctx.active_object
before = dict(sf.LAST_INFO)
n0 = len(rc.data.vertices)
res = bpy.ops.nproto.scan_refine_cage(loops=1)
V1 = np.array([v.co[:] for v in rc.data.vertices])
new = V1[n0:]
ang = np.degrees(np.arctan2(new[:, 1] / 0.05, new[:, 0] / 0.065))
side = (new[:, 2] > 0.25) & (np.abs(ang) < 90)             # up the ridge's side (the loop runs on over the cap
check("Refine: a ridge running along the scan gets a loop along it, next to the ridge "   # and up the far side)
      "(not a ring around)",
      res == {'FINISHED'} and np.ptp(new[:, 2]) > 0.15 and side.sum() >= 3 and np.abs(ang[side]).max() < 30,
      f"{len(new)} new, z span {np.ptp(new[:, 2]) * MM:.0f} mm, ridge side at {np.round(ang[side])}°")
res = bpy.ops.nproto.scan_refine_cage(loops=4)
after = dict(sf.LAST_INFO)
check("Refine: 5 loops take the RMS down by half, the cage stays all quads",
      res == {'FINISHED'} and after["dev_rms"] < 0.5 * before["dev_rms"]
      and all(len(p.vertices) == 4 for p in rc.data.polygons),
      f"{before['dev_rms'] * MM:.2f} -> {after['dev_rms'] * MM:.2f} mm, {n0} -> {len(rc.data.vertices)} vertices")
res = bpy.ops.nproto.scan_refine_cage(loops=3, tolerance=0.05)
check("Refine: nothing beyond the tolerance, no loop added", res == {'FINISHED'} and sf.LAST_INFO["loops"] == 0)
for o in ctx.scene.objects:
    o.select_set(False)
rc.select_set(True)
ctx.view_layer.objects.active = rc
res = bpy.ops.nproto.subd_to_brep()
rb = ctx.active_object
sf.deviation_mesh(rb, ridge_obj)
check("Refine: SubD to BREP of the refined cage measures as the SubD",
      res == {'FINISHED'} and store.is_nurbs(rb) and abs(sf.STATS[rb.name]["dev_rms"] - after["dev_rms"]) < 0.15 * after["dev_rms"],
      f"{sf.STATS[rb.name]['dev_rms'] * MM:.3f} vs {after['dev_rms'] * MM:.3f} mm")
# Edit Mode: only the selected faces get loops, the unselected vertices stay
for o in ctx.scene.objects:
    o.select_set(False)
rc.select_set(True)
ctx.view_layer.objects.active = rc
Vb = np.array([v.co[:] for v in rc.data.vertices])
sel = Vb[:, 2] < np.median(Vb[:, 2])
rc.data.polygons.foreach_set("select", np.zeros(len(rc.data.polygons), bool))
rc.data.edges.foreach_set("select", np.zeros(len(rc.data.edges), bool))
rc.data.vertices.foreach_set("select", sel)
bpy.ops.object.mode_set(mode='EDIT')
res = bpy.ops.nproto.scan_refine_cage(loops=2, tolerance=0.0)
bpy.ops.object.mode_set(mode='OBJECT')
Va = np.array([v.co[:] for v in rc.data.vertices])
moved = np.linalg.norm(Va[:len(Vb)] - Vb, axis=1)
check("Refine in Edit Mode: the unselected vertices stay, loops are added",
      res == {'FINISHED'} and moved[~sel].max() < 1e-9 and len(Va) > len(Vb), f"{moved[~sel].max():.1e}, +{len(Va) - len(Vb)}")

print("ALL OK" if ok else "SOME FAILED")
