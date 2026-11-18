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
from blender_nurbs import scanfit as sf, scanpatch as sp, store, subd_brep as sb  # noqa: E402

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

# -- 8. Trim Cage to Curve: the open edge held to a closed curve drawn on the scan -------------
def on_capsule(t, z, out=0.0, rx=0.065, ry=0.05, h=0.26, z0=0.2):
    sc = 1.0 + 0.12 * np.sin(np.pi * (z - z0) / h)
    p = np.stack([rx * sc * np.cos(t), ry * sc * np.sin(t), z], 1)
    nrm = np.stack([np.cos(t) / rx, np.sin(t) / ry, np.zeros_like(t)], 1)
    return p + out * nrm / np.linalg.norm(nrm, axis=1)[:, None]


def curve_obj(name, P, cyclic=True):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('POLY')
    sp.points.add(len(P) - 1)
    for q, p in zip(sp.points, P):
        q.co = (p[0], p[1], p[2], 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu)
    ctx.collection.objects.link(o)
    return o


tt = np.linspace(0, 2 * np.pi, 48, endpoint=False)
trim_z = 0.2 + 0.17 + 0.05 * np.cos(tt)                      # dips 10 cm from front to back
trim = curve_obj("Trim", on_capsule(tt, trim_z, out=0.001))  # drawn 1 mm off the scan: projected onto it
for o in ctx.scene.objects:
    o.select_set(False)
scan_obj.select_set(True)
ctx.view_layer.objects.active = scan_obj
bpy.ops.nproto.scan_tube_cage(rings=8, segments=12)
tc = ctx.active_object
trim.select_set(True)
res = bpy.ops.nproto.scan_trim_cage()
ti = dict(sf.LAST_INFO)
Vt, Ft, sht, kct = sf.cage_of(tc)
smt = sf.Sampler(len(Vt), Ft, sht, kct, 3)
Pt = smt.points(Vt)
edge = sf.poly_border_loops(smt.Fs)[0]
check("Trim Cage to Curve: the open edge lies on the trim line (both ways within 0.3 mm), the surface on the scan",
      res == {'FINISHED'} and ti["edge_max"] < 0.3e-3 and ti["edge_along"] < 0.2e-3 and ti["dev_rms"] < 0.4e-3,
      f"edge {ti['edge_max'] * MM:.3f} mm ({ti['edge_along'] * MM:.3f} along the scan), RMS {ti['dev_rms'] * MM:.3f} mm")
check("the edge follows the line's 10 cm dip, the cage remembers its trim line",
      np.ptp(Pt[edge, 2]) > 0.095 and tc.get("nproto_trim") == "Trim", f"{np.ptp(Pt[edge, 2]) * MM:.1f} mm")
for o in ctx.scene.objects:
    o.select_set(False)
tc.select_set(True)
ctx.view_layer.objects.active = tc
res = bpy.ops.nproto.scan_fit_cage(smoothness=0.2)
check("Fit Cage to Scan afterwards keeps the edge on the trim line", res == {'FINISHED'}
      and sf.LAST_INFO.get("edge_max") is not None and sf.LAST_INFO["edge_max"] < 0.3e-3,
      f"{sf.LAST_INFO.get('edge_max', 1) * MM:.3f} mm")
res = bpy.ops.nproto.scan_refine_cage(loops=2, tolerance=0.0)
check("Refine Cage to Scan keeps it too (loops across the edge add edge vertices)", res == {'FINISHED'}
      and sf.LAST_INFO["edge_max"] < 0.3e-3 and sf.LAST_INFO["loops"] == 2, f"{sf.LAST_INFO['edge_max'] * MM:.3f} mm")
# edge_need: a face whose open edge runs off the line asks to be halved along that edge
Fe = [(0, 1, 2, 3), (1, 4, 5, 2), (3, 2, 6, 7), (2, 5, 8, 6)]
Ve = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0), (1, 2, 0), (0, 2, 0), (2, 2, 0)], float)
se = sf.Sampler(len(Ve), Fe, (), False, 2)
Pe_ = se.points(Ve)
lp = sf.poly_border_loops(se.Fs)[0]
miss = ((Pe_[lp, 1] < 0.02) & (Pe_[lp, 0] > 0.4) & (Pe_[lp, 0] < 0.95)).astype(float)   # face 0's bottom edge
eu, ev = sf.edge_need(Fe, 2, Pe_, se.Fs, lp, miss, 0.5)
check("edge_need: the face whose edge misses is asked to be halved along that edge, no other",
      eu[0] > 0 and ev[0] == 0 and not eu[1:].any() and not ev[1:].any(), (eu, ev))
res = bpy.ops.nproto.scan_fit_cage(hold_trim=False)
check("Hold Trim Line off: the fit is free of it", res == {'FINISHED'} and sf.LAST_INFO.get("edge_max") is None)
open_c = curve_obj("OpenTrim", on_capsule(tt[:30], trim_z[:30]), cyclic=False)
open_c.select_set(True)
Vb = np.array([v.co[:] for v in tc.data.vertices])
res = bpy.ops.nproto.scan_trim_cage()
check("an open curve is refused, the cage left as it was",
      res == {'CANCELLED'} and np.allclose(Vb, np.array([v.co[:] for v in tc.data.vertices])))
# a trim line with a drape offset: the edge sits the offset off the scan, over the line
for o in ctx.scene.objects:
    o.select_set(False)
tc.select_set(True)
trim.select_set(True)
ctx.view_layer.objects.active = tc
res = bpy.ops.nproto.scan_trim_cage(drape=True, offset=0.003)
Vt, Ft, sht, kct = sf.cage_of(tc)
smt = sf.Sampler(len(Vt), Ft, sht, kct, 3)
Pe = smt.points(Vt)[sf.poly_border_loops(smt.Fs)[0]]
se = sf.scan_for(scan_obj).signed(Pe)
check("Trim with Drape + 3 mm: the edge stands 3 mm off the scan, on the line pushed out",
      res == {'FINISHED'} and np.nanmax(np.abs(se - 0.003)) < 0.4e-3 and sf.LAST_INFO["edge_max"] < 0.3e-3,
      f"edge off the scan {np.nanmin(se) * MM:.2f}..{np.nanmax(se) * MM:.2f} mm")

# -- 9. Drape Patch to Scan: one NURBS patch, CVs moving straight down only -----------------
rng9 = np.random.default_rng(9)
Gv = np.array([(x, y, 0.5) for x in np.linspace(-0.3, 0.3, 6) for y in np.linspace(-0.2, 0.2, 6)])
Gf = [(i * 6 + j, (i + 1) * 6 + j, (i + 1) * 6 + j + 1, i * 6 + j + 1) for i in range(5) for j in range(5)]
perm = rng9.permutation(36)                          # the grid's vertices in any order
inv = np.argsort(perm)
Gshuf = Gv[perm]
Fshuf = [tuple(int(inv[a]) for a in f) for f in Gf]
G = sp.grid_order(36, Fshuf)
rows_ok = all(np.allclose(np.ptp(Gshuf[G[i], 0 if np.ptp(Gshuf[G[0], 1]) > np.ptp(Gshuf[G[0], 0]) else 1]), 0)
              for i in range(6))
check("grid_order: a shuffled 6 x 6 quad grid comes back as rows and columns", G.shape == (6, 6) and rows_ok)
try:
    sp.grid_order(5, [(0, 1, 2, 3), (1, 4, 2)])
    refused = False
except ValueError:
    refused = True
check("grid_order: anything but a quad grid is refused", refused)
check("degree: 6 CVs -> one span of degree 5, 8 CVs -> 3 spans of degree 5, 4 CVs -> degree 3",
      sp.degree_and_knots(6)[0] == 5 and len(sp.degree_and_knots(8)[1]) == 14 and sp.degree_and_knots(4)[0] == 3)


def height_scan(name, fz, gap=None, n=240):
    """A triangulated height field z = fz(x, y) over -0.4..0.4 x -0.3..0.3; gap: (x0, width, depth) groove."""
    xs, ys = np.linspace(-0.4, 0.4, n), np.linspace(-0.3, 0.3, n)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    Z = fz(X, Y)
    if gap is not None:
        x0, w, dep = gap
        Z = Z - dep * (np.abs(X - x0) < w / 2)
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    T = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            T += [(a, a + n, a + n + 1), (a, a + n + 1, a + 1)]
    m = bpy.data.meshes.new(name)
    m.from_pydata([tuple(p) for p in P], [], T)
    m.update()
    o = bpy.data.objects.new(name, m)
    ctx.collection.objects.link(o)
    return o


poly = lambda x, y: 0.2 + 0.1 * x * x - 0.05 * x * y + 0.03 * y ** 3          # degree <= 5: exactly a patch
hs = height_scan("PolyScan", poly)
C0 = Gv.reshape(6, 6, 3)
du, Ku = sp.degree_and_knots(6)
Cp, rp = sp.drape_patch(C0, du, du, Ku, Ku, sf.scan_for(hs), smoothness=0.0)
dp = sp.deviation(Cp, du, du, Ku, Ku, sf.scan_for(hs))
check("Drape Patch: a scan that is a degree-5 surface is met exactly (RMS < 2 µm), the CVs moved only in z",
      dp["dev_rms"] < 2e-6 and np.abs(Cp[..., :2] - C0[..., :2]).max() == 0.0,
      f"RMS {dp['dev_rms'] * 1e6:.2f} µm")
gs = height_scan("GapScan", poly, gap=(0.05, 0.012, 0.006))                    # a 12 mm wide, 6 mm deep panel gap
scg = sf.scan_for(gs)
Cf_, _ = sp.drape_patch(C0, du, du, Ku, Ku, scg, smoothness=0.0)
Cd_, _ = sp.drape_patch(C0, du, du, Ku, Ku, scg, smoothness=0.0, drape=True)
ts9 = np.linspace(0, 1, 60)
S9 = np.einsum('ai,bj->abij', sp.basis(6, du, Ku, ts9), sp.basis(6, du, Ku, ts9)).reshape(-1, 36)
Pf, Pd = S9 @ Cf_.reshape(-1, 3), S9 @ Cd_.reshape(-1, 3)
away = np.abs(Pf[:, 0] - 0.05) > 0.02
ef = np.abs(Pf[away, 2] - poly(Pf[away, 0], Pf[away, 1])).max()
ed = (Pd[:, 2] - poly(Pd[:, 0], Pd[:, 1]))
check("a panel gap: the fit lets go of it (within 0.3 mm of the panel away from the gap), Drape bridges it "
      "(never more than 0.3 mm below the panel's top)",
      ef < 0.3e-3 and ed.min() > -0.3e-3, f"fit {ef * MM:.3f} mm, drape lowest {ed.min() * MM:.3f} mm")
# the operator: a NURBS patch object, updated in place when run again
me9 = bpy.data.meshes.new("Sheet9")
me9.from_pydata([tuple(p) for p in Gshuf], [], Fshuf)
me9.update()
sheet9 = bpy.data.objects.new("Sheet9", me9)
ctx.collection.objects.link(sheet9)
for o in ctx.scene.objects:
    o.select_set(False)
hs.select_set(True)
sheet9.select_set(True)
ctx.view_layer.objects.active = sheet9
res = bpy.ops.nproto.scan_drape_patch(smoothness=0.0)
p9 = ctx.active_object

from blender_nurbs.kernel import load as load_kernel  # noqa: E402
d9 = load_kernel().blender_ext.surface_cvs(store.get_brep(p9).faces()[0].surface())
check("Drape Patch to Scan: one NURBS face, degree 5 x 5, 6 x 6 CVs, on the scan",
      res == {'FINISHED'} and store.is_nurbs(p9) and (d9["degree_u"], d9["degree_v"], d9["count_u"], d9["count_v"]) == (5, 5, 6, 6)
      and sp.LAST_INFO["dev_rms"] < 5e-6, f"{sp.LAST_INFO['dev_rms'] * 1e6:.2f} µm")
for o in ctx.scene.objects:
    o.select_set(False)
hs.select_set(True)
sheet9.select_set(True)
ctx.view_layer.objects.active = sheet9
n_before = len([o for o in ctx.scene.objects if store.is_nurbs(o)])
res = bpy.ops.nproto.scan_drape_patch(offset=0.002)
check("run again (Offset 2 mm): the same patch object is updated, 2 mm above the scan",
      res == {'FINISHED'} and ctx.active_object is p9 and len([o for o in ctx.scene.objects if store.is_nurbs(o)]) == n_before
      and abs(sp.LAST_INFO["dev_mean"]) < 0.1e-3, f"mean {sp.LAST_INFO['dev_mean'] * MM + 2:.3f} mm off")

# -- 10. Drape Patch: G0 / G1 / G2 to neighbouring patches -----------------------------------
from blender_nurbs import surfeval  # noqa: E402

wavy = lambda x, y: 0.2 + 0.08 * np.cos(3 * x) * np.cos(2 * y) + 0.02 * np.sin(5 * x + 1)
ws = height_scan("WavyScan", wavy)


def grid_sheet(name, xs, ys, z=0.6):
    V = [(x, y, z) for x in xs for y in ys]
    nu_, nv_ = len(xs), len(ys)
    F = [(i * nv_ + j, (i + 1) * nv_ + j, (i + 1) * nv_ + j + 1, i * nv_ + j + 1)
         for i in range(nu_ - 1) for j in range(nv_ - 1)]
    m = bpy.data.meshes.new(name)
    m.from_pydata(V, [], F)
    m.update()
    o = bpy.data.objects.new(name, m)
    ctx.collection.objects.link(o)
    return o


def drape(sheet_obj, nbs=(), **kw):
    for o in ctx.scene.objects:
        o.select_set(False)
    ws.select_set(True)
    sheet_obj.select_set(True)
    for o in nbs:
        o.select_set(True)
    ctx.view_layer.objects.active = sheet_obj
    r = bpy.ops.nproto.scan_drape_patch(**kw)
    return r, ctx.active_object, dict(sp.LAST_INFO)


def measured(pb, edge, pa):
    """Gap, normal angle (degrees) and normal-curvature difference across the edge (%), on the kernel surfaces."""
    SB = surfeval.Surf(store.get_brep(pb).faces()[0].surface())
    SA = surfeval.Surf(store.get_brep(pa).faces()[0].surface())
    t = np.linspace(0.02, 0.98, 30)
    z0, z1 = np.zeros(30), np.ones(30)
    uv = {"u0": (z0, t), "u1": (z1, t), "v0": (t, z0), "v1": (t, z1)}[edge]
    DB = SB.at(*uv, nd=2)
    fa = SA.feet(DB[0, 0])
    DA = SA.at(fa[:, 0], fa[:, 1], nd=2)
    nA, QA = sp.second_form(DA)
    nB, QB = sp.second_form(DB)
    c = DB[1, 0] if edge[0] == "u" else DB[0, 1]
    c = c - (c * nB).sum(1)[:, None] * nB
    kA = np.einsum('mi,mij,mj->m', c, QA, c) * np.sign((nA * nB).sum(1))
    kB = np.einsum('mi,mij,mj->m', c, QB, c)
    return (float(np.linalg.norm(DA[0, 0] - DB[0, 0], axis=1).max()),
            float(np.degrees(np.arccos(np.clip(np.abs((nA * nB).sum(1)), 0, 1))).max()),
            float(np.abs(kA - kB).max() / np.abs(kA).max() * 100))


sa_ = grid_sheet("SheetA", np.linspace(-0.35, 0.0, 6), np.linspace(-0.26, 0.26, 4))    # degree 3 along the edge
_, pa_, _ = drape(sa_, match='NONE')
ys_ = np.array([-0.2, -0.13, -0.05, 0.04, 0.12, 0.2])                                    # uneven, and 4 mm off in plan
sb_ = grid_sheet("SheetB", np.linspace(0.004, 0.33, 6), ys_)
res = {}
for lv in ('G0', 'G1', 'G2'):
    r_, pb_, ib_ = drape(sb_, (pa_,), match=lv)
    res[lv] = (r_, ib_["edges"], measured(pb_, ib_["edges"][0]["edge"], pa_) if ib_["edges"] else None)
g0, g1, g2 = res["G0"][2], res["G1"][2], res["G2"][2]
check("Match Neighbours: the edge along the neighbour's (4 mm off in plan, uneven CVs, part of a degree-3 edge) "
      "is found and held",
      all(r[0] == {'FINISHED'} and len(r[1]) == 1 and r[1][0]["edge"] == "u0" for r in res.values()), res["G2"][1])
check("G0: the edges meet (measured on the two NURBS surfaces: gap < 1 µm)", g0[0] < 1e-6, g0)
check("G1: ... tangent too (normals within 0.001°), G2: ... and the curvature across the same (within 0.01 %)",
      g1[0] < 1e-6 and g1[1] < 1e-3 and g2[0] < 1e-6 and g2[1] < 1e-3 and g2[2] < 0.01 and g1[2] > 10 * g2[2],
      f"G1 {g1}, G2 {g2}")
check("the report gives the same as measured", abs(res["G2"][1][0]["result"]["angle"] - g2[1]) < 1e-5
      and abs(res["G1"][1][0]["result"]["angle"] - g1[1]) < 1e-5, (res["G2"][1][0]["result"], g2))
# a corner: a neighbour along two edges (the lower one itself G2 to the left one)
sl_ = grid_sheet("SheetL", np.linspace(-0.35, 0.0, 6), np.linspace(-0.42, 0.2, 6))
_, pl_, _ = drape(sl_, match='NONE')
sc_ = grid_sheet("SheetC", np.linspace(0.004, 0.33, 6), np.linspace(-0.42, -0.2, 6))
_, pc_, ic_ = drape(sc_, (pl_,), match='G2')
sk_ = grid_sheet("SheetK", np.linspace(0.0, 0.33, 6), np.array([-0.196, -0.13, -0.05, 0.04, 0.12, 0.2]))
r_, pk_, ik_ = drape(sk_, (pl_, pc_), match='G2')
mk = [measured(pk_, e["edge"], bpy.data.objects[e["neighbour"]]) for e in ik_["edges"]]
check("a corner between two neighbours: both edges G2 to within 20 µm, 0.1° and 1 % (exact needs the neighbours' "
      "twists to agree)",
      r_ == {'FINISHED'} and len(ik_["edges"]) == 2 and all(m[0] < 2e-5 and m[1] < 0.1 and m[2] < 1.0 for m in mk), mk)

print("ALL OK" if ok else "SOME FAILED")
