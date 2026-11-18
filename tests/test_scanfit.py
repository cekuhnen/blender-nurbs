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
def capsule(rx=0.065, ry=0.05, h=0.26, r_tip=0.055, nu=96, nv=60, z0=0.2):
    """Triangles of an elliptic tube (bulging a little at mid height) on a half-ellipsoid end."""
    pts, idx = [], {}
    lat = np.linspace(-0.5 * math.pi, 0.0, nv // 3)[1:]          # the end (pole left out: a hole)
    zs = np.linspace(0.0, h, nv)[1:]
    rows = [(math.cos(a), r_tip * math.sin(a)) for a in lat] + [(1.0 + 0.12 * math.sin(math.pi * z / h), z) for z in zs]
    for i, (sc, z) in enumerate(rows):
        for j in range(nu):
            t = 2 * math.pi * j / nu
            idx[(i, j)] = len(pts)
            pts.append((rx * sc * math.cos(t), ry * sc * math.sin(t), z0 + z))
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

print("ALL OK" if ok else "SOME FAILED")
