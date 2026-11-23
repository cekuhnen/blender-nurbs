"""Mesh to BREP: a mesh body a designer modelled -- flat faces, bevelled
edges, holes, cylinders -- rebuilt as a NURBS body by recognising what was
modelled and doing it again with the kernel, not by copying facets.

* Coplanar facets merge into planar regions. Only facet corners count: a
  point a knife line left on a straight edge is not one.
* A strip of regions whose normals stay square to one axis, whose corners lie
  on one circle about it and which is tangent to exactly two planes is a
  fillet. Blender's bevel at Profile 0.5 samples that circle, so the fitted
  radius is the bevel's (any number of segments from 2). A symmetric chamfer
  can fit a circle by chance; it is not tangent and stays a plane.
* A closed ring of such regions is a hole (facing its axis) or a cylinder.
* A boxy bevel (Profile above 0.5) is a strip of evenly wide regions square
  to one axis between two planes whose corners lie on one superellipse
  (1 - x/w1)^r + (1 - y/w2)^r = 1 in the corner's frame (Blender's Profile p
  is r = -2 ln 2 / ln p). Its setbacks are fitted with the exponent, so the
  first segments, which lie nearly flat on the faces, may merge into them.
  It needs 4 segments: with fewer, any polyline fits some exponent. It is
  rebuilt where it ends on flat faces square to it: the faces beside it stop
  at its rails, the end faces get its profile, and the strip is the profile
  swept along the edge -- a B-spline fitted to the exact curve whose first
  and last three control points lie on the faces (G2). Elsewhere (a corner
  patch with another boxy bevel) it keeps its facets.
* The patch where three fillets meet (corners on their sphere) is left for
  the kernel to make again.

The body is rebuilt sharp: each plane's outline is walked, continuing across
a fillet onto the plane beyond it, with its corners where three planes meet.
Then cylinders are added, holes cut and each radius filleted with the
kernel's rolling-ball fillet (kernel/worker.py, job "mesh_to_brep"). The
fillets become Edge Fillet steps in the object's history, so their radius
stays editable. What is not recognised keeps its facets; if the rebuild
fails, the body is made of the mesh's flat regions."""

import math

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty
from mathutils import Matrix, Vector

from . import features, jobs, store

FLAT_COS = math.cos(math.radians(0.5))      # facets this close in direction (and offset) are one plane
SEED_COS = math.cos(math.radians(60.0))     # neighbouring facets of a round turn by less than this
SQUARE = 1e-3                               # |n . axis| below this: a facet is square to a round's axis
RING_TILT = math.sin(math.radians(1.0))     # a round grows over facets tilted less than this (its circle fit decides:
                                            # Solidify's corners leave a shell's outer wall a few hundredths off)
BOXY_SEGMENTS = 4                           # a boxy bevel needs this many segments (fewer: any polyline fits)
BOXY_COS = math.cos(math.radians(88.0))     # a boxy profile's middle segment can fold by up to this
GRID_COS = math.cos(math.radians(30.6))     # free-form patches: the largest ordinary fold between two quads
TIP_COS = math.cos(math.radians(60.0))      # ... and at the tip of a smooth (boxy) profile
GRID_TURN = math.cos(math.radians(50.0))    # a grid line bending more than this at a vertex is a mitre, not one line
WIDTH_RATIO = 3.0                           # neighbouring quads of one patch: this much wider at most (not a face)


def mesh_arrays(obj, depsgraph, rel_tol):
    """World-space vertices, faces and size of the evaluated mesh, welded
    within the tolerance; a warped n-gon is split into triangles (a warped
    quad stays: it may be part of a free-form patch)."""
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.transform(ev.matrix_world)
        co = np.array([v.co for v in bm.verts], float)
        size = float(np.linalg.norm(np.ptp(co, 0))) if len(co) else 0.0
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=rel_tol * size)
        bm.verts.index_update()
        V = np.array([v.co for v in bm.verts], float)
        F = [[v.index for v in f.verts] for f in bm.faces]
        bm.free()
    finally:
        ev.to_mesh_clear()
    out = []
    for f in F:
        P = V[f]
        n = np.cross(P, np.roll(P, -1, 0)).sum(0)
        nn = np.linalg.norm(n)
        if len(f) > 4 and nn > 0 and np.abs((P - P[0]) @ (n / nn)).max() > rel_tol * size:
            out += [[f[0], f[k], f[k + 1]] for k in range(1, len(f) - 1)]
        else:
            out.append(f)
    return V, out, size


def fit_circle(P, axis):
    """(centre, radius, largest residual) of the points seen along axis."""
    u = np.cross(axis, (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    c0 = P.mean(0)
    x, y = (P - c0) @ u, (P - c0) @ w
    s = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), -(x * x + y * y), rcond=None)[0]
    cx, cy = -0.5 * s[0], -0.5 * s[1]
    r2 = cx * cx + cy * cy - s[2]
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    return c0 + cx * u + cy * w, r, float(np.abs(np.hypot(x - cx, y - cy) - r).max())


def _basis(knots, p, n, s):
    """The n B-spline basis functions of degree p at parameters s (rows)."""
    N = ((knots[:-1] <= s[:, None]) & (s[:, None] < knots[1:])).astype(float)
    N[s >= knots[-1], n - 1] = 1.0
    for d in range(1, p + 1):
        a = knots[d:-1] - knots[:-d - 1]
        b = knots[d + 1:] - knots[1:-d]
        L = np.divide(s[:, None] - knots[:-d - 1], a, out=np.zeros((len(s), len(a))), where=a > 0)
        R = np.divide(knots[d + 1:] - s[:, None], b, out=np.zeros((len(s), len(b))), where=b > 0)
        N = L * N[:, :-1] + R * N[:, 1:]
    return N[:, :n]


_PROFILES = {}


def unit_profile(r, tol=2e-5):
    """(control points (u, v), knots) of a cubic B-spline through the
    superellipse (1 - u)^r + (1 - v)^r = 1 from (1, 0) to (0, 1), in units of
    the setbacks. Its first and last three control points lie on the end
    tangents, so it meets both faces with their (zero) curvature; its knots
    are denser where it turns. Control points are added until it is within
    tol of the curve both ways."""
    key = round(float(r), 3)
    if key in _PROFILES:
        return _PROFILES[key]
    b = np.linspace(0.0, 2.0 ** (-1.0 / r), 1001)          # each half sampled evenly along its face
    a = (1.0 - b ** r) ** (1.0 / r)
    H = np.column_stack([1 - b, 1 - a])
    P = np.vstack([H, H[-2::-1, ::-1]])
    d = np.diff(P, axis=0)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]
    s /= s[-1]
    turn = np.r_[0.0, np.abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))), 0.0]
    m = np.cumsum(np.sqrt(turn) + np.sqrt(turn).mean())
    m = (m - m[0]) / (m[-1] - m[0])
    ss = np.linspace(0.0, 1.0, 1500)
    best = None
    for n in range(8, 49, 4):
        knots = np.r_[[0.0] * 4, np.interp(np.linspace(0, 1, n - 2)[1:-1], m, s), [1.0] * 4]
        N = _basis(knots, 3, n, s)
        # unknowns: steps of control points 1, 2 along the tangent at (1, 0), n-2, n-3 at (0, 1), the rest free
        A = np.zeros((2 * len(s), 4 + 2 * (n - 6)))
        A[0::2, 0], A[0::2, 1] = -N[:, 1], -N[:, 2]
        A[1::2, 2], A[1::2, 3] = -N[:, n - 2], -N[:, n - 3]
        for k in range(n - 6):
            A[0::2, 4 + 2 * k], A[1::2, 5 + 2 * k] = N[:, 3 + k], N[:, 3 + k]
        base = np.outer(N[:, 0] + N[:, 1] + N[:, 2], (1.0, 0.0)) + np.outer(N[:, n - 1] + N[:, n - 2] + N[:, n - 3], (0.0, 1.0))
        x = np.linalg.lstsq(A, (P - base).ravel(), rcond=None)[0]
        C = np.zeros((n, 2))
        C[:3] = (1.0, 0.0)
        C[n - 3:] = (0.0, 1.0)
        C[1, 0] -= x[0]
        C[2, 0] -= x[1]
        C[n - 2, 1] -= x[2]
        C[n - 3, 1] -= x[3]
        C[3:n - 3] = x[4:].reshape(-1, 2)
        Q = _basis(knots, 3, n, ss) @ C
        qa, qb = np.clip(1 - Q[:, 1], 1e-12, None), np.clip(1 - Q[:, 0], 1e-12, None)
        F = qa ** r + qb ** r - 1                              # distance to the curve: |F| / |grad F|
        err = float(np.max(np.abs(F) / np.maximum(r * np.hypot(qa ** (r - 1), qb ** (r - 1)), 1e-12)))
        if best is None or err < best[2]:
            best = (C, knots, err)
        if err < tol:
            break
    _PROFILES[key] = best
    return best


class Recognizer:
    """Planar regions of a mesh, the fillets and rings among them, and the
    sharp body's face outlines."""

    def __init__(self, V, F, size, rel_tol=1e-5, fillets=True, rings=True):
        self.V, self.F, self.size = V, F, size
        self.tol = tol = rel_tol * size + 1e-12
        n = len(F)
        N, A, corner = np.zeros((n, 3)), np.zeros(n), []
        for k, f in enumerate(F):
            P = V[f]
            Q = np.roll(P, -1, 0)
            v = 0.5 * np.cross(P, Q).sum(0)
            A[k] = np.linalg.norm(v)
            N[k] = v / (A[k] or 1.0)
            d1, d2 = P - np.roll(P, 1, 0), Q - P
            turn = np.linalg.norm(np.cross(d1, d2), axis=1) > 1e-4 * np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)
            corner.append([f[i] for i in range(len(f)) if turn[i]] or f)
        self.N, self.A = N, A
        self.E = E = {}
        for k, f in enumerate(F):
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                E.setdefault((min(a, b), max(a, b)), []).append(k)
        # planar regions: grown from the largest facets, each within the
        # tolerance of its seed's plane (pairwise merging let a chain of near-flat
        # facets -- a boxy bevel's first segments -- drift off it, and the kernel
        # cannot make a flat face of a bent outline)
        nbr = [[] for _ in range(n)]
        for fs in E.values():
            if len(fs) == 2:
                nbr[fs[0]].append(fs[1])
                nbr[fs[1]].append(fs[0])
        rid = np.full(n, -1)
        self.regs = []
        for seed in np.argsort(-A):
            if rid[seed] >= 0:
                continue
            n0, p0 = N[seed], V[F[seed][0]]
            reg, todo = [int(seed)], [int(seed)]
            rid[seed] = len(self.regs)
            while todo:
                for g in nbr[todo.pop()]:
                    if rid[g] < 0 and N[g] @ n0 > FLAT_COS and np.abs((V[F[g]] - p0) @ n0).max() < tol:
                        rid[g] = len(self.regs)
                        reg.append(g)
                        todo.append(g)
            self.regs.append(reg)
        self._fans(rid)
        self.rid = np.empty(n, int)
        for i, r in enumerate(self.regs):
            self.rid[r] = i
        rn = np.array([(N[r] * A[r, None]).sum(0) for r in self.regs])
        self.rn = rn / np.maximum(np.linalg.norm(rn, axis=1), 1e-300)[:, None]
        self.rarea = np.array([A[r].sum() for r in self.regs])
        self.rverts = [sorted({v for f in r for v in corner[f]}) for r in self.regs]
        self.rd = np.array([float(np.mean(V[vs] @ nn)) for vs, nn in zip(self.rverts, self.rn)])
        self.adj = {}
        for fs in E.values():
            if len(fs) == 2 and self.rid[fs[0]] != self.rid[fs[1]]:
                i, j = self.rid[fs[0]], self.rid[fs[1]]
                self.adj.setdefault(i, set()).add(j)
                self.adj.setdefault(j, set()).add(i)
        self.open = any(len(fs) == 1 for fs in E.values())
        self.pn, self.pd = list(self.rn), list(self.rd)        # planes: the regions', then the boxy bevels' rails
        self._kept = {}                      # boxy bevels that end in a corner: {their faces: their regions}
        self.bands = [B for B in self._merge_cuts(self._bands()) if ((B["closed"] or B.get("kind") == "cut") and rings)
                      or (not B["closed"] and B.get("kind") != "cut" and fillets)]
        if fillets:
            self.bands += self._strips({i for B in self.bands for i in B["regions"]})
        self.pn, self.pd = np.array(self.pn), np.array(self.pd)
        self.role = {}                       # region -> ("band", b) | ("plane",) | ("corner",)
        for b, B in enumerate(self.bands):
            for i in B["regions"]:
                self.role[i] = ("band", b)
        self._corners()
        self.grids = self._grids() if fillets else []
        for _ in range(3):                   # a fillet or strip beside a patch joins it (they cannot be sewn)
            near = {int(self.rid[f]) for f in self.gridface}
            near |= {j for i in near for j in self.adj.get(i, ())}
            drop = [b for b, B in enumerate(self.bands) if not B["closed"] and set(B["regions"]) & near]
            if not drop:
                break
            for b in drop:
                for i in self.bands[b]["regions"]:
                    self.role[i] = ("plane",)
            keep = [B for b, B in enumerate(self.bands) if b not in drop]
            self.role = {i: (("band", keep.index(self.bands[r[1]])) if r[0] == "band" else r) for i, r in self.role.items()}
            self.bands = keep
            self.grids = self._grids()
        for g, G in enumerate(self.grids):
            for i in G["regions"]:
                self.role[i] = ("grid", g)
        self._tan = {}

    def _fans(self, rid):
        """A spun cap -- a fan of triangles about one vertex (Spin or Screw to
        the axis) whose normals agree within FLAT_COS and whose rim is flat --
        is one flat region, also where its tip is a little off the rim's
        plane (Solidify lifts it: its corner offset is not exact). Regions
        that are only the fan's triangles merge (regs; rid is the caller's,
        rebuilt from regs)."""
        V, F, N, A, tol = self.V, self.F, self.N, self.A, self.tol
        vf = {}
        for k, f in enumerate(F):
            for v in f:
                vf.setdefault(v, []).append(k)
        drop = set()
        for v, fs in vf.items():
            if len(fs) < 6 or any(len(F[k]) != 3 for k in fs):
                continue
            rs = sorted({int(rid[k]) for k in fs})
            if len(rs) < 2 or any(r in drop or not set(self.regs[r]) <= set(fs) for r in rs):
                continue
            n0 = (N[fs] * A[fs, None]).sum(0)
            n0 /= np.linalg.norm(n0) or 1.0
            if float((N[fs] @ n0).min()) < FLAT_COS:
                continue
            rim = V[sorted({u for k in fs for u in F[k]} - {v})]
            nr = np.linalg.svd(rim - rim.mean(0))[2][2]
            if abs(float(nr @ n0)) < FLAT_COS or np.abs((rim - rim.mean(0)) @ nr).max() > tol:
                continue
            keep = rs[0]
            for r in rs[1:]:
                self.regs[keep] += self.regs[r]
                self.regs[r] = []
                drop.add(r)
            rid[fs] = keep
        if drop:
            self.regs = [r for r in self.regs if r]

    # -- recognition -------------------------------------------------------------
    def _width(self, j, ax):
        """Region j's extent across the axis ax (in its plane)."""
        d = np.cross(self.rn[j], ax)
        return float(np.ptp(self.V[self.rverts[j]] @ d)) if np.linalg.norm(d) > 1e-9 else 0.0

    def _runs_along(self, regs, ax):
        """Is each region a bevel segment along ax: an edge parallel to it, and
        at least half the strip's length? (A corner patch's facets are not.)"""
        V, F = self.V, self.F
        t = [V[self.rverts[k]] @ ax for k in regs]
        length = max(float(x.max()) for x in t) - min(float(x.min()) for x in t)
        for k, x in zip(regs, t):
            if np.ptp(x) < 0.5 * length:
                return False
            par = False
            for f in self.regs[k]:
                P = V[F[f]]
                d = np.roll(P, -1, 0) - P
                ln = np.linalg.norm(d, axis=1)
                if np.any((np.abs(d @ ax) > 0.99996 * ln) & (ln > 1e-12)):
                    par = True
                    break
            if not par:
                return False
        return True

    def _rail(self, regs, j, ax):
        """How long region j touches the strip regs along edges parallel to ax
        (a fillet's face meets it along its whole rail)."""
        V, rid, ins = self.V, self.rid, set(regs)
        total = 0.0
        for (a, b), fs in self.E.items():
            if len(fs) == 2 and {rid[fs[0]], rid[fs[1]]} & ins and j in (rid[fs[0]], rid[fs[1]]):
                d = V[b] - V[a]
                ln = float(np.linalg.norm(d))
                if ln > 1e-12 and abs(d @ ax) > 0.99996 * ln:
                    total += ln
        return total

    def _fit(self, regs, ax):
        return fit_circle(self.V[sorted({v for k in regs for v in self.rverts[k]})], ax)

    def _axis(self, regs):
        """The axis of a round's regions: the one their normals are most nearly
        square to (two: their cross product), or the one they keep the most
        nearly constant angle to (a cone's normals; on part of a ring of
        slightly tilted facets the first leans by their tilt), whichever
        their corners fit a circle about better."""
        N, w = self.rn[regs], self.rarea[regs]
        ax = np.linalg.eigh((N * w[:, None]).T @ N)[1][:, 0]
        if len(regs) < 3:
            return ax
        D = N - w @ N / w.sum()
        ax2 = np.linalg.eigh((D * w[:, None]).T @ D)[1][:, 0]
        f1, f2 = self._fit(regs, ax), self._fit(regs, ax2)
        return ax2 if f2 is not None and (f1 is None or f2[2] < f1[2]) else ax

    def _bands(self):
        """Strips (fillets) and rings (holes, cylinders): grown from pairs of
        neighbouring regions over regions (nearly) square to the axis of those
        taken so far whose corners stay on one circle. (The pair's own axis is
        off by the facets' tilt where the wall is a little off square, which
        stopped a shell's outer wall a third of the way round.)"""
        V, tol, rn = self.V, self.tol, self.rn
        used, tried, bands = set(), set(), []
        for s in np.argsort(self.rarea):
            if s in used:
                continue
            for t in sorted(self.adj.get(s, ()), key=lambda j: self.rarea[j]):
                if t in used or (s, t) in tried:
                    continue
                tried.add((s, t))
                if not SEED_COS < rn[s] @ rn[t] < FLAT_COS:
                    continue
                ax = np.cross(rn[s], rn[t])
                ax /= np.linalg.norm(ax)
                band = [s, t]
                fit = self._fit(band, ax)
                if fit is None or fit[2] > 10 * tol:
                    continue
                grown = True
                while grown:
                    grown = False
                    for i in list(band):
                        for j in self.adj.get(i, ()):
                            if j in band or j in used or abs(rn[j] @ ax) > RING_TILT:
                                continue
                            a2 = self._axis(band + [j])
                            f2 = self._fit(band + [j], a2)
                            if f2 is not None and f2[2] <= 10 * tol:
                                band.append(j)
                                ax = a2
                                grown = True
                B = self._band(band)
                if B is not None:
                    bands.append(B)
                    used.update(band)
                    break
        return bands

    def _band(self, band):
        V, rn = self.V, self.rn
        ax = np.linalg.eigh(sum(self.rarea[k] * np.outer(rn[k], rn[k]) for k in band))[1][:, 0]
        vs = sorted({v for k in band for v in self.rverts[k]})
        fit = fit_circle(V[vs], ax)
        if fit is None or fit[2] > 10 * self.tol:
            return None
        c, r, _ = fit
        d = V[vs] - c
        t = d @ ax
        d -= np.outer(t, ax)
        u = d[0] / (np.linalg.norm(d[0]) or 1.0)
        a = np.sort(np.arctan2(d @ np.cross(ax, u), d @ u))
        gaps = np.diff(np.concatenate([a, a[:1] + 2 * math.pi]))
        steps = gaps[gaps > 1e-6]
        closed = len(steps) >= 6 and gaps.max() < 2.5 * float(np.median(steps))
        k0 = band[0]
        p0 = V[self.rverts[k0]].mean(0) - c
        B = {"regions": band, "r": r, "axis": ax, "center": c, "t": (float(t.min()), float(t.max())),
             "closed": bool(closed), "inward": float((p0 - (p0 @ ax) * ax) @ rn[k0]) < 0}
        if not closed:
            if not self._runs_along(band, ax):
                return None
            nb = {j for k in band for j in self.adj.get(k, ()) if j not in band}
            wide = 2.0 * max(self._width(k, ax) for k in band)      # not a facet of a boxy bevel beside it
            tang = [j for j in nb if abs(rn[j] @ ax) < SQUARE and self._width(j, ax) > wide
                    and abs(abs(c @ rn[j] - self.rd[j]) - r) < 100 * self.tol]
            if len(tang) != 2 or len(band) < 2 or min(self._rail(band, j, ax) for j in tang) < 0.5 * float(np.ptp(t)):
                return self._cut(B, vs)
            B["planes"] = tang
        return B

    def _cut(self, B, vs):
        """An open band that is no fillet: a hole or cylinder cut through an
        edge or a corner (a notch) -- part of a cylinder, its two long sides
        on planes along its axis (its rulings) and its two ends on planes
        across it. None if it is not bordered so."""
        V, rn, rd, tol = self.V, self.rn, self.rd, self.tol
        band, ax, c, r = B["regions"], B["axis"], B["center"], B["r"]
        if len(band) < 3:
            return None
        nb = {j for k in band for j in self.adj.get(k, ()) if j not in band}
        sides = [j for j in nb if abs(rn[j] @ ax) < SQUARE]
        ends = [j for j in nb if abs(rn[j] @ ax) > 0.2]
        if len(ends) != 2 or not sides or len(sides) + len(ends) != len(nb):
            return None
        # where the band meets a side plane the mesh's vertices lie on the facets' chords, not
        # the circle: the circle is fitted to the others, the corners are where it meets the planes
        inner = [v for v in vs if all(abs(V[v] @ rn[j] - rd[j]) > 10 * tol for j in sides)]
        fit = fit_circle(V[inner], ax) if len(inner) >= 3 else None
        if fit is None or fit[2] > 10 * tol:
            return None
        c, r = fit[0], fit[1]
        B["center"], B["r"] = c, r
        u = np.cross(ax, (1.0, 0.0, 0.0) if abs(ax[0]) < 0.9 else (0.0, 1.0, 0.0))
        u /= np.linalg.norm(u)
        w = np.cross(ax, u)
        d = V[vs] - c
        ang = np.arctan2(d @ w, d @ u)
        a = np.sort(ang)
        gaps = np.diff(np.concatenate([a, a[:1] + 2 * math.pi]))
        k = int(np.argmax(gaps))
        a0 = float(a[(k + 1) % len(a)])                       # the band runs from a0 round to a0 + span
        span = float(2 * math.pi - gaps[k])
        step = float(np.delete(gaps, k).max())
        if not 0.05 < span < 2 * math.pi - 0.05 or step > math.radians(30.0):
            return None                                       # (facets of a cylinder: narrow; four corners of a
                                                              # chamfered box corner also lie on a circle)
        def on_side(th):                                      # the angle near th where a side plane cuts the circle
            pt = c + r * (math.cos(th) * u + math.sin(th) * w)
            j = min(sides, key=lambda j: abs(pt @ rn[j] - rd[j]))
            A_, B_, C_ = r * (rn[j] @ u), r * (rn[j] @ w), rd[j] - c @ rn[j]
            h = math.hypot(A_, B_)
            if h < 1e-12 or abs(C_) > h:
                return th
            base, off = math.atan2(B_, A_), math.acos(max(-1.0, min(1.0, C_ / h)))
            return min((base + sg * off for sg in (1, -1)), key=lambda x: abs((x - th + math.pi) % (2 * math.pi) - math.pi))
        t0 = on_side(a0)
        t1 = on_side(a0 + span)
        span = (t1 - t0) % (2 * math.pi)
        a0 = t0
        if not 0.05 < span < 2 * math.pi - 0.05:
            return None
        arcs, corners = {}, {}
        for j in ends:                                        # each end: the arc on that plane, its corners exact
            on = [v for v in vs if abs(V[v] @ rn[j] - rd[j]) < 10 * tol]
            if len(on) < 2:
                return None
            x = (np.arctan2((V[on] - c) @ w, (V[on] - c) @ u) - a0) % (2 * math.pi)
            x[x > span + 0.5 * (2 * math.pi - span)] -= 2 * math.pi
            if x.min() > 0.5 * step or x.max() < span - 0.5 * step:   # an end not across the whole band
                                                              # (a chord's vertex is off the corner: half a facet)
                return None
            A = self._arc(c, ax, u, w, r, a0, span, rn[j], rd[j])
            arcs[j] = A
            corners[on[int(np.argmin(x))]] = A["cvs"][0]
            corners[on[int(np.argmax(x))]] = A["cvs"][-1]
        B["corners"] = corners
        B.update(kind="cut", sides=sides, ends=ends, arcs=arcs)
        return B

    def _merge_cuts(self, bands):
        """Cut cylinders made whole: pieces on one circle (a cylinder Blender's
        boolean split) joined, and the facets it cut short at the band's sides
        taken in -- two of their corners lie on a chord, not the circle, so
        the band did not grow over them and they stayed thin flat faces."""
        V, rn, rd, tol = self.V, self.rn, self.rd, self.tol
        rest = [B for B in bands if B.get("kind") != "cut"]
        taken = {i for B in rest for i in B["regions"]}
        groups = []
        for B in (B for B in bands if B.get("kind") == "cut"):
            for g in groups:
                A = g[0]
                dc = B["center"] - A["center"]
                if abs(A["axis"] @ B["axis"]) > 1 - 1e-6 and abs(A["r"] - B["r"]) < 100 * tol \
                        and np.linalg.norm(dc - (dc @ A["axis"]) * A["axis"]) < 100 * tol:
                    g.append(B)
                    break
            else:
                groups.append([B])
        out = []
        for g in groups:
            A = g[0]
            ax, c, r = A["axis"], A["center"], A["r"]
            regs = [i for B in g for i in B["regions"]]
            widest = max(self._width(k, ax) for k in regs)
            lo = r * math.cos(min(widest / r, 0.5 * math.pi)) - 10 * tol
            n0 = len(regs)
            grown = True
            while grown:
                grown = False
                for j in sorted({j for k in regs for j in self.adj.get(k, ())} - set(regs) - taken):
                    if abs(rn[j] @ ax) > SQUARE or self._width(j, ax) > 1.05 * widest:
                        continue
                    if not lo < abs(c @ rn[j] - rd[j]) < r + 10 * tol:    # a chord's plane: the facet's
                        continue
                    q = V[self.rverts[j]] - c
                    q -= np.outer(q @ ax, ax)
                    if np.sum(np.abs(np.linalg.norm(q, axis=1) - r) < 10 * tol) < 2:   # its edge on the circle
                        continue
                    regs.append(j)
                    grown = True
            if len(g) == 1 and len(regs) == n0:
                out.append(A)
                continue
            vs = sorted({v for k in regs for v in self.rverts[k]})
            t = (V[vs] - c) @ ax
            B = self._cut({"regions": regs, "r": r, "axis": ax, "center": c, "t": (float(t.min()), float(t.max())),
                           "closed": False, "inward": A["inward"]}, vs)
            out += [B] if B is not None else g
        return rest + out

    @staticmethod
    def _arc(c, ax, u, w, r, a0, span, n, dist):
        """The circle's arc from angle a0 over span (rational quadratic, spans of
        at most 90 degrees) lifted along the axis onto the plane n . x = dist
        (an ellipse's arc there: the same weights)."""
        m = max(1, int(math.ceil(span / (0.5 * math.pi) - 1e-9)))
        h = 0.5 * span / m
        cvs, wts = [], []
        for k in range(m):
            for q, (ang, rad, wt) in enumerate(((a0 + 2 * k * h, r, 1.0), (a0 + (2 * k + 1) * h, r / math.cos(h), math.cos(h)))):
                cvs.append(c + rad * (math.cos(ang) * u + math.sin(ang) * w))
                wts.append(wt)
        cvs.append(c + r * (math.cos(a0 + span) * u + math.sin(a0 + span) * w))
        wts.append(1.0)
        cvs = [q + ax * (dist - q @ n) / (ax @ n) for q in cvs]
        knots = [0.0] * 3 + [x for k in range(1, m) for x in (k / m, k / m)] + [1.0] * 3
        return {"cvs": [list(map(float, q)) for q in cvs], "weights": wts, "knots": knots, "degree": 2}

    def _strips(self, used):
        """Boxy bevels: strips of evenly wide regions square to one axis
        (Blender spaces a bevel's segments evenly) whose corners lie on one
        superellipse between two wide planes (_strip)."""
        rn, out, tried = self.rn, [], set()
        for s in np.argsort(self.rarea):
            if s in used:
                continue
            for t in self.adj.get(s, ()):
                if t in used or (s, t) in tried or not BOXY_COS < rn[s] @ rn[t] < FLAT_COS:
                    continue
                tried.add((s, t))
                ax = np.cross(rn[s], rn[t])
                ax /= np.linalg.norm(ax)
                w0 = self._width(s, ax)
                long = lambda j: float(np.ptp(self.V[self.rverts[j]] @ ax))      # (not a corner patch's facet)
                l0 = long(s)
                strip, todo = [s], [s]
                while todo:
                    i = todo.pop()
                    for j in self.adj.get(i, ()):
                        if j not in strip and j not in used and abs(rn[j] @ ax) < SQUARE and long(j) > 0.5 * l0 \
                                and 0.5 < self._width(j, ax) / max(w0, 1e-300) < 2.0 and BOXY_COS < rn[i] @ rn[j]:
                            strip.append(j)
                            todo.append(j)
                B = self._strip(strip) if len(strip) >= BOXY_SEGMENTS else None
                if B is not None:
                    out.append(B)
                    used.update(strip)
                    break
        # an end on another boxy bevel's end, across one flat mitre: both end there
        for A in out:
            for sa, (nba, pa) in list(A["pending"].items()):
                for Bb in out:
                    if Bb is A or not nba <= set(Bb["regions"]):
                        continue
                    sb = next((k for k, (nbb, _) in Bb["pending"].items() if nbb <= set(A["regions"])), None)
                    if sb is None:
                        continue
                    X = self.V[sorted(set(pa) | set(Bb["pending"][sb][1]))]
                    c = X.mean(0)
                    n = np.linalg.svd(X - c)[2][2]
                    if np.abs((X - c) @ n).max() < 10 * self.tol:
                        self.pn.append(n)
                        self.pd.append(float(n @ c))
                        A["ends"][sa] = Bb["ends"][sb] = len(self.pn) - 1
                        del A["pending"][sa], Bb["pending"][sb]
                    break
        for A in out:
            if A["pending"]:
                self._kept[frozenset(A["planes"])] = set(A["regions"])
        return [A for A in out if not A["pending"]]

    def _strip(self, strip):
        """The superellipse of a strip between two wide planes P, Q, ending on
        flat faces square to it: setbacks w1 (on P), w2 (on Q) and exponent r
        fitted to its corners in the frame of the sharp edge; None if it is not
        one (or not rebuildable)."""
        V, rn, rd, tol = self.V, self.rn, self.rd, self.tol
        ax = np.linalg.eigh(sum(self.rarea[k] * np.outer(rn[k], rn[k]) for k in strip))[1][:, 0]
        if not self._runs_along(strip, ax):
            return None
        wide = 2.0 * max(self._width(k, ax) for k in strip)
        nb = {j for k in strip for j in self.adj.get(k, ()) if j not in strip}
        planes = [j for j in nb if abs(rn[j] @ ax) < SQUARE and self._width(j, ax) > wide]
        length = float(np.ptp(np.concatenate([V[self.rverts[k]] @ ax for k in strip])))
        if len(planes) != 2 or min(self._rail(strip, j, ax) for j in planes) < 0.5 * length:
            return None
        P, Q = planes
        if abs(rn[P] @ rn[Q]) > SEED_COS:              # (near) parallel: no edge between them (a notch's two sides)
            return None
        C0 = np.linalg.solve(np.array([rn[P], rn[Q], ax]), [rd[P], rd[Q], 0.0])
        # the segments lying within tolerance of a face merged into it: still thin
        # quads beside the strip, whose corners are on the curve too
        facets = {f for k in strip for f in self.regs[k]}
        fw = float(np.median([np.ptp(V[self.F[f]] @ np.cross(self.N[f], ax)) for f in facets]))
        extra, todo = set(), set(facets)
        while todo:
            f = todo.pop()
            for i in range(len(self.F[f])):
                a, b = self.F[f][i], self.F[f][(i + 1) % len(self.F[f])]
                for g in self.E[(min(a, b), max(a, b))]:
                    if g in facets or g in extra or self.rid[g] not in planes or len(self.F[g]) != 4:
                        continue
                    if 0.5 * fw < np.ptp(V[self.F[g]] @ np.cross(self.N[g], ax)) < 2.0 * fw:
                        extra.add(g)
                        todo.add(g)
        e = []
        for j in (P, Q):                                     # in each plane, away from the edge
            d = np.cross(ax, rn[j])
            e.append(d if (V[self.rverts[j]].mean(0) - C0) @ d > 0 else -d)
        vs = sorted({v for k in strip for v in self.rverts[k]} | {v for g in extra for v in self.F[g]})
        X = V[vs] - C0
        t = X @ ax
        xy = np.linalg.lstsq(np.column_stack(e), (X - np.outer(t, ax)).T, rcond=None)[0].T
        x, y = xy[:, 0], xy[:, 1]

        # the setbacks: the profile's outermost points (Blender ends it exactly
        # there; very boxy profiles hug the faces so closely that fitting them
        # with the exponent was ambiguous); then the exponent alone
        w1, w2 = float(x.max()), float(y.max())
        a, b = np.clip(1 - y / w2, 0.0, 1.0), np.clip(1 - x / w1, 0.0, 1.0)

        def resid(r):
            g = r * np.hypot(np.maximum(a, 1e-12) ** (r - 1) / w2, np.maximum(b, 1e-12) ** (r - 1) / w1)
            return (a ** r + b ** r - 1) / np.maximum(g, 1e-300)
        inner = (a > 1e-9) & (a < 1 - 1e-9) & (b > 1e-9) & (b < 1 - 1e-9)
        if inner.sum() < 2:
            return None
        lo, hi = 1.01, 1000.0                                  # the point nearest the corner fixes the exponent
        k = int(np.argmax(np.minimum(a, b) * inner))
        for _ in range(100):
            m = math.sqrt(lo * hi)
            lo, hi = (m, hi) if a[k] ** m + b[k] ** m > 1 else (lo, m)
        r = math.sqrt(lo * hi)
        for _ in range(20):                                    # least squares over all of them
            f = resid(r)
            h = 1e-6 * r
            J = (resid(r + h) - f) / h
            step = -float(J @ f) / max(float(J @ J), 1e-300)
            r = max(1.01, r + step)
            if abs(step) < 1e-10 * r:
                break
        if not 2.02 < r < 400.0 or np.abs(resid(r)).max() > 10 * tol:
            return None
        # its two ends: the regions across its end edges (not along the axis), and
        # their points; one flat face holding them all is an end, anything else
        # waits for _strips (a mitre with another boxy bevel, or a corner)
        ends, pending = [None, None], {}
        tmid = 0.5 * float(t.min() + t.max())
        nbs, pts = ([set(), set()] for _ in range(2))
        own = set(strip)
        for f in facets:
            fv = self.F[f]
            for k in range(len(fv)):
                a_, b_ = fv[k], fv[(k + 1) % len(fv)]
                d = V[b_] - V[a_]
                ln = float(np.linalg.norm(d))
                across = [g for g in self.E[(min(a_, b_), max(a_, b_))] if self.rid[g] not in own]
                if ln < 1e-12 or abs(d @ ax) > 0.99996 * ln or not across:
                    continue
                side = int((0.5 * (V[a_] + V[b_]) - C0) @ ax > tmid)
                nbs[side] |= {int(self.rid[g]) for g in across}
                pts[side] |= {a_, b_}
        for side in (0, 1):
            on = [j for j in nbs[side] - set(planes)
                  if np.abs(V[sorted(pts[side])] @ rn[j] - rd[j]).max() < 10 * tol]
            if len(on) == 1 and len(nbs[side] - set(planes)) == 1:
                ends[side] = on[0]
            else:
                pending[side] = (nbs[side] - set(planes), sorted(pts[side]))
        rail = {}
        for j, other, wj, ej in ((P, Q, w1, e[0]), (Q, P, w2, e[1])):
            rail[j] = len(self.pn)                          # the rail on face j: j meets the plane parallel to the other face
            self.pn.append(rn[other])
            self.pd.append(float(rn[other] @ (C0 + wj * ej)))
        return {"kind": "boxy", "regions": strip, "planes": [P, Q], "axis": ax, "corner": C0, "e": e,
                "w": (w1, w2), "r": r, "t": (float(t.min()), float(t.max())), "ends": ends, "rail": rail,
                "pending": pending, "closed": False, "inward": False}

    def profile(self, b, at, reverse=False):
        """Boxy bevel b's profile on its end plane `at` (an end face, or a
        mitre): control points (world, from its rail on P to its rail on Q),
        moved along the edge onto that plane, and knots."""
        B = self.bands[b]
        C, knots, _ = unit_profile(B["r"])
        ax, n, d = B["axis"], self.pn[at], self.pd[at]
        P3 = B["corner"] + np.outer(C[:, 0] * B["w"][0], B["e"][0]) + np.outer(C[:, 1] * B["w"][1], B["e"][1])
        P3 = P3 + np.outer((d - P3 @ n) / (n @ ax), ax)
        if reverse:
            return P3[::-1].tolist(), (1.0 - knots[::-1]).tolist()
        return P3.tolist(), knots.tolist()

    def _corners(self):
        """The rest are planes, except the patches where three or more fillets
        meet: corners on a sphere of the fillets' radius about the point where
        their axes end, bordered by no plane."""
        ends = [(B["center"] + B["axis"] * t, B["r"]) for B in self.bands
                if not B["closed"] and "center" in B and B.get("kind") != "cut" for t in B["t"]]
        for i in range(len(self.regs)):
            if i in self.role:
                continue
            P = self.V[self.rverts[i]]
            corner = any(np.abs(np.linalg.norm(P - c, axis=1) - r).max() < 0.05 * r for c, r in ends) \
                and all(self.role.get(j, ("?",))[0] != "plane" for j in self.adj.get(i, ()))
            self.role[i] = ("corner",) if corner else ("plane",)

    # -- free-form patches ---------------------------------------------------------
    # What is left smooth -- the corner where boxy bevels meet, a bevel swept
    # along a curved edge, a boxy bevel running into one -- is a grid of quads.
    # It is cut into rectangular blocks along the lines from its irregular
    # vertices, and each block becomes a C1 bicubic B-spline through its
    # vertices. The tangent at a vertex along a mesh line is one value for
    # every curve and patch through it, so neighbouring patches and the flat
    # faces beside them share their edge curves exactly, and meet tangentially
    # where the mesh is smooth.
    def _grids(self):
        V, F, N, E, rid = self.V, self.F, self.N, self.E, self.rid
        wid = {}
        for f in range(len(F)):
            if len(F[f]) == 4:
                P = V[F[f]]
                wid[f] = min(np.linalg.norm(P[0] + P[1] - P[2] - P[3]), np.linalg.norm(P[1] + P[2] - P[3] - P[0])) / 2
        # quads of flat regions (a near-flat first segment merged into a large
        # face is taken back from it: patches work on facets)
        quad = [f for f in wid if self.role[rid[f]][0] == "plane"]
        cand = set(quad)
        par = {f: f for f in quad}

        def find(f):
            while par[f] != f:
                par[f] = par[par[f]]
                f = par[f]
            return f
        for (a, b), fs in E.items():
            if len(fs) == 2 and fs[0] in cand and fs[1] in cand and self._smooth(fs[0], fs[1], a, b) \
                    and max(wid[fs[0]], wid[fs[1]]) <= WIDTH_RATIO * max(min(wid[fs[0]], wid[fs[1]]), 1e-300):
                par[find(fs[0])] = find(fs[1])
        comps = {}
        for f in quad:
            comps.setdefault(find(f), set()).add(f)
        out = []
        self.gridface = {}
        for comp in comps.values():
            if len(comp) < 3:
                continue
            for block in self._blocks(comp):
                G = self._grid(block)
                regs = {int(rid[f]) for f in block}
                if G is None or max(G["verts"].shape) < 4 or len(regs) < 3:
                    continue                                    # (3 segments at least, curved: else facets)
                for f in block:
                    self.gridface[f] = len(out)
                G["regions"] = {r for r in regs if set(self.regs[r]) <= block}
                out.append(G)
        return out

    def _smooth(self, f, g, a, b):
        """Do quads f, g (sharing edge a-b) belong to one smooth surface? A
        small fold, or a larger one at the tip of a smooth profile: the quads
        beyond them fold the same way, by at least a quarter of it, one of
        them by an ordinary fold (a chamfer's or an octagon's equal folds are
        creases)."""
        N, F, E = self.N, self.F, self.E
        c = float(N[f] @ N[g])
        if c > GRID_COS:
            return True
        if c < TIP_COS:
            return False
        ang = math.acos(c)
        e = self.V[b] - self.V[a]
        sense = np.cross(N[f], N[g]) @ e
        folds = []
        for x, y in ((f, g), (g, f)):
            rest = [v for v in F[x] if v not in (a, b)]
            if len(F[x]) != 4 or len(rest) != 2:
                return False
            far = [h for h in E[(min(rest), max(rest))] if h != x]
            if len(far) != 1 or len(F[far[0]]) != 4:
                return False
            n1, n2 = (N[far[0]], N[x]) if x is f else (N[x], N[far[0]])
            fold = math.acos(min(1.0, float(n1 @ n2)))
            if not ang / 4 <= fold <= ang + 0.0175 or np.cross(n1, n2) @ e * sense <= 0:
                return False
            folds.append(fold)
        return min(folds) <= math.acos(GRID_COS)

    def _blocks(self, comp):
        """comp cut into blocks along the edge lines from its irregular
        vertices (not four quads inside, more than two on its border)."""
        F, E = self.F, self.E
        vq, nbr = {}, {}
        for f in comp:
            for k, v in enumerate(F[f]):
                vq.setdefault(v, set()).add(f)
                w = F[f][(k + 1) % 4]
                nbr.setdefault(v, set()).add(w)
                nbr.setdefault(w, set()).add(v)
        inner = lambda a, b: [g for g in E[(min(a, b), max(a, b))] if g in comp]
        border = {v for v in nbr for w in nbr[v] if len(inner(v, w)) == 1}
        odd = lambda v: len(vq[v]) != 4 if v not in border else len(vq[v]) > 2
        cuts = set()
        for v in [v for v in nbr if odd(v)]:
            for w in nbr[v]:
                prev, cur = v, w
                for _ in range(len(nbr)):
                    cuts.add((min(prev, cur), max(prev, cur)))
                    if cur in border or odd(cur):
                        break
                    fs = set(inner(prev, cur))
                    nxt = [u for u in nbr[cur] if u != prev and not set(inner(cur, u)) & fs]
                    if len(nxt) != 1:
                        break
                    prev, cur = cur, nxt[0]
        out, seen = [], set()
        for f0 in comp:
            if f0 in seen:
                continue
            block, todo = set(), [f0]
            seen.add(f0)
            while todo:
                f = todo.pop()
                block.add(f)
                for k in range(4):
                    a, b = F[f][k], F[f][(k + 1) % 4]
                    if (min(a, b), max(a, b)) in cuts:
                        continue
                    for g in inner(a, b):
                        if g not in seen:
                            seen.add(g)
                            todo.append(g)
            out.append(block)
        return out

    def _grid(self, block):
        """{"verts": (nu+1, nv+1) vertex array} of a block that is a full
        rectangle of quads, filled from one corner; None otherwise (a ring)."""
        F = self.F
        vq, edges = {}, {}
        for f in block:
            for k, v in enumerate(F[f]):
                vq.setdefault(v, []).append(f)
                w = F[f][(k + 1) % 4]
                edges.setdefault((min(v, w), max(v, w)), []).append(f)
        corner = next((v for v, fs in vq.items() if len(fs) == 1), None)
        if corner is None:
            return None
        f = vq[corner][0]
        k = F[f].index(corner)
        row, col = [corner, F[f][(k + 1) % 4]], [corner, F[f][(k - 1) % 4]]
        for line in (row, col):                                # walk each border line from the corner
            while True:
                a, b = line[-2], line[-1]
                fs = edges[(min(a, b), max(a, b))]
                nxt = [w for g in vq[b] if g not in fs for w in (F[g][(F[g].index(b) + 1) % 4], F[g][(F[g].index(b) - 1) % 4])
                       if (min(b, w), max(b, w)) in edges and len(edges[(min(b, w), max(b, w))]) == 1 and w != a]
                if len(vq[b]) != 2 or len(nxt) != 1:
                    break
                line.append(nxt[0])
        G = np.full((len(row), len(col)), -1)
        G[:, 0], G[0, :] = row, col
        quads = {frozenset(F[f]): f for f in block}
        used = set()
        for i in range(len(row) - 1):
            for j in range(len(col) - 1):
                a, b, c = G[i, j], G[i + 1, j], G[i, j + 1]
                g = next((g for g in vq.get(a, ()) if b in F[g] and c in F[g] and g not in used), None)
                if g is None:
                    return None
                used.add(g)
                d = (set(F[g]) - {a, b, c}).pop()
                if G[i + 1, j + 1] >= 0 and G[i + 1, j + 1] != d:
                    return None
                G[i + 1, j + 1] = d
        if len(used) != len(block) or len(set(G.ravel())) != G.size:
            return None
        return {"verts": G, "facets": set(block)}

    def _cont(self, w, v):
        """The vertex after v going from w along a mesh line: straight on
        through four smooth quads, else the straightest edge that turns less
        than a mitre; None where the line ends."""
        V, F = self.V, self.F
        if not hasattr(self, "_vf"):
            self._vf, self._vn = {}, {}
            for f, fv in enumerate(F):
                for k, a in enumerate(fv):
                    self._vf.setdefault(a, []).append(f)
                    b = fv[(k + 1) % len(fv)]
                    self._vn.setdefault(a, set()).add(b)
                    self._vn.setdefault(b, set()).add(a)
        d = V[v] - V[w]
        d /= np.linalg.norm(d) or 1.0
        gf = getattr(self, "gridface", {})
        quads = [f for f in self._vf[v] if f in gf]            # within patches: the grid line goes straight on,
        inc = [f for f in quads if w in F[f]]                  # through the quad edges not beside the incoming one
        if inc:
            edge = {u for f in quads for k, u in enumerate(F[f]) if v in (F[f][k - 1], F[f][(k + 1) % 4])}
            u = [u for u in edge if u != w and not any(u in F[f] for f in inc)]
            if len(u) == 1:
                e = V[u[0]] - V[v]
                if e @ d > TIP_COS * np.linalg.norm(e):
                    return u[0]
        fs = self._vf[v]
        if len(fs) == 4 and all(len(F[f]) == 4 for f in fs) and len(self._vn[v]) == 4:
            inc = [f for f in fs if w in F[f]]
            u = [u for u in self._vn[v] if u != w and not any(u in F[f] for f in inc)]
            if len(u) == 1:
                e = V[u[0]] - V[v]
                if e @ d > GRID_TURN * np.linalg.norm(e):
                    return u[0]
        best, cos = None, math.cos(math.radians(30.0))
        for u in self._vn[v]:
            if u == w:
                continue
            e = V[u] - V[v]
            c = float(e @ d) / (np.linalg.norm(e) or 1.0)
            if c > cos:
                best, cos = u, c
        return best

    def tangent(self, v, w):
        """The shared tangent at v towards w, per unit parameter step:
        Bessel's from the chords on both sides (the flatter side's chord
        where the curvature jumps, as on a boxy profile); at a line's end,
        its chord turned into the flat face it runs into (tangent there), or
        into the face it runs along."""
        key = (v, w)
        if key in self._tan:
            return self._tan[key]
        V = self.V
        lb = float(np.linalg.norm(V[w] - V[v]))
        df = (V[w] - V[v]) / (lb or 1.0)
        c = self._cont(w, v)
        into = None                                            # a patch's line leaving it on its border:
        if c is None or not any(f in self.gridface for f in self._vf[v] if c in self.F[f]):
            into = self._face_at(v, w)                         # tangent to the flat face beside it there
        if into is not None:
            t = df - (df @ into) * into
            t = t / (np.linalg.norm(t) or 1.0) * lb
        elif c is not None:
            la = float(np.linalg.norm(V[v] - V[c]))
            db = (V[v] - V[c]) / (la or 1.0)
            t = (db * lb + df * la) / (la + lb)
            pp, nn = self._cont(v, c), self._cont(v, w)
            if pp is not None and nn is not None:              # Akima-like where the curvature jumps
                tb = math.acos(min(1.0, abs(float(db @ (V[c] - V[pp]) / (np.linalg.norm(V[c] - V[pp]) or 1.0)))))
                tf = math.acos(min(1.0, abs(float(df @ (V[nn] - V[w]) / (np.linalg.norm(V[nn] - V[w]) or 1.0)))))
                if tb + tf > 1e-9 and max(tb, tf) > 4.0 * min(tb, tf):
                    t = db * lb * tf + df * la * tb
            t = t / (np.linalg.norm(t) or 1.0) * min(la, lb)
        else:
            t = df * lb
        flat = [self._flat(f) for f in self._vf[v] if w in self.F[f]]
        flat = [n for n in flat if n is not None]
        n = flat[0] if flat else self._plane_at(v, w)          # along a flat face's border: in its plane
        if n is not None:
            t = t - (t @ n) * n
        self._tan[key] = t
        return t

    def _flat(self, f):
        """The plane a facet lies on if it belongs to a flat face (not a patch)."""
        return None if f in self.gridface else self.pn[int(self.rid[f])]

    def _face_at(self, v, w):
        """The flat face beside a patch where its line (w, v) reaches the
        patch's border at v, if they meet smoothly (tangent): its normal.
        That is the face across the patch's border edges at v."""
        own = [f for f in self._vf[v] if w in self.F[f] and f in self.gridface]
        best, cos = None, math.cos(math.radians(30.0))
        for f in own:
            fv = self.F[f]
            k = fv.index(v)
            for x in (fv[k - 1], fv[(k + 1) % len(fv)]):
                if x == w:
                    continue
                for g in self.E[(min(v, x), max(v, x))]:
                    n = self._flat(g) if g != f else None
                    if n is not None and abs(float(n @ self.N[f])) > cos:
                        best, cos = n, abs(float(n @ self.N[f]))
        return best

    def _plane_at(self, v, w):
        """The plane every curve through v keeps its tangent in: the flat face
        v lies on (one only, met smoothly along (v, w)), else the patches'
        averaged normal where only smooth patches meet at v."""
        fs = self._vf[v]
        flats = {tuple(np.round(n, 9)) for n in (self._flat(f) for f in fs) if n is not None}
        own = [f for f in fs if w in self.F[f]]
        if len(flats) == 1:
            n = np.array(next(iter(flats)))
            return n if max(abs(float(n @ self.N[g])) for g in own) > GRID_COS else None
        if not flats and all(self.N[a] @ self.N[b] > TIP_COS for a in fs for b in fs):
            n = (self.N[fs] * self.A[fs, None]).sum(0) if hasattr(self, "A") else self.N[fs].sum(0)
            return n / (np.linalg.norm(n) or 1.0)
        return None

    def hermite(self, vs):
        """(control points, knots) of the C1 cubic through the vertices vs
        with the shared tangents."""
        T = [self.tangent(vs[k], vs[k + 1]) for k in range(len(vs) - 1)] + [-self.tangent(vs[-1], vs[-2])]
        P = self.V[vs]
        cps = [P[0], P[0] + T[0] / 3]
        for k in range(1, len(vs) - 1):
            cps += [P[k] - T[k] / 3, P[k] + T[k] / 3]
        cps += [P[-1] - T[-1] / 3, P[-1]]
        n = len(vs) - 1
        knots = [0.0] * 4 + [float(k) for k in range(1, n) for _ in range(2)] + [float(n)] * 4
        return [list(map(float, c)) for c in cps], knots

    def patch(self, G):
        """(control net u-major, knots u, knots v) of a block's C1 bicubic patch."""
        vs = G["verts"]
        nu, nv = vs.shape[0] - 1, vs.shape[1] - 1
        Q = self.V[vs]
        Tu = np.array([[self.tangent(vs[i, j], vs[i + 1, j]) if i < nu else -self.tangent(vs[i, j], vs[i - 1, j])
                        for j in range(nv + 1)] for i in range(nu + 1)])
        Tv = np.array([[self.tangent(vs[i, j], vs[i, j + 1]) if j < nv else -self.tangent(vs[i, j], vs[i, j - 1])
                        for j in range(nv + 1)] for i in range(nu + 1)])
        slots = lambda n: [(0, 0), (0, 1)] + [(k, s) for k in range(1, n) for s in (-1, 1)] + [(n, -1), (n, 0)]
        net = [list(map(float, Q[i, j] + a * Tu[i, j] / 3 + b * Tv[i, j] / 3))
               for i, a in slots(nu) for j, b in slots(nv)]
        kn = lambda n: [0.0] * 4 + [float(k) for k in range(1, n) for _ in range(2)] + [float(n)] * 4
        return net, kn(nu), kn(nv)

    # -- the sharp body ---------------------------------------------------------
    def loops(self, i):
        """Region i's outlines: cycles of (vertex, region across the edge from
        it -- ("G", patch) for a patch's facet -- or None). Facets a patch took
        are not the region's."""
        F, E, rid = self.F, self.E, self.rid
        gf = getattr(self, "gridface", {})
        own = [f for f in self.regs[i] if f not in gf]
        mine = set(own)
        nxt = {}
        for f in own:
            fv = F[f]
            for k in range(len(fv)):
                a, b = fv[k], fv[(k + 1) % len(fv)]
                other = [g for g in E[(min(a, b), max(a, b))] if g not in mine]
                if other or len(E[(min(a, b), max(a, b))]) == 1:
                    g = other[0] if other else None
                    nxt[a] = (b, None if g is None else ("G", gf[g]) if g in gf else int(rid[g]))
        out = []
        while nxt:
            a = next(iter(nxt))
            lp = []
            while a in nxt:
                b, j = nxt.pop(a)
                lp.append((a, j))
                a = b
            out.append(lp)
        return out

    def label(self, i, j):
        """What region j across i's outline is for the sharp outline:
        ("P", plane, direct) -- through a fillet, the plane beyond it; along a
        boxy bevel, its rail --, ("R", ring), ("S", boxy bevel) on its end
        faces (its profile), or None (a fillet's end, a corner patch: no edge)."""
        if isinstance(j, tuple):
            return ("G", j[1], True)
        role = self.role.get(j, ("plane",)) if j is not None else ("plane",)
        if j is None:
            return ("P", -1, True)
        if role[0] == "plane":
            return ("P", j, True)
        if role[0] == "grid":
            return ("G", role[1], True)
        if role[0] == "band":
            B = self.bands[role[1]]
            if B["closed"]:
                return ("R", role[1], False)
            if B.get("kind") == "cut":
                return ("C", role[1], False)
            if i in B["planes"]:
                if B.get("kind") == "boxy":
                    return ("P", B["rail"][i], False)
                return ("P", [p for p in B["planes"] if p != i][0], False)
            if B.get("kind") == "boxy" and i in B["ends"]:
                return ("S", role[1], False)
        return None

    def _corner_point(self, i, q, s, fallback):
        """Where planes i, q, s meet; the mesh's vertex if they hardly do."""
        if q < 0 or s < 0:
            return fallback
        A = np.array([self.pn[i], self.pn[q], self.pn[s]])
        if abs(np.linalg.det(A)) < 1e-6:
            return fallback
        return np.linalg.solve(A, [self.pd[i], self.pd[q], self.pd[s]])

    def outline(self, i):
        """(outer, inners, ring ids) of plane region i on the sharp body;
        outer None when the region is a ring's cap (the cylinder makes it).
        The ring ids are those of its outermost outline: a ring through an
        inner one (the hole through a tube's end) is not shut by it."""
        V = self.V
        polys, rings, big = [], [], -1.0
        for lp in self.loops(i):
            labs = [self.label(i, j) for _, j in lp]
            keys = {l[:2] for l in labs if l is not None}
            if keys and all(k[0] == "R" for k in keys):
                P = V[[a for a, _ in lp]]
                a = float(np.linalg.norm(np.cross(P, np.roll(P, -1, 0)).sum(0)))
                if a > big:
                    big, rings = a, [k[1] for k in keys]
                continue
            n = len(lp)
            st = next((k for k in range(n) if labs[k] is not None and labs[k - 1] != labs[k]), None)
            if any(k[0] == "R" for k in keys) or st is None or self.faceted:
                pts = [V[a] for a, _ in lp]                    # a ring through the outline, facets: the mesh's outline
            else:
                P = V[[a for a, _ in lp]]
                d1, d2 = P - np.roll(P, 1, 0), np.roll(P, -1, 0) - P
                bend = np.linalg.norm(np.cross(d1, d2), axis=1) > 1e-4 * np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)
                runs = []                                      # [label, first vertex, gap before, its vertices]
                gap = False
                for k in range(n):
                    kk = (st + k) % n
                    a, l = lp[kk][0], labs[kk]
                    if l is None:
                        gap = True
                    elif runs and not gap and runs[-1][0] == l and not (l[0] == "P" and l[2] and bend[kk]):
                        runs[-1][3].append(a)                  # (along one plane the outline turns only on the mesh)
                        continue
                    else:
                        if runs and not gap:
                            runs[-1][3].append(a)
                        runs.append([l, a, gap, [a]])
                        gap = False
                    if gap and runs and runs[-1][3][-1] != a:
                        runs[-1][3].append(a)
                if not gap:
                    runs[-1][3].append(runs[0][1])
                if gap:
                    runs[0][2] = True
                pts = []
                for k in range(len(runs)):
                    (lq, _, _, _), (ls, a, g, rv) = runs[k - 1], runs[k]
                    if lq[:2] == ls[:2] and not (ls[2] and not g):
                        continue
                    if "C" in (lq[0], ls[0]):                   # at a cut cylinder: where it meets the planes
                        C = ls if ls[0] == "C" else lq
                        p = self.bands[C[1]]["corners"].get(a, V[a])
                    elif "G" in (lq[0], ls[0]):                 # beside a free-form patch: the mesh's vertex
                        p = V[a]
                    elif "S" in (lq[0], ls[0]):                 # at a boxy bevel's end: its rail on this face
                        S, q = (ls, lq) if ls[0] == "S" else (lq, ls)
                        rail = self.bands[S[1]]["rail"].get(q[1]) if q[0] == "P" else None
                        p = self._corner_point(i, q[1], rail, V[a]) if rail is not None else V[a]
                    elif not g and lq[2] and ls[2]:
                        p = V[a]                               # two planes meeting on the mesh: its vertex
                    else:
                        p = self._corner_point(i, lq[1], ls[1], V[a])
                    p = list(map(float, p))
                    if ls[0] == "G":                           # the edge from here is the patch's border curve
                        cvs, knots = self.hermite(rv)
                        cvs[0] = p
                        p = {"p": p, "cvs": cvs, "knots": knots, "degree": 3}
                    elif ls[0] == "C" and i in self.bands[ls[1]]["arcs"]:
                        A = self.bands[ls[1]]["arcs"][i]       # the edge from here is the cut cylinder's arc
                        cvs, wts, knots = [list(q) for q in A["cvs"]], list(A["weights"]), list(A["knots"])
                        if np.linalg.norm(np.array(cvs[0]) - p) > np.linalg.norm(np.array(cvs[-1]) - p):
                            cvs, wts, knots = cvs[::-1], wts[::-1], [1.0 - x for x in knots[::-1]]
                        cvs[0] = p
                        p = {"p": p, "cvs": cvs, "knots": knots, "degree": 2, "weights": wts}
                    elif ls[0] == "S":                         # the edge from here is the bevel's profile
                        B = self.bands[ls[1]]
                        cvs, knots = self.profile(ls[1], i, reverse=lq[1] != B["planes"][0])
                        cvs[0] = p
                        p = {"p": p, "cvs": cvs, "knots": knots, "degree": 3}
                    pts.append(p)
                at = lambda q: np.array(q["p"] if isinstance(q, dict) else q, float)
                keep = []                                      # one point per place (a merged near-flat facet
                for q in pts:                                  # repeats a bevel's rail): keep a curve's start
                    if keep and np.linalg.norm(at(keep[-1]) - at(q)) < 10 * self.tol:
                        if isinstance(q, dict):
                            keep[-1] = q
                        continue
                    keep.append(q)
                if len(keep) > 1 and np.linalg.norm(at(keep[-1]) - at(keep[0])) < 10 * self.tol:
                    keep = keep[:-1] if isinstance(keep[0], dict) or not isinstance(keep[-1], dict) else keep[1:]
                pts = keep
                for k, q in enumerate(pts):                    # a profile ends exactly where the next edge starts
                    if isinstance(q, dict):
                        nxt = pts[(k + 1) % len(pts)]
                        q["cvs"][-1] = nxt["p"] if isinstance(nxt, dict) else nxt
            if len(pts) >= 3 or any(isinstance(q, dict) for q in pts) and len(pts) >= 2:
                polys.append([q if isinstance(q, dict) else list(map(float, q)) for q in pts])
        if not polys:
            return None, [], rings
        at = lambda q: q["p"] if isinstance(q, dict) else q
        area = lambda P: float(np.linalg.norm(np.cross(np.array([at(q) for q in P]),
                                                       np.roll(np.array([at(q) for q in P]), -1, 0)).sum(0)))
        polys.sort(key=area, reverse=True)
        return polys[0], polys[1:], rings

    faceted = False

    def job(self, faceted=False):
        """The worker job's geometry: face outlines, cylinders, fillets,
        boxy bevels' sweeps (world). faceted: every region a face, its outline
        the mesh's."""
        if faceted:
            self.bands, self.grids, self.faceted, self.gridface = [], [], True, {}
            self.role = {i: ("plane",) for i in range(len(self.regs))}
        faces, caps = [], {}
        for i in range(len(self.regs)):
            if self.role[i][0] != "plane":
                continue
            if all(f in self.gridface for f in self.regs[i]):
                continue
            fv = self.F[self.regs[i][0]]
            if len(self.regs[i]) == 1 and len(fv) == 4 \
                    and np.abs((self.V[fv] - self.V[fv].mean(0)) @ self.rn[i]).max() > self.tol:
                faces += [[[list(map(float, self.V[q])) for q in t]] for t in ((fv[0], fv[1], fv[2]), (fv[0], fv[2], fv[3]))]
                continue                                       # a warped quad left over: two triangles
            outer, inners, rings = self.outline(i)
            if outer is not None:
                faces.append([outer] + inners)
            else:
                for b in rings:
                    caps.setdefault(b, []).append(float((self.V[self.rverts[i][0]] - self.bands[b]["center"])
                                                        @ self.bands[b]["axis"]))
        pad = max(20 * self.tol, 2e-3 * self.size)
        cyls, fillets, sweeps = [], [], []
        for b, B in enumerate(self.bands):
            if B.get("kind") == "cut":                        # ruled between its arcs on its two end planes
                A0, A1 = (B["arcs"][j] for j in B["ends"])
                sweeps.append({"cvs": A0["cvs"], "cvs2": A1["cvs"], "knots": A0["knots"], "degree": 2,
                               "weights": A0["weights"]})
                continue
            if B.get("kind") == "boxy":                       # ruled between its two ends (faces or mitres)
                (c0, knots), (c1, _) = self.profile(b, B["ends"][0]), self.profile(b, B["ends"][1])
                sweeps.append({"cvs": c0, "cvs2": c1, "knots": knots, "degree": 3})
                continue
            c, ax = B["center"], B["axis"]
            if B["closed"]:
                t0, t1 = B["t"]
                shut = caps.get(b, [])
                t0 -= 0.0 if any(abs(x - t0) < 100 * self.tol for x in shut) else pad
                t1 += 0.0 if any(abs(x - t1) < 100 * self.tol for x in shut) else pad
                cyls.append({"origin": list(map(float, c + ax * t0)), "axis": list(map(float, ax)), "r": float(B["r"]),
                             "h": float(t1 - t0), "cut": bool(B["inward"])})
            else:
                p, q = B["planes"]
                tm = c @ ax + 0.5 * (B["t"][0] + B["t"][1])
                mid = np.linalg.solve(np.array([self.rn[p], self.rn[q], ax]), [self.rd[p], self.rd[q], tm])
                fillets.append((float(B["r"]), list(map(float, mid))))
        groups = []                                            # one fillet per radius, the largest first
        for r, mid in sorted(fillets, key=lambda x: -x[0]):
            if groups and abs(groups[-1]["r"] - r) <= 100 * self.tol:
                groups[-1]["mids"].append(mid)
            else:
                groups.append({"r": r, "mids": [mid]})
        for g in groups:
            g["r"] = float(np.mean([B["r"] for B in self.bands if not B["closed"] and "center" in B and B.get("kind") != "cut"
                                    and abs(B["r"] - g["r"]) <= 100 * self.tol]))
        cyls.sort(key=lambda c: c["cut"])                      # added before cut (with no faces the first is the body)
        patches = [dict(zip(("net", "ku", "kv"), self.patch(G))) for G in self.grids]
        return {"faces": faces, "cylinders": cyls, "fillets": groups, "sweeps": sweeps, "patches": patches,
                "tol": 10 * self.tol}

    def summary(self):
        roles = [r[0] for r in self.role.values()]
        cut = lambda B: B["closed"] or B.get("kind") == "cut"
        holes = [B for B in self.bands if cut(B) and B["inward"]]
        cyls = [B for B in self.bands if cut(B) and not B["inward"]]
        fil = [B for B in self.bands if not cut(B) and "center" in B]
        boxy = [B for B in self.bands if B.get("kind") == "boxy"]
        planes = sum(1 for i, r in self.role.items() if r[0] == "plane"
                     and not all(f in self.gridface for f in self.regs[i]))
        return {"planes": planes, "fillets": [B["r"] for B in fil],
                "holes": [2 * B["r"] for B in holes], "cylinders": [2 * B["r"] for B in cyls],
                "boxy": [(math.exp(-2.0 * math.log(2.0) / B["r"]), 0.5 * (B["w"][0] + B["w"][1])) for B in boxy],
                "patches": len(self.grids)}


def _length(context, x):
    us = context.scene.unit_settings
    return bpy.utils.units.to_string(us.system, 'LENGTH', x * (us.scale_length or 1.0), precision=4)


def describe(context, s):
    """'6 planes · 3 fillets r 6.29" · 1 hole ⌀ 24"' (radii grouped)."""
    def grouped(vals, what, sym):
        if not vals:
            return []
        out = {}
        for v in vals:
            k = next((k for k in out if abs(k - v) <= 1e-6 * max(abs(v), 1.0)), v)
            out[k] = out.get(k, 0) + 1
        return [f"{n} {what}{'s' if n > 1 else ''} {sym} {_length(context, k)}" for k, n in out.items()]
    boxy = {}
    for p, w in s.get("boxy", ()):
        k = next((k for k in boxy if abs(k[0] - p) < 5e-4 and abs(k[1] - w) <= 1e-6 * max(w, 1.0)), (p, w))
        boxy[k] = boxy.get(k, 0) + 1
    return " · ".join([f"{s['planes']} plane{'s' if s['planes'] != 1 else ''}"]
                      + grouped(s["fillets"], "fillet", "r")
                      + [f"{n} bevel{'s' if n > 1 else ''} profile {p:.3g} w {_length(context, w)}" for (p, w), n in boxy.items()]
                      + grouped(s["holes"], "hole", "⌀") + grouped(s["cylinders"], "cylinder", "⌀")
                      + ([f"{s['patches']} free-form patch{'es' if s['patches'] > 1 else ''}"] if s.get("patches") else []))


def convert(context, obj, rel_tol=1e-5, fillets=True, rings=True):
    """(new NURBS object, summary text, warning or None) for a mesh object."""
    V, F, size = mesh_arrays(obj, context.evaluated_depsgraph_get(), rel_tol)
    if not F:
        raise RuntimeError("the mesh has no faces")
    rec = Recognizer(V, F, size, rel_tol, fillets, rings)
    loc, rot, _ = obj.matrix_world.decompose()
    M = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
    axis, angle = rot.to_axis_angle()
    xf = {"loc": list(loc), "axis": list(axis), "angle": math.degrees(angle), "scale": [1.0, 1.0, 1.0]}
    note = None
    try:
        info, breps = jobs.run("mesh_to_brep", {}, ["base", "result"], xform=xf, **rec.job())
        if not info.get("solid") and not rec.open:
            raise jobs.JobError("the rebuilt faces did not close into a solid")
        summary = describe(context, rec.summary())
    except jobs.JobError as exc:
        info, breps = jobs.run("mesh_to_brep", {}, ["base", "result"], xform=xf, **rec.job(faceted=True))
        summary = f"{len(rec.regs)} flat faces"
        note = f"kept the mesh's facets ({exc})"
    new = store.new_object_text(context, obj.name + " BREP", breps["result"], M)
    for coll in obj.users_collection:
        if new.name not in coll.objects:
            coll.objects.link(new)
    if context.collection and new.name in context.collection.objects and context.collection not in obj.users_collection:
        context.collection.objects.unlink(new)
    for mat in obj.data.materials:
        new.data.materials.append(mat)
    done = info.get("filleted", [])
    if done and features.history_enabled(context):
        Mi = M.inverted()
        base = breps["base"]
        for g in done:
            features.push_step(new, "edge_fillet", [], {
                "mids_local": [list(Mi @ Vector(m)) for m in g["mids"]], "shape": "CIRCULAR", "radius": g["r"],
                "end_radius": None, "chord": None, "continuity": 1, "thumbweight": 1.0, "class_a": False,
                "shrink_planes": True},
                base_text=base)
            base = None
    if info.get("failed"):
        bad = ", ".join(f"r {_length(context, r)}" for r in info["failed"])
        note = (note + "; " if note else "") + f"fillet {bad} left sharp (the kernel could not make it)"
    done = {frozenset(B["planes"]) for B in rec.bands if "planes" in B}
    kept = sum(1 for k, regs in rec._kept.items() if k not in done
               and not all(f in rec.gridface for r in regs for f in rec.regs[r]))
    if kept:
        note = (note + "; " if note else "") + f"{kept} boxy bevel{'s' if kept > 1 else ''} " \
            "ending in a corner kept as facets"
    if rec.open:
        note = (note + "; " if note else "") + "the mesh is open: a sheet, not a solid"
    return new, summary, note


class NPROTO_OT_mesh_to_brep(bpy.types.Operator):
    """Rebuild the selected mesh objects as NURBS bodies: flat faces become planes,
    round holes and cylinders exact cylinders, bevels (Profile 0.5) fillets of the
    bevel's radius, which stay editable in the Feature panel. The meshes are kept (hidden)"""
    bl_idname = "nproto.mesh_to_brep"
    bl_label = "Mesh to BREP"
    bl_options = {'REGISTER', 'UNDO'}

    fillets: BoolProperty(name="Fillets", default=True,
                          description="Bevels with a round profile (Profile 0.5, 2 or more segments) become "
                                      "fillets of the bevel's radius; off: they keep their facets")
    rings: BoolProperty(name="Holes and Cylinders", default=True,
                        description="Round holes and cylinders become exact cylinders of the mesh's diameter; "
                                    "off: they keep their facets")
    tolerance: FloatProperty(name="Tolerance", default=1e-5, min=1e-8, max=1e-2, precision=7, step=0.0001,
                             description="How far points may be off a plane or circle and still count, as a "
                                         "fraction of the object's size (raise it for imported or scanned meshes)")
    hide_original: BoolProperty(name="Hide Meshes", default=True,
                                description="Hide the mesh objects in the viewport and in renders (they stay)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(o.type == 'MESH' and not store.is_nurbs(o)
                                                for o in context.selected_objects)

    def execute(self, context):
        srcs = [o for o in context.selected_objects if o.type == 'MESH' and not store.is_nurbs(o)]
        made, lines, warn = [], [], []
        for src in srcs:
            try:
                new, summary, note = convert(context, src, self.tolerance, self.fillets, self.rings)
            except (jobs.JobError, RuntimeError) as exc:
                warn.append(f"{src.name}: not converted ({exc})")
                continue
            made.append((src, new))
            lines.append(f"{new.name}: {summary}")
            if note:
                warn.append(f"{src.name}: {note}")
        if not made:
            self.report({'WARNING'}, "; ".join(warn) or "Nothing converted")
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for src, new in made:
            if self.hide_original:
                src.hide_set(True)
                src.hide_render = True
            new.select_set(True)
        context.view_layer.objects.active = made[-1][1]
        self.report({'WARNING'} if warn else {'INFO'}, "Mesh to BREP: " + "; ".join(lines + warn))
        return {'FINISHED'}


def menu_convert(self, context):
    self.layout.operator(NPROTO_OT_mesh_to_brep.bl_idname, text="NURBS Body (NURBS Proto)", icon='MESH_CUBE')


classes = (NPROTO_OT_mesh_to_brep,)


def register():
    bpy.types.VIEW3D_MT_object_convert.append(menu_convert)


def unregister():
    bpy.types.VIEW3D_MT_object_convert.remove(menu_convert)
