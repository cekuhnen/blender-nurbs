"""SubD to BREP: a Subdivision Surface model (Catmull-Clark on a quad cage)
as a NURBS body with one face per cage face, G1 or G2 across every edge.
(Triangles and n-gons: MacroConverter -- one face per quad, the n-gons split
into the quads of Catmull-Clark's first step, T-junctions where they meet.)

Catmull-Clark's surface is a uniform bicubic B-spline wherever the mesh is
regular. So every face gets even knots (4 spans for G1, 8 for G2) and its CVs
are the cage subdivided that far, borders clamped: the CV rows run on from
face to face as the subdivided mesh does, and the faces ARE the SubD surface
and join with matching curvature (C2) everywhere but in the first span next to
a star point (a vertex where 3, 5 or more edges meet).

There, no bicubic faces can be the surface and also join smoothly, so a few
CVs are solved for instead. One extra knot very close to the star point
(1/512 of the edge, inside the ends Continuity Check leaves out) frees each
face's corner; past it the faces are made to join exactly -- parametrically
C1 (G1) or C2 (G2), linear conditions on the CVs -- while staying as close as
they can to the SubD surface and to the subdivided cage's CVs. The first CVs
of the edges at the point lie in its tangent plane as an affine-regular
n-gon (as Catmull-Clark's tangents do), so the faces meet tangentially at the
point too. One linear solve per star point.

Creases (crease 1.0) are sharp edges: each side follows its own surface."""


import math

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from mathutils import Matrix

from . import jobs, store

SPANS = (4, 8)              # G1, G2: even spans per face and direction (CVs straight from the cage subdivided that far)
SLIVER_KNOT = 1 / 512       # a knot this far from each end of an edge at a star point frees the face corners there
JOINT_K = 4                 # at a star point the CVs this far each way (corner blocks, edges' first ones) are solved for
JOINT_C2 = (False, True)    # G1, G2: past the sliver the faces join parametrically C1 (G1) or C2 (G2)
W_NATURAL = (0.02, 0.005)   # G1, G2: how strongly the solved CVs stay at the subdivided cage's (per their span)
W_FAIR = (1.0, 1.0)         # G1, G2: how strongly their rows keep bending as the cage's rows do
CHECK_ENDS = 0.015          # an edge's ends Continuity Check leaves out (and so does the report)
PRUNE_CELLS = 8             # below the base level, cells kept around an irregular corner (each level)
REGULAR = np.array([0, 0, 0, 0, 1, 1, 1, 1], float)


# -- the cage ----------------------------------------------------------------------
def cage(obj, depsgraph):
    """(V, F, sharp, matrix): the control cage in object space with the
    modifiers before Subdivision applied, its quads, its creased (1.0) and
    open edges, and the object's world matrix."""
    subs = [m for m in obj.modifiers if m.type == 'SUBSURF']
    shown = [m.show_viewport for m in subs]
    for m in subs:
        m.show_viewport = False
    try:
        depsgraph.update()
        ev = obj.evaluated_get(depsgraph)
        me = ev.to_mesh()
        V = np.array([v.co for v in me.vertices], float)
        F = [tuple(p.vertices) for p in me.polygons]
        cr = me.attributes.get("crease_edge")
        sharp = {tuple(sorted(e.vertices)) for e, d in zip(me.edges, cr.data) if d.value >= 0.999} if cr else set()
        ev.to_mesh_clear()
    finally:
        for m, s in zip(subs, shown):
            m.show_viewport = s
        depsgraph.update()
    F = [tuple(int(x) for x in f) for f in F]
    if all(len(f) == 4 for f in F):                 # (triangles / n-gons: build() splits them, MacroConverter)
        sharp = with_borders(F, sharp)
    return V, F, sharp, obj.matrix_world.copy()


def with_borders(F, sharp):
    """sharp and the open borders (edges of one face): those behave like creases."""
    uses = {}
    for f in F:
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            uses[e] = uses.get(e, 0) + 1
    return set(sharp) | {e for e, n in uses.items() if n != 2}


def refine_any(V, F, sharp):
    """One Catmull-Clark step on a mesh with any polygons (quads out): the vertex
    points first (same indices), then the edge points, then the face points;
    face k's children consecutively, child i (F[k][i], its edge point towards
    F[k][i + 1], the face point, its edge point towards F[k][i - 1])."""
    fp = np.array([V[list(f)].mean(0) for f in F])
    edges = {}
    for k, f in enumerate(F):
        for i in range(len(f)):
            edges.setdefault(tuple(sorted((f[i], f[(i + 1) % len(f)]))), []).append(k)
    ek = list(edges)
    eid = {e: len(V) + i for i, e in enumerate(ek)}
    ep = np.array([0.5 * (V[a] + V[b]) if (a, b) in sharp or len(edges[(a, b)]) != 2
                   else 0.25 * (V[a] + V[b] + fp[edges[(a, b)][0]] + fp[edges[(a, b)][1]]) for a, b in ek])
    nv = _vertex_points(V, F, fp, ek, sharp)
    V2 = np.vstack([nv, ep, fp])
    F2, s2 = [], set()
    for k, f in enumerate(F):
        c = len(V) + len(ek) + k
        for i in range(len(f)):
            a, b, d = f[i], f[(i + 1) % len(f)], f[i - 1]
            F2.append((a, eid[tuple(sorted((a, b)))], c, eid[tuple(sorted((a, d)))]))
    for a, b in sharp:
        e = eid[(a, b)]
        s2 |= {tuple(sorted((a, e))), tuple(sorted((b, e)))}
    return V2, F2, s2


def _vertex_points(V, F, fp, ek, sharp):
    vf = [[] for _ in V]
    ve = [[] for _ in V]
    for k, f in enumerate(F):
        for a in f:
            vf[a].append(k)
    for a, b in ek:
        ve[a].append(b)
        ve[b].append(a)
    out = np.empty_like(V)
    for v in range(len(V)):
        sh = [u for u in ve[v] if tuple(sorted((u, v))) in sharp]
        n = len(ve[v])
        if len(sh) >= 3 or not n:
            out[v] = V[v]
        elif len(sh) == 2:
            out[v] = (V[sh[0]] + 6 * V[v] + V[sh[1]]) / 8
        else:
            out[v] = (fp[vf[v]].mean(0) + V[v] + V[ve[v]].mean(0) + (n - 3) * V[v]) / n   # (Q + 2R + (n-3)S) / n
    return out


# -- the limit surface ---------------------------------------------------------------
def rings(F, nv):
    """Per vertex, its ring in order: (edge neighbours, the face diagonals
    between them), or None on an open border."""
    ring = [[] for _ in range(nv)]
    for f in F:
        for i in range(4):
            ring[f[i]].append((f[(i + 1) % 4], f[(i + 2) % 4], f[i - 1]))
    out = []
    for r in ring:
        nxt = {p: (a, d) for a, d, p in r}                         # the face after edge p: (its next edge, diagonal)
        e0 = r[0][2] if r else None
        es, fs = [], []
        for _ in range(len(r)):
            if e0 not in nxt:
                break
            a, d = nxt[e0]
            es.append(e0)
            fs.append(d)
            e0 = a
        out.append((es, fs) if len(r) >= 3 and len(es) == len(r) and e0 == r[0][2] else None)
    return out


def _masks(n):
    """Catmull-Clark's limit tangent masks (Halstead, Kass and DeRose 1993) on a
    ring of n: weights of the edge neighbours and of the diagonals for the
    tangents along the first edge and a quarter turn on."""
    c, sn = np.cos(2 * np.pi * np.arange(n) / n), np.sin(2 * np.pi * np.arange(n) / n)
    A = 1 + math.cos(2 * math.pi / n) + math.cos(math.pi / n) * math.sqrt(2 * (9 + math.cos(2 * math.pi / n)))
    return (A * c, c + np.roll(c, -1)), (A * sn, sn + np.roll(sn, -1))


def edge_tangent(V, ring, j):
    """Direction in which the limit surface leaves a smooth vertex along its
    j-th edge (the tangent masks turned to that edge)."""
    es, fs = ring
    (ce, cf), _ = _masks(len(es))
    d = np.roll(ce, j) @ V[es] + np.roll(cf, j) @ V[fs]
    return d / (np.linalg.norm(d) or 1.0)


def limit_normals(V, F, sharp):
    """Unit normal of the limit surface at each vertex's limit point, from
    the tangent masks on its ring; NaN at vertices on creases and open borders."""
    out = np.full((len(V), 3), np.nan)
    on = {v for e in sharp for v in e}
    for v, r in enumerate(rings(F, len(V))):
        if r is None or v in on:
            continue
        (ce, cf), (se, sf) = _masks(len(r[0]))
        E, Fd = V[r[0]], V[r[1]]
        nv = np.cross(ce @ E + cf @ Fd, se @ E + sf @ Fd)
        out[v] = nv / (np.linalg.norm(nv) or 1.0)
    return out


# -- Catmull-Clark, vectorised -----------------------------------------------------------
def _scatter(idx, vals, n):
    """Sum of vals (k x 3) per index (n x 3) -- np.add.at, but fast."""
    return np.stack([np.bincount(idx, weights=vals[:, c], minlength=n) for c in range(3)], 1)


def cc_step(V, F, sharp_keys, UV):
    """One Catmull-Clark step on quads. sharp_keys: sorted keys a * len(V) + b
    (a < b) of the creased edges (open borders count as creased too); UV: each
    quad's corners in its cage face's (u, v). Returns the same for the result."""
    nv, m = len(V), len(F)
    fp = V[F].mean(1)
    lo, hi = np.minimum(F, np.roll(F, -1, axis=1)), np.maximum(F, np.roll(F, -1, axis=1))
    uk, inv = np.unique((lo.astype(np.int64) * nv + hi).ravel(), return_inverse=True)
    inv = inv.reshape(m, 4)
    ne = len(uk)
    ea, eb = (uk // nv).astype(int), (uk % nv).astype(int)
    cnt = np.bincount(inv.ravel(), minlength=ne)
    fp4 = np.repeat(fp, 4, axis=0)
    fsum = _scatter(inv.ravel(), fp4, ne)
    crease = np.isin(uk, sharp_keys)
    hard = crease | (cnt != 2)
    mid = 0.5 * (V[ea] + V[eb])
    ep = np.where(hard[:, None], mid, 0.25 * (V[ea] + V[eb] + fsum))
    val = np.bincount(ea, minlength=nv) + np.bincount(eb, minlength=nv)
    nf = np.bincount(F.ravel(), minlength=nv)
    Q = _scatter(F.ravel(), fp4, nv)
    R = _scatter(np.r_[ea, eb], np.vstack([mid, mid]), nv)
    ns = np.bincount(ea[hard], minlength=nv) + np.bincount(eb[hard], minlength=nv)
    sv = _scatter(np.r_[ea[hard], eb[hard]], np.vstack([V[eb[hard]], V[ea[hard]]]), nv)
    n = np.maximum(val, 1).astype(float)[:, None]
    smooth = (Q / np.maximum(nf, 1)[:, None] + 2 * R / n + (n - 3) * V) / n      # (Q + 2R + (n-3)S) / n
    vp = np.where((ns >= 3)[:, None] | (val == 0)[:, None], V, np.where((ns == 2)[:, None], (sv + 6 * V) / 8, smooth))
    V2 = np.vstack([vp, ep, fp])
    fc = (np.arange(m) + nv + ne)[:, None].repeat(4, 1)
    F2 = np.stack([F, inv + nv, fc, np.roll(inv, 1, axis=1) + nv], -1).reshape(-1, 4)
    UV2 = np.stack([UV, 0.5 * (UV + np.roll(UV, -1, axis=1)), UV.mean(1, keepdims=True).repeat(4, 1),
                    0.5 * (UV + np.roll(UV, 1, axis=1))], 2).reshape(-1, 4, 2)
    nv2 = len(V2)
    ce = np.nonzero(crease)[0]
    st = np.vstack([np.c_[np.minimum(ea[ce], nv + ce), np.maximum(ea[ce], nv + ce)],
                    np.c_[np.minimum(eb[ce], nv + ce), np.maximum(eb[ce], nv + ce)]]).astype(np.int64)
    return V2, F2, np.sort(st[:, 0] * nv2 + st[:, 1]), UV2


def cc_limit(V, F, sharp_keys):
    """Each vertex's limit point."""
    nv = len(V)
    lo, hi = np.minimum(F, np.roll(F, -1, axis=1)), np.maximum(F, np.roll(F, -1, axis=1))
    uk, inv = np.unique((lo.astype(np.int64) * nv + hi).ravel(), return_inverse=True)
    ea, eb = (uk // nv).astype(int), (uk % nv).astype(int)
    hard = np.isin(uk, sharp_keys) | (np.bincount(inv, minlength=len(uk)) != 2)
    val = np.bincount(ea, minlength=nv) + np.bincount(eb, minlength=nv)
    E = _scatter(np.r_[ea, eb], np.vstack([V[eb], V[ea]]), nv)
    D = _scatter(F.ravel(), V[np.roll(F, -2, axis=1)].reshape(-1, 3), nv)
    ns = np.bincount(ea[hard], minlength=nv) + np.bincount(eb[hard], minlength=nv)
    sv = _scatter(np.r_[ea[hard], eb[hard]], np.vstack([V[eb[hard]], V[ea[hard]]]), nv)
    n = np.maximum(val, 1).astype(float)[:, None]
    return np.where((ns >= 3)[:, None], V, np.where((ns == 2)[:, None], (sv + 4 * V) / 6,
                                                     (n * n * V + 4 * E + D) / (n * (n + 5))))


def _bspline4(x):
    """The four uniform cubic B-spline weights at x in [0, 1]."""
    return np.stack([(1 - x) ** 3, 3 * x ** 3 - 6 * x ** 2 + 4, -3 * x ** 3 + 3 * x ** 2 + 3 * x + 1, x ** 3], -1) / 6


class Levels:
    """The cage subdivided to level `top`, every level kept as grids of control
    points with one ring of ghosts beyond each side (from the neighbouring
    faces, mirrored at a crease): at level `base` per cage face, below it only
    a small window at each irregular corner (all the finer levels are needed for)."""

    WINDOW = PRUNE_CELLS + 1                                       # a corner window's last grid index

    def __init__(self, V, F, sharp, top, base=0, near=()):
        """near: (face, corner index, level) at irregular vertices -- beyond
        level `base` only the cells around them are subdivided further, each
        corner down to its level (the surface elsewhere is known exactly from
        level base)."""
        nv = len(V)
        keys = np.sort(np.array([a * nv + b for a, b in sharp], np.int64))
        m = len(F)
        UV = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), (m, 1, 1))
        orig = np.arange(m)
        hard = np.array([[(min(fv[s], fv[(s + 1) % 4]), max(fv[s], fv[(s + 1) % 4])) in sharp for s in range(4)]
                         for fv in F]).reshape(m, 4)
        corner_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        near_of = [[(i, d) for g, i, d in near if g == f] for f in range(m)]
        self.base, self.ext, self.win = base, {}, {}
        W = self.WINDOW
        Vk, Fk, Kk = V.copy(), np.asarray(F), keys
        for lev in range(top + 1):
            if lev:
                Vk, Fk, Kk, UV = cc_step(Vk, Fk, Kk, UV)
                orig = np.repeat(orig, 4)
            N = 2 ** lev
            if lev > base:                                         # keep the cells near irregular corners only
                c = UV.mean(1)
                r = np.full(len(Fk), np.inf)
                for f in range(m):
                    qs = [corner_uv[i] for i, d in near_of[f] if d >= lev]
                    if qs:
                        sel = orig == f
                        for q in qs:
                            r[sel] = np.minimum(r[sel], np.abs(c[sel] - q).max(1))
                keep = r <= PRUNE_CELLS / N
                Fk, UV, orig = Fk[keep], UV[keep], orig[keep]
                used = np.unique(Fk)
                remap = -np.ones(len(Vk), int)
                remap[used] = np.arange(len(used))
                a_, b_ = Kk // len(Vk), Kk % len(Vk)
                ok = (remap[a_] >= 0) & (remap[b_] >= 0)
                na, nb = remap[a_[ok]], remap[b_[ok]]
                Kk = np.sort(np.minimum(na, nb).astype(np.int64) * len(used) + np.maximum(na, nb))
                Vk, Fk = Vk[used], remap[Fk]
                # per irregular corner a window of the grid, turned so that the corner is its (0, 0)
                wins = [(f, i) for f in range(m) for i, d in near_of[f] if d >= lev]
                grid = -np.ones((len(wins), W + 1, W + 1), int)
                ij = np.rint(UV * N).astype(int)
                for w, (f, i) in enumerate(wins):
                    sel = orig == f
                    a, b, ids = ij[sel, :, 0], ij[sel, :, 1], Fk[sel]
                    a, b = (N - a if i in (1, 2) else a), (N - b if i in (2, 3) else b)
                    inside = (a <= W) & (b <= W)
                    grid[w, a[inside], b[inside]] = ids[inside]
                sides = np.array([[hard[f, 0 if i in (0, 1) else 2], False, False, hard[f, 3 if i in (0, 3) else 1]]
                                  for f, i in wins]).reshape(len(wins), 4)
                G = self._extend(Vk, Fk, grid, W, sides, [(w, 0) for w in range(len(wins))])
                self.win[lev] = {}
                for w, (f, i) in enumerate(wins):
                    self.win[lev].setdefault(f, []).append((i, G[w]))
            elif lev == base:
                grid = -np.ones((m, N + 1, N + 1), int)
                ij = np.rint(UV * N).astype(int)
                grid[orig[:, None], ij[:, :, 0], ij[:, :, 1]] = Fk
                self.ext[lev] = self._extend(Vk, Fk, grid, N, hard, None)
                limit = cc_limit(Vk, Fk, Kk)
                self.corners = np.array([[limit[grid[f, i, j]] for i, j in ((0, 0), (N, 0), (N, N), (0, N))]
                                         for f in range(m)])
                self.limit = limit[grid]                           # (m, N + 1, N + 1, 3): the surface at (i, j) / N

    @staticmethod
    def _extend(V, F, grid, N, hard, near=None):
        nv, m = len(V), len(grid)
        lo, hi = np.minimum(F, np.roll(F, -1, axis=1)), np.maximum(F, np.roll(F, -1, axis=1))
        uk = np.unique((lo.astype(np.int64) * nv + hi).ravel())
        ea, eb = (uk // nv).astype(int), (uk % nv).astype(int)
        nb = _scatter(np.r_[ea, eb], np.vstack([V[eb], V[ea]]), nv)
        val = np.bincount(ea, minlength=nv) + np.bincount(eb, minlength=nv)
        G = np.zeros((m, N + 3, N + 3, 3))
        G[:, 1:N + 2, 1:N + 2] = V[grid]
        t = np.arange(1, N)
        for s in range(4):                                         # along each side: across from the face
            c = (np.c_[t, 0 * t], np.c_[0 * t + N, t], np.c_[t, 0 * t + N], np.c_[0 * t, t])[s]
            d = ((0, 1), (-1, 0), (0, -1), (1, 0))[s]
            al = (1, 0) if s in (0, 2) else (0, 1)
            P = grid[:, c[:, 0], c[:, 1]]
            ghost = nb[P] - V[grid[:, c[:, 0] + d[0], c[:, 1] + d[1]]] - V[grid[:, c[:, 0] + al[0], c[:, 1] + al[1]]] \
                - V[grid[:, c[:, 0] - al[0], c[:, 1] - al[1]]]
            mirror = 2 * V[P] - V[grid[:, c[:, 0] + d[0], c[:, 1] + d[1]]]
            bad = (val[P] != 4) | hard[:, s][:, None]
            G[:, c[:, 0] - d[0] + 1, c[:, 1] - d[1] + 1] = np.where(bad[..., None], mirror, ghost)
        order = np.argsort(F.ravel(), kind="stable")
        starts = np.searchsorted(F.ravel()[order], np.arange(nv + 1))
        for ci, cj in ((0, 0), (N + 2, 0), (N + 2, N + 2), (0, N + 2)):   # beyond the corners: a guess first
            di, dj = (1 if ci == 0 else -1), (1 if cj == 0 else -1)
            G[:, ci, cj] = G[:, ci + di, cj] + G[:, ci, cj + dj] - G[:, ci + di, cj + dj]
        wanted = None if near is None else set(near)
        for f in range(m):                                         # at the corners: the next edge round
            for ci_n, ((i, j), sides) in enumerate(zip(((0, 0), (N, 0), (N, N), (0, N)), ((0, 3), (0, 1), (1, 2), (2, 3)))):
                p = grid[f, i, j]
                if p < 0 or (wanted is not None and (f, ci_n) not in wanted):
                    continue
                nxt = None
                gh = []
                for s_ in sides:
                    d = ((0, 1), (-1, 0), (0, -1), (1, 0))[s_]
                    q = grid[f, i + d[0], j + d[1]]
                    al = (1 if i == 0 else -1, 0) if s_ in (0, 2) else (0, 1 if j == 0 else -1)
                    a_ = grid[f, i + al[0], j + al[1]]
                    g = None
                    if not hard[f, s_]:
                        if nxt is None:
                            nxt = {}
                            for fi in order[starts[p]:starts[p + 1]] // 4:
                                qq = F[fi]
                                k_ = int(np.nonzero(qq == p)[0][0])
                                nxt[int(qq[k_ - 1])] = int(qq[(k_ + 1) % 4])
                            prv = {b_: a2 for a2, b_ in nxt.items()}
                        g = prv.get(a_) if nxt.get(a_) == q else nxt.get(a_) if prv.get(a_) == q else None
                    G[f, i - d[0] + 1, j - d[1] + 1] = V[g] if g is not None else 2 * V[p] - V[q]
                    gh.append(g)
                if nxt is not None and len(nxt) == 4 and None not in gh:   # an ordinary corner: the true diagonal
                    for fi in order[starts[p]:starts[p + 1]] // 4:
                        qq = list(F[fi])
                        if gh[0] in qq and gh[1] in qq:
                            k_ = qq.index(p)
                            ci_, cj_ = (0 if i == 0 else N + 2), (0 if j == 0 else N + 2)
                            G[f, ci_, cj_] = V[qq[(k_ + 2) % 4]]
        return G

    def evaluate(self, f, u, v, lev):
        """Face f's level-lev uniform B-spline at (u, v): the SubD surface
        wherever the cell's 4 x 4 control points are a regular grid (below the
        base level: near an irregular corner, from its window)."""
        N = 2 ** lev
        u, v = np.asarray(u, float), np.asarray(v, float)
        if lev == self.base:
            return self._spline(self.ext[lev][f], u * N, v * N, N)
        wins = self.win[lev][f]
        cu = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])[[i for i, _ in wins]]
        r = np.maximum(np.abs(u[:, None] - cu[:, 0]), np.abs(v[:, None] - cu[:, 1]))
        pick = r.argmin(1)
        out = np.zeros((len(u), 3))
        for k, (i, G) in enumerate(wins):
            sel = pick == k
            if sel.any():
                a, b = (1 - u[sel] if i in (1, 2) else u[sel]), (1 - v[sel] if i in (2, 3) else v[sel])
                out[sel] = self._spline(G, a * N, b * N, self.WINDOW)
        return out

    @staticmethod
    def _spline(G, x, y, N):
        """The uniform cubic B-spline of the ghost-extended grid G at (x, y) in grid units."""
        I = np.clip(np.floor(x).astype(int), 0, N - 1)
        J = np.clip(np.floor(y).astype(int), 0, N - 1)
        C = G[(I[:, None] + np.arange(4))[:, :, None], (J[:, None] + np.arange(4))[:, None, :]]
        return np.einsum("na,nb,nabc->nc", _bspline4(x - I), _bspline4(y - J), C)


def samples(knots, per=3):
    """Parameters in every knot span (and the knots)."""
    br = np.unique(knots)
    return np.unique(np.r_[br, (br[:-1, None] + np.diff(br)[:, None] * np.linspace(0, 1, per + 2)[1:-1]).ravel()])


def even_knots(a, b, L):
    """Knots: 2^L even spans, and a sliver knot SLIVER_KNOT from the start (a)
    and from the end (b) where they reach an irregular vertex."""
    inner = [i / 2 ** L for i in range(1, 2 ** L)] + ([SLIVER_KNOT] if a else []) + ([1 - SLIVER_KNOT] if b else [])
    return np.r_[[0.0] * 4, np.unique(np.round(inner, 12)), [1.0] * 4]


# -- B-splines --------------------------------------------------------------------------
def basis(knots, s, der=0, p=3):
    """B-spline basis (or its derivative) of degree p at parameters s: (len(s), n)."""
    s = np.atleast_1d(np.asarray(s, float))
    n = len(knots) - p - 1
    if der:
        M = basis(knots, s, der - 1, p - 1)
        a, b = knots[p:p + n] - knots[:n], knots[p + 1:p + 1 + n] - knots[1:n + 1]
        return M[:, :n] * np.divide(p, a, out=np.zeros(n), where=a > 0) \
            - M[:, 1:n + 1] * np.divide(p, b, out=np.zeros(n), where=b > 0)
    N = ((knots[:-1] <= s[:, None]) & (s[:, None] < knots[1:])).astype(float)
    last = np.nonzero(knots[:-1] < knots[1:])[0][-1]
    N[s >= knots[-1]] = 0.0
    N[s >= knots[-1], last] = 1.0
    for d in range(1, p + 1):
        a, b = knots[d:-1] - knots[:-d - 1], knots[d + 1:] - knots[1:-d]
        N = np.divide(s[:, None] - knots[:-d - 1], a, out=np.zeros((len(s), len(a))), where=a > 0) * N[:, :-1] \
            + np.divide(knots[d + 1:] - s[:, None], b, out=np.zeros((len(s), len(b))), where=b > 0) * N[:, 1:]
    return N[:, :n]


def rows(knots, u, v, du=0, dv=0, kv=None):
    """d S / du^du dv^dv at (u, v) as combinations of the CVs (u-major; knots
    in u, kv in v, the same if not given)."""
    Bu, Bv = basis(knots, u, du), basis(knots if kv is None else kv, v, dv)
    return (Bu[:, :, None] * Bv[:, None, :]).reshape(len(Bu), -1)


def derivs(net, ku, kv, u, v, ders):
    """{(a, b): d^(a+b) S / du^a dv^b at the points (u, v)} of a face's net,
    from 1D bases (no dense rows); the degrees from the (clamped) knots."""
    pu, pv = int(np.count_nonzero(ku == ku[0])) - 1, int(np.count_nonzero(kv == kv[0])) - 1
    Bu = {a: basis(ku, u, a, pu) for a in {a for a, _ in ders}}
    Bv = {b: basis(kv, v, b, pv) for b in {b for _, b in ders}}
    X = {a: np.tensordot(B, net, axes=(1, 0)) for a, B in Bu.items()}      # (t, nv, 3); no einsum path search
    return {(a, b): np.einsum("tj,tjc->tc", Bv[b], X[a]) for a, b in ders}  # (it was 40% of einsum's time)


SIDE_UV = (lambda t: (t, 0 * t), lambda t: (1 + 0 * t, t), lambda t: (1 - t, 1 + 0 * t), lambda t: (0 * t, 1 - t))
INWARD = ((0, 1), (-1, 0), (0, -1), (1, 0))
ALONG = ((1, 0), (0, 1), (-1, 0), (0, -1))
NEAR_SIDE = (lambda m: m[:, :3], lambda m: m[-3:, :], lambda m: m[:, -3:], lambda m: m[:3, :])   # a net's 3 rows by side k


def _clamping(N):
    """C: the clamped CVs (knots 0,0,0,0,1/N,...,1,1,1,1) of a uniform cubic
    B-spline over N spans from its N+3 control points (one beyond each end)."""
    t = np.linspace(0, N, 40 * N + 1)
    ku = np.arange(-3, N + 4, dtype=float)
    kc = np.r_[[0.0] * 3, np.arange(0, N + 1, dtype=float), [float(N)] * 3]
    return np.linalg.lstsq(basis(kc, t), basis(ku, t), rcond=None)[0], kc / N


def _side_cells(N, s, M=None):
    """A face side's grid cells (in the side's direction) and the step into the
    face; N, M: the net's last index along u and along v (M = N if not given)."""
    M = N if M is None else M
    return ([(i, 0) for i in range(N + 1)], (0, 1)) if s == 0 else ([(N, j) for j in range(M + 1)], (-1, 0)) if s == 1 else \
        ([(N - i, M) for i in range(N + 1)], (0, -1)) if s == 2 else ([(0, M - j) for j in range(M + 1)], (1, 0))


# -- the conversion ------------------------------------------------------------------
class Converter:
    def __init__(self, V, F, sharp, angle_tol=0.05, curv_tol=0.02):
        self.V, self.F, self.sharp = V, F, sharp
        self.angle_tol, self.curv_tol = angle_tol, curv_tol
        self.size = float(np.linalg.norm(np.ptp(V, 0))) or 1.0
        self.edges, self.vf = {}, [[] for _ in V]
        for k, f in enumerate(F):
            for i in range(4):
                a, b = f[i], f[(i + 1) % 4]
                self.edges.setdefault((min(a, b), max(a, b)), []).append((k, i, a > b))
            for a in f:
                self.vf[a].append(k)
        self.open = any(len(u) != 2 for u in self.edges.values())
        self.ring = rings(F, len(V))
        self.creased = {v for e in sharp for v in e}
        self.nbr = [set() for _ in V]
        for a, b in self.edges:
            self.nbr[a].add(b)
            self.nbr[b].add(a)
        self.val = val = np.zeros(len(V), int)
        for a, b in self.edges:
            val[a] += 1
            val[b] += 1
        ncrease = np.zeros(len(V), int)
        for a, b in sharp:
            ncrease[a] += 1
            ncrease[b] += 1
        self.star = (val != 4) | ((ncrease != 0) & (ncrease != 2)) | np.array([len(self.vf[v]) != val[v] for v in range(len(V))])
        self.cage_normals = limit_normals(V, F, sharp)
        self._vn, self._arc = {}, {}
        self.irr = set(self.irregular())
        self.deviation = 0.0

    def _sector(self, v, f):
        """The faces around v joined to face f across smooth edges at v."""
        seen, todo = set(), [f]
        while todo:
            h = todo.pop()
            if h in seen:
                continue
            seen.add(h)
            fv = self.F[h]
            i = fv.index(v)
            for w in (fv[i - 1], fv[(i + 1) % 4]):
                e = (min(v, w), max(v, w))
                if e not in self.sharp:
                    todo += [g for g, _, _ in self.edges[e] if g not in seen]
        return seen

    def vertex_normal(self, v, f):
        """The tangent plane (unit normal) at v of the sector holding face f:
        Catmull-Clark's exact one where v is smooth; on a crease the faces'
        normals there (from the surface right next to v) turned square to the
        crease, at a corner the plane of its two creases."""
        sec = frozenset(self._sector(v, f))
        key = (v, sec)
        if key in self._vn:
            return self._vn[key]
        n = self.cage_normals[v]
        if np.isnan(n).any():
            acc = np.zeros(3)
            bound = set()
            for h in sec:
                fv = self.F[h]
                i = fv.index(v)
                q0 = self.levels.corners[h, i]
                lev = self.ctop[h][i] or self.L
                d = 2.0 ** (1 - lev)
                u, w = self._to_uv(h, v, np.array([d, 0.0]), np.array([0.0, d]))
                P = self.levels.evaluate(h, u, w, lev)
                c = np.cross(P[0] - q0, P[1] - q0)
                c /= np.linalg.norm(c) or 1.0
                acc += c if acc @ c >= 0 else -c
                bound |= {x for x in (fv[i - 1], fv[(i + 1) % 4]) if (min(v, x), max(v, x)) in self.sharp}
            n = acc / (np.linalg.norm(acc) or 1.0)
            creases = [x for x in self.nbr[v] if (min(v, x), max(v, x)) in self.sharp]
            if len(creases) == 2:                                  # along the crease: V_b - V_a
                d = self.V[creases[1]] - self.V[creases[0]]
                d /= np.linalg.norm(d) or 1.0
                n = n - (n @ d) * d
            elif len(creases) > 2 and len(bound) == 2:             # a corner: the plane of its creases
                a, b = sorted(bound)
                c = np.cross(self.V[a] - self.V[v], self.V[b] - self.V[v])
                if np.linalg.norm(c) > 1e-12:
                    n = c * np.sign(c @ n)
            n = n / (np.linalg.norm(n) or 1.0)
        self._vn[key] = n
        return n

    @staticmethod
    def _square(n, tang):
        d = tang / np.maximum(np.linalg.norm(tang, axis=1), 1e-300)[:, None]
        n = n - (n * d).sum(1)[:, None] * d
        return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]

    # -- the faces' topology, as the star-point solve (subd_g2) and the checks see it ----------------
    tedges, tmid, tq, tp = (), {}, {}, {}                          # T-junctions (MacroConverter's)

    def hard(self, e):
        """A creased or open edge: no join across it."""
        return e in self.sharp or len(self.edges[e]) != 2

    def across(self, f, e):
        """The faces across edge e from face f."""
        return [g for g, _, _ in self.edges[e] if g != f]

    def joined_at(self, x):
        """The pairs of faces joined across a smooth edge at x."""
        out = []
        for z in self.nbr[x]:
            e = (min(x, z), max(x, z))
            if len(self.edges[e]) == 2 and e not in self.sharp:
                out.append((self.edges[e][0][0], self.edges[e][1][0]))
        return out

    def joins(self):
        """Per smooth join (its key): the two faces' sides along it, (face, side, A, B): the side's parameter
        is A + B t along the join's t (from its lower vertex)."""
        return [(e, [(f, k, 1.0, -1.0) if flip else (f, k, 0.0, 1.0) for f, k, flip in uses])
                for e, uses in self.edges.items() if len(uses) == 2 and e not in self.sharp]

    def crease_curve(self, a, b):
        """The SubD's own curve along the creased (or open) edge a -> b at t in 0..1: the uniform cubic B-spline
        of the crease polygon (a ghost 2a - b beyond a corner); None where an end is a dart."""
        hn = {}
        for e, us_ in self.edges.items():
            if self.hard(e):
                for x in e:
                    hn.setdefault(x, []).append(e[1] if e[0] == x else e[0])
        if len(hn.get(a, ())) < 2 or len(hn.get(b, ())) < 2:
            return None
        V = self.V

        def outer(x, y):
            nb_ = hn[x]
            return V[nb_[0] if nb_[1] == y else nb_[1]] if len(nb_) == 2 else 2 * V[x] - V[y]
        Qm, Qa, Qb, Qp = outer(a, b), V[a], V[b], outer(b, a)
        bz = np.array([(Qm + 4 * Qa + Qb) / 6, (2 * Qa + Qb) / 3, (Qa + 2 * Qb) / 3, (Qa + 4 * Qb + Qp) / 6])

        def cub(t):
            t = np.asarray(t, float)[:, None]
            return (1 - t) ** 3 * bz[0] + 3 * t * (1 - t) ** 2 * bz[1] + 3 * t * t * (1 - t) * bz[2] + t ** 3 * bz[3]
        return cub

    def fresh_natural(self, L):
        """(nets, knots, corner limit points) of every face at level L, no sliver knots (natural)."""
        c = Converter(self.V, self.F, self.sharp)
        nets = c.natural(L, sliver=False)
        return nets, list(c.fk), c.levels.corners

    def limit_grid(self, level):
        """Per face its SubD limit points at (i, j) / 2^level (level 2 or more)."""
        c = Converter(self.V, self.F, self.sharp)
        c.natural(level, sliver=False)
        return list(c.levels.limit)

    # -- the knots -----------------------------------------------------------------------
    def irregular(self):
        """Vertices where the subdivided cage is not a regular grid: star
        points, and crease vertices other than a crease running straight through."""
        out = []
        for v in range(len(self.V)):
            creases = [w for w in self.nbr[v] if (min(v, w), max(v, w)) in self.sharp]
            straight = False
            if len(creases) == 2 and self.val[v] == 4 and self.ring[v] is not None:
                es = self.ring[v][0]
                straight = abs(es.index(creases[0]) - es.index(creases[1])) == 2
            if self.star[v] or (creases and not straight):
                out.append(v)
        return out

    def strips(self):
        """The edge rings -- the edges facing each other across the quads, whose
        faces share one knot vector along them: per edge its ring and the end
        the ring's direction starts at, per ring whether that direction holds
        all the way round."""
        ring, start, ok = {}, {}, []
        for e0 in self.edges:
            if e0 in ring:
                continue
            rid = len(ok)
            ok.append(True)
            ring[e0], start[e0] = rid, e0[0]
            todo = [e0]
            while todo:
                e = todo.pop()
                for f, k, _ in self.edges[e]:
                    fv = self.F[f]
                    pa, pb = fv[(k + 3) % 4], fv[(k + 2) % 4]      # across the face from side k's two ends
                    s = pa if start[e] == fv[k] else pb
                    eo = (min(pa, pb), max(pa, pb))
                    if eo in ring:
                        ok[rid] = ok[rid] and start[eo] == s
                        continue
                    ring[eo], start[eo] = rid, s
                    todo.append(eo)
        return ring, start, ok

    def layout(self, L, sliver=True):
        """Every face's knots (u, v), its two edge rings': 2^L even spans and a
        sliver knot at each end of the ring that reaches an irregular vertex
        (none without sliver); and per face corner how far in that knot sits
        (2^-t; 0: an ordinary corner)."""
        ring, start, ok = self.strips()
        a, b = np.zeros(len(ok), int), np.zeros(len(ok), int)
        for e, r in ring.items():
            s, t = start[e], (e[1] if start[e] == e[0] else e[0])
            a[r] |= s in self.irr
            b[r] |= t in self.irr
        for r, fine in enumerate(ok):
            if not fine:                                           # a ring that turns round: both ends alike
                a[r] = b[r] = a[r] | b[r]
        kr = [even_knots(a[r] and sliver, b[r] and sliver, L) for r in range(len(ok))]
        self.fk, self.ctop = [], []
        for fv in self.F:
            ks = []
            for k in (0, 1):
                e = (min(fv[k], fv[k + 1]), max(fv[k], fv[k + 1]))
                ks.append(kr[ring[e]] if start[e] == fv[k] else 1 - kr[ring[e]][::-1])
            ku, kv = ks
            self.fk.append((ku, kv))
            tops = []
            for i, x in enumerate(fv):
                if x not in self.irr:
                    tops.append(0)
                    continue
                hu = ku[4] if i in (0, 3) else 1 - ku[-5]
                hv = kv[4] if i in (0, 1) else 1 - kv[-5]
                tops.append(int(round(-math.log2(max(hu, hv)))))
            self.ctop.append(tops)
        self.stop = self.ctop                                      # where the SubD surface is sampled: outside the sliver

    # -- the nets ------------------------------------------------------------------------
    def _to_uv(self, f, v, s, t):
        """Face f's (u, v) at (s, t) from its corner v (s along u, t along v)."""
        i = self.F[f].index(v)
        return (1 - s if i in (1, 2) else s), (1 - t if i in (2, 3) else t)

    def _from(self, f, v):
        """Face f's knots (u, v) seen from its corner v."""
        i = self.F[f].index(v)
        ku, kv = self.fk[f]
        return (ku if i in (0, 3) else 1 - ku[::-1]), (kv if i in (0, 1) else 1 - kv[::-1])

    def exact(self, f, u, v, tops=None):
        """The SubD surface on face f at (u, v), from each point's level (the
        coarsest at which its cell is not an irregular corner's), and where the
        face's knots can follow it: everywhere but the sliver at each such
        corner (tops: how far the knots halve there, the face's if not given)."""
        lev = np.full(len(u), self.L)
        valid = np.ones(len(u), bool)
        for i, (t, x) in enumerate(zip(self.stop[f] if tops is None else tops, self.F[f])):
            if not t:
                continue
            cu, cw = ((0, 0), (1, 0), (1, 1), (0, 1))[i]
            r = np.maximum(np.abs(u - cu), np.abs(v - cw))
            lev = np.maximum(lev, np.clip(np.ceil(-np.log2(np.maximum(r, 1e-300)) - 1e-9), self.L, self.deep[x]).astype(int))
            valid &= r >= 2.0 ** -t - 1e-12
        out = np.zeros((len(u), 3))
        for lv in np.unique(lev):
            sel = lev == lv
            out[sel] = self.levels.evaluate(f, u[sel], v[sel], lv)
        return out, valid

    def corner(self, f, v):
        """Net cells of face f at its corner v: the corner, the next one along u
        and along v, the twist; and the cage vertices those two directions run to."""
        fv = self.F[f]
        i = fv.index(v)
        Mu, Mv = len(self.fk[f][0]) - 5, len(self.fk[f][1]) - 5
        ci, cj = ((0, 0), (Mu, 0), (Mu, Mv), (0, Mv))[i]
        di, dj = (1 if ci == 0 else -1), (1 if cj == 0 else -1)
        tou = (fv[1], fv[0], fv[3], fv[2])[i]
        tov = (fv[3], fv[2], fv[1], fv[0])[i]
        return (ci, cj), (ci + di, cj), (ci, cj + dj), (ci + di, cj + dj), tou, tov

    def natural(self, L, sliver=True):
        """Every face's net on its knots: the cage subdivided to level L, its
        borders clamped, the sliver knots inserted (unless not sliver), the
        corners on their exact limit points -- the SubD surface but the first
        span at irregular vertices."""
        self.layout(L, sliver)
        Lb = max(L, 2) if self.irr else L
        self.L = Lb
        deep = {}                                                  # per vertex the deepest any face there needs
        for f, ts in enumerate(self.stop):
            for x, t in zip(self.F[f], ts):
                deep[x] = max(deep.get(x, 0), t)
        self.deep = deep
        near = [(f, i, deep[x]) for f, fv in enumerate(self.F) for i, x in enumerate(fv) if deep.get(x)]
        self.top = max([Lb] + [t for _, _, t in near])
        self.levels = Levels(self.V, self.F, self.sharp, self.top, base=Lb, near=near)
        N = 2 ** Lb
        C, kL = _clamping(N)
        proj = {}

        def P(k):                                                  # level Lb's knots onto k (exact)
            key = k.tobytes()
            if key not in proj:
                ts = samples(np.union1d(k, kL), 3)
                proj[key] = np.linalg.lstsq(basis(k, ts), basis(kL, ts), rcond=None)[0]
            return proj[key]

        nets = []
        for f in range(len(self.F)):
            Q = np.einsum("ia,jb,abc->ijc", C, C, self.levels.ext[Lb][f], optimize=True)
            for (ci, cj), x in zip(((0, 0), (N + 2, 0), (N + 2, N + 2), (0, N + 2)), self.levels.corners[f]):
                Q[ci, cj] = x                                      # the corners on their exact limit points
            ku, kv = self.fk[f]
            nets.append(np.einsum("ia,jb,abc->ijc", P(ku), P(kv), Q, optimize=True))
        return nets

    def joints(self, nets, g2):
        """At every irregular vertex, the CVs within JOINT_K of it
        (each face's corner block and each edge's first ones) solved for so that
        every two faces join exactly across their edge past the sliver knot --
        parametrically C1 (G1) or C2 (G2), linear conditions on the CVs -- and
        within those, as close as can be to the SubD surface and the cage's CVs."""
        K, size = JOINT_K, self.size
        self.joint_err = 0.0
        checks = []                                                # (face, rows, SubD points): how close the faces stay
        for v in sorted(self.irr):
            var, occ = {}, []

            def key(f, a, b):
                """The unknown at face f's cell (a along u, b along v from corner v), or None (fixed)."""
                c0, cu, cv, _, wu, wv = self.corner(f, v)
                if (a == 0 and b == 0) or a > K or b > K:
                    return None
                if b == 0 or a == 0:
                    w, t = (wu, a) if b == 0 else (wv, b)
                    e = (min(v, w), max(v, w))
                    if e in self.sharp or len(self.edges[e]) != 2:
                        return None
                    return ("e", w, t)
                return ("b", f, a, b)

            def cell(f, a, b):
                c0, cu, cv = self.corner(f, v)[:3]
                return (c0[0] + a * (cu[0] - c0[0]), c0[1] + b * (cv[1] - c0[1]))

            for f in self.vf[v]:
                for a in range(K + 1):
                    for b in range(K + 1):
                        k_ = key(f, a, b)
                        if k_ is None:
                            continue
                        if k_ not in var:
                            var[k_] = len(occ)
                            occ.append([])
                        occ[var[k_]].append((f, cell(f, a, b)))
            if not occ:
                continue
            rows_c = []
            for w in self.nbr[v]:                                  # the joins: across each smooth edge at v
                e = (min(v, w), max(v, w))
                if e in self.sharp or len(self.edges[e]) != 2:
                    continue
                sides = []
                for f, _, _ in self.edges[e]:
                    c0, cu, cv, _, wu, wv = self.corner(f, v)
                    fu, fw = self._from(f, v)
                    along_u = wu == w
                    kc = fw if along_u else fu                     # the knots across the edge, from it
                    d = [basis(kc, [0.0], der)[0, :3] for der in (1, 2)]
                    sides.append((f, along_u, d))
                for t in range(1, K + 1):
                    for der in ((0, 1) if JOINT_C2[g2] else (0,)):
                        row = np.zeros(len(occ))
                        for sgn, (f, along_u, d) in zip((1.0, -1.0 if der else 1.0), sides):
                            for c in range(3):
                                k_ = key(f, t, c) if along_u else key(f, c, t)
                                if k_ is not None:
                                    row[var[k_]] += sgn * d[der][c]
                        rows_c.append(row)
            X0 = np.array([nets[lst[0][0]][lst[0][1]] for lst in occ])
            A, B = [], []
            for f in self.vf[v]:                                   # the SubD surface over the moving CVs' reach
                ku, kv = self.fk[f]
                fu, fw = self._from(f, v)
                zu, zw = fu[K + 4], fw[K + 4]
                su_, sw_ = samples(fu, 2), samples(fw, 2)
                su_, sw_ = su_[su_ <= zu + 1e-12], sw_[sw_ <= zw + 1e-12]
                S, T = np.meshgrid(su_, sw_, indexing="ij")
                u, w_ = self._to_uv(f, v, S.ravel(), T.ravel())
                target, ok = self.exact(f, u, w_)
                R = rows(ku, u[ok], w_[ok], kv=kv)
                checks.append((f, R, target[ok]))
                nv = len(kv) - 4
                cols = np.zeros((R.shape[0], len(occ)))
                fixed = nets[f].reshape(-1, 3).copy()
                for j, lst in enumerate(occ):
                    for h, cl in lst:
                        if h == f:
                            cols[:, j] += R[:, cl[0] * nv + cl[1]]
                            fixed[cl[0] * nv + cl[1]] = 0.0
                A.append(cols / size)
                B.append((target[ok] - R @ fixed) / size)
            # fairness: the moving CVs' rows keep bending as the cage's rows do (second differences along
            # both directions of every face's corner block, against the subdivided cage's)
            Xnat = X0.copy()
            for f in self.vf[v]:
                for a0 in range(K + 2):
                    for b0 in range(K + 2):
                        for da, db in ((1, 0), (0, 1)):
                            trip = [(a0 - da, b0 - db), (a0, b0), (a0 + da, b0 + db)]
                            if min(min(t_) for t_ in trip) < 0:
                                continue
                            row = np.zeros(len(occ))
                            for (a, b), cf in zip(trip, (1.0, -2.0, 1.0)):
                                k_ = key(f, a, b)
                                if k_ is not None:
                                    row[var[k_]] += cf
                            if row.any():
                                A.append(W_FAIR[g2] * row[None, :] / size)
                                B.append(W_FAIR[g2] * (row @ Xnat)[None, :] / size)
            A, B = np.vstack(A), np.vstack(B)
            n3 = 3 * len(occ)
            I3 = np.eye(3)
            # the 3D system (x, y, z of every unknown in turn, then scales): the joins per coordinate, and
            # the first CV of every smooth edge at v on Catmull-Clark's own tangent there, E(i) = v + a T(i)
            # (the tangent masks: an affine-regular n-gon in v's tangent plane), its second one on the same
            # line -- the faces then meet tangentially at v and the edges run straight into it; at a crease
            # or border vertex the first CVs just in its tangent plane
            Crows = [np.kron(np.array(rows_c), I3)] if rows_c else []
            dvals = [np.zeros(3 * len(rows_c))] if rows_c else []
            q = self.levels.corners[self.vf[v][0], self.F[self.vf[v][0]].index(v)]
            ring = self.ring[v]
            smooth = ring is not None and v not in self.creased and len(ring[0]) == self.val[v] \
                and all(("e", w, 1) in var for w in ring[0])
            extra = 1 + (len(ring[0]) if smooth else 0)
            if smooth:
                es, fs = ring
                (ce, cf), _ = _masks(len(es))
                for i, w in enumerate(es):
                    Ti = np.roll(ce, i) @ self.V[es] + np.roll(cf, i) @ self.V[fs]
                    j = var[("e", w, 1)]
                    row = np.zeros((3, n3 + extra))
                    row[:, 3 * j:3 * j + 3] = I3
                    row[:, n3] = -Ti
                    Crows.append(row)
                    dvals.append(q)
                    if ("e", w, 2) in var:                         # the second CV on the same tangent line
                        j2 = var[("e", w, 2)]
                        row = np.zeros((3, n3 + extra))
                        row[:, 3 * j2:3 * j2 + 3] = I3
                        row[:, n3 + 1 + i] = -Ti
                        Crows.append(row)
                        dvals.append(q)
            else:
                for (kind, *rest), j in var.items():
                    if kind == "e" and rest[1] == 1:
                        nrm = self.vertex_normal(v, occ[j][0][0])
                        row = np.zeros((1, n3 + extra))
                        row[0, 3 * j:3 * j + 3] = nrm * size
                        Crows.append(row)
                        dvals.append(np.array([nrm @ q * size]))
            Crows = [r if r.shape[1] == n3 + extra else np.hstack([r, np.zeros((len(r), extra))]) for r in Crows]
            # the pull to the cage's CVs per how much a move there changes the slopes: a CV next to the sliver
            # knot moves the surface's derivatives 1/span times as much as one in an even span
            h0 = 2.0 ** -self.L
            wj = np.ones(len(occ))
            for j, lst in enumerate(occ):
                f, (ci, cj) = lst[0]
                ku, kv = self.fk[f]
                du, dv = (ku[ci + 4] - ku[ci]) / 4, (kv[cj + 4] - kv[cj]) / 4
                wj[j] = h0 / max(min(du, dv), 1e-12)
            mu = np.r_[np.repeat(W_NATURAL[g2] / size * wj, 3), np.full(extra, 1e-9)]
            A3 = np.hstack([np.kron(A, I3), np.zeros((3 * len(A), extra))])   # x, y, z of each unknown in turn, scales
            B3, X03 = B.ravel(), np.r_[X0.ravel(), np.zeros(extra)]
            if Crows:
                C, dv = np.vstack(Crows), np.concatenate(dvals)
                U, Sg, Vt = np.linalg.svd(C, full_matrices=True)
                rank = int((Sg > Sg[0] * 1e-10).sum())
                xp = Vt[:rank].T @ ((U[:, :rank].T @ dv) / Sg[:rank])
                N = Vt[rank:].T
            else:
                C, dv, xp, N = None, None, np.zeros(n3 + extra), np.eye(n3 + extra)
            M = np.vstack([A3 @ N, mu[:, None] * N])
            rhs = np.r_[B3 - A3 @ xp, mu * (X03 - xp)]
            try:                                                   # small and well posed: the normal equations
                y = np.linalg.solve(M.T @ M, M.T @ rhs)
            except np.linalg.LinAlgError:
                y = np.linalg.lstsq(M, rhs, rcond=None)[0]
            Xa = xp + N @ y
            X = Xa[:n3].reshape(-1, 3)
            if C is not None:
                self.joint_err = max(self.joint_err, float(np.abs(C @ Xa - dv).max()))
            for j, lst in enumerate(occ):
                for h, cl in lst:
                    nets[h][cl] = X[j]
        self.deviation = max([0.0] + [float(np.linalg.norm(R @ nets[f].reshape(-1, 3) - P, axis=1).max())
                                      for f, R, P in checks if len(P)])
        self.zone = max([0.0] + [self._from(f, v)[i][K + 4] for v in self.irr for f in self.vf[v] for i in (0, 1)])
        return nets

    def run(self, spans=None, report=None, g2=False):
        """Nets, knots (per face) and stats. spans: even spans per face and
        direction (None: SPANS's; one on a cage without irregular vertices)."""
        self._vn, self._arc = {}, {}
        if not spans:
            spans = SPANS[g2] if self.irr else 1
        L = int(round(math.log2(spans)))
        nets = self.natural(L)
        self.deviation, self.zone, self.joint_err = 0.0, 0.0, 0.0
        todo = set()
        if self.irr:
            before = [n.copy() for n in nets]
            nets = self.joints(nets, g2)
            todo = self.moved_edges(nets, before)
        knots = list(self.fk)
        ends, every = self.edge_jumps(nets, knots, CHECK_ENDS, edges=todo, also_ends=True)
        st = {"normal": max((j for j, _ in ends.values()), default=0.0),
              "normal_all": max((j for j, _ in every.values()), default=0.0),
              "curvature": max((j for j, _ in self.curvature_jumps(nets, knots, edges=todo).values()), default=0.0)
              if g2 else None,
              "spans": 2 ** L, "sliver": SLIVER_KNOT if self.irr else 0.0, "zone": self.zone,
              "deviation": self.deviation, "joins": self.joint_err,
              "cvs": sum(n.shape[0] * n.shape[1] for n in nets), "star_points": len(self.irr)}
        if report:
            report(L, st)
        return nets, knots, st

    @staticmethod
    def _face_kmax(net, ku, kv, m=9):
        """Largest principal curvature of a face (m x m samples): the scale a
        curvature difference across its edges is judged against."""
        g = np.linspace(0, 1, m)
        G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        D = derivs(net, ku, kv, G[:, 0], G[:, 1], ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2)))
        su, sv, suu, suv, svv = D[(1, 0)], D[(0, 1)], D[(2, 0)], D[(1, 1)], D[(0, 2)]
        n = np.cross(su, sv)
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-12
        n = n / np.maximum(ln, 1e-300)[:, None]
        E, F_, G_ = (su * su).sum(1), (su * sv).sum(1), (sv * sv).sum(1)
        e, f, g_ = (suu * n).sum(1), (suv * n).sum(1), (svv * n).sum(1)
        den = np.maximum(E * G_ - F_ * F_, 1e-300)
        with np.errstate(over="ignore", invalid="ignore"):         # a 180 degree corner (a crease running through
            H = (e * G_ - 2 * f * F_ + g_ * E) / (2 * den)         # a 3-edge vertex) has none: left out below
            K = (e * g_ - f * f) / den
            k = np.abs(H) + np.sqrt(np.maximum(H * H - K, 0.0))
        return float(k[ok].max()) if ok.any() else 0.0

    def cross_curvatures(self, net, ku, kv, k, t, A, B):
        """Normal curvature across side k of a face (square to the edge, in
        the tangent plane) at edge parameters t (side parameter A + B t), with
        the face's normals."""
        u, v = SIDE_UV[k](A + B * t)
        D = derivs(net, ku, kv, u, v, ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2)))
        su, sv = D[(1, 0)], D[(0, 1)]
        n = np.cross(su, sv)
        ln = np.linalg.norm(n, axis=1)
        flat = ln < 1e-7 * np.linalg.norm(su, axis=1) * np.linalg.norm(sv, axis=1)       # a corner folded flat
        n /= np.maximum(ln, 1e-300)[:, None]
        (au, av) = ALONG[k]
        T = (-1.0 if B < 0 else 1.0) * (au * su + av * sv)
        b = np.cross(n, T)
        b /= np.maximum(np.linalg.norm(b, axis=1), 1e-300)[:, None]
        E, F_, G_ = (su * su).sum(1), (su * sv).sum(1), (sv * sv).sum(1)
        p_, q_ = (su * b).sum(1), (sv * b).sum(1)
        den = np.maximum(E * G_ - F_ * F_, 1e-300)
        a_, c_ = (G_ * p_ - F_ * q_) / den, (E * q_ - F_ * p_) / den      # b in (u, v)
        kn = a_ * a_ * (D[(2, 0)] * n).sum(1) + 2 * a_ * c_ * (D[(1, 1)] * n).sum(1) + c_ * c_ * (D[(0, 2)] * n).sum(1)
        return np.where(flat, np.nan, kn), n

    def _by_length(self, net, ku, kv, k, A, B, n, skip):
        """Edge parameters (edge direction) evenly spaced by arc length from
        skip to 1 - skip, as Continuity Check samples an edge."""
        key = (id(net), k, A, B, n, skip)
        if key not in self._arc:
            t = np.unique(np.r_[np.linspace(0, 1, 301), 2.0 ** -np.arange(3, 14), 1 - 2.0 ** -np.arange(3, 14)])
            u, v = SIDE_UV[k](A + B * t)
            P = derivs(net, ku, kv, u, v, ((0, 0),))[(0, 0)]
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            self._arc[key] = np.interp(np.linspace(skip, 1 - skip, n) * d[-1], d, t)
        return self._arc[key]

    def curvature_jumps(self, nets, knots, samples=81, edges=None):
        """Per smooth edge (or those in edges), the largest difference of the
        two faces' curvature across it, as a fraction of the larger of the
        edge's largest curvature and the faces' (as Continuity Check judges
        G2), and where (t)."""
        kmax = {}
        out = {}
        for e, uses in self.joins():
            if edges is not None and e not in edges:
                continue
            f0, k0, A0, B0 = uses[0]
            t = self._by_length(nets[f0], *knots[f0], k0, A0, B0, samples, CHECK_ENDS)
            ks = []
            for f, k, A, B in uses:
                if f not in kmax:
                    kmax[f] = self._face_kmax(nets[f], *knots[f])
                ks.append(self.cross_curvatures(nets[f], *knots[f], k, t, A, B))
            (ka, na), (kb, nb) = ks
            kb = np.where((na * nb).sum(1) < 0, -kb, kb)
            u0, v0 = SIDE_UV[k0](A0 + B0 * t)
            L = float(np.linalg.norm(np.diff(derivs(nets[f0], *knots[f0], u0, v0, ((0, 0),))[(0, 0)], axis=0),
                                     axis=1).sum()) or 1.0
            scale = max(np.nanmax(np.abs(np.r_[ka, kb, 0.0])), kmax[uses[0][0]], kmax[uses[1][0]], 0.05 / L)
            rel = np.nan_to_num(np.abs(ka - kb) / scale)                  # (none where a corner folds flat)
            out[e] = (float(rel.max()), float(t[rel.argmax()]))
        return out

    def moved_edges(self, nets, before):
        """The smooth edges worth checking after the joins: those at an
        irregular vertex or with a CV the joins moved within three rows of
        them (all a curvature across an edge depends on). Everywhere else two
        faces share the subdivided cage's rows and join exactly, by
        construction (measured: 1e-6 degrees, 1e-9 in G2)."""
        moved = [np.any(a != b, axis=2) for a, b in zip(nets, before)]
        return {e for e, uses in self.edges.items() if len(uses) == 2 and e not in self.sharp
                and (e[0] in self.irr or e[1] in self.irr or any(NEAR_SIDE[k](moved[f]).any() for f, k, _ in uses))}

    def edge_jumps(self, nets, knots, skip=0.0, edges=None, also_ends=False):
        """Per smooth edge (or those in edges), the largest angle (degrees)
        between its faces' normals, and where (t); skip: leave that much of
        each end out (as Continuity Check does). also_ends: (skipping, and
        right to the corners) from one evaluation."""
        out, out_all = {}, {}
        ends = 2.0 ** -np.arange(2, 16)
        t_all = np.unique(np.r_[np.linspace(0.0, 1.0, 81), ends, 1 - ends, 1.5 * ends, 1 - 1.5 * ends])   # by the corners too
        for e, uses in self.joins():
            if edges is not None and e not in edges:
                continue
            f0, k0, A0, B0 = uses[0]
            t = self._by_length(nets[f0], *knots[f0], k0, A0, B0, 81, skip) if skip else t_all
            n_skip = len(t)
            if also_ends:
                t = np.r_[t, t_all]
            ns, flat = [], np.zeros(len(t), bool)
            for f, k, A, B in uses:
                u, v = SIDE_UV[k](A + B * t)
                D = derivs(nets[f], *knots[f], u, v, ((1, 0), (0, 1)))
                (au, av), (iu, iv) = ALONG[k], INWARD[k]
                T = (-1.0 if B < 0 else 1.0) * (au * D[(1, 0)] + av * D[(0, 1)])
                I = iu * D[(1, 0)] + iv * D[(0, 1)]
                n = np.cross(T, I)
                ln = np.linalg.norm(n, axis=1)
                flat |= ln < 1e-7 * np.linalg.norm(T, axis=1) * np.linalg.norm(I, axis=1)   # a corner folded flat:
                ns.append(n / np.maximum(ln, 1e-300)[:, None])                                # no normal there
            ang = np.degrees(np.arccos(np.clip(np.abs((ns[0] * ns[1]).sum(1)), -1, 1)))
            ang[flat] = 0.0
            a, ta = ang[:n_skip], t[:n_skip]
            out[e] = (float(a.max()), float(ta[a.argmax()]))
            if also_ends:
                a, ta = ang[n_skip:], t[n_skip:]
                out_all[e] = (float(a.max()), float(ta[a.argmax()]))
        return (out, out_all) if also_ends else out


_CORNER_UV = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


class MacroConverter(Converter):
    """A cage with triangles or n-gons, built on its quads after one Catmull-Clark step (R, the same
    surface): one face per cage quad (its four children, 2 x 2 in its (u, v)) and one per child of an n-gon.
    A quad's side next to an n-gon meets two n-gon faces: a T-junction at the side's middle (R's edge point
    there), joined like any edge (subd_g2: the quad's half split at its middle). A lighter layout than
    one face per child of every face: the n-gons alone are split."""

    def __init__(self, V, F, sharp, angle_tol=0.05, curv_tol=0.02):
        V2, F2, s2 = refine_any(np.asarray(V, float), F, sharp)
        F2 = [tuple(int(x) for x in f) for f in F2]
        s2 = with_borders(F2, s2)
        self.R = R = Converter(V2, F2, s2, angle_tol, curv_tol)
        emid, MF, off = {}, [], 0                                  # cage edge -> its edge point in R
        for f in F:
            n = len(f)
            for i in range(n):
                emid[(min(f[i], f[(i + 1) % n]), max(f[i], f[(i + 1) % n]))] = F2[off + i][1]
            if n == 4:
                MF.append((tuple(int(x) for x in f), "q", list(range(off, off + 4))))
            else:
                MF += [(F2[c], "c", [c]) for c in range(off, off + n)]
            off += n
        self.emid = emid
        self.pt_edge = {m: e for e, m in emid.items()}
        self.V, self.F = V2, [fv for fv, _, _ in MF]
        self.kind, self.kids = [k_ for _, k_, _ in MF], [c for _, _, c in MF]
        self.angle_tol, self.curv_tol = angle_tol, curv_tol
        self.size = R.size
        self.sharp = set()
        for fv, kind, _ in MF:
            for i in range(4):
                a, b = fv[i], fv[(i + 1) % 4]
                e = (min(a, b), max(a, b))
                half = (min(a, emid[e]), max(a, emid[e])) if kind == "q" else e
                if half in s2:
                    self.sharp.add(e)
        self.edges, self.vf = {}, [[] for _ in V2]
        for k, fv in enumerate(self.F):
            for i in range(4):
                a, b = fv[i], fv[(i + 1) % 4]
                self.edges.setdefault((min(a, b), max(a, b)), []).append((k, i, a > b))
            for a in fv:
                self.vf[a].append(k)
        corners = {x for fv in self.F for x in fv}
        # the T-junctions: a quad's side used once whose middle is a corner of the n-gon faces beyond
        self.tedges, self.tmid, self.tq, self.tp = [], {}, {}, {}
        for e, uses in self.edges.items():
            f, k, _ = uses[0]
            if len(uses) != 1 or self.kind[f] != "q" or emid[e] not in corners:
                continue
            m = emid[e]
            a, b = self.F[f][k], self.F[f][(k + 1) % 4]
            pieces = []
            for x in (a, b):
                pe = (min(x, m), max(x, m))
                us_ = self.edges.get(pe, [])
                if len(us_) != 1:
                    raise ValueError("a quad's side meets more than two n-gon faces")
                pieces.append((x, us_[0][0], us_[0][1]))
                self.tp[pe] = len(self.tedges)
            self.tq[e] = self.tmid[m] = len(self.tedges)
            self.tedges.append({"e": e, "q": f, "k": k, "a": a, "b": b, "m": m, "pieces": pieces})
        self.open = any(len(u) != 2 and e not in self.tq and e not in self.tp for e, u in self.edges.items())
        self.nbr = [set() for _ in V2]
        for a, b in self.edges:
            self.nbr[a].add(b)
            self.nbr[b].add(a)
        self.val, self.creased, self.star = R.val, R.creased, R.star
        # R's irregular vertices are the faces' corners -- but for the middles of creased or open cage edges along
        # a quad's side (the crease runs straight on through them)
        on_crease = {m for m, (p_, q_) in self.pt_edge.items() if (min(p_, m), max(p_, m)) in s2}
        if not set(R.irr) <= corners | on_crease:
            raise ValueError("an irregular vertex inside a face")
        self.irr = set(R.irr) & corners
        # rings: R's, its edge points standing for the cage vertex beyond them (unless a face corner)
        self.ring = [None] * len(V2)
        for x in corners:
            r = R.ring[x]
            if r is None or x in self.tmid:
                continue
            es = []
            for y in r[0]:
                if y in corners:
                    es.append(y)
                elif y in self.pt_edge and x in self.pt_edge[y]:
                    p_, q_ = self.pt_edge[y]
                    es.append(q_ if p_ == x else p_)
                else:
                    es = None
                    break
            self.ring[x] = (es, r[1]) if es else None
        self._vn, self._arc = {}, {}
        self.deviation = 0.0
        self._nat = {}

    def hard(self, e):
        return e in self.sharp or (len(self.edges[e]) != 2 and e not in self.tq and e not in self.tp)

    def across(self, f, e):
        if e in self.tq:
            return [g for _, g, _ in self.tedges[self.tq[e]]["pieces"]]
        if e in self.tp:
            return [self.tedges[self.tp[e]]["q"]]
        return super().across(f, e)

    def joined_at(self, x):
        out = super().joined_at(x)
        for T in self.tedges:
            if T["e"] not in self.sharp:
                out += [(T["q"], g) for y, g, _ in T["pieces"] if y == x]
        return out

    def joins(self):
        out = super().joins()
        for T in self.tedges:
            if T["e"] in self.sharp:
                continue
            q, kq, m = T["q"], T["k"], T["m"]
            for x, g, kg in T["pieces"]:
                pe = (min(x, m), max(x, m))
                side_c = (0.0, 1.0) if self.F[g][kg] == pe[0] else (1.0, -1.0)
                from_x = self.F[q][kq] == x                        # the quad's side runs from x
                if x == pe[0]:
                    side_q = (0.0, 0.5) if from_x else (1.0, -0.5)
                else:
                    side_q = (0.5, -0.5) if from_x else (0.5, 0.5)
                out.append((pe, [(g, kg) + side_c, (q, kq) + side_q]))
        return out

    def crease_curve(self, a, b):
        e = (min(a, b), max(a, b))
        if e in self.R.edges:
            return self.R.crease_curve(a, b)
        m = self.emid[e]
        c1, c2 = self.R.crease_curve(a, m), self.R.crease_curve(m, b)
        if c1 is None or c2 is None:
            return None

        def cub(t):
            t = np.asarray(t, float)
            out = np.zeros((len(t), 3))
            lo = t <= 0.5
            if lo.any():
                out[lo] = c1(2 * t[lo])
            if (~lo).any():
                out[~lo] = c2(2 * t[~lo] - 1)
            return out
        return cub

    def _quad_eval(self, nets, kn, u, w):
        """The surface of a cage quad's four children (their nets, knots) at its (u, v)."""
        quad = np.where(u > 0.5, np.where(w > 0.5, 2, 1), np.where(w > 0.5, 3, 0))
        P = np.c_[u, w]
        out = np.zeros((len(u), 3))
        for i in range(4):
            sel = quad == i
            if not sel.any():
                continue
            C, du, dv = _CORNER_UV[i], _CORNER_UV[(i + 1) % 4] - _CORNER_UV[i], _CORNER_UV[i - 1] - _CORNER_UV[i]
            d = P[sel] - C
            s, t = np.clip(2 * d @ du, 0, 1), np.clip(2 * d @ dv, 0, 1)
            out[sel] = derivs(nets[i], *kn[i], s, t, ((0, 0),))[(0, 0)]
        return out

    def _fit_quad(self, f, nets, kn, kq, whole=False):
        """Least-squares net on knots kq (both ways) of a quad's surface: its children's (nets, knots), or one
        net over the whole quad (whole); and how far off it stays at the samples."""
        ts = samples(np.union1d(kq, kn[0][0]), 3)
        U, W = np.meshgrid(ts, ts, indexing="ij")
        if whole:
            S = derivs(nets[0], *kn[0], U.ravel(), W.ravel(), ((0, 0),))[(0, 0)].reshape(len(ts), len(ts), 3)
        else:
            S = self._quad_eval(nets, kn, U.ravel(), W.ravel()).reshape(len(ts), len(ts), 3)
        B = basis(kq, ts)
        Bp = np.linalg.pinv(B)
        net = np.einsum("ia,abc,jb->ijc", Bp, S, Bp, optimize=True)
        err = float(np.abs(np.einsum("ia,abc,jb->ijc", B, net, B, optimize=True) - S).max())
        return net, err

    def fresh_natural(self, L):
        """A quad at level L is its children at level L - 1 (one level more where L is 0, then one span
        if that is exact); an n-gon's child its own level L."""
        if L in self._nat:
            return self._nat[L]
        R = self.R
        lq = max(L, 1)
        nc_ = 2 ** (lq - 1)                                        # a child's cells at R's level lq - 1
        nq_ = 2 * nc_
        lev = Levels(R.V, R.F, R.sharp, lq - 1, base=lq - 1)
        Cq, kL = _clamping(nq_)
        cc = None
        if any(k_ == "c" for k_ in self.kind):
            cc = Converter(R.V, R.F, R.sharp)
            nc = cc.natural(L, sliver=False)
        m = len(self.F)
        nets, fk, cor = [None] * m, [None] * m, np.zeros((m, 4, 3))
        a, b = np.meshgrid(np.arange(nc_ + 3), np.arange(nc_ + 3), indexing="ij")
        ghost = (a == 0) | (b == 0) | (a == nc_ + 2) | (b == nc_ + 2)
        for f in range(m):
            ch = self.kids[f]
            if self.kind[f] == "c":
                nets[f], fk[f], cor[f] = nc[ch[0]], cc.fk[ch[0]], cc.levels.corners[ch[0]]
                continue
            # the quad's grid (one ring of ghosts) from its children's, as the cage subdivided would give it;
            # the ghosts first, the children's own points over them
            G = np.zeros((nq_ + 3, nq_ + 3, 3))
            for part in (ghost, ~ghost):
                for i, c in enumerate(ch):
                    C, du, dv = _CORNER_UV[i], _CORNER_UV[(i + 1) % 4] - _CORNER_UV[i], _CORNER_UV[i - 1] - _CORNER_UV[i]
                    p = C + ((a[part] - 1)[:, None] * du + (b[part] - 1)[:, None] * dv) / nq_
                    I, J = 1 + np.rint(p[:, 0] * nq_).astype(int), 1 + np.rint(p[:, 1] * nq_).astype(int)
                    ok = (I >= 0) & (J >= 0) & (I <= nq_ + 2) & (J <= nq_ + 2)
                    G[I[ok], J[ok]] = lev.ext[lq - 1][c][part][ok]
            net = np.einsum("ia,jb,abc->ijc", Cq, Cq, G, optimize=True)
            cs = [lev.corners[c, 0] for c in ch]
            net[0, 0], net[-1, 0], net[-1, -1], net[0, -1] = cs   # the corners on their exact limit points
            kq = kL
            if L == 0:                                             # one span where that is the surface exactly
                k1 = np.r_[[0.0] * 4, [1.0] * 4]
                n1, err = self._fit_quad(f, [net] * 4, [(kq, kq)] * 4, k1, whole=True)
                if err < 1e-9 * self.size:
                    net, kq = n1, k1
                    net[0, 0], net[-1, 0], net[-1, -1], net[0, -1] = cs
            nets[f], fk[f], cor[f] = net, (kq, kq), cs
        self._nat[L] = (nets, fk, cor)
        return self._nat[L]

    def limit_grid(self, level):
        R = self.R
        cq = Converter(R.V, R.F, R.sharp)
        cq.natural(level - 1, sliver=False)
        limq = cq.levels.limit
        limc = None
        out = []
        for f in range(len(self.F)):
            ch = self.kids[f]
            if self.kind[f] == "c":
                if limc is None:
                    c2 = Converter(R.V, R.F, R.sharp)
                    c2.natural(level, sliver=False)
                    limc = c2.levels.limit
                out.append(limc[ch[0]])
                continue
            n = limq.shape[1] - 1
            G = np.zeros((2 * n + 1, 2 * n + 1, 3))
            a, b = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            for i, c in enumerate(ch):
                C, du, dv = _CORNER_UV[i], _CORNER_UV[(i + 1) % 4] - _CORNER_UV[i], _CORNER_UV[i - 1] - _CORNER_UV[i]
                p = C[None, None] + (a[..., None] * du + b[..., None] * dv) / (2 * n)
                I, J = np.rint(p[..., 0] * 2 * n).astype(int), np.rint(p[..., 1] * 2 * n).astype(int)
                G[I, J] = limq[c]
            out.append(G)
        return out


def _length(context, x):
    us = context.scene.unit_settings
    return bpy.utils.units.to_string(us.system, 'LENGTH', x * (us.scale_length or 1.0), precision=3)


def build(V, F, sharp, g2=True, angle_tol=0.05, curv_tol=0.02):
    """(nets, knots, stats, conv) for a cage: faces without a star corner bicubic
    (the SubD exactly), the others 2 x 2 pieces joined exactly, of degree 6 in
    G2, 5 in G1 (subd_g2); the joins measured as Continuity Check would. A cage
    with triangles or n-gons: one face per quad, the n-gons split (MacroConverter,
    T-junctions where they meet quads); if that cannot be solved, every face
    split by one subdivision step."""
    if any(len(f) != 4 for f in F):
        try:
            return _build(MacroConverter(V, F, sharp, angle_tol, curv_tol), g2, macro=True)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            V2, F2, s2 = refine_any(np.asarray(V, float), F, sharp)
            F2 = [tuple(int(x) for x in f) for f in F2]
            nets, knots, st, conv = _build(Converter(V2, F2, with_borders(F2, s2), angle_tol, curv_tol), g2)
            st["layout_fallback"] = str(exc)
            return nets, knots, st, conv
    return _build(Converter(V, F, sharp, angle_tol, curv_tol), g2)


def _build(conv, g2, macro=False):
    import sys
    from . import subd_g2
    info, fallback = {}, None
    if conv.irr:
        try:
            nets, knots, info = subd_g2.build(sys.modules[__name__], conv, g2=g2, kernel=store._k()[1])
        except (RuntimeError, ValueError, np.linalg.LinAlgError, AttributeError) as exc:
            if macro:                                              # the earlier joints know quads only
                raise RuntimeError(str(exc)) from exc
            # the solve failed (or a kernel older than this add-on, without KKTFactor, is loaded): the earlier joints
            fallback = str(exc) if not isinstance(exc, AttributeError) else "restart Blender to load the new kernel"
            nets, knots, _ = conv.run(g2=g2)
            knots = list(knots)
    else:
        nets = conv.natural(0, sliver=False)
        knots = list(conv.fk)
    six = {f for f, (ku, _) in enumerate(knots) if np.count_nonzero(ku == ku[0]) > 4}
    edges = {e for e, uses in conv.joins() if any(u[0] in six for u in uses)}   # the others join as the SubD does: C2
    ends, every = conv.edge_jumps(nets, knots, CHECK_ENDS, edges=edges, also_ends=True)
    st = {"normal": max((j for j, _ in ends.values()), default=0.0),
          "normal_all": max((j for j, _ in every.values()), default=0.0),
          "curvature": max((j for j, _ in conv.curvature_jumps(nets, knots, edges=edges).values()), default=0.0)
          if g2 else None,
          "deviation": subd_g2.deviation(sys.modules[__name__], conv, nets, knots) if six and not fallback else conv.deviation,
          "cvs": sum(n.shape[0] * n.shape[1] for n in nets), "star_points": len(conv.irr), "split": len(six),
          "faces": len(nets), "t_junctions": len(conv.tedges),
          "degree": max((int(np.count_nonzero(knots[f][0] == 0.0)) - 1 for f in six), default=3),
          "fallback": fallback, "layout_fallback": None, "solver": info}
    return nets, knots, st, conv


def convert(context, obj, angle_tol=0.05, g2=True, curv_tol=0.02):
    """(new NURBS object, summary text, note) for a SubD (or any quad) mesh object."""
    V, F, sharp, M = cage(obj, context.evaluated_depsgraph_get())
    if not F:
        raise RuntimeError("the mesh has no faces")
    nets, knots, st, conv = build(V, F, sharp, g2, angle_tol, curv_tol)
    patches = [{"net": [list(map(float, c)) for c in net.reshape(-1, 3)], "ku": list(map(float, ku)),
                "kv": list(map(float, kv)), "degree": int(np.count_nonzero(ku == ku[0])) - 1}
               for net, (ku, kv) in zip(nets, knots)]
    tol = 1e-5 * conv.size
    info, breps = jobs.run("mesh_to_brep", {}, ["result"], xform=None, faces=[], cylinders=[], fillets=[],
                           sweeps=[], patches=patches, tol=tol)
    new = store.new_object_text(context, obj.name + " BREP", breps["result"], Matrix(M))
    for coll in obj.users_collection:
        if new.name not in coll.objects:
            coll.objects.link(new)
    if context.collection and new.name in context.collection.objects and context.collection not in obj.users_collection:
        context.collection.objects.unlink(new)
    for mat in obj.data.materials:
        new.data.materials.append(mat)
    if st["star_points"]:
        scale = abs(np.linalg.det(np.asarray(M)[:3, :3])) ** (1 / 3) or 1.0
        pieces = f"of degree {st['degree']}"
        layout = f"{st['split']} {pieces} by the {st['star_points']} star points, the rest bicubic"
        near = (f"within {_length(context, st['deviation'] * scale)} of the SubD surface "
                f"(exactly on it on the bicubic faces)")
        at_points = f" ({st['normal_all']:.3f}° at the star points)" if st["normal_all"] > st["normal"] + 5e-4 else ""
    else:
        layout, near, at_points = "one bicubic span each", "exactly the SubD surface", ""
    tj = f" ({st['t_junctions']} T-junctions by the n-gons)" if st["t_junctions"] else ""
    summary = (f"{len(nets)} faces{tj} · {layout} · {st['cvs']:,} CVs · G1 within {st['normal']:.3f}°{at_points}"
               + (f" · G2 within {100 * st['curvature']:.2f}%" if g2 else "") + f" · {near}")
    notes = [] if info.get("solid") or conv.open else ["the faces did not close into a solid"]
    if st["fallback"]:
        notes.append(f"the exact solve failed ({st['fallback']}): joined by the earlier method")
    if st["layout_fallback"]:
        notes.append(f"the n-gons could not be joined to the quads ({st['layout_fallback']}): every face split by "
                     f"one subdivision step")
    if st["normal"] > angle_tol + 1e-9:
        notes.append(f"G1 {st['normal']:.3f}°, not {angle_tol:.3f}°")
    if g2 and st["curvature"] > curv_tol + 1e-9:
        notes.append(f"G2 {100 * st['curvature']:.1f}%, not {100 * curv_tol:.1f}%")
    note = "; ".join(notes) or None
    return new, summary, note


class NPROTO_OT_subd_to_brep(bpy.types.Operator):
    """Convert the selected SubD models (Subdivision Surface on a cage) to NURBS bodies:
    one face per cage quad, each bicubic (the SubD exactly) but by star points, where faces
    of degree 6 (G2) or 5 (G1) join exactly; triangles and n-gons split into quads (meeting
    the quads beside them at T-junctions); creases (1.0) as sharp edges. The meshes are kept
    (hidden)"""
    bl_idname = "nproto.subd_to_brep"
    bl_label = "SubD to BREP"
    bl_options = {'REGISTER', 'UNDO'}

    continuity: EnumProperty(
        name="Continuity", default="G2",
        items=[("G1", "G1", "Tangent across every edge, lighter: the faces by star points (degree 5, 10 x 10 CVs) "
                            "are solved so their normals agree exactly; the bending may differ there"),
               ("G2", "G2", "Tangent and bending alike across every edge: the faces by star points (degree 6, "
                            "12 x 12 CVs) are solved so normals and curvature agree exactly")])
    angle: FloatProperty(name="Angle", default=0.05, min=0.005, max=1.0, precision=3, step=1,
                         description="Report a join whose faces' normals differ more than this along a smooth "
                                     "edge (degrees; Continuity Check's default is 0.1)")
    curvature: FloatProperty(name="Curvature %", default=2.0, min=0.1, max=20.0, precision=1,
                             description="G2: report a join whose curvature differs more than this, in percent of "
                                         "the faces' largest curvature (as Continuity Check measures it)")
    hide_original: BoolProperty(name="Hide Meshes", default=True,
                                description="Hide the SubD objects in the viewport and in renders (they stay)")

    def invoke(self, context, event):
        """Ask first: G1 or G2 (and the rest), then convert."""
        return context.window_manager.invoke_props_dialog(self, width=320, title="SubD to BREP",
                                                          confirm_text="Convert")

    def draw(self, context):
        col = self.layout.column()
        col.use_property_split = True
        col.prop(self, "continuity", expand=True)
        col.prop(self, "angle")
        if self.continuity == "G2":
            col.prop(self, "curvature")
        col.prop(self, "hide_original")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(o.type == 'MESH' and not store.is_nurbs(o)
                                                for o in context.selected_objects)

    def execute(self, context):
        srcs = [o for o in context.selected_objects if o.type == 'MESH' and not store.is_nurbs(o)]
        made, lines, warn = [], [], []
        for src in srcs:
            try:
                new, summary, note = convert(context, src, self.angle, g2=self.continuity == "G2",
                                             curv_tol=self.curvature / 100.0)
            except (jobs.JobError, RuntimeError, np.linalg.LinAlgError) as exc:
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
        self.report({'WARNING'} if warn else {'INFO'}, "SubD to BREP: " + "; ".join(lines + warn))
        return {'FINISHED'}


def menu_convert(self, context):
    self.layout.operator(NPROTO_OT_subd_to_brep.bl_idname, text="NURBS Body from SubD (NURBS Proto)", icon='MOD_SUBSURF')


classes = (NPROTO_OT_subd_to_brep,)


def register():
    bpy.types.VIEW3D_MT_object_convert.append(menu_convert)


def unregister():
    bpy.types.VIEW3D_MT_object_convert.remove(menu_convert)
