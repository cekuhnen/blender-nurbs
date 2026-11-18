"""Fit to Scan: a SubD cage whose LIMIT surface lies on a scan (or a set
distance off it), so SubD to BREP turns it into an exact NURBS body that
matches the scan; and the deviation of any surface from the scan.

A Shrinkwrap puts the cage's vertices on the scan, but the Catmull-Clark
surface pulls away from its cage (inwards on bulges, outwards in hollows).
Here the cage is solved instead:

  * every point of the limit surface is a fixed linear mix of cage vertices.
    The mix is built once per cage topology as a sparse matrix A (samples x
    cage vertices): a few Catmull-Clark steps, then the limit masks, with
    the same rules as SubD to BREP (subd_brep.cc_step / cc_limit: creases,
    open borders, Keep Corners), so the fitted surface is the NURBS body.
  * iterate: find where the surface samples are nearest the scan (and the
    scan samples nearest the surface), then one linear least-squares solve
    for the cage:
        data       point-to-plane (+ a little point-to-point) both ways,
                   robust weights; pairs whose normals disagree (the other
                   wall of a thin part) or that lie too far are dropped
        sliding    the cage's tangential spacing stays as it was
        fairness   a light pull on the cage's normal Laplacian (Smoothness)
    solved with Jacobi-preconditioned conjugate gradients on the normal
    equations (numpy only: no scipy in Blender).

Tube Cage from Scan builds a starting cage for limb-like scans (a residual
limb, a forearm, a bottle): rings around the scan's main axis, the closed
end capped with a quad grid, the open end left open.

Distances are in world units; deviation is signed: positive where the
surface stands outside the scan (along the scan's outward normal)."""

import math
import time

import numpy as np

# ---------------------------------------------------------------------------
# sparse matrices (COO), numpy only
# ---------------------------------------------------------------------------


class Sp:
    """A sparse matrix as (rows, cols, vals), duplicates summed."""

    def __init__(self, r, c, v, shape, summed=False):
        r = np.asarray(r, np.int64)
        c = np.asarray(c, np.int64)
        v = np.asarray(v, float)
        if not summed and len(r):
            key = r * shape[1] + c
            uk, inv = np.unique(key, return_inverse=True)
            v = np.bincount(inv.ravel(), weights=v, minlength=len(uk))
            keep = v != 0.0
            uk, v = uk[keep], v[keep]
            r, c = uk // shape[1], uk % shape[1]
        self.r, self.c, self.v, self.shape = r, c, v, tuple(shape)

    def __matmul__(self, X):
        if isinstance(X, Sp):
            return spmm(self, X)
        X = np.asarray(X, float)
        if X.ndim == 1:
            return np.bincount(self.r, weights=self.v * X[self.c], minlength=self.shape[0])
        return np.stack([np.bincount(self.r, weights=self.v * X[self.c, k], minlength=self.shape[0])
                         for k in range(X.shape[1])], 1)

    def rmatvec(self, Y):
        """self.T @ Y"""
        Y = np.asarray(Y, float)
        if Y.ndim == 1:
            return np.bincount(self.c, weights=self.v * Y[self.r], minlength=self.shape[1])
        return np.stack([np.bincount(self.c, weights=self.v * Y[self.r, k], minlength=self.shape[1])
                         for k in range(Y.shape[1])], 1)

    def dense(self):
        D = np.zeros(self.shape)
        np.add.at(D, (self.r, self.c), self.v)
        return D


def spmm(A, B):
    """A @ B for two Sp."""
    o = np.argsort(B.r, kind="stable")
    br, bc, bv = B.r[o], B.c[o], B.v[o]
    ptr = np.zeros(B.shape[0] + 1, np.int64)
    np.cumsum(np.bincount(br, minlength=B.shape[0]), out=ptr[1:])
    cnt = ptr[A.c + 1] - ptr[A.c]
    tot = int(cnt.sum())
    rep = np.repeat(np.arange(len(A.r)), cnt)
    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    bi = ptr[A.c][rep] + off
    return Sp(A.r[rep], bc[bi], A.v[rep] * bv[bi], (A.shape[0], B.shape[1]))


# ---------------------------------------------------------------------------
# Catmull-Clark as matrices (the rules of subd_brep.refine_any / cc_step / cc_limit)
# ---------------------------------------------------------------------------


def with_borders(F, sharp):
    """sharp edges (sorted pairs) and the open borders (edges of one face)."""
    uses = {}
    for f in F:
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            uses[e] = uses.get(e, 0) + 1
    return set(sharp) | {e for e, n in uses.items() if n != 2}


def step_any(nv, F, sharp, pin):
    """One Catmull-Clark step on any polygons, as a matrix: (S, F2 quads
    (m x 4 array), sharp2 keys, pin2). Vertex points keep their indices, then
    the edge points, then the face points (subd_brep.refine_any's order)."""
    r, c, v = [], [], []
    edges = {}
    for k, f in enumerate(F):
        for i in range(len(f)):
            edges.setdefault(tuple(sorted((f[i], f[(i + 1) % len(f)]))), []).append(k)
    ek = list(edges)
    ne = len(ek)
    eid = {e: nv + i for i, e in enumerate(ek)}
    fbase = nv + ne
    for k, f in enumerate(F):                                    # face points
        for a in f:
            r.append(fbase + k)
            c.append(a)
            v.append(1.0 / len(f))
    for i, (a, b) in enumerate(ek):                              # edge points
        fs = edges[(a, b)]
        if (a, b) in sharp or len(fs) != 2:
            r += [nv + i, nv + i]
            c += [a, b]
            v += [0.5, 0.5]
        else:
            r += [nv + i, nv + i]
            c += [a, b]
            v += [0.25, 0.25]
            for k in fs:
                for x in F[k]:
                    r.append(nv + i)
                    c.append(x)
                    v.append(0.25 / len(F[k]))
    vf = [[] for _ in range(nv)]
    ve = [[] for _ in range(nv)]
    for k, f in enumerate(F):
        for a in f:
            vf[a].append(k)
    for a, b in ek:
        ve[a].append(b)
        ve[b].append(a)
    for x in range(nv):                                          # vertex points
        sh = [u for u in ve[x] if tuple(sorted((u, x))) in sharp]
        n = len(ve[x])
        if len(sh) >= 3 or not n or (pin is not None and pin[x]):
            r.append(x)
            c.append(x)
            v.append(1.0)
        elif len(sh) == 2:
            r += [x, x, x]
            c += [sh[0], x, sh[1]]
            v += [0.125, 0.75, 0.125]
        else:                                                    # (Q + 2R + (n - 3) S) / n
            for k in vf[x]:
                for y in F[k]:
                    r.append(x)
                    c.append(y)
                    v.append(1.0 / (len(vf[x]) * len(F[k]) * n))
            for y in ve[x]:                                      # 2 R / n, R = mean of edge midpoints
                r += [x, x]
                c += [x, y]
                v += [1.0 / (n * n), 1.0 / (n * n)]
            r.append(x)
            c.append(x)
            v.append((n - 3.0) / n)
    nv2 = nv + ne + len(F)
    S = Sp(r, c, v, (nv2, nv))
    F2, s2 = [], []
    for k, f in enumerate(F):
        cp = fbase + k
        for i in range(len(f)):
            a, b, d = f[i], f[(i + 1) % len(f)], f[i - 1]
            F2.append((a, eid[tuple(sorted((a, b)))], cp, eid[tuple(sorted((a, d)))]))
    for a, b in sharp:
        e = eid.get((a, b))
        if e is None:
            continue
        for x in (a, b):
            lo, hi = min(x, e), max(x, e)
            s2.append(lo * nv2 + hi)
    pin2 = None if pin is None else np.r_[np.asarray(pin, bool), np.zeros(nv2 - nv, bool)]
    return S, np.array(F2, np.int64), np.sort(np.array(s2, np.int64)), pin2


def step_quads(nv, F, sharp_keys, pin):
    """One Catmull-Clark step on quads as a matrix (subd_brep.cc_step)."""
    m = len(F)
    lo, hi = np.minimum(F, np.roll(F, -1, axis=1)), np.maximum(F, np.roll(F, -1, axis=1))
    uk, inv = np.unique((lo.astype(np.int64) * nv + hi).ravel(), return_inverse=True)
    inv = inv.reshape(m, 4)
    ne = len(uk)
    ea, eb = (uk // nv).astype(np.int64), (uk % nv).astype(np.int64)
    cnt = np.bincount(inv.ravel(), minlength=ne)
    crease = np.isin(uk, sharp_keys)
    hard = crease | (cnt != 2)
    R, C, Vv = [], [], []
    fbase = nv + ne
    R.append(np.repeat(fbase + np.arange(m), 4))                 # face points
    C.append(F.ravel())
    Vv.append(np.full(4 * m, 0.25))
    eh = np.nonzero(hard)[0]                                     # hard edge points
    R += [nv + eh, nv + eh]
    C += [ea[eh], eb[eh]]
    Vv += [np.full(len(eh), 0.5)] * 2
    es = np.nonzero(~hard)[0]                                    # smooth edge points: ends + both face points
    R += [nv + es, nv + es]
    C += [ea[es], eb[es]]
    Vv += [np.full(len(es), 0.25)] * 2
    ki = inv.ravel()
    soft = ~hard[ki]
    rows_e = np.repeat(nv + ki[soft], 4)
    cols_e = np.repeat(F, 4, axis=0)[soft].ravel()
    R.append(rows_e)
    C.append(cols_e)
    Vv.append(np.full(len(rows_e), 0.0625))
    # vertex points
    val = np.bincount(ea, minlength=nv) + np.bincount(eb, minlength=nv)
    nf = np.bincount(F.ravel(), minlength=nv)
    ns = np.bincount(ea[hard], minlength=nv) + np.bincount(eb[hard], minlength=nv)
    still = (ns >= 3) | (val == 0)
    if pin is not None:
        still = still | pin
    cre = (~still) & (ns == 2)
    smooth = (~still) & (~cre)
    sv = np.nonzero(still)[0]
    R.append(sv)
    C.append(sv)
    Vv.append(np.ones(len(sv)))
    cv = np.nonzero(cre)[0]
    R.append(cv)
    C.append(cv)
    Vv.append(np.full(len(cv), 0.75))
    for a_, b_ in ((ea[hard], eb[hard]), (eb[hard], ea[hard])):
        ok = cre[a_]
        R.append(a_[ok])
        C.append(b_[ok])
        Vv.append(np.full(int(ok.sum()), 0.125))
    n = np.maximum(val, 1).astype(float)
    # Q / nf / n: each incident face's 4 corners at 1/4
    fv = F.ravel()
    rq = np.repeat(fv, 4)
    cq = np.repeat(F, 4, axis=0).ravel()
    ok = smooth[rq]
    R.append(rq[ok])
    C.append(cq[ok])
    Vv.append((0.25 / (np.maximum(nf, 1)[rq] * n[rq]))[ok])
    # 2 R / n / n with R = sum of edge midpoints: both ends of every edge at 1 / n^2
    for a_, b_ in ((ea, eb), (eb, ea)):
        ok = smooth[a_]
        w = 1.0 / (n[a_[ok]] ** 2)
        R += [a_[ok], a_[ok]]
        C += [a_[ok], b_[ok]]
        Vv += [w, w]
    sm = np.nonzero(smooth)[0]
    R.append(sm)
    C.append(sm)
    Vv.append((n[sm] - 3.0) / n[sm])
    nv2 = nv + ne + m
    S = Sp(np.concatenate(R), np.concatenate(C), np.concatenate(Vv), (nv2, nv))
    fc = (np.arange(m) + fbase)[:, None].repeat(4, 1)
    F2 = np.stack([F, inv + nv, fc, np.roll(inv, 1, axis=1) + nv], -1).reshape(-1, 4)
    ce = np.nonzero(crease)[0]
    st = np.vstack([np.c_[np.minimum(ea[ce], nv + ce), np.maximum(ea[ce], nv + ce)],
                    np.c_[np.minimum(eb[ce], nv + ce), np.maximum(eb[ce], nv + ce)]]).astype(np.int64)
    s2 = np.sort(st[:, 0] * nv2 + st[:, 1]) if len(st) else np.zeros(0, np.int64)
    pin2 = None if pin is None else np.r_[pin, np.zeros(nv2 - nv, bool)]
    return S, F2, s2, pin2


def limit_matrix(nv, F, sharp_keys, pin):
    """Each vertex's limit point as a matrix (subd_brep.cc_limit)."""
    lo, hi = np.minimum(F, np.roll(F, -1, axis=1)), np.maximum(F, np.roll(F, -1, axis=1))
    uk, inv = np.unique((lo.astype(np.int64) * nv + hi).ravel(), return_inverse=True)
    ea, eb = (uk // nv).astype(np.int64), (uk % nv).astype(np.int64)
    hard = np.isin(uk, sharp_keys) | (np.bincount(inv.ravel(), minlength=len(uk)) != 2)
    val = np.bincount(ea, minlength=nv) + np.bincount(eb, minlength=nv)
    ns = np.bincount(ea[hard], minlength=nv) + np.bincount(eb[hard], minlength=nv)
    still = ns >= 3
    if pin is not None:
        still = still | pin
    cre = (~still) & (ns == 2)
    smooth = (~still) & (~cre)
    n = np.maximum(val, 1).astype(float)
    R, C, Vv = [], [], []
    sv = np.nonzero(still)[0]
    R.append(sv)
    C.append(sv)
    Vv.append(np.ones(len(sv)))
    cv = np.nonzero(cre)[0]
    R.append(cv)
    C.append(cv)
    Vv.append(np.full(len(cv), 4.0 / 6.0))
    for a_, b_ in ((ea[hard], eb[hard]), (eb[hard], ea[hard])):
        ok = cre[a_]
        R.append(a_[ok])
        C.append(b_[ok])
        Vv.append(np.full(int(ok.sum()), 1.0 / 6.0))
    den = n * (n + 5.0)
    sm = np.nonzero(smooth)[0]
    R.append(sm)
    C.append(sm)
    Vv.append(n[sm] * n[sm] / den[sm])
    for a_, b_ in ((ea, eb), (eb, ea)):                          # 4 E
        ok = smooth[a_]
        R.append(a_[ok])
        C.append(b_[ok])
        Vv.append(4.0 / den[a_[ok]])
    fv = F.ravel()                                               # D: the diagonal corner of each incident face
    dg = np.roll(F, -2, axis=1).ravel()
    ok = smooth[fv]
    R.append(fv[ok])
    C.append(dg[ok])
    Vv.append(1.0 / den[fv[ok]])
    return Sp(np.concatenate(R), np.concatenate(C), np.concatenate(Vv), (nv, nv))


class Sampler:
    """The limit surface of a cage as A @ X: the limit points of the cage
    refined `levels` times (quads Fs over those points). Vertex points keep
    their indices, so row v (v < cage vertices) is cage vertex v's limit point."""

    def __init__(self, nv, F, sharp=(), keep_corners=False, levels=3):
        F = [tuple(int(x) for x in f) for f in F]
        sharp = with_borders(F, {tuple(sorted(e)) for e in sharp})
        pin = None
        if keep_corners:
            nf = np.zeros(nv, int)
            for f in F:
                for a in f:
                    nf[a] += 1
            pin = nf == 1
        self.nv, self.F, self.levels = nv, F, levels
        if all(len(f) == 4 for f in F) and levels > 0:
            Fq = np.array(F, np.int64)
            keys = np.sort(np.array([a * nv + b for a, b in sharp], np.int64)) if sharp else np.zeros(0, np.int64)
            S, Fq, keys, pin = step_quads(nv, Fq, keys, pin)
        else:
            S, Fq, keys, pin = step_any(nv, F, sharp, pin)
        n = S.shape[0]
        for _ in range(max(levels, 1) - 1):
            S2, Fq, keys, pin = step_quads(n, Fq, keys, pin)
            S = spmm(S2, S)
            n = S.shape[0]
        Lm = limit_matrix(n, Fq, keys, pin)
        self.A = spmm(Lm, S)
        self.Fs = Fq
        self.border_s = _border_vertices(Fq, n)

    def points(self, X):
        return self.A @ X


def _border_vertices(F, n):
    """Vertices on open borders of quads F."""
    lo, hi = np.minimum(F, np.roll(F, -1, axis=1)), np.maximum(F, np.roll(F, -1, axis=1))
    uk, cnt = np.unique((lo.astype(np.int64) * n + hi).ravel(), return_counts=True)
    b = uk[cnt == 1]
    out = np.zeros(n, bool)
    out[b // n] = True
    out[b % n] = True
    return out


def quad_normals(P, F):
    """Area-weighted vertex normals of a quad mesh (unit)."""
    a, b, c, d = (P[F[:, i]] for i in range(4))
    fn = np.cross(c - a, d - b)
    N = np.zeros_like(P)
    for i in range(4):
        np.add.at(N, F[:, i], fn)
    ln = np.linalg.norm(N, axis=1)
    return N / np.maximum(ln, 1e-30)[:, None]


def umbrella(nv, F):
    """The cage's umbrella Laplacian (x - mean of edge neighbours) as Sp."""
    edges = set()
    for f in F:
        for i in range(len(f)):
            edges.add(tuple(sorted((f[i], f[(i + 1) % len(f)]))))
    e = np.array(sorted(edges), np.int64).reshape(-1, 2)
    deg = np.bincount(e.ravel(), minlength=nv).astype(float)
    r = np.r_[np.arange(nv), e[:, 0], e[:, 1]]
    c = np.r_[np.arange(nv), e[:, 1], e[:, 0]]
    v = np.r_[np.ones(nv), -1.0 / np.maximum(deg[e[:, 0]], 1), -1.0 / np.maximum(deg[e[:, 1]], 1)]
    return Sp(r, c, v, (nv, nv))


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------


class Scan:
    """A triangulated scan in world space with a BVH, outward face normals
    and an even subsample of its vertices (with normals) for the scan ->
    surface term."""

    def __init__(self, V, T, max_samples=12000, seed=0):
        from mathutils.bvhtree import BVHTree
        self.V = np.asarray(V, float)
        self.T = np.asarray(T, np.int64)
        a, b, c = self.V[self.T[:, 0]], self.V[self.T[:, 1]], self.V[self.T[:, 2]]
        fn = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(fn, axis=1)
        cen = (a + b + c) / 3.0
        mid = self.V.mean(0)
        # outward: the normals point away from the middle on the whole (area weighted)
        self.flip = float(((cen - mid) * fn).sum()) < 0.0
        if self.flip:
            fn = -fn
        self.fn = fn / np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]
        vn = np.zeros_like(self.V)
        for i in range(3):
            np.add.at(vn, self.T[:, i], fn)
        self.vn = vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]
        used = np.zeros(len(self.V), bool)
        used[self.T.ravel()] = True
        idx = np.nonzero(used)[0]
        if len(idx) > max_samples:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, max_samples, replace=False))
        self.si = idx
        self.diag = float(np.linalg.norm(self.V.max(0) - self.V.min(0)))
        self.edge = float(np.median(np.linalg.norm(b - a, axis=1))) if len(a) else 0.0
        self.border = _tri_border(self.T, len(self.V))
        self.tree = BVHTree.FromPolygons(self.V.tolist(), self.T.tolist(), all_triangles=True)
        self.area = area

    def nearest(self, P, max_dist=None):
        """(Q nearest points, N their face normals, d distances, ok found)."""
        Q = np.zeros_like(P)
        N = np.zeros_like(P)
        d = np.full(len(P), np.inf)
        idx = np.full(len(P), -1, np.int64)
        tree = self.tree
        md = 1.0e30 if max_dist is None else float(max_dist)
        for i, p in enumerate(P):
            hit = tree.find_nearest(p, md)
            if hit[0] is not None:
                Q[i] = hit[0]
                idx[i] = hit[2]
                d[i] = hit[3]
        ok = idx >= 0
        N[ok] = self.fn[idx[ok]]
        return Q, N, d, ok, idx

    def signed(self, P, max_dist=None):
        """Signed distance of points P from the scan (positive outside), NaN where
        no scan lies within max_dist or the nearest point is on the scan's open
        border (the point is beside the scan, not over it)."""
        Q, N, d, ok, idx = self.nearest(P, max_dist)
        s = np.full(len(P), np.nan)
        D = P - Q
        sd = (D * N).sum(1)
        # beside a hole or past the scan's edge: the offset does not leave along the normal
        along = np.abs(sd) >= 0.5 * d
        good = ok & (along | (d < 1e-9))
        s[good] = np.sign(sd[good]) * d[good]
        return s


def _tri_border_edges(T, nv):
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    uk, cnt = np.unique(e[:, 0].astype(np.int64) * nv + e[:, 1], return_counts=True)
    b = uk[cnt == 1]
    return np.c_[b // nv, b % nv]


def _tri_border(T, nv):
    b = _tri_border_edges(T, nv)
    out = np.zeros(nv, bool)
    out[b[:, 0]] = True
    out[b[:, 1]] = True
    return out


def border_loops(V, T):
    """The scan's open borders as vertex sets, longest first (by edge length)."""
    E = _tri_border_edges(T, len(V))
    adj = {}
    for a, b in E:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen, loops = set(), []
    for s in adj:
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        cs = set(comp)
        length = sum(float(np.linalg.norm(V[a] - V[b])) for a, b in E if int(a) in cs)
        loops.append((length, np.array(sorted(cs), np.int64)))
    loops.sort(key=lambda x: -x[0])
    return loops


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------


def _pcg(apply, b, x0, diag, iters=400, tol=1e-9):
    """Preconditioned conjugate gradients for M x = b (M symmetric positive)."""
    x = x0.copy()
    r = b - apply(x)
    z = r / diag
    p = z.copy()
    rz = float((r * z).sum())
    b2 = float((b * b).sum()) or 1.0
    k = 0
    for k in range(iters):
        if float((r * r).sum()) <= tol * tol * b2:
            break
        Ap = apply(p)
        a = rz / max(float((p * Ap).sum()), 1e-300)
        x += a * p
        r -= a * Ap
        z = r / diag
        rz2 = float((r * z).sum())
        p = z + (rz2 / max(rz, 1e-300)) * p
        rz = rz2
    return x, k


def _bary(P, a, b, c):
    """Barycentric coordinates of P in triangles (a, b, c) (rows)."""
    v0, v1, v2 = b - a, c - a, P - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-300, 1e-300, d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return 1.0 - v - w, v, w


def feet_on_surface(P, Fs, Q):
    """For points Q: their nearest point on the quad mesh (P, Fs) as weights on
    its vertices: (Sp W with Q x len(P), feet, face index, distance)."""
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons(P.tolist(), Fs.tolist())
    n = len(Q)
    foot = np.zeros_like(Q)
    fi = np.full(n, -1, np.int64)
    dist = np.full(n, np.inf)
    for i, q in enumerate(Q):
        hit = tree.find_nearest(q)
        if hit[0] is not None:
            foot[i] = hit[0]
            fi[i] = hit[2]
            dist[i] = hit[3]
    ok = fi >= 0
    rows, cols, vals = [], [], []
    k = np.nonzero(ok)[0]
    F = Fs[fi[k]]
    A_, B_, C_, D_ = (P[F[:, j]] for j in range(4))
    l1 = _bary(foot[k], A_, B_, C_)
    l2 = _bary(foot[k], A_, C_, D_)
    bad1 = np.minimum(np.minimum(l1[0], l1[1]), l1[2])
    bad2 = np.minimum(np.minimum(l2[0], l2[1]), l2[2])
    first = bad1 >= bad2
    for j, (w1, w2) in enumerate(((l1[0], l2[0]), (l1[1], 0.0 * l2[0]), (l1[2], l2[1]), (0.0 * l1[0], l2[2]))):
        w = np.clip(np.where(first, w1, w2), 0.0, 1.0)
        rows.append(k)
        cols.append(F[:, j])
        vals.append(w)
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    s = np.bincount(r, weights=v, minlength=n)
    v = v / np.maximum(s[r], 1e-30)
    return Sp(r, c, v, (n, len(P))), foot, fi, dist


class FitResult(dict):
    pass


DENSE_MAX = 1500          # cage vertices up to which the normal equations are assembled and solved directly


def _gram(R, W3, n):
    """Dense (3n x 3n) sum over the rows j of R of r_j^T W3_j r_j, with r_j
    row j of R (on n vertices) and W3_j a 3 x 3 weight acting on x, y, z.
    Rows of zero weight are skipped."""
    live = np.abs(W3).reshape(len(W3), -1).max(1) > 0
    keep = live[R.r]
    r, c, v = R.r[keep], R.c[keep], R.v[keep]
    o = np.argsort(r, kind="stable")
    r, c, v = r[o], c[o], v[o]
    rows, start, cnt = np.unique(r, return_index=True, return_counts=True)
    c2 = cnt * cnt
    tot = int(c2.sum())
    out = np.zeros((n, 3, n, 3))
    if not tot:
        return out.reshape(3 * n, 3 * n)
    g = np.repeat(np.arange(len(rows)), c2)
    k = np.arange(tot) - np.repeat(np.cumsum(c2) - c2, c2)
    ia = start[g] + k // cnt[g]
    ib = start[g] + k % cnt[g]
    key = c[ia] * n + c[ib]
    vv = v[ia] * v[ib]
    Wg = W3[rows[g]]
    for p in range(3):
        for q in range(p, 3):
            blk = np.bincount(key, weights=vv * Wg[:, p, q], minlength=n * n).reshape(n, n)
            out[:, p, :, q] = blk
            if q != p:
                out[:, q, :, p] = blk.T
    return out.reshape(3 * n, 3 * n)


FAIR_SCALE = 1.0          # Smoothness 1 = this fairness weight per vertex, against the data's total of 1


def regularisers(L, Nv, X, tau, phi, border=None):
    """Scalar rows on the 3n coordinates (x of vertex v at 3v + c), their
    weights and targets:
      sliding   per vertex, the umbrella Laplacian along its 2 tangent
                directions keeps its current value (the cage keeps its
                spacing; it may still move along the normal)
      fairness  per vertex, the normal part of its Laplacian (the cage's
                curvature there) minus its neighbours' mean: curvature
                variation, so a round section stays round instead of shrinking."""
    n = len(Nv)
    h = np.array([1.0, 0.0, 0.0])
    T1 = np.cross(Nv, h)
    bad = np.linalg.norm(T1, axis=1) < 0.3
    T1[bad] = np.cross(Nv[bad], np.array([0.0, 1.0, 0.0]))
    T1 /= np.maximum(np.linalg.norm(T1, axis=1), 1e-30)[:, None]
    T2 = np.cross(Nv, T1)

    def rows_dir(D):          # rows d_v^T L_v on 3n columns
        return Sp(np.repeat(L.r, 3), (3 * L.c[:, None] + np.arange(3)).ravel(),
                  (L.v[:, None] * D[L.r]).ravel(), (n, 3 * n))
    LX = L @ X
    S1, S2 = rows_dir(T1), rows_dir(T2)
    H = rows_dir(Nv)
    K = spmm(L, H)
    if border is not None and border.any():
        # an open border's Laplacian points along the surface, not its curvature: no fairness
        # rows at or next to the border
        near = border.copy()
        near[L.r[border[L.c]]] = True
        keep = ~near[K.r]
        K = Sp(K.r[keep], K.c[keep], K.v[keep], K.shape, summed=True)
    R = Sp(np.r_[S1.r, S2.r + n, K.r + 2 * n], np.r_[S1.c, S2.c, K.c], np.r_[S1.v, S2.v, K.v],
           (3 * n, 3 * n), summed=True)
    w = np.r_[np.full(2 * n, tau), np.full(n, phi)]
    t = np.r_[(T1 * LX).sum(1), (T2 * LX).sum(1), np.zeros(n)]
    return R, w, t


def _gram1(R, w, N):
    """Dense N x N sum over the rows j of R of w_j r_j^T r_j."""
    keep = w[R.r] != 0
    r, c, v = R.r[keep], R.c[keep], R.v[keep]
    o = np.argsort(r, kind="stable")
    r, c, v = r[o], c[o], v[o]
    rows, start, cnt = np.unique(r, return_index=True, return_counts=True)
    c2 = cnt * cnt
    tot = int(c2.sum())
    if not tot:
        return np.zeros((N, N))
    g = np.repeat(np.arange(len(rows)), c2)
    k = np.arange(tot) - np.repeat(np.cumsum(c2) - c2, c2)
    ia = start[g] + k // cnt[g]
    ib = start[g] + k % cnt[g]
    return np.bincount(c[ia] * N + c[ib], weights=v[ia] * v[ib] * w[rows[g]], minlength=N * N).reshape(N, N)


def fit_cage(X0, F, scan, sharp=(), keep_corners=False, iterations=8, offset=0.0, smoothness=0.1,
             stiffness=1.0, both_ways=True, max_dist=0.0, levels=None, normal_angle=60.0, report=None,
             fixed=None, drape=False, drape_slack=0.01):
    """Solve cage vertex positions (world space, n x 3) so the cage's limit
    surface lies on the scan, `offset` off it along its outward normal.
    drape: one-sided, like a sheet laid over the scan: where the surface lies
    inside the scan (+ offset) it is pulled out at full weight, where it lies
    outside only at drape_slack -- it bridges hollows and folds (held by the
    fairness) instead of following them in.
    smoothness: fairness of the surface (0 = follow the scan as closely as the
    cage can). stiffness: how firmly the cage keeps its tangential spacing.
    fixed: boolean mask of cage vertices that stay where they are.
    Returns the new cage and a report (deviation stats, timings)."""
    t0 = time.perf_counter()
    X0 = np.asarray(X0, float)
    n = len(X0)
    if levels is None:
        levels = 3 if len(F) <= 1500 else 2
    smp = Sampler(n, F, sharp, keep_corners, levels)
    A, Fs = smp.A, smp.Fs
    m = A.shape[0]
    L = umbrella(n, F)
    cage_border = smp.border_s[:n]
    t_build = time.perf_counter() - t0
    cosmax = math.cos(math.radians(normal_angle))
    X = X0.copy()
    fixed = np.zeros(n, bool) if fixed is None else np.asarray(fixed, bool)
    # scan samples for the scan -> surface term
    Qs, Ns = scan.V[scan.si], scan.vn[scan.si]
    Qs_border = scan.border[scan.si]
    hist = []
    orient = None
    size = scan.diag or 1.0
    for it in range(int(iterations)):
        P = A @ X
        NP = quad_normals(P, Fs)
        # -- surface -> scan
        Q, NQ, d, ok, _ = scan.nearest(P)
        if orient is None:                       # cage normals against the scan's outward ones
            dots = (NP[ok] * NQ[ok]).sum(1)
            orient = 1.0 if (dots >= 0).sum() >= (dots < 0).sum() else -1.0
        NPo = NP * orient
        agree = (NPo * NQ).sum(1) >= cosmax
        Dv = P - Q
        dn = np.abs((Dv * NQ).sum(1))
        along = dn >= 0.7 * d                    # beside the scan's open edge: the offset runs along it
        use = ok & agree & (along | (d < 1e-3 * size))
        dmax = max_dist if max_dist > 0 else max(4.0 * float(np.median(d[use])) if use.any() else size, 0.01 * size)
        use &= d <= dmax
        T1 = Q + offset * NQ
        # robust weights: Geman-McClure on the distance to the target
        e1 = ((P - T1) * NQ).sum(1)                # + outside the target
        r1 = np.abs(e1)
        # (mild: the worst spots of a clean scan are what the fit must reach; outliers are already
        # dropped by distance and normal)
        sig = max(10.0 * float(np.median(r1[use])) if use.any() else size, 1e-4 * size)
        w1 = np.where(use, 1.0 / (1.0 + (r1 / sig) ** 2), 0.0)
        if drape:
            w1 = w1 * np.where(e1 > 0, drape_slack, 1.0)
        w1 = w1 / max(w1.sum(), 1e-30)
        # -- scan -> surface
        if both_ways:
            W, foot, fi, d2 = feet_on_surface(P, Fs, Qs)
            fn = NPo[Fs[np.clip(fi, 0, None)]].mean(1)
            fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]
            D2 = Qs - foot
            al2 = np.abs((D2 * fn).sum(1)) >= 0.7 * d2
            ok2 = (fi >= 0) & ((fn * Ns).sum(1) >= cosmax) & (al2 | (d2 < 1e-3 * size)) & (d2 <= dmax) & ~Qs_border
            # a scan point nearest the surface's open border is past it
            on_b = smp.border_s[Fs[np.clip(fi, 0, None)]].any(1) & ~al2
            ok2 &= ~on_b
            T2 = Qs + offset * Ns
            e2 = ((foot - T2) * Ns).sum(1)
            r2 = np.abs(e2)
            w2 = np.where(ok2, 1.0 / (1.0 + (r2 / sig) ** 2), 0.0)
            if drape:
                w2 = w2 * np.where(e2 > 0, drape_slack, 1.0)
            w2 = w2 / max(w2.sum(), 1e-30)
            B = spmm(W, A)
        # -- regularisers, as scalar rows on the 3n coordinates
        Nv = NPo[:n]
        alpha, beta = 1.0, 0.05
        R3, w3, t3 = regularisers(L, Nv, X, float(stiffness) / n, FAIR_SCALE * float(smoothness) / n, cage_border)
        mu = 1e-6

        def data(Mx, Nn, w, Y):
            Z = Mx @ Y
            return Mx.rmatvec(w[:, None] * (alpha * Nn * (Nn * Z).sum(1, keepdims=True) + beta * Z))

        def apply(x):
            Y = x.reshape(n, 3)
            out = data(A, NQ, w1, Y) + mu * Y + R3.rmatvec(w3 * (R3 @ x)).reshape(n, 3)
            if both_ways:
                out = out + data(B, Ns, w2, Y)
            out[fixed] = Y[fixed]
            return out.ravel()

        rhs = A.rmatvec(w1[:, None] * (alpha * NQ * (NQ * T1).sum(1, keepdims=True) + beta * T1))
        rhs += R3.rmatvec(w3 * t3).reshape(n, 3) + mu * X
        if both_ways:
            rhs += B.rmatvec(w2[:, None] * (alpha * Ns * (Ns * T2).sum(1, keepdims=True) + beta * T2))
        if n <= DENSE_MAX:
            # the normal equations assembled: sum over rows of r^T W r (W: 3 x 3 per row)
            I3 = np.eye(3)[None]
            M = _gram(A, w1[:, None, None] * (alpha * NQ[:, :, None] * NQ[:, None, :] + beta * I3), n)
            if both_ways:
                M += _gram(B, w2[:, None, None] * (alpha * Ns[:, :, None] * Ns[:, None, :] + beta * I3), n)
            M += _gram1(R3, w3, 3 * n)
            M[np.diag_indices(3 * n)] += mu
            b = rhs.ravel()
            if fixed.any():                      # held vertices: out of the system, exactly where they are
                fx = np.repeat(fixed, 3)
                fr = ~fx
                x = X.ravel().copy()
                x[fr] = np.linalg.solve(M[np.ix_(fr, fr)], b[fr] - M[np.ix_(fr, fx)] @ x[fx])
            else:
                x = np.linalg.solve(M, b)
            cg_it = 0
        else:
            rhs[fixed] = X[fixed]
            dg = np.bincount(A.c, weights=A.v ** 2 * w1[A.r], minlength=n)[:, None] * (alpha * NQ ** 2 + beta).mean(0)
            if both_ways:
                dg += np.bincount(B.c, weights=B.v ** 2 * w2[B.r], minlength=n)[:, None] * (alpha * Ns ** 2 + beta).mean(0)
            dg = dg.ravel() + np.bincount(R3.c, weights=R3.v ** 2 * w3[R3.r], minlength=3 * n) + mu
            dg[np.repeat(fixed, 3)] = 1.0
            x, cg_it = _pcg(apply, rhs.ravel(), X.ravel(), np.maximum(dg, 1e-30))
        Xn = x.reshape(n, 3)
        Xn[fixed] = X[fixed]
        moved = float(np.linalg.norm(Xn - X, axis=1).max())
        X = Xn
        hist.append({"it": it, "used": int(use.sum()), "used_scan": int(ok2.sum()) if both_ways else 0,
                     "rms": float(np.sqrt((w1 * r1 ** 2).sum())), "cg": cg_it, "moved": moved})
        if report:
            report(it, hist[-1])
        if moved < 1e-5 * size:
            break
    t_fit = time.perf_counter() - t0 - t_build
    out = FitResult(X=X, history=hist, samples=m, levels=levels, build_s=t_build, fit_s=t_fit,
                    orient=orient or 1.0, sampler=smp)
    out.update(deviation_stats(A @ X, scan, offset, Fs))
    return out


def vertex_areas(P, T):
    """Each vertex's share of the surface: a third of its triangles' areas (T
    triangles, or quads as two triangles)."""
    T = np.asarray(T)
    if T.shape[1] == 4:
        T = np.vstack([T[:, [0, 1, 2]], T[:, [0, 2, 3]]])
    a = 0.5 * np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), axis=1)
    w = np.zeros(len(P))
    for i in range(3):
        np.add.at(w, T[:, i], a / 3.0)
    return w


def summarize(e, w=None):
    """Area-weighted RMS, mean and 95th percentile of |e|, and the extremes."""
    w = np.ones(len(e)) if w is None else np.asarray(w, float)
    ws = max(float(w.sum()), 1e-300)
    o = np.argsort(np.abs(e))
    cw = np.cumsum(w[o]) / ws
    p95 = float(np.abs(e[o])[min(int(np.searchsorted(cw, 0.95)), len(e) - 1)])
    return {"dev_rms": float(np.sqrt((w * e ** 2).sum() / ws)), "dev_mean": float((w * e).sum() / ws),
            "dev_p95": p95, "dev_min": float(e.min()), "dev_hi": float(e.max()), "dev_max": float(np.abs(e).max())}


def deviation_stats(P, scan, offset=0.0, Fs=None, max_dist=None):
    """Signed deviation of surface points P from the scan (minus the offset)
    over the points that lie over the scan; area weighted when the faces Fs
    (over P) are given."""
    s = scan.signed(P, max_dist)
    ok = np.isfinite(s)
    e = s[ok] - offset
    if not len(e):
        return {"dev_n": 0, "dev_max": float("nan"), "dev_rms": float("nan"), "dev_mean": float("nan"),
                "dev_min": float("nan"), "dev_hi": float("nan"), "dev_p95": float("nan"), "dev_signed": s}
    w = vertex_areas(P, Fs)[ok] if Fs is not None else None
    out = summarize(e, w)
    out.update(dev_n=int(ok.sum()), dev_signed=s)
    return out


# ---------------------------------------------------------------------------
# a starting cage for limb-like scans
# ---------------------------------------------------------------------------


def _frame(axis):
    a = axis / np.linalg.norm(axis)
    h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, h)
    e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


def tube_cage(scan, rings=10, segments=12, top=0.0, margin=0.03, cap=True):
    """A quad cage around a limb-like scan: `rings` rings of `segments`
    vertices (a multiple of 4) around the scan's main axis, from its closed
    end up to `top` (distance from the closed end along the axis; 0: just
    below the lowest point of the scan's open rim, less `margin` of the
    length), the closed end capped by a (segments/4)^2 quad grid. Vertices sit
    on the scan (cast in from outside). Returns (V, F, info)."""
    V = scan.V
    used = np.zeros(len(V), bool)
    used[scan.T.ravel()] = True
    W = V[used]
    wa = np.zeros(len(V))
    for i in range(3):
        np.add.at(wa, scan.T[:, i], scan.area)
    w = wa[used]
    c = (W * w[:, None]).sum(0) / w.sum()
    C = (W - c).T @ ((W - c) * w[:, None]) / w.sum()
    ev, evec = np.linalg.eigh(C)
    axis = evec[:, -1]
    # the rim: the longest open border (small holes elsewhere, e.g. at the tip, don't count)
    loops = border_loops(V, scan.T)
    rim = loops[0][1] if loops and loops[0][0] > 0.1 * scan.diag else None
    if rim is not None:                            # the axis runs from the closed end to the open rim
        if ((V[rim] - c) @ axis).mean() < 0:
            axis = -axis
    elif axis[2] < 0:
        axis = -axis
    a, e1, e2 = _frame(axis)
    t = (W - c) @ a
    t_tip = float(t.min())
    if rim is not None:
        t_rim = float(((V[rim] - c) @ a).min())
    else:
        t_rim = float(t.max())
    span = t_rim - t_tip
    t_top = t_tip + top if top > 0 else t_rim - margin * span
    k = max(1, segments // 4)
    S = 4 * k
    g = 0.75                                         # the cap's first ring this many spacings above the tip
    sp = (t_top - t_tip) / (rings - 1 + g)
    hs = t_tip + (g + np.arange(rings)) * sp
    # the square's border, counter-clockwise from a corner: the ring directions
    sq = []
    s = np.linspace(-1.0, 1.0, k + 1)
    for x in s[:-1]:
        sq.append((x, -1.0))
    for y in s[:-1]:
        sq.append((1.0, y))
    for x in s[::-1][:-1]:
        sq.append((x, 1.0))
    for y in s[::-1][:-1]:
        sq.append((-1.0, y))
    sq = np.array(sq)
    ang = np.arctan2(sq[:, 1], sq[:, 0])
    dirs = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
    R = float(np.linalg.norm(W - c - np.outer(t, a), axis=1).max()) * 1.5 + scan.diag * 0.05
    tree = scan.tree
    ring_pts, centres = [], []
    for h in hs:
        sl = np.abs(t - h) <= 0.5 * sp
        cen = c + h * a
        if sl.sum() >= 8:
            Wc = W[sl]
            cen = Wc.mean(0)
            cen = cen - ((cen - c) @ a - h) * a      # on the ring's plane
        pts = np.full((S, 3), np.nan)
        for j, dvec in enumerate(dirs):
            o = cen + dvec * R
            hit = tree.ray_cast(o, -dvec, R)
            if hit[0] is not None:
                pts[j] = hit[0]
        miss = np.isnan(pts[:, 0])
        if miss.all():
            pts = cen + dirs * (0.5 * R)
        elif miss.any():                             # a hole: the mean radius of the hits
            rr = np.linalg.norm(pts[~miss] - cen, axis=1).mean()
            pts[miss] = cen + dirs[miss] * rr
        ring_pts.append(pts)
        centres.append(cen)
    Vc = [p for ring in ring_pts for p in ring]
    F = []
    for i in range(rings - 1):
        for j in range(S):
            j2 = (j + 1) % S
            F.append((i * S + j, i * S + j2, (i + 1) * S + j2, (i + 1) * S + j))
    if cap:
        # (k + 1)^2 grid: its border is ring 0, its inner points cast up onto the end
        gid = {}
        bi = 0
        for x in range(k):                           # the border in the ring's order
            gid[(x, 0)] = bi
            bi += 1
        for y in range(k):
            gid[(k, y)] = bi
            bi += 1
        for x in range(k, 0, -1):
            gid[(x, k)] = bi
            bi += 1
        for y in range(k, 0, -1):
            gid[(0, y)] = bi
            bi += 1
        cen0 = centres[0]
        r0 = np.linalg.norm(ring_pts[0] - cen0, axis=1).mean()
        below = cen0 - a * (2.0 * (hs[0] - t_tip) + scan.diag * 0.1)
        for y in range(1, k):
            for x in range(1, k):
                u, v = s[x], s[y]
                du, dv = u * math.sqrt(max(0.0, 1 - v * v / 2)), v * math.sqrt(max(0.0, 1 - u * u / 2))
                off = 0.8 * r0 * (du * e1 + dv * e2)
                o = below + off
                hit = tree.ray_cast(o, a, 10 * scan.diag)
                p = np.array(hit[0]) if hit[0] is not None else cen0 + off - a * 0.5 * (hs[0] - t_tip)
                gid[(x, y)] = len(Vc)
                Vc.append(p)
        for y in range(k):
            for x in range(k):
                # wound against the ring's order, as the side faces meet it (consistent normals)
                q = (gid[(x, y)], gid[(x, y + 1)], gid[(x + 1, y + 1)], gid[(x + 1, y)])
                F.append(q)
    Vc = np.array(Vc)
    F = np.array(F, np.int64)
    # outward faces: the side faces' normals point away from the axis
    f0 = F[0]
    fnrm = np.cross(Vc[f0[2]] - Vc[f0[0]], Vc[f0[3]] - Vc[f0[1]])
    mid = Vc[f0].mean(0)
    radial = (mid - c) - ((mid - c) @ a) * a
    if fnrm @ radial < 0:
        F = F[:, ::-1]
    info = {"axis": a, "centre": c, "t_tip": t_tip, "t_rim": t_rim, "t_top": t_top, "spacing": sp}
    return Vc, F, info


# ---------------------------------------------------------------------------
# refining a cage where the fit misses: one edge loop at a time
# ---------------------------------------------------------------------------
#
# A loop cut through an edge ring halves every face along it in one direction
# and keeps the cage all quads (no new star points). Which ring: after a fit,
# what is left of the deviation inside a face is what the cage cannot bend
# to. Its part that varies along u (a per-row linear trend in u taken out)
# needs the face halved in u; the same along v. A ring scores the sum over
# its faces in the direction it halves each one.

_CORNERS = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def face_grid(F, levels):
    """Where each of Sampler(levels)'s quads Fs lies in its cage face: (parent
    face per sub-quad, the sub-quad centre's (u, v) in the face's unit square
    with corner 0 at (0, 0), 1 at (1, 0), 3 at (0, 1)); (u, v) is NaN under
    faces that are not quads. Child i of a quad (a, b, c, d) is (corner i,
    edge point i, face point, edge point i - 1), as step_any / step_quads
    order them."""
    n_f = np.array([len(f) for f in F])
    off = np.r_[0, np.cumsum(n_f)]
    per = 4 ** (max(levels, 1) - 1)
    m = int(off[-1]) * per
    s = np.arange(m)
    c1 = s // per
    parent = np.searchsorted(off, c1, side='right') - 1
    corner = c1 - off[parent]
    quad = n_f[parent] == 4
    k = np.where(quad, corner, 0)
    C = _CORNERS
    o = C[k].copy()
    ax = (C[(k + 1) % 4] - C[k]) / 2
    ay = (C[(k - 1) % 4] - C[k]) / 2
    rest = s % per
    for lv in range(max(levels, 1) - 1):
        d = (rest // 4 ** (levels - 2 - lv)) % 4
        lo, lx, ly = C[d], (C[(d + 1) % 4] - C[d]) / 2, (C[(d - 1) % 4] - C[d]) / 2
        o = o + ax * lo[:, :1] + ay * lo[:, 1:]
        ax, ay = ax * lx[:, :1] + ay * lx[:, 1:], ax * ly[:, :1] + ay * ly[:, 1:]
    uv = o + 0.5 * ax + 0.5 * ay
    uv[~quad] = np.nan
    return parent, uv


def _detrended(r, w, x, groups):
    """Sum of w (r - linear fit in x)^2, the linear fit taken per group (rows of a face)."""
    sw, sx, sxx, sr, sxr, srr = (np.bincount(groups, q) for q in (w, w * x, w * x * x, w * r, w * x * r, w * r * r))
    det = sw * sxx - sx * sx
    ok = det > 1e-30 * np.maximum(sw * sxx, 1e-300)
    b = np.where(ok, (sw * sxr - sx * sr) / np.where(ok, det, 1.0), 0.0)
    a = np.where(sw > 0, (sr - b * sx) / np.maximum(sw, 1e-300), 0.0)
    # sum w (r - a - b x)^2 expanded
    return srr - 2 * a * sr - 2 * b * sxr + a * a * sw + 2 * a * b * sx + b * b * sxx


def face_need(F, levels, P, Fs, resid, tol, faces_ok=None):
    """(need_u, need_v, worst) per cage face: the deviation left in it that a
    cut halving it in u (resp. v) could take up (area x length^2), and its
    largest |deviation|. resid: the deviation at the sample points P (NaN: not
    over the scan). Faces whose worst sample stays within tol (or that are not
    quads, or not in faces_ok) need nothing."""
    nF = len(F)
    parent, uv = face_grid(F, levels)
    r4 = resid[Fs]
    good = np.isfinite(r4).all(1) & np.isfinite(uv).all(1)
    r = np.where(good, np.nan_to_num(r4).mean(1), 0.0)
    a = 0.5 * np.linalg.norm(np.cross(P[Fs[:, 2]] - P[Fs[:, 0]], P[Fs[:, 3]] - P[Fs[:, 1]]), axis=1)
    w = np.where(good, a, 0.0)
    worst = np.zeros(nF)
    np.maximum.at(worst, parent, np.where(good, np.abs(r), 0.0))
    G = 2 ** max(levels, 1)
    u, v = np.nan_to_num(uv[:, 0]), np.nan_to_num(uv[:, 1])
    iu = np.clip((u * G).astype(int), 0, G - 1)
    iv = np.clip((v * G).astype(int), 0, G - 1)
    rows_u = _detrended(r, w, u, parent * G + iv)            # rows: v fixed, varying along u
    rows_v = _detrended(r, w, v, parent * G + iu)            # columns: u fixed, varying along v
    need_u = np.bincount(np.arange(len(rows_u)) // G, rows_u, minlength=nF)[:nF]
    need_v = np.bincount(np.arange(len(rows_v)) // G, rows_v, minlength=nF)[:nF]
    hot = worst > tol
    if faces_ok is not None:
        hot &= faces_ok
    quad = np.array([len(f) == 4 for f in F])
    keep = hot & quad
    return np.where(keep, np.maximum(need_u, 0.0), 0.0), np.where(keep, np.maximum(need_v, 0.0), 0.0), worst


def edge_rings(F):
    """Every edge ring through the quads: a list of (faces, halves) where
    halves[i] is 0 when the ring halves faces[i] in u (it crosses the face's
    edges 0 and 2), 1 in v (edges 1 and 3), and the ring's edges (sorted
    vertex pairs). A ring stops at an open border and at a face that
    is not a quad (that face just gets the new vertex on its edge)."""
    ef = {}
    for k, f in enumerate(F):
        for j in range(len(f)):
            ef.setdefault(tuple(sorted((f[j], f[(j + 1) % len(f)]))), []).append(k)

    def edge(k, j):
        f = F[k]
        return tuple(sorted((f[j % 4], f[(j + 1) % 4])))

    def walk(k, j):
        """From face k leaving across its edge j: [(face, entry edge index)], the last edge, closed?"""
        k0 = k
        out = []
        while True:
            e = edge(k, j)
            nb = [h for h in ef[e] if h != k]
            if len(nb) != 1 or len(F[nb[0]]) != 4:
                return out, e, False
            h = nb[0]
            fh = F[h]
            jh = next(i for i in range(4) if tuple(sorted((fh[i], fh[(i + 1) % 4]))) == e)
            out.append((h, jh))
            k, j = h, jh + 2
            if h == k0 or len(out) > len(F):
                return out, e, False

    seen = set()
    rings = []
    for k0, f in enumerate(F):
        if len(f) != 4:
            continue
        for p in (0, 1):
            if (k0, p) in seen:
                continue
            fwd, _, _ = walk(k0, p + 2)
            closed = any(h == k0 for h, _ in fwd)
            if closed:
                fwd = fwd[:[h for h, _ in fwd].index(k0)]
                back = []
            else:
                back, _, _ = walk(k0, p)
            seq = [(h, jh % 2) for h, jh in reversed(back)] + [(k0, p)] + [(h, jh % 2) for h, jh in fwd]
            faces = [h for h, _ in seq]
            if len(set(faces)) != len(faces):          # crosses itself: leave it out
                for h, q in seq:
                    seen.add((h, q))
                continue
            for h, q in seq:
                seen.add((h, q))
            # the ring's edges: both cut edges of every face
            edges = sorted({edge(h, q + d) for h, q in seq for d in (0, 2)})
            rings.append((faces, [q for _, q in seq], edges))
    return rings


def best_ring(F, need_u, need_v, rings=None):
    """The ring whose faces need cutting most (its score), or (None, 0)."""
    rings = edge_rings(F) if rings is None else rings
    best, top = None, 0.0
    for ring in rings:
        faces, halves, _ = ring
        sc = float(sum(need_u[k] if q == 0 else need_v[k] for k, q in zip(faces, halves)))
        if sc > top:
            best, top = ring, sc
    return best, top


# ===========================================================================
# Blender: the scan cache, the operators and the Scan Deviation overlay
# ===========================================================================

import bpy  # noqa: E402
from bpy.props import BoolProperty, FloatProperty, IntProperty  # noqa: E402

LAST_INFO = {}            # the last fit's report (Adjust Last Operation panel)
_SCANS = {}               # scan object uid -> (key, Scan)
_SAMPLERS = {}            # cage topology key -> Sampler (the overlay of a SubD cage)
DEV_CACHE = {}            # object uid -> (key, (P, N, T, V)) the deviation meshes drawn
DEV_BATCHES = {}          # object uid -> (key, batch)
STATS = {}                # object name -> deviation stats shown in the Analyze panel


def _length(context, x):
    us = context.scene.unit_settings
    return bpy.utils.units.to_string(us.system, 'LENGTH', x * (us.scale_length or 1.0), precision=3)


def _scan_key(obj):
    me = obj.data
    M = tuple(round(x, 9) for row in obj.matrix_world for x in row)
    v = me.vertices
    probe = (tuple(v[0].co), tuple(v[len(v) // 2].co), tuple(v[-1].co)) if len(v) else ()
    return (me.name, len(v), len(me.polygons), M, probe)


def scan_for(obj):
    """The Scan of a mesh object (world space, its own faces, no modifiers), cached."""
    key = _scan_key(obj)
    hit = _SCANS.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    me = obj.data
    me.calc_loop_triangles()
    nv = len(me.vertices)
    V = np.empty(nv * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    T = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    T = T.reshape(-1, 3)
    T = T[~_lid_triangles(me)]
    M = np.array(obj.matrix_world)
    Vw = V @ M[:3, :3].T + M[:3, 3]
    s = Scan(Vw, T)
    _SCANS.clear()                                     # one scan at a time: they can be large
    _SCANS[obj.session_uid] = (key, s)
    return s


def _lid_triangles(me):
    """Triangles of lids: large n-gons (more than 8 sides, over 50 times the typical face) that close a
    cut-off scan -- not scanned surface. Left out, the cut stays an open rim."""
    npoly = len(me.polygons)
    lt = np.empty(npoly, np.int64)
    me.polygons.foreach_get("loop_total", lt)
    tri_poly = np.empty(len(me.loop_triangles), np.int64)
    me.loop_triangles.foreach_get("polygon_index", tri_poly)
    big = lt > 8
    if not big.any():
        return np.zeros(len(tri_poly), bool)
    area = np.empty(npoly)
    me.polygons.foreach_get("area", area)
    lid = big & (area > 50.0 * float(np.median(area)))
    return lid[tri_poly]


def _subsurf(obj):
    subs = [m for m in obj.modifiers if m.type == 'SUBSURF']
    return subs[-1] if subs else None


def is_subd(obj):
    return obj is not None and obj.type == 'MESH' and _subsurf(obj) is not None and 'nproto_brep' not in obj


def cage_of(obj):
    """(V local, F, sharp, keep_corners) of a SubD object's own mesh."""
    from . import subd_brep
    V, F, sharp = subd_brep.live_cage(obj)
    return np.asarray(V, float), F, sharp, subd_brep.keeps_corners(obj)


def pick_scan(context, cage):
    """The scan to fit to: another selected mesh, else the one the cage was made from,
    else the scene's Scan."""
    others = [o for o in context.selected_objects if o is not cage and o.type == 'MESH']
    if len(others) == 1:
        return others[0]
    name = cage.get("nproto_scan")
    if name and name in bpy.data.objects:
        return bpy.data.objects[name]
    return context.scene.nproto_settings.scan_object


def _show_deviation(context, scan_obj):
    context.scene.nproto_settings.scan_object = scan_obj
    v = getattr(context.window_manager, "nproto_view", None)
    if v is not None and not v.show_scan_dev:
        v.show_scan_dev = True


def _fit_kwargs(op):
    return dict(iterations=op.iterations, offset=op.offset, smoothness=op.smoothness, stiffness=op.stiffness,
                both_ways=op.both_ways, max_dist=op.max_distance, drape=op.drape)


class _FitProps:
    iterations: IntProperty(name="Iterations", default=10, min=1, max=60,
                            description="Rounds of matching the surface to the scan and solving the cage")
    offset: FloatProperty(name="Offset", default=0.0, subtype='DISTANCE', unit='LENGTH', soft_min=-0.01,
                          soft_max=0.02, precision=4,
                          description="Fit to the scan moved this far out along its normal (a housing's "
                                      "clearance, a liner's thickness); negative goes in")
    drape: BoolProperty(name="Drape", default=False,
                        description="Lay the surface over the scan like a sheet: it is pushed out wherever it "
                                    "dips into the scan, but bridges folds and hollows instead of following "
                                    "them in (for housings, covers and sockets that must clear the scan)")
    smoothness: FloatProperty(name="Smoothness", default=0.05, min=0.0, soft_max=2.0, max=100.0, precision=3,
                              description="How evenly the surface bends: higher evens out the curvature (scan "
                                          "noise, small bumps) at the cost of a looser fit; 0 follows the scan "
                                          "as closely as the cage can")
    stiffness: FloatProperty(name="Hold Spacing", default=1.0, min=0.0, soft_max=10.0, max=1000.0,
                             description="How firmly the cage keeps its spacing along the surface while it moves "
                                         "to the scan (low: vertices may slide, the fit gets closer but the "
                                         "cage's layout can drift)")
    both_ways: BoolProperty(name="Both Ways", default=True,
                            description="Also pull the surface to every part of the scan (not only the "
                                        "surface to its nearest scan point): bumps the surface would miss "
                                        "are reached")
    max_distance: FloatProperty(name="Max Distance", default=0.0, min=0.0, subtype='DISTANCE', unit='LENGTH',
                                precision=4,
                                description="Scan points farther than this from the surface are ignored "
                                            "(0: automatic, from the typical distance)")


def _report_lines(context, info):
    L = lambda x: _length(context, x)
    out = [f"RMS {L(info['dev_rms'])}  ·  95 % within {L(info['dev_p95'])}",
           f"most inside {L(info['dev_min'])}  ·  most outside {L(info['dev_hi'])}"]
    if info.get("loops") is not None:
        out.append(f"{info['loops']} loops added  ·  {info['verts_before']} -> {info['verts']} cage vertices")
    if info.get("time") is not None:
        out.append(f"{info['verts']} cage vertices  ·  {info['iterations']} rounds  ·  {info['time']:.1f} s")
    return out


def _draw_report(layout, context):
    if not LAST_INFO:
        return
    box = layout.box()
    col = box.column(align=True)
    col.label(text="Distance from the scan" + (" (less the offset)" if LAST_INFO.get("offset") else ""),
              icon='DRIVER_DISTANCE')
    for line in _report_lines(context, LAST_INFO):
        col.label(text=line)


def _fit_object(context, cage, scan_obj, kw, only_selected=False):
    """Fit cage (a SubD mesh object) to scan_obj in place; returns the fit report."""
    scan = scan_for(scan_obj)
    V, F, sharp, kc = cage_of(cage)
    M = np.array(cage.matrix_world)
    Xw = V @ M[:3, :3].T + M[:3, 3]
    fixed = None
    if only_selected:
        sel = np.zeros(len(V), bool)
        cage.data.vertices.foreach_get("select", sel)
        if sel.any():
            fixed = ~sel
    t0 = time.perf_counter()
    r = fit_cage(Xw, F, scan, sharp=sharp, keep_corners=kc, fixed=fixed, **kw)
    Mi = np.linalg.inv(M)
    Xl = r["X"] @ Mi[:3, :3].T + Mi[:3, 3]
    cage.data.vertices.foreach_set("co", Xl.ravel())
    cage.data.update()
    info = {k: r[k] for k in ("dev_rms", "dev_p95", "dev_min", "dev_hi", "dev_max", "dev_mean")}
    info.update(time=time.perf_counter() - t0, verts=len(V), iterations=len(r["history"]), offset=kw.get("offset", 0.0))
    return info


class NPROTO_OT_scan_fit_cage(_FitProps, bpy.types.Operator):
    """Move the active SubD cage so its smooth surface lies on the scan (the other selected mesh, or the scan
    it was made from): the Catmull-Clark surface itself is fitted, not the cage's vertices, so SubD to BREP
    then gives a NURBS body on the scan. In Edit Mode only the selected vertices move"""
    bl_idname = "nproto.scan_fit_cage"
    bl_label = "Fit Cage to Scan"
    bl_options = {'REGISTER', 'UNDO'}

    show_deviation: BoolProperty(name="Show Deviation", default=True,
                                 description="Turn on the Scan Deviation overlay (Analyze) after the fit")

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == 'MESH' and 'nproto_brep' not in o

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "offset")
        lay.prop(self, "drape")
        lay.prop(self, "smoothness")
        lay.prop(self, "stiffness")
        lay.prop(self, "iterations")
        lay.prop(self, "both_ways")
        lay.prop(self, "max_distance")
        lay.prop(self, "show_deviation")
        _draw_report(lay, context)

    def execute(self, context):
        from . import subd_brep
        cage = context.active_object
        scan_obj = pick_scan(context, cage)
        if scan_obj is None or scan_obj is cage or scan_obj.type != 'MESH':
            self.report({'WARNING'}, "Select the scan too (or set it under Analyze > Scan Deviation)")
            return {'CANCELLED'}
        if not subd_brep._plain_cage(cage):
            self.report({'WARNING'}, f"{cage.name}: apply the modifiers above Subdivision (and shape keys) first: "
                                     "the fit moves the mesh itself")
            return {'CANCELLED'}
        was_edit = cage.mode == 'EDIT'
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            info = _fit_object(context, cage, scan_obj, _fit_kwargs(self), only_selected=was_edit)
        except (ValueError, np.linalg.LinAlgError) as exc:
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'}, f"Fit Cage to Scan: {exc}")
            return {'CANCELLED'}
        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')
        cage["nproto_scan"] = scan_obj.name
        LAST_INFO.clear()
        LAST_INFO.update(info)
        if self.show_deviation:
            _show_deviation(context, scan_obj)
        self.report({'INFO'}, "Fit Cage to Scan: " + "  ·  ".join(_report_lines(context, info)[:2]))
        return {'FINISHED'}


class NPROTO_OT_scan_tube_cage(_FitProps, bpy.types.Operator):
    """A SubD cage around the active scan for limb-like shapes (a residual limb, an arm, a bottle): rings of
    quads along the scan's main axis from its closed end up to its open rim, the closed end capped with a
    quad grid, then fitted so its smooth surface lies on the scan"""
    bl_idname = "nproto.scan_tube_cage"
    bl_label = "Tube Cage from Scan"
    bl_options = {'REGISTER', 'UNDO'}

    rings: IntProperty(name="Rings", default=10, min=3, max=100,
                       description="Rings of vertices along the scan (more: closer to its detail, less smooth)")
    segments: IntProperty(name="Around", default=12, min=8, max=64, step=4,
                          description="Vertices around each ring (a multiple of 4, for the cap's grid)")
    length: FloatProperty(name="Length", default=0.0, min=0.0, subtype='DISTANCE', unit='LENGTH', precision=4,
                          description="How far up from the closed end the cage reaches (0: just below the "
                                      "scan's open rim)")
    cap: BoolProperty(name="Close the End", default=True,
                      description="Close the cage's lower end with a quad grid (off: an open tube)")
    fit: BoolProperty(name="Fit", default=True,
                      description="Fit the cage to the scan right away (Fit Cage to Scan's settings below)")

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == 'MESH' and 'nproto_brep' not in o and _subsurf(o) is None

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        col = lay.column(align=True)
        col.prop(self, "rings")
        col.prop(self, "segments")
        lay.prop(self, "length")
        lay.prop(self, "cap")
        lay.prop(self, "fit")
        if self.fit:
            col = lay.column()
            col.prop(self, "offset")
            col.prop(self, "drape")
            col.prop(self, "smoothness")
            col.prop(self, "stiffness")
            col.prop(self, "iterations")
            _draw_report(lay, context)

    def execute(self, context):
        scan_obj = context.active_object
        if len(scan_obj.data.polygons) < 20:
            self.report({'WARNING'}, "The active object should be the scan (a mesh with faces)")
            return {'CANCELLED'}
        scan = scan_for(scan_obj)
        seg = max(8, 4 * round(self.segments / 4))
        V, F, info = tube_cage(scan, rings=self.rings, segments=seg, top=self.length, cap=self.cap)
        me = bpy.data.meshes.new("Scan Cage")
        me.from_pydata([tuple(p) for p in V], [], [tuple(int(i) for i in f) for f in F])
        me.update()
        cage = bpy.data.objects.new("Scan Cage", me)
        for coll in scan_obj.users_collection:
            coll.objects.link(cage)
            break
        else:
            context.collection.objects.link(cage)
        mod = cage.modifiers.new("Subdivision", 'SUBSURF')
        mod.levels = 2
        mod.render_levels = 3
        cage.show_wire = True
        cage["nproto_scan"] = scan_obj.name
        LAST_INFO.clear()
        if self.fit:
            try:
                LAST_INFO.update(_fit_object(context, cage, scan_obj, _fit_kwargs(self)))
            except (ValueError, np.linalg.LinAlgError) as exc:
                self.report({'WARNING'}, f"Tube Cage from Scan: the fit failed ({exc}); the cage is unfitted")
        for o in context.selected_objects:
            o.select_set(False)
        cage.select_set(True)
        context.view_layer.objects.active = cage
        context.scene.nproto_settings.scan_object = scan_obj
        if self.fit and LAST_INFO:
            _show_deviation(context, scan_obj)
            self.report({'INFO'}, f"Tube Cage from Scan: {len(V)} vertices; "
                        + "  ·  ".join(_report_lines(context, LAST_INFO)[:2]))
        else:
            self.report({'INFO'}, f"Tube Cage from Scan: {len(V)} vertices, {len(F)} faces")
        return {'FINISHED'}


def insert_loop(me, F, ring):
    """Cut `ring` (from edge_rings over the faces F) into the mesh me, in
    Object Mode: a vertex at the middle of every ring edge, every ring face
    split between its two. The old vertices keep their indices (the new ones
    come after them); a split crease stays creased on both halves, the new
    loop is smooth. Returns how many vertices were added."""
    import bmesh
    faces, halves, edges = ring
    bm = bmesh.new()
    bm.from_mesh(me)
    vs = list(bm.verts)                   # (splitting edges outdates bm.verts' index table)
    n0 = len(vs)
    lay = bm.edges.layers.float.get("crease_edge")
    mid = {}
    for a, b in edges:
        e = bm.edges.get((vs[a], vs[b]))
        if e is None:
            continue
        cr = e[lay] if lay else 0.0
        ne, nv = bmesh.utils.edge_split(e, e.verts[0], 0.5)
        if lay:
            e[lay] = cr
            ne[lay] = cr
        nv.select = True
        mid[(a, b)] = nv
    for k, q in zip(faces, halves):
        f = F[k]
        v1 = mid.get(tuple(sorted((f[q], f[q + 1]))))
        v2 = mid.get(tuple(sorted((f[q + 2], f[(q + 3) % 4]))))
        if v1 is None or v2 is None:
            continue
        common = set(v1.link_faces) & set(v2.link_faces)
        if len(common) != 1:
            continue
        _, loop = bmesh.utils.face_split(common.pop(), v1, v2)
        if lay:
            loop.edge[lay] = 0.0
    added = len(bm.verts) - n0
    bm.to_mesh(me)
    bm.free()
    me.update()
    return added


class NPROTO_OT_scan_refine_cage(_FitProps, bpy.types.Operator):
    """Add edge loops to the active SubD cage where its surface misses the scan, one at a time with a fit after
    each: every loop runs through the faces whose leftover deviation it can take up best, cutting them across
    the way the deviation varies. The cage stays all quads, with no new star points. In Edit Mode only the
    selected faces are looked at and the unselected vertices stay where they are"""
    bl_idname = "nproto.scan_refine_cage"
    bl_label = "Refine Cage to Scan"
    bl_options = {'REGISTER', 'UNDO'}

    loops: IntProperty(name="Loops", default=4, min=1, max=40,
                       description="Edge loops to add at most (fewer when the surface gets within the "
                                   "tolerance everywhere first)")
    tolerance: FloatProperty(name="Tolerance", default=0.0005, min=0.0, subtype='DISTANCE', unit='LENGTH',
                             precision=4,
                             description="A face whose surface stays this close to the scan gets no loop")
    show_deviation: BoolProperty(name="Show Deviation", default=True,
                                 description="Turn on the Scan Deviation overlay (Analyze) afterwards")

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == 'MESH' and 'nproto_brep' not in o and _subsurf(o) is not None

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "loops")
        lay.prop(self, "tolerance")
        col = lay.column()
        col.prop(self, "offset")
        col.prop(self, "drape")
        col.prop(self, "smoothness")
        col.prop(self, "stiffness")
        col.prop(self, "iterations")
        lay.prop(self, "show_deviation")
        _draw_report(lay, context)

    def execute(self, context):
        from . import subd_brep
        cage = context.active_object
        scan_obj = pick_scan(context, cage)
        if scan_obj is None or scan_obj is cage or scan_obj.type != 'MESH':
            self.report({'WARNING'}, "Select the scan too (or set it under Analyze > Scan Deviation)")
            return {'CANCELLED'}
        if not subd_brep._plain_cage(cage):
            self.report({'WARNING'}, f"{cage.name}: apply the modifiers above Subdivision (and shape keys) first: "
                                     "loops are cut into the mesh itself")
            return {'CANCELLED'}
        was_edit = cage.mode == 'EDIT'
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        me = cage.data
        sel = None
        if was_edit:
            sel = np.zeros(len(me.vertices), bool)
            me.vertices.foreach_get("select", sel)
            if not sel.any():
                sel = None
        scan = scan_for(scan_obj)
        kw = _fit_kwargs(self)
        M = np.array(cage.matrix_world)
        Mi = np.linalg.inv(M)
        n_start = len(me.vertices)
        t0 = time.perf_counter()
        added = 0
        try:
            for rnd in range(self.loops + 1):
                V, F, sharp, kc = cage_of(cage)
                Xw = V @ M[:3, :3].T + M[:3, 3]
                fixed = None if sel is None else ~np.r_[sel, np.ones(len(V) - len(sel), bool)]
                last = rnd == self.loops
                # (a full fit every round: shorter in-between fits pick worse loops, about 35 % faster)
                r = fit_cage(Xw, F, scan, sharp=sharp, keep_corners=kc, fixed=fixed, **kw)
                Xl = r["X"] @ Mi[:3, :3].T + Mi[:3, 3]
                me.vertices.foreach_set("co", Xl.ravel())
                me.update()
                if last:
                    break
                smp = r["sampler"]
                P = smp.A @ r["X"]
                resid = r["dev_signed"] - kw["offset"]
                if kw["drape"]:
                    resid = np.minimum(resid, 0.0)          # standing off (bridging) is what a drape is for
                faces_ok = None
                if sel is not None:
                    s_all = np.r_[sel, np.ones(len(V) - len(sel), bool)]
                    faces_ok = np.array([s_all[list(f)].all() for f in F])
                nu, nv_, _ = face_need(F, r["levels"], P, smp.Fs, resid, self.tolerance, faces_ok)
                ring, _ = best_ring(F, nu, nv_)
                if ring is None:
                    break
                insert_loop(me, F, ring)
                added += 1
        except (ValueError, np.linalg.LinAlgError) as exc:
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'}, f"Refine Cage to Scan: {exc}")
            return {'CANCELLED'}
        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')
        cage["nproto_scan"] = scan_obj.name
        info = {k: r[k] for k in ("dev_rms", "dev_p95", "dev_min", "dev_hi", "dev_max", "dev_mean")}
        info.update(time=time.perf_counter() - t0, verts=len(me.vertices), iterations=len(r["history"]),
                    offset=kw["offset"], loops=added, verts_before=n_start)
        LAST_INFO.clear()
        LAST_INFO.update(info)
        if self.show_deviation:
            _show_deviation(context, scan_obj)
        self.report({'INFO'}, f"Refine Cage to Scan: {added} loop{'s' if added != 1 else ''} added "
                    f"({n_start} -> {len(me.vertices)} vertices)  ·  " + _report_lines(context, info)[0])
        return {'FINISHED'}


# -- the Scan Deviation overlay -------------------------------------------------

DEV_ANGLE, DEV_CHORD = 3.0, 0.001        # the NURBS analysis mesh the deviation is measured on
NO_VALUE = 1.0e30                        # drawn grey: not over the scan


def _subd_surface(obj):
    """The SubD object's limit surface (object space) as SubD to BREP builds it: (P, N, T)."""
    V, F, sharp, kc = cage_of(obj)
    key = (len(V), str(F), str(sorted(sharp)), kc)
    smp = _SAMPLERS.get(key)
    if smp is None:
        lv = 3 if len(F) <= 1500 else 2
        smp = Sampler(len(V), F, sharp, kc, lv)
        if len(_SAMPLERS) > 8:
            _SAMPLERS.clear()
        _SAMPLERS[key] = smp
    P = smp.points(V)
    N = quad_normals(P, smp.Fs)
    Fs = smp.Fs
    T = np.vstack([Fs[:, [0, 1, 2]], Fs[:, [0, 2, 3]]])
    return P, N, T


def _dev_key(obj, scan_obj):
    from . import store, subd_brep
    M = tuple(round(x, 9) for row in obj.matrix_world for x in row)
    if store.is_nurbs(obj):
        own = ("nurbs", obj.get(store.PROP_REV, 0))
    else:
        own = ("subd", subd_brep.cage_key(obj))
    return (own, M, _scan_key(scan_obj))


def deviation_mesh(obj, scan_obj):
    """(P, N, T, V): obj's surface (object space) with the signed distance from the
    scan at every vertex (NO_VALUE where it is not over the scan), cached."""
    from . import diagnostics, store
    key = _dev_key(obj, scan_obj)
    hit = DEV_CACHE.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    if store.is_nurbs(obj):
        P, N, T = diagnostics.zebra_mesh(obj, DEV_ANGLE, DEV_CHORD)
    else:
        P, N, T = _subd_surface(obj)
    P = np.asarray(P, float)
    M = np.array(obj.matrix_world)
    Pw = P @ M[:3, :3].T + M[:3, 3]
    s = scan_for(scan_obj).signed(Pw)
    ok = np.isfinite(s)
    V = np.where(ok, s, NO_VALUE).astype(np.float32)
    C = np.ascontiguousarray                  # the GPU buffers take C-ordered arrays only
    data = (C(P, np.float32), C(N, np.float32), C(T, np.int32), C(V))
    DEV_CACHE[obj.session_uid] = (key, data)
    STATS[obj.name] = summarize(s[ok], vertex_areas(Pw, np.asarray(T))[ok]) if ok.any() else {}
    return data


def deviation_targets(context, v, scan_obj):
    from . import diagnostics, store
    return [o for o in diagnostics.analysis_objects(context, v)
            if o is not scan_obj and o.type == 'MESH' and o.visible_get() and (store.is_nurbs(o) or is_subd(o))]


def draw_scan_dev(ctx):
    v = getattr(ctx.window_manager, "nproto_view", None)
    if v is None or not v.show_scan_dev:
        return
    scan_obj = ctx.scene.nproto_settings.scan_object
    if scan_obj is None or scan_obj.type != 'MESH':
        return
    from . import diagnostics
    rng = max(float(v.scan_dev_range), 1e-9)
    c = float(v.scan_dev_center)
    for obj in deviation_targets(ctx, v, scan_obj):
        try:
            P, N, T, V = deviation_mesh(obj, scan_obj)
        except Exception as exc:
            print(f"NURBS scan deviation ({obj.name}): {exc}")
            continue
        diagnostics.draw_value_mesh(ctx, obj, DEV_CACHE[obj.session_uid][0], P, N, T, V, c - rng, c + rng,
                                    bands=float(v.scan_dev_bands), opacity=float(v.zebra_opacity),
                                    cache=DEV_BATCHES)


def draw_panel(col, context):
    """The Scan Deviation settings under Analyze."""
    v = context.window_manager.nproto_view
    s = context.scene.nproto_settings
    col.prop(s, "scan_object", text="Scan")
    row = col.row(align=True)
    row.prop(v, "scan_dev_range", text="±")
    row.prop(v, "scan_dev_center", text="At")
    col.prop(v, "scan_dev_bands")
    L = lambda x: _length(context, x)
    col.label(text=f"red: {L(v.scan_dev_center + v.scan_dev_range)} out  ·  blue: "
                   f"{L(v.scan_dev_center - v.scan_dev_range)}  ·  grey: off the scan", icon='COLOR')
    o = context.active_object
    st = STATS.get(o.name) if o is not None else None
    if st:
        col.label(text=f"{o.name}: RMS {L(st['dev_rms'])}  ·  95 % {L(st['dev_p95'])}")
        col.label(text=f"inside {L(st['dev_min'])}  ·  outside {L(st['dev_hi'])}")


classes = (NPROTO_OT_scan_fit_cage, NPROTO_OT_scan_tube_cage, NPROTO_OT_scan_refine_cage)
