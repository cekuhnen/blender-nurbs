"""Textures on B-rep faces: materials and UVs on a NURBS object's own mesh.

A NURBS object's mesh is written afresh at every rebuild (store.write_mesh)
from the kernel's tessellation, which knows nothing of materials or UVs. An
object with the "nproto_tex" property gets them back at every rebuild:

  * every triangle is labelled with its B-rep face (the face attribute
    "brep_face"). The kernel meshes one face after the other, so its
    triangle list is one run per face: a binary search with a few closest-
    point queries finds where each run ends (crack fills appended after the
    runs take a neighbour's face);
  * every triangle takes the material slot of its face: Plywood sorts the
    faces into front veneer, back veneer and edge (slots 0, 1, 2); faces given
    a slot by Assign Material keep it -- each is stored with a point on it, so
    it is found again however a rebuild renumbers the faces;
  * UVs ("UVMap") in metres: a conformal unwrap (least squares, the kernel's
    sparse solver) of every island -- with Plywood each veneer side is one
    island and the edge band runs along the rim (plywood.make_uvs), otherwise
    each face is its own island -- laid side by side, the long way along u.

Nothing is done for objects without the property: their meshes stay as they
were (and as fast)."""

import json
import math

import bpy
import numpy as np
from bpy.props import EnumProperty

from . import store

PROP = "nproto_tex"
ATTR = "brep_face"
UV_NAME = "UVMap"
_cache = {}      # session_uid -> (rev, swap, per-face classes, thickness): Plywood's sort of the faces
LAST = {}        # object name -> the error of its last texture pass (None: it went through)


# -- the stored setup -------------------------------------------------------
def get(obj):
    try:
        d = json.loads(obj.get(PROP) or "null")
    except (ValueError, TypeError):
        return None
    return d if isinstance(d, dict) else None


def put(obj, d):
    if d:
        obj[PROP] = json.dumps(d)
    elif PROP in obj:
        del obj[PROP]


def is_textured(obj):
    return obj is not None and PROP in obj and store.is_nurbs(obj)


def thickness(obj):
    hit = _cache.get(obj.session_uid)
    return hit[3] if hit else None


def plywood_classes(obj):
    """Per B-rep face FRONT / BACK / EDGE from the last pass (None: not Plywood)."""
    hit = _cache.get(obj.session_uid)
    return hit[2] if hit else None


# -- which face each triangle lies on ---------------------------------------
def _size(brep):
    try:
        lo, hi = (np.array(v, float) for v in brep.bounding_box())
        return float(np.linalg.norm(hi - lo)) or 1.0
    except Exception:
        return 1.0


class _Faces:
    def __init__(self, brep):
        self.faces = brep.faces()
        self.size = _size(brep)
        pad = 1e-4 * self.size
        self.lo = np.array([np.array(f.bounding_box()[0], float) - pad for f in self.faces])
        self.hi = np.array([np.array(f.bounding_box()[1], float) + pad for f in self.faces])

    def nearest(self, c):
        """(face index, distance) of the face nearest the point c (local)."""
        inside = np.flatnonzero(((c >= self.lo) & (c <= self.hi)).all(axis=1))
        cand = inside if len(inside) else range(len(self.faces))
        best, bd = -1, math.inf
        q = tuple(float(x) for x in c)
        for k in cand:
            try:
                d = self.faces[k].closest_point(q)[1]
            except Exception:
                continue
            if d < bd:
                best, bd = int(k), d
        return best, bd


def face_labels(brep, P, polys, n_raw=None):
    """B-rep face index of every polygon of the tessellation (the kernel's
    triangles first, n_raw of them, then any crack fills)."""
    n = len(polys)
    if brep.face_count() <= 1 or n == 0:
        return np.zeros(n, np.int64)
    P = np.asarray(P, float)
    C = np.array([P[list(f)].mean(axis=0) for f in polys])
    F = _Faces(brep)
    L = np.full(n, -1, np.int64)
    memo = {}

    def lab(i):
        if i not in memo:
            memo[i] = F.nearest(C[i])[0]
        return memo[i]

    m = n if not n_raw else min(int(n_raw), n)
    stack = [(0, m - 1)] if m else []
    while stack:                    # one run per face: equal ends -> all of it
        lo, hi = stack.pop()
        a, b = lab(lo), lab(hi)
        if a == b:
            L[lo:hi + 1] = a
        elif hi - lo <= 1:
            L[lo], L[hi] = a, b
        else:
            mid = (lo + hi) // 2
            stack += [(lo, mid), (mid, hi)]
    # a face met twice would leave another without triangles: check each face
    # that got none, and label everything one by one if one is found inside a run
    have = set(np.unique(L[:m]).tolist())
    missing = [k for k in range(len(F.faces)) if k not in have]
    if missing and m > 2:
        probe = np.linspace(0, m - 1, min(m, 8 * len(F.faces))).astype(int)
        if any(lab(int(i)) != L[i] for i in probe):
            for i in range(m):
                L[i] = lab(i)
    # crack fills: a neighbour's face (a polygon sharing a vertex), else the nearest
    if m < n:
        owner = {}
        for i in range(m):
            for v in polys[i]:
                owner.setdefault(int(v), int(L[i]))
        for i in range(m, n):
            k = next((owner[int(v)] for v in polys[i] if int(v) in owner), None)
            L[i] = k if k is not None else lab(i)
    return L


# -- conformal unwrap ---------------------------------------------------------
def _ext():
    return store._k()[1]


def _arap(Un, tri, EX, EY, a2, root, hang, iters):
    """As-rigid-as-possible iterations (local / global, one factorisation):
    Un per node, the triangles' opposite edges EX, EY in their own planes."""
    if iters <= 0 or not len(tri):
        return Un
    nn = len(Un)
    A = 0.5 * a2
    G0 = -EY / a2[:, None]                        # d/dx and d/dy of the corner values
    G1 = EX / a2[:, None]
    W = A[:, None, None] * (G0[:, :, None] * G0[:, None, :] + G1[:, :, None] * G1[:, None, :])
    m = len(tri)
    mr = [np.broadcast_to(tri[:, :, None], (m, 3, 3)).ravel()]
    mc = [np.broadcast_to(tri[:, None, :], (m, 3, 3)).ravel()]
    mv = [W.ravel()]
    if len(hang):
        h, a, b, t = hang[:, 0].astype(np.int64), hang[:, 1].astype(np.int64), hang[:, 2].astype(np.int64), hang[:, 3]
        cs = np.stack([h, a, b], axis=1)
        cf = np.stack([np.ones(len(h)), t - 1.0, -t], axis=1) * 10.0
        mr.append(np.broadcast_to(cs[:, :, None], (len(h), 3, 3)).ravel())
        mc.append(np.broadcast_to(cs[:, None, :], (len(h), 3, 3)).ravel())
        mv.append((cf[:, :, None] * cf[:, None, :]).ravel())
    diag = np.zeros(nn)
    np.add.at(diag, tri.ravel(), np.einsum("tii->ti", W).ravel())
    eps = 1e-9 * (float(diag[diag > 0].mean()) if (diag > 0).any() else 1.0)
    mr.append(np.arange(nn))
    mc.append(np.arange(nn))
    mv.append(np.full(nn, eps))
    pins = np.array(sorted({int(np.flatnonzero(root == r)[0]) for r in np.unique(root[tri[:, 0]])}), np.int64)
    kf = _ext().KKTFactor(np.concatenate(mr).astype(np.int64), np.concatenate(mc).astype(np.int64),
                          np.concatenate(mv).astype(float), nn, np.arange(len(pins), dtype=np.int64), pins,
                          np.ones(len(pins)), len(pins))
    U = Un.copy()
    for _ in range(iters):
        uu, vv = U[tri, 0], U[tri, 1]
        ux, uy = (G0 * uu).sum(1), (G1 * uu).sum(1)
        vx, vy = (G0 * vv).sum(1), (G1 * vv).sum(1)
        th = np.arctan2(vx - uy, ux + vy)
        c, s = np.cos(th), np.sin(th)
        ru = np.zeros(nn)
        rv = np.zeros(nn)
        np.add.at(ru, tri.ravel(), (A[:, None] * (G0 * c[:, None] - G1 * s[:, None])).ravel())
        np.add.at(rv, tri.ravel(), (A[:, None] * (G0 * s[:, None] + G1 * c[:, None])).ravel())
        U = np.stack([np.asarray(kf.solve(ru, U[pins, 0]), float), np.asarray(kf.solve(rv, U[pins, 1]), float)],
                     axis=1)
    return U


def unwrap(P, lv, ls, lt, island, iters=40):
    """UVs per corner (loop) in metres: each island (a key per polygon, -1 =
    left at 0) unwrapped conformally on its own; each connected piece scaled to
    its true area and turned so its long axis runs along u, its lower left
    corner at (0, 0). Also the piece of every corner (-1: none)."""
    P = np.asarray(P, float)
    npoly, nl = len(ls), len(lv)
    lp = np.repeat(np.arange(npoly), lt)
    UV = np.zeros((nl, 2))
    piece = np.full(nl, -1, np.int64)
    use = island[lp] >= 0
    if not use.any():
        return UV, piece
    key = island[lp].astype(np.int64) * (len(P) + 1) + lv
    uk, inv = np.unique(key[use], return_inverse=True)
    node = np.full(nl, -1, np.int64)
    node[use] = inv
    nn = len(uk)
    Pn = np.zeros((nn, 3))
    Pn[node[use]] = P[lv[use]]
    # polygons as fans of triangles (corner indices)
    sel = np.flatnonzero((island >= 0) & (lt >= 3))
    cnt = lt[sel] - 2
    owner = np.repeat(sel, cnt)
    k = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt) + 1
    c0 = ls[owner]
    tri = np.stack([node[c0], node[c0 + k], node[c0 + k + 1]], axis=1)
    p0, p1, p2 = Pn[tri[:, 0]], Pn[tri[:, 1]], Pn[tri[:, 2]]
    e1 = p1 - p0
    nrm = np.cross(e1, p2 - p0)
    a2 = np.linalg.norm(nrm, axis=1)                  # twice the area
    l1 = np.linalg.norm(e1, axis=1)
    size = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0
    ok = (a2 > 1e-14 * size * size) & (l1 > 0)
    tri, p0, p2, e1, nrm, a2, l1 = tri[ok], p0[ok], p2[ok], e1[ok], nrm[ok], a2[ok], l1[ok]
    ex = e1 / l1[:, None]
    ey = np.cross(nrm / a2[:, None], ex)
    d2 = p2 - p0
    X = np.stack([np.zeros(len(tri)), l1, np.einsum("ij,ij->i", d2, ex)], axis=1)
    Y = np.stack([np.zeros(len(tri)), np.zeros(len(tri)), np.einsum("ij,ij->i", d2, ey)], axis=1)
    # the edge opposite each corner: from the next corner to the one after
    EX = np.roll(X, -2, axis=1) - np.roll(X, -1, axis=1)
    EY = np.roll(Y, -2, axis=1) - np.roll(Y, -1, axis=1)
    s = (1.0 / np.sqrt(2.0 * a2))[:, None]           # 1 / (2 sqrt(area))
    # residuals u_x - v_y and u_y + v_x of each triangle (unknowns: all u, then all v)
    cols = np.concatenate([tri, tri + nn], axis=1)
    R1 = np.concatenate([-EY, -EX], axis=1) * s
    R2 = np.concatenate([EX, -EY], axis=1) * s
    Rc = np.vstack([R1, R2])
    Cc = np.vstack([cols, cols])
    mr = np.broadcast_to(Cc[:, :, None], (len(Cc), 6, 6)).ravel()
    mc = np.broadcast_to(Cc[:, None, :], (len(Cc), 6, 6)).ravel()
    mv = (Rc[:, :, None] * Rc[:, None, :]).ravel()
    diag = np.zeros(2 * nn)
    np.add.at(diag, Cc.ravel(), (Rc * Rc).ravel())
    eps = 1e-9 * (float(diag[diag > 0].mean()) if (diag > 0).any() else 1.0)
    mr = np.concatenate([mr, np.arange(2 * nn)])
    mc = np.concatenate([mc, np.arange(2 * nn)])
    mv = np.concatenate([mv, np.full(2 * nn, eps)])
    # T-junctions: the kernel meshes neighbouring faces apart, so one face's edge can have
    # points the other's lacks. Unstitched, the pieces hang together at a few points and
    # unwrap at different scales: each such point is held on the other side's edge.
    hang = _hanging(Pn, tri, size)
    if len(hang):
        w = 10.0
        h, a, b, t = hang[:, 0].astype(np.int64), hang[:, 1].astype(np.int64), hang[:, 2].astype(np.int64), hang[:, 3]
        for off in (0, nn):
            cs = np.stack([h + off, a + off, b + off], axis=1)
            cf = np.stack([np.ones(len(h)), t - 1.0, -t], axis=1) * w
            mr = np.concatenate([mr, np.broadcast_to(cs[:, :, None], (len(h), 3, 3)).ravel()])
            mc = np.concatenate([mc, np.broadcast_to(cs[:, None, :], (len(h), 3, 3)).ravel()])
            mv = np.concatenate([mv, (cf[:, :, None] * cf[:, None, :]).ravel()])
    # connected pieces (shared nodes), two pins each
    parent = np.arange(nn)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    links = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]]] + ([hang[:, :2].astype(np.int64)] if len(hang) else []))
    for a, b in links.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    root = np.array([find(i) for i in range(nn)])
    in_tri = np.zeros(nn, bool)
    in_tri[tri.ravel()] = True
    cr, cc, ry = [], [], []
    for r in np.unique(root[in_tri]):
        ids = np.flatnonzero((root == r) & in_tri)
        cen = Pn[ids].mean(axis=0)
        a = ids[int(np.argmax(np.linalg.norm(Pn[ids] - cen, axis=1)))]
        b = ids[int(np.argmax(np.linalg.norm(Pn[ids] - Pn[a], axis=1)))]
        d = float(np.linalg.norm(Pn[b] - Pn[a])) or 1.0
        for var, val in ((a, 0.0), (a + nn, 0.0), (b, d), (b + nn, 0.0)):
            cr.append(len(cr))
            cc.append(var)
            ry.append(val)
    x = np.asarray(_ext().sparse_kkt(mr.astype(np.int64), mc.astype(np.int64), mv.astype(float), 2 * nn,
                                     np.array(cr, np.int64), np.array(cc, np.int64), np.ones(len(cr)), len(cr),
                                     np.zeros(2 * nn), np.array(ry, float)), float)
    Un = np.stack([x[:nn], x[nn:]], axis=1)
    t_root = root[tri[:, 0]]
    a3 = 0.5 * a2
    # the conformal map keeps angles, not areas: on a doubly curved sheet its scale drifts
    # from face to face (20 times on a chair seat). From it, at true size, as rigid as
    # possible: each triangle's nearest rotation, then the positions that follow them best
    for r in np.unique(t_root):
        ts = t_root == r
        q0, q1, q2 = Un[tri[ts, 0]], Un[tri[ts, 1]], Un[tri[ts, 2]]
        auv = 0.5 * np.abs((q1[:, 0] - q0[:, 0]) * (q2[:, 1] - q0[:, 1]) - (q2[:, 0] - q0[:, 0]) * (q1[:, 1] - q0[:, 1]))
        if auv.sum() > 1e-30:
            ids = np.flatnonzero(root == r)
            Un[ids] *= math.sqrt(float(a3[ts].sum()) / float(auv.sum()))
    Un = _arap(Un, tri, EX, EY, a2, root, hang, iters)
    # each piece: true size, long axis along u, from (0, 0)
    q0, q1, q2 = Un[tri[:, 0]], Un[tri[:, 1]], Un[tri[:, 2]]
    auv = 0.5 * ((q1[:, 0] - q0[:, 0]) * (q2[:, 1] - q0[:, 1]) - (q2[:, 0] - q0[:, 0]) * (q1[:, 1] - q0[:, 1]))
    for r in np.unique(t_root):
        ts = t_root == r
        ids = np.flatnonzero(root == r)
        A3, Auv = float(a3[ts].sum()), float(auv[ts].sum())
        if abs(Auv) < 1e-30:
            continue
        S = Un[ids] * math.sqrt(A3 / abs(Auv))
        if Auv < 0:                                   # (a conformal map keeps its turn; to be safe)
            S[:, 1] *= -1.0
        w = a3[ts]
        cen = (S[np.searchsorted(ids, tri[ts, 0])] + S[np.searchsorted(ids, tri[ts, 1])]
               + S[np.searchsorted(ids, tri[ts, 2])]) / 3.0
        m = (w[:, None] * cen).sum(axis=0) / w.sum()
        Xc = cen - m
        cov = (w[:, None, None] * Xc[:, :, None] * Xc[:, None, :]).sum(axis=0)
        ev, evec = np.linalg.eigh(cov)
        ax = evec[:, int(np.argmax(ev))]
        ang = -math.atan2(ax[1], ax[0])
        Rm = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        S = (S - m) @ Rm.T
        Un[ids] = S - S.min(axis=0)
    UV[use] = Un[node[use]]
    piece[use] = root[node[use]]
    return UV, piece


def _hanging(Pn, tri, size, batch=256):
    """(node, edge start, edge end, t) for every point on an open edge that lies
    on another open edge, inside it (t in (0, 1)) or on its end (an unwelded
    twin): the cracks of separately meshed faces."""
    E = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.minimum(E[:, 0], E[:, 1]) * (len(Pn) + 1) + np.maximum(E[:, 0], E[:, 1])
    uk, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    ob = E[idx[cnt == 1]]
    if not len(ob):
        return np.zeros((0, 4))
    nodes = np.unique(ob.ravel())
    A, B = Pn[ob[:, 0]], Pn[ob[:, 1]]
    D = B - A
    dd = np.maximum(np.einsum("ij,ij->i", D, D), 1e-300)
    seg = np.sqrt(dd)
    lo = np.minimum(A, B)
    hi = np.maximum(A, B)
    out = []
    for s in range(0, len(nodes), batch):
        q = nodes[s:s + batch]
        Q = Pn[q]
        tol = 2e-3 * size
        near = ((Q[:, None, :] >= lo[None] - tol) & (Q[:, None, :] <= hi[None] + tol)).all(axis=2)
        near &= (ob[None, :, 0] != q[:, None]) & (ob[None, :, 1] != q[:, None])
        qi, ei = np.nonzero(near)
        if not len(qi):
            continue
        t = np.einsum("ij,ij->i", Q[qi] - A[ei], D[ei]) / dd[ei]
        tc = np.clip(t, 0.0, 1.0)
        dist = np.linalg.norm(Q[qi] - (A[ei] + tc[:, None] * D[ei]), axis=1)
        ok = (t > -1e-6) & (t < 1 + 1e-6) & (dist < np.minimum(0.05 * seg[ei], tol))
        if not ok.any():
            continue
        qi, ei, tc, dist = qi[ok], ei[ok], tc[ok], dist[ok]
        best = {}
        for a_, e_, t_, d_ in zip(qi.tolist(), ei.tolist(), tc.tolist(), dist.tolist()):
            if a_ not in best or d_ < best[a_][2]:
                best[a_] = (e_, t_, d_)
        for a_, (e_, t_, _) in best.items():
            out.append((q[a_], ob[e_, 0], ob[e_, 1], t_))
    return np.array(out, float).reshape(-1, 4)


def turn(UV, rows, deg):
    """Turn the corners `rows` of UV by deg about their box's lower left, then
    put them back at (0, 0)."""
    if not deg or not len(rows):
        return
    a = math.radians(deg)
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    S = UV[rows] @ Rm.T
    UV[rows] = S - S.min(axis=0)


def shelf(UV, piece, origin=(0.0, 0.0), gap=0.05, width=None):
    """Lay the pieces side by side in rows (sizes kept), from `origin`, largest
    first. Returns the top of the last row."""
    ids = [p for p in np.unique(piece) if p >= 0]
    if not ids:
        return origin[1]
    boxes = []
    for p in ids:
        rows = np.flatnonzero(piece == p)
        lo, hi = UV[rows].min(axis=0), UV[rows].max(axis=0)
        boxes.append((rows, lo, hi - lo))
    boxes.sort(key=lambda b: -b[2][1])
    total = sum(float(b[2][0] * b[2][1]) for b in boxes) or 1.0
    if width is None:
        width = max(math.sqrt(total) * 1.3, max(float(b[2][0]) for b in boxes))
    x, y, row_h = origin[0], origin[1], 0.0
    for rows, lo, sz in boxes:
        if x > origin[0] and x + sz[0] > origin[0] + width:
            x, y, row_h = origin[0], y + row_h + gap, 0.0
        UV[rows] = UV[rows] - lo + (x, y)
        x += float(sz[0]) + gap
        row_h = max(row_h, float(sz[1]))
    return y + row_h


# -- the pass after every rebuild -------------------------------------------
def mesh_arrays(me):
    npoly, nl = len(me.polygons), len(me.loops)
    ls = np.empty(npoly, np.int64)
    lt = np.empty(npoly, np.int64)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(nl, np.int64)
    me.loops.foreach_get("vertex_index", lv)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3), lv, ls, lt


def set_uvs(me, UV):
    uvl = me.uv_layers.get(UV_NAME) or me.uv_layers.new(name=UV_NAME)
    me.uv_layers.active = uvl
    uvl.data.foreach_set("uv", np.asarray(UV, np.float32).ravel())


def _anchor_faces(F, anchors):
    """Face index of each stored (slot, point): the face the point lies on, or
    None when no face is there any more."""
    tol = 1e-4 * F.size
    out = []
    for slot, p in anchors:
        k, d = F.nearest(np.asarray(p, float))
        out.append((int(slot), k if k >= 0 and d < tol else None))
    return out


def anchor_of(brep, i):
    f = brep.faces()[i]
    ip = f.interior_point()
    return [float(c) for c in ((ip.x, ip.y, ip.z) if hasattr(ip, "x") else ip)]


def apply(obj, brep, P, polys_all):
    """After store.write_mesh: face labels, materials and UVs on obj's mesh
    (its triangles before Blender dropped the degenerate ones: polys_all)."""
    d = get(obj)
    if d is None:
        return
    try:
        _apply(obj, brep, P, polys_all, d)
        LAST[obj.name] = None
    except Exception as exc:
        LAST[obj.name] = str(exc)
        print("NURBS texture:", obj.name, exc)


def _apply(obj, brep, P, polys_all, d):
    from . import plywood
    me = obj.data
    keep = [len(set(f)) == len(f) for f in polys_all]
    L = face_labels(brep, P, polys_all, store.LAST_RAW[0] or None)[np.array(keep, bool)]
    if len(L) != len(me.polygons):
        raise ValueError(f"{len(L)} labels for {len(me.polygons)} faces")
    att = me.attributes.get(ATTR) or me.attributes.new(ATTR, 'INT', 'FACE')
    att.data.foreach_set("value", L.astype(np.int32))
    nf = brep.face_count()
    slot = np.zeros(nf, np.int64)
    ply = d.get("plywood")
    if ply is not None:
        rev = obj.get(store.PROP_REV, 0)
        swap = bool(ply.get("swap", False))
        hit = _cache.get(obj.session_uid)
        if hit is None or hit[0] != rev or hit[1] != swap or len(hit[2]) != nf:
            cls, t = plywood.classify(me, obj.matrix_world, swap=swap)
            fc = np.full(nf, plywood.EDGE, np.int64)
            for k in range(nf):
                c = cls[L == k]
                if len(c):
                    fc[k] = int(np.bincount(c, minlength=3).argmax())
            hit = (rev, swap, fc, t)
            _cache[obj.session_uid] = hit
        slot[:] = hit[2]
        obj["ply_thickness"] = hit[3]             # (the edge material follows a new thickness)
        if not ply.get("plies"):
            obj["ply_count"] = plywood._auto_plies(hit[3])
    F = None
    if d.get("faces"):
        F = _Faces(brep)
        for s, k in _anchor_faces(F, d["faces"]):
            if k is not None:
                slot[k] = s
    n_slots = max(len(me.materials), 1)
    me.polygons.foreach_set("material_index", np.clip(slot[L], 0, n_slots - 1).astype(np.int32))
    # UVs
    Pm, lv, ls, lt = mesh_arrays(me)
    if ply is not None:
        cls = hit[2][L]
        plywood.make_uvs(me, obj.matrix_world, cls, float(ply.get("grain", 0.0)))
    else:
        UV, piece = unwrap(Pm @ np.array(obj.matrix_world.to_3x3()).T, lv, ls, lt, L)
        grain = float(d.get("grain", 0.0))
        if grain:
            for p in np.unique(piece[piece >= 0]):
                turn(UV, np.flatnonzero(piece == p), grain)
        shelf(UV, piece)
        set_uvs(me, UV)
    me.update()


def rewrite(obj, context=None):
    store.write_mesh(obj, store.get_brep(obj), context)


# -- Assign Material ----------------------------------------------------------
def _picked_faces(obj):
    from . import subsel
    try:
        return list(subsel.picks(obj, "FACE"))
    except Exception:
        return []


class NPROTO_OT_face_material(bpy.types.Operator):
    """Give B-rep faces of a NURBS object a material: the active material slot
    goes on the faces picked in Face mode (all faces when none are picked).
    The object gets UVs in metres, each face its own island (Plywood: one per
    veneer side), and keeps both through every rebuild"""
    bl_idname = "nproto.face_material"
    bl_label = "Assign Material to Faces"
    bl_options = {'REGISTER', 'UNDO'}

    action: EnumProperty(
        name="Action", default="ASSIGN",
        items=[("ASSIGN", "Assign", "The active material slot on the picked faces (all faces when none are picked)"),
               ("CLEAR", "Clear", "The picked faces (all when none are picked) back to the first slot "
                                  "(or to their Plywood material)"),
               ("REMOVE", "Remove Texture", "No per-face materials, Plywood sorting or UVs any more "
                                            "(the material slots stay)")])

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return context.mode == 'OBJECT' and o is not None and store.is_nurbs(o)

    def execute(self, context):
        obj = context.active_object
        brep = store.get_brep(obj)
        if self.action == "REMOVE":
            put(obj, None)
            _cache.pop(obj.session_uid, None)
            if obj.get("nproto_plywood") is not None:
                del obj["nproto_plywood"]
            rewrite(obj, context)
            self.report({'INFO'}, f"{obj.name}: texture removed")
            return {'FINISHED'}
        picked = _picked_faces(obj)
        idx = picked or list(range(brep.face_count()))
        d = get(obj) or {}
        F = _Faces(brep)
        old = [(s, p) for (s, p), (_, k) in zip(d.get("faces", []), _anchor_faces(F, d.get("faces", [])))
               if k is not None and k not in idx]
        if self.action == "ASSIGN":
            if not obj.data.materials:
                self.report({'WARNING'}, "Add a material slot first (Material Properties)")
                return {'CANCELLED'}
            s = int(obj.active_material_index)
            old += [(s, anchor_of(brep, i)) for i in idx]
        d["faces"] = old
        if not d["faces"]:
            d.pop("faces")
        put(obj, d or {"faces": []})
        rewrite(obj, context)
        err = LAST.get(obj.name)
        if err:
            self.report({'WARNING'}, f"{obj.name}: {err}")
            return {'CANCELLED'}
        what = f"{len(idx)} picked face(s)" if picked else f"all {len(idx)} faces"
        mat = obj.data.materials[obj.active_material_index] if self.action == "ASSIGN" else None
        self.report({'INFO'}, f"{obj.name}: {what} " + (f"-> {mat.name if mat else 'empty slot'}"
                                                        if self.action == "ASSIGN" else "cleared"))
        return {'FINISHED'}


def draw_material_panel(self, context):
    """Under the material slots of a NURBS object: assign to its B-rep faces."""
    o = context.object
    if o is None or not store.is_nurbs(o) or context.mode != 'OBJECT':
        return
    lay = self.layout
    row = lay.row(align=True)
    n = len(_picked_faces(o))
    row.operator("nproto.face_material", text="Assign to Picked Faces" if n else "Assign to All Faces",
                 icon='FACESEL').action = "ASSIGN"
    row.operator("nproto.face_material", text="Clear").action = "CLEAR"
    if PROP in o:
        row.operator("nproto.face_material", text="", icon='X').action = "REMOVE"


_PANELS = ("EEVEE_MATERIAL_PT_context_material", "CYCLES_PT_context_material", "MATERIAL_PT_context_material")
classes = (NPROTO_OT_face_material,)


def register():
    for name in _PANELS:
        p = getattr(bpy.types, name, None)
        if p is not None:
            p.append(draw_material_panel)


def unregister():
    for name in _PANELS:
        p = getattr(bpy.types, name, None)
        if p is not None:
            try:
                p.remove(draw_material_panel)
            except Exception:
                pass
