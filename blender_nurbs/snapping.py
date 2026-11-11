"""Snapping for the sketch tools (CV Curve, Line).

Targets, in order of preference when several are under the mouse:
  end     curve end points, and the end points of surface edges (B-rep vertices)
  mid     the middle (by length) of an open curve or surface edge
  center  a face's centre: its area centroid. On a planar face it stays where it
          is (a disk's or a hole's centre); on a curved face it moves onto the
          face, unless the face wraps around it (a cylinder or a sphere: then
          the axis / centre point itself is kept)
  cv      curve CVs, and the CVs of surfaces whose CV net is shown
  edge    the nearest point ON a surface edge or a curve, when no point above is
          near. Exact: the kernel's closest point on the edge, or the curve's own
          Bezier spans, so a line started there really starts on the edge.
Each target remembers its source (object, edge / spline ref, END or NEAREST),
so CV Curve can make its ends G1 / G2 to it (align.py) and keep them there.
Everything is measured on screen (SNAP_PX). Where targets overlap on screen (a
box's top and bottom face centres in a top view) the one nearest the viewer wins.

Targets are collected once when the tool starts and projected with the view
matrix in numpy on every mouse move, so thousands of CVs and edge samples still
snap at mouse-move rate. A face centre costs the kernel ~10 ms, so it is worked
out only once the mouse is over that face's bounding box, and cached per
surface revision.
"""

import numpy as np

from . import curvejoin, curvetools, geom, store

SNAP_PX = 12.0
TIE_PX = 1.5            # targets this close in rank count as overlapping: nearest the viewer wins
BONUS = {"first": 4.0, "end": 4.0, "mid": 2.0, "center": 2.0, "cv": 0.0}
LABELS = {"first": "Close", "end": "End", "mid": "Mid", "center": "Center", "cv": "CV",
          "edge": "Edge", "curve": "On Curve"}
CURVE_PER = 32          # samples per Bezier span for the on-curve search
CENTER_ACC = 1e-6       # kernel centroid accuracy (1e-3 left a disk's centre 4e-6 off)
_CENTERS = {}           # session_uid -> (rev, {face index: local centre or None})
_FACES = {}             # (session_uid, rev, edge index) -> [(k, face key, local outline polylines)]


class View:
    """World -> region pixels for one 3D view: its perspective matrix, view
    matrix (for the depth) and region size."""

    def __init__(self, PM, VM, width, height):
        self.PM = np.asarray(PM, float)
        self.VM = np.asarray(VM, float)
        self.half = np.array([width, height], float) * 0.5

    def project(self, P):
        """(xy pixels (N, 2), depth in front of the viewer (N,), ok (N,))."""
        P = np.asarray(P, float).reshape(-1, 3)
        C = P @ self.PM[:, :3].T + self.PM[:, 3]
        w = C[:, 3]
        ok = w > 1e-12
        ws = np.where(ok, w, 1.0)
        xy = self.half + self.half * (C[:, :2] / ws[:, None])
        depth = -(P @ self.VM[2, :3] + self.VM[2, 3])
        return xy, depth, ok


def _choose(score, depth):
    """Index of the best score; among those within TIE_PX of it, the one nearest
    the viewer (coincident points: the better score)."""
    best = float(score.min())
    near = np.flatnonzero(score <= best + TIE_PX)
    d0 = float(depth[near].min())
    tol = 1e-6 * (abs(d0) + 1e-3)
    same = near[depth[near] <= d0 + tol]
    return int(same[np.argmin(score[same])])


class Targets:
    def __init__(self):
        self._pts, self.kinds, self.tans, self.srcs = [], [], [], []
        self._lines = []        # (world polyline (m, 3), refine(segment, s) -> world point, kind, src, meta)
        self._faces = []        # (world bbox corners (8, 3), centre() -> world point or None)

    def point(self, P, kind, tangent=None, src=None):
        """src = (object name, ref, "END" / "NEAREST"): what the point lies on."""
        self._pts.append(np.asarray(P, float))
        self.kinds.append(kind)
        self.tans.append(None if tangent is None else np.asarray(tangent, float))
        self.srcs.append(src)

    def line(self, W, refine, kind, src=None, meta=None):
        """meta: {"obj", "edge": index} of a B-rep edge, {"obj", "spline": index} of a curve."""
        if len(W) >= 2:
            self._lines.append((np.asarray(W, float), refine, kind, src, meta or {}))

    def face(self, corners, center):
        self._faces.append((np.asarray(corners, float), center))

    def done(self):
        self.P = np.array(self._pts, float).reshape(-1, 3)
        A, B, owner, seg = [], [], [], []
        for k, (W, _, _, _, _) in enumerate(self._lines):
            A.append(W[:-1])
            B.append(W[1:])
            owner.append(np.full(len(W) - 1, k))
            seg.append(np.arange(len(W) - 1))
        self.A = np.vstack(A) if A else np.zeros((0, 3))
        self.B = np.vstack(B) if B else np.zeros((0, 3))
        self.owner = np.concatenate(owner) if owner else np.zeros(0, int)
        self.seg = np.concatenate(seg) if seg else np.zeros(0, int)
        self.C = np.array([c for c, _ in self._faces], float).reshape(-1, 8, 3)
        return self

    @property
    def counts(self):
        return {k: self.kinds.count(k) for k in set(self.kinds)} | {"lines": len(self._lines),
                                                                     "faces": len(self._faces)}

    def pick(self, view, mouse, extra=(), edges=True):
        """(world point, kind, outward tangent or None, src or None) under the mouse, or None.
        extra = [(point, tangent, kind)] (the points drawn so far). edges=False:
        points only (a lock, Ctrl or Shift is in charge)."""
        mouse = np.asarray(mouse, float)
        P, kinds, tans, srcs = self.P, self.kinds, self.tans, self.srcs
        extra = list(extra)
        if extra:
            P = np.vstack([P, np.array([e[0] for e in extra], float).reshape(-1, 3)])
            kinds = kinds + [e[2] for e in extra]
            tans = tans + [e[1] for e in extra]
            srcs = srcs + [None] * len(extra)
        cands = []          # (score, depth, point, kind, tangent, src)
        if len(P):
            xy, z, ok = view.project(P)
            d = np.linalg.norm(xy - mouse, axis=1)
            for i in np.flatnonzero(ok & (d <= SNAP_PX)):
                cands.append((d[i] - BONUS.get(kinds[i], 0.0), z[i], P[i], kinds[i], tans[i], srcs[i]))
        if len(self.C):
            xy, _, ok = view.project(self.C.reshape(-1, 3))
            xy = xy.reshape(-1, 8, 2)
            ok = ok.reshape(-1, 8).all(axis=1)
            inside = ok & np.all(mouse >= xy.min(axis=1) - SNAP_PX, axis=1) \
                & np.all(mouse <= xy.max(axis=1) + SNAP_PX, axis=1)
            for f in np.flatnonzero(inside):
                c = self._faces[f][1]()
                if c is None:
                    continue
                q, z, okc = view.project(c)
                dc = float(np.linalg.norm(q[0] - mouse))
                if okc[0] and dc <= SNAP_PX:
                    cands.append((dc - BONUS["center"], z[0], np.asarray(c, float), "center", None, None))
        if cands:
            k = _choose(np.array([c[0] for c in cands]), np.array([c[1] for c in cands]))
            return cands[k][2], cands[k][3], cands[k][4], cands[k][5]
        if not edges or not len(self.A):
            return None
        a, za, oka = view.project(self.A)
        b, zb, okb = view.project(self.B)
        dv = b - a
        L2 = np.maximum((dv * dv).sum(axis=1), 1e-18)
        s = np.clip(((mouse - a) * dv).sum(axis=1) / L2, 0.0, 1.0)
        dist = np.linalg.norm(a + s[:, None] * dv - mouse, axis=1)
        hit = np.flatnonzero(oka & okb & (dist <= SNAP_PX))
        if not len(hit):
            return None
        k = hit[_choose(dist[hit], (za + s * (zb - za))[hit])]
        W, refine, kind, src, _ = self._lines[self.owner[k]]
        return np.asarray(refine(int(self.seg[k]), float(s[k])), float), kind, None, src

    def line_index(self, name, ref):
        """Index of the curve / surface edge (object name, ref), or None."""
        for k, (_, _, _, src, _) in enumerate(self._lines):
            if src is not None and src[0] == name and src[1] == ref:
                return k
        return None

    def slide(self, view, mouse, k):
        """(point, kind, src) of line k nearest the mouse on screen, wherever the
        mouse is: the point stays on that edge (exact), and snaps to its ends or
        middle within SNAP_PX. For a lock onto one edge."""
        W, refine, kind, src, meta = self._lines[k]
        mouse = np.asarray(mouse, float)
        pts = [(W[0], "end"), (W[-1], "end")] + ([(meta["mid"], "mid")] if meta.get("mid") is not None else [])
        xy, _, ok = view.project(np.array([p for p, _ in pts], float))
        d = np.linalg.norm(xy - mouse, axis=1)
        d[~ok] = np.inf
        i = int(np.argmin(d))
        if d[i] <= SNAP_PX:
            P, kd = pts[i]
            return np.asarray(P, float), kd, (src[0], src[1], "END" if kd == "end" else "NEAREST")
        a, _, oka = view.project(W[:-1])
        b, _, okb = view.project(W[1:])
        dv = b - a
        s = np.clip(((mouse - a) * dv).sum(axis=1) / np.maximum((dv * dv).sum(axis=1), 1e-18), 0.0, 1.0)
        dist = np.linalg.norm(a + s[:, None] * dv - mouse, axis=1)
        dist[~(oka & okb)] = np.inf
        j = int(np.argmin(dist))
        return np.asarray(refine(j, float(s[j])), float), kind, (src[0], src[1], "NEAREST")

    def through(self, P, first=None):
        """Every curve / surface edge passing exactly through P (an edge's point,
        or a vertex where several meet): [(src, meta, world polyline)], src with
        END where P is that edge's end, else NEAREST. The edge or curve named by
        `first` (object, ref) comes first."""
        P = np.asarray(P, float)
        if not len(self.A):
            return []
        ab = self.B - self.A
        t = np.clip(((P - self.A) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-30), 0.0, 1.0)
        dist = np.linalg.norm(self.A + t[:, None] * ab - P, axis=1)
        out = []
        for k, (W, refine, kind, src, meta) in enumerate(self._lines):
            if src is None:
                continue
            idx = np.flatnonzero(self.owner == k)
            j = idx[np.argmin(dist[idx])]
            size = float(np.ptp(W, axis=0).max()) or 1.0
            tol = 1e-7 * size + 1e-12
            if dist[j] > 1e-2 * size:
                continue
            exact = meta.get("closest")
            Q = exact(P) if exact else refine(int(self.seg[j]), float(t[j]))
            if np.linalg.norm(np.asarray(Q, float) - P) > tol:
                continue
            end = min(np.linalg.norm(P - W[0]), np.linalg.norm(P - W[-1])) <= tol
            out.append(((src[0], src[1], "END" if end else "NEAREST"), meta, W))
        if first is not None:
            out.sort(key=lambda c: (c[0][0], c[0][1]) != tuple(first[:2]))
        return out


# --------------------------------------------------------------------------
# Curves (Blender splines): ends, CVs, the middle, any point on the curve
# --------------------------------------------------------------------------

def _span_point(spans, g):
    k = int(min(max(np.floor(g), 0), len(spans) - 1))
    return curvejoin.eval_span(spans[k], [g - k])[0]


def closest_on_spans(spans, Q, g0, h, rounds=16):
    """Point of the exact curve (global span parameter g) nearest to Q, searched
    around g0 +- h."""
    n = len(spans)
    for _ in range(rounds):
        gs = np.clip(g0 + np.linspace(-h, h, 9), 0.0, n)
        X = np.array([_span_point(spans, g) for g in gs])
        g0 = float(gs[np.argmin(np.linalg.norm(X - Q, axis=1))])
        h *= 0.3
    return _span_point(spans, g0)


def _samples(spans, per=CURVE_PER):
    ts = np.linspace(0.0, 1.0, per + 1)
    pts, gs = [], []
    for i, s in enumerate(spans):
        t = ts if i == 0 else ts[1:]
        pts.append(curvejoin.eval_span(s, t))
        gs.append(i + t)
    return np.vstack(pts), np.concatenate(gs)


def _half_way(X):
    """(segment, fraction) of the point half way along polyline X."""
    d = np.linalg.norm(np.diff(X, axis=0), axis=1)
    c = np.concatenate([[0.0], np.cumsum(d)])
    if c[-1] <= 0:
        return None
    i = int(min(np.searchsorted(c, 0.5 * c[-1]) - 1, len(d) - 1))
    i = max(i, 0)
    return i, (0.5 * c[-1] - c[i]) / (d[i] or 1.0)


def add_curve(T, obj):
    M = np.array(geom.world_matrix(obj), float)
    for si, sp in enumerate(obj.data.splines):
        ref = f"S{si}"
        try:
            spans = curvejoin.transform_spans(curvejoin.spline_spans(sp), M)
        except Exception:
            continue
        closed = sp.use_cyclic_u
        if not closed:
            a, b = curvejoin._ends(spans)
            ta, tb = curvetools.end_tangents(spans)
            T.point(a, "end", -ta, (obj.name, ref, "END"))
            T.point(b, "end", tb, (obj.name, ref, "END"))
        pts = ([b.co for b in sp.bezier_points] if sp.type == 'BEZIER' else [p.co[:3] for p in sp.points])
        for c in pts:
            T.point(M[:3, :3] @ np.array(c[:3], float) + M[:3, 3], "cv")
        X, g = _samples(spans)

        def refine(i, s, X=X, g=g, spans=spans):
            Q = X[i] + s * (X[i + 1] - X[i])
            return closest_on_spans(spans, Q, g[i] + s * (g[i + 1] - g[i]), 1.0 / CURVE_PER)
        def closest(P, X=X, g=g, spans=spans):
            i = int(np.argmin(np.linalg.norm(X - P, axis=1)))
            return closest_on_spans(spans, np.asarray(P, float), g[i], 1.5 / CURVE_PER)
        hw = None if closed else _half_way(X)
        mid = None if hw is None else refine(*hw)
        T.line(X, refine, "curve", (obj.name, ref, "NEAREST"),
               {"obj": obj, "spline": si, "closest": closest, "mid": mid})
        if mid is not None:
            T.point(mid, "mid", None, (obj.name, ref, "NEAREST"))


# --------------------------------------------------------------------------
# Surfaces (B-rep objects): edge ends, edge middles, points on edges, face
# centres, shown CV nets
# --------------------------------------------------------------------------

def _edge_closest(edge, Q):
    try:
        return np.array(edge.closest_point(tuple(float(x) for x in Q))[0], float)
    except Exception:
        return np.asarray(Q, float)


def _plane_of(face, size):
    """(origin, unit normal) when the face's surface is flat, else None."""
    try:
        _, ext = store._k()
        X = np.array(ext.surface_cvs(face.surface())["points"], float).reshape(-1, 3)
    except Exception:
        return None
    o = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - o)
    n = Vt[-1]
    if np.abs((X - o) @ n).max() <= 1e-9 * size:
        return o, n
    return None


def _center_of(face):
    """Local centre of a face (see the module docstring), or None."""
    try:
        c = np.array(face.centroid(CENTER_ACC), float)
        lo, hi = (np.array(x, float) for x in face.bounding_box())
    except Exception:
        return None
    size = float(np.linalg.norm(hi - lo)) or 1.0
    pl = _plane_of(face, size)
    if pl is not None:
        o, n = pl
        return c - float((c - o) @ n) * n
    try:
        foot = lambda x: np.array(face.closest_point(tuple(float(v) for v in x))[0], float)
        q = foot(c)
        dl = 1e-3 * size
        for e in np.eye(3):
            for sg in (1.0, -1.0):
                if np.linalg.norm(foot(c + sg * dl * e) - q) > 20.0 * dl:
                    return c            # the face wraps around its centroid (cylinder, sphere)
        return q
    except Exception:
        return c


def edge_faces(obj, i):
    """The faces along edge i of a NURBS object: [(k, key, world outline
    polylines)], k = index into the edge's faces() (what align's "face" names),
    key the same for the same face reached from any of its edges."""
    rev = obj.get(store.PROP_REV, 0)
    ck = (obj.session_uid, rev, i)
    if ck not in _FACES:
        out = []
        try:
            fs = list(store.get_brep(obj).edges()[i].faces())
        except Exception:
            fs = []
        for k, f in enumerate(fs):
            try:
                lo, hi = (np.round(np.array(x, float), 9) for x in f.bounding_box())
                key = (obj.name, tuple(lo), tuple(hi), round(float(f.area()), 12))
            except Exception:
                key = (obj.name, i, k)
            polys = []
            for e in f.edges():
                try:
                    t0, t1 = e.parameter_range()
                    polys.append(np.array([e.point_at(t0 + (t1 - t0) * j / 32) for j in range(33)], float))
                except Exception:
                    pass
            out.append((k, key, polys))
        _FACES[ck] = out
    M = np.array(geom.world_matrix(obj), float)
    return [(k, key, [pl @ M[:3, :3].T + M[:3, 3] for pl in polys]) for k, key, polys in _FACES[ck]]


def face_center(obj, i):
    """Local centre of face i of a NURBS object (cached per revision)."""
    rev = obj.get(store.PROP_REV, 0)
    hit = _CENTERS.get(obj.session_uid)
    if hit is None or hit[0] != rev:
        hit = _CENTERS[obj.session_uid] = (rev, {})
    if i not in hit[1]:
        try:
            hit[1][i] = _center_of(store.get_brep(obj).faces()[i])
        except Exception:
            hit[1][i] = None
    return hit[1][i]


def add_brep(T, obj, show_cvs=False):
    M = np.array(geom.world_matrix(obj), float)
    R, t = M[:3, :3], M[:3, 3]
    world = lambda L: np.asarray(L, float) @ R.T + t
    d = store.display(obj)
    brep = store.get_brep(obj)
    try:
        bedges = list(brep.edges())
    except Exception:
        bedges = []
    for i, pl in enumerate(d.get("edges", [])):
        if len(pl) < 2:
            continue
        L = np.asarray(pl, float)
        ref = f"E{i}"
        T.point(world(L[0]), "end", None, (obj.name, ref, "END"))
        T.point(world(L[-1]), "end", None, (obj.name, ref, "END"))
        e = bedges[i] if i < len(bedges) else None

        def refine(j, s, L=L, e=e):
            Q = L[j] + s * (L[j + 1] - L[j])
            return world(Q if e is None else _edge_closest(e, Q))
        def closest(P, e=e):
            return world(_edge_closest(e, (np.asarray(P, float) - t) @ np.linalg.inv(R).T)) if e is not None else P
        closed = np.linalg.norm(L[0] - L[-1]) <= 1e-9 * (float(np.ptp(L, axis=0).max()) or 1.0)
        hw = None if closed else _half_way(L)
        mid = None if hw is None else refine(*hw)
        T.line(world(L), refine, "edge", (obj.name, ref, "NEAREST"),
               {"obj": obj, "edge": i, "closest": closest, "mid": mid})
        if mid is not None:
            T.point(mid, "mid", None, (obj.name, ref, "NEAREST"))
    try:
        faces = list(brep.faces())
    except Exception:
        faces = []
    for i, f in enumerate(faces):
        try:
            lo, hi = (np.array(x, float) for x in f.bounding_box())
        except Exception:
            continue
        corners = np.array([[(lo, hi)[a][0], (lo, hi)[b][1], (lo, hi)[c][2]]
                            for a in (0, 1) for b in (0, 1) for c in (0, 1)])

        def center(i=i):
            c = face_center(obj, i)
            return None if c is None else world(c)
        T.face(world(corners), center)
    if show_cvs:
        for net in d.get("nets", []):
            for p in world(np.asarray(net["p"], float).reshape(-1, 3)):
                T.point(p, "cv")


def collect(objs, cvs_shown=lambda o: False):
    """Snap targets of these objects (curves and NURBS surfaces); cvs_shown(obj)
    says whether a surface's CV net is drawn (its CVs snap only then)."""
    T = Targets()
    for o in objs:
        try:
            if o.type == 'CURVE':
                add_curve(T, o)
            elif store.is_nurbs(o):
                add_brep(T, o, cvs_shown(o))
        except Exception as exc:
            print("NURBS snapping skipped", o.name, exc)
    return T.done()
