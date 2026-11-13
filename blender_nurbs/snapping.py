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

Plain mesh objects (as the view shows them: modifiers applied) snap too, like
curves and surfaces, always: vertices, edges, edge middles, face centres; and,
while O has it on, as the last resort the point on the face under the mouse.
An object that can't be selected never snaps (a locked scan or reference);
holding Alt places a point without any snapping (drawtools). Each mesh is read once per session (MESH_CACHE, dropped when
its geometry changes), so a scan doesn't stall every tool start; a point cloud
(more than LOOSE_MAX loose points) snaps only on its faces. A mesh is not sampled up front: on every mouse move a few rays at
and around the mouse (MESH_RINGS) hit its faces, and only those faces give
vertices, edges and centres. So a dense mesh (a subdivided body) snaps as fast
as a cube, and what lies behind the faces in front is never offered. Mesh
points have no source: a CV Curve end on a mesh is G0 only.
Everything is measured on screen (SNAP_PX). Where targets overlap on screen (a
box's top and bottom face centres in a top view) the one nearest the viewer wins.

Targets are collected once when the tool starts and projected with the view
matrix in numpy on every mouse move, so thousands of CVs and edge samples still
snap at mouse-move rate. A face centre costs the kernel ~10 ms, so it is worked
out only once the mouse is over that face's bounding box, and cached per
surface revision.
"""

import math

import numpy as np
from bpy.app.handlers import persistent

from . import curvejoin, curvetools, geom, store

SNAP_PX = 12.0
TIE_PX = 1.5            # targets this close in rank count as overlapping: nearest the viewer wins
BONUS = {"first": 4.0, "end": 4.0, "vertex": 4.0, "mid": 2.0, "center": 2.0, "cv": 0.0}
LABELS = {"first": "Close", "end": "End", "mid": "Mid", "center": "Center", "cv": "CV",
          "edge": "Edge", "curve": "On Curve", "perp": "Perpendicular", "tan": "Tangent",
          "vertex": "Vertex", "face": "On Face", "line": "On Line"}
MESH_ELEMENTS = ("VERTEX", "EDGE", "EDGE_MIDPOINT", "FACE", "FACE_MIDPOINT")
MESH_RINGS = (0.5, 1.0)     # rays around the mouse at these fractions of SNAP_PX (8 each), and one at it
LOOSE_MAX = 200_000        # more loose vertices + edges than this (a point cloud) don't snap
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

    def ray(self, xy):
        """(origin, unit direction) in world space of the view ray through region
        pixel xy (perspective or orthographic): from the near clip plane on."""
        inv = getattr(self, "_inv", None)
        if inv is None:
            inv = self._inv = np.linalg.inv(self.PM)
        x, y = (np.asarray(xy, float) - self.half) / self.half
        a = inv @ np.array([x, y, -1.0, 1.0])
        b = inv @ np.array([x, y, 1.0, 1.0])
        a, b = a[:3] / a[3], b[:3] / b[3]
        d = b - a
        return a, d / (np.linalg.norm(d) or 1.0)


def _choose(score, depth):
    """Index of the best score; among those within TIE_PX of it, the one nearest
    the viewer (coincident points: the better score)."""
    best = float(score.min())
    near = np.flatnonzero(score <= best + TIE_PX)
    d0 = float(depth[near].min())
    tol = 1e-6 * (abs(d0) + 1e-3)
    same = near[depth[near] <= d0 + tol]
    return int(same[np.argmin(score[same])])


class MeshData:
    """A mesh object's evaluated geometry (modifiers applied) in its own space,
    kept small: single-precision vertices, edges, face loops and a BVH of its
    faces. Built once per session (MESH_CACHE) and dropped when the mesh
    changes, so a scan is read once, not every time a sketch tool starts."""

    def __init__(self, obj):
        import bpy
        from mathutils.bvhtree import BVHTree
        dg = bpy.context.evaluated_depsgraph_get()
        ob = obj.evaluated_get(dg)
        me = ob.to_mesh()
        try:
            nv, ne, nf, nl = len(me.vertices), len(me.edges), len(me.polygons), len(me.loops)
            self.V = np.empty(nv * 3, np.float32)
            me.vertices.foreach_get("co", self.V)
            self.V = self.V.reshape(-1, 3)
            self.E = np.empty(ne * 2, np.int32)
            me.edges.foreach_get("vertices", self.E)
            self.E = self.E.reshape(-1, 2)
            self.ls = np.empty(nf, np.int32)
            me.polygons.foreach_get("loop_start", self.ls)
            self.lt = np.empty(nf, np.int32)
            me.polygons.foreach_get("loop_total", self.lt)
            self.lv = np.empty(nl, np.int32)
            me.loops.foreach_get("vertex_index", self.lv)
            self.le = np.empty(nl, np.int32)
            me.loops.foreach_get("edge_index", self.le)
        finally:
            ob.to_mesh_clear()
        self.tree = BVHTree.FromObject(obj, dg) if nf else None     # hits name the evaluated polygons
        # loose vertices and edges (no face for a ray to hit: a wire outline,
        # imported lines) snap like curves, without a ray -- but a point cloud
        # (a scan as millions of loose points) would cost every mouse move: skipped
        used_v = np.zeros(nv, bool)
        used_v[self.lv] = True
        used_e = np.zeros(ne, bool)
        used_e[self.le] = True
        self.loose_v = np.flatnonzero(~used_v).astype(np.int32)
        self.loose_e = np.flatnonzero(~used_e).astype(np.int32)
        self.loose_skipped = len(self.loose_v) + len(self.loose_e) > LOOSE_MAX
        if self.loose_skipped:
            print(f"NURBS snapping: {obj.name} has {len(self.loose_v):,} loose points and {len(self.loose_e):,} "
                  f"loose edges; they don't snap (more than {LOOSE_MAX:,}), its faces do")
            self.loose_v = self.loose_v[:0]
            self.loose_e = self.loose_e[:0]


MESH_CACHE = {}         # object session_uid -> MeshData


def mesh_data(obj):
    key = obj.session_uid
    d = MESH_CACHE.get(key)
    if d is None:
        d = MESH_CACHE[key] = MeshData(obj)
    return d


def forget_meshes(depsgraph=None):
    """Drop the cached mesh of every object whose geometry changed (all of them
    without a depsgraph: a file was loaded)."""
    if depsgraph is None:
        MESH_CACHE.clear()
        return
    if not MESH_CACHE:
        return
    import bpy
    for u in depsgraph.updates:
        if not u.is_updated_geometry:
            continue
        idd = u.id.original if u.id is not None else None
        if isinstance(idd, bpy.types.Object):
            MESH_CACHE.pop(idd.session_uid, None)
        elif isinstance(idd, bpy.types.Mesh):           # every object using this mesh
            for o in bpy.data.objects:
                if o.data == idd:
                    MESH_CACHE.pop(o.session_uid, None)


class MeshTarget:
    """A plain mesh object for one sketch tool run: its cached MeshData in the
    object's current place. Vertices and centres go to world space only for
    the few faces near the mouse."""

    def __init__(self, obj):
        self.obj, self.ok, self.d = obj, None, None
        M = np.array(obj.matrix_world, float)
        self.R, self.t = M[:3, :3], M[:3, 3]
        Minv = np.linalg.inv(M)
        self.Ri, self.ti = Minv[:3, :3], Minv[:3, 3]
        self.box = np.array(obj.bound_box, float) @ self.R.T + self.t     # world corners (8, 3)

    def near(self, view, mouse):
        """Could the mouse be within SNAP_PX of this mesh? (its bounding box on screen)"""
        xy, _, ok = view.project(self.box)
        if not ok.all():
            return True             # partly behind the viewer: can't tell
        return bool(np.all(mouse >= xy.min(axis=0) - SNAP_PX) and np.all(mouse <= xy.max(axis=0) + SNAP_PX))

    def ready(self):
        if self.ok is None:
            try:
                self.d = mesh_data(self.obj)
                self.ok = True
            except Exception as exc:
                print("NURBS snapping: mesh skipped", getattr(self.obj, "name", "?"), exc)
                self.ok = False
        return self.ok

    @property
    def loose_v(self):
        return self.d.loose_v

    @property
    def loose_e(self):
        return self.d.loose_e

    def world(self, idx):
        return self.d.V[idx].astype(float) @ self.R.T + self.t

    def edges(self, es):
        """World end points (A, B) of these edges."""
        E = self.d.E[es]
        return self.world(E[:, 0]), self.world(E[:, 1])

    def centers(self, polys):
        """World centres of these faces: the mean of their vertices (as Blender's)."""
        return np.array([self.world(self.d.lv[self.d.ls[p]:self.d.ls[p] + self.d.lt[p]]).mean(axis=0)
                         for p in polys], float).reshape(-1, 3)

    def cast(self, o, d):
        """(world hit point, polygon index, distance from o) of the ray's first hit, or None."""
        from mathutils import Vector
        if self.d.tree is None:
            return None
        dl = self.Ri @ d
        dl /= np.linalg.norm(dl) or 1.0
        loc, _, idx, dist = self.d.tree.ray_cast(Vector(self.Ri @ o + self.ti), Vector(dl))
        if loc is None or idx is None:
            return None
        # the hit is single precision along the whole ray (~1e-7 of its length:
        # 1e-5 from an orthographic view's far-off near plane); cast again from
        # just before it -- nothing lies in between, it was the first hit
        back = np.array(loc, float) - (1e-4 * float(dist) + 1e-9) * dl
        loc2, _, idx2, _ = self.d.tree.ray_cast(Vector(back), Vector(dl))
        if loc2 is not None and idx2 is not None:
            loc, idx = loc2, idx2
        P = np.array(loc, float) @ self.R.T + self.t
        return P, int(idx), float(np.linalg.norm(P - o))

    def edge_normals(self, polys):
        """{edge index: world normal of the first of these polygons it bounds}
        (Newell's normal: a mesh edge's "up" for a profile standing on it)."""
        d, out = self.d, {}
        for p in polys:
            sl = slice(d.ls[p], d.ls[p] + d.lt[p])
            W = self.world(d.lv[sl])
            Wn = np.roll(W, -1, axis=0)
            n = np.array([((W[:, 1] - Wn[:, 1]) * (W[:, 2] + Wn[:, 2])).sum(),
                          ((W[:, 2] - Wn[:, 2]) * (W[:, 0] + Wn[:, 0])).sum(),
                          ((W[:, 0] - Wn[:, 0]) * (W[:, 1] + Wn[:, 1])).sum()])
            n /= np.linalg.norm(n) or 1.0
            for e in d.le[sl]:
                out.setdefault(int(e), n)
        return out

    def around(self, polys):
        """(vertex indices, edge indices) of these polygons."""
        d = self.d
        loops = np.concatenate([np.arange(d.ls[p], d.ls[p] + d.lt[p]) for p in polys])
        return np.unique(d.lv[loops]), np.unique(d.le[loops])


def _ray_segment(o, d, A, B):
    """The point of segment AB nearest the ray (o, d unit), or None when they are parallel."""
    u, w = B - A, A - o
    a, b, dd, e = float(u @ u), float(u @ d), float(u @ w), float(d @ w)
    den = a - b * b
    if den <= 1e-18 * max(a, 1e-30):
        return None
    s = min(max((b * e - dd) / den, 0.0), 1.0)
    return A + s * u


class Targets:
    def __init__(self):
        self._pts, self.kinds, self.tans, self.srcs = [], [], [], []
        self._lines = []        # (world polyline (m, 3), refine(segment, s) -> world point, kind, src, meta)
        self._faces = []        # (world bbox corners (8, 3), centre() -> world point or None)
        self._meshes = []       # MeshTarget
        self.mesh_elements = lambda: MESH_ELEMENTS      # which mesh elements snap (read on every move)

    def mesh(self, target):
        self._meshes.append(target)

    def _mesh_near(self, view, mouse):
        """The mesh elements near the mouse: {"P", "kinds"} points (vertices,
        edge middles, face centres), {"A", "B"} edges, "face" the point on the
        face under the mouse (or None), each only when switched on."""
        el = set(self.mesh_elements() or ())
        meshes = [m for m in self._meshes if m.near(view, mouse) and m.ready()] if el else []
        if not meshes:
            return None
        offs = [np.zeros(2)] + [SNAP_PX * f * np.array([math.cos(a), math.sin(a)])
                                for f in MESH_RINGS for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        polys, face = {}, None
        for j, off in enumerate(offs):
            o, d = view.ray(mouse + off)
            best = None
            for m in meshes:            # each ray: only the face nearest the viewer counts
                h = m.cast(o, d)
                if h is not None and (best is None or h[2] < best[1][2]):
                    best = (m, h)
            if best is None:
                continue
            polys.setdefault(id(best[0]), (best[0], set()))[1].add(best[1][1])
            if j == 0:
                face = best[1][0]
        P, kinds, frames, A, B, F = [], [], [], [], [], []
        hit = [(m, sorted(ps)) for m, ps in polys.values()]
        for m in meshes:
            if len(m.loose_v) or len(m.loose_e):
                hit.append((m, None))
        for m, ps in hit:
            if ps is None:          # the mesh's loose vertices and edges
                vs, es, ps = m.loose_v, m.loose_e, []
            else:
                vs, es = m.around(ps)
            ea, eb = m.edges(es)
            # an edge's frame: its direction and its face's normal (a profile stands on it)
            nmap = m.edge_normals(ps) if len(ps) else {}
            T = (eb - ea) / np.maximum(np.linalg.norm(eb - ea, axis=1), 1e-30)[:, None]
            U = np.array([nmap.get(int(e), np.zeros(3)) for e in es], float).reshape(-1, 3)
            fr = [np.array([T[i], U[i]]) for i in range(len(es))]
            if "VERTEX" in el:
                P.append(m.world(vs))
                kinds += ["vertex"] * len(vs)
                frames += [None] * len(vs)
            if "EDGE_MIDPOINT" in el:
                P.append(0.5 * (ea + eb))
                kinds += ["mid"] * len(es)
                frames += fr
            if "FACE_MIDPOINT" in el and len(ps):
                P.append(m.centers(ps))
                kinds += ["center"] * len(ps)
                frames += [None] * len(ps)
            if "EDGE" in el:
                A.append(ea)
                B.append(eb)
                F += fr
        return {"P": np.vstack(P) if P else np.zeros((0, 3)), "kinds": kinds, "frames": frames,
                "A": np.vstack(A) if A else np.zeros((0, 3)), "B": np.vstack(B) if B else np.zeros((0, 3)),
                "F": F, "face": face if "FACE" in el else None}

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
                                                                     "faces": len(self._faces),
                                                                     "meshes": len(self._meshes)}

    def pick(self, view, mouse, extra=(), edges=True, segments=()):
        """(world point, kind, outward tangent or None, src or None) under the mouse, or None.
        A mesh edge or edge centre has no src; its "tangent" is its frame: rows
        (edge direction, its face's normal -- zero for a loose edge).
        extra = [(point, tangent, kind)] (the points drawn so far), segments =
        [(A, B[, middle?[, refine]])] (the lines drawn so far: their middles
        snap as points unless middle is False -- an arc's chords --, any point on
        them as "line", put exactly onto the curve by refine when given). edges=False: points only (a lock, Ctrl or
        Shift is in charge)."""
        mouse = np.asarray(mouse, float)
        P, kinds, tans, srcs = self.P, self.kinds, self.tans, self.srcs
        extra = list(extra)
        segs, seg_refine = [], []       # (A, B[, middle snaps?[, refine(point) -> exact point]])
        for sg in segments:
            a, b = np.asarray(sg[0], float), np.asarray(sg[1], float)
            if np.linalg.norm(b - a) <= 1e-12:
                continue
            if len(sg) < 3 or sg[2]:
                extra.append((0.5 * (a + b), None, "mid"))
            segs.append((a, b))
            seg_refine.append(sg[3] if len(sg) > 3 else None)
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
        mesh = self._mesh_near(view, mouse) if self._meshes else None
        if mesh is not None and len(mesh["P"]):
            xy, z, ok = view.project(mesh["P"])
            d = np.linalg.norm(xy - mouse, axis=1)
            for i in np.flatnonzero(ok & (d <= SNAP_PX)):
                kd = mesh["kinds"][i]
                cands.append((d[i] - BONUS.get(kd, 0.0), z[i], mesh["P"][i], kd, mesh["frames"][i], None))
        if cands:
            k = _choose(np.array([c[0] for c in cands]), np.array([c[1] for c in cands]))
            return cands[k][2], cands[k][3], cands[k][4], cands[k][5]
        if not edges:
            return None
        hits = []           # (distance px, depth, point, kind, src, frame): the best curve / surface edge, mesh edge
        SA = np.array([a for a, _ in segs], float).reshape(-1, 3)
        SB = np.array([b for _, b in segs], float).reshape(-1, 3)
        for A, B, curves in ((self.A, self.B, True),) + (((mesh["A"], mesh["B"], False),) if mesh else ()) \
                + (((SA, SB, "line"),) if len(SA) else ()):
            if not len(A):
                continue
            a, za, oka = view.project(A)
            b, zb, okb = view.project(B)
            dv = b - a
            L2 = np.maximum((dv * dv).sum(axis=1), 1e-18)
            s = np.clip(((mouse - a) * dv).sum(axis=1) / L2, 0.0, 1.0)
            dist = np.linalg.norm(a + s[:, None] * dv - mouse, axis=1)
            hit = np.flatnonzero(oka & okb & (dist <= SNAP_PX))
            if not len(hit):
                continue
            depth = za + s * (zb - za)
            k = hit[_choose(dist[hit], depth[hit])]
            if curves is True:
                W, refine, kind, src, _ = self._lines[self.owner[k]]
                hits.append((dist[k], depth[k], np.asarray(refine(int(self.seg[k]), float(s[k])), float), kind, src,
                             None))
            else:           # the point of the mesh edge / drawn line under the mouse (exact in perspective too)
                Q = _ray_segment(*view.ray(mouse), A[k], B[k])
                Q = A[k] + s[k] * (B[k] - A[k]) if Q is None else Q
                if curves == "line":
                    if seg_refine[k] is not None:
                        Q = np.asarray(seg_refine[k](Q), float)
                    hits.append((dist[k], depth[k], Q, "line", None, None))
                else:
                    hits.append((dist[k], depth[k], Q, "edge", None, mesh["F"][k]))
        if hits:
            k = _choose(np.array([h[0] for h in hits]), np.array([h[1] for h in hits]))
            return hits[k][2], hits[k][3], hits[k][5], hits[k][4]
        if mesh is not None and mesh["face"] is not None:
            return np.asarray(mesh["face"], float), "face", None, None
        return None

    def near_lines(self, view, mouse, radius, limit=3):
        """[(line index, distance in px)] of the curves / edges passing within
        `radius` px of the mouse, nearest first."""
        if not len(self.A):
            return []
        mouse = np.asarray(mouse, float)
        a, _, oka = view.project(self.A)
        b, _, okb = view.project(self.B)
        dv = b - a
        s = np.clip(((mouse - a) * dv).sum(axis=1) / np.maximum((dv * dv).sum(axis=1), 1e-18), 0.0, 1.0)
        dist = np.linalg.norm(a + s[:, None] * dv - mouse, axis=1)
        best = {}
        for i in np.flatnonzero(oka & okb & (dist <= radius)):
            k = int(self.owner[i])
            if self._lines[k][3] is not None and (k not in best or dist[i] < best[k]):
                best[k] = float(dist[i])
        return sorted(best.items(), key=lambda kv: kv[1])[:limit]

    def line_info(self, k):
        """(world polyline, src) of line k."""
        return self._lines[k][0], self._lines[k][3]

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


def circle_centre(X, tol=1e-6):
    """Centre of the circle the points X lie on (a circle or an arc of one),
    or None: three spread points give the circle, all of them must be on it."""
    X = np.asarray(X, float)
    n = len(X)
    if n < 5:
        return None
    A, B, C = X[0], X[n // 3], X[(2 * n) // 3]
    ab, ac = B - A, C - A
    nn = np.cross(ab, ac)
    q = float(nn @ nn)
    if q <= 1e-24 * max(float(ab @ ab) * float(ac @ ac), 1e-300):
        return None                                   # on a line
    O = A + (float(ac @ ac) * np.cross(nn, ab) + float(ab @ ab) * np.cross(ac, nn)) / (2.0 * q)
    r = float(np.linalg.norm(A - O))
    if float(np.abs(np.linalg.norm(X - O, axis=1) - r).max()) > tol * r \
            or float(np.abs((X - O) @ (nn / math.sqrt(q))).max()) > tol * r:
        return None
    return O


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
        cen = circle_centre(X)
        if cen is not None:                     # a circle or an arc: its centre
            T.point(cen, "center")


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
        if e is not None and len(L) >= 5:              # a circular edge (a rim): its centre
            try:
                t0, t1 = e.parameter_range()
                cen = circle_centre([e.point_at(t0 + (t1 - t0) * k / 12) for k in range(13)])
            except Exception:
                cen = None
            if cen is not None:
                T.point(world(cen), "center")
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


SKETCH_MESH = ("VERTEX", "EDGE", "EDGE_MIDPOINT", "FACE_MIDPOINT")   # always on, like curves and surfaces
SNAP_STATE = {"surface": False}     # O while drawing: onto the mesh face under the mouse (kept this session)


def sketch_mesh_elements():
    """The mesh elements the sketch tools snap to: vertices, edges, edge and
    face centres always; the face itself (a point ON the mesh, off the drawing
    plane) only while O has it on."""
    return set(SKETCH_MESH) | ({"FACE"} if SNAP_STATE["surface"] else set())


@persistent
def _on_depsgraph(scene, depsgraph):
    forget_meshes(depsgraph)


@persistent
def _on_load(*_):
    forget_meshes()


def register():
    import bpy
    for lst, fn in ((bpy.app.handlers.depsgraph_update_post, _on_depsgraph), (bpy.app.handlers.load_post, _on_load)):
        if fn not in lst:
            lst.append(fn)


def unregister():
    import bpy
    for lst, fn in ((bpy.app.handlers.depsgraph_update_post, _on_depsgraph), (bpy.app.handlers.load_post, _on_load)):
        if fn in lst:
            lst.remove(fn)
    MESH_CACHE.clear()


def is_plain_mesh(obj):
    """A mesh object that is not a NURBS surface (those snap exactly, above) nor a CV cage."""
    from .cvs import is_cage
    return obj.type == 'MESH' and not store.is_nurbs(obj) and not is_cage(obj)


def collect(objs, cvs_shown=lambda o: False, mesh_elements=None):
    """Snap targets of these objects (curves, NURBS surfaces, plain meshes) --
    not of those that can't be selected; cvs_shown(obj) says whether a
    surface's CV net is drawn (its CVs snap only then); mesh_elements() -> the
    MESH_ELEMENTS on (read on every mouse move: O counts at once)."""
    T = Targets()
    if mesh_elements is not None:
        T.mesh_elements = mesh_elements
    for o in objs:
        if getattr(o, "hide_select", False):
            continue
        try:
            if o.type == 'CURVE':
                add_curve(T, o)
            elif store.is_nurbs(o):
                add_brep(T, o, cvs_shown(o))
            elif is_plain_mesh(o):
                T.mesh(MeshTarget(o))
        except Exception as exc:
            print("NURBS snapping skipped", o.name, exc)
    return T.done()
