"""BREP to Mesh: plain mesh objects from NURBS objects, for rendering,
texturing and animation -- meshed afresh at a chosen detail, not a copy of the
display mesh.

Light (default) gives each face the lightest mesh that still shades like it:
  * a flat face is one n-gon (with holes a few, as Blender's faces have none);
  * a curved face is a grid of quads along its (u, v) lines, only as dense as
    the surface bends in each direction -- a cylinder's side is one strip of
    long quads, a sphere a quad grid with triangle fans at its poles; along a
    trim the grid meets the edge with triangles.
Triangles: the same mesh split into triangles (game engines, glTF).

Every B-rep edge is sampled once and the faces on both sides use those points,
so the mesh is welded where the B-rep is joined. Each face corner gets the
exact normal of its own face (creases stay sharp, tangent joins smooth) and
texture coordinates from the face's (u, v) -- Real Size in metres (a texture
keeps its scale from face to face) or 0-1 per face. B-rep edges are UV seams,
creases over 30 degrees are marked sharp, and the face attribute "brep_face"
says which B-rep face a polygon belongs to (select by it to give faces their
own materials).

Detail sets the largest distance from the mesh to the surface (Chord, a
fraction of the object's size) and the largest turn between neighbouring
edges or grid lines (Angle)."""

import math

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from mathutils import geometry as mgeom

from . import store

DETAIL = {"LOW": (0.004, 15.0), "MEDIUM": (0.001, 7.5), "HIGH": (0.00025, 3.0)}
CREASE_DEG = 30.0
LAST = {}       # what the last conversion made, for the Adjust Last Operation panel


def _unit(x):
    x = np.asarray(x, float)
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else x


def _turn(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b)) / (na * nb))))


def _seg_dist(P, A, B):
    """Distances of the points P (k, 3) from the segment AB."""
    d = B - A
    L2 = float(d @ d)
    if L2 <= 0:
        return np.linalg.norm(P - A, axis=-1)
    t = np.clip(((P - A) @ d) / L2, 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * d), axis=-1)


def _turning(T):
    """Largest total turn (radians) of the tangents T (rows, samples, 3) along a row."""
    a, b = T[:, :-1], T[:, 1:]
    na, nb = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2)
    good = (na > 1e-12 * max(float(na.max()), 1e-300)) & (nb > 1e-12 * max(float(nb.max()), 1e-300))
    c = np.where(good, (a * b).sum(axis=2) / np.where(good, na * nb, 1.0), 1.0)
    return float(np.arccos(np.clip(c, -1.0, 1.0)).sum(axis=1).max()) if len(T) else 0.0


def dense_count(turn, ang, base=17, cap=1025):
    """Samples along a direction turning by `turn`: four per allowed angle, so
    the lines kept from them come out at most a third denser than needed."""
    return int(min(cap, max(base, 4 * math.ceil(turn / ang) + 1)))


def distribute(params, X, T, chord, ang):
    """Lines along a direction, spread evenly by how much it bends: each step
    between the samples X (rows, samples, 3), with tangents T, costs
    max(turn / ang, sqrt(turn * length / (8 chord))) on its worst row -- the
    pieces it needs for the angle and for the chord (an arc's sagitta is
    length * turn / 8). N = the total rounded up; the lines go at equal shares
    of it. So a symmetric shape gets symmetric lines and the spacing changes
    smoothly with the curvature (picking lines one after another from an end
    left a short piece at the far end); a straight direction keeps its ends."""
    a, b = T[:, :-1], T[:, 1:]
    na, nb = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2)
    good = (na > 1e-12 * max(float(na.max()), 1e-300)) & (nb > 1e-12 * max(float(nb.max()), 1e-300))
    c = np.where(good, (a * b).sum(axis=2) / np.where(good, na * nb, 1.0), 1.0)
    turn = np.arccos(np.clip(c, -1.0, 1.0))
    L = np.linalg.norm(np.diff(X, axis=1), axis=2)
    cost = np.maximum(turn / ang, np.sqrt(turn * L / (8.0 * chord))).max(axis=0)
    total = float(cost.sum())
    n = max(1, int(math.ceil(total - 1e-6)))
    params = np.asarray(params, float)
    if n == 1:
        return np.array([params[0], params[-1]])
    cum = np.concatenate([[0.0], np.cumsum(cost)])
    out = np.interp(total * np.arange(n + 1) / n, cum, params)
    out[0], out[-1] = params[0], params[-1]
    return out


def sample_edge(e, chord, ang):
    """Parameters along B-rep edge e, spread like a face's grid lines
    (distribute): within chord of the curve, turning by at most ang between
    neighbours. A straight edge keeps its two ends."""
    t0, t1 = (float(x) for x in e.parameter_range())
    tc = np.linspace(t0, t1, 65)
    Tc = np.array([e.tangent_at(float(t)) for t in tc], float)
    D = dense_count(_turning(Tc[None]), ang)
    ts = np.linspace(t0, t1, D)
    X = np.array([e.point_at(float(t)) for t in ts], float)
    T = np.array([e.tangent_at(float(t)) for t in ts], float)
    return distribute(ts, X[None], T[None], chord, ang)


def _inside(Q, loops):
    """Even-odd: which of the 2D points Q lie inside the closed polygons."""
    Q = np.asarray(Q, float).reshape(-1, 2)
    res = np.zeros(len(Q), bool)
    if not len(Q):
        return res
    x, y = Q[:, 0][:, None], Q[:, 1][:, None]
    for L in loops:
        A = np.asarray(L, float)
        B = np.roll(A, -1, axis=0)
        ax, ay, bx, by = A[:, 0][None], A[:, 1][None], B[:, 0][None], B[:, 1][None]
        cross = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = (bx - ax) * (y - ay) / (by - ay) + ax
        res ^= ((cross & (x < xc)).sum(axis=1) % 2) == 1
    return res


def _boundary_dist(Q, loops, batch=2048):
    """Distance of each 2D point to the nearest polygon edge."""
    Q = np.asarray(Q, float).reshape(-1, 2)
    A = np.vstack([np.asarray(L, float) for L in loops])
    B = np.vstack([np.roll(np.asarray(L, float), -1, axis=0) for L in loops])
    d = B - A
    L2 = np.maximum((d * d).sum(axis=1), 1e-300)
    out = np.empty(len(Q))
    for s in range(0, len(Q), batch):
        q = Q[s:s + batch][:, None]
        t = np.clip(((q - A[None]) * d[None]).sum(axis=2) / L2[None], 0.0, 1.0)
        out[s:s + batch] = np.linalg.norm(q - (A[None] + t[..., None] * d[None]), axis=2).min(axis=1)
    return out


def _area2(L):
    L = np.asarray(L, float)
    x, y = L[:, 0], L[:, 1]
    return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) * 0.5


def _plane_axis(n, pts):
    """First axis of a flat face's (u, v): along its outline's main direction (snapping.face_axis)."""
    from .snapping import face_axis
    P = np.asarray(pts, float)
    seg = np.stack([P, np.roll(P, -1, axis=0)], axis=1) if len(P) >= 2 else np.zeros((0, 2, 3))
    return _unit(face_axis(n, seg))


class Mesher:
    """Meshes one B-rep (local space): vertices P, polygons with their corner
    (u, v) and normals, the mesh edges lying on B-rep edges."""

    def __init__(self, brep, chord_rel, angle_deg, uv_mode="REAL"):
        self.brep = brep
        try:
            lo, hi = (np.array(x, float) for x in brep.bounding_box())
            self.diag = float(np.linalg.norm(hi - lo)) or 1.0
        except Exception:
            self.diag = 1.0
        self.chord = chord_rel * self.diag
        self.ang = math.radians(angle_deg)
        self.uv_mode = uv_mode
        self.P = []
        self._vkey = {}
        self._edges = {}
        self.polys = []         # (vertex indices, corner uvs, corner normals, face index, kind):
        #                         kind "flat" (an n-gon), "holes" (a flat face's triangles, around
        #                         its holes: joined into n-gons), "curved" (triangles: joined into quads)
        self.seams = set()      # (i, j), i < j: mesh edges on B-rep edges
        self.problems = []      # faces meshed coarsely or skipped, for the report
        self._req = {}          # edge -> parameters an untrimmed face's grid lines meet it at

    # -- shared points -----------------------------------------------------
    def _add(self, p):
        self.P.append(np.asarray(p, float))
        return len(self.P) - 1

    def _vertex(self, v):
        k = self._vkey.get(v)
        if k is None:
            p = v.point
            k = self._vkey[v] = self._add(p() if callable(p) else p)
        return k

    def edge(self, e):
        """(parameters, vertex indices) of B-rep edge e from its start vertex
        to its end vertex, sampled once for both of its faces."""
        r = self._edges.get(e)
        if r is None:
            a, b = self._vertex(e.start_vertex()), self._vertex(e.end_vertex())
            try:
                length = float(e.length())
            except Exception:
                length = 1.0
            if length <= 1e-12 * self.diag:
                r = (np.zeros(1), [a])
            else:
                req = self._req.get(e)
                if req is not None:                  # a side of an untrimmed face: its grid lines' points
                    t0, t1 = (float(x) for x in e.parameter_range())
                    ts = _merge(np.array([t0, t1] + sorted(req), float), 1e-9 * abs(t1 - t0))
                else:
                    ts = sample_edge(e, self.chord, self.ang)
                p0 = np.array(e.point_at(ts[0]), float)
                if np.linalg.norm(p0 - self.P[a]) > np.linalg.norm(p0 - self.P[b]) + 1e-12 * self.diag:
                    a, b = b, a                  # its parameter runs from the end vertex
                idx = [a] + [self._add(e.point_at(float(t))) for t in ts[1:-1]] + [b]
                r = (ts, idx)
                for i, j in zip(idx[:-1], idx[1:]):
                    if i != j:
                        self.seams.add((min(i, j), max(i, j)))
            self._edges[e] = r
        return r

    def _loop_indices(self, loop):
        """Vertex indices around one boundary loop: [(indices, edge)] per edge use, oriented."""
        out = []
        for e, fwd in loop:
            ts, idx = self.edge(e)
            idx = list(idx)
            sv = e.start_vertex()
            if not fwd:
                idx = idx[::-1]
            # the edge's samples may run end -> start: orient them by its start vertex
            first = self._vkey.get(sv)
            if first is not None and len(idx) > 1:
                want_first = first if fwd else self._vkey.get(e.end_vertex())
                if want_first is not None and idx[0] != want_first and idx[-1] == want_first:
                    idx = idx[::-1]
            out.append((idx, e))
        return out

    # -- faces ---------------------------------------------------------------
    def run(self):
        """Grid lines of every curved face first; an untrimmed face's lines
        then set the points on its edges (so its grid meets them in clean
        quads, and a loft's two rails get the same columns instead of each
        its own); then the faces are meshed."""
        faces = list(self.brep.faces())
        plan = {}
        for fi, f in enumerate(faces):
            try:
                s = f.surface()
                plan[fi] = (s, None if s.is_planar(1e-9 * self.diag) else self._grid(f, s))
            except Exception as exc:
                self.problems.append(f"face {fi}: {exc}")
        for s, g in plan.values():
            for e, pts in (g or {}).get("requests", ()):
                req = self._req.setdefault(e, set())
                t0, t1 = (float(x) for x in e.parameter_range())
                margin = 1e-7 * abs(t1 - t0)
                for p in pts:
                    try:
                        _, t, d = e.closest_point(tuple(float(x) for x in p))
                    except Exception:
                        continue
                    if d <= 1e-6 * self.diag and t0 + margin < t < t1 - margin:
                        req.add(float(t))
        for fi, (s, g) in plan.items():
            try:
                if g is None:
                    self._planar(fi, faces[fi], s)
                else:
                    self._curved(fi, faces[fi], s, g)
            except Exception as exc:
                self.problems.append(f"face {fi}: {exc}")
        return self

    def _outward(self, f, s, u, v):
        try:
            return _unit(f.outward_normal_at_uv(u, v))
        except Exception:
            return _unit(s.normal(u, v))

    def _planar(self, fi, f, s):
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        n = self._outward(f, s, 0.5 * (u0 + u1), 0.5 * (v0 + v1))
        loops = []
        for loop in f.boundary_loops():
            ids = []
            for idx, _ in self._loop_indices(loop):
                for i in idx:
                    if not ids or ids[-1] != i:
                        ids.append(i)
            if len(ids) > 1 and ids[0] == ids[-1]:
                ids.pop()
            if len(ids) >= 3:
                loops.append(ids)
        if not loops:
            self.problems.append(f"face {fi}: no boundary")
            return
        P = np.array(self.P)
        ua = _plane_axis(n, P[loops[0]])
        va = np.cross(n, ua)
        o = P[loops[0][0]]
        xy = lambda ids: np.stack([(P[ids] - o) @ ua, (P[ids] - o) @ va], axis=1)
        if len(loops) == 1:
            ids = loops[0]
            if _area2(xy(ids)) < 0:
                ids = ids[::-1]
            self._add_flat(fi, ids, xy(ids), n)
            return
        # holes: a few n-gons (Blender's faces have none)
        allids = [i for L in loops for i in L]
        coords = xy(allids)
        faces, k = [], 0
        for L in loops:
            faces.append(list(range(k, k + len(L))))
            k += len(L)
        # triangles with the holes left out (output 2), joined into n-gons later
        vs, _, fs, orig, _, _ = mgeom.delaunay_2d_cdt([tuple(c) for c in coords], [], faces, 2,
                                                      1e-9 * self.diag, True)
        vid = []
        for j, ov in enumerate(orig):
            if ov:
                vid.append(allids[ov[0]])
            else:                                    # a crossing the triangulation added: on the plane
                vid.append(self._add(o + vs[j][0] * ua + vs[j][1] * va))
        for fc in fs:
            ids = [vid[j] for j in fc]
            c = np.array([[vs[j][0], vs[j][1]] for j in fc])
            if _area2(c) < 0:
                ids, c = ids[::-1], c[::-1]
            self._add_flat(fi, ids, c, n, kind="holes", span=coords)

    def _add_flat(self, fi, ids, c, n, kind="flat", span=None):
        c = np.asarray(c, float)
        if self.uv_mode == "NORMALIZED":
            ref = np.asarray(span if span is not None else c, float)
            lo, size = ref.min(axis=0), max(float(np.ptp(ref, axis=0).max()), 1e-300)
            c = (c - lo) / size
        self.polys.append((list(ids), [tuple(x) for x in c], [tuple(n)] * len(ids), fi, kind))

    def _grid(self, f, s):
        """A curved face's grid lines U, V (distribute: from rows across the
        whole face), its sizes and Newton starts, and -- for an untrimmed face
        (one loop, every edge on a side of its domain) -- the points where
        its lines meet each edge, which then sample that edge."""
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        Pu, Pv = (u1 - u0) or 1.0, (v1 - v0) or 1.0
        per = store._periodic(s)
        try:
            from .store import _k
            cv = _k()[1].surface_cvs(s)
            cu, cv_ = int(cv["count_u"]), int(cv["count_v"])
        except Exception:
            cu, cv_ = 4, 4
        der = lambda uu, vv: np.array([[np.array(s.derivatives(float(a), float(b)), float) for b in vv]
                                       for a in uu])                      # (len uu, len vv, 3, 3)
        G = 33
        gu, gv = np.linspace(u0, u1, G), np.linspace(v0, v1, G)
        C = der(gu, gv)                                # a coarse grid: turning, sizes, Newton starts
        Xc = C[:, :, 0]
        Du = dense_count(_turning(C[:, :, 1].transpose(1, 0, 2)), self.ang)
        Dv = dense_count(_turning(C[:, :, 2]), self.ang)
        Ru, Rv = (min(33, max(9, 4 * (c - 1) + 1)) for c in (cu, cv_))
        fu, rv = np.linspace(u0, u1, Du), np.linspace(v0, v1, Rv)
        A = der(fu, rv).transpose(1, 0, 2, 3)          # rows along u at Rv heights
        U = distribute(fu, A[:, :, 0], A[:, :, 1], self.chord, self.ang)
        ru, fv = np.linspace(u0, u1, Ru), np.linspace(v0, v1, Dv)
        A = der(ru, fv)                                # rows along v
        V = distribute(fv, A[:, :, 0], A[:, :, 2], self.chord, self.ang)
        Lu = float(np.linalg.norm(np.diff(Xc, axis=0), axis=2).sum(axis=0).mean()) or 1.0
        Lv = float(np.linalg.norm(np.diff(Xc, axis=1), axis=2).sum(axis=1).mean()) or 1.0
        sx, sy = Lu / Pu, Lv / Pv
        flat = Xc.reshape(-1, 3)

        def feet(Q):
            k = store.nearest_rows(flat, Q)
            return np.array([store._newton_uv(s, q, gu[kk // G], gv[kk % G], ((u0, u1), (v0, v1)), per)
                             for q, kk in zip(Q, k.tolist())], float).reshape(-1, 2)

        requests = []
        loops = f.boundary_loops()
        if len(loops) == 1:
            sides = []
            for e, _fwd in loops[0]:
                try:
                    if float(e.length()) <= 1e-12 * self.diag:
                        continue
                    t0, t1 = (float(x) for x in e.parameter_range())
                    q = feet(np.array([e.point_at(t0 + (t1 - t0) * k) for k in (0.25, 0.5, 0.75)], float))
                except Exception:
                    sides = None
                    break
                on = lambda x, ref, P_: bool(np.all(np.abs(x - ref) < 1e-6 * P_))
                ends = feet(np.array([e.point_at(t0), e.point_at(t1)], float))
                if on(q[:, 1], v0, Pv) or on(q[:, 1], v1, Pv):
                    sides.append((e, "v", v0 if on(q[:, 1], v0, Pv) else v1, ends[:, 0]))
                elif on(q[:, 0], u0, Pu) or on(q[:, 0], u1, Pu):
                    sides.append((e, "u", u0 if on(q[:, 0], u0, Pu) else u1, ends[:, 1]))
                else:
                    sides = None                       # a trim: the face isn't untrimmed
                    break
            if sides:
                # where a side is split into several edges, a grid line runs through the vertex
                U = _snap_lines(U, [x for _, c, _, ends in sides if c == "v" for x in ends])
                V = _snap_lines(V, [x for _, c, _, ends in sides if c == "u" for x in ends])
                for e, c, at, _ in sides:
                    pts = [s.evaluate(float(u), float(at)) for u in U[1:-1]] if c == "v" else \
                        [s.evaluate(float(at), float(v)) for v in V[1:-1]]
                    requests.append((e, pts))
        return {"U": U, "V": V, "Lu": Lu, "Lv": Lv, "flat": flat, "gu": gu, "gv": gv, "G": G, "per": per,
                "requests": requests}

    def _curved(self, fi, f, s, g):
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        Pu, Pv = (u1 - u0) or 1.0, (v1 - v0) or 1.0
        per, U, V, Lu, Lv = g["per"], g["U"], g["V"], g["Lu"], g["Lv"]
        flat, gu, gv, Dv = g["flat"], g["gu"], g["gv"], g["G"]
        sx, sy = Lu / Pu, Lv / Pv
        uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        _, su, sv = s.derivatives(uc, vc)
        sign = 1.0 if float(np.cross(su, sv) @ self._outward(f, s, uc, vc)) >= 0 else -1.0
        tol_u, tol_v = 1e-7 * Pu, 1e-7 * Pv
        size = self.diag

        def degenerate(u, v):
            _, a, b = s.derivatives(u, v)
            return np.linalg.norm(a) * Pu < 1e-7 * size, np.linalg.norm(b) * Pv < 1e-7 * size

        def feet(Q):
            k = store.nearest_rows(flat, Q)
            return np.array([store._newton_uv(s, q, gu[kk // Dv], gv[kk % Dv], ((u0, u1), (v0, v1)), per)
                             for q, kk in zip(Q, k.tolist())], float).reshape(-1, 2)

        # -- the boundary in (u, v): each loop, its pieces joined across seams and poles
        loops_uv, loops_id, poles = [], [], []
        for loop in f.boundary_loops():
            occ = []
            for idx, e in self._loop_indices(loop):
                if len(idx) < 2:
                    continue                          # a collapsed edge (a pole): the pole segment covers it
                UV = feet(np.array([self.P[i] for i in idx]))
                for j in (0, len(UV) - 1):            # at a pole one parameter means nothing: take the neighbour's
                    du_, dv_ = degenerate(*UV[j])
                    nb = UV[1] if j == 0 else UV[-2]
                    if du_:
                        UV[j, 0] = nb[0]
                    if dv_:
                        UV[j, 1] = nb[1]
                for c, (p, P_) in enumerate(((per[0], Pu), (per[1], Pv))):
                    if p:
                        for j in range(1, len(UV)):
                            UV[j, c] += round((UV[j - 1, c] - UV[j, c]) / P_) * P_
                occ.append([idx, UV, e])
            if not occ:
                continue
            seen = {}
            ids, uvs = [], []
            ku_r = (-2, -1, 0, 1, 2) if per[0] else (0,)
            kv_r = (-2, -1, 0, 1, 2) if per[1] else (0,)
            for idx, UV, e in occ:
                if uvs:
                    prev = np.array(uvs[-1])
                    best = None
                    for ku in ku_r:
                        for kv in kv_r:
                            S = UV + (ku * Pu, kv * Pv)
                            first = seen.get(e)
                            if first is not None and len(first) == len(S) and \
                                    np.abs(S - first[::-1]).max() < 1e-6 * (Pu + Pv):
                                continue                   # a seam's second use lies a period away
                            d = math.hypot((S[0, 0] - prev[0]) * sx, (S[0, 1] - prev[1]) * sy)
                            if best is None or d < best[0] - 1e-12 * size:
                                best = (d, S)
                    UV = best[1]
                seen.setdefault(e, UV)
                for i, q in zip(idx, UV):
                    if ids and ids[-1] == i:
                        if np.abs(np.array(uvs[-1]) - q).max() > 1e-6 * (Pu + Pv):
                            poles.append((len(loops_id), len(ids) - 1))     # same point, another (u, v): a pole line
                            ids.append(i)
                            uvs.append(tuple(q))
                        continue
                    ids.append(i)
                    uvs.append(tuple(q))
            if len(ids) > 1 and ids[0] == ids[-1]:
                if np.abs(np.array(uvs[0]) - np.array(uvs[-1])).max() > 1e-6 * (Pu + Pv):
                    poles.append((len(loops_id), len(ids) - 1))
                else:
                    ids.pop()
                    uvs.pop()
            loops_id.append(ids)
            loops_uv.append(np.array(uvs, float))
        natural = False
        if not loops_uv:                              # no boundary at all (a closed torus): the whole domain
            loops_uv, loops_id, natural = None, None, True
        else:
            # the outer loop starts in the domain; holes are moved into it
            L0 = loops_uv[0]
            k0 = (math.floor((L0[:, 0].min() - u0 + 1e-6 * Pu) / Pu) if per[0] else 0,
                  math.floor((L0[:, 1].min() - v0 + 1e-6 * Pv) / Pv) if per[1] else 0)
            loops_uv[0] = L0 - (k0[0] * Pu, k0[1] * Pv)
            lo, hi = loops_uv[0].min(axis=0), loops_uv[0].max(axis=0)
            for k in range(1, len(loops_uv)):
                c = loops_uv[k].mean(axis=0)
                sh = [0.0, 0.0]
                for a, (p, P_) in enumerate(((per[0], Pu), (per[1], Pv))):
                    if p:
                        while c[a] + sh[a] < lo[a]:
                            sh[a] += P_
                        while c[a] + sh[a] > hi[a]:
                            sh[a] -= P_
                loops_uv[k] = loops_uv[k] + sh
            B = loops_uv[0]
            on = (np.abs(B[:, 0] - u0) < 1e-6 * Pu) | (np.abs(B[:, 0] - u1) < 1e-6 * Pu) | \
                 (np.abs(B[:, 1] - v0) < 1e-6 * Pv) | (np.abs(B[:, 1] - v1) < 1e-6 * Pv)
            natural = len(loops_uv) == 1 and bool(on.all()) and abs(_area2(B)) > (1 - 1e-4) * Pu * Pv
        if loops_uv is None:
            loops_uv, loops_id = self._domain_loop(s, U, V)
        # pole lines: one vertex, many (u, v) -- split them at the grid lines
        if poles:
            for li in sorted({p[0] for p in poles}, reverse=True):
                ids, uvs = loops_id[li], [tuple(q) for q in loops_uv[li]]
                new_ids, new_uv = [], []
                pole_at = {j for (l_, j) in poles if l_ == li}
                for j in range(len(ids)):
                    new_ids.append(ids[j])
                    new_uv.append(uvs[j])
                    if j in pole_at:
                        a, b = np.array(uvs[j]), np.array(uvs[(j + 1) % len(uvs)])
                        c = 0 if abs(b[0] - a[0]) > abs(b[1] - a[1]) else 1
                        lines = U if c == 0 else V
                        P_ = Pu if c == 0 else Pv
                        lo_, hi_ = sorted((a[c], b[c]))
                        cand = sorted({float(x + k * P_) for x in lines for k in (-2, -1, 0, 1, 2)
                                       if lo_ + 1e-9 * P_ < x + k * P_ < hi_ - 1e-9 * P_})
                        if a[c] > b[c]:
                            cand = cand[::-1]
                        for x in cand:
                            q = a.copy()
                            q[c] = x
                            new_ids.append(ids[j])
                            new_uv.append(tuple(q))
                loops_id[li], loops_uv[li] = new_ids, np.array(new_uv, float)
        tex = (lambda q: ((q[0] - u0) * sx, (q[1] - v0) * sy)) if self.uv_mode != "NORMALIZED" else \
            (lambda q: ((q[0] - u0) / Pu, (q[1] - v0) / Pv))

        def normal_at(q):
            uu = u0 + (q[0] - u0) % Pu if per[0] else min(max(q[0], u0), u1)
            vv = v0 + (q[1] - v0) % Pv if per[1] else min(max(q[1], v0), v1)
            n = np.array(s.normal(float(uu), float(vv)), float)
            if not np.isfinite(n).all() or abs(np.linalg.norm(n) - 1.0) > 1e-3:
                n = np.array(s.normal(float(uu + (uc - uu) * 1e-3), float(vv + (vc - vv) * 1e-3)), float)
            return tuple(sign * _unit(n))       # (at a pole: just inside it)

        if natural and len(loops_uv) == 1 and self._grid_quads(fi, s, U, V, loops_uv[0], loops_id[0], sign,
                                                              tex, normal_at):
            return
        # interior points: the grid's nodes inside, not crowding the boundary
        scale = np.array([sx, sy])
        loops_xy = [L * scale for L in loops_uv]
        lo, hi = loops_uv[0].min(axis=0), loops_uv[0].max(axis=0)
        UU = sorted({float(x + k * Pu) for x in U for k in ((-1, 0, 1, 2) if per[0] else (0,))
                     if lo[0] + 1e-9 * Pu < x + k * Pu < hi[0] - 1e-9 * Pu})
        VV = sorted({float(x + k * Pv) for x in V for k in ((-1, 0, 1, 2) if per[1] else (0,))
                     if lo[1] + 1e-9 * Pv < x + k * Pv < hi[1] - 1e-9 * Pv})
        nodes = np.array([(a, b) for a in UU for b in VV], float).reshape(-1, 2)
        if len(nodes) and not natural:
            nodes = nodes[_inside(nodes * scale, loops_xy)]
            if len(nodes):
                du = np.diff(np.array([lo[0]] + UU + [hi[0]])).min() * sx
                dv = np.diff(np.array([lo[1]] + VV + [hi[1]])).min() * sy
                nodes = nodes[_boundary_dist(nodes * scale, loops_xy) > 0.3 * min(du, dv)]
        # constrained Delaunay in (u, v) scaled to metres, holes left out
        bxy = np.vstack(loops_xy)
        bid = [i for L in loops_id for i in L]
        buv = np.vstack(loops_uv)
        nb = len(bxy)
        faces, k = [], 0
        for k_, L in enumerate(loops_xy):
            ring = list(range(k, k + len(L)))
            if k_ == 0 and _area2(L) < 0:
                ring = ring[::-1]
            faces.append(ring)
            k += len(L)
        coords = [tuple(c) for c in bxy] + [tuple(c) for c in nodes * scale]
        try:
            vs, _, fs, orig, _, _ = mgeom.delaunay_2d_cdt(coords, [], faces, 2, 1e-9 * (Lu + Lv), True)
        except Exception as exc:
            self.problems.append(f"face {fi}: triangulation failed ({exc})")
            return
        vid, vuv = [], []
        for j, ov in enumerate(orig):
            if ov and ov[0] < nb:
                vid.append(bid[ov[0]])
                vuv.append(buv[ov[0]])
            else:
                q = nodes[ov[0] - nb] if ov else np.array([vs[j][0] / sx, vs[j][1] / sy])
                uu = u0 + (q[0] - u0) % Pu if per[0] else min(max(q[0], u0), u1)
                vv = v0 + (q[1] - v0) % Pv if per[1] else min(max(q[1], v0), v1)
                vid.append(self._add(s.evaluate(float(uu), float(vv))))
                vuv.append(np.array(q, float))
        normals = {}

        def normal(j):
            if j not in normals:
                normals[j] = normal_at(vuv[j])
            return normals[j]

        for fc in fs:
            c = np.array([[vs[j][0], vs[j][1]] for j in fc])
            if _area2(c) < 0:
                fc = fc[::-1]
            if sign < 0:
                fc = fc[::-1]
            ids = [vid[j] for j in fc]
            if len(set(ids)) < 3:
                continue
            self.polys.append((ids, [tex(vuv[j]) for j in fc], [normal(j) for j in fc], fi, "curved"))

    def _grid_quads(self, fi, s, U, V, B, Bid, sign, tex, normal_at):
        """An untrimmed face whose boundary points are exactly its grid's
        (its edges were sampled at its lines): quads straight from the grid,
        no triangulation. A pole row collapses into a fan of triangles, a seam
        closes on its own (both sides are the same edge points). False when
        the boundary doesn't match the grid (a neighbour added points)."""
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        Pu, Pv = (u1 - u0) or 1.0, (v1 - v0) or 1.0
        nU, nV = len(U), len(V)
        if nU < 2 or nV < 2 or len(B) != 2 * (nU - 1) + 2 * (nV - 1):
            return False
        Bn = (np.asarray(B, float) - (u0, v0)) / (Pu, Pv)
        ids = np.full((nU, nV), -1, np.int64)
        for a in range(nU):
            for b in range(nV):
                if a in (0, nU - 1) or b in (0, nV - 1):
                    q = np.array([(U[a] - u0) / Pu, (V[b] - v0) / Pv])
                    d = np.abs(Bn - q).max(axis=1)
                    k = int(np.argmin(d))
                    if d[k] > 1e-6:
                        return False
                    ids[a, b] = Bid[k]
        for a in range(1, nU - 1):
            for b in range(1, nV - 1):
                ids[a, b] = self._add(s.evaluate(float(U[a]), float(V[b])))
        for a in range(nU - 1):
            for b in range(nV - 1):
                cell = [(a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1)]
                if sign < 0:
                    cell = cell[::-1]
                vids = [int(ids[x, y]) for x, y in cell]
                if len(set(vids)) < 3:
                    continue
                qs = [(float(U[x]), float(V[y])) for x, y in cell]
                self.polys.append((vids, [tex(q) for q in qs], [normal_at(q) for q in qs], fi, "grid"))
        return True

    def _domain_loop(self, s, U, V):
        """A face without edges (a torus): its domain's border, the same point
        on both sides of a seam shared by position."""
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        border = [(u, v0) for u in U[:-1]] + [(u1, v) for v in V[:-1]] + \
                 [(u, v1) for u in U[::-1][:-1]] + [(u0, v) for v in V[::-1][:-1]]
        ids, key = [], {}
        for q in border:
            p = np.array(s.evaluate(*q), float)
            kk = tuple(np.round(p / (1e-9 * self.diag)).astype(np.int64))
            if kk not in key:
                key[kk] = self._add(p)
            ids.append(key[kk])
        return [np.array(border, float)], [ids]


def _snap_lines(L, xs):
    """Grid lines L with a line through each x (a vertex splitting a side):
    the nearest line moves onto it when it is less than half a step away,
    else a line is added."""
    L = np.array(L, float)
    if len(L) < 2:
        return L
    lo, hi = L[0], L[-1]
    span = abs(hi - lo) or 1.0
    for x in xs:
        x = float(x)
        if not (lo + 1e-7 * span < x < hi - 1e-7 * span):
            continue
        k = int(np.argmin(np.abs(L - x)))
        if abs(L[k] - x) <= 1e-9 * span:
            continue
        step = float(np.diff(L)[min(max(k - (1 if L[k] > x else 0), 0), len(L) - 2)])
        if 0 < k < len(L) - 1 and abs(L[k] - x) < 0.5 * step:
            L[k] = x
        else:
            L = np.sort(np.append(L, x))
    return np.sort(L)


def _merge(x, tol):
    x = np.sort(np.asarray(x, float))
    out = [x[0]]
    for a in x[1:]:
        if a - out[-1] > tol:
            out.append(a)
    return np.array(out)


# -- the mesh ------------------------------------------------------------------

def build_bmesh(m, light=True, uvs=True):
    """bmesh of a Mesher's polygons: corner UVs and normals (layer "nproto_n"),
    the "brep_face" face layer; Light joins each curved face's triangles into
    quads, Triangles splits every face. Seams on B-rep edges and where the UVs
    jump; edges between faces bending by more than CREASE_DEG are sharp."""
    bm = bmesh.new()
    verts = [bm.verts.new(tuple(p)) for p in m.P]
    bm.verts.ensure_lookup_table()
    uvl = bm.loops.layers.uv.new("UVMap") if uvs else None
    nl = bm.loops.layers.float_vector.new("nproto_n")
    fl = bm.faces.layers.int.new("brep_face")
    groups = {"curved": {}, "holes": {}}
    for ids, cuv, cn, fi, kind in m.polys:
        keep = [j for j in range(len(ids)) if ids[j] != ids[j - 1]] if len(ids) > 1 else []
        if len(keep) < 3:
            continue
        try:
            bf = bm.faces.new([verts[ids[j]] for j in keep])
        except ValueError:
            continue
        bf[fl] = fi
        for loop, j in zip(bf.loops, keep):
            if uvl is not None:
                loop[uvl].uv = cuv[j]
            loop[nl] = cn[j]
        if kind in groups:
            groups[kind].setdefault(fi, []).append(bf)
    bm.normal_update()                  # join_triangles compares the faces' normals
    if light:
        for fs in groups["holes"].values():           # a flat face with holes: as few n-gons as Blender allows
            fs = set(fs)
            inner = [e for e in {e for f in fs for e in f.edges}
                     if len(e.link_faces) == 2 and all(f in fs for f in e.link_faces)]
            if inner:
                bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(1.0), use_dissolve_boundaries=False,
                                         verts=[], edges=inner, delimit=set())
        for fs in groups["curved"].values():
            fs = [f for f in fs if f.is_valid and len(f.verts) == 3]
            if len(fs) > 1:
                bmesh.ops.join_triangles(bm, faces=fs, cmp_seam=False, cmp_sharp=False, cmp_uvs=uvl is not None,
                                         cmp_vcols=False, cmp_materials=False,
                                         angle_face_threshold=math.radians(max(2.0 * math.degrees(m.ang), 20.0)),
                                         angle_shape_threshold=math.radians(60.0))
    else:
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
    cos_c = math.cos(math.radians(CREASE_DEG))
    bm.verts.index_update()
    for e in bm.edges:
        a, b = e.verts
        if (min(a.index, b.index), max(a.index, b.index)) in m.seams:
            e.seam = True
        if len(e.link_loops) != 2:
            continue
        l1, l2 = e.link_loops
        # the corners of each face at both ends of the edge
        c1 = {l1.vert: l1, l1.link_loop_next.vert: l1.link_loop_next}
        c2 = {l2.vert: l2, l2.link_loop_next.vert: l2.link_loop_next}
        sharp = jump = False
        for v in (a, b):
            x, y = c1.get(v), c2.get(v)
            if x is None or y is None:
                continue
            if x[nl].dot(y[nl]) < cos_c:
                sharp = True
            if uvl is not None and (x[uvl].uv - y[uvl].uv).length > 1e-6:
                jump = True
        if sharp:
            e.smooth = False
        if jump:
            e.seam = True
    return bm


def pack_islands(me, margin=0.02):
    """Lay the faces' UV islands (one per B-rep face) side by side in 0-1,
    largest first on shelves, keeping their proportions (for baking)."""
    uv = me.uv_layers.get("UVMap")
    att = me.attributes.get("brep_face")
    if uv is None or att is None or not len(me.polygons):
        return
    n = len(me.loops)
    UV = np.empty(n * 2)
    uv.data.foreach_get("uv", UV)
    UV = UV.reshape(-1, 2)
    fid = np.empty(len(me.polygons), np.int64)
    att.data.foreach_get("value", fid)
    ls = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_total", lt)
    lf = np.repeat(fid, lt)
    islands = []
    for f in np.unique(fid):
        sel = np.flatnonzero(lf == f)
        lo, hi = UV[sel].min(axis=0), UV[sel].max(axis=0)
        islands.append((sel, lo, hi - lo))
    islands.sort(key=lambda x: -x[2][1])
    total = sum(float(s[0] * s[1]) for _, _, s in islands) or 1.0
    width = max(math.sqrt(total) * 1.15, max(float(s[0]) for _, _, s in islands))
    gap = margin * math.sqrt(total)
    x = y = row_h = 0.0
    place = []
    for sel, lo, size in islands:
        if x > 0 and x + size[0] > width:
            x, y, row_h = 0.0, y + row_h + gap, 0.0
        place.append((sel, lo, np.array([x, y])))
        x += size[0] + gap
        row_h = max(row_h, float(size[1]))
    extent = max(width, y + row_h) or 1.0
    for sel, lo, at in place:
        UV[sel] = (UV[sel] - lo + at) / extent
    uv.data.foreach_set("uv", UV.ravel())


def mesh_from_brep(brep, name, light=True, detail="MEDIUM", chord=None, angle=None, uv_mode="REAL"):
    """A new mesh datablock (local space) and the Mesher's report."""
    c, a = DETAIL.get(detail, DETAIL["MEDIUM"])
    if detail == "CUSTOM":
        c, a = chord if chord is not None else c, angle if angle is not None else a
    m = Mesher(brep, c, a, uv_mode).run()
    bm = build_bmesh(m, light=light, uvs=uv_mode != "NONE")
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    att = me.attributes.get("nproto_n")
    N = np.empty(len(me.loops) * 3)
    if att is not None:
        att.data.foreach_get("vector", N)
        me.attributes.remove(att)
    me.shade_smooth()
    if len(me.loops):
        me.normals_split_custom_set([tuple(x) for x in N.reshape(-1, 3)])
    me.update()
    return me, m


def counts(me):
    sizes = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_total", sizes)
    return {"verts": len(me.vertices), "faces": len(me.polygons), "tris": int((sizes == 3).sum()),
            "quads": int((sizes == 4).sum()), "ngons": int((sizes > 4).sum()),
            "triangles": int((sizes - 2).sum())}


def _copy_modifiers(src, dst):
    for md in src.modifiers:
        try:
            new = dst.modifiers.new(md.name, md.type)
        except Exception:
            continue
        for p in md.bl_rna.properties:
            if p.is_readonly or p.identifier in ("name", "type"):
                continue
            try:
                setattr(new, p.identifier, getattr(md, p.identifier))
            except Exception:
                pass


class NPROTO_OT_brep_to_mesh(bpy.types.Operator):
    """Convert the selected NURBS objects to plain meshes for rendering, texturing
    and animation, meshed afresh at the chosen detail: n-gons on flat faces, quads
    on curved ones, UVs, sharp creases and seams. The NURBS objects are kept (hidden)"""
    bl_idname = "nproto.brep_to_mesh"
    bl_label = "BREP to Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    mesh_type: EnumProperty(
        name="Mesh", default="LIGHT",
        items=[("LIGHT", "Light", "N-gons on flat faces, quads on curved ones (triangles only where a trim "
                                  "meets the grid): the fewest faces"),
               ("TRIANGLES", "Triangles", "All triangles (game engines, glTF)")])
    detail: EnumProperty(
        name="Detail", default="MEDIUM",
        items=[("LOW", "Low", "Chord 0.4% of the size, 15 degrees"),
               ("MEDIUM", "Medium", "Chord 0.1% of the size, 7.5 degrees"),
               ("HIGH", "High", "Chord 0.025% of the size, 3 degrees"),
               ("CUSTOM", "Custom", "Your own chord and angle")])
    chord: FloatProperty(name="Chord", default=0.001, min=0.00001, max=0.05, precision=5, step=0.01,
                         description="Largest distance between mesh and surface, as a fraction of the object's size")
    angle: FloatProperty(name="Angle", default=7.5, min=0.5, max=45.0,
                         description="Largest turn between neighbouring mesh edges or grid lines (degrees)")
    uv_mode: EnumProperty(
        name="UVs", default="REAL",
        items=[("REAL", "Real Size", "UVs in metres from each face's (u, v): a texture keeps its scale "
                                     "from face to face"),
               ("NORMALIZED", "0-1 per Face", "Every face fills the UV square"),
               ("NONE", "None", "No UV map")])
    pack: BoolProperty(name="Pack Islands", default=False,
                       description="Lay the faces' UV islands side by side in 0-1 (for baking)")
    hide_original: BoolProperty(name="Hide NURBS Objects", default=True,
                                description="Hide the NURBS objects in the viewport and in renders "
                                            "(they stay, with their history)")
    join: BoolProperty(name="Join", default=False, description="One mesh object for all of them")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(store.is_nurbs(o) for o in context.selected_objects)

    def invoke(self, context, event):
        """The settings first (Mesh, Detail, UVs...), then Convert; they stay
        adjustable afterwards in the Adjust Last Operation panel."""
        self._dialog = True
        return context.window_manager.invoke_props_dialog(self, width=360, title="BREP to Mesh",
                                                          confirm_text="Convert")

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "mesh_type", expand=True)
        lay.prop(self, "detail")
        if self.detail == "CUSTOM":
            col = lay.column(align=True)
            col.prop(self, "chord")
            col.prop(self, "angle")
        lay.prop(self, "uv_mode")
        if self.uv_mode != "NONE":
            lay.prop(self, "pack")
        lay.prop(self, "hide_original")
        lay.prop(self, "join")
        c = LAST.get("counts")
        if c and not getattr(self, "_dialog", False):    # what it made (not shown before it runs)
            box = lay.box()
            box.label(text=f"{c['faces']:,} faces ({c['ngons']:,} n-gons, {c['quads']:,} quads, "
                           f"{c['tris']:,} triangles)", icon='MESH_DATA')
            box.label(text=f"{c['verts']:,} vertices, {c['triangles']:,} triangles when rendered")
            if LAST.get("display"):
                box.label(text=f"Display mesh: {LAST['display']:,} triangles")

    def execute(self, context):
        srcs = [o for o in context.selected_objects if store.is_nurbs(o)]
        made, total, display, problems = [], None, 0, []
        for src in srcs:
            try:
                brep = store.get_brep(src)
                me, m = mesh_from_brep(brep, src.name + " Mesh", light=self.mesh_type == "LIGHT",
                                       detail=self.detail, chord=self.chord, angle=self.angle,
                                       uv_mode=self.uv_mode)
            except Exception as exc:
                self.report({'WARNING'}, f"{src.name}: not converted ({exc})")
                continue
            problems += [f"{src.name} {p}" for p in m.problems]
            for mat in src.data.materials:
                me.materials.append(mat)
            if self.pack and self.uv_mode != "NONE":
                pack_islands(me)
            dst = bpy.data.objects.new(src.name + " Mesh", me)
            for coll in src.users_collection:
                coll.objects.link(dst)
            dst.parent = src.parent
            dst.matrix_parent_inverse = src.matrix_parent_inverse.copy()
            dst.matrix_basis = src.matrix_basis.copy()
            _copy_modifiers(src, dst)
            display += len(src.data.polygons)
            c = counts(me)
            total = c if total is None else {k: total[k] + c[k] for k in total}
            made.append((src, dst))
        if not made:
            self.report({'WARNING'}, "Nothing converted")
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for src, dst in made:
            if self.hide_original:
                src.hide_set(True)
                src.hide_render = True
            dst.select_set(True)
        context.view_layer.objects.active = made[0][1]
        if self.join and len(made) > 1:
            with context.temp_override(active_object=made[0][1], selected_editable_objects=[d for _, d in made],
                                       selected_objects=[d for _, d in made]):
                bpy.ops.object.join()
        LAST.update(counts=total, display=display)
        msg = f"BREP to Mesh: {total['faces']:,} faces ({total['ngons']:,} n-gons, {total['quads']:,} quads, " \
              f"{total['tris']:,} triangles), {total['verts']:,} vertices"
        if problems:
            self.report({'WARNING'}, msg + " -- " + "; ".join(problems[:3]))
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def menu_convert(self, context):
    self.layout.operator(NPROTO_OT_brep_to_mesh.bl_idname, text="Mesh (NURBS Proto)", icon='MESH_DATA')


classes = (NPROTO_OT_brep_to_mesh,)


def register():
    bpy.types.VIEW3D_MT_object_convert.append(menu_convert)


def unregister():
    bpy.types.VIEW3D_MT_object_convert.remove(menu_convert)
