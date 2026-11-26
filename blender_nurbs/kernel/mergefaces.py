"""Remove Edges: edges of a body taken out where the faces either side lie on
one surface, the faces merged into one -- the body stays closed and its shape
does not change (Rhino's MergeFace / MergeAllCoplanarFaces / MergeEdge,
Plasticity's Delete Redundant Topology).

When two faces are one (find):
  - both flat, on the same plane (within 1e-5 of the body's size), facing the
    same way (back to back is a fin, not one face);
  - both curved and on one surface: the same control points (pieces of one
    split surface), or every sample of one face lies on the other's surface
    (a cylinder cut in two) and they face the same way at the edge.
Anything else stays and says why: a crease, faces that only touch tangentially
(they would need a new, refitted surface), a flat face beside a curved one.

Merging (merge): the faces joined by removed edges are one cluster; two faces
of a cluster that share another edge lose that edge too (a face cannot keep an
edge inside it). Each cluster is trimmed again to the outline of the union,
with the edges' own curves (exact): a flat cluster on a new plane patch that
covers it, a curved one on the member surface every member lies on. Then the
vertices the removed edges leave behind -- where two edges now meet in a
straight or smooth line, the same two faces on either side -- are taken out
by joining the two edges into one curve (lines into one line, other curves
end to end, exact) and trimming both faces again.

Every step checks itself (each new face has the area of what it replaces, the
body stitches with no new open edges, a solid stays a solid of the same
volume); a step that does not check out is skipped and reported.

Kernel and numpy only (no bpy): the worker's merge_faces job runs merge(), the
add-on calls find() in Blender to tell picks that can go from those that can't.
"""

import math

import numpy as np

try:
    from .trimext import _cut, _edge_curve, brep_tol, flip
except ImportError:          # the worker imports the kernel folder's modules top-level
    from trimext import _cut, _edge_curve, brep_tol, flip

REL_TOL = 1e-5               # a share of the body's size: points this close are the same
SMOOTH_DEG = 1e-3            # tangents / normals this close count as the same direction
CREASE_DEG = 0.5             # a crease below this is reported as "tangent", above it as an angle
FLAT_HINT_DEG = 10.0         # two flat faces closer than this to one plane: the report names the Flat Tolerance


# -- small helpers ----------------------------------------------------------------

def _size(b):
    lo, hi = b.bounding_box()
    return float(np.linalg.norm(np.array(hi, float) - np.array(lo, float))) or 1.0


def _pt(p):
    return np.array([float(p[0]), float(p[1]), float(p[2])]) if not hasattr(p, "x") \
        else np.array([p.x, p.y, p.z], float)


def _edge_pts(e, n):
    t0, t1 = e.parameter_range()
    # the ends exactly (t0 + (t1 - t0) * 1.0 can land past t1, which an analytic line refuses)
    return [_pt(e.point_at(t0 if i == 0 else t1 if i == n - 1 else t0 + (t1 - t0) * i / (n - 1)))
            for i in range(n)]


def _edge_mid(e):
    t0, t1 = e.parameter_range()
    return _pt(e.point_at(0.5 * (t0 + t1)))


def _key(p, tol):
    return tuple(int(round(float(v) / tol)) for v in p)


def _index(faces, f):
    for k, g in enumerate(faces):
        if g == f:
            return k
    return -1


def _uv(sm, surf, p):
    _, uvs, d = sm.surface_closest_point(surf, tuple(float(x) for x in p))
    uv = uvs[0]
    return ((uv.x, uv.y) if hasattr(uv, "x") else (float(uv[0]), float(uv[1]))), float(d)


def _outward(sm, f, p):
    """Unit outward normal of face f near point p."""
    uv, _ = _uv(sm, f.surface(), p)
    n = np.array(f.outward_normal_at_uv(*uv), float)
    return n / (np.linalg.norm(n) or 1.0)


def _inner(f):
    try:
        return _pt(f.interior_point())
    except Exception:
        return None


def _samples(f, per_edge=5):
    pts = [p for e in f.edges() for p in _edge_pts(e, per_edge)]
    q = _inner(f)
    return pts + ([q] if q is not None else [])


def _cv_key(surf):
    from_ext = surf_cvs(surf)
    return (from_ext["degree_u"], from_ext["degree_v"], from_ext["count_u"], from_ext["count_v"],
            tuple(round(c, 9) for p in from_ext["points"] for c in p))


_EXT = {}


def surf_cvs(surf):
    return _EXT["ext"].surface_cvs(surf)


def _plane(sm, f, tol):
    """(point, unit outward normal) when face f is flat (its control points on
    one plane), else None."""
    d = surf_cvs(f.surface())
    P = np.array(d["points"], float)
    c = P.mean(axis=0)
    if len(P) < 3:
        return None
    _, s, vt = np.linalg.svd(P - c)
    n = vt[2]
    if np.abs((P - c) @ n).max() > tol:
        return None
    q = _inner(f)
    if q is not None:
        on = _outward(sm, f, q)
        if on @ n < 0:
            n = -n
    return c, n


class _Body:
    """Per-face facts of a brep, computed once."""

    def __init__(self, sm, b):
        _EXT["ext"] = sm.blender_ext
        self.sm, self.b = sm, b
        self.size = _size(b)
        self.tol = REL_TOL * self.size
        self.faces = b.faces()
        self._plane, self._key, self._samples = {}, {}, {}

    def plane(self, i):
        if i not in self._plane:
            try:
                self._plane[i] = _plane(self.sm, self.faces[i], self.tol)
            except Exception:
                self._plane[i] = None
        return self._plane[i]

    def cv_key(self, i):
        if i not in self._key:
            try:
                self._key[i] = _cv_key(self.faces[i].surface())
            except Exception:
                self._key[i] = None
        return self._key[i]

    def samples(self, i):
        if i not in self._samples:
            self._samples[i] = _samples(self.faces[i])
        return self._samples[i]

    def lies_on(self, i, j):
        """Does face i lie on face j's (untrimmed) surface?"""
        return self.lies_on_surface(i, self.faces[j].surface())

    def lies_on_surface(self, i, surf):
        try:
            return all(_uv(self.sm, surf, p)[1] <= 10 * self.tol for p in self.samples(i))
        except Exception:
            return False

    def edge_faces(self, e):
        """Indices of the two different faces on edge e, or None (an open edge,
        the seam of a closed face, a non-manifold edge)."""
        ids = []
        for g in e.faces():
            k = _index(self.faces, g)
            if k < 0:
                return None
            if k not in ids:
                ids.append(k)
        return ids if len(ids) == 2 else None

    def why_not(self, e):
        """'' when edge e can go (its two faces are one), else why not."""
        ij = self.edge_faces(e)
        if ij is None:
            return "an open or seam edge (no second face)"
        i, j = ij
        m = _edge_mid(e)
        try:
            ni, nj = _outward(self.sm, self.faces[i], m), _outward(self.sm, self.faces[j], m)
        except Exception:
            return "the faces could not be evaluated"
        cos = float(ni @ nj)
        same_dir = cos >= math.cos(math.radians(SMOOTH_DEG))
        pi, pj = self.plane(i), self.plane(j)
        if pi is not None and pj is not None:
            if same_dir and abs(float((pj[0] - pi[0]) @ pi[1])) <= 10 * self.tol:
                return ""
            if cos <= -math.cos(math.radians(SMOOTH_DEG)):
                return "the faces meet back to back (a fin)"
            why = self._crease(cos)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            if ang <= FLAT_HINT_DEG:
                why += f" (a Flat Tolerance of {math.ceil(ang * 100 - 1e-6) / 100:.3g}° or more merges them)"
            return why
        if not same_dir:
            why = self._crease(cos)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            if ang <= _surfjoin().TANGENT_DEG:
                why += " (a Fit Tolerance refits them onto one surface)"
            return why
        if (pi is None) != (pj is None):
            return "a flat face meets a curved one (tangent: a Fit Tolerance refits them onto one surface)"
        ki, kj = self.cv_key(i), self.cv_key(j)
        if (ki is not None and ki == kj) or self.lies_on(j, i) or self.lies_on(i, j):
            return ""
        try:
            if join_surfaces(self.sm.blender_ext, self.faces[i].surface(), self.faces[j].surface(),
                             self.tol) is not None:
                return ""
        except Exception:
            pass
        return "the faces are tangent but lie on different surfaces (a Fit Tolerance refits them onto one surface)"

    @staticmethod
    def _crease(cos):
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        if ang < CREASE_DEG:
            return f"the faces meet at {ang:.3g}° (not one surface)"
        return f"a crease of {ang:.3g}°"


# -- one surface out of two that continue each other ---------------------------------

def _grid(ext, surf):
    """(homogeneous CVs (nu, nv, 4), knots u, knots v, degree u, degree v)."""
    d = ext.surface_cvs(surf)
    nu, nv = int(d["count_u"]), int(d["count_v"])
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    w = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    H = np.concatenate([P * w[..., None], w[..., None]], axis=2)
    return H, [float(k) for k in d["knots_u"]], [float(k) for k in d["knots_v"]], \
        int(d["degree_u"]), int(d["degree_v"])


def _flip_knots(K):
    a, b = K[0], K[-1]
    return [a + b - k for k in reversed(K)]


def _variants(g):
    """The grid g with its directions reversed and / or swapped (8 ways)."""
    H, Ku, Kv, pu, pv = g
    for t in (False, True):
        G = (np.transpose(H, (1, 0, 2)), Kv, Ku, pv, pu) if t else (H, Ku, Kv, pu, pv)
        for fu in (False, True):
            for fv in (False, True):
                h, ku, kv = G[0], G[1], G[2]
                if fu:
                    h, ku = h[::-1], _flip_knots(ku)
                if fv:
                    h, kv = h[:, ::-1], _flip_knots(kv)
                yield (h, ku, kv, G[3], G[4]), t


def _same_knots(A, B):
    if len(A) != len(B):
        return False
    a0, a1, b0, b1 = A[0], A[-1], B[0], B[-1]
    if a1 - a0 <= 0 or b1 - b0 <= 0:
        return False
    return all(abs((x - a0) / (a1 - a0) - (y - b0) / (b1 - b0)) <= 1e-9 for x, y in zip(A, B))


def _cat_u(A, B, tol):
    """A then B along u, when A's last u-row is B's first (same points, same
    v knots and degrees): the joined grid (exact, a full knot at the joint)."""
    HA, KuA, KvA, pu, pv = A
    HB, KuB, KvB, qu, qv = B
    if pu != qu or pv != qv or HA.shape[1] != HB.shape[1] or not _same_knots(KvA, KvB):
        return None
    ra, rb = HA[-1], HB[0]
    if np.abs(ra[:, :3] / ra[:, 3:] - rb[:, :3] / rb[:, 3:]).max() > tol:
        return None
    s = ra[:, 3] / rb[:, 3]
    if np.abs(s - s[0]).max() > 1e-9 * abs(s[0]):
        return None                                   # weights not in proportion: not the same row
    HB = HB * s[0]
    H = np.concatenate([HA, HB[1:]], axis=0)
    t = KuA[-1]
    U = KuA[:-(pu + 1)] + [t] * pu + [x - KuB[0] + t for x in KuB[pu + 1:-(pu + 1)]] + \
        [KuB[-1] - KuB[0] + t] * (pu + 1)
    if len(U) != H.shape[0] + pu + 1:
        return None
    return H, U, KvA, pu, pv


def join_surfaces(ext, sa, sb, tol):
    """One surface for sa and sb where sb continues sa exactly across a whole
    boundary (a cylinder in two stacked pieces), in sa's own parameter
    directions; None if they do not."""
    A = _grid(ext, sa)
    B = _grid(ext, sb)
    for tA in (False, True):                          # join along A's u, or (swapped) along its v
        GA = (np.transpose(A[0], (1, 0, 2)), A[2], A[1], A[4], A[3]) if tA else A
        for GB, _ in _variants(B):
            for first, second, b_first in ((GA, GB, False), (GB, GA, True)):
                r = _cat_u(first, second, tol)
                if r is None:
                    continue
                H, U, V, pu, pv = r
                if tA:
                    H, U, V, pu, pv = np.transpose(H, (1, 0, 2)), V, U, pv, pu
                w = H[..., 2 + 1]
                P = H[..., :3] / w[..., None]
                rational = np.abs(w - 1.0).max() > 1e-12
                return ext.surface_from_cvs(pu, pv, [tuple(map(float, p)) for p in P.reshape(-1, 3)],
                                            [float(x) for x in U], [float(x) for x in V],
                                            [float(x) for x in w.reshape(-1)] if rational else None)
    return None


def has_kernel(sm):
    ext = getattr(sm, "blender_ext", None)
    return ext is not None and all(hasattr(ext, n) for n in ("trim_surface_with_3d_curves_tol", "append_brep",
                                                             "delete_faces", "surface_cvs", "curve_cvs"))


def _picked_edges(b, edges, pts, size):
    """{edge index: [pick indices]} of the edge nearest each picked point
    (None for a point no edge is near)."""
    out, miss = {}, []
    for k, p in enumerate(pts):
        q = tuple(float(x) for x in p)
        ds = [e.closest_point(q)[2] for e in edges]
        i = int(np.argmin(ds)) if ds else -1
        if i < 0 or ds[i] > 1e-2 * size:
            miss.append(k)
            continue
        out.setdefault(i, []).append(k)
    return out, miss


def find(sm, b, picked=None, flat_deg=0.0, fit_tol=0.0):
    """The edges of b that can go: {"edges": [{"edge": i, "point": mid,
    "ok": bool, "why": str, "near": angle or None, "fit": angle or None}]} --
    every edge between two faces, or (picked: points on edges) the edges
    nearest those points; "near": the angle between two flat faces within
    flat_deg of one plane (they can go by a refit, planerefit.py); "fit" (with
    fit_tol > 0): the angle between two tangent faces on different surfaces
    (they can go by a surface refit, surfjoin.py); "missed": picks near no
    edge."""
    body = _Body(sm, b)
    edges = b.edges()
    if picked is None:
        idx, miss = list(range(len(edges))), []
    else:
        hit, miss = _picked_edges(b, edges, picked, body.size)
        idx = sorted(hit)
    out = []
    for i in idx:
        e = edges[i]
        if picked is None and body.edge_faces(e) is None:
            continue
        why = body.why_not(e)
        near = None
        if why and flat_deg > 0:
            from_mod = _planerefit()
            near = from_mod.near_coplanar(body, e, flat_deg)
        fit = None
        if why and near is None and fit_tol > 0:
            sj = _surfjoin()
            fit = sj.tangent_angle(body, e)
            if fit is not None and picked is None and sj.curvature_jump(body, e) > sj.G2_REL:
                fit = None             # all edges: only curvature-continuous joints (never a fillet's edges)
        out.append({"edge": i, "point": [float(x) for x in _edge_mid(e)], "ok": not why, "why": why,
                    "near": near, "fit": fit})
    return {"edges": out, "missed": miss}


def _planerefit():
    try:
        from . import planerefit
    except ImportError:          # the worker imports the kernel folder's modules top-level
        import planerefit
    return planerefit


def _surfjoin():
    try:
        from . import surfjoin
    except ImportError:
        import surfjoin
    return surfjoin


# -- re-trimming --------------------------------------------------------------------

def _new_plane(ext, n, pts, tol):
    """A bilinear plane patch (normal n) covering the points, a little bigger."""
    pts = np.asarray(pts, float)
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, seed)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)                 # u x v = n: the patch faces along n
    o = pts.mean(axis=0)
    x, y = (pts - o) @ u, (pts - o) @ v
    pad = 0.02 * max(x.max() - x.min(), y.max() - y.min()) + 10 * tol
    x0, x1, y0, y1 = x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad
    C = [o + x0 * u + y0 * v, o + x0 * u + y1 * v, o + x1 * u + y0 * v, o + x1 * u + y1 * v]
    return ext.surface_from_cvs(1, 1, [tuple(map(float, c)) for c in C], [0.0, 0.0, 1.0, 1.0],
                                [0.0, 0.0, 1.0, 1.0], None)


def _copy_surface(ext, surf):
    d = ext.surface_cvs(surf)
    if "knots_u" not in d:
        return surf
    return ext.surface_from_cvs(d["degree_u"], d["degree_v"], list(d["points"]), list(d["knots_u"]),
                                list(d["knots_v"]), list(d["weights"]) if d.get("weights") else None)


def _span(curves):
    P = np.array([c.evaluate(t) for c in curves for t in c.parameter_range()], float) \
        if hasattr(curves[0], "evaluate") else np.zeros((1, 3))
    return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))


def _curve_ends(c):
    t0, t1 = c.parameter_range()
    return _pt(c.evaluate(t0)), _pt(c.evaluate(t1))


def _trim(sm, surf, loops, area, tol, btol):
    """surf trimmed by loops (lists of curves; the widest is taken as the
    outer one), or None unless it is one face of the given area (None: any
    area above zero)."""
    ext = sm.blender_ext
    loops = sorted(loops, key=_span, reverse=True)
    sizes = [len(lp) for lp in loops]
    curves = [c for lp in loops for c in lp]
    for tight in (True, False):
        try:
            if tight:
                cand = ext.trim_surface_with_3d_curves_tol(surf, sizes, curves, tol, btol)
            else:
                cand = sm.trim_surface_with_3d_curves(surf, sizes, curves)
        except Exception:
            continue
        if cand.face_count() != 1:
            continue
        a = cand.area(1e-6)
        if (area is None and a > 0) or (area is not None and abs(a - area) <= 1e-4 * max(area, 1e-12)):
            return cand
    return None


def _open_edges(b):
    """Open (one-face) edges, not counting a face's own seam."""
    n = 0
    for e in b.edges():
        if len(e.faces()) == 1:
            f = e.faces()[0]
            uses = sum(1 for loop in f.boundary_loops() for g, _ in loop if g == e)
            if uses < 2:
                n += 1
    return n


def _volume(b):
    try:
        return abs(float(b.volume(1e-6)))
    except Exception:
        return 0.0


def _replace(sm, b, drop, new_faces, tol, expect_faces, volume_change=0.0):
    """b without faces `drop` (indices), with new_faces appended, stitched; None
    unless it checks out (no new open edges, a solid stays one, the same
    volume -- or within volume_change of it, when faces moved)."""
    ext = sm.blender_ext
    out = b.copy()
    of = out.faces()
    ext.delete_faces(out, [of[i] for i in drop])
    for m in new_faces:
        ext.append_brep(out, m)
    was_solid = bool(b.is_manifold_solid())
    open0, v0 = _open_edges(b), _volume(b)
    for t in (tol / 10, tol, 10 * tol):
        try:
            st = sm.stitch_advanced(out.copy(), t, squeeze_small_edges=False,
                                    making_manifold_solid=was_solid)[0]
        except Exception:
            continue
        if st.face_count() != expect_faces or _open_edges(st) > open0:
            continue
        if was_solid:
            if not st.is_manifold_solid():
                continue
            v1 = _volume(st)
            if abs(v1 - v0) > 1e-6 * max(v0, 1e-12) + 1e-9 + volume_change:
                continue
        return st
    return None


# -- removing edges -----------------------------------------------------------------

def _clusters(body, go):
    """Faces joined by the edges in `go` (indices into b.edges()): union-find
    over faces; then every edge between two faces of one cluster goes too.
    Returns (clusters: [[face index]], gone: set of edge indices)."""
    edges = body.b.edges()
    nf = len(body.faces)
    parent = list(range(nf))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for k in go:
        ij = body.edge_faces(edges[k])
        if ij:
            parent[find(ij[0])] = find(ij[1])
    groups = {}
    for i in range(nf):
        groups.setdefault(find(i), []).append(i)
    clusters = [g for g in groups.values() if len(g) > 1]
    member = {i: ci for ci, g in enumerate(clusters) for i in g}
    gone = set()
    for k, e in enumerate(edges):
        ij = body.edge_faces(e)
        if ij and ij[0] in member and member.get(ij[1]) == member[ij[0]]:
            gone.add(k)
    return clusters, gone


def _carrier(body, members):
    """The surface the merged face is trimmed on: a new plane patch for a flat
    cluster, else the member surface every member lies on (None: none)."""
    ext = body.sm.blender_ext
    p = body.plane(members[0])
    if p is not None:
        pts = [q for i in members for q in body.samples(i)]
        return _new_plane(ext, p[1], pts, body.tol)
    keys = {body.cv_key(i) for i in members}
    if len(keys) == 1 and None not in keys:
        return _copy_surface(ext, body.faces[members[0]].surface())
    for j in members:
        if all(i == j or body.lies_on(i, j) for i in members):
            return _copy_surface(ext, body.faces[j].surface())
    # surfaces that continue each other: joined one by one into one surface
    surf, todo = _copy_surface(ext, body.faces[members[0]].surface()), list(members[1:])
    while todo:
        left = [i for i in todo if not body.lies_on_surface(i, surf)]
        if not left:
            return surf
        for i in left:
            try:
                s = join_surfaces(ext, surf, body.faces[i].surface(), body.tol)
            except Exception:
                s = None
            if s is not None:
                surf = s
                break
        else:
            return None
        todo = left
    return surf


def _chain_runs(runs, segs, tol):
    """Closed loops from runs of oriented edge pieces (segs: (edge, forward,
    start, end, start direction, end direction)), joined by position. Where
    more than one run starts at a point (the two sides of a closed surface's
    seam meet there), the one going on in the same direction is taken, not
    the one turning back."""
    rs, out = [list(r) for r in runs], []
    while rs:
        chain = rs.pop(0)
        start = segs[chain[0]][2]
        guard = 0
        while True:
            guard += 1
            end, d_in = segs[chain[-1]][3], segs[chain[-1]][5]
            near = [r for r in range(len(rs)) if np.linalg.norm(segs[rs[r][0]][2] - end) <= 100 * tol]
            closed = np.linalg.norm(end - start) <= 100 * tol
            if closed and (not near or float(segs[chain[0]][4] @ d_in) >= max(
                    float(segs[rs[r][0]][4] @ d_in) for r in near) - 1e-9):
                break
            if not near or guard > 10000:
                return None
            nxt = max(near, key=lambda r: float(segs[rs[r][0]][4] @ d_in))
            chain += rs.pop(nxt)
        out.append(chain)
    return out


def _seg(e, fw):
    """(edge, forward, start, end, unit direction leaving the start, unit
    direction arriving at the end) in loop order."""
    t0, t1 = e.parameter_range()
    a, z = _pt(e.point_at(t0)), _pt(e.point_at(t1))
    da, dz = np.array(e.tangent_at(t0), float), np.array(e.tangent_at(t1), float)
    da, dz = da / (np.linalg.norm(da) or 1.0), dz / (np.linalg.norm(dz) or 1.0)
    return (e, fw, a, z, da, dz) if fw else (e, fw, z, a, -dz, -da)


def _ecurve(e, fw):
    """The edge's curve over its range, in loop order (a line also when the
    kernel holds it as an analytic line)."""
    try:
        return _edge_curve(e, fw)
    except ValueError:
        t0, t1 = e.parameter_range()
        c = e.curve()
        if type(c).__name__ != "Line":
            raise
        a, z = _pt(e.point_at(t0)), _pt(e.point_at(t1))
        if not fw:
            a, z = z, a
        return _EXT["ext"].curve_from_cvs(1, [tuple(map(float, a)), tuple(map(float, z))],
                                          [0.0, 0.0, 1.0, 1.0], None)


def _merged_face(body, members, gone_keys):
    """The cluster's faces as one trimmed face, or None."""
    sm, tol = body.sm, body.tol
    btol = brep_tol(REL_TOL, body.size)
    surf = _carrier(body, members)
    if surf is None:
        return None
    segs, loops, runs = [], [], []
    for i in members:
        for loop in body.faces[i].boundary_loops():
            ids = []
            for e, fw in loop:
                if _key(_edge_mid(e), tol) in gone_keys:
                    ids.append(None)
                    continue
                segs.append(_seg(e, fw))
                ids.append(len(segs) - 1)
            if None not in ids:
                loops.append(ids)
                continue
            if all(x is None for x in ids):
                continue
            k = ids.index(None)
            cur = []
            for j in ids[k + 1:] + ids[:k + 1]:
                if j is None:
                    if cur:
                        runs.append(cur)
                    cur = []
                else:
                    cur.append(j)
            if cur:
                runs.append(cur)
    area = sum(body.faces[i].area(1e-6) for i in members)
    chained = _chain_runs(runs, segs, tol)
    made = None
    if chained is not None and (loops or chained):
        curve_loops = [[_ecurve(segs[j][0], segs[j][1]) for j in lp] for lp in loops + chained]
        made = _trim(sm, surf, curve_loops, area, tol, btol)
    if made is None and _only_seams(segs, tol):
        # nothing left but the surface's own seam (both ways) and poles: the whole surface
        try:
            cand = sm.blender_ext.brep_from_surface(surf)
            if cand.face_count() == 1 and abs(cand.area(1e-6) - area) <= 1e-4 * max(area, 1e-12):
                made = cand
        except Exception:
            pass
    return made


def _only_seams(segs, tol):
    """Is every remaining piece a closed surface's seam (run both ways) or a
    pole (no length)?"""
    count = {}
    for s in segs:
        if np.linalg.norm(s[3] - s[2]) <= tol and s[0].length() <= tol:
            continue
        k = _key(_edge_mid(s[0]), tol)
        count.setdefault(k, set()).add(bool(s[1]))
    return all(v == {True, False} for v in count.values())


def remove_edges(sm, b, go):
    """b with the edges `go` (indices into b.edges(), each between two faces on
    one surface) removed: (new brep, report) or (None, report)."""
    body = _Body(sm, b)
    edges = b.edges()
    clusters, gone = _clusters(body, go)
    rep = {"removed": 0, "failed": [], "ends": []}
    if not clusters:
        return None, rep
    gone_keys = {_key(_edge_mid(edges[k]), body.tol) for k in gone}
    new, drop, done_edges = [], [], set()
    for members in clusters:
        mf = _merged_face(body, members, gone_keys)
        mine = {k for k in gone if body.edge_faces(edges[k])[0] in members}
        if mf is None:
            rep["failed"].append({"point": [float(x) for x in _edge_mid(edges[min(mine)])],
                                  "why": "the merged face could not be trimmed"})
            continue
        new.append(mf)
        drop += members
        done_edges |= mine
    if not new:
        return None, rep
    out = _replace(sm, b, drop, new, body.tol, len(body.faces) - len(drop) + len(new))
    if out is None:
        rep["failed"].append({"point": None, "why": "the merged faces did not join up with the body"})
        return None, rep
    rep["removed"] = len(done_edges)
    for k in done_edges:            # the removed edges' ends: vertices that may be left redundant
        t0, t1 = edges[k].parameter_range()
        rep["ends"] += [[float(x) for x in _pt(edges[k].point_at(t))] for t in (t0, t1)]
    return out, rep


# -- removing redundant vertices ----------------------------------------------------

def _leaving(e, at_start):
    t0, t1 = e.parameter_range()
    d = np.array(e.tangent_at(t0 if at_start else t1), float)
    d /= (np.linalg.norm(d) or 1.0)
    return d if at_start else -d


def _redundant(body, only=None):
    """Vertices where exactly two edges meet in a straight or smooth line with
    the same faces either side: [(vertex point, edge index a, edge index b)].
    only: points; just the vertices at them."""
    edges = body.b.edges()
    tol = body.tol
    eidx = {_key(_edge_mid(e), tol): k for k, e in enumerate(edges)}
    only_keys = None if only is None else {_key(p, 100 * tol) for p in only}
    out = []
    for v in body.b.vertices():
        p = _pt(v.point())
        if only_keys is not None and _key(p, 100 * tol) not in only_keys:
            continue
        es = v.edges()
        if len(es) != 2:
            continue
        ks = [eidx.get(_key(_edge_mid(e), tol)) for e in es]
        if None in ks or ks[0] == ks[1]:
            continue
        fa = sorted(_index(body.faces, g) for g in es[0].faces())
        fb = sorted(_index(body.faces, g) for g in es[1].faces())
        if fa != fb or -1 in fa:
            continue
        dirs = []
        for e in es:
            a, z = _edge_pts(e, 2)
            if np.linalg.norm(a - z) <= tol:
                break                                  # a closed edge
            dirs.append(_leaving(e, np.linalg.norm(a - p) <= np.linalg.norm(z - p)))
        if len(dirs) != 2 or float(dirs[0] @ dirs[1]) > -math.cos(math.radians(SMOOTH_DEG)):
            continue
        out.append((p, ks[0], ks[1]))
    return out


def _hom(c):
    """(homogeneous CVs (n, 4), knots, degree) of a curve."""
    d = _EXT["ext"].curve_cvs(c)
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    return np.concatenate([P * w[:, None], w[:, None]], axis=1), [float(k) for k in d["knots"]], int(d["degree"])


def _as_degree(Q, U, p, q):
    """A straight degree-1 piece re-expressed at degree q (exact); None for
    other degree changes."""
    if p == q:
        return Q, U
    if p != 1 or len(Q) != 2:
        return None
    a, z = Q[0], Q[1]
    if abs(a[3] - 1) > 1e-12 or abs(z[3] - 1) > 1e-12:
        return None
    t = np.linspace(0.0, 1.0, q + 1)
    Qn = np.array([a + (z - a) * s for s in t])
    return Qn, [U[0]] * (q + 1) + [U[-1]] * (q + 1)


def _joined(pieces, tol):
    """One curve through the pieces (curves, in order, end to end): a line
    when they are all straight, else the pieces end to end (exact; a knot
    of full multiplicity where they meet). None when their degrees differ
    beyond straight lines."""
    ext = _EXT["ext"]
    homs = [_hom(c) for c in pieces]
    pts = np.vstack([Q[:, :3] / Q[:, 3:] for Q, _, _ in homs])
    a, z = _curve_ends(pieces[0])[0], _curve_ends(pieces[-1])[1]
    if np.linalg.norm(z - a) > tol:
        d = (z - a) / np.linalg.norm(z - a)
        off = pts - a
        if np.abs(off - np.outer(off @ d, d)).max() <= tol and all(p == 1 for _, _, p in homs):
            return ext.curve_from_cvs(1, [tuple(map(float, a)), tuple(map(float, z))], [0.0, 0.0, 1.0, 1.0], None)
    q = max(p for _, _, p in homs)
    Qs, Us = [], []
    for Q, U, p in homs:
        r = _as_degree(Q, U, p, q)
        if r is None:
            return None
        Qs.append(np.asarray(r[0], float))
        Us.append(list(r[1]))
    Qall = Qs[0]
    for Q in Qs[1:]:
        Q = Q * (Qall[-1][3] / Q[0][3])              # weights scaled to meet (the curve does not change)
        Qall = np.vstack([Qall, Q[1:]])
    # knots: each piece's inner knots shifted along, q times at each joint
    U = [Us[0][0]] * (q + 1)
    t = Us[0][0]
    for k, Uk in enumerate(Us):
        inner = Uk[q + 1:-(q + 1)]
        base = t - Uk[0]
        U += [x + base for x in inner]
        t = t + (Uk[-1] - Uk[0])
        U += [t] * (q if k < len(Us) - 1 else q + 1)
    if len(U) != len(Qall) + q + 1:
        return None
    w = Qall[:, 3]
    P = [tuple(map(float, c[:3] / c[3])) for c in Qall]
    rational = np.abs(w - 1.0).max() > 1e-12
    return ext.curve_from_cvs(q, P, [float(x) for x in U], [float(x) for x in w] if rational else None)


def _edge_chains(body, red):
    """Chains of edge indices joined at redundant vertices: [(indices in
    order, closed?)]."""
    edges = body.b.edges()
    tol = body.tol
    at = {}                                         # vertex key -> [edge indices]
    for p, ka, kb in red:
        at[_key(p, tol)] = [ka, kb]
    nbr = {}
    for (ka, kb) in at.values():
        nbr.setdefault(ka, []).append(kb)
        nbr.setdefault(kb, []).append(ka)
    seen, out = set(), []
    for k0 in nbr:
        if k0 in seen:
            continue
        # walk to one end of the chain
        prev, cur = None, k0
        while True:
            nx = [k for k in nbr.get(cur, []) if k != prev]
            if not nx or nx[0] == k0:
                break
            prev, cur = cur, nx[0]
        start = cur
        order, prev, cur = [start], None, start
        while True:
            nx = [k for k in nbr.get(cur, []) if k != prev and k not in order]
            if not nx:
                break
            prev, cur = cur, nx[0]
            order.append(cur)
        closed = len(order) > 2 and order[0] in nbr.get(order[-1], [])
        seen |= set(order)
        out.append((order, closed))
    del edges
    return out


def remove_vertices(sm, b, only=None):
    """b with its redundant vertices (or those at `only`) taken out: each run
    of edges through them joined into one curve and the faces either side
    trimmed again. (new brep, number removed) or (None, 0)."""
    body = _Body(sm, b)
    red = _redundant(body, only)
    if not red:
        return None, 0
    if not _all_bsplines(sm, b):
        # (a stitch can leave an edge on a kernel curve that is no B-spline)
        b = sm.turn_to_nurbs(b.copy())
        body = _Body(sm, b)
        red = _redundant(body, only)
        if not red:
            return None, 0
    edges = b.edges()
    tol = body.tol
    btol = brep_tol(REL_TOL, body.size)
    chains = _edge_chains(body, red)
    ekey = [_key(_edge_mid(e), tol) for e in edges]
    of_key = {}                                  # edge key -> (chain id, position)
    joined = {}
    for ci, (order, closed) in enumerate(chains):
        # orient the chain: each edge forward or backward so they run end to end
        pieces, ok = [], True
        cur_end = None
        orient = []
        for n_, k in enumerate(order):
            a, z = _edge_pts(edges[k], 2)
            if cur_end is None:
                if len(order) > 1:
                    b2 = _edge_pts(edges[order[1]], 2)
                    fw = min(np.linalg.norm(z - b2[0]), np.linalg.norm(z - b2[1])) <= \
                        min(np.linalg.norm(a - b2[0]), np.linalg.norm(a - b2[1]))
                else:
                    fw = True
            else:
                fw = np.linalg.norm(a - cur_end) <= np.linalg.norm(z - cur_end)
            orient.append(fw)
            cur_end = z if fw else a
            pieces.append(_ecurve(edges[k], fw))
        try:
            c = _joined(pieces, tol)
        except Exception:
            c = None
        if c is None:
            ok = False
        if not ok:
            continue
        joined[ci] = (c, order, orient)
        for pos, k in enumerate(order):
            of_key[ekey[k]] = (ci, pos)
    if not joined:
        return None, 0
    ext = sm.blender_ext
    touched = sorted({i for ci in joined for k in joined[ci][1] for i in body.edge_faces(edges[k]) or
                      [_index(body.faces, g) for g in edges[k].faces()]})
    new = []
    for i in touched:
        f = body.faces[i]
        loops = []
        for loop in f.boundary_loops():
            items = [(e, fw, of_key.get(_key(_edge_mid(e), tol))) for e, fw in loop]
            n = len(items)
            # start where a chain begins (or at an edge in none)
            s0 = 0
            for s in range(n):
                cur, prv = items[s][2], items[s - 1][2]
                if cur is None or prv is None or cur[0] != prv[0]:
                    s0 = s
                    break
            items = items[s0:] + items[:s0]
            curves, prev = [], None
            for e, fw, ch in items:
                if ch is None:
                    curves.append(_ecurve(e, fw))
                    prev = None
                    continue
                ci = ch[0]
                if prev == ci:
                    continue                       # the rest of a run already given as one curve
                prev = ci                          # (a seam is run twice: once each way)
                c, order, orient = joined[ci]
                # the loop runs the chain forward when this edge runs as the chain orients it
                along = (fw == orient[ch[1]])
                curves.append(c if along else _reversed(ext, c))
            loops.append(curves)
        surf = _copy_surface(ext, f.surface())
        made = _trim(sm, surf, loops, f.area(1e-6), tol, btol)
        if made is None:
            return None, 0
        new.append(made)
    out = _replace(sm, b, touched, new, tol, len(body.faces))
    if out is None:
        return None, 0
    return out, len(red)


def _all_bsplines(sm, b):
    ext = sm.blender_ext
    for e in b.edges():
        try:
            ext.curve_cvs(e.curve())
        except ValueError:
            if type(e.curve()).__name__ != "Line":
                return False
    return True


def _reversed(ext, c):
    d = ext.curve_cvs(c)
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    Q = np.concatenate([P * w[:, None], w[:, None]], axis=1)
    U = [float(k) for k in d["knots"]]
    Qr, Ur = flip(Q, U)
    Qr = np.asarray(Qr, float)
    wr = Qr[:, 3]
    rational = np.abs(wr - 1.0).max() > 1e-12
    return ext.curve_from_cvs(int(d["degree"]), [tuple(map(float, q[:3] / q[3])) for q in Qr],
                              [float(x) for x in Ur], [float(x) for x in wr] if rational else None)


# -- the whole operation ------------------------------------------------------------

def merge(sm, b, picked=None, vertices=True, flat_deg=0.0, fit_tol=0.0):
    """Remove edges of b: those nearest the picked points (world/brep space),
    or every edge that can go (picked None). With flat_deg > 0 also edges
    between flat faces within that angle of one plane: those faces are
    refitted onto one plane (planerefit.py; the shape moves a little); with
    fit_tol > 0 edges between tangent faces on different surfaces, the faces
    refitted onto one NURBS surface within fit_tol (surfjoin.py). Then
    the vertices they leave behind (vertices: also every other redundant
    vertex when nothing was picked). Returns (brep or None, report): report
    has "removed" (edges), "refitted" (edges removed by a refit), "moved"
    (largest corner move of a refit), "fitted" (edges removed by a surface
    refit), "deviation" (its largest), "vertices" (removed), "refused" [{point,
    why}], "missed" (picks near no edge), "failed" [{point, why}]."""
    if not _all_bsplines(sm, b):
        b = sm.turn_to_nurbs(b.copy())
    found = find(sm, b, picked, flat_deg, fit_tol)
    go = [x["edge"] for x in found["edges"] if x["ok"]]
    near = [x for x in found["edges"] if not x["ok"] and x.get("near") is not None]
    fits = [x for x in found["edges"] if not x["ok"] and x.get("fit") is not None]
    rep = {"removed": 0, "refitted": 0, "moved": 0.0, "fitted": 0, "deviation": 0.0, "vertices": 0,
           "missed": found["missed"], "failed": [],
           "refused": [] if picked is None else
           [{"point": x["point"], "why": x["why"]} for x in found["edges"]
            if not x["ok"] and x.get("near") is None and x.get("fit") is None]}
    out = None
    ends = None
    if go:
        out, r = remove_edges(sm, b, go)
        rep["removed"], rep["failed"] = r["removed"], r["failed"]
        ends = r["ends"]
    if near:
        base = out if out is not None else b
        r_out, r = _planerefit().refit(sm, base, [x["point"] for x in near], flat_deg)
        if r_out is not None:
            out = r_out
            rep["removed"] += r["removed"]
            rep["refitted"], rep["moved"] = r["removed"], r["moved"]
            ends = None if picked is None else (ends or []) + r["corners"]
        if picked is not None:
            rep["refused"] += [{"point": f["point"], "why": "refit: " + f["why"]} for f in r["failed"]]
    if fits:
        base = out if out is not None else b
        r_out, r = _surfjoin().refit(sm, base, [x["point"] for x in fits], fit_tol)
        if r_out is not None:
            out = r_out
            rep["removed"] += r["removed"]
            rep["fitted"], rep["deviation"] = r["removed"], r["deviation"]
            if picked is not None:
                ends = (ends or []) + r["ends"]
        if picked is not None:
            rep["refused"] += [{"point": f["point"], "why": "surface refit: " + f["why"]} for f in r["failed"]]
        elif r["failed"]:
            rep["failed"] += [{"point": f["point"], "why": "surface refit: " + f["why"]} for f in r["failed"]]
    if vertices and (out is not None or picked is None):
        base = out if out is not None else b
        v, n = remove_vertices(sm, base, None if picked is None else ends)
        if v is not None:
            out, rep["vertices"] = v, n
    return out, rep
