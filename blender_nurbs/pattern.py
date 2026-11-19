"""Pattern: linear, circular and path copies of NURBS objects and curves.

Blender's Array modifier copies only the display mesh: the copies have no
B-rep (no STEP export, no CV net, nothing to pick, trim or join). A pattern
makes each copy an object of its own carrying the source's exact geometry.

A B-rep lives in its object's local space and the object's matrix places it,
so a copy is the source's B-rep text under another matrix: copy k sits at
T_k @ M_src (T_k: the pattern's k-th move or turn, world). Nothing is
re-tessellated when the copies are made: they take the source's mesh as is.

The pattern is laid out on a widget: a frame (centre, X, Y, Z) the tool shows
and the user moves. Its centre snaps like the sketch tools (onto a flat face
it lines up with the face); a ring turns it about its Z (a row at an angle).
  Linear     up to three directions at once (the widget's X, Y, Z): a row,
             a grid or a block; each has its own count and spacing.
  Circular   copies turned about the widget's Z through its centre.
  Path       copies along a curve or surface edge (input 1), from the point
             of the path nearest the source on; turned with the surface the
             path lies on (Z the surface normal, X along the path: the
             copies stay on the surface, as stitches along a seam), with
             the path alone (a rotation-minimising frame), or only moved.
             The surface: a surface edge's own object; a curve's surface
             inputs (Offset on Surface, Project, Iso Curve); else the NURBS
             objects the curve runs on. It is the copies' third input.

With construction history each copy keeps a "pattern" feature (inputs: the
source, and for Path the path; params: the pattern and the copy's index). Its
builder returns the source's geometry at T_k @ M_src in world space, and the
copy's placement (features.PROP_PLACE = inv(the matrix it was made at))
brings that into its own local space. So when the source (or the path)
changes, the copies follow; a copy moved by hand keeps its offset. All copies
of one run share a group id: the Feature panel's Edit Pattern changes them
together and Select All selects them.

Index 0 is the source itself; linear copy (i, j, k) is i + j nx + k nx ny.
"""

import json
import math
import uuid

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty
from mathutils import Matrix, Vector

from . import draw, features, geom, hints, store
from .valuedrag import ValueDrag, angle_pick, rotate_points

OP = "pattern"
MAX_COPIES = 2000           # one run makes at most this many objects
PREVIEW_LINES = 40000       # more preview polylines than this: each copy drawn as its box
PREVIEW_TRIS = 250000       # more preview triangles than this: each copy shaded as its box
PATH_SAMPLES = 512          # samples of a path for its length and frames
CENTER_PX = 11.0            # a press this close to the widget's centre drags it
ARROW_PX = 90.0             # an unused arrow / the ring: this long on screen

MODE_ITEMS = [("LINEAR", "Linear", "Copies along the widget's X, Y and Z: a row, a grid or a block"),
              ("CIRCULAR", "Circular", "Copies turned about the widget's Z axis"),
              ("PATH", "Path", "Copies along a curve or a surface edge")]
SPACE_ITEMS = [("SPACING", "Spacing", "The distance from one copy to the next"),
               ("EXTENT", "Length", "The distance from the first copy to the last")]
AXES = "xyz"
AX_COL = ((1.0, 0.3, 0.35, 1.0), (0.5, 0.9, 0.25, 1.0), (0.3, 0.6, 1.0, 1.0))
COUNT_OF = {"distance_x": "count_x", "distance_y": "count_y", "distance_z": "count_z",
            "angle": "count", "distance": "count"}


# --------------------------------------------------------------------------
# The pattern's transforms (pure: params + source -> world 4x4 matrices)
# --------------------------------------------------------------------------

def _translation(v):
    T = np.eye(4)
    T[:3, 3] = np.asarray(v, float)
    return T


def _rotation(z, deg):
    """3x3: deg degrees about unit z, right-handed (as rotate_points)."""
    z = _unit(z)
    a = math.radians(deg)
    K = np.array([[0.0, -z[2], z[1]], [z[2], 0.0, -z[0]], [-z[1], z[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def _turn_to(a, b):
    """3x3: the smallest rotation taking unit a to unit b."""
    a, b = _unit(a), _unit(b)
    c = float(a @ b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        p = _unit(np.cross(a, (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)))
        return _rotation(p, 180.0)
    return _rotation(v / s, math.degrees(math.atan2(s, c)))


def _unit(v, fallback=(0.0, 0.0, 1.0)):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.asarray(fallback, float)


def _perp(z):
    """A unit vector square to z (X, or Y when z is near X)."""
    z = _unit(z)
    a = np.array((1.0, 0.0, 0.0)) if abs(z[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    return _unit(a - (a @ z) * z)


def world_matrix(obj):
    return np.array(geom.world_matrix(obj), float)


def frame_axes(p):
    """(x, y, z) unit axes of the widget: frame_x made square to frame_z, then
    turned by p["turn"] degrees about z."""
    z = _unit(p.get("frame_z", (0.0, 0.0, 1.0)))
    x0 = np.asarray(p.get("frame_x", (1.0, 0.0, 0.0)), float)
    x0 = x0 - (x0 @ z) * z
    x0 = _unit(x0, _perp(z)) if np.linalg.norm(x0) > 1e-9 else _perp(z)
    y0 = np.cross(z, x0)
    t = math.radians(float(p.get("turn", 0.0)))
    x = math.cos(t) * x0 + math.sin(t) * y0
    return x, np.cross(z, x), z


def normalize(p, src=None):
    """Params of the first Pattern (one direction + rows, axis enums) in
    today's form: the widget frame, a count and spacing per direction."""
    if int(p.get("v", 1)) >= 2:
        return p
    q = dict(p, v=2)
    vec = lambda name: _unit(geom.axis_vector(name, src)) if src is not None or not name.startswith("LOCAL") \
        else _unit(geom.axis_vector(name[-1]))
    if p.get("mode", "LINEAR") == "LINEAR":
        d = _unit(p["axis_dir"]) if p.get("axis") == "PICKED" else vec(p.get("axis", "X"))
        e = vec(p.get("axis_b", "Y"))
        z = np.cross(d, e)
        q.update(frame_x=list(d), frame_z=list(_unit(z, np.cross(d, _perp(d)))), turn=0.0,
                 count_x=int(p.get("count", 2)), distance_x=float(p.get("distance", 0.0)),
                 count_y=int(p.get("count_b", 1)), distance_y=float(p.get("distance_b", 0.0)),
                 count_z=1, distance_z=0.0)
        if abs(float(d @ e)) > 1 - 1e-9:            # the rows ran along the row: none
            q["count_y"] = 1
    else:
        if p.get("rot_axis") == "PICKED":
            O, z = p.get("axis_point", (0, 0, 0)), _unit(p.get("axis_dir", (0, 0, 1)))
        else:
            z = vec(p.get("rot_axis", "Z"))
            O = world_matrix(src)[:3, 3] if p.get("center") == "OBJECT" and src is not None \
                else p.get("center_point", (0, 0, 0))
        q.update(center_point=[float(c) for c in O], frame_z=list(z), frame_x=list(_perp(z)), turn=0.0)
    return q


def step(distance, n, space):
    """The move from one copy to the next."""
    if space == "EXTENT":
        return distance / max(n - 1, 1)
    return distance


def angle_step(angle, n):
    """The turn from one copy to the next: a full turn spreads n copies evenly
    (the last is not on the first), a part of one has copies at both ends."""
    if abs(abs(angle) - 360.0) < 1e-6:
        return angle / n
    return angle / max(n - 1, 1)


def counts(p):
    return tuple(max(int(p.get("count_" + a, 1)), 1) for a in AXES)


def total_count(p):
    if p.get("mode", "LINEAR") == "LINEAR":
        nx, ny, nz = counts(p)
        return nx * ny * nz
    return max(int(p.get("count", 2)), 1)


def anchor(obj):
    """Centre of obj's box (world): what a copy that is not turned keeps its place by."""
    M = world_matrix(obj)
    P = _local_points(obj)
    if P is None or not len(P):
        return M[:3, 3].copy()
    c = 0.5 * (P.min(axis=0) + P.max(axis=0))
    return M[:3, :3] @ c + M[:3, 3]


def _local_points(obj):
    if obj.type == 'MESH':
        me = obj.data
        n = len(me.vertices)
        if not n:
            return None
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        return co.reshape(-1, 3).astype(float)
    if obj.type == 'CURVE':
        pts = []
        for sp in obj.data.splines:
            if sp.type == 'BEZIER':
                pts += [tuple(b.co) for b in sp.bezier_points]
            else:
                pts += [tuple(q.co[:3]) for q in sp.points]
        return np.array(pts, float) if pts else None
    return None


# -- paths -----------------------------------------------------------------

class PathData:
    """A path (world kernel curve) sampled by arc length, with a rotation-
    minimising frame along it (columns: tangent, normal, binormal)."""

    def __init__(self, curve, n=PATH_SAMPLES):
        sm, _ = features._k()
        self.sm, self.curve = sm, curve
        t0, t1 = curve.parameter_range()
        self.t = np.linspace(t0, t1, n)
        self.P = np.array([curve.evaluate(float(t)) for t in self.t], float)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.S = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.S[-1])
        size = float(np.ptp(self.P, axis=0).max()) or 1.0
        self.closed = float(np.linalg.norm(self.P[-1] - self.P[0])) <= 1e-7 * size
        T = [self._tangent(float(t)) for t in self.t]
        F = [np.column_stack([T[0], _perp(T[0]), np.cross(T[0], _perp(T[0]))])]
        for i in range(1, n):
            F.append(_turn_to(T[i - 1], T[i]) @ F[-1])
        self.F = np.array(F)

    def _tangent(self, t):
        d = self.sm.evaluate_curve(self.curve, t)
        return _unit(d["tangent"], (1.0, 0.0, 0.0))

    def at(self, s):
        """(point, frame 3x3) at arc length s (exact point on the curve)."""
        s = float(s) % self.length if self.closed and self.length > 0 else min(max(float(s), 0.0), self.length)
        t = float(np.interp(s, self.S, self.t))
        i = int(min(max(np.searchsorted(self.S, s) - 1, 0), len(self.S) - 2))
        j = i if s - self.S[i] <= self.S[i + 1] - s else i + 1
        T = self._tangent(t)
        F = _turn_to(self.F[j][:, 0], T) @ self.F[j]
        return np.array(self.curve.evaluate(t), float), F

    def nearest(self, Q):
        """Arc length of the path point nearest Q."""
        Q = np.asarray(Q, float)
        A, B = self.P[:-1], self.P[1:]
        AB = B - A
        L2 = np.einsum("ij,ij->i", AB, AB)
        u = np.clip(np.einsum("ij,ij->i", Q - A, AB) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        d = np.linalg.norm(A + AB * u[:, None] - Q, axis=1)
        i = int(np.argmin(d))
        return float(self.S[i] + u[i] * (self.S[i + 1] - self.S[i]))

    def ray_nearest(self, ray_o, ray_d):
        """Arc length of the sample nearest the mouse ray."""
        d = _unit(ray_d)
        rel = self.P - np.asarray(ray_o, float)
        off = rel - np.outer(rel @ d, d)
        return float(self.S[int(np.argmin(np.linalg.norm(off, axis=1)))])

    def polyline(self, s0, s1, m=64):
        return np.array([self.at(s0 + (s1 - s0) * i / max(m - 1, 1))[0] for i in range(m)])


def path_data(path, cache=None):
    """PathData of a path input (object, ref), or None."""
    if path is None:
        return None
    key = (path[0].name, path[1])
    if cache is not None and key in cache:
        return cache[key]
    pd = PathData(features.input_curve(path[0], path[1]))
    if cache is not None:
        cache[key] = pd
    return pd


# -- on a surface ----------------------------------------------------------

def _is_edge(path):
    return path is not None and features.is_surface_edge(path[1])


class SurfaceFrames:
    """Frames on the surface under a path (world): Z the surface's outward
    normal, X the path's direction laid into the tangent plane, Y = Z x X.
    One face is followed along the path while the path stays on it (on an
    edge between two faces: the face the source sits on), so the frames do
    not jump from one side of a seam to the other."""

    SUB = 4                     # every 4th path sample decides the face

    def __init__(self, pd, target, anchor_pt):
        self.pd, self.tg = pd, target
        P = pd.P[::self.SUB]
        self.S = pd.S[::self.SUB]
        m, nf = len(P), len(target.faces)
        D = np.full((nf, m), np.inf)            # distance to each face (inf: not over it)
        Dany = np.full((nf, m), np.inf)         # ... over it or not (the nearest border point)
        self.UV = np.zeros((nf, m, 2))
        eps = max(target.eps, 1e-5 * target.size)
        for fi, fc in enumerate(target.faces):
            uv = fc.E.feet(P, g=17, iters=16)
            S, n, _ = fc.frame(uv)
            d = np.linalg.norm(P - S, axis=1)
            ok = fc.inside(uv, P, S, n, eps, on_trim=True)
            D[fi] = np.where(ok, d, np.inf)
            Dany[fi] = d
            self.UV[fi] = uv
        none = ~np.isfinite(D).any(axis=0)
        D[:, none] = Dany[:, none]              # off every face: the nearest one
        tol = max(1e-3 * max(pd.length, 1e-12), 10.0 * eps)
        j0 = int(np.argmin(np.abs(self.S - pd.nearest(anchor_pt))))
        cand = np.nonzero(D[:, j0] <= D[:, j0].min() + tol)[0]
        if len(cand) > 1:                       # a seam: the face nearest the source
            A = np.asarray(anchor_pt, float)[None]
            da = []
            for fi in cand:
                fc = target.faces[fi]
                uv = fc.E.feet(A, g=17, iters=16)
                S, n, _ = fc.frame(uv)
                da.append(float(np.linalg.norm(A - S)) if fc.inside(uv, A, S, n, eps, on_trim=True)[0] else np.inf)
            start = int(cand[int(np.argmin(da))]) if np.isfinite(min(da)) else int(cand[0])
        else:
            start = int(cand[0])
        face = np.zeros(m, int)

        def walk(order, cur):
            for j in order:
                if D[cur, j] > D[:, j].min() + tol:
                    cur = int(np.argmin(D[:, j]))
                face[j] = cur
            return cur

        last = walk(range(j0, m), start)
        if pd.closed:
            walk(range(0, j0), last)
        else:
            walk(range(j0 - 1, -1, -1), start)
        self.face = face

    def at(self, s):
        """(point on the surface, frame 3x3: X along the path, Z the normal) at arc length s."""
        pd = self.pd
        s = float(s) % pd.length if pd.closed and pd.length > 0 else min(max(float(s), 0.0), pd.length)
        C, F = pd.at(s)
        j = int(np.argmin(np.abs(self.S - s)))
        fi = int(self.face[j])
        fc = self.tg.faces[fi]
        uv = fc.E.feet(C[None], seed=self.UV[fi, j][None], iters=12)
        S, n, _ = fc.frame(uv)
        Z = _unit(n[0])
        T = F[:, 0]
        X = _unit(T - (T @ Z) * Z, _unit(F[:, 1] - (F[:, 1] @ Z) * Z, _perp(Z)))
        return np.asarray(S[0], float), np.column_stack([X, np.cross(Z, X), Z])


def _target(objs, cache):
    key = ("target",) + tuple(sorted(o.name for o in objs))
    tg = cache.get(key)
    if tg is None:
        from .surfoffset import target_of
        tg = cache[key] = target_of(objs)
    return tg


def surface_frames(path, surfaces, src, pd, cache):
    """SurfaceFrames of a path on its surfaces (None: no surface)."""
    if not surfaces and _is_edge(path):
        surfaces = [path[0]]
    surfaces = [o for o in (surfaces or []) if store.is_nurbs(o)]
    if not surfaces:
        return None
    key = ("sframes", src.name, path[0].name, path[1]) + tuple(sorted(o.name for o in surfaces))
    sf = cache.get(key)
    if sf is None:
        sf = cache[key] = SurfaceFrames(pd, _target(surfaces, cache), anchor(src))
    return sf


ON_SURFACE = 1e-3       # a curve this close to a surface (times its length) lies on it


def path_surfaces(path, objects, exclude=()):
    """The NURBS objects a path lies on: a surface edge's own object; a curve
    made on surfaces (its history's surface inputs: Offset on Surface, Project,
    Iso Curve, Intersect); else the objects the curve runs on (within
    ON_SURFACE of its length). [] when it lies on none."""
    if path is None:
        return []
    obj, ref = path
    if features.is_surface_edge(ref):
        return [obj]
    h = features.history(obj)
    if h:
        got = [o for o, r in h["inputs"] if r == "SURF" and store.is_nurbs(o) and o.name not in exclude]
        if got:
            return list(dict.fromkeys(got))
    try:
        pd = path_data(path)
    except Exception:
        return []
    if pd is None or pd.length <= 0:
        return []
    X = pd.P[np.linspace(0, len(pd.P) - 1, 17).astype(int)]
    tol = ON_SURFACE * pd.length
    lo, hi = X.min(axis=0) - tol, X.max(axis=0) + tol
    out = []
    for o in objects:
        if o is obj or o.name in exclude or not store.is_nurbs(o):
            continue
        M = world_matrix(o)
        B = np.array([M[:3, :3] @ np.array(c) + M[:3, 3] for c in o.bound_box], float)
        if (B.min(axis=0) > hi).any() or (B.max(axis=0) < lo).any():
            continue
        try:
            from .surfoffset import target_of
            d = target_of([o]).feet_all(X)[4]
        except Exception as exc:
            print("NURBS pattern: surface test skipped", o.name, exc)
            continue
        if float(d.max()) <= tol:
            out.append(o)
    return out


def path_steps(p, pd, s0):
    """Arc lengths of the copies (the source's s0 first) along the path."""
    n = max(int(p.get("count", 2)), 1)
    dist = float(p.get("distance", 0.0))
    if abs(dist) <= 1e-12:                      # to the path's end (a closed path: all the way round)
        if pd.closed:
            h = pd.length / n
        else:
            h = (pd.length - s0) / max(n - 1, 1)
    else:
        h = step(dist, n, p.get("space", "EXTENT"))
    return [s0 + k * h for k in range(n)]


def copy_transform(p, src, index, path=None, cache=None, surfaces=None):
    """World 4x4 that moves the source to copy `index` (None: no such copy).
    surfaces: the NURBS objects a path lies on (orient SURFACE: the copies
    turn with the surface normal; a surface edge's own object by default)."""
    cache = {} if cache is None else cache
    p = normalize(p, src)
    mode = p.get("mode", "LINEAR")
    if mode == "LINEAR":
        nx, ny, nz = counts(p)
        if not 0 < index < nx * ny * nz:
            return None
        ijk = (index % nx, (index // nx) % ny, index // (nx * ny))
        space = p.get("space", "SPACING")
        v = np.zeros(3)
        for a, n_a, i_a, d in zip(AXES, (nx, ny, nz), ijk, frame_axes(p)):
            if i_a:
                v = v + i_a * step(float(p.get("distance_" + a, 0.0)), n_a, space) * d
        return _translation(v)
    n = max(int(p.get("count", 2)), 1)
    if not 0 < index < n:
        return None
    if mode == "CIRCULAR":
        O = np.asarray(p.get("center_point", (0.0, 0.0, 0.0)), float)
        z = _unit(p.get("frame_z", (0.0, 0.0, 1.0)))
        R = _rotation(z, index * angle_step(float(p.get("angle", 360.0)), n))
        if p.get("rotate", True):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = O - R @ O
            return T
        a = cache.setdefault(("anchor", src.name), anchor(src))
        return _translation(R @ (a - O) + O - a)          # moved round the axis, not turned
    pd = path_data(path, cache)
    if pd is None or pd.length <= 0:
        return None
    pkey = (path[0].name, path[1])
    s0 = cache.get(("s0", src.name, pkey))
    if s0 is None:
        s0 = cache[("s0", src.name, pkey)] = pd.nearest(anchor(src))
    skey = ("steps", src.name, pkey, n, float(p.get("distance", 0.0)), p.get("space", "EXTENT"))
    ss = cache.get(skey)                      # (keyed on the count and spacing: a live stage changes them)
    if ss is None:
        ss = cache[skey] = path_steps(p, pd, s0)
    if not p.get("rotate", True):
        return _translation(pd.at(ss[index])[0] - pd.at(ss[0])[0])
    sf = surface_frames(path, surfaces, src, pd, cache) if p.get("orient", "PATH") == "SURFACE" else None
    if sf is not None:                        # turned with the surface, kept on it
        C0, F0 = sf.at(ss[0])
        Ck, Fk = sf.at(ss[index])
    else:
        C0, F0 = pd.at(ss[0])
        Ck, Fk = pd.at(ss[index])
    R = Fk @ F0.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = Ck - R @ C0
    return T


def copy_transforms(p, src, path=None, cache=None, surfaces=None):
    """[(index, world 4x4)] of every copy (the source, index 0, left out)."""
    cache = {} if cache is None else cache
    p = normalize(p, src)
    out = []
    for k in range(1, total_count(p)):
        T = copy_transform(p, src, k, path, cache, surfaces)
        if T is not None:
            out.append((k, T))
    return out


def circle_or_line(pts):
    """(point, unit direction, closed) of a picked polyline: an open one runs
    from its start to its end; a closed one (a circle, a hole's edge) gives its
    centre (circle fit in its plane) and its normal (right-handed with its run)."""
    P = np.asarray([tuple(q) for q in pts], float)
    size = float(np.ptp(P, axis=0).max()) if len(P) else 0.0
    if len(P) < 2 or size < 1e-12:
        return None
    closed = float(np.linalg.norm(P[-1] - P[0])) <= 1e-6 * size
    if not closed:
        return P[0], _unit(P[-1] - P[0]), False
    Q = P[:-1] if len(P) > 3 else P
    c = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - c)
    n = vt[2]
    x, y = vt[0], vt[1]
    u, v = (Q - c) @ x, (Q - c) @ y
    A = np.column_stack([u, v, np.ones_like(u)])          # u^2 + v^2 = 2a u + 2b v + k
    try:
        sol = np.linalg.lstsq(A, u * u + v * v, rcond=None)[0]
        c = c + 0.5 * sol[0] * x + 0.5 * sol[1] * y
    except np.linalg.LinAlgError:
        pass
    turn = sum(float(np.cross(a - c, b - c) @ n) for a, b in zip(P[:-1], P[1:]))
    return c, (n if turn >= 0.0 else -n), True


# --------------------------------------------------------------------------
# History: a copy follows its source
# --------------------------------------------------------------------------

@features.builder(OP)
def build_pattern(inputs, p):
    """The source's geometry at copy p["index"], in world space (the copy's
    placement takes it into its own local space). Curves: rewritten in place."""
    src, _ = inputs[0]
    path = inputs[1] if len(inputs) > 1 else None
    surfs = [o for o, r in inputs[2:] if r == "SURF"] or None
    T = copy_transform(p, src, int(p.get("index", 1)), path, None, surfs)
    if T is None:
        raise RuntimeError("the pattern has no copy %s" % p.get("index"))
    W = T @ world_matrix(src)
    if src.type == 'CURVE':
        return ("inplace", lambda obj: _write_curve(obj, src, features.placement(obj), W))
    if not store.is_nurbs(src):
        raise RuntimeError(f"{src.name} is not a NURBS object")
    if store.is_natural(src) and store.get_brep(src).face_count() == 1:
        return _moved_surface(store.get_surfaces(src)[0], W)   # an untrimmed surface stays one (Edge mode, CVs)
    b = store.brep_from_text(src[store.PROP_BREP])
    store.transform(b, W)
    return ("brep_text", store.brep_to_text(b))


def _moved_surface(surf, W):
    """A free surface: surf's CVs moved by the affine W (exact, rational ones
    too: an affine move commutes with the weights). A surface of a Brep's face
    can't be moved apart from its Brep."""
    _, ext = features._k()
    d = ext.surface_cvs(surf)
    P = np.array(d["points"], float).reshape(-1, 3) @ W[:3, :3].T + W[:3, 3]
    return ext.surface_from_cvs(d["degree_u"], d["degree_v"], [tuple(map(float, q)) for q in P],
                                list(d["knots_u"]), list(d["knots_v"]),
                                [float(x) for x in d["weights"]] if d.get("weights") else None)


def _write_curve(obj, src, A, W):
    """obj's curve data: src's, moved by A @ W (its local space when obj's
    placement is A). As made: A @ W is the identity, the data a plain copy."""
    L = W if A is None else np.asarray(A, float) @ W
    new = src.data.copy()
    if np.abs(L - np.eye(4)).max() > 1e-12:
        new.transform(Matrix(L.tolist()))
    old = obj.data
    obj.data = new
    if old is not None and old != new:
        name = old.name
        if old.users == 0:
            bpy.data.curves.remove(old)
            new.name = name


def pattern_of(obj):
    """(history dict, params) of a pattern copy, or None."""
    if obj is None or features.PROP_HIST not in obj:
        return None
    try:
        raw = json.loads(obj[features.PROP_HIST])
    except ValueError:
        return None
    if raw.get("op") != OP:
        return None
    return raw, raw.get("params", {})


def group_copies(group, objects=None):
    """Every copy of a pattern group: [(copy, source, index)]."""
    out = []
    if not group:
        return out
    for o in (objects if objects is not None else bpy.data.objects):
        pg = pattern_of(o)
        if pg is None or pg[1].get("group") != group:
            continue
        src = o.get("nproto_in_0")
        if src is not None:
            out.append((o, src, int(pg[1].get("index", 0))))
    return out


def group_sources(group):
    seen = []
    for _, src, _ in group_copies(group):
        if src not in seen:
            seen.append(src)
    return seen


# --------------------------------------------------------------------------
# Making the copies
# --------------------------------------------------------------------------

def usable(obj):
    return obj is not None and (store.is_nurbs(obj) or obj.type == 'CURVE')


def _link_like(obj, src, context):
    colls = [c for c in src.users_collection if c.name != features.HISTORY_COLL]
    if not colls:
        colls = [context.collection or context.scene.collection]
    for c in colls:
        c.objects.link(obj)


def _new_copy(context, src, F, index):
    """A new object holding src's geometry at world matrix F (its mesh is
    src's, copied: the same B-rep in the same local space looks the same)."""
    name = f"{src.name}_p{index}"
    obj = bpy.data.objects.new(name, src.data.copy())
    if src.type != 'CURVE':
        for k in (store.PROP_BREP, store.PROP_DISP, store.PROP_NATURAL, store.PROP_UNTRIM,
                  store.PROP_UNTRIM + "_natural"):
            if k in src:
                obj[k] = src[k]
        if store.PROP_NATURAL not in obj:
            obj[store.PROP_NATURAL] = 1 if store.is_natural(src) else 0
        obj[store.PROP_REV] = store._new_rev(obj)
    _link_like(obj, src, context)
    obj.matrix_world = Matrix(F.tolist())
    obj.color = src.color
    if obj.type == 'MESH':
        from .pivot import remember
        remember(obj)
    return obj


def _ensure_matrix(obj, F):
    """obj's matrix set to F (unparented, as a copy is made)."""
    if obj.parent is not None:
        obj.parent = None
    obj.matrix_basis = Matrix(F.tolist())


def make_pattern(context, sources, p, group=None, path=None, surfaces=None):
    """Copies of each source by the pattern p (path: (object, ref) for Path;
    surfaces: the objects the path lies on, inputs of the copies too).
    group: an existing pattern's group id -- its copies are moved and rebuilt
    where an index still exists, made where it is new, deleted where it is
    gone. -> (copies, errors)."""
    hist = features.history_enabled(context)
    group = group or uuid.uuid4().hex[:12]
    p = dict(p, group=group)
    if path is not None:
        p["path_ref"] = path[1]
    existing = {}
    for o, src, k in group_copies(group):
        existing[(src.name, k)] = o
    made, errors = [], []
    cache = {}
    for src in sources:
        M = world_matrix(src)
        inputs = [(src, "OBJ")] + ([path] if path is not None else [])
        if path is not None and surfaces:
            inputs += [(o, "SURF") for o in surfaces if not (_is_edge(path) and o is path[0])]
        try:
            moves = copy_transforms(p, src, path, cache, surfaces)
        except Exception as exc:
            errors.append(f"{src.name}: {exc}")
            continue
        for k, T in moves:
            F = T @ M
            obj = existing.pop((src.name, k), None)
            try:
                if obj is None:
                    obj = _new_copy(context, src, F, k)
                    fresh = True
                else:
                    _ensure_matrix(obj, F)
                    fresh = False
                if hist:
                    features.record(obj, OP, inputs, dict(p, index=k))
                    features.set_placement(obj, np.linalg.inv(F))
                    if fresh:
                        features.mark_built(obj)     # its geometry is what the feature gives
                    else:
                        features.rebuild(obj, context, force=True)
                elif not fresh:
                    _rewrite_plain(obj, src, context)
                made.append(obj)
            except Exception as exc:
                errors.append(f"{src.name} copy {k}: {exc}")
    for obj in existing.values():           # copies the pattern no longer has
        _remove(obj)
    return made, errors


def _rewrite_plain(obj, src, context):
    """A copy without history given src's geometry again (in its local space)."""
    if src.type == 'CURVE':
        _write_curve(obj, src, None, np.eye(4))
    else:
        store.set_geometry_text(obj, src[store.PROP_BREP], context, keep_untrimmed=False)
        obj[store.PROP_NATURAL] = 1 if store.is_natural(src) else 0


def _remove(obj):
    data = obj.data
    bpy.data.objects.remove(obj)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
        elif isinstance(data, bpy.types.Curve):
            bpy.data.curves.remove(data)


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------

_DIGITS = {f"NUMPAD_{i}": str(i) for i in range(10)} | {
    k: str(i) for i, k in enumerate(("ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"))}


def selected_path(context):
    """The active curve when something else is selected with it (it is a path pattern's path), or None."""
    act = context.view_layer.objects.active
    if act is not None and act.type == 'CURVE' and act.select_get() and \
            any(usable(o) and o != act for o in context.selected_objects):
        return act
    return None


def start_mode(context, last):
    """The mode the one Pattern button starts in: path for a curve active with
    something else selected, else the one used last."""
    return "PATH" if selected_path(context) is not None else last


def _snapper_class():
    from .drawtools import SketchMixin

    class _Snapper(SketchMixin):
        """The sketch tools' point picking, for the widget's centre."""
        face_planes = False

        def _status(self, context):
            pass

    return _Snapper


class NPROTO_OT_pattern(ValueDrag, bpy.types.Operator):
    """Copies of the selected NURBS objects and curves in a row, a grid, a
    block, around an axis or along a path -- real NURBS objects (Blender's
    Array copies only the mesh). From the UI a widget shows the pattern: drag
    its centre (it snaps), its arrows, its ring; number keys give the count,
    Tab types the spacing or angle; click away or Enter to make it"""
    bl_idname = "nproto.pattern"
    bl_label = "Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Pattern", items=MODE_ITEMS, default="LINEAR")
    space: EnumProperty(name="Distance", items=SPACE_ITEMS, default="SPACING")
    # linear: three directions (the widget's X, Y, Z)
    count_x: IntProperty(name="X Count", default=3, min=1, soft_max=50, max=1000,
                         description="Along the widget's X, the original included (1: not along X)")
    count_y: IntProperty(name="Y Count", default=1, min=1, soft_max=50, max=1000)
    count_z: IntProperty(name="Z Count", default=1, min=1, soft_max=50, max=1000)
    distance_x: FloatProperty(name="X Spacing", default=0.0, unit='LENGTH',
                              description="Spacing: from one copy to the next. Length: from the first to the last")
    distance_y: FloatProperty(name="Y Spacing", default=0.0, unit='LENGTH')
    distance_z: FloatProperty(name="Z Spacing", default=0.0, unit='LENGTH')
    turn: FloatProperty(name="Turn", default=0.0, soft_min=-180.0, soft_max=180.0, options={'SKIP_SAVE'},
                        description="The widget turned about its Z: the directions at an angle")
    # circular and path
    count: IntProperty(name="Count", default=6, min=2, soft_max=50, max=1000,
                       description="How many there are, the original included")
    angle: FloatProperty(name="Angle", default=360.0, min=-360.0, max=360.0,
                         description="The turn the copies spread over: 360 spreads them evenly round, "
                                     "less has copies at both ends")
    distance: FloatProperty(name="Path Distance", default=0.0, min=0.0, unit='LENGTH',
                            description="Along the path: the spacing, or the length the copies spread over "
                                        "(0: to the path's end; a closed path: all the way round)")
    rotate: BoolProperty(name="Rotate Copies", default=True,
                         description="Turn each copy with the pattern (off: only moved, kept upright)")
    orient: EnumProperty(name="Follow", default="SURFACE", items=[
        ("SURFACE", "Surface", "On a path that lies on a surface (its edge, or a curve on it): each copy turns "
                               "with the surface normal and stays on the surface, as stitches along a seam"),
        ("PATH", "Path", "Each copy turns with the path alone (a rotation-minimising frame)")])
    surfaces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                             description="JSON list of the NURBS objects the path lies on (empty: look for them)")
    # the widget (world)
    center_point: FloatVectorProperty(name="Center", size=3, subtype='XYZ', unit='LENGTH',
                                      options={'SKIP_SAVE'}, description="The widget's centre: the axis of a "
                                                                         "circular pattern runs through it")
    frame_x: FloatVectorProperty(name="Widget X", size=3, subtype='XYZ', default=(1.0, 0.0, 0.0),
                                 options={'SKIP_SAVE'})
    frame_z: FloatVectorProperty(name="Widget Z", size=3, subtype='XYZ', default=(0.0, 0.0, 1.0),
                                 options={'SKIP_SAVE'}, description="The widget's Z: a circular pattern's axis")
    path_obj: StringProperty(name="Path", options={'SKIP_SAVE'}, description="The curve (or surface) of the path")
    path_ref: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    objects: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                            description="JSON list of the object names to copy (empty: the selection)")
    group: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="An existing pattern to change (its group id)")
    auto: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'},
                       description="The toolbar's one Pattern button: path when the active object is a curve "
                                   "with something else selected, else the mode used last")

    # -- the live value stage (valuedrag.py) ----------------------------------
    angle_fields = ("angle", "turn")
    start_value = 0.0           # spacings start at 0: the copies on the original until an arrow is dragged

    @property
    def drag_fields(self):
        if self.mode == "CIRCULAR":
            return ("angle",)
        if self.mode == "PATH":
            return ("distance",)
        return ("distance_x", "distance_y", "distance_z", "turn")

    def params(self):
        p = {"v": 2, "mode": self.mode, "space": self.space, "turn": self.turn,
             "center_point": list(self.center_point), "frame_x": list(self.frame_x), "frame_z": list(self.frame_z)}
        if self.mode == "LINEAR":
            for a in AXES:
                # an arrow with no spacing makes no copies: off (a count left from an
                # earlier pattern must not stack copies on the original or hold up Enter)
                d = getattr(self, "distance_" + a)
                p["count_" + a] = getattr(self, "count_" + a) if abs(d) > 0.0 else 1
                p["distance_" + a] = d
        elif self.mode == "CIRCULAR":
            p.update(count=self.count, angle=self.angle, rotate=self.rotate)
        else:
            p.update(count=self.count, distance=self.distance, rotate=self.rotate, path_ref=self.path_ref,
                     orient=self.orient)
        return p

    def _load(self, p, src=None):
        """The operator's settings from a pattern's params (Edit Pattern)."""
        p = normalize(p, src)
        for k in ("mode", "space", "count_x", "count_y", "count_z", "distance_x", "distance_y", "distance_z",
                  "turn", "count", "angle", "distance", "rotate", "path_ref"):
            if k in p:
                setattr(self, k, p[k])
        self.orient = p.get("orient", "PATH")       # (patterns made before it: turned with the path)
        for k in ("center_point", "frame_x", "frame_z"):
            if k in p:
                setattr(self, k, tuple(float(x) for x in p[k]))

    def _sources(self, context):
        if self.objects:
            names = json.loads(self.objects)
            out = [o for o in (bpy.data.objects.get(n) for n in names) if usable(o)]
        elif self.group:
            out = [o for o in group_sources(self.group) if usable(o)]
        else:
            out = [o for o in context.selected_objects if usable(o)]
        if self.mode == "PATH" and self.path_obj:
            out = [o for o in out if o.name != self.path_obj]
        return out

    def _path(self):
        if self.mode != "PATH" or not self.path_obj:
            return None
        obj = bpy.data.objects.get(self.path_obj)
        return None if obj is None else (obj, self.path_ref or "S0")

    def _find_surfaces(self, context):
        """Look for the surfaces the path lies on (kept in self.surfaces)."""
        path = self._path()
        if path is None:
            self.surfaces = ""
            return
        srcs = {o.name for o in self._sources(context)}
        objs = [o for o in context.view_layer.objects if o.visible_get()]
        self.surfaces = json.dumps([o.name for o in path_surfaces(path, objs, srcs)])

    def _surface(self):
        """The objects the path lies on, when the copies follow the surface (else None)."""
        if self.mode != "PATH" or not self.rotate or self.orient != "SURFACE":
            return None
        path = self._path()
        if path is None:
            return None
        if _is_edge(path):
            return [path[0]]
        try:
            names = json.loads(self.surfaces) if self.surfaces else []
        except ValueError:
            names = []
        got = [o for o in (bpy.data.objects.get(n) for n in names) if o is not None and store.is_nurbs(o)]
        return got or None

    def _default_widget(self, context, srcs):
        """The widget where a new pattern starts: on the sources' middle
        (circular: the 3D cursor), lined up with the active work plane."""
        from . import workplanes
        wf = workplanes.active_frame(context.scene)
        if wf is not None:
            self.frame_x, self.frame_z = tuple(map(float, wf[1])), tuple(map(float, wf[3]))
        else:
            self.frame_x, self.frame_z = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        self.turn = 0.0
        self.center_point = tuple(map(float, context.scene.cursor.location)) if self.mode == "CIRCULAR" \
            else tuple(map(float, _middle(srcs)))

    # -- start ---------------------------------------------------------------
    def invoke(self, context, event):
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        if self.group:
            copies = group_copies(self.group)
            if not copies:
                self.report({'WARNING'}, "That pattern has no copies left")
                return {'CANCELLED'}
            act = context.view_layer.objects.active
            ref = next((c for c in copies if c[0] == act), copies[0])
            self._load(pattern_of(ref[0])[1], ref[1])
            path_in = ref[0].get("nproto_in_1")
            if path_in is not None:
                self.path_obj = path_in.name
            h = features.history(ref[0])
            if h and self.mode == "PATH":
                self.surfaces = json.dumps([o.name for o, r in h["inputs"][2:] if r == "SURF"])
        else:
            path = selected_path(context)
            if self.auto:
                self.mode = start_mode(context, self.mode)
            if self.mode == "PATH" and not self.path_obj and path is not None:
                self.path_obj, self.path_ref = path.name, "S0"    # the active curve is the path
        srcs = self._sources(context)
        if not srcs:
            self.report({'WARNING'}, "Select NURBS objects or curves to copy "
                                     "(for plain meshes Blender's Array modifier does it)")
            return {'CANCELLED'}
        if not self.objects and not self.group:      # the whole selection: in a path pattern the
            pool = [o for o in context.selected_objects if usable(o)]   # path drops out (_sources), but
            self.objects = json.dumps([o.name for o in pool])          # M to linear copies it too
        else:
            self.objects = json.dumps([o.name for o in srcs])
        if not self.group:
            self._default_widget(context, srcs)
            if self.mode == "PATH" and not self.surfaces:
                self._find_surfaces(context)
        self._keep ={f: getattr(self, f) for f in ("distance_x", "distance_y", "distance_z", "distance")} \
            if self.group else None         # an edit starts from the pattern as it is
        return self.ask(context)

    def ask(self, context):
        r = super().ask(context)
        if self._keep and getattr(self, "_dragging", False):
            for f, v in self._keep.items():
                setattr(self, f, v)
            self._drag_refresh(context)
        return r

    def _fallback(self, context):
        return context.window_manager.invoke_props_dialog(self, width=360, title=self.bl_label,
                                                          confirm_text="Make Pattern")

    def prepare_value(self, context):
        pass

    # -- make it --------------------------------------------------------------
    def execute(self, context):
        from .ops import _kernel_ok
        if not _kernel_ok(self):
            return {'CANCELLED'}
        srcs = self._sources(context)
        if not srcs:
            self.report({'WARNING'}, "Select NURBS objects or curves to copy")
            return {'CANCELLED'}
        if self.mode == "CIRCULAR" and not self.properties.is_property_set("center_point"):
            self.center_point = tuple(context.scene.cursor.location)    # (a script: the cursor as it is now)
        path = self._path()
        if self.mode == "PATH" and path is None:
            self.report({'ERROR'}, "Pattern along a path: pick the path (A in the tool)")
            return {'CANCELLED'}
        p = self.params()
        n = total_count(p) - 1
        if n * len(srcs) > MAX_COPIES:
            self.report({'ERROR'}, f"{n * len(srcs)} copies: at most {MAX_COPIES} in one pattern")
            return {'CANCELLED'}
        if self.mode == "PATH" and self.rotate and self.orient == "SURFACE" and not self.surfaces \
                and not _is_edge(path):
            self._find_surfaces(context)            # (a script: not looked for yet)
        surfs = self._surface()
        if self.mode == "PATH" and self.rotate and self.orient == "SURFACE" and surfs is None:
            self.report({'INFO'}, "Pattern: the path lies on no surface: copies turned with the path")
        act = context.view_layer.objects.active
        made, errors = make_pattern(context, srcs, p, self.group or None, path, surfs)
        for e in errors:
            self.report({'ERROR'}, "Pattern: " + e)
        if not made and errors:
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for o in srcs + made:
            try:
                o.select_set(True)
            except RuntimeError:            # not in this view layer
                pass
        keep = act if (self.group and act in made) else srcs[0]
        context.view_layer.objects.active = keep
        self.report({'INFO'}, f"Pattern: {len(made)} cop{'y' if len(made) == 1 else 'ies'}")
        return {'FINISHED'}

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.row().prop(self, "mode", expand=True)
        if self.mode == "LINEAR":
            lay.row().prop(self, "space", expand=True)
            word = "Spacing" if self.space == "SPACING" else "Length"
            for a in AXES:
                col = lay.column(align=True)
                col.prop(self, "count_" + a)
                sub = col.row()
                sub.active = getattr(self, "count_" + a) > 1
                sub.prop(self, "distance_" + a, text=f"{a.upper()} {word}")
            lay.prop(self, "turn")
            lay.prop(self, "frame_x")
            lay.prop(self, "frame_z")
        elif self.mode == "CIRCULAR":
            lay.prop(self, "count")
            lay.prop(self, "angle")
            lay.prop(self, "rotate")
            lay.prop(self, "center_point")
            lay.prop(self, "frame_z", text="Axis")
        else:
            lay.prop(self, "count")
            lay.row().prop(self, "space", expand=True)
            lay.prop(self, "distance", text=("Spacing" if self.space == "SPACING" else "Length") + " (0: to the end)")
            lay.prop(self, "rotate", text="Turn Copies")
            row = lay.row()
            row.active = self.rotate
            row.prop(self, "orient", expand=True)
            lay.label(text=f"Path: {self.path_obj or '-'}" + (f" ({self.path_ref})" if self.path_ref else ""),
                      icon='CURVE_PATH')
            if self.rotate and self.orient == "SURFACE":
                surfs = self._surface()
                lay.label(text="On: " + ", ".join(o.name for o in surfs) if surfs
                          else "On no surface: turned with the path", icon='SURFACE_NSURFACE')

    # -- the value stage --------------------------------------------------------
    def _fmt(self, context, name):
        if name == "distance" and self.distance <= 0.0:
            return "to the end"
        return super()._fmt(context, name)

    def _label(self, name):
        word = "spacing" if self.space == "SPACING" else "length"
        if name == "distance" and self.distance <= 0.0:
            return f"{self.count}×"
        if name in COUNT_OF and name != "angle":
            n = getattr(self, COUNT_OF[name])
            axis = name[-1].upper() + " " if name != "distance" else ""
            return f"{axis}{n}× {word}"
        if name == "angle":
            return f"{self.count}× over"
        return super()._label(name)

    def _too_short(self, context):
        return ""                       # drag_check says what is missing, per direction

    def drag_setup(self, context, mouse):
        srcs = self._sources(context)
        if not srcs:
            return False
        self._pv = _preview_sources(srcs)
        self._ctx = context
        self._ptmp = {}             # path data of this stage
        self._cbuf = ""             # number keys typed for the count
        self._value_mode = False    # Tab: number keys type the value instead
        self._cdrag = None          # the centre being dragged
        self._picking_axis = False
        self._base_at_mouse = False
        self._snapper = None
        self._moved_center = bool(self.group)
        if self.mode == "PATH" and self._path() is None:
            self._start_axis_pick(context)
        return True

    # widget geometry ----------------------------------------------------------
    def _frame(self):
        return frame_axes(self.params())

    def _center(self):
        return np.asarray(self.center_point, float)

    def _ppu(self, at, d):
        from .drawtools import pixels_per_unit
        region, rv3d = self._win(self._ctx)
        try:
            return pixels_per_unit(region, rv3d, at, [d]) or 0.0
        except Exception:
            return 0.0

    def _screen_len(self, at, d, px):
        ppu = self._ppu(at, d)
        return px / ppu if ppu > 1e-12 else 1.0

    def _ring_radius(self):
        C = self._center()
        x, _, _ = self._frame()
        return self._screen_len(C, x, 0.55 * ARROW_PX)

    def drag_handles(self):
        p = self.params()
        C = self._center()
        if self.mode == "LINEAR":
            out = []
            x, y, z = self._frame()
            for a, d in zip(AXES, (x, y, z)):
                field = "distance_" + a
                n = getattr(self, "count_" + a)
                k = (n - 1) if (self.space == "SPACING" and n > 1) else 1
                ext = getattr(self, field) * k
                L0 = self._screen_len(C, d, ARROW_PX)
                stub = n < 2 or abs(ext) < 0.02 * L0        # (no length on screen: grab it at its tip)
                param = (L0 if ext >= 0 else -L0) if stub else ext
                out.append({"field": field, "kind": "arrow", "origin": C, "dir": d, "param": param, "base": C,
                            "to_value": (lambda t, k=k: t / k), "axis": a, "stub": stub})
            # the ring: turns the widget about its Z
            r = self._ring_radius()
            x0, y0, _ = frame_axes(dict(p, turn=0.0))
            t = self.turn
            m = max(2, int(abs(t) / 4.0) + 2)
            arc = np.array([C + r * (math.cos(math.radians(t * i / (m - 1))) * x0 +
                                     math.sin(math.radians(t * i / (m - 1))) * y0) for i in range(m)])
            pos = arc[-1]
            back = arc[-2] if m > 2 and abs(t) > 1.0 else C + r * (math.cos(math.radians(t - 8)) * x0 +
                                                                   math.sin(math.radians(t - 8)) * y0)
            out.append({"field": "turn", "kind": "arrow", "pos": pos, "param": t, "origin": C, "dir": None,
                        "path": arc if abs(t) > 1e-6 else None, "head_from": back,
                        "pick": lambda o, r_, prev: angle_pick(C, x0, y0, z, o, r_, prev),
                        "to_value": lambda v: v, "snap": (5.0, 1.0)})
            return out
        if self.mode == "CIRCULAR":
            _, _, z = self._frame()
            a = self._anchor()
            P = self._circle_foot(a, C, z)
            if P is None:
                return []
            Cc = C + z * float((P - C) @ z)
            x = _unit(P - Cc)
            y = np.cross(z, x)
            ang = self.angle
            m = max(2, int(abs(ang) / 3.0) + 1)
            path = np.vstack([rotate_points(P[None], C, z, ang * i / (m - 1)) for i in range(m)])
            back = ang - math.copysign(min(6.0, abs(ang) * 0.5), ang) if abs(ang) > 1e-9 else 0.0
            return [{"field": "angle", "kind": "arrow", "pos": rotate_points(P[None], C, z, ang)[0], "param": ang,
                     "origin": Cc, "dir": None, "path": path, "head_from": rotate_points(P[None], C, z, back)[0],
                     "pick": lambda o, r_, prev: angle_pick(Cc, x, y, z, o, r_, prev),
                     "to_value": lambda t: t, "snap": (5.0, 1.0)}]
        # path: a handle on the path at the last copy
        pd = path_data(self._path(), self._ptmp)
        if pd is None or not self._pv:
            return []
        src = self._pv[0][0]
        s0 = pd.nearest(anchor(src))
        ss = path_steps(p, pd, s0)
        s_last = ss[-1]
        n = max(self.count, 2)
        k = (n - 1) if self.space == "SPACING" else 1
        arc = pd.polyline(s0, s_last, 48)
        return [{"field": "distance", "kind": "arrow", "pos": arc[-1], "param": s_last - s0, "origin": arc[0],
                 "dir": None, "path": arc, "head_from": arc[-2],
                 "pick": lambda o, r_, prev: _path_pick(pd, s0, o, r_),
                 "to_value": lambda t, k=k: max(t, 0.0) / k}]

    def _anchor(self):
        """The middle of all the sources' boxes (world)."""
        B = np.vstack([b for _, _, _, b in self._pv])
        return 0.5 * (B.min(axis=0) + B.max(axis=0))

    def _circle_foot(self, a, O, z):
        """Where the angle handle starts: the sources' middle, or (when that is
        on the axis) the box corner farthest from it."""
        B = np.vstack([b for _, _, _, b in self._pv])
        size = float(np.ptp(B, axis=0).max()) or 1.0
        rel = a - O
        if np.linalg.norm(rel - (rel @ z) * z) > 1e-3 * size:
            return a
        rel = B - O
        rad = np.linalg.norm(rel - np.outer(rel @ z, z), axis=1)
        if float(rad.max()) <= 1e-9 * size:
            return None
        return B[int(np.argmax(rad))]

    def drag_preview(self):
        p = self.params()
        path = self._path()
        lines, tris, guides = [], [], []
        k_all = max(total_count(p) - 1, 0)
        n_lines = sum(len(L) for _, L, _, _ in self._pv) * k_all
        n_tris = sum(len(T) // 3 for _, _, T, _ in self._pv) * k_all     # (T: 3 points a triangle)
        surfs = self._surface() if hasattr(self, "_surface") else None
        for src, L, Tw, box in self._pv:
            try:
                moves = copy_transforms(p, src, path, self._ptmp, surfs)
            except Exception as exc:
                print("NURBS pattern preview:", exc)
                moves = []
            for k, T in moves:
                R, t = T[:3, :3], T[:3, 3]
                if n_lines <= PREVIEW_LINES:
                    lines += [P @ R.T + t for P in L]
                else:
                    lines += _box_lines(box @ R.T + t)
                if n_tris <= PREVIEW_TRIS and len(Tw):
                    tris.append(Tw @ R.T + t)
                else:
                    tris.append(_box_tris(box @ R.T + t))
        if self.mode == "PATH" and path is not None:
            pd = path_data(path, self._ptmp)
            if pd is not None:
                guides.append((pd.P, (0.55, 0.85, 1.0, 0.7)))
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3)), "guides": guides}

    def _widget_overlay(self, context):
        """The widget in the sketch overlay: its centre (a dot), its axes in
        their colours, the ring, and for a circular pattern the axis line."""
        ov = draw.sketch_overlay
        refs = []
        if self.mode != "PATH":
            C = self._center()
            x, y, z = self._frame()
            V = lambda q: Vector(tuple(map(float, q)))
            if self.mode == "LINEAR":
                for h in self.drag_handles():
                    if h.get("axis"):
                        i = AXES.index(h["axis"])
                        col = AX_COL[i] if not h["stub"] else (*AX_COL[i][:3], 0.45)
                        refs.append(([V(C), V(C + h["param"] * h["dir"])], col, 3.0 if not h["stub"] else 2.0))
                r = self._ring_radius()
                x0, y0, _ = frame_axes(dict(self.params(), turn=0.0))
                ring = [V(C + r * (math.cos(a) * x0 + math.sin(a) * y0)) for a in np.linspace(0, 2 * math.pi, 65)]
                refs.append((ring, (0.75, 0.75, 0.75, 0.45), 1.5))
            else:
                big = self._screen_len(C, z, 3.0 * ARROW_PX)
                refs.append(([V(C - big * z), V(C + big * z)], AX_COL[2], 2.0))
                r = self._screen_len(C, x, 0.35 * ARROW_PX)
                ring = [V(C + r * (math.cos(a) * x + math.sin(a) * y)) for a in np.linspace(0, 2 * math.pi, 49)]
                refs.append((ring, (*AX_COL[2][:3], 0.8), 2.0))
                a = self._anchor()
                for _, T in copy_transforms(self.params(), self._pv[0][0], None, {}):
                    q = T[:3, :3] @ a + T[:3, 3]
                    foot = C + z * float((q - C) @ z)
                    refs.append(([V(foot), V(q)], (0.75, 0.75, 0.75, 0.35), 1.0))     # a spoke per copy
            ov["cursor"] = V(C)
        else:
            ov["cursor"] = None
        ov["refs"] = refs
        if self._cdrag is None:
            ov["snap"], ov["snap_label"] = None, None

    def _drag_refresh(self, context):
        msg = getattr(self, "_msg", "")         # (the base clears it)
        super()._drag_refresh(context)
        try:
            self._widget_overlay(context)
        except Exception as exc:
            print("NURBS pattern widget:", exc)
        if self._picking_axis:
            self._pick_hints(context)
        else:
            hints.show(context, self.bl_label, self._state(context) + ([msg] if msg else []), self._rows(),
                       self._mouse_hints())
        context.area.tag_redraw()

    def _drag_end(self, context):
        super()._drag_end(context)
        draw.sketch_overlay.update(refs=[], cursor=None, snap=None, snap_label=None, guides=[], face=[],
                                   face_pts=[])

    def drag_check(self):
        if self.mode == "LINEAR":
            p = self.params()
            if all(p["count_" + a] < 2 for a in AXES):
                return "No copies yet: drag an arrow out (number keys: its count, Tab: its spacing)"
        elif self.mode == "CIRCULAR":
            if abs(self.angle) < 0.05:
                return "Angle 0: drag the handle around the axis, or Tab and type an angle"
        elif self._path() is None:
            return "Pick the path first: A"
        n = (total_count(self.params()) - 1) * len(self._pv)
        if n > MAX_COPIES:
            return f"{n} copies: at most {MAX_COPIES}"
        return ""

    # keys ---------------------------------------------------------------------
    def _count_name(self):
        f = self._field_name()
        return COUNT_OF.get(f)

    def _set_field(self, name):
        if name in self.drag_fields:
            self._field = self.drag_fields.index(name)
            self._cbuf = ""

    def drag_key(self, context, event, ch):
        if ch in AXES:
            if self.mode == "LINEAR":
                self._set_field("distance_" + ch)        # that arrow takes the count / value keys
            else:
                self._axis_to(context, ch.upper())
        elif ch == "a":
            self._start_axis_pick(context)
        elif ch == "f" and self._field_name() != "distance":
            f = self._field_name()
            setattr(self, f, -getattr(self, f))
        elif ch == "m":
            order = ["LINEAR", "CIRCULAR", "PATH"]
            self.mode = order[(order.index(self.mode) + 1) % 3]
            self._field, self._cbuf, self._value_mode = 0, "", False
            srcs = self._sources(context)           # (the path is copied too when it stops being the path)
            if srcs and [s.name for s in srcs] != [s.name for s, _, _, _ in self._pv]:
                self._pv = _preview_sources(srcs)
            if not getattr(self, "_moved_center", False):
                self._default_widget(context, [s for s, _, _, _ in self._pv])
            if self.mode == "PATH" and self._path() is None:
                self._start_axis_pick(context)
            elif self.mode == "PATH" and not self.surfaces:
                self._find_surfaces(context)
        elif ch == "r" and self.mode == "PATH":
            # surface -> path -> not turned -> surface
            if self.rotate and self.orient == "SURFACE":
                self.orient = "PATH"
            elif self.rotate:
                self.rotate = False
            else:
                self.rotate, self.orient = True, "SURFACE"
        elif ch == "r" and self.mode == "CIRCULAR":
            self.rotate = not self.rotate
        elif ch == "e" and self.mode != "CIRCULAR":
            self._toggle_space()
        elif ch in "cwb" and self.mode != "PATH":
            P = {"c": lambda: np.array(context.scene.cursor.location, float), "w": lambda: np.zeros(3),
                 "b": self._anchor}[ch]()
            self.center_point = tuple(map(float, P))
            self._moved_center = True
        else:
            return False
        return True

    def _toggle_space(self):
        """Spacing <-> Length: the copies stay where they are, the values change with their meaning."""
        to_ext = self.space == "SPACING"
        self.space = "EXTENT" if to_ext else "SPACING"
        for a in AXES:
            k = max(getattr(self, "count_" + a) - 1, 1)
            v = getattr(self, "distance_" + a)
            setattr(self, "distance_" + a, v * k if to_ext else v / k)
        k = max(self.count - 1, 1)
        if self.distance > 0:
            self.distance = self.distance * k if to_ext else self.distance / k

    def _axis_to(self, context, ax):
        """X / Y / Z (circular): the widget's Z along that axis (the work plane's when one is active)."""
        from . import workplanes
        wf = workplanes.active_frame(context.scene)
        axes = {"X": np.eye(3)[0], "Y": np.eye(3)[1], "Z": np.eye(3)[2]} if wf is None else \
            {"X": np.asarray(wf[1], float), "Y": np.asarray(wf[2], float), "Z": np.asarray(wf[3], float)}
        z = axes[ax]
        nxt = {"X": "Y", "Y": "Z", "Z": "X"}[ax]
        self.frame_z, self.frame_x = tuple(map(float, z)), tuple(map(float, axes[nxt]))

    def drag_modal_hook(self, context, event):
        et, ev = event.type, event.value
        if self._picking_axis:
            return self._axis_pick_modal(context, event)
        if self._cdrag is not None:
            return self._center_modal(context, event)
        if et == 'LEFTMOUSE' and ev == 'PRESS' and self._grab is None and self.mode != "PATH":
            from .ops import PickMixin
            if PickMixin._in_view(context, event):
                m = self._mouse(context, event)
                q = self._screen(context, self._center())
                if q is not None and (q - m).length <= CENTER_PX:
                    return self._start_center(context, event)
                i = self._handle_at(context, m)
                if i is not None and self.mode == "LINEAR":
                    h = self.drag_handles()[i]
                    if h.get("axis") and getattr(self, "count_" + h["axis"]) < 2:
                        setattr(self, "count_" + h["axis"], 2)     # an unused arrow pulled out: 2 along it
                self._cbuf = ""
            return None
        if ev != 'PRESS':
            return None
        if et == 'TAB':
            if self._typed:
                self._apply_typed(context)
            self._typed = ""
            self._value_mode = not self._value_mode
            self._cbuf = ""
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        cname = self._count_name()
        if cname is None or self._value_mode or event.ctrl or event.oskey:
            return None                 # the value takes the keys (ValueDrag types it)
        ch = getattr(event, "unicode", "") or _DIGITS.get(et, "")
        if ch.isdigit():
            self._cbuf = (self._cbuf + ch)[-4:]
            self._set_count(cname, int(self._cbuf))
        elif et == 'BACK_SPACE' and self._cbuf:
            self._cbuf = self._cbuf[:-1]
            if self._cbuf:
                self._set_count(cname, int(self._cbuf))
        elif ch in ("+", "=") or et in {'UP_ARROW', 'NUMPAD_PLUS'}:
            self._set_count(cname, getattr(self, cname) + 1)
            self._cbuf = ""
        elif ch == "-" or et in {'DOWN_ARROW', 'NUMPAD_MINUS'}:
            self._set_count(cname, getattr(self, cname) - 1)
            self._cbuf = ""
        elif ch == ".":
            return {'RUNNING_MODAL'}    # (no fractions of a count)
        else:
            return None
        self._drag_refresh(context)
        return {'RUNNING_MODAL'}

    def _set_count(self, name, n):
        lo = 1 if name.startswith("count_") else 2
        setattr(self, name, max(lo, min(1000, int(n))))

    # the centre ---------------------------------------------------------------
    def _start_center(self, context, event):
        if self._snapper is None:
            self._snapper = _snapper_class()()
            self._snapper.sketch_start(context)
        sn = self._snapper
        sn._plane = sn._view_plane(context, self._center())        # a free point stays at the widget's depth
        self._cdrag = {"c": tuple(self.center_point), "x": tuple(self.frame_x), "z": tuple(self.frame_z),
                       "turn": self.turn}
        return {'RUNNING_MODAL'}

    def _center_modal(self, context, event):
        et, ev = event.type, event.value
        if et in {'MOUSEMOVE', 'LEFT_CTRL', 'RIGHT_CTRL', 'LEFT_ALT', 'RIGHT_ALT', 'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            self._move_center(context, event)
            return {'RUNNING_MODAL'}
        if et == 'LEFTMOUSE' and ev == 'RELEASE':
            self._cdrag = None
            self._moved_center = True
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        if et in {'ESC', 'RIGHTMOUSE'} and ev == 'PRESS':
            d = self._cdrag
            self.center_point, self.frame_x, self.frame_z, self.turn = d["c"], d["x"], d["z"], d["turn"]
            self._cdrag = None
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}

    def _move_center(self, context, event):
        sn = self._snapper
        try:
            P, snap = sn.pick_point(context, event, prev=None)
        except Exception as exc:
            print("NURBS pattern snap:", exc)
            P, snap = None, None
        if P is None:
            return
        P = np.asarray(P, float)
        self.center_point = tuple(map(float, P))
        d = self._cdrag
        face = None
        if not getattr(event, "alt", False):
            try:
                face = sn._snaps.flat_face(sn._view(context), np.asarray(sn._mouse(context, event), float))
            except Exception:
                face = None
        if face is not None and sn.on_face(face, P):     # on a flat face: lined up with it
            self.frame_z, self.frame_x, self.turn = tuple(map(float, face["n"])), tuple(map(float, face["u"])), 0.0
            label = (snap[3] + " · " if snap is not None and snap[3] else "") + "on face"
        else:
            self.frame_x, self.frame_z, self.turn = d["x"], d["z"], d["turn"]
            label = snap[3] if snap is not None else None
        self._drag_refresh(context)
        ov = draw.sketch_overlay
        ov["snap"] = Vector(tuple(P)) if snap is not None else None
        ov["snap_label"] = label if snap is not None else None
        context.area.tag_redraw()

    # the axis / path pick -----------------------------------------------------
    def _start_axis_pick(self, context):
        from .ops import _pick_candidates
        srcs = {s.name for s, _, _, _ in getattr(self, "_pv", [])}
        self._axis_cands = [c for c in _pick_candidates(context)
                            if float(np.ptp(np.array([tuple(q) for q in c[2]]), axis=0).max()) > 1e-9
                            and not (self.mode == "PATH" and c[0].name in srcs)]
        self._axis_hover = None
        self._picking_axis = True
        draw.pick_overlay["hover"] = None
        self._pick_hints(context)

    def _pick_hints(self, context):
        what = {"CIRCULAR": "a line, edge or circle for the axis",
                "LINEAR": "a line or edge for the widget's X (start to end)",
                "PATH": "the path: a curve or a surface edge"}[self.mode]
        hints.show(context, self.bl_label, ["click " + what],
                   [("Pick", [hints.key("LMB", "use it"), hints.key("Esc", "back, unchanged")])],
                   [("MOUSE_LMB", "Pick"), ("EVENT_ESC", "Back")])

    def _axis_pick_modal(self, context, event):
        from .ops import PickMixin, _RegionMouse, _near_candidates
        et = event.type
        if et in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event):
            return {'PASS_THROUGH'}
        if et == 'MOUSEMOVE':
            ties = _near_candidates(context, _RegionMouse(self._mouse(context, event)), self._axis_cands)
            self._axis_hover = ties[0] if ties else None
            draw.pick_overlay["hover"] = self._axis_hover[2] if self._axis_hover else None
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.value != 'PRESS':
            return {'RUNNING_MODAL'}
        if et in {'ESC', 'RIGHTMOUSE'}:
            self._end_axis_pick()
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        if et == 'LEFTMOUSE' and self._axis_hover is not None:
            obj, ref, pts = self._axis_hover
            if self.mode == "PATH":
                self.path_obj, self.path_ref = obj.name, ref
                self._ptmp = {}
                self._find_surfaces(context)
                if self.distance > 0 and self.space == "EXTENT":
                    self.distance = 0.0
            else:
                got = circle_or_line(pts)
                if got is not None:
                    O, d, closed = got
                    if self.mode == "CIRCULAR":
                        self.center_point, self.frame_z = tuple(map(float, O)), tuple(map(float, d))
                        self.frame_x = tuple(map(float, _perp(d)))
                        self._moved_center = True
                    else:
                        z = _unit(self.frame_z)
                        if abs(float(d @ z)) > 1 - 1e-6:
                            self.frame_z = tuple(map(float, _perp(d)))
                        self.frame_x, self.turn = tuple(map(float, d)), 0.0
            self._end_axis_pick()
            self._drag_refresh(context)
        return {'RUNNING_MODAL'}

    def _end_axis_pick(self):
        self._picking_axis = False
        draw.pick_overlay["hover"] = None

    # hints ----------------------------------------------------------------------
    def _state(self, context):
        f = self._field_name()
        out = []
        if self.mode == "LINEAR":
            for a in AXES:
                n = getattr(self, "count_" + a)
                if (n > 1 and abs(getattr(self, "distance_" + a)) > 0.0) or f == "distance_" + a:
                    txt = f"{a.upper()} {n}{'|' if f == 'distance_' + a and self._cbuf else ''} × " + \
                        self._fmt(context, "distance_" + a)
                    out.append(f"[{txt}]" if f == "distance_" + a else txt)
            if abs(self.turn) > 1e-6 or f == "turn":
                out.append(("[" if f == "turn" else "") + f"Turn {self.turn:.4g}°" + ("]" if f == "turn" else ""))
        elif self.mode == "CIRCULAR":
            out.append(f"Count {self.count}{'|' if self._cbuf else ''}")
            out.append(f"Angle {self.angle:.4g}°")
            if not self.rotate:
                out.append("not rotated")
        else:
            out.append(f"Count {self.count}{'|' if self._cbuf else ''}")
            path = self._path()
            out.append(f"Path {path[0].name}" if path else "no path: A")
            out.append(("spacing " if self.space == "SPACING" else "length ") +
                       (self._fmt(context, "distance") if self.distance > 0 else "to the end"))
            if not self.rotate:
                out.append("not turned")
            elif self.orient == "SURFACE":
                surfs = self._surface()
                out.append("on " + ", ".join(o.name for o in surfs) if surfs
                           else "on no surface: turned with the path")
        if self._value_mode or self._typed:
            out.append("typing " + (self._label(f) if f != "turn" else "turn") +
                       (f": {self._typed}|" if self._typed else ""))
        return out

    def _rows(self):
        f = self._field_name()
        what = {"LINEAR": "the active arrow's count", "CIRCULAR": "the count", "PATH": "the count"}[self.mode]
        value = {"LINEAR": "spacing", "CIRCULAR": "angle", "PATH": "distance"}[self.mode]
        if f == "turn":
            count = [hints.key("0-9", "type the turn (degrees)", on=True)]
        else:
            count = [hints.key("0-9", what, on=None if self._value_mode else True),
                     hints.key("+ -", "one more / fewer"),
                     hints.key("Tab", f"type the {value} instead", on=True if self._value_mode else None)]
        rows = [("Count", count)]
        if self.mode == "LINEAR":
            rows.append(("Arrows", [hints.key("LMB drag", "an arrow: to the last copy along it"),
                                    hints.cycle("X Y Z", ["x", "y", "z"],
                                                f[-1] if f.startswith("distance_") else None, "active"),
                                    hints.key("LMB drag", "the ring: turn the widget")]))
        elif self.mode == "CIRCULAR":
            rows.append(("Angle", [hints.key("LMB drag", "the arrow round the axis"), hints.key("Ctrl", "5° steps"),
                                   hints.toggle("R", "rotate copies", self.rotate)]))
        else:
            rows.append(("Path", [hints.key("LMB drag", "the arrow along the path"), hints.key("A", "pick the path"),
                                  hints.cycle("R", ["surface", "path", "not turned"],
                                              ("surface" if self.orient == "SURFACE" else "path")
                                              if self.rotate else "not turned", "turn with")]))
        if self.mode != "PATH":
            wid = [hints.key("LMB drag", "the centre: move it (snaps; on a face it lines up)"),
                   hints.key("C W B", "cursor / world / middle")]
            wid.append(hints.key("A", "pick a line / circle" if self.mode == "CIRCULAR" else "pick a line for X"))
            if self.mode == "CIRCULAR":
                wid.append(hints.key("X Y Z", "axis"))
            rows.append(("Widget", wid))
        if self.mode != "CIRCULAR":
            rows.append(("Distance", [hints.cycle("E", ["spacing", "first to last"],
                                                  "spacing" if self.space == "SPACING" else "first to last")]
                         + ([hints.key("F", "other way")] if self.mode == "LINEAR" else [])))
        else:
            rows.append(("", [hints.key("F", "other way")]))
        rows.append(("Snap", [hints.key("Ctrl", "grid"), hints.key("Alt", "no snapping")]
                     if self.mode != "PATH" else [hints.key("Shift", "precise")]))
        last = self._last_text(self._ctx)
        rows.append(("", [hints.cycle("M", ["linear", "circular", "path"], self.mode.lower())]
                     + ([hints.key("L", f"reuse {last}")] if last else [])))
        rows.append(("", [hints.note("click away from the widget or Enter: make it")]))
        return rows

    @staticmethod
    def _mouse_hints():
        return [("MOUSE_LMB_DRAG", "Drag a handle / the centre"), ("MOUSE_LMB", "Apply"), ("EVENT_ESC", "Cancel")]


def _path_pick(pd, s0, ray_o, ray_d):
    """The path handle's parameter (arc length from s0) for a mouse ray."""
    s = pd.ray_nearest(ray_o, ray_d)
    if pd.closed and s < s0:
        s += pd.length
    return s - s0


def _middle(objs):
    B = np.vstack([_box(o, world_matrix(o)) for o in objs])
    return 0.5 * (B.min(axis=0) + B.max(axis=0))


# preview geometry (world) of a source
def _preview_sources(srcs):
    out = []
    for src in srcs:
        M = world_matrix(src)
        out.append((src, _preview_lines(src, M), _preview_tris(src, M), _box(src, M)))
    return out


def _preview_lines(src, M):
    R, t = M[:3, :3], M[:3, 3]
    if src.type == 'CURVE':
        out = []
        for i in range(len(src.data.splines)):
            try:
                out.append(geom.sample_curve(features.input_curve(src, f"S{i}"), 64))
            except Exception:
                pass
        return out
    return [np.asarray(pl, float) @ R.T + t for pl in store.display(src).get("edges", []) if len(pl) > 1]


def _preview_tris(src, M):
    if src.type != 'MESH':
        return np.zeros((0, 3))
    me = src.data
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    if not nt or nt > PREVIEW_TRIS:
        return np.zeros((0, 3))
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    tri = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    P = co.reshape(-1, 3).astype(float)[tri]
    return P @ M[:3, :3].T + M[:3, 3]


def _box(src, M):
    P = _local_points(src)
    if P is None or not len(P):
        P = np.zeros((1, 3))
    lo, hi = P.min(axis=0), P.max(axis=0)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return C @ M[:3, :3].T + M[:3, 3]


_BOX_TRIS = np.array([(a, b, e, a, e, c) for a, b, c, e in
                      ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 4, 5), (2, 3, 6, 7), (0, 2, 4, 6), (1, 3, 5, 7))]).ravel()


def _box_tris(C):
    """The 12 triangles of a box from its 8 corners (_box's order)."""
    return C[_BOX_TRIS]


def _box_lines(C):
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [np.array([C[a], C[b]]) for a, b in pairs]


# --------------------------------------------------------------------------
# Feature panel: a copy's pattern
# --------------------------------------------------------------------------

class NPROTO_OT_pattern_select(bpy.types.Operator):
    """Select every copy of this pattern and its originals"""
    bl_idname = "nproto.pattern_select"
    bl_label = "Select Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    group: StringProperty(options={'HIDDEN'})

    def execute(self, context):
        copies = group_copies(self.group, context.scene.objects)
        if not copies:
            self.report({'WARNING'}, "No copies of that pattern")
            return {'CANCELLED'}
        act = context.view_layer.objects.active
        for o in context.selected_objects:
            o.select_set(False)
        for o in dict.fromkeys([s for _, s, _ in copies] + [c for c, _, _ in copies]):
            try:
                o.select_set(True)
            except RuntimeError:
                pass
        if act is not None and act.select_get():
            context.view_layer.objects.active = act
        return {'FINISHED'}


def summary(p):
    """'Linear · 4 × 20 mm' style words for a pattern's params."""
    from .valuedrag import fmt_length
    ctx = bpy.context
    p = normalize(p)
    mode = p.get("mode", "LINEAR")
    if mode == "CIRCULAR":
        txt = f"Circular · {int(p.get('count', 2))} over {float(p.get('angle', 360.0)):g}°"
        return txt + ("" if p.get("rotate", True) else " · not rotated")
    if mode == "PATH":
        d = float(p.get("distance", 0.0))
        word = "spacing" if p.get("space") == "SPACING" else "length"
        return f"Path · {int(p.get('count', 2))} · " + (f"{word} {fmt_length(ctx, d)}" if d > 0 else "to the end") \
            + ("" if p.get("rotate", True) else " · not turned") \
            + (" · on the surface" if p.get("rotate", True) and p.get("orient") == "SURFACE" else "")
    parts = []
    for a in AXES:
        n = int(p.get("count_" + a, 1))
        if n > 1:
            parts.append(f"{a.upper()} {n} × {fmt_length(ctx, float(p.get('distance_' + a, 0.0)))}")
    txt = "Linear · " + (" · ".join(parts) or "no copies")
    if abs(float(p.get("turn", 0.0))) > 1e-6:
        txt += f" · turned {float(p['turn']):g}°"
    return txt


def draw_feature(box, context, obj, h):
    """The Feature panel's part for a pattern copy (feature_ui.draw_feature)."""
    p = h["params"]
    col = box.column(align=True)
    col.label(text=summary(p), icon='MOD_ARRAY')
    copies = group_copies(p.get("group", ""), context.scene.objects)
    col.label(text=f"Copy {int(p.get('index', 0))} · {len(copies)} copies in the pattern", icon='LINKED')
    row = box.row(align=True)
    op = row.operator("nproto.pattern", text="Edit Pattern", icon='MODIFIER')
    op.group = p.get("group", "")
    row.operator("nproto.pattern_select", text="Select All", icon='RESTRICT_SELECT_OFF').group = p.get("group", "")


def patterns_from(obj, objects):
    """The patterns made from obj: [(group, params, its copies of obj)], oldest group first."""
    found = {}
    for o in objects:
        if o.get("nproto_in_0") is not obj:
            continue
        pg = pattern_of(o)
        if pg is None or not pg[1].get("group"):
            continue
        g = pg[1]["group"]
        if g not in found:
            found[g] = [pg[1], 0, o.name]
        found[g][1] += 1
    return [(g, p, n) for g, (p, n, _) in sorted(found.items(), key=lambda kv: kv[1][2])]


def draw_patterns_of(layout, context, obj):
    """The Feature panel of an original: the patterns made from it, each with
    Edit Pattern and Select All (as a copy's Feature panel has them)."""
    pats = patterns_from(obj, context.scene.objects)
    if not pats:
        return
    box = layout.box()
    box.label(text="Patterns made from this object", icon='MOD_ARRAY')
    for g, p, n in pats:
        col = box.column(align=True)
        col.label(text=f"{summary(p)} · {n} cop{'y' if n == 1 else 'ies'}")
        row = col.row(align=True)
        row.operator("nproto.pattern", text="Edit Pattern", icon='MODIFIER').group = g
        row.operator("nproto.pattern_select", text="Select All", icon='RESTRICT_SELECT_OFF').group = g


classes = (NPROTO_OT_pattern, NPROTO_OT_pattern_select)
