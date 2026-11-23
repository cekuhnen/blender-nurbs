"""Out-of-process kernel jobs (trim, fillet).

Some SMLib calls can crash the process (the surface-surface fillet is documented
to do so for some inputs, and ops that adopt an input surface crash when it is
reused). Running them here, in a separate Python, means a crash fails the
command instead of taking Blender down.

Usage:  python -I worker.py job.json      (prints one JSON line: the result)

Job JSON:
  op:        "trim" | "fillet"
  breps:     {key: path to SMLib ASCII brep (object local space)}
  xforms:    {key: {"loc": [x,y,z], "axis": [x,y,z], "angle": deg, "scale": [sx,sy,sz]}}
  curves:    [{"degree", "points", "knots", "weights"}]  (world space)
  out:       {key: path to write a result brep (written in that key's local space,
              or world space for keys without an xform)}
  ...op params
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _nproto_kernel as sm  # noqa: E402
from harden import harden  # noqa: E402
from natext import extend_curve, extend_grid  # noqa: E402

harden(sm)               # end-of-range curve evaluation that every kernel build accepts

ext = sm.blender_ext


def load(path):
    exact = getattr(ext, "read_brep_exact", None)   # no analytic conversion (keeps CVs)
    return exact(path) if exact else sm.Brep.read_from_file(path, ascii=True)


def save(brep, path):
    brep.write_to_file(path, ascii=True)


def is_identity(x):
    return (x is None or (all(abs(v) < 1e-12 for v in x["loc"]) and abs(x["angle"]) < 1e-9
                          and all(abs(s - 1.0) < 1e-12 for s in x["scale"])))


def to_world(b, x):
    if is_identity(x):
        return b
    if any(abs(s - 1.0) > 1e-12 for s in x["scale"]):
        b = sm.scale(b, tuple(x["scale"]))
    if abs(x["angle"]) > 1e-9:
        b = sm.rotate(b, (0.0, 0.0, 0.0), tuple(x["axis"]), x["angle"])
    if any(abs(v) > 1e-12 for v in x["loc"]):
        b = sm.translate(b, tuple(x["loc"]))
    return b


def to_local(b, x):
    if is_identity(x):
        return b
    if any(abs(v) > 1e-12 for v in x["loc"]):
        b = sm.translate(b, tuple(-v for v in x["loc"]))
    if abs(x["angle"]) > 1e-9:
        b = sm.rotate(b, (0.0, 0.0, 0.0), tuple(x["axis"]), -x["angle"])
    if any(abs(s - 1.0) > 1e-12 for s in x["scale"]):
        b = sm.scale(b, tuple(1.0 / s for s in x["scale"]))
    return b


def world_brep(job, key):
    return to_world(load(job["breps"][key]), job.get("xforms", {}).get(key))


def curve(d):
    return ext.curve_from_cvs(d["degree"], [tuple(p) for p in d["points"]], d["knots"],
                              d["weights"] or None)


def mid(edge):
    t0, t1 = edge.parameter_range()
    return edge.point_at(0.5 * (t0 + t1))


def dist_to_brep(p, brep):
    return min(f.closest_point(p)[1] for f in brep.faces())


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Trim: split into regions, classify, keep / discard the chosen regions
# ---------------------------------------------------------------------------

def _np():
    import numpy as np  # Blender's bundled numpy
    return np


def _poly(curve_obj, n=200):
    t0, t1 = curve_obj.parameter_range()
    return [curve_obj.evaluate(t0 + (t1 - t0) * i / (n - 1)) for i in range(n)]


def _edge_poly(e, n=24):
    try:
        t0, t1 = e.parameter_range()
        return [e.point_at(t0 + (t1 - t0) * i / (n - 1)) for i in range(n)]
    except Exception:
        try:
            return _poly(e.curve(), n)
        except Exception:
            return []


def _edge_mid(e):
    pts = _edge_poly(e, 3)
    return pts[1] if len(pts) == 3 else None


def corner_pieces(d, ang_deg=0.05):
    """Curve data (degree, points, knots, weights) split at its sharp corners:
    the interior knots of full multiplicity where the tangent turns by more
    than ang_deg. Splitting there is exact (no knot insertion): the curve
    passes through the CV at such a knot. Returns [data] when it has none."""
    np = _np()
    p = int(d["degree"])
    out, cur = [], {"degree": p, "points": [list(q) for q in d["points"]],
                    "knots": [float(k) for k in d["knots"]],
                    "weights": list(d["weights"]) if d.get("weights") else None}
    cos_max = float(np.cos(np.radians(ang_deg)))
    while True:
        U, P, W = cur["knots"], cur["points"], cur["weights"]
        n, cut = len(P), None
        for t in sorted(set(U[p + 1:len(U) - p - 1])):
            s = U.index(t)
            if U.count(t) != p or p < 1:
                continue
            j = s - 1                          # the CV the curve passes through at t
            if j < 1 or j + 1 >= n:
                continue
            a, b = np.subtract(P[j], P[j - 1]), np.subtract(P[j + 1], P[j])
            la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if la < 1e-15 or lb < 1e-15:
                continue
            if float(a @ b) / (la * lb) < cos_max:
                cut = (s, t)
                break
        if cut is None:
            out.append(cur)
            return out
        s, t = cut
        out.append({"degree": p, "points": P[:s], "knots": U[:s + p] + [t], "weights": W[:s] if W else None})
        cur = {"degree": p, "points": P[s - 1:], "knots": [t] + U[s:], "weights": W[s - 1:] if W else None}


class CurveCutter:
    """A cutting curve projected along a direction: splits by parallel
    projection; a point is 'inside' when it projects inside the curve outline."""

    def __init__(self, data, direction, reach="BOTH"):
        np = _np()
        self.data = data
        self.reach = reach      # BOTH ways from the curve, or only FRONT (along direction) / BACK
        d = np.array(direction, float)
        self.dir = d / (np.linalg.norm(d) or 1.0)
        P = np.array(_poly(curve(data)))
        a = np.cross(self.dir, [1.0, 0, 0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(self.dir, [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(self.dir, a)
        self.basis = (a, b)
        self.poly2d = np.stack([P @ a, P @ b], axis=1)
        self.origin = P.mean(axis=0)
        self.closed = np.linalg.norm(P[0] - P[-1]) <= 1e-6 * (np.ptp(P, axis=0).max() or 1.0)

    def split(self, b, ref):
        """Exact imprint of the curve swept along the projection direction (a
        cutter sheet long enough to pass through the target). The kernel's
        parallel projection can drop pieces of loops that cross a seam. A
        cutter with sharp corners (a line-arc-line notch, a polyline) is
        imprinted one smooth piece at a time: one sheet with creases let the
        intersection lose a sliver at a corner (a 0.15 mm gap in a spoon's
        notch, so the loop did not close); a piece's end is a sheet edge,
        which the imprint meets exactly."""
        for sheet in self.sheet_list(b):
            b = sm.boolean_with_options(b, sheet, operation=sm.BooleanOp.UNION, imprinting=True,
                                        keep_other_brep=True)
            if b is None:
                return None
        return b

    def sheet_list(self, b):
        """The cutting sheets (one per smooth piece of the curve), long enough
        to pass through b."""
        np = _np()
        lo, hi = (np.array(v) for v in b.bounding_box())
        P = np.array(_poly(curve(self.data), 64))
        reach = float(np.linalg.norm(hi - lo)) + float(np.max(np.linalg.norm(P - 0.5 * (lo + hi), axis=1)))
        L = 4.0 * (reach or 1.0)
        d = -self.dir if self.reach == "BACK" else self.dir
        start = -0.5 * L if self.reach == "BOTH" else 0.0
        out = []
        for piece in corner_pieces(self.data):
            c = sm.translate(curve(piece), tuple(start * d)) if start else curve(piece)
            out.append(ext.brep_from_surface(sm.create_extrude_surface(c, tuple(L * d))))
        return out

    def inside(self, p):
        if not self.closed:
            return None  # an open cutter only splits; it has no inside
        np = _np()
        if self.reach != "BOTH":     # behind a one-sided cut nothing was cut: never inside there
            ahead = float(np.dot(np.asarray(p, float) - self.origin, self.dir))
            if (ahead < 0) if self.reach == "FRONT" else (ahead > 0):
                return False
        a, bb = self.basis
        x, y = float(np.dot(p, a)), float(np.dot(p, bb))
        poly = self.poly2d
        c = False
        for i in range(len(poly) - 1):
            (x1, y1), (x2, y2) = poly[i], poly[i + 1]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                c = not c
        return c


class SurfaceCutter:
    """A cutting surface: the target is imprinted exactly along the
    intersection; 'inside' = the side of the cutter's surface normal."""

    def __init__(self, job, key, brep=None):
        np = _np()
        self.job, self.key, self.brep = job, key, brep    # brep: the cutter extended to reach across
        arr = self._brep().tessellate(chord_height_tolerance=0.0005).to_numpy_arrays()
        self.P = arr["points"].astype(float)
        self.N = arr["normals"].astype(float)

    def _brep(self):
        return self.brep.copy() if self.brep is not None else world_brep(self.job, self.key)

    def split(self, b, ref):
        return sm.boolean_with_options(b, self._brep(), operation=sm.BooleanOp.UNION,
                                       imprinting=True, keep_other_brep=True)

    def sheet_list(self, b):
        return [self._brep()]

    def inside(self, p):
        np = _np()
        i = int(np.argmin(np.linalg.norm(self.P - p, axis=1)))
        return bool(np.dot(p - self.P[i], self.N[i]) < 0)


def _regions(job, cutters):
    """Fresh split target + face groups (faces joined across ORIGINAL edges such
    as the seam of a revolve), the picked group and each group's side vector."""
    np = _np()
    ref = tuple(job["ref_point"])
    original = world_brep(job, "target")
    orig_polys = [np.array(p) for p in (_edge_poly(e, 48) for e in original.edges()) if len(p)]
    bb = original.bounding_box()
    size = float(np.linalg.norm(np.array(bb[1]) - np.array(bb[0]))) or 1.0
    tol = 1e-5 * size

    b = world_brep(job, "target")
    for c in cutters:
        b = c.split(b, ref)
        if b is None or b.face_count() == 0:
            raise RuntimeError("the cut produced nothing (does the cutter cross the surface?)")
    faces = b.faces()
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def on_original(pt):
        for P in orig_polys:
            seg = P[1:] - P[:-1]
            t = np.clip(np.einsum("ij,ij->i", pt - P[:-1], seg)
                        / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-30), 0, 1)
            if np.min(np.linalg.norm(P[:-1] + seg * t[:, None] - pt, axis=1)) <= tol * 10:
                return True
        return False

    # faces sharing an edge that runs ALONG an original edge (e.g. the seam of a
    # revolve) belong to the same region; a cut edge that merely crosses an
    # original edge somewhere does not count
    shared = {}
    for i, f in enumerate(faces):
        for e in f.edges():
            m = _edge_mid(e)
            if m is None:
                continue
            k = tuple(round(v, 6) for v in m)
            entry = shared.setdefault(k, [e, set()])
            entry[1].add(i)
    for k, (e, fs) in shared.items():
        if len(fs) < 2:
            continue
        pts = _edge_poly(e, 9)[1:-1]
        if pts and all(on_original(np.array(p)) for p in pts):
            fs = sorted(fs)
            for j in fs[1:]:
                parent[find(j)] = find(fs[0])
    groups = {}
    for i in range(len(faces)):
        groups.setdefault(find(i), []).append(i)
    groups = list(groups.values())
    picked_face = min(range(len(faces)), key=lambda i: faces[i].closest_point(ref)[1])
    picked = next(gi for gi, g in enumerate(groups) if picked_face in g)
    sides = []
    for g in groups:
        f = max((faces[i] for i in g), key=lambda fc: fc.area(1e-3))
        p = _inner_point(f)          # not on an edge, where the side is ambiguous
        sides.append(tuple(c.inside(p) for c in cutters))
    return b, faces, groups, picked, sides


def _near_edges(brep, tol):
    """Is a point on one of brep's edges (within tol)? Sampled finely: a
    48-point polyline sags off a curved edge by more than tol."""
    np = _np()
    polys = [np.array(p) for p in (_edge_poly(e, 400) for e in brep.edges()) if len(p)]

    def on(pt):
        for P in polys:
            seg = P[1:] - P[:-1]
            t = np.clip(np.einsum("ij,ij->i", pt - P[:-1], seg)
                        / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-30), 0, 1)
            if np.min(np.linalg.norm(P[:-1] + seg * t[:, None] - pt, axis=1)) <= tol:
                return True
        return False
    return on


def _rejoin_runs(lp, segs, original, tol):
    """The loop lp (indices into segs: (edge, forward, start, end)) with runs
    of consecutive pieces of ONE original edge -- split where a cut crossed
    it, at a point that is no vertex of the original -- given back as that
    edge's own curve over their range (a curve in place of the indices).
    Plain indices for everything else. Open original edges only."""
    np = _np()
    oedges = original.edges()
    overts = [np.array(e.point_at(t), float) for e in oedges for t in e.parameter_range()]

    def owner(j):
        e = segs[j][0]
        t0, t1 = e.parameter_range()
        best = None
        for f in (0.25, 0.5, 0.75):
            q = tuple(map(float, e.point_at(t0 + f * (t1 - t0))))
            ds = [oe.closest_point(q)[2] for oe in oedges]
            k = int(np.argmin(ds))
            if ds[k] > 10 * tol or (best is not None and k != best):
                return None
            best = k
        oe = oedges[best]
        a, z = oe.parameter_range()
        if np.linalg.norm(np.array(oe.point_at(a), float) - np.array(oe.point_at(z), float)) <= 10 * tol:
            return None                     # a closed rim: its pieces stay
        return best

    def joint(p):
        return not any(np.linalg.norm(p - v) <= 10 * tol for v in overts)

    try:
        own = [owner(j) for j in lp]
    except Exception:
        return list(lp)
    n = len(lp)
    if n < 2 or all(o is None for o in own):
        return list(lp)
    # start the walk at a break so a run never wraps round the loop's start
    starts = [i for i in range(n) if not (own[i] is not None and own[i] == own[i - 1] and joint(segs[lp[i]][2]))]
    if not starts:
        return list(lp)
    k0 = starts[0]
    order = [(k0 + i) % n for i in range(n)]
    out, run = [], []

    def flush():
        if len(run) == 1:
            out.append(lp[run[0]])
        elif run:
            try:
                import trimext
                oe = oedges[own[run[0]]]
                d = ext.curve_cvs(oe.curve())
                P = np.array(d["points"], float)
                w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
                Q = list(np.concatenate([P * w[:, None], w[:, None]], axis=1))
                U = [float(x) for x in d["knots"]]
                ta = oe.closest_point(tuple(map(float, segs[lp[run[0]]][2])))[1]
                tb = oe.closest_point(tuple(map(float, segs[lp[run[-1]]][3])))[1]
                Q, U = trimext._cut(Q, U, int(d["degree"]), max(min(ta, tb), U[0]), min(max(ta, tb), U[-1]))
                out.append(trimext._make_curve(Q, U, int(d["degree"]), reverse=ta > tb))
            except Exception:
                out.extend(lp[i] for i in run)
        run.clear()

    for i in order:
        if run and own[i] is not None and own[i] == own[run[-1]] and joint(segs[lp[i]][2]):
            run.append(i)
            continue
        flush()
        run.append(i)
    flush()
    return out


def heal_cut_seams(b, original):
    """Remove the cut edges that have kept surface on both sides: a trim
    imprints every cutter, and where a region on each side is kept (the far
    wall a projected circle also crossed) the imprint stayed as a seam -- two
    faces, or one face whose loop runs out and back along a slit. Each such
    set of faces on one surface is trimmed again to the outline of their
    union, exactly (the edges' own curves). Original edges (a revolve's seam)
    stay. Pieces of one original face count as one surface even when the
    kernel's split gave them different representations of it (a cubic plane
    split into bilinear patches): they are trimmed again on the original
    face's surface. Returns b unchanged when anything does not check out."""
    np = _np()
    lo, hi = (np.array(v) for v in original.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    tol = 1e-5 * size
    from trimext import brep_tol
    btol = brep_tol(1e-5, size)      # not the kernel's 0.01: stitched below, short edges stay apart
    on_original = _near_edges(original, 10 * tol)

    def key(e):
        m = _edge_mid(e)
        return None if m is None else tuple(round(float(v) / tol) for v in m)

    faces = b.faces()
    skeys = []
    for f in faces:
        try:
            skeys.append(_surface_key(f.surface()))
        except Exception:
            skeys.append(None)
    ofaces = original.faces()
    okeys = {}

    def origin(i):
        """Index of the original face piece i lies on (None: none clearly)."""
        if i not in okeys:
            try:
                p = tuple(map(float, _inner_point(faces[i])))
                ds = [of.closest_point(p)[1] for of in ofaces]
                j = int(np.argmin(ds))
                okeys[i] = j if ds[j] <= 10 * tol else None
            except Exception:
                okeys[i] = None
        return okeys[i]

    def same_surface(i, j):
        if skeys[i] is not None and skeys[i] == skeys[j]:
            return True
        return origin(i) is not None and origin(i) == origin(j)

    uses = {}       # edge key -> [(face index, forward)]
    edge_of = {}
    for i, f in enumerate(faces):
        for loop in f.boundary_loops():
            for e, fw in loop:
                k = key(e)
                if k is None:
                    return b
                uses.setdefault(k, []).append((i, fw))
                edge_of[k] = e
    seams = set()
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for k, us in uses.items():
        if len(us) != 2 or us[0][1] == us[1][1] or not same_surface(us[0][0], us[1][0]):
            continue
        pts = _edge_poly(edge_of[k], 7)[1:-1]
        if pts and all(on_original(np.array(p)) for p in pts):
            continue                               # an original edge: the seam of a revolve
        seams.add(k)
        parent[find(us[0][0])] = find(us[1][0])
    if not seams:
        return b
    clusters = {}
    for i in range(len(faces)):
        clusters.setdefault(find(i), []).append(i)
    merged, drop = [], []
    for members in clusters.values():
        if not any(k in seams for k, us in uses.items() if us[0][0] in members):
            continue
        # the union's boundary: every oriented edge of the members but the seams.
        # A loop the cut did not touch is kept as the kernel gives it (a
        # revolve's loop runs along its seam both ways, with closed circles,
        # which chaining by position would mix up); the rest are chained anew
        segs, loops, runs = [], [], []
        for i in members:
            for loop in faces[i].boundary_loops():
                ids = []
                for e, fw in loop:
                    if key(e) in seams:
                        ids.append(None)
                        continue
                    P = _edge_poly(e, 3)
                    a, z = (np.array(P[0]), np.array(P[-1])) if fw else (np.array(P[-1]), np.array(P[0]))
                    segs.append((e, fw, a, z))
                    ids.append(len(segs) - 1)
                if None not in ids:
                    loops.append(ids)
                    continue
                # the loop's runs between cut edges, each in the loop's own order
                # (closed rims meeting a revolve's seam at one vertex make
                # chaining single edges by position ambiguous)
                k = ids.index(None)
                rot = ids[k + 1:] + ids[:k + 1]
                cur = []
                for j in rot:
                    if j is None:
                        if cur:
                            runs.append(cur)
                        cur = []
                    else:
                        cur.append(j)
                if cur:
                    runs.append(cur)
        def link(join_first):
            """Loops from the runs. join_first: a run that comes back to its
            start still takes on another run starting there (a cut that met a
            revolve's seam at one vertex leaves a closed run on each side)."""
            rs, out = [list(r) for r in runs], [list(lp) for lp in loops]
            while rs:
                chain = rs.pop(0)
                start = segs[chain[0]][2]
                while True:
                    end = segs[chain[-1]][3]
                    nxt = min(range(len(rs)), key=lambda r: np.linalg.norm(segs[rs[r][0]][2] - end), default=None)
                    near = nxt is not None and np.linalg.norm(segs[rs[nxt][0]][2] - end) <= 100 * tol
                    closed = np.linalg.norm(end - start) <= 100 * tol
                    if closed and not (join_first and near):
                        break
                    if not near:
                        return None
                    chain += rs.pop(nxt)
                out.append(chain)
            return out

        def span(lp):
            P = np.array([segs[j][2] for j in lp])
            return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
        area = sum(faces[i].area(1e-6) for i in members)
        made = None
        for join_first in (False, True):
            loops_try = link(join_first)
            if loops_try is None or (join_first and not runs):
                continue
            loops_try.sort(key=span, reverse=True)          # the outer loop first
            for tight in (True, False):
                curves, sizes = [], []
                for lp in loops_try:
                    n0 = len(curves)
                    for item in _rejoin_runs(lp, segs, original, tol):
                        if not isinstance(item, int):    # a piece of an original edge, rejoined
                            curves.append(item)
                            continue
                        e, fw = segs[item][0], segs[item][1]
                        d = ext.curve_cvs(e.curve())
                        U, Pc, Wt = list(d["knots"]), list(d["points"]), d.get("weights")
                        if not fw:
                            a0, a1 = U[0], U[-1]
                            U = [a0 + a1 - k for k in reversed(U)]
                            Pc, Wt = Pc[::-1], (list(Wt)[::-1] if Wt else None)
                        curves.append(ext.curve_from_cvs(d["degree"], Pc, U, list(Wt) if Wt else None))
                    sizes.append(len(curves) - n0)
                one = len({skeys[i] for i in members}) == 1 and skeys[members[0]] is not None
                carrier = faces[members[0]].surface() if one else ofaces[origin(members[0])].surface()
                sd = ext.surface_cvs(carrier)
                surf = ext.surface_from_cvs(sd["degree_u"], sd["degree_v"], list(sd["points"]), list(sd["knots_u"]),
                                            list(sd["knots_v"]), list(sd["weights"]) if sd.get("weights") else None) \
                    if "knots_u" in sd else carrier
                try:
                    if tight and hasattr(ext, "trim_surface_with_3d_curves_tol"):
                        cand = ext.trim_surface_with_3d_curves_tol(surf, sizes, curves, tol, btol)
                    else:
                        cand = sm.trim_surface_with_3d_curves(surf, sizes, curves)
                except Exception:
                    continue
                if cand.face_count() == 1 and abs(cand.area(1e-6) - area) <= 1e-4 * max(area, 1e-12):
                    made = cand
                    break
            if made is not None:
                break
        if made is None:
            return b
        merged.append(made)
        drop += members
    if len(drop) == len(faces):
        if len(merged) == 1:
            return merged[0]
        out = merged[0]
        for m in merged[1:]:
            ext.append_brep(out, m)
        return out
    out = b.copy()
    of = out.faces()
    ext.delete_faces(out, [of[i] for i in drop])
    for m in merged:
        ext.append_brep(out, m)
    # b's open edges but the seams that go: a seam between two faces counts
    # once per edge left open (two when the cut left it unstitched, none when
    # it is one shared edge -- on a solid it is, and counting two there asked
    # for a closed result); a slit (one face both ways) is not counted open
    open_seams = sum(1 for e in b.edges() if len(e.faces()) < 2 and key(e) in seams
                     and uses[key(e)][0][0] != uses[key(e)][1][0])
    before_open = _open_edge_count(b) - open_seams
    for t in (tol / 10, tol, 10 * tol):
        st = sm.stitch_advanced(out.copy(), t, squeeze_small_edges=False, making_manifold_solid=False)[0]
        if _open_edge_count(st) <= max(before_open, 0) and st.face_count() == out.face_count():
            return st
    return b


def job_trim(job):
    """Alias-style click-to-trim.

    The target is split into regions along every cutter: projected curves
    (parallel projection) or cutting surfaces (exact imprint of the
    intersection). Faces joined across original edges (a revolve seam) form
    one region. With scope THROUGH every region on the same side of the
    cutters as the clicked one is affected (a circle in front of a vase cuts
    both walls); with PICKED only the clicked region is."""
    ref = tuple(job["ref_point"])
    cutters = build_cutters(job, strict=job.get("strict", True))
    if not cutters:
        raise RuntimeError("no cutters")

    action = job["action"]
    through = job.get("scope", "THROUGH") == "THROUGH"
    no_cut = ("the cutters do not cut this surface into regions -- "
              "check that they cross it (along the projection direction for curves)")

    stored_side = job.get("side")   # history rebuild: the side of the cutters that was kept
    marks = job.get("marks") or []  # [[world point, keep?], ...]: regions marked in the viewport

    def marked(faces, groups, picked, sides):
        """Groups to KEEP from the viewport marks: each mark labels the region
        nearest to it; a region no mark reaches (the cut changed) follows the
        side rule of the clicked one."""
        np = _np()
        owner = {i: gi for gi, g in enumerate(groups) for i in g}
        label, by_side = {}, {}
        none = tuple(None for _ in sides[picked])
        for mk in marks:
            pt, keep = mk[0], bool(mk[1])
            side = tuple(mk[2]) if len(mk) > 2 and mk[2] is not None else None
            if side is not None and side != none:
                by_side[side] = by_side.get(side, False) or keep
            q = tuple(float(x) for x in pt)
            fi = min(range(len(faces)), key=lambda i: faces[i].closest_point(q)[1])
            gi = owner[fi]
            if side is not None and sides[gi] != side:
                continue           # the cut changed (Cut flipped): that point now lies in another kind of region
            label.setdefault(gi, keep)
        same = {gi for gi, s in enumerate(sides) if s == sides[picked]} if sides[picked] != none else {picked}

        def default(gi):
            return by_side[sides[gi]] if sides[gi] in by_side else gi in same
        return {gi for gi in range(len(groups)) if label.get(gi, default(gi))}

    def chosen(groups, picked, sides):
        if stored_side is not None and through and len(stored_side) == len(sides[0]) \
                and any(v is not None for v in stored_side):
            sel = {gi for gi, s in enumerate(sides) if list(s) == list(stored_side)}
            if sel:
                return sel
        if not through or sides[picked] == tuple(None for _ in sides[picked]):
            return {picked}
        return {gi for gi, s in enumerate(sides) if s == sides[picked]}

    results, info = {}, {}
    for act in (["KEEP", "DELETE"] if action == "SPLIT" else [action]):
        b, faces, groups, picked, sides = _regions(job, cutters)
        info = {"regions": len(groups), "side": list(sides[picked])}
        if marks:   # explicit: KEEP writes the regions marked keep, DELETE the others
            keep = marked(faces, groups, picked, sides)
            if len(groups) < 2:
                raise RuntimeError(no_cut)
            if not keep:
                raise RuntimeError("every region is marked discard")
            out_g = keep if act == "KEEP" else set(range(len(groups))) - keep
            if not out_g:
                raise RuntimeError("no region is marked discard -- nothing to split off")
            info["selected"] = len(keep)
            drop = [f for i, f in enumerate(faces) if not any(i in groups[gi] for gi in out_g)]
            if drop:
                ext.delete_faces(b, drop)
            results[act] = heal_cut_seams(b, world_brep(job, "target"))
            continue
        sel = chosen(groups, picked, sides)
        info["selected"] = len(sel)
        sel_faces = {i for gi in sel for i in groups[gi]}
        if len(groups) < 2 and act == "DELETE":
            raise RuntimeError(no_cut)
        if act == "KEEP":
            drop = [f for i, f in enumerate(faces) if i not in sel_faces]
            if action == "SPLIT" and not drop:
                raise RuntimeError(no_cut)
        else:
            drop = [f for i, f in enumerate(faces) if i in sel_faces]
            if len(drop) == len(faces):
                raise RuntimeError("that would remove the whole surface")
        if drop:
            ext.delete_faces(b, drop)
        results[act] = heal_cut_seams(b, world_brep(job, "target"))
    out = {"info": info, "extended": job.get("extended", [])}
    for act, b in results.items():
        save(to_local(b, job.get("xforms", {}).get("target")), job["out"][act])
        out[act] = {"faces": b.face_count(), "area": b.area()}
    return out


def _foot(surf, q):
    np = _np()
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    best = min(((np.linalg.norm(np.array(surf.evaluate(u, v)) - q), u, v)
                for u in np.linspace(u0, u1, 17) for v in np.linspace(v0, v1, 17)))
    _, u, v = best
    for _ in range(40):
        pt, su, sv = surf.derivatives(u, v)
        r = np.array(pt) - q
        step = np.linalg.lstsq(np.array([su, sv]).T, -r, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        u = min(max(u + step[0], u0), u1)
        v = min(max(v + step[1], v0), v1)
        if np.linalg.norm(step) < 1e-13:
            break
    return u, v


def _rail_sheet(edge, base, h):
    """Degree-1 x rail sheet whose middle line is EXACTLY the rail curve."""
    return _normal_sheet(ext.curve_cvs(edge.curve()), base, h)


def _normal_sheet(d, base, h):
    """Degree-1 x curve sheet whose middle line is EXACTLY the curve (data d):
    rows = curve CVs -+ h * (base normal at the CV's Greville point), same
    weights. A cutter for a curve on (or near) the base: it stands across the
    base along its normals."""
    np = _np()
    p, U = int(d["degree"]), [float(k) for k in d["knots"]]
    P = np.array(d["points"], float)
    W = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    c = curve(d)
    faces = base.faces()
    N = []
    for i in range(len(P)):
        g = sum(U[i + 1:i + p + 1]) / p
        q = np.array(c.evaluate(g), float)
        f = min(faces, key=lambda fc: fc.closest_point(tuple(q))[1])
        s = f.surface()
        u, v = _foot(s, q)
        _, su, sv = s.derivatives(u, v)
        n = np.cross(su, sv)
        N.append(n / (np.linalg.norm(n) or 1.0))
    N = np.array(N)
    for i in range(1, len(N)):                      # keep the normals on one side
        if N[i] @ N[i - 1] < 0:
            N[i] = -N[i]
    rows = [P - h * N, P + h * N]
    pts = [tuple(map(float, r[i])) for r in rows for i in range(len(P))]
    wts = [float(w) for _ in rows for w in W]
    rational = np.abs(W - 1.0).max() > 1e-12
    srf = ext.surface_from_cvs(1, p, pts, [0.0, 0.0, 1.0, 1.0], U, wts if rational else None)
    return ext.brep_from_surface(srf)


def _insert_knot(Pw, U, p, t, times):
    """Boehm knot insertion (homogeneous CVs)."""
    np = _np()
    Pw = [np.array(x, float) for x in Pw]
    U = list(U)
    for _ in range(times):
        k = max(i for i in range(len(U) - 1) if U[i] <= t < U[i + 1])
        Q = []
        for i in range(len(Pw) + 1):
            if i <= k - p:
                Q.append(Pw[i])
            elif i > k:
                Q.append(Pw[i - 1])
            else:
                a = (t - U[i]) / (U[i + p] - U[i]) if U[i + p] != U[i] else 0.0
                Q.append((1 - a) * Pw[i - 1] + a * Pw[i])
        Pw, U = Q, U[:k + 1] + [t] + U[k + 1:]
    return Pw, U


def _is_closed(surf, direction, tol=1e-7):
    np = _np()
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    for f in (0.2, 0.5, 0.8):
        if direction == "u":
            a, b = surf.evaluate(u0, v0 + f * (v1 - v0)), surf.evaluate(u1, v0 + f * (v1 - v0))
        else:
            a, b = surf.evaluate(u0 + f * (u1 - u0), v0), surf.evaluate(u0 + f * (u1 - u0), v1)
        if np.linalg.norm(np.array(a) - np.array(b)) > tol * (1 + np.linalg.norm(np.array(a))):
            return False
    return True


def _reseamed(surf, direction, frac):
    """The same closed surface with its seam moved to `frac` of the periodic
    direction (exact: knot insertion to a full break, CVs reordered)."""
    np = _np()
    d = ext.surface_cvs(surf)
    nu, nv = d["count_u"], d["count_v"]
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    W = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    G = np.concatenate([P * W[..., None], W[..., None]], axis=2)
    if direction == "u":
        G = G.transpose(1, 0, 2)
        U, p = list(d["knots_u"]), d["degree_u"]
    else:
        U, p = list(d["knots_v"]), d["degree_v"]
    a, b = U[0], U[-1]
    t = a + frac * (b - a)
    have = sum(1 for k in U if abs(k - t) < 1e-12 * (b - a))
    if have:
        t = [k for k in U if abs(k - t) < 1e-12 * (b - a)][0]
    rows, newU = [], None
    for r in G:
        Q, newU = _insert_knot(list(r), U, p, t, max(0, p - have))
        rows.append(Q)
    rows = np.array(rows)
    idx = max(i for i, k in enumerate(newU) if k == t) - p
    R = np.concatenate([rows[:, idx:, :], rows[:, 1:idx + 1, :]], axis=1)
    eps = 1e-12 * (b - a)
    K = ([0.0] * (p + 1) + [k - t for k in newU if t + eps < k < b - eps] + [b - t] * p
         + [k - a + (b - t) for k in newU if a + eps < k < t - eps] + [b - a] * (p + 1))
    K = [k / (b - a) for k in K]
    if direction == "u":
        R = R.transpose(1, 0, 2)
    n1, n2 = R.shape[:2]
    w = R[..., 3]
    pts = [tuple(map(float, R[i, j, :3] / w[i, j])) for i in range(n1) for j in range(n2)]
    Ku = K if direction == "u" else [k / (d["knots_u"][-1] - d["knots_u"][0]) for k in d["knots_u"]]
    Kv = K if direction == "v" else [k / (d["knots_v"][-1] - d["knots_v"][0]) for k in d["knots_v"]]
    rational = np.abs(w - 1.0).max() > 1e-12
    return ext.surface_from_cvs(d["degree_u"], d["degree_v"], pts, Ku, Kv,
                                [float(x) for x in w.reshape(-1)] if rational else None)


def _surface_variant(job, key, idx):
    """Fresh surface variant idx of a job brep's face: 0 = as is, then the
    surface with the seam of a closed direction moved; None when exhausted."""
    s = world_brep(job, key).faces()[0].surface()
    if idx == 0:
        return s
    k = 1
    for direction in ("u", "v"):
        if _is_closed(s, direction):
            for frac in (0.5, 0.25, 0.75):
                if k == idx:
                    return _reseamed(s, direction, frac)
                k += 1
    return None

def _orient_fillet(f, bases):
    """The kernel's fillet face can face either way (with Flip A / B on it faces
    into the corner). Shading, matcaps and a later Join want it to face the same
    way as the surfaces it connects: along each rail the fillet is tangent to a
    base, so its normal there is plus or minus that base's. If most samples
    disagree, the face is rebuilt with v reversed. Only for a one-face fillet
    whose trim is its whole parameter domain (the surface alone describes it);
    returns the fillet unchanged otherwise or when anything fails."""
    np = _np()
    try:
        if f.face_count() != 1:
            return f
        face = f.faces()[0]
        s = face.surface()
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        (fu0, fv0), (fu1, fv1) = face.uv_domain()
        span = max(u1 - u0, v1 - v0)
        if max(abs(fu0 - u0), abs(fu1 - u1), abs(fv0 - v0), abs(fv1 - v1)) > 1e-9 * span:
            return f
        vote = 0.0
        for v in (v0, v1):
            for x in np.linspace(0.05, 0.95, 9):
                u = u0 + (u1 - u0) * x
                q = np.array(s.evaluate(u, v), float)
                n = np.array(face.normal_at_uv(u, v), float)
                best = None
                for base in bases:
                    for bf in base.faces():
                        cp, uvs, d = sm.surface_closest_point(bf.surface(), tuple(q))
                        if uvs and (best is None or d < best[0]):
                            best = (d, bf, uvs[0])
                if best is None:
                    continue
                nb = np.array(best[1].outward_normal_at_uv(best[2].x, best[2].y), float)
                vote += float(n @ nb) / max(np.linalg.norm(n) * np.linalg.norm(nb), 1e-300)
        if vote >= 0:
            return f
        d = ext.surface_cvs(s)
        nu, nv = d["count_u"], d["count_v"]
        P = np.array(d["points"], float).reshape(nu, nv, 3)[:, ::-1]
        w = np.array(d["weights"], float).reshape(nu, nv)[:, ::-1] if d.get("weights") else None
        Kv = list(d["knots_v"])
        Kv = [Kv[0] + Kv[-1] - k for k in reversed(Kv)]
        pts = [tuple(map(float, P[i, j])) for i in range(nu) for j in range(nv)]
        r = ext.surface_from_cvs(d["degree_u"], d["degree_v"], pts, list(d["knots_u"]), Kv,
                                 [float(x) for x in w.reshape(-1)] if w is not None else None)
        g = ext.brep_from_surface(r)
        if abs(g.area() - f.area()) > 1e-2 * max(f.area(), 1e-300):   # (the kernel's areas are approximate)
            return f
        return g
    except Exception:
        return f


def job_fillet(job):
    """Surface fillet. A rolling-ball circle or chamfer: the kernel's fillet.
    With job['chord'] (the rails that far apart, the ball radius changing
    with the angle between the surfaces) or job['blend'] (1 / 2 / 3: a G1 / G2 /
    G3 blend between the rolling ball's rails) or job['soft'] (the smoothing
    of a soft fillet, softfillet.py: radius_a is its minimum radius, blend its
    continuity 2 / 3): built from exact sections (secfillet.py). Either way the
    bases are trimmed back to its rails. job['ends'] EXTEND: where one surface
    ends along the crossing before the other, it is grown to the other's end
    first (filletends.py); NOTCH (default): the longer one is cut back only
    along the fillet."""
    grown = []
    if job.get("ends") == "EXTEND":
        job, grown = _grow_shorter(job)
    A = world_brep(job, "a_base" if "a_base" in job["breps"] else "a")
    B = world_brep(job, "b_base" if "b_base" in job["breps"] else "b")
    res = _job_fillet(job, A, B)
    if grown:
        res["extended"] = [{"surface": r["key"], "length": r["grown"]} for r in grown]
    return res


def _grow_shorter(job):
    """(job, records): the job with the shorter surface grown to the longer
    one's end wherever the crossing ends before the longer one does (its base
    and its trim target both); unchanged with no records."""
    import filletends
    ka = "a_base" if "a_base" in job["breps"] else "a"
    kb = "b_base" if "b_base" in job["breps"] else "b"
    A2, B2, recs = filletends.extend_to_partner(sm, world_brep(job, ka), world_brep(job, kb))
    if not recs:
        return job, recs
    job = dict(job, breps=dict(job["breps"]), xforms=dict(job.get("xforms") or {}))
    folder = job.get("out_dir") or os.path.dirname(job["breps"][ka])
    for key, base in (("a", A2), ("b", B2)):
        if not any(r["key"] == key for r in recs):
            continue
        bk = key + "_base" if key + "_base" in job["breps"] else key
        target = base
        if bk != key:           # a history replay: the trim target is the current geometry
            try:
                target = filletends.apply_records(sm, world_brep(job, key), recs, key)
            except Exception:   # already grown (the partner's own step ran): leave it
                target = world_brep(job, key)
        for k, br in ((bk, base), (key, target)):
            path = os.path.join(folder, f"grown_{k}.sm")
            save(to_local(br.copy(), job["xforms"].get(k)), path)
            job["breps"][k] = path
        if key in job.get("out", {}):        # shown grown even when it is not trimmed (or the trim fails)
            save(to_local(target.copy(), job["xforms"].get(key)), job["out"][key])
    return job, recs


def _job_fillet(job, A, B):
    chord = job.get("chord")
    blend = int(job.get("blend") or 0)
    soft = job.get("soft")
    if soft is not None:
        import secfillet
        sa = 1.0 if float(job["radius_a"]) >= 0 else -1.0
        sb = 1.0 if float(job["radius_b"]) >= 0 else -1.0
        shape = "SOFT3" if blend >= 3 else "SOFT2"
        R = abs(float(job["radius_a"]))
        f, info = secfillet.build(sm, A, B, sa, sb, R, R, shape=shape, smoothing=float(soft),
                                  tolerance=float(job.get("tolerance", 1e-4)))
        f = _orient_fillet(f, (A, B))
        job = dict(job, radius_a=sa * info["radius"][2], radius_b=sb * info["radius"][2])   # (trim sheet size)
        res = _fillet_finish(job, f, A, B)
        res["radius"] = info["radius"][1]
        res["radius_range"] = info["radius"]
        res["chord"] = info["chord"]
        res["sections"] = info["sections"]
        res["min_radius"] = info["min_radius"]
        res["smoothing"] = info["smoothing"]
        if info.get("note"):
            res["note"] = info["note"]
        return res
    radii = None if chord else job.get("radii")
    # the kernel's circular fillet takes only one radius (Radius A != B fails
    # in it: SM_ERR_NULL_POINTER); its chamfer takes two
    two = int(job["xsect"]) != 0 and abs(abs(float(job["radius_a"])) - abs(float(job["radius_b"]))) > \
        1e-9 * max(abs(float(job["radius_a"])), abs(float(job["radius_b"])))
    if chord or blend or radii or two:
        # our own sections (secfillet.py): the kernel's fillet has one ball
        # radius, a chord or a variable radius (job['radii'] = start, end;
        # start at the crossing's end nearer job['start_point']) needs one
        # per section; Radius A != B: the conic between the ball's feet
        # (secfillet.conic); a blend is built between the rails directly (no
        # circular fillet, trim and loft between its edges)
        import secfillet
        sa = 1.0 if float(job["radius_a"]) >= 0 else -1.0
        sb = 1.0 if float(job["radius_b"]) >= 0 else -1.0
        shape = {1: "G1", 2: "G2", 3: "G3"}.get(blend) or ("LINEAR" if int(job["xsect"]) == 0 else "CIRCULAR")
        f, info = secfillet.build(sm, A, B, sa, sb, abs(float(job["radius_a"])), abs(float(job["radius_b"])),
                                  chord=float(chord) if chord else None, shape=shape,
                                  tolerance=float(job.get("tolerance", 1e-4)),
                                  radii=[float(r) for r in radii] if radii else None,
                                  start=job.get("start_point"))
        f = _orient_fillet(f, (A, B))
        res = _fillet_finish(job, f, A, B)
        res["radius"] = info["radius"][1]
        res["radius_range"] = info["radius"]
        res["chord"] = info["chord"]
        res["sections"] = info["sections"]
        if info.get("note"):
            res["note"] = info["note"]
        return res
    import filletends
    if filletends.uneven_ends(sm, A, B):
        # one surface ends along the crossing before the other: the kernel's
        # ball runs on past that end in a tapering one-sided tail; the same
        # circle / chamfer from exact sections ends square to the crossing
        # there. (Where a section's ball does not fit -- a crossing leaving a
        # surface through an edge at an angle -- the kernel's fillet as before.)
        import secfillet
        sa = 1.0 if float(job["radius_a"]) >= 0 else -1.0
        sb = 1.0 if float(job["radius_b"]) >= 0 else -1.0
        try:
            f, _ = secfillet.build(sm, A, B, sa, sb, abs(float(job["radius_a"])), abs(float(job["radius_b"])),
                                   shape="LINEAR" if int(job["xsect"]) == 0 else "CIRCULAR",
                                   tolerance=float(job.get("tolerance", 1e-4)))
            return _fillet_finish(job, _orient_fillet(f, (A, B)), A, B)
        except Exception:
            pass
    return _fillet_finish(job, _surface_fillet(job, A, B), A, B)


def _surface_fillet(job, A, B):
    """The kernel fillet face between the bases (world space)."""
    # optional *_base breps: surfaces to compute the fillet from (untrimmed);
    # "a" / "b" are what gets trimmed
    f, last = None, None
    key_a = "a_base" if "a_base" in job["breps"] else "a"
    key_b = "b_base" if "b_base" in job["breps"] else "b"
    done = False
    for ia in range(7):
        if done or (ia and _surface_variant(job, key_a, ia) is None):
            break
        for ib in range(7):
            fa, fb = _surface_variant(job, key_a, ia), _surface_variant(job, key_b, ib)
            if fa is None or fb is None:
                break
            try:
                f = sm.surface_surface_fillet(
                    fa, fb, radius1=job["radius_a"], radius2=job["radius_b"], tolerance=job.get("tolerance", 1e-4),
                    xsect_type=job["xsect"], mirror=False, complement=False)
            except Exception as exc:
                f, last = None, exc
            if f is not None and f.face_count() > 0:
                done = True
                break
            if ia == 0 and ib == 0 and last is None:
                # a plain "no fillet on these sides": moving seams will not help
                raise RuntimeError("no fillet found on these sides (try Flip A / Flip B)")
    if f is None or f.face_count() == 0:
        if last is not None:
            raise last
        raise RuntimeError("no fillet found on these sides (try Flip A / Flip B)")
    return _orient_fillet(f, (A, B))


def _fillet_finish(job, f, A, B):
    """Write the fillet; trim the bases back to its rails (trim_bases)."""
    save(f, job["out"]["fillet"])
    out = {"fillet": {"faces": f.face_count(), "area": f.area()}, "trim_errors": {}}

    if job.get("trim_bases"):
        # rails = fillet boundary edges lying on each base
        edges = f.edges()
        rail = {}
        # (not an end collapsed to a point: a variable radius tapering to 0)
        span = lambda e: _np().linalg.norm(_np().subtract(e.point_at(e.parameter_range()[0]), mid(e)))
        edges = [e for e in edges if span(e) > 1e-9] or edges
        for key, base in (("a", A), ("b", B)):
            rail[key] = min(edges, key=lambda e: dist_to_brep(mid(e), base))
        rmax = max(abs(float(job["radius_a"])), abs(float(job["radius_b"])))
        import filletends
        size = filletends.size_of(A, B)
        ends = filletends.crossing_ends(sm, A, B, size)
        for key, other in (("a", "b"), ("b", "a")):
            try:
                base = world_brep(job, key)  # fresh: the fillet must not share it
                pa, pb = mid(rail[key]), mid(rail[other])
                probe = tuple(pa[i] + 0.5 * (pb[i] - pa[i]) for i in range(3))
                face = min(base.faces(), key=lambda fc: fc.closest_point(probe)[1])
                ref, d = face.closest_point(probe)  # on the base, between rail and seam
                # exact imprint along the rail (a sheet standing across the base;
                # a single-direction projection also hits the far side of closed
                # surfaces such as a revolve and deletes the wrong region)
                sheets = [_rail_sheet(rail[key], base, max(0.25 * rmax, 1e-4))]
                # a rail that stops inside the base (the other surface ends first
                # along the crossing): close the cut square to the crossing at its
                # end -- a notch, the rest of the base keeps its own edge
                t0, t1 = rail[key].parameter_range()
                for q in (rail[key].point_at(t0), rail[key].point_at(t1)):
                    end = filletends.notch_sheet(sm, base, q, ends, size, max(0.25 * rmax, 1e-4))
                    if end is not None:
                        sheets.append(end)
                import os, tempfile
                paths = {}
                for i, sh in enumerate(sheets):
                    fd, sp = tempfile.mkstemp(suffix=".sm")
                    os.close(fd)
                    save(sh, sp)
                    paths[f"sheet{i}"] = sp
                fd, op_ = tempfile.mkstemp(suffix=".sm")
                os.close(fd)
                sub = {"breps": dict(target=job["breps"][key], **paths),
                       "xforms": {"target": job.get("xforms", {}).get(key)},
                       "cutter_breps": list(paths), "ref_point": list(ref), "action": "DELETE", "scope": "PICKED",
                       "direction": [0.0, 0.0, 1.0], "out": {"DELETE": op_},
                       "extend": len(sheets) > 1}   # the notch's cuts reach across; the seams heal again
                job_trim(sub)
                import shutil
                shutil.move(op_, job["out"][key])
                for sp in paths.values():
                    os.remove(sp)
                t = load(job["out"][key])
                out[key] = {"faces": t.face_count(), "area": t.area()}
            except Exception as exc:
                out["trim_errors"][key] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Cutters that stop short: extended (naturally, exactly) until they reach across
# ---------------------------------------------------------------------------

EXTEND_STEPS = (0.05, 0.15, 0.4, 1.0, 2.0)     # fractions of the target's size, tried in turn


def _poly_dist(q, P):
    """Distance from point q to the polyline P (n, 3)."""
    np = _np()
    seg = P[1:] - P[:-1]
    t = np.clip(np.einsum("ij,ij->i", q - P[:-1], seg) / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-30), 0, 1)
    return float(np.min(np.linalg.norm(P[:-1] + seg * t[:, None] - q, axis=1)))


def _free_ends(target, cutter, edge_polys, tol):
    """End points of the intersection curves of target and cutter that lie
    inside the target (not on one of its edges): where the cut stops short
    and the cutter has to go on. None when they do not intersect at all."""
    np = _np()
    try:
        curves = sm.intersect_breps(target, cutter)[0]
    except Exception:
        return None
    if not curves:
        return None
    tips = []
    for c in curves:
        try:
            if c.is_closed():
                continue
            pts = [np.array(c.start_point(), float), np.array(c.end_point(), float)]
        except Exception:
            continue
        if np.linalg.norm(pts[0] - pts[1]) <= tol:
            continue                                   # closed after all
        tips += pts
    ends = []
    for i, q in enumerate(tips):
        # where two pieces of one cut meet (a loop split at a seam) the cut goes on
        if any(j != i and np.linalg.norm(q - r) <= tol for j, r in enumerate(tips)):
            continue
        if not any(_poly_dist(q, P) <= tol for P in edge_polys):
            ends.append(q)
    return ends


def _edge_polys(b):
    np = _np()
    return [np.array(p, float) for p in (_edge_poly(e, 64) for e in b.edges()) if len(p) > 1]


def _iso(s, edge, n=96):
    np = _np()
    (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
    ts = np.linspace(0.0, 1.0, n)
    if edge in ("U0", "U1"):
        u = u0 if edge == "U0" else u1
        return np.array([s.evaluate(u, v0 + (v1 - v0) * t) for t in ts], float)
    v = v0 if edge == "V0" else v1
    return np.array([s.evaluate(u0 + (u1 - u0) * t, v) for t in ts], float)


def _target_size(target):
    np = _np()
    lo, hi = (np.array(v, float) for v in target.bounding_box())
    return float(np.linalg.norm(hi - lo)) or 1.0


def extended_surface_cutter(target, cutter):
    """The cutting surface, extended at the boundaries where its cut stops
    inside the target until the cut reaches the target's edges (or the
    longest extension tried). A cutter that already cuts across, or that is
    not a single surface, comes back as it is. Returns (brep, edges extended)."""
    size = _target_size(target)
    tol = 1e-3 * size
    polys = _edge_polys(target)
    ends = _free_ends(target, cutter, polys, tol)
    if not ends or cutter.face_count() != 1:
        return cutter, []
    s0 = cutter.faces()[0].surface()      # a trimmed cutter: its whole surface first
    d0 = ext.surface_cvs(s0)
    best, need, s = cutter, set(), s0
    # a seam (the surface closes there) or a pole is no edge to extend
    isos = {e: _iso(s0, e) for e in ("U0", "U1", "V0", "V1")}
    np = _np()
    shut = set()
    for a, b in (("U0", "U1"), ("V0", "V1")):
        if np.max(np.linalg.norm(isos[a] - isos[b], axis=1)) <= tol:
            shut |= {a, b}
    for e, P in isos.items():
        if np.max(np.linalg.norm(P - P.mean(axis=0), axis=1)) <= tol:
            shut.add(e)
    open_edges = [e for e in ("U0", "U1", "V0", "V1") if e not in shut]
    if not open_edges:
        return cutter, []
    for f in EXTEND_STEPS:
        more = set()        # (an end at a crease or a knot of the cutter is no reason to extend)
        for q in ends:
            dist, e = min((_poly_dist(q, _iso(s, e)), e) for e in ("U0", "U1", "V0", "V1"))
            if dist <= tol and e in open_edges:
                more.add(e)
        if not more:
            break
        need |= more
        d, s = d0, None
        try:
            for e in sorted(need):
                s = ext.surface_from_cvs(*extend_grid(d, e, f * size, chunks=True))
                d = ext.surface_cvs(s)
        except Exception:
            break
        cand = ext.brep_from_surface(s)
        ends = _free_ends(target, cand, polys, tol)
        if ends is None:
            break
        best = cand
        if not ends:
            break
    return best, sorted(need)


def through_surface_cutter(target, cutter):
    """The cutting surface extended (its natural continuation) past every open
    edge until that edge lies outside the target's box, so it cuts all the
    way through (a short tube through both walls of a hood). Seams and poles
    stay. A cutter of several faces comes back as it is. Returns (brep, edges)."""
    np = _np()
    if cutter.face_count() != 1:
        return cutter, []
    size = _target_size(target)
    tol = 1e-3 * size
    lo, hi = (np.array(v) for v in target.bounding_box())
    lo, hi = lo - tol, hi + tol
    s0 = cutter.faces()[0].surface()
    isos = {e: _iso(s0, e) for e in ("U0", "U1", "V0", "V1")}
    shut = set()
    for a, b in (("U0", "U1"), ("V0", "V1")):
        if np.max(np.linalg.norm(isos[a] - isos[b], axis=1)) <= tol:
            shut |= {a, b}
    for e, P in isos.items():
        if np.max(np.linalg.norm(P - P.mean(axis=0), axis=1)) <= tol:
            shut.add(e)

    def outside(P):          # every point of the edge off the target's box
        return bool(np.all(np.any((P < lo) | (P > hi), axis=1)))
    d, s, done = ext.surface_cvs(s0), s0, []
    for e in ("U0", "U1", "V0", "V1"):
        if e in shut or outside(_iso(s, e)):
            continue
        for f in EXTEND_STEPS + (4.0,):
            try:
                cand = ext.surface_from_cvs(*extend_grid(d, e, f * size, chunks=True))
            except Exception:
                break
            if outside(_iso(cand, e)) or f == 4.0:
                d, s = ext.surface_cvs(cand), cand
                done.append(e)
                break
    if not done:
        return cutter, []
    return ext.brep_from_surface(s), done


def _curve_sheet(data, direction, L):
    np = _np()
    d = np.array(direction, float)
    c = sm.translate(curve(data), tuple(-0.5 * L * d))
    return ext.brep_from_surface(sm.create_extrude_surface(c, tuple(L * d)))


def _is_closed_data(data):
    np = _np()
    P = np.array(_poly(curve(data), 64))
    return np.linalg.norm(P[0] - P[-1]) <= 1e-6 * (np.ptp(P, axis=0).max() or 1.0)


def _curve_gap(target, data, n=33):
    """Largest distance from the curve to the target."""
    return max(dist_to_brep(tuple(q), target) for q in _poly(curve(data), n))


def extended_curve_data(target, data, sheets, direction=None):
    """An open cutting curve, extended at the ends where its cut stops inside
    the target, until it reaches the target's edges. sheets(data) gives the
    cutter sheets of a curve; direction: the projection (ends are compared
    across it), None for a curve cut along itself. Returns (data, ends
    extended: "start" / "end")."""
    np = _np()
    if _is_closed_data(data):
        return data, []                                  # closed: nothing to extend
    size = _target_size(target)
    tol = 1e-3 * size
    if direction is None:
        flat = lambda q: q
    else:
        d = np.array(direction, float)
        d /= np.linalg.norm(d) or 1.0
        flat = lambda q: q - (q @ d) * d                 # across the projection direction
    polys = _edge_polys(target)

    def free(dat):
        out, hit = [], False
        for sh in sheets(dat):
            e = _free_ends(target, sh, polys, tol)
            if e is not None:
                hit = True
                out += e
        return out if hit else None

    try:
        ends = free(data)
    except Exception:
        return data, []
    if not ends:
        return data, []
    best, need = data, set()
    P = np.array(_poly(curve(data), 64))
    a, b = P[0], P[-1]
    for f in EXTEND_STEPS:
        more = set()        # (an end at a corner of the curve is no reason to extend)
        for q in ends:
            da, db = np.linalg.norm(flat(q - a)), np.linalg.norm(flat(q - b))
            if min(da, db) <= tol:
                more.add("start" if da <= db else "end")
        if not more:
            break
        need |= more
        try:
            cand = data
            for which in sorted(need):
                cand = extend_curve(cand, which == "start", f * size, chunks=True)
            ends = free(cand)
        except Exception:
            break
        if ends is None:
            break
        best = cand
        Q = np.array(_poly(curve(cand), 64))
        a, b = Q[0], Q[-1]
        if not ends:
            break
    return best, sorted(need)


class PullCutter(SurfaceCutter):
    """A curve on (or near) the target, cut along itself -- no projection: a
    sheet standing across the target along its normals, the curve exactly in
    its middle (one sheet per smooth piece, as for projected curves). Its
    side is the side of the sheet, like a cutting surface's."""

    def __init__(self, target, data):
        np = _np()
        self.data = data
        self.sheets = pull_sheets(target, data)
        P, N = [], []
        for sh in self.sheets:
            arr = sh.tessellate(chord_height_tolerance=0.0005).to_numpy_arrays()
            P.append(arr["points"].astype(float))
            N.append(arr["normals"].astype(float))
        self.P, self.N = np.concatenate(P), np.concatenate(N)

    def split(self, b, ref):
        for sh in self.sheets:
            b = sm.boolean_with_options(b, sh.copy(), operation=sm.BooleanOp.UNION, imprinting=True,
                                        keep_other_brep=True)
            if b is None:
                return None
        return b

    def sheet_list(self, b):
        return [sh.copy() for sh in self.sheets]


def pull_sheets(target, data):
    """The sheets of a curve cut along itself: half height enough to pass
    through the target where the curve is off it, small enough not to reach
    other parts of it (the far wall of a vase)."""
    h = max(2.0 * _curve_gap(target, data), 0.02 * _target_size(target))
    return [_normal_sheet(piece, target, h) for piece in corner_pieces(data)]


def _pulled(target, c):
    """Is curve c cut along itself? c["pull"]: True, False or "auto" (when it
    lies on the target)."""
    mode = c.get("pull", False)
    if mode == "auto":
        try:
            return _curve_gap(target, c) <= 1e-3 * _target_size(target)
        except Exception:
            return False
    return bool(mode)


def build_cutters(job, strict):
    """Trim / split cutters: projected curves, curves cut along themselves
    (on the surface) and cutting surfaces. With job['extend'] a curve or
    surface whose cut stops inside the target is extended until it reaches
    across (job['extended'] lists what was). strict: a cutting surface that
    misses the target is an error (Trim), else it is left out (Split)."""
    extend = job.get("extend", False)
    target = world_brep(job, "target")
    cutters, extended = [], []
    for i, c in enumerate(job.get("curves", [])):
        if _pulled(target, c):
            if extend:
                data, ends = extended_curve_data(target, c, lambda dat: pull_sheets(target, dat))
                if ends:
                    c = data
                    extended.append({"curve": i, "ends": ends})
            cutters.append(PullCutter(target, c))
            continue
        direction = c.get("dir") or job["direction"]
        if extend:
            np = _np()
            P = np.array(_poly(curve(c), 64))
            lo, hi = (np.array(v) for v in target.bounding_box())
            L = 4.0 * (_target_size(target) + float(np.max(np.linalg.norm(P - 0.5 * (lo + hi), axis=1))))
            data, ends = extended_curve_data(target, c, lambda dat: [_curve_sheet(dat, direction, L)], direction)
            if ends:
                c = dict(data, dir=c.get("dir"))
                extended.append({"curve": i, "ends": ends})
        cutters.append(CurveCutter(c, direction, job.get("reach", "BOTH")))
    for key in job.get("cutter_breps", []):
        cut = world_brep(job, key)
        if not sm.intersect_breps(target, cut)[0]:
            if strict:
                raise RuntimeError("cutter surface does not intersect the target")
            continue
        brep = None
        if job.get("through"):      # all the way through the part
            brep, edges = through_surface_cutter(target, cut)
            if edges:
                extended.append({"surface": key, "edges": edges})
            else:
                brep = None
        elif extend:
            brep, edges = extended_surface_cutter(target, cut)
            if edges:
                extended.append({"surface": key, "edges": edges})
            else:
                brep = None
        cutters.append(SurfaceCutter(job, key, brep))
    job["extended"] = extended
    return cutters


def _inner_point(f, n=13):
    """A point well inside face f: of a grid of (u, v) the face calls inside,
    the one farthest from its edges. The kernel's interior_point() can lie on
    an edge (the top of a notch's arc)."""
    np = _np()
    try:
        (u0, v0), (u1, v1) = f.uv_domain()
        surf = f.surface()
        E = [np.array(p) for p in (_edge_poly(e, 64) for e in f.edges()) if len(p)]
        E = np.vstack(E) if E else None
        best, best_d = None, -1.0
        for u in np.linspace(u0, u1, n + 2)[1:-1]:
            for v in np.linspace(v0, v1, n + 2)[1:-1]:
                if f.classify_uv(float(u), float(v)) != "inside":
                    continue
                p = np.array(surf.evaluate(float(u), float(v)))
                d = float(np.min(np.linalg.norm(E - p, axis=1))) if E is not None else 0.0
                if d > best_d:
                    best, best_d = p, d
        if best is not None:
            return best
    except Exception:
        pass
    return np.array(f.interior_point())


def _untrim_regions(job):
    """The surface as it was before its trims ("untrimmed") cut along the
    trimmed one's ("target") cut edges -- the edges that are not the
    untrimmed surface's own boundary. Returns (b, faces, groups, present,
    removed, untrimmed, current, own-edge test): present groups are on the
    trimmed surface, removed ones were trimmed away."""
    np = _np()
    cur = world_brep(job, "target")
    unt = world_brep(job, "untrimmed")
    lo, hi = (np.array(v) for v in unt.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    tol = 1e-5 * size
    on_own = _near_edges(unt, 10 * tol)
    cut_edges = []
    for e in cur.edges():
        pts = _edge_poly(e, 9)
        if pts and not all(on_own(p) for p in np.array(pts)[1:-1]):
            cut_edges.append(e)
    if not cut_edges:
        raise RuntimeError("nothing is trimmed on this surface")
    cutters = []
    for e in cut_edges:
        d = ext.curve_cvs(e.curve())
        data = {"degree": d["degree"], "points": [list(p) for p in d["points"]], "knots": list(d["knots"]),
                "weights": list(d["weights"]) if d.get("weights") else None}
        cutters.append(PullCutter(unt, data))
    lo2, hi2 = (np.array(v) for v in cur.bounding_box())
    sub = dict(job, ref_point=list(job.get("point") or (0.5 * (lo2 + hi2)).tolist()))
    sub["breps"] = dict(job["breps"], target=job["breps"]["untrimmed"])
    sub["xforms"] = dict(job.get("xforms", {}), target=job.get("xforms", {}).get("untrimmed"))
    b, faces, groups, _, _ = _regions(sub, cutters)
    cur_faces = cur.faces()

    def on_current(p):
        """Is p (on the untrimmed surface) inside the trimmed one? The face's own
        trim test on the surface parameters -- a distance test called a removed
        area present when its sample point lay near the cut edge."""
        q = tuple(float(v) for v in p)
        for f in cur_faces:
            try:
                _, uvs, d = sm.surface_closest_point(f.surface(), q)
                if d <= 100 * tol and any(f.classify_uv(float(uv.x), float(uv.y)) in ("inside", "boundary")
                                          for uv in uvs):
                    return True
            except Exception:
                if f.closest_point(q)[1] <= 10 * tol:
                    return True
        return False
    present, removed = [], []
    for gi, g in enumerate(groups):
        f = max((faces[i] for i in g), key=lambda fc: fc.area(1e-3))
        (present if on_current(_inner_point(f)) else removed).append(gi)
    area_present = sum(faces[i].area(1e-6) for gi in present for i in groups[gi])
    if abs(area_present - cur.area(1e-6)) > 1e-3 * max(cur.area(1e-6), 1e-12):
        raise RuntimeError("this surface has more on it than trims (a fillet?) -- use Untrim for all of it")
    if not removed:
        raise RuntimeError("no trimmed-away area found")
    return b, faces, groups, present, removed, unt, cur, on_own


def job_untrim_preview(job):
    """The areas trimmed away (out_region<i>.sm, the untrimmed body's local
    space) and a world point inside each, for picking them in the viewport."""
    b, faces, groups, present, removed, unt, cur, _ = _untrim_regions(job)
    x = job.get("xforms", {}).get("untrimmed")
    points = []
    for k, gi in enumerate(removed):
        part = b.copy()
        pf = part.faces()
        ext.delete_faces(part, [f for i, f in enumerate(pf) if i not in groups[gi]])
        save(to_local(part, x), os.path.join(job["out_dir"], f"out_region{k}.sm"))
        f = max((faces[i] for i in groups[gi]), key=lambda fc: fc.area(1e-3))
        points.append([float(v) for v in _inner_point(f)])
    return {"regions": len(removed), "points": points}


def job_untrim_region(job):
    """Untrim areas: the trimmed-away areas nearest job["points"] (world;
    or job["point"], a point on a cut edge) come back; the cut edges between
    them and the rest are merged away. The other trims stay."""
    np = _np()
    pts = job.get("points") or [job["point"]]
    if job.get("point") is not None and not job.get("points"):
        # a click on an edge: it must be a cut edge, not the surface's own
        cur = world_brep(job, "target")
        unt = world_brep(job, "untrimmed")
        lo, hi = (np.array(v) for v in unt.bounding_box())
        on_own = _near_edges(unt, 1e-4 * (float(np.linalg.norm(hi - lo)) or 1.0))
        q = np.array(job["point"], float)
        best = min((e for e in cur.edges() if _edge_poly(e, 9)),
                   key=lambda e: float(np.min(np.linalg.norm(np.array(_edge_poly(e, 9)) - q, axis=1))))
        if all(on_own(p) for p in np.array(_edge_poly(best, 9))[1:-1]):
            raise RuntimeError("that edge is the surface's own boundary: nothing was trimmed there")
    b, faces, groups, present, removed, unt, cur, on_own = _untrim_regions(job)
    back = set()
    for pt in pts:
        q = tuple(float(v) for v in pt)
        back.add(min(removed, key=lambda gi: min(faces[i].closest_point(q)[1] for i in groups[gi])))
    keep = set(present) | back
    drop = [f for i, f in enumerate(faces) if not any(i in groups[gi] for gi in keep)]
    if drop:
        ext.delete_faces(b, drop)
    b = heal_cut_seams(b, unt)
    left = sum(1 for e in b.edges() if _edge_poly(e, 9)       # (world, as on_own: before to_local moves b)
               and not all(on_own(p) for p in np.array(_edge_poly(e, 9))[1:-1]))
    n_faces, area = b.face_count(), b.area(1e-6)
    save(to_local(b, job.get("xforms", {}).get("untrimmed")), job["out"]["result"])
    return {"faces": n_faces, "area": area, "cuts_left": left,
            "restored": float(sum(faces[i].area(1e-6) for gi in back for i in groups[gi]))}


def _solid_caps(solid, cutters):
    """Each cutter sheet cut down to its part inside the solid, split along
    the other cutters' sheets: the faces that close the pieces of a split."""
    per = [c.sheet_list(solid) for c in cutters]
    caps = []
    for ci, sheets in enumerate(per):
        for sh in sheets:
            try:
                cap = sm.boolean_with_options(sh, solid.copy(), operation=sm.BooleanOp.INTERSECTION)
            except Exception:
                cap = None
            if cap is None or cap.face_count() == 0:
                continue
            for cj, others in enumerate(per):
                if cj == ci:
                    continue
                for o in others:
                    cut = sm.boolean_with_options(cap, o.copy(), operation=sm.BooleanOp.UNION, imprinting=True,
                                                  keep_other_brep=True)
                    if cut is not None and cut.face_count():
                        cap = cut
            caps.append(cap)
    return caps


def _solid_cells(b, caps, size):
    """The closed pieces of a solid cut along sheets. b: the solid imprinted
    along every cut (outward normals); caps: the sheets' parts inside it.
    Round every edge, the faces meeting there (b's and the caps') are put in
    angular order; each wedge between two neighbours joins the sides of the
    two faces that look into it. A b face's outer side is the outside; a
    group of face sides that never reaches it is a piece. Returns
    [(b face indices, {cap index: face indices})]; a cap face with the same
    piece on both sides (a cut that stops inside the body) is left out."""
    np = _np()
    F = [(f, None) for f in b.faces()]                 # (face, cap index or None for b)
    for k, cap in enumerate(caps):
        F += [(f, k) for f in cap.faces()]
    nb = b.face_count()
    delta, near = 2e-3 * size, 1e-4 * size
    E = []                                             # every face's edges: (face index, edge, polyline)
    for gi, (f, _) in enumerate(F):
        for e in f.edges():
            P = np.array(_edge_poly(e, 24), float)
            if len(P) >= 2 and e.length() > near:
                E.append((gi, e, P))

    def normal(gi, q):
        f, k = F[gi]
        _, uvs, _ = sm.surface_closest_point(f.surface(), tuple(map(float, q)))
        u, v = uvs[0].x, uvs[0].y
        n = np.array(f.outward_normal_at_uv(u, v) if k is None else f.normal_at_uv(u, v), float)
        return n / (np.linalg.norm(n) or 1.0)

    def into(gi, m, t):
        """Unit direction from m into face gi, square to the edge (None: both ways, a seam)."""
        n = normal(gi, m)
        d = np.cross(n, t)
        d /= np.linalg.norm(d) or 1.0
        f = F[gi][0]
        ins = [f.closest_point(tuple(map(float, m + s * delta * d)))[1] < 0.3 * delta for s in (1.0, -1.0)]
        if ins[0] == ins[1]:
            return None, n
        return (d if ins[0] else -d), n

    OUT = 2 * len(F)
    parent = list(range(OUT + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, c):
        parent[find(a)] = find(c)

    def node(gi, side):
        if F[gi][1] is None and side > 0:
            return OUT
        return 2 * gi + (0 if side > 0 else 1)

    for gi, e, P in E:
        t0, t1 = e.parameter_range()
        tm = 0.5 * (t0 + t1)
        m = np.array(e.point_at(tm), float)
        t = np.array(e.tangent_at(tm), float)
        t /= np.linalg.norm(t) or 1.0
        ring = {gi}
        for gj, e2, P2 in E:
            if gj in ring or np.min(np.linalg.norm(P2 - m, axis=1)) > 0.5 * float(
                    np.max(np.linalg.norm(np.diff(P2, axis=0), axis=1))) + near:
                continue
            if e2.closest_point(tuple(map(float, m)))[2] <= near:
                ring.add(gj)
        ent = []
        for gj in ring:
            d, n = into(gj, m, t)
            if d is not None:
                ent.append((gj, d, n))
        if len(ent) < len(ring) or not ent:
            continue                                   # a seam: the face meets itself
        a = ent[0][1]
        c = np.cross(t, a)
        ent.sort(key=lambda x: float(np.arctan2(x[1] @ c, x[1] @ a)))
        for i in range(len(ent)):
            (ga, da, na), (gb, db, nb_) = ent[i], ent[(i + 1) % len(ent)]
            sa = 1 if float(na @ np.cross(t, da)) > 0 else -1           # the wedge turns on from A ...
            sb = 1 if float(nb_ @ -np.cross(t, db)) > 0 else -1         # ... and ends at B
            union(node(ga, sa), node(gb, sb))
    out_root = find(OUT)
    cells = {}
    for gi in range(len(F)):
        for side in (1, -1):
            nd = node(gi, side)
            if nd == OUT or find(nd) == out_root:
                continue
            cells.setdefault(find(nd), {}).setdefault(gi, set()).add(side)
    result = []
    for members in cells.values():
        bf = sorted(gi for gi in members if gi < nb)
        if not bf:
            continue
        cf = {}
        for gi, sides in members.items():
            if gi >= nb and len(sides) == 1:
                k = F[gi][1]
                cf.setdefault(k, []).append(gi - nb - sum(caps[j].face_count() for j in range(k)))
        result.append((bf, cf))
    return result


def _cell_brep(b, caps, cell, tol):
    """One piece of a split solid: its faces of b and of the caps stitched
    into a closed body."""
    bf, cf = cell
    part = b.copy()
    keep = set(bf)
    drop = [f for i, f in enumerate(part.faces()) if i not in keep]
    if drop:
        ext.delete_faces(part, drop)
    for k, idx in cf.items():
        cp = caps[k].copy()
        keep = set(idx)
        drop = [f for i, f in enumerate(cp.faces()) if i not in keep]
        if drop:
            ext.delete_faces(cp, drop)
        ext.append_brep(part, cp)
    res = sm.stitch_advanced(part, tol, squeeze_small_edges=False, making_manifold_solid=True)
    out = res[0]
    try:
        sm.unify_normals(out, max(out.faces(), key=lambda f: f.area()), 100)
    except Exception:
        pass
    try:
        h = sm.heal_brep(out)
        if h is not None and h.is_manifold_solid() and abs(h.volume(1e-6)) > 0:
            out = h
    except Exception:
        pass
    return out


def _job_split_solid(job, cutters, solid):
    """job_split for a closed solid: every piece a closed solid of its own."""
    np = _np()
    lo, hi = (np.array(v, float) for v in solid.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    b = solid.copy()
    for c in cutters:
        b = c.split(b, tuple(job["ref_point"]))
        if b is None or b.face_count() == 0:
            raise RuntimeError("the cut produced nothing (does the cutter cross the solid?)")
    caps = _solid_caps(solid, cutters)
    cells = _solid_cells(b, caps, size)
    if len(cells) < 2:
        raise RuntimeError("the cutters do not split this solid into separate pieces")
    tol = 1e-4 * size
    x = job.get("xforms", {}).get("target")
    keep = job.get("keep")
    if keep is not None:                    # the marked pieces joined into one body
        chosen = [cells[i] for i, k in enumerate(keep) if k and i < len(cells)]
        bf = sorted({i for c in chosen for i in c[0]})
        count = {}
        for c in chosen:
            for k, idx in c[1].items():
                for i in idx:
                    count[(k, i)] = count.get((k, i), 0) + 1
        cf = {}
        for (k, i), n in count.items():
            if n == 1:                      # a cap between two kept pieces goes
                cf.setdefault(k, []).append(i)
        part = _cell_brep(b, caps, (bf, cf), tol)
        try:              # no seam where the dropped wall met the faces (as Trim's merge)
            healed = heal_cut_seams(part, solid)
            if healed is not part and healed.is_manifold_solid():
                part = healed
        except Exception:
            pass
        save(to_local(part, x), job["out"]["KEEP"] if "KEEP" in job.get("out", {})
             else os.path.join(job["out_dir"], "out_KEEP.sm"))
        return {"regions": len(cells), "solid": True, "closed": bool(part.is_manifold_solid())}
    areas, points, closed, volumes = [], [], [], []
    for gi, cell in enumerate(cells):
        part = _cell_brep(b, caps, cell, tol)
        f = max(part.faces(), key=lambda fc: fc.area(1e-3))
        points.append([float(v) for v in _inner_point(f)])
        ok = bool(part.is_manifold_solid())
        closed.append(ok)
        try:
            volumes.append(abs(float(part.volume(1e-6))) if ok else 0.0)
        except Exception:
            volumes.append(0.0)
        areas.append(part.area())
        save(to_local(part, x), os.path.join(job["out_dir"], f"out_region{gi}.sm"))
    return {"regions": len(cells), "areas": areas, "extended": job.get("extended", []), "points": points,
            "sides": [[None] * len(cutters) for _ in cells], "picked": 0, "solid": True, "closed": closed,
            "volumes": volumes}


def job_split(job):
    """Split the target along all cutters and write EVERY region as its own
    brep (out_region<i>.sm in out_dir). Cutting surfaces that miss the target
    are ignored; nothing is removed. With job["solids"] a closed solid comes
    apart into closed solids (each piece capped where it was cut); then
    job["keep"] (one bool per piece) writes the kept pieces joined into one
    body as out_KEEP.sm."""
    cutters = build_cutters(job, strict=False)
    if not cutters:
        raise RuntimeError("nothing crosses this surface")
    if "ref_point" not in job:
        lo, hi = world_brep(job, "target").bounding_box()
        job["ref_point"] = [0.5 * (lo[i] + hi[i]) for i in range(3)]
    if job.get("solids"):
        target = world_brep(job, "target")
        try:
            is_solid = bool(target.is_manifold_solid())
        except Exception:
            is_solid = False
        if is_solid:
            return _job_split_solid(job, cutters, target)
    b, faces, groups, picked, sides = _regions(job, cutters)
    if len(groups) < 2:
        raise RuntimeError("the cutters do not split this surface into separate regions")
    x = job.get("xforms", {}).get("target")
    areas, points = [], []
    for g in groups:   # a point inside each region (world), the one its side is judged at
        f = max((faces[i] for i in g), key=lambda fc: fc.area(1e-3))
        points.append([float(v) for v in _inner_point(f)])
    for gi, g in enumerate(groups):
        part = b.copy()
        pf = part.faces()
        keep = set(g)
        drop = [f for i, f in enumerate(pf) if i not in keep]
        if drop:
            ext.delete_faces(part, drop)
        save(to_local(part, x), os.path.join(job["out_dir"], f"out_region{gi}.sm"))
        areas.append(part.area())
    return {"regions": len(groups), "areas": areas, "extended": job.get("extended", []),
            "points": points, "sides": [list(sd) for sd in sides], "picked": picked}


def _shared_edges(b):
    return sum(1 for e in b.edges() if len(e.faces()) == 2)


def job_join(job):
    """Merge several breps (world space) into one and stitch coincident edges
    within `tolerance`. Written in the local space of job['into']."""
    keys = job["order"]
    breps = [world_brep(job, k) for k in keys]
    before = sum(_shared_edges(b) for b in breps)
    # the faces are copied in as they are: a MERGE boolean would intersect
    # them first, and surfaces that touch tangentially along an edge (a blend
    # built G1/G2 into a trimmed face) break it or come out with damaged trims
    merged = breps[0]
    for b in breps[1:]:
        ext.append_brep(merged, b)
    res = sm.stitch_advanced(merged, float(job.get("tolerance", 1e-3)), squeeze_small_edges=False,
                             making_manifold_solid=False)
    b = res[0]
    shared = _shared_edges(b)
    if shared - before <= 0:
        raise RuntimeError("no edges within the stitch tolerance -- the surfaces do not touch "
                           "(raise Tolerance in the F9 panel if they are close)")
    # consistent face orientation (stitched sheets can come in flipped; the
    # fillet then cannot tell which side of an edge is outside)
    try:
        start = max(b.faces(), key=lambda f: f.area())
        sm.unify_normals(b, start, 100)
    except Exception:
        pass
    try:
        solid = bool(b.is_manifold_solid())
    except Exception:
        solid = False
    volume = 0.0
    if solid:
        # a closed shell from stitching has no consistent orientation / inside
        # region yet; healing turns it into a real solid with a volume
        try:
            h = sm.heal_brep(b)
            v = h.volume(1e-6) if h is not None else 0.0
            if h is not None and abs(v) > 0:
                b, volume = h, abs(v)
        except Exception:
            pass
    save(to_local(b, job.get("xforms", {}).get(job.get("into"))), job["out"]["joined"])
    return {"faces": b.face_count(), "stitched": shared - before, "max_gap": float(res[4]) if len(res) > 4 else 0.0,
            "solid": solid, "volume": volume}


def _edge_copy(e):
    """A fresh exact copy of edge e's curve over its own range (the cap faces
    adopt the curves they are given, so never the brep's own). The range is cut
    out by knot insertion when the edge uses only part of its curve."""
    np = _np()
    d = ext.curve_cvs(e.curve())
    p, U = int(d["degree"]), [float(k) for k in d["knots"]]
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    Pw = [np.append(x * wi, wi) for x, wi in zip(P, w)]
    t0, t1 = e.parameter_range()
    span = (U[-1] - U[0]) or 1.0
    eps = 1e-12 * span
    for t, keep_left in ((t1, True), (t0, False)):    # cut the far end first: t0's knots stay put
        if not U[0] + eps < t < U[-1] - eps:
            continue
        have = sum(1 for k in U if abs(k - t) <= eps)
        Pw, U = _insert_knot(Pw, U, p, t, max(0, p - have))
        first = min(i for i in range(len(U)) if abs(U[i] - t) <= eps)   # t now p times from here
        if keep_left:            # CV first-1 is the curve's point at t
            Pw, U = Pw[:first], U[:first] + [t] * (p + 1)
        else:
            Pw, U = Pw[first - 1:], [t] * (p + 1) + U[first + p:]
    pts = [tuple(map(float, x[:3] / x[3])) for x in Pw]
    wts = [float(x[3]) for x in Pw]
    rational = max(abs(x - 1.0) for x in wts) > 1e-12
    return ext.curve_from_cvs(p, pts, U, wts if rational else None)


def _edge_points(e, n=17):
    np = _np()
    t0, t1 = e.parameter_range()
    return np.array([e.point_at(float(t)) for t in np.linspace(t0, t1, n)], float)


def _plane_of(P, tol):
    """(unit normal, point) of the plane through points P, or None when they
    stray from it by more than tol (or lie on a line)."""
    np = _np()
    c = P.mean(axis=0)
    _, s, vt = np.linalg.svd(P - c)
    n = vt[2]
    if s[1] <= tol or np.abs((P - c) @ n).max() > tol:
        return None
    return n, c


def _open_loops(b, tol):
    """The open boundary of b as loops of edges (each a list of edges in
    order). An edge is open when its one face uses it once (a periodic seam
    is used twice by its face). Zero-length (pole) edges are left out."""
    uses = {}
    for f in b.faces():
        for loop in f.boundary_loops():
            for e, _ in loop:
                uses[e] = uses.get(e, 0) + 1
    free = [e for e in b.edges() if len(e.faces()) == 1 and uses.get(e, 0) == 1 and e.length() > tol]
    ends = {}
    for e in free:
        t0, t1 = e.parameter_range()
        ends[e] = (_np().array(e.point_at(t0), float), _np().array(e.point_at(t1), float))
    loops, left = [], list(free)
    while left:
        e = left.pop(0)
        chain, head, tail = [e], ends[e][0], ends[e][1]
        while _np().linalg.norm(head - tail) > tol:
            nxt = None
            for f in left:
                a, z = ends[f]
                if _np().linalg.norm(a - tail) <= tol:
                    nxt, tail = f, z
                elif _np().linalg.norm(z - tail) <= tol:
                    nxt, tail = f, a
                if nxt is not None:
                    break
            if nxt is None:
                raise RuntimeError("the open edges do not close into loops")
            left.remove(nxt)
            chain.append(nxt)
        loops.append(chain)
    return loops


def _cap_pieces(loop, tol):
    """[(plane, curves)] that close one open loop: the loop itself when it is
    flat, else its runs of edges that share a plane, each closed by a straight
    line (a revolve of less than a full turn whose profile ends on the axis:
    two half caps meeting on the axis)."""
    np = _np()
    pts = [_edge_points(e) for e in loop]
    pl = _plane_of(np.vstack(pts), tol)
    if pl is not None:
        return [(pl, [_edge_copy(e) for e in loop])]
    runs, cur = [], [0]
    for i in range(1, len(loop)):
        if _plane_of(np.vstack([pts[j] for j in cur + [i]]), tol) is not None:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    if len(runs) and _plane_of(np.vstack([pts[j] for j in cur + runs[0]]), tol) is not None:
        runs[0] = cur + runs[0]              # the loop started inside a run
    else:
        runs.append(cur)
    if len(runs) < 2:
        raise RuntimeError("an open end is not flat -- it cannot be capped")
    out = []
    for run in runs:
        P = np.vstack([pts[j] for j in run])
        pl = _plane_of(P, tol)
        if pl is None:
            raise RuntimeError("an open end is not flat -- it cannot be capped")
        # the run's two loose ends: points used by one edge of the run only
        E = [(pts[j][0], pts[j][-1]) for j in run]
        loose = []
        for a in [q for pair in E for q in pair]:
            if sum(1 for pair in E for q in pair if np.linalg.norm(q - a) <= tol) == 1:
                loose.append(a)
        if len(loose) != 2:
            raise RuntimeError("an open end is not flat -- it cannot be capped")
        out.append((pl, [_edge_copy(loop[j]) for j in run]
                    + [sm.create_line_segment(tuple(map(float, loose[0])), tuple(map(float, loose[1])))]))
    return out


def _capped(walls, tol):
    """(brep, cap planes) of walls closed by flat caps, stitched and oriented
    like the first wall's first face (raises when it does not close)."""
    b = walls[0].copy()
    for w in walls[1:]:
        ext.append_brep(b, w)
    if len(walls) > 1:                # walls meeting each other (half turns): their shared edges are not ends
        b = sm.stitch_advanced(b, max(tol, 1e-7), squeeze_small_edges=False, making_manifold_solid=False)[0]
    planes = []                                       # [(normal, point, [curves])]
    for loop in _open_loops(b, tol):
        for (n, c), curves in _cap_pieces(loop, tol):
            for q in planes:
                if abs(abs(float(n @ q[0])) - 1.0) < 1e-9 and abs(float((c - q[1]) @ q[0])) <= tol:
                    q[2].extend(curves)
                    break
            else:
                planes.append((n, c, list(curves)))
    for _, _, curves in planes:
        try:
            cap = sm.create_planar_faces(curves)
        except Exception as exc:
            raise RuntimeError(f"a cap could not be made ({exc})")
        ext.append_brep(b, cap)
    # the walls come first in b: its first face is the first wall's
    b = sm.stitch_advanced(b, max(tol, 1e-7), squeeze_small_edges=False, making_manifold_solid=True)[0]
    first = walls[0].faces()[0]

    def off_first(f):
        try:
            return first.closest_point(f.interior_point())[1]
        except Exception:
            return float("inf")
    start = min(b.faces(), key=off_first)
    try:
        sm.unify_normals(b, start, 100)
    except Exception:
        pass
    try:
        solid = bool(b.is_manifold_solid())
    except Exception:
        solid = False
    if not solid:
        n_open = sum(1 for e in b.edges() if len(e.faces()) == 1)
        raise RuntimeError(f"the walls and caps do not close into a solid ({n_open} open edges) -- "
                           "use closed profiles, or open ones whose ends lie on the revolve axis")
    return b, len(planes)


def _volume(b, accuracy=1e-6):
    try:
        return float(b.volume(accuracy))
    except Exception:
        return 0.0


def _vol(b):
    """The volume for the face jobs' checks and reports: 1e-4 relative is ~5x
    quicker than 1e-6 on curved bodies (a cone: 0.5 s instead of 2.4 s)."""
    return _volume(b, 1e-4)


def _turned_wall(w):
    """A wall (one untrimmed face) with its normal turned over."""
    return ext.brep_from_surface(_reverse_v(w.faces()[0].surface()))


def job_cap_solid(job):
    """A closed solid from walls (Extrude's / Revolve's sides, world space):
    every open end is closed by a planar face -- loops in one plane make one
    face, so a profile inside another is its hole -- then stitched, oriented
    and healed. The walls set the orientation: when the solid comes out inside
    out (negative volume) it is built again from the walls turned over.
    Written in the local space of job['into']."""
    np = _np()
    walls = [world_brep(job, k) for k in job["order"]]
    lo = np.min([np.array(w.bounding_box()[0], float) for w in walls], axis=0)
    hi = np.max([np.array(w.bounding_box()[1], float) for w in walls], axis=0)
    tol = 1e-6 * (float(np.linalg.norm(hi - lo)) or 1.0)
    def build(ws):
        b, caps = _capped(ws, tol)
        v = _volume(b)
        if abs(v) <= 0:               # no inside region yet: the healer makes one
            try:
                h = sm.heal_brep(b)
                if h is not None and h.face_count() and h.is_manifold_solid():
                    b, v = h, _volume(h)
            except Exception:
                pass
        return b, caps, v
    b, caps, volume = build(walls)
    if volume < 0 and all(w.face_count() == 1 for w in walls):
        b, caps, volume = build([_turned_wall(w) for w in walls])
    if volume <= 0:
        raise RuntimeError("the solid came out inside out (the kernel could not orient it)")
    save(to_local(b, job.get("xforms", {}).get(job.get("into"))), job["out"]["result"])
    return {"faces": b.face_count(), "caps": caps, "volume": volume}


# ---------------------------------------------------------------------------
# Push / Pull: a flat face moved along its normal (material added or removed)
# ---------------------------------------------------------------------------

def _moved_curve(c, v):
    np = _np()
    d = ext.curve_cvs(c)
    P = np.array(d["points"], float) + np.asarray(v, float)
    return ext.curve_from_cvs(int(d["degree"]), [tuple(map(float, q)) for q in P],
                              [float(k) for k in d["knots"]], d.get("weights") or None)


def _face_at(b, p):
    return min(b.faces(), key=lambda f: f.closest_point(tuple(float(x) for x in p))[1])


def _loop_edges(face, tol):
    """The edges round a face's loops, each once (a seam used twice and
    zero-length pole edges left out)."""
    uses, order = {}, []
    for loop in face.boundary_loops():
        for e, _ in loop:
            if e not in uses:
                order.append(e)
            uses[e] = uses.get(e, 0) + 1
    return [e for e in order if uses[e] == 1 and e.length() > tol]


def _face_prism(face, n, dist, eps, tol):
    """(prism, walls) the face sweeps moving dist along n, reaching eps back
    into the body so the boolean never meets the face exactly (a cap lying on
    the face crashes it). Keep the walls alive with the prism."""
    start = -eps if dist > 0 else eps
    walls = [ext.brep_from_surface(sm.create_extrude_surface(_moved_curve(_edge_copy(e), start * n),
                                                             tuple(float(x) for x in (dist - start) * n)))
             for e in _loop_edges(face, tol)]
    return _capped(walls, tol)[0], walls


def _extruded_face(body, p, n, dist, tol):
    """(solid, kept) of body with the face at p replaced by the walls its loops
    sweep moving dist along n and the face moved there, stitched: Extrude Face,
    no boolean. Exact when the walls meet nothing else of the body."""
    np = _np()
    b = body.copy()
    F = _face_at(b, p)
    # a face the move leaves alone sets the orientation afterwards
    others = [f for f in b.faces() if f != F]
    ref = max((np.array(f.interior_point(), float) for f in others),
              key=lambda q: float(np.linalg.norm(q - np.asarray(p, float))))
    v = tuple(float(x) for x in dist * n)
    edges = _loop_edges(F, tol)
    walls = [ext.brep_from_surface(sm.create_extrude_surface(_edge_copy(e), v)) for e in edges]
    if F.surface().is_planar():
        cap = sm.create_planar_faces([_moved_curve(_edge_copy(e), v) for e in edges])
    else:                                # a curved face: its copy, moved
        cap = _translated(_face_sheet(body, p), v)
    ext.delete_faces(b, [F])
    for w in walls:
        ext.append_brep(b, w)
    ext.append_brep(b, cap)
    b = sm.stitch_advanced(b, tol, squeeze_small_edges=False, making_manifold_solid=True)[0]
    try:
        sm.unify_normals(b, _face_at(b, ref), 100)
    except Exception:
        pass
    return b, (walls, cap)


def _offset_slab(body, p, dist, eps):
    """(slab, kept) the face at p of body sweeps when it is offset dist along
    its outward normal (a curved face: the face, its offset and walls along
    the normals at its edges -- Thicken's shell), reaching eps back into the
    body so the boolean never meets the face exactly."""
    np = _np()
    b = body.copy()
    F = _face_at(b, p)
    ip = np.array(F.interior_point(), float)
    out_n = _face_normal(F, ip)
    _, uvs, _ = sm.surface_closest_point(F.surface(), tuple(map(float, ip)))
    sn = np.array(sm.evaluate_surface_normal(F.surface(), *_uvxy(uvs[0])), float)
    sgn = 1.0 if float(sn @ out_n) > 0 else -1.0      # the face alone offsets along its surface's normal
    ext.delete_faces(b, [f for f in b.faces() if f != F])
    start = -eps if dist > 0 else eps
    why = "the face cannot be offset that far (the distance may be bigger than its radius of curvature)"
    try:
        s0 = sm.offset_brep(b, sgn * start, extended_offset=True, self_intersection=True)
        slab = sm.shell_brep(s0, sgn * (dist - start), extended_offset=True, self_intersection=True,
                             create_solid=True)
    except Exception:
        raise RuntimeError(why)
    if slab is None or not slab.face_count() or not slab.is_manifold_solid():
        raise RuntimeError(why)
    return slab, (b, s0)


def _lean(body, F, n_samples=5):
    """How far (degrees) the face's neighbours are from square to it along
    their shared edges: the walls go along the face's normal, so a neighbour
    at an angle is left with a step about distance x sin(this)."""
    np = _np()
    worst = 0.0
    for e in F.edges():
        other = [g for g in e.faces() if g != F]
        if not other:
            continue
        t0, t1 = e.parameter_range()
        for t in np.linspace(t0, t1, n_samples + 2)[1:-1]:
            q = np.array(e.point_at(t), float)
            try:
                a, b = _face_normal(F, q), _face_normal(other[0], q)
            except Exception:
                continue
            worst = max(worst, float(np.degrees(np.arcsin(min(1.0, abs(float(a @ b)))))))
    return worst


# ---------------------------------------------------------------------------
# Replace Face: a face of a solid replaced by a surface, its sides extended /
# cut back to meet it
# ---------------------------------------------------------------------------

def _translated(b, v):
    return sm.translate(b.copy(), tuple(float(x) for x in v))


def _swept_solid(sheet, start, vec, tol):
    """(solid, kept): the open sheet moved by start, then swept along vec --
    the sheet there, its copy moved by vec, and walls from its open edges
    extruded along vec, stitched and turned outward."""
    np = _np()
    b0 = _translated(sheet, start)
    b1 = _translated(b0, vec)
    walls = [ext.brep_from_surface(sm.create_extrude_surface(_edge_copy(e), tuple(float(x) for x in vec)))
             for loop in _open_loops(b0, tol) for e in loop]
    b = b0.copy()
    ext.append_brep(b, b1)
    for w in walls:
        ext.append_brep(b, w)
    b = sm.stitch_advanced(b, tol, squeeze_small_edges=False, making_manifold_solid=True)[0]
    try:
        sm.unify_normals(b, b.faces()[0], 100)
    except Exception:
        pass
    if not b.is_manifold_solid():
        raise RuntimeError("the swept face does not close into a solid")
    if _volume(b) < 0:
        b = sm.heal_brep(b) or b
    return b, (b0, b1, walls)


def job_extrude_face(job):
    """Extrude Face: the face of the body at face_pt (world) swept along dir,
    from start to distance (signed world lengths along dir): the face there,
    its copy moved and walls from its loops, stitched into a closed solid
    (world). A start a hair behind the face (Join / Cut) keeps the boolean
    from meeting the face exactly."""
    np = _np()
    body = world_brep(job, "target")
    lo, hi = (np.array(v, float) for v in body.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    tol = 1e-6 * size
    p = np.asarray(job["face_pt"], float)
    F = _face_at(body, p)
    if F.closest_point(tuple(float(x) for x in p))[1] > 1e-3 * size:
        raise RuntimeError("the face is no longer on the body")
    d = np.asarray(job["dir"], float)
    d /= np.linalg.norm(d) or 1.0
    a, b = float(job.get("start", 0.0)), float(job["distance"])
    if abs(b - a) <= 1e-9 * size:
        raise RuntimeError("the distance is 0")
    sheet = _face_sheet(body, p)
    why = ("the face swept along the direction does not close into a solid (the direction runs along the "
           "face somewhere, or the face wraps round, like a tube's side)")
    try:
        solid, _kept = _swept_solid(sheet, a * d, (b - a) * d, tol)
    except RuntimeError:
        raise RuntimeError(why)
    v = _volume(solid, 1e-4)
    area = float(sheet.area()) if hasattr(sheet, "area") else 0.0
    if v <= 1e-6 * max(area, size * size) * abs(b - a):
        raise RuntimeError(why)
    save(to_local(solid, job.get("xforms", {}).get(job.get("into"))), job["out"]["result"])
    return {"faces": solid.face_count(), "volume": v}


def _heights(b, n, o, m=9):
    """Signed heights along n (from the point o) of points spread over b's faces."""
    np = _np()
    hs = []
    for f in b.faces():
        for e in f.edges():
            t0, t1 = e.parameter_range()
            hs += [float((np.array(e.point_at(t), float) - o) @ n) for t in np.linspace(t0, t1, m)]
        hs.append(float((np.array(f.interior_point(), float) - o) @ n))
    return hs


# ---------------------------------------------------------------------------
# Exact face move: the neighbours' own surfaces extended / cut back to meet the
# face's new surface (no walls along the normal, no boolean)
# ---------------------------------------------------------------------------

def _curve_piece(c, t0, t1):
    """An exact copy of curve c between parameters t0 and t1 (knot insertion)."""
    np = _np()
    if t0 > t1:
        t0, t1 = t1, t0
    d = ext.curve_cvs(c)
    p, U = int(d["degree"]), [float(k) for k in d["knots"]]
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    Pw = [np.append(x * wi, wi) for x, wi in zip(P, w)]
    eps = 1e-12 * ((U[-1] - U[0]) or 1.0)
    for t, keep_left in ((t1, True), (t0, False)):
        if not U[0] + eps < t < U[-1] - eps:
            continue
        have = sum(1 for k in U if abs(k - t) <= eps)
        Pw, U = _insert_knot(Pw, U, p, t, max(0, p - have))
        first = min(i for i in range(len(U)) if abs(U[i] - t) <= eps)
        if keep_left:
            Pw, U = Pw[:first], U[:first] + [t] * (p + 1)
        else:
            Pw, U = Pw[first - 1:], [t] * (p + 1) + U[first + p:]
    pts = [tuple(map(float, x[:3] / x[3])) for x in Pw]
    wts = [float(x[3]) for x in Pw]
    return ext.curve_from_cvs(p, pts, U, wts if max(abs(x - 1.0) for x in wts) > 1e-12 else None)


def _reversed_curve(c):
    d = ext.curve_cvs(c)
    U = [float(k) for k in d["knots"]]
    a, b = U[0], U[-1]
    return ext.curve_from_cvs(int(d["degree"]), [tuple(map(float, q)) for q in reversed(d["points"])],
                              [a + b - k for k in reversed(U)],
                              list(reversed(d["weights"])) if d.get("weights") else None)


def _extended_curve(c, at_start, dist):
    import natext
    np = _np()
    d = ext.curve_cvs(c)
    P = np.array(d["points"], float)
    if int(d["degree"]) == 1 and len(P) == 2 and not d.get("weights"):     # a line: just longer
        t = (P[1] - P[0]) / (np.linalg.norm(P[1] - P[0]) or 1.0) * dist
        if at_start:
            P[0] = P[0] - t
        else:
            P[1] = P[1] + t
        return ext.curve_from_cvs(1, [tuple(map(float, q)) for q in P], [0.0, 0.0, 1.0, 1.0], None)
    e = natext.extend_curve({"degree": d["degree"], "points": d["points"], "knots": d["knots"],
                             "weights": d.get("weights")}, at_start, dist)
    return ext.curve_from_cvs(int(e["degree"]), [tuple(q) for q in e["points"]], e["knots"], e["weights"])


def _closed_dirs(srf, tol):
    """Is the surface closed round in u, in v (its two iso ends coincide)?"""
    np = _np()
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    ts = np.linspace(0.0, 1.0, 7)
    cu = all(np.linalg.norm(np.array(srf.evaluate(u0, v0 + t * (v1 - v0)), float)
                            - np.array(srf.evaluate(u1, v0 + t * (v1 - v0)), float)) <= tol for t in ts)
    cv = all(np.linalg.norm(np.array(srf.evaluate(u0 + t * (u1 - u0), v0), float)
                            - np.array(srf.evaluate(u0 + t * (u1 - u0), v1), float)) <= tol for t in ts)
    return cu, cv


def _extended_surface(srf, dist, tol):
    """The surface continued as the same polynomial by dist past each side it
    is open at (the old part stays exactly as it was)."""
    import natext
    np = _np()
    d = ext.surface_cvs(srf)
    if srf.is_planar():
        # a plane: a parallelogram patch on it, over its CVs grown by dist (the
        # trims are 3D curves, so the parameters need not be kept)
        P = np.array(d["points"], float)
        c = P.mean(axis=0)
        (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
        su, sv = (np.array(srf.evaluate(u1, v0), float) - np.array(srf.evaluate(u0, v0), float),
                  np.array(srf.evaluate(u0, v1), float) - np.array(srf.evaluate(u0, v0), float))
        nrm = np.cross(su, sv)
        if np.linalg.norm(nrm) > 1e-12:
            nrm /= np.linalg.norm(nrm)
            a = su / np.linalg.norm(su)
            b = np.cross(nrm, a)
            x, y = (P - c) @ a, (P - c) @ b
            x0, x1, y0, y1 = x.min() - dist, x.max() + dist, y.min() - dist, y.max() + dist
            Q = [c + x0 * a + y0 * b, c + x0 * a + y1 * b, c + x1 * a + y0 * b, c + x1 * a + y1 * b]
            return ext.surface_from_cvs(1, 1, [tuple(map(float, q)) for q in Q], [0.0, 0.0, 1.0, 1.0],
                                        [0.0, 0.0, 1.0, 1.0])
    cu, cv = _closed_dirs(srf, tol)
    for side in [s for s, closed in (("U0", cu), ("U1", cu), ("V0", cv), ("V1", cv)) if not closed]:
        du, dv, pts, Ku, Kv, w = natext.extend_grid(d, side, dist)
        nu, nv = len(Ku) - du - 1, len(Kv) - dv - 1
        d = {"degree_u": du, "degree_v": dv, "count_u": nu, "count_v": nv, "points": pts,
             "knots_u": Ku, "knots_v": Kv, "weights": w}
    return ext.surface_from_cvs(d["degree_u"], d["degree_v"], d["points"], d["knots_u"], d["knots_v"],
                                d.get("weights"))


def _chain(curves, tol):
    """The curves in head-to-tail order (each turned round where needed), or None."""
    np = _np()
    if not curves:
        return None
    ends = lambda c: (np.array(c.start_point(), float), np.array(c.end_point(), float))
    out = [curves[0]]
    left = list(curves[1:])
    while left:
        tail = ends(out[-1])[1]
        for i, c in enumerate(left):
            a, b = ends(c)
            if np.linalg.norm(a - tail) <= tol:
                out.append(left.pop(i))
                break
            if np.linalg.norm(b - tail) <= tol:
                out.append(_reversed_curve(left.pop(i)))
                break
        else:
            return None
    return out


def _new_vertex(side, v, G, reach, tol):
    """(new point, side curve from its far end to it) of the vertex v of the
    moved face: the side edge -- extended exactly past v -- meeting the new
    surface G nearest v."""
    np = _np()
    c = _edge_copy(side)
    a, b = np.array(c.start_point(), float), np.array(c.end_point(), float)
    at_start = np.linalg.norm(a - v) < np.linalg.norm(b - v)
    far = b if at_start else a
    cx = _extended_curve(c, at_start, reach)
    _, points = sm.intersect_curve_surface(cx, G)
    if not points:
        raise RuntimeError("a side edge does not reach the new face")
    q = min((np.array(x, float) for x in points), key=lambda x: float(np.linalg.norm(x - v)))
    _, tq, _ = sm.curve_closest_point(cx, tuple(map(float, q)))
    _, tf, _ = sm.curve_closest_point(cx, tuple(map(float, far)))
    _, tv, _ = sm.curve_closest_point(cx, tuple(map(float, v)))
    if (tq - tf) * (tv - tf) <= 0 or abs(tq - tf) <= 1e-9:
        raise RuntimeError("the move runs past a side edge's far end (the face would vanish)")
    return q, (cx, tf, tq)


def _new_edge(srf_ext, G, qa, qb, closed, tol):
    """The curve(s) where the neighbour's extended surface meets G, from qa to qb
    (a closed edge -- a full circle -- comes back to qa: two pieces)."""
    np = _np()
    best = None
    for c in sm.intersect_surfaces(srf_ext, G):
        pa, ta, da = sm.curve_closest_point(c, tuple(map(float, qa)))
        pb, tb, db = sm.curve_closest_point(c, tuple(map(float, qb)))
        if best is None or da + db < best[0]:
            best = (da + db, c, ta, tb)
    if best is None or best[0] > 1e3 * tol:
        raise RuntimeError("a neighbour does not meet the new face")
    _, c, ta, tb = best
    t0, t1 = c.parameter_range()
    if closed:
        return [p_ for p_ in (_curve_piece(c, t0, ta), _curve_piece(c, ta, t1)) if p_.length() > tol]
    return [_curve_piece(c, ta, tb)]


def _surface_piece(srf, along_v, t0, t1):
    """srf cut exactly (knot insertion) to v in [t0, t1] (along_v) or u in [t0, t1]."""
    np = _np()
    d = ext.surface_cvs(srf)
    nu, nv, du, dv = d["count_u"], d["count_v"], int(d["degree_u"]), int(d["degree_v"])
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    w = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    G = np.concatenate([P * w[..., None], w[..., None]], axis=2)
    Ku, Kv = [float(k) for k in d["knots_u"]], [float(k) for k in d["knots_v"]]
    if not along_v:                                   # work along v: transpose
        G, Ku, Kv, du, dv = G.transpose(1, 0, 2), Kv, Ku, dv, du
    if t0 > t1:
        t0, t1 = t1, t0
    rows, K = [], None
    for i in range(G.shape[0]):
        Pw, U = [np.array(x) for x in G[i]], list(Kv)
        eps = 1e-12 * ((U[-1] - U[0]) or 1.0)
        for t, keep_left in ((t1, True), (t0, False)):
            if not U[0] + eps < t < U[-1] - eps:
                continue
            have = sum(1 for k in U if abs(k - t) <= eps)
            Pw, U = _insert_knot(Pw, U, dv, t, max(0, dv - have))
            first = min(j for j in range(len(U)) if abs(U[j] - t) <= eps)
            if keep_left:
                Pw, U = Pw[:first], U[:first] + [t] * (dv + 1)
            else:
                Pw, U = Pw[first - 1:], [t] * (dv + 1) + U[first + dv:]
        rows.append(Pw)
        K = U
    G = np.array(rows, float)
    if not along_v:
        G, Ku, K, du, dv = G.transpose(1, 0, 2), K, Ku, dv, du
    W = G[..., 3]
    pts = [tuple(map(float, G[i, j, :3] / W[i, j])) for i in range(G.shape[0]) for j in range(G.shape[1])]
    rational = np.abs(W - 1.0).max() > 1e-12
    return ext.surface_from_cvs(du, dv, pts, Ku, K, [float(x) for x in W.reshape(-1)] if rational else None)


def _boundary_iso(srf, along_v, at_end):
    """The exact boundary curve of srf at its v (along_v) or u start / end."""
    d = ext.surface_cvs(srf)
    nu, nv = d["count_u"], d["count_v"]
    P = d["points"]
    w = d.get("weights")
    if along_v:                      # v fixed: the curve runs in u
        j = nv - 1 if at_end else 0
        idx = [i * nv + j for i in range(nu)]
        deg, K = d["degree_u"], d["knots_u"]
    else:
        i = nu - 1 if at_end else 0
        idx = [i * nv + j for j in range(nv)]
        deg, K = d["degree_v"], d["knots_v"]
    return ext.curve_from_cvs(int(deg), [tuple(map(float, P[k])) for k in idx], [float(x) for x in K],
                              [float(w[k]) for k in idx] if w else None)


def _iso_cut(N, e, srf_ext, G, q_new, tol):
    """(new face, new edge curve) for a closed natural neighbour N whose edge e
    with the moved face is one of its isos, when G meets its extension along an
    iso too (a cap moved along a cone's / a revolve's axis): the extended
    surface cut exactly at that iso -- no trimming. None if it is not so."""
    np = _np()
    srf = N.surface()
    mid = np.array(e.point_at(sum(e.parameter_range()) / 2), float)
    _, uvs, _ = sm.surface_closest_point(srf_ext, tuple(map(float, mid)))
    u_e, v_e = _uvxy(uvs[0])
    ts = np.linspace(*e.parameter_range(), 9)
    uvp = []
    for t in ts:
        _, uv_, _ = sm.surface_closest_point(srf_ext, tuple(map(float, e.point_at(t))))
        uvp.append(_uvxy(uv_[0]))
    uvp = np.array(uvp)
    (u0, u1), (v0, v1) = srf_ext.u_domain(), srf_ext.v_domain()
    if np.ptp(uvp[:, 1]) <= 1e-7 * (v1 - v0):
        along_v = True                       # e is a v-iso
    elif np.ptp(uvp[:, 0]) <= 1e-7 * (u1 - u0):
        along_v = False
    else:
        return None
    _, uvq, _ = sm.surface_closest_point(srf_ext, tuple(map(float, q_new)))
    t_new = _uvxy(uvq[0])[1 if along_v else 0]
    # G must hold that whole iso
    for a in np.linspace(0.0, 1.0, 13):
        uv = (u0 + a * (u1 - u0), t_new) if along_v else (t_new, v0 + a * (v1 - v0))
        x = np.array(srf_ext.evaluate(*uv), float)
        if sm.surface_closest_point(G, tuple(map(float, x)))[2] > 10 * tol:
            return None
    # the far iso of the old face: its point farthest (in parameter) from e's
    far_pts = [np.array(f_.point_at(sum(f_.parameter_range()) / 2), float) for f_ in N.edges() if f_ != e]
    t_old = t_new
    for x in far_pts:
        _, uv_, _ = sm.surface_closest_point(srf_ext, tuple(map(float, x)))
        t = _uvxy(uv_[0])[1 if along_v else 0]
        if abs(t - (v_e if along_v else u_e)) > abs(t_old - (v_e if along_v else u_e)):
            t_old = t
    if abs(t_old - t_new) <= 1e-9:
        return None
    piece = _surface_piece(srf_ext, along_v, t_old, t_new)
    at_end = t_new > t_old
    return ext.brep_from_surface(piece), _boundary_iso(piece, along_v, at_end)


def _basis(U, p, t):
    """All B-spline basis functions of degree p on knots U at t (a vector)."""
    np = _np()
    n = len(U) - p - 1
    t = min(max(t, U[p]), U[n])
    N = np.zeros(len(U) - 1)
    k = n - 1 if t >= U[n] else max(i for i in range(p, n) if U[i] <= t)
    N[k] = 1.0
    for d in range(1, p + 1):
        M = np.zeros(len(U) - 1)
        for i in range(len(U) - d - 1):
            a = (t - U[i]) / (U[i + d] - U[i]) * N[i] if U[i + d] > U[i] else 0.0
            b = (U[i + d + 1] - t) / (U[i + d + 1] - U[i + 1]) * N[i + 1] if U[i + d + 1] > U[i + 1] else 0.0
            M[i] = a + b
        N = M
    return N[:n]


def _refit_like(srf, at, tol, k=4):
    """A B-spline surface with srf's degrees, knots and weights through the
    points at(u, v) (srf's parameters) -- least squares on a grid; exact when
    the target is representable so (a cone's or a cylinder's offset, with the
    arc weights). Raises if it misses by more than tol on a finer grid."""
    np = _np()
    d = ext.surface_cvs(srf)
    nu, nv, du, dv = d["count_u"], d["count_v"], int(d["degree_u"]), int(d["degree_v"])
    Ku, Kv = [float(x) for x in d["knots_u"]], [float(x) for x in d["knots_v"]]
    w = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    us = np.linspace(Ku[du], Ku[nu], k * nu + 1)
    vs = np.linspace(Kv[dv], Kv[nv], k * nv + 1)
    Bu = [_basis(Ku, du, u) for u in us]
    Bv = [_basis(Kv, dv, v) for v in vs]
    rows, rhs = [], []
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            a = np.outer(Bu[i], Bv[j]) * w
            W = a.sum()
            rows.append((a / W).reshape(-1))
            rhs.append(np.array(at(u, v), float))
    X = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    out = ext.surface_from_cvs(du, dv, [tuple(map(float, q)) for q in X], Ku, Kv,
                               [float(x) for x in w.reshape(-1)] if d.get("weights") else None)
    for u in np.linspace(Ku[du], Ku[nu], 2 * k * nu + 3)[1::2]:
        for v in np.linspace(Kv[dv], Kv[nv], 2 * k * nv + 3)[1::2]:
            if np.linalg.norm(np.array(out.evaluate(u, v), float) - np.array(at(u, v), float)) > tol:
                raise RuntimeError("the offset surface is not exact in the face's own form")
    return out


def _offset_like(srf, dist, tol):
    """srf offset by dist along its own normal, as a B-spline in srf's form."""
    off = sm.create_offset_surface(srf, dist)
    return _refit_like(srf, lambda u, v: sm.evaluate_surface_point(off, u, v), tol)


def _exact_move_closed(body, p, G, reach, tol):
    """(solid, kept): _exact_move for a face closed round (a cone's or a
    revolve's side, its seam an edge of its own) between flat-ish neighbours
    (caps): the new face is G cut exactly at the isos where each extended
    neighbour meets it, the neighbours re-trimmed to those isos."""
    np = _np()
    b = body.copy()
    F = _face_at(b, p)
    ext_d = 2.0 * reach + 10.0 * tol
    rims = [e for e in F.edges() if len([f for f in e.faces() if f != F]) == 1]
    seams = [e for e in F.edges() if not [f for f in e.faces() if f != F]]
    if not rims or not seams:
        raise RuntimeError("not a closed face between caps")
    cu, cv = _closed_dirs(G, tol)
    if cu == cv:
        raise RuntimeError("the new face is not closed round one way")
    G = _extended_surface(G, ext_d, tol)          # past where the neighbours meet it (open way only)
    along_v = cu                         # closed in u: its rims are v-isos
    (u0, u1), (v0, v1) = G.u_domain(), G.v_domain()
    # G's seam line, extended past both ends: where each extended neighbour meets it
    seam_c = _boundary_iso(G, not along_v, False)
    seam_x = _extended_curve(_extended_curve(seam_c, True, ext_d), False, ext_d)
    cuts, nbr = [], {}
    for e in rims:
        N = [f for f in e.faces() if f != F][0]
        if N in nbr:
            raise RuntimeError("a neighbour meets the face twice")
        S = _extended_surface(N.surface(), ext_d, tol)
        _, pts = sm.intersect_curve_surface(seam_x, S)
        if not pts:
            raise RuntimeError("a neighbour does not reach the new face")
        mid = np.array(e.point_at(sum(e.parameter_range()) / 2), float)
        q = min((np.array(x, float) for x in pts), key=lambda x: float(np.linalg.norm(x - mid)))
        _, uvq, _ = sm.surface_closest_point(G, tuple(map(float, q)))
        t = _uvxy(uvq[0])[1 if along_v else 0]
        for a in np.linspace(0.0, 1.0, 13):            # G meets it along that whole iso
            uv = (u0 + a * (u1 - u0), t) if along_v else (t, v0 + a * (v1 - v0))
            x = np.array(G.evaluate(*uv), float)
            if sm.surface_closest_point(S, tuple(map(float, x)))[2] > 10 * tol:
                raise RuntimeError("a neighbour does not meet the new face along one of its circles")
        cuts.append(t)
        nbr[N] = (e, S, t)
    if len(cuts) != 2:
        raise RuntimeError("the face is not between two neighbours")
    piece = _surface_piece(G, along_v, min(cuts), max(cuts))
    new_face = ext.brep_from_surface(piece)
    built = [new_face]
    for N, (e, S, t) in nbr.items():
        rim = _boundary_iso(piece, along_v, t == max(cuts))
        loops = []
        for loop in N.boundary_loops():
            cs = []
            for x, _ in loop:
                if x == e:
                    cs.append(ext.curve_from_cvs(*_cv_tuple(ext.curve_cvs(rim))))
                elif x.length() > tol:
                    cs.append(_edge_copy(x))
            ch = _chain(cs, 1e3 * tol)
            if ch is None:
                raise RuntimeError("a neighbour's new outline does not close")
            loops.append(ch)
        if N.surface().is_planar():
            built.append(sm.create_planar_faces([c for l in loops for c in l]))
        else:
            built.append(ext.trim_surface_with_3d_curves_tol(S, [len(l) for l in loops],
                                                            [c for l in loops for c in l], 10 * tol, 1e2 * tol))
    gone = [F] + list(nbr)
    keep_pts = [np.array(f.interior_point(), float) for f in b.faces() if f not in gone]
    ext.delete_faces(b, gone)
    for nb in built:
        ext.append_brep(b, nb)
    b = sm.stitch_advanced(b, 10 * tol, squeeze_small_edges=False, making_manifold_solid=True)[0]
    try:
        sm.unify_normals(b, _face_at(b, keep_pts[0]) if keep_pts else b.faces()[1], 100)
    except Exception:
        pass
    if not b.is_manifold_solid():
        raise RuntimeError("the moved face and its neighbours do not close into a solid")
    if _vol(b) <= 0:
        raise RuntimeError("the moved face turned the body inside out")
    return b, (built, piece)


def _exact_move(body, p, G, reach, tol):
    """(solid, kept): body with the face at p replaced by the untrimmed surface
    G, each neighbour's own surface extended or cut back to meet it and each
    vertex slid along its side edge -- no walls along the normal, so a leaning
    neighbour stays on its plane / surface. Needs three faces at every vertex
    of the face and neighbours that reach G; raises otherwise (the caller
    falls back)."""
    np = _np()
    b = body.copy()
    F = _face_at(b, p)
    Fedges = set(F.edges())
    ext_d = 2.0 * reach + 10.0 * tol
    new_v = {}                                   # vertex -> (point, (curve, t_far, t_new), side edge)
    for e in Fedges:
        for v in e.vertices():
            if v in new_v:
                continue
            sides = [x for x in v.edges() if x not in Fedges]
            if len(sides) != 1:
                raise RuntimeError("a vertex of the face has more than three faces")
            q = np.array(v.point(), float)
            new_v[v] = _new_vertex(sides[0], q, G, ext_d, tol) + (sides[0],)
    side_of = {sd: (v, q, piece) for v, (q, piece, sd) in new_v.items()}
    nbrs = {}
    for e in Fedges:
        others = [f for f in e.faces() if f != F]
        if len(others) != 1:
            raise RuntimeError("an edge of the face is not shared with one neighbour")
        nbrs.setdefault(others[0], []).append(e)
    ext_srf = {N: _extended_surface(N.surface(), ext_d, tol) for N in nbrs}
    new_c = {}                                   # face edge -> its new curve(s) (fresh copies made from these)
    iso_faces = {}                               # closed neighbour -> its new natural face
    for N, es in nbrs.items():
        if not any(_closed_dirs(N.surface(), tol)):
            continue
        if len(es) != 1:
            raise RuntimeError("a closed neighbour meets the face along several edges")
        e = es[0]
        cut = _iso_cut(N, e, ext_srf[N], G, new_v[e.start_vertex()][0], tol)
        if cut is None:
            raise RuntimeError("a closed neighbour (a cone, a revolve) does not meet the new face along "
                               "one of its circles")
        iso_faces[N], c = cut
        new_c[e] = [c]
    for e in Fedges:
        if e in new_c:
            continue
        N = [f for f in e.faces() if f != F][0]
        va, vb = e.start_vertex(), e.end_vertex()
        closed = va == vb
        new_c[e] = _new_edge(ext_srf[N], G, new_v[va][0], new_v[vb][0], closed, tol)
    copy_c = lambda c: ext.curve_from_cvs(*_cv_tuple(ext.curve_cvs(c)))

    def side_piece(sd):
        cx, tf, tq = side_of[sd][2]
        return _curve_piece(cx, tf, tq)

    built, kept = [], []
    # the neighbours, re-trimmed on their extended surfaces
    for N in nbrs:
        if N in iso_faces:
            built.append(iso_faces[N])
            continue
        loops = []
        for loop in N.boundary_loops():
            cs = []
            for x, _ in loop:
                if x in new_c:
                    cs += [copy_c(c) for c in new_c[x]]
                elif x in side_of:
                    cs.append(side_piece(x))
                elif x.length() > tol:
                    cs.append(_edge_copy(x))
            ch = _chain(cs, 1e3 * tol)
            if ch is None:
                raise RuntimeError("a neighbour's new outline does not close")
            loops.append(ch)
        srf = _extended_surface(N.surface(), ext_d, tol)
        built.append(ext.trim_surface_with_3d_curves_tol(srf, [len(l) for l in loops],
                                                        [c for l in loops for c in l], 10 * tol, 1e2 * tol))
    # the new face: G trimmed to the new outline(s)
    loops = []
    for loop in F.boundary_loops():
        ch = _chain([copy_c(c) for x, _ in loop for c in new_c[x]], 1e3 * tol)
        if ch is None:
            raise RuntimeError("the new face's outline does not close")
        loops.append(ch)
    if G.is_planar():
        built.append(sm.create_planar_faces([c for l in loops for c in l]))
    else:
        Gc = ext.surface_from_cvs(*_srf_tuple(ext.surface_cvs(G)))
        built.append(ext.trim_surface_with_3d_curves_tol(Gc, [len(l) for l in loops],
                                                        [c for l in loops for c in l], 10 * tol, 1e2 * tol))
    # the body without the face and its neighbours, the new faces in, stitched
    gone = [F] + list(nbrs)
    keep_pts = [np.array(f.interior_point(), float) for f in b.faces() if f not in gone]
    if not keep_pts:
        raise RuntimeError("nothing of the body stays as it was")
    ext.delete_faces(b, gone)
    for nb in built:
        ext.append_brep(b, nb)
    b = sm.stitch_advanced(b, 10 * tol, squeeze_small_edges=False, making_manifold_solid=True)[0]
    try:
        sm.unify_normals(b, _face_at(b, keep_pts[0]), 100)
    except Exception:
        pass
    if not b.is_manifold_solid():
        raise RuntimeError("the moved face and its neighbours do not close into a solid")
    if _vol(b) <= 0:
        raise RuntimeError("the moved face turned the body inside out")
    return b, (built, ext_srf)


def _cv_tuple(d):
    return (int(d["degree"]), [tuple(map(float, q)) for q in d["points"]], [float(k) for k in d["knots"]],
            d.get("weights") or None)


def _srf_tuple(d):
    return (int(d["degree_u"]), int(d["degree_v"]), [tuple(map(float, q)) for q in d["points"]],
            [float(k) for k in d["knots_u"]], [float(k) for k in d["knots_v"]], d.get("weights") or None)


def _moved_surface(srf, v):
    d = ext.surface_cvs(srf)
    a = _srf_tuple(d)
    return ext.surface_from_cvs(a[0], a[1], [tuple(float(x + y) for x, y in zip(q, v)) for q in a[2]],
                                a[3], a[4], a[5])


def _face_sheet(body, p):
    """The face of body at p alone, as an open sheet."""
    b = body.copy()
    F = _face_at(b, p)
    ext.delete_faces(b, [f for f in b.faces() if f != F])
    return b


def _find_face(body, p, want_n, was_flat, size):
    """(face, point inside it) a job means by the point p: when it was flat and
    its normal is known, the nearest flat face facing the same way (a rebuild
    may have moved it); else the face at p, which must be there."""
    np = _np()
    F = _face_at(body, p)
    if want_n is not None and was_flat:
        same = []
        for f in body.faces():
            try:
                if f.surface().is_planar() and float(
                        _face_normal(f, np.array(f.interior_point(), float)) @ np.asarray(want_n, float)) > 0.9999:
                    same.append(f)
            except Exception:
                pass
        if same:
            F = min(same, key=lambda f: f.closest_point(tuple(float(x) for x in p))[1])
            return F, np.array(F.interior_point(), float)
    elif F.closest_point(tuple(float(x) for x in p))[1] > 1e-3 * size:
        raise RuntimeError("a picked face is no longer on the body")
    return F, np.asarray(p, float)


def _covers(sheet, face, n, reach, tol, per_edge=6):
    """Does every line along n through the face's edge points meet the sheet?"""
    np = _np()
    for e in face.edges():
        t0, t1 = e.parameter_range()
        for t in np.linspace(t0, t1, per_edge + 2)[1:-1]:
            q = np.array(e.point_at(t), float)
            ln = sm.create_line_segment(tuple(map(float, q - reach * n)), tuple(map(float, q + reach * n)))
            hit = False
            for f in sheet.faces():
                try:
                    curves, points = sm.intersect_curve_face(ln, f)
                    if points or curves:
                        hit = True
                        break
                except Exception:
                    pass
            if not hit:
                return False
    return True


def _over_face(sheet, face, n, reach, k=6):
    """Points of the sheet over the inside of the face, along n: from the
    face's inner point toward its edge points (each pulled back onto it)."""
    np = _np()
    o = np.array(face.interior_point(), float)
    out = []
    for e in face.edges():
        t0, t1 = e.parameter_range()
        for t in np.linspace(t0, t1, k + 2)[1:-1]:
            q = 0.5 * (o + np.array(e.point_at(t), float))
            q = np.array(face.closest_point(tuple(map(float, q)))[0], float)
            ln = sm.create_line_segment(tuple(map(float, q - reach * n)), tuple(map(float, q + reach * n)))
            for f in sheet.faces():
                try:
                    _, points = sm.intersect_curve_face(ln, f)
                except Exception:
                    continue
                if points:
                    out.append(np.array(points[0], float))
                    break
    return out


def _surface_sign(F):
    """+1 when the face's surface normal is its outward normal, else -1."""
    np = _np()
    ip = np.array(F.interior_point(), float)
    _, uvs, _ = sm.surface_closest_point(F.surface(), tuple(map(float, ip)))
    sn = np.array(sm.evaluate_surface_normal(F.surface(), *_uvxy(uvs[0])), float)
    return 1.0 if float(sn @ _face_normal(F, ip)) > 0 else -1.0


def _try_exact(body, F, p, n, dist, G, reach, tol):
    """The face moved exactly (its neighbours extended / cut back on their own
    surfaces) to the new surface G -- or for Push / Pull (G None) its plane
    moved by dist / its surface offset by dist. None when it can't be done
    (the caller keeps the walls along the normal)."""
    closed = any(not [f for f in e.faces() if f != F] for e in F.edges())     # a seam: closed round
    try:
        if G is None:
            if F.surface().is_planar():
                G = _moved_surface(F.surface(), dist * n)
            else:
                G = _offset_like(F.surface(), _surface_sign(F) * dist, tol)
        if closed:
            out, kept = _exact_move_closed(body, p, G, reach, tol)
        else:
            out, kept = _exact_move(body, p, _extended_surface(G, 2.0 * reach + 10.0 * tol, tol), reach, tol)
    except Exception:
        return None
    if not out.is_manifold_solid():
        return None
    if dist:                                          # a pull must add, a push remove (when it tells)
        V0, V1 = _volume(body, 1e-3), _volume(out, 1e-3)
        if abs(V1 - V0) > 2e-3 * max(abs(V0), 1e-12) and (V1 - V0) * dist <= 0:
            return None
    return out, kept


def _replace_one(body, F, p, sheet, size, tol, eps, exact=True):
    """(body, how, lean, kept): the one face F of body (at p) replaced by the
    open sheet, as job_replace_face does it. exact: leaning neighbours are
    first tried on their own surfaces (_try_exact)."""
    np = _np()
    keep = []
    o = np.array(F.interior_point(), float)
    n = _face_normal(F, o)
    flat = bool(F.surface().is_planar())
    if not flat:
        # swept along one direction: the face must not turn away from it
        for e in F.edges():
            t0, t1 = e.parameter_range()
            for t in np.linspace(t0, t1, 5):
                if float(_face_normal(F, np.array(e.point_at(t), float)) @ n) < 0.2:
                    raise RuntimeError("the face turns too far to be replaced along one direction "
                                       "(pick a flatter face)")
    hs = _heights(sheet, n, o)
    try:
        face_lean = _lean(body, F)
    except Exception:
        face_lean = 0.0
    if exact and face_lean > 0.5 and sheet.face_count() == 1:      # leaning neighbours: on their own surfaces
        ex = _try_exact(body, F, p, n, 0.0, sheet.faces()[0].surface(),
                        max(abs(h) for h in hs) + 0.05 * size, tol)
        if ex is not None:
            return ex[0], "exact", face_lean, [ex[1]]
    reach = max(abs(h) for h in hs) + size
    if not _covers(sheet, F, n, reach, tol):
        raise RuntimeError("the surface does not cover the whole face (seen along its normal): "
                           "extend it past the face's edges")
    hi, lo = max(hs), min(hs)
    V0 = _vol(body)
    U, how = body, "cut"
    if hi > tol:                  # the face's column joined, up to past the surface
        H = hi + 0.1 * size
        out = None
        try:
            tool, kept = (_face_prism(F, n, H, eps, tol) if flat else
                          _swept_solid(_face_sheet(body, p), -eps * n, (H + eps) * n, tol))
            keep.append((tool, kept))
            out = sm.boolean(body.copy(), tool.copy(), operation=sm.BooleanOp.UNION)
        except Exception:
            out = None
        if out is None or not out.face_count() or not out.is_manifold_solid() or abs(_vol(out) - V0) <= tol:
            out, kept = _extruded_face(body, p, n, H, tol)
            keep.append(kept)
            how = "extrude"
        else:
            how = "boolean"
        if not out.is_manifold_solid():
            raise RuntimeError("the face's column could not be joined to the body")
        U = out
    top = max(hi, 0.0) + 0.2 * size
    depth = max(-lo, 0.0) + 0.2 * size
    col, k1 = _swept_solid(_face_sheet(body, p), -depth * n, (depth + top) * n, tol)
    far, k2 = _swept_solid(sheet, np.zeros(3), (top + depth + size) * n, tol)
    keep += [k1, k2, (col, far)]
    try:
        cut = sm.boolean(col.copy(), far.copy(), operation=sm.BooleanOp.INTERSECTION)
    except Exception as exc:
        raise RuntimeError(f"the part beyond the surface could not be found ({exc})")
    if cut is None or not cut.face_count():
        res = U                   # nothing beyond the surface: the column ends exactly there
    else:
        res = sm.boolean(U.copy(), cut.copy(), operation=sm.BooleanOp.DIFFERENCE)
        if res is None or not res.face_count():
            raise RuntimeError("nothing left: the surface cuts the whole body away")
        # the surface over the inside of the face must now be the body's boundary
        # (a boolean that silently did nothing leaves those points inside it)
        if not res.is_manifold_solid() or any(dist_to_brep(tuple(map(float, h)), res) > 1e-4 * size
                                              for h in _over_face(sheet, F, n, reach)):
            raise RuntimeError("the cut back to the surface failed (the boolean)")
    return res, how, face_lean, keep


def job_replace_face(job):
    """Replace Face (world space): each face of the solid job['target'] at
    job['face_pts'] is replaced by the open surface job['surface'], its sides
    extended or cut back along the face's normal to meet it -- the surface may
    lie above the face, below it or cross it. Material up to past the surface
    is joined first (the face's column: the boolean, or the face extruded and
    stitched when its side lies on a curved neighbour), then the part of the
    face's column beyond the surface is cut away: the column intersected with
    the surface swept away from the body. Accepted when the volumes add up.
    The surface must cover the whole face (seen along its normal). Written in
    the local space of job['into']."""
    np = _np()
    body = world_brep(job, "target")
    sheet = world_brep(job, "surface")
    if not bool(body.is_manifold_solid()):
        raise RuntimeError("Replace Face needs a closed solid (all edges shared by two faces)")
    if bool(sheet.is_manifold_solid()):
        raise RuntimeError("the replacing surface is a closed solid: use an open surface")
    lo_, hi_ = (np.array(v, float) for v in body.bounding_box())
    size = float(np.linalg.norm(hi_ - lo_)) or 1.0
    tol, eps = 1e-6 * size, 1e-5 * size
    pts = job["face_pts"]
    normals = job.get("face_normals") or [None] * len(pts)
    flats = job.get("face_flat") or [True] * len(pts)
    keep, hows, lean = [], [], 0.0
    for p, want_n, was_flat in zip(pts, normals, flats):
        F, p = _find_face(body, p, want_n, was_flat, size)
        body, how, face_lean, kept = _replace_one(body, F, p, sheet, size, tol, eps)
        keep += kept
        hows.append(how)
        if how != "exact":
            lean = max(lean, face_lean)
    volume = _volume(body, 1e-3)               # for the report
    save(to_local(body, job.get("xforms", {}).get(job.get("into"))), job["out"]["result"])
    return {"faces": body.face_count(), "volume": volume, "how": hows, "lean": lean}


def job_push_pull(job):
    """Push / Pull faces of a closed solid (world space): each face at
    job['face_pts'] moves job['distance'] along its outward normal -- out adds
    material, in removes it. A flat face: its prism is joined to / cut from
    the body (the boolean merges faces that end up in one plane); when the
    boolean leaves the body as it was (a prism whose side lies on a curved
    neighbour, a cylinder's top pulled), the face is extruded and stitched
    instead, accepted when its volume is the body's plus area x distance.
    A curved face is offset: the slab between it and its offset (walls along
    its normals) is joined / cut (a hole or boss changes radius). Either way
    the walls go along the face's normal; 'lean' reports how far the
    neighbours are from square to it (a step is left where they lean).
    job['face_flat'] (optional): which faces were flat when picked -- a flat
    one is found again on rebuild by its normal (job['face_normals']).
    Written in the local space of job['into']."""
    np = _np()
    body = world_brep(job, "target")
    if not bool(body.is_manifold_solid()):
        raise RuntimeError("Push / Pull needs a closed solid (all edges shared by two faces)")
    dist = float(job["distance"])
    lo, hi = (np.array(v, float) for v in body.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    tol, eps = 1e-6 * size, 1e-5 * size
    if abs(dist) <= tol:
        raise RuntimeError("the distance is 0")
    hows, keep, lean = [], [], 0.0
    pts = job["face_pts"]
    normals = job.get("face_normals") or [None] * len(pts)
    flats = job.get("face_flat") or [True] * len(pts)
    op = sm.BooleanOp.UNION if dist > 0 else sm.BooleanOp.DIFFERENCE
    for p, want_n, was_flat in zip(pts, normals, flats):
        F, p = _find_face(body, p, want_n, was_flat, size)
        flat = bool(F.surface().is_planar())
        try:
            face_lean = _lean(body, F)
        except Exception:
            face_lean = 0.0
        n = _face_normal(F, np.array(F.interior_point(), float))
        res = None
        if face_lean > 0.5:           # leaning neighbours: extended on their own surfaces
            ex = _try_exact(body, F, p, n, dist, None, abs(dist) + 0.05 * size, tol)
            if ex is not None:
                keep.append(ex[1])
                body = ex[0]
                hows.append("exact")
                continue
        lean = max(lean, face_lean)
        area = float(sm.face_area(F, 1e-6))
        V = _vol(body)
        change = area * dist
        try:
            if flat:
                tool, kept = _face_prism(F, n, dist, eps, tol)
            else:
                tool, kept = _offset_slab(body, p, dist, eps)
            keep.append((tool, kept))
            out = sm.boolean(body.copy(), tool.copy(), operation=op)
            if out is not None and out.face_count():
                if out.is_manifold_solid() and abs(_vol(out) - V) > 1e-3 * abs(change):
                    res = (out, "boolean" if flat else "offset")
            elif dist < 0:
                raise RuntimeError("nothing left: the push removes the whole body")
        except RuntimeError as exc:
            if "nothing left" in str(exc) or not flat:
                raise
        except Exception as exc:
            if not flat:
                raise RuntimeError(f"the offset face could not be joined to the body ({exc})")
        if res is None and flat:
            try:
                out, kept = _extruded_face(body, p, n, dist, tol)
                keep.append(kept)
                if out.is_manifold_solid() and abs(_vol(out) - (V + change)) <= 1e-3 * abs(change) + 2e-4 * abs(V) + tol:
                    res = (out, "extrude")
            except Exception:
                pass
        if res is None:
            raise RuntimeError("the face could not be moved (the boolean failed" +
                               (" and the extruded face does not close into a solid)" if flat else ")"))
        body, how = res
        hows.append(how)
    volume = _volume(body, 1e-3)               # for the report
    if volume <= 0:
        raise RuntimeError("nothing left: the push removes the whole body")
    save(to_local(body, job.get("xforms", {}).get(job.get("into"))), job["out"]["result"])
    return {"faces": body.face_count(), "volume": volume, "how": hows, "lean": lean}


def _rotation(z, ang):
    """3x3: ang radians about unit z, right-handed."""
    np = _np()
    z = np.asarray(z, float)
    K = np.array([[0.0, -z[2], z[1]], [z[2], 0.0, -z[0]], [-z[1], z[0], 0.0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def _rotated_surface(srf, O, R):
    """A free copy of srf turned by R about the point O (its CVs moved: exact,
    rational surfaces too)."""
    np = _np()
    a = _srf_tuple(ext.surface_cvs(srf))
    pts = [tuple(map(float, O + R @ (np.asarray(q, float) - O))) for q in a[2]]
    return ext.surface_from_cvs(a[0], a[1], pts, a[3], a[4], a[5])


def _ruled_rows(srf):
    """(points (2, n, 3), ruling-first?) of a surface ruled straight between two
    CV rows along one direction (an Extrude wall: degree 1, two rows), or
    None."""
    np = _np()
    d = ext.surface_cvs(srf)
    nu, nv = int(d["count_u"]), int(d["count_v"])
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    if int(d["degree_u"]) == 1 and nu == 2:
        return P, True
    if int(d["degree_v"]) == 1 and nv == 2:
        return np.transpose(P, (1, 0, 2)), False
    return None


def _ruled_cap_move(body, p, G, tol):
    """(solid, kept): the flat face at p (a cap) moved onto the plane G where
    every neighbour is a closed wall ruled straight along one direction (an
    extruded cylinder or tube, untrimmed): each wall's row on the cap's side
    slid along its rulings onto G (an affine map of its CVs: exact, rational
    ones too), the cap rebuilt on those rows. No booleans: the walls stay one
    face each. Raises when the face is not such a cap."""
    np = _np()
    if not G.is_planar():
        raise RuntimeError("the new face is not flat")
    b = body.copy()
    F = _face_at(b, p)
    (u0, u1), (v0, v1) = G.u_domain(), G.v_domain()
    g0 = np.array(G.evaluate(u0, v0), float)
    gn = np.cross(np.array(G.evaluate(u1, v0), float) - g0, np.array(G.evaluate(u0, v1), float) - g0)
    if np.linalg.norm(gn) <= 1e-300:
        raise RuntimeError("the new face's plane has no normal")
    gn /= np.linalg.norm(gn)
    walls = {}
    for e in F.edges():
        others = [f for f in e.faces() if f != F]
        if len(others) != 1:
            raise RuntimeError("an edge of the cap is not shared with one wall")
        walls.setdefault(others[0], []).append(e)
    built, rims = [], []
    for N, es in walls.items():
        srf = N.surface()
        if not any(_closed_dirs(srf, tol)):
            raise RuntimeError("a neighbour is not closed round")
        rr = _ruled_rows(srf)
        if rr is None:
            raise RuntimeError("a neighbour is not a straight extrusion")
        P, ruling_first = rr
        d = ext.surface_cvs(srf)
        D = P[1] - P[0]
        if np.linalg.norm(D - D.mean(axis=0), axis=1).max() > 1e-6 * (np.linalg.norm(D, axis=1).max() or 1.0):
            raise RuntimeError("a neighbour's rulings are not parallel")
        dirn = D.mean(axis=0)
        # which row is on the cap's side: the one the cap's edges lie on
        mids = [np.array(e.point_at(sum(e.parameter_range()) / 2), float) for e in es]

        def row_curve(R, k, d=d, ruling_first=ruling_first):
            return ext.curve_from_cvs(int(d["degree_v" if ruling_first else "degree_u"]),
                                      [tuple(map(float, q)) for q in R],
                                      [float(x) for x in d["knots_v" if ruling_first else "knots_u"]],
                                      _row_weights(d, k, ruling_first))
        rows = [row_curve(P[k], k) for k in (0, 1)]
        dist = [max(sm.curve_closest_point(rows[k], tuple(map(float, m)))[2] for m in mids) for k in (0, 1)]
        k = int(np.argmin(dist))
        if dist[k] > 1e3 * tol:
            raise RuntimeError("the cap does not sit on a wall's end row")
        # the wall must be natural: its own boundary is its two end rows (and the seam)
        for e in N.edges():
            if e in es or not e.faces() or len(set(e.faces())) == 1:
                continue
            m = np.array(e.point_at(sum(e.parameter_range()) / 2), float)
            if sm.curve_closest_point(rows[1 - k], tuple(map(float, m)))[2] > 1e3 * tol:
                raise RuntimeError("a wall is trimmed")
        den = float(dirn @ gn)
        if abs(den) <= 1e-6 * np.linalg.norm(dirn):
            raise RuntimeError("the new face runs along a wall")
        Q = P.copy()
        Q[k] = P[k] + np.outer(((g0 - P[k]) @ gn) / den, dirn)
        # the wall keeps its direction: the cap's row stays on its side of the far row
        s = (Q[k] - Q[1 - k]) @ dirn * (1.0 if k == 1 else -1.0)
        if s.min() <= 1e3 * tol:
            raise RuntimeError("the face would run through the far end of a wall")
        pts = Q if ruling_first else np.transpose(Q, (1, 0, 2))
        flat = [tuple(map(float, q)) for q in pts.reshape(-1, 3)]
        built.append(ext.brep_from_surface(ext.surface_from_cvs(
            int(d["degree_u"]), int(d["degree_v"]), flat, [float(x) for x in d["knots_u"]],
            [float(x) for x in d["knots_v"]], d.get("weights") or None)))
        rims.append(row_curve(Q[k], k))
    built.append(sm.create_planar_faces(rims))
    gone = [F] + list(walls)
    keep_pts = [np.array(f.interior_point(), float) for f in b.faces() if f not in gone]
    if not keep_pts:
        raise RuntimeError("nothing of the body stays as it was")
    ext.delete_faces(b, gone)
    for nb in built:
        ext.append_brep(b, nb)
    b = sm.stitch_advanced(b, 10 * tol, squeeze_small_edges=False, making_manifold_solid=True)[0]
    try:
        sm.unify_normals(b, _face_at(b, keep_pts[0]), 100)
    except Exception:
        pass
    if not b.is_manifold_solid():
        raise RuntimeError("the moved cap and its walls do not close into a solid")
    if _vol(b) <= 0:
        raise RuntimeError("the moved cap turned the body inside out")
    return b, built


def _row_weights(d, k, ruling_first):
    """The weights of CV row k (along the rulings' cross direction), or None."""
    np = _np()
    if not d.get("weights"):
        return None
    W = np.array(d["weights"], float).reshape(int(d["count_u"]), int(d["count_v"]))
    w = W[k] if ruling_first else W[:, k]
    return [float(x) for x in w] if np.abs(w - 1.0).max() > 1e-12 else None


def _own_hinge(F, O, z, size):
    """A point of the face's straight edge along z nearest the line through O
    (a rebuild moved the face: its hinge edge moved with it); O when it has none."""
    np = _np()
    best = None
    for e in F.edges():
        a, b = np.array(e.start_vertex().point(), float), np.array(e.end_vertex().point(), float)
        L = float(np.linalg.norm(b - a))
        if L <= 1e-9 * size:
            continue
        d = (b - a) / L
        if abs(float(d @ z)) < 1.0 - 1e-9:
            continue
        m = np.array(e.point_at(sum(e.parameter_range()) / 2), float)
        r = m - a
        if np.linalg.norm(r - (r @ d) * d) > 1e-6 * size:          # not straight
            continue
        w = O - a
        dist = float(np.linalg.norm(w - (w @ d) * d))
        if best is None or dist < best[0]:
            best = (dist, a + (w @ d) * d)
    return O if best is None else best[1]


def job_rotate_face(job):
    """Rotate Face (world space): each face of the solid job['target'] at
    job['face_pts'] turned by job['angle'] degrees (right-handed) about the
    axis through job['axis_point'] along job['axis_dir'] -- usually one of
    its own edges, a hinge. The face keeps its shape (its surface turned);
    each neighbour keeps its own surface, extended or cut back to meet it,
    and each corner slides along its side edge, as Push / Pull's exact move.
    A corner with more than three faces, a side edge that never meets the
    turned face or a neighbour that runs out (the face would vanish) is
    refused. job['hinge_on_face'] (optional, per face): the hinge was that
    face's own edge -- on a rebuild it is the face's straight edge along the
    axis nearest the stored one (the face may have moved). Written in the
    local space of job['into']."""
    np = _np()
    body = world_brep(job, "target")
    if not bool(body.is_manifold_solid()):
        raise RuntimeError("Rotate Face needs a closed solid (all edges shared by two faces)")
    ang = float(np.radians(float(job["angle"])))
    if abs(ang) <= 1e-9:
        raise RuntimeError("the angle is 0")
    O = np.asarray(job["axis_point"], float)
    z = np.asarray(job["axis_dir"], float)
    if np.linalg.norm(z) <= 1e-12:
        raise RuntimeError("the axis has no direction")
    z = z / np.linalg.norm(z)
    R = _rotation(z, ang)
    lo, hi = (np.array(v, float) for v in body.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    tol = 1e-6 * size
    pts = job["face_pts"]
    normals = job.get("face_normals") or [None] * len(pts)
    flats = job.get("face_flat") or [True] * len(pts)
    eps = 1e-5 * size
    on_face = job.get("hinge_on_face") or [False] * len(pts)
    keep, hows, lean, V0 = [], [], 0.0, _vol(body)
    for p, want_n, was_flat, own in zip(pts, normals, flats, on_face):
        F, p = _find_face(body, p, want_n, was_flat, size)
        Of = _own_hinge(F, O, z, size) if own else O     # hinged on its own edge: that edge, where it is now
        G = _rotated_surface(F.surface(), Of, R)
        moved = max((float(np.linalg.norm((R - np.eye(3)) @ (np.array(v.point(), float) - Of)))
                     for e in F.edges() for v in e.vertices()), default=0.0)
        closed = any(not [f for f in e.faces() if f != F] for e in F.edges())     # a seam: closed round
        out, why = None, ""
        for r in (moved + 0.25 * size, moved + size):     # a slanting side edge slides further than a corner turns
            try:
                if closed:
                    out, kept = _exact_move_closed(body, p, G, r, tol)
                else:
                    out, kept = _exact_move(body, p, _extended_surface(G, 2.0 * r + 10.0 * tol, tol), r, tol)
                if out.is_manifold_solid():
                    break
                out = None
            except Exception as exc:
                out, why = None, str(exc)
        if out is None and closed is False and G.is_planar():
            try:                          # a cap on extruded walls (a cylinder's or a tube's end)
                out, kept = _ruled_cap_move(body, p, G, tol)
            except Exception as exc:
                why = why or str(exc)
                out = None
        if out is not None:
            keep.append(kept)
            body = out
            hows.append("exact")
            continue
        # a neighbour that can't be extended to meet it (a cylinder round a
        # tilted cap): the face replaced by its turned surface, the walls along
        # its normal (Replace Face)
        try:
            sheet = ext.brep_from_surface(_extended_surface(G, size, tol))
            body, how, face_lean, kept = _replace_one(body, F, p, sheet, size, tol, eps, exact=False)
        except Exception as exc:
            if "SM_ERR" in why or "vanish" in why or "reach" in why or "far end" in why:
                raise RuntimeError("the face can't turn that far: a side edge would no longer meet it "
                                   "(a neighbour would vanish)")
            raise RuntimeError(f"the face could not be turned ({why or exc})")
        keep.append(kept)
        hows.append(how)
        lean = max(lean, face_lean)
    volume = _volume(body, 1e-3)
    if volume <= 0:
        raise RuntimeError("the turned face turned the body inside out")
    save(to_local(body, job.get("xforms", {}).get(job.get("into"))), job["out"]["result"])
    return {"faces": body.face_count(), "volume": volume, "change": volume - V0, "how": hows, "lean": lean}


def _loop_polygon(loop, k=24):
    """Points along a face loop in its own order (each edge sampled k times)."""
    np = _np()
    out = []
    for e, fw in loop:
        t0, t1 = e.parameter_range()
        ts = np.linspace(t0, t1, k + 1)[:-1] if fw else np.linspace(t1, t0, k + 1)[:-1]
        out += [np.array(e.point_at(float(t)), float) for t in ts]
    return np.array(out)


def _moved_curve(c, d):
    deg, pts, U, w = _cv_tuple(ext.curve_cvs(c))
    return ext.curve_from_cvs(deg, [tuple(float(a + b) for a, b in zip(q, d)) for q in pts], U, w)


def _rigid_curve(c, M):
    """A free copy of curve c carried by the rigid motion M (its CVs moved:
    exact, rational curves too)."""
    deg, pts, U, w = _cv_tuple(ext.curve_cvs(c))
    return ext.curve_from_cvs(deg, [tuple(map(float, M(q))) for q in pts], U, w)


def _feature_turns(job):
    """[(axis point, unit axis, degrees)] of a Move Face job, in the order
    they apply: job['turns'], else the one job['angle'] about job['axis_point']
    / job['axis_dir']."""
    np = _np()
    out = []
    for O, z, ang in job.get("turns") or []:
        if abs(float(ang)) > 1e-9:
            z = np.asarray(z, float)
            out.append((np.asarray(O, float), z / np.linalg.norm(z), float(ang)))
    ang = float(job.get("angle", 0.0) or 0.0)
    if not out and abs(ang) > 1e-9:
        z = np.asarray(job["axis_dir"], float)
        out.append((np.asarray(job["axis_point"], float), z / np.linalg.norm(z), ang))
    return out


def _feature_motion(job, size):
    """The rigid motion M (points (..., 3) -> moved) of a Move Face job: its
    turns (_feature_turns) one after the other, then slid by job['vector'];
    None when it moves nothing."""
    np = _np()
    d = np.asarray(job.get("vector", (0.0, 0.0, 0.0)), float)
    turns = [(O, _rotation(z, np.radians(a))) for O, z, a in _feature_turns(job)]
    if np.linalg.norm(d) <= 1e-6 * size and not turns:
        return None

    def M(P):
        P = np.asarray(P, float)
        for O, R in turns:
            P = (P - O) @ R.T + O
        return P + d
    return M


def _fit_sphere_srf(G, size):
    """(centre, radius) when the surface is a sphere (a grid of it fits one
    within 1e-7 of the size), else None."""
    np = _np()
    (u0, u1), (v0, v1) = G.u_domain(), G.v_domain()
    P = np.array([G.evaluate(float(u), float(v)) for u in np.linspace(u0, u1, 13) for v in np.linspace(v0, v1, 13)],
                 float)
    m = P.mean(axis=0)
    q = P - m
    sol = np.linalg.lstsq(np.c_[2 * q, np.ones(len(q))], (q * q).sum(axis=1), rcond=None)[0]
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + c @ c, 0.0)))
    if r <= 1e-9 * size or float(np.abs(np.linalg.norm(q - c, axis=1) - r).max()) > 1e-7 * max(size, r):
        return None
    return c + m, r


def _cut_curve(c, a, b):
    """The piece [a, b] of a free curve (exact)."""
    np = _np()
    from trimext import _cut, _make_curve
    d = ext.curve_cvs(c)
    p = int(d["degree"])
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    Q, U = _cut(list(np.c_[P * w[:, None], w]), [float(k) for k in d["knots"]], p, a, b)
    return _make_curve(Q, U, p)


def _fib_dirs(n):
    np = _np()
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    t = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.c_[r * np.cos(t), r * np.sin(t), z]


class _SeatTrim:
    """A face a Move Face feature sits in, trimmed again with some of its
    cycles (featregion.face_cycles), the feature's own moved by M.

    An open face: its outline and holes as loops. A face closed round (a
    cylinder's side, a sphere): the kernel trims it only with no loop across
    its seam or round a pole of its parameters but the rims that run round
    it, so the cycles are laid out in a frame -- the surface reseamed
    (exactly) where no hole is, a sphere first turned about its centre so its
    poles sit clear of every hole -- and the outline built by how many cycles
    run round that frame: two (a band: rim, seam, rim, seam back), one (a
    cap: rim, seam to the pole, back) or none (a whole sphere: the seam from
    pole to pole and back); every other cycle is a hole."""

    def __init__(self, S, rim_edges, M, size, tol, btol, rims_all=None):
        from featregion import face_cycles, pole_ends
        np = _np()
        self.S, self.M, self.tol, self.btol = S, M, tol, btol
        cycles, self.k = face_cycles(S)
        rims_all = rims_all or {}
        self.specs = []                          # (cycle, kind, role: mine | hole | outline)
        for c, kind in cycles:
            if all(e in rim_edges for e, _ in c):
                if kind not in ("inner", "pole"):
                    raise RuntimeError("the feature runs out to the edge of the face it sits in")
                role = "mine"
            elif kind == "inner" or (kind == "pole" and all(e in rims_all for e, _ in c)):
                role = "hole"
            else:
                role = "outline"
            self.specs.append((c, kind, role))
        self.combo, self.R, self.frac, self.pre = {}, None, 0.0, None
        if self.k is None:
            return
        self.pts = [M(_loop_polygon(c, 8)) if role == "mine" else _loop_polygon(c, 8) for c, _, role in self.specs]
        G = self._free()
        lo_, hi_ = (np.array(v, float) for v in S.bounding_box())
        self.size = float(np.linalg.norm(hi_ - lo_)) or size
        if self.k == 2:                          # closed both ways (a torus)
            wd = [(self._winds_uv(G, P, 0), self._winds_uv(G, P, 1)) for P in self.pts]
            if not any(a or b for a, b in wd):   # a whole torus: holes only, both seams moved clear
                self.poles, self.wind = [], [False] * len(self.specs)
                self.fracs = tuple(self._seam_frac(G, d, range(len(self.specs))) for d in (0, 1))
                return
            dirs = {0 if a else 1 for a, b in wd if a or b}
            if any(a and b for a, b in wd) or len(dirs) != 1:
                raise RuntimeError("the torus the feature sits in is cut in a way the move can't follow")
            d = dirs.pop()                       # a band round the tube (an elbow): its other seam into the cut-away part
            self.pre = (1 - d, self._outside_frac(G, 1 - d))
            self.k = d
            G = self._framed(reseam=False)
        self.poles = pole_ends(G, self.k)
        w = [self._winds_uv(G, P) for P in self.pts]
        if sum(w) <= 2 and self._clear_of_poles(G):
            self.wind = w
        else:
            sph = _fit_sphere_srf(G, size)
            if sph is None or len(self.poles) != 2:
                raise RuntimeError("the feature would run over a pole of the face it sits in")
            self._sphere_frame(G, *sph)
        # a seam clear of every cycle that does not run round the frame
        self.frac = self._seam_frac(self._framed(reseam=False), self.k,
                                    [i for i in range(len(self.specs)) if not self.wind[i]])

    def _outside_frac(self, G, e):
        """A share of direction e's domain whose whole iso lies outside the
        face (in the part a band of a torus was cut from)."""
        np = _np()
        lo, hi = (G.u_domain(), G.v_domain())[e]
        dlo, dhi = (G.u_domain(), G.v_domain())[1 - e]
        ts = np.linspace(lo, hi, 64, endpoint=False)
        out = []
        for t in ts:
            cl = [self.S.classify_uv(*((float(t), float(q)) if e == 0 else (float(q), float(t))))
                  for q in np.linspace(dlo, dhi, 9, endpoint=False)]
            out.append(all(c == "outside" for c in cl))
        if not any(out):
            raise RuntimeError("the torus band the feature sits in has no part cut away to put its seam in")
        best, run, start = None, 0, 0
        for i in range(2 * len(ts)):                 # the middle of the longest run (round the end)
            if out[i % len(ts)]:
                if run == 0:
                    start = i
                run += 1
                if best is None or run > best[0]:
                    best = (run, start)
            else:
                run = 0
        n, i0 = best
        n = min(n, len(ts))
        t = lo + ((i0 + (n - 1) / 2.0) % len(ts)) * (hi - lo) / len(ts)
        return float((t - lo) / (hi - lo))

    def _seam_frac(self, G, d, idx):
        """Where (a share of direction d's domain) to put G's seam in d so it
        crosses none of the cycles idx (as they will be)."""
        lo, hi = (G.u_domain(), G.v_domain())[d]
        P = hi - lo
        spans = []
        for i in idx:
            cs = [float((uv.y if d else uv.x)) for uv in
                  (sm.surface_closest_point(G, tuple(map(float, q)))[1][0] for q in self.pts[i])]
            un = [cs[0]]
            for x in cs[1:]:
                dd = (x - un[-1]) % P
                un.append(un[-1] + (dd - P if dd > P / 2 else dd))
            spans.append((min(un), max(un)))
        m = 0.02 * P

        def clear(t):
            return all((t - a + m) % P > (b - a) + 2 * m for a, b in spans)
        seam = lo
        if not clear(lo):
            cand = [(b + (a2 - b) % P / 2.0, (a2 - b) % P) for a, b in spans for a2, _ in spans]
            cand = [(t, g) for t, g in cand if clear(t)]
            if not cand:
                raise RuntimeError("the holes and features go all round the face they sit in")
            seam = max(cand, key=lambda x: x[1])[0]
        frac = ((seam - lo) % P) / P
        return 0.0 if frac > 1 - 1e-9 else frac

    # -- the frame ----------------------------------------------------------------
    def _winds_uv(self, G, Q, d=None):
        np = _np()
        d = self.k if d is None else d
        lo, hi = (G.u_domain(), G.v_domain())[d]
        P = hi - lo
        cs = [float(uv.y if d else uv.x) for uv in
              (sm.surface_closest_point(G, tuple(map(float, q)))[1][0] for q in Q)]
        net = 0.0
        for a, b in zip(cs, cs[1:] + cs[:1]):
            d = (b - a) % P
            net += d - P if d > P / 2 else d
        return abs(net) > P / 2

    def _pole_points(self, G):
        lo, hi = (G.u_domain(), G.v_domain())[self.k]
        c = (lo + hi) / 2
        return [(G.evaluate(c, e) if self.k == 0 else G.evaluate(e, c)) for e in self.poles]

    def _clear_of_poles(self, G):
        np = _np()
        Q = np.vstack(self.pts)
        return all(float(np.linalg.norm(Q - np.array(p, float), axis=1).min()) > 0.02 * self.size
                   for p in self._pole_points(G))

    def _sphere_frame(self, G, O, r):
        """Turn the sphere about its centre so both poles sit clear of every
        cycle and at most two cycles run round its axis."""
        np = _np()
        p0, p1 = (np.array(p, float) for p in self._pole_points(G))
        a0 = (p1 - p0) / np.linalg.norm(p1 - p0)
        U = [(Q - O) / np.linalg.norm(Q - O, axis=1)[:, None] for Q in self.pts]
        best = None
        for d in np.vstack([a0[None], _fib_dirs(160)]):
            x = np.cross(d, [1.0, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1.0, 0])
            x /= np.linalg.norm(x)
            y = np.cross(d, x)
            w = []
            for V in U:
                ang = np.arctan2(V @ y, V @ x)
                dd = (np.diff(np.r_[ang, ang[:1]]) + np.pi) % (2 * np.pi) - np.pi
                w.append(abs(float(dd.sum())) > np.pi)
            clear = min(float(np.arccos(np.clip(np.abs(V @ d), 0.0, 1.0).max())) for V in U)   # to the nearer pole
            if sum(w) <= 2 and (best is None or clear > best[0]):
                best = (clear, d, w)
        if best is None or best[0] < 0.02:
            raise RuntimeError("the feature would run over a pole of the face it sits in")
        _, d, w = best
        ax = np.cross(a0, d)
        s_, c_ = float(np.linalg.norm(ax)), float(a0 @ d)
        if s_ < 1e-12:
            if c_ > 0:
                Rm = np.eye(3)
            else:
                ax = np.cross(a0, [1.0, 0, 0]) if abs(a0[0]) < 0.9 else np.cross(a0, [0, 1.0, 0])
                Rm = _rotation(ax / np.linalg.norm(ax), np.pi)
        else:
            Rm = _rotation(ax / s_, np.arctan2(s_, c_))
        self.R = (np.asarray(O, float), Rm)
        self.wind = w

    def _free(self):
        return ext.surface_from_cvs(*_srf_tuple(ext.surface_cvs(self.S.surface())))

    def _framed(self, reseam=True):
        G = self._free()
        if self.pre is not None and self.pre[1] > 0.0:
            G = _reseamed(G, "uv"[self.pre[0]], self.pre[1])
        if self.R is not None:
            G = _rotated_surface(G, *self.R)
        if reseam and self.k == 2:
            for d, f in enumerate(self.fracs):
                if f > 0.0:
                    G = _reseamed(G, "uv"[d], f)
        elif reseam and self.k is not None and self.frac > 0.0:
            G = _reseamed(G, "uv"[self.k], self.frac)
        return G

    def _surface(self):
        return self._framed()

    # -- the loops ------------------------------------------------------------------
    def _curves(self, spec):
        from trimext import _edge_curve
        c, _, role = spec
        out = [_edge_curve(e, fw) for e, fw in c if e.length() > self.tol]
        return [_rigid_curve(x, self.M) for x in out] if role == "mine" else out

    def _rim_from(self, G, seam, spec):
        """(curves of a cycle running round the frame, from where it crosses
        the seam, that point)."""
        from trimext import _piece
        np = _np()
        curves = self._curves(spec)
        for j, c in enumerate(curves):
            try:
                pts, t1s, _ = sm.intersect_curves(c, seam)
            except Exception:
                continue
            if not pts:
                continue
            X = np.array(pts[0], float)
            _, t, _ = sm.curve_closest_point(c, tuple(map(float, X)))
            t0, t1 = c.parameter_range()
            t = min(max(float(t), t0), t1)
            out = []
            if abs(t1 - t) > 1e-9 * abs(t1 - t0):
                out.append(_cut_curve(c, t, t1))
            out += [ext.curve_from_cvs(*_cv_tuple(ext.curve_cvs(x))) for x in curves[j + 1:] + curves[:j]]
            if abs(t - t0) > 1e-9 * abs(t1 - t0):
                out.append(_cut_curve(c, t0, t))
            return [x for x in out if x.length() > self.tol], X
        raise RuntimeError("a rim round the face the feature sits in does not cross its seam")

    def _outline(self, G, form, W, combo):
        """The outline loop of a closed face in its frame."""
        from trimext import _iso_piece
        np = _np()
        rev = lambda cs: [_reversed_curve(x) for x in reversed(cs)]
        val = (G.u_domain(), G.v_domain())[self.k][0]
        olo, ohi = (G.u_domain(), G.v_domain())[1 - self.k]
        seam = _boundary_iso(G, self.k == 1, False)

        def q(X):
            a = sm.surface_closest_point(G, tuple(map(float, X)))[1][0]
            return float(a.y) if self.k == 0 else float(a.x)
        iso = lambda a, b: _iso_piece(G, self.k == 0, val, a, b)
        if form == "band":
            (r1, X1), (r2, X2) = (self._rim_from(G, seam, self.specs[i]) for i in W)
            r1 = rev(r1) if combo[0] else r1
            r2 = rev(r2) if combo[1] else r2
            return r1 + [iso(q(X1), q(X2))] + r2 + [iso(q(X2), q(X1))]
        if form == "cap":
            r1, X1 = self._rim_from(G, seam, self.specs[W[0]])
            r1 = rev(r1) if combo[0] else r1
            pole = ohi if combo[1] else olo
            return r1 + [iso(q(X1), pole), iso(pole, q(X1))]
        a, b = (ohi, olo) if combo[0] else (olo, ohi)        # a whole sphere: pole to pole and back
        return [iso(a, b), iso(b, a)]

    def face(self, which, area=None):
        """The face trimmed with the cycles which(spec) keeps (the outline
        always); area: the one it must come out with (it settles which way
        a closed face's outline runs)."""
        chosen = [i for i, sp in enumerate(self.specs) if sp[2] == "outline" or which(sp)]
        if self.k == 2:
            return self._torus_face(chosen, area)
        if self.k is None or (not self.poles and not any(self.wind[i] for i in chosen)):
            outer = next(i for i, sp in enumerate(self.specs) if sp[1] == "outer")
            loops = [self._curves(self.specs[outer])] + [self._curves(self.specs[i]) for i in chosen if i != outer]
            return self._trim(self._surface(), loops)
        W = [i for i in chosen if self.wind[i]]
        holes = [i for i in chosen if not self.wind[i]]
        form = {2: "band", 1: "cap", 0: "sphere"}.get(len(W))
        if form is None:
            raise RuntimeError("more than two rims run round the face the feature sits in")
        combos = [self.combo[form]] if form in self.combo else \
            ([(False, False), (False, True), (True, False), (True, True)] if form != "sphere" else [(False,), (True,)])
        err = None
        for combo in combos:
            try:
                G = self._surface()
                loops = [self._outline(G, form, W, combo)] + [self._curves(self.specs[i]) for i in holes]
                out = self._trim(self._surface(), loops)
            except Exception as exc:
                err = exc
                continue
            got = float(sum(f.area(1e-6) for f in out.faces()))
            if out.face_count() == 1 and (area is None or abs(got - area) <= 1e-4 * area + self.tol):
                self.combo[form] = combo
                return out
        raise err or RuntimeError("the face the feature sits in does not trim again")

    def _torus_face(self, chosen, area):
        """A whole torus with holes: its outline the two seam circles, as the
        kernel gives it (A, B, A back, B back)."""
        rev = _reversed_curve
        orders = [self.combo["torus"]] if "torus" in self.combo else [0, 1, 2, 3]
        err = None
        for order in orders:
            try:
                G = self._surface()
                A, B = _boundary_iso(G, True, False), _boundary_iso(G, False, False)
                outline = [[A, B, rev(A), rev(B)], [B, A, rev(B), rev(A)], [rev(A), B, A, rev(B)],
                           [rev(B), A, B, rev(A)]][order]
                out = self._trim(self._surface(), [outline] + [self._curves(self.specs[i]) for i in chosen])
            except Exception as exc:
                err = exc
                continue
            got = float(sum(f.area(1e-6) for f in out.faces()))
            if out.face_count() == 1 and (area is None or abs(got - area) <= 1e-4 * area + self.tol):
                self.combo["torus"] = order
                return out
        raise err or RuntimeError("the torus the feature sits in does not trim again")

    def _trim(self, G, loops):
        return ext.trim_surface_with_3d_curves_tol(G, [len(l) for l in loops], [c for l in loops for c in l],
                                                  10 * self.tol, self.btol)


def _p3(p):
    np = _np()
    return np.array([p.x, p.y, p.z], float) if hasattr(p, "x") else np.array(p, float)


def _reseat_motion(job, seats):
    """(M, rotation (axis, degrees) or None, A, A2) of a Move Face on a free-form
    face: the frame at the anchor's foot on its seat (its point and outward
    normal) moved to the foot of anchor + vector; with follow, turned so the
    feature keeps square to the face."""
    np = _np()
    rs = job["reseat"]
    anchor = np.asarray(rs["anchor"], float)
    S0 = min(seats, key=lambda S: S.closest_point(tuple(map(float, anchor)))[1])
    G0 = S0.surface()
    A = _p3(sm.surface_closest_point(G0, tuple(map(float, anchor)))[0])
    A2 = _p3(sm.surface_closest_point(G0, tuple(map(float, A + np.asarray(rs["vector"], float))))[0])
    n0, n1 = _face_normal(S0, A), _face_normal(S0, A2)
    R, rot = np.eye(3), None
    ax = np.cross(n0, n1)
    if rs.get("follow", True) and np.linalg.norm(ax) > 1e-12:
        ang = float(np.arctan2(np.linalg.norm(ax), float(n0 @ n1)))
        R = _rotation(ax / np.linalg.norm(ax), ang)
        rot = (ax / np.linalg.norm(ax), float(np.degrees(ang)))
    return (lambda P: (np.asarray(P, float) - A) @ R.T + A2), rot, A, A2


def _move_reseat(job, body, region, seats, rim_edges, rims_all, M, rot, A, A2, size, tol, btol):
    """Move Face on free-form seats: the feature as a closed tool -- its faces,
    the walls continued straight past each rim (their own rulings: an
    Extrude's walls), capped with the seat's patch inside the rim moved along
    -- carried rigidly, then cut from (a hole, a pocket) or joined to (a boss)
    the body with the feature filled in: the boolean meets the walls with the
    seat exactly, so the rim takes the face's shape where it lands."""
    np = _np()
    from trimext import _edge_curve
    faces = body.faces()
    fsize = float(np.linalg.norm(np.ptp(np.vstack([_loop_polygon([(e, True)], 6) for e in rim_edges]), axis=0)))
    # the rims, each with the seat it sits in
    sts = {S: _SeatTrim(S, rim_edges, M, size, tol, btol, rims_all) for S in seats}
    rims = [(S, sp) for S, st in sts.items() for sp in st.specs if sp[2] == "mine"]
    qc = np.mean([np.array(f.interior_point(), float) for f in region], axis=0)
    # the walls' straight direction: their side edges (from a rim vertex out)
    rim_v = {}
    for S, (c, _, _) in rims:
        for e, _ in c:
            for v in e.vertices():
                rim_v[v] = (S, c)
    sides, d = [], None
    for F in region:
        for e in F.edges():
            if e in rim_edges or e in sides:
                continue
            ends = [v for v in e.vertices() if v in rim_v]
            if not ends:
                continue
            a, b = np.array(e.start_vertex().point(), float), np.array(e.end_vertex().point(), float)
            if abs(e.length() - float(np.linalg.norm(b - a))) > 1e-6 * max(e.length(), size):
                raise RuntimeError("the feature's walls are not straight (drafted or rounded): on a free-form "
                                   "face only a feature with straight walls (an Extrude's) can move")
            u = (b - a) / np.linalg.norm(b - a)
            if d is not None and np.linalg.norm(np.cross(u, d)) > 1e-6:
                raise RuntimeError("the feature's walls do not run one way: on a free-form face only a "
                                   "feature with straight walls (an Extrude's) can move")
            d = u if d is None else d
            sides.append(e)
    if d is None:
        raise RuntimeError("the feature's walls have no straight side to continue past the face")
    # each rim: which way is out of the feature (away from its middle), and how far to go
    out = {}
    for S, (c, _, _) in rims:
        P = _loop_polygon(c, 8)
        pc = P.mean(axis=0)
        s_ = 1.0 if float((pc - qc) @ d) > 0 else -1.0
        G = S.surface()
        Pm = M(P)
        dev = max(float(sm.surface_closest_point(G, tuple(map(float, q)))[2]) for q in Pm)
        out[id(c)] = (s_ * d, 2.0 * dev + 0.25 * fsize)
    n_anchor = _face_normal(min(seats, key=lambda S: S.closest_point(tuple(map(float, A)))[1]), A)
    S_a = min(seats, key=lambda S: S.closest_point(tuple(map(float, A)))[1])
    adds = float((qc - A) @ n_anchor) > 0           # the feature stands out of the face: a boss
    # where each moved rim lands: inside its seat, clear of its other holes
    for S, st in sts.items():
        mine = [sp for sp in st.specs if sp[2] == "mine"]
        if not mine:
            continue
        G = S.surface()
        Q = np.vstack([M(_loop_polygon(c, 8)) for c, _, _ in mine])
        Q = np.array([_p3(sm.surface_closest_point(G, tuple(map(float, q)))[0]) for q in Q], float)
        F0 = st.face(lambda sp: False).faces()[0]
        cls = lambda F, P: [F.classify_uv(*_uvxy(sm.surface_closest_point(F.surface(), tuple(map(float, q)))[1][0]))
                            for q in P]
        if any(c_ != "inside" for c_ in cls(F0, Q)):
            raise RuntimeError("the feature would leave the face it sits in (or touch its edge)")
        if any(sp[2] == "hole" for sp in st.specs):
            F1 = st.face(lambda sp: sp[2] != "mine").faces()[0]
            if any(c_ != "inside" for c_ in cls(F1, Q)):
                raise RuntimeError("the feature would run into another hole or feature in that face")
    # the tool: walls continued straight past their rims, capped
    ext_d = 4.0 * max(e_ for _, e_ in out.values()) + 10 * tol
    vmove = {}
    for S, (c, _, _) in rims:
        for e, _ in c:
            for v in e.vertices():
                vmove[v] = out[id(c)]
    emove = {e: out[id(c)] for S, (c, _, _) in rims for e, _ in c}
    idx = [i for i, f in enumerate(faces) if f in region]
    built = []
    for F in [faces[i] for i in idx]:
        if not any(e in emove or e in sides for e in F.edges()):
            built.append(None)
            continue
        loops = []
        for loop in F.boundary_loops():               # in the loop's own order (a closed wall's seam form)
            cs = []
            for e, fw in loop:
                if e in emove:
                    dv, ev = emove[e]
                    cs.append(_moved_curve(_edge_curve(e, fw), ev * dv))
                elif e in sides:
                    a, b = np.array(e.start_vertex().point(), float), np.array(e.end_vertex().point(), float)
                    if e.start_vertex() in vmove:
                        a = a + vmove[e.start_vertex()][1] * vmove[e.start_vertex()][0]
                    if e.end_vertex() in vmove:
                        b = b + vmove[e.end_vertex()][1] * vmove[e.end_vertex()][0]
                    a, b = (a, b) if fw else (b, a)
                    cs.append(ext.curve_from_cvs(1, [tuple(map(float, a)), tuple(map(float, b))],
                                                 [0.0, 0.0, 1.0, 1.0], None))
                elif e.length() > tol:
                    cs.append(_edge_curve(e, fw))
            loops.append(cs)
        srf = _extended_surface(F.surface(), ext_d, tol)
        for lp in loops:                              # the continued wall must lie on its own surface
            for cv in lp:
                t0, t1 = cv.parameter_range()
                for t in np.linspace(t0, t1, 5):
                    if sm.surface_closest_point(srf, tuple(map(float, cv.evaluate(float(t)))))[2] > 1e3 * tol:
                        raise RuntimeError("the feature's walls are not straight (drafted or rounded): on a "
                                           "free-form face only a feature with straight walls (an Extrude's) "
                                           "can move")
        built.append(ext.trim_surface_with_3d_curves_tol(srf, [len(l) for l in loops], [x for l in loops for x in l],
                                                         10 * tol, btol))
    tool = body.copy()
    tf = tool.faces()
    redo = {i for i, nb in zip(idx, built) if nb is not None}
    ext.delete_faces(tool, [tf[i] for i in range(len(tf)) if i not in set(idx) or i in redo])
    for nb in built:
        if nb is not None:
            ext.append_brep(tool, nb)
    for S, (c, _, _) in rims:
        dv, ev = out[id(c)]
        G = ext.surface_from_cvs(*_srf_tuple(ext.surface_cvs(S.surface())))
        plug = ext.trim_surface_with_3d_curves_tol(G, [len(c)], [_edge_curve(e, fw) for e, fw in c], 10 * tol, btol)
        ext.append_brep(tool, sm.translate(plug, tuple(map(float, ev * dv))))
    tool = sm.stitch_advanced(tool, 10 * tol, squeeze_small_edges=False, making_manifold_solid=True)[0]
    if not tool.is_manifold_solid():
        raise RuntimeError("the feature does not close into a solid when continued past the face")
    if rot is not None:
        sm.rotate(tool, tuple(map(float, A)), tuple(map(float, rot[0])), rot[1])
    tool = sm.translate(tool, tuple(map(float, A2 - A)))
    # the body with the feature filled in
    b = body.copy()
    bf = b.faces()
    gone = set(idx) | {i for i, f in enumerate(faces) if f in seats}
    keep_pt = next(np.array(bf[i].interior_point(), float) for i in range(len(bf)) if i not in gone)
    filled = [st.face(lambda sp: sp[2] != "mine") for st in sts.values()]
    ext.delete_faces(b, [bf[i] for i in gone])
    for nb in filled:
        ext.append_brep(b, nb)
    b = sm.stitch_advanced(b, 10 * tol, squeeze_small_edges=False, making_manifold_solid=True)[0]
    try:
        sm.unify_normals(b, _face_at(b, keep_pt), 100)
    except Exception:
        pass
    if not b.is_manifold_solid():
        raise RuntimeError("the face does not close up where the feature was")
    try:
        res = sm.boolean(b, tool, operation=sm.BooleanOp.UNION if adds else sm.BooleanOp.DIFFERENCE)
    except Exception as exc:
        raise RuntimeError(f"the feature does not go into the face where it lands ({exc})")
    if res is None or not res.is_manifold_solid():
        raise RuntimeError("the moved feature and the face do not close into a solid")
    if res.face_count() != body.face_count():
        raise RuntimeError("the feature would run into something where it lands (the faces came out "
                           f"{res.face_count()} instead of {body.face_count()})")
    return res


def job_move_face(job):
    """Move Face (world space): the feature holding the faces at
    job['face_pts'] -- the faces enclosed by an inner loop of the face(s) it
    sits in (featregion: a hole, pocket, boss, slot, letter) -- carried along
    those seat faces by a rigid motion: turned about axes (job['turns']:
    [[point, direction, degrees], ...] in order, or one job['angle'] about
    job['axis_point'] / job['axis_dir']), then slid by job['vector']. The
    feature moves rigidly, each seat keeps its surface and is trimmed again
    with that loop moved: exact, the volume stays. The motion must carry the
    feature's rim along every seat's surface (a flat face: any slide in it; a
    cylinder: a turn about and a slide along its axis; a surface of
    revolution (a cone, a torus): a turn about its axis; a sphere: any turn
    about its centre);
    the moved loop must stay inside its
    seat, clear of its other holes. Written in the local space of
    job['into']."""
    np = _np()
    from featregion import feature_region
    from trimext import brep_tol
    body = world_brep(job, "target")
    if not bool(body.is_manifold_solid()):
        raise RuntimeError("Move Face needs a closed solid (all edges shared by two faces)")
    lo, hi = (np.array(v, float) for v in body.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    tol = 1e-6 * size
    reseat = job.get("reseat")
    M = _feature_motion(job, size) if not reseat else None
    if M is None and not reseat:
        raise RuntimeError("the move is 0")
    faces = body.faces()
    start = {_face_at(body, p) for p in job["face_pts"]}
    from featregion import rims as _rims
    rims_all = _rims(body)
    reg = feature_region(body, start, rims_all)
    if reg is None:
        raise RuntimeError("the face is not part of a feature sitting inside a face (a hole, pocket, boss "
                           "or slot): Push / Pull or Rotate Face move a face of the body itself")
    region, seats, rim_edges = reg
    btol = brep_tol(1e-5, size)
    if reseat:                                    # a free-form face: carried by frames, the rim cut anew
        if float(np.linalg.norm(np.asarray(reseat["vector"], float))) <= tol:
            raise RuntimeError("the move is 0")
        M, rot, A, A2 = _reseat_motion(job, seats)
        b = _move_reseat(job, body, region, seats, rim_edges, rims_all, M, rot, A, A2, size, tol, btol)
        save(to_local(b, job.get("xforms", {}).get(job.get("into"))), job["out"]["result"])
        return {"faces": b.face_count(), "volume": _volume(b, 1e-3), "feature": len(region),
                "seats": len(seats), "turned": rot[1] if rot else 0.0}

    def free_surface(S):
        return ext.surface_from_cvs(*_srf_tuple(ext.surface_cvs(S.surface())))

    def off_surface(G, P):
        return max(float(sm.surface_closest_point(G, tuple(map(float, q)))[2]) for q in P)

    def classes(F, P):
        G = F.surface()
        out = []
        for q in P:
            uvs = sm.surface_closest_point(G, tuple(map(float, q)))[1]
            out.append(F.classify_uv(float(uvs[0].x), float(uvs[0].y)) if uvs else "outside")
        return out

    # the seats again, their loops round the feature moved
    built = []
    for S in seats:
        G = free_surface(S)
        st = _SeatTrim(S, rim_edges, M, size, tol, btol, rims_all)
        mine_specs = [sp for sp in st.specs if sp[2] == "mine"]
        others = [sp for sp in st.specs if sp[2] == "hole"]
        for c, _, _ in mine_specs:
            P = _loop_polygon(c, 12)
            if off_surface(G, M(P)) > off_surface(G, P) + max(btol, 1e-5 * size):
                raise RuntimeError("the move takes the feature off the face it sits in: slide it along "
                                   "a flat face, slide or turn it about a cylinder's axis, turn or tilt it "
                                   "on a sphere, turn it about a cone's or a torus's axis (Push / Pull "
                                   "lifts or sinks a face)")
        A0 = float(S.area(1e-6))
        try:
            nb = st.face(lambda sp: True, area=A0)
        except Exception:
            nb = None
        # the moved loops inside the seat with the feature filled in, clear of its other holes
        Q = np.vstack([M(_loop_polygon(c, 8)) for c, _, _ in mine_specs])
        if any(c != "inside" for c in classes(st.face(lambda sp: False).faces()[0], Q)):
            raise RuntimeError("the feature would leave the face it sits in (or touch its edge)")
        if others:
            hit = any(c != "inside" for c in classes(st.face(lambda sp: sp[2] != "mine").faces()[0], Q))
            if not hit:                              # another hole swallowed by the moved feature
                T = st.face(lambda sp: sp[2] == "mine").faces()[0]
                hit = any(c == "outside" for c in
                          classes(T, np.vstack([_loop_polygon(c, 8) for c, _, _ in others])))
            if hit:
                raise RuntimeError("the feature would run into another hole or feature in that face")
        if nb is None:
            raise RuntimeError("the face the feature sits in does not trim again round the moved feature")
        A1 = float(sum(f.area(1e-6) for f in nb.faces()))
        if nb.face_count() != 1 or abs(A1 - A0) > 1e-4 * A0 + tol:
            raise RuntimeError("the feature would run into another hole or feature in that face")
        built.append(nb)
    # the feature's faces, moved
    idx = [i for i, f in enumerate(faces) if f in region]
    sheet = body.copy()
    sf = sheet.faces()
    ext.delete_faces(sheet, [sf[i] for i in range(len(sf)) if i not in set(idx)])
    for O, z, ang in _feature_turns(job):
        sm.rotate(sheet, tuple(map(float, O)), tuple(map(float, z)), ang)
    d = np.asarray(job.get("vector", (0.0, 0.0, 0.0)), float)
    if np.linalg.norm(d) > 0:
        sheet = sm.translate(sheet, tuple(map(float, d)))
    gone = set(idx) | {i for i, f in enumerate(faces) if f in seats}
    b = body.copy()
    bf = b.faces()
    keep_pts = [np.array(bf[i].interior_point(), float) for i in range(len(bf)) if i not in gone]
    ext.delete_faces(b, [bf[i] for i in gone])
    for nb in built + [sheet]:
        ext.append_brep(b, nb)
    b = sm.stitch_advanced(b, 10 * tol, squeeze_small_edges=False, making_manifold_solid=True)[0]
    try:
        sm.unify_normals(b, _face_at(b, keep_pts[0]), 100)
    except Exception:
        pass
    if not b.is_manifold_solid():
        raise RuntimeError("the moved feature and its face do not close into a solid")
    V0, V1 = _vol(body), _vol(b)
    if abs(V1 - V0) > 1e-4 * abs(V0) + tol:
        raise RuntimeError(f"the move changed the volume ({V0:.6g} to {V1:.6g}): the feature runs into "
                           "something")
    save(to_local(b, job.get("xforms", {}).get(job.get("into"))), job["out"]["result"])
    return {"faces": b.face_count(), "volume": _volume(b, 1e-3), "feature": len(region), "seats": len(seats)}


def job_boolean(job):
    """Union / difference / intersection of closed solids (world space): the
    target job['order'][0] combined with each tool in turn. Written in the local
    space of job['into']."""
    keys = job["order"]
    opname = job["operation"]
    op = {"UNION": sm.BooleanOp.UNION, "DIFFERENCE": sm.BooleanOp.DIFFERENCE,
          "INTERSECTION": sm.BooleanOp.INTERSECTION}[opname]
    breps = [world_brep(job, k) for k in keys]
    names = job.get("names", {})
    for k, b in zip(keys, breps):
        try:
            solid = bool(b.is_manifold_solid())
        except Exception:
            solid = False
        if not solid:
            raise RuntimeError(f"{names.get(k, k)} is not a closed solid -- Join its surfaces into a "
                               "closed body first (all edges shared by two faces)")
    res = breps[0]
    for k, b in zip(keys[1:], breps[1:]):
        try:
            out = sm.boolean(res, b, operation=op)
        except Exception as exc:
            raise RuntimeError(f"the boolean with {names.get(k, k)} failed ({exc})")
        if out is None or out.face_count() == 0:
            if opname == "INTERSECTION":
                raise RuntimeError(f"nothing left: {names.get(k, k)} does not overlap the body")
            if opname == "DIFFERENCE":
                raise RuntimeError(f"nothing left: {names.get(k, k)} encloses the whole body")
            raise RuntimeError(f"the boolean with {names.get(k, k)} gave an empty result")
        res = out
    try:
        volume = abs(float(res.volume(1e-6)))
    except Exception:
        volume = 0.0
    save(to_local(res, job.get("xforms", {}).get(job.get("into"))), job["out"]["result"])
    return {"faces": res.face_count(), "volume": volume}


def _seg_dist(P, pts):
    """Distance of each point of pts to the polyline P."""
    np = _np()
    if len(P) < 2:
        return np.full(len(pts), np.inf)
    A, S = P[:-1], P[1:] - P[:-1]
    ss = np.maximum(np.einsum("ij,ij->i", S, S), 1e-300)
    out = []
    for q in pts:
        t = np.clip(np.einsum("ij,ij->i", q - A, S) / ss, 0.0, 1.0)
        out.append(float(np.min(np.linalg.norm(A + S * t[:, None] - q, axis=1))))
    return np.array(out)


def job_boundary_fill(job):
    """Boundary fill (Rhino's CreateSolid): the closed solid the selected
    surfaces enclose. Every surface is split along the others; pieces with an
    edge that no piece of another surface shares (overhangs, the parts outside
    the enclosure) are dropped until none is left; the rest is stitched into a
    solid. World space in; local space of job['into'] out."""
    np = _np()
    keys = job["order"]
    names = job.get("names", {})
    worlds = {k: world_brep(job, k) for k in keys}
    lo = np.min([np.array(worlds[k].bounding_box()[0], float) for k in keys], axis=0)
    hi = np.max([np.array(worlds[k].bounding_box()[1], float) for k in keys], axis=0)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    tol = 1e-4 * size

    pieces = []                                   # (source index, brep)
    for i, k in enumerate(keys):
        sub = {"breps": dict(job["breps"], target=job["breps"][k]),
               "xforms": dict(job.get("xforms", {}), target=job.get("xforms", {}).get(k)),
               "ref_point": [float(x) for x in worlds[k].bounding_box()[0]]}
        cutters = []
        for j, other in enumerate(keys):
            if j == i:
                continue
            c = SurfaceCutter(sub, other)
            try:              # keep only surfaces that really cut this one
                n0 = worlds[k].face_count()
                cut = c.split(world_brep(job, k), None)
                if cut is not None and cut.face_count() > n0:
                    cutters.append(c)
            except Exception:
                pass
        if not cutters:
            raise RuntimeError(f"{names.get(k, k)} does not cross any other selected surface")
        b, faces, groups, _, _ = _regions(sub, cutters)
        for g in groups:
            part = b.copy()
            drop = [f for fi, f in enumerate(part.faces()) if fi not in set(g)]
            if drop:
                ext.delete_faces(part, drop)
            pieces.append((i, part))

    # each piece's open edges, as polylines
    def key(e):
        m = _edge_mid(e)
        return None if m is None else tuple(round(float(v) / tol) for v in m)

    def open_edges(part):
        # a seam (a closed face meeting itself) is used twice by its face's loop
        seams = set()
        for f in part.faces():
            seen = set()
            for loop in f.boundary_loops():
                for e, _ in loop:
                    kk = key(e)
                    if kk in seen:
                        seams.add(kk)
                    seen.add(kk)
        out = []
        for e in part.edges():
            if len(e.faces()) < 2 and key(e) not in seams:
                P = np.array(_edge_poly(e, 24), float)
                if len(P) >= 2:
                    out.append(P)
        return out
    edges = [open_edges(part) for _, part in pieces]
    alive = set(range(len(pieces)))
    while True:
        drop = set()
        for a in alive:
            src = pieces[a][0]
            for ei, P in enumerate(edges[a]):
                probe = P[1:-1:3] if len(P) > 4 else P
                covered = False
                # a seam: another open edge of the same piece lies on it
                for ej, Q in enumerate(edges[a]):
                    if ej != ei and (_seg_dist(Q, probe) <= tol).all():
                        covered = True
                        break
                if not covered:
                    for b_ in alive:
                        if b_ == a or pieces[b_][0] == src:
                            continue
                        if any((_seg_dist(Q, probe) <= tol).all() for Q in edges[b_]):
                            covered = True
                            break
                if not covered:
                    drop.add(a)
                    break
        if not drop:
            break
        alive -= drop
    if len({pieces[a][0] for a in alive}) < 2:
        raise RuntimeError("the surfaces do not enclose a volume (every piece has an open edge) -- "
                           "extend them so that they cross each other all round")
    parts = [pieces[a][1] for a in sorted(alive)]
    merged = sm.merge_breps(parts, operation=sm.BooleanOp.MERGE) if len(parts) > 1 else parts[0]
    res = sm.stitch_advanced(merged, tol, squeeze_small_edges=False, making_manifold_solid=True)
    b = res[0]
    try:
        solid = bool(b.is_manifold_solid())
    except Exception:
        solid = False
    if not solid:
        n_open = sum(1 for e in b.edges() if len(e.faces()) < 2)
        n_multi = sum(1 for e in b.edges() if len(e.faces()) > 2)
        if n_multi:
            raise RuntimeError("the surfaces enclose more than one volume -- select only the surfaces "
                               "around the one you want")
        raise RuntimeError(f"the pieces do not close up ({n_open} open edges)")
    try:
        sm.unify_normals(b, max(b.faces(), key=lambda f: f.area()), 100)
    except Exception:
        pass
    volume = 0.0
    try:
        h = sm.heal_brep(b)
        v = h.volume(1e-6) if h is not None else 0.0
        if h is not None and abs(v) > 0:
            b, volume = h, abs(v)
    except Exception:
        pass
    save(to_local(b, job.get("xforms", {}).get(job.get("into"))), job["out"]["result"])
    return {"faces": b.face_count(), "volume": volume, "pieces": len(pieces), "kept": len(alive)}


def job_unjoin(job):
    """One brep per face (local space of the source object)."""
    b = load(job["breps"]["target"])
    n = b.face_count()
    if n < 2:
        raise RuntimeError("this object has a single face -- nothing to unjoin")
    areas = []
    for i in range(n):
        part = b.copy()
        pf = part.faces()
        ext.delete_faces(part, [f for j, f in enumerate(pf) if j != i])
        save(part, os.path.join(job["out_dir"], f"out_face{i}.sm"))
        areas.append(part.area())
    return {"faces": n, "areas": areas}


# ---------------------------------------------------------------------------
# Edge fillet by chord: exact sections (secfillet.py) where the kernel's one
# radius per edge cannot keep the rails the chord apart
# ---------------------------------------------------------------------------

class _EdgeCurve:
    """An edge as a curve over its own range (for secfillet's crossing)."""

    def __init__(self, e):
        self.e = e

    def parameter_range(self):
        return self.e.parameter_range()

    def evaluate(self, t):
        return self.e.point_at(t)


def _face_key(f):
    q = f.interior_point()
    q = (q.x, q.y, q.z) if hasattr(q, "x") else tuple(q)
    return tuple(round(float(x), 7) for x in q)


def _surface_copy(s):
    d = ext.surface_cvs(s)
    return ext.surface_from_cvs(d["degree_u"], d["degree_v"], [tuple(p) for p in d["points"]],
                                list(d["knots_u"]), list(d["knots_v"]), d.get("weights") or None)


def _surface_extended(s, distance):
    """A copy of s carried on naturally by distance past each open edge (a
    closed direction is left alone)."""
    out = _surface_copy(s)
    for edge, direction in (("U0", "u"), ("U1", "u"), ("V0", "v"), ("V1", "v")):
        if _is_closed(out, direction):
            continue
        try:
            out = ext.surface_from_cvs(*extend_grid(ext.surface_cvs(out), edge, distance))
        except Exception:
            pass
    return out


def _section_groups(b, sel, tol):
    """[(face P, face Q, [edges])]: the picked edges by the two faces they
    join, when every chain is closed or ends where no other face meets (a
    sheet's boundary). None otherwise: fillets that end against a face or
    meet at a corner are the kernel's."""
    np = _np()
    groups = {}
    for e in sel:
        fs = e.faces()
        k = tuple(sorted(_face_key(f) for f in fs))
        groups.setdefault(k, [fs[0], fs[1], []])[2].append(e)
    shared = [e for e in b.edges() if len(e.faces()) == 2]
    for fp, fq, es in groups.values():
        own = [np.array(_edge_mid(e), float) for e in es]
        pts = []
        for e in es:
            t0, t1 = e.parameter_range()
            pts += [np.array(e.point_at(t0), float), np.array(e.point_at(t1), float)]
        free = [p for p in pts if sum(float(np.linalg.norm(p - q)) <= tol for q in pts) == 1]
        for p in free:
            for e in shared:
                m = np.array(_edge_mid(e), float)
                if any(float(np.linalg.norm(m - q)) <= tol for q in own):
                    continue                # this chain's own edge
                t0, t1 = e.parameter_range()
                if min(float(np.linalg.norm(np.array(e.point_at(t), float) - p)) for t in (t0, t1)) <= tol:
                    return None
    return list(groups.values())


def _section_edge_fillet(b, sel, chord, shape, soft=None):
    """(body, info) with the picked edges filleted from exact sections at a
    constant chord, or None when the kernel has to do it (see _section_groups).
    soft = (R, smoothing): a soft fillet (shape SOFT2 / SOFT3) of minimum
    radius R instead (chord is then only the reach: how far to carry the
    faces on and to stand the trim sheets)."""
    np = _np()
    import os
    import tempfile
    import secfillet
    lo, hi = (np.array(v) for v in b.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    groups = _section_groups(b, sel, 1e-5 * size)
    if groups is None:
        return None
    fillets, radii, chords, sections, peaks = [], [], [], 0, []
    for fp, fq, es in groups:
        # the faces' surfaces carried on past their edges: a face that ends at
        # the filleted edge has no foot for the ball there otherwise (nor a
        # twisted one at the sheet's ends, where the section leans past it)
        A = ext.brep_from_surface(_surface_extended(fp.surface(), 2.0 * chord))
        B = ext.brep_from_surface(_surface_extended(fq.surface(), 2.0 * chord))
        if soft:
            f, info = secfillet.build(sm, A, B, None, None, soft[0], soft[0], shape=shape, smoothing=soft[1],
                                      curves=[_EdgeCurve(e) for e in es], on_faces=(fp, fq))
            peaks += list(info["min_radius"])
        else:
            f, info = secfillet.build(sm, A, B, None, None, chord, chord, chord=chord, shape=shape,
                                      curves=[_EdgeCurve(e) for e in es], on_faces=(fp, fq))
        fillets.append((f, A, B, fp, fq))
        radii += list(info["radius"])
        chords += list(info["chord"])
        sections += info["sections"]
    # each face the fillets touch is trimmed on its own (a trim of the joined
    # body by the rail cutter fails in the kernel's intersector), then the
    # faces, the fillets and the rest of the body are stitched again
    faces = b.faces()
    keys = [_face_key(f) for f in faces]
    involved = sorted({keys.index(_face_key(f)) for _, _, _, fp, fq in fillets for f in (fp, fq)})

    def alone(i):
        part = b.copy()
        ext.delete_faces(part, [f for j, f in enumerate(part.faces()) if j != i])
        return part
    pieces = {i: alone(i) for i in involved}
    tmp = []

    def temp(brep=None):
        fd, path = tempfile.mkstemp(suffix=".sm")
        os.close(fd)
        if brep is not None:
            save(brep, path)
        tmp.append(path)
        return path
    try:
        for f, A, B, fp, fq in fillets:
            rails = {}
            for key, face in (("a", fp), ("b", fq)):
                rails[key] = min(f.edges(), key=lambda e: face.closest_point(tuple(mid(e)))[1])
            for key, other, base, face in (("a", "b", A, fp), ("b", "a", B, fq)):
                i = keys.index(_face_key(face))
                pa, pb = mid(rails[key]), mid(rails[other])
                probe = tuple(pa[k] + 0.5 * (pb[k] - pa[k]) for k in range(3))
                ref = face.closest_point(probe)[0]
                sheet = _rail_sheet(rails[key], base, max(0.25 * chord, 1e-4))
                out = temp()
                job_trim({"breps": {"target": temp(pieces[i]), "sheet": temp(sheet)}, "xforms": {"target": None},
                          "cutter_breps": ["sheet"], "ref_point": [float(x) for x in ref], "action": "DELETE",
                          "scope": "PICKED", "direction": [0.0, 0.0, 1.0], "out": {"DELETE": out},
                          "extend": True})          # an open rail ends on the face's edge: carried on across it
                pieces[i] = load(out)
        parts = [pieces[i] for i in involved] + [f for f, *_ in fillets]
        if len(involved) < len(faces):
            rest = b.copy()
            ext.delete_faces(rest, [f for j, f in enumerate(rest.faces()) if j in involved])
            parts.insert(0, rest)
        before = sum(_shared_edges(x) for x in parts)
        merged = sm.merge_breps(parts, operation=sm.BooleanOp.MERGE)
        res = sm.stitch_advanced(merged, max(1e-5, 2e-4 * size), squeeze_small_edges=False,
                                 making_manifold_solid=False)
        r = res[0]
        if _shared_edges(r) - before < 2 * len(fillets):
            raise RuntimeError("the chord fillet did not stitch to both faces")
        try:
            sm.unify_normals(r, max(r.faces(), key=lambda fc: fc.area()), 100)
        except Exception:
            pass
    finally:
        for path in tmp:
            try:
                os.remove(path)
            except OSError:
                pass
    out = {"radius": (min(radii), float(np.mean(radii)), max(radii)),
           "chord": (min(chords), float(np.mean(chords)), max(chords)), "sections": sections}
    if peaks:
        out["min_radius"] = (min(peaks), max(peaks))
    return r, out


def _measured_chords(before, r):
    """(min, mean, max) distance between the rails of the fillet faces the
    kernel made (across each face, the direction whose ends lie on the
    original faces), or None."""
    np = _np()
    try:
        on_in = _input_test(before)
        out = []
        for f in r.faces():
            if on_in(f):
                continue
            s = f.surface()
            (u0, v0), (u1, v1) = f.uv_domain()
            um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            on = lambda q: dist_to_brep(tuple(q), before)
            eu = on(s.evaluate(u0, vm)) + on(s.evaluate(u1, vm))
            ev = on(s.evaluate(um, v0)) + on(s.evaluate(um, v1))
            if eu <= ev:
                out += [float(np.linalg.norm(np.array(s.evaluate(u0, v)) - np.array(s.evaluate(u1, v))))
                        for v in np.linspace(v0, v1, 9)]
            else:
                out += [float(np.linalg.norm(np.array(s.evaluate(u, v0)) - np.array(s.evaluate(u, v1))))
                        for u in np.linspace(u0, u1, 9)]
        return (min(out), float(np.mean(out)), max(out)) if out else None
    except Exception:
        return None


def job_edge_fillet(job):
    """Fillet / chamfer / blend the edges whose world-space midpoints are given.
    Works on joined sheets and on solids; written back in the object's local space."""
    np = _np()
    b = world_brep(job, "target")
    b_orig = b.copy() if job.get("class_a") else None
    b_ring = b.copy()
    edges = b.edges()
    mids = []
    for e in edges:
        m = _edge_mid(e)
        mids.append(np.array(m) if m is not None else np.array([np.inf] * 3))
    mids = np.array(mids)
    lo, hi = (np.array(v) for v in b.bounding_box())
    # history rebuilds: upstream edits move the edges, so match more loosely
    tol = (0.08 if job.get("loose") else 1e-3) * (float(np.linalg.norm(hi - lo)) or 1.0)
    shared_idx = [i for i, e in enumerate(edges) if len(e.faces()) == 2]
    picked = []
    for m in job["edge_mids"]:
        d = np.linalg.norm(mids - np.array(m), axis=1)
        if job.get("loose") and shared_idx:   # history rebuild: nearest edge that CAN be filleted
            i = min(shared_idx, key=lambda k: d[k])
        else:
            i = int(np.argmin(d))
        if d[i] > tol and not job.get("loose"):
            raise RuntimeError("an edge could not be found again (did the object change?)")
        if len(edges[i].faces()) != 2:
            raise RuntimeError("only edges shared by two faces can be filleted -- Join the surfaces first")
        if i not in picked:
            picked.append(i)
    sel = [edges[i] for i in picked]
    if job.get("shape") == "SOFT":
        return _soft_edge_fillet(job, b, b_ring, sel)
    xsect = {"CHAMFER": 0, "CIRCULAR": 1, "BLEND": 2}[job.get("shape", "CIRCULAR")]
    r0 = float(job["radius"])
    r1 = job.get("end_radius")
    chord_info = chord_note = None
    if job.get("chord") and not (r1 is not None and abs(float(r1) - r0) > 1e-12):
        shape = {0: "LINEAR", 1: "CIRCULAR"}.get(xsect) or {1: "G1", 2: "G2", 3: "G3"}.get(int(job.get("continuity", 1)))
        sec, chord_note = None, None
        if shape and abs(float(job.get("thumbweight", 1.0)) - 1.0) < 1e-9:
            try:
                sec = _section_edge_fillet(b, sel, float(job["chord"]), shape)
                if sec is None:
                    chord_note = "the fillets meet another face or each other: one ball radius per edge"
            except Exception as exc:
                chord_note = f"exact chord failed ({exc}): one ball radius per edge"
        if sec is not None:
            r, info = sec
            try:
                solid = bool(r.is_manifold_solid())
            except Exception:
                solid = False
            new_faces = []
            try:
                on_in = _input_test(b_ring)
                for f in r.faces():
                    if not on_in(f):
                        q = f.interior_point()
                        new_faces.append([float(q.x), float(q.y), float(q.z)] if hasattr(q, "x") else [float(x) for x in q])
            except Exception:
                new_faces = []
            save(to_local(r, job.get("xforms", {}).get("target")), job["out"]["result"])
            return {"edges": len(sel), "faces": r.face_count(), "solid": solid, "new_faces": new_faces, "rings": [],
                    "rims_only": False, "radius": info["radius"][1], "radius_range": info["radius"],
                    "chord": info["chord"], "sections": info["sections"]}
    if job.get("chord"):
        # chord = distance between the rails = 2 r sin(theta / 2), theta = the
        # angle between the two faces' normals across the edge
        c = float(job["chord"])
        th = []
        for e in sel:
            t0, t1 = e.parameter_range()
            for t in np.linspace(t0, t1, 9)[1:-1]:
                p = np.array(e.point_at(t), float)
                n0, n1 = (_face_normal(fc, p) for fc in e.faces()[:2])
                th.append(float(np.arccos(np.clip(n0 @ n1, -1.0, 1.0))))
        th = np.maximum(np.array(th), np.radians(1.0))       # (a smooth edge needs no fillet)
        rs = c / (2.0 * np.sin(0.5 * th))
        r0, r1 = float(np.mean(rs)), None
        chord_info = True             # measured on the kernel's fillet below
    if r1 is not None and abs(float(r1) - r0) > 1e-12:
        r = sm.variable_radius_fillet(b, sel, start_radius=r0, end_radius=float(r1), xsect_type=xsect,
                                      continuity=int(job.get("continuity", 1)))
    else:
        r = sm.fillet_edges(b, sel, radius=r0, xsect_type=xsect, continuity=int(job.get("continuity", 1)),
                            thumbweight=float(job.get("thumbweight", 1.0)))
    if r is None or r.face_count() == 0:
        raise RuntimeError("the kernel returned no result (radius too large for the neighbouring faces?)")
    r = _fix_coiled_faces(b_ring, _fix_ring_faces(b_ring, r))
    rings = []
    if r1 is None or abs(float(r1) - r0) <= 1e-12:
        # rims (Class-A or not): one smooth ring each; Class-A leaves them be
        r, rings = _smooth_rings(b_ring, r, xsect, r0)
    rims_only = False
    if rings:            # every fillet face is a rebuilt ring: Class-A has nothing left to do
        on_in = _input_test(b_ring)
        rims_only = sum(1 for f in r.faces() if not on_in(f)) == len(rings)
    class_a = None
    if job.get("class_a"):
        r, class_a = _class_a(b_orig, r, job, xsect, r0 if r1 is None or abs(float(r1) - r0) <= 1e-12 else None,
                              skip=[g["at"] for g in rings])
        if rings:
            class_a["rings"] = [{k: v for k, v in g.items() if k != "at"} for g in rings]
    try:
        solid = bool(r.is_manifold_solid())
    except Exception:
        solid = False
    new_faces = []                        # a point inside each face the fillet made (world): its panel highlights them
    try:
        on_in = _input_test(b_ring)
        for f in r.faces():
            if not on_in(f):
                q = f.interior_point()
                new_faces.append([float(q.x), float(q.y), float(q.z)] if hasattr(q, "x") else [float(x) for x in q])
    except Exception:
        new_faces = []
    try:                                  # (Mesh to BREP's fillets) the planes beside it: down to what is left
        on_in = _input_test(b_ring) if job.get("shrink_planes") else None
        if on_in is not None:
            r = _shrink_planes(r, lambda f: on_in(f) and any(not on_in(g) for e in f.edges() for g in e.faces()))
    except Exception:
        pass
    ch = _measured_chords(b_ring, r) if chord_info is not None else None    # (world: before to_local moves r)
    save(to_local(r, job.get("xforms", {}).get("target")), job["out"]["result"])
    out = {"edges": len(sel), "faces": r.face_count(), "solid": solid, "new_faces": new_faces,
           "rings": [{k: v for k, v in g.items() if k != "at"} for g in rings], "rims_only": rims_only}
    if chord_info is not None:
        out["radius"] = r0
        if ch is not None:
            out["chord"] = ch
            if chord_note and ch[2] - ch[0] > 1e-3 * ch[1]:     # one radius is exact where the angle holds
                out["chord_note"] = chord_note
    if class_a is not None:
        out["class_a"] = class_a
    return out


def _new_face_points(before, r):
    """A point inside each face of r that is not one of before's (world)."""
    out = []
    try:
        on_in = _input_test(before)
        for f in r.faces():
            if not on_in(f):
                q = f.interior_point()
                out.append([float(q.x), float(q.y), float(q.z)] if hasattr(q, "x") else [float(x) for x in q])
    except Exception:
        return []
    return out


def _soft_edge_fillet(job, b, b_ring, sel):
    """Soft fillet (softfillet.py) of the picked edges, minimum radius
    job['radius'], job['smoothing'], G2 / G3 (job['continuity']). Edges whose
    fillets end at the sheet's boundary or close on themselves are built from
    exact sections, each sized from its own angle (secfillet). Otherwise the
    kernel's circular fillet, its ball sized from the mean angle, gives the
    rails, and every fillet strip is rebuilt with soft sections between them
    (classa); the corner patches where fillets meet are removed, and the
    openings left for Square (G2, Match Neighbours)."""
    np = _np()
    import math
    import softfillet
    cont = 3 if int(job.get("continuity", 2)) >= 3 else 2
    R = float(job["radius"])
    smooth = softfillet.clamp_smoothing(float(job.get("smoothing", 0.6)), cont)
    shape = "SOFT3" if cont == 3 else "SOFT2"
    xf = job.get("xforms", {}).get("target")
    note = None
    try:
        sec = _section_edge_fillet(b, sel, 2.0 * softfillet.ball_radius(R, 0.5 * math.pi, smooth, cont), shape,
                                   soft=(R, smooth))
    except Exception as exc:
        sec, note = None, f"exact sections failed ({exc}): kernel rails"
    if sec is not None:
        r, info = sec
        try:
            solid = bool(r.is_manifold_solid())
        except Exception:
            solid = False
        new_faces = _new_face_points(b_ring, r)
        save(to_local(r, xf), job["out"]["result"])
        return {"edges": len(sel), "faces": r.face_count(), "solid": solid, "new_faces": new_faces, "rings": [],
                "rims_only": False, "radius": R, "ball_radius": info["radius"], "min_radius": info["min_radius"],
                "smoothing": smooth, "sections": info["sections"], "soft": "sections"}
    # the kernel's rails: one ball for all the edges, from their mean angle
    th = []
    for e in sel:
        t0, t1 = e.parameter_range()
        for t in np.linspace(t0, t1, 9)[1:-1]:
            p = np.array(e.point_at(t), float)
            n0, n1 = (_face_normal(fc, p) for fc in e.faces()[:2])
            th.append(float(np.arccos(np.clip(n0 @ n1, -1.0, 1.0))))
    turn = max(float(np.mean(th)), math.radians(2.0))
    rho = softfillet.ball_radius(R, turn, smooth, cont)
    r = sm.fillet_edges(b, sel, radius=rho, xsect_type=1, continuity=1, thumbweight=1.0)
    if r is None or r.face_count() == 0:
        raise RuntimeError("the kernel returned no result (radius too large for the neighbouring faces?)")
    r = _fix_coiled_faces(b_ring, _fix_ring_faces(b_ring, r))
    import classa
    on_in = _input_test(b_ring)
    fil = [i for i, f in enumerate(r.faces()) if not on_in(f)]
    keep = r.copy()
    try:
        out, rep = classa.rebuild_body(r, sm, fil, kind="SOFT", cont=cont,
                                       tol=float(job.get("ca_tolerance", 1e-4)),
                                       ang_tol=float(job.get("ca_angle", 0.1)),
                                       max_degree=max(int(job.get("ca_degree", 7)), softfillet.degree_of(cont)),
                                       max_spans=int(job.get("ca_spans", 8)), radius=rho, soft=(smooth, cont))
    except Exception as exc:
        raise RuntimeError(f"the soft sections failed ({exc})")
    if isinstance(rep, list) or not rep.get("patches"):
        raise RuntimeError("no fillet strip could be rebuilt with soft sections")
    lo, hi = (np.array(v) for v in out.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    sl = [f for f in out.faces() if f.area() < 1e-10 * size * size]
    if sl:
        ext.delete_faces(out, sl)
        out = sm.stitch_advanced(out, rep["stitch_tol"], squeeze_small_edges=False, making_manifold_solid=False)[0]
    lamina = sum(1 for e in out.edges() if len(e.faces()) == 1)
    nonman = sum(1 for e in out.edges() if len(e.faces()) > 2)
    if lamina > rep["lamina_before"] + rep["opening_edges"] or nonman:
        raise RuntimeError(f"the soft strips did not stitch back ({lamina - rep['lamina_before']} open, "
                           f"{nonman} non-manifold edges)")
    try:
        solid = bool(out.is_manifold_solid())
    except Exception:
        solid = False
    peaks = [x["min_radius"] for x in rep["faces"] if "min_radius" in x]
    res = {"edges": len(sel), "faces": out.face_count(), "solid": solid, "new_faces": _new_face_points(b_ring, out),
           "rings": [], "rims_only": False, "radius": R, "ball_radius": (rho, rho, rho), "smoothing": smooth,
           "soft": "kernel rails", "openings": rep["openings"], "patches": rep["patches"],
           "class_a": {k: v for k, v in rep.items() if k in ("faces", "patches", "openings")}}
    if peaks:
        res["min_radius"] = (min(p[0] for p in peaks), max(p[1] for p in peaks))
    if note:
        res["note"] = note
    del keep
    save(to_local(out, xf), job["out"]["result"])
    return res


def _open_edge_count(b):
    """Open (one-face) edges of b, not counting seams (an edge a face's own
    loop uses twice)."""
    n = 0
    seams = set()
    for f in b.faces():
        seen = set()
        for loop in f.boundary_loops():
            for e, _ in loop:
                k = id(e) if _edge_mid(e) is None else tuple(round(float(v), 7) for v in _edge_mid(e))
                if k in seen:
                    seams.add(k)
                seen.add(k)
    for e in b.edges():
        m = _edge_mid(e)
        k = id(e) if m is None else tuple(round(float(v), 7) for v in m)
        if len(e.faces()) < 2 and k not in seams:
            n += 1
    return n


def _fix_ring_faces(before, r):
    """The kernel's chamfer and blend fillets of a CLOSED edge (the rim of a
    cylinder) come back with a broken trim: the face is bounded by the two
    closed rails but has no seam, so it covers a sliver of its surface and the
    fillet shows as a gap. The surface itself spans exactly rail to rail all
    the way round, so such a face is replaced by its untrimmed surface and
    stitched back in. Keeps r when that does not stitch as well."""
    np = _np()
    orig = {_surface_key(f.surface()) for f in before.faces()}
    bad = []
    for f in r.faces():
        loops = f.boundary_loops()
        if len(loops) < 2 or any(len(lp) != 1 for lp in loops):
            continue                       # only rings bounded by closed rails alone
        try:
            if _surface_key(f.surface()) in orig:
                continue                   # an input face (a tube with two rims)
            nat = ext.brep_from_surface(f.surface())
            if f.area() < 0.5 * nat.faces()[0].area():
                bad.append((f, nat))
        except Exception:
            continue
    if not bad:
        return r
    lo, hi = (np.array(v) for v in r.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    out = r.copy()
    try:
        drop = [f for f in out.faces() for g, _ in bad if _surface_key(f.surface()) == _surface_key(g.surface())]
        ext.delete_faces(out, drop)
        for _, nat in bad:
            ext.append_brep(out, nat)
        before_open = _open_edge_count(r)
        for tol in (1e-6 * size, 1e-5 * size, 1e-4 * size):
            st = sm.stitch_advanced(out.copy(), tol, squeeze_small_edges=False, making_manifold_solid=False)[0]
            if _open_edge_count(st) <= before_open and st.face_count() == r.face_count():
                try:
                    sm.unify_normals(st, max(st.faces(), key=lambda f: f.area()), 100)
                except Exception:
                    pass
                return st
    except Exception:
        pass
    return r


def _fix_coiled_faces(before, r):
    """The kernel's fillet can carry a neighbouring face on over a surface of
    its own that coils round past the input's ends (a half cylinder grown into
    one and a half turns), the trim then covering the face more than once: the
    face has the wrong area and fails to mesh -- a hole in the solid. Each
    face of r that lies on an input face's surface without being that surface
    is trimmed out of the input's surface again by its own 3D edges; where
    that gives another area the kernel's face is the broken one and the
    re-trimmed face takes its place. Keeps r when that does not stitch as well."""
    np = _np()
    lo, hi = (np.array(v) for v in before.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    tol = 1e-5 * size
    origs = [(_surface_key(f.surface()), f.surface()) for f in before.faces()]
    keys = {k for k, _ in origs}
    carriers = {}
    swap = []
    try:
        for i, f in enumerate(r.faces()):
            if _surface_key(f.surface()) in keys:
                continue
            q = f.interior_point()
            pts = [(q.x, q.y, q.z) if hasattr(q, "x") else tuple(q)]
            for e in f.edges():
                t0, t1 = e.parameter_range()
                pts += [tuple(map(float, e.point_at(t))) for t in np.linspace(t0, t1, 7)]
            for k, (_, S) in enumerate(origs):
                if k not in carriers:      # (copies: the kernel adopts a surface it is given)
                    carriers[k] = ext.brep_from_surface(_surface_copy(S))
                if any(carriers[k].faces()[0].closest_point(tuple(map(float, p)))[1] > tol for p in pts):
                    continue
                loops = [_chain([_edge_copy(e) for e, _ in lp], 10 * tol) for lp in f.boundary_loops()]
                if not loops or any(lp is None for lp in loops):
                    break
                try:
                    cand = ext.trim_surface_with_3d_curves_tol(_surface_copy(S), [len(lp) for lp in loops],
                                                               [c for lp in loops for c in lp], 10 * tol, 1e2 * tol)
                except Exception:
                    break
                if cand.face_count() == 1 and abs(cand.area(1e-6) - f.area()) > 1e-3 * f.area():
                    swap.append((i, cand))
                break
        if not swap:
            return r
        out = r.copy()
        faces = out.faces()
        ext.delete_faces(out, [faces[i] for i, _ in swap])
        for _, cand in swap:
            ext.append_brep(out, cand)
        open_before, solid = _open_edge_count(r), bool(r.is_manifold_solid())
        for t in (tol, 10 * tol):
            st = sm.stitch_advanced(out.copy(), t, squeeze_small_edges=False, making_manifold_solid=False)[0]
            if st.face_count() == r.face_count() and _open_edge_count(st) <= open_before \
                    and (bool(st.is_manifold_solid()) or not solid):
                try:
                    sm.unify_normals(st, max(st.faces(), key=lambda fc: fc.area()), 100)
                except Exception:
                    pass
                return st
    except Exception:
        pass
    return r


def _input_test(before):
    """on_input(face): face is one of `before`'s faces (maybe trimmed or
    merged), not a new fillet face."""
    np = _np()
    orig = {_surface_key(f.surface()) for f in before.faces()}
    lo, hi = (np.array(v) for v in before.bounding_box())
    tol = 1e-6 * (float(np.linalg.norm(hi - lo)) or 1.0)

    def on_input(f):
        if _surface_key(f.surface()) in orig:
            return True
        try:
            p = f.interior_point()
        except Exception:
            return False
        p = (p.x, p.y, p.z) if hasattr(p, "x") else tuple(p)
        return dist_to_brep(p, before) < tol
    return on_input


def _periodic_basis(n, p, ts, der=0):
    """Uniform periodic B-spline basis (n functions, n spans on [0, 1))."""
    import classa
    np = _np()
    B = classa.basis(np.arange(-p, n + p + 1) / n, p, np.asarray(ts, float) % 1.0, der)
    out = np.zeros((B.shape[0], n))
    for j in range(B.shape[1]):
        out[:, j % n] += B[:, j]
    return out


RING_DEGREE = 5          # along a rebuilt ring: curvature ripple ~1e-5 with 12-16 spans (cubic: ~1e-2)


def _ring_surface(face, is_fillet, kind, radius, size):
    """One smooth closed surface for a kernel fillet face that runs all round a
    closed edge, or None. Sections are exact (rolling ball; the Blend profile
    with the kernel's own fullness), at even arc length all round. When they
    are all one section turned about an axis, the ring is that section's exact
    surface of revolution (_revolved_ring); otherwise they are fitted along the
    ring with a uniform periodic degree-5 B-spline (C4 across the seam, evenly
    spaced CVs), the fewest spans within 1e-6 of the size. Returns (surface,
    description for the report) or None."""
    import classa
    np = _np()
    lay = classa.face_layout(face, sm, is_fillet)
    if isinstance(lay, str) or not lay.get("closed"):
        return None
    # a ring has no ends: its seam is just one of its sections (classa turns
    # the sections near an end into the end's plane, over 30% of the length
    # each way -- on a ring that tilts a third of them, by up to ~7 deg)
    lay["T_end"] = (None, None)
    d = ext.surface_cvs(lay["surf"])
    m = d["degree_u"] if lay["across"] == "u" else d["degree_v"]
    kcv = [d["count_v"], d["count_u"]] if lay["across"] == "u" else [d["count_u"], d["count_v"]]   # along x across
    blend_k = None
    if kind == "BLEND":
        if not (3 <= m <= 9 and m % 2 == 1):
            return None
        ks = classa.blend_fullness(lay, sm, m)
        if not ks:
            return None
        blend_k = float(np.median(ks))
    secs, _ = classa._sample_along(lay, kind, m, sm, radius, n=385, blend_k=blend_k)
    X = (secs[..., :3] / secs[..., 3:4]).mean(axis=1)
    gaps = np.linalg.norm(np.diff(X, axis=0), axis=1)
    h = float(np.median(gaps))
    # the kernel's ring face may stop a little short of its start or run a
    # little past it: drop the last sections that lie at (or past) the first
    t0 = (X[1] - X[0]) / (np.linalg.norm(X[1] - X[0]) or 1.0)
    keep = len(X)
    while keep > len(X) // 2 and (X[0] - X[keep - 1]) @ t0 < 0.5 * h:
        keep -= 1
    secs, X = secs[:keep], X[:keep]
    seg = np.r_[np.linalg.norm(np.diff(X, axis=0), axis=1), np.linalg.norm(X[0] - X[-1])]
    if seg.max() > 3.0 * h:
        return None                               # not one even loop of sections
    nrows = secs.shape[1]
    rev = _revolved_ring(secs, 1e-6 * size)
    if rev is not None:
        return rev, {"ring": "revolution", "round": 9, "kernel_cvs": kcv}
    s = np.r_[0.0, np.cumsum(seg[:-1])] / seg.sum()
    Q = secs.reshape(keep, -1)
    us = np.linspace(0.0, 1.0, 9)
    Xs = classa.eval_sections(secs, us)
    p, tol = RING_DEGREE, 1e-6 * size
    fit = None
    for n in (8, 12, 16, 24, 32, 48, 64):
        if 5 * n > keep:
            break
        C = np.linalg.lstsq(_periodic_basis(n, p, s), Q, rcond=None)[0]
        F = (_periodic_basis(n, p, s) @ C).reshape(secs.shape)
        if (F[..., 3] <= 0).any():
            continue
        if float(np.linalg.norm(classa.eval_sections(F, us) - Xs, axis=2).max()) <= tol:
            fit = (n, C)
            break
    if fit is None:
        return None
    n, C = fit
    # the same spline on a clamped knot vector (a periodic spline restricted to
    # [0, 1] lies in that space: the change of basis is exact)
    Uc = [0.0] * (p + 1) + [k / n for k in range(1, n)] + [1.0] * (p + 1)
    ss = np.linspace(0.0, 1.0, 20 * n + 1)
    Bc = classa.basis(Uc, p, ss)
    P = (np.linalg.lstsq(Bc, _periodic_basis(n, p, ss), rcond=None)[0] @ C).reshape(n + p, nrows, 4)
    pts, wts = [], []
    for i in range(nrows):                        # u = across (Bezier), v = along the ring
        for j in range(n + p):
            hj = P[j, i]
            pts.append(tuple(float(x) for x in hj[:3] / hj[3]))
            wts.append(float(hj[3]))
    rational = any(abs(w - 1.0) > 1e-12 for w in wts)
    return (ext.surface_from_cvs(nrows - 1, p, pts, [0.0] * nrows + [1.0] * nrows, Uc, wts if rational else None),
            {"ring": "periodic", "round": n + p, "degree": p, "kernel_cvs": kcv})


def _revolved_ring(secs, tol):
    """When every section is the same one turned about an axis (the rim of a
    cylinder, cone or any revolved part), the exact surface of revolution of
    that section: rational, 9 CVs round (the circle's own), its seam where
    the kernel's was. None when the sections differ by more than tol."""
    np = _np()
    X = secs[..., :3] / secs[..., 3:4]                    # (N, rows, 3)
    R0 = X[:, 0]
    m0 = R0.mean(axis=0)
    _, _, vt = np.linalg.svd(R0 - m0)
    a = vt[2]
    if np.abs((R0 - m0) @ a).max() > tol:
        return None                                       # the rail is not planar
    e1, e2 = vt[0], vt[1]
    if np.cross(a, R0[0] - m0) @ (R0[1] - R0[0]) < 0:
        a = -a                                            # turn the way the sections run (as the kernel face)
    xy = np.c_[(R0 - m0) @ e1, (R0 - m0) @ e2]
    A = np.c_[2.0 * xy, np.ones(len(xy))]
    cx, cy, _ = np.linalg.lstsq(A, (xy ** 2).sum(axis=1), rcond=None)[0]
    c = m0 + cx * e1 + cy * e2
    D = X - c
    z = D @ a                                              # (N, rows)
    rad = D - z[..., None] * a
    r = np.linalg.norm(rad, axis=2)
    if r.min() < tol:
        return None                                       # a row on the axis
    x0 = rad[:, :1] / r[:, :1, None]                       # each section's own radial direction
    y0 = np.cross(a, x0)
    phi = np.arctan2((rad * y0).sum(axis=2), (rad * x0).sum(axis=2))
    w = secs[..., 3]
    if (np.ptp(r, axis=0).max() > tol or np.ptp(z, axis=0).max() > tol
            or (np.ptp(phi, axis=0) * r.max(axis=0)).max() > tol or np.ptp(w, axis=0).max() > 1e-6):
        return None
    rm, zm, pm, wm = r.mean(axis=0), z.mean(axis=0), phi.mean(axis=0), w.mean(axis=0)
    xs = x0[0, 0]
    ys = np.cross(a, xs)
    s2 = np.sqrt(0.5)
    pts, wts = [], []
    for i in range(secs.shape[1]):                         # u = across (the section), v = round
        for k in range(9):
            ang = pm[i] + k * np.pi / 4.0
            rr = rm[i] if k % 2 == 0 else rm[i] / s2
            q = c + zm[i] * a + rr * (np.cos(ang) * xs + np.sin(ang) * ys)
            pts.append(tuple(float(x) for x in q))
            wts.append(float(wm[i] * (1.0 if k % 2 == 0 else s2)))
    nr = secs.shape[1]
    return ext.surface_from_cvs(nr - 1, 2, pts, [0.0] * nr + [1.0] * nr,
                                [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0], wts)


def _smooth_rings(before, r, xsect, radius):
    """The kernel's fillet round a CLOSED edge (a cylinder's rim) is one dense
    face (150-200 CVs along it, knots of multiplicity 3: only C0 inside) whose
    section shape follows the base surfaces' parametrisation, so it differs a
    little all the way round: on a round part the curvature map shows blobs
    and the CVs bunch up unevenly, most of all at the seam. Each such face is
    rebuilt as one smooth closed surface (_ring_surface) and stitched back in.
    Keeps r (and that face) when anything misses. Returns (brep, rings): one
    description per rebuilt ring, with "at" = a point inside it (to find the
    face again after the stitch)."""
    np = _np()
    kind = {0: "CHAMFER", 1: "CIRCULAR", 2: "BLEND"}[xsect]
    lo, hi = (np.array(v) for v in r.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    on_input = _input_test(before)
    fil = [f for f in r.faces() if not on_input(f)]
    if not fil:
        return r, []
    import classa

    def is_fillet(f):
        return any(classa._same_face(f, g) for g in fil)
    new = []
    for f in fil:
        try:
            res = _ring_surface(f, is_fillet, kind, radius, size)
        except Exception:
            res = None
        if res is not None:
            surf, desc = res
            (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
            desc["at"] = [float(x) for x in surf.evaluate(0.5 * (u0 + u1), 0.5 * (v0 + v1))]
            new.append((_surface_key(f.surface()), surf, desc))
    if not new:
        return r, []
    try:
        out = r.copy()
        keys = {k for k, _, _ in new}
        ext.delete_faces(out, [f for f in out.faces() if _surface_key(f.surface()) in keys])
        for _, surf, _ in new:
            ext.append_brep(out, ext.brep_from_surface(surf))
        open0 = _open_edge_count(r)
        for tol in (1e-6 * size, 1e-5 * size):
            st = sm.stitch_advanced(out.copy(), tol, squeeze_small_edges=False, making_manifold_solid=False)[0]
            if _open_edge_count(st) <= open0 and st.face_count() == r.face_count():
                try:      # agree with the original faces (the new one may face either way)
                    base = [f for f in st.faces() if on_input(f)] or st.faces()
                    sm.unify_normals(st, max(base, key=lambda f: f.area()), 100)
                except Exception:
                    pass
                return st, [desc for _, _, desc in new]
    except Exception:
        pass
    return r, []


def _face_normal(face, p):
    """Unit outward normal of face at (near) the point p."""
    np = _np()
    _, uvs, _ = sm.surface_closest_point(face.surface(), tuple(float(x) for x in p))
    uv = uvs[0]
    n = np.array(face.outward_normal_at_uv(*((uv.x, uv.y) if hasattr(uv, "x") else uv)), float)
    return n / (np.linalg.norm(n) or 1.0)


def _surface_key(surf):  # noqa: E302
    d = ext.surface_cvs(surf)
    return (d["degree_u"], d["degree_v"], d["count_u"], d["count_v"],
            tuple(round(c, 9) for p in d["points"] for c in p))


def _class_a(before, r, job, xsect, radius, skip=()):
    """Rebuild the kernel's dense fillet faces as single-span patches
    (kernel/classa.py). Keeps the kernel result if anything goes wrong.
    Faces through a point of `skip` (rims already rebuilt as one smooth
    ring, _smooth_rings) are left as they are."""
    import classa
    np = _np()
    orig = {_surface_key(f.surface()) for f in before.faces()}
    lo, hi = (np.array(v) for v in before.bounding_box())
    tol = 1e-6 * (float(np.linalg.norm(hi - lo)) or 1.0)

    def on_input(f):          # a (trimmed / merged) input face, not a fillet
        if _surface_key(f.surface()) in orig:
            return True
        try:
            p = f.interior_point()
        except Exception:
            return False
        p = (p.x, p.y, p.z) if hasattr(p, "x") else tuple(p)
        return dist_to_brep(p, before) < tol
    def is_ring(f):
        return any(f.closest_point(tuple(q))[1] < tol for q in skip)
    fil = [i for i, f in enumerate(r.faces()) if not on_input(f) and not is_ring(f)]
    if not fil and skip:
        return r, {"faces": [], "patches": 0}
    kind = {0: "CHAMFER", 1: "CIRCULAR", 2: "BLEND"}[xsect]
    keep = r.copy()
    try:
        out, rep = classa.rebuild_body(r, sm, fil, kind=kind, cont=int(job.get("ca_continuity", 2)),
                                       tol=float(job.get("ca_tolerance", 1e-4)),
                                       ang_tol=float(job.get("ca_angle", 0.1)),
                                       max_degree=int(job.get("ca_degree", 7)),
                                       max_spans=int(job.get("ca_spans", 8)), radius=radius)
    except Exception as exc:
        return keep, {"error": f"Class-A rebuild failed ({exc}); kept the kernel fillet"}
    if isinstance(rep, list):          # nothing to rebuild
        return keep, {"faces": rep, "patches": 0, "note": "the fillet faces are already single-span"}
    # the merge imprints zero-area slivers where a patch lies tangent to a
    # plane: drop them and stitch again
    lo, hi = (np.array(v) for v in out.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    sl = [f for f in out.faces() if f.area() < 1e-10 * size * size]
    if sl:
        ext.delete_faces(out, sl)
        out = sm.stitch_advanced(out, rep["stitch_tol"], squeeze_small_edges=False,
                                 making_manifold_solid=False)[0]
    lamina = sum(1 for e in out.edges() if len(e.faces()) == 1)
    nonman = sum(1 for e in out.edges() if len(e.faces()) > 2)
    if lamina > rep["lamina_before"] or nonman:
        return keep, {"error": f"the patches did not stitch back ({lamina - rep['lamina_before']} open, "
                               f"{nonman} non-manifold edges); kept the kernel fillet", "faces": rep["faces"]}
    try:
        if bool(keep.is_manifold_solid()) and not bool(out.is_manifold_solid()):
            h = sm.heal_brep(out)
            if h is not None and bool(h.is_manifold_solid()):
                out = h
    except Exception:
        pass
    rep["lamina_after"] = lamina
    return out, rep


def job_retarget(job):
    """Trimmed-surface CV edit: the target's (single) face gets a new B-spline
    surface built from `cvs`; trims stay fixed in UV space. Local space."""
    m = job["meta"]
    b = load(job["breps"]["target"])
    mk = lambda: ext.surface_from_cvs(m["degree_u"], m["degree_v"], [tuple(p) for p in job["cvs"]],
                                      m["knots_u"], m["knots_v"], m["weights"] or None)
    # all faces share one underlying surface (checked in Blender): each gets its own copy
    ext.replace_face_surfaces_keep_uv(b, [mk() for _ in range(b.face_count())])
    save(b, job["out"]["result"])
    # the edited surface without trims (for Untrim)
    plain = ext.brep_from_surface(ext.surface_from_cvs(m["degree_u"], m["degree_v"], [tuple(p) for p in job["cvs"]],
                                                       m["knots_u"], m["knots_v"], m["weights"] or None))
    save(plain, job["out"]["untrimmed"])
    return {"faces": b.face_count()}


def _reverse_v(s):
    """The same surface with v reversed (its normal turns over)."""
    np = _np()
    d = ext.surface_cvs(s)
    nu, nv = d["count_u"], d["count_v"]
    P = np.array(d["points"], float).reshape(nu, nv, 3)[:, ::-1]
    w = np.array(d["weights"], float).reshape(nu, nv)[:, ::-1] if d.get("weights") else None
    K = list(d["knots_v"])
    Kv = [K[0] + K[-1] - k for k in reversed(K)]
    pts = [tuple(map(float, P[i, j])) for i in range(nu) for j in range(nv)]
    return ext.surface_from_cvs(d["degree_u"], d["degree_v"], pts, list(d["knots_u"]), Kv,
                                [float(x) for x in w.reshape(-1)] if w is not None else None)


def _uvxy(uv):
    return (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])


def _normal_near(b, p):
    """Outward normal of b's face nearest the point p."""
    np = _np()
    f = min(b.faces(), key=lambda f: f.closest_point(tuple(p))[1])
    _, uvs, _ = sm.surface_closest_point(f.surface(), tuple(p))
    return np.array(f.outward_normal_at_uv(*_uvxy(uvs[0])), float)


def flip_brep(b):
    """Every face of an open sheet / shell turned over, trims kept. The kernel
    only turns faces over to agree with a neighbour (unify_normals), so a
    small helper strip is stitched to one open edge facing the other way,
    the faces are unified to the strip, and the strip is removed again."""
    np = _np()
    try:
        solid = bool(b.is_manifold_solid())
    except Exception:
        solid = False
    if solid:
        raise RuntimeError("a closed solid always faces outward -- nothing to flip")
    n0 = b.face_count()
    lo, hi = (np.array(v, float) for v in b.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    bnd = [e for e in b.edges() if len(e.faces()) == 1]
    if not bnd:
        raise RuntimeError("no open edge (the faces are not joined into one sheet?)")
    e = max(bnd, key=lambda e: e.length())
    t0, t1 = e.parameter_range()
    P = np.array([e.point_at(t) for t in np.linspace(t0, t1, 41)], float)
    m = P[20]
    nrm = _normal_near(b, m)
    nrm /= np.linalg.norm(nrm) or 1.0
    h = 0.05 * (e.length() or size)
    # the strip leaves the edge at 45 deg between the normal and the outward
    # direction: straight along the normal its side edges could lie on a
    # neighbouring face's edge (a wall at a corner) and get stitched to it
    T = P[21] - P[19]
    out = np.cross(nrm, T)
    out /= np.linalg.norm(out) or 1.0
    f0 = e.faces()[0]
    if f0.closest_point(tuple(m + 1e-3 * h * out))[1] < f0.closest_point(tuple(m - 1e-3 * h * out))[1]:
        out = -out
    w = (nrm + out) / np.sqrt(2.0)
    c1 = sm.create_curve_interp_points([tuple(map(float, p)) for p in P])
    c2 = sm.create_curve_interp_points([tuple(map(float, p + h * w)) for p in P])
    strip = sm.create_ruled_surface(c1, c2)
    probe = m + 0.5 * h * w
    # the interpolated edge strays from the real one by ~(edge / 40)^4 x curvature:
    # stitch tight first, a little looser if that does not connect
    for tol in (1e-6 * size, 1e-5 * size, 1e-4 * size):
        for helper in (strip, _reverse_v(strip)):
            merged = sm.merge_breps([b.copy(), ext.brep_from_surface(helper)], operation=sm.BooleanOp.MERGE)
            st = sm.stitch_advanced(merged, tol, squeeze_small_edges=False, making_manifold_solid=False)[0]
            hf = min(st.faces(), key=lambda f: f.closest_point(tuple(probe))[1])
            try:
                sm.unify_normals(st, hf, 100)
            except Exception:
                continue
            ext.delete_faces(st, [hf])
            if st.face_count() != n0:
                continue
            if float(_normal_near(st, m) @ nrm) < 0:
                return st
    raise RuntimeError("the kernel could not turn these faces over")


def job_flip(job):
    """Flip Normal of a trimmed face / joined sheet (local space)."""
    b = load(job["breps"]["target"])
    save(flip_brep(b), job["out"]["result"])
    return {"faces": b.face_count()}


def _to_origin(b):
    """Move b (in place) so its box is centred on the origin; returns the
    move back. The kernel's offsets fit their surfaces finer the farther the
    part is from the origin: a lofted dome's offset had 20 x 15 CVs at the
    origin, 45 x 156 six units out (and Thicken took 57 s instead of 0.2)."""
    lo, hi = b.bounding_box()
    c = tuple(0.5 * (float(a) + float(z)) for a, z in zip(lo, hi))
    if max(abs(v) for v in c) > 0.0:
        sm.translate(b, tuple(-v for v in c))
    return c


def _from_origin(r, c):
    return sm.translate(r, c) if r is not None and max(abs(v) for v in c) > 0.0 else r


def job_offset(job):
    """Offset every face of a (joined) sheet or solid by `distance` along its
    normals; corners between faces extended until they meet. World space in,
    local space of the target out (the distance is a world distance)."""
    b = world_brep(job, "target")
    d = float(job["distance"])
    if abs(d) <= 0:
        raise RuntimeError("the distance is zero")
    c = _to_origin(b)
    r = _from_origin(sm.offset_brep(b, d, extended_offset=True, self_intersection=True), c)
    if r is None or r.face_count() == 0:
        raise RuntimeError("no offset (the distance may be bigger than a concave radius)")
    save(to_local(r, job.get("xforms", {}).get("target")), job["out"]["result"])
    return {"faces": r.face_count()}


def job_thicken(job):
    """A closed solid from a (joined) sheet: the sheet, its offset by
    `distance` and the side walls between them. symmetric: half each way."""
    b = world_brep(job, "target")
    d = float(job["distance"])
    if abs(d) <= 0:
        raise RuntimeError("the thickness is zero")
    try:
        solid = bool(b.is_manifold_solid())
    except Exception:
        solid = False
    if solid:
        raise RuntimeError("this is already a closed solid -- Thicken works on open surfaces (use Offset)")
    c = _to_origin(b)
    if job.get("symmetric"):
        b = sm.offset_brep(b, -0.5 * d, extended_offset=True, self_intersection=True)
    r = _from_origin(sm.shell_brep(b, d, extended_offset=True, self_intersection=True, create_solid=True), c)
    if r is None or r.face_count() == 0:
        raise RuntimeError("no solid (the thickness may be bigger than a concave radius)")
    volume = 0.0
    try:
        if r.is_manifold_solid():
            volume = abs(r.volume(1e-6))
    except Exception:
        pass
    save(to_local(r, job.get("xforms", {}).get("target")), job["out"]["result"])
    return {"faces": r.face_count(), "volume": volume}


def _flat_quad(loops, tol):
    """The four corners of a face outline that is one convex, flat loop of four
    straight sides (no holes, no curves), else None. Its bilinear patch is
    then exactly the face -- flat, its edges the outline -- with no trim."""
    np = _np()
    if len(loops) != 1 or len(loops[0]) != 4 or any(isinstance(q, dict) for q in loops[0]):
        return None
    Q = np.array(loops[0], float)
    n = np.cross(Q[2] - Q[0], Q[3] - Q[1])
    if np.linalg.norm(n) <= 1e-12:
        return None
    n /= np.linalg.norm(n)
    if np.abs((Q - Q.mean(0)) @ n).max() > tol:            # not flat: a bilinear patch would be warped
        return None
    turns = [np.cross(Q[(k + 1) % 4] - Q[k], Q[(k + 2) % 4] - Q[(k + 1) % 4]) @ n for k in range(4)]
    side = max(np.linalg.norm(Q[(k + 1) % 4] - Q[k]) for k in range(4))
    if min(turns) <= 1e-6 * side * side:                   # concave or a straight "corner": it would fold
        return None
    return [tuple(map(float, q)) for q in Q]


def _open_edges(b):
    return sum(1 for e in b.edges() if len(e.faces()) == 1)


def _tight_plane(f, pad=0.005):
    """The surface of planar face f cut down to its trim, or None: a plane
    made as a bilinear patch on a parallelogram (the kernel's planar faces,
    our quads) maps uv affinely, so the same patch over the sub-domain round
    the trim's uv box (grown by pad of it) is the same plane -- the trims keep
    their uv curves -- just no bigger than it needs to be. None if the
    surface is not such a plane or is no more than 0.1% too big already."""
    np = _np()
    d = ext.surface_cvs(f.surface())
    if d.get("degree_u") != 1 or d.get("degree_v") != 1 or d.get("count_u") != 2 or d.get("count_v") != 2 \
            or d.get("weights"):
        return None
    (a0, a1), (c0, c1) = (d["knots_u"][0], d["knots_u"][-1]), (d["knots_v"][0], d["knots_v"][-1])
    if a1 - a0 <= 0 or c1 - c0 <= 0:
        return None
    C = np.array(d["points"], float)                      # u-major: (a0,c0) (a0,c1) (a1,c0) (a1,c1)
    o, du, dv = C[0], (C[2] - C[0]) / (a1 - a0), (C[1] - C[0]) / (c1 - c0)
    if np.linalg.norm(C[3] - C[2] - (C[1] - C[0])) > 1e-9 * (np.linalg.norm(C[2] - C[0]) + np.linalg.norm(C[1] - C[0])):
        return None                                       # not a parallelogram: uv is not affine on it
    Q = []
    for e in f.edges():
        t0, t1 = e.parameter_range()
        Q += [e.point_at(t) for t in np.linspace(t0, t1, 33)]
    if not Q:
        return None
    UV = np.linalg.lstsq(np.stack([du, dv], 1), (np.array(Q, float) - o).T, rcond=None)[0].T + (a0, c0)
    (u0, v0), (u1, v1) = UV.min(0), UV.max(0)
    g = pad * max(u1 - u0, v1 - v0)
    u0, v0, u1, v1 = max(u0 - g, a0), max(v0 - g, c0), min(u1 + g, a1), min(v1 + g, c1)
    if (u1 - u0) * (v1 - v0) > 0.999 * (a1 - a0) * (c1 - c0):
        return None
    S = lambda u, v: tuple(map(float, o + (u - a0) * du + (v - c0) * dv))
    return ext.surface_from_cvs(1, 1, [S(u0, v0), S(u0, v1), S(u1, v0), S(u1, v1)],
                                [u0, u0, u1, u1], [v0, v0, v1, v1])


def _shrink_plane(b):
    """A one-face planar BREP (create_planar_faces) with its plane cut down to its trim."""
    srf = _tight_plane(b.faces()[0])
    if srf is None:
        return b
    out = b.copy()
    try:
        ext.replace_face_surfaces_keep_uv(out, [srf])
    except Exception:
        return b
    return out if abs(out.area(1e-6) - b.area(1e-6)) <= 1e-6 * max(b.area(1e-6), 1e-12) else b


def _shrink_planes(b, pick=None):
    """Body b with the planes of its faces (those pick(face) accepts; all if
    None) cut down to their trims -- after a fillet the faces beside it keep
    the sharp face's plane. b itself if nothing changes or the result is not
    the same body (open edges, solidity or area changed)."""
    faces = b.faces()
    srfs, n = [], 0
    for f in faces:
        t = _tight_plane(f) if pick is None or pick(f) else None
        n += t is not None
        srfs.append(t if t is not None else f.surface())
    if not n:
        return b
    out = b.copy()
    try:
        ext.replace_face_surfaces_keep_uv(out, srfs)
        same = _open_edges(out) == _open_edges(b) and bool(out.is_manifold_solid()) == bool(b.is_manifold_solid()) \
            and abs(out.area(1e-6) - b.area(1e-6)) <= 1e-6 * max(b.area(1e-6), 1e-12)
    except Exception:
        return b
    return out if same else b


def job_mesh_to_brep(job):
    """Mesh to BREP (mesh_brep.py): planar faces from their outlines (outer
    loop, then holes) sewn into a body, cylinders added and holes cut
    ({origin, axis, r, h, cut}), then each fillet group ({r, mids}, the sharp
    edges' midpoints) rounded, the largest radius first. Built in world space;
    "base" (before the fillets, if asked for) and "result" written in job['xform']'s space;
    the volume only with job["volume"]."""
    np = _np()
    tol = float(job.get("tol", 1e-6))
    body = None
    parts, tight = [], {}
    at = lambda q: tuple(q["p"]) if isinstance(q, dict) else tuple(q)
    for loops in job["faces"]:
        quad = _flat_quad(loops, tol)
        if quad is not None:                  # four straight sides: the bilinear patch on its corners, untrimmed
            q0, q1, q2, q3 = quad
            parts.append(ext.brep_from_surface(ext.surface_from_cvs(1, 1, [q0, q3, q1, q2],
                                                                    [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0])))
            continue
        cs = []
        for P in loops:                       # a point: a line to the next; {p, cvs, knots, degree}: a curve to it
            for k, q in enumerate(P):
                if isinstance(q, dict):
                    cs.append(ext.curve_from_cvs(int(q["degree"]), [tuple(c) for c in q["cvs"]], list(q["knots"]),
                                                 list(q["weights"]) if q.get("weights") else None))
                else:
                    cs.append(sm.create_line_segment(at(q), at(P[(k + 1) % len(P)])))
        parts.append(sm.create_planar_faces(cs))
        tight[len(parts) - 1] = _shrink_plane(parts[-1])   # its plane cut down to the outline
    for g in job.get("patches", []):          # a free-form patch: a B-spline (u-major net; bicubic unless said)
        d = int(g.get("degree", 3))
        parts.append(ext.brep_from_surface(ext.surface_from_cvs(d, d, [tuple(c) for c in g["net"]],
                                                                list(g["ku"]), list(g["kv"]))))
    for w in job.get("sweeps", []):           # a boxy bevel / cut cylinder: ruled between its curves at its two ends
        pts = [tuple(c) for a, b in zip(w["cvs"], w["cvs2"]) for c in (a, b)]
        wts = [float(x) for x in w["weights"] for _ in (0, 1)] if w.get("weights") else None
        parts.append(ext.brep_from_surface(ext.surface_from_cvs(int(w["degree"]), 1, pts, list(w["knots"]),
                                                                [0.0, 0.0, 1.0, 1.0], wts)))
    def sew(ps):
        b = ps[0].copy()
        for q in ps[1:]:                      # faces side by side (no intersecting: they only share edges)
            ext.append_brep(b, q)
        return sm.stitch_advanced(b, tol, squeeze_small_edges=False, making_manifold_solid=True)[0]
    if parts:
        # the planes cut down to their outlines; their trim edges move by ~1e-6 doing so, which can
        # keep a fine profile (a 16-segment boxy bevel) from stitching: then the kernel's own planes
        body = sew([tight.get(k, q) for k, q in enumerate(parts)])
        if any(tight[k] is not parts[k] for k in tight) and not body.is_manifold_solid():
            plain = sew(parts)
            if plain.is_manifold_solid() or _open_edges(plain) < _open_edges(body):
                body = plain
        try:
            sm.unify_normals(body, max(body.faces(), key=lambda f: f.area()), 100)
        except Exception:
            pass
        if body.is_manifold_solid():
            h = sm.heal_brep(body)
            body = h if h is not None and h.face_count() else body
    for c in job.get("cylinders", []):
        cyl = sm.create_cylinder((0.0, 0.0, 0.0), float(c["r"]), float(c["h"]))
        ax = np.array(c["axis"], float)
        if ax[2] < 1 - 1e-12:
            k = np.cross((0.0, 0.0, 1.0), ax)
            k = k / np.linalg.norm(k) if np.linalg.norm(k) > 1e-12 else np.array([1.0, 0.0, 0.0])
            cyl = sm.rotate(cyl, (0.0, 0.0, 0.0), tuple(map(float, k)), float(np.degrees(np.arccos(np.clip(ax[2], -1, 1)))))
        cyl = sm.translate(cyl, tuple(map(float, c["origin"])))
        if body is None:
            body = cyl
            continue
        op = sm.BooleanOp.DIFFERENCE if c["cut"] else sm.BooleanOp.UNION
        out = sm.boolean(body, cyl, operation=op)
        if out is None or out.face_count() == 0:
            raise RuntimeError("a hole or cylinder could not be combined with the body")
        body = out
    if body is None:
        raise RuntimeError("nothing to build")
    solid = bool(body.is_manifold_solid())
    base = body.copy()
    done, failed = [], []
    for g in job.get("fillets", []):
        b = body.copy()
        edges = b.edges()
        mids = np.array([_edge_mid(e) if _edge_mid(e) is not None else [np.inf] * 3 for e in edges], float)
        sel = []
        for m in g["mids"]:
            i = int(np.argmin(np.linalg.norm(mids - np.array(m), axis=1)))
            if len(edges[i].faces()) == 2 and edges[i] not in sel:
                sel.append(edges[i])
        try:
            r = sm.fillet_edges(b, sel, radius=float(g["r"])) if sel else None
        except Exception:
            r = None
        if r is None or r.face_count() == 0:
            failed.append(g["r"])
            continue
        body = r
        done.append(g)
    if done or job.get("cylinders"):         # planes a fillet or a hole cut into: down to what is left of them
        body = _shrink_planes(body)
    xf = job.get("xform")
    if "base" in job["out"]:                  # (SubD to BREP has no fillets: it asks for the result only)
        save(to_local(base, xf), job["out"]["base"])
    save(to_local(body, xf), job["out"]["result"])
    volume = None                             # only on request: it took 0.1-0.25 s of a SubD body's 0.3 s here
    if job.get("volume") and solid:
        try:
            volume = abs(float(body.volume(1e-6)))
        except Exception:
            volume = 0.0
    return {"solid": solid and bool(body.is_manifold_solid()), "faces": body.face_count(), "volume": volume,
            "filleted": done, "failed": failed}


JOBS = {"retarget": job_retarget, "edge_fillet": job_edge_fillet, "trim": job_trim, "untrim_region": job_untrim_region, "untrim_preview": job_untrim_preview, "fillet": job_fillet, "split": job_split, "join": job_join, "unjoin": job_unjoin, "boolean": job_boolean, "push_pull": job_push_pull, "rotate_face": job_rotate_face, "move_face": job_move_face, "replace_face": job_replace_face, "extrude_face": job_extrude_face,
        "flip": job_flip, "cap_solid": job_cap_solid, "boundary_fill": job_boundary_fill, "offset": job_offset, "thicken": job_thicken,
        "mesh_to_brep": job_mesh_to_brep}


RESULT_MARK = "@@nproto-result@@ "   # (serve mode: the line that carries a job's result)


def _run(path):
    with open(path) as f:
        job = json.load(f)
    try:
        return {"ok": True, "result": JOBS[job["op"]](job)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()}


def serve():
    """Stay up: one job file path per line on stdin, each answered by a RESULT_MARK line on stdout (jobs.py
    keeps one of these running so a job does not pay for starting Python and loading the kernel)."""
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        res = _run(path)
        sys.stdout.write("\n" + RESULT_MARK + json.dumps(res) + "\n")
        sys.stdout.flush()


def main():
    if sys.argv[1] == "--serve":
        serve()
        return
    res = _run(sys.argv[1])
    sys.stdout.write("\n" + json.dumps(res) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
