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
        np = _np()
        lo, hi = (np.array(v) for v in b.bounding_box())
        P = np.array(_poly(curve(self.data), 64))
        reach = float(np.linalg.norm(hi - lo)) + float(np.max(np.linalg.norm(P - 0.5 * (lo + hi), axis=1)))
        L = 4.0 * (reach or 1.0)
        d = -self.dir if self.reach == "BACK" else self.dir
        start = -0.5 * L if self.reach == "BOTH" else 0.0
        for piece in corner_pieces(self.data):
            c = sm.translate(curve(piece), tuple(start * d)) if start else curve(piece)
            sheet = ext.brep_from_surface(sm.create_extrude_surface(c, tuple(L * d)))
            b = sm.boolean_with_options(b, sheet, operation=sm.BooleanOp.UNION, imprinting=True,
                                        keep_other_brep=True)
            if b is None:
                return None
        return b

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


def heal_cut_seams(b, original):
    """Remove the cut edges that have kept surface on both sides: a trim
    imprints every cutter, and where a region on each side is kept (the far
    wall a projected circle also crossed) the imprint stayed as a seam -- two
    faces, or one face whose loop runs out and back along a slit. Each such
    set of faces on one surface is trimmed again to the outline of their
    union, exactly (the edges' own curves). Original edges (a revolve's seam)
    stay. Returns b unchanged when anything does not check out."""
    np = _np()
    lo, hi = (np.array(v) for v in original.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    tol = 1e-5 * size
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
        if len(us) != 2 or us[0][1] == us[1][1] or skeys[us[0][0]] is None \
                or skeys[us[0][0]] != skeys[us[1][0]]:
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
                curves = []
                for lp in loops_try:
                    for j in lp:
                        e, fw = segs[j][0], segs[j][1]
                        d = ext.curve_cvs(e.curve())
                        U, Pc, Wt = list(d["knots"]), list(d["points"]), d.get("weights")
                        if not fw:
                            a0, a1 = U[0], U[-1]
                            U = [a0 + a1 - k for k in reversed(U)]
                            Pc, Wt = Pc[::-1], (list(Wt)[::-1] if Wt else None)
                        curves.append(ext.curve_from_cvs(d["degree"], Pc, U, list(Wt) if Wt else None))
                sd = ext.surface_cvs(faces[members[0]].surface())
                surf = ext.surface_from_cvs(sd["degree_u"], sd["degree_v"], list(sd["points"]), list(sd["knots_u"]),
                                            list(sd["knots_v"]), list(sd["weights"]) if sd.get("weights") else None) \
                    if "knots_u" in sd else faces[members[0]].surface()
                try:
                    if tight and hasattr(ext, "trim_surface_with_3d_curves_tol"):
                        cand = ext.trim_surface_with_3d_curves_tol(surf, [len(lp) for lp in loops_try], curves, tol,
                                                                   0.01)
                    else:
                        cand = sm.trim_surface_with_3d_curves(surf, [len(lp) for lp in loops_try], curves)
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
    before_open = _open_edge_count(b) - 2 * len(seams)
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
    G3 blend between the rolling ball's rails): built from exact sections
    (secfillet.py). Either way the bases are trimmed back to its rails."""
    A = world_brep(job, "a_base" if "a_base" in job["breps"] else "a")
    B = world_brep(job, "b_base" if "b_base" in job["breps"] else "b")
    chord = job.get("chord")
    blend = int(job.get("blend") or 0)
    if chord or blend:
        # our own sections (secfillet.py): the kernel's fillet has one ball
        # radius, a chord needs one per section; a blend is built between the
        # rails directly (no circular fillet, trim and loft between its edges)
        import secfillet
        sa = 1.0 if float(job["radius_a"]) >= 0 else -1.0
        sb = 1.0 if float(job["radius_b"]) >= 0 else -1.0
        shape = {1: "G1", 2: "G2", 3: "G3"}.get(blend) or ("LINEAR" if int(job["xsect"]) == 0 else "CIRCULAR")
        f, info = secfillet.build(sm, A, B, sa, sb, abs(float(job["radius_a"])), abs(float(job["radius_b"])),
                                  chord=float(chord) if chord else None, shape=shape,
                                  tolerance=float(job.get("tolerance", 1e-4)))
        f = _orient_fillet(f, (A, B))
        res = _fillet_finish(job, f, A, B)
        res["radius"] = info["radius"][1]
        res["radius_range"] = info["radius"]
        res["chord"] = info["chord"]
        res["sections"] = info["sections"]
        if info.get("note"):
            res["note"] = info["note"]
        return res
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
        for key, base in (("a", A), ("b", B)):
            rail[key] = min(edges, key=lambda e: dist_to_brep(mid(e), base))
        rmax = max(abs(float(job["radius_a"])), abs(float(job["radius_b"])))
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
                sheet = _rail_sheet(rail[key], base, max(0.25 * rmax, 1e-4))
                import os, tempfile
                fd, sp = tempfile.mkstemp(suffix=".sm")
                os.close(fd)
                save(sheet, sp)
                fd, op_ = tempfile.mkstemp(suffix=".sm")
                os.close(fd)
                sub = {"breps": {"target": job["breps"][key], "sheet": sp},
                       "xforms": {"target": job.get("xforms", {}).get(key)},
                       "cutter_breps": ["sheet"], "ref_point": list(ref), "action": "DELETE", "scope": "PICKED",
                       "direction": [0.0, 0.0, 1.0], "out": {"DELETE": op_}}
                job_trim(sub)
                import shutil
                shutil.move(op_, job["out"][key])
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


def job_split(job):
    """Split the target along all cutters and write EVERY region as its own
    brep (out_region<i>.sm in out_dir). Cutting surfaces that miss the target
    are ignored; nothing is removed."""
    cutters = build_cutters(job, strict=False)
    if not cutters:
        raise RuntimeError("nothing crosses this surface")
    if "ref_point" not in job:
        lo, hi = world_brep(job, "target").bounding_box()
        job["ref_point"] = [0.5 * (lo[i] + hi[i]) for i in range(3)]
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
    merged = breps[0] if len(breps) == 1 else sm.merge_breps(breps, operation=sm.BooleanOp.MERGE)
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


def _section_edge_fillet(b, sel, chord, shape):
    """(body, info) with the picked edges filleted from exact sections at a
    constant chord, or None when the kernel has to do it (see _section_groups)."""
    np = _np()
    import os
    import tempfile
    import secfillet
    lo, hi = (np.array(v) for v in b.bounding_box())
    size = float(np.linalg.norm(hi - lo)) or 1.0
    groups = _section_groups(b, sel, 1e-5 * size)
    if groups is None:
        return None
    fillets, radii, chords, sections = [], [], [], 0
    for fp, fq, es in groups:
        # the faces' surfaces carried on past their edges: a face that ends at
        # the filleted edge has no foot for the ball there otherwise (nor a
        # twisted one at the sheet's ends, where the section leans past it)
        A = ext.brep_from_surface(_surface_extended(fp.surface(), 2.0 * chord))
        B = ext.brep_from_surface(_surface_extended(fq.surface(), 2.0 * chord))
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
    return r, {"radius": (min(radii), float(np.mean(radii)), max(radii)),
               "chord": (min(chords), float(np.mean(chords)), max(chords)), "sections": sections}


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
    r = _fix_ring_faces(b_ring, r)
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


def job_offset(job):
    """Offset every face of a (joined) sheet or solid by `distance` along its
    normals; corners between faces extended until they meet. World space in,
    local space of the target out (the distance is a world distance)."""
    b = world_brep(job, "target")
    d = float(job["distance"])
    if abs(d) <= 0:
        raise RuntimeError("the distance is zero")
    r = sm.offset_brep(b, d, extended_offset=True, self_intersection=True)
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
    if job.get("symmetric"):
        b = sm.offset_brep(b, -0.5 * d, extended_offset=True, self_intersection=True)
    r = sm.shell_brep(b, d, extended_offset=True, self_intersection=True, create_solid=True)
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


JOBS = {"retarget": job_retarget, "edge_fillet": job_edge_fillet, "trim": job_trim, "untrim_region": job_untrim_region, "untrim_preview": job_untrim_preview, "fillet": job_fillet, "split": job_split, "join": job_join, "unjoin": job_unjoin, "boolean": job_boolean,
        "flip": job_flip, "boundary_fill": job_boundary_fill, "offset": job_offset, "thicken": job_thicken,
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
