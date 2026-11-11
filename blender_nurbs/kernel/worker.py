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


class CurveCutter:
    """A cutting curve projected along a direction: splits by parallel
    projection; a point is 'inside' when it projects inside the curve outline."""

    def __init__(self, data, direction):
        np = _np()
        self.data = data
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
        self.closed = np.linalg.norm(P[0] - P[-1]) <= 1e-6 * (np.ptp(P, axis=0).max() or 1.0)

    def split(self, b, ref):
        """Exact imprint of the curve swept along the projection direction (a
        cutter sheet long enough to pass through the target). The kernel's
        parallel projection can drop pieces of loops that cross a seam."""
        np = _np()
        lo, hi = (np.array(v) for v in b.bounding_box())
        P = np.array(_poly(curve(self.data), 64))
        reach = float(np.linalg.norm(hi - lo)) + float(np.max(np.linalg.norm(P - 0.5 * (lo + hi), axis=1)))
        L = 4.0 * (reach or 1.0)
        c = sm.translate(curve(self.data), tuple(-0.5 * L * self.dir))
        sheet = ext.brep_from_surface(sm.create_extrude_surface(c, tuple(L * self.dir)))
        return sm.boolean_with_options(b, sheet, operation=sm.BooleanOp.UNION, imprinting=True,
                                       keep_other_brep=True)

    def inside(self, p):
        if not self.closed:
            return None  # an open cutter only splits; it has no inside
        np = _np()
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

    def __init__(self, job, key):
        np = _np()
        self.job, self.key = job, key
        arr = world_brep(job, key).tessellate(chord_height_tolerance=0.0005).to_numpy_arrays()
        self.P = arr["points"].astype(float)
        self.N = arr["normals"].astype(float)

    def split(self, b, ref):
        return sm.boolean_with_options(b, world_brep(self.job, self.key), operation=sm.BooleanOp.UNION,
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
        p = np.array(f.interior_point())
        sides.append(tuple(c.inside(p) for c in cutters))
    return b, faces, groups, picked, sides


def job_trim(job):
    """Alias-style click-to-trim.

    The target is split into regions along every cutter: projected curves
    (parallel projection) or cutting surfaces (exact imprint of the
    intersection). Faces joined across original edges (a revolve seam) form
    one region. With scope THROUGH every region on the same side of the
    cutters as the clicked one is affected (a circle in front of a vase cuts
    both walls); with PICKED only the clicked region is."""
    ref = tuple(job["ref_point"])
    cutters = []
    for c in job.get("curves", []):
        cutters.append(CurveCutter(c, c.get("dir") or job["direction"]))
    for key in job.get("cutter_breps", []):
        if not sm.intersect_breps(world_brep(job, "target"), world_brep(job, key))[0]:
            raise RuntimeError(f"cutter surface does not intersect the target")
        cutters.append(SurfaceCutter(job, key))
    if not cutters:
        raise RuntimeError("no cutters")

    action = job["action"]
    through = job.get("scope", "THROUGH") == "THROUGH"
    no_cut = ("the cutters do not cut this surface into regions -- "
              "check that they cross it (along the projection direction for curves)")

    stored_side = job.get("side")   # history rebuild: the side of the cutters that was kept

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
        sel = chosen(groups, picked, sides)
        info = {"regions": len(groups), "selected": len(sel), "side": list(sides[picked])}
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
        results[act] = b
    out = {"info": info}
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
    """Degree-1 x rail sheet whose middle line is EXACTLY the rail curve: rows =
    rail CVs -+ h * (base normal at the CV's Greville point), same weights."""
    np = _np()
    d = ext.curve_cvs(edge.curve())
    p, U = d["degree"], list(d["knots"])
    P = np.array(d["points"], float)
    W = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    c = edge.curve()
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
    # optional *_base breps: surfaces to compute the fillet from (untrimmed);
    # "a" / "b" are what gets trimmed
    A = world_brep(job, "a_base" if "a_base" in job["breps"] else "a")
    B = world_brep(job, "b_base" if "b_base" in job["breps"] else "b")
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
    f = _orient_fillet(f, (A, B))
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


def _cutters_from_job(job):
    cutters = [CurveCutter(c, c.get("dir") or job["direction"]) for c in job.get("curves", [])]
    for key in job.get("cutter_breps", []):
        if sm.intersect_breps(world_brep(job, "target"), world_brep(job, key))[0]:
            cutters.append(SurfaceCutter(job, key))
    return cutters


def job_split(job):
    """Split the target along all cutters and write EVERY region as its own
    brep (out_region<i>.sm in out_dir). Cutting surfaces that miss the target
    are ignored; nothing is removed."""
    cutters = _cutters_from_job(job)
    if not cutters:
        raise RuntimeError("nothing crosses this surface")
    if "ref_point" not in job:
        lo, hi = world_brep(job, "target").bounding_box()
        job["ref_point"] = [0.5 * (lo[i] + hi[i]) for i in range(3)]
    b, faces, groups, picked, sides = _regions(job, cutters)
    if len(groups) < 2:
        raise RuntimeError("the cutters do not split this surface into separate regions")
    x = job.get("xforms", {}).get("target")
    areas = []
    for gi, g in enumerate(groups):
        part = b.copy()
        pf = part.faces()
        keep = set(g)
        drop = [f for i, f in enumerate(pf) if i not in keep]
        if drop:
            ext.delete_faces(part, drop)
        save(to_local(part, x), os.path.join(job["out_dir"], f"out_region{gi}.sm"))
        areas.append(part.area())
    return {"regions": len(groups), "areas": areas}


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


def job_edge_fillet(job):
    """Fillet / chamfer / blend the edges whose world-space midpoints are given.
    Works on joined sheets and on solids; written back in the object's local space."""
    np = _np()
    b = world_brep(job, "target")
    b_orig = b.copy() if job.get("class_a") else None
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
    if r1 is not None and abs(float(r1) - r0) > 1e-12:
        r = sm.variable_radius_fillet(b, sel, start_radius=r0, end_radius=float(r1), xsect_type=xsect,
                                      continuity=int(job.get("continuity", 1)))
    else:
        r = sm.fillet_edges(b, sel, radius=r0, xsect_type=xsect, continuity=int(job.get("continuity", 1)),
                            thumbweight=float(job.get("thumbweight", 1.0)))
    if r is None or r.face_count() == 0:
        raise RuntimeError("the kernel returned no result (radius too large for the neighbouring faces?)")
    class_a = None
    if job.get("class_a"):
        r, class_a = _class_a(b_orig, r, job, xsect, r0 if r1 is None or abs(float(r1) - r0) <= 1e-12 else None)
    try:
        solid = bool(r.is_manifold_solid())
    except Exception:
        solid = False
    save(to_local(r, job.get("xforms", {}).get("target")), job["out"]["result"])
    out = {"edges": len(sel), "faces": r.face_count(), "solid": solid}
    if class_a is not None:
        out["class_a"] = class_a
    return out


def _surface_key(surf):  # noqa: E302
    d = ext.surface_cvs(surf)
    return (d["degree_u"], d["degree_v"], d["count_u"], d["count_v"],
            tuple(round(c, 9) for p in d["points"] for c in p))


def _class_a(before, r, job, xsect, radius):
    """Rebuild the kernel's dense fillet faces as single-span patches
    (kernel/classa.py). Keeps the kernel result if anything goes wrong."""
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
    fil = [i for i, f in enumerate(r.faces()) if not on_input(f)]
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


JOBS = {"retarget": job_retarget, "edge_fillet": job_edge_fillet, "trim": job_trim, "fillet": job_fillet, "split": job_split, "join": job_join, "unjoin": job_unjoin, "boolean": job_boolean}


def main():
    with open(sys.argv[1]) as f:
        job = json.load(f)
    try:
        res = {"ok": True, "result": JOBS[job["op"]](job)}
    except Exception as exc:
        res = {"ok": False, "error": str(exc), "trace": traceback.format_exc()}
    sys.stdout.write("\n" + json.dumps(res) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
