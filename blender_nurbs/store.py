"""Per-object storage of exact kernel geometry + its Blender display mesh.

A NURBS object is a regular mesh object (the tessellation) carrying:
  obj["nproto_brep"]   kernel Brep in SMLib ASCII format (the truth, local space)
  obj["nproto_rev"]    revision counter, bumps on every rebuild (cache key)
  obj["nproto_disp"]   JSON: edge polylines + CV nets for the overlay
  obj["nproto_hist"]   JSON construction history (op + params), optional
  obj["nproto_in_<i>"] ID pointers to the history inputs
"""

import json
import math
import random
import os
import tempfile

import bpy
import numpy as np

from .kernel import load as _load_kernel

PROP_BREP = "nproto_brep"
PROP_REV = "nproto_rev"
PROP_DISP = "nproto_disp"

_cache = {}  # obj.session_uid -> (rev, Brep)
_disp_cache = {}  # obj.session_uid -> (rev, dict)


def _k():
    sm = _load_kernel()
    if sm is None:
        raise RuntimeError("USD BRep kernel not loaded")
    return sm, sm.blender_ext


def is_nurbs(obj):
    return obj is not None and obj.type == 'MESH' and PROP_BREP in obj


# --------------------------------------------------------------------------
# Brep (de)serialisation
# --------------------------------------------------------------------------

def brep_to_text(brep):
    fd, path = tempfile.mkstemp(suffix=".sm")
    os.close(fd)
    try:
        brep.write_to_file(path, ascii=True)
        with open(path, "r") as f:
            return f.read()
    finally:
        os.remove(path)


def brep_from_text(text):
    sm, _ = _k()
    fd, path = tempfile.mkstemp(suffix=".sm")
    with os.fdopen(fd, "w", newline="\n") as f:      # the same bytes on Windows (no \r added)
        f.write(text)
    try:
        # keep B-splines exactly as written: the stock reader turns e.g. a flat
        # cubic surface into a degree-1 analytic plane (losing its CVs)
        exact = getattr(sm.blender_ext, "read_brep_exact", None)
        return exact(path) if exact else sm.Brep.read_from_file(path, ascii=True)
    finally:
        os.remove(path)


def _new_rev(obj):
    """A fresh revision token for the caches. Random rather than a counter: undo
    restores an older counter value, and a later edit would count up to a value
    the caches already hold for different geometry (stale edges / brep)."""
    old = obj.get(PROP_REV, 0)
    while True:
        r = random.randrange(1, 2 ** 31 - 1)
        if r != old:
            return r


def get_brep(obj):
    rev = obj.get(PROP_REV, 0)
    hit = _cache.get(obj.session_uid)
    if hit and hit[0] == rev:
        return hit[1]
    brep = brep_from_text(obj[PROP_BREP])
    _cache[obj.session_uid] = (rev, brep)
    return brep


def get_surfaces(obj):
    return [f.surface() for f in get_brep(obj).faces()]


def fresh_surfaces(obj):
    """Uncached copies, for kernel ops that may adopt or modify their input."""
    return [f.surface() for f in brep_from_text(obj[PROP_BREP]).faces()]


PROP_NATURAL = "nproto_natural"   # 1 = one untrimmed face (iso boundaries are its edges)
PROP_UNTRIM = "nproto_untrimmed"  # brep text before the first trim (for Untrim)


def is_natural(obj):
    if PROP_NATURAL in obj:
        return bool(obj[PROP_NATURAL])
    return _infer_natural(obj)  # objects made before the flag existed


def _infer_natural(obj):
    """One face whose boundary runs only along the surface's iso boundaries."""
    try:
        brep = get_brep(obj)
        if brep.face_count() != 1:
            return False
        face = brep.faces()[0]
        surf = face.surface()
        (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
        iso = []
        for k in range(33):
            t = k / 32
            iso += [surf.evaluate(u0, v0 + (v1 - v0) * t), surf.evaluate(u1, v0 + (v1 - v0) * t),
                    surf.evaluate(u0 + (u1 - u0) * t, v0), surf.evaluate(u0 + (u1 - u0) * t, v1)]
        iso = np.array(iso)
        size = float(np.linalg.norm(iso.max(axis=0) - iso.min(axis=0))) or 1.0
        for e in face.edges():
            t0, t1 = e.parameter_range()
            for k in (0.25, 0.5, 0.75):
                p = np.array(e.point_at(t0 + (t1 - t0) * k))
                if np.min(np.linalg.norm(iso - p, axis=1)) > 0.02 * size:
                    return False
        return True
    except Exception:
        return False


def set_geometry_text(obj, text, context=None, keep_untrimmed=True):
    """Replace an object's B-rep with kernel ASCII (e.g. a worker result)."""
    if keep_untrimmed and PROP_UNTRIM not in obj and PROP_BREP in obj:
        obj[PROP_UNTRIM] = obj[PROP_BREP]
        obj[PROP_UNTRIM + "_natural"] = obj.get(PROP_NATURAL, 0)
    brep = brep_from_text(text)
    obj[PROP_BREP] = text
    obj[PROP_NATURAL] = 0
    obj[PROP_REV] = _new_rev(obj)
    obj[PROP_DISP] = json.dumps(_display_data(brep))
    _cache[obj.session_uid] = (obj[PROP_REV], brep)
    write_mesh(obj, brep, context)


# --------------------------------------------------------------------------
# Building / updating objects
# --------------------------------------------------------------------------

def to_brep(geom):
    """Accept a kernel Surface or Brep."""
    sm, ext = _k()
    if isinstance(geom, sm.Brep):
        return geom
    return ext.brep_from_surface(geom)


def tess_settings(context=None):
    scn = (context or bpy.context).scene
    s = getattr(scn, "nproto_settings", None)
    if s is None:
        return 0.0005, 5.0
    return s.chord_tolerance, s.angle_tolerance


def edge_angle_setting(context=None):
    scn = (context or bpy.context).scene
    s = getattr(scn, "nproto_settings", None)
    return float(getattr(s, "edge_angle_tolerance", 2.0)) if s is not None else 2.0


def max_edge_setting(context=None):
    scn = (context or bpy.context).scene
    s = getattr(scn, "nproto_settings", None)
    return float(getattr(s, "max_edge_rel", 1.0 / 30.0)) if s is not None else 1.0 / 30.0


def _split_knots(P, W, U, p, k, tol):
    """Split CV rows (axis 0 of P / W) of a clamped direction at its C0 knot k
    (multiplicity p): (left, right) as (P, W, knots), exactly the same shape."""
    s = next(i for i, x in enumerate(U) if abs(x - k) <= tol)
    m = sum(1 for x in U if abs(x - k) <= tol)
    b = s - 1                                   # the CV the curve passes through at k
    left = (P[:b + 1], W[:b + 1], list(U[:s]) + [k] * (p + 1))
    right = (P[b:], W[b:], [k] * (p + 1) + list(U[s + m:]))
    return left, right


def _crease_pieces(surf, angle_deg):
    """An untrimmed surface with folds -- C0 knot lines where the normal jumps
    by more than angle_deg, like a loft through rectangles -- as smooth pieces
    split exactly at those lines (None when it has none). Tessellated piece by
    piece, the creases stay sharp: one mesh across a fold gets vertex normals
    from one side only (dark streaks), and the refinement keeps splitting the
    triangles there (normals more than angle_deg apart) into slivers."""
    _, ext = _k()
    try:
        d = ext.surface_cvs(surf)
    except Exception:
        return None
    nu, nv, du, dv = d["count_u"], d["count_v"], int(d["degree_u"]), int(d["degree_v"])
    Uu, Uv = [float(x) for x in d["knots_u"]], [float(x) for x in d["knots_v"]]
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    W = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    cos_c = math.cos(math.radians(angle_deg))

    def normal(u, v):
        _, su, sv = surf.derivatives(u, v)
        n = np.cross(su, sv)
        return n / (np.linalg.norm(n) or 1.0)

    def folds(U, p, other, along_u):
        lo, hi = U[p], U[-p - 1]
        tol = 1e-9 * (hi - lo or 1.0)
        eps = 1e-6 * (hi - lo or 1.0)
        out = []
        if abs(U[0] - lo) > tol or abs(U[-1] - hi) > tol:
            return out                         # an unclamped (periodic) knot vector: left whole
        for k in sorted({x for x in U if lo + tol < x < hi - tol}):
            if sum(1 for x in U if abs(x - k) <= tol) != p:
                continue                       # smoother than C0 (or a gap): nothing to split
            o0, o1 = other
            for t in np.linspace(o0, o1, 7)[1:-1]:
                a = normal(k - eps, t) if along_u else normal(t, k - eps)
                b = normal(k + eps, t) if along_u else normal(t, k + eps)
                if float(a @ b) < cos_c:
                    out.append(k)
                    break
        return out
    ku = folds(Uu, du, (Uv[dv], Uv[-dv - 1]), True)
    kv = folds(Uv, dv, (Uu[du], Uu[-du - 1]), False)
    if not ku and not kv or (len(ku) + 1) * (len(kv) + 1) > 64:
        return None
    rows = [(P, W, Uu)]
    for k in ku:
        Pl, Wl, Ul = rows.pop()
        left, right = _split_knots(Pl, Wl, Ul, du, k, 1e-9)
        rows += [left, right]
    pieces = []
    for Pr, Wr, Ur in rows:
        cols = [(Pr.transpose(1, 0, 2), Wr.T, Uv)]
        for k in kv:
            Pc, Wc, Uc = cols.pop()
            left, right = _split_knots(Pc, Wc, Uc, dv, k, 1e-9)
            cols += [left, right]
        for Pc, Wc, Uc in cols:
            Pq, Wq = Pc.transpose(1, 0, 2), Wc.T                # back to (u, v)
            rational = bool(np.abs(Wq - 1.0).max() > 1e-12)
            pieces.append(ext.surface_from_cvs(du, dv, [tuple(map(float, x)) for x in Pq.reshape(-1, 3)], Ur, Uc,
                                               [float(w) for w in Wq.ravel()] if rational else None))
    return pieces


def _tessellate(brep, chord, angle, edge_angle=None, max_edge_rel=1.0 / 30.0, natural=False, diag=None):
    """edge_angle: separate (finer) angle tolerance for the edges. Neighbouring
    objects are tessellated independently; along a shared curved edge each
    mesh's chords cut inside the curve, leaving slits between the meshes. Fine
    edge sampling keeps those slits sub-pixel without densifying the faces.
    natural: the brep is one untrimmed face -- a surface with folds is then
    tessellated piece by piece (_crease_pieces). diag: size the relative
    tolerances refer to (default: the brep's own)."""
    if diag is None:
        bb = brep.bounding_box()
        try:
            lo, hi = np.array(bb[0]), np.array(bb[1])
            diag = float(np.linalg.norm(hi - lo)) or 1.0
        except Exception:  # bbox format differs: fall back to absolute
            diag = 1.0
    if natural and brep.face_count() == 1:
        pieces = _crease_pieces(brep.faces()[0].surface(), angle)
        if pieces:
            Ps, Ns, Ts, off = [], [], [], 0
            for srf in pieces:
                p_, n_, t_ = _tessellate(to_brep(srf), chord, angle, edge_angle, max_edge_rel, diag=diag)
                Ps.append(np.asarray(p_, np.float32))
                Ns.append(np.asarray(n_, np.float32))
                Ts.append(_tri_array(t_) + off)
                off += len(p_)
            return np.vstack(Ps), np.vstack(Ns), np.vstack(Ts)
    # a flat face shades the same however it is split (one normal): no length
    # limit there, which only added triangles
    flat = brep.face_count() == 1 and _flat(brep.faces()[0].surface(), diag)
    if flat:
        max_edge_rel = 0.0
    pb = brep.tessellate(chord_height_tolerance=chord * diag,
                         curve_angle_tolerance_deg=min(angle, edge_angle) if edge_angle else angle,
                         surface_angle_tolerance_deg=angle,
                         max_edge_length=(max_edge_rel * diag) if max_edge_rel else 0.0,
                         allow_partial=True, return_failures=True)
    pb, failures = pb
    if failures:
        print(f"NURBS: {len(failures)} face(s) failed to tessellate")
    a = pb.to_numpy_arrays()
    pts = a["points"].astype(np.float32)
    nrm = a["normals"].astype(np.float32)
    f = np.asarray(a["faces"])
    if len(f) % 4 == 0 and np.all(f[0::4] == 3):       # all triangles (the usual case): one array
        tris = f.reshape(-1, 4)[:, 1:].astype(np.int64)
    else:
        tris = []
        i = 0
        while i < len(f):
            c = int(f[i])
            tris.append(f[i + 1:i + 1 + c])
            i += 1 + c
    try:
        tris = _zip(pts, nrm, tris, edge_pts=lambda: _edge_samples(brep), tol=4.0 * chord * diag)
    except Exception as exc:
        print("NURBS: crack repair skipped:", exc)
    try:
        tris = _unfold(pts, nrm, tris)          # before refining: a split fold is several folds
    except Exception as exc:
        print("NURBS: mesh repair skipped:", exc)
    if max_edge_rel:
        try:
            if brep.face_count() == 1:
                pts, nrm, tris = _refine_single(brep.faces()[0].surface(), pts, nrm, tris, max_edge_rel * diag,
                                                max_angle=angle)
        except Exception as exc:
            print("NURBS: mesh refinement skipped:", exc)
    try:
        tris = _unfold(pts, nrm, tris)
    except Exception as exc:
        print("NURBS: mesh repair skipped:", exc)
    if brep.face_count() == 1:
        try:
            pts, nrm = _relax_folds(brep.faces()[0].surface(), pts, nrm, tris)
        except Exception as exc:
            print("NURBS: fold relax skipped:", exc)
    return pts, nrm, tris


def _flat(surf, size):
    try:
        return bool(surf.is_planar(1e-7 * size))
    except Exception:
        return False


def _tri_array(tris):
    """(m, 3) int array of a face list (larger polygons as fans); an (m, 3)
    array passes through. The mesh helpers keep triangles in one array: a
    list of 100k small arrays cost more than the work done on them."""
    if isinstance(tris, np.ndarray) and tris.ndim == 2 and tris.shape[1] == 3:
        return tris.astype(np.int64, copy=False)
    T = []
    for t in tris:
        if len(t) == 3:
            T.append((int(t[0]), int(t[1]), int(t[2])))
            continue
        t = [int(x) for x in t]
        for k in range(1, len(t) - 1):
            T.append((t[0], t[k], t[k + 1]))
    return np.asarray(T, np.int64).reshape(-1, 3)


def _folded(P, Nn, T, good=0.3):
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    ar = np.linalg.norm(fn, axis=1)
    nv = Nn[T[:, 0]] + Nn[T[:, 1]] + Nn[T[:, 2]]
    cos = np.sum(fn * nv, axis=1) / np.maximum(ar * np.linalg.norm(nv, axis=1), 1e-300)
    return np.nonzero((cos < good) & (ar > 1e-6 * (float(np.median(ar)) or 1.0)))[0]


def _relax_folds(surf, pts, nrm, tris, rounds=4):
    """Last resort for folds the edge flips can't undo (a sliver strip along a
    crack in the kernel's mesh, its two sides crossing over): every inner
    vertex of a folded triangle moves to the surface point at the average
    (u, v) of its neighbours -- on the surface exactly, with its exact
    normal -- until the triangles face along the surface again."""
    if isinstance(tris, np.ndarray):
        T = _tri_array(tris)
    else:
        T = np.array([[int(x) for x in t[:3]] for t in tris if len(t) == 3], int).reshape(-1, 3)
    if not len(T):
        return pts, nrm
    P = np.asarray(pts, float).copy()
    Nn = np.asarray(nrm, float).copy()
    Nn /= np.maximum(np.linalg.norm(Nn, axis=1), 1e-30)[:, None]
    bad = _folded(P, Nn, T)
    if not len(bad):
        return pts, nrm
    boundary, start, nbrs = _mesh_topology(T, len(P))
    foot = _make_foot(surf)
    (u0_, u1_), (v0_, v1_) = surf.u_domain(), surf.v_domain()
    pu_, pv_ = _periodic(surf)
    du_p, dv_p = (u1_ - u0_) if pu_ else 0.0, (v1_ - v0_) if pv_ else 0.0

    def _near(x, ref, period):
        """x moved by whole periods to within half a period of ref."""
        return x if not period else x - period * round((x - ref) / period)
    uv = {}
    for _ in range(rounds):
        verts = np.unique(T[bad])
        verts = verts[~boundary[verts]].tolist()
        if not verts:
            break
        for v in verts:
            nb = nbrs[start[v]:start[v + 1]].tolist()
            if not nb:
                continue
            for w in nb + [v]:
                if w not in uv:
                    uv[w] = foot(P[w])
            u = float(np.mean([_near(uv[w][0], uv[v][0], du_p) for w in nb]))
            vv = float(np.mean([_near(uv[w][1], uv[v][1], dv_p) for w in nb]))
            if du_p:
                u = u0_ + (u - u0_) % du_p
            if dv_p:
                vv = v0_ + (vv - v0_) % dv_p
            pt, su, sv = surf.derivatives(u, vv)
            n = np.cross(su, sv)
            n /= (np.linalg.norm(n) or 1.0)
            if float(n @ Nn[v]) < 0:
                n = -n
            P[v], Nn[v], uv[v] = np.array(pt, float), n, (u, vv)
        bad = _folded(P, Nn, T)
        if not len(bad):
            break
    return np.asarray(P, np.float32), np.asarray(Nn, np.float32)


def _mesh_topology(T, n):
    """(boundary vertex mask, start, nbrs) of a triangle mesh with n vertices:
    a vertex is on the boundary when one of its edges has other than two
    triangles; vertex v's neighbours are nbrs[start[v]:start[v + 1]]. Whole
    arrays at once (a Python dict per edge cost ~0.2 s on a zebra mesh)."""
    T = np.asarray(T, np.int64)
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E.sort(axis=1)
    keys, cnt = np.unique(E[:, 0] * n + E[:, 1], return_counts=True)
    boundary = np.zeros(n, bool)
    bk = keys[cnt != 2]
    boundary[bk // n] = True
    boundary[bk % n] = True
    pair = np.unique(np.concatenate([E[:, 0] * n + E[:, 1], E[:, 1] * n + E[:, 0]]))
    start = np.searchsorted(pair // n, np.arange(n + 1))
    return boundary, start, pair % n


def nearest_rows(flat, Q, batch=4096):
    """Index of the row of flat nearest to each point of Q: |q|^2 - 2 q.f + |f|^2
    as one matrix product per batch (an (m, k, 3) difference array per batch
    was most of the cost of finding thousands of (u, v) feet)."""
    flat = np.asarray(flat, float)
    Q = np.asarray(Q, float).reshape(-1, 3)
    f2 = np.einsum("ij,ij->i", flat, flat)
    out = np.empty(len(Q), np.int64)
    for s0 in range(0, len(Q), batch):
        q = Q[s0:s0 + batch]
        out[s0:s0 + batch] = np.argmin(f2[None, :] - 2.0 * (q @ flat.T), axis=1)
    return out


def surface_feet(surf, Q, g=33):
    """Closest (u, v) on surf of every point of Q: each from the nearest point
    of a g x g grid, then Newton -- one matrix product for all the seeds."""
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    gu, gv = np.linspace(u0, u1, g), np.linspace(v0, v1, g)
    flat = np.array([[surf.evaluate(u, v) for v in gv] for u in gu]).reshape(-1, 3)
    per, dom = _periodic(surf), ((u0, u1), (v0, v1))
    Q = np.asarray(Q, float).reshape(-1, 3)
    return np.array([_newton_uv(surf, q, gu[k // g], gv[k % g], dom, per)
                     for q, k in zip(Q, nearest_rows(flat, Q).tolist())], float).reshape(-1, 2)


def _make_foot(surf, g=33):
    """Closest (u, v) on surf for a point: grid start, then Newton."""
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    gu, gv = np.linspace(u0, u1, g), np.linspace(v0, v1, g)
    flat = np.array([[surf.evaluate(u, v) for v in gv] for u in gu]).reshape(-1, 3)

    per = _periodic(surf)

    def foot(q):
        k = int(nearest_rows(flat, q)[0])
        return _newton_uv(surf, q, gu[k // g], gv[k % g], ((u0, u1), (v0, v1)), per)
    return foot


def _zip(pts, nrm, tris, max_angle=25.0, edge_pts=None, tol=0.0):
    """Close cracks in the kernel's tessellation (zippering). Its mesh of a
    trimmed face can leave thin gaps: two needle triangles fanning out of a
    corner in nearly the same direction without sharing an edge, or a long
    edge facing a chain of short ones (T-junction). Harmless while hairline,
    but refinement moves points on one side onto the surface and not the
    other, which opens a visible crack. Wherever an open edge arrives at a
    vertex and another leaves it in nearly the same direction (< max_angle),
    the thin triangle between them is added, and so on along the gap. Along
    the trim, consecutive open edges point opposite ways and are left."""
    P = np.asarray(pts, float)
    Nn = np.asarray(nrm, float)
    Ta = _tri_array(tris)
    if not len(Ta):
        return tris
    half = set(map(tuple, Ta[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).tolist()))   # per triangle, in order
    B = {h for h in half if (h[1], h[0]) not in half}
    if not B:
        return tris
    outb, inb = {}, {}
    for p_, q_ in B:
        outb.setdefault(p_, set()).add(q_)
        inb.setdefault(q_, set()).add(p_)
    cmax = float(np.cos(np.radians(max_angle)))
    queue = [v for v in outb if v in inb]
    added = []
    guard = 4 * len(B) + 16
    while queue and guard > 0:
        guard -= 1
        v = queue.pop()
        best = None
        for p_ in list(inb.get(v, ())):
            dp = P[p_] - P[v]
            lp = np.linalg.norm(dp)
            for q_ in list(outb.get(v, ())):
                if q_ == p_:
                    continue
                dq = P[q_] - P[v]
                lq = np.linalg.norm(dq)
                if lp < 1e-30 or lq < 1e-30:
                    continue
                c = float(dp @ dq) / (lp * lq)
                if c > cmax and (best is None or c > best[0]):
                    best = (c, p_, q_)
        if best is None:
            continue
        _, p_, q_ = best
        # the new triangle (v, p, q) must face along the surface
        fn = np.cross(P[p_] - P[v], P[q_] - P[v])
        nv = Nn[v] + Nn[p_] + Nn[q_]
        if float(fn @ nv) < 0:
            continue
        added.append((v, p_, q_))
        B.discard((p_, v)); B.discard((v, q_))
        inb[v].discard(p_); outb[p_].discard(v)
        outb[v].discard(q_); inb[q_].discard(v)
        if (q_, p_) in B:                               # closes the gap completely
            B.discard((q_, p_)); outb[q_].discard(p_); inb[p_].discard(q_)
        else:
            B.add((p_, q_)); outb.setdefault(p_, set()).add(q_); inb.setdefault(q_, set()).add(p_)
        queue += [p_, q_, v]
    if edge_pts is not None:
        added += _fill_small_loops(P, Nn, B, outb, edge_pts, tol)
    if not added:
        return tris
    return np.vstack([Ta, np.asarray(added, np.int64).reshape(-1, 3)])


def _edge_samples(brep, n=120):
    out = []
    for e in brep.edges():
        try:
            t0, t1 = e.parameter_range()
            out += [list(e.point_at(t0 + (t1 - t0) * i / (n - 1))) for i in range(n)]
        except Exception:
            pass
    return np.asarray(out, float).reshape(-1, 3)


def _fill_small_loops(P, Nn, B, outb, edge_pts, tol, share=0.1):
    """Holes left after zippering: closed loops of open edges much shorter
    than the longest loop (the trim), lying inside the face (most of their
    vertices away from the B-rep's own edges -- a real trimmed hole is kept),
    are filled by ear clipping in the plane of their averaged normal, facing
    the same way as the mesh around them."""
    used = set()
    loops = []
    for h in list(B):
        if h in used:
            continue
        loop, cur = [h[0]], h
        used.add(h)
        ok = True
        for _ in range(100000):
            v = cur[1]
            if v == loop[0]:
                break
            nxt = [q for q in outb.get(v, ()) if (v, q) in B and (v, q) not in used]
            if not nxt:
                ok = False
                break
            cur = (v, nxt[0])
            used.add(cur)
            loop.append(v)
        if ok and len(loop) >= 3:
            loops.append(loop)
    if len(loops) < 2:
        return []
    per = [float(sum(np.linalg.norm(P[l[i]] - P[l[(i + 1) % len(l)]]) for i in range(len(l)))) for l in loops]
    big = max(per)
    out = []
    E = None
    for loop, pr in zip(loops, per):
        if pr >= share * big or len(loop) > 400:
            continue
        if E is None:
            E = edge_pts()
        if len(E):
            d = np.min(np.linalg.norm(P[loop][:, None, :] - E[None, :, :], axis=2), axis=1)
            if np.mean(d < max(tol, 1e-3 * pr)) > 0.5:     # on the B-rep's edges: a real hole
                continue
        poly = loop[::-1]                         # the fill runs against the open edges
        n = Nn[poly].sum(axis=0)
        if np.linalg.norm(n) < 1e-30:
            continue
        n = n / np.linalg.norm(n)
        e1 = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 0.1:
            e1 = np.cross(n, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        Q = np.c_[(P[poly] - P[poly].mean(axis=0)) @ e1, (P[poly] - P[poly].mean(axis=0)) @ e2]
        area = 0.5 * float(np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]))
        if area <= 0:
            continue
        idx = list(range(len(poly)))
        tri = []
        guard = 4 * len(idx) * len(idx) + 8
        while len(idx) > 3 and guard > 0:
            guard -= 1
            clipped = False
            for k in range(len(idx)):
                i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
                a, b, c = Q[i0], Q[i1], Q[i2]
                cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if cr <= 0:
                    continue
                inside = False
                for j in idx:
                    if j in (i0, i1, i2):
                        continue
                    p_ = Q[j]
                    d1 = (b[0] - a[0]) * (p_[1] - a[1]) - (b[1] - a[1]) * (p_[0] - a[0])
                    d2 = (c[0] - b[0]) * (p_[1] - b[1]) - (c[1] - b[1]) * (p_[0] - b[0])
                    d3 = (a[0] - c[0]) * (p_[1] - c[1]) - (a[1] - c[1]) * (p_[0] - c[0])
                    if d1 > 0 and d2 > 0 and d3 > 0:
                        inside = True
                        break
                if inside:
                    continue
                tri.append((poly[i0], poly[i1], poly[i2]))
                idx.pop(k)
                clipped = True
                break
            if not clipped:
                break
        if len(idx) == 3:
            tri.append((poly[idx[0]], poly[idx[1]], poly[idx[2]]))
            out += tri
    return out


def _unfold(pts, nrm, tris, passes=4, good=0.3):
    """Repair folded triangles: the kernel's tessellation of a surface with a
    strongly uneven parameterization can leave triangles whose plane stands
    across (or against) the surface normals at their corners -- they overlap
    their neighbours and show as spikes in shading and zebra. Such a triangle
    gets the edge it shares with a neighbour flipped (the other diagonal of
    their quad) when both new triangles face along the surface; vertices and
    normals are unchanged. Only triangles inside one smooth region count
    (corner normals within ~45 degrees); zero-area slivers are harmless."""
    import math
    P = np.asarray(pts, float)
    Nn = np.asarray(nrm, float)
    Nu = Nn / np.maximum(np.linalg.norm(Nn, axis=1), 1e-30)[:, None]
    Ta = _tri_array(tris)
    if not len(Ta):
        return tris
    fn = np.cross(P[Ta[:, 1]] - P[Ta[:, 0]], P[Ta[:, 2]] - P[Ta[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    nv = Nu[Ta[:, 0]] + Nu[Ta[:, 1]] + Nu[Ta[:, 2]]
    den = area * np.linalg.norm(nv, axis=1)
    cos = np.where(den > 1e-30, np.sum(fn * nv, axis=1) / np.maximum(den, 1e-300), 1.0)
    smooth = np.minimum.reduce([np.sum(Nu[Ta[:, i]] * Nu[Ta[:, j]], axis=1) for i, j in ((0, 1), (1, 2), (2, 0))]) > 0.7
    scale = float(np.median(area)) if len(area) else 0.0
    bad = np.nonzero((cos < good) & smooth & (area > 1e-6 * scale))[0]
    if not len(bad):
        return tris
    T = Ta.tolist()
    cos = cos.tolist()
    Pl, Nl = P.tolist(), Nu.tolist()

    def cosf(a, b, c):
        pa, pb, pc = Pl[a], Pl[b], Pl[c]
        e1 = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        e2 = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        f = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
        na, nb, nc = Nl[a], Nl[b], Nl[c]
        v = (na[0] + nb[0] + nc[0], na[1] + nb[1] + nc[1], na[2] + nb[2] + nc[2])
        l = math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
        lv = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return -1.0 if l < 1e-30 or lv < 1e-30 else (f[0] * v[0] + f[1] * v[1] + f[2] * v[2]) / (l * lv)

    # edge -> triangles, first only around the folds (every triangle with a
    # vertex there: exact for any edge with an end there); the whole mesh as
    # soon as a lookup leaves that region
    near_v = np.zeros(len(P), bool)
    near_v[Ta[bad].ravel()] = True
    for _ in range(3):
        near_v[Ta[near_v[Ta].any(axis=1)].ravel()] = True
    near_t = np.nonzero(near_v[Ta].any(axis=1))[0]
    emap = {}
    whole = [len(near_t) == len(Ta)]

    def add_all(ids):
        for i in ids:
            a, b, c = T[i]
            for x, y in ((a, b), (b, c), (c, a)):
                emap.setdefault((min(x, y), max(x, y)), []).append(i)
    add_all(near_t.tolist())

    def edge_tris(x, y):
        if not whole[0] and not (near_v[x] or near_v[y]):
            emap.clear()
            add_all(range(len(T)))
            whole[0] = True
        return emap.get((min(x, y), max(x, y)), [])

    todo = sorted(bad.tolist(), key=lambda i: cos[i])
    for _ in range(passes):
        nxt = []
        for i in todo:
            if cos[i] >= good:
                continue
            best = None
            a0, b0, c0 = T[i]
            for x, y, z in ((a0, b0, c0), (b0, c0, a0), (c0, a0, b0)):
                nb = [j for j in edge_tris(x, y) if j != i]
                if len(nb) != 1:
                    continue
                j = nb[0]
                d = [v for v in T[j] if v not in (x, y)]
                if len(d) != 1 or d[0] == z:
                    continue
                d = d[0]
                nd, nz = Nl[d], Nl[z]
                if nd[0] * nz[0] + nd[1] * nz[1] + nd[2] * nz[2] < 0.7:     # the quad must be one smooth patch
                    continue
                t1, t2 = [x, d, z], [d, y, z]
                c1, c2 = cosf(*t1), cosf(*t2)
                if min(c1, c2) > max(min(cos[i], cos[j]), 0.0) and (best is None or min(c1, c2) > best[0]):
                    best = (min(c1, c2), j, t1, t2, c1, c2)
            if best is None:
                continue
            _, j, t1, t2, c1, c2 = best
            for tri in (T[i], T[j], t1, t2):     # (switch to the whole mesh now if needed, before anything changes)
                for x, y in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    edge_tris(x, y)
            for k_, tri in ((i, T[i]), (j, T[j])):
                for x, y in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    lst = emap.get((min(x, y), max(x, y)), [])
                    if k_ in lst:
                        lst.remove(k_)
            T[i], T[j], cos[i], cos[j] = t1, t2, c1, c2
            for k_, tri in ((i, t1), (j, t2)):
                for x, y in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    emap.setdefault((min(x, y), max(x, y)), []).append(k_)
            nxt += [k for k in (i, j) if cos[k] < good]
        todo = [k for k in set(nxt) if cos[k] < good]
        if not todo:
            break
    return np.asarray(T, np.int64).reshape(-1, 3)


def _periodic(surf):
    """(u, v): whether the surface wraps around in that direction -- periodic,
    or closed (a fillet ring along a circular edge: its two ends are the same
    curve though the knots are clamped). Averaging parameters across such a
    seam lands on the far side of the ring."""
    try:
        pu, pv = bool(surf.is_periodic_u()), bool(surf.is_periodic_v())
    except Exception:
        pu, pv = False, False
    if pu and pv:
        return pu, pv
    try:
        (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
        t = np.linspace(0.0, 1.0, 7)
        A = np.array([surf.evaluate(u0, v0 + (v1 - v0) * x) for x in t] +
                     [surf.evaluate(u0 + (u1 - u0) * x, v0) for x in t], float)
        B = np.array([surf.evaluate(u1, v0 + (v1 - v0) * x) for x in t] +
                     [surf.evaluate(u0 + (u1 - u0) * x, v1) for x in t], float)
        G = np.array([surf.evaluate(u0 + (u1 - u0) * x, v0 + (v1 - v0) * y) for x in t for y in t], float)
        tol = 1e-6 * (float(np.ptp(G, axis=0).max()) or 1.0)
        d = np.linalg.norm(A - B, axis=1)
        return (pu or bool(d[:7].max() < tol)), (pv or bool(d[7:].max() < tol))
    except Exception:
        return pu, pv


def _newton_uv(surf, q, u, v, dom, per=(False, False)):
    """Closest (u, v) on surf to point q from a start: Newton on the 2 x 2
    normal equations in plain floats (numpy's lstsq on a 3 x 2 matrix costs
    ~10 us a call; it is only used where the equations are degenerate). In a
    periodic direction the parameter wraps around instead of stopping at the
    domain's end: a point just past a tube's seam has u near the end, not 0
    (clamped, it landed on the seam, and points made from it folded the mesh)."""
    import math
    (u0, u1), (v0, v1) = dom
    pu, pv = per
    qx, qy, qz = float(q[0]), float(q[1]), float(q[2])
    eps = 1e-12 * (abs(u1 - u0) + abs(v1 - v0))
    for _ in range(20):
        pt, su, sv = surf.derivatives(u, v)
        rx, ry, rz = qx - pt[0], qy - pt[1], qz - pt[2]
        a = su[0] * su[0] + su[1] * su[1] + su[2] * su[2]
        b = su[0] * sv[0] + su[1] * sv[1] + su[2] * sv[2]
        c = sv[0] * sv[0] + sv[1] * sv[1] + sv[2] * sv[2]
        g1 = su[0] * rx + su[1] * ry + su[2] * rz
        g2 = sv[0] * rx + sv[1] * ry + sv[2] * rz
        det = a * c - b * b
        if det > 1e-12 * a * c:
            du, dv = (c * g1 - b * g2) / det, (a * g2 - b * g1) / det
        else:
            st = np.linalg.lstsq(np.array([su, sv], float).T, np.array([rx, ry, rz]), rcond=None)[0]
            du, dv = float(st[0]), float(st[1])
        if not (math.isfinite(du) and math.isfinite(dv)):
            break
        u = (u0 + (u + du - u0) % (u1 - u0)) if pu else min(max(u + du, u0), u1)
        v = (v0 + (v + dv - v0) % (v1 - v0)) if pv else min(max(v + dv, v0), v1)
        if math.sqrt(du * du + dv * dv) < eps:
            break
    return u, v


def _refine_single(surf, pts, nrm, tris, lmax, max_tris=120000, passes=8, max_angle=None):
    """The kernel's tessellation of a TRIMMED face can leave triangles far
    longer than max_edge_length (a trim crossing its coarse cells is filled
    with long fans), which shade as facets and wedges on a curved face. Split
    every triangle's over-long longest edge at its midpoint in the surface's
    (u, v) -- the new point lies exactly on the surface, with the exact normal
    -- until none is longer than lmax (single-face objects: the surface is
    unambiguous). Each pass works on whole arrays (the split pattern, the new
    triangles, the midpoints); new points are numbered in the order a scan of
    the triangles first meets their edges."""
    P = np.asarray(pts, float)
    Nn = np.asarray(nrm, float)
    T = _tri_array(tris)
    cmin = float(np.cos(np.radians(max_angle))) if max_angle else None
    BIG = np.int64(1) << 32

    def to_split(T):
        """Edges to split (as sorted vertex pairs), chosen for all triangles at
        once: each triangle's longest edge if it is longer than lmax, else its
        longest edge whose end normals differ by more than max_angle (linear
        interpolation of the normals across it would bend zebra stripes)."""
        if not len(T):
            return np.zeros(0, np.int64)
        Na = Nn / np.maximum(np.linalg.norm(Nn, axis=1), 1e-30)[:, None]
        E = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)       # (m, 3, 2)
        L = np.linalg.norm(P[E[:, :, 0]] - P[E[:, :, 1]], axis=2)               # (m, 3)
        k = np.argmax(L, axis=1)
        rows = np.arange(len(T))
        pick = np.full(len(T), -1)
        long_ = L[rows, k] > lmax
        pick[long_] = k[long_]
        if cmin is not None:
            dots = np.sum(Na[E[:, :, 0]] * Na[E[:, :, 1]], axis=2)              # (m, 3)
            bent = (dots < cmin) & (L > 1e-4 * lmax)
            kb = np.argmax(np.where(bent, L, -1.0), axis=1)
            sel = (~long_) & bent.any(axis=1)
            pick[sel] = kb[sel]
        idx = np.nonzero(pick >= 0)[0]
        ed = E[idx, pick[idx]]
        ed.sort(axis=1)
        return np.unique(ed[:, 0] * BIG + ed[:, 1])                              # edge keys

    def key(x, y):
        return np.minimum(x, y) * BIG + np.maximum(x, y)

    split = to_split(T)
    if not len(split):
        return pts, nrm, tris
    dom = (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    per = _periodic(surf)
    gu, gv = np.linspace(u0, u1, 33), np.linspace(v0, v1, 33)
    flat = np.array([[surf.evaluate(u, v) for v in gv] for u in gu]).reshape(-1, 3)

    def feet(Q):
        """Closest (u, v) of every point: nearest grid point, then Newton."""
        out = np.empty((len(Q), 2))
        for r, kk in enumerate(nearest_rows(flat, Q).tolist()):
            out[r] = _newton_uv(surf, Q[r], gu[kk // 33], gv[kk % 33], dom, per)
        return out

    UV = np.full((len(P), 2), np.nan)            # (u, v) of the points found so far
    for it in range(passes):
        if it:
            split = to_split(T)
        if not len(split) or len(T) > max_tris:
            break
        m = len(T)
        hit = np.stack([np.isin(key(T[:, 0], T[:, 1]), split), np.isin(key(T[:, 1], T[:, 2]), split),
                        np.isin(key(T[:, 2], T[:, 0]), split)], axis=1)
        nh = hit.sum(axis=1)
        r = np.zeros(m, np.int64)                # rotate so the pattern starts at edge 0
        r[nh == 1] = np.argmax(hit[nh == 1], axis=1)
        r[nh == 2] = (np.argmin(hit[nh == 2], axis=1) + 1) % 3
        rows = np.arange(m)
        A, B, C = T[rows, r], T[rows, (r + 1) % 3], T[rows, (r + 2) % 3]
        # midpoints, numbered in the order the scan meets them: per triangle (a, b), (b, c), (c, a)
        calls = np.full((m, 3), -1, np.int64)
        calls[nh >= 1, 0] = key(A, B)[nh >= 1]
        calls[nh >= 2, 1] = key(B, C)[nh >= 2]
        calls[nh == 3, 2] = key(C, A)[nh == 3]
        flat_calls = calls.ravel()
        flat_calls = flat_calls[flat_calls >= 0]
        uk, first = np.unique(flat_calls, return_index=True)
        order = np.argsort(first, kind="stable")
        newk = uk[order]
        ids = np.empty(len(uk), np.int64)
        ids[order] = len(P) + np.arange(len(uk))
        ea, eb = newk // BIG, newk % BIG
        need = np.unique(np.concatenate([ea, eb]))
        need = need[np.isnan(UV[need, 0])]
        if len(need):
            UV[need] = feet(P[need])
        ua, ub = UV[ea], UV[eb]
        muv = 0.5 * (ua + ub)
        seam = (np.abs(ua[:, 0] - ub[:, 0]) > 0.5 * (u1 - u0)) | (np.abs(ua[:, 1] - ub[:, 1]) > 0.5 * (v1 - v0))
        if seam.any():
            muv[seam] = feet(0.5 * (P[ea[seam]] + P[eb[seam]]))
        D = [surf.derivatives(u, v) for u, v in muv.tolist()]
        newP = np.array([d[0] for d in D], float).reshape(-1, 3)
        n = np.cross(np.array([d[1] for d in D], float).reshape(-1, 3), np.array([d[2] for d in D], float).reshape(-1, 3))
        ln = np.linalg.norm(n, axis=1)
        n /= np.where(ln > 0, ln, 1.0)[:, None]
        n[np.sum(n * (Nn[ea] + Nn[eb]), axis=1) < 0] *= -1.0
        P = np.vstack([P, newP])
        Nn = np.vstack([Nn, n])
        UV = np.vstack([UV, muv])

        def mid(k_):
            return ids[np.searchsorted(uk, k_)]
        cnt = np.array([1, 2, 3, 4])[nh]
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        out = np.empty((int(cnt.sum()), 3), np.int64)
        s0 = nh == 0
        out[start[s0]] = T[s0]
        for h_ in (1, 2, 3):
            w = np.nonzero(nh == h_)[0]
            if not len(w):
                continue
            a_, b_, c_ = A[w], B[w], C[w]
            m0 = mid(key(a_, b_))
            if h_ == 1:
                new_t = [(a_, m0, c_), (m0, b_, c_)]
            elif h_ == 2:
                m1 = mid(key(b_, c_))
                new_t = [(a_, m0, c_), (m0, b_, m1), (m0, m1, c_)]
            else:
                m1, m2 = mid(key(b_, c_)), mid(key(c_, a_))
                new_t = [(a_, m0, m2), (m0, b_, m1), (m2, m1, c_), (m0, m1, m2)]
            for j, t_ in enumerate(new_t):
                out[start[w] + j] = np.stack(t_, axis=1)
        T = out
    return P.astype(np.float32), Nn.astype(np.float32), T


def _display_data(brep):
    """Edge polylines and CV nets in local space, for the overlay."""
    _, ext = _k()
    edges = []
    n = 40
    for e in brep.edges():
        poly = []
        try:
            t0, t1 = e.parameter_range()
            poly = [list(e.point_at(t0 + (t1 - t0) * i / (n - 1))) for i in range(n)]
        except Exception:
            try:  # some edges (e.g. fillet rails) only evaluate through their curve
                c = e.curve()
                t0, t1 = c.parameter_range()
                poly = [list(c.evaluate(t0 + (t1 - t0) * i / (n - 1))) for i in range(n)]
            except Exception:
                poly = []  # keep the index: refs "E<i>" follow brep.edges() order
        edges.append(poly)
    shared = []
    for i, e in enumerate(brep.edges()):
        try:
            if len(e.faces()) == 2:
                shared.append(i)
        except Exception:
            pass
    nets = []
    for k, f in enumerate(brep.faces()):
        try:
            d = ext.surface_cvs(f.surface())
        except Exception:
            continue
        nets.append({"nu": d["count_u"], "nv": d["count_v"], "f": k,      # f: the face (Face mode picks)
                     "p": [list(p) for p in d["points"]]})
    return {"edges": edges, "nets": nets, "shared": shared}


def write_mesh(obj, brep, context=None):
    chord, angle = tess_settings(context)
    pts, nrm, tris = _tessellate(brep, chord, angle, edge_angle_setting(context), max_edge_setting(context),
                                 natural=is_natural(obj))
    me = obj.data
    me.clear_geometry()
    faces = tris.tolist() if isinstance(tris, np.ndarray) else [list(map(int, t)) for t in tris]
    # a face using one vertex twice (the kernel's mesh of a closed seam can have
    # them) is not kept by Blender: drop it here so the per-corner normals
    # below count the same corners as the mesh
    faces = [f for f in faces if len(set(f)) == len(f)]
    me.from_pydata(pts.tolist(), [], faces)
    me.update()
    try:
        me.shade_smooth()
        loop_n = _corner_normals(brep, pts, nrm, faces)
        if loop_n is None or len(loop_n) != len(me.loops):
            # (a wrong-length list crashes Blender inside normals_split_custom_set)
            me.normals_split_custom_set_from_vertices(nrm.tolist())
        else:
            me.normals_split_custom_set(loop_n)
    except Exception as exc:
        print("NURBS normals:", exc)
    me.update()


CREASE_DEG = 30.0


def _corner_normals(brep, pts, nrm, tris):
    """Per-corner normals that keep B-rep creases sharp. The tessellator shares
    one vertex (with ONE averaged normal) between faces meeting at an edge, so
    a 90 degree cap rim would shade as if rounded. At such crease vertices each
    triangle gets the exact normal of its own face. None = no creases."""
    if brep.face_count() < 2 or not len(tris):
        return None
    P = np.asarray(pts, float)
    T = [list(map(int, t)) for t in tris]
    tri_n = []
    for t in T:
        n = np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])
        tri_n.append(n / (np.linalg.norm(n) or 1.0))
    tri_n = np.array(tri_n)
    inc = [[] for _ in range(len(P))]
    for k, t in enumerate(T):
        for v in t:
            inc[v].append(k)
    cos_c = np.cos(np.radians(CREASE_DEG))
    crease = np.zeros(len(P), bool)
    for v, ks in enumerate(inc):
        if len(ks) > 1:
            N_ = tri_n[ks]
            if (N_ @ N_.T).min() < cos_c:
                crease[v] = True
    if not crease.any():
        return None
    from .square import _foot
    faces = brep.faces()
    surf_cache = {}
    face_of = {}
    loops = []
    for k, t in enumerate(T):
        if any(crease[v] for v in t) and k not in face_of:
            c = P[t].mean(axis=0)
            face_of[k] = min(range(len(faces)), key=lambda i: faces[i].closest_point(tuple(c))[1])
    for k, t in enumerate(T):
        for v in t:
            if not crease[v]:
                loops.append(tuple(nrm[v]))
                continue
            fi = face_of[k]
            f = faces[fi]
            s = surf_cache.setdefault(fi, f.surface())
            try:
                uv = _foot(s, P[v], None)
                n = np.array(f.outward_normal_at_uv(*uv), float)
                n /= np.linalg.norm(n) or 1.0
                if n @ tri_n[k] < 0:
                    n = -n
            except Exception:
                n = tri_n[k]
            loops.append(tuple(n))
    return loops


def set_geometry(obj, geom, context=None):
    sm, _ = _k()
    obj[PROP_NATURAL] = 0 if isinstance(geom, sm.Brep) else 1
    brep = to_brep(geom)
    obj[PROP_BREP] = brep_to_text(brep)
    obj[PROP_REV] = _new_rev(obj)
    obj[PROP_DISP] = json.dumps(_display_data(brep))
    _cache[obj.session_uid] = (obj[PROP_REV], brep)
    write_mesh(obj, brep, context)


def new_object(context, name, geom, matrix=None):
    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    coll = context.collection or context.scene.collection
    coll.objects.link(obj)
    if matrix is not None:
        obj.matrix_world = matrix
    set_geometry(obj, geom, context)
    return obj


def new_object_text(context, name, text, matrix=None):
    """New NURBS object from kernel ASCII brep text (local space = matrix)."""
    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    coll = context.collection or context.scene.collection
    coll.objects.link(obj)
    if matrix is not None:
        obj.matrix_world = matrix
    set_geometry_text(obj, text, context, keep_untrimmed=False)
    return obj


def display(obj):
    rev = obj.get(PROP_REV, 0)
    hit = _disp_cache.get(obj.session_uid)
    if hit and hit[0] == rev:
        return hit[1]
    try:
        d = json.loads(obj.get(PROP_DISP, "{}"))
    except ValueError:
        d = {}
    _disp_cache[obj.session_uid] = (rev, d)
    return d


def shared_edges(obj):
    """Indices (into display edges / brep.edges()) of edges between two faces."""
    d = display(obj)
    if "shared" not in d:
        d["shared"] = [i for i, e in enumerate(get_brep(obj).edges()) if len(e.faces()) == 2]
    return d["shared"]


def retessellate(obj, context=None):
    write_mesh(obj, get_brep(obj), context)
