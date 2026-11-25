"""Imprint (Project with Output: Edge): curves, edges or surfaces put onto a
surface or solid as edges of its own. Nothing is removed and the body stays
one body (a closed solid stays closed): its faces are only divided, and an
open curve that stops inside a face leaves an edge that ends there.

The imprint itself is Split's (worker.CurveCutter / PullCutter /
SurfaceCutter): the curve swept along the projection direction into a sheet,
and the target imprinted exactly where the sheet meets it. That sheet runs
through the whole part, so a sketch above a box also lands on the bottom and
on the walls it runs across. Which of those cut edges stay is the scope:

  NEAREST  the faces the projection meets first, seen from the curve (a
           point of a cut edge is hidden when the line from the curve to
           it passes through the part before it). Plasticity's "Hide
           occlusion". Curves that lie on the part (cut along themselves)
           and cutting surfaces keep every edge.
  FACES    only on the picked faces (Shapr3D's "Project Onto").
  THROUGH  every face the projection crosses.

The edges that do not stay are taken off again by worker.heal_cut_seams (the
faces on either side re-trimmed to their union), with the kept edges counted
as original edges so they stay. The vertices a kept edge put on a face's
border stay on the neighbour too: a line across a box top splits the top
edges of the walls it meets.

Pure functions on Breps (no bpy); worker.job_imprint calls run()."""


def _cut_edges(W, b, original, sheets, tol):
    """The edges of b that the imprint made: on a cutter sheet and not along
    an edge of the original. (A sheet test first: an edge that is on no sheet
    is old, and the costly original-edge test runs only on the few that are.)"""
    np = W._np()
    on_orig = W._near_edges(original, 10 * tol)
    out = []
    for e in b.edges():
        pts = W._edge_poly(e, 9)[1:-1]
        if not pts:
            continue
        m = tuple(map(float, pts[len(pts) // 2]))
        try:
            near = min(s.closest_point(m)[1] if hasattr(s, "closest_point") else
                       min(f.closest_point(m)[1] for f in s.faces()) for s in sheets)
        except Exception:
            near = 0.0
        if near > 100 * tol:
            continue
        if all(on_orig(np.array(p)) for p in pts):
            continue
        out.append(e)
    return out


def _face_origin(W, f, ofaces, tol):
    """Index of the original face that face f (a piece of it) lies on."""
    np = W._np()
    try:
        p = tuple(map(float, W._inner_point(f)))
    except Exception:
        return None
    ds = [of.closest_point(p)[1] for of in ofaces]
    j = int(np.argmin(ds))
    return j if ds[j] <= 100 * tol else None


class _Sight:
    """Is a point of a cut edge seen from its curve? The line from the curve
    point above it (along the projection) to it must not meet the part
    before it does."""

    def __init__(self, W, cutter, original, tol):
        np = W._np()
        self.W, self.np, self.tol = W, np, tol
        self.original = original
        self.near = 100 * tol
        self.d = np.array(cutter.dir, float)
        a, b = cutter.basis
        self.a, self.b = np.array(a, float), np.array(b, float)
        self.P = np.array(W._poly(W.curve(cutter.data), 400), float)
        self.Q = np.stack([self.P @ self.a, self.P @ self.b], axis=1)

    def owns(self, pts):
        """Do these points lie on this cutter's sheet (in the plane across it)?"""
        np = self.np
        return max(self._base(np.asarray(p, float))[1] for p in pts) <= 1e3 * self.tol

    def _base(self, m):
        """(the curve point whose projection line runs through m, the
        across-distance of m from that line)."""
        np = self.np
        q = np.array([m @ self.a, m @ self.b])
        A, B = self.Q[:-1], self.Q[1:]
        s = B - A
        t = np.clip(np.einsum("ij,ij->i", q - A, s) / np.maximum(np.einsum("ij,ij->i", s, s), 1e-30), 0, 1)
        X = A + s * t[:, None]
        dist = np.linalg.norm(X - q, axis=1)
        i = int(np.argmin(dist))
        c = self.P[i] + (self.P[i + 1] - self.P[i]) * t[i]
        c = m - float((m - c) @ self.d) * self.d          # on m's own projection line
        return c, float(dist[i])

    def depth(self, m):
        """How far m lies from its curve along the projection direction (signed)."""
        c, _ = self._base(m)
        return float((m - c) @ self.d)

    def seen(self, m):
        np, W, sm, ext = self.np, self.W, self.W.sm, self.W.ext
        c, _ = self._base(m)
        L = float(np.linalg.norm(m - c))
        if L <= 10 * self.tol:
            return True                    # the curve lies right there
        seg = ext.curve_from_cvs(1, [tuple(map(float, c)), tuple(map(float, m))], [0.0, 0.0, 1.0, 1.0], None)
        try:
            curves, points = sm.intersect_curve_brep(seg, self.original)
        except Exception:
            return True
        near = self.near                  # m's own face: the kernel's hit there is only this close
        for p in points:
            if float(np.linalg.norm(np.asarray(tuple(p), float) - m)) > near:
                return False
        for cv in curves:                  # the line runs along a face (a wall along the projection)
            Q = np.asarray(W._poly(cv, 9), float)
            if float(np.max(np.linalg.norm(Q - m, axis=1))) > near:
                return False
        return True


def _keep_nearest(W, cut, cutters, original, tol):
    """The cut edges the projection meets first (see _Sight): an edge stays
    when most of its points are seen from its curve. Edges of curves that lie
    on the part, and of cutting surfaces, all stay."""
    sights = [_Sight(W, c, original, tol) for c in cutters if isinstance(c, W.CurveCutter)
              and not isinstance(c, W.PullCutter)]
    keep = []
    for e in cut:
        pts = [W._np().array(p, float) for p in W._edge_poly(e, 11)[1:-1]]
        own = next((s for s in sights if s.owns(pts)), None)
        if own is None:
            keep.append(e)
            continue
        seen = sum(1 for p in pts if own.seen(p))
        if 2 * seen >= len(pts):
            keep.append(e)
    return keep


def _flat_sketches(W, job, original, tol):
    """A sketch lying on flat faces it was drawn on (its "dir" their normal)
    is imprinted along that normal, not along itself: the same edges, and an
    extension past the face's edges (Extend to Edges) stays well defined (a
    sheet along the surface normals has no normal off the face)."""
    np = W._np()
    sm = W.sm
    faces = original.faces()
    for c in job.get("curves", []):
        if c.get("pull") != "auto" or not c.get("dir"):
            continue
        if not W._pulled(original, c):
            continue
        d = np.array(c["dir"], float)
        d /= np.linalg.norm(d) or 1.0
        flat = True
        for p in np.asarray(W._poly(W.curve(c), 24), float):
            q = tuple(map(float, p))
            f = min(faces, key=lambda g: g.closest_point(q)[1])
            if not f.surface().is_planar():
                flat = False
                break
            _, uvs, _ = sm.surface_closest_point(f.surface(), q)
            n = np.array(f.outward_normal_at_uv(*W._uvxy(uvs[0])), float)
            if abs(float(n @ d)) < 1.0 - 1e-6 * (np.linalg.norm(n) or 1.0):
                flat = False
                break
        if flat:
            c["pull"] = False


def _on_faces(W, data, original, tol):
    """Indices of the original faces a curve (cut along itself) lies on: a
    face counts when more than an end point of the curve is on it."""
    np = W._np()
    faces = original.faces()
    P = np.asarray(W._poly(W.curve(data), 64), float)
    D = np.array([[f.closest_point(tuple(map(float, p)))[1] for f in faces] for p in P])
    on = D <= D.min(axis=1)[:, None] + 10 * tol
    need = max(2, int(0.05 * len(P)))
    return {j for j in range(len(faces)) if int(on[:, j].sum()) >= need}


def _pull_kept(W, cut, cutters, job, original, sheets_of, tol):
    """The cut edges of curves cut along themselves that lie on the faces the
    curve lies on (not where its sheet stood over an edge into a wall)."""
    ofaces = original.faces()
    curves = job.get("curves", [])
    owned = []
    for i, c in enumerate(cutters):
        if isinstance(c, W.PullCutter) and i < len(curves):
            owned.append((sheets_of[i], _on_faces(W, curves[i], original, tol)))
    if not owned:
        return cut
    out = []
    for e in cut:
        pts = W._edge_poly(e, 9)[1:-1]
        m = tuple(map(float, pts[len(pts) // 2])) if pts else None
        mine = None
        for sh, on in owned:
            if m is not None and min(min(f.closest_point(m)[1] for f in s.faces()) for s in sh) <= 100 * tol:
                mine = on
                break
        if mine is None or any(_face_origin(W, f, ofaces, tol) in mine for f in e.faces()):
            out.append(e)
    return out


def run(job, W):
    """job: the target brep, the cutters as for Split (curves with their
    direction, "pull" curves, cutter_breps), "scope" (NEAREST / FACES /
    THROUGH), "faces" (world points on the faces to imprint, FACES), "extend"
    (open curves that stop inside a face extended to its edges). Writes
    out["result"] (target local space)."""
    np = W._np()
    sm, ext = W.sm, W.ext
    original = W.world_brep(job, "target")
    size = W._target_size(original)
    tol = 1e-5 * size
    _flat_sketches(W, job, original, tol)
    cutters = W.build_cutters(job, strict=False)
    if not cutters:
        raise RuntimeError("nothing crosses this surface")
    lo, hi = original.bounding_box()
    ref = tuple(0.5 * (lo[i] + hi[i]) for i in range(3))
    sheets_of = [c.sheet_list(original) for c in cutters]
    sheets = [s for sh in sheets_of for s in sh]
    b = original.copy()
    for c in cutters:
        nb = c.split(b, ref)
        if nb is None or nb.face_count() == 0:
            raise RuntimeError("the imprint failed (does the curve reach the surface?)")
        b = nb
    cut_all = _cut_edges(W, b, original, sheets, tol)
    if not cut_all:
        raise RuntimeError("nothing was imprinted: the curves miss the surface along the projection")
    made = len(cut_all)
    cut = _pull_kept(W, cut_all, cutters, job, original, sheets_of, tol)

    scope = job.get("scope", "NEAREST")
    if scope == "FACES" and not job.get("faces"):
        raise RuntimeError("Picked Faces: no faces picked (pick them on the target in Face mode)")
    if scope == "FACES":
        ofaces = original.faces()
        picked = set()
        for p in job["faces"]:
            ds = [f.closest_point(tuple(map(float, p)))[1] for f in ofaces]
            picked.add(int(np.argmin(ds)))
        keep = [e for e in cut if any(_face_origin(W, f, ofaces, tol) in picked for f in e.faces())]
    elif scope == "NEAREST":
        keep = _keep_nearest(W, cut, cutters, original, tol)
    else:
        keep = list(cut)
    if not keep:
        raise RuntimeError("nothing was imprinted on " + ("the picked faces" if scope == "FACES" else
                                                           "the faces nearest the curves"))
    healed, result, n = True, b, len(keep)
    if len(keep) < made:
        dropped = [e for e in cut_all if not any(e == k for k in keep)]
        again = _second_pass(W, original, cutters, sheets_of, keep, dropped, size, tol)
        if again is not None:
            result, n, healed = again
        else:                         # the others taken off again; the kept ones count as original
            h = W.heal_cut_seams(b.copy(), original, keep=[W._edge_poly(e, 400) for e in keep])
            left = _cut_edges(W, h, original, sheets, tol)
            healed = len(left) == len(keep)
            if len(left) >= len(keep):
                result = h
    try:
        solid = bool(result.is_manifold_solid())
    except Exception:
        solid = False
    W.save(W.to_local(result, job.get("xforms", {}).get("target")), job["out"]["result"])
    return {"edges": n, "dropped": max(0, made - n), "healed": healed, "solid": solid,
            "faces": result.face_count(), "extended": job.get("extended", [])}


def _band_sheets(W, cutter, t0, t1):
    """The cutter's sheets only between depths t0 and t1 along its direction."""
    np = W._np()
    sm, ext = W.sm, W.ext
    d = np.array(cutter.dir, float)
    out = []
    for piece in W.corner_pieces(cutter.data):
        k = W.curve(piece)
        if abs(t0) > 0:
            k = sm.translate(k, tuple(float(t0) * d))
        out.append(ext.brep_from_surface(sm.create_extrude_surface(k, tuple(float(t1 - t0) * d))))
    return out


def _second_pass(W, original, cutters, sheets_of, keep, dropped, size, tol):
    """The imprint again with each projected curve's sheet cut back to the
    depths where its kept edges are (the far side of the part, or the back of
    a tube, never touched: a cut there across a closed surface's seam can not
    be healed away). What it still makes beyond the kept edges (a sliver down
    a wall the projection runs along) is healed away. A curve none of whose
    edges stay is left out. Returns (brep, kept edges, healed?) or None."""
    np = W._np()
    sm = W.sm

    def pts(e, n=33):
        return [np.asarray(p, float) for p in W._edge_poly(e, n)]
    sheets2 = []
    for i, c in enumerate(cutters):
        if not isinstance(c, W.CurveCutter):
            sheets2 += sheets_of[i]
            continue
        s = _Sight(W, c, original, tol)
        mk = [e for e in keep if s.owns(pts(e, 9)[1:-1])]
        md = [e for e in dropped if s.owns(pts(e, 9)[1:-1])]
        if not mk:
            continue
        if not md:
            sheets2 += sheets_of[i]
            continue
        tk = [s.depth(p) for e in mk for p in pts(e)]
        td = [s.depth(p) for e in md for p in pts(e)]
        t0, t1 = min(tk), max(tk)
        gaps = [t0 - t for t in td if t < t0 - 100 * tol] + [t - t1 for t in td if t > t1 + 100 * tol]
        m = max(100 * tol, min([0.01 * size] + [0.25 * g for g in gaps]))
        sheets2 += _band_sheets(W, c, t0 - m, t1 + m)
    try:
        b = original.copy()
        for sh in sheets2:
            b = sm.boolean_with_options(b, sh.copy(), operation=sm.BooleanOp.UNION, imprinting=True,
                                        keep_other_brep=True)
            if b is None or b.face_count() == 0:
                return None
        cut = _cut_edges(W, b, original, sheets2, tol)
    except Exception:
        return None
    near = W._near_polys([W._edge_poly(e, 400) for e in keep], 100 * tol)
    kept = []
    for e in cut:
        P = pts(e, 11)[1:-1]
        if P and 2 * sum(1 for p in P if near(p)) >= len(P):
            kept.append(e)
    if len(kept) < len(keep):
        return None
    if len(kept) == len(cut):
        return b, len(kept), True
    h = W.heal_cut_seams(b.copy(), original, keep=[W._edge_poly(e, 400) for e in kept])
    left = _cut_edges(W, h, original, sheets2, tol)
    if len(left) < len(kept):
        return None
    return (h, len(kept), True) if len(left) == len(kept) else (b, len(kept), False)
