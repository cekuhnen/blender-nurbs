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
    result = _merge_seam_splits(W, result, original, tol)
    try:
        solid = bool(result.is_manifold_solid())
    except Exception:
        solid = False
    W.save(W.to_local(result, job.get("xforms", {}).get("target")), job["out"]["result"])
    return {"edges": n, "dropped": max(0, made - n), "healed": healed, "solid": solid,
            "faces": result.face_count(), "extended": job.get("extended", [])}


def _seam_dir(W, srf, pts):
    """0 (u) or 1 (v): the closed direction of srf whose seam the points
    lie on; None when they are on no seam."""
    np = W._np()
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    uv = []
    for p in pts:
        try:
            _, uvs, _ = W.sm.surface_closest_point(srf, tuple(map(float, p)))
            uv.append(W._uvxy(uvs[0]))
        except Exception:
            return None
    uv = np.array(uv, float)
    for k, (a, b) in enumerate(((u0, u1), (v0, v1))):
        span = (b - a) or 1.0
        x = (uv[:, k] - a) / span
        if np.all(np.minimum(np.abs(x), np.abs(1.0 - x)) <= 1e-6):
            return k
    return None


def _free_frac(W, srf, k, pts):
    """A place for the seam in direction k (0..1 of its range) as far as
    possible from the points: the middle of the widest gap between them."""
    np = W._np()
    a, b = (srf.u_domain(), srf.v_domain())[k]
    xs = []
    for p in pts:
        _, uvs, _ = W.sm.surface_closest_point(srf, tuple(map(float, p)))
        xs.append(((W._uvxy(uvs[0])[k] - a) / ((b - a) or 1.0)) % 1.0)
    xs = np.sort(np.array(xs, float))
    gaps = np.diff(np.concatenate([xs, [xs[0] + 1.0]]))
    i = int(np.argmax(gaps))
    if gaps[i] < 0.05:
        return None                          # the region goes all the way round
    return float((xs[i] + 0.5 * gaps[i]) % 1.0)


def _merge_seam_splits(W, b, original, tol):
    """One face again where the imprint left two on either side of a closed
    surface's seam (a groove outline across a cylinder's seam): the kernel's
    trims can not cross a seam, so they are trimmed again as one face on the
    same surface with its seam moved away from them (worker._reseamed, exact),
    and stitched back in. A region that goes all the way round keeps the seam.
    Returns b unchanged when anything does not check out."""
    np = W._np()
    sm, ext = W.sm, W.ext
    faces = b.faces()
    ofaces = original.faces()
    on_orig = W._near_edges(original, 10 * tol)
    origin = [_face_origin(W, f, ofaces, tol) for f in faces]

    def key(e):
        m = W._edge_mid(e)
        return None if m is None else tuple(round(float(v) / tol) for v in m)
    uses, edge_of = {}, {}
    for i, f in enumerate(faces):
        for loop in f.boundary_loops():
            for e, fw in loop:
                k = key(e)
                if k is None:
                    return b
                uses.setdefault(k, []).append((i, fw))
                edge_of[k] = e
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seams = set()
    for k, us in uses.items():
        if len(us) != 2 or us[0][0] == us[1][0]:
            continue
        i, j = us[0][0], us[1][0]
        if origin[i] is None or origin[i] != origin[j]:
            continue
        pts = [np.asarray(p, float) for p in W._edge_poly(edge_of[k], 7)[1:-1]]
        if not pts or not all(on_orig(p) for p in pts):
            continue                                       # a cut edge, not a piece of the seam
        if _seam_dir(W, ofaces[origin[i]].surface(), pts) is None:
            continue
        seams.add(k)
        parent[find(i)] = find(j)
    if not seams:
        return b
    clusters = {}
    for i in range(len(faces)):
        clusters.setdefault(find(i), []).append(i)
    out = b.copy()
    drop, made = [], []
    for members in clusters.values():
        if len(members) < 2:
            continue
        segs = []                                # (edge, forward, start, end): the union's boundary
        for i in members:
            for loop in faces[i].boundary_loops():
                for e, fw in loop:
                    if key(e) in seams:
                        continue
                    P = W._edge_poly(e, 3)
                    a, z = (np.array(P[0]), np.array(P[-1])) if fw else (np.array(P[-1]), np.array(P[0]))
                    segs.append((e, fw, a, z))
        loops, todo = [], list(range(len(segs)))
        while todo:
            chain = [todo.pop(0)]
            while np.linalg.norm(segs[chain[-1]][3] - segs[chain[0]][2]) > 100 * tol:
                end = segs[chain[-1]][3]
                nxt = min(todo, key=lambda j: np.linalg.norm(segs[j][2] - end), default=None)
                if nxt is None or np.linalg.norm(segs[nxt][2] - end) > 100 * tol:
                    return b
                todo.remove(nxt)
                chain.append(nxt)
            loops.append(chain)
        srf0 = ofaces[origin[members[0]]].surface()
        seam_pts = [np.asarray(p, float) for k in seams if uses[k][0][0] in members
                    for p in W._edge_poly(edge_of[k], 5)[1:-1]]
        kdir = _seam_dir(W, srf0, seam_pts)
        bpts = [p for j in range(len(segs)) for p in W._edge_poly(segs[j][0], 9)]
        frac = None if kdir is None else _free_frac(W, srf0, kdir, bpts)
        if frac is None:
            continue
        area = sum(faces[i].area(1e-6) for i in members)

        def curves_of(lp, flip):
            cs = []
            for j in (reversed(lp) if flip else lp):
                e, fw = segs[j][0], segs[j][1]
                d = ext.curve_cvs(e.curve())
                U, Pc, Wt = list(d["knots"]), list(d["points"]), d.get("weights")
                if fw == flip:
                    a0, a1 = U[0], U[-1]
                    U = [a0 + a1 - u for u in reversed(U)]
                    Pc, Wt = Pc[::-1], (list(Wt)[::-1] if Wt else None)
                cs.append(ext.curve_from_cvs(d["degree"], Pc, U, list(Wt) if Wt else None))
            return cs
        face = None
        for flip in (False, True):
            G = W._reseamed(ext.surface_from_cvs(*W._srf_tuple(ext.surface_cvs(srf0))), "uv"[kdir], frac)
            cs = [curves_of(lp, flip) for lp in loops]
            try:
                cand = ext.trim_surface_with_3d_curves_tol(G, [len(c) for c in cs], [c for l in cs for c in l],
                                                           10 * tol, 100 * tol)
            except Exception:
                continue
            if cand.face_count() == 1 and abs(cand.area(1e-6) - area) <= 1e-4 * max(area, 1e-12):
                face = cand
                break
        if face is None:
            continue
        drop += members
        made.append(face)
    if not made:
        return b
    of = out.faces()
    ext.delete_faces(out, [of[i] for i in drop])
    for m in made:
        ext.append_brep(out, m)
    try:
        solid = bool(b.is_manifold_solid())
    except Exception:
        solid = False
    want = len(faces) - len(drop) + len(made)
    for t in (tol, 10 * tol, 100 * tol):
        st = sm.stitch_advanced(out.copy(), t, squeeze_small_edges=False, making_manifold_solid=solid)[0]
        if st.face_count() != want:
            continue
        try:
            ok = bool(st.is_manifold_solid()) if solid else W._open_edge_count(st) <= W._open_edge_count(b)
        except Exception:
            ok = False
        if ok and abs(st.area(1e-6) - b.area(1e-6)) <= 1e-4 * max(b.area(1e-6), 1e-12):
            return st
    return b


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
