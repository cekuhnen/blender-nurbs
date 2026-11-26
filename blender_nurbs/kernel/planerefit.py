"""Remove Edges with a Flat Tolerance: flat faces that are almost on one plane
(within an angle) merged onto one plane fitted through them. Unlike the exact
merge (mergefaces) the shape changes a little: corners move onto the fitted
planes, and the faces round them follow.

A cluster: flat faces joined by the edges to remove (each between two flat
faces facing almost the same way). Its plane: through the cluster's corners
(least squares), facing the way of its area-weighted normal. Every other face
keeps its plane. All clusters are fitted together, then every corner that
touches a cluster is put where the planes round it meet: three or more
different planes -- the point on all of them (least squares, and it must be
one point: four faces round a corner that no longer meet in one are refused);
two -- the nearest point of the line where they cross (a corner on a straight
run of the outline; the edges either side are joined after); one -- a corner
inside a cluster, gone.

Then each cluster is trimmed once on a patch of its plane, and every face
round it on a new patch of its own plane, with straight edges between the
moved corners (other edges as they were). Accepted when the body stitches
closed with the face count it should have and its volume moved no more than
the corners did (area x largest move).

Refused (the edges stay and say why): a curved face at a moved corner (its
surface would have to change), a curved edge ending at one, corners that no
longer meet in one point, a move larger than MAX_MOVE of the size.
"""

import math

import numpy as np

try:
    from . import mergefaces as mf
    from .trimext import brep_tol
except ImportError:          # the worker imports the kernel folder's modules top-level
    import mergefaces as mf
    from trimext import brep_tol

MAX_MOVE = 0.02              # a share of the body's size: a refit that moves corners further is refused


def near_coplanar(body, e, flat_deg):
    """The angle (degrees) between the two faces of edge e when both are flat,
    face the same way and lie within flat_deg of one plane; else None."""
    ij = body.edge_faces(e)
    if ij is None or flat_deg <= 0:
        return None
    pi, pj = body.plane(ij[0]), body.plane(ij[1])
    if pi is None or pj is None:
        return None
    cos = float(pi[1] @ pj[1])
    if cos <= 0:
        return None
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return ang if ang <= flat_deg else None


def _straight(e, tol):
    a, z = mf._edge_pts(e, 2)
    L = np.linalg.norm(z - a)
    if L <= tol:
        return False
    d = (z - a) / L
    for q in mf._edge_pts(e, 7)[1:-1]:
        off = q - a
        if np.linalg.norm(off - (off @ d) * d) > tol:
            return False
    return True


def _fit(points, normal_hint):
    P = np.asarray(points, float)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[2]
    if n @ normal_hint < 0:
        n = -n
    return c, n / np.linalg.norm(n)


def _meet(planes, p, tol):
    """Where the planes [(point, normal)] meet, nearest p: (point, why)."""
    N = np.array([n for _, n in planes], float)
    d = np.array([n @ c for c, n in planes], float)
    # distinct directions: two planes meeting in a line, or more meeting in a point
    U, s, Vt = np.linalg.svd(N)
    rank = int(np.sum(s > 1e-6 * s[0]))
    if rank < 2:
        return None, "the faces round a corner are parallel"
    if rank == 2:
        # the line: the nearest point to p of {x: N x = d}
        x0 = np.linalg.lstsq(N, d, rcond=None)[0]
        line = Vt[2]
        x = x0 + ((p - x0) @ line) * line
    else:
        x = np.linalg.lstsq(N, d, rcond=None)[0]
    if np.abs(N @ x - d).max() > 10 * tol:
        return None, "the faces round a corner no longer meet in one point"
    return x, ""


def _refit_all(sm, b, go):
    """b with the clusters joined by edges `go` (indices into b.edges())
    merged onto fitted planes: (brep, largest move, edges removed, moved
    corners) or (None, why, 0, [])."""
    body = mf._Body(sm, b)
    tol = body.tol
    faces, edges = body.faces, b.edges()
    ext = sm.blender_ext
    key = lambda p: mf._key(p, tol)  # noqa: E731
    clusters, gone = mf._clusters(body, go)
    if not clusters:
        return None, "nothing to merge", 0, []
    group = {i: ci for ci, g in enumerate(clusters) for i in g}       # face -> cluster
    # each final face's plane: a cluster's fitted one, else the face's own
    corner_pts, vert_faces, vert_edges = {}, {}, {}
    for k, e in enumerate(edges):
        a, z = mf._edge_pts(e, 2)
        fs = [mf._index(faces, g) for g in e.faces()]
        for p in (a, z):
            v = key(p)
            corner_pts[v] = p
            vert_edges.setdefault(v, set()).add(k)
            vert_faces.setdefault(v, set()).update(fs)
    plane_of = {}
    for ci, g in enumerate(clusters):
        pts = [corner_pts[key(p)] for i in g for e in faces[i].edges() for p in mf._edge_pts(e, 2)]
        area_n = sum(faces[i].area(1e-6) * body.plane(i)[1] for i in g)
        plane_of[("c", ci)] = _fit(pts, area_n)
    final = lambda i: ("c", group[i]) if i in group else ("f", i)  # noqa: E731
    # the corners that move: every corner of a cluster face
    touched = {key(p) for i in group for e in faces[i].edges() for p in mf._edge_pts(e, 2)}
    moved = {}
    for v in touched:
        ids = {final(i) for i in vert_faces[v] if i >= 0}
        planes = []
        for fid in sorted(ids, key=str):
            if fid[0] == "f":
                pl = body.plane(fid[1])
                if pl is None:
                    return None, "a curved face meets it at a corner (its surface would have to change)", 0, []
                plane_of[fid] = pl
            planes.append(plane_of[fid])
        if len(planes) == 1:
            continue                                   # inside a cluster: the corner goes
        x, why = _meet(planes, corner_pts[v], tol)
        if x is None:
            return None, why, 0, []
        moved[v] = x
    move = max((float(np.linalg.norm(moved[v] - corner_pts[v])) for v in moved), default=0.0)
    if move > MAX_MOVE * body.size:
        return None, f"it would move corners by {move:.3g} (more than {MAX_MOVE:.0%} of the size)", 0, []
    gone_keys = {mf._key(mf._edge_mid(edges[k]), tol) for k in gone}

    def piece(e, fw):
        """(curve, start, end) of edge e in loop order, its corners moved."""
        a, z = mf._edge_pts(e, 2)
        if not fw:
            a, z = z, a
        ka, kz = key(a), key(z)
        if ka not in moved and kz not in moved:
            return mf._ecurve(e, fw), a, z
        if not _straight(e, tol):
            raise ValueError("a curved edge ends at a moved corner")
        a2, z2 = moved.get(ka, a), moved.get(kz, z)
        if np.linalg.norm(z2 - a2) <= tol:
            raise ValueError("an edge would shrink to nothing")
        return ext.curve_from_cvs(1, [tuple(map(float, a2)), tuple(map(float, z2))],
                                  [0.0, 0.0, 1.0, 1.0], None), a2, z2

    btol = brep_tol(mf.REL_TOL, body.size)
    # the final faces that change: the clusters, and every face at a moved corner
    redo = {("c", ci) for ci in range(len(clusters))}
    for v in moved:
        redo |= {final(i) for i in vert_faces[v] if i >= 0}
    new, drop = [], []
    try:
        for fid in sorted(redo, key=str):
            members = clusters[fid[1]] if fid[0] == "c" else [fid[1]]
            c, n = plane_of[fid] if fid in plane_of else body.plane(fid[1])
            if fid[0] == "f":                         # one face: its loops as they are
                loops, P = [], []
                for loop in faces[fid[1]].boundary_loops():
                    cvs = []
                    for e, fw in loop:
                        cv, a2, z2 = piece(e, fw)
                        cvs.append(cv)
                        P += [a2, z2]
                    loops.append(cvs)
            else:                                     # a cluster: its outline, chained by the moved corners
                segs, curves = [], []
                for i in members:
                    for loop in faces[i].boundary_loops():
                        for e, fw in loop:
                            if mf._key(mf._edge_mid(e), tol) in gone_keys:
                                continue
                            cv, a2, z2 = piece(e, fw)
                            d = (z2 - a2) / (np.linalg.norm(z2 - a2) or 1.0)
                            segs.append((None, fw, a2, z2, d, d))
                            curves.append(cv)
                chained = mf._chain_runs([[j] for j in range(len(segs))], segs, tol)
                if chained is None:
                    raise ValueError("a merged face's outline did not close")
                loops = [[curves[j] for j in lp] for lp in chained]
                P = [s[2] for s in segs]
            m = mf._trim(sm, mf._new_plane(ext, n, P, tol), loops, None, tol, btol)
            if m is None:
                raise ValueError("a face could not be trimmed again")
            new.append(m)
            drop += members
    except ValueError as exc:
        return None, str(exc), 0, []
    area = sum(faces[i].area(1e-6) for i in group)
    out = mf._replace(sm, b, drop, new, tol, len(faces) - len(drop) + len(new),
                      volume_change=2.0 * area * move + 1e-9)
    if out is None:
        return None, "the refitted faces did not join up with the body", 0, []
    return out, move, len(gone), [[float(x) for x in q] for q in moved.values()]


def refit(sm, b, go_pts, flat_deg):
    """Merge onto fitted planes the clusters joined by the edges at go_pts
    (points on near-coplanar edges): all together; failing that, cluster by
    cluster. Returns (brep or None, report): "removed" (edges), "moved"
    (largest corner move), "failed" [{point, why}]."""
    rep = {"removed": 0, "moved": 0.0, "failed": [], "corners": []}
    body = mf._Body(sm, b)
    edges = b.edges()
    hit, _ = mf._picked_edges(b, edges, go_pts, body.size)
    go = [k for k in hit if near_coplanar(body, edges[k], flat_deg) is not None]
    if not go:
        return None, rep
    out, res, n, corners = _refit_all(sm, b, go)
    if out is not None:
        rep.update(removed=n, moved=res, corners=corners)
        return out, rep
    # one cluster at a time (found again by a point on its edges on each result)
    clusters, _ = mf._clusters(body, go)
    groups = [[mf._edge_mid(edges[k]) for k in go if body.edge_faces(edges[k])[0] in g] for g in clusters]
    cur = b
    for pts in groups:
        cb = mf._Body(sm, cur)
        ce = cur.edges()
        h, _ = mf._picked_edges(cur, ce, pts, cb.size)
        g2 = [k for k in h if near_coplanar(cb, ce[k], flat_deg) is not None]
        o, r, n, corners = _refit_all(sm, cur, g2) if g2 else (None, "its edges were not found again", 0, [])
        if o is None:
            rep["failed"].append({"point": [float(x) for x in pts[0]], "why": r})
            continue
        cur = o
        rep["removed"] += n
        rep["moved"] = max(rep["moved"], r)
        rep["corners"] += corners
    return (cur if cur is not b else None), rep
