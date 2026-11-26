"""Remove Faces (heal): faces of a solid taken away and the gap closed by
extending the faces round them until they meet -- Fusion's Delete Face,
NX's Delete Face, Plasticity's Delete Face with heal. A hole's wall goes and
the plate is whole again; a boss (wall and top) goes and the face under it
closes; a chamfer or a sliver face goes and its neighbours meet in a sharp
edge. Unlike Delete Faces the body stays closed.

The picked faces are grouped by the edges they share (a hole's wall and its
floor are one group: the kernel removes a group as one feature). All groups
go in one defeature call (the kernel's SmFeatureExecutive, the faces round
each group extended until they close it); when that fails, group by group,
and a single face also by DeleteFilletFace (it catches a sliver between two
faces the feature solver gives up on). Each try runs with the kernel's own
extension distance first, then longer ones: its own is too short where the
faces round the gap meet at a shallow angle.

A result counts only when it is a valid closed solid (if the body was one),
has fewer faces, and no picked face is left on it (the kernel can come back
with some of them still there). A group that can't be closed -- a box's top:
its walls run on upward and never meet -- is reported and left; Delete
Faces (X) is the way to open the body there.

Kernel and numpy only (no bpy): the worker's remove_faces job runs remove().
"""

import numpy as np

REACH = (0.0, 0.02, 0.1, 0.5, 2.0)    # extension distances to try: the kernel's own, then shares of the size


def has_kernel(sm):
    ext = getattr(sm, "blender_ext", None)
    return ext is not None and all(hasattr(ext, n) for n in ("defeature_faces", "delete_fillet_face"))


def _size(b):
    lo, hi = b.bounding_box()
    return float(np.linalg.norm(np.array(hi, float) - np.array(lo, float))) or 1.0


def _point(f):
    p = f.interior_point()
    return np.array([p.x, p.y, p.z], float) if hasattr(p, "x") else np.array([float(x) for x in p])


def _face_near(b, p, tol):
    q = tuple(float(x) for x in p)
    best, bd = None, None
    for f in b.faces():
        d = f.closest_point(q)[1]
        if bd is None or d < bd:
            best, bd = f, d
    return best if bd is not None and bd <= tol else None


def _index(faces, f):
    for k, g in enumerate(faces):
        if g == f:
            return k
    return -1


def _groups(faces, idx):
    """The picked faces (indices) in groups joined by shared edges."""
    idx = list(dict.fromkeys(idx))
    pick = set(idx)
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in idx:
        for e in faces[i].edges():
            for g in e.faces():
                j = _index(faces, g)
                if j in pick and j != i:
                    parent[find(i)] = find(j)
    out = {}
    for i in idx:
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _ok(before, after, gone, tol):
    """Does after count as before with the faces at `gone` (points) removed?"""
    if after is None or after.face_count() == 0 or after.face_count() >= before.face_count():
        return False
    if before.is_manifold_solid() and not after.is_manifold_solid():
        return False
    return not any(_face_near(after, p, tol) is not None for p in gone)


def _defeature(sm, b, groups_pts, size, tol):
    """b with each group of faces (given by a point on each face) removed by
    the kernel's defeature, trying each extension distance; (brep, "") or
    (None, why)."""
    why = "the faces round it could not be extended to meet"
    gone = [p for g in groups_pts for p in g]
    for r in REACH:
        c = b.copy()
        groups = [[_face_near(c, p, tol) for p in g] for g in groups_pts]
        if any(f is None for g in groups for f in g):
            return None, "a picked face was not found again"
        try:
            sm.blender_ext.defeature_faces(c, groups, r * size)
        except Exception as exc:
            why = str(exc).split(" at ")[0]
            continue
        if _ok(b, c, gone, tol):
            return c, ""
    return None, why


def _delete_one(sm, b, p, tol):
    """b with the face at p removed by DeleteFilletFace (its neighbours run on
    to meet), or None."""
    c = b.copy()
    f = _face_near(c, p, tol)
    if f is None:
        return None
    try:
        sm.blender_ext.delete_fillet_face(c, f)
    except Exception:
        return None
    return c if _ok(b, c, [p], tol) else None


def _enclosed(faces, group):
    """The faces the group closes in: each face all of whose neighbours are
    in the group or already added (a blind hole's floor when only its wall is
    picked, a boss's top). Returns the added indices."""
    have, added = set(group), []
    while True:
        more = []
        for j in range(len(faces)):
            if j in have:
                continue
            nb = {_index(faces, g) for e in faces[j].edges() for g in e.faces()} - {j}
            if nb and nb <= have:
                more.append(j)
        if not more:
            return added
        have |= set(more)
        added += more


def _fillet_faces(sm, b):
    """{face index: point} of the faces Remove Fillets handles (fillets and
    the corner patches between them), or {} without its kernel functions."""
    try:
        from . import defeature
    except ImportError:                  # the worker imports the kernel folder's modules top-level
        import defeature
    if not defeature.has_kernel(sm):
        return {}, None
    try:
        found = defeature.find(sm, b)
    except Exception:
        return {}, None
    faces = b.faces()
    out = {}
    for x in found["fillets"] + found["corners"]:
        out[x["face"]] = _point(faces[x["face"]])
    return out, defeature


def remove(sm, b, picked):
    """Remove the faces of b nearest the picked points (b's space) and heal
    the gaps. Fillets (and rounded corners) among them go by Remove Fillets'
    rules, each with its chain, the chain's other fillets put back: the
    kernel can hang for minutes on part of a rounded corner. Returns (brep
    or None, report): report has "removed" (faces), "groups" (closed),
    "added" (enclosed faces taken along), "fillets" (removed as fillets),
    "failed" [{point, why}], "missed" (picks on no face)."""
    size = _size(b)
    tol = 1e-5 * size
    faces = b.faces()
    idx, miss = [], []
    for k, p in enumerate(picked):
        q = tuple(float(x) for x in p)
        ds = [f.closest_point(q)[1] for f in faces]
        i = int(np.argmin(ds)) if ds else -1
        if i < 0 or ds[i] > 1e-2 * size:
            miss.append(k)
            continue
        idx.append(i)
    rep = {"removed": 0, "groups": 0, "added": 0, "fillets": 0, "failed": [], "missed": miss}
    idx = list(dict.fromkeys(idx))
    if not idx:
        return None, rep
    if len(idx) >= len(faces):
        rep["failed"].append({"point": None, "why": "every face of the body is picked"})
        return None, rep
    # fillets first, by Remove Fillets' rules
    cur = b
    fil, defeature = _fillet_faces(sm, b)
    fil_pts = [fil[i] for i in idx if i in fil]
    rest = [_point(faces[i]) for i in idx if i not in fil]
    if fil_pts:
        r, frep = defeature.remove(sm, b, picked=fil_pts, keep_others=True)
        if r is None:
            why = "; ".join(dict.fromkeys(f["why"] for f in frep.get("failed", []))) or "not removed"
            rep["failed"].append({"point": [float(x) for x in fil_pts[0]], "why": "fillet: " + why})
        else:
            cur = r
            rep["fillets"] = rep["removed"] = len(fil_pts)
            rep["groups"] += 1
    if not rest:
        return (cur if cur is not b else None), rep
    cf = cur.faces()
    ridx = [i for i in (_index(cf, _face_near(cur, p, 10 * tol)) for p in rest) if i >= 0]
    groups = _groups(cf, ridx)
    # all at once
    out, why = _defeature(sm, cur, [[_point(cf[i]) for i in g] for g in groups], size, tol)
    if out is not None:
        rep["removed"] += len(ridx)
        rep["groups"] += len(groups)
        return out, rep
    # group by group (each on the result of the ones before); a group that fails
    # is tried again with the faces it closes in (its floor, its top)
    for g in groups:
        pts = [_point(cf[i]) for i in g]
        nxt, why = _defeature(sm, cur, [pts], size, tol)
        if nxt is None and len(g) == 1:
            nxt = _delete_one(sm, cur, pts[0], tol)
        if nxt is None:
            cfaces = cur.faces()
            gi = [_index(cfaces, _face_near(cur, p, 10 * tol)) for p in pts]
            add = _enclosed(cfaces, gi) if all(i >= 0 for i in gi) else []
            if add and len(gi) + len(add) < len(cfaces):
                nxt, why2 = _defeature(sm, cur, [pts + [_point(cfaces[j]) for j in add]], size, tol)
                if nxt is not None:
                    rep["added"] += len(add)
                    rep["removed"] += len(add)
        if nxt is None:
            rep["failed"].append({"point": [float(x) for x in pts[0]], "why": why})
            continue
        cur = nxt
        rep["removed"] += len(g)
        rep["groups"] += 1
    return (cur if cur is not b else None), rep
