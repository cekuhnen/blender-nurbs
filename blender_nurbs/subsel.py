"""Face / edge selection: Body / Face / Edge modes, as in Plasticity.

Body mode is Blender's own object selection. In Face or Edge mode a click in
the viewport picks a face or edge of a NURBS body whatever tool is active (a
background handler takes the clicks and mouse moves; everything else passes
through): a click picks one (and selects its body), Shift+click adds or
removes one, a click on nothing clears the picks; what is under the mouse
lights up. Face and Edge can be on together (Shift+click the second mode
button): an edge within 10 px of the mouse is picked, else the face under it.
Clicks on the navigation gizmo and the continuity labels go to them. The
picks live on the object (an ID property, so undo keeps them), each with a
point on it, so they still mean the same face or edge after a rebuild
renumbers them; they count while their body is selected. What follows them:

* CVs / Hulls: a body with picked faces shows only those faces' nets.
* Edge Fillet, Continuity Check, Dome take the picked edges (no picking), and
  so do Loft, Planar, Square, Boundary, Patch, Loft Rails, Duplicate Edge,
  Extend Surface, Pipe and Revolve when the picks fit them (Alt+click a tool:
  pick by hand instead).
* Unjoin splits the picked faces off; X / Delete removes the picked faces
  (the body opens), Shift+X / Shift+Delete removes them and heals the gap
  (Remove Faces), or (only edges picked) X / Delete removes the picked edges
  where their faces merge into one (Remove Edges). Those keys are keymap
  items (removeedges.py), so F9 adjusts what they did.
"""

import json

import numpy as np

import bpy
from bpy_extras import view3d_utils
from mathutils import Vector

from . import store
from .kernel import load as _load_kernel

PROP = "nproto_subsel"
HOVER = {"key": None}          # (object name, "FACE" / "EDGE", index) under the mouse
COL_PICK = (1.0, 0.55, 0.1, 0.35)
COL_PICK_LINE = (1.0, 0.55, 0.1, 1.0)
COL_HOVER = (1.0, 0.95, 0.3, 0.22)
COL_HOVER_LINE = (1.0, 0.95, 0.2, 1.0)
COL_STEP = (0.2, 0.85, 1.0, 0.16)          # what the open history step made (as the hidden-input outline)
COL_STEP_LINE = (0.2, 0.85, 1.0, 0.9)

_valid = {}    # session_uid -> (rev, raw, kind, [indices])
_ftris = {}    # (session_uid, rev, face) -> (points, triangles, boundary polylines), local space
_epts = {}     # session_uid -> (rev, [polyline or None])


def kinds(context=None):
    """The sub-object kinds being picked: a subset of {"FACE", "EDGE"}
    (empty in Body mode)."""
    v = getattr((context or bpy.context).window_manager, "nproto_view", None)
    if v is None:
        return frozenset()
    m = v.sel_mode
    m = {m} if isinstance(m, str) else set(m)
    return frozenset(m & {"FACE", "EDGE"})


def mode(context=None):
    """"BODY", "FACE", "EDGE" or "FACE_EDGE" (both on)."""
    k = kinds(context)
    if not k:
        return "BODY"
    return "FACE_EDGE" if len(k) == 2 else next(iter(k))


def _size(obj):
    lo, hi = (np.array(v, float) for v in store.get_brep(obj).bounding_box())
    return float(np.linalg.norm(hi - lo)) or 1.0


def _edge_polys(obj):
    rev = obj.get(store.PROP_REV, 0)
    hit = _epts.get(obj.session_uid)
    if hit and hit[0] == rev:
        return hit[1]
    polys = [np.array(p, float) if len(p) > 1 else None for p in store.display(obj).get("edges", [])]
    _epts[obj.session_uid] = (rev, polys)
    return polys


def _dist_poly(P, q):
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(((q - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - q, axis=1).min())


def _anchor(obj, kind, i):
    """A point on the face / edge (local space): finds it again after a rebuild."""
    if kind == "FACE":
        p = store.get_brep(obj).faces()[i].interior_point()
        return [float(p.x), float(p.y), float(p.z)] if hasattr(p, "x") else [float(x) for x in p]
    P = _edge_polys(obj)[i]
    return [float(x) for x in P[len(P) // 2]]


def _locate(obj, kind, i, anchor, size):
    """Index of the face / edge a stored pick means in the current geometry."""
    q = np.array(anchor, float)
    if kind == "FACE":
        faces = store.get_brep(obj).faces()

        def d(k):
            return faces[k].closest_point(tuple(q))[1]
    else:
        polys = _edge_polys(obj)

        def d(k):
            return _dist_poly(polys[k], q) if polys[k] is not None else np.inf
        faces = polys
    if 0 <= i < len(faces) and d(i) < 1e-4 * size:
        return i
    ds = [d(k) for k in range(len(faces))]
    if not ds:
        return None
    k = int(np.argmin(ds))
    # only the same face / edge, found again after a renumbering (its point on
    # it, up to the tessellation); never a neighbour that took its place
    return k if ds[k] < max(2e-3, 4.0 * store.tess_settings()[0]) * size else None


KINDS = ("FACE", "EDGE")


def _raw(obj):
    """{"FACE": [[i, anchor], ...], "EDGE": [...]} as stored (older files: one
    kind, {"kind", "items"})."""
    try:
        d = json.loads(obj.get(PROP) or "{}")
    except (ValueError, TypeError):
        return {}
    if "kind" in d:
        return {d["kind"]: d.get("items", [])} if d.get("kind") in KINDS else {}
    return {k: d.get(k, []) for k in KINDS if d.get(k)}


def get_all(obj):
    """{"FACE": [indices], "EDGE": [indices]} picked on obj, checked against
    its current geometry; empty lists when nothing is picked or the body is
    not selected."""
    empty = {k: [] for k in KINDS}
    if obj is None or not store.is_nurbs(obj) or not obj.select_get():
        return empty
    raw = obj.get(PROP)
    if not raw:
        return empty
    rev = obj.get(store.PROP_REV, 0)
    hit = _valid.get(obj.session_uid)
    if hit and hit[0] == rev and hit[1] == raw:
        return hit[2]
    out = {k: [] for k in KINDS}
    try:
        size = _size(obj)
        for kind, items in _raw(obj).items():
            for i, a in items:
                j = _locate(obj, kind, int(i), a, size)
                if j is not None and j not in out[kind]:
                    out[kind].append(j)
    except Exception as exc:
        print("NURBS select:", obj.name, exc)
    _valid[obj.session_uid] = (rev, raw, out)
    return out


def get(obj):
    """(kind, [indices]) picked on obj: "FACE" or "EDGE" when one kind is
    picked, "MIXED" (faces then edges) when both, (None, []) when nothing."""
    a = get_all(obj)
    have = [k for k in KINDS if a[k]]
    if not have:
        return None, []
    if len(have) == 2:
        return "MIXED", a["FACE"] + a["EDGE"]
    return have[0], a[have[0]]


def picks(obj, kind):
    return get_all(obj)[kind]


def set_picks(obj, kind, idx):
    """Store the picked faces / edges of obj (an empty list removes that kind's)."""
    idx = list(dict.fromkeys(int(i) for i in idx))
    d = _raw(obj)
    if idx:
        d[kind] = [[i, _anchor(obj, kind, i)] for i in idx]
    else:
        d.pop(kind, None)
    if any(d.get(k) for k in KINDS):
        obj[PROP] = json.dumps({k: d[k] for k in KINDS if d.get(k)})
    elif PROP in obj:
        del obj[PROP]


def drop_kind(scene, kind):
    """Remove one kind's picks everywhere (that mode was switched off)."""
    for o in scene.objects:
        if PROP in o:
            d = _raw(o)
            if kind in d:
                d.pop(kind)
                if any(d.get(k) for k in KINDS):
                    o[PROP] = json.dumps({k: d[k] for k in KINDS if d.get(k)})
                else:
                    del o[PROP]


def clear_all(scene):
    for o in scene.objects:
        if PROP in o:
            del o[PROP]


_stepf = {}    # session_uid -> (key, [indices])


def surface_key(face):
    """A short id of the surface under a face: the same while later steps only
    trim or split the face (they keep its surface), new when it is rebuilt."""
    import hashlib
    d = store._k()[1].surface_cvs(face.surface())
    s = repr((d["degree_u"], d["degree_v"], d["count_u"], d["count_v"],
              [round(c, 7) + 0.0 for p in d["points"] for c in p]))
    return hashlib.md5(s.encode()).hexdigest()[:16]


def face_keys(brep, pts):
    """The surface key of the face of `brep` each point lies on (None: none there)."""
    faces = brep.faces()
    lo, hi = (np.array(v, float) for v in brep.bounding_box())
    tol = 1e-4 * (float(np.linalg.norm(hi - lo)) or 1.0)
    out = []
    for p in pts:
        ds = [f.closest_point(tuple(p))[1] for f in faces]
        k = int(np.argmin(ds)) if ds else -1
        out.append(surface_key(faces[k]) if k >= 0 and ds[k] < tol else None)
    return out


def _pushed_points(obj, p):
    """Local points on the faces a Push / Pull step moved, where they are now:
    each stored face point carried the distance along its outward normal (in
    world, as the step moves them), as Push / Pull re-picks them."""
    from mathutils import Vector
    from . import geom
    M = geom.world_matrix(obj)
    Mi, N = M.inverted(), M.to_3x3().inverted().transposed()
    d = float(p.get("distance", 0.0))
    nrm = p.get("normals_local") or []
    out = []
    for k, q in enumerate(p.get("faces_local", [])):
        w = M @ Vector(q)
        if k < len(nrm):
            w = w + d * (N @ Vector(nrm[k])).normalized()
        out.append([float(x) for x in Mi @ w])
    return out


def _turned(P, O, z, deg):
    """World points P turned deg degrees about the axis through O along unit z."""
    from .pushpull import _rot3
    return O + (np.asarray(P, float) - O) @ _rot3(z, deg).T


def _step_world(obj, p, key):
    """The stored local points p[key] in world, and the body's matrix."""
    from . import geom
    M = geom.world_matrix(obj)
    return np.array([list(M @ Vector(q)) for q in p.get(key, [])], float).reshape(-1, 3), M


def _rotated_points(obj, p):
    """Local points on the faces a Rotate Face step turned, where they are
    now: turned about the stored hinge, as Rotate Face re-picks them."""
    W, M = _step_world(obj, p, "faces_local")
    O = np.array(M @ Vector(p["axis_point_local"]), float)
    z = np.array((M.to_3x3() @ Vector(p["axis_dir_local"])).normalized(), float)
    Mi = M.inverted()
    return [[float(x) for x in Mi @ Vector(q)] for q in _turned(W, O, z, float(p.get("angle", 0.0)))]


def _moved_points(obj, p):
    """Local points on the faces a Move Face step moved, where they are now:
    carried as the step moves the feature (steps.apply_move_face): slid on a
    flat face; tilted, turned and slid on a curved one; carried by the frames
    over a free-form seat (the seat surface stays, so its frames are found
    on the body as it is now)."""
    W, M = _step_world(obj, p, "faces_local")
    L = M.to_3x3()
    O = np.array(M @ Vector(p["axis_point_local"]), float)
    mode = p.get("mode", "PLANE")
    if mode in ("PLANE", "FREE"):
        v = np.array(L @ (float(p.get("move_x", 0.0)) * Vector(p["axis_x_local"])
                          + float(p.get("move_y", 0.0)) * Vector(p["axis_y_local"])), float)
        if mode == "FREE":
            from .pushpull import free_frame
            faces = store.get_brep(obj).faces()
            q = tuple(p["axis_point_local"])
            s = min(range(len(faces)), key=lambda k: faces[k].closest_point(q)[1])
            R, A, A2, _ = free_frame(obj, s, O, v, bool(p.get("follow", True)))
            W = (W - A) @ R.T + A2
        else:
            W = W + v
    else:
        z = np.array((L @ Vector(p["axis_x_local"])).normalized(), float)
        if mode == "SPHERE" and abs(float(p.get("tilt", 0.0))) > 0.0:
            W = _turned(W, O, np.array((L @ Vector(p["axis_y_local"])).normalized(), float), float(p["tilt"]))
        if abs(float(p.get("turn", 0.0))) > 0.0:
            W = _turned(W, O, z, float(p["turn"]))
        W = W + float(p.get("slide", 0.0)) * z
    Mi = M.inverted()
    return [[float(x) for x in Mi @ Vector(q)] for q in W]


def _replaced_points(obj, p, surf):
    """Local points on the faces a Replace Face step put in: each stored
    point's foot on the replacement surface (the step's input) as it is now."""
    from . import geom
    W, M = _step_world(obj, p, "faces_local")
    if surf is None or not store.is_nurbs(surf):
        return []
    S = geom.world_matrix(surf)
    Si, Mi = S.inverted(), M.inverted()
    sf = store.get_brep(surf).faces()
    out = []
    for q in W:
        ql = tuple(Si @ Vector(q))
        best = min((f.closest_point(ql) for f in sf), key=lambda r: r[1], default=None)
        if best is not None:
            c = best[0]
            c = Vector((c.x, c.y, c.z) if hasattr(c, "x") else tuple(c))
            out.append([float(x) for x in Mi @ (S @ c)])
    return out


def _sculpted_face(obj, p):
    """Index of the face a Face CVs step sculpted: the face nearest its stored
    point whose surface has the step's CV layout (nu x nv, its degrees; u
    and v either way round)."""
    deg = tuple(p.get("deg") or ())
    want = (p["nu"], p["nv"]) + (deg if len(deg) == 2 else (3, 3))
    faces = store.get_brep(obj).faces()
    k = store._k()[1]
    q = tuple(float(x) for x in p["face_local"])
    best = None
    for j, f in enumerate(faces):
        d = k.surface_cvs(f.surface())
        lay = (d["count_u"], d["count_v"], d["degree_u"], d["degree_v"])
        if lay not in (want, (want[1], want[0], want[3], want[2])):     # u and v may come back swapped
            continue
        dist = f.closest_point(q)[1]
        if best is None or dist < best[0]:
            best = (dist, j)
    return None if best is None else best[1]


STEP_FACE_OPS = ("edge_fillet", "push_pull", "rotate_face", "move_face", "replace_face", "face_cvs",
                 "shell", "remove_fillets", "merge_faces", "remove_faces")
MADE_OPS = ("shell", "remove_fillets", "merge_faces", "remove_faces")   # faces kept by steps.store_made


def step_faces(obj):
    """The faces the history step open in the panel made or moved (an edge
    fillet: its fillet faces; Push / Pull, Rotate Face, Replace Face: the
    faces it moved or put in; Move Face: the whole feature it moved; Face
    CVs: the sculpted face; Shell, Remove Fillets / Edges / Faces: the faces
    it made or changed, kept by steps.store_made), while that body is the
    active one; [] otherwise."""
    pg = getattr(obj, "nproto_step", None)
    if (pg is None or not pg.focus or pg.index < 0 or obj != bpy.context.view_layer.objects.active
            or not store.is_nurbs(obj)):
        return []
    try:
        raw = json.loads(obj.get("nproto_stack", "[]"))
        e = raw[pg.index]
    except (ValueError, IndexError, TypeError):
        return []
    op = e.get("op")
    if op not in STEP_FACE_OPS:
        return []
    from . import steps
    p = e.get("params") or {}
    surf = obj.get(f"nproto_sin_{pg.index}_0") if op == "replace_face" else None
    key = (obj.get(store.PROP_REV, 0), pg.index, json.dumps(e, sort_keys=True),
           tuple(tuple(r) for r in obj.matrix_world),
           obj.get(steps.PROP_RIMS, "") if op == "edge_fillet" else obj.get(steps.PROP_MADE, "") if op in MADE_OPS else "",
           (surf.name, surf.get(store.PROP_REV, 0), tuple(tuple(r) for r in surf.matrix_world))
           if surf is not None else None)
    hit = _stepf.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    skeys = []
    try:
        if op == "edge_fillet":
            info = steps.rims_of(obj, p.get("mids_local", [])) or {}
            pts = info.get("faces") or []
            skeys = info.get("keys") or []
        elif op == "push_pull":
            pts = _pushed_points(obj, p)
        elif op == "rotate_face":
            pts = _rotated_points(obj, p)
        elif op == "move_face":
            pts = _moved_points(obj, p)
        elif op == "replace_face":
            pts = _replaced_points(obj, p, surf)
        elif op == "face_cvs":
            j = _sculpted_face(obj, p)
            idx = [] if j is None else [j]
            _stepf[obj.session_uid] = (key, idx)
            return idx
        else:
            info = steps.made_of(obj, op, p) or {}
            pts = info.get("faces") or []
            skeys = info.get("keys") or []
    except Exception as exc:
        print("NURBS step faces:", exc)
        pts = []
    idx = []
    try:
        size = _size(obj)
        now = [surface_key(f) for f in store.get_brep(obj).faces()] if skeys else []
        for n, p in enumerate(pts):
            # by its surface first (a later step may have cut away the point,
            # or split the face: every piece counts); else the point
            sk = skeys[n] if n < len(skeys) else None
            js = [j for j, k in enumerate(now) if sk and k == sk]
            if not js:
                j = _locate(obj, "FACE", -1, p, size)
                js = [] if j is None else [j]
            idx += [j for j in js if j not in idx]
        if op == "move_face" and idx:        # the whole feature lights up, as when it is picked
            from .pushpull import feature_of
            feat = feature_of(obj, idx)
            if feat is not None:
                idx = sorted(set(idx) | set(feat[0]))
    except Exception as exc:
        print("NURBS step faces:", exc)
    _stepf[obj.session_uid] = (key, idx)
    return idx


_stepe = {}    # session_uid -> (key, [edge indices])
STEP_EDGE_OPS = ("imprint", "extend_surface", "match_surface", "surface_align")


def _iso_edges(obj, isos, at=None):
    """Indices of obj's edges on its surface's own boundary sides isos
    (U0/U1/V0/V1); with `at` (a point along that side, picked on a trimmed
    surface) only the pieces there, all of the side when none is."""
    from .kernel import trimext
    on = [(i, a0, a1) for i, (iso, a0, a1) in trimext.boundary_edges(store.get_brep(obj)).items() if iso in isos]
    if at is not None:
        here = [i for i, a0, a1 in on if min(a0, a1) - 1e-9 <= float(at) <= max(a0, a1) + 1e-9]
        if here:
            return sorted(here)
    return sorted(i for i, _, _ in on)


def step_edges(obj):
    """The edges the history step open in the panel made or acted on
    (Imprint: its new edges, kept by steps.store_made; Extend: the side it
    grew; Match / Align: the sides it matched), while that body is the
    active one; [] otherwise."""
    pg = getattr(obj, "nproto_step", None)
    if (pg is None or not pg.focus or pg.index < 0 or obj != bpy.context.view_layer.objects.active
            or not store.is_nurbs(obj)):
        return []
    try:
        raw = json.loads(obj.get("nproto_stack", "[]"))
        e = raw[pg.index]
    except (ValueError, IndexError, TypeError):
        return []
    op = e.get("op")
    if op not in STEP_EDGE_OPS:
        return []
    from . import steps
    p = e.get("params") or {}
    names = steps.step_input_names(obj, pg.index, len(e.get("refs") or []))
    key = (obj.get(store.PROP_REV, 0), pg.index, json.dumps(e, sort_keys=True), tuple(names),
           obj.get(steps.PROP_MADE, "") if op == "imprint" else "")
    hit = _stepe.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    idx = []
    try:
        if op == "imprint":
            size = _size(obj)
            for q in (steps.made_of(obj, op, p, names) or {}).get("edges") or []:
                j = _locate(obj, "EDGE", -1, q, size)
                if j is not None and j not in idx:
                    idx.append(j)
            idx.sort()
        elif op == "extend_surface":
            idx = _iso_edges(obj, {p.get("edge")}, p.get("at"))
        else:
            idx = _iso_edges(obj, set(p.get("edges") or ()))
    except Exception as exc:
        print("NURBS step edges:", exc)
    _stepe[obj.session_uid] = (key, idx)
    return idx


def faces_shown(obj):
    """The faces whose CV nets CVs / Hulls shows on obj: the picked ones, else
    the ones the open history step made, else None for all."""
    f = picks(obj, "FACE") or step_faces(obj)
    return set(f) if f else None


def selected_edges(context):
    """[(object, "E<i>")] -- the picked edges of the selected bodies."""
    out = []
    for o in context.selected_objects:
        for i in picks(o, "EDGE"):
            out.append((o, f"E{i}"))
    return out


def summary(obj):
    """'2 faces, 1 edge picked: F0, F3, E5' / '' for the Active Object panel."""
    a = get_all(obj)
    parts, refs = [], []
    for k, noun, tag in (("FACE", "face", "F"), ("EDGE", "edge", "E")):
        n = len(a[k])
        if n:
            parts.append(f"{n} {noun}{'s' if n > 1 else ''}")
            refs += [tag + str(i) for i in a[k]]
    if not parts:
        return ""
    return " and ".join(parts) + " picked: " + ", ".join(refs[:8]) + (" ..." if len(refs) > 8 else "")


# --------------------------------------------------------------------------
# geometry for the highlight: one face on its own
# --------------------------------------------------------------------------

def face_tris(obj, i):
    """(points (n,3), triangles (m,3), boundary polylines) of face i, local."""
    rev = obj.get(store.PROP_REV, 0)
    key = (obj.session_uid, rev, i)
    hit = _ftris.get(key)
    if hit is not None:
        return hit
    for k in [k for k in _ftris if k[0] == obj.session_uid and k[1] != rev]:
        del _ftris[k]
    sm = _load_kernel()
    ext = sm.blender_ext
    b = store.get_brep(obj).copy()
    faces = b.faces()
    others = [f for k, f in enumerate(faces) if k != i]
    if others:
        ext.delete_faces(b, others)
    chord, angle = store.tess_settings()
    size = _size(obj)
    pb, _ = b.tessellate(chord_height_tolerance=chord * size, curve_angle_tolerance_deg=angle,
                         surface_angle_tolerance_deg=angle, allow_partial=True, return_failures=True)
    a = pb.to_numpy_arrays()
    pts = np.asarray(a["points"], float)
    f = np.asarray(a["faces"])
    tris = []
    j = 0
    while j < len(f):
        c = int(f[j])
        poly = f[j + 1:j + 1 + c]
        for t in range(1, c - 1):
            tris.append((poly[0], poly[t], poly[t + 1]))
        j += 1 + c
    polys = []
    for e in b.edges():
        try:
            polys.append(np.array([e.point_at(t) for t in store.edge_params(e, 40)], float))
        except Exception:
            pass
    out = (pts, np.array(tris, dtype=np.int64).reshape(-1, 3), polys)
    _ftris[key] = out
    return out


# --------------------------------------------------------------------------
# what is under the mouse
# --------------------------------------------------------------------------

def _ray(context, xy):
    region, rv3d = context.region, context.region_data
    return (view3d_utils.region_2d_to_origin_3d(region, rv3d, xy),
            view3d_utils.region_2d_to_vector_3d(region, rv3d, xy))


def face_under(context, xy):
    """(body, face index) under the mouse, or None (nothing, or not a NURBS body)."""
    o, d = _ray(context, xy)
    hit, loc, _, _, ob, _ = context.scene.ray_cast(context.evaluated_depsgraph_get(), o, d)
    if not hit or ob is None:
        return None
    ob = ob.original
    if not store.is_nurbs(ob):
        return None
    sm = _load_kernel()
    b = store.get_brep(ob)
    q = ob.matrix_world.inverted() @ loc
    f = sm.find_face(b, (float(q.x), float(q.y), float(q.z)))
    i = next((k for k, g in enumerate(b.faces()) if g == f), None)
    return None if i is None else (ob, i)


def edge_under(context, xy, px=10.0):
    """(body, edge index) of the visible B-rep edge nearest the mouse (within
    px pixels), or None. An edge hidden behind a surface is not picked."""
    region, rv3d = context.region, context.region_data
    W, H = region.width, region.height
    PM = np.array(rv3d.perspective_matrix)
    m = np.array(xy, float)
    cands = []
    for ob in context.visible_objects:
        if not store.is_nurbs(ob):
            continue
        M = np.array(ob.matrix_world)
        T = PM @ M
        for i, P in enumerate(_edge_polys(ob)):
            if P is None:
                continue
            h = np.c_[P, np.ones(len(P))] @ T.T
            ok = h[:, 3] > 1e-9
            if not ok.all():
                continue
            s = np.c_[(h[:, 0] / h[:, 3] * 0.5 + 0.5) * W, (h[:, 1] / h[:, 3] * 0.5 + 0.5) * H]
            a, b = s[:-1], s[1:]
            ab = b - a
            t = np.clip(((m - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
            dd = np.linalg.norm(a + t[:, None] * ab - m, axis=1)
            k = int(np.argmin(dd))
            if dd[k] <= px:
                p = P[k] + t[k] * (P[k + 1] - P[k])
                cands.append((float(dd[k]), ob, i, M[:3, :3] @ p + M[:3, 3], a[k] + t[k] * ab[k]))
    if not cands:
        return None
    dg = context.evaluated_depsgraph_get()
    for _, ob, i, pw, sxy in sorted(cands, key=lambda c: c[0]):
        o, d = _ray(context, (float(sxy[0]), float(sxy[1])))
        dist = float((np.array(pw) - np.array(o)) @ np.array(d))
        hit, loc, _, _, hob, _ = context.scene.ray_cast(dg, o, d)
        if hit and (np.array(loc) - np.array(o)) @ np.array(d) < dist - 0.01 * _size(ob) * max(ob.scale):
            continue                     # behind a surface
        return ob, i
    return None


def _redraw(context):
    for w in context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


# --------------------------------------------------------------------------
# picking: a background handler while Face / Edge mode is on
# --------------------------------------------------------------------------

RUN = {"on": False, "last": None}


def _view_region_at(window, x, y):
    """(area, main region) of the 3D viewport under the window point (x, y),
    or None -- also over its toolbar, sidebar or headers."""
    for area in window.screen.areas:
        if area.type != 'VIEW_3D' or not (area.x <= x < area.x + area.width and area.y <= y < area.y + area.height):
            continue
        main = None
        for r in area.regions:
            if not (r.x <= x < r.x + r.width and r.y <= y < r.y + r.height) or r.width < 2 or r.height < 2:
                continue
            if r.type == 'WINDOW':
                main = r
            else:
                return None                  # over a panel / header drawn on the viewport
        return (area, main) if main is not None else None
    return None


def _hit(context, ks, xy):
    """(body, index, kind) under the mouse for the kinds being picked: an edge
    within 10 px wins, else the face under the mouse; None for nothing."""
    if isinstance(ks, str):
        ks = {"FACE", "EDGE"} if ks == "FACE_EDGE" else {ks}
    if "EDGE" in ks:
        h = edge_under(context, xy)
        if h is not None:
            return h[0], h[1], "EDGE"
    if "FACE" in ks:
        h = face_under(context, xy)
        if h is not None:
            return h[0], h[1], "FACE"
    return None


def _over_gizmo(context, area, region, x, y):
    """The mouse (window x, y) is over the navigation gizmo (the axis ball and
    its buttons) or a continuity label: those clicks go to them, not to picking."""
    from . import contlabels
    if contlabels.hover.get("key") is not None:
        return True
    return over_navigation(context, area, region, x, y)


def over_navigation(context, area, region, x, y):
    """The mouse (window x, y) is over the 3D view's navigation gizmo: the axis
    ball and the zoom / pan / camera / perspective buttons."""
    sp = area.spaces.active
    if not getattr(sp, "show_gizmo", True) or not getattr(sp, "show_gizmo_navigate", True):
        return False
    prefs = context.preferences
    v = prefs.view
    ball = getattr(v, "mini_axis_type", 'GIZMO') == 'GIZMO'
    buttons = getattr(v, "show_navigate_ui", True)
    if not ball and not buttons:
        return False
    size = float(getattr(v, "gizmo_size_navigate_v3d", 80)) * (getattr(prefs.system, "ui_scale", 1.0) or 1.0)
    from .hints import _insets
    right = _insets(area, region)[1]
    top = region.y + region.height
    cut = sum(r.height for r in area.regions if r.type in ('HEADER', 'TOOL_HEADER') and r.height > 1
              and region.y <= r.y and r.y + r.height <= top + 1 and r.y + r.height >= top - 100)
    x1, y1 = region.x + region.width - right, top - cut
    h = (1.4 * size if ball else 0.0) + (2.4 * size if buttons else 0.0)
    return x1 - 1.6 * size <= x <= x1 and y1 - h <= y <= y1


def apply_pick(context, kind, hit, extend):
    """A click in Face / Edge mode: hit = (body, index[, kind]) or None."""
    if hit is None:
        if not extend:
            clear_all(context.scene)
            for o in context.selected_objects:
                o.select_set(False)
        return
    ob, i = hit[0], hit[1]
    kind = hit[2] if len(hit) > 2 else kind
    if extend:
        idx = list(picks(ob, kind))
        if i in idx:
            idx.remove(i)
        else:
            idx.append(i)
    else:
        clear_all(context.scene)
        for o in context.selected_objects:
            o.select_set(False)
        idx = [i]
    ob.select_set(True)
    context.view_layer.objects.active = ob
    set_picks(ob, kind, idx)


class NPROTO_OT_subsel_picker(bpy.types.Operator):
    """Face / Edge mode: picks faces or edges with any tool active (a click
    picks, Shift+click adds or removes; the face / edge under the mouse lights
    up). A click on no NURBS face / edge clears the picks and goes on to
    Blender, which selects what is there (a mesh body) or deselects. Everything
    else passes through"""
    bl_idname = "nproto.subsel_picker"
    bl_label = "Face / Edge Picking"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        if RUN["on"] or context.window is None:
            return {'CANCELLED'}
        RUN["on"], RUN["last"] = True, None
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _stop(self, context):
        RUN["on"] = False
        HOVER["key"] = None
        _redraw(context)
        return {'FINISHED', 'PASS_THROUGH'}

    def modal(self, context, event):
        ks = kinds(context)
        if not ks or not RUN["on"]:
            return self._stop(context)
        if context.mode != 'OBJECT' or event.type not in ('MOUSEMOVE', 'LEFTMOUSE'):
            return {'PASS_THROUGH'}
        try:
            win = context.window
            where = _view_region_at(win, event.mouse_x, event.mouse_y) if win is not None else None
            if where is not None and _over_gizmo(context, where[0], where[1], event.mouse_x, event.mouse_y):
                if HOVER["key"] is not None:
                    HOVER["key"] = None
                    _redraw(context)
                return {'PASS_THROUGH'}             # the navigation gizmo / a label takes it
            if event.type == 'MOUSEMOVE':
                key = None
                if where is not None and (event.mouse_x, event.mouse_y) != RUN["last"]:
                    RUN["last"] = (event.mouse_x, event.mouse_y)
                    area, region = where
                    with context.temp_override(window=win, area=area, region=region):
                        h = _hit(bpy.context, ks, (event.mouse_x - region.x, event.mouse_y - region.y))
                    key = None if h is None else (h[0].name, h[2], h[1])
                elif where is not None:
                    key = HOVER["key"]
                if key != HOVER["key"]:
                    HOVER["key"] = key
                    _redraw(context)
                return {'PASS_THROUGH'}
            # a click (press and release without a drag): Blender sends it after the release
            if event.value != 'CLICK' or event.ctrl or event.alt or event.oskey or where is None:
                return {'PASS_THROUGH'}
            area, region = where
            with context.temp_override(window=win, area=area, region=region):
                h = _hit(bpy.context, ks, (event.mouse_x - region.x, event.mouse_y - region.y))
                if h is None:
                    if not event.shift:
                        clear_all(bpy.context.scene)
                else:
                    apply_pick(bpy.context, None, h, event.shift)
            _redraw(context)
            if h is None:
                # not on a NURBS face / edge: Blender's own click selects (a mesh
                # body, a curve, a NURBS body away from its edges) or deselects
                return {'PASS_THROUGH'}
            bpy.ops.ed.undo_push(message="Pick " + h[2].title())
            return {'RUNNING_MODAL'}
        except Exception as exc:
            print("NURBS select:", exc)
            return {'PASS_THROUGH'}


def _start_picker():
    """Start the background picker in the first window with a 3D viewport."""
    if RUN["on"] or not kinds():
        return None
    wm = bpy.context.window_manager
    for win in wm.windows:
        area = next((a for a in win.screen.areas if a.type == 'VIEW_3D'), None)
        if area is None:
            continue
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        try:
            with bpy.context.temp_override(window=win, area=area, region=region):
                bpy.ops.nproto.subsel_picker('INVOKE_DEFAULT')
        except Exception as exc:
            print("NURBS select: picker not started", exc)
        break
    return None


def ensure_running():
    """Face / Edge mode on but no picker (a file was loaded, or it was stopped):
    start it again (from a timer: never from inside a draw)."""
    if kinds() and not RUN["on"] and not bpy.app.timers.is_registered(_start_picker):
        bpy.app.timers.register(_start_picker, first_interval=0.05)


_MODE = {"prev": {"BODY"}, "busy": False}


def _sync_modes():
    """Remember the mode set as it is now (after loading a file or the add-on),
    so the next switch knows which picks to keep."""
    try:
        v = bpy.context.window_manager.nproto_view
        m = {v.sel_mode} if isinstance(v.sel_mode, str) else set(v.sel_mode)
        if not m:                              # none set (an older single-mode value): Body
            _MODE["busy"] = True
            try:
                v.sel_mode = {"BODY"}
            finally:
                _MODE["busy"] = False
            m = {"BODY"}
        _MODE["prev"] = m
    except Exception:
        pass
    return None


def settle_modes(new, prev):
    """The mode set after a click on the mode buttons: Face and Edge go together
    (Shift+click adds one), Body stands alone -- added, it is the only one;
    Face / Edge added beside it replace it; none left means Body."""
    new = set(new)
    if "BODY" in new and len(new) > 1:
        new = {"BODY"} if "BODY" not in prev else new - {"BODY"}
    return new or {"BODY"}


def mode_changed(view, context):
    """Body / Face / Edge switched. Face / Edge start the picker; Body ends it.
    Picks of a kind switched off go; Body clears all of them."""
    if _MODE["busy"]:
        return
    cur = {view.sel_mode} if isinstance(view.sel_mode, str) else set(view.sel_mode)
    prev = _MODE["prev"]
    new = settle_modes(cur, prev)
    if new != cur:
        _MODE["busy"] = True
        try:
            view.sel_mode = new
        finally:
            _MODE["busy"] = False
    if "BODY" in new or "BODY" in prev:
        clear_all(context.scene)
    else:
        for k in prev - new:
            drop_kind(context.scene, k)
    _MODE["prev"] = set(new)
    HOVER["key"] = None
    if "BODY" not in new:
        ensure_running()
    _redraw(context)


@bpy.app.handlers.persistent
def _on_load(_):
    RUN["on"] = False                       # loading a file ends every running operator
    HOVER["key"] = None
    TREE_PICK["name"] = None
    for o in bpy.data.objects:              # (nothing is picked in the History list after loading)
        pg = getattr(o, "nproto_step", None)
        if pg is not None and pg.focus:
            pg.focus = False
    _sync_modes()
    ensure_running()


def draw_tool_header(self, context):
    """Body | Face | Edge in the tool settings bar, right after Blender's own
    tool settings (its selection-mode icons; see ui._hook_tool_settings):
    always on top, whatever tool is active."""
    v = getattr(context.window_manager, "nproto_view", None)
    if v is None or context.mode != 'OBJECT' or _load_kernel() is None:
        return
    self.layout.separator()
    row = self.layout.row(align=True)
    row.prop(v, "sel_mode", expand=True)
    ensure_running()


def draw_mode_buttons(layout, context):
    """The same as icons in the viewport header -- only while the tool
    settings bar is hidden, so the switch is always somewhere on top."""
    v = getattr(context.window_manager, "nproto_view", None)
    sd = getattr(context, "space_data", None)
    if v is None or context.mode != 'OBJECT' or getattr(sd, "show_region_tool_header", False):
        return
    row = layout.row(align=True)
    row.prop(v, "sel_mode", expand=True, icon_only=True)
    ensure_running()


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _biased(P, M, bias):
    W = P @ np.array(M.to_3x3()).T + np.array(M.translation)
    if bias is None:
        return W
    from .draw import pull_toward_viewer
    return pull_toward_viewer(W, bias)


def _draw_face(ctx, ob, i, fill, line, width, draw_lines, draw_tris, edge_bias):
    pts, tris, polys = face_tris(ob, i)
    bias = edge_bias(ctx, ob, store.tess_settings(ctx)[0])
    if len(tris):
        W = _biased(pts, ob.matrix_world, bias).astype(np.float32)
        draw_tris(W[tris.reshape(-1)], fill)
    lines = []
    for P in polys:
        W = _biased(P, ob.matrix_world, bias)
        for k in range(len(W) - 1):
            lines += [tuple(W[k]), tuple(W[k + 1])]
    draw_lines(lines, line, width)


def _draw_edge(ctx, ob, i, color, width, draw_lines, edge_bias):
    P = _edge_polys(ob)[i]
    if P is None:
        return
    W = _biased(P, ob.matrix_world, edge_bias(ctx, ob, store.tess_settings(ctx)[0]))
    draw_lines([tuple(W[k // 2 + k % 2]) for k in range(2 * (len(W) - 1))], color, width)


# the object last clicked in the History list: drawn in the step blue (a body
# filled, a sketch as a thick curve) while it stays the active, selected one
TREE_PICK = {"name": None}
_mtris = {}    # session_uid -> (rev, points (n,3) local, triangles (m,3))
_cpoly = {}    # session_uid -> (signature, [world polylines])


def picked_in_tree(ctx):
    """The object clicked in the History list while it is still active and
    selected (a hidden input can't be selected: active is enough), else None."""
    act = ctx.view_layer.objects.active
    if act is None or act.name != TREE_PICK["name"]:
        return None
    try:
        if not act.select_get() and act.visible_get():
            return None
    except RuntimeError:
        pass
    return act


def tree_pick(ctx):
    """The History-picked object to draw blue, or None (also while one of its
    steps is open: then only that step's faces are blue)."""
    act = picked_in_tree(ctx)
    if act is None or getattr(getattr(act, "nproto_step", None), "focus", False):
        return None
    return act


def clear_tree_pick():
    """Nothing in the History list is picked any more: the object shows its
    own feature again (an open step closes)."""
    ob = bpy.data.objects.get(TREE_PICK["name"] or "")
    TREE_PICK["name"] = None
    pg = getattr(ob, "nproto_step", None) if ob is not None else None
    if pg is not None and pg.focus:
        pg.focus = False


@bpy.app.handlers.persistent
def _tree_pick_watch(scene, depsgraph=None):
    """Selecting something else in the viewport (or nothing) ends the History pick."""
    if TREE_PICK["name"] is None:
        return
    try:
        if picked_in_tree(bpy.context) is None:
            clear_tree_pick()
    except Exception as exc:
        print("NURBS history pick:", exc)


def _mesh_tris(ob):
    """The body's own display mesh (already tessellated): local points, triangles."""
    rev = ob.get(store.PROP_REV, 0)
    hit = _mtris.get(ob.session_uid)
    if hit and hit[0] == rev:
        return hit[1], hit[2]
    me = ob.data
    me.calc_loop_triangles()
    pts = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", pts)
    tris = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    out = (pts.reshape(-1, 3).astype(float), tris.reshape(-1, 3))
    _mtris[ob.session_uid] = (rev, *out)
    return out


def _curve_polys(ob):
    """The curve object's splines as world polylines (cached on its points and placement)."""
    M = ob.matrix_world
    sig = (tuple(round(x, 9) for r in M for x in r),
           tuple((s.type, s.use_cyclic_u, s.order_u, s.use_endpoint_u,
                  tuple(round(c, 9) for p in (s.bezier_points if s.type == 'BEZIER' else s.points)
                        for c in (tuple(p.co) + (tuple(p.handle_left) + tuple(p.handle_right)
                                                 if s.type == 'BEZIER' else ()))))
                 for s in ob.data.splines))
    hit = _cpoly.get(ob.session_uid)
    if hit and hit[0] == sig:
        return hit[1]
    from . import geom
    polys = []
    for s in ob.data.splines:
        try:
            polys.append(geom.sample_curve(geom.spline_to_kernel_curve(s, M), 200))
        except Exception:
            pts = s.bezier_points if s.type == 'BEZIER' else s.points
            P = [M @ p.co.to_3d() if len(p.co) == 4 else M @ p.co for p in pts]
            if s.use_cyclic_u and P:
                P.append(P[0])
            polys.append(np.array(P, float))
    _cpoly[ob.session_uid] = (sig, polys)
    return polys


def _draw_tree_pick(ctx, ob, draw_lines, draw_tris, edge_bias):
    import gpu
    if ob.type == 'CURVE':
        gpu.state.depth_test_set('NONE')         # a sketch inside a body still shows
        lines = []
        for W in _curve_polys(ob):
            lines += [tuple(W[k // 2 + k % 2]) for k in range(2 * (len(W) - 1))]
        draw_lines(lines, COL_STEP_LINE, 4.0)
        gpu.state.depth_test_set('LESS_EQUAL')
        return
    if not store.is_nurbs(ob):
        return
    bias = edge_bias(ctx, ob, store.tess_settings(ctx)[0])
    pts, tris = _mesh_tris(ob)
    if len(tris):
        W = _biased(pts, ob.matrix_world, bias).astype(np.float32)
        draw_tris(W[tris.reshape(-1)], COL_STEP)
    lines = []
    for P in store.display(ob).get("edges", []):
        if len(P) > 1:
            W = _biased(np.asarray(P, float), ob.matrix_world, bias)
            lines += [tuple(W[k // 2 + k % 2]) for k in range(2 * (len(W) - 1))]
    draw_lines(lines, COL_STEP_LINE, 2.5)


def draw_view(ctx, draw_lines, draw_tris, edge_bias):
    import gpu
    if ctx.mode != 'OBJECT':
        return
    tp = tree_pick(ctx)
    if tp is not None and not get(tp)[1]:
        try:
            _draw_tree_pick(ctx, tp, draw_lines, draw_tris, edge_bias)
        except Exception as exc:
            print("NURBS history pick: draw", exc)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('ALPHA')
    for ob in ctx.selected_objects:
        if not store.is_nurbs(ob) or not ob.visible_get():
            continue
        a = get_all(ob)
        for kind in KINDS:
            for i in a[kind]:
                try:
                    if kind == "FACE":
                        _draw_face(ctx, ob, i, COL_PICK, COL_PICK_LINE, 2.5, draw_lines, draw_tris, edge_bias)
                    else:
                        _draw_edge(ctx, ob, i, COL_PICK_LINE, 4.0, draw_lines, edge_bias)
                except Exception as exc:
                    print("NURBS select: draw", exc)
    act = ctx.view_layer.objects.active
    if act is not None and act.visible_get() and not get(act)[1]:
        for i in step_faces(act):
            try:
                _draw_face(ctx, act, i, COL_STEP, COL_STEP_LINE, 2.5, draw_lines, draw_tris, edge_bias)
            except Exception as exc:
                print("NURBS step faces: draw", exc)
        for i in step_edges(act):
            try:
                _draw_edge(ctx, act, i, COL_STEP_LINE, 4.0, draw_lines, edge_bias)
            except Exception as exc:
                print("NURBS step edges: draw", exc)
    key = HOVER["key"]
    if key is not None and key[1] in kinds(ctx):
        ob = bpy.data.objects.get(key[0])
        if ob is not None and store.is_nurbs(ob) and ob.visible_get():
            try:
                if key[1] == "FACE":
                    _draw_face(ctx, ob, key[2], COL_HOVER, COL_HOVER_LINE, 1.5, draw_lines, draw_tris, edge_bias)
                else:
                    _draw_edge(ctx, ob, key[2], COL_HOVER_LINE, 4.0, draw_lines, edge_bias)
            except Exception:
                pass


classes = (NPROTO_OT_subsel_picker,)


def register():
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    if _tree_pick_watch not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_tree_pick_watch)
    bpy.app.timers.register(_sync_modes, first_interval=0.05)
    ensure_running()


def unregister():
    RUN["on"] = False                       # a running picker ends at its next event
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    if _tree_pick_watch in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_tree_pick_watch)
    if bpy.app.timers.is_registered(_start_picker):
        bpy.app.timers.unregister(_start_picker)
