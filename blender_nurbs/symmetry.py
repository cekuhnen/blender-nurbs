"""Symmetry: an object whose CVs come in mirrored pairs across a plane (as
Alias's Symmetric Modeling). The Mirror tool's Make Symmetric sets it up; the
tool's other output, Mirror Copy, is a pattern copy (pattern.py, mode MIRROR).

Make Symmetric pairs every CV with its twin across the plane and copies one
half (the master side) onto the other:
  * a curve spline: CV i with CV n-1-i (a closed spline: with CV s-i, the s
    that fits the mirror best); a Bezier point's left handle twins its
    twin's right handle;
  * a surface: its CV rows reversed along U or V, the direction whose
    reversal fits the mirror best (a trimmed surface keeps its trims, a
    surface with history gets a CV Tweak step: the cage's Done does both).
A CV that is its own twin (the middle of an odd count) is put on the plane.
The master side is the start of the curve / the first rows; Flip Master Side
(F in the tool) takes the other half. The knots must read the same both ways
(Blender's uniform and endpoint knots do); weights are copied with the CVs.

While the object, or its CV cage (Edit CVs), is in Edit Mode a depsgraph
handler keeps the pairs: a CV that moved puts its twin at its mirror image --
whatever moved it (G / R / S, proportional editing, snapping, typed values).
When both twins moved, the selected one leads; both or neither selected: the
master side. A CV on the plane stays on it.

The plane is kept on the object in its own space (PROP), so it moves with
the object. Remove Symmetry drops it and leaves the CVs where they are.
In local space the plane is a point o, the direction d a CV is mirrored along
and the covector c of the plane (c . d = 1): f(x) = c . (x - o) is the
height above it, x - 2 f d the mirror image. For an object scaled unevenly
d and c differ; the mirror in world stays the true one.
"""

import json

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import geom, store

PROP = "nproto_sym"
COL_PLANE = (0.3, 0.78, 1.0, 0.75)
COL_TWIN = (0.25, 0.55, 1.0, 1.0)

_busy = [False]
_last = {}          # object name -> (layout key, rows (surface / curve local), weights) after the last enforcement


# --------------------------------------------------------------------------
# The plane (pure)
# --------------------------------------------------------------------------

def _unit(v, fallback=(0.0, 0.0, 1.0)):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.asarray(fallback, float)


def plane_from_world(M, O, n):
    """The plane through world point O, normal n, in the local space of an
    object at world matrix M: {"o", "d", "c"} (c . d = 1)."""
    M = np.asarray(M, float)
    n = _unit(n)
    Mi = np.linalg.inv(M)
    o = Mi[:3, :3] @ np.asarray(O, float) + Mi[:3, 3]
    d = Mi[:3, :3] @ n
    c = M[:3, :3].T @ n
    k = float(c @ d)
    return {"o": o, "d": d, "c": c / k}


def world_plane(M, sym):
    """(point, unit normal) of the plane in world space."""
    M = np.asarray(M, float)
    O = M[:3, :3] @ sym["o"] + M[:3, 3]
    n = _unit(np.linalg.inv(M[:3, :3]).T @ sym["c"])
    return O, n


def height(sym, P):
    P = np.asarray(P, float)
    return (P[:, :3] - sym["o"]) @ sym["c"]


def reflect(sym, P):
    """P (n, 3 or more: extra columns, the weight, kept) mirrored."""
    P = np.array(P, float)
    P[:, :3] -= 2.0 * np.outer(height(sym, P), sym["d"])
    return P


def project(sym, P):
    P = np.array(P, float)
    P[:, :3] -= np.outer(height(sym, P), sym["d"])
    return P


def to_json(sym):
    return {k: [float(x) for x in np.asarray(v, float)] if k in ("o", "d", "c") else v for k, v in sym.items()}


def from_json(raw):
    try:
        s = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    for k in ("o", "d", "c"):
        if k not in s:
            return None
        s[k] = np.asarray(s[k], float)
    return s


def sym_of(obj):
    """obj's symmetry (plane in its local space and pairing), or None."""
    if obj is None or PROP not in obj:
        return None
    return from_json(obj[PROP])


# --------------------------------------------------------------------------
# Pairs (pure)
# --------------------------------------------------------------------------

def chain_twins(n, cyclic=False, shift=None):
    """Twin of each CV of a spline: n-1-i, or (shift - i) mod n closed."""
    i = np.arange(n)
    if cyclic:
        return (int(shift or 0) - i) % n
    return n - 1 - i


def best_shift(P, sym):
    """The s for which CV s-i (mod n) is nearest CV i's mirror image."""
    P = np.asarray(P, float)[:, :3]
    n = len(P)
    R = reflect(sym, P)
    best, s_best = None, 0
    for s in range(n):
        e = float(((R - P[(s - np.arange(n)) % n]) ** 2).sum())
        if best is None or e < best - 1e-15:
            best, s_best = e, s
    return s_best


def bezier_twins(n, cyclic=False, shift=None):
    """Rows [left handle, point, right handle] per point: row 3i+k twins 3t+2-k."""
    t = chain_twins(n, cyclic, shift)
    out = np.empty(3 * n, int)
    for i in range(n):
        for k in range(3):
            out[3 * i + k] = 3 * t[i] + 2 - k
    return out


def grid_twins(nu, nv, direction):
    """Twins of a u-major CV grid (index iu * nv + iv), reversed along U or V."""
    iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    if direction == "U":
        iu = nu - 1 - iu
    else:
        iv = nv - 1 - iv
    return (iu * nv + iv).ravel()


def best_direction(G, sym):
    """"U" or "V": the direction whose reversed rows lie nearest the mirror images."""
    G = np.asarray(G, float)
    nu, nv = G.shape[:2]
    P = G.reshape(-1, 3)
    R = reflect(sym, P)
    err = {}
    for dr in ("U", "V"):
        t = grid_twins(nu, nv, dr)
        err[dr] = float(((R - P[t]) ** 2).sum())
    return "U" if err["U"] <= err["V"] else "V"


def master_mask(twins, master=0):
    """True for the CVs that lead: the first of each pair (master 0) or the
    second (master 1); a CV that is its own twin leads itself."""
    i = np.arange(len(twins))
    return np.where(twins == i, True, (i < twins) if not master else (i > twins))


def symmetrize(P, twins, lead, sym):
    """P with each pair's follower put at its leader's mirror image and the
    CVs that are their own twins put on the plane."""
    P = np.asarray(P, float)
    Q = P.copy()
    i = np.arange(len(P))
    fixed = twins == i
    if fixed.any():
        Q[fixed] = project(sym, P[fixed])
    fol = ~lead
    if fol.any():
        Q[fol] = reflect(sym, P[twins[fol]])
    return Q


def enforce(P, P0, twins, sel, lead, sym, tol):
    """P (just edited) with its pairs mirrored again: a CV that moved since P0
    puts its twin at its mirror image; both moved: the selected one leads,
    else the master side (lead). -> (Q, changed)."""
    P = np.asarray(P, float)
    P0 = np.asarray(P0, float)
    moved = np.abs(P - P0).max(axis=1) > tol
    if not moved.any():
        return P, False
    Q = P.copy()
    n = len(P)
    for i in np.nonzero(moved)[0]:
        t = int(twins[i])
        if t == i:
            Q[i] = project(sym, P[i:i + 1])[0]
            continue
        if moved[t]:
            if bool(sel[i]) != bool(sel[t]):
                if not sel[i]:
                    continue                    # the twin leads (its turn in the loop)
            elif not lead[i]:
                continue
        Q[t] = reflect(sym, P[i:i + 1])[0]
    changed = bool(np.abs(Q - P).max() > tol) if n else False
    return Q, changed


def resymmetrize(P, twins, sel, lead, sym):
    """P with every pair mirrored again, whichever moved: the selected twin
    leads, else the master side (lead); CVs that are their own twins on the
    plane. After an undo: the step Blender stored may hold a move without the
    twin the handler wrote just after it."""
    sel = np.asarray(sel, bool)
    t = np.asarray(twins)
    leads = np.where(sel != sel[t], sel, np.asarray(lead, bool))
    return symmetrize(P, t, leads, sym)


def knots_symmetric(U, tol=1e-9):
    U = np.asarray(U, float)
    if not len(U):
        return True
    return bool(np.abs((U[0] + U[-1] - U[::-1]) - U).max() <= tol * (1.0 + abs(U[-1] - U[0])))


# --------------------------------------------------------------------------
# Blender data
# --------------------------------------------------------------------------

def _spline_rows(sp):
    """(kind, rows, sel): NURBS / POLY rows (n, 4) xyzw; BEZIER rows (3n, 3)."""
    if sp.type == 'BEZIER':
        bp = sp.bezier_points
        n = len(bp)
        rows = np.empty((n, 3, 3))
        sel = np.empty((n, 3), bool)
        for k, (attr, sattr) in enumerate((("handle_left", "select_left_handle"), ("co", "select_control_point"),
                                           ("handle_right", "select_right_handle"))):
            a = np.empty(n * 3)
            bp.foreach_get(attr, a)
            rows[:, k] = a.reshape(n, 3)
            s = np.empty(n, bool)
            bp.foreach_get(sattr, s)
            sel[:, k] = s
        return "BEZIER", rows.reshape(-1, 3), sel.ravel()
    pts = sp.points
    n = len(pts)
    a = np.empty(n * 4)
    pts.foreach_get("co", a)
    s = np.empty(n, bool)
    pts.foreach_get("select", s)
    return "NURBS", a.reshape(n, 4), s


def _write_spline_rows(sp, kind, rows):
    if kind == "BEZIER":
        n = len(sp.bezier_points)
        R = np.asarray(rows, float).reshape(n, 3, 3)
        for k, attr in enumerate(("handle_left", "co", "handle_right")):
            sp.bezier_points.foreach_set(attr, R[:, k].ravel())
    else:
        sp.points.foreach_set("co", np.asarray(rows, float).ravel())


def _spline_twins(sp, kind, sym, si, rows=None):
    shifts = sym.get("shift") or {}
    n = len(sp.bezier_points) if kind == "BEZIER" else len(sp.points)
    shift = None
    if sp.use_cyclic_u:
        shift = shifts.get(str(si))
        if shift is None:
            P = rows if rows is not None else _spline_rows(sp)[1]
            co = P.reshape(n, 3, 3)[:, 1] if kind == "BEZIER" else P[:, :3]
            shift = best_shift(co, sym)
    if kind == "BEZIER":
        return bezier_twins(n, sp.use_cyclic_u, shift), shift
    return chain_twins(n, sp.use_cyclic_u, shift), shift


def surface_reason(obj):
    """None if obj's surface CVs can be paired (Edit CVs works on it), else why not."""
    from . import cvs
    return cvs.editable_reason(obj)


def _surface_grid(obj):
    from .kernel import load as load_kernel
    ext = load_kernel().blender_ext
    d = ext.surface_cvs(store.get_brep(obj).faces()[0].surface())
    nu, nv = d["count_u"], d["count_v"]
    return d, np.asarray(d["points"], float).reshape(nu, nv, 3)


def usable(obj):
    return obj is not None and (obj.type == 'CURVE' or store.is_nurbs(obj))


def target_of(obj):
    """The object whose symmetry an Edit Mode object follows: a curve itself,
    a CV cage its surface."""
    if obj is None:
        return None
    if obj.type == 'MESH':
        from . import cvs
        if cvs.is_cage(obj):
            return bpy.data.objects.get(obj[cvs.CAGE_OF])
    return obj


def master_side(obj, sym, flip=False):
    """+1 / -1: the side of the plane (local height sign) the master half
    mostly lies on (for a preview of Make Symmetric)."""
    hs = []
    if obj.type == 'CURVE':
        for si, sp in enumerate(obj.data.splines):
            kind, rows, _ = _spline_rows(sp)
            tw, _ = _spline_twins(sp, kind, sym, si, rows)
            lead = master_mask(tw, 1 if flip else 0)
            hs.append(height(sym, rows[lead & (tw != np.arange(len(tw)))]))
    elif store.is_nurbs(obj) and surface_reason(obj) is None:
        _, G = _surface_grid(obj)
        dr = best_direction(G, sym)
        tw = grid_twins(G.shape[0], G.shape[1], dr)
        lead = master_mask(tw, 1 if flip else 0)
        hs.append(height(sym, G.reshape(-1, 3)[lead & (tw != np.arange(len(tw)))]))
    h = np.concatenate(hs) if hs else np.zeros(0)
    return -1.0 if len(h) and float(h.sum()) < 0 else 1.0


def make_symmetric(context, obj, O, n, flip=False):
    """Pair obj's CVs across the world plane (O, n), copy the master half onto
    the other and keep the plane on obj. -> (moved: the largest CV move,
    notes: [str]). Raises RuntimeError when obj can't be paired."""
    M = np.array(geom.world_matrix(obj), float)
    sym = plane_from_world(M, O, n)
    sym["master"] = 1 if flip else 0
    notes = []
    moved = 0.0
    if obj.type == 'CURVE':
        if not len(obj.data.splines):
            raise RuntimeError("the curve has no splines")
        shifts = {}
        for si, sp in enumerate(obj.data.splines):
            kind, rows, _ = _spline_rows(sp)
            tw, shift = _spline_twins(sp, kind, dict(sym, shift={}), si, rows)
            if shift is not None:
                shifts[str(si)] = int(shift)
            if kind == "NURBS" and sp.type == 'NURBS' and not sp.use_cyclic_u:
                U = geom.blender_knots(len(sp.points), sp.order_u, False, sp.use_endpoint_u, sp.use_bezier_u)
                if not knots_symmetric(U):
                    notes.append(f"spline {si + 1}: its knots are not the same both ways, so the curve is "
                                 "only nearly symmetric (Endpoint knots make it exact)")
            lead = master_mask(tw, sym["master"])
            Q = symmetrize(rows, tw, lead, sym)
            moved = max(moved, float(np.abs(Q[:, :3] - rows[:, :3]).max()) if len(Q) else 0.0)
            _write_spline_rows(sp, kind, Q)
            if kind == "BEZIER":
                bp = sp.bezier_points
                N = len(bp)
                lt = [p.handle_left_type for p in bp]
                rt = [p.handle_right_type for p in bp]
                t = chain_twins(N, sp.use_cyclic_u, shift)
                lead_pt = master_mask(t, sym["master"])
                for i in range(N):
                    if not lead_pt[i]:
                        j = int(t[i])
                        bp[i].handle_left_type, bp[i].handle_right_type = rt[j], lt[j]
                _write_spline_rows(sp, kind, Q)        # (a handle type change can move the handles)
        sym["shift"] = shifts
        obj.data.update_tag()
    else:
        why = surface_reason(obj)
        if why:
            raise RuntimeError(why)
        from . import cvs
        d, G = _surface_grid(obj)
        nu, nv = G.shape[:2]
        dr = best_direction(G, sym)
        K = d["knots_u"] if dr == "U" else d["knots_v"]
        if not knots_symmetric(K):
            notes.append(f"its {dr} knots are not the same both ways, so the surface is only nearly symmetric")
        tw = grid_twins(nu, nv, dr)
        lead = master_mask(tw, sym["master"])
        P = G.reshape(-1, 3)
        Wt = np.asarray(d.get("weights") or [], float)
        if len(Wt) == len(P):
            P = np.c_[P, Wt]
        Q = symmetrize(P, tw, lead, sym)
        moved = float(np.abs(Q[:, :3] - P[:, :3]).max()) if len(Q) else 0.0
        sym["dir"] = dr
        if moved > 0.0 or (Q.shape[1] > 3 and np.abs(Q[:, 3] - P[:, 3]).max() > 0):
            cage = cvs.cage_for(obj)
            if cage is not None:
                raise RuntimeError("its CVs are being edited: Done first")
            if context.object is not None and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            cage = cvs.make_cage(context, obj)
            if Q.shape[1] > 3:
                meta = json.loads(cage[cvs.CAGE_META])
                meta["weights"] = [float(w) for w in Q[:, 3]]
                cage[cvs.CAGE_META] = json.dumps(meta)
            cage.data.vertices.foreach_set("co", Q[:, :3].astype(np.float32).ravel())
            cage.data.update()
            vl = context.view_layer
            act, sel = vl.objects.active, [o for o in context.selected_objects]
            cvs.finish_cage(cage, vl)
            for o in vl.objects:
                if o is not None and o.select_get():
                    o.select_set(False)
            for o in sel:
                if o.name in bpy.data.objects:
                    try:
                        o.select_set(True)
                    except RuntimeError:
                        pass
            if act is not None and act.name in bpy.data.objects:
                vl.objects.active = act
    obj[PROP] = json.dumps(to_json(sym))
    _last.pop(obj.name, None)
    return moved, notes


def remove(obj):
    if obj is not None and PROP in obj:
        del obj[PROP]
        _last.pop(obj.name, None)
        return True
    return False


def twins_world(obj):
    """World points of the twins of the selected CVs of an object in Edit
    Mode (a symmetric curve, or a symmetric surface's cage), for drawing."""
    tgt = target_of(obj)
    sym = sym_of(tgt)
    if sym is None or obj.mode != 'EDIT':
        return []
    out = []
    if obj.type == 'CURVE':
        M = np.array(geom.world_matrix(obj), float)
        for si, sp in enumerate(obj.data.splines):
            kind, rows, sel = _spline_rows(sp)
            tw, _ = _spline_twins(sp, kind, sym, si, rows)
            idx = [int(tw[i]) for i in np.nonzero(sel)[0] if tw[i] != i and not sel[tw[i]]]
            if idx:
                P = rows[idx, :3]
                out += list(P @ M[:3, :3].T + M[:3, 3])
        return out
    pts, sel = _cage_rows(obj)
    if pts is None:
        return out
    meta = json.loads(obj[_cvs().CAGE_META])
    tw = grid_twins(meta["nu"], meta["nv"], sym.get("dir", "U"))
    if len(tw) != len(pts):
        return out
    M = np.array(geom.world_matrix(obj), float)
    idx = [int(tw[i]) for i in np.nonzero(sel)[0] if tw[i] != i and not sel[tw[i]]]
    if idx:
        out += list(pts[idx] @ M[:3, :3].T + M[:3, 3])
    return out


def _cvs():
    from . import cvs
    return cvs


def _cage_rows(cage):
    import bmesh
    try:
        bm = bmesh.from_edit_mesh(cage.data)
    except Exception:
        return None, None
    bm.verts.ensure_lookup_table()
    n = len(bm.verts)
    P = np.array([tuple(v.co) for v in bm.verts], float).reshape(n, 3)
    S = np.array([v.select for v in bm.verts], bool)
    return P, S


# --------------------------------------------------------------------------
# Edit Mode: keep the pairs
# --------------------------------------------------------------------------

def _tol(P):
    size = float(np.abs(np.asarray(P, float)[:, :3]).max()) if len(P) else 1.0
    return 2e-6 * (1.0 + size)


def keep_curve(obj, sym, whole=False):
    """One pass over a symmetric curve in Edit Mode (whole: every pair, not
    only the ones that moved -- after an undo). -> True if it wrote."""
    key = tuple((sp.type, len(sp.bezier_points) if sp.type == 'BEZIER' else len(sp.points), sp.use_cyclic_u)
                for sp in obj.data.splines)
    data = [_spline_rows(sp) for sp in obj.data.splines]
    prev = _last.get(obj.name)
    if not whole and (prev is None or prev[0] != key):
        _last[obj.name] = (key, [r.copy() for _, r, _ in data])
        return False
    wrote = False
    lead_side = int(sym.get("master", 0))
    snap = []
    for si, (sp, (kind, rows, sel), r0) in enumerate(zip(obj.data.splines, data, prev[1] if not whole else
                                                         [None] * len(data))):
        tw, _ = _spline_twins(sp, kind, sym, si, rows)
        if whole:
            Q = resymmetrize(rows, tw, sel, master_mask(tw, lead_side), sym)
            changed = bool(len(Q)) and float(np.abs(Q - rows).max()) > _tol(rows)
        else:
            Q, changed = enforce(rows, r0, tw, sel, master_mask(tw, lead_side), sym, _tol(rows))
        if changed:
            _write_spline_rows(sp, kind, Q)
            wrote = True
        snap.append(Q)
    _last[obj.name] = (key, snap)
    if wrote:
        obj.data.update_tag()
    return wrote


def keep_cage(cage, surf, sym, whole=False):
    import bmesh
    P, S = _cage_rows(cage)
    if P is None:
        return False
    cvs = _cvs()
    meta = json.loads(cage[cvs.CAGE_META])
    tw = grid_twins(meta["nu"], meta["nv"], sym.get("dir", "U"))
    if len(tw) != len(P):
        return False
    # the cage's space -> the surface's (where the plane is kept)
    A = np.array(geom.world_matrix(surf), float)
    A = np.linalg.inv(A) @ np.array(geom.world_matrix(cage), float)
    Ps = P @ A[:3, :3].T + A[:3, 3]
    key = (len(P),)
    prev = _last.get(cage.name)
    lead = master_mask(tw, int(sym.get("master", 0)))
    if whole:
        Q = resymmetrize(Ps, tw, S, lead, sym)
        changed = float(np.abs(Q - Ps).max()) > _tol(Ps)
    elif prev is None or prev[0] != key:
        _last[cage.name] = (key, Ps.copy())
        return False
    else:
        Q, changed = enforce(Ps, prev[1], tw, S, lead, sym, _tol(Ps))
    _last[cage.name] = (key, Q)
    if not changed:
        return False
    Ai = np.linalg.inv(A)
    Qc = Q @ Ai[:3, :3].T + Ai[:3, 3]
    bm = bmesh.from_edit_mesh(cage.data)
    bm.verts.ensure_lookup_table()
    for v, q in zip(bm.verts, Qc):
        if (v.co - _vec(q)).length > 0.0:
            v.co = _vec(q)
    bmesh.update_edit_mesh(cage.data, loop_triangles=False, destructive=False)
    return True


def _vec(q):
    from mathutils import Vector
    return Vector((float(q[0]), float(q[1]), float(q[2])))


def _pass(scene, whole=False):
    live = set()
    for obj in scene.objects:
        if obj.mode != 'EDIT' or obj.type not in {'CURVE', 'MESH'}:
            continue
        tgt = target_of(obj)
        sym = sym_of(tgt)
        if sym is None:
            continue
        live.add(obj.name)
        try:
            if obj.type == 'CURVE':
                keep_curve(obj, sym, whole)
            else:
                keep_cage(obj, tgt, sym, whole)
        except Exception as exc:
            print(f"NURBS symmetric CV edit ({obj.name}): {exc}")
    for k in [k for k in _last if k not in live]:
        del _last[k]


@persistent
def _on_depsgraph(scene, depsgraph):
    if _busy[0]:
        return
    if not any(isinstance(u.id, (bpy.types.Curve, bpy.types.Mesh, bpy.types.Object)) for u in depsgraph.updates):
        return
    _busy[0] = True
    try:
        _pass(scene)
    finally:
        _busy[0] = False


@persistent
def _on_undo(scene, *_):
    """After Ctrl+Z / Ctrl+Shift+Z in Edit Mode: every pair mirrored again
    (Blender stores a step when the operator ends, before the handler has
    moved the twin of a move made in one go -- a script, a redo panel)."""
    if _busy[0]:
        return
    _busy[0] = True
    try:
        _pass(scene, whole=True)
    finally:
        _busy[0] = False


# --------------------------------------------------------------------------
# Drawing: the plane of a selected symmetric object, the twins of the selected CVs
# --------------------------------------------------------------------------

def plane_outline(O, n, pts, grow=0.08):
    """A rectangle in the plane (O, n) round the points' shadow on it (closed polyline)."""
    from .pattern import _perp
    n = _unit(n)
    u = _perp(n)
    v = np.cross(n, u)
    P = np.asarray(pts, float) - O
    a, b = P @ u, P @ v
    if not len(P):
        a = b = np.array([-1.0, 1.0])
    a0, a1, b0, b1 = a.min(), a.max(), b.min(), b.max()
    span = max(a1 - a0, b1 - b0, 1e-6)
    a0, a1, b0, b1 = a0 - grow * span, a1 + grow * span, b0 - grow * span, b1 + grow * span
    C = [O + a0 * u + b0 * v, O + a1 * u + b0 * v, O + a1 * u + b1 * v, O + a0 * u + b1 * v]
    return C + [C[0]]


def _world_box(obj):
    M = np.array(geom.world_matrix(obj), float)
    B = np.array([tuple(c) for c in obj.bound_box], float)
    return B @ M[:3, :3].T + M[:3, 3]


def draw(ctx, draw_lines, draw_points, lines_from_polylines, ident):
    act = ctx.view_layer.objects.active if ctx.view_layer else None
    shown = []
    for obj in ctx.visible_objects:
        if PROP not in obj and not (obj.type == 'MESH' and obj.mode == 'EDIT'):
            continue
        tgt = target_of(obj)
        sym = sym_of(tgt)
        if sym is None or not (obj.select_get() or obj.mode == 'EDIT' or obj == act):
            continue
        if tgt in shown:
            continue
        shown.append(tgt)
        O, n = world_plane(np.array(geom.world_matrix(tgt), float), sym)
        out = plane_outline(O, n, _world_box(obj))
        draw_lines(lines_from_polylines([[_vec(q) for q in out]], ident), COL_PLANE, 1.5)
        if obj.mode == 'EDIT':
            tw = twins_world(obj)
            if tw:
                draw_points([_vec(q) for q in tw], (0.0, 0.0, 0.0, 1.0), 13.0)
                draw_points([_vec(q) for q in tw], COL_TWIN, 10.0)


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

class NPROTO_OT_remove_symmetry(bpy.types.Operator):
    """Stop moving this object's CVs in mirrored pairs (the CVs stay where they are)"""
    bl_idname = "nproto.remove_symmetry"
    bl_label = "Remove Symmetry"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        done = 0
        objs = set(context.selected_objects) | ({context.active_object} if context.active_object else set())
        for o in objs:
            if remove(target_of(o)):
                done += 1
        if not done:
            self.report({'WARNING'}, "Nothing selected is symmetric")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Symmetry removed from {done} object{'s' if done > 1 else ''}")
        return {'FINISHED'}


def draw_panel(layout, context, obj):
    """Active Object panel: a symmetric object's plane and Remove Symmetry."""
    tgt = target_of(obj)
    sym = sym_of(tgt)
    if sym is None:
        return
    O, n = world_plane(np.array(geom.world_matrix(tgt), float), sym)
    box = layout.box()
    box.label(text="Symmetric: CVs move in mirrored pairs", icon='MOD_MIRROR')
    box.label(text="Plane at ({:.4g}, {:.4g}, {:.4g}), normal ({:.3g}, {:.3g}, {:.3g})".format(*O, *n))
    box.operator("nproto.remove_symmetry", icon='X')


classes = (NPROTO_OT_remove_symmetry,)


_HANDLERS = (("depsgraph_update_post", _on_depsgraph), ("undo_post", _on_undo), ("redo_post", _on_undo))


def register():
    for lst, fn in _HANDLERS:
        hs = getattr(bpy.app.handlers, lst)
        if fn not in hs:
            hs.append(fn)


def unregister():
    for lst, fn in _HANDLERS:
        hs = getattr(bpy.app.handlers, lst)
        if fn in hs:
            hs.remove(fn)
    _last.clear()
