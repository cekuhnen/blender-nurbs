"""Nudge CVs: fine CV moves on the CV cage, as Alias's Transform CV, with
continuity guides at the edges that meet another surface.

Runs in the CV cage's Edit Mode (Tab on a surface) on the selected CVs:

    click a CV          select it (Shift: add / remove); press and drag to move
    drag anywhere       the selection moves along the active direction, 1 / Sensitivity
                        of the mouse (Shift: a tenth of that, Ctrl: whole Steps)
    arrow keys          one Step along it (Up / Right +, Down / Left -; Shift a tenth, Ctrl ten)
    N U V               each CV's own normal / U hull / V hull direction (Alias NUV)
    X Y Z               world axes;  F  free, in the view plane
    a number + Enter    exactly that far along the direction (units work: 2mm)
    [ ]                 halve / double the Step
    C                   guides: Off, G0, G1, G2
    L                   lock: every guided CV stays on its guide while it moves
    P                   put the selected guided CVs onto their guides
    Backspace, Ctrl+Z   take the last nudge back
    Enter, Space, RMB   done (one undo step);  Esc: everything back as it was

Guides. Every boundary edge of the surface that lies on an edge of another
surface (found as Match Surface finds it) gets the conditions Match Surface
sets, CV by CV (see match.py; row 0 is the edge, rows 1 and 2 the next ones
in):

    G0  row 0 CV j on the target's edge CV P_j                     (a point)
    G1  row 1 CV on the line x0 + s D_j: the target's hull line     (a line)
        run on straight across the edge (s > 0)
    G2  row 2 CV in the plane where the curvature across the edge   (a plane)
        is the target's: c2 ((x2 - x1)/d2 - (x1 - x0)/d1) . n
        = k_t |c1 (x1 - x0)|^2, n the normal of the target's tangent
        plane at CV j, k_t = DD_j . n / |D_j|^2

These are the per-CV conditions (sufficient, the ones Match Surface makes
exact); the measured continuity of the whole edge (Continuity Check's
analysis) is shown next to them, so the two can be compared. Guides need the
two layouts along the edge to be one (degree and knots); where they are not,
the edge says so and only the measurement is shown (Match Surface makes the
layouts one).

The cage rebuilds the surface as it moves (cvs.py), so history is as with
any CV edit: Done on the cage after a real move adds a CV Tweak step
(cvtweak.py), which keeps the CVs that end on their guide on it.
"""

import math

import bmesh
import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

from . import cvs, draw, hints, match, salign, store

DIRS = ("N", "U", "V", "X", "Y", "Z", "FREE")
LEVELS = ("OFF", "G0", "G1", "G2")
PICK_PX = 12.0
DRAG_PX = 3.0               # a press becomes a drag once the mouse moved this far
COL_OK = (0.3, 1.0, 0.45, 1.0)
COL_BAD = (1.0, 0.32, 0.25, 1.0)
COL_GUIDE = (0.35, 0.85, 1.0, 0.9)
COL_AXIS = (1.0, 0.85, 0.3, 0.9)
COL_PLANE = (0.35, 0.85, 1.0, 0.15)
PLANE_PX = 22.0   # half the G2 plane square's width, in pixels
WORLD = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

overlay = {"on": False, "guides": [], "sel": set(), "axis": None, "labels": []}
_handles = []


# --------------------------------------------------------------------------
# The CV net (world space) and the frames of its CVs
# --------------------------------------------------------------------------

def _unit(V):
    V = np.asarray(V, float)
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    return V / np.maximum(n, 1e-300)


def frames(G):
    """Each CV's own directions (Alias NUV): U and V along its hull lines
    (from neighbour to neighbour, one-sided at the ends), N across them.
    G (nu, nv, 3) -> {"N", "U", "V"}: (nu, nv, 3) unit vectors."""
    G = np.asarray(G, float)

    def along(axis):
        Gt = G if axis == 0 else G.transpose(1, 0, 2)
        D = np.empty_like(Gt)
        if Gt.shape[0] < 2:
            D[:] = 0.0
        else:
            D[1:-1] = Gt[2:] - Gt[:-2]
            D[0] = Gt[1] - Gt[0]
            D[-1] = Gt[-1] - Gt[-2]
        return D if axis == 0 else D.transpose(1, 0, 2)

    U, V = along(0), along(1)
    N = np.cross(U, V)
    # a collapsed hull (a pole: all of a row on one point) has no V there: use the next row's
    bad = np.linalg.norm(N, axis=-1) < 1e-12 * max(float(np.ptp(G.reshape(-1, 3), axis=0).max()), 1e-30)
    if bad.any():
        for i, j in zip(*np.nonzero(bad)):
            for di in (1, -1, 2, -2):
                ii = i + di
                if 0 <= ii < G.shape[0] and not bad[ii, j]:
                    N[i, j] = N[ii, j]
                    break
    return {"N": _unit(N), "U": _unit(U), "V": _unit(V)}


def net_of_cage(cage, pts=None):
    """match.Net of the cage's CVs (world space, knots 0..1), or None when
    the surface is rational (its weights make the per-CV guides wrong)."""
    import json
    meta = json.loads(cage[cvs.CAGE_META])
    w = meta.get("weights") or []
    if w and max(abs(x - 1.0) for x in w) > 1e-12:
        return None
    pts = cvs.cage_points(cage) if pts is None else pts
    M = np.array(cage.matrix_world, float)
    P = np.asarray(pts, float) @ M[:3, :3].T + M[:3, 3]
    G = P.reshape(meta["nu"], meta["nv"], 3)
    return match.Net(G, (match._norm(meta["knots_u"]), match._norm(meta["knots_v"])),
                     (meta["degree_u"], meta["degree_v"]))


# --------------------------------------------------------------------------
# Partners: the surfaces across the edges
# --------------------------------------------------------------------------

def side_on_layout(net, edge, sd):
    """The target side sd on the net's own layout along `edge` (the
    target's rows refined to it, exactly), or None when the net would have
    to change (a degree raised or knots inserted on its side)."""
    Gs, Ua, Ue, p, q = net.std(edge)
    keys, item = match._side_item(sd)
    try:
        q2, U2, arrs = match.compatible([(q, Ue, Gs[0]), item])
    except Exception:
        return None
    if q2 != q or len(arrs[0]) != Gs.shape[1]:
        return None
    from . import geom
    return geom._side_from(keys, q2, U2, arrs[1])


def find_partners(context, surf, net, exclude=()):
    """{edge: {"obj", "ref", "side" (on the net's layout) or None, "max" (the
    highest level it allows), "why"}} for the edges of the net that lie on
    another surface's edge or a curve."""
    from .ops import _pick_candidates
    skip = {surf.name} | {o.name for o in exclude}
    cands = [c for c in _pick_candidates(context) if c[0].name not in skip and not cvs.is_cage(c[0])]
    out = {}
    for edge in salign.EDGES:
        Gs, Ua, Ue, p, q = net.std(edge)
        E = Gs[0]
        if float(np.sum(np.linalg.norm(np.diff(E, axis=0), axis=1))) < 1e-12:
            continue                                    # a pole: no edge
        try:
            tgt = salign.find_neighbour(surf, edge, cands)
        except Exception:
            tgt = None
        if tgt is None or isinstance(tgt, list):
            continue
        out[edge] = partner_of(net, edge, *tgt)
    return out


def partner_of(net, edge, tobj, tref):
    """{"obj", "ref", "side" (on the net's layout) or None, "max", "why"}:
    the target (tobj, tref) as the partner of the net's `edge`."""
    Gs, Ua, Ue, p, q = net.std(edge)
    E = Gs[0]
    e = {"obj": tobj, "ref": tref, "side": None, "max": 0, "why": None}
    try:
        sd = match.fit_side(match.target_side(tobj, tref), E[0], E[-1], "EDGE")
    except Exception as exc:
        e["why"] = str(exc)
        return e
    s = side_on_layout(net, edge, sd)
    if s is None:
        e["why"] = "the CV layouts along the edge differ (Match Surface makes them one)"
    else:
        e["side"] = s
        e["max"] = 0 if s.get("D") is None else min(2, p, Gs.shape[0] - 1)
    return e


# --------------------------------------------------------------------------
# Guides
# --------------------------------------------------------------------------

def _basis(U, p, n, ts):
    """(len(ts), n) values of the B-spline basis (degree 0 included)."""
    if p > 0:
        return salign.basis_matrix(U, p, n, ts)
    B = np.zeros((len(ts), n))
    for r, t in enumerate(ts):
        i = max(k for k in range(n) if U[k] <= t or k == 0)
        B[r, min(i, n - 1)] = 1.0
    return B


def basis_ders(U, p, n, ts):
    """Basis values and their first and second derivatives at ts:
    (B, B1, B2), each (len(ts), n)."""
    U = [float(x) for x in U]
    ts = np.asarray(ts, float)
    B = _basis(U, p, n, ts)
    z = np.zeros_like(B)
    if p < 1 or n < 2:
        return B, z, z

    def diff(U, p, n):            # control points of the derivative from the CVs: (n - 1, n)
        Dm = np.zeros((n - 1, n))
        for i in range(n - 1):
            h = U[i + p + 1] - U[i + 1]
            if h > 0:
                Dm[i, i], Dm[i, i + 1] = -p / h, p / h
        return Dm

    D1 = diff(U, p, n)
    B1 = _basis(U[1:-1], p - 1, n - 1, ts) @ D1
    if p < 2 or n < 3:
        return B, B1, z
    D2 = diff(U[1:-1], p - 1, n - 1) @ D1
    B2 = _basis(U[2:-2], p - 2, n - 2, ts) @ D2
    return B, B1, B2


def _rowdot(A, B):
    return np.einsum("ij,ij->i", A, B)


def g2_system(net, edge, sd):
    """The G2 condition across `edge`, sampled along it, as a linear system
    in the heights of the row 2 CVs: moving row 2 CV j by h_j along nu_j
    changes the curvature mismatch at the samples by A[:, j] h_j.
    -> {"A" (m, n_e), "e" (m: the target's curvature across the edge minus
    the surface's, both in the direction perpendicular to the edge), "nu"
    (n_e, 3), "kscale", "k2" (flat indices of row 2)} or None.

    The curvatures across the edge (normal curvature in the tangent-plane
    direction w perpendicular to the edge, II(w, w) / |w|^2) use the full
    second fundamental form, so a hull leaning along the edge (a tangent
    length that changes from CV to CV) is accounted for."""
    nu_, nv_ = net.G.shape[:2]
    Gs, Ua, Ue, p, q = net.std(edge)
    if p < 2 or Gs.shape[0] < 3:
        return None
    n_e = Gs.shape[1]
    kc = match._consts(Ua, p)
    m = max(8, 4 * n_e)
    ts = np.linspace(0.0, 1.0, m)
    B, B1, B2 = basis_ders(Ue, q, n_e, ts)
    P, D, DD = (np.asarray(sd[k], float) for k in ("P", "D", "DD"))
    P1, P2, Dt, D1, DDt = B1 @ P, B2 @ P, B @ D, B1 @ D, B @ DD
    n = _unit(np.cross(P1, Dt))
    # the target's curvature across the edge
    bt = -_rowdot(Dt, P1) / np.maximum(_rowdot(P1, P1), 1e-300)
    wt = Dt + bt[:, None] * P1
    kt = _rowdot(DDt + 2 * bt[:, None] * D1 + (bt ** 2)[:, None] * P2, n) / np.maximum(_rowdot(wt, wt), 1e-300)
    # the surface's, all but row 2's part
    x0, x1, x2 = Gs[0], Gs[1], Gs[2]
    Sa, Sat = kc["c1"] * (B @ (x1 - x0)), kc["c1"] * (B1 @ (x1 - x0))
    St, Stt = B1 @ x0, B2 @ x0
    bs = -_rowdot(Sa, St) / np.maximum(_rowdot(St, St), 1e-300)
    ws = Sa + bs[:, None] * St
    w2 = np.maximum(_rowdot(ws, ws), 1e-300)
    Saa = kc["c2"] * (B @ ((x2 - x1) / kc["d2"] - (x1 - x0) / kc["d1"]))
    ks = (_rowdot(Saa, n) + 2 * bs * _rowdot(Sat, n) + bs ** 2 * _rowdot(Stt, n)) / w2
    # each row 2 CV moves along the target's normal at its own place along the edge
    xi = np.array([float(np.mean(Ue[j + 1:j + q + 1])) if q > 0 else float(Ue[j]) for j in range(n_e)])
    Bx, B1x, _ = basis_ders(Ue, q, n_e, xi)
    nu = _unit(np.cross(B1x @ P, Bx @ D))
    A = (kc["c2"] / kc["d2"]) * B * (n @ nu.T) / w2[:, None]
    L = float(np.sum(np.linalg.norm(np.diff(B @ P, axis=0), axis=1))) or 1.0
    kscale = max(float(np.max(np.abs(kt))), float(np.max(np.abs(ks))), 0.05 / L)
    ix = salign._grid_index(edge, nu_, nv_)
    return {"A": A, "e": kt - ks, "nu": nu, "kscale": kscale, "k2": [ix(2, j) for j in range(n_e)]}


def solve_heights(sysm, free):
    """Least-squares heights h (along nu) of the row 2 CVs `free` (indices j
    along the edge), the others staying: -> {j: h}."""
    free = sorted(free)
    if not free:
        return {}
    A = sysm["A"][:, free]
    ridge = 1e-12 * (float(np.sum(A * A)) or 1.0)
    h = np.linalg.solve(A.T @ A + ridge * np.eye(len(free)), A.T @ sysm["e"])
    return dict(zip(free, h))


def guides(net, partners, level):
    """Per-CV guides of the partnered edges up to `level` (0, 1, 2):
    [{"k" (flat CV index), "edge", "row", "j", "kind" ("point", "line",
    "plane"), "O", "d" (line direction / plane normal), "err", "unit"}].

    G0: the target's edge CV; err the distance to it.
    G1: the target's hull line run on straight from this CV's row 0 CV; err
        the angle (degrees) between the hull leg and it.
    G2: the plane (through the best place for this CV, its neighbours
        staying where they are) in which it keeps the curvature across the
        edge closest to the target's; err how much of that mismatch moving
        it there takes away, relative to the largest curvature along the
        edge (as Continuity Check's %). Every row 2 CV on its plane = the
        whole row is the least-squares best."""
    nu, nv = net.G.shape[:2]
    X = net.G.reshape(-1, 3)
    out = []
    for edge, e in partners.items():
        sd = e.get("side")
        top = min(level, e.get("max", 0))
        if sd is None or top < 0:
            continue
        Gs, Ua, Ue, p, q = net.std(edge)
        n_e = Gs.shape[1]
        ix = salign._grid_index(edge, nu, nv)
        P = np.asarray(sd["P"], float)
        for j in range(n_e):
            k = ix(0, j)
            out.append({"k": k, "edge": edge, "row": 0, "j": j, "kind": "point", "O": P[j], "d": None,
                        "err": float(np.linalg.norm(X[k] - P[j])), "unit": "G0"})
        if top < 1:
            continue
        Du = _unit(np.asarray(sd["D"], float))
        for j in range(n_e):
            k0, k1 = ix(0, j), ix(1, j)
            leg = X[k1] - X[k0]
            ang = math.degrees(math.atan2(float(np.linalg.norm(np.cross(leg, Du[j]))), float(leg @ Du[j])))
            out.append({"k": k1, "edge": edge, "row": 1, "j": j, "kind": "line", "O": X[k0].copy(), "d": Du[j],
                        "err": ang, "unit": "G1"})
        if top < 2:
            continue
        sysm = g2_system(net, edge, sd)
        if sysm is None:
            continue
        for j in range(n_e):
            h = solve_heights(sysm, [j])[j]
            k = sysm["k2"][j]
            fix = float(np.max(np.abs(sysm["A"][:, j] * h)))
            out.append({"k": k, "edge": edge, "row": 2, "j": j, "kind": "plane", "O": X[k] + h * sysm["nu"][j],
                        "d": sysm["nu"][j], "err": fix / sysm["kscale"], "unit": "G2"})
    return out


def fit_row2(net, partners, level, free):
    """The net's CVs (flat, world) with the row 2 CVs in `free` (flat
    indices) moved together to their least-squares best G2 places."""
    X = net.G.reshape(-1, 3).copy()
    if level < 2:
        return X
    for edge, e in partners.items():
        if e.get("side") is None or e.get("max", 0) < 2:
            continue
        sysm = g2_system(match.Net(X.reshape(net.G.shape), net.knots, net.deg), edge, e["side"])
        if sysm is None:
            continue
        js = [j for j, k in enumerate(sysm["k2"]) if k in free]
        for j, h in solve_heights(sysm, js).items():
            X[sysm["k2"][j]] += h * sysm["nu"][j]
    return X


def holds(g, tol):
    return g["err"] <= tol[g["unit"]]


def onto(x, g):
    """x put onto guide g (the nearest point of it)."""
    x = np.asarray(x, float)
    if g["kind"] == "point":
        return np.array(g["O"], float)
    if g["kind"] == "line":
        s = max(float((x - g["O"]) @ g["d"]), 0.0)      # s > 0: a leg back over the edge folds it
        return g["O"] + s * g["d"]
    return x - float((x - g["O"]) @ g["d"]) * g["d"]


def first_guides(gs):
    """{CV index: its guide of the lowest row} (a CV two edges guide: the one
    nearer its edge rules, e.g. row 1 of one edge over row 2 of the other)."""
    best = {}
    for g in gs:
        b = best.get(g["k"])
        if b is None or g["row"] < b["row"]:
            best[g["k"]] = g
    return best


def tolerances(context=None):
    v = getattr((context or bpy.context).window_manager, "nproto_view", None)
    if v is None:
        return {"G0": 1e-5, "G1": 0.1, "G2": 0.05}
    return {"G0": v.cont_gap_tol, "G1": v.cont_angle_tol, "G2": v.cont_curv_tol / 100.0}


def fmt_err(g, context=None):
    from . import contcheck
    return f"{g['unit']} " + contcheck.fmt(g["unit"], g["err"], context)


# --------------------------------------------------------------------------
# Moves (pure: world positions in, world positions out)
# --------------------------------------------------------------------------

def direction(dirs, k, axis, view_dirs=None):
    """The direction CV k moves along for `axis` (unit)."""
    if axis in WORLD:
        return np.array(WORLD[axis], float)
    if axis in ("N", "U", "V"):
        return dirs[axis].reshape(-1, 3)[k]
    return np.asarray(view_dirs[0], float)


class Guided:
    """The cage's CV layout and the partners of its edges: guides and fits
    for any positions X (flat, world)."""

    def __init__(self, net0, partners):
        self.shape, self.knots, self.deg = net0.G.shape, net0.knots, net0.deg
        self.partners = partners

    def net(self, X):
        return match.Net(np.asarray(X, float).reshape(self.shape), self.knots, self.deg)

    def guides(self, X, level):
        return guides(self.net(X), self.partners, level) if level >= 0 and self.partners else []

    def fit(self, X, level, free):
        return fit_row2(self.net(X), self.partners, level, free) if free else np.array(X, float)


def put_on(gd, X, ks, level):
    """X with the CVs ks put onto their guides: row 0 first (the next rows'
    guides start there), then row 1, then the row 2 CVs together."""
    Y = np.array(X, float)
    for row in (0, 1):
        best = first_guides(gd.guides(Y, level))
        for k in ks:
            g = best.get(k)
            if g is not None and g["row"] == row:
                Y[k] = onto(Y[k], g)
    best = first_guides(gd.guides(Y, level))
    return gd.fit(Y, level, {k for k in ks if k in best and best[k]["row"] == 2})


def nudge(X, sel, delta_of, gd=None, level=-1, lock=False, held=()):
    """New positions: X (n, 3) with every CV k in sel moved by delta_of(k).
    lock: the guided CVs of sel end on their guides (rows 0, 1: the point /
    line; row 2: in its plane), and the row 2 CVs in `held` (on their guide
    before) that are not in sel are fitted again together, so they follow
    a row 1 CV that moved."""
    Y = np.array(X, float)
    for k in sel:
        Y[k] = Y[k] + np.asarray(delta_of(k), float)
    if not lock or gd is None or level < 0:
        return Y
    for row in (0, 1, 2):
        best = first_guides(gd.guides(Y, level))
        for k in sel:
            g = best.get(k)
            if g is not None and g["row"] == row:
                Y[k] = onto(Y[k], g)
    best = first_guides(gd.guides(Y, level))
    return gd.fit(Y, level, {k for k in held if k not in sel and k in best and best[k]["row"] == 2})


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------

def _view_settings(context):
    return context.window_manager.nproto_view


class NPROTO_OT_cv_nudge(bpy.types.Operator):
    """Move CVs in fine steps along their normal, hull or the world axes
    (Alias Transform CV), with G0 / G1 / G2 guides where the surface meets
    another one. Runs on the CV cage (Tab on a surface)"""
    bl_idname = "nproto.cv_nudge"
    bl_label = "Nudge CVs"
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and cvs.is_cage(obj) and obj.mode == 'EDIT' and context.area is not None \
            and context.area.type == 'VIEW_3D'

    # -- set-up -----------------------------------------------------------------
    def invoke(self, context, event):
        self._cage = context.active_object
        self._surf = bpy.data.objects.get(self._cage[cvs.CAGE_OF])
        if self._surf is None:
            self.report({'WARNING'}, "Nudge CVs: the cage's surface is gone")
            return {'CANCELLED'}
        bm = self._bm()
        self._start = np.array([tuple(v.co) for v in bm.verts], float)       # cage local
        self._sel = {v.index for v in bm.verts if v.select}
        self._undo = []             # positions before each nudge (Backspace)
        self._typed = ""
        self._press = None          # {"mouse", "X", "k", "t0" | "p0", "drag"}
        self._free_h = True         # FREE: the last arrow key was Left / Right
        import json
        meta = json.loads(self._cage[cvs.CAGE_META])
        self._shape = (meta["nu"], meta["nv"], 3)
        self._partners = {}
        self._gd = None
        net0 = net_of_cage(self._cage)
        if net0 is not None:
            try:
                self._partners = find_partners(context, self._surf, net0, (self._cage,))
            except Exception as exc:
                print("NURBS Nudge CVs: partners:", exc)
            self._gd = Guided(net0, self._partners)
        self._measured = {}
        self._meas_rev = None
        self._measure(context)
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        overlay["on"] = True
        self._refresh(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _bm(self):
        bm = bmesh.from_edit_mesh(self._cage.data)
        bm.verts.ensure_lookup_table()
        return bm

    def _win(self, context):
        area = context.area
        region = next((r for r in area.regions if r.type == 'WINDOW'), context.region)
        return region, area.spaces.active.region_3d

    def _mouse(self, context, event):
        region, _ = self._win(context)
        return np.array([event.mouse_x - region.x, event.mouse_y - region.y], float)

    def _ray(self, context, m):
        region, rv3d = self._win(context)
        o = view3d_utils.region_2d_to_origin_3d(region, rv3d, Vector(tuple(m)))
        d = view3d_utils.region_2d_to_vector_3d(region, rv3d, Vector(tuple(m)))
        return np.array(o, float), np.array(d, float)

    # -- positions --------------------------------------------------------------
    def _M(self):
        return np.array(self._cage.matrix_world, float)

    def _world(self, L=None):
        M = self._M()
        L = np.array([tuple(v.co) for v in self._bm().verts], float) if L is None else L
        return L @ M[:3, :3].T + M[:3, 3]

    def _write(self, W):
        M = self._M()
        L = (np.asarray(W, float) - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T
        bm = self._bm()
        for v, p in zip(bm.verts, L):
            v.co = Vector(tuple(p))
        bmesh.update_edit_mesh(self._cage.data, loop_triangles=False, destructive=False)

    def _level(self, context):
        return LEVELS.index(_view_settings(context).cv_guides) - 1

    def _guides(self, context, W):
        if self._gd is None:
            return []
        return self._gd.guides(W, self._level(context))

    def _dirs(self, W):
        return frames(np.asarray(W, float).reshape(self._shape))

    def _view_dirs(self, context):
        _, rv3d = self._win(context)
        R = rv3d.view_rotation
        return np.array(R @ Vector((1, 0, 0)), float), np.array(R @ Vector((0, 1, 0)), float)

    # -- the move ---------------------------------------------------------------
    def _apply(self, context, W0, delta_of):
        """Move the selection from W0 by delta_of(k) (world); one nudge."""
        if not self._sel:
            return False
        s = _view_settings(context)
        held = ()
        if s.cv_lock and self._gd is not None:
            tol = tolerances(context)
            held = {k for k, g in first_guides(self._guides(context, W0)).items() if holds(g, tol)}
        W = nudge(W0, self._sel, delta_of, self._gd, self._level(context), s.cv_lock, held)
        if np.abs(W - W0).max() <= 0.0:
            return False
        self._write(W)
        return True

    def _step_move(self, context, sign, scale):
        s = _view_settings(context)
        W0 = self._world()
        amount = sign * s.cv_step * scale
        axis = s.cv_axis
        dirs = self._dirs(W0)
        vd = self._view_dirs(context)
        if axis == "FREE":
            delta = lambda k: amount * vd[0 if self._free_h else 1]
        else:
            delta = lambda k: amount * direction(dirs, k, axis)
        self._undo.append(W0)
        if not self._apply(context, W0, delta):
            self._undo.pop()

    def _typed_move(self, context):
        from .drawtools import parse_length
        try:
            amount = parse_length(context, self._typed)
        except Exception:
            self.report({'WARNING'}, f"Nudge CVs: {self._typed!r} is not a length")
            return
        s = _view_settings(context)
        if s.cv_axis == "FREE":
            self.report({'INFO'}, "Nudge CVs: pick a direction (N U V / X Y Z) to type a distance")
            return
        W0 = self._world()
        dirs = self._dirs(W0)
        self._undo.append(W0)
        if not self._apply(context, W0, lambda k: amount * direction(dirs, k, s.cv_axis)):
            self._undo.pop()

    def _drag_to(self, context, event):
        pr = self._press
        s = _view_settings(context)
        W0 = pr["X"]
        dirs = pr["dirs"]
        m = self._mouse(context, event)
        o, d = self._ray(context, m)
        sens = max(s.cv_sensitivity, 1e-6) * (10.0 if event.shift else 1.0)
        ref = W0[pr["k"]]
        if s.cv_axis == "FREE":
            n = pr["view_n"]
            den = float(d @ n)
            if abs(den) < 1e-9:
                return
            p = o + d * float((ref - o) @ n) / den
            move = (p - pr["p0"]) / sens
            if event.ctrl and s.cv_step > 0:
                move = np.round(move / s.cv_step) * s.cv_step
            delta = lambda k: move
        else:
            from .valuedrag import axis_param
            a = direction(dirs, pr["k"], s.cv_axis)
            t = axis_param(o, d, ref, a)
            if t is None or pr["t0"] is None:
                return
            amount = (t - pr["t0"]) / sens
            if event.ctrl and s.cv_step > 0:
                amount = round(amount / s.cv_step) * s.cv_step
            pr["amount"] = amount
            delta = lambda k: amount * direction(dirs, k, s.cv_axis)
        self._apply(context, W0, delta)

    def _snap_onto(self, context):
        W0 = self._world()
        best = first_guides(self._guides(context, W0))
        ks = [k for k in self._sel if k in best]
        if not ks:
            self.report({'INFO'}, "Nudge CVs: no selected CV has a guide")
            return
        W = put_on(self._gd, W0, ks, self._level(context))
        self._undo.append(W0)
        self._write(W)

    # -- picking ----------------------------------------------------------------
    def _cv_at(self, context, m):
        region, rv3d = self._win(context)
        best, bd = None, PICK_PX
        for k, p in enumerate(self._world()):
            q = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(tuple(p)))
            if q is None:
                continue
            dd = math.hypot(q.x - m[0], q.y - m[1])
            if dd < bd:
                best, bd = k, dd
        return best

    def _select(self, k, add):
        bm = self._bm()
        if not add:
            for v in bm.verts:
                v.select = False
            for e in bm.edges:
                e.select = False
        v = bm.verts[k]
        v.select = not v.select if add else True
        bm.select_history.clear()
        if v.select:
            bm.select_history.add(v)
        bm.select_flush(False)
        bmesh.update_edit_mesh(self._cage.data, loop_triangles=False, destructive=False)
        self._sel = {u.index for u in bm.verts if u.select}

    def _start_press(self, context, event, k):
        W0 = self._world()
        m = self._mouse(context, event)
        o, d = self._ray(context, m)
        s = _view_settings(context)
        if k is None:
            act = self._bm().select_history.active
            k = act.index if isinstance(act, bmesh.types.BMVert) and act.select else min(self._sel)
        pr = {"mouse": m, "X": W0, "k": k, "drag": False, "dirs": self._dirs(W0), "amount": 0.0}
        ref = W0[k]
        if s.cv_axis == "FREE":
            _, rv3d = self._win(context)
            n = np.array(rv3d.view_rotation @ Vector((0, 0, 1)), float)
            den = float(d @ n)
            pr["view_n"] = n
            pr["p0"] = o + d * float((ref - o) @ n) / den if abs(den) > 1e-9 else ref
        else:
            from .valuedrag import axis_param
            pr["t0"] = axis_param(o, d, ref, direction(pr["dirs"], k, s.cv_axis))
            if pr["t0"] is None:
                self.report({'INFO'}, "Nudge CVs: the direction points at you -- turn the view, or use the arrow keys")
        self._press = pr

    # -- display ----------------------------------------------------------------
    def _measure(self, context):
        """Continuity Check's measurement of every partnered edge against its
        target surface (cached until the surface rebuilds)."""
        rev = (self._surf.get(store.PROP_REV, 0), len(self._surf.get(store.PROP_BREP, "")))
        if rev == self._meas_rev:
            return
        self._meas_rev = rev
        self._measured = {}
        if not store.is_natural(self._surf):
            return
        for edge, e in self._partners.items():
            if e["obj"].type == 'CURVE':
                continue
            try:
                self._measured[edge] = match.measure(self._surf, edge, e["obj"], e["ref"])
            except Exception as exc:
                self._measured[edge] = str(exc) or exc.__class__.__name__

    def _refresh(self, context):
        s = _view_settings(context)
        W = self._world()
        gs = self._guides(context, W)
        tol = tolerances(context)
        overlay["guides"] = [dict(g, ok=holds(g, tol), x=W[g["k"]]) for g in gs]
        overlay["sel"] = set(self._sel)
        overlay["axis"] = None
        if self._sel and s.cv_axis != "FREE":
            dirs = self._dirs(W)
            span = float(np.ptp(W, axis=0).max()) or 1.0
            overlay["axis"] = [(W[k], direction(dirs, k, s.cv_axis), 0.08 * span) for k in self._sel]
        self._status(context, gs, tol, W)
        for a in context.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()

    def _status(self, context, gs, tol, W):
        from .valuedrag import fmt_length
        s = _view_settings(context)
        moved = float(np.linalg.norm(W - self._world(self._start), axis=1).max()) if len(W) else 0.0
        names = {"N": "normal", "U": "U hull", "V": "V hull", "X": "X", "Y": "Y", "Z": "Z", "FREE": "view plane"}
        state = [f"{len(self._sel)} CV" + ("s" if len(self._sel) != 1 else "") if self._sel else "click a CV",
                 "along " + names[s.cv_axis], "Step " + fmt_length(context, s.cv_step)
                 + (f" · 1/{s.cv_sensitivity:g}" if s.cv_sensitivity != 1 else ""),
                 f"typed {self._typed}" if self._typed else "",
                 f"moved {fmt_length(context, moved)}" if moved > 0 else ""]
        best = first_guides(gs)
        on = [best[k] for k in self._sel if k in best]
        if on:
            worst = max(on, key=lambda g: (g["row"], g["err"] / max(tol[g["unit"]], 1e-300)))
            bad = [g for g in on if not holds(g, tol)]
            state.append((f"{len(bad)} off guide, " if bad else "on guide, ") + "worst " + fmt_err(worst, context))
        for edge, e in self._partners.items():
            r = self._measured.get(edge)
            tgt = f"{edge} → {e['obj'].name}"
            if e.get("why") and not isinstance(r, dict):
                state.append(f"{tgt}: no guides")
            elif isinstance(r, dict):
                from . import contcheck
                parts = ["G0 " + contcheck.fmt("G0", r["gap"], context)]
                if r.get("angle") is not None:
                    parts.append("G1 " + contcheck.fmt("G1", r["angle"], context))
                    parts.append("G2 " + contcheck.fmt("G2", r["curv"], context))
                state.append(f"{tgt} measured " + " ".join(parts))
        lv = s.cv_guides
        rows = [("Move", [hints.key("Drag", "along the direction"), hints.key("← →", "one Step"),
                          hints.key("Shift", "a tenth"), hints.key("Ctrl", "ten Steps / whole Steps"),
                          hints.key("0-9", "exact distance"), hints.key("[ ]", "Step ÷2 ×2")]),
                ("Direction", [hints.cycle("N U V", ["N", "U", "V"], s.cv_axis if s.cv_axis in "NUV" else None,
                                           "the CV's own"),
                               hints.cycle("X Y Z", ["X", "Y", "Z"], s.cv_axis if s.cv_axis in "XYZ" else None, "world"),
                               hints.key("F", "view plane", on=True if s.cv_axis == "FREE" else None)]),
                ("Guides", [hints.cycle("C", ["Off", "G0", "G1", "G2"], {"OFF": "Off"}.get(lv, lv)),
                            hints.toggle("L", "lock", s.cv_lock), hints.key("P", "put on guide")]
                 + ([hints.note("no edge of this surface meets another surface")] if not self._partners else [])),
                ("Nudge", [hints.key("Backspace", "take back", on=None if self._undo else False),
                           hints.key("Enter", "done")])]
        hints.show(context, "Nudge CVs", state, rows,
                   [("MOUSE_LMB", "Pick / drag"), ("MOUSE_RMB", "Done"), ("EVENT_ESC", "Cancel")])

    # -- modal ------------------------------------------------------------------
    def modal(self, context, event):
        from .drawtools import PASS_KEYS
        from .ops import PickMixin
        if self._cage.name not in bpy.data.objects or self._cage.mode != 'EDIT':
            return self._end(context, cancel=False)
        et, ev = event.type, event.value
        dragging = self._press is not None
        if et == 'TIMER':
            self._measure(context)
            self._refresh(context)
            return {'PASS_THROUGH'}
        if not dragging and (et in PASS_KEYS and et not in ('INBETWEEN_MOUSEMOVE',) or
                             (et in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event))):
            return {'PASS_THROUGH'}
        if et == 'INBETWEEN_MOUSEMOVE':
            return {'PASS_THROUGH'}
        if hints.handle(event):
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        s = _view_settings(context)
        if et == 'MOUSEMOVE':
            if dragging:
                pr = self._press
                if not pr["drag"] and np.linalg.norm(self._mouse(context, event) - pr["mouse"]) >= DRAG_PX:
                    pr["drag"] = True
                    self._undo.append(pr["X"])
                if pr["drag"]:
                    self._drag_to(context, event)
                    self._refresh(context)
            return {'RUNNING_MODAL'}
        if et == 'LEFTMOUSE' and ev == 'PRESS':
            k = self._cv_at(context, self._mouse(context, event))
            if k is not None and (event.shift or k not in self._sel):
                self._select(k, event.shift)
            if self._sel and not (event.shift and k is not None and k not in self._sel):
                self._start_press(context, event, k if k in self._sel else None)
            self._refresh(context)
            return {'RUNNING_MODAL'}
        if et == 'LEFTMOUSE' and ev == 'RELEASE':
            self._press = None
            self._refresh(context)
            return {'RUNNING_MODAL'}
        if ev != 'PRESS':
            return {'RUNNING_MODAL'}
        if dragging:
            return {'RUNNING_MODAL'}
        if et in {'RET', 'NUMPAD_ENTER'} and self._typed:
            self._typed_move(context)
            self._typed = ""
        elif et in {'RET', 'NUMPAD_ENTER', 'SPACE', 'RIGHTMOUSE'}:
            return self._end(context, cancel=False)
        elif et == 'ESC':
            if self._typed:
                self._typed = ""
            else:
                return self._end(context, cancel=True)
        elif et == 'BACK_SPACE' and self._typed:
            self._typed = self._typed[:-1]
        elif et == 'BACK_SPACE' or (et == 'Z' and (event.ctrl or event.oskey)):
            if self._undo:
                self._write(self._undo.pop())
        elif et in {'UP_ARROW', 'RIGHT_ARROW', 'DOWN_ARROW', 'LEFT_ARROW'}:
            self._free_h = et in {'RIGHT_ARROW', 'LEFT_ARROW'}
            sign = 1.0 if et in {'UP_ARROW', 'RIGHT_ARROW'} else -1.0
            self._step_move(context, sign, 0.1 if event.shift else (10.0 if event.ctrl else 1.0))
        elif et in {'N', 'U', 'V', 'X', 'Y', 'Z'} and not (event.ctrl or event.alt or event.oskey):
            s.cv_axis = et
        elif et == 'F' and not event.ctrl:
            s.cv_axis = "FREE"
        elif et == 'C' and not event.ctrl:
            s.cv_guides = LEVELS[(LEVELS.index(s.cv_guides) + (-1 if event.shift else 1)) % len(LEVELS)]
        elif et == 'L':
            s.cv_lock = not s.cv_lock
        elif et == 'P':
            self._snap_onto(context)
        elif et == 'LEFT_BRACKET':
            s.cv_step = s.cv_step / 2.0
        elif et == 'RIGHT_BRACKET':
            s.cv_step = s.cv_step * 2.0
        else:
            from .drawtools import NUMPAD_CHARS
            ch = getattr(event, "unicode", "") or NUMPAD_CHARS.get(et, "")
            if ch and (self._typed or ch in "0123456789.-+") and not event.ctrl:
                self._typed += ch
        self._refresh(context)
        return {'RUNNING_MODAL'}

    def _end(self, context, cancel):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        overlay.update(on=False, guides=[], sel=set(), axis=None)
        hints.clear(context)
        for a in context.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()
        if self._cage.name not in bpy.data.objects:
            return {'CANCELLED'}
        moved = bool(np.abs(self._world() - self._world(self._start)).max() > 0.0) if self._cage.mode == 'EDIT' else False
        if cancel:
            if moved:
                self._write(self._world(self._start))
            return {'CANCELLED'}
        return {'FINISHED'} if moved else {'CANCELLED'}


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _world_per_px(p):
    """World length of one pixel at the depth of point `p` in the region
    being drawn (1.0 when it can't be told, e.g. behind the view)."""
    ctx = bpy.context
    region, rv3d = ctx.region, ctx.region_data
    if region is None or rv3d is None:
        return 1.0
    p = Vector(tuple(p))
    q = view3d_utils.location_3d_to_region_2d(region, rv3d, p)
    if q is None:
        return 1.0
    w = view3d_utils.region_2d_to_location_3d(region, rv3d, q + Vector((1.0, 0.0)), p)
    return float((w - p).length) or 1.0


def _draw_view():
    if not overlay["on"]:
        return
    import gpu
    try:
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        sel = overlay["sel"]
        rings_ok, rings_bad = [], []
        for g in overlay["guides"]:
            x = Vector(tuple(g["x"]))
            (rings_ok if g["ok"] else rings_bad).append(x)
            if g["k"] not in sel:
                continue
            col = COL_OK if g["ok"] else COL_BAD
            O = np.asarray(g["O"], float)
            if g["kind"] == "point":
                draw._draw_lines([x, Vector(tuple(O))], col, 1.5)
                draw._draw_points([Vector(tuple(O))], COL_GUIDE, 9.0)
            elif g["kind"] == "line":
                L = 2.0 * (float(np.linalg.norm(np.asarray(g["x"]) - O)) or 1.0)
                draw._draw_lines([Vector(tuple(O)), Vector(tuple(O + L * g["d"]))], COL_GUIDE, 2.0)
                foot = onto(g["x"], g)
                draw._draw_lines([x, Vector(tuple(foot))], col, 1.5)
            else:
                foot = onto(g["x"], g)
                n = np.asarray(g["d"], float)
                a = np.cross(n, (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0))
                a /= np.linalg.norm(a) or 1.0
                b = np.cross(n, a)
                # the plane is a direction, not a size: draw it the same on screen
                # wherever the CV is (the gap shows in the line and the %)
                r = _world_per_px(foot) * PLANE_PX
                sq = [Vector(tuple(foot + r * (u * a + v * b))) for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                draw._draw_tris([sq[0], sq[1], sq[2], sq[0], sq[2], sq[3]], COL_PLANE)
                ed = []
                for i in range(4):
                    ed += [sq[i], sq[(i + 1) % 4]]
                draw._draw_lines(ed, COL_GUIDE, 1.5)
                draw._draw_lines([x, Vector(tuple(foot))], col, 1.5)
        draw._draw_points(rings_bad, COL_BAD, 12.0)
        draw._draw_points(rings_ok, COL_OK, 12.0)
        if overlay["axis"]:
            seg = []
            for p, d, L in overlay["axis"]:
                p = np.asarray(p, float)
                seg += [Vector(tuple(p - L * d)), Vector(tuple(p + L * d))]
            draw._draw_lines(seg, COL_AXIS, 1.5)
        # the cage's own CVs on top of the status discs
        draw._draw_cage(bpy.context)
        gpu.state.depth_test_set('LESS_EQUAL')
    except Exception as exc:
        print("NURBS Nudge CVs draw:", exc)


def _draw_px():
    if not overlay["on"]:
        return
    import blf
    ctx = bpy.context
    region, rv3d = ctx.region, ctx.region_data
    if rv3d is None:
        return
    try:
        font = 0
        blf.size(font, 11 * ctx.preferences.view.ui_scale)
        for g in overlay["guides"]:
            if g["k"] not in overlay["sel"]:
                continue
            q = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(tuple(g["x"])))
            if q is None:
                continue
            txt = fmt_err(g, ctx)
            w, h = blf.dimensions(font, txt)
            x, y = q.x + 12, q.y + 8 + (h + 8) * g["row"]
            draw._box(x - 4, y - 4, w + 8, h + 8, (0.05, 0.05, 0.05, 0.8))
            blf.color(font, *(COL_OK if g["ok"] else COL_BAD))
            blf.position(font, x, y, 0)
            blf.draw(font, txt)
    except Exception as exc:
        print("NURBS Nudge CVs labels:", exc)


def draw_panel(lay, context):
    """The Nudge settings, under the cage's line in the Active Object panel."""
    v = _view_settings(context)
    col = lay.column(align=True)
    col.operator(NPROTO_OT_cv_nudge.bl_idname, icon='EMPTY_ARROWS')
    row = col.row(align=True)
    row.prop(v, "cv_step")
    row.prop(v, "cv_sensitivity", text="1 /")
    row = col.row(align=True)
    row.prop(v, "cv_guides", expand=True)
    col.prop(v, "cv_lock")


classes = (NPROTO_OT_cv_nudge,)


def register():
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), 'WINDOW', 'POST_VIEW'))
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_px, (), 'WINDOW', 'POST_PIXEL'))
    cvtweak.register()


def unregister():
    for h in _handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
        except Exception:
            pass
    _handles.clear()
    overlay.update(on=False, guides=[], sel=set(), axis=None)


# the CV Tweak history step (CV edits kept in the history) lives next to the
# tool; importing it here registers its replay with the history stack
from . import cvtweak  # noqa: E402
