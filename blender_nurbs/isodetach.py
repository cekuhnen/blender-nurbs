"""Detach at Isoparm: cut an untrimmed NURBS surface along isoparms (constant
u and / or constant v lines) into untrimmed surfaces of their own -- Alias'
Detach, Rhino's Split at Isocurve + Shrink.

Exact: the cut knots are inserted until the surface breaks there (Boehm, in
homogeneous space, so rational surfaces stay exact) and the CV net is sliced
at the breaks. Every piece is the original surface over its sub-domain: same
degree, same weights, no trims, its own CVs. A closed direction (a revolve, a
tube) is first opened: one cut there moves the seam to the cut and leaves one
piece; more cuts give that many pieces (a cut at the seam itself counts).

The pieces are a history feature ("iso_detach", one per piece) when
Construction History is on: the source is kept hidden and each piece follows
it. Cut positions are fractions of the source's domain, so they hold when the
source is rebuilt with other knots."""

import json

import bpy
import numpy as np
from bpy.props import EnumProperty, StringProperty
from mathutils import Vector

from . import draw, features, geom, hints, store

EPS = 1e-9          # cut positions closer than this (fraction of the domain) are the same
SNAP_PX = 12        # a cut within this many pixels of a span line lands on it
COL_U, COL_V = 0, 3  # pick colours (draw.COL_PICKS): constant-u cuts cyan, constant-v amber


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


# --------------------------------------------------------------------------
# Exact splitting of a homogeneous CV net (axis 0 is the direction cut)
# --------------------------------------------------------------------------

def net(surf):
    """(H (nu, nv, 4) homogeneous CVs, knots_u, knots_v, degree_u, degree_v, rational)."""
    _, ext = _k()
    d = ext.surface_cvs(surf)
    nu, nv = int(d["count_u"]), int(d["count_v"])
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    W = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    H = np.concatenate([P * W[..., None], W[..., None]], axis=2)
    return (H, [float(k) for k in d["knots_u"]], [float(k) for k in d["knots_v"]],
            int(d["degree_u"]), int(d["degree_v"]), bool(d.get("weights")))


def _mult(U, t, tol):
    return sum(1 for k in U if abs(k - t) <= tol)


def insert(G, U, p, t, times):
    """Boehm knot insertion of t, `times` times, along axis 0 of G."""
    G = np.asarray(G, float)
    U = list(U)
    for _ in range(times):
        k = max(i for i in range(len(U) - 1) if U[i] <= t < U[i + 1])
        n = G.shape[0]
        Q = np.empty((n + 1,) + G.shape[1:])
        Q[:k - p + 1] = G[:k - p + 1]
        Q[k + 1:] = G[k:]
        for i in range(k - p + 1, k + 1):
            den = U[i + p] - U[i]
            a = (t - U[i]) / den if den else 0.0
            Q[i] = (1.0 - a) * G[i - 1] + a * G[i]
        G, U = Q, U[:k + 1] + [t] + U[k + 1:]
    return G, U


def split(G, U, p, t, tol):
    """The curve rows G (knots U, degree p) cut at t: (left, right), each (G, U),
    clamped at the cut. t is snapped onto a knot within tol."""
    near = [k for k in U if abs(k - t) <= tol]
    if near:
        t = near[0]
    s = _mult(U, t, tol)
    G, U = insert(G, U, p, t, max(0, p - s))
    U = [t if abs(k - t) <= tol else k for k in U]
    k = max(i for i, x in enumerate(U) if x == t)
    L = (G[:k - p + 1], U[:k + 1] + [t])
    R = (G[k - p:], [t] + U[k - p + 1:])
    return L, R


def clamp(G, U, p, tol):
    """A clamped copy (end knots of multiplicity p + 1) of an unclamped (periodic) row set."""
    a, b = U[p], U[len(U) - p - 1]
    if any(abs(x - a) > tol for x in U[:p + 1]):
        G, U = split(G, U, p, a, tol)[1]
    if any(abs(x - b) > tol for x in U[-p - 1:]):
        G, U = split(G, U, p, b, tol)[0]
    return G, U


def _join_wrap(R, L, p, tol):
    """Piece R (it ends at the seam) and piece L (it starts there) as one:
    the seam becomes an inner knot of multiplicity p. Exact when the seam
    rows are the same homogeneous points up to one factor (rescaled here)."""
    GR, UR = R
    GL, UL = L
    a, b = GR[-1], GL[0]
    w = a[..., 3] / b[..., 3]
    if np.ptp(w) > 1e-9 * abs(w).max():
        raise ValueError("the seam's weights differ along it: it can't be opened exactly")
    GL = GL * w.mean()
    gap = np.abs(GR[-1] - GL[0]).max()
    if gap > 1e-7 * (np.abs(GR).max() or 1.0):
        raise ValueError("the surface is not closed in this direction")
    G = np.concatenate([GR[:-1], [0.5 * (GR[-1] + GL[0])], GL[1:]], axis=0)
    off = UR[-1] - UL[0]
    U = UR[:-(p + 1)] + [UR[-1]] * p + [x + off for x in UL[p + 1:]]
    return G, U


def cut_rows(G, U, p, fracs, closed):
    """Pieces [(G, U)] of the rows G cut at domain fractions `fracs` (a closed
    direction: opened first, the piece across the seam joined back)."""
    tol = 1e-12 * ((U[-1] - U[0]) or 1.0)
    G, U = clamp(G, U, p, tol)
    a, b = U[0], U[-1]
    tol = 1e-12 * (b - a)
    fr = sorted({round(float(f), 12) for f in fracs})
    seam = closed and any(f <= EPS or f >= 1.0 - EPS for f in fr)
    ts = [a + f * (b - a) for f in fr if EPS < f < 1.0 - EPS]
    if not ts and not seam:
        return [(G, U)]
    pieces, rest = [], (G, U)
    for t in ts:
        left, rest = split(rest[0], rest[1], p, t, tol)
        pieces.append(left)
    pieces.append(rest)
    if closed and not seam:
        pieces = [_join_wrap(pieces[-1], pieces[0], p, tol)] + pieces[1:-1]
    return pieces


def _normalized(U):
    a, b = U[0], U[-1]
    return [(x - a) / (b - a) for x in U]


def detach(surf, cuts_u=(), cuts_v=(), closed=None):
    """The untrimmed surfaces surf is cut into along constant-u lines at
    fractions cuts_u and constant-v lines at cuts_v of its domain, in grid
    order (u outer, v inner). Each piece's cut directions run over 0..1."""
    _, ext = _k()
    H, Ku, Kv, pu, pv, rational = net(surf)
    if closed is None:
        closed = store._periodic(surf)
    cu, cv = bool(cuts_u), bool(cuts_v)
    cols = cut_rows(H, Ku, pu, cuts_u, closed[0]) if cu else [(H, Ku)]
    out = []
    for G, U in cols:
        rows = cut_rows(G.transpose(1, 0, 2), Kv, pv, cuts_v, closed[1]) if cv else [(G.transpose(1, 0, 2), Kv)]
        for Gv, V in rows:
            A = Gv.transpose(1, 0, 2)
            w = A[..., 3]
            pts = [tuple(map(float, x)) for x in (A[..., :3] / w[..., None]).reshape(-1, 3)]
            ku = _normalized(U) if cu else U
            kv = _normalized(V) if cv else V
            wl = [float(x) for x in w.reshape(-1)] if rational or np.abs(w - 1.0).max() > 1e-12 else None
            out.append(ext.surface_from_cvs(pu, pv, pts, ku, kv, wl))
    return out


def piece_domains(cuts_u, cuts_v, closed):
    """[(fu0, fu1, fv0, fv1)] the source-domain fractions each piece covers,
    in detach() order (a piece across a seam has fu1 > 1: it wraps)."""
    def spans(fr, cl):
        ts = sorted({round(float(f), 12) for f in fr if EPS < f < 1.0 - EPS})
        seam = cl and any(f <= EPS or f >= 1.0 - EPS for f in fr)
        edges = [0.0] + ts + [1.0]
        sp = list(zip(edges[:-1], edges[1:]))
        if fr and cl and not seam:
            sp = [(ts[-1], 1.0 + ts[0])] + sp[1:-1]
        return sp
    return [(a, b, c, d) for a, b in spans(cuts_u, closed[0]) for c, d in spans(cuts_v, closed[1])]


# --------------------------------------------------------------------------
# History feature: one piece of the source's detach
# --------------------------------------------------------------------------

@features.builder("iso_detach")
def build_iso_detach(inputs, p):
    obj = inputs[0][0]
    if not store.is_natural(obj):
        raise RuntimeError(f"{obj.name} is trimmed: Detach at Isoparm needs an untrimmed surface")
    pieces = detach(store.get_surfaces(obj)[0], p.get("cuts_u", []), p.get("cuts_v", []))
    i = int(p.get("piece", 0))
    if not 0 <= i < len(pieces):
        raise RuntimeError(f"{obj.name} now detaches into {len(pieces)} pieces, not {i + 1}")
    return pieces[i]


def apply(context, obj, cuts_u, cuts_v, op=None):
    """Detach obj (one untrimmed face) at the cuts. Returns the piece objects
    (with history: new objects following the hidden source; without: the
    first piece stays obj itself)."""
    surf = store.get_surfaces(obj)[0]
    pieces = detach(surf, cuts_u, cuts_v)
    if len(pieces) < 2 and not (cuts_u and store._periodic(surf)[0]) and not (cuts_v and store._periodic(surf)[1]):
        raise ValueError("the cuts don't divide the surface")
    M = geom.world_matrix(obj)
    keep = features.history_enabled(context)
    made = []
    if not keep:
        if features.has_history(obj):
            features.clear_history(obj)
            if op is not None:
                op.report({'INFO'}, f"{obj.name}: construction history removed")
        for key in (store.PROP_UNTRIM, store.PROP_UNTRIM + "_natural"):
            if key in obj:
                del obj[key]
        store.set_geometry(obj, pieces[0], context)
        made.append(obj)
    params = {"cuts_u": [float(f) for f in cuts_u], "cuts_v": [float(f) for f in cuts_v]}
    for i, s in enumerate(pieces):
        if not keep and i == 0:
            continue
        o = store.new_object(context, f"{obj.name}_detach{i + 1}", s, M)
        for m in obj.data.materials:
            o.data.materials.append(m)
        if keep:
            features.record(o, "iso_detach", [(obj, "SURF")], {**params, "piece": i})
        made.append(o)
    if keep:
        features.keep_inputs(context, [obj])
    return made


# --------------------------------------------------------------------------
# Picking isoparms
# --------------------------------------------------------------------------

def _dir_fracs(surf, d):
    """Interior span-line fractions of direction d ('U' / 'V'), plus the seam (0) of a closed one."""
    _, ext = _k()
    c = ext.surface_cvs(surf)
    U = [float(k) for k in (c["knots_u"] if d == "U" else c["knots_v"])]
    p = int(c["degree_u"] if d == "U" else c["degree_v"])
    a, b = U[p], U[len(U) - p - 1]
    out = sorted({round((k - a) / (b - a), 12) for k in U if a < k < b})
    if store._periodic(surf)[0 if d == "U" else 1]:
        out = [0.0] + out
    return out


def iso_points(surf, d, f, n=64):
    """Local-space points of the isoparm at fraction f of direction d."""
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    ts = np.linspace(0.0, 1.0, n)
    if d == "U":
        u = u0 + (u1 - u0) * min(max(f, 0.0), 1.0)
        return np.array([surf.evaluate(u, v0 + (v1 - v0) * t) for t in ts], float)
    v = v0 + (v1 - v0) * min(max(f, 0.0), 1.0)
    return np.array([surf.evaluate(u0 + (u1 - u0) * t, v) for t in ts], float)


def _screen(context, P):
    """Region pixel coordinates (n, 2) of world points P (n, 3); NaN behind the eye."""
    M = np.array(context.region_data.perspective_matrix, float)
    Q = np.c_[P, np.ones(len(P))] @ M.T
    w = Q[:, 3:4]
    with np.errstate(divide='ignore', invalid='ignore'):
        n = np.where(w > 1e-9, Q[:, :2] / w, np.nan)
    r = context.region
    return np.c_[(n[:, 0] + 1.0) * 0.5 * r.width, (n[:, 1] + 1.0) * 0.5 * r.height]


class NPROTO_OT_detach_iso(bpy.types.Operator):
    """Cut an untrimmed surface along isoparms into untrimmed surfaces of its
    own (exact: knot insertion, no trims). Move over the surface to see the
    isoparm, click to add a cut (it snaps to span lines; Shift: free), Tab
    switches U / V, Backspace takes the last cut back, Enter detaches. A
    closed direction opens at a single cut. With history each piece follows
    the source, which is kept hidden"""
    bl_idname = "nproto.detach_iso"
    bl_label = "Detach at Isoparm"
    bl_options = {'REGISTER', 'UNDO'}

    target: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    cuts_u: StringProperty(name="Cuts U", default="[]", description="Constant-u cuts: fractions of the u domain (JSON list)")
    cuts_v: StringProperty(name="Cuts V", default="[]", description="Constant-v cuts: fractions of the v domain (JSON list)")
    iso_dir: EnumProperty(name="Direction", items=[("U", "U", "Constant-u lines"), ("V", "V", "Constant-v lines")],
                          default="U")

    def draw(self, context):
        lay = self.layout
        for d, txt in (("U", self.cuts_u), ("V", self.cuts_v)):
            fr = json.loads(txt or "[]")
            if fr:
                lay.label(text=f"{d}: " + ", ".join(f"{f:.4g}" for f in fr), icon='UV_EDGESEL')

    # ---- run
    def _cuts(self):
        return json.loads(self.cuts_u or "[]"), json.loads(self.cuts_v or "[]")

    def execute(self, context):
        from .ops import _kernel_ok, _select_only
        if not _kernel_ok(self):
            return {'CANCELLED'}
        obj = bpy.data.objects.get(self.target) if self.target else context.active_object
        if obj is None or not store.is_nurbs(obj):
            self.report({'WARNING'}, "Detach at Isoparm: select a NURBS surface")
            return {'CANCELLED'}
        if not store.is_natural(obj):
            self.report({'WARNING'}, f"{obj.name} is trimmed or has several faces: Detach at Isoparm needs one "
                                     "untrimmed surface (for a trimmed one: Iso Curve, then Split On Surface)")
            return {'CANCELLED'}
        cu, cv = self._cuts()
        if not cu and not cv:
            self.report({'WARNING'}, "Detach at Isoparm: no cuts")
            return {'CANCELLED'}
        try:
            made = apply(context, obj, cu, cv, self)
        except Exception as exc:
            self.report({'ERROR'}, f"Detach at Isoparm failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, made)
        self.report({'INFO'}, f"Detached {obj.name} into {len(made)} surface(s)")
        return {'FINISHED'}

    # ---- interactive
    def invoke(self, context, event):
        from .ops import _kernel_ok
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D' or context.region_data is None:
            return self.execute(context)
        self._obj, self._surf, self._foot = None, None, None
        self._cuts_l = []                 # [(dir, fraction, world polyline)]
        self._hover = None                # (dir, fraction, snapped?)
        self._snap_cache = {}
        a = context.active_object
        self._pre = a is not None and store.is_nurbs(a) and store.is_natural(a) and a.select_get()
        if self._pre:
            self._set_target(a)
        draw.pick_overlay["picked"], draw.pick_overlay["colors"] = [], []
        draw.pick_overlay["hover"] = None
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {'RUNNING_MODAL'}

    def _set_target(self, obj):
        self._obj = obj
        self._surf = store.get_surfaces(obj)[0]
        self._foot = store._make_foot(self._surf)
        self._M = np.array(geom.world_matrix(obj), float)
        self._snap_cache = {}

    def _world(self, P):
        return P @ self._M[:3, :3].T + self._M[:3, 3]

    def _snap_lines(self, d):
        """[(fraction, world points)] of direction d's span lines (cached)."""
        if d not in self._snap_cache:
            self._snap_cache[d] = [(f, self._world(iso_points(self._surf, d, f, 32)))
                                   for f in _dir_fracs(self._surf, d)[:200]]
        return self._snap_cache[d]

    def _pick(self, context, event):
        """(dir, fraction, snapped) of the isoparm under the mouse, or None."""
        from .ops_trim import _ray_nurbs
        obj, loc, _, _ = _ray_nurbs(context, event)
        if obj is None:
            return None
        if self._obj is None:
            if not store.is_natural(obj):
                return ("REFUSED", obj)
            self._set_target(obj)
        if obj != self._obj:
            return None
        q = np.linalg.solve(self._M, np.r_[np.array(loc, float), 1.0])[:3]
        u, v = self._foot(q)
        (u0, u1), (v0, v1) = self._surf.u_domain(), self._surf.v_domain()
        d = self.iso_dir
        f = (u - u0) / (u1 - u0) if d == "U" else (v - v0) / (v1 - v0)
        f = min(max(f, 0.0), 1.0)
        if event.shift:
            return (d, f, False)
        m = np.array([event.mouse_region_x, event.mouse_region_y], float)
        best = None
        for g, W in self._snap_lines(d):
            S = _screen(context, W)
            dist = np.nanmin(np.linalg.norm(S - m, axis=1)) if np.isfinite(S).any() else np.inf
            if dist < SNAP_PX and (best is None or dist < best[0]):
                best = (dist, g)
        if best is not None:
            return (d, best[1], True)
        return (d, f, False)

    def _polyline(self, d, f):
        return [Vector(p) for p in self._world(iso_points(self._surf, d, f))]

    def _refresh(self, context):
        draw.pick_overlay["picked"] = [pl for _, _, pl in self._cuts_l]
        draw.pick_overlay["colors"] = [COL_U if d == "U" else COL_V for d, _, _ in self._cuts_l]
        h = self._hover
        draw.pick_overlay["hover"] = self._polyline(h[0], h[1]) if h and h[0] in "UV" else None
        draw.pick_overlay["refused"] = None
        if h and h[0] == "REFUSED":
            M = h[1].matrix_world
            pts = []
            for e in store.display(h[1]).get("edges", []):
                pts += [M @ Vector(p) for p in e]
            draw.pick_overlay["refused"] = pts or None
        self._status(context)
        if context.area is not None:
            context.area.tag_redraw()

    def _status(self, context):
        nu = sum(1 for d, _, _ in self._cuts_l if d == "U")
        nv = len(self._cuts_l) - nu
        h = self._hover
        state = [self._obj.name if self._obj is not None else "click a surface",
                 f"cuts U {nu} · V {nv}"]
        if h and h[0] in "UV":
            state.append(f"{h[0]} {h[1]:.4f}" + (" (span line)" if h[2] else ""))
        elif h and h[0] == "REFUSED":
            state.append(f"{h[1].name} is trimmed: not detachable")
        rows = [("Cut", [hints.key("Click", "add a cut"), hints.cycle("Tab", ["U", "V"], self.iso_dir),
                         hints.key("Shift", "no snapping to span lines"),
                         hints.key("Backspace", "take the last cut back", on=None if self._cuts_l else False)]),
                ("Detach", [hints.key("Enter", "detach", on=None if self._cuts_l else False), hints.key("Esc", "cancel")])]
        hints.show(context, self.bl_label, state, rows,
                   mouse=[('MOUSE_LMB', "Add Cut"), ('EVENT_TAB', "U / V"), ('EVENT_RETURN', "Detach"),
                          ('EVENT_ESC', "Cancel")])

    def _end(self, context):
        draw.pick_overlay["picked"], draw.pick_overlay["colors"] = [], []
        draw.pick_overlay["hover"] = draw.pick_overlay["refused"] = None
        hints.clear(context)
        if context.area is not None:
            context.area.header_text_set(None)
            context.area.tag_redraw()

    def modal(self, context, event):
        from .drawtools import PASS_KEYS
        if hints.handle(event):
            return {'RUNNING_MODAL'}
        if event.type in PASS_KEYS:           # in-between moves too: Blender's drags (gizmos) are built from them
            return {'PASS_THROUGH'}
        if event.type in {'MOUSEMOVE', 'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            self._hover = self._pick(context, event)
            self._refresh(context)
        elif event.type in {'TAB', 'U', 'V'} and event.value == 'PRESS':
            self.iso_dir = event.type if event.type in {'U', 'V'} else ("V" if self.iso_dir == "U" else "U")
            self._hover = self._pick(context, event)
            self._refresh(context)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            h = self._pick(context, event)
            if h and h[0] in "UV":
                d, f, _ = h
                ends = not store._periodic(self._surf)[0 if d == "U" else 1]
                if ends and not (EPS < f < 1.0 - EPS):
                    self.report({'WARNING'}, "That is the surface's edge: nothing to cut there")
                elif any(c[0] == d and abs(c[1] - f) <= 1e-6 for c in self._cuts_l):
                    pass
                else:
                    self._cuts_l.append((d, f, self._polyline(d, f)))
            self._hover = h
            self._refresh(context)
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS' and self._cuts_l:
            self._cuts_l.pop()
            if not self._cuts_l and not self._pre:
                self._obj = None          # no cuts left: another surface may be picked
            self._refresh(context)
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if not self._cuts_l:
                self.report({'WARNING'}, "Click on a surface to add a cut first")
                return {'RUNNING_MODAL'}
            self.target = self._obj.name
            self.cuts_u = json.dumps([f for d, f, _ in self._cuts_l if d == "U"])
            self.cuts_v = json.dumps([f for d, f, _ in self._cuts_l if d == "V"])
            self._end(context)
            return self.execute(context)
        elif event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}


classes = (NPROTO_OT_detach_iso,)
