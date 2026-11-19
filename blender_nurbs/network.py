"""Curve Network: one surface through a grid of crossing curves.

The inputs fall into two families: every curve of one crosses every curve
of the other (Rhino NetworkSrf, Alias "Square" grown to a net). They are
told apart by who touches whom (the first input and the curves it does not
touch are one family), ordered along each other and turned to run the same
way. The outermost four are the patch's sides, as in Square: surface edges
there take G0 / G1 / G2 to their surface. Every inner curve becomes a
position condition inside the patch, solved with the boundary, so the
surface passes through all the curves.

Parameters: v of the u-curves is where they cross the v-curves (averaged
over the crossings), u likewise; each curve is re-parameterised piecewise
linearly so it meets every curve of the other family at that common u / v,
which makes the conditions agree at the crossings. Curves that miss each
other by a little are met as well as they allow (least squares); a gap
larger than the tolerance is refused with the two curves named.

The CV layout: the inputs' own when they all have one and it holds the net
(Square's matched layout over both families), else the fewest uniform spans
that keep every curve within tolerance (1e-4 of the size; 1e-3 with a trim
edge on the boundary), degree 3, or 5 with a G2 side.
"""

import numpy as np

from . import features, geom, square
from .loft_rails import TOUCH_REL, _contact, _min_spans, _nearest_t, _remap
from .square import CONT

LAST_INFO = {}


def families(sides, tol):
    """(u-family indices, v-family indices): a 2-colouring of the touch graph,
    the first input in the u-family. Raises when it isn't a full network."""
    from .loft_rails import _touch_dist
    n = len(sides)
    touch = [[i != j and _touch_dist(sides[i], sides[j]) <= tol for j in range(n)] for i in range(n)]
    colour = [None] * n
    colour[0] = 0
    todo = [0]
    while todo:
        i = todo.pop()
        for j in range(n):
            if touch[i][j]:
                if colour[j] is None:
                    colour[j] = 1 - colour[i]
                    todo.append(j)
                elif colour[j] == colour[i]:
                    raise RuntimeError(f"{_name(sides[i])} and {_name(sides[j])} touch but run the same way: "
                                       "a network needs two families of curves, each crossing the other")
    if any(c is None for c in colour):
        k = colour.index(None)
        raise RuntimeError(f"{_name(sides[k])} does not touch the other curves")
    A = [i for i in range(n) if colour[i] == 0]
    B = [i for i in range(n) if colour[i] == 1]
    for a in A:
        for b in B:
            if not touch[a][b]:
                raise RuntimeError(f"{_name(sides[a])} and {_name(sides[b])} do not cross: "
                                   "every curve of one family must cross every curve of the other")
    if len(A) < 2 or len(B) < 2:
        raise RuntimeError("a network needs at least two curves each way (four around the outside)")
    return A, B


def _name(sd):
    return f"{sd.src[0]} {sd.src[1]}" if getattr(sd, "src", None) else "a curve"


def _order_and_orient(us, vs, tol):
    """Orders both families along each other and turns every curve to run from
    the first curve of the other family to the last."""
    hit = [_contact(vs[j], us[0])[1] for j in range(len(vs))]      # where each v-curve crosses u-curve 0
    vs = [vs[j] for j in np.argsort(hit)]
    hit = [_contact(us[k], vs[0])[1] for k in range(len(us))]
    us = [us[k] for k in np.argsort(hit)]
    for k, sd in enumerate(us):
        if _contact(sd, vs[0])[0] > _contact(sd, vs[-1])[0]:
            us[k] = sd.reversed()
    for j, sd in enumerate(vs):
        if _contact(sd, us[0])[0] > _contact(sd, us[-1])[0]:
            vs[j] = sd.reversed()
    return us, vs


def crossings(us, vs, tol):
    """a[k][j]: parameter on u-curve k where it crosses v-curve j; b[j][k] the
    same on v-curve j. Raises on a gap above tol."""
    a = np.zeros((len(us), len(vs)))
    b = np.zeros((len(vs), len(us)))
    worst = 0.0
    for k, su in enumerate(us):
        for j, sv in enumerate(vs):
            ta, tb, d = _contact(su, sv)
            if d > tol:
                raise RuntimeError(f"{_name(su)} and {_name(sv)} miss each other by {d:.3g}")
            a[k, j], b[j, k] = ta, tb
            worst = max(worst, d)
    return a, b, worst


def _ends_on(sd, first, last, tol, what):
    for end, other in ((0, first), (-1, last)):
        d = _nearest_t(other, sd.P[end])[1]
        if d > tol:
            raise RuntimeError(f"{_name(sd)} does not end on {_name(other)} (gap {d:.3g}): "
                               f"the outer {what} must close the network")


def build(inputs, p):
    m = int(p.get("samples", 48))
    raw = [_side(o, r, m) for o, r in inputs]
    size = float(np.linalg.norm(np.ptp(np.vstack([s.P for s in raw]), axis=0))) or 1.0
    tol = float(p.get("nw_gap", 0.0) or 0.0) or TOUCH_REL * size * 10
    A, B = families(raw, tol)
    us, vs = _order_and_orient([raw[i] for i in A], [raw[i] for i in B], tol)
    for sd in us:
        _ends_on(sd, vs[0], vs[-1], tol, "curves across")
    for sd in vs:
        _ends_on(sd, us[0], us[-1], tol, "curves along")
    a, b, worst = crossings(us, vs, tol)
    a[:, 0], a[:, -1] = 0.0, 1.0          # (the ends lie on the outer curves)
    b[:, 0], b[:, -1] = 0.0, 1.0
    # common parameters: u_j where v-curve j crosses the u-curves, v_k likewise
    u = a.mean(axis=0)
    v = b.mean(axis=0)
    u = (u - u[0]) / ((u[-1] - u[0]) or 1.0)
    v = (v - v[0]) / ((v[-1] - v[0]) or 1.0)
    if np.any(np.diff(u) < 1e-3) or np.any(np.diff(v) < 1e-3):
        raise RuntimeError("two curves of one family cross the others at the same place")
    for k, sd in enumerate(us):
        _remap(sd, list(a[k]), list(u))
    for j, sd in enumerate(vs):
        _remap(sd, list(b[j]), list(v))
    inner_u, inner_v = us[1:-1], vs[1:-1]
    extra = []
    for k, sd in enumerate(inner_u, 1):
        extra += [(float(t), float(v[k]), P.copy()) for t, P in zip(sd.t, sd.P)]
    for j, sd in enumerate(inner_v, 1):
        extra += [(float(u[j]), float(t), P.copy()) for t, P in zip(sd.t, sd.P)]
    # sides as in Square: bottom = first u-curve, right = last v-curve, top = last u-curve, left = first v-curve
    sides = [us[0], vs[-1], us[-1], vs[0]]
    names = ["bottom", "right", "top", "left"]
    conts = [p.get("nw_cont", "G0")] * 4         # every outer curve that is a surface edge
    ks = [CONT[c] if sd.has_surface and not sd.free else 0 for c, sd in zip(conts, sides)]
    weights = [1.0] * 4
    square.EXTRA_G0[:] = extra
    try:
        surf, info = _solve(sides, ks, weights, int(p.get("nw_degree", 0) or 0), int(p.get("nw_spans_u", 0)),
                            int(p.get("nw_spans_v", 0)), inner_u, inner_v)
        info["curve_gap"] = _curve_gap(surf, extra)       # (in the solve's own parameters: before turning it)
        surf = square._orient_like_neighbours(surf, sides)
    finally:
        square.EXTRA_G0[:] = []
    info["curves"] = (len(us), len(vs))
    info["crossing_gap"] = worst
    info["edge_sides"] = {n: bool(sd.has_surface and not sd.free) for n, sd in zip(names, sides)}
    info["corner_notes"] = list(square.LAST_NOTES)
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return surf


def _side(o, r, m):
    sd = square.sample_side(o, r, m)
    sd.src = (o.name, r)
    if sd.knots is None and not features.is_surface_edge(r):
        from .loft_rails import _curve_layout
        sd.degree, sd.knots = _curve_layout(features.input_curve(o, r))
    return sd


def _curve_gap(surf, extra):
    if not extra:
        return 0.0
    step = max(1, len(extra) // 400)
    return max(float(np.linalg.norm(np.array(surf.evaluate(u, v)) - P)) for u, v, P in extra[::step])


def _matched_layout(sides, ks, inner_u, inner_v):
    """The inputs' own layout over both families (as Loft Rails'), or None."""
    bottom, right, top, left = sides
    out = []
    for group, (ka, kb) in (([bottom, top] + list(inner_u), (ks[3], ks[1])),
                            ([left, right] + list(inner_v), (ks[0], ks[2]))):
        if any(sd.knots is None or sd.degree is None for sd in group):
            return None
        need = 1 + ka + kb
        deg = max(max(sd.degree for sd in group), need)
        if deg > 7:
            return None
        interior = square._merge_knots_mult([(sd.knots, deg - sd.degree + 1) for sd in group])
        while deg + 1 + len(interior) < ka + kb + 2:
            pts = [0.0] + interior + [1.0]
            j = int(np.argmax(np.diff(pts)))
            interior = sorted(interior + [0.5 * (pts[j] + pts[j + 1])])
        out.append((deg, [0.0] * (deg + 1) + interior + [1.0] * (deg + 1)))
    return out


def _solve(sides, ks, weights, degree, spans_u, spans_v, inner_u, inner_v):
    """Square's net with the inner curves as position conditions: the
    lightest layout whose gaps (sides and inner curves), angles and curvature
    stay within tolerance."""
    _, ext = square._k()
    conflicts = square.corner_fades(sides, ks)
    square.LAST_NOTES[:] = square.corner_notes(sides, conflicts)
    size = float(np.linalg.norm(np.ptp(np.vstack([sd.P for sd in sides]), axis=0))) or 1.0

    def make(r):
        S, (Uu, Uv), _, info = r
        nu, nv = S.shape[:2]
        pts = [tuple(map(float, S[i, j])) for i in range(nu) for j in range(nv)]
        info["folds"] = square.net_folds(S)
        return ext.surface_from_cvs(info["degree"][0], info["degree"][1], pts, Uu, Uv), info

    trim_side = any(sd.has_surface and sd.knots is None and not sd.free for sd in sides)
    gap_rel = 1e-3 if trim_side else 1e-4

    def score_of(q_):
        sc = max(q_["gap"] / (gap_rel * q_["size"]), q_["angle"] / 0.1, q_["curv"] / 0.10)
        return sc + 100.0 * q_["folds"] if q_.get("folds") else sc

    tol = gap_rel * size
    cands = []
    lay = None if (degree or spans_u or spans_v) else _matched_layout(sides, ks, inner_u, inner_v)
    if lay is not None:
        (p_, Uu), (q_, Uv) = lay
        try:
            r = square._build_knots(sides, ks, weights, Uu, p_, Uv, q_, const_alpha=True)
            surf, info = make(r)
            info.update(square.measure(surf, sides, ks))
            info["matched"] = True
            sc = score_of(info)
            if sc <= 1.0:
                return surf, info
            cands.append((sc, surf, info))
        except RuntimeError:
            pass
    degrees = [degree] if degree else ([5, 3] if max(ks) >= 2 else [3, 5])
    for p_ in degrees:
        u_rows = [(sd.t, sd.P) for sd in [sides[0], sides[2]] + list(inner_u)]
        v_rows = [(sd.t, sd.P) for sd in [sides[1], sides[3]] + list(inner_v)]
        su0 = spans_u or max(_min_spans(u_rows, p_, tol), len(inner_v) + 2 - p_)
        sv0 = spans_v or max(_min_spans(v_rows, p_, tol), len(inner_u) + 2 - p_)
        su0, sv0 = max(1, su0), max(1, sv0)
        while p_ + su0 < ks[1] + ks[3] + 2:
            su0 += 1
        while p_ + sv0 < ks[0] + ks[2] + 2:
            sv0 += 1
        tries = [(su0, sv0)] if (spans_u and spans_v) else \
            sorted({(su0 + a_, sv0 + b_) for a_ in range(0, 5) for b_ in range(0, 5)
                    if (not spans_u or a_ == 0) and (not spans_v or b_ == 0)},
                   key=lambda s: ((p_ + s[0]) * (p_ + s[1]), s))
        for su, sv in tries[:14]:
            try:
                r = square._build(sides, ks, weights, p_, su, sv)
            except RuntimeError:
                continue
            surf, info = make(r)
            info.update(square.measure(surf, sides, ks))
            sc = score_of(info)
            cands.append((sc, surf, info))
            if sc <= 1.0:
                break
    if not cands:
        raise RuntimeError("too few CVs for the chosen continuity")
    return square._pick_lightest(cands)


@features.builder("network")
def build_network(inputs, p):
    return build(inputs, p)


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import EnumProperty, IntProperty, StringProperty  # noqa: E402

from .ops import PickMixin, _decode_inputs, _encode_inputs, _kernel_ok, _select_only  # noqa: E402

NW_CONT = [("G0", "G0", "Position"), ("G1", "G1", "Tangent"), ("G2", "G2", "Curvature")]
NW_DEG = [("0", "Auto", "Degree 5 with G2, else 3 (or the curves' own layout)"), ("3", "3", ""), ("5", "5", "")]


class NPROTO_OT_network(PickMixin, bpy.types.Operator):
    """One surface through a network of crossing curves (two families, every
    curve of one crossing every curve of the other). The outer four are its
    sides: surface edges there take G0 / G1 / G2. Select the curves, or click
    them in any order and press Enter"""
    bl_idname = "nproto.network"
    bl_label = "Curve Network"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 4
    pick_max = 256
    finish_on_min = False
    extra_help = "   (all the curves, any order: two families crossing each other)"
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    nw_cont: EnumProperty(name="Edges", items=NW_CONT, default="G0",
                          description="Continuity to the surfaces whose edges are among the outer four curves")
    nw_degree: EnumProperty(name="Degree", items=NW_DEG, default="0")
    nw_spans_u: IntProperty(name="Spans U", default=0, min=0, max=48, description="0 = auto")
    nw_spans_v: IntProperty(name="Spans V", default=0, min=0, max=48, description="0 = auto")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not self.inputs and not event.alt:
            crv = [o for o in context.selected_objects if o.type == 'CURVE' and len(o.data.splines)]
            ins = [(o, f"S{i}") for o in crv for i in range(len(o.data.splines))]
            if len(ins) >= 4:
                self.inputs = _encode_inputs(ins)
                return self.execute(context)
        return self.start_picking(context, event)

    def _params(self):
        return {"nw_cont": self.nw_cont, "nw_degree": int(self.nw_degree), "nw_spans_u": self.nw_spans_u,
                "nw_spans_v": self.nw_spans_v}

    def draw(self, context):
        lay = self.layout
        inf = LAST_INFO
        row = lay.row(align=True)
        row.label(text="Edges")
        sub = row.row(align=True)
        sub.enabled = any(inf.get("edge_sides", {}).values()) if inf else True
        sub.prop(self, "nw_cont", expand=True)
        lay.prop(self, "nw_degree")
        row = lay.row(align=True)
        row.prop(self, "nw_spans_u")
        row.prop(self, "nw_spans_v")
        if inf:
            lay.label(text=f"{inf['curves'][0]} x {inf['curves'][1]} curves · {inf['cvs'][0]} x {inf['cvs'][1]} CVs · "
                           f"degree {inf['degree'][0]}", icon='MESH_GRID')
            lay.label(text=f"gap to the curves {max(inf.get('gap', 0), inf.get('curve_gap', 0)):.1e} · "
                           f"angle {inf.get('angle', 0):.3f}° · curvature {100 * inf.get('curv', 0):.1f}%",
                      icon='DRIVER_DISTANCE')
            for note in inf.get("corner_notes", []):
                lay.label(text=note, icon='ERROR')

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            inputs = _decode_inputs(self.inputs)
            if len(inputs) < 4:
                raise ValueError("pick at least four curves (two each way)")
            params = self._params()
            g = features.run("network", inputs, params)
            from . import store
            obj = store.new_object(context, "Network", g)
            if features.history_enabled(context):
                features.record(obj, "network", inputs, params)
        except Exception as exc:
            self.report({'ERROR'}, f"Curve Network: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        inf = LAST_INFO
        self.report({'INFO'}, f"Curve Network: {inf['curves'][0]} x {inf['curves'][1]} curves, "
                              f"{inf['cvs'][0]}x{inf['cvs'][1]} CVs, gap {max(inf.get('gap', 0), inf.get('curve_gap', 0)):.1e}")
        return {'FINISHED'}


classes = (NPROTO_OT_network,)
