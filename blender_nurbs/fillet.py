"""Fillet (Create Curves > Between Curves): round the corner between two curves.

Click the first curve, then the second, near the corner to round: the parts
you click are the parts kept. The radius follows the mouse along the corner's
bisector, or type it and press Enter. The exact arc goes in tangent to both;
the curves are trimmed back to the tangent points (T: trim on / off) and, with
Join (J), the three become one curve. Two segments of one polyline (the same
spline) are rounded in place: the corner between them goes, the arc joins
them, and it stays one curve.

Geometry: the curves' exact Bezier spans (curvejoin / curvetrim). The tangent
lines at the clicked points give the first guess (an exact corner circle for
straight lines); Newton then solves P1(g1) + r N1 = P2(g2) + r N2 on the
curves themselves, N the in-plane normal towards the arc's centre. The arc is
a rational quadratic (exact), the pieces are de Casteljau splits (exact).
"""

import json
import math

import numpy as np

from . import curvejoin, curvetrim, geom
from .drawshapes import _unit
from .drawtools import arc_bezier, arc_circle

# --------------------------------------------------------------------------
# Geometry on span lists (homogeneous Bezier rows, global parameter g = span + t)
# --------------------------------------------------------------------------


def span_eval(spans, g):
    """(point, first derivative) at global parameter g."""
    n = len(spans)
    g = min(max(float(g), 0.0), float(n))
    k = min(int(g), n - 1)
    return curvetrim.eval_d(spans[k], g - k)


def nearest_g(spans, Q):
    """Global parameter of the point of the span list nearest Q."""
    pts, gs = curvetrim.sample_spans(spans, 48)
    Q = np.asarray(Q, float)
    g = float(gs[int(np.argmin(np.linalg.norm(pts - Q, axis=1)))])
    n = float(len(spans))
    for _ in range(60):
        P, D = span_eval(spans, g)
        dd = float(D @ D)
        if dd < 1e-30:
            break
        step = float((P - Q) @ D) / dd
        g2 = min(max(g - step, 0.0), n)
        if abs(g2 - g) < 1e-15:
            break
        g = g2
    return g


def _tangent_lines_guess(Q1, T1, Q2, T2, n, r):
    """The corner of the two tangent lines (through the clicked points), the
    rays towards the clicked points, the bisector and the line fillet's tangent
    points for radius r (None: the lines are parallel)."""
    M = np.stack([T1, -T2], axis=1)
    ab, *_ = np.linalg.lstsq(M, Q2 - Q1, rcond=None)
    X = Q1 + ab[0] * T1
    e1 = T1 if float((Q1 - X) @ T1) >= 0 else -T1
    e2 = T2 if float((Q2 - X) @ T2) >= 0 else -T2
    c = min(max(float(e1 @ e2), -1.0), 1.0)
    half = 0.5 * math.acos(c)
    if half < 1e-9 or math.pi / 2 - half < 1e-9:
        return None
    u = _unit(e1 + e2)
    return {"X": X, "e1": e1, "e2": e2, "u": u, "half": half,
            "t1": X + (r / math.tan(half)) * e1, "t2": X + (r / math.tan(half)) * e2,
            "C": X + (r / math.sin(half)) * u}


def radius_from(guess, P):
    """A radius from the mouse: its foot on the corner's bisector is the centre."""
    s = float((np.asarray(P, float) - guess["X"]) @ guess["u"])
    return max(s, 1e-12) * math.sin(guess["half"])


def solve_fillet(sp1, sp2, Q1, Q2, r, n=None):
    """The fillet of radius r between two curves (world span lists), in the
    corner the clicked points Q1, Q2 pick: {g1, g2, t1, t2, centre, radius,
    normal, middle} or None (parallel, or the arc doesn't fit on the curves)."""
    Q1, Q2 = np.asarray(Q1, float), np.asarray(Q2, float)
    g1, g2 = nearest_g(sp1, Q1), nearest_g(sp2, Q2)
    P1, D1 = span_eval(sp1, g1)
    P2, D2 = span_eval(sp2, g2)
    T1, T2 = _unit(D1), _unit(D2)
    if n is None:
        n = np.cross(T1, T2)
        if float(np.linalg.norm(n)) < 1e-9:
            return None
    n = _unit(n)
    guess = _tangent_lines_guess(P1, T1, P2, T2, n, float(r))
    if guess is None:
        return None
    # the centre's side of each curve: towards the other curve's kept part
    s1 = 1.0 if float(guess["e2"] @ np.cross(n, T1)) >= 0 else -1.0
    s2 = 1.0 if float(guess["e1"] @ np.cross(n, T2)) >= 0 else -1.0
    g1, g2 = nearest_g(sp1, guess["t1"]), nearest_g(sp2, guess["t2"])
    size = max(float(np.ptp(np.vstack([P1, P2, guess["X"]]), axis=0).max()), float(r), 1e-9)

    def F(a, b):
        A, DA = span_eval(sp1, a)
        B, DB = span_eval(sp2, b)
        return (A + r * s1 * _unit(np.cross(n, DA))) - (B + r * s2 * _unit(np.cross(n, DB)))

    n1, n2 = float(len(sp1)), float(len(sp2))
    for _ in range(80):
        f0 = F(g1, g2)
        if float(np.linalg.norm(f0)) < 1e-13 * size:
            break
        h = 1e-7
        J = np.stack([(F(g1 + h, g2) - F(g1 - h, g2)) / (2 * h), (F(g1, g2 + h) - F(g1, g2 - h)) / (2 * h)], axis=1)
        step, *_ = np.linalg.lstsq(J, -f0, rcond=None)
        lim = 0.25
        step = np.clip(step, -lim, lim)
        g1, g2 = min(max(g1 + step[0], 0.0), n1), min(max(g2 + step[1], 0.0), n2)
    if float(np.linalg.norm(F(g1, g2))) > 1e-9 * size:
        return None                         # the arc doesn't fit (it would leave a curve's end)
    A, DA = span_eval(sp1, g1)
    B, DB = span_eval(sp2, g2)
    C = A + r * s1 * _unit(np.cross(n, DA))
    mid = C + r * _unit(_unit(A - C) + _unit(B - C)) if np.linalg.norm(_unit(A - C) + _unit(B - C)) > 1e-12 else None
    if mid is None:
        return None
    return {"g1": g1, "g2": g2, "t1": A, "t2": B, "centre": C, "radius": float(r), "normal": n, "middle": mid}


def arc_spans(S, M, E):
    """The exact arc S -> M -> E as homogeneous rational quadratic spans."""
    pts, w = arc_bezier(S, M, E)
    out = []
    for k in range(0, len(pts) - 1, 2):
        rows = [np.append(np.asarray(pts[k + j], float) * w[k + j], w[k + j]) for j in range(3)]
        out.append(np.array(rows))
    return out


def _toward_arc(spans, g, t, middle):
    """Does the curve, running forward (rising g) from its tangent point t, head
    into the arc's side -- the corner the arc replaces?"""
    D = span_eval(spans, g)[1]
    return float(D @ (np.asarray(middle, float) - np.asarray(t, float))) > 0.0


def _piece_keeping(spans, g_cut, g_keep, closed):
    """(span list, starts at the cut) of the piece of `spans` cut at g_cut that holds g_keep."""
    if g_cut <= 1e-9 or g_cut >= len(spans) - 1e-9:
        return [s.copy() for s in spans], g_cut <= 1e-9
    pieces = curvetrim.split_spans(spans, [g_cut], closed)
    k = curvetrim.piece_at(pieces, g_keep, len(spans))
    sp, a, b = pieces[k]
    return sp, abs(a - g_cut) < 1e-9


def fillet_plan(c1, c2, Q1, Q2, r, same=False, closed=False):
    """Everything the fillet makes (world spans): {sol, keep1, keep2 (span
    lists), arc (spans), joined (one span list: kept 1 -> arc -> kept 2)} or None.
    same: both picks on one spline (c1 is c2): the corner piece between the
    tangent points goes."""
    sol = solve_fillet(c1, c2, Q1, Q2, r)
    if sol is None:
        return None
    arc = arc_spans(sol["t1"], sol["middle"], sol["t2"])
    if same:
        cuts = sorted(g for g in (sol["g1"], sol["g2"]) if 1e-9 < g < len(c1) - 1e-9)
        pieces = curvetrim.split_spans(c1, cuts, closed)
        # the corner piece: the one leaving the first tangent point towards the arc
        fwd = _toward_arc(c1, sol["g1"], sol["t1"], sol["middle"])
        at = lambda a, b: abs(a - b) < 1e-9 or (closed and abs(abs(a - b) - len(c1)) < 1e-9)
        corner = [k for k, (_, a, b) in enumerate(pieces) if at(a if fwd else b, sol["g1"])]
        if len(corner) != 1:
            return None
        rest = curvetrim.remove_piece(pieces, corner[0], closed)
        end = lambda sp, i: curvejoin._cart(sp[i])[i]
        close = lambda a, b: float(np.linalg.norm(a - b)) <= 1e-7 * max(1.0, float(np.linalg.norm(a)))
        if closed:
            # one piece left: it runs from one tangent point round to the other; the arc closes it
            piece = rest[0] if rest else []
            a = arc if piece and close(end(piece, -1), end(arc, 0)) else curvejoin._reverse(arc)
            joined = piece + a
        else:
            before, after = (rest + [[], []])[:2]
            if before and close(end(before, -1), end(arc, -1)):
                arc = curvejoin._reverse(arc)
            joined = before + arc + after
        return {"sol": sol, "joined": joined, "arc": arc, "same": True}
    # the part kept runs AWAY from the arc (wherever along it you clicked: the
    # click picks the corner, not the cut -- a big radius can put the tangent
    # points past the clicks)
    n1, n2 = len(c1), len(c2)
    away1 = 0.5 * sol["g1"] if _toward_arc(c1, sol["g1"], sol["t1"], sol["middle"]) else 0.5 * (sol["g1"] + n1)
    away2 = 0.5 * sol["g2"] if _toward_arc(c2, sol["g2"], sol["t2"], sol["middle"]) else 0.5 * (sol["g2"] + n2)
    keep1, starts1 = _piece_keeping(c1, sol["g1"], away1, False)
    keep2, starts2 = _piece_keeping(c2, sol["g2"], away2, False)
    k1 = curvejoin._reverse(keep1) if starts1 else keep1          # ends at the tangent point
    k2 = keep2 if starts2 else curvejoin._reverse(keep2)          # starts at the tangent point
    return {"sol": sol, "keep1": keep1, "keep2": keep2, "arc": arc, "joined": k1 + arc + k2, "same": False}


def polyline(spans, per=16):
    return [p for p in curvetrim.sample_spans(spans, per)[0]] if spans else []


# --------------------------------------------------------------------------
# Blender side
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty, FloatProperty, StringProperty  # noqa: E402
from mathutils import Vector  # noqa: E402

from . import draw, hints, features  # noqa: E402
from .drawshapes import _ShapeTool  # noqa: E402
from .drawtools import SNAP_PX, _select_new  # noqa: E402

COL_REMOVED = (0.55, 0.55, 0.55, 0.7)


def _spline_world(obj, i):
    """(local spans, world spans, closed) of spline i of a curve object."""
    sp = obj.data.splines[i]
    loc = curvejoin.spline_spans(sp)
    world = curvejoin.transform_spans(loc, geom.world_matrix(obj))
    return loc, world, curvetrim.is_closed(loc, sp.use_cyclic_u)


def _to_local(spans_world, obj):
    Minv = np.linalg.inv(np.array(geom.world_matrix(obj), float))
    return curvejoin.transform_spans(spans_world, Minv)


def apply_fillet(context, picks, radius, join=False, trim=True, report=None):
    """Make the fillet. picks: [(object name, spline index, world point)] x 2.
    Returns (the objects made or changed, info text) or (None, why)."""
    (n1, i1, Q1), (n2, i2, Q2) = picks
    o1, o2 = bpy.data.objects.get(n1), bpy.data.objects.get(n2)
    if o1 is None or o2 is None or o1.type != 'CURVE' or o2.type != 'CURVE':
        return None, "pick two curves"
    _, w1, cl1 = _spline_world(o1, i1)
    same = o1 == o2 and i1 == i2
    _, w2, cl2 = (None, w1, cl1) if same else _spline_world(o2, i2)
    plan = fillet_plan(w1, w2, Q1, Q2, float(radius), same=same, closed=cl1 if same else False)
    if plan is None:
        return None, "no fillet of that radius fits there (parallel curves, or it runs off a curve's end)"
    for o in {o1, o2}:
        if trim or join:
            curvetrim._drop_history(_Reporter(report), o)
    if plan["same"]:
        curvetrim.put_back(o1, i1, [_to_local(plan["joined"], o1)])
        return [o1], "corner rounded"
    if join:
        curvetrim.put_back(o1, i1, [_to_local(plan["joined"], o1)])
        cu2 = o2.data
        if len(cu2.splines) == 1:
            bpy.data.objects.remove(o2)
        else:
            curvetrim.put_back(o2, i2, [])
        return [o1], "joined into one curve"
    made = []
    if trim:
        if not cl1:
            curvetrim.put_back(o1, i1, [_to_local(plan["keep1"], o1)])
        if not cl2:
            curvetrim.put_back(o2, i2, [_to_local(plan["keep2"], o2)])
    from .primitives import _add_spline
    cu = bpy.data.curves.new("Fillet", 'CURVE')
    cu.dimensions = '3D'
    arc_obj = bpy.data.objects.new("Fillet", cu)
    (context.collection or context.scene.collection).objects.link(arc_obj)
    _add_spline(cu, curvejoin.spans_to_spec(plan["arc"], simplify=False))
    made.append(arc_obj)
    return made, "trimmed" if trim else "curves kept whole"


class _Reporter:
    def __init__(self, fn):
        self.fn = fn

    def report(self, kind, msg):
        if self.fn is not None:
            self.fn(kind, msg)


class NPROTO_OT_fillet_curves(_ShapeTool, bpy.types.Operator):
    """Round the corner between two curves: click the first, then the second
    near the corner (the parts you click stay), then the radius (mouse, or type
    it). The arc is exact and tangent to both; the curves are trimmed to it (T)
    and can be joined into one curve (J). Two segments of one polyline get
    their corner rounded in place"""
    bl_idname = "nproto.fillet_curves"
    bl_label = "Fillet"
    bl_options = {'REGISTER', 'UNDO'}
    MODES = [("RADIUS", "Radius", "An arc of a radius, tangent to both curves")]
    TITLE = "Fillet"

    mode: bpy.props.EnumProperty(name="Mode", items=MODES, default="RADIUS")
    radius: FloatProperty(name="Radius", default=0.25, min=1e-6, unit='LENGTH')
    join: BoolProperty(name="Join", default=False, description="The trimmed curves and the arc become one curve")
    trim: BoolProperty(name="Trim", default=True, description="Trim both curves back to the arc's tangent points")
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.prop(self, "radius")
        self.layout.prop(self, "trim")
        self.layout.prop(self, "join")

    # -- picking curves -----------------------------------------------------------
    def _reset(self):
        _ShapeTool._reset(self)
        self._picks = []

    def _needs_line(self):
        return len(getattr(self, "_picks", [])) < 2

    def typing_ready(self):
        return len(getattr(self, "_picks", [])) == 2

    def _fields(self):
        return ["R"] if len(getattr(self, "_picks", [])) == 2 else []

    def _line_under(self, context, event):
        """(point on it, None, polyline, (object name, spline index)) of the curve under the mouse."""
        view, mouse = self._view(context), self._mouse(context, event)
        for k, _ in self._snaps.near_lines(view, mouse, SNAP_PX, limit=4):
            W, src = self._snaps.line_info(k)
            if src is None or not str(src[1]).startswith("S"):
                continue
            o = bpy.data.objects.get(src[0])
            if o is None or o.type != 'CURVE':
                continue
            P, _, _ = self._snaps.slide(view, mouse, k)
            return (np.asarray(P, float), None, W, (src[0], int(str(src[1])[1:])))
        return None

    def _step(self):
        return ["click the first curve, near the corner", "click the second curve, near the corner",
                "the radius: move, or type it"][min(len(self._picks), 2)]

    def _extra_key(self, context, event):
        if event.value == 'PRESS' and event.type in {'J', 'T'} and not self._buf:
            if event.type == 'J':
                self.join = not self.join
            else:
                self.trim = not self.trim
            return True
        return False

    def _extra_rows(self):
        return [("Result", [hints.toggle("T", "trim", self.trim), hints.toggle("J", "join", self.join)])]

    def _click(self, context, event):
        if len(self._picks) < 2:
            ln = self._line_under(context, event)
            if ln is None:
                self.report({'WARNING'}, "Fillet: click a curve")
                return None
            self._picks.append((ln[3][0], ln[3][1], ln[0], ln[2]))
            if self._plane is None:
                self.fix_plane(context, ln[0])
            self._reset_typing()
            return None
        return self._finish(context, self._shape(context, event, final=True))

    # -- the fillet under way ---------------------------------------------------------
    def _plan(self, context, event, r=None):
        (n1, i1, Q1, _), (n2, i2, Q2, _) = self._picks
        cache = getattr(self, "_spans_cache", None)
        key = (n1, i1, n2, i2)
        if cache is None or cache[0] != key:
            o1, o2 = bpy.data.objects.get(n1), bpy.data.objects.get(n2)
            _, w1, cl1 = _spline_world(o1, i1)
            same = o1 == o2 and i1 == i2
            w2 = w1 if same else _spline_world(o2, i2)[1]
            self._spans_cache = cache = (key, w1, w2, same, cl1)
        _, w1, w2, same, cl1 = cache
        typed = self._typed(context)
        if r is None:
            r = typed.get("R")
        if r is None:                                    # from the mouse, along the corner's bisector
            P1, D1 = span_eval(w1, nearest_g(w1, Q1))
            P2, D2 = span_eval(w2, nearest_g(w2, Q2))
            nn = np.cross(_unit(D1), _unit(D2))
            if float(np.linalg.norm(nn)) < 1e-9:
                return None, None
            g0 = _tangent_lines_guess(P1, _unit(D1), P2, _unit(D2), _unit(nn), 1.0)
            if g0 is None:
                return None, None
            M = self._mouse_on(context, event, g0["X"], nn)
            if M is None:
                return None, None
            r = radius_from(g0, M)
        return fillet_plan(w1, w2, Q1, Q2, float(r), same=same, closed=cl1 if same else False), float(r)

    def _shape(self, context, event, final=False):
        if len(self._picks) < 2:
            return None
        plan, r = self._plan(context, event)
        if plan is None:
            return None
        lines = [polyline(plan["joined"] if (plan["same"] or self.join) else plan["arc"])]
        if not plan["same"] and not self.join:
            lines += [polyline(plan["keep1"] if self.trim else self._spans_cache[1]),
                      polyline(plan["keep2"] if self.trim else self._spans_cache[2])]
        return {"kind": "fillet", "poly": lines[0], "extra": lines[1:], "radius": r, "vals": {"R": r}, "plan": plan}

    def _preview(self, context, event):
        _ShapeTool._preview(self, context, event)
        ov = draw.sketch_overlay
        sh = self._shape(context, event) if len(self._picks) == 2 else None
        if sh is not None:
            ov["curves"] = [[Vector(p) for p in pl] for pl in [sh["poly"]] + sh["extra"] if len(pl) > 1]
        for _, _, Q, W in getattr(self, "_picks", []):
            ov["refs"].append(([Vector(p) for p in W], COL_REMOVED, 1.5))
        ov["cvs"] = [Vector(Q) for _, _, Q, _ in getattr(self, "_picks", [])]
        context.area.tag_redraw()

    def _finish(self, context, sh):
        if sh is None:
            self.report({'WARNING'}, "Fillet: no arc of that radius fits there")
            return None
        self.radius = sh["radius"]
        self.data = json.dumps({"picks": [[n, i, list(map(float, Q))] for n, i, Q, _ in self._picks]})
        self.sketch_end(context)
        return self.execute(context)

    def execute(self, context):
        try:
            d = json.loads(self.data or "{}")
        except ValueError:
            d = {}
        if not d.get("picks"):
            self.report({'WARNING'}, "Fillet is interactive: start it from the panel or menu")
            return {'CANCELLED'}
        picks = [(n, int(i), np.array(Q, float)) for n, i, Q in d["picks"]]
        made, info = apply_fillet(context, picks, self.radius, self.join, self.trim, self.report)
        if made is None:
            self.report({'WARNING'}, f"Fillet: {info}")
            return {'CANCELLED'}
        _select_new(context, made)
        self.report({'INFO'}, f"Fillet, radius {self.radius:.4g}: {info}")
        return {'FINISHED'}


classes = (NPROTO_OT_fillet_curves,)
