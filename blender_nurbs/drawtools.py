"""Sketch tools: CV Curve and Line (interactive, click in the 3D view).

CV Curve
  CV mode            -- every click places a CV.
  Through Points     -- the curve passes through every click.
  Either way the result is one clean uniform NURBS spline (Endpoint on, no
  Bezier knot stacks: curvature-continuous, any CV can be moved without
  making a corner). Degree 1-5; C closes the curve (periodic).
Line
  Two clicks per line; each line is its own 2-CV curve (or, with Polyline,
  connected lines become one degree-1 curve).

Points land on the drawing plane: the grid plane in an axis-aligned
orthographic view (Top / Front / Side), else the view plane, through the 3D
cursor (or through the first snapped point). Clicks snap (see snapping.py) to
the ends, middles, CVs and any point of visible curves, to the vertices,
middles and any point of surface edges, to face centres, to the CVs of
surfaces whose CV net is shown, and to the points drawn so far. Ctrl snaps to
the grid, Shift to 15 degree angles from the previous point.
Locks: H / V horizontal / vertical in the drawing plane; X / Y / Z along that
world axis from the previous point (from the 3D cursor for the first);
Shift+X / Y / Z in the plane of the other two axes. A lock, Ctrl or Shift
leaves only the point snaps on, not the edges.
CV Curve start / end conditions (S / E keys, or the F9 panel): an end snapped
onto a curve or a surface edge is G0, G1 or G2 to it, solved exactly like
Align Curve (align.py): only the end CVs move in CV mode, Through Points adds
CVs so the curve still passes through every click. With G1 / G2 and
Construction History on, the curve gets an Align Curve history step at once,
so its ends stay locked to their targets when either one changes.
In curve Edit Mode the new splines go into the object being edited.
"""

import json
import math

import numpy as np

from . import curvejoin, curvetools, geom, snapping

SNAP_PX = snapping.SNAP_PX
ANGLE_STEP = math.radians(15.0)


# --------------------------------------------------------------------------
# Curve construction (pure numpy)
# --------------------------------------------------------------------------

def _chord(S, closed):
    P = np.vstack([S, S[:1]]) if closed else S
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    if d.sum() <= 0:
        return None
    t = np.concatenate([[0.0], np.cumsum(d)]) / d.sum()
    return t[:-1] if closed else t


def interpolate(S, deg, closed=False):
    """CVs of the uniform (clamped, or periodic when closed) B-spline of degree
    deg through the points S (as many CVs as points). Chord-length parameters,
    uniform ones if that system is badly conditioned."""
    S = np.asarray(S, float)
    n = len(S)
    p = max(1, min(deg, n - 1))
    if p == 1 or n <= 2:
        return S.copy(), p
    B = curvetools.Basis(n, p, closed)
    best = None
    for kind in ("chord", "uniform"):
        t = _chord(S, closed) if kind == "chord" else (
            np.arange(n) / n if closed else np.linspace(0.0, 1.0, n))
        if t is None:
            continue
        A = B.matrix(t)
        c = np.linalg.cond(A)
        if best is None or c < best[0]:
            best = (c, A)
        if c < 1e8:
            break
    X = np.linalg.solve(best[1], S)
    return X, p


def apply_tangents(X, start_out=None, end_out=None):
    """CV mode: put the 2nd (and last-but-one) CV on the tangent line of the
    curve end it was snapped to (G1 to that curve)."""
    X = np.array(X, float)
    n = len(X)
    for which, out in ((0, start_out), (1, end_out)):
        if out is None or n < 2:
            continue
        o = np.asarray(out, float)
        o = o / np.linalg.norm(o)
        if which == 0:
            P, k = X[0], 1
        else:
            P, k = X[-1], n - 2
        if which == 1 and start_out is not None and n - 2 <= 1:
            break          # too few CVs to follow both tangents: the start wins
        s = float((X[k] - P) @ o)
        L = float(np.linalg.norm(X[k] - P))
        X[k] = P + max(s, 0.25 * L, 1e-6) * o
    return X


def build_spec(clicks, mode, deg, closed, start_out=None, end_out=None):
    """World-space spline spec from the clicked points."""
    from .primitives import spec
    S = np.asarray(clicks, float)
    if closed and len(S) > 2 and np.linalg.norm(S[0] - S[-1]) < 1e-9:
        S = S[:-1]
    n = len(S)
    if n < 2:
        raise ValueError("place at least 2 points")
    closed = closed and n >= 3
    if mode == "THROUGH":
        X, p = interpolate(S, deg, closed)
    else:
        X = S if closed else apply_tangents(S, start_out, end_out)
        p = max(1, min(deg, n - 1))
    if closed:
        return spec([tuple(x) for x in X], order=p + 1, cyclic=True, endpoint=False)
    return spec([tuple(x) for x in X], order=p + 1)


def through_ends(S, deg, ends):
    """CVs and degree of a uniform spline through the points S whose ends are
    aligned (ends as align.align_open). Each G1 / G2 end adds one / two CVs:
    the end CVs come from align_start applied to the plain interpolation's end
    derivatives, the others are solved so the curve still passes through every
    point. Returns (X, p, specs)."""
    from . import align
    S = np.asarray(S, float)
    n = len(S)
    specs = {e: dict(sp_) for e, (sp_, _) in ends.items()}
    ks = align.CONT[specs["0"]["cont"]] if "0" in specs else 0
    ke = align.CONT[specs["1"]["cont"]] if "1" in specs else 0
    X0, p0 = interpolate(S, deg)
    if ks == 0 and ke == 0:
        X, p, specs = align.align_open(X0, p0, ends)        # G0: the ends sit on their targets
        return X, p, specs
    N = n + ks + ke
    p = min(max(deg, align.degree_for([s_["cont"] for s_ in specs.values()])), N - 1)
    U0 = geom.clamped_uniform_knots(n, p0)
    U = geom.clamped_uniform_knots(N, p)
    Q = np.zeros((N, 3))
    I = np.eye(4)
    for e, k in (("0", ks), ("1", ke)):
        A, Ua = (X0, U0) if e == "0" else align._reverse(X0, U0)
        d1, d2, _, _ = align._derivs(A, Ua, p0) if p0 >= 1 else (A[1] - A[0], np.zeros(3), 1, 1)
        seg, Ub = (Q, U) if e == "0" else align._reverse(Q, U)
        seg = align._set_start(seg, Ub, p, min(k, 2) if p >= 2 else min(k, 1), A[0], d1, d2)
        if e in ends:
            sp_ = specs[e]
            P, D, DD, surf = ends[e][1]
            if sp_.get("snap") == "NEAREST" and sp_.get("mode", "EDGE") == "EDGE" and D @ d1 < 0:
                D = -D
                sp_["flip"] = not sp_.get("flip", False)
            seg = align.align_start(seg, Ub, p, sp_, (P, D, DD, surf), I)
        Q = seg if e == "0" else align._reverse(seg, Ub)[0]
    fixed = list(range(ks + 1)) + list(range(N - 1 - ke, N))
    free = [i for i in range(N) if i not in fixed]
    if free:
        t = _chord(S, False)
        if t is None:
            t = np.linspace(0.0, 1.0, n)
        A = curvetools.Basis(N, p, False).matrix(t[1:-1])
        rhs = S[1:-1] - A[:, fixed] @ Q[fixed]
        Q[free] = np.linalg.lstsq(A[:, free], rhs, rcond=None)[0]
    return Q, p, specs


def build_spec_ends(clicks, mode, deg, ends):
    """World-space spline spec of an open sketch curve with end conditions
    (see through_ends / align.align_open), and the end specs used."""
    from . import align
    from .primitives import spec
    S = np.asarray(clicks, float)
    if len(S) < 2:
        raise ValueError("place at least 2 points")
    if mode == "THROUGH":
        X, p, specs = through_ends(S, deg, ends)
    else:
        X, p, specs = align.align_open(S, max(1, min(deg, len(S) - 1)), ends)
    return spec([tuple(x) for x in X], order=p + 1), specs


def spec_polyline(sp, per=24):
    deg, P, W, U = geom.blender_nurbs_to_clamped(sp["points"], sp["weights"], sp["order"], sp["cyclic"],
                                                 sp["endpoint"], sp["bezier"])
    from .curvetrim import sample_spans
    return sample_spans(curvejoin._clamped_to_spans(deg, P, W, U), per)[0]


def polar_point(prev, u, v, length, angle_deg):
    """The point `length` away from prev at angle_deg in the drawing plane
    (0 = along u, the horizontal; counter-clockwise towards v)."""
    a = math.radians(angle_deg)
    return np.asarray(prev, float) + length * (math.cos(a) * np.asarray(u, float) + math.sin(a) * np.asarray(v, float))


def plane_angle(d, u, v):
    return math.degrees(math.atan2(float(d @ v), float(d @ u)))


REF_UNIT = {"INCHES": "in", "FEET": "ft", "THOU": "thou", "MILES": "mi", "MILLIMETERS": "mm",
            "CENTIMETERS": "cm", "METERS": "m", "KILOMETERS": "km", "MICROMETERS": "um"}


def parse_length(context, text):
    """Typed length in Blender units: unit letters are honoured (2ft 6in, 10mm),
    a plain number is in the scene's display length unit."""
    us = context.scene.unit_settings
    if us.system == 'NONE':
        return float(text)
    ref = REF_UNIT.get(us.length_unit, "in" if us.system == 'IMPERIAL' else "m")
    meters = bpy.utils.units.to_value(us.system, 'LENGTH', text, str_ref_unit=ref)
    return meters / (us.scale_length or 1.0)


def chain_lines(lines, tol=1e-9):
    """Connected lines -> polylines (lists of points), in drawing order."""
    out = []
    for a, b in lines:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if out and np.linalg.norm(out[-1][-1] - a) <= tol:
            out[-1].append(b)
        else:
            out.append([a, b])
    return out


AXES = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}
AXIS_COL = {"X": (1.0, 0.2, 0.32, 1.0), "Y": (0.55, 0.86, 0.0, 1.0), "Z": (0.16, 0.56, 1.0, 1.0)}
COL_TANGENT = (1.0, 0.85, 0.2, 1.0)
COL_REF_FACE = (0.95, 0.72, 0.5)


def ref_label(kind, cand, n, i, locked=False):
    """Snap label, naming the reference only when that tells something: a face
    ('Edge · Face'), or a choice ('End · Edge 2/3 (T)'); ' · locked (L)' while
    locked onto an edge."""
    from . import snapping
    txt = snapping.LABELS.get(kind, kind)
    if cand is not None and (cand["src"][3] is not None or n > 1):
        txt += " · Face" if cand["src"][3] is not None else (" · Curve" if cand["src"][1].startswith("S")
                                                             else " · Edge")
    if n > 1:
        txt += f" {i % n + 1}/{n} (T)"
    if locked:
        txt += " · locked (L)"
    return txt


def ray_line_point(o, d, base, axis):
    """Point of the line base + s * axis nearest to the ray o + t * d. A line seen
    end-on gives base."""
    o, d, base, axis = (np.asarray(x, float) for x in (o, d, base, axis))
    w = base - o
    a, b, c = float(axis @ axis), float(axis @ d), float(d @ d)
    den = a * c - b * b
    if den <= 1e-9 * a * c:
        return base.copy()
    s = (b * float(d @ w) - c * float(axis @ w)) / den
    return base + s * axis


def ray_plane_point(o, d, base, normal):
    """Ray o + t * d through the plane (base, normal); None when the plane is
    seen edge-on."""
    o, d, base, normal = (np.asarray(x, float) for x in (o, d, base, normal))
    den = float(d @ normal)
    if abs(den) < 1e-3 * float(np.linalg.norm(d)) * float(np.linalg.norm(normal)):
        return None
    return o + d * float((base - o) @ normal) / den


def plane_axes(vec):
    """Two unit directions spanning the plane normal to vec: the world axes for
    X / Y / Z, else a stable pair."""
    vec = np.asarray(vec, float) / (np.linalg.norm(vec) or 1.0)
    for k, e in AXES.items():
        if abs(abs(float(vec @ e)) - 1.0) < 1e-12:
            return [AXES[j] for j in "XYZ" if j != k]
    a = AXES["Z"] if abs(vec[2]) < 0.9 else AXES["X"]
    e1 = np.cross(a, vec)
    e1 /= np.linalg.norm(e1)
    return [e1, np.cross(vec, e1)]


def axis_lock_point(lock, base, ray, snapped=None, plane_point=None, step=None, vec=None):
    """The point for an axis lock ('X' / 'Y' / 'Z': along that world axis from
    base; '-X' / '-Y' / '-Z': in the plane through base of the other two;
    'T' / '-T': along / normal to `vec`, the tangent of an edge or the normal of
    a face). ray = (origin, direction) under the mouse. A snapped point is
    projected onto the line / plane, so the result lines up with it.
    plane_point (the drawing plane under the mouse) stands in when the lock
    plane is seen edge-on. step: grid increment for the offsets from base (Ctrl)."""
    vec = AXES[lock[-1]] if vec is None else np.asarray(vec, float) / (np.linalg.norm(vec) or 1.0)
    base = np.asarray(base, float)
    if lock[0] != "-":
        P = np.asarray(snapped, float) if snapped is not None else ray_line_point(ray[0], ray[1], base, vec)
        r = float((P - base) @ vec)
        if step:
            r = round(r / step) * step
        return base + r * vec
    if snapped is not None:
        P = np.asarray(snapped, float)
    else:
        P = ray_plane_point(ray[0], ray[1], base, vec)
        if P is None:
            if plane_point is None:
                return None
            P = np.asarray(plane_point, float)
    off = P - base
    off -= float(off @ vec) * vec
    if step:
        for e in plane_axes(vec):
            c = float(off @ e)
            off += (round(c / step) * step - c) * e
    return base + off


def plane_rect(A, B, u, v, from_center=False, square=False, size=(None, None)):
    """(centre, size X, size Y, signs) of a rectangle on the drawing plane (u, v)
    from the first click A to B; A is the centre with from_center. Typed sizes
    (full sizes) override the mouse, which then only gives the side. square:
    both sizes the larger one."""
    A, B, u, v = (np.asarray(x, float) for x in (A, B, u, v))
    dx, dy = float((B - A) @ u), float((B - A) @ v)
    k = 2.0 if from_center else 1.0
    X = float(size[0]) if size[0] is not None else k * abs(dx)
    Y = float(size[1]) if size[1] is not None else k * abs(dy)
    if square and (size[0] is None or size[1] is None):
        X = Y = float(size[0] if size[0] is not None else size[1] if size[1] is not None else max(X, Y))
    sx, sy = (1.0 if dx >= 0 else -1.0), (1.0 if dy >= 0 else -1.0)
    C = A if from_center else A + 0.5 * sx * X * u + 0.5 * sy * Y * v
    return C, X, Y, (sx, sy)


def plane_matrix(C, u, v):
    """Object matrix of a plane drawn on (u, v) about C: local X = u, Y = v,
    Z = u x v (its normal faces the view it was drawn in)."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) - (np.asarray(v, float) @ u) * u
    v /= np.linalg.norm(v)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = u, v, np.cross(u, v), np.asarray(C, float)
    return M


def next_start(lines, P, chain, tol=1e-9):
    """Where the next line starts after one ended at P: at P when the lines
    chain (Polyline on, or a typed line), so each click adds a segment --
    unless that click closed the chain on its first point; else nowhere."""
    if not chain:
        return None
    chains = chain_lines(lines, tol)
    if chains and len(chains[-1]) > 3 and np.linalg.norm(chains[-1][0] - chains[-1][-1]) <= tol:
        return None
    return np.asarray(P, float)


# --------------------------------------------------------------------------
# Blender side
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty  # noqa: E402
from bpy_extras import view3d_utils  # noqa: E402
from mathutils import Vector  # noqa: E402

from . import draw  # noqa: E402
from .primitives import PROFILE_SHAPES  # noqa: E402

DEGREE_ITEMS = [(str(d), str(d), "") for d in range(1, curvejoin.MAX_DEGREE + 1)]
MODE_ITEMS = [("CV", "CVs", "Every click places a control point"),
              ("THROUGH", "Through Points", "The curve passes through every click")]
PASS_KEYS = {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE',
             'NDOF_MOTION', 'NUMPAD_0', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_5', 'NUMPAD_6',
             'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_PERIOD', 'NUMPAD_PLUS', 'NUMPAD_MINUS', 'ACCENT_GRAVE'}


def add_curve(context, name, specs_world, edit_obj=None):
    """Write world-space specs as a new curve object (origin at the first
    point), or as new splines of the curve being edited. Returns the object."""
    from .primitives import _add_spline
    if edit_obj is not None:
        Minv = np.linalg.inv(np.array(geom.world_matrix(edit_obj), float))
        cu = edit_obj.data
        for sp in specs_world:
            P = np.array(sp["points"], float) @ Minv[:3, :3].T + Minv[:3, 3]
            _add_spline(cu, dict(sp, points=[tuple(p) for p in P]))
            from .curvetrim import select_spline
            select_spline(cu.splines[-1])
        edit_obj.data.update_tag()
        return edit_obj
    origin = np.array(specs_world[0]["points"][0], float)
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    obj = bpy.data.objects.new(name, cu)
    (context.collection or context.scene.collection).objects.link(obj)
    obj.location = tuple(origin)
    for sp in specs_world:
        P = np.array(sp["points"], float) - origin
        _add_spline(cu, dict(sp, points=[tuple(p) for p in P]))
    return obj


def _edit_target(context):
    if context.mode == 'EDIT_CURVE' and context.active_object is not None and context.active_object.type == 'CURVE':
        return context.active_object
    return None


def _select_new(context, objs):
    for o in context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    if objs:
        context.view_layer.objects.active = objs[-1]


class SketchMixin:
    """Point picking on a drawing plane with snapping."""

    def sketch_start(self, context):
        self._plane = None          # (origin, normal, u, v), fixed by the first point
        self._snaps = self._snap_targets(context)
        self._hover = None          # (point, snap info)
        self._refs = None           # references at the hovered snap: {"keys", "cands", "i"}
        self._hover_cand = None
        self._edge_lock = None      # (object, ref): L locked the point onto this edge / curve
        self._last_edge = None      # the edge last slid along: it stays the reference at its ends
        draw.sketch_overlay.update(curves=[], hull=[], cvs=[], snap=None, snap_label=None, guide=[], axes=[],
                                   refs=[], cursor=None)

    def sketch_end(self, context):
        draw.sketch_overlay.update(curves=[], hull=[], cvs=[], snap=None, snap_label=None, guide=[], axes=[],
                                   refs=[], cursor=None)
        context.area.header_text_set(None)
        context.area.tag_redraw()

    # -- the 3D view's drawing region (the tool may be started from the sidebar,
    #    whose region is not the one we draw in) --------------------------
    def _win(self, context):
        area = context.area
        region = next((r for r in area.regions if r.type == 'WINDOW'), context.region) if area else context.region
        rv3d = getattr(area.spaces.active, "region_3d", None) if area else None
        return region, rv3d or context.region_data

    def _mouse(self, context, event):
        region, _ = self._win(context)
        return np.array([event.mouse_x - region.x, event.mouse_y - region.y], float)

    # -- drawing plane ----------------------------------------------------
    def _view_plane(self, context, through):
        rv3d = self._win(context)[1]
        vdir = np.array(rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)), float)
        axis = int(np.argmax(np.abs(vdir)))
        if not rv3d.is_perspective and abs(abs(vdir[axis]) - 1.0) < 1e-4:
            n = np.zeros(3)
            n[axis] = 1.0
            u, v = {2: ((1, 0, 0), (0, 1, 0)), 1: ((1, 0, 0), (0, 0, 1)), 0: ((0, 1, 0), (0, 0, 1))}[axis]
            u, v = np.array(u, float), np.array(v, float)
        else:
            n = -vdir
            u = np.array(rv3d.view_rotation @ Vector((1.0, 0.0, 0.0)), float)
            v = np.cross(n, u)
        return np.asarray(through, float), n, u, v

    def _ray_point(self, context, mouse, plane):
        region, rv3d = self._win(context)
        o = np.array(view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse), float)
        d = np.array(view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse), float)
        P0, n = plane[0], plane[1]
        den = float(d @ n)
        if abs(den) < 1e-12:
            return None
        return o + d * float((P0 - o) @ n) / den

    def _to_px(self, context, p):
        region, rv3d = self._win(context)
        q = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(p))
        return None if q is None else np.array(q, float)

    # -- snapping ---------------------------------------------------------
    def _snap_targets(self, context):
        """Snap targets of the visible curves and surfaces (snapping.Targets)."""
        return snapping.collect(context.visible_objects, lambda o: draw.cv_net_shown(context, o))

    def _view(self, context):
        region, rv3d = self._win(context)
        return snapping.View(rv3d.perspective_matrix, rv3d.view_matrix, region.width, region.height)

    def _grid_step(self, context, at):
        """A grid increment that is at least ~12 px on screen here."""
        p0 = self._to_px(context, at)
        u = self._plane[2] if self._plane else np.array([1.0, 0, 0])
        p1 = self._to_px(context, at + u)
        if p0 is None or p1 is None:
            return 1.0
        ppu = max(np.linalg.norm(p1 - p0), 1e-9)
        us = context.scene.unit_settings
        scale = us.scale_length or 1.0
        if us.system == 'IMPERIAL':
            inch = 0.0254 / scale
            steps = [inch * f for f in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 3, 6, 12, 36, 120, 1200)]
        else:
            steps = [10.0 ** k * f / scale for k in range(-4, 5) for f in (1, 2, 5)]
        for st in steps:
            if st * ppu >= SNAP_PX:
                return st
        return steps[-1]

    def pick_point(self, context, event, extra=(), prev=None):
        """(world point, snap) under the mouse. snap = (kind, outward tangent,
        reference, label): the reference (object, ref, END / NEAREST, face) is
        the edge or face a G1 / G2 end at this point follows (T cycles it)."""
        self._hover_cand = None
        P, snap = self._pick(context, event, extra, prev)
        kind = snap[0] if snap is not None else None
        if kind in ("edge", "curve", "mid") and snap[2] is not None:
            self._last_edge = tuple(snap[2][:2])      # sliding along this edge
        elif kind != "end":
            self._last_edge = None
        if snap is not None and snap[2] is not None:
            src, cand, n = self._choose_ref(P, snap[2], kind)
            self._hover_cand = cand
            snap = (snap[0], snap[1], src, ref_label(snap[0], cand, n, self._refs["i"] if n else 0,
                                                     getattr(self, "_edge_lock", None) is not None))
        elif snap is not None:
            snap = (snap[0], snap[1], None, snapping.LABELS.get(snap[0]))
        return P, snap

    def _pick(self, context, event, extra=(), prev=None):
        mouse = self._mouse(context, event)
        plane = self._plane or self._view_plane(context, np.array(context.scene.cursor.location, float))
        O, n, u, v = plane
        el = getattr(self, "_edge_lock", None)
        k = self._snaps.line_index(*el) if el else None
        if k is not None:
            # locked onto an edge (L): the point slides along it, wherever the mouse is
            P, kind, src = self._snaps.slide(self._view(context), mouse, k)
            return P, (kind, None, src)
        lock = getattr(self, "_lock", None)
        if lock and lock[-1] in "XYZT":
            # world axis lock (X / Y / Z) or plane lock (Shift+X / Y / Z), from
            # the previous point, or from the 3D cursor for the first one
            base = prev if prev is not None else np.array(context.scene.cursor.location, float)
            hit = self._snaps.pick(self._view(context), mouse, extra, edges=False)
            region, rv3d = self._win(context)
            ray = (np.array(view3d_utils.region_2d_to_origin_3d(region, rv3d, Vector(mouse)), float),
                   np.array(view3d_utils.region_2d_to_vector_3d(region, rv3d, Vector(mouse)), float))
            step = self._grid_step(context, base) if event.ctrl and hit is None else None
            P = axis_lock_point(lock, base, ray, None if hit is None else hit[0],
                                self._ray_point(context, Vector(mouse), plane), step,
                                getattr(self, "_tvec", None) if lock[-1] == "T" else None)
            return P, None
        axis = None
        if prev is not None:
            alock = getattr(self, "_angle_lock", None)
            if lock in ('H', 'V'):
                axis = u if lock == 'H' else v
            elif alock is not None:
                axis = math.cos(math.radians(alock)) * u + math.sin(math.radians(alock)) * v
        # points (ends, middles, centres, CVs) always snap; edges only when
        # nothing else is in charge (a lock, Ctrl grid, Shift angles)
        hit = self._snaps.pick(self._view(context), mouse, extra,
                               edges=axis is None and not event.ctrl and not event.shift)
        best = None if hit is None else (0.0, hit[0], hit[2], hit[1])
        if best is not None and axis is None:
            return best[1], (best[3], best[2], hit[3])
        if axis is not None:
            # locked direction (horizontal / vertical / typed angle): slide along
            # the line through the previous point; a snapped point is projected
            # onto it, so the line lines up with that point
            P = best[1] if best is not None else self._ray_point(context, Vector(mouse), plane)
            if P is None:
                return None, None
            r = float((P - prev) @ axis)
            if event.ctrl and best is None:
                st = self._grid_step(context, P)
                r = round(r / st) * st
            return prev + r * axis, None
        P = self._ray_point(context, Vector(mouse), plane)
        if P is None:
            return None, None
        if event.shift and prev is not None:
            d = P - prev
            a = math.atan2(float(d @ v), float(d @ u))
            a = round(a / ANGLE_STEP) * ANGLE_STEP
            r = float(np.linalg.norm([d @ u, d @ v]))
            if event.ctrl:
                st = self._grid_step(context, P)
                r = max(st, round(r / st) * st)
            P = prev + r * (math.cos(a) * u + math.sin(a) * v)
        elif event.ctrl:
            st = self._grid_step(context, P)
            x, y = float((P - O) @ u), float((P - O) @ v)
            ox, oy = float(O @ u), float(O @ v)       # snap to the world grid, not to the cursor
            x = round((x + ox) / st) * st - ox
            y = round((y + oy) / st) * st - oy
            P = O + x * u + y * v
        return P, None

    def toggle_lock(self, which):
        self._lock = None if getattr(self, "_lock", None) == which else which
        self._angle_lock = None
        self._edge_lock = None

    def edge_lock_key(self):
        """L: lock the point onto the highlighted edge / curve (the reference T
        picked), so it slides only along it; again: unlock."""
        if getattr(self, "_edge_lock", None):
            self._edge_lock = None
            return
        cand = getattr(self, "_hover_cand", None)
        if cand is not None:
            self._edge_lock = tuple(cand["src"][:2])
            self._lock = None
            self._angle_lock = None

    def lock_text(self):
        lk = getattr(self, "_lock", None)
        if lk == "T":
            return " · tangent to the edge"
        if lk == "-T":
            return " · in the face's tangent plane"
        if lk and lk[-1] in "XYZ":
            if lk[0] == "-":
                return " · plane " + "XYZ".replace(lk[-1], "") + f" (not {lk[-1]})"
            return f" · along {lk}"
        if lk:
            return " · " + ("horizontal" if lk == 'H' else "vertical")
        if getattr(self, "_angle_lock", None) is not None:
            return f" · angle {self._angle_lock:g}°"
        return ""

    def axis_key(self, event):
        """X / Y / Z (Shift: exclude that axis) pressed: toggle that lock."""
        self.toggle_lock(("-" if event.shift else "") + event.type)

    def axis_guides(self, context, prev):
        """Coloured guide lines of the active world axis lock through its base."""
        lk = getattr(self, "_lock", None)
        out = []
        if lk and lk[-1] in "XYZT":
            base = np.asarray(prev if prev is not None else context.scene.cursor.location, float)
            rv3d = self._win(context)[1]
            big = 50.0 * max(float(getattr(rv3d, "view_distance", 10.0)), 1e-3)
            if lk[-1] == "T":
                t = np.asarray(self._tvec, float)
                dirs = plane_axes(t) if lk[0] == "-" else [t / (np.linalg.norm(t) or 1.0)]
                cols = [COL_TANGENT] * len(dirs)
            else:
                ks = "XYZ".replace(lk[-1], "") if lk[0] == "-" else lk
                dirs, cols = [AXES[k] for k in ks], [AXIS_COL[k] for k in ks]
            for e, c in zip(dirs, cols):
                out.append(([Vector(base - big * e), Vector(base + big * e)], c))
        draw.sketch_overlay["axes"] = out

    # -- references: which edge / face a G1 / G2 end follows ----------------
    def ref_mode(self):
        """SURFACE: tangent to a face along the edge; EDGE: continue the edge."""
        return "EDGE"

    def ref_candidates(self, P, src):
        """Every reference at P: the faces along the edges through it (SURFACE),
        or the edges / curves themselves (EDGE). src's edge first."""
        mode = self.ref_mode()
        out, seen = [], set()
        for s3, meta, W in self._snaps.through(P, src):
            if mode == "SURFACE" and "edge" in meta:
                faces = snapping.edge_faces(meta["obj"], meta["edge"])
                for k, key, outline in faces:
                    if key not in seen:
                        seen.add(key)
                        out.append({"src": s3 + (k,), "edge": W, "face": outline})
                if faces:
                    continue
            key = s3[:2]
            if key not in seen:
                seen.add(key)
                out.append({"src": s3 + (None,), "edge": W, "face": None})
        return out

    def _choose_ref(self, P, src, kind=None):
        """(reference src, candidate, count) at a snapped point; the choice made
        with T sticks while the same references are under the mouse. Locked onto
        an edge (L): only that edge's. At an end where edges meet, the edge just
        slid along comes first."""
        el = getattr(self, "_edge_lock", None)
        last = getattr(self, "_last_edge", None)
        first = el or (last if kind == "end" and last else None) or src
        cands = self.ref_candidates(P, first)
        if el:
            cands = [c for c in cands if tuple(c["src"][:2]) == tuple(el)] or cands
        keys = [c["src"][:2] + c["src"][3:] for c in cands]
        st = getattr(self, "_refs", None)
        if st is None or st["keys"] != keys:
            st = self._refs = {"keys": keys, "cands": cands, "i": 0}
        else:
            st["cands"] = cands
        if not cands:
            return tuple(src) + (None,), None, 0
        c = cands[st["i"] % len(cands)]
        return c["src"], c, len(cands)

    def cycle_ref(self, which=None):
        """T: the next reference at the hovered snap (or in `which`, a stored list)."""
        st = which if which is not None else getattr(self, "_refs", None)
        if st and st["cands"]:
            st["i"] = (st["i"] + 1) % len(st["cands"])
            return True
        return False

    def show_refs(self, extra=()):
        """Highlight the hovered reference (edge white, its face outlined) and
        `extra` candidates (the start's), dimmer."""
        out = []
        for cand, alpha in [(self._hover_cand, 1.0)] + [(c, 0.45) for c in extra]:
            if not cand:
                continue
            for pl in cand.get("face") or []:
                out.append(([Vector(p) for p in pl], (COL_REF_FACE[0], COL_REF_FACE[1], COL_REF_FACE[2], alpha), 2.5))
            out.append(([Vector(p) for p in cand["edge"]], (1.0, 1.0, 1.0, alpha), 3.5))
        draw.sketch_overlay["refs"] = out

    def fix_plane(self, context, P):
        if self._plane is None:
            self._plane = self._view_plane(context, P)

    def passthrough(self, context, event):
        from .ops import PickMixin
        if event.type in PASS_KEYS:
            return True
        return event.type in {'LEFTMOUSE', 'RIGHTMOUSE'} and not PickMixin._in_view(context, event)


# --------------------------------------------------------------------------
# CV Curve
# --------------------------------------------------------------------------

CONT_ITEMS = [("G0", "G0", "Position: the end only sits on what it was snapped to"),
              ("G1", "G1", "Tangent: the end leaves tangent to the curve or surface edge it was snapped to"),
              ("G2", "G2", "Curvature: tangent, and the curvature matches as well")]
EDGE_MODES = [("SURFACE", "Tangent to Surface", "On a surface edge the curve stays tangent to the surface "
                                                "(G2: matches its normal curvature); curves are continued"),
              ("EDGE", "Continue Edge", "On a surface edge the curve continues the edge itself, like on a curve")]
LAST_INFO = {"text": ""}


class NPROTO_OT_draw_curve(SketchMixin, bpy.types.Operator):
    """Draw a NURBS curve by clicking: CVs, or points the curve passes through.
    The result is one clean uniform spline (no kinks when you move CVs later).
    An end snapped onto a curve or surface edge can be G0 / G1 / G2 to it"""
    bl_idname = "nproto.draw_curve"
    bl_label = "CV Curve"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Mode", items=MODE_ITEMS, default="CV")
    degree: EnumProperty(name="Degree", items=DEGREE_ITEMS, default="3")
    closed: BoolProperty(name="Closed", default=False, description="Close the curve (smooth, periodic)")
    start_cont: EnumProperty(name="Start", items=CONT_ITEMS, default="G1",
                             description="Continuity of the start with the curve or surface edge it was snapped "
                                         "to (G1 / G2 add an Align history step, so the end stays there)")
    end_cont: EnumProperty(name="End", items=CONT_ITEMS, default="G1",
                           description="Continuity of the end with the curve or surface edge it was snapped to "
                                       "(G1 / G2 add an Align history step, so the end stays there)")
    edge_mode: EnumProperty(name="On Surface Edges", items=EDGE_MODES, default="SURFACE")
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.row().prop(self, "mode", expand=True)
        lay.row().prop(self, "degree", expand=True)
        lay.prop(self, "closed")
        col = lay.column()
        col.active = not self.closed
        col.row().prop(self, "start_cont", expand=True)
        col.row().prop(self, "end_cont", expand=True)
        col.prop(self, "edge_mode")
        if LAST_INFO["text"]:
            for line in LAST_INFO["text"].split("\n"):
                lay.label(text=line)

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, f"{self.bl_label}: run it in a 3D View")
            return {'CANCELLED'}
        if context.mode not in ('OBJECT', 'EDIT_CURVE'):
            bpy.ops.object.mode_set(mode='OBJECT')
        self._edit = _edit_target(context)
        self._pts, self._snapinfo = [], []
        self._frames = {}
        self._start_refs = None     # the start's references (T cycles them once the start is placed)
        self.sketch_start(context)
        context.window_manager.modal_handler_add(self)
        self._status(context)
        self._preview(context, self.pick_point(context, event))     # the dot shows right away
        return {'RUNNING_MODAL'}

    def _status(self, context):
        n = len(self._pts)
        conds = "" if self.closed else f", start {self.start_cont} · end {self.end_cont}"
        context.area.header_text_set(
            f"CV Curve [{'CVs' if self.mode == 'CV' else 'Through Points'}, degree {self.degree}{conds}"
            f"{self.lock_text()}]: {n} point(s)   "
            "LMB place · Ctrl grid · Shift 15° · H/V horizontal/vertical · X/Y/Z axis · Shift+X/Y/Z plane · "
            "S/E start/end G0-G2 · T which edge / face · L lock onto the edge · F face or edge "
            f"[{'face' if self.edge_mode == 'SURFACE' else 'edge'}] · M mode · 1-5 degree · C close · "
            "Backspace undo · Enter/RMB done · Esc cancel")

    # -- end conditions ---------------------------------------------------
    def _cond(self, src, P, cont):
        """(align end spec, world frame) for an end snapped onto src = (object
        name, ref, END / NEAREST); None if it can't be evaluated."""
        from . import align, store
        name, ref, snap, face = (list(src) + [None])[:4]
        obj = bpy.data.objects.get(name)
        if obj is None:
            return None
        mode = self.edge_mode if store.is_nurbs(obj) else "EDGE"
        face = face if mode == "SURFACE" else None
        frames = getattr(self, "_frames", None)
        if frames is None:
            frames = self._frames = {}
        key = (name, ref, snap, mode, face, tuple(np.round(np.asarray(P, float), 12)))
        if key not in frames:
            try:
                frames[key] = align.end_frame(obj, ref, P, snap, mode, face=face or 0)
            except Exception as exc:
                print("CV Curve: no frame on", name, ref, exc)
                frames[key] = None
        fr = frames[key]
        if fr is None:
            return None
        spec = {"cont": cont, "weight": 1.0, "mode": mode, "snap": snap, "flip": False}
        if face is not None:
            spec["face"] = int(face)
        return spec, fr

    def ref_mode(self):
        return self.edge_mode

    def _ends(self, pts, srcs):
        """{"0" / "1": (spec, frame)} for the ends snapped onto curves / edges."""
        if self.closed or len(pts) < 2:
            return {}
        out = {}
        for e, src, P, cont in (("0", srcs[0], pts[0], self.start_cont), ("1", srcs[-1], pts[-1], self.end_cont)):
            if src:
                c = self._cond(tuple(src), P, cont)
                if c is not None:
                    out[e] = c
        return out

    @staticmethod
    def _src(snap):
        return snap[2] if snap and len(snap) > 2 else None

    def _spec(self, pts, srcs):
        ends = self._ends(pts, srcs)
        if ends:
            return build_spec_ends(pts, self.mode, int(self.degree), ends)
        return build_spec(pts, self.mode, int(self.degree), False), {}

    def _preview(self, context, hover=None):
        pts = list(self._pts) + ([hover[0]] if hover and hover[0] is not None else [])
        sn = list(self._snapinfo) + ([hover[1]] if hover and hover[0] is not None else [])
        ov = draw.sketch_overlay
        ov["snap"] = Vector(hover[0]) if hover and hover[1] is not None else None
        ov["snap_label"] = hover[1][3] if hover and hover[1] is not None else None
        ov["cursor"] = Vector(hover[0]) if hover and hover[0] is not None else None
        ov["curves"], ov["hull"], ov["cvs"] = [], [], [Vector(p) for p in self._pts]
        self.axis_guides(context, self._pts[-1] if self._pts else None)
        st = getattr(self, "_start_refs", None)
        self.show_refs([st["cands"][st["i"]]] if st and st["cands"] and not self.closed else [])
        if len(pts) >= 2:
            try:
                sp, _ = self._spec(pts, [self._src(x) for x in sn])
                ov["curves"] = [[Vector(p) for p in spec_polyline(sp, 16)]]
                ov["hull"] = [Vector(p) for p in sp["points"]]
            except Exception:
                pass
        context.area.tag_redraw()

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        prev = self._pts[-1] if self._pts else None
        first = [(self._pts[0], None, "first")] if len(self._pts) >= 3 else []
        hover = lambda: self.pick_point(context, event, first, prev)
        if event.type in {'MOUSEMOVE', 'LEFT_CTRL', 'RIGHT_CTRL', 'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            self._preview(context, hover())
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            P, snap = hover()
            if P is None:
                return {'RUNNING_MODAL'}
            if snap and snap[0] == "first":          # clicked the first point: close
                self.closed = True
                return self._finish(context)
            self.fix_plane(context, P)
            self._pts.append(P)
            self._snapinfo.append(snap)
            if len(self._pts) == 1:
                self._start_refs = dict(self._refs, cands=list(self._refs["cands"])) \
                    if snap and snap[2] is not None and self._refs else None
            if getattr(self, "_lock", None) in ("X", "Y", "Z", "-X", "-Y", "-Z"):
                self._lock = None            # an axis lock holds for one point, like G X in Blender
            self._edge_lock = None           # so does a lock onto an edge
            self._preview(context, (P, None))
            self._status(context)
        elif event.value == 'PRESS' and event.type == 'BACK_SPACE' and self._pts:
            self._pts.pop()
            self._snapinfo.pop()
            if not self._pts:
                self._plane = None
                self._start_refs = None
            self._preview(context)
            self._status(context)
        elif event.value == 'PRESS' and event.type in {'H', 'V'}:
            self.toggle_lock(event.type)
            self._preview(context, hover())
            self._status(context)
        elif event.value == 'PRESS' and event.type in {'X', 'Y', 'Z'}:
            self.axis_key(event)
            self._preview(context, hover())
            self._status(context)
        elif event.value == 'PRESS' and event.type == 'T':
            # the hovered snap's references, or else the start's
            P, snap = hover()
            if not (snap and snap[2] is not None and self.cycle_ref()):
                st = self._start_refs
                if st and self.cycle_ref(st):
                    c = st["cands"][st["i"]]
                    sn = self._snapinfo[0]
                    self._snapinfo[0] = (sn[0], sn[1], c["src"], ref_label(sn[0], c, len(st["cands"]), st["i"]))
            self._preview(context, hover())
            self._status(context)
        elif event.value == 'PRESS' and event.type == 'L':
            hover()
            self.edge_lock_key()
            self._preview(context, hover())
            self._status(context)
        elif event.value == 'PRESS' and event.type == 'F':
            self.edge_mode = "EDGE" if self.edge_mode == "SURFACE" else "SURFACE"
            if self._pts and self._snapinfo[0] and self._snapinfo[0][2] is not None:
                sn = self._snapinfo[0]
                self._refs = None
                src, cand, n = self._choose_ref(self._pts[0], sn[2])
                self._start_refs = dict(self._refs, cands=list(self._refs["cands"])) if cand else None
                self._snapinfo[0] = (sn[0], sn[1], src, ref_label(sn[0], cand, n, 0))
            self._refs = None
            self._preview(context, hover())
            self._status(context)
        elif event.value == 'PRESS' and event.type in {'S', 'E'}:
            prop = "start_cont" if event.type == 'S' else "end_cont"
            setattr(self, prop, {"G0": "G1", "G1": "G2", "G2": "G0"}[getattr(self, prop)])
            self._preview(context, hover())
            self._status(context)
        elif event.value == 'PRESS' and event.type == 'M':
            self.mode = "THROUGH" if self.mode == "CV" else "CV"
            self._preview(context, hover())
            self._status(context)
        elif event.value == 'PRESS' and event.type in {'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'}:
            self.degree = str(['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'].index(event.type) + 1)
            self._preview(context, hover())
            self._status(context)
        elif event.value == 'PRESS' and event.type == 'C':
            if len(self._pts) >= 3:
                self.closed = True
                return self._finish(context)
        elif event.value == 'PRESS' and event.type in {'RET', 'NUMPAD_ENTER', 'SPACE', 'RIGHTMOUSE'}:
            if len(self._pts) >= 2:
                return self._finish(context)
            self.sketch_end(context)
            return {'CANCELLED'}
        elif event.value == 'PRESS' and event.type == 'ESC':
            self.sketch_end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        srcs = [self._src(x) for x in self._snapinfo]
        self.data = json.dumps({"pts": [list(map(float, p)) for p in self._pts],
                                "start": list(srcs[0]) if srcs and srcs[0] else None,
                                "end": list(srcs[-1]) if len(srcs) > 1 and srcs[-1] else None,
                                "edit": self._edit.name if self._edit else ""})
        self.sketch_end(context)
        return self.execute(context)

    def execute(self, context):
        from . import align, features
        try:
            d = json.loads(self.data or "{}")
        except ValueError:
            d = {}
        if not d.get("pts"):
            self.report({'WARNING'}, "CV Curve is interactive: start it from the panel or menu")
            return {'CANCELLED'}
        pts = d["pts"]
        srcs = [d.get("start")] + [None] * max(len(pts) - 2, 0) + [d.get("end")]
        try:
            if self.closed:
                sp, specs = build_spec(pts, self.mode, int(self.degree), True), {}
            elif not d.get("start") and not d.get("end") and (d.get("start_out") or d.get("end_out")):
                # direction-only tangents (no target object): G1 / G2 put the 2nd CV on them
                so = d.get("start_out") if self.start_cont != "G0" else None
                eo = d.get("end_out") if self.end_cont != "G0" else None
                sp, specs = build_spec(pts, self.mode, int(self.degree), False, so, eo), {}
            else:
                sp, specs = self._spec(pts, srcs)
        except Exception as exc:
            self.report({'ERROR'}, f"CV Curve: {exc}")
            return {'CANCELLED'}
        edit = bpy.data.objects.get(d.get("edit", "")) if d.get("edit") else None
        if edit is not None and context.mode != 'EDIT_CURVE':
            edit = None
        obj = add_curve(context, "NURBS Curve", [sp], edit)
        if edit is None:
            _select_new(context, [obj])
        info = f"Curve: degree {sp['order'] - 1}, {len(sp['points'])} CVs" + (", closed" if sp["cyclic"] else "")
        lines = [info]
        for e in ("0", "1"):
            if e in specs:
                src = d["start"] if e == "0" else d["end"]
                lines.append(f"{'Start' if e == '0' else 'End'}: {specs[e]['cont']} to {src[0]} {src[1]}"
                             + ("" if specs[e]["mode"] == "EDGE" else
                                " (tangent to the surface" + (f", its face {specs[e]['face'] + 1})"
                                                              if "face" in specs[e] else ")")))
        if any(align.CONT[s_["cont"]] >= 1 for s_ in specs.values()):
            if edit is not None:
                lines.append("No history in Edit Mode: the ends are aligned once")
            elif not features.history_enabled(context):
                lines.append("Construction History is off: the ends are aligned once")
            else:
                inputs, ends_h = [(obj, "S0")], {}
                for e in ("0", "1"):
                    if e in specs:
                        src = d["start"] if e == "0" else d["end"]
                        inputs.append((bpy.data.objects[src[0]], src[1]))
                        ends_h[e] = {k: specs[e][k] for k in ("cont", "weight", "mode", "snap", "flip", "face")
                                     if k in specs[e]}
                        ends_h[e]["input"] = len(inputs) - 1
                try:
                    # once through Align's own solve, so geometry and history agree exactly
                    kind, deg_, Q, _ = align.solve(obj, ends_h, inputs)
                    align.write(obj, kind, deg_, Q)
                    features.record(obj, "align_curve", inputs, {"ends": ends_h})
                    lines.append("Align history added: the ends stay on their targets")
                except Exception as exc:
                    lines.append(f"Align history not added: {exc}")
        LAST_INFO["text"] = "\n".join(lines)
        self.report({'INFO'}, " · ".join(lines))
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Line
# --------------------------------------------------------------------------

class NPROTO_OT_draw_line(SketchMixin, bpy.types.Operator):
    """Draw straight lines: click the start, then click the end or type a
    length (Tab: angle) and press Enter. H / V lock horizontal / vertical,
    X / Y / Z a world axis (Shift+X / Y / Z: the plane of the other two),
    Shift snaps to 15 degrees, Ctrl to the grid. S: a start snapped onto an
    edge or curve leaves tangent to it (T picks which edge / face, F tangent to
    the face or the edge). Each line is its own 2-CV curve"""
    bl_idname = "nproto.draw_line"
    bl_label = "Line"
    bl_options = {'REGISTER', 'UNDO'}

    start_cont: EnumProperty(name="Start", items=CONT_ITEMS[:2], default="G0", options={'HIDDEN'},
                             description="G1: a line started on an edge or curve leaves tangent to it")
    line_mode: EnumProperty(name="On Surface Edges", items=EDGE_MODES, default="EDGE", options={'HIDDEN'})

    polyline: BoolProperty(name="Polyline", default=False,
                           description="Each click adds a segment to the line drawn so far (click its first point "
                                       "to close it, Backspace steps back); the lines become one degree-1 curve")
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        self.layout.prop(self, "polyline")

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, f"{self.bl_label}: run it in a 3D View")
            return {'CANCELLED'}
        if context.mode not in ('OBJECT', 'EDIT_CURVE'):
            bpy.ops.object.mode_set(mode='OBJECT')
        self._edit = _edit_target(context)
        self._lines, self._start, self._lock = [], None, None
        self._start_refs = None
        self._reset_typing()
        self.sketch_start(context)
        context.window_manager.modal_handler_add(self)
        self._preview(context, self._target(context, event))       # the dot shows right away
        return {'RUNNING_MODAL'}

    # -- typed input ------------------------------------------------------
    def _reset_typing(self):
        self._buf, self._field, self._len, self._ang = "", "LEN", None, None
        self._angle_lock = None

    def _live(self, context):
        """(length, angle) typed so far, the field being typed included."""
        L, A = self._len, self._ang
        if self._buf:
            try:
                if self._field == "LEN":
                    L = parse_length(context, self._buf)
                else:
                    A = float(eval(self._buf, {"__builtins__": {}}, {}))
            except Exception:
                pass
        return L, A

    def _target(self, context, event):
        """End point under the mouse with locks, snapping and typed values."""
        L, A = self._live(context)
        if self._start is not None:
            self._angle_lock = A
            if A is not None:
                self._lock = None
        P, snap = self.pick_point(context, event, self._extra(), self._start)
        if self._start is None or L is None:
            return P, snap
        O, n, u, v = self._plane
        if A is not None:
            d = math.cos(math.radians(A)) * u + math.sin(math.radians(A)) * v
        elif getattr(self, "_lock", None) in ('H', 'V', 'X', 'Y', 'Z', 'T'):
            d = {'H': u, 'V': v, 'T': np.asarray(getattr(self, "_tvec", u), float)}.get(self._lock, AXES.get(self._lock))
            d = d / (np.linalg.norm(d) or 1.0)
            if P is not None and float((P - self._start) @ d) < 0:
                d = -d
        elif P is not None and np.linalg.norm(P - self._start) > 1e-12:
            d = (P - self._start) / np.linalg.norm(P - self._start)
        else:
            d = u
        return self._start + L * d, None

    def _status(self, context, length=None, angle=None):
        us = context.scene.unit_settings
        fmt = lambda x: bpy.utils.units.to_string(us.system, 'LENGTH', x * (us.scale_length or 1.0), precision=4)
        txt = f"Line: {len(self._lines)} drawn · " + (
            ("click the next point (on the first one closes it)" if self.polyline and self._lines
             and self._start is not None and np.linalg.norm(self._start - self._lines[-1][1]) <= 1e-9
             else "place the end") if self._start is not None else "click the start")
        if self._start is not None:
            L, A = self._live(context)
            lt = (self._buf + "_" if self._field == "LEN" and self._buf else
                  (fmt(L) if L is not None else (fmt(length) if length is not None else "-")))
            at = (self._buf + "_" if self._field == "ANG" and self._buf else
                  (f"{A:g}°" if A is not None else (f"{angle:.1f}°" if angle is not None else "-")))
            mark = lambda f, t: f"[{t}]" if self._field == f else t
            txt += f"   Length {mark('LEN', lt)}  Angle {mark('ANG', at)}{self.lock_text()}"
        context.area.header_text_set(
            txt + "   type a length · Tab angle · Enter place · H/V horizontal/vertical · X/Y/Z axis · "
            "Shift+X/Y/Z plane · Shift 15° · Ctrl grid · "
            f"S tangent start [{self.start_cont}] · T which edge / face · L lock onto the edge · "
            f"F face or edge [{'face' if self.line_mode == 'SURFACE' else 'edge'}] · "
            f"P polyline [{'on' if self.polyline else 'off'}] · Backspace undo · RMB done · Esc cancel")

    # -- G1 start: tangent to the edge / curve it was snapped onto -----------
    def ref_mode(self):
        return self.line_mode

    def _apply_tangent(self):
        """Lock the line's direction to the start reference: along the edge's
        tangent (continue it), or in the face's tangent plane."""
        from . import align, store
        if getattr(self, "_lock", None) in ("T", "-T"):
            self._lock = None
        st = self._start_refs
        if self._start is None or self.start_cont != "G1" or not st or not st["cands"]:
            return
        name, ref, snap, face = st["cands"][st["i"]]["src"]
        obj = bpy.data.objects.get(name)
        if obj is None:
            return
        mode = self.line_mode if store.is_nurbs(obj) else "EDGE"
        try:
            P, D, DD, surf = align.end_frame(obj, ref, self._start, snap, mode, face=face or 0)
        except Exception as exc:
            print("Line: no tangent at", name, ref, exc)
            return
        if mode == "SURFACE" and surf is not None:
            self._lock, self._tvec = "-T", np.asarray(surf[0], float)
        elif np.linalg.norm(D) > 1e-12:
            self._lock, self._tvec = "T", np.asarray(D, float)

    def _new_start_refs(self, snap):
        self._start_refs = dict(self._refs, cands=list(self._refs["cands"])) \
            if snap and snap[2] is not None and getattr(self, "_refs", None) else None

    def _extra(self):
        out = []
        for a, b in self._lines:
            out += [(a, None, "end"), (b, None, "end")]
        return out

    def _preview(self, context, hover=None):
        ov = draw.sketch_overlay
        lines = [[Vector(a), Vector(b)] for a, b in self._lines]
        length = angle = None
        guide = []
        if self._start is not None and hover and hover[0] is not None:
            lines.append([Vector(self._start), Vector(hover[0])])
            d = hover[0] - self._start
            length = float(np.linalg.norm(d))
            if self._plane is not None:
                O, n, u, v = self._plane
                angle = plane_angle(d, u, v)
                big = 1e3 * max(length, 1e-3)
                guide = [Vector(self._start - big * u), Vector(self._start + big * u)]   # horizontal reference
        ov["curves"] = lines
        ov["guide"] = guide
        ov["cvs"] = [Vector(p) for ab in self._lines for p in ab] + ([Vector(self._start)] if self._start is not None else [])
        ov["snap"] = Vector(hover[0]) if hover and hover[1] is not None else None
        ov["snap_label"] = hover[1][3] if hover and hover[1] is not None else None
        ov["cursor"] = Vector(hover[0]) if hover and hover[0] is not None else None
        self.axis_guides(context, self._start)
        st = getattr(self, "_start_refs", None)
        self.show_refs([st["cands"][st["i"]]] if self._start is not None and st and st["cands"]
                       and self.start_cont == "G1" else [])
        self._status(context, length, angle)
        context.area.tag_redraw()

    def _commit(self, P, chain):
        if np.linalg.norm(P - self._start) <= 1e-9:
            return
        self._lines.append((self._start, P))
        # typed lines, and every line with Polyline on, chain on from their end
        self._start = next_start(self._lines, P, chain)
        self._reset_typing()
        self._lock = None
        self._edge_lock = None
        self._start_refs = None

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if not hasattr(self, "_buf"):
            self._reset_typing()
        typing = self._start is not None
        ch = event.unicode if event.value == 'PRESS' else ""
        if typing and ch and (ch in "0123456789.-+*/'\"" or (self._buf and (ch.isalpha() or ch == " "))) \
                and not (event.ctrl or event.alt or event.oskey):
            self._buf += ch
            self._preview(context, self._target(context, event))
            return {'RUNNING_MODAL'}
        if event.type in {'MOUSEMOVE', 'LEFT_CTRL', 'RIGHT_CTRL', 'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            self._preview(context, self._target(context, event))
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            P, snap = self._target(context, event)
            if P is None:
                return {'RUNNING_MODAL'}
            if self._start is None:
                self.fix_plane(context, P)
                self._start = P
                self._reset_typing()
                if getattr(self, "_lock", None) in ("X", "Y", "Z", "-X", "-Y", "-Z"):
                    self._lock = None            # an axis lock holds for one point
                self._edge_lock = None           # so does a lock onto an edge
                self._new_start_refs(snap)
                self._apply_tangent()
            else:
                self._commit(P, chain=self.polyline)     # Polyline: keep clicking to add segments
            self._preview(context, (P, None) if self._start is None else self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'TAB' and typing:
            L, A = self._live(context)
            self._len, self._ang = L, A
            self._buf = ""
            self._field = "ANG" if self._field == "LEN" else "LEN"
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'BACK_SPACE':
            if self._buf:
                self._buf = self._buf[:-1]
            elif self._start is not None and (self._len is not None or self._ang is not None):
                self._reset_typing()
            elif self._start is not None and self._lines and np.linalg.norm(self._start - self._lines[-1][1]) <= 1e-9:
                self._start = np.asarray(self._lines.pop()[0], float)    # chaining: step back one segment
                self._lock = None
            elif self._start is not None:
                self._start = None
                self._lock = None
                self._start_refs = None
            elif self._lines:
                self._lines.pop()
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type in {'H', 'V'} and not self._buf:
            self.toggle_lock(event.type)
            self._ang = None
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type in {'X', 'Y', 'Z'} and not self._buf:
            self.axis_key(event)
            self._ang = None
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'S' and not self._buf:
            self.start_cont = "G0" if self.start_cont == "G1" else "G1"
            self._apply_tangent()
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'T' and not self._buf:
            if self._start is not None and self._start_refs:
                self.cycle_ref(self._start_refs)
                self._apply_tangent()
            else:
                self._target(context, event)
                self.cycle_ref()
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'L' and not self._buf:
            self._target(context, event)
            self.edge_lock_key()
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'F' and not self._buf:
            self.line_mode = "EDGE" if self.line_mode == "SURFACE" else "SURFACE"
            self._refs = None
            if self._start is not None and self._start_refs:
                src = self._start_refs["cands"][self._start_refs["i"]]["src"]
                self._choose_ref(self._start, src)
                self._new_start_refs((None, None, src))
                self._refs = None
                self._apply_tangent()
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'P' and not self._buf:
            self.polyline = not self.polyline
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type in {'RET', 'NUMPAD_ENTER'} and typing and (
                self._buf or self._len is not None or self._ang is not None):
            P, _ = self._target(context, event)
            if P is not None:
                self._commit(P, chain=True)
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type in {'RET', 'NUMPAD_ENTER', 'SPACE', 'RIGHTMOUSE', 'ESC'}:
            self.sketch_end(context)
            if not self._lines or event.type == 'ESC':
                return {'CANCELLED'}
            self.data = json.dumps({"lines": [[list(map(float, a)), list(map(float, b))] for a, b in self._lines],
                                    "edit": self._edit.name if self._edit else ""})
            return self.execute(context)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from .primitives import spec
        try:
            d = json.loads(self.data or "{}")
        except ValueError:
            d = {}
        lines = d.get("lines") or []
        if not lines:
            self.report({'WARNING'}, "Line is interactive: start it from the panel or menu")
            return {'CANCELLED'}
        edit = bpy.data.objects.get(d.get("edit", "")) if d.get("edit") else None
        if edit is not None and context.mode != 'EDIT_CURVE':
            edit = None
        groups = chain_lines(lines) if self.polyline else [[np.array(a), np.array(b)] for a, b in lines]
        made = []
        for pl in groups:
            closed = bool(len(pl) > 3 and np.linalg.norm(pl[0] - pl[-1]) < 1e-9)
            pts = pl[:-1] if closed else pl
            sp = spec([tuple(p) for p in pts], order=2, cyclic=closed, endpoint=not closed)
            made.append(add_curve(context, "NURBS Line", [sp], edit))
        if edit is None:
            _select_new(context, made)
        self.report({'INFO'}, f"{len(lines)} line(s)" + (f" as {len(groups)} polyline(s)" if self.polyline else ""))
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Plane (a surface: click two corners, or type X / Y)
# --------------------------------------------------------------------------

class NPROTO_OT_draw_plane(SketchMixin, bpy.types.Operator):
    """Draw a NURBS plane: click one corner, then the opposite one, or type its
    X size (Tab: Y size) and press Enter. C draws from the centre, Shift makes
    it square, Ctrl snaps to the grid. Degree and spans per direction in the
    Adjust Last Operation panel, later in the Feature panel"""
    bl_idname = "nproto.draw_plane"
    bl_label = "Draw Plane"
    bl_options = {'REGISTER', 'UNDO'}

    size_x: FloatProperty(name="Size X", default=2.0, min=1e-6, unit='LENGTH')
    size_y: FloatProperty(name="Size Y", default=2.0, min=1e-6, unit='LENGTH')
    from_center: BoolProperty(name="From Centre", default=False,
                              description="The first click is the plane's centre, not a corner")
    degree_u: IntProperty(name="Degree U", default=3, min=1, max=7, description="Degree along X")
    degree_v: IntProperty(name="Degree V", default=3, min=1, max=7, description="Degree along Y")
    spans_u: IntProperty(name="Spans U", default=1, min=1, max=64, description="Spans along X")
    spans_v: IntProperty(name="Spans V", default=1, min=1, max=64, description="Spans along Y")
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        col = lay.column(align=True)
        col.prop(self, "size_x")
        col.prop(self, "size_y")
        lay.prop(self, "from_center")
        col = lay.column(align=True)
        col.prop(self, "degree_u")
        col.prop(self, "degree_v")
        col = lay.column(align=True)
        col.prop(self, "spans_u")
        col.prop(self, "spans_v")
        lay.label(text=f"{self.degree_u + self.spans_u} x {self.degree_v + self.spans_v} CVs")

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, f"{self.bl_label}: run it in a 3D View")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self._start, self._lock = None, None
        self._reset_typing()
        self.sketch_start(context)
        context.window_manager.modal_handler_add(self)
        self._preview(context, self.pick_point(context, event))
        return {'RUNNING_MODAL'}

    def _reset_typing(self):
        self._buf, self._field, self._sx, self._sy = "", "X", None, None

    def _live(self, context):
        X, Y = self._sx, self._sy
        if self._buf:
            try:
                val = parse_length(context, self._buf)
                X, Y = (val, Y) if self._field == "X" else (X, val)
            except Exception:
                pass
        return X, Y

    def _rect(self, context, event, P):
        """(centre, X, Y, signs) for the second corner P (None: nothing yet)."""
        if self._start is None or P is None:
            return None
        O, n, u, v = self._plane
        P = np.asarray(P, float)
        P = P - float((P - self._start) @ n) * n          # a snapped point off the plane: onto it
        return plane_rect(self._start, P, u, v, self.from_center, event.shift, self._live(context))

    def _status(self, context, rect=None):
        us = context.scene.unit_settings
        fmt = lambda x: bpy.utils.units.to_string(us.system, 'LENGTH', x * (us.scale_length or 1.0), precision=4)
        txt = "Plane: " + ("click the opposite corner" if self._start is not None else
                           "click the centre" if self.from_center else "click a corner")
        if self._start is not None:
            X, Y = self._live(context)
            xs = (self._buf + "_" if self._field == "X" and self._buf else
                  fmt(X) if X is not None else (fmt(rect[1]) if rect else "-"))
            ys = (self._buf + "_" if self._field == "Y" and self._buf else
                  fmt(Y) if Y is not None else (fmt(rect[2]) if rect else "-"))
            mark = lambda f, t: f"[{t}]" if self._field == f else t
            txt += f"   X {mark('X', xs)}  Y {mark('Y', ys)}"
        context.area.header_text_set(
            txt + f"   degree {self.degree_u} x {self.degree_v}, {self.spans_u} x {self.spans_v} span(s) · "
            "type X · Tab Y · Enter place · "
            f"C from centre [{'on' if self.from_center else 'off'}] · Shift square · Ctrl grid · "
            "X/Y/Z axis (first click) · Backspace undo · Esc cancel")

    def _preview(self, context, hover=None, event=None):
        ov = draw.sketch_overlay
        ov["snap"] = Vector(hover[0]) if hover and hover[1] is not None else None
        ov["snap_label"] = hover[1][3] if hover and hover[1] is not None else None
        ov["cursor"] = Vector(hover[0]) if hover and hover[0] is not None else None
        ov["curves"], ov["cvs"], ov["refs"] = [], [Vector(self._start)] if self._start is not None else [], []
        rect = None
        if self._start is not None and hover and hover[0] is not None and event is not None:
            rect = self._rect(context, event, hover[0])
        if rect is not None and rect[1] > 0 and rect[2] > 0:
            C, X, Y, _ = rect
            O, n, u, v = self._plane
            corner = lambda a, b: Vector(C + a * 0.5 * X * u + b * 0.5 * Y * v)
            ov["curves"] = [[corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1), corner(-1, -1)]]
            # the CV net it will get (uniform: the CVs of a flat plane are evenly spaced)
            nu, nv = self.degree_u + self.spans_u, self.degree_v + self.spans_v
            col = (0.55, 0.25, 0.85, 0.8)
            for i in range(nu):
                a = -1 + 2 * i / (nu - 1)
                ov["refs"].append(([corner(a, -1), corner(a, 1)], col, 1.0))
            for j in range(nv):
                b = -1 + 2 * j / (nv - 1)
                ov["refs"].append(([corner(-1, b), corner(1, b)], col, 1.0))
        self.axis_guides(context, None)
        self._status(context, rect)
        context.area.tag_redraw()

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        typing = self._start is not None
        ch = event.unicode if event.value == 'PRESS' else ""
        if typing and ch and (ch in "0123456789.-+*/'\"" or (self._buf and (ch.isalpha() or ch == " "))) \
                and not (event.ctrl or event.alt or event.oskey):
            self._buf += ch
            self._preview(context, self.pick_point(context, event), event)
            return {'RUNNING_MODAL'}
        hover = lambda: self.pick_point(context, event)
        if event.type in {'MOUSEMOVE', 'LEFT_CTRL', 'RIGHT_CTRL', 'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            self._preview(context, hover(), event)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            P, snap = hover()
            if P is None:
                return {'RUNNING_MODAL'}
            if self._start is None:
                self.fix_plane(context, P)
                self._start = np.asarray(P, float)
                self._lock = None
                self._reset_typing()
                self._preview(context, (P, None), event)
            else:
                return self._finish(context, self._rect(context, event, P))
        elif event.value == 'PRESS' and event.type == 'TAB' and typing:
            self._sx, self._sy = self._live(context)
            self._buf = ""
            self._field = "Y" if self._field == "X" else "X"
            self._preview(context, hover(), event)
        elif event.value == 'PRESS' and event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and typing:
            X, Y = self._live(context)
            P, _ = hover()
            if X is not None or Y is not None:
                return self._finish(context, self._rect(context, event, P if P is not None else self._start))
        elif event.value == 'PRESS' and event.type == 'BACK_SPACE':
            if self._buf:
                self._buf = self._buf[:-1]
            elif self._start is not None and (self._sx is not None or self._sy is not None):
                self._reset_typing()
            elif self._start is not None:
                self._start, self._plane = None, None
            self._preview(context, hover(), event)
        elif event.value == 'PRESS' and event.type == 'C' and not self._buf:
            self.from_center = not self.from_center
            self._preview(context, hover(), event)
        elif event.value == 'PRESS' and event.type in {'X', 'Y', 'Z'} and not self._buf and self._start is None:
            self.axis_key(event)
            self._preview(context, hover(), event)
        elif event.value == 'PRESS' and event.type in {'ESC', 'RIGHTMOUSE'}:
            self.sketch_end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context, rect):
        if rect is None or rect[1] <= 1e-9 or rect[2] <= 1e-9:
            self.report({'WARNING'}, "Plane: give it a size in X and in Y")
            return {'RUNNING_MODAL'}
        C, X, Y, (sx, sy) = rect
        O, n, u, v = self._plane
        self.size_x, self.size_y = X, Y
        self.data = json.dumps({"anchor": list(map(float, self._start)), "u": list(map(float, u)),
                                "v": list(map(float, v)), "sx": sx, "sy": sy})
        self.sketch_end(context)
        return self.execute(context)

    def execute(self, context):
        from mathutils import Matrix
        from . import features, store
        try:
            d = json.loads(self.data or "{}")
        except ValueError:
            d = {}
        if not d.get("anchor"):
            self.report({'WARNING'}, "Draw Plane is interactive: start it from the panel or menu")
            return {'CANCELLED'}
        A, u, v = (np.array(d[k], float) for k in ("anchor", "u", "v"))
        X, Y = float(self.size_x), float(self.size_y)
        C = A if self.from_center else A + 0.5 * d.get("sx", 1.0) * X * u + 0.5 * d.get("sy", 1.0) * Y * v
        params = {"width": X, "depth": Y, "pl_degree_u": int(self.degree_u), "pl_degree_v": int(self.degree_v),
                  "spans_u": int(self.spans_u), "spans_v": int(self.spans_v)}
        try:
            g = features.run("prim_plane", [], params)
        except Exception as exc:
            self.report({'ERROR'}, f"Plane failed: {exc}")
            return {'CANCELLED'}
        obj = store.new_object(context, "NURBS Plane", g, Matrix(plane_matrix(C, u, v).tolist()))
        features.record(obj, "prim_plane", [], params)
        _select_new(context, [obj])
        self.report({'INFO'}, f"Plane {X:.4g} x {Y:.4g}: degree {self.degree_u} x {self.degree_v}, "
                              f"{self.degree_u + self.spans_u} x {self.degree_v + self.spans_v} CVs")
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Profile: a circle / ellipse / square / rectangle perpendicular to a curve or
# surface edge (a section for Sweep and Pipe)
# --------------------------------------------------------------------------

def profile_size(shape, o, X, Y):
    """Profile sizes from the offset o of the mouse from the centre (in the
    profile's plane, axes X / Y): {radius, radius2} or {width, depth}."""
    dx, dy = abs(float(o @ X)), abs(float(o @ Y))
    if shape == "CIRCLE":
        return {"radius": float(np.hypot(dx, dy))}
    if shape == "ELLIPSE":
        return {"radius": dx, "radius2": dy}
    if shape == "SQUARE":
        return {"width": 2.0 * max(dx, dy)}
    return {"width": 2.0 * dx, "depth": 2.0 * dy}


class NPROTO_OT_draw_profile(SketchMixin, bpy.types.Operator):
    """Place a circle, ellipse, square or rectangle perpendicular to a curve or
    a surface edge: click on it for the centre, then drag (or type) the size.
    On a surface edge it stands square to the surface. With Construction
    History it stays perpendicular there when the curve or edge changes"""
    bl_idname = "nproto.draw_profile"
    bl_label = "Profile"
    bl_options = {'REGISTER', 'UNDO'}

    shape: EnumProperty(name="Shape", items=PROFILE_SHAPES, default="CIRCLE")
    radius: FloatProperty(name="Radius", default=0.25, min=1e-6, unit='LENGTH')
    radius2: FloatProperty(name="Radius 2", default=0.15, min=1e-6, unit='LENGTH')
    width: FloatProperty(name="Size X", default=0.5, min=1e-6, unit='LENGTH')
    depth: FloatProperty(name="Size Y", default=0.3, min=1e-6, unit='LENGTH')
    at: FloatProperty(name="Position", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
                      description="Where along the curve / edge (0 = its start, 1 = its end)")
    rotation: FloatProperty(name="Rotation °", default=0.0, soft_min=-180.0, soft_max=180.0,
                            description="Turn the profile about the curve's tangent")
    degree: EnumProperty(name="Degree", items=[("0", "Exact", "")] + DEGREE_ITEMS, default="0")
    spans_u: IntProperty(name="Spans", default=8, min=1, max=64)
    data: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.row().prop(self, "shape", expand=True)
        col = lay.column(align=True)
        if self.shape in ("CIRCLE", "ELLIPSE"):
            col.prop(self, "radius")
            if self.shape == "ELLIPSE":
                col.prop(self, "radius2")
            lay.row().prop(self, "degree", expand=True)
            if self.degree != "0":
                lay.prop(self, "spans_u")
        else:
            col.prop(self, "width", text="Size" if self.shape == "SQUARE" else "Size X")
            if self.shape == "RECTANGLE":
                col.prop(self, "depth")
        lay.prop(self, "at")
        lay.prop(self, "rotation")

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, f"{self.bl_label}: run it in a 3D View")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self._center, self._src, self._frame = None, None, None
        self._lock = None
        self._buf, self._field, self._s1, self._s2 = "", 1, None, None
        self.sketch_start(context)
        context.window_manager.modal_handler_add(self)
        self._preview(context, event)
        return {'RUNNING_MODAL'}

    # the "up" of a profile on a surface edge comes from one of its faces: every
    # (edge, face) pair at the point is a choice (T), corners included
    def ref_mode(self):
        return "SURFACE"

    def ref_candidates(self, P, src):
        out = []
        for s3, meta, W in self._snaps.through(P, src):
            faces = snapping.edge_faces(meta["obj"], meta["edge"]) if "edge" in meta else []
            for k, key, outline in faces:
                out.append({"src": s3 + (k,), "edge": W, "face": outline})
            if not faces:
                out.append({"src": s3 + (None,), "edge": W, "face": None})
        return out

    def _sizes(self):
        if self.shape in ("CIRCLE", "ELLIPSE"):
            return {"radius": self.radius, "radius2": self.radius2}
        return {"width": self.width, "depth": self.depth}

    def _params(self, sizes=None, at=None, face=None):
        p = {"pp_shape": self.shape, "radius": self.radius, "radius2": self.radius2, "width": self.width,
             "depth": self.depth, "pp_at": self.at if at is None else at, "pp_rotation": self.rotation,
             "pp_face": int(face or 0), "degree": int(self.degree), "spans_u": int(self.spans_u)}
        p.update(sizes or {})
        return p

    def _frame_at(self, src, P):
        from .primitives import curve_param_at, profile_frame
        obj = bpy.data.objects.get(src[0])
        at = curve_param_at(obj, src[1], P)
        return at, profile_frame(obj, src[1], at, self.rotation, src[3] or 0)

    def _live(self, context):
        a, b = self._s1, self._s2
        if self._buf:
            try:
                v = parse_length(context, self._buf)
                a, b = (v, b) if self._field == 1 else (a, v)
            except Exception:
                pass
        return a, b

    def _typed(self, context):
        """Typed sizes as profile params (the first field is the radius / size X)."""
        a, b = self._live(context)
        out = {}
        if self.shape in ("CIRCLE", "ELLIPSE"):
            if a is not None:
                out["radius"] = a
            if b is not None and self.shape == "ELLIPSE":
                out["radius2"] = b
        else:
            if a is not None:
                out["width"] = a
            if b is not None and self.shape == "RECTANGLE":
                out["depth"] = b
        return out

    def _mouse_sizes(self, context, event):
        """Sizes from the mouse, in the profile's plane (a plane seen edge-on:
        the point of the mouse ray nearest the centre)."""
        region, rv3d = self._win(context)
        m = Vector(self._mouse(context, event))
        o = np.array(view3d_utils.region_2d_to_origin_3d(region, rv3d, m), float)
        d = np.array(view3d_utils.region_2d_to_vector_3d(region, rv3d, m), float)
        M = self._frame
        X, Y, T, P = M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3]
        Q = ray_plane_point(o, d, P, T) if abs(float(d @ T)) > 0.05 * np.linalg.norm(d) else None
        if Q is None:
            Q = o + d * float((P - o) @ d) / float(d @ d)
        off = Q - P
        off -= float(off @ T) * T
        sz = profile_size(self.shape, off, X, Y)
        if event.ctrl:
            st = self._grid_step(context, P)
            sz = {k: max(st, round(v / st) * st) for k, v in sz.items()}
        sz.update(self._typed(context))
        return sz

    def _status(self, context, sizes=None):
        us = context.scene.unit_settings
        fmt = lambda x: bpy.utils.units.to_string(us.system, 'LENGTH', x * (us.scale_length or 1.0), precision=4)
        txt = f"Profile [{self.shape.title()}]: " + ("drag or type the size" if self._center is not None
                                                     else "click on a curve or surface edge")
        if self._center is not None and sizes:
            txt += "   " + "  ".join(f"{k.replace('width', 'X').replace('depth', 'Y').title()} {fmt(v)}"
                                     for k, v in sizes.items() if k in ("radius", "radius2", "width", "depth"))
            if self._buf:
                txt += f"   [{self._buf}_]"
        context.area.header_text_set(
            txt + "   C circle · E ellipse · S square · R rectangle · T which edge / face · L lock onto the edge · "
            "type the size · Tab second size · Enter place · Ctrl grid · Esc cancel")

    def _preview(self, context, event):
        from .primitives import profile_spec
        ov = draw.sketch_overlay
        ov["curves"], ov["cvs"], ov["refs"], ov["hull"] = [], [], [], []
        sizes = None
        if self._center is None:
            P, snap = self.pick_point(context, event)
            ov["snap"] = Vector(P) if snap is not None else None
            ov["snap_label"] = snap[3] if snap is not None else None
            ov["cursor"] = Vector(P) if P is not None else None
            self.show_refs()
            if snap is not None and snap[2] is not None:
                try:
                    _, M = self._frame_at(snap[2], P)
                except Exception:
                    M = None
            else:
                M = None
        else:
            M = self._frame
            sizes = self._mouse_sizes(context, event)
            ov["snap"], ov["snap_label"], ov["cursor"] = None, None, Vector(self._center)
            ov["cvs"] = [Vector(self._center)]
        if M is not None:
            sp = profile_spec(self._params(sizes))
            pl = spec_polyline(sp, 24)
            ov["curves"] = [[Vector(M[:3, :3] @ p + M[:3, 3]) for p in pl]]
            n = M[:3, 2]
            big = 0.5 * max(max(sizes.values()) if sizes else max(self._sizes().values()), 1e-3)
            ov["axes"] = [([Vector(M[:3, 3] - big * n), Vector(M[:3, 3] + big * n)], COL_TANGENT)]
        else:
            ov["axes"] = []
        self._status(context, sizes)
        context.area.tag_redraw()

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        placed = self._center is not None
        ch = event.unicode if event.value == 'PRESS' else ""
        if placed and ch and (ch in "0123456789.-+*/'\"" or (self._buf and (ch.isalpha() or ch == " "))) \
                and not (event.ctrl or event.alt or event.oskey):
            self._buf += ch
            self._preview(context, event)
            return {'RUNNING_MODAL'}
        if event.type in {'MOUSEMOVE', 'LEFT_CTRL', 'RIGHT_CTRL'}:
            self._preview(context, event)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not placed:
                P, snap = self.pick_point(context, event)
                if snap is None or snap[2] is None:
                    self.report({'WARNING'}, "Profile: click on a curve or a surface edge")
                    return {'RUNNING_MODAL'}
                try:
                    self.at, self._frame = self._frame_at(snap[2], P)
                except Exception as exc:
                    self.report({'WARNING'}, f"Profile: {exc}")
                    return {'RUNNING_MODAL'}
                self._center, self._src = np.asarray(P, float), snap[2]
                self._edge_lock = None
                self._preview(context, event)
            else:
                return self._finish(context, self._mouse_sizes(context, event))
        elif event.value == 'PRESS' and event.type == 'TAB' and placed:
            self._s1, self._s2 = self._live(context)
            self._buf = ""
            self._field = 2 if self._field == 1 else 1
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and placed:
            return self._finish(context, self._typed(context))     # nothing typed: the panel's sizes
        elif event.value == 'PRESS' and event.type == 'BACK_SPACE':
            if self._buf:
                self._buf = self._buf[:-1]
            elif self._s1 is not None or self._s2 is not None:
                self._s1 = self._s2 = None
            elif placed:
                self._center, self._src, self._frame = None, None, None
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type in {'C', 'E', 'S', 'R'} and not self._buf:
            self.shape = {"C": "CIRCLE", "E": "ELLIPSE", "S": "SQUARE", "R": "RECTANGLE"}[event.type]
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type == 'T' and not placed:
            self.pick_point(context, event)
            self.cycle_ref()
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type == 'L' and not placed:
            self.pick_point(context, event)
            self.edge_lock_key()
            self._preview(context, event)
        elif event.value == 'PRESS' and event.type in {'ESC', 'RIGHTMOUSE'}:
            self.sketch_end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context, sizes):
        for k, v in sizes.items():
            if v is not None and v > 1e-9:
                setattr(self, k, v)
        name, ref, snap, face = (list(self._src) + [None])[:4]
        self.data = json.dumps({"obj": name, "ref": ref, "face": face})
        self.sketch_end(context)
        return self.execute(context)

    def execute(self, context):
        from mathutils import Matrix
        from . import features
        from .primitives import profile_frame, profile_spec, write_spline_object
        try:
            d = json.loads(self.data or "{}")
        except ValueError:
            d = {}
        path = bpy.data.objects.get(d.get("obj", "")) if d.get("obj") else None
        if path is None:
            self.report({'WARNING'}, "Profile is interactive: start it from the panel or menu")
            return {'CANCELLED'}
        params = self._params(face=d.get("face"))
        try:
            M = profile_frame(path, d["ref"], self.at, self.rotation, d.get("face") or 0)
            sp = profile_spec(params)
        except Exception as exc:
            self.report({'ERROR'}, f"Profile failed: {exc}")
            return {'CANCELLED'}
        name = "Profile " + self.shape.title()
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        obj = bpy.data.objects.new(name, cu)
        (context.collection or context.scene.collection).objects.link(obj)
        obj.matrix_world = Matrix(M.tolist())
        write_spline_object(obj, sp)
        if features.history_enabled(context):
            features.record(obj, "perp_profile", [(path, d["ref"])], params)
            hist = f", follows {path.name} {d['ref']}"
        else:
            if self.shape in ("CIRCLE", "ELLIPSE"):
                features.record(obj, "prim_circle", [], {k: params[k] for k in ("radius", "radius2", "degree",
                                                                                  "spans_u")}
                                | {"elliptic": self.shape == "ELLIPSE"})
            else:
                features.record(obj, "prim_rectangle", [], {"width": params["width"],
                                                             "depth": params["width"] if self.shape == "SQUARE"
                                                             else params["depth"]})
            hist = ""
        _select_new(context, [obj])
        self.report({'INFO'}, f"{name} on {path.name} {d['ref']} at {self.at:.3f}{hist}")
        return {'FINISHED'}


classes = (NPROTO_OT_draw_curve, NPROTO_OT_draw_line, NPROTO_OT_draw_plane, NPROTO_OT_draw_profile)
