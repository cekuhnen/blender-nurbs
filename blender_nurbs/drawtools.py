"""Sketch tools: CV Curve and Line (interactive, click in the 3D view).

CV Curve
  CV mode            -- every click places a CV.
  Through Points     -- the curve passes through every click.
  Either way the result is one clean uniform NURBS spline (Endpoint on, no
  Bezier knot stacks: curvature-continuous, any CV can be moved without
  making a corner). Degree 1-5; C closes the curve (periodic).
Line
  Two clicks per line; each line is its own 2-CV curve (or, with Polyline,
  connected lines become one degree-1 curve). Once the start is placed, the
  end also snaps where the line meets a curve or edge at a right angle
  (Perpendicular) or touches it (Tangent; planar curves, in their plane).
  S: the start leaves tangent to, or square to, the curve it was snapped on;
  such a start slides along that curve to meet a second curve Perpendicular /
  Tangent (a line tangent to two circles). A / Shift+A over a line or curve:
  the direction stays parallel / square to it (linesolve.py).

Points land on the drawing plane, chosen by the first point: the plane of
the flat face under it (a mesh's or a NURBS surface's; axes along the face's
edges; outlined before the click; snapping.Targets.flat_face), else -- over a
curved surface, nothing, or with Alt -- the grid plane in an axis-aligned
orthographic view (Top / Front / Side), else the world's XY plane (the
floor grid; the upright XZ or YZ plane when the floor is seen nearly
edge-on), through the world origin (or through the first snapped point).
The 3D cursor plays no part: it places new objects (Shift+A), not sketches.
Ctrl snaps to that plane's grid. Clicks snap (see snapping.py) to
the ends, middles, CVs and any point of visible curves, to the vertices,
middles and any point of surface edges, to face centres, to the CVs of
surfaces whose CV net is shown, to the points drawn so far, and (Blender's
snap menu: the magnet and its elements) to mesh vertices, edges, edge and face
centres and faces. Ctrl snaps to
the grid, Shift to 15 degree angles from the previous point.
Locks: X / Y / Z along that world axis from the previous point (from the
world origin for the first), stopping where it crosses a smart guide;
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

from . import curvejoin, curvetools, geom, hints, linesolve, snapping

SNAP_PX = snapping.SNAP_PX
SPECIAL_PX = 2.0 * SNAP_PX      # how near a Perpendicular / Tangent point the mouse must be
MOUSE_SKETCH = [("MOUSE_LMB", "Place"), ("MOUSE_RMB", "Done"), ("EVENT_ESC", "Cancel")]
MOUSE_SHAPE = [("MOUSE_LMB", "Place"), ("EVENT_ESC", "Cancel")]
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


def preview_comb(sp, samples, scale):
    """Curvature comb (diagnostics.comb_geometry) of a spline spec: the CV
    Curve preview's, while it is drawn."""
    from . import diagnostics, store
    _, ext = store._k()
    deg, P, W, U = geom.blender_nurbs_to_clamped(sp["points"], sp["weights"], sp["order"], sp["cyclic"],
                                                 sp["endpoint"], sp["bezier"])
    rational = any(abs(w - 1.0) > 1e-12 for w in W)
    kc = ext.curve_from_cvs(int(deg), [tuple(map(float, q)) for q in P], [float(k) for k in U],
                            [float(w) for w in W] if rational else None)
    return diagnostics.kernel_comb(kc, samples, scale)


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


# --------------------------------------------------------------------------
# Tangent arcs in a polyline (Draw Line, R): an arc leaves the last point
# along the previous segment's end direction and ends where you click
# --------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, float)
    return v / (float(np.linalg.norm(v)) or 1.0)


def tangent_arc(S, t, E, tol=1e-6):
    """Middle point of the circular arc that leaves S along the unit direction t
    and ends at E (its sweep is twice the angle between t and the chord), or
    None when E lies on t's line (no arc: a straight segment)."""
    S, t, E = (np.asarray(x, float) for x in (S, t, E))
    d = E - S
    c = float(np.linalg.norm(d))
    if c < 1e-12:
        return None
    dh = d / c
    cos_phi = float(t @ dh)
    perp = t - cos_phi * dh
    sin_phi = float(np.linalg.norm(perp))
    if sin_phi < tol:
        return None
    r = c / (2.0 * sin_phi)
    return S + 0.5 * d + r * (1.0 - cos_phi) * (perp / sin_phi)


def arc_circle(S, M, E):
    """(centre, radius, unit x towards S, unit y the way the arc runs, sweep in
    radians) of the circle through S, M, E, running S -> M -> E."""
    S, M, E = (np.asarray(x, float) for x in (S, M, E))
    a, b = S - M, E - M
    axb = np.cross(a, b)
    den = 2.0 * float(axb @ axb)
    C = M + (np.cross(axb, a) * float(b @ b) + np.cross(b, axb) * float(a @ a)) / den
    r = float(np.linalg.norm(S - C))
    x = (S - C) / r
    m = M - C
    y = m - float(m @ x) * x
    y /= float(np.linalg.norm(y))
    half = math.atan2(float(m @ y), float(m @ x))       # S -> M: half the sweep (up to 180 degrees)
    return C, r, x, y, 2.0 * half


def arc_points(S, M, E, n=24):
    """n + 1 points along the arc S -> M -> E."""
    C, r, x, y, th = arc_circle(S, M, E)
    return [C + r * (math.cos(a) * x + math.sin(a) * y) for a in np.linspace(0.0, th, n + 1)]


def seg_end_tangent(a, b, mid=None):
    """The unit direction a segment ends with: a line's own, an arc's tangent at its end."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if mid is None:
        d = b - a
        return d / (float(np.linalg.norm(d)) or 1.0)
    C, r, x, y, th = arc_circle(a, mid, b)
    return -math.sin(th) * x + math.cos(th) * y


def arc_bezier(S, M, E):
    """(points, weights) of the exact arc S -> M -> E as rational quadratic
    Bezier pieces of at most 90 degrees (order 3, Bezier knots)."""
    C, r, x, y, th = arc_circle(S, M, E)
    k = max(1, int(math.ceil(th / (math.pi / 2) - 1e-9)))
    dt = th / k
    at = lambda a, rr=r: C + rr * (math.cos(a) * x + math.sin(a) * y)
    pts, w = [np.asarray(S, float)], [1.0]
    for j in range(k):
        t0, t1 = j * dt, (j + 1) * dt
        pts.append(at(0.5 * (t0 + t1), r / math.cos(dt / 2)))
        w.append(math.cos(dt / 2))
        pts.append(np.asarray(E, float) if j == k - 1 else at(t1))
        w.append(1.0)
    return pts, w


def chain_segments(segs, tol=1e-9):
    """Connected segments (a, b, arc middle or None) -> chains, in drawing order."""
    out = []
    for a, b, m in segs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if out and np.linalg.norm(out[-1][-1][1] - a) <= tol:
            out[-1].append((a, b, m))
        else:
            out.append([(a, b, m)])
    return out


def segments_spec(chain, tol=1e-9):
    """One spline spec for a chain of lines and arcs: exact, order 3 with
    Bezier knots (a line is a straight Bezier piece), cyclic when it closes."""
    from .primitives import spec
    pts, w = [], []
    for k, (a, b, m) in enumerate(chain):
        if m is None:
            P, W = [np.asarray(a, float), 0.5 * (np.asarray(a, float) + np.asarray(b, float)), np.asarray(b, float)], \
                [1.0, 1.0, 1.0]
        else:
            P, W = arc_bezier(a, m, b)
        if k:
            P, W = P[1:], W[1:]
        pts += P
        w += W
    closed = bool(len(chain) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= tol)
    if closed:
        pts, w = pts[:-1], w[:-1]
    return spec([tuple(p) for p in pts], w, order=3, cyclic=closed, endpoint=True, bezier=True)


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
# number-pad keys that type a value once a tool takes typing (Blender's Emulate
# Numpad sends the top-row digits as these too); before that they change the view
DEGREE_PAD = {f"NUMPAD_{i}": i for i in range(1, 6)}
NUMPAD_CHARS = {f"NUMPAD_{i}": str(i) for i in range(10)} | {
    "NUMPAD_PERIOD": ".", "NUMPAD_MINUS": "-", "NUMPAD_PLUS": "+", "NUMPAD_SLASH": "/", "NUMPAD_ASTERIX": "*"}
GUIDE_PX = 6.0          # a smart guide snaps within this many pixels
EDGE_ON_DEG = 12.0      # a free view this close to level with the floor draws on an upright world plane
GUIDE_LABEL = {"align": "Aligned", "tangent": "Tangent", "extend": "Extension", "perp": "Perpendicular",
               "parallel": "Parallel"}
COL_GUIDE = (0.75, 0.75, 0.75, 0.9)
POINT_KINDS = {"first", "end", "mid", "center", "cv", "vertex"}
REFRESH_KEYS = {'MOUSEMOVE', 'LEFT_CTRL', 'RIGHT_CTRL', 'LEFT_SHIFT', 'RIGHT_SHIFT', 'LEFT_ALT', 'RIGHT_ALT'}
PASS_KEYS = {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE',
             'NDOF_MOTION', 'NUMPAD_0', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_5', 'NUMPAD_6',
             'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_PERIOD', 'NUMPAD_PLUS', 'NUMPAD_MINUS', 'ACCENT_GRAVE', 'INBETWEEN_MOUSEMOVE'}


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


# --------------------------------------------------------------------------
# The viewport grid (as Blender draws it), for Ctrl grid snapping
# --------------------------------------------------------------------------

# Blender's length units (its unit table), smallest first, in metres; the grid
# has a line level at each, then x10 levels to fill GRID_STEPS_LEN
GRID_UNITS = {"METRIC": [1e-6, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],        # um mm cm dm m dam hm km
              "IMPERIAL": [0.0000254, 0.0254, 0.3048, 0.9144, 20.1168, 201.168, 1609.344]}  # thou in ft yd ch fur mi
GRID_STEPS_LEN = 8
GRID_FADE_PX = 4.0      # a grid level starts to fade in when its lines are this far apart


def grid_steps(system, scale_length=1.0, grid_scale=1.0, subdivisions=10):
    """The viewport grid's line levels in Blender units, finest first (Blender's
    ED_view3d_grid_steps for an axis view): one per unit of the unit system,
    or Grid Scale times powers of the Subdivisions without units."""
    units = GRID_UNITS.get(system)
    if units is None:
        return [grid_scale * subdivisions ** i for i in range(GRID_STEPS_LEN)]
    steps = [u * grid_scale / scale_length for u in units]
    while len(steps) < GRID_STEPS_LEN:
        steps.append(10.0 * steps[-1])
    return steps


# the base unit Blender's floor grid stops at in any view but Top / Front / Side
# (perspective, or orthographic turned): it skips the levels below it there,
# which the axis views draw (ED_view3d_grid_steps)
GRID_BASE = {"METRIC": 1.0, "IMPERIAL": 0.3048}


def fine_grid_levels(system, scale_length=1.0, grid_scale=1.0, subdivisions=10):
    """(levels, base) for the fine floor grid (draw.py): base is the finest
    level Blender draws on the floor of a perspective or turned orthographic
    view (1 m, 1 ft, times the Grid Scale); levels are four of the axis-view
    levels, finest first, starting at most three below base (mm, cm, 10 cm;
    thou, inch), so those views get the levels a Top view has. None when the floor lacks none
    (no units)."""
    steps = grid_steps(system, scale_length, grid_scale, subdivisions)
    if system not in GRID_BASE:
        return None
    base = GRID_BASE[system] * grid_scale / (scale_length or 1.0)
    fine = [k for k, st in enumerate(steps) if st < base * (1 - 1e-9)]
    if not fine:
        return None
    i0 = max(0, len(fine) - 3)
    lv = steps[i0:i0 + 4]
    lv += [lv[-1]] * (4 - len(lv))
    return lv, base


def visible_grid_step(steps, ppu):
    """The finest grid level drawn at least half visible at ppu pixels per unit.
    The viewport fades a level in from GRID_FADE_PX apart (invisible) to the
    level below it reaching GRID_FADE_PX (fully drawn), alpha cubed; below half
    its lines are hard to see, so the next level up is the one to snap to."""
    px = [s * ppu for s in steps]
    i = next((k for k, p in enumerate(px) if p >= GRID_FADE_PX), len(px) - 1)
    s0 = px[i - 1] if i > 0 else 0.0
    t = (GRID_FADE_PX - s0) / max(px[i] - s0, 1e-12)
    blend = 1.0 - min(max(t, 0.0), 1.0)
    if blend ** 3 >= 0.5 or i + 1 >= len(steps):
        return steps[i]
    return steps[i + 1]


def pixels_per_unit(region, rv3d, at, axes):
    """Screen pixels per world unit at the point `at` along each of `axes`,
    the smallest: the exact derivative of the view's projection there (a
    finite step of a whole unit went behind the camera, or far off, in a
    close perspective view). None behind the camera."""
    M = np.array(rv3d.perspective_matrix, float)
    q = M @ np.append(np.asarray(at, float), 1.0)
    if q[3] <= 1e-12 or not axes:
        return None
    half = np.array([region.width, region.height], float) * 0.5
    rates = []
    for a in axes:
        da = M[:, :3] @ np.asarray(a, float)
        rates.append(float(np.linalg.norm((da[:2] * q[3] - q[:2] * da[3]) / (q[3] * q[3]) * half)))
    return max(min(rates), 1e-9)


class SketchMixin:
    """Point picking on a drawing plane with snapping."""

    face_planes = True          # a first point on a flat face draws on that face's plane

    def sketch_start(self, context):
        self._plane = None          # (origin, normal, u, v), fixed by the first point
        self._plane_face = None     # the flat face (snapping.Targets.flat_face) the plane lies on, or None
        self._face_hover = None     # the flat face under the mouse the next first point would draw on
        self._snaps = self._snap_targets(context)
        self._hover = None          # (point, snap info)
        self._refs = None           # references at the hovered snap: {"keys", "cands", "i"}
        self._hover_cand = None
        self._edge_lock = None      # (object, ref): L locked the point onto this edge / curve
        self._last_edge = None      # the edge last slid along: it stays the reference at its ends
        self._acquired = []         # points snapped to lately: smart guides align to them too
        draw.sketch_overlay.update(curves=[], hull=[], cvs=[], snap=None, snap_label=None, axes=[],
                                   refs=[], cursor=None, comb=None, guides=[], face=[], face_pts=[])

    def sketch_end(self, context):
        draw.sketch_overlay.update(curves=[], hull=[], cvs=[], snap=None, snap_label=None, axes=[],
                                   refs=[], cursor=None, comb=None, guides=[], face=[], face_pts=[])
        hints.clear(context)

    # -- key hints (hints.py) shared by the sketch tools ----------------------
    def direction_hints(self, extra=()):
        """Direction keys; the active lock lit."""
        lk = getattr(self, "_lock", None)
        on = lambda c: True if c else None
        return ([hints.key("X Y Z", "axis", on=on(lk in ("X", "Y", "Z"))),
                 hints.key("Shift+X/Y/Z", "plane", on=on(lk in ("-X", "-Y", "-Z")))]
                + list(extra) + [hints.key("Shift", "15°")])

    def reference_hints(self, mode):
        """Which edge / face an end follows: T next, L lock onto it, F edge or face."""
        return [hints.key("T", "next edge / face"),
                hints.key("L", "lock onto the edge", on=True if getattr(self, "_edge_lock", None) else None),
                hints.cycle("F", ["edge", "face"], "face" if mode == "SURFACE" else "edge")]

    def hints_key(self, context, event):
        """? / F1: collapse or expand the key panel. True when handled."""
        if hints.handle(event):
            context.area.tag_redraw()
            return True
        return False

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
        """The drawing plane where no flat face chose one: in a Top / Front /
        Side view that view's grid plane; in any other view (perspective or
        turned) the world's XY plane -- the floor grid -- or, seen nearly
        edge-on (within EDGE_ON_DEG), whichever of the world's XZ / YZ planes
        faces the view more. Never the screen plane (it turned with every
        orbit and matched no grid), never the 3D cursor's (Blender turns the
        cursor to the view when it is placed, which tilted the sketch)."""
        rv3d = self._win(context)[1]
        vdir = np.array(rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)), float)
        axis = int(np.argmax(np.abs(vdir)))
        if not rv3d.is_perspective and abs(abs(vdir[axis]) - 1.0) < 1e-4:
            n = np.zeros(3)
            n[axis] = 1.0
            u, v = {2: ((1, 0, 0), (0, 1, 0)), 1: ((1, 0, 0), (0, 0, 1)), 0: ((0, 1, 0), (0, 0, 1))}[axis]
            u, v = np.array(u, float), np.array(v, float)
            return np.asarray(through, float), n, u, v
        ex, ey, ez = np.eye(3)
        if abs(float(vdir[2])) >= math.sin(math.radians(EDGE_ON_DEG)):
            n = ez if vdir[2] < 0 else -ez                   # facing the viewer
            u = ex
            v = np.cross(n, u)                               # seen from below the floor: its Y flips
        else:
            n = ey if abs(float(vdir[1])) >= abs(float(vdir[0])) else ex
            n = n if float(vdir @ n) < 0 else -n
            v = ez                                           # upright: up stays up
            u = np.cross(v, n)
        return np.asarray(through, float), n, u, v

    @staticmethod
    def _grid_origin(O, face=None):
        """Where Ctrl's grid lines cross on a plane through O: the world origin
        when its axes are world axes (the grid Blender draws), else -- a
        slanted face -- the face's centre, so the grid sits square on it."""
        if face is None:
            return np.zeros(3)
        aligned = lambda a: float(np.abs(np.asarray(a, float)).max()) > 1 - 1e-9
        if aligned(face["u"]) and aligned(face["v"]):
            return np.zeros(3)
        return np.asarray(face.get("c", O), float)

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
        """Snap targets of the visible curves, surfaces and meshes (snapping.Targets)."""
        return snapping.collect(context.visible_objects, lambda o: draw.cv_net_shown(context, o),
                                snapping.sketch_mesh_elements)

    def _view(self, context):
        region, rv3d = self._win(context)
        return snapping.View(rv3d.perspective_matrix, rv3d.view_matrix, region.width, region.height)

    def _grid_step(self, context, at, u=None, v=None):
        """The grid step Ctrl snaps to here: the finest grid level the view
        draws at least half visible (visible_grid_step), so points land on
        lines you can see, never between them. The pixels per unit are taken
        along the drawing plane's axes u and v at the point (the plane's, or
        before the first point the one the point lands on), the shorter of
        the two -- as the fine floor grid (draw.py) fades its levels -- from
        the projection's derivative, so a close perspective view measures
        right too."""
        if u is None:
            pl = self._plane or self._view_plane(context, at)
            u, v = pl[2], pl[3]
        ppu = pixels_per_unit(*self._win(context), at, [a for a in (u, v) if a is not None])
        if ppu is None:
            return 1.0
        us = context.scene.unit_settings
        ov = getattr(getattr(context.area, "spaces", None) and context.area.spaces.active, "overlay", None)
        steps = grid_steps(us.system, us.scale_length or 1.0, getattr(ov, "grid_scale", 1.0) or 1.0,
                           getattr(ov, "grid_subdivisions", 10) or 10)
        return visible_grid_step(steps, ppu)

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
            label = snapping.LABELS.get(snap[0]) or (snap[0][6:] if snap[0].startswith("guide:") else None)
            snap = (snap[0], snap[1], None, label)
        if snap is not None and snap[0] in POINT_KINDS and P is not None:
            self._acquire(P)            # guides align to the points you snap to
        return P, snap

    def drawn_segments(self):
        """[(A, B)] the lines drawn so far in this tool run: they snap too
        (their middles, any point on them). Draw Line has them."""
        return ()

    # -- smart guides: lines in the drawing plane the point can snap onto ------
    use_guides = True

    def guide_points(self):
        """Points of this run that guides align to (horizontal / vertical in the plane)."""
        return []

    def guide_dirs(self, prev):
        """[(unit direction, kind)] of guides through the previous point."""
        return []

    def guide_axes(self):
        """Directions of the sketch's straight lines, newest first: guides run
        parallel and perpendicular to them, through every guide point."""
        return []

    def _acquire(self, P):
        """A point snapped to: guides align to it for a while (the 3 latest)."""
        P = np.asarray(P, float)
        acq = [q for q in (getattr(self, "_acquired", None) or []) if np.linalg.norm(q - P) > 1e-9]
        self._acquired = [P] + acq[:2]

    def _guide_lines(self, context, prev, plane):
        """[(base, unit direction, kind)] in the drawing plane."""
        O, n, u, v = plane
        flat = lambda p: np.asarray(p, float) - float((np.asarray(p, float) - O) @ n) * n
        pts = []
        for p in list(self.guide_points()) + list(getattr(self, "_acquired", None) or []):
            q = flat(p)
            if all(np.linalg.norm(q - r) > 1e-9 for r in pts):
                pts.append(q)
        dirs = [(u, "align"), (v, "align")]
        n_sketch = 0
        for d in self.guide_axes():
            d = np.asarray(d, float)
            d = d - float(d @ n) * n
            if np.linalg.norm(d) < 1e-9:
                continue
            d = _unit(d)
            fresh = False
            for e, kind in ((d, "parallel"), (_unit(np.cross(n, d)), "perp")):
                if all(abs(float(e @ f)) < 1 - 1e-6 for f, _ in dirs):
                    dirs.append((e, kind))
                    fresh = True
            n_sketch += fresh
            if n_sketch >= 4:               # the 4 newest directions of the sketch
                break
        out = [(q, e, kind) for q in pts for e, kind in dirs]
        if prev is not None:
            out += [(flat(prev), _unit(d), k) for d, k in self.guide_dirs(prev)]
        return out

    def _dashes(self, context, a, b):
        """Dashes (2-point polylines) from a to b, about 6 px long."""
        pa, pb = self._to_px(context, a), self._to_px(context, b)
        n = 1 if pa is None or pb is None else int(min(max(np.linalg.norm(pb - pa) / 6.0, 1), 400))
        out = []
        for k in range(0, n, 2):
            out.append([Vector(a + (b - a) * k / n), Vector(a + (b - a) * min(k + 1, n) / n)])
        return out

    def _guide_snap(self, context, mouse, prev, plane, along=None):
        """The point the smart guides give near the mouse, or None: (point,
        label, dashes). Where two guides cross near the mouse, that crossing;
        else the nearest guide within GUIDE_PX. along=(base, direction): only
        where that line (a lock) crosses a guide."""
        if not getattr(self, "use_guides", True):
            return None
        mouse = np.asarray(mouse, float)
        lines = self._guide_lines(context, prev, plane)
        if along is not None:
            lines = [(b, e, k) for b, e, k in lines if abs(float(_unit(along[1]) @ e)) < 1 - 1e-6]
        near = []
        if lines:
            view = self._view(context)
            Bs = np.array([b for b, _, _ in lines], float)
            Es = np.array([e for _, e, _ in lines], float)
            a, _, oka = view.project(Bs)
            b, _, okb = view.project(Bs + Es * max(float(np.ptp(Bs, axis=0).max()), 1.0))
            w = b - a
            L = np.linalg.norm(w, axis=1)
            good = oka & okb & (L > 1e-9)
            w = w / np.where(L > 1e-9, L, 1.0)[:, None]
            dist = np.abs((mouse - a)[:, 0] * w[:, 1] - (mouse - a)[:, 1] * w[:, 0])
            for k in np.flatnonzero(good & ((dist <= GUIDE_PX) if along is None else True)):
                near.append((float(dist[k]), lines[k][0], lines[k][1], lines[k][2]))
        if not near:
            return None
        near.sort(key=lambda g: g[0])

        def cross(b1, e1, b2, e2):
            M = np.stack([e1, -e2], axis=1)
            st, *_ = np.linalg.lstsq(M, b2 - b1, rcond=None)
            X = b1 + st[0] * e1
            return X if np.linalg.norm(X - (b2 + st[1] * e2)) < 1e-9 * (1 + np.linalg.norm(X)) else None

        best = None
        if along is not None:
            for dist, base, e, kind in near:
                X = cross(np.asarray(along[0], float), _unit(along[1]), base, e)
                px = self._to_px(context, X) if X is not None else None
                if px is not None and np.linalg.norm(px - mouse) <= GUIDE_PX * 1.5 \
                        and (best is None or np.linalg.norm(px - mouse) < best[0]):
                    best = (float(np.linalg.norm(px - mouse)), X, [(base, kind)])
        else:
            for i in range(min(len(near), 6)):
                for j in range(i + 1, min(len(near), 6)):
                    _, b1, e1, k1 = near[i]
                    _, b2, e2, k2 = near[j]
                    if abs(float(e1 @ e2)) > 1 - 1e-6:
                        continue
                    X = cross(b1, e1, b2, e2)
                    px = self._to_px(context, X) if X is not None else None
                    if px is not None and np.linalg.norm(px - mouse) <= GUIDE_PX * 1.5 \
                            and (best is None or np.linalg.norm(px - mouse) < best[0]):
                        best = (float(np.linalg.norm(px - mouse)), X, [(b1, k1), (b2, k2)])
            if best is None:
                _, base, e, kind = near[0]
                Pm = self._ray_point(context, Vector(mouse), plane)
                if Pm is None:
                    return None
                X = base + float((Pm - base) @ e) * e
                best = (0.0, X, [(base, kind)])
        if best is None:
            return None
        X = best[1]
        labels = []
        for _, kind in best[2]:
            if GUIDE_LABEL[kind] not in labels:
                labels.append(GUIDE_LABEL[kind])
        dashes = [d for base, _ in best[2] for d in self._dashes(context, base, X)]
        return X, " · ".join(labels), dashes, len(best[2])

    def _guided(self, g):
        """A guide snap as _pick's (point, snap): its dashes drawn, its label shown."""
        draw.sketch_overlay["guides"] = [(g[2], COL_GUIDE)]
        return g[0], ("guide:" + g[1], None, None)

    def snap_key(self, context, event):
        """O: snap onto the mesh face under the mouse (the point lies on the
        mesh, off the drawing plane) on / off; kept for the session. True when
        it was that key (the caller refreshes the preview)."""
        if event.type != 'O' or event.value != 'PRESS' or event.ctrl or getattr(event, "alt", False) \
                or getattr(event, "oskey", False) or getattr(self, "_buf", ""):
            return False
        snapping.SNAP_STATE["surface"] = not snapping.SNAP_STATE["surface"]
        try:
            self._status(context)
        except Exception:
            pass
        return True

    @staticmethod
    def snap_hints():
        return [hints.key("Ctrl", "grid"), hints.key("Alt", "no snapping"),
                hints.toggle("O", "on surface", snapping.SNAP_STATE["surface"])]

    # -- drawing on a flat face ---------------------------------------------
    def plane_open(self):
        """Does the next click choose the drawing plane? (no point yet)"""
        return self._plane is None

    def _face_under(self, context, mouse, free=False):
        """The flat face under the mouse whose plane the next click draws on
        (snapping.Targets.flat_face), kept for fix_plane and outlined; None
        once the plane is chosen, with Alt, or over no flat face."""
        f = None
        if self.face_planes and not free and self.plane_open():
            try:
                f = self._snaps.flat_face(self._view(context), np.asarray(mouse, float))
            except Exception as exc:
                print("NURBS face plane:", exc)
        self._face_hover = f
        shown = self._shown_face()
        draw.sketch_overlay["face"] = list(shown["seg"]) if shown is not None else []
        draw.sketch_overlay["face_pts"] = [] if shown is None else \
            [tuple(map(float, q)) for q in shown.get("corners", ())] + [tuple(map(float, q)) for q in shown.get("mids", ())]
        return f

    def _shown_face(self):
        """The flat face outlined now: the one under the mouse before the first
        point, then the one the sketch is drawn on."""
        f = getattr(self, "_face_hover", None)
        if f is None and self._plane is not None:
            f = getattr(self, "_plane_face", None)
        return f

    @staticmethod
    def on_face(f, Q):
        """Does point Q lie on the plane of the flat face f?"""
        return f is not None and Q is not None \
            and abs(float((np.asarray(Q, float) - f["P"]) @ f["n"])) <= 1e-4 * f["size"] + 1e-9

    def fix_plane(self, context, P, on=()):
        """Choose the drawing plane at the first point P: the plane of the flat
        face under the mouse when P (and the points `on`) lie on it -- axes
        along its edges --, else the view plane through P (the grid plane in
        a Top / Front / Side view)."""
        if not self.plane_open():
            return
        f = getattr(self, "_face_hover", None)
        if f is not None and all(self.on_face(f, Q) for Q in (P, *on)):
            self._plane = (np.asarray(P, float), f["n"], f["u"], f["v"])
            self._plane_face = f
        elif self._plane is None or getattr(self, "_plane_face", None) is not None:
            self._plane = self._view_plane(context, P)
            self._plane_face = None

    def _pick(self, context, event, extra=(), prev=None):
        mouse = self._mouse(context, event)
        free = bool(getattr(event, "alt", False))       # Alt held: a free point, nothing snaps
        draw.sketch_overlay["guides"] = []
        f = self._face_under(context, mouse, free)
        shown = self._shown_face()
        if shown is not None and shown.get("snap"):     # a mesh face's corners and edge middles (the squares) snap
            extra = list(extra) + list(shown["snap"])
        if f is not None:                               # before the first point: on the flat face under the mouse
            plane = (f["P"], f["n"], f["u"], f["v"])
        else:
            plane = self._plane or self._view_plane(context, np.zeros(3))
        O, n, u, v = plane
        el = getattr(self, "_edge_lock", None)
        k = self._snaps.line_index(*el) if el else None
        if k is not None:
            # locked onto an edge (L): the point slides along it, wherever the mouse is
            P, kind, src = self._snaps.slide(self._view(context), mouse, k)
            return P, (kind, None, src)
        lock = getattr(self, "_lock", None)
        if lock and (lock[-1] in "XYZT" or (lock == "A" and prev is not None)):
            # world axis lock (X / Y / Z) or plane lock (Shift+X / Y / Z), from
            # the previous point, or from the world origin for the first one; a
            # direction taken from a line (A) only once there is a previous point
            base = prev if prev is not None else np.zeros(3)
            hit = None if free else self._snaps.pick(self._view(context), mouse, extra, edges=False,
                                                     segments=self.drawn_segments())
            region, rv3d = self._win(context)
            ray = (np.array(view3d_utils.region_2d_to_origin_3d(region, rv3d, Vector(mouse)), float),
                   np.array(view3d_utils.region_2d_to_vector_3d(region, rv3d, Vector(mouse)), float))
            step = self._grid_step(context, base, u, v) if event.ctrl and hit is None else None
            vec = getattr(self, "_tvec", None) if lock[-1] == "T" else getattr(self, "_avec", None) \
                if lock == "A" else None
            if hit is None and not free and not event.ctrl and lock[0] != "-":
                along = AXES.get(lock, vec)
                if along is not None and np.linalg.norm(along) > 1e-12:
                    g = self._guide_snap(context, mouse, prev, plane, along=(base, np.asarray(along, float)))
                    if g is not None:
                        return self._guided(g)
            P = axis_lock_point(lock, base, ray, None if hit is None else hit[0],
                                self._ray_point(context, Vector(mouse), plane), step, vec)
            return P, None
        axis = None
        if prev is not None:
            alock = getattr(self, "_angle_lock", None)
            if alock is not None:
                axis = math.cos(math.radians(alock)) * u + math.sin(math.radians(alock)) * v
        # points (ends, middles, centres, CVs) always snap; edges only when
        # nothing else is in charge (a lock, Ctrl grid, Shift angles); Alt: nothing
        hit = None if free else self._snaps.pick(self._view(context), mouse, extra,
                                                 edges=axis is None and not event.ctrl and not event.shift,
                                                 segments=self.drawn_segments())
        best = None if hit is None else (0.0, hit[0], hit[2], hit[1])
        guides = not free and not event.ctrl and not event.shift
        if best is not None and axis is None:
            if hit[1] not in POINT_KINDS and guides:
                g = self._guide_snap(context, mouse, prev, plane)
                if g is not None and g[3] >= 2:         # two guides crossing beat an edge
                    return self._guided(g)
            return best[1], (best[3], best[2], hit[3])
        if axis is not None:
            # locked direction (a typed angle): slide along
            # the line through the previous point; a snapped point is projected
            # onto it, so the line lines up with that point -- or with a guide
            if best is None and not free and not event.ctrl:
                g = self._guide_snap(context, mouse, prev, plane, along=(prev, axis))
                if g is not None:
                    return self._guided(g)
            P = best[1] if best is not None else self._ray_point(context, Vector(mouse), plane)
            if P is None:
                return None, None
            r = float((P - prev) @ axis)
            if event.ctrl and best is None:
                st = self._grid_step(context, P, u, v)
                r = round(r / st) * st
            return prev + r * axis, None
        if guides:
            g = self._guide_snap(context, mouse, prev, plane)
            if g is not None:
                return self._guided(g)
        P = self._ray_point(context, Vector(mouse), plane)
        if P is None:
            return None, None
        if event.shift and prev is not None:
            d = P - prev
            a = math.atan2(float(d @ v), float(d @ u))
            a = round(a / ANGLE_STEP) * ANGLE_STEP
            r = float(np.linalg.norm([d @ u, d @ v]))
            if event.ctrl:
                st = self._grid_step(context, P, u, v)
                r = max(st, round(r / st) * st)
            P = prev + r * (math.cos(a) * u + math.sin(a) * v)
        elif event.ctrl:
            st = self._grid_step(context, P, u, v)
            face = f if f is not None else (getattr(self, "_plane_face", None) if self._plane is not None else None)
            g0 = self._grid_origin(O, face)           # the world grid, or a slanted face's own
            x, y = float((P - O) @ u), float((P - O) @ v)
            ox, oy = float((O - g0) @ u), float((O - g0) @ v)
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
        if lk == "A":
            return " · " + ("parallel to" if getattr(self, "_akind", "parallel") == "parallel" else "square to") \
                + " the line (A)"
        if lk == "T":
            return " · square to the edge" if getattr(self, "_tkind", "tangent") == "square" else " · tangent to the edge"
        if lk == "-T":
            return " · in the face's tangent plane"
        if lk and lk[-1] in "XYZ":
            if lk[0] == "-":
                return " · plane " + "XYZ".replace(lk[-1], "") + f" (not {lk[-1]})"
            return f" · along {lk}"
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
        if lk and (lk[-1] in "XYZT" or (lk == "A" and prev is not None)):
            base = np.asarray(prev if prev is not None else np.zeros(3), float)
            rv3d = self._win(context)[1]
            big = 50.0 * max(float(getattr(rv3d, "view_distance", 10.0)), 1e-3)
            if lk == "A":
                dirs, cols = [np.asarray(self._avec, float)], [COL_TANGENT]
            elif lk[-1] == "T":
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

    def typing_ready(self):
        """Typed values are taken now (the tool's first point is placed)."""
        return False

    def numpad_keys(self, event):
        """Number-pad keys this tool uses itself (not the view)."""
        return False

    @staticmethod
    def typed_char(event):
        """The character a key press types: its text, or a number-pad key's."""
        if event.value != 'PRESS':
            return ""
        return getattr(event, "unicode", "") or NUMPAD_CHARS.get(event.type, "")

    def passthrough(self, context, event):
        from .ops import PickMixin
        if event.type in NUMPAD_CHARS and (self.typing_ready() or self.numpad_keys(event)):
            return False            # a digit being typed (or a degree key), not a view change
        if event.type in PASS_KEYS:
            return True
        # outside the view (the sidebar, a header): clicks and mouse moves go to
        # the UI -- its buttons only respond after the mouse has moved over them
        return event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event)


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
START_MODES = [("FREE", "Free", "The line leaves its start in any direction"),
               ("TANGENT", "Tangent", "A line started on an edge or curve leaves tangent to it, and slides along it "
                                      "to meet a second curve Tangent / Perpendicular"),
               ("PERP", "Perpendicular", "A line started on an edge or curve leaves square to it, and slides along "
                                         "it to meet a second curve Tangent / Perpendicular")]
START_TEXT = {"FREE": "free", "TANGENT": "tangent", "PERP": "perpendicular"}
SLIDE_LABEL = {("tan", "tan"): "Tangent to both", ("perp", "perp"): "Perpendicular to both",
               ("tan", "perp"): "Tangent · Perpendicular", ("perp", "tan"): "Perpendicular · Tangent"}


class NPROTO_OT_draw_curve(SketchMixin, bpy.types.Operator):
    """Draw a NURBS curve by clicking: CVs, or points the curve passes through.
    The result is one clean uniform spline (no kinks when you move CVs later).
    An end snapped onto a curve or surface edge can be G0 / G1 / G2 to it"""
    bl_idname = "nproto.draw_curve"
    bl_label = "CV Curve"
    bl_options = {'REGISTER', 'UNDO'}

    def guide_points(self):
        return list(getattr(self, "_pts", []))

    def guide_axes(self):
        P = [np.asarray(p, float) for p in getattr(self, "_pts", [])]
        return [_unit(P[k + 1] - P[k]) for k in range(len(P) - 2, -1, -1) if np.linalg.norm(P[k + 1] - P[k]) > 1e-12]

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
        conts = ["G0", "G1", "G2"]
        rows = [("Place", [hints.key("Backspace", "undo"), hints.toggle("C", "close", self.closed),
                           hints.key("Enter", "done")]),
                ("Direction", self.direction_hints())]
        if not self.closed:
            rows.append(("Ends", [hints.cycle("S", conts, self.start_cont, "start"),
                                  hints.cycle("E", conts, self.end_cont, "end")]))
        rows += [("Snap", self.snap_hints()), ("Reference", self.reference_hints(self.edge_mode)),
                 ("Curve", [hints.cycle("M", ["CVs", "through points"],
                                        "CVs" if self.mode == "CV" else "through points"),
                            hints.cycle("1-5", [str(d) for d in range(1, 6)], str(self.degree), "degree")]),
                ("Show", [hints.toggle("K", "curvature comb",
                                       bool(getattr(context.window_manager.nproto_view, "show_comb", False)))])]
        hints.show(context, "CV Curve", [f"{n} point(s)", self.lock_text().strip(" ·")], rows, MOUSE_SKETCH)

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
        ov["comb"] = None
        if len(pts) >= 2:
            try:
                sp, _ = self._spec(pts, [self._src(x) for x in sn])
                ov["curves"] = [[Vector(p) for p in spec_polyline(sp, 16)]]
                ov["hull"] = [Vector(p) for p in sp["points"]]
                v = getattr(context.window_manager, "nproto_view", None)
                if v is not None and v.show_comb:
                    ov["comb"] = preview_comb(sp, int(v.comb_samples), v.comb_scale)
            except Exception:
                pass
        context.area.tag_redraw()

    def numpad_keys(self, event):
        # Emulate Numpad sends the top-row 1-5 as number-pad keys: here they set the degree
        return event.type in DEGREE_PAD and bpy.context.preferences.inputs.use_emulate_numpad

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        prev = self._pts[-1] if self._pts else None
        first = [(self._pts[0], None, "first")] if len(self._pts) >= 3 else []
        hover = lambda: self.pick_point(context, event, first, prev)
        if event.type in REFRESH_KEYS or self.snap_key(context, event):
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
        elif event.value == 'PRESS' and (event.type in {'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'}
                                         or self.numpad_keys(event)):
            self.degree = str(DEGREE_PAD.get(event.type) or ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'].index(event.type) + 1)
            self._preview(context, hover())
            self._status(context)
        elif event.value == 'PRESS' and event.type == 'K':
            v = context.window_manager.nproto_view
            v.show_comb = not v.show_comb             # the same switch as Analyze > Curvature Comb
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
    """Draw straight lines and, in a polyline, tangent arcs (R): click the start, then click the end or type a
    length (Tab: angle) and press Enter. The end snaps Perpendicular and
    Tangent to curves and edges. X / Y / Z lock a world axis (Shift+X / Y /
    Z: the plane of the other two), smart guides align it, A / Shift+A
    parallel / square to the line or curve under the mouse, Shift snaps to 15
    degrees, Ctrl to the grid. S: a start snapped onto an edge or curve leaves
    tangent or square to it, and slides along it to meet a second curve
    Tangent / Perpendicular (T picks which edge / face, F the face or the
    edge). Each line is its own 2-CV curve"""
    bl_idname = "nproto.draw_line"
    bl_label = "Line"
    bl_options = {'REGISTER', 'UNDO'}

    start_mode: EnumProperty(name="Start", items=START_MODES, default="FREE", options={'HIDDEN'},
                             description="How a line started on an edge or curve leaves it")
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
        self._arcs, self._arc_mode = [], False      # per segment: its arc's middle, or None (a line)
        self._start_refs = None
        self._seg_start = None          # where the line starts while the start slides (Tangent to both, ...)
        self._curves, self._solved = {}, {}
        self._reset_typing()
        self.sketch_start(context)
        context.window_manager.modal_handler_add(self)
        self._preview(context, self._target(context, event))       # the dot shows right away
        return {'RUNNING_MODAL'}

    # -- typed input ------------------------------------------------------
    def typing_ready(self):
        return getattr(self, "_start", None) is not None

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
        self._seg_start = None
        if self._start is not None:
            arcing = self._arc_from() is not None
            self._angle_lock = None if arcing else A      # an arc's angle is its own (from its start direction)
            if A is not None and not arcing:
                self._lock = None
        P, snap = self.pick_point(context, event, self._extra(), self._start)
        if snap is not None and snap[0] == "cv" and snap[2] is None and P is not None:
            # a CV that lies on its curve (a circle's quarter points): that curve is its reference
            on = self._snaps.through(P)
            if on:
                src, cand, n = self._choose_ref(P, on[0][0], "cv")
                self._hover_cand = cand
                snap = (snap[0], snap[1], src, snap[3])
        if self._start is not None and L is None and A is None and not getattr(event, "alt", False) \
                and self._arc_from() is None:
            sp = self._special(context, event, P, snap)
            if sp is not None:
                return sp
        arc = self._arc_from()
        if arc is not None and (L is not None or A is not None):
            return self._arc_end(arc, P, L, A), None
        if self._start is None or L is None:
            return P, snap
        O, n, u, v = self._plane
        if A is not None:
            d = math.cos(math.radians(A)) * u + math.sin(math.radians(A)) * v
        elif getattr(self, "_lock", None) in ('X', 'Y', 'Z', 'T', 'A'):
            d = {'T': np.asarray(getattr(self, "_tvec", u), float),
                 'A': np.asarray(getattr(self, "_avec", u), float)}.get(self._lock, AXES.get(self._lock))
            d = d / (np.linalg.norm(d) or 1.0)
            if P is not None and float((P - self._start) @ d) < 0:
                d = -d
        elif P is not None and np.linalg.norm(P - self._start) > 1e-12:
            d = (P - self._start) / np.linalg.norm(P - self._start)
        else:
            d = u
        return self._start + L * d, None

    def _arc_end(self, arc, P, R, A):
        """The arc's end from a typed Radius and / or Angle. The angle is the
        turn from the direction the arc leaves to its chord: 90 gives a half
        circle (it sweeps twice the angle); the mouse picks the side."""
        S, t = arc
        n = self._plane[1]
        m = _unit(np.cross(n, t))
        ref = np.asarray(P, float) if P is not None else S + m
        if float((ref - S) @ m) < 0:
            m = -m
        if A is not None:
            phi = math.radians(min(max(abs(float(A)), 0.01), 179.99))
            d = math.cos(phi) * t + math.sin(phi) * m
            if R is not None:
                return S + 2.0 * float(R) * math.sin(phi) * d
            return S + max(float((ref - S) @ d), 1e-9) * d
        C = S + float(R) * m                  # Radius only: the end follows the mouse round that circle
        w = ref - C
        w -= float(w @ n) * n
        if np.linalg.norm(w) < 1e-12:
            w = -m
        return C + float(R) * _unit(w)

    def _status(self, context, length=None, angle=None):
        us = context.scene.unit_settings
        fmt = lambda x: bpy.utils.units.to_string(us.system, 'LENGTH', x * (us.scale_length or 1.0), precision=4)
        step = (("click the next point (on the first one closes it)" if self.polyline and self._lines
                 and self._start is not None and np.linalg.norm(self._start - self._lines[-1][1]) <= 1e-9
                 else "place the end") if self._start is not None else "click the start")
        if getattr(self, "_arc_mode", False):
            step = ("click where the arc ends" if self._arc_from() is not None
                    else "arc: draw a line first, the arc continues from it")
        state = [step, f"{len(self._lines)} drawn" if self._lines else ""]
        if self._start is not None:
            L, A = self._live(context)
            lt = (self._buf + "_" if self._field == "LEN" and self._buf else
                  (fmt(L) if L is not None else (fmt(length) if length is not None else "-")))
            at = (self._buf + "_" if self._field == "ANG" and self._buf else
                  (f"{A:g}°" if A is not None else (f"{angle:.1f}°" if angle is not None else "-")))
            mark = lambda f, t: f"[{t}]" if self._field == f else t
            shown = getattr(self, "_arc_shown", None) if self._arc_from() is not None else None
            if self._arc_from() is not None:
                lt = (self._buf + "_" if self._field == "LEN" and self._buf else
                      (fmt(L) if L is not None else (fmt(shown[0]) if shown else "-")))
                at = (self._buf + "_" if self._field == "ANG" and self._buf else
                      (f"{abs(A):g}°" if A is not None else (f"{shown[1]:.1f}°" if shown else "-")))
                sweep = 2.0 * abs(A) if A is not None else (2.0 * shown[1] if shown else None)
                state += [f"Radius {mark('LEN', lt)}", f"Angle {mark('ANG', at)}"
                          + (f" (arc {sweep:.0f}°)" if sweep is not None else "")]
            else:
                state += [f"Length {mark('LEN', lt)}", f"Angle {mark('ANG', at)}"]
        state.append(self.lock_text().strip(" ·"))
        if self.start_mode != "FREE":
            state.append(f"start {START_TEXT[self.start_mode]}")
        lk, ak = getattr(self, "_lock", None), getattr(self, "_akind", None)
        on = lambda c: True if c else None
        ready = self._start is not None
        arcing = self._arc_from() is not None
        rows = [("Type", [hints.key("123", ("a radius" if arcing else "an exact length (3, 2' 6\", 10mm)") if ready else
                                    "an exact length, after the first click",
                                    on=on(ready and self._field == "LEN" and self._buf) if ready else False),
                          hints.key("Tab", "switch to the angle (90: half circle)" if arcing else "switch to the angle",
                                    on=on(self._field == "ANG")),
                          hints.key("Enter", "place it"), hints.key("Backspace", "delete / undo")]),
                ("Direction", self.direction_hints([hints.key("A", "parallel", on=on(lk == "A" and ak == "parallel")),
                                                    hints.key("Shift+A", "square",
                                                              on=on(lk == "A" and ak == "square"))])),
                ("Snap", self.snap_hints()),
                ("Start", [hints.cycle("S", [START_TEXT[m[0]] for m in START_MODES], START_TEXT[self.start_mode])]),
                ("Reference", self.reference_hints(self.line_mode)),
                ("Polyline", [hints.toggle("P", "", self.polyline),
                              hints.cycle("R", ["line", "arc"], "arc" if getattr(self, "_arc_mode", False) else "line",
                                          "segment")])]
        hints.show(context, "Line", state, rows, MOUSE_SKETCH)

    # -- Tangent / Perpendicular start: the edge / curve it was snapped onto ---
    def ref_mode(self):
        return self.line_mode

    def _apply_start(self):
        """Lock the line's direction to the start reference. Tangent: along the
        edge's tangent (continue it), or in the face's tangent plane.
        Perpendicular: square to the edge, in the curve's plane (a planar curve),
        in the face's tangent plane (F), else in the drawing plane."""
        from . import align, store
        if getattr(self, "_lock", None) in ("T", "-T"):
            self._lock = None
        st = self._start_refs
        if self._start is None or self.start_mode == "FREE" or not st or not st["cands"]:
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
        if self.start_mode == "PERP":
            d = None
            if mode == "SURFACE" and surf is not None:
                d = np.cross(np.asarray(surf[0], float), D)
            else:
                c = self._curve((name, ref))
                if c is not None:
                    d = linesolve.side_direction(c, linesolve.nearest_t(c, self._start),
                                                 self._plane[1] if self._plane is not None else None)
            if d is not None and np.linalg.norm(d) > 1e-12:
                self._lock, self._tvec, self._tkind = "T", np.asarray(d, float), "square"
            return
        if mode == "SURFACE" and surf is not None:
            self._lock, self._tvec, self._tkind = "-T", np.asarray(surf[0], float), "tangent"
        elif np.linalg.norm(D) > 1e-12:
            self._lock, self._tvec, self._tkind = "T", np.asarray(D, float), "tangent"

    # -- Perpendicular / Tangent end snaps -----------------------------------
    def _curve(self, key):
        """The curve / edge (object name, ref) for linesolve, cached; None if
        it cannot be evaluated."""
        if getattr(self, "_curves", None) is None:
            self._curves = {}
        if key not in self._curves:
            from . import features, store
            c = None
            obj = bpy.data.objects.get(key[0])
            if obj is not None:
                try:
                    c = linesolve.kernel_curve(features.input_curve(obj, key[1]), store._k()[0])
                except Exception as exc:
                    print("Line: cannot evaluate", key, exc)
            self._curves[key] = c
        return self._curves[key]

    def _start_curve(self):
        """(key, curve, t, kind) of a Tangent / Perpendicular start on a curve, or None."""
        st = self._start_refs
        if self.start_mode == "FREE" or not st or not st["cands"] or self._start is None:
            return None
        src = st["cands"][st["i"]]["src"]
        key = (src[0], src[1])
        c = self._curve(key)
        if c is None:
            return None
        return key, c, linesolve.nearest_t(c, self._start), "tan" if self.start_mode == "TANGENT" else "perp"

    def _solutions(self, key, c):
        """[(kind label, start point, end point)] on curve `key` for the current
        start: from the fixed start point, or -- a Tangent / Perpendicular start
        on a curve -- sliding along that curve. Cached per start."""
        sc = self._start_curve()
        ck = (key, tuple(np.round(self._start, 12)), sc[0] if sc else None, sc[3] if sc else None,
              self.line_mode, self._start_refs["i"] if sc else 0)
        if getattr(self, "_solved", None) is None:
            self._solved = {}
        if ck in self._solved:
            return self._solved[ck]
        out = []
        if sc is None:
            for kind in ("perp", "tan"):
                for _, Q in linesolve.from_point(c, self._start, kind):
                    out.append((snapping.LABELS[kind], np.asarray(self._start, float), Q))
        elif key != sc[0]:
            k1 = sc[3]
            for k2 in ("tan", "perp"):
                for _, _, Q1, Q2 in linesolve.between(sc[1], k1, c, k2, linesolve.seeds_from(sc[1], sc[2], c)):
                    out.append((SLIDE_LABEL[(k1, k2)], Q1, Q2))
        self._solved[ck] = out
        return out

    def _special(self, context, event, P, snap):
        """(end point, snap) when the mouse is near a point where the line
        meets the curve / edge under it Perpendicular or Tangent, else None.
        Not with a lock of your own, Ctrl, Shift or a lock onto an edge; a
        point snap (end, middle, centre, CV) nearer the mouse still wins."""
        if getattr(self, "_lock", None) not in (None, "T", "-T") or event.ctrl or event.shift \
                or getattr(self, "_edge_lock", None) or getattr(self, "_angle_lock", None) is not None:
            return None
        view, mouse = self._view(context), self._mouse(context, event)
        best = None
        for k, _ in self._snaps.near_lines(view, mouse, SPECIAL_PX):
            W, src = self._snaps.line_info(k)
            c = self._curve((src[0], src[1]))
            if c is None:
                continue
            for label, A, Q in self._solutions((src[0], src[1]), c):
                q = self._to_px(context, Q)
                if q is None:
                    continue
                d = float(np.linalg.norm(q - mouse))
                if d <= SPECIAL_PX and (best is None or d < best[0]):
                    best = (d, label, A, Q, W, src)
        if best is None:
            return None
        if snap is not None and P is not None and snap[0] not in ("edge", "curve"):
            q = self._to_px(context, P)
            if q is not None and float(np.linalg.norm(q - mouse)) <= best[0]:
                return None
        d, label, A, Q, W, src = best
        if np.linalg.norm(A - self._start) > 1e-12:
            self._seg_start = A
        self._hover_cand = {"src": tuple(src[:2]) + ("NEAREST", None), "edge": W, "face": None}
        return Q, ("tan" if label.startswith("Tan") else "perp", None, tuple(src[:2]) + ("NEAREST", None), label)

    # -- A / Shift+A: parallel / square to a line or curve ---------------------
    def _direction_key(self, context, event):
        """Lock the direction parallel (A) or square (Shift+A) to the line /
        curve under the mouse, at the point nearest it; again: unlock."""
        kind = "square" if event.shift else "parallel"
        if getattr(self, "_lock", None) == "A" and getattr(self, "_akind", None) == kind:
            self._lock = None
            return
        self._target(context, event)
        cand = getattr(self, "_hover_cand", None)
        key = tuple(cand["src"][:2]) if cand else None
        W = cand["edge"] if cand else None
        if key is None:
            near = self._snaps.near_lines(self._view(context), self._mouse(context, event), SNAP_PX, limit=1)
            if not near:
                self.report({'INFO'}, "A: hover a line or curve to draw parallel / square to it")
                return
            W, src = self._snaps.line_info(near[0][0])
            key = (src[0], src[1])
        c = self._curve(key)
        if c is None:
            return
        P = self._ray_point(context, Vector(self._mouse(context, event)),
                            self._plane or self._view_plane(context, np.zeros(3)))
        t = linesolve.nearest_t(c, P if P is not None else c.W[0]) if not c.straight else c.t0
        if kind == "parallel":
            d = c.at(t)[2]
        else:
            plane = self._plane or self._view_plane(context, np.zeros(3))
            d = linesolve.side_direction(c, t, plane[1])
        if d is None or np.linalg.norm(d) <= 1e-12:
            return
        self._lock, self._avec, self._akind = "A", np.asarray(d, float) / np.linalg.norm(d), kind
        self._aref = [Vector(p) for p in W] if W is not None else None
        self._angle_lock = None
        self._edge_lock = None

    def _new_start_refs(self, snap):
        self._start_refs = dict(self._refs, cands=list(self._refs["cands"])) \
            if snap and snap[2] is not None and getattr(self, "_refs", None) else None

    def _arc_list(self):
        """Each drawn segment's arc middle (None: a line), as long as _lines."""
        arcs = getattr(self, "_arcs", None)
        if arcs is None or len(arcs) != len(self._lines):
            arcs = self._arcs = (list(arcs or []) + [None] * len(self._lines))[:len(self._lines)]
        return arcs

    def _arc_from(self):
        """(start, unit direction) the next arc leaves with: the chain's last
        point and its last segment's end direction; None out of arc mode, or
        with no segment to continue."""
        if not getattr(self, "_arc_mode", False) or self._start is None or not self._lines \
                or getattr(self, "_seg_start", None) is not None:
            return None
        a, b = self._lines[-1]
        if np.linalg.norm(np.asarray(b, float) - self._start) > 1e-9:
            return None
        return np.asarray(self._start, float), seg_end_tangent(a, b, self._arc_list()[-1])

    def _extra(self):
        out = []
        for (a, b), m in zip(self._lines, self._arc_list()):
            out += [(a, None, "end"), (b, None, "end")]
            if m is not None:
                out.append((np.asarray(m, float), None, "mid"))     # an arc's middle
        return out

    def guide_points(self):
        pts = [p for ab in self._lines for p in ab]
        return pts + ([self._start] if self._start is not None else [])

    def guide_axes(self):
        out = []
        for (a, b), m in reversed(list(zip(self._lines, self._arc_list()))):
            if m is None and np.linalg.norm(np.asarray(b) - np.asarray(a)) > 1e-12:
                out.append(_unit(np.asarray(b, float) - np.asarray(a, float)))
        return out

    def guide_dirs(self, prev):
        """From the last point: straight on (Extension after a line, Tangent
        after an arc) and square to it; parallel to the straight lines drawn.
        None while an arc is being placed (its end only aligns)."""
        if self._arc_from() is not None:
            return []
        out = []
        if self._lines and np.linalg.norm(np.asarray(self._lines[-1][1], float) - np.asarray(prev, float)) <= 1e-9:
            m = self._arc_list()[-1]
            t = seg_end_tangent(*self._lines[-1], m)
            out.append((t, "extend" if m is None else "tangent"))
            if self._plane is not None:
                out.append((_unit(np.cross(self._plane[1], t)), "perp"))
        for (a, b), m in zip(self._lines, self._arc_list()):
            if m is None and np.linalg.norm(np.asarray(b) - np.asarray(a)) > 1e-12:
                d = _unit(np.asarray(b, float) - np.asarray(a, float))
                if all(abs(float(d @ e)) < 1 - 1e-6 for e, _ in out):
                    out.append((d, "parallel"))
        return out

    def plane_open(self):
        """Every new line (not one chained on) may start on another flat face."""
        return getattr(self, "_start", None) is None

    def drawn_segments(self):
        out = []
        for (a, b), m in zip(self._lines, self._arc_list()):
            if m is None:
                out.append((a, b))
                continue
            C, r, x, y, th = arc_circle(a, m, b)
            onto = lambda Q, C=C, r=r, x=x, y=y: C + r * _unit(((Q - C) @ x) * x + ((Q - C) @ y) * y)
            pts = arc_points(a, m, b, 32)
            out += [(pts[k], pts[k + 1], False, onto) for k in range(len(pts) - 1)]
        return out

    def _preview(self, context, hover=None):
        ov = draw.sketch_overlay
        seg = lambda a, b, m: [Vector(a), Vector(b)] if m is None else [Vector(p) for p in arc_points(a, m, b)]
        lines = [seg(a, b, m) for (a, b), m in zip(self._lines, self._arc_list())]
        length = angle = None
        a = self._seg_start if getattr(self, "_seg_start", None) is not None else self._start
        if self._start is not None and hover and hover[0] is not None:
            arc = self._arc_from()
            m = tangent_arc(arc[0], arc[1], hover[0]) if arc is not None else None
            lines.append(seg(a, hover[0], m))
            if m is not None:
                _, rr, _, _, th = arc_circle(a, m, hover[0])
                self._arc_shown = (rr, math.degrees(th) / 2.0)
            else:
                self._arc_shown = None
            d = hover[0] - a
            length = float(np.linalg.norm(d))
            if self._plane is not None:
                O, n, u, v = self._plane
                angle = plane_angle(d, u, v)
        ov["curves"] = lines
        ov["cvs"] = [Vector(p) for ab in self._lines for p in ab] + ([Vector(a)] if a is not None else [])
        ov["snap"] = Vector(hover[0]) if hover and hover[1] is not None else None
        ov["snap_label"] = hover[1][3] if hover and hover[1] is not None else None
        ov["cursor"] = Vector(hover[0]) if hover and hover[0] is not None else None
        self.axis_guides(context, self._start)
        st = getattr(self, "_start_refs", None)
        extra = [st["cands"][st["i"]]] if self._start is not None and st and st["cands"] \
            and self.start_mode != "FREE" else []
        if getattr(self, "_lock", None) == "A" and getattr(self, "_aref", None):
            extra.append({"edge": self._aref, "face": None})
        self.show_refs(extra)
        self._status(context, length, angle)
        context.area.tag_redraw()

    def _commit(self, P, chain):
        a = self._seg_start if getattr(self, "_seg_start", None) is not None else self._start
        if np.linalg.norm(P - a) <= 1e-9:
            return
        arc = self._arc_from()
        mid = tangent_arc(arc[0], arc[1], P) if arc is not None else None
        self._arc_list().append(mid)
        self._lines.append((np.asarray(a, float), P))
        self._seg_start = None
        # typed lines, and every line with Polyline on, chain on from their end
        self._start = next_start(self._lines, P, chain)
        self._reset_typing()
        self._lock = None
        self._edge_lock = None
        self._start_refs = None

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        if not hasattr(self, "_buf"):
            self._reset_typing()
        typing = self._start is not None
        ch = self.typed_char(event)
        if typing and ch and (ch in "0123456789.-+*/'\"" or (self._buf and (ch.isalpha() or ch == " "))) \
                and not (event.ctrl or event.alt or event.oskey):
            self._buf += ch
            self._preview(context, self._target(context, event))
            return {'RUNNING_MODAL'}
        if event.type in REFRESH_KEYS or self.snap_key(context, event):
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
                self._apply_start()
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
                self._arc_list().pop()
                self._start = np.asarray(self._lines.pop()[0], float)    # chaining: step back one segment
                self._lock = None
            elif self._start is not None:
                self._start = None
                self._lock = None
                self._start_refs = None
            elif self._lines:
                self._arc_list().pop()
                self._lines.pop()
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type in {'X', 'Y', 'Z'} and not self._buf:
            self.axis_key(event)
            self._ang = None
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'S' and not self._buf:
            order = [m[0] for m in START_MODES]
            self.start_mode = order[(order.index(self.start_mode) + 1) % len(order)]
            self._apply_start()
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'R' and not self._buf \
                and not (event.ctrl or getattr(event, "alt", False) or getattr(event, "oskey", False)):
            # the next segments are tangent arcs (or lines again); an arc needs a segment to continue
            self._arc_mode = not getattr(self, "_arc_mode", False)
            if self._arc_mode and not self.polyline:
                self.polyline = True
                if self._start is None and self._lines:
                    self._start = np.asarray(self._lines[-1][1], float)
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'A' and not self._buf:
            self._direction_key(context, event)
            self._ang = None
            self._preview(context, self._target(context, event))
        elif event.value == 'PRESS' and event.type == 'T' and not self._buf:
            if self._start is not None and self._start_refs:
                self.cycle_ref(self._start_refs)
                self._apply_start()
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
                self._apply_start()
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
                                    "arcs": [None if m is None else list(map(float, m)) for m in self._arc_list()],
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
        arcs = (d.get("arcs") or []) + [None] * len(lines)
        segs = [(np.array(a, float), np.array(b, float), None if m is None else np.array(m, float))
                for (a, b), m in zip(lines, arcs)]
        groups = chain_segments(segs) if self.polyline else [[sg] for sg in segs]
        made = []
        for g in groups:
            if all(m is None for _, _, m in g):
                pl = [g[0][0]] + [b for _, b, _ in g]
                closed = bool(len(pl) > 3 and np.linalg.norm(pl[0] - pl[-1]) < 1e-9)
                pts = pl[:-1] if closed else pl
                sp = spec([tuple(p) for p in pts], order=2, cyclic=closed, endpoint=not closed)
            else:
                sp = segments_spec(g)           # lines and tangent arcs: one exact curve
            made.append(add_curve(context, "NURBS Line", [sp], edit))
        if edit is None:
            _select_new(context, made)
        n_arcs = sum(1 for _, _, m in segs if m is not None)
        self.report({'INFO'}, f"{len(lines) - n_arcs} line(s)" + (f", {n_arcs} arc(s)" if n_arcs else "")
                    + (f" as {len(groups)} polyline(s)" if self.polyline else ""))
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

    def typing_ready(self):
        return getattr(self, "_start", None) is not None

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
        state = ["click the opposite corner" if self._start is not None else
                 "click the centre" if self.from_center else "click a corner"]
        if self._start is not None:
            X, Y = self._live(context)
            xs = (self._buf + "_" if self._field == "X" and self._buf else
                  fmt(X) if X is not None else (fmt(rect[1]) if rect else "-"))
            ys = (self._buf + "_" if self._field == "Y" and self._buf else
                  fmt(Y) if Y is not None else (fmt(rect[2]) if rect else "-"))
            mark = lambda f, t: f"[{t}]" if self._field == f else t
            state += [f"X {mark('X', xs)}", f"Y {mark('Y', ys)}"]
        state.append(f"degree {self.degree_u} x {self.degree_v}, {self.spans_u} x {self.spans_v} span(s)")
        lk = getattr(self, "_lock", None)
        ready = self._start is not None
        rows = [("Type", [hints.key("123", "an exact X size" if ready else "an exact X size, after the first click",
                                    on=(True if self._field == "X" and self._buf else None) if ready else False),
                          hints.key("Tab", "switch to the Y size", on=True if self._field == "Y" else None),
                          hints.key("Enter", "place it"), hints.key("Backspace", "delete / undo")]),
                ("Shape", [hints.toggle("C", "from centre", self.from_center), hints.key("Shift", "square")]),
                ("Direction", [hints.key("X Y Z", "axis (first click)",
                                         on=True if lk and lk[-1] in "XYZ" else None)])]
        rows.append(("Snap", self.snap_hints()))
        hints.show(context, "Plane", state, rows, MOUSE_SHAPE)

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
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        typing = self._start is not None
        ch = self.typed_char(event)
        if typing and ch and (ch in "0123456789.-+*/'\"" or (self._buf and (ch.isalpha() or ch == " "))) \
                and not (event.ctrl or event.alt or event.oskey):
            self._buf += ch
            self._preview(context, self.pick_point(context, event), event)
            return {'RUNNING_MODAL'}
        hover = lambda: self.pick_point(context, event)
        if event.type in REFRESH_KEYS or self.snap_key(context, event):
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
    """Place a circle, ellipse, square or rectangle perpendicular to a curve, a
    surface edge or a mesh edge: click on it for the centre, then drag (or
    type) the size. On a surface or mesh edge it stands square to the surface.
    With Construction History it stays perpendicular to a curve or surface edge
    when that changes (a mesh edge is a one-off placement)"""
    bl_idname = "nproto.draw_profile"
    bl_label = "Profile"
    bl_options = {'REGISTER', 'UNDO'}
    use_guides = False      # a profile needs a real edge to stand on
    face_planes = False     # it stands on the edge, square to it

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
        if '"mesh": true' not in (self.data or ""):     # a mesh edge has no position along it to move
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

    def typing_ready(self):
        return getattr(self, "_center", None) is not None

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

    def _mesh_frame(self, snap, P):
        """The profile's frame on a mesh edge or edge centre (its snap carries
        the edge's direction and its face's normal), or None."""
        from .primitives import profile_axes
        fr = snap[1] if snap is not None and snap[2] is None and snap[0] in ("edge", "mid") else None
        if not isinstance(fr, np.ndarray) or fr.shape != (2, 3):
            return None
        return profile_axes(P, fr[0], fr[1] if np.linalg.norm(fr[1]) > 0.5 else None, self.rotation)

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
        state = ["drag or type the size" if self._center is not None else "click on a curve, surface edge or mesh edge"]
        if self._center is not None and sizes:
            state += [f"{k.replace('width', 'X').replace('depth', 'Y').title()} {fmt(v)}"
                      for k, v in sizes.items() if k in ("radius", "radius2", "width", "depth")]
            if self._buf:
                state.append(f"[{self._buf}_]")
        rows = [("Shape", [hints.cycle("C E S R", [p[1].lower() for p in PROFILE_SHAPES], self.shape.lower())]),
                ("Type", [hints.key("123", "an exact size" if self._center is not None else
                                    "an exact size, after the first click",
                                    on=None if self._center is not None else False),
                          hints.key("Tab", "switch to the second size"),
                          hints.key("Enter", "place it"), hints.key("Ctrl", "grid")]),
                ("Reference", [hints.key("T", "next edge / face"),
                               hints.key("L", "lock onto the edge",
                                         on=True if getattr(self, "_edge_lock", None) else None)])]
        hints.show(context, f"Profile ({self.shape.lower()})", state, rows, MOUSE_SHAPE)

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
                M = self._mesh_frame(snap, P)
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
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        placed = self._center is not None
        ch = self.typed_char(event)
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
                Mm = self._mesh_frame(snap, P) if snap is not None and P is not None else None
                if snap is None or (snap[2] is None and Mm is None):
                    self.report({'WARNING'}, "Profile: click on a curve, a surface edge or a mesh edge")
                    return {'RUNNING_MODAL'}
                if Mm is not None:          # a mesh edge: placed there, no history
                    self.at, self._frame = 0.0, Mm
                    self._mesh = (np.asarray(P, float), np.asarray(snap[1][0], float), np.asarray(snap[1][1], float))
                else:
                    try:
                        self.at, self._frame = self._frame_at(snap[2], P)
                    except Exception as exc:
                        self.report({'WARNING'}, f"Profile: {exc}")
                        return {'RUNNING_MODAL'}
                    self._mesh = None
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
        if self._src is None and getattr(self, "_mesh", None) is not None:
            P, T, up = self._mesh
            self.data = json.dumps({"mesh": True, "P": P.tolist(), "T": T.tolist(), "up": up.tolist()})
        else:
            name, ref, snap, face = (list(self._src) + [None])[:4]
            self.data = json.dumps({"obj": name, "ref": ref, "face": face})
        self.sketch_end(context)
        return self.execute(context)

    def execute(self, context):
        from mathutils import Matrix
        from . import features
        from .primitives import profile_axes, profile_frame, profile_spec, write_spline_object
        try:
            d = json.loads(self.data or "{}")
        except ValueError:
            d = {}
        on_mesh = bool(d.get("mesh"))
        path = bpy.data.objects.get(d.get("obj", "")) if d.get("obj") else None
        if path is None and not on_mesh:
            self.report({'WARNING'}, "Profile is interactive: start it from the panel or menu")
            return {'CANCELLED'}
        params = self._params(face=d.get("face"))
        try:
            if on_mesh:                 # on a mesh edge: square to its face, turned by Rotation (F9)
                up = np.asarray(d["up"], float)
                M = profile_axes(d["P"], d["T"], up if np.linalg.norm(up) > 0.5 else None, self.rotation)
            else:
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
        if features.history_enabled(context) and not on_mesh:
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
        self.report({'INFO'}, f"{name} on a mesh edge" if on_mesh else f"{name} on {path.name} {d['ref']} at {self.at:.3f}{hist}")
        return {'FINISHED'}


classes = (NPROTO_OT_draw_curve, NPROTO_OT_draw_line, NPROTO_OT_draw_plane, NPROTO_OT_draw_profile)
