"""NURBS primitives (curves and surfaces), registered as history features.

Every primitive is built in its object's LOCAL space (the object sits at the
3D cursor), so its parameters stay editable in the Feature panel later.

Degree 0 = "exact": rational quadratic circles / arcs / surfaces of
revolution. Degree 1..5 = non-rational B-splines with the requested number
of spans; closed directions are built periodic (smooth across the seam) and
then clamped, exactly like Blender's own cyclic NURBS.
"""

import json
import math

import numpy as np
from mathutils import Vector

from . import features, geom
from .kernel import load as _load_kernel

W45 = math.sqrt(0.5)


def _k():
    sm = _load_kernel()
    return sm, sm.blender_ext


# --------------------------------------------------------------------------
# Spline specs (Blender NURBS splines) for curve primitives
# --------------------------------------------------------------------------

def spec(points, weights=None, order=4, cyclic=False, endpoint=True, bezier=False):
    return {"points": [tuple(map(float, p)) for p in points],
            "weights": [float(w) for w in (weights or [1.0] * len(points))],
            "order": int(order), "cyclic": cyclic, "endpoint": endpoint, "bezier": bezier}


def exact_circle_spec(r1, r2=None):
    """Blender's exact NURBS circle (order 3, cyclic, endpoint+bezier knots)."""
    r2 = r1 if r2 is None else r2
    pts = [(0, -r2, 0), (-r1, -r2, 0), (-r1, 0, 0), (-r1, r2, 0), (0, r2, 0), (r1, r2, 0), (r1, 0, 0), (r1, -r2, 0)]
    w = [1, W45, 1, W45, 1, W45, 1, W45]
    return spec(pts, w, order=3, cyclic=True, endpoint=True, bezier=True)


def exact_arc_spec(r, angle_deg):
    """Rational quadratic arc in Bezier knot mode (exact), from +X counter-clockwise."""
    a = math.radians(max(1e-3, min(angle_deg, 360.0)))
    segs = max(1, int(math.ceil(a / (math.pi / 2) - 1e-9)))
    d = a / segs
    pts, w = [], []
    for s in range(segs):
        t0, t1 = s * d, (s + 1) * d
        p0 = (r * math.cos(t0), r * math.sin(t0), 0.0)
        m = (t0 + t1) / 2
        rr = r / math.cos(d / 2)
        p1 = (rr * math.cos(m), rr * math.sin(m), 0.0)
        if s == 0:
            pts.append(p0); w.append(1.0)
        pts.append(p1); w.append(math.cos(d / 2))
        pts.append((r * math.cos(t1), r * math.sin(t1), 0.0)); w.append(1.0)
    return spec(pts, w, order=3, cyclic=False, endpoint=True, bezier=True)


def periodic_ring(n_cvs, degree, r1, r2=None):
    """CVs of a uniform periodic B-spline that approximates a circle / ellipse
    (radius scaled so the curve's mean radius is exact)."""
    r2 = r1 if r2 is None else r2
    p = min(degree, n_cvs - 1)
    ang = [2 * math.pi * j / n_cvs for j in range(n_cvs)]
    unit = [(math.cos(t), math.sin(t), 0.0) for t in ang]
    deg, P, W, U = geom.blender_nurbs_to_clamped(unit, [1.0] * n_cvs, p + 1, True, False, False)
    rad = [np.linalg.norm(geom.eval_bspline(P, W, U, deg, k / 720)[:2]) for k in range(721)]
    s = 1.0 / float(np.mean(rad))
    return [(r1 * s * x, r2 * s * y, 0.0) for x, y, _ in unit], p


def blender_spline_to_clamped(sp):
    """Clamped (degree, points, weights, knots) of a spline spec."""
    deg, P, W, U = geom.blender_nurbs_to_clamped(sp["points"], sp["weights"], sp["order"], sp["cyclic"],
                                                 sp["endpoint"], sp["bezier"])
    return deg, [tuple(map(float, p)) for p in P], [float(w) for w in W], [float(u) for u in U]


def spec_to_kernel_curve(sp):
    _, ext = _k()
    deg, P, W, U = blender_spline_to_clamped(sp)
    rational = any(abs(w - 1.0) > 1e-12 for w in W)
    return ext.curve_from_cvs(deg, P, U, W if rational else None)


# --------------------------------------------------------------------------
# Curve primitives
# --------------------------------------------------------------------------

def _line_points(a, b, n):
    return [tuple(a[i] + (b[i] - a[i]) * k / (n - 1) for i in range(3)) for k in range(n)]


@features.builder("prim_line")
def prim_line(inputs, p):
    deg = max(1, int(p.get("degree", 1)))
    spans = max(1, int(p.get("spans_u", 1)))
    L = p.get("length", 2.0)
    pts = _line_points((-L / 2, 0, 0), (L / 2, 0, 0), deg + spans)
    return ("spline", spec(pts, order=deg + 1))


@features.builder("prim_circle")
def prim_circle(inputs, p):
    r1 = p.get("radius", 1.0)
    r2 = p.get("radius2", r1) if p.get("elliptic") else r1
    deg = int(p.get("degree", 0))
    if deg == 0:
        return ("spline", exact_circle_spec(r1, r2))
    pts, pp = periodic_ring(max(deg + 1, int(p.get("spans_u", 8))), deg, r1, r2)
    return ("spline", spec(pts, order=pp + 1, cyclic=True, endpoint=False))


@features.builder("prim_arc")
def prim_arc(inputs, p):
    r = p.get("radius", 1.0)
    ang = p.get("angle", 90.0)
    deg = int(p.get("degree", 0))
    if deg == 0:
        return ("spline", exact_arc_spec(r, ang))
    n = max(deg + 1, int(p.get("spans_u", 4)) + deg)
    a = math.radians(ang)
    samples = [(r * math.cos(a * k / 127), r * math.sin(a * k / 127), 0.0) for k in range(128)]
    return ("spline", _fit_open_spec(samples, deg, n))


def _fit_open_spec(samples, deg, n_cvs):
    """Least-squares open clamped fit with UNIFORM knots (Blender endpoint mode)."""
    S = np.array(samples)
    d = np.linalg.norm(np.diff(S, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(d)]) / d.sum()
    U = geom.clamped_uniform_knots(n_cvs, deg)
    A = geom.interp_matrix(t, U, deg, n_cvs)
    X = np.zeros((n_cvs, 3))
    X[0], X[-1] = S[0], S[-1]
    R = S - np.outer(A[:, 0], X[0]) - np.outer(A[:, -1], X[-1])
    X[1:-1] = np.linalg.lstsq(A[:, 1:-1], R, rcond=None)[0]
    return spec([tuple(x) for x in X], order=deg + 1)


@features.builder("prim_rectangle")
def prim_rectangle(inputs, p):
    w, h = p.get("width", 2.0) / 2, p.get("depth", 1.0) / 2
    return ("spline", spec([(-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)], order=2, cyclic=True, endpoint=False))


@features.builder("prim_polygon")
def prim_polygon(inputs, p):
    n = max(3, int(p.get("sides", 6)))
    r = p.get("radius", 1.0)
    pts = [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), 0.0) for k in range(n)]
    return ("spline", spec(pts, order=2, cyclic=True, endpoint=False))


@features.builder("prim_helix")
def prim_helix(inputs, p):
    r, turns, height = p.get("radius", 1.0), max(0.1, p.get("turns", 3.0)), p.get("height", 3.0)
    deg = max(2, int(p.get("degree", 3)) or 3)
    samples = [(r * math.cos(2 * math.pi * turns * k / 400), r * math.sin(2 * math.pi * turns * k / 400),
                height * k / 400) for k in range(401)]
    n = max(deg + 1, int(round(turns * max(4, int(p.get("spans_u", 8))))) + deg)
    return ("spline", _fit_open_spec(samples, deg, n))


# --------------------------------------------------------------------------
# Profiles perpendicular to a curve / edge (Sweep and Pipe sections)
# --------------------------------------------------------------------------

PROFILE_SHAPES = [("CIRCLE", "Circle", ""), ("ELLIPSE", "Ellipse", ""), ("SQUARE", "Square", ""),
                  ("RECTANGLE", "Rectangle", "")]


def profile_frame(path_obj, ref, at, rotation=0.0, face=0):
    """World matrix of a profile perpendicular to a curve / surface edge at
    `at` (0..1 along its parameter): local Z along the tangent, local Y towards
    the surface normal (a surface edge; `face` of its faces) or world up (a
    curve), then turned `rotation` degrees about the tangent."""
    from . import store
    sm, _ = _k()
    kc = features.input_curve(path_obj, ref)
    t0, t1 = kc.parameter_range()
    d = sm.evaluate_curve(kc, t0 + min(max(float(at), 0.0), 1.0) * (t1 - t0))
    P = np.array(d["point"], float)
    T = np.array(d["tangent"], float)
    T /= np.linalg.norm(T) or 1.0
    up = None
    if features.is_surface_edge(ref) and store.is_nurbs(path_obj):
        try:
            from .align import surface_frame
            up = np.asarray(surface_frame(path_obj, ref, P, face)[0], float)
        except Exception:
            up = None
    if up is None or np.linalg.norm(up - (up @ T) * T) < 1e-6:
        up = np.array([0.0, 0.0, 1.0]) if abs(T[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    Y = up - (up @ T) * T
    Y /= np.linalg.norm(Y)
    X = np.cross(Y, T)
    a = math.radians(float(rotation))
    X, Y = math.cos(a) * X + math.sin(a) * Y, -math.sin(a) * X + math.cos(a) * Y
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = X, Y, T, P
    return M


def curve_param_at(path_obj, ref, P):
    """0..1 position along the curve / edge of its point nearest P."""
    sm, _ = _k()
    kc = features.input_curve(path_obj, ref)
    t0, t1 = kc.parameter_range()
    t = float(sm.curve_closest_point(kc, tuple(float(x) for x in P))[1])
    return min(max((t - t0) / ((t1 - t0) or 1.0), 0.0), 1.0)


def profile_spec(p):
    """Local spline spec of a profile (in its own XY plane, centred)."""
    shape = p.get("pp_shape", "CIRCLE")
    if shape in ("CIRCLE", "ELLIPSE"):
        return prim_circle([], dict(p, elliptic=shape == "ELLIPSE"))[1]
    w = float(p.get("width", 1.0))
    return prim_rectangle([], {"width": w, "depth": w if shape == "SQUARE" else float(p.get("depth", 1.0))})[1]


@features.builder("perp_profile")
def perp_profile(inputs, p):
    """A circle / ellipse / square / rectangle perpendicular to its path (input
    0) at pp_at: it follows when the path changes."""
    from mathutils import Matrix
    path, ref = inputs[0]
    M = profile_frame(path, ref, p.get("pp_at", 0.0), p.get("pp_rotation", 0.0), p.get("pp_face", 0))
    sp = profile_spec(p)

    def write(o):
        o.matrix_world = Matrix(M.tolist())
        write_spline_object(o, sp)
    return ("inplace", write)


# --------------------------------------------------------------------------
# Surface primitives (kernel geometry, local space)
# --------------------------------------------------------------------------

def _clamped(n, d):
    return geom.clamped_uniform_knots(n, d)


@features.builder("prim_plane")
def prim_plane(inputs, p):
    _, ext = _k()
    old = int(p.get("degree", 3)) or 3           # planes made before Degree U / V: one degree
    du, dv = max(1, int(p.get("pl_degree_u", old))), max(1, int(p.get("pl_degree_v", old)))
    nu, nv = max(1, int(p.get("spans_u", 1))) + du, max(1, int(p.get("spans_v", 1))) + dv
    w, h = p.get("width", 2.0), p.get("depth", 2.0)
    pts = [(-w / 2 + w * i / (nu - 1), -h / 2 + h * j / (nv - 1), 0.0) for i in range(nu) for j in range(nv)]
    return ext.surface_from_cvs(du, dv, pts, _clamped(nu, du), _clamped(nv, dv))


def _ring_clamped(deg, spans, r1=1.0, r2=None):
    """Clamped (degree, unit-ish ring CVs, weights, knots) for a closed direction."""
    if deg == 0:
        return blender_spline_to_clamped(exact_circle_spec(r1, r2))
    pts, pp = periodic_ring(max(deg + 1, spans), deg, r1, r2)
    return blender_spline_to_clamped(spec(pts, order=pp + 1, cyclic=True, endpoint=False))


def _revolve_tensor(ring, profile_pts, profile_w, profile_deg, profile_knots):
    """Surface of revolution about Z as a tensor product: ring (u) x profile (v).
    Profile points are (radius, 0, z)."""
    _, ext = _k()
    rdeg, rP, rW, rU = ring
    pts, wts = [], []
    for (cx, cy, _), cw in zip(rP, rW):
        for (px, _, pz), pw in zip(profile_pts, profile_w):
            pts.append((cx * px, cy * px, pz))
            wts.append(cw * pw)
    rational = any(abs(w - 1.0) > 1e-12 for w in wts)
    return ext.surface_from_cvs(rdeg, profile_deg, pts, rU, profile_knots, wts if rational else None)


def _profile_line(a, b, deg, spans):
    deg = max(1, deg)
    n = deg + max(1, spans)
    pts = _line_points(a, b, n)
    return pts, [1.0] * n, deg, _clamped(n, deg)


def _profile_arc(r, a0, a1, deg, spans, center=(0.0, 0.0)):
    """Arc in the XZ plane (x = radius, z = height) from angle a0 to a1 (degrees)."""
    cx, cz = center
    if deg == 0:
        sp = exact_arc_spec(r, a1 - a0)
        d, P, W, U = blender_spline_to_clamped(sp)
        c, s = math.cos(math.radians(a0)), math.sin(math.radians(a0))
        P = [(cx + c * x - s * y, 0.0, cz + s * x + c * y) for x, y, _ in P]
        return P, W, d, U
    samples = [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / 127)), 0.0,
                cz + r * math.sin(math.radians(a0 + (a1 - a0) * k / 127))) for k in range(128)]
    sp = _fit_open_spec(samples, deg, deg + max(1, spans))
    d, P, W, U = blender_spline_to_clamped(sp)
    return P, W, d, U


def _deg(p):
    return int(p.get("degree", 0))


def _caps(surface_brep, circles):
    """Join a revolved side with planar caps into a healed solid."""
    sm, ext = _k()
    parts = [surface_brep] + [sm.create_planar_faces([c]) for c in circles]
    b = sm.merge_breps(parts, operation=sm.BooleanOp.MERGE)
    b = sm.stitch_advanced(b, 1e-6, squeeze_small_edges=False, making_manifold_solid=False)[0]
    try:
        h = sm.heal_brep(b)
        if h is not None and h.face_count():
            b = h
    except Exception:
        pass
    return b


def _cap_circle(r, z, deg, spans):
    sm, ext = _k()
    if deg == 0:
        c = spec_to_kernel_curve(exact_circle_spec(r))
    else:
        pts, pp = periodic_ring(max(deg + 1, spans), deg, r)
        c = spec_to_kernel_curve(spec(pts, order=pp + 1, cyclic=True, endpoint=False))
    return sm.translate(c, (0.0, 0.0, z))


@features.builder("prim_cylinder")
def prim_cylinder(inputs, p):
    _, ext = _k()
    r, h, deg = p.get("radius", 1.0), p.get("height", 2.0), _deg(p)
    su, sv = int(p.get("spans_u", 8)), int(p.get("spans_v", 1))
    ring = _ring_clamped(deg, su)
    prof = _profile_line((r, 0, -h / 2), (r, 0, h / 2), max(1, deg) if deg else 1, sv)
    s = _revolve_tensor(ring, *prof)
    if p.get("caps"):
        return _caps(ext.brep_from_surface(s), [_cap_circle(r, -h / 2, deg, su), _cap_circle(r, h / 2, deg, su)])
    return s


@features.builder("prim_cone")
def prim_cone(inputs, p):
    _, ext = _k()
    r, h, deg = p.get("radius", 1.0), p.get("height", 2.0), _deg(p)
    r2 = max(0.0, p.get("radius2", 0.0))
    su, sv = int(p.get("spans_u", 8)), int(p.get("spans_v", 1))
    ring = _ring_clamped(deg, su)
    prof = _profile_line((r, 0, 0), (r2, 0, h), max(1, deg) if deg else 1, sv)
    s = _revolve_tensor(ring, *prof)
    if p.get("caps"):
        caps = [_cap_circle(r, 0.0, deg, su)] + ([_cap_circle(r2, h, deg, su)] if r2 > 1e-9 else [])
        return _caps(ext.brep_from_surface(s), caps)
    return s


@features.builder("prim_sphere")
def prim_sphere(inputs, p):
    r, deg = p.get("radius", 1.0), _deg(p)
    ring = _ring_clamped(deg, int(p.get("spans_u", 8)))
    prof = _profile_arc(r, -90.0, 90.0, deg, int(p.get("spans_v", 4)))
    return _revolve_tensor(ring, *prof)


@features.builder("prim_torus")
def prim_torus(inputs, p):
    R, r, deg = p.get("radius", 1.0), p.get("radius2", 0.3), _deg(p)
    ring = _ring_clamped(deg, int(p.get("spans_u", 8)))
    if deg == 0:
        sp = exact_circle_spec(r)
    else:
        pts, pp = periodic_ring(max(deg + 1, int(p.get("spans_v", 6))), deg, r)
        sp = spec(pts, order=pp + 1, cyclic=True, endpoint=False)
    d, P, W, U = blender_spline_to_clamped(sp)
    P = [(R + x, 0.0, y) for x, y, _ in P]       # tube circle in the XZ plane, offset by R
    return _revolve_tensor(ring, P, W, d, U)


@features.builder("prim_disk")
def prim_disk(inputs, p):
    sm, _ = _k()
    return sm.create_planar_faces([_cap_circle(p.get("radius", 1.0), 0.0, _deg(p), int(p.get("spans_u", 8)))])


@features.builder("prim_box")
def prim_box(inputs, p):
    sm, ext = _k()
    w, d, h = p.get("width", 2.0) / 2, p.get("depth", 2.0) / 2, p.get("height", 2.0) / 2
    deg = max(1, _deg(p) or 1)
    s = max(1, int(p.get("spans_u", 1)))
    n = deg + s
    knots = _clamped(n, deg)

    def quad(o, eu, ev):
        o, eu, ev = np.array(o), np.array(eu), np.array(ev)
        pts = [tuple(o + eu * i / (n - 1) + ev * j / (n - 1)) for i in range(n) for j in range(n)]
        return ext.brep_from_surface(ext.surface_from_cvs(deg, deg, pts, knots, knots))

    faces = [quad((-w, -d, -h), (2 * w, 0, 0), (0, 2 * d, 0)), quad((-w, -d, h), (2 * w, 0, 0), (0, 2 * d, 0)),
             quad((-w, -d, -h), (2 * w, 0, 0), (0, 0, 2 * h)), quad((-w, d, -h), (2 * w, 0, 0), (0, 0, 2 * h)),
             quad((-w, -d, -h), (0, 2 * d, 0), (0, 0, 2 * h)), quad((w, -d, -h), (0, 2 * d, 0), (0, 0, 2 * h))]
    b = sm.merge_breps(faces, operation=sm.BooleanOp.MERGE)
    b = sm.stitch_advanced(b, 1e-6, squeeze_small_edges=False, making_manifold_solid=False)[0]
    try:
        hb = sm.heal_brep(b)
        if hb is not None and hb.face_count():
            b = hb
    except Exception:
        pass
    return b


CURVE_KINDS = {"LINE": "prim_line", "ARC": "prim_arc", "CIRCLE": "prim_circle", "ELLIPSE": "prim_circle",
               "RECTANGLE": "prim_rectangle", "POLYGON": "prim_polygon", "HELIX": "prim_helix"}
SURFACE_KINDS = {"PLANE": "prim_plane", "BOX": "prim_box", "CYLINDER": "prim_cylinder", "CONE": "prim_cone",
                 "SPHERE": "prim_sphere", "TORUS": "prim_torus", "DISK": "prim_disk"}


def write_spline_object(obj, sp):
    """Write one spline spec, or a list of them, (local space) into a curve object."""
    cu = obj.data
    cu.dimensions = '3D'
    cu.splines.clear()
    specs = sp if isinstance(sp, list) else [sp]
    for one in specs:
        _add_spline(cu, one)
    # joints of a Join Curves result (Unjoin splits there)
    if any("joints" in one for one in specs):
        obj["nproto_joints"] = json.dumps([one.get("joints") for one in specs])
    elif "nproto_joints" in obj:
        del obj["nproto_joints"]


def _add_spline(cu, sp):
    s = cu.splines.new('NURBS')
    s.points.add(len(sp["points"]) - 1)
    for pt, co, w in zip(s.points, sp["points"], sp["weights"]):
        pt.co = (*co, w)
    s.order_u = sp["order"]
    s.use_cyclic_u = sp["cyclic"]
    s.use_endpoint_u = sp["endpoint"]
    s.use_bezier_u = sp["bezier"]
    s.resolution_u = 24
