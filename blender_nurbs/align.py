"""Align Curve (Alias "Align"): move a drawn curve's end CVs so the end is
G0 / G1 / G2 to an edge, keeping the rest of the curve.

Per end, with the end parameter t = 0 of a clamped curve of degree p, knots U:
    C(0)   = Q0
    C'(0)  = p / u1 * (Q1 - Q0)                         u1 = U[p+1] - U[0]
    C''(0) = (p-1)/u1 * ( p (Q2-Q1)/u2 - p (Q1-Q0)/u1 )   u2 = U[p+2] - U[0]
so G0 sets Q0, G1 sets Q1, G2 sets Q2, and G3 (C''') Q3.

Modes
  EDGE     continue the edge: C'(0) = lam * D (edge's outward derivative),
           C''(0) = lam^2 * DD, C''' = lam^3 DDD -> the curve and the edge
           form one G1/G2/G3 curve.
  SURFACE  stay tangent to the edge's surface: C'(0) = the curve's own end
           tangent projected into the tangent plane; the normal component of
           C''(0) matches the surface's normal curvature in that direction,
           and for G3 that of C''' the surface's third-order form (contact of
           order 3 with the surface).
  NORMAL   perpendicular to the surface: the end continues the surface's
           normal line at that point (a straight line), so C' runs along the
           normal and G2 / G3 leave no curvature (nor its rate) across it.
  SYMMETRY a symmetry (mirror) plane: C' along the normal (G1), and the
           curve joins its mirror image C2 / C3: G2 n.C'' = 0 (the bend,
           C'' in the plane, stays free), G3 also C''' in the plane = 0 (its
           part along the normal stays free). Meant for planes; on a curved
           surface the mirror is its tangent plane there.
Targets: a curve or a surface edge (Snap: END, the nearest end of the edge,
or NEAREST, the nearest point on it), or a face of a surface ("F<i>": the
point of the face nearest the end, Tangent to Surface or Perpendicular).

The curve itself is the feature's first input, so moving its other CVs keeps
the ends locked to the edges (history), like Alias with construction history on.
"""

import json

import numpy as np
from mathutils import Vector

from . import features, geom, store

CONT = {"G0": 0, "G1": 1, "G2": 2, "G3": 3}
MODES = ("EDGE", "SURFACE", "NORMAL", "SYMMETRY")
SQUARE = ("NORMAL", "SYMMETRY")      # modes that leave along the surface normal


def is_face_ref(ref):
    """A face of a NURBS object (F<i>), as a target of its own."""
    return isinstance(ref, str) and len(ref) > 1 and ref[0] == "F" and ref[1:].isdigit()


class Frame(tuple):
    """An end target: (P, D, DD, surf) world, plus d3 (the third derivative,
    or None). Unpacks as the 4-tuple it always was."""
    d3 = None


def frame(P, D, DD, surf=None, d3=None):
    f = Frame((np.asarray(P, float), np.asarray(D, float), np.asarray(DD, float), surf))
    f.d3 = None if d3 is None else np.asarray(d3, float)
    return f


def frame_d3(fr):
    return getattr(fr, "d3", None)


def flipped(fr):
    """The frame turned round (odd derivatives change sign)."""
    d3 = frame_d3(fr)
    return frame(fr[0], -np.asarray(fr[1], float), fr[2], fr[3], None if d3 is None else -d3)


class SurfFrame(tuple):
    """(n, k_n) as surface_frame returned it, plus the exact forms of the
    surface at the point (world): n unit normal, Q the second fundamental form
    (3x3, on tangent vectors), C the third-order form (3x3x3) or None."""
    n = Q = C = None


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


# --------------------------------------------------------------------------
# The curve being aligned (object-local space)
# --------------------------------------------------------------------------

def curve_state(obj, si=0):
    """(kind, p, Q (n,3) local, U) of the object's spline si (the first by
    default; a sketch drawn in Edit Mode has several). NURBS splines are used
    as clamped uniform (endpoint) curves; Bezier splines as their first and
    last cubic segments."""
    sp = obj.data.splines[si]
    if sp.use_cyclic_u:
        raise RuntimeError("closed (cyclic) curves have no ends to align")
    if sp.type == 'BEZIER':
        bp = sp.bezier_points
        Q = []
        for i, b in enumerate(bp):
            if i:
                Q.append(np.array(b.handle_left))
            Q.append(np.array(b.co))
            if i < len(bp) - 1:
                Q.append(np.array(b.handle_right))
        return "BEZIER", 3, np.array(Q, float), None
    pts = np.array([tuple(p.co) for p in sp.points], float)
    if sp.type == 'POLY':
        return "NURBS", 1, pts[:, :3], geom.clamped_uniform_knots(len(pts), 1)
    p = sp.order_u - 1
    if not sp.use_endpoint_u or sp.use_bezier_u or np.abs(pts[:, 3] - 1).max() > 1e-9:
        return "REFIT", p, pts[:, :3], None
    return "NURBS", p, pts[:, :3], geom.clamped_uniform_knots(len(pts), p)


def _refit(obj, deg, n, si=0):
    """Endpoint uniform NURBS fit of the current curve shape (local space)."""
    from .primitives import _fit_open_spec
    kc = features.input_curve(obj, f"S{si}")
    t0, t1 = kc.parameter_range()
    Mi = np.linalg.inv(np.array(geom.world_matrix(obj), float))
    S = [tuple(Mi[:3, :3] @ np.array(kc.evaluate(t)) + Mi[:3, 3]) for t in np.linspace(t0, t1, 400)]
    spec = _fit_open_spec(S, deg, n)
    return np.array(spec["points"], float), geom.clamped_uniform_knots(n, deg)


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

def edge_frame(edge_obj, ref, near, snap="END", flip=False, third=False):
    """World-space point, outward derivative D and second derivative DD on the
    edge: at its end nearest to `near` (END) or at the nearest point (NEAREST).
    third: also the third derivative DDD (outward)."""
    sm, _ = _k()
    kc = features.input_curve(edge_obj, ref)
    if snap == "NEAREST":
        t = curve_param_near(kc, near)
        d = sm.evaluate_curve(kc, t)
        Pt, D, DD = (np.array(d[k], float) for k in ("point", "tangent", "curvature"))
        D3 = geom.curve_derivs(kc, t, 3)[3] if third else None
        # continue in the direction the curve already heads away from the edge
        if flip:
            D = -D
            D3 = None if D3 is None else -D3
        return (Pt, D, DD, D3) if third else (Pt, D, DD)
    a, b = geom.curve_end_frame(kc, 0, third), geom.curve_end_frame(kc, 1, third)
    f = min((a, b), key=lambda f: np.linalg.norm(f[0] - near))
    Pt, D, DD = f[:3]
    if not third:
        return Pt, (-D if flip else D), DD
    return Pt, (-D if flip else D), DD, (-f[3] if flip else f[3])


def curve_param_near(kc, near):
    """Parameter of the point of kernel curve kc nearest `near` (world)."""
    sm, _ = _k()
    t0, t1 = kc.parameter_range()
    ts = np.linspace(t0, t1, 401)
    P = np.array([kc.evaluate(t) for t in ts])
    t = float(ts[int(np.argmin(np.linalg.norm(P - near, axis=1)))])
    for _ in range(30):                                  # Newton on |C(t) - near|^2
        d = sm.evaluate_curve(kc, t)
        c, c1, c2 = (np.array(d[k], float) for k in ("point", "tangent", "curvature"))
        g, h = (c - near) @ c1, c1 @ c1 + (c - near) @ c2
        if abs(h) < 1e-30:
            break
        t = min(max(t - g / h, t0), t1)
    return t


def target_face(obj, ref, face=0):
    """The B-rep face a target lies on: face F<i> itself, one of edge E<i>'s
    faces (face: which, for an edge between two), else the object's first."""
    brep = store.get_brep(obj)
    if is_face_ref(ref):
        return brep.faces()[int(ref[1:])]
    if features.is_brep_edge_ref(ref):
        fs = brep.edges()[int(ref[1:])].faces()
        return fs[min(max(int(face or 0), 0), len(fs) - 1)]
    return brep.faces()[0]


def surface_jet(obj, ref, point, face=0, uv=None):
    """The surface of a target at the point nearest `point` (world), or at
    (u, v) = uv (fractions of the face surface's domain): (P, n, Q, C, (u, v)
    fractions, kernel face). n the unit Su x Sv normal, Q the second
    fundamental form (3x3 on tangent vectors), C the third-order form
    (3x3x3, None when the surface can't give it), all world space and exact
    (surfeval: from the CVs)."""
    from .square import _foot, _second
    M = np.array(geom.world_matrix(obj), float)
    R, tr = M[:3, :3], M[:3, 3]
    f = target_face(obj, ref, face)
    s = f.surface()
    (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
    if uv is not None:
        u, v = u0 + float(uv[0]) * (u1 - u0), v0 + float(uv[1]) * (v1 - v0)
    else:
        u, v = _foot(s, np.linalg.solve(R, np.asarray(point, float) - tr))
    P = R @ np.array(s.evaluate(u, v), float) + tr
    try:
        from . import surfeval
        D = surfeval.Surf(s).at(np.array([u]), np.array([v]), 3)
        n, Q, C = surfeval.forms({kl: a @ R.T for kl, a in D.items()})
        n, Q, C = n[0], Q[0], C[0]
    except Exception:
        su, sv, suu, suv, svv = (R @ x for x in _second(s, u, v))
        n = np.cross(su, sv)
        n /= np.linalg.norm(n) or 1.0
        Jp = np.linalg.pinv(np.array([su, sv]).T)
        H = np.array([[suu @ n, suv @ n], [suv @ n, svv @ n]])
        Q, C = Jp.T @ H @ Jp, None
    fr = ((u - u0) / ((u1 - u0) or 1.0), (v - v0) / ((v1 - v0) or 1.0))
    return P, n, Q, C, fr, f


def surface_frame(edge_obj, ref, point, face=0):
    """(normal n, function k_n(direction)) of the edge's (or face's) surface
    at `point` (world). face: which of the edge's faces (an edge between two
    faces). The result also carries n, Q, C (SurfFrame), for G3."""
    _, n, Q, C, _, _ = surface_jet(edge_obj, ref, point, face)

    def k_n(t):
        t = np.asarray(t, float)
        t = t - (t @ n) * n
        tt = float(t @ t)
        return float(t @ Q @ t) / tt if tt > 1e-30 else 0.0
    out = SurfFrame((n, k_n))
    out.n, out.Q, out.C = n, Q, C
    return out


def surf_d3(surf, A, d1, d2, d3):
    """Third derivative (local) of an end tangent to the surface: d3's part
    in the tangent plane kept, its normal part what contact of order 3 needs,
    n.C3 = 3 Q(C1, C2t) + C(C1, C1, C1) (C2t: the tangent part of C2).
    A: world -> local (the curve object's inverse matrix, 3x3)."""
    n = surf.n
    Ai = np.linalg.inv(A)
    w1, w2, w3 = Ai @ d1, Ai @ d2, Ai @ d3
    w2t = w2 - (w2 @ n) * n
    w3t = w3 - (w3 @ n) * n
    c = 3.0 * float(w1 @ surf.Q @ w2t)
    if surf.C is not None:
        c += float(np.einsum("ijk,i,j,k->", surf.C, w1, w1, w1))
    return A @ (w3t + c * n)


def _outward_across(f, obj, P, n, e):
    """Unit direction in the tangent plane at P, square to the edge direction
    e, leading off face f (world)."""
    b = np.cross(n, e)
    b /= np.linalg.norm(b) or 1.0
    M = np.array(geom.world_matrix(obj), float)
    R, tr = M[:3, :3], M[:3, 3]
    lo, hi = (np.array(x, float) for x in f.bounding_box())
    eps = 1e-3 * (float(np.linalg.norm(hi - lo)) or 1.0)
    pl = np.linalg.solve(R, P - tr)
    bl = np.linalg.solve(R, b)
    bl /= np.linalg.norm(bl) or 1.0
    dp = f.closest_point(tuple(map(float, pl + eps * bl)))[1]
    dm = f.closest_point(tuple(map(float, pl - eps * bl)))[1]
    return -b if dm > dp else b         # +b goes back onto the face: off it is -b


def end_frame(src_obj, ref, near, snap="END", mode="EDGE", flip=False, face=0):
    """World frame (P, D, DD, surf) of an align target, as solve() builds it
    (a Frame: .d3 holds the third derivative). NORMAL: the surface's normal
    line at the point (D = the unit normal, turned round by flip; DD = DDD = 0),
    solved like an edge. A face target (F<i>) lies at the point of the face
    nearest `near`: NORMAL, or SURFACE (D then any tangent)."""
    near = np.asarray(near, float)
    mode = solve_mode(mode, src_obj, ref)
    z = np.zeros(3)
    if is_face_ref(ref):
        P, n, _, _, _, _ = surface_jet(src_obj, ref, near)
        if mode == "SURFACE":
            a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t = a - (a @ n) * n
            return frame(P, t / np.linalg.norm(t), z, surface_frame(src_obj, ref, P), z)
        return frame(P, -n if flip else n, z, None, z)
    Pt, D, DD, D3 = edge_frame(src_obj, ref, near, snap, False if mode in SQUARE else flip, third=True)
    if mode in SQUARE:
        _, n, _, _, _, _ = surface_jet(src_obj, ref, Pt, face)
        return frame(Pt, -n if flip else n, z, None, z)
    surf = surface_frame(src_obj, ref, Pt, face) if mode == "SURFACE" else None
    return frame(Pt, D, DD, surf, D3)


def solve_mode(mode, src_obj, ref):
    """How a target is solved: a curve is always continued (EDGE); a face is
    never (NORMAL instead); otherwise as asked."""
    if src_obj.type == 'CURVE' or not store.is_nurbs(src_obj):
        return "EDGE"
    if is_face_ref(ref) and mode not in ("SURFACE",) + SQUARE:
        return "NORMAL"
    return mode if mode in MODES else "EDGE"


def orient(spec, fr, toward):
    """Turn a frame round when it leads away from `toward` (a world vector
    from the end into the curve), toggling spec["flip"] to match, so the end
    leaves on the side the curve is. -> the frame."""
    if np.asarray(fr[1], float) @ np.asarray(toward, float) < 0:
        spec["flip"] = not spec.get("flip", False)
        return flipped(fr)
    return fr


def needs_orient(spec):
    """Ends whose direction is not fixed by the target alone: a point along an
    edge (EDGE NEAREST) and the normal line (NORMAL)."""
    m = spec.get("mode", "EDGE")
    return m in SQUARE or (m == "EDGE" and spec.get("snap") == "NEAREST")


# --------------------------------------------------------------------------
# Blend Curve's sides (features.blend_frames)
# --------------------------------------------------------------------------

def side_point(obj, ref, end=1, at=None, uv=None):
    """World point a blend side starts from: a curve / edge end, a point along
    it (at: 0..1 of its parameter range), or a face point (uv fractions)."""
    if is_face_ref(ref):
        return surface_jet(obj, ref, None, uv=uv or (0.5, 0.5))[0]
    kc = features.input_curve(obj, ref)
    t0, t1 = kc.parameter_range()
    t = (t1 if int(end) else t0) if at is None else t0 + float(at) * (t1 - t0)
    return np.array(kc.evaluate(t), float)


def side_frame(obj, ref, end=1, at=None, uv=None, mode="EDGE", face=0, third=False, toward=None):
    """(P, C1, C2[, C3]) world of what a blend side carries on:
      EDGE     the curve / edge itself: at an end outward (as always), at a
               point along it (at) forward, or toward `toward` (the other
               side's point) when given;
      SURFACE  tangent to the edge's surface, square to the edge and off the
               face (C2 / C3 normal to it: the surface's normal curvature and
               its rate in that direction, contact of order 2 / 3); on a face
               (F<i>), the direction toward `toward` in its tangent plane;
      NORMAL   perpendicular to the surface: its normal line (toward the
               other side), C2 = C3 = 0."""
    mode = solve_mode(mode, obj, ref)
    toward = None if toward is None else np.asarray(toward, float)
    z = np.zeros(3)
    if mode == "EDGE":
        kc = features.input_curve(obj, ref)
        if at is None:
            return geom.curve_end_frame(kc, int(end), third)
        t0, t1 = kc.parameter_range()
        P, D, DD, D3 = geom.curve_derivs(kc, t0 + float(at) * (t1 - t0), 3)
        if toward is not None and D @ (toward - P) < 0:
            D, D3 = -D, -D3
        return (P, D, DD, D3) if third else (P, D, DD)
    if is_face_ref(ref):
        P, n, Q, C, _, f = surface_jet(obj, ref, None, uv=uv or (0.5, 0.5))
    else:
        P = side_point(obj, ref, end, at)
        _, n, Q, C, _, f = surface_jet(obj, ref, P, face)
    if mode in SQUARE:
        D = n if toward is None or n @ (toward - P) >= 0 else -n
        if mode == "SYMMETRY":
            # the blend bends (Curvature) in the plane, toward the other side
            u = (toward - P) if toward is not None else np.array([1.0, 0.0, 0.0])
            u = u - (u @ n) * n
            if np.linalg.norm(u) < 1e-12:
                a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                u = a - (a @ n) * n
            return geom.sym_frame(P, D, u / np.linalg.norm(u), third)
        return (P, D, z, z) if third else (P, D, z)
    if is_face_ref(ref):
        t = (toward - P) if toward is not None else np.array([1.0, 0.0, 0.0])
        t = t - (t @ n) * n
        if np.linalg.norm(t) < 1e-12:
            a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t = a - (a @ n) * n
        b = t / np.linalg.norm(t)
    else:
        kc = features.input_curve(obj, ref)
        e = geom.curve_derivs(kc, curve_param_near(kc, P), 1)[1]
        b = _outward_across(f, obj, P, n, e / (np.linalg.norm(e) or 1.0))
    DD = float(b @ Q @ b) * n
    if not third:
        return P, b, DD
    c3 = float(np.einsum("ijk,i,j,k->", C, b, b, b)) if C is not None else 0.0
    return P, b, DD, c3 * n


# --------------------------------------------------------------------------
# End CV solve (local space of the curve object)
# --------------------------------------------------------------------------

def _derivs(Q, U, p):
    u1 = U[p + 1] - U[0]
    u2 = (U[p + 2] - U[0]) if p + 2 < len(U) else u1
    d1 = p / u1 * (Q[1] - Q[0])
    d2 = (p - 1) / u1 * (p * (Q[2] - Q[1]) / u2 - p * (Q[1] - Q[0]) / u1) if p >= 2 and len(Q) > 2 else np.zeros(3)
    return d1, d2, u1, u2


def _start_basis(U, p, k):
    """(k + 1, p + 1): the j-th derivative of basis function r at the start
    of a clamped knot vector (zero for r > j)."""
    U = np.asarray(U, float)
    key = (p, k, tuple(np.round(U[:min(len(U), 2 * p + 2)] - U[0], 13)))   # the knots that matter at the start
    hit = _START_BASIS.get(key)
    if hit is None:
        from .surfeval import basis_ders
        _, N = basis_ders(U, p, len(U) - p - 1, np.array([U[0]]), k)
        if len(_START_BASIS) > 256:
            _START_BASIS.clear()
        hit = _START_BASIS[key] = N[0]
    return hit


_START_BASIS = {}       # _start_basis results (CV Curve's fair Through Points asks thousands of times)


def _d3(Q, U, p):
    """C''' at the start of (Q, U, p) (zero below degree 3)."""
    if p < 3 or len(Q) < 4:
        return np.zeros(3)
    N = _start_basis(U, p, 3)
    return sum(N[3, r] * np.asarray(Q[r], float) for r in range(4))


def _set_start(Q, U, p, k, P0, d1, d2, d3=None):
    Q = Q.copy()
    if k >= 3:
        # Q0..Q3 from C, C', C'', C''' (the clamped start's triangular basis)
        N = _start_basis(U, p, 3)
        ders = [np.asarray(P0, float), d1, d2, np.zeros(3) if d3 is None else d3]
        for j in range(4):
            Q[j] = (ders[j] - sum(N[j, r] * Q[r] for r in range(j))) / N[j, j]
        return Q
    u1 = U[p + 1] - U[0]
    u2 = (U[p + 2] - U[0]) if p + 2 < len(U) else u1
    Q[0] = P0
    if k >= 1:
        Q[1] = P0 + (u1 / p) * d1
    if k >= 2:
        Q[2] = Q[1] + (u2 / p) * (d2 * u1 / (p - 1) + p * (Q[1] - Q[0]) / u1)
    return Q


def align_start(Q, U, p, spec, frame_world, Mi):
    """Align the start of (Q, U, p) to an edge frame. spec: cont, weight, mode.
    frame_world: (P, D, DD, surf) as end_frame (its d3 used for G3). NORMAL
    frames are the normal line, solved as an edge."""
    k = CONT[spec.get("cont", "G1")]
    w = float(spec.get("weight", 1.0))
    P, D, DD, surf = frame_world
    DDD = frame_d3(frame_world)
    A = Mi[:3, :3]
    P0 = A @ P + Mi[:3, 3]
    d1_cur, d2_cur, _, _ = _derivs(Q, U, p)
    d3_cur = _d3(Q, U, p) if k >= 3 else np.zeros(3)
    m = float(np.linalg.norm(d1_cur)) or float(np.linalg.norm(Q[-1] - Q[0])) or 1.0
    d3 = None
    if spec.get("mode", "EDGE") == "SYMMETRY":
        # frame: (P, the normal, toward the curve); C' along it, the mirror image joins C2 / C3
        n = A @ D
        n /= np.linalg.norm(n) or 1.0
        d1 = (m * w) * n
        d2 = d2_cur - (d2_cur @ n) * n                    # the bend kept, nothing along the normal
        d3 = float(d3_cur @ n) * n if k >= 3 else None    # nothing in the plane
        return _set_start(Q, U, p, k, P0, d1, d2, d3)
    if spec.get("mode", "EDGE") == "SURFACE" and surf is not None:
        n_w, k_n = surf
        n = A @ n_w
        n /= np.linalg.norm(n) or 1.0
        t = d1_cur - (d1_cur @ n) * n
        if np.linalg.norm(t) < 1e-12 * m:
            t = A @ D
            t -= (t @ n) * n
        t = t / (np.linalg.norm(t) or 1.0) * m * w
        d1 = t
        # normal curvature in the curve's direction (world direction for k_n)
        kn = k_n(np.linalg.solve(A, t / np.linalg.norm(t)))
        d2 = d2_cur - (d2_cur @ n) * n + kn * (t @ t) * n
        if k >= 3:
            d3 = surf_d3(surf, A, d1, d2, d3_cur) if getattr(surf, "Q", None) is not None else d3_cur
    else:
        Dl, DDl = A @ D, A @ DD
        dn2 = float(Dl @ Dl) or 1.0
        lam1 = m / dn2 ** 0.5
        lam = w * lam1
        d1 = lam * Dl
        # a fresh align: C'' = lam^2 C_e'' exactly. A rebuild (spec "keep"):
        # keep the curve's own part of C'' along the tangent (f'' of the
        # reparameterisation) -- still G2, and an aligned curve, CV-edited or
        # not, solves to itself
        mu = float((d2_cur - lam1 * lam1 * DDl) @ Dl) / dn2 if spec.get("keep") else 0.0
        d2 = lam * lam * DDl + mu * Dl
        if k >= 3:
            # C''' = lam^3 C_e''' + 3 lam mu C_e'' + nu C_e' (f''' = nu kept on a rebuild)
            D3l = A @ DDD if DDD is not None else np.zeros(3)
            nu = float((d3_cur - lam1 ** 3 * D3l - 3 * lam1 * mu * DDl) @ Dl) / dn2 if spec.get("keep") else 0.0
            d3 = lam ** 3 * D3l + 3 * lam * mu * DDl + nu * Dl
    return _set_start(Q, U, p, k, P0, d1, d2, d3)


# --------------------------------------------------------------------------
# An aligned end's free numbers (constrained CV editing, blendedit.py)
# --------------------------------------------------------------------------

class EndShape:
    """The CVs Q0..Qk an aligned start of (seg, Ub, p) may take, as a function
    of a few numbers x -- everything else would break the alignment:
      EDGE     x = (s, mu[, nu]): Q1 = P + s T on the edge's tangent; C'' =
               lam^2 C_e'' + mu C_e' (lam from s), so Q2 only slides along the
               tangent beyond the height its curvature needs; G3: C''' =
               lam^3 C_e''' + 3 lam mu C_e'' + nu C_e'. NORMAL is this on the
               normal line.
      SURFACE  x = (t1, t2, e1, e2[, f1, f2]): C' = t in the tangent plane,
               C'' = e in it plus the surface's normal curvature along t times
               |t|^2, G3: C''' = f in it plus what order-3 contact needs.
      SYMMETRY x = (s[, b1, b2[, h]]): Q1 = P + s n, C'' = b in the plane (the
               bend), C''' = h n.
    frame_world = (P, D, DD, surf) as align_start; A, b = the curve's world ->
    local matrix (rotation part, offset)."""

    def __init__(self, Ub, p, k, mode, frame_world, A, b, size):
        self.Ub, self.p, self.k = Ub, p, k
        P, D, DD, surf = frame_world
        self.A = A
        self.P0 = A @ P + b
        self.mode = "SURFACE" if mode == "SURFACE" and surf is not None else (
            "SYMMETRY" if mode == "SYMMETRY" else "EDGE")
        self.smin = 1e-3 * size
        if self.mode == "SYMMETRY":
            n = A @ D
            self.n = self.T = n / (np.linalg.norm(n) or 1.0)
            a = np.array([1.0, 0.0, 0.0]) if abs(self.n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e1 = a - (a @ self.n) * self.n
            self.e1 = e1 / np.linalg.norm(e1)
            self.e2 = np.cross(self.n, self.e1)
        elif self.mode == "EDGE":
            self.D, self.DD = A @ D, A @ DD
            d3 = frame_d3(frame_world)
            self.D3 = A @ d3 if d3 is not None else np.zeros(3)
            self.T = self.D / (np.linalg.norm(self.D) or 1.0)
        else:
            self.surf = surf
            n_w, self.k_n = surf
            n = A @ n_w
            self.n = n / (np.linalg.norm(n) or 1.0)
            a = np.array([1.0, 0.0, 0.0]) if abs(self.n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e1 = a - (a @ self.n) * self.n
            self.e1 = e1 / np.linalg.norm(e1)
            self.e2 = np.cross(self.n, self.e1)

    def size(self):
        if self.mode == "SYMMETRY":
            return [0, 1, 3, 4][self.k]
        return self.k if self.mode == "EDGE" else 2 * self.k

    def clamp(self, x):
        x = np.array(x, float)
        if self.mode in ("EDGE", "SYMMETRY") and len(x):
            x[0] = max(x[0], self.smin)
        return x

    def rows(self, x, seg):
        """Q0..Qk for the numbers x (seg: the curve's CVs from this end, for _set_start's shape)."""
        p, Ub, k = self.p, self.Ub, self.k
        u1 = Ub[p + 1] - Ub[0]
        if k == 0:
            return [self.P0]
        d3 = None
        if self.mode == "SYMMETRY":
            d1 = (p / u1) * max(float(x[0]), self.smin) * self.n
            d2 = x[1] * self.e1 + x[2] * self.e2 if k >= 2 else np.zeros(3)
            d3 = float(x[3]) * self.n if k >= 3 else None
        elif self.mode == "EDGE":
            s = max(float(x[0]), self.smin)
            d1 = (p / u1) * s * self.T
            if k >= 2:
                lam = np.linalg.norm(d1) / (np.linalg.norm(self.D) or 1.0)
                d2 = lam * lam * self.DD + float(x[1]) * self.D
                if k >= 3:
                    d3 = lam ** 3 * self.D3 + 3 * lam * float(x[1]) * self.DD + float(x[2]) * self.D
            else:
                d2 = np.zeros(3)
        else:
            d1 = x[0] * self.e1 + x[1] * self.e2
            d2 = np.zeros(3)
            if k >= 2:
                tn = float(np.linalg.norm(d1))
                kn = self.k_n(np.linalg.solve(self.A, d1 / tn)) if tn > 1e-300 else 0.0
                d2 = x[2] * self.e1 + x[3] * self.e2 + kn * tn * tn * self.n
            if k >= 3:
                d3t = x[4] * self.e1 + x[5] * self.e2
                d3 = surf_d3(self.surf, self.A, d1, d2, d3t) if getattr(self.surf, "Q", None) is not None else d3t
        Q = _set_start(np.asarray(seg, float), Ub, p, k, self.P0, d1, d2, d3)
        return [Q[i] for i in range(k + 1)]

    def numbers(self, seg):
        """x read off the curve's present end CVs (projected where they are off)."""
        p, Ub, k = self.p, self.Ub, self.k
        if k == 0:
            return np.zeros(0)
        seg = np.asarray(seg, float).copy()
        seg[0] = self.P0
        d1, d2, _, _ = _derivs(seg, Ub, p)
        d3 = _d3(seg, Ub, p) if k >= 3 else None
        if self.mode == "SYMMETRY":
            x = [float((seg[1] - seg[0]) @ self.n)]
            if k >= 2:
                x += [float(d2 @ self.e1), float(d2 @ self.e2)]
            if k >= 3:
                x.append(float(d3 @ self.n))
            return self.clamp(x)
        if self.mode == "EDGE":
            s = float((seg[1] - seg[0]) @ self.T)
            x = [s]
            dn2 = float(self.D @ self.D) or 1.0
            if k >= 2:
                lam = (p / (Ub[p + 1] - Ub[0])) * max(s, self.smin) / (np.linalg.norm(self.D) or 1.0)
                mu = float((d2 - lam * lam * self.DD) @ self.D) / dn2
                x.append(mu)
                if k >= 3:
                    x.append(float((d3 - lam ** 3 * self.D3 - 3 * lam * mu * self.DD) @ self.D) / dn2)
            return self.clamp(x)
        x = [float(d1 @ self.e1), float(d1 @ self.e2)]
        if k >= 2:
            x += [float(d2 @ self.e1), float(d2 @ self.e2)]
        if k >= 3:
            x += [float(d3 @ self.e1), float(d3 @ self.e2)]
        return np.array(x)


def _reverse(Q, U):
    a, b = U[0], U[-1]
    return Q[::-1].copy(), [a + b - u for u in reversed(U)]


# --------------------------------------------------------------------------
# Aligning CVs before there is an object (CV Curve's start / end conditions)
# --------------------------------------------------------------------------

def degree_for(conts):
    """Lowest degree whose end CVs can hold these conditions (as solve() needs):
    G1 2, G2 3 (curvature continuous along the curve), G3 4 (its rate too)."""
    k = max([CONT[c] for c in conts] + [0])
    return 4 if k >= 3 else (3 if k >= 2 else (2 if k >= 1 else 1))


def refit_count(n_cvs, need, deg):
    """CVs of a curve that must be refitted for its aligned ends (a degree
    raised, or too few CVs; need: each end's G level). The ends take
    sum(need) CVs from it besides their end points: those are added, so the
    CVs the curve had stay free to keep its shape (else G3, raising a 5-CV
    cubic to a 5-CV quartic and fixing 4 of them, bends it far off)."""
    k_need = sum(need) + len(need)
    return max(n_cvs + sum(need), k_need, deg + 1)


def _refit_points(Q, p, n, q):
    """n CVs of degree q (endpoint uniform) fitting the curve of CVs Q, degree p."""
    from .curvetools import Basis, fit
    S = Basis(len(Q), p, False).matrix(np.linspace(0.0, 1.0, 40 * len(Q) + 1)) @ Q
    return fit(S, q, n, False)[0]


def _insert_cv(Q, e):
    """One more CV on the leg next to end e ("0" / "1"), a third of the way in."""
    Q = np.asarray(Q, float)
    if e == "1":
        return _insert_cv(Q[::-1], "0")[::-1].copy()
    return np.vstack([Q[:1], Q[:1] + (Q[1] - Q[0]) / 3.0, Q[1:]])


def align_open(Q, p, ends, want=None):
    """World CVs (and degree) of an open endpoint-uniform spline with its ends
    aligned. ends = {"0" / "1": (spec, frame)}: spec as an align_curve end
    (cont, weight, mode, snap), frame from end_frame(). The degree is raised and
    the curve refitted with more CVs (same shape) when the ends need it, as
    solve() would, so the result recorded as an align_curve feature rebuilds
    unchanged. want: the degree asked for (CV Curve): p is that degree clamped
    to the clicks, and when the ends add CVs the degree rises towards want, up
    to the final CV count - 1 (else two clicks with G2 ends always came out
    cubic). A NEAREST target in EDGE mode is flipped to leave the way the
    curve already heads, a NORMAL one to the side the curve is (spec["flip"]
    set). Returns (Q, p, specs)."""
    Q = np.asarray(Q, float)
    specs = {e: dict(s) for e, (s, _) in ends.items()}
    need = [CONT[s.get("cont", "G1")] for s in specs.values()]
    if not need:
        return Q, p, specs
    q = max(p, degree_for([s.get("cont", "G1") for s in specs.values()]))
    # CV mode: the clicks are the CVs, and an end moves the ones next to it (G1
    # the 2nd, G2 the 3rd too). A G3 end adds a CV on its first leg instead of
    # taking a 4th click, so it moves the same clicks as G2 and the shape holds
    Q0 = Q
    for e, (s_, _) in sorted(ends.items()):
        if CONT[s_.get("cont", "G1")] >= 3 and len(Q) >= 2:
            Q = _insert_cv(Q, e)
    refit = max(len(Q), sum(need) + len(need), q + 1) != len(Q)     # too few CVs: refit, with room
    # A refit counts the ends' CVs itself, so it starts from the clicks without
    # the G3 legs' CVs: counted twice, they left spare CVs on the clicks'
    # chord, and the curve dipped to it between the ends (2 inflections).
    n_out = refit_count(len(Q0), need, q) if refit else len(Q)
    if want is not None:
        q = max(q, min(int(want), n_out - 1))
    if refit:
        Q = _refit_points(Q0, p, n_out, q)
    p = q
    U = geom.clamped_uniform_knots(len(Q), p)
    I = np.eye(4)
    for e in ("0", "1"):
        if e not in ends:
            continue
        spec = specs[e]
        fr = ends[e][1]
        seg, Ub = (Q, U) if e == "0" else _reverse(Q, U)
        if spec.get("mode", "EDGE") in SQUARE:
            fr = orient(spec, fr, np.mean(seg[1:], axis=0) - seg[0])
        elif spec.get("snap") == "NEAREST" and spec.get("mode", "EDGE") == "EDGE":
            fr = orient(spec, fr, _derivs(seg, Ub, p)[0])
        new = align_start(seg, Ub, p, spec, fr, I)
        Q = new if e == "0" else _reverse(new, Ub)[0]
    return Q, p, specs


def move_end(obj, si, end, P):
    """Put the end CV (end 0 / 1) of spline si at world point P (a Bezier end
    takes its handles along)."""
    sp = obj.data.splines[si]
    Mi = np.linalg.inv(np.array(geom.world_matrix(obj), float))
    q = Mi[:3, :3] @ np.asarray(P, float) + Mi[:3, 3]
    if sp.type == 'BEZIER':
        b = sp.bezier_points[-1 if end else 0]
        d = q - np.array(b.co, float)
        b.co = tuple(q)
        b.handle_left = tuple(np.array(b.handle_left, float) + d)
        b.handle_right = tuple(np.array(b.handle_right, float) + d)
    else:
        pt = sp.points[-1 if end else 0]
        pt.co = (*map(float, q), pt.co[3])
    obj.data.update_tag()


def spline_index(inputs):
    """Which spline of the curve object an Align feature moves (its first input, S<i>)."""
    try:
        r = inputs[0][1]
        return int(r[1:]) if r.startswith("S") and r[1:].isdigit() else 0
    except Exception:
        return 0


def solve(obj, ends, inputs, fresh=()):
    """New local CVs (and knots/degree) for the object's curve with the given
    ends aligned. ends: {"0"/"1": spec}; inputs: [(obj, ref)] (index 0 = the curve).
    fresh: the ends aligned just now, whose Weight scales the curve's tangent.
    The others already carry their weight: a rebuild must not scale them again
    (it compounded on every rebuild)."""
    si = spline_index(inputs)
    kind, p, Q, U = curve_state(obj, si)
    M = np.array(geom.world_matrix(obj), float)
    Mi = np.linalg.inv(M)
    need = [CONT[ends[e].get("cont", "G1")] for e in ("0", "1") if e in ends]
    k_need = sum(need) + len(need)          # CVs used by the ends
    if kind == "BEZIER":
        nseg = (len(Q) - 1) // 3
        if 3 not in need and (nseg >= 2 or k_need <= 4):
            kind_out = "BEZIER"
        else:
            kind = "REFIT"                  # (G3 needs a whole cubic span per end: refit)
    g3 = 3 in need
    if kind == "REFIT" or (kind == "NURBS" and (len(Q) < max(k_need, 2) or p < max(need + [1]) + (1 if 2 in need else 0)
                                                  or (p == 1 and max(need + [0]) >= 1) or (g3 and p < 4))):
        deg = max(4 if g3 else 3, p) if kind != "REFIT" or p < (4 if g3 else 2) else p
        n = refit_count(len(Q), need, deg)
        Q, U = _refit(obj, deg, n, si)
        p, kind = deg, "NURBS"
    if kind == "NURBS":
        kind_out = "NURBS"

    def world_end(local_pt):
        return M[:3, :3] @ local_pt + M[:3, 3]

    for e in ("0", "1"):
        if e not in ends:
            continue
        spec = ends[e]
        src, ref = inputs[int(spec["input"])]
        if kind_out == "BEZIER":
            seg = Q[:4] if e == "0" else Q[-4:][::-1]
            Ub = [0.0] * 4 + [1.0] * 4
        else:
            seg, Ub = (Q, U) if e == "0" else _reverse(Q, U)
        near = world_end(seg[0])
        fr = end_frame(src, ref, near, spec.get("snap", "END"), spec.get("mode", "EDGE"),
                       spec.get("flip", False), spec.get("face", 0))
        if e not in fresh:
            spec = dict(spec, weight=1.0, keep=True)
        spec = dict(spec, mode=solve_mode(spec.get("mode", "EDGE"), src, ref))
        new = align_start(seg, Ub, 3 if kind_out == "BEZIER" else p, spec, fr, Mi)
        if kind_out == "BEZIER":
            if e == "0":
                Q[:4] = new
            else:
                Q[-4:] = new[::-1]
        elif e == "0":
            Q = new
        else:
            Q, U = _reverse(new, Ub)
    return kind_out, p, Q, U


def _spline_data(sp):
    """Everything needed to make spline sp again (_respline)."""
    d = {"type": sp.type, "order_u": sp.order_u, "use_endpoint_u": sp.use_endpoint_u,
         "use_bezier_u": sp.use_bezier_u, "use_cyclic_u": sp.use_cyclic_u, "resolution_u": sp.resolution_u,
         "material_index": sp.material_index, "use_smooth": sp.use_smooth}
    if sp.type == 'BEZIER':
        d["bp"] = [(tuple(b.co), tuple(b.handle_left), tuple(b.handle_right), b.handle_left_type,
                    b.handle_right_type, b.radius, b.tilt) for b in sp.bezier_points]
    else:
        d["pts"] = [(tuple(q.co), q.radius, q.tilt) for q in sp.points]
    return d


def _respline(cu, d):
    sp = cu.splines.new(d["type"])
    if d["type"] == 'BEZIER':
        sp.bezier_points.add(len(d["bp"]) - 1)
        for b, (co, hl, hr, tl, tr, r, t) in zip(sp.bezier_points, d["bp"]):
            b.co, b.handle_left_type, b.handle_right_type = co, tl, tr
            b.handle_left, b.handle_right, b.radius, b.tilt = hl, hr, r, t
    else:
        sp.points.add(len(d["pts"]) - 1)
        for q, (co, r, t) in zip(sp.points, d["pts"]):
            q.co, q.radius, q.tilt = co, r, t
        sp.order_u = d["order_u"]
        sp.use_endpoint_u, sp.use_bezier_u = d["use_endpoint_u"], d["use_bezier_u"]
    sp.use_cyclic_u, sp.resolution_u = d["use_cyclic_u"], d["resolution_u"]
    sp.material_index, sp.use_smooth = d["material_index"], d["use_smooth"]
    return sp


def _nurbs_spline(cu, si, n):
    """Spline si of cu as a NURBS spline of n points; the other splines kept
    as they are, in their order (Blender can't remove points or put a new
    spline at an index: the splines are made again when it must)."""
    sp = cu.splines[si]
    if sp.type == 'NURBS' and len(sp.points) == n:
        return sp
    if sp.type == 'NURBS' and len(sp.points) < n:
        sp.points.add(n - len(sp.points))
        return sp
    keep = [_spline_data(s_) for s_ in cu.splines]
    mat = keep[si]["material_index"]
    keep[si] = {"type": 'NURBS', "order_u": 2, "use_endpoint_u": True, "use_bezier_u": False,
                "use_cyclic_u": False, "resolution_u": 24, "material_index": mat,
                "use_smooth": keep[si]["use_smooth"], "pts": [((0.0, 0.0, 0.0, 1.0), 1.0, 0.0)] * n}
    cu.splines.clear()
    for d in keep:
        _respline(cu, d)
    return cu.splines[si]


def write(obj, kind, p, Q, si=0):
    """Write the solved CVs back into spline si of the curve object (local space)."""
    cu = obj.data
    sp = cu.splines[si]
    if kind == "BEZIER":
        bp = sp.bezier_points
        for i, b in enumerate(bp):
            b.handle_left_type = b.handle_right_type = 'FREE'
        k = 0
        for i, b in enumerate(bp):
            if i:
                b.handle_left = Q[k]
                k += 1
            b.co = Q[k]
            k += 1
            if i < len(bp) - 1:
                b.handle_right = Q[k]
                k += 1
        # end handles outside the curve: keep them on the tangent line
        b0, b1 = bp[0], bp[-1]
        b0.handle_left = 2 * np.array(b0.co) - np.array(b0.handle_right)
        b1.handle_right = 2 * np.array(b1.co) - np.array(b1.handle_left)
    else:
        if sp.type != 'NURBS' or len(sp.points) != len(Q) or sp.order_u != p + 1:
            if len(cu.splines) == 1:
                mat = [sp.material_index]
                cu.splines.clear()
                sp = cu.splines.new('NURBS')
                sp.points.add(len(Q) - 1)
                sp.material_index = mat[0]
                sp.resolution_u = 24
            else:
                sp = _nurbs_spline(cu, si, len(Q))
        for pt, q in zip(sp.points, Q):
            pt.co = (*map(float, q), 1.0)
        sp.order_u = p + 1
        sp.use_endpoint_u = True
        sp.use_bezier_u = False
        sp.use_cyclic_u = False
    cu.update_tag()


@features.builder("align_curve")
def build_align(inputs, p):
    obj = inputs[0][0]
    ends = p.get("ends", {})
    kind, deg, Q, _ = solve(obj, ends, inputs)          # (no fresh end: the weights are in the curve already)
    si = spline_index(inputs)
    return ("inplace", lambda o: write(o, kind, deg, Q, si))


def measure(obj, end, edge_obj, ref, snap="END"):
    """(gap, tangent angle deg, curvature difference) between the curve end and
    the edge continuation (EDGE mode check)."""
    sm, _ = _k()
    kc = features.input_curve(obj, "S0")
    fc = geom.curve_end_frame(kc, int(end))
    P, D, DD = fc
    Pt, De, DDe = edge_frame(edge_obj, ref, P, snap)
    t_c = -D / np.linalg.norm(D)                  # curve leaving the edge end
    t_e = De / np.linalg.norm(De)
    ang = float(np.degrees(np.arccos(np.clip(t_c @ t_e, -1, 1))))
    def kvec(d1, d2):
        return np.cross(np.cross(d1, d2), d1) / (d1 @ d1) ** 2
    kc_ = kvec(-D, DD)
    ke = kvec(De, DDe)
    return float(np.linalg.norm(P - Pt)), ang, float(np.linalg.norm(kc_ - ke))
