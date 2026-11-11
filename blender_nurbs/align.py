"""Align Curve (Alias "Align"): move a drawn curve's end CVs so the end is
G0 / G1 / G2 to an edge, keeping the rest of the curve.

Per end, with the end parameter t = 0 of a clamped curve of degree p, knots U:
    C(0)   = Q0
    C'(0)  = p / u1 * (Q1 - Q0)                         u1 = U[p+1] - U[0]
    C''(0) = (p-1)/u1 * ( p (Q2-Q1)/u2 - p (Q1-Q0)/u1 )   u2 = U[p+2] - U[0]
so G0 sets Q0, G1 sets Q1, G2 sets Q2.

Modes
  EDGE     continue the edge: C'(0) = lam * D (edge's outward derivative),
           C''(0) = lam^2 * DD  -> the curve and the edge form one G1/G2 curve.
  SURFACE  stay tangent to the edge's surface: C'(0) = the curve's own end
           tangent projected into the tangent plane; the normal component of
           C''(0) matches the surface's normal curvature in that direction.
Snap: END (the nearest end of the edge) or NEAREST (nearest point on the edge).

The curve itself is the feature's first input, so moving its other CVs keeps
the ends locked to the edges (history), like Alias with construction history on.
"""

import json

import numpy as np
from mathutils import Vector

from . import features, geom, store

CONT = {"G0": 0, "G1": 1, "G2": 2}


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


# --------------------------------------------------------------------------
# The curve being aligned (object-local space)
# --------------------------------------------------------------------------

def curve_state(obj):
    """(kind, p, Q (n,3) local, U) of the object's first spline. NURBS splines
    are used as clamped uniform (endpoint) curves; Bezier splines as their first
    and last cubic segments."""
    sp = obj.data.splines[0]
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


def _refit(obj, deg, n):
    """Endpoint uniform NURBS fit of the current curve shape (local space)."""
    from .primitives import _fit_open_spec
    kc = features.input_curve(obj, "S0")
    t0, t1 = kc.parameter_range()
    Mi = np.linalg.inv(np.array(geom.world_matrix(obj), float))
    S = [tuple(Mi[:3, :3] @ np.array(kc.evaluate(t)) + Mi[:3, 3]) for t in np.linspace(t0, t1, 400)]
    spec = _fit_open_spec(S, deg, n)
    return np.array(spec["points"], float), geom.clamped_uniform_knots(n, deg)


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

def edge_frame(edge_obj, ref, near, snap="END", flip=False):
    """World-space point, outward derivative D and second derivative DD on the
    edge: at its end nearest to `near` (END) or at the nearest point (NEAREST)."""
    sm, _ = _k()
    kc = features.input_curve(edge_obj, ref)
    t0, t1 = kc.parameter_range()
    if snap == "NEAREST":
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
        d = sm.evaluate_curve(kc, t)
        Pt, D, DD = (np.array(d[k], float) for k in ("point", "tangent", "curvature"))
        # continue in the direction the curve already heads away from the edge
        if flip:
            D = -D
        return Pt, D, DD
    a, b = geom.curve_end_frame(kc, 0), geom.curve_end_frame(kc, 1)
    Pt, D, DD = min((a, b), key=lambda f: np.linalg.norm(f[0] - near))
    return Pt, (-D if flip else D), DD


def surface_frame(edge_obj, ref, point, face=0):
    """(normal n, function k_n(direction)) of the edge's surface at `point`
    (world). face: which of the edge's faces (an edge between two faces)."""
    from .square import _foot, _second
    M = np.array(geom.world_matrix(edge_obj), float)
    R = M[:3, :3]
    brep = store.get_brep(edge_obj)
    pl = np.linalg.solve(R, point - M[:3, 3])
    if features.is_brep_edge_ref(ref):
        fs = brep.edges()[int(ref[1:])].faces()
        face = fs[min(max(int(face or 0), 0), len(fs) - 1)]
    else:
        face = brep.faces()[0]
    s = face.surface()
    u, v = _foot(s, pl)
    su, sv, suu, suv, svv = (R @ x for x in _second(s, u, v))
    n = np.cross(su, sv)
    n /= np.linalg.norm(n) or 1.0

    def k_n(t):
        a, c = np.linalg.lstsq(np.array([su, sv]).T, t, rcond=None)[0]
        num = a * a * (suu @ n) + 2 * a * c * (suv @ n) + c * c * (svv @ n)
        den = a * a * (su @ su) + 2 * a * c * (su @ sv) + c * c * (sv @ sv)
        return num / den if den > 1e-30 else 0.0
    return n, k_n


# --------------------------------------------------------------------------
# End CV solve (local space of the curve object)
# --------------------------------------------------------------------------

def _derivs(Q, U, p):
    u1 = U[p + 1] - U[0]
    u2 = (U[p + 2] - U[0]) if p + 2 < len(U) else u1
    d1 = p / u1 * (Q[1] - Q[0])
    d2 = (p - 1) / u1 * (p * (Q[2] - Q[1]) / u2 - p * (Q[1] - Q[0]) / u1) if p >= 2 and len(Q) > 2 else np.zeros(3)
    return d1, d2, u1, u2


def _set_start(Q, U, p, k, P0, d1, d2):
    Q = Q.copy()
    u1 = U[p + 1] - U[0]
    u2 = (U[p + 2] - U[0]) if p + 2 < len(U) else u1
    Q[0] = P0
    if k >= 1:
        Q[1] = P0 + (u1 / p) * d1
    if k >= 2:
        Q[2] = Q[1] + (u2 / p) * (d2 * u1 / (p - 1) + p * (Q[1] - Q[0]) / u1)
    return Q


def align_start(Q, U, p, spec, frame_world, Mi):
    """Align the start of (Q, U, p) to an edge frame. spec: cont, weight, mode."""
    k = CONT[spec.get("cont", "G1")]
    w = float(spec.get("weight", 1.0))
    P, D, DD, surf = frame_world
    A = Mi[:3, :3]
    P0 = A @ P + Mi[:3, 3]
    d1_cur, d2_cur, _, _ = _derivs(Q, U, p)
    m = float(np.linalg.norm(d1_cur)) or float(np.linalg.norm(Q[-1] - Q[0])) or 1.0
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
    else:
        Dl = A @ D
        lam = w * m / (np.linalg.norm(Dl) or 1.0)
        d1 = lam * Dl
        d2 = lam * lam * (A @ DD)
    return _set_start(Q, U, p, k, P0, d1, d2)


def _reverse(Q, U):
    a, b = U[0], U[-1]
    return Q[::-1].copy(), [a + b - u for u in reversed(U)]


# --------------------------------------------------------------------------
# Aligning CVs before there is an object (CV Curve's start / end conditions)
# --------------------------------------------------------------------------

def degree_for(conts):
    """Lowest degree whose end CVs can hold these conditions (as solve() needs)."""
    k = max([CONT[c] for c in conts] + [0])
    return 3 if k >= 2 else (2 if k >= 1 else 1)


def end_frame(src_obj, ref, near, snap="END", mode="EDGE", flip=False, face=0):
    """World frame (P, D, DD, surf) of an align target, as solve() builds it."""
    Pt, D, DD = edge_frame(src_obj, ref, np.asarray(near, float), snap, flip)
    surf = surface_frame(src_obj, ref, Pt, face) if mode == "SURFACE" and store.is_nurbs(src_obj) else None
    return Pt, D, DD, surf


def _refit_points(Q, p, n, q):
    """n CVs of degree q (endpoint uniform) fitting the curve of CVs Q, degree p."""
    from .curvetools import Basis, fit
    S = Basis(len(Q), p, False).matrix(np.linspace(0.0, 1.0, 40 * len(Q) + 1)) @ Q
    return fit(S, q, n, False)[0]


def align_open(Q, p, ends):
    """World CVs (and degree) of an open endpoint-uniform spline with its ends
    aligned. ends = {"0" / "1": (spec, frame)}: spec as an align_curve end
    (cont, weight, mode, snap), frame from end_frame(). The degree is raised and
    the curve refitted with more CVs (same shape) when the ends need it, as
    solve() would, so the result recorded as an align_curve feature rebuilds
    unchanged. A NEAREST target in EDGE mode is flipped to leave the way the
    curve already heads (spec["flip"] set). Returns (Q, p, specs)."""
    Q = np.asarray(Q, float)
    specs = {e: dict(s) for e, (s, _) in ends.items()}
    need = [CONT[s.get("cont", "G1")] for s in specs.values()]
    if not need:
        return Q, p, specs
    q = max(p, degree_for([s.get("cont", "G1") for s in specs.values()]))
    n = max(len(Q), sum(need) + len(need), q + 1)
    if n != len(Q):
        Q = _refit_points(Q, p, n, q)
    p = q
    U = geom.clamped_uniform_knots(len(Q), p)
    I = np.eye(4)
    for e in ("0", "1"):
        if e not in ends:
            continue
        spec = specs[e]
        P, D, DD, surf = ends[e][1]
        seg, Ub = (Q, U) if e == "0" else _reverse(Q, U)
        if spec.get("snap") == "NEAREST" and spec.get("mode", "EDGE") == "EDGE":
            if D @ _derivs(seg, Ub, p)[0] < 0:
                D = -D
                spec["flip"] = not spec.get("flip", False)
        new = align_start(seg, Ub, p, spec, (P, D, DD, surf), I)
        Q = new if e == "0" else _reverse(new, Ub)[0]
    return Q, p, specs


def solve(obj, ends, inputs):
    """New local CVs (and knots/degree) for the object's curve with the given
    ends aligned. ends: {"0"/"1": spec}; inputs: [(obj, ref)] (index 0 = the curve)."""
    kind, p, Q, U = curve_state(obj)
    M = np.array(geom.world_matrix(obj), float)
    Mi = np.linalg.inv(M)
    need = [CONT[ends[e].get("cont", "G1")] for e in ("0", "1") if e in ends]
    k_need = sum(need) + len(need)          # CVs used by the ends
    if kind == "BEZIER":
        nseg = (len(Q) - 1) // 3
        if nseg >= 2 or k_need <= 4:
            kind_out = "BEZIER"
        else:
            kind = "REFIT"
    if kind == "REFIT" or (kind == "NURBS" and (len(Q) < max(k_need, 2) or p < max(need + [1]) + (1 if 2 in need else 0)
                                                  or (p == 1 and max(need + [0]) >= 1))):
        deg = max(3, p) if kind != "REFIT" or p < 2 else p
        n = max(len(Q), k_need, deg + 1)
        Q, U = _refit(obj, deg, n)
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
        Pt, D, DD = edge_frame(src, ref, near, spec.get("snap", "END"), spec.get("flip", False))
        surf = None
        if spec.get("mode") == "SURFACE" and store.is_nurbs(src):
            surf = surface_frame(src, ref, Pt, spec.get("face", 0))
        new = align_start(seg, Ub, 3 if kind_out == "BEZIER" else p, spec, (Pt, D, DD, surf), Mi)
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


def write(obj, kind, p, Q):
    """Write the solved CVs back into the curve object (local space)."""
    cu = obj.data
    sp = cu.splines[0]
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
            mat = [sp.material_index]
            cu.splines.clear()
            sp = cu.splines.new('NURBS')
            sp.points.add(len(Q) - 1)
            sp.material_index = mat[0]
            sp.resolution_u = 24
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
    kind, deg, Q, _ = solve(obj, ends, inputs)
    return ("inplace", lambda o: write(o, kind, deg, Q))


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
