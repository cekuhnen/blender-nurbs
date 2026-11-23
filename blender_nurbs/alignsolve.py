"""Align solver: the rigid move that mates two snap elements.

Pure numpy (no bpy), so it is unit tested on its own. The elements are the
dicts of snapframes.element ({"type", "o", "d", "pin", "hint", ...}, world
space); make() builds one by hand.

mate(a, b) moves the part a belongs to so that a meets b:

1. Directions. Each direction plays one of three roles:
     plane   a PLANE's normal                   signed (points out of the face)
     axis    a RING's or an AXIS's axis         a RING's is signed (out of the
                                                face its rim is on), an AXIS's not
     line    a LINE's direction, or the edge    unsigned
             a mid point carries (its hint)
   A plane and a line want the line in the plane (perpendicular to the
   normal); any other pair wants the directions parallel: opposed when both
   are signed (faces flush and facing each other, a peg's end ring onto a
   hole's rim), otherwise whichever sense is nearer (axes coaxial, edges
   parallel). The rotation is the smallest one that does it. Two flat faces
   then also turn their edge runs into line (the nearest quarter turn).
   A point without a direction turns nothing.
2. Position. The part turns about a's point. Each element leaves its
   point some freedom: a pinned one (point, ring centre, face centre, axis
   middle) none, a line or an axis along itself, a face in its plane. The
   move is the smallest one that brings a's freedom onto b's (the part of
   the gap between their points that neither freedom covers).
3. Refinements, about the point p where the two meet (on both freedoms,
   nearest where a's point landed, so the mate holds through them):
     flip     half a turn: the other sense (faces back to back), or for a
              line in a plane the other side of the plane
     spin     radians about the mated direction (spin_axis: b's, or the
              plane normal for a line in a plane)
     second   a second pair (a2, b2) of elements picked before the move:
              the spin that lines a2 up with b2 (directions where both have
              one across the spin axis, signed ones the same way round as
              flush side faces; otherwise their points round the axis)
     gap      a distance along b's outward direction (away from b)

Result: {"M" 4x4 world matrix (new = M @ old), "R", "t", "p" (the meeting
point, before the gap), "axis" (spin axis or None), "gap_dir" (or None),
"relation" ("parallel", "opposed", "in plane", "none"), "angle" (degrees
turned), "dist" (how far a's point moved), "spin" (radians used, the
second pair's included)}.
"""

import math

import numpy as np

EPS = 1e-12


def make(type, o, d=None, pin=None, hint=None, r=None):
    """An element by hand (tests, typed input). pin defaults to the type's
    usual: points and rings pinned, lines, faces and axes free."""
    if pin is None:
        pin = type in ("POINT", "RING")
    u = lambda v: None if v is None else _unit(np.asarray(v, float))
    return {"type": type, "o": np.asarray(o, float), "d": u(d), "pin": bool(pin), "hint": u(hint), "r": r}


# --------------------------------------------------------------------------
# small vector helpers
# --------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else None


def _perp(n):
    """A unit vector perpendicular to unit n."""
    a = np.array((1.0, 0, 0)) if abs(n[0]) < 0.9 else np.array((0, 1.0, 0))
    return _unit(np.cross(n, a))


def rotation(axis, angle):
    """3x3 rotation by angle (radians) about unit axis (right hand)."""
    k = np.asarray(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def rotation_between(u, v, fallback=None):
    """The smallest rotation taking unit u onto unit v; half a turn about
    fallback (or any axis across u) when they are opposed."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = float(np.clip(u @ v, -1.0, 1.0))
    w = np.cross(u, v)
    s = float(np.linalg.norm(w))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        k = None
        if fallback is not None:
            k = _unit(np.asarray(fallback, float) - float(np.asarray(fallback, float) @ u) * u)
        return rotation(k if k is not None else _perp(u), math.pi)
    return rotation(w / s, math.atan2(s, c))


def signed_angle(u, v, axis):
    """Angle (radians, -pi..pi) turning u onto v about axis, both taken across it."""
    axis = np.asarray(axis, float)
    u = np.asarray(u, float) - float(np.asarray(u, float) @ axis) * axis
    v = np.asarray(v, float) - float(np.asarray(v, float) @ axis) * axis
    if np.linalg.norm(u) < EPS or np.linalg.norm(v) < EPS:
        return 0.0
    return math.atan2(float(np.cross(u, v) @ axis), float(u @ v))


def _snap_angle(phi, period):
    """phi brought into (-period/2, period/2] by whole periods."""
    return phi - period * round(phi / period)


def matrix(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _about(Q, p):
    """4x4 of rotation Q about point p."""
    return matrix(Q, p - Q @ p)


# --------------------------------------------------------------------------
# what each element contributes
# --------------------------------------------------------------------------

def direction(e):
    """(unit direction, role, signed) of an element, or None."""
    t = e["type"]
    d = e.get("d")
    if t == "PLANE" and d is not None:
        return np.asarray(d, float), "plane", True
    if t in ("RING", "AXIS") and d is not None:
        return np.asarray(d, float), "axis", t == "RING"
    if t == "LINE" and d is not None:
        return np.asarray(d, float), "line", False
    if t == "POINT" and e.get("hint") is not None:     # an edge's middle: the edge
        return np.asarray(e["hint"], float), "line", False
    return None


def freedom(e):
    """Unit vectors spanning where the element's point may slide."""
    if e.get("pin"):
        return []
    t, d = e["type"], e.get("d")
    if d is None:
        return []
    d = np.asarray(d, float)
    if t in ("LINE", "AXIS"):
        return [d]
    if t == "PLANE":
        u = _perp(d)
        return [u, np.cross(d, u)]
    return []


def _residual(v, span):
    """v less its part in span (vectors, any count)."""
    v = np.asarray(v, float)
    if not span:
        return v
    A = np.array(span, float).T
    Q, s, _ = np.linalg.svd(A, full_matrices=False)
    Q = Q[:, s > 1e-9 * max(float(s.max()), EPS)]
    return v - Q @ (Q.T @ v)


def _face_spin_hints(a, b, R, axis):
    """The extra turn about axis that lines a's face edges up with b's (after
    R), the nearest quarter turn, or 0."""
    if a["type"] != "PLANE" or b["type"] != "PLANE" or a.get("hint") is None or b.get("hint") is None:
        return 0.0
    phi = signed_angle(R @ np.asarray(a["hint"], float), np.asarray(b["hint"], float), axis)
    return _snap_angle(phi, 0.5 * math.pi)


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------

def orient(a, b):
    """(R, relation, spin axis, flip axis) of the direction step (1.)."""
    da, db = direction(a), direction(b)
    if da is None or db is None:
        ax = db[0] if db is not None else (da[0] if da is not None else None)
        return np.eye(3), "none", ax, None
    (ua, ra, sa), (ub, rb, sb) = da, db
    hint_b = b.get("hint")
    if {ra, rb} == {"plane", "line"}:
        if rb == "plane":          # a's line into b's face: its direction off the normal
            v = _unit(ua - float(ua @ ub) * ub)
            if v is None:
                v = _unit(hint_b) if hint_b is not None else _perp(ub)
            R = rotation_between(ua, v)
            return R, "in plane", ub, v
        # a's face onto b's line: a's normal across the line
        v = _unit(ua - float(ua @ ub) * ub)
        if v is None:
            v = _perp(ub)
        R = rotation_between(ua, v)
        return R, "in plane", R @ ua, ub
    if sa and sb:
        target, rel = -ub, "opposed"
    else:
        target, rel = (ub if float(ua @ ub) >= 0 else -ub), "parallel"
    fb = hint_b if hint_b is not None else None
    R = rotation_between(ua, target, fallback=fb)
    extra = _face_spin_hints(a, b, R, ub)
    if extra:
        R = rotation(ub, extra) @ R
    flip_axis = None
    if fb is not None:
        flip_axis = _unit(np.asarray(fb, float) - float(np.asarray(fb, float) @ ub) * ub)
    if flip_axis is None and a.get("hint") is not None:
        h = R @ np.asarray(a["hint"], float)
        flip_axis = _unit(h - float(h @ ub) * ub)
    if flip_axis is None:
        flip_axis = _perp(ub)
    return R, rel, ub, flip_axis


def place(a, b, R):
    """The smallest move t that brings a's point onto b's freedom once the
    part has turned by R about that point (2.)."""
    span = [R @ v for v in freedom(a)] + freedom(b)
    return _residual(np.asarray(b["o"], float) - np.asarray(a["o"], float), span)


def pivot(a, b, R, p):
    """The point where the mated elements meet, nearest p (a's landed point):
    on a's freedom (turned by R) through p and on b's. Flip and spin turn
    about it, so both keep meeting."""
    SA = [R @ v for v in freedom(a)]
    if not SA:
        return p
    A = np.array(SA, float).T
    w = _residual(p - np.asarray(b["o"], float), freedom(b))
    Ab = np.array([_residual(A[:, k], freedom(b)) for k in range(A.shape[1])]).T
    U, sv, Vt = np.linalg.svd(Ab, full_matrices=False)
    keep = sv > 1e-9                        # unit columns: an absolute cut (parallel freedoms give ~0)
    c = Vt[keep].T @ ((U[:, keep].T @ -w) / sv[keep])
    return p + A @ c


def second_spin(a2, b2, axis, p, M4):
    """The turn (radians) about the axis through p that lines a2 (carried by
    the 4x4 M4) up with b2: their directions when both have one across the
    axis, otherwise their points round it."""
    R = M4[:3, :3]
    da, db = direction(a2), direction(b2)
    if da is not None and db is not None:
        ua, ub = R @ da[0], db[0]
        across = lambda v: float(np.linalg.norm(v - float(v @ axis) * axis)) > 1e-6
        if across(ua) and across(ub):
            phi = signed_angle(ua, ub, axis)
            if not (da[2] and db[2]):              # an unsigned one: the nearer sense
                phi = _snap_angle(phi, math.pi)
            return phi
    pa = M4[:3, :3] @ np.asarray(a2["o"], float) + M4[:3, 3]
    return signed_angle(pa - p, np.asarray(b2["o"], float) - p, axis)


def gap_direction(a, b, R, relation):
    """Unit direction a moves for a positive gap: out of b (its normal or
    ring axis), else back along a's own outward direction, else b's direction."""
    db, da = direction(b), direction(a)
    if relation == "in plane":
        if db is not None and db[1] == "plane":
            return db[0]
        return -(R @ da[0])
    if db is not None and db[2]:
        return db[0]
    if da is not None and da[2]:
        return -(R @ da[0])
    if db is not None:
        return db[0]
    if da is not None:
        return R @ da[0]
    return None


def mate(a, b, flip=False, spin=0.0, gap=0.0, second=None):
    """The rigid move mating element a (on the moving part) with b; see the
    module docstring."""
    R0, relation, axis, flip_axis = orient(a, b)
    oa = np.asarray(a["o"], float)
    t0 = place(a, b, R0)
    p = oa + t0                             # where a's point lands
    M = matrix(R0, p - R0 @ oa)             # turned about a's point, then moved
    p = pivot(a, b, R0, p)                  # where the two meet
    if flip and relation != "none":
        M = _about(rotation(flip_axis, math.pi), p) @ M
        if relation == "in plane" and direction(b)[1] != "plane":
            axis = M[:3, :3] @ direction(a)[0]       # a's normal, turned over
    if axis is not None:
        axis = _unit(axis)
    spin_used = float(spin)
    if axis is not None and second is not None:
        spin_used += second_spin(second[0], second[1], axis, p, M)
    if axis is not None and spin_used:
        M = _about(rotation(axis, spin_used), p) @ M
    g = gap_direction(a, b, M[:3, :3], relation) if gap else None
    if g is not None:
        M[:3, 3] += float(gap) * g
    R = M[:3, :3]
    ang = math.degrees(math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    o = np.asarray(a["o"], float)
    return {"M": M, "R": R, "t": M[:3, 3].copy(), "p": p, "axis": axis, "gap_dir": g, "relation": relation,
            "angle": ang, "dist": float(np.linalg.norm(R @ o + M[:3, 3] - o)), "spin": spin_used}


def moved(e, M):
    """Element e carried by the 4x4 M (directions turned, the point moved)."""
    R, t = M[:3, :3], M[:3, 3]
    out = dict(e)
    out["o"] = R @ np.asarray(e["o"], float) + t
    for k in ("d", "hint"):
        if e.get(k) is not None:
            out[k] = R @ np.asarray(e[k], float)
    return out


def mated(a, b, M, tol=1e-9):
    """True when a, carried by M, meets b as mate asks (direction rule and
    position), within tol (lengths relative to 1 + the points' spread)."""
    am = moved(a, M)
    da, db = direction(am), direction(b)
    if da is not None and db is not None:
        (ua, ra, sa), (ub, rb, sb) = da, db
        c = float(ua @ ub)
        if {ra, rb} == {"plane", "line"}:
            if abs(c) > 1e-7:
                return False
        elif abs(abs(c) - 1.0) > 1e-7:
            return False
    scale = 1.0 + float(np.linalg.norm(np.asarray(b["o"], float) - np.asarray(a["o"], float)))
    r = _residual(np.asarray(b["o"], float) - am["o"], freedom(am) + freedom(b))
    return float(np.linalg.norm(r)) <= tol * scale
