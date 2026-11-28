"""Two-element measurements: the shortest distance, the angle, centre to
centre (Measure, and the selection readout when two things are picked).

An element is a point, a curve or surface edge, or a NURBS face, in world
space, recognised as what it is exactly:

* point;
* line (a straight edge or curve: its ends and direction);
* circle (an exact circle or an arc of one, or a fillet's fitted rim: centre,
  axis, radius);
* curve (anything else);
* plane (a flat face: a point and its outward normal);
* cylinder (its axis and radius) and sphere (its centre and radius);
* surface (any other face).

relate(a, b) gives:

* the shortest distance between them and the two points it runs between
  (exact for points, lines and planes; otherwise from the nearest of their
  samples, refined by projecting back and forth onto each, exact to round-off
  on smooth elements);
* the angle between them (lines, planes, axes): two lines meeting at an end
  or two faces meeting along an edge give the angle between them there (a
  box's corner 90°, a 45° chamfer and its face 135°), otherwise the acute
  angle between the two directions;
* parallel ones their distance apart (two faces' gap, two axes', a line's to
  a plane);
* centres: circles, spheres and cylinders measure from their centre or axis
  (hole to hole, a hole's axis to a face).

Nothing in the scene changes.
"""

import math

import numpy as np

from . import store

PAR = 1e-7                  # sin of the angle below which two directions are parallel
TOUCH = 1e-6                # times the size: two elements this close touch
MAX_SAMPLES = 48            # samples of an element tried as starting points
_world = {}                 # object name -> (key, world Brep copy)


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-300 else v * 0.0


def _thin(W, k=MAX_SAMPLES):
    W = np.asarray(W, float)
    if len(W) <= k:
        return W
    return W[np.linspace(0, len(W) - 1, k).round().astype(int)]


# --------------------------------------------------------------------------
# Elements
# --------------------------------------------------------------------------

def point_el(P, key=None):
    P = np.asarray(P, float)
    return {"type": "point", "what": "point", "key": key or ("pt",) + tuple(np.round(P, 12)), "P": P,
            "samples": P[None, :], "size": 0.0, "closest": lambda Q: P}


def _seg_closest(A, B):
    AB = B - A
    L2 = float(AB @ AB)

    def f(Q):
        if L2 <= 1e-300:
            return A
        t = min(max(float((np.asarray(Q, float) - A) @ AB) / L2, 0.0), 1.0)
        return A + t * AB
    return f


def curve_el(kc, key=None, what="edge"):
    """An element on a world-space kernel curve (an edge or a curve spline)."""
    from . import linesolve
    from .measure import curve_radius
    from .measureinfo import _arc_of, curve_length, fit_circle
    c = linesolve.kernel_curve(kc, store._k()[0])
    W = c.W
    size = float(np.ptp(W, axis=0).max())
    el = {"what": what, "key": key, "W": W, "size": size, "P": W[len(W) // 2], "samples": _thin(W)}
    if c.straight:
        A, B = W[0].copy(), W[-1].copy()
        el.update(type="line", A=A, B=B, d=_unit(B - A), closest=_seg_closest(A, B), length=float(np.linalg.norm(B - A)))
        return el
    el["closest"] = lambda Q: c.at(linesolve.nearest_t(c, Q))[1]
    try:
        L = curve_length(kc)
        arc = _arc_of(kc, L)
    except Exception:
        L, arc = None, None
    el["length"] = L
    if arc is not None:
        m = curve_radius(c, W[0])
        if m is not None and m["source"] == "circle":
            C, n = m["centre"], m["normal"]
        else:
            C, _, n, _ = fit_circle(W)
        el.update(type="circle", C=np.asarray(C, float), n=_unit(n), r=float(arc[0]), arc=arc)
        return el
    el["type"] = "curve"
    return el


def world_brep(obj):
    """obj's Brep moved into world space (a copy, cached on its geometry and transform)."""
    from . import geom
    from .measureinfo import _world_brep
    M = np.array(geom.world_matrix(obj), float)
    k = (obj.get(store.PROP_REV, 0), tuple(np.round(M, 12).ravel()))
    hit = _world.get(obj.name)
    if hit is not None and hit[0] == k:
        return hit[1]
    b = _world_brep(obj, M)
    _world[obj.name] = (k, b)
    return b


def edge_el(obj, i):
    from . import features
    return curve_el(features.input_curve(obj, f"E{i}"), key=("E", obj.name, i), what="edge")


def spline_el(obj, ref):
    from . import features
    return curve_el(features.input_curve(obj, ref), key=("S", obj.name, ref), what="curve")


def _lines_point(S, N):
    """The point nearest all the lines (S_i along N_i), least squares, and the rank."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for s, n in zip(S, N):
        P = np.eye(3) - np.outer(n, n)
        A += P
        b += P @ s
    x, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    return x, rank


def face_el(obj, i, n=9):
    """An element on face i of the NURBS object obj, in world space."""
    from .measureinfo import face_radius
    F = world_brep(obj).faces()[i]
    (u0, v0), (u1, v1) = F.uv_domain()           # ((u min, v min), (u max, v max))
    g = np.linspace(0.03, 0.97, n)
    S, N = [], []
    for u in u0 + g * (u1 - u0):
        for v in v0 + g * (v1 - v0):
            if F.classify_uv(float(u), float(v)) == "outside":
                continue
            S.append(F.point_at_uv(float(u), float(v)))
            N.append(F.outward_normal_at_uv(float(u), float(v)))
    if not S:                                       # a sliver between the grid lines
        S = [F.closest_point(tuple(float(x) for x in F.centroid()))[0]]
    S = np.array(S, float).reshape(-1, 3)
    N = np.array([_unit(x) for x in N], float).reshape(-1, 3)
    size = float(np.ptp(S, axis=0).max()) if len(S) > 1 else 0.0
    el = {"what": "face", "key": ("F", obj.name, i), "face": F, "samples": _thin(S, 81), "size": size,
          "P": S[len(S) // 2] if len(S) else np.zeros(3), "S": S, "N": N,
          "closest": lambda Q: np.array(F.closest_point(tuple(float(x) for x in Q))[0], float)}
    if len(N) >= 3:
        nm = _unit(N.mean(axis=0))
        O = S.mean(axis=0)
        if float(np.abs(N - nm).max()) <= 1e-7 and float(np.abs((S - O) @ nm).max()) <= 1e-7 * max(size, 1e-300):
            el.update(type="plane", O=O, n=nm)
            return el
        try:
            fr = face_radius(F)
        except Exception:
            fr = None
        if fr is not None and fr[1] in ("cylinder", "sphere"):
            r = fr[0]
            C, rank = _lines_point(S, N)
            if fr[1] == "sphere":
                el.update(type="sphere", C=C, r=r)
            else:
                a = np.linalg.svd(N, full_matrices=False)[2][-1]      # square to every normal
                C = C - ((C - S.mean(axis=0)) @ a) * a                # the axis point beside the face's middle
                el.update(type="cylinder", C=C, n=_unit(a), r=r)
            return el
    el["type"] = "surface"
    return el


def outline(e, k=24):
    """Polylines that show an element (a curve itself, a face's boundary)."""
    if e["type"] == "point":
        return []
    if "W" in e:
        return [e["W"]]
    F = e.get("face")
    if F is None:
        return []
    out = []
    try:
        for ed in F.outer_loop_edges():
            t0, t1 = ed.parameter_range()
            out.append(np.array([ed.point_at(t0 + (t1 - t0) * j / k) for j in range(k + 1)], float))
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------

def _seg_seg(A, B, C, D):
    """Closest points of segments AB and CD."""
    d1, d2, r = B - A, D - C, A - C
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= 1e-300 and e <= 1e-300:
        return A, C
    if a <= 1e-300:
        return A, C + min(max(f / e, 0.0), 1.0) * d2
    c = float(d1 @ r)
    if e <= 1e-300:
        return A + min(max(-c / a, 0.0), 1.0) * d1, C
    b = float(d1 @ d2)
    den = a * e - b * b
    s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-300 * a * e else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, min(max(-c / a, 0.0), 1.0)
    elif t > 1.0:
        t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return A + s * d1, C + t * d2


def _refine(a, b, P, tol, it=100):
    """Project back and forth from P on a: (point on a, point on b)."""
    Q = b["closest"](P)
    for _ in range(it):
        P2 = a["closest"](Q)
        Q2 = b["closest"](P2)
        moved = float(np.linalg.norm(P2 - P)) + float(np.linalg.norm(Q2 - Q))
        P, Q = P2, Q2
        if moved <= tol:
            break
    return P, Q


def closest_pair(a, b):
    """(point on a, point on b, distance): the shortest distance between them."""
    ta, tb = a["type"], b["type"]
    if ta == "point" and tb == "point":
        return a["P"], b["P"], float(np.linalg.norm(b["P"] - a["P"]))
    if ta == "line" and tb == "line":
        P, Q = _seg_seg(a["A"], a["B"], b["A"], b["B"])
        return P, Q, float(np.linalg.norm(Q - P))
    if ta == "point":
        Q = b["closest"](a["P"])
        return a["P"], Q, float(np.linalg.norm(Q - a["P"]))
    if tb == "point":
        P = a["closest"](b["P"])
        return P, b["P"], float(np.linalg.norm(b["P"] - P))
    size = max(a["size"], b["size"], 1e-300)
    tol = 1e-13 * size
    starts = []
    for x, y, flip in ((a, b, False), (b, a, True)):
        S = x["samples"]
        d = [float(np.linalg.norm(y["closest"](s) - s)) for s in S]
        for k in np.argsort(d)[:3]:
            starts.append((S[k], flip))
    best = None
    for s, flip in starts:
        if flip:
            Q, P = _refine(b, a, s, tol)
        else:
            P, Q = _refine(a, b, s, tol)
        d = float(np.linalg.norm(Q - P))
        if best is None or d < best[2]:
            best = (P, Q, d)
    return best


def _axis(e):
    """(point, unit direction) of an element's line or axis, or None."""
    t = e["type"]
    if t == "line":
        return e["A"], e["d"]
    if t in ("circle", "cylinder"):
        return e["C"], e["n"]
    return None


def _line_point_dist(O, d, P):
    v = np.asarray(P, float) - O
    return float(np.linalg.norm(v - (v @ d) * d))


def _lines_dist(O1, d1, O2, d2):
    n = np.cross(d1, d2)
    nn = float(np.linalg.norm(n))
    if nn <= PAR:
        return _line_point_dist(O1, d1, O2)
    return abs(float((O2 - O1) @ n)) / nn


def _acute(u, v):
    c = abs(float(np.clip(np.asarray(u) @ np.asarray(v), -1.0, 1.0)))
    return math.degrees(math.acos(min(c, 1.0)))


def _parallel(u, v):
    return float(np.linalg.norm(np.cross(u, v))) <= PAR


def _shared_end(a, b, tol):
    """(the common end, the other end of a, of b) of two lines meeting at an end, or None."""
    for P, P2 in ((a["A"], a["B"]), (a["B"], a["A"])):
        for Q, Q2 in ((b["A"], b["B"]), (b["B"], b["A"])):
            if float(np.linalg.norm(P - Q)) <= tol:
                return P, P2, Q2
    return None


def _angle(a, b, touching, tol):
    """(angle in degrees, how) between two elements with a direction, or None."""
    ta, tb = a["type"], b["type"]
    if ta == "line" and tb == "line":
        sh = _shared_end(a, b, tol)
        if sh is not None:
            V, P, Q = sh
            u, v = _unit(P - V), _unit(Q - V)
            return math.degrees(math.acos(float(np.clip(u @ v, -1.0, 1.0)))), "corner"
        return _acute(a["d"], b["d"]), "lines"
    if ta == "plane" and tb == "plane":
        if touching and not _parallel(a["n"], b["n"]):
            e = _unit(np.cross(a["n"], b["n"]))             # along the edge they meet at
            Nm = np.array([a["n"], b["n"]])                  # a point on that line, near a's middle
            M = a["P"] + np.linalg.lstsq(Nm, np.array([a["n"] @ (a["O"] - a["P"]), b["n"] @ (b["O"] - a["P"])]),
                                         rcond=None)[0]
            u = a["S"].mean(axis=0) - M
            v = b["S"].mean(axis=0) - M
            u, v = _unit(u - (u @ e) * e), _unit(v - (v @ e) * e)
            if float(np.linalg.norm(u)) > 0 and float(np.linalg.norm(v)) > 0:
                return math.degrees(math.acos(float(np.clip(u @ v, -1.0, 1.0)))), "faces"
        return _acute(a["n"], b["n"]), "planes"
    if {ta, tb} == {"circle", "plane"} or (ta == "circle" and tb == "circle"):
        return _acute(a["n"], b["n"]), "planes"              # a circle's plane against a face's / another's
    dirs = []
    for e in (a, b):
        if e["type"] == "plane":
            dirs.append(("normal", e["n"]))
        elif _axis(e) is not None:
            dirs.append(("dir", _axis(e)[1]))
        else:
            return None
    (ka, u), (kb, v) = dirs
    if ka == kb:                                         # two axes / an axis and a line
        return _acute(u, v), "axes"
    return 90.0 - _acute(u, v), "to plane"               # a line or an axis against a plane


def _centre_of(e):
    return e["C"] if e["type"] in ("circle", "sphere") else None


def relate(a, b):
    """What two elements measure against each other:
    {"min": (P, Q, d), "delta": Q - P (two points), "angle": (deg, how) or None,
     "parallel": (distance, what) or None, "centres": (C1, C2, distance, what) or None}."""
    P, Q, d = closest_pair(a, b)
    size = max(a["size"], b["size"], d, 1e-300)
    touching = d <= TOUCH * size
    out = {"a": a, "b": b, "min": (P, Q, d), "delta": None, "angle": None, "parallel": None, "centres": None,
           "touching": bool(touching)}
    if a["type"] == "point" and b["type"] == "point":
        out["delta"] = Q - P
        return out
    ang = _angle(a, b, touching, 1e-7 * size)
    ta, tb = a["type"], b["type"]
    # parallel: their distance apart
    if ta == "plane" and tb == "plane" and _parallel(a["n"], b["n"]):
        g = abs(float((b["O"] - a["O"]) @ a["n"]))
        out["parallel"] = (g, "faces")
        ang = None
    elif {ta, tb} <= {"line", "circle", "cylinder"} and _parallel(_axis(a)[1], _axis(b)[1]):
        (O1, d1), (O2, d2) = _axis(a), _axis(b)
        out["parallel"] = (_lines_dist(O1, d1, O2, d2), "axes" if "line" not in (ta, tb) else "lines")
        ang = None
    elif "plane" in (ta, tb):
        pl, other = (a, b) if ta == "plane" else (b, a)
        ax = _axis(other)
        if other["type"] == "circle":
            if ang is not None and ang[0] <= math.degrees(PAR):
                ang = None                                   # parallel: the centre's distance says it
        elif ax is not None and abs(float(ax[1] @ pl["n"])) <= PAR:
            out["parallel"] = (abs(float((ax[0] - pl["O"]) @ pl["n"])),
                               "line to face" if other["type"] == "line" else "axis to face")
            ang = None
    out["angle"] = ang
    # centres: circles and spheres from the centre, cylinders from the axis
    ca, cb = _centre_of(a), _centre_of(b)
    if ca is not None and cb is not None:
        out["centres"] = (ca, cb, float(np.linalg.norm(cb - ca)), "centres")
    elif ca is not None or cb is not None:
        C, other, flip = (ca, b, False) if ca is not None else (cb, a, True)
        t = other["type"]
        R = None
        if t == "point":
            R = other["P"]
        elif t == "plane":
            R = C - float((C - other["O"]) @ other["n"]) * other["n"]
        elif t in ("line", "cylinder"):
            O, dd = _axis(other)
            R = O + float((C - O) @ dd) * dd
        if R is not None:
            out["centres"] = ((R, C) if flip else (C, R)) + (float(np.linalg.norm(R - C)), "centre")
    elif "cylinder" in (ta, tb) and out["parallel"] is None:
        cyl, other = (a, b) if ta == "cylinder" else (b, a)
        if other["type"] == "point":
            O, dd = _axis(cyl)
            R = O + float((other["P"] - O) @ dd) * dd
            out["centres"] = (R, other["P"], float(np.linalg.norm(other["P"] - R)), "axis")
    return out


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

ANGLE_WHAT = {"corner": "corner", "faces": "between the faces", "lines": "", "planes": "", "axes": "", "to plane": ""}
PAR_WHAT = {"faces": "Parallel faces", "axes": "Axes", "lines": "Parallel lines", "axis to face": "Axis to face",
            "line to face": "Line to face"}


def deg(x):
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{s}°"


def rows(context, r, axis=None):
    """[(label, value)] for a relation. axis: 'X' / 'Y' / 'Z' shows only that
    component of a distance between points."""
    from .measureinfo import fmt
    out = []
    P, Q, d = r["min"]
    if r["delta"] is not None:
        dx, dy, dz = r["delta"]
        if axis in ("X", "Y", "Z"):
            v = {"X": dx, "Y": dy, "Z": dz}[axis]
            out.append((f"Δ{axis}", fmt(context, abs(v))))
        else:
            out.append(("Distance", f"{fmt(context, d)}   ΔX {fmt(context, dx)}  ΔY {fmt(context, dy)}  "
                                    f"ΔZ {fmt(context, dz)}"))
        return out
    shown = []

    def same(x):
        return any(abs(x - y) <= 1e-9 * max(abs(y), d, 1e-300) for y in shown)
    if r["centres"] is not None:
        C1, C2, dc, what = r["centres"]
        lab = {"centres": "Centres", "centre": "From the centre", "axis": "From the axis"}[what]
        out.append((lab, fmt(context, dc)))
        shown.append(dc)
    if r["parallel"] is not None:
        g, what = r["parallel"]
        if not same(g):
            out.append((PAR_WHAT[what], fmt(context, g) if g > 1e-12 * max(d, 1.0) else
                        ("coaxial" if what == "axes" else "in line" if what == "lines" else "flush")))
            shown.append(g)
    if r["angle"] is not None:
        a, how = r["angle"]
        extra = ANGLE_WHAT.get(how, "")
        out.append(("Angle", deg(a) + (f"  ({extra})" if extra else "")))
    if r["touching"]:
        if not (r["angle"] is not None and r["angle"][1] in ("faces", "corner")):   # (meeting there goes without saying)
            out.append(("Shortest", "0 (touching)"))
    elif not same(d):
        out.append(("Shortest", fmt(context, d)))
    return out


def text(context, r, axis=None):
    return "  ·  ".join(f"{a} {b}" for a, b in rows(context, r, axis))
