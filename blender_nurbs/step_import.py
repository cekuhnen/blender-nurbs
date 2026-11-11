"""STEP (ISO 10303-21: AP203 / AP214 / AP242) import of B-rep models.

Reads MANIFOLD_SOLID_BREP, BREP_WITH_VOIDS and SHELL_BASED_SURFACE_MODEL
bodies and turns every ADVANCED_FACE into an exact trimmed NURBS face:

* Surfaces -- B-spline surfaces (rational, unclamped / periodic knots
  included) as they are. Analytic surfaces become exact rational NURBS sized
  to the face: PLANE, CYLINDRICAL_, CONICAL_, SPHERICAL_, TOROIDAL_SURFACE,
  SURFACE_OF_REVOLUTION, SURFACE_OF_LINEAR_EXTRUSION; OFFSET_SURFACE via the
  kernel's offset; RECTANGULAR_TRIMMED_SURFACE via its basis.
* Edges -- LINE, CIRCLE, ELLIPSE, B-spline (rational), POLYLINE, and the
  3D curve of SURFACE_CURVE / SEAM_CURVE / TRIMMED_CURVE, cut exactly
  between the edge's vertices (end points snapped onto them, so the two
  faces of an edge get the same curve).
* Faces are trimmed with their 3D edge loops by the kernel, the faces of a
  body are merged and stitched (solids come back as closed bodies).

Units (mm, cm, m, inch, foot...) are converted to Blender units, and
assembly placements (REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION,
MAPPED_ITEM) are applied as object transforms. Each body becomes one NURBS
object named after its product.
"""

import math
import os
import re
import time

import numpy as np

from . import curvejoin, curvetrim, store
from .extend import _deboor

# --------------------------------------------------------------------------
# Part 21 parser
# --------------------------------------------------------------------------

_ENTITY = re.compile(r"#(\d+)\s*=\s*((?:[^;']|'(?:[^']|'')*')*);", re.S)
_TOKEN = re.compile(r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<ref>#\d+)|(?P<enum>\.[A-Z_0-9]+\.)|"
                    r"(?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|(?P<id>[A-Z_][A-Z_0-9]*)|"
                    r"(?P<p>[(),$*]))")
_POINT = re.compile(r"^\s*CARTESIAN_POINT\s*\(\s*'(?:[^']|'')*'\s*,\s*\(([^)]*)\)\s*\)\s*$", re.S)


class Ref(int):
    pass


class Enum(str):
    pass


class Typed:
    def __init__(self, name, args):
        self.name, self.args = name, args


def _parse(text):
    toks = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if m is None:
            if text[pos:].strip() == "":
                break
            raise ValueError(f"STEP syntax near: {text[pos:pos + 40]!r}")
        pos = m.end()
        k = m.lastgroup
        v = m.group(k)
        toks.append((k, v))
    i = 0

    def value():
        nonlocal i
        k, v = toks[i]
        i += 1
        if k == "p":
            if v == "(":
                out = []
                if toks[i] == ("p", ")"):
                    i += 1
                    return out
                while True:
                    out.append(value())
                    k2, v2 = toks[i]
                    i += 1
                    if v2 == ")":
                        return out
            if v in "$*":
                return None
            raise ValueError("unexpected " + v)
        if k == "ref":
            return Ref(int(v[1:]))
        if k == "num":
            return float(v)
        if k == "str":
            return v[1:-1].replace("''", "'")
        if k == "enum":
            return Enum(v[1:-1])
        if k == "id":
            if i < len(toks) and toks[i] == ("p", "("):
                args = value()
                return Typed(v, args)
            return Typed(v, [])
        raise ValueError(v)

    if toks and toks[0] == ("p", "("):          # complex entity: (A(..) B(..) ...)
        i = 1
        parts = {}
        while toks[i] != ("p", ")"):
            t = value()
            parts[t.name] = t.args
        return parts
    return value()


class StepFile:
    def __init__(self, text):
        d0 = text.find("DATA;")
        body = text[d0 + 5:] if d0 >= 0 else text
        self.raw = {int(m.group(1)): m.group(2) for m in _ENTITY.finditer(body)}
        self._cache = {}
        self._types = None

    def __getitem__(self, ref):
        ref = int(ref)
        e = self._cache.get(ref)
        if e is None:
            e = _parse(self.raw[ref])
            self._cache[ref] = e
        return e

    def type(self, ref):
        e = self[ref]
        return e.name if isinstance(e, Typed) else "COMPLEX"

    def args(self, ref, name=None):
        e = self[ref]
        if isinstance(e, Typed):
            return e.args
        return e.get(name) if name else None

    def has(self, ref, name):
        e = self[ref]
        return (e.name == name) if isinstance(e, Typed) else (name in e)

    def of_type(self, *names):
        """Refs whose (simple) entity type is one of names -- found by text,
        parsed lazily."""
        out = []
        pat = re.compile(r"^\s*\(?\s*(?:[A-Z_0-9]+\s*\([^;]*?\)\s*)*?(" + "|".join(names) + r")\s*\(")
        for k, raw in self.raw.items():
            head = raw.lstrip()[:60]
            if any(head.startswith(nm) for nm in names) or (head.startswith("(") and any(nm in raw for nm in names)):
                e = self[k]
                if isinstance(e, Typed) and e.name in names:
                    out.append(k)
                elif isinstance(e, dict) and any(nm in e for nm in names):
                    out.append(k)
        return out

    def point(self, ref):
        raw = self.raw[int(ref)]
        m = _POINT.match(raw)
        if m:
            return np.array([float(x) for x in m.group(1).split(",")], float)
        return np.array(self.args(ref)[1], float)

    def direction(self, ref):
        d = np.array(self.args(ref)[1], float)
        return d / np.linalg.norm(d)


# --------------------------------------------------------------------------
# Units, placements
# --------------------------------------------------------------------------

def length_factor(sf, ctx_ref):
    """Metres per file length unit, and radians per file angle unit, of a
    representation context. (0.001, 1.0) for millimetre / radian files."""
    L, A = 0.001, 1.0
    e = sf[ctx_ref]
    units = e.get("GLOBAL_UNIT_ASSIGNED_CONTEXT") if isinstance(e, dict) else None
    for u in (units[0] if units else []):
        ue = sf[u]
        if not isinstance(ue, dict):
            continue
        val = _unit_value(sf, u)
        if "LENGTH_UNIT" in ue and val is not None:
            L = val
        elif "PLANE_ANGLE_UNIT" in ue and val is not None:
            A = val
    return L, A


_SI = {"EXA": 1e18, "PETA": 1e15, "TERA": 1e12, "GIGA": 1e9, "MEGA": 1e6, "KILO": 1e3, "HECTO": 1e2, "DECA": 1e1,
       "DECI": 1e-1, "CENTI": 1e-2, "MILLI": 1e-3, "MICRO": 1e-6, "NANO": 1e-9, "PICO": 1e-12}


def _unit_value(sf, u):
    ue = sf[u]
    if not isinstance(ue, dict):
        return None
    if "SI_UNIT" in ue:
        pre, name = ue["SI_UNIT"]
        return _SI.get(str(pre), 1.0) if pre else 1.0
    if "CONVERSION_BASED_UNIT" in ue:
        _, mw = ue["CONVERSION_BASED_UNIT"]
        a = sf.args(mw)
        meas = a[0]
        v = float(meas.args[0]) if isinstance(meas, Typed) else float(meas)
        base = _unit_value(sf, a[1])
        return v * (base if base is not None else 1.0)
    return None


def uncertainty(sf, ctx_ref):
    e = sf[ctx_ref]
    if isinstance(e, dict) and "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT" in e:
        for u in e["GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT"][0]:
            a = sf.args(u)
            if a and isinstance(a[0], Typed):
                return float(a[0].args[0])
    return None


def placement(sf, ref, f=1.0):
    """(origin, x, y, z) of an AXIS2_PLACEMENT_3D (lengths scaled by f)."""
    a = sf.args(ref)
    o = sf.point(a[1]) * f
    z = sf.direction(a[2]) if a[2] is not None else np.array([0.0, 0, 1])
    x = sf.direction(a[3]) if len(a) > 3 and a[3] is not None else None
    if x is None:
        x = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return o, x, y, z


def placement_matrix(sf, ref, f=1.0):
    o, x, y, z = placement(sf, ref, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, o
    return M


# --------------------------------------------------------------------------
# Bezier spans (homogeneous rows) helpers
# --------------------------------------------------------------------------

def bspline_spans(Pw, U, p):
    """[(span, t0, t1)] Bezier spans of a B-spline with any knot vector
    (clamped or not): each span's CVs are blossoms of its polynomial piece."""
    Pw = [np.asarray(x, float) for x in Pw]
    n = len(Pw) - 1
    out = []
    for k in range(p, n + 1):
        a, b = U[k], U[k + 1]
        if b - a <= 1e-14 * max(1.0, abs(b)):
            continue
        cv = [_deboor(Pw, U, p, k, [a] * (p - i) + [b] * i) for i in range(p + 1)]
        out.append((np.array(cv), float(a), float(b)))
    return out


def arc_spans(o, x, y, r1, r2, a0, a1):
    """Rational quadratic spans of an ellipse arc from angle a0 to a1 (a1 > a0)."""
    total = a1 - a0
    nseg = max(1, int(math.ceil(total / (math.pi / 2) - 1e-9)))
    out = []
    pt = lambda t: o + r1 * math.cos(t) * x + r2 * math.sin(t) * y
    for k in range(nseg):
        a, b = a0 + total * k / nseg, a0 + total * (k + 1) / nseg
        h = 0.5 * (b - a)
        w = math.cos(h)
        m = o + (r1 * math.cos(a + h) * x + r2 * math.sin(a + h) * y) / w
        out.append(np.array([[*pt(a), 1.0], [*(m * w), w], [*pt(b), 1.0]]))
    return out


def _cart(s):
    return s[:, :3] / s[:, 3:4]


def sub_spans(spans, ta, tb):
    """Exact piece [ta, tb] (ta < tb) of [(span, t0, t1)]."""
    out = []
    for s, t0, t1 in spans:
        if t1 <= ta + 1e-12 or t0 >= tb - 1e-12:
            continue
        a = max(0.0, (ta - t0) / (t1 - t0))
        b = min(1.0, (tb - t0) / (t1 - t0))
        seg = s
        if b < 1.0 - 1e-12:
            seg, _ = curvetrim.split_bezier(seg, b)
        if a > 1e-12:
            _, seg = curvetrim.split_bezier(seg, a / b if b > 0 else 0.0)
        out.append(seg)
    return out


def closest_param(spans, P):
    best = None
    for s, t0, t1 in spans:
        ts = np.linspace(0, 1, 17)
        Q = curvejoin.eval_span(s, ts)
        d = np.linalg.norm(Q - P, axis=1)
        k = int(np.argmin(d))
        if best is None or d[k] < best[0]:
            best = (d[k], s, t0, t1, ts[k])
    _, s, t0, t1, t = best
    for _ in range(30):
        C, dC = curvetrim.eval_d(s, t)
        h = 1e-6
        C2, dC2 = curvetrim.eval_d(s, min(1.0, t + h))
        dd = (dC2 - dC) / h
        r = C - P
        den = float(dC @ dC + r @ dd)
        if abs(den) < 1e-20:
            break
        t2 = float(np.clip(t - float(r @ dC) / den, 0.0, 1.0))
        if abs(t2 - t) < 1e-14:
            t = t2
            break
        t = t2
    return t0 + t * (t1 - t0)


def spans_to_kernel(ext, spans):
    """Kernel curve holding these spans exactly (Bezier knots)."""
    q = max(len(s) - 1 for s in spans)
    spans = [curvejoin.elevate(s, q) for s in spans]
    for k in range(1, len(spans)):
        a, b = spans[k - 1][-1, 3], spans[k][0, 3]
        if b != 0:
            spans[k] = spans[k] * (a / b)
    rows = [spans[0][0]] + [r for s in spans for r in s[1:]]
    rows = np.array(rows)
    W = rows[:, 3]
    P = rows[:, :3] / W[:, None]
    n = len(spans)
    U = [0.0] * (q + 1) + [float(k) for k in range(1, n) for _ in range(q)] + [float(n)] * (q + 1)
    U = [u / n for u in U]
    rational = np.abs(W - W[0]).max() > 1e-12 or abs(W[0] - 1) > 1e-12
    return ext.curve_from_cvs(q, [tuple(map(float, p)) for p in P], U, [float(w) for w in W] if rational else None)


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------

class Importer:
    def __init__(self, sf, f=1.0, angle=1.0, tol=1e-6):
        from .kernel import load
        self.sm = load()
        self.ext = self.sm.blender_ext
        self.sf, self.f, self.ang, self.tol = sf, f, angle, tol
        self._edge = {}
        self.warnings = []

    # -- generic B-spline curve from a STEP entity -------------------------
    def bspline_curve(self, ref):
        sf = self.sf
        e = sf[ref]
        if isinstance(e, Typed):
            a = e.args
            deg, cps = int(a[1]), a[2]
            mults, knots = a[6], a[7]
            W = None
        else:
            b = e["B_SPLINE_CURVE"]           # (degree, control points, form, closed, self_intersect)
            deg, cps = int(b[0]), b[1]
            k = e["B_SPLINE_CURVE_WITH_KNOTS"]
            mults, knots = k[0], k[1]
            W = e["RATIONAL_B_SPLINE_CURVE"][0] if "RATIONAL_B_SPLINE_CURVE" in e else None
        P = np.array([sf.point(c) for c in cps]) * self.f
        W = np.ones(len(P)) if W is None else np.array(W, float)
        U = [float(u) for u, m in zip(knots, mults) for _ in range(int(m))]
        Pw = np.hstack([P * W[:, None], W[:, None]])
        return bspline_spans(Pw, U, deg)

    def curve_spans(self, ref, pts=None):
        """[(span, t0, t1)] of a bounded curve (infinite LINEs bounded by pts)."""
        sf = self.sf
        t = sf.type(ref)
        if t == "COMPLEX" and (sf.has(ref, "B_SPLINE_CURVE_WITH_KNOTS")):
            return self.bspline_curve(ref)
        a = sf.args(ref)
        if t == "B_SPLINE_CURVE_WITH_KNOTS":
            return self.bspline_curve(ref)
        if t in ("SURFACE_CURVE", "SEAM_CURVE", "INTERSECTION_CURVE"):
            return self.curve_spans(a[1], pts)
        if t == "TRIMMED_CURVE":
            return self.curve_spans(a[1], pts)
        if t == "POLYLINE":
            P = [sf.point(c) * self.f for c in a[1]]
            return [(np.array([[*P[k], 1.0], [*P[k + 1], 1.0]]), float(k), float(k + 1)) for k in range(len(P) - 1)]
        if t == "LINE":
            o = sf.point(a[1]) * self.f
            va = sf.args(a[2])
            d = sf.direction(va[1])
            ss = [float((p - o) @ d) for p in (pts if pts is not None and len(pts) else [o, o + d])]
            s0, s1 = min(ss), max(ss)
            pad = 0.05 * (s1 - s0) + 1e-6
            s0, s1 = s0 - pad, s1 + pad
            return [(np.array([[*(o + s0 * d), 1.0], [*(o + s1 * d), 1.0]]), s0, s1)]
        if t in ("CIRCLE", "ELLIPSE"):
            o, x, y, z = placement(sf, a[1], self.f)
            r1 = float(a[2]) * self.f
            r2 = float(a[3]) * self.f if t == "ELLIPSE" else r1
            spans = arc_spans(o, x, y, r1, r2, 0.0, 2 * math.pi)
            n = len(spans)
            return [(s, 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n) for k, s in enumerate(spans)]
        raise ValueError(f"unsupported curve {t}")

    def _ellipse_angle(self, o, x, y, r1, r2, P):
        d = P - o
        return math.atan2(float(d @ y) / r2, float(d @ x) / r1)

    def edge_spans(self, edge_ref):
        """Homogeneous Bezier spans of an EDGE_CURVE from its start vertex to its end vertex."""
        hit = self._edge.get(edge_ref)
        if hit is not None:
            return hit
        sf = self.sf
        a = sf.args(edge_ref)
        Ps = sf.point(sf.args(a[1])[1]) * self.f
        Pe = sf.point(sf.args(a[2])[1]) * self.f
        geo, same = a[3], (a[4] != "F")
        g = geo
        while sf.type(g) in ("SURFACE_CURVE", "SEAM_CURVE", "INTERSECTION_CURVE", "TRIMMED_CURVE"):
            g = sf.args(g)[1]
        t = sf.type(g)
        closed_v = np.linalg.norm(Ps - Pe) <= 1e-9 * (1 + np.linalg.norm(Ps))
        if t == "LINE":
            if closed_v:
                raise ValueError("zero-length line edge")
            spans = [np.array([[*Ps, 1.0], [*Pe, 1.0]])]
        elif t in ("CIRCLE", "ELLIPSE"):
            ga = sf.args(g)
            o, x, y, z = placement(sf, ga[1], self.f)
            r1 = float(ga[2]) * self.f
            r2 = float(ga[3]) * self.f if t == "ELLIPSE" else r1
            a0 = self._ellipse_angle(o, x, y, r1, r2, Ps)
            a1 = self._ellipse_angle(o, x, y, r1, r2, Pe)
            if same:
                while a1 <= a0 + 1e-12:
                    a1 += 2 * math.pi
                if closed_v:
                    a1 = a0 + 2 * math.pi
                spans = arc_spans(o, x, y, r1, r2, a0, a1)
            else:
                while a0 <= a1 + 1e-12:
                    a0 += 2 * math.pi
                if closed_v:
                    a0 = a1 + 2 * math.pi
                spans = curvejoin._reverse(arc_spans(o, x, y, r1, r2, a1, a0))
        else:
            cs = self.curve_spans(g, [Ps, Pe])
            tlo, thi = cs[0][1], cs[-1][2]
            curve_closed = np.linalg.norm(_cart(cs[0][0])[0] - _cart(cs[-1][0])[-1]) <= 1e-7 * (1 + np.linalg.norm(Ps))
            ta, tb = closest_param(cs, Ps), closest_param(cs, Pe)
            if not same:
                ta, tb = tb, ta
            if closed_v and curve_closed:
                spans = sub_spans(cs, ta, thi) + sub_spans(cs, tlo, ta) if ta > tlo + 1e-12 else [c[0] for c in cs]
            elif tb > ta + 1e-12:
                spans = sub_spans(cs, ta, tb)
            elif curve_closed:
                spans = sub_spans(cs, ta, thi) + sub_spans(cs, tlo, tb)
            else:
                raise ValueError("edge range not found on its curve")
            spans = [s for s in spans if len(s)]
            if not same:
                spans = curvejoin._reverse(spans)
        spans = [np.array(s, float) for s in spans]
        # snap the ends onto the vertices (both faces of the edge get the same ends)
        spans[0][0, :3] = Ps * spans[0][0, 3]
        spans[-1][-1, :3] = Pe * spans[-1][-1, 3]
        self._edge[edge_ref] = spans
        return spans

    # -- faces -----------------------------------------------------------
    def face_loops(self, face_ref):
        """[(is_outer, [spans of each oriented edge, in loop order])], seam edge refs."""
        sf = self.sf
        bounds = sf.args(face_ref)[1]
        loops, seen, seams = [], {}, set()
        for b in bounds:
            bt = sf.type(b)
            ba = sf.args(b)
            loop = ba[1]
            orient = ba[2] != "F"
            lt = sf.type(loop)
            if lt == "VERTEX_LOOP":
                continue
            edges = []
            if lt == "POLY_LOOP":
                P = [sf.point(c) * self.f for c in sf.args(loop)[1]]
                for k in range(len(P)):
                    edges.append([np.array([[*P[k], 1.0], [*P[(k + 1) % len(P)], 1.0]])])
            else:
                for oe in sf.args(loop)[1]:
                    oa = sf.args(oe)
                    ec = oa[3]
                    seen[ec] = seen.get(ec, 0) + 1
                    if seen[ec] > 1:
                        seams.add(ec)
                    sp = self.edge_spans(ec)
                    edges.append(sp if oa[4] != "F" else curvejoin._reverse(sp))
            if not orient:
                edges = [curvejoin._reverse(e) for e in reversed(edges)]
            loops.append((bt == "FACE_OUTER_BOUND", edges))
        if loops and not any(o for o, _ in loops):        # no outer bound flagged: the biggest loop
            size = lambda L: np.ptp(np.vstack([_cart(s) for e in L[1] for s in e]), axis=0).max()
            k = max(range(len(loops)), key=lambda i: size(loops[i]))
            loops[k] = (True, loops[k][1])
        loops.sort(key=lambda L: not L[0])
        return loops, seams

    def surface(self, ref, pts, seam_pts=None):
        """(du, dv, grid (nu, nv, 4) homogeneous, Uu, Uv) of a face's surface,
        analytic ones bounded to the points pts of the face's edges."""
        sf = self.sf
        t = sf.type(ref)
        a = sf.args(ref)
        if t == "COMPLEX" or t == "B_SPLINE_SURFACE_WITH_KNOTS":
            return self.bspline_surface(ref)
        if t == "RECTANGULAR_TRIMMED_SURFACE":
            return self.surface(a[1], pts, seam_pts)
        if t == "PLANE":
            o, x, y, z = placement(sf, a[1], self.f)
            u = (pts - o) @ x
            v = (pts - o) @ y
            pu, pv = 0.02 * np.ptp(u) + 1e-6, 0.02 * np.ptp(v) + 1e-6
            u0, u1, v0, v1 = u.min() - pu, u.max() + pu, v.min() - pv, v.max() + pv
            G = np.zeros((2, 2, 4))
            for i, uu in enumerate((u0, u1)):
                for j, vv in enumerate((v0, v1)):
                    G[i, j] = [*(o + uu * x + vv * y), 1.0]
            return 1, 1, G, [0, 0, 1, 1], [0, 0, 1, 1]
        if t in ("CYLINDRICAL_SURFACE", "CONICAL_SURFACE"):
            o, x, y, z = placement(sf, a[1], self.f)
            R = float(a[2]) * self.f
            tana = math.tan(float(a[3]) * self.ang) if t == "CONICAL_SURFACE" else 0.0
            h = (pts - o) @ z
            ph = 0.02 * np.ptp(h) + 1e-6
            h0, h1 = h.min() - ph, h.max() + ph
            if tana != 0.0 and R + h0 * tana < 0 < R + h1 * tana or (tana and R + h1 * tana < 0 < R + h0 * tana):
                apex = -R / tana                            # don't cross the apex
                h0, h1 = (apex, h1) if R + h1 * tana > 0 else (h0, apex)
            th0, th1 = self._angle_range(pts, o, x, y, seam_pts)
            prof = np.array([[R + h0 * tana, h0, 1.0], [R + h1 * tana, h1, 1.0]])
            return self._revolve(prof, 1, [0, 0, 1, 1], o, x, y, z, th0, th1)
        if t == "SPHERICAL_SURFACE":
            o, x, y, z = placement(sf, a[1], self.f)
            R = float(a[2]) * self.f
            d = pts - o
            zz = np.clip((d @ z) / R, -1, 1)
            ph = np.arcsin(zz)
            pad = 0.02 * np.ptp(ph) + 1e-4
            p0, p1 = max(-math.pi / 2, ph.min() - pad), min(math.pi / 2, ph.max() + pad)
            far = np.linalg.norm(d - np.outer(d @ z, z), axis=1) > 1e-6 * R
            th0, th1 = self._angle_range(pts[far] if far.any() else pts, o, x, y, seam_pts)
            prof_sp = arc_spans(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), R, R, p0, p1)
            prof, U = self._spans_rows(prof_sp)
            return self._revolve(prof, 2, U, o, x, y, z, th0, th1)
        if t == "TOROIDAL_SURFACE":
            o, x, y, z = placement(sf, a[1], self.f)
            Rm, rm = float(a[2]) * self.f, float(a[3]) * self.f
            d = pts - o
            rho = np.linalg.norm(d - np.outer(d @ z, z), axis=1)
            psi = np.arctan2(d @ z, rho - Rm)
            q0, q1 = _cover(psi, None)
            th0, th1 = self._angle_range(pts, o, x, y, seam_pts)
            prof_sp = arc_spans(np.array([Rm, 0, 0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), rm, rm, q0, q1)
            prof, U = self._spans_rows(prof_sp)
            return self._revolve(prof, 2, U, o, x, y, z, th0, th1)
        if t == "SURFACE_OF_REVOLUTION":
            ax = sf.args(a[2])
            o = sf.point(ax[1]) * self.f
            z = sf.direction(ax[2])
            cs = self.curve_spans(a[1], pts)
            # profile in the (radial, axial) frame of its own first point
            P0 = _cart(cs[0][0])[0] - o
            xr = P0 - (P0 @ z) * z
            if np.linalg.norm(xr) < 1e-12:
                Pm = curvejoin.eval_span(cs[len(cs) // 2][0], [0.5])[0] - o
                xr = Pm - (Pm @ z) * z
            x = xr / np.linalg.norm(xr)
            y = np.cross(z, x)
            spans = [s for s, _, _ in cs]
            th0, th1 = self._angle_range(pts, o, x, y, seam_pts, from_zero=True)
            return self._revolve_3d(spans, o, x, y, z, th0, th1)
        if t == "SURFACE_OF_LINEAR_EXTRUSION":
            cs = self.curve_spans(a[1], pts)
            va = sf.args(a[2])
            d = sf.direction(va[1])
            C = np.vstack([curvejoin.eval_span(s, np.linspace(0, 1, 9)) for s, _, _ in cs])
            s_lo = float((pts @ d).min() - (C @ d).max())
            s_hi = float((pts @ d).max() - (C @ d).min())
            pad = 0.02 * (s_hi - s_lo) + 1e-6
            s_lo, s_hi = s_lo - pad, s_hi + pad
            rows, U = self._spans_rows([s for s, _, _ in cs], cart3=True)
            q = len(cs[0][0]) - 1
            G = np.zeros((len(rows), 2, 4))
            for i, r in enumerate(rows):
                w = r[3]
                G[i, 0] = [*((r[:3] / w + s_lo * d) * w), w]
                G[i, 1] = [*((r[:3] / w + s_hi * d) * w), w]
            return q, 1, G, U, [0, 0, 1, 1]
        if t == "OFFSET_SURFACE":
            dist = float(a[2]) * self.f
            base = self.surface(a[1], pts, seam_pts)
            return ("offset", base, dist)
        raise ValueError(f"unsupported surface {t}")

    def bspline_surface(self, ref):
        sf = self.sf
        e = sf[ref]
        if isinstance(e, Typed):
            a = e.args
            du, dv, cps = int(a[1]), int(a[2]), a[3]
            mu, mv, ku, kv = a[8], a[9], a[10], a[11]
            W = None
        else:
            b = e["B_SPLINE_SURFACE"]
            du, dv, cps = int(b[0]), int(b[1]), b[2]
            k = e["B_SPLINE_SURFACE_WITH_KNOTS"]
            mu, mv, ku, kv = k[0], k[1], k[2], k[3]
            W = e["RATIONAL_B_SPLINE_SURFACE"][0] if "RATIONAL_B_SPLINE_SURFACE" in e else None
        P = np.array([[sf.point(c) for c in row] for row in cps]) * self.f
        W = np.ones(P.shape[:2]) if W is None else np.array(W, float)
        G = np.concatenate([P * W[..., None], W[..., None]], axis=2)
        Uu = [float(u) for u, m in zip(ku, mu) for _ in range(int(m))]
        Uv = [float(u) for u, m in zip(kv, mv) for _ in range(int(m))]
        G, Uu = _clamp_dir(G, Uu, du)
        Gt, Uv = _clamp_dir(G.transpose(1, 0, 2), Uv, dv)
        return du, dv, Gt.transpose(1, 0, 2), Uu, Uv

    # -- surfaces of revolution -----------------------------------------
    def _angle_range(self, pts, o, x, y, seam_pts, from_zero=False):
        d = pts - o
        th = np.arctan2(d @ y, d @ x)
        seam = None
        if seam_pts is not None and len(seam_pts):
            ds = seam_pts - o
            seam = float(np.arctan2(ds @ y, ds @ x).mean())
        return _cover(th, seam)

    @staticmethod
    def _spans_rows(spans, cart3=False):
        """Clamped CV rows + knots of a span list (Bezier knots)."""
        q = max(len(s) - 1 for s in spans)
        spans = [curvejoin.elevate(np.asarray(s, float), q) for s in spans]
        for k in range(1, len(spans)):
            a_, b_ = spans[k - 1][-1, 3], spans[k][0, 3]
            if b_ != 0:
                spans[k] = spans[k] * (a_ / b_)
        rows = np.array([spans[0][0]] + [r for s in spans for r in s[1:]])
        n = len(spans)
        U = [0.0] * (q + 1) + [k / n for k in range(1, n) for _ in range(q)] + [1.0] * (q + 1)
        if cart3:
            return rows, U
        # rows of a profile drawn in the plane: (radial, axial, w) from (x, y, z, w)
        return np.stack([rows[:, 0], rows[:, 1], rows[:, 3]], axis=1), U

    def _revolve(self, prof, dv, Uv, o, x, y, z, th0, th1):
        """prof rows (radial*w, axial*w, w) revolved about z from th0 to th1."""
        cs = arc_spans(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 1.0, 1.0, th0, th1)
        crow, Uu = self._spans_rows(cs, cart3=True)
        G = np.zeros((len(crow), len(prof), 4))
        for i, c in enumerate(crow):
            wc = c[3]
            cx, cy = c[0] / wc, c[1] / wc
            for j, pr in enumerate(prof):
                wp = pr[2]
                r, h = pr[0] / wp, pr[1] / wp
                P = o + h * z + r * (cx * x + cy * y)
                w = wc * wp
                G[i, j] = [*(P * w), w]
        return 2, dv, G, Uu, Uv

    def _revolve_3d(self, spans, o, x, y, z, th0, th1):
        rows, Uv = self._spans_rows(spans, cart3=True)
        dv = len(spans[0]) - 1 if all(len(s) == len(spans[0]) for s in spans) else max(len(s) for s in spans) - 1
        cs = arc_spans(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 1.0, 1.0, th0, th1)
        crow, Uu = self._spans_rows(cs, cart3=True)
        G = np.zeros((len(crow), len(rows), 4))
        for i, c in enumerate(crow):
            wc = c[3]
            cx, cy = c[0] / wc, c[1] / wc
            for j, r in enumerate(rows):
                wp = r[3]
                P = r[:3] / wp - o
                h = float(P @ z)
                rho = P - h * z
                Q = o + h * z + cx * rho + cy * np.cross(z, rho)
                w = wc * wp
                G[i, j] = [*(Q * w), w]
        return 2, dv, G, Uu, Uv

    # -- one face --------------------------------------------------------
    def make_surface(self, data, flip):
        if isinstance(data, tuple) and data and data[0] == "offset":
            _, base, dist = data
            # offset(flip(base), -d) is offset(base, d) with the normal reversed
            s = self.make_surface(base, flip)
            return self.sm.create_offset_surface(s, -dist if flip else dist)
        du, dv, G, Uu, Uv = data
        if flip:          # reverse u: the face's normal is opposite the surface's
            G = G[::-1].copy()
            a, b = Uu[0], Uu[-1]
            Uu = [a + b - u for u in reversed(Uu)]
        norm = lambda K: [(k - K[0]) / (K[-1] - K[0]) for k in K]
        nu, nv = G.shape[:2]
        w = G[..., 3]
        pts = [tuple(map(float, G[i, j, :3] / w[i, j])) for i in range(nu) for j in range(nv)]
        rational = np.abs(w - 1.0).max() > 1e-12
        return self.ext.surface_from_cvs(int(du), int(dv), pts, norm(Uu), norm(Uv),
                                         [float(v) for v in w.reshape(-1)] if rational else None)

    def face(self, face_ref):
        sf = self.sf
        a = sf.args(face_ref)
        loops, seams = self.face_loops(face_ref)
        if not loops:
            raise ValueError("face without edge loops")
        pts = np.vstack([curvejoin.eval_span(s, np.linspace(0, 1, 9)) for _, L in loops for e in L for s in e])
        seam_pts = None
        if seams:
            seam_pts = np.vstack([curvejoin.eval_span(s, [0.5]) for ec in seams for s in self.edge_spans(ec)])
        data = self.surface(a[2], pts, seam_pts)
        same_sense = a[3] != "F"
        surf = self.make_surface(data, not same_sense)
        counts, curves = [], []
        for _, L in loops:
            counts.append(len(L))
            for e in L:
                curves.append(spans_to_kernel(self.ext, e))
        try:
            return self.sm.trim_surface_with_3d_curves(surf, counts, curves)
        except Exception:
            # a face whose single loop runs along its surface's own boundary is
            # the whole surface: use it untrimmed (the kernel's trimming refuses
            # some of these, e.g. very thin sliver faces)
            if len(loops) == 1 and not isinstance(data[0], str) and _on_boundary(data, pts):
                return self.ext.brep_from_surface(self.make_surface(data, not same_sense))
        # edges a little off their surface (an approximate offset surface in the
        # source CAD): pull the loops onto the surface and trim again
        surf = self.make_surface(data, not same_sense)
        probe = self.make_surface(data, not same_sense)
        curves = [self._projected_curve(probe, e) for _, L in loops for e in L]
        return self.sm.trim_surface_with_3d_curves(surf, counts, curves)

    def _projected_curve(self, surf, spans):
        from .drawtools import interpolate
        from . import geom
        n = max(8, 6 * len(spans))
        S = []
        for k, s in enumerate(spans):
            ts = np.linspace(0, 1, 7)[(1 if k else 0):]
            S.append(curvejoin.eval_span(s, ts))
        S = np.vstack(S)
        idx = np.linspace(0, len(S) - 1, min(len(S), max(n, 4))).round().astype(int)
        S = S[np.unique(idx)]
        Q = []
        for p in S:
            q = self.sm.surface_closest_point(surf, tuple(map(float, p)))
            q = q[0] if isinstance(q, tuple) else q
            Q.append([q.x, q.y, q.z])
        Q = np.array(Q)
        X, deg = interpolate(Q, 3)
        U = geom.clamped_uniform_knots(len(X), deg)
        return self.ext.curve_from_cvs(deg, [tuple(map(float, x)) for x in X], U, None)


def _on_boundary(data, pts, rel=1e-6):
    """Do all the points lie on the boundary curves of the surface?"""
    du, dv, G, Uu, Uv = data
    samples = []
    for row, U, p in ((G[0], Uv, dv), (G[-1], Uv, dv), (G[:, 0], Uu, du), (G[:, -1], Uu, du)):
        for s, _, _ in bspline_spans(list(row), U, p):
            samples.append(curvejoin.eval_span(s, np.linspace(0, 1, 65)))
    B = np.vstack(samples)
    size = float(np.linalg.norm(np.ptp(B, axis=0))) or 1.0
    from .curvetools import _seg_dist
    return float(_seg_dist(np.asarray(pts), B).max()) <= max(rel * size * 100, 1e-6)


def _clamp_dir(G, U, p):
    """Grid (n, m, 4) with knot vector U in the first direction made clamped
    (Bezier extraction of each column when U is not clamped)."""
    n = G.shape[0]
    clamped = (all(abs(U[i] - U[0]) < 1e-12 for i in range(p + 1)) and
               all(abs(U[-1 - i] - U[-1]) < 1e-12 for i in range(p + 1)))
    if clamped and len(U) == n + p + 1:
        return G, U
    cols = []
    for j in range(G.shape[1]):
        sp = bspline_spans([G[i, j] for i in range(n)], U, p)
        cols.append(sp)
    k = len(cols[0])
    rows = [cols[0][0][0][0]]
    Gn = np.zeros((k * p + 1, G.shape[1], 4))
    for j, sp in enumerate(cols):
        r = [sp[0][0][0]] + [x for s, _, _ in sp for x in s[1:]]
        Gn[:, j] = np.array(r)
    t = [s[1] for s in cols[0]] + [cols[0][-1][2]]
    Un = [t[0]] * (p + 1) + [x for x in t[1:-1] for _ in range(p)] + [t[-1]] * (p + 1)
    return Gn, Un


def _cover(th, seam):
    """Smallest angle interval [a0, a1] (a1 > a0) covering the angles th.
    With a seam (the face wraps all round) the full circle from the seam."""
    th = np.sort(np.mod(np.asarray(th, float), 2 * math.pi))
    if seam is not None:
        return seam, seam + 2 * math.pi
    if len(th) == 0:
        return 0.0, 2 * math.pi
    gaps = np.diff(np.concatenate([th, [th[0] + 2 * math.pi]]))
    k = int(np.argmax(gaps))
    if gaps[k] < math.radians(10):
        return 0.0, 2 * math.pi
    a0 = th[(k + 1) % len(th)]
    a1 = th[k]
    if a1 <= a0:
        a1 += 2 * math.pi
    span = a1 - a0
    pad = min(0.02 * span + math.radians(0.5), (2 * math.pi - span) / 2)
    return a0 - pad, a1 + pad


# --------------------------------------------------------------------------
# Bodies, products, assembly placements
# --------------------------------------------------------------------------

BODY_TYPES = ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS", "SHELL_BASED_SURFACE_MODEL")
REP_TYPES = ("ADVANCED_BREP_SHAPE_REPRESENTATION", "MANIFOLD_SURFACE_SHAPE_REPRESENTATION", "SHAPE_REPRESENTATION",
             "FACETED_BREP_SHAPE_REPRESENTATION")


def body_faces(sf, ref):
    t = sf.type(ref)
    a = sf.args(ref)
    shells = []
    if t == "MANIFOLD_SOLID_BREP":
        shells = [a[1]]
    elif t == "BREP_WITH_VOIDS":
        shells = [a[1]] + list(a[2])
    elif t == "SHELL_BASED_SURFACE_MODEL":
        shells = list(a[1])
    faces = []
    for s in shells:
        sa = sf.args(s)
        if sf.type(s) == "ORIENTED_CLOSED_SHELL":
            sa = sf.args(sa[2])
        faces += list(sa[1])
    return faces, t != "SHELL_BASED_SURFACE_MODEL"


def _rep_items(sf, rep):
    a = sf.args(rep)
    if a is None:          # complex representation
        e = sf[rep]
        for k in ("REPRESENTATION", "SHAPE_REPRESENTATION"):
            if k in e:
                a = e[k]
                break
    return a[1] if a else [], a[2] if a else None


def scene_bodies(sf):
    """[(body ref, name, 4x4 transform in file units, context ref)]."""
    reps = {}
    for r in sf.of_type(*REP_TYPES):
        items, ctx = _rep_items(sf, r)
        reps[r] = (items, ctx)
    # product names per representation
    names = {}
    for sdr in sf.of_type("SHAPE_DEFINITION_REPRESENTATION"):
        pds, rep = sf.args(sdr)
        try:
            pd = sf.args(pds)[2]
            pdf = sf.args(pd)[2]
            prod = sf.args(pdf)[2]
            names[rep] = sf.args(prod)[0] or sf.args(prod)[1]
        except Exception:
            pass
    # parent links: child rep -> [(parent rep, transform child -> parent)]
    parents = {}
    for r in sf.of_type("SHAPE_REPRESENTATION_RELATIONSHIP", "REPRESENTATION_RELATIONSHIP"):
        e = sf[r]
        if isinstance(e, dict):
            rr = e.get("REPRESENTATION_RELATIONSHIP")
            tr = e.get("REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
            if not rr:
                continue
            r1, r2 = rr[2], rr[3]
            T = np.eye(4)
            if tr and sf.type(tr[0]) == "ITEM_DEFINED_TRANSFORMATION":
                ia = sf.args(tr[0])
                T = placement_matrix(sf, ia[3]) @ np.linalg.inv(placement_matrix(sf, ia[2]))
            parents.setdefault(r1, []).append((r2, T))
        else:
            a = e.args
            parents.setdefault(a[2], []).append((a[3], np.eye(4)))      # same part, no transform
    # mapped items: rep R contains MAPPED_ITEM(source rep) at a target placement
    for r, (items, ctx) in reps.items():
        for it in items:
            if sf.type(it) == "MAPPED_ITEM":
                ma = sf.args(it)
                rmap = sf.args(ma[1])
                src = rmap[1]
                T = placement_matrix(sf, ma[2]) @ np.linalg.inv(placement_matrix(sf, rmap[0]))
                parents.setdefault(src, []).append((r, T))

    def paths(rep, depth=0):
        if depth > 32:
            return [np.eye(4)]
        ups = parents.get(rep, [])
        if not ups:
            return [np.eye(4)]
        out = []
        for p, T in ups:
            for M in paths(p, depth + 1):
                out.append(M @ T)
        return out

    def name_of(rep, depth=0):
        if rep in names:
            return names[rep]
        for p, _ in parents.get(rep, []):
            if depth < 8:
                n = name_of(p, depth + 1)
                if n:
                    return n
        return None

    out, used = [], set()
    for r, (items, ctx) in reps.items():
        for it in items:
            if sf.type(it) in BODY_TYPES and it not in used:
                nm = sf.args(it)[0] or name_of(r) or ""
                nm_prod = name_of(r)
                for M in paths(r):
                    out.append((it, nm_prod or nm, M, ctx))
                used.add(it)
    # bodies not referenced from any representation
    for b in sf.of_type(*BODY_TYPES):
        if b not in used:
            out.append((b, sf.args(b)[0] or "", np.eye(4), None))
    return out


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------

FORCE_FALLBACK = False


def _append(sm, ext, dst, src):
    """Add the faces of src to dst without intersecting them. append_brep comes
    with this add-on's kernel build; an older loaded kernel (Blender not
    restarted since an update) uses piecewise_merge."""
    if hasattr(ext, "append_brep") and not FORCE_FALLBACK:
        ext.append_brep(dst, src)
        return dst
    try:
        return sm.piecewise_merge(dst, src)
    except Exception:
        raise RuntimeError("the kernel loaded in this Blender session is older than the add-on (no append_brep): "
                           "restart Blender to load the updated kernel, then import again")


def read_step(path, scale=1.0, stitch=True, progress=None):
    """Import a STEP file. Returns ([(name, brep, 4x4 matrix in Blender units, info)], report).
    scale = Blender units per metre."""
    from .kernel import load
    sm = load()
    if sm is None:
        raise RuntimeError("USD BRep kernel not available")
    ext = sm.blender_ext
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    t_start = time.time()
    sf = StepFile(text)
    bodies = scene_bodies(sf)
    if not bodies:
        raise ValueError("no B-rep bodies in this file (only B-rep solids and surface models are read)")
    results = []
    report = {"bodies": 0, "faces": 0, "failed": 0, "solids": 0, "reasons": {}}
    cache = {}
    total = sum(len(body_faces(sf, b)[0]) for b, _, _, _ in bodies)
    done = 0
    for body, name, M, ctx in bodies:
        Lf, Af = length_factor(sf, ctx) if ctx is not None else (0.001, 1.0)
        f = Lf * scale                                   # file units -> Blender units
        # faces are built and stitched in millimetres (the kernel's tolerances
        # suit part-sized numbers; a 20 mm part is 0.02 Blender units), then the
        # finished body is scaled to Blender units
        fk = Lf / 0.001
        unc = uncertainty(sf, ctx) if ctx is not None else None
        key = (body, f)
        if key not in cache:
            faces, solid = body_faces(sf, body)
            imp = Importer(sf, fk, Af)
            brep, nok, nfail = None, 0, 0
            for fr in faces:
                done += 1
                if progress:
                    progress(done, total)
                try:
                    fb = imp.face(fr)
                except Exception as exc:
                    nfail += 1
                    why = str(exc).split("\n")[0][:80]
                    report["reasons"][why] = report["reasons"].get(why, 0) + 1
                    continue
                nok += 1
                if brep is None:
                    brep = fb
                else:
                    brep = _append(sm, ext, brep, fb)
            if brep is None:
                cache[key] = None
                report["failed"] += nfail
                continue
            closed = False
            if stitch and nok > 1:
                P = np.array(sm.brep_bounding_box(brep)) if hasattr(sm, "brep_bounding_box") else None
                size = float(np.linalg.norm(np.ptp(np.array(P).reshape(-1, 3), axis=0))) if P is not None else 1.0
                tol = max((unc or 0.0) * fk * 10, 1e-6 * size, 1e-3)    # mm
                try:
                    sm.stitch_advanced(brep, tol, True, True, solid, False)
                except Exception as exc:
                    report["reasons"]["stitch: " + str(exc)[:60]] = 1
                try:
                    closed = bool(sm.is_manifold_solid(brep))
                except Exception:
                    closed = False
            s_mm = 0.001 * scale                         # millimetres -> Blender units
            if abs(s_mm - 1.0) > 1e-15:
                sm.scale(brep, (s_mm, s_mm, s_mm))
            cache[key] = (brep, nok, nfail, closed)
            report["faces"] += nok
            report["failed"] += nfail
        hit = cache[key]
        if hit is None:
            continue
        brep, nok, nfail, closed = hit
        Mb = np.array(M, float)
        Mb[:3, 3] *= f                                   # placements are in file units
        results.append((name or os.path.splitext(os.path.basename(path))[0], brep, Mb,
                        {"faces": nok, "failed": nfail, "solid": closed}))
        report["bodies"] += 1
        report["solids"] += int(closed)
    report["seconds"] = time.time() - t_start
    return results, report


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty, FloatProperty, StringProperty  # noqa: E402
from bpy_extras.io_utils import ImportHelper  # noqa: E402
from mathutils import Matrix  # noqa: E402


def import_into_scene(context, path, stitch=True, extra_scale=1.0):
    sc = (context.scene.unit_settings.scale_length or 1.0)
    wm = context.window_manager
    wm.progress_begin(0, 100)
    try:
        res, rep = read_step(path, extra_scale / sc, stitch,
                             progress=lambda d, t: wm.progress_update(int(100 * d / max(t, 1))))
    finally:
        wm.progress_end()
    stem = os.path.splitext(os.path.basename(path))[0]
    coll = bpy.data.collections.new(stem)
    (context.collection or context.scene.collection).children.link(coll)
    objs = []
    texts = {}
    for name, brep, M, info in res:
        key = id(brep)
        if key not in texts:
            texts[key] = store.brep_to_text(brep)
        me = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, me)
        coll.objects.link(obj)
        obj.matrix_world = Matrix([list(r) for r in M])
        store.set_geometry_text(obj, texts[key], context, keep_untrimmed=False)
        obj["nproto_step_source"] = os.path.basename(path)
        objs.append(obj)
    return objs, rep


class NPROTO_OT_import_step(bpy.types.Operator, ImportHelper):
    """Import B-rep solids and surfaces from a STEP file (AP203 / AP214 / AP242) as exact trimmed NURBS"""
    bl_idname = "nproto.import_step"
    bl_label = "Import STEP"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

    filename_ext = ".step"
    filter_glob: StringProperty(default="*.step;*.stp;*.STEP;*.STP", options={'HIDDEN'})
    stitch: BoolProperty(name="Stitch Faces", default=True,
                         description="Sew the faces of each body together (solids come back closed)")
    extra_scale: FloatProperty(name="Scale", default=1.0, min=1e-6, soft_max=1000.0,
                               description="Extra scale on top of the file's own units")

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            objs, rep = import_into_scene(context, self.filepath, self.stitch, self.extra_scale)
        except Exception as exc:
            self.report({'ERROR'}, f"STEP import failed: {exc}")
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        if objs:
            context.view_layer.objects.active = objs[-1]
        msg = (f"Imported {rep['bodies']} bodies, {rep['faces']} faces ({rep['solids']} closed solids) "
               f"in {rep['seconds']:.1f}s")
        if rep["failed"]:
            top = sorted(rep["reasons"].items(), key=lambda kv: -kv[1])[:3]
            msg += f" -- {rep['failed']} faces could not be built: " + "; ".join(f"{k} ({v})" for k, v in top)
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def menu_import(self, context):
    self.layout.operator(NPROTO_OT_import_step.bl_idname, text="NURBS (.step)")


classes = (NPROTO_OT_import_step,)


def register():
    bpy.types.TOPBAR_MT_file_import.append(menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
