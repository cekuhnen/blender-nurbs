"""Continuity Check: G0 / G1 / G2 measured along picked edges and drawn in the
viewport (label + comb), live while the geometry changes.

Pick ONE edge of a joined body (an edge between two faces: checked between
those faces) or the edge of a separate surface: its partner edge on another
surface is found by itself, else pick it as the second edge. Every checked edge
gets a label at its middle -- "G2 ✓ 1.2%": the level shown, whether it holds
and its worst value -- and a comb along it:

Each level draws its deviation as a comb standing on the surface, scaled to
the edge's own largest value (red past the tolerance; a faint orange line at
the tolerance height when it is in range):

* G0: the gap between the two sides.
* G1: the angle between the two sides' normals.
* G2: the difference of the two sides' curvature across the edge. Optionally (Curvature Combs) each side's own curvature comb too
  (white / magenta): where the edge is G2 they lie on each other.

G0 measures the SURFACES: across a seam, how far the surface's two ends are
apart (a program that compares the shared edge with itself reports 0 there).

The checked edges stay on screen after the tool ends (Analyze panel: level,
comb scale, tolerances, the list, Clear), so they update while you edit.

The labels of Square / Loft / Patch ... (Display > Feature Labels) are the
feature's own SETTINGS (clickable); this tool MEASURES any edge.
"""

import math

import numpy as np

import bpy
from bpy.props import IntProperty
from mathutils import Vector

from . import draw, geom, store

PAIRS = []           # [{"a": (obj name, ref, local mid), "b": (...) | None}]
_cache = {}          # id(pair) -> (key, result | error string)
PENDING = {"pts": None, "text": ""}   # the tool's first pick, waiting for the edge it meets
TOOL = {"running": False, "stop": False}  # the picking tool, so its button can switch it off


def active(context=None):
    """Is Continuity Check on (picking, or its checks on screen)? Its button
    shows pressed then, and a click switches it off -- like Zebra."""
    v = getattr((context or bpy.context).window_manager, "nproto_view", None)
    return TOOL["running"] or bool(v is not None and v.cont_show_checks and PAIRS)


def stop(context=None):
    """Switch Continuity Check off: end the picking and hide the checks (the
    list stays; switching it on shows it again)."""
    v = getattr((context or bpy.context).window_manager, "nproto_view", None)
    if TOOL["running"]:
        TOOL["stop"] = True
    if v is not None:
        v.cont_show_checks = False
    _drop_pending()
    draw.pick_overlay["hover"] = None
N_SAMPLES = 160
LEVELS = ("G0", "G1", "G2")
from .diagnostics import COMB_BAD, COMB_ENV_W, COMB_ENVELOPE, COMB_HAIR, COMB_HAIR_W, COMB_SPAN  # noqa: E402

COL_OK = (0.3, 0.95, 0.45, 1.0)          # the edge line / label: holds
COL_BAD = COMB_BAD                        # fails
COL_B = (0.3, 0.75, 1.0, 0.9)            # Curvature Combs: the second side
COL_SKIP = (0.55, 0.55, 0.55, 0.8)
COL_TOL = (1.0, 0.75, 0.2, 0.55)      # the tolerance height over a comb
COL_LOW = (1.0, 0.55, 0.1, 1.0)       # holds at this level, but a lower level fails there
LEVEL_COL = {"G0": (0.9, 0.9, 0.9, 1), "G1": (0.35, 0.8, 1.0, 1), "G2": (0.4, 1.0, 0.5, 1)}


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------

def _edge_mid_local(e):
    t0, t1 = e.parameter_range()
    for f in (0.5, 0.49, 0.51):
        try:
            return np.array(e.point_at(t0 + (t1 - t0) * f), float)
        except Exception:
            continue
    return None


def side_key(obj, ref):
    """(name, ref, local midpoint): the midpoint finds a brep edge again when a
    rebuild renumbers the edges."""
    mid = None
    if ref.startswith("E"):
        try:
            m = _edge_mid_local(store.get_brep(obj).edges()[int(ref[1:])])
            mid = None if m is None else tuple(float(x) for x in m)
        except Exception:
            mid = None
    return (obj.name, ref, mid)


def _find_edge(brep, ref, mid):
    edges = brep.edges()
    i = int(ref[1:])
    if mid is None:
        return edges[i] if i < len(edges) else None
    best, bd = None, None
    for e in edges:
        m = _edge_mid_local(e)
        if m is None:
            continue
        d = float(np.linalg.norm(m - np.asarray(mid)))
        if bd is None or d < bd:
            best, bd = e, d
    return best


def _is_seam(e):
    """A face meeting itself (the seam of a closed surface): one face whose
    loops use the edge twice."""
    fs = e.faces()
    if len(fs) != 1:
        return False
    n = 0
    for lp in fs[0].boundary_loops():
        for ed, _ in lp:
            if ed == e:
                n += 1
    return n >= 2


def edge_kind(obj, ref):
    """"shared" (between two faces of a joined body), "seam" (a closed face
    meeting itself) or "open" (a boundary: its partner is on another surface)."""
    if not ref.startswith("E"):
        return "open"
    try:
        e = store.get_brep(obj).edges()[int(ref[1:])]
    except Exception:
        return "open"
    if len(e.faces()) == 2:
        return "shared"
    return "seam" if _is_seam(e) else "open"


def edge_faces(obj, ref):
    """How many faces the picked edge has (2 = a stitched edge of a joined body)."""
    if not ref.startswith("E"):
        return 1
    try:
        return len(store.get_brep(obj).edges()[int(ref[1:])].faces())
    except Exception:
        return 0


class _Side:
    """One picked edge: its curve in world space and the face(s) it bounds."""

    def __init__(self, obj, ref, mid):
        self.obj = obj
        Md = np.array(geom.world_matrix(obj), float)
        self.R, self.T = Md[:3, :3], Md[:3, 3]
        self.Ri = np.linalg.inv(self.R)
        b = store.get_brep(obj)
        if ref.startswith("E"):
            e = _find_edge(b, ref, mid)
            if e is None:
                raise ValueError(f"{obj.name}: the edge is gone")
            self.faces = list(e.faces())
            self.seam = _is_seam(e)
            self.uv_at = None
            self.t0, self.t1 = e.parameter_range()
            self._pt = lambda t: e.point_at(t)
        else:
            face = b.faces()[0]
            surf = face.surface()
            (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
            self.faces = [face]
            self.seam = False
            if ref in ("U0", "U1"):
                uc = u0 if ref == "U0" else u1
                self.t0, self.t1 = v0, v1
                self._pt = lambda t: surf.evaluate(uc, t)
                self.uv_at = lambda t: (uc, t)
            else:
                vc = v0 if ref == "V0" else v1
                self.t0, self.t1 = u0, u1
                self._pt = lambda t: surf.evaluate(t, vc)
                self.uv_at = lambda t: (t, vc)

    def world(self, t):
        t = min(max(t, self.t0), self.t1)
        try:
            p = self._pt(t)
        except Exception:              # some kernels refuse the very end of the range
            span = self.t1 - self.t0
            p = self._pt(min(max(t, self.t0 + 1e-7 * span), self.t1 - 1e-7 * span))
        return self.R @ np.asarray(p, float) + self.T

    def closest(self, p, ts, P):
        """Parameter and point of this edge nearest to p (world): the nearest
        of the samples P (at ts), refined by golden section."""
        i = int(np.argmin(np.linalg.norm(P - p, axis=1)))
        a, b = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
        g = 0.5 * (math.sqrt(5.0) - 1.0)
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = np.linalg.norm(self.world(c) - p), np.linalg.norm(self.world(d) - p)
        for _ in range(40):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = np.linalg.norm(self.world(c) - p)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = np.linalg.norm(self.world(d) - p)
        t = 0.5 * (a + b)
        return t, self.world(t)


def _eval(face, side, p, T, uv0):
    """Foot of p (world) on the face's surface: (uv, point, unit normal -- the
    face's outward one when the kernel knows it --, normal curvature across T)."""
    from .square import _foot, _second
    surf = face.surface()
    q = side.Ri @ (p - side.T)
    uv = _foot(surf, q, uv0)
    su, sv, suu, suv, svv = (side.R @ np.asarray(x, float) for x in _second(surf, *uv))
    pt = side.R @ np.asarray(surf.evaluate(*uv), float) + side.T
    n = np.cross(su, sv)
    n /= np.linalg.norm(n) or 1.0
    try:
        on = side.R @ np.asarray(face.outward_normal_at_uv(*uv), float)
        if n @ on < 0:
            n = -n
    except Exception:
        pass
    w = np.cross(n, T)
    w /= np.linalg.norm(w) or 1.0
    a, c = np.linalg.lstsq(np.array([su, sv]).T, w, rcond=None)[0]
    du = a * su + c * sv
    k = (a * a * (suu @ n) + 2 * a * c * (suv @ n) + c * c * (svv @ n)) / max(du @ du, 1e-300)
    return uv, pt, n, float(k)


def _fresh_if_lost(face, side, p, T, uv0, tol):
    """_eval started from the previous sample's foot (it follows the edge);
    when that lands off p -- the face is closed and its seam lies elsewhere
    than the edge's start, or it runs the other way: the search jams against
    the seam -- search the whole face again, and from across the seam (a foot
    stuck on one end of the domain starts again from the other end), and
    keep the nearest foot."""
    r = _eval(face, side, p, T, uv0)
    if np.linalg.norm(r[1] - p) <= tol:
        return r
    tried = [r] + ([_eval(face, side, p, T, None)] if uv0 is not None else [])
    (u0, u1), (v0, v1) = face.surface().u_domain(), face.surface().v_domain()
    eu, ev = 1e-9 * (u1 - u0), 1e-9 * (v1 - v0)
    for q in list(tried):
        u, v = q[0]
        for uv in (((u1 if abs(u - u0) <= eu else u0), v) if min(abs(u - u0), abs(u - u1)) <= eu else None,
                   (u, (v1 if abs(v - v0) <= ev else v0)) if min(abs(v - v0), abs(v - v1)) <= ev else None):
            if uv is not None:
                try:
                    tried.append(_eval(face, side, p, T, uv))
                except Exception:
                    pass
    return min(tried, key=lambda q: np.linalg.norm(q[1] - p))


def _face_kmax(face, side, m=9):
    """Largest principal curvature of the face's surface (m x m grid over its
    domain, world space): the scale a G2 curvature difference is judged
    against -- where a G2 blend meets a straight wall both sides are ~flat AT
    the edge, and a residual there means nothing next to the blend's own 1/r."""
    from .square import _second
    surf = face.surface()
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    best = 0.0
    for u in np.linspace(u0, u1, m):
        for v in np.linspace(v0, v1, m):
            try:
                su, sv, suu, suv, svv = (side.R @ np.asarray(x, float) for x in _second(surf, u, v))
            except Exception:
                continue
            n = np.cross(su, sv)
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue                      # a pole
            n /= ln
            E, F, G = su @ su, su @ sv, sv @ sv
            e, f, g = suu @ n, suv @ n, svv @ n
            den = E * G - F * F
            if den <= 1e-300:
                continue
            H = (e * G - 2 * f * F + g * E) / (2 * den)
            K = (e * g - f * f) / den
            best = max(best, abs(H) + math.sqrt(max(H * H - K, 0.0)))
    return best


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def _measure(A, fa, B, fb, separate, n):
    """Samples along A's edge, even by arc length: gap, normal angle, both
    cross curvatures. An open edge leaves its very ends out (corners, where
    a third face takes over); a closed one (a rim) is sampled all the way
    round, its seam included -- the seams of the faces often sit there and
    are where they kink. Separate edges: samples off B's edge are dropped."""
    ts = np.linspace(A.t0, A.t1, 4 * n + 1)
    PA = np.array([A.world(t) for t in ts])
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(PA, axis=0), axis=1))])
    L = float(d[-1]) or 1.0
    closed = float(np.linalg.norm(PA[0] - PA[-1])) < 1e-4 * L     # (kernel trim curves close to ~1e-6)
    frac = np.arange(n) / n if closed else np.linspace(0.015, 0.985, n)
    tsel = np.interp(frac * L, d, ts)
    if separate:
        tsB = np.linspace(B.t0, B.t1, 8 * n + 1)
        PB = np.array([B.world(t) for t in tsB])
    reach = 0.02 * L
    rows = []
    uva = uvb = None
    for t in tsel:
        p = A.world(t)
        dt = 1e-4 * (A.t1 - A.t0)
        T = A.world(t + dt) - A.world(t - dt)
        T /= np.linalg.norm(T) or 1.0
        if separate:
            tb, qb = B.closest(p, tsB, PB)
            if np.linalg.norm(qb - p) > reach:
                rows.append(None)
                continue
        try:
            # an untrimmed surface's own boundary: start on it -- the two ends of a
            # closed surface lie on each other and a free search may take either
            uva, pa, na, ka = _fresh_if_lost(fa, A, p, T, A.uv_at(t) if A.uv_at else uva, 1e-6 * L)
            uvb, pb, nb, kb = _fresh_if_lost(fb, B, qb if separate else p, T,
                                             B.uv_at(tb) if separate and B.uv_at else uvb, 1e-6 * L)
        except Exception:
            rows.append(None)
            continue
        gv = (qb - p) if separate else (pb - pa)
        if na @ nb < 0:          # sides oriented apart (separate surfaces): compare like for like
            nb, kb = -nb, -kb
        ang = math.degrees(math.acos(min(1.0, max(-1.0, float(na @ nb)))))
        rows.append((p, na, gv, float(np.linalg.norm(gv)), ang, ka, kb))
    return rows, L, closed


def recommended(faces):
    """Samples for a true comb: 4 per CV of the faces' surfaces (capped). A
    dense kernel fit (a blend with ~200 CVs round a rim) deviates between its
    fit points, and fewer samples alias that into random spikes. Only advice:
    the check uses the Samples you set (the panel says when it is fewer)."""
    from .kernel import load
    ext = load().blender_ext
    most = 0
    for f in faces:
        try:
            d = ext.surface_cvs(f.surface())
            most = max(most, d["count_u"], d["count_v"])
        except Exception:
            pass
    return int(min(4 * most, 1200))


def samples(context=None):
    v = getattr((context or bpy.context).window_manager, "nproto_view", None)
    return int(v.cont_samples) if v is not None else N_SAMPLES


def _measure_seam(A, face, n):
    """Across a seam: the face's surface at its two parameter ends (u = u0 and
    u = u1, or v), which meet along the edge."""
    surf = face.surface()
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    ts = np.linspace(A.t0, A.t1, 4 * n + 1)
    PA = np.array([A.world(t) for t in ts])
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(PA, axis=0), axis=1))])
    L = float(d[-1]) or 1.0
    tsel = np.interp(np.linspace(0.015, 0.985, n) * L, d, ts)
    # which direction closes: the surface's two ends lie on each other there
    mid = A.Ri @ (PA[len(PA) // 2] - A.T)
    from .square import _foot
    um, vm = _foot(surf, mid, None)
    du = float(np.linalg.norm(np.asarray(surf.evaluate(u0, vm)) - np.asarray(surf.evaluate(u1, vm))))
    dv = float(np.linalg.norm(np.asarray(surf.evaluate(um, v0)) - np.asarray(surf.evaluate(um, v1))))
    in_u = du <= dv
    rows = []
    last = None
    for t in tsel:
        p = A.world(t)
        dt = 1e-4 * (A.t1 - A.t0)
        T = A.world(t + dt) - A.world(t - dt)
        T /= np.linalg.norm(T) or 1.0
        q = A.Ri @ (p - A.T)
        try:
            u, v = _foot(surf, q, last)
            last = (u, v)
            sa, sb = ((u0, v), (u1, v)) if in_u else ((u, v0), (u, v1))
            _, pa, na, ka = _eval(face, A, p, T, sa)
            _, pb, nb, kb = _eval(face, A, p, T, sb)
        except Exception:
            rows.append(None)
            continue
        gv = pb - pa
        ang = math.degrees(math.acos(min(1.0, max(-1.0, float(na @ nb)))))
        rows.append((p, na, gv, float(np.linalg.norm(gv)), ang, ka, kb))
    return rows, L, False


def analyze(pair, n=None):
    """{P, N, D (gap vectors), gap, ang, kA, kB, crel, L, mid, ok mask}."""
    n = n or samples()
    a = pair["a"]
    oa = bpy.data.objects.get(a[0])
    if oa is None:
        raise ValueError(f"{a[0]} is gone")
    A = _Side(oa, a[1], a[2])
    if pair["b"] is None and A.seam:
        rec = recommended(A.faces)
        rows, L, closed = _measure_seam(A, A.faces[0], n)
    elif pair["b"] is None:
        if len(A.faces) != 2:
            raise ValueError("the edge is not shared by two faces any more")
        rec = recommended(A.faces)
        rows, L, closed = _measure(A, A.faces[0], A, A.faces[1], False, n)
    else:
        b = pair["b"]
        ob = bpy.data.objects.get(b[0])
        if ob is None:
            raise ValueError(f"{b[0]} is gone")
        B = _Side(ob, b[1], b[2])
        rec = recommended(A.faces[:1] + B.faces[:1])
        rows, L, closed = _measure(A, A.faces[0], B, B.faces[0], True, n)
        if sum(r is not None for r in rows) < 0.3 * n:       # B is the shorter one: measure along it
            rows2, L2, closed2 = _measure(B, B.faces[0], A, A.faces[0], True, n)
            if sum(r is not None for r in rows2) > sum(r is not None for r in rows):
                rows, L, closed = rows2, L2, closed2
    faces = (A, A.faces[0], A, A.faces[-1]) if pair["b"] is None else (A, A.faces[0], B, B.faces[0])
    good = [r for r in rows if r is not None]
    if len(good) < max(3, 0.2 * n):
        raise ValueError("the edges don't meet")
    P, N, D, gap, ang, kA, kB = (np.array(x, float) for x in zip(*good))
    used = np.array([r is not None for r in rows])
    Lc = L * float(used.mean())
    kref = max(float(np.max(np.abs(kA))), float(np.max(np.abs(kB))))
    kscale = max([kref] + [_face_kmax(f, sd) for sd, f in ((faces[0], faces[1]), (faces[2], faces[3]))])
    crel = np.abs(kA - kB) / max(kscale, 0.05 / (Lc or 1.0))
    return {"P": P, "N": N, "D": D, "gap": gap, "ang": ang, "kA": kA, "kB": kB, "crel": crel,
            "L": Lc, "kref": kref, "kscale": kscale, "mid": P[len(P) // 2], "covered": float(used.mean()),
            "closed": bool(closed and used.all()), "recommended": rec}


def _key(pair):
    out = []
    for s in (pair["a"], pair["b"]):
        if s is None:
            continue
        o = bpy.data.objects.get(s[0])
        if o is None:
            return None
        out.append((o.get(store.PROP_REV), o.get("nproto_build_key"), tuple(round(x, 9) for r in o.matrix_world
                                                                            for x in r)))
    return tuple(out) + (samples(),)


def result(pair):
    """Cached analysis (recomputed when either object rebuilds or moves); a
    string when it can't be measured."""
    k = _key(pair)
    hit = _cache.get(id(pair))
    if hit is not None and hit[0] == k:
        return hit[1]
    try:
        r = analyze(pair) if k is not None else "an object is gone"
    except Exception as exc:
        r = str(exc) or exc.__class__.__name__
    _cache[id(pair)] = (k, r)
    return r


def tolerances(v):
    return {"G0": v.cont_gap_tol, "G1": v.cont_angle_tol, "G2": v.cont_curv_tol / 100.0}


def judge(res, v):
    """{level: (worst value, holds, per-sample failing mask)}; a level holds
    only when the levels below it hold too."""
    tol = tolerances(v)
    vals = {"G0": res["gap"], "G1": res["ang"], "G2": res["crel"]}
    out, below_ok, below_bad = {}, True, np.zeros(len(res["gap"]), bool)
    for lv in LEVELS:
        bad = vals[lv] > tol[lv]
        worst = float(vals[lv].max())
        ok = below_ok and not bad.any()
        below_bad = below_bad | bad
        out[lv] = (worst, ok, below_bad.copy())
        below_ok = ok
    return out


UNITS = {"KILOMETERS": (1000.0, " km"), "METERS": (1.0, " m"), "CENTIMETERS": (0.01, " cm"),
         "MILLIMETERS": (0.001, " mm"), "MICROMETERS": (1e-6, " µm"), "MILES": (1609.344, " mi"),
         "FEET": (0.3048, "'"), "INCHES": (0.0254, '"'), "THOU": (0.0000254, " thou")}


def why_not(res, v, lv):
    """The lower level that makes lv fail although lv's own value is within
    tolerance (G2 ✗ with a fine curvature but a gap), else None."""
    tol = tolerances(v)
    vals = {"G0": res["gap"], "G1": res["ang"], "G2": res["crel"]}
    if vals[lv].max() > tol[lv]:
        return None
    for low in LEVELS[:LEVELS.index(lv)]:
        if vals[low].max() > tol[low]:
            return low
    return None


def fmt(lv, value, context=None):
    if lv == "G0":
        us = (context or bpy.context).scene.unit_settings
        x = value * us.scale_length
        if us.system == 'NONE':
            return f"{x:.6f}"
        unit = us.length_unit
        if unit == 'ADAPTIVE':
            unit = 'INCHES' if us.system == 'IMPERIAL' else 'MILLIMETERS'
        f, suf = UNITS.get(unit, (1.0, " m"))
        return f"{x / f:.5f}{suf}"
    if lv == "G1":
        return f"{value:.3f}°"
    return f"{100.0 * value:.2f}%"


def pair_name(pair):
    a, b = pair["a"], pair["b"]
    if b is None:
        return f"{a[0]} {a[1]}"
    return f"{a[0]} {a[1]} ↔ {b[0]} {b[1]}" if a[0] != b[0] else f"{a[0]} {a[1]} ↔ {b[1]}"


def _same(s, obj, ref):
    return s is not None and s[0] == obj.name and s[1] == ref


def find_pair(obj, ref):
    for i, p in enumerate(PAIRS):
        if _same(p["a"], obj, ref) or _same(p["b"], obj, ref):
            return i
    return None


def add(obj, ref, partner=None):
    """Check an edge (and its partner edge on another surface)."""
    pair = {"a": side_key(obj, ref), "b": side_key(*partner) if partner else None}
    PAIRS.append(pair)
    return pair


def add_selected(context):
    """Check the edges picked in Edge mode (subsel): a joined edge or a seam
    on its own, an open edge with the edge it meets. Returns how many."""
    from . import subsel
    from .ops import _pick_candidates
    picked = subsel.selected_edges(context)
    if not picked:
        return 0
    cands = None
    n = 0
    have = {(p["a"][0], p["a"][1]) for p in PAIRS if p.get("a")}
    for obj, ref in picked:
        if (obj.name, ref) in have:
            continue
        if edge_kind(obj, ref) == "open":
            if cands is None:
                cands = [c for c in _pick_candidates(context, True) if not c[1].startswith("S") and store.is_nurbs(c[0])]
            own = next((c for c in cands if c[0] == obj and c[1] == ref), None)
            partner = find_partner(obj, ref, own[2], cands) if own else None
            if partner is None:
                continue                          # an open edge that meets nothing: nothing to check
            add(obj, ref, partner)
        else:
            add(obj, ref)
        n += 1
    return n


def clear(index=-1):
    if index < 0:
        PAIRS.clear()
        _cache.clear()
    elif index < len(PAIRS):
        _cache.pop(id(PAIRS[index]), None)
        PAIRS.pop(index)


def find_partner(obj, ref, pts, cands):
    """The edge of another surface (or another open edge of the same one) that
    runs along this open edge: most of its samples within 1% of its length."""
    P = np.array([tuple(p) for p in pts], float)
    L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) or 1.0
    tol = 0.01 * L
    best = None
    for o, r, q in cands:
        if (o == obj and r == ref) or r.startswith("S"):
            continue
        if r.startswith("E") and edge_kind(o, r) != "open":
            continue
        Q = np.array([tuple(x) for x in q], float)
        dPQ = np.min(np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2), axis=1)
        dQP = np.min(np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2), axis=1)
        share = max(float(np.mean(dPQ < tol)), float(np.mean(dQP < tol)))
        if share < 0.6:
            continue
        score = (share, -float(np.mean(np.minimum(dPQ, tol))))
        if best is None or score > best[0]:
            best = (score, (o, r))
    return best[1] if best else None


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _v(p):
    return Vector((float(p[0]), float(p[1]), float(p[2])))


def draw_view(ctx, draw_lines):
    """Combs and the coloured edge (POST_VIEW)."""
    v = getattr(ctx.window_manager, "nproto_view", None)
    if v is None or not PAIRS or not v.cont_show_checks:
        return
    import gpu
    gpu.state.depth_test_set('NONE')
    lv = v.cont_level
    tol = tolerances(v)
    for pair in list(PAIRS):
        res = result(pair)
        if isinstance(res, str):
            continue
        J = judge(res, v)
        # the line: red where THIS level fails, orange where only a lower level
        # does (a G2 view over a gap: the curvature is fine, G0 is not), green
        # where everything up to this level holds
        own = {"G0": res["gap"], "G1": res["ang"], "G2": res["crel"]}[lv] > tolerances(v)[lv]
        k = LEVELS.index(lv)
        low = J[LEVELS[k - 1]][2] if k > 0 else np.zeros(len(own), bool)
        P, N, L = res["P"], res["N"], res["L"]
        links = [(i, i + 1) for i in range(len(P) - 1)] + ([(len(P) - 1, 0)] if res.get("closed") else [])
        ok_seg, bad_seg, low_seg = [], [], []
        for i, j in links:
            seg = bad_seg if own[i] or own[j] else low_seg if low[i] or low[j] else ok_seg
            seg.extend([_v(P[i]), _v(P[j])])
        # every level: the DEVIATION between the two sides (gap, normal angle,
        # curvature difference) as a comb standing on the surface (outward
        # normal), scaled to this edge's own largest value so its shape always
        # shows -- also well within tolerance, as in Plasticity. Hairs past the
        # tolerance are red; a faint line marks the tolerance height when it is
        # in range. A deviation far below the tolerance (< 5% of it: noise)
        # stays small instead of being blown up.
        val = {"G0": res["gap"], "G1": res["ang"], "G2": res["crel"]}[lv]
        H = COMB_SPAN * L * v.cont_scale
        norm = max(float(val.max()), 0.05 * tol[lv], 1e-30)
        h = val / norm * H
        tips = P + N * h[:, None]
        own_bad = val > tol[lv]
        hg, hb, env = [], [], []
        for i in range(len(P)):
            (hb if own_bad[i] else hg).extend([_v(P[i]), _v(tips[i])])
        for i, j in links:
            env.extend([_v(tips[i]), _v(tips[j])])
        draw_lines(hg, COMB_HAIR, COMB_HAIR_W)
        draw_lines(hb, COMB_BAD, COMB_ENV_W)
        draw_lines(env, COMB_ENVELOPE, COMB_ENV_W)
        ht = tol[lv] / norm * H
        if ht <= 2.0 * H:
            tl = P + N * ht
            line = []
            for i, j in links:
                line.extend([_v(tl[i]), _v(tl[j])])
            draw_lines(line, COL_TOL, 1.0)
        if lv == "G2" and v.cont_kcombs:
            # optional: each side's own curvature across the edge (outward = convex),
            # scaled to the faces' largest curvature; on a G2 edge they coincide
            s_ = COMB_SPAN * L * v.cont_scale / max(res["kscale"], 1e-12)
            if res["kscale"] * L < 1e-6:
                s_ = 0.0
            ta = P - N * (res["kA"] * s_)[:, None]
            tb = P - N * (res["kB"] * s_)[:, None]
            ha, ea, eb = [], [], []
            for i in range(len(P)):
                ha.extend([_v(P[i]), _v(ta[i])])
            for i, j in links:
                ea.extend([_v(ta[i]), _v(ta[j])])
                eb.extend([_v(tb[i]), _v(tb[j])])
            draw_lines(ha, (0.85, 0.85, 0.85, 0.35), COMB_HAIR_W)
            draw_lines(ea, (0.85, 0.85, 0.85, 0.9), COMB_ENV_W)
            draw_lines(eb, COL_B, COMB_ENV_W)
        draw_lines(ok_seg, COL_OK, 3.5)
        draw_lines(bad_seg, COL_BAD, 3.5)
        draw_lines(low_seg, COL_LOW, 3.5)
    gpu.state.depth_test_set('LESS_EQUAL')


def draw_px(ctx, region, rv3d, box):
    """Labels "G2 ✓ 1.20%" at the middle of every checked edge (POST_PIXEL).
    box(x, y, w, h, color) draws a filled rectangle."""
    import blf
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    v = getattr(ctx.window_manager, "nproto_view", None)
    font = 0
    blf.size(font, 12)
    if PENDING["pts"]:          # a first pick waiting for its partner: say so where you look
        pts = PENDING["pts"]
        q = location_3d_to_region_2d(region, rv3d, pts[len(pts) // 2])
        if q is not None:
            t = PENDING["text"]
            w, h = blf.dimensions(font, t)
            box(q.x + 4, q.y - 22, w + 12, h + 10, (0.08, 0.08, 0.08, 0.85))
            blf.color(font, *draw.COL_PICKS[0])
            blf.position(font, q.x + 10, q.y - 17, 0)
            blf.draw(font, t)
    if v is None or not PAIRS or not v.cont_show_checks:
        return
    lv = v.cont_level
    for pair in list(PAIRS):
        res = result(pair)
        if isinstance(res, str):
            continue
        q = location_3d_to_region_2d(region, rv3d, _v(res["mid"]))
        if q is None:
            continue
        J = judge(res, v)
        worst, ok, _ = J[lv]
        low = why_not(res, v, lv)          # this level is within tolerance, a lower one is not
        mark = COL_OK if ok else (COL_LOW if low is not None else COL_BAD)
        parts = [(lv + " ", LEVEL_COL[lv]), ("✓ " if ok else "✗ ", mark),
                 (fmt(lv, worst, ctx), (0.92, 0.92, 0.92, 1.0))]
        if low is not None:
            parts.append((f"  ({low} ✗ {fmt(low, J[low][0], ctx)})", COL_LOW))
        ws = [blf.dimensions(font, t)[0] for t, _ in parts]
        h = blf.dimensions(font, "G0")[1]
        x, y = q.x + 10, q.y + 8
        box(x - 6, y - 5, sum(ws) + 12, h + 10, (0.08, 0.08, 0.08, 0.85))
        for (t, col), w in zip(parts, ws):
            blf.color(font, *col)
            blf.position(font, x, y, 0)
            blf.draw(font, t)
            x += w


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------

def _drop_pending():
    draw.pick_overlay["picked"] = []
    PENDING["pts"], PENDING["text"] = None, ""


def _redraw(context):
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type in ('VIEW_3D', 'UI') or area.type == 'PROPERTIES':
                area.tag_redraw()


class NPROTO_OT_continuity(bpy.types.Operator):
    """Check G0 / G1 / G2 along edges: click an edge of a joined body, or the edges
    of two surfaces that meet. Labels and combs show where it holds and where not;
    they stay on screen and update while you edit. Click the button again to
    switch it off (and once more to bring the checks back and pick more)"""
    bl_idname = "nproto.continuity"
    bl_label = "Continuity Check"
    bl_options = {'REGISTER'}

    gap_tol: bpy.props.FloatProperty(name="Gap Tolerance", default=1e-3, min=0.0, unit='LENGTH')
    angle_tol: bpy.props.FloatProperty(name="Angle Tolerance", default=1.0, min=0.0)
    curv_tol: bpy.props.FloatProperty(name="Curvature Tolerance %", default=2.0, min=0.0, soft_max=20.0,
                                      description="G2: largest difference of the curvature across the edge, "
                                                  "in percent of the larger one")

    def execute(self, context):
        """Two selected surfaces, no picking (scripts / F3 with a selection)."""
        from .ops_sweep import continuity_report
        import json
        sel = [o for o in context.selected_objects if store.is_nurbs(o)]
        if len(sel) != 2:
            self.report({'WARNING'}, "Select two NURBS surfaces")
            return {'CANCELLED'}
        r = continuity_report(sel[0], sel[1]) or continuity_report(sel[1], sel[0])
        if r is None:
            self.report({'WARNING'}, "The surfaces don't meet")
            return {'CANCELLED'}
        g0 = "G0 ok" if r["gap"] <= self.gap_tol else "G0 FAIL"
        g1 = "G1 ok" if r["angle"] <= self.angle_tol else "G1 FAIL"
        msg = f"{g0} (max gap {r['gap']:.2e}) · {g1} (max normal angle {r['angle']:.3f}°)"
        if "curv_diff" in r:
            g2 = "G2 ok" if r["curv_diff"] * 100.0 <= self.curv_tol else "G2 FAIL"
            rad = lambda k: "flat" if abs(k) < 1e-12 else f"r {1.0 / abs(k):.4g}"
            msg += (f" · {g2} (max curvature difference {100.0 * r['curv_diff']:.2f}%: "
                    f"{rad(r['k_a'])} vs {rad(r['k_b'])})")
        self.report({'INFO'}, msg + f" · {r['points']} points")
        context.scene["nproto_last_continuity"] = json.dumps(r)
        return {'FINISHED'}

    # ---- interactive picking ----
    def invoke(self, context, event):
        from .ops import _pick_candidates
        if active(context):                       # the button again: switch it off
            stop(context)
            _redraw(context)
            self.report({'INFO'}, "Continuity Check off")
            return {'FINISHED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        if add_selected(context):                 # Edge mode: check the picked edges, no picking
            context.window_manager.nproto_view.cont_show_checks = True
            _redraw(context)
            self.report({'INFO'}, f"Continuity Check: {len(PAIRS)} edge(s)")
            return {'FINISHED'}
        self._cands = [c for c in _pick_candidates(context, True)
                       if not c[1].startswith("S") and store.is_nurbs(c[0])]
        if not self._cands:
            self.report({'WARNING'}, "No NURBS edges to check")
            return {'CANCELLED'}
        self._first = None
        self._ties, self._cycle = [], 0
        context.window_manager.nproto_view.cont_show_checks = True
        _drop_pending()
        draw.pick_overlay["hover"] = None
        context.window_manager.modal_handler_add(self)
        TOOL["running"], TOOL["stop"] = True, False
        self._status(context)
        _redraw(context)
        return {'RUNNING_MODAL'}

    def _status(self, context, extra=""):
        from . import hints
        v = context.window_manager.nproto_view
        if self._first is not None:
            msg = f"{self._first[0].name} {self._first[1]}: pick the edge it meets"
        else:
            msg = "click an edge (a joined edge, or a surface edge); click a checked edge to remove it"
        rows = [("Show", [hints.cycle("0 1 2", ["G0", "G1", "G2"], v.cont_level),
                          hints.key("+ -", "comb size")]),
                ("Edges", [hints.key("Tab", "cycle overlapping", on=True if len(self._ties) > 1 else None),
                           hints.key("Backspace", "undo"), hints.key("C", "clear all")])]
        hints.show(context, "Continuity", [msg, extra.strip(" ·")], rows,
                   [("MOUSE_LMB", "Pick edge"), ("EVENT_RETURN", "Done"), ("EVENT_ESC", "Done")])

    def _end(self, context):
        TOOL["running"], TOOL["stop"] = False, False
        _drop_pending()
        draw.pick_overlay["hover"] = None
        if context.area is not None:
            from . import hints
            hints.clear(context)
        _redraw(context)

    def modal(self, context, event):
        from .ops import PickMixin, _near_candidates
        v = context.window_manager.nproto_view
        if TOOL["stop"]:                          # switched off from its button
            self._end(context)
            return {'CANCELLED'}
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event):
            return {'PASS_THROUGH'}
        from . import hints
        if hints.handle(event):
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE':
            self._hover(context, event)
            return {'RUNNING_MODAL'}
        if event.value != 'PRESS':
            return {'RUNNING_MODAL'}
        if event.type == 'TAB' and self._ties:
            self._cycle += 1
            draw.pick_overlay["hover"] = self._ties[self._cycle % len(self._ties)][2]
            _redraw(context)
        elif event.type in {'ZERO', 'NUMPAD_0', 'ONE', 'NUMPAD_1', 'TWO', 'NUMPAD_2'}:
            v.cont_level = {"ZERO": "G0", "NUMPAD_0": "G0", "ONE": "G1", "NUMPAD_1": "G1"}.get(event.type, "G2")
            self._status(context)
            _redraw(context)
        elif event.type in {'NUMPAD_PLUS', 'EQUAL', 'PLUS'}:
            v.cont_scale = min(v.cont_scale * 1.25, 50.0)
        elif event.type in {'NUMPAD_MINUS', 'MINUS'}:
            v.cont_scale = max(v.cont_scale / 1.25, 0.02)
        elif event.type == 'C':
            clear()
            self._first = None
            _drop_pending()
            self._status(context)
            _redraw(context)
        elif event.type == 'BACK_SPACE':
            if self._first is not None:
                self._first = None
                _drop_pending()
            elif PAIRS:
                clear(len(PAIRS) - 1)
            self._status(context)
            _redraw(context)
        elif event.type == 'LEFTMOUSE':
            hit = self._hover(context, event)
            if hit is not None:
                self._pick(context, hit)
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE', 'ESC', 'RIGHTMOUSE'}:
            self._end(context)
            return {'FINISHED'}
        return {'RUNNING_MODAL'}

    def _hover(self, context, event):
        from .ops import _near_candidates
        ties = _near_candidates(context, event, self._cands)
        key = [(c[0].name, c[1]) for c in ties]
        if key != [(c[0].name, c[1]) for c in self._ties]:
            self._ties, self._cycle = ties, 0
        cur = self._ties[self._cycle % len(self._ties)] if self._ties else None
        draw.pick_overlay["hover"] = cur[2] if cur else None
        context.area.tag_redraw()
        return cur

    def _pick(self, context, hit):
        obj, ref, pts = hit
        if self._first is not None:
            if (obj, ref) == self._first[:2]:
                return
            add(self._first[0], self._first[1], (obj, ref))
            self._first = None
            _drop_pending()
            self._report_last(context)
            return
        k = find_pair(obj, ref)
        if k is not None:                       # a checked edge: take it off
            clear(k)
            self._status(context, "   (removed)")
            _redraw(context)
            return
        if edge_kind(obj, ref) in ("shared", "seam"):     # one click: its two sides are known
            add(obj, ref)
            self._report_last(context)
            return
        partner = find_partner(obj, ref, pts, self._cands)
        if partner is not None:
            add(obj, ref, partner)
            self._report_last(context)
            return
        self._first = (obj, ref, pts)
        draw.pick_overlay["picked"] = [pts]
        PENDING["pts"], PENDING["text"] = pts, "now click the edge this one meets"
        self.report({'INFO'}, f"{obj.name} {ref} is an open edge and nothing runs along it: "
                              "click the edge it meets (Backspace: drop it)")
        self._status(context)
        _redraw(context)

    def _report_last(self, context):
        v = context.window_manager.nproto_view
        res = result(PAIRS[-1])
        if isinstance(res, str):
            self.report({'WARNING'}, f"{pair_name(PAIRS[-1])}: {res}")
            clear(len(PAIRS) - 1)
        else:
            J = judge(res, v)
            self.report({'INFO'}, f"{pair_name(PAIRS[-1])}: " + " · ".join(
                f"{lv} {'ok' if J[lv][1] else 'FAILS'} {fmt(lv, J[lv][0], context)}" for lv in LEVELS))
        self._status(context)
        _redraw(context)


class NPROTO_OT_continuity_clear(bpy.types.Operator):
    """Take a checked edge off (or all of them)"""
    bl_idname = "nproto.continuity_clear"
    bl_label = "Clear Continuity Checks"
    bl_options = {'INTERNAL'}

    index: IntProperty(default=-1)

    def execute(self, context):
        clear(self.index)
        _redraw(context)
        return {'FINISHED'}


def draw_panel(col, context):
    """The checked edges, the level shown and the tolerances (Analyze panel)."""
    v = context.window_manager.nproto_view
    col.operator("nproto.continuity", icon='DRIVER_DISTANCE', depress=active(context))
    if not PAIRS or not v.cont_show_checks:
        return
    box = col.box()
    row = box.row(align=True)
    row.prop(v, "cont_level", expand=True)
    row = box.row(align=True)
    row.prop(v, "cont_scale")
    row.prop(v, "cont_samples")
    if v.cont_level == 'G2':
        box.prop(v, "cont_kcombs")
    need = max([r_["recommended"] for r_ in (result(p_) for p_ in PAIRS) if not isinstance(r_, str)] or [0])
    if need > v.cont_samples:
        box.label(text=f"A dense surface: {need} samples show its true comb", icon='INFO')
    for i, pair in enumerate(PAIRS):
        res = result(pair)
        r = box.row(align=True)
        if isinstance(res, str):
            r.label(text=f"{pair_name(pair)}: {res}", icon='ERROR')
        else:
            J = judge(res, v)
            r.label(text=pair_name(pair), icon='CHECKMARK' if J[v.cont_level][1] else 'ERROR')
            r.label(text="  ".join(f"{lv}{'✓' if J[lv][1] else '✗'}" for lv in LEVELS))
        r.operator("nproto.continuity_clear", text="", icon='X', emboss=False).index = i
    sub = box.column(align=True)
    sub.prop(v, "cont_gap_tol")
    sub.prop(v, "cont_angle_tol")
    sub.prop(v, "cont_curv_tol")
    box.operator("nproto.continuity_clear", text="Clear All", icon='TRASH').index = -1


classes = (NPROTO_OT_continuity, NPROTO_OT_continuity_clear)
