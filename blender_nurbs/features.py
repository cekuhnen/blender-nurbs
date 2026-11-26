"""Feature builders: pure functions (inputs, params) -> kernel geometry.

Every modelling command is registered here so construction history can rebuild
it later. An input is (object, ref) -- see geom.py for the ref strings.
"""

import json
import math

import numpy as np
from mathutils import Vector

from . import geom, store
from .kernel import load as _load_kernel

try:            # keep registrations from other modules across an add-on reload
    BUILDERS
except NameError:
    BUILDERS = {}


def builder(name):
    def deco(fn):
        BUILDERS[name] = fn
        return fn
    return deco


def _k():
    sm = _load_kernel()
    if sm is None:
        raise RuntimeError("USD BRep kernel not loaded")
    return sm, sm.blender_ext


# --------------------------------------------------------------------------
# Input resolution
# --------------------------------------------------------------------------

def is_edge_ref(ref):
    return ref in ("U0", "U1", "V0", "V1")


def is_brep_edge_ref(ref):
    return ref.startswith("E")


def is_surface_edge(ref):
    """An edge of a NURBS surface: untrimmed boundary (U0..V1) or trimmed / B-rep edge (E#)."""
    return is_edge_ref(ref) or is_brep_edge_ref(ref)


def input_curve(obj, ref):
    """Kernel curve in world space for a curve spline or a surface edge (only
    the stretch a loft's slid end covers, while EDGE_RANGES names it)."""
    c = _input_curve_full(obj, ref)
    rng = edge_range(obj, ref)
    return c if rng is None else _sub_curve(c, *rng)


def _input_curve_full(obj, ref):
    sm, ext = _k()
    if is_brep_edge_ref(ref) or is_edge_ref(ref):
        # EXACT edge curves (an interpolation through sample points would make
        # caps / lofts whose edges miss their neighbour by the chord error;
        # Join then splits edges and leaves sliver faces in the gap)
        try:
            return _exact_edge_curve(obj, ref)
        except Exception:
            pass
    if is_brep_edge_ref(ref):
        poly = store.display(obj)["edges"][int(ref[1:])]
        M = geom.world_matrix(obj)
        return sm.create_curve_interp_points([tuple(M @ Vector(p)) for p in poly])
    if is_edge_ref(ref):
        surf = store.get_surfaces(obj)[0]
        M = geom.world_matrix(obj)
        pts = [tuple(M @ Vector(p)) for p in geom.boundary_points(surf, ref, 48)]
        return sm.create_curve_interp_points(pts)
    idx = int(ref[1:]) if ref.startswith("S") else 0
    spline = obj.data.splines[idx]
    return geom.spline_to_kernel_curve(spline, geom.world_matrix(obj))


def _exact_edge_curve(obj, ref):
    """World-space exact kernel curve of a surface edge: the boundary CV row of
    an untrimmed surface (U0/U1/V0/V1), or the B-rep edge's own curve cut to its
    range (E#)."""
    sm, ext = _k()
    if is_brep_edge_ref(ref):
        from .square import _edge_curve
        d = ext.curve_cvs(_edge_curve(store.get_brep(obj).edges()[int(ref[1:])]))
        P, U, W, deg = d["points"], list(d["knots"]), list(d["weights"]) if d.get("weights") else None, d["degree"]
    else:
        d = ext.surface_cvs(store.get_surfaces(obj)[0])
        nu, nv = d["count_u"], d["count_v"]
        Pg = np.array(d["points"], float).reshape(nu, nv, 3)
        Wg = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else None
        if ref in ("U0", "U1"):
            i = 0 if ref == "U0" else nu - 1
            P, W, U, deg = Pg[i], (Wg[i] if Wg is not None else None), list(d["knots_v"]), d["degree_v"]
        else:
            j = 0 if ref == "V0" else nv - 1
            P, W, U, deg = Pg[:, j], (Wg[:, j] if Wg is not None else None), list(d["knots_u"]), d["degree_u"]
        W = None if W is None else [float(x) for x in W]
    M = np.array(geom.world_matrix(obj), dtype=float)
    Pw = [tuple(map(float, M[:3, :3] @ np.asarray(p, float) + M[:3, 3])) for p in P]
    if W is not None and max(abs(w - 1.0) for w in W) < 1e-12:
        W = None
    return ext.curve_from_cvs(int(deg), Pw, [float(k) for k in U], W)


def brep_edge_iso(obj, ref, n=9):
    """(face surface, iso ref, lo, hi) when B-rep edge `ref` runs along a side
    of its face's surface: "U0" / "U1" / "V0" / "V1", and the stretch of that
    side it covers (lo..hi, 0..1 along the side). That is an untrimmed side of
    a trimmed face, whole or cut short by a trim. None for a trim curve, an
    edge between two faces, or an edge off the surface."""
    sm, _ = _k()
    try:
        edges = store.get_brep(obj).edges()
        e = edges[int(ref[1:])]
        faces = e.faces()
    except Exception:
        return None
    if len(faces) != 1:
        return None
    s = faces[0].surface()
    (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
    t0, t1 = e.parameter_range()
    pts = np.array([e.point_at(t0 + (t1 - t0) * k / (n - 1)) for k in range(n)], float)
    size = float(np.ptp(pts, axis=0).max()) or 1.0
    uv = []
    for q in pts:
        cp = sm.surface_closest_point(s, tuple(map(float, q)))
        w = cp[1][0]
        u, v = float(w.x), float(w.y)
        if np.linalg.norm(np.array(s.evaluate(u, v), float) - q) > 1e-7 * size:
            return None
        uv.append((u, v))
    uv = np.array(uv)
    for iso, col, val in (("U0", 0, u0), ("U1", 0, u1), ("V0", 1, v0), ("V1", 1, v1)):
        span = (u1 - u0) if col == 0 else (v1 - v0)
        if np.all(np.abs(uv[:, col] - val) <= 1e-6 * span):
            a, b = (v0, v1) if col == 0 else (u0, u1)
            w = (uv[:, 1 - col] - a) / (b - a)
            lo, hi = float(w.min()), float(w.max())
            lo = 0.0 if lo < 1e-6 else lo
            hi = 1.0 if hi > 1.0 - 1e-6 else hi
            return (s, iso, lo, hi) if hi - lo > 1e-6 else None
    return None


def _iso_side(ext, surf, ref, R, T):
    """Exact rows of the surface's side `ref` (U0 / U1 / V0 / V1), world space."""
    d = ext.surface_cvs(surf)
    if d.get("weights") and max(abs(w - 1.0) for w in d["weights"]) > 1e-12:
        return None
    nu, nv = d["count_u"], d["count_v"]
    G = np.array(d["points"], float).reshape(nu, nv, 3)
    if ref in ("V0", "V1"):
        G = G.transpose(1, 0, 2)
        Ua, deg_a, Ue, deg_e = list(d["knots_v"]), d["degree_v"], list(d["knots_u"]), d["degree_u"]
    else:
        Ua, deg_a, Ue, deg_e = list(d["knots_u"]), d["degree_u"], list(d["knots_v"]), d["degree_v"]
    end = ref[1] == "1"
    if end:                      # outward = increasing parameter at the far end: mirror it to the start
        G = G[::-1]
        lo, hi = Ua[0], Ua[-1]
        Ua = [lo + hi - u for u in reversed(Ua)]
    p = deg_a
    P0 = G[0]
    D = DD = DDD = None
    if p >= 1 and len(G) > 1:
        d1 = p / (Ua[p + 1] - Ua[1])
        Su = d1 * (G[1] - G[0])                  # dS/du at the edge (u increasing inward)
        D = -Su                                  # outward
        DD = np.zeros_like(P0)
        if p >= 2 and len(G) > 2:
            a1 = (G[1] - G[0]) / (Ua[p + 1] - Ua[1])
            a2 = (G[2] - G[1]) / (Ua[p + 2] - Ua[2])
            DD = p * (p - 1) / (Ua[p + 1] - Ua[2]) * (a2 - a1)
        # third: odd, so outward is minus the inward one (zero below degree 3)
        w3 = geom.start_coeffs(Ua, p, 3)
        DDD = -sum(w * G[i] for i, w in enumerate(w3)) if p >= 3 and len(G) > 3 else np.zeros_like(P0)
    lo, hi = Ue[0], Ue[-1]
    knots = [(u - lo) / (hi - lo) for u in Ue]
    tw = lambda A: None if A is None else A @ R.T
    return {"deg": int(deg_e), "knots": knots, "P": P0 @ R.T + T, "D": tw(D), "DD": tw(DD), "DDD": tw(DDD)}


def exact_side(obj, ref):
    """Exact CV rows of a loft input on its own layout: {"deg", "knots" (0..1),
    "P", "D", "DD"} in world space; D / DD are the outward cross first / second
    derivative rows of a surface edge (None for a curve). A B-rep edge counts
    when it runs along a side of its face's surface (an untrimmed side of a
    trimmed face): that side's rows, cut exactly to the edge's stretch. None
    when not available exactly (a trim curve, rational)."""
    sm, ext = _k()
    M = np.array(geom.world_matrix(obj), dtype=float)
    R, T = M[:3, :3], M[:3, 3]
    rng = edge_range(obj, ref)
    if is_edge_ref(ref):
        if not store.is_natural(obj):
            return None
        surf = store.get_surfaces(obj)[0]
        side = _iso_side(ext, surf, ref, R, T)
        if side is None or rng is None:
            return side
        return geom.sub_side(side, *_domain_share(surf, ref, rng))
    if is_brep_edge_ref(ref):
        hit = brep_edge_iso(obj, ref)
        if hit is None:
            return None
        surf, iso, lo, hi = hit
        if rng is not None:
            lo, hi = _iso_stretch(obj, ref, surf, iso, rng)
        side = _iso_side(ext, surf, iso, R, T)
        if side is None or (lo == 0.0 and hi == 1.0):
            return side
        return geom.sub_side(side, lo, hi)
    d = ext.curve_cvs(input_curve(obj, ref))       # already world space
    if d.get("weights") and max(abs(w - 1.0) for w in d["weights"]) > 1e-12:
        return None
    U = list(d["knots"])
    lo, hi = U[0], U[-1]
    return {"deg": int(d["degree"]), "knots": [(u - lo) / (hi - lo) for u in U],
            "P": np.array(d["points"], float), "D": None, "DD": None}


def _same_layout(sa, sb):
    return (sa["deg"] == sb["deg"] and len(sa["knots"]) == len(sb["knots"])
            and np.allclose(sa["knots"], sb["knots"], atol=1e-9))


# --------------------------------------------------------------------------
# Slid ends: a two-sided loft over a stretch of each side
# --------------------------------------------------------------------------

# {(object name, ref): (t0, t1)} while a loft with slid ends builds or is
# measured: the stretch of that input it spans, in the parameter of the
# input's own exact curve (a surface edge: its surface's parameter along it;
# a B-rep edge: the edge's). input_curve, exact_side, input_samples and
# square.sample_side all cut the input to it, so every path of the loft (the
# exact rows, the sampled G1 / G2 fits, Align Edges / Surface) and the
# continuity check see the same shorter edge.
EDGE_RANGES = {}
SLIDE_KEYS = ("a_from", "a_to", "b_from", "b_to")
SLIDE_MIN = 0.01          # a slid side spans at least this share of its length


def edge_range(obj, ref):
    if not EDGE_RANGES:
        return None
    return EDGE_RANGES.get((getattr(obj, "name", None), ref))


def _sub_curve(c, t0, t1):
    """The kernel curve c cut exactly to t0..t1 of its parameter (0..1 after)."""
    sm, ext = _k()
    d = ext.curve_cvs(c)
    U = [float(k) for k in d["knots"]]
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else None
    a, b = U[0], U[-1]
    f = lambda t: min(max((t - a) / (b - a), 0.0), 1.0)
    # rational: homogeneous points, the weight carried as the first column of a second row
    side = {"deg": int(d["degree"]), "knots": [f(k) for k in U],
            "P": P if w is None else P * w[:, None],
            "D": None if w is None else np.c_[w, np.zeros_like(w), np.zeros_like(w)]}
    cut = geom.sub_side(side, f(t0), f(t1))
    Q = cut["P"]
    W = None
    if w is not None:
        W = cut["D"][:, 0]
        Q = Q / W[:, None]
        W = None if np.allclose(W, 1.0, atol=1e-12) else [float(x) for x in W]
    return ext.curve_from_cvs(int(cut["deg"]), [tuple(map(float, q)) for q in Q],
                              [float(k) for k in cut["knots"]], W)


def _domain_share(surf, ref, rng):
    """t0..t1 of a natural edge's surface parameter as shares of its domain."""
    a, b = surf.v_domain() if ref in ("U0", "U1") else surf.u_domain()
    t0, t1 = sorted(((rng[0] - a) / (b - a), (rng[1] - a) / (b - a)))
    return max(t0, 0.0), min(t1, 1.0)


def _iso_stretch(obj, ref, surf, iso, rng):
    """lo..hi (0..1 along the surface's side `iso`) of B-rep edge ref's stretch rng."""
    sm, _ = _k()
    e = store.get_brep(obj).edges()[int(ref[1:])]
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    out = []
    for t in rng:
        w = sm.surface_closest_point(surf, tuple(map(float, e.point_at(t))))[1][0]
        out.append((float(w.y) - v0) / (v1 - v0) if iso in ("U0", "U1") else (float(w.x) - u0) / (u1 - u0))
    lo, hi = sorted(out)
    return max(lo, 0.0), min(hi, 1.0)


def side_slides(p):
    """((from, to) of side A, of side B): the share of each side's length the loft spans."""
    out = []
    for s in "ab":
        lo = min(max(float(p.get(f"{s}_from", 0.0)), 0.0), 1.0)
        hi = min(max(float(p.get(f"{s}_to", 1.0)), 0.0), 1.0)
        out.append(tuple(sorted((lo, hi))))
    return out


def has_slides(inputs, p):
    """Does this loft span only a stretch of a side? (Two sides, no rails, open.)"""
    if p.get("closed") or loft_rail_count(inputs, p) or len(loft_sides(inputs, p)) != 2:
        return False
    return any(lo > 1e-6 or hi < 1.0 - 1e-6 for lo, hi in side_slides(p))


def _arc_table(c, n=512):
    """(parameters, arc length to each) along a kernel curve: chord sums at
    two spacings, Richardson-extrapolated (fourth order: 1e-9 of the length)."""
    t0, t1 = c.parameter_range()
    tt = np.linspace(t0, t1, 2 * n + 1)
    P = np.array([c.evaluate(float(t)) for t in tt], float)
    fine = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])[::2]
    coarse = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[::2], axis=0), axis=1))])
    return tt[::2], (4.0 * fine - coarse) / 3.0


def _arc_param(c, tt, s, target, k=64):
    """The parameter at arc length `target` along c (its _arc_table tt, s): the
    table's step around it resampled k times, so the point is exact to ~1e-10
    of the length (linear in the table alone it was off by 1e-6)."""
    if target <= s[0]:
        return float(tt[0])
    if target >= s[-1]:
        return float(tt[-1])
    i = int(min(max(np.searchsorted(s, target) - 1, 0), len(s) - 2))
    ts = np.linspace(tt[i], tt[i + 1], 2 * k + 1)
    P = np.array([c.evaluate(float(t)) for t in ts], float)
    fine = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])[::2]
    coarse = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[::2], axis=0), axis=1))])
    loc = (4.0 * fine - coarse) / 3.0
    loc = s[i] + loc * ((s[i + 1] - s[i]) / (loc[-1] or 1.0))     # agree with the table at the step's ends
    return float(np.interp(target, loc, ts[::2]))


def _slide_side(side, lo, hi, name):
    """[(obj, ref)] of the pieces of a side that the stretch lo..hi of its
    length (along the side as chained) covers, and {(name, ref): (t0, t1)}
    of the pieces it cuts."""
    chain = chain_side(side)
    curves = [input_curve(o, r) for o, r, _ in chain]
    tabs = [_arc_table(c) for c in curves]
    total = sum(s[-1] for _, s in tabs) or 1.0
    a, b = lo * total, hi * total
    eps = 1e-9 * total
    pieces, rng, acc = [], {}, 0.0
    for (o, r, rev), c, (tt, s) in zip(chain, curves, tabs):
        L = float(s[-1])
        f0, f1 = max(a - acc, 0.0), min(b - acc, L)       # along the side, within this piece
        acc += L
        if f1 - f0 <= eps:
            continue
        if rev:
            f0, f1 = L - f1, L - f0
        pieces.append((o, r))
        if f0 > eps or f1 < L - eps:
            rng[(o.name, r)] = (_arc_param(c, tt, s, f0), _arc_param(c, tt, s, f1))
    if not pieces:
        raise ValueError(f"side {name} spans nothing: its From and To must differ")
    return pieces, rng


def _b_runs_back(sides, p):
    """Does side B run the other way from A (so its From / To count from its far end)?"""
    if not p.get("auto_align", True):
        return False
    pa = geom.sample_curve(side_curve(sides[0]), 16)
    pb = geom.sample_curve(side_curve(sides[1]), 16)
    return bool(geom.should_flip(pa, pb))


def can_slide(inputs, p):
    """Can this loft's ends slide (two sides, no rails, open)?"""
    return not p.get("closed") and not loft_rail_count(inputs, p) and len(loft_sides(inputs, p)) == 2


def slide_tracks(inputs, p, n=401):
    """[(points, runs_back)] of side A and B, whole (not slid): n points evenly
    spaced along the side's length (world), and whether its From / To count
    from the far end of them (side B running against A)."""
    saved = dict(EDGE_RANGES)
    EDGE_RANGES.clear()
    try:
        sides = loft_sides(inputs, p)
        back = _b_runs_back(sides, p)
        out = []
        for i, sd in enumerate(sides):
            c = side_curve(sd)
            tt, s = _arc_table(c)
            ts = np.interp(np.linspace(0.0, s[-1], n), s, tt)
            out.append((np.array([c.evaluate(float(t)) for t in ts], float), back and i == 1))
        return out
    finally:
        EDGE_RANGES.clear()
        EDGE_RANGES.update(saved)


def slide_inputs(inputs, p):
    """(inputs, p, ranges) of a two-sided loft whose ends slide along its
    sides (From / To, shares of each side's length; B counted from the same
    end as A): the inputs cut to the pieces the stretches cover, p's side
    counts to match, and the stretches of the pieces cut (EDGE_RANGES).
    The inputs as they are when nothing slides."""
    if not has_slides(inputs, p):
        return inputs, p, {}
    saved = dict(EDGE_RANGES)
    EDGE_RANGES.clear()
    try:
        sides = loft_sides(inputs, p)
        (la, ha), (lb, hb) = side_slides(p)
        for nm, lo, hi in (("A", la, ha), ("B", lb, hb)):
            if hi - lo < SLIDE_MIN:
                raise ValueError(f"side {nm}: From and To are too close (at least {SLIDE_MIN:g} of its length)")
        if _b_runs_back(sides, p):
            lb, hb = 1.0 - hb, 1.0 - lb
        sa, ra = _slide_side(sides[0], la, ha, "A")
        sb, rb = _slide_side(sides[1], lb, hb, "B")
    finally:
        EDGE_RANGES.clear()
        EDGE_RANGES.update(saved)
    if set(ra) & set(rb):
        raise ValueError("a slid loft cannot take the same edge on both sides")
    return sa + sb, dict(p, sides=[len(sa), len(sb)]), {**ra, **rb}


class slid_ends:
    """with slid_ends(inputs, p) as (inputs, p): the loft's inputs and params
    cut to its slid stretches, EDGE_RANGES set while inside."""

    def __init__(self, inputs, p):
        self.ins, self.p, self.rng = slide_inputs(inputs, p)

    def __enter__(self):
        self.saved = dict(EDGE_RANGES)
        EDGE_RANGES.update(self.rng)
        return self.ins, self.p

    def __exit__(self, *exc):
        EDGE_RANGES.clear()
        EDGE_RANGES.update(self.saved)
        return False


def slid_scope(obj):
    """slid_ends of obj's loft history (nothing slid for anything else)."""
    h = history(obj)
    if h is None or h.get("op") != "loft":
        return slid_ends([], {})
    try:
        return slid_ends(h["inputs"], h["params"])
    except Exception as exc:
        print("NURBS loft slid ends:", getattr(obj, "name", obj), exc)
        return slid_ends([], {})


# --------------------------------------------------------------------------
# Loft sides: one or more edges / curves joined end to end
# --------------------------------------------------------------------------

def loft_sides(inputs, p):
    """The inputs grouped into loft sides: p["sides"] holds how many inputs
    (in order) each side takes; without it every input is a side of its own."""
    groups = p.get("sides") or []
    if not groups or sum(int(g) for g in groups) != len(inputs) or any(int(g) < 1 for g in groups):
        return [[x] for x in inputs]
    out, i = [], 0
    for g in groups:
        out.append(list(inputs[i:i + int(g)]))
        i += int(g)
    return out


def _curve_ends(c):
    t0, t1 = c.parameter_range()
    return np.array(c.evaluate(t0), float), np.array(c.evaluate(t1), float)


def chain_side(side, mixed=False):
    """The pieces of a side in order along it: [(obj, ref, reversed)], each
    piece's end on the next one's start. Raises when they do not touch.
    mixed: curves and surface edges may share the chain (Extrude's profile)."""
    if len(side) == 1:
        return [(side[0][0], side[0][1], False)]
    kinds = {is_surface_edge(r) and o.type != 'CURVE' for o, r in side}
    if len(kinds) > 1 and not mixed:
        raise ValueError("a loft side is either surface edges or curves, not both")
    ends = [_curve_ends(input_curve(o, r)) for o, r in side]
    allp = np.vstack([e for pair in ends for e in pair])
    tol = 1e-4 * (float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0)

    def partner(q, skip):
        return min((float(np.linalg.norm(q - e)), j) for j, pair in enumerate(ends) if j != skip for e in pair)

    start, rev = 0, False
    for j, (a, b) in enumerate(ends):        # an open chain starts at a free end
        if partner(a, j)[0] > tol:
            start, rev = j, False
            break
        if partner(b, j)[0] > tol:
            start, rev = j, True
            break
    order = [(start, rev)]
    used = {start}
    cur = ends[start][0 if rev else 1]
    while len(used) < len(side):
        d, j, r = min((float(np.linalg.norm(cur - e)), j, k == 1)
                      for j, pair in enumerate(ends) if j not in used for k, e in enumerate(pair))
        if d > tol:
            raise ValueError("the edges of a loft side must touch end to end")
        order.append((j, r))
        used.add(j)
        cur = ends[j][0 if r else 1]
    return [(side[j][0], side[j][1], r) for j, r in order]


def side_curve(side, mixed=False):
    """One exact kernel curve along a whole loft side (world space)."""
    sm, ext = _k()
    chain = chain_side(side, mixed)
    if len(chain) == 1:
        o, r, _ = chain[0]
        return input_curve(o, r)
    items, speeds, prev_w = [], [], None
    for o, r, rev in chain:
        c = input_curve(o, r)
        if rev:
            c = geom.reverse_curve(c)
        d = ext.curve_cvs(c)
        P = np.array(d["points"], float)
        w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
        if prev_w is not None and w[0] > 0:
            w = w * (prev_w / w[0])            # same weight at the joint (a rational curve is unchanged)
        prev_w = w[-1]
        items.append((int(d["degree"]), d["knots"], np.c_[P * w[:, None], w]))
        speeds.append(geom.end_speeds(int(d["degree"]), d["knots"], P))
    q, U, A = geom.concat_arrays(items, geom.speed_widths(speeds))
    W = A[:, 3]
    P = A[:, :3] / W[:, None]
    Wl = None if np.allclose(W, 1.0, atol=1e-12) else [float(x) for x in W]
    return ext.curve_from_cvs(int(q), [tuple(map(float, x)) for x in P], [float(k) for k in U], Wl)


JOINT_EPS = 0.02     # a crease joint is blended over this share of each piece next to it
LOFT_NOTES = []      # what the last loft build wants the user to know (creased joints)


def _localise_end(s, at_end, eps=JOINT_EPS):
    """Insert a knot eps from one end of an exact side, so its end CV rows
    (the joint's, to be averaged with the next piece's) only reach that far."""
    from .kernel.classa import insert_knot
    keys, A = geom._side_array(s)
    U = [float(k) for k in s["knots"]]
    p = int(s["deg"])
    t = 1.0 - eps if at_end else eps
    inner = U[p + 1:-p - 1]
    near = (inner[-1] if at_end else inner[0]) if inner else (0.0 if at_end else 1.0)
    if (at_end and near >= t) or (not at_end and near <= t):
        return s                     # a knot already that close
    A, U = insert_knot(A, U, p, t)
    return geom._side_from(keys, p, U, np.asarray(A, float))


def side_exact(side, g3=False):
    """Exact rows of a whole loft side ({"deg", "knots", "P", "D", "DD", "DDD"}, see
    exact_side) with the pieces joined end to end, or None. The cross
    derivatives of each piece are scaled by a constant (a reparameterisation
    of its face: G1 / G2 stay exact) so they meet at the joints. Where they
    cannot meet (the faces are creased there, or their curvature jumps) the
    joint's rows are averaged over a short stretch (JOINT_EPS of each piece):
    the loft holds exactly along everything else. "creases" lists those
    joints as (angle in degrees, point). g3: the third rows must meet too
    (else they are left out, as for a G0 - G2 loft)."""
    chain = chain_side(side)
    sides = []
    for o, r, rev in chain:
        s = exact_side(o, r)
        if s is None:
            return None
        sides.append(geom.reverse_side(s) if rev else s)
    if len(sides) == 1:
        sides[0]["creases"] = []
        return sides[0]
    has_d = sides[0]["D"] is not None
    if has_d != all(s["D"] is not None for s in sides):
        return None
    has_dd = has_d and all(s["DD"] is not None for s in sides)
    has_ddd = g3 and has_dd and all(s.get("DDD") is not None for s in sides)
    creases = []
    scales = [1.0]
    bad = [False] * (len(sides) - 1)
    for i, (s0, s1) in enumerate(zip(sides[:-1], sides[1:])):
        if not has_d:
            break
        a, b = scales[-1] * s0["D"][-1], s1["D"][0]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-15 or nb < 1e-15:
            return None
        c = na / nb
        scales.append(c)
        cosang = float(np.dot(a, b)) / (na * nb)
        if cosang < np.cos(1e-4):
            bad[i] = True
            creases.append((float(np.degrees(np.arccos(max(-1.0, min(1.0, cosang))))), s0["P"][-1].copy()))
        elif has_dd:
            dda, ddb = scales[-2] ** 2 * s0["DD"][-1], c ** 2 * s1["DD"][0]
            ref = max(float(np.linalg.norm(dda)), float(np.linalg.norm(ddb)), na * na * 1e-9)
            if float(np.linalg.norm(dda - ddb)) > 1e-6 * ref:
                bad[i] = True          # curvature jumps here: G2 is local to each piece
            elif has_ddd:
                d3a, d3b = scales[-2] ** 3 * s0["DDD"][-1], c ** 3 * s1["DDD"][0]
                ref = max(float(np.linalg.norm(d3a)), float(np.linalg.norm(d3b)), na ** 3 * 1e-9)
                if float(np.linalg.norm(d3a - d3b)) > 1e-6 * ref:
                    bad[i] = True      # the curvature rate jumps: G3 is local to each piece
    sides = [dict(s) for s in sides]
    for i, flag in enumerate(bad):
        if flag:
            sides[i] = _localise_end(sides[i], True)
            sides[i + 1] = _localise_end(sides[i + 1], False)
    keys = ["P"] + (["D"] if has_d else []) + (["DD"] if has_dd else []) + (["DDD"] if has_ddd else [])
    items, speeds = [], []
    for s, k in zip(sides, scales):
        cols = [s["P"]] + ([k * s["D"]] if has_d else []) + ([k * k * s["DD"]] if has_dd else []) + \
            ([k ** 3 * s["DDD"]] if has_ddd else [])
        items.append((s["deg"], s["knots"], np.hstack(cols)))
        speeds.append(geom.end_speeds(s["deg"], s["knots"], s["P"]))
    q, U, A = geom.concat_arrays(items, geom.speed_widths(speeds))
    out = geom._side_from(keys, q, U, A)
    out["creases"] = creases
    return out


def _piece_lengths(chain):
    lens = []
    for o, r, _ in chain:
        P = geom.sample_curve(input_curve(o, r), 16)
        lens.append(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
    return lens


def side_sample_count(side, n):
    """Samples a side wants: n along its longest piece, the others by length."""
    chain = chain_side(side)
    if len(chain) == 1:
        return n
    lens = _piece_lengths(chain)
    big = max(lens) or 1.0
    return sum(max(8, int(round(n * L / big))) for L in lens) - (len(lens) - 1)


def side_samples(side, n):
    """EdgeSample along a whole loft side, exactly n samples: the pieces'
    samples end to end, each piece getting its share by length."""
    chain = chain_side(side)
    if len(chain) == 1:
        o, r, _ = chain[0]
        return input_samples(o, r, n)
    lens = _piece_lengths(chain)
    k = len(chain)
    total = n + k - 1                       # joints are sampled twice, then dropped once
    tot_len = sum(lens) or 1.0
    counts = [max(4, int(round(total * L / tot_len))) for L in lens]
    counts[int(np.argmax(lens))] += total - sum(counts)
    if min(counts) < 2:
        raise ValueError("too few samples for this loft side: raise Samples")
    parts, raw = [], []
    for (o, r, rev), m in zip(chain, counts):
        es = input_samples(o, r, m)
        raw.append(es)
        parts.append(es.reversed() if rev else es)
    cat = lambda key: None if getattr(parts[0], key) is None else np.vstack(
        [getattr(parts[0], key)] + [getattr(e, key)[1:] for e in parts[1:]])
    P, D, DD = cat("P"), cat("D"), cat("DD")
    if D is not None:
        # A trimmed edge's samples carry the unit direction across the edge,
        # square to the edge's tangent -- taken from its neighbouring samples,
        # one-sided at its ends. Where two edges meet that leaves a kink in
        # the row (each end leans its own way: the spoon's two halves 1.2°
        # apart) and the fit wobbled through it. Square them again to the
        # tangent of the whole side (second order, across the joints): same
        # tangent plane, no kink. (Natural edges carry the face's own
        # derivative, which G2 pairs with its second one: left as it is.)
        Dn = np.linalg.norm(D, axis=1)
        if np.allclose(Dn, 1.0, atol=1e-6):
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            if np.all(np.diff(s) > 0):
                T = np.gradient(P, s, axis=0, edge_order=2)
                T /= np.linalg.norm(T, axis=1, keepdims=True)
                D = D - np.sum(D * T, axis=1, keepdims=True) * T
                D /= np.linalg.norm(D, axis=1, keepdims=True)
    joints = list(np.cumsum([m - 1 for m in counts])[:-1])
    es = geom.EdgeSample(P, D, DD, [int(j) for j in joints])
    if D is not None:
        sds = [getattr(e, "_side", None) for e in raw]
        if all(s is not None for s in sds):
            # trimmed edges: from the faces' forms, for the cross direction as squared above
            es._ddd_fn = lambda: _ddd_from_sides(sds, [rev for _, _, rev in chain], D, DD)
        else:
            es._ddd_fn = lambda: (None if any(e.DDD is None for e in parts) else
                                  np.vstack([parts[0].DDD] + [e.DDD[1:] for e in parts[1:]]))
    return es


# --------------------------------------------------------------------------
# Paired breakpoints: both sides sampled stretch by stretch
# --------------------------------------------------------------------------

PAIR_MAX = 0.15    # breakpoints pair when their places along the two sides differ by at most this share
PAIR_DENSE = 256   # samples a piece is located on (its curvature jumps found as shares of it)


def _piece_sampler(o, r):
    """(s0, s1, m) -> EdgeSample of m samples over shares s0..s1 of a loft
    input's own parameter (a natural edge's surface parameter, as
    input_samples takes it, or a curve's); None for a B-rep edge (square
    samples those whole)."""
    if is_edge_ref(r):
        surf = store.get_surfaces(o)[0]
        rng = edge_range(o, r)
        lo, hi = (0.0, 1.0) if rng is None else _domain_share(surf, r, rng)
        M = geom.world_matrix(o)
        return lambda s0, s1, m: geom.sample_surface_edge(
            surf, r, M, m, (lo + (hi - lo) * s0, lo + (hi - lo) * s1))
    if is_brep_edge_ref(r):
        return None
    c = input_curve(o, r)
    t0, t1 = c.parameter_range()
    return lambda s0, s1, m: geom.EdgeSample(np.array(
        [c.evaluate(t0 + (t1 - t0) * (s0 + (s1 - s0) * i / (m - 1))) for i in range(m)], float))


def _side_stretches(side):
    """A loft side cut at its breakpoints -- the joints between its pieces
    and the curvature jumps inside them (geom.curve_bend_points) -- in chain
    order: [(sampler, s0, s1, reversed, length)]; None when a piece cannot be
    sampled by shares."""
    out = []
    for o, r, rev in chain_side(side):
        smp = _piece_sampler(o, r)
        if smp is None:
            return None
        P = smp(0.0, 1.0, PAIR_DENSE + 1).P
        size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
        shares = []
        for q in geom.curve_bend_points(input_curve(o, r)):
            f = geom.polyline_index(P, q, 1e-3 * size)
            if f is None:
                continue
            # onto the point itself (located on the samples it was 0.19 mm
            # into a slot's arc: the joint sample's tangent off 0.2°)
            s, d = f / PAIR_DENSE, 1e-7
            for _ in range(8):
                e2 = smp(s, min(s + d, 1.0), 2)
                p0, t_ = e2.P[0], (e2.P[1] - e2.P[0]) / max(min(s + d, 1.0) - s, 1e-300)
                step = float((q - p0) @ t_) / max(float(t_ @ t_), 1e-300)
                s = min(max(s + step, 0.0), 1.0 - d)
                if abs(step) < 1e-13:
                    break
            if 1e-6 < s < 1.0 - 1e-6:
                shares.append(s)
        cuts = [0.0] + sorted(shares) + [1.0]
        piece = []
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            Q = smp(s0, s1, 33).P
            piece.append((smp, s0, s1, rev, float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)))))
        out += piece[::-1] if rev else piece
    return out


def _stretch_samples(stretches, counts):
    """EdgeSample of a side's stretches end to end, counts[k] samples on the
    k-th (its ends shared); every breakpoint a joint."""
    parts = []
    for (smp, s0, s1, rev, _), m in zip(stretches, counts):
        es = smp(s0, s1, m)
        parts.append(es.reversed() if rev else es)
    cat = lambda key: None if getattr(parts[0], key) is None else np.vstack(
        [getattr(parts[0], key)] + [getattr(e, key)[1:] for e in parts[1:]])
    joints = [int(j) for j in np.cumsum([m - 1 for m in counts])[:-1]]
    es = geom.EdgeSample(cat("P"), cat("D"), cat("DD"), joints)
    if all(e.T is not None for e in parts):
        # unit tangents: the stretches' parameter speeds differ
        T = cat("T")
        es.T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-300)
    if es.D is not None:
        es._ddd_fn = lambda: (None if any(e.DDD is None for e in parts) else
                              np.vstack([parts[0].DDD] + [e.DDD[1:] for e in parts[1:]]))
    return es


def _paired_or(sides, n, p, a, b):
    """paired_samples for the plain samples a, b (both in their own
    direction) when the sides pair, else a, b as they are. The pair must
    flip as the plain samples do (auto align decides on the samples)."""
    flip = (bool(p.get("auto_align", True)) and geom.should_flip(a.P, b.P)) != bool(p.get("flip_b"))
    pr = paired_samples(sides, n, flip)
    if pr is None:
        return a, b
    if ((bool(p.get("auto_align", True)) and geom.should_flip(pr[0].P, pr[1].P)) != bool(p.get("flip_b"))) != flip:
        return a, b
    return pr


def paired_samples(sides, n, flip):
    """Both sides of a two-sided loft sampled stretch by stretch (see
    _side_stretches), the k-th stretch of each with the same count: their
    breakpoints land on the same samples, so one knot line runs from a
    breakpoint on one side to its partner on the other. Sampled each by its
    own parameter, a sketch's line-to-arc joints sat at 0.255 / 0.5 / 0.754
    along it and its extrude's at 0.286 / 0.5 / 0.786: the knots one side
    needs crossed the other's smooth stretch and the loft kinked 0.5° inside.
    flip: side b runs the other way (auto align / Flip B). Returns (a, b), b
    in its own direction (the caller flips it), or None when the sides have
    no breakpoints, not as many, ones too far apart (PAIR_MAX), or a piece
    sampled otherwise (a B-rep edge)."""
    try:
        la, lb = _side_stretches(sides[0]), _side_stretches(sides[1])
    except Exception as exc:
        print("Loft: sides not paired,", exc)
        return None
    if la is None or lb is None or len(la) != len(lb) or len(la) < 2:
        return None
    lb_ = lb[::-1] if flip else lb
    wa = np.array([s[4] for s in la]); wb = np.array([s[4] for s in lb_])
    if wa.sum() <= 0 or wb.sum() <= 0:
        return None
    wa, wb = wa / wa.sum(), wb / wb.sum()
    if float(np.max(np.abs(np.cumsum(wa) - np.cumsum(wb)))) > PAIR_MAX:
        return None
    w = 0.5 * (wa + wb)
    counts = [max(4, int(round((n - 1) * x)) + 1) for x in w]
    return _stretch_samples(la, counts), _stretch_samples(lb, counts[::-1] if flip else counts)


# --------------------------------------------------------------------------
# Match Edges: a G1 / G2 side whose loft carries on the source's own edges
# --------------------------------------------------------------------------

def _edge_end_frame(e, at_end, M):
    """(point, unit tangent arriving at that end, curvature vector) of a B-rep
    edge's end, world space: the edge as it comes into the end point."""
    t0, t1 = e.parameter_range()
    h = 1e-3 * (t1 - t0)
    R, T = M[:3, :3], M[:3, 3]
    pt = lambda t: R @ np.array(e.point_at(t), float) + T
    if at_end:
        p0, p1, p2 = pt(t1), pt(t1 - h), pt(t1 - 2 * h)
        d1 = (3 * p0 - 4 * p1 + p2) / (2 * h)             # d/dt, forward = arriving
    else:
        p0, p1, p2 = pt(t0), pt(t0 + h), pt(t0 + 2 * h)
        d1 = -(-3 * p0 + 4 * p1 - p2) / (2 * h)           # -d/dt: arriving at the start
    d2 = (p0 - 2 * p1 + p2) / (h * h)
    sp = float(np.linalg.norm(d1)) or 1.0
    tan = d1 / sp
    kvec = (d2 - (d2 @ tan) * tan) / (sp * sp)            # curvature vector (either way round)
    return p0, tan, kvec


def edge_into_corner(obj, corner, side_pts, outward, tol):
    """The source's own edge that runs into a corner of a loft side (the rim
    of a cut spoon at the end of its cut): (unit tangent arriving at the
    corner, curvature vector), or None. Boundary edges (one face) not on the
    side, whose way on from the corner leaves the face (outward > 0)."""
    try:
        edges = store.get_brep(obj).edges()
    except Exception:
        return None
    M = np.array(geom.world_matrix(obj), float)
    best = None
    for e in edges:
        try:
            if len(e.faces()) != 1:
                continue
            t0, t1 = e.parameter_range()
            for at_end in (False, True):
                q, tan, kvec = _edge_end_frame(e, at_end, M)
                if float(np.linalg.norm(q - corner)) > tol:
                    continue
                mid = M[:3, :3] @ np.array(e.point_at(0.5 * (t0 + t1)), float) + M[:3, 3]
                if float(np.min(np.linalg.norm(side_pts - mid, axis=1))) <= tol:
                    continue                       # a piece of the side itself
                go = float(tan @ outward)
                if go > 0.05 and (best is None or go > best[0]):
                    best = (go, tan, kvec)
        except Exception:
            continue
    return None if best is None else (best[1], best[2])


MATCH_MAX_DEG = 75.0     # the cross direction leans at most this far towards an edge


MATCH_FADE = 0.3        # the edges' bend fades out over this share of the side from each corner
MATCH_LEAN_FADE = 1.0   # the lean fades out over this share (1: straight from one corner's lean to the other's)


def match_edges_samples(side, n, g2=False, lean_scale=1.0, bend_scale=1.0):
    """EdgeSample of a surface side whose cross direction follows the
    source's own edges at the side's corners: at each corner it is the
    direction the edge running into it carries on in, and in between it
    leans from one to the other (in the tangent plane: G1 holds). The
    second row is the source's normal curvature in that direction; with g2
    the corners also take the edges' bend, so the loft's outline carries on
    each edge in curvature. None when the side is not surface edges or
    nothing leans (the plain side is then used)."""
    from .square import sample_side
    chain = chain_side(side)
    if any(not (is_surface_edge(r) and o.type != 'CURVE') for o, r, _ in chain):
        return None
    if len(chain) == 1:
        counts = [n]
    else:
        lens = _piece_lengths(chain)
        total = n + len(chain) - 1
        counts = [max(4, int(round(total * L / (sum(lens) or 1.0)))) for L in lens]
        counts[int(np.argmax(lens))] += total - sum(counts)
    parts = []
    sds = []
    for (o, r, rev), m in zip(chain, counts):
        sd = sample_side(o, r, m)
        if sd.Q is None or sd.N is None or sd.B is None:
            return None
        arr = [np.asarray(sd.P, float), np.asarray(sd.N, float), np.asarray(sd.Q, float), np.asarray(sd.B, float)]
        parts.append([a[::-1] for a in arr] if rev else arr)
        sds.append((sd, rev))
    P, N, Q, B = (np.concatenate([parts[0][k]] + [pt[k][1:] for pt in parts[1:]]) for k in range(4))
    s_ = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if not np.all(np.diff(s_) > 0):
        return None
    L = float(s_[-1])
    T = np.gradient(P, s_, axis=0, edge_order=2)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    b = np.cross(N, T)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    b[np.sum(b * B, axis=1) < 0] *= -1.0                  # outward, as the source says
    tol = 1e-4 * (float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0)
    lim = np.tan(np.radians(MATCH_MAX_DEG))
    lean, bend = [0.0, 0.0], [np.zeros(3), np.zeros(3)]
    for k, i in ((0, 0), (1, len(P) - 1)):
        hit = edge_into_corner(chain[0 if k == 0 else -1][0], P[i], P, b[i], tol)
        if hit is None:
            continue
        w, kvec = hit
        w = w - (w @ N[i]) * N[i]                          # in the tangent plane (it is, up to the fit)
        if w @ b[i] <= 1e-6:
            continue
        lean[k] = float(np.clip((w @ T[i]) / (w @ b[i]), -lim, lim))
        g = kvec - (kvec @ N[i]) * N[i]                    # the edge's bend within the surface
        wu = w / np.linalg.norm(w)
        bend[k] = g - (g @ wu) * wu
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    bent = g2 and max(float(np.linalg.norm(bend[0])), float(np.linalg.norm(bend[1]))) * size > 1e-3
    if max(abs(lean[0]), abs(lean[1])) < 1e-3 and not bent:
        return None                           # square already (a rim square to the side): the plain rows
    if not g2:
        bend = [np.zeros(3), np.zeros(3)]
    t = s_ / L
    lean = [lean_scale * x for x in lean]
    if MATCH_LEAN_FADE >= 1.0:
        lam = (1 - t) * lean[0] + t * lean[1]
    else:
        fa_ = np.clip(1.0 - t / MATCH_LEAN_FADE, 0.0, 1.0) ** 2
        fb_ = np.clip(1.0 - (1.0 - t) / MATCH_LEAN_FADE, 0.0, 1.0) ** 2
        lam = fa_ * lean[0] + fb_ * lean[1]
    W = b + lam[:, None] * T
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    # second derivative for a unit-speed cross direction W: its normal part is
    # the source's normal curvature in W (G2); the part in the tangent plane is
    # free -- the corners' edges' bend, fading in between
    kn = np.einsum("ij,ijk,ik->i", W, Q, W)
    DD = kn[:, None] * N
    if g2:
        # only near the corners: the outline takes the edge's bend there, the
        # rows in between are left to the surface (faded in across the whole
        # side, a spoon's tight rims pulled the middle rows over each other)
        fa = np.clip(1.0 - t / MATCH_FADE, 0.0, 1.0) ** 2
        fb = np.clip(1.0 - (1.0 - t) / MATCH_FADE, 0.0, 1.0) ** 2
        G = bend_scale * (fa[:, None] * bend[0] + fb[:, None] * bend[1])
        G = G - np.sum(G * N, axis=1, keepdims=True) * N
        G = G - np.sum(G * W, axis=1, keepdims=True) * W
        DD = DD + G
    joints = list(np.cumsum([m - 1 for m in counts])[:-1])
    es = geom.EdgeSample(P, W, DD, [int(j) for j in joints])
    es._ddd_fn = lambda: _ddd_from_sides([s for s, _ in sds], [r for _, r in sds], W, DD)
    es.leans = (float(np.degrees(np.arctan(lean[0]))), float(np.degrees(np.arctan(lean[1]))))
    es.N = N
    es.scales = (lean_scale, bend_scale)
    es.redo = lambda m: match_edges_samples(side, m, False, lean_scale, bend_scale)
    return es


def surface_flow_samples(side, n, lean_scale=1.0):
    """EdgeSample of a surface side whose cross direction runs along the
    source's own parameter lines across the edge (Align to Surface): the
    face's iso direction that crosses the edge, in its tangent plane, with
    the face's normal curvature in it. None when every piece is an
    untrimmed edge (its exact rows run along those lines already) or the
    side is not surface edges."""
    from .square import sample_side, _foot
    chain = chain_side(side)
    if any(not (is_surface_edge(r) and o.type != 'CURVE') for o, r, _ in chain):
        return None
    if all(is_edge_ref(r) for _, r, _ in chain):
        return None
    if len(chain) == 1:
        counts = [n]
    else:
        lens = _piece_lengths(chain)
        total = n + len(chain) - 1
        counts = [max(4, int(round(total * L / (sum(lens) or 1.0)))) for L in lens]
        counts[int(np.argmax(lens))] += total - sum(counts)
    parts = []
    sds = []
    for (o, r, rev), m in zip(chain, counts):
        sd = sample_side(o, r, m)
        if sd.Q is None or sd.N is None or sd.B is None:
            return None
        b_ = store.get_brep(o)
        surf = (b_.edges()[int(r[1:])].faces()[0] if is_brep_edge_ref(r) else b_.faces()[0]).surface()
        M = np.array(geom.world_matrix(o), float)
        R, Ri, T = M[:3, :3], np.linalg.inv(M[:3, :3]), M[:3, 3]
        Pw = np.asarray(sd.P, float)
        Nw = np.asarray(sd.N, float)
        Bw = np.asarray(sd.B, float)
        Wd, prev = [], None
        for q, nq, bq in zip(Pw, Nw, Bw):
            prev = _foot(surf, Ri @ (q - T), prev)
            _, su, sv = surf.derivatives(*prev)
            su, sv = R @ np.array(su, float), R @ np.array(sv, float)
            bu = abs(su @ bq) / (np.linalg.norm(su) or 1.0)
            bv = abs(sv @ bq) / (np.linalg.norm(sv) or 1.0)
            d = su if bu >= bv else sv
            if d @ bq < 0:
                d = -d
            nn = nq / (np.linalg.norm(nq) or 1.0)
            d = d - (d @ nn) * nn
            Wd.append(d / (np.linalg.norm(d) or 1.0))
        arr = [Pw, Nw, np.asarray(sd.Q, float), np.array(Wd)]
        parts.append([a[::-1] for a in arr] if rev else arr)
        sds.append((sd, rev))
    P, N, Q, W = (np.concatenate([parts[0][k]] + [pt[k][1:] for pt in parts[1:]]) for k in range(4))
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    # the surface's flow, smoothed: its lines can fan out right at a corner
    # (a spoon's, 16° to 43° within the last sample) -- rows cannot follow
    # that; a cubic in arc length through the lean keeps the flow and drops
    # the fan. Eased towards square by lean_scale.
    s_ = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    T = np.gradient(P, s_, axis=0, edge_order=2)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    b = np.cross(N, T)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    b[np.sum(b * W, axis=1) < 0] *= -1.0
    lim = np.tan(np.radians(MATCH_MAX_DEG))
    lam = np.clip(np.sum(W * T, axis=1) / np.maximum(np.sum(W * b, axis=1), 1e-6), -lim, lim)
    tt = s_ / (s_[-1] or 1.0)
    if len(P) >= 8:
        lam = np.polyval(np.polyfit(tt, lam, 3), tt)
    W = b + (lean_scale * lam)[:, None] * T
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    kn = np.einsum("ij,ijk,ik->i", W, Q, W)
    joints = list(np.cumsum([m - 1 for m in counts])[:-1])
    es = geom.EdgeSample(P, W, kn[:, None] * N, [int(j) for j in joints])
    es._ddd_fn = lambda: _ddd_from_sides([s for s, _ in sds], [r for _, r in sds], es.D, es.DD)
    es.N = N
    es.redo = lambda m: surface_flow_samples(side, m, lean_scale)
    return es


EDGE_MODES = ("FREE", "EDGES", "SURFACE")


def side_edge_modes(p):
    """Each side's edge mode: FREE (square to the side, or along an
    untrimmed face's own lines), EDGES (Align Edges: carry on the surface's
    edges at the corners), SURFACE (Align to Surface: along the face's own
    lines across the side). Lofts made with the older Match Edges switch
    align edges on both sides."""
    old = "EDGES" if p.get("match_edges") else "FREE"
    return tuple(p.get(k) if p.get(k) in EDGE_MODES else old for k in ("edges_a", "edges_b"))


def input_samples(obj, ref, n):
    if is_edge_ref(ref):
        surf = store.get_surfaces(obj)[0]
        rng = edge_range(obj, ref)
        share = None if rng is None else _domain_share(surf, ref, rng)
        return geom.sample_surface_edge(surf, ref, geom.world_matrix(obj), n, share)
    if is_brep_edge_ref(ref):
        # trimmed / B-rep edge: exact points on the face, cross tangent and
        # cross curvature from the face (same sampling as Square)
        from .square import sample_side
        sd = sample_side(obj, ref, n)
        es = geom.EdgeSample(sd.P, sd.B, sd.D2)
        es._side = sd
        es._ddd_fn = lambda: _ddd_from_sides([sd], [False], es.D, es.DD)
        return es
    return geom.EdgeSample(geom.sample_curve(input_curve(obj, ref), n))


def ddd_normal(C, Q, N, D, DD):
    """Third derivative (its part along the unit normals N) of a cross curve
    on the source with c' = D, c'' = DD: n.c''' = C(D, D, D) + 3 II(D, P DD)
    (surfeval.forms; P the tangent part). The part in the tangent plane is
    free -- a reparameterisation -- and left 0."""
    from . import surfeval
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    PDD = DD - np.sum(DD * N, axis=1, keepdims=True) * N
    k3 = surfeval.cubic(C, D) + 3.0 * np.einsum("ij,ijk,ik->i", D, Q, PDD)
    return k3[:, None] * N


def _ddd_from_sides(sds, revs, D, DD):
    """DDD of a side sampled piece by piece (square.Side each, reversed where
    revs says; joints shared) for its final rows D, DD: None without the
    faces' third-order forms."""
    Cs, Qs, Ns = [], [], []
    for sd, rev in zip(sds, revs):
        C, Q, N = sd.C, sd.Q, sd.N
        if C is None or Q is None or N is None:
            return None
        if rev:
            C, Q, N = C[::-1], Q[::-1], N[::-1]
        k = 0 if not Cs else 1
        Cs.append(np.asarray(C, float)[k:])
        Qs.append(np.asarray(Q, float)[k:])
        Ns.append(np.asarray(N, float)[k:])
    return ddd_normal(np.concatenate(Cs), np.concatenate(Qs), np.concatenate(Ns), D, DD)


def default_ref(obj):
    return "S0" if obj.type == 'CURVE' else "U0"


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

LOFT_FOLD_DEG = 1.0     # a loft surface folding more than this along a knot line is split there


def split_folds(g, rows=None):
    """A surface with folds (C0 knot lines where its normal jumps, e.g. a loft
    through rectangles) as faces joined along them, like Rhino's polysurface
    from kinked curves: the folds become real edges, which Edge Fillet can
    round and every tool sees as edges. Anything else is returned as it is."""
    sm, _ = _k()
    if not isinstance(g, sm.Surface):
        return g
    pieces = store._crease_pieces(g, LOFT_FOLD_DEG, rows)
    if not pieces:
        return g
    from . import jobs
    ident = {"loc": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "angle": 0.0, "scale": [1.0, 1.0, 1.0]}
    objs = {f"s{i}": (store.brep_to_text(store.to_brep(pc)), ident) for i, pc in enumerate(pieces)}
    b = store.to_brep(g).bounding_box()
    diag = float(np.linalg.norm(np.array(b[1], float) - np.array(b[0], float))) or 1.0
    _, breps = jobs.run("join", objs, ["joined"], order=list(objs), into="s0", tolerance=1e-6 * diag)
    return store.brep_from_text(breps["joined"])


def loft_rail_count(inputs, p):
    """How many of the Loft's side groups (the last ones) are rails."""
    n = int(p.get("rails", 0) or 0)
    return n if 0 < n < len(loft_sides(inputs, p)) - 1 else 0


@builder("loft")
def build_loft(inputs, p):
    LAST_SPANS[0] = None
    if loft_rail_count(inputs, p):
        return _build_loft_with_rails(inputs, p)
    g = _build_loft(inputs, p)
    if _is_skin(inputs, p):
        # through curve sections: sections with corners give folds -> faces.
        # A G1 / G2 blend between two edges is never split (a fold there would
        # be a defect, not hidden as a polysurface)
        g = split_folds(g, _section_rows(g, inputs, p))
    elif p.get("fit") == "SPANS" and p.get("span_faces") and LAST_SPANS[0]:
        g = split_spans(g, inputs, p)
    return g


LAST_SPANS = [None]      # how many Bezier spans the last loft build made (None: not a span fit)
# the last span fit's worst miss over its own target (half the labels'
# tolerances) at all its samples: the labels look at 33 (a tangent swung 0.8°
# between them next to a slot's curvature jump, the labels said 0.1°), and a
# fit short of its target held a uniform-knot slot 0.015° where it can hold 0.001°
LAST_FIT = [0.0]


def _build_loft_with_rails(inputs, p):
    """Loft with guide curves: the profiles (side groups) and the rails (the
    last `rails` groups) go to Loft Rails' solver -- with two rails the four
    sides of a Square-style net, each surface edge among them G0 / G1 / G2
    with its weight; one rail is a guide the surface passes through. Loft's
    side A / B settings are the first / last profile's; rail A / B take
    cont_ra / cont_rb and their weights tan_ra / tan_rb."""
    from . import loft_rails
    LOFT_NOTES.clear()
    if "G3" in (p.get("cont_a"), p.get("cont_b")):
        # Loft Rails' solver holds G0 - G2
        p = dict(p, **{k: "G2" for k in ("cont_a", "cont_b") if p.get(k) == "G3"})
        LOFT_NOTES.append("with rails the profiles hold G2 at most -- G3 built as G2")
    q = {"sides": p.get("sides"), "rails": loft_rail_count(inputs, p),
         "lr_start": p.get("cont_a", "G0"), "lr_end": p.get("cont_b", "G0"),
         "lr_ra": p.get("cont_ra", "G0"), "lr_rb": p.get("cont_rb", "G0"),
         "lw_start": float(p.get("tan_a", 1.0)), "lw_end": float(p.get("tan_b", 1.0)),
         "lw_ra": float(p.get("tan_ra", 1.0)), "lw_rb": float(p.get("tan_rb", 1.0)),
         "samples": max(int(p.get("samples", 24)), 48), "lr_degree": 0,
         "lr_ease": float(p.get("corner_ease", 0.0) or 0.0), "lr_even": bool(p.get("even_net"))}
    # the solver's Auto accepts itself by its own, looser measure (a G2 spoon
    # edge 15 % off by the labels' check at 6 x 8 CVs): take Auto when the
    # labels hold, else the lightest uniform net of more spans that holds
    groups = loft_sides(inputs, p)
    nr = q["rails"]
    profs, rails = groups[:len(groups) - nr], groups[len(groups) - nr:]
    checks = [(profs[0], p.get("cont_a", "G0")), (profs[-1], p.get("cont_b", "G0"))]
    if nr == 2:
        checks += [(rails[0], p.get("cont_ra", "G0")), (rails[1], p.get("cont_rb", "G0"))]
    else:
        checks += [(rails[0], "G0")]                   # a guide: the loft passes through it
    sides_, conts_ = [c[0] for c in checks], [c[1] for c in checks]
    eases = loft_rail_eases(inputs, p)
    degs = (5,) if "G2" in conts_ else (3, 5)
    tries = [None] + sorted(((d, sp, sp) for d in degs for sp in RAIL_SPANS), key=lambda t: (t[0] + t[1], t[0]))
    # set by hand (Patch Quad Sided's Degree / Spans): that net only, or the
    # lightest that holds with the rest left to the search
    fd, fu, fv = (int(p.get(k, 0) or 0) for k in ("rail_degree", "rail_spans_u", "rail_spans_v"))
    if fd or fu or fv:
        ds = (fd,) if fd else degs
        su = (fu,) if fu else RAIL_SPANS
        sv = (fv,) if fv else RAIL_SPANS
        tries = sorted({(d, a, b) for d in ds for a in su for b in sv}, key=lambda t: (t[0] + t[1] + t[2], t[0]))
    best = None
    for tr in tries:
        spans = None if tr is None else tr[1]
        qq = dict(q) if tr is None else dict(q, lr_degree=tr[0], lr_spans_u=tr[1], lr_spans_v=tr[2])
        try:
            g = loft_rails.build(inputs, qq)
        except RuntimeError:
            continue
        worst, what = _verify_sides(g, sides_, conts_, eases)
        notes = list(loft_rails.LAST_INFO.get("corner_notes", []))
        if best is None or worst < best[0] - 1e-9:
            best = (worst, g, what, notes, spans)
        if worst <= 1.0:
            best = (worst, g, what, notes, spans)
            break
    if best is None:
        raise RuntimeError("the loft with rails could not be built")
    worst, g, what, notes, spans = best
    LOFT_NOTES.extend(notes)
    LAST_EASES.clear()
    LAST_EASES.update(eases)
    if p.get("span_faces"):
        g = split_spans(g, inputs, p, both=True)
    if worst > 1.0:
        LOFT_NOTES.append(f"{what} -- past the continuity tolerance with up to {RAIL_SPANS[-1]} spans"
                          + (" (the corners' tangent planes differ)" if notes else ""))
    return g


RAIL_SPANS = (4, 6, 8, 10, 12)
LAST_EASES = {}                     # the last loft with rails' eased corner zones (for the labels)     # loft with rails: the span counts tried after Auto, lightest first


def split_spans(g, inputs, p, both=False):
    """A span-fitted loft as one Bezier patch per span (both: across the
    profiles too -- a loft with rails), joined into a body (oriented first:
    run() turns only a single surface the right way)."""
    sm, _ = _k()
    if not isinstance(g, sm.Surface):
        return g
    flip = edge_vote(g, inputs) < 0
    if p.get("flip_normal"):
        flip = not flip
    if flip:
        from .square import reverse_u
        g = reverse_u(g)
    pieces = geom.split_uv_spans(g) if both else geom.split_v_spans(g)
    if len(pieces) < 2:
        return g
    from . import jobs
    ident = {"loc": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "angle": 0.0, "scale": [1.0, 1.0, 1.0]}
    objs = {f"s{i}": (store.brep_to_text(store.to_brep(pc)), ident) for i, pc in enumerate(pieces)}
    b = store.to_brep(g).bounding_box()
    diag = float(np.linalg.norm(np.array(b[1], float) - np.array(b[0], float))) or 1.0
    _, breps = jobs.run("join", objs, ["joined"], order=list(objs), into="s0", tolerance=1e-6 * diag)
    return store.brep_from_text(breps["joined"])


def _section_rows(g, inputs, p=None):
    """The u values where a loft surface holds its sections (None if not found)."""
    sm, _ = _k()
    try:
        rows = []
        for side in loft_sides(inputs, p or {}):
            P = geom.sample_curve(side_curve(side), 5)[1:4]
            us = []
            for q in P:
                uv = sm.surface_closest_point(g, tuple(map(float, q)))[1][0]
                us.append(float(uv.x if hasattr(uv, "x") else uv[0]))
            rows.append(float(np.median(us)))
        return rows
    except Exception:
        return None


def _is_skin(inputs, p):
    any_edge = any(is_surface_edge(r) for _, r in inputs)
    smooth = p.get("cont_a", "G0") != "G0" or p.get("cont_b", "G0") != "G0"
    return bool(p.get("closed")) or len(loft_sides(inputs, p)) > 2 or not (any_edge or smooth)


def _build_loft(inputs, p):
    sm, _ = _k()
    LOFT_NOTES.clear()
    if len(inputs) < 2:
        raise ValueError("Loft needs at least two inputs")
    any_edge = any(is_surface_edge(r) for _, r in inputs)
    smooth = p.get("cont_a", "G0") != "G0" or p.get("cont_b", "G0") != "G0"

    sides = loft_sides(inputs, p)
    if len(sides) < 2:
        raise ValueError("Loft needs at least two sides")
    closed = bool(p.get("closed"))
    if closed and len(sides) < 3:
        raise ValueError("a closed loft needs at least 3 sections")
    if closed or len(sides) > 2 or not (any_edge or smooth):
        # Plain skin through curves: exact, done by the kernel (closed: our
        # periodic skin, smooth across the seam). A side of several edges is
        # one curve through them all.
        curves = [side_curve(sd) for sd in sides]
        if closed:
            curves = [curves[i] for i in geom.ring_order(curves)]
        if p.get("auto_align", True):
            prev = geom.sample_curve(curves[0], 16)
            for i in range(1, len(curves)):
                if geom.is_closed_curve(curves[i]) and geom.is_closed_curve(curves[i - 1]):
                    # closed sections: the same way round, starting in the same direction
                    curves[i] = geom.align_closed(curves[i - 1], curves[i])
                    prev = geom.sample_curve(curves[i], 16)
                    continue
                cur = geom.sample_curve(curves[i], 16)
                if geom.should_flip(prev, cur):
                    curves[i] = geom.reverse_curve(curves[i])
                    cur = cur[::-1]
                prev = cur
        if p.get("flip_b"):
            curves[-1] = geom.reverse_curve(curves[-1])
        if closed:
            return geom.closed_skin(curves)
        if len(curves) == 2 and p.get("ruled"):
            return sm.create_ruled_surface(curves[0], curves[1])
        try:
            # our own skin: the kernel's spaces the sections by its own measure and
            # can overshoot (sections of different layouts: the loft rose past
            # the top section and folded back down to it)
            return geom.open_skin([c for c in curves])
        except Exception:
            return sm.create_skin_surface(curves)

    # Two inputs with G1 / G2 / G3 to surfaces: sampled blend surface.
    n = int(p.get("samples", 24))
    if p.get("fit") == "SPANS":
        n = max(n, SPAN_MIN_SAMPLES)
    if side_edge_modes(p) != ("FREE", "FREE") and not _g3_on_iso_edges(sides, p):
        g = _build_matched(sides, n, p)
        if g is not None:
            return g
    n = max(side_sample_count(sides[0], n), side_sample_count(sides[1], n))
    a = side_samples(sides[0], n)
    b = side_samples(sides[1], n)
    a, b = _paired_or(sides, n, p, a, b)
    if p.get("auto_align", True) and geom.should_flip(a.P, b.P):
        b = b.reversed()
    if p.get("flip_b"):
        b = b.reversed()
    p = _g3_ready(p, a, b)
    # exact: both inputs on the same layout along the edge -> keep it (a line stays a line)
    try:
        sa = side_exact(sides[0], p.get("cont_a") == "G3")
        sb = side_exact(sides[1], p.get("cont_b") == "G3")
    except Exception:
        sa = sb = None
    for nm, sd, cont in (("A", sa, p.get("cont_a", "G0")), ("B", sb, p.get("cont_b", "G0"))):
        if sd is not None and cont != "G0" and sd.get("creases"):
            worst = max(a for a, _ in sd["creases"])
            LOFT_NOTES.append(f"side {nm}: its faces meet at {worst:.2g}° at a joint -- "
                              f"the loft is {cont} along the edges except right at that joint")
    if sa is not None and sb is not None:
        if np.linalg.norm(sb["P"][0] - b.P[0]) > np.linalg.norm(sb["P"][-1] - b.P[0]):
            sb = geom.reverse_side(sb)           # follow the samples' orientation (auto align / flip)
        if np.linalg.norm(sa["P"][0] - a.P[0]) > np.linalg.norm(sa["P"][-1] - a.P[0]):
            sa = geom.reverse_side(sa)
        if not _same_layout(sa, sb):
            # two exact layouts that differ (degree 2 beside 3, other knots): raise and
            # refine both exactly to one layout, rather than fitting samples
            try:
                sa, sb = geom.compatible_sides(sa, sb)
            except Exception as exc:
                print("Loft: exact sides not compatible, sampling", exc)
        if _same_layout(sa, sb):
            return geom.blend_loft_exact(sa, sb, a, b, p.get("cont_a", "G0"), p.get("cont_b", "G0"),
                                         p.get("tan_a", 1.0), p.get("tan_b", 1.0),
                                         p.get("cur_a", 0.0), p.get("cur_b", 0.0))
    # along the edge: the inputs' own degree when both are known (a line stays a
    # line); a side without an exact layout (a trimmed edge) is curved: cubic at least
    deg_in = max((s["deg"] for s in (sa, sb) if s is not None), default=3)
    if sa is None or sb is None:
        deg_in = max(deg_in, 3)
    bends = (side_bend_points(sides[0]), side_bend_points(sides[1])) if p.get("fit") == "SPANS" else None
    LAST_FIT[0] = 0.0
    g = _sampled_blend(a, b, p, deg_in, bends)
    if deg_in == 3 and any(is_surface_edge(r) for _, r in inputs) and \
            {p.get("cont_a"), p.get("cont_b")} & {"G1", "G2", "G3"}:
        # held as the labels check it (and the fit at all its samples), else with more samples
        if max(_verify_sides(g, sides, (p.get("cont_a", "G0"), p.get("cont_b", "G0")))[0], LAST_FIT[0]) > 1.0:
            # short of the labels, or of the fit's own target: more samples
            m2 = int(p.get("samples", 24)) * 2
            if p.get("fit") == "SPANS":
                m2 = max(m2, SPAN_MIN_SAMPLES)   # fewer than a span fit starts with cannot do better
            g2 = _build_matched(sides, m2, p, force=True)
            if g2 is not None:
                g = g2
    return g


def _g3_on_iso_edges(sides, p):
    """A G3 loft whose sides are all untrimmed surface edges (or curves):
    the edges running into such a side's corners are its surface's own iso
    lines, which the exact rows carry on in direction and curvature already,
    so Align Edges takes them (exact G3). Its sampled rows lean, and leaning,
    G3 also hangs on the fitted edge row's third derivative along the edge: a
    smooth wavy pair held G2 to 3e-5 that way but missed G3 by 7 %."""
    if "G3" not in (p.get("cont_a"), p.get("cont_b")):
        return False
    return all(is_edge_ref(r) or o.type == 'CURVE' or not is_surface_edge(r) for sd in sides for o, r in sd)


def _g3_ready(p, a, b):
    """p with a G3 side whose samples have no third derivative (a source
    without a CV net) taken as G2, with a note."""
    for nm, es in (("a", a), ("b", b)):
        if p.get("cont_" + nm) == "G3" and es.has_surface and es.DDD is None:
            p = dict(p, **{"cont_" + nm: "G2"})
            LOFT_NOTES.append(f"side {nm.upper()}: G3 needs a source surface with a CV net -- built G2")
    return p


MATCH_MAX_SAMPLES = 200
SPAN_MAX_SAMPLES = 1000   # Bezier spans: the samples only feed the fit (CVs come from the tolerance)


def _build_matched(sides, n, p, force=False):
    """Match Edges: the cross direction follows the source's edges at the
    corners (sampled -- the exact rows hold the square direction only).
    The direction turns along the side, so the fit is checked between its
    samples and the samples doubled until G1 holds there (a short wide side
    held G1 at 49 Samples, not at 24). None when no side has anything to
    match."""
    conts = (p.get("cont_a", "G0"), p.get("cont_b", "G0"))
    modes = side_edge_modes(p)
    best = None
    m = n
    bends = (side_bend_points(sides[0]), side_bend_points(sides[1])) if p.get("fit") == "SPANS" else None
    while True:
        LOFT_NOTES.clear()
        k = max(side_sample_count(sides[0], m), side_sample_count(sides[1], m))
        plain = [side_samples(sides[0], k), side_samples(sides[1], k)]
        ma, mb = (_matched_side(sides[i], k, conts[i], p.get("tan_" + "ab"[i], 1.0), p.get("cur_" + "ab"[i], 0.0),
                                plain, conts, "AB"[i], modes[i]) for i in (0, 1))
        if ma is None and mb is None and not force:
            return best[1] if best else None
        if ma is None and mb is None:
            plain = list(_paired_or(sides, k, p, *plain))
        a = ma if ma is not None else plain[0]
        b = mb if mb is not None else plain[1]
        if p.get("auto_align", True) and geom.should_flip(a.P, b.P):
            b = b.reversed()
        if p.get("flip_b"):
            b = b.reversed()
        p_ = _g3_ready(p, a, b)
        LAST_FIT[0] = 0.0
        g = _sampled_blend(a, b, p_, 3, bends)
        worst, what = _verify_sides(g, sides, (p_.get("cont_a", "G0"), p_.get("cont_b", "G0")))
        # the labels' measure (between their samples too) decides the note; the
        # fit's own target, half of that, whether more samples are worth it
        if 0.5 * LAST_FIT[0] > worst:
            worst, what = 0.5 * LAST_FIT[0], f"the fit between the labels' samples: {0.5 * LAST_FIT[0]:.2g}x the tolerance"
        goal = max(worst, LAST_FIT[0])
        if best is None or goal < best[4]:
            best = (worst, g, list(LOFT_NOTES), what, goal)
        cap = SPAN_MAX_SAMPLES if p.get("fit") == "SPANS" else MATCH_MAX_SAMPLES
        if goal <= 1.0 or 2 * k > cap:
            break
        m *= 2
    LOFT_NOTES[:] = best[2]
    if best[0] > 1.0:
        LOFT_NOTES.append(f"{best[3]} at {k} samples -- past the continuity tolerance")
    return best[1]


def ease_zones(side, frac, note="eased at the corner (Ease Corners)", ends=(0, 1)):
    """{(object name, ref): [(end 0/1, share of that piece, note)]} for a
    loft side eased over `frac` of its length toward its ends (both, or those
    in `ends`): the first and last pieces of a chained side carry the side's ends."""
    if frac <= 0:
        return {}
    chain = chain_side(side)
    lens = _piece_lengths(chain)
    L = sum(lens) or 1.0
    out = {}
    for k, ((o, r, rev), Lp) in enumerate(zip(chain, lens)):
        f = min(1.0, frac * L / (Lp or 1.0))
        zones = []
        if k == 0 and 0 in ends:
            zones.append((1 if rev else 0, f, note))
        if k == len(chain) - 1 and 1 in ends:
            zones.append((0 if rev else 1, f, note))
        if zones:
            out[(o.name, r)] = zones
    return out


CORNER_ZONE = 0.1       # share of a profile, toward a corner no surface can hold G1 at, that is not held to G1 / G2


def _side_end(side, end, m=65):
    """(point, unit normal or None, unit tangent pointing into the side) at a
    side's start (end 0) or finish (end 1), in world space."""
    from .square import sample_side
    chain = chain_side(side)
    o, r, rev = chain[0] if end == 0 else chain[-1]
    sd = sample_side(o, r, m)
    first = (end == 0) != rev                   # the side's end is this piece's first sample
    P = np.asarray(sd.P, float)
    i, j = (0, 1) if first else (-1, -2)
    t = P[j] - P[i]
    t /= np.linalg.norm(t) or 1.0
    n = None
    if sd.has_surface and sd.N is not None:
        n = np.asarray(sd.N[i], float)
        n = n / (np.linalg.norm(n) or 1.0)
    return P[i], n, t


def loft_rail_eases(inputs, p):
    """{(object name, ref): ease zones} of a loft with two rails' profiles:
    a G2 profile eased toward both rails by Ease Corners, and every G1 / G2
    profile end where no surface can hold it -- the rail leaves the profile's
    tangent plane there (or, a G1 / G2 rail, its own surface's normal differs)
    by more than the labels' angle tolerance. Those corners are left out of
    the check over CORNER_ZONE of the profile rather than chased with more CVs."""
    if loft_rail_count(inputs, p) != 2:
        return {}
    from . import contlabels
    tol_a = contlabels._tols()[0]
    groups = loft_sides(inputs, p)
    profs, rails = groups[:-2], groups[-2:]
    out = {}

    def add(zs):
        for key, lst in zs.items():
            out.setdefault(key, []).extend(lst)

    frac = float(p.get("corner_ease", 0.0) or 0.0)
    rail_ends = []
    for rail, rc in zip(rails, (p.get("cont_ra", "G0"), p.get("cont_rb", "G0"))):
        for e in (0, 1):
            try:
                rail_ends.append((_side_end(rail, e), rc))
            except Exception:
                pass
    for side, cont in ((profs[0], p.get("cont_a", "G0")), (profs[-1], p.get("cont_b", "G0"))):
        if cont not in ("G1", "G2"):
            continue
        if cont == "G2" and frac > 0:
            add(ease_zones(side, frac))
        L = sum(_piece_lengths(chain_side(side))) or 1.0
        for e in (0, 1):
            try:
                P, n, _ = _side_end(side, e)
            except Exception:
                continue
            if n is None or not rail_ends:
                continue
            (Q, m, t), rc = min(rail_ends, key=lambda x: float(np.linalg.norm(x[0][0] - P)))
            if np.linalg.norm(Q - P) > 1e-3 * L:
                continue
            deg = float(np.degrees(np.arcsin(min(1.0, abs(float(t @ n))))))
            what = f"the rail leaves {deg:.2g}° off the tangent plane"
            if rc in ("G1", "G2") and m is not None:
                dn = float(np.degrees(np.arccos(min(1.0, abs(float(n @ m))))))
                if dn > deg:
                    deg, what = dn, f"the rail's surface is {dn:.2g}° off the tangent plane"
            if deg > tol_a:
                add(ease_zones(side, CORNER_ZONE, f"{what} at this corner: no surface is {cont} there",
                               ends=(e,)))
    return out


def _verify_sides(g, sides, conts, eases=None):
    """How a loft holds its sides, measured as the labels measure it
    (contlabels.measure: the gap along every edge or curve of a side -- G0
    included -- and for G1 / G2 the angle and curvature straight across it,
    between the samples the fit went through): (worst value over its
    tolerance, what it was). <= 1 holds."""
    from types import SimpleNamespace
    from mathutils import Matrix
    from . import contlabels
    tol_a, tol_c = contlabels._tols()
    tol_3 = contlabels._tol_g3()
    shim = SimpleNamespace(matrix_basis=Matrix.Identity(4), parent=None)     # the loft is in world space
    try:
        brep = store.to_brep(g)
    except Exception:
        return 0.0, ""
    worst, what = 0.0, ""
    for side, cont in zip(sides, conts):
        k = geom.CONT_ORDER.get(cont, 0)
        for o, r in side:
            edge = is_surface_edge(r) and o.type != 'CURVE'
            kk = k if edge else 0                      # a curve holds position only
            try:
                res = contlabels.measure(shim, o, r, kk, brep=brep, ease=(eases or {}).get((o.name, r)))
                L = _piece_lengths([(o, r, False)])[0] or 1.0
            except Exception:
                continue
            q = res["gap"] / max(contlabels.TOL_GAP_REL * L, 1e-15)
            if q > worst:
                worst, what = q, f"{o.name} {r}: gap {res['gap']:.2g}"
            if kk == 0:
                continue
            if res.get("eased"):
                # holds everywhere outside the eased corner zones (measure says
                # "eased" only then): that is what was asked for
                res = dict(res, angle=0.0, curv=0.0, rate=0.0)
            rate = res.get("rate", 0.0) if k >= 3 else 0.0
            rate = 1e9 if rate != rate else rate          # nan: no CV net to measure it on
            for val, tol, name in ((res["angle"], tol_a, f"G1 {res['angle']:.2g}°"),
                                   (res["curv"] if k >= 2 else 0.0, tol_c, f"G2 {100 * res['curv']:.2g}%"),
                                   (rate, tol_3, f"G3 {100 * rate:.2g}%")):
                q = val / max(tol, 1e-12)
                if q > worst:
                    worst, what = q, f"{o.name} {r}: {name}"
    return worst, what


ROW_SPACING_MIN = 0.05   # a CV row keeps at least this share of the edge's spacing (else the fit wobbles)


def _rows_cross(es, cont, tangent, curvature, n_deg, scale):
    """Do the side's CV rows cross over one another along the side, or come
    so close to it (squeezed under ROW_SPACING_MIN of the edge's spacing)
    that the fit through them wobbles?"""
    try:
        rows = geom._side_rows(es, cont, tangent, curvature, n_deg, scale)
    except ValueError:                       # G3 without a third derivative: built G2 (_g3_ready)
        rows = geom._side_rows(es, "G2", tangent, curvature, n_deg, scale)
    d0 = np.diff(rows[0], axis=0)
    dd = np.sum(d0 * d0, axis=1)
    return any(np.any(np.sum(np.diff(R, axis=0) * d0, axis=1) < ROW_SPACING_MIN * dd) for R in rows[1:])


def _matched_side(side, n, cont, tangent, curvature, plain, conts, name, mode="EDGES"):
    """The samples of one side in its edge mode: Align Edges eased (the
    edges' bend, then their lean, scaled back) until its rows do not cross;
    Align to Surface as it is (square to the side when its rows would
    cross). None for a G0 or Free side, or one with nothing to align."""
    if cont == "G0" or mode == "FREE":
        return None
    ka, kb = (geom.CONT_ORDER[c] for c in conts)
    n_deg = 1 + ka + kb
    scale = float(np.mean(np.linalg.norm(plain[0].P - plain[1].P[::(1 if not geom.should_flip(
        plain[0].P, plain[1].P) else -1)], axis=1))) or 1.0
    if mode == "SURFACE":
        for f in (1.0, 0.75, 0.5, 0.25):
            es = surface_flow_samples(side, n, f)
            if es is None:
                return None
            if not _rows_cross(es, cont, tangent, curvature, n_deg, scale):
                if f != 1.0:
                    LOFT_NOTES.append(f"side {name}: the surface's lines followed only {int(round(100 * f))} % "
                                      f"-- fully, its CV rows would cross (lower Tangent {name} to follow them fully)")
                return es
        LOFT_NOTES.append(f"side {name}: along the surface its CV rows would cross -- square to the side")
        return None
    steps = [(1.0, 1.0), (1.0, 0.5), (1.0, 0.25), (1.0, 0.0), (0.75, 0.0), (0.5, 0.0), (0.25, 0.0)]
    for lean_s, bend_s in steps:
        es = match_edges_samples(side, n, cont in ("G2", "G3"), lean_s, bend_s)
        if es is None:
            return None
        if not _rows_cross(es, cont, tangent, curvature, n_deg, scale):
            if (lean_s, bend_s) != (1.0, 1.0):
                what = "the edges' curvature" if lean_s == 1.0 else "the edges' direction"
                LOFT_NOTES.append(f"side {name}: {what} matched only {int(round(100 * (bend_s if lean_s == 1.0 else lean_s)))} % "
                                  f"-- fully, its CV rows would cross (lower Tangent {name} to match it fully)")
            return es
    LOFT_NOTES.append(f"side {name}: the edges could not be matched without crossing CV rows -- square to the side")
    return None


SPAN_MIN_SAMPLES = 120   # Bezier spans: samples along a side at least (cheap; spans come from tolerance)


def side_bend_points(side):
    """World points inside the pieces of a loft side where the curvature
    jumps (geom.curve_bend_points on each piece's exact curve): a slot's
    line-to-arc joints, also when Extrude has joined them into one edge."""
    out = []
    for o, r in side:
        try:
            out += geom.curve_bend_points(input_curve(o, r))
        except Exception as exc:
            print("Loft: no bends for", o.name, r, exc)
    return out


def _bend_indices(es, pts):
    """The bend points that lie on the samples es, as fractional sample indices."""
    if not pts:
        return []
    size = float(np.linalg.norm(np.ptp(es.P, axis=0))) or 1.0
    out = (geom.polyline_index(es.P, q, 1e-3 * size) for q in pts)
    return [f for f in out if f is not None]


def _ordered_bends(a, b, p, bends):
    """Both sides' bends as fractional sample indices, the side that must hold
    G1 / G2 first: where the two sides' bends crowd one another only one gets
    a knot, and a G0 side's needs it least (only its position counts)."""
    if not bends:
        return []
    out = [(_bend_indices(a, bends[0]), geom.CONT_ORDER.get(p.get("cont_a", "G0"), 0) if a.has_surface else 0),
           (_bend_indices(b, bends[1]), geom.CONT_ORDER.get(p.get("cont_b", "G0"), 0) if b.has_surface else 0)]
    out.sort(key=lambda t: -t[1])
    return out[0][0] + out[1][0]


def _sampled_blend(a, b, p, deg_in, bends=None):
    """The sampled G0 / G1 / G2 blend between two sides' samples. Fit
    "SPANS": a few Bezier spans of one degree, knots placed by tolerance
    (geom.blend_loft_spans), full knots at the bends too (side_bend_points
    of each side, `bends` = (side a's, side b's)). Otherwise a least-squares net; a chained side
    is then fitted piece by piece between its joints (a piece needs 4
    samples at least, else its joint is left to the fit)."""
    if p.get("fit") == "SPANS":
        from . import contlabels
        tol_a, tol_c = contlabels._tols()
        g, nspans, worst = geom.blend_loft_spans(
            a, b, sorted(set(a.joints) | set(b.joints)), p.get("cont_a", "G0"), p.get("cont_b", "G0"),
            p.get("tan_a", 1.0), p.get("tan_b", 1.0), p.get("cur_a", 0.0), p.get("cur_b", 0.0),
            degree=int(p.get("span_degree", 5)), tol_angle=tol_a, tol_curv=tol_c, tol_gap_rel=contlabels.TOL_GAP_REL,
            tol_rate=contlabels._tol_g3(),
            bends=_ordered_bends(a, b, p, bends))
        LAST_SPANS[0] = nspans
        LAST_FIT[0] = worst                  # over its own target: half the labels' tolerances
        return g
    breaks, last = [], 0
    for j in sorted(set(a.joints) | set(b.joints)):
        if j - last >= 3 and len(a.P) - 1 - j >= 3:
            breaks.append(j)
            last = j
    if breaks:
        return geom.blend_loft_pieces(a, b, breaks, p.get("cont_a", "G0"), p.get("cont_b", "G0"),
                                      p.get("tan_a", 1.0), p.get("tan_b", 1.0),
                                      p.get("cur_a", 0.0), p.get("cur_b", 0.0), degree_v=deg_in)
    return geom.blend_loft(a, b,
                           p.get("cont_a", "G0"), p.get("cont_b", "G0"),
                           p.get("tan_a", 1.0), p.get("tan_b", 1.0),
                           p.get("cur_a", 0.0), p.get("cur_b", 0.0), degree_v=deg_in)


def extrude_direction(obj, curve, p):
    """Unit direction an extrude of `curve` (an input of `obj`) goes, before the
    distance's sign: its Direction, then the tilt. The Extrude tool's live
    preview uses it too, so the preview and the result always agree."""
    mode = p.get("direction", "NORMAL")
    if mode == "NORMAL":
        d = geom.profile_normal(curve, p.get("normal_hint"))
        if d.length < 0.5:
            d = Vector((0, 0, 1))
    elif mode == "VECTOR":
        d = Vector(p.get("dir_vec", (0.0, 0.0, 1.0)))
        if d.length < 1e-12:
            d = Vector((0, 0, 1))
    else:
        d = geom.axis_vector(mode, obj)
    return geom.tilt_direction(d.normalized(), float(p.get("tilt", 0.0)), float(p.get("tilt_azimuth", 0.0)))


def profile_groups(inputs):
    """The inputs of one Extrude in groups that touch end to end (curves and
    surface edges alike), each in the order given; a group that branches
    (three ends at one point) comes apart into single inputs."""
    if len(inputs) < 2:
        return [list(inputs)]
    ends = [_curve_ends(input_curve(o, r)) for o, r in inputs]
    allp = np.vstack([e for pair in ends for e in pair])
    tol = 1e-4 * (float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0)
    parent = list(range(len(inputs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(inputs)):
        for j in range(i + 1, len(inputs)):
            if min(np.linalg.norm(a - b) for a in ends[i] for b in ends[j]) <= tol:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(inputs)):
        groups.setdefault(root(i), []).append(i)
    out = []
    for idx in sorted(groups.values()):
        side = [inputs[i] for i in idx]
        try:
            chain_side(side, mixed=True)
            out.append(side)
        except ValueError:
            out += [[x] for x in side]
    return out


def profile_curve(inputs):
    """One exact kernel curve through inputs that touch end to end (a closed
    chain ends exactly where it starts), or the single input's curve."""
    if len(inputs) == 1:
        return input_curve(*inputs[0])
    _, ext = _k()
    c = side_curve(inputs, mixed=True)
    d = ext.curve_cvs(c)
    P = np.array(d["points"], float)
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    gap = float(np.linalg.norm(P[0] - P[-1]))
    if 0.0 < gap <= 1e-4 * size:          # closed within the chaining tolerance: one point at the seam
        P[-1] = P[0]
        return ext.curve_from_cvs(int(d["degree"]), [tuple(map(float, x)) for x in P],
                                  [float(k) for k in d["knots"]], d.get("weights") or None)
    return c


@builder("extrude")
def build_extrude(inputs, p):
    if p.get("caps"):
        return _solid(inputs, p, _extrude_side, joints=True)
    obj, ref = inputs[0]
    return _extrude_side(obj, profile_curve(inputs), p)


def _extrude_side(obj, curve, p):
    """The extrude's wall (a surface) of one profile curve."""
    sm, _ = _k()
    d = extrude_direction(obj, curve, p)
    h = float(p.get("distance", 1.0))
    draft = float(p.get("draft", 0.0))
    if p.get("start"):                       # Extrude Face: the walls start a hair behind the face
        curve = geom.translate_curve(curve, tuple(d * float(p["start"])))
    if abs(draft) > 1e-9:
        # Multi-Draft: height h along d, the far edge moved outward by h tan(draft)
        return geom.draft_extrude(curve, d, h, draft, flip=bool(p.get("draft_flip")),
                                  symmetric=bool(p.get("symmetric")), centre=p.get("draft_centre"))
    vec = d * h
    if p.get("symmetric"):
        curve = geom.translate_curve(curve, tuple(-vec))
        vec = vec * 2.0
    return sm.create_extrude_surface(curve, tuple(vec))


def is_face_ref(ref):
    """A face of a NURBS object as an input: F<index> (Extrude Face)."""
    return isinstance(ref, str) and len(ref) > 1 and ref[0] == "F" and ref[1:].isdigit()


def face_at(obj, p, ref=None):
    """The index of the face an Extrude Face means: the one at its point
    (face_local, the object's local space), else the one its ref names."""
    faces = store.get_brep(obj).faces()
    q = p.get("face_local")
    if q is not None:
        return min(range(len(faces)), key=lambda k: faces[k].closest_point(tuple(map(float, q)))[1])
    i = int(ref[1:]) if ref else 0
    if not 0 <= i < len(faces):
        raise RuntimeError(f"{obj.name} has no face {i}")
    return i


def face_direction(obj, i, p):
    """Unit direction (world) an Extrude Face goes before the distance's sign:
    Face Normal is the face's outward normal (a flat face's plane normal, a
    curved face's averaged over it), the other Directions as Extrude's."""
    if p.get("direction", "NORMAL") != "NORMAL":
        return extrude_direction(obj, None, p)
    return geom.tilt_direction(Vector(face_normal(obj, i)), float(p.get("tilt", 0.0)),
                               float(p.get("tilt_azimuth", 0.0)))


def face_normal(obj, i):
    """World unit outward normal of face i: a flat face's, or a curved face's
    averaged over its tessellation points (a bump's comes out straight up)."""
    from .pushpull import face_frame, normals_at
    _, _, n, flat = face_frame(obj, i)
    if not flat:
        from . import subsel
        pts = subsel.face_tris(obj, i)[0]
        if len(pts):
            m = normals_at(obj, i, pts[::max(1, len(pts) // 200)]).mean(axis=0)
            if np.linalg.norm(m) > 1e-6:
                n = m / np.linalg.norm(m)
    return np.asarray(n, float)


def face_loop_inputs(obj, i):
    """The edges round face i as inputs ((obj, ref), each once: seams and
    zero-length pole edges left out) -- an untrimmed surface's sides U0 / U1 /
    V0 / V1, else its B-rep edges E<k>."""
    if store.is_natural(obj):
        surf = store.get_surfaces(obj)[0]
        out = []
        for which in ("U0", "U1", "V0", "V1"):
            P = geom.boundary_points(surf, which, 9, corners=True)
            if np.linalg.norm(np.ptp(np.asarray(P, float), axis=0)) > 1e-9:
                out.append((obj, which))
        return out
    b = store.get_brep(obj)
    F = b.faces()[i]
    uses = {}
    for loop in F.boundary_loops():
        for e, _ in loop:
            uses[e] = uses.get(e, 0) + 1
    edges = b.edges()
    return [(obj, f"E{k}") for k, e in enumerate(edges) if uses.get(e) == 1 and e.length() > 1e-9]


@builder("extrude_face")
def build_extrude_face(inputs, p):
    """A closed solid from a face swept along a direction (Fusion's Extrude
    of a face): the face, its copy moved by Distance and walls from its loops.
    Draft (a flat face only) leans the walls as Extrude's Multi-Draft, holes
    the other way. reach: the solid starts that far behind the face, so a Join
    or Cut with the face's own body never meets the face exactly."""
    from . import jobs
    obj, ref = inputs[0]
    i = face_at(obj, p, ref)
    from .pushpull import face_frame
    _, pw, _, flat = face_frame(obj, i)
    d = np.array(face_direction(obj, i, p), float)
    h = float(p.get("distance", 1.0))
    if abs(h) <= 0.0:
        raise RuntimeError("the distance is 0")
    reach = float(p.get("reach", 0.0))
    sym = bool(p.get("symmetric"))
    a, b = (-abs(h), abs(h)) if sym else (-reach if h > 0 else reach, h)
    if abs(float(p.get("draft", 0.0))) > 1e-9:
        if not flat:
            raise RuntimeError("Draft needs a flat face: a curved face extrudes straight (Draft 0)")
        q = dict(p, direction="VECTOR", dir_vec=[float(x) for x in d], tilt=0.0, tilt_azimuth=0.0)
        q.pop("draft_centre", None)
        if not sym:
            q["start"], q["distance"] = a, b - a
        return _solid(face_loop_inputs(obj, i), q, _extrude_side, joints=True)
    info, breps = jobs.run("extrude_face", {"target": obj}, ["result"], face_pt=[float(x) for x in pw],
                           dir=[float(x) for x in d], start=a, distance=b, into="_world_")
    return ("brep_text", breps["result"])


def revolve_exact(curve, origin, axis, angle_deg, start_deg=0.0):
    """Exact surface of revolution as a rational tensor product: an exact
    rational arc (u) x the profile's own CVs (v). The kernel's
    create_surface_revolution only revolves the FIRST span of a multi-span
    profile, so it is not used. Works for any profile (rational too), axis and
    angle; start_deg turns the start (seam) around the axis."""
    from .primitives import exact_arc_spec
    sm, ext = _k()
    cv = ext.curve_cvs(curve)
    P = np.array(cv["points"], float)
    Wp = np.array(cv["weights"], float) if cv.get("weights") else np.ones(len(P))
    z = np.asarray(axis, float)
    z /= np.linalg.norm(z)
    O = np.asarray(origin, float)
    # frame: x towards the profile's start point (projected off the axis)
    rel = P - O
    radial = rel - np.outer(rel @ z, z)
    # CVs a hair off the axis go onto it: a profile end meant to lie there (a
    # rectangle's corner, float32 in Blender and turned by a rotation not
    # exactly 90 degrees, ends 2e-9 off) would revolve into a ring a few
    # nanometres wide instead of a pole -- sliver triangles whose normals
    # point anywhere: a bright spot at the axis
    rl = np.linalg.norm(radial, axis=1)
    on = rl <= 1e-6 * (float(np.ptp(P, axis=0).max()) or 1.0)
    if on.any() and not on.all():
        rel[on] -= radial[on]
        radial[on] = 0.0
    k = int(np.argmax(np.linalg.norm(radial, axis=1)))
    x = radial[k] / (np.linalg.norm(radial[k]) or 1.0)
    if np.linalg.norm(radial[k]) < 1e-12:
        x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x -= (x @ z) * z
        x /= np.linalg.norm(x)
    y = np.cross(z, x)
    a, b_, h = rel @ x, rel @ y, rel @ z            # profile in the axis frame
    ang = float(angle_deg)
    sgn = 1.0 if ang >= 0 else -1.0
    arc = exact_arc_spec(1.0, min(abs(ang), 360.0))
    from .geom import blender_nurbs_to_clamped
    rdeg, rP, rW, rU = blender_nurbs_to_clamped(arc["points"], arc["weights"], arc["order"], False, True, True)
    c0, s0 = np.cos(np.radians(start_deg)), np.sin(np.radians(start_deg))
    pts, wts = [], []
    for (cx, cy, _), cw in zip(rP, rW):
        cy *= sgn
        # rotate the ring CV by the start angle (a rotation keeps the weights)
        cx, cy = c0 * cx - s0 * cy, s0 * cx + c0 * cy
        for ai, bi, hi, wp in zip(a, b_, h, Wp):
            # R(theta) (a, b) = a (cos, sin) + b (-sin, cos): linear in the ring CV
            q = O + (ai * cx - bi * cy) * x + (ai * cy + bi * cx) * y + hi * z
            pts.append(tuple(map(float, q)))
            wts.append(float(cw * wp))
    rational = any(abs(w - 1.0) > 1e-12 for w in wts)
    return ext.surface_from_cvs(int(rdeg), int(cv["degree"]), pts, [float(t) for t in rU],
                                [float(t) for t in cv["knots"]], wts if rational else None)


def revolve_axis(obj, p):
    """(point, unit direction) of a revolve's axis (world): a world or local
    axis through the pivot, or a picked line (axis_point, axis_dir)."""
    if p.get("axis") == "PICKED":
        d = Vector(p.get("axis_dir", (0.0, 0.0, 1.0)))
        return Vector(p.get("axis_point", (0.0, 0.0, 0.0))), (d.normalized() if d.length > 1e-12 else Vector((0, 0, 1)))
    axis = geom.axis_vector(p.get("axis", "Z"), obj)
    origin = geom.world_matrix(obj).translation if p.get("pivot", "OBJECT") == "OBJECT" else Vector()
    return Vector(origin), axis.normalized()


@builder("revolve")
def build_revolve(inputs, p):
    if p.get("caps"):
        return _solid(inputs, p, _revolve_solid_side, closed_only=False)
    obj, ref = inputs[0]
    return _revolve_side(obj, input_curve(obj, ref), p)


def _revolve_solid_side(obj, curve, p):
    """A solid's revolved wall: a closed profile turned all the way round is
    two half turns (one face closed both ways -- a torus-like ring -- has no
    volume in the kernel: its integration needs a seam it can cross)."""
    angle = float(p.get("angle", 360.0))
    origin, axis = revolve_axis(obj, p)
    if (abs(abs(angle) - 360.0) > 1e-9 or not is_closed_curve(curve)
            or _off_axis_runs(curve, origin, axis)[0] is not curve):    # a side on the axis: open runs
        return _revolve_side(obj, curve, p, solid=True)
    seam = float(p.get("seam_angle", 0.0))
    sm, ext = _k()
    copy = lambda: ext.curve_from_cvs(*_cv_args(ext.curve_cvs(curve)))
    return [revolve_exact(copy(), tuple(origin), tuple(axis), 180.0, seam),
            revolve_exact(copy(), tuple(origin), tuple(axis), 180.0, seam + 180.0)]


def _cv_args(d):
    return (int(d["degree"]), [tuple(map(float, q)) for q in d["points"]], [float(k) for k in d["knots"]],
            d.get("weights") or None)


def _revolve_side(obj, curve, p, solid=False):
    """The revolve's wall (a surface) of one profile curve. Stretches of the
    profile lying on the axis (a rectangle's side on it) turn into no area --
    the kernel refuses such a face -- so they are left out: the rest is
    revolved (one surface; a Brep of several when the axis cuts the profile
    into separate runs, or a list of them for a solid's walls)."""
    origin, axis = revolve_axis(obj, p)
    angle = float(p.get("angle", 360.0))
    seam = float(p.get("seam_angle", 0.0)) if abs(abs(angle) - 360.0) < 1e-9 else 0.0
    runs = _off_axis_runs(curve, origin, axis)
    if p.get("split_kinks") and not solid:      # one face per smooth stretch: its corners become edges
        runs = [q for c in runs for q in _split_at_kinks(c)]
    walls = [revolve_exact(c, tuple(origin), tuple(axis), angle, seam) for c in runs]
    if len(walls) == 1 or solid:
        return walls[0] if len(walls) == 1 else walls
    sm, ext = _k()
    merged = ext.brep_from_surface(walls[0])
    for w in walls[1:]:
        ext.append_brep(merged, ext.brep_from_surface(w))
    size = float(np.ptp(geom.sample_curve(curve, 33), axis=0).max()) or 1.0
    return sm.stitch_advanced(merged, max(1e-5, 1e-6 * size), squeeze_small_edges=False,
                              making_manifold_solid=False)[0]


def _off_axis_runs(curve, origin, axis, rel=1e-6):
    """The profile's runs off the revolve axis: the curve cut exactly at the
    ends of its spans that lie on the axis (each run one curve; a closed
    profile's run across its start is joined back into one). [curve] when no
    span lies on it, or when all do (dropped elsewhere)."""
    sm, ext = _k()
    d = ext.curve_cvs(curve)
    U = [float(k) for k in d["knots"]]
    brk = sorted(set(U))
    spans = [(a, b) for a, b in zip(brk, brk[1:]) if b - a > 1e-12 * (brk[-1] - brk[0])]
    O, z = np.asarray(origin, float), np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    S = geom.sample_curve(curve, 33)
    tol = rel * (float(np.ptp(S, axis=0).max()) or 1.0)

    def on(a, b):
        X = np.array([curve.evaluate(a + (b - a) * s) for s in np.linspace(0.0, 1.0, 9)])
        rel_ = X - O
        return float(np.linalg.norm(rel_ - np.outer(rel_ @ z, z), axis=1).max()) <= tol

    flags = [on(a, b) for a, b in spans]
    if not any(flags) or all(flags):
        return [curve]
    runs, cur = [], None
    for (a, b), f in zip(spans, flags):
        if f:
            if cur:
                runs.append(cur)
            cur = None
        else:
            cur = [cur[0], b] if cur else [a, b]
    if cur:
        runs.append(cur)
    pieces = [_sub_curve(curve, a, b) for a, b in runs]
    if len(runs) > 1 and not flags[0] and not flags[-1] and is_closed_curve(curve):
        pieces = pieces[1:-1] + [_join_curves_c0(pieces[-1], pieces[0])]
    return pieces


def _turn_deg(curve, t, h):
    """How far the curve's direction turns at t (degrees): the chords just
    before and just after it. None where a chord has no length."""
    a, m, b = (np.asarray(curve.evaluate(x), float) for x in (t - h, t, t + h))
    u, v = m - a, b - m
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if lu <= 0.0 or lv <= 0.0:
        return None
    return math.degrees(math.acos(min(1.0, max(-1.0, float(u @ v) / (lu * lv)))))


def _split_at_kinks(curve, min_deg=0.5):
    """The curve cut at its corners (knots as multiple as its degree where it
    turns by more than min_deg): a revolved polyline gets one face per stretch,
    so its creases are edges that can be picked (fillets, chamfers) and shade
    sharp. A closed curve smooth across its start keeps that stretch whole.
    [curve] when it has no corner."""
    sm, ext = _k()
    d = ext.curve_cvs(curve)
    U, deg = [float(k) for k in d["knots"]], int(d["degree"])
    t0, t1 = U[0], U[-1]
    h = 1e-6 * (t1 - t0)
    kinks = [k for k in geom._corner_knots(U, deg, t0, t1) if (_turn_deg(curve, k, h) or 0.0) > min_deg]
    if not kinks:
        return [curve]
    cuts = [t0] + kinks + [t1]
    pieces = [_sub_curve(curve, a, b) for a, b in zip(cuts, cuts[1:])]
    if is_closed_curve(curve):
        a0, a1 = (np.asarray(curve.evaluate(x), float) for x in (t0, t0 + h))
        b0, b1 = (np.asarray(curve.evaluate(x), float) for x in (t1 - h, t1))
        u, v = b1 - b0, a1 - a0
        if np.linalg.norm(u) > 0 and np.linalg.norm(v) > 0 and \
                math.degrees(math.acos(min(1.0, max(-1.0, float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))))) <= min_deg:
            pieces = pieces[1:-1] + [_join_curves_c0(pieces[-1], pieces[0])]
    return pieces


def _join_curves_c0(c1, c2):
    """Two kernel curves of one degree where c1 ends at c2's start, as one
    (a C0 knot where they meet; c2's weights scaled to c1's at the joint)."""
    sm, ext = _k()
    d1, d2 = ext.curve_cvs(c1), ext.curve_cvs(c2)
    p = int(d1["degree"])
    if int(d2["degree"]) != p:
        raise RuntimeError("revolve: the profile's pieces off the axis differ in degree")
    U1, U2 = [float(k) for k in d1["knots"]], [float(k) for k in d2["knots"]]
    f1 = lambda t: (t - U1[0]) / (U1[-1] - U1[0])
    f2 = lambda t: 1.0 + (t - U2[0]) / (U2[-1] - U2[0])
    U = [f1(t) for t in U1[:-1]] + [f2(t) for t in U2[p + 1:]]
    P = [tuple(map(float, q)) for q in d1["points"]] + [tuple(map(float, q)) for q in d2["points"][1:]]
    W = None
    if d1.get("weights") or d2.get("weights"):
        w1 = [float(w) for w in (d1.get("weights") or [1.0] * len(d1["points"]))]
        w2 = [float(w) for w in (d2.get("weights") or [1.0] * len(d2["points"]))]
        k = w1[-1] / w2[0]
        W = w1 + [w * k for w in w2[1:]]
    return ext.curve_from_cvs(p, P, U, W)


# --------------------------------------------------------------------------
# Solids: Extrude / Revolve with Caps (Body: New Solid, Join, Cut, Intersect)
# --------------------------------------------------------------------------

def is_closed_curve(curve, rel=1e-6):
    P = geom.sample_curve(curve, 9)
    size = float(np.ptp(P, axis=0).max()) or 1.0
    return float(np.linalg.norm(P[0] - P[-1])) <= rel * size


def _in_polygon(q, P, n):
    """Is point q inside the closed polygon P (the plane with unit normal n)?"""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = a - (a @ n) * n
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    x, y = P @ a, P @ b
    qx, qy = float(q @ a), float(q @ b)
    inside = False
    for i in range(len(P) - 1):
        x0, y0, x1, y1 = x[i], y[i], x[i + 1], y[i + 1]
        if (y0 > qy) != (y1 > qy) and qx < x0 + (qy - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def nesting(curves):
    """Depth of each closed profile: how many of the others (flat, in the same
    plane) it lies inside. Even depth: an outline; odd: a hole in the one
    around it. Open or curved profiles have depth 0."""
    samples = [geom.sample_curve(c, 97) for c in curves]
    planes = []
    for P in samples:
        c = P.mean(axis=0)
        _, sv, vt = np.linalg.svd(P - c)
        size = float(np.ptp(P, axis=0).max()) or 1.0
        flat = sv[1] > 1e-9 * size and np.abs((P - c) @ vt[2]).max() <= 1e-6 * size
        closed = np.linalg.norm(P[0] - P[-1]) <= 1e-6 * size
        planes.append((vt[2], c, size) if flat and closed else None)
    depth = [0] * len(curves)
    for j, Q in enumerate(samples):
        for i, P in enumerate(samples):
            if i == j or planes[i] is None or planes[j] is None:
                continue
            n, c, size = planes[i]
            if abs(abs(float(n @ planes[j][0])) - 1.0) > 1e-6 or np.abs((Q - c) @ n).max() > 1e-6 * size:
                continue
            if _in_polygon(Q[0], P, n):
                depth[j] += 1
    return depth


def solid_regions(inputs):
    """The inputs of a Caps tool grouped into the solids to make: each outline
    with the profiles directly inside it (its holes). A list of input lists."""
    groups = profile_groups(inputs)
    curves = [profile_curve(g) for g in groups]
    depth = nesting(curves)
    samples = [geom.sample_curve(c, 97) for c in curves]
    regions = {i: list(groups[i]) for i in range(len(groups)) if depth[i] % 2 == 0}
    for j in range(len(groups)):
        if depth[j] % 2 == 0:
            continue
        # the outline it is a hole of: the one around it, one level up
        owners = [i for i in regions if depth[i] == depth[j] - 1 and nesting([curves[i], curves[j]])[1] == 1]
        if owners:
            regions[owners[0]] += groups[j]
        else:
            regions[j] = list(groups[j])
    return [regions[i] for i in sorted(regions)]


def _solid(inputs, p, side_of, closed_only=True, joints=False):
    """A closed solid: the walls of every profile in inputs (an outline and its
    holes), flat caps on the open ends (in the worker). Holes draft the other
    way, so the material's walls all lean the same way. joints: extruded walls
    get a face per profile piece (split_wall_at_joints)."""
    groups = profile_groups(inputs)
    curves = [profile_curve(g) for g in groups]
    if closed_only:
        open_ = [g[0][0].name for g, c in zip(groups, curves) if not is_closed_curve(c)]
        if open_:
            raise RuntimeError("a solid needs closed profiles: " + ", ".join(open_) + _is_are(open_) +
                               "(Body: Surface extrudes it as a wall)")
    depth = nesting(curves)
    walls = []
    for i, (g, c) in enumerate(zip(groups, curves)):
        q = dict(p)
        q.pop("draft_centre", None)          # each closed profile drafts away from its own middle
        if depth[i] % 2:
            q["draft_flip"] = not q.get("draft_flip", False)
        w = side_of(g[0][0], c, q)
        if joints:
            w = split_wall_at_joints(w, profile_joint_points(g))
        walls += w if isinstance(w, list) else [w]
    return cap_walls(walls)


def profile_joint_points(group):
    """World points where the pieces of one profile meet: the joints a Join
    Curves recorded on its splines, and the ends of chained inputs."""
    from . import curvejoin
    pts = []
    if len(group) > 1:
        for o, r in group:
            pts += list(_curve_ends(input_curve(o, r)))
    for o, r in group:
        if o.type != 'CURVE' or not (isinstance(r, str) and r[:1] == "S" and r[1:].isdigit()):
            continue
        recs, i = curvejoin.joint_records(o), int(r[1:])
        rec = recs[i] if i < len(recs) else None
        if not isinstance(rec, dict):
            continue
        try:
            spans = curvejoin.input_spans(o, r)
        except Exception:
            continue
        if rec.get("nseg") != len(spans):
            continue                         # the spline was edited since: its record no longer fits
        pts += [curvejoin._cart(spans[j])[0] for j in rec.get("joints") or [] if 0 < j < len(spans)]
    return pts


def _span_shape(s, tol):
    """What a rational Bezier span (world) is: ("line", point, unit direction),
    ("arc", centre, radius) or ("free",)."""
    from . import curvejoin, snapping
    X = curvejoin.eval_span(s, np.linspace(0.0, 1.0, 9))
    C = curvejoin._cart(s)
    ch = X[-1] - X[0]
    L = float(np.linalg.norm(ch))
    if L <= tol:
        return ("free",)
    t = ch / L
    off = (C - X[0]) - np.outer((C - X[0]) @ t, t)
    if float(np.linalg.norm(off, axis=1).max()) <= tol:
        return ("line", X[0], t)
    cen = snapping.circle_centre(X, 1e-5)
    if cen is not None:
        return ("arc", cen, float(np.linalg.norm(X[0] - cen)))
    return ("free",)


def _shape_changes(a, b, tol):
    """Whether two neighbouring spans are different lines / arcs (a line runs
    into an arc, an arc into another circle's arc ...): not two pieces of one
    line or one circle, and not free-form on both sides."""
    if a[0] == "free" and b[0] == "free":
        return False
    if a[0] != b[0]:
        return True
    if a[0] == "line":
        d = b[1] - a[1]
        return (float(np.linalg.norm(np.cross(a[2], b[2]))) > 1e-6
                or float(np.linalg.norm(d - (d @ a[2]) * a[2])) > tol)
    return float(np.linalg.norm(a[1] - b[1])) > tol or abs(a[2] - b[2]) > tol


def split_wall_at_joints(surf, joint_pts):
    """An extruded wall (u across the extrusion, degree 1; v along the
    profile) as pieces split exactly along its C0 profile knots where the
    profile's pieces meet: at joint_pts (a Join's joints, chained inputs) and
    where a line runs into an arc or one arc into another. Each piece of the
    profile then gets a face of its own with its own edges, so an arc's rim is
    a true circular edge (its centre snaps) even where it joins a line
    tangentially -- Rhino / Shapr3D give one face per segment the same way.
    Folds are split later (cap_walls). Anything else is returned as it is."""
    from . import curvejoin
    sm, ext = _k()
    if not isinstance(surf, sm.Surface):
        return surf
    try:
        d = ext.surface_cvs(surf)
    except Exception:
        return surf
    nu, nv, du, dv = d["count_u"], d["count_v"], int(d["degree_u"]), int(d["degree_v"])
    if du != 1 or dv < 1:
        return surf
    Uu, Uv = [float(x) for x in d["knots_u"]], [float(x) for x in d["knots_v"]]
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    W = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    size = float(np.linalg.norm(np.ptp(P.reshape(-1, 3), axis=0))) or 1.0
    tol = 1e-5 * size
    lo, hi = Uv[dv], Uv[len(Uv) - dv - 1]
    ktol = 1e-12 * max(1.0, abs(hi - lo))
    if abs(Uv[0] - lo) > ktol or abs(Uv[-1] - hi) > ktol:
        return surf                                    # unclamped (periodic): left whole
    inner = sorted({u for u in Uv if lo + ktol < u < hi - ktol})   # as curvejoin._clamped_to_spans
    if not inner:
        return surf
    try:
        spans = curvejoin._clamped_to_spans(dv, P[0], W[0], Uv)
    except Exception:
        return surf
    if len(spans) != len(inner) + 1:
        return surf
    shapes = [_span_shape(s, tol) for s in spans]
    J = np.asarray(joint_pts, float).reshape(-1, 3)

    def near_joint(col):
        for a, b in zip(col[:-1], col[1:]):           # the knot line: straight between the CV rows
            ab = b - a
            L2 = float(ab @ ab) or 1.0
            t = np.clip((J - a) @ ab / L2, 0.0, 1.0)
            if len(J) and float(np.linalg.norm(J - (a + t[:, None] * ab), axis=1).min()) <= tol:
                return True
        return False
    cuts = []
    for i, k in enumerate(inner):
        idx = [j for j, x in enumerate(Uv) if abs(x - k) <= 1e-9 * (hi - lo or 1.0)]
        if len(idx) != dv:
            continue                                   # smoother than C0: one piece of the profile
        col = P[:, idx[0] - 1]                         # the CVs the knot line passes through
        if near_joint(col) or _shape_changes(shapes[i], shapes[i + 1], tol):
            cuts.append(k)
    if not cuts:
        return surf
    pieces, cols = [], [(P.transpose(1, 0, 2), W.T, Uv)]
    for k in cuts:
        Pc, Wc, Uc = cols.pop()
        left, right = store._split_knots(Pc, Wc, Uc, dv, k, 1e-9 * (hi - lo or 1.0))
        cols += [left, right]
    for Pc, Wc, Uc in cols:
        Pq, Wq = Pc.transpose(1, 0, 2), Wc.T
        rational = bool(np.abs(Wq - 1.0).max() > 1e-12)
        pieces.append(ext.surface_from_cvs(du, dv, [tuple(map(float, x)) for x in Pq.reshape(-1, 3)], Uu,
                                           [float(x) for x in Uc],
                                           [float(w) for w in Wq.ravel()] if rational else None))
    return pieces


def cap_walls(pieces):
    """A closed solid (('brep_text', world text)) from walls in world space --
    surfaces, Breps or ('brep_text', text) -- with flat caps on their open
    ends (the worker's cap_solid)."""
    from . import jobs
    from .steps import IDENT
    sm, ext = _k()
    flat = []
    for piece in pieces:
        # a wall folded at its corners (an extruded rectangle) as one face per side,
        # so each side is a face of its own (Push / Pull, Edge Fillet picks)
        split = store._crease_pieces(piece, LOFT_FOLD_DEG) if isinstance(piece, sm.Surface) else None
        flat += split or [piece]
    walls = {}
    for k, piece in enumerate(flat):
        if isinstance(piece, tuple) and piece and piece[0] == "brep_text":
            text = piece[1]
        else:
            b = piece if isinstance(piece, sm.Brep) else ext.brep_from_surface(piece)
            text = store.brep_to_text(b)
        walls[f"w{k}"] = (text, IDENT)
    _, breps = jobs.run("cap_solid", walls, ["result"], order=list(walls), into="_world_")
    return ("brep_text", breps["result"])


def _is_are(names):
    return " is open " if len(names) == 1 else " are open "


def _closed_side(side):
    try:
        return is_closed_curve(profile_curve(side))
    except Exception:
        return False


def check_cap_inputs(op, inputs, p):
    """Refuse a Sweep / Loft solid of open profiles up front, with a reason."""
    if op == "sweep":
        if not is_closed_curve(input_curve(*inputs[0])):
            raise RuntimeError(f"a solid needs a closed profile: {inputs[0][0].name} is open "
                               "(Body: Surface sweeps it as a wall)")
    elif op == "loft":
        sides = loft_sides(inputs, p)
        sides = sides[:len(sides) - loft_rail_count(inputs, p)]
        open_ = [s[0][0].name for s in sides if not _closed_side(s)]
        if open_:
            raise RuntimeError("a solid needs closed sections: " + ", ".join(open_) + _is_are(open_) +
                               "(Body: Surface lofts them as a wall)")


# Features whose wall becomes a solid with caps=True (Extrude and Revolve do it
# in their builders: they group profiles into outlines and holes; Pipe too: its
# caps can be round)
CAP_OPS = {"sweep", "loft"}


@builder("offset")
def build_offset(inputs, p):
    """Result in the source object's local space (same matrix). One plain
    surface: the kernel's exact offset surface. Trimmed faces / joined sheets /
    solids: every face offset, trims kept, corners extended until they meet."""
    sm, _ = _k()
    obj, _ = inputs[0]
    d = p.get("distance", 0.1)
    if store.is_natural(obj) and store.get_brep(obj).face_count() == 1:
        surf = store.fresh_surfaces(obj)[0]  # fresh: the offset keeps a pointer to its base
        return sm.create_offset_surface(surf, d)
    from . import jobs
    _, breps = jobs.run("offset", {"target": obj}, ["result"], distance=d)
    return ("brep_text", breps["result"])


@builder("thicken")
def build_thicken(inputs, p):
    """A closed solid from a (joined) sheet: the sheet, its offset and the walls
    between them (in the source object's local space)."""
    obj, _ = inputs[0]
    from . import jobs
    _, breps = jobs.run("thicken", {"target": obj}, ["result"], distance=p.get("distance", 0.1),
                        symmetric=bool(p.get("symmetric", False)))
    return ("brep_text", breps["result"])


@builder("planar")
def build_planar(inputs, p):
    sm, _ = _k()
    curves = [input_curve(o, r) for o, r in inputs]
    return sm.create_planar_faces(curves)


# a Blend Curve side's own settings besides G0-G3 / weight / curvature / rate / flip:
#   bc_mode_<s>  EDGE (continue the curve / edge), SURFACE (tangent to the edge's
#                surface, across the edge), NORMAL (perpendicular to the surface)
#   bc_on_<s>    start at a point along the curve / edge (bc_at_<s>: 0..1 of its
#                parameter) instead of its end (end_<s>)
#   uv_<s>       a face input (F<i>): the point on it (fractions of its domain)
#   face_<s>     an edge between two faces: which face SURFACE / NORMAL use
#   bc_ang_<s>   SURFACE on an edge: degrees from the edge round the normal
#                (missing: 90, square to it)
BLEND_SIDE_KEYS = ("bc_mode_a", "bc_mode_b", "bc_on_a", "bc_on_b", "bc_at_a", "bc_at_b", "uv_a", "uv_b",
                   "face_a", "face_b", "bc_ang_a", "bc_ang_b")


def _blend_side(inputs, p, s):
    """keyword arguments of align.side_frame for side s ('a' / 'b')."""
    obj, ref = inputs[0 if s == "a" else 1]
    on = bool(p.get(f"bc_on_{s}")) and not (ref.startswith("F") and ref[1:].isdigit())
    return obj, ref, dict(end=int(p.get(f"end_{s}", 1 if s == "a" else 0)),
                          at=float(p.get(f"bc_at_{s}", 0.5)) if on else None, uv=p.get(f"uv_{s}"),
                          mode=p.get(f"bc_mode_{s}", "EDGE"), face=int(p.get(f"face_{s}", 0) or 0),
                          third=p.get(f"cont_{s}") == "G3", angle=p.get(f"bc_ang_{s}"))


def blend_frames(inputs, p):
    """A Blend Curve's two end frames: (P, C', C''[, C''']) at each side,
    turned round where Flip is on. A side is a curve or surface edge (its end,
    outward, or a point along it), or a face point; on a surface it is the
    edge continued, tangent to the surface or perpendicular to it
    (align.side_frame). Directions that are not outward by themselves (a point
    along a curve, the normal) lead toward the other side."""
    from . import align
    oa, ra, ka = _blend_side(inputs, p, "a")
    ob, rb, kb = _blend_side(inputs, p, "b")
    pa = align.side_point(oa, ra, ka["end"], ka["at"], ka["uv"])
    pb = align.side_point(ob, rb, kb["end"], kb["at"], kb["uv"])
    fa = align.side_frame(oa, ra, toward=pb, **ka)
    fb = align.side_frame(ob, rb, toward=pa, **kb)
    flip = geom.flip_frame                                               # odd derivatives turn round
    if p.get("flip_a"):
        fa = flip(fa)
    if p.get("flip_b"):
        fb = flip(fb)
    return fa, fb


@builder("blend_curve")
def build_blend_curve(inputs, p):
    """Returns ('curve', degree, world CVs): written into a Blender curve object."""
    fa, fb = blend_frames(inputs, p)
    deg, pts = geom.blend_curve_cvs(fa, fb, p.get("cont_a", "G1"), p.get("cont_b", "G1"),
                                    p.get("w_a", 1.0), p.get("w_b", 1.0), p.get("c_a", 0.0), p.get("c_b", 0.0),
                                    p.get("d_a", 0.0), p.get("d_b", 0.0))
    return ("curve", deg, pts)


# --------------------------------------------------------------------------
# History records
# --------------------------------------------------------------------------

PROP_HIST = "nproto_hist"


PROP_PLACE = "nproto_place"   # 4x4 (16 numbers): where the feature's result sits in the object's local space


def placement(obj):
    """The feature's placement in obj (numpy 4x4), None = as built. Set Origin /
    Apply Transform change the object's matrix and move its mesh the other way
    (pivot.py): the result keeps its place in the world through this."""
    v = obj.get(PROP_PLACE)
    if v is None:
        return None
    A = np.array(list(v), float).reshape(4, 4)
    return None if np.abs(A - np.eye(4)).max() <= 1e-14 else A


def set_placement(obj, A):
    if A is None or np.abs(np.asarray(A, float) - np.eye(4)).max() <= 1e-14:
        if PROP_PLACE in obj:
            del obj[PROP_PLACE]
    else:
        obj[PROP_PLACE] = [float(x) for x in np.asarray(A, float).ravel()]


def placed(obj, g):
    """A builder's result g put where obj's feature sits (its placement)."""
    A = placement(obj)
    if A is None:
        return g
    turn = np.linalg.det(A[:3, :3]) < 0.0   # a mirror: turned over too, facing the side Blender showed (pivot.py)
    if isinstance(g, tuple):
        if not g or g[0] != "brep_text":
            return g                    # curves: not moved (Blender moves a curve's own points)
        b = store.brep_from_text(g[1])
        store.transform(b, A)
        text = store.brep_to_text(b)
    else:
        store.transform(g, A)           # a fresh Surface / Brep of the builder's
        sm = _k()[0]
        if not turn:
            return g
        if not isinstance(g, sm.Brep):
            from .square import reverse_u
            return reverse_u(g)
        text = store.brep_to_text(g)
    if turn:
        from .steps import flip_text
        text = flip_text(text)
    return ("brep_text", text)


def record(obj, op, inputs, params):
    if PROP_PLACE in obj:
        del obj[PROP_PLACE]      # the caller built it as the feature gives it
    for key in [k for k in obj.keys() if k.startswith("nproto_in_")]:
        del obj[key]
    for i, (src, _) in enumerate(inputs):
        obj[f"nproto_in_{i}"] = src
    obj[PROP_HIST] = json.dumps({"op": op, "refs": [r for _, r in inputs], "params": params})
    if op in ORIENT_OPS:     # the caller just built it with run(): how it came out
        obj[PROP_FLIPPED] = int(LAST_FLIP[0] or 0)
    mark_built(obj)          # the caller just built it from exactly these inputs
    try:  # show it in the Feature panel right away
        from .feature_ui import sync_from_history
        sync_from_history(obj)
    except Exception:
        pass


# --------------------------------------------------------------------------
# Kept inputs: Join / Boolean / Boundary Fill / Thicken keep what they were
# built from (their history rebuilds from it) in one hidden collection
# --------------------------------------------------------------------------

HISTORY_COLL = "NURBS History"
PROP_HOME = "nproto_home"      # JSON list: the collections an input was in before it was kept
KEEP_OPS = ("join", "join_curves", "boolean", "thicken", "boundary_fill", "split_region", "iso_detach",
            "extrude_face", "emboss")   # results that keep their inputs


def history_collection(context, create=True):
    """The scene's hidden NURBS History collection (made on first use)."""
    import bpy
    scene = context.scene
    coll = next((c for c in scene.collection.children_recursive if c.name.startswith(HISTORY_COLL)
                 and c.get("nproto_history_coll")), None)
    if coll is None and create:
        coll = bpy.data.collections.new(HISTORY_COLL)
        coll["nproto_history_coll"] = 1
        scene.collection.children.link(coll)
    if coll is not None:
        coll.hide_render = True
        for vl in scene.view_layers:
            lc = _layer_coll(vl.layer_collection, coll)
            if lc is not None and not lc.hide_viewport:
                lc.hide_viewport = True          # its eye off: Unhide All leaves the inputs hidden
    return coll


def include_history(context):
    """Put the History collection back in the view layers that exclude it (its
    checkbox off in the Outliner): kept inputs must stay in the view layer so
    the Construction Tree can make them active and show them. Its eye stays
    off, so they stay hidden. Call it from operators only (not from draw)."""
    coll = history_collection(context, create=False)
    if coll is None:
        return False
    changed = False
    for vl in context.scene.view_layers:
        lc = _layer_coll(vl.layer_collection, coll)
        if lc is not None and lc.exclude:
            lc.exclude = False
            lc.hide_viewport = True
            changed = True
    return changed


def _layer_coll(lc, coll):
    if lc.collection is coll:
        return lc
    for ch in lc.children:
        r = _layer_coll(ch, coll)
        if r is not None:
            return r
    return None


def keep_inputs(context, objs):
    """Hide the inputs a result was built from in the NURBS History collection
    (their old collections are remembered for Restore Inputs)."""
    coll = history_collection(context)
    include_history(context)
    for o in objs:
        homes = [c.name for c in o.users_collection if c is not coll]
        if homes:
            o[PROP_HOME] = json.dumps(homes)
        if coll not in o.users_collection:
            coll.objects.link(o)
        for c in list(o.users_collection):
            if c is not coll:
                c.objects.unlink(o)
        o.hide_render = True
        if PROP_PEEK in o:
            del o[PROP_PEEK]
        try:
            o.hide_set(True)
        except RuntimeError:
            pass


PROP_PEEK = "nproto_peek"      # a kept input shown for a look (also linked to the scene collection)


def set_shown(context, obj, show):
    """Show / hide an object from the Construction Tree. A kept input lives in
    the hidden History collection, so showing it also links it to the scene
    collection for the moment; hiding it again takes that link away."""
    coll = history_collection(context, create=False)
    include_history(context)
    kept = coll is not None and obj.name in coll.objects
    scol = context.scene.collection
    if show and kept and not keepers_of(obj):
        # nothing keeps it any more (its Join was removed): back where it came from, for good
        release_inputs(context, [obj])
        return
    if show:
        if kept and obj.name not in scol.objects:
            scol.objects.link(obj)
            obj[PROP_PEEK] = 1
        try:
            obj.hide_set(False)
        except RuntimeError:
            pass
    else:
        try:
            obj.hide_set(True)
        except RuntimeError:
            pass
        if obj.get(PROP_PEEK):
            if kept and obj.name in scol.objects:
                scol.objects.unlink(obj)
            del obj[PROP_PEEK]


def is_shown(obj):
    try:
        return obj.visible_get()
    except RuntimeError:
        return False


def release_inputs(context, objs):
    """Back from the History collection to where they came from, shown."""
    import bpy
    coll = history_collection(context, create=False)
    for o in objs:
        homes = [bpy.data.collections.get(n) if n != context.scene.collection.name else context.scene.collection
                 for n in json.loads(o.get(PROP_HOME, "[]"))]
        homes = [c for c in homes if c is not None]
        if not homes and not any(c is not coll for c in o.users_collection):
            homes = [context.collection or context.scene.collection]   # an input never stashed stays put
        for c in homes:
            if o.name not in c.objects:
                c.objects.link(o)
        if coll is not None and o.name in coll.objects:
            coll.objects.unlink(o)
        if PROP_HOME in o:
            del o[PROP_HOME]
        if PROP_PEEK in o:
            del o[PROP_PEEK]
        o.hide_render = False
        try:
            o.hide_set(False)
        except RuntimeError:
            pass


def is_kept(context, obj):
    """obj is stashed as a kept input (in the NURBS History collection, or
    hidden there with its old collections remembered)."""
    coll = history_collection(context, create=False)
    return (coll is not None and obj.name in coll.objects) or PROP_HOME in obj


def keepers_of(obj):
    """The features that keep obj as a hidden input: a Join, Thicken, Boolean...
    (KEEP_OPS) built from it. A Loft or Boundary from its edges leaves it shown."""
    import bpy
    out = []
    for d in bpy.data.objects:
        if d is obj or not any(d.get(k) is obj for k in d.keys() if k.startswith("nproto_in_")):
            continue
        try:
            op = json.loads(d.get(PROP_HIST, "{}")).get("op")
        except (ValueError, TypeError):
            op = None
        if op in KEEP_OPS:
            out.append(d)
    return out


def users_of(obj):
    """Objects whose history (feature or steps) takes obj as an input."""
    import bpy
    return [d for d in bpy.data.objects if d is not obj
            and any(d.get(k) is obj for k in d.keys() if k.startswith(("nproto_in_", "nproto_sin_")))]


def purge_inputs(context, candidates):
    """Delete the kept inputs among candidates (hidden, or in the History
    collection) that no history uses any more -- and, in turn, the kept
    inputs they were built from. Visible objects (curves you work with)
    stay. Returns the deleted names."""
    import bpy
    coll = history_collection(context, create=False)
    todo, gone = list(candidates), []
    while todo:
        o = todo.pop()
        try:
            name = o.name
        except ReferenceError:
            continue
        if name not in bpy.data.objects or users_of(o):
            continue
        try:
            hidden = o.hide_get()
        except RuntimeError:            # not in this view layer
            hidden = True
        kept = (coll is not None and name in coll.objects) or hidden
        if not kept:
            continue
        todo += history_inputs(o)
        data = o.data
        bpy.data.objects.remove(o)
        if data is not None and getattr(data, "users", 1) == 0:
            try:
                (bpy.data.meshes if isinstance(data, bpy.types.Mesh) else bpy.data.curves).remove(data)
            except Exception:
                pass
        gone.append(name)
    return gone


def history(obj):
    if PROP_HIST not in obj:
        return None
    h = json.loads(obj[PROP_HIST])
    ins = []
    for i, ref in enumerate(h["refs"]):
        src = obj.get(f"nproto_in_{i}")
        if src is None:
            return None
        ins.append((src, ref))
    h["inputs"] = ins
    return h


SPLIT_OPS = {"split_region", "iso_detach"}     # features whose result is a piece trimmed out of their first input


def untrimmed_base(obj):
    """(text, natural) of the surface a split piece was cut out of -- what
    Untrim and Untrim Region give back -- in obj's local space; None when
    obj's feature is not a split (or lost its input)."""
    h = history(obj)
    if h is None or h["op"] not in SPLIT_OPS or not h["inputs"]:
        return None
    src = h["inputs"][0][0]
    if h["op"] == "iso_detach":          # a detached piece: its own carrier, not the whole source
        from .isodetach import piece_carrier_text
        text = piece_carrier_text(src, h["params"])
        return None if text is None else (placed(obj, ("brep_text", text))[1], True)
    text = src.get(store.PROP_UNTRIM)
    if text:
        nat = bool(src.get(store.PROP_UNTRIM + "_natural", 0))
    else:
        text, nat = src.get(store.PROP_BREP), store.is_natural(src)
    if not text:
        return None
    g = placed(obj, ("brep_text", text))      # the split's pieces are in the split surface's local space
    return g[1], nat


NATURAL_STEPS = {"refit_trim", "merge_patches", "rebuild_surface", "extend_surface"}   # result is an untrimmed surface


def _natural_result(obj, step):
    """A NATURAL_STEPS step's result is untrimmed -- but Extend of a trimmed
    surface keeps it trimmed (and after Untrim the same step extends the plain one)."""
    if step["op"] != "extend_surface":
        return True
    from .extend import result_natural
    return result_natural(obj)


KEEP_STEPS = {"flip_normal"}     # result is plain / trimmed exactly when its input was
PROP_STACK = "nproto_stack"   # JSON list of in-place steps: {"op", "refs", "params"}
PROP_BASE = "nproto_base"     # brep text the stack starts from when there is no base feature


def stack(obj):
    """Stack steps with resolved inputs. Raises if an input was deleted."""
    if PROP_STACK not in obj:
        return []
    out = []
    for k, e in enumerate(json.loads(obj[PROP_STACK])):
        ins = []
        for i, ref in enumerate(e["refs"]):
            src = obj.get(f"nproto_sin_{k}_{i}")
            if src is None:
                raise RuntimeError(f"history step {k + 1} ({e['op']}) lost an input")
            ins.append((src, ref))
        out.append({"op": e["op"], "params": e["params"], "inputs": ins})
    return out


def _write_stack(obj, entries):
    for key in [k for k in obj.keys() if k.startswith("nproto_sin_")]:
        del obj[key]
    raw = []
    for k, e in enumerate(entries):
        for i, (src, _) in enumerate(e["inputs"]):
            obj[f"nproto_sin_{k}_{i}"] = src
        raw.append({"op": e["op"], "refs": [r for _, r in e["inputs"]], "params": e["params"]})
    if raw:
        obj[PROP_STACK] = json.dumps(raw)
    elif PROP_STACK in obj:
        del obj[PROP_STACK]


def push_step(obj, op, inputs, params, base_text=None):
    """Record an in-place step. base_text = the geometry before the first step
    (only used when the object has no base feature of its own)."""
    from . import store
    entries = stack(obj)
    # no base feature, or one that can no longer run (an input was deleted:
    # history() is None and the rebuild starts from the stored base)
    if PROP_BASE not in obj and history(obj) is None:
        obj[PROP_BASE] = base_text if base_text is not None else obj[store.PROP_BREP]
        obj[PROP_BASE + "_natural"] = 1 if store.is_natural(obj) else 0
    entries.append({"op": op, "inputs": list(inputs), "params": params})
    _write_stack(obj, entries)
    mark_built(obj)


def remove_step(obj, index):
    entries = stack(obj)
    if 0 <= index < len(entries):
        del entries[index]
    _write_stack(obj, entries)


def remove_steps(obj, ops):
    entries = [e for e in stack(obj) if e["op"] not in ops]
    _write_stack(obj, entries)


def copy_history(src, dst):
    """Give dst the same base feature, snapshot and stack as src."""
    clear_history(dst)
    for k in src.keys():
        if k.startswith(("nproto_in_", "nproto_sin_")) or k in (PROP_HIST, PROP_STACK, PROP_BASE, PROP_BASE + "_natural"):
            dst[k] = src[k]


def history_enabled(context=None):
    import bpy
    s = getattr((context or bpy.context).scene, "nproto_settings", None)
    return bool(s and s.construction_history)


def base_text(obj):
    """Brep text of an object's BASE geometry (its own feature, or the snapshot
    before its history steps) -- without any trims / fillets applied."""
    h = history(obj)
    if h is not None:
        g = run(h["op"], h["inputs"], h["params"])
        if h["op"] == "square":
            from .square import store_info
            store_info(obj)
        elif h["op"] == "patch":
            from .patch import store_info
            store_info(obj)
        elif h["op"] == "unroll":
            from .unroll import store_info
            store_info(obj)
        g = placed(obj, g)
        if isinstance(g, tuple) and g and g[0] == "brep_text":
            return g[1]
        if isinstance(g, tuple):
            return None  # curves have no brep
        return store.brep_to_text(store.to_brep(g))
    if PROP_BASE in obj:
        return obj[PROP_BASE]
    return obj.get(store.PROP_BREP)


def has_history(obj):
    return PROP_HIST in obj or PROP_STACK in obj


def bake(obj, step=-1):
    """Bake (Alias' Delete History): keep obj as it is and drop its history
    up to `step` -- step -1: just its own feature (it stops following its
    inputs); step k: the feature and steps 0..k, which become a snapshot the
    later steps keep building on. Returns the objects it no longer uses."""
    from . import steps as _steps
    entries = stack(obj)
    h = history(obj)
    freed = [src for src, _ in h["inputs"]] if h is not None else []
    if step < 0 and not entries or obj.type == 'CURVE':
        clear_history(obj)                 # nothing builds on top: the geometry is all there is
        return freed
    # the geometry the kept steps build on
    if h is not None:
        g = placed(obj, run(h["op"], h["inputs"], h["params"]))
        if isinstance(g, tuple) and g and g[0] == "brep_text":
            text, natural = g[1], False
        elif isinstance(g, tuple):
            clear_history(obj)
            return freed
        else:
            sm = _k()[0]
            text, natural = store.brep_to_text(store.to_brep(g)), not isinstance(g, sm.Brep)
    else:
        text = obj.get(PROP_BASE, obj.get(store.PROP_BREP))
        natural = bool(obj.get(PROP_BASE + "_natural", 0))
    upto = min(step, len(entries) - 1)
    for e in entries[:upto + 1]:
        freed += [src for src, _ in e["inputs"]]
        if not e["params"].get("_off"):
            text = _steps.APPLY[e["op"]](obj, text, e["inputs"], e["params"])
            natural = False
    keep = entries[upto + 1:]
    for key in [k for k in obj.keys() if k.startswith("nproto_in_")] + [PROP_HIST]:
        if key in obj:
            del obj[key]
    _write_stack(obj, keep)
    if keep:
        obj[PROP_BASE] = text
        obj[PROP_BASE + "_natural"] = 1 if natural else 0
    else:
        for key in (PROP_BASE, PROP_BASE + "_natural"):
            if key in obj:
                del obj[key]
    mark_built(obj)
    return [o for o in dict.fromkeys(freed) if o is not None]


def clear_history(obj):
    for key in [k for k in obj.keys() if k.startswith(("nproto_in_", "nproto_sin_")) or
                k in (PROP_HIST, PROP_STACK, PROP_BASE, PROP_BASE + "_natural", PROP_PLACE)]:
        del obj[key]


def history_inputs(obj):
    return [obj[k] for k in obj.keys() if k.startswith(("nproto_in_", "nproto_sin_")) and obj[k] is not None]


def depends_on(obj, other, limit=500):
    """Is obj built (directly or through other objects) from `other`?"""
    seen, todo = set(), [obj]
    while todo:
        o = todo.pop()
        for src in history_inputs(o):
            if src is other:
                return True
            if src.name not in seen and len(seen) < limit:
                seen.add(src.name)
                todo.append(src)
    return False


# Features whose surface faces whichever way its inputs happen to run: built
# from edges of other surfaces it is turned to face the same way as they do
# (matcaps / shading / a later Join agree across the edges); Flip Normal turns
# it over on top of that. Patch, Square and Loft Rails orient themselves.
ORIENT_OPS = {"loft", "boundary", "sweep", "dome"}
PROP_FLIPPED = "nproto_flipped"   # 1 = the last build reversed u (see _rebuild)
LAST_FLIP = [None]                # the last run()'s decision for an ORIENT_OPS surface (None: not one)


def run(op, inputs, params):
    if op == "loft" and has_slides(inputs, params):
        with slid_ends(inputs, params) as (ins, prm):
            return _run(op, ins, prm)
    return _run(op, inputs, params)


def _run(op, inputs, params):
    if op not in BUILDERS:
        raise RuntimeError(f"no builder for feature '{op}' (reload the add-on)")
    capped = bool(params.get("caps")) and op in CAP_OPS
    if capped:
        check_cap_inputs(op, inputs, params)
    g = BUILDERS[op](inputs, params)
    LAST_FLIP[0] = None
    if capped:
        return cap_walls([g])      # the worker turns the solid outward
    if op in ORIENT_OPS:
        sm, _ = _k()
        if isinstance(g, sm.Surface):
            flip = edge_vote(g, inputs) < 0
            if params.get("flip_normal"):
                flip = not flip
            if flip:
                from .square import reverse_u
                g = reverse_u(g)
            LAST_FLIP[0] = int(flip)
    return g


def edge_vote(surf, inputs, m=12):
    """> 0: surf faces the same way as the surfaces whose edges are among the
    inputs, < 0: the other way, 0: no surface edges (or nothing to compare)."""
    from .square import sample_side, grid_starts, _foot
    vote = 0.0
    for obj, ref in inputs:
        if not is_surface_edge(ref) or not store.is_nurbs(obj):
            continue
        try:
            sd = sample_side(obj, ref, m)
        except Exception:
            continue
        if sd.Nout is None:
            continue
        ks = list(range(1, len(sd.P) - 1))
        starts = grid_starts(surf, [sd.P[k] for k in ks])
        for k, uv0 in zip(ks, starts):
            try:
                _, su, sv = surf.derivatives(*_foot(surf, np.asarray(sd.P[k], float), uv0))
            except Exception:
                continue
            n = np.cross(su, sv)
            vote += float(n @ sd.Nout[k]) / (np.linalg.norm(n) or 1.0)
    return vote


_U_SWAP = {"U0": "U1", "U1": "U0"}


def _swap_u(ref):
    return _U_SWAP.get(ref, ref) if isinstance(ref, str) else ref


def _dependents(obj):
    """Objects with history that take obj as an input."""
    import bpy
    out = []
    for o in bpy.data.objects:
        if o is obj:
            continue
        if any(o.get(k) is obj for k in o.keys() if k.startswith(("nproto_in_", "nproto_sin_"))):
            out.append(o)
    return out


def _edge_marks(obj):
    """Each display edge of obj as (start, mid, end) in local space."""
    try:
        polys = store.display(obj).get("edges", [])
    except Exception:
        return []
    out = []
    for pl in polys:
        P = np.asarray(pl, float)
        if len(P):
            out.append(np.array([P[0], P[len(P) // 2], P[-1]]))
    return out


def _swap_own(obj, entries):
    """obj's base surface turned over (u reversed): its iso edges U0 / U1
    swapped. Its stack steps that name them (Match, Extend) follow."""
    for e in entries:
        prm = e["params"]
        if isinstance(prm.get("edges"), list):
            prm["edges"] = [_swap_u(x) for x in prm["edges"]]
        if "edge" in prm:
            prm["edge"] = _swap_u(prm["edge"])
    if entries:
        _write_stack(obj, entries)


def edge_moves(old_marks, new_marks):
    """{old number: new number} for B-rep edges renumbered by a rebuild (a trim
    imprinted another way, an input that changed shape). An edge keeps its
    number unless another edge is clearly nearer to where it was (less than
    half as far); two edges never move onto the same one (then both stay)."""
    if not old_marks or not new_marks:
        return {}
    pts = np.concatenate([np.asarray(m, float) for m in old_marks])
    tiny = 1e-9 * max(float(np.ptp(pts, axis=0).max()), 1e-12)

    def dist(a, b):
        return min(np.abs(a - b).max(), np.abs(a - b[::-1]).max())

    moves = {}
    for i, a in enumerate(old_marks):
        d = [dist(a, b) for b in new_marks]
        j = int(np.argmin(d))
        if j != i and not (i < len(new_marks) and d[i] <= 2.0 * d[j] + tiny):
            moves[i] = j
    taken = {}
    for i, j in moves.items():
        taken.setdefault(j, []).append(i)
    kept = set(range(min(len(old_marks), len(new_marks)))) - set(moves)   # edges that stay put keep theirs
    return {i: j for i, j in moves.items() if len(taken[j]) == 1 and j not in kept}


def _names_edges(obj):
    """Does a feature built from obj name its B-rep edges (E#)?"""
    return any('"E' in str(o.get(k, "")) for o in _dependents(obj) for k in (PROP_HIST, PROP_STACK))


def _remap_dependents(obj, old_marks, turned=True):
    """obj rebuilt: its B-rep edges may be numbered differently, and when it
    turned over (turned) its iso edges U0 / U1 swapped too. Every feature
    built from its edges is renamed to keep the same edges (old_marks: obj's
    edges before, from _edge_marks)."""
    moves = edge_moves(old_marks, _edge_marks(obj))

    def remap(ref):
        if turned and is_edge_ref(ref):
            return _swap_u(ref)
        if is_brep_edge_ref(ref):
            try:
                return f"E{moves[int(ref[1:])]}"
            except (ValueError, KeyError):
                return ref
        return ref

    for o in _dependents(obj):
        if PROP_HIST in o:
            h = json.loads(o[PROP_HIST])
            refs = [remap(r) if o.get(f"nproto_in_{i}") is obj else r for i, r in enumerate(h["refs"])]
            if refs != h["refs"]:
                h["refs"] = refs
                o[PROP_HIST] = json.dumps(h)
        if PROP_STACK in o:
            raw = json.loads(o[PROP_STACK])
            changed = False
            for k, e in enumerate(raw):
                refs = [remap(r) if o.get(f"nproto_sin_{k}_{i}") is obj else r for i, r in enumerate(e["refs"])]
                if refs != e["refs"]:
                    e["refs"], changed = refs, True
            if changed:
                o[PROP_STACK] = json.dumps(raw)


PROP_KEY = "nproto_build_key"


def build_key(obj, _seen=None, matrix=None, place=False):
    """Fingerprint of everything a rebuild depends on: the feature and its steps,
    the object's transform and -- recursively -- its inputs (their geometry,
    transforms and their own inputs). Equal key = the rebuild would produce what
    the object already holds. Recursion matters for mutually dependent objects
    (two surfaces trimmed by one fillet), whose rebuild order is arbitrary."""
    import hashlib
    seen = set() if _seen is None else _seen
    seen.add(obj.name)
    hsh = hashlib.sha1()
    hsh.update(str(obj.get(PROP_HIST, "")).encode())
    hsh.update(str(obj.get(PROP_STACK, "")).encode())
    hsh.update(hashlib.sha1(str(obj.get(PROP_BASE, "")).encode()).digest())
    # (matrix / place: the key as it was under another matrix and placement -- pivot.py)
    M = geom.world_matrix(obj) if matrix is None else matrix
    hsh.update(np.round(np.array(M, float), 10).tobytes())
    C = placement(obj) if place is False else place
    if C is not None:
        hsh.update(np.round(np.array(C, float).ravel(), 10).tobytes())
    for src in history_inputs(obj):
        hsh.update(src.name.encode())
        hsh.update(np.round(np.array(geom.world_matrix(src), float), 10).tobytes())
        if store.is_nurbs(src):
            hsh.update(hashlib.sha1(str(src.get(store.PROP_BREP, "")).encode()).digest())
            if src.name not in seen and has_history(src):
                hsh.update(build_key(src, seen).encode())
        elif src.type == 'CURVE':
            from .diagnostics import _curve_key
            hsh.update(str(_curve_key(src)).encode())
            if src.name not in seen and has_history(src):
                hsh.update(build_key(src, seen).encode())
        elif src.type == 'MESH' and '"subd_to_brep"' in str(obj.get(PROP_HIST, "")):
            from .subd_brep import cage_key          # a live SubD to BREP body: its cage (any quad mesh)
            hsh.update(cage_key(src).encode())
    return hsh.hexdigest()


def mark_built(*objs):
    """Record that these objects' current geometry IS their rebuild result (an
    operator just computed it), so the history handler does not redo the work."""
    for o in objs:
        if o is not None and has_history(o):
            try:
                o[PROP_KEY] = build_key(o)
            except Exception:
                pass


PROP_PENDING = "nproto_pending"   # set while Pause Rebuild holds a settings edit back


def paused(context=None):
    """Pause Rebuild is on: settings edits wait for Rebuild."""
    import bpy
    try:
        s = getattr((context or bpy.context).scene, "nproto_settings", None)
    except AttributeError:
        return False
    return bool(s and getattr(s, "pause_rebuild", False))


def pending(obj):
    """obj's settings changed while Pause Rebuild was on: its geometry is older than its history."""
    return bool(obj.get(PROP_PENDING))


def edit_rebuild(obj, context=None, force=False):
    """After a settings edit: rebuild now, or -- while Pause Rebuild is on --
    only note that obj waits (Rebuild in the Feature panel, or switching the
    pause off, builds it)."""
    if paused(context):
        obj[PROP_PENDING] = 1
        return False
    return rebuild(obj, context, force=force)


def rebuild(obj, context=None, force=False):
    """Base feature (or snapshot), then every stack step in order. Skipped when
    nothing it depends on changed since the last build (unless force)."""
    from . import steps
    h = history(obj)
    entries = stack(obj)
    if h is None and not entries:
        return False
    if obj.type == 'CURVE' and obj.mode == 'EDIT' and h is not None and h["op"] == "align_curve":
        return False         # never rewrite a curve while its CVs are being edited
    key = build_key(obj)
    if not force and obj.get(PROP_KEY) == key:
        if PROP_PENDING in obj:
            del obj[PROP_PENDING]
        return False
    result = _rebuild(obj, context, h, entries, steps)
    obj[PROP_KEY] = build_key(obj)
    if PROP_PENDING in obj:
        del obj[PROP_PENDING]
    return result


def _rebuild(obj, context, h, entries, steps):
    turned = [None]
    before = _edge_marks(obj) if obj.type == 'MESH' and _names_edges(obj) else None
    result = _rebuild_inner(obj, context, h, entries, steps, turned)
    if turned[0] is not None:
        _remap_dependents(obj, turned[0])
    elif before:
        _remap_dependents(obj, before, turned=False)     # features on its edges follow them
    return result


def _rebuild_inner(obj, context, h, entries, steps, turned):
    natural = False
    if h is not None:
        g = run(h["op"], h["inputs"], h["params"])
        if h["op"] in ORIENT_OPS and LAST_FLIP[0] is not None and LAST_FLIP[0] != int(obj.get(PROP_FLIPPED, 0)):
            # it came out turned the other way than last time: the stack's
            # steps name the base's edges -- swap U0 / U1 before they run
            turned[0] = _edge_marks(obj)
            obj[PROP_FLIPPED] = LAST_FLIP[0]
            _swap_own(obj, entries)
        if h["op"] == "square":
            from .square import store_info
            store_info(obj)
        elif h["op"] == "patch":
            from .patch import store_info
            store_info(obj)
        elif h["op"] == "unroll":
            from .unroll import store_info
            store_info(obj)
        if isinstance(g, tuple) and g and g[0] == "inplace":
            g[1](obj)
            return True
        if isinstance(g, tuple) and g and g[0] == "curve":
            geom.write_curve_object(obj, g[1], g[2])
            return True
        if isinstance(g, tuple) and g and g[0] in ("spline", "splines"):
            from .primitives import write_spline_object
            write_spline_object(obj, g[1])
            return True
        g = placed(obj, g)
        if isinstance(g, tuple) and g and g[0] == "brep_text":
            base_text = g[1]
        else:
            if not entries:
                store.set_geometry(obj, g, context)
                if store.PROP_UNTRIM in obj:
                    del obj[store.PROP_UNTRIM]
                return True
            sm = _k()[0]
            natural = not isinstance(g, sm.Brep)
            base_text = store.brep_to_text(store.to_brep(g))
    else:
        base_text = obj[PROP_BASE]
        natural = bool(obj.get(PROP_BASE + "_natural", 0))
    text = base_text
    steps.REBUILD_BASE[0] = base_text
    try:
        for e in entries:
            if e["params"].get("_off"):   # muted in the history list
                continue
            text = steps.APPLY[e["op"]](obj, text, e["inputs"], e["params"])
    finally:
        steps.REBUILD_BASE[0] = None
    store.set_geometry_text(obj, text, context, keep_untrimmed=False)
    active = [e for e in entries if not e["params"].get("_off")]
    # Flip Normal keeps a surface as it is (plain or trimmed): look past it
    core = [e for e in active if e["op"] not in KEEP_STEPS]
    flips = sum(1 for e in active if e["op"] == "flip_normal") % 2
    if core and core[-1]["op"] == "untrim":   # a split piece back to the surface it was cut out of
        whole = untrimmed_base(obj)
        obj[store.PROP_NATURAL] = 1 if whole is None or whole[1] else 0
        if store.PROP_UNTRIM in obj:
            del obj[store.PROP_UNTRIM]
    elif (core and core[-1]["op"] in NATURAL_STEPS and _natural_result(obj, core[-1])) or (not core and natural):
        obj[store.PROP_NATURAL] = 1          # a plain surface (again)
        if store.PROP_UNTRIM in obj:
            del obj[store.PROP_UNTRIM]
    elif entries or (h is not None and h["op"] in SPLIT_OPS):
        # a split piece's untrimmed surface is the one it was cut out of (as it
        # is now: it follows that surface), not the piece the split gives
        whole = untrimmed_base(obj) if h is not None else None
        if whole is not None:
            base_text, natural = whole
        if flips and natural:                # Untrim gives the surface facing the same way
            try:
                base_text = steps.flip_text(base_text)
            except Exception:
                pass
        obj[store.PROP_NATURAL] = 0
        obj[store.PROP_UNTRIM] = base_text
        obj[store.PROP_UNTRIM + "_natural"] = 1 if natural else 0
    else:
        obj[store.PROP_NATURAL] = 1 if natural else 0
    return True
