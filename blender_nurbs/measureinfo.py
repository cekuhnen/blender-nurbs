"""Selection readout: what the selection measures, always on (Shapr3D style).

Whatever is selected is measured, with no tool running, in a small panel at
the bottom right of the 3D view (and in the Analyze panel, with Copy):

* picked edges (Edge mode): their length, the total for several;
* picked faces (Face mode): their area, the total for several;
* a NURBS body with nothing picked: a closed solid its volume, area and
  centroid (a cross marks it in the view); an open sheet its area;
  several bodies: the totals (the centroid of them all when all are solids);
* curve objects (NURBS, Bezier, poly): their length, the total of every
  spline.

Exact, from the kernel: an edge's arc length, a face's area and a solid's mass
properties (relative accuracy 1e-6), a curve's length by Gauss-Legendre over
its knot spans. A body that takes longer than 15 ms to measure (hundreds of
faces) is measured once it has stopped changing for 0.25 s, so a live CV drag
stays smooth; meanwhile its last values show dimmed. In world space: a transform that is a rotation and a uniform
scale (and a mirror) is applied to the local values (lengths x s, areas x s²,
volumes x s³), so moving or turning a body costs nothing; a non-uniform scale
measures a moved copy, a shear is refused. Readings are cached per object
(its geometry revision, transform and picks) and only recomputed when one of
those changes. Nothing in the scene changes.
"""

import math
import time

import numpy as np

import bpy

from . import store

ACC = 1e-6                  # relative accuracy of the kernel's area / volume / length
FAST = 0.015                # s: a body measured faster than this is measured live
SETTLE = 0.25               # s: a slower one once it has stayed unchanged this long
_GL = np.polynomial.legendre.leggauss(12)
_cache = {}                 # object name -> (key, reading)
_errors = set()             # (object name, message) already printed
_cost = {}                  # object name -> seconds its last reading took
_seen = {}                  # object name -> (key, time first seen) of a change not yet measured
_pairs = {}                 # what two picked things measure against each other: key -> reading


# --------------------------------------------------------------------------
# Lengths of kernel curves
# --------------------------------------------------------------------------

def curve_length(kc):
    """Arc length of a kernel curve: Gauss-Legendre (12 points) on four
    pieces of every knot span (rational spans too, to ~1e-12)."""
    sm, ext = store._k()
    t0, t1 = (float(x) for x in kc.parameter_range())
    if not t1 > t0:
        return 0.0
    try:
        knots = sorted({float(k) for k in ext.curve_cvs(kc)["knots"] if t0 < float(k) < t1})
        pieces = 4
    except Exception:
        knots, pieces = [], 64
    brk = [t0] + knots + [t1]
    x, w = _GL
    total = 0.0
    for a, b in zip(brk[:-1], brk[1:]):
        sub = np.linspace(a, b, pieces + 1)
        for p, q in zip(sub[:-1], sub[1:]):
            h = 0.5 * (q - p)
            sp = [float(np.linalg.norm(sm.evaluate_curve(kc, float(h * xi + 0.5 * (p + q)))["tangent"])) for xi in x]
            total += h * float(np.dot(w, sp))
    return total


# --------------------------------------------------------------------------
# Readings (pure, given the objects)
# --------------------------------------------------------------------------

def _similarity(M):
    """(s, ok): the uniform scale of the 4x4 M, and whether M is a rotation
    (or mirror) times that scale plus a move."""
    L = np.asarray(M, float)[:3, :3]
    d = abs(float(np.linalg.det(L)))
    if d < 1e-300:
        return 0.0, False
    s = d ** (1.0 / 3.0)
    return s, bool(np.abs(L.T @ L / (s * s) - np.eye(3)).max() <= 1e-6)


def _scale(obj, M):
    """(s, similarity) of obj's world matrix M; s exact (no float32 rounding of
    a rotation) when the object has no parent and a uniform scale."""
    s, sim = _similarity(M)
    if sim and obj.parent is None:
        sc = [abs(float(x)) for x in obj.scale]
        if max(sc) - min(sc) <= 1e-7 * max(sc) and abs(sc[0] - s) <= 1e-5 * s:
            s = sc[0]
    return s, sim


def _world_brep(obj, M):
    b = store.get_brep(obj).copy()
    store.transform(b, M)               # raises on a shear
    return b


# --------------------------------------------------------------------------
# Radii: circle / arc edges and curves, constant-radius faces
# --------------------------------------------------------------------------

R_TOL = 1e-3                # relative spread of a face's radius still read as constant (fitted fillets)
FLAT_R = 1e3                # a radius over this times the face's size / the edge's length: flat / straight
BOW_TOL = 5e-4              # a fitted arc: off its circle by at most this much of how far it bows
                            # (a kernel fillet's rim 1.5e-4; free-form edges 1.3e-3 and up)
SHEET_FACES = 8             # an open body of at most this many faces: its faces' radii too


def fit_circle(W):
    """(centre, radius, normal, spread) of the circle best through the points
    W (n, 3): a plane by SVD, then the circle in it by least squares;
    spread = the largest distance off it (in and out of the plane) / radius.
    None when the points lie on a line."""
    C0 = W.mean(axis=0)
    X = W - C0
    _, sv, Vt = np.linalg.svd(X, full_matrices=False)
    if sv[1] <= 1e-9 * max(sv[0], 1e-300):
        return None
    a, b, n = Vt[0], Vt[1], Vt[2]
    x, y = X @ a, X @ b
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    (cx, cy, k), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    r = math.sqrt(max(k + cx * cx + cy * cy, 0.0))
    if r <= 0.0:
        return None
    C = C0 + cx * a + cy * b
    off = np.hypot(np.linalg.norm(W - C - np.outer((W - C) @ n, n), axis=1) - r, (W - C) @ n)
    return C, r, n, float(off.max() / r)


def _arc_of(kc, length):
    """(radius, angle in degrees, full circle, spread) of a kernel curve that
    is a circle or an arc of one: exact (Measure's test, every sample on one
    circle to 1e-7: spread 0), or fitted: within R_TOL of the radius and
    BOW_TOL of how far it bows (a fillet's rim is a fitted curve: spread is
    how far it strays from the circle, over the radius). Else None."""
    from . import linesolve
    from .measure import curve_radius
    c = linesolve.kernel_curve(kc, store._k()[0])
    if c.straight:
        return None
    m = curve_radius(c, c.W[0])
    if m is not None and m["source"] == "circle":
        r = float(m["radius"])
        if r > FLAT_R * max(length, 1e-300):
            return None                 # all but straight (round-off on a line)
        full = bool(m.get("full")) or c.closed
        return r, (360.0 if full else math.degrees(length / r)), full, 0.0
    fit = fit_circle(c.W)
    if fit is None or fit[3] > R_TOL:
        return None
    r = fit[1]
    if r > FLAT_R * max(length, 1e-300):
        return None
    A, B = c.W[0], c.W[-1]
    ab = B - A
    L2 = float(ab @ ab)
    if c.closed or L2 <= (1e-9 * r) ** 2:
        bow = r
    else:                               # the sagitta: how far it bows off its chord
        t = np.clip((c.W - A) @ ab / L2, 0.0, 1.0)
        bow = float(np.linalg.norm(A + np.outer(t, ab) - c.W, axis=1).max())
    if fit[3] * r > BOW_TOL * bow:      # a short free-form edge fits some big circle within R_TOL
        return None
    return r, (360.0 if c.closed else math.degrees(length / r)), bool(c.closed), fit[3]


def edge_arc(edge):
    """_arc_of for a Brep edge (in the Brep's own units)."""
    from .square import _edge_curve
    return _arc_of(_edge_curve(edge), float(edge.length(ACC)))


def face_radius(face, n=17):
    """(radius, kind, spread) of a face with one constant principal radius,
    else None. kind: "sphere" (the same both ways), "cylinder" (straight the
    other way), "fillet" (constant one way, as a rolling-ball fillet or a
    torus); spread: the radius's relative spread over the samples (0 for an
    exact one). Sampled on an n x n grid of its surface inside the trim,
    curvatures from the exact first and second derivatives."""
    from . import surfeval
    E = surfeval.Surf(face.surface())
    (u0, u1), (v0, v1) = E.u_domain, E.v_domain
    g = np.linspace(0.02, 0.98, n)
    U, V = np.meshgrid(u0 + g * (u1 - u0), v0 + g * (v1 - v0), indexing="ij")
    U, V = U.ravel(), V.ravel()
    keep = np.array([face.classify_uv(float(a), float(b)) != "outside" for a, b in zip(U, V)])
    if keep.sum() < 6:
        return None
    U, V = U[keep], V[keep]
    D = E.at(U, V, 2)
    su, sv = D[1, 0], D[0, 1]
    nn = np.cross(su, sv)
    ln = np.linalg.norm(nn, axis=1)
    ok = ln > 1e-9 * max(float(ln.max()), 1e-300)            # not at a pole
    if ok.sum() < 6:
        return None
    su, sv, nn, ln = su[ok], sv[ok], nn[ok], ln[ok]
    nrm = nn / ln[:, None]
    Ee, Fe, Ge = (su * su).sum(1), (su * sv).sum(1), (sv * sv).sum(1)
    Le, Me, Ne = ((D[2, 0][ok] * nrm).sum(1), (D[1, 1][ok] * nrm).sum(1), (D[0, 2][ok] * nrm).sum(1))
    den = Ee * Ge - Fe * Fe
    K = (Le * Ne - Me * Me) / den
    H = (Ee * Ne - 2.0 * Fe * Me + Ge * Le) / (2.0 * den)
    root = np.sqrt(np.maximum(H * H - K, 0.0))
    k1, k2 = np.abs(H + root), np.abs(H - root)
    ka, kb = np.maximum(k1, k2), np.minimum(k1, k2)           # the tighter way, the other
    if float(ka.min()) <= 1e-12 * float(ka.max()) or float(ka.max()) <= 0.0:
        return None                                           # flat somewhere: no constant radius
    ra = 1.0 / ka
    size = float(np.ptp(D[0, 0][ok], axis=0).max())
    if float(ra.max()) > FLAT_R * max(size, 1e-300):
        return None                                           # flat (curvature round-off)
    spread = float((ra.max() - ra.min()) / ra.mean())
    if spread > R_TOL:
        return None
    r = float(np.median(ra))
    if float(np.abs(kb * r - 1.0).max()) <= R_TOL:
        kind = "sphere"
    elif float((kb * r).max()) <= 1e-6:
        kind = "cylinder"
    else:
        kind = "fillet"
    return r, kind, (0.0 if spread <= 1e-7 else spread)


def _picked(obj):
    from . import subsel
    a = subsel.get_all(obj)
    return list(a["FACE"]), list(a["EDGE"])


def body_reading(obj, faces=(), edges=()):
    """What a NURBS object measures: {"faces": [areas], "edges": [lengths]}
    for its picks, else {"solid": bool, "area", "volume", "centroid"} for the
    whole body. World space."""
    from . import geom
    M = np.array(geom.world_matrix(obj), float)
    s, sim = _scale(obj, M)
    if sim:
        b, f_len, f_area, f_vol = store.get_brep(obj), s, s * s, s ** 3
    else:
        b, f_len, f_area, f_vol = _world_brep(obj, M), 1.0, 1.0, 1.0
    out = {"object": obj.name}
    if faces or edges:
        F, E = b.faces(), b.edges()
        faces = [i for i in faces if 0 <= i < len(F)]
        edges = [i for i in edges if 0 <= i < len(E)]
        out["faces"] = [abs(float(F[i].area(ACC))) * f_area for i in faces]
        out["edges"] = [float(E[i].length(ACC)) * f_len for i in edges]
        out["face_radii"] = [_scaled_r(_safe(face_radius, F[i]), f_len) for i in faces]
        out["edge_arcs"] = [_scaled_r(_safe(edge_arc, E[i]), f_len) for i in edges]
        return out
    solid = bool(b.is_manifold_solid())
    out["solid"] = solid
    if solid:
        mp = b.mass_properties(ACC)
        c = np.array(mp["centroid"], float)
        out["volume"] = abs(float(mp["volume"])) * f_vol
        out["area"] = float(mp["area"]) * f_area
        out["centroid"] = (M[:3, :3] @ c + M[:3, 3]) if sim else c
    else:
        out["area"] = float(b.area(ACC)) * f_area
        F = b.faces()
        if len(F) <= SHEET_FACES:                 # a fillet surface on its own: its radius
            out["face_radii"] = [_scaled_r(_safe(face_radius, f), f_len) for f in F]
    return out


def _safe(fn, x):
    try:
        return fn(x)
    except Exception as exc:
        print("NURBS selection readout: radius:", exc)
        return None


def _scaled_r(t, f):
    """A radius reading (r, ...) with r times the uniform scale f."""
    return None if t is None else (t[0] * f,) + tuple(t[1:])


def curve_reading(obj):
    """{"curves": [length of each spline]} of a curve object, world space."""
    from mathutils import Matrix
    from . import geom
    M = np.array(geom.world_matrix(obj), float)
    s, sim = _scale(obj, M)
    mat = Matrix.Identity(4) if sim else geom.world_matrix(obj)
    lengths, arcs = [], []
    f = s if sim else 1.0
    for sp in obj.data.splines:
        try:
            kc = geom.spline_to_kernel_curve(sp, mat)
            L = curve_length(kc)
        except Exception:
            continue                    # a spline of one point
        lengths.append(L * f)
        arcs.append(_scaled_r(_safe(lambda k: _arc_of(k, L), kc), f))
    return {"object": obj.name, "curves": lengths, "curve_arcs": arcs}


def _key(obj):
    """What a reading depends on: the geometry, the transform, the picks."""
    from . import geom, subsel
    M = tuple(round(x, 12) for row in geom.world_matrix(obj) for x in row)
    if store.is_nurbs(obj):
        return ("N", obj.get(store.PROP_REV, 0), M, obj.get(subsel.PROP), obj.select_get())
    pts = []
    for sp in obj.data.splines:
        if sp.type == 'BEZIER':
            for p in sp.bezier_points:
                pts += [*p.co, *p.handle_left, *p.handle_right]
        else:
            for p in sp.points:
                pts += [*p.co]
        pts.append(float(sp.use_cyclic_u) + 10.0 * sp.order_u + 1000.0 * len(sp.points))
    return ("C", hash(tuple(round(x, 12) for x in pts)), M)


def _settled():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()
    return None


def _reading(obj, settle=False):
    """obj's reading (cached). settle: a body whose last reading took longer
    than FAST is measured only once it has stayed unchanged for SETTLE
    seconds (a live CV drag rebuilds it every frame); until then the old
    reading comes back marked "stale" (or {"pending"} when there is none)."""
    k = _key(obj)
    hit = _cache.get(obj.name)
    if hit is not None and hit[0] == k:
        return hit[1]
    if settle and _cost.get(obj.name, 0.0) > FAST:
        now = time.monotonic()
        seen = _seen.get(obj.name)
        if seen is None or seen[0] != k:
            _seen[obj.name] = (k, now)
            if not bpy.app.timers.is_registered(_settled):
                bpy.app.timers.register(_settled, first_interval=SETTLE + 0.02)
            old = dict(hit[1]) if hit is not None else {"object": obj.name, "pending": True}
            old["stale"] = True
            return old
        if now - seen[1] < SETTLE:
            if not bpy.app.timers.is_registered(_settled):
                bpy.app.timers.register(_settled, first_interval=SETTLE - (now - seen[1]) + 0.02)
            old = dict(hit[1]) if hit is not None else {"object": obj.name, "pending": True}
            old["stale"] = True
            return old
    t0 = time.perf_counter()
    try:
        if store.is_nurbs(obj):
            faces, edges = _picked(obj)
            r = body_reading(obj, faces, edges)
        else:
            r = curve_reading(obj)
    except Exception as exc:
        msg = str(exc)
        if (obj.name, msg) not in _errors:
            _errors.add((obj.name, msg))
            print("NURBS selection readout:", obj.name, msg)
        r = {"object": obj.name, "error": "can't measure a sheared object" if "shear" in msg else msg}
    _cost[obj.name] = time.perf_counter() - t0
    _cache[obj.name] = (k, r)
    _seen.pop(obj.name, None)
    return r


def measurable(obj):
    return obj is not None and (store.is_nurbs(obj) or (obj.type == 'CURVE' and len(obj.data.splines) > 0))


def gather(context, settle=False):
    """The selection's totals:
    {"edges": [..], "faces": [..], "curves": [..], "solids": [(volume, area, centroid)],
     "sheets": [area], "errors": [(name, msg)], "stale": bool}, or None when
    nothing measurable is selected. settle: slow bodies wait until they stop
    changing (see _reading); "stale" says some value is an old one."""
    objs = [o for o in context.selected_objects if measurable(o)]
    if not objs:
        return None
    info = {"edges": [], "faces": [], "curves": [], "solids": [], "sheets": [], "errors": [], "stale": False,
            "edge_arcs": [], "face_radii": [], "curve_arcs": [], "sheet_radii": []}
    for o in sorted(objs, key=lambda o: o.name):
        r = _reading(o, settle)
        info["stale"] |= bool(r.get("stale"))
        if r.get("pending"):
            continue
        if "error" in r:
            info["errors"].append((o.name, r["error"]))
        elif "curves" in r:
            info["curves"] += r["curves"]
            info["curve_arcs"] += r.get("curve_arcs", [None] * len(r["curves"]))
        elif "faces" in r:
            info["faces"] += r["faces"]
            info["edges"] += r["edges"]
            info["face_radii"] += r.get("face_radii", [None] * len(r["faces"]))
            info["edge_arcs"] += r.get("edge_arcs", [None] * len(r["edges"]))
        elif r.get("solid"):
            info["solids"].append((r["volume"], r["area"], np.asarray(r["centroid"], float)))
        else:
            info["sheets"].append(r["area"])
            info["sheet_radii"] += r.get("face_radii", [None])
    if not info["stale"]:
        try:
            _pair_into(info, objs)
        except Exception as exc:
            msg = str(exc)
            if ("pair", msg) not in _errors:
                _errors.add(("pair", msg))
                print("NURBS selection readout: two things:", msg)
    return info


def _pair_els(objs):
    """([(kind, object, index)] of the picked edges / faces and curve splines,
    [whole bodies]) of the selected objects; None when more than two of either."""
    els, bodies = [], []
    for o in sorted(objs, key=lambda o: o.name):
        if store.is_nurbs(o):
            f, e = _picked(o)
            if f or e:
                els += [("F", o, i) for i in f] + [("E", o, i) for i in e]
            else:
                bodies.append(o)
        else:
            els += [("S", o, f"S{k}") for k in range(len(o.data.splines))]
        if len(els) > 2 or len(bodies) > 2:
            return None
    return els, bodies


def pair_refs(context):
    """The two picked things (kind, object, index) the readout measures against
    each other, or None."""
    got = _pair_els([o for o in context.selected_objects if measurable(o)])
    if got is None or len(got[0]) != 2 or got[1]:
        return None
    return got[0]


def _pair_into(info, objs):
    """Two things selected (picked edges / faces, curve splines) or two whole
    bodies: info["pair"] = what they measure against each other
    (measurepair.relate), info["clearance"] = (gap, how) of two bodies."""
    from . import measurepair
    info["pair"], info["clearance"] = None, None
    got = _pair_els(objs)
    if got is None:
        return
    els, bodies = got
    if len(els) == 2 and not bodies:
        k = ("pair",) + tuple((t, o.name, i, _key(o)) for t, o, i in els)
        if k not in _pairs:
            mk = {"F": measurepair.face_el, "E": measurepair.edge_el, "S": measurepair.spline_el}
            a, b = (mk[t](o, i) for t, o, i in els)
            _pairs.clear()
            _pairs[k] = measurepair.relate(a, b)
        info["pair"] = _pairs[k]
    elif len(bodies) == 2 and not els:
        k = ("bodies",) + tuple((o.name, _key(o)) for o in bodies)
        if k not in _pairs:
            A, B = (measurepair.world_brep(o) for o in bodies)
            rel = A.relationship_to(B)
            _pairs.clear()
            _pairs[k] = (float(rel.get("distance", A.distance_to(B))), str(rel.get("relationship", "")))
        info["clearance"] = _pairs[k]


def centroid(info):
    """The volume-weighted centroid of the selected solids (None unless only
    solids are selected: no sheets, curves or picks)."""
    if not info or not info["solids"] or info["sheets"] or info["faces"] or info["edges"] or info["curves"]:
        return None
    V = sum(v for v, _, _ in info["solids"])
    if V <= 0.0:
        return None
    return sum(v * c for v, _, c in info["solids"]) / V


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

# the scene's Length unit: (metres per unit, symbol)
UNITS = {"KILOMETERS": (1000.0, "km"), "METERS": (1.0, "m"), "CENTIMETERS": (0.01, "cm"),
         "MILLIMETERS": (0.001, "mm"), "MICROMETERS": (1e-6, "µm"), "MILES": (1609.344, "mi"),
         "FEET": (0.3048, "ft"), "INCHES": (0.0254, "in"), "THOU": (2.54e-5, "thou")}
SIG = 6                     # significant digits shown


def _num(v):
    """v with SIG significant digits, no exponent, trailing zeros dropped (-0 is 0)."""
    if v == 0.0 or not math.isfinite(v):
        return "0" if v == 0.0 else f"{v}"
    dec = max(0, min(9, SIG - 1 - int(math.floor(math.log10(abs(v))))))
    t = f"{v:.{dec}f}"
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return "0" if t in ("-0", "") else t


def fmt(context, x, dim=1):
    """A length (dim 1), area (2) or volume (3) in the scene's units: in its
    Length unit when one is set (mm, in, ...), else Blender's adaptive units."""
    us = context.scene.unit_settings
    if us.system == 'NONE' or not math.isfinite(x):
        return _num(x)
    sx = x * (us.scale_length or 1.0) ** dim               # in metres (^dim)
    u = UNITS.get(us.length_unit)
    if u is not None:
        sym = u[1] + {1: "", 2: "²", 3: "³"}[dim]
        return f"{_num(sx / u[0] ** dim)} {sym}"
    cat = {1: 'LENGTH', 2: 'AREA', 3: 'VOLUME'}[dim]
    try:
        return bpy.utils.units.to_string(us.system, cat, 0.0 if abs(sx) < 1e-15 else sx, precision=5)
    except Exception:
        return _num(x)


def _sig(x, spread):
    """x rounded to the digits a fitted value's spread supports (2 to 6)."""
    if spread <= 0.0 or x == 0.0:
        return x
    n = max(2, min(6, int(math.floor(-math.log10(spread)))))
    return round(x, n - 1 - int(math.floor(math.log10(abs(x)))))


def _same(rs):
    return max(rs) - min(rs) <= 1e-6 * max(rs)


def arcs_text(context, arcs):
    """'R 10 mm  ·  90° arc' / 'R 10 mm  ·  Ø 20 mm' for one circle or arc
    (≈ when fitted), 'R 10 mm each' / 'R 5 – 10 mm' for several; None unless
    every one is a circle or an arc of one."""
    if not arcs or any(a is None for a in arcs):
        return None
    sp = max(a[3] for a in arcs)
    rs = [_sig(a[0], sp) for a in arcs]
    approx = "≈ " if sp > 0.0 else ""
    if len(arcs) == 1:
        r, (_, ang, full, _) = rs[0], arcs[0]
        return f"R {approx}{fmt(context, r)}  ·  " + (f"Ø {approx}{fmt(context, 2 * r)}" if full
                                                     else f"{_num(round(ang, 1 if approx else 4))}° arc")
    if _same(rs) or (approx and max(rs) - min(rs) <= R_TOL * max(rs)):
        return f"R {approx}{fmt(context, _sig(float(np.median(rs)), sp))} each"
    return f"R {fmt(context, min(rs))} – {fmt(context, max(rs))}"


def radii_text(context, radii):
    """'R 10 mm fillet' (≈ when fitted: its radius spreads over the face),
    'R 10 mm each' / 'R 5 – 10 mm' for several; None unless every face has
    one constant radius."""
    if not radii or any(r is None for r in radii):
        return None
    sp = max(r[2] for r in radii)
    rs = [_sig(r[0], sp) for r in radii]
    approx = "≈ " if sp > 0.0 else ""
    if len(radii) == 1:
        return f"R {approx}{fmt(context, rs[0])} {radii[0][1]}"
    if _same(rs) or (approx and max(rs) - min(rs) <= R_TOL * max(rs)):
        kinds = {r[1] for r in radii}
        return f"R {approx}{fmt(context, _sig(float(np.median(rs)), sp))} each" + (f" {kinds.pop()}" if len(kinds) == 1 else "")
    return f"R {fmt(context, min(rs))} – {fmt(context, max(rs))}"


def _with(value, extra):
    return value if extra is None else f"{value}  ·  {extra}"


def _count(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def rows(context, info):
    """[(label, value)] for the readout; [] for nothing."""
    if not info:
        return []
    out = []
    if info["edges"]:
        n = len(info["edges"])
        out.append((_count(n, "edge"), _with(("Total length " if n > 1 else "Length ") + fmt(context, sum(info["edges"])),
                                             arcs_text(context, info["edge_arcs"]))))
    if info["curves"]:
        n = len(info["curves"])
        out.append((_count(n, "curve"), _with(("Total length " if n > 1 else "Length ") + fmt(context, sum(info["curves"])),
                                              arcs_text(context, info["curve_arcs"]))))
    if info["faces"]:
        n = len(info["faces"])
        out.append((_count(n, "face"), _with(("Total area " if n > 1 else "Area ") + fmt(context, sum(info["faces"]), 2),
                                             radii_text(context, info["face_radii"]))))
    if info["solids"]:
        n = len(info["solids"])
        V = sum(v for v, _, _ in info["solids"])
        A = sum(a for _, a, _ in info["solids"])
        lab = "Solid" if n == 1 else _count(n, "solid")
        out.append((lab, f"Volume {fmt(context, V, 3)}  ·  Area {fmt(context, A, 2)}"))
    if info["sheets"]:
        n = len(info["sheets"])
        lab = "Open surface" if n == 1 else _count(n, "open surface")
        out.append((lab, _with(f"Area {fmt(context, sum(info['sheets']), 2)}", radii_text(context, info["sheet_radii"]))
                    + "  ·  no volume"))
    c = centroid(info)
    if c is not None:
        tiny = 1e-9 * sum(v for v, _, _ in info["solids"]) ** (1.0 / 3.0)
        out.append(("Centroid", "  ".join(f"{a} {fmt(context, 0.0 if abs(x) < tiny else float(x))}"
                                          for a, x in zip("XYZ", c))))
    if info.get("pair") is not None:
        from . import measurepair
        out += measurepair.rows(context, info["pair"])
    if info.get("clearance") is not None:
        g, how = info["clearance"]
        out.append(("Between", f"Clearance {fmt(context, g)}" if g > 0 else (how.replace("_", " ") or "touching")))
    for name, msg in info["errors"]:
        out.append((name, msg))
    if not out and info.get("stale"):
        out.append(("", "Measuring…"))
    return out


def text(context, info=None):
    """The readout as plain text, one reading a line (for Copy and the Info log)."""
    info = gather(context) if info is None else info
    return "\n".join(f"{a}: {b}" for a, b in rows(context, info))


# --------------------------------------------------------------------------
# Drawing: bottom right of the 3D view
# --------------------------------------------------------------------------

def enabled(context=None):
    from . import hints
    p = hints._prefs()
    return bool(getattr(p, "selection_readout", True)) if p is not None else True


def current(context):
    """The selection's info for the view and the panel (cached per object;
    slow bodies settle first)."""
    try:
        return gather(context, settle=True)
    except Exception as exc:
        print("NURBS selection readout:", exc)
        return None


def draw_px(ctx, region, rv3d, box):
    """The readout panel (bottom right of the free part of the view, above the
    key hints when they would meet) and, when switched on, the centroid's cross."""
    import blf
    from . import hints
    if ctx.mode != 'OBJECT' or not enabled(ctx):
        return
    sp = ctx.space_data
    if sp is None or not getattr(sp.overlay, "show_overlays", True):
        return
    info = current(ctx)
    lines = rows(ctx, info)
    if not lines:
        return
    s = getattr(ctx.preferences.system, "ui_scale", 1.0) or 1.0
    font = hints.FONT
    blf.size(font, hints.SIZE * s)
    gap, pad, lh = 12 * s, 7 * s, 16 * s
    wl = max(blf.dimensions(font, a)[0] for a, _ in lines)
    wr = max(blf.dimensions(font, b)[0] for _, b in lines)
    w, h = wl + gap + wr + 2 * pad, lh * len(lines) + 2 * pad - (lh - blf.dimensions(font, "Ag")[1])
    left, right, _ = hints._insets(ctx.area, region)
    m = 10 * s
    x, y = region.width - right - m - w, m
    hud = hints.HUD.get("box")
    if hud is not None and x < hud[0] + hud[2] + m:
        y = hud[1] + hud[3] + m                      # the key hints are in the way: above them
    if x < left + m:
        x = left + m
    box(x, y, w, h, hints.COL_BG)
    asc = blf.dimensions(font, "A")[1]
    top = y + h - pad
    for j, (a, b) in enumerate(lines):
        yy = top - j * lh - asc
        blf.color(font, *hints.COL_LABEL)
        blf.position(font, x + pad, yy, 0)
        blf.draw(font, a)
        blf.color(font, *(hints.COL_OFF if info.get("stale") else hints.COL_TEXT))   # dim: being measured
        blf.position(font, x + pad + wl + gap, yy, 0)
        blf.draw(font, b)
    if info.get("pair") is not None:
        _pair_lines(region, rv3d, info["pair"], s)
    c = centroid(info) if show_centroid(ctx) else None
    if c is not None:
        if centroid_style(ctx) == 'COG':
            _cog(region, rv3d, c, 9 * s)
        else:
            _cross(region, rv3d, c, 7 * s)


def _pair_lines(region, rv3d, r, s):
    """The shortest distance (and the centres' line) between two picked things."""
    import gpu
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from gpu_extras.batch import batch_for_shader
    segs = [r["min"][:2]]
    if r.get("centres") is not None:
        segs.append(r["centres"][:2])
    pts = []
    for A, B in segs:
        if float(np.linalg.norm(np.asarray(B) - np.asarray(A))) <= 0.0:
            continue
        a = location_3d_to_region_2d(region, rv3d, tuple(float(x) for x in A))
        b = location_3d_to_region_2d(region, rv3d, tuple(float(x) for x in B))
        if a is not None and b is not None:
            pts += [(a.x, a.y), (b.x, b.y)]
    if not pts:
        return
    sh = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    for col, lw in (((0.0, 0.0, 0.0, 0.6), 3.5 * s), ((1.0, 0.85, 0.2, 1.0), 1.5 * s)):
        b = batch_for_shader(sh, 'LINES', {"pos": pts})
        sh.bind()
        sh.uniform_float("color", col)
        sh.uniform_float("lineWidth", lw)
        sh.uniform_float("viewportSize", (region.width, region.height))
        b.draw(sh)
    gpu.state.blend_set('NONE')


def show_centroid(context):
    """The centroid's cross is off unless switched on (Analyze > Selection,
    the eye on the Centroid row): the object's origin is the point to keep
    clear in the view."""
    from . import hints
    p = hints._prefs()
    return bool(getattr(p, "selection_centroid", False)) if p is not None else False


def centroid_style(context):
    """'CROSS' (a small yellow plus) or 'COG' (the centre-of-mass symbol)."""
    from . import hints
    p = hints._prefs()
    return getattr(p, "centroid_style", 'CROSS') if p is not None else 'CROSS'


def _cog(region, rv3d, P, r):
    """The centre-of-mass symbol: a ring split in black and white quarters,
    open in the middle so an origin dot on the same spot still shows through."""
    import math
    import gpu
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from gpu_extras.batch import batch_for_shader
    q = location_3d_to_region_2d(region, rv3d, tuple(float(x) for x in P))
    if q is None:
        return
    ri, n = 0.45 * r, 8                              # inner radius, segments per quarter

    def ring(k0, k1, rad):
        return [(q.x + rad * math.cos(math.pi / 2 * k / n), q.y + rad * math.sin(math.pi / 2 * k / n))
                for k in range(k0, k1 + 1)]

    gpu.state.blend_set('ALPHA')
    flat = gpu.shader.from_builtin('UNIFORM_COLOR')
    for quarter, col in enumerate(((1.0, 1.0, 1.0, 0.95), (0.08, 0.08, 0.08, 0.95)) * 2):
        o, i = ring(quarter * n, quarter * n + n, r), ring(quarter * n, quarter * n + n, ri)
        tris = []
        for k in range(n):
            tris += [o[k], o[k + 1], i[k], i[k], o[k + 1], i[k + 1]]
        b = batch_for_shader(flat, 'TRIS', {"pos": tris})
        flat.bind()
        flat.uniform_float("color", col)
        b.draw(flat)
    line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    for rad in (r, ri):
        b = batch_for_shader(line, 'LINE_LOOP', {"pos": ring(0, 4 * n - 1, rad)})
        line.bind()
        line.uniform_float("color", (0.0, 0.0, 0.0, 0.85))
        line.uniform_float("lineWidth", 1.5)
        line.uniform_float("viewportSize", (region.width, region.height))
        b.draw(line)
    gpu.state.blend_set('NONE')


def _cross(region, rv3d, P, r):
    """A small yellow plus, open in the middle so an origin dot on the same
    spot (a box's origin is often at its centroid) still shows through."""
    import gpu
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from gpu_extras.batch import batch_for_shader
    q = location_3d_to_region_2d(region, rv3d, tuple(float(x) for x in P))
    if q is None:
        return
    g = 0.5 * r                                      # the open middle
    segs = [(q.x - r, q.y), (q.x - g, q.y), (q.x + g, q.y), (q.x + r, q.y),
            (q.x, q.y - r), (q.x, q.y - g), (q.x, q.y + g), (q.x, q.y + r)]
    sh = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    for col, lw in (((0.0, 0.0, 0.0, 0.7), 4.0), ((1.0, 0.85, 0.2, 1.0), 2.0)):
        b = batch_for_shader(sh, 'LINES', {"pos": segs})
        sh.bind()
        sh.uniform_float("color", col)
        sh.uniform_float("lineWidth", lw)
        sh.uniform_float("viewportSize", (region.width, region.height))
        b.draw(sh)
    gpu.state.blend_set('NONE')


# --------------------------------------------------------------------------
# Panel and Copy
# --------------------------------------------------------------------------

def draw_panel(col, context):
    """The readout in the Analyze panel, with Copy and the on-screen switch."""
    from . import hints
    info = current(context)
    lines = rows(context, info)
    box = col.box()
    head = box.row(align=True)
    head.label(text="Selection", icon='DRIVER_DISTANCE')
    p = hints._prefs()
    if p is not None:
        head.prop(p, "selection_readout", text="", icon='HIDE_OFF' if p.selection_readout else 'HIDE_ON',
                  emboss=False)
    sub = head.row(align=True)
    sub.enabled = bool(lines)
    if info and info.get("pair") is not None:
        sub.operator("nproto.measure_pin_selection", text="", icon='PINNED', emboss=False)
    if info and (info["edges"] or info["curves"]):
        sub.operator("nproto.measure_pin_comb", text="", icon='IPO_ELASTIC', emboss=False)
    sub.operator("nproto.measure_copy", text="", icon='COPYDOWN', emboss=False)
    if not lines:
        box.label(text="Select curves, bodies, or pick faces / edges")
        return
    for a, b in lines:
        r = box.row()
        r.label(text=a)
        r.label(text=b)
        if a.startswith("Centroid") and p is not None:      # its cross in the view: off at first
            if p.selection_centroid:
                r.prop(p, "centroid_style", text="", icon_only=True)
            r.prop(p, "selection_centroid", text="", emboss=False,
                   icon='HIDE_OFF' if p.selection_centroid else 'HIDE_ON')


class NPROTO_OT_measure_copy(bpy.types.Operator):
    """Copy what the selection measures (lengths, areas, volume, centroid) to
    the clipboard, one reading a line, and write it to the Info log"""
    bl_idname = "nproto.measure_copy"
    bl_label = "Copy Measurements"
    bl_options = {'REGISTER'}

    def execute(self, context):
        txt = text(context)
        if not txt:
            self.report({'WARNING'}, "Nothing measurable selected")
            return {'CANCELLED'}
        context.window_manager.clipboard = txt
        self.report({'INFO'}, txt.replace("\n", "   |   "))
        return {'FINISHED'}


classes = (NPROTO_OT_measure_copy,)
