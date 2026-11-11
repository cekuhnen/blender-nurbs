"""Feature builders: pure functions (inputs, params) -> kernel geometry.

Every modelling command is registered here so construction history can rebuild
it later. An input is (object, ref) -- see geom.py for the ref strings.
"""

import json

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
    """Kernel curve in world space for a curve spline or a surface edge."""
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


def exact_side(obj, ref):
    """Exact CV rows of a loft input on its own layout: {"deg", "knots" (0..1),
    "P", "D", "DD"} in world space; D / DD are the outward cross first / second
    derivative rows of an untrimmed surface edge (None for a curve). None when
    not available exactly (trimmed B-rep edge, rational)."""
    sm, ext = _k()
    M = np.array(geom.world_matrix(obj), dtype=float)
    R, T = M[:3, :3], M[:3, 3]
    if is_edge_ref(ref):
        if not store.is_natural(obj):
            return None
        d = ext.surface_cvs(store.get_surfaces(obj)[0])
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
        D = DD = None
        if p >= 1 and len(G) > 1:
            d1 = p / (Ua[p + 1] - Ua[1])
            Su = d1 * (G[1] - G[0])                  # dS/du at the edge (u increasing inward)
            D = -Su                                  # outward
            DD = np.zeros_like(P0)
            if p >= 2 and len(G) > 2:
                a1 = (G[1] - G[0]) / (Ua[p + 1] - Ua[1])
                a2 = (G[2] - G[1]) / (Ua[p + 2] - Ua[2])
                DD = p * (p - 1) / (Ua[p + 1] - Ua[2]) * (a2 - a1)
        lo, hi = Ue[0], Ue[-1]
        knots = [(u - lo) / (hi - lo) for u in Ue]
        tw = lambda A: None if A is None else A @ R.T
        return {"deg": int(deg_e), "knots": knots, "P": P0 @ R.T + T, "D": tw(D), "DD": tw(DD)}
    if is_brep_edge_ref(ref):
        return None
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


def input_samples(obj, ref, n):
    if is_edge_ref(ref):
        surf = store.get_surfaces(obj)[0]
        return geom.sample_surface_edge(surf, ref, geom.world_matrix(obj), n)
    if is_brep_edge_ref(ref):
        # trimmed / B-rep edge: exact points on the face, cross tangent and
        # cross curvature from the face (same sampling as Square)
        from .square import sample_side
        sd = sample_side(obj, ref, n)
        return geom.EdgeSample(sd.P, sd.B, sd.D2)
    return geom.EdgeSample(geom.sample_curve(input_curve(obj, ref), n))


def default_ref(obj):
    return "S0" if obj.type == 'CURVE' else "U0"


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

LOFT_FOLD_DEG = 1.0     # a loft surface folding more than this along a knot line is split there


def split_folds(g):
    """A surface with folds (C0 knot lines where its normal jumps, e.g. a loft
    through rectangles) as faces joined along them, like Rhino's polysurface
    from kinked curves: the folds become real edges, which Edge Fillet can
    round and every tool sees as edges. Anything else is returned as it is."""
    sm, _ = _k()
    if not isinstance(g, sm.Surface):
        return g
    pieces = store._crease_pieces(g, LOFT_FOLD_DEG)
    if not pieces:
        return g
    from . import jobs
    ident = {"loc": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "angle": 0.0, "scale": [1.0, 1.0, 1.0]}
    objs = {f"s{i}": (store.brep_to_text(store.to_brep(pc)), ident) for i, pc in enumerate(pieces)}
    b = store.to_brep(g).bounding_box()
    diag = float(np.linalg.norm(np.array(b[1], float) - np.array(b[0], float))) or 1.0
    _, breps = jobs.run("join", objs, ["joined"], order=list(objs), into="s0", tolerance=1e-6 * diag)
    return store.brep_from_text(breps["joined"])


@builder("loft")
def build_loft(inputs, p):
    g = _build_loft(inputs, p)
    if _is_skin(inputs, p):
        # through curve sections: sections with corners give folds -> faces.
        # A G1 / G2 blend between two edges is never split (a fold there would
        # be a defect, not hidden as a polysurface)
        g = split_folds(g)
    return g


def _is_skin(inputs, p):
    any_edge = any(is_surface_edge(r) for _, r in inputs)
    smooth = p.get("cont_a", "G0") != "G0" or p.get("cont_b", "G0") != "G0"
    return bool(p.get("closed")) or len(inputs) > 2 or not (any_edge or smooth)


def _build_loft(inputs, p):
    sm, _ = _k()
    if len(inputs) < 2:
        raise ValueError("Loft needs at least two inputs")
    any_edge = any(is_surface_edge(r) for _, r in inputs)
    smooth = p.get("cont_a", "G0") != "G0" or p.get("cont_b", "G0") != "G0"

    closed = bool(p.get("closed"))
    if closed and len(inputs) < 3:
        raise ValueError("a closed loft needs at least 3 sections")
    if closed or len(inputs) > 2 or not (any_edge or smooth):
        # Plain skin through curves: exact, done by the kernel (closed: our
        # periodic skin, smooth across the seam).
        curves = [input_curve(o, r) for o, r in inputs]
        if closed:
            curves = [curves[i] for i in geom.ring_order(curves)]
        if p.get("auto_align", True):
            prev = geom.sample_curve(curves[0], 16)
            for i in range(1, len(curves)):
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
        return sm.create_skin_surface(curves)

    # Two inputs with G1/G2 to surfaces: sampled blend surface.
    n = int(p.get("samples", 24))
    a = input_samples(*inputs[0], n)
    b = input_samples(*inputs[1], n)
    if p.get("auto_align", True) and geom.should_flip(a.P, b.P):
        b = b.reversed()
    if p.get("flip_b"):
        b = b.reversed()
    # exact: both inputs on the same layout along the edge -> keep it (a line stays a line)
    try:
        sa, sb = exact_side(*inputs[0]), exact_side(*inputs[1])
    except Exception:
        sa = sb = None
    if sa is not None and sb is not None:
        if np.linalg.norm(sb["P"][0] - b.P[0]) > np.linalg.norm(sb["P"][-1] - b.P[0]):
            sb = geom.reverse_side(sb)           # follow the samples' orientation (auto align / flip)
        if np.linalg.norm(sa["P"][0] - a.P[0]) > np.linalg.norm(sa["P"][-1] - a.P[0]):
            sa = geom.reverse_side(sa)
        if _same_layout(sa, sb):
            return geom.blend_loft_exact(sa, sb, a, b, p.get("cont_a", "G0"), p.get("cont_b", "G0"),
                                         p.get("tan_a", 1.0), p.get("tan_b", 1.0),
                                         p.get("cur_a", 0.0), p.get("cur_b", 0.0))
    # along the edge: the inputs' own degree when both are known (a line stays a
    # line); a side without an exact layout (a trimmed edge) is curved: cubic at least
    deg_in = max((s["deg"] for s in (sa, sb) if s is not None), default=3)
    if sa is None or sb is None:
        deg_in = max(deg_in, 3)
    return geom.blend_loft(a, b,
                           p.get("cont_a", "G0"), p.get("cont_b", "G0"),
                           p.get("tan_a", 1.0), p.get("tan_b", 1.0),
                           p.get("cur_a", 0.0), p.get("cur_b", 0.0), degree_v=deg_in)


@builder("extrude")
def build_extrude(inputs, p):
    sm, _ = _k()
    obj, ref = inputs[0]
    curve = input_curve(obj, ref)
    mode = p.get("direction", "NORMAL")
    if mode == "NORMAL":
        d, _ = geom.curve_plane_normal(curve)
        if d.length < 0.5:
            d = Vector((0, 0, 1))
    elif mode == "VECTOR":
        d = Vector(p.get("dir_vec", (0.0, 0.0, 1.0)))
        if d.length < 1e-12:
            d = Vector((0, 0, 1))
    else:
        d = geom.axis_vector(mode, obj)
    d = geom.tilt_direction(d.normalized(), float(p.get("tilt", 0.0)), float(p.get("tilt_azimuth", 0.0)))
    h = float(p.get("distance", 1.0))
    draft = float(p.get("draft", 0.0))
    if abs(draft) > 1e-9:
        # Multi-Draft: height h along d, the far edge moved outward by h tan(draft)
        return geom.draft_extrude(curve, d, h, draft, flip=bool(p.get("draft_flip")),
                                  symmetric=bool(p.get("symmetric")))
    vec = d * h
    if p.get("symmetric"):
        curve = geom.translate_curve(curve, tuple(-vec))
        vec = vec * 2.0
    return sm.create_extrude_surface(curve, tuple(vec))


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


@builder("revolve")
def build_revolve(inputs, p):
    obj, ref = inputs[0]
    curve = input_curve(obj, ref)
    axis = geom.axis_vector(p.get("axis", "Z"), obj)
    origin = geom.world_matrix(obj).translation if p.get("pivot", "OBJECT") == "OBJECT" else Vector()
    angle = float(p.get("angle", 360.0))
    seam = float(p.get("seam_angle", 0.0)) if abs(abs(angle) - 360.0) < 1e-9 else 0.0
    return revolve_exact(curve, tuple(origin), tuple(axis), angle, seam)


@builder("offset")
def build_offset(inputs, p):
    sm, _ = _k()
    obj, _ = inputs[0]
    surf = store.fresh_surfaces(obj)[0]  # fresh: the offset keeps a pointer to its base
    # Result lives in the source object's local space (same matrix).
    return sm.create_offset_surface(surf, p.get("distance", 0.1))


@builder("planar")
def build_planar(inputs, p):
    sm, _ = _k()
    curves = [input_curve(o, r) for o, r in inputs]
    return sm.create_planar_faces(curves)


@builder("blend_curve")
def build_blend_curve(inputs, p):
    """Returns ('curve', degree, world CVs): written into a Blender curve object."""
    (oa, ra), (ob, rb) = inputs[0], inputs[1]
    fa = geom.curve_end_frame(input_curve(oa, ra), int(p.get("end_a", 1)))
    fb = geom.curve_end_frame(input_curve(ob, rb), int(p.get("end_b", 0)))
    if p.get("flip_a"):
        fa = (fa[0], -fa[1], fa[2])
    if p.get("flip_b"):
        fb = (fb[0], -fb[1], fb[2])
    deg, pts = geom.blend_curve_cvs(fa, fb, p.get("cont_a", "G1"), p.get("cont_b", "G1"),
                                    p.get("w_a", 1.0), p.get("w_b", 1.0), p.get("c_a", 0.0), p.get("c_b", 0.0))
    return ("curve", deg, pts)


# --------------------------------------------------------------------------
# History records
# --------------------------------------------------------------------------

PROP_HIST = "nproto_hist"


def record(obj, op, inputs, params):
    for key in [k for k in obj.keys() if k.startswith("nproto_in_")]:
        del obj[key]
    for i, (src, _) in enumerate(inputs):
        obj[f"nproto_in_{i}"] = src
    obj[PROP_HIST] = json.dumps({"op": op, "refs": [r for _, r in inputs], "params": params})
    mark_built(obj)          # the caller just built it from exactly these inputs
    try:  # show it in the Feature panel right away
        from .feature_ui import sync_from_history
        sync_from_history(obj)
    except Exception:
        pass


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


NATURAL_STEPS = {"refit_trim", "rebuild_surface", "extend_surface"}   # result is an untrimmed surface
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
    if PROP_HIST not in obj and PROP_BASE not in obj:
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


def clear_history(obj):
    for key in [k for k in obj.keys() if k.startswith(("nproto_in_", "nproto_sin_")) or
                k in (PROP_HIST, PROP_STACK, PROP_BASE, PROP_BASE + "_natural")]:
        del obj[key]


def history_inputs(obj):
    return [obj[k] for k in obj.keys() if k.startswith(("nproto_in_", "nproto_sin_")) and obj[k] is not None]


def run(op, inputs, params):
    if op not in BUILDERS:
        raise RuntimeError(f"no builder for feature '{op}' (reload the add-on)")
    return BUILDERS[op](inputs, params)


PROP_KEY = "nproto_build_key"


def build_key(obj, _seen=None):
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
    hsh.update(np.round(np.array(geom.world_matrix(obj), float), 10).tobytes())
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
    natural = False
    if h is not None:
        g = run(h["op"], h["inputs"], h["params"])
        if h["op"] == "square":
            from .square import store_info
            store_info(obj)
        elif h["op"] == "patch":
            from .patch import store_info
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
    for e in entries:
        if e["params"].get("_off"):   # muted in the history list
            continue
        text = steps.APPLY[e["op"]](obj, text, e["inputs"], e["params"])
    store.set_geometry_text(obj, text, context, keep_untrimmed=False)
    active = [e for e in entries if not e["params"].get("_off")]
    if active and active[-1]["op"] in NATURAL_STEPS:
        obj[store.PROP_NATURAL] = 1          # the last step made a plain surface again
        if store.PROP_UNTRIM in obj:
            del obj[store.PROP_UNTRIM]
    elif entries:
        obj[store.PROP_UNTRIM] = base_text
        obj[store.PROP_UNTRIM + "_natural"] = 1 if natural else 0
    else:
        obj[store.PROP_NATURAL] = 1 if natural else 0
    return True
