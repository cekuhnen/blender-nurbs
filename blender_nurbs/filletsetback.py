"""Surface Fillet, Ends: Set Back -- surfaces meeting edge to edge.

Both surfaces are cut back along their WHOLE edges that meet (each by the
fillet's setback from the crossing: the distance from the edge to that
side's rail), and a Loft edge blend (G1 / G2 / G3, a few Bezier spans: a
light patch) joins the two new edges. Where one surface is shorter than the
other the blend runs from the one edge's whole length to the other's and its
end is an S curve from the longer surface's corner to the shorter one's --
neither surface is grown or notched.

It is the modelling recipe done by hand -- Offset on Surface of each edge,
Trim along it, Loft between the new edges -- so the result is a Loft (its
Tangent / Curvature / Degree editable in its panel) and each surface gets a
Trim step; the cut-back curves are kept as the trims' inputs in the NURBS
History collection (deleted when history is off).
"""

import json

import bpy
import numpy as np

from . import curvejoin, features, store, surfoffset
from .kernel import filletends, load
from .sweeps import _world_brep


def crossing_edges(a, b):
    """(edge index on a, edge index on b): the boundary edges the two
    surfaces meet along (the crossing lies on both), None when they do not
    meet edge to edge."""
    sm = load()
    wa, wb = _world_brep(a), _world_brep(b)
    size = filletends.size_of(wa, wb)
    cr = filletends.crossing(sm, wa, wb, size)
    if cr is None or cr.closed:
        return None
    tol = 2e-4 * size
    pts = [cr.at(s)[0] for s in (0.25, 0.5, 0.75)]
    out = []
    for w in (wa, wb):
        hit = None
        for i, e in enumerate(w.edges()):
            if all(e.closest_point(tuple(map(float, p)))[2] <= tol for p in pts):
                hit = i
                break
        if hit is None:
            return None
        out.append(hit)
    return tuple(out)


def edge_samples(obj, ei, n=17):
    """(points (n,3), unit tangents, unit directions INTO the surface square
    to the edge) along edge ei of obj, world, by arc length."""
    sm = load()
    w = _world_brep(obj)
    e = w.edges()[ei]
    t0, t1 = e.parameter_range()
    ts = np.linspace(t0, t1, 257)
    P = np.array([e.point_at(float(t)) for t in ts], float)
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    tk = np.interp(np.linspace(0.0, L[-1], n), L, ts)
    P = np.array([e.point_at(float(t)) for t in tk], float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-300
    f = w.faces()[0] if w.face_count() == 1 else None
    size = float(L[-1]) or 1.0
    D = []
    for p, t in zip(P, T):
        fc = f or min(w.faces(), key=lambda g: g.closest_point(tuple(p))[1])
        _, uvs, _ = sm.surface_closest_point(fc.surface(), tuple(map(float, p)))
        uv = uvs[0]
        u, v = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
        nn = np.array(fc.normal_at_uv(u, v), float)
        d = np.cross(nn, t)
        d /= np.linalg.norm(d) or 1.0
        if not _inside(fc, p + 1e-3 * size * d):
            d = -d
        D.append(d)
    return P, T, np.array(D)


def _inside(face, p):
    """Is p (just off an edge) on the face? Off its surface's own boundary the
    closest point is clamped to it (a distance); off a trim, classify says."""
    sm = load()
    q, d = face.closest_point(tuple(map(float, p)))
    if d > 1e-9 + 1e-6 * float(np.linalg.norm(p)):
        return False
    try:
        _, uvs, _ = sm.surface_closest_point(face.surface(), tuple(map(float, p)))
        uv = uvs[0]
        u, v = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
        return face.classify_uv(float(u), float(v)) != "outside"
    except Exception:
        return True


def preview_samples(a, b):
    """For the live preview: ((P, T, D) along a's edge, the same along b's,
    b's turned to run the way a's does), None when not edge to edge."""
    ee = crossing_edges(a, b)
    if ee is None:
        return None
    sa_, sb_ = edge_samples(a, ee[0]), edge_samples(b, ee[1])
    Pa, Pb = sa_[0], sb_[0]
    if np.linalg.norm(Pa[0] - Pb[-1]) + np.linalg.norm(Pa[-1] - Pb[0]) < \
            np.linalg.norm(Pa[0] - Pb[0]) + np.linalg.norm(Pa[-1] - Pb[-1]):
        sb_ = tuple(x[::-1] for x in sb_)
    return sa_, sb_


def preview_sections(samples, da, db, deg, k=9):
    """Flat sections (n, k, 3) of the blend between the two cut-back edges:
    a Bezier of degree deg from a's new edge to b's, square across each
    (deg 1: straight), CVs h = chord / deg apart (the Loft's Tangent 1)."""
    from math import comb
    (Pa, _, Da), (Pb, _, Db) = samples
    A = Pa + da * Da
    B = Pb + db * Db
    w = np.linspace(0.0, 1.0, k)[None, :, None]
    if deg <= 1:
        return A[:, None] * (1 - w) + B[:, None] * w
    h = np.linalg.norm(B - A, axis=1, keepdims=True) / deg
    m = (deg + 1) // 2
    cv = [A - i * h * Da for i in range(m)] + [B - i * h * Db for i in range(m)][::-1]
    return sum(comb(deg, i) * w ** i * (1 - w) ** (deg - i) * c[:, None] for i, c in enumerate(cv))


def _offset_specs(obj, ei, dist):
    """Spline specs (world) of edge ei moved back `dist` onto obj (Offset on
    Surface, the side that stays on the surface)."""
    tg = surfoffset.target_of([obj])
    pieces = [curvejoin.input_spans(obj, f"E{ei}")]
    last = None
    for flip in (False, True):
        try:
            return surfoffset.offset_on_surface(tg, pieces, {"distance": float(dist), "flip": flip,
                                                             "os_corner": "SHARP", "tolerance": 1e-5})
        except Exception as exc:
            last = exc
    raise RuntimeError(f"{obj.name}: its edge cannot be set back {dist:.4g} m ({last})")


def _spec_mid(spec):
    from .primitives import spec_to_kernel_curve
    c = spec_to_kernel_curve(spec)
    t0, t1 = c.parameter_range()
    return np.array(c.evaluate(0.5 * (t0 + t1)), float)


def _edge_near(obj, q):
    w = _world_brep(obj)
    return min(range(len(w.edges())), key=lambda i: w.edges()[i].closest_point(tuple(map(float, q)))[2])


def build(context, a, b, da, db, cont="G2", degree=5):
    """Cut a and b back along the edges they meet along (by da, db) and join
    the new edges with a Loft edge blend. Returns the Loft object."""
    from .flow import new_curve_object
    ee = crossing_edges(a, b)
    if ee is None:
        raise RuntimeError("Set Back needs surfaces that meet edge to edge (their edges along the crossing)")
    sm = load()
    plans = []
    for obj, ei, d in ((a, ee[0], da), (b, ee[1], db)):
        specs = _offset_specs(obj, ei, d)           # both before either is trimmed
        w = _world_brep(obj)
        e = w.edges()[ei]
        t0, t1 = e.parameter_range()
        em = np.array(e.point_at(0.5 * (t0 + t1)), float)
        plans.append((obj, specs, em))
    keep_history = features.history_enabled(context)
    mids, cutters = [], []
    for obj, specs, em in plans:
        cut = new_curve_object(context, obj, obj.name + "_setback", specs)
        cutters.append(cut)
        cm = _spec_mid(specs[0])
        keep = cm + 2.0 * (cm - em)                 # well inside the part that stays
        f = min(_world_brep(obj).faces(), key=lambda g: g.closest_point(tuple(map(float, keep)))[1])
        _, uvs, _ = sm.surface_closest_point(f.surface(), tuple(map(float, keep)))
        uv = uvs[0]
        u, v = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
        n = np.array(f.normal_at_uv(u, v), float)
        r = bpy.ops.nproto.trim(target=obj.name, cutters=json.dumps([[cut.name, "S0"]]), ref_point=tuple(keep),
                                normal=tuple(n), view_dir=tuple(-n), action="KEEP")
        if r != {'FINISHED'}:
            raise RuntimeError(f"{obj.name}: the cut back along its edge failed")
        mids.append(cm)
    if keep_history:
        features.keep_inputs(context, cutters)
    else:
        for c in cutters:
            data = c.data
            bpy.data.objects.remove(c)
            if data.users == 0:
                bpy.data.curves.remove(data)
    ea, eb = _edge_near(a, mids[0]), _edge_near(b, mids[1])
    kw = dict(cont_a=cont, cont_b=cont)
    if cont == "G0":
        kw["ruled"] = True
    else:
        kw.update(fit="SPANS", span_degree=int(degree))
    r = bpy.ops.nproto.loft(inputs=json.dumps([[a.name, f"E{ea}"], [b.name, f"E{eb}"]]), **kw)
    if r != {'FINISHED'}:
        raise RuntimeError("the blend between the cut-back edges failed")
    lo = context.active_object
    lo.name = "Fillet"
    return lo
