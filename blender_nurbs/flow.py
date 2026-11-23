"""Flow Along Surface: bend geometry from a base surface onto a target surface.

A point over the base surface is known by where it is on the base -- its
foot's (u, v) and its height h above the base along the normal. Flow puts it
at the same place on the target: the target at the matching (u, v), plus h
along the target's normal. Curves are sampled, mapped and fitted again
(corners stay corners), NURBS surfaces are refitted over their own (u, v)
domain, trims and all (surfmap: each face's edges are mapped too and the new
surface is trimmed with them), so a flowed solid is still one body.

The (u, v) match:
* General: the base's (u, v) as a share of its domain is the same share of
  the target's domain (untrimmed surfaces: like Rhino's FlowAlongSrf, the
  base is usually a flat rectangle the geometry was drawn on). Flip U / Flip
  V / Swap UV turn the target's directions round when they run the other way.
* Fold back: a base made by Unroll Surface from the target. Every flat face
  maps to the 3D face it was unrolled from at the very same (u, v) (the flat
  sheet is that face's isometric map over its own domain), so holes, slots or
  lettering drawn on the sheet land exactly where the sheet will be bent to.

Rigid moves each object as a whole instead (no bending): turned and moved by
the map's frame at the object's centre (e.g. rivets or bosses on a sheet).

Points off the base's domain carry on along the target's tangents from its
edge (the step past the base's edge, in the base's tangent plane, is taken
over by the target's (u, v) directions).
"""

import json

import bpy
import numpy as np
from bpy.props import BoolProperty, StringProperty

from . import curvejoin, curvetools, curvetrim, features, geom, store, surfeval, surfmap

OP = "flow"
CURVE_TOL = 1e-5          # flowed curves within this share of the target's size of the map
SURF_TOL = 1e-6           # flowed surfaces: the same for their nets
SURF_BUDGET = 40000       # grid nodes per face sampled for its refit
REFIT_TOL = 1e-5          # a refit missing by more than this share of the size is sampled finer
LAST_INFO = {}


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def world_brep(obj):
    b = store.get_brep(obj).copy()
    return store.transform(b, np.array(geom.world_matrix(obj), float))


def brep_size(b):
    from .unroll import _box
    lo, hi = _box(b)
    return float(np.linalg.norm(hi - lo)) or 1.0


# --------------------------------------------------------------------------
# The map
# --------------------------------------------------------------------------

class Pair:
    """One base face and where it goes: the target surface St and the (u, v)
    match uv -> (uv on St, its 2x2 Jacobian). sigma: the target normal's
    side that the base normal's side goes to (+1 / -1), 0 to drop the height
    (onto the target itself)."""

    def __init__(self, face, Sb, St, uvmap, sigma=1.0):
        self.face, self.Sb, self.St, self.uvmap, self.sigma = face, Sb, St, uvmap, sigma

    def feet(self, Q):
        uv = self.Sb.feet(Q)
        B = self.Sb.at(uv[:, 0], uv[:, 1], 0)[0, 0]
        return uv, np.linalg.norm(Q - B, axis=1)

    def apply(self, Q, uv):
        D = self.Sb.at(uv[:, 0], uv[:, 1], 1)
        B, Bu, Bv = D[0, 0], D[1, 0], D[0, 1]
        N = _unit(np.cross(Bu, Bv))
        r = Q - B
        h = np.einsum("ij,ij->i", r, N)
        t = r - h[:, None] * N
        # the step past the base's edge in (u, v) (least squares in its tangent plane)
        a, b, c = (np.einsum("ij,ij->i", Bu, Bu), np.einsum("ij,ij->i", Bu, Bv), np.einsum("ij,ij->i", Bv, Bv))
        g1, g2 = np.einsum("ij,ij->i", Bu, t), np.einsum("ij,ij->i", Bv, t)
        det = a * c - b * b
        ok = det > 1e-14 * np.maximum(a * c, 1e-300)
        safe = np.where(ok, det, 1.0)
        du = np.where(ok, (c * g1 - b * g2) / safe, 0.0)
        dv = np.where(ok, (a * g2 - b * g1) / safe, 0.0)
        uvt, J = self.uvmap(uv)
        d = np.stack([du, dv], 1) @ J.T
        E = self.St.at(uvt[:, 0], uvt[:, 1], 1)
        T, Tu, Tv = E[0, 0], E[1, 0], E[0, 1]
        Nt = _unit(np.cross(Tu, Tv))
        return T + Tu * d[:, :1] + Tv * d[:, 1:] + (self.sigma * h)[:, None] * Nt


def general_uvmap(Sb, St, flip_u=False, flip_v=False, swap=False):
    """The base's (u, v) as a share of its domain -> the same share of the
    target's (turned by the flips / the swap)."""
    lo_b = np.array([Sb.u_domain[0], Sb.v_domain[0]])
    sb = np.array([Sb.u_domain[1], Sb.v_domain[1]]) - lo_b
    lo_t = np.array([St.u_domain[0], St.v_domain[0]])
    st = np.array([St.u_domain[1], St.v_domain[1]]) - lo_t
    J = np.diag(1.0 / sb)
    if swap:
        J = J[::-1]
    sgn = np.array([-1.0 if flip_u else 1.0, -1.0 if flip_v else 1.0])
    J = np.diag(st * sgn) @ J
    off = lo_t + st * np.array([1.0 if flip_u else 0.0, 1.0 if flip_v else 0.0])

    def m(uv):
        return (uv - lo_b) @ J.T + off, J
    return m


def same_uvmap(flip, Sb):
    """Fold back: the flat face's (u, v) is its source face's, as Unify
    Normals left it (flip 1: u reversed, 2: v reversed, 3: u and v swapped)."""
    (u0, u1), (v0, v1) = Sb.u_domain, Sb.v_domain
    if flip == 1:
        J, off = np.diag([-1.0, 1.0]), np.array([u0 + u1, 0.0])
    elif flip == 2:
        J, off = np.diag([1.0, -1.0]), np.array([0.0, v0 + v1])
    elif flip == 3:
        J, off = np.array([[0.0, 1.0], [1.0, 0.0]]), np.zeros(2)
    else:
        J, off = np.eye(2), np.zeros(2)

    def m(uv):
        return uv @ J.T + off, J
    return m


class FlowMap:
    """World points over the base -> world points on / over the target."""

    def __init__(self, pairs, size):
        self.pairs, self.size = pairs, size

    def map(self, Q):
        return self.map_labeled(Q)[0]

    def map_labeled(self, Q):
        """(points, the pair each point went through)."""
        Q = np.asarray(Q, float).reshape(-1, 3)
        lab = np.zeros(len(Q), int)
        if not len(Q):
            return Q.copy(), lab
        if len(self.pairs) == 1:
            p = self.pairs[0]
            return p.apply(Q, p.feet(Q)[0]), lab
        tol = 1e-9 * self.size
        res = [p.feet(Q) for p in self.pairs]
        D = np.stack([d for _, d in res], 1)                  # (m, pairs)
        lab = np.argmin(D, axis=1)
        near = D <= D.min(axis=1, keepdims=True) + tol
        for i in np.nonzero(near.sum(axis=1) > 1)[0]:
            # on several faces' surfaces (a flat sheet's faces share one plane): the face it is
            # inside, else the one it is on the edge of
            got = {}
            for k in np.nonzero(near[i])[0]:
                f = self.pairs[k].face
                u, v = res[k][0][i]
                c = "inside" if f is None else f.classify_uv(float(u), float(v))
                got.setdefault(c, k)
                if c == "inside":
                    break
            for c in ("inside", "boundary", "vertex"):
                if c in got:
                    lab[i] = got[c]
                    break
        out = np.empty_like(Q)
        for k, p in enumerate(self.pairs):
            sel = np.nonzero(lab == k)[0]
            if len(sel):
                out[sel] = p.apply(Q[sel], res[k][0][sel])
        return out, lab

    def map_by(self, Q, k):
        """Q through pair k (whichever face it is nearest)."""
        Q = np.asarray(Q, float).reshape(-1, 3)
        p = self.pairs[k]
        return p.apply(Q, p.feet(Q)[0])


def unroll_link(base, target):
    """The face table of a base made by Unroll Surface from target: [[source
    face, flip], ...] per base face, or None."""
    from .unroll import PROP_UNROLL
    try:
        info = json.loads(base.get(PROP_UNROLL, ""))
    except Exception:
        return None
    if info.get("source") != target.name:
        return None
    faces = info.get("faces") or []
    nb, nt = store.get_brep(base).face_count(), store.get_brep(target).face_count()
    if len(faces) != nb or any(k < 0 or k >= nt for k, _ in faces):
        return None
    return faces


def largest_face(b):
    return max(b.faces(), key=lambda f: f.area())


def build_map(base, target, flip_u=False, flip_v=False, swap=False, fold=True):
    """(FlowMap, mode): "fold" for an unrolled sheet onto its source, else "general"."""
    bb, tb = world_brep(base), world_brep(target)
    size = max(brep_size(bb), brep_size(tb))
    link = unroll_link(base, target) if fold else None
    if link is not None:
        tf = tb.faces()
        pairs = []
        for f, (k, flip) in zip(bb.faces(), link):
            Sb = surfeval.Surf(f.surface())
            pairs.append(Pair(f, Sb, surfeval.Surf(tf[k].surface()), same_uvmap(flip, Sb),
                              -1.0 if flip else 1.0))
        return FlowMap(pairs, size), "fold", (bb, tb)
    fb, ft = largest_face(bb), largest_face(tb)
    Sb, St = surfeval.Surf(fb.surface()), surfeval.Surf(ft.surface())
    return FlowMap([Pair(None, Sb, St, general_uvmap(Sb, St, flip_u, flip_v, swap))], size), "general", (bb, tb)


# --------------------------------------------------------------------------
# Mapping geometry
# --------------------------------------------------------------------------

def _pick_degree(p_src, T, X, U, tol):
    """1 if a degree-1 direction's samples stay linear in it (1D fit on its
    own knots within tol), else at least 3."""
    if p_src != 1:
        return max(3, int(p_src))
    B = surfmap.basis(U, 1, T)
    R = B @ surfmap._lstsq(B, X.reshape(len(T), -1)) - X.reshape(len(T), -1)
    return 1 if float(np.abs(R).max(initial=0.0)) <= 0.5 * tol else 3


def map_surface(S, fmap, tol):
    """(C, U, p, V, q, err) of the surface S taken through fmap: a net over
    S's own (u, v) domain fitted to the mapped grid."""
    from .unroll import axes_for
    au, av = axes_for(S, SURF_BUDGET)
    for _ in range(4):
        P = S.grid(au.t, av.t, 0)[0, 0]
        X = fmap.map(P.reshape(-1, 3)).reshape(P.shape)
        p = _pick_degree(S.p, au.t, X, surfmap.fit_knots(S.U, 1, 1, *S.u_domain), tol)
        q = _pick_degree(S.q, av.t, X.transpose(1, 0, 2), surfmap.fit_knots(S.V, 1, 1, *S.v_domain), tol)
        U0 = surfmap.fit_knots(S.U, S.p, p, *S.u_domain)
        V0 = surfmap.fit_knots(S.V, S.q, q, *S.v_domain)
        C, U, V, err = surfmap.fit_grid(au.t, av.t, X, U0, p, V0, q, tol,
                                        min_u=0.5 * float(au.h.min()), min_v=0.5 * float(av.h.min()))
        # spans stop at half a sampling panel: a face sampled as plainly as its own shape (a flat
        # cap: 4 panels a side) that the map bends more (folded back onto a dome) gets finer
        # panels until it is within REFIT_TOL -- else its trim, held to 1e-4 of the size, fails
        if err <= max(tol, REFIT_TOL * fmap.size) or 4 * au.n * av.n > SURF_BUDGET:
            break
        au = surfmap.Axis(surfmap.breakpoints(S.U, *S.u_domain, 2 * au.P))
        av = surfmap.Axis(surfmap.breakpoints(S.V, *S.v_domain, 2 * av.P))
    return C, U, p, V, q, err


def tight_plane(f, S, pad=0.02):
    """S cut down to face f's trim, for a flat face on a bilinear plane patch:
    the same plane (its (u, v) runs affinely) over the trim's (u, v) box grown
    by pad of it. A word's letters are capped by one plane spanning the whole
    word; mapped whole, its far parts land where the map bends (the next face
    of an unrolled sheet, past the edge of a face a projection hits) and the
    fit of a letter's cap fails there. S itself when it is no such plane or
    no smaller than 80 % of it would do."""
    if S.p != 1 or S.q != 1 or S.nu != 2 or S.nv != 2 or S.rational:
        return S
    try:
        uv = np.vstack([q for lp in f.boundary_loops() if lp for q, _ in surfmap.loop_uvs(S, lp)])
    except Exception:
        return S
    (u0, u1), (v0, v1) = S.u_domain, S.v_domain
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    g = pad * (hi - lo) + 1e-9 * np.array([u1 - u0, v1 - v0])
    lo = np.maximum(lo - g, [u0, v0])
    hi = np.minimum(hi + g, [u1, v1])
    if (hi[0] - lo[0]) * (hi[1] - lo[1]) > 0.8 * (u1 - u0) * (v1 - v0):
        return S
    _, ext = store._k()
    C = S.at([lo[0], lo[0], hi[0], hi[0]], [lo[1], hi[1], lo[1], hi[1]], 0)[0, 0]     # (u, v) order: u major
    try:
        return surfeval.Surf(ext.surface_from_cvs(1, 1, [tuple(map(float, c)) for c in C],
                                                  [float(lo[0])] * 2 + [float(hi[0])] * 2,
                                                  [float(lo[1])] * 2 + [float(hi[1])] * 2, None))
    except Exception:
        return S


def flow_brep(b, fmap):
    """The world-space brep b taken through fmap: (Brep, worst fit miss)."""
    sm, ext = store._k()
    size = fmap.size
    tol = SURF_TOL * size
    parts, worst = [], 0.0
    faces = list(b.faces())
    for f in faces:
        try:
            S = surfeval.Surf(f.surface())
        except Exception:
            raise ValueError("a face is not a B-spline surface")
        if len(faces) > 1:
            S = tight_plane(f, S)
        C, U, p, V, q, err = map_surface(S, fmap, tol)
        worst = max(worst, err)
        make = lambda C=C, U=U, p=p, V=V, q=q: surfmap.surface_from_net(C, U, p, V, q)
        if len(faces) == 1 and surfmap.is_natural_face(f, S):
            parts.append(ext.brep_from_surface(make()))
        else:
            def to_world(uv, S=S):
                return fmap.map(S.at(uv[:, 0], uv[:, 1], 0)[0, 0])
            parts.append(surfmap.trimmed_face(f, S, make, to_world, size))
    out = parts[0]
    for o in parts[1:]:
        ext.append_brep(out, o)
    if len(parts) > 1:
        out = sm.stitch_advanced(out, 1e-5 * size, squeeze_small_edges=False, making_manifold_solid=False)[0]
        try:
            h = sm.heal_brep(out)
            if h is not None and h.face_count():
                out = h
        except Exception:
            pass
    return out, worst


def _closed(spans, cyclic):
    try:
        return curvetrim.is_closed(spans, cyclic)
    except Exception:
        P0, P1 = curvejoin._cart(spans[0])[0], curvejoin._cart(spans[-1])[-1]
        return cyclic or np.linalg.norm(P1 - P0) <= 1e-9 * curvejoin._size(spans)


def _straight(X, size):
    d = X[-1] - X[0]
    L = float(np.linalg.norm(d))
    if L <= 1e-12 * size:
        return False
    off = X - X[0] - np.outer((X - X[0]) @ d / (L * L), d)
    return float(np.linalg.norm(off, axis=1).max()) <= 1e-9 * size


def _line_span(a, b):
    return np.array([[*a, 1.0], [*b, 1.0]], float)


def _fit_spans(X, closed, tol):
    """(open spans, closed CVs) of a cubic fitted to the samples X within tol
    (open: ends kept)."""
    keep = np.concatenate([[True], np.linalg.norm(np.diff(X, axis=0), axis=1) > 1e-12 * max(tol, 1e-300)])
    X = X[keep]
    C, _ = curvetools.fit_auto(X, 3, closed, None, tol)
    if closed:
        return None, C
    U = geom.clamped_uniform_knots(len(C), 3)
    return curvejoin._clamped_to_spans(3, C, None, U), None


SPAN_SAMPLES = 48


def _eval(seg, g):
    i = min(int(g), len(seg) - 1)
    return curvejoin.eval_span(seg[i], [g - i])[0]


def _dense_map(seg, fmap, tol, per=SPAN_SAMPLES, cap=4000):
    """(P, g, X, labels): samples of the spans and their images, dense enough
    that the images' chords sag less than tol / 10 (the fit is held to the
    chords too)."""
    for _ in range(4):
        P, g = curvetrim.sample_spans(seg, per)
        X, lab = fmap.map_labeled(P)
        if len(X) < 3:
            break
        sag = np.linalg.norm(X[1:-1] - 0.5 * (X[:-2] + X[2:]), axis=1)
        sag = sag[(lab[:-2] == lab[1:-1]) & (lab[1:-1] == lab[2:])]
        worst = float(sag.max(initial=0.0)) / 4.0           # a two-chord sag is 4x one chord's
        if worst <= 0.1 * tol or per >= cap:
            break
        per = min(cap, int(per * min(8.0, np.ceil(np.sqrt(worst / (0.1 * tol))))))
    return P, g, X, lab


def _crossing(seg, fmap, g0, g1, a, b, iters=40):
    """Where the curve (spans seg, local parameter g) passes from fmap's pair a
    to pair b between g0 and g1: (the point through a, through b)."""
    for _ in range(iters):
        g = 0.5 * (g0 + g1)
        if fmap.map_labeled(_eval(seg, g)[None])[1][0] == a:
            g0 = g
        else:
            g1 = g
    P = _eval(seg, 0.5 * (g0 + g1))[None]
    return fmap.map_by(P, a)[0], fmap.map_by(P, b)[0]


def _pieces(seg, fmap, tol):
    """The mapped samples of a smooth stretch, cut where it crosses from one
    face of the map to another: [(points, starts a new curve)] (the first
    piece never does). A crossing whose two sides land together is a corner
    (a fold), else a cut."""
    P, g, X, lab = _dense_map(seg, fmap, tol)
    out, first = [], False
    cur = [X[0]]
    for k in range(len(X) - 1):
        if lab[k] == lab[k + 1]:
            cur.append(X[k + 1])
            continue
        xa, xb = _crossing(seg, fmap, g[k], g[k + 1], lab[k], lab[k + 1])
        cur.append(xa)
        out.append((np.array(cur), first))
        first = float(np.linalg.norm(xa - xb)) > 100.0 * tol
        cur = [xb, X[k + 1]]
    out.append((np.array(cur), first))
    return out


def _piece_spans(X, size, tol):
    if len(X) < 2:
        return []
    if _straight(X, size) or len(X) < 5:
        return [_line_span(X[0], X[-1])]
    return _fit_spans(X, False, tol)[0]


def _chain_spec(chain):
    """One spline spec from consecutive spans, joined exactly."""
    out = [chain[0]]
    for sp in chain[1:]:
        sp = sp.copy()
        w = sp[0, 3]
        sp[0] = np.r_[out[-1][-1, :3] / out[-1][-1, 3] * w, w]
        out.append(sp)
    return curvejoin.spans_to_spec(out, simplify=True)


def flow_spans(spans, cyclic, fmap, size, tol):
    """Spline specs of the world spans taken through fmap: smooth stretches
    fitted as one cubic each, the curve's corners kept, a corner added where
    it crosses a fold of the map and the curve cut where the map is cut."""
    closed = _closed(spans, cyclic)
    cuts = curvejoin.split_points(spans)
    if closed:
        a, b = curvejoin._tangent(spans[-1], 1), curvejoin._tangent(spans[0], 0)
        seam = float(a @ b) < np.cos(np.radians(curvejoin.CORNER_DEG))
        if not cuts and not seam:
            P, _, X, lab = _dense_map(spans, fmap, tol)
            if np.all(lab == lab[0]):
                _, C = _fit_spans(X[:-1], True, tol)
                return [curvetools.rebuilt_spec(C, 3, True)]
        elif cuts and not seam:
            k = cuts[0]
            spans = spans[k:] + spans[:k]
            cuts = [c - k for c in cuts[1:]]
    bounds = [0] + list(cuts) + [len(spans)]
    chains = [[]]
    for a, b in zip(bounds[:-1], bounds[1:]):
        for k, (X, new) in enumerate(_pieces(spans[a:b], fmap, tol)):
            if new and chains[-1]:
                chains.append([])
            chains[-1] += _piece_spans(X, size, tol)
    chains = [c for c in chains if c]
    if closed and len(chains) > 1:
        e0, e1 = chains[-1][-1], chains[0][0]
        if np.linalg.norm(e0[-1, :3] / e0[-1, 3] - e1[0, :3] / e1[0, 3]) <= 100.0 * tol:
            chains = [chains[-1] + chains[0]] + chains[1:-1]
    return [_chain_spec(c) for c in chains]


def new_curve_object(context, src, name, specs):
    """A curve object of world-space spline specs, in src's collections."""
    from .primitives import write_spline_object
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    obj = bpy.data.objects.new(name, cu)
    write_spline_object(obj, specs)
    for c in (src.users_collection or [context.collection]):
        c.objects.link(obj)
    return obj


def curve_specs(obj, fmap, size, tol):
    """Spline specs (world space) of a curve object's splines through fmap."""
    M = np.array(geom.world_matrix(obj), float)
    out = []
    for sp in obj.data.splines:
        try:
            spans = curvejoin.transform_spans(curvejoin.spline_spans(sp), M)
        except Exception:
            continue
        out += flow_spans(spans, bool(sp.use_cyclic_u), fmap, size, tol)
    return out


def rigid_matrix(obj, fmap):
    """4x4 world move of obj as a whole: the map's turn and move at its centre."""
    if store.is_nurbs(obj):
        from .unroll import _box
        lo, hi = _box(world_brep(obj))
        c = 0.5 * (lo + hi)
    else:
        M = np.array(geom.world_matrix(obj), float)
        P = np.array([M[:3, :3] @ np.array(v) + M[:3, 3] for v in obj.bound_box], float)
        c = P.mean(axis=0)
    eps = 1e-4 * fmap.size
    Sb = fmap.pairs[0].Sb
    uv = Sb.feet(c[None])
    D = Sb.at(uv[:, 0], uv[:, 1], 1)
    e1 = _unit(D[1, 0][0])
    n = _unit(np.cross(D[1, 0][0], D[0, 1][0]))
    e2 = np.cross(n, e1)
    Q = fmap.map(np.array([c, c + eps * e1, c + eps * e2]))
    f1 = _unit(Q[1] - Q[0])
    f2 = Q[2] - Q[0]
    f2 = _unit(f2 - (f2 @ f1) * f1)
    f3 = np.cross(f1, f2)
    R = np.stack([f1, f2, f3], 1) @ np.stack([e1, e2, np.cross(e1, e2)], 1).T
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = Q[0] - R @ c
    return A


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------

def _params_map(inputs, p):
    base, target = inputs[1][0], inputs[2][0]
    fmap, mode, _ = build_map(base, target, bool(p.get("fw_flip_u")), bool(p.get("fw_flip_v")),
                              bool(p.get("fw_swap")), bool(p.get("fw_fold", True)))
    return fmap, mode


@features.builder(OP)
def build_flow(inputs, p):
    """inputs: (object to flow, ref), (base, "SURF"), (target, "SURF"); the
    result in world space (curves: splines, NURBS: brep text)."""
    obj = inputs[0][0]
    fmap, mode = _params_map(inputs, p)
    LAST_INFO.clear()
    LAST_INFO.update(mode=mode)
    if p.get("fw_rigid"):
        A = rigid_matrix(obj, fmap)
        if obj.type == 'CURVE':
            M = A @ np.array(geom.world_matrix(obj), float)
            return ("splines", [curvejoin.spans_to_spec(curvejoin.transform_spans(curvejoin.spline_spans(sp), M),
                                                        simplify=False) for sp in obj.data.splines])
        b = world_brep(obj)
        store.transform(b, A)
        return ("brep_text", store.brep_to_text(b))
    if obj.type == 'CURVE':
        return ("splines", curve_specs(obj, fmap, fmap.size, CURVE_TOL * fmap.size))
    b, err = flow_brep(world_brep(obj), fmap)
    LAST_INFO.update(fit=err / fmap.size)
    return ("brep_text", store.brep_to_text(b))


# --------------------------------------------------------------------------
# The operator
# --------------------------------------------------------------------------

def _points_of(obj, n=200):
    M = np.array(geom.world_matrix(obj), float)
    if obj.type == 'CURVE':
        P = []
        for sp in obj.data.splines:
            try:
                P.append(curvetrim.sample_spans(curvejoin.spline_spans(sp), 8)[0])
            except Exception:
                pass
        P = np.vstack(P) if P else np.zeros((0, 3))
        return P @ M[:3, :3].T + M[:3, 3]
    try:
        d = store.display(obj)
        P = np.array([p for pl in d.get("edges", []) for p in pl], float).reshape(-1, 3)
    except Exception:
        P = np.zeros((0, 3))
    P = P[:: max(1, len(P) // n)]
    return P @ M[:3, :3].T + M[:3, 3]


def guess_roles(context):
    """(objects, base, target) from the selection: the target is the active
    NURBS object; the base is a sheet unrolled from it, else the selected
    NURBS object the other selected geometry lies nearest to."""
    sel = [o for o in context.selected_objects if store.is_nurbs(o) or o.type == 'CURVE']
    act = context.view_layer.objects.active
    target = act if act in sel and store.is_nurbs(act) else None
    if target is None:
        return [], None, None
    cands = [o for o in sel if o is not target and store.is_nurbs(o)]
    base = next((o for o in cands if unroll_link(o, target) is not None), None)
    if base is None and cands:
        others = [o for o in sel if o is not target]
        best, bd = None, None
        for c in cands:
            try:
                bb = world_brep(c)
                Sb = surfeval.Surf(largest_face(bb).surface())
            except Exception:
                continue
            d = 0.0
            for o in others:
                if o is c:
                    continue
                P = _points_of(o)
                if len(P):
                    uv = Sb.feet(P)
                    d += float(np.linalg.norm(P - Sb.at(uv[:, 0], uv[:, 1], 0)[0, 0], axis=1).mean())
            if len(others) == 1:
                d = 0.0          # only the base and the target: nothing else to flow
            if bd is None or d < bd:
                best, bd = c, d
        base = best
    objs = [o for o in sel if o is not target and o is not base]
    return objs, base, target


class NPROTO_OT_flow(bpy.types.Operator):
    """Bend curves and surfaces from a base surface onto a target surface:
    same place in (u, v), same height over the surface. Select the geometry
    and the base, then the target last (active). A sheet made by Unroll
    Surface as the base folds back onto its source exactly"""
    bl_idname = "nproto.flow"
    bl_label = "Flow Along Surface"
    bl_options = {'REGISTER', 'UNDO'}

    base: StringProperty(name="Base", description="The surface the geometry lies on now")
    target: StringProperty(name="Target", description="The surface to bend the geometry onto")
    rigid: BoolProperty(name="Rigid", default=False,
                        description="Move and turn each object as a whole instead of bending it")
    flip_u: BoolProperty(name="Flip U", default=False, description="The target's U the other way round")
    flip_v: BoolProperty(name="Flip V", default=False, description="The target's V the other way round")
    swap: BoolProperty(name="Swap UV", default=False, description="The base's U goes along the target's V")
    fold: BoolProperty(name="Fold Back", default=True,
                       description="A sheet unrolled from the target maps face by face onto it (exact)")
    objects: StringProperty(options={'HIDDEN'})          # JSON names (the redo panel keeps them)

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def draw(self, context):
        lay = self.layout
        lay.prop_search(self, "base", context.scene, "objects")
        lay.prop_search(self, "target", context.scene, "objects")
        lay.prop(self, "rigid")
        link = self._link(context)
        if link:
            lay.prop(self, "fold")
        if not (link and self.fold):
            row = lay.row(align=True)
            row.prop(self, "flip_u", toggle=True)
            row.prop(self, "flip_v", toggle=True)
            row.prop(self, "swap", toggle=True)

    def _link(self, context):
        b, t = bpy.data.objects.get(self.base), bpy.data.objects.get(self.target)
        return b is not None and t is not None and store.is_nurbs(b) and store.is_nurbs(t) \
            and unroll_link(b, t) is not None

    def invoke(self, context, event):
        objs, base, target = guess_roles(context)
        if target is None or base is None:
            self.report({'WARNING'}, "Select the geometry and the base surface, then the target surface last")
            return {'CANCELLED'}
        if not objs:
            self.report({'WARNING'}, "Select the curves or surfaces to flow too")
            return {'CANCELLED'}
        self.base, self.target = base.name, target.name
        self.objects = json.dumps([o.name for o in objs])
        return self.execute(context)

    def execute(self, context):
        from .ops import _kernel_ok, _select_only
        if not _kernel_ok(self):
            return {'CANCELLED'}
        base, target = bpy.data.objects.get(self.base), bpy.data.objects.get(self.target)
        if base is None or target is None or not store.is_nurbs(base) or not store.is_nurbs(target):
            self.report({'WARNING'}, "Pick a NURBS base and target surface")
            return {'CANCELLED'}
        try:
            names = json.loads(self.objects or "[]")
        except Exception:
            names = []
        objs = [bpy.data.objects.get(n) for n in names]
        objs = [o for o in objs if o is not None and o is not base and o is not target]
        if not objs:
            objs = [o for o in context.selected_objects if o not in (base, target)
                    and (store.is_nurbs(o) or o.type == 'CURVE')]
        if not objs:
            self.report({'WARNING'}, "Nothing to flow")
            return {'CANCELLED'}
        params = {"fw_rigid": self.rigid, "fw_flip_u": self.flip_u, "fw_flip_v": self.flip_v, "fw_swap": self.swap,
                  "fw_fold": self.fold}
        hist = context.scene.nproto_settings.construction_history
        made, failed, mode = [], [], ""
        for o in objs:
            inputs = [(o, "OBJ"), (base, "SURF"), (target, "SURF")]
            try:
                g = features.run(OP, inputs, params)
            except Exception as exc:
                failed.append(f"{o.name}: {exc}")
                continue
            mode = LAST_INFO.get("mode", mode)
            r = _result_object(context, o, g)
            if hist:
                features.record(r, OP, inputs, params)
            made.append(r)
        if not made:
            self.report({'ERROR'}, "Flow failed: " + "; ".join(failed))
            return {'CANCELLED'}
        _select_only(context, made)
        how = "folded back face by face" if mode == "fold" else ("moved rigidly" if self.rigid else "flowed")
        msg = f"{len(made)} object{'s' if len(made) != 1 else ''} {how} onto {target.name}"
        if failed:
            msg += " -- failed: " + "; ".join(failed)
        self.report({'WARNING'} if failed else {'INFO'}, msg)
        return {'FINISHED'}


def _result_object(context, src, g):
    """A new object holding the flowed geometry g (world space), in the
    source's collections, named after it."""
    name = src.name + "_flow"
    if g[0] == "splines":
        return new_curve_object(context, src, name, g[1])
    obj = store.new_object_text(context, name, g[1])
    if src.users_collection and context.collection not in src.users_collection:
        for c in src.users_collection:
            if obj.name not in c.objects:
                c.objects.link(obj)
        for c in list(obj.users_collection):
            if c not in src.users_collection:
                c.objects.unlink(obj)
    return obj


classes = (NPROTO_OT_flow,)
