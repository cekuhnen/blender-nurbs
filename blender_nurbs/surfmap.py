"""Maps of surfaces, sampled and fitted: the shared ground of Unroll Surface
(unroll.py) and Flow Along Surface (flow.py).

Both commands build a new surface over the SAME (u, v) domain as a source
surface: Unroll the source's isometric map into the plane, Flow the source
moved through a base -> target map. Keeping the domain keeps the trims, which
live in (u, v) (ext.replace_face_surfaces_keep_uv), and it is what lets Flow
fold an unrolled sheet back exactly: the flat face and its 3D face share
their parameters point for point.

* Axis: one parameter direction cut into panels (the knot spans, subdivided)
  with an 8-point Gauss-Legendre rule in each. Its nodes are the panel ends
  plus the Gauss points, and `cumulative` integrates a function sampled at
  the Gauss points up to every node (the antiderivative of the degree-7
  interpolant in each panel: exact for polynomials of degree 7 per panel).
  A knot is always a panel end, so the integrands are smooth in each panel.
* fit_grid: the least-squares tensor B-spline through samples on a grid of
  nodes, its knots refined (span halving) until it is within tolerance.
"""

import math

import numpy as np

from . import store
from .surfeval import _basis_matrix

K = 8                                           # Gauss points per panel
_x, _w = np.polynomial.legendre.leggauss(K)
GT = (_x + 1.0) / 2.0                           # on [0, 1]
GW = _w / 2.0
_V = GT[:, None] ** np.arange(K)
_A = GT[:, None] ** (np.arange(K) + 1) / (np.arange(K) + 1)
QM = _A @ np.linalg.inv(_V)                     # QM[i, j] = integral of the j-th Lagrange basis over [0, GT[i]]


def breakpoints(knots, lo, hi, n_min=16, sub_min=2, n_max=None):
    """Panel ends in [lo, hi]: every distinct knot, each span cut into equal
    parts so there are at least n_min panels (and at least sub_min per span).
    n_max caps the subdivision (never below one panel per span)."""
    ks = np.unique(np.round(np.asarray(knots, float), 14))
    eps = 1e-12 * max(1.0, hi - lo)
    ks = ks[(ks > lo + eps) & (ks < hi - eps)]
    b = np.concatenate([[lo], ks, [hi]])
    spans = len(b) - 1
    s = max(sub_min, int(math.ceil(n_min / spans)))
    if n_max is not None:
        s = max(1, min(s, n_max // spans))
    out = [b[0]]
    for a, c in zip(b[:-1], b[1:]):
        out += list(a + (c - a) * np.arange(1, s + 1) / s)
    out[-1] = hi
    return np.array(out, float)


class Axis:
    """Panels along one parameter direction and their Gauss points."""

    def __init__(self, b):
        self.b = np.asarray(b, float)
        self.h = np.diff(self.b)
        P = len(self.h)
        self.P = P
        self.g = self.b[:-1, None] + self.h[:, None] * GT[None, :]          # (P, K)
        n = P * (K + 1) + 1
        self.n = n
        self.bi = np.arange(P + 1) * (K + 1)                                # panel ends
        self.gi = (np.arange(P) * (K + 1))[:, None] + 1 + np.arange(K)[None, :]
        t = np.empty(n)
        t[self.bi] = self.b
        t[self.gi] = self.g
        self.t = t

    @property
    def gauss(self):
        return self.g.ravel()

    def cumulative(self, f):
        """Integral from b[0] to every node of f, given (..., P*K) at the
        Gauss points; (..., n)."""
        f = np.asarray(f, float)
        f = f.reshape(f.shape[:-1] + (self.P, K))
        tot = (f @ GW) * self.h                                             # (..., P)
        ends = np.concatenate([np.zeros(f.shape[:-2] + (1,)), np.cumsum(tot, axis=-1)], axis=-1)
        inner = ends[..., :-1, None] + self.h[:, None] * (f @ QM.T)         # (..., P, K)
        out = np.empty(f.shape[:-2] + (self.n,))
        out[..., self.bi] = ends
        out[..., self.gi] = inner
        return out

    def node_of(self, t):
        """Index of the node at parameter t (a panel end)."""
        return int(self.bi[int(np.argmin(np.abs(self.b - t)))])

    def excl_sum(self, jumps):
        """(..., n): the sum of jumps (..., P+1; one per panel end) at the
        panel ends strictly before each node (a panel end itself excluded)."""
        jumps = np.asarray(jumps, float)
        c = np.cumsum(jumps, axis=-1)                                       # through end k
        before = np.concatenate([np.zeros(jumps.shape[:-1] + (1,)), c[..., :-1]], axis=-1)   # ends < k
        out = np.empty(jumps.shape[:-1] + (self.n,))
        out[..., self.bi] = before
        out[..., self.gi] = np.repeat(c[..., :-1, None], K, axis=-1)       # ends <= panel start
        return out


# --------------------------------------------------------------------------
# Fitting a tensor B-spline to grid samples
# --------------------------------------------------------------------------

def fit_knots(knots, p_src, p, lo, hi):
    """A clamped knot vector of degree p over [lo, hi] with the source's
    distinct interior knots, each repeated so the fit may be as little
    smooth there as the source is (C^(p_src - m) at a knot of multiplicity m)."""
    U = np.asarray(knots, float)
    eps = 1e-12 * max(1.0, hi - lo)
    out = [lo] * (p + 1)
    for k in np.unique(np.round(U, 14)):
        if lo + eps < k < hi - eps:
            m = int(np.sum(np.abs(U - k) <= 1e-12 * max(1.0, abs(k))))
            cont = p_src - m                              # the source's continuity there
            out += [float(k)] * max(1, min(p, p - cont))
    out += [hi] * (p + 1)
    return out


def _with(knots, p, extra):
    U = list(knots[p + 1:-(p + 1)])
    U = sorted(U + [float(x) for x in extra])
    return list(knots[:p + 1]) + U + list(knots[-(p + 1):])


def _spans(U):
    ks = np.unique(np.asarray(U, float))
    return ks


def _span_index(ks, t):
    return np.clip(np.searchsorted(ks, t, side="right") - 1, 0, len(ks) - 2)


def basis(U, p, t):
    U = np.asarray(U, float)
    return _basis_matrix(U, p, len(U) - p - 1, t, 0)[0]


def _lstsq(B, X):
    """Least squares B C = X (X: (m, d)); C (n, d)."""
    return np.linalg.lstsq(B, X, rcond=None)[0]


def fit_grid(tu, tv, X, U, p, V, q, tol, min_u=None, min_v=None, max_spans=400, rounds=10):
    """(C, U, V, err): the degree (p, q) B-spline with control net C (nu, nv, d)
    on knots U, V nearest the samples X (len(tu), len(tv), d) at the grid
    (tu x tv), the knots halved in the spans that miss by more than tol
    (each direction on its own: the 1D fits along it), at most `rounds`
    times. min_u / min_v: spans are not cut below this width (the sample
    spacing). err: the largest miss of the final fit at the samples."""
    tu, tv = np.asarray(tu, float), np.asarray(tv, float)
    X = np.asarray(X, float)
    nu_s, nv_s, d = X.shape
    U, V = list(U), list(V)
    for _ in range(rounds):
        Bu, Bv = basis(U, p, tu), basis(V, q, tv)
        # 1D fits: along u for every v column, along v for every u row
        Cu = _lstsq(Bu, X.reshape(nu_s, -1))
        Ru = np.linalg.norm((Bu @ Cu).reshape(X.shape) - X, axis=2).max(axis=1)      # (nu_s,)
        Cv = _lstsq(Bv, X.transpose(1, 0, 2).reshape(nv_s, -1))
        Rv = np.linalg.norm((Bv @ Cv).reshape(nv_s, nu_s, d) - X.transpose(1, 0, 2), axis=2).max(axis=1)
        add_u, add_v = [], []
        for R, T, W, add, mn in ((Ru, tu, U, add_u, min_u), (Rv, tv, V, add_v, min_v)):
            ks = _spans(W)
            if len(ks) - 1 >= max_spans:
                continue
            si = _span_index(ks, T)
            for s in np.unique(si[R > 0.5 * tol]):
                a, b = ks[s], ks[s + 1]
                if mn is None or (b - a) > 1.5 * mn:
                    add.append(0.5 * (a + b))
        if not add_u and not add_v:
            break
        U = _with(U, p, add_u) if add_u else U
        V = _with(V, q, add_v) if add_v else V
    Bu, Bv = basis(U, p, tu), basis(V, q, tv)
    C1 = _lstsq(Bu, X.reshape(nu_s, -1)).reshape(-1, nv_s, d)                # (nu, nv_s, d)
    C = _lstsq(Bv, C1.transpose(1, 0, 2).reshape(nv_s, -1)).reshape(-1, C1.shape[0], d).transpose(1, 0, 2)
    Y = np.einsum("ia,abd,jb->ijd", Bu, C, Bv)
    err = float(np.linalg.norm(Y - X, axis=2).max())
    return C, U, V, err


def degree_for(p_src, linear_ok):
    """The fit's degree along a direction: 1 where the map stays linear along
    the source's straight iso lines, else at least cubic."""
    if p_src == 1 and linear_ok:
        return 1
    return max(3, int(p_src))


def surface_from_net(C, U, p, V, q):
    """A kernel surface from a non-rational net C (nu, nv, 3)."""
    _, ext = store._k()
    nu, nv = C.shape[:2]
    pts = [tuple(map(float, C[i, j])) for i in range(nu) for j in range(nv)]
    return ext.surface_from_cvs(int(p), int(q), pts, [float(x) for x in U], [float(x) for x in V], None)


# --------------------------------------------------------------------------
# A face rebuilt on a new surface, its trims taken along
# --------------------------------------------------------------------------

EDGE_SAMPLES = 64
BREP_TOL = 1e-5        # a rebuilt face's own tolerance, a share of the size (the stitch's)


def edge_samples(e, n=9):
    """(n, 3) points along a kernel edge, evenly in its parameter."""
    t0, t1 = e.parameter_range()
    return np.array([e.point_at(t0 + (t1 - t0) * k / (n - 1)) for k in range(n)], float)


def loop_uvs(S, loop):
    """[(uv (n, 2), points (n, 3))] of each edge of a boundary loop, in loop
    order, the parameters of a periodic surface unwrapped along the loop (a
    seam edge, used twice, lies once on each side of the domain)."""
    pts = []
    for e, fwd in loop:
        P = edge_samples(e, EDGE_SAMPLES)
        pts.append(P if fwd else P[::-1])
    allp = np.vstack(pts)
    uv = S.feet(allp)
    pu, pv = S.periodic()
    for c, per, (a, b) in ((0, pu, S.u_domain), (1, pv, S.v_domain)):
        if not per:
            continue
        T = b - a
        x = uv[:, c].copy()
        for k in range(1, len(x)):
            x[k] += T * np.round((x[k - 1] - x[k]) / T)
        # the whole loop moved by whole periods to lie in the domain
        best = min(range(-2, 3), key=lambda m: float(np.sum(np.maximum(a - (x + m * T), 0.0)
                                                            + np.maximum((x + m * T) - b, 0.0))))
        uv[:, c] = np.clip(x + best * T, a, b)
    out, k = [], 0
    for P in pts:
        out.append((uv[k:k + len(P)], P))
        k += len(P)
    return out


def _on_border(S, uv, tol=1e-9):
    (u0, u1), (v0, v1) = S.u_domain, S.v_domain
    du, dv = tol * (u1 - u0), tol * (v1 - v0)
    return bool(np.all(np.abs(uv[:, 0] - u0) <= du) or np.all(np.abs(uv[:, 0] - u1) <= du)
                or np.all(np.abs(uv[:, 1] - v0) <= dv) or np.all(np.abs(uv[:, 1] - v1) <= dv))


def is_natural_face(face, S):
    """One loop, every edge along a side of the surface's domain."""
    loops = [lp for lp in face.boundary_loops() if lp]
    if len(loops) != 1:
        return False
    return all(_on_border(S, uv) for uv, _ in loop_uvs(S, loops[0]))


def curve_through(X, size):
    """A kernel curve through the world points X (a straight segment when
    they are on a line)."""
    sm, _ = store._k()
    X = np.asarray(X, float)
    d = X[-1] - X[0]
    L = float(np.linalg.norm(d))
    if L > 1e-12 * size:
        off = X - X[0] - np.outer((X - X[0]) @ d / (L * L), d)
        if float(np.linalg.norm(off, axis=1).max()) <= 1e-9 * size:
            return sm.create_line_segment(tuple(map(float, X[0])), tuple(map(float, X[-1])))
    keep = np.concatenate([[True], np.linalg.norm(np.diff(X, axis=0), axis=1) > 1e-12 * size])
    return sm.create_curve_interp_points([tuple(map(float, x)) for x in X[keep]])


def trimmed_face(face, S, make_surface, uv_to_world, size):
    """A new face for `face`: the surface make_surface() gives, trimmed by
    the face's edges taken through uv_to_world ((m, 2) parameters of S ->
    (m, 3) world points), one curve per edge so that edges shared with other
    rebuilt faces meet edge to edge. make_surface is called for every try
    (the kernel takes the surface and curves it trims with).

    The face's own tolerance follows size (BREP_TOL), not the kernel's
    default 0.01 model units: the faces are stitched together again by
    their vertices, and two corners closer than twice that tolerance (a
    polygon ring of short edges) were merged into one, the wrong edges
    paired and the body left open."""
    sm, ext = store._k()
    loops = [lp for lp in face.boundary_loops() if lp]
    recipes = [[uv_to_world(uv) for uv, _ in loop_uvs(S, lp)] for lp in loops]
    sizes = [len(lp) for lp in recipes]
    last = None
    for tol in (1e-7, 1e-5, 1e-4):
        try:
            curves = [curve_through(X, size) for lp in recipes for X in lp]
            b = ext.trim_surface_with_3d_curves_tol(make_surface(), sizes, curves, tol * size,
                                                    max(BREP_TOL, 10.0 * tol) * size)
            if b is not None and b.face_count():
                return b
        except Exception as exc:
            last = exc
    curves = [curve_through(X, size) for lp in recipes for X in lp]
    b = sm.trim_surface_with_3d_curves(make_surface(), sizes, curves)
    if b is None or not b.face_count():
        raise RuntimeError(f"the trim failed ({last})")
    return b
