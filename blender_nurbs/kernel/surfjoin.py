"""Remove Edges with a Fit Tolerance: faces that meet tangentially but lie on
different surfaces (patches of an import or of SubD to BREP, a NURBS face
beside a slightly different one, a flat face running on into a curved one)
merged into ONE face on one new NURBS surface fitted through them, within the
tolerance (Rhino's MergeSrf Smooth, here inside a body: the faces round them
keep their own surfaces and edges).

The faces' surfaces must line up side to side: each face's patch (its
surface over the face's parameter box, cut out exactly; a flat face gets a
rectangle lined up with the edge it shares) is a cell, and the cells form a
grid in which every removed edge runs along a whole shared side (a row of
patches, or rows of them). One chain of cells at a time (rows, then the rows
as one chain) is made one reference surface: the cells side by side, each
over a share of the parameter range in proportion to its speed at the seams
(so the reference is smooth in its parameters too), and the parameter along
each seam taken over from the cell before (a 1D reparameterization of the
next cell, so the two sides of a seam meet at the same parameters even when
the patches are parameterized differently). That reference is fitted by
least squares on a B-spline net (degree 3, or the cells' own when higher),
spans added where the fit is off, until it is within the tolerance (a grid
of rows: half each time). A side that closes on itself (a ring) is closed
again on the fit, with a smooth seam.

The merged face is the fitted surface trimmed by the cluster's outline (the
edges as they were, within the tolerance of the new surface), or the whole
surface when every face was a whole patch. Accepted when it stitches back
into the body with the face count it should have, a solid staying a solid
(its volume within area x tolerance).

Refused (the edges stay and say why): a crease over TANGENT_DEG (a refit
would smooth it), faces whose surfaces do not line up side to side, a ring
of faces going all the way round, a fit that would need more than MAX_CVS
control points a way.
"""

import math

import numpy as np

try:
    from . import mergefaces as mf
    from .classa import basis
    from .trimext import brep_tol
except ImportError:          # the worker imports the kernel folder's modules top-level
    import mergefaces as mf
    from classa import basis
    from trimext import brep_tol

TANGENT_DEG = 0.5            # faces meeting at more than this are a crease: never refitted
MAX_CVS = 80                 # control points a way, at most
MAX_DEGREE = 7
G2_REL = 0.05                # Remove Redundant Edges refits only joints whose curvature matches this closely
SIDES = ("u0", "u1", "v0", "v1")


# -- nets: {"H": homogeneous CVs (nu, nv, 4), "Uu", "Uv": knots on 0..1, "pu", "pv"} ------

def _net(ext, surf):
    d = ext.surface_cvs(surf)
    nu, nv = int(d["count_u"]), int(d["count_v"])
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    w = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    H = np.concatenate([P * w[..., None], w[..., None]], axis=2)
    Ku, Kv = np.array(d["knots_u"], float), np.array(d["knots_v"], float)
    dom = (Ku[0], Ku[-1], Kv[0], Kv[-1])
    return {"H": H, "Uu": (Ku - Ku[0]) / (Ku[-1] - Ku[0]), "Uv": (Kv - Kv[0]) / (Kv[-1] - Kv[0]),
            "pu": int(d["degree_u"]), "pv": int(d["degree_v"])}, dom


def _grid(net, su, sv, du=0, dv=0):
    """Points (du = dv = 0) or first derivatives on the grid su x sv."""
    Bu = basis(net["Uu"], net["pu"], su)
    Bv = basis(net["Uv"], net["pv"], sv)
    H = np.einsum("ia,abk,jb->ijk", Bu, net["H"], Bv)
    S = H[..., :3] / H[..., 3:]
    if not (du or dv):
        return S
    Bu1 = basis(net["Uu"], net["pu"], su, du) if du else Bu
    Bv1 = basis(net["Uv"], net["pv"], sv, dv) if dv else Bv
    D = np.einsum("ia,abk,jb->ijk", Bu1, net["H"], Bv1)
    return (D[..., :3] - D[..., 3:] * S) / H[..., 3:]


def _pts(net, u, v):
    """Points and first derivatives at scattered parameters (u[i], v[i])."""
    u, v = np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)
    Bu, Bv = basis(net["Uu"], net["pu"], u), basis(net["Uv"], net["pv"], v)
    Bu1, Bv1 = basis(net["Uu"], net["pu"], u, 1), basis(net["Uv"], net["pv"], v, 1)
    H = np.einsum("ia,abk,ib->ik", Bu, net["H"], Bv)
    Hu = np.einsum("ia,abk,ib->ik", Bu1, net["H"], Bv)
    Hv = np.einsum("ia,abk,ib->ik", Bu, net["H"], Bv1)
    w = H[:, 3:]
    S = H[:, :3] / w
    return S, (Hu[:, :3] - Hu[:, 3:] * S) / w, (Hv[:, :3] - Hv[:, 3:] * S) / w


def _invert(net, Q, n=25, iters=12):
    """Parameters (u, v) of the points Q on the net and their distance."""
    Q = np.asarray(Q, float).reshape(-1, 3)
    t = np.linspace(0, 1, n)
    X = _grid(net, t, t).reshape(-1, 3)
    k = np.array([int(np.argmin(((X - q) ** 2).sum(1))) for q in Q])
    u, v = t[k // n].copy(), t[k % n].copy()
    for _ in range(iters):
        S, Su, Sv = _pts(net, u, v)
        r = S - Q
        a, b, c = (Su * Su).sum(1), (Su * Sv).sum(1), (Sv * Sv).sum(1)
        gu, gv = (Su * r).sum(1), (Sv * r).sum(1)
        det = a * c - b * b
        det = np.where(np.abs(det) > 1e-300, det, 1e-300)
        u = np.clip(u - (c * gu - b * gv) / det, 0.0, 1.0)
        v = np.clip(v - (a * gv - b * gu) / det, 0.0, 1.0)
    return u, v, np.linalg.norm(_pts(net, u, v)[0] - Q, axis=1)


def _orient(net, tr, fu, fv):
    H, Uu, Uv, pu, pv = net["H"], net["Uu"], net["Uv"], net["pu"], net["pv"]
    if tr:
        H, Uu, Uv, pu, pv = H.transpose(1, 0, 2), Uv, Uu, pv, pu
    if fu:
        H, Uu = H[::-1], 1.0 - Uu[::-1]
    if fv:
        H, Uv = H[:, ::-1], 1.0 - Uv[::-1]
    return {"H": H, "Uu": Uu, "Uv": Uv, "pu": pu, "pv": pv}


def _orientations(net):
    return [_orient(net, tr, fu, fv) for tr in (0, 1) for fu in (0, 1) for fv in (0, 1)]


def _side(net, s, m):
    """The boundary curve s sampled at m parameters (along +v for u sides, +u for v sides)."""
    t = np.linspace(0, 1, m)
    if s in ("u0", "u1"):
        return _grid(net, [0.0 if s == "u0" else 1.0], t)[0]
    return _grid(net, t, [0.0 if s == "v0" else 1.0])[:, 0]


# -- cutting a patch out of a surface (exact) ----------------------------------------

def _insert(H, U, p, t):
    """Knot t inserted once along axis 0 (Boehm)."""
    U = list(U)
    k = int(np.searchsorted(U, t, side="right") - 1)
    s = sum(1 for x in U if abs(x - t) <= 1e-12)
    n = H.shape[0]
    Q = np.empty((n + 1,) + H.shape[1:])
    Q[:k - p + 1] = H[:k - p + 1]
    for i in range(k - p + 1, k - s + 1):
        a = (t - U[i]) / (U[i + p] - U[i])
        Q[i] = a * H[i] + (1 - a) * H[i - 1]
    Q[k - s + 1:] = H[k - s:]
    return Q, U[:k + 1] + [t] + U[k + 1:]


def _clip_axis(H, U, p, a, b):
    """The piece [a, b] (0 <= a < b <= 1) along axis 0, knots on 0..1 again."""
    U = [float(x) for x in U]
    eps = 1e-12
    for t in (a, b):
        if U[0] + eps < t < U[-1] - eps:
            s = sum(1 for x in U if abs(x - t) <= eps)
            for _ in range(max(0, p - s)):
                H, U = _insert(H, U, p, t)
    r = max(i for i in range(len(U)) if U[i] <= a + eps)
    s = min(i for i in range(len(U)) if U[i] >= b - eps)
    H = H[r - p:s]
    K = np.array([a] * (p + 1) + U[r + 1:s] + [b] * (p + 1))
    return H, (K - a) / (b - a)


def _subnet(net, box):
    """The net over the box (u0, u1, v0, v1) of its 0..1 parameters, exactly."""
    u0, u1, v0, v1 = box
    H, Uu = _clip_axis(net["H"], net["Uu"], net["pu"], u0, u1)
    Ht, Uv = _clip_axis(H.transpose(1, 0, 2), net["Uv"], net["pv"], v0, v1)
    return {"H": Ht.transpose(1, 0, 2), "Uu": Uu, "Uv": Uv, "pu": net["pu"], "pv": net["pv"]}


def _reseam_axis(H, U, p, t):
    """A closed run of CVs along axis 0 started again at t (exact): [t, 1]
    then [0, t], joined where the old seam was."""
    if t < 1e-6 or t > 1 - 1e-6:
        return H, np.asarray(U, float)
    H1, U1 = _clip_axis(H, U, p, t, 1.0)
    H2, U2 = _clip_axis(H, U, p, 0.0, t)
    a, b = H1[-1], H2[0]
    s = float(np.mean(a[..., 3] / b[..., 3]))
    if np.abs(a[..., :3] / a[..., 3:] - b[..., :3] / b[..., 3:]).max() > 1e-9 * (1 + np.abs(a).max()):
        return None, None
    H2 = H2 * s
    w1 = 1.0 - t
    K = np.r_[[0.0] * (p + 1), w1 * U1[p + 1:-(p + 1)], [w1] * p, w1 + t * U2[p + 1:-(p + 1)], [1.0] * (p + 1)]
    return np.concatenate([H1, H2[1:]], axis=0), K


def reseam(net, d, t):
    """The net with its closed direction d (0: u, 1: v) started at t, or None."""
    if d == 0:
        H, U = _reseam_axis(net["H"], net["Uu"], net["pu"], t)
        return None if H is None else dict(net, H=H, Uu=U)
    H, U = _reseam_axis(net["H"].transpose(1, 0, 2), net["Uv"], net["pv"], t)
    return None if H is None else dict(net, H=H.transpose(1, 0, 2), Uv=U)


def _bilinear(c, e1, e2, x0, x1, y0, y1):
    C = np.array([[c + x0 * e1 + y0 * e2, c + x0 * e1 + y1 * e2],
                  [c + x1 * e1 + y0 * e2, c + x1 * e1 + y1 * e2]], float)
    H = np.concatenate([C, np.ones((2, 2, 1))], axis=2)
    K = np.array([0.0, 0.0, 1.0, 1.0])
    return {"H": H, "Uu": K, "Uv": K.copy(), "pu": 1, "pv": 1}


def _cell(body, i, shared):
    """Face i's patch: its surface over the face's parameter box, or for a flat
    face a rectangle lined up with one of its shared straight edges. None
    when it has no clean box (a face across a closed surface's seam)."""
    f = body.faces[i]
    pts = [p for e in f.edges() for p in mf._edge_pts(e, 9)]
    pl = body.plane(i)
    if pl is not None:
        for e in shared:
            a, z = mf._edge_pts(e, 2)
            d = z - a
            if np.linalg.norm(d) <= body.tol or not _straight(e, body.tol):
                continue
            e1 = d / np.linalg.norm(d)
            n = pl[1]
            e1 = e1 - (e1 @ n) * n
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            P = np.asarray(pts) - a
            x, y = P @ e1, P @ e2
            return _bilinear(a, e1, e2, x.min(), x.max(), y.min(), y.max())
    net, dom = _net(body.sm.blender_ext, f.surface())
    uv = []
    for p in pts:
        (u, v), _ = mf._uv(body.sm, f.surface(), p)
        uv.append(((u - dom[0]) / (dom[1] - dom[0]), (v - dom[2]) / (dom[3] - dom[2])))
    uv = np.clip(np.array(uv), 0.0, 1.0)
    box = []
    for d, (s0, s1) in enumerate((("u0", "u1"), ("v0", "v1"))):
        x = uv[:, d]
        lo, hi = float(x.min()), float(x.max())
        if np.linalg.norm(_side(net, s0, 9) - _side(net, s1, 9), axis=1).max() <= 10 * body.tol:
            # a closed direction: a point on the seam is at 0 or 1, so the box is what the
            # widest gap between the face's parameters (round the seam) leaves
            xs = np.sort(x)
            gaps = np.r_[np.diff(xs), xs[0] + 1.0 - xs[-1]]
            k = int(np.argmax(gaps))
            mid = 0.5 * (xs[k] + xs[k + 1]) if k < len(xs) - 1 else (0.5 * (xs[-1] + xs[0] + 1.0)) % 1.0
            other = float(np.median(uv[:, 1 - d]))
            q = _pts(net, *(([mid], [other]) if d == 0 else ([other], [mid])))[0][0]
            if float(f.closest_point(tuple(float(c) for c in q))[1]) <= 10 * body.tol:
                lo, hi = 0.0, 1.0                  # the face goes all the way round
            elif k < len(xs) - 1:
                lo_w, hi_w = float(xs[k + 1]), float(xs[k])
                if lo_w <= 1 - 1e-6 and hi_w >= 1e-6:
                    return None                    # across the seam: no box
                lo, hi = (lo_w, 1.0) if hi_w < 1e-6 else (0.0, hi_w)
        box += [lo, hi]
    box = [0.0 if x < 1e-6 else 1.0 if x > 1 - 1e-6 else float(x) for x in box]
    if box[1] - box[0] < 1e-6 or box[3] - box[2] < 1e-6:
        return None
    if box == [0.0, 1.0, 0.0, 1.0]:
        return net
    return _subnet(net, box)


def _straight(e, tol):
    a, z = mf._edge_pts(e, 2)
    L = np.linalg.norm(z - a)
    if L <= tol:
        return False
    d = (z - a) / L
    for q in mf._edge_pts(e, 7)[1:-1]:
        off = q - a
        if np.linalg.norm(off - (off @ d) * d) > tol:
            return False
    return True


# -- the cells as a grid ------------------------------------------------------------

NEXT = {"u1": ("u0", (1, 0)), "u0": ("u1", (-1, 0)), "v1": ("v0", (0, 1)), "v0": ("v1", (0, -1))}


def _on(P, Q, tol):
    """Do the points P lie on the polyline Q (each within tol of a segment)?"""
    A, B = Q[:-1], Q[1:]
    D = B - A
    L2 = (D * D).sum(1).clip(1e-300)
    for p in P:
        t = (((p - A) * D).sum(1) / L2).clip(0.0, 1.0)
        if np.linalg.norm(A + t[:, None] * D - p, axis=1).min() > tol:
            return False
    return True


def _meets(a, s, b, t, tol):
    """Does side t of net b run along side s of net a, the same way?"""
    S, T = _side(a, s, 41), _side(b, t, 41)
    if np.linalg.norm(S[0] - T[0]) > tol or np.linalg.norm(S[-1] - T[-1]) > tol:
        return False
    return _on(T[::4], S, tol) and _on(S[::4], T, tol)


def _reseamed_to(a, s, b, t, tol):
    """b with its seam moved so that its side t (a closed curve, as side s of
    a is) starts where side s does, or None."""
    S, T = _side(a, s, 41), _side(b, t, 41)
    if np.linalg.norm(S[0] - S[-1]) > tol or np.linalg.norm(T[0] - T[-1]) > tol:
        return None
    if not (_on(T[::4], S, tol) and _on(S[::4], T, tol)):
        return None
    d = 1 if t in ("u0", "u1") else 0               # the direction the side runs along
    m = 401
    TT = _side(b, t, m)
    x = float(np.argmin(np.linalg.norm(TT - S[0], axis=1))) / (m - 1)
    for _ in range(8):                              # Newton along the side
        uv = ([0.0 if t == "u0" else 1.0], [x]) if d == 1 else ([x], [0.0 if t == "v0" else 1.0])
        P, Su, Sv = _pts(b, np.array(uv[0]), np.array(uv[1]))
        D = Sv[0] if d == 1 else Su[0]
        x = float(np.clip(x - (P[0] - S[0]) @ D / max(D @ D, 1e-300), 0.0, 1.0))
    return reseam(b, d, x)


def layout(cells, links, tol):
    """{(i, j): oriented net} of the cells (a dict index -> net) on a grid in
    which every link (pair of cell indices joined by removed edges) is a whole
    shared side, or None."""
    ids = list(cells)
    first = ids[0]
    where = {first: (0, 0)}
    grid = {(0, 0): cells[first]}
    todo = [first]
    nbr = {i: set() for i in ids}
    for a, b in links:
        nbr[a].add(b)
        nbr[b].add(a)
    while todo:
        a = todo.pop()
        ia, ja = where[a]
        na = grid[(ia, ja)]
        for b in sorted(nbr[a]):
            hit = None
            for s, (t, (di, dj)) in NEXT.items():
                for nb in _orientations(cells[b]):
                    if _meets(na, s, nb, t, tol):
                        hit = ((ia + di, ja + dj), nb)
                        break
                    rb = None if b in where else _reseamed_to(na, s, nb, t, tol)
                    if rb is not None and _meets(na, s, rb, t, tol):
                        hit = ((ia + di, ja + dj), rb)
                        break
                if hit:
                    break
            if hit is None:
                return None
            cell, nb = hit
            if b in where:
                if where[b] != cell:
                    return None
                continue
            if cell in grid:
                return None
            where[b], grid[cell] = cell, nb
            todo.append(b)
    if len(where) != len(ids):
        return None
    i0, j0 = min(i for i, _ in grid), min(j for _, j in grid)
    grid = {(i - i0, j - j0): v for (i, j), v in grid.items()}
    I, J = 1 + max(i for i, _ in grid), 1 + max(j for _, j in grid)
    if len(grid) != I * J:
        return None
    return grid, I, J


# -- one reference surface out of a chain of nets along u ----------------------------

def _speed(net, s):
    """Mean |dS/du| along side s (u0 / u1)."""
    t = np.linspace(0, 1, 9)
    D = _grid(net, [0.0 if s == "u0" else 1.0], t, du=1)[0]
    return float(np.linalg.norm(D, axis=1).mean())


class _Chain:
    """Nets side by side along u (each u1 side the next one's u0) as one
    reference S(u, v) on 0..1 x 0..1: net k over [a_k, a_k + w_k] in u, and
    its v parameter the one that meets the net before it at the seam."""

    def __init__(self, nets):
        self.nets = nets
        w = [1.0]
        for a, b in zip(nets[:-1], nets[1:]):
            r = _speed(b, "u0") / max(_speed(a, "u1"), 1e-300)
            w.append(w[-1] * min(10.0, max(0.1, r)))
        w = np.array(w) / sum(w)
        self.w, self.a = w, np.r_[0.0, np.cumsum(w)[:-1]]

    def warp(self, k, v):
        """Net k's v parameter for the chain's v (that of net 0)."""
        v = np.asarray(v, float)
        if k == 0:
            return v
        prev = self.warp(k - 1, v)
        q = _pts(self.nets[k - 1], np.ones(len(v)), prev)[0]
        nb = self.nets[k]
        t = np.linspace(0, 1, 201)
        side = _grid(nb, [0.0], t)[0]
        x = t[np.argmin(((side[None] - q[:, None]) ** 2).sum(2), axis=1)]
        for _ in range(10):
            S, _, Sv = _pts(nb, np.zeros(len(x)), x)
            x = np.clip(x - ((S - q) * Sv).sum(1) / (Sv * Sv).sum(1).clip(1e-300), 0.0, 1.0)
        return x

    def __call__(self, us, vs):
        us = np.asarray(us, float)
        out = np.empty((len(us), len(vs), 3))
        k_of = np.clip(np.searchsorted(self.a, us, side="right") - 1, 0, len(self.nets) - 1)
        for k, net in enumerate(self.nets):
            m = k_of == k
            if m.any():
                s = ((us[m] - self.a[k]) / self.w[k]).clip(0.0, 1.0)
                vk = self.warp(k, vs)
                Bu = basis(net["Uu"], net["pu"], s)
                Bv = basis(net["Uv"], net["pv"], vk)
                # vk differs per v sample but not per u: one row of basis per v
                H = np.einsum("ia,abk,jb->ijk", Bu, net["H"], Bv)
                out[m] = H[..., :3] / H[..., 3:]
        return out

    def knots_u(self):
        out = [x for k, n in enumerate(self.nets) for x in self.a[k] + self.w[k] * np.unique(n["Uu"])]
        return out


def _inner(U):
    U = np.unique(np.round(np.asarray(U, float), 12))
    return [float(x) for x in U if 1e-9 < x < 1 - 1e-9]


def _knots(p, inner):
    return np.array([0.0] * (p + 1) + sorted(inner) + [1.0] * (p + 1))


def _params(U, per):
    br = np.unique(np.round(np.asarray(U, float), 12))
    t = [br[0]]
    for a, b in zip(br[:-1], br[1:]):
        t += list(np.linspace(a, b, per + 1)[1:])
    return np.array(t)


def _thin(inner, gap=1e-3):
    out = []
    for x in sorted(inner):
        if (not out or x - out[-1] > gap) and gap < x < 1 - gap:
            out.append(x)
    return out


def _closed_v(ref, tol):
    t = np.linspace(0, 1, 21)
    G = ref(t, [0.0, 1.0])
    return float(np.linalg.norm(G[:, 0] - G[:, 1], axis=1).max()) <= tol


def _close_v(C, Uv):
    """The net closed again along v (first and last columns one, a smooth seam)."""
    C = C.copy()
    nv = C.shape[1]
    if nv < 4:
        return C
    P0 = 0.5 * (C[:, 0] + C[:, -1])
    h0, h1 = Uv[pvp(Uv) + 1] - Uv[0], Uv[-1] - Uv[-pvp(Uv) - 2]
    d = 0.5 * ((C[:, 1] - C[:, 0]) / h0 + (C[:, -1] - C[:, -2]) / h1)
    C[:, 0] = C[:, -1] = P0
    C[:, 1], C[:, -2] = P0 + d * h0, P0 - d * h1
    return C


def pvp(U):
    """The degree of a clamped knot vector (its end multiplicity - 1)."""
    return sum(1 for x in U if abs(x - U[0]) <= 1e-12) - 1


def fit(ref, pu, pv, Uu0, Uv0, tol, closed_v=False):
    """Least-squares net of ref on knots grown from Uu0 x Uv0 until within
    tol: (net, deviation) or (None, why)."""
    inu, inv = _thin(_inner(Uu0)), _thin(_inner(Uv0))
    for _ in range(14):
        Uu, Uv = _knots(pu, inu), _knots(pv, inv)
        nu, nv = len(Uu) - pu - 1, len(Uv) - pv - 1
        if nu > MAX_CVS or nv > MAX_CVS:
            return None, f"the fit needs more than {MAX_CVS} control points a way"
        su, sv = _params(Uu, pu + 3), _params(Uv, pv + 3)
        G = ref(su, sv)
        Bu, Bv = basis(Uu, pu, su), basis(Uv, pv, sv)
        X1 = np.linalg.lstsq(Bu, G.reshape(len(su), -1), rcond=None)[0]           # (nu, Mv*3)
        Eu = np.linalg.norm((Bu @ X1 - G.reshape(len(su), -1)).reshape(len(su), len(sv), 3), axis=2).max(1)
        Gt = G.transpose(1, 0, 2).reshape(len(sv), -1)
        Xv = np.linalg.lstsq(Bv, Gt, rcond=None)[0]
        Ev = np.linalg.norm((Bv @ Xv - Gt).reshape(len(sv), len(su), 3), axis=2).max(1)
        Y = X1.reshape(nu, len(sv), 3).transpose(1, 0, 2).reshape(len(sv), -1)
        C = np.linalg.lstsq(Bv, Y, rcond=None)[0].reshape(nv, nu, 3).transpose(1, 0, 2)
        if closed_v:
            C = _close_v(C, Uv)
        net = {"H": np.concatenate([C, np.ones((nu, nv, 1))], axis=2), "Uu": Uu, "Uv": Uv, "pu": pu, "pv": pv}
        tu, tv = _params(Uu, 2 * pu + 4), _params(Uv, 2 * pv + 4)
        E = np.linalg.norm(_grid(net, tu, tv) - ref(tu, tv), axis=2)
        dev = float(E.max())
        if dev <= tol:
            return net, dev
        grow = False
        for (inner, U, e, s, p) in ((inu, Uu, Eu, su, pu), (inv, Uv, Ev, sv, pv)):
            br = np.unique(np.round(U, 12))
            emax = max(float(e.max()), 1e-300)
            for a, b in zip(br[:-1], br[1:]):
                m = (s >= a) & (s <= b)
                if m.any() and e[m].max() > 0.3 * tol and e[m].max() >= 0.2 * emax and b - a > 2e-3:
                    inner.append(0.5 * (a + b))
                    grow = True
        if not grow:          # the 1D fits are fine but not the surface: refine where the surface is off
            iu, iv = np.unravel_index(int(np.argmax(E)), E.shape)
            for inner, U, t in ((inu, Uu, tu[iu]), (inv, Uv, tv[iv])):
                br = np.unique(np.round(U, 12))
                k = int(np.clip(np.searchsorted(br, t) - 1, 0, len(br) - 2))
                inner.append(0.5 * (br[k] + br[k + 1]))
        inu.sort()
        inv.sort()
    return None, f"the fit stays {dev:.3g} off (more than the tolerance)"


def fit_chain(nets, tol):
    """One net for nets side by side along u, within tol: (net, dev) or (None, why)."""
    ref = _Chain(nets)
    pu = min(MAX_DEGREE, max(3, max(n["pu"] for n in nets)))
    pv = min(MAX_DEGREE, max(3, max(n["pv"] for n in nets)))
    closed = _closed_v(ref, tol)
    Uu0 = _knots(pu, list(ref.knots_u()))
    inv = [x for n in nets[:1] for x in _inner(n["Uv"])]
    Uv0 = _knots(pv, inv + list(np.linspace(0, 1, 5)[1:-1]))
    Uu0 = _knots(pu, list(_inner(Uu0)) + list(np.linspace(0, 1, 2 * len(nets) + 1)[1:-1]))
    return fit(ref, pu, pv, Uu0, Uv0, tol, closed)


def fit_grid(grid, I, J, tol):
    """One net for the grid of nets, within tol (rows within tol / 2, then
    the rows as one chain within tol / 2)."""
    if J == 1 or I == 1:
        nets = [grid[(i, 0)] for i in range(I)] if J == 1 else [_orient(grid[(0, j)], 1, 0, 0) for j in range(J)]
        net, dev = fit_chain(nets, tol)
        return (net if J == 1 or net is None else _orient(net, 1, 0, 0)), dev
    rows, devs = [], []
    for j in range(J):
        r, d = fit_chain([grid[(i, j)] for i in range(I)], 0.5 * tol)
        if r is None:
            return None, d
        rows.append(_orient(r, 1, 0, 0))
        devs.append(d)
    net, dev = fit_chain(rows, 0.5 * tol)
    if net is None:
        return None, dev
    return _orient(net, 1, 0, 0), dev + max(devs)


# -- candidates, and the merge ---------------------------------------------------------

def tangent_angle(body, e):
    """The largest angle (degrees) between the two faces along edge e when it
    can go by a refit (at most TANGENT_DEG, not both flat: planerefit does
    those), else None."""
    ij = body.edge_faces(e)
    if ij is None:
        return None
    i, j = ij
    if body.plane(i) is not None and body.plane(j) is not None:
        return None
    worst = 0.0
    try:
        for p in mf._edge_pts(e, 5):
            ni, nj = mf._outward(body.sm, body.faces[i], p), mf._outward(body.sm, body.faces[j], p)
            worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, float(ni @ nj))))))
    except Exception:
        return None
    return worst if worst <= TANGENT_DEG else None


def curvature_jump(body, e):
    """How much the two faces' normal curvature across edge e differs, as a
    share of the larger (or of 1 / size): ~0 where they join curvature-
    continuously (patches of one smooth shape), ~1 at a fillet's edge."""
    i, j = body.edge_faces(e)
    t0, t1 = e.parameter_range()
    tm = 0.5 * (t0 + t1)
    p = mf._pt(e.point_at(tm))
    t = np.array(e.tangent_at(tm), float)
    t /= np.linalg.norm(t) or 1.0
    h = 2e-3 * body.size
    ks = []
    for k in (i, j):
        f = body.faces[k]
        n = mf._outward(body.sm, f, p)
        d = np.cross(n, t)
        d /= np.linalg.norm(d) or 1.0
        best = None
        for sgn in (1.0, -1.0):             # the side of the edge the face is on: the nearer foot
            r = f.closest_point(tuple(float(x) for x in p + sgn * h * d))
            if best is None or float(r[1]) < best[0]:
                best = (float(r[1]), mf._pt(r[0]), sgn)
        q = best[1]
        nq = mf._outward(body.sm, f, q)
        turn = nq - n
        ks.append(float(turn @ (best[2] * d)) / max(float(np.linalg.norm(q - p)), 1e-300))
    k = max(abs(ks[0]), abs(ks[1]), 1.0 / body.size)
    return abs(ks[0] - ks[1]) / k


def _surface(ext, net):
    H = net["H"]
    nu, nv = H.shape[:2]
    w = H[..., 3]
    P = H[..., :3] / w[..., None]
    rational = np.abs(w - 1.0).max() > 1e-12
    return ext.surface_from_cvs(int(net["pu"]), int(net["pv"]), [tuple(map(float, p)) for p in P.reshape(-1, 3)],
                                [float(x) for x in net["Uu"]], [float(x) for x in net["Uv"]],
                                [float(x) for x in w.reshape(-1)] if rational else None)


def _whole(body, i):
    """Is face i its whole surface (no trim of its own)?"""
    try:
        f = body.faces[i]
        ext = body.sm.blender_ext
        if "knots_u" not in ext.surface_cvs(f.surface()):
            return False                    # no copy of it to make a face from (a brep owns its surface)
        a = ext.brep_from_surface(mf._copy_surface(ext, f.surface())).area(1e-6)
        return abs(a - f.area(1e-6)) <= 1e-5 * max(a, 1e-12)
    except Exception:
        return False


def _merged(body, members, gone, tol):
    """The cluster as one face on a fitted surface: (face brep, deviation) or (None, why)."""
    sm, ext = body.sm, body.sm.blender_ext
    edges = body.b.edges()
    shared = {i: [edges[k] for k in gone if i in body.edge_faces(edges[k])] for i in members}
    cells = {}
    for i in members:
        c = _cell(body, i, shared[i])
        if c is None:
            return None, "a face crosses its surface's seam"
        cells[i] = c
    links = {tuple(sorted(body.edge_faces(edges[k]))) for k in gone}
    lay = layout(cells, links, max(100 * body.tol, 0.5 * tol))
    if lay is None:
        return None, "the faces' surfaces do not line up side to side (a row or grid of four-sided patches)"
    net, dev = fit_grid(*lay, tol)
    if net is None:
        return None, dev
    # the fitted surface faces the way the faces do
    f0 = body.faces[members[0]]
    q = mf._inner(f0)
    if q is not None:
        u, v, _ = _invert(net, [q])
        _, Su, Sv = _pts(net, u, v)
        if float(np.cross(Su[0], Sv[0]) @ mf._outward(sm, f0, q)) < 0:
            net = _orient(net, 0, 0, 1)
    surf = _surface(ext, net)
    area = sum(body.faces[i].area(1e-6) for i in members)
    if all(_whole(body, i) for i in members):
        try:
            m = ext.brep_from_surface(surf)
            if m.face_count() == 1:
                return m, dev
        except Exception:
            pass
    gone_keys = {mf._key(mf._edge_mid(edges[k]), body.tol) for k in gone}
    segs, curves = [], []
    for i in members:
        for loop in body.faces[i].boundary_loops():
            for e, fw in loop:
                if mf._key(mf._edge_mid(e), body.tol) in gone_keys:
                    continue
                segs.append(mf._seg(e, fw))
                curves.append(mf._ecurve(e, fw))
    chained = mf._chain_runs([[j] for j in range(len(segs))], segs, body.tol)
    if chained is None:
        return None, "the merged face's outline did not close"
    loops = [[curves[j] for j in lp] for lp in chained]
    t_trim = max(body.tol, 2.0 * dev)
    m = mf._trim(sm, surf, loops, None, t_trim, brep_tol(max(mf.REL_TOL, 2.0 * dev / body.size), body.size))
    if m is None or abs(m.area(1e-6) - area) > 1e-3 * area + 1e-9:
        return None, "the fitted surface could not be trimmed to the faces' outline"
    return m, dev


def refit(sm, b, go_pts, tol):
    """Merge the clusters joined by the edges at go_pts (points on edges
    between tangent faces) onto fitted surfaces within tol. Returns (brep or
    None, report): "removed" (edges), "deviation" (largest), "failed"
    [{point, why}], "ends" (the removed edges' ends)."""
    rep = {"removed": 0, "deviation": 0.0, "failed": [], "ends": []}
    if tol <= 0:
        return None, rep
    cur = b
    pending = [list(p) for p in go_pts]
    while pending:
        body = mf._Body(sm, cur)
        edges = cur.edges()
        hit, _ = mf._picked_edges(cur, edges, pending, body.size)
        go = [k for k in hit if tangent_angle(body, edges[k]) is not None]
        if not go:
            break
        clusters, gone = mf._clusters(body, go)
        if not clusters:
            break
        # one cluster per pass (the body changes; the rest are found again by their points)
        members = clusters[0]
        mine = {k for k in gone if body.edge_faces(edges[k])[0] in members}
        here = [p for k in hit if k in mine for p in (pending[x] for x in hit[k])]
        pending = [p for k in hit if k not in mine and k in go for p in (pending[x] for x in hit[k])]
        try:
            m, dev = _merged(body, members, mine, tol)
        except Exception as exc:          # a kernel call that throws: this cluster stays
            m, dev = None, f"the refit failed ({exc})"
        if m is None:
            rep["failed"].append({"point": here[0] if here else None, "why": dev})
            continue
        area = sum(body.faces[i].area(1e-6) for i in members)
        out = mf._replace(sm, cur, members, [m], max(body.tol, 2.0 * dev),
                          len(body.faces) - len(members) + 1, volume_change=2.0 * area * dev + 1e-9)
        if out is None:
            rep["failed"].append({"point": here[0] if here else None,
                                  "why": "the fitted face did not join up with the body"})
            continue
        for k in mine:
            t0, t1 = edges[k].parameter_range()
            rep["ends"] += [[float(x) for x in mf._pt(edges[k].point_at(t))] for t in (t0, t1)]
        rep["removed"] += len(mine)
        rep["deviation"] = max(rep["deviation"], dev)
        cur = out
    return (cur if cur is not b else None), rep
