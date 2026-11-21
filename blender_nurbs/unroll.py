"""Unroll Surface: lay a developable surface out flat (sheet metal patterns).

A developable surface (plane, cylinder, cone, an extruded or ruled sheet, a
bent strip) can be flattened without stretching: there is a map F from its
(u, v) domain into the plane that keeps every length. That map is found from
the surface's intrinsic geometry alone:

* the base iso line v = v* (a panel end near the middle of v) is laid out as
  a plane curve with the same arc length and the same geodesic curvature
  (its turning rate inside the surface, (S_uu . (N x S_u)) / |S_u|^2 per u);
* from each point of it, the cross iso line u = const leaves at the angle the
  surface has between S_u and S_v there, and is laid out the same way.

On a developable surface this is the isometry (up to the integration error,
about 1e-12 with 8 Gauss points per panel). Where the surface has a crease
(a knot of full multiplicity: a polyline extruded, a folded loft), the iso
line's flat direction jumps by the difference of the one-sided angles to the
crease line, so folded sheets unroll in one piece. On a surface that is not
developable the result depends on the order (its stretch is reported; Relax
spreads it out by least squares on lengths, like Rhino's Smash).

The flat sheet is a B-spline over the SAME (u, v) domain as the face's
surface (fitted to the map within tolerance, surfmap.fit_grid), so a trimmed
face keeps its trims (ext.replace_face_surfaces_keep_uv) and Flow Along
Surface can fold anything drawn on the sheet back onto the 3D face exactly.

Joined surfaces (polysurfaces, sheet-metal parts) unroll face by face; each
face is turned and moved in the plane to meet a face already laid out along
their shared edge (a tree over the faces, breadth first from the largest),
unless it would overlap the layout: then it starts a new piece. The faces of
each piece are stitched, so the bend lines stay as edges of one flat sheet.
"""

import json

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty

from . import features, geom, store, surfeval, surfmap
from .surfmap import Axis

OP = "unroll"
PROP_UNROLL = "nproto_unroll"      # JSON on a flat sheet: {"source", "faces": [source face index per face], ...}
DEVELOPABLE_K = 1e-6              # |Gaussian curvature| x size^2 below this: developable (an exact unroll)
FIT_TOL = 1e-6                    # flat sheet within this share of the face's size of the map
LAST_INFO = {}                    # the last run, for the report and the panel

PLACE_ITEMS = [("BESIDE", "Beside", "Flat on the ground (the object's lowest Z), beside the object"),
               ("TOUCHING", "Touching", "In the tangent plane at the middle of the surface, touching it there")]


# --------------------------------------------------------------------------
# The isometric map of one surface (pure numpy)
# --------------------------------------------------------------------------

def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def _angle(a, b):
    """Angle from a to b in (0, pi) (inside the surface: the normal is a x b)."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), _dot(a, b))


def _turn(D2, D1, N):
    """Turning rate of an iso line inside the surface per parameter:
    (c'' . (N x c')) / |c'|^2 (its geodesic curvature times its speed)."""
    return _dot(D2, np.cross(N, D1)) / np.maximum(_dot(D1, D1), 1e-300)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def axes_for(S, budget=200000):
    """Integration panels along u and v (knots always panel ends), at most
    about `budget` grid nodes."""
    (u0, u1), (v0, v1) = S.u_domain, S.v_domain
    def want(U, p):
        straight = p == 1 and len(np.unique(U)) == 2
        return 4 if straight else 24
    nu, nv = want(S.U, S.p), want(S.V, S.q)
    for _ in range(6):
        au = Axis(surfmap.breakpoints(S.U, u0, u1, nu))
        av = Axis(surfmap.breakpoints(S.V, v0, v1, nv))
        if au.n * av.n <= budget:
            break
        nu, nv = max(4, nu // 2), max(4, nv // 2)
    return au, av


def develop(S, au, av):
    """F (au.n, av.n, 2): the flat position of every grid node (au.t x av.t)
    and the base (u*, v*) it was laid out from (F there is 0, S_u along +x)."""
    (u0, u1), (v0, v1) = S.u_domain, S.v_domain
    epu, epv = 1e-9 * (u1 - u0), 1e-9 * (v1 - v0)

    # the base row: a v panel end near the middle where S_u does not vanish
    order = sorted(range(1, av.P), key=lambda j: abs(j - av.P / 2.0)) or [0]
    for js in order:
        vs = av.b[js]
        D = S.grid(au.gauss, [vs], 2)
        Su = D[1, 0][:, 0]
        if np.linalg.norm(Su, axis=1).min() > 1e-12 * max(1.0, np.linalg.norm(Su, axis=1).max()):
            break
    Suu, Sv = D[2, 0][:, 0], D[0, 1][:, 0]
    N = _unit(np.cross(Su, Sv))
    fu = _turn(Suu, Su, N)
    spu = np.linalg.norm(Su, axis=1)
    # jumps of the base line's flat direction at creases (panel ends)
    R = S.grid(au.b, [vs], 1)
    L = S.grid(np.maximum(au.b - epu, u0), [vs], 1)
    jump_u = _angle(L[1, 0][:, 0], L[0, 1][:, 0]) - _angle(R[1, 0][:, 0], R[0, 1][:, 0])
    jump_u[0] = jump_u[-1] = 0.0
    i0 = au.node_of(au.b[au.P // 2])
    Iu = au.cumulative(fu)
    Ju = au.excl_sum(jump_u)
    phi_u = Iu - Iu[i0] + Ju - Ju[i0]                     # left limit 0 at u*
    vel = spu[:, None] * np.stack([np.cos(phi_u[au.gi].ravel()), np.sin(phi_u[au.gi].ravel())], axis=1)
    a = np.stack([au.cumulative(vel[:, 0]), au.cumulative(vel[:, 1])], axis=1)
    a -= a[i0]

    # the cross lines from every u node (panel ends: the left side, as phi_u there)
    ue = au.t.copy()
    ue[au.bi[1:]] = np.maximum(ue[au.bi[1:]] - epu, u0)
    D = S.grid(ue, av.gauss, 2)
    Sv, Svv, Su = D[0, 1], D[0, 2], D[1, 0]
    N = _unit(np.cross(Su, Sv))
    fv = _turn(Svv, Sv, N)                                # (nu, Pv*K)
    spv = np.linalg.norm(Sv, axis=2)
    Rb = S.grid(ue, av.b, 1)
    Lb = S.grid(ue, np.maximum(av.b - epv, v0), 1)
    bR = _angle(Rb[1, 0], Rb[0, 1])                       # (nu, Pv+1)
    bL = _angle(Lb[1, 0], Lb[0, 1])
    jump_v = bR - bL
    jump_v[:, 0] = jump_v[:, -1] = 0.0
    j0 = av.node_of(vs)
    phi0 = phi_u + bL[:, js]                              # the left limit at v*
    Iv = av.cumulative(fv)
    Jv = av.excl_sum(jump_v)
    phi_v = phi0[:, None] + Iv - Iv[:, j0:j0 + 1] + Jv - Jv[:, j0:j0 + 1]
    pg = phi_v[:, av.gi.ravel()]
    F = np.empty((au.n, av.n, 2))
    for c, fn in ((0, np.cos), (1, np.sin)):
        X = av.cumulative(spv * fn(pg))
        F[:, :, c] = a[:, c:c + 1] + X - X[:, j0:j0 + 1]
    return F, (au.b[au.P // 2], vs)


def strain(S, C, U, p, V, q, au, av):
    """(largest, rms) |stretch - 1| of the flat net C (2D or 3D) against S at
    the Gauss grid (principal stretches of the map), skipping degenerate points."""
    tu, tv = au.gauss[::2], av.gauss[::2]
    D = S.grid(tu, tv, 1)
    Bu = surfeval._basis_matrix(np.asarray(U, float), p, len(U) - p - 1, tu, 1)
    Bv = surfeval._basis_matrix(np.asarray(V, float), q, len(V) - q - 1, tv, 1)
    Fu = np.einsum("ia,abd,jb->ijd", Bu[1], C, Bv[0])
    Fv = np.einsum("ia,abd,jb->ijd", Bu[0], C, Bv[1])
    Gs = np.stack([np.stack([_dot(D[1, 0], D[1, 0]), _dot(D[1, 0], D[0, 1])], -1),
                   np.stack([_dot(D[1, 0], D[0, 1]), _dot(D[0, 1], D[0, 1])], -1)], -2)
    Gf = np.stack([np.stack([_dot(Fu, Fu), _dot(Fu, Fv)], -1),
                   np.stack([_dot(Fu, Fv), _dot(Fv, Fv)], -1)], -2)
    det = np.linalg.det(Gs)
    good = det > 1e-10 * np.maximum(np.trace(Gs, axis1=-2, axis2=-1) ** 2, 1e-300)
    if not good.any():
        return 0.0, 0.0
    M = np.linalg.solve(Gs[good], Gf[good])
    ev = np.linalg.eigvals(M).real
    e = np.abs(np.sqrt(np.maximum(ev, 0.0)) - 1.0)
    return float(e.max()), float(np.sqrt(np.mean(e * e)))


def flat_net(surf, tol_rel=FIT_TOL, relax=False):
    """(C (nu, nv, 2), U, p, V, q, info) of the flat map of a kernel surface:
    the net of a B-spline over the surface's own domain, within tol_rel of
    its size of the isometric map. info: "strain" / "rms" (the largest and
    the rms stretch), "fit" (its miss / size), "developable", "relaxed",
    "base" (the (u, v) laid out at the origin, S_u along +x), "size"."""
    S = surfeval.Surf(surf)
    au, av = axes_for(S)
    F, base = develop(S, au, av)
    P = S.grid(au.t, av.t, 0)[0, 0]
    size = float(np.linalg.norm(P.reshape(-1, 3).max(axis=0) - P.reshape(-1, 3).min(axis=0))) or 1.0
    tol = tol_rel * size
    relaxed = False
    p, q = surfmap.degree_for(S.p, True), surfmap.degree_for(S.q, True)
    U0 = surfmap.fit_knots(S.U, S.p, p, *S.u_domain)
    V0 = surfmap.fit_knots(S.V, S.q, q, *S.v_domain)
    mu, mv = 0.5 * float(au.h.min()), 0.5 * float(av.h.min())
    C, U, V, err = surfmap.fit_grid(au.t, av.t, F, U0, p, V0, q, tol, min_u=mu, min_v=mv)
    st, rms = strain(S, C, U, p, V, q, au, av)
    kmax = gauss_curvature(S, au, av) * size * size
    developable = kmax < DEVELOPABLE_K
    if relax and not developable:
        F2 = relax_map(P, F)
        p, q = max(p, 3), max(q, 3)
        U0 = surfmap.fit_knots(S.U, S.p, p, *S.u_domain)
        V0 = surfmap.fit_knots(S.V, S.q, q, *S.v_domain)
        C2, U2, V2, err2 = surfmap.fit_grid(au.t, av.t, F2, U0, p, V0, q, tol, min_u=mu, min_v=mv)
        st2, rms2 = strain(S, C2, U2, p, V2, q, au, av)
        if rms2 < rms:
            C, U, V, err, st, rms, relaxed = C2, U2, V2, err2, st2, rms2, True
    return C, U, p, V, q, {"strain": st, "rms": rms, "fit": err / size, "relaxed": relaxed, "base": base,
                           "size": size, "developable": developable, "curvature": kmax}


def gauss_curvature(S, au, av):
    """Largest |Gaussian curvature| at the Gauss grid (0 on a developable surface)."""
    D = S.grid(au.gauss[::2], av.gauss[::2], 2)
    K = surfeval.curvatures(D)[0]
    return float(np.abs(K).max(initial=0.0))


def relax_map(P, F, iters=40):
    """F moved so the grid's edge and diagonal lengths come as near those of
    the 3D grid P as the plane allows: Gauss-Newton on the relative length
    misfits of a coarse subgrid (sparse normal equations, the rigid motions
    pinned), the fine grid following its moves bilinearly."""
    _, ext = store._k()
    nu, nv = F.shape[:2]
    iu = np.unique(np.r_[np.arange(0, nu, max(1, nu // 48)), nu - 1])
    iv = np.unique(np.r_[np.arange(0, nv, max(1, nv // 48)), nv - 1])
    Pc, F0 = P[np.ix_(iu, iv)].reshape(-1, 3), F[np.ix_(iu, iv)]
    m, n = len(iu), len(iv)
    idx = np.arange(m * n).reshape(m, n)
    pairs = [(idx[:-1, :], idx[1:, :]), (idx[:, :-1], idx[:, 1:]),
             (idx[:-1, :-1], idx[1:, 1:]), (idx[1:, :-1], idx[:-1, 1:])]
    A = np.concatenate([a.ravel() for a, _ in pairs])
    B = np.concatenate([b.ravel() for _, b in pairs])
    L = np.linalg.norm(Pc[A] - Pc[B], axis=1)
    keep = L > 1e-9 * max(float(L.max(initial=0.0)), 1e-300)
    A, B, L = A[keep], B[keep], L[keep]
    if not len(A):
        return F
    x = F0.reshape(-1, 2).copy()
    nvar = 2 * m * n
    k2 = int(idx[min(1, m - 1), 0] if m > 1 else idx[0, min(1, n - 1)])
    cr = np.arange(3, dtype=np.int64)
    cc = np.array([0, 1, 2 * k2 + 1], np.int64)
    for _ in range(iters):
        d = x[A] - x[B]
        cur = np.maximum(np.linalg.norm(d, axis=1), 1e-300)
        r = cur / L - 1.0
        g = d / (cur * L)[:, None]
        ec = np.stack([2 * A, 2 * A + 1, 2 * B, 2 * B + 1], 1)
        ev = np.concatenate([g, -g], 1)
        Jtr = np.zeros(nvar)
        np.add.at(Jtr, ec.ravel(), (ev * r[:, None]).ravel())
        Mr = np.concatenate([np.repeat(ec, 4, axis=1).ravel(), np.arange(nvar)])
        Mc = np.concatenate([np.tile(ec, (1, 4)).ravel(), np.arange(nvar)])
        Mv = np.concatenate([(ev[:, :, None] * ev[:, None, :]).ravel(), np.full(nvar, 1e-10)])
        dx = ext.sparse_kkt(Mr.astype(np.int64), Mc.astype(np.int64), Mv, nvar, cr, cc, np.ones(3), 3,
                            -Jtr, np.zeros(3))
        dx = np.asarray(dx).reshape(-1, 2)
        x += dx
        if np.abs(dx).max() < 1e-10 * float(L.max()):
            break
    Dc = x.reshape(m, n, 2) - F0
    fu = np.interp(np.arange(nu), iu, np.arange(m))
    fv = np.interp(np.arange(nv), iv, np.arange(n))
    i0 = np.clip(np.floor(fu).astype(int), 0, max(m - 2, 0))
    j0 = np.clip(np.floor(fv).astype(int), 0, max(n - 2, 0))
    i1, j1 = np.minimum(i0 + 1, m - 1), np.minimum(j0 + 1, n - 1)
    a = (fu - i0)[:, None, None]
    b = (fv - j0)[None, :, None]
    return F + ((1 - a) * (1 - b) * Dc[i0][:, j0] + a * (1 - b) * Dc[i1][:, j0]
                + (1 - a) * b * Dc[i0][:, j1] + a * b * Dc[i1][:, j1])


# --------------------------------------------------------------------------
# Faces of a B-rep laid out in the plane
# --------------------------------------------------------------------------

class Flat:
    """One face and its flat map: the net C (nu, nv, 2) over the face
    surface's domain (mirrored so the face's outward side is +Z), and its
    placement in the layout (x -> R x + t)."""

    def __init__(self, index, face, relax, tol_rel):
        self.index = index
        self.face = face
        surf = face.surface()
        try:
            self.S = surfeval.Surf(surf)
        except Exception:
            raise ValueError(f"face {index} is not a B-spline surface")
        self.C, self.U, self.p, self.V, self.q, self.info = flat_net(surf, tol_rel, relax)
        self.uv_in = inside_uvs(face, self.S)
        u, v = self.uv_in[len(self.uv_in) // 2] if len(self.uv_in) else (
            0.5 * sum(self.S.u_domain), 0.5 * sum(self.S.v_domain))
        n = np.array(face.normal_at_uv(float(u), float(v)))
        o = np.array(face.outward_normal_at_uv(float(u), float(v)))
        self.mirror = float(n @ o) < 0.0
        if self.mirror:
            self.C = self.C.copy()
            self.C[..., 1] *= -1.0
        self.R, self.t = np.eye(2), np.zeros(2)
        self.area = float(face.area()) if hasattr(face, "area") else 1.0
        self._flat_surf = None
        self.island = -1

    def flat(self, uv):
        """Flat (unplaced) positions of the face at (u, v) (m, 2)."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        Bu, Bv = surfmap.basis(self.U, self.p, uv[:, 0]), surfmap.basis(self.V, self.q, uv[:, 1])
        return np.einsum("ma,abd,mb->md", Bu, self.C, Bv)

    def placed(self, uv):
        return self.flat(uv) @ self.R.T + self.t

    def flat_surf(self):
        """The unplaced flat net as a numpy surface (z = 0), for feet in the plane."""
        if self._flat_surf is None:
            C3 = np.concatenate([self.C, np.zeros(self.C.shape[:2] + (1,))], axis=2)
            self._flat_surf = surfeval.Surf(surfmap.surface_from_net(C3, self.U, self.p, self.V, self.q))
        return self._flat_surf

    def contains(self, X, tol):
        """Which placed 2D points X (m, 2) lie inside the face (not on its edge)."""
        X = np.asarray(X, float).reshape(-1, 2)
        if not len(X):
            return np.zeros(0, bool)
        L = (X - self.t) @ self.R                   # back into the unplaced flat
        Fs = self.flat_surf()
        uv = Fs.feet(np.c_[L, np.zeros(len(L))])
        Q = Fs.at(uv[:, 0], uv[:, 1], 0)[0, 0][:, :2]
        out = np.linalg.norm(Q - L, axis=1) <= tol
        for k in np.nonzero(out)[0]:
            out[k] = self.face.classify_uv(float(uv[k, 0]), float(uv[k, 1])) == "inside"
        return out

    def box(self):
        P = self.placed(self.uv_in) if len(self.uv_in) else np.zeros((1, 2)) + self.t
        return P.min(axis=0), P.max(axis=0)


def inside_uvs(face, S, n=17):
    """(m, 2) parameters of a grid over the face's trim box that lie inside it."""
    (a, b), (c, d) = face.uv_domain()
    out = []
    for u in np.linspace(a, c, n + 2)[1:-1]:
        for v in np.linspace(b, d, n + 2)[1:-1]:
            if face.classify_uv(float(u), float(v)) == "inside":
                out.append((u, v))
    return np.array(out, float).reshape(-1, 2)


def _rigid(src, dst):
    """(R, t): the turn and move (no mirror) taking 2D points src nearest dst."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    Uh, _, Vt = np.linalg.svd(H)
    R = Vt.T @ Uh.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ Uh.T
    return R, cd - R @ cs


def shared_edges(brep, faces):
    """[(i, j, points)] of the edges between two different faces."""
    out = []
    for e in brep.edges():
        fs = e.faces()
        if len(fs) != 2:
            continue
        ij = []
        for f in fs:
            for k, g in enumerate(faces):
                if f == g:
                    ij.append(k)
                    break
        if len(ij) == 2 and ij[0] != ij[1]:
            out.append((ij[0], ij[1], surfmap.edge_samples(e, 9)))
    return out


def layout(brep, relax=False, tol_rel=FIT_TOL):
    """The faces of a (world space) brep unrolled and laid out: (flats,
    islands). Each island is a list of face indices whose flats were turned
    and moved to meet along shared edges without overlapping."""
    faces = list(brep.faces())
    flats = [Flat(k, f, relax, tol_rel) for k, f in enumerate(faces)]
    size = max(fl.info["size"] for fl in flats)
    tol = 1e-6 * size
    links = {}
    for i, j, P in shared_edges(brep, faces):
        if len(P) < 2 or np.linalg.norm(P[-1] - P[0]) + np.linalg.norm(P[len(P) // 2] - P[0]) <= tol:
            continue
        uv_i = flats[i].S.feet(P)
        uv_j = flats[j].S.feet(P)
        a, b = flats[i].flat(uv_i), flats[j].flat(uv_j)
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        links.setdefault(i, []).append((L, j, a, b))
        links.setdefault(j, []).append((L, i, b, a))
    for v in links.values():
        v.sort(key=lambda x: -x[0])
    placed, islands = set(), []
    order = sorted(range(len(flats)), key=lambda k: -flats[k].area)
    for root in order:
        if root in placed:
            continue
        isl = [root]
        placed.add(root)
        flats[root].island = len(islands)
        queue = [root]
        while queue:
            i = queue.pop(0)
            for _, j, a, b in links.get(i, []):
                if j in placed:
                    continue
                dst = a @ flats[i].R.T + flats[i].t
                R, t = _rigid(b, dst)
                if float(np.linalg.norm(b @ R.T + t - dst, axis=1).max()) > 1e-4 * size:
                    continue                      # the edge is not the same curve flat on both sides
                flats[j].R, flats[j].t = R, t
                if _overlaps(flats[j], [flats[k] for k in isl], tol, size):
                    flats[j].R, flats[j].t = np.eye(2), np.zeros(2)
                    continue
                placed.add(j)
                flats[j].island = len(islands)
                isl.append(j)
                queue.append(j)
        islands.append(isl)
    return flats, islands


def _overlaps(new, others, tol, size):
    lo, hi = new.box()
    X = new.placed(new.uv_in)
    for o in others:
        olo, ohi = o.box()
        if np.any(hi < olo - 1e-3 * size) or np.any(ohi < lo - 1e-3 * size):
            continue
        if new.contains(o.placed(o.uv_in), tol).any() or o.contains(X, tol).any():
            return True
    return False


def world_frames(flats, islands, src_box, place, gap_rel=0.08):
    """A 3x4 matrix per island: its layout (x, y) -> world (x, y, 0, 1)."""
    lo, hi = np.asarray(src_box[0], float), np.asarray(src_box[1], float)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    gap = gap_rel * size
    boxes = []
    for isl in islands:
        P = np.vstack([flats[k].placed(flats[k].uv_in) for k in isl if len(flats[k].uv_in)] or [np.zeros((1, 2))])
        Q = np.vstack([flats[k].placed(_grid_uv(flats[k])) for k in isl])
        P = np.vstack([P, Q])
        boxes.append((P.min(axis=0), P.max(axis=0)))
    out = []
    if place == "TOUCHING":
        f0 = flats[islands[0][0]]
        u, v = f0.info["base"]
        D = f0.S.at([u], [v], 1)
        O, Su, Sv = D[0, 0][0], D[1, 0][0], D[0, 1][0]
        ex = Su / (np.linalg.norm(Su) or 1.0)
        n = np.array(f0.face.outward_normal_at_uv(float(u), float(v)))
        ey = np.cross(n, ex)
        ey /= np.linalg.norm(ey) or 1.0
        x = 0.0
        for k, (blo, bhi) in enumerate(boxes):
            M = np.zeros((3, 3))
            M[:, 0], M[:, 1] = ex, ey
            shift = 0.0 if k == 0 else x - blo[0]
            M[:, 2] = O + shift * ex
            out.append(M)
            x = (boxes[0][1][0] if k == 0 else x + bhi[0] - blo[0]) + gap
        return out
    x = hi[0] + gap
    for blo, bhi in boxes:
        M = np.zeros((3, 3))
        M[0, 0], M[1, 1] = 1.0, 1.0
        M[:, 2] = (x - blo[0], lo[1] - blo[1], lo[2])
        out.append(M)
        x += (bhi[0] - blo[0]) + gap
    return out


def _grid_uv(fl, n=9):
    (u0, u1), (v0, v1) = fl.S.u_domain, fl.S.v_domain
    uu, vv = np.meshgrid(np.linspace(u0, u1, n), np.linspace(v0, v1, n), indexing="ij")
    uv = np.c_[uu.ravel(), vv.ravel()]
    keep = [fl.face.classify_uv(float(a), float(b)) != "outside" for a, b in uv]
    return uv[np.array(keep, bool)] if any(keep) else uv[:1]


def placed_surface(fl, M):
    """The face's flat surface in world space: its net placed (R, t) then
    taken into the world by the island's frame M (3x3: x, y, origin)."""
    C3 = to_world(fl.C @ fl.R.T + fl.t, M)
    return surfmap.surface_from_net(C3, fl.U, fl.p, fl.V, fl.q), C3


def to_world(X2, M):
    """Layout points (..., 2) into the world by an island frame M."""
    X2 = np.asarray(X2, float)
    return X2[..., :1] * M[:, 0] + X2[..., 1:2] * M[:, 1] + M[:, 2]


def trimmed_flat(fl, M, size):
    """The face laid flat with its trims: its flat surface trimmed by its
    edges mapped into the plane (one curve per edge, so edges shared with
    other flat faces meet edge to edge)."""
    return surfmap.trimmed_face(fl.face, fl.S, lambda: placed_surface(fl, M)[0],
                                lambda uv: to_world(fl.placed(uv), M), size)


def build_islands(flats, islands, frames):
    """[(Brep, [source face index per face])]: each island's flat faces,
    trims kept, stitched where they meet."""
    sm, ext = store._k()
    size = max(fl.info["size"] for fl in flats)
    out = []
    for isl, M in zip(islands, frames):
        parts, nets = [], []
        for k in isl:
            fl = flats[k]
            surf, C3 = placed_surface(fl, M)
            nets.append((k, C3))
            parts.append(ext.brep_from_surface(surf) if surfmap.is_natural_face(fl.face, fl.S)
                         else trimmed_flat(fl, M, size))
        b = parts[0]
        for o in parts[1:]:
            ext.append_brep(b, o)
        if len(parts) > 1:
            b = sm.stitch_advanced(b, 1e-5 * size, squeeze_small_edges=False, making_manifold_solid=False)[0]
            up = [f for f, (k, _) in zip(b.faces(), nets) if not flats[k].mirror]
            try:
                sm.unify_normals(b, max(up or list(b.faces()), key=lambda f: f.area()), 100)
            except Exception:
                pass
        out.append((b, _face_sources(b, nets, size)))
    return out


def _face_sources(b, nets, size):
    """[source face index, flip] of every face of b, matched by its net:
    flip 0 the same (u, v), 1 u reversed, 2 v reversed, 3 u and v swapped
    (Unify Normals turns a face over by changing its surface: it swaps u and
    v); [-1, 0] when nothing matches."""
    _, ext = store._k()
    out = []
    for f in b.faces():
        try:
            d = ext.surface_cvs(f.surface())
            P = np.asarray(d["points"], float).reshape(int(d["count_u"]), int(d["count_v"]), 3)
        except Exception:
            out.append([-1, 0])
            continue
        best, bd = [-1, 0], None
        for k, C3 in nets:
            for flip, C in ((0, C3), (1, C3[::-1]), (2, C3[:, ::-1]), (3, C3.transpose(1, 0, 2))):
                if C.shape != P.shape:
                    continue
                dd = float(np.abs(C - P).max())
                if bd is None or dd < bd:
                    best, bd = [k, flip], dd
        out.append(best if bd is not None and bd <= 1e-6 * size else [-1, 0])
    return out


def unroll_brep(brep_world, place="BESIDE", relax=False):
    """(pieces, flats, islands): the flat sheets of a world-space brep, each
    (Brep, source face per face), placed by `place`."""
    flats, islands = layout(brep_world, relax)
    lo, hi = _box(brep_world)
    frames = world_frames(flats, islands, (lo, hi), place)
    return build_islands(flats, islands, frames), flats, islands


def _box(brep):
    bb = brep.bounding_box()
    try:
        lo, hi = bb
        return np.array(lo, float), np.array(hi, float)
    except Exception:
        return np.array(bb.min, float), np.array(bb.max, float)


# --------------------------------------------------------------------------
# Curves on the surface come along
# --------------------------------------------------------------------------

def curve_map(flats, islands, frames):
    """A flow.FlowMap from the 3D faces to their placed flat faces."""
    from . import flow
    frame_of = {k: M for isl, M in zip(islands, frames) for k in isl}
    pairs = []
    for fl in flats:
        St = surfeval.Surf(placed_surface(fl, frame_of[fl.index])[0])
        pairs.append(flow.Pair(fl.face, fl.S, St, flow.same_uvmap(0, fl.S), sigma=0.0))
    return flow.FlowMap(pairs, max(fl.info["size"] for fl in flats))


def lies_on(obj, flats, size, rel=2e-3):
    """Whether a curve object lies on the faces (every sample within rel x
    size: projected curves miss by about 1e-3)."""
    from .flow import _points_of
    P = _points_of(obj)
    if not len(P):
        return False
    best = np.full(len(P), np.inf)
    for fl in flats:
        uv = fl.S.feet(P)
        best = np.minimum(best, np.linalg.norm(P - fl.S.at(uv[:, 0], uv[:, 1], 0)[0, 0], axis=1))
    return bool(best.max() <= rel * size)


def unrolled_curve_specs(obj, cmap, size):
    """World spline specs of a curve on the surface, laid out with it; pieces
    that cross a cut between faces (a jump in the layout) come apart."""
    from . import curvejoin, flow
    M = np.array(geom.world_matrix(obj), float)
    out = []
    for sp in obj.data.splines:
        try:
            spans = curvejoin.transform_spans(curvejoin.spline_spans(sp), M)
        except Exception:
            continue
        out += flow.flow_spans(spans, bool(sp.use_cyclic_u), cmap, size, flow.CURVE_TOL * size)
    return out


# --------------------------------------------------------------------------
# History and the operator
# --------------------------------------------------------------------------

def _summary(flats):
    dev = all(fl.info["developable"] for fl in flats)
    st = max(fl.info["strain"] for fl in flats)
    return {"developable": dev, "strain": st, "relaxed": any(fl.info["relaxed"] for fl in flats)}


def unroll_object(obj, place, relax):
    b = store.get_brep(obj).copy()
    store.transform(b, np.array(geom.world_matrix(obj), float))
    return unroll_brep(b, place, relax)


@features.builder(OP)
def build_unroll(inputs, p):
    """inputs: (the surface, "SURF"); params ur_place, ur_relax, island. The
    island-th flat sheet in world space."""
    src = inputs[0][0]
    pieces, flats, islands = unroll_object(src, p.get("ur_place", "BESIDE"), bool(p.get("ur_relax", True)))
    k = int(p.get("island", 0))
    if k >= len(pieces):
        raise RuntimeError(f"the unroll now has {len(pieces)} piece(s), not {k + 1}")
    b, faces = pieces[k]
    LAST_INFO.clear()
    LAST_INFO.update(_summary(flats), source=src.name, faces=faces, pieces=len(pieces))
    return ("brep_text", store.brep_to_text(b))


def store_info(obj):
    """After a rebuild: the face table Flow folds back with."""
    if LAST_INFO.get("faces") is not None:
        obj[PROP_UNROLL] = json.dumps({"source": LAST_INFO["source"], "faces": LAST_INFO["faces"]})


class NPROTO_OT_unroll(bpy.types.Operator):
    """Lay developable surfaces out flat (cones, cylinders, bent and folded
    sheets: sheet-metal patterns). Joined faces stay joined along their bend
    lines where they don't overlap; curves on the surface come along. Flow
    Along Surface folds what you draw on the sheet back onto the surface"""
    bl_idname = "nproto.unroll"
    bl_label = "Unroll Surface"
    bl_options = {'REGISTER', 'UNDO'}

    place: EnumProperty(name="Place", items=PLACE_ITEMS, default="BESIDE")
    relax: BoolProperty(name="Relax", default=True,
                        description="A surface that is not developable is flattened with the least stretch "
                                    "(an approximate pattern) instead of laid out from its middle")
    curves: BoolProperty(name="Curves Along", default=True,
                         description="Selected curves lying on the surface are unrolled with it")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        from .ops import _kernel_ok, _select_only
        if not _kernel_ok(self):
            return {'CANCELLED'}
        srcs = [o for o in context.selected_objects if store.is_nurbs(o)]
        curves = [o for o in context.selected_objects if o.type == 'CURVE'] if self.curves else []
        if not srcs:
            self.report({'WARNING'}, "Select NURBS surfaces to unroll")
            return {'CANCELLED'}
        hist = context.scene.nproto_settings.construction_history
        params = {"ur_place": self.place, "ur_relax": self.relax}
        made, notes, bad = [], [], []
        for src in srcs:
            try:
                pieces, flats, islands = unroll_object(src, self.place, self.relax)
            except Exception as exc:
                bad.append(f"{src.name}: {exc}")
                continue
            for k, (b, faces) in enumerate(pieces):
                name = src.name + "_flat" + (f".{k + 1}" if len(pieces) > 1 else "")
                obj = store.new_object_text(context, name, store.brep_to_text(b))
                obj[PROP_UNROLL] = json.dumps({"source": src.name, "faces": faces})
                if hist:
                    features.record(obj, OP, [(src, "SURF")], dict(params, island=k))
                made.append(obj)
            size = max(fl.info["size"] for fl in flats)
            on = [c for c in curves if lies_on(c, flats, size)]
            if on:
                from . import flow
                frames = world_frames(flats, islands, _box(_world(src)), self.place)
                cmap = curve_map(flats, islands, frames)
                for c in on:
                    specs = unrolled_curve_specs(c, cmap, size)
                    if specs:
                        made.append(flow.new_curve_object(context, c, c.name + "_flat", specs))
            s = _summary(flats)
            if s["developable"]:
                notes.append(f"{src.name}: {len(pieces)} piece{'s' if len(pieces) != 1 else ''}, exact")
            else:
                notes.append(f"{src.name}: not developable, up to {100 * s['strain']:.2g} % stretch"
                             + (" (relaxed)" if s["relaxed"] else ""))
        if not made:
            self.report({'ERROR'}, "Unroll failed: " + "; ".join(bad))
            return {'CANCELLED'}
        _select_only(context, made)
        msg = "Unrolled " + "; ".join(notes) + ("" if not bad else " -- failed: " + "; ".join(bad))
        exact = all("exact" in n for n in notes)
        self.report({'INFO'} if exact and not bad else {'WARNING'}, msg)
        return {'FINISHED'}


def _world(obj):
    b = store.get_brep(obj).copy()
    return store.transform(b, np.array(geom.world_matrix(obj), float))


classes = (NPROTO_OT_unroll,)
