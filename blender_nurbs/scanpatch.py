"""Drape Patch to Scan: one NURBS patch laid straight down onto a scan.

The patch's control net is a quad grid the user lays out above the scan (a
6 x 6 grid = one span of degree 5 each way: three rows of CVs from every
edge, for G0 / G1 / G2 to a neighbour later, and no more). Every CV may only
move along one direction (world Z by default), so the patch keeps its plan
view: its footprint, its CV columns and its parametrisation stay as laid out.

That makes the fit linear and done in one go: every surface point is a
fixed mix of CVs, so its position across the direction is fixed too, and the
scan's height under it is found once, by a ray cast straight down. Then

    minimise   sum_k w_k (n_k . up)^2 (S_k t - h_k)^2  +  smoothness * fairness(t)

over the CV heights t, where S_k are the patch's basis weights at sample k,
h_k the scan's height there (plus the clearance), and (n . up)^2 turns the
height difference into a distance along the scan's normal. Robust weights
(re-solved a few times) let go of panel gaps and trim strips; Drape makes it
one-sided, like a sheet: the patch is pushed up where it dips into the scan
and only lightly pulled down where it stands above it, so it bridges gaps."""

import math
import time

import numpy as np

from . import geom


# ---------------------------------------------------------------------------
# the control grid
# ---------------------------------------------------------------------------


def grid_order(n_verts, faces):
    """The vertices of a quad grid mesh as an (nu, nv) array of indices, row 0
    along one border from a corner. Raises ValueError for anything else."""
    if not faces or any(len(f) != 4 for f in faces):
        raise ValueError("the sheet must be a grid of quads (a 6 x 6 vertex grid for one degree-5 patch)")
    nb = [set() for _ in range(n_verts)]
    nf = np.zeros(n_verts, int)
    for f in faces:
        for i in range(4):
            a, b = int(f[i]), int(f[(i + 1) % 4])
            nb[a].add(b)
            nb[b].add(a)
            nf[a] += 1
    used = np.nonzero(nf)[0]
    corners = [v for v in used if nf[v] == 1]
    if len(corners) != 4:
        raise ValueError("the sheet must be one rectangular grid of quads (4 corners)")
    c0 = corners[0]
    border = [v for v in used if nf[v] <= 2]
    bset = set(border)

    def walk(a, b):
        """Along the border from corner a through b to the next corner."""
        line = [a, b]
        while nf[line[-1]] != 1:
            nxt = [y for y in nb[line[-1]] if y in bset and y != line[-2] and nf[y] <= 2]
            nxt = [y for y in nxt if y not in line]
            if not nxt:
                raise ValueError("the sheet's border is not a clean grid border")
            line.append(nxt[0])
        return line

    d1, d2 = [y for y in nb[c0] if y in bset]
    row0 = walk(c0, d1)
    col0 = walk(c0, d2)
    nu, nv = len(col0), len(row0)
    if nu * nv != len(used) or len(faces) != (nu - 1) * (nv - 1):
        raise ValueError("the sheet must be one rectangular grid of quads")
    G = np.full((nu, nv), -1, np.int64)
    G[0] = row0
    for i in range(1, nu):
        G[i, 0] = col0[i]
        for j in range(1, nv):
            # the vertex next to both G[i, j-1] and G[i-1, j] that is not G[i-1, j-1]
            cand = (nb[G[i, j - 1]] & nb[G[i - 1, j]]) - {int(G[i - 1, j - 1])}
            if len(cand) != 1:
                raise ValueError("the sheet must be one rectangular grid of quads")
            G[i, j] = cand.pop()
    if len(set(G.ravel().tolist())) != nu * nv:
        raise ValueError("the sheet must be one rectangular grid of quads")
    return G


def degree_and_knots(n, max_degree=5):
    """The degree (n - 1 up to max_degree: one span when it fits) and clamped uniform knots for n CVs."""
    p = max(1, min(int(max_degree), n - 1))
    return p, geom.clamped_uniform_knots(n, p)


def basis(n, p, U, t):
    return np.asarray(geom.interp_matrix(np.asarray(t, float), U, p, n), float)


def _fair_rows(nu, nv):
    """Second differences of a value per CV over the net (along u, along v, and the twist): rows over nu * nv."""
    rows = []
    idx = np.arange(nu * nv).reshape(nu, nv)
    for i in range(nu - 2):
        for j in range(nv):
            rows.append(({idx[i, j]: 1.0, idx[i + 1, j]: -2.0, idx[i + 2, j]: 1.0}))
    for i in range(nu):
        for j in range(nv - 2):
            rows.append(({idx[i, j]: 1.0, idx[i, j + 1]: -2.0, idx[i, j + 2]: 1.0}))
    for i in range(nu - 1):
        for j in range(nv - 1):
            rows.append(({idx[i, j]: 1.0, idx[i + 1, j]: -1.0, idx[i, j + 1]: -1.0, idx[i + 1, j + 1]: 1.0}))
    D = np.zeros((len(rows), nu * nv))
    for k, r in enumerate(rows):
        for c, v in r.items():
            D[k, c] = v
    return D


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# edges matched to a neighbouring patch: G0 / G1 / G2 as rows on the CV heights
# ---------------------------------------------------------------------------
#
# The patch meets a neighbour along an edge that lies on the neighbour's edge
# in plan view (the edge's CVs are snapped there first). With the CVs moving
# only along `up`, every condition is linear in the heights t:
#   G0   up . B(s)            = up . A          (the same point)
#   G1   n_A . B_c(s)         = 0               (the cross tangent in A's tangent plane)
#   G2   n_A . B_cc(s)        = II_A(B_c, B_c)  (the same normal curvature across)
# B_c, B_cc: the patch's first and second derivative across the edge; II_A the
# neighbour's second fundamental form. G0 and G1 along a shared curve already
# make the normal curvature along the edge and the twist term agree, so the
# one scalar G2 row per point is all G2 asks. II_A(B_c, B_c) depends on B_c,
# i.e. on the heights: it is taken from the last solve (a few rounds).

EDGES = ("u0", "u1", "v0", "v1")


def edge_tangent_rows(nu, nv, du, dv, Ku, Kv, edge, s):
    """The patch's derivative along `edge` at the parameters s, as rows over the nu * nv CVs."""
    from .surfeval import _basis_matrix
    s = np.asarray(s, float)
    if edge in ("u0", "u1"):
        Bc = _basis_matrix(Ku, du, nu, np.array([0.0 if edge == "u0" else 1.0]), 0)[0, 0]
        Ba = _basis_matrix(Kv, dv, nv, s, 1)[1]
        return np.einsum('i,mj->mij', Bc, Ba).reshape(len(s), -1)
    Bc = _basis_matrix(Kv, dv, nv, np.array([0.0 if edge == "v0" else 1.0]), 0)[0, 0]
    Ba = _basis_matrix(Ku, du, nu, s, 1)[1]
    return np.einsum('mi,j->mij', Ba, Bc).reshape(len(s), -1)


def edge_rows(nu, nv, du, dv, Ku, Kv, edge, s):
    """(R0, R1, R2, rows of the edge CVs): the patch's position and its first
    and second derivatives across `edge` (into the patch) at the parameters s
    along it, as rows over the nu * nv CVs."""
    from .surfeval import _basis_matrix
    s = np.asarray(s, float)
    if edge in ("u0", "u1"):
        c = np.array([0.0 if edge == "u0" else 1.0])
        Bc = _basis_matrix(Ku, du, nu, c, 2)[:, 0, :]             # (3, nu)
        Ba = _basis_matrix(Kv, dv, nv, s, 0)[0]                   # (m, nv)
        sign = 1.0 if edge == "u0" else -1.0
        R = [np.einsum('i,mj->mij', Bc[k] * sign ** k, Ba).reshape(len(s), -1) for k in range(3)]
        rows = [[(i if edge == "u0" else nu - 1 - i) * nv + j for j in range(nv)] for i in range(3)]
    else:
        c = np.array([0.0 if edge == "v0" else 1.0])
        Bc = _basis_matrix(Kv, dv, nv, c, 2)[:, 0, :]
        Ba = _basis_matrix(Ku, du, nu, s, 0)[0]
        sign = 1.0 if edge == "v0" else -1.0
        R = [np.einsum('mi,j->mij', Ba, Bc[k] * sign ** k).reshape(len(s), -1) for k in range(3)]
        rows = [[i * nv + (j if edge == "v0" else nv - 1 - j) for i in range(nu)] for j in range(3)]
    return R[0], R[1], R[2], rows


def edge_plan(C, du, dv, Ku, Kv, edge, m=48):
    """Points along an edge of the patch with CVs C (world), and their parameters."""
    nu, nv = C.shape[:2]
    s = np.linspace(0.0, 1.0, m)
    R0 = edge_rows(nu, nv, du, dv, Ku, Kv, edge, s)[0]
    return R0 @ C.reshape(-1, 3), s


def _flat(P, up):
    return P - np.outer(P @ up, up)


def continuity(C, du, dv, Ku, Kv, edge, match):
    """How the patch meets its neighbour along `edge` (match: the data from
    the neighbour): the largest gap, normal angle (degrees) and difference of
    the normal curvature across the edge (relative to the neighbour's largest
    there, in %)."""
    nu, nv = C.shape[:2]
    R0, R1, R2, _ = edge_rows(nu, nv, du, dv, Ku, Kv, edge, match["s"])
    Cf = C.reshape(-1, 3)
    P, Bc, Bcc = R0 @ Cf, R1 @ Cf, R2 @ Cf
    Ta = edge_tangent_rows(nu, nv, du, dv, Ku, Kv, edge, match["s"]) @ Cf
    nB = np.cross(Ta, Bc)
    nB /= np.maximum(np.linalg.norm(nB, axis=1), 1e-300)[:, None]
    nA = match["n"]
    gap = float(np.linalg.norm(P - match["A"], axis=1).max())
    ang = float(np.degrees(np.arccos(np.clip(np.abs((nA * nB).sum(1)), 0.0, 1.0))).max())
    w2 = (Bc * Bc).sum(1)
    kB = (nA * Bcc).sum(1) / np.maximum(w2, 1e-300)
    kA = np.einsum('mi,mij,mj->m', Bc, match["Q"], Bc) / np.maximum(w2, 1e-300)
    scale = max(float(np.abs(kA).max()), 1.0 / max(match["L"], 1e-9))
    curv = float(100.0 * np.abs(kB - kA).max() / scale)
    return {"gap": gap, "angle": ang, "curv": curv}


def drape_patch(C0, du, dv, Ku, Kv, scan, up=(0.0, 0.0, 1.0), smoothness=0.01, drape=False, offset=0.0,
                samples=64, iterations=6, fixed=None, drape_slack=0.01, steep=0.3, edges=()):
    """Move the CVs C0 (nu x nv x 3, world) of a NURBS patch (degrees du, dv;
    knots Ku, Kv) along `up` only, so the patch lies on the scan seen from
    above (offset: a clearance along the scan's normal). fixed: (nu, nv)
    mask of CVs that stay. edges: matches to neighbours (match_edge), each
    held at its level (0, 1, 2 for G0, G1, G2). Returns (CVs, report)."""
    t0 = time.perf_counter()
    C0 = np.asarray(C0, float)
    nu, nv = C0.shape[:2]
    n = nu * nv
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    s = int(samples)
    ts = (np.arange(s) + 0.5) / s
    Bu, Bv = basis(nu, du, Ku, ts), basis(nv, dv, Kv, ts)
    S = np.einsum('ai,bj->abij', Bu, Bv).reshape(s * s, n)       # sample (a, b) -> CV (i, j)
    Cf = C0.reshape(n, 3)
    Pb = S @ Cf                                                  # the samples as laid out
    # the scan straight below (and above) every sample: the first hit coming down from above the scan
    from mathutils import Vector
    lo, hi = scan.V.min(0), scan.V.max(0)
    reach = float(np.linalg.norm(hi - lo)) + 1.0
    top = max(float((scan.V @ up).max()), float((Pb @ up).max())) + 0.01 * reach
    h = np.full(len(Pb), np.nan)
    nz = np.zeros(len(Pb))
    tree = scan.tree
    down = Vector(tuple(-up))
    n_down = 0
    for k, p in enumerate(Pb):
        o = p + (top - float(p @ up)) * up
        hit = tree.ray_cast(Vector(tuple(o)), down, 2 * reach)
        if hit[0] is None:
            continue
        nrm = np.array(hit[1])
        if scan.flip:
            nrm = -nrm
        c = float(nrm @ up)
        if c < 0:                                                # a face seen from behind: its outward side is down
            c = -c
            n_down += 1
        q = np.array(hit[0])
        h[k] = float((q - p) @ up) + (offset / max(c, steep))
        nz[k] = c
    ok = np.isfinite(h) & (nz >= steep)                          # walls seen from above are not the patch's to fit
    if ok.sum() < max(3 * n, 50):
        raise ValueError("the scan is not below the sheet (or seen almost edge-on from above): "
                         f"{int(ok.sum())} of {len(h)} sample points over it")
    fixed = np.zeros(n, bool) if fixed is None else np.asarray(fixed, bool).ravel()
    free = ~fixed
    D = _fair_rows(nu, nv)
    Sk, hk, ck = S[ok], h[ok], nz[ok]
    t = np.zeros(n)
    hist = []
    w_rob = np.ones(len(hk))
    ups = []
    for e in edges:                                              # the rows of every matched edge, once
        R0, R1, R2, _ = edge_rows(nu, nv, du, dv, Ku, Kv, e["edge"], e["s"])
        nA = e["n"]
        cu = nA @ up                                             # (n_A . up): how a height moves the condition
        L = e["L"]
        m = len(e["s"])
        ups.append((e, R0, R1, R2, nA, cu, L, m))
    if edges:
        iterations = max(int(iterations), 6)
    for it in range(int(iterations)):
        r = (Sk @ t - hk) * ck                                   # along the scan's normal; + above the scan
        if it > 0:
            sig = max(10.0 * float(np.median(np.abs(r))), 1e-6 * reach)
            w_rob = 1.0 / (1.0 + (r / sig) ** 2)
        w = w_rob * ck * ck
        if drape:
            w = w * np.where(r > 0, drape_slack, 1.0)
        w = w / max(float(w.sum()), 1e-300)
        M = (Sk * w[:, None]).T @ Sk + float(smoothness) * (D.T @ D) / max(len(D), 1) + 1e-12 * np.eye(n)
        b = (Sk * w[:, None]).T @ hk
        # the edge conditions first, as exactly as they can hold (least squares among themselves), then the
        # scan with what freedom is left: t = t_c + N z, N spanning the heights that leave every condition be
        C_now = Cf + t[:, None] * up
        Crows, crhs = [], []
        for e, R0, R1, R2, nA, cu, L, m in ups:
            Crows.append(R0)
            crhs.append(e["A"] @ up - (R0 @ Cf) @ up)
            if e["level"] >= 1:
                Bc = R1 @ C_now
                wl = np.maximum(np.linalg.norm(Bc, axis=1), 1e-300)
                Crows.append(R1 * (L * cu / wl)[:, None])
                crhs.append(-L * ((R1 @ Cf) * nA).sum(1) / wl)
            if e["level"] >= 2:
                Bc = R1 @ C_now
                w2 = np.maximum((Bc * Bc).sum(1), 1e-300)
                qa = np.einsum('mi,mij,mj->m', Bc, e["Q"], Bc)
                Crows.append(R2 * (L * L * cu / w2)[:, None])
                crhs.append(L * L * (qa - ((R2 @ Cf) * nA).sum(1)) / w2)
        if fixed.any():
            Crows.append(np.eye(n)[fixed])
            crhs.append(np.zeros(int(fixed.sum())))
        if Crows:
            Cm, cr = np.vstack(Crows), np.concatenate(crhs)
            U_, sv, Vt = np.linalg.svd(Cm, full_matrices=True)
            rank = int((sv > 1e-9 * max(float(sv[0]), 1e-300)).sum())
            tc = Vt[:rank].T @ ((U_[:, :rank].T @ cr) / sv[:rank])  # the conditions' least-squares heights
            N = Vt[rank:].T                                          # heights that leave them as they are
            if N.shape[1]:
                z = np.linalg.solve(N.T @ M @ N, N.T @ (b - M @ tc))
                tn = tc + N @ z
            else:
                tn = tc
        else:
            tn = np.linalg.solve(M, b)
        moved = float(np.abs(tn - t).max())
        t = tn
        hist.append({"it": it, "moved": moved})
        if it > 1 and moved < 1e-7 * reach and not (edges and it < 4):
            break
    C = (Cf + t[:, None] * up).reshape(nu, nv, 3)
    for e in edges:
        e["result"] = continuity(C, du, dv, Ku, Kv, e["edge"], e)
    rep = {"samples": int(ok.sum()), "of": len(h), "iterations": len(hist), "time": time.perf_counter() - t0,
           "down": n_down / max(int(np.isfinite(h).sum()), 1),
           "degree": (du, dv), "cvs": (nu, nv), "spans": (nu - du, nv - dv)}
    return C, rep


def deviation(C, du, dv, Ku, Kv, scan, offset=0.0, samples=80):
    """Signed distance of the patch (CVs C) from the scan over the part that lies over it, area weighted."""
    from . import scanfit
    nu, nv = C.shape[:2]
    ts = np.linspace(0.0, 1.0, samples)
    S = np.einsum('ai,bj->abij', basis(nu, du, Ku, ts), basis(nv, dv, Kv, ts)).reshape(samples * samples, nu * nv)
    P = S @ C.reshape(-1, 3)
    i = np.arange(samples - 1)
    a = (i[:, None] * samples + i[None, :]).ravel()
    F = np.stack([a, a + samples, a + samples + 1, a + 1], 1)
    return scanfit.deviation_stats(P, scan, offset, F)


# ===========================================================================
# Blender
# ===========================================================================

import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty  # noqa: E402

LAST_INFO = {}


def sheet_grid(obj):
    """(CVs nu x nv x 3 in world space, vertex grid) of a quad-grid mesh object (its own mesh, no modifiers)."""
    me = obj.data
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    G = grid_order(len(V), [tuple(p.vertices) for p in me.polygons])
    M = np.array(geom.world_matrix(obj))
    Vw = V @ M[:3, :3].T + M[:3, 3]
    return Vw[G], G


class Neighbour:
    """A neighbouring NURBS patch (one face) in world space: its four parameter edges and its geometry on them."""

    def __init__(self, obj):
        from . import store, surfeval
        faces = store.get_brep(obj).faces()
        if len(faces) != 1:
            raise ValueError(f"{obj.name}: a neighbour must be one NURBS face")
        self.S = surfeval.Surf(faces[0].surface())
        self.M = np.array(geom.world_matrix(obj))
        self.Lm = self.M[:3, :3]
        self.name = obj.name

    def iso(self, k, n=400):
        """(u, v) along the surface's parameter edge k (0: u min, 1: u max, 2: v min, 3: v max)."""
        (u0, u1), (v0, v1) = self.S.u_domain, self.S.v_domain
        t = np.linspace(0.0, 1.0, n)
        if k < 2:
            return np.full(n, u0 if k == 0 else u1), v0 + t * (v1 - v0)
        return u0 + t * (u1 - u0), np.full(n, v0 if k == 2 else v1)

    def eval(self, u, v, nd=2):
        D = self.S.at(u, v, nd)
        return {kl: (a @ self.Lm.T + (self.M[:3, 3] if kl == (0, 0) else 0.0)) for kl, a in D.items()}


def second_form(D):
    """(unit normal, second fundamental form as a world tensor on tangent vectors) of an evaluation dict (nd >= 2)."""
    from .surfeval import normals
    su, sv = D[1, 0], D[0, 1]
    n, _ = normals(D)
    J = np.stack([su, sv], axis=-1)
    JtJ = np.einsum("mia,mib->mab", J, J)
    Jp = np.linalg.solve(JtJ, np.transpose(J, (0, 2, 1)))         # (m, 2, 3): world tangent -> (du, dv)
    H = np.stack([np.stack([(D[2, 0] * n).sum(1), (D[1, 1] * n).sum(1)], -1),
                  np.stack([(D[1, 1] * n).sum(1), (D[0, 2] * n).sum(1)], -1)], -2)
    return n, np.einsum("mab,mai,mbj->mij", H, Jp, Jp)


def find_matches(C, du, dv, Ku, Kv, neighbours, up, tol=0.05):
    """For each edge of the patch, the neighbour edge lying along it in plan view, if one lies within tol (of the
    edge's length) on average: {edge: (neighbour, its edge k, mean plan distance)}."""
    from .scanfit import polyline_nearest
    out = {}
    for e in EDGES:
        P, _ = edge_plan(C, du, dv, Ku, Kv, e)
        Pf = _flat(P, up)
        Lb = float(np.linalg.norm(np.diff(Pf, axis=0), axis=1).sum())
        best = None
        for nb in neighbours:
            for k in range(4):
                A = nb.eval(*nb.iso(k, 200), nd=0)[0, 0]
                d = float(polyline_nearest(Pf, _flat(A, up), closed=False)[4].mean())
                if best is None or d < best[2]:
                    best = (nb, k, d)
        if best is not None and best[2] < tol * Lb:
            out[e] = best
    return out


def snap_edge(C, du, dv, Ku, Kv, edge, nb, k, up):
    """Move the patch's edge CVs (across `up` only) so its edge lies on the
    neighbour's edge k in plan view with the neighbour's parametrisation: the
    edge is the neighbour's edge between the points nearest its two ends, at
    the same parameters (the same CV spacing along it: a neighbour edge of
    one span and the same or lower degree is then met exactly). The edge's
    second and third CV rows shift along with it, so the CV columns stay
    straight."""
    from .scanfit import polyline_nearest
    C = C.copy()
    nu, nv = C.shape[:2]
    u, v = nb.iso(k, 801)
    Af = _flat(nb.eval(u, v, nd=0)[0, 0], up)
    s = np.linspace(0.0, 1.0, 81)
    R0, _, _, rows = edge_rows(nu, nv, du, dv, Ku, Kv, edge, s)
    Cf = C.reshape(-1, 3)
    ends = _flat(R0[[0, -1]] @ Cf, up)
    _, _, seg, lam, _ = polyline_nearest(ends, Af, closed=False)
    tau = (seg + lam) / (len(u) - 1)                                # the ends' places along the neighbour edge
    tt = tau[0] + s * (tau[1] - tau[0])
    T = _flat(nb.eval(np.interp(tt, np.linspace(0, 1, len(u)), u), np.interp(tt, np.linspace(0, 1, len(v)), v),
                      nd=0)[0, 0], up)
    r0 = rows[0]
    X = np.linalg.lstsq(R0[:, r0], T, rcond=None)[0]
    delta = _flat(X, up) - _flat(Cf[r0], up)
    for r in rows[:3]:
        Cf[r] += delta
    return Cf.reshape(nu, nv, 3)


def edge_match(C, du, dv, Ku, Kv, edge, nb, k, up, level, m=40):
    """The neighbour's geometry at the patch's edge points (by plan position): the data drape_patch holds the
    edge to."""
    from .scanfit import polyline_nearest
    P, s = edge_plan(C, du, dv, Ku, Kv, edge, m)
    u, v = nb.iso(k, 800)
    A = nb.eval(u, v, nd=0)[0, 0]
    _, _, seg, lam, _ = polyline_nearest(_flat(P, up), _flat(A, up), closed=False)
    uu = u[seg] + lam * (u[np.minimum(seg + 1, len(u) - 1)] - u[seg])
    vv = v[seg] + lam * (v[np.minimum(seg + 1, len(v) - 1)] - v[seg])
    D = nb.eval(uu, vv, nd=2)
    n, Q = second_form(D)
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return {"edge": edge, "level": int(level), "s": s, "A": D[0, 0], "n": n, "Q": Q, "L": L,
            "neighbour": nb.name, "k": k}


class NPROTO_OT_scan_drape_patch(bpy.types.Operator):
    """Lay one NURBS patch straight down onto the scan: the active mesh is the patch's control net (a quad grid
    above the scan; 6 x 6 vertices = one span of degree 5 each way, three CV rows from every edge for G0, G1, G2
    to a neighbour later). Its CVs only move up or down, so the patch keeps its plan view, footprint and CV
    columns; the scan is the other selected mesh or the Scan under Analyze"""
    bl_idname = "nproto.scan_drape_patch"
    bl_label = "Drape Patch to Scan"
    bl_options = {'REGISTER', 'UNDO'}

    max_degree: IntProperty(name="Max Degree", default=5, min=1, max=9,
                            description="Highest degree each way: a grid of degree + 1 vertices is one span; more "
                                        "vertices than that make more spans")
    direction: EnumProperty(name="Direction", default='Z',
                            items=[('Z', "World Z", "CVs move up and down (straight down onto the scan)"),
                                   ('NORMAL', "Sheet Normal", "CVs move along the sheet's own normal")],
                            description="The one direction the CVs move in")
    drape: BoolProperty(name="Drape", default=False,
                        description="Lay the patch on the scan like a sheet: pushed up wherever it dips into the "
                                    "scan, only lightly pulled down where it stands above it, so it bridges panel "
                                    "gaps and hollows instead of following them in")
    offset: FloatProperty(name="Offset", default=0.0, subtype='DISTANCE', unit='LENGTH', precision=4,
                          description="Fit this far above the scan (along its normal); negative goes in")
    smoothness: FloatProperty(name="Smoothness", default=0.0001, min=0.0, soft_max=1.0, max=100.0, precision=4,
                              description="How evenly the CV net bends: higher keeps the patch calmer where the "
                                          "scan is noisy or has gaps, at the cost of a looser fit")
    match: EnumProperty(name="Match Neighbours", default='G2',
                        items=[('NONE', "Off", "Fit the scan only"),
                               ('G0', "G0", "Edges meet the neighbouring patches' edges (position)"),
                               ('G1', "G1", "... and are tangent to them"),
                               ('G2', "G2", "... and have the same curvature across the edge")],
                        description="Continuity to the selected NURBS patches whose edge lies along an edge of the "
                                    "sheet (in plan view): the edge's first one, two or three CV rows are set by it")
    snap_edges: BoolProperty(name="Snap Edges", default=True,
                             description="Move a matched edge's CVs (across the drape direction only) so the edge "
                                         "lies on the neighbour's edge in plan view")
    show_deviation: BoolProperty(name="Show Deviation", default=True,
                                 description="Turn on the Scan Deviation overlay (Analyze) afterwards")

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == 'MESH' and 'nproto_brep' not in o and len(o.data.polygons) > 0

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "max_degree")
        lay.prop(self, "direction")
        lay.prop(self, "drape")
        lay.prop(self, "offset")
        lay.prop(self, "smoothness")
        lay.prop(self, "match")
        if self.match != 'NONE':
            lay.prop(self, "snap_edges")
        lay.prop(self, "show_deviation")
        if LAST_INFO:
            from . import scanfit
            box = lay.box()
            col = box.column(align=True)
            L = lambda x: scanfit._length(context, x)
            i = LAST_INFO
            col.label(text=f"degree {i['degree'][0]} x {i['degree'][1]}  ·  {i['cvs'][0]} x {i['cvs'][1]} CVs  ·  "
                           f"{i['spans'][0]} x {i['spans'][1]} span{'s' if max(i['spans']) > 1 else ''}",
                      icon='SURFACE_NSURFACE')
            col.label(text=f"RMS {L(i['dev_rms'])}  ·  95 % within {L(i['dev_p95'])}")
            col.label(text=f"most inside {L(i['dev_min'])}  ·  most outside {L(i['dev_hi'])}")
            for e in i.get("edges", ()):
                r = e["result"]
                txt = f"{e['edge']} -> {e['neighbour']}: G{e['level']}  gap {L(r['gap'])}"
                if e["level"] >= 1:
                    txt += f"  ·  {r['angle']:.3f}°"
                if e["level"] >= 2:
                    txt += f"  ·  curvature {r['curv']:.2f} %"
                col.label(text=txt, icon='LINKED')

    def execute(self, context):
        from . import scanfit, store
        sheet = context.active_object
        others = [o for o in context.selected_objects
                  if o is not sheet and o.type == 'MESH' and len(o.data.polygons) and not store.is_nurbs(o)]
        scan_obj = others[0] if len(others) == 1 else context.scene.nproto_settings.scan_object
        if scan_obj is None or scan_obj is sheet or scan_obj.type != 'MESH':
            self.report({'WARNING'}, "Select the scan too (or set it under Analyze > Scan Deviation)")
            return {'CANCELLED'}
        if sheet.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            C0, _ = sheet_grid(sheet)
            nu, nv = C0.shape[:2]
            du, Ku = degree_and_knots(nu, self.max_degree)
            dv, Kv = degree_and_knots(nv, self.max_degree)
            if self.direction == 'Z':
                up = np.array([0.0, 0.0, 1.0])
            else:
                P = C0.reshape(-1, 3)
                up = np.linalg.svd(P - P.mean(0))[2][2]
                if up[2] < 0:
                    up = -up
            scan = scanfit.scan_for(scan_obj)
            own = sheet.get("nproto_patch")
            nbs = [o for o in context.selected_objects if o is not sheet and store.is_nurbs(o) and o.name != own]
            if not nbs and self.match != 'NONE':          # (run again later: the neighbours it was matched to)
                nbs = [bpy.data.objects[n] for n in sheet.get("nproto_neighbours", "").split("\t")
                       if n and n in bpy.data.objects and n != own]
            matches = []
            if self.match != 'NONE' and nbs:
                level = int(self.match[1])
                found = find_matches(C0, du, dv, Ku, Kv, [Neighbour(o) for o in nbs], up)
                for e, (nb, k, _) in found.items():
                    if self.snap_edges:
                        C0 = snap_edge(C0, du, dv, Ku, Kv, e, nb, k, up)
                for e, (nb, k, _) in found.items():
                    deg = du if e in ("u0", "u1") else dv
                    matches.append(edge_match(C0, du, dv, Ku, Kv, e, nb, k, up, min(level, deg)))
            C, rep = drape_patch(C0, du, dv, Ku, Kv, scan, up=up, smoothness=self.smoothness, drape=self.drape,
                                 offset=self.offset, edges=matches)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.report({'WARNING'}, f"Drape Patch to Scan: {exc}")
            return {'CANCELLED'}
        if rep["down"] > 0.5:
            # most of the scan seen from above faces down: an open scan (a single panel) whose outside was
            # guessed the wrong way; the side the patch is draped from is the outside
            scan.flip_normals()
            scanfit.DEV_CACHE.clear()
        _, ext = store._k()
        pts = [tuple(map(float, C[i, j])) for i in range(nu) for j in range(nv)]
        surf = ext.surface_from_cvs(du, dv, pts, Ku, Kv)
        name = sheet.get("nproto_patch")
        patch = bpy.data.objects.get(name) if name else None
        if patch is not None and store.is_nurbs(patch) and patch.name in context.scene.objects:
            patch.matrix_world = Matrix.Identity(4)
            store.set_geometry(patch, surf, context)              # the sheet's own patch: updated in place
        else:
            patch = store.new_object(context, sheet.name + " Patch", surf)
        sheet["nproto_patch"] = patch.name
        if matches:
            sheet["nproto_neighbours"] = "\t".join(sorted({m["neighbour"] for m in matches}))
        patch["nproto_scan"] = scan_obj.name
        dev = deviation(C, du, dv, Ku, Kv, scan, self.offset)
        info = dict(rep)
        info["edges"] = [{"edge": m["edge"], "neighbour": m["neighbour"], "level": m["level"], "result": m["result"]}
                         for m in matches]
        info.update({k: dev[k] for k in ("dev_rms", "dev_p95", "dev_min", "dev_hi", "dev_max", "dev_mean")})
        LAST_INFO.clear()
        LAST_INFO.update(info)
        for o in context.selected_objects:
            o.select_set(False)
        patch.select_set(True)
        context.view_layer.objects.active = patch
        if self.show_deviation:
            scanfit._show_deviation(context, scan_obj)
        L = lambda x: scanfit._length(context, x)
        msg = (f"Drape Patch to Scan: degree {du} x {dv}, {nu} x {nv} CVs  ·  RMS {L(info['dev_rms'])}"
               f"  ·  95 % within {L(info['dev_p95'])}")
        if matches:
            msg += "  ·  " + ", ".join(f"G{m['level']} to {m['neighbour']}" for m in matches)
        elif self.match != 'NONE' and nbs:
            msg += "  ·  no neighbour edge lies along the sheet's edges (in plan view)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


classes = (NPROTO_OT_scan_drape_patch,)
