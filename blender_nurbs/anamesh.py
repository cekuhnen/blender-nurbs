"""Analysis meshes: fine triangle meshes of the exact B-rep for the zebra,
draft angle, matcap and curvature map overlays, with the EXACT normal (and
curvatures) at every vertex.

The kernel's tessellator costs about 5 us per triangle; at the fineness the
stripes need (about 1 degree between neighbouring vertex normals) a SubD body
became 400k - 1.2M triangles and several seconds. Here:

* An untrimmed face (its boundary runs along the surface's iso boundaries:
  every face of a SubD or Mesh to BREP body, most single surfaces) is meshed
  as a grid in (u, v). The samples follow the knot spans -- each span gets as
  many steps as its normals turn (angle tolerance) and its sag needs (chord
  tolerance) -- and the whole grid is evaluated with two matrix products
  (surfeval). C0 knot lines (creases) split the grid, so each side keeps its
  own normals.
* A trimmed face is tessellated alone by the kernel at a coarse setting, then
  every triangle is split uniformly in (u, v) (feet of its corners, the new
  points evaluated exactly), finely enough for the same angle tolerance.

Every vertex lies on the surface; normals are oriented outwards (the face's
outward normal), curvatures are signed so that convex is positive.
"""

import functools
import math

import numpy as np

from . import store, surfeval

CURV_KINDS = ("GAUSS", "MEAN", "MAX", "MIN")
MAX_STEPS = 4000          # per knot span (a safety cap)
MAX_FACE_VERTS = 2000000  # per face: beyond this the tolerances are relaxed
COARSE_ANGLE = 10.0       # the kernel's tessellation of a trimmed face, before the split
MAX_SPLIT = 16            # most pieces per triangle edge
MAX_EDGE_REL = 1.0 / 30.0  # longest triangle edge, relative to the body's size (as the kernel's mesh had)


# -- untrimmed faces -----------------------------------------------------------

def _c0_knots(K, p, lo, hi):
    """Interior knots of multiplicity >= p (where the surface may crease)."""
    tol = 1e-12 * (hi - lo or 1.0)
    vals, cnt = np.unique(np.round(K / (hi - lo or 1.0), 12) * (hi - lo or 1.0), return_counts=True)
    return [float(x) for x, c in zip(vals, cnt) if c >= p and lo + tol < x < hi - tol]


def _spans(K, lo, hi):
    """Distinct knot values in [lo, hi] (span ends)."""
    k = np.unique(K[(K >= lo) & (K <= hi)])
    if not len(k) or k[0] > lo:
        k = np.r_[lo, k]
    if k[-1] < hi:
        k = np.r_[k, hi]
    out = [k[0]]
    for x in k[1:]:
        if x - out[-1] > 1e-12 * (hi - lo):
            out.append(x)
    return np.array(out)


def _probe(E, eu, ev, probe=8):
    """Normals, bending and speed |S_t| in both directions on a probe grid of
    probe samples per knot span, evaluated once. Bending: the part of S_tt
    across the iso curve's tangent -- also within the tangent plane, so a flat
    face with a curved outline still gets the steps its outline's sag needs."""
    tu = np.concatenate([np.linspace(a, b, probe + 1)[:-1] for a, b in zip(eu[:-1], eu[1:])] + [eu[-1:]])
    tv = np.concatenate([np.linspace(a, b, probe + 1)[:-1] for a, b in zip(ev[:-1], ev[1:])] + [ev[-1:]])
    D = E.grid(tu, tv, 2)
    n, ln = surfeval.normals(D)
    # where su x sv vanishes (a pole) the normal is undefined: a zero vector,
    # which _steps' turning ignores (cos 1 against anything is not asked)
    n[ln <= 1e-9 * max(float(np.median(ln)), 1e-300)] = 0.0
    su, sv = np.linalg.norm(D[1, 0], axis=2), np.linalg.norm(D[0, 1], axis=2)
    bu = np.linalg.norm(np.cross(D[2, 0], D[1, 0]), axis=2) / np.where(su > 0, su, 1.0)
    bv = np.linalg.norm(np.cross(D[0, 2], D[0, 1]), axis=2) / np.where(sv > 0, sv, 1.0)
    # unit tangents of the iso curves (zero where the curve stands still, a pole)
    ref = 1e-9 * max(float(np.median(su)), float(np.median(sv)), 1e-300)
    du = np.where((su > ref)[..., None], D[1, 0] / np.where(su > 0, su, 1.0)[..., None], 0.0)
    dv = np.where((sv > ref)[..., None], D[0, 1] / np.where(sv > 0, sv, 1.0)[..., None], 0.0)
    return n, bu, bv, su, sv, du, dv


def _turning(d):
    """Angle between consecutive unit vectors along axis 0 (0 where one is
    undefined, a zero vector)."""
    cos = np.clip(np.sum(d[1:] * d[:-1], axis=-1), -1.0, 1.0)
    defined = (np.sum(d[1:] * d[1:], axis=-1) > 0.5) & (np.sum(d[:-1] * d[:-1], axis=-1) > 0.5)
    return np.where(defined, np.arccos(cos), 0.0)


def _steps(ends, n, bend, speed, tang, angle_rad, chord_abs, edge_abs, probe=8):
    """Parameters along one direction (axis 0 of the probe arrays): every span
    [ends[i], ends[i + 1]] split into equal steps, as many as its normals turn
    by (angle_rad each, the most of any probe row) -- and the iso curves along
    the face's two borders turn by (tang: their unit tangents; the outline
    then follows a curved edge as the display mesh's does) --, its sag allows
    (chord_abs) and its length needs (edge_abs: no triangle edge longer; 0 =
    no limit)."""
    turn = _turning(n)                                       # (len(ts) - 1, rows)
    rim = _turning(tang[:, [0, -1]])                         # (len(ts) - 1, 2)
    out = []
    for s in range(len(ends) - 1):
        a, b = ends[s], ends[s + 1]
        rows = slice(s * probe, (s + 1) * probe + 1)
        seg = slice(s * probe, (s + 1) * probe)
        k = 1
        if angle_rad > 0:
            k = int(math.ceil(max(float(turn[seg].sum(axis=0).max()), float(rim[seg].sum(axis=0).max())) / angle_rad))
        if chord_abs > 0:
            kap = float(bend[rows].max())
            if kap > 0:
                k = max(k, int(math.ceil((b - a) / math.sqrt(8.0 * chord_abs / kap))))
        if edge_abs > 0:
            k = max(k, int(math.ceil((b - a) * float(speed[rows].max()) / edge_abs)))
        k = min(max(k, 1), MAX_STEPS)
        out.append(np.linspace(a, b, k + 1)[:-1])
    out.append(np.array([ends[-1]]))
    return np.concatenate(out)


def _grid_tris(mu, mv, P):
    """Two triangles per grid cell, split along the shorter diagonal."""
    i, j = np.meshgrid(np.arange(mu - 1), np.arange(mv - 1), indexing="ij")
    a = (i * mv + j).ravel()
    b = a + mv           # (i + 1, j)
    c = b + 1            # (i + 1, j + 1)
    d = a + 1            # (i, j + 1)
    d1 = np.einsum("ij,ij->i", P[a] - P[c], P[a] - P[c])
    d2 = np.einsum("ij,ij->i", P[b] - P[d], P[b] - P[d])
    short = d1 <= d2
    T1 = np.where(short[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1))
    T2 = np.where(short[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))
    return np.vstack([T1, T2])


def _fix_poles(E, uu, vv, D, nrm, ln):
    """Normals where su x sv vanishes (a collapsed edge such as a sphere's pole):
    the limit, from a point a tiny step towards the middle of the domain."""
    ref = float(np.median(ln)) if ln.size else 0.0
    bad = ln <= 1e-9 * max(ref, 1e-300)
    if not bad.any():
        return nrm
    (u0, u1), (v0, v1) = E.u_domain, E.v_domain
    uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    U, V = uu[bad], vv[bad]
    for step in (1e-6, 1e-4, 1e-3, 1e-2):
        Ds = E.at(U + (uc - U) * step, V + (vc - V) * step, 1)
        n2, l2 = surfeval.normals(Ds)
        if (l2 > 1e-9 * max(ref, 1e-300)).all() or step == 1e-2:
            nrm[bad] = n2
            break
    return nrm


def grid_mesh(E, angle_deg, chord_abs, curv=False, edge_abs=0.0):
    """(P, N, T, uv, D) of an untrimmed surface, P / N / uv per vertex; D: the
    evaluation dict on the vertices (second derivatives when curv). edge_abs:
    longest triangle edge (not on a flat surface, whose normals are all one)."""
    (u0, u1), (v0, v1) = E.u_domain, E.v_domain
    ang = math.radians(angle_deg)
    cu = [u0] + _c0_knots(E.U, E.p, u0, u1) + [u1]
    cv = [v0] + _c0_knots(E.V, E.q, v0, v1) + [v1]
    eu, ev = _spans(E.U, u0, u1), _spans(E.V, v0, v1)
    n, bu, bv, su, sv, du, dv = _probe(E, eu, ev)
    ref = n.reshape(-1, 3)[np.argmax(np.einsum("ij,ij->i", n.reshape(-1, 3), n.reshape(-1, 3)))]
    if float(np.einsum("ijk,k->ij", n, ref)[np.einsum("ijk,ijk->ij", n, n) > 0.5].min()) > 1.0 - 1e-12:
        edge_abs = 0.0                                       # flat: two triangles shade it exactly
    f = 1.0
    while True:
        us_all = _steps(eu, n, bu, su, du, ang * f, chord_abs * f * f, edge_abs * f)
        vs_all = _steps(ev, n.transpose(1, 0, 2), bv.T, sv.T, dv.transpose(1, 0, 2), ang * f, chord_abs * f * f,
                        edge_abs * f)
        if len(us_all) * len(vs_all) <= MAX_FACE_VERTS:
            break
        f *= 1.25 * math.sqrt(len(us_all) * len(vs_all) / MAX_FACE_VERTS)
    Ps, Ns, Ts, UVs, Ds, base = [], [], [], [], [], 0
    nd = 2 if curv else 1
    hu, hv = 1e-11 * (u1 - u0), 1e-11 * (v1 - v0)
    for a0, a1 in zip(cu[:-1], cu[1:]):
        us = us_all[(us_all >= a0 - hu) & (us_all <= a1 + hu)].copy()
        us = np.unique(np.r_[a0, us, a1])
        ue = us.copy()
        ue[-1] = a1 - (hu if a1 < u1 else 0.0)       # the piece's own side of a crease
        ue[0] = a0 + (hu if a0 > u0 else 0.0)
        for b0, b1 in zip(cv[:-1], cv[1:]):
            vs = vs_all[(vs_all >= b0 - hv) & (vs_all <= b1 + hv)].copy()
            vs = np.unique(np.r_[b0, vs, b1])
            ve = vs.copy()
            ve[-1] = b1 - (hv if b1 < v1 else 0.0)
            ve[0] = b0 + (hv if b0 > v0 else 0.0)
            D = E.grid(ue, ve, nd)
            mu, mv = len(ue), len(ve)
            P = D[0, 0].reshape(-1, 3)
            n, ln = surfeval.normals(D)
            n, ln = n.reshape(-1, 3), ln.ravel()
            uu, vv = np.repeat(ue, mv), np.tile(ve, mu)
            n = _fix_poles(E, uu, vv, D, n, ln)
            Ps.append(P)
            Ns.append(n)
            Ts.append(_grid_tris(mu, mv, P) + base)
            UVs.append(np.stack([uu, vv], 1))
            Ds.append({k: x.reshape(-1, 3) for k, x in D.items()})
            base += len(P)
    D = {k: np.vstack([d[k] for d in Ds]) for k in Ds[0]}
    return np.vstack(Ps), np.vstack(Ns), np.vstack(Ts), np.vstack(UVs), D


def is_untrimmed(face, E, size):
    """Whether the face is its whole surface: the trim box is the surface's
    domain and every edge runs along an iso boundary (a tube's two end loops
    do; a hole's loop, a notch or a trim across the surface does not)."""
    try:
        (u0, u1), (v0, v1) = E.u_domain, E.v_domain
        (fu0, fv0), (fu1, fv1) = face.uv_domain()
        span = max(u1 - u0, v1 - v0) or 1.0
        if max(abs(fu0 - u0), abs(fu1 - u1), abs(fv0 - v0), abs(fv1 - v1)) > 1e-7 * span:
            return False
        pts = []
        for e in face.edges():
            t0, t1 = e.parameter_range()
            pts += [e.point_at(t0 + (t1 - t0) * k) for k in (0.2, 0.5, 0.8)]
        if not pts:
            return False
        return bool(_on_iso_boundary(E, np.asarray(pts, float), 2e-6 * size).all())
    except Exception:
        return False


def _on_iso_boundary(E, Q, tol, n=1025):
    """Whether each point lies on one of the surface's four iso boundaries:
    its distance to their fine polylines (segments: off the curve by their sag
    only, ~1e-7 of the size) is within tol. Only the two segments at the
    nearest polyline point are measured."""
    (u0, u1), (v0, v1) = E.u_domain, E.v_domain
    tu, tv = np.linspace(u0, u1, n), np.linspace(v0, v1, n)
    rows = np.concatenate([E.grid([u0, u1], tv, 0)[0, 0], E.grid(tu, [v0, v1], 0)[0, 0].transpose(1, 0, 2)])
    flat = rows.reshape(-1, 3)
    k = store.nearest_rows(flat, Q)
    j = k % n
    best = np.full(len(Q), np.inf)
    for a, b in ((j - 1, j), (j, j + 1)):
        ok = (a >= 0) & (b < n)
        a, b = np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)
        A, B = flat[k - j + a], flat[k - j + b]
        AB = B - A
        L2 = np.einsum("ij,ij->i", AB, AB)
        w = np.clip(np.einsum("ij,ij->i", Q - A, AB) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        d = np.linalg.norm(A + w[:, None] * AB - Q, axis=1)
        best = np.where(ok, np.minimum(best, d), best)
    return best <= tol


# -- trimmed faces -------------------------------------------------------------

def _single_face(brep, fi):
    if brep.face_count() == 1:
        return brep
    _, ext = store._k()
    b = brep.copy()
    ext.delete_faces(b, [f for i, f in enumerate(b.faces()) if i != fi])
    return b


@functools.lru_cache(maxsize=4096)
def _lattice(k, ka=0, kb=0, kc=0):
    """A triangle split into k x k: (barycentric weights (m, 3), sub-triangles
    as indices into them). ka, kb, kc: fewer pieces along the edges 0-1, 1-2,
    2-0 -- the points there move to the nearest of ka (kb, kc) equal steps
    (in order, so no sub-triangle turns over; those left without area are
    dropped), and the edge meets a neighbouring triangle split ka (kb, kc)
    times point for point."""
    idx = {}
    W = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            idx[i, j] = len(W)
            W.append([1.0 - (i + j) / k, i / k, j / k])
    W = np.asarray(W, float)

    def snap(x, ke):                       # fraction x / k along an edge -> nearest of ke steps
        return round(x * ke / k) / ke
    if 0 < ka < k:                         # edge 0-1: j = 0, t = i / k from corner 0 to 1
        for i in range(k + 1):
            t = snap(i, ka)
            W[idx[i, 0]] = (1.0 - t, t, 0.0)
    if 0 < kb < k:                         # edge 1-2: i + j = k, t = j / k from corner 1 to 2
        for j in range(k + 1):
            t = snap(j, kb)
            W[idx[k - j, j]] = (0.0, 1.0 - t, t)
    if 0 < kc < k:                         # edge 2-0: i = 0, t = (k - j) / k from corner 2 to 0
        for j in range(k + 1):
            t = snap(k - j, kc)
            W[idx[0, j]] = (t, 0.0, 1.0 - t)
    T = []
    for i in range(k):
        for j in range(k - i):
            T.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))
            if i + j < k - 1:
                T.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
    T = np.asarray(T, np.int64)
    # merge points that landed on the same spot, drop what lost its area
    key = np.round(W * (4 * k * max(ka, kb, kc, 1))).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    T = first[inv.ravel()][T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 2] != T[:, 0])]
    # where each point lies: kind 0 a corner (which), 1 on an edge (which, and
    # its step from the edge's first corner, of that edge's pieces), 2 inside
    kind = np.full(len(W), 2, np.int64)
    which = np.zeros(len(W), np.int64)
    step = np.zeros(len(W), np.int64)
    pieces = (ka or k, kb or k, kc or k)
    for e, (zero, along) in enumerate(((2, 1), (0, 2), (1, 0))):    # edge 0-1: w2 = 0, t = w1; ...
        on = (np.abs(W[:, zero]) < 1e-12) & (kind == 2)
        kind[on] = 1
        which[on] = e
        step[on] = np.round(W[on, along] * pieces[e]).astype(np.int64)
    corner = W.max(axis=1) > 1.0 - 1e-12
    kind[corner] = 0
    which[corner] = np.argmax(W[corner], axis=1)
    for a in (W, T, kind, which, step):
        a.setflags(write=False)
    return W, T, kind, which, step


def _flip_needles(P, N, T, thin=0.02, passes=4):
    """Needles (a triangle whose height is under thin x its longest edge: the
    kernel fans them out of trim corners) flipped away: the longest edge
    becomes the other diagonal of the quad it forms with the neighbour across
    it, when both new triangles face along the surface and the thinner of them
    is fatter than the needle. Split, a needle stays one; flipped, its corner
    joins well-shaped triangles."""
    T = np.array(T, np.int64)
    if not len(T):
        return T

    Pl, Nl = P.tolist(), N.tolist()

    def fat(t):
        """(height / longest edge, how far the triangle faces along the normals)."""
        a, b, c = Pl[t[0]], Pl[t[1]], Pl[t[2]]
        e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        f = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
        L = max(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2, e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2,
                (c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2 + (c[2] - b[2]) ** 2)
        na, nb, nc = Nl[t[0]], Nl[t[1]], Nl[t[2]]
        up = sum(f[i] * (na[i] + nb[i] + nc[i]) for i in range(3))
        return (math.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2) / L if L > 0 else 0.0), up
    for _ in range(passes):
        Pt = P[T]
        L2 = np.sum((Pt - np.roll(Pt, -1, axis=1)) ** 2, axis=2)          # edges 0-1, 1-2, 2-0
        area2 = np.linalg.norm(np.cross(Pt[:, 1] - Pt[:, 0], Pt[:, 2] - Pt[:, 0]), axis=1)
        E3 = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1)
        Es = np.sort(E3, axis=2)
        key = Es[:, :, 0] * (len(P) + 1) + Es[:, :, 1]
        uk, inv, cnt = np.unique(key.ravel(), return_inverse=True, return_counts=True)
        longest = np.argmax(L2, axis=1)
        lk = inv.reshape(-1, 3)[np.arange(len(T)), longest]              # each triangle's longest edge
        # needles whose longest edge has a triangle on its other side (along the trim it has none)
        needles = np.nonzero((area2 < thin * L2.max(axis=1)) & (cnt[lk] == 2))[0]
        if not len(needles):
            break
        owner = {}
        for t in np.nonzero(np.isin(inv.reshape(-1, 3), lk[needles]).any(axis=1))[0].tolist():
            for e in inv[3 * t:3 * t + 3].tolist():
                owner.setdefault(e, []).append(t)
        used = set()
        changed = False
        for t in needles.tolist():
            if t in used:
                continue
            e = int(longest[t])
            x, y, z = (int(v) for v in (T[t, e], T[t, (e + 1) % 3], T[t, (e + 2) % 3]))
            others = [j for j in owner.get(int(lk[t]), ()) if j != t]
            if len(others) != 1:
                continue
            if others[0] in used:               # its neighbour was just flipped: the next pass
                continue
            j = others[0]
            w = [v for v in T[j].tolist() if v not in (x, y)]
            if len(w) != 1:
                continue
            w = w[0]
            t1, t2 = (z, x, w), (z, w, y)
            (f1, u1), (f2, u2) = fat(t1), fat(t2)
            if u1 <= 0 or u2 <= 0 or min(f1, f2) <= fat(T[t].tolist())[0]:
                continue
            T[t], T[j] = t1, t2
            used.update((t, j))
            changed = True
        if not changed:
            break
    return T


def trimmed_mesh(brep, fi, E, angle_deg, chord_rel, diag, curv=False, edge_abs=0.0):
    """(P, N, T, uv, D) of a trimmed face: the kernel's coarse tessellation of
    the face alone (its trim sampled as finely as the angle tolerance asks),
    every triangle split in (u, v) as finely as it needs, neighbouring
    triangles meeting point for point along their shared edges."""
    single = _single_face(brep, fi)
    pb, _ = single.tessellate(chord_height_tolerance=max(chord_rel, 0.002) * diag,
                              curve_angle_tolerance_deg=angle_deg,
                              surface_angle_tolerance_deg=COARSE_ANGLE,
                              max_edge_length=4.0 * edge_abs,
                              allow_partial=True, return_failures=True)
    a = pb.to_numpy_arrays()
    P0 = np.asarray(a["points"], float)
    f = np.asarray(a["faces"])
    if not len(f):
        return None
    T0 = store._tri_array(f.reshape(-1, 4)[:, 1:] if (len(f) % 4 == 0 and np.all(f[0::4] == 3)) else
                          [f[i + 1:i + 1 + f[i]] for i in _poly_starts(f)])
    uv0 = E.feet(P0)
    # the kernel's mesh of a trimmed face can have folded triangles and thin
    # cracks: repaired as for the display mesh (cheap at this size), with the
    # exact normals turned to the kernel's winding
    n0, _ = surfeval.normals(E.at(uv0[:, 0], uv0[:, 1], 1))
    fn = np.cross(P0[T0[:, 1]] - P0[T0[:, 0]], P0[T0[:, 2]] - P0[T0[:, 0]])
    if float(np.sum(fn * (n0[T0[:, 0]] + n0[T0[:, 1]] + n0[T0[:, 2]]))) < 0:
        n0 = -n0
    try:
        T0 = store._tri_array(store._zip(P0, n0, T0, edge_pts=lambda: store._edge_samples(single),
                                         tol=4.0 * max(chord_rel, 0.002) * diag))
        T0 = store._tri_array(store._unfold(P0, n0, T0))
        T0 = _flip_needles(P0, n0, T0)
    except Exception as exc:
        print("NURBS analysis mesh: repair skipped:", exc)
    (u0, u1), (v0, v1) = E.u_domain, E.v_domain
    pu, pv = E.periodic()
    # corner (u, v) per triangle, unwrapped across a periodic seam
    C = uv0[T0]                                               # (m, 3, 2)
    for ax, per, lo, hi in ((0, pu, u0, u1), (1, pv, v0, v1)):
        if per:
            L = hi - lo
            ref = C[:, :1, ax]
            C[:, :, ax] -= L * np.round((C[:, :, ax] - ref) / L)
    # how finely: per triangle, its corner normals' spread over the angle
    # tolerance and its longest edge over the edge limit; an edge is split as
    # finely as the finer of its two triangles wants, a triangle as finely as
    # its finest edge
    Dc = E.at(C[:, :, 0].ravel(), C[:, :, 1].ravel(), 1)
    nc, _ = surfeval.normals(Dc)
    nc = nc.reshape(-1, 3, 3)
    cosn = np.stack([np.einsum("ij,ij->i", nc[:, x], nc[:, y]) for x, y in ((0, 1), (1, 2), (2, 0))], 1)
    want = np.arccos(np.clip(cosn.min(axis=1), -1.0, 1.0)) / math.radians(angle_deg)
    if edge_abs > 0 and float(cosn.min()) < 1.0 - 1e-12:              # (a flat face needs no split)
        Pc = P0[T0]
        L = np.linalg.norm(Pc - np.roll(Pc, -1, axis=1), axis=2)      # edges 0-1, 1-2, 2-0
        want = np.maximum(want, L.max(axis=1) / edge_abs)
    kt = np.clip(np.ceil(want - 1e-9), 1, MAX_SPLIT).astype(np.int64)
    m = len(T0)
    E3 = np.stack([T0[:, [0, 1]], T0[:, [1, 2]], T0[:, [2, 0]]], 1)     # (m, 3, 2)
    E3.sort(axis=2)
    keys, inv = np.unique(E3[:, :, 0] * (len(P0) + 1) + E3[:, :, 1], return_inverse=True)
    ke = np.zeros(len(keys), np.int64)
    np.maximum.at(ke, inv.ravel(), np.repeat(kt, 3))
    ke3 = ke[inv.reshape(m, 3)]                                       # (m, 3) pieces per edge
    kt = ke3.max(axis=1)
    pat = np.stack([kt, ke3[:, 0], ke3[:, 1], ke3[:, 2]], 1)
    pats, pinv = np.unique(pat, axis=0, return_inverse=True)
    pinv = pinv.ravel()
    inv3 = inv.reshape(m, 3)
    # every lattice point gets a key: a coarse corner its vertex, a point on a
    # coarse edge (edge, step from its lower-numbered end), an inner point its
    # own -- the triangles on both sides of an edge share its points
    NV, S1 = len(P0), MAX_SPLIT + 1
    inner = NV + len(keys) * S1
    Ks, UVs, Ts, base = [], [], [], 0
    for pi, (k, ka, kb, kc) in enumerate(pats.tolist()):
        sel = np.nonzero(pinv == pi)[0]
        W, Tl, kind, which, step = _lattice(int(k), int(ka), int(kb), int(kc))
        ms, L = len(sel), len(W)
        K = np.empty((ms, L), np.int64)
        cm = kind == 0
        K[:, cm] = T0[sel][:, which[cm]]
        for e, (x, y) in enumerate(((0, 1), (1, 2), (2, 0))):
            em = (kind == 1) & (which == e)
            if not em.any():
                continue
            g = inv3[sel, e]
            fwd = (T0[sel, x] < T0[sel, y])[:, None]
            j = step[em][None, :]
            K[:, em] = NV + g[:, None] * S1 + np.where(fwd, j, ke[g][:, None] - j)
        im = kind == 2
        K[:, im] = inner + np.arange(ms * int(im.sum())).reshape(ms, -1)
        inner += ms * int(im.sum())
        Ks.append(K.ravel())
        UVs.append(np.einsum("lc,mcd->mld", W, C[sel]).reshape(-1, 2))
        Ts.append((Tl[None, :, :] + (base + np.arange(ms) * L)[:, None, None]).reshape(-1, 3))
        base += ms * L
    _, first, vid = np.unique(np.concatenate(Ks), return_index=True, return_inverse=True)
    UV = np.vstack(UVs)[first]
    T = vid.ravel()[np.vstack(Ts)]
    for ax, per, lo, hi in ((0, pu, u0, u1), (1, pv, v0, v1)):
        if per:
            UV[:, ax] = lo + (UV[:, ax] - lo) % (hi - lo)
        else:
            UV[:, ax] = np.clip(UV[:, ax], lo, hi)
    D = E.at(UV[:, 0], UV[:, 1], 2 if curv else 1)
    n, ln = surfeval.normals(D)
    n = _fix_poles(E, UV[:, 0], UV[:, 1], D, n, ln)
    return D[0, 0], n, T, UV, D


def _poly_starts(f):
    i, out = 0, []
    while i < len(f):
        out.append(i)
        i += 1 + int(f[i])
    return out


# -- whole bodies --------------------------------------------------------------

def _orient(face, E):
    """+1 when su x sv is the face's outward normal, else -1. Asked at the
    best-defined of a few points of the domain (not a pole: the kernel's
    normal there means nothing); any (u, v) of the surface will do, the
    outward normal is the surface's turned to the face's side."""
    try:
        (u0, u1), (v0, v1) = E.u_domain, E.v_domain
        f = np.array([0.5, 0.3, 0.7, 0.15, 0.85])
        uu, vv = np.meshgrid(u0 + (u1 - u0) * f, v0 + (v1 - v0) * f, indexing="ij")
        n, ln = surfeval.normals(E.at(uu.ravel(), vv.ravel(), 1))
        k = int(np.argmax(ln))
        out = np.array(face.outward_normal_at_uv(float(uu.ravel()[k]), float(vv.ravel()[k])), float)
        return 1.0 if float(n[k] @ out) >= 0 else -1.0
    except Exception:
        return 1.0


def _wound(P, N, T):
    """The triangles wound like the normals (counter-clockwise seen from where
    the normals point), as a mesh of the face would be."""
    T = np.asarray(T, np.int64)
    if not len(T):
        return T
    S = T[::max(1, len(T) // 4000)]                     # one face's mesh is wound one way: a sample tells
    fn = np.cross(P[S[:, 1]] - P[S[:, 0]], P[S[:, 2]] - P[S[:, 0]])
    if float(np.sum(fn * (N[S[:, 0]] + N[S[:, 1]] + N[S[:, 2]]))) < 0:
        T = T[:, [0, 2, 1]]
    return T


def _curvatures(E, uv, D):
    """(K, H, k1, k2) at the vertices; where the first fundamental form
    degenerates (a pole) the limit, from a point a tiny step inside."""
    K, H, k1, k2, good = surfeval.curvatures(D)
    if not good.all():
        (u0, u1), (v0, v1) = E.u_domain, E.v_domain
        bad = np.nonzero(~good)[0]
        U, V = uv[bad, 0], uv[bad, 1]
        for step in (1e-4, 1e-3, 4e-3, 1e-2):
            Ds = E.at(U + (0.5 * (u0 + u1) - U) * step, V + (0.5 * (v0 + v1) - V) * step, 2)
            r = surfeval.curvatures(Ds)
            if r[4].all() or step == 1e-2:
                for arr, val in zip((K, H, k1, k2), r[:4]):
                    arr[bad] = val
                break
    return K, H, k1, k2


def _kernel_face(brep, fi, angle_deg, chord_rel, diag, curv):
    """The fallback for a face surfeval can't read (not a B-spline): the
    kernel's tessellation of the face alone, kernel normals, and the
    curvatures vertex by vertex (slow, but rare)."""
    from .diagnostics import curvatures_at
    from .square import _foot
    face = brep.faces()[fi]
    surf = face.surface()
    P, N, T = store._tessellate(_single_face(brep, fi), chord_rel, angle_deg, diag=diag)
    P, N, T = np.asarray(P, float), np.asarray(N, float), store._tri_array(T)
    uv = _foot(surf, np.array(face.interior_point(), float))
    _, su, sv = surf.derivatives(*uv)
    out = np.array(face.outward_normal_at_uv(*uv), float)
    sgn = 1.0 if float(np.cross(su, sv) @ out) >= 0 else -1.0
    k = int(np.argmin(np.linalg.norm(P - np.array(surf.evaluate(*uv)), axis=1)))
    if float(N[k] @ out) < 0:
        N = -N
    vals = None
    if curv:
        vals = []
        for w in store.surface_feet(surf, P).tolist():
            K, H, k1, k2 = curvatures_at(surf, *w)
            vals.append((K, -sgn * H, -sgn * k2, -sgn * k1))
        vals = np.array(vals, float).reshape(-1, 4)
    return P, N, T, vals


_prep = {}       # object -> (revision, per face (Surf or None, untrimmed, outward sign))


def _faces(brep, natural, diag, key):
    """Per face: its surface in numpy form (None: the kernel's fallback), whether
    it is untrimmed, and the sign that turns su x sv outwards. Kept per object
    and revision (key): the zebra, the curvature map and every quality setting
    share it."""
    if key is not None:
        hit = _prep.get(key[0])
        if hit and hit[0] == key[1]:
            return hit[1]
    faces = brep.faces()
    info = []
    for fi, face in enumerate(faces):
        try:
            E = surfeval.Surf(face.surface())
            whole = natural if (natural is not None and len(faces) == 1) else is_untrimmed(face, E, diag)
            info.append((E, bool(whole), _orient(face, E)))
        except Exception:
            info.append((None, False, 1.0))
    if key is not None:
        _prep[key[0]] = (key[1], info)
    return info


def body_mesh(brep, angle_deg, chord_rel, curv=False, natural=None, stats=None, max_edge_rel=MAX_EDGE_REL,
              key=None):
    """Analysis mesh of a whole body: (P, N, T, values) with values a dict
    {GAUSS, MEAN, MAX, MIN} per vertex when curv, else None; None when no face
    could be meshed. natural: True when the body is known to be one untrimmed
    face. key: (object id, revision) to keep the per-face set-up under. Faces
    surfeval can't describe, or that fail here, fall back to the kernel."""
    bb = brep.bounding_box()
    diag = float(np.linalg.norm(np.array(bb[1]) - np.array(bb[0]))) or 1.0
    chord_abs = chord_rel * diag
    edge_abs = max_edge_rel * diag
    Ps, Ns, Ts, Vs, base = [], [], [], [], 0
    for fi, (E, whole, sgn) in enumerate(_faces(brep, natural, diag, key)):
        res, kind = None, "kernel"
        if E is not None:
            try:
                if whole:
                    res, kind = grid_mesh(E, angle_deg, chord_abs, curv, edge_abs), "grid"
                else:
                    res, kind = trimmed_mesh(brep, fi, E, angle_deg, chord_rel, diag, curv, edge_abs), "trim"
            except Exception as exc:
                print(f"NURBS analysis mesh, face {fi}: {exc}")
                res = None
        if res is not None:
            P, N, T, uv, D = res
            N = N * sgn
            T = _wound(P, N, T)
            vals = None
            if curv:
                K, H, k1, k2 = _curvatures(E, uv, D)
                # outward normal: curvature towards the inside is positive (convex > 0)
                vals = np.stack([K, -sgn * H, -sgn * k2, -sgn * k1], 1)
        else:
            try:
                P, N, T, vals = _kernel_face(brep, fi, angle_deg, chord_rel, diag, curv)
                kind = "kernel"
            except Exception as exc:
                print(f"NURBS analysis mesh, face {fi} (kernel): {exc}")
                continue
        if stats is not None:
            stats[kind] = stats.get(kind, 0) + 1
        if not len(T):
            continue
        Ps.append(np.asarray(P, float))
        Ns.append(np.asarray(N, float))
        Ts.append(np.asarray(T, np.int64) + base)
        if curv:
            Vs.append(vals)
        base += len(P)
    if not Ps:
        return None
    vals = None
    if curv:
        V = np.vstack(Vs)
        vals = {k: V[:, i].astype(np.float32) for i, k in enumerate(CURV_KINDS)}
    return (np.vstack(Ps).astype(np.float32), np.vstack(Ns).astype(np.float32),
            np.vstack(Ts).astype(np.int32), vals)
