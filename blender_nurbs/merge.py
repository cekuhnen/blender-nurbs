"""Merge Patches: rebuild a joined set of faces whose outer boundary is four-
sided as ONE untrimmed NURBS surface.

Two routes:

* Grid (exact). The faces are untrimmed and form a rectangular grid of
  patches, each seam a full edge on both sides. Their nets are given one
  layout across every seam (degree raise + knot insertion) and joined end to
  end with a full-multiplicity knot at each seam: one surface identical to the
  pieces. How closely the result follows them is then the user's choice:
    EXACT   keep every knot (deviation 0, seams stay as knot lines)
    REDUCE  remove knots greedily while the surface stays within tolerance
    REFIT   fewest uniform spans (chosen degree) within tolerance
* Projected (any layout: trimmed faces, T-junctions). The outer loop is split
  into 4 sides at its corners, a Coons grid spans them in 3D, its points are
  snapped to the nearest face and fitted (always a refit).

Deviation is measured against the original faces on a dense grid.
"""

import numpy as np

from . import geom, store

MODES = ("EXACT", "REDUCE", "REFIT")


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


def _basis(U, p, ts, der=0):
    from .kernel.classa import basis
    return basis(U, p, ts, der)


# ---------------------------------------------------------------------------
# nets: (P homogeneous (nu, nv, 4), Uu, Uv, pu, pv), knots on 0..1
# ---------------------------------------------------------------------------

def _norm_knots(U):
    U = np.asarray(U, float)
    return (U - U[0]) / ((U[-1] - U[0]) or 1.0)


def _face_net(ext, surf):
    d = ext.surface_cvs(surf)
    nu, nv = int(d["count_u"]), int(d["count_v"])
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    W = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    H = np.concatenate([P * W[..., None], W[..., None]], axis=2)
    return {"P": H, "Uu": _norm_knots(d["knots_u"]), "Uv": _norm_knots(d["knots_v"]),
            "pu": int(d["degree_u"]), "pv": int(d["degree_v"])}


def _xyz(H):
    return H[..., :3] / H[..., 3:4]


def _orient(net, tr, fu, fv):
    P, Uu, Uv, pu, pv = net["P"], net["Uu"], net["Uv"], net["pu"], net["pv"]
    if tr:
        P, Uu, Uv, pu, pv = P.transpose(1, 0, 2), Uv, Uu, pv, pu
    if fu:
        P, Uu = P[::-1], 1.0 - Uu[::-1]
    if fv:
        P, Uv = P[:, ::-1], 1.0 - Uv[::-1]
    return {"P": P, "Uu": Uu, "Uv": Uv, "pu": pu, "pv": pv}


def _orientations(net):
    return [_orient(net, tr, fu, fv) for tr in (0, 1) for fu in (0, 1) for fv in (0, 1)]


def _side(net, s):
    """Boundary iso curve as 3D CVs along +v (u sides) or +u (v sides)."""
    X = _xyz(net["P"])
    return {"u0": X[0], "u1": X[-1], "v0": X[:, 0], "v1": X[:, -1]}[s]


def eval_net(net, su, sv):
    Bu = _basis(net["Uu"], net["pu"], su)
    Bv = _basis(net["Uv"], net["pv"], sv)
    H = np.einsum("ia,abk,jb->ijk", Bu, net["P"], Bv)
    return _xyz(H)


def _side_samples(net, s, m=9):
    t = np.linspace(0, 1, m)
    if s in ("u0", "u1"):
        return eval_net(net, [0.0 if s == "u0" else 1.0], t)[0]
    return eval_net(net, t, [0.0 if s == "v0" else 1.0])[:, 0]


# ---------------------------------------------------------------------------
# grid route
# ---------------------------------------------------------------------------

def _untrimmed(face, net, tol):
    """The face's edges are its surface's four boundary isos."""
    edges = face.edges()
    if len(edges) != 4:
        return False
    mids = []
    for e in edges:
        t0, t1 = e.parameter_range()
        mids.append(np.array(e.point_at(0.5 * (t0 + t1)), float))
    for s in ("u0", "u1", "v0", "v1"):
        q = _side_samples(net, s, 3)[1]
        if min(np.linalg.norm(q - m_) for m_ in mids) > tol:
            return False
    return True


NEXT = {"u1": ("u0", (1, 0)), "u0": ("u1", (-1, 0)), "v1": ("v0", (0, 1)), "v0": ("v1", (0, -1))}


def grid_layout(nets, tol):
    """{(i, j): oriented net} with every seam a full shared side, or None."""
    n = len(nets)
    cells = {(0, 0): nets[0]}
    where = {0: (0, 0)}
    todo = [0]
    while todo:
        a = todo.pop()
        ia, ja = where[a]
        na = cells[(ia, ja)]
        for s, (t, (di, dj)) in NEXT.items():
            S = _side_samples(na, s)
            for b in range(n):
                if b == a:
                    continue
                hit = None
                for nb in _orientations(nets[b]):
                    if np.linalg.norm(_side_samples(nb, t) - S, axis=1).max() <= tol:
                        hit = nb
                        break
                if hit is None:
                    continue
                cell = (ia + di, ja + dj)
                if b in where:
                    if where[b] != cell:
                        return None
                    continue
                if cell in cells:
                    return None
                cells[cell] = hit
                where[b] = cell
                todo.append(b)
    if len(where) != n:
        return None
    i0 = min(i for i, _ in cells)
    j0 = min(j for _, j in cells)
    cells = {(i - i0, j - j0): v for (i, j), v in cells.items()}
    I = 1 + max(i for i, _ in cells)
    J = 1 + max(j for _, j in cells)
    if len(cells) != I * J:
        return None
    return cells, I, J


def _speeds_u(net):
    """Mean |dS/du| at u=0 and u=1 over the v rows (on 0..1)."""
    from .geom import end_speeds
    X = _xyz(net["P"])
    s = [end_speeds(net["pu"], net["Uu"], X[:, j]) for j in range(X.shape[1])]
    return float(np.mean([a for a, _ in s])), float(np.mean([b for _, b in s]))


def _join_u(nets, tol):
    """Nets side by side along u (each u1 meets the next u0) -> one net."""
    from .geom import compatible_arrays, concat_arrays, speed_widths
    items = []
    for nt in nets:
        nu, nv, _ = nt["P"].shape
        items.append((nt["pv"], nt["Uv"], nt["P"].transpose(1, 0, 2).reshape(nv, nu * 4)))
    qv, Uv, arrs = compatible_arrays(items, 1e-9)
    nets2 = []
    for nt, A in zip(nets, arrs):
        nu = nt["P"].shape[0]
        P = A.reshape(len(A), nu, 4).transpose(1, 0, 2)
        nets2.append({"P": P, "Uu": nt["Uu"], "Uv": np.asarray(Uv), "pu": nt["pu"], "pv": qv})
    for a, b in zip(nets2[:-1], nets2[1:]):
        gap = float(np.linalg.norm(_xyz(a["P"][-1]) - _xyz(b["P"][0]), axis=1).max())
        if gap > tol:
            raise ValueError(f"seam CVs differ by {gap:.2e} (the two sides are parameterized differently)")
    widths = speed_widths([_speeds_u(nt) for nt in nets2])
    nv = len(Uv) - qv - 1
    qu, Uu, A = concat_arrays([(nt["pu"], nt["Uu"], nt["P"].reshape(nt["P"].shape[0], nv * 4)) for nt in nets2],
                              widths)
    return {"P": A.reshape(len(A), nv, 4), "Uu": np.asarray(Uu), "Uv": np.asarray(Uv), "pu": qu, "pv": qv}


def _transpose(net):
    return _orient(net, 1, 0, 0)


def merge_grid(cells, I, J, tol):
    rows = [_join_u([cells[(i, j)] for i in range(I)], tol) for j in range(J)]
    return _transpose(_join_u([_transpose(r) for r in rows], tol))


# ---------------------------------------------------------------------------
# accuracy: knot removal / refit, measured against the reference surface
#
# Edges: each outer side has a continuity k (-1 free, 0 G0, 1 G1, 2 G2). The
# fits keep the reference's derivatives across that side up to order k: the
# first k+1 CV rows from that side are solved from the derivative curves
# (exact in Reduce; fitted to the edge tolerance in Refit). Matching the
# original patches' own cross-boundary derivatives keeps whatever continuity
# they had with their neighbours.
# ---------------------------------------------------------------------------

SIDES = ("u0", "u1", "v0", "v1")
ANGLE_TOL = 0.1          # degrees, G1 sides
CURV_TOL = 0.02          # relative normal curvature across the edge, G2 sides


def eval_d(net, su, sv, du=0, dv=0):
    """d^(du+dv) S / du^du dv^dv on the grid su x sv (rational: Leibniz)."""
    from math import comb
    P = net["P"]
    Bu = [_basis(net["Uu"], net["pu"], su, k) for k in range(du + 1)]
    Bv = [_basis(net["Uv"], net["pv"], sv, k) for k in range(dv + 1)]
    if np.allclose(P[..., 3], 1.0):
        return np.einsum("ia,abk,jb->ijk", Bu[du], P[..., :3], Bv[dv])
    H = {(k, l): np.einsum("ia,abk,jb->ijk", Bu[k], P, Bv[l]) for k in range(du + 1) for l in range(dv + 1)}
    A = {kl: h[..., :3] for kl, h in H.items()}
    w = {kl: h[..., 3:4] for kl, h in H.items()}
    S = {}
    for k in range(du + 1):
        for l in range(dv + 1):
            v = A[(k, l)].copy()
            for j in range(1, l + 1):
                v -= comb(l, j) * w[(0, j)] * S[(k, l - j)]
            for i in range(1, k + 1):
                v -= comb(k, i) * w[(i, 0)] * S[(k - i, l)]
                for j in range(1, l + 1):
                    v -= comb(k, i) * comb(l, j) * w[(i, j)] * S[(k - i, l - j)]
            S[(k, l)] = v / w[(0, 0)]
    return S[(du, dv)]


def _dense_params(U, per_span=8):
    br = sorted(set(float(k) for k in U))
    t = [br[0]]
    for a, b in zip(br[:-1], br[1:]):
        t += list(np.linspace(a, b, per_span + 1)[1:])
    return np.array(t)


def fit1d(t, D, U, p, start=(), end=()):
    """Least squares B-spline (knots U, degree p) through data D (m, d) at t,
    with the derivatives start[r] / end[r] (order r) held exactly at the ends.
    When the curve has too few CVs to hold them all, the highest orders are
    dropped (the edge checks decide whether the result is good enough)."""
    U = np.asarray(U, float)
    n = len(U) - p - 1
    start, end = list(start), list(end)
    while len(start) + len(end) > n:
        (start if len(start) >= len(end) else end).pop()
    D = np.asarray(D, float)
    P = np.zeros((n, D.shape[1]))
    fixed = np.zeros(n, bool)
    for r, Dr in enumerate(start):
        b = _basis(U, p, [U[0]], r)[0]
        P[r] = (Dr - b[:r] @ P[:r]) / b[r]
        fixed[r] = True
    for r, Dr in enumerate(end):
        b = _basis(U, p, [U[-1]], r)[0]
        i = n - 1 - r
        P[i] = (Dr - b[i + 1:] @ P[i + 1:]) / b[i]
        fixed[i] = True
    A = _basis(U, p, t)
    free = ~fixed
    if free.any():
        rhs = D - A[:, fixed] @ P[fixed]
        P[free] = np.linalg.lstsq(A[:, free], rhs, rcond=None)[0]
    res = float(np.abs(A @ P - D).max()) if len(D) else 0.0
    return P, res


def _end_derivs(U, p, C, k_start, k_end):
    """Derivatives (orders 0..k) of the curve with CVs C at its two ends."""
    U = np.asarray(U, float)
    s = [(_basis(U, p, [U[0]], r)[0] @ C) for r in range(k_start + 1)]
    e = [(_basis(U, p, [U[-1]], r)[0] @ C) for r in range(k_end + 1)]
    return s, e


def _side_arrays(new, ref, cont, t, kmin=0.0):
    """Per constrained side, per sample t along it: position gap, normal angle
    (deg) and relative normal-curvature difference across the edge."""
    t = np.asarray(t, float)
    out = {}
    for side in SIDES:
        k = cont.get(side, -1)
        if k < 0:
            continue
        across_u = side[0] == "u"
        c = [0.0 if side[1] == "0" else 1.0]
        args = (c, t) if across_u else (t, c)

        def frame(f):
            S = f(*args, 0, 0).reshape(-1, 3)
            Su = f(*args, 1, 0).reshape(-1, 3)
            Sv = f(*args, 0, 1).reshape(-1, 3)
            Scc = f(*args, 2, 0).reshape(-1, 3) if across_u else f(*args, 0, 2).reshape(-1, 3)
            Sc = Su if across_u else Sv
            n = np.cross(Su, Sv)
            n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-300)
            kap = np.einsum("ij,ij->i", Scc, n) / np.einsum("ij,ij->i", Sc, Sc).clip(1e-300)
            return S, n, kap
        S1, n1, k1 = frame(lambda a, b, du, dv: eval_d(new, a, b, du, dv))
        S0, n0, k0 = frame(ref)
        e = {"pos": np.linalg.norm(S1 - S0, axis=1)}
        if k >= 1:
            e["angle"] = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n0, n1), -1, 1)))
        if k >= 2:
            scale = max(float(np.abs(k0).max()), kmin, 1e-12)
            e["curv"] = np.abs(k1 - k0) / scale
        out[side] = e
    return out


def side_errors(new, ref, cont, m=41, kmin=0.0):
    """Per constrained side: max position gap, normal angle (deg) and relative
    normal-curvature difference across the edge. ref(su, sv, du, dv)."""
    arr = _side_arrays(new, ref, cont, np.linspace(0, 1, m), kmin)
    return {s: {k: float(v.max()) for k, v in e.items()} for s, e in arr.items()}


def edges_ok(errs, edge_tol):
    for e in errs.values():
        if e["pos"] > edge_tol or e.get("angle", 0.0) > ANGLE_TOL or e.get("curv", 0.0) > CURV_TOL:
            return False
    return True


def _worst(errs):
    w = {"pos": 0.0, "angle": 0.0, "curv": 0.0}
    for e in errs.values():
        for k in w:
            w[k] = max(w[k], e.get(k, 0.0))
    return w


def _fit_dir(net, Unew, along, cont=None):
    """Least squares of the net's CV curves (along u or v) on knots Unew,
    keeping their end derivatives up to the continuity of the two sides the
    curves run into (so those edges and their cross derivatives stay exact)."""
    cont = cont or {}
    if along == "v":
        tc = {"u0": cont.get("v0", -1), "u1": cont.get("v1", -1)}
        return _transpose(_fit_dir(_transpose(net), Unew, "u", tc))
    p, U, P = net["pu"], net["Uu"], net["P"]
    nu, nv, _ = P.shape
    C = P.reshape(nu, nv * 4)
    t = _dense_params(U, 2 * p + 2)
    D = _basis(U, p, t) @ C
    s, e = _end_derivs(U, p, C, max(0, cont.get("u0", -1)), max(0, cont.get("u1", -1)))
    Q, _ = fit1d(t, D, Unew, p, s, e)
    return {"P": Q.reshape(len(Q), nv, 4), "Uu": np.asarray(Unew), "Uv": net["Uv"], "pu": p, "pv": net["pv"]}


def reduce_knots(net, ref, surf_err, tol, cont, edge_tol, kmin=0.0, max_iter=400):
    """Greedy knot removal: each round removes the one knot (u or v) whose
    removal keeps the surface closest to the reference, while the surface
    stays within tol and the constrained edges within their checks."""
    cur, err = net, 0.0
    for _ in range(max_iter):
        best = None
        for along in ("u", "v"):
            U = cur["Uu"] if along == "u" else cur["Uv"]
            p = cur["pu"] if along == "u" else cur["pv"]
            for k in sorted(set(float(x) for x in U[p + 1:len(U) - p - 1])):
                idx = int(np.where(np.isclose(U, k))[0][-1])
                cand = _fit_dir(cur, np.delete(U, idx), along, cont)
                e = surf_err(cand)
                if e > tol or (best is not None and e >= best[0]):
                    continue
                if not edges_ok(side_errors(cand, ref, cont, kmin=kmin), edge_tol):
                    continue
                best = (e, cand)
        if best is None:
            break
        err, cur = best
    return cur, err


def fit_surface(ref, p, Uu, Uv, cont, m):
    """Tensor-product least squares of the reference on knots Uu x Uv (degree
    p), keeping each constrained side's derivative curves (two passes: along v
    for every sample row and for the u-derivative rows at the u ends, then
    along u)."""
    t = np.linspace(0, 1, m)
    kv0, kv1 = max(-1, cont.get("v0", -1)), max(-1, cont.get("v1", -1))
    ku0, ku1 = max(-1, cont.get("u0", -1)), max(-1, cont.get("u1", -1))

    def along_v(us, du):
        G = ref(us, t, du, 0)
        st = [ref(us, [0.0], du, s_)[:, 0] for s_ in range(kv0 + 1)]
        en = [ref(us, [1.0], du, s_)[:, 0] for s_ in range(kv1 + 1)]
        rows, res = [], 0.0
        for i in range(len(us)):
            Y, r = fit1d(t, G[i], Uv, p, [a[i] for a in st], [a[i] for a in en])
            rows.append(Y)
            res = max(res, r) if du == 0 else res
        return np.array(rows), res
    Y, res_v = along_v(t, 0)
    nv = Y.shape[1]
    Ys = [along_v([0.0], r)[0][0].ravel() for r in range(ku0 + 1)]
    Ye = [along_v([1.0], r)[0][0].ravel() for r in range(ku1 + 1)]
    X, res_u = fit1d(t, Y.reshape(m, -1), Uu, p, Ys, Ye)
    X = X.reshape(len(X), nv, 3)
    H = np.concatenate([X, np.ones(X.shape[:2] + (1,))], axis=2)
    return {"P": H, "Uu": np.asarray(Uu), "Uv": np.asarray(Uv), "pu": p, "pv": p}, res_u, res_v


def seam_report(net, kmin=0.0, m=21):
    """Continuity across the inner knot lines where the net can break G2 (knot
    multiplicity >= degree - 1: the patch seams, and any sharp line inside a
    patch). Per line: direction ("u": the line u = t), t, normal angle (deg),
    relative curvature jump across it, and kind: KINK (tangent break), G1
    (tangent but curvature jumps) or G2."""
    out = []
    eps = 1e-9
    s = np.linspace(0.0, 1.0, m)
    for d, U, p in (("u", net["Uu"], net["pu"]), ("v", net["Uv"], net["pv"])):
        U = np.round(np.asarray(U, float), 12)
        for k in sorted(set(U[p + 1:len(U) - p - 1])):
            if int(np.sum(U == k)) < max(1, p - 1):
                continue

            def frame(c):
                a = ([c], s) if d == "u" else (s, [c])
                Su, Sv = eval_d(net, *a, 1, 0).reshape(-1, 3), eval_d(net, *a, 0, 1).reshape(-1, 3)
                Scc = (eval_d(net, *a, 2, 0) if d == "u" else eval_d(net, *a, 0, 2)).reshape(-1, 3)
                Sc = Su if d == "u" else Sv
                n = np.cross(Su, Sv)
                ln = np.linalg.norm(n, axis=1)
                n = n / ln.clip(1e-300)[:, None]
                kap = np.einsum("ij,ij->i", Scc, n) / np.einsum("ij,ij->i", Sc, Sc).clip(1e-300)
                return n, kap, ln
            n0, k0, l0 = frame(k - eps)
            n1, k1, l1 = frame(k + eps)
            ok = (l0 > 1e-12 * max(l0.max(), 1e-300)) & (l1 > 1e-12 * max(l1.max(), 1e-300))   # skip poles
            if not ok.any():
                continue
            ang = float(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n0, n1)[ok], -1, 1))).max())
            scale = max(float(np.abs(np.r_[k0[ok], k1[ok]]).max()), kmin, 1e-12)
            cj = float(np.abs(k1 - k0)[ok].max() / scale)
            kind = "KINK" if ang > ANGLE_TOL else ("G1" if cj > CURV_TOL else "G2")
            out.append({"dir": d, "t": float(k), "angle": ang, "curv": cj, "kind": kind})
    return out


def seam_knots(report, p):
    """Knots Refit keeps for the seams that are not G2: (u list, v list) of
    (t, multiplicity): degree p at a kink (C0), p - 1 where only the
    curvature jumps (C1)."""
    out = ([], [])
    for r in report:
        if r["kind"] == "G2":
            continue
        mult = p if r["kind"] == "KINK" else max(1, p - 1)
        out[0 if r["dir"] == "u" else 1].append((r["t"], mult))
    return out


def _with_seams(U, p, sm):
    """Knot vector U with the seam knots sm [(t, mult)] in it: each seam takes
    the place of the nearest inner knot (so the span count stays) with its
    multiplicity. U unchanged when it has too few inner knots."""
    if not sm:
        return U
    inner = _inner(U)
    if len(inner) < len(sm):
        return U
    mult = {}
    for k, mu in sm:
        i = min((i for i in range(len(inner)) if i not in mult), key=lambda i: abs(inner[i] - k))
        inner[i] = k
        mult[i] = mu
    full = sorted(x for i, k in enumerate(inner) for x in [k] * mult.get(i, 1))
    return _knots(p, full)


def _inner(U):
    """Distinct inner knots of a knot vector (on 0..1)."""
    k = sorted(set(np.round(np.asarray(U, float), 12)))
    return list(k[1:-1])


def _knots(p, inner):
    return np.r_[[0.0] * (p + 1), np.asarray(inner, float), [1.0] * (p + 1)]


def _uniform(p, spans):
    return np.asarray(geom.clamped_uniform_knots(p + spans, p), float)


KNOT_PLACEMENTS = ("ADAPTIVE", "UNIFORM")
PLACE_TRIES = 3          # knot re-placements per span count before adding a span


def refit_net(ref, degree, spans_u, spans_v, tol, cont, edge_tol, m=None, surf_err=None, max_spans=40, kmin=0.0,
              knots="ADAPTIVE", breaks=None, grid_only=False, seams=None):
    """Fewest spans (per direction) of the chosen degree with the surface
    within tol and the constrained edges within their checks, or the spans
    given. Grows the direction that fails until both hold.
    knots: UNIFORM keeps even spans; ADAPTIVE moves the knots to where the
    error is (surface and edge checks, each relative to its limit) before
    each extra span, and keeps the best placement it found. breaks: the
    reference's own inner knots (u, v), tried as the knots when their count
    fits (the patches' knot lines are where their shape changes).
    grid_only: ref is only known at the m x m sample parameters (a projected
    grid), so every check samples there. seams: knots kept in every knot
    vector tried, (u list, v list) of (t, multiplicity) (see seam_knots);
    a direction whose spans are set too few for its seams smooths them."""
    p = int(degree)
    m = m or max(41, 6 * (p + 1))
    t = np.linspace(0, 1, m)
    G = ref(t, t, 0, 0)
    tt = t if grid_only else np.linspace(0, 1, 2 * m - 1)
    Gd = ref(tt, tt, 0, 0)
    edge_m = m if grid_only else 41
    if surf_err is None:
        surf_err = lambda nt: float(np.linalg.norm(eval_net(nt, tt, tt) - Gd, axis=2).max())
    adaptive = knots != "UNIFORM"

    breaks = breaks or (None, None)
    seams = [list(x) for x in (seams or ([], []))]
    for d, fixed in ((0, spans_u), (1, spans_v)):
        if fixed and int(fixed) < len(seams[d]) + 1:
            seams[d] = []
    fix = lambda U, d: _with_seams(U, p, seams[d])

    def fit1(B, U):              # unconstrained 1D fit of all rows: error per sample
        A = _basis(U, p, t)
        X = np.linalg.lstsq(A, B, rcond=None)[0]
        return np.linalg.norm((A @ X - B).reshape(m, -1, 3), axis=2).max(axis=1)

    def from_breaks(brk, spans):
        if brk is None or len(brk) != spans - 1:
            return None
        return _knots(p, brk)

    def candidates(U, spans, err, brk):
        """Knot vectors to try at this span count: the reference's own knots
        when they fit, then the current ones moved to where the error is."""
        out = []
        Ub = from_breaks(brk, spans)
        if Ub is not None:
            out.append(Ub)
        if spans > 1 and err is not None:
            out.append(place(U, spans, err))
        return out

    def guess(rows, fixed, brk, d):
        """Fewest spans whose 1D fit of the rows stays within half the
        tolerance, and the knots that got there."""
        B = np.hstack(rows)
        n_s = len(seams[d])
        lo, hi = (int(fixed), int(fixed)) if fixed else (n_s + 1, max(n_s + 1, m // 2 - p))
        best = None
        for s_ in range(lo, hi + 1):
            U = fix(_uniform(p, s_), d)
            e = fit1(B, U)
            if adaptive:
                for _ in range(PLACE_TRIES):
                    for Uc in candidates(U, s_, e, brk):
                        Uc = fix(Uc, d)
                        ec = fit1(B, Uc)
                        if ec.max() < e.max():
                            U, e = Uc, ec
            if best is None or e.max() < best[2]:
                best = (s_, U, float(e.max()))
            if e.max() <= 0.5 * tol:
                return s_, U
        return best[0], (best[1] if fixed else fix(_uniform(p, best[0]), d))

    def profiles(net):
        """Error along u and along v, each sample relative to its limit."""
        E = np.linalg.norm(eval_net(net, tt, tt) - Gd, axis=2) / tol
        eu, ev = E.max(axis=1), E.max(axis=0)
        lim = {"pos": edge_tol, "angle": ANGLE_TOL, "curv": CURV_TOL}
        for side, e in _side_arrays(net, ref, cont, tt, kmin).items():
            r = np.max([v / lim[k] for k, v in e.items()], axis=0)
            if side[0] == "v":           # v = const: the edge runs along u
                eu = np.maximum(eu, r)
            else:
                ev = np.maximum(ev, r)
        return eu, ev

    def place(U, spans, err):
        from .kernel.classa import equidistribute
        ts = t if len(err) == len(t) else tt
        return _knots(p, equidistribute(ts, err, U, p, spans))

    def run(Uu, Uv):
        Uu, Uv = fix(Uu, 0), fix(Uv, 1)
        net, res_u, res_v = fit_surface(ref, p, Uu, Uv, cont, m)
        dev = surf_err(net)
        errs = side_errors(net, ref, cont, m=edge_m, kmin=kmin)
        score = max([dev / tol] + [max(e["pos"] / edge_tol, e.get("angle", 0.0) / ANGLE_TOL,
                                       e.get("curv", 0.0) / CURV_TOL) for e in errs.values()])
        return {"net": net, "res": (res_u, res_v), "dev": dev, "errs": errs, "score": score, "U": (Uu, Uv)}

    sv, Uv0 = guess([G[i] for i in range(m)], spans_v, breaks[1], 1)
    su, Uu0 = guess([G[:, j] for j in range(m)], spans_u, breaks[0], 0)
    cur = run(Uu0, Uv0)
    best_here, tries, best = cur, 0, cur
    while True:
        net, dev, errs = cur["net"], cur["dev"], cur["errs"]
        res_u, res_v = cur["res"]
        bad_u = [s for s in ("v0", "v1") if s in errs and not edges_ok({s: errs[s]}, edge_tol)]   # curves along u
        bad_v = [s for s in ("u0", "u1") if s in errs and not edges_ok({s: errs[s]}, edge_tol)]   # curves along v
        if dev <= tol and not bad_u and not bad_v:
            best = cur
            break
        if cur["score"] < best_here["score"]:
            best_here = cur
        if cur["score"] < best["score"]:
            best = cur
        if adaptive and tries < PLACE_TRIES:     # same spans, knots moved to where the error is
            tries += 1
            eu, ev = profiles(best_here["net"])
            Uu, Uv = best_here["U"]
            nUu = Uu if su == 1 else place(Uu, su, eu)
            nUv = Uv if sv == 1 else place(Uv, sv, ev)
            if tries == 1:                        # first retry: the reference's knot lines, when they fit
                Bu, Bv = from_breaks(breaks[0], su), from_breaks(breaks[1], sv)
                if Bu is not None or Bv is not None:
                    nUu = Bu if Bu is not None else Uu
                    nUv = Bv if Bv is not None else Uv
            nUu, nUv = fix(nUu, 0), fix(nUv, 1)
            if len(nUu) == len(Uu) and len(nUv) == len(Uv) and np.allclose(nUu, Uu, atol=1e-6) \
                    and np.allclose(nUv, Uv, atol=1e-6):
                tries = PLACE_TRIES
            else:
                cur = run(nUu, nUv)
                continue
        grow_u = not spans_u and su < max_spans
        grow_v = not spans_v and sv < max_spans
        # grow the direction at fault; when that one is fixed by the user, stop
        # (more spans the other way cannot help)
        res_u, res_v = best_here["res"]
        dev, errs = best_here["dev"], best_here["errs"]
        bad_u = [s for s in ("v0", "v1") if s in errs and not edges_ok({s: errs[s]}, edge_tol)]
        bad_v = [s for s in ("u0", "u1") if s in errs and not edges_ok({s: errs[s]}, edge_tol)]
        if dev > tol:
            want_u = res_u >= res_v
        else:
            want_u = len(bad_u) >= len(bad_v)
        if want_u and grow_u:
            su += 1
        elif not want_u and grow_v:
            sv += 1
        else:
            break
        Uu, Uv = best_here["U"]
        if adaptive:
            eu, ev = profiles(best_here["net"])
            if want_u:
                Uu = from_breaks(breaks[0], su)
                Uu = place(best_here["U"][0], su, eu) if Uu is None else Uu
            else:
                Uv = from_breaks(breaks[1], sv)
                Uv = place(best_here["U"][1], sv, ev) if Uv is None else Uv
        else:
            Uu, Uv = _uniform(p, su), _uniform(p, sv)
        cur = run(Uu, Uv)
        best_here, tries = cur, 0
    su, sv = len(set(np.round(best["U"][0], 12))) - 1, len(set(np.round(best["U"][1], 12))) - 1
    return best["net"], (su, sv), best["dev"], best["errs"]


# ---------------------------------------------------------------------------
# projected route (any layout)
# ---------------------------------------------------------------------------

def _outer_sides(brep, size):
    from .refit import _edge_polyline, _chain, _four_sides
    outer = [e for e in brep.edges() if len(e.faces()) == 1]
    if not outer:
        raise RuntimeError("the faces have no open boundary")
    polys = [_edge_polyline(e) for e in outer]
    try:
        loop = _chain(polys, 1e-3 * size)
    except RuntimeError:
        raise RuntimeError("the open boundary of the faces is not one loop (Merge Patches needs a "
                           "four-sided outer boundary)") from None
    try:
        return _four_sides(loop)
    except RuntimeError as exc:
        raise RuntimeError(str(exc).replace("Refit to Trim", "Merge Patches").replace("trim boundary",
                           "outer boundary")) from None


def projected_grid(brep, size, m=61, smooth=3):
    """Coons grid over the 4 outer sides, snapped onto the nearest face."""
    from .refit import _resample
    s0, s1, s2, s3 = _outer_sides(brep, size)
    B, R, T, L = _resample(s0, m), _resample(s1, m), _resample(s2[::-1], m), _resample(s3[::-1], m)
    s = np.linspace(0, 1, m)
    c00, c10, c01, c11 = B[0], B[-1], T[0], T[-1]
    G = ((1 - s)[None, :, None] * B[:, None] + s[None, :, None] * T[:, None]
         + (1 - s)[:, None, None] * L[None] + s[:, None, None] * R[None]
         - ((1 - s)[:, None, None] * (1 - s)[None, :, None] * c00 + s[:, None, None] * (1 - s)[None, :, None] * c10
            + (1 - s)[:, None, None] * s[None, :, None] * c01 + s[:, None, None] * s[None, :, None] * c11))
    faces = brep.faces()

    def snap(G):
        out = G.copy()
        for i in range(1, m - 1):
            for j in range(1, m - 1):
                q = tuple(float(x) for x in G[i, j])
                best = min((f.closest_point(q) for f in faces), key=lambda r: r[1])
                out[i, j] = best[0]
        return out
    G = snap(G)
    for _ in range(smooth):            # even out the spacing the snap pulled together, then snap back
        H = G.copy()
        H[1:-1, 1:-1] = 0.25 * (G[2:, 1:-1] + G[:-2, 1:-1] + G[1:-1, 2:] + G[1:-1, :-2])
        G = snap(H)
    return G


def _grid_eval(G):
    """Bilinear lookup in a dense (m x m) sample grid on 0..1 x 0..1."""
    m = G.shape[0]

    def ev(su, sv):
        su, sv = np.asarray(su, float), np.asarray(sv, float)
        fu, fv = su * (m - 1), sv * (m - 1)
        iu, iv = np.clip(fu.astype(int), 0, m - 2), np.clip(fv.astype(int), 0, m - 2)
        au, av = (fu - iu)[:, None, None], (fv - iv)[None, :, None]
        g = lambda a, b: G[np.ix_(a, b)]
        return ((1 - au) * (1 - av) * g(iu, iv) + au * (1 - av) * g(iu + 1, iv)
                + (1 - au) * av * g(iu, iv + 1) + au * av * g(iu + 1, iv + 1))
    return ev


def _deviation_to_faces(net, faces, m=25):
    t = np.linspace(0, 1, m)
    X = eval_net(net, t, t).reshape(-1, 3)
    return max(min(f.closest_point(tuple(float(c) for c in q))[1] for f in faces) for q in X)


# ---------------------------------------------------------------------------

def _cont_dict(cont):
    """cont: one value for every side or {side: value}; values -1/0/1/2 or
    'FREE'/'G0'/'G1'/'G2'."""
    conv = lambda c: c if isinstance(c, int) else {"FREE": -1, "G0": 0, "G1": 1, "G2": 2}[str(c).upper()]
    if isinstance(cont, dict):
        return {s: conv(cont.get(s, -1)) for s in SIDES}
    return {s: conv(cont) for s in SIDES}


def merge_text(text, mode="REDUCE", tol=0.0, degree=3, spans_u=0, spans_v=0, tol_rel=1e-4, cont="G2", edge_tol=0.0,
               knots="ADAPTIVE", seams="KEEP"):
    """Brep text of the single merged surface. Returns (text, info).
    tol: absolute tolerance (0 = tol_rel x size); cont: edge continuity kept
    (per side or one for all); edge_tol: edge position tolerance (0 = tol/10)."""
    _, ext = _k()
    brep = store.brep_from_text(text)
    faces = brep.faces()
    if len(faces) < 1:
        raise RuntimeError("nothing to merge")
    lo, hi = brep.bounding_box()
    size = float(np.linalg.norm(np.subtract(hi, lo))) or 1.0
    tol = float(tol) if tol and tol > 0 else tol_rel * size
    edge_tol = float(edge_tol) if edge_tol and edge_tol > 0 else 0.1 * tol
    cont = _cont_dict(cont)
    kmin = 1.0 / size          # curvature differences are relative to this at least (flat edges)
    seam_tol = 1e-6 * size
    nets = [_face_net(ext, f.surface()) for f in faces]
    route, layout, note = "projected", None, ""
    if all(_untrimmed(f, n_, 1e-5 * size) for f, n_ in zip(faces, nets)):
        layout = grid_layout(nets, 1e-5 * size)
        if layout is None:
            note = "faces are not a full grid of patches"
    else:
        note = "some faces are trimmed"
    merged = None
    if layout is not None:
        cells, I, J = layout
        try:
            merged = merge_grid(cells, I, J, seam_tol * 10)
            route = "grid"
        except ValueError as exc:
            note = str(exc)
    info = {"route": route, "size": size, "tolerance": tol, "edge_tolerance": edge_tol, "faces": len(faces),
            "note": note, "continuity": dict(cont)}
    errs = {}
    if merged is not None:
        info["grid"] = (I, J)
        ref = lambda a, b, du=0, dv=0: eval_d(merged, a, b, du, dv)
        su_ref = _dense_params(merged["Uu"], 6)
        sv_ref = _dense_params(merged["Uv"], 6)
        Rpts = eval_net(merged, su_ref, sv_ref)
        surf_err = lambda nt: float(np.linalg.norm(eval_net(nt, su_ref, sv_ref) - Rpts, axis=2).max())
        info["exact_cvs"] = merged["P"].shape[:2]
        info["seams"] = seam_report(merged, kmin)
        keep = seam_knots(info["seams"], int(degree)) if seams == "KEEP" else ([], [])
        info["seams_kept"] = sum(len(x) for x in keep) if mode == "REFIT" else 0
        if mode == "EXACT":
            out, dev = merged, 0.0
            errs = {s: {"pos": 0.0} for s in SIDES if cont[s] >= 0}
        elif mode == "REDUCE":
            out, dev = reduce_knots(merged, ref, surf_err, tol, cont, edge_tol, kmin)
            errs = side_errors(out, ref, cont, kmin=kmin)
        else:
            out, spans, dev, errs = refit_net(ref, degree, spans_u, spans_v, tol, cont, edge_tol, surf_err=surf_err,
                                              kmin=kmin, knots=knots,
                                              breaks=(_inner(merged["Uu"]), _inner(merged["Uv"])), seams=keep)
            fits = [not sp or sp >= len(k_) + 1 for sp, k_ in ((spans_u, keep[0]), (spans_v, keep[1]))]
            info["seams_kept"] = sum(len(k_) for f_, k_ in zip(fits, keep) if f_)
            if info["seams_kept"] < sum(len(x) for x in keep):
                info["note"] = (info["note"] + "; " if info["note"] else "") + "spans too few to keep the seams sharp"
            info["spans"] = spans
    else:
        G = projected_grid(brep, size)
        m = G.shape[0]
        t = np.linspace(0, 1, m)

        def ref(a, b, du=0, dv=0):            # the sample grid: positions at its own parameters only
            if du or dv:
                raise ValueError("no derivatives on a projected grid")
            ia = np.rint(np.asarray(a, float) * (m - 1)).astype(int)
            ib = np.rint(np.asarray(b, float) * (m - 1)).astype(int)
            return G[np.ix_(ia, ib)]
        if any(c > 0 for c in cont.values()):
            note = (note + "; " if note else "") + "edges kept at G0 (no tangents on a projected grid)"
            info["note"] = note
        cont0 = {s: min(c, 0) for s, c in cont.items()}
        info["continuity"] = dict(cont0)
        surf_err = lambda nt: float(np.linalg.norm(eval_net(nt, t, t) - G, axis=2).max())

        def ref_pos(a, b, du=0, dv=0):        # side_errors also asks for derivatives: none to compare
            return ref(a, b) if not (du or dv) else np.zeros((len(a), len(b), 3))
        out, spans, dev, errs = refit_net(ref_pos, degree, spans_u, spans_v, tol, cont0, edge_tol, m=m,
                                          surf_err=surf_err, knots=knots, grid_only=True)
        errs = {s: {"pos": e["pos"]} for s, e in errs.items()}
        info["spans"] = spans
        info["mode_used"] = "REFIT"
        dev = max(dev, _deviation_to_faces(out, faces))
    if merged is not None and mode != "EXACT":
        dev = max(dev, _deviation_to_faces(out, faces, 17))
    P, nu, nv = out["P"], out["P"].shape[0], out["P"].shape[1]
    X = _xyz(P)
    W = P[..., 3]
    pts = [tuple(map(float, X[i, j])) for i in range(nu) for j in range(nv)]
    wts = None if np.allclose(W, 1.0) else [float(W[i, j]) for i in range(nu) for j in range(nv)]
    s = ext.surface_from_cvs(int(out["pu"]), int(out["pv"]), pts, [float(k) for k in out["Uu"]],
                             [float(k) for k in out["Uv"]], wts)
    info.update({"cvs": (nu, nv), "degree": (int(out["pu"]), int(out["pv"])), "deviation": float(dev),
                 "knots_u": len(set(np.round(out["Uu"], 12))) - 2, "knots_v": len(set(np.round(out["Uv"], 12))) - 2,
                 "edges": errs, "edge_worst": _worst(errs), "edges_ok": edges_ok(errs, edge_tol)})
    return store.brep_to_text(ext.brep_from_surface(s)), info


LAST_INFO = {}

CONT_KEYS = {s: f"merge_cont_{s}" for s in SIDES}


def params_cont(p):
    return {s: p.get(CONT_KEYS[s], "G2") for s in SIDES}


def apply_merge(obj, text, inputs, p):
    out, info = merge_text(text, p.get("merge_mode", "REDUCE"), float(p.get("merge_tolerance", 0.0)),
                           int(p.get("merge_degree", 3)), int(p.get("merge_spans_u", 0)),
                           int(p.get("merge_spans_v", 0)), cont=params_cont(p),
                           edge_tol=float(p.get("merge_edge_tolerance", 0.0)),
                           knots=p.get("merge_knots", "ADAPTIVE"), seams=p.get("merge_seams", "KEEP"))
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return out


def _register_step():
    from . import steps
    steps.APPLY["merge_patches"] = apply_merge
    steps.LABEL["merge_patches"] = lambda p: f"Merge Patches · {p.get('merge_mode', 'REDUCE').title()}"


_register_step()
