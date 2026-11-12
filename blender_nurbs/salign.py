"""Surface Align (Alias "Align" for surfaces): make one or more boundary edges
of an untrimmed surface G0 / G1 / G2 to the neighbouring surface edge (or G0
to a curve), by moving only the first 1 / 2 / 3 CV rows at each edge.

Per edge, with the edge at a = 0 of the cross direction (degree p, clamped
knots Ua) and CV rows R0, R1, R2 along it (basis Ne along the edge):
    S(0, e)   = Ne R0
    Sa(0, e)  = c1 Ne (R1 - R0)                          c1 = p / d1
    Saa(0, e) = c2 Ne ((R2 - R1) / d2 - (R1 - R0) / d1)  c2 = p (p-1) / (Ua[p+1] - Ua[2])
    d1 = Ua[p+1] - Ua[1],  d2 = Ua[p+2] - Ua[2]
so each condition is a linear least-squares fit of one row (end CVs exact).

  G0  the edge runs along the neighbour edge (matched by arc length)
  G1  the cross tangent turns into the neighbour's tangent plane, pointing
      away from it; its length and its along-edge part are kept (x Tangent)
  G2  the normal part of the cross second derivative gives the neighbour's
      normal curvature in that direction; its tangential part is kept
When the surface has too few rows for the conditions at both ends of a
direction, knots are inserted first (the shape does not change).

Recorded as a history step ("surface_align"): moving the neighbour realigns.
"""

import numpy as np
from mathutils import Vector

from . import features, geom, store

EDGES = ("U0", "U1", "V0", "V1")
CONT = {"G0": 0, "G1": 1, "G2": 2}


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


# --------------------------------------------------------------------------
# B-spline helpers
# --------------------------------------------------------------------------

def basis_matrix(U, p, n, ts):
    """(len(ts), n) basis values of a clamped B-spline with n CVs."""
    A = np.zeros((len(ts), n))
    for r, t in enumerate(ts):
        s = geom.find_span(n - 1, p, t, U)
        for j, v in enumerate(geom.basis_funs(s, t, p, U)):
            A[r, s - p + j] = v
    return A


def insert_knot(G, U, p, t):
    """Boehm: insert knot t along axis 0 of the CV grid G (n, m, 3)."""
    n = G.shape[0] - 1
    k = geom.find_span(n, p, t, U)
    Q = np.zeros((n + 2,) + G.shape[1:])
    Q[:k - p + 1] = G[:k - p + 1]
    Q[k + 1:] = G[k:]
    for i in range(k - p + 1, k + 1):
        a = (t - U[i]) / (U[i + p] - U[i])
        Q[i] = a * G[i] + (1 - a) * G[i - 1]
    return Q, list(U[:k + 1]) + [t] + list(U[k + 1:])


def ensure_rows(G, U, p, need):
    """Insert knots (middle of the longest span) until G has `need` rows."""
    while G.shape[0] < need:
        spans = [(U[i + 1] - U[i], i) for i in range(p, len(U) - p - 1) if U[i + 1] > U[i]]
        w, i = max(spans)
        G, U = insert_knot(G, U, p, 0.5 * (U[i] + U[i + 1]))
    return G, U


def _arclen(P):
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return d / (d[-1] or 1.0)


def _unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=1), 1e-30)[:, None]


# --------------------------------------------------------------------------
# Edge orientation: every edge is solved as "a = 0 of axis 0"
# --------------------------------------------------------------------------

def _to_std(G, Uu, Uv, du, dv, edge):
    if edge in ("V0", "V1"):
        G, Uu, Uv, du, dv = G.transpose(1, 0, 2), Uv, Uu, dv, du
    if edge in ("U1", "V1"):
        a, b = Uu[0], Uu[-1]
        G, Uu = G[::-1], [a + b - x for x in reversed(Uu)]
    return np.ascontiguousarray(G), list(Uu), list(Uv), du, dv


def _from_std(G, Ua, Ue, p, q, edge):
    """Inverse of _to_std: (G, Uu, Uv, du, dv)."""
    if edge in ("U1", "V1"):
        a, b = Ua[0], Ua[-1]
        G, Ua = G[::-1], [a + b - x for x in reversed(Ua)]
    if edge in ("V0", "V1"):
        return np.ascontiguousarray(G.transpose(1, 0, 2)), list(Ue), list(Ua), q, p
    return np.ascontiguousarray(G), list(Ua), list(Ue), p, q


# --------------------------------------------------------------------------
# The neighbour, resampled at the edge's own points
# --------------------------------------------------------------------------

def target_side(obj, ref, m=160):
    from .square import sample_side
    return sample_side(obj, ref, m)


def _matched(side, E, mode="arc"):
    """Target data at the edge points E (m, 3, uniform in the edge's parameter),
    oriented like E. mode: "arc" equal arc length, "project" closest points,
    "param" equal (normalised) parameter of edge and neighbour."""
    P = side.P
    if (np.linalg.norm(P[0] - E[0]) + np.linalg.norm(P[-1] - E[-1])
            > np.linalg.norm(P[-1] - E[0]) + np.linalg.norm(P[0] - E[-1])):
        side = side.reversed()
    s_t, s_e = _arclen(side.P), _arclen(E)
    if mode == "param":         # the neighbour's own parameter ~ the edge's
        s_t, s_e = np.asarray(side.t, float), np.linspace(0.0, 1.0, len(E))
    elif mode == "project":     # closest points
        a, b = side.P[:-1], side.P[1:]
        ab = b - a
        L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-30)
        t = np.clip(np.einsum("ksd,sd->ks", E[:, None, :] - a[None], ab) / L2, 0, 1)
        foot = a[None] + t[..., None] * ab[None]
        k = np.argmin(np.linalg.norm(foot - E[:, None, :], axis=2), axis=1)
        idx = np.arange(len(E))
        s_e = s_t[k] + t[idx, k] * (s_t[k + 1] - s_t[k])
    lerp = lambda A: np.array([np.interp(s_e, s_t, A[:, k]) for k in range(A.shape[1])]).T
    out = {"P": lerp(side.P)}
    if side.has_surface:
        out["B"] = _unit(lerp(np.asarray(side.B, float)))
        out["N"] = _unit(lerp(np.asarray(side.N, float)))
        Q = np.asarray(side.Q, float).reshape(len(side.P), 9) if side.Q is not None else None
        out["Q"] = None if Q is None else lerp(Q).reshape(-1, 3, 3)
    return out


def edge_conditions(G, Ua, p, Ue, q, side, cont, tangent=1.0, mode="project"):
    """Linear conditions of one edge on the std grid G (n_a, n_e, 3), world
    space, per level (G0, G1, G2): [(C, rhs, normal)] with C = {row a: (m, n_e)}.
    normal None: sum_a C_a @ G[a] = rhs (m, 3); else the scalar condition
    normal . (sum_a C_a @ G[a]) = rhs (m,). All in length units."""
    n_e = G.shape[1]
    m = max(80, 8 * n_e)
    ts = np.linspace(Ue[q], Ue[-q - 1], m)
    Ne = basis_matrix(Ue, q, n_e, ts)
    tg = _matched(side, Ne @ G[0], mode)
    out = [[({0: Ne}, tg["P"], None)]]
    if cont >= 1:
        d1 = Ua[p + 1] - Ua[1]
        c1 = p / d1
        Sa = c1 * (Ne @ (G[1] - G[0]))
        T = _unit(np.gradient(tg["P"], axis=0))
        Nn = tg["N"]
        b = _unit(np.cross(Nn, T))
        b *= np.sign(np.sum(b * tg["B"], axis=1))[:, None]
        along = np.sum(Sa * T, axis=1)
        h = np.sqrt(np.maximum(np.sum(Sa * Sa, axis=1) - along ** 2, 0.0))
        h = np.where(h > 1e-9 * (np.linalg.norm(Sa, axis=1) + 1e-30), h, np.linalg.norm(Sa, axis=1))
        Sa_d = tangent * (along[:, None] * T + h[:, None] * b)
        out.append([({1: Ne, 0: -Ne}, Sa_d / c1, None),                 # direction / length (soft)
                    ({1: Ne, 0: -Ne}, np.zeros(m), Nn)])                # in the tangent plane (strict)
        if cont >= 2:
            d2 = Ua[p + 2] - Ua[2]
            c2 = p * (p - 1) / (Ua[p + 1] - Ua[2])
            Saa = c2 * (Ne @ ((G[2] - G[1]) / d2 - (G[1] - G[0]) / d1))
            kn = np.einsum("ki,kij,kj->k", Sa_d, tg["Q"], Sa_d)
            Saa_d = Saa - np.sum(Saa * Nn, axis=1)[:, None] * Nn + kn[:, None] * Nn
            r = d2 / d1
            C = {2: Ne, 1: -(1 + r) * Ne, 0: r * Ne}
            out.append([(C, d2 * Saa_d / c2, None), (C, d2 * kn / c2, Nn)])
    return out


def _grid_index(edge, nu, nv):
    """(a, e) of the std grid -> flat index into the (nu, nv) grid."""
    if edge == "U0":
        return lambda a, e: a * nv + e
    if edge == "U1":
        return lambda a, e: (nu - 1 - a) * nv + e
    if edge == "V0":
        return lambda a, e: e * nv + a
    return lambda a, e: e * nv + (nv - 1 - a)


STRICT = 1000.0      # weight of the exact G1 / G2 conditions against the soft shape terms


def solve_joint(G, Uu, Uv, du, dv, work, sides, modes, tangent=1.0, iters=4, lam=1e-4):
    """Level by level (G0 rows, then G1 rows, then G2 rows), each a least-squares
    solve over that row of every edge at once (edges meeting at a corner share
    its CVs; a CV already set by a lower level stays). Repeated with the
    targets re-matched to the new edges. modes: {edge: "param" | "project"}."""
    nu, nv = G.shape[:2]
    ncv = nu * nv
    top = max(c for _, c in work)
    # an edge far from its neighbour is first matched by arc length
    near = {}
    for edge, c in work:
        Gs, Ua, Ue, p, q = _to_std(G, Uu, Uv, du, dv, edge)
        E = basis_matrix(Ue, q, Gs.shape[1], np.linspace(Ue[q], Ue[-q - 1], 40)) @ Gs[0]
        length = float(np.sum(np.linalg.norm(np.diff(E, axis=0), axis=1))) or 1.0
        near[edge] = _poly_dist(E, sides[edge].P) < 0.02 * length
    for it in range(iters):
        # the edge ends sit on the neighbour edge's ends (shared corners: the mean)
        ends = {}
        for edge, c in work:
            Gs, Ua, Ue, p, q = _to_std(G, Uu, Uv, du, dv, edge)
            ix = _grid_index(edge, nu, nv)
            n_e = Gs.shape[1]
            Ne = basis_matrix(Ue, q, n_e, np.array([Ue[q], Ue[-q - 1]]))
            P = _matched(sides[edge], Ne @ Gs[0])["P"]
            ends.setdefault(ix(0, 0), []).append(P[0])
            ends.setdefault(ix(0, n_e - 1), []).append(P[-1])
        X = G.reshape(-1, 3).copy()
        for k, pts in ends.items():
            X[k] = np.mean(pts, axis=0)
        G = X.reshape(nu, nv, 3)
        fixed = set(ends)
        for level in range(top + 1):
            rows, rhs, free = [], [], set()
            for edge, c in work:
                if c < level:
                    continue
                Gs, Ua, Ue, p, q = _to_std(G, Uu, Uv, du, dv, edge)
                ix = _grid_index(edge, nu, nv)
                n_e = Gs.shape[1]
                mode = modes[edge] if modes[edge] == "param" else ("project" if it > 0 or near[edge] else "arc")
                conds = edge_conditions(Gs, Ua, p, Ue, q, sides[edge], level, tangent, mode)[level]
                free.update(ix(level, e) for e in range(n_e))
                for C, y, nrm in conds:
                    m = len(y)
                    A = np.zeros((m, ncv))
                    for a, Ca in C.items():
                        A[:, [ix(a, e) for e in range(n_e)]] += Ca
                    if nrm is None:
                        w = 1.0 / np.sqrt(m)
                        for d in range(3):
                            R = np.zeros((m, 3 * ncv))
                            R[:, d::3] = A
                            rows.append(w * R)
                            rhs.append(w * y[:, d])
                    else:
                        w = STRICT / np.sqrt(m)
                        R = np.zeros((m, 3 * ncv))
                        for d in range(3):
                            R[:, d::3] = A * nrm[:, d:d + 1]
                        rows.append(w * R)
                        rhs.append(w * y)
            F = np.array(sorted(free - fixed), int)
            if not len(F):
                continue
            A, Y = np.vstack(rows), np.concatenate(rhs)
            x = G.reshape(-1).copy()
            cols = (3 * F[:, None] + np.arange(3)).reshape(-1)
            mask = np.zeros(3 * ncv, bool)
            mask[cols] = True
            Yr = Y - A[:, ~mask] @ x[~mask]
            AF = np.vstack([A[:, cols], lam * np.eye(len(cols))])
            YF = np.concatenate([Yr, lam * x[cols]])
            x[cols] = np.linalg.lstsq(AF, YF, rcond=None)[0]
            G = x.reshape(nu, nv, 3)
            fixed.update(F.tolist())
    return G


def check_edge(G, Ua, p, Ue, q, side, m=60):
    """(max gap, max normal angle in degrees, max relative curvature error) of
    the std grid's a = 0 edge against the neighbour."""
    n_e = G.shape[1]
    ts = np.linspace(Ue[q], Ue[-q - 1], m)
    Ne = basis_matrix(Ue, q, n_e, ts)
    h = 1e-6 * (ts[-1] - ts[0])
    lo, hi = np.maximum(ts - h, ts[0]), np.minimum(ts + h, ts[-1])
    dNe = (basis_matrix(Ue, q, n_e, hi) - basis_matrix(Ue, q, n_e, lo)) / (hi - lo)[:, None]
    E = Ne @ G[0]
    tg = _matched(side, E, "project")
    gap = float(np.max(np.linalg.norm(E - tg["P"], axis=1)))
    if "N" not in tg:
        return gap, None, None
    d1 = Ua[p + 1] - Ua[1]
    Sa = p / d1 * (Ne @ (G[1] - G[0]))
    Se = dNe @ G[0]
    n = _unit(np.cross(Sa, Se))
    cosang = np.abs(np.sum(n * tg["N"], axis=1))
    ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).max())
    kerr = None
    if p >= 2 and G.shape[0] > 2 and tg.get("Q") is not None:
        d2 = Ua[p + 2] - Ua[2]
        c2 = p * (p - 1) / (Ua[p + 1] - Ua[2])
        Saa = c2 * (Ne @ ((G[2] - G[1]) / d2 - (G[1] - G[0]) / d1))
        ss = np.sum(Sa * Sa, axis=1)
        k_own = np.sum(Saa * tg["N"], axis=1) / ss
        k_t = np.einsum("ki,kij,kj->k", Sa, tg["Q"], Sa) / ss
        ref = float(np.abs(k_t).max())
        kerr = float(np.abs(k_own - k_t).max() / ref) if ref > 1e-12 else float(np.abs(k_own).max())
    return gap, ang, kerr


# --------------------------------------------------------------------------
# Whole surface
# --------------------------------------------------------------------------

def align_text(obj, text, jobs_, tangent=1.0, report=None, measure_only=False):
    """Brep text of the (untrimmed, single) surface in `text` with the edges in
    jobs_ = [(edge, (target_obj, ref), cont)] aligned. Coordinates: obj local."""
    sm, ext = _k()
    brep = store.brep_from_text(text)
    if brep.face_count() != 1:
        raise RuntimeError("Surface Align works on a single untrimmed surface")
    d = ext.surface_cvs(brep.faces()[0].surface())
    if d.get("weights") and max(abs(w - 1.0) for w in d["weights"]) > 1e-12:
        raise RuntimeError("rational surfaces can't be aligned (Rebuild Surface first)")
    nu, nv = d["count_u"], d["count_v"]
    M = np.array(geom.world_matrix(obj), float)
    R, T = M[:3, :3], M[:3, 3]
    G = np.array(d["points"], float).reshape(nu, nv, 3) @ R.T + T
    Uu, Uv = list(d["knots_u"]), list(d["knots_v"])
    du, dv = int(d["degree_u"]), int(d["degree_v"])

    sides = {}
    work = []
    for edge, (tobj, tref), cont in jobs_:
        if edge not in EDGES:
            raise RuntimeError(f"{edge} is not a boundary edge")
        sd = target_side(tobj, tref)
        c = CONT[cont] if isinstance(cont, str) else int(cont)
        if not sd.has_surface and c > 0:
            c = 0                        # a curve only gives position
            if report is not None:
                report.append(f"{edge}: {tobj.name} is a curve, aligned G0")
        sides[edge] = sd
        work.append((edge, c))
    cmap = dict(work)

    # enough rows / degree in each direction
    for pair, axis in ((("U0", "U1"), "u"), (("V0", "V1"), "v")):
        cs = [cmap[e] for e in pair if e in cmap]
        if not cs or measure_only:
            continue
        deg = du if axis == "u" else dv
        if max(cs) > deg:
            raise RuntimeError(f"degree {deg} in {axis.upper()} is too low for G{max(cs)} (Rebuild Surface first)")
        need = sum(c + 1 for c in cs) + (1 if len(cs) == 1 else 0)
        if axis == "u":
            G, Uu = ensure_rows(G, Uu, du, need)
        else:
            Gt, Uv = ensure_rows(G.transpose(1, 0, 2), Uv, dv, need)
            G = np.ascontiguousarray(Gt.transpose(1, 0, 2))

    if not measure_only:
        # two ways to pair each edge with its neighbour: by parameter (exact when
        # they share a layout) and by closest points; each edge keeps the closer
        modes = {}
        for edge, _ in work:
            gaps = {}
            for mode in ("param", "project"):
                Gm = solve_joint(G, Uu, Uv, du, dv, [(edge, 0)], sides, {edge: mode}, iters=2)
                Gs, Ua, Ue, p, q = _to_std(Gm, Uu, Uv, du, dv, edge)
                gaps[mode] = check_edge(Gs, Ua, p, Ue, q, sides[edge])[0]
            modes[edge] = min(gaps, key=gaps.get)
        G = solve_joint(G, Uu, Uv, du, dv, work, sides, modes, tangent)
    if report is not None:
        for edge, c in work:
            Gs, Ua, Ue, p, q = _to_std(G, Uu, Uv, du, dv, edge)
            gap, ang, kerr = check_edge(Gs, Ua, p, Ue, q, sides[edge])
            msg = f"{edge} G{c}: gap {gap:.2g}"
            if ang is not None and c >= 1:
                msg += f", angle {ang:.2g}°"
            if kerr is not None and c >= 2:
                msg += f", curvature {100 * kerr:.2g}%"
            report.append(msg)
    if measure_only:
        return text

    Ri = np.linalg.inv(R)
    L = (G - T) @ Ri.T
    nu, nv = L.shape[:2]
    pts = [tuple(map(float, L[i, j])) for i in range(nu) for j in range(nv)]
    norm = lambda K: [(x - K[0]) / (K[-1] - K[0]) for x in K]
    s = ext.surface_from_cvs(du, dv, pts, norm(Uu), norm(Uv))
    return store.brep_to_text(ext.brep_from_surface(s))


# --------------------------------------------------------------------------
# Finding the neighbour of an edge
# --------------------------------------------------------------------------

def _poly_dist(A, B):
    """Mean distance from the points A to the polyline B."""
    a, b = B[:-1], B[1:]
    ab = b - a
    L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-30)
    out = []
    for x in A:
        t = np.clip(np.sum((x - a) * ab, axis=1) / L2, 0, 1)
        out.append(np.min(np.linalg.norm(a + t[:, None] * ab - x, axis=1)))
    return float(np.mean(out))


def _inward(obj, edge):
    """World point at the middle of `edge` and the direction into the surface there."""
    surf = store.get_surfaces(obj)[0]
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    um, vm, eu, ev = 0.5 * (u0 + u1), 0.5 * (v0 + v1), 1e-3 * (u1 - u0), 1e-3 * (v1 - v0)
    a, b = {"U0": ((u0, vm), (u0 + eu, vm)), "U1": ((u1, vm), (u1 - eu, vm)),
            "V0": ((um, v0), (um, v0 + ev)), "V1": ((um, v1), (um, v1 - ev))}[edge]
    M = geom.world_matrix(obj)
    p, q = np.array(M @ Vector(surf.evaluate(*a))), np.array(M @ Vector(surf.evaluate(*b)))
    return p, q - p


def overlaps(obj, edge, tobj, tref):
    """True when the target surface lies on the SAME side of the edge as obj
    (e.g. another fill of the same opening): aligning to it would fold obj back."""
    if tobj.type == 'CURVE' or not features.is_surface_edge(tref):
        return False
    p, inward = _inward(obj, edge)
    sd = target_side(tobj, tref, 21)
    k = int(np.argmin(np.linalg.norm(np.asarray(sd.P) - p, axis=1)))
    return float(np.asarray(sd.B[k]) @ inward) < 0.0     # B points away from the target face


def find_neighbour(obj, edge, cands, rel=0.05):
    """(obj, ref) of the candidate edge / curve lying along `edge`, or None.
    cands: [(obj, ref, world points)] as the pick candidates. A surface lying on
    the same side as obj (overlapping it) is only taken when nothing else fits."""
    surf = store.get_surfaces(obj)[0]
    M = geom.world_matrix(obj)
    E = np.array([tuple(M @ Vector(p)) for p in geom.boundary_points(surf, edge, 32)])
    length = float(np.sum(np.linalg.norm(np.diff(E, axis=0), axis=1))) or 1.0
    scored = []
    for o, r, pts in cands:
        if o == obj:
            continue
        P = np.array([tuple(p) for p in pts], float)
        if len(P) < 2:
            continue
        d = max(_poly_dist(E, P), _poly_dist(P[:: max(1, len(P) // 32)], E) if o.type == 'CURVE' else 0.0)
        if d <= rel * length:
            scored.append((d + (1e-9 if o.type == 'CURVE' else 0.0), o, r))   # prefer a surface edge on a tie
    scored.sort(key=lambda x: x[0])
    for _, o, r in scored:
        try:
            if not overlaps(obj, edge, o, r):
                return o, r
        except Exception:
            return o, r
    return (scored[0][1], scored[0][2]) if scored else None


def apply_surface_align(obj, text, inputs, p):
    jobs_ = [(e, inp, c) for e, inp, c in zip(p["edges"], inputs, p["cont"])]
    return align_text(obj, text, jobs_, float(p.get("tangent", 1.0)))


def _register_step():
    from . import steps
    steps.APPLY["surface_align"] = apply_surface_align
    steps.LABEL["surface_align"] = lambda p: "Align · " + ", ".join(
        f"{e} {c}" for e, c in zip(p.get("edges", []), p.get("cont", [])))
    features.NATURAL_STEPS.add("surface_align")


_register_step()


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import EnumProperty, FloatProperty, StringProperty  # noqa: E402

from .ops import (PickMixin, _decode_inputs, _encode_inputs, _kernel_ok, _near_candidates,  # noqa: E402
                  _object_under_mouse, _pick_candidates, _select_only)
from . import draw as _draw  # noqa: E402

_ITEMS = [("G0", "G0", "Position: the edge lies on the neighbour edge"),
          ("G1", "G1", "Tangent to the neighbouring surface"),
          ("G2", "G2", "Curvature continuous with the neighbouring surface")]


def _cont_prop(edge):
    return EnumProperty(name=edge, items=_ITEMS, default="G1",
                        description=f"Continuity of edge {edge} to its neighbour")


def parse_picks(picks):
    """picks [(obj, ref)] -> (surface, [(edge, (target obj, ref) | None)]).
    The first pick names the surface; another of its edges adds an edge; a
    pick on any other object is the target of the edge before it."""
    if not picks:
        raise ValueError("pick an edge of the surface to align")
    obj = picks[0][0]
    out = []
    for o, r in picks:
        if o == obj:
            if r not in EDGES:
                raise ValueError(f"{r} of {o.name} is not a boundary edge of an untrimmed surface")
            if r not in [e for e, _ in out]:
                out.append([r, None])
        elif out:
            out[-1][1] = (o, r)
    return obj, [tuple(x) for x in out]


class NPROTO_OT_surface_align(PickMixin, bpy.types.Operator):
    """Align boundary edges of an untrimmed surface G0 / G1 / G2 to the surfaces they
    touch (moves only the CV rows at those edges). Pick one or more edges of the
    surface (Enter to finish); picking another surface's edge right after one sets
    its target, otherwise the touching edge is found"""
    bl_idname = "nproto.surface_align"
    bl_label = "Surface Align"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 1
    pick_max = 32
    finish_on_min = False
    extra_help = ("   Click ON a surface near its edge = that edge; then click the neighbour across it "
                  "= its target (optional)")
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    cont_U0: _cont_prop("U0")
    cont_U1: _cont_prop("U1")
    cont_V0: _cont_prop("V0")
    cont_V1: _cont_prop("V1")
    tangent: FloatProperty(name="Tangent", default=1.0, min=0.05, soft_max=3.0,
                           description="Scale of the cross tangent at the aligned edges (G1 / G2)")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        return self.start_picking(context, event)

    def _update_hover(self, context, event):
        """Hovering a surface picks its nearest edge, however far the edge is on
        screen (click inside the face, near the edge); off the surfaces, edges
        and curves within reach as usual. The first pick must be an edge of an
        untrimmed surface."""
        from mathutils import Vector as _V
        cands = self._cands
        if not self._picks:
            cands = [c for c in cands if c[0].type != 'CURVE' and c[1] in EDGES and store.is_natural(c[0])]
        under = _object_under_mouse(context, _V((event.mouse_region_x, event.mouse_region_y)))
        ties = []
        if under is not None:
            own = [c for c in cands if c[0] == under]
            if own:
                ties = _near_candidates(context, event, own, max_px=1e9, tie_px=0.0)[:1]
        if not ties:
            ties = _near_candidates(context, event, cands)
        key = [(c[0].name, c[1]) for c in ties]
        if key != getattr(self, "_tie_key", None):
            self._tie_key, self._ties, self._cycle = key, ties, 0
        cur = self._ties[self._cycle % len(self._ties)] if self._ties else None
        _draw.pick_overlay["hover"] = cur[2] if cur else None
        self._status(context)
        context.area.tag_redraw()
        return cur

    def _jobs(self, context):
        obj, edges = parse_picks(_decode_inputs(self.inputs))
        if not store.is_natural(obj) or store.get_brep(obj).face_count() != 1:
            raise ValueError(f"{obj.name} is trimmed: only untrimmed single surfaces can be aligned "
                             "(Refit to Trim first)")
        cands = None
        jobs_ = []
        for e, tgt in edges:
            if tgt is None:
                if cands is None:
                    cands = [c for c in _pick_candidates(context) if c[0] != obj]
                tgt = find_neighbour(obj, e, cands)
                if tgt is None:
                    raise ValueError(f"no surface edge or curve touches {e}: pick its target right after the edge")
            jobs_.append((e, tgt, getattr(self, "cont_" + e)))
        return obj, jobs_

    def draw(self, context):
        lay = self.layout
        try:
            _, edges = parse_picks(_decode_inputs(self.inputs))
        except Exception:
            edges = []
        col = lay.column(align=True)
        for e, _ in edges:
            row = col.row(align=True)
            row.label(text=e)
            row.prop(self, "cont_" + e, expand=True)
        lay.prop(self, "tangent")

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            obj, jobs_ = self._jobs(context)
        except Exception as exc:
            self.report({'WARNING'}, str(exc))
            return {'CANCELLED'}
        msgs = []
        hist = features.history_enabled(context)
        try:
            if hist:
                p = {"edges": [e for e, _, _ in jobs_], "cont": [c for _, _, c in jobs_],
                     "tangent": float(self.tangent)}
                features.push_step(obj, "surface_align", [t for _, t, _ in jobs_], p)
                features.rebuild(obj, context, force=True)
                align_text(obj, obj[store.PROP_BREP], jobs_, report=msgs, measure_only=True)
            else:
                text = align_text(obj, obj[store.PROP_BREP], jobs_, float(self.tangent), report=msgs)
                store.set_geometry_text(obj, text, context, keep_untrimmed=False)
            obj[store.PROP_NATURAL] = 1
        except Exception as exc:
            if hist:
                try:
                    features.remove_step(obj, len(features.stack(obj)) - 1)
                    features.rebuild(obj, context, force=True)
                except Exception:
                    pass
            self.report({'ERROR'}, f"Surface Align failed: {exc}")
            return {'CANCELLED'}
        bad = []
        try:
            from .contlabels import measure
            for e, t, c in jobs_:
                if overlaps(obj, e, *t):
                    bad.append(f"{e}: {t[0].name} {t[1]} lies on the same side (overlapping surface)")
                chk = measure(obj, t[0], t[1], CONT[c])
                if not chk["ok"]:
                    bad.append(f"{e} {c} {chk['text']}")
        except Exception as exc:
            print("NURBS surface align check:", exc)
        if bad:
            self.report({'WARNING'}, "Surface Align: " + "; ".join(bad))
        names = ", ".join(f"{e}→{t[0].name} {t[1]} {c}" for e, t, c in jobs_)
        self.report({'INFO'}, f"Aligned {names}" + (f"  ({'; '.join(msgs)})" if msgs else ""))
        _select_only(context, [obj])
        return {'FINISHED'}


classes = (NPROTO_OT_surface_align,)
