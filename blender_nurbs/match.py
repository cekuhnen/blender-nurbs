"""Match Surface (Rhino MatchSrf; Alias Align onto the Master's layout): make a
boundary edge of an untrimmed surface exactly G0 / G1 / G2 / G3 to a target
edge by setting its first 1 / 2 / 3 / 4 CV rows from the target's own CV rows.

The target's rows (features.exact_side; world space, edge parameter t in 0..1):
    P(t)  = sum N_j(t) P_j      the target edge
    D(t)  = sum N_j(t) D_j      its cross derivative, pointing out of the target
    DD(t) = sum N_j(t) DD_j     its cross second derivative
    DDD(t) = sum N_j(t) DDD_j   its cross third derivative
The surface's edge at a = 0 of its cross direction (salign._to_std: degree p,
knots Ua), rows G0, G1, G2 on the same basis N_j along the edge:
    S(0, t)   = sum N_j G0_j
    Sa(0, t)  = c1 sum N_j (G1_j - G0_j)                          c1 = p / d1
    Saa(0, t) = c2 sum N_j ((G2_j - G1_j) / d2 - (G1_j - G0_j) / d1)
Match sets them CV by CV:
    G0  G0_j = P_j
    G1  c1 (G1_j - G0_j) = lam D_j                       (lam > 0, one per edge)
    G2  c2 ((G2_j - G1_j) / d2 - (G1_j - G0_j) / d1) = lam^2 DD_j + rho c1 (G1_j - G0_j)
    G3  sum_i w_i G_i,j (i = 0..3) = lam^3 DDD_j + 3 rho lam^2 DD_j + sigma c1 (G1_j - G0_j)
        (w: geom.start_coeffs, the third derivative's weights)
so Sa = lam D, Saa = lam^2 DD + rho Sa and Saaa = lam^3 DDD + 3 rho lam^2 DD
+ sigma Sa all along the edge: S(a, t) = target(f(a), t) to third order for
the cross reparameterisation f' = lam, f'' = rho lam, f^(3) = sigma lam -- one
tangent plane, the same normal curvature across the edge (II(w, w) / |w|^2 is
the same for w and lam w; rho Sa is tangential) and the same curvature rate.
The surface continues the target's isocurves: the CV hull lines run on
straight across the edge, as if the two were one surface split there.
lam keeps the surface's own cross tangent length (times Tangent); rho slides
row 2 along the hull lines to where it sat relative to row 1, sigma row 3
likewise. Where another matched edge meets the edge at a corner, that edge's
row already holds this one's corner CVs: lam, rho and sigma are read from
them instead, so edges cut from one surface come back exactly.

Layouts that differ along the edge are made one first, exactly: the lower
degree raised (each knot's multiplicity up by the same amount: the fewest CVs)
and the knots merged, so the surface takes the target's knots there (its shape
does not change). Across the edge, too low a degree is raised and missing rows
are inserted; Keep Other End decides how many rows at the far edge stay.

Average (Rhino's Average Surfaces): both surfaces move to meet halfway -- one
edge row (the mean), one cross direction (the mean of the two scaled cross
tangents) and one curvature row; each surface keeps its own tangent length.

Recorded as a history step ("match_surface") unless averaging.
"""

import numpy as np

from . import features, geom, salign, store

EDGES = salign.EDGES
CONT = salign.CONT
KEEP = {"NONE": 0, "G0": 1, "G1": 2, "G2": 3, "G3": 4}    # rows at the far edge that must not move
CROSS = {"U0": 0, "U1": 0, "V0": 1, "V1": 1}     # the axis the edge's rows step along
SNAP = 1e-6       # an edge end this close to the target's end (relative to its length) sits on it


class MatchError(RuntimeError):
    """What Match can't do, and what to do instead."""


def _k():
    return salign._k()


# --------------------------------------------------------------------------
# B-spline layouts
# --------------------------------------------------------------------------

def _norm(U):
    a, b = float(U[0]), float(U[-1])
    return [(float(x) - a) / (b - a) for x in U]


def _span_samples(U, p):
    """p + 2 parameters in every span: a fit on them is fully determined."""
    ts = []
    for a, b in zip(U[:-1], U[1:]):
        if b > a:
            ts.extend(np.linspace(a, b, p + 3)[:-1])
    ts.append(U[-1])
    return np.array(ts)


def raise_degree(A, U, p, q):
    """CV rows A (n, k) of a clamped B-spline (knots U, degree p) as degree q:
    the same curve with each interior knot's multiplicity up by q - p (the
    fewest CVs, no Bezier split). The fit in the bigger space reproduces the
    curve exactly. -> (rows, knots, q)."""
    A = np.asarray(A, float)
    if q <= p:
        return A, list(U), p
    U = [float(x) for x in U]
    inner = U[p + 1:len(U) - p - 1]
    U2 = [U[0]] * (q + 1) + [v for v in sorted(set(inner)) for _ in range(inner.count(v) + q - p)] + [U[-1]] * (q + 1)
    n2 = len(U2) - q - 1
    ts = _span_samples(U2, q)
    F = salign.basis_matrix(U, p, len(A), ts) @ A
    X = np.linalg.lstsq(salign.basis_matrix(U2, q, n2, ts), F, rcond=None)[0]
    X[0], X[-1] = A[0], A[-1]
    return X, U2, q


def compatible(items):
    """[(degree, knots 0..1, rows (n, k))] on one degree and knot vector,
    exactly: degrees raised to the highest, then every knot inserted into
    every item to its highest multiplicity. -> (degree, knots, [rows])."""
    q = max(int(p) for p, _, _ in items)
    up = [raise_degree(A, U, int(p), q) for p, U, A in items]
    return geom.compatible_arrays([(q, U_, A_) for A_, U_, _ in up], tol=1e-7)


class Net:
    """A surface's CV grid in world space: G (nu, nv, 3), knots (0..1) and
    degree per axis (0 = u, 1 = v)."""

    def __init__(self, G, knots, deg):
        self.G = np.ascontiguousarray(G, float)
        self.knots = [[float(k) for k in knots[0]], [float(k) for k in knots[1]]]
        self.deg = [int(deg[0]), int(deg[1])]

    @classmethod
    def of(cls, obj, text=None):
        _, ext = _k()
        brep = store.brep_from_text(text) if text is not None else store.get_brep(obj)
        if brep.face_count() != 1 or (text is None and not store.is_natural(obj)):
            raise MatchError(f"{obj.name} is trimmed or joined: Match works on one untrimmed surface "
                             "(Refit to Trim or Unjoin first)")
        try:
            d = ext.surface_cvs(brep.faces()[0].surface())
        except Exception:
            raise MatchError(f"{obj.name} has no CV net (e.g. an exact offset): Rebuild Surface first")
        if d.get("weights") and max(abs(w - 1.0) for w in d["weights"]) > 1e-12:
            raise MatchError(f"{obj.name} is rational (arcs in its CV net): Rebuild Surface first")
        nu, nv = d["count_u"], d["count_v"]
        M = np.array(geom.world_matrix(obj), float)
        G = np.array(d["points"], float).reshape(nu, nv, 3) @ M[:3, :3].T + M[:3, 3]
        return cls(G, (_norm(d["knots_u"]), _norm(d["knots_v"])), (d["degree_u"], d["degree_v"]))

    def copy(self):
        return Net(self.G.copy(), self.knots, self.deg)

    def count(self, axis):
        return self.G.shape[axis]

    def rows(self, axis):
        """The CV rows that step along `axis`, side by side: (count(axis), other * 3)."""
        Gt = self.G if axis == 0 else self.G.transpose(1, 0, 2)
        return np.ascontiguousarray(Gt).reshape(Gt.shape[0], -1)

    def set_rows(self, axis, deg, knots, A):
        A = np.asarray(A, float)
        Gt = A.reshape(len(A), self.G.shape[1 - axis], 3)
        self.G = np.ascontiguousarray(Gt if axis == 0 else Gt.transpose(1, 0, 2))
        self.deg[axis], self.knots[axis] = int(deg), [float(k) for k in knots]

    def raise_degree(self, axis, q):
        A, U, q = raise_degree(self.rows(axis), self.knots[axis], self.deg[axis], q)
        self.set_rows(axis, q, U, A)

    def ensure_rows(self, axis, need):
        A, U = salign.ensure_rows(self.rows(axis), self.knots[axis], self.deg[axis], need)
        self.set_rows(axis, self.deg[axis], U, A)

    def std(self, edge):
        """(grid with the edge at row 0, knots across, knots along, degree across, along)."""
        return salign._to_std(self.G, self.knots[0], self.knots[1], self.deg[0], self.deg[1], edge)

    def edge_row(self, edge):
        return self.std(edge)[0][0]

    def text(self, obj):
        """Brep text of the surface in obj's local space."""
        _, ext = _k()
        M = np.array(geom.world_matrix(obj), float)
        L = (self.G - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T
        nu, nv = L.shape[:2]
        pts = [tuple(map(float, L[i, j])) for i in range(nu) for j in range(nv)]
        s = ext.surface_from_cvs(self.deg[0], self.deg[1], pts, self.knots[0], self.knots[1])
        return store.brep_to_text(ext.brep_from_surface(s))


def _eval(deg, knots, A, ts):
    return salign.basis_matrix(knots, int(deg), len(A), np.asarray(ts, float)) @ np.asarray(A, float)


def _length(deg, knots, A):
    P = _eval(deg, knots, A, np.linspace(0.0, 1.0, 65))
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


# --------------------------------------------------------------------------
# The target edge
# --------------------------------------------------------------------------

def target_side(tobj, tref):
    """The target edge's exact rows ({"deg", "knots", "P", "D", "DD"}, world
    space, D / DD None for a curve), or a MatchError saying why it has none."""
    try:
        s = features.exact_side(tobj, tref)
    except Exception:
        s = None
    if s is not None:
        return s
    name = f"{tobj.name} {tref}"
    if tobj.type == 'CURVE':
        raise MatchError(f"{name} is a rational curve (an arc or circle): Match copies CVs, so it needs a "
                         "plain B-spline (Rebuild Curve it, or use Surface Align)")
    if features.is_brep_edge_ref(tref):
        raise MatchError(f"{name} is a trim curve, not a side of its surface's CV net: Match can only copy "
                         "the rows of an untrimmed side (use Surface Align, which fits any edge)")
    raise MatchError(f"{name} is rational (arcs in its CV net): Rebuild Surface it first, or use Surface Align")


def _closest_t(s, x):
    """Parameter (0..1) of the target edge's point nearest to x."""
    ts = np.linspace(0.0, 1.0, 801)
    P = _eval(s["deg"], s["knots"], s["P"], ts)
    i = int(np.argmin(np.linalg.norm(P - x, axis=1)))
    a, b = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
    f = lambda t: float(np.linalg.norm(_eval(s["deg"], s["knots"], s["P"], [t])[0] - x))
    g = 0.5 * (5 ** 0.5 - 1)
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def fit_side(s, E0, E1, ends="EDGE"):
    """The target side running the way the edge E0 -> E1 does. ends "EDGE":
    the whole target edge (the edge is stretched end to end onto it);
    "CLOSEST": the stretch between the target's points nearest E0 and E1,
    cut out exactly (part of a longer edge)."""
    if ends != "CLOSEST":
        if geom.should_flip(np.array([E0, E1]), np.array([s["P"][0], s["P"][-1]])):
            s = geom.reverse_side(s)
        return s
    t0, t1 = _closest_t(s, E0), _closest_t(s, E1)
    if t0 > t1:
        s, t0, t1 = geom.reverse_side(s), 1.0 - t0, 1.0 - t1
    L = _length(s["deg"], s["knots"], s["P"]) or 1.0
    P0, P1 = s["P"][0], s["P"][-1]
    at = lambda t: _eval(s["deg"], s["knots"], s["P"], [t])[0]
    if np.linalg.norm(at(t0) - P0) <= SNAP * L:
        t0 = 0.0
    if np.linalg.norm(at(t1) - P1) <= SNAP * L:
        t1 = 1.0
    if t1 - t0 < 1e-6:
        raise MatchError("both ends of the edge are nearest to one point of the target")
    return s if (t0, t1) == (0.0, 1.0) else geom.sub_side(s, t0, t1)


# --------------------------------------------------------------------------
# Making the layouts fit
# --------------------------------------------------------------------------

def prepare_across(net, conts, keep, notes=None, name=""):
    """Degree and rows across each matched edge: degree >= its level, rows for
    every matched edge of that direction (plus the far edge's kept rows when
    only one end is matched). conts: {edge: level}."""
    for axis, pair in ((0, ("U0", "U1")), (1, ("V0", "V1"))):
        cs = [conts[e] for e in pair if e in conts]
        if not cs:
            continue
        uv = "UV"[axis]
        top = max(cs)
        if top > net.deg[axis]:
            if notes is not None:
                notes.append(f"{name}degree {uv} {net.deg[axis]} → {top} for G{top}")
            net.raise_degree(axis, top)
        need = sum(c + 1 for c in cs) + (KEEP.get(keep, 1) if len(cs) == 1 else 0)
        if net.count(axis) < need:
            if notes is not None:
                notes.append(f"{name}{need - net.count(axis)} {uv} rows added")
            net.ensure_rows(axis, need)


def _side_item(s):
    keys, A = geom._side_array(s)
    return keys, (s["deg"], s["knots"], A)


# --------------------------------------------------------------------------
# Setting the rows
# --------------------------------------------------------------------------

def _consts(Ua, p):
    d1 = Ua[p + 1] - Ua[1]
    k = {"d1": d1, "c1": p / d1}
    if p >= 2:
        k["d2"] = Ua[p + 2] - Ua[2]
        k["c2"] = p * (p - 1) / (Ua[p + 1] - Ua[2])
    if p >= 3:
        k["w3"] = geom.start_coeffs(Ua, p, 3)          # Saaa(0) = sum_i w3_i row_i
    return k


def _speed(Ne, A):
    return float(np.mean(np.linalg.norm(Ne @ A, axis=1)))


def solve(net, work, tangent=1.0, notes=None):
    """Set the net's rows from the sides, level by level -- every edge's row 0,
    then row 1, then row 2 -- so a CV two edges want at one level (a corner)
    gets their mean, and one a lower level placed stays. work: [(edge, level,
    side)], each side on the net's layout along its edge, running its way.

    lam (the tangent length), rho (row 2's slide) and sigma (row 3's) are one
    number per edge. Where another matched edge meets this one, its edge row
    already holds this edge's row 1 / 2 / 3 CV at that corner: lam, rho and
    sigma are taken from those CVs, so both edges keep their rows (two edges
    cut from one surface are matched exactly). Elsewhere lam keeps the
    surface's own tangent length, and rho / sigma keep row 2 / 3 where it sat
    relative to the row before it."""
    nu, nv = net.G.shape[:2]
    X0 = net.G.reshape(-1, 3).copy()
    X = X0.copy()
    size = float(np.linalg.norm(np.ptp(X0, axis=0))) or 1.0
    ts = np.linspace(0.0, 1.0, 65)
    edges = []
    for edge, c, sd in work:
        Gs, Ua, Ue, p, q = net.std(edge)
        n_e = Gs.shape[1]
        if len(sd["P"]) != n_e:
            raise RuntimeError(f"{edge}: the layouts along the edge differ ({len(sd['P'])} / {n_e} CVs)")
        e = {"edge": edge, "c": c, "k": _consts(Ua, p), "ix": salign._grid_index(edge, nu, nv), "n": n_e,
             "P": np.asarray(sd["P"], float), "lam": 1.0, "rho": 0.0,
             "D": None if sd.get("D") is None else np.asarray(sd["D"], float),
             "DD": None if sd.get("DD") is None else np.asarray(sd["DD"], float),
             "DDD": None if sd.get("DDD") is None else np.asarray(sd["DDD"], float)}
        if c >= 1:
            Ne = salign.basis_matrix(Ue, q, n_e, ts)
            own, tgt = _speed(Ne, e["k"]["c1"] * (Gs[1] - Gs[0])), _speed(Ne, e["D"])
            if tgt <= 1e-300:
                raise MatchError(f"{edge}: the target has no cross tangent there")
            e["lam"] = own / tgt if own > 1e-12 * size else 1.0
        edges.append(e)
    fixed, spread = set(), 0.0
    for level in range(max((c for _, c, _ in work), default=-1) + 1):
        want = {}
        for e in edges:
            if e["c"] < level:
                continue
            ix, n_e, k = e["ix"], e["n"], e["k"]
            rows = [[ix(a, j) for j in range(n_e)] for a in range(level + 1)]
            corners = [j for j in (0, n_e - 1) if rows[level][j] in fixed]
            if level == 0:
                vals = e["P"]
            elif level == 1:
                x0, D = X[rows[0]], e["D"]
                est = [k["c1"] * float((X[rows[1][j]] - x0[j]) @ D[j]) / float(D[j] @ D[j])
                       for j in corners if D[j] @ D[j] > 1e-300]
                est = [x for x in est if x > 0.0]
                e["lam"] = tangent * (float(np.mean(est)) if est else e["lam"])
                vals = x0 + e["lam"] * D / k["c1"]
            else:
                x0, x1 = X[rows[0]], X[rows[1]]
                lam = e["lam"]
                if level == 2:
                    base = x1 + k["d2"] * ((x1 - x0) / k["d1"] + lam * lam * e["DD"] / k["c2"])
                    slide = (k["d2"] / k["c2"]) * k["c1"] * (x1 - x0)     # row 2 along the hull lines
                else:
                    w3, x2 = k["w3"], X[rows[2]]
                    rate = lam ** 3 * e["DDD"] + 3.0 * e["rho"] * lam * lam * e["DD"]
                    base = (rate - w3[0] * x0 - w3[1] * x1 - w3[2] * x2) / w3[3]
                    slide = k["c1"] * (x1 - x0) / w3[3]                  # row 3 along the hull lines
                use = np.zeros(n_e, bool)
                if corners:
                    ref, use[corners] = X[rows[level]], True
                else:
                    ref = X0[rows[level]] + (X[rows[level - 1]] - X0[rows[level - 1]])
                    use[:] = [r not in fixed for r in rows[level]]
                den = float(np.sum(slide[use] ** 2))
                rho = float(np.sum((ref - base)[use] * slide[use])) / den if den > 1e-300 else 0.0
                if level == 2:
                    e["rho"] = rho
                vals = base + rho * slide
            for j in range(n_e):
                kk = rows[level][j]
                if kk not in fixed:
                    want.setdefault(kk, []).append(vals[j])
        for kk, vs in want.items():
            X[kk] = np.mean(vs, axis=0)
            if len(vs) > 1:
                spread = max(spread, float(max(np.linalg.norm(v - X[kk]) for v in vs)))
        fixed.update(want)
    if notes is not None and spread > 1e-9 * size:
        notes.append(f"the targets disagree at a shared corner (by {spread:.2g}): met halfway there")
    net.G = X.reshape(nu, nv, 3)
    return net


# --------------------------------------------------------------------------
# The whole match
# --------------------------------------------------------------------------

def _level(cont):
    return CONT[cont] if isinstance(cont, str) else int(cont)


def match_nets(S, jobs_, tangent=1.0, keep="G0", ends="EDGE", average=False, notes=None, obj=None):
    """S (Net, world) with the edges in jobs_ = [(edge, (target obj, ref),
    level)] matched. average: the targets (untrimmed surface edges) move too,
    to meet halfway. -> (S, {target name: (target obj, its Net)}); the dict is
    empty unless averaging."""
    notes = notes if notes is not None else []
    work, seen = [], set()
    targets = {}
    for edge, (tobj, tref), cont in jobs_:
        if edge not in EDGES:
            raise MatchError(f"{edge} is not a boundary edge of an untrimmed surface")
        if edge in seen:
            raise MatchError(f"{edge} is picked twice")
        seen.add(edge)
        if obj is not None and tobj == obj:
            raise MatchError(f"{edge}: pick an edge of another surface as its target")
        c = _level(cont)
        Gs, Ua, Ue, p, q = S.std(edge)
        if _length(q, Ue, Gs[0]) < 1e-12:
            raise MatchError(f"{edge} is a point (a pole): there is no edge to match")
        E0, E1 = Gs[0, 0], Gs[0, -1]
        if average:
            if not features.is_edge_ref(tref) or tobj.type == 'CURVE' or not store.is_natural(tobj):
                raise MatchError(f"Average moves both surfaces: {tobj.name} {tref} must be an edge of an "
                                 "untrimmed surface")
            if tobj.name in targets:
                raise MatchError(f"Average: {tobj.name} is the target of two edges")
            T = Net.of(tobj)
            Hr = T.edge_row(tref)
            flip = geom.should_flip(np.array([E0, E1]), np.array([Hr[0], Hr[-1]]))
            targets[tobj.name] = (tobj, T)
            work.append({"edge": edge, "c": c, "T": T, "tref": tref, "flip": flip, "name": tobj.name})
        else:
            sd = target_side(tobj, tref)
            if sd["D"] is None and c > 0:
                notes.append(f"{edge}: {tobj.name} is a curve -- matched G0")
                c = 0
            if c >= 3 and sd.get("DDD") is None:
                notes.append(f"{edge}: {tobj.name} {tref} has no exact third derivative rows -- matched G2")
                c = 2
            work.append({"edge": edge, "c": c, "side": fit_side(sd, E0, E1, ends), "name": tobj.name})

    conts = {w["edge"]: w["c"] for w in work}
    prepare_across(S, conts, keep, notes)
    for w in work:
        if average:
            prepare_across(w["T"], {w["tref"]: w["c"]}, keep, notes, f"{w['name']}: ")

    # one layout along every edge: the surface's rows, the target rows (and
    # the target surfaces when averaging) merged per direction
    for along, pair in ((1, ("U0", "U1")), (0, ("V0", "V1"))):
        ws = [w for w in work if w["edge"] in pair]
        if not ws:
            continue
        n_before, d_before = S.count(along), S.deg[along]
        items, keys = [(S.deg[along], S.knots[along], S.rows(along))], []
        for w in ws:
            if average:
                T, ax = w["T"], 1 - CROSS[w["tref"]]
                A, U = T.rows(ax), T.knots[ax]
                if w["flip"]:
                    A, U = A[::-1], [1.0 - x for x in reversed(U)]
                items.append((T.deg[ax], U, A))
            else:
                ky, it = _side_item(w["side"])
                keys.append(ky)
                items.append(it)
        try:
            q, U, arrs = compatible(items)
        except ValueError as exc:
            raise MatchError(f"the layouts along {', '.join(w['edge'] for w in ws)} could not be made one: {exc}")
        S.set_rows(along, q, U, arrs[0])
        for i, w in enumerate(ws):
            if average:
                T, ax = w["T"], 1 - CROSS[w["tref"]]
                A, Ut = arrs[1 + i], U
                if w["flip"]:
                    A, Ut = A[::-1], [1.0 - x for x in reversed(U)]
                T.set_rows(ax, q, Ut, A)
            else:
                w["side"] = geom._side_from(keys[i], q, U, arrs[1 + i])
        if S.count(along) != n_before or q != d_before:
            what = (f"degree {d_before} → {q}, " if q != d_before else "") + f"{n_before} → {S.count(along)} CVs"
            notes.append(f"{'/'.join(w['edge'] for w in ws)}: along the edge {what} (the target's layout)")

    if average:
        tjobs = []
        for w in work:
            w["side"], tside = _halfway(S, w)
            tjobs.append((w["T"], w["tref"], w["c"], tside))
        for T, tref, c, tside in tjobs:
            solve(T, [(tref, c, tside)], tangent)
    solve(S, [(w["edge"], w["c"], w["side"]) for w in work], tangent, notes)
    return S, targets


def _halfway(S, w):
    """The common rows two averaged surfaces meet on: (side for S, side for the
    target, running the target's way). Each keeps its own cross tangent
    length; the direction and curvature rows are the means of the two."""
    edge, c, T, tref = w["edge"], w["c"], w["T"], w["tref"]
    Gs, Ua, Ue, p, q = S.std(edge)
    Hs, Va, _, pt, _ = T.std(tref)
    if w["flip"]:
        Hs = Hs[:, ::-1]
    n_e = Gs.shape[1]
    out = {"deg": q, "knots": list(Ue), "P": 0.5 * (Gs[0] + Hs[0]), "D": None, "DD": None, "DDD": None}
    if c >= 1:
        ks, kt = _consts(Ua, p), _consts(Va, pt)
        Ne = salign.basis_matrix(Ue, q, n_e, np.linspace(0.0, 1.0, 65))
        A, B = ks["c1"] * (Gs[1] - Gs[0]), kt["c1"] * (Hs[1] - Hs[0])
        aS, aT = _speed(Ne, A), _speed(Ne, B)
        if aS <= 1e-300 or aT <= 1e-300:
            raise MatchError(f"{edge}: a surface has no cross tangent there (collapsed rows)")
        W = 0.5 * (A / aS - B / aT)              # into S; -W into the target
        out["D"] = W
        if c >= 2:
            DS = ks["c2"] * ((Gs[2] - Gs[1]) / ks["d2"] - (Gs[1] - Gs[0]) / ks["d1"])
            DT = kt["c2"] * ((Hs[2] - Hs[1]) / kt["d2"] - (Hs[1] - Hs[0]) / kt["d1"])
            out["DD"] = 0.5 * (DS / aS ** 2 + DT / aT ** 2)
        if c >= 3:              # odd: the target's own one points into it, so it enters with a minus
            TS = sum(wi * Gs[i] for i, wi in enumerate(ks["w3"]))
            TT = sum(wi * Hs[i] for i, wi in enumerate(kt["w3"]))
            out["DDD"] = 0.5 * (TS / aS ** 3 - TT / aT ** 3)
    tside = dict(out)
    if out["D"] is not None:
        tside["D"] = -out["D"]
    if out["DDD"] is not None:
        tside["DDD"] = -out["DDD"]
    if w["flip"]:
        r = lambda X: None if X is None else X[::-1].copy()
        tside = {"deg": q, "knots": [1.0 - x for x in reversed(Ue)], "P": r(tside["P"]),
                 "D": r(tside["D"]), "DD": r(tside["DD"]), "DDD": r(tside["DDD"])}
    return out, tside


def match_text(obj, text, jobs_, tangent=1.0, keep="G0", ends="EDGE", report=None):
    """Brep text of the (untrimmed, single) surface in `text` with the edges in
    jobs_ = [(edge, (target obj, ref), level)] matched. Coordinates: obj local."""
    S, _ = match_nets(Net.of(obj, text), jobs_, tangent, keep, ends, False, report, obj)
    return S.text(obj)


# --------------------------------------------------------------------------
# Measuring (the Continuity Check's analysis)
# --------------------------------------------------------------------------

def measure(obj, edge, tobj, tref):
    """Worst {"gap", "angle" (degrees), "curv" (relative), "rate" (relative,
    G3)} along the edge against its target, as Continuity Check measures them;
    angle / curv / rate None for a curve target."""
    from . import contcheck
    if tobj.type != 'CURVE':
        r = contcheck.analyze({"a": contcheck.side_key(obj, edge), "b": contcheck.side_key(tobj, tref)})
        return {"gap": float(r["gap"].max()), "angle": float(r["ang"].max()), "curv": float(r["crel"].max()),
                "rate": contcheck._worst(r["g3rel"])}
    kc = features.input_curve(tobj, tref)
    t0, t1 = kc.parameter_range()
    ts = np.linspace(t0, t1, 801)
    C = np.array([kc.evaluate(t) for t in ts], float)
    M = np.array(geom.world_matrix(obj), float)          # double precision (mathutils is float32)
    surf = store.get_surfaces(obj)[0]
    E = np.array(geom.boundary_points(surf, edge, 61), float) @ M[:3, :3].T + M[:3, 3]
    gap = 0.0
    g = 0.5 * (5 ** 0.5 - 1)
    for x in E:
        i = int(np.argmin(np.linalg.norm(C - x, axis=1)))
        a, b = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
        f = lambda t: float(np.linalg.norm(np.array(kc.evaluate(t), float) - x))
        for _ in range(60):
            c_, d_ = b - g * (b - a), a + g * (b - a)
            if f(c_) < f(d_):
                b = d_
            else:
                a = c_
        gap = max(gap, f(0.5 * (a + b)))
    return {"gap": gap, "angle": None, "curv": None, "rate": None}


# --------------------------------------------------------------------------
# History step
# --------------------------------------------------------------------------

def apply_match(obj, text, inputs, p):
    jobs_ = [(e, inp, c) for e, inp, c in zip(p["edges"], inputs, p["cont"])]
    return match_text(obj, text, jobs_, float(p.get("tangent", 1.0)), p.get("keep", "G0"), p.get("ends", "EDGE"))


def _register_step():
    from . import steps
    steps.APPLY["match_surface"] = apply_match
    steps.LABEL["match_surface"] = lambda p: "Match · " + ", ".join(
        f"{e} {c}" for e, c in zip(p.get("edges", []), p.get("cont", [])))
    features.NATURAL_STEPS.add("match_surface")


_register_step()


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty  # noqa: E402

from .ops import _kernel_ok, _select_only  # noqa: E402

_ITEMS = [("G0", "G0", "Position: the edge row is the target's edge row"),
          ("G1", "G1", "Tangent: the next row continues the target's tangent rows"),
          ("G2", "G2", "Curvature: the third row continues the target's curvature rows"),
          ("G3", "G3", "Curvature rate: the fourth row continues the target's third-derivative rows "
                       "(degree 3 or more across, 4 rows at the edge)")]
_KEEP_ITEMS = [("NONE", "None", "The far edge may change"),
               ("G0", "Position", "The far edge stays where it is"),
               ("G1", "Tangent", "The far edge keeps its position and tangent"),
               ("G2", "Curvature", "The far edge keeps its position, tangent and curvature"),
               ("G3", "Curvature Rate", "The far edge keeps its position, tangent, curvature and curvature rate")]
_END_ITEMS = [("EDGE", "Whole Edge", "The edge runs end to end along the whole target edge"),
              ("CLOSEST", "Closest Points", "The edge runs along the stretch of the target between the points "
                                            "nearest its ends (part of a longer edge)")]


def _cont_prop(edge):
    return EnumProperty(name=edge, items=_ITEMS, default="G2",
                        description=f"Continuity of edge {edge} to its target")


class NPROTO_OT_match_surface(salign.EdgePicking, bpy.types.Operator):
    """Match boundary edges of an untrimmed surface to target edges by their CVs:
    the edge row, tangent row, curvature row and curvature-rate row are set from
    the target's rows, exactly G0 / G1 / G2 / G3 (layouts that differ are made one first). Pick one or
    more edges of the surface (Enter to finish); picking another surface's edge
    right after one sets its target, otherwise the touching edge is found"""
    bl_idname = "nproto.match_surface"
    bl_label = "Match Surface"
    bl_options = {'REGISTER', 'UNDO'}

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    cont_U0: _cont_prop("U0")
    cont_U1: _cont_prop("U1")
    cont_V0: _cont_prop("V0")
    cont_V1: _cont_prop("V1")
    tangent: FloatProperty(name="Tangent", default=1.0, min=0.05, soft_max=3.0,
                           description="Scale of the cross tangent at the matched edges (1 = the surface's own length)")
    keep: EnumProperty(name="Keep Other End", items=_KEEP_ITEMS, default="G0",
                       description="What the edge opposite a matched edge keeps (rows are inserted when needed)")
    ends: EnumProperty(name="Edge Ends", items=_END_ITEMS, default="EDGE",
                       description="Which stretch of the target edge the edge runs along")
    average: BoolProperty(name="Average Both", default=False,
                          description="Move both surfaces to meet halfway (the targets must be untrimmed surfaces; "
                                      "no history: both are edited like Edit CVs)")

    def draw(self, context):
        lay = self.layout
        try:
            _, edges = salign.parse_picks(salign._decode_inputs(self.inputs))
        except Exception:
            edges = []
        col = lay.column(align=True)
        for e, _ in edges:
            row = col.row(align=True)
            row.label(text=e)
            row.prop(self, "cont_" + e, expand=True)
        lay.prop(self, "tangent")
        lay.prop(self, "keep")
        row = lay.row()
        row.enabled = not self.average
        row.prop(self, "ends")
        lay.prop(self, "average")

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            obj, jobs_ = self.picked_jobs(context, "matched")
        except Exception as exc:
            self.report({'WARNING'}, str(exc))
            return {'CANCELLED'}
        notes = []
        hist = features.history_enabled(context) and not self.average
        pushed = False
        try:
            for e, (t, _), _ in jobs_:
                if hist and t is not obj and features.depends_on(t, obj):
                    raise MatchError(f"{t.name} is built from {obj.name}: matching {e} to it would loop "
                                     "(Bake one of them first)")
            if self.average:
                S, Ts = match_nets(Net.of(obj), jobs_, self.tangent, self.keep, "EDGE", True, notes, obj)
                baked = []
                for o, net in [(obj, S)] + list(Ts.values()):
                    store.set_geometry_text(o, net.text(o), context, keep_untrimmed=False)
                    o[store.PROP_NATURAL] = 1
                    if features.has_history(o):
                        features.clear_history(o)
                        baked.append(o.name)
                if baked:
                    notes.append(f"construction history removed from {', '.join(baked)} (Average edits both)")
            else:
                text = match_text(obj, obj[store.PROP_BREP], jobs_, self.tangent, self.keep, self.ends, notes)
                if hist:
                    p = {"edges": [e for e, _, _ in jobs_], "cont": [c for _, _, c in jobs_],
                         "tangent": float(self.tangent), "keep": self.keep, "ends": self.ends}
                    features.push_step(obj, "match_surface", [t for _, t, _ in jobs_], p)
                    pushed = True
                    features.rebuild(obj, context, force=True)
                else:
                    store.set_geometry_text(obj, text, context, keep_untrimmed=False)
                obj[store.PROP_NATURAL] = 1
        except Exception as exc:
            if pushed:
                try:
                    features.remove_step(obj, len(features.stack(obj)) - 1)
                    features.rebuild(obj, context, force=True)
                except Exception:
                    pass
            if isinstance(exc, MatchError):
                self.report({'WARNING'}, f"Match Surface: {exc}")
            else:
                self.report({'ERROR'}, f"Match Surface failed: {exc}")
            return {'CANCELLED'}
        self.report(*self._summary(context, obj, jobs_, notes))
        _select_only(context, [obj])
        return {'FINISHED'}

    def _summary(self, context, obj, jobs_, notes):
        """(level, text): what each edge reaches, as Continuity Check measures it."""
        from . import contcheck
        v = getattr(context.window_manager, "nproto_view", None)
        tol = contcheck.tolerances(v) if v is not None else {"G0": 1e-5, "G1": 0.1, "G2": 0.05, "G3": 0.05}
        parts, bad = [], []
        for e, (t, r), c in jobs_:
            lv = _level(c) if t.type != 'CURVE' else 0
            try:
                m = measure(obj, e, t, r)
            except Exception as exc:
                parts.append(f"{e}→{t.name} {r}: not measured ({exc})")
                continue
            txt = f"{e}→{t.name} {r} G{lv}: gap {m['gap']:.2g}"
            fails = m["gap"] > tol["G0"]
            if lv >= 1 and m["angle"] is not None:
                txt += f", {m['angle']:.3g}°"
                fails |= m["angle"] > tol["G1"]
            if lv >= 2 and m["curv"] is not None:
                txt += f", curvature {100 * m['curv']:.2g}%"
                fails |= m["curv"] > tol["G2"]
            if lv >= 3 and m["rate"] is not None:
                txt += f", curvature rate {contcheck.fmt('G3', m['rate'])}"
                fails |= not m["rate"] <= tol["G3"]
            try:
                if salign.overlaps(obj, e, t, r):
                    txt += " (the target lies on the surface's own side: overlapping)"
                    fails = True
            except Exception:
                pass
            (bad if fails else parts).append(txt)
        text = "Matched " + "; ".join(parts + bad) + (f"  ({'; '.join(notes)})" if notes else "")
        return ({'WARNING'} if bad else {'INFO'}), text


classes = (NPROTO_OT_match_surface,)
