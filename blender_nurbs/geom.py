"""Geometry helpers: Blender curves -> kernel curves, sampling, blend lofts.

Conventions
-----------
* Kernel geometry of a NURBS result object lives in that object's LOCAL space;
  world = obj.matrix_world @ local. Inputs are always converted to WORLD space.
* Surface CV nets from `ext.surface_cvs` are u-major: P[iu * count_v + iv].
* An input "ref" is a string: "S<i>" = spline i of a curve object,
  "U0"/"U1"/"V0"/"V1" = an iso boundary of a NURBS object's surface.
"""

import math

import numpy as np
from mathutils import Matrix, Vector

from .kernel import load as _load_kernel


def world_matrix(obj):
    """World matrix from stored data (loc/rot/scale + parent chain).

    obj.matrix_world is runtime data in Blender 5.x: right after an undo (which
    the Adjust Last Operation panel does on every change) it can read as
    identity until the depsgraph runs again. matrix_basis is always valid."""
    M = obj.matrix_basis.copy()
    child = obj
    while child.parent is not None:
        par = child.parent
        M = par.matrix_basis @ child.matrix_parent_inverse @ M
        child = par
    return M


def _k():
    sm = _load_kernel()
    if sm is None:
        raise RuntimeError("USD BRep kernel not loaded")
    return sm, sm.blender_ext


# ---------------------------------------------------------------------------
# B-spline basics (pure numpy) -- used for Blender's own knot modes and for
# fitting compatible blend rows.
# ---------------------------------------------------------------------------

def find_span(n, p, t, U):
    """n = last CV index, p = degree, U = full knot vector."""
    if t >= U[n + 1]:
        return n
    if t <= U[p]:
        return p
    lo, hi = p, n + 1
    mid = (lo + hi) // 2
    while t < U[mid] or t >= U[mid + 1]:
        if t < U[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def basis_funs(i, t, p, U):
    N = [0.0] * (p + 1)
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = t - U[i + 1 - j]
        right[j] = U[i + j] - t
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            tmp = N[r] / denom if denom != 0.0 else 0.0
            N[r] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        N[j] = saved
    return N


def eval_bspline(P, W, U, p, t):
    """Evaluate a (rational) B-spline curve with arbitrary knots."""
    n = len(P) - 1
    span = find_span(n, p, t, U)
    N = basis_funs(span, t, p, U)
    acc = np.zeros(3)
    wsum = 0.0
    for j in range(p + 1):
        k = span - p + j
        w = W[k] if W is not None else 1.0
        acc += N[j] * w * np.asarray(P[k])
        wsum += N[j] * w
    return acc / wsum if wsum else acc


def clamped_uniform_knots(n_cvs, degree):
    inner = n_cvs - degree - 1
    return ([0.0] * (degree + 1)
            + [(i + 1) / (inner + 1) for i in range(inner)]
            + [1.0] * (degree + 1))


def interp_matrix(params, U, p, n_cvs):
    A = np.zeros((len(params), n_cvs))
    for r, t in enumerate(params):
        span = find_span(n_cvs - 1, p, t, U)
        N = basis_funs(span, t, p, U)
        A[r, span - p:span + 1] = N
    return A


def row_params(rows):
    """Averaged chord-length parameters (0..1) shared by point rows."""
    N = len(rows[0])
    acc = np.zeros(N)
    for R in rows:
        d = np.linalg.norm(np.diff(R, axis=0), axis=1)
        tot = d.sum()
        if tot > 0:
            acc[1:] += np.cumsum(d) / tot
    params = acc / len(rows)
    params[0], params[-1] = 0.0, 1.0
    if np.min(np.diff(params)) < 1e-6 / N:
        # repeated samples (e.g. a sweep folding over a cusp): blend in uniform
        # parameters, else the interpolation matrix is singular and CVs explode
        params = 0.8 * params + 0.2 * np.linspace(0.0, 1.0, N)
    return params


def fit_rows(rows, degree=3, tol=None, fallback=True, max_cvs=None, params=None):
    """Fit several point rows (same sample count) with ONE shared
    parameterization and knot vector, so the resulting curves are compatible.

    rows: list of (N,3) arrays. With `tol` (one value, or one per row), uses
    the fewest CVs that fit every row within its tol; otherwise interpolates
    (N CVs). params: the shared parameters (default: row_params(rows)).
    Returns (cv_rows list of (n,3), knots, degree)."""
    N = len(rows[0])
    p = min(degree, N - 1)
    params = row_params(rows) if params is None else np.asarray(params, float)
    B = np.hstack(rows)  # (N, 3*len(rows))

    # Lightest net first: least-squares fit with the fewest CVs that keeps every
    # row within `tol` of its samples (ends exact). Falls back to interpolation.
    tols = None if tol is None else np.broadcast_to(np.asarray(tol, float), (len(rows),))
    if tols is not None and np.all(tols > 0):
        for n_cvs in range(p + 1, N if max_cvs is None else min(N, max_cvs + 1)):
            U = _ls_knots(params, n_cvs, p)
            if _empty_span(params, U):
                continue
            X = _ls_fit(params, B, U, p, n_cvs)
            A = interp_matrix(params, U, p, n_cvs)
            err = np.abs(A @ X - B).reshape(N, -1, 3)
            if np.all(np.linalg.norm(err, axis=2).max(axis=0) <= tols):
                return [X[:, 3 * i:3 * i + 3] for i in range(len(rows))], U, p
        if not fallback:
            return None

    # knot averaging (Piegl & Tiller eq. 9.8)
    U = [0.0] * (p + 1)
    for j in range(1, N - p):
        U.append(sum(params[j:j + p]) / p)
    U += [1.0] * (p + 1)
    A = interp_matrix(params, U, p, N)
    X = np.linalg.solve(A, B)
    out = [X[:, 3 * i:3 * i + 3] for i in range(len(rows))]
    return out, U, p


def _ls_knots(params, n_cvs, p):
    """Knot placement for least squares (Piegl & Tiller eq. 9.68/9.69)."""
    m = len(params) - 1
    n = n_cvs - 1
    U = [0.0] * (p + 1)
    d = (m + 1) / (n - p + 1)
    for j in range(1, n - p + 1):
        i = int(j * d)
        a = j * d - i
        U.append((1 - a) * params[i - 1] + a * params[i])
    U += [1.0] * (p + 1)
    return U


def _empty_span(params, U):
    """True when a knot span holds no sample strictly inside it: nothing pins
    the curve there, and a fit with nearly as many CVs as samples swings its
    end CVs freely (a sweep with 28 CVs on 32 rows got a first span of 0.009
    with no sample in it and folded back behind its profile)."""
    t = np.asarray(params, float)
    ks = np.unique(np.asarray(U, float))
    eps = 1e-12 * max(1.0, float(ks[-1] - ks[0]))
    for a, b in zip(ks[:-1], ks[1:]):
        if not np.any((t > a + eps) & (t < b - eps)):
            return True
    return False


def _ls_fit(params, B, U, p, n_cvs):
    """Least squares with first/last CV pinned to the first/last samples."""
    A = interp_matrix(params, U, p, n_cvs)
    X = np.zeros((n_cvs, B.shape[1]))
    X[0], X[-1] = B[0], B[-1]
    if n_cvs > 2:
        R = B - np.outer(A[:, 0], X[0]) - np.outer(A[:, -1], X[-1])
        Ai = A[:, 1:-1]
        X[1:-1] = np.linalg.lstsq(Ai, R, rcond=None)[0]
    return X


# ---------------------------------------------------------------------------
# Blender curve objects -> kernel curves (world space)
# ---------------------------------------------------------------------------

def _blender_knots(n_cvs, order, endpoint, bezier, cyclic):
    """Knot vector Blender uses for a legacy NURBS spline (non-cyclic)."""
    p = order - 1
    m = n_cvs + order
    if endpoint:
        return clamped_uniform_knots(n_cvs, p), True
    if bezier:
        # Bezier knot mode: knots repeated `p` times, endpoints clamped-ish.
        U = []
        k = 0.0
        while len(U) < m:
            U.extend([k] * p)
            k += 1.0
        U = U[:m]
        return [u / U[-1] for u in U], False
    return [i / (m - 1) for i in range(m)], False


def spline_to_kernel_curve(spline, matrix):
    """Exact where possible (Bezier, endpoint NURBS, poly), otherwise sampled
    and interpolated by the kernel."""
    sm, ext = _k()
    M = matrix

    if spline.type == 'BEZIER':
        bp = spline.bezier_points
        segs = list(range(len(bp) - 1))
        if spline.use_cyclic_u and len(bp) > 1:
            segs.append(len(bp) - 1)
        if not segs:
            raise ValueError("Bezier spline needs at least 2 points")
        pts = []
        for s in segs:
            a, b = bp[s], bp[(s + 1) % len(bp)]
            if not pts:
                pts.append(M @ a.co)
            pts += [M @ a.handle_right, M @ b.handle_left, M @ b.co]
        knots = [0.0] * 4
        for s in range(1, len(segs)):
            knots += [float(s)] * 3
        knots += [float(len(segs))] * 4
        return ext.curve_from_cvs(3, [tuple(p) for p in pts], knots)

    pts4 = [p.co for p in spline.points]
    pts = [M @ Vector(c[:3]) for c in pts4]
    weights = [c[3] for c in pts4]
    if spline.type == 'POLY':
        if spline.use_cyclic_u:
            pts.append(pts[0])
        knots = [0.0, 0.0] + [float(i) for i in range(1, len(pts) - 1)] + [float(len(pts) - 1)] * 2
        return ext.curve_from_cvs(1, [tuple(p) for p in pts], knots)

    # NURBS: exact, for every Blender knot mode (cyclic included)
    order = max(2, min(spline.order_u, len(pts)))
    rational = any(abs(w - 1.0) > 1e-9 for w in weights)
    deg, P, W, U = blender_nurbs_to_clamped(
        [tuple(v) for v in pts], weights, order, spline.use_cyclic_u,
        spline.use_endpoint_u, spline.use_bezier_u)
    return ext.curve_from_cvs(deg, [tuple(map(float, v)) for v in P], [float(u) for u in U],
                              [float(w) for w in W] if rational else None)


def blender_knots(points_num, order, cyclic, endpoint, bezier):
    """Blender's knot vector for a NURBS spline (mirrors Blender's
    curve_nurbs calculate_knots, used by legacy and new curves alike)."""
    is_bezier = bezier
    is_end_point = endpoint
    repeat_inner = order - 1 if is_bezier else 1
    head = (order - (1 if cyclic else 0)) if is_end_point else (min(2, repeat_inner) if is_bezier else 1)
    tail = 2 * order - 1 if cyclic else (order if is_end_point else 0)
    knot_count = points_num + order + (order - 1 if cyclic else 0)
    knots = [0.0] * knot_count
    r = head
    current = 0.0
    offset = 1 if (is_end_point and cyclic) else 0
    if offset:
        knots[0] = current
        current += 1.0
    for i in range(offset, knot_count - tail):
        knots[i] = current
        r -= 1
        if r == 0:
            current += 1.0
            r = repeat_inner
    tail_index = knot_count - tail
    for i in range(tail):
        knots[tail_index + i] = current + (knots[i] - knots[0])
    return knots


def _insert_knot(Pw, U, p, t):
    """Boehm single knot insertion on homogeneous control points."""
    n = len(Pw) - 1
    k = max(i for i in range(len(U) - 1) if U[i] <= t + 1e-12)
    k = min(k, n)
    s = sum(1 for u in U if abs(u - t) <= 1e-12)
    Q = [Pw[i] for i in range(0, k - p + 1)]
    for i in range(k - p + 1, k - s + 1):
        a = (t - U[i]) / (U[i + p] - U[i])
        Q.append(a * Pw[i] + (1.0 - a) * Pw[i - 1])
    Q += [Pw[i] for i in range(k - s, n + 1)]
    return Q, U[:k + 1] + [t] + U[k + 1:]


def _clamp_start(Pw, U, p, a):
    """Make the curve start exactly at parameter a with an (p+1)-fold knot."""
    while sum(1 for u in U if abs(u - a) <= 1e-12) < p:
        Pw, U = _insert_knot(Pw, U, p, a)
    first = next(i for i, u in enumerate(U) if abs(u - a) <= 1e-12)
    last = max(i for i, u in enumerate(U) if abs(u - a) <= 1e-12)
    ia = max(first, last - p + 1)
    return list(Pw[ia - 1:]), [a] + list(U[ia:])


def blender_nurbs_to_clamped(points, weights, order, cyclic, endpoint, bezier):
    """Exact clamped (degree, CVs, weights, knots) of a Blender NURBS spline."""
    p = order - 1
    n = len(points)
    if bezier and cyclic and p > 1 and n % p != 0:
        raise ValueError(f"Blender cannot evaluate this spline: a cyclic order-{order} spline with "
                         f"Bezier knots needs a multiple of {p} points (has {n})")
    Pw = [np.array([x * w, y * w, z * w, w]) for (x, y, z), w in zip(points, weights)]
    if cyclic:
        Pw = Pw + Pw[:p]
    U = blender_knots(n, order, cyclic, endpoint, bezier)
    U = [float(u) for u in U[:len(Pw) + order]]
    a, b = U[p], U[len(Pw)]
    Pw, U = _clamp_start(Pw, U, p, a)
    # clamp the end by mirroring: reversed curve, clamp its start, mirror back
    lo, hi = U[0], U[-1]
    Pr, Ur = _clamp_start(Pw[::-1], [lo + hi - u for u in reversed(U)], p, lo + hi - b)
    P_new = Pr[::-1]
    U_new = [lo + hi - u for u in reversed(Ur)]
    span = (b - a) or 1.0
    U_new = [(u - a) / span for u in U_new]
    W = [float(q[3]) for q in P_new]
    P = [q[:3] / q[3] for q in P_new]
    return p, P, W, U_new


def ring_order(curves, n=16):
    """Order of the sections around a closed loft: the given one, unless a
    nearest-neighbour ring through their centres is clearly shorter (a
    selection comes in no particular order)."""
    C = [sample_curve(c, n).mean(axis=0) for c in curves]
    def length(order):
        return sum(np.linalg.norm(C[order[i]] - C[order[(i + 1) % len(order)]]) for i in range(len(order)))
    given = list(range(len(C)))
    greedy, left = [0], set(given[1:])
    while left:
        k = min(left, key=lambda j: np.linalg.norm(C[j] - C[greedy[-1]]))
        greedy.append(k)
        left.remove(k)
    return greedy if length(greedy) < 0.98 * length(given) else given


def chain_order(curves, n=16):
    """Order of the sections of an open loft: the given one, unless a chain
    through their centres from one end to the other is clearly shorter (a
    selection comes in no particular order: three stacked circles taken
    bottom, top, middle loft up and fold back down)."""
    C = [sample_curve(c, n).mean(axis=0) for c in curves]
    k = len(C)
    if k < 3:
        return list(range(k))

    def length(order):
        return sum(np.linalg.norm(C[order[i]] - C[order[i + 1]]) for i in range(len(order) - 1))
    given = list(range(k))
    best = given
    for start in range(k):                     # every start: greedy nearest neighbour
        chain, left = [start], set(given) - {start}
        while left:
            j = min(left, key=lambda q: np.linalg.norm(C[q] - C[chain[-1]]))
            chain.append(j)
            left.remove(j)
        if length(chain) < length(best) - 1e-12:
            best = chain
    if length(best) >= 0.98 * length(given):
        return given
    return best if best[0] <= best[-1] else best[::-1]


def is_closed_curve(curve, tol=1e-6):
    t0, t1 = curve.parameter_range()
    a, b = np.array(curve.evaluate(t0)), np.array(curve.evaluate(t1))
    P = sample_curve(curve, 16)
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    return float(np.linalg.norm(a - b)) <= tol * size


def _area_vector(P):
    c = P.mean(axis=0)
    return 0.5 * np.cross(P[:-1] - c, P[1:] - c).sum(axis=0)


def reseat_closed(curve, t):
    """The same closed curve starting (and ending) at parameter t: t inserted
    to full multiplicity, the two pieces swapped and joined at the old start.
    Exact (rational curves too); the parameter range stays the same."""
    from .kernel.classa import insert_knot
    _, ext = _k()
    d = ext.curve_cvs(curve)
    p = int(d["degree"])
    U = [float(k) for k in d["knots"]]
    a, b = U[0], U[-1]
    eps = 1e-9 * (b - a)
    if t - a <= 1e-6 * (b - a) or b - t <= 1e-6 * (b - a):      # starts there already
        return curve
    t = next((k for k in U if abs(k - t) <= 1e-6 * (b - a)), t)  # (next to) a knot: use it
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    Pw = np.c_[P * w[:, None], w]
    while sum(abs(k - t) <= 1e-12 for k in U) < p:
        Pw, U = insert_knot(Pw, U, p, t)
    Pw = np.asarray(Pw, float)
    s0 = next(i for i, k in enumerate(U) if abs(k - t) <= 1e-12)    # t sits at U[s0 .. s0+p-1]
    j = s0 - 1                                                       # the CV the curve passes at t
    first, second = Pw[:j + 1], Pw[j:]
    first = first * (second[-1, 3] / first[0, 3])                    # same point (and weight) where they join
    K2 = [t] * (p + 1) + U[s0 + p:len(U) - (p + 1)] + [b] * (p + 1)
    K1 = [a] * (p + 1) + U[p + 1:s0] + [t] * (p + 1)
    knots = K2[:-1] + [k + (b - a) for k in K1[p + 1:]]
    H = np.vstack([second, first[1:]])
    knots = [k - (t - a) for k in knots]
    knots[:p + 1] = [a] * (p + 1)                    # exactly (0.9999999999999999 is another knot to a merge)
    knots[-(p + 1):] = [b] * (p + 1)
    for i in range(1, len(knots)):
        if abs(knots[i] - knots[i - 1]) <= eps:
            knots[i] = knots[i - 1]
    pts = [tuple(float(x) for x in q) for q in H[:, :3] / H[:, 3:]]
    wts = [float(x) for x in H[:, 3]]
    rational = any(abs(x - 1.0) > 1e-12 for x in wts)
    return ext.curve_from_cvs(p, pts, knots, wts if rational else None)


def _turn(a, b):
    """The smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if c < -1.0 + 1e-9:
        return -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def align_closed(prev, cur, n=720, keep_deg=1.0):
    """A closed section turned to run like the one before it (the same way
    round) and to start where it starts (the same direction from its centre,
    carried over from the previous section's plane by the smallest turn):
    with different seams the loft twists between them. A start within
    keep_deg of that is left as it is."""
    if not (is_closed_curve(prev) and is_closed_curve(cur)):
        return cur
    A = sample_curve(prev, 64)
    B = sample_curve(cur, 64)
    na, nb = _area_vector(A), _area_vector(B)
    la, lb = float(np.linalg.norm(na)), float(np.linalg.norm(nb))
    if la < 1e-12 or lb < 1e-12:
        return cur
    na, nb = na / la, nb / lb
    ca, cb = A[:-1].mean(axis=0), B[:-1].mean(axis=0)
    d = cb - ca
    dl = float(np.linalg.norm(d))
    if dl > 1e-12 and abs(na @ d) > 0.2 * dl and abs(nb @ d) > 0.2 * dl:
        same = (na @ d > 0) == (nb @ d > 0)          # both wound the same way along the loft
    else:
        same = float(na @ nb) >= 0.0
    if not same:
        cur = reverse_curve(cur)
        B = B[::-1]
        nb = -nb
    ref = _turn(na, nb) @ (A[0] - ca)
    ref = ref - (ref @ nb) * nb
    if np.linalg.norm(ref) < 1e-12:
        return cur
    ref /= np.linalg.norm(ref)
    t0, t1 = cur.parameter_range()

    def wrap(t):
        return t0 + (t - t0) % (t1 - t0)

    def score(t):                                     # cos of the angle to the start direction
        v = np.array(cur.evaluate(wrap(t))) - cb
        v = v - (v @ nb) * nb
        return float(v @ ref) / (np.linalg.norm(v) or 1.0)
    if score(t0) >= math.cos(math.radians(keep_deg)):
        return cur                                    # starts there already
    ts = np.linspace(t0, t1, n, endpoint=False)
    i = int(np.argmax([score(t) for t in ts]))
    h = (t1 - t0) / n
    lo, hi = ts[i] - h, ts[i] + h
    for _ in range(30):                               # golden section around the best sample
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if score(m1) > score(m2):
            hi = m2
        else:
            lo = m1
    return reseat_closed(cur, wrap(0.5 * (lo + hi)))


def compatible_arrays(items, tol=1e-7):
    """Clamped B-splines on one degree and one knot vector, exactly. items:
    [(degree, knots, CV array (n, dim))] -- any columns (homogeneous CVs, or
    an edge's rows with its cross derivatives side by side). Each range is
    mapped to 0..1, degrees raised (by Bezier pieces) to the highest, knots
    closer than tol taken as one, then every knot inserted into every item to
    its highest multiplicity. Returns (degree, knots, [arrays])."""
    from .kernel.classa import insert_knot, bezier_pieces
    from .curvejoin import elevate
    data = []
    for p, U, A in items:
        p = int(p)
        U = [float(k) for k in U]
        a, b = U[0], U[-1]
        U = [(k - a) / (b - a) for k in U]
        U[:p + 1], U[-(p + 1):] = [0.0] * (p + 1), [1.0] * (p + 1)
        data.append([p, U, np.asarray(A, float)])
    q = max(it[0] for it in data)
    for it in data:
        p, U, A = it
        if p == q:
            continue
        br = sorted(set(U))
        pieces = [elevate(np.asarray(s_, float), q) for s_ in bezier_pieces(A, U, p)]
        it[2] = np.vstack([pieces[0]] + [s_[1:] for s_ in pieces[1:]])
        it[1] = [0.0] * (q + 1) + [k for k in br[1:-1] for _ in range(q)] + [1.0] * (q + 1)
        it[0] = q
    # one value for knots closer than tol, across all items
    vals = sorted({k for _, U, _ in data for k in U[q + 1:-q - 1]})
    reps = []
    for v in vals:
        if reps and v - reps[-1][-1] <= tol:
            reps[-1].append(v)
        else:
            reps.append([v])
    snap = {v: g[0] for g in reps for v in g}
    for it in data:
        it[1] = [snap.get(k, k) for k in it[1]]
    for v in [g[0] for g in reps]:
        mult = max(sum(1 for k in U if k == v) for _, U, _ in data)
        for it in data:
            while sum(1 for k in it[1] if k == v) < mult:
                A, U = insert_knot(it[2], it[1], q, v)
                it[2], it[1] = np.asarray(A, float), U
    U0 = data[0][1]
    if any(len(U) != len(U0) or any(abs(x - y) > 1e-12 for x, y in zip(U, U0)) for _, U, _ in data):
        raise ValueError("the curves could not be given one layout")
    return q, U0, [A for _, _, A in data]


def compatible_curves(curves, tol=1e-7):
    """The sections on one degree and one knot vector, exactly (see
    compatible_arrays). Returns (degree, knots, [homogeneous CVs (m, 4)]).
    (The kernel's own merge nudges knots -- 0.25 to 0.2499999994, 1 to
    0.99999998788 beside 1 -- and leaves knot vectors the kernel then refuses.)"""
    _, ext = _k()
    items = []
    for c in curves:
        d = ext.curve_cvs(c)
        P = np.array(d["points"], float)
        w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
        items.append((int(d["degree"]), d["knots"], np.c_[P * w[:, None], w]))
    try:
        return compatible_arrays(items, tol)
    except ValueError:
        raise ValueError("the sections could not be given one layout")


_SIDE_ROWS = ("P", "D", "DD", "DDD")


def start_coeffs(U, p, k):
    """Weights w (k + 1,) of the k-th derivative at the start of a clamped
    B-spline (degree p, knots U) in its first k + 1 CVs: C^(k)(U[0]) =
    sum_i w_i P_i. The derivative CVs P^(r)_i = (p - r + 1) / (U[i+p+1] -
    U[i+r]) (P^(r-1)_{i+1} - P^(r-1)_i), whose first one is the derivative
    there. Zero (all weights) for k > p."""
    W = np.eye(k + 1)
    if k > p:
        return np.zeros(k + 1)
    for r in range(1, k + 1):
        W = np.array([(p - r + 1) / (U[i + p + 1] - U[i + r]) * (W[i + 1] - W[i]) for i in range(len(W) - 1)])
    return W[0]


def _side_array(s):
    keys = [k for k in _SIDE_ROWS if s.get(k) is not None]
    return keys, np.hstack([np.asarray(s[k], float) for k in keys])


def _side_from(keys, deg, knots, A):
    out = {"deg": int(deg), "knots": [float(k) for k in knots], "P": None, "D": None, "DD": None, "DDD": None}
    for i, k in enumerate(keys):
        out[k] = A[:, 3 * i:3 * i + 3].copy()
    return out


def compatible_sides(sa, sb, tol=1e-9):
    """Two exact loft sides ({"deg", "knots", "P", "D", "DD"}) on one degree and
    one knot vector, exactly: the edge rows and their cross-derivative rows
    are curves on the same layout, so they are raised and refined together."""
    (ka, A), (kb, B) = _side_array(sa), _side_array(sb)
    q, U, (A2, B2) = compatible_arrays([(sa["deg"], sa["knots"], A), (sb["deg"], sb["knots"], B)], tol)
    return _side_from(ka, q, U, A2), _side_from(kb, q, U, B2)


def raise_array(p, U, A, q):
    """A clamped B-spline (CV array, any columns) raised from degree p to q,
    exactly (Bezier pieces elevated, knots at full multiplicity)."""
    from .kernel.classa import bezier_pieces
    from .curvejoin import elevate
    A = np.asarray(A, float)
    if p >= q:
        return p, [float(k) for k in U], A
    br = sorted(set(float(k) for k in U))
    pieces = [elevate(np.asarray(s_, float), q) for s_ in bezier_pieces(A, U, p)]
    A = np.vstack([pieces[0]] + [s_[1:] for s_ in pieces[1:]])
    return q, [br[0]] * (q + 1) + [k for k in br[1:-1] for _ in range(q)] + [br[-1]] * (q + 1), A


def concat_arrays(items, widths):
    """Clamped B-splines joined end to end into ONE (exact): items [(degree,
    knots, CV array)] whose last CV meets the next one's first (they are
    averaged), widths the share of 0..1 each gets. Joints are knots of full
    multiplicity (C0 in the parameter; the geometry is as smooth as the
    pieces meet). Returns (degree, knots, array)."""
    q = max(int(p) for p, _, _ in items)
    tot = float(sum(widths)) or 1.0
    widths = [w / tot for w in widths]
    knots, rows, s = [0.0] * (q + 1), None, 0.0
    for (p, U, A), w in zip(items, widths):
        U = [float(k) for k in U]
        a, b = U[0], U[-1]
        U = [(k - a) / (b - a) for k in U]
        _, U, A = raise_array(int(p), U, A, q)
        if rows is None:
            rows = A
        else:
            rows = np.vstack([rows[:-1], 0.5 * (rows[-1] + A[0]), A[1:]])
            knots += [s] * q
        knots += [s + w * k for k in U[q + 1:-q - 1]]
        s += w
    knots += [1.0] * (q + 1)
    return q, knots, rows


def end_speeds(p, U, P):
    """|dC/dt| at the start and the end of a clamped B-spline on 0..1 (CVs P)."""
    U = [float(k) for k in U]
    a, b = U[0], U[-1]
    U = [(k - a) / (b - a) for k in U]
    s0 = p / ((U[p + 1] - U[1]) or 1.0) * float(np.linalg.norm(P[1] - P[0]))
    n = len(P) - 1
    s1 = p / ((U[n + p] - U[n]) or 1.0) * float(np.linalg.norm(P[n] - P[n - 1]))
    return s0, s1


def speed_widths(speeds):
    """Parameter widths for pieces joined end to end so the speed along the
    joined curve does not jump at a joint (C1 where the pieces are tangent):
    speeds [(start, end)] of each piece on its own 0..1."""
    w = [1.0]
    for (_, e), (s, _) in zip(speeds[:-1], speeds[1:]):
        w.append(w[-1] * (s / e) if e > 1e-15 and s > 1e-15 else w[-1])
    return w


def sub_side(s, lo, hi):
    """The stretch lo..hi (0..1 along the edge) of an exact loft side, exactly
    (knot insertion), with its knots mapped back to 0..1."""
    from .kernel.classa import insert_knot
    p = int(s["deg"])
    keys, A = _side_array(s)
    U = [float(k) for k in s["knots"]]
    snap = lambda t: next((k for k in U if abs(k - t) <= 1e-9), t)
    lo, hi = snap(lo), snap(hi)
    for t in (lo, hi):
        if 0.0 < t < 1.0:
            while sum(1 for k in U if k == t) < p:
                A, U = insert_knot(A, U, p, t)
                A = np.asarray(A, float)
    last_lo = max(i for i, k in enumerate(U) if k == lo)
    inner = [k for k in U if lo < k < hi]
    n = p + 1 + len(inner)
    s0 = last_lo - p
    knots = [0.0] * (p + 1) + [(k - lo) / (hi - lo) for k in inner] + [1.0] * (p + 1)
    return _side_from(keys, p, knots, A[s0:s0 + n])


def open_skin(curves, degree=3):
    """Open loft through the sections in order: the sections made compatible
    (one knot vector), each column of matching CVs interpolated in homogeneous
    coordinates on the same averaged chord-length parameters (as closed_skin).
    The surface holds every section exactly. U runs along the loft, V along
    the sections (like the kernel's skin, so its edges keep their names)."""
    sm, ext = _k()
    n = len(curves)
    if n < 2:
        raise ValueError("a loft needs at least 2 sections")
    pu, Uu, Hs = compatible_curves(curves)
    m = len(Hs[0])
    H = np.array(Hs, float)                                                 # (n, m, 4) homogeneous
    P = H[..., :3] / H[..., 3:]
    d = np.linalg.norm(np.diff(P, axis=0), axis=2)                          # (n-1, m)
    tot = d.sum(axis=0)
    d = np.where(tot > 1e-12, d / np.where(tot > 1e-12, tot, 1.0), 1.0 / (n - 1))
    t = np.concatenate([[0.0], np.cumsum(d.mean(axis=1))])
    t /= t[-1]
    q = max(1, min(int(degree), n - 1))
    Uv = [0.0] * (q + 1) + [float(np.mean(t[j:j + q])) for j in range(1, n - q)] + [1.0] * (q + 1)
    A = interp_matrix(t, Uv, q, n)
    X = np.linalg.solve(A, H.reshape(n, -1)).reshape(n, m, 4)
    if (X[..., 3] <= 1e-9).any():
        raise ValueError("these sections give a loft with non-positive weights")
    pts = [tuple(float(x) for x in X[i, j, :3] / X[i, j, 3]) for i in range(n) for j in range(m)]
    wts = [float(X[i, j, 3]) for i in range(n) for j in range(m)]
    rational = any(abs(w - 1.0) > 1e-12 for w in wts)
    return ext.surface_from_cvs(q, pu, pts, Uv, Uu, wts if rational else None)


def closed_skin(curves, degree=3):
    """Closed (periodic) loft: through every section in order and back to the
    first, smooth all round (C2 for degree 3, the seam included). The sections
    are made compatible (one knot vector), then each column of matching CVs is
    interpolated by a periodic spline in homogeneous coordinates, on the same
    averaged chord-length parameters for every column -- so the surface holds
    every section exactly, rational ones (exact circles) too."""
    from . import curvetools
    sm, ext = _k()
    n = len(curves)
    if n < 3:
        raise ValueError("a closed loft needs at least 3 sections")
    pu, Uu, Hs = compatible_curves(curves)
    m = len(Hs[0])
    H = np.array(Hs, float)                                                 # (n, m, 4) homogeneous
    P = H[..., :3] / H[..., 3:]
    d = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=2)                  # section i -> i+1, round
    tot = d.sum(axis=0)
    d = np.where(tot > 1e-12, d / np.where(tot > 1e-12, tot, 1.0), 1.0 / n)
    t = np.concatenate([[0.0], np.cumsum(d.mean(axis=1))[:-1]])
    q = max(1, min(int(degree), n - 1))
    A = curvetools.Basis(n, q, True).matrix(t)
    X = np.linalg.solve(A, H.reshape(n, -1)).reshape(n, m, 4)
    if (X[..., 3] <= 1e-9).any():
        raise ValueError("these sections give a closed loft with non-positive weights")
    cols = []
    for j in range(m):
        dv, Pc, Wc, Uv = blender_nurbs_to_clamped((X[:, j, :3] / X[:, j, 3:4]).tolist(), X[:, j, 3].tolist(),
                                                  q + 1, True, False, False)
        cols.append((Pc, Wc))
    nl = len(cols[0][0])
    # like the kernel's own skin: U runs along the loft (through the sections), V along each section
    pts = [tuple(map(float, cols[j][0][i])) for i in range(nl) for j in range(m)]
    wts = [float(cols[j][1][i]) for i in range(nl) for j in range(m)]
    rational = any(abs(w - 1.0) > 1e-12 for w in wts)
    return ext.surface_from_cvs(dv, pu, pts, [float(u) for u in Uv], Uu, wts if rational else None)


def curve_object_splines(obj):
    if obj.type != 'CURVE':
        return []
    return list(obj.data.splines)


def reverse_curve(curve):
    _, ext = _k()
    d = ext.curve_cvs(curve)
    k = d["knots"]
    a, b = k[0], k[-1]
    rk = [a + b - x for x in reversed(k)]
    w = list(reversed(d["weights"])) if d["weights"] else None
    return ext.curve_from_cvs(d["degree"], list(reversed(d["points"])), rk, w)


def translate_curve(curve, vec):
    _, ext = _k()
    d = ext.curve_cvs(curve)
    pts = [(p[0] + vec[0], p[1] + vec[1], p[2] + vec[2]) for p in d["points"]]
    return ext.curve_from_cvs(d["degree"], pts, d["knots"], d["weights"] or None)


def sample_curve(curve, n):
    t0, t1 = curve.parameter_range()
    return np.array([curve.evaluate(t0 + (t1 - t0) * i / (n - 1)) for i in range(n)])


def sample_curve_kinks(curve, n):
    """sample_curve plus a sample exactly on every knot where the curve may
    kink (multiplicity >= degree): even samples alone cut a chained profile's
    sharp corners, so its preview showed chamfers the result does not have."""
    t0, t1 = curve.parameter_range()
    eps = 1e-9 * (t1 - t0)
    try:
        cv = _load_kernel().blender_ext.curve_cvs(curve)
        deg, U = int(cv["degree"]), [float(k) for k in cv["knots"]]
    except Exception:
        return sample_curve(curve, n)
    kinks, i = [], 0
    while i < len(U):
        j = i
        while j + 1 < len(U) and abs(U[j + 1] - U[i]) <= eps:
            j += 1
        if j - i + 1 >= deg and t0 + eps < U[i] < t1 - eps:
            kinks.append(U[i])
        i = j + 1
    if not kinks:
        return sample_curve(curve, n)
    T = [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]
    T = sorted([t for t in T if all(abs(t - k) > 1e-4 * (t1 - t0) for k in kinks)] + kinks)
    return np.array([curve.evaluate(t) for t in T])


def curve_plane_normal(curve, n=32):
    P = sample_curve(curve, n)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    return Vector(vt[2]).normalized(), Vector(c)


def _square_to(h, L):
    """Unit h made square to the unit line direction L (a world axis when h runs along L)."""
    for cand in (h, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        c = np.asarray(cand, float)
        c = c - (c @ L) * L
        if np.linalg.norm(c) > 1e-6:
            return c / np.linalg.norm(c)
    return np.array([0.0, 0.0, 1.0])


def profile_normal(curve, hint=None, n=32):
    """The plane normal an extrude of `curve` goes along ("Curve Normal").
    A straight curve lies in every plane through it, so the plane alone picks
    a direction at random (rounding decided it): it gets the hint (the work
    plane's normal when the tool started, else world Z) made square to it, as
    Rhino's ExtrudeCrv takes the CPlane's Z. A planar curve's normal is turned
    to the hint's side when one is given (older extrudes without one keep the
    plane's own sign)."""
    P = sample_curve(curve, n)
    c = P.mean(axis=0)
    _, S, vt = np.linalg.svd(P - c)
    h = np.asarray(hint if hint is not None else (0.0, 0.0, 1.0), float)
    h = h / (np.linalg.norm(h) or 1.0)
    if S[0] <= 1e-300 or S[1] <= 1e-7 * S[0]:
        return Vector(_square_to(h, vt[0]))
    nrm = vt[2]
    if hint is not None:
        dot = float(nrm @ h)
        if abs(dot) < 1e-6:                 # square to the hint: a stable choice
            k = int(np.argmax(np.abs(nrm)))
            if nrm[k] < 0:
                nrm = -nrm
        elif dot < 0:
            nrm = -nrm
    return Vector(nrm).normalized()


# ---------------------------------------------------------------------------
# Surface edge sampling for G0/G1/G2/G3 lofts
# ---------------------------------------------------------------------------

class EdgeSample:
    """Samples along a loft input: positions P, outward cross tangent D,
    outward cross second derivative DD (None for plain curves), and the
    third DDD for G3 -- the third derivative of the same cross curve on the
    source (only its part along the normal matters for G3). DDD is worked out
    on first use (_ddd_fn) and is None when the source has no CV net."""

    def __init__(self, P, D=None, DD=None, joints=None, DDD=None):
        self.P = P
        self.D = D
        self.DD = DD
        self.joints = list(joints or [])     # sample indices where the pieces of a chained side meet
        self.N = None                        # the source's unit normals, when known (Align Edges / Surface)
        self._DDD = DDD
        self._ddd_fn = None                  # () -> DDD, called on first use

    @property
    def has_surface(self):
        return self.D is not None

    @property
    def DDD(self):
        if self._DDD is None and self._ddd_fn is not None:
            fn, self._ddd_fn = self._ddd_fn, None
            try:
                self._DDD = fn()
            except Exception as exc:
                print("NURBS loft third derivative:", exc)
        return self._DDD

    @DDD.setter
    def DDD(self, v):
        self._DDD, self._ddd_fn = v, None

    def _pass_ddd(self, out, f):
        """out's DDD as f of ours -- lazily, when ours is not worked out yet."""
        if self._DDD is not None:
            out._DDD = f(self._DDD)
        elif self._ddd_fn is not None:
            out._ddd_fn = lambda: (lambda d: None if d is None else f(d))(self.DDD)
        return out

    def reversed(self):
        r = lambda a: None if a is None else a[::-1].copy()
        n = len(self.P)
        out = EdgeSample(self.P[::-1].copy(), r(self.D), r(self.DD), sorted(n - 1 - j for j in self.joints))
        out.N = r(self.N)
        return self._pass_ddd(out, r)

    def piece(self, i0, i1):
        c = lambda a: None if a is None else a[i0:i1 + 1].copy()
        out = EdgeSample(c(self.P), c(self.D), c(self.DD))
        out.N = c(self.N)
        return self._pass_ddd(out, c)


def sample_surface_edge(surface, which, matrix, n, share=None):
    """EdgeSample along an iso edge; share: only lo..hi of it (shares of its domain)."""
    (u0, u1), (v0, v1) = surface.u_domain(), surface.v_domain()
    s_lo, s_hi = share if share is not None else (0.0, 1.0)
    M = np.array(matrix)
    R = M[:3, :3]
    T = M[:3, 3]
    P, D, DD, UV = [], [], [], []
    along_v = which in ("U0", "U1")
    span = (u1 - u0) if along_v else (v1 - v0)
    h = 1e-4 * span
    for i in range(n):
        s = s_lo + (s_hi - s_lo) * i / (n - 1)
        if along_v:
            u = u0 if which == "U0" else u1
            v = v0 + (v1 - v0) * s
            sign = -1.0 if which == "U0" else 1.0
            pt, du, _ = surface.derivatives(u, v)
            _, du2, _ = surface.derivatives(u - sign * h, v)
            d1 = np.array(du)
            d2 = (np.array(du) - np.array(du2)) / (sign * h)  # one-sided d2/du2
            d = sign * d1
            dd = d2
            UV.append((u, v))
        else:
            v = v0 if which == "V0" else v1
            u = u0 + (u1 - u0) * s
            sign = -1.0 if which == "V0" else 1.0
            pt, _, dv = surface.derivatives(u, v)
            _, _, dv2 = surface.derivatives(u, v - sign * h)
            d1 = np.array(dv)
            d2 = (np.array(dv) - np.array(dv2)) / (sign * h)
            d = sign * d1
            dd = d2
            UV.append((u, v))
        P.append(R @ np.array(pt) + T)
        D.append(R @ d)
        DD.append(R @ dd)
    es = EdgeSample(np.array(P), np.array(D), np.array(DD))

    def ddd():
        # the exact third derivative across (odd: outward is sign times it) --
        # its part along the normal: the rest is a reparameterisation, and
        # kept, it set the fit's tolerance (a share of the row) 30x too loose
        from . import surfeval
        uv = np.array(UV, float)
        D3 = {kl: x @ R.T for kl, x in surfeval.Surf(surface).at(uv[:, 0], uv[:, 1], 3).items()}
        n, _ = surfeval.normals(D3)
        d3 = sign * (D3[3, 0] if along_v else D3[0, 3])
        return np.sum(d3 * n, axis=1, keepdims=True) * n
    es._ddd_fn = ddd
    return es


def _corner_knots(knots, deg, a, b):
    """Knots inside (a, b) with multiplicity >= deg: where the edge may have a corner."""
    eps = 1e-9 * max(abs(b - a), 1e-12)
    out, i = [], 0
    while i < len(knots):
        j = i
        while j < len(knots) and abs(knots[j] - knots[i]) <= eps:
            j += 1
        if j - i >= deg and a + eps < knots[i] < b - eps:
            out.append(float(knots[i]))
        i = j
    return out


def boundary_points(surface, which, n=48, corners=False):
    """n evenly spaced points along an iso edge (U0 U1 V0 V1). corners=True
    adds the points at the edge's corners (C0 knots, every knot of a degree-1
    direction), so a drawn edge of an extruded rectangle doesn't cut them;
    the result then has more than n points."""
    (u0, u1), (v0, v1) = surface.u_domain(), surface.v_domain()
    along_v = which in ("U0", "U1")
    a, b = (v0, v1) if along_v else (u0, u1)
    ts = [a + (b - a) * i / (n - 1) for i in range(n)]
    if corners:
        try:
            _, ext = _k()
            d = ext.surface_cvs(surface)
            k = d["knots_v"] if along_v else d["knots_u"]
            deg = d["degree_v"] if along_v else d["degree_u"]
            tol = 1e-7 * max(abs(b - a), 1e-12)
            ts = sorted(ts + [t for t in _corner_knots(k, int(deg), a, b) if min(abs(t - x) for x in ts) > tol])
        except Exception:
            pass
    if along_v:
        uu = u0 if which == "U0" else u1
        return [surface.evaluate(uu, t) for t in ts]
    vv = v0 if which == "V0" else v1
    return [surface.evaluate(t, vv) for t in ts]


# ---------------------------------------------------------------------------
# Blend loft between two inputs with G0 / G1 / G2 / G3 at each side
# ---------------------------------------------------------------------------

CONT_ORDER = {"G0": 0, "G1": 1, "G2": 2, "G3": 3}


def _side_rows(es, cont, tangent, curvature, n_deg, scale):
    """Rows of CVs at one side of the blend (starting with the boundary)."""
    rows = [es.P]
    k = CONT_ORDER[cont] if es.has_surface else 0
    if k == 0:
        return rows
    Dn = np.linalg.norm(es.D, axis=1, keepdims=True)
    Dn[Dn == 0] = 1.0
    Th = es.D / Dn
    # S'(0) = tangent * scale * Th  ->  P1 = P0 + S'(0) / n
    lam = tangent * scale / Dn  # per-sample reparam factor
    P1 = es.P + (tangent * scale / n_deg) * Th
    rows.append(P1)
    if k >= 2:
        # S''(0) = lam^2 * C''  (+ free term along Th = curvature slider)
        S2 = (lam ** 2) * es.DD + (curvature * scale) * Th
        P2 = 2 * P1 - es.P + S2 / (n_deg * (n_deg - 1))
        rows.append(P2)
    if k >= 3:
        # the source's cross curve c reparameterised by f (f' = lam, f'' = c_):
        # S''' = f'^3 c''' + 3 f' f'' c'' + f''' c', with f''' = 0
        DDD = es.DDD
        if DDD is None:
            raise ValueError("G3 needs the source's third derivative (a surface with a CV net)")
        c_ = curvature * scale / Dn
        S3 = (lam ** 3) * DDD + 3.0 * lam * c_ * es.DD
        rows.append(S3 / (n_deg * (n_deg - 1) * (n_deg - 2)) + 3 * P2 - 3 * P1 + es.P)
    return rows


def _difference_rows(rows_a, rows_b, tol, size, normals=(None, None)):
    """Side rows [R0, R1, R2, R3] as their differences [R0, R1 - R0,
    R2 - 2 R1 + R0, R3 - 3 R2 + 3 R1 - R0] (b's reversed, as in the surface)
    and a tolerance for each: for an edge row tol, at most 1e-5 of that
    edge's length (the continuity check allows a gap of 1e-4 of it; tol, 1e-4
    of the whole loft, let a trimmed edge's row stray past that); a 1e-4
    angle for the tangent rows, 0.1 % for the curvature and the curvature
    rate rows."""
    def side(R):
        out = [R[0]]
        if len(R) > 1:
            out.append(R[1] - R[0])
        if len(R) > 2:
            out.append(R[2] - 2 * R[1] + R[0])
        if len(R) > 3:
            out.append(R[3] - 3 * R[2] + 3 * R[1] - R[0])
        return out
    def tols(F, N):
        if tol is None:
            return [None] * len(F)
        edge = float(np.sum(np.linalg.norm(np.diff(F[0], axis=0), axis=1)))
        t = [min(tol, max(1e-5 * edge, 1e-9 * size))]
        if len(F) > 1:
            t.append(max(1e-4 * float(np.median(np.linalg.norm(F[1], axis=1))), 1e-7 * size))
        if len(F) > 2:
            # 0.1 % of the curvature row -- of its part along the normal when the
            # normals are known: a part in the tangent plane (Align Edges' bend
            # towards the corners) does not change the curvature, and measured
            # with it the curvature was fitted 10x looser (4.5 % off there)
            for Fk in F[2:]:
                big = np.abs(np.sum(Fk * N, axis=1)) if N is not None and len(N) == len(Fk) else \
                    np.linalg.norm(Fk, axis=1)
                t.append(max(1e-3 * float(np.max(big)), 1e-7 * size))
        return t
    Fa, Fb = side(rows_a), side(rows_b)
    ta, tb = tols(Fa, normals[0]), tols(Fb, normals[1])
    F = Fa + Fb[::-1]
    t = ta + tb[::-1]
    return F, (None if tol is None else t)


def _undifference_rows(G, na, nb):
    """CV rows back from fitted difference rows (inverse of _difference_rows)."""
    def side(H):
        out = [H[0]]
        if len(H) > 1:
            out.append(H[1] + H[0])
        if len(H) > 2:
            out.append(H[2] + 2 * out[1] - out[0])
        if len(H) > 3:
            out.append(H[3] + 3 * out[2] - 3 * out[1] + out[0])
        return out
    return side(G[:na]) + side(G[na:][::-1])[::-1]


def blend_loft_pieces(a, b, breaks, cont_a="G0", cont_b="G0", tan_a=1.0, tan_b=1.0, cur_a=0.0, cur_b=0.0,
                      degree_v=3):
    """blend_loft fitted piece by piece between the sample indices `breaks`
    (where the edges of a chained side meet) and joined there into one
    surface: one fit across a joint, where the trim curves of two faces meet
    with a slight kink, needed ten times the CVs and put knots so close the
    CVs flew off (a folded loft). Every piece uses the whole loft's scale, so
    the rows meet at the joints (the fit pins its end samples)."""
    _, ext = _k()
    scale = float(np.mean(np.linalg.norm(a.P - b.P, axis=1))) or 1.0
    idx = [0] + list(breaks) + [len(a.P) - 1]
    items, widths, nu, deg_u, knots_u = [], [], None, None, None
    for i0, i1 in zip(idx[:-1], idx[1:]):
        g = blend_loft(a.piece(i0, i1), b.piece(i0, i1), cont_a, cont_b, tan_a, tan_b, cur_a, cur_b,
                       degree_v=degree_v, scale=scale)
        d = ext.surface_cvs(g)
        nu, nv = d["count_u"], d["count_v"]
        deg_u, knots_u = d["degree_u"], d["knots_u"]
        G = np.array(d["points"], float).reshape(nu, nv, 3)
        items.append((d["degree_v"], d["knots_v"], G.transpose(1, 0, 2).reshape(nv, 3 * nu)))
        seg = lambda P: float(np.sum(np.linalg.norm(np.diff(P[i0:i1 + 1], axis=0), axis=1)))
        widths.append(0.5 * (seg(a.P) + seg(b.P)) or 1.0)
    q, U, A = concat_arrays(items, widths)
    G = A.reshape(len(A), nu, 3).transpose(1, 0, 2)
    pts = [tuple(map(float, x)) for x in G.reshape(-1, 3)]
    return ext.surface_from_cvs(int(deg_u), int(q), pts, [float(k) for k in knots_u], [float(k) for k in U])


SPAN_MAX = 32          # Bezier spans: at most this many along the side


def _pinned_fit(params, B, U, p, n_cvs, pins):
    """Least squares on a given knot vector, with the CVs `pins` {cv index:
    sample index} held on their samples (the ends, the side's joints)."""
    A = interp_matrix(params, U, p, n_cvs)
    X = np.zeros((n_cvs, B.shape[1]))
    fixed = sorted(pins)
    for c in fixed:
        X[c] = B[pins[c]]
    free = [c for c in range(n_cvs) if c not in pins]
    if free:
        R = B - A[:, fixed] @ X[fixed]
        X[free] = np.linalg.lstsq(A[:, free], R, rcond=None)[0]
    return X, A


def _knot_vector(p, inner):
    """Clamped knot vector on 0..1 from {value: multiplicity}."""
    U = [0.0] * (p + 1)
    for t in sorted(inner):
        U += [t] * inner[t]
    return U + [1.0] * (p + 1)


def blend_loft_spans(a, b, breaks, cont_a="G0", cont_b="G0", tan_a=1.0, tan_b=1.0, cur_a=0.0, cur_b=0.0,
                     degree=5, tol_angle=0.1, tol_curv=0.05, tol_gap_rel=1e-4, scale=None, tol_rate=0.05):
    """The G0 / G1 / G2 / G3 blend between two sides' samples as a few Bezier
    spans of one degree along the side, not a dense least-squares net.
    Knots go where they are needed, not where the samples are: full
    multiplicity at the side's joints `breaks` (sample indices -- the faces
    there meet as they meet), then a G2 (C2) knot in the middle of every span
    that does not hold, until each holds half the labels' tolerances (gap,
    angle, curvature, curvature rate, as contlabels measures them). Symmetric sides give
    symmetric knots. Returns (surface, number of spans, worst error over its
    tolerance)."""
    _, ext = _k()
    ka = CONT_ORDER[cont_a] if a.has_surface else 0
    kb = CONT_ORDER[cont_b] if b.has_surface else 0
    n_deg = 1 + ka + kb
    if scale is None:
        scale = float(np.mean(np.linalg.norm(a.P - b.P, axis=1))) or 1.0
    rows_a = _side_rows(a, cont_a, tan_a, cur_a, n_deg, scale)
    rows_b = _side_rows(b, cont_b, tan_b, cur_b, n_deg, scale)
    rows = rows_a + rows_b[::-1]
    allp = np.vstack(rows)
    size = float(np.linalg.norm(allp.max(axis=0) - allp.min(axis=0))) or 1.0
    F, _ = _difference_rows(rows_a, rows_b, None, size)
    q = max(int(degree), 3)
    N = len(rows[0])

    # what each fitted row is, and how far it may stray (half the labels' tolerances)
    def kinds(n_rows, normals):
        return [("edge", None), ("tan", None), ("curv", normals), ("rate", normals)][:n_rows]
    kk = kinds(len(rows_a), getattr(a, "N", None)) + kinds(len(rows_b), getattr(b, "N", None))[::-1]
    tols = []
    for Fi, (kind, nrm) in zip(F, kk):
        if kind == "edge":
            L = float(np.sum(np.linalg.norm(np.diff(Fi, axis=0), axis=1)))
            tols.append(np.full(N, max(0.5 * tol_gap_rel * L, 1e-9 * size)))
        elif kind == "tan":
            tols.append(np.maximum(np.tan(np.radians(0.5 * tol_angle)) * np.linalg.norm(Fi, axis=1), 1e-9 * size))
        else:
            big = np.abs(np.sum(Fi * nrm, axis=1)) if nrm is not None and len(nrm) == N else np.linalg.norm(Fi, axis=1)
            tk = tol_curv if kind == "curv" else tol_rate
            tols.append(np.full(N, max(0.5 * tk * float(np.max(big)), 1e-9 * size)))
    params = row_params(rows)
    B = np.hstack(F)
    joints = sorted({int(j) for j in breaks if 0 < j < N - 1})
    # G2 sides that know their normals: the edge row must also bend like the
    # source along the edge -- a loft leaving at a lean measures its curvature
    # straight across partly along the edge (lean^2), and a span within 5 um
    # of a trim curve can still bend 10 % differently
    bend_checks = []
    for es, rows_s, kidx in ((a, rows_a, 0), (b, rows_b, len(F) - 1)):
        Ns = getattr(es, "N", None)
        if Ns is None or len(rows_s) < 3 or len(Ns) != N:
            continue
        Ns = Ns if kidx == 0 else Ns          # rows_b's samples run as a's (reversed with them)
        P0 = rows_s[0]
        Pt = np.gradient(P0, params, axis=0, edge_order=2)
        Ptt = np.gradient(Pt, params, axis=0, edge_order=2)
        k_src = np.sum(Ptt * Ns, axis=1) / np.maximum(np.sum(Pt * Pt, axis=1), 1e-30)
        k_ref = max(float(np.max(np.abs(np.sum(F[kidx + (2 if kidx == 0 else -2)] * Ns, axis=1)))) /
                    max((scale / n_deg) ** 2 / (n_deg - 1) * n_deg, 1e-30), float(np.max(np.abs(k_src))), 1e-9)
        # and the tangent row must turn along the edge as the source's normal
        # does (the mixed curvature: how the loft's normal turns along the edge,
        # times the lean, is part of the curvature straight across)
        tidx = kidx + 1 if kidx == 0 else kidx - 1
        F1 = F[tidx]
        m_src = np.sum(np.gradient(F1, params, axis=0, edge_order=2) * Ns, axis=1) / \
            np.maximum(np.linalg.norm(F1, axis=1) * np.linalg.norm(Pt, axis=1), 1e-30)
        bend_checks.append((kidx, Ns, k_src, 0.25 * tol_curv * k_ref, tidx, m_src))
    h = 1e-4
    # the lightest first: cubic spans with knots only at the joints (a straight
    # edge stays 4 CVs); the span degree, splitting as needed, when they do not hold
    for q, may_split in (((3, False), (q, True)) if q > 3 else ((q, True),)):
        inner = {float(params[j]): q for j in joints}
        res = _span_fit(q, inner, may_split, params, B, F, N, tols, joints, bend_checks)
        if res[3] <= 1.0 or may_split:
            break
    U, X, inner, worst = res
    G = _undifference_rows([X[:, 3 * i:3 * i + 3] for i in range(len(F))], len(rows_a), len(rows_b))
    pts = [tuple(map(float, x)) for R in G for x in R]
    knots_u = [0.0] * (n_deg + 1) + [1.0] * (n_deg + 1)
    srf = ext.surface_from_cvs(n_deg, q, pts, knots_u, [float(k) for k in U])
    return srf, len(sorted({0.0, 1.0, *inner})) - 1, worst


def _span_fit(q, inner, may_split, params, B, F, N, tols, joints, bend_checks):
    """The span loop of blend_loft_spans: fit on the knots `inner`, then split
    every span out of tolerance (when may_split), until all hold. Returns
    (knots, CV array, inner knots, worst error over its tolerance)."""
    h = 1e-4
    worst = None
    for _ in range(4 * SPAN_MAX):
        U = _knot_vector(q, inner)
        n_cvs = len(U) - q - 1
        A = interp_matrix(params, U, q, n_cvs)
        pins = {0: 0, n_cvs - 1: N - 1}
        for j in joints:                         # the CV the joint's sample sits on
            c = int(np.argmax(A[j]))
            if A[j, c] > 1.0 - 1e-9:
                pins[c] = j
        X, A = _pinned_fit(params, B, U, q, n_cvs, pins)
        err = (A @ X - B).reshape(N, len(F), 3)
        ratio = np.max(np.stack([np.linalg.norm(err[:, i], axis=1) / tols[i] for i in range(len(F))]), axis=0)
        if bend_checks:
            tp, tm = np.clip(params + h, 0.0, 1.0), np.clip(params - h, 0.0, 1.0)
            Ap, Am = interp_matrix(tp, U, q, n_cvs), interp_matrix(tm, U, q, n_cvs)
            for kidx, Ns, k_src, ktol, tidx, m_src in bend_checks:
                Xe = X[:, 3 * kidx:3 * kidx + 3]
                Pp, P0f, Pm = Ap @ Xe, A @ Xe, Am @ Xe
                dt = (tp - tm)[:, None]
                Pt = (Pp - Pm) / np.where(dt > 0, dt, 1.0)
                Ptt = (Pp - 2 * P0f + Pm) / (h * h)
                inner_ = (tp - params > 0.5 * h) & (params - tm > 0.5 * h)
                k_fit = np.sum(Ptt * Ns, axis=1) / np.maximum(np.sum(Pt * Pt, axis=1), 1e-30)
                Xt = X[:, 3 * tidx:3 * tidx + 3]
                F1f = A @ Xt
                dF1 = (Ap @ Xt - Am @ Xt) / np.where(dt > 0, dt, 1.0)
                m_fit = np.sum(dF1 * Ns, axis=1) / np.maximum(np.linalg.norm(F1f, axis=1) * np.linalg.norm(Pt, axis=1), 1e-30)
                r_ = np.where(inner_, np.maximum(np.abs(k_fit - k_src), np.abs(m_fit - m_src)) / ktol, 0.0)
                for j in joints:                   # a joint's two sides bend as their faces do
                    r_[max(j - 1, 0):j + 2] = 0.0
                ratio = np.maximum(ratio, r_)
        worst = float(ratio.max())
        if worst <= 1.0 or not may_split:
            break
        # split every span that does not hold, in its middle (C2) -- if it
        # still has samples enough for its new CVs
        edges_ = sorted({0.0, 1.0, *inner})
        spans_ = list(zip(edges_[:-1], edges_[1:]))
        r_span = {sp: float(ratio[(params >= sp[0]) & (params <= sp[1])].max()) for sp in spans_}
        todo = {sp for sp in spans_ if r_span[sp] > 1.0}
        for lo, hi in list(todo):                 # its mirror too, when close (symmetric sides stay symmetric)
            mir = min(spans_, key=lambda sp: abs(sp[0] - (1 - hi)) + abs(sp[1] - (1 - lo)))
            if abs(mir[0] - (1 - hi)) + abs(mir[1] - (1 - lo)) < 1e-6 and r_span[mir] > 0.5:
                todo.add(mir)
        split = False
        for lo, hi in sorted(todo):
            if np.sum((params >= lo) & (params <= hi)) < 2 * (q + 1):
                continue
            if len(edges_) - 1 + split >= SPAN_MAX:
                break
            inner[0.5 * (lo + hi)] = max(q - 2, 1)
            split = True
        if not split:
            break
    return U, X, inner, worst


def split_uv_spans(surf):
    """An untrimmed surface cut at all its interior knots, both ways: one
    Bezier patch per span (exact)."""
    _, ext = _k()
    out = []
    for piece in split_v_spans(surf):
        d = ext.surface_cvs(piece)
        nu, nv = d["count_u"], d["count_v"]
        G = np.array(d["points"], float).reshape(nu, nv, 3).transpose(1, 0, 2)
        tr = ext.surface_from_cvs(int(d["degree_v"]), int(d["degree_u"]), [tuple(map(float, x)) for x in G.reshape(-1, 3)],
                                  [float(k) for k in d["knots_v"]], [float(k) for k in d["knots_u"]])
        for sub in split_v_spans(tr):              # (u, v swapped): back the other way round
            e = ext.surface_cvs(sub)
            H = np.array(e["points"], float).reshape(e["count_u"], e["count_v"], 3).transpose(1, 0, 2)
            out.append(ext.surface_from_cvs(int(e["degree_v"]), int(e["degree_u"]),
                                            [tuple(map(float, x)) for x in H.reshape(-1, 3)],
                                            [float(k) for k in e["knots_v"]], [float(k) for k in e["knots_u"]]))
    return out


def split_v_spans(surf):
    """An untrimmed surface cut at its interior v knots into its spans, each
    a Bezier patch in v (knots raised to full multiplicity, exactly)."""
    from .kernel.classa import insert_knot
    _, ext = _k()
    d = ext.surface_cvs(surf)
    nu, nv, du, dv = d["count_u"], d["count_v"], int(d["degree_u"]), int(d["degree_v"])
    if d.get("weights") and max(abs(w - 1.0) for w in d["weights"]) > 1e-12:
        return [surf]
    G = np.array(d["points"], float).reshape(nu, nv, 3)
    A = G.transpose(1, 0, 2).reshape(nv, 3 * nu)
    U = [float(k) for k in d["knots_v"]]
    for t in sorted(set(U[dv + 1:-dv - 1])):
        while sum(1 for k in U if k == t) < dv:
            A, U = insert_knot(A, U, dv, t)
            A = np.asarray(A, float)
    br = sorted(set(U))
    out = []
    for j in range(len(br) - 1):
        blk = A[j * dv:j * dv + dv + 1].reshape(dv + 1, nu, 3).transpose(1, 0, 2)
        pts = [tuple(map(float, x)) for x in blk.reshape(-1, 3)]
        out.append(ext.surface_from_cvs(du, dv, pts, [float(k) for k in d["knots_u"]],
                                        [0.0] * (dv + 1) + [1.0] * (dv + 1)))
    return out


def blend_loft(a, b, cont_a="G0", cont_b="G0", tan_a=1.0, tan_b=1.0, cur_a=0.0, cur_b=0.0, fit_tol=1e-4,
               degree_v=3, scale=None):
    """a, b: EdgeSample. Returns a kernel B-spline surface; u runs a -> b.
    scale: the length the tangent / curvature rows are measured in (default:
    the mean distance between a and b)."""
    _, ext = _k()
    ka = CONT_ORDER[cont_a] if a.has_surface else 0
    kb = CONT_ORDER[cont_b] if b.has_surface else 0
    n_rows = 2 + ka + kb
    n_deg = n_rows - 1
    if scale is None:
        scale = float(np.mean(np.linalg.norm(a.P - b.P, axis=1))) or 1.0
    rows_a = _side_rows(a, cont_a, tan_a, cur_a, n_deg, scale)
    rows_b = _side_rows(b, cont_b, tan_b, cur_b, n_deg, scale)
    rows = rows_a + rows_b[::-1]
    allp = np.vstack(rows)
    size = float(np.linalg.norm(allp.max(axis=0) - allp.min(axis=0))) or 1.0
    tol = fit_tol * size if fit_tol else None
    # The G1 / G2 rows are fitted as differences from the edge row (the cross
    # tangent, the cross curvature term), each held to its own tolerance -- an
    # angle and a curvature, not a distance: fitted within tol in position, the
    # rows of a handle-to-bowl loft left its tangent 0.1-0.8 degrees off, the
    # result depending on Samples. The fit is linear in the rows, so the CVs
    # are the same as fitting the rows themselves.
    F, tols = _difference_rows(rows_a, rows_b, tol, size, (getattr(a, "N", None), getattr(b, "N", None)))
    params = row_params(rows)
    fit = None
    if tol and degree_v < 3:
        # the inputs' own (lower) degree first, as ONE span: a line stays a
        # line. More spans of it are not taken: a polyline through the samples
        # of a curved edge passes them "within tol" yet kinks at every one
        q = max(1, degree_v)
        fit = fit_rows(F, q, tol=tols, fallback=False, max_cvs=q + 1, params=params)
    if fit is None:
        fit = fit_rows(F, 3, tol=tols, params=params)
    G, knots_v, deg_v = fit
    cv_rows = _undifference_rows(G, len(rows_a), len(rows_b))
    pts = []
    for R in cv_rows:
        pts.extend(tuple(map(float, p)) for p in R)
    knots_u = [0.0] * (n_deg + 1) + [1.0] * (n_deg + 1)
    return ext.surface_from_cvs(n_deg, deg_v, pts, knots_u, [float(x) for x in knots_v])


def _mean_row_length(D, deg, knots, n=17):
    """Mean length of the vector curve with CVs D (degree, knots) along 0..1."""
    _, ext = _k()
    try:
        c = ext.curve_from_cvs(int(deg), [tuple(map(float, q)) for q in D], [float(k) for k in knots], None)
        t0, t1 = c.parameter_range()
        return float(np.mean([np.linalg.norm(np.array(c.evaluate(float(t)), float))
                              for t in np.linspace(t0, t1, n)]))
    except Exception:
        return 0.0


def blend_loft_exact(sa, sb, a, b, cont_a="G0", cont_b="G0", tan_a=1.0, tan_b=1.0, cur_a=0.0, cur_b=0.0):
    """Blend loft on the inputs' OWN layout along the edge: sa / sb are exact
    sides {"deg", "knots", "P", "D", "DD", "DDD"} (CV rows, D / DD / DDD =
    outward cross derivative rows or None) sharing degree and knots. Rows are built with a
    constant scale per side, so they stay linear in the CVs and the result has
    exactly the inputs' degree and knots along v (G1 / G2 / G3 exact: a constant
    reparameterisation of each neighbour). a / b are the EdgeSamples (for the
    distance and the mean cross-tangent length)."""
    _, ext = _k()
    ka = CONT_ORDER[cont_a] if sa["D"] is not None else 0
    kb = CONT_ORDER[cont_b] if sb["D"] is not None else 0
    n_deg = 1 + ka + kb
    scale = float(np.mean(np.linalg.norm(a.P - b.P, axis=1))) or 1.0

    def side(s, es, k, tangent, curvature):
        rows = [s["P"]]
        if k == 0:
            return rows
        # the mean length of the cross derivative the rows are built from (the exact
        # D rows, evaluated along the edge). The EdgeSample of a B-rep edge holds unit
        # directions: dividing by its mean (1) left a trimmed bowl's side rows 11x too
        # short, so the loft hugged the bowl and turned sharply away from it.
        dm = _mean_row_length(s["D"], s["deg"], s["knots"]) or float(np.mean(np.linalg.norm(es.D, axis=1))) or 1.0
        lam = tangent * scale / dm
        P1 = s["P"] + (lam / n_deg) * s["D"]
        rows.append(P1)
        if k >= 2:
            S2 = lam ** 2 * s["DD"] + (curvature * scale / dm) * s["D"]
            rows.append(2 * P1 - s["P"] + S2 / (n_deg * (n_deg - 1)))
        if k >= 3:
            # S''' = f'^3 DDD + 3 f' f'' DD (f''' = 0): a constant reparameterisation
            # plus the curvature slider's f'', exact G3 on the inputs' own layout
            if s.get("DDD") is None:
                raise ValueError("G3 needs the source's third derivative (a surface with a CV net)")
            S3 = lam ** 3 * s["DDD"] + 3.0 * lam * (curvature * scale / dm) * s["DD"]
            rows.append(S3 / (n_deg * (n_deg - 1) * (n_deg - 2)) + 3 * rows[2] - 3 * rows[1] + rows[0])
        return rows

    rows = side(sa, a, ka, tan_a, cur_a) + side(sb, b, kb, tan_b, cur_b)[::-1]
    pts = [tuple(map(float, p)) for R in rows for p in R]
    knots_u = [0.0] * (n_deg + 1) + [1.0] * (n_deg + 1)
    return ext.surface_from_cvs(n_deg, int(sa["deg"]), pts, knots_u, [float(x) for x in sa["knots"]])


def reverse_side(s):
    k = s["knots"]
    r = lambda A: None if A is None else A[::-1].copy()
    return {"deg": s["deg"], "knots": [1.0 - x for x in reversed(k)], "P": r(s["P"]), "D": r(s["D"]), "DD": r(s["DD"]),
            "DDD": r(s.get("DDD"))}


def should_flip(Pa, Pb):
    straight = np.linalg.norm(Pa[0] - Pb[0]) + np.linalg.norm(Pa[-1] - Pb[-1])
    crossed = np.linalg.norm(Pa[0] - Pb[-1]) + np.linalg.norm(Pa[-1] - Pb[0])
    return crossed < straight


def axis_vector(name, obj=None):
    v = {"X": Vector((1, 0, 0)), "Y": Vector((0, 1, 0)), "Z": Vector((0, 0, 1))}[name[-1]]
    if name.startswith("LOCAL") and obj is not None:
        v = (world_matrix(obj).to_3x3() @ v).normalized()
    return v


# ---------------------------------------------------------------------------
# Blend curve between two curve ends (G0 / G1 / G2 / G3 at each side)
# ---------------------------------------------------------------------------

def curve_end_frame(kcurve, end, third=False):
    """Point, outward 1st derivative and 2nd derivative at a curve end
    (end 0 = start, 1 = end). 'Outward' = continuing the curve past that end.
    third: also the outward 3rd derivative (exact, from the CVs)."""
    sm, _ = _k()
    t0, t1 = kcurve.parameter_range()
    d = sm.evaluate_curve(kcurve, t1 if end else t0)
    P = np.array(d["point"], float)
    D = np.array(d["tangent"], float) * (1.0 if end else -1.0)
    DD = np.array(d["curvature"], float)
    if not third:
        return P, D, DD
    return P, D, DD, curve_derivs(kcurve, t1 if end else t0, 3)[3] * (1.0 if end else -1.0)


def curve_derivs(kcurve, t, nd=3):
    """[C, C', ..., C^(nd)] at t, exact from the curve's CVs (rational: the
    quotient rule)."""
    from math import comb
    from . import surfeval
    _, ext = _k()
    d = ext.curve_cvs(kcurve)
    p, U = int(d["degree"]), np.asarray(d["knots"], float)
    P = np.asarray(d["points"], float)
    w = np.asarray(d["weights"], float) if d.get("weights") else np.ones(len(P))
    span, N = surfeval.basis_ders(U, p, len(P), np.array([float(t)]), nd)
    Pw = np.c_[P * w[:, None], w][int(span[0]) - p:int(span[0]) + 1]
    A = [N[0, k] @ Pw for k in range(nd + 1)]
    C = []
    for k in range(nd + 1):
        v = A[k][:3] - sum(comb(k, i) * A[i][3] * C[k - i] for i in range(1, k + 1))
        C.append(v / A[0][3])
    return C


def blend_side_cvs(P, D, DD, D3, k, n, L, w=1.0, c=0.0, d=0.0):
    """The CVs one blend side fixes, P0 .. Pk (k = 0 for G0 .. 3 for G3).

    Same construction as the G1/G2/G3 blend loft: P1 = P0 + w*L/n * T (G1),
    P2 = 2 P1 - P0 + (lam^2 C'' + c*L*T) / (n(n-1)) (G2) with lam = w*L/|C'|,
    P3 = 3 P2 - 3 P1 + P0 + (lam^3 C''' + 3 lam mu C'' + d*L*T) / (n(n-1)(n-2))
    (G3), mu = c L / |C'|: the input curve reparameterised with f' = lam,
    f'' = mu, f''' = d L / |C'|. Every w > 0, c, d gives exact Gk and nothing
    else does: these three numbers are all a side's CVs can change."""
    P, D, DD = (np.asarray(v, float) for v in (P, D, DD))
    rows = [P]
    if k == 0:
        return rows
    dn = float(np.linalg.norm(D)) or 1.0
    T = D / dn
    P1 = P + (w * L / n) * T
    rows.append(P1)
    lam = w * L / dn
    if k >= 2:
        P2 = 2 * P1 - P + (lam * lam * DD + c * L * T) / (n * (n - 1))
        rows.append(P2)
    if k >= 3:
        if D3 is None:
            raise ValueError("G3 needs the curve's third derivative")
        S3 = lam ** 3 * np.asarray(D3, float) + 3.0 * lam * (c * L / dn) * DD + d * L * T
        rows.append(3 * P2 - 3 * P1 + P + S3 / (n * (n - 1) * (n - 2)))
    return rows


def blend_curve_cvs(frame_a, frame_b, cont_a="G1", cont_b="G1", w_a=1.0, w_b=1.0, c_a=0.0, c_b=0.0,
                    d_a=0.0, d_b=0.0):
    """CVs of a single Bezier-span blend from end A to end B: side A's CVs,
    then side B's reversed (blend_side_cvs), n = degree = 1 + kA + kB, L =
    distance between the ends. A frame is (P, C', C'') or, for G3,
    (P, C', C'', C''')."""
    ka, kb = CONT_ORDER[cont_a], CONT_ORDER[cont_b]
    n = 1 + ka + kb
    L = float(np.linalg.norm(np.asarray(frame_b[0], float) - np.asarray(frame_a[0], float))) or 1.0
    a = side_cvs(frame_a, ka, n, L, w_a, c_a, d_a)
    b = side_cvs(frame_b, kb, n, L, w_b, c_b, d_b)
    return n, [tuple(map(float, p)) for p in a + b[::-1]]


class SymFrame(tuple):
    '''A blend side on a symmetry plane: (P, the normal, 0[, 0]) and sym_u,
    the direction in the plane the blend bends toward (Curvature).'''
    sym_u = None


def sym_frame(P, D, u, third=False):
    z = np.zeros(3)
    f = SymFrame((np.asarray(P, float), np.asarray(D, float), z, z) if third else
                 (np.asarray(P, float), np.asarray(D, float), z))
    f.sym_u = np.asarray(u, float)
    return f


def flip_frame(f):
    '''The frame turned round (odd derivatives change sign); a symmetry side stays one.'''
    out = (f[0], -np.asarray(f[1], float), f[2]) + tuple(-np.asarray(x, float) for x in f[3:])
    if getattr(f, "sym_u", None) is not None:
        g = SymFrame(out)
        g.sym_u = f.sym_u
        return g
    return out


def blend_sym_side_cvs(P, D, u, k, n, L, w=1.0, c=0.0, d=0.0):
    '''A blend side on a symmetry plane, P0 .. Pk: P1 = P0 + w L/n T (T the
    normal), P2 at twice P1's height (n.C2 = 0: the blend and its mirror
    image join C2) moved by c L/(n(n-1)) along u in the plane (the bend at
    the plane), P3 = 3 P2 - 3 P1 + P0 + d L/(n(n-1)(n-2)) T (C3 along the
    normal only: C3 with the mirror image).'''
    P, D, u = (np.asarray(v, float) for v in (P, D, u))
    rows = [P]
    if k == 0:
        return rows
    T = D / (float(np.linalg.norm(D)) or 1.0)
    P1 = P + (w * L / n) * T
    rows.append(P1)
    if k >= 2:
        P2 = 2 * P1 - P + (c * L / (n * (n - 1))) * u
        rows.append(P2)
    if k >= 3:
        rows.append(3 * P2 - 3 * P1 + P + (d * L / (n * (n - 1) * (n - 2))) * T)
    return rows


def side_cvs(frame, k, n, L, w=1.0, c=0.0, d=0.0):
    '''One blend side's CVs from its frame (blend_side_cvs, or a symmetry plane's).'''
    if getattr(frame, "sym_u", None) is not None:
        return blend_sym_side_cvs(frame[0], frame[1], frame.sym_u, k, n, L, w, c, d)
    return blend_side_cvs(*frame[:3], frame[3] if len(frame) > 3 else None, k, n, L, w, c, d)


def write_curve_object(obj, degree, points_world):
    """Replace a Blender curve object's splines with one endpoint NURBS spline
    (a single Bezier span: order = degree + 1, clamped). A spline of that
    shape already there only gets its CVs moved: in Edit Mode (a blend curve's
    CVs being edited, blendedit.py) it keeps its point selection."""
    cu = obj.data
    cu.dimensions = '3D'
    Mi = world_matrix(obj).inverted()
    if len(cu.splines) == 1:
        sp = cu.splines[0]
        if sp.type == 'NURBS' and len(sp.points) == len(points_world) and sp.order_u == degree + 1 \
                and sp.use_endpoint_u and not sp.use_bezier_u and not sp.use_cyclic_u:
            co = np.array([(*(Mi @ Vector(w)), 1.0) for w in points_world], float).ravel()
            sp.points.foreach_set("co", co)
            cu.update_tag()
            return
    cu.splines.clear()
    sp = cu.splines.new('NURBS')
    sp.points.add(len(points_world) - 1)
    for p, w in zip(sp.points, points_world):
        p.co = (*(Mi @ Vector(w)), 1.0)
    sp.order_u = degree + 1
    sp.use_endpoint_u = True
    sp.use_bezier_u = False
    sp.resolution_u = 24
    cu.update_tag() if hasattr(cu, "update_tag") else None


def cutter_direction(kcurve, fallback):
    """Plane normal of a planar cutting curve, else the fallback (view) direction."""
    P = sample_curve(kcurve, 48)
    c = P.mean(axis=0)
    size = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0
    _, sv, vt = np.linalg.svd(P - c)
    flat = sv[2] / max(len(P) ** 0.5, 1.0) <= 1e-4 * size
    if flat and sv[1] > 1e-6 * size:
        return Vector(vt[2]).normalized()
    d = Vector(fallback)
    return d.normalized() if d.length > 1e-9 else Vector((0, 0, 1))


# ---------------------------------------------------------------------------
# Extrude: tilted direction, draft (Alias Multi-Draft)
# ---------------------------------------------------------------------------

def _perp_frame(d):
    d = np.asarray(d, float)
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = a - (a @ d) * d
    a /= np.linalg.norm(a)
    return a, np.cross(d, a)


def tilt_direction(d, tilt_deg, azimuth_deg):
    """Direction d tilted by `tilt` degrees away from itself, towards the
    perpendicular at `azimuth` degrees (0 = towards world X, or Y if d ~ X)."""
    if abs(tilt_deg) < 1e-12:
        return Vector(d)
    dn = np.asarray(d, float) / np.linalg.norm(d)
    a, b = _perp_frame(dn)
    az, t = np.radians(azimuth_deg), np.radians(tilt_deg)
    side = np.cos(az) * a + np.sin(az) * b
    return Vector(np.cos(t) * dn + np.sin(t) * side)


def draft_reference(P, closed, centre=None):
    """The point a draft opens away from: a closed profile's own centre; an
    open one's the given centre (the middle of everything extruded together,
    so open pieces of one outline all lean out of it), else its own."""
    if closed or centre is None:
        return np.asarray(P, float).mean(axis=0)
    return np.asarray(centre, float)


def _draft_normals(P, d, closed, flip, centre=None):
    """Unit in-plane 'outward' directions (perpendicular to the curve and to d)
    at curve samples P. Sharp corners get a mitred normal (length 1/cos(half
    angle)), so offset polylines keep straight, parallel sides."""
    d = np.asarray(d, float)
    n = len(P)
    seg = np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0)    # closed: include the closing segment
    seg_n = np.cross(seg, d)
    seg_n /= np.linalg.norm(seg_n, axis=1, keepdims=True) + 1e-300
    out = np.zeros((n, 3))
    for i in range(n):
        prev = seg_n[i - 1] if (i > 0 or closed) else None
        nxt = seg_n[i] if (i < len(seg_n) and (i < n - 1 or closed)) else None
        if prev is None:
            out[i] = nxt
        elif nxt is None:
            out[i] = prev
        else:
            m = prev + nxt
            c = np.linalg.norm(m) / 2.0                      # cos(half angle between the sides)
            out[i] = (m / (np.linalg.norm(m) or 1.0)) / max(c, 0.05)
    # outward = away from the curve's centre (closed) or the given one; flip on request
    c = draft_reference(P, closed, centre)
    rel = P - c
    rel -= np.outer(rel @ d, d)
    sgn = 1.0 if np.sum(out * rel) >= 0 else -1.0
    if flip:
        sgn = -sgn
    return sgn * out


def _exact_draft(cv, d, h, off, flip, symmetric, centre=None):
    """Draft rows on the profile's own basis: the base row IS the profile (its
    CVs, knots and weights unchanged), the far row(s) are fitted on the same
    knots and weights -- exact for lines, circles and arcs (a cone), with knots
    added to every row (exactly) only where an offset needs them. At a sharp
    corner (a knot of full multiplicity where the tangents disagree) the
    smooth pieces are drafted one by one and the far rows mitred: each
    piece's end is moved along its tangent to where the two pieces' far edges
    meet (exact for straight pieces). Returns (rows, knots, degree, weights),
    or None when the profile is not clamped or cannot be drafted this way."""
    from .kernel.classa import basis, insert_knot
    p = int(cv["degree"])
    U = [float(k) for k in cv["knots"]]
    P = np.array(cv["points"], float)
    w = np.array(cv["weights"], float) if cv.get("weights") else np.ones(len(P))
    if p < 2 or len(U) != len(P) + p + 1 or len(w) != len(P) or (w <= 0).any():
        return None
    if any(abs(U[i] - U[0]) > 1e-12 or abs(U[-1 - i] - U[-1]) > 1e-12 for i in range(p + 1)):
        return None
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    closed = np.linalg.norm(P[0] - P[-1]) < 1e-9 * size
    tol = 1e-5 * size
    Pw = np.c_[P * w[:, None], w]
    # the offset's parameter speed kinks wherever the profile's curvature
    # jumps (at its knots): one more multiplicity there lets the far row
    # follow (inserting a knot leaves the profile itself unchanged)
    for k in sorted(set(U[p + 1:-p - 1])):
        if U.count(k) < p:
            Pw, U = insert_knot(Pw, U, p, k)
            Pw = np.asarray(Pw, float)
    P = Pw[:, :3] / Pw[:, 3:]

    def frame(Pw_, U_, ts):
        H, H1 = basis(U_, p, ts) @ Pw_, basis(U_, p, ts, 1) @ Pw_
        C = H[:, :3] / H[:, 3:]
        return C, (H1[:, :3] - C * H1[:, 3:]) / H[:, 3:]

    def unit_n(T):
        nrm = np.cross(T, d)
        nn = np.linalg.norm(nrm, axis=1)
        if (nn < 1e-9 * (np.linalg.norm(T, axis=1) + 1e-300)).any():
            return None                       # the profile runs along the direction
        return nrm / nn[:, None]

    def bends(T):
        return T[0] @ T[1] < (1.0 - 1e-9) * np.linalg.norm(T[0]) * np.linalg.norm(T[1])

    br = sorted(set(U))
    e = 1e-7 * (br[-1] - br[0])
    corners = []
    for k in br[1:-1]:
        if bends(frame(Pw, U, [k - e, k + e])[1]):
            if U.count(k) != p:
                return None
            corners.append(k)
    seam_corner = closed and bends(np.vstack([frame(Pw, U, [U[-1] - e])[1], frame(Pw, U, [U[0] + e])[1]]))
    # outward = away from the curve's centre, as _draft_normals (whole curve)
    ts = np.concatenate([np.linspace(a, b, 18)[1:-1] for a, b in zip(br[:-1], br[1:])])
    C, T = frame(Pw, U, ts)
    nrm = unit_n(T)
    if nrm is None:
        return None
    rel = C - draft_reference(C, closed, centre)
    rel -= np.outer(rel @ d, d)
    sgn = 1.0 if np.sum(nrm * rel) >= 0 else -1.0
    if flip:
        sgn = -sgn
    heights = [h] + ([-h] if symmetric else [])          # far rows: top (and bottom)

    def unit(v):
        return v / (np.linalg.norm(v) or 1.0)

    def tangent_at(Q, i0, i1, i2):
        """The profile's unit tangent at its CV i1 where the curve passes
        through it (a knot of full multiplicity, or a clamped end): exactly
        the directions to and from its neighbours i0, i2 (positive weights),
        averaged (they agree: no corner there)."""
        return unit(unit(Q[i1] - Q[i0]) + unit(Q[i2] - Q[i1]))

    # a closed profile smooth through its seam: the far row's two seam ends
    # must leave along one tangent too
    seam_t = None
    if closed and not seam_corner:
        seam_t = unit(unit(P[-1] - P[-2]) + unit(P[1] - P[0]))

    def fit_piece(Pw_, U_, t_start=None, t_end=None):
        """The far rows of one smooth piece on its own basis (ends exact). At
        a knot of full multiplicity the basis is only C0, so a free fit would
        put a small tangent crease there (and in the wall): the far row's CV
        at such a knot is held on the target and its two neighbours on the
        profile's tangent line through it -- the row and the ruled wall are
        then G1 there, as the profile is. t_start / t_end: the same at the
        piece's ends (a smooth seam)."""
        n0 = len(Pw_)
        for _ in range(16):
            w_ = Pw_[:, 3]
            n_cv = len(Pw_)
            br_ = sorted(set(U_))
            fixed = {0: U_[0], n_cv - 1: U_[-1]}              # CV -> parameter (on the target)
            Q = Pw_[:, :3] / w_[:, None]
            anchors = []
            for k in br_[1:-1]:
                if U_.count(k) >= p:
                    j = U_.index(k) - 1
                    fixed[j] = k
                    anchors.append((j, tangent_at(Q, j - 1, j, j + 1)))
            if t_start is not None:
                anchors.append((0, t_start))
            if t_end is not None:
                anchors.append((n_cv - 1, t_end))
            ties = {}                                         # CV -> [(anchor CV, unit tangent)]
            for j, T in anchors:
                for i in (j - 1, j + 1):
                    if 0 <= i < n_cv and i not in fixed:
                        ties.setdefault(i, []).append((j, T))
            ts_ = np.concatenate([np.linspace(a, b, 16, endpoint=False) for a, b in zip(br_[:-1], br_[1:])]
                                 + [[br_[-1]]])
            at = {k: 16 * i for i, k in enumerate(br_)}       # the sample index of each knot
            at[br_[-1]] = len(ts_) - 1
            C_, T_ = frame(Pw_, U_, ts_)
            n_ = unit_n(T_)
            if n_ is None:
                return None
            B = basis(U_, p, ts_)
            R = B * w_[None, :] / (B @ w_)[:, None]           # the rational basis
            tgs = [C_ + hh * d + off * sgn * n_ for hh in heights]
            # one CV between two anchors (degree 2: every span between double
            # knots) lies on both tangent lines: where they meet (one line
            # when they are the same: a straight span); else split the span
            meet, split = {}, None
            for i, tl in ties.items():
                if len(tl) < 2:
                    continue
                (j1, T1), (j2, T2) = tl
                pts = []
                for tg in tgs:
                    A1, A2 = tg[at[fixed[j1]]], tg[at[fixed[j2]]]
                    c = T1 @ T2
                    if 1.0 - abs(c) < 1e-12:                  # parallel: the same line, or no meeting
                        if np.linalg.norm(np.cross(A2 - A1, T1)) > 1e-3 * tol:
                            break
                        continue
                    s1, s2 = np.linalg.solve(np.array([[1.0, -c], [c, -1.0]]), [(A2 - A1) @ T1, (A2 - A1) @ T2])
                    q1, q2 = A1 + s1 * T1, A2 + s2 * T2
                    if np.linalg.norm(q1 - q2) > 1e-3 * tol or s1 * (A2 - A1) @ T1 < 0:
                        break                                 # skew lines, or a meeting behind
                    pts.append(0.5 * (q1 + q2))
                else:
                    if pts:
                        meet[i] = pts
                    else:
                        ties[i] = tl[:1]
                    continue
                split = 0.5 * (fixed[j1] + fixed[j2])
                break
            if split is not None:
                Pw_, U_ = insert_knot(Pw_, U_, p, split)
                Pw_ = np.asarray(Pw_, float)
                continue
            # X = X0 + M z: fixed and met CVs in X0, a tied CV = its anchor + s * tangent
            cols = []
            for i in range(n_cv):
                if i in meet or i in fixed:
                    continue
                cols.append((i, [ties[i][0][1]] if i in ties else list(np.eye(3))))
            M = np.zeros((3 * n_cv, sum(len(c) for _, c in cols)))
            c0 = 0
            for i, vs in cols:
                for v in vs:
                    M[3 * i:3 * i + 3, c0] = v
                    c0 += 1
            RR = np.kron(R, np.eye(3))
            rows, worst = [], np.zeros(len(ts_))
            for r_, tg in enumerate(tgs):
                X = np.zeros((n_cv, 3))
                for i, k in fixed.items():
                    X[i] = tg[at[k]]
                for i, tl in ties.items():
                    X[i] = meet[i][r_] if i in meet else X[tl[0][0]]
                if M.shape[1]:
                    z = np.linalg.lstsq(RR @ M, (tg - R @ X).ravel(), rcond=None)[0]
                    X = X + (M @ z).reshape(n_cv, 3)
                rows.append(X)
                worst = np.maximum(worst, np.linalg.norm(R @ X - tg, axis=1))
            if worst.max() <= tol:
                return Pw_, U_, rows
            spans = sorted({max(i for i, k in enumerate(br_) if k <= t and i < len(br_) - 1)
                            for t, err in zip(ts_, worst) if err > tol})
            for i in spans:
                Pw_, U_ = insert_knot(Pw_, U_, p, 0.5 * (br_[i] + br_[i + 1]))
            Pw_ = np.asarray(Pw_, float)
            if len(Pw_) > 8 * n0 + 200:
                return None
        return None

    cuts = [U[0]] + corners + [U[-1]]
    pieces = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        ka = 0 if a == U[0] else U.index(a) - 1
        kb = len(P) - 1 if b == U[-1] else U.index(b) - 1
        res = fit_piece(Pw[ka:kb + 1], [a] * (p + 1) + [u for u in U if a < u < b] + [b] * (p + 1),
                        seam_t if a == U[0] else None, seam_t if b == U[-1] else None)
        if res is None:
            return None
        pieces.append(res)

    def mitre(Xa, Xb):
        """Move Xa's last and Xb's first CV to where their end tangents meet."""
        p0, ta, q0, tb = Xa[-1], Xa[-1] - Xa[-2], Xb[0], Xb[1] - Xb[0]
        M = np.array([[ta @ ta, -(ta @ tb)], [ta @ tb, -(tb @ tb)]])
        m = 0.5 * (p0 + q0)
        if abs(np.linalg.det(M)) > 1e-12 * (ta @ ta) * (tb @ tb):
            s_, u_ = np.linalg.solve(M, np.array([(q0 - p0) @ ta, (q0 - p0) @ tb]))
            mm = 0.5 * (p0 + s_ * ta + q0 + u_ * tb)
            if np.linalg.norm(mm - m) < 20.0 * abs(off) + tol:    # (a hairpin: no sane mitre)
                m = mm
        Xa[-1] = Xb[0] = m

    for i in range(len(pieces) - 1 + (1 if seam_corner else 0)):
        j = (i + 1) % len(pieces)
        for r in range(len(heights)):
            mitre(pieces[i][2][r], pieces[j][2][r])
    base, far, Ut = [], [[] for _ in heights], [U[0]] * (p + 1)
    for i, (Pw_, U_, rows) in enumerate(pieces):
        k0 = 0 if i == 0 else 1                               # a corner CV is shared
        base.append(Pw_[k0:])
        for r, X in enumerate(rows):
            far[r].append(X[k0:])
        Ut += [u for u in U_ if U_[0] < u < U_[-1]] + ([U_[-1]] * p if i < len(pieces) - 1 else [])
    Ut += [U[-1]] * (p + 1)
    Pw = np.vstack(base)
    w = Pw[:, 3]
    P = Pw[:, :3] / w[:, None]
    far = [np.vstack(f) for f in far]
    if closed:                                                # the seam: one point
        for f in far:
            f[-1] = f[0]
    rows = [P, far[0]] if not symmetric else [far[1], P, far[0]]
    return rows, Ut, p, w


def draft_extrude(kcurve, d, h, draft_deg, flip=False, symmetric=False, centre=None):
    """Ruled draft surface: height h along d, outward offset h*tan(draft).
    Degree-1 profiles (polylines) are exact with mitred corners. Smooth
    profiles keep their own CVs, knots and weights for the base row, and the
    far row is fitted on that basis (a drafted circle is an exact cone, CVs
    as even as the circle's; sharp corners are mitred). Only a profile that
    cannot be drafted so is sampled and both rows fitted on one
    parameterization (tolerance 1e-5 of the size)."""
    _, ext = _k()
    d = np.asarray(d, float) / np.linalg.norm(d)
    off = h * np.tan(np.radians(draft_deg))
    cv = ext.curve_cvs(kcurve)
    closed = False
    wts = None
    exact = None if cv["degree"] == 1 else _exact_draft(cv, d, h, off, flip, symmetric, centre)
    if cv["degree"] == 1 and not cv.get("weights"):
        P = np.array(cv["points"], float)
        closed = np.linalg.norm(P[0] - P[-1]) < 1e-9 * (np.ptp(P, axis=0).max() or 1.0)
        nrm = _draft_normals(P[:-1] if closed else P, d, closed, flip, centre)
        if closed:
            nrm = np.vstack([nrm, nrm[:1]])
        rows = _draft_rows(P, nrm, d, h, off, symmetric)
        knots_c = list(cv["knots"])
        deg_c = 1
        X = rows
    elif exact is not None:
        X, knots_c, deg_c, w = exact
        if np.abs(w - 1.0).max() > 1e-12:
            wts = [float(x) for _ in X for x in w]
    else:
        t0, t1 = kcurve.parameter_range()
        P = np.array([kcurve.evaluate(t0 + (t1 - t0) * i / 199) for i in range(200)], float)
        closed = np.linalg.norm(P[0] - P[-1]) < 1e-7 * (np.ptp(P, axis=0).max() or 1.0)
        if closed:                      # the last sample repeats the first
            nrm = _draft_normals(P[:-1], d, True, flip)
            nrm = np.vstack([nrm, nrm[:1]])
        else:
            nrm = _draft_normals(P, d, False, flip, centre)
        rows = _draft_rows(P, nrm, d, h, off, symmetric)
        size = float(np.linalg.norm(np.ptp(np.vstack(rows), axis=0))) or 1.0
        X, knots_c, deg_c = fit_rows(rows, 3, tol=1e-5 * size)
    nr = len(X)
    knots_e = [0.0, 0.0, 1.0, 1.0] if nr == 2 else [0.0, 0.0, 0.5, 1.0, 1.0]
    pts = []
    for r in X:
        pts.extend(tuple(map(float, p)) for p in r)
    # u = across the extrusion (degree 1), v = along the curve
    return ext.surface_from_cvs(1, deg_c, pts, knots_e, [float(k) for k in knots_c], wts)


def _draft_rows(P, nrm, d, h, off, symmetric):
    top = P + h * d + off * nrm
    if not symmetric:
        return [P, top]
    bottom = P - h * d + off * nrm
    return [bottom, P, top]
