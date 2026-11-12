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


_SIDE_ROWS = ("P", "D", "DD")


def _side_array(s):
    keys = [k for k in _SIDE_ROWS if s.get(k) is not None]
    return keys, np.hstack([np.asarray(s[k], float) for k in keys])


def _side_from(keys, deg, knots, A):
    out = {"deg": int(deg), "knots": [float(k) for k in knots], "P": None, "D": None, "DD": None}
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


def curve_plane_normal(curve, n=32):
    P = sample_curve(curve, n)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    return Vector(vt[2]).normalized(), Vector(c)


# ---------------------------------------------------------------------------
# Surface edge sampling for G0/G1/G2 lofts
# ---------------------------------------------------------------------------

class EdgeSample:
    """Samples along a loft input: positions P, outward cross tangent D,
    outward cross second derivative DD (None for plain curves)."""

    def __init__(self, P, D=None, DD=None):
        self.P = P
        self.D = D
        self.DD = DD

    @property
    def has_surface(self):
        return self.D is not None

    def reversed(self):
        r = lambda a: None if a is None else a[::-1].copy()
        return EdgeSample(self.P[::-1].copy(), r(self.D), r(self.DD))


def sample_surface_edge(surface, which, matrix, n):
    (u0, u1), (v0, v1) = surface.u_domain(), surface.v_domain()
    M = np.array(matrix)
    R = M[:3, :3]
    T = M[:3, 3]
    P, D, DD = [], [], []
    along_v = which in ("U0", "U1")
    span = (u1 - u0) if along_v else (v1 - v0)
    h = 1e-4 * span
    for i in range(n):
        s = i / (n - 1)
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
        P.append(R @ np.array(pt) + T)
        D.append(R @ d)
        DD.append(R @ dd)
    return EdgeSample(np.array(P), np.array(D), np.array(DD))


def boundary_points(surface, which, n=48):
    (u0, u1), (v0, v1) = surface.u_domain(), surface.v_domain()
    pts = []
    for i in range(n):
        s = i / (n - 1)
        if which in ("U0", "U1"):
            pts.append(surface.evaluate(u0 if which == "U0" else u1, v0 + (v1 - v0) * s))
        else:
            pts.append(surface.evaluate(u0 + (u1 - u0) * s, v0 if which == "V0" else v1))
    return pts


# ---------------------------------------------------------------------------
# Blend loft between two inputs with G0 / G1 / G2 at each side
# ---------------------------------------------------------------------------

CONT_ORDER = {"G0": 0, "G1": 1, "G2": 2}


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
    return rows


def _difference_rows(rows_a, rows_b, tol, size):
    """Side rows [R0, R1, R2] as [R0, R1 - R0, R2 - 2 R1 + R0] (b's reversed,
    as in the surface) and a tolerance for each: for an edge row tol, at most
    1e-5 of that edge's length (the continuity check allows a gap of 1e-4 of
    it; tol, 1e-4 of the whole loft, let a trimmed edge's row stray past
    that); a 1e-4 angle for the tangent rows, 0.1 % for the curvature rows."""
    def side(R):
        out = [R[0]]
        if len(R) > 1:
            out.append(R[1] - R[0])
        if len(R) > 2:
            out.append(R[2] - 2 * R[1] + R[0])
        return out
    def tols(F):
        if tol is None:
            return [None] * len(F)
        edge = float(np.sum(np.linalg.norm(np.diff(F[0], axis=0), axis=1)))
        t = [min(tol, max(1e-5 * edge, 1e-9 * size))]
        if len(F) > 1:
            t.append(max(1e-4 * float(np.median(np.linalg.norm(F[1], axis=1))), 1e-7 * size))
        if len(F) > 2:
            t.append(max(1e-3 * float(np.max(np.linalg.norm(F[2], axis=1))), 1e-7 * size))
        return t
    Fa, Fb = side(rows_a), side(rows_b)
    ta, tb = tols(Fa), tols(Fb)
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
        return out
    return side(G[:na]) + side(G[na:][::-1])[::-1]


def blend_loft(a, b, cont_a="G0", cont_b="G0", tan_a=1.0, tan_b=1.0, cur_a=0.0, cur_b=0.0, fit_tol=1e-4,
               degree_v=3):
    """a, b: EdgeSample. Returns a kernel B-spline surface; u runs a -> b."""
    _, ext = _k()
    ka = CONT_ORDER[cont_a] if a.has_surface else 0
    kb = CONT_ORDER[cont_b] if b.has_surface else 0
    n_rows = 2 + ka + kb
    n_deg = n_rows - 1
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
    F, tols = _difference_rows(rows_a, rows_b, tol, size)
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
    sides {"deg", "knots", "P", "D", "DD"} (CV rows, D / DD = outward cross
    derivative rows or None) sharing degree and knots. Rows are built with a
    constant scale per side, so they stay linear in the CVs and the result has
    exactly the inputs' degree and knots along v (G1 / G2 exact: a constant
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
        return rows

    rows = side(sa, a, ka, tan_a, cur_a) + side(sb, b, kb, tan_b, cur_b)[::-1]
    pts = [tuple(map(float, p)) for R in rows for p in R]
    knots_u = [0.0] * (n_deg + 1) + [1.0] * (n_deg + 1)
    return ext.surface_from_cvs(n_deg, int(sa["deg"]), pts, knots_u, [float(x) for x in sa["knots"]])


def reverse_side(s):
    k = s["knots"]
    r = lambda A: None if A is None else A[::-1].copy()
    return {"deg": s["deg"], "knots": [1.0 - x for x in reversed(k)], "P": r(s["P"]), "D": r(s["D"]), "DD": r(s["DD"])}


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
# Blend curve between two curve ends (G0 / G1 / G2 at each side)
# ---------------------------------------------------------------------------

def curve_end_frame(kcurve, end):
    """Point, outward 1st derivative and 2nd derivative at a curve end
    (end 0 = start, 1 = end). 'Outward' = continuing the curve past that end."""
    sm, _ = _k()
    t0, t1 = kcurve.parameter_range()
    d = sm.evaluate_curve(kcurve, t1 if end else t0)
    P = np.array(d["point"], float)
    D = np.array(d["tangent"], float) * (1.0 if end else -1.0)
    DD = np.array(d["curvature"], float)
    return P, D, DD


def blend_curve_cvs(frame_a, frame_b, cont_a="G1", cont_b="G1", w_a=1.0, w_b=1.0, c_a=0.0, c_b=0.0):
    """CVs of a single Bezier-span blend from end A to end B.

    Same construction as the G1/G2 blend loft: P1 = P0 + w*L/n * T (G1),
    P2 = 2 P1 - P0 + (lam^2 C'' + c*L*T) / (n(n-1)) (G2) with lam = w*L/|C'|,
    where L = distance between the ends and n = degree = 1 + kA + kB."""
    PA, DA, DDA = frame_a
    PB, DB, DDB = frame_b
    ka, kb = CONT_ORDER[cont_a], CONT_ORDER[cont_b]
    n = 1 + ka + kb
    L = float(np.linalg.norm(PB - PA)) or 1.0

    def side(P, D, DD, k, w, c):
        rows = [P]
        if k == 0:
            return rows
        dn = float(np.linalg.norm(D)) or 1.0
        T = D / dn
        P1 = P + (w * L / n) * T
        rows.append(P1)
        if k >= 2:
            lam = w * L / dn
            P2 = 2 * P1 - P + (lam * lam * DD + c * L * T) / (n * (n - 1))
            rows.append(P2)
        return rows

    a = side(PA, DA, DDA, ka, w_a, c_a)
    b = side(PB, DB, DDB, kb, w_b, c_b)
    return n, [tuple(map(float, p)) for p in a + b[::-1]]


def write_curve_object(obj, degree, points_world):
    """Replace a Blender curve object's splines with one endpoint NURBS spline
    (a single Bezier span: order = degree + 1, clamped)."""
    cu = obj.data
    cu.dimensions = '3D'
    cu.splines.clear()
    sp = cu.splines.new('NURBS')
    sp.points.add(len(points_world) - 1)
    Mi = world_matrix(obj).inverted()
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


def _draft_normals(P, d, closed, flip):
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
    # outward = away from the curve's centre (closed), else the side of +x; flip on request
    c = P.mean(axis=0)
    rel = P - c
    rel -= np.outer(rel @ d, d)
    sgn = 1.0 if np.sum(out * rel) >= 0 else -1.0
    if flip:
        sgn = -sgn
    return sgn * out


def _exact_draft(cv, d, h, off, flip, symmetric):
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
    rel = C - C.mean(axis=0)
    rel -= np.outer(rel @ d, d)
    sgn = 1.0 if np.sum(nrm * rel) >= 0 else -1.0
    if flip:
        sgn = -sgn
    heights = [h] + ([-h] if symmetric else [])          # far rows: top (and bottom)

    def fit_piece(Pw_, U_):
        """The far rows of one smooth piece on its own basis (ends exact)."""
        n0 = len(Pw_)
        for _ in range(16):
            w_ = Pw_[:, 3]
            br_ = sorted(set(U_))
            ts_ = np.concatenate([np.linspace(a, b, 16, endpoint=False) for a, b in zip(br_[:-1], br_[1:])]
                                 + [[br_[-1]]])
            C_, T_ = frame(Pw_, U_, ts_)
            n_ = unit_n(T_)
            if n_ is None:
                return None
            B = basis(U_, p, ts_)
            R = B * w_[None, :] / (B @ w_)[:, None]           # the rational basis
            rows, worst = [], np.zeros(len(ts_))
            for hh in heights:
                tg = C_ + hh * d + off * sgn * n_
                X = np.zeros((len(Pw_), 3))
                X[0], X[-1] = tg[0], tg[-1]
                if len(Pw_) > 2:
                    rhs = tg - np.outer(R[:, 0], X[0]) - np.outer(R[:, -1], X[-1])
                    X[1:-1] = np.linalg.lstsq(R[:, 1:-1], rhs, rcond=None)[0]
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
        res = fit_piece(Pw[ka:kb + 1], [a] * (p + 1) + [u for u in U if a < u < b] + [b] * (p + 1))
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


def draft_extrude(kcurve, d, h, draft_deg, flip=False, symmetric=False):
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
    exact = None if cv["degree"] == 1 else _exact_draft(cv, d, h, off, flip, symmetric)
    if cv["degree"] == 1 and not cv.get("weights"):
        P = np.array(cv["points"], float)
        closed = np.linalg.norm(P[0] - P[-1]) < 1e-9 * (np.ptp(P, axis=0).max() or 1.0)
        nrm = _draft_normals(P[:-1] if closed else P, d, closed, flip)
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
            nrm = _draft_normals(P, d, False, flip)
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
