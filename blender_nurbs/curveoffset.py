"""Offset Curve.

Offsets curves by a distance within their plane (the curve's own plane, or
the active work plane for straight or non-planar curves): every point moves
along n = N x T, N the plane normal, T the curve's direction.

* Lines and circular arcs offset EXACTLY (a line moves, an arc keeps its
  centre and weights and changes its radius). An arc that would shrink to
  nothing is dropped and its neighbours meet instead.
* Free-form spans are fitted as uniform splines within Tolerance, through
  their end points and with the base curve's end directions, so they stay G1
  to the exact pieces next to them. Where the distance is larger than the
  radius of curvature on that side, the offset would fold into a small loop:
  the loop is cut out and the offset gets a corner there.
* At corners (tangent breaks) the offset pieces either overlap (inside of the
  turn: both are trimmed where they cross) or leave a gap (outside), closed
  as Sharp (both pieces extended by tangent lines until they meet), Round (an
  exact arc about the corner) or Chamfer (a straight line).

Individual offsets each curve on its own (corners inside it are still
handled); Joined first chains the curves whose ends touch, and offsets each
chain as one curve. Positive distances go outward on closed curves and to the
left (seen from the plane's normal) on open ones; Flip turns that round.
"""

import numpy as np

from . import curvejoin, curvetools, curvetrim, features, geom

LAST_INFO = {}
CHAIN_TOL = 1e-3             # Joined: curve ends this close are chained
PER_SPAN = 48                # samples per free-form span (at least)
MAX_PER_SPAN = 1500


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-300 else v * 0.0


def _start(spans):
    return curvejoin._cart(spans[0])[0]


def _end(spans):
    return curvejoin._cart(spans[-1])[-1]


def _line(a, b):
    return np.array([[*a, 1.0], [*b, 1.0]])


def _samples(spans, per=24):
    return curvetrim.sample_spans(spans, per)[0]


def plane_normal(spans_list, hint):
    """(normal, planar): the plane the curves lie in, oriented like hint; the
    hint itself for straight or non-planar curves."""
    hint = _unit(np.asarray(hint, float))
    P = np.vstack([_samples(s, 16) for s in spans_list])
    c = P.mean(axis=0)
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    _, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    if len(S) < 3 or S[1] < 1e-7 * size * np.sqrt(len(P)):
        return hint, True                     # a straight line: in any plane through it
    n = Vt[2]
    if float(np.abs((P - c) @ n).max()) > 1e-5 * size:
        return hint, False
    dot = float(n @ hint)
    if abs(dot) < 1e-6:                       # square to the hint: a stable choice
        k = int(np.argmax(np.abs(n)))
        if n[k] < 0:
            n = -n
    elif dot < 0:
        n = -n
    return _unit(n), True


def signed_area(spans, N):
    P = _samples(spans, 16)
    return 0.5 * float(np.cross(P, np.roll(P, -1, axis=0)).sum(axis=0) @ N)


def _basis(N):
    a = np.array([1.0, 0, 0]) if abs(N[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(np.cross(N, a))
    return e1, np.cross(N, e1)


# --------------------------------------------------------------------------
# Exact offsets: lines and circular arcs
# --------------------------------------------------------------------------

def arc_centre(seg):
    """Centre and radius of a circular rational quadratic span."""
    P = curvejoin._cart(seg)
    M = 0.5 * (P[0] + P[2])
    h = M - P[1]
    hl = float(np.linalg.norm(h))
    a2 = float((P[0] - P[1]) @ (P[0] - P[1]))
    O = P[1] + h / hl * (a2 / hl)
    return O, float(np.linalg.norm(P[0] - O))


def span_kind(seg, N):
    if curvetools.is_straight(seg):
        return "LINE"
    if curvetools.is_circular(seg):
        P = curvejoin._cart(seg)
        an = _unit(np.cross(P[1] - P[0], P[2] - P[1]))
        if abs(abs(float(an @ N)) - 1.0) < 1e-6:     # the arc lies square to the offset plane
            return "ARC"
    return "FREE"


def offset_line(seg, N, d):
    P = curvejoin._cart(seg)
    v = np.cross(N, _unit(P[-1] - P[0]))
    if np.linalg.norm(v) < 1e-9:
        raise ValueError("a line runs along the offset plane's normal: choose another plane")
    out = seg.copy()
    out[:, :3] += np.outer(seg[:, 3], d * _unit(v))
    return out


def offset_arc(seg, N, d):
    """The concentric arc, or None where it shrinks to nothing."""
    O, R = arc_centre(seg)
    C, dC = curvetrim.eval_d(seg, 0.5)
    v = _unit(np.cross(N, _unit(dC)))
    k = (R + d * float(v @ ((C - O) / R))) / R
    if k <= 1e-9:
        return None
    P = curvejoin._cart(seg)
    out = seg.copy()
    out[:, :3] = (O + k * (P - O)) * seg[:, 3:4]
    return out


# --------------------------------------------------------------------------
# Free-form runs: sampled, loops cut out, fitted
# --------------------------------------------------------------------------

def _span_count(seg, N, d, tol, per=PER_SPAN):
    """Samples on one span so that the offset polyline's chords stay within
    a tenth of tol of the offset (the fit's error is measured against them)."""
    ts = np.linspace(0.0, 1.0, per + 1)
    Q = []
    for t in ts:
        p, dp = curvetrim.eval_d(seg, float(t))
        Q.append(p + d * _unit(np.cross(N, _unit(dp))))
    Q = np.array(Q)
    sag = float(np.linalg.norm(Q[:-2] - 2 * Q[1:-1] + Q[2:], axis=1).max()) / 8.0
    return int(min(MAX_PER_SPAN, max(per, np.ceil(per * np.sqrt(sag / (0.1 * tol))))))


def _offset_samples(run, N, d, tol):
    C, Q, T = [], [], []
    for i, seg in enumerate(run):
        per = _span_count(seg, N, d, tol)
        for t in np.linspace(0.0, 1.0, per + 1)[(1 if i else 0):]:
            p, dp = curvetrim.eval_d(seg, float(t))
            tt = _unit(dp)
            v = np.cross(N, tt)
            if np.linalg.norm(v) < 1e-9:
                raise ValueError("the curve runs along the offset plane's normal: choose another plane")
            C.append(p)
            T.append(tt)
            Q.append(p + d * _unit(v))
    return np.array(C), np.array(Q), np.array(T)


def _seg_cross_2d(a0, a1, B0, B1):
    """Which of the segments B0[k]-B1[k] cross segment a0-a1 (2D), and where (s along a)."""
    r = a1 - a0
    S = B1 - B0
    den = r[0] * S[:, 1] - r[1] * S[:, 0]
    q = B0 - a0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (q[:, 0] * S[:, 1] - q[:, 1] * S[:, 0]) / den
        u = (q[:, 0] * r[1] - q[:, 1] * r[0]) / den
    hit = (np.abs(den) > 1e-300) & (s >= 0) & (s <= 1) & (u >= 0) & (u <= 1)
    return hit, s


def remove_loops(Q, C, N):
    """Split the offset polyline Q (base polyline C) where it folds back: each
    fold's loop is cut out at the crossing that closes it. Returns
    ([polylines], unresolved folds)."""
    e1, e2 = _basis(N)
    out, bad = [], 0
    cur_Q, cur_C = Q, C
    for _ in range(200):
        dq, dc = np.diff(cur_Q, axis=0), np.diff(cur_C, axis=0)
        rev = np.einsum("ij,ij->i", dq, dc) <= 0.0
        if not rev.any():
            break
        r0 = int(np.argmax(rev))
        r1 = r0
        while r1 + 1 < len(rev) and rev[r1 + 1]:
            r1 += 1
        q2 = np.column_stack([cur_Q @ e1, cur_Q @ e2])
        nseg = len(q2) - 1
        best = None
        for i in range(r0 - 1, -1, -1):
            if best is not None and (r0 - 1 - i) > best[0]:
                break
            js = np.arange(r1 + 1, nseg)
            if not len(js):
                break
            hit, s = _seg_cross_2d(q2[i], q2[i + 1], q2[js], q2[js + 1])
            if hit.any():
                k = int(np.argmax(hit))
                j = int(js[k])
                if best is None or j - i < best[0]:
                    best = (j - i, i, j, float(s[k]))
        if best is None:
            bad += 1
            break
        _, i, j, s = best
        X = cur_Q[i] + s * (cur_Q[i + 1] - cur_Q[i])
        out.append((np.vstack([cur_Q[:i + 1], X]), np.vstack([cur_C[:i + 1], cur_C[i + 1]])))
        cur_Q = np.vstack([X, cur_Q[j + 1:]])
        cur_C = np.vstack([cur_C[j], cur_C[j + 1:]])
    out.append((cur_Q, cur_C))
    return [q for q, _ in out], bad


def _dedupe(S):
    keep = np.concatenate([[True], np.linalg.norm(np.diff(S, axis=0), axis=1) > 1e-12])
    return S[keep]


def fit_polyline(S, T0, T1, deg, tol, closed=False):
    """Piece dict of a uniform spline through the offset samples S."""
    S = _dedupe(S)
    if closed:
        X, e = curvetools.fit_auto(S, deg, True, None, tol)
        sp = curvetools.rebuilt_spec(X, deg, True)
        d_, P, W, U = geom.blender_nurbs_to_clamped(sp["points"], sp["weights"], sp["order"], True,
                                                    sp["endpoint"], sp["bezier"])
        return {"spans": curvejoin._clamped_to_spans(d_, P, W, U), "fit": (X, deg, True), "err": e}
    if len(S) < deg + 2 or np.linalg.norm(S[-1] - S[0]) < 1e-12:
        return {"spans": [_line(S[0], S[-1])], "fit": None, "err": 0.0}
    X, e = curvetools.fit_auto(S, deg, False, (_unit(T0), _unit(T1)), tol)
    U = geom.clamped_uniform_knots(len(X), deg)
    return {"spans": curvejoin._clamped_to_spans(deg, X, None, U), "fit": (X, deg, False), "err": e}


def offset_free_run(run, N, d, tol, closed=False):
    """Pieces offsetting a run of free-form spans (a fold splits it)."""
    deg = max(3, min(curvejoin.MAX_DEGREE, max(len(s) - 1 for s in run)))
    C, Q, T = _offset_samples(run, N, d, tol)
    if closed:
        rev = np.einsum("ij,ij->i", np.diff(Q, axis=0), np.diff(C, axis=0)) <= 0
        if not rev.any():
            return [fit_polyline(Q, None, None, deg, tol, closed=True)], 0, 0
    polys, bad = remove_loops(Q, C, N)
    pieces = []
    for k, S in enumerate(polys):
        T0 = T[0] if k == 0 else S[min(1, len(S) - 1)] - S[0]
        T1 = T[-1] if k == len(polys) - 1 else S[-1] - S[max(len(S) - 2, 0)]
        pieces.append(fit_polyline(S, T0, T1, deg, tol))
    return pieces, len(polys) - 1, bad


# --------------------------------------------------------------------------
# A chain: smooth segments, their pieces, the corners between them
# --------------------------------------------------------------------------

def _segments(spans, closed):
    """[(spans, corner point after it or None)] split at tangent breaks; a
    closed chain with corners is turned to start at one."""
    cuts = curvejoin.split_points(spans)
    n = len(spans)
    if closed:
        a, b = curvejoin._tangent(spans[-1], 1), curvejoin._tangent(spans[0], 0)
        seam_corner = float(a @ b) < np.cos(np.radians(curvejoin.CORNER_DEG))
        if cuts and not seam_corner:
            k = cuts[0]
            spans = spans[k:] + spans[:k]
            cuts = [c - k for c in cuts[1:]]
            seam_corner = True
        if not seam_corner and not cuts:
            return [(spans, None)], False
    bounds = [0] + cuts + [n]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        segs.append((spans[a:b], _end(spans[a:b])))
    if not closed:
        segs[-1] = (segs[-1][0], None)
    return segs, True


def _rotate_to_kind_change(spans, kinds):
    for k in range(len(kinds)):
        if (kinds[k] == "FREE") != (kinds[k - 1] == "FREE"):
            return spans[k:] + spans[:k], kinds[k:] + kinds[:k]
    return spans, kinds


def chain_pieces(spans, closed, N, d, tol, info):
    """Raw offset pieces of one chain: [{"spans", "fit", "corner"}], corner =
    the base corner point after the piece (None: a smooth joint or a cut)."""
    segs, cornered = _segments(spans, closed)
    pieces = []
    for seg, corner in segs:
        kinds = [span_kind(s, N) for s in seg]
        smooth_closed = closed and not cornered
        if smooth_closed and all(k == "FREE" for k in kinds):
            ps, folds, bad = offset_free_run(seg, N, d, tol, closed=True)
            info["folds"] += folds
            info["unresolved"] += bad
            for p in ps:
                p["corner"] = None
            pieces += ps
            continue
        if smooth_closed:
            seg, kinds = _rotate_to_kind_change(seg, kinds)
        k = 0
        seg_pieces = []
        while k < len(seg):
            if kinds[k] == "FREE":
                j = k
                while j < len(seg) and kinds[j] == "FREE":
                    j += 1
                ps, folds, bad = offset_free_run(seg[k:j], N, d, tol)
                info["folds"] += folds
                info["unresolved"] += bad
                seg_pieces += [dict(p, corner=None) for p in ps]
                k = j
                continue
            o = offset_line(seg[k], N, d) if kinds[k] == "LINE" else offset_arc(seg[k], N, d)
            if o is None:
                info["collapsed"] += 1
                seg_pieces.append(None)          # a break: the neighbours meet across it
            else:
                info["exact"] += 1
                seg_pieces.append({"spans": [o], "fit": None, "corner": None})
            k += 1
        # merge exact pieces that follow on smoothly; a dropped arc stays a break
        merged = []
        brk = False
        for p in seg_pieces:
            if p is None:
                brk = True
                continue
            if merged and not brk and merged[-1]["fit"] is None and p["fit"] is None:
                merged[-1]["spans"] += p["spans"]
            else:
                merged.append(p)
            brk = False
        if merged:
            merged[-1]["corner"] = corner
        pieces += merged
    return pieces


def _trim_end(spans, g):
    if g >= len(spans) - 1e-9:
        return spans
    if g <= 1e-9:
        return []
    return curvetrim.split_spans(spans, [g], False)[0][0]


def _trim_start(spans, g):
    if g <= 1e-9:
        return spans
    if g >= len(spans) - 1e-9:
        return []
    return curvetrim.split_spans(spans, [g], False)[-1][0]


def _arc_spans(O, a, b):
    """Exact arc about O from a to b (the short way)."""
    u0, u1 = a - O, b - O
    r = 0.5 * (np.linalg.norm(u0) + np.linalg.norm(u1))
    e1 = _unit(u0)
    nn = _unit(np.cross(u0, u1))
    e2 = np.cross(nn, e1)
    phi = float(np.arctan2(np.linalg.norm(np.cross(u0, u1)), float(u0 @ u1)))
    m = max(1, int(np.ceil(phi / (np.pi / 2) - 1e-9)))
    pt = lambda t: O + r * (np.cos(t) * e1 + np.sin(t) * e2)
    out = []
    for k in range(m):
        t0, t1 = phi * k / m, phi * (k + 1) / m
        h = 0.5 * (t1 - t0)
        w = np.cos(h)
        mid = O + (r / w) * (np.cos(t0 + h) * e1 + np.sin(t0 + h) * e2)
        p0 = a if k == 0 else pt(t0)
        p1 = b if k == m - 1 else pt(t1)
        out.append(np.array([[*p0, 1.0], [*(mid * w), w], [*p1, 1.0]]))
    return out


def _rays(a, ta, b, tb):
    """s, t with a + s ta = b - t tb (least squares), and the miss."""
    M = np.column_stack([ta, tb])
    st, *_ = np.linalg.lstsq(M, b - a, rcond=None)
    s, t = float(st[0]), float(st[1])
    return s, t, float(np.linalg.norm(a + s * ta - (b - t * tb)))


def junction(A, B, corner, mode, d, tol):
    """Resolve where piece A ends and piece B starts. Returns
    (A spans, B spans, bridge spans, status) with status 'ok' or 'drop'."""
    a, b = _end(A), _start(B)
    if np.linalg.norm(a - b) <= tol:
        A, B = [s.copy() for s in A], [s.copy() for s in B]
        curvejoin._snap(A[-1], -1, B[0], 0)
        return A, B, [], "ok"
    hits = curvetrim.intersect(A, B, tol=min(tol, 1e-6 * curvetrim.spans_size(A + B) + 1e-9))
    if hits:
        gA, gB, P = min(hits, key=lambda h: (len(A) - h[0]) + h[1])
        A2, B2 = _trim_end(A, gA), _trim_start(B, gB)
        if A2 and B2:
            A2, B2 = [s.copy() for s in A2], [s.copy() for s in B2]
            A2[-1][-1, :3] = P * A2[-1][-1, 3]
            B2[0][0, :3] = P * B2[0][0, 3]
            return A2, B2, [], "ok"
    ta, tb = curvejoin._tangent(A[-1], 1), curvejoin._tangent(B[0], 0)
    s, t, miss = _rays(a, ta, b, tb)
    gap = float(np.linalg.norm(b - a))
    parallel = abs(float(ta @ tb)) > 1.0 - 1e-9
    if not parallel and s > 0 and t > 0 and miss <= 1e-6 * max(gap, 1.0):
        if mode == "ROUND" and corner is not None and \
                abs(np.linalg.norm(a - corner) - np.linalg.norm(b - corner)) <= 1e-6 * max(abs(d), 1.0):
            return A, B, _arc_spans(np.asarray(corner, float), a, b), "ok"
        if mode == "CHAMFER":
            return A, B, [_line(a, b)], "ok"
        X = a + s * ta
        return A, B, [_line(a, X), _line(X, b)], "ok"
    if not parallel and s < 0 and t < 0:
        return A, B, [], "drop"         # they overlap but never cross: one of them is used up
    return A, B, [_line(a, b)], "ok"   # parallel ends (a slot narrower than the offset), etc.


def resolve(pieces, closed, mode, d, tol, info):
    """Trim / extend / bridge the pieces into one list of spans. A piece that
    the offset uses up (a short edge between two inside corners) is dropped
    and its neighbours are made to meet instead."""
    P = [dict(p, spans=[s.copy() for s in p["spans"]]) for p in pieces]
    br = [[] for _ in P]                  # what bridges the gap after piece k
    j, guard = 0, 0
    while len(P) > 1 and j < (len(P) if closed else len(P) - 1) and guard < 4 * len(pieces) + 8:
        guard += 1
        m = len(P)
        i2 = (j + 1) % m
        A, B, bridge, status = junction(P[j]["spans"], P[i2]["spans"], P[j]["corner"], mode, d, tol)
        if status == "drop":
            if m > 2 and (closed or j > 0):
                drop = j                  # trimmed at its start already: used up
            elif m > 2 and (closed or i2 < m - 1):
                drop = i2
            else:
                br[j] = [_line(_end(A), _start(B))]
                info["unresolved"] += 1
                j += 1
                continue
            info["dropped"] += 1
            if drop == j:
                prev = (j - 1) % m
                P[prev]["corner"] = P[j]["corner"] if P[j]["corner"] is not None else P[prev]["corner"]
                br[prev] = []
                P.pop(j)
                br.pop(j)
                j = prev if prev < j else len(P) - 1
            else:
                P[j]["corner"] = P[i2]["corner"] if P[i2]["corner"] is not None else P[j]["corner"]
                br[j] = []
                P.pop(i2)
                br.pop(i2)
                if i2 < j:
                    j -= 1
            continue
        if not A or not B:
            info["unresolved"] += 1
        P[j]["spans"], P[i2]["spans"] = A, B
        br[j] = bridge
        j += 1
    out = []
    for k, p in enumerate(P):
        out += p["spans"]
        out += br[k]
    return out, P


def offset_chain(spans, closed, N, d, mode, tol, info):
    """Spline spec (world) of one chain's offset."""
    pieces = chain_pieces(spans, closed, N, d, tol, info)
    if not pieces:
        raise ValueError("the offset is empty: the distance is larger than the curve allows")
    info["err"] = max([info["err"]] + [p.get("err", 0.0) for p in pieces])
    if len(pieces) == 1 and pieces[0]["fit"] is not None and (pieces[0]["fit"][2] or not closed):
        X, deg, cl = pieces[0]["fit"]
        return curvetools.rebuilt_spec(X, deg, cl), pieces[0]["spans"]
    out, _ = resolve(pieces, closed, mode, d, tol, info)
    out = [s for s in out if np.linalg.norm(curvejoin._cart(s)[-1] - curvejoin._cart(s)[0]) > 1e-12
           or len(s) > 2]
    if not out:
        raise ValueError("the offset is empty: the distance is larger than the curve allows")
    if closed and np.linalg.norm(_end(out) - _start(out)) <= max(tol, 1e-9):
        curvejoin._snap(out[-1], -1, out[0], 0)
    return curvejoin.spans_to_spec(out, simplify=True), out


def crosses_itself(spans, closed, tol):
    if len(spans) < 2:
        return False
    hits = curvetrim.intersect(spans, spans, tol=max(tol, 1e-6), same=True)
    seam = _start(spans)
    eps = 10 * max(tol, 1e-6)
    return any(not (closed and np.linalg.norm(P - seam) <= eps) for _, _, P in hits)


def offset_curves(pieces, p):
    """Specs (world) offsetting the input span lists with the params p."""
    d0 = float(p.get("distance", 0.1))
    if abs(d0) < 1e-12:
        raise ValueError("Distance is 0")
    tol = float(p.get("tolerance", 1e-4))
    mode = p.get("oc_corner", "SHARP")
    hint = np.asarray(p.get("plane_n", (0.0, 0.0, 1.0)), float)
    if p.get("oc_mode", "JOINED") == "JOINED":
        chains = [(s, c) for s, _, c, _ in curvejoin.chain_pieces(pieces, CHAIN_TOL)]
    else:
        chains = [(s, curvetrim.is_closed(s)) for s in pieces]
    info = {"chains": len(chains), "exact": 0, "folds": 0, "collapsed": 0, "dropped": 0, "unresolved": 0,
            "err": 0.0, "crossings": 0, "planar": True}
    specs = []
    sides = (1.0, -1.0) if p.get("both_sides") else (1.0,)
    for spans, closed in chains:
        if p.get("oc_plane") == "WORKPLANE":
            N, planar = _unit(hint), True
        else:
            N, planar = plane_normal([spans], hint)
        info["planar"] &= planar
        base = d0 * (-1.0 if p.get("flip") else 1.0)
        if closed and signed_area(spans, N) > 0:
            base = -base                   # counter-clockwise: left is inside, positive goes out
        for sgn in sides:
            sp, out = offset_chain(spans, closed, N, sgn * base, mode, tol, info)
            try:
                if crosses_itself(out, closed, tol):
                    info["crossings"] += 1
            except Exception:
                pass
            specs.append(sp)
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return specs


@features.builder("offset_curve")
def build_offset_curve(inputs, p):
    pieces = [curvejoin.input_spans(o, r) for o, r in inputs]
    return ("splines", offset_curves(pieces, p))


def report_text(info):
    msg = f"Offset {info['chains']} curve(s)"
    notes = []
    if info["exact"]:
        notes.append("lines / arcs exact")
    if info["err"] > 0:
        notes.append(f"fit within {info['err']:.2g}")
    if info["folds"]:
        notes.append(f"{info['folds']} fold(s) cut out")
    if info["collapsed"]:
        notes.append(f"{info['collapsed']} arc(s) too small, dropped")
    if info["dropped"]:
        notes.append(f"{info['dropped']} piece(s) used up")
    if not info["planar"]:
        notes.append("not planar: offset square to the work plane's normal")
    return msg + (" · " + ", ".join(notes) if notes else "")


def warnings(info):
    out = []
    if info["crossings"]:
        out.append("the offset crosses itself: the distance is too large for this shape")
    if info["unresolved"]:
        out.append(f"{info['unresolved']} place(s) could not be trimmed cleanly (bridged with a line)")
    return out


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import json  # noqa: E402

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty  # noqa: E402

MODE_ITEMS = [("INDIVIDUAL", "Individual", "Offset each curve on its own (one result per selected object)"),
              ("JOINED", "Joined", "Chain the curves whose ends touch and offset each chain as one curve: "
                                   "corners between them are trimmed or filled")]
CORNER_ITEMS = [("SHARP", "Sharp", "Extend the pieces by tangent lines until they meet"),
                ("ROUND", "Round", "Fill the corner with an exact arc about the original corner"),
                ("CHAMFER", "Chamfer", "Connect the pieces with a straight line")]
PLANE_ITEMS = [("AUTO", "Curve Plane", "The plane the curves lie in (the work plane for straight lines "
                                       "and non-planar curves)"),
               ("WORKPLANE", "Work Plane", "Offset square to the active work plane's normal (World: Z)")]


def _hint(context):
    from . import workplanes
    fr = workplanes.active_frame(context.scene)
    return [0.0, 0.0, 1.0] if fr is None else [float(x) for x in fr[3]]


class NPROTO_OT_offset_curve(bpy.types.Operator):
    """Offset curves within their plane: lines and arcs exactly, free-form curves
    within Tolerance. Corners are trimmed or filled (Sharp, Round, Chamfer).
    Individual offsets each curve, Joined chains touching curves first"""
    bl_idname = "nproto.offset_curve"
    bl_label = "Offset Curve"
    bl_options = {'REGISTER', 'UNDO'}

    distance: FloatProperty(name="Distance", default=0.1, unit='LENGTH', soft_min=-10.0, soft_max=10.0,
                            description="Positive: outward on closed curves, to the left (seen from the "
                                        "plane's normal) on open ones")
    oc_mode: EnumProperty(name="Curves", items=MODE_ITEMS, default="JOINED")
    oc_corner: EnumProperty(name="Corners", items=CORNER_ITEMS, default="SHARP")
    oc_plane: EnumProperty(name="Plane", items=PLANE_ITEMS, default="AUTO")
    flip: BoolProperty(name="Flip", default=False, description="Offset to the other side")
    both_sides: BoolProperty(name="Both Sides", default=False, description="Offset to both sides")
    tolerance: FloatProperty(name="Tolerance", default=1e-4, min=1e-7, soft_max=0.01, precision=6, unit='LENGTH',
                             description="Largest deviation of a fitted free-form offset from the exact one")
    names: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def _objs(self, context):
        if not self.names:
            self.names = json.dumps([o.name for o in context.selected_objects
                                     if o.type == 'CURVE' and len(o.data.splines)])
        return [bpy.data.objects[n] for n in json.loads(self.names) if n in bpy.data.objects]

    def invoke(self, context, event):
        curvetrim._object_mode(context)
        self.names = ""
        if not self._objs(context):
            self.report({'WARNING'}, "Select curves to offset")
            return {'CANCELLED'}
        if bpy.app.background:
            return self.execute(context)
        return context.window_manager.invoke_props_dialog(self, width=320, title=self.bl_label,
                                                          confirm_text=self.bl_label)

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "distance")
        lay.row().prop(self, "oc_mode", expand=True)
        lay.row().prop(self, "oc_corner", expand=True)
        lay.prop(self, "oc_plane")
        lay.prop(self, "flip")
        lay.prop(self, "both_sides")
        lay.prop(self, "tolerance")

    def _params(self, context):
        return {"distance": self.distance, "oc_mode": self.oc_mode, "oc_corner": self.oc_corner,
                "flip": self.flip, "both_sides": self.both_sides, "tolerance": self.tolerance,
                "plane_n": _hint(context)} | ({"oc_plane": "WORKPLANE"} if self.oc_plane == "WORKPLANE" else {})

    def execute(self, context):
        from .ops import _select_only
        from .primitives import write_spline_object
        objs = self._objs(context)
        if not objs:
            self.report({'WARNING'}, "Select curves to offset")
            return {'CANCELLED'}
        params = self._params(context)
        groups = [objs] if self.oc_mode == "JOINED" else [[o] for o in objs]
        made, info_all, warn = [], [], []
        for grp in groups:
            inputs = [(o, f"S{i}") for o in grp for i in range(len(o.data.splines))]
            try:
                g = features.run("offset_curve", inputs, params)
            except Exception as exc:
                self.report({'ERROR'}, f"Offset Curve ({', '.join(o.name for o in grp)}): {exc}")
                continue
            name = (grp[0].name + "_offset") if len(grp) == 1 else "OffsetCurve"
            cu = bpy.data.curves.new(name, 'CURVE')
            out = bpy.data.objects.new(name, cu)
            (context.collection or context.scene.collection).objects.link(out)
            write_spline_object(out, g[1])
            if features.history_enabled(context):
                features.record(out, "offset_curve", inputs, params)
            made.append(out)
            info_all.append(dict(LAST_INFO))
            warn += warnings(LAST_INFO)
        if not made:
            return {'CANCELLED'}
        _select_only(context, made)
        for w in dict.fromkeys(warn):
            self.report({'WARNING'}, w)
        if not warn:
            merged = {k: (max(i[k] for i in info_all) if k == "err" else all(i[k] for i in info_all) if k == "planar"
                          else sum(i[k] for i in info_all)) for k in info_all[0]}
            self.report({'INFO'}, report_text(merged))
        return {'FINISHED'}


classes = (NPROTO_OT_offset_curve,)
