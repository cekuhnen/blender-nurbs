"""Extend / shrink a TRIMMED surface at an edge that lies on its own boundary.

The carrier surface is extended exactly (natext: its last span continued as
the same polynomial) and the face is trimmed again:

- Extending: the picked edge (with the edges next to it on the same iso
  boundary: the run) is replaced by three exact iso curves of the extended
  surface -- out along the surface at the run's two ends, across along the
  new boundary -- and the extended surface is trimmed with that loop. The
  rest of the trim stays as it was; the band the extension adds is bounded by
  the iso lines through the run's ends (the surface's own flow).
- Shrinking: the trim loop is clipped to the kept side of the iso line at the
  new boundary -- every edge that crosses it cut exactly there, the line
  itself the new edge -- and the cut-back surface is trimmed with each piece
  (more than one face when the trim winds back across that line).

Kernel and numpy only: used by extend.py in Blender, testable without it.
"""

import numpy as np

try:
    from .natext import deboor, end_curve, extend_u_end, flip, span
except ImportError:          # the worker imports the kernel folder's modules top-level
    from natext import deboor, end_curve, extend_u_end, flip, span

EDGES = ("U0", "U1", "V0", "V1")


def _k():
    try:
        from . import load
        sm = load()
    except ImportError:      # in the worker
        import _nproto_kernel as sm
    return sm, sm.blender_ext


# --------------------------------------------------------------------------
# which edges lie on the surface's own boundary
# --------------------------------------------------------------------------

def _uv(sm, surf, q):
    _, uvs, _ = sm.surface_closest_point(surf, tuple(float(x) for x in q))
    uv = uvs[0]
    return (uv.x, uv.y) if hasattr(uv, "x") else (float(uv[0]), float(uv[1]))


def surface_size(surf):
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    P = np.array([surf.evaluate(u0 + (u1 - u0) * a, v0 + (v1 - v0) * b)
                  for a in (0, 0.5, 1) for b in (0, 0.5, 1)], float)
    return float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0


def edge_on_iso(surf, edge, tol=None, n=9):
    """(iso, along at the edge's start, along at its end) when the edge runs
    along the iso boundary iso (U0/U1/V0/V1) of surf; None otherwise. along:
    the surface parameter along that boundary (v for U0/U1, u for V0/V1)."""
    sm, _ = _k()
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    if tol is None:
        tol = 1e-3 * surface_size(surf)
    t0, t1 = edge.parameter_range()
    ts = np.linspace(t0, t1, n)
    pts = [np.array(edge.point_at(float(t)), float) for t in ts]
    uvs = [_uv(sm, surf, p) for p in pts]
    for iso in EDGES:
        across_u = iso in ("U0", "U1")
        c = (u1 if iso == "U1" else u0) if across_u else (v1 if iso == "V1" else v0)
        ok = True
        for p, (u, v) in zip(pts, uvs):
            a = v if across_u else u
            q = surf.evaluate(c, a) if across_u else surf.evaluate(a, c)
            if np.linalg.norm(np.array(q, float) - p) > tol:
                ok = False
                break
        if ok:
            al = [(v if across_u else u) for u, v in uvs]
            a0, a1 = al[0], al[-1]
            if abs(a1 - a0) <= 1e-9:
                if np.linalg.norm(pts[-1] - pts[0]) > tol or max(np.linalg.norm(q - pts[0]) for q in pts) <= tol:
                    continue         # a point (a pole), not a run along the boundary
                # a closed edge (the rim of a cylinder): the whole boundary, the way it runs
                lo, hi = (u0, u1) if not across_u else (v0, v1)
                a0, a1 = (lo, hi) if al[len(al) // 2 + 1] > al[len(al) // 2] else (hi, lo)
            return iso, float(a0), float(a1)
    return None


def boundary_edges(brep, tol=None):
    """{edge index in brep.edges(): (iso, along start, along end)} of a
    one-face brep: the edges that lie on the carrier's own boundary."""
    if brep.face_count() != 1:
        return {}
    surf = brep.faces()[0].surface()
    if tol is None:
        tol = 1e-3 * surface_size(surf)
    out = {}
    for i, e in enumerate(brep.edges()):
        try:
            r = edge_on_iso(surf, e, tol)
        except Exception:
            r = None
        if r is not None:
            out[i] = r
    return out


def is_natural(brep, tol=None):
    """One face whose every edge lies on its carrier's boundary (an untrimmed surface)."""
    if brep.face_count() != 1:
        return False
    surf = brep.faces()[0].surface()
    size = surface_size(surf)
    on = boundary_edges(brep, tol)
    for i, e in enumerate(brep.edges()):
        if i in on:
            continue
        try:
            short = e.length() <= 1e-7 * size      # a pole (revolve, cone tip): no boundary to trim
        except Exception:
            short = False
        if not short:
            return False
    return True


# --------------------------------------------------------------------------
# exact curves on the surface
# --------------------------------------------------------------------------

def _grid(d):
    nu, nv = d["count_u"], d["count_v"]
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    w = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    return (np.concatenate([P * w[..., None], w[..., None]], axis=2),
            list(map(float, d["knots_u"])), list(map(float, d["knots_v"])), int(d["degree_u"]), int(d["degree_v"]))


def extended_surface(surf, iso, distance):
    """surf extended at iso by distance, knots NOT normalised: the old part
    keeps its parameters (a trim in UV stays where it was)."""
    _, ext = _k()
    G, Uu, Uv, du, dv = _grid(ext.surface_cvs(surf))
    if iso in ("V0", "V1"):
        G, Uu, Uv, du, dv = G.transpose(1, 0, 2).copy(), Uv, Uu, dv, du
    start = iso in ("U0", "V0")
    far = Uu[-1]
    if start:
        G, Uu = flip(G, Uu)
    G, Uu = extend_u_end(G, Uu, du, Uv, dv, distance)
    if start:
        G, Uu = flip(G, Uu)
        Uu = [x + far - Uu[-1] for x in Uu]       # flipping back shifts them: the far end stays put
    if iso in ("V0", "V1"):
        G, Uu, Uv, du, dv = G.transpose(1, 0, 2).copy(), Uv, Uu, dv, du
    w = G[..., 3]
    pts = [tuple(map(float, G[i, j, :3] / w[i, j])) for i in range(G.shape[0]) for j in range(G.shape[1])]
    rational = np.abs(w - 1.0).max() > 1e-12
    return ext.surface_from_cvs(du, dv, pts, Uu, Uv, [float(x) for x in w.reshape(-1)] if rational else None)


def extended_surface_to(surf, iso, value):
    """surf extended at iso so that boundary sits at the parameter `value`
    (beyond the old one): the same polynomial continued, knots NOT
    normalised, so the old part keeps its parameters. Exact: the target is a
    parameter, not an averaged arc length."""
    _, ext = _k()
    G, Uu, Uv, du, dv = _grid(ext.surface_cvs(surf))
    if iso in ("V0", "V1"):
        G, Uu, Uv, du, dv = G.transpose(1, 0, 2).copy(), Uv, Uu, dv, du
    start = iso in ("U0", "V0")
    far = Uu[-1]
    t = float(value)
    if start:
        t = Uu[0] + Uu[-1] - t
        G, Uu = flip(G, Uu)
    if t <= Uu[-1] + 1e-12:
        raise ValueError("the new boundary lies inside the surface")
    cols = [end_curve([G[i, j] for i in range(G.shape[0])], Uu, du, t) for j in range(G.shape[1])]
    Uu = cols[0][1]
    G = np.array([[cols[j][0][i] for j in range(G.shape[1])] for i in range(len(cols[0][0]))])
    if start:
        G, Uu = flip(G, Uu)
        Uu = [x + far - Uu[-1] for x in Uu]
    if iso in ("V0", "V1"):
        G, Uu, Uv, du, dv = G.transpose(1, 0, 2).copy(), Uv, Uu, dv, du
    w = G[..., 3]
    pts = [tuple(map(float, G[i, j, :3] / w[i, j])) for i in range(G.shape[0]) for j in range(G.shape[1])]
    rational = np.abs(w - 1.0).max() > 1e-12
    return ext.surface_from_cvs(du, dv, pts, Uu, Uv, [float(x) for x in w.reshape(-1)] if rational else None)


def _iso_cvs(surf, fixed_u, value):
    """(homogeneous CVs, knots, degree) of the iso curve of surf at u = value
    (fixed_u) or v = value: exact."""
    _, ext = _k()
    G, Uu, Uv, du, dv = _grid(ext.surface_cvs(surf))
    if fixed_u:                    # a curve in v: collapse the u direction at value
        k = span(Uu, du, G.shape[0] - 1, value)
        Q = [deboor(G[:, j], Uu, du, k, [value] * du) for j in range(G.shape[1])]
        return Q, Uv, dv
    k = span(Uv, dv, G.shape[1] - 1, value)
    Q = [deboor(G[i], Uv, dv, k, [value] * dv) for i in range(G.shape[0])]
    return Q, Uu, du


def _cut(Q, U, p, lo, hi):
    """The piece [lo, hi] of a curve (homogeneous CVs Q, clamped knots U): exact."""
    if hi < U[-1] - 1e-12:
        Q, U = end_curve(Q, U, p, hi)
    if lo > U[0] + 1e-12:
        far = U[-1]
        Qf, Uf = flip(np.array(Q), U)
        Qf, Uf = end_curve(list(Qf), Uf, p, U[0] + U[-1] - lo)
        Q, U = flip(np.array(Qf), Uf)
        U = [x + far - U[-1] for x in U]
    return list(Q), list(U)


def _make_curve(Q, U, p, reverse=False):
    _, ext = _k()
    Q = np.array(Q, float)
    if reverse:
        Q, U = flip(Q, U)
    w = Q[:, 3]
    pts = [tuple(map(float, q[:3] / q[3])) for q in Q]
    rational = np.abs(w - 1.0).max() > 1e-12
    return ext.curve_from_cvs(int(p), pts, [float(x) for x in U], [float(x) for x in w] if rational else None)


def _iso_piece(surf, fixed_u, value, a, b):
    """Curve on surf at u (fixed_u) or v = value, from parameter a to b."""
    Q, U, p = _iso_cvs(surf, fixed_u, value)
    lo, hi = min(a, b), max(a, b)
    Q, U = _cut(Q, U, p, lo, hi)
    return _make_curve(Q, U, p, reverse=a > b)


def _edge_curve(e, forward):
    """The edge's own curve over its range, in loop order."""
    _, ext = _k()
    d = ext.curve_cvs(e.curve())
    p = int(d["degree"])
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    Q = np.concatenate([P * w[:, None], w[:, None]], axis=1)
    U = [float(k) for k in d["knots"]]
    t0, t1 = e.parameter_range()
    Q, U = _cut(list(Q), U, p, max(t0, U[0]), min(t1, U[-1]))
    return _make_curve(Q, U, p, reverse=not forward)


# --------------------------------------------------------------------------
# the run of boundary edges a pick means
# --------------------------------------------------------------------------

def _run(loop, info, iso, at):
    """Indices (in loop order) of the consecutive loop edges on iso around the
    one whose along range holds `at` (else the nearest), and the along values
    where the run starts and ends (loop order)."""
    on = [info[k] if info[k] is not None and info[k][0] == iso else None for k in range(len(loop))]
    cand = [k for k in range(len(loop)) if on[k] is not None]
    if not cand:
        return None

    def miss(k):
        a, b = on[k][1], on[k][2]
        lo, hi = min(a, b), max(a, b)
        return 0.0 if lo <= at <= hi else min(abs(at - lo), abs(at - hi))
    k0 = min(cand, key=miss)
    n = len(loop)
    run = [k0]
    while len(run) < n and on[(run[0] - 1) % n] is not None:
        run.insert(0, (run[0] - 1) % n)
    while len(run) < n and on[(run[-1] + 1) % n] is not None:
        run.append((run[-1] + 1) % n)
    if len(run) == n:
        return None                  # the whole loop on one iso: nothing to grow from
    return run, on[run[0]][1], on[run[-1]][2]


def _loop_info(surf, loop, tol):
    out = []
    for e, fw in loop:
        try:
            r = edge_on_iso(surf, e, tol)
        except Exception:
            r = None
        if r is not None and not fw:
            r = (r[0], r[2], r[1])
        out.append(r)
    return out


def _band_area(surf, iso, c_old, c_new, ta, tb, n=24):
    """Area of the piece of surf between the iso lines across = c_old .. c_new, along = ta .. tb."""
    across_u = iso in ("U0", "U1")
    a = np.linspace(min(c_old, c_new), max(c_old, c_new), n + 1)
    b = np.linspace(min(ta, tb), max(ta, tb), n + 1)
    tot = 0.0
    for i in range(n):
        for j in range(n):
            x, y = 0.5 * (a[i] + a[i + 1]), 0.5 * (b[j] + b[j + 1])
            u, v = (x, y) if across_u else (y, x)
            _, su, sv = surf.derivatives(u, v)
            tot += float(np.linalg.norm(np.cross(su, sv))) * (a[i + 1] - a[i]) * (b[j + 1] - b[j])
    return tot


def _boundary_value(surf, iso):
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    return {"U0": u0, "U1": u1, "V0": v0, "V1": v1}[iso]


def pick_info(brep, edge_index, tol=None):
    """(iso, along at the edge's middle, along start, along end) of a picked
    edge of a one-face brep, None when it is not on the carrier's boundary."""
    if brep.face_count() != 1:
        return None
    surf = brep.faces()[0].surface()
    r = edge_on_iso(surf, brep.edges()[edge_index], tol)
    if r is None:
        return None
    return r[0], 0.5 * (r[1] + r[2]), r[1], r[2]


def run_range(brep, iso, at, tol=None):
    """(along start, along end) of the run a pick at `at` on iso means."""
    face = brep.faces()[0]
    surf = face.surface()
    if tol is None:
        tol = 1e-3 * surface_size(surf)
    loop = face.boundary_loops()[0]
    r = _run(loop, _loop_info(surf, loop, tol), iso, at)
    return None if r is None else (r[1], r[2])


# --------------------------------------------------------------------------
# extend / shrink
# --------------------------------------------------------------------------

def extend_trimmed(brep, iso, at, distance, to=None):
    """A new Brep: the one-face trimmed brep extended (distance > 0) or cut
    back (< 0) at the run of boundary edges on iso around along = at. to:
    grow until that boundary sits at this parameter instead (distance unused)."""
    if brep.face_count() != 1:
        raise RuntimeError("only single surfaces can be extended (Unjoin first)")
    if to is not None:
        return _grow(brep, iso, at, None, to)
    if abs(distance) <= 1e-12:
        return brep.copy()
    if distance > 0:
        return _grow(brep, iso, at, distance)
    return _shrink(brep, iso, at, distance)


def _grow(brep, iso, at, distance, to=None):
    sm, ext = _k()
    face = brep.faces()[0]
    surf = face.surface()
    size = surface_size(surf)
    tol = 1e-3 * size
    loops = face.boundary_loops()
    outer = loops[0]
    run = _run(outer, _loop_info(surf, outer, tol), iso, at)
    if run is None:
        raise RuntimeError("this edge does not lie on the surface's own boundary")
    idx, ta, tb = run
    across_u = iso in ("U0", "U1")
    c_old = _boundary_value(surf, iso)
    area0 = face.area(1e-6)

    def attempt(tight):
        big = extended_surface(surf, iso, distance) if to is None else extended_surface_to(surf, iso, to)
        c_new = _boundary_value(big, iso)
        n = len(outer)
        start = (idx[-1] + 1) % n        # the loop from just after the run back round to it
        curves = []
        k = start
        while k != idx[0]:
            e, fw = outer[k]
            curves.append(_edge_curve(e, fw))
            k = (k + 1) % n
        # the run replaced: out at ta, along the new boundary, back in at tb
        curves.append(_iso_piece(big, not across_u, ta, c_old, c_new))
        curves.append(_iso_piece(big, across_u, c_new, ta, tb))
        curves.append(_iso_piece(big, not across_u, tb, c_new, c_old))
        sizes = [len(curves)]
        for lp in loops[1:]:
            if not lp:
                continue
            curves += [_edge_curve(e, fw) for e, fw in lp]
            sizes.append(len(lp))
        band = _band_area(big, iso, c_old, c_new, ta, tb)
        if tight:
            out = ext.trim_surface_with_3d_curves_tol(big, sizes, curves, 1e-4 * size, 0.01)
        else:
            out = sm.trim_surface_with_3d_curves(big, sizes, curves)
        if out is None or out.face_count() != 1:
            return None
        a = out.area(1e-6)
        if abs(a - (area0 + band)) > 0.02 * band + 1e-4 * area0:
            return None
        return out

    for tight in (True, False):
        try:
            out = attempt(tight)
        except Exception:
            out = None
        if out is not None:
            return out
    raise RuntimeError("the extended trim could not be built")


def _piece(e, forward, t0, t1):
    """Curve of edge e between its parameters t0 and t1 (loop order: t0 first)."""
    _, ext = _k()
    d = ext.curve_cvs(e.curve())
    p = int(d["degree"])
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    Q = np.concatenate([P * w[:, None], w[:, None]], axis=1)
    U = [float(k) for k in d["knots"]]
    lo, hi = max(min(t0, t1), U[0]), min(max(t0, t1), U[-1])
    Q, U = _cut(list(Q), U, p, lo, hi)
    return _make_curve(Q, U, p, reverse=t0 > t1)


def _edge_track(sm, surf, e, forward, n=33):
    """Edge parameters in loop order and the (u, v) of the edge there."""
    t0, t1 = e.parameter_range()
    ts = np.linspace(t0, t1, n) if forward else np.linspace(t1, t0, n)
    return ts, [_uv(sm, surf, e.point_at(float(t))) for t in ts]


def _clip_loop(sm, surf, loop, across_u, c, keep_high):
    """The loop clipped to the kept side of across = c: (arcs, crossings).
    arcs: [(start crossing along, end crossing along, [(edge, forward, t0, t1)])],
    each a run of the loop on the kept side from one crossing to the next;
    None if the loop does not cross (then 'all' or 'none' says where it lies)."""
    kept = (lambda a: a > c) if keep_high else (lambda a: a < c)
    pieces = []          # (edge, fw, t0, t1, kept?) in loop order, split at the crossings
    marks = []           # (index in pieces where a crossing starts a piece, along there)
    for e, fw in loop:
        ts, uvs = _edge_track(sm, surf, e, fw)
        acr = [(u if across_u else v) for u, v in uvs]
        cuts = []
        for k in range(len(ts) - 1):
            if kept(acr[k]) != kept(acr[k + 1]):
                lo, hi = ts[k], ts[k + 1]
                for _ in range(50):           # the crossing: bisect along the edge
                    mid = 0.5 * (lo + hi)
                    uv = _uv(sm, surf, e.point_at(float(mid)))
                    if kept(uv[0] if across_u else uv[1]) == kept(acr[k]):
                        lo = mid
                    else:
                        hi = mid
                t = 0.5 * (lo + hi)
                uv = _uv(sm, surf, e.point_at(float(t)))
                cuts.append((t, uv[1] if across_u else uv[0]))
        start = ts[0]
        state = kept(acr[0])
        for t, along in cuts:
            pieces.append((e, fw, start, t, state))
            marks.append((len(pieces), along))
            start, state = t, not state
        pieces.append((e, fw, start, ts[-1], state))
    if not marks:
        return None, ("all" if pieces and pieces[0][4] else "none")
    # rotate so the pieces start at a crossing, then group into arcs between crossings
    m0 = marks[0][0]
    order = pieces[m0:] + pieces[:m0]
    shift = {(i - m0) % len(pieces): al for i, al in marks}
    arcs = []
    cur, cur_along = [], None
    for i, pc in enumerate(order):
        if i in shift:
            if cur:
                arcs.append((cur_along, shift[i], cur))
            cur, cur_along = [], shift[i]
        cur.append(pc)
    arcs.append((cur_along, shift[0], cur))
    return [(a0, a1, [x[:4] for x in pcs]) for a0, a1, pcs in arcs if pcs[0][4]], "cross"


def _uv_poly(sm, surf, loop):
    P = []
    for e, fw in loop:
        P += _edge_track(sm, surf, e, fw, 17)[1][:-1]
    return np.array(P)


def _inside_poly(pt, P):
    x, y = pt
    c = False
    for i in range(len(P)):
        (x1, y1), (x2, y2) = P[i - 1], P[i]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def _shrink(brep, iso, at, distance):
    """Cut back: the face clipped to the kept side of the iso line at the new
    boundary, every edge crossing it cut exactly there, the line itself the
    new edge; each piece trimmed on the cut-back carrier."""
    sm, ext = _k()
    face = brep.faces()[0]
    surf = face.surface()
    size = surface_size(surf)
    small = extended_surface(surf, iso, distance)
    c = _boundary_value(small, iso)
    across_u = iso in ("U0", "U1")
    keep_high = iso in ("U0", "V0")
    loops = [lp for lp in face.boundary_loops() if lp]
    arcs, where = _clip_loop(sm, surf, loops[0], across_u, c, keep_high)
    if arcs is None:
        raise RuntimeError("the cut back removes the whole surface" if where == "none"
                           else "the cut back line does not cross this surface")
    holes = []
    for lp in loops[1:]:
        h, hw = _clip_loop(sm, surf, lp, across_u, c, keep_high)
        if h is not None:
            raise RuntimeError("the cut back line runs through a hole")
        if hw == "all":
            holes.append(lp)
    # pair the crossings along the line: the kept region's boundary runs along
    # the line between crossings 1-2, 3-4, ... (sorted along it)
    ends = sorted([a[1] for a in arcs] + [a[0] for a in arcs])
    partner = {}
    for k in range(0, len(ends) - 1, 2):
        partner[round(ends[k], 12)] = ends[k + 1]
        partner[round(ends[k + 1], 12)] = ends[k]
    by_start = {round(a[0], 12): a for a in arcs}
    used, faces = set(), []
    for a in arcs:
        if round(a[0], 12) in used:
            continue
        chain, cur = [], a
        while cur is not None and round(cur[0], 12) not in used:
            used.add(round(cur[0], 12))
            chain.append(cur)
            nxt = partner.get(round(cur[1], 12))
            cur = by_start.get(round(nxt, 12)) if nxt is not None else None
        faces.append(chain)
    area0 = face.area(1e-6)
    out = None
    for chain in faces:
        def build(tight):
            big = extended_surface(surf, iso, distance)
            curves = []
            poly = []
            for a0, a1, pcs in chain:
                for e, fw, t0, t1 in pcs:
                    curves.append(_piece(e, fw, t0, t1))
                    poly += _edge_track(sm, surf, e, fw, 9)[1]
                nxt = partner[round(a1, 12)]
                curves.append(_iso_piece(big, across_u, c, a1, nxt))
            mine = []
            for lp in holes:          # the holes inside this piece
                q = _uv_poly(sm, surf, lp)[0]
                if _inside_poly(q, np.array(poly)):
                    mine.append(lp)
            sizes = [len(curves)]
            for lp in mine:
                curves += [_edge_curve(e, fw) for e, fw in lp]
                sizes.append(len(lp))
            if tight:
                b = ext.trim_surface_with_3d_curves_tol(big, sizes, curves, 1e-4 * size, 0.01)
            else:
                b = sm.trim_surface_with_3d_curves(big, sizes, curves)
            return b if b is not None and b.face_count() == 1 and 0 < b.area(1e-6) < area0 * (1 + 1e-4) else None
        made = None
        for tight in (True, False):
            try:
                made = build(tight)
            except Exception:
                made = None
            if made is not None:
                break
        if made is None:
            raise RuntimeError("the cut back trim could not be built")
        if out is None:
            out = made
        else:
            ext.append_brep(out, made)
    return out
