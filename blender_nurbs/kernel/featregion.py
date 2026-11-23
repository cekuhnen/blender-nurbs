"""The feature a face belongs to, for Move Face: the faces enclosed by an
inner loop of the face(s) they sit in -- a hole's wall, a pocket's walls and
floor, a boss's walls and top, an embossed letter. Pure topology on the
kernel's Brep (no Blender), so the add-on (hover, preview) and the worker
(the move itself) find the same faces.

A flood fill from the picked faces crosses every edge except one that leaves
the feature into its seat: an edge of an inner loop of the face on its other
side. So features inside the feature (a hole in a boss's top) come along, and
starting on a face that is no feature (a box's side, a plate's top) floods the
whole body and finds nothing.

A face closed round (a cylinder's or a revolve's side) has a seam edge, and a
hole the seam runs through is no inner loop: the kernel joins its two halves
into the outer loop along the seam. So the loops are read as cycles with the
seam edges left out (face_cycles): the cycles winding round the face are its
boundary, the rest are holes, wherever the seam is."""


def _k():
    try:
        from . import load
        return load()
    except ImportError:      # in the worker
        import _nproto_kernel as sm
        return sm


def _ends(e, fw):
    t0, t1 = e.parameter_range()
    a, b = e.point_at(float(t0)), e.point_at(float(t1))
    return (a, b) if fw else (b, a)


def _dist(a, b):
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def closed_dir(srf):
    """0 when the surface closes round in u (its u ends meet), 1 in v, else
    None (a torus, closed both ways, counts as None too)."""
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    out = []
    for k in (0, 1):
        pts = []
        gap = 0.0
        for i in range(5):
            t = (i + 0.5) / 5
            if k == 0:
                a, b = srf.evaluate(u0, v0 + t * (v1 - v0)), srf.evaluate(u1, v0 + t * (v1 - v0))
            else:
                a, b = srf.evaluate(u0 + t * (u1 - u0), v0), srf.evaluate(u0 + t * (u1 - u0), v1)
            pts += [a, b]
            gap = max(gap, _dist(a, b))
        span = max(_dist(p, q) for p in pts for q in pts) or 1.0
        out.append(gap <= 1e-7 * span and span > 0)
    return 0 if out[0] and not out[1] else 1 if out[1] and not out[0] else None


def seam_edges(face):
    """The edges along which the face meets itself (a closed face's seam)."""
    out = set()
    for e in face.edges():
        if all(f == face for f in e.faces()) and e.length() > 0:
            out.add(e)
    return out


def _winds(sm, srf, k, cycle):
    """Does the cycle (oriented edges) run once round the closed direction k?"""
    lo, hi = (srf.u_domain(), srf.v_domain())[k]
    P = hi - lo
    cs = []
    for e, fw in cycle:
        t0, t1 = e.parameter_range()
        for i in range(8):
            t = t0 + (t1 - t0) * (i / 8 if fw else 1 - i / 8)
            uv = sm.surface_closest_point(srf, tuple(float(x) for x in e.point_at(float(t))))[1][0]
            cs.append(float(uv.y if k else uv.x))
    net = 0.0
    for a, b in zip(cs, cs[1:] + cs[:1]):
        d = (b - a) % P
        net += d - P if d > P / 2 else d
    return abs(net) > P / 2


def face_cycles(face):
    """(cycles, k): the face's boundary as closed cycles of (edge, forward),
    each with its kind -- 'outer' (the outline), 'inner' (a hole or a
    feature's rim) or 'winding' (a closed face's rim running round it: a
    cylinder side's two circles) -- and k, the direction it closes in (None:
    an open face, whose loops are the cycles). On a closed face the seam edges
    are left out and the runs between them joined up."""
    outer = {e for e, _ in face.boundary_outer_loop()}
    seams = seam_edges(face)
    loops = face.boundary_loops()
    k = closed_dir(face.surface()) if seams else None
    if k is None:
        return [([(e, fw) for e, fw in L], "outer" if any(e in outer for e, _ in L) else "inner")
                for L in loops], None
    # runs between the seam edges, each in its loop's order (pole edges dropped)
    runs = []
    for L in loops:
        L = [(e, fw) for e, fw in L if e in seams or e.length() > 0]
        if not any(e in seams for e, _ in L):
            runs.append(L)
            continue
        i = next(j for j, (e, _) in enumerate(L) if e in seams)
        cur = []
        for e, fw in L[i + 1:] + L[:i + 1]:
            if e in seams:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append((e, fw))
        if cur:
            runs.append(cur)
    size = max((e.length() for e in face.edges()), default=1.0) or 1.0
    tol = 1e-6 * size
    cycles = []
    while runs:
        cyc = runs.pop(0)
        start = _ends(*cyc[0])[0]
        while _dist(_ends(*cyc[-1])[1], start) > tol:
            end = _ends(*cyc[-1])[1]
            j = min(range(len(runs)), key=lambda r: _dist(_ends(*runs[r][0])[0], end), default=None)
            if j is None or _dist(_ends(*runs[j][0])[0], end) > 1e3 * tol:
                raise RuntimeError("a closed face's boundary does not close round")
            cyc += runs.pop(j)
        cycles.append(cyc)
    sm = _k()
    kinds = ["winding" if _winds(sm, face.surface(), k, c) else "inner" for c in cycles]
    if "winding" not in kinds:                    # a patch the seam runs across: the widest is its outline
        def span(c):
            P = [e.point_at(float(t)) for e, _ in c for t in e.parameter_range()]
            return max(_dist(p, q) for p in P for q in P)
        kinds[max(range(len(cycles)), key=lambda i: span(cycles[i]))] = "outer"
    return list(zip(cycles, kinds)), k


def inner_loop_edges(face):
    """The edges of the face's holes and feature rims (its inner loops; on a
    closed face its cycles that do not run round it)."""
    cycles, _ = face_cycles(face)
    return {e for c, kind in cycles if kind == "inner" for e, _ in c}


def rims(brep):
    """{edge: the face it is an inner-loop edge of} over the whole body."""
    out = {}
    for f in brep.faces():
        for e in inner_loop_edges(f):
            out[e] = f
    return out


def feature_region(brep, start, rim=None):
    """(region faces, seat faces, the rim edges between them) of the feature
    holding the faces start; None when they are no feature (the fill reached
    the whole body, or never met a seat)."""
    rim = rims(brep) if rim is None else rim
    region = set(start)
    stack = list(start)
    seats, edges = set(), set()
    while stack:
        F = stack.pop()
        for e in F.edges():
            for G in e.faces():
                if G == F or G in region:
                    continue
                if rim.get(e) == G:            # leaving the feature into the face it sits in
                    seats.add(G)
                    edges.add(e)
                    continue
                region.add(G)
                stack.append(G)
    seats -= region                            # a seat reached some other way is no seat
    edges = {e for e in edges if rim.get(e) in seats}
    if not seats or len(region) >= brep.face_count():
        return None
    return region, seats, edges
