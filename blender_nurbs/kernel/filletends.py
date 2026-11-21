"""Surface Fillet between surfaces that meet along only part of an edge.

A wall shorter than the floor it stands on, or two sheets meeting edge to
edge where one is shorter (a tent with one short side): the crossing ends
where the shorter surface ends, while the longer one carries on. Three things
follow, all handled here:

- Length. The fillet must end at the crossing's ends, each end section in
  the plane square to the crossing there. The kernel's rolling ball runs on
  past the end of the shorter surface in a tapering one-sided tail, so with
  `uneven_ends` the worker builds circles and chamfers from exact sections
  (secfillet.py, as chord, blend and soft fillets always are), which end
  exactly there.
- Notch (Ends: Notch). The fillet's rail on the longer surface stops inside
  it, so the rail alone does not cut it into regions. `notch_sheet` closes
  the cut: a plane square to the crossing through its end, from the crossing
  out past the rail. The longer surface loses only the strip under the
  fillet; past the shorter surface's end it keeps its own edge, with a step
  the size of the fillet's setback.
- Extend (Ends: Extend). `extend_to_partner` grows the shorter surface first,
  naturally (its last span continued as the same polynomial, natext.py), at
  its boundary that the crossing runs out through, until that boundary sits
  exactly where the longer surface ends along the crossing (a parameter, not
  an averaged distance: trimext.extended_surface_to). The fillet then runs
  the whole edge and both surfaces trim edge to edge.

Kernel and numpy only: the worker imports it top-level, Blender as
blender_nurbs.kernel.filletends (the Surface Fillet preview)."""

import numpy as np

try:
    from . import secfillet, trimext
except ImportError:          # in the worker: the kernel folder is on sys.path
    import secfillet
    import trimext


def size_of(*breps):
    lo, hi = [], []
    for b in breps:
        a, c = b.bounding_box()
        lo.append(a)
        hi.append(c)
    return float(np.linalg.norm(np.max(np.array(hi, float), 0) - np.min(np.array(lo, float), 0))) or 1.0


def dist_to_face(brep, p):
    return min(f.closest_point(tuple(map(float, p)))[1] for f in brep.faces())


def dist_to_edges(brep, p):
    return min(e.closest_point(tuple(map(float, p)))[2] for e in brep.edges())


def crossing(sm, A, B, size=None):
    """The longest intersection chain of A and B (secfillet.Crossing), None
    when they do not cross."""
    size = size or size_of(A, B)
    try:
        curves, _ = sm.intersect_breps(A, B)
    except RuntimeError:
        return None
    cr = [secfillet.Crossing(ch) for ch in secfillet.chains(list(curves or []), 1e-5 * size)]
    cr = [c for c in cr if c.length > 1e-9]
    return max(cr, key=lambda c: c.length) if cr else None


def crossing_ends(sm, A, B, size=None):
    """[(end point, unit tangent pointing out of the crossing)] of A and B's
    crossing: two for an open one, none for a closed one or none at all."""
    cr = crossing(sm, A, B, size)
    if cr is None or cr.closed:
        return []
    out = []
    for s, sign in ((0.0, -1.0), (1.0, 1.0)):
        p, t = cr.at(s)
        out.append((np.asarray(p, float), sign * np.asarray(t, float)))
    return out


def uneven_ends(sm, A, B, size=None):
    """Ends of the crossing where one surface carries on past it and the other
    does not: [(end, tangent out, 'a' or 'b' = the SHORTER surface)]."""
    size = size or size_of(A, B)
    out = []
    for c, t in crossing_ends(sm, A, B, size):
        h = 1e-3 * size
        on_a = dist_to_face(A, c + h * t) < 0.1 * h
        on_b = dist_to_face(B, c + h * t) < 0.1 * h
        if on_a != on_b:
            out.append((c, t, "b" if on_a else "a"))
    return out


def _normalized(ext, surf):
    """The surface with its knots scaled to 0..1 (a natural surface's usual form)."""
    d = ext.surface_cvs(surf)

    def nrm(K):
        K = [float(k) for k in K]
        return [(k - K[0]) / (K[-1] - K[0]) for k in K]
    return ext.surface_from_cvs(d["degree_u"], d["degree_v"], [tuple(map(float, p)) for p in d["points"]],
                                nrm(d["knots_u"]), nrm(d["knots_v"]), d.get("weights") or None)


def _across(sm, surf, iso, p):
    _, uvs, _ = sm.surface_closest_point(surf, tuple(map(float, p)))
    uv = uvs[0]
    u, v = (uv.x, uv.y) if hasattr(uv, "x") else (float(uv[0]), float(uv[1]))
    return u if iso in ("U0", "U1") else v


def _grown(sm, X, iso, at, to):
    """X (one face) with its boundary iso moved out to the parameter `to`."""
    ext = sm.blender_ext
    surf = X.faces()[0].surface()
    if trimext.is_natural(X):
        return ext.brep_from_surface(_normalized(ext, trimext.extended_surface_to(surf, iso, to)))
    return trimext.extend_trimmed(X, iso, at, None, to=to)


def extend_to_partner(sm, A, B):
    """(A, B, [record]) with the shorter surface grown at every uneven end of
    their crossing until its boundary reaches the longer one's end there.
    record: {'key': 'a' / 'b', 'iso', 'at', 'to', 'grown': world length}.
    Breps are world space and not modified (new ones are returned)."""
    ext = sm.blender_ext
    size = size_of(A, B)
    tol = 2e-4 * size
    breps = {"a": A, "b": B}
    records = []
    for c, t, key in uneven_ends(sm, A, B, size):
        X, Y = breps[key], breps["b" if key == "a" else "a"]
        if X.face_count() != 1:
            raise RuntimeError("Extend: the shorter surface has several faces (Unjoin it first)")
        surf = X.faces()[0].surface()
        h = 1e-3 * size
        probe = c + h * t
        best = None
        for i, (iso, a0, a1) in trimext.boundary_edges(X).items():
            if X.edges()[i].closest_point(tuple(map(float, c)))[2] > tol:
                continue
            try:
                big = trimext.extended_surface(surf, iso, 2.0 * size)
            except Exception:
                continue
            bigb = ext.brep_from_surface(big)
            if dist_to_face(bigb, probe) < 0.1 * h:
                best = (iso, 0.5 * (a0 + a1), big, bigb)
                break
        if best is None:
            raise RuntimeError("Extend: the shorter surface's edge at the end of the crossing "
                               "is not on its own boundary")
        iso, at, big, bigb = best
        # where the crossing with the grown surface runs out: the longer surface's end
        cr = crossing(sm, Y, bigb, size)
        if cr is None or cr.closed:
            raise RuntimeError("Extend: the grown surface does not reach along the longer one")
        e0, e1 = cr.at(0.0)[0], cr.at(1.0)[0]
        target = max((e0, e1), key=lambda e: float((np.asarray(e) - c) @ t))
        if float((np.asarray(target) - c) @ t) <= tol:
            continue
        to = _across(sm, big, iso, target)
        old = trimext._boundary_value(surf, iso)
        if (to - old) * (1.0 if iso in ("U1", "V1") else -1.0) <= 1e-12:
            continue
        breps[key] = _grown(sm, X, iso, at, to)
        records.append({"key": key, "iso": iso, "at": float(at), "to": float(to),
                        "grown": float(np.linalg.norm(np.asarray(target) - c))})
    return breps["a"], breps["b"], records


def apply_records(sm, X, records, key):
    """The same growth on another brep of the same surface (a trim target
    that differs from the fillet's base on a history replay)."""
    for r in records:
        if r["key"] == key:
            X = _grown(sm, X, r["iso"], r["at"], r["to"])
    return X


def notch_sheet(sm, X, q, ends, size, reach):
    """A plane cutter closing the trim of X where its rail ends at q inside
    it: square to the crossing at the crossing's end nearest q, from beyond
    that end out past q (the trim extends it across). None when no notch is
    needed (q on X's boundary). Raises when the crossing ends inside X too:
    there is no edge of X to notch back to."""
    ext = sm.blender_ext
    tol = 2e-4 * size
    q = np.asarray(q, float)
    if dist_to_edges(X, q) <= tol or not ends:
        return None
    c, t = min(ends, key=lambda ct: float(np.linalg.norm(q - ct[0])))
    if dist_to_edges(X, c) > tol:
        raise RuntimeError("the crossing ends inside this surface: there is no edge to notch it back to "
                           "(trim it by hand, or use Ends: Extend)")
    d1 = (q - c) - float((q - c) @ t) * t
    L = float(np.linalg.norm(d1))
    if L <= tol:
        return None
    d1 /= L
    d2 = np.cross(t, d1)
    H = max(reach, L)
    Q = [c + s * L * d1 + w * H * d2 for s in (-0.5, 1.5) for w in (-1.0, 1.0)]
    srf = ext.surface_from_cvs(1, 1, [tuple(map(float, x)) for x in Q], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0])
    return ext.brep_from_surface(srf)
