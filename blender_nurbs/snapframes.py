"""Snap elements with a frame: what the Align tool mates.

A sketch snap (snapping.Targets.pick) is a point. Aligning one part to
another also needs the direction the element stands in, so element() turns
the snap under the mouse into one of these (all world space):

  type    o (origin)                d (unit)                pin   label
  POINT   the point                 None                    yes   Vertex, End, Mid, CV, Point, Sphere
  LINE    the point on the edge     the edge's direction    no    Edge
  PLANE   the point on the face     its normal              no    Face
          the face's centre         its normal              yes   Center
  RING    the circle's centre       its axis                yes   Ring, Arc  (r: its radius)
  AXIS    the axis point nearest    the axis                no    Axis       (r: radius, or None for a cone)
          the axis middle (centre)  the axis                yes   Axis Mid

pin: the origin itself must meet the other element's; otherwise the element
may slide (a line along d, a plane in itself, an axis along d). A normal and
an axis point towards the viewer, the side being looked at (the outside of a
solid's visible face). "hint" is a second direction where there is one (an
edge middle: the edge; a flat face: the run of its edges), for the spin about
d. "P" is the point the mouse is on, "obj" the object the element belongs to,
"glyph" [(world polyline, style)] what to draw (style "main", "axis", "dash").

Where they come from:
  rings   circular B-rep edges and curves (snapping.Targets.rings, exact), and
          a plain mesh's sharp or open edge loops that lie on a circle
          (mesh_rings: once per mesh)
  faces   the face under the mouse: a NURBS face's surface is sampled and
          fitted (plane, cylinder, cone, sphere, any surface of revolution;
          step_export's fits, cached per surface revision); a mesh face's
          flat region is a plane, its smooth region (faces bending by less
          than 30 degrees) a cylinder when its normals lie in one plane and its
          vertices on a circle round the axis, a sphere when on one
Priority: a point snap (end, vertex, CV) nearer than a ring wins, a ring beats
the edge it is and the mesh vertices and edge middles lying on it, an edge
beats the face behind it, and over a face its own kind.
"""

import math

import numpy as np

from . import snapping, store

RING_PX = snapping.SNAP_PX
RING_MIN = 6                # a mesh loop needs this many vertices to count as a circle
RING_TOL = 1e-4             # ... on a circle within this share of its radius
FIT_TOL = 1e-6              # NURBS face fits: within this share of the face's size
SMOOTH_COS = math.cos(math.radians(30.0))
SMOOTH_MAX = 50_000         # faces in a smooth mesh region looked at
FEATURE_MAX = 400_000       # sharp edges of a mesh looked at for rings
GLYPH_PX = 40.0             # arrows this long on screen

_FACE_KIND = {}             # NURBS session_uid -> (rev, {face index: local kind or None})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-300 else v


def _perp(n):
    n = _unit(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = _unit(np.cross(n, a))
    return x, np.cross(n, x)


def circle_points(c, n, r, k=64):
    x, y = _perp(n)
    t = np.linspace(0.0, 2 * math.pi, k + 1)
    return np.asarray(c, float) + r * (np.cos(t)[:, None] * x + np.sin(t)[:, None] * y)


def px_size(view, P, px=1.0):
    """World length of px pixels at world point P (across the view)."""
    P = np.asarray(P, float)
    xy, _, ok = view.project(P)
    if not ok[0]:
        return 1.0
    o1, d1 = view.ray(xy[0])
    o2, d2 = view.ray(xy[0] + (px, 0.0))
    t1 = float((P - o1) @ d1)
    Q1, Q2 = o1 + t1 * d1, o2 + float((P - o2) @ d1) / max(float(d2 @ d1), 1e-12) * d2
    return float(np.linalg.norm(Q2 - Q1)) or 1.0


def _towards(d, view_dir):
    """d turned towards the viewer (against the view ray)."""
    d = _unit(d)
    return -d if float(d @ view_dir) > 0 else d


def _poly_px(view, W, mouse):
    """Screen distance (px) from the mouse to the polyline W, and the depth there."""
    a, za, oka = view.project(W[:-1])
    b, zb, okb = view.project(W[1:])
    dv = b - a
    s = np.clip(((mouse - a) * dv).sum(axis=1) / np.maximum((dv * dv).sum(axis=1), 1e-18), 0.0, 1.0)
    dist = np.linalg.norm(a + s[:, None] * dv - mouse, axis=1)
    dist[~(oka & okb)] = np.inf
    j = int(np.argmin(dist))
    return float(dist[j]), float(za[j] + s[j] * (zb[j] - za[j]))


def _ray_line_foot(o, d, P, a):
    """The point of the line (P, a) nearest the ray (o, d)."""
    w = P - o
    b, dd, e = float(a @ d), float(a @ w), float(d @ w)
    den = 1.0 - b * b
    if den < 1e-12:
        return P
    return P + ((b * e - dd) / den) * a


# --------------------------------------------------------------------------
# mesh rings and smooth regions (local space, cached on the MeshData)
# --------------------------------------------------------------------------

def sharp_edges(d):
    """The edges of mesh data d bending by 30 degrees or more, open edges and
    those of more than two faces: indices (once per mesh; empty for a mesh
    too dense to look at)."""
    got = getattr(d, "_sharp", None)
    if got is not None:
        return got
    if not len(d.ls) or len(d.lv) > snapping.FLAT_MAX_LOOPS:
        d._sharp = np.zeros(0, np.int64)
        return d._sharp
    N, _, _, lp, order, start, _ = d._faces_geometry()
    cnt = start[1:] - start[:-1]
    sharp = cnt != 2
    two = np.flatnonzero(cnt == 2)
    f0, f1 = lp[order[start[two]]], lp[order[start[two] + 1]]
    sharp[two] = (N[f0] * N[f1]).sum(axis=1) < SMOOTH_COS
    sharp &= cnt > 0
    d._sharp = np.flatnonzero(sharp)
    return d._sharp


def on_sharp(d, q, tol):
    """Is the local point q on a sharp edge of mesh data d (within tol)?"""
    es = sharp_edges(d)
    if not len(es):
        return False
    A, B = d.V[d.E[es, 0]].astype(float), d.V[d.E[es, 1]].astype(float)
    ab = B - A
    t = np.clip(((q - A) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-30), 0.0, 1.0)
    return float(np.linalg.norm(A + t[:, None] * ab - q, axis=1).min()) <= tol


def mesh_rings(d):
    """The sharp (bending by 30 degrees or more) or open edge loops of mesh
    data d that lie on a circle: [(centre, unit normal, radius, vertex
    indices)], local. Once per mesh."""
    got = getattr(d, "_rings", None)
    if got is not None:
        return got
    out = []
    d._rings = out
    es = sharp_edges(d)
    if not len(es) or len(es) > FEATURE_MAX:
        return out
    size = d._faces_geometry()[6]
    for vs, closed in snapping._edge_loops(d.E, es):
        if not closed or len(vs) < RING_MIN:
            continue
        ci = snapping.circle_frame(d.V[vs].astype(float), RING_TOL)
        if ci is not None and ci[2] > 1e-9 * size:
            out.append((ci[0], ci[1], ci[2], vs))
    return out


def _smooth_region(d, poly):
    """The faces reached from face `poly` across edges bending by less than
    30 degrees (a cylinder's side, a sphere), up to SMOOTH_MAX: indices."""
    N, _, _, lp, order, start, _ = d._faces_geometry()
    seen = {int(poly)}
    front = [int(poly)]
    while front and len(seen) < SMOOTH_MAX:
        nxt = []
        for p in front:
            for e in d.le[d.ls[p]:d.ls[p] + d.lt[p]]:
                for q in lp[order[start[e]:start[e + 1]]]:
                    q = int(q)
                    if q not in seen and float(N[q] @ N[p]) >= SMOOTH_COS:
                        seen.add(q)
                        nxt.append(q)
        front = nxt
    return np.fromiter(seen, np.int64)


def mesh_face_kind(d, poly):
    """What the face `poly` of mesh data d is part of (local): ("PLANE", c, n,
    seg), ("AXIS", point, axis, radius), ("SPHERE", centre, radius) or None."""
    cache = getattr(d, "_kinds", None)
    if cache is None:
        cache = d._kinds = {}
    key = int(poly)
    if key in cache:
        return cache[key]
    reg = d.flat_region(poly)
    if reg is None:
        cache[key] = None
        return None
    if reg["flat"]:
        out = ("PLANE", reg["c"], reg["n"], reg["seg"])
        cache[key] = out
        return out
    polys = _smooth_region(d, poly)
    N, A = d._faces_geometry()[:2]
    size = d._faces_geometry()[6]
    vs = np.unique(d.lv[snapping._ranges(d.ls[polys], d.lt[polys])])
    X = d.V[vs].astype(float)
    out = None
    S = (N[polys] * A[polys][:, None]).T @ N[polys]
    lam, vec = np.linalg.eigh(S)
    if lam[2] > 0 and lam[0] <= 1e-4 * lam[2] and lam[1] >= 1e-3 * lam[2]:
        a = vec[:, 0]                       # the normals all lie across it: a cylinder's axis
        x, y = _perp(a)
        Q = np.column_stack([X @ x, X @ y])
        from .step_export import _circle_2d
        c2, r = _circle_2d(Q)
        if r > 1e-9 * size and float(np.abs(np.linalg.norm(Q - c2, axis=1) - r).max()) <= 2e-3 * r:
            out = ("AXIS", c2[0] * x + c2[1] * y + float(X.mean(axis=0) @ a) * a, a, r)
    if out is None and lam[0] > 1e-2 * lam[2]:
        from .step_export import fit_sphere
        sp = fit_sphere(X, 2e-3 * max(float(np.ptp(X, axis=0).max()), 1e-12), size)
        if sp is not None:
            out = ("SPHERE", sp[0], sp[1])
    for p in polys:
        cache[int(p)] = out
    cache[key] = out
    return out


# --------------------------------------------------------------------------
# NURBS faces (local, cached per surface revision)
# --------------------------------------------------------------------------

def brep_face_kind(obj, i):
    """What face i of a NURBS object is (local): ("PLANE", c, n, None),
    ("AXIS", point, axis, radius or None), ("SPHERE", centre, radius) or
    None, from its surface sampled on a grid."""
    rev = obj.get(store.PROP_REV, 0)
    hit = _FACE_KIND.get(obj.session_uid)
    if hit is None or hit[0] != rev:
        hit = _FACE_KIND[obj.session_uid] = (rev, {})
    if i in hit[1]:
        return hit[1][i]
    out = None
    try:
        from . import step_export as sx
        from .surfeval import Surf
        f = store.get_brep(obj).faces()[i]
        lo, hi = (np.array(x, float) for x in f.bounding_box())
        size = float(np.linalg.norm(hi - lo)) or 1.0
        E = Surf(f.surface())
        nu, nv = (min(4 * (n - 1) + 1, 33) for n in (E.nu, E.nv))
        G = E.grid(np.linspace(*E.u_domain, max(nu, 9)), np.linspace(*E.v_domain, max(nv, 9)), 0)[0, 0]
        gsize = float(np.linalg.norm(np.ptp(G.reshape(-1, 3), axis=0))) or size
        tol = FIT_TOL * gsize
        pl = sx.fit_plane(G, tol)
        if pl is not None:
            out = ("PLANE", pl[0], pl[1], None)
        else:
            cy = sx.fit_cylinder(G, tol, gsize)
            if cy is not None:
                out = ("AXIS", cy[0], cy[1], cy[2])
            else:
                co = sx.fit_cone(G, tol, gsize)
                if co is not None:
                    out = ("AXIS", co[0], co[1], None)
                else:
                    sp = sx.fit_sphere(G, tol, gsize)
                    if sp is not None:
                        out = ("SPHERE", sp[0], sp[1])
                    else:
                        rv = sx.revolution_axis(G, tol, gsize)
                        if rv is not None:
                            out = ("AXIS", rv[0], rv[1], None)
    except Exception as exc:
        print("NURBS align: face kind skipped", getattr(obj, "name", "?"), i, exc)
        out = None
    hit[1][i] = out
    return out


def forget():
    """Drop the cached NURBS face kinds (a file was loaded)."""
    _FACE_KIND.clear()


# --------------------------------------------------------------------------
# the element under the mouse
# --------------------------------------------------------------------------

def _scale_of(R):
    """The uniform scale of a 3x3 matrix, or None when it is not a similarity."""
    s = abs(float(np.linalg.det(R))) ** (1.0 / 3.0)
    if s <= 0 or np.abs(R.T @ R - s * s * np.eye(3)).max() > 1e-6 * s * s:
        return None
    return s


def _world_kind(m, kind):
    """A local face kind (mesh_face_kind / brep_face_kind) in world space."""
    if kind is None:
        return None
    R, t = m.R, m.t
    s = _scale_of(R)
    if kind[0] == "PLANE":
        n = _unit(m.Ri.T @ kind[2])
        seg = None if kind[3] is None else np.asarray(kind[3], float).reshape(-1, 3) @ R.T + t
        return ("PLANE", R @ kind[1] + t, n, None if seg is None else seg.reshape(-1, 2, 3))
    if kind[0] == "AXIS":
        r = None if (kind[3] is None or s is None) else kind[3] * s
        return ("AXIS", R @ kind[1] + t, _unit(R @ kind[2]), r)
    if kind[0] == "SPHERE":
        return ("SPHERE", R @ kind[1] + t, None if s is None else kind[2] * s)
    return None


def _target_rings(T, view, mouse):
    """[(ring dict, rim px, centre px, depth, is_mesh)] of the rings within
    RING_PX of the mouse, on the rim or at the centre."""
    out = []
    for g in T.rings:
        out.append((g, False))
    for m in T._meshes:
        if not (m.near(view, mouse) and m.ready()):
            continue
        rings = m.__dict__.get("_world_rings")
        if rings is None:
            s = _scale_of(m.R)
            rings = []
            for c, n, r, vs in mesh_rings(m.d):
                W = m.world(np.append(vs, vs[0]))
                rings.append({"c": m.R @ c + m.t, "n": _unit(m.Ri.T @ n),
                              "r": float(np.linalg.norm(W[0] - (m.R @ c + m.t))) if s is None else r * s,
                              "W": W, "obj": m.obj.name, "ref": None})
            m._world_rings = rings
        out += [(g, True) for g in rings]
    hits = []
    for g, is_mesh in out:
        xy, z, ok = view.project(g["c"])
        dc = float(np.linalg.norm(xy[0] - mouse)) if ok[0] else np.inf
        dr, zr = _poly_px(view, g["W"], mouse)
        if min(dc, dr) <= RING_PX:
            hits.append((g, dr, dc, zr if dr <= dc else float(z[0]), is_mesh))
    return hits


def _on_ring(g, P):
    """Is world point P on ring g's circle (or arc)?"""
    v = np.asarray(P, float) - g["c"]
    h = float(v @ g["n"])
    rho = float(np.linalg.norm(v - h * g["n"]))
    return abs(h) <= 1e-5 * g["r"] + 1e-9 and abs(rho - g["r"]) <= RING_TOL * 10 * g["r"] + 1e-9


def _is_full(g):
    W = g["W"]
    return float(np.linalg.norm(W[0] - W[-1])) <= 1e-6 * g["r"]


def _ring_element(g, view_dir, P=None, scale=1.0):
    n = _towards(g["n"], view_dir)
    full = _is_full(g)
    glyph = [(circle_points(g["c"], n, g["r"]) if full else g["W"], "main"),
             (np.array([g["c"], g["c"] + GLYPH_PX * scale * n]), "axis")]
    return {"type": "RING", "label": "Ring" if full else "Arc", "o": g["c"], "d": n, "r": g["r"], "pin": True,
            "hint": None, "P": g["c"] if P is None else np.asarray(P, float), "obj": g["obj"],
            "src": (g["obj"], g["ref"], "CENTER") if g.get("ref") else None, "glyph": glyph}


def _point_element(P, label, obj, src=None, hint=None):
    return {"type": "POINT", "label": label, "o": np.asarray(P, float), "d": None, "r": None, "pin": True,
            "hint": None if hint is None else _unit(hint), "P": np.asarray(P, float), "obj": obj, "src": src,
            "glyph": []}


def _face_element(T, view, mouse, hit, scale, centre=None):
    """The element of the face under the mouse (hit = Targets.ray_hit), or,
    with centre (a face centre snap), the pinned one there."""
    m, (P, poly, _) = hit
    o, vd = view.ray(np.asarray(mouse, float))
    if isinstance(m, snapping.SurfaceTarget):
        try:
            i = m.face_index(poly, P)
        except Exception:
            i = -1
        kind = brep_face_kind(m.obj, i) if i >= 0 else None
        if kind is not None and kind[0] == "PLANE":
            f = m.flat(poly, P)                 # the exact outline
            kind = ("PLANE", kind[1], kind[2], None if f is None else (f["seg"] - m.t) @ np.linalg.inv(m.R).T)
        src = (m.obj.name, f"F{i}", "NEAREST") if i >= 0 else None
    else:
        kind, src = mesh_face_kind(m.d, poly), None
    w = _world_kind(m, kind)
    name = m.obj.name
    if w is not None and w[0] == "PLANE":
        n = _towards(w[2], vd)
        seg = w[3]
        hint = snapping.face_axis(n, seg) if seg is not None else None
        at = P if centre is None else np.asarray(centre, float)
        glyph = ([(s, "main") for s in seg] if seg is not None else []) + \
            [(np.array([at, at + GLYPH_PX * scale * n]), "axis")]
        return {"type": "PLANE", "label": "Face" if centre is None else "Center", "o": np.asarray(at, float),
                "d": n, "r": None, "pin": centre is not None, "hint": hint, "P": np.asarray(at, float),
                "obj": name, "src": src, "glyph": glyph}
    if w is not None and w[0] == "AXIS":
        a = _towards(w[2], vd) if abs(float(w[2] @ vd)) > 1e-3 else w[2]
        foot = _ray_line_foot(o, vd, w[1], a) if centre is None else np.asarray(centre, float)
        foot = w[1] + float((foot - w[1]) @ a) * a
        ext = _axis_extent(m, poly, w[1], a, foot)
        glyph = [(np.array(ext), "dash")]
        if w[3] is not None:
            glyph.append((circle_points(foot, a, w[3]), "main"))
        pinned = centre is not None
        return {"type": "AXIS", "label": "Axis Mid" if pinned else "Axis", "o": foot, "d": a, "r": w[3],
                "pin": pinned, "hint": None, "P": foot if pinned else np.asarray(P, float), "obj": name,
                "src": src, "glyph": glyph}
    if w is not None and w[0] == "SPHERE":
        el = _point_element(w[1], "Sphere", name, src)
        if w[2] is not None:
            el["glyph"] = [(circle_points(w[1], -vd, w[2]), "main")]
        el["P"] = np.asarray(P, float)
        return el
    # any other face: its tangent plane at the point
    n = _towards(m.Ri.T @ m.d._faces_geometry()[0][poly], vd)
    return {"type": "PLANE", "label": "On Face", "o": np.asarray(P, float), "d": n, "r": None, "pin": False,
            "hint": None, "P": np.asarray(P, float), "obj": name, "src": src,
            "glyph": [(np.array([P, P + GLYPH_PX * scale * n]), "axis")]}


def _axis_extent(m, poly, c, a, foot):
    """The axis segment over the face (a little past it): [start, end]."""
    try:
        if isinstance(m, snapping.SurfaceTarget):
            i = m.face_index(poly, foot)
            lo, hi = (np.array(x, float) for x in store.get_brep(m.obj).faces()[i].bounding_box())
            X = np.array([[(lo, hi)[p][0], (lo, hi)[q][1], (lo, hi)[k][2]]
                          for p in (0, 1) for q in (0, 1) for k in (0, 1)]) @ m.R.T + m.t
        else:
            polys = _smooth_region(m.d, poly)
            X = m.world(np.unique(m.d.lv[snapping._ranges(m.d.ls[polys], m.d.lt[polys])]))
        h = (X - c) @ a
        h0, h1 = float(h.min()), float(h.max())
    except Exception:
        h0 = h1 = float((foot - c) @ a)
    pad = 0.15 * (h1 - h0) or 1.0
    return [c + (h0 - pad) * a, c + (h1 + pad) * a]


def element(T, view, mouse):
    """The snap element under the mouse (see the module docstring), or None."""
    mouse = np.asarray(mouse, float)
    o, vd = view.ray(mouse)
    h = T.pick_info(view, mouse)
    rings = _target_rings(T, view, mouse)
    hit = None
    hit_done = False

    def face_hit():
        nonlocal hit, hit_done
        if not hit_done:
            hit, hit_done = T.ray_hit(view, mouse), True
        return hit

    at = h["P"] if h is not None else None
    scale = px_size(view, at if at is not None else (o + 10.0 * vd))
    kind = h["kind"] if h is not None else None
    hpx = np.inf
    if h is not None:
        xy, _, ok = view.project(h["P"])
        hpx = float(np.linalg.norm(xy[0] - mouse)) if ok[0] else np.inf
    # the best ring: the rim (as near as the edge it is) or its centre
    best = None
    for g, dr, dc, z, is_mesh in rings:
        score = min(dr - 0.5, dc - snapping.BONUS["center"])
        if best is None or score < best[0] - snapping.TIE_PX or (score <= best[0] + snapping.TIE_PX and z < best[2]):
            best = (score, g, z, is_mesh, dc <= dr)
    if best is not None:
        _, g, _, is_mesh, at_centre = best
        strong = kind in ("end", "vertex", "cv", "first")
        lies_on = h is not None and (is_mesh or kind in ("edge", "curve")) and _on_ring(g, h["P"])
        centre_hit = kind == "center" and float(np.linalg.norm(h["P"] - g["c"])) <= 1e-6 * g["r"] + 1e-12
        h_score = hpx - snapping.BONUS.get(kind, 0.0) if h is not None else np.inf
        if lies_on or centre_hit or h is None or kind == "face" or (not strong and best[0] <= h_score) \
                or (strong and best[0] < h_score - snapping.TIE_PX):
            P = g["c"] if at_centre else (h["P"] if lies_on else g["W"][0])
            if not at_centre and not lies_on:
                q = _nearest_on_poly(view, g["W"], mouse)
                P = q if q is not None else P
            return _ring_element(g, vd, P, scale)
    if h is not None and h["src"] is None and kind in ("vertex", "mid", "center", "edge"):
        # a facet's corner, edge or centre on a curved mesh (a cylinder's side, a
        # sphere) is noise: the surface's axis or centre is meant, unless the
        # point is on one of its sharp edges
        fh = face_hit()
        if fh is not None and type(fh[0]) is snapping.MeshTarget and fh[0].obj.name == h["obj"]:
            m = fh[0]
            k = mesh_face_kind(m.d, fh[1][1])
            if k is not None and k[0] in ("AXIS", "SPHERE"):
                q = m.Ri @ np.asarray(h["P"], float) + m.ti
                if not on_sharp(m.d, q, 1e-6 * m.d._faces_geometry()[6]):
                    return _face_element(T, view, mouse, fh, scale)
    if h is None or kind == "face":
        fh = face_hit()
        if fh is not None:
            return _face_element(T, view, mouse, fh, scale)
        return None if h is None else _point_element(h["P"], "Point", h["obj"], h["src"])
    label = snapping.LABELS.get(kind, kind.title())
    if kind in ("end", "vertex", "cv", "first"):
        el = _point_element(h["P"], label, h["obj"], h["src"])
        return el
    if kind == "mid":
        hint = _edge_dir(T, h)
        el = _point_element(h["P"], label, h["obj"], h["src"], hint)
        if hint is not None:
            el["glyph"] = [(np.array([h["P"] - GLYPH_PX * 0.5 * scale * el["hint"],
                                      h["P"] + GLYPH_PX * 0.5 * scale * el["hint"]]), "axis")]
        return el
    if kind == "center":
        fh = face_hit()
        if fh is not None and fh[0].obj.name == h["obj"]:
            el = _face_element(T, view, mouse, fh, scale, centre=h["P"])
            if el["type"] in ("PLANE", "AXIS") and _lies_in(el, h["P"]):
                return el
        return _point_element(h["P"], label, h["obj"], h["src"])
    if kind in ("edge", "curve"):
        W, d = _edge_line(T, h)
        if d is not None:
            return {"type": "LINE", "label": "Edge", "o": np.asarray(h["P"], float), "d": d, "r": None,
                    "pin": False, "hint": None, "P": np.asarray(h["P"], float), "obj": h["obj"], "src": h["src"],
                    "glyph": [(W, "main")] if W is not None else []}
        el = _point_element(h["P"], "On Curve" if kind == "curve" else "Edge", h["obj"], h["src"],
                            _edge_dir(T, h))
        if W is not None:
            el["glyph"] = [(W, "main")]
        return el
    return _point_element(h["P"], label, h["obj"], h["src"])


def _lies_in(el, P):
    v = np.asarray(P, float) - el["o"]
    tol = 1e-6 * (float(np.linalg.norm(v)) + (el["r"] or 0.0) + 1e-3)
    if el["type"] == "PLANE":
        return abs(float(v @ el["d"])) <= tol + 1e-9
    return float(np.linalg.norm(v - float(v @ el["d"]) * el["d"])) <= tol + 1e-9


def _nearest_on_poly(view, W, mouse):
    a, _, oka = view.project(W[:-1])
    b, _, okb = view.project(W[1:])
    dv = b - a
    s = np.clip(((mouse - a) * dv).sum(axis=1) / np.maximum((dv * dv).sum(axis=1), 1e-18), 0.0, 1.0)
    dist = np.linalg.norm(a + s[:, None] * dv - mouse, axis=1)
    dist[~(oka & okb)] = np.inf
    j = int(np.argmin(dist))
    if not np.isfinite(dist[j]):
        return None
    return W[j] + s[j] * (W[j + 1] - W[j])


def _edge_line(T, h):
    """(world polyline, unit direction) of the straight edge the snap h is
    on, (polyline, None) when it is curved, (None, None) when unknown. A mesh
    edge's direction is its frame's first row."""
    src, tan = h["src"], h["tan"]
    if src is None:
        if tan is not None and np.asarray(tan).shape == (2, 3) and np.linalg.norm(tan[0]) > 0.5:
            return None, _unit(tan[0])
        return None, None
    k = T.line_index(src[0], src[1])
    if k is None:
        return None, None
    W, _ = T.line_info(k)
    a, b = W[0], W[-1]
    L = float(np.linalg.norm(b - a))
    if L <= 1e-12:
        return W, None
    u = (b - a) / L
    v = W - a
    off = np.linalg.norm(v - np.outer(v @ u, u), axis=1)
    return (W, u) if float(off.max()) <= 1e-7 * L else (W, None)


def _edge_dir(T, h):
    """The direction of the edge or curve at the snap h (None when unknown)."""
    W, d = _edge_line(T, h)
    if d is not None:
        return d
    if W is not None and len(W) >= 2:
        j = int(np.argmin(np.linalg.norm(W - h["P"], axis=1)))
        a, b = W[max(j - 1, 0)], W[min(j + 1, len(W) - 1)]
        if np.linalg.norm(b - a) > 0:
            return _unit(b - a)
    tan = h["tan"]
    if tan is not None and np.asarray(tan).shape == (2, 3):
        return _unit(tan[0]) if np.linalg.norm(tan[0]) > 0.5 else None
    return None


def overlay(el, color=(0.25, 1.0, 0.45, 1.0)):
    """sketch_overlay["elements"] entries for element el."""
    return [] if el is None else [(pl, style, color) for pl, style in el["glyph"]]
