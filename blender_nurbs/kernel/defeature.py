"""Remove Fillets: find the fillets of a solid and take them away, the faces
round them extended until they meet in the sharp edges again.

Kernel side, pure functions on a Brep (no bpy): the worker's remove_fillets
job runs them, and the add-on calls find() in Blender to light the fillets
up. Needs the kernel's blender_ext.fillet_info / defeature_faces /
delete_fillet_face (kernel_build/ext/SmPyBlenderExt.cpp); has_kernel() says
whether the loaded kernel has them.

Which faces are fillets (find): the kernel's own fillet recognition
(SmFace::IdentifyFillet: rail edges tangent to the faces either side, a
circular, blend or variable section), then three rules the whole-body
upstream remove_fillet lacks, found on real parts:
  - a "linear" section is no fillet (the round end of a slot came back as one);
  - a face whose normals turn more than FULL_ROUND_DEG is a full round (a
    slot's half-round end, a rib's rounded top): its sides never meet;
  - a face whose rail edges run along other fillets is part of a free-form
    skin (a spaceship STEP had 572 of 576 faces 'fillets'), not a fillet.
Corner patches (curved faces all of whose neighbours are fillets: where three
rounded edges meet) go with their fillets. A chain is the fillets and corners
joined by edges; it is removed as a whole, because the kernel can't rebuild
part of a rounded corner (a partial pick hung it for minutes).

Removing (remove): all chains in one defeature call (the kernel's
SmFeatureExecutive, the faces round each chain extended until they close it);
if that fails, chain by chain, each also tried by DeleteFilletFace (the two
catch different cases: a concave L edge only the second, a hole's rim only
the first). Fillets of a chain that were not picked are put back after it
is gone: the new sharp edge filleted again with the radius and section the
kernel read off them."""

import math

import numpy as np

FULL_ROUND_DEG = 170.0          # a face turning more than this is a full round, not a fillet
TANGENCY_DEG = 5.0              # rails within this of tangent count (the kernel's own default)
XS_NAMES = {0: "circular", 1: "circular", 2: "linear", 3: "G1 blend", 4: "G2 blend", 5: "G3 blend",
            6: "blend"}


def has_kernel(sm):
    ext = getattr(sm, "blender_ext", None)
    return ext is not None and all(hasattr(ext, n) for n in ("fillet_info", "defeature_faces",
                                                             "delete_fillet_face"))


def _size(b):
    lo, hi = b.bounding_box()
    return float(np.linalg.norm(np.array(hi, float) - np.array(lo, float))) or 1.0


def _index(faces, f):
    for k, g in enumerate(faces):
        if g == f:
            return k
    return -1


def _neighbours(faces):
    out = []
    for i, f in enumerate(faces):
        nb = set()
        for e in f.edges():
            for g in e.faces():
                j = _index(faces, g)
                if j >= 0 and j != i:
                    nb.add(j)
        out.append(nb)
    return out


def _normal(sm, f, p):
    _, uvs, _ = sm.surface_closest_point(f.surface(), tuple(float(x) for x in p))
    uv = uvs[0]
    uv = (uv.x, uv.y) if hasattr(uv, "x") else (float(uv[0]), float(uv[1]))
    n = np.array(f.outward_normal_at_uv(*uv), float)
    return n / (np.linalg.norm(n) or 1.0)


def _turn(sm, f, d):
    """How far (degrees) the fillet turns across its section: the angle
    between its normals at a point of one rail and the nearest point of the
    other, at a few stations (180 for a slot's half-round end, 90 for a
    square edge's round; a ring round a hole turns 90 too, though its normals
    span every direction along it)."""
    a_rails, b_rails = list(d["rails_min"]), list(d["rails_max"])
    if not a_rails or not b_rails:
        return 0.0
    best = 0.0
    for e in a_rails:
        t0, t1 = e.parameter_range()
        for s in (0.25, 0.5, 0.75):
            p = e.point_at(t0 + s * (t1 - t0))
            q = min((x.closest_point(p) for x in b_rails), key=lambda r: r[2])[0]
            try:
                c = float(_normal(sm, f, p) @ _normal(sm, f, q))
            except Exception:
                continue
            best = max(best, math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return best


def _probe(d):
    """Where the sharp edge comes back once the fillet is gone, roughly: the
    middle of a chord across it (rail to rail), and the rails' direction."""
    a_rails, b_rails = list(d["rails_min"]), list(d["rails_max"])
    if not a_rails or not b_rails:
        return None, None
    e = max(a_rails, key=lambda x: x.length())
    t0, t1 = e.parameter_range()
    t = 0.5 * (t0 + t1)
    p = np.array(e.point_at(t), float)
    q = np.array(min((x.closest_point(tuple(p)) for x in b_rails), key=lambda r: r[2])[0], float)
    tan = np.array(e.tangent_at(t), float)
    tan /= np.linalg.norm(tan) or 1.0
    return [float(x) for x in 0.5 * (p + q)], [float(x) for x in tan]


def _point(f):
    ip = f.interior_point()
    return [float(ip.x), float(ip.y), float(ip.z)] if hasattr(ip, "x") else [float(x) for x in ip]


def _inner_point(f, n=9):
    """A point well inside face f (the middle of a fillet's strip): of a grid
    of (u, v) the face calls inside, the one farthest from its edges. The
    kernel's interior_point() can lie a few microns off a rail: once the
    fillet is gone the new sharp corner passes that close to it."""
    try:
        (u0, v0), (u1, v1) = f.uv_domain()
        surf = f.surface()
        E = []
        for e in f.edges():
            t0, t1 = e.parameter_range()
            E += [e.point_at(float(t)) for t in np.linspace(t0, t1, 25)]
        E = np.array(E, float) if E else None
        best, best_d = None, -1.0
        for u in np.linspace(u0, u1, n + 2)[1:-1]:
            for v in np.linspace(v0, v1, n + 2)[1:-1]:
                if f.classify_uv(float(u), float(v)) != "inside":
                    continue
                q = np.array(surf.evaluate(float(u), float(v)), float)
                d = float(np.min(np.linalg.norm(E - q, axis=1))) if E is not None else 0.0
                if d > best_d:
                    best, best_d = q, d
        if best is not None:
            return [float(x) for x in best]
    except Exception:
        pass
    return _point(f)


def find(sm, b, exclude=()):
    """The fillets of Brep b: {"fillets": [{face, point, radius, radius_min,
    radius_max, kind, chain}], "corners": [{face, point, chain}],
    "chains": [[face indices]], "skipped": [{face, point, why}]}. Face indices
    are b.faces() order; points lie inside the faces (b's space). exclude:
    face indices that are no fillet whatever they look like (Face CVs: a
    sculpted face tangent to fillets all round reads as one, and then its
    fillets -- their rails on it -- did not)."""
    ext = sm.blender_ext
    faces = list(b.faces())
    tol = max(1e-6, 1e-4 * _size(b))
    info = []
    for f in faces:
        try:
            info.append(ext.fillet_info(f, tol, TANGENCY_DEG))
        except Exception:
            info.append({"is_fillet": False})
    exclude = set(exclude)
    cand = {i for i, d in enumerate(info) if d.get("is_fillet") and d.get("cross_section") != 2} - exclude
    skipped = [{"face": i, "why": "a straight (linear) section"} for i, d in enumerate(info)
               if d.get("is_fillet") and d.get("cross_section") == 2]
    nb = _neighbours(faces)
    fillets = set()
    for i in sorted(cand):
        d = info[i]
        rail_faces = set()
        for e in list(d["rails_min"]) + list(d["rails_max"]):
            for g in e.faces():
                j = _index(faces, g)
                if j >= 0 and j != i:
                    rail_faces.add(j)
        if rail_faces & cand:
            continue                      # rails on fillets: a free-form skin, or a corner patch (below)
        turn = _turn(sm, faces[i], d)
        if turn > FULL_ROUND_DEG:
            skipped.append({"face": i, "why": f"a full round ({turn:.0f}°): its sides never meet"})
            continue
        fillets.add(i)
    corners = set()
    for i, f in enumerate(faces):
        if i in fillets or i in exclude or len(nb[i]) < 3 or not nb[i] <= fillets:
            continue
        try:
            if f.surface().is_planar(tol):
                continue
        except Exception:
            pass
        corners.add(i)
    members = fillets | corners
    chains, chain_of = [], {}
    for i in sorted(members):
        if i in chain_of:
            continue
        stack, chain = [i], []
        chain_of[i] = len(chains)
        while stack:
            a = stack.pop()
            chain.append(a)
            for c in nb[a]:
                if c in members and c not in chain_of:
                    chain_of[c] = len(chains)
                    stack.append(c)
        chains.append(sorted(chain))
    out_f = []
    for i in sorted(fillets):
        d = info[i]
        r, rmin, rmax = float(d["radius"]), float(d["radius_min"]), float(d["radius_max"])
        if r <= 0.0 and rmax > 0.0:
            r = rmax
        try:
            probe, along = _probe(d)
        except Exception:
            probe, along = None, None
        out_f.append({"face": i, "point": _point(faces[i]), "probe": probe, "along": along,
                      "radius": r, "radius_min": rmin if rmin > 0 else r,
                      "radius_max": rmax if rmax > 0 else r, "kind": XS_NAMES.get(int(d["cross_section"]), "blend"),
                      "cross_section": int(d["cross_section"]), "solver": int(d["solver"]),
                      "chain": chain_of[i]})
    out_c = [{"face": i, "point": _point(faces[i]), "chain": chain_of[i]} for i in sorted(corners)]
    for s in skipped:
        s["point"] = _point(faces[s["face"]])
    return {"fillets": out_f, "corners": out_c, "chains": chains, "skipped": skipped}


def _face_near(b, p, tol):
    faces = list(b.faces())
    best, bd = None, None
    for f in faces:
        d = f.closest_point(tuple(float(x) for x in p))[1]
        if bd is None or d < bd:
            best, bd = f, d
    return best if bd is not None and bd <= tol else None


def _solid_ok(before, after):
    if after is None or after.face_count() == 0:
        return False
    if before.is_manifold_solid() and not after.is_manifold_solid():
        return False
    return after.face_count() < before.face_count()


def _reaches(rmax, size):
    """Extension distances to try for the faces round a chain: the kernel's
    own first, then longer ones. Its own is too short where the faces meet
    at a shallow angle (a chamfer's cone and a flat face, 150 degrees apart):
    the body comes back unchanged."""
    return [0.0] + [d for d in (3.0 * rmax, 10.0 * rmax, 0.05 * size) if d > 0.0]


def _defeatured(sm, b, groups_of, reaches, tol):
    """b with the face groups removed (a new Brep) by the kernel's defeature,
    trying each extension distance in turn; (None, why) when none works.
    groups_of(copy) -> the groups as the copy's faces. A result must have
    lost every face of the groups (a point inside each no longer on the body:
    the kernel can come back with some of them left); one that kept a face
    the extensions added (a bevel cone's strip down to the floor, with the
    kernel's own distance) gives way to a later distance that leaves the
    faces as they were but for the removed ones."""
    why = "the faces round it could not be extended to meet"
    best = None
    for d in reaches:
        c = b.copy()
        groups = groups_of(c)
        if groups is None:
            return None, "its faces were not found again"
        gone = [_inner_point(f) for g in groups for f in g]
        want = b.face_count() - len(gone)
        try:
            sm.blender_ext.defeature_faces(c, groups, d)
        except Exception as exc:
            why = str(exc).split(" at ")[0]
            continue
        if not _solid_ok(b, c) or any(_face_near(c, p, tol) is not None for p in gone):
            continue
        if c.face_count() <= want:
            return c, ""
        if best is None or c.face_count() < best.face_count():
            best = c
    return (best, "") if best is not None else (None, why)


def _remove_chain(sm, b, pts, tol, reaches=(0.0,)):
    """b with the chain of faces at pts removed (a new Brep), or (None, why)."""
    ext = sm.blender_ext

    def groups_of(c):
        faces = [_face_near(c, p, tol) for p in pts]
        return None if any(f is None for f in faces) else [faces]
    c, why = _defeatured(sm, b, groups_of, reaches, tol)
    if c is not None:
        return c, ""
    if len(pts) == 1:
        c = b.copy()
        f = _face_near(c, pts[0], tol)
        try:
            ext.delete_fillet_face(c, f)
            if _solid_ok(b, c):
                return c, ""
        except Exception:
            pass
    return None, why


def _refillet(sm, b, kept, tol):
    """Fillet again the sharp edges where the kept fillets were: each the edge
    nearest the middle of the fillet's chord that runs the rails' way, with
    its radius and section. Returns (brep, restored count) or (None, why)."""
    c = b.copy()                                  # the edges must be the copy's own
    edges, radii, blends = [], [], []
    for k in kept:
        if k.get("probe") is None:
            return None, "a kept fillet has no rails to find its edge by"
        p = tuple(float(x) for x in k["probe"])
        along = np.array(k["along"], float)
        best, bd = None, None
        for e in c.edges():
            if len(e.faces()) != 2:
                continue
            q, t, d = e.closest_point(p)
            tan = np.array(e.tangent_at(t), float)
            if abs(float(tan @ along)) < 0.9 * (np.linalg.norm(tan) or 1.0):
                continue                          # crosses the fillet's way (an edge at its end)
            if bd is None or d < bd:
                best, bd = e, d
        if best is None or bd > 2.0 * max(k["radius_max"], tol):
            return None, "the edge under a kept fillet was not found"
        if any(best == e for e in edges):
            continue
        if k["solver"] == 2:
            return None, "a kept fillet has a varying radius"
        edges.append(best)
        radii.append(float(k["radius"]))
        blends.append(k["cross_section"] in (3, 4, 5))
    if not edges:
        return b, 0
    X = sm.FilletXSect
    try:
        if any(blends):
            conts = {k["cross_section"] for k in kept if k["cross_section"] in (3, 4, 5)}
            if not all(blends) or len(conts) > 1:
                return None, "kept fillets of different kinds (blends and rounds)"
            sm.fillet_edges_per_edge(c, edges, radii, [X.BLEND] * len(edges), continuity=conts.pop() - 2)
        else:
            sm.fillet_edges_per_edge(c, edges, radii, [X.CIRCULAR] * len(edges))
    except Exception as exc:
        return None, "filleting the kept edges again failed: " + str(exc).split(" at ")[0]
    if b.is_manifold_solid() and not c.is_manifold_solid():
        return None, "filleting the kept edges again opened the body"
    return c, len(edges)


def remove(sm, b, picked=None, max_radius=0.0, keep_others=True):
    """Remove fillets from Brep b (not changed). picked: points (b's space) on
    the fillets to remove -- each takes its whole chain away, the chain's
    other fillets put back when keep_others; None: every fillet (no larger
    than max_radius when that is > 0). Returns (new Brep or None, report)."""
    found = find(sm, b)
    size = _size(b)
    tol = max(1e-6, 1e-6 * size)
    fil = {f["face"]: f for f in found["fillets"]}
    faces = list(b.faces())
    want = set()
    if picked is None:
        for i, f in fil.items():
            if max_radius <= 0.0 or f["radius_max"] <= max_radius * (1.0 + 1e-9):
                want.add(i)
    else:
        corner_faces = {c["face"] for c in found["corners"]}
        for p in picked:
            f = _face_near(b, p, max(tol, 1e-4 * size))
            i = _index(faces, f) if f is not None else -1
            if i in fil:
                want.add(i)
            elif i in corner_faces:             # a corner patch: the fillets round it
                ch = next(c["chain"] for c in found["corners"] if c["face"] == i)
                want |= {j for j, x in fil.items() if x["chain"] == ch}
    report = {"found": len(fil), "removed": 0, "restored": 0, "chains": 0, "failed": [], "warnings": []}
    if not want:
        report["failed"].append({"point": None, "why": "no removable fillet" + (" picked" if picked else "")})
        return None, report
    chains = sorted({fil[i]["chain"] for i in want})
    plan = []                                   # (points of the chain's faces, kept fillets, n removed, reaches)
    for ch in chains:
        members = found["chains"][ch]
        kept = [fil[i] for i in members if i in fil and i not in want] if keep_others else []
        rmax = max((fil[i]["radius_max"] for i in members if i in fil), default=0.0)
        plan.append(([_point(faces[i]) for i in members], kept, sum(1 for i in members if i in want),
                     _reaches(rmax, size)))
    # all chains at once: one rebuild
    cur, _ = _defeatured(sm, b, lambda c: [[list(c.faces())[i] for i in found["chains"][ch]] for ch in chains],
                         _reaches(max((f["radius_max"] for f in fil.values()), default=0.0), size)[:2],
                         max(tol, 1e-5 * size))
    if cur is not None:
        report["chains"] = len(chains)
        report["removed"] = sum(n for _, _, n, _ in plan)
    if cur is None:                             # chain by chain
        cur = b
        for pts, kept, n, reaches in plan:
            nxt, why = _remove_chain(sm, cur, pts, max(tol, 1e-5 * size), reaches)
            if nxt is None:
                report["failed"].append({"point": pts[0], "why": why})
                continue
            cur = nxt
            report["chains"] += 1
            report["removed"] += n
        if cur is b:
            return None, report
    kept_all = [k for (pts, kept, n, _), ch in zip(plan, chains) for k in kept
                if not any(f["point"] == pts[0] for f in report["failed"])]
    if kept_all:
        res, n = _refillet(sm, cur, kept_all, max(tol, 1e-5 * size))
        if res is None:
            report["warnings"].append(f"{len(kept_all)} other fillet{'s' if len(kept_all) != 1 else ''} of the "
                                      f"same chain went too ({n})")
            report["removed"] += len(kept_all)
        else:
            cur = res
            report["restored"] = n
            if any(k["cross_section"] not in (0, 1) for k in kept_all):
                report["warnings"].append("the other fillets of the chain were put back as the kernel's blends "
                                          "of the same size: a soft fillet's shape can differ")
    return cur, report
