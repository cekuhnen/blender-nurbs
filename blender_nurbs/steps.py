"""History stack steps: in-place operations replayed on top of an object's
base geometry (its own feature, or a snapshot). Each applier takes the
current brep text (object local space) and returns the new text."""

import json

from mathutils import Matrix, Vector

from . import geom, jobs, store


def _world(obj):
    return geom.world_matrix(obj)


def untrimmed_cutters(cutters):
    """Cutting surfaces as their UNTRIMMED surface. Used when cutting with the
    trimmed one fails -- e.g. two surfaces trimmed by each other: the cutter's
    hole lies exactly on the cut, so the trimmed cutter barely touches it."""
    out = {}
    for k, src in cutters.items():
        if isinstance(src, tuple) or store.PROP_UNTRIM not in src:
            out[k] = src
        else:
            out[k] = (src[store.PROP_UNTRIM], jobs.xform(src))
    return out


def cutter_curve_json(kc, projection, view, up=None):
    """Worker data of a cutting curve for Trim / Split. CURVE: a planar curve
    projects along its own normal, others along the view; a curve lying on the
    surface cuts along itself ("auto"). SURFACE: always along itself.
    up: the curve object's Z axis (world): the sketch normal points its way,
    so a one-sided trim (reach FRONT) goes where the blue axis points."""
    cj = jobs.curve_json(kc)
    if projection == "CURVE":
        d = geom.cutter_direction(kc, view)
        if up is not None and Vector(up).length > 1e-9 and abs(d.dot(Vector(up).normalized())) > 0.1:
            d = d if d.dot(Vector(up)) > 0 else -d
        cj["dir"] = list(d)
    cj["pull"] = True if projection == "SURFACE" else ("auto" if projection == "CURVE" else False)
    return cj


def curve_up(obj):
    """World Z axis of a curve object (its sketch normal); None for surfaces."""
    if getattr(obj, "type", None) != 'CURVE':
        return None
    return (obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()


def extended_notes(extended, curve_inputs, surfaces, target_name):
    """Report lines for what a trim / split job extended. curve_inputs: the
    (object, ref) of each curve in job order; surfaces: {job key: object}."""
    out = []
    for e in extended or []:
        if "curve" in e:
            o, r = curve_inputs[e["curve"]]
            ends = "both ends" if len(e["ends"]) > 1 else "its " + e["ends"][0]
            out.append(f"{o.name}{'' if r == 'S0' else ' ' + r}: extended at {ends} to cut across {target_name}")
        else:
            o = surfaces[e["surface"]]
            name = o.name if hasattr(o, "name") else e["surface"]
            out.append(f"{name}: extended at " + ", ".join(e["edges"]) + f" to cut across {target_name}")
    return out


def run_trim(target, cutters, outs, **kw):
    try:
        return jobs.run("trim", {"target": target, **cutters}, outs, cutter_breps=list(cutters), **kw)
    except jobs.JobError:
        alt = untrimmed_cutters(cutters)
        if not cutters or all(alt[k] is cutters[k] for k in cutters):
            raise
        return jobs.run("trim", {"target": target, **alt}, outs, cutter_breps=list(alt), **kw)


def trim_regions(target, cutters, **kw):
    """The regions a trim would cut `target` into, for marking before it is
    committed: [{"text": local brep text, "point": world point inside it,
    "side": its side of the cutters}], and the index of the clicked one."""
    try:
        info, breps = jobs.run("split", {"target": target, **cutters}, [], cutter_breps=list(cutters), **kw)
    except jobs.JobError:
        alt = untrimmed_cutters(cutters)
        if not cutters or all(alt[k] is cutters[k] for k in cutters):
            raise
        info, breps = jobs.run("split", {"target": target, **alt}, [], cutter_breps=list(alt), **kw)
    regions = [{"text": breps[f"region{i}"], "point": info["points"][i], "side": info["sides"][i]}
               for i in range(info["regions"])]
    return regions, info["picked"], info


def region_mesh(text, M):
    """World triangles (n, 3, 3) and outline segments (m, 2, 3) of a region,
    coarse: only for the trim preview overlay."""
    import numpy as np
    b = store.brep_from_text(text)
    lo, hi = (np.array(v) for v in b.bounding_box())
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    pb, _ = b.tessellate(chord_height_tolerance=0.002 * diag, curve_angle_tolerance_deg=10.0,
                         surface_angle_tolerance_deg=15.0, allow_partial=True, return_failures=True)
    a = pb.to_numpy_arrays()
    P = np.asarray(a["points"], float)
    f = np.asarray(a["faces"])
    tris, i = [], 0
    while i < len(f):
        c = int(f[i])
        poly = f[i + 1:i + 1 + c]
        tris += [(poly[0], poly[k], poly[k + 1]) for k in range(1, c - 1)]
        i += 1 + c
    T = np.array(tris, int).reshape(-1, 3)
    Mw = np.array(M, float)
    W = P @ Mw[:3, :3].T + Mw[:3, 3]
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    border = uniq[cnt == 1]
    return W[T].astype(np.float32), W[border].astype(np.float32)


def apply_trim(obj, text, inputs, p):
    from . import features
    M = _world(obj)
    ref = M @ Vector(p["ref_local"])
    nrm = (M.to_3x3() @ Vector(p.get("normal_local", (0, 0, 1)))).normalized()
    view = (M.to_3x3() @ Vector(p.get("view_local", (0, 0, -1)))).normalized()
    curves, cutters = [], {}
    for src, ref_s in inputs:
        if ref_s == "SURF":
            cutter = src
            if features.depends_on(src, obj):
                # two surfaces trimmed by each other: the cutter's trimmed shape is
                # itself cut by this surface, so whichever of the two rebuilds first
                # would cut with the other's OLD shape (and the pair stops following
                # its inputs). Cut with its untrimmed shape as it is now.
                try:
                    bt = features.base_text(src)
                except Exception:
                    bt = None
                if bt:
                    cutter = (bt, jobs.xform(src))
            cutters[f"c{len(cutters)}"] = cutter
        else:
            curves.append(cutter_curve_json(features.input_curve(src, ref_s), p.get("projection", "CURVE"), view,
                                            curve_up(src)))
    proj = p.get("projection", "CURVE")
    if proj in ("NORMAL", "CURVE", "SURFACE"):
        d = nrm
    elif proj == "VIEW":
        d = view
    else:
        d = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[proj])
    action = p.get("action", "KEEP")
    _, breps = run_trim((text, jobs.xform(obj)), cutters, [action],
                        curves=curves, direction=list(d.normalized()),
                        ref_point=list(ref), action=action, scope=p.get("scope", "THROUGH"), side=p.get("side"),
                        extend=bool(p.get("extend", False)), reach=p.get("reach", "BOTH"),
                        through=bool(p.get("through", False)),
                        marks=[[list(M @ Vector(mk[0]))] + list(mk[1:]) for mk in p.get("marks_local") or []])
    return breps[action]


PROP_RIMS = "nproto_rims"


def _mids_key(mids_local):
    return json.dumps([[round(float(x), 6) for x in m] for m in mids_local])


def store_rims(obj, mids_local, info, Mi=None, text=None):
    """Keep what an edge fillet step made, keyed by the step's edges, for its
    panel: its faces (a point inside each, local, and the key of the surface
    under it from the step's result `text`: highlighted while the step is
    open), what became of its rims (closed edges), and whether every face of
    the fillet is a rim (then the Class-A settings have nothing to do)."""
    try:
        d = json.loads(obj.get(PROP_RIMS, "{}"))
    except ValueError:
        d = {}
    key = _mids_key(mids_local)
    if Mi is None:
        Mi = _world(obj).inverted()
    entry = {}
    if info.get("rings"):
        entry.update(rings=info["rings"], only=bool(info.get("rims_only")))
    if info.get("new_faces"):
        entry["faces"] = [[float(x) for x in (Mi @ Vector(p))] for p in info["new_faces"]]
        if text:
            # the surface under each face too: a later step that trims the face (a
            # chamfer cutting its end off) keeps its surface, not that one point
            try:
                from . import subsel
                entry["keys"] = subsel.face_keys(store.brep_from_text(text), entry["faces"])
            except Exception as exc:
                print("NURBS fillet face keys:", exc)
    if entry:
        d[key] = entry
    else:
        d.pop(key, None)
    if d:
        obj[PROP_RIMS] = json.dumps(d)
    elif PROP_RIMS in obj:
        del obj[PROP_RIMS]


def rims_of(obj, mids_local):
    """{"rings": [...], "only": bool, "faces": [points]} for the fillet step
    on these edges (keys present when they apply), or None."""
    try:
        return json.loads(obj.get(PROP_RIMS, "{}")).get(_mids_key(mids_local))
    except (ValueError, TypeError):
        return None


def apply_edge_fillet(obj, text, inputs, p):
    M = _world(obj)
    mids = [list(M @ Vector(m)) for m in p["mids_local"]]
    info, breps = jobs.run("edge_fillet", {"target": (text, jobs.xform(obj))}, ["result"], edge_mids=mids,
                        shape=p.get("shape", "CIRCULAR"), radius=p["radius"], end_radius=p.get("end_radius"),
                        chord=p["radius"] if p.get("chord") else None,
                        continuity=int(p.get("continuity", 1)), thumbweight=p.get("thumbweight", 1.0), loose=True,
                        shrink_planes=bool(p.get("shrink_planes")),
                        **class_a_params(p))
    if "class_a" in info:
        obj["nproto_class_a"] = json.dumps(info["class_a"])
    elif "nproto_class_a" in obj:
        del obj["nproto_class_a"]
    store_rims(obj, p["mids_local"], info, M.inverted(), breps["result"])
    return breps["result"]


def class_a_params(p):
    """Worker parameters of the Class-A fillet rebuild (history params / operator)."""
    if not p.get("class_a"):
        return {}
    return {"class_a": True, "ca_tolerance": float(p.get("ca_tolerance", 1e-4)),
            "ca_angle": float(p.get("ca_angle", 0.1)), "ca_degree": int(p.get("ca_degree", 7)),
            "ca_continuity": int(p.get("ca_continuity", 2))}


def apply_fillet_trim(obj, text, inputs, p):
    """Trim this surface back to the rail of its fillet with the partner surface."""
    from . import features
    partner = inputs[0][0]
    role = p.get("role", "a")   # replay the original fillet (same a / b order) and take our side
    me = (text, jobs.xform(obj))
    # the fillet is computed from the BASE surfaces (trims may have shrunk them);
    # only the trim is applied to the current geometry
    me_base = (features.base_text(obj) or text, jobs.xform(obj))
    pa_base = (features.base_text(partner) or partner[store.PROP_BREP], jobs.xform(partner))
    if role == "a":
        objs = {"a": me, "b": partner, "a_base": me_base, "b_base": pa_base}
    else:
        objs = {"a": partner, "b": me, "a_base": pa_base, "b_base": me_base}
    trim = p.get("trim", True)
    res, breps = jobs.run("fillet", objs, ["fillet", "a", "b"], radius_a=p["radius_a"], radius_b=p["radius_b"],
                          xsect=p["xsect"], chord=abs(p["radius_a"]) if p.get("chord") else None,
                          blend=p.get("blend", 0), tolerance=1e-4, trim_bases=trim, ends=p.get("ends", "NOTCH"))
    if role not in breps and not trim:
        return text                 # Extend without trimming: this side was not the one grown
    if role not in breps:
        raise RuntimeError(res.get("trim_errors", {}).get(role, "fillet trim failed"))
    return breps[role]


IDENT = {"loc": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "angle": 0.0, "scale": [1.0, 1.0, 1.0]}


def natural_surface(brep):
    """The surface of a one-face brep whose face is the whole surface (no
    trim), else None."""
    if brep.face_count() != 1:
        return None
    f = brep.faces()[0]
    s = f.surface()
    try:
        (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
        (fu0, fv0), (fu1, fv1) = f.uv_domain()
        if len(f.boundary_loops()) != 1:
            return None
    except Exception:
        return None
    span = max(u1 - u0, v1 - v0) or 1.0
    if max(abs(fu0 - u0), abs(fu1 - u1), abs(fv0 - v0), abs(fv1 - v1)) > 1e-9 * span:
        return None
    return s


def flip_text(text):
    """The brep turned over (every face's normal reversed), as text. An
    untrimmed surface: u reversed here (exact; U0 / U1 swap). Trimmed faces /
    joined sheets: in the worker (the kernel turns them over, trims kept)."""
    b = store.brep_from_text(text)
    s = natural_surface(b)
    if s is not None:
        from .square import reverse_u
        _, ext = store._k()
        return store.brep_to_text(ext.brep_from_surface(reverse_u(s)))
    _, breps = jobs.run("flip", {"target": (text, IDENT)}, ["result"])
    return breps["result"]


def apply_flip(obj, text, inputs, p):
    return flip_text(text)


REBUILD_BASE = [None]     # the geometry before a rebuild's steps (features sets it): Untrim Region's reference


def untrim_region_text(text, untrimmed_text, xf, point=None, points=None):
    """(result text, info): text with trimmed-away areas given back, from the
    untrimmed surface: the one at the cut edge nearest `point`, or the ones
    nearest each of `points` (world points inside removed areas)."""
    kw = {"points": [[float(v) for v in q] for q in points]} if points else {"point": [float(v) for v in point]}
    info, breps = jobs.run("untrim_region", {"target": (text, xf), "untrimmed": (untrimmed_text, xf)}, ["result"],
                           **kw)
    return breps["result"], info


def untrim_preview(text, untrimmed_text, xf):
    """The areas trimmed away from text: [(local brep text, world point inside)]."""
    info, breps = jobs.run("untrim_preview", {"target": (text, xf), "untrimmed": (untrimmed_text, xf)}, [])
    return [(breps[f"region{i}"], info["points"][i]) for i in range(info["regions"])]


def untrim_reference(obj):
    """The surface Untrim Region cuts the trimmed-away areas from: the one a
    split piece was cut out of, else the object's base (before its steps)."""
    from . import features
    whole = features.untrimmed_base(obj)
    if whole is not None:
        return whole[0]
    return REBUILD_BASE[0] or features.base_text(obj)


def apply_untrim_region(obj, text, inputs, p):
    base = untrim_reference(obj)
    M = _world(obj)
    if p.get("points_local"):
        out, _ = untrim_region_text(text, base, jobs.xform(obj), points=[M @ Vector(q) for q in p["points_local"]])
    else:
        out, _ = untrim_region_text(text, base, jobs.xform(obj), M @ Vector(p["point_local"]))
    return out


def apply_untrim(obj, text, inputs, p):
    """A split piece back to the whole surface it was cut out of."""
    from . import features
    whole = features.untrimmed_base(obj)
    return whole[0] if whole is not None else text


def apply_push_pull(obj, text, inputs, p):
    """Push / Pull on rebuild: the faces at the stored points moved again."""
    M = _world(obj)
    pts = [list(M @ Vector(q)) for q in p["faces_local"]]
    N = M.to_3x3().inverted().transposed()
    nrm = [list((N @ Vector(n)).normalized()) for n in p["normals_local"]] if p.get("normals_local") else None
    _, breps = jobs.run("push_pull", {"target": (text, jobs.xform(obj))}, ["result"], face_pts=pts,
                        face_normals=nrm, face_flat=p.get("flat"), distance=float(p["distance"]), into="target")
    return breps["result"]


def apply_replace_face(obj, text, inputs, p):
    """Replace Face on rebuild: the faces at the stored points replaced by the
    surface (the step's input) as it is now."""
    M = _world(obj)
    pts = [list(M @ Vector(q)) for q in p["faces_local"]]
    N = M.to_3x3().inverted().transposed()
    nrm = [list((N @ Vector(n)).normalized()) for n in p["normals_local"]] if p.get("normals_local") else None
    _, breps = jobs.run("replace_face", {"target": (text, jobs.xform(obj)), "surface": inputs[0][0]}, ["result"],
                        face_pts=pts, face_normals=nrm, face_flat=p.get("flat"), into="target")
    return breps["result"]


APPLY = {"replace_face": apply_replace_face, "push_pull": apply_push_pull, "trim": apply_trim, "edge_fillet": apply_edge_fillet, "fillet_trim": apply_fillet_trim,
         "flip_normal": apply_flip, "untrim_region": apply_untrim_region, "untrim": apply_untrim}
LABEL = {
    "trim": lambda p: "Trim · " + {"KEEP": "Keep", "DELETE": "Discard"}.get(p.get("action"), p.get("action", "")),
    "edge_fillet": lambda p: f"Edge {p.get('shape', 'CIRCULAR').title()} · "
                             + ("chord" if p.get("chord") else "r") + f" {p.get('radius', 0):.3g}"
                             + (" · Class-A" if p.get("class_a") else ""),
    "fillet_trim": lambda p: ("Fillet trim" if p.get("trim", True) else "Fillet extend"),
    "flip_normal": lambda p: "Flip Normal",
    "untrim_region": lambda p: "Untrim Region",
    "untrim": lambda p: "Untrim",
    "replace_face": lambda p: "Replace Face" + (f" · {len(p.get('faces_local', []))} faces"
                                                if len(p.get("faces_local", [])) > 1 else ""),
    "push_pull": lambda p: ("Pull" if float(p.get("distance", 0)) > 0 else "Push")
                           + f" · {abs(float(p.get('distance', 0))):.3g}"
                           + (f" · {len(p.get('faces_local', []))} faces" if len(p.get("faces_local", [])) > 1 else ""),
}


# --------------------------------------------------------------------------
# Split: every region of the split body is its own object with a "split
# region" feature (inputs: the body, then what cut it); the body is kept as a
# hidden input, like Join's. The region is found again by its centroid.
# --------------------------------------------------------------------------

_split_cache = {}      # inputs + settings -> region texts (one split serves all its regions)


def _split_job(inputs, p):
    """Worker objects and settings of a split of inputs[0] by inputs[1:]."""
    from . import features
    tgt = inputs[0][0]
    others = {f"c{i}": o for i, (o, r) in enumerate(inputs[1:]) if r == "SURF"}
    view = Vector(p.get("view_dir", (0.0, 0.0, -1.0)))
    if view.length < 1e-9:
        view = Vector((0.0, 0.0, -1.0))
    curves = [cutter_curve_json(features.input_curve(o, r), p.get("projection", "CURVE"), view, curve_up(o))
              for o, r in inputs[1:] if r != "SURF"]
    proj = p.get("projection", "CURVE")
    d = view if proj in ("VIEW", "CURVE", "SURFACE") else Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[proj])
    kw = {"curves": curves, "cutter_breps": list(others), "direction": list(d.normalized()),
          "extend": bool(p.get("extend", True)), "reach": p.get("reach", "BOTH"),
          "through": bool(p.get("through", False))}
    return {"target": tgt, **others}, kw, others


def split_info(inputs, p, notes=None):
    """The split of inputs[0] by the other inputs (surfaces "SURF", curves
    "S<i>" and surface edges), in the worker's region order: {"texts": local
    brep texts, "points": a world point inside each, "areas", "sides"}. With
    p["extend"] (the default) a cutter whose cut stops inside the body is
    extended until it reaches across; `notes` (a list) receives what was."""
    import hashlib
    objs, kw, others = _split_job(inputs, p)
    tgt = inputs[0][0]
    h = hashlib.sha1()
    for o in [tgt] + list(others.values()):
        h.update(o.get(store.PROP_BREP, "").encode())
        h.update(json.dumps(jobs.xform(o), sort_keys=True).encode())
    h.update(json.dumps([kw["curves"], kw["direction"], kw["extend"], kw["reach"], kw["through"]],
                        sort_keys=True).encode())
    key = h.hexdigest()
    if key in _split_cache:
        out, said = _split_cache[key]
        if notes is not None:
            notes += said
        return out
    info, breps = jobs.run("split", objs, [], **kw)
    said = extended_notes(info.get("extended"), [(o, r) for o, r in inputs[1:] if r != "SURF"], others, tgt.name)
    if notes is not None:
        notes += said
    n = info["regions"]
    out = {"texts": [breps[f"region{i}"] for i in range(n)], "points": info.get("points", []),
           "areas": info.get("areas", []), "sides": info.get("sides", [])}
    if len(_split_cache) > 16:
        _split_cache.clear()
    _split_cache[key] = (out, said)
    return out


def split_regions(inputs, p, notes=None):
    """Brep texts of the regions of inputs[0] split by the other inputs, largest first."""
    out = split_info(inputs, p, notes)
    areas = out["areas"]
    order = sorted(range(len(out["texts"])), key=lambda i: -(areas[i] if i < len(areas) else 0))
    return [out["texts"][i] for i in order]


def merge_regions(inputs, p, keep):
    """One brep text (local) of the regions of a split marked keep[i] (in
    split_info order), joined where they meet: no seam along the cuts
    between them (the Trim job with marks does the merging)."""
    out = split_info(inputs, p)
    picked = [i for i, k in enumerate(keep) if k]
    if len(picked) == 1:
        return out["texts"][picked[0]]
    objs, kw, _ = _split_job(inputs, p)
    marks = [[pt, bool(k), sd] for pt, k, sd in zip(out["points"], keep, out["sides"])]
    _, breps = jobs.run("trim", objs, ["KEEP"], ref_point=out["points"][picked[0]], action="KEEP", scope="PICKED",
                        marks=marks, strict=False, **kw)
    return breps["KEEP"]


def region_point(text):
    """A region's area centroid (local space): what tells it from the other
    regions of its split. (A point inside it is not: when a cutter moves, the
    new cut can run right through that point.)"""
    b = store.brep_from_text(text)
    acc, tot = [0.0, 0.0, 0.0], 0.0
    for f in b.faces():
        a = f.area()
        c = f.centroid()
        c = (c.x, c.y, c.z) if hasattr(c, "x") else tuple(c)
        acc = [acc[i] + a * float(c[i]) for i in range(3)]
        tot += a
    return [x / (tot or 1.0) for x in acc]


def _text_dist(text, at):
    c = region_point(text)
    return sum((c[i] - float(at[i])) ** 2 for i in range(3))


# --------------------------------------------------------------------------
# Features that run in the worker (their result is brep text)
# --------------------------------------------------------------------------

def _register():
    from . import features

    @features.builder("join")
    def build_join(inputs, p):
        objs = {f"s{i}": o for i, (o, _) in enumerate(inputs)}
        _, breps = jobs.run("join", objs, ["joined"], order=list(objs), into="_world_",
                            tolerance=p.get("tolerance", 1e-3))
        return ("brep_text", breps["joined"])

    @features.builder("split_region")
    def build_split_region(inputs, p):
        if p.get("keep_at"):        # several regions joined into one piece (Split's preview)
            out = split_info(inputs, p)
            cs = [region_point(t) for t in out["texts"]]

            def near(c, ats):
                return min(sum((c[i] - float(a[i])) ** 2 for i in range(3)) for a in ats) if ats else float("inf")
            keep = [near(c, p["keep_at"]) <= near(c, p.get("drop_at") or []) for c in cs]
            return ("brep_text", merge_regions(inputs, p, keep))
        regions = split_regions(inputs, p)
        at = p.get("at")
        best = min(regions, key=lambda t: _text_dist(t, at)) if at is not None else regions[0]
        return ("brep_text", best)

    @features.builder("boolean")
    def build_boolean(inputs, p):
        objs = {f"s{i}": o for i, (o, _) in enumerate(inputs)}
        _, breps = jobs.run("boolean", objs, ["result"], order=list(objs), into="_world_",
                            operation=p.get("bool_op", "UNION"), names={k: o.name for k, o in objs.items()})
        return ("brep_text", breps["result"])

    @features.builder("boundary_fill")
    def build_boundary_fill(inputs, p):
        objs = {f"s{i}": o for i, (o, _) in enumerate(inputs)}
        _, breps = jobs.run("boundary_fill", objs, ["result"], order=list(objs), into="_world_",
                            names={k: o.name for k, o in objs.items()}, timeout=600.0)
        return ("brep_text", breps["result"])

    @features.builder("surface_fillet")
    def build_surface_fillet(inputs, p):
        (a, _), (b, _) = inputs[0], inputs[1]
        objs = {"a": (features.base_text(a) or a[store.PROP_BREP], jobs.xform(a)),
                "b": (features.base_text(b) or b[store.PROP_BREP], jobs.xform(b))}
        _, breps = jobs.run("fillet", objs, ["fillet"], radius_a=p["radius_a"],
                            radius_b=p["radius_b"], xsect=p["xsect"],
                            chord=abs(p["radius_a"]) if p.get("chord") else None, blend=p.get("blend", 0),
                            tolerance=1e-4, trim_bases=False, ends=p.get("ends", "NOTCH"))
        return ("brep_text", breps["fillet"])


_register()
