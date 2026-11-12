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


def run_trim(target, cutters, outs, **kw):
    try:
        return jobs.run("trim", {"target": target, **cutters}, outs, cutter_breps=list(cutters), **kw)
    except jobs.JobError:
        alt = untrimmed_cutters(cutters)
        if not cutters or all(alt[k] is cutters[k] for k in cutters):
            raise
        return jobs.run("trim", {"target": target, **alt}, outs, cutter_breps=list(alt), **kw)


def apply_trim(obj, text, inputs, p):
    from . import features
    M = _world(obj)
    ref = M @ Vector(p["ref_local"])
    nrm = (M.to_3x3() @ Vector(p.get("normal_local", (0, 0, 1)))).normalized()
    view = (M.to_3x3() @ Vector(p.get("view_local", (0, 0, -1)))).normalized()
    curves, cutters = [], {}
    for src, ref_s in inputs:
        if ref_s == "SURF":
            cutters[f"c{len(cutters)}"] = src
        else:
            kc = features.input_curve(src, ref_s)
            cj = jobs.curve_json(kc)
            if p.get("projection", "CURVE") == "CURVE":
                cj["dir"] = list(geom.cutter_direction(kc, view))
            curves.append(cj)
    proj = p.get("projection", "CURVE")
    if proj in ("NORMAL", "CURVE"):
        d = nrm
    elif proj == "VIEW":
        d = view
    else:
        d = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[proj])
    action = p.get("action", "KEEP")
    _, breps = run_trim((text, jobs.xform(obj)), cutters, [action],
                        curves=curves, direction=list(d.normalized()),
                        ref_point=list(ref), action=action, scope=p.get("scope", "THROUGH"), side=p.get("side"))
    return breps[action]


PROP_RIMS = "nproto_rims"


def _mids_key(mids_local):
    return json.dumps([[round(float(x), 6) for x in m] for m in mids_local])


def store_rims(obj, mids_local, info, Mi=None):
    """Keep what an edge fillet step made, keyed by the step's edges, for its
    panel: its faces (a point inside each, local: highlighted while the step
    is open), what became of its rims (closed edges), and whether every face
    of the fillet is a rim (then the Class-A settings have nothing to do)."""
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
                        **class_a_params(p))
    if "class_a" in info:
        obj["nproto_class_a"] = json.dumps(info["class_a"])
    elif "nproto_class_a" in obj:
        del obj["nproto_class_a"]
    store_rims(obj, p["mids_local"], info, M.inverted())
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
    res, breps = jobs.run("fillet", objs, ["fillet", "a", "b"], radius_a=p["radius_a"], radius_b=p["radius_b"],
                          xsect=p["xsect"], chord=abs(p["radius_a"]) if p.get("chord") else None,
                          tolerance=1e-4, trim_bases=True)
    if role not in breps:
        raise RuntimeError(res.get("trim_errors", {}).get(role, "fillet trim failed"))
    return breps[role]


def blend_between(a, b, fillet_text, cont):
    """G1 / G2 blend surface (world space) from a's edge to b's edge where the
    circular fillet `fillet_text` (world space) meets them -- a and b already
    trimmed back to those rails. cont: 1 or 2."""
    import numpy as np
    from . import features
    fb = store.brep_from_text(fillet_text)
    rails = []
    for e in fb.edges():
        t0, t1 = e.parameter_range()
        rails.append(np.array([e.point_at(t0 + (t1 - t0) * k / 40.0) for k in range(41)], float))

    def match(obj):
        M = geom.world_matrix(obj)
        best = None
        for i, e in enumerate(store.get_brep(obj).edges()):
            t0, t1 = e.parameter_range()
            P = np.array([tuple(M @ Vector(e.point_at(t0 + (t1 - t0) * k / 6.0))) for k in range(7)], float)
            for R in rails:
                d = max(float(np.min(np.linalg.norm(R - q, axis=1))) for q in P)
                if best is None or d < best[0]:
                    best = (d, i)
        if best is None:
            raise RuntimeError(f"{obj.name} has no edge along the fillet")
        return f"E{best[1]}"
    g = f"G{int(cont)}"
    p = {"cont_a": g, "cont_b": g, "tan_a": 1.0, "tan_b": 1.0, "cur_a": 0.0, "cur_b": 0.0,
         "auto_align": True, "flip_b": False, "ruled": False, "closed": False, "samples": 24}
    return features.run("loft", [(a, match(a)), (b, match(b))], p)


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


APPLY = {"trim": apply_trim, "edge_fillet": apply_edge_fillet, "fillet_trim": apply_fillet_trim,
         "flip_normal": apply_flip}
LABEL = {
    "trim": lambda p: "Trim · " + {"KEEP": "Keep", "DELETE": "Discard"}.get(p.get("action"), p.get("action", "")),
    "edge_fillet": lambda p: f"Edge {p.get('shape', 'CIRCULAR').title()} · "
                             + ("chord" if p.get("chord") else "r") + f" {p.get('radius', 0):.3g}"
                             + (" · Class-A" if p.get("class_a") else ""),
    "fillet_trim": lambda p: "Fillet trim",
    "flip_normal": lambda p: "Flip Normal",
}


# --------------------------------------------------------------------------
# Split: every region of the split body is its own object with a "split
# region" feature (inputs: the body, then what cut it); the body is kept as a
# hidden input, like Join's. The region is found again by its centroid.
# --------------------------------------------------------------------------

_split_cache = {}      # inputs + settings -> region texts (one split serves all its regions)


def split_regions(inputs, p):
    """Brep texts of the regions of inputs[0] split by the other inputs
    (surfaces "SURF" and curves "S<i>"), largest first, in the body's local space."""
    import hashlib
    from . import features
    from .ops_trim import _curve_direction
    tgt = inputs[0][0]
    others = {f"c{i}": o for i, (o, r) in enumerate(inputs[1:]) if r == "SURF"}
    view = Vector(p.get("view_dir", (0.0, 0.0, -1.0)))
    if view.length < 1e-9:
        view = Vector((0.0, 0.0, -1.0))
    curves = []
    for o, r in inputs[1:]:
        if r != "SURF":
            kc = features.input_curve(o, r)
            cj = jobs.curve_json(kc)
            if p.get("projection", "CURVE") == "CURVE":
                cj["dir"] = list(_curve_direction(kc, view))
            curves.append(cj)
    proj = p.get("projection", "CURVE")
    d = view if proj in ("VIEW", "CURVE") else Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[proj])
    h = hashlib.sha1()
    for o in [tgt] + list(others.values()):
        h.update(o.get(store.PROP_BREP, "").encode())
        h.update(json.dumps(jobs.xform(o), sort_keys=True).encode())
    h.update(json.dumps([curves, list(d.normalized())], sort_keys=True).encode())
    key = h.hexdigest()
    if key in _split_cache:
        return _split_cache[key]
    info, breps = jobs.run("split", {"target": tgt, **others}, [], curves=curves,
                           cutter_breps=list(others), direction=list(d.normalized()))
    parts = sorted(((int(k[6:]), t) for k, t in breps.items() if k.startswith("region")))
    areas = info.get("areas", [])
    order = sorted(range(len(parts)), key=lambda i: -(areas[i] if i < len(areas) else 0))
    regions = [parts[i][1] for i in order]
    if len(_split_cache) > 16:
        _split_cache.clear()
    _split_cache[key] = regions
    return regions


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
                            chord=abs(p["radius_a"]) if p.get("chord") else None,
                            tolerance=1e-4, trim_bases=False)
        if p.get("blend"):
            return blend_between(a, b, breps["fillet"], p["blend"])
        return ("brep_text", breps["fillet"])


_register()
