"""History stack steps: in-place operations replayed on top of an object's
base geometry (its own feature, or a snapshot). Each applier takes the
current brep text (object local space) and returns the new text."""

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


def apply_edge_fillet(obj, text, inputs, p):
    M = _world(obj)
    mids = [list(M @ Vector(m)) for m in p["mids_local"]]
    info, breps = jobs.run("edge_fillet", {"target": (text, jobs.xform(obj))}, ["result"], edge_mids=mids,
                        shape=p.get("shape", "CIRCULAR"), radius=p["radius"], end_radius=p.get("end_radius"),
                        continuity=int(p.get("continuity", 1)), thumbweight=p.get("thumbweight", 1.0), loose=True,
                        **class_a_params(p))
    if "class_a" in info:
        import json
        obj["nproto_class_a"] = json.dumps(info["class_a"])
    elif "nproto_class_a" in obj:
        del obj["nproto_class_a"]
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
                          xsect=p["xsect"], tolerance=1e-4, trim_bases=True)
    if role not in breps:
        raise RuntimeError(res.get("trim_errors", {}).get(role, "fillet trim failed"))
    return breps[role]


APPLY = {"trim": apply_trim, "edge_fillet": apply_edge_fillet, "fillet_trim": apply_fillet_trim}
LABEL = {
    "trim": lambda p: "Trim · " + {"KEEP": "Keep", "DELETE": "Discard"}.get(p.get("action"), p.get("action", "")),
    "edge_fillet": lambda p: f"Edge {p.get('shape', 'CIRCULAR').title()} · r {p.get('radius', 0):.3g}"
                             + (" · Class-A" if p.get("class_a") else ""),
    "fillet_trim": lambda p: "Fillet trim",
}


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

    @features.builder("boolean")
    def build_boolean(inputs, p):
        objs = {f"s{i}": o for i, (o, _) in enumerate(inputs)}
        _, breps = jobs.run("boolean", objs, ["result"], order=list(objs), into="_world_",
                            operation=p.get("bool_op", "UNION"), names={k: o.name for k, o in objs.items()})
        return ("brep_text", breps["result"])

    @features.builder("surface_fillet")
    def build_surface_fillet(inputs, p):
        (a, _), (b, _) = inputs[0], inputs[1]
        objs = {"a": (features.base_text(a) or a[store.PROP_BREP], jobs.xform(a)),
                "b": (features.base_text(b) or b[store.PROP_BREP], jobs.xform(b))}
        _, breps = jobs.run("fillet", objs, ["fillet"], radius_a=p["radius_a"],
                            radius_b=p["radius_b"], xsect=p["xsect"], tolerance=1e-4, trim_bases=False)
        return ("brep_text", breps["fillet"])


_register()
