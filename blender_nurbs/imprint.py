"""Imprint: Project with Output: Edge (and Intersect with Output: Edge).

Curves, surface edges or cutting surfaces put onto a surface or solid as
edges of its own: its faces are divided where the projection lands, nothing
is removed, and a closed solid stays closed. The new faces take Push / Pull,
Extrude Face, Move Face, Face CVs and the rest; the new edges are targets for
Blend, Fillet, Offset on Surface and Sweep. An open curve that stops inside a
face leaves an edge that ends there (Extend to Edges carries it on to the
face's edges).

Which faces take the edges (scope): NEAREST the faces the projection meets
first seen from the curve, FACES the faces picked on the target (Face mode),
THROUGH every face the projection crosses. Kernel side: kernel/imprint.py.

A history step on the target (op "imprint"), its inputs what was imprinted:
move the sketch and the edges follow."""

from mathutils import Vector

from . import features, geom, jobs, steps, store

OP = "imprint"

PROJECTIONS = [
    ("CURVE", "Sketch Normal", "Each curve along the normal of the plane it was drawn on (a planar curve "
                               "along its own normal); other curves along the work plane normal. A curve "
                               "lying on the surface is imprinted along itself"),
    ("WORKPLANE", "Work Plane", "Along the active work plane's normal (World Z when the World is active)"),
    ("VIEW", "View", "Along the view direction"),
    ("SURFACE", "On Surface", "No projection: along the curve itself, where it lies on or near the surface "
                              "(Output: Edge only)"),
    ("X", "World X", ""), ("Y", "World Y", ""), ("Z", "World Z", ""),
]
SCOPES = [
    ("NEAREST", "Nearest Faces", "The faces the projection meets first, seen from the curve: not the far side "
                                 "of the part, nor walls the projection runs along"),
    ("FACES", "Picked Faces", "Only the faces picked on the target in Face mode"),
    ("THROUGH", "Through", "Every face the projection crosses, all the way through the part"),
]
REACHES = [
    ("BOTH", "Both Ways", "From the curve along the projection on both sides"),
    ("FRONT", "Along Normal", "Only where the projection direction points from the curve"),
    ("BACK", "Against Normal", "Only on the other side of the curve"),
]
AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def plane_direction(scene):
    """World direction along the active work plane (into it: World -Z)."""
    from . import workplanes
    fr = workplanes.active_frame(scene)
    return Vector((0, 0, -1)) if fr is None else -Vector(tuple(map(float, fr[3]))).normalized()


def direction(projection, view, plane):
    """The one direction of a projection mode (Sketch Normal: the fallback
    for curves without a plane of their own)."""
    if projection in AXES:
        return Vector(AXES[projection])
    if projection == "VIEW":
        return Vector(view).normalized()
    return Vector(plane).normalized()


def curve_data(target, inputs, projection, view, plane):
    """Worker data of the curves among inputs ((object, ref) pairs; surfaces
    "SURF" are left out), each with its own direction for Sketch Normal."""
    out = []
    for o, r in inputs:
        if r == "SURF":
            continue
        kc = features.input_curve(o, r)
        if projection == "CURVE":
            sk = steps.sketch_normal(o, kc, target, Vector(view))
            out.append(steps.cutter_curve_json(kc, "CURVE", Vector(plane), steps.curve_up(o), sk))
        else:
            out.append(steps.cutter_curve_json(kc, "SURFACE" if projection == "SURFACE" else projection,
                                               Vector(view)))
    return out


def face_points(obj, idx):
    """World points well inside the faces idx of obj (what FACES scope stores)."""
    b = store.get_brep(obj)
    faces = b.faces()
    M = geom.world_matrix(obj)
    out = []
    for i in idx:
        if 0 <= i < len(faces):
            p = faces[i].interior_point()
            out.append(M @ Vector(tuple(map(float, (p.x, p.y, p.z) if hasattr(p, "x") else p))))
    return out


def run(target, inputs, p, text=None):
    """Imprint target (its brep text `text` when given: a history rebuild) with
    inputs. p: projection, reach, scope, extend and, in target local space,
    dir_local (the direction), view_local, faces_local (FACES scope). Returns
    (worker info, local brep text)."""
    M = geom.world_matrix(target)
    R = M.to_3x3()
    view = (R @ Vector(p.get("view_local", (0.0, 0.0, -1.0)))).normalized()
    d = (R @ Vector(p.get("dir_local", (0.0, 0.0, -1.0)))).normalized()
    proj = p.get("projection", "CURVE")
    objs = {"target": (text, jobs.xform(target)) if text is not None else target}
    keys = []
    for o, r in inputs:
        if r != "SURF":
            continue
        cutter = o
        if features.depends_on(o, target):
            # imprinted by each other (Intersect, Output Edge, both ways): the
            # other one's shape before its own imprint (the same shape)
            try:
                bt = features.base_text(o)
            except Exception:
                bt = None
            if bt:
                cutter = (bt, jobs.xform(o))
        keys.append(f"c{len(keys)}")
        objs[keys[-1]] = cutter
    curves = curve_data(target, inputs, proj, view, d)
    faces = [list(M @ Vector(q)) for q in p.get("faces_local") or []]
    info, breps = jobs.run(OP, objs, ["result"], curves=curves, cutter_breps=keys, direction=list(d),
                           reach=p.get("reach", "BOTH"), extend=bool(p.get("extend", False)),
                           scope=p.get("scope", "NEAREST"), faces=faces, through=False)
    return info, breps["result"]


def local_params(target, projection, view, d, reach, scope, extend, faces_world=()):
    """Step parameters (directions and face points in target local space)."""
    M = geom.world_matrix(target)
    Mi = M.inverted()
    Ri = M.to_3x3().inverted()
    return {"projection": projection, "reach": reach, "scope": scope, "extend": bool(extend),
            "view_local": list((Ri @ Vector(view)).normalized()), "dir_local": list((Ri @ Vector(d)).normalized()),
            "faces_local": [list(Mi @ Vector(q)) for q in faces_world]}


def notes(info, inputs, target):
    """Report lines for what the imprint did beyond its edges."""
    out = steps.extended_notes(info.get("extended"), [(o, r) for o, r in inputs if r != "SURF"],
                               {f"c{i}": o for i, (o, r) in enumerate([x for x in inputs if x[1] == "SURF"])},
                               target.name)
    if not info.get("healed", True):
        out.append(f"{target.name}: some edges also stayed on faces behind the picked ones "
                   "(they could not be taken off again)")
    return out


def apply_imprint(obj, text, inputs, p):
    return run(obj, inputs, p, text)[1]


def label(p):
    scope = p.get("scope", "NEAREST")
    n = len(p.get("faces_local") or [])
    what = {"NEAREST": "Nearest", "THROUGH": "Through"}.get(scope, f"{n} face{'s' if n != 1 else ''}")
    return f"Imprint · {what}"


def _register_step():
    from . import history_ui
    steps.APPLY[OP] = apply_imprint
    steps.LABEL[OP] = label
    history_ui.STEP_ICON[OP] = 'MOD_SHRINKWRAP'
    history_ui.STEP_FIELDS[OP] = ["imprint_scope", "reach", "imprint_extend"]


_register_step()
