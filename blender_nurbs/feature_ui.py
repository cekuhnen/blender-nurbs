"""Live feature editing ("query edit"): every object built with construction
history shows its feature's settings in the sidebar. Changing one rebuilds
the object (and, through history, everything built from it).

Also hosts the Add Primitive operator and the Create menus."""

import json

import bpy
from bpy.props import FloatVectorProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from mathutils import Matrix

from . import features, primitives, store
from .kernel import load as load_kernel
from .ops import AXIS_ITEMS, CONT_ITEMS

_syncing = [False]

DEGREE_ITEMS = [("0", "Exact", "Rational, exact circles / spheres"), ("1", "1", ""), ("2", "2", ""),
                ("3", "3", ""), ("4", "4", ""), ("5", "5", "")]
DIR_ITEMS = [("NORMAL", "Curve Normal", "")] + AXIS_ITEMS + [("VECTOR", "Vector", "")]

# which settings each feature shows (in this order)
FIELDS = {
    "loft": ["closed", "cont_a", "tan_a", "cur_a", "cont_b", "tan_b", "cur_b", "auto_align", "flip_b", "ruled", "samples", "flip_normal"],
    "blend_curve": ["cont_a", "w_a", "c_a", "flip_a", "cont_b", "w_b", "c_b", "flip_b"],
    "extrude": ["distance", "direction", "dir_vec", "tilt", "tilt_azimuth", "draft", "draft_flip", "symmetric"],
    "revolve": ["angle", "axis", "pivot", "seam_angle"],
    "offset": ["distance"],
    "thicken": ["distance", "symmetric"],
    "planar": [],
    "join": ["tolerance"],
    "boolean": ["bool_op"],
    "boundary_fill": [],
    "loft_rails": ["lr_mode", "lr_start", "lw_start", "lr_end", "lw_end", "lr_ra", "lw_ra", "lr_rb", "lw_rb",
                   "lr_degree", "lr_spans_u", "lr_spans_v"],
    "join_curves": ["jc_mode", "tolerance", "jc_degree", "jc_cvs"],
    "sweep": ["samples", "tangent", "flip_tangent", "scale_end", "twist", "keep_height", "flip_normal"],
    "pipe": ["radius", "samples"],
    "patch": ["pt_cvs", "pt_cvs_v", "pt_auto_cvs", "pt_degree", "pt_fair", "pt_surface_fair"],
    "dome": ["dome_layout", "dome_cont", "dome_height", "dome_fullness", "dome_cvs"],
    "boundary": ["bd_exact", "sq_degree_u", "sq_degree_v", "sq_spans_u", "sq_spans_v", "samples", "flip_normal"],
    "square": ["e0", "w0", "e1", "w1", "e2", "w2", "e3", "w3", "sq_match", "sq_degree_u", "sq_degree_v", "sq_spans_u", "sq_spans_v"],
    "iso_curve": ["iso_dir", "param"],
    "surface_fillet": ["radius_a", "radius_b"],
    "prim_line": ["length", "degree", "spans_u"],
    "prim_arc": ["radius", "angle", "degree", "spans_u"],
    "prim_circle": ["radius", "elliptic", "radius2", "degree", "spans_u"],
    "prim_rectangle": ["width", "depth", "corner"],
    "prim_slot": ["length", "width"],
    "prim_polygon": ["radius", "sides"],
    "prim_helix": ["radius", "height", "turns", "taper", "left", "degree", "spans_u"],
    "prim_plane": ["width", "depth", "pl_degree_u", "pl_degree_v", "spans_u", "spans_v"],
    "perp_profile": ["pp_shape", "radius", "radius2", "width", "depth", "degree", "spans_u", "pp_at", "pp_rotation"],
    "prim_box": ["width", "depth", "height", "degree", "spans_u"],
    "prim_cylinder": ["radius", "height", "degree", "spans_u", "spans_v", "caps"],
    "prim_cone": ["radius", "radius2", "height", "degree", "spans_u", "spans_v", "caps"],
    "prim_sphere": ["radius", "degree", "spans_u", "spans_v"],
    "prim_torus": ["radius", "radius2", "degree", "spans_u", "spans_v"],
    "prim_disk": ["radius", "degree", "spans_u"],
}
LABELS = {"tan_a": "Tangent A", "tan_b": "Tangent B", "cur_a": "Curvature A", "cur_b": "Curvature B",
          "w_a": "Weight A", "w_b": "Weight B", "c_a": "Curvature A", "c_b": "Curvature B",
          "spans_u": "Spans U", "spans_v": "Spans V", "radius2": "Radius 2", "width": "Size X", "depth": "Size Y",
          "e0": "Edge 1", "e1": "Edge 2", "e2": "Edge 3", "e3": "Edge 4", "w0": "Weight 1", "w1": "Weight 2",
          "w2": "Weight 3", "w3": "Weight 4", "sq_spans_u": "Spans U (0 auto)", "sq_spans_v": "Spans V (0 auto)"}
PRETTY = {"patch": "Patch", "dome": "Dome", "square": "Square", "sweep": "Sweep", "pipe": "Pipe", "boundary": "Boundary", "iso_curve": "Iso Curve",
          "dup_edge": "Duplicate Edge", "project_curve": "Projected Curve", "intersect": "Intersection",
          "join": "Join", "split_region": "Split", "boolean": "Boolean", "boundary_fill": "Boundary Fill", "join_curves": "Join Curves", "loft_rails": "Loft Rails", "surface_fillet": "Surface Fillet", "loft": "Loft", "blend_curve": "Blend Curve", "extrude": "Extrude", "revolve": "Revolve",
          "offset": "Offset", "thicken": "Thicken", "planar": "Planar", "perp_profile": "Profile"}


def _on_change(self, context):
    if _syncing[0]:
        return
    obj = self.id_data
    h = features.history(obj)
    if h is None:
        return
    params = dict(h["params"])
    for name in FIELDS.get(h["op"], []):
        v = getattr(self, name)
        if name in ("degree", "square_degree", "sq_degree_u", "sq_degree_v", "lr_degree"):
            v = int(v)
        elif hasattr(v, "__len__") and not isinstance(v, str):
            v = [float(x) for x in v]          # vectors: plain lists in the history JSON
        params[name] = v
    raw = json.loads(obj[features.PROP_HIST])
    raw["params"] = params
    obj[features.PROP_HIST] = json.dumps(raw)
    obj["nproto_feature_synced"] = obj[features.PROP_HIST]
    try:
        features.edit_rebuild(obj, context)
    except Exception as exc:
        print(f"NURBS feature edit ({obj.name}): {exc}")


def _F(**kw):
    kw.setdefault("update", _on_change)
    return kw


def _patch_obj(pg):
    """The Patch object a Feature-panel property belongs to (None for an
    operator's copy of the edge list: its redo rebuilds)."""
    obj = pg.id_data
    if _syncing[0] or not isinstance(obj, bpy.types.Object):
        return None
    h = features.history(obj)
    return obj if h is not None and h["op"] == "patch" else None


def _patch_write(obj, raw, context):
    obj[features.PROP_HIST] = json.dumps(raw)
    obj["nproto_feature_synced"] = obj[features.PROP_HIST]
    try:
        features.edit_rebuild(obj, context)
    except Exception as exc:
        print(f"NURBS feature edit ({obj.name}): {exc}")


def cont_all(p, ins):
    """What the Patch's "All edges" row shows: the last value set for all
    edges, else the one every surface edge shares, else MIXED."""
    if p.get("pt_all") in ("AUTO", "G0", "G1", "G2"):
        return p["pt_all"]
    conts, guides = p.get("cont", []), set(p.get("pt_guides", []))
    vals = {conts[k] for k, (o, r) in enumerate(ins) if k < len(conts) and k not in guides
            and features.is_surface_edge(r) and o.type != 'CURVE'}
    return vals.pop() if len(vals) == 1 else "MIXED"


def _sync_patch_edges(obj, p):
    """Edge rows and "All edges" row follow the history (values only)."""
    pg = obj.nproto_feature
    n = len(features.history(obj)["inputs"])
    conts = (list(p.get("cont", [])) + ["G0"] * n)[:n]
    tens = (list(p.get("pt_tension", [])) + [1.0] * n)[:n]
    _syncing[0] = True
    try:
        for it in pg.pt_edges:
            if it.index < n:
                if not it.curve:
                    it.cont = conts[it.index] if conts[it.index] in ("G0", "G1", "G2") else "G0"
                it.tension = float(tens[it.index])
        pg.pt_cont_all = cont_all(p, features.history(obj)["inputs"])
    finally:
        _syncing[0] = False


def _on_patch_edge(self, context):
    """A Patch edge's continuity or tension changed in the panel: into the feature's lists."""
    obj = _patch_obj(self)
    if obj is None:
        return
    raw = json.loads(obj[features.PROP_HIST])
    p = raw["params"]
    n = len(raw["refs"])
    t = (list(p.get("pt_tension", [])) + [1.0] * n)[:n]
    t[self.index] = float(self.tension)
    if all(abs(x - 1.0) < 1e-9 for x in t):
        p.pop("pt_tension", None)
    else:
        p["pt_tension"] = t
    c = (list(p.get("cont", [])) + ["G0"] * n)[:n]
    if not self.curve and c[self.index] != self.cont:
        c[self.index] = self.cont
        p["cont"] = c
        p.pop("pt_all", None)                 # set edge by edge now
    _patch_write(obj, raw, context)
    _syncing[0] = True
    try:
        obj.nproto_feature.pt_cont_all = cont_all(p, features.history(obj)["inputs"])
    finally:
        _syncing[0] = False


def _on_patch_all(self, context):
    """The "All edges" continuity: every edge of the Patch at once (Auto: as a new Patch picks them)."""
    obj = _patch_obj(self)
    if obj is None or self.pt_cont_all == "MIXED":
        return
    from . import patch as _pt
    h = features.history(obj)
    raw = json.loads(obj[features.PROP_HIST])
    p = raw["params"]
    ins = h["inputs"]
    v = self.pt_cont_all
    if v == "AUTO":
        conts = _pt.auto_conts(ins, min(int(p.get("pt_cvs", 24)), 24), int(p.get("pt_degree", 5)),
                               float(p.get("pt_fair", 1.0)), p=p)
    else:
        conts = [v if features.is_surface_edge(r) and o.type != 'CURVE' else "G0" for o, r in ins]
    for k in p.get("pt_guides", []):
        conts[k] = "G0"
    p["cont"], p["pt_all"] = conts, v
    _patch_write(obj, raw, context)
    _sync_patch_edges(obj, p)


def _on_patch_all_tension(self, context):
    """The "All edges" tension: every edge of the Patch at once."""
    obj = _patch_obj(self)
    if obj is None:
        return
    raw = json.loads(obj[features.PROP_HIST])
    p = raw["params"]
    n = len(raw["refs"])
    guides = set(p.get("pt_guides", []))
    t = float(self.pt_tension_all)
    if abs(t - 1.0) < 1e-9:
        p.pop("pt_tension", None)
    else:
        p["pt_tension"] = [1.0 if k in guides else t for k in range(n)]
    _patch_write(obj, raw, context)
    _sync_patch_edges(obj, p)


PATCH_CONT = [("G0", "G0", "Position only"), ("G1", "G1", "Tangent to the neighbouring surface"),
              ("G2", "G2", "Curvature-continuous to the neighbouring surface")]
PATCH_ALL = [("AUTO", "Auto", "G2 where the fill meets the neighbour roughly tangentially (G1 where a corner "
                              "makes curvature impossible), G0 where the neighbour stands across it and on curves"),
             ("G0", "G0", "Position only on every edge"),
             ("G1", "G1", "Tangent to every neighbouring surface (curves: G0)"),
             ("G2", "G2", "Curvature-continuous to every neighbouring surface (curves: G0)"),
             ("MIXED", "Mixed", "Edges set one by one")]
SURFACE_FAIR_DESC = ("Measure the fairness on the surface itself (two more solves): where the layout bends the "
                     "parameter lines -- a corner inside a side, like an elbow's floor-to-wall corners -- the zebra "
                     "stays smooth and the CV rows even. Off: measured in the (u, v) grid only, quicker, as Patch "
                     "worked before")
TENSION_DESC = ("How far this edge's direction carries into the patch: the patch is stiffer (more) or softer "
                "(less) against bending next to it; 1 = as solved")


class NprotoPatchEdge(bpy.types.PropertyGroup):
    """One edge of a Patch (Feature panel, and the Patch operator's redo panel)."""
    index: IntProperty()
    label: StringProperty()
    curve: BoolProperty(description="Only position (G0): a curve, or an edge without a surface")
    cont: EnumProperty(name="Continuity", items=PATCH_CONT, default="G0", update=_on_patch_edge)
    tension: FloatProperty(name="Tension", min=0.25, max=4.0, soft_min=0.5, soft_max=2.0, default=1.0,
                           update=_on_patch_edge, description=TENSION_DESC)


class NprotoFeature(bpy.types.PropertyGroup):
    dome_layout: EnumProperty(name="Layout", items=[("POLAR", "Polar", "Rows from the rim to one top point (a round rim: an exact revolution)"),
                              ("DRAPED", "Draped", "One even four-sided surface over the opening, trimmed "
                                                   "to the rim (no pole: easy to edit)")], **_F())
    dome_cvs: IntProperty(name="CVs", min=8, max=24, description="Draped: CVs in each direction", **_F())
    dome_cont: EnumProperty(name="Continuity", items=[("G1", "G1", "Tangent to the surface"),
                                                      ("G2", "G2", "Tangent, with the surface's curvature carried on")],
                            **_F())
    dome_height: FloatProperty(name="Height", min=0.05, soft_max=3.0,
                               description="1: round (the sphere that carries the surface on over the rim); "
                                           "less: flatter, more: taller", **_F())
    dome_fullness: FloatProperty(name="Fullness", min=0.3, max=2.0,
                                 description="How far the rim's direction carries into the dome", **_F())
    cont_a: EnumProperty(name="Continuity A", items=CONT_ITEMS, **_F())
    cont_b: EnumProperty(name="Continuity B", items=CONT_ITEMS, **_F())
    tan_a: FloatProperty(name="Tangent A", min=0.0, soft_max=3.0, **_F())
    tan_b: FloatProperty(name="Tangent B", min=0.0, soft_max=3.0, **_F())
    cur_a: FloatProperty(name="Curvature A", soft_min=-2.0, soft_max=2.0, **_F())
    cur_b: FloatProperty(name="Curvature B", soft_min=-2.0, soft_max=2.0, **_F())
    w_a: FloatProperty(name="Weight A", min=0.0, soft_max=3.0, **_F())
    w_b: FloatProperty(name="Weight B", min=0.0, soft_max=3.0, **_F())
    c_a: FloatProperty(name="Curvature A", soft_min=-2.0, soft_max=2.0, **_F())
    c_b: FloatProperty(name="Curvature B", soft_min=-2.0, soft_max=2.0, **_F())
    flip_a: BoolProperty(name="Flip A", **_F())
    flip_b: BoolProperty(name="Flip B", **_F())
    auto_align: BoolProperty(name="Auto Align", **_F())
    ruled: BoolProperty(name="Ruled", **_F())
    symmetric: BoolProperty(name="Symmetric", **_F())
    caps: BoolProperty(name="Caps (solid)", **_F())
    elliptic: BoolProperty(name="Ellipse", **_F())
    samples: IntProperty(name="Samples", min=6, max=200, **_F())
    distance: FloatProperty(name="Distance", unit='LENGTH', **_F())
    direction: EnumProperty(name="Direction", items=DIR_ITEMS, **_F())
    dir_vec: FloatVectorProperty(name="Vector", size=3, subtype='XYZ', **_F())
    tilt: FloatProperty(name="Tilt °", soft_min=-89.0, soft_max=89.0, **_F())
    tilt_azimuth: FloatProperty(name="Tilt Around °", soft_min=-180.0, soft_max=180.0, **_F())
    draft: FloatProperty(name="Draft °", soft_min=-45.0, soft_max=45.0, **_F())
    draft_flip: BoolProperty(name="Flip Draft", **_F())
    angle: FloatProperty(name="Angle", min=-360.0, max=360.0, **_F())
    axis: EnumProperty(name="Axis", items=AXIS_ITEMS, **_F())
    seam_angle: FloatProperty(name="Seam Angle", soft_min=-180.0, soft_max=180.0, **_F())
    pivot: EnumProperty(name="Pivot", items=[("OBJECT", "Object Origin", ""), ("WORLD", "World Origin", "")], **_F())
    radius: FloatProperty(name="Radius", min=1e-4, unit='LENGTH', **_F())
    radius2: FloatProperty(name="Radius 2", min=0.0, unit='LENGTH', **_F())
    length: FloatProperty(name="Length", min=1e-4, unit='LENGTH', **_F())
    width: FloatProperty(name="Width", min=1e-4, unit='LENGTH', **_F())
    depth: FloatProperty(name="Depth", min=1e-4, unit='LENGTH', **_F())
    corner: FloatProperty(name="Corner Radius", min=0.0, unit='LENGTH', **_F())
    height: FloatProperty(name="Height", min=1e-4, unit='LENGTH', **_F())
    turns: FloatProperty(name="Turns", min=0.1, soft_max=20.0, **_F())
    left: BoolProperty(name="Left-Handed", **_F(description="The helix turns the other way as it rises"))
    taper: FloatProperty(name="Taper °", min=-80.0, max=80.0, soft_min=-45.0, soft_max=45.0, **_F(
        description="Lean the helix in (positive: narrower on top) or out (negative: wider), like a thread on a cone"))
    sides: IntProperty(name="Sides", min=3, max=64, **_F())
    degree: EnumProperty(name="Degree", items=DEGREE_ITEMS, **_F())
    spans_u: IntProperty(name="Spans U", min=1, max=64, **_F())
    spans_v: IntProperty(name="Spans V", min=1, max=64, **_F())
    pl_degree_u: IntProperty(name="Degree U", min=1, max=7, description="Degree along X", **_F())
    pp_shape: EnumProperty(name="Shape", items=primitives.PROFILE_SHAPES, **_F())
    closed: BoolProperty(name="Closed", description="Loft through the sections and back to the first (3 or more)",
                         **_F())
    pp_at: FloatProperty(name="Position", min=0.0, max=1.0, subtype='FACTOR',
                         description="Where along the curve / edge (0 = its start, 1 = its end)", **_F())
    pp_rotation: FloatProperty(name="Rotation °", soft_min=-180.0, soft_max=180.0,
                               description="Turn the profile about the curve's tangent", **_F())
    pl_degree_v: IntProperty(name="Degree V", min=1, max=7, description="Degree along Y", **_F())
    tangent: BoolProperty(name="Tangent to Surface", **_F())
    flip_tangent: BoolProperty(name="Flip Tangent", **_F())
    flip_normal: BoolProperty(name="Flip Normal", description="Turn the surface over (it faces like the surfaces it is built from; this reverses that)", **_F())
    scale_end: FloatProperty(name="End Scale", min=0.01, soft_max=5.0, **_F())
    twist: FloatProperty(name="Twist", soft_min=-720.0, soft_max=720.0, **_F())
    keep_height: BoolProperty(name="Keep Height (2 rails)", **_F())
    iso_dir: EnumProperty(name="Direction", items=[("U", "U", ""), ("V", "V", "")], **_F())
    param: FloatProperty(name="Position", min=0.0, max=1.0, subtype='FACTOR', **_F())
    tolerance: FloatProperty(name="Tolerance", min=1e-6, soft_max=0.1, precision=5, unit='LENGTH', **_F())
    radius_a: FloatProperty(name="Radius A (signed: side)", unit='LENGTH', **_F())
    radius_b: FloatProperty(name="Radius B (signed: side)", unit='LENGTH', **_F())
    e0: EnumProperty(name="Edge 1", items=[("FREE", "Free", ""), ("G0", "G0", ""), ("G1", "G1", ""), ("G2", "G2", "")], **_F())
    e1: EnumProperty(name="Edge 2", items=[("FREE", "Free", ""), ("G0", "G0", ""), ("G1", "G1", ""), ("G2", "G2", "")], **_F())
    e2: EnumProperty(name="Edge 3", items=[("FREE", "Free", ""), ("G0", "G0", ""), ("G1", "G1", ""), ("G2", "G2", "")], **_F())
    e3: EnumProperty(name="Edge 4", items=[("FREE", "Free", ""), ("G0", "G0", ""), ("G1", "G1", ""), ("G2", "G2", "")], **_F())
    w0: FloatProperty(name="Weight 1", min=0.05, soft_max=3.0, **_F())
    w1: FloatProperty(name="Weight 2", min=0.05, soft_max=3.0, **_F())
    w2: FloatProperty(name="Weight 3", min=0.05, soft_max=3.0, **_F())
    w3: FloatProperty(name="Weight 4", min=0.05, soft_max=3.0, **_F())
    sq_degree_u: EnumProperty(name="Degree U", items=[("0", "Auto", "")] + [(str(d), str(d), "") for d in range(1, 8)], **_F())
    sq_degree_v: EnumProperty(name="Degree V", items=[("0", "Auto", "")] + [(str(d), str(d), "") for d in range(1, 8)], **_F())
    sq_match: BoolProperty(name="Match Neighbours", default=True, **_F())
    bd_exact: BoolProperty(name="Exact Edges", default=True, **_F())
    pt_cvs: IntProperty(name="CVs U", min=4, max=40, **_F())
    pt_cvs_v: IntProperty(name="CVs V", min=4, max=40, **_F())
    pt_auto_cvs: BoolProperty(name="Lightest", description="The fewest CVs (per direction, up to CVs U / V) at which "
                              "every edge still holds its continuity", **_F())
    pt_edges: CollectionProperty(type=NprotoPatchEdge)
    pt_cont_all: EnumProperty(name="All Edges", items=PATCH_ALL, default="MIXED", update=_on_patch_all,
                              description="Continuity of every edge at once")
    pt_tension_all: FloatProperty(name="Tension", min=0.25, max=4.0, soft_min=0.5, soft_max=2.0, default=1.0,
                                  update=_on_patch_all_tension,
                                  description="Tension of every edge at once. " + TENSION_DESC)
    pt_degree: IntProperty(name="Degree", min=2, max=5, **_F())
    pt_fair: FloatProperty(name="Fairness", min=0.0, soft_max=20.0, **_F())
    pt_surface_fair: BoolProperty(name="Fair on Surface", default=True, description=SURFACE_FAIR_DESC, **_F())
    sq_spans_u: IntProperty(name="Spans U", min=0, max=32, **_F())
    sq_spans_v: IntProperty(name="Spans V", min=0, max=32, **_F())
    lr_start: EnumProperty(name="Start Profile", items=[("G0", "G0", ""), ("G1", "G1", ""), ("G2", "G2", "")], **_F())
    lr_end: EnumProperty(name="End Profile", items=[("G0", "G0", ""), ("G1", "G1", ""), ("G2", "G2", "")], **_F())
    lr_ra: EnumProperty(name="Rail A", items=[("G0", "G0", ""), ("G1", "G1", ""), ("G2", "G2", "")], **_F())
    lr_rb: EnumProperty(name="Rail B", items=[("G0", "G0", ""), ("G1", "G1", ""), ("G2", "G2", "")], **_F())
    lw_start: FloatProperty(name="Weight", min=0.05, soft_max=3.0, **_F())
    lw_end: FloatProperty(name="Weight", min=0.05, soft_max=3.0, **_F())
    lw_ra: FloatProperty(name="Weight", min=0.05, soft_max=3.0, **_F())
    lw_rb: FloatProperty(name="Weight", min=0.05, soft_max=3.0, **_F())
    lr_degree: EnumProperty(name="Degree", items=[("0", "Auto", ""), ("3", "3", ""), ("5", "5", "")], **_F())
    lr_mode: EnumProperty(name="Mode", items=[("RAILS", "Rails", ""), ("CENTRELINE", "Centreline", "")], **_F())
    lr_spans_u: IntProperty(name="Spans U (0 auto)", min=0, max=32, **_F())
    lr_spans_v: IntProperty(name="Spans V (0 auto)", min=0, max=32, **_F())
    bool_op: EnumProperty(name="Operation", items=[("UNION", "Union", ""), ("DIFFERENCE", "Difference", ""),
                                                   ("INTERSECTION", "Intersection", "")], **_F())
    jc_mode: EnumProperty(name="Mode", items=[("EXACT", "Exact", "Keep every piece's shape exactly"),
                                              ("REBUILD", "Rebuild", "Fit one clean uniform curve")], **_F())
    jc_degree: IntProperty(name="Degree (Rebuild)", min=1, max=5, **_F())
    jc_cvs: IntProperty(name="CVs (Rebuild, 0 auto)", min=0, max=200, **_F())


def sync_from_history(obj):
    """Copy an object's history params into its Feature panel properties."""
    h = features.history(obj)
    if h is None:
        return False
    pg = obj.nproto_feature
    _syncing[0] = True
    try:
        params = dict(h["params"])
        if h["op"] == "square" and "square_degree" in params:     # older squares: one degree for both
            params.setdefault("sq_degree_u", params["square_degree"])
            params.setdefault("sq_degree_v", params["square_degree"])
        if h["op"] == "prim_plane":                                # older planes: one degree for both
            params.setdefault("pl_degree_u", int(params.get("degree", 3)) or 3)
            params.setdefault("pl_degree_v", int(params.get("degree", 3)) or 3)
        if h["op"] == "boundary":                                  # older boundaries: fitted, auto layout
            for k_, v_ in (("bd_exact", False), ("sq_degree_u", 0), ("sq_degree_v", 0), ("sq_spans_u", 0), ("sq_spans_v", 0)):
                params.setdefault(k_, v_)
        if h["op"] in features.ORIENT_OPS:
            params.setdefault("flip_normal", False)
        if h["op"] == "patch":                                     # older patches: square nets, fixed size
            params.setdefault("pt_cvs_v", params.get("pt_cvs", 20))
            params.setdefault("pt_auto_cvs", False)
            params.setdefault("pt_surface_fair", True)           # on unless switched off
            pg.pt_edges.clear()
            tens = params.get("pt_tension", [])
            conts = params.get("cont", [])
            guides = set(params.get("pt_guides", []))
            for i, (src, ref) in enumerate(h["inputs"]):
                if i in guides:
                    continue
                it = pg.pt_edges.add()
                it.index, it.label = i, f"{src.name} {ref}"
                it.curve = not (features.is_surface_edge(ref) and src.type != 'CURVE')
                c_ = conts[i] if i < len(conts) else "G0"
                it.cont = "G0" if it.curve or c_ not in ("G0", "G1", "G2") else c_
                it.tension = float(tens[i]) if i < len(tens) else 1.0
            pg.pt_cont_all = cont_all(params, h["inputs"])
            vals = {round(float(tens[i]), 9) if i < len(tens) else 1.0 for i in range(len(h["inputs"])) if i not in guides}
            pg.pt_tension_all = vals.pop() if len(vals) == 1 else 1.0
        for name in FIELDS.get(h["op"], []):
            if name not in params:
                continue
            v = params[name]
            try:
                setattr(pg, name, str(v) if name in ("degree", "square_degree", "sq_degree_u", "sq_degree_v", "lr_degree") else v)
            except (TypeError, ValueError):
                pass
    finally:
        _syncing[0] = False
    obj["nproto_feature_synced"] = obj.get(features.PROP_HIST, "")
    return True


class NPROTO_OT_sync_feature(bpy.types.Operator):
    """Load this object's feature settings into the Feature panel"""
    bl_idname = "nproto.sync_feature"
    bl_label = "Edit Feature"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        return {'FINISHED'} if context.view_layer.objects.active and sync_from_history(context.view_layer.objects.active) else {'CANCELLED'}


def shows_feature(obj, context):
    """True when draw_feature has something to show for obj: its own history,
    objects built from it, or a place in another object's history tree."""
    from . import history_ui
    if obj is None:
        return False
    if features.has_history(obj):
        return True
    if (store.is_nurbs(obj) or obj.type == 'CURVE') and history_ui.dependents(obj, context.scene):
        return True
    return history_ui.tree_root(context, obj) != obj


def draw_feature(layout, context):
    from . import history_ui
    obj = context.view_layer.objects.active
    if not shows_feature(obj, context):
        return
    hist = features.has_history(obj)
    head = layout.row()
    head.alert = obj.hide_get()
    head.label(text=f"Editing: {obj.name}" + ("  (hidden input)" if obj.hide_get() else ""),
               icon='HIDE_ON' if obj.hide_get() else 'OBJECT_DATA')
    if not hist:
        box = layout.box()
        row = box.row()
        row.label(text="Baked: no history of its own", icon='FREEZE')
        row.operator("nproto.delete_feature", text="", icon='TRASH').name = obj.name
        ff = history_ui.from_feeds(obj, context.scene)
        if ff:
            box.label(text=ff, icon='LINKED')
    pg_s = obj.nproto_step
    if hist and pg_s.focus and pg_s.index >= 0:
        # a step picked in the Construction Tree: only that step, not the feature under it
        try:
            entries = features.stack(obj)
        except RuntimeError:
            entries = []
        if pg_s.index < len(entries):
            # laid out like any feature: its name, Pause Rebuild, its settings
            e = entries[pg_s.index]
            box = layout.box()
            row = box.row()
            row.label(text=f"Feature: {history_ui.step_label(e).partition(' · ')[0]}",
                      icon=history_ui.STEP_ICON.get(e["op"], 'MODIFIER'))
            bk = row.operator("nproto.bake_feature", text="", icon='FREEZE')
            bk.name, bk.step = obj.name, pg_s.index
            row.operator("nproto.remove_step", text="", icon='TRASH').index = pg_s.index
            if e["params"].get("_off"):
                box.label(text="Off: skipped (switch it on in the Construction Tree)", icon='INFO')
            from .ui import draw_pause
            draw_pause(box, context)
            history_ui.draw_steps(box, obj, only=pg_s.index, bare=True)
            history_ui.draw_tree(layout, context, obj)
            return
    if hist:
        h = features.history(obj)
        box = layout.box()
        if h is not None:
            op = h["op"]
            title = PRETTY.get(op, op.replace("prim_", "").title())
            row = box.row()
            row.label(text=f"Feature: {title}", icon='MODIFIER')
            if op in features.KEEP_OPS:
                row.operator("nproto.restore_inputs", text="Restore Inputs", icon='LOOP_BACK')
            bk = row.operator("nproto.bake_feature", text="", icon='FREEZE')
            bk.name, bk.step = obj.name, -1
            row.operator("nproto.delete_feature", text="", icon='TRASH').name = obj.name
            ff = history_ui.from_feeds(obj, context.scene)
            if ff:
                box.label(text=ff, icon='LINKED')
            if op == "surface_fillet":       # how it was built (set in the Fillet's redo panel)
                prm = h["params"]
                shape = ("Blend G%d" % prm["blend"]) if prm.get("blend") else \
                    ("Chamfer" if prm.get("xsect") == 0 else "Circular")
                box.label(text=f"Shape: {shape} · Rail: {'Chord' if prm.get('chord') else 'Rolling Ball'}",
                          icon='MOD_BEVEL')
            from .ui import draw_pause
            draw_pause(box, context)
            if obj.get("nproto_feature_synced") != obj.get(features.PROP_HIST):
                box.operator("nproto.sync_feature", icon='FILE_REFRESH')
            else:
                col = box.column(align=True)
                pg = obj.nproto_feature
                if op == "patch":
                    _draw_patch(box, obj, pg, h)
                for name in FIELDS.get(op, []) if op != "patch" else []:
                    if op == "square" and name[0] in "ew" and name[1:].isdigit():
                        if name[0] == "e":
                            r_ = col.row(align=True)
                            r_.label(text=f"Edge {int(name[1]) + 1}")
                            r_.prop(pg, name, expand=True)
                            r_.prop(pg, "w" + name[1], text="")
                        continue
                    if op == "loft_rails" and pg.lr_mode == "CENTRELINE" and name != "lr_mode":
                        continue
                    if op == "loft_rails" and name.startswith("lr_") and name[3:] in ("start", "end", "ra", "rb"):
                        r_ = col.row(align=True)
                        r_.label(text=pg.bl_rna.properties[name].name)
                        r_.prop(pg, name, expand=True)
                        r_.prop(pg, "lw_" + name[3:], text="")
                        continue
                    if op == "loft_rails" and name.startswith("lw_"):
                        continue
                    if op in ("square", "boundary") and name in ("sq_degree_u", "sq_spans_u"):
                        r_ = col.row(align=True)
                        r_.label(text="Degree" if name == "sq_degree_u" else "Spans (0 auto)")
                        r_.prop(pg, name, text="")
                        r_.prop(pg, name.replace("_u", "_v"), text="")
                        continue
                    if op in ("square", "boundary") and name in ("sq_degree_v", "sq_spans_v"):
                        continue
                    if op == "boundary" and name == "bd_exact":
                        r_ = col.row(align=True)
                        r_.prop(pg, "bd_exact", text="Exact" if pg.bd_exact else "Fit", toggle=True,
                                icon='CHECKBOX_HLT' if pg.bd_exact else 'CHECKBOX_DEHLT')
                        continue
                    if op == "boundary" and name == "samples":
                        r_ = col.row(align=True)
                        r_.active = not pg.bd_exact          # used by Fit (and when Exact falls back)
                        r_.prop(pg, "samples")
                        continue
                    if op == "perp_profile":
                        shp = pg.pp_shape
                        if name in ("radius", "degree") and shp not in ("CIRCLE", "ELLIPSE") or \
                                name == "radius2" and shp != "ELLIPSE" or name == "width" and shp in ("CIRCLE", "ELLIPSE") \
                                or name == "depth" and shp != "RECTANGLE" or \
                                name == "spans_u" and (shp not in ("CIRCLE", "ELLIPSE") or pg.degree == "0"):
                            continue
                        if name == "width" and shp == "SQUARE":
                            col.prop(pg, name, text="Size")
                            continue
                    if op == "dome" and name == "dome_cvs" and pg.dome_layout != "DRAPED":
                        continue
                    if name in ("degree", "square_degree", "jc_mode", "bool_op", "lr_degree", "lr_mode", "pp_shape",
                                "dome_cont", "dome_layout"):
                        col.row().prop(pg, name, expand=True)
                    elif name in ("jc_degree", "jc_cvs") and pg.jc_mode != "REBUILD":
                        continue
                    else:
                        col.prop(pg, name, text=LABELS.get(name, pg.bl_rna.properties[name].name))
                if op == "join_curves" and obj.type == 'CURVE':
                    for sp in obj.data.splines:
                        kind = "piecewise Bezier" if sp.use_bezier_u else "uniform"
                        box.label(text=f"{len(sp.points)} CVs · degree {sp.order_u - 1} · {kind}", icon='CURVE_NCURVE')
                if op == "square":
                    from .draw import topology
                    t = topology(obj)
                    if t:
                        box.label(text=f"{t['count'][0]} x {t['count'][1]} CVs · degree {t['degree'][0]} · "
                                       f"spans {t['spans'][0]} x {t['spans'][1]}  (CVs shown in the viewport)",
                                  icon='MESH_GRID')
                    try:
                        qi = json.loads(obj.get("nproto_square_info", "{}"))
                    except ValueError:
                        qi = {}
                    if qi:
                        box.label(text=f"gap {qi.get('gap', 0):.1e} · max angle {qi.get('angle', 0):.3f}° · "
                                       f"curvature err {100 * qi.get('curv', 0):.1f}%", icon='DRIVER_DISTANCE')
                        for note in qi.get("corner_notes", []):
                            box.label(text=note, icon='ERROR')
        elif features.PROP_HIST in obj:
            box.label(text="Feature: an input was deleted (history broken)", icon='ERROR')
        # (its steps -- fillets, trims ... -- are rows of the Construction Tree: click one to edit it)
    history_ui.draw_tree(layout, context, obj)


_resyncing = set()


def _defer_sync(name):
    """Resync an object's Feature panel right after this redraw (a panel may not write while drawing)."""
    if name in _resyncing:
        return
    _resyncing.add(name)

    def run():
        _resyncing.discard(name)
        o = bpy.data.objects.get(name)
        if o is not None:
            sync_from_history(o)
            for win in bpy.context.window_manager.windows:
                for area in win.screen.areas:
                    area.tag_redraw()
        return None
    bpy.app.timers.register(run, first_interval=0.0)


def _draw_patch(box, obj, pg, h):
    from . import patch as _pt
    p = h["params"]
    conts = p.get("cont", [])
    loop = [k for k in range(len(h["inputs"])) if k not in set(p.get("pt_guides", []))]
    if [e.index for e in pg.pt_edges] != loop or pg.pt_cont_all != cont_all(p, h["inputs"]) or any(
            not e.curve and e.index < len(conts) and e.cont != (conts[e.index] if conts[e.index] in ("G0", "G1", "G2")
                                                                else "G0") for e in pg.pt_edges):
        _defer_sync(obj.name)          # rows from before they carried continuity (or out of step)
    _pt.draw_settings(box, pg, pg.pt_edges, _pt.patch_info(obj), "pt_cont_all", obj=obj, h=h)


class NPROTO_OT_remove_step(bpy.types.Operator):
    """Remove this history step and rebuild (the object keeps its other steps)"""
    bl_idname = "nproto.remove_step"
    bl_label = "Remove History Step"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty()

    def execute(self, context):
        obj = context.view_layer.objects.active
        if obj is None:
            return {'CANCELLED'}
        features.remove_step(obj, self.index)
        obj.nproto_step.index = -1
        if features.PROP_STACK not in obj and features.PROP_HIST not in obj and features.PROP_BASE in obj:
            text, nat = obj[features.PROP_BASE], obj.get(features.PROP_BASE + "_natural", 0)
            store.set_geometry_text(obj, text, context, keep_untrimmed=False)
            obj[store.PROP_NATURAL] = nat
            del obj[features.PROP_BASE]
            if store.PROP_UNTRIM in obj:
                del obj[store.PROP_UNTRIM]
        else:
            try:
                features.rebuild(obj, context)
            except Exception as exc:
                self.report({'ERROR'}, f"Rebuild failed: {exc}")
                return {'CANCELLED'}
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Add Primitive
# --------------------------------------------------------------------------

KIND_ITEMS = [(k, k.title(), "") for k in list(primitives.CURVE_KINDS) + list(primitives.SURFACE_KINDS)]
DEFAULTS = {"radius": 1.0, "radius2": 0.4, "length": 2.0, "width": 2.0, "depth": 2.0, "height": 2.0, "corner": 0.0, "taper": 0.0, "left": False,
            "turns": 3.0, "sides": 6, "angle": 90.0, "spans_u": 8, "spans_v": 4, "degree": 0, "caps": False,
            "elliptic": False}


def view_size(context):
    """A size for a new shape that suits the view: about a quarter of the 3D
    view's width at the 3D cursor, rounded to the grid step the view shows
    (so 2" or 1' in an inch scene, never a fixed 2 m). None without a 3D view."""
    from bpy_extras import view3d_utils
    from .drawtools import grid_steps, visible_grid_step
    area = context.area
    if area is None or area.type != 'VIEW_3D':
        return None
    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
    rv3d = getattr(area.spaces.active, "region_3d", None)
    if region is None or rv3d is None or region.width < 2:
        return None
    cur = context.scene.cursor.location
    c = view3d_utils.location_3d_to_region_2d(region, rv3d, cur)
    y = c.y if c is not None else region.height * 0.5
    a = view3d_utils.region_2d_to_location_3d(region, rv3d, (0.0, y), cur)
    b = view3d_utils.region_2d_to_location_3d(region, rv3d, (float(region.width), y), cur)
    W = (b - a).length
    if not W > 1e-9:
        return None
    us = context.scene.unit_settings
    ov = getattr(area.spaces.active, "overlay", None)
    steps = grid_steps(us.system, us.scale_length or 1.0, getattr(ov, "grid_scale", 1.0) or 1.0,
                       getattr(ov, "grid_subdivisions", 10) or 10)
    step = visible_grid_step(steps, region.width / W)
    return max(step, round(W / 4.0 / step) * step)


def place_primitive(context, name, op, params, M, g=None):
    """A primitive feature (op, params: built in its own space) as a new object
    with world matrix M; it stays editable in the Feature panel. g: the built
    geometry when it is at hand. Returns the object."""
    if g is None:
        g = features.run(op, [], params)
    if isinstance(g, tuple) and g[0] == "spline":
        cu = bpy.data.curves.new(name, 'CURVE')
        obj = bpy.data.objects.new(name, cu)
        (context.collection or context.scene.collection).objects.link(obj)
        obj.matrix_world = M
        primitives.write_spline_object(obj, g[1])
    else:
        obj = store.new_object(context, name, g, M)
    features.record(obj, op, [], params)
    sync_from_history(obj)
    return obj


def primitive_params(op, **values):
    """The full parameter set of a primitive feature: its fields' defaults, then values."""
    p = {n: DEFAULTS[n] for n in FIELDS[op] if n in DEFAULTS}
    p.update(values)
    return p


class NPROTO_OT_add_primitive(bpy.types.Operator):
    """Add a NURBS curve or surface primitive at the 3D cursor (editable later in the Feature panel)"""
    bl_idname = "nproto.add_primitive"
    bl_label = "Add NURBS Primitive"
    bl_options = {'REGISTER', 'UNDO'}

    kind: EnumProperty(name="Kind", items=KIND_ITEMS, default="CIRCLE")
    radius: FloatProperty(name="Radius", default=1.0, min=1e-4, unit='LENGTH')
    radius2: FloatProperty(name="Radius 2", default=0.4, min=0.0, unit='LENGTH')
    length: FloatProperty(name="Length", default=2.0, min=1e-4, unit='LENGTH')
    width: FloatProperty(name="Size X", default=2.0, min=1e-4, unit='LENGTH')
    depth: FloatProperty(name="Size Y", default=2.0, min=1e-4, unit='LENGTH')
    corner: FloatProperty(name="Corner Radius", default=0.0, min=0.0, unit='LENGTH',
                          description="Round the rectangle's corners (exact quarter circles); 0: sharp")
    height: FloatProperty(name="Height", default=2.0, min=1e-4, unit='LENGTH')
    turns: FloatProperty(name="Turns", default=3.0, min=0.1, soft_max=20.0)
    taper: FloatProperty(name="Taper °", default=0.0, min=-80.0, max=80.0, soft_min=-45.0, soft_max=45.0,
                         description="Lean the helix in (positive: narrower on top) or out (negative: wider), "
                                     "like a thread on a cone")
    left: BoolProperty(name="Left-Handed", default=False, description="The helix turns the other way as it rises")
    sides: IntProperty(name="Sides", default=6, min=3, max=64)
    angle: FloatProperty(name="Angle", default=90.0, min=0.01, max=360.0)
    degree: EnumProperty(name="Degree", items=DEGREE_ITEMS, default="0")
    spans_u: IntProperty(name="Spans U", default=8, min=1, max=64)
    spans_v: IntProperty(name="Spans V", default=4, min=1, max=64)
    pl_degree_u: IntProperty(name="Degree U", default=3, min=1, max=7, description="Degree along X")
    pl_degree_v: IntProperty(name="Degree V", default=3, min=1, max=7, description="Degree along Y")
    caps: BoolProperty(name="Caps (solid)", default=False)
    elliptic: BoolProperty(name="Ellipse", default=False)

    def _op(self):
        return primitives.CURVE_KINDS.get(self.kind) or primitives.SURFACE_KINDS[self.kind]

    def _params(self):
        op = self._op()
        p = {n: (int(getattr(self, n)) if n == "degree" else getattr(self, n)) for n in FIELDS[op]}
        if self.kind == "ELLIPSE":
            p["elliptic"] = True
        if self.kind in ("LINE", "PLANE", "BOX", "HELIX") and p.get("degree") == 0:
            p["degree"] = 1 if self.kind in ("LINE", "BOX") else 3   # no 'exact' variant: straight / fitted
        return p

    def invoke(self, context, event):
        # sized to the view (a quarter of its width, on the visible grid), not a fixed 2 m
        S = view_size(context)
        if S:
            self.length = self.width = self.depth = self.height = S
            self.radius = 0.5 * S
            self.radius2 = 0.5 * S * (0.3 if self.kind == "TORUS" else 0.4)
        # sensible per-kind defaults on first use
        if self.kind in ("LINE",):
            self.spans_u = 1
        if self.kind in ("PLANE", "BOX"):
            self.spans_u, self.spans_v = 1, 1
        if self.kind == "TORUS" and not S:
            self.radius2 = 0.3
        if self.kind == "CONE":
            self.radius2 = 0.0
        return self.execute(context)

    def execute(self, context):
        if load_kernel() is None:
            self.report({'ERROR'}, "USD BRep kernel not available")
            return {'CANCELLED'}
        op, params = self._op(), self._params()
        try:
            g = features.run(op, [], params)
        except Exception as exc:
            self.report({'ERROR'}, f"{self.kind.title()} failed: {exc}")
            return {'CANCELLED'}
        M = Matrix.Translation(context.scene.cursor.location)
        obj = place_primitive(context, "NURBS " + self.kind.title(), op, params, M, g)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {'FINISHED'}

    def draw(self, context):
        lay = self.layout
        op = self._op()
        lay.label(text=self.kind.title())
        col = lay.column(align=True)
        is_ellipse = self.kind == "ELLIPSE"
        for n in FIELDS[op]:
            if n == "elliptic":
                if not is_ellipse:          # the Ellipse kind is always elliptic
                    col.prop(self, n)
                continue
            if n == "degree":
                col.row().prop(self, "degree", expand=True)
            elif n == "radius2" and op == "prim_circle":
                row = col.row()
                row.active = is_ellipse or self.elliptic   # Radius 2 only shapes an ellipse
                row.prop(self, n)
            else:
                col.prop(self, n)


classes = (NprotoPatchEdge, NprotoFeature, NPROTO_OT_sync_feature, NPROTO_OT_remove_step, NPROTO_OT_add_primitive)


def register():
    bpy.types.Object.nproto_feature = PointerProperty(type=NprotoFeature)


def unregister():
    del bpy.types.Object.nproto_feature
