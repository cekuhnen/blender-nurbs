"""Operators: Sweep, Pipe, Boundary, Iso Curve, Duplicate Edge, Project Curve,
Intersect, Rebuild Surface, Continuity Check."""

import json
import math

import bpy
import numpy as np
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from mathutils import Vector

from . import draw, features, geom, imprint, store, sweeps, workplanes
from . import hints
from .kernel import load as load_kernel
from .ops import (CHAIN_ITEMS, AskValue, PickMixin, SolidBody, _curve_inputs, _decode_inputs, _encode_inputs,
                  _kernel_ok, _make_result, _select_only, tangent_chain)
from .feature_ui import NprotoPipePoint, NprotoScalePoint, pipe_points_param, sweep_points_param
from .valuedrag import BASE_PX, ValueDrag, fmt_length, ruled_preview


def _new_surface(context, name, op, inputs, params):
    g = features.run(op, inputs, params)
    obj = store.new_object(context, name, g)
    if features.history_enabled(context):
        features.record(obj, op, inputs, params)
    return obj


def _new_curves(context, name, op, inputs, params):
    from .primitives import write_spline_object
    g = features.run(op, inputs, params)
    cu = bpy.data.curves.new(name, 'CURVE')
    obj = bpy.data.objects.new(name, cu)
    (context.collection or context.scene.collection).objects.link(obj)
    write_spline_object(obj, g[1])
    if features.history_enabled(context):
        features.record(obj, op, inputs, params)
    return obj


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

SCALE_TAPER_ITEMS = [("LINEAR", "Linear", "The scale runs straight from point to point (a crease at each point)"),
                     ("SMOOTH", "Smooth", "A smooth curve through the points, level at both ends")]

RAIL_MAX_POINTS = 32       # points along a rail in a value stage (each has its own drag fields)


class RailPoints:
    """Points along a rail in a value stage (Pipe's radius points, Sweep's
    scale points): each has a size arrow and a diamond sliding it along the
    rail. A press on the rail away from the handles adds a point there and
    takes its arrow; X or Delete over a point's arrow or diamond removes it.
    A tool sets point_coll (its collection's name), point_attr (the size
    attribute of an item) and point_prefix (the size and the where fields'
    prefixes, 3 letters each: field i is prefix + i), and gives _rail_s()
    ({"g": rail length at each sample, "P", "total"}), _points_live() and
    _point_value(f) (the size a new point at share f starts with)."""

    point_coll = "points"
    point_attr = "radius"
    point_prefix = ("pr_", "pa_")

    def _rail_points(self):
        return getattr(self, self.point_coll)

    def _point_fields(self):
        out = ()
        for i in range(min(len(self._rail_points()), RAIL_MAX_POINTS)):
            out += (f"{self.point_prefix[0]}{i}", f"{self.point_prefix[1]}{i}")
        return out

    def _point_n(self, i):
        """Point i's number along the rail (1 = nearest the start)."""
        pts = self._rail_points()
        return 1 + sorted(range(len(pts)), key=lambda j: pts[j].at).index(i)

    def _rail_share(self, ray_o, ray_d):
        """The share of the rail's length nearest the mouse ray."""
        rs = self._rail_s()
        d = ray_d / (np.linalg.norm(ray_d) or 1.0)
        rel = rs["P"] - ray_o
        off = rel - np.outer(rel @ d, d)
        k = int(np.argmin(np.linalg.norm(off, axis=1)))
        return float(rs["g"][k]) / (rs["total"] or 1.0)

    def _at_handle(self, i, C, T):
        """The diamond sliding point i along the rail (C, T: the rail there)."""
        return {"field": f"{self.point_prefix[1]}{i}", "kind": "draft", "origin": C, "dir": T,
                "param": self._rail_points()[i].at, "pos": C, "pick": lambda o, r, prev: self._rail_share(o, r),
                "to_value": lambda t: min(0.995, max(0.005, t)), "snap": (0.05, 0.01)}

    def _points_hook(self, context, event):
        if not self._points_live():
            return None
        if event.type == 'DEL' and event.value == 'PRESS':
            return self._remove_hot(context)
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS' or self._grab is not None:
            return None
        if not PickMixin._in_view(context, event):
            return None
        m = self._mouse(context, event)
        if self._handle_at(context, m) is not None:
            return None
        if not self._near_screen(context, self._rail_s()["P"], m):
            return None
        pts = self._rail_points()
        if len(pts) >= RAIL_MAX_POINTS:
            self._msg = f"at most {RAIL_MAX_POINTS} points"
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        region, rv3d = self._win(context)
        from bpy_extras import view3d_utils
        o = np.array(view3d_utils.region_2d_to_origin_3d(region, rv3d, m), float)
        r = np.array(view3d_utils.region_2d_to_vector_3d(region, rv3d, m), float)
        f = min(0.995, max(0.005, self._rail_share(o, r)))
        value = self._point_value(f)
        pt = pts.add()
        pt.at = f
        setattr(pt, self.point_attr, value)
        name = f"{self.point_prefix[0]}{len(pts) - 1}"
        hi = next(j for j, h in enumerate(self.drag_handles()) if h["field"] == name)
        self._start_grab(context, hi, m)
        self._drag_refresh(context)
        return {'RUNNING_MODAL'}

    def _remove_hot(self, context):
        hs = self.drag_handles()
        if self._hot is None or self._hot >= len(hs) or not hs[self._hot]["field"].startswith(self.point_prefix):
            return None
        self._rail_points().remove(int(hs[self._hot]["field"][3:]))
        self._hot, self._field, self._typed = None, 0, ""
        self._drag_refresh(context)
        return {'RUNNING_MODAL'}

    def _point_label(self, name, size_word):
        n = self._point_n(int(name[3:]))
        return f"Point {n} At" if name.startswith(self.point_prefix[1]) else f"{size_word} {n}"


def _point_at(i, coll):
    def get(self):
        return getattr(self, coll)[i].at

    def set_(self, v):
        getattr(self, coll)[i].at = min(0.995, max(0.005, float(v)))
    return property(get, set_)


class NPROTO_OT_sweep(SolidBody, RailPoints, ValueDrag, AskValue, PickMixin, bpy.types.Operator):
    """Sweep a profile along one rail, two rails (Shift+click the 1st rail), or a
    surface edge (tangent to that surface). Select the profile, then the path
    (last); or with nothing selected click the profile, then the path. From
    the UI one rail gives a live preview: drag the arrow at the end for End
    Scale, click the rail to add a scale point (T: linear / smooth, X: remove
    a point); click away or Enter to make it"""
    bl_idname = "nproto.sweep"
    bl_label = "Sweep"
    bl_options = {'REGISTER', 'UNDO'}
    op_name = "sweep"

    pick_min = 2
    pick_max = 3
    ask_dialog = False      # no preview (two rails, Value Input = Dialog): made at once, as it always was
    point_coll = "scale_points"
    point_attr = "scale"
    point_prefix = ("ss_", "sa_")

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    samples: IntProperty(name="Samples", default=32, min=6, max=300, description="Frames along the rail")
    tangent: BoolProperty(name="Tangent to Surface", default=True,
                          description="Rail is a surface edge: make the sweep G1 to that surface")
    flip_tangent: BoolProperty(name="Flip Tangent", default=False)
    scale_end: FloatProperty(name="End Scale", default=1.0, min=0.01, soft_max=5.0)
    scale_points: CollectionProperty(type=NprotoScalePoint, options={'SKIP_SAVE'},
                                     description="The profile's scale at points along the rail")
    scale_taper: EnumProperty(name="Scale Taper", items=SCALE_TAPER_ITEMS, default="LINEAR",
                              description="How the scale runs from point to point: straight (a crease at each "
                                          "point) or a smooth curve, level at both ends")
    add_scale_point: BoolProperty(name="Add Scale Point", default=False, options={'SKIP_SAVE'},
                                  description="A scale point in the middle of the longest stretch between points")
    twist: FloatProperty(name="Twist", default=0.0, soft_min=-720.0, soft_max=720.0)
    keep_height: BoolProperty(name="Keep Height (2 rails)", default=False,
                              description="Stretch the profile between the rails without scaling its height")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        self.body_prefer(context)
        if self.inputs:                     # picked already (see PickMixin._finish)
            return self.ask(context)
        crv = [o for o in context.selected_objects if o.type == 'CURVE']
        act = context.active_object
        if len(crv) in (2, 3) and act in crv and not event.alt:
            prof = sweep_profile(crv, act)
            rails = [o for o in crv if o != prof]
            rails.sort(key=lambda o: o == act)          # the path selected last
            self.inputs = _encode_inputs([(prof, "S0")] + [(o, "S0") for o in rails])
            return self.ask(context)
        return self.start_picking(context, event)

    def pick_state(self):
        n = len(self._picks)
        return ["click the profile", "click the path (Shift+click: a first rail of two)",
                "click the second rail"][min(n, 2)]

    def _params(self):
        p = {k: getattr(self, k) for k in ("samples", "tangent", "flip_tangent", "scale_end", "twist", "keep_height")}
        if len(self.scale_points) or self.scale_taper != "LINEAR":
            p.update(scale_points=sweep_points_param(self.scale_points), scale_taper=self.scale_taper)
        return p

    def _edit_points(self):
        """The redo panel's buttons: rows marked Remove go, Add puts one in the
        middle of the longest stretch between points (at the scale there)."""
        for i in reversed(range(len(self.scale_points))):
            if self.scale_points[i].remove:
                self.scale_points.remove(i)
        if self.add_scale_point:
            self.add_scale_point = False
            pts = sweep_points_param(self.scale_points)
            fs = [0.0] + [a for a, _ in pts] + [1.0]
            k = max(range(len(fs) - 1), key=lambda i: fs[i + 1] - fs[i])
            at = 0.5 * (fs[k] + fs[k + 1])
            law = sweeps.sweep_scale_law({"scale_end": self.scale_end, "scale_taper": self.scale_taper,
                                          "scale_points": pts})
            it = self.scale_points.add()
            it.at, it.scale = at, float(law(at))

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._edit_points()
        try:
            inputs = _decode_inputs(self.inputs)
            if len(inputs) < 2:
                raise ValueError("pick a profile and a rail (Shift+click for a second rail)")
            if self.body != "SURFACE":
                return self.execute_solid(context, inputs, self._params())
            obj = _make_result(context, "Sweep", "sweep", inputs, self._params())   # a surface, or pieces joined
        except Exception as exc:
            self.report({'ERROR'}, f"Sweep failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        return {'FINISHED'}


    def solid_groups(self, ins):
        return [ins]                  # the profile and its rail(s): one solid

    def draw(self, context):
        lay = self.layout
        self.draw_body(lay)
        col = lay.column()
        for k in ("samples", "tangent", "flip_tangent", "scale_end", "twist", "keep_height"):
            col.prop(self, k)
        two_rails = len(_decode_inputs(self.inputs)) >= 3 if self.inputs else False
        box = lay.box()
        box.active = not two_rails
        row = box.row(align=True)
        row.label(text="Scale" + ("  (two rails: set by the rails)" if two_rails else ""))
        row.prop(self, "scale_taper", expand=True)
        pc = box.column(align=True)
        for n, i in enumerate(sorted(range(len(self.scale_points)), key=lambda i: self.scale_points[i].at)):
            it = self.scale_points[i]
            r = pc.row(align=True)
            r.prop(it, "at", text=f"{n + 1}  At", slider=True)
            r.prop(it, "scale", text="×")
            r.prop(it, "remove", text="", icon='X', toggle=True)
        box.prop(self, "add_scale_point", icon='ADD', toggle=True)

    # -- the live value stage (valuedrag.py): one rail ---------------------------
    PREVIEW_ROWS = 49           # sections along the rail
    PREVIEW_RING = 49           # points along the profile

    @property
    def drag_fields(self):
        return ("scale_end",) + self._point_fields()

    @property
    def ratio_fields(self):
        return self.drag_fields

    def _label(self, name):
        if name.startswith(self.point_prefix):
            return self._point_label(name, "Scale")
        return ValueDrag._label(self, name)

    def _fmt(self, context, name):
        if name.startswith(self.point_prefix[1]):
            return f"{100.0 * getattr(self, name):.3g} %"
        return ValueDrag._fmt(self, context, name)

    def drag_setup(self, context, mouse):
        """The rail's frames and the profile on them (as the sweep places it),
        and the profile's point farthest from the rail: the scale arrows
        stand on it."""
        ins = _decode_inputs(self.inputs)
        if len(ins) != 2:
            return False                # two rails set the scale: made at once
        sr = sweeps.sweep_rail(ins, self._params(), 64)
        q = sweeps.sweep_local(sr, geom.sample_curve(sr["prof_k"], self.PREVIEW_RING))
        d = np.linalg.norm(q, axis=1)
        k = int(np.argmax(d))
        if d[k] < 1e-9:
            return False
        self._sr, self._q, self._tip, self._tip_len = sr, q, q[k], float(d[k])
        total = float(np.sum(np.linalg.norm(np.diff(sr["P"], axis=0), axis=1)))
        self._rs = {"P": sr["P"], "g": np.linspace(0.0, total, len(sr["P"])), "total": total}
        return True

    def _rail_s(self):
        return self._rs

    def _points_live(self):
        return getattr(self, "_rs", None) is not None

    def _law(self):
        return sweeps.sweep_scale_law(self._params())

    def _point_value(self, f):
        return max(0.01, float(self._law()(f)))

    def _tip_at(self, f):
        """(rail point, rail tangent, unit direction to the far profile point) at share f."""
        Pk, Tk, Nk, Bk = sweeps.sweep_frame_at(self._sr, f)
        d = sweeps.sweep_row(self._sr, self._tip[None], lambda _f: 1.0, Pk, Tk, Nk, Bk, f)[0] - Pk
        return Pk, Tk, d / (np.linalg.norm(d) or 1.0)

    def drag_modal_hook(self, context, event):
        return self._points_hook(context, event)

    def drag_handles(self):
        L = self._tip_len

        def scale_of(t, L=L):
            return max(0.01, abs(t) / L)
        C, _, d = self._tip_at(1.0)
        out = [{"field": "scale_end", "kind": "arrow", "origin": C, "dir": d, "param": self.scale_end * L,
                "base": C, "to_value": scale_of, "snap": (0.1, 0.01)}]
        for i in range(min(len(self.scale_points), RAIL_MAX_POINTS)):
            pt = self.scale_points[i]
            C, T, d = self._tip_at(pt.at)
            out.append({"field": f"ss_{i}", "kind": "arrow", "origin": C, "dir": d, "param": pt.scale * L,
                        "base": C, "to_value": scale_of, "snap": (0.1, 0.01)})
            out.append(self._at_handle(i, C, T))
        return out

    def drag_preview(self):
        law = self._law()
        cuts = [float(x) for x in law.f[1:-1]] if law.taper == "LINEAR" else []
        fs = sorted(set(np.linspace(0.0, 1.0, self.PREVIEW_ROWS).tolist()) | set(cuts))
        rows = [sweeps.sweep_row(self._sr, self._q, law, *sweeps.sweep_frame_at(self._sr, f), f) for f in fs]
        lines, tris = ruled_preview(rows, ribs=8, row_every=4)
        # the rows at the points, so a crease shows where it is
        lines += [rows[fs.index(f)] for f in cuts]
        return {"lines": lines, "tris": tris, "guides": []}

    def drag_key(self, context, event, ch):
        if ch == "t":
            self.scale_taper = "SMOOTH" if self.scale_taper == "LINEAR" else "LINEAR"
        elif ch == "x":
            return self._remove_hot(context) is not None
        elif ch == "f" and self._sr["tangent"]:
            self.flip_tangent = not self.flip_tangent
            self.drag_setup(context, None)
        elif not self.body_key(ch):
            return False
        return True

    def drag_handle_hints(self):
        return [hints.key("LMB drag", "arrows: end / point scale"),
                hints.key("LMB drag", "diamond on the rail: slide a point"),
                hints.key("LMB", "on the rail: add a scale point"),
                hints.key("X", "remove the point under the mouse")]

    def drag_rows(self):
        rows = [("Scale", [hints.cycle("T", ["linear", "smooth"], self.scale_taper.lower())])]
        if self._sr["tangent"]:
            rows.append(("Surface", [hints.toggle("F", "flip tangent", self.flip_tangent)]))
        return rows + [self.body_row()]

    def drag_state(self):
        n = len(self.scale_points)
        return ([f"{n} scale point{'s' if n > 1 else ''}"] if n else []) + self.body_state()


def _scale_size(i):
    def get(self):
        return self.scale_points[i].scale

    def set_(self, v):
        self.scale_points[i].scale = max(0.01, float(v))
    return property(get, set_)


for _i in range(RAIL_MAX_POINTS):          # the value stage's fields for scale point i: ss_i (scale), sa_i (where)
    setattr(NPROTO_OT_sweep, f"ss_{_i}", _scale_size(_i))
    setattr(NPROTO_OT_sweep, f"sa_{_i}", _point_at(_i, "scale_points"))


def sweep_profile(crv, act):
    """The profile among selected curves, the path selected last (active):
    with two, the one not active; with three (a profile on two rails), the
    one that touches both others, as a profile spans its rails."""
    others = [o for o in crv if o != act]
    if len(crv) == 2:
        return others[0]
    from . import features
    pts = {o: geom.sample_curve(features.input_curve(o, "S0"), 64) for o in crv}

    def gap(a, b):
        A, B = pts[a], pts[b]
        return float(np.min(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)))
    return min(crv, key=lambda c: sum(gap(c, o) for o in crv if o != c))


PIPE_CORNER_ITEMS = [("MITRE", "Mitre", "Sharp corners of the rail: the tubes cut on the plane halving the turn"),
                     ("ROUND", "Round", "Sharp corners of the rail: an exact torus elbow, G1 to straight legs")]

PIPE_MAX_POINTS = RAIL_MAX_POINTS       # radius points along a rail (each has its own drag fields pr_i / pa_i)

PIPE_TAPER_ITEMS = [("LINEAR", "Linear", "The radius changes evenly along the rail (a cone on a line)"),
                    ("SMOOTH", "Smooth", "The radius eases out of the start and into the end (level at both ends)")]

PIPE_CAP_ITEMS = [("NONE", "None", "Open ends (a solid gets flat caps)"),
                  ("FLAT", "Flat", "Flat discs close the ends: a closed body"),
                  ("ROUND", "Round", "Half spheres close the ends, tangent to the tube: a closed body")]


class NPROTO_OT_pipe(SolidBody, RailPoints, ValueDrag, AskValue, PickMixin, bpy.types.Operator):
    """Sweep an exact circle along curves or surface edges (selected, or pick
    them: a click takes a tangent chain, Shift+click keeps picking). Pieces
    that touch end to end make one pipe. From the UI the radius has an arrow
    to drag on a live preview (D: diameter, V: variable radius, C: caps,
    M: corners, J: join); type a value, click away or Enter to make it"""
    bl_idname = "nproto.pipe"
    bl_label = "Pipe"
    bl_options = {'REGISTER', 'UNDO'}
    op_name = "pipe"

    accepts_picks = True
    pick_min = 1
    pick_max = 256          # a click takes an edge (its tangent chain); Shift+click keeps picking, Enter finishes
    extra_keys = (("C", "tangent chain on / off"),)
    extra_help = "(pieces that touch end to end make one pipe; separate ones a pipe each)"
    fit_view = "shrink"     # the last radius stays, unless it would swamp the view
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    radius: FloatProperty(name="Radius", default=0.1, min=1e-4, unit='LENGTH')
    diameter: FloatProperty(name="Diameter", default=0.2, min=2e-4, unit='LENGTH')
    # each run writes the size into both fields, so switching it keeps the size
    use_diameter: BoolProperty(name="Diameter", default=False,
                               description="Give the pipe's size as its diameter instead of its radius")
    use_variable: BoolProperty(name="Variable Radius", default=False,
                               description="The radius runs from Radius at the rail's start to End Radius at its end")
    radius_end: FloatProperty(name="End Radius", default=0.05, min=1e-4, unit='LENGTH')
    diameter_end: FloatProperty(name="End Diameter", default=0.1, min=2e-4, unit='LENGTH')
    taper: EnumProperty(name="Taper", items=PIPE_TAPER_ITEMS, default="LINEAR",
                        description="How a variable radius runs along the rail")
    points: CollectionProperty(type=NprotoPipePoint, options={'SKIP_SAVE'},
                               description="Radius points along the rail, between its start and end")
    cap_style: EnumProperty(name="Cap", items=PIPE_CAP_ITEMS, default="NONE",
                            description="How the pipe's ends are closed (a closed rail has no ends)")
    samples: IntProperty(name="Samples", default=32, min=6, max=300,
                         description="Frames along each curve or edge of the rail")
    join: BoolProperty(name="Join Touching", default=True,
                       description="Curves and edges that touch end to end make one pipe (it can't turn a sharp "
                                   "corner); off: a pipe each")
    chain: EnumProperty(name="Pick", items=CHAIN_ITEMS, default="TANGENT",
                        description="What one click on an edge picks")
    chain_angle: FloatProperty(name="Chain Angle", default=2.0, min=0.01, max=sweeps.PIPE_CORNER_DEG,
                               description="Tangent Chain: edges meeting within this angle (degrees) are one run")
    corner_style: EnumProperty(name="Corners", items=PIPE_CORNER_ITEMS, default="MITRE",
                               description="How the pipe turns a sharp corner of its rail")
    bend: FloatProperty(name="Bend", default=2.0, min=1.0, soft_max=10.0,
                        description="Round corners: the elbow's centreline radius, as a multiple of the pipe's "
                                    "radius (1: the inside of the bend comes to a point)")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        self.body_prefer(context)
        if self.inputs:                     # picked already (see PickMixin._finish)
            return self.ask(context)
        crv = _curve_inputs(context)          # every spline of the selected curves
        if crv and not event.alt:
            self.inputs = _encode_inputs(crv)
            return self.ask(context)
        return self.start_picking(context, event)

    # -- picking ------------------------------------------------------------------
    def expand_pick(self, cand):
        if self.chain == 'TANGENT':
            have = {(o.name, r) for o, r in getattr(self, "_picks", [])}
            return [c for c in tangent_chain(cand, self._cands, self.chain_angle) if (c[0].name, c[1]) not in have]
        return [cand]

    def pick_state(self):
        mode = "tangent chain" if self.chain == 'TANGENT' else "single edge"
        n = len(getattr(self, "_picks", []))
        if not n:
            return f"click a curve or edge ({mode})"
        return f"{n} picked · Shift+click more ({mode})"

    def tool_key(self, context, event):
        if event.type == 'C' and not (event.ctrl or event.oskey or event.alt):
            self.chain = 'SINGLE' if self.chain == 'TANGENT' else 'TANGENT'
            return True
        return False

    def _groups(self, ins):
        """The pipes to make: pieces that touch end to end together (Join Touching)."""
        return features.profile_groups(ins) if self.join else [[x] for x in ins]

    def _radius(self):
        return 0.5 * self.diameter if self.use_diameter else self.radius

    def _radius_end(self):
        if not self.use_variable:
            return self._radius()
        return 0.5 * self.diameter_end if self.use_diameter else self.radius_end

    def params(self):
        p = {"radius": self._radius(), "samples": self.samples, "cap_style": self.cap_style,
             "corner_style": self.corner_style, "bend": self.bend}
        if self.use_variable:
            p.update(variable=True, radius_end=self._radius_end(), taper=self.taper)
            if len(self.points):
                p["radius_points"] = pipe_points_param(self.points)
        return p

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if self.use_diameter:               # both fields hold the size made (the redo panel, the next run)
            self.radius = max(1e-4, 0.5 * self.diameter)
            self.radius_end = max(1e-4, 0.5 * self.diameter_end)
        else:
            self.diameter = 2.0 * self.radius
            self.diameter_end = 2.0 * self.radius_end
        params = self.params()
        if self.body != "SURFACE":
            try:
                inputs = _decode_inputs(self.inputs)
            except ValueError as exc:
                self.report({'ERROR'}, f"Pipe failed: {exc}")
                return {'CANCELLED'}
            return self.execute_solid(context, inputs, params)
        made, failed = [], []
        try:
            groups = self._groups(_decode_inputs(self.inputs))
        except Exception as exc:
            self.report({'ERROR'}, f"Pipe failed: {exc}")
            return {'CANCELLED'}
        for g in groups:           # one rail that fails doesn't take the others with it
            try:
                made.append(_make_result(context, "Pipe", "pipe", g, params))
            except Exception as exc:
                failed.append(f"{g[0][0].name}: {exc}" if len(groups) > 1 else str(exc))
        if failed:
            self.report({'ERROR'} if not made else {'WARNING'}, "Pipe failed: " + "; ".join(failed))
        if not made:
            return {'CANCELLED'}
        _select_only(context, made)
        return {'FINISHED'}

    def solid_groups(self, ins):
        return self._groups(ins)      # a solid rod per rail

    def draw(self, context):
        lay = self.layout
        self.draw_body(lay)
        n_in = len(_decode_inputs(self.inputs)) if self.inputs else 0
        if n_in > 1:
            lay.prop(self, "join")
        row = lay.row(align=True)
        row.prop(self, "diameter" if self.use_diameter else "radius",
                 text=("Start " if self.use_variable else "") + ("Diameter" if self.use_diameter else "Radius"))
        row.prop(self, "use_diameter", text="", icon='ARROW_LEFTRIGHT')
        lay.prop(self, "use_variable")
        if self.use_variable:
            col = lay.column(align=True)
            col.prop(self, "diameter_end" if self.use_diameter else "radius_end")
            col.row(align=True).prop(self, "taper", expand=True)
            if len(self.points):
                pc = lay.column(align=True)
                pc.label(text="Radius Points (X over one in the preview removes it)")
                for n, i in enumerate(sorted(range(len(self.points)), key=lambda i: self.points[i].at)):
                    row = pc.row(align=True)
                    row.prop(self.points[i], "at", text=f"{n + 1}  At", slider=True)
                    row.prop(self.points[i], "radius", text="R")
        lay.prop(self, "cap_style", expand=True)
        if self.body != "SURFACE" and self.cap_style == "NONE":
            lay.label(text="A solid gets flat caps", icon='INFO')
        row = lay.row(align=True)
        row.prop(self, "corner_style", expand=True)
        sub = row.row(align=True)
        sub.active = self.corner_style == "ROUND"
        sub.prop(self, "bend", text="Bend ×")
        lay.prop(self, "samples")
        row = lay.row(align=True)
        row.prop(self, "chain", text="")
        sub = row.row(align=True)
        sub.active = self.chain == 'TANGENT'
        sub.prop(self, "chain_angle", text="°")

    # -- the live value stage (valuedrag.py) ----------------------------------
    @property
    def drag_fields(self):
        size = "diameter" if self.use_diameter else "radius"
        out = (size,) + ((size + "_end",) if self.use_variable else ())
        if self.use_variable:
            out += self._point_fields()
        return out + (("bend",) if self.corner_style == "ROUND" and getattr(self, "_corner", None) else ())

    @property
    def ratio_fields(self):
        return ("bend",) + tuple(f"pa_{i}" for i in range(min(len(self.points), PIPE_MAX_POINTS)))

    def _label(self, name):
        if name.startswith(self.point_prefix):
            return self._point_label(name, "Diameter" if self.use_diameter else "Radius")
        return ValueDrag._label(self, name)

    def _fmt(self, context, name):
        if name.startswith("pa_"):
            return f"{100.0 * getattr(self, name):.3g} %"
        return ValueDrag._fmt(self, context, name)

    PREVIEW_ROWS = 48           # circles along the rail
    PREVIEW_RING = 33           # points round a circle (closed)

    def drag_setup(self, context, mouse):
        """Per rail: its spans (one, or a stretch per corner: mitred), its open
        ends (for the caps' preview) and the largest radius its mitres allow."""
        self._pp, self._bad = [], []
        self._n_in = len(_decode_inputs(self.inputs))
        for g in self._groups(_decode_inputs(self.inputs)):      # curves and surface edges alike
            rail = sweeps.pipe_rail(g)
            closed = features.is_closed_curve(rail)
            try:
                tracks = sweeps.pipe_tracks(rail, closed, self.corner_style)
            except RuntimeError as exc:         # a rail turning back on itself
                self._bad.append((str(exc), [c["point"] for c in sweeps.rail_corners(rail) if c["deg"] > 175.0]))
                tracks = None
            if tracks is None:
                P, T, N, B = sweeps.pipe_frames(rail, min(240, self.PREVIEW_ROWS * len(g)), closed)
                gk = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
                spans = [{"P": P, "T": T, "N": N, "B": B, "track": None, "g": gk, "s0": 0.0}]
                total = float(gk[-1])
                limit = math.inf
            else:
                spans, total = [], 0.0
                for tr in tracks:
                    T, N = tr["T"], tr["N"]
                    spans.append({"P": tr["P"], "T": T, "N": N, "B": tr["h"] * np.cross(T, N), "track": tr,
                                  "s0": total})
                    total += float(tr["s"][-1])
                # Mitre: the largest radius; Round: the largest elbow centreline radius
                limit = sweeps.elbow_limit(tracks) if self.corner_style == "ROUND" else sweeps.mitre_limit(tracks)
            ends = []
            if not closed:
                a_, b_ = spans[0], spans[-1]
                ends = [(a_["P"][0], -a_["T"][0], a_["N"][0], a_["B"][0]),
                        (b_["P"][-1], b_["T"][-1], b_["N"][-1], b_["B"][-1])]
            G, Rp, Rt, Rn, Rb = [], [], [], [], []       # the rail sampled: rail length from its start, frames
            for sp in spans:
                G.append(sp["g"] if sp["track"] is None else sp["s0"] + sp["track"]["s"])
                Rp.append(sp["P"]), Rt.append(sp["T"]), Rn.append(sp["N"]), Rb.append(sp["B"])
            rail_s = {"g": np.concatenate(G), "P": np.vstack(Rp), "T": np.vstack(Rt), "N": np.vstack(Rn),
                      "B": np.vstack(Rb), "total": total}
            self._pp.append({"spans": spans, "ends": ends, "limit": limit, "inputs": g, "total": total,
                             "closed": closed, "rail": rail_s})
        if not self._pp:
            return False
        # the corner the Bend handle stands at: the first one of the first rail
        self._corner = next((sp["track"]["end"] for pp in self._pp for sp in pp["spans"]
                             if sp["track"] is not None and sp["track"]["end"] is not None), None)
        spans = self._pp[0]["spans"]
        mid = spans[len(spans) // 2]
        kb, self._base_at_mouse = (len(spans) // 2, len(mid["P"]) // 2), False
        if mouse is not None:
            best = None
            for i, sp in enumerate(spans):
                for k, q in enumerate(sp["P"]):
                    sq = self._screen(context, q)
                    if sq is not None and (sq - mouse).length <= BASE_PX and (best is None or (sq - mouse).length < best[0]):
                        best = ((sq - mouse).length, (i, k))
            if best is not None:
                kb, self._base_at_mouse = best[1], True
        self._kb = kb
        self._dir = self._arrow_dir(context)
        return True

    def _foot(self):
        i, k = self._kb
        sp = self._pp[0]["spans"][i]
        return sp["P"][k], sp["T"][k], sp["N"][k], sp["B"][k]

    def drag_on_input(self, context, mouse):
        if self.use_variable and self._pp[0]["ends"]:
            return False                # a press on the rail adds a radius point there (drag_modal_hook)
        return self._near_screen(context, np.vstack([sp["P"] for sp in self._pp[0]["spans"]]), mouse)

    # -- radius points ------------------------------------------------------------
    def _rail_at(self, f):
        """(point, T, N, B) on the first rail at share f of its length."""
        rs = self._pp[0]["rail"]
        g = min(1.0, max(0.0, f)) * self._pp[0]["total"]
        k = int(np.argmin(np.abs(rs["g"] - g)))
        return rs["P"][k], rs["T"][k], rs["N"][k], rs["B"][k]

    def _rail_s(self):
        return self._pp[0]["rail"]

    def _points_live(self):
        return self.use_variable and bool(getattr(self, "_pp", None)) and bool(self._pp[0]["ends"])

    def _point_value(self, f):
        return max(1e-4, float(self._law(self._pp[0])(f * self._pp[0]["total"])))

    def _point_handles(self, k):
        out = []
        law = self._law(self._pp[0])
        for i in range(min(len(self.points), PIPE_MAX_POINTS)):
            pt = self.points[i]
            C, T, N, B = self._rail_at(pt.at)
            d = self._side(N, B, T)
            out.append({"field": f"pr_{i}", "kind": "arrow", "origin": C, "dir": d, "param": pt.radius,
                        "base": C, "to_value": lambda t, k=k: max(1e-4 * k, abs(t) * k)})
            out.append(self._at_handle(i, C, T))
        return out

    def drag_modal_hook(self, context, event):
        """Variable radius: a press on the rail adds a radius point (RailPoints)."""
        return self._points_hook(context, event)

    def _arrow_dir(self, context):
        """Out of the rail at the arrow's foot, square to it: of the frame's
        four directions the one most across the view, so it is seen side on."""
        P, T, N, B = self._foot()
        _, rv3d = self._win(context)
        if rv3d is None:
            return N
        view = np.array(rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)), float)
        up = np.array(rv3d.view_rotation @ Vector((0.0, 1.0, 0.0)), float)
        side = np.cross(view, T)
        side = side if np.linalg.norm(side) > 1e-6 else up
        return max((N, B, -N, -B), key=lambda d: float(d @ side))

    def _law(self, pp):
        pts = [(p.at, p.radius) for p in self.points] if self.use_variable else ()
        return sweeps.RadiusLaw(self._radius(), self._radius_end(), self.taper, pp["total"], pts)

    def drag_handles(self):
        k = 2.0 if self.use_diameter else 1.0
        pp = self._pp[0]
        if self.use_variable and pp["ends"]:
            # Variable: an arrow at each end of the first rail, square to it
            out = []
            for field, (C, out_, N, B), r_ in ((self.drag_fields[0], pp["ends"][0], self._radius()),
                                                (self.drag_fields[1], pp["ends"][1], self._radius_end())):
                d = self._side(N, B, out_)
                out.append({"field": field, "kind": "arrow", "origin": C, "dir": d, "param": r_, "base": C,
                            "to_value": lambda t, k=k: max(1e-4 * k, abs(t) * k)})
            out += self._point_handles(k)
        else:
            B0 = self._foot()[0]
            r_ = float(self._law(pp)(self._foot_g())) if self.use_variable else self._radius()
            out = [{"field": self.drag_fields[0], "kind": "arrow", "origin": B0, "dir": self._dir,
                    "param": r_, "base": B0,
                    "to_value": lambda t, k=k: max(1e-4 * k, abs(t) * k)}]
        c = getattr(self, "_corner", None)
        if c is not None and self.corner_style == "ROUND":
            # a diamond where the elbow meets the leg after the corner, slid along that leg
            tn = math.tan(math.radians(c["deg"]) / 2.0)
            r = max(self._radius(), self._radius_end())     # elbows: Bend x the largest radius
            out.append({"field": "bend", "kind": "draft", "origin": c["C"], "dir": c["t_out"],
                        "param": sweeps.elbow_reach(c, self.bend * r),
                        "to_value": lambda t, tn=tn, r=r: max(1.0, abs(t) / tn / r) if tn > 1e-9 else None,
                        "snap": (0.1, 0.01)})
        return out

    def _side(self, N, B, T):
        """Of the frame's four directions square to the rail, the one most across the view."""
        rv3d = self._win(bpy.context)[1]
        if rv3d is None:
            return N
        view = np.array(rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)), float)
        side = np.cross(view, T)
        if np.linalg.norm(side) < 1e-6:
            side = np.array(rv3d.view_rotation @ Vector((0.0, 1.0, 0.0)), float)
        return max((N, B, -N, -B), key=lambda d: float(d @ side))

    def _foot_g(self):
        i, k = self._kb
        sp = self._pp[0]["spans"][i]
        if sp["track"] is None:
            return float(sp["g"][k])
        return sp["s0"] + float(sp["track"]["s"][k])

    def drag_preview(self):
        r = self._radius()
        style = sweeps.pipe_cap_style({"cap_style": self.cap_style, "caps": self.body != "SURFACE"})
        a = np.linspace(0.0, 2.0 * math.pi, self.PREVIEW_RING)
        ca, sa = np.cos(a), np.sin(a)
        ring = r * np.stack([ca, sa], axis=1)
        lines, tris, guides = [], [], []
        rnd = self.corner_style == "ROUND"
        unit = np.stack([ca, sa], axis=1)
        for pp in self._pp:
            law = self._law(pp)
            rmax = law.rmax
            bend = self.bend * rmax
            too_big = (bend if rnd else rmax) > pp["limit"] * (1.0 + (1e-6 if rnd else 1e-9))
            for sp in pp["spans"]:
                tr = sp["track"]
                pieces = []
                if tr is None:
                    P, N, B = sp["P"], sp["N"], sp["B"]
                    rk = law(sp["g"])
                    pieces.append(([P[k] + rk[k] * (np.outer(unit[:, 0], N[k]) + np.outer(unit[:, 1], B[k]))
                                    for k in range(len(P))], 6))
                else:           # a stretch between corners: its columns cut off on the mitres / where the elbows start
                    L_ = float(tr["s"][-1])
                    used = sum(sweeps.elbow_reach(tr[k], bend) for k in ("start", "end")
                               if rnd and tr[k] is not None)
                    if too_big or L_ - used > 1e-7 * L_:
                        pieces.append((list(sweeps.mitred_rows(tr, unit, 25, bend, law, sp["s0"])), 4))
                    if rnd and tr["end"] is not None and not too_big:
                        c = tr["end"]
                        Pk, Tk, Nk, Bk = sweeps._frame_end(tr, -1)
                        p0, O, ax, th = sweeps.elbow_geometry(c, bend)
                        rc = law(sp["s0"] + L_)
                        X = p0 + rc * (np.outer(unit[:, 0], Nk) + np.outer(unit[:, 1], Bk))
                        pieces.append(([O + sweeps._rot_rows(X - O, ax, f) for f in np.linspace(0.0, th, 13)], 3))
                for rows, every in pieces:
                    L, Tr = ruled_preview(rows, ribs=8, row_every=every)
                    if too_big:                     # the mitres / elbows don't fit: shown in red, not made
                        guides += [(ln, (1.0, 0.25, 0.2, 1.0)) for ln in L]
                        continue
                    lines += L
                    tris.append(Tr)
            if style == "NONE" or too_big:
                continue
            for (C, out, N, B), gk, sgn in zip(pp["ends"], (0.0, pp["total"]), (-1.0, 1.0)):
                rk = law(gk)
                dirs = np.outer(unit[:, 0], N) + np.outer(unit[:, 1], B)
                circle = C + rk * dirs
                if style == "FLAT":
                    fan = np.stack([np.repeat(C[None], len(circle) - 1, axis=0), circle[:-1], circle[1:]], axis=1)
                    tris.append(fan.reshape(-1, 3))
                else:           # the round cap: latitude rings over its sphere, from the end circle to the pole
                    off, R, th0 = sweeps.cap_meridian(rk, sgn * law.slope(gk))
                    lat = np.linspace(th0, 0.5 * math.pi, 7)
                    cap = [C + off * out + R * math.cos(f) * dirs + R * math.sin(f) * out for f in lat]
                    L, Tr = ruled_preview(cap, ribs=8, row_every=2)
                    lines += L
                    tris.append(Tr)
        for _, pts in self._bad:                # a red ring round a corner the pipe can't take
            for q in pts:
                guides.append((q + 1.6 * r * np.outer(ca, self._ring_x(q)) + 1.6 * r * np.outer(sa, self._ring_y(q)),
                               (1.0, 0.25, 0.2, 1.0)))
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3)), "guides": guides}

    def _ring_x(self, q):
        rv3d = self._win(bpy.context)[1]
        return np.array(rv3d.view_rotation @ Vector((1.0, 0.0, 0.0)), float)

    def _ring_y(self, q):
        rv3d = self._win(bpy.context)[1]
        return np.array(rv3d.view_rotation @ Vector((0.0, 1.0, 0.0)), float)

    def drag_check(self):
        if self._bad:
            return self._bad[0][0] + (" · J: a pipe each" if self.join and self._n_in > 1 else "")
        r = self._law(self._pp[0]).rmax if self.use_variable else self._radius()
        for pp in self._pp:
            sep = " · J: a pipe each" if self.join and len(pp["inputs"]) > 1 else ""
            if self.use_variable and pp["closed"] and abs(self._radius_end() - self._radius()) > 1e-12:
                return "a closed rail has no start and end for a varying radius (V: one radius)"
            if self.corner_style == "ROUND" and self.bend * r > pp["limit"] * (1.0 + 1e-6):
                return f"the elbows don't fit: Bend at most ×{pp['limit'] / r:.2f} at this radius" + sep
            if self.corner_style != "ROUND" and r > pp["limit"] * (1.0 + 1e-9):
                lim = pp["limit"] * (2.0 if self.use_diameter else 1.0)
                return f"too large for the mitred corners: at most {fmt_length(bpy.context, lim)}" + sep
        return "" if r > 1e-4 else "Radius 0: drag the arrow out, or type a radius"

    def drag_key(self, context, event, ch):
        if ch == "d":                   # the same size, given the other way
            r, re = self._radius(), (0.5 * self.diameter_end if self.use_diameter else self.radius_end)
            self.use_diameter = not self.use_diameter
            self.radius, self.diameter = max(1e-4, r), max(2e-4, 2.0 * r)
            self.radius_end, self.diameter_end = max(1e-4, re), max(2e-4, 2.0 * re)
        elif ch == "v":
            if not self.use_variable:   # start the end where the start is: drag its arrow from there
                r = self._radius()
                self.radius_end, self.diameter_end = r, 2.0 * r
            self.use_variable = not self.use_variable
        elif ch == "t" and self.use_variable:
            self.taper = "SMOOTH" if self.taper == "LINEAR" else "LINEAR"
        elif ch == "x" and self.use_variable:
            return self._remove_hot(context) is not None
        elif ch == "c":
            keys = [k for k, _, _ in PIPE_CAP_ITEMS]
            self.cap_style = keys[(keys.index(self.cap_style) + 1) % len(keys)]
        elif ch == "j" and self._n_in > 1:
            self.join = not self.join
            self.drag_setup(context, None)
        elif ch == "m" and getattr(self, "_corner", None) is not None:
            self.corner_style = "ROUND" if self.corner_style == "MITRE" else "MITRE"
            self.drag_setup(context, None)
        elif not self.body_key(ch):
            return False
        return True

    def drag_handle_hints(self):
        word = "diameter" if self.use_diameter else "radius"
        out = ([hints.key("LMB drag", f"arrows: start / end / point {word}"),
                hints.key("LMB drag", "diamond on the rail: slide a point"),
                hints.key("LMB", "on the rail: add a radius point"),
                hints.key("X", "remove the point under the mouse")] if self.use_variable
               else [hints.key("LMB drag", "arrow: " + word)])
        if self.corner_style == "ROUND" and getattr(self, "_corner", None) is not None:
            out.append(hints.key("LMB drag", "diamond: bend"))
        return out

    def drag_rows(self):
        names = {"NONE": "none", "FLAT": "flat", "ROUND": "round"}
        rows = [("Size", [hints.toggle("D", "diameter", self.use_diameter),
                          hints.toggle("V", "variable", self.use_variable)]
                 + ([hints.cycle("T", ["linear", "smooth"], self.taper.lower())] if self.use_variable else [])),
                ("Caps", [hints.cycle("C", list(names.values()), names[self.cap_style])])]
        if getattr(self, "_corner", None) is not None:
            rows.append(("Corners", [hints.cycle("M", ["mitre", "round"], self.corner_style.lower())]))
        if self._n_in > 1:
            rows.append(("Rails", [hints.toggle("J", "join touching", self.join)]))
        return rows + [self.body_row()]

    def drag_state(self):
        style = sweeps.pipe_cap_style({"cap_style": self.cap_style, "caps": self.body != "SURFACE"})
        caps = {"NONE": "Open", "FLAT": "Flat caps", "ROUND": "Round caps"}[style]
        n = len(self._pp)
        mitres = sum(len(pp["spans"]) - (1 if pp["ends"] else 0)        # a closed rail: one more joint
                     for pp in self._pp if pp["spans"][0]["track"] is not None)
        word = "elbow" if self.corner_style == "ROUND" else "mitre"
        npts = len(self.points) if self.use_variable else 0
        return (([f"{n} pipes"] if n > 1 else []) + ([f"{mitres} {word}{'s' if mitres > 1 else ''}"] if mitres else [])
                + ([f"{npts} radius point{'s' if npts > 1 else ''}"] if npts else []) + [caps] + self.body_state())


def _point_size(i):
    def get(self):
        r = self.points[i].radius
        return 2.0 * r if self.use_diameter else r

    def set_(self, v):
        self.points[i].radius = max(1e-4, 0.5 * v if self.use_diameter else v)
    return property(get, set_)


for _i in range(PIPE_MAX_POINTS):          # the value stage's fields for radius point i: pr_i (size), pa_i (where)
    setattr(NPROTO_OT_pipe, f"pr_{_i}", _point_size(_i))
    setattr(NPROTO_OT_pipe, f"pa_{_i}", _point_at(_i, "points"))


BD_DEG = [("0", "Auto", "Auto: exact edges keep the curves' own degree; a fit uses 3")] + \
    [(str(d), str(d), "") for d in range(1, 8)]


class NPROTO_OT_boundary(PickMixin, bpy.types.Operator):
    """Four-sided surface from 4 boundary curves or edges (selected, or pick them)"""
    bl_idname = "nproto.boundary"
    bl_label = "Boundary"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 4
    pick_max = 4
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    bd_method: EnumProperty(name="Method", default="EXACT", items=[
        ("EXACT", "Exact", "Build the patch from the curves' own CVs: the edges ARE the inputs and the net is as "
                           "light as the curves (polynomial curves / untrimmed surface edges). Falls back to Fit "
                           "when not possible"),
        ("FIT", "Fit", "Read Samples points from each curve, blend them and fit a net of the chosen degree / spans")])
    sq_degree_u: EnumProperty(name="Degree U", items=BD_DEG, default="0")
    sq_degree_v: EnumProperty(name="Degree V", items=BD_DEG, default="0")
    sq_spans_u: IntProperty(name="Spans U", default=0, min=0, max=32,
                            description="0 = auto (exact: the curves' own knots; fit: fewest within tolerance)")
    sq_spans_v: IntProperty(name="Spans V", default=0, min=0, max=32,
                            description="0 = auto (exact: the curves' own knots; fit: fewest within tolerance)")
    samples: IntProperty(name="Samples", default=24, min=6, max=200,
                         description="Fit: points read from each curve. More = the edges follow the curves "
                                     "more closely (and with Spans 0, usually more CVs)")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        crv = [o for o in context.selected_objects if o.type == 'CURVE']
        if len(crv) == 4 and not event.alt:
            self.inputs = _encode_inputs([(o, "S0") for o in crv])
            return self.execute(context)
        return self.start_picking(context, event)

    def draw(self, context):
        from .sweeps import boundary_info
        lay = self.layout
        lay.row().prop(self, "bd_method", expand=True)
        fitted = self.bd_method == "FIT" or (boundary_info and not boundary_info.get("exact"))
        if self.bd_method == "EXACT" and boundary_info.get("why"):
            box = lay.box()
            box.label(text="Fitted: exact not possible", icon='INFO')
            box.label(text=boundary_info["why"])
        col = lay.column(align=True)
        row = col.row(align=True)
        row.label(text="Degree")
        row.prop(self, "sq_degree_u", text="")
        row.prop(self, "sq_degree_v", text="")
        row = col.row(align=True)
        row.label(text="Spans (0 auto)")
        row.prop(self, "sq_spans_u", text="")
        row.prop(self, "sq_spans_v", text="")
        if fitted:
            col.prop(self, "samples")
        if boundary_info:
            (nu, nv), (du, dv) = boundary_info["cvs"], boundary_info["degree"]
            how = "exact edges" if boundary_info["exact"] else "fitted"
            lay.label(text=f"{nu} x {nv} CVs · degree {du} x {dv} · {how}", icon='MESH_GRID')

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            inputs = _decode_inputs(self.inputs)
            if len(inputs) != 4:
                raise ValueError("needs exactly 4 boundary curves")
            obj = _new_surface(context, "Boundary", "boundary", inputs, {
                "samples": self.samples, "bd_exact": self.bd_method == "EXACT",
                "sq_degree_u": int(self.sq_degree_u), "sq_degree_v": int(self.sq_degree_v),
                "sq_spans_u": self.sq_spans_u, "sq_spans_v": self.sq_spans_v})
        except Exception as exc:
            self.report({'ERROR'}, f"Boundary failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        return {'FINISHED'}


SQ_CONT = [("FREE", "Free", "No condition: the edge isn't held to this boundary, it follows from the other sides"),
           ("G0", "G0", "Position only"), ("G1", "G1", "Tangent to the neighbouring surface"),
           ("G2", "G2", "Curvature continuous with the neighbouring surface")]
SQ_DEG = [("0", "Auto", "Auto: the lightest layout within tolerance (G2 sides -> 5, G1 -> 3)")] + \
    [(str(d), str(d), "") for d in range(1, 8)]


class NPROTO_OT_square(PickMixin, bpy.types.Operator):
    """Four-sided patch with G0 / G1 / G2 to the neighbouring surfaces on each side
    (Alias Square). Pick the 4 boundaries (surface edges or curves, any order),
    Enter after 3 for a three-sided patch"""
    bl_idname = "nproto.square"
    bl_label = "Square"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 3
    pick_max = 4
    finish_on_min = False
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    e0: EnumProperty(name="Edge 1", items=SQ_CONT, default="G0")
    e1: EnumProperty(name="Edge 2", items=SQ_CONT, default="G0")
    e2: EnumProperty(name="Edge 3", items=SQ_CONT, default="G0")
    e3: EnumProperty(name="Edge 4", items=SQ_CONT, default="G0")
    w0: FloatProperty(name="Weight 1", default=1.0, min=0.05, soft_max=3.0)
    w1: FloatProperty(name="Weight 2", default=1.0, min=0.05, soft_max=3.0)
    w2: FloatProperty(name="Weight 3", default=1.0, min=0.05, soft_max=3.0)
    w3: FloatProperty(name="Weight 4", default=1.0, min=0.05, soft_max=3.0)
    sq_degree_u: EnumProperty(name="Degree U", items=SQ_DEG, default="0",
                              description="Degree along U (bottom / top); Auto picks the lightest layout")
    sq_degree_v: EnumProperty(name="Degree V", items=SQ_DEG, default="0",
                              description="Degree along V (left / right); Auto picks the lightest layout")
    sq_spans_u: IntProperty(name="Spans U", default=0, min=0, max=32, description="0 = auto (fewest within tolerance)")
    sq_spans_v: IntProperty(name="Spans V", default=0, min=0, max=32, description="0 = auto (fewest within tolerance)")
    sq_retrim: BoolProperty(name="Re-trim Planar Targets", default=False,
                            description="A side that is the trim curve of a planar face becomes one clean curve in "
                                        "the matched layout, and that face is re-trimmed to it (keeps the square light)")
    sq_move: FloatProperty(name="Max Edge Move %", default=5.0, min=0.01, max=20.0,
                           description="Re-trim: how far the planar face's edge may move, in % of the patch size "
                                       "(larger = lighter net, e.g. single span)")
    sq_match: BoolProperty(name="Match Neighbours", default=True,
                           description="Take degree and knots from the neighbouring surfaces' edges, so the CV rows "
                                       "line up with theirs (used with Degree Auto and Spans 0)")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        crv = [o for o in context.selected_objects if o.type == 'CURVE']
        if len(crv) == 4 and not event.alt:
            self.inputs = _encode_inputs([(o, "S0") for o in crv])
            return self.execute(context)
        return self.start_picking(context, event)

    def _params(self):
        p = {f"e{i}": getattr(self, f"e{i}") for i in range(4)}
        p.update({f"w{i}": getattr(self, f"w{i}") for i in range(4)})
        p.update({"sq_degree_u": int(self.sq_degree_u), "sq_degree_v": int(self.sq_degree_v), "sq_spans_u": self.sq_spans_u,
                  "sq_spans_v": self.sq_spans_v, "sq_match": self.sq_match, "sq_retrim": self.sq_retrim,
                  "sq_move": self.sq_move})
        return p

    def draw(self, context):
        from . import square
        lay = self.layout
        ins = _decode_inputs(self.inputs) if self.inputs else []
        for i in range(max(3, min(4, len(ins) or 4))):
            row = lay.row(align=True)
            name = f"{ins[i][0].name} {ins[i][1]}" if i < len(ins) else f"Edge {i + 1}"
            row.label(text=name)
            row.prop(self, f"e{i}", expand=True)
            row.prop(self, f"w{i}", text="")
        lay.prop(self, "sq_match")
        sub = lay.row()
        sub.enabled = self.sq_match
        sub.prop(self, "sq_retrim")
        sub2 = lay.row()
        sub2.enabled = self.sq_match and self.sq_retrim
        sub2.prop(self, "sq_move")
        row = lay.row(align=True)
        row.label(text="Degree")
        row.prop(self, "sq_degree_u", text="")
        row.prop(self, "sq_degree_v", text="")
        row = lay.row(align=True)
        row.label(text="Spans")
        row.prop(self, "sq_spans_u", text="")
        row.prop(self, "sq_spans_v", text="")
        inf = square.LAST_INFO
        if inf:
            lay.label(text=f"{inf['cvs'][0]} x {inf['cvs'][1]} CVs · degree {inf['degree'][0]} · "
                           f"spans {inf['spans'][0]} x {inf['spans'][1]}", icon='MESH_GRID')
            lay.label(text=f"gap {inf.get('gap', 0):.1e} · max angle {inf.get('angle', 0):.3f}° · "
                           f"curvature err {100 * inf.get('curv', 0):.1f}%", icon='DRIVER_DISTANCE')
            for note in inf.get("corner_notes", []):
                lay.label(text=note, icon='ERROR')

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            inputs = _decode_inputs(self.inputs)
            if len(inputs) not in (3, 4):
                raise ValueError("needs 4 boundaries (or 3)")
            from . import square as _sq
            params = self._params()
            g = features.run("square", inputs, params)
            clean = list(_sq.LAST_INFO.get("clean_edges", []))
            info = dict(_sq.LAST_INFO)
            obj = store.new_object(context, "Square", g)
            # re-trim the planar targets to the clean edges, then point the
            # square's history at the new edges (the face itself has no history)
            if clean:
                fixed = []
                for c in clean:
                    tgt = bpy.data.objects[c["src"][0]]
                    features.clear_history(tgt)
                    new_ref = _sq.retrim_planar(tgt, c["src"][1], c["cvs"], c["knots"], c["degree"])
                    fixed.append((c["src"], new_ref))
                    self.report({'INFO'}, f"{tgt.name}: re-trimmed to the square's edge "
                                          f"(edge moved up to {c['deviation']:.1e})")
                inputs = [(o, dict(((a, b), n) for (a, b), n in fixed).get((o.name, r), r)) for o, r in inputs]
                params["sq_retrim"] = False      # done: the faces now carry the clean edges
            if features.history_enabled(context):
                features.record(obj, "square", inputs, params)
            _sq.LAST_INFO.clear()
            _sq.LAST_INFO.update(info)
            _sq.store_info(obj)
        except Exception as exc:
            self.report({'ERROR'}, f"Square failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Curves from surfaces
# --------------------------------------------------------------------------

class NPROTO_OT_iso_curve(bpy.types.Operator):
    """Extract an isoparametric curve from the active (untrimmed) surface"""
    bl_idname = "nproto.iso_curve"
    bl_label = "Iso Curve"
    bl_options = {'REGISTER', 'UNDO'}

    iso_dir: EnumProperty(name="Direction", items=[("U", "U", ""), ("V", "V", "")], default="U")
    param: FloatProperty(name="Position", default=0.5, min=0.0, max=1.0, subtype='FACTOR')

    def execute(self, context):
        obj = context.active_object
        if not (_kernel_ok(self) and store.is_nurbs(obj)):
            self.report({'WARNING'}, "Select a NURBS surface")
            return {'CANCELLED'}
        try:
            c = _new_curves(context, obj.name + "_iso", "iso_curve", [(obj, "SURF")],
                            {"iso_dir": self.iso_dir, "param": self.param})
        except Exception as exc:
            self.report({'ERROR'}, f"Iso curve failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [c])
        return {'FINISHED'}


class NPROTO_OT_dup_edge(PickMixin, bpy.types.Operator):
    """Duplicate surface edges as Blender NURBS curves (pick edges, Enter)"""
    bl_idname = "nproto.dup_edge"
    bl_label = "Duplicate Edge"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 1
    pick_max = 256
    finish_on_min = False
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def pick_candidates(self, context):
        return [c for c in super().pick_candidates(context) if c[0].type != 'CURVE']

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        return self.start_picking(context, event)

    def execute(self, context):
        try:
            c = _new_curves(context, "EdgeCurve", "dup_edge", _decode_inputs(self.inputs), {})
        except Exception as exc:
            self.report({'ERROR'}, f"Duplicate edge failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [c])
        return {'FINISHED'}


OUTPUT_ITEMS = [
    ("CURVE", "Curve", "New curves lying on the surface; the surface itself is not changed"),
    ("EDGE", "Edge", "Imprint: edges of the surface itself, its faces divided where they land. Nothing is "
                     "removed and a solid stays closed; the new faces and edges take Push / Pull, Extrude Face, "
                     "Blend, Fillet and the rest"),
]


def _project_inputs(context, target):
    """What Project puts onto target: every spline of the selected curves (a
    stray one-point spline, a click Draw Line left behind, is no curve and is
    left out), and the edges picked (Edge mode) on other selected NURBS objects."""
    from . import subsel
    out = []
    for o in context.selected_objects:
        if o == target:
            continue
        if o.type == 'CURVE':
            out += [(o, f"S{i}") for i, sp in enumerate(o.data.splines)
                    if len(sp.bezier_points if sp.type == 'BEZIER' else sp.points) >= 2]
        elif store.is_nurbs(o):
            out += [(o, f"E{i}") for i in subsel.picks(o, "EDGE")]
    return out


def imprint_object(op, context, tgt, inputs, p, pre=None):
    """Imprint tgt with inputs (params p, see imprint.run) and record the step.
    pre: the result computed already (info, text). Returns the info, or None
    (reported) when nothing was imprinted."""
    from . import jobs, subsel
    from .ops_trim import _frozen
    try:
        info, text = pre if pre is not None else imprint.run(tgt, inputs, p)
    except jobs.JobError as exc:
        op.report({'WARNING'}, f"Imprint on {tgt.name}: {exc}")
        return None
    base = tgt[store.PROP_BREP]
    keep = features.history_enabled(context)
    if not keep:
        _frozen(tgt, op)
    store.set_geometry_text(tgt, text, context)
    subsel.set_picks(tgt, "FACE", [])          # the face numbers changed
    subsel.set_picks(tgt, "EDGE", [])
    if keep:                                   # the edges follow what made them
        features.push_step(tgt, imprint.OP, inputs, p, base_text=base)
    for msg in imprint.notes(info, inputs, tgt):
        op.report({'INFO'}, msg)
    return info


class NPROTO_OT_project_curve(bpy.types.Operator):
    """Project the selected curves (and edges picked in Edge mode on other surfaces) onto the active surface
    or solid. Output Curve: new curves lying on it. Output Edge (Imprint): edges of its own, its faces divided
    where they land -- nothing is removed and a solid stays closed. Onto picks which faces take them: the
    nearest ones seen from the curve, the faces picked in Face mode, or all the way through"""
    bl_idname = "nproto.project_curve"
    bl_label = "Project / Imprint"
    bl_options = {'REGISTER', 'UNDO'}

    output: EnumProperty(name="Output", items=OUTPUT_ITEMS, default="CURVE")
    direction: EnumProperty(name="Direction", items=imprint.PROJECTIONS, default="WORKPLANE",
                            description="Curve: along the work plane normal unless set. Edge: along each sketch's "
                                        "normal (Sketch Normal) unless set")
    scope: EnumProperty(name="Onto", items=imprint.SCOPES, default="NEAREST")
    reach: EnumProperty(name="Reach", items=imprint.REACHES, default="BOTH")
    extend: BoolProperty(name="Extend to Edges", default=False,
                         description="An open curve that stops inside a face is carried on (its natural "
                                     "continuation) until it reaches the face's edges")
    view_dir: bpy.props.FloatVectorProperty(subtype='XYZ', options={'HIDDEN'}, default=(0, 0, -1))

    def invoke(self, context, event):
        from . import subsel
        rv3d = getattr(context, "region_data", None)
        if rv3d is not None:
            self.view_dir = rv3d.view_rotation @ Vector((0, 0, -1))
        tgt = context.active_object
        if self.output == "EDGE" and tgt is not None and store.is_nurbs(tgt) and subsel.picks(tgt, "FACE"):
            self.scope = "FACES"                  # faces picked on the target: onto those
        return self.execute(context)

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "output", expand=True)
        lay.prop(self, "direction")
        if self.output == "EDGE":
            lay.prop(self, "scope")
            lay.prop(self, "reach")
            lay.prop(self, "extend")

    def execute(self, context):
        from . import subsel
        surf = context.active_object
        if not (_kernel_ok(self) and store.is_nurbs(surf)):
            self.report({'WARNING'}, "Select curves, then the surface (active)")
            return {'CANCELLED'}
        if self.output == "EDGE" and not self.properties.is_property_set("direction"):
            self.direction = "CURVE"              # Imprint: a sketch drawn on a face goes along its normal
        inputs = _project_inputs(context, surf)
        if not inputs:
            self.report({'WARNING'}, "Select curves (or pick edges of other surfaces in Edge mode), "
                                     "then the surface (active)")
            return {'CANCELLED'}
        d = imprint.direction(self.direction, self.view_dir, imprint.plane_direction(context.scene))
        along = {"CURVE": "the sketch normal", "WORKPLANE": "the work plane normal", "VIEW": "the view",
                 "SURFACE": "the curve itself"}.get(self.direction, "World " + self.direction)
        if self.output == "CURVE":
            if self.direction == "SURFACE":
                self.report({'WARNING'}, "Project: On Surface makes edges, not curves -- set Output to Edge")
                return {'FINISHED'}
            try:
                c = _new_curves(context, "Projected", "project_curve", inputs + [(surf, "SURF")],
                                {"dir": list(d.normalized()), "projection": self.direction,
                                 "view": list(Vector(self.view_dir))})
            except sweeps.ProjectMiss:
                # FINISHED with nothing made keeps Adjust Last Operation open, so
                # the direction can be changed right there.
                self.report({'WARNING'}, f"Project: the {'curve misses' if len(inputs) == 1 else 'curves miss'} "
                                         f"the surface along {along}; pick another Direction "
                                         "in Adjust Last Operation")
                return {'FINISHED'}
            except Exception as exc:
                self.report({'ERROR'}, f"Project failed: {exc}")
                return {'CANCELLED'}
            _select_only(context, [c])
            return {'FINISHED'}
        faces = imprint.face_points(surf, subsel.picks(surf, "FACE")) if self.scope == "FACES" else []
        if self.scope == "FACES" and not faces:
            self.report({'WARNING'}, "Imprint: Onto Picked Faces, but no faces are picked on "
                                     f"{surf.name} (Face mode) -- pick them, or set Onto to Nearest Faces")
            return {'FINISHED'}
        p = imprint.local_params(surf, self.direction, self.view_dir, d, self.reach, self.scope, self.extend,
                                 faces)
        info = imprint_object(self, context, surf, inputs, p)
        if info is None:
            self.report({'WARNING'}, f"Imprint: nothing was imprinted along {along}; change Direction or Onto "
                                     "in Adjust Last Operation")
            return {'FINISHED'}
        n = int(info.get("edges", 0))
        self.report({'INFO'}, f"Imprint: {n} edge{'s' if n != 1 else ''} on {surf.name}")
        _select_only(context, [surf])
        return {'FINISHED'}


class NPROTO_OT_intersect(bpy.types.Operator):
    """Where two selected surfaces meet. Output Curve: intersection curves. Output Edge (Imprint): edges of
    the surfaces themselves along the intersection, their faces divided there -- nothing is removed"""
    bl_idname = "nproto.intersect"
    bl_label = "Intersect"
    bl_options = {'REGISTER', 'UNDO'}

    output: EnumProperty(name="Output", items=OUTPUT_ITEMS, default="CURVE")
    onto: EnumProperty(name="Onto", items=[("BOTH", "Both", "Each surface takes the edges where the other "
                                                            "meets it"),
                                           ("ACTIVE", "Active", "Only the active surface")], default="BOTH")

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "output", expand=True)
        if self.output == "EDGE":
            lay.prop(self, "onto")

    def execute(self, context):
        from . import jobs
        sel = [o for o in context.selected_objects if store.is_nurbs(o)]
        if not (_kernel_ok(self) and len(sel) == 2):
            self.report({'WARNING'}, "Select two NURBS surfaces")
            return {'CANCELLED'}
        if self.output == "CURVE":
            try:
                c = _new_curves(context, "Intersection", "intersect", [(sel[0], "SURF"), (sel[1], "SURF")], {})
            except Exception as exc:
                self.report({'ERROR'}, f"Intersect failed: {exc}")
                return {'CANCELLED'}
            _select_only(context, [c])
            return {'FINISHED'}
        act = context.active_object if context.active_object in sel else sel[0]
        pairs = [(act, sel[1] if act == sel[0] else sel[0])]
        if self.onto == "BOTH":
            pairs.append((pairs[0][1], act))
        # both from the shapes as they are, then applied
        done = []
        for tgt, other in pairs:
            p = imprint.local_params(tgt, "CURVE", (0, 0, -1), (0, 0, -1), "BOTH", "THROUGH", False)
            try:
                pre = imprint.run(tgt, [(other, "SURF")], p)
            except jobs.JobError as exc:
                self.report({'WARNING'}, f"Intersect on {tgt.name}: {exc}")
                continue
            done.append((tgt, other, p, pre))
        made = [tgt for tgt, other, p, pre in done
                if imprint_object(self, context, tgt, [(other, "SURF")], p, pre) is not None]
        if not made:
            self.report({'ERROR'}, "Intersect: the surfaces do not meet -- nothing was imprinted")
            return {'FINISHED'}
        self.report({'INFO'}, "Intersect: imprinted on " + ", ".join(o.name for o in made))
        _select_only(context, made)
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Rebuild Surface, Continuity Check
# --------------------------------------------------------------------------

class NPROTO_OT_extend(ValueDrag, AskValue, PickMixin, bpy.types.Operator):
    """Extend (or with a negative distance shrink) a surface at one boundary edge,
    continuing its own shape exactly (Alias "Natural" extend). Pick the edge,
    then drag the arrow on it (back into the surface: shrink), type a value,
    click away or Enter to make it. A trimmed surface extends at the edges
    that still lie on its own boundary; its trim edges cannot be extended"""
    bl_idname = "nproto.extend"
    bl_label = "Extend Surface"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 1
    pick_max = 1
    finish_on_min = True
    start_value = 0.0      # the arrow stands on the edge: nothing is extended until it is dragged
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    distance: FloatProperty(name="Distance", default=0.5, soft_min=-10.0, soft_max=10.0, unit='LENGTH',
                            description="How far the edge moves along the surface (negative: shrink)")

    # -- picking: the edges it can take, and why the others are refused ------
    def pick_candidates(self, context):
        from . import extend
        ok, self._refused, self._why_of = [], [], {}
        for c in super().pick_candidates(context):
            if c[0].type == 'CURVE':
                continue
            try:
                why = extend.refusal(c[0], c[1])
            except Exception as exc:
                why = f"Cannot extend this edge ({exc})"
            if why:
                self._refused.append(c)
                self._why_of[(c[0].name, c[1])] = why
            else:
                ok.append(c)
        return ok

    def start_picking(self, context, event=None):
        if not self.pick_candidates(context) and getattr(self, "_refused", None):
            self.report({'WARNING'}, "Nothing to extend. " + self._why_of[(self._refused[0][0].name,
                                                                          self._refused[0][1])])
            return {'CANCELLED'}
        self._why = ""
        return super().start_picking(context, event)

    def _update_hover(self, context, event):
        from .ops import _near_candidates
        cur = super()._update_hover(context, event)
        why, line = "", None
        if cur is None and getattr(self, "_refused", None):
            near = _near_candidates(context, event, self._refused)
            if near:
                why, line = self._why_of[(near[0][0].name, near[0][1])], near[0][2]
        draw.pick_overlay["refused"] = line
        if why != getattr(self, "_why", ""):
            self._why = why
            self._status(context)
            context.area.tag_redraw()
        return cur

    def tool_key(self, context, event):
        if event.type == 'LEFTMOUSE' and getattr(self, "_why", "") and self._in_view(context, event):
            if self._update_hover(context, event) is None and self._why:
                self.report({'WARNING'}, self._why)
                return True
        return False

    def pick_state(self):
        return self._why if getattr(self, "_why", "") else "click a boundary edge of a surface"

    def pick_rows(self):
        rows = super().pick_rows()
        rows.append(("", [hints.note("Edges on the surface's own boundary extend, on trimmed surfaces too. "
                                     "Trim edges (red) do not: Untrim or Refit to Trim first. "
                                     "Joined surfaces: Unjoin first")]))
        return rows

    def _end(self, context):
        draw.pick_overlay["refused"] = None
        super()._end(context)

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if self.inputs:                     # picked already (see PickMixin._finish)
            self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
            return self.ask(context)
        return self.start_picking(context, event)

    # -- the live value stage (valuedrag.py) ----------------------------------
    def drag_setup(self, context, mouse):
        from . import extend
        picks = _decode_inputs(self.inputs)
        if not picks:
            return False
        obj, ref = picks[0]
        r = extend.resolve(obj, ref)
        if r is None:
            return False
        edge, self._xat, (lo, hi), self._xtrimmed = r
        from .kernel import load
        self._ext = load().blender_ext
        surf = store.get_surfaces(obj)[0]
        self._xs, self._xobj, self._xedge, self._xrange = surf, obj, edge, (lo, hi)
        self._xface = store.get_brep(obj).faces()[0]
        self._M = geom.world_matrix(obj)
        self._scale = sum(self._M.to_scale()) / 3.0 or 1.0
        # the arrow: where the edge was clicked (else its middle), along the surface out of it
        (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
        across_u = edge in ("U0", "U1")
        at_end = edge in ("U1", "V1")

        def uv_at(f):
            a = lo + (hi - lo) * f
            if across_u:
                return (u1 if at_end else u0), a
            return a, (v1 if at_end else v0)
        self._xj, self._base_at_mouse = 16, False
        if mouse is not None:
            best = None
            for j in range(33):
                q = self._screen(context, self._M @ Vector(surf.evaluate(*uv_at(j / 32))))
                if q is not None and (q - mouse).length <= BASE_PX and (best is None or (q - mouse).length < best[0]):
                    best = ((q - mouse).length, j)
            if best is not None:
                self._xj, self._base_at_mouse = best[1], True
        self._xpts = [self._M @ Vector(surf.evaluate(*uv_at(j / 32))) for j in range(33)]   # the picked edge, world
        pt, du, dv = surf.derivatives(*uv_at(self._xj / 32))
        d = np.array(du if across_u else dv, float) * (1.0 if at_end else -1.0)
        M3 = np.array(self._M.to_3x3(), float)
        dw = M3 @ d
        self._xB = np.array(self._M @ Vector(pt), float)
        self._xD = dw / (np.linalg.norm(dw) or 1.0)
        self._xcache = None
        return True

    def drag_on_input(self, context, mouse):
        return self._near_screen(context, self._xpts, mouse)

    def _strip(self):
        """(rows of world points from the old edge to the new one, the new edge
        surface's arrow path): the band the extension adds (or, shrinking, cuts)."""
        from .kernel import trimext
        key = round(float(self.distance), 12)
        if self._xcache is not None and self._xcache[0] == key:
            return self._xcache[1]
        dist = float(self.distance) / self._scale
        edge = self._xedge
        across_u = edge in ("U0", "U1")
        new = trimext.extended_surface(self._xs, edge, dist)      # same parameters as the old part
        c_old, c_new = trimext._boundary_value(self._xs, edge), trimext._boundary_value(new, edge)
        big = new if dist >= 0 else self._xs           # the surface the band lies on
        a, b = (c_old, c_new) if dist >= 0 else (c_new, c_old)    # inner end first, out to big's edge
        lo, hi = self._xrange
        if dist < 0 and self._xtrimmed:
            # cutting back a trimmed surface: the band spans what the new edge line crosses of the face
            (u0, u1), (v0, v1) = self._xs.u_domain(), self._xs.v_domain()
            A0, A1 = (v0, v1) if across_u else (u0, u1)
            ins = [A0 + (A1 - A0) * k / 64 for k in range(65)]
            ins = [t for t in ins if self._xface.classify_uv(*((c_new, t) if across_u else (t, c_new))) != "outside"]
            if ins:
                lo, hi = min(ins + [lo]), max(ins + [hi])

        def P(x, t):
            return big.evaluate(x, t) if across_u else big.evaluate(t, x)
        grid = [[P(a + (b - a) * i / 8, lo + (hi - lo) * j / 32) for j in range(33)] for i in range(9)]
        M = np.array(self._M, float)
        rows = [np.asarray(r, float) @ M[:3, :3].T + M[:3, 3] for r in grid]
        path = np.array([r[self._xj] for r in rows])
        self._xcache = (key, (rows, path))
        return rows, path

    def drag_handles(self):
        h = {"field": "distance", "kind": "arrow", "origin": self._xB, "dir": self._xD, "param": self.distance,
             "base": self._xB, "to_value": lambda t: t}
        try:
            rows, path = self._strip()
            if self.distance >= 0:
                h["pos"], h["path"] = path[-1], path
                h["head_from"] = path[-2]
            else:                             # shrinking: the arrow points back into the surface
                h["pos"], h["path"] = path[0], path[::-1]
                h["head_from"] = path[1]
        except Exception as exc:
            print("NURBS extend preview:", exc)
        return [h]

    def drag_preview(self):
        rows, _ = self._strip()
        lines, tris = ruled_preview(rows, ribs=8, row_every=8)
        new_edge = rows[-1] if self.distance >= 0 else rows[0]
        return {"lines": lines + [new_edge], "tris": tris}

    def drag_check(self):
        return "" if abs(self.distance) > 1e-12 else "Distance 0: drag the arrow out, or type a value"

    def drag_rows(self):
        return [("Shrink", [hints.note("drag the arrow back into the surface, or type a negative value")])]

    def execute(self, context):
        from . import extend
        picks = _decode_inputs(self.inputs)
        if not picks:
            self.report({'WARNING'}, "Pick a boundary edge of a surface")
            return {'CANCELLED'}
        obj, ref = picks[0]
        why = extend.refusal(obj, ref) if store.is_nurbs(obj) else "Pick a boundary edge of a surface"
        r = None if why else extend.resolve(obj, ref)
        if r is None:
            self.report({'WARNING'}, why or extend.TRIM_EDGE)
            return {'CANCELLED'}
        edge, at, _, trimmed = r
        scale = sum(geom.world_matrix(obj).to_scale()) / 3.0 or 1.0
        p = {"edge": edge, "distance": float(self.distance) / scale}
        if at is not None:
            p["at"] = float(at)
        try:
            if features.history_enabled(context):
                features.push_step(obj, "extend_surface", [], p)
                features.rebuild(obj, context, force=True)
            else:
                store.set_geometry_text(obj, extend.extend_surface_text(obj[store.PROP_BREP], edge, p["distance"], at),
                                        context, keep_untrimmed=False)
                obj[store.PROP_NATURAL] = 1 if extend.result_natural(obj) else 0
        except Exception as exc:
            if features.history_enabled(context):
                try:
                    features.remove_step(obj, len(features.stack(obj)) - 1)
                except Exception:
                    pass
            self.report({'ERROR'}, f"Extend failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        return {'FINISHED'}


class NPROTO_OT_refit_trim(bpy.types.Operator):
    """Turn a trimmed four-sided surface into an untrimmed one whose edges are the
    trim boundary, keeping the original shape (CVs fitted into the boundary)"""
    bl_idname = "nproto.refit_trim"
    bl_label = "Refit to Trim"
    bl_options = {'REGISTER', 'UNDO'}

    refit_degree: IntProperty(name="Degree", default=3, min=1, max=7)
    refit_spans_u: IntProperty(name="Spans U", default=0, min=0, max=32, description="0 = auto (fewest within 1e-4 of the size)")
    refit_spans_v: IntProperty(name="Spans V", default=0, min=0, max=32, description="0 = auto (fewest within 1e-4 of the size)")

    def draw(self, context):
        from . import refit
        lay = self.layout
        lay.prop(self, "refit_degree")
        row = lay.row(align=True)
        row.prop(self, "refit_spans_u")
        row.prop(self, "refit_spans_v")
        inf = refit.LAST_INFO
        if inf:
            lay.label(text=f"{inf['cvs'][0]} x {inf['cvs'][1]} CVs · spans {inf['spans'][0]} x {inf['spans'][1]} · "
                           f"deviation {inf['deviation']:.1e}", icon='MESH_GRID')

    def execute(self, context):
        from . import refit
        if not _kernel_ok(self):
            return {'CANCELLED'}
        objs = [o for o in context.selected_objects if store.is_nurbs(o)]
        trimmed = [o for o in objs if not store.is_natural(o) and store.get_brep(o).face_count() == 1]
        if not trimmed:
            self.report({'WARNING'}, "Select a trimmed single surface (four-sided trim boundary)")
            return {'CANCELLED'}
        p = {"refit_degree": self.refit_degree, "refit_spans_u": self.refit_spans_u, "refit_spans_v": self.refit_spans_v}
        done = []
        for o in trimmed:
            try:
                if features.history_enabled(context):
                    features.push_step(o, "refit_trim", [], p)
                    features.rebuild(o, context, force=True)
                else:
                    text, info = refit.refit_text(o[store.PROP_BREP], self.refit_degree, self.refit_spans_u,
                                                  self.refit_spans_v)
                    refit.LAST_INFO.clear()
                    refit.LAST_INFO.update(info)
                    store.set_geometry_text(o, text, context, keep_untrimmed=False)
                    o[store.PROP_NATURAL] = 1
                done.append(o)
            except Exception as exc:
                if features.history_enabled(context):
                    try:
                        st = features.stack(o)
                        if st and st[-1]["op"] == "refit_trim":
                            features.remove_step(o, len(st) - 1)
                    except Exception:
                        pass
                self.report({'ERROR'}, f"{o.name}: {exc}")
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}


MERGE_CONT = [("FREE", "Free", "The edge may move (plain fit)"),
              ("G0", "G0", "Keep the edge where it is"),
              ("G1", "G1", "Keep the edge and the tangent plane across it"),
              ("G2", "G2", "Keep the edge, the tangent plane and the curvature across it")]
MERGE_SIDE_NAMES = (("u0", "U start"), ("u1", "U end"), ("v0", "V start"), ("v1", "V end"))


def _merge_set_all(self, context):
    for s_, _ in MERGE_SIDE_NAMES:
        setattr(self, f"merge_cont_{s_}", self.merge_edges)


MERGE_MODES = [
    ("EXACT", "Exact", "Keep every CV: identical to the patches, seams stay as knot lines"),
    ("REDUCE", "Reduce", "Remove seam and span knots while the surface stays within the tolerance (keeps the degree)"),
    ("REFIT", "Refit", "Fewest spans of the chosen degree within the tolerance (or the spans given)"),
]

MERGE_SEAMS = [
    ("KEEP", "Keep Kinks", "Seams where the patches meet with a kink (or a curvature jump) stay sharp: "
                           "Refit puts a full knot there"),
    ("SMOOTH", "Smooth", "Refit runs smoothly across every seam (a kink is rounded off within the tolerance)"),
]

MERGE_KNOTS = [
    ("ADAPTIVE", "Adaptive", "Move the knots to where the surface needs them: fewer spans for the same tolerance"),
    ("UNIFORM", "Uniform", "Evenly spaced knots"),
]


class NPROTO_OT_merge_patches(bpy.types.Operator):
    """Rebuild joined faces whose outer boundary is four-sided as ONE untrimmed surface.
    A grid of untrimmed patches merges exactly; anything else is refitted over the outer boundary"""
    bl_idname = "nproto.merge_patches"
    bl_label = "Merge Patches"
    bl_options = {'REGISTER', 'UNDO'}

    merge_mode: EnumProperty(name="Match", items=MERGE_MODES, default="REDUCE")
    merge_tolerance: FloatProperty(name="Tolerance", default=0.0, min=0.0, soft_max=0.01, precision=5, unit='LENGTH',
                                   description="Largest allowed distance from the patches (0 = 1e-4 of the size)")
    merge_degree: IntProperty(name="Degree", default=3, min=1, max=7)
    merge_spans_u: IntProperty(name="Spans U (0 = auto)", default=0, min=0, max=32,
                               description="Pieces between knots along U; 1 = one piece with no inner knot, "
                                           "0 = the fewest within the tolerance")
    merge_spans_v: IntProperty(name="Spans V (0 = auto)", default=0, min=0, max=32,
                               description="Pieces between knots along V; 1 = one piece with no inner knot, "
                                           "0 = the fewest within the tolerance")
    merge_knots: EnumProperty(name="Knots", items=MERGE_KNOTS, default="ADAPTIVE",
                              description="Where Refit puts its knots")
    merge_reduce_degree: BoolProperty(name="Reduce Degree", default=False,
                                      description="Also lower the degree (u and v on their own, e.g. 6 to 3) while "
                                                  "the surface and the kept edges stay within their tolerances")
    merge_keep_original: BoolProperty(name="Keep Original", default=False,
                                      description="Merge into a new object and keep the joined patches (hidden) "
                                                  "to compare against")
    merge_show_dev: BoolProperty(name="Deviation Map", default=True,
                                 description="Colour the result by its distance from the patches while this panel "
                                             "is open: blue below, green on, red above (the ramp spans "
                                             "minus to plus the tolerance; contour lines every tenth)")
    merge_seams: EnumProperty(name="Seams", items=MERGE_SEAMS, default="KEEP",
                              description="What Refit does at seams that are not smooth")
    merge_edges: EnumProperty(name="Edges", items=MERGE_CONT, default="G2", update=_merge_set_all,
                              description="What the outer edges keep from the patches, on all four sides. "
                                          "Keeping their tangents and curvature keeps the continuity they had "
                                          "with neighbouring surfaces")
    merge_cont_u0: EnumProperty(name="U start", items=MERGE_CONT, default="G2")
    merge_cont_u1: EnumProperty(name="U end", items=MERGE_CONT, default="G2")
    merge_cont_v0: EnumProperty(name="V start", items=MERGE_CONT, default="G2")
    merge_cont_v1: EnumProperty(name="V end", items=MERGE_CONT, default="G2")
    merge_per_side: BoolProperty(name="Each Side", default=False, description="Set the edge continuity per side")
    merge_edge_tolerance: FloatProperty(name="Edge Tolerance", default=0.0, min=0.0, soft_max=0.001, precision=6,
                                        unit='LENGTH', description="Largest allowed gap at the kept edges "
                                                                   "(0 = a tenth of the tolerance)")

    def _params(self):
        p = {"merge_mode": self.merge_mode, "merge_tolerance": self.merge_tolerance, "merge_degree": self.merge_degree,
             "merge_spans_u": self.merge_spans_u, "merge_spans_v": self.merge_spans_v,
             "merge_edge_tolerance": self.merge_edge_tolerance, "merge_knots": self.merge_knots,
             "merge_seams": self.merge_seams, "merge_reduce_degree": self.merge_reduce_degree}
        for s_, _ in MERGE_SIDE_NAMES:
            p[f"merge_cont_{s_}"] = getattr(self, f"merge_cont_{s_}")
        return p

    def draw(self, context):
        from . import merge
        lay = self.layout
        inf = merge.LAST_INFO

        def dist(x):
            us = context.scene.unit_settings
            m = x * (us.scale_length or 1.0)
            if us.system == 'IMPERIAL':
                return f"{m / 0.0254:.5f}\""
            if us.system == 'METRIC':
                return f"{m * 1e6:.1f} µm" if m < 1e-4 else f"{m * 1e3:.4f} mm"
            return f"{x:.1e}"
        lay.prop(self, "merge_mode", expand=True)
        lay.prop(self, "merge_keep_original")
        if self.merge_mode != "EXACT":
            row = lay.row(align=True)
            row.prop(self, "merge_tolerance")
            if self.merge_tolerance <= 0 and inf:
                row.label(text=f"auto {dist(inf['tolerance'])}")
        refit = self.merge_mode == "REFIT" or (inf and inf.get("route") == "projected")
        if self.merge_mode == "REDUCE" and not refit:
            lay.prop(self, "merge_reduce_degree")
        if refit:
            lay.prop(self, "merge_degree")
            row = lay.row(align=True)
            row.prop(self, "merge_spans_u", text="Spans U")
            row.prop(self, "merge_spans_v", text="Spans V")
            lay.label(text="Spans: 0 = auto · 1 = one piece, no inner knot")
            lay.row(align=True).prop(self, "merge_knots", expand=True)
            if inf and any(r_["kind"] != "G2" for r_ in inf.get("seams", [])):
                lay.row(align=True).prop(self, "merge_seams", expand=True)
        if self.merge_mode != "EXACT":
            row = lay.row(align=True)
            row.prop(self, "merge_edges", expand=True)
            row.prop(self, "merge_per_side", text="", icon='MOD_EDGESPLIT')
            if self.merge_per_side:
                col = lay.column(align=True)
                for s_, name in MERGE_SIDE_NAMES:
                    r_ = col.row(align=True)
                    r_.label(text=name)
                    r_.prop(self, f"merge_cont_{s_}", expand=True)
            row = lay.row(align=True)
            row.prop(self, "merge_edge_tolerance", text="Edge Tolerance")
            if self.merge_edge_tolerance <= 0 and inf:
                row.label(text=f"auto {dist(inf['edge_tolerance'])}")
        if inf:
            if inf.get("route") == "grid":
                lay.label(text=f"{inf['grid'][0]} x {inf['grid'][1]} patch grid · exact merge "
                               f"{inf['exact_cvs'][0]} x {inf['exact_cvs'][1]} CVs", icon='CHECKMARK')
                seams = inf.get("seams", [])
                if seams:
                    kinks = [r_ for r_ in seams if r_["kind"] == "KINK"]
                    g1 = [r_ for r_ in seams if r_["kind"] == "G1"]
                    parts = []
                    if kinks:
                        parts.append(f"{len(kinks)} kink{'s' if len(kinks) > 1 else ''} "
                                     f"{max(r_['angle'] for r_ in kinks):.2f}°")
                    if g1:
                        parts.append(f"{len(g1)} G1 (curvature {100 * max(r_['curv'] for r_ in g1):.0f}%)")
                    n2 = len(seams) - len(kinks) - len(g1)
                    if n2:
                        parts.append(f"{n2} G2")
                    txt = "seams " + " · ".join(parts)
                    if kinks or g1:
                        if self.merge_mode == "REFIT":
                            txt += " · kept sharp" if inf.get("seams_kept") else " · smoothed"
                        elif self.merge_mode == "EXACT":
                            txt += " · kept"
                    lay.label(text=txt, icon='ERROR' if kinks else ('INFO' if g1 else 'CHECKMARK'))
                if inf.get("note"):
                    lay.label(text=inf["note"], icon='INFO')
            else:
                lay.label(text=f"Refitted over the outer boundary ({inf.get('note', '')})", icon='INFO')
            su, sv = inf["knots_u"] + 1, inf["knots_v"] + 1
            tag = lambda fixed: "" if not refit else (" set" if fixed else " auto")
            lay.label(text=f"{inf['cvs'][0]} x {inf['cvs'][1]} CVs · degree {inf['degree'][0]}/{inf['degree'][1]} · "
                           f"spans {su}{tag(self.merge_spans_u)} x {sv}{tag(self.merge_spans_v)}", icon='MESH_GRID')
            over = self.merge_mode != "EXACT" and inf["deviation"] > inf["tolerance"] * (1 + 1e-6)
            lay.alert = over
            lay.label(text=f"deviation {dist(inf['deviation'])}" + (f" > tolerance {dist(inf['tolerance'])}" if over else ""),
                      icon='ERROR' if over else 'CHECKMARK')
            lay.alert = False
            row = lay.row(align=True)
            row.prop(self, "merge_show_dev")
            if self.merge_show_dev and "devmap_max" in inf:
                out_ = inf.get("devmap_outside", 0.0)
                row.label(text=f"max {dist(inf['devmap_max'])}" + (f" · {100 * out_:.1f}% over" if out_ > 0 else ""))
                lay.label(text=f"blue -{dist(inf['tolerance'])} · green on · red +{dist(inf['tolerance'])}",
                          icon='BLANK1')
            cont = inf.get("continuity", {})
            kept = [s_ for s_ in ("u0", "u1", "v0", "v1") if cont.get(s_, -1) >= 0]
            if self.merge_mode != "EXACT" and kept:
                w = inf.get("edge_worst", {})
                txt = f"edges gap {dist(w.get('pos', 0.0))}"
                if any(cont[s_] >= 1 for s_ in kept):
                    txt += f" · angle {w.get('angle', 0.0):.3f}°"
                if any(cont[s_] >= 2 for s_ in kept):
                    txt += f" · curvature {100 * w.get('curv', 0.0):.1f}%"
                bad = not inf.get("edges_ok", True)
                lay.alert = bad
                lay.label(text=txt + (" · over the edge checks" if bad else ""), icon='ERROR' if bad else 'CHECKMARK')
                lay.alert = False
            elif self.merge_mode != "EXACT":
                lay.label(text="edges free: they may move by up to the deviation", icon='INFO')

    def execute(self, context):
        from . import merge
        if not _kernel_ok(self):
            return {'CANCELLED'}
        objs = [o for o in context.selected_objects if store.is_nurbs(o) and store.get_brep(o).face_count() > 1]
        if not objs:
            self.report({'WARNING'}, "Select joined surfaces (two or more faces, four-sided outer boundary)")
            return {'CANCELLED'}
        p = self._params()
        done = []
        merge.WANT_DEVMAP[0] = bool(self.merge_show_dev)
        try:
            self._run(context, objs, p, done)
        finally:
            merge.WANT_DEVMAP[0] = False
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}

    def _run(self, context, objs, p, done):
        from . import merge
        for o in objs:
            if self.merge_keep_original:       # merge a plain copy; the joined patches stay as they are, hidden
                src = o
                o = store.new_object_text(context, f"{src.name} Merged", src[store.PROP_BREP], src.matrix_world.copy())
                for mat in src.data.materials:
                    o.data.materials.append(mat)
                src.hide_set(True)
            try:
                if features.history_enabled(context):
                    features.push_step(o, "merge_patches", [], p)
                    features.rebuild(o, context, force=True)
                else:
                    text = merge.apply_merge(o, o[store.PROP_BREP], [], p)
                    store.set_geometry_text(o, text, context, keep_untrimmed=False)
                    o[store.PROP_NATURAL] = 1
                done.append(o)
            except Exception as exc:
                if features.history_enabled(context):
                    try:
                        st = features.stack(o)
                        if st and st[-1]["op"] == "merge_patches":
                            features.remove_step(o, len(st) - 1)
                    except Exception:
                        pass
                self.report({'ERROR'}, f"{o.name}: {exc}")
                if self.merge_keep_original:      # no merge: drop the copy, show the patches again
                    bpy.data.objects.remove(o)
                    src.hide_set(False)


class NPROTO_OT_rebuild_surface(bpy.types.Operator):
    """Refit untrimmed surfaces with a chosen degree and number of spans (uniform knots)"""
    bl_idname = "nproto.rebuild_surface"
    bl_label = "Rebuild Surface"
    bl_options = {'REGISTER', 'UNDO'}

    degree_u: IntProperty(name="Degree U", default=3, min=1, max=7)
    degree_v: IntProperty(name="Degree V", default=3, min=1, max=7)
    spans_u: IntProperty(name="Spans U", default=4, min=1, max=64)
    spans_v: IntProperty(name="Spans V", default=4, min=1, max=64)

    def draw(self, context):
        lay = self.layout
        col = lay.column(align=True)
        col.prop(self, "degree_u")
        col.prop(self, "degree_v")
        col = lay.column(align=True)
        col.prop(self, "spans_u")
        col.prop(self, "spans_v")
        lay.label(text=f"Result: {self.degree_u + self.spans_u} x {self.degree_v + self.spans_v} CVs "
                       f"({self.spans_u} x {self.spans_v} spans)", icon='MESH_GRID')
        lay.label(text="Viewport: amber = span isoparms, purple = CV hull", icon='HIDE_OFF')

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        done = 0
        for o in [o for o in context.selected_objects if store.is_nurbs(o)]:
            if not store.is_natural(o) or store.get_brep(o).face_count() != 1:
                self.report({'WARNING'}, f"{o.name}: only untrimmed single surfaces can be rebuilt")
                continue
            p = {"degree_u": self.degree_u, "degree_v": self.degree_v, "spans_u": self.spans_u, "spans_v": self.spans_v}
            try:
                if features.history_enabled(context):
                    features.push_step(o, "rebuild_surface", [], p)
                    features.rebuild(o, context, force=True)
                    o[store.PROP_NATURAL] = 1
                else:
                    store.set_geometry_text(o, sweeps.rebuild_surface_text(o[store.PROP_BREP], self.degree_u,
                                                                           self.degree_v, self.spans_u, self.spans_v),
                                            context, keep_untrimmed=False)
                    o[store.PROP_NATURAL] = 1
                done += 1
            except Exception as exc:
                self.report({'ERROR'}, f"{o.name}: {exc}")
        return {'FINISHED'} if done else {'CANCELLED'}


def continuity_report(a, b, samples=60):
    """Exact (kernel) max gap and max normal angle where surface a's edges
    touch surface b."""
    from .sweeps import _normal_near
    # in double precision: mathutils is float32, ~6e-5 off at 1000 m from the
    # origin -- a "gap" that is only rounding, and a G2 blend's curvature read
    # that far inside it (where it is already rising) failed the check
    Ma, Mb = (np.array(geom.world_matrix(o), dtype=float) for o in (a, b))
    Mai, Mbi = np.linalg.inv(Ma), np.linalg.inv(Mb)
    Ra, Rb = Ma[:3, :3], Mb[:3, :3]
    xf = lambda M, p: M[:3, :3] @ np.asarray(p, float) + M[:3, 3]
    ba, bb = store.get_brep(a), store.get_brep(b)
    per_edge = []
    for e in ba.edges():
        t0, t1 = e.parameter_range()
        pts = []
        for k in range(1, samples):
            try:
                pts.append(xf(Ma, e.point_at(t0 + (t1 - t0) * k / samples)))
            except Exception:
                pass
        if pts:
            per_edge.append(pts)
    if not per_edge:
        return None
    allp = np.array([p for pts in per_edge for p in pts])
    span = float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0
    near = []
    for pts in per_edge:          # only edges that run ALONG the other surface count
        hits = []
        for p in pts:
            pl = tuple(float(x) for x in xf(Mbi, p))
            fb, (q, d) = min(((f, f.closest_point(pl)) for f in bb.faces()), key=lambda x: x[1][1])
            hits.append((p, d, fb, np.array(q)))
        if sum(1 for h in hits if h[1] < 0.01 * span) >= 0.8 * len(hits):
            near += hits
    if not near:
        return None
    gap, ang = 0.0, 0.0
    curv = []                 # (relative difference, k_a, k_b) of the curvature across the edge
    for idx, (p, d, fb, q) in enumerate(near):
        pa = xf(Mai, p)
        fa = min(ba.faces(), key=lambda f: f.closest_point(tuple(pa))[1])
        na = Ra @ _normal_near(fa.surface(), np.array(fa.closest_point(tuple(pa))[0]))
        nb = Rb @ _normal_near(fb.surface(), q)
        c = abs(float(na @ nb) / ((np.linalg.norm(na) * np.linalg.norm(nb)) or 1.0))
        gap = max(gap, d)
        ang = max(ang, math.degrees(math.acos(min(1.0, c))))
        # G2: normal curvature of both faces in the direction across the edge
        prev, nxt = near[max(idx - 1, 0)][0], near[min(idx + 1, len(near) - 1)][0]
        T = np.asarray(nxt, float) - np.asarray(prev, float)
        if np.linalg.norm(T) == 0:
            continue
        try:
            ka, nwa = _cross_curvature(fa.surface(), np.array(fa.closest_point(tuple(pa))[0]), Ra, T)
            kb, nwb = _cross_curvature(fb.surface(), q, Rb, T)
        except Exception:
            continue
        if nwa @ nwb < 0:
            kb = -kb
        floor = 1.0 / span            # curvatures below "flat over the whole part" count as equal
        curv.append((abs(ka - kb) / max(abs(ka), abs(kb), floor), ka, kb))
    out = {"points": len(near), "gap": gap, "angle": ang}
    if curv:
        worst = max(curv, key=lambda x: x[0])
        out.update(curv_diff=worst[0], k_a=worst[1], k_b=worst[2])
    return out


def _cross_curvature(surf, q, R, T):
    """Normal curvature (world space) of surf at its point q (local) in the
    direction across the edge running along T (world). Returns (k, normal):
    k > 0 when the surface bends towards that normal."""
    from .square import _foot, _second
    from .kernel import load
    try:        # started at the kernel's global foot: a grid start can sit across a closed surface's seam
        _, uvs, _ = load().surface_closest_point(surf, tuple(float(x) for x in q))
        uv0 = (uvs[0].x, uvs[0].y) if hasattr(uvs[0], "x") else (uvs[0][0], uvs[0][1])
    except Exception:
        uv0 = None
    u, v = _foot(surf, np.asarray(q, float), uv0)
    su, sv, suu, suv, svv = (R @ x for x in _second(surf, u, v))
    n = np.cross(su, sv)
    n /= np.linalg.norm(n) or 1.0
    w = np.cross(n, T)
    w /= np.linalg.norm(w) or 1.0
    a, b = np.linalg.lstsq(np.array([su, sv]).T, w, rcond=None)[0]
    du = a * su + b * sv
    k = (a * a * (suu @ n) + 2 * a * b * (suv @ n) + b * b * (svv @ n)) / max(du @ du, 1e-300)
    return float(k), n


classes = (NPROTO_OT_extend, NPROTO_OT_refit_trim, NPROTO_OT_merge_patches, NPROTO_OT_square, NPROTO_OT_sweep, NPROTO_OT_pipe, NPROTO_OT_boundary, NPROTO_OT_iso_curve, NPROTO_OT_dup_edge,
           NPROTO_OT_project_curve, NPROTO_OT_intersect, NPROTO_OT_rebuild_surface)
