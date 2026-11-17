"""Operators: Sweep, Pipe, Boundary, Iso Curve, Duplicate Edge, Project Curve,
Intersect, Rebuild Surface, Continuity Check."""

import json
import math

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from mathutils import Vector

from . import features, geom, store, sweeps
from . import hints
from .kernel import load as load_kernel
from .ops import AskValue, PickMixin, _decode_inputs, _encode_inputs, _kernel_ok, _select_only
from .valuedrag import BASE_PX, ValueDrag, ruled_preview


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

class NPROTO_OT_sweep(PickMixin, bpy.types.Operator):
    """Sweep a profile along one rail, two rails (Shift+click the 1st rail), or a
    surface edge (tangent to that surface). Select the profile, then the path
    (last); or with nothing selected click the profile, then the path"""
    bl_idname = "nproto.sweep"
    bl_label = "Sweep"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 2
    pick_max = 3

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    samples: IntProperty(name="Samples", default=32, min=6, max=300, description="Frames along the rail")
    tangent: BoolProperty(name="Tangent to Surface", default=True,
                          description="Rail is a surface edge: make the sweep G1 to that surface")
    flip_tangent: BoolProperty(name="Flip Tangent", default=False)
    scale_end: FloatProperty(name="End Scale", default=1.0, min=0.01, soft_max=5.0)
    twist: FloatProperty(name="Twist", default=0.0, soft_min=-720.0, soft_max=720.0)
    keep_height: BoolProperty(name="Keep Height (2 rails)", default=False,
                              description="Stretch the profile between the rails without scaling its height")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        crv = [o for o in context.selected_objects if o.type == 'CURVE']
        act = context.active_object
        if len(crv) in (2, 3) and act in crv and not event.alt:
            prof = sweep_profile(crv, act)
            rails = [o for o in crv if o != prof]
            rails.sort(key=lambda o: o == act)          # the path selected last
            self.inputs = _encode_inputs([(prof, "S0")] + [(o, "S0") for o in rails])
            return self.execute(context)
        return self.start_picking(context, event)

    def pick_state(self):
        n = len(self._picks)
        return ["click the profile", "click the path (Shift+click: a first rail of two)",
                "click the second rail"][min(n, 2)]

    def _params(self):
        return {k: getattr(self, k) for k in ("samples", "tangent", "flip_tangent", "scale_end", "twist", "keep_height")}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            inputs = _decode_inputs(self.inputs)
            if len(inputs) < 2:
                raise ValueError("pick a profile and a rail (Shift+click for a second rail)")
            obj = _new_surface(context, "Sweep", "sweep", inputs, self._params())
        except Exception as exc:
            self.report({'ERROR'}, f"Sweep failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        return {'FINISHED'}


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


class NPROTO_OT_pipe(AskValue, PickMixin, bpy.types.Operator):
    """Sweep an exact circle along curves (selected, or pick one)"""
    bl_idname = "nproto.pipe"
    bl_label = "Pipe"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 1
    pick_max = 1
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    radius: FloatProperty(name="Radius", default=0.1, min=1e-4, unit='LENGTH')
    samples: IntProperty(name="Samples", default=32, min=6, max=300)

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if self.inputs:                     # picked already (see PickMixin._finish)
            return self.ask(context)
        crv = [o for o in context.selected_objects if o.type == 'CURVE']
        if crv and not event.alt:
            self.inputs = _encode_inputs([(o, "S0") for o in crv])
            return self.ask(context)
        return self.start_picking(context, event)

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        made = []
        try:
            for inp in _decode_inputs(self.inputs):
                made.append(_new_surface(context, "Pipe", "pipe", [inp], {"radius": self.radius, "samples": self.samples}))
        except Exception as exc:
            self.report({'ERROR'}, f"Pipe failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, made)
        return {'FINISHED'} if made else {'CANCELLED'}


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


class NPROTO_OT_project_curve(bpy.types.Operator):
    """Project the selected curves onto the active surface (curves on the surface)"""
    bl_idname = "nproto.project_curve"
    bl_label = "Project Curve"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(name="Direction", items=[("VIEW", "View", ""), ("X", "World X", ""), ("Y", "World Y", ""),
                                                     ("Z", "World Z", "")], default="VIEW")
    view_dir: bpy.props.FloatVectorProperty(subtype='XYZ', options={'HIDDEN'}, default=(0, 0, -1))

    def invoke(self, context, event):
        rv3d = getattr(context, "region_data", None)
        if rv3d is not None:
            self.view_dir = rv3d.view_rotation @ Vector((0, 0, -1))
        return self.execute(context)

    def execute(self, context):
        surf = context.active_object
        crv = [o for o in context.selected_objects if o.type == 'CURVE']
        if not (_kernel_ok(self) and store.is_nurbs(surf) and crv):
            self.report({'WARNING'}, "Select curves, then the surface (active)")
            return {'CANCELLED'}
        d = Vector(self.view_dir) if self.direction == "VIEW" else Vector(
            {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[self.direction])
        try:
            c = _new_curves(context, "Projected", "project_curve",
                            [(o, "S0") for o in crv] + [(surf, "SURF")], {"dir": list(d.normalized())})
        except Exception as exc:
            self.report({'ERROR'}, f"Project failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [c])
        return {'FINISHED'}


class NPROTO_OT_intersect(bpy.types.Operator):
    """Intersection curves of two selected surfaces"""
    bl_idname = "nproto.intersect"
    bl_label = "Intersect"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sel = [o for o in context.selected_objects if store.is_nurbs(o)]
        if not (_kernel_ok(self) and len(sel) == 2):
            self.report({'WARNING'}, "Select two NURBS surfaces")
            return {'CANCELLED'}
        try:
            c = _new_curves(context, "Intersection", "intersect", [(sel[0], "SURF"), (sel[1], "SURF")], {})
        except Exception as exc:
            self.report({'ERROR'}, f"Intersect failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [c])
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Rebuild Surface, Continuity Check
# --------------------------------------------------------------------------

class NPROTO_OT_extend(ValueDrag, AskValue, PickMixin, bpy.types.Operator):
    """Extend (or with a negative distance shrink) a surface at one boundary edge,
    continuing its own shape exactly (Alias "Natural" extend). Pick the edge,
    then drag the arrow on it (back into the surface: shrink), type a value,
    click away or Enter to make it"""
    bl_idname = "nproto.extend"
    bl_label = "Extend Surface"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 1
    pick_max = 1
    finish_on_min = True
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    distance: FloatProperty(name="Distance", default=0.5, soft_min=-10.0, soft_max=10.0, unit='LENGTH',
                            description="How far the edge moves along the surface (negative: shrink)")

    def pick_candidates(self, context):
        from .extend import EDGES
        return [c for c in super().pick_candidates(context)
                if c[0].type != 'CURVE' and c[1] in EDGES and store.is_natural(c[0])]

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if self.inputs:                     # picked already (see PickMixin._finish)
            self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
            return self.ask(context)
        return self.start_picking(context, event)

    # -- the live value stage (valuedrag.py) ----------------------------------
    def drag_setup(self, context, mouse):
        from .extend import EDGES
        picks = _decode_inputs(self.inputs)
        if not picks:
            return False
        obj, edge = picks[0]
        if edge not in EDGES or not store.is_natural(obj) or store.get_brep(obj).face_count() != 1:
            return False
        from .kernel import load
        self._ext = load().blender_ext
        surf = store.get_surfaces(obj)[0]
        self._xd = self._ext.surface_cvs(surf)
        self._xs, self._xobj, self._xedge = surf, obj, edge
        self._M = geom.world_matrix(obj)
        self._scale = sum(self._M.to_scale()) / 3.0 or 1.0
        # the arrow: where the edge was clicked (else its middle), along the surface out of it
        (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
        across_u = edge in ("U0", "U1")
        at_end = edge in ("U1", "V1")

        def uv_at(f):
            if across_u:
                return (u1 if at_end else u0), v0 + (v1 - v0) * f
            return u0 + (u1 - u0) * f, (v1 if at_end else v0)
        self._xj, self._base_at_mouse = 16, False
        if mouse is not None:
            best = None
            for j in range(33):
                q = self._screen(context, self._M @ Vector(surf.evaluate(*uv_at(j / 32))))
                if q is not None and (q - mouse).length <= BASE_PX and (best is None or (q - mouse).length < best[0]):
                    best = ((q - mouse).length, j)
            if best is not None:
                self._xj, self._base_at_mouse = best[1], True
        pt, du, dv = surf.derivatives(*uv_at(self._xj / 32))
        d = np.array(du if across_u else dv, float) * (1.0 if at_end else -1.0)
        M3 = np.array(self._M.to_3x3(), float)
        dw = M3 @ d
        self._xB = np.array(self._M @ Vector(pt), float)
        self._xD = dw / (np.linalg.norm(dw) or 1.0)
        self._xcache = None
        return True

    def _strip(self):
        """(rows of world points from the old edge to the new one, the new edge
        surface's arrow path): the band the extension adds (or, shrinking, cuts)."""
        from .extend import extend_grid
        key = round(float(self.distance), 12)
        if self._xcache is not None and self._xcache[0] == key:
            return self._xcache[1]
        sm = load_kernel()
        ext = self._ext
        dist = float(self.distance) / self._scale
        edge = self._xedge
        across_u = edge in ("U0", "U1")
        at_end = edge in ("U1", "V1")
        new = ext.surface_from_cvs(*extend_grid(self._xd, edge, dist))
        big = new if dist >= 0 else self._xs           # the surface the band lies on
        small = self._xs if dist >= 0 else new
        (bu0, bu1), (bv0, bv1) = big.u_domain(), big.v_domain()
        (su0, su1), (sv0, sv1) = small.u_domain(), small.v_domain()
        # the band's inner end: the small surface's edge, found on the big one
        sm_u = (su1 if at_end else su0) if across_u else 0.5 * (su0 + su1)
        sm_v = 0.5 * (sv0 + sv1) if across_u else (sv1 if at_end else sv0)
        q = small.evaluate(sm_u, sm_v)
        _, uvs, _ = sm.surface_closest_point(big, tuple(q))
        uv = uvs[0]
        cu, cv = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
        if across_u:
            a, b = cu, (bu1 if at_end else bu0)
            grid = [[big.evaluate(a + (b - a) * i / 8, bv0 + (bv1 - bv0) * j / 32) for j in range(33)] for i in range(9)]
        else:
            a, b = cv, (bv1 if at_end else bv0)
            grid = [[big.evaluate(bu0 + (bu1 - bu0) * j / 32, a + (b - a) * i / 8) for j in range(33)] for i in range(9)]
        M = np.array(self._M, float)
        # inner end first: the old edge (extending) or the new one (shrinking), out to the big surface's edge
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
            self.report({'WARNING'}, "Pick a boundary edge of an untrimmed surface")
            return {'CANCELLED'}
        obj, edge = picks[0]
        if edge not in extend.EDGES or not store.is_natural(obj) or store.get_brep(obj).face_count() != 1:
            self.report({'WARNING'}, "Only boundary edges of untrimmed single surfaces can be extended")
            return {'CANCELLED'}
        scale = sum(geom.world_matrix(obj).to_scale()) / 3.0 or 1.0
        p = {"edge": edge, "distance": float(self.distance) / scale}
        try:
            if features.history_enabled(context):
                features.push_step(obj, "extend_surface", [], p)
                features.rebuild(obj, context, force=True)
            else:
                store.set_geometry_text(obj, extend.extend_surface_text(obj[store.PROP_BREP], edge, p["distance"]),
                                        context, keep_untrimmed=False)
            obj[store.PROP_NATURAL] = 1
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
    Ma, Mb = geom.world_matrix(a), geom.world_matrix(b)
    Mbi = Mb.inverted()
    Ra, Rb = np.array(Ma.to_3x3()), np.array(Mb.to_3x3())
    ba, bb = store.get_brep(a), store.get_brep(b)
    per_edge = []
    for e in ba.edges():
        t0, t1 = e.parameter_range()
        pts = []
        for k in range(1, samples):
            try:
                pts.append(np.array(Ma @ Vector(e.point_at(t0 + (t1 - t0) * k / samples))))
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
            pl = tuple(Mbi @ Vector(p))
            fb, (q, d) = min(((f, f.closest_point(pl)) for f in bb.faces()), key=lambda x: x[1][1])
            hits.append((p, d, fb, np.array(q)))
        if sum(1 for h in hits if h[1] < 0.01 * span) >= 0.8 * len(hits):
            near += hits
    if not near:
        return None
    Mai = Ma.inverted()
    gap, ang = 0.0, 0.0
    curv = []                 # (relative difference, k_a, k_b) of the curvature across the edge
    for idx, (p, d, fb, q) in enumerate(near):
        pa = np.array(Mai @ Vector(p))
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
    u, v = _foot(surf, np.asarray(q, float), None)
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
