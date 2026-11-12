"""Trim / Split / Untrim and Fillet / Chamfer.

Both run their kernel work in the worker process (jobs.py): the kernel can
crash on some inputs, and that must not take Blender down.
"""

import json

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty
from bpy_extras import view3d_utils
from mathutils import Vector

from . import draw, features, geom, jobs, steps, store
from .ops import AskValue, PickMixin, _kernel_ok, _near_candidates, _pick_candidates, _select_only


def _ray_nurbs(context, event):
    """NURBS object, world hit point and normal under the mouse."""
    region, rv3d = context.region, context.region_data
    mouse = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    hit, loc, nrm, _, obj, _ = context.scene.ray_cast(context.evaluated_depsgraph_get(), origin, direction)
    if hit and obj is not None and store.is_nurbs(obj.original):
        return obj.original, loc, nrm, direction
    return None, None, None, direction


def _curve_direction(kcurve, fallback):
    """Plane normal of a planar cutting curve, else the fallback (view) direction."""
    P = geom.sample_curve(kcurve, 48)
    c = P.mean(axis=0)
    size = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0
    _, sv, vt = np.linalg.svd(P - c)
    flat = sv[2] / max(len(P) ** 0.5, 1.0) <= 1e-4 * size
    if flat and sv[1] > 1e-6 * size:  # planar and not a straight line
        return Vector(vt[2]).normalized()
    d = Vector(fallback)
    return d.normalized() if d.length > 1e-9 else Vector((0, 0, 1))


def _frozen(obj, op):
    """Trimming replaces the B-rep: construction history can no longer rebuild it."""
    if features.has_history(obj):
        features.clear_history(obj)
        op.report({'INFO'}, f"{obj.name}: construction history removed")


# --------------------------------------------------------------------------
# Trim
# --------------------------------------------------------------------------

class NPROTO_OT_trim(bpy.types.Operator):
    """Trim a surface with projected curves or intersecting surfaces.
    Click the surface on the side to keep, then pick cutting curves / edges / surfaces"""
    bl_idname = "nproto.trim"
    bl_label = "Trim"
    bl_options = {'REGISTER', 'UNDO'}

    target: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    cutters: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    ref_point: FloatVectorProperty(name="Pick Point", subtype='XYZ', options={'HIDDEN'})
    normal: FloatVectorProperty(name="Surface Normal", subtype='XYZ', options={'HIDDEN'})
    view_dir: FloatVectorProperty(name="View Direction", subtype='XYZ', options={'HIDDEN'})
    action: EnumProperty(name="Action", items=[
        ("KEEP", "Keep", "Keep the picked region"),
        ("DELETE", "Discard", "Remove the picked region"),
        ("SPLIT", "Split", "Keep both sides as separate objects"),
    ], default="KEEP")
    scope: EnumProperty(name="Scope", items=[
        ("THROUGH", "Through", "Act on every region on the same side of the cutters as the clicked one "
                               "(a circle in front of a vase cuts both walls)"),
        ("PICKED", "Picked Region Only", "Act only on the clicked region"),
    ], default="THROUGH")
    projection: EnumProperty(name="Project", items=[
        ("CURVE", "Curve Normal", "Project each planar cutting curve along its own plane normal "
                                  "(cookie cutter); non-planar curves use the view direction"),
        ("VIEW", "View", "Project along the view direction"),
        ("NORMAL", "Surface Normal", "Project along the surface normal at the pick point"),
        ("X", "World X", ""), ("Y", "World Y", ""), ("Z", "World Z", ""),
    ], default="CURVE")

    # ---- interactive picking
    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._stage = 0
        self._cuts = []
        self._target = None
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["hover"] = None
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {'RUNNING_MODAL'}

    def _status(self, context, extra=""):
        from . import hints
        if self._stage == 0:
            state = ["click the surface region to keep / discard", "Action and Project in F9"]
        else:
            state = [f"{self._target.name}: pick cutting curves / edges, or click a cutting surface",
                     f"{len(self._cuts)} picked"]
        state.append(extra.strip(" ·[]"))
        rows = [("Pick", [hints.key("Tab", "cycle overlapping", on=True if getattr(self, "_ties", None) else None),
                          hints.key("Backspace", "undo"),
                          hints.key("Enter", "trim", on=True if self._stage and self._cuts else False)])]
        hints.show(context, "Trim", state, rows, [("MOUSE_LMB", "Pick"), ("EVENT_ESC", "Cancel")])

    def _end(self, context):
        from . import hints
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["hover"] = None
        hints.clear(context)

    def modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        from .ops import PickMixin
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event):
            return {'PASS_THROUGH'}            # the sidebar / headers stay usable
        from . import hints
        if hints.handle(event):
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._end(context)
            return {'CANCELLED'}
        if self._stage == 0:
            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                obj, loc, nrm, vdir = _ray_nurbs(context, event)
                if obj is None:
                    return {'RUNNING_MODAL'}
                self._target = obj
                self.ref_point, self.normal, self.view_dir = loc, nrm, vdir
                self._cands = [c for c in _pick_candidates(context) if c[0] != obj]
                self._ties, self._cycle = [], 0
                self._stage = 1
                self._status(context)
            return {'RUNNING_MODAL'}

        # stage 1: cutters
        if event.type == 'MOUSEMOVE':
            ties = _near_candidates(context, event, self._cands)
            if [(c[0].name, c[1]) for c in ties] != [(c[0].name, c[1]) for c in self._ties]:
                self._ties, self._cycle = ties, 0
            cur = self._ties[self._cycle % len(self._ties)] if self._ties else None
            draw.pick_overlay["hover"] = cur[2] if cur else None
            context.area.tag_redraw()
        elif event.type == 'TAB' and event.value == 'PRESS' and self._ties:
            self._cycle += 1
            draw.pick_overlay["hover"] = self._ties[self._cycle % len(self._ties)][2]
            context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            ties = _near_candidates(context, event, self._cands)
            if ties:
                cur = ties[self._cycle % len(ties)] if ties == self._ties else ties[0]
                self._cuts.append((cur[0], cur[1]))
                draw.pick_overlay["picked"].append(cur[2])
            else:
                obj, _, _, _ = _ray_nurbs(context, event)
                if obj is not None and obj != self._target:
                    self._cuts.append((obj, "SURF"))
                    poly = []
                    for e in store.display(obj).get("edges", []):
                        poly += [obj.matrix_world @ Vector(p) for p in e]
                    draw.pick_overlay["picked"].append(poly)
            self._status(context)
            context.area.tag_redraw()
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS' and self._cuts:
            self._cuts.pop()
            draw.pick_overlay["picked"].pop()
            self._status(context)
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS' and self._cuts:
            self._end(context)
            self.target = self._target.name
            self.cutters = json.dumps([[o.name, r] for o, r in self._cuts])
            return self.execute(context)
        return {'RUNNING_MODAL'}

    # ---- the actual trim
    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        tgt = bpy.data.objects.get(self.target)
        if tgt is None or not store.is_nurbs(tgt):
            self.report({'ERROR'}, "Trim: no target surface")
            return {'CANCELLED'}
        curves, cutter_objs = [], {}
        try:
            for name, ref in json.loads(self.cutters or "[]"):
                obj = bpy.data.objects[name]
                if ref == "SURF":
                    cutter_objs[f"c{len(cutter_objs)}"] = obj
                else:
                    kc = features.input_curve(obj, ref)
                    cj = jobs.curve_json(kc)
                    if self.projection == "CURVE":
                        cj["dir"] = list(_curve_direction(kc, Vector(self.view_dir)))
                    curves.append(cj)
        except Exception as exc:
            self.report({'ERROR'}, f"Trim: bad cutter ({exc})")
            return {'CANCELLED'}
        if self.projection in ("NORMAL", "CURVE"):  # CURVE: used for cutting surfaces only
            d = Vector(self.normal)
        elif self.projection == "VIEW":
            d = Vector(self.view_dir)
        else:
            d = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[self.projection])
        if d.length < 1e-9:
            d = Vector((0, 0, 1))
        outs = ["KEEP", "DELETE"] if self.action == "SPLIT" else [self.action]
        pre_text = tgt[store.PROP_BREP]
        try:
            from .steps import run_trim
            res, breps = run_trim(tgt, cutter_objs, outs, curves=curves,
                                  direction=list(d.normalized()), ref_point=list(self.ref_point),
                                action=self.action, scope=self.scope)
        except jobs.JobError as exc:
            self.report({'ERROR'}, f"Trim failed: {exc}")
            return {'CANCELLED'}
        keep_history = features.history_enabled(context)
        if not keep_history:
            _frozen(tgt, self)
        main = breps["KEEP"] if self.action in ("KEEP", "SPLIT") else breps["DELETE"]
        store.set_geometry_text(tgt, main, context)
        made = [tgt]
        if keep_history:
            Mi = geom.world_matrix(tgt).inverted()
            R = Mi.to_3x3()
            inputs = [(bpy.data.objects[n], r) for n, r in json.loads(self.cutters or "[]")]
            params = {"action": "KEEP" if self.action == "SPLIT" else self.action, "scope": self.scope,
                      "projection": self.projection, "ref_local": list(Mi @ Vector(self.ref_point)),
                      "normal_local": list(R @ Vector(self.normal)), "view_local": list(R @ Vector(self.view_dir)),
                      "side": (res.get("info") or {}).get("side")}
            features.push_step(tgt, "trim", inputs, params, base_text=pre_text)
        if self.action == "SPLIT":
            other = store.new_object_text(context, tgt.name + "_split", breps["DELETE"], geom.world_matrix(tgt))
            other[store.PROP_UNTRIM] = tgt[store.PROP_UNTRIM]
            if keep_history:  # same history, last step keeps the other side
                features.copy_history(tgt, other)
                entries = features.stack(other)
                entries[-1]["params"] = dict(entries[-1]["params"], action="DELETE")
                features._write_stack(other, entries)
            made.append(other)
        _select_only(context, made)
        return {'FINISHED'}


class NPROTO_OT_untrim(bpy.types.Operator):
    """Restore selected surfaces to how they were before their first trim"""
    bl_idname = "nproto.untrim"
    bl_label = "Untrim"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        n = 0
        for o in context.selected_objects:
            if store.is_nurbs(o) and features.PROP_STACK in o and any(
                    e["op"] in ("trim", "fillet_trim") for e in features.stack(o)):
                features.remove_steps(o, {"trim", "fillet_trim"})
                if features.PROP_STACK not in o and features.PROP_HIST not in o:
                    text, nat = o[features.PROP_BASE], o.get(features.PROP_BASE + "_natural", 0)
                    store.set_geometry_text(o, text, context, keep_untrimmed=False)
                    o[store.PROP_NATURAL] = nat
                    del o[features.PROP_BASE]
                    if store.PROP_UNTRIM in o:
                        del o[store.PROP_UNTRIM]
                else:
                    features.rebuild(o, context, force=True)
                n += 1
                continue
            if store.is_nurbs(o) and store.PROP_UNTRIM in o:
                natural = o.get(store.PROP_UNTRIM + "_natural", 0)
                store.set_geometry_text(o, o[store.PROP_UNTRIM], context, keep_untrimmed=False)
                o[store.PROP_NATURAL] = natural
                del o[store.PROP_UNTRIM]
                n += 1
        self.report({'INFO'}, f"Untrimmed {n} object(s)")
        return {'FINISHED'} if n else {'CANCELLED'}


# --------------------------------------------------------------------------
# Fillet / Chamfer
# --------------------------------------------------------------------------

def part_size(objs):
    """The smallest of the objects' world bounding-box sizes (the longest side
    of each box), or None when there is nothing to measure."""
    sizes = []
    for o in objs:
        M = geom.world_matrix(o)
        P = np.array([M @ Vector(c) for c in o.bound_box], float)
        s = float(np.ptp(P, axis=0).max()) if len(P) else 0.0
        if s > 1e-9:
            sizes.append(s)
    return min(sizes) if sizes else None


def nice_length(x):
    """x rounded down to 1, 2 or 5 times a power of ten."""
    e = 10.0 ** np.floor(np.log10(x))
    return float(max(m for m in (1.0, 2.0, 5.0, 10.0) if m * e <= x * (1 + 1e-9)) * e)


def fit_radius(op, objs, names):
    """The radius remembered from the last use (or the default) may be far too
    big for this part: a 0.25 m fillet on a 5 cm part finds no fillet at all.
    If the largest of the properties *names* is over a fifth of the part's size,
    scale them all down so the largest is a twentieth of it (rounded), keeping
    their ratios. Only for a call from the UI (a script's radius is kept), and
    only once per call, so the redo panel's values stay yours."""
    if not op.fit_pending:
        return
    op.fit_pending = False
    size = part_size(objs)
    big = max(getattr(op, n) for n in names)
    if size is None or big <= 0.2 * size:
        return
    k = nice_length(0.05 * size) / big
    for n in names:
        setattr(op, n, getattr(op, n) * k)
    op.report({'INFO'}, f"Radius {big:g} m is too big for this part ({size:g} m): "
                        f"using {getattr(op, names[0]):g} m")


class NPROTO_OT_fillet(AskValue, bpy.types.Operator):
    """Fillet or chamfer between two surfaces (select both)"""
    bl_idname = "nproto.fillet"
    bl_label = "Fillet / Chamfer"
    bl_options = {'REGISTER', 'UNDO'}

    surf_a: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    surf_b: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_pending: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})  # set by invoke: check the radius once
    shape: EnumProperty(name="Shape", items=[
        ("CIRCULAR", "Circular", "Exact circular cross-section"),
        ("CHAMFER", "Chamfer", "Straight (linear) cross-section"),
        ("BLEND", "Blend", "Free-form blend, G1 or G2 to both surfaces"),
    ], default="CIRCULAR")
    continuity: EnumProperty(name="Continuity", items=[("1", "G1", "Tangent to both surfaces"),
                                                       ("2", "G2", "Curvature continuous to both surfaces")],
                             default="2")
    rail_type: EnumProperty(name="Rail", items=[
        ("RADIUS", "Rolling Ball", "The size is the radius of a ball rolling between the faces"),
        ("CHORD", "Chord", "The size is the distance between the two rails (where the fillet meets the faces)"),
    ], default="RADIUS")
    radius: FloatProperty(name="Radius A", default=0.25, min=1e-5, soft_max=10.0, unit='LENGTH',
                          description="Distance from surface A to its rail")
    radius_b: FloatProperty(name="Radius B", default=0.25, min=1e-5, soft_max=10.0, unit='LENGTH')
    same_radius: BoolProperty(name="Same Radius", default=True)
    flip_a: BoolProperty(name="Flip A", default=False, description="Put the fillet on the other side of A")
    flip_b: BoolProperty(name="Flip B", default=False, description="Put the fillet on the other side of B")
    auto_flip: BoolProperty(name="Auto Side", default=True,
                            description="If there is no fillet on the chosen sides, try the others")
    trim_bases: BoolProperty(name="Trim Surfaces", default=True,
                             description="Trim both surfaces back to the fillet rails")

    def invoke(self, context, event):
        sel = [o for o in context.selected_objects if store.is_nurbs(o)]
        if len(sel) != 2:
            self.report({'WARNING'}, "Select exactly two NURBS surfaces")
            return {'CANCELLED'}
        act = context.active_object
        sel.sort(key=lambda o: o == act)  # active = B
        self.surf_a, self.surf_b = sel[0].name, sel[1].name
        self.fit_pending = True
        return self.ask(context)

    def prepare_value(self, context):
        a, b = bpy.data.objects.get(self.surf_a), bpy.data.objects.get(self.surf_b)
        if a is not None and b is not None:
            fit_radius(self, [a, b], ["radius", "radius_b"])     # a size that fits, shown in the dialog

    def _chord(self):
        return self.radius if self.rail_type == "CHORD" else None

    def _try(self, a, b, fa, fb):
        chord = self._chord()
        rb = self.radius if (self.same_radius or chord) else self.radius_b
        self._used = {"radius_a": -self.radius if fa else self.radius, "radius_b": -rb if fb else rb,
                      "xsect": 0 if self.shape == "CHAMFER" else 2, "chord": chord,
                      "blend": int(self.continuity) if self.shape == "BLEND" else 0}
        return jobs.run("fillet", {"a": a, "b": b}, ["fillet", "a", "b"],
                        radius_a=-self.radius if fa else self.radius,
                        radius_b=-rb if fb else rb,
                        xsect=0 if self.shape == "CHAMFER" else 2, chord=chord,
                        tolerance=1e-4, trim_bases=self.trim_bases or self.shape == "BLEND")

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        a = bpy.data.objects.get(self.surf_a)
        b = bpy.data.objects.get(self.surf_b)
        if a is None or b is None:
            self.report({'ERROR'}, "Fillet: select two NURBS surfaces")
            return {'CANCELLED'}
        fit_radius(self, [a, b], ["radius", "radius_b"])
        order = [(self.flip_a, self.flip_b)]
        if self.auto_flip:
            order += [(fa, fb) for fa in (False, True) for fb in (False, True) if (fa, fb) != order[0]]
        last, res, breps = None, None, {}
        for fa, fb in order:
            try:
                res, breps = self._try(a, b, fa, fb)
                break
            except jobs.JobError as exc:
                # wrong-side combinations fail with "no fillet found" OR a kernel
                # error, depending on the surfaces: keep trying the other sides
                last = exc
                continue
        if res is None or "fillet" not in breps:
            tried = "all four side combinations" if self.auto_flip else "this side combination"
            size = part_size([a, b])
            rb = self.radius if self.same_radius else self.radius_b
            hint = (f" (the radius {max(self.radius, rb):g} m may be too big for surfaces {size:g} m across)"
                    if size and max(self.radius, rb) > 0.1 * size else "")
            self.report({'ERROR'}, f"Fillet failed on {tried}: {last}{hint}")
            return {'CANCELLED'}
        if (fa, fb) != (self.flip_a, self.flip_b):
            self.flip_a, self.flip_b = fa, fb  # show what was used in the redo panel
        fil = store.new_object_text(context, "Chamfer" if self.shape == "CHAMFER" else "Fillet", breps["fillet"])
        keep_history = features.history_enabled(context)
        u = self._used
        for key, obj in (("a", a), ("b", b)):
            if key in breps:
                pre = obj[store.PROP_BREP]
                if not keep_history:
                    _frozen(obj, self)
                store.set_geometry_text(obj, breps[key], context)
                if keep_history:
                    partner = b if key == "a" else a
                    features.push_step(obj, "fillet_trim", [(partner, "SURF")], dict(u, role=key), base_text=pre)
        for key, err in res.get("trim_errors", {}).items():
            self.report({'WARNING'}, f"Could not trim {(a if key == 'a' else b).name}: {err}")
        if u["blend"]:
            # the circular fillet trimmed both surfaces back to its rails: the
            # blend is a G1 / G2 loft between those two new edges
            try:
                store.set_geometry(fil, steps.blend_between(a, b, breps["fillet"], u["blend"]), context)
            except Exception as exc:
                self.report({'WARNING'}, f"Blend failed ({exc}): kept the circular fillet")
                u["blend"] = 0
        if keep_history:
            features.record(fil, "surface_fillet", [(a, "SURF"), (b, "SURF")], dict(u))
        if res.get("chord"):
            lo, mean, hi = res["chord"]
            spread = (f"{lo:.4g}..{hi:.4g}" if hi - lo > 1e-3 * mean else f"{mean:.4g}")
            self.report({'INFO'}, f"Chord {spread} m (ball radius {res['radius']:.4g} m)")
        if keep_history:          # all three are final now: no history re-run needed
            features.mark_built(fil, a, b)
        _select_only(context, [fil])
        return {'FINISHED'}

    def draw(self, context):
        lay = self.layout
        lay.prop(self, "shape", expand=True)
        if self.shape == "BLEND":
            lay.row(align=True).prop(self, "continuity", expand=True)
        lay.row(align=True).prop(self, "rail_type", expand=True)
        col = lay.column(align=True)
        if self.rail_type == "CHORD":
            col.prop(self, "radius", text="Chord")
        else:
            col.prop(self, "radius", text="Distance A" if self.shape == "CHAMFER" else "Radius A")
            col.prop(self, "same_radius")
            sub = col.row()
            sub.enabled = not self.same_radius
            sub.prop(self, "radius_b", text="Distance B" if self.shape == "CHAMFER" else "Radius B")
        row = lay.row(align=True)
        row.prop(self, "flip_a", toggle=True)
        row.prop(self, "flip_b", toggle=True)
        lay.prop(self, "auto_flip")
        lay.prop(self, "trim_bases")


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

class NPROTO_OT_split(bpy.types.Operator):
    """Split the selected surfaces along each other and/or along the selected
    curves. Every region becomes its own object; nothing is removed. With
    history the split surfaces are kept as hidden inputs and each region
    follows them (and what cut them)"""
    bl_idname = "nproto.split"
    bl_label = "Split"
    bl_options = {'REGISTER', 'UNDO'}

    surfaces_cut: BoolProperty(name="Surfaces Split Each Other", default=True,
                               description="Each selected surface is split by the other selected surfaces")
    projection: EnumProperty(name="Curve Projection", items=[
        ("CURVE", "Curve Normal", "Each planar curve along its own normal; non-planar curves use the view"),
        ("VIEW", "View", "Along the view direction"),
        ("X", "World X", ""), ("Y", "World Y", ""), ("Z", "World Z", ""),
    ], default="CURVE")
    view_dir: FloatVectorProperty(subtype='XYZ', options={'HIDDEN'})
    targets: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    curves: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        rv3d = getattr(context, "region_data", None)
        if rv3d is not None:
            self.view_dir = rv3d.view_rotation @ Vector((0, 0, -1))
        surfs = [o for o in context.selected_objects if store.is_nurbs(o)]
        crvs = [o for o in context.selected_objects if o.type == 'CURVE']
        self.targets = json.dumps([o.name for o in surfs])
        self.curves = json.dumps([[o.name, f"S{i}"] for o in crvs for i in range(len(o.data.splines))])
        return self.execute(context)

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not self.targets:  # called from a script: use the selection
            self.targets = json.dumps([o.name for o in context.selected_objects if store.is_nurbs(o)])
            self.curves = json.dumps([[o.name, f"S{i}"] for o in context.selected_objects if o.type == 'CURVE'
                                      for i in range(len(o.data.splines))])
        surfs = [bpy.data.objects[n] for n in json.loads(self.targets) if n in bpy.data.objects]
        curve_refs = json.loads(self.curves or "[]")
        if not surfs or (len(surfs) < 2 and not curve_refs):
            self.report({'WARNING'}, "Select two or more surfaces, or surfaces and curves")
            return {'CANCELLED'}
        params = {"surfaces_cut": bool(self.surfaces_cut), "projection": self.projection,
                  "view_dir": [float(x) for x in self.view_dir]}
        crv_inputs = [(bpy.data.objects[n], r) for n, r in curve_refs if n in bpy.data.objects]

        # compute every split from the ORIGINAL geometry first, then apply
        results, skipped = {}, []
        for tgt in surfs:
            inputs = [(tgt, "SURF")] + ([(s, "SURF") for s in surfs if s != tgt] if self.surfaces_cut else []) \
                + crv_inputs
            if len(inputs) < 2:
                continue
            try:
                results[tgt.name] = (inputs, steps.split_regions(inputs, params))
            except jobs.JobError as exc:
                skipped.append(f"{tgt.name}: {exc}")
            except Exception as exc:
                skipped.append(f"{tgt.name}: bad curve ({exc})")
        if not results:
            self.report({'ERROR'}, "Split: nothing was split. " + "; ".join(skipped))
            return {'CANCELLED'}
        made = []
        keep = features.history_enabled(context)
        for name, (inputs, texts) in results.items():
            tgt = bpy.data.objects[name]
            original = tgt.get(store.PROP_UNTRIM, tgt[store.PROP_BREP])
            if not keep:                  # no history: the largest region stays in the surface itself
                _frozen(tgt, self)
                store.set_geometry_text(tgt, texts[0], context)
                made.append(tgt)
            for i, text in enumerate(texts if keep else texts[1:], 1):
                o = store.new_object_text(context, f"{name}_split{i}", text, geom.world_matrix(tgt))
                o[store.PROP_UNTRIM] = original
                if tgt.data.materials:
                    for m in tgt.data.materials:
                        o.data.materials.append(m)
                if keep:              # each region follows the split surface and what cut it
                    features.record(o, "split_region", inputs, {**params, "at": steps.region_point(text)})
                made.append(o)
        if keep:
            features.keep_inputs(context, [bpy.data.objects[n] for n in results])
        for msg in skipped:
            self.report({'WARNING'}, msg)
        _select_only(context, made)
        self.report({'INFO'}, f"Split into {len(made)} pieces")
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Join / Unjoin
# --------------------------------------------------------------------------

def _remove_object(obj):
    data = obj.data
    bpy.data.objects.remove(obj)
    if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
        bpy.data.meshes.remove(data)
    elif data is not None and data.users == 0 and isinstance(data, bpy.types.Curve):
        bpy.data.curves.remove(data)


class NPROTO_OT_join(bpy.types.Operator):
    """Join the selected surfaces into one object and stitch their coincident
    edges (within Tolerance). The active object receives the result"""
    bl_idname = "nproto.join"
    bl_label = "Join"
    bl_options = {'REGISTER', 'UNDO'}

    tolerance: FloatProperty(name="Tolerance", default=0.001, min=1e-6, soft_max=0.1, precision=5, unit='LENGTH',
                             description="Largest gap between edges that is still stitched")
    names: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        sel = [o for o in context.selected_objects if store.is_nurbs(o)]
        if not sel and sum(1 for o in context.selected_objects if o.type == 'CURVE') >= 1:
            return bpy.ops.nproto.join_curves('INVOKE_DEFAULT')
        act = context.active_object
        sel.sort(key=lambda o: o == act)  # active last = receives the result
        self.names = json.dumps([o.name for o in sel])
        return self.execute(context)

    def execute(self, context):
        if not self.names and not any(store.is_nurbs(o) for o in context.selected_objects) and \
                any(o.type == 'CURVE' for o in context.selected_objects):
            return bpy.ops.nproto.join_curves('INVOKE_DEFAULT')
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not self.names:
            act = context.active_object
            sel = sorted((o for o in context.selected_objects if store.is_nurbs(o)), key=lambda o: o == act)
            self.names = json.dumps([o.name for o in sel])
        objs = [bpy.data.objects[n] for n in json.loads(self.names) if n in bpy.data.objects]
        if len(objs) < 2:
            self.report({'WARNING'}, "Select two or more NURBS surfaces")
            return {'CANCELLED'}
        into = objs[-1]
        keys = {f"s{i}": o for i, o in enumerate(objs)}
        if features.history_enabled(context):
            try:
                g = features.run("join", [(o, "SURF") for o in objs], {"tolerance": self.tolerance})
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"Join failed: {exc}")
                return {'CANCELLED'}
            joined = store.new_object_text(context, into.name + "_joined", g[1])
            features.record(joined, "join", [(o, "SURF") for o in objs], {"tolerance": self.tolerance})
            features.keep_inputs(context, objs)
            _select_only(context, [joined])
            b = store.get_brep(joined)
            self.report({'INFO'}, f"Joined {len(objs)} surfaces into {b.face_count()} faces "
                                  "(sources kept hidden: history; Unjoin brings them back)")
            return {'FINISHED'}
        try:
            info, breps = jobs.run("join", keys, ["joined"], order=list(keys),
                                   into=f"s{len(objs) - 1}", tolerance=self.tolerance)
        except jobs.JobError as exc:
            self.report({'ERROR'}, f"Join failed: {exc}")
            return {'CANCELLED'}
        for o in objs:
            _frozen(o, self)
        store.set_geometry_text(into, breps["joined"], context, keep_untrimmed=False)
        if store.PROP_UNTRIM in into:
            del into[store.PROP_UNTRIM]
        for o in objs[:-1]:
            _remove_object(o)
        _select_only(context, [into])
        msg = f"Joined {len(objs)} surfaces into {info['faces']} faces, {info['stitched']} edge(s) stitched"
        if info.get("solid"):
            msg += f" -- closed solid, volume {info.get('volume', 0):.4g}"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class NPROTO_OT_unjoin(bpy.types.Operator):
    """Break the selected multi-face surfaces into one object per face, keeping
    the current shape (fillets and trims included). To go back to the surfaces
    a Join was made from, use Restore Inputs in the Feature panel"""
    bl_idname = "nproto.unjoin"
    bl_label = "Unjoin"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not any(store.is_nurbs(o) for o in context.selected_objects) and \
                any(o.type == 'CURVE' for o in context.selected_objects):
            return bpy.ops.nproto.unjoin_curves()
        if not _kernel_ok(self):
            return {'CANCELLED'}
        objs = [o for o in context.selected_objects if store.is_nurbs(o) and store.get_brep(o).face_count() > 1]
        if not objs:
            self.report({'WARNING'}, "Select a joined surface (more than one face)")
            return {'CANCELLED'}
        made = []
        for obj in objs:
            try:
                info, breps = jobs.run("unjoin", {"target": obj}, [])
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"{obj.name}: {exc}")
                continue
            parts = sorted(((int(k[4:]), t) for k, t in breps.items() if k.startswith("face")))
            areas = info.get("areas", [])
            order = sorted(range(len(parts)), key=lambda i: -(areas[i] if i < len(areas) else 0))
            texts = [parts[i][1] for i in order]
            h = features.history(obj)
            if h and h["op"] == "join":
                hidden = [src.name for src, _ in h["inputs"] if src.hide_get()]
                if hidden:
                    self.report({'INFO'}, "The Join's original surfaces stay hidden in the scene: " + ", ".join(hidden))
            _frozen(obj, self)
            M = geom.world_matrix(obj)
            store.set_geometry_text(obj, texts[0], context, keep_untrimmed=False)
            made.append(obj)
            for i, text in enumerate(texts[1:], 1):
                o = store.new_object_text(context, f"{obj.name}_face{i}", text, M)
                for m in obj.data.materials:
                    o.data.materials.append(m)
                made.append(o)
        if not made:
            return {'CANCELLED'}
        _select_only(context, made)
        self.report({'INFO'}, f"Unjoined into {len(made)} surfaces")
        return {'FINISHED'}


class NPROTO_OT_restore_inputs(bpy.types.Operator):
    """Undo the Join: delete the joined object and show the surfaces it was
    made from again. History steps on the joined object (e.g. edge fillets)
    are lost -- use Unjoin to split the current shape instead"""
    bl_idname = "nproto.restore_inputs"
    bl_label = "Restore Join Inputs"
    bl_options = {'REGISTER', 'UNDO'}

    def _targets(self, context):
        out = []
        for o in context.selected_objects or [context.active_object]:
            h = features.history(o) if o is not None and (store.is_nurbs(o) or o.type == 'CURVE') else None
            if h and h["op"] in features.KEEP_OPS:
                out.append(o)
        act = context.active_object
        if act is not None and act not in out:
            h = features.history(act) if (store.is_nurbs(act) or act.type == 'CURVE') else None
            if h and h["op"] in features.KEEP_OPS:
                out.append(act)
        return out

    def invoke(self, context, event):
        objs = self._targets(context)
        if not objs:
            self.report({'WARNING'}, "Select a joined surface or curve that has construction history")
            return {'CANCELLED'}
        lost = []
        for o in objs:
            try:
                lost += [f"{o.name}: {e['op'].replace('_', ' ')}" for e in features.stack(o)]
            except RuntimeError:
                pass
        if lost:
            msg = "These history steps will be lost: " + "; ".join(lost)
            try:
                return context.window_manager.invoke_confirm(self, event, title="Restore Join Inputs",
                                                             message=msg, confirm_text="Restore", icon='WARNING')
            except TypeError:
                return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def execute(self, context):
        objs = self._targets(context)
        if not objs:
            self.report({'WARNING'}, "Select a joined surface or curve that has construction history")
            return {'CANCELLED'}
        restored = []
        for o in objs:
            srcs = list(dict.fromkeys(src for src, _ in features.history(o)["inputs"]))
            _remove_object(o)
            features.release_inputs(context, srcs)
            restored += srcs
        _select_only(context, restored)
        self.report({'INFO'}, f"Restored {len(restored)} source objects")
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Edge fillet (joined surfaces / solids)
# --------------------------------------------------------------------------

def _edge_mid_world(obj, idx):
    e = store.get_brep(obj).edges()[idx]
    t0, t1 = e.parameter_range()
    try:
        p = e.point_at(0.5 * (t0 + t1))
    except Exception:
        c = e.curve()
        a, b = c.parameter_range()
        p = c.evaluate(0.5 * (a + b))
    return list(geom.world_matrix(obj) @ Vector(p))


def seam_crossings(obj, idxs):
    """Explain a kernel fillet failure caused by an edge that crosses the seam
    of a closed (periodic) face, e.g. an opening cut across a revolve's seam."""
    from .square import _foot
    brep = store.get_brep(obj)
    bad = []
    for i in idxs:
        e = brep.edges()[i]
        t0, t1 = e.parameter_range()
        size_e = e.length() or 1.0
        if np.linalg.norm(np.array(e.point_at(t0)) - np.array(e.point_at(t1))) < 1e-6 * size_e:
            continue        # a closed loop around a closed surface (e.g. a tube rim) is fine
        for f in e.faces():
            s = f.surface()
            (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
            mid_v, mid_u = 0.5 * (v0 + v1), 0.5 * (u0 + u1)
            tol = 1e-5 * (float(np.linalg.norm(np.ptp(np.array(f.bounding_box(), float), axis=0))) or 1.0)
            closed_u = np.linalg.norm(np.array(s.evaluate(u0, mid_v)) - np.array(s.evaluate(u1, mid_v))) < tol
            closed_v = np.linalg.norm(np.array(s.evaluate(mid_u, v0)) - np.array(s.evaluate(mid_u, v1))) < tol
            if not (closed_u or closed_v):
                continue
            # independent (global) foot points: a seam crossing shows as a jump u0 <-> u1
            uv = np.array([_foot(s, np.array(e.point_at(t)), None) for t in np.linspace(t0, t1, 15)])
            jump_u = closed_u and np.abs(np.diff(uv[:, 0])).max() > 0.5 * (u1 - u0)
            jump_v = closed_v and np.abs(np.diff(uv[:, 1])).max() > 0.5 * (v1 - v0)
            if jump_u or jump_v:
                bad.append(f"E{i}")
                break
    if not bad:
        return None
    return (f"edge(s) {', '.join(bad)} run across the seam of a single closed face (e.g. a revolve) -- a likely "
            "cause. Rebuild with Trim (which splits such faces at the seam), or turn the seam away "
            "(Revolve > Seam Angle) and trim again")


def tangent_chain(cand, cands, max_deg=10.0):
    """cand plus every candidate edge reachable through end-to-end joins where
    the direction changes by less than max_deg (world polylines)."""
    def ends(pl):
        a, b = np.array(pl[0]), np.array(pl[-1])
        ta = np.array(pl[min(1, len(pl) - 1)]) - a
        tb = np.array(pl[-1]) - np.array(pl[max(len(pl) - 2, 0)])
        n = lambda v: v / (np.linalg.norm(v) or 1.0)
        return [(a, -n(ta)), (b, n(tb))]           # (point, outward direction at that end)
    allp = np.vstack([np.array(c[2]) for c in cands]) if cands else np.zeros((1, 3))
    tol = 1e-4 * (float(np.linalg.norm(np.ptp(allp, axis=0))) or 1.0)
    cos_max = np.cos(np.radians(max_deg))
    chain, todo = [cand], [cand]
    seen = {cand[1]}
    while todo:
        c = todo.pop()
        for p, d in ends(c[2]):
            for o in cands:
                if o[1] in seen:
                    continue
                for q, e in ends(o[2]):
                    # joined here, and the other edge leaves in the direction we were going
                    if np.linalg.norm(p - q) < tol and float(-e @ d) > cos_max:
                        seen.add(o[1])
                        chain.append(o)
                        todo.append(o)
                        break
    return chain


class NPROTO_OT_edge_fillet(AskValue, PickMixin, bpy.types.Operator):
    """Fillet, chamfer or blend shared edges of joined surfaces and solids.
    Click the edges to fillet (click again to un-pick, A = all), Enter to finish.
    Alt+click the button: all shared edges of the selected objects at once"""
    bl_idname = "nproto.edge_fillet"
    bl_label = "Edge Fillet"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 1
    pick_max = 4096
    finish_on_min = False
    toggle_picks = True
    extra_keys = [("Alt+LMB", "single edge"), ("A", "all edges")]
    extra_help = "Tangent chains are picked together; click a picked edge again to un-pick it. Enter fillets."

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_pending: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})  # set by invoke: check the radius once
    shape: EnumProperty(name="Shape", items=[
        ("CIRCULAR", "Circular", "Circular cross-section"),
        ("CHAMFER", "Chamfer", "Straight cross-section"),
        ("BLEND", "Blend", "Free-form blend with G1/G2/G3 continuity to both faces"),
    ], default="CIRCULAR")
    rail_type: EnumProperty(name="Rail", items=[
        ("RADIUS", "Rolling Ball", "The size is the radius of a ball rolling between the faces"),
        ("CHORD", "Chord", "The size is the distance between the two rails (where the fillet meets the faces)"),
    ], default="RADIUS")
    radius: FloatProperty(name="Radius", default=0.1, min=1e-4, soft_max=5.0, unit='LENGTH')
    variable: BoolProperty(name="Variable Radius", default=False)
    end_radius: FloatProperty(name="End Radius", default=0.2, min=1e-4, soft_max=5.0, unit='LENGTH')
    continuity: EnumProperty(name="Continuity", items=[("1", "G1", ""), ("2", "G2", ""), ("3", "G3", "")], default="2")
    thumbweight: FloatProperty(name="Fullness", default=1.0, min=0.1, soft_max=3.0,
                               description="Blend shape: >1 fuller, <1 thinner")
    tangent_chain: BoolProperty(name="Tangent Chain", default=True,
                                description="Picking an edge also picks the edges that continue it tangentially")
    chain_angle: FloatProperty(name="Chain Angle", default=10.0, min=0.5, max=45.0,
                               description="Max change of direction between chained edges (degrees)")
    class_a: BoolProperty(name="Class-A Patches", default=False,
                          description="Rebuild the kernel's dense fillet faces (along curved edges) as the fewest "
                                      "single-span patches inside the tolerance, G1 / G2 to the next fillet")
    ca_tolerance: FloatProperty(name="Tolerance", default=1e-4, min=1e-6, soft_max=0.01, precision=5, unit='LENGTH',
                                description="Largest deviation from the exact rolling-ball fillet")
    ca_angle: FloatProperty(name="Angle Tolerance", default=0.1, min=0.001, soft_max=1.0, precision=3,
                            description="Largest normal angle to the filleted faces along the rails (degrees)")
    ca_degree: IntProperty(name="Max Degree", default=7, min=3, max=9,
                           description="Highest degree along the edge; more patches are used if it is not enough")
    ca_continuity: EnumProperty(name="Ends", items=[("1", "G1", "Tangent to the next fillet face"),
                                                    ("2", "G2", "Curvature continuous to the next fillet face")],
                                default="2")

    def pick_candidates(self, context):
        # with joined objects selected, only their edges are offered
        sel = [o for o in context.selected_objects if store.is_nurbs(o)]
        pool = [o for o in sel if self._shared(o)] or [o for o in context.visible_objects if store.is_nurbs(o)]
        cands = []
        for obj in pool:
            shared = self._shared(obj)
            if not shared:
                continue
            M = geom.world_matrix(obj)
            for i, poly in enumerate(store.display(obj).get("edges", [])):
                if i in shared and len(poly) > 1:
                    cands.append((obj, f"E{i}", [M @ Vector(p) for p in poly]))
        return cands

    def expand_pick(self, cand):
        """One click picks the whole tangent chain: shared edges of the same
        object joined end to end with (almost) the same direction."""
        if not self.tangent_chain or getattr(self, "_alt", False):
            return [cand]
        return tangent_chain(cand, [c for c in self._cands if c[0] == cand[0]], self.chain_angle)

    @staticmethod
    def _shared(obj):
        try:
            return set(store.shared_edges(obj))
        except Exception:
            return set()

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self.fit_pending = True
        if self.inputs:                     # picked already (see PickMixin._finish)
            return self.ask(context)
        from . import subsel
        picked = subsel.selected_edges(context)     # Edge mode: the picked edges, no picking
        if picked:
            ok = [(o, r) for o, r in picked if int(r[1:]) in self._shared(o)]
            if not ok:
                self.report({'WARNING'}, "None of the picked edges lies between two faces: Join the surfaces first")
                return {'CANCELLED'}
            if len(ok) < len(picked):
                self.report({'WARNING'}, f"{len(picked) - len(ok)} picked edge(s) skipped (open: only edges "
                                         "between two faces can be filleted)")
            self.inputs = json.dumps([[o.name, r] for o, r in ok])
            return self.ask(context)
        if event.alt:   # all shared edges of the selection at once
            sel = [o for o in context.selected_objects if store.is_nurbs(o) and store.shared_edges(o)]
            if sel:
                self.inputs = json.dumps([[o.name, f"E{i}"] for o in sel for i in store.shared_edges(o)])
                return self.ask(context)
        if not self.pick_candidates(context):
            self.report({'WARNING'}, "No shared edges: Join surfaces first (only edges between two faces can be filleted)")
            return {'CANCELLED'}
        return self.start_picking(context)

    def prepare_value(self, context):
        try:
            names = {n for n, _ in json.loads(self.inputs or "[]")}
        except ValueError:
            names = set()
        objs = [bpy.data.objects[n] for n in names if n in bpy.data.objects]
        if objs:
            fit_radius(self, objs, ["radius", "end_radius"])     # a size that fits, shown in the dialog

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._alt = event.alt
        if event.type == 'A' and event.value == 'PRESS':
            have = {(o.name, r) for o, r in self._picks}
            for c in self._cands:
                if (c[0].name, c[1]) not in have:
                    self._picks.append((c[0], c[1]))
                    draw.pick_overlay["picked"].append(c[2])
            self._groups = [1] * len(self._picks)
            self._status(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        return super().modal(context, event)

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            picks = json.loads(self.inputs or "[]")
        except ValueError:
            picks = []
        by_obj = {}
        for name, ref in picks:
            if name in bpy.data.objects and ref.startswith("E"):
                by_obj.setdefault(name, []).append(int(ref[1:]))
        if not by_obj:
            self.report({'WARNING'}, "Pick shared edges of a joined surface")
            return {'CANCELLED'}
        fit_radius(self, [bpy.data.objects[n] for n in by_obj], ["radius", "end_radius"])
        chord = self.radius if self.rail_type == "CHORD" else None
        done = []
        for name, idxs in by_obj.items():
            obj = bpy.data.objects[name]
            try:
                mids = [_edge_mid_world(obj, i) for i in idxs]
                info, breps = jobs.run("edge_fillet", {"target": obj}, ["result"], edge_mids=mids,
                                       shape=self.shape, radius=self.radius,
                                       end_radius=self.end_radius if self.variable and not chord else None,
                                       chord=chord,
                                       continuity=int(self.continuity), thumbweight=self.thumbweight,
                                       **steps.class_a_params(self._ca_params()))
            except (jobs.JobError, IndexError) as exc:
                why = seam_crossings(obj, idxs)
                if why:
                    self.report({'ERROR'}, f"Edge fillet on {name} failed: {why}")
                else:
                    self.report({'ERROR'}, f"Edge fillet on {name} failed: {exc}")
                continue
            pre = obj[store.PROP_BREP]
            keep_history = features.history_enabled(context)
            if not keep_history:
                _frozen(obj, self)
            store.set_geometry_text(obj, breps["result"], context, keep_untrimmed=False)
            if keep_history:
                Mi = geom.world_matrix(obj).inverted()
                steps.store_rims(obj, [list(Mi @ Vector(m)) for m in mids], info, Mi)
                features.push_step(obj, "edge_fillet", [], {
                    "mids_local": [list(Mi @ Vector(m)) for m in mids], "shape": self.shape, "radius": self.radius,
                    "end_radius": self.end_radius if self.variable and not chord else None, "chord": chord,
                    "continuity": int(self.continuity), "thumbweight": self.thumbweight,
                    **self._ca_params()}, base_text=pre)
            done.append(obj)
            msg = f"{name}: {info['edges']} edge(s) filleted" + (" (solid)" if info.get("solid") else "")
            if info.get("chord"):
                lo, mean, hi = info["chord"]
                msg += (f" · chord {lo:.4g}..{hi:.4g} m" if hi - lo > 1e-3 * mean else f" · chord {mean:.4g} m") \
                    + f" (ball radius {info['radius']:.4g} m)"
            level = 'INFO'
            if info.get("rings") and "class_a" not in info:
                msg += " · " + ring_summary(info["rings"])
            if "class_a" in info:
                ca_msg, ok = class_a_summary(info["class_a"])
                msg += " · " + ca_msg
                level = 'INFO' if ok else 'WARNING'
                obj["nproto_class_a"] = json.dumps(info["class_a"])
            elif "nproto_class_a" in obj:
                del obj["nproto_class_a"]
            self.report({level}, msg)
        if not done:
            return {'CANCELLED'}
        _select_only(context, done)
        return {'FINISHED'}

    def _ca_params(self):
        return {"class_a": bool(self.class_a), "ca_tolerance": self.ca_tolerance, "ca_angle": self.ca_angle,
                "ca_degree": int(self.ca_degree), "ca_continuity": int(self.ca_continuity)}

    def draw(self, context):
        lay = self.layout
        lay.prop(self, "shape", expand=True)
        lay.row(align=True).prop(self, "rail_type", expand=True)
        col = lay.column(align=True)
        if self.rail_type == "CHORD":
            col.prop(self, "radius", text="Chord")
        else:
            col.prop(self, "radius", text="Distance" if self.shape == "CHAMFER" else "Radius")
            col.prop(self, "variable")
            sub = col.row()
            sub.enabled = self.variable
            sub.prop(self, "end_radius")
        if self.shape == "BLEND":
            row = lay.row(align=True)
            row.prop(self, "continuity", expand=True)
            lay.prop(self, "thumbweight")
        box = lay.box().column(align=True)
        box.prop(self, "class_a")
        if self.class_a:
            box.prop(self, "ca_tolerance")
            box.prop(self, "ca_angle")
            box.prop(self, "ca_degree")
            box.row(align=True).prop(self, "ca_continuity", expand=True)


def ring_summary(rings):
    """What became of the fillet faces round closed edges (rims)."""
    out = []
    for g in rings or []:
        k = g.get("kernel_cvs", [0, 0])
        out.append(f"rim {k[0]}x{k[1]} CVs -> " + ("exact revolution, 9 CVs round" if g.get("ring") == "revolution"
                                                  else f"one even ring, {g.get('round')} CVs round "
                                                       f"(degree {g.get('degree', 5)})"))
    return "; ".join(out)


def class_a_summary(rep):
    """One line for the Class-A rebuild report; (text, all within tolerance)."""
    rings = ring_summary(rep.get("rings"))
    if rep.get("error"):
        return rep["error"] + (f" · {rings}" if rings else ""), False
    faces = [f for f in rep.get("faces", []) if "skipped" not in f]
    skipped = [f["skipped"] for f in rep.get("faces", []) if "skipped" in f]
    if not faces:
        if rings:
            return "Class-A: " + rings + (" · kept: " + "; ".join(skipped) if skipped else ""), not skipped
        return "Class-A: " + (rep.get("note") or ("; ".join(skipped) or "nothing to rebuild")), not skipped
    parts = []
    for f in faces:
        k = f.get("kernel_cvs", [0, 0])
        n = f["spans"]
        parts.append(f"{k[0]}x{k[1]} CVs -> {n} single-span patch{'es' if n > 1 else ''} "
                     f"{f.get('rows', 3)}x{f['degree'] + 1} (dev {f['dev']:.2g}, {f['angle']:.2g} deg, "
                     f"ends {'/'.join(f['ends'])})")
    ok = all(f["within"] for f in faces) and not skipped
    txt = "Class-A: " + "; ".join(parts + ([rings] if rings else []))
    if not all(f["within"] for f in faces):
        txt += " -- outside tolerance (raise Tolerance or Max Degree)"
    if skipped:
        txt += " · kept: " + "; ".join(skipped)
    return txt, ok


BOOL_ITEMS = [("UNION", "Union", "Material from any of the bodies"),
              ("DIFFERENCE", "Difference", "The active body minus the other selected bodies"),
              ("INTERSECTION", "Intersection", "Only the material the bodies share")]
BOOL_SUFFIX = {"UNION": "union", "DIFFERENCE": "difference", "INTERSECTION": "intersection"}


class NPROTO_OT_boolean(bpy.types.Operator):
    """Union, difference or intersection of closed solids. The active object is
    the target (for Difference, the body that is cut); the other selected solids
    are the tools. With history the inputs are hidden and the result follows them"""
    bl_idname = "nproto.boolean"
    bl_label = "Boolean"
    bl_options = {'REGISTER', 'UNDO'}

    operation: EnumProperty(name="Operation", items=BOOL_ITEMS, default="UNION")
    names: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})   # target first, then the tools

    def invoke(self, context, event):
        act = context.active_object
        sel = [o for o in context.selected_objects if store.is_nurbs(o)]
        sel.sort(key=lambda o: o != act)  # active first = the target
        self.names = json.dumps([o.name for o in sel])
        return self.execute(context)

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not self.names:
            act = context.active_object
            self.names = json.dumps([o.name for o in sorted(
                (o for o in context.selected_objects if store.is_nurbs(o)), key=lambda o: o != act)])
        objs = [bpy.data.objects[n] for n in json.loads(self.names) if n in bpy.data.objects]
        if len(objs) < 2:
            self.report({'WARNING'}, "Select two or more closed NURBS solids (the active one is the target)")
            return {'CANCELLED'}
        target = objs[0]
        name = f"{target.name}_{BOOL_SUFFIX[self.operation]}"
        if features.history_enabled(context):
            try:
                g = features.run("boolean", [(o, "SURF") for o in objs], {"bool_op": self.operation})
            except jobs.JobError as exc:
                self.report({'ERROR'}, f"Boolean failed: {exc}")
                return {'CANCELLED'}
            res = store.new_object_text(context, name, g[1])
            features.record(res, "boolean", [(o, "SURF") for o in objs], {"bool_op": self.operation})
            features.keep_inputs(context, objs)
            features.mark_built(res)
            _select_only(context, [res])
            self.report({'INFO'}, f"Boolean {self.operation.lower()} of {len(objs)} solids "
                                  "(inputs kept hidden: history; Restore Inputs brings them back)")
            return {'FINISHED'}
        keys = {f"s{i}": o for i, o in enumerate(objs)}
        try:
            info, breps = jobs.run("boolean", keys, ["result"], order=list(keys), into="s0",
                                   operation=self.operation, names={k: o.name for k, o in keys.items()})
        except jobs.JobError as exc:
            self.report({'ERROR'}, f"Boolean failed: {exc}")
            return {'CANCELLED'}
        _frozen(target, self)
        store.set_geometry_text(target, breps["result"], context, keep_untrimmed=False)
        if store.PROP_UNTRIM in target:
            del target[store.PROP_UNTRIM]
        for o in objs[1:]:
            _remove_object(o)
        _select_only(context, [target])
        self.report({'INFO'}, f"Boolean {self.operation.lower()}: {info['faces']} faces, "
                              f"volume {info.get('volume', 0):.4g}")
        return {'FINISHED'}


class NPROTO_OT_boundary_fill(bpy.types.Operator):
    """Boundary fill: the closed solid that the selected surfaces enclose. Each
    surface is trimmed by the others and the pieces around the enclosed volume
    are joined into a solid (everything outside is discarded). With history
    the surfaces are hidden and the solid follows them"""
    bl_idname = "nproto.boundary_fill"
    bl_label = "Boundary Fill"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        from .cvs import is_cage
        objs = [o for o in context.selected_objects if store.is_nurbs(o) and not is_cage(o)]
        if len(objs) < 2:
            self.report({'WARNING'}, "Select the surfaces around the volume (two or more)")
            return {'CANCELLED'}
        objs.sort(key=lambda o: o.name)
        ins_ = [(o, "SURF") for o in objs]
        try:
            g = features.run("boundary_fill", ins_, {})
        except jobs.JobError as exc:
            self.report({'ERROR'}, f"Boundary Fill failed: {exc}")
            return {'CANCELLED'}
        res = store.new_object_text(context, "Solid", g[1])
        if features.history_enabled(context):
            features.record(res, "boundary_fill", ins_, {})
            features.mark_built(res)
            features.keep_inputs(context, objs)        # Restore Inputs brings them back
        else:
            for o in objs:
                _remove_object(o)                       # the solid replaces them
        _select_only(context, [res])
        b = store.get_brep(res)
        try:
            vol = abs(b.volume(1e-6)) if b.is_manifold_solid() else 0.0
        except Exception:
            vol = 0.0
        self.report({'INFO'}, f"Boundary Fill: solid of {b.face_count()} faces, volume {vol:.4g}")
        return {'FINISHED'}


class NPROTO_OT_flip_normal(bpy.types.Operator):
    """Turn the selected surfaces over: their normals (shading, matcaps, offset
    direction) point the other way. Works on trimmed faces and joined sheets;
    closed solids always face outward"""
    bl_idname = "nproto.flip_normal"
    bl_label = "Flip Normal"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        from .cvs import is_cage
        objs = [o for o in context.selected_objects if store.is_nurbs(o) and not is_cage(o)]
        if not objs:
            self.report({'WARNING'}, "Select NURBS surfaces")
            return {'CANCELLED'}
        done = []
        for o in objs:
            old = features._edge_marks(o)
            try:
                if features.history_enabled(context):
                    entries = features.stack(o)
                    if entries and entries[-1]["op"] == "flip_normal":
                        features.remove_step(o, len(entries) - 1)      # flipping back: drop the step
                        if features.history(o) is None and not features.stack(o) and features.PROP_BASE in o:
                            # nothing left but the snapshot: it is the geometry again
                            nat = bool(o.get(features.PROP_BASE + "_natural", 0))
                            store.set_geometry_text(o, o[features.PROP_BASE], context, keep_untrimmed=False)
                            o[store.PROP_NATURAL] = 1 if nat else 0
                            for k in (features.PROP_BASE, features.PROP_BASE + "_natural"):
                                if k in o:
                                    del o[k]
                        else:
                            features.rebuild(o, context, force=True)
                    else:
                        features.push_step(o, "flip_normal", [], {})
                        features.rebuild(o, context, force=True)
                else:
                    nat = store.is_natural(o)
                    untrim = o.get(store.PROP_UNTRIM)
                    store.set_geometry_text(o, steps.flip_text(o[store.PROP_BREP]), context, keep_untrimmed=False)
                    o[store.PROP_NATURAL] = 1 if nat else 0
                    if untrim and not nat:
                        o[store.PROP_UNTRIM] = steps.flip_text(untrim)
                features._remap_dependents(o, old)
                done.append(o)
            except Exception as exc:
                self.report({'ERROR'}, f"{o.name}: {exc}")
        if not done:
            return {'CANCELLED'}
        features.mark_built(*done)
        self.report({'INFO'}, "Flipped " + ", ".join(o.name for o in done))
        return {'FINISHED'}


classes = (NPROTO_OT_trim, NPROTO_OT_untrim, NPROTO_OT_fillet, NPROTO_OT_split, NPROTO_OT_join, NPROTO_OT_unjoin, NPROTO_OT_restore_inputs,
           NPROTO_OT_edge_fillet, NPROTO_OT_boolean, NPROTO_OT_flip_normal, NPROTO_OT_boundary_fill)
