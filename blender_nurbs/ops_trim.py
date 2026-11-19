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

from . import draw, features, geom, hints, jobs, steps, store
from .ops import AskValue, PickMixin, _kernel_ok, _near_candidates, _pick_candidates, _select_only
from .valuedrag import ValueDrag, ruled_preview


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


def _frozen(obj, op):
    """Trimming replaces the B-rep: construction history can no longer rebuild it."""
    if features.has_history(obj):
        features.clear_history(obj)
        op.report({'INFO'}, f"{obj.name}: construction history removed")


# --------------------------------------------------------------------------
# Trim
# --------------------------------------------------------------------------

def region_under(context, event, regions):
    """Index of the region (dicts with world "tris") the mouse ray meets first."""
    region, rv3d = context.region, context.region_data
    mouse = (event.mouse_region_x, event.mouse_region_y)
    o = np.array(view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse), float)
    dvec = np.array(view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse), float)
    best, best_t = None, np.inf
    for ri, r in enumerate(regions):
        T = r["tris"].astype(float)
        if not len(T):
            continue
        e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        pv = np.cross(dvec, e2)
        det = np.einsum("ij,ij->i", e1, pv)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - T[:, 0]
        u = np.einsum("ij,ij->i", tv, pv) * inv
        qv = np.cross(tv, e1)
        v = (qv @ dvec) * inv
        t = np.einsum("ij,ij->i", e2, qv) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        if hit.any():
            tm = float(t[hit].min())
            if tm < best_t:
                best, best_t = ri, tm
    return best


class NPROTO_OT_trim(bpy.types.Operator):
    """Trim a surface with projected curves or intersecting surfaces.
    Click the surface on the side to keep, pick cutting curves / edges / surfaces,
    press Enter to see the regions, click regions to keep / discard them, Enter to trim"""
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
                                  "(cookie cutter); non-planar curves use the view direction. A curve "
                                  "lying on the surface cuts along itself"),
        ("VIEW", "View", "Project along the view direction"),
        ("NORMAL", "Surface Normal", "Project along the surface normal at the pick point"),
        ("SURFACE", "On Surface", "No projection: cut along the curve itself, where it lies on or near the "
                                  "surface (curves on the surface do this in Curve Normal too)"),
        ("X", "World X", ""), ("Y", "World Y", ""), ("Z", "World Z", ""),
    ], default="CURVE")
    extend: BoolProperty(name="Extend Cutters", default=True,
                         description="A curve, line, edge or surface whose cut stops inside the surface is "
                                     "extended (its natural continuation) until it cuts all the way across")
    reach: EnumProperty(name="Cut", items=[
        ("BOTH", "Both Ways", "A projected curve cuts on both sides of itself (a circle inside a vase "
                              "cuts the front and the back wall)"),
        ("FRONT", "Along Normal", "Only where the projection direction points from the curve: for Curve "
                                  "Normal, the curve object's blue Z axis (the sketch normal)"),
        ("BACK", "Against Normal", "Only on the other side of the curve"),
    ], default="BOTH")
    through: BoolProperty(name="Surfaces Through", default=False,
                          description="A cutting surface is extended (its natural continuation) past its open "
                                      "edges until it cuts all the way through the part (a short tube through "
                                      "both walls)")
    # regions marked in the viewport before trimming: [[world point, keep?], ...]
    marks: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    # ---- interactive picking
    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._stage = 0
        self._cuts = []
        self._target = None
        self._regions, self._keep, self._hover = [], [], None
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["hover"] = None
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {'RUNNING_MODAL'}

    def _status(self, context, extra=""):
        from . import hints
        if self._stage == 0:
            state = ["click the surface region to keep / discard", "Action and Project in F9"]
        elif self._stage == 1:
            state = [f"{self._target.name}: pick cutting curves / edges, or click a cutting surface",
                     f"{len(self._cuts)} picked"]
        else:
            k = sum(self._keep)
            state = [f"{self._target.name}: click a region to keep / discard it",
                     f"{k} kept · {len(self._keep) - k} discarded"]
        state.append(extra.strip(" ·[]"))
        if self._stage < 2:
            rows = [("Pick", [hints.key("Tab", "cycle overlapping", on=True if getattr(self, "_ties", None) else None),
                              hints.key("Backspace", "undo"),
                              hints.key("Enter", "show regions", on=True if self._stage and self._cuts else False)])]
        else:
            reach = {"BOTH": "both ways", "FRONT": "along normal", "BACK": "against normal"}[self.reach]
            keys = []
            if any(r != "SURF" for _, r in self._cuts):         # only a projected curve has a direction
                keys.append(hints.key("F", f"cut: {reach}"))
            if any(r == "SURF" for _, r in self._cuts):
                keys.append(hints.key("E", "surfaces through", on=self.through))
            rows = [("Regions", keys + [hints.key("I", "invert"),
                                 hints.key("Backspace", "back to cutters"),
                                 hints.key("Enter", "trim", on=True)])]
        hints.show(context, "Trim", state, rows, [("MOUSE_LMB", "Pick"), ("EVENT_ESC", "Cancel")])

    def _end(self, context):
        from . import hints
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["hover"] = None
        draw.pick_overlay["regions"] = []
        draw.pick_overlay["region_hover"] = None
        draw.pick_overlay["region_pal"] = None
        hints.clear(context)

    def _job_inputs(self, cutter_pairs):
        """(curves, cutting surfaces, direction) of the worker job."""
        curves, cutter_objs = [], {}
        for obj, ref in cutter_pairs:
            if ref == "SURF":
                cutter_objs[f"c{len(cutter_objs)}"] = obj
            else:
                curves.append(steps.cutter_curve_json(features.input_curve(obj, ref), self.projection,
                                                      Vector(self.view_dir), steps.curve_up(obj)))
        if self.projection in ("NORMAL", "CURVE", "SURFACE"):  # CURVE: used for cutting surfaces only
            d = Vector(self.normal)
        elif self.projection == "VIEW":
            d = Vector(self.view_dir)
        else:
            d = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[self.projection])
        if d.length < 1e-9:
            d = Vector((0, 0, 1))
        return curves, cutter_objs, d.normalized()

    def _show_regions(self, context, prefer=None):
        """Cut the target into its regions (nothing is changed yet) and mark
        each keep / discard by the Action and Scope rule. prefer: {side of the
        cutters: keep?} from the regions shown before (Cut changed with F), so
        the same side stays kept wherever the click now falls. False: it failed."""
        try:
            curves, cutter_objs, d = self._job_inputs(self._cuts)
            regions, picked, _ = steps.trim_regions(self._target, cutter_objs, curves=curves, direction=list(d),
                                                    ref_point=list(self.ref_point), extend=bool(self.extend),
                                                    reach=self.reach, through=self.through)
        except (jobs.JobError, Exception) as exc:
            self.report({'WARNING'}, f"Trim: {exc}")
            return False
        if len(regions) < 2:
            self.report({'WARNING'}, "Trim: the cutters do not cut this surface into regions -- "
                                     "check that they cross it (along the projection direction for curves)")
            return False
        sides = [tuple(r["side"]) for r in regions]
        none = tuple(None for _ in sides[picked])
        if self.scope == "THROUGH" and sides[picked] != none:
            sel = {i for i, sd in enumerate(sides) if sd == sides[picked]}
        else:
            sel = {picked}
        self._keep = [(i in sel) == (self.action != "DELETE") for i in range(len(regions))]
        for i, sd in enumerate(sides):
            if prefer and sd != none and sd in prefer:
                self._keep[i] = prefer[sd]
        M = self._target.matrix_world
        self._regions = []
        for r in regions:
            tris, border = steps.region_mesh(r["text"], M)
            self._regions.append({"point": r["point"], "tris": tris, "border": border, "side": tuple(r["side"])})
        self._hover = None
        self._sync_regions()
        return True

    def _sync_regions(self):
        draw.pick_overlay["regions"] = [(r["tris"], r["border"], k) for r, k in zip(self._regions, self._keep)]
        draw.pick_overlay["region_hover"] = self._hover

    def _region_under(self, context, event):
        return region_under(context, event, self._regions)


    def modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        from .ops import PickMixin
        if event.type == 'INBETWEEN_MOUSEMOVE':
            return {'PASS_THROUGH'}       # not used here; a gizmo drag needs them unhandled
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

        if self._stage == 2:
            return self._modal_regions(context, event)

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
            context.window.cursor_set('WAIT')
            shown = self._show_regions(context)
            context.window.cursor_set('DEFAULT')
            if shown:
                self._stage = 2
                draw.pick_overlay["hover"] = None
            self._status(context)
            context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _modal_regions(self, context, event):
        """Stage 2: the regions are shown (green keep, red discard); a click
        flips one, Enter trims."""
        if event.type == 'MOUSEMOVE':
            h = self._region_under(context, event)
            if h != self._hover:
                self._hover = h
                self._sync_regions()
                context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            h = self._region_under(context, event)
            if h is not None:
                self._keep[h] = not self._keep[h]
                self._sync_regions()
                self._status(context)
                context.area.tag_redraw()
        elif event.type == 'E' and event.value == 'PRESS' and any(r == "SURF" for _, r in self._cuts):
            prefer = {}
            for r, k in zip(self._regions, self._keep):
                prefer[r["side"]] = prefer.get(r["side"], False) or k
            self.through = not self.through
            context.window.cursor_set('WAIT')
            if not self._show_regions(context, prefer):
                self.through = not self.through
                self._show_regions(context, prefer)
            context.window.cursor_set('DEFAULT')
            self._status(context)
            context.area.tag_redraw()
        elif event.type == 'F' and event.value == 'PRESS' and any(r != "SURF" for _, r in self._cuts):
            order = ["BOTH", "FRONT", "BACK"]
            prefer = {}       # keep the same side of the cutters (a side with mixed marks: kept if any is)
            for r, k in zip(self._regions, self._keep):
                prefer[r["side"]] = prefer.get(r["side"], False) or k
            self.reach = order[(order.index(self.reach) + 1) % 3]
            context.window.cursor_set('WAIT')
            if not self._show_regions(context, prefer):        # nothing cut that way: try the next one
                self.reach = order[(order.index(self.reach) + 1) % 3]
                self._show_regions(context, prefer)
            context.window.cursor_set('DEFAULT')
            self._status(context)
            context.area.tag_redraw()
        elif event.type == 'I' and event.value == 'PRESS':
            self._keep = [not k for k in self._keep]
            self._sync_regions()
            self._status(context)
            context.area.tag_redraw()
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS':
            self._stage = 1
            draw.pick_overlay["regions"] = []
            draw.pick_overlay["region_hover"] = None
            self._status(context)
            context.area.tag_redraw()
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if not any(self._keep):
                self.report({'WARNING'}, "Trim: every region is marked discard")
                return {'RUNNING_MODAL'}
            self._end(context)
            self.target = self._target.name
            self.cutters = json.dumps([[o.name, r] for o, r in self._cuts])
            self.marks = json.dumps([[list(r["point"]), bool(k), list(r["side"])]
                                     for r, k in zip(self._regions, self._keep)])
            if self.action == "DELETE" or (self.action == "SPLIT" and all(self._keep)):
                self.action = "KEEP"          # the marks say what goes; Split needs something discarded
            return self.execute(context)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "action")
        if self.marks:
            try:
                m = json.loads(self.marks)
            except ValueError:
                m = []
            k = sum(1 for mk in m if mk[1])
            col = lay.column(align=True)
            col.label(text=f"Regions marked in the viewport: {k} kept, {len(m) - k} discarded")
            col.label(text="Discard inverts the marks; Split makes the discarded ones an object")
        else:
            lay.prop(self, "scope")
        lay.prop(self, "projection")
        lay.prop(self, "reach")
        lay.prop(self, "through")
        lay.prop(self, "extend")

    # ---- the actual trim
    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        tgt = bpy.data.objects.get(self.target)
        if tgt is None or not store.is_nurbs(tgt):
            self.report({'ERROR'}, "Trim: no target surface")
            return {'CANCELLED'}
        try:
            pairs = [(bpy.data.objects[n], r) for n, r in json.loads(self.cutters or "[]")]
            curves, cutter_objs, d = self._job_inputs(pairs)
            marks = json.loads(self.marks) if self.marks else []
        except Exception as exc:
            self.report({'ERROR'}, f"Trim: bad cutter ({exc})")
            return {'CANCELLED'}
        outs = ["KEEP", "DELETE"] if self.action == "SPLIT" else [self.action]
        pre_text = tgt[store.PROP_BREP]
        try:
            from .steps import run_trim
            res, breps = run_trim(tgt, cutter_objs, outs, curves=curves,
                                  direction=list(d.normalized()), ref_point=list(self.ref_point),
                                  action=self.action, scope=self.scope, extend=bool(self.extend), marks=marks,
                                  reach=self.reach, through=self.through)
        except jobs.JobError as exc:
            self.report({'ERROR'}, f"Trim failed: {exc}")
            return {'CANCELLED'}
        crv_in = [(bpy.data.objects[n], r) for n, r in json.loads(self.cutters or "[]") if r != "SURF"]
        for msg in steps.extended_notes(res.get("extended"), crv_in, cutter_objs, tgt.name):
            self.report({'INFO'}, msg)
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
                      "side": (res.get("info") or {}).get("side"), "extend": bool(self.extend),
                      "reach": self.reach, "through": self.through}
            if marks:
                params["marks_local"] = [[list(Mi @ Vector(mk[0]))] + list(mk[1:]) for mk in marks]
            features.push_step(tgt, "trim", inputs, params, base_text=pre_text)
        if self.action == "SPLIT":
            other = store.new_object_text(context, tgt.name + "_split", breps["DELETE"], geom.world_matrix(tgt))
            other[store.PROP_UNTRIM] = tgt[store.PROP_UNTRIM]
            other[store.PROP_UNTRIM + "_natural"] = tgt.get(store.PROP_UNTRIM + "_natural", 0)
            if keep_history:  # same history, last step keeps the other side
                features.copy_history(tgt, other)
                entries = features.stack(other)
                entries[-1]["params"] = dict(entries[-1]["params"], action="DELETE")
                features._write_stack(other, entries)
            made.append(other)
        _select_only(context, made)
        return {'FINISHED'}


TRIM_OPS = {"trim", "fillet_trim", "untrim_region"}


def _split_piece(o):
    """A piece of a Split with history: its feature gives the trimmed piece,
    so untrims are history steps (set straight on it, a rebuild undoes them)."""
    return store.is_nurbs(o) and features.untrimmed_base(o) is not None


def untrim_all(o, context):
    """Back to the surface before its first trim. False: nothing to undo."""
    if _split_piece(o):
        if store.is_natural(o):
            return False
        entries = [e for e in features.stack(o) if e["op"] not in TRIM_OPS | {"untrim"}]
        features._write_stack(o, [{"op": "untrim", "inputs": [], "params": {}}] + entries)
        features.rebuild(o, context, force=True)
        return True
    if store.is_nurbs(o) and features.PROP_STACK in o and any(
            e["op"] in ("trim", "fillet_trim") for e in features.stack(o)):
        features.remove_steps(o, TRIM_OPS)
        if features.PROP_STACK not in o and features.PROP_HIST not in o:
            text, nat = o[features.PROP_BASE], o.get(features.PROP_BASE + "_natural", 0)
            store.set_geometry_text(o, text, context, keep_untrimmed=False)
            o[store.PROP_NATURAL] = nat
            del o[features.PROP_BASE]
            if store.PROP_UNTRIM in o:
                del o[store.PROP_UNTRIM]
        else:
            features.rebuild(o, context, force=True)
        return True
    if store.is_nurbs(o) and store.PROP_UNTRIM in o:
        natural = o.get(store.PROP_UNTRIM + "_natural", 0)
        store.set_geometry_text(o, o[store.PROP_UNTRIM], context, keep_untrimmed=False)
        o[store.PROP_NATURAL] = natural
        del o[store.PROP_UNTRIM]
        return True
    return False


class NPROTO_OT_untrim(bpy.types.Operator):
    """Restore selected surfaces to how they were before their first trim"""
    bl_idname = "nproto.untrim"
    bl_label = "Untrim"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        n = sum(1 for o in context.selected_objects if untrim_all(o, context))
        self.report({'INFO'}, f"Untrimmed {n} object(s)")
        return {'FINISHED'} if n else {'CANCELLED'}


def _trim_history(o):
    return features.PROP_STACK in o and any(e["op"] in ("trim", "fillet_trim") for e in features.stack(o))


def _untrimmable(o):
    if _split_piece(o):
        return not store.is_natural(o)
    return store.is_nurbs(o) and (store.PROP_UNTRIM in o or _trim_history(o))


def _untrim_reference(o):
    """(untrimmed text, as a history step?) for Untrim Region on o."""
    if _split_piece(o):
        return steps.untrim_reference(o), True
    if _trim_history(o):
        return features.base_text(o), True
    return o[store.PROP_UNTRIM], False


class NPROTO_OT_untrim_region(bpy.types.Operator):
    """Untrim some areas of trimmed surfaces: what was trimmed away shows in red;
    click the areas to bring back (green), Enter untrims them. The other trims
    stay. Works on the selected trimmed surfaces, or every visible one"""
    bl_idname = "nproto.untrim_region"
    bl_label = "Untrim Region"
    bl_options = {'REGISTER', 'UNDO'}

    target: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    point: FloatVectorProperty(name="Edge Point", subtype='XYZ', options={'HIDDEN', 'SKIP_SAVE'})
    points: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})     # JSON: world points inside removed areas

    # ---- the work (one object)
    def untrim(self, context, obj, point=None, points=None):
        """Give back the area at the cut edge nearest `point`, or the areas
        nearest `points`. Returns an error text or None."""
        try:
            untrimmed, hist = _untrim_reference(obj)
            text, info = steps.untrim_region_text(obj[store.PROP_BREP], untrimmed, jobs.xform(obj), point, points)
        except (jobs.JobError, KeyError) as exc:
            return str(exc)
        if info.get("cuts_left", 1) == 0:        # nothing trimmed is left: the whole surface again
            untrim_all(obj, context)
            return None
        if hist:
            Mi = geom.world_matrix(obj).inverted()
            p = {"points_local": [list(Mi @ Vector(q)) for q in points]} if points \
                else {"point_local": list(Mi @ Vector(point))}
            features.push_step(obj, "untrim_region", [], p)
            features.rebuild(obj, context, force=True)
        else:
            store.set_geometry_text(obj, text, context)        # keeps the untrimmed surface for Untrim
        return None

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if self.points:       # from the viewport (or a script): {object: [points]}
            done, errs = 0, []
            for name, pts in json.loads(self.points).items():
                obj = bpy.data.objects.get(name)
                err = "not a trimmed surface" if obj is None or not _untrimmable(obj) else \
                    self.untrim(context, obj, points=[Vector(q) for q in pts])
                if err:
                    errs.append(f"{name}: {err}")
                else:
                    done += 1
            for e in errs:
                self.report({'WARNING'}, f"Untrim Region: {e}")
            return {'FINISHED'} if done else {'CANCELLED'}
        obj = bpy.data.objects.get(self.target)
        if obj is None or not _untrimmable(obj):
            self.report({'ERROR'}, "Untrim Region: pick an edge of a trimmed surface")
            return {'CANCELLED'}
        err = self.untrim(context, obj, Vector(self.point))
        if err:
            self.report({'ERROR'}, f"Untrim Region: {err}")
            return {'CANCELLED'}
        return {'FINISHED'}

    # ---- interactive: the removed areas show; click the ones to bring back
    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        objs = [o for o in context.selected_objects if _untrimmable(o)] \
            or [o for o in context.visible_objects if _untrimmable(o)]
        if not objs:
            self.report({'WARNING'}, "Untrim Region: no trimmed surface (select one)")
            return {'CANCELLED'}
        self._regions, self._keep, self._hover = [], [], None
        context.window.cursor_set('WAIT')
        for o in objs:
            try:
                untrimmed, _ = _untrim_reference(o)
                for text, pt in steps.untrim_preview(o[store.PROP_BREP], untrimmed, jobs.xform(o)):
                    tris, border = steps.region_mesh(text, o.matrix_world)
                    self._regions.append({"target": o.name, "point": pt, "tris": tris, "border": border})
                    self._keep.append(False)
            except (jobs.JobError, KeyError) as exc:
                self.report({'WARNING'}, f"Untrim Region: {o.name}: {exc}")
        context.window.cursor_set('DEFAULT')
        if not self._regions:
            return {'CANCELLED'}
        self._sync()
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {'RUNNING_MODAL'}

    def _sync(self):
        draw.pick_overlay["regions"] = [(r["tris"], r["border"], k) for r, k in zip(self._regions, self._keep)]
        draw.pick_overlay["region_hover"] = self._hover

    def _status(self, context):
        from . import hints
        k = sum(self._keep)
        state = ["click what was trimmed away (red) to bring it back (green)",
                 f"{k} of {len(self._keep)} to untrim"]
        rows = [("Regions", [hints.key("A", "all"), hints.key("Enter", "untrim", on=bool(k))])]
        hints.show(context, "Untrim Region", state, rows, [("MOUSE_LMB", "Switch area"), ("EVENT_ESC", "Cancel")])

    def _end(self, context):
        from . import hints
        draw.pick_overlay["regions"] = []
        draw.pick_overlay["region_hover"] = None
        hints.clear(context)

    def modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM',
                          'INBETWEEN_MOUSEMOVE'}:
            return {'PASS_THROUGH'}
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event):
            return {'PASS_THROUGH'}
        from . import hints
        if hints.handle(event):
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._end(context)
            context.area.tag_redraw()
            return {'CANCELLED'}
        if event.type == 'MOUSEMOVE':
            h = region_under(context, event, self._regions)
            if h != self._hover:
                self._hover = h
                self._sync()
                context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            h = region_under(context, event, self._regions)
            if h is not None:
                self._keep[h] = not self._keep[h]
                self._sync()
                self._status(context)
                context.area.tag_redraw()
        elif event.type == 'A' and event.value == 'PRESS':
            self._keep = [not all(self._keep)] * len(self._keep)
            self._sync()
            self._status(context)
            context.area.tag_redraw()
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if not any(self._keep):
                return {'RUNNING_MODAL'}
            pts = {}
            for r, k in zip(self._regions, self._keep):
                if k:
                    pts.setdefault(r["target"], []).append(list(r["point"]))
            self.points = json.dumps(pts)
            self._end(context)
            context.window.cursor_set('WAIT')
            res = self.execute(context)
            context.window.cursor_set('DEFAULT')
            return res
        return {'RUNNING_MODAL'}


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


class NPROTO_OT_fillet(ValueDrag, AskValue, bpy.types.Operator):
    """Fillet or chamfer between two surfaces: select both, or click them.
    Where they cross, the fillet is previewed with a radius arrow: drag it,
    type a value, A / B put it on the other side of a surface (where there is
    room), click away or Enter to make it"""
    bl_idname = "nproto.fillet"
    bl_label = "Fillet / Chamfer"
    bl_options = {'REGISTER', 'UNDO'}

    surf_a: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    surf_b: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_pending: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})  # set by invoke: check the radius once
    shape: EnumProperty(name="Shape", items=[
        ("CIRCULAR", "Circular", "Exact circular cross-section"),
        ("CHAMFER", "Chamfer", "Straight (linear) cross-section"),
        ("BLEND", "Blend", "Free-form blend, G1, G2 or G3 to both surfaces"),
    ], default="CIRCULAR")
    continuity: EnumProperty(name="Continuity", items=[("1", "G1", "Tangent to both surfaces"),
                                                       ("2", "G2", "Curvature continuous to both surfaces"),
                                                       ("3", "G3", "Curvature continuous, and changing at the "
                                                               "surfaces' rate (degree 7 across)")],
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
        given = [bpy.data.objects.get(n) for n in (self.surf_a, self.surf_b) if n]
        if not (len(given) == 2 and all(o is not None and store.is_nurbs(o) for o in given)):
            sel = [o for o in context.selected_objects if store.is_nurbs(o)]
            if len(sel) != 2:
                # one selected surface is A: click the other; none or many: click both
                return self._start_surface_pick(context, sel if len(sel) == 1 else [])
            act = context.active_object
            sel.sort(key=lambda o: o == act)  # active = B
            self.surf_a, self.surf_b = sel[0].name, sel[1].name
        self.fit_pending = True
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        return self.ask(context)

    # -- clicking the two surfaces (nothing or one selected) -----------------
    def _start_surface_pick(self, context, first):
        if context.area is None or context.area.type != 'VIEW_3D' or bpy.app.background:
            self.report({'WARNING'}, "Select two NURBS surfaces")
            return {'CANCELLED'}
        self._surf_picks = list(first)
        self._sel0 = ([o.name for o in context.selected_objects],
                      context.view_layer.objects.active.name if context.view_layer.objects.active else None)
        self._show_picks(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _show_picks(self, context):
        for o in context.selected_objects:
            o.select_set(False)
        for o in self._surf_picks:
            o.select_set(True)
        if self._surf_picks:
            context.view_layer.objects.active = self._surf_picks[-1]
        which = "A" if not self._surf_picks else "B"
        state = [f"click surface {which}"] + ([f"A: {self._surf_picks[0].name}"] if self._surf_picks else [])
        hints.show(context, self.bl_label, state,
                   [("Pick", [hints.key("Backspace", "undo", on=True if self._surf_picks else False)])],
                   [("MOUSE_LMB", "Pick"), ("EVENT_ESC", "Cancel")])
        context.area.tag_redraw()

    def _pick_modal(self, context, event):
        from .ops import PickMixin
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM',
                          'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            return {'PASS_THROUGH'}
        if event.type.startswith(("NUMPAD_", "NDOF_")) and event.type != 'NUMPAD_ENTER':
            return {'PASS_THROUGH'}          # turning the view
        if hints.handle(event):
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE'} and not PickMixin._in_view(context, event):
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            obj = self._surface_under(context, event)
            if obj is not None and obj not in self._surf_picks:
                self._surf_picks.append(obj)
                self._show_picks(context)
                if len(self._surf_picks) == 2:
                    hints.clear(context)
                    # the value stage is a modal of its own: start the command again with both
                    a, b = self._surf_picks
                    bpy.ops.nproto.fillet('INVOKE_DEFAULT', surf_a=a.name, surf_b=b.name)
                    return {'CANCELLED'}
            return {'RUNNING_MODAL'}
        if event.type == 'BACK_SPACE' and event.value == 'PRESS' and self._surf_picks:
            self._surf_picks.pop()
            self._show_picks(context)
            return {'RUNNING_MODAL'}
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            names, act = self._sel0
            for o in context.selected_objects:
                o.select_set(False)
            for n in names:
                if n in bpy.data.objects:
                    bpy.data.objects[n].select_set(True)
            context.view_layer.objects.active = bpy.data.objects.get(act) if act else None
            hints.clear(context)
            context.area.tag_redraw()
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _surface_under(self, context, event):
        # the 3D view's main region, whichever region the tool was started from (a sidebar button)
        region, rv3d = self._win(context)
        m = (event.mouse_x - region.x, event.mouse_y - region.y)
        o = view3d_utils.region_2d_to_origin_3d(region, rv3d, m)
        d = view3d_utils.region_2d_to_vector_3d(region, rv3d, m)
        hit, _, _, _, obj, _ = context.scene.ray_cast(context.evaluated_depsgraph_get(), o, d)
        return obj.original if hit and obj is not None and store.is_nurbs(obj.original) else None

    def modal(self, context, event):
        if getattr(self, "_surf_picks", None) is not None:
            return self._pick_modal(context, event)
        return ValueDrag.modal(self, context, event)

    # -- the live value stage (valuedrag.py) ----------------------------------
    fit_view = "shrink"         # the radius fit_radius found for the part: only ever made smaller
    start_value = 0.0           # the arrow at the crossing: no fillet until it is dragged (L: the last radius)
    SHAPES = ("CIRCULAR", "CHAMFER", "BLEND")

    @property
    def drag_fields(self):
        return ("radius",) if (self.same_radius or self.rail_type == "CHORD") else ("radius", "radius_b")

    def _label(self, name):
        word = "Distance" if self.shape == "CHAMFER" else "Radius"
        if name == "radius" and self.rail_type == "CHORD":
            return "Chord"
        if name == "radius" and (self.same_radius or self.rail_type == "CHORD"):
            return word
        return f"{word} {'A' if name == 'radius' else 'B'}"

    def _sides(self):
        return (-1.0 if self.flip_a else 1.0), (-1.0 if self.flip_b else 1.0)

    def drag_setup(self, context, mouse):
        a, b = bpy.data.objects.get(self.surf_a), bpy.data.objects.get(self.surf_b)
        if a is None or b is None:
            return False
        self._sf = surface_fillet_frames(a, b)
        if not self._sf:
            return False               # surfaces that do not cross: the dialog
        try:
            self._fits = surface_fillet_sides(a, b, self._sf, part_size([a, b]))
        except Exception as exc:
            print("NURBS fillet sides:", exc)
            self._fits = {s: 1.0 for s in SIDES}
        best = max(self._fits.values())
        if best > 0 and self._fits[(self.flip_a, self.flip_b)] < best:
            # the side asked for has no room (surfaces meeting at their edges):
            # show the one the kernel's Auto Side would make
            self.flip_a, self.flip_b = next(s for s in SIDES if self._fits[s] == best)
        return True

    def _toggle_side(self, which):
        fa, fb = (not self.flip_a, self.flip_b) if which == "a" else (self.flip_a, not self.flip_b)
        fits = getattr(self, "_fits", None)
        if fits and fits[(fa, fb)] < max(fits.values()):
            several = sum(v == max(fits.values()) for v in fits.values()) > 1
            self._msg = (f"the other side of {which.upper()} has no room: a surface ends at the crossing"
                         if several else "only this side has room: the surfaces end at the crossing")
            return
        self.flip_a, self.flip_b = fa, fb

    def _chord_k(self, P, NA, NB):
        """2 sin(th/2) per sample (chord = k * ball radius), th the turn between the faces."""
        sa, sb = self._sides()
        ca, cb, A, B = surface_fillet_rails(P, NA, NB, sa, sb)
        D1, D2 = ca - A + cb, ca + cb - B
        D1 /= np.linalg.norm(D1, axis=1, keepdims=True) + 1e-300
        D2 /= np.linalg.norm(D2, axis=1, keepdims=True) + 1e-300
        th = np.pi - np.arccos(np.clip(np.sum(D1 * D2, axis=1), -0.9999, 0.9999))
        return 2.0 * np.sin(0.5 * np.maximum(th, np.radians(1.0)))

    def _radii(self, P, NA, NB):
        if self.rail_type == "CHORD":
            r = self.radius / self._chord_k(P, NA, NB)
            return r, r
        rb = self.radius if self.same_radius else self.radius_b
        return np.full(len(P), self.radius), np.full(len(P), rb)

    def drag_preview(self):
        sa, sb = self._sides()
        lines, tris = [], []
        for P, NA, NB in self._sf:
            ra, rb = self._radii(P, NA, NB)
            X = surface_fillet_sections(P, NA, NB, ra, rb, sa, sb, self.shape, continuity=self.continuity)
            L, T = ruled_preview(list(X), ribs=2)
            lines += L
            tris.append(T)
        return {"lines": lines, "tris": np.vstack(tris)}

    def _handle(self, field):
        P, NA, NB = self._sf[0]
        j = len(P) // 2
        sa, sb = self._sides()
        ca, cb, A, B = (x[0] for x in surface_fillet_rails(P[j:j + 1], NA[j:j + 1], NB[j:j + 1], sa, sb))
        ra, rb = (x[j] for x in self._radii(P, NA, NB))
        if field == "radius":           # the rail on A: P + ra (ca - A) + rb cb, along D1
            D = ca - A + cb
            D /= np.linalg.norm(D) or 1.0
            al, be = float((ca - A) @ D), float(cb @ D)
            if self.rail_type == "CHORD":
                k = float(self._chord_k(P[j:j + 1], NA[j:j + 1], NB[j:j + 1])[0])
                to_value, param = (lambda t: t * k / (al + be)), ra * (al + be)
            elif self.same_radius:
                to_value, param = (lambda t: t / (al + be)), ra * (al + be)
            else:
                to_value, param = (lambda t: (t - rb * be) / al), ra * al + rb * be
        else:                           # the rail on B: P + ra ca + rb (cb - B), along D2
            D = ca + cb - B
            D /= np.linalg.norm(D) or 1.0
            al, be = float(ca @ D), float((cb - B) @ D)
            to_value, param = (lambda t: (t - ra * al) / be), ra * al + rb * be
        return {"field": field, "kind": "arrow", "origin": P[j], "dir": D, "base": P[j], "param": param,
                "to_value": to_value}

    def drag_handles(self):
        return [self._handle(f) for f in self.drag_fields]

    def drag_key(self, context, event, ch):
        if ch == "c":
            self.shape = self.SHAPES[(self.SHAPES.index(self.shape) + 1) % len(self.SHAPES)]
        elif ch == "r":
            self.rail_type = "CHORD" if self.rail_type == "RADIUS" else "RADIUS"
            self._field = 0
        elif ch in ("a", "b"):
            self._toggle_side(ch)
        elif ch == "s":
            self.same_radius = not self.same_radius
            if not self.same_radius:
                self.radius_b = self.radius
            self._field = 0
        elif ch == "g" and self.shape == "BLEND":
            self.continuity = {"1": "2", "2": "3"}.get(self.continuity, "1")
        elif ch == "t":
            self.trim_bases = not self.trim_bases
        else:
            return False
        return True

    def drag_handle_hints(self):
        if len(self.drag_fields) > 1:
            return [hints.key("LMB drag", "arrows: radius on A, on B")]
        return [hints.key("LMB drag", "arrow: chord" if self.rail_type == "CHORD" else "arrow: radius")]

    def drag_rows(self):
        rows = [("Shape", [hints.cycle("C", ["circular", "chamfer", "blend"], self.shape.lower()),
                           hints.cycle("R", ["rolling ball", "chord"],
                                       "rolling ball" if self.rail_type == "RADIUS" else "chord", "size"),
                           hints.toggle("S", "same radius", self.same_radius)]
                 + ([hints.cycle("G", ["G1", "G2", "G3"], "G" + self.continuity)] if self.shape == "BLEND" else [])),
                ("Side", [hints.key("A", "other side of A"), hints.key("B", "other side of B"),
                          hints.toggle("T", "trim the surfaces", self.trim_bases)])]
        return rows

    def drag_state(self):
        a, b = bpy.data.objects.get(self.surf_a), bpy.data.objects.get(self.surf_b)
        return [self.shape.title(), f"A {a.name if a else '?'}, B {b.name if b else '?'}",
                "the preview is approximate: the kernel makes the real fillet"]

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
                        xsect=0 if self.shape == "CHAMFER" else 2, chord=chord, blend=self._used["blend"],
                        tolerance=1e-4, trim_bases=self.trim_bases)

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
        if res.get("note"):
            self.report({'WARNING'}, res["note"])
        if keep_history:
            features.record(fil, "surface_fillet", [(a, "SURF"), (b, "SURF")], dict(u))
        if u["chord"] and res.get("radius_range"):
            lo, mean, hi = res["radius_range"]
            spread = (f"{lo:.4g}..{hi:.4g}" if hi - lo > 1e-3 * mean else f"{mean:.4g}")
            self.report({'INFO'}, f"Chord {res['chord'][1]:.4g} m (ball radius {spread} m)")
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

def _split_selection(context):
    """(surfaces to split, curve refs) from the selection: every spline of the
    selected curves and lines, and the edges picked in Edge mode (a surface
    with picked edges cuts with them and is not split itself)."""
    from . import subsel
    surfs, refs = [], []
    for o in context.selected_objects:
        if o.type == 'CURVE':
            refs += [[o.name, f"S{i}"] for i in range(len(o.data.splines))]
        elif store.is_nurbs(o):
            edges = subsel.picks(o, "EDGE")
            if edges:
                refs += [[o.name, f"E{i}"] for i in edges]
            else:
                surfs.append(o)
    return surfs, refs


COL_SPLIT_JOINED = (0.25, 0.6, 1.0)     # regions joined into one piece; green: each its own piece


class NPROTO_OT_split(bpy.types.Operator):
    """Split the selected surfaces along each other and/or along the selected
    curves, lines and picked edges (projected onto the surface). A cutter that
    stops short is extended until it cuts across. The regions show first, all
    green: each becomes its own object. Click regions to turn them blue: the
    blue ones are joined into one piece. Enter splits. Nothing is removed. With history the
    split surfaces are kept as hidden inputs and each piece follows them (and
    what cut them)"""
    bl_idname = "nproto.split"
    bl_label = "Split"
    bl_options = {'REGISTER', 'UNDO'}

    surfaces_cut: BoolProperty(name="Surfaces Split Each Other", default=True,
                               description="Each selected surface is split by the other selected surfaces")
    projection: EnumProperty(name="Curve Projection", items=[
        ("CURVE", "Curve Normal", "Each planar curve along its own normal; non-planar curves use the view. "
                                  "A curve lying on the surface cuts along itself"),
        ("VIEW", "View", "Along the view direction"),
        ("SURFACE", "On Surface", "No projection: cut along the curve itself, where it lies on or near the surface"),
        ("X", "World X", ""), ("Y", "World Y", ""), ("Z", "World Z", ""),
    ], default="CURVE")
    reach: EnumProperty(name="Cut", items=[
        ("BOTH", "Both Ways", "A projected curve cuts on both sides of itself"),
        ("FRONT", "Along Normal", "Only where the projection direction points from the curve: for Curve "
                                  "Normal, the curve object's blue Z axis (the sketch normal)"),
        ("BACK", "Against Normal", "Only on the other side of the curve"),
    ], default="BOTH")
    through: BoolProperty(name="Surfaces Through", default=False,
                          description="A cutting surface is extended (its natural continuation) past its open "
                                      "edges until it cuts all the way through the part (a short tube through "
                                      "both walls)")
    extend: BoolProperty(name="Extend Cutters", default=True,
                         description="A curve, line, edge or surface whose cut stops inside the surface is "
                                     "extended (its natural continuation) until it cuts all the way across")
    view_dir: FloatVectorProperty(subtype='XYZ', options={'HIDDEN'})
    targets: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    curves: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    cutter_surfs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})   # surfaces that only cut (picked as cutters)
    split_all: BoolProperty(name="Split All", default=False,
                            description="Every region becomes its own object (the marks are ignored)")
    # regions marked in the preview: {surface name: [[world point, stays?, side], ...]}
    marks: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def _params(self):
        return {"surfaces_cut": bool(self.surfaces_cut), "projection": self.projection,
                "view_dir": [float(x) for x in self.view_dir], "extend": bool(self.extend), "reach": self.reach,
                "through": bool(self.through)}

    def _inputs(self, tgt, surfs, crv_inputs):
        cutters = [bpy.data.objects[n] for n in json.loads(self.cutter_surfs or "[]") if n in bpy.data.objects]
        return [(tgt, "SURF")] + ([(s, "SURF") for s in surfs if s != tgt] if self.surfaces_cut else []) \
            + [(c, "SURF") for c in cutters if c != tgt] + crv_inputs

    def _selection(self):
        surfs = [bpy.data.objects[n] for n in json.loads(self.targets) if n in bpy.data.objects]
        crv_inputs = [(bpy.data.objects[n], r) for n, r in json.loads(self.curves or "[]") if n in bpy.data.objects]
        return surfs, crv_inputs

    def _has_curves(self):
        return bool(json.loads(self.curves or "[]"))

    def _has_surfaces(self):
        return bool(json.loads(self.cutter_surfs or "[]")) or (self.surfaces_cut and len(json.loads(self.targets or "[]")) > 1)

    def _enough(self, surfs, crv_inputs):
        return bool(surfs) and (len(surfs) >= 2 or bool(crv_inputs) or bool(json.loads(self.cutter_surfs or "[]")))

    # ---- preview
    def invoke(self, context, event):
        rv3d = getattr(context, "region_data", None)
        if rv3d is not None:
            self.view_dir = rv3d.view_rotation @ Vector((0, 0, -1))
        surfs, refs = _split_selection(context)
        self.targets = json.dumps([o.name for o in surfs])
        self.curves = json.dumps(refs)
        self.marks = ""
        self.cutter_surfs = ""
        if not _kernel_ok(self):
            return {'CANCELLED'}
        self._regions, self._keep, self._hover = [], [], None
        self._ties, self._cycle = [], 0
        if not self._enough(surfs, refs):        # pick what to split and what cuts it, like Trim
            self._stage = "pick"
            self._targets = list(surfs)
            self._cuts = [(bpy.data.objects[n], r) for n, r in refs]
            draw.pick_overlay["picked"], draw.pick_overlay["colors"] = [], []
            for o in self._targets:
                self._mark_pick(o, None, None, target=True)
            for o, r in self._cuts:
                self._mark_pick(o, r, None)
            self._order = [("target", o, None) for o in self._targets] + [("curve", o, r) for o, r in self._cuts]
            self._refresh_cands(context)
            context.window_manager.modal_handler_add(self)
            self._status(context)
            return {'RUNNING_MODAL'}
        context.window.cursor_set('WAIT')
        shown = self._show_regions(context)
        context.window.cursor_set('DEFAULT')
        if not shown:
            return {'CANCELLED'}
        self._stage = "regions"
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {'RUNNING_MODAL'}

    # ---- pick stage
    def _refresh_cands(self, context):
        self._cands = [c for c in _pick_candidates(context) if c[0] not in self._targets]

    def _mark_pick(self, obj, ref, pts, target=False):
        """Highlight a pick: a target's or cutting surface's edges, or a curve / edge."""
        if pts is None:
            M = obj.matrix_world
            pts = []
            if store.is_nurbs(obj) and ref and ref.startswith("E"):
                pts = [M @ Vector(p) for p in store.display(obj).get("edges", [])[int(ref[1:])]]
            elif store.is_nurbs(obj):
                for e in store.display(obj).get("edges", []):
                    pts += [M @ Vector(p) for p in e]
            elif obj.type == 'CURVE' and ref:
                c = geom.spline_to_kernel_curve(obj.data.splines[int(ref[1:])], geom.world_matrix(obj))
                pts = [Vector(p) for p in geom.sample_curve(c, 48)]
        draw.pick_overlay["picked"].append(pts)
        draw.pick_overlay["colors"].append(1 if target else 0)

    def _picks(self):
        """[(kind, object, ref)] in pick order, matching the overlay."""
        return getattr(self, "_order", [])

    def _modal_pick(self, context, event):
        if event.type == 'MOUSEMOVE':
            if not self._targets:
                return {'RUNNING_MODAL'}
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
            obj, _, _, _ = _ray_nurbs(context, event)
            if not self._targets or (event.shift and obj is not None and obj not in self._targets):
                if obj is None:
                    return {'RUNNING_MODAL'}
                self._targets.append(obj)                   # a surface to split
                self._order = self._picks() + [("target", obj, None)]
                self._mark_pick(obj, None, None, target=True)
                self._refresh_cands(context)
            else:
                ties = _near_candidates(context, event, self._cands)
                if ties:
                    cur = ties[self._cycle % len(ties)] if ties == self._ties else ties[0]
                    self._cuts.append((cur[0], cur[1]))
                    self._order = self._picks() + [("curve", cur[0], cur[1])]
                    self._mark_pick(cur[0], cur[1], cur[2])
                elif obj is not None and obj not in self._targets:
                    self._cuts.append((obj, "SURF"))       # a cutting surface: it cuts, it is not split
                    self._order = self._picks() + [("curve", obj, "SURF")]
                    self._mark_pick(obj, None, None)
            self._status(context)
            context.area.tag_redraw()
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS' and self._picks():
            kind, obj, ref = self._order.pop()
            if kind == "target":
                self._targets.remove(obj)
                self._refresh_cands(context)
            else:
                self._cuts.remove((obj, ref))
            draw.pick_overlay["picked"].pop()
            draw.pick_overlay["colors"].pop()
            self._status(context)
            context.area.tag_redraw()
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            self.targets = json.dumps([o.name for o in self._targets])
            self.curves = json.dumps([[o.name, r] for o, r in self._cuts if r != "SURF"])
            self.cutter_surfs = json.dumps([o.name for o, r in self._cuts if r == "SURF"])
            surfs, crv_inputs = self._selection()
            if not self._enough(surfs, crv_inputs):
                self.report({'WARNING'}, "Split: click a surface to split, then what cuts it")
                return {'RUNNING_MODAL'}
            context.window.cursor_set('WAIT')
            shown = self._show_regions(context)
            context.window.cursor_set('DEFAULT')
            if shown:
                self._stage = "regions"
                draw.pick_overlay["picked"], draw.pick_overlay["colors"] = [], []
                draw.pick_overlay["hover"] = None
            self._status(context)
            context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _show_regions(self, context, prefer=None):
        """Split every target (nothing changes yet) and show its regions. By
        default every region is its own piece (none joined); prefer:
        {(surface, side): joined?} carried over when Cut changes."""
        surfs, crv_inputs = self._selection()
        regions, keep, skipped = [], [], []
        for tgt in surfs:
            inputs = self._inputs(tgt, surfs, crv_inputs)
            if len(inputs) < 2:
                continue
            try:
                info = steps.split_info(inputs, self._params())
            except Exception as exc:
                skipped.append(f"{tgt.name}: {exc}")
                continue
            if len(info["texts"]) < 2:
                skipped.append(f"{tgt.name}: the cutters do not split it into separate regions")
                continue
            M = tgt.matrix_world
            for i, text in enumerate(info["texts"]):
                tris, border = steps.region_mesh(text, M)
                side = tuple(info["sides"][i]) if i < len(info["sides"]) else ()
                regions.append({"target": tgt.name, "point": info["points"][i], "side": side,
                                "tris": tris, "border": border})
                k = False                  # each region its own piece
                if prefer and (tgt.name, side) in prefer and any(v is not None for v in side):
                    k = prefer[(tgt.name, side)]
                keep.append(k)
        for msg in skipped:
            self.report({'WARNING'}, f"Split: {msg}")
        if not regions:
            self.report({'ERROR'}, "Split: nothing was split")
            return False
        self._regions, self._keep = regions, keep
        draw.pick_overlay["region_pal"] = (COL_SPLIT_JOINED, draw.COL_REGION_KEEP)   # joined: blue, own piece: green
        self._hover = None
        self._sync()
        return True

    def _sync(self):
        draw.pick_overlay["regions"] = [(r["tris"], r["border"], k) for r, k in zip(self._regions, self._keep)]
        draw.pick_overlay["region_hover"] = self._hover

    def _status(self, context):
        from . import hints
        if self._stage == "pick":
            if not self._targets:
                state = ["click the surface to split", "or select surfaces and curves, then Split"]
            else:
                state = [", ".join(o.name for o in self._targets) + ": pick cutting curves / edges, or click a "
                         "cutting surface (Shift+click: another surface to split)", f"{len(self._cuts)} picked"]
            rows = [("Pick", [hints.key("Tab", "cycle overlapping", on=True if self._ties else None),
                              hints.key("Backspace", "undo"),
                              hints.key("Enter", "show regions", on=bool(self._targets and self._cuts))])]
            hints.show(context, "Split", state, rows, [("MOUSE_LMB", "Pick"), ("EVENT_ESC", "Cancel")])
            return
        k = sum(self._keep)
        n = len(self._keep) - k
        state = ["green: each its own piece · click regions to join them (blue: one piece together)",
                 f"{n + (1 if k else 0)} pieces" + (f" · {k} joined" if k else "")]
        reach = {"BOTH": "both ways", "FRONT": "along normal", "BACK": "against normal"}[self.reach]
        keys = [hints.key("A", "split all", on=not any(self._keep))]
        if self._has_curves():                # only a projected curve has a direction
            keys.append(hints.key("F", f"cut: {reach}"))
        if self._has_surfaces():
            keys.append(hints.key("E", "surfaces through", on=self.through))
        rows = [("Regions", keys + [hints.key("I", "invert"),
                             hints.key("Enter", "split", on=True)])]
        hints.show(context, "Split", state, rows, [("MOUSE_LMB", "Switch region"), ("EVENT_ESC", "Cancel")])

    def _end(self, context):
        from . import hints
        draw.pick_overlay["regions"] = []
        draw.pick_overlay["region_hover"] = None
        draw.pick_overlay["region_pal"] = None
        draw.pick_overlay["picked"], draw.pick_overlay["colors"] = [], []
        draw.pick_overlay["hover"] = None
        hints.clear(context)

    def modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM',
                          'INBETWEEN_MOUSEMOVE'}:
            return {'PASS_THROUGH'}
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event):
            return {'PASS_THROUGH'}
        from . import hints
        if hints.handle(event):
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._end(context)
            context.area.tag_redraw()
            return {'CANCELLED'}
        if self._stage == "pick":
            return self._modal_pick(context, event)
        if event.type == 'MOUSEMOVE':
            h = region_under(context, event, self._regions)
            if h != self._hover:
                self._hover = h
                self._sync()
                context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            h = region_under(context, event, self._regions)
            if h is not None:
                self._keep[h] = not self._keep[h]
                self._sync()
                self._status(context)
                context.area.tag_redraw()
        elif event.type == 'A' and event.value == 'PRESS':
            if any(self._keep):                     # every region its own piece
                self._saved_keep = list(self._keep)
                self._keep = [False] * len(self._keep)
            else:                                   # again: back to what it was
                self._keep = getattr(self, "_saved_keep", None) or self._keep
            self._sync()
            self._status(context)
            context.area.tag_redraw()
        elif event.type == 'I' and event.value == 'PRESS':
            self._keep = [not k for k in self._keep]
            self._sync()
            self._status(context)
            context.area.tag_redraw()
        elif event.type == 'E' and event.value == 'PRESS' and self._has_surfaces():
            prefer = {}
            for r, k in zip(self._regions, self._keep):
                key = (r["target"], r["side"])
                prefer[key] = prefer.get(key, False) or k
            self.through = not self.through
            context.window.cursor_set('WAIT')
            if not self._show_regions(context, prefer):
                self.through = not self.through
                self._show_regions(context, prefer)
            context.window.cursor_set('DEFAULT')
            self._status(context)
            context.area.tag_redraw()
        elif event.type == 'F' and event.value == 'PRESS' and self._has_curves():
            prefer = {}
            for r, k in zip(self._regions, self._keep):
                key = (r["target"], r["side"])
                prefer[key] = prefer.get(key, False) or k
            order = ["BOTH", "FRONT", "BACK"]
            old = self.reach
            context.window.cursor_set('WAIT')
            for _ in range(2):            # nothing split that way: the next one
                self.reach = order[(order.index(self.reach) + 1) % 3]
                if self._show_regions(context, prefer):
                    break
            else:
                self.reach = old
                self._show_regions(context, prefer)
            context.window.cursor_set('DEFAULT')
            self._status(context)
            context.area.tag_redraw()
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            marks = {}
            for r, k in zip(self._regions, self._keep):
                marks.setdefault(r["target"], []).append([list(r["point"]), bool(k), list(r["side"])])
            self.marks = json.dumps(marks)
            self.split_all = not any(self._keep)
            self._end(context)
            return self.execute(context)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        for name in ("surfaces_cut", "projection", "reach", "through", "extend", "split_all"):
            lay.prop(self, name)
        if self.marks and not self.split_all:
            lay.label(text="Pieces as marked in the viewport")

    # ---- the split
    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not self.targets:  # called from a script: use the selection
            surfs, refs = _split_selection(context)
            self.targets = json.dumps([o.name for o in surfs])
            self.curves = json.dumps(refs)
        surfs, crv_inputs = self._selection()
        if not self._enough(surfs, crv_inputs):
            self.report({'WARNING'}, "Select two or more surfaces, or surfaces and curves / lines "
                                     "(or edges picked in Edge mode)")
            return {'CANCELLED'}
        params = self._params()
        all_marks = json.loads(self.marks) if self.marks and not self.split_all else {}

        # compute every split from the ORIGINAL geometry first, then apply
        results, skipped, notes = {}, [], []
        for tgt in surfs:
            inputs = self._inputs(tgt, surfs, crv_inputs)
            if len(inputs) < 2:
                continue
            try:
                info = steps.split_info(inputs, params, notes)
                if len(info["texts"]) < 2:
                    raise jobs.JobError("the cutters do not split this surface into separate regions")
                stays = self._stays(info, all_marks.get(tgt.name))
                if all(stays):     # every region joined: this surface only cuts the others
                    skipped.append(f"{tgt.name}: every region joined, left as it is")
                    continue
                pieces = []        # (text, [region indices])
                if any(stays):
                    pieces.append((steps.merge_regions(inputs, params, stays), [i for i, k in enumerate(stays) if k]))
                off = sorted((i for i, k in enumerate(stays) if not k), key=lambda i: -info["areas"][i])
                pieces += [(info["texts"][i], [i]) for i in off]          # largest first
                results[tgt.name] = (inputs, info, pieces)
            except jobs.JobError as exc:
                skipped.append(f"{tgt.name}: {exc}")
            except Exception as exc:
                skipped.append(f"{tgt.name}: bad curve ({exc})")
        if not results:
            self.report({'ERROR'}, "Split: nothing was split. " + "; ".join(skipped))
            return {'CANCELLED'}
        made = []
        keep = features.history_enabled(context)
        for name, (inputs, info, pieces) in results.items():
            tgt = bpy.data.objects[name]
            original = tgt.get(store.PROP_UNTRIM, tgt[store.PROP_BREP])
            original_nat = tgt.get(store.PROP_UNTRIM + "_natural", 0) if store.PROP_UNTRIM in tgt \
                else (1 if store.is_natural(tgt) else 0)
            if not keep:                  # no history: the first piece stays in the surface itself
                _frozen(tgt, self)
                store.set_geometry_text(tgt, pieces[0][0], context)
                made.append(tgt)
            centroids = [steps.region_point(t) for t in info["texts"]]
            for i, (text, idx) in enumerate(pieces if keep else pieces[1:], 1):
                o = store.new_object_text(context, f"{name}_split{i}", text, geom.world_matrix(tgt))
                o[store.PROP_UNTRIM] = original
                o[store.PROP_UNTRIM + "_natural"] = original_nat
                if tgt.data.materials:
                    for m in tgt.data.materials:
                        o.data.materials.append(m)
                if keep:              # each piece follows the split surface and what cut it
                    if len(idx) == 1:
                        p = {**params, "at": steps.region_point(text)}
                    else:
                        p = {**params, "keep_at": [centroids[j] for j in idx],
                             "drop_at": [centroids[j] for j in range(len(centroids)) if j not in idx]}
                    features.record(o, "split_region", inputs, p)
                made.append(o)
        if keep:
            features.keep_inputs(context, [bpy.data.objects[n] for n in results])
        for msg in skipped:
            self.report({'WARNING'}, msg)
        for msg in notes:
            self.report({'INFO'}, msg)
        _select_only(context, made)
        self.report({'INFO'}, f"Split into {len(made)} pieces")
        return {'FINISHED'}

    @staticmethod
    def _stays(info, marks):
        """Which regions (split_info order) stay one piece: the marks (each
        region takes the nearest mark of its side), else none -- every region
        its own piece, as before the preview."""
        n = len(info["texts"])
        if not marks:
            return [False] * n
        out = []
        for i in range(n):
            q = Vector(info["points"][i])
            side = list(info["sides"][i]) if i < len(info["sides"]) else []
            cand = [m for m in marks if list(m[2]) == side] or marks
            best = min(cand, key=lambda m: (Vector(m[0]) - q).length)
            out.append(bool(best[1]))
        return out


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


def keep_faces(obj, idx):
    """Brep text of obj's faces idx (indices), the others deleted."""
    _, ext = store._k()
    b = store.get_brep(obj).copy()
    keep = set(int(i) for i in idx)
    ext.delete_faces(b, [f for k, f in enumerate(b.faces()) if k not in keep])
    return store.brep_to_text(b)


def split_faces(context, obj, idx, op):
    """The faces idx of obj become objects of their own (one each); the rest
    stays obj. Returns the new objects."""
    from . import subsel
    n = store.get_brep(obj).face_count()
    idx = sorted(set(int(i) for i in idx if 0 <= int(i) < n))
    rest = [k for k in range(n) if k not in idx]
    if not idx or not rest:
        return []
    M = geom.world_matrix(obj)
    texts = [keep_faces(obj, [i]) for i in idx]
    rest_text = keep_faces(obj, rest)
    _frozen(obj, op)
    store.set_geometry_text(obj, rest_text, context, keep_untrimmed=False)
    made = []
    for i, text in zip(idx, texts):
        o = store.new_object_text(context, f"{obj.name}_face{i}", text, M)
        for m in obj.data.materials:
            o.data.materials.append(m)
        made.append(o)
    subsel.set_picks(obj, "FACE", [])
    return made


class NPROTO_OT_unjoin(bpy.types.Operator):
    """Break the selected multi-face surfaces into one object per face, keeping
    the current shape (fillets and trims included). With faces picked (Face
    mode) only those come off, one object each; the rest stays joined. To go
    back to the surfaces a Join was made from, use Restore Inputs in the
    Feature panel"""
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
        from . import subsel
        picked = {o.name: subsel.picks(o, "FACE") for o in objs}
        if any(picked.values()):
            # Face mode: just the picked faces come off
            made = []
            for obj in objs:
                if picked[obj.name]:
                    made += [obj] + split_faces(context, obj, picked[obj.name], self)
            if not made:
                self.report({'WARNING'}, "Pick some faces, not all of a body's faces")
                return {'CANCELLED'}
            _select_only(context, made)
            self.report({'INFO'}, f"Split off {len(made) - sum(1 for o in made if o.name in picked)} face(s)")
            return {'FINISHED'}
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


class NPROTO_OT_delete_faces(bpy.types.Operator):
    """Delete the picked faces (Face mode) from their bodies; a body whose faces
    are all picked is deleted. X / Delete in Face mode does this"""
    bl_idname = "nproto.delete_faces"
    bl_label = "Delete Faces"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        from . import subsel
        todo = [(o, subsel.picks(o, "FACE")) for o in context.selected_objects if store.is_nurbs(o)]
        todo = [(o, f) for o, f in todo if f]
        if not todo:
            self.report({'WARNING'}, "Pick faces first (Face mode)")
            return {'CANCELLED'}
        gone, faces = [], 0
        for obj, idx in todo:
            n = store.get_brep(obj).face_count()
            faces += len(set(idx))
            if len(set(idx)) >= n:
                gone.append(obj.name)
                bpy.data.objects.remove(obj)
                continue
            text = keep_faces(obj, [k for k in range(n) if k not in set(idx)])
            _frozen(obj, self)
            store.set_geometry_text(obj, text, context, keep_untrimmed=False)
            subsel.set_picks(obj, "FACE", [])
        self.report({'INFO'}, f"Deleted {faces} face(s)" + (f"; removed {', '.join(gone)}" if gone else ""))
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


def _v3(x):
    return np.array([x.x, x.y, x.z] if hasattr(x, "x") else list(x), float)


def fillet_frames(obj, idx, n=15):
    """Where a fillet of shared edge idx of obj sits, per sample along it
    (world): (P (n,3) the edge's points, D1, D2 (n,3) unit directions into its
    two faces, across the edge, in each face's tangent plane). The rolling
    ball of radius r sits in the wedge between D1 and D2 (on the material side
    of a convex edge, outside a concave one): rails at P + r / tan(a/2) D,
    a = the angle between D1 and D2. Sampled once; any radius is then a
    formula (the Edge Fillet preview)."""
    from .kernel import load
    sm = load()
    b = store.get_brep(obj)
    e = b.edges()[idx]
    fs = e.faces()
    if len(fs) != 2:
        return None
    M = geom.world_matrix(obj)
    M3 = np.array(M.to_3x3(), float)
    Mt = np.array(M.translation, float)
    t0, t1 = e.parameter_range()
    ts = np.linspace(t0, t1, n)
    h = 1e-4 * (t1 - t0)
    pts = np.array([_v3(e.point_at(t)) for t in np.linspace(t0, t1, 33)])
    eps = 1e-3 * float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))     # a step into a face
    Ps, D1, D2 = [], [], []
    for t in ts:
        P = _v3(e.point_at(t))
        a, c = _v3(e.point_at(max(t0, t - h))), _v3(e.point_at(min(t1, t + h)))
        T = c - a
        T /= np.linalg.norm(T) or 1.0
        ds = []
        for f in fs:
            surf = f.surface()
            _, uvs, _ = sm.surface_closest_point(surf, tuple(P))
            uv = uvs[0]
            u, v = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
            nrm = np.array(f.outward_normal_at_uv(u, v), float)
            d = np.cross(nrm, T)
            d /= np.linalg.norm(d) or 1.0
            # into the face: a small step that way in its (u, v) must stay inside its trims
            Dv = surf.derivatives(u, v)
            J = np.column_stack([np.array(Dv[1], float), np.array(Dv[2], float)])
            step = np.linalg.lstsq(J, d * eps, rcond=None)[0]
            (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()

            def out(uu, vv):        # past the surface's own range (an untrimmed side), or outside its trims
                tu, tv = 1e-9 * (u1 - u0), 1e-9 * (v1 - v0)
                if not (u0 - tu <= uu <= u1 + tu and v0 - tv <= vv <= v1 + tv):
                    return True
                return f.classify_uv(uu, vv) == "outside"
            if out(u + step[0], v + step[1]) and not out(u - step[0], v - step[1]):
                d = -d
            ds.append(d)
        Ps.append(P)
        D1.append(ds[0])
        D2.append(ds[1])
    P = np.array(Ps) @ M3.T + Mt
    to_w = lambda D: (lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-300))(np.array(D) @ M3.T)
    return P, to_w(D1), to_w(D2)


def fillet_sections(P, D1, D2, r, shape="CIRCULAR", k=9, fullness=1.0):
    """Cross-sections of a fillet of radius r (per sample, scalar or (n,))
    over the frames of fillet_frames: (n, k, 3), rail on face 1 first.
    CIRCULAR: the rolling ball's arc; CHAMFER: the straight line between the
    rails; BLEND: a curve tangent to both faces (fuller with fullness > 1)."""
    r = np.broadcast_to(np.asarray(r, float), (len(P),))
    cosa = np.clip(np.sum(D1 * D2, axis=1), -0.9999, 0.9999)
    half = 0.5 * np.arccos(cosa)                     # half the wedge's angle
    s = r / np.tan(half)                             # edge -> rail
    T1, T2 = P + s[:, None] * D1, P + s[:, None] * D2
    w = np.linspace(0.0, 1.0, k)[None, :, None]
    if shape == "CHAMFER":
        return T1[:, None] * (1 - w) + T2[:, None] * w
    if shape == "BLEND":
        g = np.clip(0.55 * fullness, 0.05, 1.0)
        A, B = T1 + g * (P - T1), T2 + g * (P - T2)
        return ((1 - w) ** 3 * T1[:, None] + 3 * (1 - w) ** 2 * w * A[:, None]
                + 3 * (1 - w) * w ** 2 * B[:, None] + w ** 3 * T2[:, None])
    bis = D1 + D2
    bis /= np.linalg.norm(bis, axis=1, keepdims=True) + 1e-300
    C = P + (r / np.sin(half))[:, None] * bis
    a, b = (T1 - C) / r[:, None], (T2 - C) / r[:, None]
    om = np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0))
    so = np.sin(om)
    so = np.where(so < 1e-9, 1.0, so)
    wa = np.sin((1 - w[..., 0]) * om[:, None]) / so[:, None]
    wb = np.sin(w[..., 0] * om[:, None]) / so[:, None]
    return C[:, None] + r[:, None, None] * (wa[..., None] * a[:, None] + wb[..., None] * b[:, None])


def surface_fillet_frames(a, b, n=15):
    """Where a fillet between surfaces a and b sits: per intersection curve,
    (P (n,3) points on it, NA, NB (n,3) the surfaces' outward normals there),
    world. None when they do not cross. With the ball on the normals' sides
    (Flip A / Flip B off) its centre C has n_A . (C - P) = r_A, n_B . (C - P)
    = r_B: any radius is then a formula (the Surface Fillet preview)."""
    from .kernel import load
    from .sweeps import _world_brep
    sm = load()
    wa, wb = _world_brep(a), _world_brep(b)
    try:
        curves, _ = sm.intersect_breps(wa, wb)
    except RuntimeError:            # the kernel raises when they do not meet
        return None
    out = []
    for c in curves or []:
        t0, t1 = c.parameter_range()
        P = np.array([_v3(c.evaluate(t)) for t in np.linspace(t0, t1, n)])
        if float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) < 1e-9:
            continue
        NA, NB = [], []
        for p in P:
            for br, acc in ((wa, NA), (wb, NB)):
                f = min(br.faces(), key=lambda f: f.closest_point(tuple(p))[1])
                _, uvs, _ = sm.surface_closest_point(f.surface(), tuple(p))
                uv = uvs[0]
                u, v = (uv.x, uv.y) if hasattr(uv, "x") else (uv[0], uv[1])
                nn = np.array(f.outward_normal_at_uv(u, v), float)
                acc.append(nn / (np.linalg.norm(nn) or 1.0))
        out.append((P, np.array(NA), np.array(NB)))
    return out or None


SIDES = ((False, False), (False, True), (True, False), (True, True))    # (flip A, flip B), the kernel's order


def surface_fillet_sides(a, b, frames, size):
    """Which sides (flip A, flip B) have room for a fillet: {sides: share of
    the samples where it fits}. The rails leave the crossing along the
    surfaces (ca - A + cb on A, ca + cb - B on B); a short step that way must
    land on the surface itself, not on its extension -- two surfaces meeting
    at their edges (a tent) have one side, two crossing through each other
    four."""
    from .sweeps import _world_brep
    wa, wb = _world_brep(a), _world_brep(b)
    h = 0.02 * (size or 1.0)

    def dist(br, p):
        return min(f.closest_point(tuple(p))[1] for f in br.faces())

    out = {}
    for fa, fb in SIDES:
        ok = n = 0
        for P, NA, NB in frames:
            idx = sorted({len(P) // 4, len(P) // 2, (3 * len(P)) // 4})
            ca, cb, A, B = surface_fillet_rails(P[idx], NA[idx], NB[idx], -1.0 if fa else 1.0, -1.0 if fb else 1.0)
            D1, D2 = ca - A + cb, ca + cb - B
            for p, d1, d2 in zip(P[idx], D1, D2):
                d1 = d1 / (np.linalg.norm(d1) or 1.0)
                d2 = d2 / (np.linalg.norm(d2) or 1.0)
                n += 1
                ok += dist(wa, p + h * d1) < 0.25 * h and dist(wb, p + h * d2) < 0.25 * h
        out[(fa, fb)] = ok / max(n, 1)
    return out


def surface_fillet_rails(P, NA, NB, sa, sb):
    """Per sample (n,3) each: (ca, cb, A, B) with the ball centre at
    C = P + r_A ca + r_B cb and its rails at C - r_A A (on surface A) and
    C - r_B B (on B); A, B are the normals turned to the ball's side. sa, sb:
    +1 for the ball on that surface's normal side, -1 for the other (Flip)."""
    A, B = sa * NA, sb * NB
    g = np.sum(A * B, axis=1)
    det = np.clip(1.0 - g * g, 1e-6, None)
    # least-norm x with A.x = ra, B.x = rb: x = alpha A + beta B
    ca = (A - g[:, None] * B) / det[:, None]          # x for (ra, rb) = (1, 0)
    cb = (B - g[:, None] * A) / det[:, None]          # ... (0, 1)
    return ca, cb, A, B


def surface_fillet_sections(P, NA, NB, ra, rb, sa, sb, shape="CIRCULAR", k=9, continuity=2):
    """Cross-sections (n, k, 3) of a fillet between two crossing surfaces,
    rail on A first: the ball (radius ra from A, rb from B; an equal pair is
    the rolling ball's arc), a straight chamfer, or a blend tangent to both
    (the worker's G1 cubic / G2 quintic / G3 degree 7, secfillet.py, with the
    surfaces' curvature left out)."""
    n = len(P)
    ra = np.broadcast_to(np.asarray(ra, float), (n,))
    rb = np.broadcast_to(np.asarray(rb, float), (n,))
    ca, cb, A, B = surface_fillet_rails(P, NA, NB, sa, sb)
    C = P + ra[:, None] * ca + rb[:, None] * cb
    T1, T2 = C - ra[:, None] * A, C - rb[:, None] * B
    w = np.linspace(0.0, 1.0, k)[None, :, None]
    if shape == "CHAMFER":
        return T1[:, None] * (1 - w) + T2[:, None] * w
    if shape == "BLEND":
        D = T2 - T1
        d = np.linalg.norm(D, axis=1, keepdims=True)
        ta = D - np.sum(D * A, axis=1, keepdims=True) * A
        tb = -D - np.sum(-D * B, axis=1, keepdims=True) * B
        ta /= np.linalg.norm(ta, axis=1, keepdims=True) + 1e-300
        tb /= np.linalg.norm(tb, axis=1, keepdims=True) + 1e-300
        deg = {1: 3, 3: 7}.get(int(continuity), 5)
        h = d / deg
        k = (deg + 1) // 2                          # CVs from each side: on along its tangent
        cv = [T1 + i * h * ta for i in range(k)] + [T2 + i * h * tb for i in range(k)][::-1]
        from math import comb
        return sum(comb(deg, i) * w ** i * (1 - w) ** (deg - i) * c[:, None] for i, c in enumerate(cv))
    a, b = -A, -B                                    # centre -> rails, unit
    om = np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0))
    so = np.where(np.sin(om) < 1e-9, 1.0, np.sin(om))
    wa = np.sin((1 - w[..., 0]) * om[:, None]) / so[:, None]
    wb = np.sin(w[..., 0] * om[:, None]) / so[:, None]
    rr = ra[:, None] + (rb - ra)[:, None] * w[..., 0]          # equal radii: the ball's arc
    return C[:, None] + rr[..., None] * (wa[..., None] * a[:, None] + wb[..., None] * b[:, None])


class NPROTO_OT_edge_fillet(ValueDrag, AskValue, PickMixin, bpy.types.Operator):
    """Fillet, chamfer or blend shared edges of joined surfaces and solids.
    Click the edges to fillet (click again to un-pick, A = all), Enter to finish.
    Alt+click the button: all shared edges of the selected objects at once.
    Then the cross-sections are previewed: drag the radius handle, type a value,
    click away or Enter to make it"""
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
        return self.start_picking(context, event)

    def prepare_value(self, context):
        try:
            names = {n for n, _ in json.loads(self.inputs or "[]")}
        except ValueError:
            names = set()
        objs = [bpy.data.objects[n] for n in names if n in bpy.data.objects]
        if objs:
            fit_radius(self, objs, ["radius", "end_radius"])     # a size that fits, shown in the dialog

    # -- the live value stage (valuedrag.py) ----------------------------------
    angle_fields = ()
    fit_view = "shrink"         # the radius fit_radius found for the part: only ever made smaller
    start_value = 0.0           # the radius handle on the edges: no fillet until it is dragged (L: the last radius)
    SHAPES = ("CIRCULAR", "CHAMFER", "BLEND")

    @property
    def drag_fields(self):
        return ("radius", "end_radius") if self.variable and self.rail_type == "RADIUS" else ("radius",)

    def drag_setup(self, context, mouse):
        self._ef = []
        for name, ref in json.loads(self.inputs or "[]"):
            obj = bpy.data.objects.get(name)
            if obj is None or not ref.startswith("E"):
                continue
            fr = fillet_frames(obj, int(ref[1:]))
            if fr is not None:
                self._ef.append(fr)
        if not self._ef:
            return False
        n_all = sum(len(P) for P, _, _ in self._ef)
        self._frac = []                 # each sample's place along the picks, 0..1 (a variable radius)
        k = 0
        for P, _, _ in self._ef:
            self._frac.append(np.arange(k, k + len(P)) / max(n_all - 1, 1))
            k += len(P)
        return True

    def _r_of(self, value, cosa):
        """Ball radius for a field value at a wedge of cos(angle) cosa."""
        if self.rail_type == "CHORD":
            th = np.pi - np.arccos(np.clip(cosa, -0.9999, 0.9999))       # angle between the normals
            return value / (2.0 * np.sin(0.5 * max(th, np.radians(1.0))))
        return value

    def _radii(self, i):
        f = self._frac[i]
        P, D1, D2 = self._ef[i]
        cosa = np.sum(D1 * D2, axis=1)
        if self.variable and self.rail_type == "RADIUS":
            vals = self.radius + (self.end_radius - self.radius) * f
        else:
            vals = np.full(len(P), self.radius)
        return np.array([self._r_of(v, c) for v, c in zip(vals, cosa)])

    def drag_preview(self):
        lines, tris = [], []
        for i in range(len(self._ef)):
            P, D1, D2 = self._ef[i]
            X = fillet_sections(P, D1, D2, self._radii(i), self.shape, fullness=self.thumbweight)
            L, T = ruled_preview(list(X), ribs=2)
            lines += L
            tris.append(T)
        return {"lines": lines, "tris": np.vstack(tris)}

    def _handle(self, field, i, j):
        P, D1, D2 = self._ef[i]
        cosa = float(D1[j] @ D2[j])
        tan_half = np.tan(0.5 * np.arccos(np.clip(cosa, -0.9999, 0.9999)))
        value = getattr(self, field)
        if self.rail_type == "CHORD":     # chord -> rail distance and back
            th = np.pi - np.arccos(np.clip(cosa, -0.9999, 0.9999))
            k = 2.0 * np.sin(0.5 * max(th, np.radians(1.0)))
        else:
            k = 1.0
        return {"field": field, "kind": "arrow", "origin": P[j], "dir": D1[j], "base": P[j],
                "param": value / k / tan_half, "to_value": lambda t: t * tan_half * k}

    def drag_handles(self):
        P0 = self._ef[0][0]
        hs = [self._handle("radius", 0, len(P0) // 2 if len(self._ef) == 1 or not self.variable else 1)]
        if "end_radius" in self.drag_fields:
            Pl = self._ef[-1][0]
            hs.append(self._handle("end_radius", len(self._ef) - 1, len(Pl) - 2))
        return hs

    def drag_key(self, context, event, ch):
        if ch == "c":
            self.shape = self.SHAPES[(self.SHAPES.index(self.shape) + 1) % len(self.SHAPES)]
        elif ch == "v":
            self.variable = not self.variable
            self._field = 0
        elif ch == "r":
            self.rail_type = "CHORD" if self.rail_type == "RADIUS" else "RADIUS"
            self._field = 0
        else:
            return False
        return True

    def drag_handle_hints(self):
        hs = [hints.key("LMB drag", "arrow: radius" if self.rail_type == "RADIUS" else "arrow: chord")]
        if "end_radius" in self.drag_fields:
            hs.append(hints.key("LMB drag", "2nd arrow: end radius"))
        return hs

    def drag_rows(self):
        return [("Shape", [hints.cycle("C", ["circular", "chamfer", "blend"], self.shape.lower()),
                           hints.cycle("R", ["rolling ball", "chord"],
                                       "rolling ball" if self.rail_type == "RADIUS" else "chord", "size"),
                           hints.toggle("V", "variable", self.variable)])]

    def drag_state(self):
        n = len(self._ef)
        return [self.shape.title(), f"{n} edge" + ("s" if n != 1 else ""),
                "the preview is approximate: the kernel makes the real fillet"]

    def modal(self, context, event):
        if getattr(self, "_dragging", False):
            return super().modal(context, event)      # the value stage (valuedrag.py)
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
                steps.store_rims(obj, [list(Mi @ Vector(m)) for m in mids], info, Mi, breps["result"])
                features.push_step(obj, "edge_fillet", [], {
                    "mids_local": [list(Mi @ Vector(m)) for m in mids], "shape": self.shape, "radius": self.radius,
                    "end_radius": self.end_radius if self.variable and not chord else None, "chord": chord,
                    "continuity": int(self.continuity), "thumbweight": self.thumbweight,
                    **self._ca_params()}, base_text=pre)
            done.append(obj)
            from . import subsel
            subsel.set_picks(obj, "EDGE", [])     # the picked edges are gone: no pick moves onto a rail
            msg = f"{name}: {info['edges']} edge(s) filleted" + (" (solid)" if info.get("solid") else "")
            if info.get("chord"):
                lo, mean, hi = info["chord"]
                rr = info.get("radius_range") or (info["radius"],) * 3
                ball = (f"{rr[0]:.4g}..{rr[2]:.4g}" if rr[2] - rr[0] > 1e-3 * rr[1] else f"{rr[1]:.4g}")
                msg += (f" · chord {lo:.4g}..{hi:.4g} m" if hi - lo > 1e-3 * mean else f" · chord {mean:.4g} m") \
                    + f" (ball radius {ball} m)"
            level = 'INFO'
            if info.get("chord_note"):
                msg += " · " + info["chord_note"]
                level = 'WARNING'
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


classes = (NPROTO_OT_trim, NPROTO_OT_untrim, NPROTO_OT_untrim_region, NPROTO_OT_fillet, NPROTO_OT_split, NPROTO_OT_join, NPROTO_OT_unjoin,
           NPROTO_OT_delete_faces, NPROTO_OT_restore_inputs,
           NPROTO_OT_edge_fillet, NPROTO_OT_boolean, NPROTO_OT_flip_normal, NPROTO_OT_boundary_fill)
