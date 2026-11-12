"""Modelling operators (viewport tools)."""

import json

import bpy
import numpy as np
from bpy.props import FloatVectorProperty, BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras import view3d_utils
from mathutils import Vector

from . import draw, features, geom, store
from .kernel import load as load_kernel

CONT_ITEMS = [
    ("G0", "G0 Position", "Edges touch only"),
    ("G1", "G1 Tangent", "Tangent continuous with the source surface"),
    ("G2", "G2 Curvature", "Curvature continuous with the source surface"),
]
AXIS_ITEMS = [
    ("X", "World X", ""), ("Y", "World Y", ""), ("Z", "World Z", ""),
    ("LOCAL_X", "Local X", ""), ("LOCAL_Y", "Local Y", ""), ("LOCAL_Z", "Local Z", ""),
]


def _kernel_ok(op):
    if load_kernel() is None:
        from . import kernel
        op.report({'ERROR'}, kernel.error or "USD BRep kernel not available")
        return False
    return True


def _make_result(context, name, op, inputs, params, matrix=None, keep_history=True):
    g = features.run(op, inputs, params)
    if isinstance(g, tuple) and g and g[0] == "brep_text":     # built in the worker
        obj = store.new_object_text(context, name, g[1], matrix)
    else:
        obj = store.new_object(context, name, g, matrix)
    if keep_history and context.scene.nproto_settings.construction_history:
        features.record(obj, op, inputs, params)
    return obj


def _select_only(context, objs):
    for o in context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    if objs:
        context.view_layer.objects.active = objs[-1]


def _encode_inputs(inputs):
    return json.dumps([[o.name, r] for o, r in inputs])


def _decode_inputs(text):
    out = []
    for name, ref in json.loads(text or "[]"):
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise ValueError(f"Input object '{name}' not found")
        out.append((obj, ref))
    return out


# --------------------------------------------------------------------------
# Pick candidates (curve splines and NURBS surface boundaries)
# --------------------------------------------------------------------------

def _pick_candidates(context, allow_edges=True):
    cands = []
    for obj in context.visible_objects:
        try:
            if obj.type == 'CURVE':
                for i, sp in enumerate(obj.data.splines):
                    c = geom.spline_to_kernel_curve(sp, geom.world_matrix(obj))
                    pts = [Vector(p) for p in geom.sample_curve(c, 48)]
                    cands.append((obj, f"S{i}", pts))
            elif allow_edges and store.is_nurbs(obj) and not store.is_natural(obj):
                # trimmed / multi-face: the real B-rep edges (G0 inputs)
                M = geom.world_matrix(obj)
                for i, poly in enumerate(store.display(obj).get("edges", [])):
                    pts = [M @ Vector(p) for p in poly]
                    if len(pts) > 1 and sum((pts[k + 1] - pts[k]).length for k in range(len(pts) - 1)) > 1e-6:
                        cands.append((obj, f"E{i}", pts))
            elif allow_edges and store.is_nurbs(obj):
                surf = store.get_surfaces(obj)[0]
                M = geom.world_matrix(obj)
                for which in ("U0", "U1", "V0", "V1"):
                    pts = [M @ Vector(p) for p in geom.boundary_points(surf, which, 48)]
                    if (pts[0] - pts[len(pts) // 2]).length + (pts[-1] - pts[len(pts) // 2]).length < 1e-6:
                        continue  # degenerate edge (pole of a revolve, cone tip...)
                    cands.append((obj, which, pts))
        except Exception as exc:  # unusable input, skip it
            print("NURBS pick: skipping", obj.name, exc)
    return cands


def _object_under_mouse(context, mouse):
    region, rv3d = context.region, context.region_data
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    try:
        hit, _, _, _, obj, _ = context.scene.ray_cast(context.evaluated_depsgraph_get(), origin, direction)
    except Exception:
        return None
    return obj.original if hit and obj is not None else None


def _near_candidates(context, event, cands, max_px=18.0, tie_px=4.0):
    """All candidates within tie_px of the closest one (closest first), with
    the object under the cursor moved to the front. Coincident edges (a blend
    edge on its source edge, a surface edge on its input curve) are ties."""
    region, rv3d = context.region, context.region_data
    mouse = Vector((event.mouse_region_x, event.mouse_region_y))
    scored = []
    for c in cands:
        best = None
        prev = None
        for p in c[2]:
            q = view3d_utils.location_3d_to_region_2d(region, rv3d, p)
            if q is not None and prev is not None:
                seg = q - prev
                L = seg.length_squared
                t = 0.0 if L == 0 else max(0.0, min(1.0, (mouse - prev).dot(seg) / L))
                d = (prev + seg * t - mouse).length
                if best is None or d < best:
                    best = d
            prev = q
        if best is not None and best <= max_px:
            scored.append((best, c))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0])
    ties = [c for d, c in scored if d <= scored[0][0] + tie_px]
    under = _object_under_mouse(context, mouse)
    if under is not None:
        ties.sort(key=lambda c: c[0] != under)  # stable: object under cursor first
    return ties


def _nearest(context, event, cands, max_px=18.0):
    ties = _near_candidates(context, event, cands, max_px)
    return ties[0] if ties else None


class AskValue:
    """Commands with a size (distance, radius, thickness): started from the UI
    they first ask for it in a small dialog -- prefilled with the last value
    used (prepare_value may size it to the part) -- and compute on OK. The
    redo panel still adjusts it afterwards. Scripts (EXEC) never ask."""

    def prepare_value(self, context):
        pass

    def ask(self, context):
        self.prepare_value(context)
        if bpy.app.background:
            return self.execute(context)
        return context.window_manager.invoke_props_dialog(self, width=320, title=self.bl_label,
                                                          confirm_text=self.bl_label)


class PickMixin:
    """Modal picking of curves / surface edges. Subclasses set:
    pick_count (min), pick_max, allow_edges, and implement finish_picks()."""

    pick_min = 2
    pick_max = 2
    allow_edges = True
    finish_on_min = True   # finish as soon as pick_min inputs are picked (unless Shift)

    def expand_pick(self, cand):
        """Inputs added by one click (Planar chains whole edge loops)."""
        return [cand]

    def pick_candidates(self, context):
        return _pick_candidates(context, self.allow_edges)

    def closes_loop(self):
        return False

    def _status(self, context):
        n = len(self._picks)
        msg = f"{self.bl_label}: pick input {n + 1}" + (
            f" (Enter to finish)" if n >= self.pick_min else "")
        ties = getattr(self, "_ties", [])
        if len(ties) > 1:
            cur = ties[self._cycle % len(ties)]
            msg += f"   [{cur[0].name} {cur[1]} · {len(ties)} overlapping, Tab to cycle]"
        context.area.header_text_set(msg + "   LMB pick · Shift+LMB keep picking · Backspace undo · Esc cancel"
                                     + getattr(self, "extra_help", ""))

    def _update_hover(self, context, event):
        ties = _near_candidates(context, event, self._cands)
        key = [(c[0].name, c[1]) for c in ties]
        if key != getattr(self, "_tie_key", None):
            self._tie_key, self._ties, self._cycle = key, ties, 0
        cur = self._ties[self._cycle % len(self._ties)] if self._ties else None
        draw.pick_overlay["hover"] = cur[2] if cur else None
        self._status(context)
        context.area.tag_redraw()
        return cur

    def start_picking(self, context):
        self._cands = self.pick_candidates(context)
        if not self._cands:
            self.report({'WARNING'}, "Nothing to pick: add curves or NURBS surfaces first")
            return {'CANCELLED'}
        self._picks = []
        self._groups = []
        self._ties, self._tie_key, self._cycle = [], None, 0
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["hover"] = None
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {'RUNNING_MODAL'}

    def _end(self, context):
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["hover"] = None
        context.area.header_text_set(None)
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE'} and not self._in_view(context, event):
            # a click on the sidebar / header (e.g. the CVs / Hulls toggle): let the UI have it
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            self._update_hover(context, event)
        elif event.type == 'TAB' and event.value == 'PRESS':
            if getattr(self, "_ties", None):
                self._cycle += 1
                cur = self._ties[self._cycle % len(self._ties)]
                draw.pick_overlay["hover"] = cur[2]
                self._status(context)
                context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = self._update_hover(context, event)
            self._last_mouse = Vector((event.mouse_region_x, event.mouse_region_y))
            if hit:
                self._tie_key = None
                have = {(o.name, r) for o, r in self._picks}
                if getattr(self, "toggle_picks", False) and (hit[0].name, hit[1]) in have:
                    # click a picked edge again: un-pick it
                    k = [(o.name, r) for o, r in self._picks].index((hit[0].name, hit[1]))
                    self._picks.pop(k)
                    draw.pick_overlay["picked"].pop(k)
                    self._groups = [1] * len(self._picks)
                    self._status(context)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                group = [c for c in self.expand_pick(hit) if (c[0].name, c[1]) not in have]
                for c in group:
                    self._picks.append((c[0], c[1]))
                    draw.pick_overlay["picked"].append(c[2])
                if group:
                    self._groups.append(len(group))
                context.area.tag_redraw()
                if self.closes_loop() and not (event.shift and getattr(self, "shift_extends", False)):
                    return self._finish(context)
                if len(self._picks) >= self.pick_max or (
                        self.finish_on_min and len(self._picks) >= self.pick_min and not event.shift):
                    return self._finish(context)
                self._status(context)
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS' and self._picks:
            for _ in range(self._groups.pop() if self._groups else 1):
                self._picks.pop()
                draw.pick_overlay["picked"].pop()
            self._status(context)
            context.area.tag_redraw()
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if len(self._picks) >= self.pick_min:
                return self._finish(context)
        elif event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    @staticmethod
    def _in_view(context, event):
        """Is the mouse over the 3D view's drawing area (not a sidebar / header
        overlapping it)?"""
        area = context.area
        if area is None:
            return True
        x, y = event.mouse_x, event.mouse_y
        win = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if win is None or not (win.x <= x < win.x + win.width and win.y <= y < win.y + win.height):
            return False
        for r in area.regions:   # overlapping regions (region overlap draws the sidebar over the view)
            if r.type in {'UI', 'HEADER', 'TOOL_HEADER', 'TOOLS', 'HUD', 'ASSET_SHELF'} and r.width > 1 and r.height > 1:
                if r.x <= x < r.x + r.width and r.y <= y < r.y + r.height:
                    return False
        return True

    def _finish(self, context):
        self._end(context)
        self.inputs = _encode_inputs(self._picks)
        if isinstance(self, AskValue) and not bpy.app.background:
            # a dialog cannot open from this pick loop (the loop would keep
            # running behind it): start the command again with the picks
            # given -- its invoke asks for the value
            # a running operator reports "NPROTO_OT_edge_fillet", its class "nproto.edge_fillet"
            idn = self.bl_idname
            mod, name = idn.split("_OT_", 1) if "_OT_" in idn else idn.split(".", 1)
            getattr(getattr(bpy.ops, mod.lower()), name)('INVOKE_DEFAULT', inputs=self.inputs)
            return {'CANCELLED'}
        return self.execute(context)


# --------------------------------------------------------------------------
# Loft
# --------------------------------------------------------------------------

def loft_selection_order(sel, active=None, closed=False):
    """The order to loft selected curves in. A selection has none (Blender's
    list is not the order they were clicked), so three or more open-loft
    sections go along the chain through them; two keep the active one last.
    Picking them one by one keeps your order."""
    sel = sorted(sel, key=lambda o: o == active)  # active last
    if len(sel) >= 3 and not closed:
        try:
            order = geom.chain_order([features.input_curve(o, "S0") for o in sel])
            sel = [sel[i] for i in order]
        except Exception:
            pass
    return sel


class NPROTO_OT_loft(PickMixin, bpy.types.Operator):
    """Loft a surface between curves or surface edges (G0 / G1 / G2)"""
    bl_idname = "nproto.loft"
    bl_label = "Loft"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 2
    pick_max = 32

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    use_selection: BoolProperty(name="Use Selection", default=False, options={'SKIP_SAVE'},
                                description="Loft the selected curves instead of picking")
    cont_a: EnumProperty(name="Continuity A", items=CONT_ITEMS, default="G0")
    cont_b: EnumProperty(name="Continuity B", items=CONT_ITEMS, default="G0")
    tan_a: FloatProperty(name="Tangent A", default=1.0, min=0.0, soft_max=3.0,
                         description="G1 magnitude at input A (relative to loft span)")
    tan_b: FloatProperty(name="Tangent B", default=1.0, min=0.0, soft_max=3.0)
    cur_a: FloatProperty(name="Curvature A", default=0.0, soft_min=-2.0, soft_max=2.0,
                         description="G2 shape at input A (moves the 3rd CV row along the tangent)")
    cur_b: FloatProperty(name="Curvature B", default=0.0, soft_min=-2.0, soft_max=2.0)
    auto_align: BoolProperty(name="Auto Align", default=True,
                             description="Flip inputs, and start closed sections in the same direction, so the loft does not twist")
    flip_b: BoolProperty(name="Flip B", default=False)
    ruled: BoolProperty(name="Ruled", default=False, description="Straight (degree 1) loft between 2 curves")
    closed: BoolProperty(name="Closed", default=False,
                         description="Loft through the sections and back to the first: a closed ring, smooth all "
                                     "round, the seam included (3 or more sections)")
    samples: IntProperty(name="Samples", default=24, min=6, max=200,
                         description="Samples along the edges for G1/G2 lofts")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        sel = [o for o in context.selected_objects if o.type == 'CURVE']
        if self.use_selection or (len(sel) >= 2 and not event.alt):
            sel = loft_selection_order(sel, context.active_object, self.closed)
            self.inputs = _encode_inputs([(o, "S0") for o in sel])
            return self.execute(context)
        return self.start_picking(context)

    def _params(self):
        return {k: getattr(self, k) for k in (
            "cont_a", "cont_b", "tan_a", "tan_b", "cur_a", "cur_b",
            "auto_align", "flip_b", "ruled", "closed", "samples")}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            inputs = _decode_inputs(self.inputs)
            obj = _make_result(context, "Loft", "loft", inputs, self._params())
        except Exception as exc:
            self.report({'ERROR'}, f"Loft failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        return {'FINISHED'}

    def draw(self, context):
        lay = self.layout
        try:
            refs = [r for _, r in json.loads(self.inputs or "[]")]
        except ValueError:
            refs = []
        two = len(refs) == 2
        for side, ref in zip("ab", refs[:2] if two else []):
            box = lay.box()
            edge = features.is_surface_edge(ref)
            box.label(text=f"Input {side.upper()}: " + (("trimmed edge" if features.is_brep_edge_ref(ref) else
                                                          "surface edge") if edge else "curve"))
            col = box.column(align=True)
            col.enabled = edge
            col.prop(self, f"cont_{side}", text="")
            row = col.row(align=True)
            row.prop(self, f"tan_{side}", text="Tangent")
            row.prop(self, f"cur_{side}", text="Curvature")
        col = lay.column()
        row = col.row()
        row.active = len(refs) >= 3
        row.prop(self, "closed")
        col.prop(self, "auto_align")
        col.prop(self, "flip_b")
        if two and not any(features.is_surface_edge(r) for r in refs):
            col.prop(self, "ruled")
        col.prop(self, "samples")


# --------------------------------------------------------------------------
# Single-curve tools
# --------------------------------------------------------------------------

def _curve_inputs(context):
    out = []
    for o in context.selected_objects:
        if o.type == 'CURVE':
            out += [(o, f"S{i}") for i in range(len(o.data.splines))]
    return out


class _CurveTool(PickMixin):
    pick_min = 1
    pick_max = 1
    allow_edges = True

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        go = self.ask if isinstance(self, AskValue) else self.execute
        if self.inputs:                     # picked already (see PickMixin._finish)
            return go(context)
        ins = _curve_inputs(context)
        if ins and not event.alt:
            self.inputs = _encode_inputs(ins)
            return go(context)
        return self.start_picking(context)

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        made = []
        try:
            for inp in _decode_inputs(self.inputs):
                made.append(_make_result(context, self.bl_label, self.op_name, [inp], self.params()))
        except Exception as exc:
            self.report({'ERROR'}, f"{self.bl_label} failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, made)
        return {'FINISHED'}


class NPROTO_OT_extrude(AskValue, _CurveTool, bpy.types.Operator):
    """Extrude curves (or surface edges) into surfaces"""
    bl_idname = "nproto.extrude"
    bl_label = "Extrude"
    bl_options = {'REGISTER', 'UNDO'}
    op_name = "extrude"

    distance: FloatProperty(name="Distance", default=1.0, unit='LENGTH',
                            description="Height along the direction (with draft: the far edge also moves outward)")
    direction: EnumProperty(name="Direction", items=[("NORMAL", "Curve Normal", "")] + AXIS_ITEMS +
                            [("VECTOR", "Vector", "Any direction")], default="NORMAL")
    dir_vec: FloatVectorProperty(name="Vector", default=(0.0, 0.0, 1.0), subtype='XYZ', size=3)
    tilt: FloatProperty(name="Tilt °", default=0.0, soft_min=-89.0, soft_max=89.0,
                        description="Tilt the direction away from itself by this angle")
    tilt_azimuth: FloatProperty(name="Tilt Around °", default=0.0, soft_min=-180.0, soft_max=180.0,
                                description="Which way to tilt (turns around the direction)")
    draft: FloatProperty(name="Draft °", default=0.0, soft_min=-45.0, soft_max=45.0,
                         description="Multi-Draft: the far edge moves outward by height x tan(draft)")
    draft_flip: BoolProperty(name="Flip Draft", default=False, description="Draft towards the inside")
    symmetric: BoolProperty(name="Symmetric", default=False,
                            description="Both ways; with draft, both sides open away from the curve (parting line)")

    def params(self):
        return {"distance": self.distance, "direction": self.direction, "dir_vec": list(self.dir_vec),
                "tilt": self.tilt, "tilt_azimuth": self.tilt_azimuth, "draft": self.draft,
                "draft_flip": self.draft_flip, "symmetric": self.symmetric}


class NPROTO_OT_revolve(_CurveTool, bpy.types.Operator):
    """Revolve curves around an axis"""
    bl_idname = "nproto.revolve"
    bl_label = "Revolve"
    bl_options = {'REGISTER', 'UNDO'}
    op_name = "revolve"

    angle: FloatProperty(name="Angle", default=360.0, min=-360.0, max=360.0)
    axis: EnumProperty(name="Axis", items=AXIS_ITEMS, default="LOCAL_Z")
    pivot: EnumProperty(name="Pivot", items=[("OBJECT", "Object Origin", ""), ("WORLD", "World Origin", "")])
    seam_angle: FloatProperty(name="Seam Angle", default=0.0, soft_min=-180.0, soft_max=180.0,
                              description="Full revolves: where the surface's seam sits around the axis. Turn it "
                                          "away from openings you will trim -- fillets cannot cross a seam")

    def params(self):
        return {"angle": self.angle, "axis": self.axis, "pivot": self.pivot, "seam_angle": self.seam_angle}


def _edge_points(obj, ref):
    """Local-space polyline of a surface edge ref."""
    if features.is_edge_ref(ref):
        return geom.boundary_points(store.get_surfaces(obj)[0], ref, 24)
    return store.display(obj)["edges"][int(ref[1:])]


def edge_loop_refs(obj, ref):
    """All refs of the boundary loop that contains this surface edge.

    An edge that is closed by itself (the rim of a revolve, a circular hole)
    is its own loop: chaining on would run through seams and the other rim."""
    if features.is_edge_ref(ref) or features.is_brep_edge_ref(ref):
        pts = [Vector(p) for p in _edge_points(obj, ref)]
        if len(pts) > 2:
            size = max((p - pts[0]).length for p in pts)
            if size > 0 and (pts[0] - pts[-1]).length <= 1e-5 * size:
                return [ref]
    if features.is_edge_ref(ref):  # open untrimmed surface: its non-degenerate iso boundaries
        out = []
        for r in ("U0", "U1", "V0", "V1"):
            pts = [Vector(p) for p in _edge_points(obj, r)]
            if sum((pts[k + 1] - pts[k]).length for k in range(len(pts) - 1)) > 1e-9:
                out.append(r)
        return out
    if not features.is_brep_edge_ref(ref):
        return [ref]
    brep = store.get_brep(obj)
    edges = brep.edges()

    def key(e):
        try:
            t0, t1 = e.parameter_range()
            return tuple(round(c, 7) for c in e.point_at(0.5 * (t0 + t1)))
        except Exception:
            return None

    index = {key(e): i for i, e in enumerate(edges)}
    target = key(edges[int(ref[1:])])
    for face in brep.faces():
        for loop in face.boundary_loops():
            keys = [key(e) for e, _ in loop]
            if target in keys:
                return [f"E{index[k]}" for k in keys if k in index]
    return [ref]


def polylines_closed(polys, tol):
    """True when the polylines chain into closed loop(s)."""
    ends = []
    for pl in polys:
        if len(pl) < 2:
            return False
        if (pl[0] - pl[-1]).length <= tol:
            continue  # closed by itself
        ends += [pl[0], pl[-1]]
    if not ends and polys:
        return True
    used = [False] * len(ends)
    for i, p in enumerate(ends):
        if used[i]:
            continue
        j = next((j for j in range(len(ends)) if j != i and not used[j] and j // 2 != i // 2
                  and (ends[j] - p).length <= tol), None)
        if j is None:
            return False
        used[i] = used[j] = True
    return True


def _polylines_planar(polys, rel=1e-4):
    P = np.array([tuple(p) for pl in polys for p in pl], float)
    if len(P) < 4:
        return True
    c = P.mean(axis=0)
    _, sv, vt = np.linalg.svd(P - c, full_matrices=False)
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    return float(np.abs((P - c) @ vt[2]).max()) <= rel * size


class NPROTO_OT_planar(PickMixin, bpy.types.Operator):
    """Fill a closed planar profile (curves and/or surface edges) with a trimmed
    planar surface. Pick an edge to take its whole loop; finishes when closed"""
    bl_idname = "nproto.planar"
    bl_label = "Planar"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 1
    pick_max = 512
    finish_on_min = False

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    chain: BoolProperty(name="Chain Edge Loops", default=True, options={'SKIP_SAVE'},
                        description="Picking a surface edge picks its whole boundary loop")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        ins = _curve_inputs(context)
        if ins and not event.alt:
            self.inputs = _encode_inputs(ins)
            return self.execute(context)
        return self.start_picking(context)

    def _status(self, context):
        super()._status(context)
        if self._picks:
            context.area.header_text_set(
                f"Planar: {len(self._picks)} edge(s)/curve(s) picked, profile not closed yet · "
                "keep picking (Enter fills what you have)   Tab cycle · Backspace undo · Esc cancel")

    def expand_pick(self, cand):
        obj, ref = cand[0], cand[1]
        if not self.chain or obj.type == 'CURVE':
            return [cand]
        refs = edge_loop_refs(obj, ref)
        byref = {(c[0].name, c[1]): c for c in self._cands}
        out = [byref[(obj.name, r)] for r in refs if (obj.name, r) in byref]
        # only a PLANAR loop is taken whole (a planar face's boundary); on a
        # curved surface the picked edge is one piece of a profile made of
        # edges from several surfaces (e.g. a wall's top edge + a cap's edge)
        if len(out) > 1 and not _polylines_planar([c[2] for c in out]):
            return [cand]
        # an edge that continues the profile being picked (touches the end of an
        # already picked edge / curve) is just that edge: its face's loop would
        # add a second, overlapping loop (e.g. a cap's straight edge closing a
        # U-shaped opening must not bring the whole top face's outline)
        if len(out) > 1 and self._picks and self._touches_picked(cand):
            return [cand]
        return out or [cand]

    def _touches_picked(self, cand):
        byref = {(c[0].name, c[1]): c[2] for c in self._cands}
        ends = [pl[i] for o, r in self._picks if (o.name, r) in byref for pl in [byref[(o.name, r)]] for i in (0, -1)]
        if not ends:
            return False
        allp = [p for pl in [byref[(o.name, r)] for o, r in self._picks if (o.name, r) in byref] for p in pl]
        size = max((a - b).length for a in allp[:1] for b in allp) or 1.0
        tol = 1e-3 * size
        return any((cand[2][i] - e).length <= tol for i in (0, -1) for e in ends)

    def closes_loop(self):
        byref = {(c[0].name, c[1]): c[2] for c in self._cands}
        polys = [byref[(o.name, r)] for o, r in self._picks if (o.name, r) in byref]
        if not polys:
            return False
        allp = [p for pl in polys for p in pl]
        size = max((a - b).length for a in allp[:1] for b in allp) or 1.0
        return polylines_closed(polys, 1e-4 * size)

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            ins = _decode_inputs(self.inputs) if self.inputs else _curve_inputs(context)
        except Exception as exc:
            self.report({'ERROR'}, f"Planar: {exc}")
            return {'CANCELLED'}
        if not ins:
            self.report({'WARNING'}, "Select closed planar curves, or Alt+click Planar to pick edges")
            return {'CANCELLED'}
        try:
            obj = _make_result(context, "Planar", "planar", ins, {})
        except Exception as exc:
            msg = str(exc)
            if "CreatePlanarFaces" in msg:
                msg = ("the profile is not closed or not planar -- pick every edge / curve around the opening "
                       f"({len(ins)} picked)")
            self.report({'ERROR'}, f"Planar failed: {msg}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        return {'FINISHED'}


class NPROTO_OT_offset(AskValue, bpy.types.Operator):
    """Offset NURBS surfaces along their normal"""
    bl_idname = "nproto.offset"
    bl_label = "Offset"
    bl_options = {'REGISTER', 'UNDO'}

    distance: FloatProperty(name="Distance", default=0.1, unit='LENGTH',
                            description="Along the normals (negative: the other side)")

    def invoke(self, context, event):
        if not any(store.is_nurbs(o) for o in context.selected_objects):
            self.report({'WARNING'}, "Select NURBS surfaces")
            return {'CANCELLED'}
        return self.ask(context)

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        made = []
        try:
            for o in [o for o in context.selected_objects if store.is_nurbs(o)]:
                made.append(_make_result(context, o.name + "_offset", "offset", [(o, "U0")],
                                         {"distance": self.distance}, geom.world_matrix(o)))
        except Exception as exc:
            self.report({'ERROR'}, f"Offset failed: {exc}")
            return {'CANCELLED'}
        if not made:
            self.report({'WARNING'}, "Select NURBS surfaces")
            return {'CANCELLED'}
        _select_only(context, made)
        return {'FINISHED'}


class NPROTO_OT_thicken(AskValue, bpy.types.Operator):
    """Thicken surfaces (single or joined) into closed solids"""
    bl_idname = "nproto.thicken"
    bl_label = "Thicken"
    bl_options = {'REGISTER', 'UNDO'}

    distance: FloatProperty(name="Thickness", default=0.01, unit='LENGTH',
                            description="Wall thickness along the normals (negative: the other side)")
    symmetric: BoolProperty(name="Symmetric", default=False, description="Half the thickness on each side")

    def invoke(self, context, event):
        if not any(store.is_nurbs(o) for o in context.selected_objects):
            self.report({'WARNING'}, "Select NURBS surfaces")
            return {'CANCELLED'}
        return self.ask(context)

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        made = []
        for o in [o for o in context.selected_objects if store.is_nurbs(o)]:
            try:
                made.append(_make_result(context, o.name + "_thick", "thicken", [(o, "SURF")],
                                         {"distance": self.distance, "symmetric": self.symmetric},
                                         geom.world_matrix(o)))
            except Exception as exc:
                self.report({'ERROR'}, f"Thicken {o.name} failed: {exc}")
        if not made:
            if not any(store.is_nurbs(o) for o in context.selected_objects):
                self.report({'WARNING'}, "Select NURBS surfaces")
            return {'CANCELLED'}
        srcs = [o for o in context.selected_objects if store.is_nurbs(o) and o not in made]
        if features.history_enabled(context):
            features.keep_inputs(context, srcs)          # Restore Inputs brings the sheets back
        else:
            for o in srcs:
                bpy.data.objects.remove(o)               # the solid replaces its sheet
        _select_only(context, made)
        return {'FINISHED'}


class NPROTO_OT_from_surface(bpy.types.Operator):
    """Convert Blender NURBS surface objects into kernel NURBS surfaces"""
    bl_idname = "nproto.from_blender_surface"
    bl_label = "Convert Blender Surface"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        from .kernel import ext
        made = []
        for o in [o for o in context.selected_objects if o.type == 'SURFACE']:
            for sp in o.data.splines:
                nu, nv = sp.point_count_u, sp.point_count_v
                if not (sp.use_endpoint_u and sp.use_endpoint_v):
                    self.report({'WARNING'}, f"{o.name}: enable Endpoint U and V for an exact conversion; skipped")
                    continue
                pu, pv = sp.order_u - 1, sp.order_v - 1
                M = geom.world_matrix(o)
                pts, w = [], []
                for iu in range(nu):
                    for iv in range(nv):
                        co = sp.points[iv * nu + iu].co
                        pts.append(tuple(M @ Vector(co[:3])))
                        w.append(co[3])
                rational = any(abs(x - 1.0) > 1e-9 for x in w)
                s = ext.surface_from_cvs(pu, pv, pts, geom.clamped_uniform_knots(nu, pu),
                                         geom.clamped_uniform_knots(nv, pv), w if rational else None)
                made.append(store.new_object(context, o.name + "_nurbs", s))
        if not made:
            return {'CANCELLED'}
        _select_only(context, made)
        return {'FINISHED'}


# --------------------------------------------------------------------------
# History and display
# --------------------------------------------------------------------------

class NPROTO_OT_delete_history(bpy.types.Operator):
    """Remove construction history (the surfaces stop following their inputs)"""
    bl_idname = "nproto.delete_history"
    bl_label = "Delete History"
    bl_options = {'REGISTER', 'UNDO'}

    scope: EnumProperty(items=[("SELECTED", "Selected", ""), ("ALL", "All", "")], default="SELECTED")

    def execute(self, context):
        objs = list(context.selected_objects if self.scope == "SELECTED" else context.scene.objects)
        n, freed = 0, []
        for o in objs:
            if features.has_history(o):
                freed += features.history_inputs(o)
                features.clear_history(o)
                n += 1
        # the inputs those results kept (hidden) are not needed any more
        gone = features.purge_inputs(context, list(dict.fromkeys(freed)))
        msg = f"History removed from {n} object(s)"
        if gone:
            msg += f"; deleted {len(gone)} kept input(s) nothing uses now: " + ", ".join(gone)
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class NPROTO_OT_collect_inputs(bpy.types.Operator):
    """Move the surfaces / curves that Join, Boolean, Boundary Fill and Thicken
    results keep for their history into the hidden NURBS History collection
    (tidies scenes made before it existed, or after Unhide All)"""
    bl_idname = "nproto.collect_inputs"
    bl_label = "Collect Inputs"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        kept = []
        for o in context.scene.objects:
            h = features.history(o) if (store.is_nurbs(o) or o.type == 'CURVE') else None
            if h and h["op"] in features.KEEP_OPS:
                kept += [src for src, _ in h["inputs"] if src not in kept]
        if not kept:
            self.report({'INFO'}, "No kept inputs in this scene")
            return {'CANCELLED'}
        features.keep_inputs(context, kept)
        self.report({'INFO'}, f"{len(kept)} kept input(s) in the hidden '{features.HISTORY_COLL}' collection")
        return {'FINISHED'}


class NPROTO_OT_rebuild(bpy.types.Operator):
    """Rebuild selected NURBS objects from history, or re-tessellate them"""
    bl_idname = "nproto.rebuild"
    bl_label = "Rebuild / Re-tessellate"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        for o in context.selected_objects:
            if store.is_nurbs(o):
                try:
                    if not features.rebuild(o, context, force=True):
                        store.retessellate(o, context)
                except Exception as exc:
                    self.report({'ERROR'}, f"{o.name}: {exc}")
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Blend curve
# --------------------------------------------------------------------------

def _closest_ends(ca, cb):
    """End indices (0/1) of two kernel curves whose endpoints are nearest each other."""
    pa = [np.array(ca.start_point()), np.array(ca.end_point())]
    pb = [np.array(cb.start_point()), np.array(cb.end_point())]
    best = min(((i, j) for i in (0, 1) for j in (0, 1)), key=lambda ij: np.linalg.norm(pa[ij[0]] - pb[ij[1]]))
    return best


ALIGN_MODES = [("EDGE", "Continue Edge", "The curve continues the edge (curve + edge form one G1/G2 curve)"),
               ("SURFACE", "Tangent to Surface", "The curve stays tangent to the edge's surface (G2: matches its "
                                                 "normal curvature)")]


class NPROTO_OT_align_curve(PickMixin, bpy.types.Operator):
    """Align a curve end to an edge with G0 / G1 / G2 by moving only the end CVs.
    Click the curve near the end to align, then the edge (surface edge or curve)"""
    bl_idname = "nproto.align_curve"
    bl_label = "Align Curve"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 2
    pick_max = 2

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    end: IntProperty(name="End", default=0, min=0, max=1, options={'HIDDEN'})
    cont: EnumProperty(name="Continuity", items=CONT_ITEMS, default="G2")
    weight: FloatProperty(name="Weight", default=1.0, min=0.01, soft_max=3.0,
                          description="Tangent length at the end, relative to the curve's own")
    mode: EnumProperty(name="Mode", items=ALIGN_MODES, default="EDGE")
    snap: EnumProperty(name="Snap", items=[("END", "Edge End", "The nearest end of the edge"),
                                           ("NEAREST", "Nearest Point", "The nearest point on the edge")],
                       default="END")
    flip: BoolProperty(name="Flip", default=False, description="Reverse the edge direction used for G1 / G2")

    def expand_pick(self, cand):
        if not getattr(self, "_picks", None):          # first pick: the curve; which end?
            obj, ref, pts = cand
            region, rv3d = bpy.context.region, bpy.context.region_data
            a = view3d_utils.location_3d_to_region_2d(region, rv3d, pts[0])
            b = view3d_utils.location_3d_to_region_2d(region, rv3d, pts[-1])
            m = getattr(self, "_last_mouse", None)
            if a is not None and b is not None and m is not None:
                self.end = 0 if (a - m).length < (b - m).length else 1
        return [cand]

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        return self.start_picking(context)

    def execute(self, context):
        from . import align
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            picks = _decode_inputs(self.inputs)
            if len(picks) != 2 or picks[0][0].type != 'CURVE':
                raise ValueError("pick the curve (near the end to align), then the edge")
            obj, target = picks[0][0], picks[1]
            if target[0] == obj:
                raise ValueError("pick an edge of another object")
            h = features.history(obj)
            if h is not None and h["op"] == "align_curve":
                inputs = list(h["inputs"])
                ends = dict(h["params"].get("ends", {}))
            else:
                if features.has_history(obj):
                    features.clear_history(obj)
                    self.report({'INFO'}, f"{obj.name}: its construction history was replaced by Align")
                inputs, ends = [(obj, "S0")], {}
            # the other end keeps its alignment; this end gets the new edge
            keep = {e: v for e, v in ends.items() if e != str(self.end)}
            used = sorted({int(v["input"]) for v in keep.values()})
            new_inputs = [(obj, "S0")] + [inputs[i] for i in used]
            remap = {old: k + 1 for k, old in enumerate(used)}
            for v in keep.values():
                v["input"] = remap[int(v["input"])]
            new_inputs.append(target)
            keep[str(self.end)] = {"input": len(new_inputs) - 1, "cont": self.cont, "weight": self.weight,
                                   "mode": self.mode, "snap": self.snap, "flip": self.flip}
            kind, deg, Q, _ = align.solve(obj, keep, new_inputs)
            align.write(obj, kind, deg, Q)
            if features.history_enabled(context):
                features.record(obj, "align_curve", new_inputs, {"ends": keep})
        except Exception as exc:
            self.report({'ERROR'}, f"Align failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        return {'FINISHED'}

    def draw(self, context):
        lay = self.layout
        lay.label(text=f"End {'start' if self.end == 0 else 'end'} of the curve", icon='CURVE_DATA')
        lay.row().prop(self, "cont", expand=True)
        lay.prop(self, "mode")
        lay.prop(self, "snap")
        row = lay.row(align=True)
        row.prop(self, "weight")
        row.prop(self, "flip", toggle=True)


class NPROTO_OT_blend_curve(PickMixin, bpy.types.Operator):
    """Blend curve between two curve ends with G0 / G1 / G2 and a weight per side.
    Click each curve near the end to blend from"""
    bl_idname = "nproto.blend_curve"
    bl_label = "Blend Curve"
    bl_options = {'REGISTER', 'UNDO'}

    pick_min = 2
    pick_max = 2

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    end_a: IntProperty(name="End A", default=1, min=0, max=1, options={'HIDDEN'})
    end_b: IntProperty(name="End B", default=0, min=0, max=1, options={'HIDDEN'})
    cont_a: EnumProperty(name="Continuity A", items=CONT_ITEMS, default="G2")
    cont_b: EnumProperty(name="Continuity B", items=CONT_ITEMS, default="G2")
    w_a: FloatProperty(name="Weight A", default=1.0, min=0.0, soft_max=3.0,
                       description="Tangent weight at A (how far the blend follows curve A's direction)")
    w_b: FloatProperty(name="Weight B", default=1.0, min=0.0, soft_max=3.0)
    c_a: FloatProperty(name="Curvature A", default=0.0, soft_min=-2.0, soft_max=2.0,
                       description="G2 shape at A (moves the 3rd CV along the tangent)")
    c_b: FloatProperty(name="Curvature B", default=0.0, soft_min=-2.0, soft_max=2.0)
    flip_a: BoolProperty(name="Flip A", default=False, description="Reverse the tangent direction at A")
    flip_b: BoolProperty(name="Flip B", default=False)

    def expand_pick(self, cand):
        # which end of the picked curve is nearer the click (in screen space)?
        obj, ref, pts = cand
        region, rv3d = bpy.context.region, bpy.context.region_data
        a = view3d_utils.location_3d_to_region_2d(region, rv3d, pts[0])
        b = view3d_utils.location_3d_to_region_2d(region, rv3d, pts[-1])
        m = getattr(self, "_last_mouse", None)
        end = 1
        if a is not None and b is not None and m is not None:
            end = 0 if (a - m).length < (b - m).length else 1
        if not hasattr(self, "_ends"):
            self._ends = []
        self._ends = self._ends[:len(self._picks)] + [end]
        return [cand]

    def _finish(self, context):
        ends = getattr(self, "_ends", [1, 0])
        self.end_a, self.end_b = ends[0], ends[1] if len(ends) > 1 else 0
        return super()._finish(context)

    def _from_selection(self, context):
        sel = [o for o in context.selected_objects if o.type == 'CURVE' and features.PROP_HIST not in o]
        if len(sel) != 2:
            return False
        act = context.active_object
        sel.sort(key=lambda o: o == act)
        ca = features.input_curve(sel[0], "S0")
        cb = features.input_curve(sel[1], "S0")
        self.end_a, self.end_b = _closest_ends(ca, cb)
        self.inputs = _encode_inputs([(sel[0], "S0"), (sel[1], "S0")])
        return True

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not event.alt and self._from_selection(context):
            return self.execute(context)
        self._ends = []
        return self.start_picking(context)

    def _params(self):
        return {"end_a": self.end_a, "end_b": self.end_b, "cont_a": self.cont_a, "cont_b": self.cont_b,
                "w_a": self.w_a, "w_b": self.w_b, "c_a": self.c_a, "c_b": self.c_b,
                "flip_a": self.flip_a, "flip_b": self.flip_b}

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not self.inputs:
            self._from_selection(context)  # scripts / no viewport: use the selection
        try:
            inputs = _decode_inputs(self.inputs)
            if len(inputs) != 2:
                raise ValueError("select or pick two curves")
            g = features.run("blend_curve", inputs, self._params())
        except Exception as exc:
            self.report({'ERROR'}, f"Blend curve failed: {exc}")
            return {'CANCELLED'}
        cu = bpy.data.curves.new("BlendCurve", 'CURVE')
        obj = bpy.data.objects.new("BlendCurve", cu)
        (context.collection or context.scene.collection).objects.link(obj)
        geom.write_curve_object(obj, g[1], g[2])
        if context.scene.nproto_settings.construction_history:
            features.record(obj, "blend_curve", inputs, self._params())
        _select_only(context, [obj])
        return {'FINISHED'}

    def draw(self, context):
        lay = self.layout
        for side in "ab":
            box = lay.box()
            box.label(text=f"Side {side.upper()}")
            box.row().prop(self, f"cont_{side}", expand=True)
            col = box.column(align=True)
            col.prop(self, f"w_{side}", text="Weight")
            sub = col.row()
            sub.enabled = getattr(self, f"cont_{side}") == "G2"
            sub.prop(self, f"c_{side}", text="Curvature")
            box.prop(self, f"flip_{side}")


classes = (NPROTO_OT_align_curve, 
    NPROTO_OT_blend_curve,
    NPROTO_OT_loft, NPROTO_OT_extrude, NPROTO_OT_revolve, NPROTO_OT_planar, NPROTO_OT_offset, NPROTO_OT_thicken,
    NPROTO_OT_from_surface, NPROTO_OT_delete_history, NPROTO_OT_collect_inputs, NPROTO_OT_rebuild,
)
