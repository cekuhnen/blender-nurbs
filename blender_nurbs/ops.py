"""Modelling operators (viewport tools)."""

import json
import math

import bpy
import numpy as np
from bpy.props import FloatVectorProperty, BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras import view3d_utils
from mathutils import Vector

from . import draw, features, geom, hints, store
from .valuedrag import ValueDrag, angle_pick, draft_angle, rotate_points, ruled_preview


def _seg_dist_np(p, a, b):
    ab = b - a
    L = float(ab @ ab)
    t = 0.0 if L == 0 else min(1.0, max(0.0, float((p - a) @ ab) / L))
    return float(np.linalg.norm(a + t * ab - p))

MOUSE_PICK = [("MOUSE_LMB", "Pick"), ("EVENT_RETURN", "Finish"), ("EVENT_ESC", "Cancel")]
from .kernel import load as load_kernel

CONT_ITEMS = [
    ("G0", "G0 Position", "Edges touch only"),
    ("G1", "G1 Tangent", "Tangent continuous with the source surface"),
    ("G2", "G2 Curvature", "Curvature continuous with the source surface"),
]
# Loft's sides: also G3 (the curvature's rate of change carries on too)
LOFT_CONT_ITEMS = CONT_ITEMS + [
    ("G3", "G3 Curvature Rate", "Curvature continuous, and the curvature changes across the edge at the source "
                                "surface's rate (a source with a CV net; up to G2 with rails)"),
]
AXIS_ITEMS = [
    ("X", "World X", ""), ("Y", "World Y", ""), ("Z", "World Z", ""),
    ("LOCAL_X", "Local X", ""), ("LOCAL_Y", "Local Y", ""), ("LOCAL_Z", "Local Z", ""),
]
# Revolve: also a line picked in the view (Axis Point + Axis Direction, world)
REVOLVE_AXIS_ITEMS = AXIS_ITEMS + [("PICKED", "Picked Line", "A line, open curve or straight edge picked for it: "
                                                               "from its start to its end (A while revolving)")]


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
                    pts = [M @ Vector(p) for p in geom.boundary_points(surf, which, 48, corners=True)]
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


def _straight_line(curve, rel_tol=1e-4):
    """(start, unit direction) of a curve that is a straight line (every
    point within rel_tol of its length off the chord), else None."""
    P = geom.sample_curve(curve, 17)
    d = P[-1] - P[0]
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        return None
    u = d / L
    rel = P - P[0]
    off = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    return (P[0], u) if float(off.max()) <= rel_tol * L else None


def _on_axis(S, O, z, rel_tol=1e-6):
    """Do the points S lie on the axis through O along z (all, within a
    millionth of their size)?"""
    S = np.asarray(S, float)
    rel = S - np.asarray(O, float)
    z = np.asarray(z, float)
    rad = np.linalg.norm(rel - np.outer(rel @ z, z), axis=1)
    size = float(np.ptp(S, axis=0).max()) or float(np.linalg.norm(rel, axis=1).max()) or 1.0
    return float(rad.max()) <= rel_tol * size


class _RegionMouse:
    """A mouse position (region pixels), standing in for an event."""

    def __init__(self, m):
        self.mouse_region_x, self.mouse_region_y = float(m[0]), float(m[1])


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


def _carry_props(op):
    """The operator's settings, to start it again with the picks (PickMixin._finish).
    A command started from Python gets its defaults, not the settings it ran with,
    so without them a direction or size set when the tool started was lost."""
    out = {"inputs": op.inputs}
    for p in op.properties.bl_rna.properties:      # the operator's properties only
        k = p.identifier
        if k in ("rna_type", "inputs") or p.is_readonly or p.type in ('POINTER', 'COLLECTION'):
            continue
        if p.is_skip_save or not op.properties.is_property_set(k):
            continue
        v = getattr(op, k)
        out[k] = v[:] if getattr(p, "is_array", False) else v
    return out


GIZMO = {"held": False}     # the left button went down on the navigation gizmo (PickMixin._in_view)


def natural_side(obj, ref):
    """The side (U0 / U1 / V0 / V1) of an untrimmed surface that its edge
    "E<i>" (Edge mode's name for it) lies on; None if it is not one."""
    if not (ref.startswith("E") and ref[1:].isdigit() and store.is_nurbs(obj) and store.is_natural(obj)):
        return None
    try:
        from .kernel import trimext
        brep = store.get_brep(obj)
        r = trimext.edge_on_iso(brep.faces()[0].surface(), brep.edges()[int(ref[1:])])
    except Exception:
        return None
    return r[0] if r is not None else None


def main_region(area):
    """The 3D view's drawing region of an area (None if it has none)."""
    if area is None:
        return None
    return next((r for r in area.regions if r.type == 'WINDOW'), None)


class InView:
    """The event with its mouse counted from the 3D view's main region."""

    def __init__(self, event, region):
        self._e = event
        self.mouse_region_x = event.mouse_x - region.x
        self.mouse_region_y = event.mouse_y - region.y

    def __getattr__(self, name):
        return getattr(self._e, name)


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

    def pick_count(self):
        """How many inputs count toward pick_min (Loft: its sides)."""
        return len(self._picks)

    def after_pick(self, group, event):
        """Called with the candidates one click added."""

    def after_unpick(self, n):
        """Called after Backspace took back the last n picks."""

    def tool_key(self, context, event):
        """A key of the tool's own while picking: True when handled."""
        return False

    def _status(self, context):
        n = self.pick_count()
        state = [self.pick_state() if hasattr(self, "pick_state") else f"pick input {n + 1}",
                 "Enter to finish" if n >= self.pick_min else ""]
        ties = getattr(self, "_ties", [])
        if len(ties) > 1:
            cur = ties[self._cycle % len(ties)]
            state.append(f"{cur[0].name} {cur[1]} · {len(ties)} overlapping")
        hints.show(context, self.bl_label, state, self.pick_rows(), MOUSE_PICK)

    def pick_rows(self):
        """The picking keys for the hint panel (hints.py), plus the tool's own
        (extra_keys: [(keys, text)]) and its advice (extra_help)."""
        n = (self.pick_count() if hasattr(self, "pick_count") else len(self._picks)) if hasattr(self, "_picks") else 0
        rows = [("Pick", [hints.key("Shift+LMB", "keep picking"),
                          hints.key("Tab", "cycle overlapping", on=True if len(getattr(self, "_ties", [])) > 1 else None),
                          hints.key("Backspace", "undo"),
                          hints.key("Enter", "finish", on=True if n >= self.pick_min else False)])]
        extra = getattr(self, "extra_keys", ())
        if extra:
            rows.append(("Also", [hints.key(k, t) for k, t in extra]))
        tip = getattr(self, "extra_help", "").strip().strip("·").strip()
        if tip.startswith("(") and tip.endswith(")"):
            tip = tip[1:-1]
        if tip:
            rows.append(("", [hints.note(tip)]))
        return rows

    def hints_key(self, context, event):
        if hints.handle(event):
            context.area.tag_redraw()
            return True
        return False

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

    accepts_picks = False   # picked edges (Face / Edge mode) that fit become the inputs

    def picked_inputs(self, context):
        """The edges picked in Edge mode as this tool's inputs ((obj, ref) in
        pick order, a Planar edge chained to its loop), when every one is
        something it can take and their number fits; else None."""
        from . import subsel
        picked = subsel.selected_edges(context)
        if not picked:
            return None
        byref = {(c[0].name, c[1]): c for c in self._cands}
        out = []
        for o, r in picked:
            cand = byref.get((o.name, r))
            if cand is None:
                # an untrimmed surface's candidates are its sides U0 / U1 / V0 / V1, Edge mode names E<i>
                cand = byref.get((o.name, natural_side(o, r)))
            if cand is None:
                return None
            for c in self.expand_pick(cand):
                if (c[0], c[1]) not in out:
                    out.append((c[0], c[1]))
        return out if self.pick_min <= len(out) <= self.pick_max else None

    def start_picking(self, context, event=None):
        self._cands = self.pick_candidates(context)
        if not self._cands:
            self.report({'WARNING'}, "Nothing to pick: add curves or NURBS surfaces first")
            return {'CANCELLED'}
        self._picks = []
        self._groups = []
        if self.accepts_picks and not (event is not None and event.alt):
            pre = self.picked_inputs(context)      # (expand_pick may look at self._picks: empty here)
            if pre:
                self._picks, self._groups = pre, [1] * len(pre)
                return self._finish(context)
        self._ties, self._tie_key, self._cycle = [], None, 0
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["colors"] = []
        draw.pick_overlay["hover"] = None
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {'RUNNING_MODAL'}

    def _end(self, context):
        draw.pick_overlay["picked"] = []
        draw.pick_overlay["colors"] = []
        draw.pick_overlay["hover"] = None
        hints.clear(context)

    def modal(self, context, event):
        # started from a sidebar panel the tool's region is the panel's: no
        # view there, the mouse counted from its corner. Pick in the 3D view's region.
        reg = main_region(context.area)
        if reg is None or reg == context.region:
            return self._pick_modal(context, event)
        with context.temp_override(region=reg):
            return self._pick_modal(bpy.context, InView(event, reg))

    def _pick_modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and self.tool_key(context, event):
            self._status(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'INBETWEEN_MOUSEMOVE':
            return {'PASS_THROUGH'}       # not used here; a gizmo drag needs them unhandled
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not self._in_view(context, event):
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
                    self.after_pick(group, event)
                context.area.tag_redraw()
                if self.closes_loop() and not (event.shift and getattr(self, "shift_extends", False)):
                    return self._finish(context)
                if len(self._picks) >= self.pick_max or (
                        self.finish_on_min and self.pick_count() >= self.pick_min and not event.shift):
                    return self._finish(context)
                self._status(context)
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS' and self._picks:
            n = self._groups.pop() if self._groups else 1
            for _ in range(n):
                self._picks.pop()
                draw.pick_overlay["picked"].pop()
            self.after_unpick(n)
            self._status(context)
            context.area.tag_redraw()
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if self.pick_count() >= self.pick_min:
                return self._finish(context)
        elif event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    @staticmethod
    def _in_view(context, event):
        """Is the mouse over the 3D view's drawing area (not a sidebar / header
        overlapping it, nor the navigation gizmo: clicks and drags there go to
        Blender, so the view can be turned while a tool runs)?
        A press on the gizmo is remembered (GIZMO): Blender makes a drag of
        that press only from mouse moves nobody else handled, so until the drag
        starts (its rotate / pan / zoom takes over) or the button comes up,
        the moves go to Blender too."""
        if GIZMO["held"]:
            etype = getattr(event, "type", None)
            if etype in ('MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'):
                return False
            if etype == 'LEFTMOUSE':
                GIZMO["held"] = False
                if getattr(event, "value", None) != 'PRESS':    # released, or the drag has started
                    return False
        area = context.area
        if area is None:
            return True
        x, y = event.mouse_x, event.mouse_y
        win = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if win is None or not (win.x <= x < win.x + win.width and win.y <= y < win.y + win.height):
            return False
        for r in area.regions:   # overlapping regions (region overlap draws the sidebar over the view)
            if r.type in {'UI', 'HEADER', 'TOOL_HEADER', 'TOOLS', 'HUD', 'ASSET_SHELF'} and r.width > 1 and r.height > 1:
                if r.type == 'HUD' and r.y < win.y:
                    continue         # a hidden Adjust Last Operation panel keeps its size, parked below the view
                if r.x <= x < r.x + r.width and r.y <= y < r.y + r.height:
                    return False
        from .subsel import over_navigation
        try:
            on_gizmo = over_navigation(context, area, win, x, y)
        except Exception:
            on_gizmo = False
        if on_gizmo:
            if getattr(event, "type", None) == 'LEFTMOUSE' and getattr(event, "value", None) == 'PRESS':
                GIZMO["held"] = True
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
            getattr(getattr(bpy.ops, mod.lower()), name)('INVOKE_DEFAULT', **_carry_props(self))
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


def _is_curve_cand(c):
    return c[0].type == 'CURVE'


def _poly_ends(pts):
    """((start point, outward direction), (end point, outward direction)) of a
    pick polyline; directions from three points (second order)."""
    P = [Vector(p) for p in pts]
    if len(P) < 3:
        d = (P[-1] - P[0]).normalized()
        return (P[0], -d), (P[-1], d)
    f0 = -3 * P[0] + 4 * P[1] - P[2]
    fe = 3 * P[-1] - 4 * P[-2] + P[-3]
    return (P[0], -f0.normalized()), (P[-1], fe.normalized())


def _touch_tol(polys):
    pts = [Vector(p) for pl in polys for p in (pl[0], pl[-1])]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return max(1e-3 * (hi - lo).length, 1e-7)


def tangent_chain(hit, cands, max_deg=2.0):
    """The candidates that continue `hit` tangentially (within max_deg) at
    either end, end to end, the same kind (surface edges / curves): the
    whole smooth run of edges one click on a side means."""
    chain = [hit]
    keyof = lambda c: (c[0].name, c[1])
    used = {keyof(hit)}
    kind = _is_curve_cand(hit)
    pool = [c for c in cands if _is_curve_cand(c) == kind and len(c[2]) > 1]
    tol = _touch_tol([c[2] for c in pool] or [hit[2]])
    cos_max = np.cos(np.radians(max_deg))
    ends = {k: _poly_ends(c[2]) for c in pool for k in [keyof(c)]}
    for side in (1, 0):                       # grow past the end, then past the start
        cur = ends[keyof(hit)][side]
        while True:
            best = None
            for c in pool:
                k = keyof(c)
                if k in used:
                    continue
                for j, (q, out) in enumerate(ends[k]):
                    if (q - cur[0]).length > tol:
                        continue
                    cosang = float(cur[1].dot(-out))      # the next piece goes on the way this one went
                    if cosang >= cos_max and (best is None or cosang > best[0]):
                        best = (cosang, c, j)
            if best is None:
                break
            _, c, j = best
            used.add(keyof(c))
            if side == 1:
                chain.append(c)
            else:
                chain.insert(0, c)
            cur = ends[keyof(c)][1 - j]          # carry on from its far end
    return chain


def touches(polys_a, polys_b):
    """Does an end of one polyline in polys_b meet an end of one in polys_a?"""
    if not polys_a or not polys_b:
        return False
    tol = _touch_tol(list(polys_a) + list(polys_b))
    ea = [Vector(p) for pl in polys_a for p in (pl[0], pl[-1])]
    return any((Vector(q) - e).length <= tol for pl in polys_b for q in (pl[0], pl[-1]) for e in ea)


def group_into_sides(cands):
    """Side index of each candidate (in order): one joins the side before it
    when it touches it end to end and is the same kind; otherwise it starts
    a new side."""
    out = []
    for i, c in enumerate(cands):
        if out:
            cur = [cands[j] for j in range(i) if out[j] == out[-1]]
            if _is_curve_cand(cur[0]) == _is_curve_cand(c) and touches([x[2] for x in cur], [c[2]]):
                out.append(out[-1])
                continue
        out.append(out[-1] + 1 if out else 0)
    return out


def side_counts(side_of):
    counts = []
    for i, s in enumerate(side_of):
        if i == 0 or s != side_of[i - 1]:
            counts.append(0)
        counts[-1] += 1
    return counts


EDGE_ITEMS = [
    ("FREE", "Free Edges", "The loft leaves the side square to it (an untrimmed edge: along its face's own lines)",
     'MOD_CURVE', 0),
    ("EDGES", "Align Edges", "At the side's corners the loft carries on the surface's own edges, so its outline "
     "flows out of them (G2: in their curvature too)", 'MOD_SIMPLEDEFORM', 1),
    ("SURFACE", "Align to Surface", "The loft carries on the surface's own parameter lines across the side",
     'SURFACE_NSURFACE', 2),
]

CHAIN_ITEMS = [("TANGENT", "Tangent Chain", "A click picks the edge and every edge that continues it tangentially"),
               ("SINGLE", "Single Edge", "A click picks just that edge (Shift+click more edges that touch it)")]


class NPROTO_OT_loft(PickMixin, bpy.types.Operator):
    """Loft a surface between curves or surface edges (G0 / G1 / G2 / G3). A side
    can be several edges end to end: picked edges that touch form one side
    (Tangent Chain picks a smooth run of edges with one click)"""
    bl_idname = "nproto.loft"
    bl_label = "Loft"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 2
    pick_max = 64
    extra_keys = (("C", "tangent chain on / off"), ("N", "start the next side"), ("R", "pick rails / profiles"))
    extra_help = "(edges that touch end to end make one side; R: up to two rails, guide curves along the loft)"

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    sides: StringProperty(options={'HIDDEN', 'SKIP_SAVE'},
                          description="How many inputs each side takes, in order (JSON list)")
    chain: EnumProperty(name="Pick", items=CHAIN_ITEMS, default="TANGENT",
                        description="What one click on an edge picks")
    chain_angle: FloatProperty(name="Chain Angle", default=2.0, min=0.01, max=30.0,
                               description="Tangent Chain: edges meeting within this angle (degrees) are one run")
    use_selection: BoolProperty(name="Use Selection", default=False, options={'SKIP_SAVE'},
                                description="Loft the selected curves instead of picking")
    cont_a: EnumProperty(name="Continuity A", items=LOFT_CONT_ITEMS, default="G0")
    cont_b: EnumProperty(name="Continuity B", items=LOFT_CONT_ITEMS, default="G0")
    tan_a: FloatProperty(name="Tangent A", default=1.0, min=0.0, soft_max=3.0,
                         description="G1 magnitude at input A (relative to loft span)")
    tan_b: FloatProperty(name="Tangent B", default=1.0, min=0.0, soft_max=3.0)
    cur_a: FloatProperty(name="Curvature A", default=0.0, soft_min=-2.0, soft_max=2.0,
                         description="G2 / G3 shape at input A (moves the 3rd CV row along the tangent)")
    cur_b: FloatProperty(name="Curvature B", default=0.0, soft_min=-2.0, soft_max=2.0)
    auto_align: BoolProperty(name="Auto Align", default=True,
                             description="Flip inputs, and start closed sections in the same direction, so the loft does not twist")
    flip_b: BoolProperty(name="Flip B", default=False)
    ruled: BoolProperty(name="Ruled", default=False, description="Straight (degree 1) loft between 2 curves")
    closed: BoolProperty(name="Closed", default=False,
                         description="Loft through the sections and back to the first: a closed ring, smooth all "
                                     "round, the seam included (3 or more sections)")
    samples: IntProperty(name="Samples", default=24, min=6, max=200,
                         description="Samples along the edges for G1 / G2 / G3 lofts")
    fit: EnumProperty(name="Fit", items=[
        ("SPANS", "Bezier Spans", "G1 / G2 to trimmed edges: a few Bezier spans of one degree, knots where the "
         "continuity needs them (light, symmetric nets)"),
        ("DENSE", "Fitted", "G1 / G2 to trimmed edges: a least-squares net through the samples (many CVs)")],
        default="SPANS")
    span_degree: IntProperty(name="Span Degree", default=5, min=3, max=9,
                             description="Bezier Spans: degree along the sides")
    span_faces: BoolProperty(name="Split into Faces", default=False,
                             description="Bezier Spans: one Bezier patch per span, joined into one body")
    edges_a: EnumProperty(name="Edges A", items=EDGE_ITEMS, default="EDGES",
                          description="G1 / G2: which way the loft leaves side A")
    edges_b: EnumProperty(name="Edges B", items=EDGE_ITEMS, default="EDGES",
                          description="G1 / G2: which way the loft leaves side B")
    rails: IntProperty(options={'HIDDEN', 'SKIP_SAVE'}, default=0,
                       description="How many of the last sides are rails")
    cont_ra: EnumProperty(name="Rail A", items=CONT_ITEMS, default="G0",
                          description="Continuity along rail A where it is a surface edge")
    cont_rb: EnumProperty(name="Rail B", items=CONT_ITEMS, default="G0",
                          description="Continuity along rail B where it is a surface edge")
    tan_ra: FloatProperty(name="Weight A", default=1.0, min=0.05, soft_max=3.0, description="Rail A: how hard it holds")
    tan_rb: FloatProperty(name="Weight B", default=1.0, min=0.05, soft_max=3.0, description="Rail B: how hard it holds")
    corner_ease: FloatProperty(name="Ease Corners", default=0.15, min=0.0, max=0.5, subtype='FACTOR',
                               description="With two rails: G2 at the profiles eases off over this share of each "
                                           "profile toward the rails (a source whose curvature runs wild at the "
                                           "corner is not chased there: a much lighter net). 0: G2 all along")
    even_net: BoolProperty(name="Even Net", default=False,
                           description="With rails: space the CVs evenly (a cleaner, more regular net; G1 can "
                                       "need a few more CVs). Off: the net follows the boundaries' own layout")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        sel = [o for o in context.selected_objects if o.type == 'CURVE']
        if self.use_selection or (len(sel) >= 2 and not event.alt):
            sel = loft_selection_order(sel, context.active_object, self.closed)
            self.inputs = _encode_inputs([(o, "S0") for o in sel])
            self.sides = ""
            return self.execute(context)
        self._side_of, self._new_side = [], False
        self.rails = 0
        return self.start_picking(context, event)

    # --- picking: sides of one or more edges --------------------------------

    def expand_pick(self, cand):
        if self.chain == 'TANGENT':
            have = {(o.name, r) for o, r in getattr(self, "_picks", [])}
            return [c for c in tangent_chain(cand, self._cands, self.chain_angle) if (c[0].name, c[1]) not in have]
        return [cand]

    def pick_count(self):
        so = getattr(self, "_side_of", None)
        if not so:
            return len(self._picks)
        return len({g for g in so if g not in getattr(self, "_rail_groups", set())})     # profiles

    def _label(self, group):
        rails = getattr(self, "_rail_groups", set())
        so = getattr(self, "_side_of", [])
        groups = sorted(set(so) | {group})
        if group in rails:
            return f"rail {'AB'[min([g for g in groups if g in rails].index(group), 1)]}"
        return f"side {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[min([g for g in groups if g not in rails].index(group), 25)]}"

    def pick_state(self):
        so = getattr(self, "_side_of", [])
        rail = getattr(self, "_rail_mode", False)
        mode = "Tangent Chain" if self.chain == 'TANGENT' else "Single Edge"
        if not so or self._new_side:
            nxt = (so[-1] + 1) if so else 0
            if rail:
                self._rail_groups.add(nxt)
                lab = self._label(nxt)
                self._rail_groups.discard(nxt)
                return f"{lab}: pick its edges · R: back to profiles"
            return f"{self._label(nxt)}: pick its edges" + (" · R: rails" if so else "")
        side = so[-1]
        n = sum(1 for s in so if s == side)
        return f"{self._label(side)}: {n} edge{'s' if n > 1 else ''} · {mode}" + \
            (" · R: back to profiles" if rail else " · R: rails")

    def _recolour(self):
        rails = getattr(self, "_rail_groups", set())
        profs = sorted({g for g in self._side_of if g not in rails})
        draw.pick_overlay["colors"] = [3 if g in rails else profs.index(g) % 3 for g in self._side_of]

    def after_pick(self, group, event):
        so = self._side_of
        rail = getattr(self, "_rail_mode", False)
        if not so:
            side = 0
        elif self._new_side or ((so[-1] in self._rail_groups) != rail):
            side = so[-1] + 1                      # a new side (or rail / profile mode changed)
        else:
            done = len(self._picks) - len(group)
            cur = [self._cands_by_key[(o.name, r)] for (o, r), s in zip(self._picks[:done], so) if s == so[-1]]
            same = cur and _is_curve_cand(cur[0]) == _is_curve_cand(group[0])
            side = so[-1] if same and touches([c[2] for c in cur], [c[2] for c in group]) else so[-1] + 1
        so.extend([side] * len(group))
        if rail:
            self._rail_groups.add(side)
        self._new_side = False
        self._recolour()

    def after_unpick(self, n):
        del self._side_of[len(self._side_of) - n:]
        self._rail_groups &= set(self._side_of)
        self._new_side = False
        self._recolour()

    def tool_key(self, context, event):
        if getattr(event, "ctrl", False) or getattr(event, "oskey", False) or getattr(event, "alt", False):
            return False
        if event.type == 'N' and getattr(self, "_side_of", None):
            self._new_side = True
            return True
        if event.type == 'C':
            self.chain = 'SINGLE' if self.chain == 'TANGENT' else 'TANGENT'
            return True
        if event.type == 'R':
            self._rail_mode = not getattr(self, "_rail_mode", False)
            self._new_side = bool(self._side_of)
            return True
        return False

    def start_picking(self, context, event=None):
        self._side_of, self._new_side = [], False
        self._rail_groups, self._rail_mode = set(), False
        self._cands_by_key = {}
        r = super().start_picking(context, event)
        self._cands_by_key = {(c[0].name, c[1]): c for c in getattr(self, "_cands", [])}
        return r

    def picked_inputs(self, context):
        pre = super().picked_inputs(context)
        if pre:
            byref = {(c[0].name, c[1]): c for c in self._cands}
            self._side_of = group_into_sides([byref[(o.name, r)] for o, r in pre])
            if len(set(self._side_of)) < 2:
                self._side_of = list(range(len(pre)))     # one run of touching edges: each its own section
        return pre

    def _finish(self, context):
        so = getattr(self, "_side_of", [])
        rails = getattr(self, "_rail_groups", set())
        if so and len(so) == len(self._picks) and rails:
            # profiles first, then the rails (in the order picked)
            order = sorted(range(len(so)), key=lambda i: (so[i] in rails, so[i], i))
            self._picks = [self._picks[i] for i in order]
            so = [so[i] for i in order]
            self._side_of = so
        self.rails = len(rails)
        self.sides = json.dumps(side_counts(so)) if so and len(so) == len(self._picks) else ""
        return super()._finish(context)

    def _side_counts(self):
        try:
            v = json.loads(self.sides) if self.sides else []
        except ValueError:
            v = []
        return [int(x) for x in v]

    def _params(self):
        p = {k: getattr(self, k) for k in (
            "cont_a", "cont_b", "tan_a", "tan_b", "cur_a", "cur_b",
            "auto_align", "flip_b", "ruled", "closed", "samples", "edges_a", "edges_b",
            "fit", "span_degree", "span_faces")}
        counts = self._side_counts()
        if counts and (any(c > 1 for c in counts) or self.rails):
            p["sides"] = counts
        if self.rails:
            p.update({"rails": int(self.rails), "cont_ra": self.cont_ra, "cont_rb": self.cont_rb,
                      "tan_ra": self.tan_ra, "tan_rb": self.tan_rb, "corner_ease": self.corner_ease,
                      "even_net": self.even_net})
        return p

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            inputs = _decode_inputs(self.inputs)
            features.LOFT_NOTES.clear()
            obj = _make_result(context, "Loft", "loft", inputs, self._params())
        except Exception as exc:
            self.report({'ERROR'}, f"Loft failed: {exc}")
            return {'CANCELLED'}
        for note in features.LOFT_NOTES:
            self.report({'WARNING'}, note)
        _select_only(context, [obj])
        return {'FINISHED'}

    def draw(self, context):
        lay = self.layout
        try:
            refs = [r for _, r in json.loads(self.inputs or "[]")]
        except ValueError:
            refs = []
        counts = self._side_counts()
        if not counts or sum(counts) != len(refs):
            counts = [1] * len(refs)
        firsts = [sum(counts[:i]) for i in range(len(counts))]
        nr = self.rails if 0 < self.rails < len(counts) - 1 else 0
        nprof = len(counts) - nr
        two = nprof == 2
        shown = [(0, firsts[0], counts[0]), (1, firsts[nprof - 1], counts[nprof - 1])] if (two or nr) else []
        for side, i0, k in (("ab"[j], i0, k) for j, i0, k in shown):
            ref = refs[i0]
            box = lay.box()
            edge = features.is_surface_edge(ref)
            what = (("trimmed edge" if features.is_brep_edge_ref(ref) else "surface edge") if edge else "curve")
            box.label(text=f"Input {side.upper()}: " + (f"{k} {what}s" if k > 1 else what))
            col = box.column(align=True)
            col.enabled = edge
            row = col.row(align=True)
            row.prop(self, f"cont_{side}", text="")
            if not nr:
                sub = row.row(align=True)
                sub.active = getattr(self, f"cont_{side}") != "G0"
                sub.prop(self, f"edges_{side}", text="")
            row = col.row(align=True)
            row.prop(self, f"tan_{side}", text="Weight" if nr else "Tangent")
            if not nr:
                row.prop(self, f"cur_{side}", text="Curvature")
        for j in range(nr):
            i0, k = firsts[nprof + j], counts[nprof + j]
            ref = refs[i0]
            box = lay.box()
            edge = features.is_surface_edge(ref)
            what = (("trimmed edge" if features.is_brep_edge_ref(ref) else "surface edge") if edge else "curve")
            box.label(text=f"Rail {'AB'[j]}: " + (f"{k} {what}s" if k > 1 else what) +
                      ("" if nr == 2 else " (a guide: the loft passes through it)"))
            row = box.row(align=True)
            row.enabled = edge and nr == 2
            row.prop(self, f"cont_r{'ab'[j]}", text="")
            row.prop(self, f"tan_r{'ab'[j]}", text="Weight")
        if nr:
            col = lay.column()
            row = col.row()
            row.active = nr == 2 and bool({"G2", "G3"} & {self.cont_a, self.cont_b})
            row.prop(self, "corner_ease")
            col.prop(self, "even_net")
            col.prop(self, "span_faces")
            col.prop(self, "samples")
            row = col.row(align=True)
            row.prop(self, "chain", text="Pick")
            sub = row.row(align=True)
            sub.active = self.chain == 'TANGENT'
            sub.prop(self, "chain_angle", text="°")
            return
        col = lay.column()
        row = col.row()
        row.active = len(counts) >= 3
        row.prop(self, "closed")
        col.prop(self, "auto_align")
        col.prop(self, "flip_b")
        if two and not any(features.is_surface_edge(r) for r in refs):
            col.prop(self, "ruled")
        sampled = any(features.is_surface_edge(r) for r in refs) and {self.cont_a, self.cont_b} != {"G0"}
        row = col.row(align=True)
        row.active = sampled
        row.prop(self, "fit", text="")
        sub = row.row(align=True)
        sub.active = sampled and self.fit == "SPANS"
        sub.prop(self, "span_degree", text="Degree")
        row = col.row()
        row.active = sampled and self.fit == "SPANS"
        row.prop(self, "span_faces")
        col.prop(self, "samples")
        row = col.row(align=True)
        row.prop(self, "chain", text="Pick")
        sub = row.row(align=True)
        sub.active = self.chain == 'TANGENT'
        sub.prop(self, "chain_angle", text="°")


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
    accepts_picks = True
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
        ins = self.selection_inputs(context)
        if ins and not event.alt:
            self.inputs = _encode_inputs(ins)
            return go(context)
        return self.start_picking(context, event)

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            ins = self.usable_inputs(_decode_inputs(self.inputs))
        except Exception as exc:
            self.report({'ERROR'}, f"{self.bl_label} failed: {exc}")
            return {'CANCELLED'}
        made, failed = [], []
        for inp in ins:            # one curve that fails doesn't take the others with it
            try:
                made.append(_make_result(context, self.bl_label, self.op_name, [inp], self.params()))
            except Exception as exc:
                failed.append(f"{inp[0].name}: {exc}")
        if failed:
            self.report({'ERROR'} if not made else {'WARNING'}, f"{self.bl_label} failed on " + "; ".join(failed))
        if not made:
            return {'CANCELLED'}
        _select_only(context, made)
        return {'FINISHED'}

    def usable_inputs(self, ins):
        """The inputs to make results from (a tool may leave some out, saying why)."""
        return ins

    def selection_inputs(self, context):
        """The inputs a selection gives (every spline of the selected curves)."""
        return _curve_inputs(context)


def _smooth_draft_normals(P, d, closed, flip, centre=None):
    """Outward directions at samples of a smooth curve, from its tangents
    (second-order differences, also at open ends): a chord's normal at an
    end is off by half the turn of the last chord. Signed as _draft_normals."""
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T = np.gradient(P, axis=0, edge_order=2)
    n = np.cross(T, d)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
    rel = P - geom.draft_reference(P, closed, centre)
    rel -= np.outer(rel @ d, d)
    sgn = 1.0 if np.sum(n * rel) >= 0 else -1.0
    return -sgn * n if flip else sgn * n


def extrude_rows(P, d, h, draft=0.0, flip=False, symmetric=False, smooth=True, centre=None):
    """Rows of points of the extrude of samples P (world): what build_extrude
    makes, for its live preview -- the far row moved by h along d and, with a
    draft, outward by h tan(draft) as draft_extrude does. smooth=False: P are
    a polyline's corners (mitred, as the builder does a degree-1 profile)."""
    P = np.asarray(P, float)
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) or 1.0)
    if abs(draft) > 1e-9:
        closed = len(P) > 2 and np.linalg.norm(P[0] - P[-1]) < 1e-7 * (np.ptp(P, axis=0).max() or 1.0)
        normals = _smooth_draft_normals if smooth else geom._draft_normals
        nrm = normals(P[:-1] if closed else P, d, closed, flip, centre)
        if closed:
            nrm = np.vstack([nrm, nrm[:1]])
        return geom._draft_rows(P, nrm, d, h, h * np.tan(np.radians(draft)), symmetric)
    if symmetric:
        return [P - h * d, P + h * d]
    return [P, P + h * d]


class NPROTO_OT_extrude(ValueDrag, AskValue, _CurveTool, bpy.types.Operator):
    """Extrude curves (or surface edges) into surfaces. From the UI the distance
    follows the mouse with a preview; type a value, Tab for the draft, LMB to apply"""
    bl_idname = "nproto.extrude"
    bl_label = "Extrude"
    bl_options = {'REGISTER', 'UNDO'}
    op_name = "extrude"
    start_value = 0.0      # the arrow stands on the curve: nothing is extruded until it is dragged
    pick_max = 256         # a click takes one curve or edge; Shift+click keeps picking, Enter finishes

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
    chain: BoolProperty(name="Chain", default=True,
                        description="Curves and edges that touch end to end make one surface (a draft mitres "
                                    "their corners); off: one surface each")
    normal_hint: FloatVectorProperty(subtype='XYZ', size=3, default=(0.0, 0.0, 1.0), options={'HIDDEN'},
                                     description="Curve Normal of a straight curve, and the side a planar "
                                                 "curve's normal points to: the work plane's normal")

    def params(self):
        return {"distance": self.distance, "direction": self.direction, "dir_vec": list(self.dir_vec),
                "tilt": self.tilt, "tilt_azimuth": self.tilt_azimuth, "draft": self.draft,
                "draft_flip": self.draft_flip, "symmetric": self.symmetric,
                "normal_hint": [float(x) for x in self.normal_hint]}

    def invoke(self, context, event):
        from . import workplanes
        fr = workplanes.active_frame(context.scene)
        self.normal_hint = [float(x) for x in fr[3]] if fr is not None else (0.0, 0.0, 1.0)
        return super().invoke(context, event)

    def _groups(self, ins):
        """The surfaces to make: inputs that touch end to end together (Chain)."""
        return features.profile_groups(ins) if self.chain else [[x] for x in ins]

    @staticmethod
    def _centre(curves):
        """Where open pieces extruded together draft away from: the middle of all
        of them (None for one profile: it uses its own centre)."""
        if len(curves) < 2:
            return None
        P = np.vstack([geom.sample_curve(c, 33) for c in curves])
        return [float(x) for x in 0.5 * (P.min(axis=0) + P.max(axis=0))]

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            groups = self._groups(_decode_inputs(self.inputs))
            centre = self._centre([features.profile_curve(g) for g in groups])
        except Exception as exc:
            self.report({'ERROR'}, f"{self.bl_label} failed: {exc}")
            return {'CANCELLED'}
        made, failed = [], []
        for g in groups:           # one profile that fails doesn't take the others with it
            p = self.params()
            if centre is not None:
                p["draft_centre"] = centre
            try:
                made.append(_make_result(context, self.bl_label, self.op_name, g, p))
            except Exception as exc:
                failed.append(f"{g[0][0].name}: {exc}")
        if failed:
            self.report({'ERROR'} if not made else {'WARNING'}, f"{self.bl_label} failed on " + "; ".join(failed))
        if not made:
            return {'CANCELLED'}
        _select_only(context, made)
        return {'FINISHED'}

    # -- the live value stage (valuedrag.py) ----------------------------------
    drag_fields = ("distance", "draft")
    angle_fields = ("draft",)
    AXES = {"x": "X", "y": "Y", "z": "Z"}

    def drag_setup(self, context, mouse):
        self._ex = []
        from .kernel import load
        ext = load().blender_ext
        for g in self._groups(_decode_inputs(self.inputs)):
            obj, curve = g[0][0], features.profile_curve(g)
            cv = ext.curve_cvs(curve)
            if cv["degree"] == 1 and not cv.get("weights"):     # a polyline: its corners, exact
                self._ex.append((obj, curve, np.array(cv["points"], float), False))
            else:
                self._ex.append((obj, curve, geom.sample_curve(curve, 97), True))
        if not self._ex:
            return False
        self._ctr = self._centre([c for _, c, _, _ in self._ex])
        # the handles stand on the first curve: the arrow where it was clicked
        # (when it was picked by a click), else halfway along it
        obj, curve, P, smooth = self._ex[0]
        S = P if smooth else geom.sample_curve(curve, 97)
        n = len(S)
        closed = np.linalg.norm(S[0] - S[-1]) < 1e-7 * (np.ptp(S, axis=0).max() or 1.0)
        kb, self._base_at_mouse = n // 2, False
        if mouse is not None:
            best = None
            for k, q in enumerate(S):
                sq = self._screen(context, q)
                if sq is not None and (sq - mouse).length <= 40.0 and (best is None or (sq - mouse).length < best[0]):
                    best = ((sq - mouse).length, k)
            if best is not None:
                kb, self._base_at_mouse = best[1], True
        if closed:
            kd = (kb + (n - 1) // 5) % (n - 1)
        else:
            kd = kb + n // 5 if kb < n // 2 else kb - n // 5
        self._S, self._kb, self._kd, self._closed = S, kb, kd, closed
        return True

    def drag_on_input(self, context, mouse):
        return self._near_screen(context, self._S, mouse)

    def _dirs(self):
        p = self.params()
        return [np.array(features.extrude_direction(obj, curve, p)) for obj, curve, _, _ in self._ex]

    def _draft_normal(self, d):
        """Outward direction of the draft at the diamond's foot (sample kd),
        the way the preview and the builder move the top edge there."""
        obj, curve, P, smooth = self._ex[0]
        S, k, closed = self._S, self._kd, self._closed
        if smooth:
            nrm = _smooth_draft_normals(S[:-1] if closed else S, d, closed, self.draft_flip, self._ctr)
            return nrm[k % len(nrm)]
        # a polyline: the normal of the side the foot is on, signed as the corners' mitres
        cl = len(P) > 2 and np.linalg.norm(P[0] - P[-1]) < 1e-7 * (np.ptp(P, axis=0).max() or 1.0)
        mit = geom._draft_normals(P[:-1] if cl else P, d, cl, self.draft_flip, self._ctr)
        if cl:
            mit = np.vstack([mit, mit[:1]])
        q = S[k]
        j = min(range(len(P) - 1), key=lambda i: _seg_dist_np(q, P[i], P[i + 1]))
        sn = np.cross(P[j + 1] - P[j], d)
        sn /= np.linalg.norm(sn) or 1.0
        return sn if sn @ mit[j] >= 0 else -sn

    def drag_handles(self):
        d = self._dirs()[0]
        d = d / (np.linalg.norm(d) or 1.0)
        h = self.distance
        B = self._S[self._kb]
        foot = self._S[self._kd] + d * h
        arrow = {"field": "distance", "kind": "arrow", "origin": B, "dir": d, "param": h,
                 "base": B, "to_value": lambda t: t}
        if abs(h) <= 1e-9 * (float(np.ptp(self._S, axis=0).max()) or 1.0):
            return [arrow]        # no height yet: no top edge to draft (the diamond would sit on the curve)
        nd = self._draft_normal(d)
        return [arrow,
                {"field": "draft", "kind": "draft", "origin": foot, "dir": nd,
                 "param": h * np.tan(np.radians(self.draft)),
                 "to_value": lambda t, h=h: draft_angle(t, h)}]

    def drag_preview(self):
        lines, tris = [], []
        for (obj, curve, P, smooth), d in zip(self._ex, self._dirs()):
            rows = extrude_rows(P, d, self.distance, self.draft, self.draft_flip, self.symmetric, smooth, self._ctr)
            L, T = ruled_preview(rows, ribs=8 if smooth else len(P) - 1)
            lines += L
            tris.append(T)
        return {"lines": lines, "tris": np.vstack(tris)}

    def drag_check(self):
        size = max(float(np.ptp(np.vstack([P for _, _, P, _ in self._ex]), axis=0).max()), 1e-9)
        if abs(self.distance) <= 1e-6 * size:
            return "Distance 0: drag the arrow out, or type a distance"
        return ""

    def drag_key(self, context, event, ch):
        if ch in self.AXES:         # X Y Z: that world axis; again: the object's local one
            ax = self.AXES[ch]
            self.direction = ("LOCAL_" + ax) if self.direction == ax else ax
        elif ch == "n":
            self.direction = "NORMAL"
        elif ch == "f":
            self.distance = -self.distance
        elif ch == "s":
            self.symmetric = not self.symmetric
        elif ch == "d":
            self.draft_flip = not self.draft_flip
        elif ch == "c" and len(_decode_inputs(self.inputs)) > 1:
            self.chain = not self.chain
            self.drag_setup(bpy.context, None)
        else:
            return False
        return True

    def drag_handle_hints(self):
        return [hints.key("LMB drag", "arrow: distance"), hints.key("LMB drag", "diamond: draft")]

    def drag_rows(self):
        cur = {"NORMAL": "normal"}.get(self.direction, self.direction.replace("LOCAL_", "").lower())
        return [("Direction", [hints.cycle("N X Y Z", ["normal", "x", "y", "z"], cur),
                               hints.key("F", "flip"), hints.toggle("S", "symmetric", self.symmetric)]),
                ("Draft", [hints.toggle("D", "inward", self.draft_flip)]
                 + ([hints.toggle("C", "chain", self.chain)] if len(_decode_inputs(self.inputs)) > 1 else []))]

    def drag_state(self):
        names = {"NORMAL": "Curve Normal", "VECTOR": "Vector"}
        d = names.get(self.direction, self.direction.replace("LOCAL_", "Local ").replace("_", " "))
        if d in ("X", "Y", "Z"):
            d = "World " + d
        return [d] + (["Symmetric"] if self.symmetric else [])


class NPROTO_OT_revolve(ValueDrag, AskValue, _CurveTool, bpy.types.Operator):
    """Revolve curves around an axis. From the UI the angle has a handle to drag
    around the axis on a live preview; type a value, click away or Enter to make it"""
    bl_idname = "nproto.revolve"
    bl_label = "Revolve"
    bl_options = {'REGISTER', 'UNDO'}
    op_name = "revolve"

    angle: FloatProperty(name="Angle", default=360.0, min=-360.0, max=360.0)
    axis: EnumProperty(name="Axis", items=REVOLVE_AXIS_ITEMS, default="LOCAL_Z")
    pivot: EnumProperty(name="Pivot", items=[("OBJECT", "Object Origin", ""), ("WORLD", "World Origin", "")],
                        description="Where a world or local axis passes through (a picked line has its own place)")
    axis_point: FloatVectorProperty(name="Axis Point", size=3, subtype='XYZ', unit='LENGTH',
                                    description="Picked Line: a point the axis passes through (world)")
    axis_dir: FloatVectorProperty(name="Axis Direction", size=3, subtype='XYZ', default=(0.0, 0.0, 1.0),
                                  description="Picked Line: the axis' direction (world)")
    seam_angle: FloatProperty(name="Seam Angle", default=0.0, soft_min=-180.0, soft_max=180.0,
                              description="Full revolves: where the surface's seam sits around the axis. Turn it "
                                          "away from openings you will trim -- fillets cannot cross a seam")

    def params(self):
        return {"angle": self.angle, "axis": self.axis, "pivot": self.pivot, "seam_angle": self.seam_angle,
                "axis_point": list(self.axis_point), "axis_dir": list(self.axis_dir)}

    # -- the live value stage (valuedrag.py) ----------------------------------
    drag_fields = ("angle",)
    angle_fields = ("angle",)
    ask_dialog = False          # Value Input = Dialog: revolve at once, as it always did
    AXES = {"x": "X", "y": "Y", "z": "Z"}

    # nothing selected: click the profile, then the axis (Enter after the profile keeps the axis setting)
    pick_max = 2
    finish_on_min = False

    def pick_state(self):
        if not self._picks:
            return "click the profile"
        return "click the axis: a line or straight edge  ·  Enter: " + self._axis_name()

    def _axis_name(self):
        if self.axis == "PICKED":
            return "the line picked last time"
        return (self.axis.replace("LOCAL_", "Local ") if self.axis.startswith("LOCAL_") else "World " + self.axis)

    def _finish(self, context):
        if len(self._picks) == 2:           # the second click was the axis
            pts = draw.pick_overlay["picked"][1]
            p0, p1 = np.asarray(pts[0], float), np.asarray(pts[-1], float)
            if np.linalg.norm(p1 - p0) > 1e-9:
                self.axis_point, self.axis_dir = tuple(p0), tuple((p1 - p0) / np.linalg.norm(p1 - p0))
                self.axis = "PICKED"
            self._picks = self._picks[:1]
        return super()._finish(context)

    def selection_inputs(self, context):
        """Profile(s) and axis selected, the axis last (active): a straight
        line made active is the axis, from its start to its end; the other
        selected curves are the profiles. (Only a straight line: two
        profiles revolved about World Z stay two profiles.)"""
        ins = _curve_inputs(context)
        act = context.view_layer.objects.active
        curves = {o for o, _ in ins}
        if act is None or act.type != 'CURVE' or act not in curves or len(curves) < 2 or len(act.data.splines) != 1:
            return ins
        line = _straight_line(features.input_curve(act, "S0"))
        if line is None:
            return ins
        self.axis_point, self.axis_dir = tuple(line[0]), tuple(line[1])
        self.axis = "PICKED"
        return [(o, r) for o, r in ins if o != act]

    def usable_inputs(self, ins):
        """Leave out a curve lying on the axis (the line picked for it, still
        selected): turned about itself it sweeps no surface, and the kernel
        refuses to make a face of it."""
        keep, skip = [], []
        for obj, ref in ins:
            S = geom.sample_curve(features.input_curve(obj, ref), 33)
            (skip if _on_axis(S, *features.revolve_axis(obj, self.params())) else keep).append((obj, ref))
        if skip:
            self.report({'INFO'} if keep else {'ERROR'},
                        ", ".join(o.name for o, _ in skip) + (" lies" if len(skip) == 1 else " lie")
                        + " on the axis: not revolved" + ("" if keep else " -- pick a profile off the axis"))
        return keep

    def drag_setup(self, context, mouse):
        self._rv = []
        for obj, ref in _decode_inputs(self.inputs):
            curve = features.input_curve(obj, ref)
            self._rv.append((obj, geom.sample_curve(curve, 97), ref))
        self._rv_all = list(self._rv)
        self._drop_on_axis()
        if not self._rv:
            return False
        self._mouse0 = mouse
        self._ctx = context
        self._place(context)
        return True

    def _axis(self):
        O, z = features.revolve_axis(self._rv_all[0][0], self.params())
        return np.array(O, float), np.array(z, float)

    def _drop_on_axis(self):
        """The previewed profiles: those off the axis (as usable_inputs)."""
        keep = [r for r in self._rv_all if not _on_axis(r[1], *features.revolve_axis(r[0], self.params()))]
        self._rv = keep or self._rv_all[:1]

    def _place(self, context):
        """The handle's foot: the profile point clicked (when it was picked
        by a click), else the one farthest from the axis."""
        O, z = self._axis()
        S = self._rv[0][1]
        rel = S - O
        rad = np.linalg.norm(rel - np.outer(rel @ z, z), axis=1)
        k = int(np.argmax(rad))
        self._base_at_mouse = False
        if self._mouse0 is not None:
            best = None
            for i, q in enumerate(S):
                sq = self._screen(context, q)
                if sq is not None and rad[i] > 1e-6 * (rad.max() or 1.0):
                    dd = (sq - self._mouse0).length
                    if dd <= 40.0 and (best is None or dd < best[0]):
                        best = (dd, i)
            if best is not None:
                k, self._base_at_mouse = best[1], True
        self._k = k

    def drag_handles(self):
        O, z = self._axis()
        P = self._rv[0][1][self._k]
        C = O + z * float((P - O) @ z)
        x = P - C
        rho = float(np.linalg.norm(x))
        if rho < 1e-12:
            return []
        x /= rho
        y = np.cross(z, x)
        a = self.angle
        n = max(2, int(abs(a) / 3.0) + 1)
        path = np.vstack([rotate_points(P[None], O, z, a * i / (n - 1)) for i in range(n)])
        back = a - math.copysign(min(6.0, abs(a) * 0.5), a) if abs(a) > 1e-9 else 0.0
        return [{"field": "angle", "kind": "arrow", "pos": rotate_points(P[None], O, z, a)[0], "param": a,
                 "origin": C, "dir": None, "path": path, "head_from": rotate_points(P[None], O, z, back)[0],
                 "pick": lambda o, r, prev: angle_pick(C, x, y, z, o, r, prev),
                 "to_value": lambda t: t, "snap": (5.0, 1.0)}]

    def drag_preview(self):
        O, z = self._axis()
        a = self.angle
        m = max(2, int(math.ceil(abs(a) / 7.5)) + 1)
        lines, tris = [], []
        for obj, S, _ in self._rv:
            rows = [rotate_points(S, O, z, a * i / (m - 1)) for i in range(m)]
            L, T = ruled_preview(rows, ribs=8, row_every=3)
            lines += L
            tris.append(T)
        # the axis, a little longer than the profile reaches along it
        h = np.concatenate([(S - O) @ z for _, S, _ in self._rv])
        size = float(np.ptp(np.vstack([S for _, S, _ in self._rv]), axis=0).max()) or 1.0
        pad = max(0.15 * float(np.ptp(h)), 0.5 * size)     # visible even when the profile hardly runs along it
        axis = np.array([O + z * (h.min() - pad), O + z * (h.max() + pad)])
        return {"lines": lines, "tris": np.vstack(tris), "guides": [(axis, (0.55, 0.85, 1.0, 0.9))]}

    def drag_check(self):
        if abs(self.angle) < 0.05:
            return "Angle 0: drag the handle around the axis, or type an angle"
        if all(_on_axis(r[1], *features.revolve_axis(r[0], self.params())) for r in self._rv):
            return "the profile lies on the axis: pick another axis (A, X Y Z)"
        return ""

    def drag_key(self, context, event, ch):
        if ch in self.AXES:         # X Y Z: that world axis; again: the object's local one
            ax = self.AXES[ch]
            self.axis = ("LOCAL_" + ax) if self.axis == ax else ax
        elif ch == "a":
            self._start_axis_pick(context)
            return True
        elif ch == "o":
            self.pivot = "WORLD" if self.pivot == "OBJECT" else "OBJECT"
        elif ch == "f":
            self.angle = -self.angle
        else:
            return False
        self._drop_on_axis()
        self._place(context)
        return True

    def drag_handle_hints(self):
        return [hints.key("LMB drag", "the arrow around the axis: angle")]

    def drag_rows(self):
        picked = self.axis == "PICKED"
        cur = None if picked else self.axis.replace("LOCAL_", "").lower()
        return [("Axis", [hints.cycle("X Y Z", ["x", "y", "z"], cur),
                          hints.key("A", "pick a line / edge", on=True if picked else None),
                          hints.cycle("O", ["object", "world"], None if picked else self.pivot.lower(), "pivot"),
                          hints.key("F", "other way"), hints.key("Ctrl", "5° steps")])]

    def drag_state(self):
        if self.axis == "PICKED":
            return ["Axis: picked line"]
        ax = self.axis.replace("LOCAL_", "Local ") if self.axis.startswith("LOCAL_") else "World " + self.axis
        return [f"Axis {ax}", "Pivot " + ("Object Origin" if self.pivot == "OBJECT" else "World Origin")]

    # -- A: pick the axis (a line, an open curve, a straight edge) ------------
    def _start_axis_pick(self, context):
        # a selected curve may be the axis too (select the profile and the line, then A)
        self._axis_cands = [c for c in _pick_candidates(context)
                            if np.linalg.norm(np.asarray(c[2][-1]) - np.asarray(c[2][0])) > 1e-9]
        self._axis_hover = None
        self._picking_axis = True
        draw.pick_overlay["hover"] = None
        self._axis_hints(context)

    def _axis_hints(self, context):
        hints.show(context, self.bl_label, ["click a line, open curve or edge for the axis (start to end)"],
                   [("Axis", [hints.key("LMB", "use it"), hints.key("Esc", "back, axis unchanged")])],
                   [("MOUSE_LMB", "Pick the axis"), ("EVENT_ESC", "Back")])

    def _end_axis_pick(self, context):
        self._picking_axis = False
        draw.pick_overlay["hover"] = None

    def drag_modal_hook(self, context, event):
        """While A is picking the axis: hover and click a candidate."""
        if not getattr(self, "_picking_axis", False):
            return None
        et = event.type
        if et in {'LEFTMOUSE', 'RIGHTMOUSE', 'MOUSEMOVE'} and not PickMixin._in_view(context, event):
            return {'PASS_THROUGH'}
        if et == 'MOUSEMOVE':
            m = self._mouse(context, event)
            ties = _near_candidates(context, _RegionMouse(m), self._axis_cands)
            self._axis_hover = ties[0] if ties else None
            draw.pick_overlay["hover"] = self._axis_hover[2] if self._axis_hover else None
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.value != 'PRESS':
            return {'RUNNING_MODAL'}
        if et in {'ESC', 'RIGHTMOUSE'}:
            self._end_axis_pick(context)
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        if et == 'LEFTMOUSE' and self._axis_hover is not None:
            key = (self._axis_hover[0].name, self._axis_hover[1])
            ins = [(o, r) for o, r in _decode_inputs(self.inputs)]
            rest = [(o, r) for o, r in ins if (o.name, r) != key]
            if not rest:
                self._msg = "that is the profile: pick another line for the axis"
                return {'RUNNING_MODAL'}
            if len(rest) < len(ins):            # the picked line was selected with the profile: no longer one
                self.inputs = _encode_inputs(rest)
                self._rv_all = [r for r in self._rv_all if (r[0].name, r[2]) != key]
            pts = self._axis_hover[2]
            p0, p1 = np.asarray(pts[0], float), np.asarray(pts[-1], float)
            self.axis_point = tuple(p0)
            self.axis_dir = tuple((p1 - p0) / np.linalg.norm(p1 - p0))
            self.axis = "PICKED"
            self._end_axis_pick(context)
            self._drop_on_axis()
            self._place(context)
            self._drag_refresh(context)
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}


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

    accepts_picks = True
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
        return self.start_picking(context, event)

    def _status(self, context):
        super()._status(context)
        if self._picks:
            hints.show(context, "Planar", [f"{len(self._picks)} edge(s) / curve(s) picked, profile not closed yet",
                                           "keep picking (Enter fills what you have)"], self.pick_rows(), MOUSE_PICK)

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


def offset_frame(obj, cos_deg=1.0):
    """The display mesh of obj, set up to be offset: (world vertices V (n,3),
    their offset directions W (n,3), triangles (m,3)). V + d W is the offset
    by d: W is the normal where the surface is smooth; where faces meet at an
    edge or corner, W moves the point onto every face's offset at once (the
    least-norm w with n_i . w = 1 for each face normal n_i there), so offset
    faces meet with extended corners, as the real offset makes them."""
    me = obj.data
    nv, nl = len(me.vertices), len(me.loops)
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    M = np.array(geom.world_matrix(obj), float)
    V = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    lv = np.empty(nl, np.int64)
    me.loops.foreach_get("vertex_index", lv)
    cn = np.empty(nl * 3)
    me.corner_normals.foreach_get("vector", cn)
    Nm = np.linalg.inv(M[:3, :3]).T                     # normals go by the inverse transpose
    N = cn.reshape(-1, 3) @ Nm.T
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-300
    mean = np.zeros((nv, 3))
    np.add.at(mean, lv, N)
    W = mean / (np.linalg.norm(mean, axis=1, keepdims=True) + 1e-300)
    sharp = np.zeros(nv, bool)
    np.logical_or.at(sharp, lv, np.sum(N * W[lv], axis=1) < np.cos(np.radians(cos_deg)))
    if sharp.any():
        order = np.argsort(lv, kind="stable")
        starts = np.searchsorted(lv[order], np.arange(nv))
        ends = np.searchsorted(lv[order], np.arange(nv), side="right")
        cmin = np.cos(np.radians(cos_deg))
        for v in np.nonzero(sharp)[0]:
            groups = []
            for n in N[order[starts[v]:ends[v]]]:
                if not any(n @ g > cmin for g in groups):
                    groups.append(n)
            A = np.array(groups)
            W[v] = np.linalg.lstsq(A, np.ones(len(A)), rcond=None)[0]     # least norm: on every offset face
    me.calc_loop_triangles()
    T = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    # one plain surface is offset along ITS normal (du x dv), which the shaded
    # face may show the other way (a sphere's points inward): follow it
    try:
        if store.is_natural(obj) and store.get_brep(obj).face_count() == 1:
            surf = store.get_surfaces(obj)[0]
            (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
            D = surf.derivatives(0.5 * (u0 + u1), 0.5 * (v0 + v1))
            p = M[:3, :3] @ np.array(D[0], float) + M[:3, 3]
            ns = Nm @ np.cross(np.array(D[1], float), np.array(D[2], float))
            k = int(np.argmin(np.linalg.norm(V - p, axis=1)))
            if ns @ W[k] < 0:
                W = -W
    except Exception:
        pass
    return V, W, T.reshape(-1, 3), sharp


def _directions_at(bvh, V, W, T, P):
    """Offset directions at points P on the mesh (V, W, T as offset_frame
    gives them), blended over the nearest triangle: a point on an edge takes
    the directions of the edge's own vertices, not those of a vertex inside
    the face that happens to be nearer (a curved sheet turns in between)."""
    out = np.empty((len(P), 3))
    for i, p in enumerate(P):
        loc, _, f, _ = bvh.find_nearest(Vector(p))
        if f is None:
            out[i] = W[int(np.argmin(np.linalg.norm(V - p, axis=1)))]
            continue
        a, b, c = V[T[f]]
        q = np.array(loc, float)
        v0, v1, v2 = b - a, c - a, q - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        den = d00 * d11 - d01 * d01
        if den <= 0.0:
            out[i] = W[T[f][0]]
            continue
        y = (d11 * (v2 @ v0) - d01 * (v2 @ v1)) / den
        z = (d00 * (v2 @ v1) - d01 * (v2 @ v0)) / den
        out[i] = (1.0 - y - z) * W[T[f][0]] + y * W[T[f][1]] + z * W[T[f][2]]
    return out


def offset_frames(context):
    """The selected NURBS objects set up for an offset preview (the active one
    first): [(obj, V, W, T, sharp, edges)] with V, W, T, sharp as offset_frame
    gives them and edges [(world polyline E, its offset directions, open: on
    one face only)], and the vertex of the first one the arrow stands on (a
    smooth spot near its middle)."""
    from mathutils.bvhtree import BVHTree
    objs = [o for o in context.selected_objects if store.is_nurbs(o)]
    act = context.view_layer.objects.active
    objs.sort(key=lambda o: o != act)               # the handle stands on the active one
    out = []
    for o in objs:
        V, W, T, sharp = offset_frame(o)
        if not len(V):
            continue
        bvh = BVHTree.FromPolygons([Vector(v) for v in V], T.tolist())
        M = geom.world_matrix(o)
        shared = set(store.shared_edges(o))
        edges = []
        for k, pl in enumerate(store.display(o).get("edges", [])):
            if len(pl) < 2:
                continue
            E = np.array([M @ Vector(p) for p in pl], float)
            edges.append((E, _directions_at(bvh, V, W, T, E), k not in shared))
        out.append((o, V, W, T, sharp, edges))
    if not out:
        return out, 0
    _, V, W, T, sharp, _ = out[0]
    cand = np.nonzero(~sharp)[0]
    if not len(cand):
        cand = np.arange(len(V))
    c = V.mean(axis=0)
    return out, int(cand[np.argmin(np.linalg.norm(V[cand] - c, axis=1))])


class NPROTO_OT_offset(ValueDrag, AskValue, bpy.types.Operator):
    """Offset NURBS surfaces along their normal. From the UI an arrow on the
    surface shows which side it goes; drag it (through the surface for the
    other side), type a value, click away or Enter to make it"""
    bl_idname = "nproto.offset"
    bl_label = "Offset"
    bl_options = {'REGISTER', 'UNDO'}

    distance: FloatProperty(name="Distance", default=0.1, unit='LENGTH',
                            description="Along the normals (negative: the other side)")

    # -- the live value stage (valuedrag.py) ----------------------------------
    MAX_FILL = 60000            # more triangles than this: the preview draws only the edges
    start_value = 0.0           # the arrow on the surface: no offset until it is dragged (L: the last distance)

    def drag_setup(self, context, mouse):
        self._of, self._hv = offset_frames(context)
        return bool(self._of)

    def drag_handles(self):
        o, V, W, T, sharp, _ = self._of[0]
        B = V[self._hv]
        n = W[self._hv] / (np.linalg.norm(W[self._hv]) or 1.0)
        return [{"field": "distance", "kind": "arrow", "origin": B, "dir": n, "param": self.distance,
                 "base": B, "to_value": lambda t: t}]

    def drag_preview(self):
        d = self.distance
        lines, tris = [], []
        total = sum(len(T) for _, _, _, T, _, _ in self._of)
        for o, V, W, T, sharp, edges in self._of:
            lines += [E + d * We for E, We, _ in edges]
            if total <= self.MAX_FILL:
                tris.append((V + d * W)[T].reshape(-1, 3))
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3))}

    def drag_key(self, context, event, ch):
        if ch == "f":
            self.distance = -self.distance
            return True
        return False

    def drag_rows(self):
        return [("Side", [hints.key("F", "the other side"),
                          hints.note("or drag the arrow through the surface")])]

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


class NPROTO_OT_thicken(ValueDrag, AskValue, bpy.types.Operator):
    """Thicken surfaces (single or joined) into closed solids. From the UI the
    solid shows as a preview with an arrow on the surface for the thickness;
    drag it (through the surface for the other side), type a value, click away
    or Enter to make it"""
    bl_idname = "nproto.thicken"
    bl_label = "Thicken"
    bl_options = {'REGISTER', 'UNDO'}

    distance: FloatProperty(name="Thickness", default=0.01, unit='LENGTH',
                            description="Wall thickness along the normals (negative: the other side)")
    symmetric: BoolProperty(name="Symmetric", default=False, description="Half the thickness on each side")

    # -- the live value stage (valuedrag.py) ----------------------------------
    MAX_FILL = 60000            # more triangles than this: the preview draws only the edges
    fit_view = "shrink"         # a wall: the last thickness, made smaller only if it runs off the view
    start_value = 0.0           # the arrow on the surface: nothing until it is dragged (L: the last thickness)

    def drag_setup(self, context, mouse):
        self._of, self._hv = offset_frames(context)
        return bool(self._of)

    def _sides(self):
        """The offsets of the two skins: the sheet and its offset, or half
        the thickness each way."""
        d = self.distance
        return (-0.5 * d, 0.5 * d) if self.symmetric else (0.0, d)

    def drag_handles(self):
        o, V, W, T, sharp, _ = self._of[0]
        B = V[self._hv]
        n = W[self._hv] / (np.linalg.norm(W[self._hv]) or 1.0)
        a, b = self._sides()
        k = 2.0 if self.symmetric else 1.0             # symmetric: the arrow tip is half the thickness out
        return [{"field": "distance", "kind": "arrow", "origin": B, "dir": n, "param": b,
                 "base": B + a * n, "to_value": lambda t, k=k: k * t}]

    def drag_preview(self):
        a, b = self._sides()
        skins = [s for s in (a, b) if s != 0.0]        # the sheet itself is drawn already
        lines, tris = [], []
        total = sum(len(T) for _, _, _, T, _, _ in self._of) * len(skins)
        for o, V, W, T, sharp, edges in self._of:
            for E, We, open_ in edges:
                lines += [E + s * We for s in skins]
                if open_:                              # a side wall along it
                    L, Tw = ruled_preview([E + a * We, E + b * We], ribs=6)
                    lines += L[2:]
                    tris.append(Tw)
            if total <= self.MAX_FILL:
                tris += [(V + s * W)[T].reshape(-1, 3) for s in skins]
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3))}

    def drag_check(self):
        if abs(self.distance) <= 0.0:
            return "Thickness 0: drag the arrow out, or type a thickness"
        return ""

    def drag_key(self, context, event, ch):
        if ch == "f":
            self.distance = -self.distance
        elif ch == "s":
            self.symmetric = not self.symmetric
        else:
            return False
        return True

    def drag_rows(self):
        return [("Side", [hints.key("F", "the other side"), hints.toggle("S", "symmetric", self.symmetric),
                          hints.note("or drag the arrow through the surface")])]

    def drag_state(self):
        return ["Symmetric"] if self.symmetric else []

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
        return self.start_picking(context, event)

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
        return self.start_picking(context, event)

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
