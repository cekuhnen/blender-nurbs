"""Curve selection in Edit Curve mode: CV / Hull / Curve, like the mesh's
Vertex / Edge / Face.

Blender's curve Edit Mode picks only control points (L picks a whole curve,
but only near one of its CVs). A background handler adds what is missing,
with the select and transform tools active:

* CV (default): a click near a CV is Blender's own. A click on a hull line
  away from the CVs selects that span's two CVs; a click on the drawn curve
  selects the whole curve (the nearer of the two wins). Double-click a CV:
  its whole curve. Box / lasso stay Blender's.
* Hull: a click picks the nearest hull span (its two CVs); on a Bezier curve
  the piece between two points. A click on nothing deselects.
* Curve: a click on a CV, hull line or the curve picks the whole curve. Box,
  lasso and circle select whole curves: any curve the box touches (a CV in
  it, or its line through it) comes in completely; a deselecting box takes
  whole curves out.

Shift+click adds or removes; what a click would pick lights up under the
mouse. The picks are the CVs' own select flags (edit-mode undo, the
transform tools and everything else see them)."""

import bpy
import numpy as np
from bpy.app.handlers import persistent

CV_PX = 12.0          # a CV this near the mouse is the CV's (Blender's own pick)
LINE_PX = 8.0         # a hull line / curve this near is picked
COL_HOVER = (1.0, 0.95, 0.2, 1.0)
SELECT_TOOLS = {"builtin.select", "builtin.select_box", "builtin.select_circle", "builtin.select_lasso",
                "builtin.move", "builtin.rotate", "builtin.scale", "builtin.transform"}
SELECT_OPS = {"VIEW3D_OT_select", "VIEW3D_OT_select_box", "VIEW3D_OT_select_lasso", "VIEW3D_OT_select_circle"}

RUN = {"on": False, "last": None, "where": None, "press": None}
HOVER = {"key": None, "lines": None}     # key: (object name, spline, kind, points); lines: world polylines
SNAP = {}                                # object name -> per-spline select flags (last seen)
LAST_OP = {"key": None}
BUSY = {"on": False}


def mode(context=None):
    """"CV", "HULL" or "CURVE"."""
    v = getattr((context or bpy.context).window_manager, "nproto_view", None)
    return getattr(v, "curve_sel_mode", "CV") if v is not None else "CV"


def _objs(context):
    return [o for o in (getattr(context, "objects_in_mode", None) or [])
            if o.type == 'CURVE' and o.visible_get()]


def _pts(s):
    return s.bezier_points if s.type == 'BEZIER' else s.points


# --------------------------------------------------------------------------
# select flags
# --------------------------------------------------------------------------

def _is_sel(s, p):
    if s.type == 'BEZIER':
        return p.select_control_point or p.select_left_handle or p.select_right_handle
    return p.select


def _set_sel(s, p, on):
    if p.hide:
        return
    if s.type == 'BEZIER':
        p.select_control_point = p.select_left_handle = p.select_right_handle = on
    else:
        p.select = on


def flags(ob):
    return tuple(tuple(bool(_is_sel(s, p)) for p in _pts(s)) for s in ob.data.splines)


def restore(ob, fl):
    """Put select flags taken by flags() back (same layout only)."""
    spl = ob.data.splines
    if fl is None or len(fl) != len(spl):
        return
    for s, f in zip(spl, fl):
        P = _pts(s)
        if len(P) != len(f):
            continue
        for p, on in zip(P, f):
            if bool(_is_sel(s, p)) != on:
                _set_sel(s, p, on)


def deselect_all(objs):
    for ob in objs:
        for s in ob.data.splines:
            for p in _pts(s):
                if _is_sel(s, p):
                    _set_sel(s, p, False)


def apply(context, hit, extend, before=None):
    """Select what a click hit: hit = (object, spline, kind, point indices) or
    None. extend: Shift -- adds, or removes when all of it was selected
    (judged on `before`, the flags when the button went down)."""
    objs = _objs(context)
    if before:
        for ob in objs:
            restore(ob, before.get(ob.name))
    if hit is None:
        if not extend:
            deselect_all(objs)
        _tag(objs)
        return
    ob, si, _, idx = hit
    s = ob.data.splines[si]
    P = _pts(s)
    if extend:
        was = (before or {}).get(ob.name)
        fl = was[si] if was is not None and si < len(was) and len(was[si]) == len(P) else flags(ob)[si]
        on = not all(fl[k] for k in idx if not P[k].hide)
    else:
        deselect_all(objs)
        on = True
    for k in idx:
        _set_sel(s, P[k], on)
    _tag(objs)


def _tag(objs):
    for ob in objs:
        ob.data.update_tag()
        SNAP[ob.name] = flags(ob)


# --------------------------------------------------------------------------
# what is under the mouse
# --------------------------------------------------------------------------

def _project(context, W):
    """Region coordinates (n, 2) of world points (n, 3); rows behind the eye NaN."""
    region, rv3d = context.region, context.region_data
    W = np.asarray(W, float).reshape(-1, 3)
    h = np.c_[W, np.ones(len(W))] @ np.array(rv3d.perspective_matrix).T
    s = np.full((len(W), 2), np.nan)
    ok = h[:, 3] > 1e-9
    s[ok, 0] = (h[ok, 0] / h[ok, 3] * 0.5 + 0.5) * region.width
    s[ok, 1] = (h[ok, 1] / h[ok, 3] * 0.5 + 0.5) * region.height
    return s


def _dist_poly2(S, m):
    """Pixel distance from m to polyline S (n, 2)."""
    if len(S) < 2:
        return np.inf
    a, b = S[:-1], S[1:]
    ok = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    if not ok.any():
        return np.inf
    a, b = a[ok], b[ok]
    ab = b - a
    t = np.clip(((m - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - m, axis=1).min())


def _co(M, p):
    c = p.co
    return np.array(M @ (c.to_3d() if len(c) == 4 else c), float)


def _bezier_piece(M, p0, p1, n=24):
    a, b, c, d = (np.array(M @ v, float) for v in (p0.co, p0.handle_right, p1.handle_left, p1.co))
    t = np.linspace(0.0, 1.0, n)[:, None]
    u = 1.0 - t
    return u ** 3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t ** 3 * d


def spans(ob, si):
    """[(point indices, world polyline)] of spline si's hull spans: CV k to
    k + 1 (a Bezier curve: the piece between its points), the closing span
    of a cyclic one too."""
    s = ob.data.splines[si]
    P = _pts(s)
    n = len(P)
    M = ob.matrix_world
    pairs = [(k, k + 1) for k in range(n - 1)]
    if s.use_cyclic_u and n > 2:
        pairs.append((n - 1, 0))
    out = []
    for i, j in pairs:
        if P[i].hide or P[j].hide:
            continue
        if s.type == 'BEZIER':
            out.append(((i, j), _bezier_piece(M, P[i], P[j])))
        else:
            out.append(((i, j), np.array([_co(M, P[i]), _co(M, P[j])])))
    return out


def curve_poly(ob, si):
    """The drawn curve of spline si, a world polyline."""
    from . import subsel
    polys = subsel._curve_polys(ob)
    return np.asarray(polys[si], float) if si < len(polys) else np.zeros((0, 3))


def _visible(s):
    return not getattr(s, "hide", False) and any(not p.hide for p in _pts(s))


def _all_points(s):
    return tuple(k for k, p in enumerate(_pts(s)) if not p.hide)


def _nearest_cv(context, objs, m):
    best = None
    for ob in objs:
        M = ob.matrix_world
        for si, s in enumerate(ob.data.splines):
            if not _visible(s):
                continue
            P = _pts(s)
            W = []
            for k, p in enumerate(P):
                if p.hide:
                    continue
                W.append((k, _co(M, p)))
                if s.type == 'BEZIER':
                    W += [(k, np.array(M @ p.handle_left, float)), (k, np.array(M @ p.handle_right, float))]
            if not W:
                continue
            S = _project(context, [w for _, w in W])
            d = np.linalg.norm(S - m, axis=1)
            d[np.isnan(d)] = np.inf
            j = int(np.argmin(d))
            if d[j] <= CV_PX and (best is None or d[j] < best[0]):
                best = (float(d[j]), ob, si, W[j][0])
    return best


def _nearest_span(context, objs, m, px):
    best = None
    for ob in objs:
        for si, s in enumerate(ob.data.splines):
            if not _visible(s):
                continue
            for idx, W in spans(ob, si):
                d = _dist_poly2(_project(context, W), m)
                if d <= px and (best is None or d < best[0]):
                    best = (d, ob, si, idx, W)
    return best


def _nearest_curve(context, objs, m, px):
    best = None
    for ob in objs:
        for si, s in enumerate(ob.data.splines):
            if not _visible(s):
                continue
            W = curve_poly(ob, si)
            d = _dist_poly2(_project(context, W), m)
            if d <= px and (best is None or d < best[0]):
                best = (d, ob, si, W)
    return best


def hit_at(context, xy, md=None):
    """What a click at region point xy picks in mode md:
    (object, spline, kind, point indices, world polylines to light up) with
    kind "CV" (a CV mode click near a CV: Blender's own), "SPAN" or "CURVE";
    None for nothing."""
    md = md or mode(context)
    objs = _objs(context)
    if not objs:
        return None
    m = np.array(xy, float)
    if md == "HULL":
        sp = _nearest_span(context, objs, m, max(CV_PX, LINE_PX))
        return None if sp is None else (sp[1], sp[2], "SPAN", sp[3], [sp[4]])
    cv = _nearest_cv(context, objs, m)
    if md == "CV" and cv is not None:
        return (cv[1], cv[2], "CV", (cv[3],), [])
    sp = _nearest_span(context, objs, m, LINE_PX)
    cu = _nearest_curve(context, objs, m, LINE_PX)
    if md == "CURVE":
        cand = [c for c in (cv and (cv[0], cv[1], cv[2]), sp and (sp[0], sp[1], sp[2]),
                            cu and (cu[0], cu[1], cu[2])) if c]
        if not cand:
            return None
        _, ob, si = min(cand, key=lambda c: c[0])
        s = ob.data.splines[si]
        return (ob, si, "CURVE", _all_points(s), [curve_poly(ob, si)])
    # CV mode away from the CVs: the hull span or the whole curve, the nearer
    # (a curve on its own hull line -- a straight piece -- counts as the curve)
    if cu is not None and (sp is None or cu[0] <= sp[0] + 1.0):
        s = cu[1].data.splines[cu[2]]
        return (cu[1], cu[2], "CURVE", _all_points(s), [cu[3]])
    if sp is not None:
        return (sp[1], sp[2], "SPAN", sp[3], [sp[4]])
    return None


# --------------------------------------------------------------------------
# box / lasso / circle in Curve mode: whole curves
# --------------------------------------------------------------------------

def _rect_touch(S, x0, x1, y0, y1):
    """Polyline S (n, 2) has a point in the rectangle or a segment through it."""
    S = S[~np.isnan(S).any(axis=1)]
    if not len(S):
        return False
    if ((S[:, 0] >= x0) & (S[:, 0] <= x1) & (S[:, 1] >= y0) & (S[:, 1] <= y1)).any():
        return True
    for a, b in zip(S[:-1], S[1:]):           # Liang-Barsky clip
        d = b - a
        t0, t1, ok = 0.0, 1.0, True
        for p, q in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
            if abs(p) < 1e-12:
                if q < 0:
                    ok = False
                    break
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                ok = False
                break
        if ok:
            return True
    return False


def _in_poly(S, L):
    """Which points S (n, 2) lie inside polygon L (m, 2) (even-odd)."""
    inside = np.zeros(len(S), bool)
    x, y = S[:, 0], S[:, 1]
    for (ax, ay), (bx, by) in zip(L, np.roll(L, -1, axis=0)):
        c = (ay > y) != (by > y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xc = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= c & (x < xc)
    return inside


def _touch_test(op):
    """A function (polyline (n, 2) -> touched) for the box / lasso / circle
    operator op, or None."""
    name = op.bl_idname
    p = op.properties
    try:
        if name == "VIEW3D_OT_select_box":
            x0, x1 = sorted((p.xmin, p.xmax))
            y0, y1 = sorted((p.ymin, p.ymax))
            return lambda S: _rect_touch(S, x0, x1, y0, y1)
        if name == "VIEW3D_OT_select_lasso":
            L = np.array([tuple(q.loc) for q in p.path], float)
            if len(L) < 3:
                return None
            return lambda S: bool(_in_poly(S[~np.isnan(S).any(axis=1)], L).any())
        if name == "VIEW3D_OT_select_circle":
            c, r = np.array((p.x, p.y), float), float(p.radius)
            return lambda S: _dist_poly2(S, c) <= r
    except Exception:
        return None
    return None


def _op_mode(op):
    try:
        return op.properties.mode
    except Exception:
        try:
            return "ADD" if op.properties.extend else ("SUB" if op.properties.deselect else "SET")
        except Exception:
            return "SET"


def _op_key(op):
    if op is None:
        return None
    p = op.properties
    vals = []
    for k in ("xmin", "xmax", "ymin", "ymax", "x", "y", "radius", "mode", "location"):
        try:
            v = getattr(p, k)
            vals.append(tuple(v) if hasattr(v, "__len__") and not isinstance(v, str) else v)
        except Exception:
            pass
    return (op.bl_idname, op.as_pointer(), tuple(vals))


def _whole_curves(context, op):
    """After a select operator in Curve mode: every curve it changed (or whose
    line the box / lasso crossed) goes in or out completely."""
    md = _op_mode(op)
    test = None
    where = RUN.get("where")
    if where is not None and op.bl_idname != "VIEW3D_OT_select":
        test = _touch_test(op)
    changed = []
    for ob in _objs(context):
        now = flags(ob)
        before = SNAP.get(ob.name)
        if before is None or len(before) != len(now) or any(len(a) != len(b) for a, b in zip(before, now)):
            before = None
        s_all = ob.data.splines
        for si, f in enumerate(now):
            s = s_all[si]
            if not _visible(s):
                continue
            hid = [p.hide for p in _pts(s)]
            vis = [x for x, h in zip(f, hid) if not h]
            b = [x for x, h in zip(before[si], hid) if not h] if before is not None else vis
            gained = any(n and not o for n, o in zip(vis, b))
            lost = any(o and not n for n, o in zip(vis, b))
            partial = any(vis) and not all(vis)
            target = None
            if gained or (partial and md != "SUB"):
                target = True
            elif lost or partial:
                target = False
            elif test is not None and md in ("SET", "ADD", "SUB", "XOR"):
                try:
                    area, region = where
                    with context.temp_override(area=area, region=region):
                        touched = test(_project(bpy.context, curve_poly(ob, si)))
                except Exception:
                    touched = False
                if touched:
                    target = (md != "SUB") if md != "XOR" else not all(b)
            if target is not None and (target != all(vis) or (not target and any(vis))):
                for p in _pts(s):
                    _set_sel(s, p, target)
                changed.append(ob)
    for ob in set(changed):
        ob.data.update_tag()


@persistent
def _on_depsgraph(scene, depsgraph=None):
    if BUSY["on"]:
        return
    ctx = bpy.context
    try:
        if ctx.mode != 'EDIT_CURVE':
            if SNAP:
                SNAP.clear()
            return
        ops = ctx.window_manager.operators
        op = ops[-1] if len(ops) else None
        key = _op_key(op)
        fresh = key != LAST_OP["key"] and op is not None and op.bl_idname in SELECT_OPS
        LAST_OP["key"] = key
        if mode(ctx) != "CURVE":                # (switching to Curve takes a fresh snapshot)
            return
        BUSY["on"] = True
        try:
            if fresh:
                _whole_curves(ctx, op)
            for ob in _objs(ctx):
                SNAP[ob.name] = flags(ob)
        finally:
            BUSY["on"] = False
    except Exception as exc:
        print("NURBS curve select:", exc)


# --------------------------------------------------------------------------
# the background picker
# --------------------------------------------------------------------------

def _redraw(context):
    for w in context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


def _tool_ok(context):
    try:
        t = context.workspace.tools.from_space_view3d_mode('EDIT_CURVE', create=False)
    except Exception:
        return True
    return t is None or t.idname in SELECT_TOOLS


def _other_modal(win):
    """Another modal operator runs in the window (a drawing tool): its clicks."""
    try:
        return any(op.bl_idname != "NPROTO_OT_curve_picker" for op in win.modal_operators)
    except Exception:
        return False


class NPROTO_OT_curve_picker(bpy.types.Operator):
    """Edit Curve mode: picks hull spans and whole curves by clicking their
    lines (CV / Hull / Curve modes in the header); everything else passes
    through"""
    bl_idname = "nproto.curve_picker"
    bl_label = "Curve Picking"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        if RUN["on"] or context.window is None:
            return {'CANCELLED'}
        RUN.update(on=True, last=None, press=None)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _stop(self, context):
        RUN["on"] = False
        self._hover(context, None)
        return {'FINISHED', 'PASS_THROUGH'}

    @staticmethod
    def _hover(context, h):
        key = None if h is None or h[2] == "CV" else (h[0].name, h[1], h[2], tuple(h[3]))
        if key != HOVER["key"]:
            HOVER["key"] = key
            HOVER["lines"] = None if key is None else h[4]
            _redraw(context)

    def modal(self, context, event):
        if not RUN["on"] or context.mode != 'EDIT_CURVE':
            return self._stop(context)
        if event.type not in ('MOUSEMOVE', 'LEFTMOUSE'):
            return {'PASS_THROUGH'}
        try:
            from . import subsel
            win = context.window
            where = subsel._view_region_at(win, event.mouse_x, event.mouse_y) if win is not None else None
            busy = (where is None or not _tool_ok(context) or _other_modal(win)
                    or subsel._over_gizmo(context, where[0], where[1], event.mouse_x, event.mouse_y))
            if busy:
                self._hover(context, None)
                return {'PASS_THROUGH'}
            area, region = where
            xy = (event.mouse_x - region.x, event.mouse_y - region.y)
            md = mode(context)

            def hit():
                with context.temp_override(window=win, area=area, region=region):
                    return hit_at(bpy.context, xy, md)

            if event.type == 'MOUSEMOVE':
                if (event.mouse_x, event.mouse_y) != RUN["last"]:
                    RUN["last"] = (event.mouse_x, event.mouse_y)
                    self._hover(context, hit())
                return {'PASS_THROUGH'}
            if event.value == 'PRESS':
                # the flags before Blender's own press-select: a click puts them
                # back and applies its pick (Shift toggles on what was there)
                RUN["where"] = where
                RUN["press"] = {o.name: flags(o) for o in _objs(context)}
                self._hover(context, None)
                return {'PASS_THROUGH'}
            if event.ctrl or event.alt or event.oskey:
                return {'PASS_THROUGH'}
            if event.value == 'DOUBLE_CLICK':
                if md != "CV":
                    return {'PASS_THROUGH'}
                h = hit()
                if h is None or h[2] != "CV":
                    return {'PASS_THROUGH'}
                # the first click picked the CV: the whole curve now (Shift adds it)
                ob, si = h[0], h[1]
                s = ob.data.splines[si]
                BUSY["on"] = True
                try:
                    if not event.shift:
                        deselect_all(_objs(context))
                    for p in _pts(s):
                        _set_sel(s, p, True)
                    _tag(_objs(context))
                finally:
                    BUSY["on"] = False
                _redraw(context)
                bpy.ops.ed.undo_push(message="Select Curve")
                return {'RUNNING_MODAL'}
            if event.value != 'CLICK':
                return {'PASS_THROUGH'}
            h = hit()
            if md == "CV" and (h is None or h[2] == "CV"):
                return {'PASS_THROUGH'}             # a CV, or nothing: Blender's own click
            before = RUN.get("press")
            with context.temp_override(window=win, area=area, region=region):
                BUSY["on"] = True
                try:
                    apply(bpy.context, None if h is None else h[:4], event.shift, before)
                finally:
                    BUSY["on"] = False
            self._hover(context, h)
            _redraw(context)
            bpy.ops.ed.undo_push(message="Deselect All" if h is None else
                                 ("Select Curve" if h[2] == "CURVE" else "Select Hull Span"))
            return {'RUNNING_MODAL'}
        except Exception as exc:
            print("NURBS curve select:", exc)
            return {'PASS_THROUGH'}


def _start_picker():
    if RUN["on"] or bpy.context.mode != 'EDIT_CURVE':
        return None
    wm = bpy.context.window_manager
    for win in wm.windows:
        area = next((a for a in win.screen.areas if a.type == 'VIEW_3D'), None)
        if area is None:
            continue
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        try:
            with bpy.context.temp_override(window=win, area=area, region=region):
                bpy.ops.nproto.curve_picker('INVOKE_DEFAULT')
        except Exception as exc:
            print("NURBS curve select: picker not started", exc)
        break
    return None


def ensure_running():
    """In Edit Curve mode with no picker: start it (from a timer, never inside a draw)."""
    try:
        if (not RUN["on"] and bpy.context.mode == 'EDIT_CURVE'
                and not bpy.app.timers.is_registered(_start_picker)):
            bpy.app.timers.register(_start_picker, first_interval=0.05)
    except Exception:
        pass


def mode_changed(view, context):
    HOVER["key"] = HOVER["lines"] = None
    RUN["last"] = None
    if view.curve_sel_mode == "CURVE" and context.mode == 'EDIT_CURVE':
        # existing partial picks become whole curves
        objs = _objs(context)
        for ob in objs:
            for s in ob.data.splines:
                P = _pts(s)
                if any(_is_sel(s, p) for p in P if not p.hide):
                    for p in P:
                        _set_sel(s, p, True)
        _tag(objs)
    _redraw(context)


def _editing_curve(context):
    ob = context.active_object
    return context.mode == 'EDIT_CURVE' and ob is not None and ob.type == 'CURVE'


def draw_tool_header(self, context):
    """CV | Hull | Curve in the tool settings bar in Edit Curve mode."""
    v = getattr(context.window_manager, "nproto_view", None)
    if v is None or not _editing_curve(context):
        return
    self.layout.separator()
    row = self.layout.row(align=True)
    row.prop(v, "curve_sel_mode", expand=True)
    ensure_running()


def draw_mode_buttons(layout, context):
    """The same as icons in the viewport header while the tool settings bar is hidden."""
    v = getattr(context.window_manager, "nproto_view", None)
    sd = getattr(context, "space_data", None)
    if v is None or not _editing_curve(context):
        return
    ensure_running()
    if getattr(sd, "show_region_tool_header", False):
        return
    row = layout.row(align=True)
    row.prop(v, "curve_sel_mode", expand=True, icon_only=True)


def draw_view(ctx, draw_lines):
    """The span / curve a click would pick, lit under the mouse."""
    if ctx.mode != 'EDIT_CURVE' or HOVER["key"] is None or not HOVER["lines"]:
        return
    import gpu
    gpu.state.depth_test_set('NONE')
    lines = []
    for W in HOVER["lines"]:
        W = np.asarray(W, float)
        lines += [tuple(W[k // 2 + k % 2]) for k in range(2 * (len(W) - 1))]
    draw_lines(lines, COL_HOVER, 4.0)
    gpu.state.depth_test_set('LESS_EQUAL')


@persistent
def _on_load(_):
    RUN.update(on=False, last=None, where=None, press=None)
    HOVER["key"] = HOVER["lines"] = None
    SNAP.clear()
    LAST_OP["key"] = None
    ensure_running()


classes = (NPROTO_OT_curve_picker,)


def register():
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    ensure_running()


def unregister():
    RUN["on"] = False
    for lst, f in ((bpy.app.handlers.load_post, _on_load), (bpy.app.handlers.depsgraph_update_post, _on_depsgraph)):
        if f in lst:
            lst.remove(f)
    if bpy.app.timers.is_registered(_start_picker):
        bpy.app.timers.unregister(_start_picker)
