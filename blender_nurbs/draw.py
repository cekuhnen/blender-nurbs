"""Viewport overlay: B-rep edges, CV nets, and pick highlights."""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import store

_handle = None
_handle_px = None
_label_cache = {}
pick_overlay = {"hover": None, "picked": []}  # filled by the modal pick operator
# filled by the sketch tools (CV Curve / Line): world-space polylines and points
sketch_overlay = {"curves": [], "hull": [], "cvs": [], "snap": None, "snap_label": None, "guide": [], "axes": [],
                  "refs": [], "cursor": None}
COL_CURSOR = (1.0, 1.0, 1.0, 1.0)
COL_SNAP = (0.25, 1.0, 0.45, 1.0)
COL_GUIDE = (0.6, 0.6, 0.6, 0.8)

COL_EDGE = (0.05, 0.05, 0.05, 1.0)
COL_EDGE_SEL = (1.0, 0.55, 0.1, 1.0)
COL_HULL = (0.55, 0.25, 0.85, 0.8)
COL_CV = (0.85, 0.3, 1.0, 1.0)
COL_CV_SEL = (1.0, 0.62, 0.12, 1.0)
COL_CV_ACT = (1.0, 1.0, 1.0, 1.0)
COL_HULL_SEL = (1.0, 0.62, 0.12, 0.9)
COL_ISO = (1.0, 0.78, 0.25, 1.0)
COL_GHOST = (0.2, 0.85, 1.0, 0.9)
# the hidden object picked in the Construction Tree, whose edges are drawn as a
# ghost; an object hidden with H while active is not one
ghost = {"name": None}
COL_HOVER = (1.0, 0.95, 0.2, 1.0)
COL_PICKS = [(0.1, 0.8, 1.0, 1.0), (1.0, 0.25, 0.6, 1.0), (0.3, 1.0, 0.4, 1.0)]


def _lines_from_polylines(polys, M, bias=None):
    """World-space line pairs; `bias` = (eye, view_dir, amount) pulls every point
    toward the viewer (see _edge_bias)."""
    out = []
    for pl in polys:
        w = [M @ Vector(p) for p in pl]
        if bias is not None:
            eye, vdir, amount = bias
            if eye is not None:
                w = [p + (eye - p).normalized() * amount for p in w]
            else:
                w = [p + vdir * amount for p in w]
        for i in range(len(w) - 1):
            out += [w[i], w[i + 1]]
    return out


def _edge_bias(ctx, obj, chord):
    """Exact B-rep edges lie ON the surface, but the shaded mesh is a tessellation
    whose flat triangles bulge up to the chord tolerance off it -- on convex areas
    (fillets, cylinders, seams) the mesh then hides parts of the edges. Pull the
    edge lines toward the viewer by a bit more than that chord error."""
    rv3d = ctx.region_data
    if rv3d is None:
        return None
    M = obj.matrix_world
    corners = [M @ Vector(c) for c in obj.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    amount = 1.5 * chord * ((hi - lo).length or 1.0) + 1e-5
    vinv = rv3d.view_matrix.inverted()
    if rv3d.is_perspective:
        return vinv.translation.copy(), None, amount
    return None, (vinv.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized(), amount


def _draw_lines(coords, color, width):
    if not coords:
        return
    sh = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    sh.uniform_float("viewportSize", (region.width, region.height))
    sh.uniform_float("lineWidth", width)
    sh.uniform_float("color", color)
    batch_for_shader(sh, 'LINES', {"pos": coords}).draw(sh)


def _draw_points(coords, color, size):
    if not coords:
        return
    sh = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
    gpu.state.point_size_set(size)
    sh.uniform_float("color", color)
    batch_for_shader(sh, 'POINTS', {"pos": coords}).draw(sh)


_topo_cache = {}   # session_uid -> (rev, data)


def topology(obj):
    """CV hull, CV points and span isoparms (lines at the knots) of a
    single-face surface, in object space; None if it has no CV net."""
    rev = obj.get(store.PROP_REV, 0)
    hit = _topo_cache.get(obj.session_uid)
    if hit and hit[0] == rev:
        return hit[1]
    data = None
    try:
        from .kernel import load as _load
        ext = _load().blender_ext
        brep = store.get_brep(obj)
        if brep.face_count() == 1:
            surf = brep.faces()[0].surface()
            d = ext.surface_cvs(surf)
            nu, nv, P = d["count_u"], d["count_v"], [tuple(p) for p in d["points"]]
            hull = []
            for iu in range(nu):
                hull.append([P[iu * nv + iv] for iv in range(nv)])
            for iv in range(nv):
                hull.append([P[iu * nv + iv] for iu in range(nu)])
            (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
            ku = sorted({k for k in d["knots_u"] if u0 - 1e-12 <= k <= u1 + 1e-12})
            kv = sorted({k for k in d["knots_v"] if v0 - 1e-12 <= k <= v1 + 1e-12})
            n = 48
            iso = [[tuple(surf.evaluate(u, v0 + (v1 - v0) * i / n)) for i in range(n + 1)] for u in ku]
            iso += [[tuple(surf.evaluate(u0 + (u1 - u0) * i / n, v)) for i in range(n + 1)] for v in kv]
            data = {"hull": hull, "cvs": P, "iso": iso, "spans": (len(ku) - 1, len(kv) - 1),
                    "count": (nu, nv), "degree": (d["degree_u"], d["degree_v"])}
    except Exception:
        data = None
    _topo_cache[obj.session_uid] = (rev, data)
    return data


_PREVIEW_OPS = ("NPROTO_OT_rebuild_surface", "nproto.rebuild_surface", "NPROTO_OT_square", "nproto.square",
                "NPROTO_OT_refit_trim", "nproto.refit_trim")


def _rebuild_preview_active(ctx):
    ops = ctx.window_manager.operators
    return bool(ops) and ops[-1].bl_idname in _PREVIEW_OPS


def _square_feature(obj):
    h = obj.get("nproto_hist")
    return isinstance(h, str) and '"op": "square"' in h


def _rebuild_step_open(obj):
    """A Rebuild history step is open for editing in the Feature panel."""
    k = getattr(getattr(obj, "nproto_step", None), "index", -1)
    if k < 0 or "nproto_stack" not in obj:
        return False
    import json
    try:
        raw = json.loads(obj["nproto_stack"])
    except ValueError:
        return False
    return k < len(raw) and raw[k]["op"] in ("rebuild_surface", "refit_trim") and not raw[k]["params"].get("_off")


def cv_net_shown(ctx, obj):
    """Is this NURBS object's CV net drawn (CVs / Hulls on and selected, or the
    active Square / open Rebuild step)? The sketch tools snap to those CVs."""
    if not store.is_nurbs(obj):
        return False
    v = getattr(ctx.window_manager, "nproto_view", None)
    if v is not None and v.show_cvs and obj.select_get():
        return True
    act = ctx.view_layer.objects.active
    return obj == act and (_rebuild_step_open(act) or (act.select_get() and _square_feature(act)))


def _draw_cage(ctx):
    """CV editing: draw the cage like Show CVs (purple hull, square CVs), on top
    of Blender's own edit-mesh wire; selected CVs / hull legs in orange."""
    obj = ctx.view_layer.objects.active
    if obj is None or obj.mode != 'EDIT' or obj.type != 'MESH' or "nproto_cage_of" not in obj:
        return
    import bmesh
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        if not bm.is_valid:
            return
        M = obj.matrix_world.copy()
        W = [M @ v.co for v in bm.verts]
        vsel = [v.select for v in bm.verts]
        hull, hull_sel = [], []
        for e in bm.edges:
            i, j = e.verts[0].index, e.verts[1].index
            (hull_sel if e.select else hull).extend((W[i], W[j]))
        act = bm.select_history.active
        act_i = act.index if isinstance(act, bmesh.types.BMVert) else -1
        del bm, act
    except Exception:
        return
    pts = [W[i] for i in range(len(W)) if not vsel[i]]
    sel = [W[i] for i in range(len(W)) if vsel[i] and i != act_i]
    gpu.state.depth_test_set('NONE')
    _draw_lines(hull, COL_HULL, 1.5)
    _draw_lines(hull_sel, COL_HULL_SEL, 2.0)
    _draw_points(pts, COL_CV, 7.0)
    _draw_points(sel, COL_CV_SEL, 8.0)
    if 0 <= act_i < len(W):
        _draw_points([W[act_i]], COL_CV_ACT, 8.0)
    gpu.state.depth_test_set('LESS_EQUAL')


def _draw():
    ctx = bpy.context
    scn = ctx.scene
    v = getattr(ctx.window_manager, "nproto_view", None)
    show_edges = v.show_edges if v else True
    show_cvs = v.show_cvs if v else False
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    from . import diagnostics
    if v is not None:
        diagnostics.draw_zebra(ctx)          # under the edges
    chord = store.tess_settings(ctx)[0]
    for obj in ctx.visible_objects:
        if not store.is_nurbs(obj):
            continue
        d = store.display(obj)
        M = obj.matrix_world
        sel = obj.select_get()
        if show_edges:
            _draw_lines(_lines_from_polylines(d.get("edges", []), M, _edge_bias(ctx, obj, chord)),
                        COL_EDGE_SEL if sel else COL_EDGE, 2.0 if sel else 1.5)
        if show_cvs and sel:
            gpu.state.depth_test_set('NONE')
            for net in d.get("nets", []):
                nu, nv, P = net["nu"], net["nv"], net["p"]
                W = [M @ Vector(p) for p in P]
                hull = []
                for iu in range(nu):
                    for iv in range(nv - 1):
                        hull += [W[iu * nv + iv], W[iu * nv + iv + 1]]
                for iv in range(nv):
                    for iu in range(nu - 1):
                        hull += [W[iu * nv + iv], W[(iu + 1) * nv + iv]]
                _draw_lines(hull, COL_HULL, 1.0)
                _draw_points(W, COL_CV, 6.0)
            gpu.state.depth_test_set('LESS_EQUAL')

    _draw_cage(ctx)

    if v is not None:
        diagnostics.draw_combs(ctx, _draw_lines)

    # Rebuild Surface: while its F9 panel is up, show the new topology
    # (span isoparms + CV hull) of the rebuilt surfaces
    show_f9 = _rebuild_preview_active(ctx)
    act = ctx.view_layer.objects.active
    step_open = act is not None and store.is_nurbs(act) and (_rebuild_step_open(act) or
                                                              (act.select_get() and _square_feature(act)))
    if show_f9 or step_open:
        gpu.state.depth_test_set('NONE')
        for obj in (ctx.selected_objects if show_f9 else [act]):
            if not store.is_nurbs(obj):
                continue
            t = topology(obj)
            if t is None:
                continue
            M = obj.matrix_world
            _draw_lines(_lines_from_polylines(t["iso"], M), COL_ISO, 2.5)
            _draw_lines(_lines_from_polylines(t["hull"], M), COL_HULL, 1.0)
            _draw_points([M @ Vector(p) for p in t["cvs"]], COL_CV, 6.0)
        gpu.state.depth_test_set('LESS_EQUAL')

    # the active object may be a hidden history input picked in the Construction
    # Tree: show its edges so you can see what you are editing
    act = ctx.view_layer.objects.active
    if (show_edges and act is not None and act.name == ghost["name"] and store.is_nurbs(act)
            and act.hide_get()):
        gpu.state.depth_test_set('NONE')
        _draw_lines(_lines_from_polylines(store.display(act).get("edges", []), act.matrix_world),
                    COL_GHOST, 2.5)

    # pick highlights (world-space polylines)
    gpu.state.depth_test_set('NONE')
    for i, pl in enumerate(pick_overlay["picked"]):
        _draw_lines(_lines_from_polylines([pl], _IDENT), COL_PICKS[min(i, 2)], 4.0)
    if pick_overlay["hover"]:
        _draw_lines(_lines_from_polylines([pick_overlay["hover"]], _IDENT), COL_HOVER, 4.0)
    so = sketch_overlay
    if so["guide"]:
        _draw_lines(_lines_from_polylines([so["guide"]], _IDENT), COL_GUIDE, 1.0)
    for pl, col in so.get("axes") or []:          # X / Y / Z lock: the axis (or the plane's two axes)
        _draw_lines(_lines_from_polylines([pl], _IDENT), col, 1.5)
    for pl, col, w in so.get("refs") or []:       # the edge (white) / face (beige) a G1 / G2 end follows
        _draw_lines(_lines_from_polylines([pl], _IDENT), col, w)
    if so["hull"]:
        _draw_lines(_lines_from_polylines([so["hull"]], _IDENT), COL_HULL, 1.5)
    for pl in so["curves"]:
        _draw_lines(_lines_from_polylines([pl], _IDENT), COL_HOVER, 3.0)
    if so["cvs"]:
        _draw_points(so["cvs"], COL_CV, 7.0)
    if so["snap"] is not None:
        _draw_points([so["snap"]], COL_SNAP, 13.0)
    if so.get("cursor") is not None:     # where the next point / CV lands
        _draw_points([so["cursor"]], (0.0, 0.0, 0.0, 1.0), 10.0)
        _draw_points([so["cursor"]], COL_CURSOR, 7.0)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


class _Ident:
    def __matmul__(self, v):
        return v


_IDENT = _Ident()


def _draw_paused(ctx, region):
    """Pause Rebuild is on: say so at the bottom of every 3D view."""
    import blf
    from . import history
    s = getattr(ctx.scene, "nproto_settings", None)
    if s is None or not getattr(s, "pause_rebuild", False):
        return
    n = len(history.waiting(ctx.scene))
    txt = "Rebuild paused" + (f" · {n} waiting · press Rebuild in the Feature panel" if n else "")
    font = 0
    blf.size(font, 13)
    w, h = blf.dimensions(font, txt)
    x, y = (region.width - w) / 2, 28
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    sh.uniform_float("color", (0.05, 0.05, 0.05, 0.75))
    a, b, c_, d_ = (x - 8, y - 6), (x + w + 8, y - 6), (x + w + 8, y + h + 6), (x - 8, y + h + 6)
    batch_for_shader(sh, 'TRIS', {"pos": [a, b, c_, a, c_, d_]}).draw(sh)
    blf.color(font, 1.0, 0.75, 0.3, 1.0)
    blf.position(font, x, y, 0)
    blf.draw(font, txt)


def _draw_px():
    """Continuity labels (Free / G0 / G1 / G2) on the edges of selected Square,
    Loft, Loft Rails and Blend Curve features (clickable: see contlabels)."""
    import blf
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from . import contlabels
    ctx = bpy.context
    region, rv3d = ctx.region, ctx.region_data
    if rv3d is None:
        return
    _draw_paused(ctx, region)
    _draw_snap_label(region, rv3d)
    font = 0
    blf.size(font, 11)
    cols = {"FREE": (0.62, 0.62, 0.62, 1), "G0": (0.9, 0.9, 0.9, 1), "G1": (0.35, 0.8, 1.0, 1),
            "G2": (0.4, 1.0, 0.5, 1)}
    for obj in contlabels.visible_label_objects(ctx):
        for i, L in enumerate(contlabels.labels(obj)):
            q = location_3d_to_region_2d(region, rv3d, L["pos"])
            if q is None:
                continue
            txt = "Guide" if L.get("guide") else contlabels.SHORT[L["value"]]
            w, h = blf.dimensions(font, txt)
            r = max(w, h) * 0.5 + 5
            hot = contlabels.hover["key"] == (obj.name, i)
            bad = L.get("check") is not None and not L["check"]["ok"]
            eased = bad and L["check"].get("eased")
            if hot:
                _disc(q, r + (5.0 if bad else 2.5), (1.0, 0.85, 0.2, 1.0))      # hover ring: click to change
            if L.get("pending"):
                _disc(q, r + 2.5, (0.55, 0.6, 0.75, 1.0))     # grey-blue: paused, not built / checked yet
            elif eased:
                _disc(q, r + 2.5, (1.0, 0.55, 0.1, 1.0))      # orange: holds, eased at an impossible corner
            elif bad:
                _disc(q, r + 2.5, (1.0, 0.15, 0.15, 1.0))     # red ring: the edge does NOT hold this
            _disc(q, r, (0.05, 0.05, 0.05, 0.9) if len(L["values"]) > 1 else (0.05, 0.05, 0.05, 0.55))
            blf.color(font, *cols.get(L["value"], cols["G0"]))
            blf.position(font, q.x - w / 2, q.y - h / 2 + 1, 0)
            blf.draw(font, txt)
            if abs(L.get("tension", 1.0) - 1.0) > 1e-6:     # Patch: this edge's tension, beside the disc
                t_ = f"T {L['tension']:.2g}"
                blf.color(font, 0.85, 0.85, 0.85, 1.0)
                blf.position(font, q.x + r + 4, q.y - h / 2 + 1, 0)
                blf.draw(font, t_)
        # Patch: its base's four corners (white diamonds) and the other
        # junctions between its inputs (hollow dots); click a dot to move the
        # nearest corner there
        from .patch import corner_points
        for i, c in enumerate(corner_points(obj) if contlabels._is_patch(obj) else []):
            q = location_3d_to_region_2d(region, rv3d, c["pos"])
            if q is None:
                continue
            hot = contlabels.hover["key"] == ("corner", obj.name, i)
            if c["corner"]:
                _diamond(q, 8.5 if hot else 7.0, (0.05, 0.05, 0.05, 0.9))
                _diamond(q, 6.0 if hot else 4.8, (1.0, 0.85, 0.2, 1.0) if hot else (1.0, 1.0, 1.0, 1.0))
            else:
                _disc(q, 6.5 if hot else 5.0, (0.05, 0.05, 0.05, 0.85))
                _disc(q, 4.5 if hot else 3.2, (1.0, 0.85, 0.2, 1.0) if hot else (0.75, 0.75, 0.75, 1.0))
                _disc(q, 2.0 if hot else 1.6, (0.05, 0.05, 0.05, 0.85))
        # XNURBS-style: a square on every corner where an edge's continuity is
        # eased (no surface can follow both neighbours there), with the angle
        for m in contlabels.corner_marks(obj):
            q = location_3d_to_region_2d(region, rv3d, m["pos"])
            if q is None:
                continue
            _square(q, 6.0, (0.05, 0.05, 0.05, 0.9))
            _square(q, 4.5, (1.0, 0.55, 0.1, 1.0))
            if m.get("deg"):
                t_ = f"{m['deg']:.1f}°"
                blf.color(font, 1.0, 0.62, 0.25, 1.0)
                blf.position(font, q.x + 9, q.y - 4, 0)
                blf.draw(font, t_)


def _draw_snap_label(region, rv3d):
    """Sketch tools: what the cursor snapped to (End, Mid, Center, CV, Edge...),
    beside the green snap dot."""
    import blf
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    so = sketch_overlay
    if so.get("snap") is None or not so.get("snap_label"):
        return
    q = location_3d_to_region_2d(region, rv3d, so["snap"])
    if q is None:
        return
    font = 0
    blf.size(font, 11)
    txt = so["snap_label"]
    w, h = blf.dimensions(font, txt)
    x, y = q.x + 11, q.y + 7
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    sh.uniform_float("color", (0.05, 0.05, 0.05, 0.75))
    a, b, c_, d_ = (x - 4, y - 3), (x + w + 4, y - 3), (x + w + 4, y + h + 3), (x - 4, y + h + 3)
    batch_for_shader(sh, 'TRIS', {"pos": [a, b, c_, a, c_, d_]}).draw(sh)
    blf.color(font, *COL_SNAP)
    blf.position(font, x, y, 0)
    blf.draw(font, txt)


def _square(c, h, color):
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    sh.uniform_float("color", color)
    a, b, c_, d_ = (c.x - h, c.y - h), (c.x + h, c.y - h), (c.x + h, c.y + h), (c.x - h, c.y + h)
    batch_for_shader(sh, 'TRIS', {"pos": [a, b, c_, a, c_, d_]}).draw(sh)


def _diamond(c, h, color):
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    sh.uniform_float("color", color)
    a, b, c_, d_ = (c.x, c.y - h), (c.x + h, c.y), (c.x, c.y + h), (c.x - h, c.y)
    batch_for_shader(sh, 'TRIS', {"pos": [a, b, c_, a, c_, d_]}).draw(sh)


def _disc(c, r, color):
    import math
    pts = [(c.x + r * math.cos(a), c.y + r * math.sin(a)) for a in [i * 2 * math.pi / 24 for i in range(24)]]
    tris = []
    for i in range(24):
        tris += [(c.x, c.y), pts[i], pts[(i + 1) % 24]]
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    sh.uniform_float("color", color)
    batch_for_shader(sh, 'TRIS', {"pos": tris}).draw(sh)


def register():
    global _handle, _handle_px
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
    if _handle_px is None:
        _handle_px = bpy.types.SpaceView3D.draw_handler_add(_draw_px, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _handle, _handle_px
    if _handle_px is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle_px, 'WINDOW')
        _handle_px = None
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
