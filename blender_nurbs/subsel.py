"""Face / edge selection: Body / Face / Edge modes, as in Plasticity.

Body mode is Blender's own object selection. In Face or Edge mode a click in
the viewport picks a face or edge of a NURBS body whatever tool is active (a
background handler takes the clicks and mouse moves; everything else passes
through): a click picks one (and selects its body), Shift+click adds or
removes one, a click on nothing clears the picks; what is under the mouse
lights up. The picks live on the object (an ID property, so
undo keeps them), each with a point on it, so they still mean the same face
or edge after a rebuild renumbers them; they count while their body is
selected. What follows them:

* CVs / Hulls: a body with picked faces shows only those faces' nets.
* Edge Fillet and Continuity Check take the picked edges (no picking).
"""

import json

import numpy as np

import bpy
from bpy_extras import view3d_utils

from . import store
from .kernel import load as _load_kernel

PROP = "nproto_subsel"
HOVER = {"key": None}          # (object name, "FACE" / "EDGE", index) under the mouse
COL_PICK = (1.0, 0.55, 0.1, 0.35)
COL_PICK_LINE = (1.0, 0.55, 0.1, 1.0)
COL_HOVER = (1.0, 0.95, 0.3, 0.22)
COL_HOVER_LINE = (1.0, 0.95, 0.2, 1.0)
COL_STEP = (0.2, 0.85, 1.0, 0.16)          # what the open history step made (as the hidden-input outline)
COL_STEP_LINE = (0.2, 0.85, 1.0, 0.9)

_valid = {}    # session_uid -> (rev, raw, kind, [indices])
_ftris = {}    # (session_uid, rev, face) -> (points, triangles, boundary polylines), local space
_epts = {}     # session_uid -> (rev, [polyline or None])


def mode(context=None):
    v = getattr((context or bpy.context).window_manager, "nproto_view", None)
    return v.sel_mode if v is not None else "BODY"


def _size(obj):
    lo, hi = (np.array(v, float) for v in store.get_brep(obj).bounding_box())
    return float(np.linalg.norm(hi - lo)) or 1.0


def _edge_polys(obj):
    rev = obj.get(store.PROP_REV, 0)
    hit = _epts.get(obj.session_uid)
    if hit and hit[0] == rev:
        return hit[1]
    polys = [np.array(p, float) if len(p) > 1 else None for p in store.display(obj).get("edges", [])]
    _epts[obj.session_uid] = (rev, polys)
    return polys


def _dist_poly(P, q):
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(((q - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - q, axis=1).min())


def _anchor(obj, kind, i):
    """A point on the face / edge (local space): finds it again after a rebuild."""
    if kind == "FACE":
        p = store.get_brep(obj).faces()[i].interior_point()
        return [float(p.x), float(p.y), float(p.z)] if hasattr(p, "x") else [float(x) for x in p]
    P = _edge_polys(obj)[i]
    return [float(x) for x in P[len(P) // 2]]


def _locate(obj, kind, i, anchor, size):
    """Index of the face / edge a stored pick means in the current geometry."""
    q = np.array(anchor, float)
    if kind == "FACE":
        faces = store.get_brep(obj).faces()

        def d(k):
            return faces[k].closest_point(tuple(q))[1]
    else:
        polys = _edge_polys(obj)

        def d(k):
            return _dist_poly(polys[k], q) if polys[k] is not None else np.inf
        faces = polys
    if 0 <= i < len(faces) and d(i) < 1e-4 * size:
        return i
    ds = [d(k) for k in range(len(faces))]
    if not ds:
        return None
    k = int(np.argmin(ds))
    return k if ds[k] < 0.02 * size else None


def get(obj):
    """(kind, [indices]) picked on obj, checked against its current geometry;
    (None, []) when nothing is picked or the body is not selected."""
    if obj is None or not store.is_nurbs(obj) or not obj.select_get():
        return None, []
    raw = obj.get(PROP)
    if not raw:
        return None, []
    rev = obj.get(store.PROP_REV, 0)
    hit = _valid.get(obj.session_uid)
    if hit and hit[0] == rev and hit[1] == raw:
        return hit[2], hit[3]
    try:
        d = json.loads(raw)
    except ValueError:
        return None, []
    kind, idx = d.get("kind"), []
    try:
        size = _size(obj)
        for i, a in d.get("items", []):
            j = _locate(obj, kind, int(i), a, size)
            if j is not None and j not in idx:
                idx.append(j)
    except Exception as exc:
        print("NURBS select:", obj.name, exc)
    _valid[obj.session_uid] = (rev, raw, kind, idx)
    return kind, idx


def picks(obj, kind):
    k, idx = get(obj)
    return idx if k == kind else []


def set_picks(obj, kind, idx):
    """Store the picked faces / edges of obj (an empty list removes them)."""
    idx = list(dict.fromkeys(int(i) for i in idx))
    if not idx:
        if PROP in obj:
            del obj[PROP]
        return
    obj[PROP] = json.dumps({"kind": kind, "items": [[i, _anchor(obj, kind, i)] for i in idx]})


def clear_all(scene):
    for o in scene.objects:
        if PROP in o:
            del o[PROP]


_stepf = {}    # session_uid -> (key, [indices])


def step_faces(obj):
    """The faces the history step open in the panel made (an edge fillet:
    its fillet faces), while that body is the active one; [] otherwise."""
    pg = getattr(obj, "nproto_step", None)
    if (pg is None or not pg.focus or pg.index < 0 or obj != bpy.context.view_layer.objects.active
            or not store.is_nurbs(obj)):
        return []
    try:
        raw = json.loads(obj.get("nproto_stack", "[]"))
        e = raw[pg.index]
    except (ValueError, IndexError, TypeError):
        return []
    if e.get("op") != "edge_fillet":
        return []
    from . import steps
    info = steps.rims_of(obj, e["params"].get("mids_local", [])) or {}
    pts = info.get("faces") or []
    key = (obj.get(store.PROP_REV, 0), pg.index, json.dumps(pts))
    hit = _stepf.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    idx = []
    try:
        size = _size(obj)
        for p in pts:
            j = _locate(obj, "FACE", -1, p, size)
            if j is not None and j not in idx:
                idx.append(j)
    except Exception as exc:
        print("NURBS step faces:", exc)
    _stepf[obj.session_uid] = (key, idx)
    return idx


def faces_shown(obj):
    """The faces whose CV nets CVs / Hulls shows on obj: the picked ones, else
    the ones the open history step made, else None for all."""
    f = picks(obj, "FACE") or step_faces(obj)
    return set(f) if f else None


def selected_edges(context):
    """[(object, "E<i>")] -- the picked edges of the selected bodies."""
    out = []
    for o in context.selected_objects:
        for i in picks(o, "EDGE"):
            out.append((o, f"E{i}"))
    return out


def summary(obj):
    """'2 faces picked' / '' for the Active Object panel."""
    k, idx = get(obj)
    if not idx:
        return ""
    noun = "face" if k == "FACE" else "edge"
    return f"{len(idx)} {noun}{'s' if len(idx) > 1 else ''} picked: " + ", ".join(
        ("F" if k == "FACE" else "E") + str(i) for i in idx[:8]) + (" ..." if len(idx) > 8 else "")


# --------------------------------------------------------------------------
# geometry for the highlight: one face on its own
# --------------------------------------------------------------------------

def face_tris(obj, i):
    """(points (n,3), triangles (m,3), boundary polylines) of face i, local."""
    rev = obj.get(store.PROP_REV, 0)
    key = (obj.session_uid, rev, i)
    hit = _ftris.get(key)
    if hit is not None:
        return hit
    for k in [k for k in _ftris if k[0] == obj.session_uid and k[1] != rev]:
        del _ftris[k]
    sm = _load_kernel()
    ext = sm.blender_ext
    b = store.get_brep(obj).copy()
    faces = b.faces()
    others = [f for k, f in enumerate(faces) if k != i]
    if others:
        ext.delete_faces(b, others)
    chord, angle = store.tess_settings()
    size = _size(obj)
    pb, _ = b.tessellate(chord_height_tolerance=chord * size, curve_angle_tolerance_deg=angle,
                         surface_angle_tolerance_deg=angle, allow_partial=True, return_failures=True)
    a = pb.to_numpy_arrays()
    pts = np.asarray(a["points"], float)
    f = np.asarray(a["faces"])
    tris = []
    j = 0
    while j < len(f):
        c = int(f[j])
        poly = f[j + 1:j + 1 + c]
        for t in range(1, c - 1):
            tris.append((poly[0], poly[t], poly[t + 1]))
        j += 1 + c
    polys = []
    for e in b.edges():
        try:
            t0, t1 = e.parameter_range()
            polys.append(np.array([e.point_at(t0 + (t1 - t0) * s / 39) for s in range(40)], float))
        except Exception:
            pass
    out = (pts, np.array(tris, dtype=np.int64).reshape(-1, 3), polys)
    _ftris[key] = out
    return out


# --------------------------------------------------------------------------
# what is under the mouse
# --------------------------------------------------------------------------

def _ray(context, xy):
    region, rv3d = context.region, context.region_data
    return (view3d_utils.region_2d_to_origin_3d(region, rv3d, xy),
            view3d_utils.region_2d_to_vector_3d(region, rv3d, xy))


def face_under(context, xy):
    """(body, face index) under the mouse, or None (nothing, or not a NURBS body)."""
    o, d = _ray(context, xy)
    hit, loc, _, _, ob, _ = context.scene.ray_cast(context.evaluated_depsgraph_get(), o, d)
    if not hit or ob is None:
        return None
    ob = ob.original
    if not store.is_nurbs(ob):
        return None
    sm = _load_kernel()
    b = store.get_brep(ob)
    q = ob.matrix_world.inverted() @ loc
    f = sm.find_face(b, (float(q.x), float(q.y), float(q.z)))
    i = next((k for k, g in enumerate(b.faces()) if g == f), None)
    return None if i is None else (ob, i)


def edge_under(context, xy, px=10.0):
    """(body, edge index) of the visible B-rep edge nearest the mouse (within
    px pixels), or None. An edge hidden behind a surface is not picked."""
    region, rv3d = context.region, context.region_data
    W, H = region.width, region.height
    PM = np.array(rv3d.perspective_matrix)
    m = np.array(xy, float)
    cands = []
    for ob in context.visible_objects:
        if not store.is_nurbs(ob):
            continue
        M = np.array(ob.matrix_world)
        T = PM @ M
        for i, P in enumerate(_edge_polys(ob)):
            if P is None:
                continue
            h = np.c_[P, np.ones(len(P))] @ T.T
            ok = h[:, 3] > 1e-9
            if not ok.all():
                continue
            s = np.c_[(h[:, 0] / h[:, 3] * 0.5 + 0.5) * W, (h[:, 1] / h[:, 3] * 0.5 + 0.5) * H]
            a, b = s[:-1], s[1:]
            ab = b - a
            t = np.clip(((m - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
            dd = np.linalg.norm(a + t[:, None] * ab - m, axis=1)
            k = int(np.argmin(dd))
            if dd[k] <= px:
                p = P[k] + t[k] * (P[k + 1] - P[k])
                cands.append((float(dd[k]), ob, i, M[:3, :3] @ p + M[:3, 3], a[k] + t[k] * ab[k]))
    if not cands:
        return None
    dg = context.evaluated_depsgraph_get()
    for _, ob, i, pw, sxy in sorted(cands, key=lambda c: c[0]):
        o, d = _ray(context, (float(sxy[0]), float(sxy[1])))
        dist = float((np.array(pw) - np.array(o)) @ np.array(d))
        hit, loc, _, _, hob, _ = context.scene.ray_cast(dg, o, d)
        if hit and (np.array(loc) - np.array(o)) @ np.array(d) < dist - 0.01 * _size(ob) * max(ob.scale):
            continue                     # behind a surface
        return ob, i
    return None


def _redraw(context):
    for w in context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


# --------------------------------------------------------------------------
# picking: a background handler while Face / Edge mode is on
# --------------------------------------------------------------------------

RUN = {"on": False, "last": None}


def _view_region_at(window, x, y):
    """(area, main region) of the 3D viewport under the window point (x, y),
    or None -- also over its toolbar, sidebar or headers."""
    for area in window.screen.areas:
        if area.type != 'VIEW_3D' or not (area.x <= x < area.x + area.width and area.y <= y < area.y + area.height):
            continue
        main = None
        for r in area.regions:
            if not (r.x <= x < r.x + r.width and r.y <= y < r.y + r.height) or r.width < 2 or r.height < 2:
                continue
            if r.type == 'WINDOW':
                main = r
            else:
                return None                  # over a panel / header drawn on the viewport
        return (area, main) if main is not None else None
    return None


def _hit(context, kind, xy):
    return face_under(context, xy) if kind == "FACE" else edge_under(context, xy)


def apply_pick(context, kind, hit, extend):
    """A click in Face / Edge mode: hit = (body, index) or None."""
    if hit is None:
        if not extend:
            clear_all(context.scene)
            for o in context.selected_objects:
                o.select_set(False)
        return
    ob, i = hit
    if extend:
        idx = list(picks(ob, kind))
        if i in idx:
            idx.remove(i)
        else:
            idx.append(i)
    else:
        clear_all(context.scene)
        for o in context.selected_objects:
            o.select_set(False)
        idx = [i]
    ob.select_set(True)
    context.view_layer.objects.active = ob
    set_picks(ob, kind, idx)


class NPROTO_OT_subsel_picker(bpy.types.Operator):
    """Face / Edge mode: picks faces or edges with any tool active (a click
    picks, Shift+click adds or removes, a click on nothing clears; the face /
    edge under the mouse lights up). Everything else passes through"""
    bl_idname = "nproto.subsel_picker"
    bl_label = "Face / Edge Picking"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        if RUN["on"] or context.window is None:
            return {'CANCELLED'}
        RUN["on"], RUN["last"] = True, None
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _stop(self, context):
        RUN["on"] = False
        HOVER["key"] = None
        _redraw(context)
        return {'FINISHED', 'PASS_THROUGH'}

    def modal(self, context, event):
        kind = mode(context)
        if kind == "BODY" or not RUN["on"]:
            return self._stop(context)
        if context.mode != 'OBJECT' or event.type not in ('MOUSEMOVE', 'LEFTMOUSE'):
            return {'PASS_THROUGH'}
        try:
            win = context.window
            where = _view_region_at(win, event.mouse_x, event.mouse_y) if win is not None else None
            if event.type == 'MOUSEMOVE':
                key = None
                if where is not None and (event.mouse_x, event.mouse_y) != RUN["last"]:
                    RUN["last"] = (event.mouse_x, event.mouse_y)
                    area, region = where
                    with context.temp_override(window=win, area=area, region=region):
                        h = _hit(bpy.context, kind, (event.mouse_x - region.x, event.mouse_y - region.y))
                    key = None if h is None else (h[0].name, kind, h[1])
                elif where is not None:
                    key = HOVER["key"]
                if key != HOVER["key"]:
                    HOVER["key"] = key
                    _redraw(context)
                return {'PASS_THROUGH'}
            # a click (press and release without a drag): Blender sends it after the release
            if event.value != 'CLICK' or event.ctrl or event.alt or event.oskey or where is None:
                return {'PASS_THROUGH'}
            area, region = where
            with context.temp_override(window=win, area=area, region=region):
                h = _hit(bpy.context, kind, (event.mouse_x - region.x, event.mouse_y - region.y))
                apply_pick(bpy.context, kind, h, event.shift)
            bpy.ops.ed.undo_push(message=f"Pick {kind.title()}")
            _redraw(context)
            return {'RUNNING_MODAL'}
        except Exception as exc:
            print("NURBS select:", exc)
            return {'PASS_THROUGH'}


def _start_picker():
    """Start the background picker in the first window with a 3D viewport."""
    if RUN["on"] or mode() == "BODY":
        return None
    wm = bpy.context.window_manager
    for win in wm.windows:
        area = next((a for a in win.screen.areas if a.type == 'VIEW_3D'), None)
        if area is None:
            continue
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        try:
            with bpy.context.temp_override(window=win, area=area, region=region):
                bpy.ops.nproto.subsel_picker('INVOKE_DEFAULT')
        except Exception as exc:
            print("NURBS select: picker not started", exc)
        break
    return None


def ensure_running():
    """Face / Edge mode on but no picker (a file was loaded, or it was stopped):
    start it again (from a timer: never from inside a draw)."""
    if mode() != "BODY" and not RUN["on"] and not bpy.app.timers.is_registered(_start_picker):
        bpy.app.timers.register(_start_picker, first_interval=0.05)


def mode_changed(view, context):
    """Body / Face / Edge switched: the picks go (a new selection). Face / Edge
    start the picker; Body ends it."""
    clear_all(context.scene)
    HOVER["key"] = None
    if view.sel_mode != "BODY":
        ensure_running()
    _redraw(context)


@bpy.app.handlers.persistent
def _on_load(_):
    RUN["on"] = False                       # loading a file ends every running operator
    HOVER["key"] = None
    ensure_running()


def draw_tool_header(self, context):
    """Body | Face | Edge in the tool settings bar, right after Blender's own
    tool settings (its selection-mode icons; see ui._hook_tool_settings):
    always on top, whatever tool is active."""
    v = getattr(context.window_manager, "nproto_view", None)
    if v is None or context.mode != 'OBJECT' or _load_kernel() is None:
        return
    self.layout.separator()
    row = self.layout.row(align=True)
    row.prop(v, "sel_mode", expand=True)
    ensure_running()


def draw_mode_buttons(layout, context):
    """The same as icons in the viewport header -- only while the tool
    settings bar is hidden, so the switch is always somewhere on top."""
    v = getattr(context.window_manager, "nproto_view", None)
    sd = getattr(context, "space_data", None)
    if v is None or context.mode != 'OBJECT' or getattr(sd, "show_region_tool_header", False):
        return
    row = layout.row(align=True)
    row.prop(v, "sel_mode", expand=True, icon_only=True)
    ensure_running()


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _biased(P, M, bias):
    W = P @ np.array(M.to_3x3()).T + np.array(M.translation)
    if bias is None:
        return W
    eye, vdir, amount = bias
    if eye is not None:
        d = np.array(eye) - W
        return W + d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None] * amount
    return W + np.array(vdir) * amount


def _draw_face(ctx, ob, i, fill, line, width, draw_lines, draw_tris, edge_bias):
    pts, tris, polys = face_tris(ob, i)
    bias = edge_bias(ctx, ob, store.tess_settings(ctx)[0])
    if len(tris):
        W = _biased(pts, ob.matrix_world, bias).astype(np.float32)
        draw_tris(W[tris.reshape(-1)], fill)
    lines = []
    for P in polys:
        W = _biased(P, ob.matrix_world, bias)
        for k in range(len(W) - 1):
            lines += [tuple(W[k]), tuple(W[k + 1])]
    draw_lines(lines, line, width)


def _draw_edge(ctx, ob, i, color, width, draw_lines, edge_bias):
    P = _edge_polys(ob)[i]
    if P is None:
        return
    W = _biased(P, ob.matrix_world, edge_bias(ctx, ob, store.tess_settings(ctx)[0]))
    draw_lines([tuple(W[k // 2 + k % 2]) for k in range(2 * (len(W) - 1))], color, width)


def draw_view(ctx, draw_lines, draw_tris, edge_bias):
    import gpu
    if ctx.mode != 'OBJECT':
        return
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('ALPHA')
    for ob in ctx.selected_objects:
        if not store.is_nurbs(ob) or not ob.visible_get():
            continue
        kind, idx = get(ob)
        for i in idx:
            try:
                if kind == "FACE":
                    _draw_face(ctx, ob, i, COL_PICK, COL_PICK_LINE, 2.5, draw_lines, draw_tris, edge_bias)
                else:
                    _draw_edge(ctx, ob, i, COL_PICK_LINE, 4.0, draw_lines, edge_bias)
            except Exception as exc:
                print("NURBS select: draw", exc)
    act = ctx.view_layer.objects.active
    if act is not None and act.visible_get() and not get(act)[1]:
        for i in step_faces(act):
            try:
                _draw_face(ctx, act, i, COL_STEP, COL_STEP_LINE, 2.5, draw_lines, draw_tris, edge_bias)
            except Exception as exc:
                print("NURBS step faces: draw", exc)
    key = HOVER["key"]
    if key is not None and mode(ctx) == key[1]:
        ob = bpy.data.objects.get(key[0])
        if ob is not None and store.is_nurbs(ob) and ob.visible_get():
            try:
                if key[1] == "FACE":
                    _draw_face(ctx, ob, key[2], COL_HOVER, COL_HOVER_LINE, 1.5, draw_lines, draw_tris, edge_bias)
                else:
                    _draw_edge(ctx, ob, key[2], COL_HOVER_LINE, 4.0, draw_lines, edge_bias)
            except Exception:
                pass


classes = (NPROTO_OT_subsel_picker,)


def register():
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    ensure_running()


def unregister():
    RUN["on"] = False                       # a running picker ends at its next event
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    if bpy.app.timers.is_registered(_start_picker):
        bpy.app.timers.unregister(_start_picker)
