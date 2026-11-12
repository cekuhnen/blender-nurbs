"""Clickable continuity labels (Free / G0 / G1 / G2) on the edges of selected
history surfaces: Square, Loft, Loft Rails and Blend Curve.

Each label sits at the middle of the input it belongs to (Blend Curve: at the
end it blends from). A click steps that edge's setting to the next value
(Ctrl / Shift+click: the previous one), through the Feature panel's property,
so the surface rebuilds from its history like any other edit (one undo step).

Picking is done by a 3D-view gizmo group whose gizmos only hit-test the label
discs in screen space (test_select); the discs themselves are drawn by
draw._draw_px.
"""

import numpy as np

import bpy
from bpy.props import IntProperty, StringProperty
from mathutils import Vector

from . import features, store

OPS = ("square", "loft", "loft_rails", "blend_curve", "patch")
ORDER = ("FREE", "G0", "G1", "G2")
SHORT = {"FREE": "Free", "G0": "G0", "G1": "G1", "G2": "G2"}
RADIUS_PX = 13.0

_cache = {}          # session_uid -> (key, labels)
hover = {"key": None}


def _mid(obj, ref):
    kc = features.input_curve(obj, ref)
    t0, t1 = kc.parameter_range()
    return Vector(kc.evaluate(0.5 * (t0 + t1)))


def _edge_values(obj, ref, with_free=False):
    """Settings an input allows: a curve carries position only."""
    base = ["G0", "G1", "G2"] if features.is_surface_edge(ref) and obj.type != 'CURVE' else ["G0"]
    return (["FREE"] + base) if with_free else base


def _compute(obj):
    out = _compute_base(obj)
    try:
        entries = features.stack(obj)
    except Exception:
        return out
    if not any(e["op"] == "surface_align" for e in entries):
        return out
    from . import geom, store
    from mathutils import Vector
    surf = store.get_surfaces(obj)[0]
    M = geom.world_matrix(obj)
    at = 0.3 if out else 0.5          # off the middle when the base feature has labels there too
    for k, e in enumerate(entries):
        if e["op"] != "surface_align":
            continue
        for i, (edge, (src, ref)) in enumerate(zip(e["params"]["edges"], e["inputs"])):
            pts = geom.boundary_points(surf, edge, 21)
            vals = ["G0", "G1", "G2"] if features.is_surface_edge(ref) and src.type != 'CURVE' else ["G0"]
            cur = e["params"]["cont"][i]
            out.append({"pos": M @ Vector(pts[int(round(at * 20))]), "param": f"stack:{k}:{i}",
                        "value": cur if cur in vals else vals[0], "values": vals,
                        "name": f"{edge} → {src.name} {ref}", "src": (src, ref)})
    return out


def _compute_base(obj):
    h = features.history(obj)
    if h is None or h["op"] not in OPS:
        return []
    op, ins, p = h["op"], h["inputs"], h["params"]
    out = []
    if op == "square":
        for i, (src, ref) in enumerate(ins):
            vals = _edge_values(src, ref, with_free=True)
            cur = p.get(f"e{i}", "G0")
            if cur not in vals:
                cur = "G0" if "G0" in vals else vals[0]
            out.append({"pos": _mid(src, ref), "param": f"e{i}", "value": cur, "values": vals,
                        "name": f"{src.name} {ref}", "src": (src, ref)})
    elif op == "patch":
        conts = p.get("cont", [])
        tens = p.get("pt_tension", [])
        guides = set(p.get("pt_guides", []))
        for i, (src, ref) in enumerate(ins):
            g = i in guides
            vals = ["G0"] if g else _edge_values(src, ref)          # Patch: G0 / G1 / G2; guides: G0
            cur = "G0" if g else (conts[i] if i < len(conts) else "G0")
            out.append({"pos": _mid(src, ref), "param": f"pc:{i}", "value": cur if cur in vals else vals[0],
                        "values": vals, "name": f"{src.name} {ref}" + (" (guide)" if g else ""), "src": (src, ref),
                        "guide": g, "tension": float(tens[i]) if i < len(tens) and not g else 1.0})
    elif op == "loft":
        for (src, ref), param in ((ins[0], "cont_a"), (ins[-1], "cont_b")):
            vals = _edge_values(src, ref)
            cur = p.get(param, "G0")
            out.append({"pos": _mid(src, ref), "param": param, "value": cur if cur in vals else vals[0],
                        "values": vals, "name": f"{src.name} {ref}", "src": (src, ref)})
    elif op == "loft_rails":
        if p.get("lr_mode", "RAILS") == "CENTRELINE":
            return []
        from . import loft_rails as lr, square as sq
        raw = [sq.sample_side(o, r, 24) for o, r in ins]
        size = float(np.linalg.norm(np.ptp(np.vstack([s.P for s in raw]), axis=0))) or 1.0
        prof_idx, rail_idx = lr.classify(raw, lr.TOUCH_REL * size * 10)
        if len(prof_idx) < 2:
            return []
        if rail_idx:
            hit = [lr._contact(raw[i], raw[rail_idx[0]])[1] for i in prof_idx]
            prof_idx = [prof_idx[k] for k in np.argsort(hit)]
        roles = [(prof_idx[0], "lr_start"), (prof_idx[-1], "lr_end")]
        if len(rail_idx) == 2:
            roles += [(rail_idx[0], "lr_ra"), (rail_idx[1], "lr_rb")]
        for i, param in roles:
            src, ref = ins[i]
            vals = _edge_values(src, ref)
            cur = p.get(param, "G0")
            out.append({"pos": _mid(src, ref), "param": param, "value": cur if cur in vals else vals[0],
                        "values": vals, "name": f"{src.name} {ref}", "src": (src, ref)})
    elif op == "blend_curve" and obj.type == 'CURVE' and obj.data.splines:
        sp = obj.data.splines[0]
        M = obj.matrix_world
        pts = [M @ Vector(q.co[:3]) for q in sp.points] if sp.type != 'BEZIER' else \
            [M @ b.co for b in sp.bezier_points]
        if len(pts) >= 2:
            for (src, ref), param, P in ((ins[0], "cont_a", pts[0]), (ins[1], "cont_b", pts[-1])):
                vals = ["G0", "G1", "G2"]
                cur = p.get(param, "G1")
                out.append({"pos": P, "param": param, "value": cur if cur in vals else "G1", "values": vals,
                            "name": f"{src.name} end"})
    return out


def _patch_eases(obj, out):
    """({label index: ease zones} for a Patch's edge labels (see measure),
    [corner marks]): a mark is {"pos", "text"} at every corner where an edge's
    continuity had to be eased (XNURBS-style)."""
    h = features.history(obj)
    if not h or h.get("op") != "patch" or not any(L["param"].startswith("pc:") for L in out):
        return {}, []
    try:
        from .patch import input_eases
        ins = h["inputs"]
        conts = h["params"].get("cont", [])
        res = input_eases(ins, conts, p=h["params"])
    except Exception as exc:
        print("NURBS patch eases:", exc)
        return {}, []
    names = lambda k: f"{ins[k][0].name} {ins[k][1]}"
    why = lambda k, j, kind, deg: (f"{names(k)} G1 / {names(j)}: tangent planes differ {deg:.2g}°" if kind == "G1"
                                   else f"{names(k)} G2 / {names(j)}: curvatures differ")
    out_, marks = {}, []
    for i, L in enumerate(out):
        if not L["param"].startswith("pc:"):
            continue
        k = int(L["param"][3:])
        lst = res.get(k)
        if lst:
            out_[i] = [(end, f, (f"neighbours' tangent planes differ {deg:.2g}° at the corner with {names(j)}"
                                 if kind == "G1" else f"curvatures differ at the corner with {names(j)}"))
                       for end, f, j, kind, deg, _ in lst]
    for k, lst in res.items():
        for end, f, j, kind, deg, pos in lst:
            near = [m for m in marks if (m["pos"] - Vector(pos)).length < 1e-6 + 1e-4 * (m["pos"].length or 1.0)]
            if near:
                near[0]["text"] += "; " + why(k, j, kind, deg)
                if kind == "G1" and deg is not None:
                    near[0]["deg"] = max(near[0]["deg"] or 0.0, deg)
                continue
            marks.append({"pos": Vector(pos), "deg": deg if kind == "G1" else None, "kind": kind,
                          "text": "Eased corner: " + why(k, j, kind, deg)})
    return out_, marks


_marks = {}          # session_uid -> (key, corner marks)


def corner_marks(obj):
    """[{pos, deg, kind, text}]: the corners of a Patch where continuity is
    eased because no surface can follow both neighbours there."""
    labels(obj)
    hit = _marks.get(obj.session_uid)
    return hit[1] if hit else []


_base_cache = {}
CUT_STEPS = {"edge_fillet", "trim", "fillet_trim"}     # steps that only cut / round the feature's result


def _base_brep(obj):
    """The feature's own result, before its history steps, when those steps
    only cut it (Edge Fillet, trims); None otherwise. Its labels describe that
    feature, so its edges are checked there -- a fillet rounding a loft's
    corners away from its sections is not a G0 failure of the loft. Steps that
    reshape it (Surface Align, Rebuild) are checked on the result, as before.
    The base stored at the last rebuild when there is one, else the feature
    built once (cached per build)."""
    active = [e for e in features.stack(obj) if not e["params"].get("_off")]
    if not active or any(e["op"] not in CUT_STEPS for e in active):
        return None
    from . import store
    text = obj.get(store.PROP_UNTRIM)
    if text is None:
        h = features.history(obj)
        if h is None:
            return None
        key = (obj.session_uid, obj.get(features.PROP_HIST), obj.get("nproto_build_key"))
        if key not in _base_cache:
            g = features.run(h["op"], h["inputs"], h["params"])
            if isinstance(g, tuple):
                g = g[1] if g and g[0] == "brep_text" else None
            elif g is not None:
                g = store.brep_to_text(store.to_brep(g))
            if len(_base_cache) > 32:
                _base_cache.clear()
            _base_cache[key] = g
        text = _base_cache[key]
    return None if text is None else store.brep_from_text(text)


def labels(obj):
    """[{pos, param, value, values, name}] of an object's continuity labels (cached per build)."""
    key = (obj.get("nproto_build_key"), obj.get(features.PROP_HIST), obj.get(features.PROP_STACK),
           tuple(obj.matrix_world.translation) if obj.type == 'CURVE' else None, features.pending(obj), _tols())
    hit = _cache.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1]
    try:
        out = _compute(obj)
    except Exception as exc:
        print("NURBS continuity labels:", obj.name, exc)
        out = []
    eases, marks = _patch_eases(obj, out)
    _marks[obj.session_uid] = (key, marks)
    waiting = features.pending(obj)          # Pause Rebuild: the geometry is older than these settings
    base = None
    if out and obj.type == 'MESH' and not waiting:
        try:
            base = _base_brep(obj)
        except Exception as exc:
            print("NURBS continuity check (base):", obj.name, exc)
    for i, L in enumerate(out):
        L["check"] = None
        L["pending"] = waiting
        if "src" in L and L["value"] != "FREE" and obj.type == 'MESH' and not waiting:
            try:
                skip = guide_skip(obj) if L.get("guide") else None
                own = not str(L.get("param", "")).startswith("stack:")    # the feature's label, not a step's
                L["check"] = measure(obj, *L["src"], ORDER.index(L["value"]) - 1, ease=eases.get(i), skip_near=skip,
                                     brep=base if own else None)
            except Exception as exc:
                print("NURBS continuity check:", obj.name, L["name"], exc)
    _cache[obj.session_uid] = (key, out)
    return out


# --------------------------------------------------------------------------
# What the surface really achieves at an edge
# --------------------------------------------------------------------------

TOL_GAP_REL = 1e-4      # gap, relative to the edge's length
TOL_ANGLE = 0.1         # degrees between the normals
TOL_CURV = 0.05         # normal curvature across the edge, relative to the neighbour's largest


def _tols():
    """Angle / curvature tolerances: the Continuity Check's (Analyze panel), so
    a feature label's red ring and the check agree."""
    v = getattr(bpy.context.window_manager, "nproto_view", None)
    if v is None:
        return TOL_ANGLE, TOL_CURV
    return float(v.cont_angle_tol), float(v.cont_curv_tol) / 100.0


def guide_skip(obj, ref_pts=None):
    """(loop points, distance, loop size): a Patch's guide curves are checked
    only farther than distance from its loop (near an edge the edge's own
    continuity wins, see patch.GUIDE_FADE), their gap against the loop's size."""
    from . import patch as _pt
    h = features.history(obj)
    if h is None or h["op"] != "patch":
        return None
    sides, _, _, _, size = _pt.loop_sides(h["inputs"], h["params"], 16)
    return np.vstack([sd.P for sd in sides]), _pt.GUIDE_FADE * size, size


def measure(obj, src, ref, cont, n=33, ease=None, skip_near=None, brep=None):
    """Gap / normal angle / cross curvature of obj's surface along the input
    (src, ref), sampled away from the very ends: {"gap", "angle", "curv",
    "ok", "text", "eased"}. cont = 0 / 1 / 2: which conditions are checked.
    ease: [(end 0/1, fraction, note)] -- zones toward a corner where the
    feature eased this edge on purpose (no surface can follow both
    neighbours there): angle / curvature there don't count against the edge;
    the result is then ok=False, eased=True when it holds everywhere else.
    skip_near: (points, distance[, size]) -- samples that close to any of the
    points aren't checked at all (a Patch's guide curve where it meets the
    loop); its gap is judged against size when that is longer.
    brep: the geometry to check (default the object's own; labels pass the
    feature's result before its history steps)."""
    from .square import _foot, _second, sample_side
    from . import geom, store
    TOL_ANGLE, TOL_CURV = _tols()
    sd = sample_side(src, ref, n + 4)
    M = np.array(geom.world_matrix(obj), float)
    R, T = M[:3, :3], M[:3, 3]
    Ri = np.linalg.inv(R)
    P = np.asarray(sd.P, float)
    length = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) or 1.0
    brep = brep if brep is not None else store.get_brep(obj)
    faces = list(brep.faces())
    surfs = [f.surface() for f in faces]
    mid = Ri @ (P[len(P) // 2] - T)
    fi_mid = min(range(len(faces)), key=lambda i: faces[i].closest_point(tuple(mid))[1])
    surf, fi_prev = surfs[fi_mid], fi_mid
    gap = ang = ang_e = 0.0
    kerr, kref, kerr_e = [], 0.0, []
    prev = None
    frac = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]) / length
    zone = np.zeros(len(P), bool)
    for end, f, _ in (ease or []):
        zone |= (frac < f) if end == 0 else (frac > 1.0 - f)
    far = np.ones(len(P), bool)
    if skip_near is not None:
        pts_, dist_ = skip_near[0], skip_near[1]
        far = np.min(np.linalg.norm(P[:, None, :] - np.asarray(pts_)[None, :, :], axis=2), axis=1) > dist_
    for k in range(2, len(P) - 2):
        if not far[k]:
            continue
        q = Ri @ (P[k] - T)
        if len(faces) > 1:
            # several faces (a loft split at its folds): the input runs over
            # all of them, each sample is measured on the face it lies on
            fi = min(range(len(faces)), key=lambda i: faces[i].closest_point(tuple(q))[1])
            if fi != fi_prev:
                surf, fi_prev, prev = surfs[fi], fi, None
        prev = _foot(surf, q, prev)
        p_ = np.array(surf.evaluate(*prev), float)
        g_k = float(np.linalg.norm(R @ p_ + T - P[k]))
        if g_k > TOL_GAP_REL * length:
            # following on from the last sample can stop on the wrong side of
            # a fold (a loft through rectangles) or at the far end of a closed
            # surface's seam: start this one afresh from a grid
            alt = _foot(surf, q, None)
            p2 = np.array(surf.evaluate(*alt), float)
            g2 = float(np.linalg.norm(R @ p2 + T - P[k]))
            if g2 < g_k:
                prev, p_, g_k = alt, p2, g2
        gap = max(gap, g_k)
        if cont < 1 or not sd.has_surface:
            continue
        su, sv, suu, suv, svv = (R @ x for x in _second(surf, *prev))
        nrm = np.cross(su, sv)
        nrm /= np.linalg.norm(nrm) or 1.0
        Nt = np.asarray(sd.N[k], float)
        if nrm @ Nt < 0:
            nrm = -nrm
        a_k = float(np.degrees(np.arccos(min(1.0, abs(float(nrm @ Nt))))))
        if zone[k]:
            ang_e = max(ang_e, a_k)
        else:
            ang = max(ang, a_k)
        if cont < 2:
            continue
        # straight across the edge in the tangent plane (sd.B of a natural edge is the
        # neighbour's own, possibly skewed, cross derivative; sd.K belongs to this b)
        t_ = P[k + 1] - P[k - 1]
        b = np.cross(Nt, t_)
        b /= np.linalg.norm(b) or 1.0
        if b @ np.asarray(sd.B[k], float) < 0:
            b = -b
        a, c = np.linalg.lstsq(np.array([su, sv]).T, b, rcond=None)[0]
        k_own = (a * a * (suu @ nrm) + 2 * a * c * (suv @ nrm) + c * c * (svv @ nrm))
        (kerr_e if zone[k] else kerr).append(abs(k_own - float(sd.K[k])))
        kref = max(kref, abs(float(sd.K[k])))
    curv = (max(kerr) / max(kref, 0.05 / length)) if kerr else 0.0
    fails = []
    # a guide curve inside a Patch: relative to the opening, not to itself
    if gap > TOL_GAP_REL * max(length, skip_near[2] if skip_near is not None and len(skip_near) > 2 else 0.0):
        fails.append(f"gap {gap:.3g}")
    if cont >= 1 and sd.has_surface and ang > TOL_ANGLE:
        fails.append(f"angle {ang:.2g}°")
    if cont >= 2 and sd.has_surface and curv > TOL_CURV:
        fails.append(f"curvature {100 * curv:.2g}%")
    parts = [f"gap {gap:.2g}"] + ([f"angle {ang:.2g}°"] if cont >= 1 and sd.has_surface else []) + \
            ([f"curvature {100 * curv:.2g}%"] if cont >= 2 and sd.has_surface else [])
    curv_e = (max(kerr_e) / max(kref, 0.05 / length)) if kerr_e else 0.0
    eased_off = (cont >= 1 and sd.has_surface and ang_e > TOL_ANGLE) or \
        (cont >= 2 and sd.has_surface and curv_e > TOL_CURV)
    if eased_off and not fails:
        notes = "; ".join(nt for _, _, nt in ease)
        what = (f"angle up to {ang_e:.2g}°" if ang_e > TOL_ANGLE else "") + \
            (", " if ang_e > TOL_ANGLE and curv_e > TOL_CURV else "") + \
            (f"curvature up to {100 * curv_e:.2g}%" if curv_e > TOL_CURV else "")
        return {"gap": gap, "angle": max(ang, ang_e), "curv": max(curv, curv_e), "ok": False, "eased": True,
                "text": f"holds except at the corner ({', '.join(parts)}); eased there, {what}: {notes}"}
    return {"gap": gap, "angle": max(ang, ang_e), "curv": max(curv, curv_e), "ok": not fails, "eased": False,
            "text": ("FAILS: " + ", ".join(fails)) if fails else "holds: " + ", ".join(parts)}


def has_labels(obj):
    h = obj.get(features.PROP_HIST)
    if bool(h) and any(f'"op": "{op}"' in str(h) for op in OPS):
        return True
    return '"surface_align"' in str(obj.get(features.PROP_STACK, ""))


def visible_label_objects(context):
    v = getattr(context.window_manager, "nproto_view", None)
    if v is not None and not v.show_cont_labels:
        return []
    if context.mode != 'OBJECT':
        return []
    return [o for o in context.selected_objects if has_labels(o)]


def next_value(values, cur, step):
    if cur not in values:
        return values[0]
    return values[(values.index(cur) + step) % len(values)]


# --------------------------------------------------------------------------
# Operator: step one edge's continuity
# --------------------------------------------------------------------------

class NPROTO_OT_cycle_continuity(bpy.types.Operator):
    """Step this edge's continuity (Free → G0 → G1 → G2); Ctrl or Shift+click steps back"""
    bl_idname = "nproto.cycle_continuity"
    bl_label = "Edge Continuity"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    obj_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    index: IntProperty(default=-1, options={'HIDDEN', 'SKIP_SAVE'})
    step: IntProperty(default=1, options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def description(cls, context, props):
        obj = bpy.data.objects.get(props.obj_name)
        if obj is None:
            return cls.__doc__
        ls = labels(obj)
        if 0 <= props.index < len(ls):
            L = ls[props.index]
            nxt = next_value(L["values"], L["value"], 1)
            prv = next_value(L["values"], L["value"], -1)
            chk = L.get("check")
            state = " -- waiting for Rebuild (paused)" if L.get("pending") else (f" -- {chk['text']}" if chk else "")
            return f"{L['name']}: {SHORT[L['value']]}{state} (click: {SHORT[nxt]}; Ctrl+click: {SHORT[prv]})"
        return cls.__doc__

    def invoke(self, context, event):
        if event.ctrl or event.shift:
            self.step = -1
        return self.execute(context)

    def execute(self, context):
        from . import feature_ui
        obj = bpy.data.objects.get(self.obj_name)
        if obj is None:
            return {'CANCELLED'}
        ls = labels(obj)
        if not 0 <= self.index < len(ls):
            return {'CANCELLED'}
        L = ls[self.index]
        if len(L["values"]) < 2:
            self.report({'INFO'}, f"{L['name']}: a guide curve only holds position (G0)" if L.get("guide")
                        else f"{L['name']} is a curve: it can only be G0")
            return {'CANCELLED'}
        new = next_value(L["values"], L["value"], self.step)
        if L["param"].startswith("pc:"):                 # a Patch edge: its entry in the feature's list
            import json
            i = int(L["param"][3:])
            raw = json.loads(obj[features.PROP_HIST])
            conts = list(raw["params"].get("cont", []))
            while len(conts) <= i:
                conts.append("G0")
            old_raw = obj[features.PROP_HIST]
            conts[i] = new
            raw["params"]["cont"] = conts
            raw["params"].pop("pt_all", None)          # set edge by edge now
            obj[features.PROP_HIST] = json.dumps(raw)
            try:
                features.edit_rebuild(obj, context, force=True)
            except Exception as exc:
                obj[features.PROP_HIST] = old_raw
                self.report({'ERROR'}, f"Continuity: {exc}")
                return {'CANCELLED'}
            try:
                feature_ui.sync_from_history(obj)          # the Feature panel's edge rows follow
            except Exception as exc:
                print("NURBS continuity panel sync:", exc)
            self.report({'INFO'}, f"{obj.name} · {L['name']}: {SHORT[new]}"
                        + ("  (paused: Rebuild builds it)" if features.paused(context) else ""))
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return {'FINISHED'}
        if L["param"].startswith("stack:"):              # a Surface Align step: edit it in place
            _, k, i = L["param"].split(":")
            entries = features.stack(obj)
            entries[int(k)]["params"]["cont"][int(i)] = new
            try:
                features._write_stack(obj, entries)
                features.edit_rebuild(obj, context, force=True)
            except Exception as exc:
                entries[int(k)]["params"]["cont"][int(i)] = L["value"]
                features._write_stack(obj, entries)
                self.report({'ERROR'}, f"Continuity: {exc}")
                return {'CANCELLED'}
            self.report({'INFO'}, f"{obj.name} · {L['name']}: {SHORT[new]}")
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return {'FINISHED'}
        feature_ui.sync_from_history(obj)
        try:
            setattr(obj.nproto_feature, L["param"], new)      # rebuilds through the feature's history
        except Exception as exc:
            self.report({'ERROR'}, f"Continuity: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"{obj.name} · {L['name']}: {SHORT[new]}")
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return {'FINISHED'}


# --------------------------------------------------------------------------
# Gizmos: hit-test the label discs
# --------------------------------------------------------------------------

def _to_px(context, p):
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    return location_3d_to_region_2d(context.region, context.region_data, p)


class NPROTO_GT_cont_label(bpy.types.Gizmo):
    bl_idname = "NPROTO_GT_cont_label"

    __slots__ = ("obj_name", "index")

    def setup(self):
        self.obj_name, self.index = "", -1

    def draw(self, context):
        pass                               # the disc is drawn by draw._draw_px

    def draw_select(self, context, select_id):
        pass

    def test_select(self, context, location):
        obj = bpy.data.objects.get(self.obj_name)
        hit = False
        if obj is not None:
            ls = labels(obj)
            if 0 <= self.index < len(ls):
                q = _to_px(context, ls[self.index]["pos"])
                if q is not None:
                    hit = (Vector(location) - q).length <= RADIUS_PX
        key = (self.obj_name, self.index)
        if hit:
            if hover["key"] != key:
                hover["key"] = key
                context.area.tag_redraw()
            return 0
        if hover["key"] == key:
            hover["key"] = None
            context.area.tag_redraw()
        return -1


CORNER_PX = 8.0


class NPROTO_GT_patch_corner(bpy.types.Gizmo):
    """A Patch's corner / junction dot (drawn by draw._draw_px)."""
    bl_idname = "NPROTO_GT_patch_corner"

    __slots__ = ("obj_name", "index")

    def setup(self):
        self.obj_name, self.index = "", -1

    def draw(self, context):
        pass

    def draw_select(self, context, select_id):
        pass

    def test_select(self, context, location):
        from .patch import corner_points
        obj = bpy.data.objects.get(self.obj_name)
        hit = False
        if obj is not None:
            pts = corner_points(obj)
            if 0 <= self.index < len(pts):
                q = _to_px(context, pts[self.index]["pos"])
                if q is not None:
                    hit = (Vector(location) - q).length <= CORNER_PX
        key = ("corner", self.obj_name, self.index)
        if hit:
            if hover["key"] != key:
                hover["key"] = key
                context.area.tag_redraw()
            return 0
        if hover["key"] == key:
            hover["key"] = None
            context.area.tag_redraw()
        return -1


def _is_patch(obj):
    return obj.get("nproto_patch_info") is not None or '"op": "patch"' in str(obj.get(features.PROP_HIST, ""))


def _corner_count(obj):
    if not _is_patch(obj):
        return 0
    from .patch import corner_points
    return len(corner_points(obj))


class NPROTO_GGT_cont_labels(bpy.types.GizmoGroup):
    bl_idname = "NPROTO_GGT_cont_labels"
    bl_label = "NURBS continuity labels"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'PERSISTENT', 'SHOW_MODAL_ALL'}   # 2D group: hit-tested with test_select

    @classmethod
    def poll(cls, context):
        return bool(visible_label_objects(context))

    def _sync(self, context):
        objs = visible_label_objects(context)
        want = [("L", o.name, i) for o in objs for i in range(len(labels(o)))] + \
            [("C", o.name, i) for o in objs for i in range(_corner_count(o))]
        have = [("C" if isinstance(g, NPROTO_GT_patch_corner) else "L", g.obj_name, g.index)
                for g in self.gizmos]
        if want == have:
            return
        self.gizmos.clear()
        for kind, name, i in want:
            if kind == "L":
                g = self.gizmos.new(NPROTO_GT_cont_label.bl_idname)
                props = g.target_set_operator("nproto.cycle_continuity")
            else:
                g = self.gizmos.new(NPROTO_GT_patch_corner.bl_idname)
                props = g.target_set_operator("nproto.patch_corner")
            g.obj_name, g.index = name, i
            props.obj_name, props.index = name, i
            g.use_draw_modal = False

    def setup(self, context):
        self._sync(context)

    def refresh(self, context):
        self._sync(context)

    def draw_prepare(self, context):
        # refresh only runs on some updates; selection / rebuild changes must
        # still give every visible label its gizmo
        self._sync(context)


classes = (NPROTO_OT_cycle_continuity, NPROTO_GT_cont_label, NPROTO_GT_patch_corner, NPROTO_GGT_cont_labels)
