"""Blend Curve: CV edits keep the continuity at both ends.

A blend curve has no free CVs (degree = 1 + kA + kB): side A fixes its first
kA + 1 CVs, side B its last kB + 1. Within the continuity a side can only
change three numbers (geom.blend_side_cvs): Weight (P1 along the tangent),
Curvature (P2 along the tangent; its height comes from the weight) and, for
G3, Rate (P3 along the tangent). So while the curve is in Edit Mode a
depsgraph handler turns whatever the user did to the CVs (G / R / S, snapping,
typed values) into those numbers -- a least-squares fit to the CVs that moved,
the others kept where they were as far as the fit leaves them free -- and puts
every CV back on the result. Moving P2 off its line slides P1 to give it that
height; the end points stay on their curves.

The numbers go into the feature's history (and its Feature panel), so the
edit is part of the blend: it survives a rebuild when an input curve moves.
A curve whose CV count or order was changed in Edit Mode is no longer a blend
and is left alone. Delete History edits it freely.

Align Curve's ends get the same: an aligned curve's own CVs are its state
(the curve is the feature's first input), so only the end CVs are held --
each end's free numbers (align.EndShape: tangent length and the tangential
part of C'' at an edge; tangent and C'' within the plane at a surface) are
fitted to the CVs, the selected ones (being moved) first. The inner CVs stay
free, and leaving Edit Mode rebuilds nothing new (align.solve of an aligned
curve gives it back).
"""

import json

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import features, geom, store

OP = "blend_curve"
_busy = [False]
_frames = {}         # object name -> (key, frame A, frame B): the input ends, for the current inputs
_WMIN = 0.02         # weight can not reach 0 (P1 on P0: no tangent left to keep)
_KEEP = 1e-6         # pull of the CVs the user did not move, relative to the ones moved


def is_blend(obj):
    if obj is None or obj.type != 'CURVE' or features.PROP_HIST not in obj:
        return False
    return f'"{OP}"' in str(obj[features.PROP_HIST])


def _inputs_key(h):
    """What the end frames depend on: the inputs (shape, placement) and which end of each, flipped or not."""
    from .diagnostics import _curve_key
    p = h["params"]
    parts = [json.dumps([r for _, r in h["inputs"]]),
             json.dumps([p.get(k) for k in ("end_a", "end_b", "flip_a", "flip_b", "cont_a", "cont_b") + features.BLEND_SIDE_KEYS])]
    for src, _ in h["inputs"]:
        parts.append(src.name)
        parts.append(np.round(np.array(geom.world_matrix(src), float), 10).tobytes().hex())
        if store.is_nurbs(src):
            parts.append(str(hash(str(src.get(store.PROP_BREP, "")))))
        elif src.type == 'CURVE':
            parts.append(str(_curve_key(src)))
    return "|".join(parts)


def _end_frames(obj, h):
    key = _inputs_key(h)
    hit = _frames.get(obj.name)
    if hit is not None and hit[0] == key:
        return hit[1], hit[2]
    fa, fb = features.blend_frames(h["inputs"], h["params"])
    _frames[obj.name] = (key, fa, fb)
    return fa, fb


def _side_x(p, s):
    return np.array([float(p.get(f"w_{s}", 1.0)), float(p.get(f"c_{s}", 0.0)), float(p.get(f"d_{s}", 0.0))])


def side_rows(frame, k, n, L, x):
    return geom.side_cvs(frame, k, n, L, *x)


def lm(res, x0, clamp=lambda x: x):
    """Levenberg-Marquardt on a few numbers: x minimising |res(x)|^2 from x0,
    kept in bounds by clamp (numeric Jacobian; the residuals are low-degree
    polynomials in x)."""
    xk = clamp(np.array(x0, float))
    if not len(xk):
        return xk
    r = res(xk)
    f = r @ r
    mu = 1e-3
    for _ in range(80):
        J = np.empty((len(r), len(xk)))
        for j in range(len(xk)):
            h = 1e-7 * (1.0 + abs(xk[j]))
            e = np.zeros(len(xk))
            e[j] = h
            J[:, j] = (res(xk + e) - r) / h
        A, g = J.T @ J, J.T @ r
        stepped = False
        while mu < 1e10:
            xn = clamp(xk + np.linalg.solve(A + mu * np.diag(np.diag(A) + 1e-12), -g))
            rn = res(xn)
            fn = rn @ rn
            if fn < f:
                stepped = abs(f - fn) > 1e-20 and float(np.abs(xn - xk).max()) > 1e-13
                xk, r, f = xn, rn, fn
                mu = max(mu / 3.0, 1e-12)
                break
            mu *= 4.0
        if not stepped:
            break
    return xk


def fit_side(frame, k, n, L, x0, target, moved):
    """Weight / Curvature / Rate (x, as many as the side has: k) that bring the
    side's CVs 1..k nearest the target: the moved ones first, the others
    (pulled only by _KEEP) where the moved ones leave freedom."""
    x0 = np.asarray(x0, float)
    if k == 0 or not moved:
        return x0
    wt = np.sqrt(np.array([1.0 if i in moved else _KEEP for i in range(1, k + 1)]))
    tgt = [np.asarray(target[i], float) for i in range(1, k + 1)]
    scale = 1.0 / (L or 1.0)

    def full(xk):
        x = x0.copy()
        x[:k] = xk
        return x

    def clamp(xk):
        xk = np.array(xk, float)
        xk[0] = max(xk[0], _WMIN)
        return xk

    def res(xk):
        rows = side_rows(frame, k, n, L, full(xk))
        r = [wt[i - 1] * (rows[i] - tgt[i - 1]) * scale for i in range(1, k + 1)]
        return np.concatenate(r + [1e-6 * (xk - x0[:k])])

    return full(lm(res, x0[:k], clamp))


def _world_cvs(obj, sp):
    co = np.empty(len(sp.points) * 4)
    sp.points.foreach_get("co", co)
    M = np.array(geom.world_matrix(obj), float)
    loc = co.reshape(-1, 4)[:, :3]
    return loc @ M[:3, :3].T + M[:3, 3]


def constrain(obj):
    """Put a blend curve's CVs back on its continuity after an edit; the fit
    becomes the feature's Weight / Curvature / Rate. -> True if it changed something."""
    h = features.history(obj)
    if h is None or h["op"] != OP or len(h["inputs"]) != 2:
        return False
    p = h["params"]
    ka, kb = geom.CONT_ORDER[p.get("cont_a", "G1")], geom.CONT_ORDER[p.get("cont_b", "G1")]
    n = 1 + ka + kb
    cu = obj.data
    if len(cu.splines) != 1:
        return False
    sp = cu.splines[0]
    if sp.type != 'NURBS' or len(sp.points) != n + 1 or sp.order_u != n + 1:
        return False                      # no longer a blend (points added / removed): left as edited
    W = _world_cvs(obj, sp)
    fa, fb = _end_frames(obj, h)
    L = float(np.linalg.norm(np.asarray(fb[0], float) - np.asarray(fa[0], float))) or 1.0
    tol = 2e-6 * (L + float(np.abs(W).max()))      # Blender keeps CVs in single precision
    xs = {"a": _side_x(p, "a"), "b": _side_x(p, "b")}
    sides = {"a": (fa, ka, [W[i] for i in range(ka + 1)]),
             "b": (fb, kb, [W[n - i] for i in range(kb + 1)])}
    new_x, changed = {}, False
    for s, (frame, k, tgt) in sides.items():
        base = side_rows(frame, k, n, L, xs[s])
        off = [i for i in range(k + 1) if float(np.linalg.norm(tgt[i] - base[i])) > tol]
        if not off:
            new_x[s] = xs[s]
            continue
        changed = True
        moved = {i for i in off if i > 0}            # P0 is the input's end: it only snaps back
        target = [tgt[i] if i in moved else base[i] for i in range(k + 1)]
        new_x[s] = fit_side(frame, k, n, L, xs[s], target, moved)
    if not changed:
        return False
    rows_a = side_rows(fa, ka, n, L, new_x["a"])
    rows_b = side_rows(fb, kb, n, L, new_x["b"])
    P = np.array(rows_a + rows_b[::-1], float)
    Mi = np.linalg.inv(np.array(geom.world_matrix(obj), float))
    loc = P @ Mi[:3, :3].T + Mi[:3, 3]
    co = np.c_[loc, np.ones(len(loc))].ravel()
    sp.points.foreach_set("co", co)
    cu.update_tag()
    # the fit is the blend's own setting now
    raw = json.loads(obj[features.PROP_HIST])
    for s in "ab":
        k = ka if s == "a" else kb
        for j, name in enumerate(("w", "c", "d")[:k]):
            raw["params"][f"{name}_{s}"] = round(float(new_x[s][j]), 9)
    obj[features.PROP_HIST] = json.dumps(raw)
    try:
        obj[features.PROP_KEY] = features.build_key(obj)
    except Exception:
        pass
    try:
        from .feature_ui import sync_from_history
        sync_from_history(obj)
    except Exception:
        pass
    return True


# --------------------------------------------------------------------------
# Align Curve: the aligned ends' CVs (the rest of the curve is free)
# --------------------------------------------------------------------------

ALIGN = "align_curve"


def is_aligned(obj):
    if obj is None or obj.type != 'CURVE' or features.PROP_HIST not in obj:
        return False
    return f'"{ALIGN}"' in str(obj[features.PROP_HIST])


def _write_local(sp, Q):
    co = np.c_[np.asarray(Q, float), np.ones(len(Q))].ravel()
    sp.points.foreach_set("co", co)


def constrain_align(obj):
    """Put an aligned curve's end CVs back on their alignment after an edit.
    Each end's free numbers (align.EndShape) are fitted to its CVs: the
    selected ones (what the user moves) first, the others kept where the fit
    leaves freedom. A NEAREST end slides along its edge with the end CV; an
    END end snaps back. -> True if it changed something."""
    from . import align
    h = features.history(obj)
    if h is None or h["op"] != ALIGN:
        return False
    ends, inputs = h["params"].get("ends", {}), h["inputs"]
    si = align.spline_index(inputs)
    if not ends or len(obj.data.splines) <= si:
        return False
    kind, p, Q, U = align.curve_state(obj, si)
    if kind == "REFIT":
        return False
    need = [align.CONT[ends[e].get("cont", "G1")] for e in ("0", "1") if e in ends]
    k_need = sum(need) + len(need)
    M = np.array(geom.world_matrix(obj), float)
    Mi = np.linalg.inv(M)
    size = float(np.linalg.norm(np.ptp(Q, axis=0))) or 1.0
    tol = 2e-6 * (size + float(np.abs(Q).max()))
    if kind == "BEZIER":
        # handles: put the ends back as a rebuild would (no fit)
        kind2, p2, Q2, _ = align.solve(obj, ends, inputs)
        if kind2 != "BEZIER" or len(Q2) != len(Q) or float(np.abs(Q2 - Q).max()) <= tol:
            return False
        align.write(obj, kind2, p2, Q2, si)
        return True
    if len(Q) < max(k_need, 2) or p < max(need + [1]) + (1 if 2 in need else 0) or (p == 1 and max(need) >= 1) \
            or (3 in need and p < 3):
        return False                                      # too few CVs / too low a degree now: left as edited
    sp = obj.data.splines[si]
    sel = np.array([q.select for q in sp.points], bool)
    Qn, changed = Q.copy(), False
    for e in ("0", "1"):
        if e not in ends:
            continue
        spec = ends[e]
        k = align.CONT[spec.get("cont", "G1")]
        seg, Ub = (Qn, U) if e == "0" else align._reverse(Qn, U)
        sseg = sel if e == "0" else sel[::-1]
        src, ref = inputs[int(spec["input"])]
        near = M[:3, :3] @ seg[0] + M[:3, 3]
        fw = align.end_frame(src, ref, near, spec.get("snap", "END"), spec.get("mode", "EDGE"),
                             spec.get("flip", False), spec.get("face", 0))
        shape = align.EndShape(Ub, p, k, spec.get("mode", "EDGE"), fw, Mi[:3, :3], Mi[:3, 3], size)
        x0 = shape.numbers(seg)
        on = shape.rows(x0, seg)
        if max(float(np.linalg.norm(on[i] - seg[i])) for i in range(k + 1)) <= tol:
            continue
        wt = np.sqrt(np.array([1.0 if sseg[i] else _KEEP for i in range(1, k + 1)]))
        tgt = [seg[i].copy() for i in range(k + 1)]

        def res(x, shape=shape, seg=seg, wt=wt, tgt=tgt, k=k, x0=x0):
            rows = shape.rows(x, seg)
            r = [wt[i - 1] * (rows[i] - tgt[i]) / size for i in range(1, k + 1)]
            return np.concatenate(r + [1e-6 * (x - x0)])

        x = lm(res, x0, shape.clamp) if k else x0
        rows = shape.rows(x, seg)
        seg = seg.copy()
        for i in range(k + 1):
            seg[i] = rows[i]
        Qn = seg if e == "0" else seg[::-1].copy()
        changed = True
    if not changed:
        return False
    _write_local(sp, Qn)
    obj.data.update_tag()
    return True


@persistent
def _on_depsgraph(scene, depsgraph):
    if _busy[0]:
        return
    curves = set()
    for u in depsgraph.updates:
        idb = u.id
        if isinstance(idb, bpy.types.Curve):
            curves.add(idb.original.name)
        elif isinstance(idb, bpy.types.Object) and idb.type == 'CURVE' and u.is_updated_geometry:
            curves.add(idb.original.data.name)
    if not curves:
        return
    _busy[0] = True
    try:
        for obj in scene.objects:
            if obj.type != 'CURVE' or obj.mode != 'EDIT' or obj.data.name not in curves:
                continue
            try:
                if is_blend(obj):
                    constrain(obj)
                elif is_aligned(obj):
                    constrain_align(obj)
            except Exception as exc:
                print(f"NURBS constrained CV edit ({obj.name}): {exc}")
    finally:
        _busy[0] = False


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    _frames.clear()
