"""Surface Combs (Alias "Curvature U / V" on isoparms): curvature combs on a
set of iso-curves of every face, along U and / or along V.

Each comb belongs to one iso-curve C(t) = S(u, t) (or S(t, v)) of the face's
surface and is exact: from the surface's first and second derivatives,

    K = (C'' - (C'' . T) T) / |C'|^2,    T = C' / |C'|

the curve's curvature vector (pointing to its centre of curvature); a hair
goes from the curve away from that centre, as on a curve comb. Trimmed faces
show only the parts of the iso-curves inside them (Face.classify_uv). The
scale is per object, so the longest hairs are about a fifth of its size at
Comb Scale 1, and stays put while you work on it.
"""

import numpy as np

import bpy
from mathutils import Vector

from . import geom, store

from .diagnostics import COMB_ENV_W, COMB_ENVELOPE, COMB_HAIR, COMB_HAIR_W, COMB_SPAN  # noqa: E402

COL_ISO = (0.85, 0.85, 0.85, 0.5)

_cache = {}      # object name -> (key, runs, k95, size)


def _iso_runs(face, R, T, along, k, n, samples):
    """The k-th of n iso-curves of the face running `along` U or V: runs of
    (world points, world curvature vectors) inside the trimmed face."""
    from .square import _second
    surf = face.surface()
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    f = (k + 1) / (n + 1)
    if along == "U":
        vc = v0 + (v1 - v0) * f
        uvs = [(u, vc) for u in np.linspace(u0, u1, samples)]
    else:
        uc = u0 + (u1 - u0) * f
        uvs = [(uc, v) for v in np.linspace(v0, v1, samples)]
    runs, cur = [], []
    for uv in uvs:
        try:
            inside = face.classify_uv(*uv) != "outside"
        except Exception:
            inside = True
        if not inside:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
            continue
        try:
            su, sv, suu, suv, svv = (R @ np.asarray(x, float) for x in _second(surf, *uv))
            p = R @ np.asarray(surf.evaluate(*uv), float) + T
        except Exception:
            continue
        d1, d2 = (su, suu) if along == "U" else (sv, svv)
        sp = float(np.linalg.norm(d1))
        if sp < 1e-14:                 # a pole: no curve speed, no curvature
            continue
        t = d1 / sp
        K = (d2 - (d2 @ t) * t) / (sp * sp)
        cur.append((p, K))
    if len(cur) > 1:
        runs.append(cur)
    return [(np.array([p for p, _ in r]), np.array([K for _, K in r])) for r in runs]


def compute(obj, v):
    key = (obj.get(store.PROP_REV), obj.get("nproto_build_key"), tuple(round(x, 9) for r in obj.matrix_world for x in r),
           v.iso_along_u, v.iso_along_v, v.iso_count_u, v.iso_count_v, v.iso_samples)
    hit = _cache.get(obj.name)
    if hit and hit[0] == key:
        return hit[1:]
    M = np.array(geom.world_matrix(obj), float)
    R, T = M[:3, :3], M[:3, 3]
    brep = store.get_brep(obj)
    runs = []
    for face in brep.faces():
        for along, on, n in (("U", v.iso_along_u, v.iso_count_u), ("V", v.iso_along_v, v.iso_count_v)):
            if not on:
                continue
            for k in range(n):
                runs += _iso_runs(face, R, T, along, k, n, v.iso_samples)
    lo, hi = (np.array(x, float) for x in brep.bounding_box())
    size = float(np.linalg.norm((R @ (hi - lo)))) or 1.0
    ks = np.concatenate([np.linalg.norm(K, axis=1) for _, K in runs]) if runs else np.zeros(0)
    k95 = float(np.percentile(ks, 95)) if len(ks) else 0.0
    _cache[obj.name] = (key, runs, k95, size)
    return runs, k95, size


def _v(p):
    return Vector((float(p[0]), float(p[1]), float(p[2])))


def draw_view(ctx, draw_lines):
    v = getattr(ctx.window_manager, "nproto_view", None)
    if v is None or not v.show_iso_combs:
        return
    import gpu
    objs = [o for o in (ctx.selected_objects if v.diag_selected_only else ctx.visible_objects)
            if store.is_nurbs(o) and o.visible_get()]
    gpu.state.depth_test_set('LESS_EQUAL')
    for obj in objs:
        try:
            runs, k95, size = compute(obj, v)
        except Exception as exc:
            print("NURBS surface combs:", obj.name, exc)
            continue
        fac = v.iso_scale * COMB_SPAN * size / k95 if k95 * size > 1e-9 else 0.0
        iso, hairs, env = [], [], []
        for P, K in runs:
            tips = P - K * fac
            for i in range(len(P)):
                hairs.extend([_v(P[i]), _v(tips[i])])
            for i in range(len(P) - 1):
                iso.extend([_v(P[i]), _v(P[i + 1])])
                env.extend([_v(tips[i]), _v(tips[i + 1])])
        draw_lines(iso, COL_ISO, 1.0)
        draw_lines(hairs, COMB_HAIR, COMB_HAIR_W)
        draw_lines(env, COMB_ENVELOPE, COMB_ENV_W)
