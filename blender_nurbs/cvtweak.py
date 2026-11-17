"""CV Tweak: CV edits kept in the construction history.

Done in the CV cage (Edit CVs, Nudge CVs) on a surface that has history
adds a "cv_tweak" step instead of removing the history. The step stores

- each moved CV's offset in that CV's own frame: along its U hull line,
  across it in the tangent plane, and along its normal. Replayed on a
  surface that has changed (its inputs moved), the offsets turn and lean
  with the surface, so the same push stays a push "out of" the surface.
- the moved CVs that sat on their continuity guide (a neighbour's edge
  point, G1 line or G2 plane, as Nudge CVs shows them) when Done was
  pressed. The neighbour becomes an input of the step: when it changes,
  those CVs are put back onto its guides, so the edge keeps its G0 / G1 /
  G2 with it.

A tweak needs the surface's CV layout as it was (counts and degrees): if
an earlier step changes it, the step reports that and stops the rebuild
(mute or delete the step in the history list).
"""

import json

import numpy as np
from mathutils import Vector

from . import cvnudge, cvs, features, geom, jobs, match, store

OP = "cv_tweak"


# --------------------------------------------------------------------------
# Frames and offsets (pure)
# --------------------------------------------------------------------------

def local_frames(G):
    """(T1, T2, N), each (nu * nv, 3): an orthonormal frame per CV -- T1
    along its U hull line, N its normal (Nudge CVs' N), T2 = N x T1."""
    f = cvnudge.frames(G)
    N = f["N"].reshape(-1, 3)
    U = f["U"].reshape(-1, 3)
    T1 = U - np.einsum("ij,ij->i", U, N)[:, None] * N
    bad = np.linalg.norm(T1, axis=1) < 1e-9          # U along N (a degenerate hull): any tangent
    if bad.any():
        alt = np.where(np.abs(N[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        T1[bad] = np.cross(N[bad], alt[bad])
    T1 = cvnudge._unit(T1)
    T2 = np.cross(N, T1)
    return T1, T2, N


def offsets(X0, X1, shape):
    """[[k, a, b, c]] for every CV k that moved from X0 to X1 (flat (n, 3),
    one space): the move in the frame of X0 (a T1 + b T2 + c N)."""
    X0, X1 = np.asarray(X0, float), np.asarray(X1, float)
    T1, T2, N = local_frames(X0.reshape(shape))
    out = []
    for k in np.nonzero(np.any(X1 != X0, axis=1))[0]:
        d = X1[k] - X0[k]
        out.append([int(k), float(d @ T1[k]), float(d @ T2[k]), float(d @ N[k])])
    return out


def moved(X, moves, shape):
    """X (flat) with the recorded offsets applied in X's own frames."""
    X = np.array(X, float)
    T1, T2, N = local_frames(X.reshape(shape))
    Y = X.copy()
    for k, a, b, c in moves:
        Y[k] = X[k] + a * T1[k] + b * T2[k] + c * N[k]
    return Y


# --------------------------------------------------------------------------
# The surface in a brep text
# --------------------------------------------------------------------------

def _ext():
    return store._k()[1]


def surface_of(text):
    """(cvs dict of the kernel, natural): the one surface the brep's faces
    lie on and whether the brep is that whole surface (no trim)."""
    from . import steps
    b = store.brep_from_text(text)
    ext = _ext()
    nets = [ext.surface_cvs(f.surface()) for f in b.faces()]
    a = nets[0]
    for o in nets[1:]:
        if (o["count_u"], o["count_v"]) != (a["count_u"], a["count_v"]) or \
                np.abs(np.asarray(o["points"], float) - np.asarray(a["points"], float)).max() > 1e-9:
            raise RuntimeError("CV Tweak: the surface is now several different surfaces")
    return a, steps.natural_surface(b) is not None


def _rational(d):
    w = d.get("weights") or []
    return bool(w) and max(abs(x - 1.0) for x in w) > 1e-12


def text_with_cvs(obj, text, d, natural, pts):
    """The brep text with its surface's CVs replaced by pts (local); trims
    stay where they are in UV (the CV cage's rule)."""
    meta = {"degree_u": d["degree_u"], "degree_v": d["degree_v"], "knots_u": list(d["knots_u"]),
            "knots_v": list(d["knots_v"]), "weights": list(d["weights"] or [])}
    pts = [tuple(map(float, p)) for p in pts]
    if natural:
        ext = _ext()
        s = ext.surface_from_cvs(meta["degree_u"], meta["degree_v"], pts, meta["knots_u"], meta["knots_v"],
                                 meta["weights"] or None)
        return store.brep_to_text(ext.brep_from_surface(s))
    _, breps = jobs.run("retarget", {"target": (text, jobs.xform(obj))}, ["result", "untrimmed"],
                        meta=meta, cvs=[list(p) for p in pts])
    return breps["result"]


# --------------------------------------------------------------------------
# Guides: CVs that stay on a neighbour's G0 / G1 / G2 guide
# --------------------------------------------------------------------------

def _world(obj):
    M = np.array(geom.world_matrix(obj), float)
    return M[:3, :3], M[:3, 3]


def world_net(X, d, shape, A, t):
    """match.Net (world) of local CVs X."""
    G = (np.asarray(X, float) @ A.T + t).reshape(shape)
    return match.Net(G, (match._norm(d["knots_u"]), match._norm(d["knots_v"])), (d["degree_u"], d["degree_v"]))


def on_guides(context, surf, X0w, X1w, net0, ks, exclude=()):
    """[(edge, (partner, ref), level, [CVs])]: of the CVs ks (moved), those
    on their guide in X1w, grouped by the edge whose guide it is. Partners
    built from surf are left out (they follow it already)."""
    partners = cvnudge.find_partners(context, surf, net0, exclude)
    partners = {e: p for e, p in partners.items()
                if p.get("side") is not None and not features.depends_on(p["obj"], surf)}
    if not partners:
        return []
    gd = cvnudge.Guided(net0, partners)
    tol = cvnudge.tolerances(context)
    best = cvnudge.first_guides(gd.guides(X1w, 2))
    out = {}
    for k in ks:
        g = best.get(k)
        if g is None or not cvnudge.holds(g, tol):
            continue
        out.setdefault(g["edge"], []).append((int(k), g["row"]))
    res = []
    for edge, kr in out.items():
        p = partners[edge]
        res.append((edge, (p["obj"], p["ref"]), max(r for _, r in kr), sorted(k for k, _ in kr)))
    return res


def put_back(net_in, Xw, holds, inputs):
    """Xw (world) with each hold's CVs put onto the guides of its partner as
    the partner is now (row 0, row 1, then row 2 together). Holds whose
    partner no longer fits the layout are left as they are; returns
    (Xw, [edges left])."""
    left = []
    for h, (tobj, tref) in zip(holds, inputs):
        e = cvnudge.partner_of(net_in, h["edge"], tobj, tref)
        if e.get("side") is None:
            left.append(h["edge"])
            continue
        gd = cvnudge.Guided(net_in, {h["edge"]: e})
        Xw = cvnudge.put_on(gd, Xw, set(h["ks"]), min(int(h["level"]), e["max"]))
    return Xw, left


# --------------------------------------------------------------------------
# History step
# --------------------------------------------------------------------------

LAST_NOTES = []      # what the last replay could not do (a partner that no longer fits)


def apply_cv_tweak(obj, text, inputs, p):
    d, natural = surface_of(text)
    nu, nv = d["count_u"], d["count_v"]
    if (nu, nv) != (p["nu"], p["nv"]) or [d["degree_u"], d["degree_v"]] != list(p["deg"]):
        raise RuntimeError(f"CV Tweak: the surface's CV layout changed ({p['nu']} x {p['nv']}, degree "
                           f"{p['deg'][0]} x {p['deg'][1]} -> {nu} x {nv}, degree {d['degree_u']} x "
                           f"{d['degree_v']}): mute or delete the step")
    shape = (nu, nv, 3)
    X = np.asarray(d["points"], float)
    Y = moved(X, p["moves"], shape)
    holds = p.get("holds") or []
    LAST_NOTES.clear()
    if holds and not _rational(d):
        A, t = _world(obj)
        Yw, left = put_back(world_net(X, d, shape, A, t), Y @ A.T + t, holds, inputs)
        Y = (Yw - t) @ np.linalg.inv(A).T
        if left:
            LAST_NOTES.append(f"{obj.name}: CV Tweak: the guides of {', '.join(left)} no longer fit")
    return text_with_cvs(obj, text, d, natural, Y)


def label(p):
    n = len(p.get("moves", []))
    s = f"CV Tweak · {n} CV{'s' if n != 1 else ''}"
    held = sum(len(h["ks"]) for h in p.get("holds") or [])
    if held:
        s += " · " + ", ".join(f"{h['edge']} G{h['level']}" for h in p["holds"])
    return s


def record(surf, cage, context=None):
    """Add the cage's edit to surf's history as a CV Tweak step. Returns its
    params, or None when no CV moved."""
    meta = json.loads(cage[cvs.CAGE_META])
    shape = (meta["nu"], meta["nv"], 3)
    X0 = np.asarray(json.loads(cage[cvs.CAGE_CVS0]), float)
    M = geom.world_matrix(surf).inverted() @ geom.world_matrix(cage)
    pts, _ = cvs._snap(cage, [tuple(M @ Vector(q)) for q in cvs.cage_points(cage)])
    X1 = np.asarray(pts, float)
    moves = offsets(X0, X1, shape)
    if not moves:
        return None
    holds, inputs = [], []
    d = {"knots_u": meta["knots_u"], "knots_v": meta["knots_v"], "degree_u": meta["degree_u"],
         "degree_v": meta["degree_v"], "weights": meta.get("weights")}
    if not _rational(d):
        try:
            import bpy
            A, t = _world(surf)
            net0 = world_net(X0, d, shape, A, t)
            for edge, inp, level, ks in on_guides(context or bpy.context, surf, X0 @ A.T + t, X1 @ A.T + t, net0,
                                                  [m[0] for m in moves], (cage,)):
                holds.append({"edge": edge, "level": int(level), "ks": ks})
                inputs.append(inp)
        except Exception as exc:
            print("NURBS CV Tweak: guides not kept:", exc)
            holds, inputs = [], []
    params = {"nu": meta["nu"], "nv": meta["nv"], "deg": [meta["degree_u"], meta["degree_v"]],
              "moves": moves, "holds": holds}
    features.push_step(surf, OP, inputs, params, base_text=cage[cvs.CAGE_ORIG])
    return params


def keep_edit(surf, cage, context=None):
    """After a CV edit was kept on a surface with history: record it as a
    step. -> "step" (recorded), "cleared" (it could not be: history removed,
    as before CV Tweak) or None (nothing moved)."""
    try:
        return "step" if record(surf, cage, context) is not None else None
    except Exception as exc:
        print(f"NURBS CV Tweak: {surf.name}: not recorded ({exc}); construction history removed")
        features.clear_history(surf)
        return "cleared"


def _register_step():
    from . import steps
    steps.APPLY[OP] = apply_cv_tweak
    steps.LABEL[OP] = label
    features.KEEP_STEPS.add(OP)      # plain stays plain, trimmed stays trimmed


_register_step()


def register():
    from . import history_ui
    history_ui.STEP_ICON[OP] = 'VERTEXSEL'
    history_ui.STEP_NOTE[OP] = "Edit CVs again (Tab) to add another tweak"


def unregister():
    pass
