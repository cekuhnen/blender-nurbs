"""Set Origin and Apply Transform on NURBS objects.

Blender's Set Origin (and Apply Location / Rotation / Scale, Ctrl+A) moves a
mesh object's vertices and changes its matrix so that it stays where it is.
A NURBS object's mesh is only the display of its B-rep (local space), which
Blender knows nothing about: left alone, everything that reads the B-rep
(Loft, Blend, Trim ... built from it) sees it moved by the origin's offset,
and its next rebuild puts it there too.

So every mesh write remembers a few vertices and the world matrix
(PROP_SIG). When a NURBS object's vertices are found moved by exactly the
inverse of its matrix change (the object stayed where it was in the world),
its B-rep texts are moved the same way, and a feature's placement
(features.PROP_PLACE) takes the change, so its rebuilds stay put too.
"""

import json

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import geom, store

PROP_SIG = "nproto_msig"   # JSON {"n", "i": vertex indices, "p": their local coords, "m": world matrix, "s": size}


def _matrix(obj):
    return np.array(geom.world_matrix(obj), float)


def remember(obj):
    """Note the mesh just written (a few vertices) and the matrix it was written under."""
    me = obj.data
    n = len(me.vertices)
    if n == 0:
        if PROP_SIG in obj:
            del obj[PROP_SIG]
        return
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    size = float(np.linalg.norm(co.max(axis=0) - co.min(axis=0)))
    obj[PROP_SIG] = json.dumps({"n": n, "i": idx, "p": co[idx].astype(float).tolist(),
                                "m": _matrix(obj).ravel().tolist(), "s": size})


def _transform_text(text, A):
    b = store.brep_from_text(text)
    store.transform(b, A)
    return store.brep_to_text(b)


def apply(obj, A):
    """obj's mesh was moved by A (local space, 4x4) with the object staying in
    place: move its B-rep (and the texts it keeps for Untrim / its history
    base) the same way, and its feature's placement."""
    from . import features
    hist = features.has_history(obj)
    C = features.placement(obj)
    if hist:     # was it built from its inputs as they are? (not waiting for a rebuild)
        s = json.loads(obj[PROP_SIG])
        M0 = np.array(s["m"], float).reshape(4, 4)
        built = obj.get(features.PROP_KEY) == features.build_key(obj, matrix=M0.tolist(), place=C)
    texts = {}
    for k in (store.PROP_BREP, store.PROP_UNTRIM, features.PROP_BASE):
        if isinstance(obj.get(k), str):
            texts[k] = _transform_text(obj[k], A)        # all of them first: a failure changes nothing
    for k, t in texts.items():
        obj[k] = t
    brep = store.brep_from_text(obj[store.PROP_BREP])
    obj[store.PROP_REV] = store._new_rev(obj)
    obj[store.PROP_DISP] = json.dumps(store._display_data(brep))
    store._cache[obj.session_uid] = (obj[store.PROP_REV], brep)
    if features.history(obj) is not None:
        features.set_placement(obj, A if C is None else A @ C)
    if hist and built:
        obj[features.PROP_KEY] = features.build_key(obj)    # what it holds is still what a rebuild gives
    remember(obj)                                              # (the mesh Blender moved shows it already)


def check(obj):
    """Has obj's mesh been moved under it by Set Origin / Apply Transform?
    Then follow with its B-rep. -> True when it did."""
    if not store.is_nurbs(obj) or obj.mode == 'EDIT':
        return False
    raw = obj.get(PROP_SIG)
    if not raw:
        return False
    try:
        s = json.loads(raw)
    except ValueError:
        return False
    me = obj.data
    if len(me.vertices) != s["n"]:
        remember(obj)                   # some other mesh: start over from it
        return False
    P = np.array([tuple(me.vertices[i].co) for i in s["i"]], float)
    P0 = np.array(s["p"], float)
    M = _matrix(obj)
    M0 = np.array(s["m"], float).reshape(4, 4)
    tol = 1e-5 * max(s["s"], 1e-9)
    if np.abs(P - P0).max() <= tol:
        if np.abs(M - M0).max() > 1e-12:        # only moved (G / R / S): the new matrix is the reference
            s["m"] = M.ravel().tolist()
            obj[PROP_SIG] = json.dumps(s)
        return False
    try:
        A = np.linalg.solve(M, M0)               # local change that keeps the world place: M @ A = M0
    except np.linalg.LinAlgError:
        remember(obj)
        return False
    Q = P0 @ A[:3, :3].T + A[:3, 3]
    scale = max(float(np.abs(A[:3, :3]).max()), 1.0)
    if np.abs(Q - P).max() > tol * scale + 1e-6 * float(np.abs(Q).max()):   # (the mesh is float32)
        remember(obj)                   # changed some other way: nothing to follow
        return False
    try:
        apply(obj, A)
    except Exception as exc:
        print(f"NURBS: {obj.name}: could not follow the origin / transform change: {exc}")
        remember(obj)
        return False
    return True


def check_all(objs):
    return [o.name for o in objs if check(o)]


@persistent
def _on_depsgraph(scene, depsgraph):
    names = set()
    for u in depsgraph.updates:
        idb = u.id
        if isinstance(idb, bpy.types.Object) and (u.is_updated_geometry or u.is_updated_transform):
            names.add(idb.original.name)
    objs = [o for o in (scene.objects.get(n) for n in names) if o is not None and store.is_nurbs(o)]
    if objs:
        check_all(objs)


@persistent
def _on_undo(*_):
    # an undo / redo step saved right after Set Origin (before the handler above
    # ran) holds the moved mesh with the old B-rep: follow it again
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        check_all([o for o in scene.objects if store.is_nurbs(o)])


_HANDLERS = (("depsgraph_update_post", _on_depsgraph), ("undo_post", _on_undo), ("redo_post", _on_undo),
             ("load_post", _on_undo))


def register():
    for lst, fn in _HANDLERS:
        hs = getattr(bpy.app.handlers, lst)
        if fn not in hs:
            hs.append(fn)


def unregister():
    for lst, fn in _HANDLERS:
        hs = getattr(bpy.app.handlers, lst)
        if fn in hs:
            hs.remove(fn)
