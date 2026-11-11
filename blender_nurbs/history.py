"""Construction history: rebuild results when their inputs change.

Watches depsgraph updates; when an object that some NURBS result was built
from changes (transform or geometry, including CV edits in edit mode), the
result is rebuilt on a short timer so dragging stays responsive. Results that
are themselves inputs propagate downstream in dependency order.
"""

import bpy
from bpy.app.handlers import persistent

from . import features, store

_dirty = set()       # names of result objects to rebuild
_just_built = set()  # rebuilt by us: their own mesh update must not re-trigger
_held = set()       # rebuilds Pause Rebuild holds back (inputs that changed meanwhile)
_busy = False
_DELAY = 0.08


def _dependents_map(scene):
    deps = {}
    for obj in scene.objects:
        if not features.has_history(obj):
            continue
        for src in features.history_inputs(obj):
            deps.setdefault(src.name, set()).add(obj.name)
    return deps


@persistent
def _on_depsgraph(scene, depsgraph):
    if _busy:
        return
    s = getattr(scene, "nproto_settings", None)
    if s is None or not s.construction_history or not s.live_update:
        return
    changed = set()
    for u in depsgraph.updates:
        idb = u.id
        if isinstance(idb, bpy.types.Object) and (u.is_updated_geometry or u.is_updated_transform):
            changed.add(idb.original.name)
        elif isinstance(idb, bpy.types.Curve):
            for o in scene.objects:
                if o.data == idb.original:
                    changed.add(o.name)
    changed -= _just_built
    _just_built.clear()
    if not changed:
        return
    deps = _dependents_map(scene)
    hit = set()
    for name in changed:
        hit |= deps.get(name, set())
    if hit and getattr(s, "pause_rebuild", False):
        _held.update(hit)                # built by Rebuild / when the pause ends
        return
    if hit:
        _dirty.update(hit)
        if not bpy.app.timers.is_registered(_process):
            bpy.app.timers.register(_process, first_interval=_DELAY)


def _order(names, scene):
    """Topological order so upstream results rebuild first."""
    objs = {n: scene.objects.get(n) for n in names}
    done, out = set(), []

    def visit(n):
        if n in done or objs.get(n) is None:
            return
        done.add(n)
        for src in features.history_inputs(objs[n]):
            if src.name in objs:
                visit(src.name)
        out.append(objs[n])

    for n in names:
        visit(n)
    return out


def _process():
    global _busy
    scene = bpy.context.scene
    if not _dirty:
        return None
    # pull in downstream dependents of everything we rebuild
    deps = _dependents_map(scene)
    todo = set(_dirty)
    stack = list(todo)
    while stack:
        n = stack.pop()
        for d in deps.get(n, ()):
            if d not in todo:
                todo.add(d)
                stack.append(d)
    _dirty.clear()
    _busy = True
    try:
        for obj in _order(todo, scene):
            try:
                features.rebuild(obj)
                _just_built.add(obj.name)
            except Exception as exc:
                print(f"NURBS history: rebuild of {obj.name} failed: {exc}")
    finally:
        _busy = False
    return None


def waiting(scene):
    """Names of the objects Pause Rebuild holds back: edited settings, or
    inputs that changed meanwhile."""
    return {o.name for o in scene.objects if o.get(features.PROP_PENDING)} | \
        {n for n in _held if scene.objects.get(n) is not None}


def rebuild_waiting(context):
    """Build everything Pause Rebuild held back, upstream first, and what is
    built from it. -> (number of objects that waited, [error texts])."""
    global _busy
    scene = context.scene
    flagged = {o.name for o in scene.objects if o.get(features.PROP_PENDING)}
    names = waiting(scene)
    _held.clear()
    if not names:
        return 0, []
    deps = _dependents_map(scene)
    todo, stack = set(names), list(names)
    while stack:
        for d in deps.get(stack.pop(), ()):
            if d not in todo:
                todo.add(d)
                stack.append(d)
    errors = []
    _busy = True
    try:
        for obj in _order(todo, scene):
            try:
                features.rebuild(obj, context, force=obj.name in flagged)
                _just_built.add(obj.name)
            except Exception as exc:
                obj[features.PROP_PENDING] = 1           # still not built from its settings
                errors.append(f"{obj.name}: {exc}")
    finally:
        _busy = False
    return len(names), errors


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    if bpy.app.timers.is_registered(_process):
        bpy.app.timers.unregister(_process)
