"""Dynamic Section (after Alias Evaluate > Dynamic Section): a set of parallel
planes cut through the NURBS surfaces, the section curves drawn live with
their curvature combs.

The section is an Empty ("NURBS Section", one per scene): its Z axis is the
planes' normal and the planes are centred on it. Move and turn it like any
object -- G / R, the toolbar's Move / Rotate tools, or the Translate / Rotate
fields in the Analyze panel -- and the sections follow while you drag. Its settings
(number of planes, step, comb scale, samples, Visual Clip...) live on the
Empty, so the section is saved with the file; hiding the Empty hides it.

The planes are cut exactly by the kernel (intersect_brep_with_plane, ~1 ms a
plane). The combs do NOT come from those fitted section curves: at every sample
the curvature of the section is computed from the surface itself --

    t   = n_plane x n_surface          (the section's tangent)
    k_n = II(t, t)                     (normal curvature of the surface along t)
    K   = k_n / (b . n_surface) * b    (b = n_plane x t: the curve's in-plane normal)

(Meusnier), so the comb shows the surface, not a fit's wiggles. All planes of a
section share one comb scale, so their combs compare.
"""

import math

import numpy as np

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from mathutils import Vector

from . import geom, store

from .diagnostics import COMB_ENV_W, COMB_ENVELOPE, COMB_HAIR, COMB_HAIR_W, COMB_SPAN  # noqa: E402

COL_SECTION = (0.95, 0.95, 0.95, 1.0)
COL_PLANE = (0.35, 0.65, 1.0, 0.45)

_cache = {}          # section object name -> (key, sections)
_breps = {}          # target name -> (key, world brep)


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

def _redraw(self=None, context=None):
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class NprotoSection(bpy.types.PropertyGroup):
    is_section: BoolProperty(default=False)
    count: IntProperty(name="Planes", default=5, min=1, max=200, update=_redraw,
                       description="Number of parallel section planes")
    step: FloatProperty(name="Step", default=0.1, min=1e-6, unit='LENGTH', update=_redraw,
                        description="Distance between the planes")
    offset: FloatProperty(name="Position", default=0.0, unit='LENGTH', options={'HIDDEN'},   # unused (older files)
                          description="Slide the planes along their normal (drag to sweep them through the model)")
    curvature: BoolProperty(name="Curvature", default=True, update=_redraw,
                            description="Curvature combs on the section curves")
    scale: FloatProperty(name="Scale", default=1.0, min=0.0, soft_max=10.0, update=_redraw,
                         description="Length of the combs (the same for every plane, so they compare)")
    samples: IntProperty(name="Samples", default=80, min=8, max=2000, update=_redraw,
                         description="Points per section curve")
    show_plane: BoolProperty(name="Plane", default=True, update=_redraw,
                             description="Outline of the middle plane")
    targets: StringProperty(name="Surfaces", default="", update=lambda s, c: _clip_changed(s),
                            description="The objects it cuts (one name per line); empty: all visible NURBS objects")
    clip: BoolProperty(name="Visual Clip", default=False, update=lambda s, c: _clip_changed(s),
                       description="Show only the part of the model behind the middle plane (the sections stay visible)")
    flip: BoolProperty(name="Flip", default=False, update=lambda s, c: _clip_changed(s),
                       description="Show the other side")
    slab: BoolProperty(name="Clip Offset", default=False, update=lambda s, c: _clip_changed(s),
                       description="A second plane: only the slab between the two stays visible")
    slab_width: FloatProperty(name="Width", default=0.1, min=0.0, unit='LENGTH', update=lambda s, c: _clip_changed(s),
                              description="Distance of the second clipping plane")


def is_section(obj):
    return obj is not None and obj.type == 'EMPTY' and getattr(obj, "nproto_section", None) is not None \
        and obj.nproto_section.is_section


def targets(sec):
    s = sec.nproto_section
    names = [n for n in s.targets.split("\n") if n]
    if names:
        objs = [bpy.data.objects.get(n) for n in names]
        return [o for o in objs if o is not None and store.is_nurbs(o) and o.visible_get()]
    return [o for o in bpy.context.visible_objects if store.is_nurbs(o)]


def _size(objs):
    """Centre and diagonal of the objects' exact B-reps (world): the shaded
    mesh may be clipped (Visual Clip), the B-rep is not."""
    lo, hi = None, None
    for o in objs:
        try:
            a, b = (np.array(v, float) for v in store.get_brep(o).bounding_box())
            corners = [(x, y, z) for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])]
        except Exception:
            corners = [tuple(c) for c in o.bound_box]
        for c in corners:
            p = np.array(o.matrix_world @ Vector(c), float)
            lo = p if lo is None else np.minimum(lo, p)
            hi = p if hi is None else np.maximum(hi, p)
    if lo is None:
        return np.zeros(3), 1.0
    return 0.5 * (lo + hi), float(np.linalg.norm(hi - lo)) or 1.0


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def _world_brep(obj):
    key = (obj.get(store.PROP_REV), obj.get("nproto_build_key"), tuple(round(x, 9) for r in obj.matrix_world for x in r))
    hit = _breps.get(obj.name)
    if hit and hit[0] == key:
        return hit[1]
    from .sweeps import _world_brep as wb
    b = wb(obj)
    _breps[obj.name] = (key, b)
    return b


def planes(sec):
    """[(point, unit normal)] in world space, centred on the Empty."""
    s = sec.nproto_section
    M = sec.matrix_world
    n = (M.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    o = M.translation
    k = s.count
    return [(o + n * ((i - 0.5 * (k - 1)) * s.step), n) for i in range(k)]


def _resample(kc, m):
    """m points evenly by arc length along a kernel curve: (points, params)."""
    t0, t1 = kc.parameter_range()
    ts = np.linspace(t0, t1, 4 * m + 1)
    P = np.array([kc.evaluate(t) for t in ts], float)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if d[-1] <= 0:
        return P[:1], ts[:1]
    tt = np.interp(np.linspace(0.0, d[-1], m), d, ts)
    return np.array([kc.evaluate(t) for t in tt], float), tt


def _curvature(faces, P, npl, size, state):
    """Exact curvature vector of the plane section at each point of P (world),
    from the surface it lies on (see the module note). None where the plane
    is (nearly) tangent to the surface -- filled from the neighbours."""
    from .kernel import load
    from .square import _second
    sm = load()
    out = []
    tol = 1e-5 * size
    for i, p in enumerate(P):
        T = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        T /= np.linalg.norm(T) or 1.0
        pt = tuple(float(x) for x in p)
        order = ([state["fi"]] if state.get("fi") is not None else []) + \
            [k for k in range(len(faces)) if k != state.get("fi")]
        face, best = None, None
        for k in order:
            try:
                d = faces[k].closest_point(pt)[1]
            except Exception:
                continue
            if best is None or d < best[0]:
                best = (d, k)
            if d <= tol:
                break
        if best is None:
            out.append(None)
            continue
        state["fi"] = best[1]
        face = faces[best[1]]
        surf = face.surface()
        try:
            _, uvs, _ = sm.surface_closest_point(surf, pt)
            uv = (uvs[0].x, uvs[0].y) if hasattr(uvs[0], "x") else tuple(uvs[0])
            su, sv, suu, suv, svv = (np.asarray(x, float) for x in _second(surf, *uv))
        except Exception:
            out.append(None)
            continue
        ns = np.cross(su, sv)
        ln = np.linalg.norm(ns)
        if ln < 1e-14:
            out.append(None)
            continue
        ns /= ln
        t = np.cross(npl, ns)
        lt = np.linalg.norm(t)
        if lt < 1e-9:                       # the plane touches the surface here
            out.append(None)
            continue
        t /= lt
        if t @ T < 0:
            t = -t
        b = np.cross(npl, t)
        den = float(b @ ns)
        if abs(den) < 0.05:                 # grazing cut: Meusnier blows up
            out.append(None)
            continue
        a, c = np.linalg.lstsq(np.array([su, sv]).T, t, rcond=None)[0]
        du = a * su + c * sv
        kn = (a * a * (suu @ ns) + 2 * a * c * (suv @ ns) + c * c * (svv @ ns)) / max(du @ du, 1e-300)
        out.append((kn / den) * b)
    # fill the gaps from the neighbours (a short run where the plane grazes the surface)
    K = np.zeros((len(P), 3))
    have = [i for i, k in enumerate(out) if k is not None]
    if not have:
        return K, False
    for i in range(len(P)):
        if out[i] is not None:
            K[i] = out[i]
        else:
            j = min(have, key=lambda h: abs(h - i))
            K[i] = out[j]
    return K, True


def compute(sec):
    """[{"plane": (point, normal), "curves": [{"P", "K"}]}] for a section Empty
    (cached: recomputed when it moves, a setting changes or a target rebuilds)."""
    from .kernel import load
    sm = load()
    s = sec.nproto_section
    objs = targets(sec)
    key = (tuple(round(x, 9) for r in sec.matrix_world for x in r), s.count, round(s.step, 12), round(s.offset, 12),
           s.samples, s.curvature, tuple((o.name, o.get(store.PROP_REV), tuple(round(x, 9) for r in o.matrix_world
                                                                                for x in r)) for o in objs))
    hit = _cache.get(sec.name)
    if hit and hit[0] == key:
        return hit[1]
    _, size = _size(objs)
    out = []
    breps = []
    for o in objs:
        try:
            breps.append(_world_brep(o))
        except Exception as exc:
            print("NURBS section:", o.name, exc)
    for p, n in planes(sec):
        npl = np.array(n, float)
        curves = []
        for b in breps:
            try:
                cs = sm.intersect_brep_with_plane(b, tuple(p), tuple(n))
            except Exception:
                continue
            faces = list(b.faces())
            for kc in cs or []:
                try:
                    L = kc.length()
                except Exception:
                    L = 0.0
                m = max(8, int(round(s.samples * min(1.0, max(L / size, 0.15)) * 2.0)))
                P, _ = _resample(kc, m)
                if len(P) < 2:
                    continue
                K, ok = (_curvature(faces, P, npl, size, {}) if s.curvature else (None, False))
                curves.append({"P": P, "K": K if ok else None})
        out.append({"plane": (np.array(p, float), npl), "curves": curves})
    _cache[sec.name] = (key, out)
    return out


# --------------------------------------------------------------------------
# Visual Clip: a Geometry Nodes modifier on the cut objects drops the faces
# past the middle plane (and, with Clip Offset, before a second one), driven
# live by the section Empty; the B-rep edge overlay is clipped the same way
# --------------------------------------------------------------------------

CLIP = "NURBS Section Clip"
CLIP_VERSION = 3


def _clip_group():
    ng = bpy.data.node_groups.get(CLIP)
    if ng is not None and ng.get("nproto_clip") == CLIP_VERSION:
        return ng
    if ng is not None:
        ng.name = CLIP + " (old)"           # modifiers still using it are pointed at the new one by apply_clip
    ng = bpy.data.node_groups.new(CLIP, 'GeometryNodeTree')
    ng["nproto_clip"] = CLIP_VERSION
    it = ng.interface
    it.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    it.new_socket("Section", in_out='INPUT', socket_type='NodeSocketObject')
    it.new_socket("Offset", in_out='INPUT', socket_type='NodeSocketFloat')
    it.new_socket("Flip", in_out='INPUT', socket_type='NodeSocketBool')
    it.new_socket("Slab", in_out='INPUT', socket_type='NodeSocketBool')
    it.new_socket("Width", in_out='INPUT', socket_type='NodeSocketFloat')
    it.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, L = ng.nodes, ng.links
    gi, go = N.new('NodeGroupInput'), N.new('NodeGroupOutput')
    oi = N.new('GeometryNodeObjectInfo')
    oi.transform_space = 'RELATIVE'                 # the section in this object's own space
    L.new(gi.outputs["Section"], oi.inputs["Object"])
    rv = N.new('FunctionNodeRotateVector')
    rv.inputs["Vector"].default_value = (0.0, 0.0, 1.0)
    L.new(oi.outputs["Rotation"], rv.inputs["Rotation"])
    nrm = N.new('ShaderNodeVectorMath')
    nrm.operation = 'NORMALIZE'
    L.new(rv.outputs[0], nrm.inputs[0])
    pos = N.new('GeometryNodeInputPosition')
    sub = N.new('ShaderNodeVectorMath')
    sub.operation = 'SUBTRACT'
    L.new(pos.outputs[0], sub.inputs[0])
    L.new(oi.outputs["Location"], sub.inputs[1])
    dot = N.new('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    L.new(sub.outputs[0], dot.inputs[0])
    L.new(nrm.outputs[0], dot.inputs[1])
    d = N.new('ShaderNodeMath')
    d.operation = 'SUBTRACT'
    L.new(dot.outputs["Value"], d.inputs[0])
    L.new(gi.outputs["Offset"], d.inputs[1])
    sg = N.new('ShaderNodeMath')
    sg.operation = 'MULTIPLY_ADD'                   # 1 - 2 flip
    L.new(gi.outputs["Flip"], sg.inputs[0])
    sg.inputs[1].default_value = -2.0
    sg.inputs[2].default_value = 1.0
    side = N.new('ShaderNodeMath')
    side.operation = 'MULTIPLY'
    L.new(d.outputs[0], side.inputs[0])
    L.new(sg.outputs[0], side.inputs[1])
    past = N.new('ShaderNodeMath')
    past.operation = 'GREATER_THAN'
    L.new(side.outputs[0], past.inputs[0])
    past.inputs[1].default_value = 0.0
    negw = N.new('ShaderNodeMath')
    negw.operation = 'MULTIPLY'
    L.new(gi.outputs["Width"], negw.inputs[0])
    negw.inputs[1].default_value = -1.0
    before = N.new('ShaderNodeMath')
    before.operation = 'LESS_THAN'
    L.new(side.outputs[0], before.inputs[0])
    L.new(negw.outputs[0], before.inputs[1])
    band = N.new('FunctionNodeBooleanMath')
    band.operation = 'AND'
    L.new(gi.outputs["Slab"], band.inputs[0])
    L.new(before.outputs[0], band.inputs[1])
    cut = N.new('FunctionNodeBooleanMath')
    cut.operation = 'OR'
    L.new(past.outputs[0], cut.inputs[0])
    L.new(band.outputs[0], cut.inputs[1])
    # judged per CORNER, then per face: a face goes only when all its corners
    # are past the plane, so the faces crossing it stay ...
    cap = N.new('GeometryNodeCaptureAttribute')
    cap.domain = 'POINT'
    cap.capture_items.new('BOOLEAN', "Cut")
    L.new(gi.outputs["Geometry"], cap.inputs["Geometry"])
    L.new(cut.outputs[0], cap.inputs["Cut"])
    dg = N.new('GeometryNodeDeleteGeometry')
    dg.domain = 'FACE'
    L.new(cap.outputs["Geometry"], dg.inputs["Geometry"])
    L.new(cap.outputs["Cut"], dg.inputs["Selection"])
    # ... and their corners past it move onto it: the cut is a clean line ON the
    # plane, not the mesh's sawtooth (Clip Offset: the same at the second plane).
    # They move ALONG THE SURFACE (in their tangent plane), not along the
    # plane's normal: a long triangle of a straight wall cut by a tilted plane
    # would otherwise be pulled far off the surface
    o1 = N.new('ShaderNodeMath')
    o1.operation = 'MAXIMUM'
    L.new(side.outputs[0], o1.inputs[0])
    o1.inputs[1].default_value = 0.0
    mw = N.new('ShaderNodeMath')
    mw.operation = 'SUBTRACT'                       # -width - side: > 0 before the second plane
    L.new(negw.outputs[0], mw.inputs[0])
    L.new(side.outputs[0], mw.inputs[1])
    o2r = N.new('ShaderNodeMath')
    o2r.operation = 'MAXIMUM'
    L.new(mw.outputs[0], o2r.inputs[0])
    o2r.inputs[1].default_value = 0.0
    o2 = N.new('ShaderNodeMath')
    o2.operation = 'MULTIPLY'
    L.new(o2r.outputs[0], o2.inputs[0])
    L.new(gi.outputs["Slab"], o2.inputs[1])
    dl = N.new('ShaderNodeMath')
    dl.operation = 'SUBTRACT'
    L.new(o2.outputs[0], dl.inputs[0])
    L.new(o1.outputs[0], dl.inputs[1])
    dls = N.new('ShaderNodeMath')
    dls.operation = 'MULTIPLY'                      # along the kept side's normal (Flip)
    L.new(dl.outputs[0], dls.inputs[0])
    L.new(sg.outputs[0], dls.inputs[1])
    off = N.new('ShaderNodeVectorMath')            # fallback: along the plane's normal
    off.operation = 'SCALE'
    L.new(nrm.outputs[0], off.inputs[0])
    L.new(dls.outputs[0], off.inputs["Scale"])
    vn = N.new('GeometryNodeInputNormal')          # the vertex normal: its tangent plane
    ndn = N.new('ShaderNodeVectorMath')
    ndn.operation = 'DOT_PRODUCT'
    L.new(nrm.outputs[0], ndn.inputs[0])
    L.new(vn.outputs[0], ndn.inputs[1])
    nn = N.new('ShaderNodeVectorMath')
    nn.operation = 'SCALE'
    L.new(vn.outputs[0], nn.inputs[0])
    L.new(ndn.outputs["Value"], nn.inputs["Scale"])
    tn = N.new('ShaderNodeVectorMath')              # n - (n.N) N: the plane normal in the tangent plane
    tn.operation = 'SUBTRACT'
    L.new(nrm.outputs[0], tn.inputs[0])
    L.new(nn.outputs[0], tn.inputs[1])
    t2 = N.new('ShaderNodeVectorMath')
    t2.operation = 'DOT_PRODUCT'
    L.new(tn.outputs[0], t2.inputs[0])
    L.new(tn.outputs[0], t2.inputs[1])
    inv = N.new('ShaderNodeMath')                   # moving along tn changes the plane distance by |tn|^2
    inv.operation = 'DIVIDE'
    L.new(dls.outputs[0], inv.inputs[0])
    L.new(t2.outputs["Value"], inv.inputs[1])
    offs = N.new('ShaderNodeVectorMath')
    offs.operation = 'SCALE'
    L.new(tn.outputs[0], offs.inputs[0])
    L.new(inv.outputs[0], offs.inputs["Scale"])
    graze = N.new('ShaderNodeMath')                 # surface nearly parallel to the plane: along the normal
    graze.operation = 'LESS_THAN'
    L.new(t2.outputs["Value"], graze.inputs[0])
    graze.inputs[1].default_value = 0.04
    sw = N.new('GeometryNodeSwitch')
    sw.input_type = 'VECTOR'
    L.new(graze.outputs[0], sw.inputs["Switch"])
    L.new(offs.outputs[0], sw.inputs["False"])
    L.new(off.outputs[0], sw.inputs["True"])
    sp = N.new('GeometryNodeSetPosition')
    L.new(dg.outputs[0], sp.inputs["Geometry"])
    L.new(sw.outputs[0], sp.inputs["Offset"])
    L.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def _ident(ng, name):
    return next(i.identifier for i in ng.interface.items_tree
                if getattr(i, "in_out", "") == 'INPUT' and i.name == name)


def _mset(m, name, value):
    """Set a Geometry Nodes modifier input (Blender 5: modifier.properties.inputs;
    before: modifier[identifier])."""
    ident = _ident(m.node_group, name)
    props = getattr(m, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        getattr(props.inputs, ident).value = value
    else:
        m[ident] = value


def _mget(m, name):
    ident = _ident(m.node_group, name)
    props = getattr(m, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        return getattr(props.inputs, ident).value
    return m[ident]


def clip_section(m):
    """The section Empty a clip modifier follows (its Section input)."""
    try:
        return _mget(m, "Section") if m.node_group is not None else None
    except Exception:
        return None


def _clip_changed(settings):
    sec = settings.id_data
    try:
        apply_clip(sec)
    except Exception as exc:
        print("NURBS section clip:", exc)
    _redraw()


def apply_clip(sec):
    """Add / update / remove the clip modifiers this section drives."""
    s = sec.nproto_section
    want = {o.name: o for o in targets(sec)} if (s.is_section and s.clip) else {}
    for o in bpy.data.objects:
        m = o.modifiers.get(CLIP) if o.type == 'MESH' else None
        if m is not None and clip_section(m) == sec and o.name not in want:
            o.modifiers.remove(m)
    if not want:
        return
    ng = _clip_group()
    for o in want.values():
        if o.type != 'MESH':
            continue
        m = o.modifiers.get(CLIP)
        if m is None:
            m = o.modifiers.new(CLIP, 'NODES')
            m.show_in_editmode = False
        m.node_group = ng
        _mset(m, "Section", sec)
        _mset(m, "Offset", 0.0)
        _mset(m, "Flip", bool(s.flip))
        _mset(m, "Slab", bool(s.slab))
        _mset(m, "Width", float(s.slab_width))
        o.update_tag()


def clip_planes(obj):
    """[(point, normal, slab width | None)] world planes clipping obj (from its
    clip modifier), for the edge overlay; [] when not clipped. A point is
    kept when (x - point) . normal <= 0 (and >= -width with Clip Offset)."""
    m = obj.modifiers.get(CLIP) if obj.type == 'MESH' else None
    if m is None or not m.show_viewport:
        return []
    sec = clip_section(m)
    if not is_section(sec) or not sec.nproto_section.clip:
        return []
    s = sec.nproto_section
    n = (sec.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    p = sec.matrix_world.translation.copy()
    return [(p, -n if s.flip else n, s.slab_width if s.slab else None)]


def keep_segments(coords, planes_):
    """Line pairs clipped to the kept side of the clip planes: a segment
    crossing a plane is cut exactly where it crosses (like the shaded mesh)."""
    if not planes_:
        return coords
    out = []
    for i in range(0, len(coords) - 1, 2):
        a, b = coords[i], coords[i + 1]
        for p, n, w in planes_:
            for lim in ([0.0] if w is None else [0.0, -w]):
                sgn = 1.0 if lim == 0.0 else -1.0       # keep d <= 0, and d >= -w for the second plane
                da = sgn * ((a - p).dot(n) - lim)
                db = sgn * ((b - p).dot(n) - lim)
                if da > 0 and db > 0:
                    a = None
                    break
                if da > 0 or db > 0:
                    x = a + (b - a) * (da / (da - db))
                    a, b = (x, b) if da > 0 else (a, x)
            if a is None:
                break
        if a is not None:
            out.extend([a, b])
    return out


def _sync_clip(scene):
    """Clip modifiers follow their section: removed when it is gone or no
    longer clips (an empty Section input would clip everything past z = 0),
    paused while it is hidden (Dynamic Section switched off, or H)."""
    names = None
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        m = o.modifiers.get(CLIP)
        if m is None:
            continue
        if names is None:
            names = {x.name for x in scene.objects}
        sec = clip_section(m)
        if not is_section(sec) or not sec.nproto_section.clip or sec.name not in names:
            o.modifiers.remove(m)
            continue
        if m.node_group.get("nproto_clip") != CLIP_VERSION:   # a file saved with an older clip: update it
            apply_clip(sec)
            m = o.modifiers.get(CLIP)
            if m is None:
                continue
        on = sec.visible_get()
        if m.show_viewport != on:
            m.show_viewport = on


@bpy.app.handlers.persistent
def _clip_cleanup(scene, depsgraph=None):
    _sync_clip(scene)


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _v(p):
    return Vector((float(p[0]), float(p[1]), float(p[2])))


_kref_cache = {}


def _kref(objs):
    """The targets' reference curvature: the 95th percentile of the largest
    principal curvature on a grid over every face (exact, world space). It is
    a property of the surfaces, not of where the planes cut, so the comb
    length stays put while you move or turn the section."""
    from .square import _second
    key = tuple((o.name, o.get(store.PROP_REV), tuple(round(x, 9) for r in o.matrix_world for x in r)) for o in objs)
    hit = _kref_cache.get(key)
    if hit is not None:
        return hit
    ks = []
    for o in objs:
        try:
            faces = list(_world_brep(o).faces())
        except Exception:
            continue
        for f in faces:
            surf = f.surface()
            (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
            for u in np.linspace(u0, u1, 9):
                for v in np.linspace(v0, v1, 9):
                    try:
                        su, sv, suu, suv, svv = (np.asarray(x, float) for x in _second(surf, u, v))
                    except Exception:
                        continue
                    n = np.cross(su, sv)
                    ln = np.linalg.norm(n)
                    if ln < 1e-12:
                        continue
                    n /= ln
                    E, F, G = su @ su, su @ sv, sv @ sv
                    e, f_, g = suu @ n, suv @ n, svv @ n
                    den = E * G - F * F
                    if den <= 1e-300:
                        continue
                    H = (e * G - 2 * f_ * F + g * E) / (2 * den)
                    K = (e * g - f_ * f_) / den
                    ks.append(abs(H) + math.sqrt(max(H * H - K, 0.0)))
    k = float(np.percentile(ks, 95)) if ks else 0.0
    if len(_kref_cache) > 16:
        _kref_cache.clear()
    _kref_cache[key] = k
    return k


def comb_factor(s, objs, size):
    """Hair length per unit curvature: at Scale 1 a hair at the surfaces'
    reference curvature is a fifth of their size (as on every other comb)."""
    k = _kref(objs)
    return s.scale * COMB_SPAN * size / k if k * size > 1e-9 else 0.0


def section_objects(context):
    return [o for o in context.visible_objects if is_section(o)]


def draw_view(ctx, draw_lines):
    import gpu
    secs = section_objects(ctx)
    if not secs:
        return
    gpu.state.depth_test_set('NONE')
    for sec in secs:
        s = sec.nproto_section
        try:
            data = compute(sec)
        except Exception as exc:
            print("NURBS section:", sec.name, exc)
            continue
        objs = targets(sec)
        centre, size = _size(objs)
        comb = comb_factor(s, objs, size)        # hair length = curvature x this (one scale for all planes)
        lines, hairs, env = [], [], []
        for sect in data:
            for c in sect["curves"]:
                P = c["P"]
                for i in range(len(P) - 1):
                    lines.extend([_v(P[i]), _v(P[i + 1])])
                if s.curvature and c["K"] is not None:
                    tips = P - c["K"] * comb
                    for i in range(len(P)):
                        hairs.extend([_v(P[i]), _v(tips[i])])
                    for i in range(len(P) - 1):
                        env.extend([_v(tips[i]), _v(tips[i + 1])])
        if s.show_plane and data:
            # the middle plane's outline, the size of the model
            M = sec.matrix_world.to_3x3().normalized()
            ax, ay = M @ Vector((1, 0, 0)), M @ Vector((0, 1, 0))
            o = _v(data[len(data) // 2]["plane"][0])
            h = 0.5 * size
            cs = [o + (ax * sx + ay * sy) * h for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            draw_lines([cs[0], cs[1], cs[1], cs[2], cs[2], cs[3], cs[3], cs[0]], COL_PLANE, 1.0)
        draw_lines(hairs, COMB_HAIR, COMB_HAIR_W)
        draw_lines(env, COMB_ENVELOPE, COMB_ENV_W)
        draw_lines(lines, COL_SECTION, 2.0)
    gpu.state.depth_test_set('LESS_EQUAL')


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

def _orient(normal):
    """Rotation (Euler) turning +Z onto normal."""
    return Vector((0, 0, 1)).rotation_difference(Vector(normal).normalized()).to_euler()


def _view_axis(context):
    """The world axis closest to the view's horizontal: planes then run across
    the screen like Alias's default."""
    rv3d = context.region_data
    if rv3d is None:
        return Vector((1, 0, 0))
    right = rv3d.view_rotation @ Vector((1, 0, 0))
    k = max(range(3), key=lambda i: abs(right[i]))
    v = Vector((0, 0, 0))
    v[k] = 1.0
    return v


def scene_section(context):
    """The scene's Dynamic Section (there is one at most), or None."""
    return next((o for o in context.scene.objects if is_section(o)), None)


def _select_only(context, obj):
    for o in context.selected_objects:
        o.select_set(False)
    if obj.hide_get():
        obj.hide_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


def section_active(context):
    """Is the Dynamic Section on (the scene's section shown)? Its button shows
    pressed then, and a click switches it off -- like Zebra."""
    sec = scene_section(context)
    return sec is not None and sec.visible_get()


class NPROTO_OT_section_add(bpy.types.Operator):
    """Dynamic Section: parallel planes through the surfaces with their section
    curves and curvature combs, live while you move or turn the section.
    Cuts the selected NURBS objects, or all visible ones when none is selected.
    Click again to switch it off (the section is hidden, Visual Clip paused);
    once more brings it back as it was. One section per scene"""
    bl_idname = "nproto.section_add"
    bl_label = "Dynamic Section"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        old = scene_section(context)
        if old is not None and old.visible_get():          # on: switch it off
            old.select_set(False)
            old.hide_set(True)
            _sync_clip(context.scene)
            _redraw()
            self.report({'INFO'}, "Dynamic Section off")
            return {'FINISHED'}
        if old is not None:                                 # off: bring it back as it was
            _select_only(context, old)
            _sync_clip(context.scene)
            _redraw()
            return {'FINISHED'}
        sel = [o for o in context.selected_objects if store.is_nurbs(o)]
        objs = sel or [o for o in context.visible_objects if store.is_nurbs(o)]
        if not objs:
            self.report({'WARNING'}, "No NURBS surfaces to section")
            return {'CANCELLED'}
        centre, size = _size(objs)
        e = bpy.data.objects.new("NURBS Section", None)
        e.empty_display_type = 'SINGLE_ARROW'
        e.empty_display_size = 0.25 * size
        e.show_in_front = True
        (context.collection or context.scene.collection).objects.link(e)
        e.location = Vector(centre)
        e.rotation_euler = _orient(_view_axis(context))
        s = e.nproto_section
        s.is_section = True
        s.count = 5
        s.step = size / 8.0
        s.targets = "\n".join(o.name for o in sel)
        _select_only(context, e)
        self.report({'INFO'}, f"Dynamic Section through {len(objs)} object(s): move / turn it with G / R (or the "
                              "Move / Rotate tools); settings in the Analyze panel")
        return {'FINISHED'}


class NPROTO_OT_section_remove(bpy.types.Operator):
    """Delete the Dynamic Section (its sections, combs and Visual Clip go with it)"""
    bl_idname = "nproto.section_remove"
    bl_label = "Remove Section"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(options={'HIDDEN'})

    def execute(self, context):
        e = bpy.data.objects.get(self.name) if self.name else scene_section(context)
        if not is_section(e):
            return {'CANCELLED'}
        e.nproto_section.clip = False          # drops its clip modifiers
        _cache.pop(e.name, None)
        bpy.data.objects.remove(e)
        _redraw()
        return {'FINISHED'}


class NPROTO_OT_section_orient(bpy.types.Operator):
    """Turn the active section so its planes face an axis (Rotation Reset: +Z)"""
    bl_idname = "nproto.section_orient"
    bl_label = "Orient Section"
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(items=[("X", "X", "Planes across X"), ("Y", "Y", "Planes across Y"), ("Z", "Z", "Planes across Z"),
                              ("VIEW", "View", "Planes facing the view"), ("RESET", "Reset", "Rotation 0, 0, 0")])

    @classmethod
    def poll(cls, context):
        return is_section(context.active_object)

    def execute(self, context):
        e = context.active_object
        if self.axis == "RESET":
            e.rotation_euler = (0.0, 0.0, 0.0)
        elif self.axis == "VIEW":
            rv3d = context.region_data
            if rv3d is None:
                return {'CANCELLED'}
            e.rotation_euler = _orient(rv3d.view_rotation @ Vector((0, 0, 1)))
        else:
            v = Vector((0, 0, 0))
            v["XYZ".index(self.axis)] = 1.0
            e.rotation_euler = _orient(v)
        _redraw()
        return {'FINISHED'}


class NPROTO_OT_section_targets(bpy.types.Operator):
    """Cut the selected NURBS objects (or, with none selected, all visible ones)"""
    bl_idname = "nproto.section_targets"
    bl_label = "Use Selection"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(options={'HIDDEN'})

    def execute(self, context):
        e = bpy.data.objects.get(self.name)
        if not is_section(e):
            return {'CANCELLED'}
        e.nproto_section.targets = "\n".join(o.name for o in context.selected_objects if store.is_nurbs(o))
        _redraw()
        return {'FINISHED'}


class NPROTO_OT_section_curves(bpy.types.Operator):
    """Make the current sections real curves (one curve object, world space)"""
    bl_idname = "nproto.section_curves"
    bl_label = "Create Curves"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_section(context.active_object)

    def execute(self, context):
        from .kernel import load
        from .primitives import write_spline_object
        from .sweeps import _kcurve_poly, _poly_to_spec
        sm = load()
        sec = context.active_object
        specs = []
        for p, n in planes(sec):
            for o in targets(sec):
                try:
                    cs = sm.intersect_brep_with_plane(_world_brep(o), tuple(p), tuple(n))
                except Exception:
                    continue
                for kc in cs or []:
                    try:
                        specs.append(_poly_to_spec(_kcurve_poly(kc)))
                    except Exception as exc:
                        print("NURBS section curve:", exc)
        if not specs:
            self.report({'WARNING'}, "The planes cut nothing")
            return {'CANCELLED'}
        cu = bpy.data.curves.new("Section Curves", 'CURVE')
        obj = bpy.data.objects.new("Section Curves", cu)
        (context.collection or context.scene.collection).objects.link(obj)
        write_spline_object(obj, specs)
        self.report({'INFO'}, f"{len(specs)} section curve(s)")
        return {'FINISHED'}


def draw_panel(col, context):
    """Analyze panel: the button (adds the scene's one section, or selects it),
    and its settings while it is the active object."""
    e = context.active_object
    col.operator("nproto.section_add", icon='MOD_EDGESPLIT', text="Dynamic Section", depress=section_active(context))
    if not is_section(e) or not e.visible_get():
        return
    s = e.nproto_section
    box = col.box()
    row = box.row(align=True)
    row.label(text=e.name, icon='MOD_EDGESPLIT')
    row.operator("nproto.section_remove", text="", icon='X').name = e.name
    row = box.row(align=True)
    row.prop(s, "count")
    row.prop(s, "step")
    row = box.row(align=True)
    row.prop(s, "curvature", toggle=True, icon='IPO_ELASTIC')
    row.prop(s, "show_plane", toggle=True)
    row = box.row(align=True)
    sub = row.row(align=True)
    sub.enabled = s.curvature
    sub.prop(s, "scale")
    row.prop(s, "samples")
    sub = box.column(align=True)
    sub.prop(e, "location", text="Translate")
    sub.prop(e, "rotation_euler", text="Rotate")
    row = box.row(align=True)
    for ax in ("X", "Y", "Z", "VIEW"):
        row.operator("nproto.section_orient", text=ax.title() if ax == "VIEW" else ax).axis = ax
    row.operator("nproto.section_orient", text="", icon='LOOP_BACK').axis = "RESET"
    row = box.row(align=True)
    row.prop(s, "clip", toggle=True, icon='MOD_BOOLEAN')
    sub = row.row(align=True)
    sub.enabled = s.clip
    sub.prop(s, "flip", toggle=True)
    sub.prop(s, "slab", toggle=True)
    if s.clip and s.slab:
        box.prop(s, "slab_width")
    names = [n for n in s.targets.split("\n") if n]
    row = box.row(align=True)
    row.label(text=(f"Cuts: {', '.join(names)}" if names else "Cuts: all visible NURBS objects"))
    row.operator("nproto.section_targets", text="", icon='RESTRICT_SELECT_OFF').name = e.name
    box.operator("nproto.section_curves", icon='CURVE_DATA')


classes = (NprotoSection, NPROTO_OT_section_add, NPROTO_OT_section_remove, NPROTO_OT_section_orient,
           NPROTO_OT_section_targets,
           NPROTO_OT_section_curves)


def register():
    bpy.types.Object.nproto_section = PointerProperty(type=NprotoSection)
    if _clip_cleanup not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_clip_cleanup)


def unregister():
    if _clip_cleanup in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_clip_cleanup)
    if hasattr(bpy.types.Object, "nproto_section"):
        del bpy.types.Object.nproto_section
