"""Twist, Bend and Taper: deform NURBS objects (trimmed faces and solids too)
and curves along an axis, in place.

A point is known by where it sits along the axis (s, from the axis start, 0
to L at its end) and across it (a along the bend direction x, b along
y = z x x). Each deformation is a smooth map of those:

* Twist   turns the point about the axis by angle * g(s / L)
* Bend    bends the axis into an arc (angle * g: the turn of its tangent);
          the point keeps its distance a across the bent axis, so the axis
          itself is the neutral layer (lengths along it are kept)
* Taper   scales a and b (Flat: only a) by 1 + (factor - 1) * g(s / L)

g runs from 0 to 1 over the axis: straight (constant rate) or Smooth (a
quintic ease, no rate jump at the ends: a twisted or bent stretch blends
into the rest without a crease). Outside the axis the part carries on
rigidly (Bend: straight along the end tangent), unless Infinite: the rate
goes on past the ends.

The map goes through Flow's machinery (flow.map_surface, surfmap.trimmed_face):
every face's surface is refitted over its own (u, v) domain to the mapped
grid, and trimmed by its mapped edges, so a solid stays one closed body. A
face's surface is first cut down to its trim's (u, v) box (exactly, by knot
insertion): the parts of an untrimmed surface far outside its trim would
otherwise be refitted too -- past the axis end, where a straight twist turns
a corner, that cost thousands of CVs for nothing.

The axis: Auto (the part's longest extent), the object's X / Y / Z, or a
picked straight line (select the line last, active: its start to its end;
with history the deformation follows the line when it moves). For the bend
direction the part's thinnest extent across the axis is used, turned by
Direction. History: an in-place step ("deform"), editable in the History
panel.
"""

import json
import math
import time

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, StringProperty

from . import curvejoin, curvetrim, features, flow, geom, hints, store, surfeval, surfmap
from .valuedrag import ValueDrag, angle_pick, rotate_points

OP = "deform"
SURF_TOL = 1e-6          # refit tolerance, a share of the size (as Flow)
CURVE_TOL = 1e-5
FOLD_MIN = 0.02          # the map may squeeze a part down to this share of its size, not further
LAST_INFO = {}

KINDS = [("TWIST", "Twist", "Turn the part about the axis, more and more along it"),
         ("BEND", "Bend", "Bend the axis into an arc; the part bends with it"),
         ("TAPER", "Taper", "Scale the part across the axis, more and more along it")]
AXES = [("AUTO", "Auto", "Along the part's longest extent"),
        ("X", "Local X", "The object's X axis, over the part's extent"),
        ("Y", "Local Y", "The object's Y axis, over the part's extent"),
        ("Z", "Local Z", "The object's Z axis, over the part's extent"),
        ("LINE", "Line", "A picked straight line, from its start to its end")]
LABEL = {k: n for k, n, _ in KINDS}


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def ease(t, smooth):
    """(g, dg/dt) of the ramp at t (already clipped where it should be)."""
    if not smooth:
        return t, np.ones_like(t)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0), 30.0 * t * t * (t - 1.0) ** 2


# --------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------

def points_of_brep(b, n=17):
    """Points that show a brep's extent: samples along every edge, and the
    corners of its box (a face bulging past its edges)."""
    from .refit import _edge_polyline
    out = []
    for e in b.edges():
        try:
            out.append(_edge_polyline(e, n))
        except Exception:
            pass
    try:
        lo, hi = (np.array(v, float) for v in b.bounding_box())
        if not out:
            out.append(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                 for z in (lo[2], hi[2])]))
    except Exception:
        pass
    return np.vstack(out) if out else np.zeros((0, 3))


def points_of_curve(obj, per=16):
    """Local sample points of a curve object's splines."""
    out = []
    for sp in obj.data.splines:
        try:
            P, _ = curvetrim.sample_spans(curvejoin.spline_spans(sp), per)
            out.append(np.asarray(P, float))
        except Exception:
            continue
    return np.vstack(out) if out else np.zeros((0, 3))


class Frame:
    """The axis (start O, unit z, length L) and the bend direction x (y = z x x)."""

    def __init__(self, O, z, x, L):
        self.O, self.z, self.x, self.L = (np.asarray(O, float), _unit(z), _unit(x), float(L))
        self.y = np.cross(self.z, self.x)

    def coords(self, Q):
        R = np.asarray(Q, float).reshape(-1, 3) - self.O
        return R @ self.z, R @ self.x, R @ self.y

    def point(self, s, a, b):
        return self.O + np.outer(s, self.z) + np.outer(a, self.x) + np.outer(b, self.y)


def _thinnest(P, z, prefer=()):
    """The unit direction across z along which the points P are thinnest
    (one of `prefer` when it is as thin, to within a hair)."""
    t = _unit(np.cross(z, [1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.cross(z, [0.0, 1.0, 0.0]))
    w = np.cross(z, t)
    cands = [np.cos(a) * t + np.sin(a) * w for a in np.radians(np.arange(0.0, 180.0, 2.0))]
    best = min(cands, key=lambda d: float(np.ptp(P @ d)))
    width = float(np.ptp(P @ best))
    for d in prefer:
        d = _unit(d - (d @ z) * z)
        if np.linalg.norm(d) > 0.5 and float(np.ptp(P @ d)) <= width * (1.0 + 1e-3) + 1e-12:
            return d
    return best


def frame_for(P, p, line=None):
    """The deformation's frame for the (local) points P of the part and the
    step's params; line: the axis' (start, end), local, for axis LINE."""
    P = np.asarray(P, float).reshape(-1, 3)
    axes = np.eye(3)
    if line is not None:
        A, B = (np.asarray(v, float) for v in line)
        z, O, L = _unit(B - A), A, float(np.linalg.norm(B - A))
        if L < 1e-12:
            raise ValueError("the axis line has no length")
    else:
        if not len(P):
            raise ValueError("nothing to deform")
        lo, hi = P.min(axis=0), P.max(axis=0)
        k = {"X": 0, "Y": 1, "Z": 2}.get(p.get("deform_axis", "AUTO"), int(np.argmax(hi - lo)))
        z = axes[k]
        L = float(hi[k] - lo[k])
        c = 0.5 * (lo + hi)
        O = c.copy()
        O[k] = lo[k]
        if L < 1e-9 * (float(np.linalg.norm(hi - lo)) or 1.0):
            raise ValueError("the part has no extent along that axis")
    x = _thinnest(P, z, prefer=[a for a in axes if abs(a @ z) < 0.5]) if len(P) else _unit(np.cross(z, axes[0]))
    turn = float(p.get("deform_direction", 0.0))
    if turn:
        x = rotate_points(x[None], np.zeros(3), z, turn)[0]
    fr = Frame(O, z, x, L)
    fr.spans_part = line is None        # the axis runs over the whole part: nothing lies past its ends
    return fr


# --------------------------------------------------------------------------
# The map
# --------------------------------------------------------------------------

class DeformMap:
    """Local points -> deformed local points; also what Flow's mapping code
    asks of a map (map_labeled / map_by: one piece)."""

    def __init__(self, kind, frame, p, size):
        self.kind, self.f, self.size = kind, frame, float(size)
        self.smooth = bool(p.get("deform_smooth", False))
        # past the ends of an axis that spans the part there is only the rest of an untrimmed
        # surface (and the pad a face is cut down with): the rate carries on there, so the map
        # has no corner the refit would have to follow (a straight ramp held at its ends does)
        self.infinite = (bool(p.get("deform_infinite", False)) or getattr(frame, "spans_part", False)) \
            and not self.smooth
        self.flat = bool(p.get("deform_flat", False))
        self.angle = math.radians(float(p.get("deform_angle", 90.0)))
        self.factor = float(p.get("deform_factor", 0.5))
        self._table = None

    # g(s) and its slope along s
    def ramp(self, s):
        L = self.f.L
        t = s / L
        if not self.infinite:
            t = np.clip(t, 0.0, 1.0)
        g, dg = ease(t, self.smooth)
        inside = (s >= 0.0) & (s <= L) if not self.infinite else np.ones_like(s, bool)
        return g, np.where(inside, dg / L, 0.0)

    # -- bend: the bent axis C(s) = integral of (cos theta, sin theta) ---------------
    def _arc(self, sc):
        """(along z, along x) of the bent axis at sc (inside the bent stretch)."""
        if self.smooth:
            return self._smooth_arc(sc)
        k = self.angle / self.f.L
        x = k * sc
        small = np.abs(x) < 1e-4
        xs = np.where(small, 1.0, x)
        cz = np.where(small, sc * (1.0 - x * x / 6.0 + x ** 4 / 120.0), np.sin(xs) / np.where(small, 1.0, k))
        cx = np.where(small, sc * (x / 2.0 - x ** 3 / 24.0), (1.0 - np.cos(xs)) / np.where(small, 1.0, k))
        return cz, cx

    def _smooth_arc(self, sc):
        if self._table is None:
            ax = surfmap.Axis(np.linspace(0.0, self.f.L, 129))
            th = self.angle * ease(ax.gauss / self.f.L, True)[0]
            cz = ax.cumulative(np.cos(th))
            cx = ax.cumulative(np.sin(th))
            tn = self.angle * ease(ax.t / self.f.L, True)[0]
            self._table = (ax.t, cz, cx, np.cos(tn), np.sin(tn))
        t, cz, cx, dz, dx = self._table
        i = np.clip(np.searchsorted(t, sc) - 1, 0, len(t) - 2)
        h = t[i + 1] - t[i]
        u = (sc - t[i]) / h
        h00, h10, h01, h11 = (2 * u ** 3 - 3 * u * u + 1, u ** 3 - 2 * u * u + u, -2 * u ** 3 + 3 * u * u, u ** 3 - u * u)
        return (h00 * cz[i] + h10 * h * dz[i] + h01 * cz[i + 1] + h11 * h * dz[i + 1],
                h00 * cx[i] + h10 * h * dx[i] + h01 * cx[i + 1] + h11 * h * dx[i + 1])

    def map(self, Q):
        Q = np.asarray(Q, float).reshape(-1, 3)
        if not len(Q):
            return Q.copy()
        s, a, b = self.f.coords(Q)
        g, _ = self.ramp(s)
        if self.kind == "TWIST":
            phi = self.angle * g
            c, sn = np.cos(phi), np.sin(phi)
            return self.f.point(s, a * c - b * sn, a * sn + b * c)
        if self.kind == "TAPER":
            k = 1.0 + (self.factor - 1.0) * g
            return self.f.point(s, a * k, b if self.flat else b * k)
        # BEND
        L = self.f.L
        sc = s if self.infinite else np.clip(s, 0.0, L)
        th = self.angle * g
        cz, cx = self._arc(sc)
        over = s - sc                                     # past an end: straight on along its tangent
        T = (np.cos(th), np.sin(th))
        N = (-np.sin(th), np.cos(th))
        f = self.f
        return (f.O + np.outer(cz + over * T[0] + a * N[0], f.z) + np.outer(cx + over * T[1] + a * N[1], f.x)
                + np.outer(b, f.y))

    def squeeze(self, Q):
        """The least local volume ratio over the points Q (<= 0: the map folds there)."""
        s, a, b = self.f.coords(Q)
        g, dg = self.ramp(s)
        if self.kind == "TWIST" or not len(Q):
            return 1.0
        if self.kind == "TAPER":
            k = 1.0 + (self.factor - 1.0) * g
            return float((np.abs(k) * np.sign(k) if self.flat else k * np.abs(k)).min())
        return float((1.0 - self.angle * dg * a).min())

    def map_labeled(self, Q):
        X = self.map(Q)
        return X, np.zeros(len(X), int)

    def map_by(self, Q, k):
        return self.map(Q)


def fold_message(fmap, P):
    """Why the map can't be applied to the part (points P), or ''."""
    if not len(P):
        return ""
    q = fmap.squeeze(P)
    if q > FOLD_MIN:
        return ""
    if fmap.kind == "TAPER":
        return "The taper squeezes the part to nothing: use a factor above 0"
    s, a, _ = fmap.f.coords(P)
    R = fmap.f.L / max(abs(fmap.angle) * (1.875 if fmap.smooth else 1.0), 1e-12)
    return (f"Bent too tight: the inside folds over (bend radius {R:.4g} < the part's "
            f"{float(np.abs(a).max()):.4g} from the axis) -- less angle, a longer axis or Smooth off")


# --------------------------------------------------------------------------
# Brep and curves through the map
# --------------------------------------------------------------------------

def _snap(U, t, eps):
    k = int(np.argmin(np.abs(U - t)))
    return float(U[k]) if abs(U[k] - t) <= eps else float(t)


def _crop_dir(G, U, p, a, b):
    """The homogeneous net G (n, m, 4) cut down to [a, b] in its first
    direction (Boehm knot insertion: exact, same parameters)."""
    from .kernel.natext import insert_knot
    U = np.asarray(U, float)
    eps = 1e-10 * (U[-1] - U[0] or 1.0)
    a, b = _snap(U, a, eps), _snap(U, b, eps)
    rows, Ul = [np.array(r) for r in G], list(map(float, U))
    for t in (a, b):
        if t <= Ul[p] + eps or t >= Ul[len(rows)] - eps:
            continue
        m = sum(1 for k in Ul if abs(k - t) <= eps)
        for _ in range(max(0, p - m)):
            rows, Ul = insert_knot(rows, Ul, p, t)
    Ua = np.array(Ul)
    ia = int(np.nonzero(np.abs(Ua - a) <= eps)[0].max())
    ib = int(np.nonzero(np.abs(Ua - b) <= eps)[0].min())
    r = max(1, ia - p + 1)
    s = min(len(rows), ib)
    out = np.stack(rows[r - 1:s])
    knots = [a] + list(Ua[r:s + p]) + [b]
    return out, np.array(knots)


def trim_box(face, S):
    """((u0, v0), (u1, v1)): the (u, v) box of face's trim loops on S (its
    uv_domain() is the whole surface's domain on some faces, e.g. a cap plane)."""
    uv = np.vstack([q for lp in face.boundary_loops() if lp for q, _ in surfmap.loop_uvs(S, lp)])
    return tuple(uv.min(axis=0)), tuple(uv.max(axis=0))


def crop_to_trim(face, S, pad=0.02, box=None):
    """S cut down to face's trim box grown by pad of it (the same surface over
    the same parameters), or S itself when that would hardly be smaller."""
    try:
        (u0, v0), (u1, v1) = box if box is not None else trim_box(face, S)
    except Exception:
        return S
    (a, b), (c, d) = S.u_domain, S.v_domain
    gu, gv = pad * (u1 - u0), pad * (v1 - v0)
    lo = (max(a, u0 - gu), max(c, v0 - gv))
    hi = (min(b, u1 + gu), min(d, v1 + gv))
    cut_u = lo[0] > a + 0.01 * (b - a) or hi[0] < b - 0.01 * (b - a)
    cut_v = lo[1] > c + 0.01 * (d - c) or hi[1] < d - 0.01 * (d - c)
    if not (cut_u or cut_v) or hi[0] <= lo[0] or hi[1] <= lo[1]:
        return S
    G, U, V = S.Pw, S.U, S.V
    try:
        if cut_u:
            G, U = _crop_dir(G, U, S.p, lo[0], hi[0])
        if cut_v:
            Gt, V = _crop_dir(np.swapaxes(G, 0, 1), V, S.q, lo[1], hi[1])
            G = np.swapaxes(Gt, 0, 1)
        _, ext = store._k()
        w = G[..., 3]
        P = (G[..., :3] / w[..., None]).reshape(-1, 3)
        W = [float(x) for x in w.ravel()] if S.rational else None
        return surfeval.Surf(ext.surface_from_cvs(S.p, S.q, [tuple(map(float, q)) for q in P],
                                                  [float(x) for x in U], [float(x) for x in V], W))
    except Exception:
        return S


def deform_brep(b, fmap):
    """The brep b (local) through fmap: (Brep, worst fit miss)."""
    sm, ext = store._k()
    size = fmap.size
    tol = SURF_TOL * size
    faces = list(b.faces())
    parts, worst = [], 0.0
    for f in faces:
        try:
            S = surfeval.Surf(f.surface())
        except Exception:
            raise ValueError("a face is not a B-spline surface")
        natural = len(faces) == 1 and surfmap.is_natural_face(f, S)
        if not natural:
            S = crop_to_trim(f, S)
        C, U, p, V, q, err = flow.map_surface(S, fmap, tol)
        worst = max(worst, err)
        make = lambda C=C, U=U, p=p, V=V, q=q: surfmap.surface_from_net(C, U, p, V, q)
        if natural:
            parts.append(ext.brep_from_surface(make()))
        else:
            def to_world(uv, S=S):
                return fmap.map(S.at(uv[:, 0], uv[:, 1], 0)[0, 0])
            parts.append(surfmap.trimmed_face(f, S, make, to_world, size))
    out = parts[0]
    for o in parts[1:]:
        ext.append_brep(out, o)
    if len(parts) > 1:
        out = sm.stitch_advanced(out, 1e-5 * size, squeeze_small_edges=False, making_manifold_solid=False)[0]
        try:
            h = sm.heal_brep(out)
            if h is not None and h.face_count():
                out = h
        except Exception:
            pass
    return out, worst


def curve_specs(obj, fmap):
    """Spline specs (local) of a curve object's splines through fmap."""
    out = []
    for sp in obj.data.splines:
        try:
            spans = curvejoin.spline_spans(sp)
        except Exception:
            continue
        out += flow.flow_spans(spans, bool(sp.use_cyclic_u), fmap, fmap.size, CURVE_TOL * fmap.size)
    return out


def _size(P):
    return float(np.linalg.norm(np.ptp(P, axis=0))) if len(P) else 1.0


def line_local(obj, line_obj):
    """(start, end) of a straight line object, in obj's local space."""
    kc = features.input_curve(line_obj, "S0")
    P = geom.sample_curve(kc, 2)
    Mi = np.linalg.inv(np.array(geom.world_matrix(obj), float))
    return [Mi[:3, :3] @ q + Mi[:3, 3] for q in (np.asarray(P[0], float), np.asarray(P[-1], float))]


def step_line(obj, inputs, p):
    """The axis line of a step (local), or None: the live line input, else the
    one recorded with the step."""
    if p.get("deform_axis") != "LINE":
        return None
    if inputs:
        try:
            return line_local(obj, inputs[0][0])
        except Exception:
            pass
    ln = p.get("deform_line")
    if not ln:
        raise ValueError("the axis line is missing")
    return [np.asarray(v, float) for v in ln]


def map_for(P, p, line=None):
    fr = frame_for(P, p, line)
    size = max(_size(P), fr.L)
    return DeformMap(p.get("deform_kind", "TWIST"), fr, p, size)


def deform_text(text, p, line=None):
    """Brep text (local) deformed: (text, info)."""
    b = store.brep_from_text(text)
    P = points_of_brep(b)
    fmap = map_for(P, p, line)
    why = fold_message(fmap, P)
    if why:
        raise ValueError(why)
    t0 = time.time()
    out, err = deform_brep(b, fmap)
    info = {"fit": err / fmap.size, "faces": out.face_count(), "time": time.time() - t0,
            "closed": bool(b.is_manifold_solid()) and bool(out.is_manifold_solid()),
            "opened": bool(b.is_manifold_solid()) and not bool(out.is_manifold_solid())}
    return store.brep_to_text(out), info


def apply_deform(obj, text, inputs, p):
    out, info = deform_text(text, p, step_line(obj, inputs, p))
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return out


def step_label(p):
    kind = p.get("deform_kind", "TWIST")
    if kind == "TAPER":
        what = f"×{float(p.get('deform_factor', 0.5)):.3g}" + (" flat" if p.get("deform_flat") else "")
    else:
        what = f"{float(p.get('deform_angle', 90.0)):.3g}°"
    extra = [n for n, k in (("smooth", "deform_smooth"), ("infinite", "deform_infinite")) if p.get(k)]
    axis = p.get("deform_axis", "AUTO")
    return f"{LABEL.get(kind, kind.title())} · {what}" + (f" · {axis.title()}" if axis != "AUTO" else "") \
        + "".join(" · " + e for e in extra)


def _register_step():
    from . import steps
    steps.APPLY[OP] = apply_deform
    steps.LABEL[OP] = step_label
    features.KEEP_STEPS.add(OP)          # plain stays plain, trimmed stays trimmed


_register_step()


# --------------------------------------------------------------------------
# The operator
# --------------------------------------------------------------------------

def _drop_last_step(obj, context):
    """Take back a deform step that failed (as Remove History Step does)."""
    st = features.stack(obj)
    if not st or st[-1]["op"] != OP:
        return
    features.remove_step(obj, len(st) - 1)
    if features.PROP_STACK not in obj and features.PROP_HIST not in obj and features.PROP_BASE in obj:
        nat = obj.get(features.PROP_BASE + "_natural", 0)
        store.set_geometry_text(obj, obj[features.PROP_BASE], context, keep_untrimmed=False)
        obj[store.PROP_NATURAL] = nat
        del obj[features.PROP_BASE]
        for k in (features.PROP_BASE + "_natural",):
            if k in obj:
                del obj[k]
    else:
        features.rebuild(obj, context, force=True)


def _is_line(obj):
    if obj is None or obj.type != 'CURVE' or len(obj.data.splines) != 1:
        return False
    from .ops import _straight_line
    try:
        return _straight_line(features.input_curve(obj, "S0")) is not None
    except Exception:
        return False


class NPROTO_OT_deform(ValueDrag, bpy.types.Operator):
    """Twist, bend or taper NURBS objects (solids too) and curves along an axis,
    in place. The axis is the part's longest extent, the object's X / Y / Z, or a
    straight line selected last. From the UI the angle (taper: the end scale)
    has a handle on a live preview; type a value, click away or Enter to apply"""
    bl_idname = "nproto.deform"
    bl_label = "Deform"
    bl_options = {'REGISTER', 'UNDO'}

    kind: EnumProperty(name="Deform", items=KINDS, default="TWIST")
    angle: FloatProperty(name="Angle", default=90.0, soft_min=-720.0, soft_max=720.0,
                         description="Twist: the turn at the axis end. Bend: how far the axis end turns")
    factor: FloatProperty(name="End Scale", default=0.5, soft_min=0.0, soft_max=4.0,
                          description="Taper: the scale across the axis at its end (1 at its start)")
    axis: EnumProperty(name="Axis", items=AXES, default="AUTO")
    direction: FloatProperty(name="Direction", default=0.0, soft_min=-180.0, soft_max=180.0,
                             description="Bend / Flat Taper: the bend direction turned about the axis from the "
                                         "part's thinnest side (degrees)")
    smooth: BoolProperty(name="Smooth", default=False,
                         description="Ease in and out at the axis ends: no crease where the deformed stretch "
                                     "meets the rest")
    infinite: BoolProperty(name="Infinite", default=False,
                           description="Go on deforming past the axis ends (else the rest moves rigidly)")
    flat: BoolProperty(name="Flat", default=False, description="Taper: scale along the bend direction only")
    line_start: FloatVectorProperty(name="Line Start", size=3, subtype='XYZ', unit='LENGTH',
                                    description="Axis Line: where it starts (world)")
    line_end: FloatVectorProperty(name="Line End", size=3, subtype='XYZ', unit='LENGTH', default=(0, 0, 1),
                                  description="Axis Line: where it ends (world)")
    line: StringProperty(name="Line", description="The line the axis follows (history: live)")
    objects: StringProperty(options={'HIDDEN'})          # JSON names (the redo panel keeps them)

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    @classmethod
    def description(cls, context, props):
        k = getattr(props, "kind", "TWIST")
        return {"TWIST": "Twist NURBS objects and curves about an axis, more and more along it",
                "BEND": "Bend NURBS objects and curves: the axis bends into an arc, the part with it",
                "TAPER": "Taper NURBS objects and curves: scale them across an axis, more and more along it"}[k]

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "kind")
        if self.kind == "TAPER":
            lay.prop(self, "factor")
            lay.prop(self, "flat")
        else:
            lay.prop(self, "angle")
        lay.prop(self, "axis")
        if self.axis == "LINE":
            lay.prop_search(self, "line", context.scene, "objects")
        if self.kind == "BEND" or (self.kind == "TAPER" and self.flat):
            lay.prop(self, "direction")
        lay.prop(self, "smooth")
        sub = lay.row()
        sub.active = not self.smooth
        sub.prop(self, "infinite")
        inf = LAST_INFO
        if inf.get("fit") is not None:
            lay.label(text=f"Fit {inf['fit']:.1e} of the size · {inf.get('time', 0):.2f} s", icon='MESH_GRID')

    # -- inputs -------------------------------------------------------------------------
    def _objects(self, context):
        try:
            names = json.loads(self.objects or "[]")
        except Exception:
            names = []
        objs = [bpy.data.objects.get(n) for n in names]
        objs = [o for o in objs if o is not None and o.name != self.line]
        if not objs:
            objs = [o for o in context.selected_objects if (store.is_nurbs(o) or o.type == 'CURVE')
                    and o.name != self.line]
        return objs

    def params(self, obj=None):
        p = {"deform_kind": self.kind, "deform_angle": float(self.angle), "deform_factor": float(self.factor),
             "deform_axis": self.axis, "deform_direction": float(self.direction),
             "deform_smooth": bool(self.smooth), "deform_infinite": bool(self.infinite),
             "deform_flat": bool(self.flat)}
        if self.axis == "LINE" and obj is not None:
            p["deform_line"] = [list(map(float, v)) for v in self._line_local(obj)]
        return p

    def _line_world(self):
        lo = bpy.data.objects.get(self.line) if self.line else None
        if lo is not None and _is_line(lo):
            P = geom.sample_curve(features.input_curve(lo, "S0"), 2)
            return np.asarray(P[0], float), np.asarray(P[-1], float)
        return np.array(self.line_start, float), np.array(self.line_end, float)

    def _line_local(self, obj):
        Mi = np.linalg.inv(np.array(geom.world_matrix(obj), float))
        return [Mi[:3, :3] @ q + Mi[:3, 3] for q in self._line_world()]

    def _local_points(self, obj):
        if store.is_nurbs(obj):
            return points_of_brep(store.get_brep(obj))
        return points_of_curve(obj)

    def _map(self, obj):
        p = self.params(obj)
        return map_for(self._local_points(obj), p, p.get("deform_line"))

    def _gather(self, context):
        """The objects from the selection; a straight line selected last (active)
        with them is the axis. False when there is nothing to deform."""
        act = context.view_layer.objects.active
        sel = [o for o in context.selected_objects if store.is_nurbs(o) or o.type == 'CURVE']
        if act in sel and len(sel) > 1 and _is_line(act):
            self.axis, self.line = "LINE", act.name
            a, b = self._line_world()
            self.line_start, self.line_end = tuple(a), tuple(b)
            sel = [o for o in sel if o is not act]
        elif self.axis == "LINE" and not self.line:
            self.axis = "AUTO"
        self.objects = json.dumps([o.name for o in sel])
        return bool(sel)

    def invoke(self, context, event):
        self.line = ""
        if not self._gather(context):
            self.report({'WARNING'}, "Select NURBS objects or curves to deform (and a straight line last for the axis)")
            return {'CANCELLED'}
        return self.ask(context)

    def execute(self, context):
        from .ops import _kernel_ok, _select_only
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not self.objects:              # a script: the selection, as the UI takes it
            self._gather(context)
        if self.axis == "LINE":
            lo = bpy.data.objects.get(self.line) if self.line else None
            if lo is not None and not _is_line(lo):
                self.report({'WARNING'}, f"{lo.name} is not a straight line: Auto axis used")
                self.axis, self.line = "AUTO", ""
        objs = self._objects(context)
        if not objs:
            self.report({'WARNING'}, "Select NURBS objects or curves to deform")
            return {'CANCELLED'}
        hist = features.history_enabled(context)
        done, failed, worst = [], [], 0.0
        line_obj = bpy.data.objects.get(self.line) if self.axis == "LINE" and self.line else None
        for o in objs:
            p = self.params(o)
            try:
                if o.type == 'CURVE':
                    if features.has_history(o):
                        raise ValueError("has history -- bake it first (curves deform without history)")
                    P = points_of_curve(o)
                    fmap = map_for(P, p, p.get("deform_line"))
                    why = fold_message(fmap, P)
                    if why:
                        raise ValueError(why)
                    from .primitives import write_spline_object
                    write_spline_object(o, curve_specs(o, fmap))
                elif hist:
                    inputs = [(line_obj, "S0")] if line_obj is not None else []
                    features.push_step(o, OP, inputs, p)
                    try:
                        features.rebuild(o, context, force=True)
                    except Exception:
                        _drop_last_step(o, context)
                        raise
                    worst = max(worst, LAST_INFO.get("fit", 0.0))
                else:
                    natural = store.is_natural(o)
                    text, info = deform_text(o[store.PROP_BREP], p, p.get("deform_line"))
                    LAST_INFO.clear()
                    LAST_INFO.update(info)
                    store.set_geometry_text(o, text, context, keep_untrimmed=False)
                    if store.PROP_UNTRIM in o:
                        del o[store.PROP_UNTRIM]          # the untrimmed surface no longer fits
                    if natural and store.get_brep(o).face_count() == 1:
                        o[store.PROP_NATURAL] = 1
                    worst = max(worst, info["fit"])
                done.append(o)
            except Exception as exc:
                failed.append(f"{o.name}: {exc}")
        if not done:
            self.report({'ERROR'}, f"{LABEL[self.kind]} failed: " + "; ".join(failed))
            return {'CANCELLED'}
        _select_only(context, done)
        msg = f"{LABEL[self.kind]}: {len(done)} object{'s' if len(done) != 1 else ''}"
        if worst:
            msg += f" · fit {worst:.1e} of the size"
        if LAST_INFO.get("opened"):
            msg += " · the solid came out open"
        if failed:
            msg += " -- failed: " + "; ".join(failed)
        self.report({'WARNING'} if failed or LAST_INFO.get("opened") else {'INFO'}, msg)
        return {'FINISHED'}

    # -- the live value stage (valuedrag.py) ------------------------------------------
    angle_fields = ("angle",)
    ratio_fields = ("factor",)
    ask_dialog = False
    fit_view = "none"
    MAX_FILL = 80000

    def prepare_value(self, context):
        pass

    def drag_setup(self, context, mouse):
        self.drag_fields = ("factor",) if self.kind == "TAPER" else ("angle",)
        self._dv = []
        for o in self._objects(context):
            M = np.array(geom.world_matrix(o), float)
            if store.is_nurbs(o):
                me = o.data
                me.calc_loop_triangles()
                n = len(me.vertices)
                V = np.empty(n * 3)
                me.vertices.foreach_get("co", V)
                T = np.empty(len(me.loop_triangles) * 3, np.int64)
                me.loop_triangles.foreach_get("vertices", T)
                E = [np.asarray(pl, float) for pl in store.display(o).get("edges", []) if len(pl) > 1]
                self._dv.append((o, M, V.reshape(-1, 3), T.reshape(-1, 3), E))
            else:
                E = []
                for sp in o.data.splines:
                    try:
                        P, _ = curvetrim.sample_spans(curvejoin.spline_spans(sp), 24)
                        E.append(np.asarray(P, float))
                    except Exception:
                        pass
                self._dv.append((o, M, np.zeros((0, 3)), np.zeros((0, 3), np.int64), E))
        return bool(self._dv)

    def _world(self, M, X):
        return X @ M[:3, :3].T + M[:3, 3]

    def _frame_world(self):
        o, M = self._dv[0][0], self._dv[0][1]
        fm = self._map(o)
        f = fm.f
        R = M[:3, :3]
        z, x = _unit(R @ f.z), _unit(R @ f.x)
        P = self._local_points(o)
        s, a, b = f.coords(P) if len(P) else (np.zeros(1), np.zeros(1), np.zeros(1))
        r = float(np.sqrt(a * a + b * b).max(initial=0.0)) or 0.1 * f.L
        return fm, self._world(M, f.O[None])[0], z, x, np.cross(z, x), f.L * float(np.linalg.norm(R @ f.z)), r, M

    def drag_handles(self):
        fm, O, z, x, y, L, r, M = self._frame_world()
        if self.kind == "TAPER":
            E = O + L * z
            return [{"field": "factor", "kind": "arrow", "origin": E, "dir": x, "param": self.factor * r,
                     "base": E, "to_value": lambda t, r=r: t / r, "snap": (0.1 * r, 0.01 * r)}]
        if self.kind == "TWIST":
            E = O + L * z
            a = self.angle
            P0 = E + r * 1.15 * x
            n = max(2, int(abs(a) / 3.0) + 1)
            path = np.vstack([rotate_points(P0[None], E, z, a * i / (n - 1)) for i in range(n)])
            back = a - math.copysign(min(6.0, abs(a) * 0.5), a) if abs(a) > 1e-9 else 0.0
            return [{"field": "angle", "kind": "arrow", "pos": rotate_points(P0[None], E, z, a)[0], "param": a,
                     "origin": E, "dir": None, "path": path, "head_from": rotate_points(P0[None], E, z, back)[0],
                     "pick": lambda o, rr, prev: angle_pick(E, x, y, z, o, rr, prev),
                     "to_value": lambda t: t, "snap": (5.0, 1.0)}]
        # BEND: the bent axis' end; its chord from the start turns half the angle
        Ml = self._dv[0][1]
        f = fm.f
        n = 33
        loc = fm.map(f.point(np.linspace(0.0, f.L, n), np.zeros(n), np.zeros(n)))
        path = self._world(Ml, loc)

        def pick(ro, rd, prev):
            t = angle_pick(O, z, x, y, ro, rd, 0.5 * prev)
            return None if t is None else max(-358.0, min(358.0, 2.0 * t))
        return [{"field": "angle", "kind": "arrow", "pos": path[-1], "param": self.angle, "origin": O, "dir": None,
                 "path": path, "head_from": path[-2], "pick": pick, "to_value": lambda t: t, "snap": (5.0, 1.0)}]

    def drag_preview(self):
        lines, tris, guides = [], [], []
        total = sum(len(T) for _, _, _, T, _ in self._dv)
        for o, M, V, T, E in self._dv:
            try:
                fm = self._map(o)
            except Exception:
                continue
            for pl in E:
                lines.append(self._world(M, fm.map(pl)))
            if len(T) and total <= self.MAX_FILL:
                tris.append(self._world(M, fm.map(V))[T].reshape(-1, 3))
        try:
            fm, O, z, x, y, L, r, M = self._frame_world()
            guides.append((np.array([O, O + L * z]), (0.55, 0.85, 1.0, 0.9)))
        except Exception:
            pass
        return {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3)), "guides": guides}

    def drag_check(self):
        for o, *_ in self._dv:
            try:
                fm = self._map(o)
                why = fold_message(fm, self._local_points(o))
            except Exception as exc:
                why = str(exc)
            if why:
                return why
        return ""

    def drag_key(self, context, event, ch):
        if ch == "s":
            self.smooth = not self.smooth
        elif ch == "i":
            self.infinite = not self.infinite
        elif ch == "f" and self.kind == "TAPER":
            self.flat = not self.flat
        elif ch == "d":
            self.direction = (self.direction + 90.0 + 180.0) % 360.0 - 180.0
        elif ch in ("x", "y", "z", "a") and self.axis != "LINE":
            self.axis = {"x": "X", "y": "Y", "z": "Z", "a": "AUTO"}[ch]
        elif ch == "t":
            order = ["TWIST", "BEND", "TAPER"]
            self.kind = order[(order.index(self.kind) + 1) % 3]
            self.drag_fields = ("factor",) if self.kind == "TAPER" else ("angle",)
            self._field = 0
        else:
            return False
        return True

    def drag_rows(self):
        rows = [("Ends", [hints.toggle("S", "smooth", self.smooth), hints.toggle("I", "infinite", self.infinite)])]
        if self.axis != "LINE":
            rows.append(("Axis", [hints.key("X Y Z", "local axis"), hints.key("A", "auto"),
                                  hints.note(self.axis.title())]))
        keys = [hints.key("T", "twist / bend / taper")]
        if self.kind == "BEND" or (self.kind == "TAPER" and self.flat):
            keys.append(hints.key("D", "turn the direction 90°"))
        if self.kind == "TAPER":
            keys.append(hints.toggle("F", "flat", self.flat))
        rows.append(("Shape", keys))
        return rows

    def drag_state(self):
        out = [LABEL[self.kind]]
        if self.smooth:
            out.append("Smooth")
        if self.infinite and not self.smooth:
            out.append("Infinite")
        return out


classes = (NPROTO_OT_deform,)
