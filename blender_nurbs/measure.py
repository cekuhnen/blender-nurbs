"""Measure (Analyze), Rhino / Alias / Shapr3D style: no modes, it reads
whatever is under the mouse.

* Hover: a point (an end, a middle, a centre, a CV: the drawing tools'
  snapping) reads its X Y Z; a curve or a surface edge its radius (a true
  circle or an arc of one: exact radius and centre; any other curve: the
  radius of curvature at the nearest point, the osculating circle drawn;
  straight) and its whole length; a NURBS face its two principal radii there
  (a cylinder: its radius and flat; a sphere: the same both ways), with the
  osculating circles.
* Click one thing, then hover or click a second: what the two measure against
  each other (measurepair): two points their distance and ΔX ΔY ΔZ (X / Y / Z
  lock the second point to that axis, as in the drawing tools); anything else
  the shortest distance, the angle (two edges meeting at a corner, two faces
  along their edge: the angle between them there), parallel faces' / lines' /
  axes' distance apart, and centre to centre for circles, holes and spheres.
  Click the first thing again to keep its own reading.
* D: radius / diameter. Alt: a plain point (no edge or face). Ctrl+C copies
  the last reading. P pins the reading under the mouse (else the last kept):
  it stays in the file and follows the model (measurepin). K pins a
  curvature comb on the curve or edge under the mouse.

A kept reading stays on screen while the tool runs and goes to the Info log;
Backspace drops the first pick or the last reading, Esc / right click ends.
Nothing in the scene changes. No driving dimensions: readings only.
"""

import math

import numpy as np

from . import features, linesolve, store, surfeval

STRAIGHT_K = 1e-9          # curvature (x size) below this: straight / flat
OVERLAY = {"items": [], "live": None, "first": []}


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-300 else v * 0.0


# --------------------------------------------------------------------------
# Readings (pure)
# --------------------------------------------------------------------------

def distance(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    d = B - A
    return {"kind": "distance", "A": A, "B": B, "length": float(np.linalg.norm(d)), "delta": d}


def _circle3(A, B, C):
    ab, ac = B - A, C - A
    n = np.cross(ab, ac)
    nn = float(n @ n)
    if nn <= 1e-24 * max(float(ab @ ab) * float(ac @ ac), 1e-300):
        return None
    O = A + (float(ac @ ac) * np.cross(n, ab) + float(ab @ ab) * np.cross(ac, n)) / (2.0 * nn)
    return O, float(np.linalg.norm(A - O)), n / math.sqrt(nn)


def curve_radius(c, P):
    """Reading of a linesolve Curve at its point nearest P: an exact circle
    (all of it on one circle: its radius and centre), else the radius of
    curvature there, or straight."""
    P = np.asarray(P, float)
    W = c.W
    if not c.straight:
        c3 = _circle3(W[0], W[len(W) // 3], W[(2 * len(W)) // 3])
        if c3 is not None:
            O, r, n = c3
            if float(np.abs(np.linalg.norm(W - O, axis=1) - r).max()) <= 1e-7 * r \
                    and float(np.abs((W - O) @ n).max()) <= 1e-7 * r:
                q = P - O
                q -= (q @ n) * n
                Q = O + r * _unit(q if np.linalg.norm(q) > 1e-12 else W[0] - O)
                return {"kind": "radius", "source": "circle", "radius": r, "centre": O, "point": Q, "normal": n,
                        "full": bool(c.closed)}
    t = linesolve.nearest_t(c, P)
    _, Q, D, DD = c.at(t)
    sp = float(np.linalg.norm(D))
    if sp < 1e-300:
        return None
    k = float(np.linalg.norm(np.cross(D, DD))) / sp ** 3
    if k * c.size < STRAIGHT_K:
        return {"kind": "radius", "source": "straight", "radius": math.inf, "point": Q, "tangent": D / sp}
    T = D / sp
    N = _unit(DD - (DD @ T) * T)
    return {"kind": "radius", "source": "curvature", "radius": 1.0 / k, "centre": Q + N / k, "point": Q,
            "normal": _unit(np.cross(T, N)), "full": False, "tangent": T}


def surface_radius(obj, P):
    """Principal radii of curvature of the NURBS object's face nearest P:
    {"point", "normal", "principal": [(radius or inf, unit direction, centre or None)]}."""
    from .surfoffset import target_of
    tg = target_of([obj])
    F, UV, S, N, d = tg.feet_all(np.atleast_2d(np.asarray(P, float)))
    if F[0] < 0:
        return None
    E = tg.faces[F[0]].E
    u, v = UV[0]
    D = E.at(np.array([u]), np.array([v]), 2)
    D1 = {kl: a[0] for kl, a in D.items()}
    su, sv = D1[1, 0], D1[0, 1]
    n = _unit(np.cross(su, sv))
    I = np.array([[su @ su, su @ sv], [su @ sv, sv @ sv]])
    II = np.array([[D1[2, 0] @ n, D1[1, 1] @ n], [D1[1, 1] @ n, D1[0, 2] @ n]])
    try:
        Sh = np.linalg.solve(I, II)                  # the shape operator in (u, v)
    except np.linalg.LinAlgError:
        return None
    ks, vecs = np.linalg.eig(Sh)
    size = tg.size
    out = []
    for j in np.argsort(-np.abs(ks.real)):
        k = float(ks.real[j])
        dvec = _unit(vecs[0, j].real * su + vecs[1, j].real * sv)
        if abs(k) * size < STRAIGHT_K:
            out.append((math.inf, dvec, None))
        else:
            out.append((1.0 / abs(k), dvec, S[0] + n / k))
    if len(out) == 2 and np.linalg.norm(np.cross(out[0][1], out[1][1])) < 1e-6:   # an umbilic: any two directions
        t2 = _unit(np.cross(n, out[0][1]))
        out[1] = (out[1][0], t2, out[1][2])
    return {"kind": "surface", "point": S[0], "normal": n, "principal": out, "object": obj.name}


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

def _fmt(context, x):
    from .measureinfo import fmt
    return fmt(context, x)


def label(context, m, diameter=False):
    if m is None:
        return ""
    if m["kind"] == "pair":
        from . import measurepair
        return measurepair.text(context, m["r"])
    if m["kind"] == "point":
        x, y, z = m["point"]
        return f"X {_fmt(context, x)}  Y {_fmt(context, y)}  Z {_fmt(context, z)}"
    if m["kind"] == "distance":
        dx, dy, dz = m["delta"]
        return (f"{_fmt(context, m['length'])}   ΔX {_fmt(context, dx)}  ΔY {_fmt(context, dy)}  "
                f"ΔZ {_fmt(context, dz)}")
    sym, f = ("Ø", 2.0) if diameter else ("R", 1.0)
    if m["kind"] == "radius":
        if m["source"] == "straight":
            return "straight" + (f"  ·  L {_fmt(context, m['length'])}" if m.get("length") is not None else "")
        what = {"circle": "circle", "curvature": "curvature"}[m["source"]]
        txt = f"{sym} {_fmt(context, f * m['radius'])}  ·  {what}"
        return txt + (f"  ·  L {_fmt(context, m['length'])}" if m.get("length") is not None else "")
    parts = []
    for r, _, _ in m["principal"]:
        parts.append("flat" if math.isinf(r) else f"{sym} {_fmt(context, f * r)}")
    a, b = parts
    if a == b:
        return f"{a}  ·  both ways"
    return f"{a}  ·  {b}  (principal)"


# --------------------------------------------------------------------------
# Drawing (own handlers while the tool runs)
# --------------------------------------------------------------------------

COL_LIVE = (1.0, 0.85, 0.2, 1.0)
COL_KEPT = (0.85, 0.9, 1.0, 0.95)
COL_FIRST = (0.3, 0.85, 1.0, 1.0)      # the first of two picks
_handles = []


def _circle_pts(C, r, n, k=96, x=None):
    n = _unit(n)
    x = _unit(x if x is not None else np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]))
    x = _unit(x - (x @ n) * n)
    y = np.cross(n, x)
    a = np.linspace(0, 2 * math.pi, k + 1)
    return np.asarray(C, float) + r * (np.outer(np.cos(a), x) + np.outer(np.sin(a), y))


def geometry(m):
    """(polylines, points, anchor for the label) to draw for a reading."""
    if m is None:
        return [], [], None
    if m["kind"] == "distance":
        return [np.array([m["A"], m["B"]])], [m["A"], m["B"]], 0.5 * (m["A"] + m["B"])
    if m["kind"] == "point":
        return [], [m["point"]], m["point"]
    if m["kind"] == "pair":
        r = m["r"]
        P, Q, _ = r["min"]
        lines, pts = [np.array([P, Q])], [P, Q]
        at = 0.5 * (P + Q)
        if r["centres"] is not None:
            C1, C2 = r["centres"][:2]
            lines.append(np.array([C1, C2]))
            pts += [C1, C2]
            at = 0.5 * (C1 + C2)
        return lines, pts, at
    if m["kind"] == "radius":
        if m["source"] == "straight":
            return [], [m["point"]], m["point"]
        C, Q, r = m["centre"], m["point"], m["radius"]
        lines = [np.array([C, Q])]
        if m.get("full") or m["source"] == "circle" or r < 1e6:
            lines.append(_circle_pts(C, r, m["normal"], x=Q - C))
        return lines, [C, Q], Q
    lines, pts = [], [m["point"]]
    for j, (r, dvec, C) in enumerate(m["principal"]):
        if C is None:
            continue
        nrm = _unit(np.cross(dvec, m["normal"]))          # the normal section's plane
        lines.append(_circle_pts(C, r, nrm, x=m["point"] - C))
        if j == 0:
            lines.append(np.array([C, m["point"]]))
            pts.append(C)
    return lines, pts, m["point"]


def _draw_view():
    import gpu
    from gpu_extras.batch import batch_for_shader
    sh = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = getattr(__import__("bpy").context, "region", None)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    first = [(pl, COL_FIRST) for pl in OVERLAY.get("first") or []]
    for m, col in [(it["m"], COL_KEPT) for it in OVERLAY["items"]] + \
            ([(OVERLAY["live"]["m"], COL_LIVE)] if OVERLAY.get("live") else []) + [(None, None)]:
        lines = geometry(m)[0] if m is not None else []
        if m is None:
            lines, col = [pl for pl, _ in first], COL_FIRST
        for pl in lines:
            co = [tuple(map(float, p)) for p in pl]
            segs = [x for a, b in zip(co[:-1], co[1:]) for x in (a, b)]
            if not segs:
                continue
            b = batch_for_shader(sh, 'LINES', {"pos": segs})
            sh.bind()
            sh.uniform_float("color", col)
            sh.uniform_float("lineWidth", 2.0)
            if region is not None:
                sh.uniform_float("viewportSize", (region.width, region.height))
            b.draw(sh)
    gpu.state.blend_set('NONE')


def _draw_px():
    import blf
    import bpy
    import gpu
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from gpu_extras.batch import batch_for_shader
    context = bpy.context
    region, rv3d = context.region, context.region_data
    if region is None or rv3d is None:
        return
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    font = 0
    blf.size(font, 13)
    entries = [(it, COL_KEPT) for it in OVERLAY["items"]] + ([(OVERLAY["live"], COL_LIVE)] if OVERLAY.get("live") else [])
    for it, col in entries:
        lines, pts, at = geometry(it["m"])
        for p in pts:                                     # the points: small squares
            q = location_3d_to_region_2d(region, rv3d, tuple(map(float, p)))
            if q is None:
                continue
            s = 3.5
            b = batch_for_shader(sh, 'TRIS', {"pos": [(q.x - s, q.y - s), (q.x + s, q.y - s), (q.x + s, q.y + s),
                                                       (q.x - s, q.y - s), (q.x + s, q.y + s), (q.x - s, q.y + s)]})
            sh.bind()
            sh.uniform_float("color", col)
            b.draw(sh)
        if at is None or not it.get("text"):
            continue
        q = location_3d_to_region_2d(region, rv3d, tuple(map(float, at)))
        if q is None:
            continue
        w, h = blf.dimensions(font, it["text"])
        x, y = q.x + 12, q.y + 10
        b = batch_for_shader(sh, 'TRIS', {"pos": [(x - 5, y - 5), (x + w + 5, y - 5), (x + w + 5, y + h + 5),
                                                   (x - 5, y - 5), (x + w + 5, y + h + 5), (x - 5, y + h + 5)]})
        sh.bind()
        sh.uniform_float("color", (0.08, 0.08, 0.08, 0.85))
        b.draw(sh)
        blf.position(font, x, y, 0)
        blf.color(font, *col)
        blf.draw(font, it["text"])
    gpu.state.blend_set('NONE')


def _add_handlers():
    import bpy
    if not _handles:
        _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), 'WINDOW', 'POST_VIEW'))
        _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_px, (), 'WINDOW', 'POST_PIXEL'))


def _remove_handlers():
    import bpy
    for h, kind in zip(_handles, ('WINDOW', 'WINDOW')):
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, kind)
        except Exception:
            pass
    _handles.clear()
    OVERLAY["items"], OVERLAY["live"], OVERLAY["first"] = [], None, []


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty  # noqa: E402
from bpy_extras import view3d_utils  # noqa: E402
from mathutils import Vector  # noqa: E402

from . import hints  # noqa: E402
from .drawtools import REFRESH_KEYS, SNAP_PX, SketchMixin  # noqa: E402

MOUSE = [("MOUSE_LMB", "Pick / keep"), ("MOUSE_RMB", "Done"), ("EVENT_ESC", "Done")]


class NPROTO_OT_measure(SketchMixin, bpy.types.Operator):
    """Measure what is under the mouse (a point, a curve or edge, a face:
    coordinates, radius, length), or between two things you click: distance,
    angle, centre to centre. D: radius / diameter; click keeps a reading on
    screen and in the Info log; Esc ends. Nothing in the scene changes"""
    bl_idname = "nproto.measure"
    bl_label = "Measure"
    bl_options = {'REGISTER'}

    diameter: BoolProperty(name="Diameter", default=False,
                           description="Read diameters instead of radii (D switches while measuring)")

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Measure: run it in a 3D View")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self.sketch_start(context)
        self._A = None                       # the first pick (a measurepair element)
        self._curves = {}
        self._els = {}
        self._pairs = {}
        self._count = 0
        self._last = ""
        self._P = None
        _add_handlers()
        OVERLAY["items"], OVERLAY["live"], OVERLAY["first"] = [], None, []
        context.window_manager.modal_handler_add(self)
        self._update(context, event)
        return {'RUNNING_MODAL'}

    # -- what is under the mouse ------------------------------------------------
    def _curve(self, key):
        """(linesolve Curve, its length) of a curve spline or surface edge, world space."""
        if key not in self._curves:
            c, L = None, None
            obj = bpy.data.objects.get(key[0])
            if obj is not None:
                try:
                    kc = features.input_curve(obj, key[1])
                    c = linesolve.kernel_curve(kc, store._k()[0])
                    from .measureinfo import curve_length
                    L = curve_length(kc)
                except Exception as exc:
                    print("Measure: cannot evaluate", key, exc)
            self._curves[key] = (c, L)
        return self._curves[key]

    def _el(self, key):
        """The measurepair element for a key ("L", object, ref) / ("F", object, face)."""
        from . import measurepair
        if key not in self._els:
            el = None
            obj = bpy.data.objects.get(key[1])
            if obj is not None:
                try:
                    if key[0] == "L":
                        el = measurepair.curve_el(features.input_curve(obj, key[2]), key=key,
                                                  what="edge" if store.is_nurbs(obj) else "curve")
                    else:
                        el = measurepair.face_el(obj, key[2])
                except Exception as exc:
                    print("Measure: cannot evaluate", key, exc)
            self._els[key] = el
        return self._els[key]

    def _target(self, context, event):
        """(what is under the mouse, its own reading): ("point", P) / ("L", key, point on it) /
        ("F", key, hit point), or None."""
        from .drawtools import POINT_KINDS
        prev = self._A["P"] if self._A is not None and self._A["type"] == "point" else None
        P, snap = self.pick_point(context, event, prev=prev)
        self._P = P
        kind = snap[0] if snap is not None else None
        src = snap[2][0] if snap is not None and snap[2] is not None else None
        self._snap_line = tuple(snap[2][:2]) if src is not None and len(snap[2]) > 1 and isinstance(snap[2][1], str) \
            else None                        # the curve / edge a snapped point lies on (K combs it)
        self._snap_obj = bpy.data.objects.get(src if isinstance(src, str) else getattr(src, "name", "")) \
            if src is not None else None
        if P is not None and (event.alt or kind in POINT_KINDS or getattr(self, "_lock", None)):
            return ("point", np.asarray(P, float))
        view, mouse = self._view(context), self._mouse(context, event)
        hit = self._ray(context, event)
        for k_line, _ in self._snaps.near_lines(view, mouse, SNAP_PX, limit=4):
            W, src = self._snaps.line_info(k_line)
            c, L = self._curve(tuple(src[:2])) if src is not None else (None, None)
            if c is None:
                continue
            a, _, ok = view.project(c.W)
            k = int(np.argmin(np.where(ok, np.linalg.norm(a - np.asarray(mouse, float), axis=1), np.inf)))
            if hit is not None and self._hidden(context, event, c.W[k], hit[1]):
                continue                     # behind the surface under the mouse (a rim's back): not this one
            return ("L", ("L", src[0], src[1]), c.W[k])
        if hit is not None:
            obj, Hp = hit
            i = self._face_at(obj, Hp)
            if i is not None:
                return ("F", ("F", obj.name, i), Hp)
        if P is not None:
            return ("point", np.asarray(P, float))
        return None

    def _face_at(self, obj, Hp):
        from . import geom
        b = store.get_brep(obj)
        q = np.linalg.inv(np.array(geom.world_matrix(obj), float)) @ np.r_[Hp, 1.0]
        try:
            f = store._k()[0].find_face(b, tuple(float(x) for x in q[:3]))
        except Exception:
            return None
        return next((k for k, g in enumerate(b.faces()) if g == f), None)

    def _single(self, tg):
        """The reading of one thing on its own."""
        if tg[0] == "point":
            return {"kind": "point", "point": tg[1]}
        if tg[0] == "L":
            c, L = self._curve(tg[1][1:])
            m = curve_radius(c, tg[2]) if c is not None else None
            if m is not None and L is not None:
                m["length"] = L
            return m
        obj = bpy.data.objects.get(tg[1][1])
        try:
            return surface_radius(obj, tg[2]) if obj is not None else None
        except Exception as exc:
            print("Measure:", exc)
            return None

    def _ref(self, tg):
        """What a pin keeps of a target (measurepin): the point (on the object it
        snapped to), the edge, curve or face and where on it."""
        from . import measurepin
        try:
            if tg[0] == "point":
                return measurepin.point_ref(tg[1], getattr(self, "_snap_obj", None))
            return measurepin.element_ref(tg[1], at=tg[2])
        except Exception as exc:
            print("Measure: cannot pin", tg[:2], exc)
            return None

    def _element(self, tg):
        from . import measurepair
        if tg[0] == "point":
            return measurepair.point_el(tg[1])
        return self._el(tg[1])

    def _pair(self, el):
        """The reading between the first pick and el."""
        from . import measurepair
        k = (self._A["key"], el["key"])
        if k not in self._pairs or self._A["type"] == "point" or el["type"] == "point":
            try:
                self._pairs[k] = {"kind": "pair", "r": measurepair.relate(self._A, el)}
            except Exception as exc:
                print("Measure:", exc)
                self._pairs[k] = None
        return self._pairs[k]

    def _reading(self, context, event):
        """(target, the live reading) under the mouse."""
        tg = self._target(context, event)
        if tg is None:
            return None, None
        if self._A is None:
            return tg, self._single(tg)
        el = self._element(tg)
        if el is None:
            return tg, None
        if el["key"] == self._A["key"]:
            return tg, self._single(tg)      # the first pick again: its own reading (click keeps it)
        return tg, self._pair(el)

    def _hidden(self, context, event, Q, H):
        """Whether the curve point Q lies behind the surface hit H, seen from the view."""
        region, rv3d = self._win(context)
        o = np.array(view3d_utils.region_2d_to_origin_3d(region, rv3d, Vector(self._mouse(context, event))), float)
        dq, dh = float(np.linalg.norm(Q - o)), float(np.linalg.norm(H - o))
        return dq > dh + 1e-3 * max(dh, 1e-9)

    def _ray(self, context, event):
        region, rv3d = self._win(context)
        m = Vector(self._mouse(context, event))
        o = view3d_utils.region_2d_to_origin_3d(region, rv3d, m)
        d = view3d_utils.region_2d_to_vector_3d(region, rv3d, m)
        try:
            hit, loc, _, _, obj, _ = context.scene.ray_cast(context.evaluated_depsgraph_get(), o, d)
        except Exception:
            return None
        if not hit or obj is None:
            return None
        obj = obj.original
        return (obj, np.array(loc, float)) if store.is_nurbs(obj) else None

    def _update(self, context, event):
        tg, m = self._reading(context, event)
        self._tg, self._live = tg, m
        OVERLAY["live"] = {"m": m, "text": label(context, m, self.diameter)} if m is not None else None
        from . import draw
        draw.sketch_overlay["snap"] = Vector(self._P) if tg is not None and tg[0] == "point" else None
        self._status(context)
        context.area.tag_redraw()

    def _status(self, context):
        live = OVERLAY.get("live")
        if self._A is None:
            state = ["move over a point, an edge, a curve or a face; click it to measure from it"]
        else:
            state = [f"from the {self._A['what']}: move over or click a second one (the {self._A['what']} again: "
                     "keep its own reading)" + self.lock_text()]
        if live and live.get("text"):
            state.append(live["text"])
        rows = [("Keep", [hints.key("LMB", "pick / keep"), hints.key("P", "pin (stays, follows the model)"),
                          hints.key("K", "pin a comb on the curve / edge"),
                          hints.key("Backspace", "drop the last"), hints.key("Ctrl C", "copy"), hints.key("Esc", "done")]),
                ("Show", [hints.toggle("D", "diameter", self.diameter)])]
        if self._A is not None and self._A["type"] == "point":
            rows.append(("Axis", [hints.key("X Y Z", "along an axis",
                                            on=getattr(self, "_lock", None) in ("X", "Y", "Z", "-X", "-Y", "-Z"))]))
        rows.append(("Snap", self.snap_hints()))
        hints.show(context, "Measure", state, rows, MOUSE)

    def _keep(self, context, m, refs=None):
        txt = label(context, m, self.diameter)
        OVERLAY["items"].append({"m": m, "text": txt, "dia": self.diameter, "refs": refs})
        self._count += 1
        self._last = txt
        what = {"distance": "Distance", "point": "Point", "pair": "Between",
                "radius": "Diameter" if self.diameter else "Radius",
                "surface": "Surface " + ("diameters" if self.diameter else "radii")}[m["kind"]]
        if m["kind"] == "radius" and m["source"] == "straight":
            what = "Length"
        self.report({'INFO'}, f"{what}: {txt}")
        print(f"Measure · {what}: {txt}")

    def _set_first(self, el, ref=None):
        from . import measurepair
        self._A = el
        self._A_ref = ref
        OVERLAY["first"] = measurepair.outline(el) if el is not None else []

    def _end(self, context):
        self.sketch_end(context)
        _remove_handlers()
        from . import draw
        draw.sketch_overlay["snap"] = None
        context.area.tag_redraw()

    def _click(self, context, event):
        tg, m = self._reading(context, event)
        if tg is None:
            return
        el = self._element(tg)
        if el is None:
            return
        if self._A is None:
            self._set_first(el, self._ref(tg))
            if el["type"] == "point":
                self.fix_plane(context, el["P"])
            return
        if el["key"] == self._A["key"]:
            if m is not None and not (m["kind"] == "radius" and m["source"] == "straight" and m.get("length") is None):
                self._keep(context, m, (self._ref(tg), None))
        elif m is not None:
            self._keep(context, m, (self._A_ref, self._ref(tg)))
        self._set_first(None)
        if getattr(self, "_lock", None) in ("X", "Y", "Z", "-X", "-Y", "-Z"):
            self._lock = None

    def _pin(self, context, event):
        """P: pin the reading under the mouse (else the last one kept)."""
        from . import measurepin
        tg, m = self._reading(context, event)
        refs = None
        if tg is not None and m is not None:
            el = self._element(tg)
            if self._A is None or (el is not None and el["key"] == self._A["key"]):
                refs = (self._ref(tg), None)
            else:
                refs = (self._A_ref, self._ref(tg))
        elif OVERLAY["items"]:
            refs = OVERLAY["items"][-1].get("refs")
        if not refs or refs[0] is None or (refs[1] is None and m is not None and m.get("kind") == "pair"):
            self.report({'WARNING'}, "Nothing to pin here")
            return
        p = measurepin.add(context.scene, refs[0], refs[1], self.diameter)
        txt, _ = measurepin.text(context, p)
        bpy.ops.ed.undo_push(message="Pin Measurement")
        self.report({'INFO'}, f"Pinned {p.label}: {txt}")
        if self._A is not None and refs[1] is not None:
            self._set_first(None)

    def _pin_comb(self, context, event):
        """K: pin a curvature comb on the curve or edge under the mouse."""
        from . import measurepin
        tg = self._target(context, event)
        if tg is not None and tg[0] == "point" and getattr(self, "_snap_line", None):
            tg = ("L", ("L",) + self._snap_line, tg[1])          # a point snapped on an edge: that edge
        if tg is None or tg[0] != "L":
            self.report({'WARNING'}, "K pins a comb: move over a curve or an edge")
            return
        ref = measurepin.element_ref(tg[1])
        if ref is None:
            return
        p = measurepin.add(context.scene, ref, comb=True)
        txt, _ = measurepin.text(context, p)
        bpy.ops.ed.undo_push(message="Pin Curvature Comb")
        self.report({'INFO'}, f"Pinned {p.label}: {txt}")

    def modal(self, context, event):
        if self.passthrough(context, event):
            return {'PASS_THROUGH'}
        if self.hints_key(context, event):
            return {'RUNNING_MODAL'}
        if event.type in REFRESH_KEYS or self.snap_key(context, event):
            self._update(context, event)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._click(context, event)
            self._update(context, event)
        elif event.value == 'PRESS' and event.type == 'P' and not event.ctrl:
            self._pin(context, event)
            self._update(context, event)
        elif event.value == 'PRESS' and event.type == 'K' and not event.ctrl:
            self._pin_comb(context, event)
            self._update(context, event)
        elif event.value == 'PRESS' and event.type == 'D' and not event.ctrl:
            self.diameter = not self.diameter
            for it in OVERLAY["items"]:
                it["text"] = label(context, it["m"], self.diameter)
            self._update(context, event)
        elif event.value == 'PRESS' and event.type == 'C' and (event.ctrl or event.oskey):
            txt = (OVERLAY.get("live") or {}).get("text") or self._last
            if txt:
                context.window_manager.clipboard = txt
                self.report({'INFO'}, f"Copied: {txt}")
        elif event.value == 'PRESS' and event.type in {'X', 'Y', 'Z'} and self._A is not None \
                and self._A["type"] == "point":
            self.axis_key(event)
            self._update(context, event)
        elif event.value == 'PRESS' and event.type == 'BACK_SPACE':
            if self._A is not None:
                self._set_first(None)
            elif OVERLAY["items"]:
                OVERLAY["items"].pop()
            self._update(context, event)
        elif event.value == 'PRESS' and event.type in {'ESC', 'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            self._end(context)
            return {'FINISHED'} if self._count else {'CANCELLED'}
        return {'RUNNING_MODAL'}


classes = (NPROTO_OT_measure,)
