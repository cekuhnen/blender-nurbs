"""Rail Revolve: a profile turned around an axis while its end follows a rail.

Axis: through the rail's centre square to its plane (default), a world or
local axis, or a picked line.

At every point of the rail the profile is turned to the rail's angle about
the axis and stretched away from the axis so that its end lands on the rail
(Rhino RailRevolve). A circle round the axis through the profile's end gives
a plain revolve; a rounded rectangle gives a lamp shade with a rectangular
foot, the profile's shape kept in every section.

It is exact, with no fitting: in the axis frame the profile point is
A + rho(t) e_r + z(t) e_z, rho and z affine in the point, and the section at
rail point R(s) is A + rho(t) / rho_end * R_perp(s) + z(t) e_z, R_perp the
rail point's offset square to the axis. Both factors are affine in the CVs,
so the surface's CVs are A + (rho_i / rho_end) R_perp_j + z_i e_z with
weights w_i v_j (the profile's degree and knots one way, the rail's the
other). Height: Keep (the profile's heights; the rail's own height is not
used), Shift (the profile moves along the axis with the rail's height) or
Scale (it stretches along the axis from its other end so that its end
meets the rail's height) -- still exact (sums and products of affine terms).

The profile should lie in a half-plane through the axis; points off it are
taken by their distance from the axis (and the report says how far off).
The end that follows the rail is the one nearer to it.
"""

import math

import numpy as np
from mathutils import Vector

from . import features, geom

LAST_INFO = {}


def _cvs(kc):
    _, ext = features._k()
    d = ext.curve_cvs(kc)
    P = np.array(d["points"], float)
    w = np.array(d["weights"], float) if d.get("weights") else np.ones(len(P))
    return int(d["degree"]), P, w, [float(k) for k in d["knots"]]


def rail_axis(kc):
    """(centre, unit normal) of a rail: its plane's normal (world Z for a
    straight or non-planar rail; turned to point up), through its centre -- the
    area centroid of a closed rail, the fitted circle's centre of an open one."""
    P = geom.sample_curve(kc, 400)
    size = float(np.ptp(P, axis=0).max()) or 1.0
    c = P.mean(axis=0)
    _, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    planar = S[1] > 1e-6 * size * np.sqrt(len(P)) and float(np.abs((P - c) @ n).max()) <= 1e-3 * size
    if not planar:
        n = np.array([0.0, 0.0, 1.0])
    if n[2] < 0 or (abs(n[2]) < 1e-9 and (n[0] < 0 or (abs(n[0]) < 1e-9 and n[1] < 0))):
        n = -n
    e1 = Vt[0] if planar else np.array([1.0, 0.0, 0.0])
    e1 = e1 - (e1 @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    q = np.c_[(P - c) @ e1, (P - c) @ e2]
    closed = np.linalg.norm(P[0] - P[-1]) <= 1e-6 * size
    if closed:
        x, y = q[:, 0], q[:, 1]
        cr = x[:-1] * y[1:] - x[1:] * y[:-1]
        area = 0.5 * cr.sum()
        if abs(area) > 1e-12 * size * size:
            cx = ((x[:-1] + x[1:]) * cr).sum() / (6 * area)
            cy = ((y[:-1] + y[1:]) * cr).sum() / (6 * area)
            return c + cx * e1 + cy * e2, n
    # open: the least-squares circle through the points (Kasa)
    M = np.c_[2 * q, np.ones(len(q))]
    sol = np.linalg.lstsq(M, (q ** 2).sum(axis=1), rcond=None)[0]
    return c + sol[0] * e1 + sol[1] * e2, n


def axis_of(inputs, p, rail=None):
    """(A, z) of the axis (world, unit z). Rail Centre needs the rail (obj, ref)."""
    ax = p.get("rr_axis", "RAIL")
    if ax == "RAIL":
        if rail is None:
            raise ValueError("Rail Centre: no rail")
        return rail_axis(features.input_curve(*rail))
    if ax == "PICKED":
        if len(inputs) < 3:
            raise ValueError("Axis Picked Line: pick a line for the axis as the third input")
        o, r = inputs[2]
        P = geom.sample_curve(features.input_curve(o, r), 9)
        d = P[-1] - P[0]
        if np.linalg.norm(d) < 1e-12:
            raise ValueError("the axis line has no length")
        return P[0], d / np.linalg.norm(d)
    obj = inputs[0][0]
    A, z = features.revolve_axis(obj, {"axis": ax, "pivot": p.get("pivot", "WORLD")})
    return np.array(A, float), np.array(z, float)


def _sweep(P, A, z):
    """Angle the points sweep round the axis (radians)."""
    rel = P - A
    rel = rel - np.outer(rel @ z, z)
    keep = np.linalg.norm(rel, axis=1) > 1e-9 * (np.ptp(P, axis=0).max() or 1.0)
    rel = rel[keep]
    if len(rel) < 2:
        return 0.0
    e1 = rel[0] / np.linalg.norm(rel[0])
    e2 = np.cross(z, e1)
    ang = np.unwrap(np.arctan2(rel @ e2, rel @ e1))
    return float(np.ptp(ang))


def order_inputs(inputs, p):
    """(profile, rail): the profile sweeps (almost) no angle round the axis, the
    rail goes round it (with Rail Centre: each way tried, the clearer taken)."""
    S = [geom.sample_curve(features.input_curve(o, r), 33) for o, r in inputs[:2]]
    best = None
    for k in (0, 1):
        prof, rail = inputs[1 - k], inputs[k]
        A, z = axis_of(inputs, p, rail)
        score = _sweep(S[k], A, z) - _sweep(S[1 - k], A, z)
        if best is None or score > best[0]:
            best = (score, prof, rail)
        if p.get("rr_axis", "RAIL") != "RAIL":
            # a fixed axis: one evaluation decides
            other = _sweep(S[1 - k], A, z)
            mine = _sweep(S[k], A, z)
            return (inputs[1 - k], inputs[k]) if mine >= other else (inputs[k], inputs[1 - k])
    return best[1], best[2]


def build(inputs, p):
    _, ext = features._k()
    if len(inputs) < 2:
        raise ValueError("pick the profile and the rail")
    prof, rail = order_inputs(inputs, p)
    A, z = axis_of(inputs, p, rail)
    kp = features.input_curve(*prof)
    kr = features.input_curve(*rail)
    pdeg, Q, wq, Uq = _cvs(kp)
    rdeg, R, wr, Ur = _cvs(kr)
    size = float(np.linalg.norm(np.ptp(np.vstack([Q, R]), axis=0))) or 1.0
    # which end follows the rail: the one nearer to it
    Rs = geom.sample_curve(kr, 200)
    ends = [Q[0], Q[-1]]
    dist = [float(np.linalg.norm(Rs - e, axis=1).min()) for e in ends]
    k_end = int(np.argmin(dist))
    E = ends[k_end]
    relE = E - A
    rE = relE - (relE @ z) * z
    rho_e = float(np.linalg.norm(rE))
    if rho_e < 1e-9 * size:
        raise ValueError("the profile's end that should follow the rail lies on the axis")
    er = rE / rho_e
    rel = Q - A
    zq = rel @ z
    rho = rel @ er
    off = float(np.linalg.norm(rel - np.outer(zq, z) - np.outer(rho, er), axis=1).max())
    z_e = float(relE @ z)
    z_s = float(zq[-1 if k_end == 0 else 0])        # the profile's other end: Scale keeps it
    relR = R - A
    hr = relR @ z
    Rp = relR - np.outer(hr, z)
    mode = p.get("rr_height", "KEEP")
    if mode == "SCALE" and abs(z_e - z_s) < 1e-9 * size:
        raise ValueError("Scale Height: the profile's two ends are at one height (nothing to scale)")
    S = np.empty((len(Q), len(R), 3))
    for i in range(len(Q)):
        if mode == "SHIFT":
            hz = zq[i] + hr - z_e
        elif mode == "SCALE":
            hz = z_s + (zq[i] - z_s) * (hr - z_s) / (z_e - z_s)
        else:
            hz = np.full(len(R), zq[i])
        S[i] = A + (rho[i] / rho_e) * Rp + np.outer(hz, z)
    W = np.outer(wq, wr)
    rational = np.abs(W - 1.0).max() > 1e-12
    # facing away from the axis (as a revolve's outside) at the middle of the rail
    if _faces_axis(S, W, pdeg, Uq, rdeg, Ur, A, z):
        S, W = S[:, ::-1], W[:, ::-1]
        Ur = [Ur[0] + Ur[-1] - k for k in reversed(Ur)]
    pts = [tuple(map(float, S[i, j])) for i in range(len(Q)) for j in range(len(R))]
    wts = [float(W[i, j]) for i in range(len(Q)) for j in range(len(R))] if rational else None
    surf = ext.surface_from_cvs(pdeg, rdeg, pts, Uq, Ur, wts)
    gap = float(np.linalg.norm(Rs - _end_curve(S, W, k_end, rdeg, Ur, 200), axis=1).max()) \
        if mode != "KEEP" or np.ptp(hr) < 1e-9 * size else None
    LAST_INFO.clear()
    LAST_INFO.update({"cvs": (len(Q), len(R)), "degree": (pdeg, rdeg), "off_plane": off, "rail_gap": gap,
                      "profile": f"{prof[0].name} {prof[1]}", "rail": f"{rail[0].name} {rail[1]}"})
    return surf


def _faces_axis(S, W, p, U, q, V, A, z):
    from .kernel.classa import basis
    t = 0.5 * (U[p] + U[-p - 1])
    s = 0.5 * (V[q] + V[-q - 1])
    Bu, Bu1 = basis(U, p, [t])[0], basis(U, p, [t], 1)[0]
    Bv, Bv1 = basis(V, q, [s])[0], basis(V, q, [s], 1)[0]
    H = np.concatenate([S * W[..., None], W[..., None]], axis=2)
    def ev(a, b):
        return np.einsum("i,j,ijc->c", a, b, H)
    h, hu, hv = ev(Bu, Bv), ev(Bu1, Bv), ev(Bu, Bv1)
    P = h[:3] / h[3]
    du = (hu[:3] - P * hu[3]) / h[3]
    dv = (hv[:3] - P * hv[3]) / h[3]
    n = np.cross(du, dv)
    out = (P - A) - ((P - A) @ z) * z
    return float(n @ out) < 0


def _end_curve(S, W, k_end, q, V, n):
    from .kernel.classa import basis
    row = S[0 if k_end == 0 else -1]
    w = W[0 if k_end == 0 else -1]
    ts = np.linspace(V[q], V[-q - 1], n)
    B = basis(V, q, ts)
    H = B @ np.c_[row * w[:, None], w]
    return H[:, :3] / H[:, 3:]


@features.builder("rail_revolve")
def build_rail_revolve(inputs, p):
    return build(inputs, p)


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import EnumProperty, FloatVectorProperty, StringProperty  # noqa: E402

from .ops import AXIS_ITEMS, PickMixin, _decode_inputs, _encode_inputs, _kernel_ok, _make_result, _select_only  # noqa: E402

RR_AXIS = [("RAIL", "Rail Centre", "Through the rail's centre, square to its plane (world Z for a non-planar rail)")] + \
    AXIS_ITEMS + [("PICKED", "Picked Line", "A line picked as the third input: from its start to its end")]
HEIGHT_ITEMS = [("KEEP", "Keep", "The profile keeps its heights; only its distance from the axis follows the rail"),
                ("SHIFT", "Shift", "The profile also moves along the axis with the rail's height"),
                ("SCALE", "Scale", "The profile also stretches along the axis (from its other end) so its end "
                                   "meets the rail's height")]


class NPROTO_OT_rail_revolve(PickMixin, bpy.types.Operator):
    """Revolve a profile around an axis while its end follows a rail: the
    profile is stretched away from the axis to meet the rail at every angle
    (exact: no fitting). Select or click the profile and the rail (any order;
    with Axis: Picked Line, then the axis line) and press Enter"""
    bl_idname = "nproto.rail_revolve"
    bl_label = "Rail Revolve"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 2
    pick_max = 3
    finish_on_min = False
    extra_help = "   (profile and rail in any order; the axis line third with Axis: Picked Line)"
    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    axis: EnumProperty(name="Axis", items=RR_AXIS, default="RAIL")
    pivot: EnumProperty(name="Pivot", items=[("WORLD", "World Origin", ""), ("OBJECT", "Object Origin",
                                                                              "The profile object's origin")],
                        default="WORLD")
    rr_height: EnumProperty(name="Height", items=HEIGHT_ITEMS, default="KEEP")

    def invoke(self, context, event):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not self.inputs and not event.alt:
            crv = [o for o in context.selected_objects if o.type == 'CURVE' and len(o.data.splines)]
            if len(crv) in (2, 3):
                self.inputs = _encode_inputs([(o, "S0") for o in crv])
                return self.execute(context)
        return self.start_picking(context, event)

    def _params(self):
        return {"rr_axis": self.axis, "pivot": self.pivot, "rr_height": self.rr_height}

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop(self, "axis")
        if self.axis not in ("PICKED", "RAIL"):
            lay.prop(self, "pivot")
        lay.row().prop(self, "rr_height", expand=True)
        inf = LAST_INFO
        if inf:
            lay.label(text=f"profile {inf['profile']} · rail {inf['rail']}", icon='CURVE_DATA')
            lay.label(text=f"{inf['cvs'][0]} x {inf['cvs'][1]} CVs, exact", icon='MESH_GRID')
            if inf.get("off_plane", 0) > 1e-6:
                lay.label(text=f"profile off a plane through the axis by {inf['off_plane']:.2g}", icon='ERROR')

    def execute(self, context):
        if not _kernel_ok(self):
            return {'CANCELLED'}
        try:
            inputs = _decode_inputs(self.inputs)
            if len(inputs) < 2:
                raise ValueError("pick the profile and the rail")
            params = self._params()
            obj = _make_result(context, "RailRevolve", "rail_revolve", inputs, params)
        except Exception as exc:
            self.report({'ERROR'}, f"Rail Revolve: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        inf = LAST_INFO
        msg = f"Rail Revolve: {inf['cvs'][0]} x {inf['cvs'][1]} CVs (exact)"
        if inf.get("off_plane", 0) > 1e-6:
            self.report({'WARNING'}, msg + f"; the profile is off a plane through the axis by {inf['off_plane']:.2g}")
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


classes = (NPROTO_OT_rail_revolve,)
