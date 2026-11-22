"""Emboss / Deboss: raised or sunk lettering, logos and pads on any face of a
solid, not only flat or cylindrical ones.

The profiles (closed curves: outlines and the holes inside them, any number
of letters at once) are extruded into a prism -- Emboss: from Overlap under
the surface to Height over it; Deboss: from Height under it to Overlap over
it -- the prism is laid onto the body and joined to it (Emboss) or cut from
it (Deboss). The Overlap keeps the prism's far end off the body's own
surface: a Boolean of faces lying on one another is the fragile case.

How the prism is laid on:
* Wrap: the profiles lie on a sheet. A sheet made by Unroll Surface from the
  body folds back face by face (exact on developable faces, the unroll's
  spread-out stretch on others): lettering keeps its size and shape. Any
  other flat sheet maps by (u, v) share onto one face of the body (Flow
  Along Surface's General: the face picked, else the largest).
* Project: straight along the profiles' plane normal, as a light throws a
  shadow: the walls stay parallel to that direction (no undercut for a
  mould pulled that way) and the top is the surface moved along it by
  Height. The nearest surface on each line is used (Far Side: the farthest).
* Place: moved and turned as a whole, tangent to the surface at the
  profiles' centre (small logos, pads: no bending at all; the Overlap must
  cover the surface's sag under them, about width^2 / (8 radius)).

The material side is the body's outside where the profiles land: Height is
measured outward, whichever way the sheet's or the surface's own normal
points."""

import json

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

from . import features, flow, geom, store, surfeval, surfmap

OP = "emboss"
MODE_ITEMS = [("WRAP", "Wrap", "Bend the profiles from the sheet they lie on onto the body (an unrolled sheet "
                               "folds back exactly)"),
              ("PROJECT", "Project", "Throw the profiles straight along their plane's normal onto the body "
                                     "(walls parallel to it)"),
              ("PLACE", "Place", "Move and turn the profiles as a whole onto the surface (no bending)")]
OP_ITEMS = [("EMBOSS", "Emboss", "Raised: joined to the body"),
            ("DEBOSS", "Deboss", "Sunk: cut into the body")]
LAST_INFO = {}


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


# --------------------------------------------------------------------------
# The inputs
# --------------------------------------------------------------------------

def split_inputs(inputs):
    """(body, face index or None, base or None, profile inputs) of an
    Emboss's inputs: the body first (ref SURF, or F<i>: the face a General
    wrap goes onto), a sheet (ref SURF) if it wraps, then the profiles."""
    body, ref = inputs[0]
    face = int(ref[1:]) if features.is_face_ref(ref) else None
    base = None
    profiles = []
    for o, r in inputs[1:]:
        if r == "SURF" and base is None:
            base = o
        else:
            profiles.append((o, r))
    return body, face, base, profiles


def profile_plane(curves):
    """(centre, unit normal, size) of the plane the profile curves lie in."""
    P = np.vstack([geom.sample_curve(c, 65) for c in curves])
    c = P.mean(axis=0)
    _, sv, vt = np.linalg.svd(P - c)
    size = float(np.linalg.norm(np.ptp(P, axis=0))) or 1.0
    if sv[1] <= 1e-9 * size or float(np.abs((P - c) @ vt[2]).max()) > 1e-4 * size:
        raise RuntimeError("the profiles are not flat in one plane")
    return c, vt[2], size


def _hit_face(B, X):
    """(face, (u, v), outward unit normal) of the body face nearest X."""
    faces = list(B.faces())
    d = [f.closest_point(tuple(map(float, X)))[1] for f in faces]
    f = faces[int(np.argmin(d))]
    S = surfeval.Surf(f.surface())
    uv = S.feet(np.asarray(X, float)[None])[0]
    return f, uv, np.asarray(f.outward_normal_at_uv(float(uv[0]), float(uv[1])), float)


# --------------------------------------------------------------------------
# Project: along a fixed direction onto the body
# --------------------------------------------------------------------------

def _in_loops(loops, uv):
    """Even-odd: which (u, v) lie inside a face's trim loops ((k, 2) polylines)."""
    inside = np.zeros(len(uv), bool)
    x, y = uv[:, 0][:, None], uv[:, 1][:, None]
    for L in loops:
        a, b = L, np.roll(L, -1, axis=0)
        for s in range(0, len(a), 512):
            ax, ay, bx, by = a[s:s + 512, 0], a[s:s + 512, 1], b[s:s + 512, 0], b[s:s + 512, 1]
            cross = (ay > y) != (by > y)
            xi = ax + (y - ay) * (bx - ax) / np.where(by != ay, by - ay, 1.0)
            inside ^= (np.count_nonzero(cross & (x < xi), axis=1) % 2).astype(bool)
    return inside


class _ProjFace:
    """One body face for Project: its surface, trim loops (None: natural),
    a seed grid and that grid's footprint on the profiles' plane."""
    G = 17                                             # seed grid per side

    def __init__(self, f, e1, e2):
        self.face = f
        self.S = surfeval.Surf(f.surface())
        self.natural = surfmap.is_natural_face(f, self.S)
        self.loops = None if self.natural else [
            np.vstack([uv for uv, _ in surfmap.loop_uvs(self.S, lp)])
            for lp in f.boundary_loops() if lp]
        (u0, u1), (v0, v1) = self.S.u_domain, self.S.v_domain
        self.gu, self.gv = np.linspace(u0, u1, self.G), np.linspace(v0, v1, self.G)
        P = self.S.grid(self.gu, self.gv, 0)[0, 0]
        self.cell = float(np.linalg.norm(np.ptp(P, axis=(0, 1)))) / (self.G - 1)
        self.P = P.reshape(-1, 3)
        self.ab = np.stack([self.P @ e1, self.P @ e2], 1)
        pad = 2.0 * self.cell
        self.lo, self.hi = self.ab.min(axis=0) - pad, self.ab.max(axis=0) + pad

    def covers(self, ab):
        return np.all((ab >= self.lo) & (ab <= self.hi), axis=1)

    def hits(self, B, ab, n, far):
        """(t, in the domain, inside the trim, line distance) of the lines
        B + t n on this face: Newton from the seed point nearest each line
        (among the close ones the nearest along it; Far: the farthest)."""
        S = self.S
        d2 = ((self.ab[None] - ab[:, None]) ** 2).sum(axis=2)          # (m, G*G)
        tg = (self.P @ n)[None] - (B @ n)[:, None]
        lim = (np.sqrt(d2.min(axis=1)) + 2.0 * self.cell) ** 2
        close = d2 <= lim[:, None]
        # the side of the line wanted (nearest / farthest hit), then the seed nearest the line there
        # (not a pole of a sphere, whose zero derivative would hold Newton)
        side = np.where(close, -np.abs(tg) if far else np.abs(tg), np.inf)
        ext = side.min(axis=1, keepdims=True)
        k = np.argmin(np.where(close & (side <= ext + 2.0 * self.cell), d2, np.inf), axis=1)
        g = len(self.gv)
        uv = np.stack([self.gu[k // g], self.gv[k % g]], 1)
        t = tg[np.arange(len(B)), k]
        (u0, u1), (v0, v1) = S.u_domain, S.v_domain
        act = np.arange(len(B))
        for _ in range(25):
            D = S.at(uv[act, 0], uv[act, 1], 1)
            F = D[0, 0] - B[act] - t[act, None] * n
            J = np.stack([D[1, 0], D[0, 1], np.broadcast_to(-n, D[1, 0].shape)], 2)
            step = (np.linalg.pinv(J, rcond=1e-10) @ -F[..., None])[..., 0]    # a pole: least squares
            uv[act] += step[:, :2]
            t[act] += step[:, 2]
            uv[act, 0] = np.clip(uv[act, 0], u0, u1)
            uv[act, 1] = np.clip(uv[act, 1], v0, v1)
            act = act[np.abs(step).max(axis=1) > 1e-13 * (1.0 + np.abs(uv[act]).max(axis=1))]
            if not len(act):
                break
        D = S.at(uv[:, 0], uv[:, 1], 1)
        X = D[0, 0]
        res = np.linalg.norm(X - B - t[:, None] * n, axis=1)
        dom = res <= 1e-9 * self.cell * self.G
        # off the surface's domain: carry on in the tangent plane at its edge
        N = _unit(np.cross(D[1, 0], D[0, 1]))
        dn = N @ n
        tp = np.einsum("ij,ij->i", X - B, N) / np.where(np.abs(dn) > 1e-6, dn, 1.0)
        t = np.where(dom, t, tp)
        line = np.linalg.norm((X - B) - ((X - B) @ n)[:, None] * n, axis=1)
        # a face running nearly along the lines (a box's top seen from the front) carries
        # them on only when no other face can: its tangent plane meets them far off or never
        line = np.where(np.abs(dn) > 0.2, line, 1e6 + line)
        inside = np.ones(len(B), bool) if self.natural else np.zeros(len(B), bool)
        if not self.natural and dom.any():
            inside[dom] = _in_loops(self.loops, uv[dom])
        return t, dom, inside & dom, np.where(dom, 0.0, line)


class ProjectMap:
    """World points -> moved along n onto the body: Q + t n, t the line's
    hit on the body through Q's foot on the profiles' plane. Duck-types
    flow.FlowMap (map, size) for flow.flow_brep."""

    def __init__(self, body_brep, centre, normal, size, far=False):
        self.c, self.n, self.size, self.far = np.asarray(centre, float), _unit(np.asarray(normal, float)), size, far
        a = np.array([1.0, 0.0, 0.0]) if abs(self.n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.e1 = _unit(a - (a @ self.n) * self.n)
        self.e2 = np.cross(self.n, self.e1)
        self.faces = [_ProjFace(f, self.e1, self.e2) for f in body_brep.faces()]
        self.memo = {}                                       # foot key -> t (edges are mapped once per face)

    def shift(self, B):
        """t of each foot B (on the profiles' plane): inside a face's trim
        first, then on a surface past its trim, then past a surface's edge
        (the nearest to the line); among equals the nearest hit (Far: the
        farthest)."""
        m = len(B)
        ab = np.stack([B @ self.e1, B @ self.e2], 1)
        rank = np.full((m, len(self.faces)), np.inf)
        T = np.zeros((m, len(self.faces)))
        cover = [pf.covers(ab) for pf in self.faces]
        if not np.any(cover, axis=0).all():                  # off every face: all of them may extend
            miss = ~np.any(cover, axis=0)
            cover = [c | miss for c in cover]
        for k, pf in enumerate(self.faces):
            sel = np.nonzero(cover[k])[0]
            if not len(sel):
                continue
            t, dom, inside, line = pf.hits(B[sel], ab[sel], self.n, self.far)
            rank[sel, k] = np.where(inside, -1.0, np.where(dom, 0.0, 1.0 + line))
            T[sel, k] = t
        best = rank.min(axis=1, keepdims=True)
        tie = rank <= best + 1e-12
        sc = np.where(tie, -np.abs(T) if self.far else np.abs(T), np.inf)
        return T[np.arange(m), np.argmin(sc, axis=1)]

    def map(self, Q):
        Q = np.asarray(Q, float).reshape(-1, 3)
        if not len(Q):
            return Q.copy()
        h = (Q - self.c) @ self.n
        B = Q - h[:, None] * self.n
        # points over one foot (a wall's lines along n) share its t
        key = np.round(np.stack([B @ self.e1, B @ self.e2], 1) / (1e-12 * self.size))
        uk, first, back = np.unique(key, axis=0, return_index=True, return_inverse=True)
        keys = [tuple(k) for k in uk.astype(np.int64).tolist()]
        t = np.array([self.memo.get(k, np.nan) for k in keys], float)
        todo = np.nonzero(np.isnan(t))[0]
        for s in range(0, len(todo), 2000):
            sel = todo[s:s + 2000]
            t[sel] = self.shift(B[first[sel]])
        for i in todo.tolist():
            self.memo[keys[i]] = float(t[i])
        return Q + t[back.ravel()][:, None] * self.n


# --------------------------------------------------------------------------
# Wrap: Flow's map from a sheet
# --------------------------------------------------------------------------

def wrap_map(base, body, face, p):
    """(FlowMap, mode): the sheet folds back onto the body if Unroll made it
    from the body, else maps by (u, v) share onto one of its faces."""
    if face is None:
        fmap, mode, _ = flow.build_map(base, body, bool(p.get("fw_flip_u")), bool(p.get("fw_flip_v")),
                                       bool(p.get("fw_swap")), bool(p.get("fw_fold", True)))
        return fmap, mode
    bb, tb = flow.world_brep(base), flow.world_brep(body)
    faces = tb.faces()
    if not 0 <= face < len(faces):
        raise RuntimeError(f"{body.name} has no face {face}")
    Sb, St = surfeval.Surf(flow.largest_face(bb).surface()), surfeval.Surf(faces[face].surface())
    pair = flow.Pair(None, Sb, St, flow.general_uvmap(Sb, St, bool(p.get("fw_flip_u")), bool(p.get("fw_flip_v")),
                                                      bool(p.get("fw_swap"))))
    return flow.FlowMap([pair], max(flow.brep_size(bb), flow.brep_size(tb))), "general"


def base_frame(base, c):
    """(foot of c on the sheet, unit normal there, two unit tangents)."""
    bb = flow.world_brep(base)
    S = surfeval.Surf(flow.largest_face(bb).surface()) if bb.face_count() == 1 else None
    best = None
    for f in ([None] if S is not None else bb.faces()):
        Sf = S if f is None else surfeval.Surf(f.surface())
        uv = Sf.feet(np.asarray(c, float)[None])
        D = Sf.at(uv[:, 0], uv[:, 1], 1)
        d = float(np.linalg.norm(D[0, 0][0] - c))
        if best is None or d < best[0]:
            best = (d, D)
    D = best[1]
    e1 = _unit(D[1, 0][0])
    n = _unit(np.cross(D[1, 0][0], D[0, 1][0]))
    return D[0, 0][0], n, e1, np.cross(n, e1)


def rigid_matrix(m, c, n, e1, size):
    """4x4 world move taking the frame (c; e1, n x e1, n) to where the map m
    takes it (its turn and move at c, no bending)."""
    e2 = np.cross(n, e1)
    eps = 1e-4 * size
    Q = m(np.array([c, c + eps * e1, c + eps * e2]))
    f1 = _unit(Q[1] - Q[0])
    f2 = Q[2] - Q[0]
    f2 = _unit(f2 - (f2 @ f1) * f1)
    R = np.stack([f1, f2, np.cross(f1, f2)], 1) @ np.stack([e1, e2, n], 1).T
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = Q[0] - R @ c
    return A


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------

def _prism(profiles, d, lo, hi):
    """A closed solid of the profiles (outlines and their holes) from lo
    behind their plane to hi in front of it, along the unit direction d."""
    q = {"direction": "VECTOR", "dir_vec": tuple(map(float, d)), "distance": float(lo + hi), "start": -float(lo)}
    g = features._solid(profiles, q, features._extrude_side)
    return store.brep_from_text(g[1])


def crossing_outlines(curves, n, size):
    """The first pair (i, j) of profile curves whose outlines cross or touch
    (kerned letters: a W running into the A after it), else None. They can't
    be one region's outline and hole, nor two regions side by side."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(a - (a @ n) * n)
    e2 = np.cross(n, e1)
    polys = []
    for c in curves:
        P = geom.sample_curve(c, 193)
        polys.append(np.stack([P @ e1, P @ e2], 1))
    gap = 1e-5 * size

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            A, B = polys[i], polys[j]
            if np.any(A.max(axis=0) < B.min(axis=0) - gap) or np.any(B.max(axis=0) < A.min(axis=0) - gap):
                continue
            a0, a1 = A[:-1, None], A[1:, None]
            b0, b1 = B[None, :-1], B[None, 1:]
            d1, d2 = orient(a0, a1, b0), orient(a0, a1, b1)
            d3, d4 = orient(b0, b1, a0), orient(b0, b1, a1)
            if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
                return i, j
            if float(np.linalg.norm(A[:, None] - B[None], axis=2).min()) <= gap:
                return i, j
    return None


def emboss_tool(inputs, p):
    """The prism laid onto the body (world Brep), and how: (Brep, info)."""
    body, face, base, profiles = split_inputs(inputs)
    if not profiles:
        raise RuntimeError("no profiles: select closed curves to emboss")
    mode = p.get("em_mode", "PROJECT")
    if mode == "WRAP" and base is None:
        raise RuntimeError("Wrap needs the sheet the profiles lie on (Unroll the body, draw on the sheet)")
    H = float(p.get("em_height", 0.1))
    D = float(p.get("em_overlap", 0.025))
    if H <= 0.0:
        raise RuntimeError("Height must be more than 0")
    if D < 0.0:
        raise RuntimeError("Overlap can't be negative")
    deboss = p.get("em_op", "EMBOSS") == "DEBOSS"
    groups = features.profile_groups(profiles)
    curves = [features.profile_curve(g) for g in groups]
    open_ = [g[0][0].name for g, c in zip(groups, curves) if not features.is_closed_curve(c)]
    if open_:
        raise RuntimeError("Emboss needs closed profiles: " + ", ".join(open_) + features._is_are(open_).rstrip())
    c, n_plane, psize = profile_plane(curves)
    hit = crossing_outlines(curves, n_plane, psize)
    if hit is not None:
        names = sorted({groups[k][0][0].name for k in hit})
        raise RuntimeError(f"two outlines of {' and '.join(names)} cross or touch (kerned letters?): space them "
                           "apart (Letter Spacing) or join them into one outline first")
    tb = flow.world_brep(body)
    if not tb.is_manifold_solid():
        raise RuntimeError(f"{body.name} is not a closed solid")
    size = max(flow.brep_size(tb), psize)
    info = {"mode": mode}
    if base is not None and mode != "PROJECT":
        fmap, how = wrap_map(base, body, face, p)
        info["wrap"] = how
        foot, n, e1, _ = base_frame(base, c)
        m = fmap.map
    else:
        n = n_plane
        e1 = _unit(np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]))
        fmap = ProjectMap(tb, c, n, size, far=bool(p.get("em_far")))
        foot = c
        m = fmap.map
    if mode == "PLACE":
        A = rigid_matrix(m, foot, n, e1, size)
        m = lambda Q, A=A: np.asarray(Q, float) @ A[:3, :3].T + A[:3, 3]
    # the side: a step along n from the profiles' centre must go out of the body
    eps = 1e-3 * psize
    X0, X1 = m(np.array([foot, foot + eps * n]))
    _, _, nout = _hit_face(tb, X0)
    if float((X1 - X0) @ nout) == 0.0:
        raise RuntimeError("the profiles' plane is edge-on to the surface there")
    s = 1.0 if float((X1 - X0) @ nout) > 0.0 else -1.0
    lo, hi = (H, D) if deboss else (D, H)
    prism = _prism(profiles, s * n, lo, hi)
    if mode == "PLACE":
        store.transform(prism, A)
        out = prism
    else:
        try:
            out, err = flow.flow_brep(prism, fmap)
        except Exception as exc:
            raise RuntimeError("the lettering could not be laid on: it crosses a step or an edge of the body "
                               f"where the surface under it jumps ({exc})") from exc
        info["fit"] = err / fmap.size
    if not out.is_manifold_solid():
        raise RuntimeError("the laid-on profiles came out open (try a larger Overlap, or Place for small ones)")
    return out, info


@features.builder(OP)
def build_emboss(inputs, p):
    """inputs: (body, SURF or F<i>), [(sheet, SURF)], profiles...; the
    embossed / debossed body in world space."""
    from . import jobs
    from .steps import IDENT
    body = inputs[0][0]
    tool, info = emboss_tool(inputs, p)
    LAST_INFO.clear()
    LAST_INFO.update(info)
    op = "DIFFERENCE" if p.get("em_op", "EMBOSS") == "DEBOSS" else "UNION"
    objs = {"s0": body, "s1": (store.brep_to_text(tool), IDENT)}
    res, breps = jobs.run("boolean", objs, ["result"], order=list(objs), into="_world_", operation=op,
                          names={"s0": body.name, "s1": "the lettering"})
    LAST_INFO.update(faces=res.get("faces"), volume=res.get("volume"))
    return ("brep_text", breps["result"])


# --------------------------------------------------------------------------
# The operator
# --------------------------------------------------------------------------

def _is_solid(o):
    try:
        return store.is_nurbs(o) and store.get_brep(o).is_manifold_solid()
    except Exception:
        return False


def guess_roles(context):
    """(body, sheet or None, profile curve objects) from the selection: the
    active closed solid is the body, a NURBS sheet the profiles' base, the
    curves the profiles."""
    act = context.active_object
    sel = list(context.selected_objects)
    body = act if _is_solid(act) else next((o for o in sel if _is_solid(o)), None)
    curves = [o for o in sel if o.type == 'CURVE']
    sheets = [o for o in sel if o is not body and store.is_nurbs(o) and not _is_solid(o)]
    return body, (sheets[0] if sheets else None), curves


def profile_inputs(curves):
    return [(o, f"S{i}") for o in curves for i in range(len(o.data.splines))]


class NPROTO_OT_emboss(bpy.types.Operator):
    """Raised or sunk lettering, logos and pads on any face of a solid. Select
    the closed profile curves (and the sheet they lie on, to wrap), then the
    solid last (active). Wrap bends them from the sheet (an unrolled sheet
    folds back exactly), Project throws them along their plane's normal,
    Place moves them as a whole"""
    bl_idname = "nproto.emboss"
    bl_label = "Emboss"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Mode", items=MODE_ITEMS, default="PROJECT")
    operation: EnumProperty(name="Operation", items=OP_ITEMS, default="EMBOSS")
    height: FloatProperty(name="Height", default=0.1, min=0.0, unit='LENGTH', soft_max=10.0,
                          description="How far the lettering stands out (Deboss: sinks in), measured outward")
    overlap: FloatProperty(name="Overlap", default=0.025, min=0.0, unit='LENGTH', soft_max=1.0,
                           description="How far the lettering runs on through the surface (into the body for "
                                       "Emboss, out of it for Deboss), so the Boolean never meets faces lying on "
                                       "the surface")
    far: BoolProperty(name="Far Side", default=False,
                      description="Project: onto the farthest surface along each line, not the nearest")
    face: IntProperty(name="Target Face", default=-1, min=-1,
                      description="Wrap from a sheet not unrolled from the body: the face it maps onto "
                                  "(-1: the largest)")
    flip_u: BoolProperty(name="Flip U", default=False)
    flip_v: BoolProperty(name="Flip V", default=False)
    swap: BoolProperty(name="Swap UV", default=False)
    body: StringProperty(name="Body", description="The solid to emboss")
    base: StringProperty(name="Sheet", description="Wrap: the sheet the profiles lie on")
    objects: StringProperty(options={'HIDDEN'})          # JSON names of the profile curves

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def draw(self, context):
        lay = self.layout
        lay.row().prop(self, "operation", expand=True)
        lay.row().prop(self, "mode", expand=True)
        lay.prop(self, "height")
        lay.prop(self, "overlap")
        lay.prop_search(self, "body", context.scene, "objects")
        if self.mode != "PROJECT":
            lay.prop_search(self, "base", context.scene, "objects")
        if self.mode == "PROJECT":
            lay.prop(self, "far")
        elif not self._folds():
            lay.prop(self, "face")
            row = lay.row(align=True)
            row.prop(self, "flip_u", toggle=True)
            row.prop(self, "flip_v", toggle=True)
            row.prop(self, "swap", toggle=True)

    def _folds(self):
        b, t = bpy.data.objects.get(self.base), bpy.data.objects.get(self.body)
        return b is not None and t is not None and flow.unroll_link(b, t) is not None

    def _from_selection(self, context):
        """Body, sheet and profiles from the selection (a re-run keeps the names
        it was given); False when they are not there."""
        body, sheet, curves = guess_roles(context)
        if body is None:
            self.report({'WARNING'}, "Select the profile curves, then the solid to emboss last (active)")
            return False
        if not curves:
            self.report({'WARNING'}, "Select closed profile curves to emboss too")
            return False
        self.body = body.name
        self.base = sheet.name if sheet is not None else ""
        self.objects = json.dumps([o.name for o in curves])
        if sheet is not None and not self.properties.is_property_set("mode"):
            self.mode = "WRAP"
        if not self.properties.is_property_set("height"):
            try:
                _, _, psize = profile_plane([features.input_curve(o, r) for o, r in profile_inputs(curves)])
                self.height = float(f"{0.05 * psize:.2g}")
                if not self.properties.is_property_set("overlap"):
                    self.overlap = float(f"{0.25 * self.height:.2g}")
            except Exception:
                pass
        return True

    def invoke(self, context, event):
        if not self._from_selection(context):
            return {'CANCELLED'}
        return self.execute(context)

    def params(self):
        return {"em_mode": self.mode, "em_op": self.operation, "em_height": self.height, "em_overlap": self.overlap,
                "em_far": self.far, "fw_flip_u": self.flip_u, "fw_flip_v": self.flip_v, "fw_swap": self.swap,
                "fw_fold": True}

    def execute(self, context):
        from .ops import _kernel_ok, _select_only
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if not self.body and not self._from_selection(context):
            return {'CANCELLED'}
        body = bpy.data.objects.get(self.body)
        if body is None or not _is_solid(body):
            self.report({'WARNING'}, "Pick a closed NURBS solid to emboss")
            return {'CANCELLED'}
        try:
            names = json.loads(self.objects or "[]")
        except Exception:
            names = []
        curves = [o for o in (bpy.data.objects.get(n) for n in names) if o is not None and o.type == 'CURVE']
        if not curves:
            curves = [o for o in context.selected_objects if o.type == 'CURVE']
        if not curves:
            self.report({'WARNING'}, "No profile curves")
            return {'CANCELLED'}
        base = bpy.data.objects.get(self.base) if self.base else None
        if self.mode == "WRAP" and (base is None or not store.is_nurbs(base)):
            self.report({'WARNING'}, "Wrap needs the sheet the profiles lie on: Unroll the body and draw on the "
                                     "sheet, or pick a flat sheet")
            return {'CANCELLED'}
        use_base = base is not None and self.mode != "PROJECT"
        ref = f"F{self.face}" if use_base and self.face >= 0 and not self._folds() else "SURF"
        inputs = [(body, ref)] + ([(base, "SURF")] if use_base else []) + profile_inputs(curves)
        params = self.params()
        try:
            g = features.run(OP, inputs, params)
        except Exception as exc:
            self.report({'ERROR'}, f"Emboss failed: {exc}")
            return {'CANCELLED'}
        if features.history_enabled(context):
            res = store.new_object_text(context, f"{body.name}_{self.operation.lower()}", g[1])
            for c in body.users_collection:
                if res.name not in c.objects:
                    c.objects.link(res)
            for c in list(res.users_collection):
                if c not in body.users_collection and body.users_collection:
                    c.objects.unlink(res)
            features.record(res, OP, inputs, params)
            features.keep_inputs(context, [body] + curves + ([base] if use_base else []))
            features.mark_built(res)
        else:
            from .ops_trim import _frozen
            _frozen(body, self)
            store.set_geometry_text(body, g[1], context, keep_untrimmed=False)
            if store.PROP_UNTRIM in body:
                del body[store.PROP_UNTRIM]
            res = body
        _select_only(context, [res])
        how = {"PROJECT": "projected", "PLACE": "placed"}.get(self.mode, "folded back" if LAST_INFO.get("wrap") ==
                                                              "fold" else "wrapped")
        self.report({'INFO'}, f"{self.operation.title()}: {len(curves)} profile object"
                              f"{'s' if len(curves) != 1 else ''} {how} onto {body.name}")
        return {'FINISHED'}


classes = (NPROTO_OT_emboss,)
