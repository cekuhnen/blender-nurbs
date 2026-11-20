"""Flange and Draft Surface.

A ruled surface grown from a surface edge (or a curve on a surface), its
rulings square to the edge:

* Flange: at an Angle to the surface. 0 degrees continues the surface
  tangentially (G1 across the edge), 90 stands square to it along its
  outward normal, negative angles go to the back. The angle is measured in
  the plane square to the edge at every point, from the surface's own
  direction out of the edge.
* Draft: at a constant Draft angle to a Pull direction (a mould's). The
  rulings tilt away from the face (positive angles; negative: over it), and
  the angle is kept for the surface itself, not just its rulings: where the
  edge runs slanted to the pull, the ruling leans further (sin b = sin a /
  |P across the edge|), so the surface's draft is the same all along.
  Length along the ruling or as Height along the pull (a flat top edge for a
  planar parting line).

The base row IS the edge: its own CVs, knots and weights (knots added only
where the far edge needs them, which leaves the edge unchanged), so the
flange meets the surface exactly. The far row is fitted on that basis within
Tolerance. Edges are split at their corners; each smooth piece is one face.
Where two faces meet at a corner (two picked edges, or a corner inside one)
Corners decides what happens there: Mitre moves both far edges to where they
meet (an outside gap closes, an inside overlap is cut back), Blend fills an
outside gap with a round corner (a cone from the corner point, its far edge
an arc tangent to both far edges; inside corners are mitred), Open leaves them
as they are. The faces are stitched into one body where they meet.

Edges: a free edge of a face (on a trimmed face too); the side away from the
face is found by probing it. Curves on a surface: the left side seen from
the surface's outside (Flip: the right, Both Sides: both).
"""

import math

import numpy as np

from . import curvejoin, features, geom, store, surfeval
from . import surfoffset as so

LAST_INFO = {}


def _unit_rows(V):
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    return V / np.where(n > 1e-300, n, 1.0)


def _dot(A, B):
    return np.einsum("ij,ij->i", A, B)


# --------------------------------------------------------------------------
# Inputs: the edge's pieces and the face it bounds
# --------------------------------------------------------------------------

def base_pieces(obj, ref):
    """The input as clamped homogeneous B-splines, split at its corners:
    [(degree, Pw (n, 4), knots)] (world)."""
    spans = curvejoin.input_spans(obj, ref)
    cuts = curvejoin.split_points(spans)
    bounds = [0] + cuts + [len(spans)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sp = curvejoin.spans_to_spec(spans[a:b], simplify=True)
        d, P, W, U = geom.blender_nurbs_to_clamped(sp["points"], sp["weights"], sp["order"], sp.get("cyclic", False),
                                                   sp["endpoint"], sp["bezier"])
        P = np.asarray(P, float)
        W = np.asarray(W, float)
        out.append((int(d), np.c_[P * W[:, None], W], [float(k) for k in U]))
    return out, len(cuts)


def _single_face_target(face, size):
    tg = so.Target.__new__(so.Target)
    tg.size = size
    tg.faces = [so._Face(face, size)]
    tg.eps = 1e-7 * size
    return tg


def edge_face(obj, ref):
    """(target of the one face the edge bounds, None) or (None, why not)."""
    from .sweeps import _world_brep
    if not store.is_nurbs(obj):
        return None, "Pick an edge of a NURBS surface"
    brep = _world_brep(obj)
    bb = brep.bounding_box()
    size = float(np.linalg.norm(np.array(bb[1]) - np.array(bb[0]))) or 1.0
    if features.is_brep_edge_ref(ref):
        edge = brep.edges()[int(ref[1:])]
        faces = edge.faces()
        if len(faces) != 1:
            return None, "This edge joins two faces: a flange grows from a free edge (Unjoin first)"
        face = faces[0]
    else:
        face = brep.faces()[0]
    return _single_face_target(face, size), None


# --------------------------------------------------------------------------
# The direction field along a piece
# --------------------------------------------------------------------------

def _frame(Pw, U, p, ts):
    from .kernel.classa import basis
    H, H1 = basis(U, p, ts) @ Pw, basis(U, p, ts, 1) @ Pw
    C = H[:, :3] / H[:, 3:]
    return C, (H1[:, :3] - C * H1[:, 3:]) / H[:, 3:]


class Field:
    """Directions square to a piece: d(C, T) -> (unit directions, lengths)."""

    def __init__(self, tg, p, sigma):
        self.tg, self.p, self.sigma = tg, p, sigma
        self.mode = p.get("fl_mode", "FLANGE")
        self.length = float(p.get("fl_length", 1.0))
        self.angle = math.radians(float(p.get("angle", 90.0)))
        self.draft = math.radians(float(p.get("draft_angle", 3.0)))
        pull = np.asarray(p.get("pull", [0.0, 0.0, 1.0]), float)
        self.pull = pull / (np.linalg.norm(pull) or 1.0) * (-1.0 if p.get("pull_flip") else 1.0)
        self.height = p.get("measure", "RULING") == "HEIGHT"
        self.steep = 0           # samples where the edge runs too close to the pull for the draft

    def frame(self, C, T):
        """(normal N, tangent t, out-of-face direction b) at the points C."""
        F, UV, S, N, dist = self.tg.feet_all(C)
        bad = F < 0
        if bad.all():
            raise ValueError("the edge is not on the surface")
        if bad.any():                    # (a point a little off the face: its nearest neighbour's normal)
            good = np.nonzero(~bad)[0]
            for i in np.nonzero(bad)[0]:
                N[i] = N[good[np.argmin(np.abs(good - i))]]
        t = _unit_rows(T - _dot(T, N)[:, None] * N)
        b = np.cross(N, t) * self.sigma
        return N, t, b

    def __call__(self, C, T):
        N, t, b = self.frame(C, T)
        if self.mode == "FLANGE":
            d = math.cos(self.angle) * b + math.sin(self.angle) * N
            return d, np.full(len(C), self.length)
        P = self.pull[None, :]
        Pp = P - _dot(np.repeat(P, len(t), 0), t)[:, None] * t
        mag = np.linalg.norm(Pp, axis=1)
        Pu = Pp / np.maximum(mag, 1e-300)[:, None]
        s = np.cross(t, Pu)
        s *= np.where(_dot(s, b) >= 0, 1.0, -1.0)[:, None]      # tilts away from the face
        sinb = math.sin(self.draft) / np.maximum(mag, 1e-300)
        self.steep = int(np.count_nonzero(np.abs(sinb) > 1.0))
        beta = np.arcsin(np.clip(sinb, -1.0, 1.0))
        d = np.cos(beta)[:, None] * Pu + np.sin(beta)[:, None] * s
        if self.height:
            up = d @ self.pull                  # = cos(b) |P across|: the ruling's rise per length
            L = self.length / np.where(np.abs(up) > 1e-6, up, 1e-6)
        else:
            L = np.full(len(C), self.length)
        return d, L

    def normal_hint(self, C, T):
        """The side the flange should face: the surface's outside turned with it."""
        N, t, b = self.frame(C, T)
        d, _ = self(C, T)
        m = np.cross(d, t)
        sgn = np.sign(_dot(np.cross(b, t), N))          # b x t is +-N: turn so that d = b gives N
        return m * sgn[:, None]


def edge_sigma(tg, Pw, U, p):
    """+1 when N x t points away from the face along the piece, else -1 (a
    probe a little to that side lands off the face)."""
    ts = np.linspace(U[p], U[-p - 1], 23)[1:-1]
    C, T = _frame(Pw, U, p, ts)
    F, UV, S, N, _ = tg.feet_all(C)
    ok = F >= 0
    if not ok.any():
        raise ValueError("the edge is not on the surface")
    t = _unit_rows(T - _dot(T, N)[:, None] * N)
    nb = np.cross(N, t)
    eps = 2e-3 * tg.size
    Y = S[ok] + eps * nb[ok]
    _, _, _, _, inside = tg.project(Y, F[ok], UV[ok], gap=eps)
    return -1.0 if inside.mean() > 0.5 else 1.0


# --------------------------------------------------------------------------
# The far row, fitted on the edge's own basis
# --------------------------------------------------------------------------

def _span_params(U, p, per):
    br = sorted(set(U[p:len(U) - p]))
    return np.concatenate([np.linspace(a, b, per + 1)[(1 if i else 0):] for i, (a, b) in enumerate(zip(br[:-1], br[1:]))])


def fit_far(Pw, U, p, field, tol, rounds=12, per=10):
    """(Pw, U, Q, error): the base (knots added where needed) and far CVs Q
    (Euclidean, on the base's weights) of the far edge C + L d."""
    from .kernel.classa import basis, insert_knot
    Pw = np.asarray(Pw, float)
    U = list(U)
    err = np.inf
    Q = None
    for _ in range(rounds):
        ts = _span_params(U, p, per)
        C, T = _frame(Pw, U, p, ts)
        d, L = field(C, T)
        F = C + L[:, None] * d
        w = Pw[:, 3]
        B = basis(U, p, ts)
        R = B * w[None, :] / (B @ w)[:, None]
        n = len(w)
        Q = np.zeros((n, 3))
        Q[0], Q[-1] = F[0], F[-1]               # clamped: the ends exactly
        if n > 2:
            rhs = F - np.outer(R[:, 0], Q[0]) - np.outer(R[:, -1], Q[-1])
            Q[1:-1] = np.linalg.lstsq(R[:, 1:-1], rhs, rcond=None)[0]
        tm = 0.5 * (ts[:-1] + ts[1:])
        Cm, Tm = _frame(Pw, U, p, tm)
        dm, Lm = field(Cm, Tm)
        Bm = basis(U, p, tm)
        Rm = Bm * w[None, :] / (Bm @ w)[:, None]
        E = np.linalg.norm(Rm @ Q - (Cm + Lm[:, None] * dm), axis=1)
        err = float(E.max()) if len(E) else 0.0
        if err <= tol:
            break
        # a knot in the middle of every span that misses (the base is unchanged by it)
        br = sorted(set(U[p:len(U) - p]))
        new = []
        for a, b in zip(br[:-1], br[1:]):
            k = (tm >= a) & (tm <= b)
            if k.any() and E[k].max() > tol:
                new.append(0.5 * (a + b))
        if not new or len(Pw) > 2000:
            break
        for k in new:
            Pw, U = insert_knot(Pw, U, p, k)
            Pw = np.asarray(Pw, float)
    return Pw, U, Q, err


def ruled_surface(Pw, U, p, Q):
    """Kernel surface: u across (degree 1, base row first), v along the edge."""
    _, ext = so.store._k()
    w = Pw[:, 3]
    P = Pw[:, :3] / w[:, None]
    pts = [tuple(map(float, x)) for x in P] + [tuple(map(float, x)) for x in Q]
    wts = None if np.abs(w - 1.0).max() < 1e-12 else [float(x) for x in w] * 2
    return ext.surface_from_cvs(1, p, pts, [0.0, 0.0, 1.0, 1.0], [float(k) for k in U], wts)


def _faces_hint(surf, Pw, U, p, field):
    """+1 when the surface faces the way the field wants, else -1."""
    E = surfeval.Surf(surf)
    tm = np.array([U[p] + (U[-p - 1] - U[p]) * f for f in (0.3, 0.5, 0.7)])
    n, _ = surfeval.normals(E.at(np.full(3, 0.0), tm, 1))
    C, T = _frame(Pw, U, p, tm)
    m = field.normal_hint(C, T)
    return 1.0 if float(np.sum(n * m)) >= 0 else -1.0


# --------------------------------------------------------------------------
# Corners: where two faces meet at one base point
# --------------------------------------------------------------------------

def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-300 else v


def _end(B, F, e):
    """(base point, far point, the far edge's direction leaving the face) at end e (0 start, 1 end)."""
    if e:
        return B[-1], F[-1], _unit(F[-1] - F[-2])
    return B[0], F[0], _unit(F[0] - F[1])


def corner_pairs(rows, size):
    """[((i, e), (j, f))]: ends of faces (rows [(base, far)]) that meet at one
    base point, paired by the closest ruling directions (on both sides of a
    curve, each side meets its own side)."""
    tol = 1e-4 * size
    ends = [(i, e) for i in range(len(rows)) for e in (0, 1)]
    cand = []
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            (i, e), (j, f) = ends[a], ends[b]
            Va, Fa, _ = _end(*rows[i], e)
            Vb, Fb, _ = _end(*rows[j], f)
            if np.linalg.norm(Va - Vb) > tol:
                continue
            dot = float(_unit(Fa - Va) @ _unit(Fb - Vb))
            if dot > -0.5:
                cand.append((-dot, a, b))
    cand.sort()
    used, pairs = set(), []
    for _, a, b in cand:
        if a not in used and b not in used:
            used |= {a, b}
            pairs.append((ends[a], ends[b]))
    return pairs


def corner_meet(rows, pair, size):
    """(kind, M, phi): where the two far edges' end directions meet. kind:
    'meets' (the far ends already coincide), 'smooth' (the faces run on
    tangentially: snap them together), 'outside' (a gap: both far edges reach
    on to M), 'inside' (an overlap: both are cut back to M), 'skew' (one
    each), 'open' (a hairpin, no sane meeting point). phi: the turn at M."""
    (i, e), (j, f) = pair
    Va, Fa, ta = _end(*rows[i], e)
    Vb, Fb, tb = _end(*rows[j], f)
    tol = 1e-6 * size
    mid = 0.5 * (Fa + Fb)
    if np.linalg.norm(Fa - Fb) <= tol:
        return "meets", mid, 0.0
    A = np.array([[ta @ ta, -(ta @ tb)], [ta @ tb, -(tb @ tb)]])
    phi = math.acos(max(-1.0, min(1.0, float(ta @ -tb))))
    if abs(np.linalg.det(A)) < 1e-10 or phi < math.radians(0.5):
        return "smooth", mid, phi
    s_, u_ = np.linalg.solve(A, np.array([(Fb - Fa) @ ta, (Fb - Fa) @ tb]))
    M = 0.5 * (Fa + s_ * ta + Fb + u_ * tb)
    reach = max(float(np.linalg.norm(Fa - Va)), float(np.linalg.norm(Fb - Vb)), tol)
    if np.linalg.norm(M - mid) > 20.0 * reach:
        return "open", mid, phi
    if s_ > 0 and u_ > 0:
        return "outside", M, phi
    if s_ < 0 and u_ < 0:
        return "inside", M, phi
    return "skew", M, phi


def _straight(X, size):
    if len(X) <= 2:
        return True
    a, b = X[0], X[-1]
    L = _unit(b - a)
    R = (X - a) - np.outer((X - a) @ L, L)
    return float(np.linalg.norm(R, axis=1).max()) <= 1e-7 * size


def move_far_end(B, F, e, delta, size, movable=None):
    """Move the far edge's end e by delta, the rest following: a straight face
    (base and far edge lines) is stretched along its line (an affine map: it
    stays straight and flat), a curved one only near the corner (over about
    3 |delta|; with movable -- a CV row -- only those CVs, whose spans lie
    near the corner, ramping down along them). False when the far edge would
    fold back on itself."""
    n = len(F)
    if n == 2 or (_straight(B, size) and _straight(F, size)):
        seg = np.linalg.norm(np.diff(F, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        tot = cum[-1]
        if tot <= 1e-300 or (delta @ _end(B, F, e)[2] < 0 and np.linalg.norm(delta) > 0.9 * tot):
            return False
        g = cum / tot if e else 1.0 - cum / tot
    else:
        order = np.arange(n)[::-1] if e else np.arange(n)
        seg = np.linalg.norm(np.diff(F[order], axis=0), axis=1)
        dist = np.concatenate([[0.0], np.cumsum(seg)])           # from the end, along the far edge
        tot = dist[-1]
        r = min(0.5 * tot, 3.0 * float(np.linalg.norm(delta)))
        if movable is not None:
            mv = np.asarray(movable, bool)[order]
            k = int(np.argmin(mv)) if not mv.all() else n - 1     # the first CV that must stay
            r = max(dist[k], 1e-300)
        if delta @ _end(B, F, e)[2] < 0 and np.linalg.norm(delta) > 0.9 * r:
            return False
        g = np.zeros(n)
        g[order] = np.maximum(0.0, 1.0 - dist / r)
        if movable is not None:
            g[~np.asarray(movable, bool)] = 0.0
    F += np.outer(g, delta)
    return True


def conic(Fa, M, Fb, w, n):
    """Points of the rational quadratic arc Fa -> Fb with middle CV M (weight w)."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    b0, b1, b2 = (1 - t) ** 2, 2 * t * (1 - t) * w, t ** 2
    return (b0 * Fa + b1 * M + b2 * Fb) / (b0 + b1 + b2)


def apply_corners(rows, mode, size, movable=None):
    """Corner the faces (rows [(base, far)], far rows moved in place) as mode
    says: returns (fans [(V, Fa, M, Fb, w)] for Blend, counts). movable:
    {(face, end): CV mask} for CV rows (see move_far_end)."""
    movable = movable or {}
    counts = {"mitre": 0, "blend": 0, "open": 0}
    fans = []
    for pair in corner_pairs(rows, size):
        (i, e), (j, f) = pair
        kind, M, phi = corner_meet(rows, pair, size)
        if kind == "meets":
            continue
        Fa, Fb = _end(*rows[i], e)[1], _end(*rows[j], f)[1]
        if kind == "smooth":                      # tangent faces: one far point
            move_far_end(*rows[i], e, M - Fa, size, movable.get((i, e)))
            move_far_end(*rows[j], f, M - Fb, size, movable.get((j, f)))
            continue
        if mode == "OPEN" or kind == "open":
            counts["open"] += 1
            continue
        if mode == "BLEND" and kind == "outside":
            fans.append((_end(*rows[i], e)[0].copy(), Fa.copy(), M, Fb.copy(), math.cos(0.5 * phi)))
            counts["blend"] += 1
            continue
        Fa0, Fb0 = rows[i][1].copy(), rows[j][1].copy()
        if move_far_end(*rows[i], e, M - Fa, size, movable.get((i, e))) and \
                move_far_end(*rows[j], f, M - Fb, size, movable.get((j, f))):
            counts["mitre"] += 1
        else:                                     # (folds: put both back)
            rows[i][1][:] = Fa0
            rows[j][1][:] = Fb0
            counts["open"] += 1
    return fans, counts


def _refine_ends(fs, pairs, rows, size):
    """Curved faces with a corner to move: degree + 1 knots |delta| apart at
    that end (the edge and far edge unchanged), so the move stays near the
    corner. Returns {(face, end): the CVs whose spans lie in that stretch}."""
    from .kernel.classa import insert_knot
    marks = {}
    for pair in pairs:
        kind, M, _ = corner_meet(rows, pair, size)
        if kind in ("meets", "open"):
            continue
        for i, e in pair:
            f = fs[i]
            B, F = rows[i]
            if _straight(B, size) and _straight(F, size):
                continue
            step = float(np.linalg.norm(M - _end(B, F, e)[1]))
            p, U, Q = f["deg"], f["U"], f["Q"]
            a, b = U[p], U[-p - 1]
            n = len(Q) - 1
            # the far edge's speed at that end, so the knots sit |delta|, 2|delta|, 3|delta| from it
            if e:
                speed = p / ((U[n + p] - U[n]) or 1.0) * float(np.linalg.norm(Q[n] - Q[n - 1]))
            else:
                speed = p / ((U[p + 1] - U[1]) or 1.0) * float(np.linalg.norm(Q[1] - Q[0]))
            if speed <= 1e-300 or step <= 1e-12 * size:
                continue
            dt = min(step / speed, (b - a) / (2.0 * (p + 1)))
            marks[(i, e)] = dt
            for k in range(1, p + 2):
                t = b - k * dt if e else a + k * dt
                if any(abs(t - u) < 1e-9 * (b - a) for u in f["U"]):
                    continue
                w = f["Pw"][:, 3:4]
                A, U2 = insert_knot(np.hstack([f["Pw"], f["Q"] * w]), f["U"], p, t)
                A = np.asarray(A, float)
                f["Pw"], f["U"], f["Q"] = A[:, :4], list(U2), A[:, 4:] / A[:, 3:4]
            rows[i] = (f["Pw"][:, :3] / f["Pw"][:, 3:4], f["Q"])
    movable = {}
    for (i, e), dt in marks.items():      # (after every insertion: a face can have a corner at both ends)
        p, U = fs[i]["deg"], fs[i]["U"]
        a, b = U[p], U[-p - 1]
        n = len(fs[i]["Q"])
        lim = (p + 1) * dt * (1.0 + 1e-9)
        movable[(i, e)] = [(b - U[k]) <= lim if e else (U[k + p + 1] - a) <= lim for k in range(n)]
    return movable


def fan_surface(V, Fa, M, Fb, w):
    """A Blend corner: rulings from V to the arc Fa -> Fb (a cone when the arc is round)."""
    _, ext = so.store._k()
    pts = [tuple(map(float, V))] * 3 + [tuple(map(float, x)) for x in (Fa, M, Fb)]
    return ext.surface_from_cvs(1, 2, pts, [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                [1.0, float(w), 1.0] * 2)


def _normal_at(surf, u, v):
    E = surfeval.Surf(surf)
    n, _ = surfeval.normals(E.at(np.array([u]), np.array([v]), 1))
    return n[0]


# --------------------------------------------------------------------------
# Whole inputs
# --------------------------------------------------------------------------

def _targets_for(inputs):
    """[(obj, ref, target, sides)] for the curve / edge inputs."""
    curves = [(o, r) for o, r in inputs if r != "SURF"]
    surfs = [o for o, r in inputs if r == "SURF"]
    out = []
    shared = None
    for o, r in curves:
        if features.is_surface_edge(r):
            tg, why = edge_face(o, r)
            if tg is None:
                raise ValueError(why)
            out.append((o, r, tg, None))
        else:
            if shared is None:
                if not surfs:
                    raise ValueError("a curve needs the surface it lies on")
                shared = so.target_of(surfs)
            out.append((o, r, shared, "CURVE"))
    return out


def prepare(inputs):
    """What does not change with the settings: per input piece (degree, Pw,
    knots, target, kind, the edge's away-from-face sign); and the corner count."""
    out, corners = [], 0
    for o, r, tg, kind in _targets_for(inputs):
        pieces, c = base_pieces(o, r)
        corners += c
        for deg, Pw, U in pieces:
            s = None if kind == "CURVE" else edge_sigma(tg, Pw, U, deg)
            out.append((deg, Pw, U, tg, kind, s))
    return out, corners


def _sigmas(kind, s, p):
    if kind == "CURVE":
        s = -1.0 if p.get("flip") else 1.0
        return (s, -s) if p.get("both_sides") else (s,)
    return (-s,) if p.get("flip") else (s,)


def faces_rows(prepared, p, n=48):
    """Preview: per face, the base and far rows of points (no fitting), the
    corners done as the build does them (Blend corners as extra faces)."""
    rows = []
    for deg, Pw, U, tg, kind, s in prepared:
        for sigma in _sigmas(kind, s, p):
            field = Field(tg, p, sigma)
            ts = np.linspace(U[deg], U[-deg - 1], n)
            C, T = _frame(Pw, U, deg, ts)
            d, L = field(C, T)
            rows.append((C, C + L[:, None] * d))
    if not rows:
        return []
    size = float(np.linalg.norm(np.ptp(np.vstack([r[0] for r in rows]), axis=0))) or 1.0
    fans, _ = apply_corners(rows, p.get("fl_corner", "OPEN"), size)
    out = [[B, F] for B, F in rows]
    for V, Fa, M, Fb, w in fans:
        out.append([np.repeat(V[None, :], 17, axis=0), conic(Fa, M, Fb, w, 17)])
    return out


def anchor(prepared, p, at):
    """The frame at the input point nearest at (world), for the handles:
    dict(C, t, N, b, d, L, Pu, s, mag) or None."""
    best = None
    for deg, Pw, U, tg, kind, s in prepared:
        ts = np.linspace(U[deg], U[-deg - 1], 33)
        C, T = _frame(Pw, U, deg, ts)
        k = int(np.argmin(np.linalg.norm(C - np.asarray(at, float), axis=1)))
        dist = float(np.linalg.norm(C[k] - at))
        if best is None or dist < best[0]:
            best = (dist, C[k:k + 1], T[k:k + 1], tg, _sigmas(kind, s, p)[0])
    if best is None:
        return None
    _, C, T, tg, sigma = best
    field = Field(tg, p, sigma)
    N, t, b = field.frame(C, T)
    d, L = field(C, T)
    P = field.pull
    Pp = P - float(P @ t[0]) * t[0]
    mag = float(np.linalg.norm(Pp))
    Pu = Pp / (mag or 1.0)
    sv = np.cross(t[0], Pu)
    if float(sv @ b[0]) < 0:
        sv = -sv
    return {"C": C[0], "t": t[0], "N": N[0], "b": b[0], "d": d[0], "L": float(L[0]), "Pu": Pu, "s": sv, "mag": mag,
            "pull": P}


def build(inputs, p, prepared=None):
    """The flange / draft faces of every input, in one brep."""
    _, ext = so.store._k()
    if abs(float(p.get("fl_length", 1.0))) < 1e-12:
        raise ValueError("Length is 0")
    if prepared is None:
        prepared = prepare(inputs)
    pieces, corners = prepared
    mode = p.get("fl_corner", "OPEN")
    info = {"faces": 0, "corners": corners, "err": 0.0, "steep": 0, "cvs": 0, "mitre": 0, "blend": 0, "open": 0}
    tol = float(p.get("tolerance", 1e-4))
    fs = []
    for deg, Pw0, U0, tg, kind, s in pieces:
        for sigma in _sigmas(kind, s, p):
            field = Field(tg, p, sigma)
            Pw, U, Q, err = fit_far(Pw0, U0, deg, field, tol)
            info["steep"] += field.steep
            info["err"] = max(info["err"], err)
            fs.append({"Pw": np.asarray(Pw, float), "U": list(U), "deg": deg, "Q": Q, "field": field})
    if not fs:
        raise ValueError("nothing to grow a flange from")
    # corners, on the CV rows (a CV row's end and end direction are the edge's)
    rows = [(f["Pw"][:, :3] / f["Pw"][:, 3:4], f["Q"]) for f in fs]
    size = float(np.linalg.norm(np.ptp(np.vstack([r[0] for r in rows]), axis=0))) or 1.0
    movable = _refine_ends(fs, corner_pairs(rows, size), rows, size) if mode != "OPEN" else {}
    fans, counts = apply_corners(rows, mode, size, movable)
    info.update(counts)
    parts, surfs = [], []
    for f, (_, Q) in zip(fs, rows):
        Pw, U, deg = f["Pw"], f["U"], f["deg"]
        surf = ruled_surface(Pw, U, deg, Q)
        flipped = _faces_hint(surf, Pw, U, deg, f["field"]) < 0
        if flipped:
            # turned over: the far row first
            w = Pw[:, 3]
            surf = ruled_surface(np.c_[Q * w[:, None], w], U, deg, Pw[:, :3] / w[:, None])
        surfs.append((surf, Pw[:, :3] / Pw[:, 3:4], U, deg))
        info["faces"] += 1
        info["cvs"] += 2 * len(Q)
        parts.append(ext.brep_from_surface(surf))
    for V, Fa, M, Fb, w in fans:
        surf = fan_surface(V, Fa, M, Fb, w)
        # face the way the flange it starts from does (its ruling to Fa)
        k = min(range(len(rows)), key=lambda i: min(np.linalg.norm(rows[i][1][0] - Fa),
                                                    np.linalg.norm(rows[i][1][-1] - Fa)))
        nb, B, U, deg = surfs[k]
        at_end = np.linalg.norm(rows[k][1][-1] - Fa) < np.linalg.norm(rows[k][1][0] - Fa)
        v = U[-deg - 1] if at_end else U[deg]
        v += (-1e-3 if at_end else 1e-3) * (U[-deg - 1] - U[deg])
        if float(_normal_at(surf, 0.5, 0.0) @ _normal_at(nb, 0.5, v)) < 0:
            surf = fan_surface(V, Fb, M, Fa, w)
        info["faces"] += 1
        info["cvs"] += 6
        parts.append(ext.brep_from_surface(surf))
    brep = parts[0]
    for b in parts[1:]:
        ext.append_brep(brep, b)
    if len(parts) > 1:
        sm, _ = so.store._k()
        try:
            brep = sm.stitch_advanced(brep, 1e-6 * size, squeeze_small_edges=False,
                                      making_manifold_solid=False)[0]
        except Exception as exc:
            print("NURBS flange stitch:", exc)
    LAST_INFO.clear()
    LAST_INFO.update(info)
    return brep


@features.builder("flange")
def build_flange(inputs, p):
    return build(inputs, p)


def report_text(info, mode):
    msg = f"{'Flange' if mode == 'FLANGE' else 'Draft surface'}: {info['faces']} face(s)"
    notes = [f"far edge within {info['err']:.2g}"]
    for k, word in (("mitre", "mitred"), ("blend", "blended"), ("open", "left open")):
        if info.get(k):
            notes.append(f"{info[k]} corner(s) {word}")
    return msg + " · " + ", ".join(notes)


def warnings(info):
    out = []
    if info["steep"]:
        out.append("the edge runs closer to the pull direction than the draft angle in places: "
                   "the surface stands square to the edge there (less draft)")
    return out


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------

import json  # noqa: E402

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, StringProperty  # noqa: E402

from . import hints  # noqa: E402
from .ops import (AskValue, PickMixin, _decode_inputs, _encode_inputs, _kernel_ok, _make_result,  # noqa: E402
                  _select_only)
from .valuedrag import BASE_PX, ValueDrag, angle_pick, ruled_preview  # noqa: E402

MODE_ITEMS = [("FLANGE", "Flange", "At an angle to the surface: 0 continues it, 90 stands square to it"),
              ("DRAFT", "Draft", "At a constant draft angle to a pull direction, opening away from the face")]
PULL_ITEMS = [("Z", "World Z", ""), ("X", "World X", ""), ("Y", "World Y", ""),
              ("WORKPLANE", "Work Plane", "The active work plane's normal (World: Z)"),
              ("VIEW", "View", "Towards the viewer, as the view was when the tool started")]
MEASURE_ITEMS = [("RULING", "Along Surface", "Length along the rulings"),
                 ("HEIGHT", "Height", "Height along the pull direction (a planar edge gives a flat top)")]
PULLS = ("Z", "X", "Y", "WORKPLANE", "VIEW")
CORNER_ITEMS = [("MITRE", "Mitre", "Both far edges run on (or are cut back) to where they meet: a sharp corner"),
                ("BLEND", "Blend", "An outside corner gets a round corner (a cone from the corner point, tangent "
                                   "to both); inside corners are mitred"),
                ("OPEN", "Open", "Leave the corners as they are (a gap outside, an overlap inside)")]
CORNERS = ("MITRE", "BLEND", "OPEN")


def _free_edge(obj, ref):
    """Edge-pick candidates a flange can take: curves, untrimmed sides, free brep edges."""
    if not features.is_brep_edge_ref(ref):
        return True
    try:
        return len(store.get_brep(obj).edges()[int(ref[1:])].faces()) == 1
    except Exception:
        return False


class NPROTO_OT_flange(ValueDrag, AskValue, PickMixin, bpy.types.Operator):
    """Grow a ruled surface from surface edges (or curves on a surface): a
    Flange at an angle to the surface (0 continues it tangentially, 90 stands
    square to it), or a Draft surface at a constant draft angle to a pull
    direction. The base is the edge exactly. Click edges and Enter (or select
    curves on a surface), then drag the arrow and the angle, type, Enter"""
    bl_idname = "nproto.flange"
    bl_label = "Flange"
    bl_options = {'REGISTER', 'UNDO'}

    accepts_picks = True
    pick_min = 1
    pick_max = 256
    finish_on_min = False
    start_value = 0.0           # the arrow on the edge: nothing until it is dragged (L: the last length)
    angle_fields = ("angle", "draft")

    inputs: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    surfaces: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fl_mode: EnumProperty(name="Mode", items=MODE_ITEMS, default="FLANGE")
    length: FloatProperty(name="Length", default=0.5, unit='LENGTH', soft_min=-10.0, soft_max=10.0,
                          description="How far the surface reaches from the edge (negative: the other way)")
    angle: FloatProperty(name="Angle °", default=90.0, soft_min=-180.0, soft_max=180.0,
                         description="Flange: 0 continues the surface, 90 stands along its outward normal, "
                                     "negative angles to the back")
    draft: FloatProperty(name="Draft °", default=3.0, soft_min=-45.0, soft_max=45.0,
                               description="Draft: the surface's angle to the pull direction; positive opens "
                                           "away from the face")
    fl_pull: EnumProperty(name="Pull", items=PULL_ITEMS, default="Z")
    pull_flip: BoolProperty(name="Flip Pull", default=False, description="Pull the other way")
    measure: EnumProperty(name="Length", items=MEASURE_ITEMS, default="RULING")
    flip: BoolProperty(name="Flip Side", default=False,
                       description="Edges: grow over the face instead of away from it. Curves: the right side")
    both_sides: BoolProperty(name="Both Sides", default=False, description="Curves on a surface: both sides")
    fl_corner: EnumProperty(name="Corners", items=CORNER_ITEMS, default="MITRE",
                            description="Where two flanges meet at a corner")
    tolerance: FloatProperty(name="Tolerance", default=1e-4, min=1e-7, soft_max=0.01, precision=6, unit='LENGTH',
                             description="Largest deviation of the fitted far edge")
    view_dir: FloatVectorProperty(subtype='XYZ', options={'HIDDEN'}, default=(0.0, 0.0, 1.0))
    work_n: FloatVectorProperty(subtype='XYZ', options={'HIDDEN'}, default=(0.0, 0.0, 1.0))

    @property
    def drag_fields(self):
        return ("length", "angle") if self.fl_mode == "FLANGE" else ("length", "draft")

    # -- inputs ---------------------------------------------------------------
    def pick_candidates(self, context):
        return [c for c in super().pick_candidates(context) if _free_edge(c[0], c[1])]

    def pick_state(self):
        n = len(getattr(self, "_picks", []))
        return "click free surface edges (or curves on a surface)" + (f" · {n} picked, Enter" if n else "")

    def _inputs(self, context):
        picks = _decode_inputs(self.inputs)
        crv = [(o, r) for o, r in picks if not features.is_surface_edge(r)]
        if not crv:
            return picks
        objs = []
        if self.surfaces:
            objs = [bpy.data.objects[n] for n in json.loads(self.surfaces) if n in bpy.data.objects]
        if not objs:
            objs = [o for o in context.selected_objects if o.type != 'CURVE' and store.is_nurbs(o)]
        if not objs:
            cands = [o for o in context.visible_objects if o.type != 'CURVE' and store.is_nurbs(o)]
            objs = so.find_targets([curvejoin.input_spans(o, r) for o, r in crv], cands, sources=[o for o, _ in crv])
        self.surfaces = json.dumps([o.name for o in objs])
        return picks + [(o, "SURF") for o in objs]

    def invoke(self, context, event):
        from . import curvetrim, workplanes
        if not _kernel_ok(self):
            return {'CANCELLED'}
        curvetrim._object_mode(context)
        rv3d = getattr(context, "region_data", None)
        if rv3d is not None:
            from mathutils import Vector
            self.view_dir = rv3d.view_rotation @ Vector((0, 0, 1))
        fr = workplanes.active_frame(context.scene)
        if fr is not None:
            self.work_n = [float(x) for x in fr[3]]
        self.surfaces = ""
        if not self.inputs:
            crv = [o for o in context.selected_objects if o.type == 'CURVE' and len(o.data.splines)]
            if crv and not event.alt:
                self.inputs = _encode_inputs([(o, f"S{i}") for o in crv for i in range(len(o.data.splines))])
            else:
                return self.start_picking(context, event)
        self._start_event = (event.mouse_x, event.mouse_y, event.type, event.value)
        try:
            ins = self._inputs(context)
        except ValueError as exc:
            self.report({'WARNING'}, str(exc))
            return {'CANCELLED'}
        if any(r != "SURF" and not features.is_surface_edge(r) for _, r in ins) and not any(r == "SURF" for _, r in ins):
            self.report({'WARNING'}, "No surface under the curves: select the surface they lie on too")
            return {'CANCELLED'}
        return self.ask(context)

    def _p(self):
        pull = {"Z": (0, 0, 1), "X": (1, 0, 0), "Y": (0, 1, 0)}.get(self.fl_pull)
        if pull is None:
            pull = tuple(self.view_dir) if self.fl_pull == "VIEW" else tuple(self.work_n)
        return {"fl_mode": self.fl_mode, "fl_length": self.length, "angle": self.angle,
                "draft_angle": self.draft, "pull": [float(x) for x in pull], "pull_flip": self.pull_flip,
                "measure": self.measure, "flip": self.flip, "both_sides": self.both_sides,
                "fl_corner": self.fl_corner, "tolerance": self.tolerance}

    # -- the live value stage (valuedrag.py) ----------------------------------
    def drag_setup(self, context, mouse):
        try:
            self._ins = self._inputs(context)
            self._prep = prepare(self._ins)
        except Exception as exc:
            self.report({'WARNING'}, str(exc))
            return False
        if not self._prep[0]:
            return False
        P = np.vstack([_frame(Pw, U, deg, np.linspace(U[deg], U[-deg - 1], 33))[0]
                       for deg, Pw, U, *_ in self._prep[0]])
        self._pts = P
        self._size = float(np.ptp(P, axis=0).max()) or 1.0
        self._anchor = P[len(P) // 2]
        self._base_at_mouse = False
        if mouse is not None:
            best = None
            for q in P:
                sq = self._screen(context, q)
                if sq is not None and (sq - mouse).length <= BASE_PX and (best is None or (sq - mouse).length < best[0]):
                    best = ((sq - mouse).length, q)
            if best is not None:
                self._anchor, self._base_at_mouse = best[1], True
        self._cache = None
        return True

    def drag_on_input(self, context, mouse):
        return self._near_screen(context, self._pts, mouse)

    def drag_handles(self):
        p = self._p()
        a = anchor(self._prep[0], p, self._anchor)
        if a is None:
            return []
        C, d = a["C"], a["d"]
        rise = float(d @ a["pull"]) if (self.fl_mode == "DRAFT" and self.measure == "HEIGHT") else 1.0
        rise = rise if abs(rise) > 1e-6 else 1e-6
        hs = [{"field": "length", "kind": "arrow", "origin": C, "dir": d, "param": self.length / rise, "base": C,
               "to_value": lambda t, r=rise: t * r}]
        R = 0.6 * max(abs(a["L"]), 0.15 * self._size)
        t = a["t"]
        if self.fl_mode == "FLANGE":
            x, y, ang = a["b"], a["N"], self.angle
            field, to_value = "angle", (lambda v: v)
        else:
            x, y = a["Pu"], a["s"]
            mag = a["mag"]
            sb = math.sin(math.radians(self.draft)) / (mag or 1e-9)
            ang = math.degrees(math.asin(max(-1.0, min(1.0, sb))))
            field = "draft"
            to_value = (lambda v, m=mag: math.degrees(math.asin(max(-1.0, min(1.0, math.sin(math.radians(v)) * m)))))
        n = max(2, int(abs(ang) / 3.0) + 1)
        arc = np.array([C + R * (math.cos(math.radians(ang * i / (n - 1))) * x +
                                 math.sin(math.radians(ang * i / (n - 1))) * y) for i in range(n)])
        back = ang - math.copysign(min(6.0, abs(ang) * 0.5), ang) if abs(ang) > 1e-9 else 0.0
        hs.append({"field": field, "kind": "arrow", "pos": arc[-1], "param": ang, "origin": C, "dir": None,
                   "path": arc, "head_from": C + R * (math.cos(math.radians(back)) * x + math.sin(math.radians(back)) * y),
                   "pick": lambda o, r, prev, C=C, x=x, y=y, t=t: angle_pick(C, x, y, t, o, r, prev),
                   "to_value": to_value, "snap": (5.0, 1.0)})
        return hs

    def drag_preview(self):
        p = self._p()
        key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in p.items()))
        if self._cache is None or self._cache[0] != key:
            lines, tris = [], []
            try:
                for rows in faces_rows(self._prep[0], p):
                    L, T = ruled_preview(rows, ribs=8)
                    lines += L
                    tris.append(T)
            except Exception as exc:
                print("NURBS flange preview:", exc)
            self._cache = (key, {"lines": lines, "tris": np.vstack(tris) if tris else np.zeros((0, 3))})
        return self._cache[1]

    def drag_check(self):
        return "" if abs(self.length) > 1e-12 else "Length 0: drag the arrow out, or type a value"

    def drag_key(self, context, event, ch):
        if ch == "m":
            self.fl_mode = "DRAFT" if self.fl_mode == "FLANGE" else "FLANGE"
            self._field = 0
        elif ch == "f":
            self.flip = not self.flip
        elif ch == "b":
            self.both_sides = not self.both_sides
        elif ch == "p" and self.fl_mode == "DRAFT":
            self.fl_pull = PULLS[(PULLS.index(self.fl_pull) + 1) % len(PULLS)]
        elif ch == "h" and self.fl_mode == "DRAFT":
            self.measure = "HEIGHT" if self.measure == "RULING" else "RULING"
        elif ch == "c":
            self.fl_corner = CORNERS[(CORNERS.index(self.fl_corner) + 1) % len(CORNERS)]
        else:
            return False
        return True

    def drag_rows(self):
        rows = [("Mode", [hints.cycle("M", ["flange", "draft"], self.fl_mode.lower()),
                          hints.key("Tab", "length / angle")]),
                ("Side", [hints.key("F", "flip side"), hints.toggle("B", "both sides (curves)", self.both_sides)]),
                ("Corners", [hints.cycle("C", ["mitre", "blend", "open"], self.fl_corner.lower())])]
        if self.fl_mode == "DRAFT":
            rows.append(("Pull", [hints.cycle("P", ["Z", "X", "Y", "work plane", "view"],
                                              {"WORKPLANE": "work plane", "VIEW": "view"}.get(self.fl_pull, self.fl_pull)),
                                  hints.toggle("H", "height", self.measure == "HEIGHT")]))
        return rows

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.row().prop(self, "fl_mode", expand=True)
        lay.prop(self, "length")
        if self.fl_mode == "FLANGE":
            lay.prop(self, "angle")
        else:
            lay.prop(self, "draft")
            lay.prop(self, "fl_pull")
            lay.prop(self, "pull_flip")
            lay.row().prop(self, "measure", expand=True)
        lay.prop(self, "flip")
        lay.prop(self, "both_sides")
        lay.row().prop(self, "fl_corner", expand=True)
        lay.prop(self, "tolerance")

    def execute(self, context):
        try:
            ins = self._inputs(context)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if not ins:
            self.report({'WARNING'}, "Pick free surface edges, or select curves on a surface")
            return {'CANCELLED'}
        p = self._p()
        name = "Flange" if self.fl_mode == "FLANGE" else "Draft"
        try:
            obj = _make_result(context, name, "flange", ins, p)
        except Exception as exc:
            self.report({'ERROR'}, f"{name}: {exc}")
            return {'CANCELLED'}
        _select_only(context, [obj])
        info = dict(LAST_INFO)
        warn = warnings(info)
        for w in warn:
            self.report({'WARNING'}, w)
        if not warn:
            self.report({'INFO'}, report_text(info, self.fl_mode))
        return {'FINISHED'}


classes = (NPROTO_OT_flange,)
