"""Detach at Isoparm: cut a NURBS surface along isoparms (constant u and / or
constant v lines) into surfaces of their own -- Alias' Detach, Rhino's Split
at Isocurve + Shrink.

Exact: the cut knots are inserted until the surface breaks there (Boehm, in
homogeneous space, so rational surfaces stay exact) and the CV net is sliced
at the breaks. Every piece is the original surface over its sub-domain: same
degree, same weights, its own CVs. A closed direction (a revolve, a tube) is
first opened: one cut there moves the seam to the cut and leaves one piece;
more cuts give that many pieces (a cut at the seam itself counts).

A trimmed surface: each piece's surface is cut the same way and the trim is
clipped to it -- every trim edge crossing a cut line cut exactly there, the
cut line (an exact isoparm) the new edge. A piece the trim does not cut into
comes out untrimmed, one it does not reach into is left out, and a trimmed
piece's Untrim gives the cut-down surface, not the whole source.

The pieces are a history feature ("iso_detach", one per piece) when
Construction History is on: the source is kept hidden and each piece follows
it. Cut positions are fractions of the source's domain, so they hold when the
source is rebuilt with other knots."""

import json

import bpy
import numpy as np
from bpy.props import EnumProperty, StringProperty
from mathutils import Vector

from . import draw, features, geom, hints, store

EPS = 1e-9          # cut positions closer than this (fraction of the domain) are the same
SNAP_PX = 12        # a cut within this many pixels of a span line, edge middle or CV lands on it
SNAP_LABELS = {"mid": "Mid", "cv": "CV"}                      # beside the snap dot
SNAP_STATE = {"span": "span line", "mid": "edge middle", "cv": "CV"}   # in the header
COL_U, COL_V = 0, 3  # pick colours (draw.COL_PICKS): constant-u cuts cyan, constant-v amber


def _k():
    from .kernel import load
    m = load()
    return m, m.blender_ext


# --------------------------------------------------------------------------
# Exact splitting of a homogeneous CV net (axis 0 is the direction cut)
# --------------------------------------------------------------------------

def net(surf):
    """(H (nu, nv, 4) homogeneous CVs, knots_u, knots_v, degree_u, degree_v, rational)."""
    _, ext = _k()
    d = ext.surface_cvs(surf)
    nu, nv = int(d["count_u"]), int(d["count_v"])
    P = np.array(d["points"], float).reshape(nu, nv, 3)
    W = np.array(d["weights"], float).reshape(nu, nv) if d.get("weights") else np.ones((nu, nv))
    H = np.concatenate([P * W[..., None], W[..., None]], axis=2)
    return (H, [float(k) for k in d["knots_u"]], [float(k) for k in d["knots_v"]],
            int(d["degree_u"]), int(d["degree_v"]), bool(d.get("weights")))


def _mult(U, t, tol):
    return sum(1 for k in U if abs(k - t) <= tol)


def insert(G, U, p, t, times):
    """Boehm knot insertion of t, `times` times, along axis 0 of G."""
    G = np.asarray(G, float)
    U = list(U)
    for _ in range(times):
        k = max(i for i in range(len(U) - 1) if U[i] <= t < U[i + 1])
        n = G.shape[0]
        Q = np.empty((n + 1,) + G.shape[1:])
        Q[:k - p + 1] = G[:k - p + 1]
        Q[k + 1:] = G[k:]
        for i in range(k - p + 1, k + 1):
            den = U[i + p] - U[i]
            a = (t - U[i]) / den if den else 0.0
            Q[i] = (1.0 - a) * G[i - 1] + a * G[i]
        G, U = Q, U[:k + 1] + [t] + U[k + 1:]
    return G, U


def split(G, U, p, t, tol):
    """The curve rows G (knots U, degree p) cut at t: (left, right), each (G, U),
    clamped at the cut. t is snapped onto a knot within tol."""
    near = [k for k in U if abs(k - t) <= tol]
    if near:
        t = near[0]
    s = _mult(U, t, tol)
    G, U = insert(G, U, p, t, max(0, p - s))
    U = [t if abs(k - t) <= tol else k for k in U]
    k = max(i for i, x in enumerate(U) if x == t)
    L = (G[:k - p + 1], U[:k + 1] + [t])
    R = (G[k - p:], [t] + U[k - p + 1:])
    return L, R


def clamp(G, U, p, tol):
    """A clamped copy (end knots of multiplicity p + 1) of an unclamped (periodic) row set."""
    a, b = U[p], U[len(U) - p - 1]
    if any(abs(x - a) > tol for x in U[:p + 1]):
        G, U = split(G, U, p, a, tol)[1]
    if any(abs(x - b) > tol for x in U[-p - 1:]):
        G, U = split(G, U, p, b, tol)[0]
    return G, U


def _join_wrap(R, L, p, tol):
    """Piece R (it ends at the seam) and piece L (it starts there) as one:
    the seam becomes an inner knot of multiplicity p. Exact when the seam
    rows are the same homogeneous points up to one factor (rescaled here)."""
    GR, UR = R
    GL, UL = L
    a, b = GR[-1], GL[0]
    w = a[..., 3] / b[..., 3]
    if np.ptp(w) > 1e-9 * abs(w).max():
        raise ValueError("the seam's weights differ along it: it can't be opened exactly")
    GL = GL * w.mean()
    gap = np.abs(GR[-1] - GL[0]).max()
    if gap > 1e-7 * (np.abs(GR).max() or 1.0):
        raise ValueError("the surface is not closed in this direction")
    G = np.concatenate([GR[:-1], [0.5 * (GR[-1] + GL[0])], GL[1:]], axis=0)
    off = UR[-1] - UL[0]
    U = UR[:-(p + 1)] + [UR[-1]] * p + [x + off for x in UL[p + 1:]]
    return G, U


def cut_rows(G, U, p, fracs, closed):
    """Pieces [(G, U)] of the rows G cut at domain fractions `fracs` (a closed
    direction: opened first, the piece across the seam joined back)."""
    tol = 1e-12 * ((U[-1] - U[0]) or 1.0)
    G, U = clamp(G, U, p, tol)
    a, b = U[0], U[-1]
    tol = 1e-12 * (b - a)
    fr = sorted({round(float(f), 12) for f in fracs})
    seam = closed and any(f <= EPS or f >= 1.0 - EPS for f in fr)
    ts = [a + f * (b - a) for f in fr if EPS < f < 1.0 - EPS]
    if not ts and not seam:
        return [(G, U)]
    pieces, rest = [], (G, U)
    for t in ts:
        left, rest = split(rest[0], rest[1], p, t, tol)
        pieces.append(left)
    pieces.append(rest)
    if closed and not seam:
        pieces = [_join_wrap(pieces[-1], pieces[0], p, tol)] + pieces[1:-1]
    return pieces


def _normalized(U):
    a, b = U[0], U[-1]
    return [(x - a) / (b - a) for x in U]


def detach(surf, cuts_u=(), cuts_v=(), closed=None, clamp_all=False):
    """The untrimmed surfaces surf is cut into along constant-u lines at
    fractions cuts_u and constant-v lines at cuts_v of its domain, in grid
    order (u outer, v inner). Each piece's cut directions run over 0..1; an
    uncut direction keeps its knots (clamp_all: clamped, same domain)."""
    _, ext = _k()
    H, Ku, Kv, pu, pv, rational = net(surf)
    if closed is None:
        closed = store._periodic(surf)
    cu, cv = bool(cuts_u), bool(cuts_v)
    cols = cut_rows(H, Ku, pu, cuts_u, closed[0]) if cu or clamp_all else [(H, Ku)]
    out = []
    for G, U in cols:
        rows = (cut_rows(G.transpose(1, 0, 2), Kv, pv, cuts_v, closed[1]) if cv or clamp_all
                else [(G.transpose(1, 0, 2), Kv)])
        for Gv, V in rows:
            A = Gv.transpose(1, 0, 2)
            w = A[..., 3]
            pts = [tuple(map(float, x)) for x in (A[..., :3] / w[..., None]).reshape(-1, 3)]
            ku = _normalized(U) if cu else U
            kv = _normalized(V) if cv else V
            wl = [float(x) for x in w.reshape(-1)] if rational or np.abs(w - 1.0).max() > 1e-12 else None
            out.append(ext.surface_from_cvs(pu, pv, pts, ku, kv, wl))
    return out


def piece_domains(cuts_u, cuts_v, closed):
    """[(fu0, fu1, fv0, fv1)] the source-domain fractions each piece covers,
    in detach() order (a piece across a seam has fu1 > 1: it wraps)."""
    def spans(fr, cl):
        ts = sorted({round(float(f), 12) for f in fr if EPS < f < 1.0 - EPS})
        seam = cl and any(f <= EPS or f >= 1.0 - EPS for f in fr)
        edges = [0.0] + ts + [1.0]
        sp = list(zip(edges[:-1], edges[1:]))
        if fr and cl and not seam:
            sp = [(ts[-1], 1.0 + ts[0])] + sp[1:-1]
        return sp
    return [(a, b, c, d) for a, b in spans(cuts_u, closed[0]) for c, d in spans(cuts_v, closed[1])]


# --------------------------------------------------------------------------
# Trimmed surfaces: each piece's carrier is cut exactly as above, and the
# face's trim is clipped to the piece -- every trim edge that crosses a cut
# line cut exactly there, the cut line itself (an exact isoparm of the
# carrier) the new edge -- then the carrier is trimmed with the clipped loops.
# A piece the trim does not reach into is left out; one the trim does not cut
# into is the carrier itself, untrimmed.
# --------------------------------------------------------------------------

def copy(surf):
    """A fresh copy of an untrimmed surface (the kernel takes ownership of a
    surface it trims or wraps into a Brep)."""
    _, ext = _k()
    d = ext.surface_cvs(surf)
    return ext.surface_from_cvs(int(d["degree_u"]), int(d["degree_v"]), [tuple(map(float, p)) for p in d["points"]],
                                [float(k) for k in d["knots_u"]], [float(k) for k in d["knots_v"]],
                                [float(w) for w in d["weights"]] if d.get("weights") else None)


def _frac(sm, surf, q, dom):
    """Source-domain fractions (fu, fv) of point q on surf."""
    from .kernel.trimext import _uv
    u, v = _uv(sm, surf, q)
    (u0, u1), (v0, v1) = dom
    return np.array([(u - u0) / (u1 - u0), (v - v0) / (v1 - v0)])


def _cycles(face, size):
    """The face's boundary as closed 3D cycles [(edge, forward)] in loop order.
    Seam edges -- used twice in one loop, where a closed carrier wraps -- are
    left out: a cut piece's carrier is open there, or the seam lies inside it."""
    tol = 1e-6 * size
    out = []
    for lp in face.boundary_loops():
        lp = [(e, fw) for e, fw in lp if sum(1 for x, _ in lp if x == e) == 1]
        if not lp:
            continue
        ends = []
        for e, fw in lp:
            t0, t1 = e.parameter_range()
            a, b = np.array(e.point_at(t0), float), np.array(e.point_at(t1), float)
            ends.append((a, b) if fw else (b, a))
        used = [False] * len(lp)
        for i0 in range(len(lp)):
            if used[i0]:
                continue
            cyc, i = [], i0
            while True:
                used[i] = True
                cyc.append(lp[i])
                end = ends[i][1]
                if np.linalg.norm(end - ends[i0][0]) <= tol:
                    break
                order = list(range(i + 1, len(lp))) + list(range(i))
                i = next((j for j in order if not used[j] and np.linalg.norm(ends[j][0] - end) <= tol), None)
                if i is None:
                    raise ValueError("a trim loop of the surface does not close")
            out.append(cyc)
    return out


def _track(sm, surf, cyc, dom, n=33):
    """Per edge of a cycle: (edge parameters in loop order, source fractions there)."""
    out = []
    for e, fw in cyc:
        t0, t1 = e.parameter_range()
        ts = np.linspace(t0, t1, n) if fw else np.linspace(t1, t0, n)
        out.append((ts, [_frac(sm, surf, e.point_at(float(t)), dom) for t in ts]))
    return out


def _circ(x):
    return abs((x + 0.5) % 1.0 - 0.5)


class _Box:
    """One piece's sub-domain of the source (fractions; a piece across a
    closed direction's seam wraps). Its sides that are cut lines bound the
    trim; local coordinates run 0..1 over the piece in each cut direction.
    The cut lines form chains the clipped trim runs along between crossings:
    the rectangle "R" (perimeter coordinate 0..4, counter-clockwise from the
    corner (0, 0)) or, in a closed direction left uncut (a tube cut only
    across), the circles V0 / V1 (or U0 / U1), each 0..1 the way the trim
    runs along it."""

    def __init__(self, dom, cut, closed):
        self.lo, self.hi = (dom[0], dom[2]), (dom[1], dom[3])
        self.cut, self.closed = cut, closed
        self.ring = tuple(closed[i] and not cut[i] for i in (0, 1))
        self.sides = set()
        for i, d in enumerate("UV"):
            if cut[i]:
                if closed[i] or self.lo[i] > EPS:
                    self.sides.add(d + "0")
                if closed[i] or self.hi[i] < 1.0 - EPS:
                    self.sides.add(d + "1")

    def local(self, f):
        out = []
        for i in (0, 1):
            if not self.cut[i]:
                out.append(float(f[i]))
                continue
            x = f[i] - self.lo[i]
            if self.closed[i]:
                x %= 1.0
            out.append(x / (self.hi[i] - self.lo[i]))
        return out

    def source(self, L):
        """Source fractions of local coordinates L."""
        out = []
        for i in (0, 1):
            if not self.cut[i]:
                out.append(L[i])
                continue
            x = self.lo[i] + L[i] * (self.hi[i] - self.lo[i])
            out.append(x % 1.0 if self.closed[i] else x)
        return out

    def kept(self, f):
        L = self.local(f)
        for s in self.sides:
            x = L[0 if s[0] == "U" else 1]
            if (s[1] == "0" and not x > 0.0) or (s[1] == "1" and not x < 1.0):
                return False
        return True

    def side_of(self, line, L):
        """The side of the piece the cut line `line` (direction, fraction) is,
        L: local coordinates of a point inside the piece next to the crossing
        (in a closed direction cut once, both sides are that one line)."""
        i, c = line
        d = "UV"[i]
        near = (lambda x, y: _circ(x - y) <= 1e-12) if self.closed[i] else (lambda x, y: abs(x - y) <= 1e-12)
        cand = [d + k for k, x in (("0", self.lo[i]), ("1", self.hi[i])) if d + k in self.sides and near(c, x)]
        if not cand:
            return None
        if len(cand) == 2:
            return cand[0] if L[i] < 0.5 else cand[1]
        return cand[0]

    def on_side(self, s, L):
        L = list(L)
        L[0 if s[0] == "U" else 1] = 0.0 if s[1] == "0" else 1.0
        return L

    def chain(self, s, L):
        """(chain, coordinate along it) of local point L on side s."""
        u, v = L
        if self.ring[0]:
            return s, (u % 1.0) if s == "V0" else ((1.0 - u) % 1.0)
        if self.ring[1]:
            return s, ((1.0 - v) % 1.0) if s == "U0" else (v % 1.0)
        return "R", {"V0": u, "U1": 1.0 + v, "V1": 3.0 - u, "U0": 4.0 - v}[s] % 4.0

    @staticmethod
    def period(ch):
        return 4.0 if ch == "R" else 1.0

    def closed_chains(self):
        if self.ring[0] or self.ring[1]:
            return sorted(self.sides)
        return ["R"] if len(self.sides) == 4 else []

    def chain_point(self, ch):
        """Local coordinates of a point on chain ch (to tell whether it runs inside the face)."""
        return {"R": (0.5, 0.0), "V0": (0.5, 0.0), "V1": (0.5, 1.0), "U0": (0.0, 0.5), "U1": (1.0, 0.5)}[ch]

    def path(self, carrier, ch, s0, length):
        """Exact iso curves of the carrier along chain ch from s0 over length
        (as recipes: the kernel consumes the curves it trims with, so every
        attempt builds its own), and local points along them."""
        from .kernel.trimext import _iso_piece
        (cu0, cu1), (cv0, cv1) = carrier.u_domain(), carrier.v_domain()
        P = self.period(ch)
        curves, pts = [], []
        s, end = s0, s0 + length
        while s < end - 1e-12:
            k = np.floor(s + 1e-12)
            nxt = min(end, k + 1.0)
            a, b = s - k, nxt - k
            side = ch if ch != "R" else ("V0", "U1", "V1", "U0")[int(k % P)]
            if side not in self.sides:
                raise ValueError("the clipped trim runs off the surface's edge")
            if ch == "R":
                ends = {"V0": ((a, 0.0), (b, 0.0)), "U1": ((1.0, a), (1.0, b)),
                        "V1": ((1.0 - a, 1.0), (1.0 - b, 1.0)), "U0": ((0.0, 1.0 - a), (0.0, 1.0 - b))}[side]
            elif ch in ("V0", "V1"):
                y = 0.0 if ch == "V0" else 1.0
                ends = ((a, y), (b, y)) if ch == "V0" else ((1.0 - a, y), (1.0 - b, y))
            else:
                x = 0.0 if ch == "U0" else 1.0
                ends = ((x, 1.0 - a), (x, 1.0 - b)) if ch == "U0" else ((x, a), (x, b))
            (ua, va), (ub, vb) = ends
            if side[0] == "V":
                args = (False, cv0 + va * (cv1 - cv0), cu0 + ua * (cu1 - cu0), cu0 + ub * (cu1 - cu0))
            else:
                args = (True, cu0 + ua * (cu1 - cu0), cv0 + va * (cv1 - cv0), cv0 + vb * (cv1 - cv0))
            curves.append(lambda a=args: _iso_piece(carrier, *a))
            pts += [(ua + (ub - ua) * t, va + (vb - va) * t) for t in np.linspace(0.0, 1.0, 5)[:-1]]
            s = nxt
        return curves, pts


def _segments(sm, surf, cyc, track, cuts, closed, dom):
    """A cycle cut at every crossing with a cut line: [(entry, exit, pieces,
    probe)] in loop order. entry / exit: (line (direction, fraction),
    fractions at the crossing), None for a cycle no line crosses (then one
    segment, the whole cycle); pieces: [(edge, forward, t0, t1, fractions
    along it)]; probe: fractions of a point inside the segment (which piece of
    the detach it lies in)."""
    def g_of(f, i, c):
        g = f[i] - c
        return (g + 0.5) % 1.0 - 0.5 if closed[i] else g

    def bisect(e, ta, tb, i, c):
        pa = g_of(_frac(sm, surf, e.point_at(float(ta)), dom), i, c) >= 0.0
        for _ in range(52):
            tm = 0.5 * (ta + tb)
            if (g_of(_frac(sm, surf, e.point_at(float(tm)), dom), i, c) >= 0.0) == pa:
                ta = tm
            else:
                tb = tm
        t = 0.5 * (ta + tb)
        f = _frac(sm, surf, e.point_at(float(t)), dom)
        f[i] = c % 1.0 if closed[i] else c
        return t, f

    segs, pieces, entry = [], [], None
    for (e, fw), (ts, F) in zip(cyc, track):
        G = np.array(F)
        evs = []
        for i in (0, 1):
            for c in cuts[i]:
                g = np.array([g_of(f, i, c) for f in G])
                pos = g >= 0.0
                for k in range(len(ts) - 1):
                    if pos[k] != pos[k + 1] and abs(g[k] - g[k + 1]) < 0.5:
                        t, f = bisect(e, ts[k], ts[k + 1], i, c)
                        evs.append((k, abs(t - ts[k]), t, (i, c), f))
        evs.sort(key=lambda x: (x[0], x[1]))
        t0, pts, ki = ts[0], [G[0]], 0
        for k, _, t, line, f in evs:
            while ki < k:
                ki += 1
                pts.append(G[ki])
            pieces.append((e, fw, t0, t, pts + [f]))
            segs.append([entry, (line, f), pieces])
            pieces, entry, t0, pts = [], (line, f), t, [f]
        while ki < len(ts) - 1:
            ki += 1
            pts.append(G[ki])
        pieces.append((e, fw, t0, ts[-1], pts))
    if not segs:
        segs = [[None, None, pieces]]
    else:
        segs[0] = [entry, segs[0][1], pieces + segs[0][2]]
    out = []
    for ent, ex, pcs in segs:
        e, fw, t0, t1, _ = max(pcs, key=lambda x: abs(x[3] - x[2]))
        out.append((ent, ex, pcs, _frac(sm, surf, e.point_at(float(0.5 * (t0 + t1))), dom)))
    return out


def _signed_area(P):
    P = np.asarray(P, float)
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _inside(pt, P):
    from .kernel.trimext import _inside_poly
    return _inside_poly(pt, np.asarray(P, float))


FIT_TOLS = (1e-7, 1e-5, 1e-4)    # UV trim curve fit tolerances tried in turn (x the surface size)


def _trim(sm, ext, carrier, groups, size):
    """The carrier trimmed with each group of loops ([curve recipes] per loop,
    the outer first): a Brep, one face per group. The UV trim curves are
    fitted as tightly as the kernel manages (an edge's UV curve is what the
    face's area and later edits go by). Each piece's own tolerance follows
    the size (surfmap.BREP_TOL), not the kernel's default 0.01 model units:
    joined back, a strip narrower than about twice that had its corners
    merged and nothing stitched."""
    from .surfmap import BREP_TOL
    out = None
    for loops in groups:
        sizes = [len(c) for c in loops]
        made = None
        for tol in FIT_TOLS + (None,):
            try:
                curves = [make() for c in loops for make in c]
                if tol is not None:
                    b = ext.trim_surface_with_3d_curves_tol(copy(carrier), sizes, curves, tol * size,
                                                            max(BREP_TOL, 10.0 * tol) * size)
                else:
                    b = sm.trim_surface_with_3d_curves(copy(carrier), sizes, curves)
            except Exception:
                b = None
            if b is not None and b.face_count() == 1 and b.area(1e-6) > 0.0:
                made = b
                break
        if made is None:
            raise ValueError("a piece's trim could not be built")
        if out is None:
            out = made
        else:
            ext.append_brep(out, made)
    return out


def _region(sm, ext, face, dom, carrier, box, segments, size):
    """The face clipped to one piece's box, on that piece's carrier: a Brep,
    the carrier itself (the trim does not cut into the piece) or None (the
    trim does not reach into it)."""
    from .kernel.trimext import _edge_curve, _piece
    (u0, u1), (v0, v1) = dom
    arcs, loops = [], []      # arcs: (entry (chain, s), exit (chain, s), [(e, fw, t0, t1, _, local points)])
    for segs in segments:
        for ent, ex, pcs, probe in segs:
            if not box.kept(probe):
                continue
            loc = [(e, fw, t0, t1, None, [box.local(f) for f in pts]) for e, fw, t0, t1, pts in pcs]
            if ent is None:             # a whole cycle inside the piece
                loops.append(([lambda e=e, fw=fw: _edge_curve(e, fw) for e, fw, _, _, _ in pcs],
                              [p for x in loc for p in x[5]]))
                continue
            first, last = loc[0][5], loc[-1][5]
            near_in = first[1] if len(first) > 2 else box.local(probe)
            near_out = last[-2] if len(last) > 2 else box.local(probe)
            sd_in, sd_out = box.side_of(ent[0], near_in), box.side_of(ex[0], near_out)
            if sd_in is None or sd_out is None:
                raise ValueError("a trim edge crosses a cut line where it should not")
            L_in, L_out = box.on_side(sd_in, box.local(ent[1])), box.on_side(sd_out, box.local(ex[1]))
            first[0], last[-1] = L_in, L_out
            arcs.append((box.chain(sd_in, L_in), box.chain(sd_out, L_out), loc))
    # the arcs joined into loops along the cut lines (counter-clockwise: the
    # region on the left, like the face's own outer loop)
    starts = {}
    for i, (ent, _, _) in enumerate(arcs):
        starts.setdefault(ent[0], []).append((ent[1], i))
    used = set()
    for i0 in range(len(arcs)):
        if i0 in used:
            continue
        curves, poly, i = [], [], i0
        while i not in used:
            used.add(i)
            _, (ch, s_out), pcs = arcs[i]
            for e, fw, t0, t1, _, pts in pcs:
                if abs(t1 - t0) > 1e-12 * (abs(t0) + abs(t1) + 1.0):
                    curves.append(lambda a=(e, fw, t0, t1): _piece(*a))
                poly += pts
            P = box.period(ch)
            cand = starts.get(ch, [])
            if not cand:
                raise ValueError("the clipped trim does not close")
            s_in, j = min(cand, key=lambda x: (x[0] - s_out + 1e-12) % P)
            c2, p2 = box.path(carrier, ch, s_out, (s_in - s_out + 1e-12) % P - 1e-12)
            curves += c2
            poly += p2
            i = j
        if i != i0:
            raise ValueError("the clipped trim does not close")
        loops.append((curves, poly))
    hit = {a[0][0] for a in arcs} | {a[1][0] for a in arcs}
    for ch in box.closed_chains():
        if ch in hit:
            continue
        fu, fv = box.source(box.chain_point(ch))
        try:
            inside = str(face.classify_uv(u0 + fu * (u1 - u0), v0 + fv * (v1 - v0))).lower().endswith("inside")
        except Exception:
            inside = False
        if inside:
            loops.append(box.path(carrier, ch, 0.0, box.period(ch)))
    if not loops:
        return None
    if box.ring[0] or box.ring[1]:
        groups = [[c for c, _ in loops]]           # a closed carrier: its loops go round it, all in one face
    else:
        # nesting: a loop inside an even number of others bounds a face, the
        # loops just inside it are its holes
        n = len(loops)
        A = [abs(_signed_area(p)) for _, p in loops]
        ins = [[j for j in range(n) if j != i and A[j] > A[i] and _inside(loops[i][1][0], loops[j][1])]
               for i in range(n)]
        groups = []
        for i in range(n):
            if len(ins[i]) % 2:
                continue
            holes = [j for j in range(n) if len(ins[j]) == len(ins[i]) + 1 and i in ins[j]]
            groups.append([loops[i][0]] + [loops[j][0] for j in holes])
    if len(groups) == 1 and len(groups[0]) == 1 and len(loops) == 1:
        whole = store.to_brep(copy(carrier)).area(1e-6)
        b = _trim(sm, ext, carrier, groups, size)
        if abs(b.area(1e-6) - whole) <= 1e-5 * whole:
            return carrier
        return b
    return _trim(sm, ext, carrier, groups, size)


def detach_trimmed(brep, cuts_u=(), cuts_v=()):
    """The one-face trimmed brep cut along isoparms of its carrier, in
    piece_domains() order: [(geometry or None, carrier)] -- geometry a Brep
    (the trimmed piece), the carrier itself (an untrimmed piece) or None (the
    trim does not reach into the piece)."""
    sm, ext = _k()
    from .kernel.trimext import surface_size
    if brep.face_count() != 1:
        raise ValueError("Detach at Isoparm needs a single surface (Unjoin first)")
    face = brep.faces()[0]
    surf = face.surface()
    closed = store._periodic(surf)
    dom = (surf.u_domain(), surf.v_domain())
    carriers = detach(surf, cuts_u, cuts_v, closed, clamp_all=True)
    doms = piece_domains(cuts_u, cuts_v, closed)
    size = surface_size(surf)
    cycles = _cycles(face, size)
    cuts = ([float(f) for f in cuts_u], [float(f) for f in cuts_v])
    segments = [_segments(sm, surf, c, _track(sm, surf, c, dom), cuts, closed, dom) for c in cycles]
    out = []
    for carrier, d in zip(carriers, doms):
        box = _Box(d, (bool(cuts_u), bool(cuts_v)), closed)
        out.append((_region(sm, ext, face, dom, carrier, box, segments, size), carrier))
    total = face.area(1e-6)
    got = sum(store.to_brep(copy(g)).area(1e-6) if not isinstance(g, sm.Brep) else g.area(1e-6)
              for g, _ in out if g is not None)
    if abs(got - total) > 1e-3 * total:
        raise ValueError("the pieces do not add up to the surface (a cut along a trim edge, or through one of "
                         "its corners? move it a little)")
    return out


# --------------------------------------------------------------------------
# History feature: one piece of the source's detach
# --------------------------------------------------------------------------

_trim_cache = {}        # source brep + cuts -> the trimmed detach's pieces as text (one detach serves all pieces)


def detachable(obj):
    """One face (untrimmed or trimmed): what Detach at Isoparm cuts."""
    try:
        return store.is_nurbs(obj) and (store.is_natural(obj) or store.get_brep(obj).face_count() == 1)
    except Exception:
        return False


def _trimmed(obj, cuts_u, cuts_v):
    """detach_trimmed() of obj as [None | ("T" | "N", brep text)], cached."""
    import hashlib
    sm, ext = _k()
    key = hashlib.sha1((obj.get(store.PROP_BREP, "") + json.dumps([list(cuts_u), list(cuts_v)])).encode()).hexdigest()
    hit = _trim_cache.get(key)
    if hit is None:
        hit = []
        for g, _ in detach_trimmed(store.get_brep(obj), cuts_u, cuts_v):
            if g is None:
                hit.append(None)
            elif isinstance(g, sm.Brep):
                hit.append(("T", store.brep_to_text(g)))
            else:
                hit.append(("N", store.brep_to_text(ext.brep_from_surface(g))))
        if len(_trim_cache) > 8:
            _trim_cache.clear()
        _trim_cache[key] = hit
    return hit


def _geometry(entry):
    """A fresh kernel object of a cached piece: a Surface (untrimmed) or a Brep."""
    kind, text = entry
    b = store.brep_from_text(text)
    return copy(b.faces()[0].surface()) if kind == "N" else b


def pieces_of(obj, cuts_u, cuts_v):
    """[geometry or None] of obj detached at the cuts (piece_domains order):
    Surfaces for an untrimmed source; Brep / Surface / None for a trimmed one."""
    if store.is_natural(obj):
        return detach(store.get_surfaces(obj)[0], cuts_u, cuts_v)
    if store.get_brep(obj).face_count() != 1:
        raise ValueError(f"{obj.name} has several faces: Detach at Isoparm cuts a single surface (Unjoin first)")
    return [None if e is None else _geometry(e) for e in _trimmed(obj, cuts_u, cuts_v)]


def piece_carrier_text(src, p):
    """Brep text of the untrimmed surface a piece was cut from (its Untrim), None if gone."""
    _, ext = _k()
    try:
        surf = store.get_surfaces(src)[0]
        carriers = detach(surf, p.get("cuts_u", []), p.get("cuts_v", []), clamp_all=not store.is_natural(src))
        i = int(p.get("piece", 0))
        return store.brep_to_text(ext.brep_from_surface(carriers[i])) if 0 <= i < len(carriers) else None
    except Exception:
        return None


@features.builder("iso_detach")
def build_iso_detach(inputs, p):
    obj = inputs[0][0]
    pieces = pieces_of(obj, p.get("cuts_u", []), p.get("cuts_v", []))
    i = int(p.get("piece", 0))
    if not 0 <= i < len(pieces):
        raise RuntimeError(f"{obj.name} now detaches into {len(pieces)} pieces, not {i + 1}")
    if pieces[i] is None:
        raise RuntimeError(f"the trim of {obj.name} no longer reaches into this piece")
    return pieces[i]


def apply(context, obj, cuts_u, cuts_v, op=None):
    """Detach obj (one face, untrimmed or trimmed) at the cuts. Returns the
    piece objects (with history: new objects following the hidden source;
    without: the first piece stays obj itself). A trimmed piece's Untrim
    gives the carrier it was cut out of."""
    sm, ext = _k()
    natural = store.is_natural(obj)
    surf = store.get_surfaces(obj)[0]
    closed = store._periodic(surf)
    geoms = pieces_of(obj, cuts_u, cuts_v)
    real = [i for i, g in enumerate(geoms) if g is not None]
    opened = (cuts_u and closed[0]) or (cuts_v and closed[1])
    if len(real) < 2 and not opened:
        raise ValueError("the cuts don't divide the surface" + ("" if natural else " (they miss its trimmed part)"))
    carriers = None if natural else detach(surf, cuts_u, cuts_v, clamp_all=True)

    def untrim_base(o, i):
        for key in (store.PROP_UNTRIM, store.PROP_UNTRIM + "_natural"):
            if key in o:
                del o[key]
        if carriers is not None and isinstance(geoms[i], sm.Brep):
            o[store.PROP_UNTRIM] = store.brep_to_text(ext.brep_from_surface(copy(carriers[i])))
            o[store.PROP_UNTRIM + "_natural"] = 1

    M = geom.world_matrix(obj)
    keep = features.history_enabled(context)
    made = []
    if not keep:
        if features.has_history(obj):
            features.clear_history(obj)
            if op is not None:
                op.report({'INFO'}, f"{obj.name}: construction history removed")
        store.set_geometry(obj, geoms[real[0]], context)
        untrim_base(obj, real[0])
        made.append(obj)
    params = {"cuts_u": [float(f) for f in cuts_u], "cuts_v": [float(f) for f in cuts_v]}
    for i in real:
        if not keep and i == real[0]:
            continue
        o = store.new_object(context, f"{obj.name}_detach{i + 1}", geoms[i], M)
        for m in obj.data.materials:
            o.data.materials.append(m)
        if keep:
            features.record(o, "iso_detach", [(obj, "SURF")], {**params, "piece": i})
        else:
            untrim_base(o, i)
        made.append(o)
    if keep:
        features.keep_inputs(context, [obj])
    return made


# --------------------------------------------------------------------------
# Picking isoparms
# --------------------------------------------------------------------------

def _dir_fracs(surf, d):
    """Interior span-line fractions of direction d ('U' / 'V'), plus the seam (0) of a closed one."""
    _, ext = _k()
    c = ext.surface_cvs(surf)
    U = [float(k) for k in (c["knots_u"] if d == "U" else c["knots_v"])]
    p = int(c["degree_u"] if d == "U" else c["degree_v"])
    a, b = U[p], U[len(U) - p - 1]
    out = sorted({round((k - a) / (b - a), 12) for k in U if a < k < b})
    if store._periodic(surf)[0 if d == "U" else 1]:
        out = [0.0] + out
    return out


def iso_points(surf, d, f, n=64):
    """Local-space points of the isoparm at fraction f of direction d."""
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    ts = np.linspace(0.0, 1.0, n)
    if d == "U":
        u = u0 + (u1 - u0) * min(max(f, 0.0), 1.0)
        return np.array([surf.evaluate(u, v0 + (v1 - v0) * t) for t in ts], float)
    v = v0 + (v1 - v0) * min(max(f, 0.0), 1.0)
    return np.array([surf.evaluate(u0 + (u1 - u0) * t, v) for t in ts], float)


def iso_runs(surf, face, d, f, n=96):
    """Local-space polylines of the isoparm at fraction f of direction d where
    it runs on the face (face None: the whole isoparm, one polyline)."""
    P = iso_points(surf, d, f, n)
    if face is None:
        return [P]
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    ts = np.linspace(0.0, 1.0, n)
    g = min(max(f, 0.0), 1.0)
    uv = [(u0 + (u1 - u0) * g, v0 + (v1 - v0) * t) if d == "U" else (u0 + (u1 - u0) * t, v0 + (v1 - v0) * g)
          for t in ts]
    on = []
    for u, v in uv:
        try:
            on.append(not str(face.classify_uv(u, v)).lower().endswith("outside"))
        except Exception:
            on.append(True)
    runs, cur = [], []
    for p, k in zip(P, on):
        if k:
            cur.append(p)
        elif cur:
            runs.append(np.array(cur))
            cur = []
    if cur:
        runs.append(np.array(cur))
    return [r for r in runs if len(r) > 1]


def _greville_fracs(surf, d):
    """Fraction of direction d at each CV row's Greville point (the average of
    its p knots: where that row of CVs pulls hardest), one per row index."""
    _, ext = _k()
    c = ext.surface_cvs(surf)
    U = [float(k) for k in (c["knots_u"] if d == "U" else c["knots_v"])]
    p = int(c["degree_u"] if d == "U" else c["degree_v"])
    n = int(c["count_u"] if d == "U" else c["count_v"])
    a, b = U[p], U[len(U) - p - 1]
    return [(sum(U[i + 1:i + p + 1]) / p - a) / (b - a) for i in range(n)]


def _length_middle(fn, t0, t1, n=129):
    """(point halfway along the curve t -> fn(t) by length, length)."""
    ts = np.linspace(t0, t1, n)
    P = np.array([fn(float(t)) for t in ts], float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    tm = float(np.interp(0.5 * s[-1], s, ts))
    return np.array(fn(tm), float), float(s[-1])


def snap_points(surf, brep, face, d, cvs=True):
    """[(fraction, local point, kind)] a cut in direction d snaps to: the middle
    (by length) of every edge of the face -- of the untrimmed surface's four
    sides when brep is None -- ("mid"), and every CV at its row's Greville
    fraction ("cv"). Fractions at an open direction's ends are left out (that
    is the surface's edge), as are CV rows whose isoparm the trim cuts away."""
    closed = store._periodic(surf)[0 if d == "U" else 1]
    (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
    H = net(surf)[0]
    P = H[..., :3] / H[..., 3:4]
    size = float(np.linalg.norm(P.reshape(-1, 3).max(axis=0) - P.reshape(-1, 3).min(axis=0))) or 1.0
    foot = store._make_foot(surf)
    spans = np.array(_dir_fracs(surf, d) + [0.0, 1.0], float)

    def frac(f):
        k = int(np.argmin(np.abs(spans - f)))
        if abs(spans[k] - f) < 1e-9:             # on a span line (a knot already there): exactly on it
            f = float(spans[k])
        if closed:
            f %= 1.0
            return 0.0 if f > 1.0 - EPS else f
        return f if EPS < f < 1.0 - EPS else None

    out = []
    if brep is None:
        sides = [(lambda t, v=v: surf.evaluate(u0 + (u1 - u0) * t, v), 0.0, 1.0) for v in (v0, v1)] + \
                [(lambda t, u=u: surf.evaluate(u, v0 + (v1 - v0) * t), 0.0, 1.0) for u in (u0, u1)]
    else:
        sides = [(e.point_at, *e.parameter_range()) for e in brep.edges()]
    for fn, t0, t1 in sides:
        M, L = _length_middle(fn, t0, t1)
        if L < 1e-9 * size:                      # a pole
            continue
        u, v = foot(M)
        f = frac((u - u0) / (u1 - u0) if d == "U" else (v - v0) / (v1 - v0))
        if f is not None:
            out.append((f, M, "mid"))
    if cvs:
        on = {}
        for i, g in enumerate(_greville_fracs(surf, d)):
            f = frac(g)
            if f is None:
                continue
            if f not in on:
                on[f] = face is None or bool(iso_runs(surf, face, d, f, 48))
            if on[f]:
                rows = P[i] if d == "U" else P[:, i]
                out += [(f, p, "cv") for p in rows]
    return out


def _screen(context, P):
    """Region pixel coordinates (n, 2) of world points P (n, 3); NaN behind the eye."""
    M = np.array(context.region_data.perspective_matrix, float)
    Q = np.c_[P, np.ones(len(P))] @ M.T
    w = Q[:, 3:4]
    with np.errstate(divide='ignore', invalid='ignore'):
        n = np.where(w > 1e-9, Q[:, :2] / w, np.nan)
    r = context.region
    return np.c_[(n[:, 0] + 1.0) * 0.5 * r.width, (n[:, 1] + 1.0) * 0.5 * r.height]


class NPROTO_OT_detach_iso(bpy.types.Operator):
    """Cut a surface along isoparms into surfaces of its own (exact: knot
    insertion). An untrimmed surface gives untrimmed pieces; a trimmed one
    gives each piece its share of the trim on the cut-down surface (one the
    trim does not cut into comes out untrimmed). Move over the surface to see
    the isoparm, click to add a cut (it snaps to span lines, edge middles and
    shown CVs; Shift: free), Tab
    switches U / V, Backspace takes the last cut back, Enter detaches. A
    closed direction opens at a single cut. With history each piece follows
    the source, which is kept hidden"""
    bl_idname = "nproto.detach_iso"
    bl_label = "Detach at Isoparm"
    bl_options = {'REGISTER', 'UNDO'}

    target: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    cuts_u: StringProperty(name="Cuts U", default="[]", description="Constant-u cuts: fractions of the u domain (JSON list)")
    cuts_v: StringProperty(name="Cuts V", default="[]", description="Constant-v cuts: fractions of the v domain (JSON list)")
    iso_dir: EnumProperty(name="Direction", items=[("U", "U", "Constant-u lines"), ("V", "V", "Constant-v lines")],
                          default="U")

    def draw(self, context):
        lay = self.layout
        for d, txt in (("U", self.cuts_u), ("V", self.cuts_v)):
            fr = json.loads(txt or "[]")
            if fr:
                lay.label(text=f"{d}: " + ", ".join(f"{f:.4g}" for f in fr), icon='UV_EDGESEL')

    # ---- run
    def _cuts(self):
        return json.loads(self.cuts_u or "[]"), json.loads(self.cuts_v or "[]")

    def execute(self, context):
        from .ops import _kernel_ok, _select_only
        if not _kernel_ok(self):
            return {'CANCELLED'}
        obj = bpy.data.objects.get(self.target) if self.target else context.active_object
        if obj is None or not store.is_nurbs(obj):
            self.report({'WARNING'}, "Detach at Isoparm: select a NURBS surface")
            return {'CANCELLED'}
        if not detachable(obj):
            self.report({'WARNING'}, f"{obj.name} has several faces: Detach at Isoparm cuts a single surface "
                                     "(Unjoin first)")
            return {'CANCELLED'}
        cu, cv = self._cuts()
        if not cu and not cv:
            self.report({'WARNING'}, "Detach at Isoparm: no cuts")
            return {'CANCELLED'}
        try:
            made = apply(context, obj, cu, cv, self)
        except Exception as exc:
            self.report({'ERROR'}, f"Detach at Isoparm failed: {exc}")
            return {'CANCELLED'}
        _select_only(context, made)
        self.report({'INFO'}, f"Detached {obj.name} into {len(made)} surface(s)")
        return {'FINISHED'}

    # ---- interactive
    def invoke(self, context, event):
        from .ops import _kernel_ok
        if not _kernel_ok(self):
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D' or context.region_data is None:
            return self.execute(context)
        self._obj, self._surf, self._foot = None, None, None
        self._cuts_l = []                 # [(dir, fraction, world polyline)]
        self._hover = None                # (dir, fraction, snap kind or None, world snap point or None)
        self._snap_cache = {}
        a = context.active_object
        self._pre = a is not None and detachable(a) and a.select_get()
        if self._pre:
            self._set_target(a)
        draw.pick_overlay["picked"], draw.pick_overlay["colors"] = [], []
        draw.pick_overlay["hover"] = None
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {'RUNNING_MODAL'}

    def _set_target(self, obj):
        self._obj = obj
        self._surf = store.get_surfaces(obj)[0]
        self._brep = None if store.is_natural(obj) else store.get_brep(obj)
        self._face = None if self._brep is None else self._brep.faces()[0]
        self._foot = store._make_foot(self._surf)
        self._M = np.array(geom.world_matrix(obj), float)
        self._snap_cache = {}

    def _world(self, P):
        return P @ self._M[:3, :3].T + self._M[:3, 3]

    def _snap_lines(self, d):
        """[(fraction, world points)] of direction d's span lines (cached)."""
        if d not in self._snap_cache:
            lines = []
            for f in _dir_fracs(self._surf, d)[:200]:
                runs = iso_runs(self._surf, self._face, d, f, 32 if self._face is None else 48)
                if runs:          # (a span line the trim cuts away entirely does not snap)
                    lines.append((f, self._world(np.concatenate(runs))))
            self._snap_cache[d] = lines
        return self._snap_cache[d]

    def _snap_points(self, context, d):
        """(fractions, world points, kinds) a cut in direction d snaps to: edge
        middles, and the CVs while the net is shown (cached)."""
        cvs = draw.cv_net_shown(context, self._obj)
        key = ("P", d, cvs)
        if key not in self._snap_cache:
            pts = snap_points(self._surf, self._brep, self._face, d, cvs)
            self._snap_cache[key] = (np.array([f for f, _, _ in pts], float),
                                     self._world(np.array([p for _, p, _ in pts], float).reshape(-1, 3)),
                                     [k for _, _, k in pts])
        return self._snap_cache[key]

    def _point_snap(self, context, event, d):
        """(d, fraction, kind, world point) of the edge middle / CV nearest the
        mouse within SNAP_PX, or None."""
        F, W, K = self._snap_points(context, d)
        if not len(F):
            return None
        S = _screen(context, W)
        dist = np.linalg.norm(S - np.array([event.mouse_region_x, event.mouse_region_y], float), axis=1)
        dist[~np.isfinite(dist)] = np.inf
        i = int(np.argmin(dist))
        return (d, float(F[i]), K[i], W[i]) if dist[i] < SNAP_PX else None

    def _pick(self, context, event):
        """(dir, fraction, snap kind or None, world snap point or None) of the
        isoparm under the mouse, or None. An edge middle or CV snaps even with
        the mouse off the surface (a CV often is)."""
        from .ops_trim import _ray_nurbs
        obj, loc, _, _ = _ray_nurbs(context, event)
        if obj is None or (self._obj is not None and obj != self._obj):
            if self._obj is not None and not event.shift:
                return self._point_snap(context, event, self.iso_dir)
            return None
        if self._obj is None:
            if not detachable(obj):
                return ("REFUSED", obj)
            self._set_target(obj)
        q = np.linalg.solve(self._M, np.r_[np.array(loc, float), 1.0])[:3]
        u, v = self._foot(q)
        (u0, u1), (v0, v1) = self._surf.u_domain(), self._surf.v_domain()
        d = self.iso_dir
        f = (u - u0) / (u1 - u0) if d == "U" else (v - v0) / (v1 - v0)
        f = min(max(f, 0.0), 1.0)
        if event.shift:
            return (d, f, None, None)
        snap = self._point_snap(context, event, d)
        if snap is not None:
            return snap
        m = np.array([event.mouse_region_x, event.mouse_region_y], float)
        best = None
        for g, W in self._snap_lines(d):
            S = _screen(context, W)
            dist = np.nanmin(np.linalg.norm(S - m, axis=1)) if np.isfinite(S).any() else np.inf
            if dist < SNAP_PX and (best is None or dist < best[0]):
                best = (dist, g)
        if best is not None:
            return (d, best[1], "span", None)
        return (d, f, None, None)

    def _polylines(self, d, f):
        """World polylines of the isoparm, where it runs on the (trimmed) face."""
        return [[Vector(p) for p in self._world(r)] for r in iso_runs(self._surf, self._face, d, f)]

    def _refresh(self, context):
        draw.pick_overlay["picked"] = [pl for _, _, pls in self._cuts_l for pl in pls]
        draw.pick_overlay["colors"] = [COL_U if d == "U" else COL_V for d, _, pls in self._cuts_l for _ in pls]
        h = self._hover
        draw.pick_overlay["hover"] = (self._polylines(h[0], h[1]) or None) if h and h[0] in "UV" else None
        draw.pick_overlay["refused"] = None
        snap = h[3] if h and h[0] in "UV" else None
        draw.sketch_overlay["snap"] = Vector(tuple(snap)) if snap is not None else None
        draw.sketch_overlay["snap_label"] = SNAP_LABELS.get(h[2]) if snap is not None else None
        if h and h[0] == "REFUSED":
            M = h[1].matrix_world
            pts = []
            for e in store.display(h[1]).get("edges", []):
                pts += [M @ Vector(p) for p in e]
            draw.pick_overlay["refused"] = pts or None
        self._status(context)
        if context.area is not None:
            context.area.tag_redraw()

    def _status(self, context):
        nu = sum(1 for d, _, _ in self._cuts_l if d == "U")
        nv = len(self._cuts_l) - nu
        h = self._hover
        state = [self._obj.name if self._obj is not None else "click a surface",
                 f"cuts U {nu} · V {nv}"]
        if h and h[0] in "UV":
            state.append(f"{h[0]} {h[1]:.4f}" + (f" ({SNAP_STATE[h[2]]})" if h[2] else ""))
        elif h and h[0] == "REFUSED":
            state.append(f"{h[1].name} has several faces: Unjoin first")
        rows = [("Cut", [hints.key("Click", "add a cut"), hints.cycle("Tab", ["U", "V"], self.iso_dir),
                         hints.key("Shift", "no snapping"),
                         hints.key("Backspace", "take the last cut back", on=None if self._cuts_l else False)]),
                ("Detach", [hints.key("Enter", "detach", on=None if self._cuts_l else False), hints.key("Esc", "cancel")])]
        hints.show(context, self.bl_label, state, rows,
                   mouse=[('MOUSE_LMB', "Add Cut"), ('EVENT_TAB', "U / V"), ('EVENT_RETURN', "Detach"),
                          ('EVENT_ESC', "Cancel")])

    def _end(self, context):
        draw.pick_overlay["picked"], draw.pick_overlay["colors"] = [], []
        draw.pick_overlay["hover"] = draw.pick_overlay["refused"] = None
        draw.sketch_overlay["snap"] = draw.sketch_overlay["snap_label"] = None
        hints.clear(context)
        if context.area is not None:
            context.area.header_text_set(None)
            context.area.tag_redraw()

    def modal(self, context, event):
        from .drawtools import PASS_KEYS
        if hints.handle(event):
            return {'RUNNING_MODAL'}
        if event.type in PASS_KEYS:           # in-between moves too: Blender's drags (gizmos) are built from them
            return {'PASS_THROUGH'}
        if event.type in {'MOUSEMOVE', 'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            self._hover = self._pick(context, event)
            self._refresh(context)
        elif event.type in {'TAB', 'U', 'V'} and event.value == 'PRESS':
            self.iso_dir = event.type if event.type in {'U', 'V'} else ("V" if self.iso_dir == "U" else "U")
            self._hover = self._pick(context, event)
            self._refresh(context)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            h = self._pick(context, event)
            if h and h[0] in "UV":
                d, f = h[0], h[1]
                ends = not store._periodic(self._surf)[0 if d == "U" else 1]
                if ends and not (EPS < f < 1.0 - EPS):
                    self.report({'WARNING'}, "That is the surface's edge: nothing to cut there")
                elif any(c[0] == d and abs(c[1] - f) <= 1e-6 for c in self._cuts_l):
                    pass
                else:
                    self._cuts_l.append((d, f, self._polylines(d, f)))
            self._hover = h
            self._refresh(context)
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS' and self._cuts_l:
            self._cuts_l.pop()
            if not self._cuts_l and not self._pre:
                self._obj = None          # no cuts left: another surface may be picked
            self._refresh(context)
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if not self._cuts_l:
                self.report({'WARNING'}, "Click on a surface to add a cut first")
                return {'RUNNING_MODAL'}
            self.target = self._obj.name
            self.cuts_u = json.dumps([f for d, f, _ in self._cuts_l if d == "U"])
            self.cuts_v = json.dumps([f for d, f, _ in self._cuts_l if d == "V"])
            self._end(context)
            return self.execute(context)
        elif event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._end(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}


classes = (NPROTO_OT_detach_iso,)
