"""SubD to BREP at star points: faces of 2 x 2 polynomial pieces joined exactly G1 or G2.

Faces without a star corner stay one bicubic span: the SubD surface exactly.
A face with one becomes 2 x 2 polynomial pieces, C1 across its middle lines:
of degree 6 for G2 (knots [0^7, .5^5, 1^7], 12 x 12 CVs, G2 across the middle
lines too), of degree 5 for G1 (knots [0^6, .5^4, 1^6], 10 x 10 CVs). It is
fitted to the SubD and then moved as little and as smoothly as possible until
it joins its neighbours exactly:

  G1 along every edge half:  f_t + g_s = beta f_u
  G2 along every edge half:  (f_u x f_t) . (g_ss - f_tt + 2 beta f_ut - beta^2 f_uu) = 0

with beta = 2 cos(theta / k) (1 - u)^2 on the half from a star point (it must
vanish with its slope at the edge's middle, where the pieces only join C1:
a linear beta makes G1 there force f_u = 0; k faces
in its sector, theta = 2 pi around a smooth point, pi on a crease or border,
the creases' angle at a corner) and 0 elsewhere; beta f_u then makes the edge
on a star half one degree lower (4 in G1, 5 in G2). Lower degrees cost more
than they save: bicubic pieces stay 4 % of the size off a jack's SubD,
biquartic ones 0.7-0.9 %, degree 5 0.2 %. The conditions are polynomials along
each half, sampled densely enough that they hold everywhere, not only at the
samples.
Each Newton step is the least move (identity plus second differences of the
moves, per face grid) meeting the linearised conditions, from one sparse
factorisation of the KKT system (kernel.KKTFactor), reused for a few cheap
steps after it. Star points stay on the SubD's limit points, crease and border
curves are the SubD's own (exact cubics), the CV strips by faces that stay
bicubic are the SubD's, so those faces join exactly too."""
import math
import time

import numpy as np

G2_SAMPLES = 20      # along each half: its G2 condition is a polynomial of degree 17, two samples may go as implied
G1_SAMPLES = {6: 9, 5: 8, 4: 7, 3: 6}   # G1 along a half is a polynomial of degree p + 1
NEWTON_STEPS = 14
REUSE_STEPS = 3      # cheap steps on one factorisation after each Newton step
ANCHORED = 6         # the first Newton steps also pull the CVs back towards the SubD fit
TOL = 5e-5           # largest condition left (each normalised by its own scale): G2 then within 0.001 %
FAIR = 3.0           # weight of the moves' second differences against their size


def _bez(sb, x, d=0, p=6):
    """Bernstein basis of degree p (or its d-th derivative) at x: (len(x), p + 1)."""
    return sb.basis(np.r_[[0.0] * (p + 1), [1.0] * (p + 1)], np.atleast_1d(np.asarray(x, float)), d, p)


def _split(p):
    """(2p + 1, p + 1): a Bezier curve's CVs -> its two halves' (de Casteljau at 1/2, sharing the middle one)."""
    L = np.zeros((p + 1, p + 1))
    Rr = np.zeros((p + 1, p + 1))
    for j in range(p + 1):
        for k in range(j + 1):
            L[j, k] = math.comb(j, k) / 2 ** j
        for k in range(j, p + 1):
            Rr[j, k] = math.comb(p - j, k - j) / 2 ** (p - j)
    return np.vstack([L, Rr[1:]])


def knots_for(p):
    """The faces' knots at star points: 2 x 2 pieces of degree p, C1 at 0.5 (2p CVs a direction)."""
    return np.r_[[0.0] * (p + 1), [0.5] * (p - 1), [1.0] * (p + 1)]


def _piece(sb, net, ku, kv, lo_u, lo_v, Pi, s):
    """Least-squares Bezier piece (Pi: the pseudo-inverse of its basis at s) over [lo_u, lo_u + .5] x
    [lo_v, lo_v + .5] of a cubic net (exact where the net is one polynomial there)."""
    U, V = np.meshgrid(lo_u + 0.5 * s, lo_v + 0.5 * s, indexing="ij")
    S = sb.derivs(net, ku, kv, U.ravel(), V.ravel(), ((0, 0),))[(0, 0)].reshape(len(s), len(s), 3)
    return np.einsum("ia,abc,jb->ijc", Pi, S, Pi)


def build(sb, conv, g2=True, kernel=None, opts=None):
    """(nets, knots, info) for every cage face of conv (a subd_brep.Converter with star points): bicubic
    one-span nets, and at star points 2 x 2 pieces of degree 6 (G2: 12 x 12 CVs) or 5 (G1: 10 x 10).
    kernel: the B-rep kernel extension (KKTFactor)."""
    opts = opts or {}
    p = int(opts.get("degree", 6 if g2 else 5))
    if g2 and p < 6:
        raise ValueError("G2 needs degree 6 at the star points")
    N, Mid, R = 2 * p, p, p + 1                                    # last grid index, middle row, piece / view size
    q = int(opts.get("beta_power", 2))                            # beta's power (see above)
    T0 = time.perf_counter()
    size = conv.size
    irr = conv.irr
    V, F = conv.V, conv.F
    m = len(F)
    nat0, k0, _ = conv.fresh_natural(0)
    nat2, fk2, cor2 = conv.fresh_natural(2)
    star = [[x in irr for x in fv] for fv in F]
    smooth_star = {x for x in irr if conv.ring[x] is not None and x not in conv.creased}
    sext = [any(s_) for s_ in star]
    # a T-junction's middle (where a quad's side meets two n-gon faces): their corner there is the quad's
    # middle cell on that side
    tal = {T["m"]: ("e", T["e"][0], T["e"][1], Mid) for T in conv.tedges if sext[T["q"]] and T["e"] not in conv.sharp}

    # ---- the grids ((2p + 1)^2: 2 x 2 Bezier pieces sharing the middle rows) and their shared cells
    s15 = np.linspace(0, 1, 2 * p + 3)
    Pi6 = np.linalg.pinv(_bez(sb, s15, 0, p))
    corner_of = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}
    grids = {}
    for f in range(m):
        if not sext[f]:
            continue
        G = np.zeros((N + 1, N + 1, 3))
        # the star pieces first: the exact pieces then overwrite the rows they share
        order = sorted(((qu, qv) for qu in (0, 1) for qv in (0, 1)), key=lambda q: not star[f][corner_of[q]])
        for qu, qv in order:
            G[p * qu:p * qu + R, p * qv:p * qv + R] = _piece(sb, nat2[f], *fk2[f], 0.5 * qu, 0.5 * qv, Pi6, s15)
        for k in range(4):
            if star[f][k]:
                G[(0, N, N, 0)[k], (0, 0, N, N)[k]] = cor2[f][k]   # on the limit point
        grids[f] = G
    faces_i = sorted(grids)

    def side_cell(k, t):
        return [(t, 0), (N, t), (N - t, N), (0, N - t)][k]

    key = {}
    for f in faces_i:
        for i in range(N + 1):
            for j in range(N + 1):
                key[(f, i, j)] = ("c", f, i, j)
        fv = F[f]
        for k in range(4):
            a, b = fv[k], fv[(k + 1) % 4]
            for t in range(N + 1):
                i, j = side_cell(k, t)
                key[(f, i, j)] = tal.get(a, ("v", a)) if t == 0 else (tal.get(b, ("v", b)) if t == N else
                                                       ("e", min(a, b), max(a, b), t if a < b else N - t))
    uniq = sorted(set(key.values()), key=str)
    kid = {k_: n for n, k_ in enumerate(uniq)}
    P = np.zeros((len(uniq), 3))
    for (f, i, j), k_ in key.items():
        P[kid[k_]] = grids[f][i, j]

    # crease and border curves: the uniform cubic B-spline of the crease polygon (a ghost 2a - b beyond a
    # corner), which is what the subdivision makes of them; left as fitted where an end is a dart
    for f in faces_i:
        for k in range(4):
            a, b = F[f][k], F[f][(k + 1) % 4]
            if not conv.hard((min(a, b), max(a, b))):
                continue
            cub = conv.crease_curve(a, b)
            if cub is None:
                continue
            row = np.vstack([Pi6 @ cub(0.5 * s15), (Pi6 @ cub(0.5 + 0.5 * s15))[1:]])
            for t in range(N + 1):
                P[kid[key[(f,) + side_cell(k, t)]]] = row[t]

    # ---- the joins: per half-line, (p + 1)^2 views of cell ids on either side (a along the line, b across)
    def half_view(f, x, y):
        fv = F[f]
        k = fv.index(x)
        ci, cj = (0, N, N, 0)[k], (0, 0, N, N)[k]
        di, dj = (1 if ci == 0 else -1), (1 if cj == 0 else -1)
        tou = (fv[1], fv[0], fv[3], fv[2])[k]
        return [[kid[key[(f,) + ((ci + di * a, cj + dj * b) if tou == y else (ci + di * b, cj + dj * a))]]
                 for b in range(R)] for a in range(R)]

    def inner_view(f, k):
        rg = range(R)
        if k == 0:
            A = [[(Mid - b, a) for b in rg] for a in rg]; B = [[(Mid + b, a) for b in rg] for a in rg]
        elif k == 1:
            A = [[(N - a, Mid - b) for b in rg] for a in rg]; B = [[(N - a, Mid + b) for b in rg] for a in rg]
        elif k == 2:
            A = [[(Mid + b, N - a) for b in rg] for a in rg]; B = [[(Mid - b, N - a) for b in rg] for a in rg]
        else:
            A = [[(a, Mid + b) for b in rg] for a in rg]; B = [[(a, Mid - b) for b in rg] for a in rg]
        return [[kid[key[(f,) + c]] for c in r] for r in A], [[kid[key[(f,) + c]] for c in r] for r in B]

    # ---- T-junctions: along each half of the quad's side, its corner piece split at the half's middle (de
    # Casteljau) -- cells tied to the quad's by linear conditions, whose first row IS the n-gon face's side
    # (one curve on both sides); the n-gon face then joins that split piece as it would a neighbour's
    Ssplit = _split(p)
    tblocks, lin, extra = [], [], []
    for T in conv.tedges:
        tq_ = T["q"]
        if T["e"] in conv.sharp or not sext[tq_]:
            continue
        for x, c, kc in T["pieces"]:
            Qv = np.array(half_view(tq_, x, T["b"] if x == T["a"] else T["a"]))   # (along from x, across)
            from_x = F[c][kc] == x
            VB = np.zeros((N + 1, R), int)
            for i in range(N + 1):
                VB[i, 0] = kid[key[(c,) + side_cell(kc, i if from_x else N - i)]]
                for bb in range(1, R):
                    VB[i, bb] = len(uniq) + len(extra)
                    extra.append(("x", c, x, i, bb))
            tblocks.append((x, c, VB, T))
            for i in range(N + 1):
                for bb in range(R):
                    w = Ssplit[i]
                    if VB[i, bb] == Qv[int(np.argmax(w)), bb] and abs(w.max() - 1.0) < 1e-12:
                        continue                                   # (the piece's own end: the same cell)
                    lin.append((VB[i, bb], Qv[:, bb], w))
    if extra:
        uniq = uniq + extra
        P = np.vstack([P, np.zeros((len(extra), 3))])
    for vid, ids, w in lin:
        P[vid] = w @ P[ids]
    LV = np.array([v_ for v_, _, _ in lin], int)
    LI = np.array([i_ for _, i_, _ in lin], int).reshape(len(lin), R)
    LW = np.array([w_ for _, _, w_ in lin], float).reshape(len(lin), R)

    # ---- held cells: star corners; the edge curve of creases and borders; the rows by faces staying bicubic
    # that the join depends on (three for G2, two for G1)
    fixed = np.zeros(len(uniq), bool)
    for k_ in uniq:
        if k_[0] == "v" and k_[1] in irr:
            fixed[kid[k_]] = True
    for f in faces_i:
        fv = F[f]
        for k in range(4):
            a, b = fv[k], fv[(k + 1) % 4]
            e = (min(a, b), max(a, b))
            hard = conv.hard(e)
            if hard or not all(sext[g] for g in conv.across(f, e)):
                for t in range(N + 1):
                    i, j = side_cell(k, t)
                    for d in range(1 if hard else (3 if g2 else 2)):
                        di, dj = (0, -d, 0, d)[k], (d, 0, -d, 0)[k]
                        fixed[kid[key[(f, i + di, j + dj)]]] = True
    free = np.nonzero(~fixed)[0]
    col = -np.ones(len(uniq), int)
    col[free] = np.arange(len(free))
    n = len(free)

    sharp_at = {}
    for e in conv.sharp:
        for x in e:
            sharp_at.setdefault(x, []).append(e)

    def edge_beta(x, y):
        """beta at x on the smooth edge (x, y): 2 cos(theta / k) for the sector of k faces it lies in (theta =
        2 pi around a smooth point or a dart, pi on a crease or border, the creases' own angle at a corner);
        None: not handled."""
        if x not in irr:
            return 0.0
        nsh = len(sharp_at.get(x, ()))
        if nsh == 0:
            return 2 * math.cos(2 * math.pi / int(conv.val[x])) if conv.ring[x] is not None else None
        if nsh == 1:
            return 2 * math.cos(2 * math.pi / len(conv.vf[x]))
        parent = {f: f for f in conv.vf[x]}

        def root(f):
            while parent[f] != f:
                parent[f] = parent[parent[f]]
                f = parent[f]
            return f
        for f_, g_ in conv.joined_at(x):
            parent[root(f_)] = root(g_)
        r0 = root(conv.edges[(min(x, y), max(x, y))][0][0])
        sector = [f for f in conv.vf[x] if root(f) == r0]
        theta = math.pi                                            # a crease or border runs smoothly through x
        if nsh >= 3:
            # a corner: the angle between the sector's two creases (their tangents at x point along the cage
            # edges), through the sector -- the one nearer the cage's own angle there
            bounds, cage_angle = [], 0.0
            for f in sector:
                fv = F[f]
                i = fv.index(x)
                for z in (fv[(i + 1) % 4], fv[i - 1]):
                    e2 = (min(x, z), max(x, z))
                    if conv.hard(e2):
                        bounds.append(z)
                da, db = V[fv[(i + 1) % 4]] - V[x], V[fv[i - 1]] - V[x]
                cage_angle += math.acos(max(-1.0, min(1.0, float(da @ db / (np.linalg.norm(da) * np.linalg.norm(db) or 1.0)))))
            if len(bounds) == 2:
                ea, eb = V[bounds[0]] - V[x], V[bounds[1]] - V[x]
                alpha = math.acos(max(-1.0, min(1.0, float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) or 1.0)))))
                theta = alpha if abs(alpha - cage_angle) <= abs(2 * math.pi - alpha - cage_angle) else 2 * math.pi - alpha
                if abs(theta - math.pi) < 0.35:
                    # creases running on nearly straight through x (a T): no fan fits their unequal tangents, so the
                    # sector's corner folds flat whatever beta; a quarter turn keeps the joins exact (the SubD's
                    # own corner is as flat)
                    theta = math.pi / 2
            else:
                theta = math.pi / 2
        return 2 * math.cos(theta / len(sector))

    VF, VG, BETA, ENDS, ALPHA, BSHAPE = [], [], [], [], [], []
    for e, uses in conv.edges.items():
        if len(uses) != 2 or e in conv.sharp:
            continue
        f, g = uses[0][0], uses[1][0]
        if not (sext[f] and sext[g]):
            continue
        for x, y in ((e[0], e[1]), (e[1], e[0])):
            b_ = edge_beta(x, y)
            if b_ is None:
                continue
            VF.append(half_view(f, x, y))
            VG.append(half_view(g, x, y))
            BETA.append(b_)
            ALPHA.append(1.0)
            BSHAPE.append(0)
            ENDS.append((("v", x, y), ("m", e, x)))
    for x, c, VB, T in tblocks:                                   # an n-gon face's side against the split piece
        tm = T["m"]
        pe = (min(x, tm), max(x, tm))
        b_ = edge_beta(x, tm)
        if b_ is None:
            continue
        for frm, to in ((x, tm), (tm, x)):
            VF.append(half_view(c, frm, to))
            VG.append([[int(VB[a if frm == x else N - a, bb]) for bb in range(R)] for a in range(R)])
            # the n-gon face runs at half the quad's speed both along and across (its pieces are half as big):
            # f_t and f_u count twice; beta is the quad's own, beta (1 - U)^2 over its half side, in the n-gon
            # face's halves (U = u / 2 from x, 1 - u / 2 from the middle): the same at the star point, vanishing
            # with its slope where the quad's pieces meet
            BETA.append(2.0 * b_ if frm == x else -2.0 * b_)     # (from the middle f_u runs back towards x)
            ALPHA.append(2.0)
            BSHAPE.append(1 if frm == x else 2)
            ENDS.append((("v", frm, to), ("m", pe, frm)))
    HE = len(VF)                                                   # edge halves first, then the middle half-lines
    if g2:
        for f in faces_i:
            for k in range(4):
                A, B = inner_view(f, k)
                VF.append(A)
                VG.append(B)
                BETA.append(0.0)
                ALPHA.append(1.0)
                BSHAPE.append(0)
                a, b = F[f][k], F[f][(k + 1) % 4]
                e = (min(a, b), max(a, b))
                if e in conv.tq and conv.tedges[conv.tq[e]]["m"] in tal:   # ends at a T-junction's middle
                    ENDS.append((("v", conv.tedges[conv.tq[e]]["m"], ("q", f)), ("c", f, k)))
                else:
                    ENDS.append((("m", e, ("f", f)), ("c", f, k)))
    VF = np.array(VF, int).reshape(-1, R, R)
    VG = np.array(VG, int).reshape(-1, R, R)
    BETA = np.array(BETA, float)
    ALPHA = np.array(ALPHA, float)
    BSHAPE = np.array(BSHAPE, int)

    def bshape(u, kinds):
        """beta's shape along each half-line (its value at the half's start is BETA's)."""
        u = np.asarray(u)[None, :]
        return np.where((kinds == 1)[:, None], (1 - u / 2) ** q,
                        np.where((kinds == 2)[:, None], (u / 2) ** q, (1 - u) ** q))
    H = len(VF) if g2 else 0
    S = opts.get("ns", G2_SAMPLES)
    keep2 = np.ones((H, S), bool)
    # where several half-lines cross, some of their G2 samples follow from the others and from G1: those go
    at = {}
    for h, (e0, e1) in enumerate(ENDS[:H]):
        at.setdefault(("V", e0[1]) if e0[0] == "v" else ("M", e0[1]), []).append((h, 0, e0))
        at.setdefault(("M", e1[1]) if e1[0] == "m" else ("C", e1[1]), []).append((h, S - 1, e1))
    for (kd, p_), lst in at.items():
        drop = []
        if kd == "V" and p_ in tal:
            # a T-junction's middle: four lines cross as at an ordinary vertex (the two halves of the quad's side,
            # the quad's middle line, the n-gon faces' joint): one of each opposite two goes
            far = conv.tedges[conv.tmid[p_]]["b"]
            drop = [(h, s_) for h, s_, en in lst if en[2] == far or isinstance(en[2], tuple)]
        elif kd == "V":
            ring_ = conv.ring[p_]
            if ring_ is None:
                continue
            n_ = int(conv.val[p_])
            es = list(ring_[0])
            pos = {es.index(en[2]): (h, s_) for h, s_, en in lst}
            if p_ in smooth_star and len(pos) == n_:
                keep_n = max(n_ - 3, 0)                            # n - 3 consecutive ones around a star point
                drop = [hs_ for i, hs_ in pos.items() if i >= keep_n]
            elif n_ == 4 and not sharp_at.get(p_):
                drop = [hs_ for i, hs_ in pos.items() if (i + 2) % 4 in pos and (i + 2) % 4 < i]
        elif kd == "M":
            if not conv.hard(p_):                                 # the two halves are opposite, so are the lines
                inners = [(h, s_) for h, s_, en in lst if isinstance(en[2], tuple)]
                halves = [(h, s_) for h, s_, en in lst if not isinstance(en[2], tuple)]
                drop = halves[1:] + inners[1:]
        else:
            drop = [(h, s_) for h, s_, en in lst if en[2] >= 2]   # a face centre: two of its four lines
        for h, s_ in drop:
            keep2[h, s_] = False
    # C1 across the middle lines: cells (p - 1, p, p + 1) across each line
    c1 = []
    for f in faces_i:
        for t in range(N + 1):
            c1.append((kid[key[(f, Mid - 1, t)]], kid[key[(f, Mid, t)]], kid[key[(f, Mid + 1, t)]]))
            c1.append((kid[key[(f, t, Mid - 1)]], kid[key[(f, t, Mid)]], kid[key[(f, t, Mid + 1)]]))
    c1 = np.array(c1, int).reshape(-1, 3)

    # ---- the conditions' weights per sample
    us = np.linspace(0, 1, S)
    Bu = {d: _bez(sb, us, d, p) for d in range(3)}
    B0 = {d: _bez(sb, [0.0], d, p)[0] for d in range(3)}
    W = {(i, j): np.einsum("sa,b->sab", Bu[i], B0[j]) for i in range(3) for j in range(3) if i + j <= 2}
    S1 = opts.get("ns1", G1_SAMPLES.get(p, 2 * p))
    us1 = np.linspace(0, 1, S1)
    Bu1 = {d: _bez(sb, us1, d, p) for d in range(2)}
    W1 = {(i, j): np.einsum("sa,b->sab", Bu1[i], B0[j]) for i, j in ((0, 1), (1, 0))}
    beta1 = BETA[:HE, None] * bshape(us1, BSHAPE[:HE])
    Wf1 = ALPHA[:HE, None, None, None] * W1[(0, 1)][None] - beta1[:, :, None, None] * W1[(1, 0)][None]   # (HE, S1, R, R)
    beta = BETA[:H, None] * bshape(us, BSHAPE[:H])
    al = ALPHA[:H, None, None, None]
    Wq = al ** 2 * W[(0, 2)][None] - 2 * al * beta[:, :, None, None] * W[(1, 1)][None] \
        + (beta ** 2)[:, :, None, None] * W[(2, 0)][None]
    cf, cg = col[VF], col[VG]
    if H:
        # a G2 sample whose second derivatives are all held: only the normal could turn to meet it -- it goes
        fq = (np.einsum("sab,hab->hs", (np.abs(W[(0, 2)]) > 1e-12).astype(float), (cg[:H] >= 0).astype(float)) +
              np.einsum("hsab,hab->hs", (np.abs(Wq) > 1e-12).astype(float), (cf[:H] >= 0).astype(float)))
        keep2 &= fq > 0
    g2map = -np.ones(H * S, int)
    g2map[keep2.ravel()] = np.arange(int(keep2.sum()))
    n2 = int(keep2.sum())

    def dsum(Pc, views, Wm):
        G = Pc[views]                                                               # (h, R, R, 3)
        if Wm.ndim == 3:
            return np.einsum("sab,habc->hsc", Wm, G)
        return np.einsum("hsab,habc->hsc", Wm, G)

    den = None

    def residual(Pc, jac=True):
        nonlocal den
        g1 = dsum(Pc, VF[:HE], Wf1) + dsum(Pc, VG[:HE], W1[(0, 1)])               # (HE, S1, 3)
        sc1 = np.maximum(np.linalg.norm(dsum(Pc, VF[:HE], W1[(0, 1)]), axis=2), 1e-300)
        r1 = g1 / sc1[:, :, None]
        if H:
            fu, ft = dsum(Pc, VF[:H], W[(1, 0)]), dsum(Pc, VF[:H], W[(0, 1)])
            Q = dsum(Pc, VG[:H], W[(0, 2)]) - dsum(Pc, VF[:H], Wq)
            N = np.cross(fu, ft)
            if den is None:
                den = np.maximum(np.linalg.norm(fu, axis=2) * np.linalg.norm(ft, axis=2) ** 3, 1e-300) / size
            g2v = ((N * Q).sum(2) / den).ravel()[keep2.ravel()]
        else:
            g2v = np.zeros(0)
        r3 = Pc[c1[:, 0]] + Pc[c1[:, 2]] - 2 * Pc[c1[:, 1]]
        sc3 = np.maximum(np.linalg.norm(Pc[c1[:, 2]] - Pc[c1[:, 0]], axis=1), 1e-300)
        rl = (Pc[LV] - np.einsum("lk,lkc->lc", LW, Pc[LI])) / size if len(LV) else np.zeros((0, 3))
        res = np.concatenate([r1.ravel(), g2v, (r3 / sc3[:, None]).ravel(), rl.ravel()])
        if not jac:
            return res
        rows, cols, vals = [], [], []
        hs = np.arange(HE * S1).reshape(HE, S1)
        for views, Wm in ((cf[:HE], Wf1), (cg[:HE], W1[(0, 1)])):
            Wfull = Wm if Wm.ndim == 4 else np.broadcast_to(Wm[None], (HE, S1, R, R))
            for a in range(R):
                for b in range(R):
                    cc = views[:, a, b]
                    ok = cc >= 0
                    if not ok.any():
                        continue
                    w = Wfull[ok][:, :, a, b] / sc1[ok]
                    rr = hs[ok]
                    for c in range(3):
                        rows.append((rr * 3 + c).ravel())
                        cols.append(np.repeat(cc[ok], S1) * 3 + c)
                        vals.append(w.ravel())
        off = HE * S1 * 3
        if H:
            hs2 = np.arange(H * S).reshape(H, S)
            a1, a2 = np.cross(ft, Q), np.cross(Q, fu)
            for a in range(R):
                for b in range(R):
                    ccf = cf[:H, a, b]
                    okf = ccf >= 0
                    if okf.any():
                        vec = (W[(1, 0)][:, a, b][None, :, None] * a1 + W[(0, 1)][:, a, b][None, :, None] * a2
                               - Wq[:, :, a, b][:, :, None] * N)[okf] / den[okf][:, :, None]
                        rr2 = g2map[hs2[okf]].ravel()
                        for c in range(3):
                            rows.append(np.where(rr2 >= 0, off + rr2, -1))
                            cols.append(np.repeat(ccf[okf], S) * 3 + c)
                            vals.append(vec[:, :, c].ravel())
                    ccg = cg[:H, a, b]
                    okg = ccg >= 0
                    if okg.any():
                        vec = (W[(0, 2)][:, a, b][None, :, None] * N)[okg] / den[okg][:, :, None]
                        rr2 = g2map[hs2[okg]].ravel()
                        for c in range(3):
                            rows.append(np.where(rr2 >= 0, off + rr2, -1))
                            cols.append(np.repeat(ccg[okg], S) * 3 + c)
                            vals.append(vec[:, :, c].ravel())
        off2 = off + n2
        ci = np.arange(len(c1))
        for k_, w in ((0, 1.0), (2, 1.0), (1, -2.0)):
            cc = col[c1[:, k_]]
            ok = cc >= 0
            for c in range(3):
                rows.append(off2 + ci[ok] * 3 + c)
                cols.append(cc[ok] * 3 + c)
                vals.append(w / sc3[ok])
        off3 = off2 + len(c1) * 3                                  # the T-junctions' split cells: linear
        li = np.arange(len(LV))
        for ids, w in [(LV, np.ones(len(LV)))] + [(LI[:, k_], -LW[:, k_]) for k_ in range(R)]:
            cc = col[ids] if len(ids) else np.zeros(0, int)
            ok = (cc >= 0) & (w != 0)
            for c in range(3):
                rows.append(off3 + li[ok] * 3 + c)
                cols.append(cc[ok] * 3 + c)
                vals.append(w[ok] / size)
        R_, C_, V_ = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        ok = R_ >= 0
        return res, (R_[ok], C_[ok], V_[ok])

    # ---- the move metric: identity plus the moves' second differences along every grid row and column
    tr = []
    for f in faces_i:
        for i in range(N + 1):
            for j in range(N + 1):
                for di, dj in ((1, 0), (0, 1)):
                    if i - di >= 0 and i + di <= N and j - dj >= 0 and j + dj <= N:
                        tr.append((col[kid[key[(f, i - di, j - dj)]]], col[kid[key[(f, i, j)]]],
                                   col[kid[key[(f, i + di, j + dj)]]]))
    tr = np.array(tr, int).reshape(-1, 3)
    wf = opts.get("fair", FAIR)
    wts = (1.0, -2.0, 1.0)
    Mr_l, Mc_l, Mv_l = [np.arange(3 * n)], [np.arange(3 * n)], [np.ones(3 * n)]
    for p_ in range(3):
        for q_ in range(3):
            ok = (tr[:, p_] >= 0) & (tr[:, q_] >= 0)
            for c in range(3):
                Mr_l.append(tr[ok, p_] * 3 + c)
                Mc_l.append(tr[ok, q_] * 3 + c)
                Mv_l.append(np.full(int(ok.sum()), wf * wts[p_] * wts[q_]))
    Mr, Mc = np.concatenate(Mr_l).astype(np.int64), np.concatenate(Mc_l).astype(np.int64)
    Mv = np.concatenate(Mv_l)

    def apply(Pc, dx):
        Pn = Pc.copy()
        Pn[free] += dx.reshape(-1, 3)
        return Pn

    def factor_at(res, J):
        r_, c_, v_ = J
        nrm = np.bincount(r_, weights=v_ * v_, minlength=len(res))
        live = nrm > 1e-28
        mapr = -np.ones(len(res), int)
        mapr[live] = np.arange(int(live.sum()))
        sel = live[r_]
        rr = mapr[r_[sel]]
        sq = np.sqrt(nrm[live])
        K = kernel.KKTFactor(Mr, Mc, Mv, 3 * n, rr.astype(np.int64), c_[sel].astype(np.int64), v_[sel] / sq[rr],
                             int(live.sum()), opts.get("delta", 1e-13), bool(opts.get("eigen_only", False)))
        return K, live, sq

    def solve_with(K, live, sq, res, Pc, anchored, refine):
        rx = np.zeros(3 * n)
        if anchored:
            d_ = (Pc[free] - P[free]).ravel()
            rx = -np.bincount(Mr, weights=Mv * d_[Mc], minlength=3 * n)
        return np.asarray(K.solve(rx, -res[live] / sq, refine))

    Pc = P.copy()
    hist = []
    if n:
        res, J = residual(Pc)
        hist.append(float(np.abs(res).max()) if len(res) else 0.0)
        tol = opts.get("tol", TOL)
        steps = opts.get("iters", NEWTON_STEPS if g2 else 2)
        for it in range(steps):
            if not len(res) or hist[-1] < tol:
                break
            K, live, sq = factor_at(res, J)
            cur = hist[-1]
            anchored = it < opts.get("anchored", ANCHORED)
            ok = True
            for inner in range(1 + opts.get("reuse", REUSE_STEPS)):
                dx = solve_with(K, live, sq, res, Pc, anchored, 3 if inner == 0 else 1)
                s_, ok = 1.0, False
                while s_ > 1e-3:
                    Pt = apply(Pc, s_ * dx)
                    rt = residual(Pt, jac=False)
                    if float(np.abs(rt).max()) < cur or (inner == 0 and anchored and it > 1):
                        ok = True
                        break
                    if inner:
                        break                                      # a reuse step that does not help: refactorise
                    s_ *= 0.5
                if not ok:
                    break
                Pc, res = Pt, rt
                cur = float(np.abs(res).max())
                hist.append(cur)
                if cur < tol:
                    break
            if (not ok and inner == 0) or cur < tol:
                break
            res, J = residual(Pc)
    info = {"newton": hist, "unknowns": 3 * n, "conditions": HE * S1 * 3 + n2 + len(c1) * 3 + len(LV) * 3,
            "t_junctions": len(tblocks) // 2,
            "max_move": float(np.abs(Pc - P).max() / size) if len(P) else 0.0,
            "seconds": time.perf_counter() - T0, "degree": p}

    nets, knots = [], []
    for f in range(m):
        if not sext[f]:
            nets.append(nat0[f].copy())
            knots.append(k0[f])
        else:
            G = np.array([[Pc[kid[key[(f, i, j)]]] for j in range(N + 1)] for i in range(N + 1)])
            keep = [i for i in range(N + 1) if i != Mid]           # C1 at the middle: the shared row is redundant
            nets.append(G[np.ix_(keep, keep)])
            knots.append((knots_for(p), knots_for(p)))
    return nets, knots, info


def deviation(sb, conv, nets, knots, level=3):
    """Largest distance from the SubD surface to the faces built here: its exact limit points at the level's
    (i, j) / 2^level on every face by a star point, each projected onto its face (one-span faces are exact)."""
    lims = conv.limit_grid(level)
    worst = 0.0
    for f, (net, (ku, kv)) in enumerate(zip(nets, knots)):
        if np.count_nonzero(ku == ku[0]) <= 4:
            continue
        N = lims[f].shape[0] - 1
        g = np.arange(N + 1) / N
        uv = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        q = lims[f].reshape(-1, 3)
        best = None
        for it in range(6):                                      # Newton onto the face from the same parameters
            D = sb.derivs(net, ku, kv, uv[:, 0], uv[:, 1], ((0, 0), (1, 0), (0, 1)))
            r = D[(0, 0)] - q
            d = np.linalg.norm(r, axis=1)
            best = d if best is None else np.minimum(best, d)    # (a step may jump off near a folded corner)
            if it == 5:
                break
            a11, a12, a22 = (D[(1, 0)] ** 2).sum(1), (D[(1, 0)] * D[(0, 1)]).sum(1), (D[(0, 1)] ** 2).sum(1)
            b1, b2 = (D[(1, 0)] * r).sum(1), (D[(0, 1)] * r).sum(1)
            dt = np.maximum(a11 * a22 - a12 ** 2, 1e-300)
            step = np.c_[(a22 * b1 - a12 * b2) / dt, (a11 * b2 - a12 * b1) / dt]
            uv = np.clip(uv - np.clip(step, -0.1, 0.1), 0, 1)
        worst = max(worst, float(best.max()))
    return worst
