"""Soft fillet sections: one Bezier span, G2 (degree 5) or G3 (degree 7) to
both faces, whose curvature ramps from the faces' own up to a peak of exactly
1 / R in the middle.

The shape is our own construction. The ideal corner is a circular arc of
radius R between two curvature ramps (a clothoid, kappa rising linearly with
arc length, for G2; a smoothstep ramp, kappa' = 0 at the face too, for G3).
Smoothing = the share of the turn taken by the ramps: 0 is a plain circle,
1 has no arc left (ramp, peak, ramp). The section is the single span closest
to that ideal with its first cont + 1 CVs on each face's tangent line
(spacings along the line are the only unknowns, mirror symmetric), fitted
with exact foot-point parameters. Its own peak curvature then sets the size:
with the contact points at distance L from where the two tangent lines meet,
the peak is g / L, so L = g R gives a minimum radius of exactly R. On planes
this is exact; on curved faces the same spacings are used per section,
measured on that section's own L, with the faces' normal curvature (and, G3,
its rate) added on CVs 2 and 3.

Pure numpy (no bpy, no kernel): imported by secfillet.py, classa.py and the
add-on's preview."""

import math

import numpy as np

SMOOTH_MIN, SMOOTH_MAX = 0.1, 1.0
_N_IDEAL = 241                  # ideal points the span is fitted to
_N_INT = 1601                   # integration nodes of the ideal curve


def clamp_smoothing(smoothing, cont):
    """The smoothing a single span can hold monotone: G3 (degree 7) needs at
    least 0.5 (below, its curvature bumps on the ramp)."""
    lo = 0.5 if int(cont) >= 3 else SMOOTH_MIN
    return float(min(max(float(smoothing), lo), SMOOTH_MAX))


def degree_of(cont):
    return 7 if int(cont) >= 3 else 5


def _bern(n, u):
    u = np.asarray(u, float)[:, None]
    i = np.arange(n + 1)[None, :]
    c = np.array([math.comb(n, k) for k in range(n + 1)], float)[None, :]
    return c * u ** i * (1.0 - u) ** (n - i)


def _ders(P, n, u, d):
    """d-th derivative of the degree-n Bezier with CVs P at the params u."""
    Q = np.asarray(P, float)
    k = n
    for _ in range(d):
        Q = k * np.diff(Q, axis=0)
        k -= 1
    if k < 0:
        return np.zeros((len(u), Q.shape[1] if Q.ndim > 1 else 2))
    return _bern(k, u) @ Q


def curvature(P, u):
    """Signed curvature of a planar Bezier (CVs P, n x 2) at the params u."""
    n = len(P) - 1
    d1, d2 = _ders(P, n, u, 1), _ders(P, n, u, 2)
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3


def ideal(phi, smoothing, cont):
    """Ideal corner with R = 1 turning by phi: (points (N x 2) by arc length,
    setback L). It starts at the origin heading +x and turns left; L is the
    distance from the start to where the two end tangent lines meet."""
    psi = 0.5 * phi * min(max(smoothing, 0.0), 1.0)         # turn in each ramp
    Lr = 2.0 * psi                                           # both ramp kinds average kappa = 1/2
    La = phi - 2.0 * psi
    S = 2.0 * Lr + La
    s = np.linspace(0.0, S, _N_INT)

    def ramp(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x) if cont >= 3 else x
    if Lr > 0:
        k = np.where(s < Lr, ramp(s / Lr), np.where(s > Lr + La, ramp((S - s) / Lr), 1.0))
    else:
        k = np.ones_like(s)
    ds = np.diff(s)
    th = np.r_[0.0, np.cumsum(0.5 * (k[1:] + k[:-1]) * ds)]
    th *= phi / th[-1]                                       # (trapezoid rounding)
    c, sn = np.cos(th), np.sin(th)
    x = np.r_[0.0, np.cumsum(0.5 * (c[1:] + c[:-1]) * ds)]
    y = np.r_[0.0, np.cumsum(0.5 * (sn[1:] + sn[:-1]) * ds)]
    E = np.array([x[-1], y[-1]])
    L = float(E[0] - E[1] / math.tan(phi))
    P = np.c_[x, y]
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    q = np.linspace(0.0, seg[-1], _N_IDEAL)
    return np.c_[np.interp(q, seg, x), np.interp(q, seg, y)], L


def planar_cvs(spacing, phi, L=1.0):
    """CVs (2 x (cont + 1) rows of 2D points) of the symmetric span: contact at
    the origin, the tangent lines meeting at (L, 0), line 2 leaving it at angle
    phi. spacing: CV-to-CV distances along the line (fractions of L)."""
    xs = np.r_[0.0, np.cumsum(spacing)] * L
    e2 = np.array([math.cos(phi), math.sin(phi)])
    C = np.array([L, 0.0])
    one = [np.array([x, 0.0]) for x in xs]
    two = [C + (L - x) * e2 for x in xs]
    return np.array(one + two[::-1])


_cache = {}
_GRID = math.radians(0.5)          # turn-angle step of the solved shapes (interpolated between)


def _fit(phi, smoothing, cont):
    """Exact solve at one turn angle (see shape)."""
    m = degree_of(cont)
    Q, Li = ideal(phi, smoothing, cont)
    Q = Q / Li                                               # contact to corner = 1
    nsp = cont                                               # spacings per side
    base = planar_cvs(np.zeros(nsp), phi)
    dirs = [planar_cvs(np.eye(nsp)[j], phi) - base for j in range(nsp)]

    def cvs(x):
        return base + sum(xj * D for xj, D in zip(x, dirs))

    def feet(P, u):
        for _ in range(4):
            F, F1, F2 = _bern(m, u) @ P, _ders(P, m, u, 1), _ders(P, m, u, 2)
            r = F - Q
            den = (F1 * F1).sum(1) + (r * F2).sum(1)
            du = np.where(np.abs(den) > 1e-14, (r * F1).sum(1) / np.where(np.abs(den) > 1e-14, den, 1.0), 0.0)
            du[0] = du[-1] = 0.0
            u = np.clip(u - du, 0.0, 1.0)
        return u

    def sag(x):
        """d2 kappa / du2 at the middle: > 0 is a dip (two peaks)."""
        h = 2e-3
        k = np.abs(curvature(cvs(x), np.array([0.5 - h, 0.5, 0.5 + h])))
        return (k[0] - 2.0 * k[1] + k[2]) / (h * h)

    u = np.linspace(0.0, 1.0, len(Q))
    x = np.full(nsp, 1.0 / (nsp + 1.5))
    for _ in range(30):                                      # unconstrained: linear in x
        B = _bern(m, u)
        F0 = B @ base
        J = np.stack([(B @ D).ravel() for D in dirs], 1)
        x_new = np.linalg.lstsq(J, (Q - F0).ravel(), rcond=None)[0]
        u = feet(cvs(x_new), u)
        done = float(np.abs(x_new - x).max()) < 1e-13
        x = x_new
        if done:
            break
    flat = False
    if sag(x) > 0.0:
        # the span cannot hold the arc's plateau: it overshoots and the curvature
        # dips in the middle. Hold the middle flat instead (kappa'' = 0 there),
        # the closest such span (Gauss-Newton, the flat middle weighted in)
        flat = True
        w = 1e3
        for _ in range(60):
            B = _bern(m, u)
            r = np.r_[(B @ cvs(x) - Q).ravel(), w * sag(x) * 1e-3]
            Jc = []
            for j in range(nsp):
                e = np.zeros(nsp)
                e[j] = 1e-7
                rj = np.r_[(B @ cvs(x + e) - Q).ravel(), w * sag(x + e) * 1e-3]
                Jc.append((rj - r) / 1e-7)
            Jc = np.array(Jc).T
            dx = np.linalg.lstsq(Jc, -r, rcond=None)[0]
            x = x + dx
            u = feet(cvs(x), u)
            if float(np.abs(dx).max()) < 1e-12:
                break
    P = cvs(x)
    k = np.abs(curvature(P, np.linspace(0.0, 0.5, 601)))
    monotone = bool(np.all(np.diff(k) >= -1e-6 * max(k.max(), 1e-12)))
    g = float(abs(curvature(P, np.array([0.5]))[0]))
    g = max(g, float(k.max()))
    err = float(np.linalg.norm(_bern(m, u) @ P - Q, axis=1).max()) * Li
    return {"degree": m, "cont": cont, "spacing": [float(v) for v in x], "g": g, "err": err,
            "monotone": monotone, "flat": flat, "ok": bool(np.all(x > 0) and x.sum() < 1.0)}


def _node(i, smoothing, cont):
    key = (i, round(smoothing, 12), cont)
    hit = _cache.get(key)
    if hit is None:
        if len(_cache) > 20000:
            _cache.clear()
        hit = _cache[key] = _fit(i * _GRID, smoothing, cont)
    return hit


def shape(phi, smoothing, cont=2):
    """Normalised soft section for a turn of phi (radians, between the two
    faces' tangents across the edge): dict(degree, spacing = CV gaps along each
    tangent line as fractions of L, g = peak curvature x L, err = largest
    distance to the ideal / R, monotone = curvature rises steadily to the
    middle, flat = the middle was held flat, the span being unable to follow
    the arc). Solved on a 0.5 deg grid of turns and interpolated (cubic) in
    between, so the sections change smoothly along a fillet whose angle does."""
    cont = 3 if int(cont) >= 3 else 2
    phi = float(min(max(phi, math.radians(2.0)), math.radians(178.0)))
    smoothing = float(min(max(smoothing, SMOOTH_MIN), SMOOTH_MAX))
    f = phi / _GRID
    i = int(math.floor(f))
    t = f - i
    if t < 1e-9:
        return _node(i, smoothing, cont)
    ns = [_node(i + j, smoothing, cont) for j in (-1, 0, 1, 2)]
    # 4-point Lagrange weights at t (nodes at -1, 0, 1, 2)
    wts = [-t * (t - 1) * (t - 2) / 6.0, (t + 1) * (t - 1) * (t - 2) / 2.0,
           -(t + 1) * t * (t - 2) / 2.0, (t + 1) * t * (t - 1) / 6.0]
    sp = [sum(w * n["spacing"][k] for w, n in zip(wts, ns)) for k in range(cont)]
    out = dict(ns[1])
    out.update(spacing=sp, g=sum(w * n["g"] for w, n in zip(wts, ns)),
               err=max(ns[1]["err"], ns[2]["err"]), monotone=ns[1]["monotone"] and ns[2]["monotone"],
               flat=ns[1]["flat"] or ns[2]["flat"])
    return out


def setback(R, phi, smoothing, cont=2):
    """L: contact point to tangent-line corner, for a minimum radius R."""
    return shape(phi, smoothing, cont)["g"] * float(R)


def ball_radius(R, phi, smoothing, cont=2):
    """The rolling ball whose contact points are the soft section's: L / tan(phi / 2)."""
    return setback(R, phi, smoothing, cont) / math.tan(0.5 * max(phi, 1e-9))


def section_cvs(pa, ta, na, ka, pb, tb, nb, kb, L, spacing, degree, ka3=0.0, kb3=0.0):
    """3D CVs (degree + 1, 3) of a soft section from contact pa to pb.
    t*: unit tangent at the contact pointing across toward the other rail,
    n*: unit face normal turned toward the fillet's centre (the side the
    section bends to), k*: the face's normal curvature in direction t*,
    positive when it bends toward n*, k*3: its rate (G3 only). L: contact to
    the tangent-line corner (the same on both sides: symmetric section)."""
    m = int(degree)
    c = len(spacing)
    sides = []
    for p, t, n, k, k3 in ((pa, ta, na, ka, ka3), (pb, tb, nb, kb, kb3)):
        p, t, n = (np.asarray(v, float) for v in (p, t, n))
        d = np.r_[0.0, np.cumsum(spacing)] * L               # along the line
        cv = [p + d[j] * t for j in range(c + 1)]
        a = d[1]
        if c >= 2:
            # G2: C''(0) . n = m (m - 1) h2 = kappa (m a)^2
            h2 = k * m * a * a / (m - 1)
            cv[2] = cv[2] + h2 * n
            if c >= 3:
                # G3: with v = m a and tau = m (m - 1) (b - a) the tangential part of
                # C''(0): C'''(0) . n = v^3 kappa_s + 3 kappa v tau
                # = m (m - 1) (m - 2) (h3 - 3 h2)
                b = d[2] - d[1]
                v, tau = m * a, m * (m - 1) * (b - a)
                h3 = 3.0 * h2 + (v ** 3 * k3 + 3.0 * k * v * tau) / (m * (m - 1) * (m - 2))
                cv[3] = cv[3] + h3 * n
        sides.append(cv)
    return np.array(sides[0] + sides[1][::-1])


def peak_radius(cvs, n=161):
    """Smallest radius of curvature of a 3D Bezier section: sampled, then
    refined by golden section around the sharpest sample (smooth in the CVs,
    so it can be solved for)."""
    P = np.asarray(cvs, float)
    m = len(P) - 1

    def k(u):
        u = np.atleast_1d(np.asarray(u, float))
        d1, d2 = _ders(P, m, u, 1), _ders(P, m, u, 2)
        return np.linalg.norm(np.cross(d1, d2), axis=1) / np.linalg.norm(d1, axis=1) ** 3
    u = np.linspace(0.0, 1.0, n)
    ks = k(u)
    i = int(np.argmax(ks))
    lo, hi = u[max(i - 1, 0)], u[min(i + 1, n - 1)]
    gr = 0.5 * (math.sqrt(5.0) - 1.0)
    a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
    ka, kb = k(a)[0], k(b)[0]
    for _ in range(40):
        if ka > kb:
            hi, b, kb = b, a, ka
            a = hi - gr * (hi - lo)
            ka = k(a)[0]
        else:
            lo, a, ka = a, b, kb
            b = lo + gr * (hi - lo)
            kb = k(b)[0]
    kmax = max(float(ks[i]), float(ka), float(kb))
    return 1.0 / kmax if kmax > 0 else float("inf")
