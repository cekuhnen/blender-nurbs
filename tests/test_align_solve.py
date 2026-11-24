"""Align solver (alignsolve.py, pure numpy):
    blender -b --factory-startup --python tests/test_align_solve.py
(or any Python with numpy: the module is loaded straight from its file)."""
import importlib.util, math, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("alignsolve", os.path.join(ROOT, "blender_nurbs", "alignsolve.py"))
al = importlib.util.module_from_spec(spec)
spec.loader.exec_module(al)
make, mate, mated, moved = al.make, al.mate, al.mated, al.moved

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

close = lambda a, b, tol=1e-9: bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))
X, Y, Z = np.eye(3)

def rigid(M):
    R = M[:3, :3]
    return close(R.T @ R, np.eye(3), 1e-12) and abs(np.linalg.det(R) - 1) < 1e-12

def apply(M, P):
    return M[:3, :3] @ np.asarray(P, float) + M[:3, 3]


# --- face onto face: flush, facing each other, smallest move -------------
a = make("PLANE", (5, 3, 1), Z, hint=X)          # the top of a part, away from the origin
b = make("PLANE", (0, 0, 0), Z, hint=X)          # a floor facing up
r = mate(a, b)
check("face/face: opposed", r["relation"] == "opposed" and close(r["R"] @ Z, -Z))
check("face/face: mated, rigid", mated(a, b, r["M"]) and rigid(r["M"]))
check("face/face: turns about its own point", close(r["p"], (5, 3, 0)) and close(apply(r["M"], a["o"]), (5, 3, 0)),
      r["p"])
check("face/face: half a turn", abs(r["angle"] - 180) < 1e-9 and abs(r["dist"] - 1) < 1e-9)
check("face/face: edges kept in line", close(abs(r["R"] @ X), X))

# already facing: only the move
a = make("PLANE", (2, 2, 4), -Z)
r = mate(a, b)
check("face/face facing: no turn", close(r["R"], np.eye(3)) and close(r["t"], (0, 0, -4)))

# --- pinned centres meet --------------------------------------------------
a = make("PLANE", (5, 3, 1), Z, pin=True, hint=X)
b = make("PLANE", (1, 1, 0), Z, pin=True, hint=X)
r = mate(a, b)
check("center/center: points meet", close(apply(r["M"], a["o"]), b["o"]) and close(r["R"] @ Z, -Z))
a = make("PLANE", (5, 3, 1), Z, pin=True)
b = make("PLANE", (1, 1, 0), Z)                  # a free face: the centre only lands in it
r = mate(a, b)
check("center/face: lands in the face", close(apply(r["M"], a["o"]), (5, 3, 0)))

# --- peg ring into hole ring ---------------------------------------------
a = make("RING", (10, 0, 0), X, r=0.5)           # a peg lying along X, its end ring
b = make("RING", (0, 0, 0), Z, r=0.5)            # a hole's rim on a floor
r = mate(a, b)
check("ring/ring: centres meet, axes opposed", close(apply(r["M"], a["o"]), b["o"]) and close(r["R"] @ X, -Z)
      and r["relation"] == "opposed")
check("ring/ring: a quarter turn", abs(r["angle"] - 90) < 1e-9)

# ring onto an axis: coaxial, keeps its height
a = make("RING", (3, 1, 2), (0, 0.1, 1))
b = make("AXIS", (0, 0, 0), -Z)
r = mate(a, b)
check("ring/axis: parallel, nearest sense", r["relation"] == "parallel"
      and close(r["R"] @ al._unit((0, 0.1, 1)), Z))
check("ring/axis: centre on the axis", close(apply(r["M"], a["o"])[:2], (0, 0)) and abs(apply(r["M"], a["o"])[2] - 2) < 1e-9)

# --- axis to axis: coaxial with the smallest turn -------------------------
d = al._unit((0, 0.1, 1))
a = make("AXIS", (2, 0, 0), d)
b = make("AXIS", (0, 0, 0), -Z)
r = mate(a, b)
check("axis/axis: small turn (nearest sense)", abs(r["angle"] - math.degrees(math.atan(0.1))) < 1e-9, r["angle"])
check("axis/axis: coaxial", mated(a, b, r["M"]))

# --- line in a plane, plane on a line --------------------------------------
a = make("LINE", (1, 1, 1), (1, 0, 1))
b = make("PLANE", (0, 0, 0), Z)
r = mate(a, b)
check("line/face: in plane", r["relation"] == "in plane" and abs(r["angle"] - 45) < 1e-9 and mated(a, b, r["M"]))
check("line/face: spin about the normal", close(r["axis"], Z))
a = make("LINE", (1, 1, 1), Z)                   # along the normal: any way down, still in plane
r = mate(a, b)
check("line/face: degenerate", mated(a, b, r["M"]) and abs(r["angle"] - 90) < 1e-9)
a = make("PLANE", (0, 0, 2), al._unit((0, 1, 1)))
b = make("LINE", (0, 0, 0), X)
r = mate(a, b)
check("face/line: line in the face", mated(a, b, r["M"]) and abs(float(r["R"] @ a["d"] @ X)) < 1e-12)

# --- points ---------------------------------------------------------------
a = make("POINT", (3, 4, 5))
r = mate(a, make("PLANE", (0, 0, 0), Z))
check("point/face: straight down", close(r["R"], np.eye(3)) and close(r["t"], (0, 0, -5)))
r = mate(a, make("LINE", (0, 0, 0), X))
check("point/line: onto the line", close(r["t"], (0, -4, -5)))
r = mate(a, make("AXIS", (0, 0, 0), Z, pin=True))
check("point/axis mid: onto the middle", close(r["t"], (-3, -4, -5)))
r = mate(a, make("POINT", (1, 1, 1)))
check("point/point", close(r["t"], (-2, -3, -4)) and r["relation"] == "none" and r["axis"] is None)

# edge middles: edges parallel, middles together
a = make("POINT", (2, 2, 0), hint=al._unit((1, 1, 0)))
b = make("POINT", (0, 0, 0), hint=X)
r = mate(a, b)
check("mid/mid: edges parallel", close(np.abs(r["R"] @ a["hint"]), X) and close(apply(r["M"], a["o"]), (0, 0, 0))
      and abs(r["angle"] - 45) < 1e-9)

# --- face edge runs line up (nearest quarter turn) ------------------------
for deg, want in ((20, 20), (70, -20), (-30, -30), (130, 40)):
    h = (math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0)
    a = make("PLANE", (0, 0, 3), -Z, hint=h)
    r = mate(a, make("PLANE", (0, 0, 0), Z, hint=X))
    hh = r["R"] @ a["hint"]
    check(f"face hints {deg} deg: edges in line, turn {want}", abs(abs(hh[0]) - 1) < 1e-9 or abs(abs(hh[1]) - 1) < 1e-9)
    check(f"face hints {deg} deg: smallest", abs(r["angle"] - abs(want)) < 1e-9, r["angle"])

# --- flip ----------------------------------------------------------------
a = make("PLANE", (5, 3, 1), Z, hint=X)
b = make("PLANE", (0, 0, 0), Z, hint=X)
r = mate(a, b, flip=True)
check("flip face/face: back to back", close(r["R"] @ Z, Z) and mated(a, b, r["M"]) and close(r["p"], (5, 3, 0)))
a = make("LINE", (1, 1, 1), (1, 0, 1))
b = make("PLANE", (0, 0, 0), Z)
r0, r1 = mate(a, b), mate(a, b, flip=True)
side = lambda r, q: float(apply(r["M"], q)[2])
q = (0, 1, 2)                                    # a point of the part off its line, across it
check("flip line/face: other side of the plane", side(r0, q) * side(r1, q) < 0 and mated(a, b, r1["M"]))
a = make("AXIS", (2, 0, 0), Z)
b = make("AXIS", (0, 0, 0), Z)
r = mate(a, b, flip=True)
check("flip axis/axis: other sense", close(r["R"] @ Z, -Z) and mated(a, b, r["M"]))

# --- spin ----------------------------------------------------------------
a = make("PLANE", (5, 3, 1), Z, hint=X)
b = make("PLANE", (0, 0, 0), Z, hint=X)
r = mate(a, b, spin=math.pi / 2)
check("spin: about the landed point", close(r["p"], (5, 3, 0)) and close(apply(r["M"], a["o"]), (5, 3, 0))
      and mated(a, b, r["M"]))
check("spin: quarter turn", close(np.abs(r["R"] @ X), Y))
r = mate(make("POINT", (1, 0, 0)), make("POINT", (0, 0, 0)), spin=1.0)
check("spin: nothing to turn about", close(r["R"], np.eye(3)))

# --- gap -----------------------------------------------------------------
r = mate(a, b, gap=0.5)
check("gap face/face: off b's face", close(apply(r["M"], a["o"]), (5, 3, 0.5)) and close(r["gap_dir"], Z))
a = make("RING", (10, 0, 0), X)
b = make("RING", (0, 0, 0), Z)
r = mate(a, b, gap=-0.25)
check("gap ring/ring: negative sinks in", close(apply(r["M"], a["o"]), (0, 0, -0.25)))
a = make("PLANE", (0, 0, 2), -Z)
b = make("LINE", (0, 0, 0), X)
r = mate(a, b, gap=1)
check("gap face/line: away from the line", close(apply(r["M"], a["o"]), (0, 0, 1)))

# --- second pair: the spin only ----------------------------------------------
th = math.radians(33)
Rz = al.rotation(Z, th)
a = make("PLANE", (5, 3, 0), -Z, hint=Rz @ X)      # a box's bottom, turned 33 deg
b = make("PLANE", (0, 0, 0), Z)                    # no hint: only the second pair turns it
a2 = make("PLANE", Rz @ (6, 3, 0.5), Rz @ X)       # the box's side face
b2 = make("PLANE", (2, 0, 0.5), Y)                 # the target's side face
r = mate(a, b, second=(a2, b2))
check("second pair: side faces the same way", close(r["R"] @ (Rz @ X), Y))
check("second pair: first mate kept", mated(a, b, r["M"]) and close(r["p"], (5, 3, 0)))
r = mate(make("RING", (0, 0, 0), -Z), make("RING", (0, 0, 0), Z),
         second=(make("POINT", (1, 0, 0)), make("POINT", (0, 2, 0))))
check("second pair: points round the axis", close(r["R"] @ X, Y) and close(r["R"] @ Z, Z))
r = mate(make("AXIS", (0, 0, 0), Z), make("AXIS", (0, 0, 0), Z),
         second=(make("LINE", (1, 0, 0), al._unit((1, 1, 0))), make("LINE", (0, 0, 0), -X)))
check("second pair: unsigned lines, nearer sense", abs(abs(r["spin"]) - math.pi / 4) < 1e-9, r["spin"])

# --- fuzz: every pair of types mates, rigid, also flipped and spun -----------
rng = np.random.default_rng(7)
types = [("POINT", False), ("POINT", True), ("LINE", False), ("PLANE", False), ("PLANE", True),
         ("RING", True), ("AXIS", False), ("AXIS", True)]
def rnd(t, pin):
    d = None if t == "POINT" else rng.normal(size=3)
    h = rng.normal(size=3) if (t == "POINT" and rng.random() < 0.5) or (t == "PLANE" and rng.random() < 0.5) else None
    if h is not None and d is not None:
        d = al._unit(d)
        h = h - (h @ d) * d
    return make(t, rng.normal(size=3) * 5, d, pin=pin, hint=h)
bad = []
for _ in range(60):
    for ta in types:
        for tb in types:
            a, b = rnd(*ta), rnd(*tb)
            second = (rnd(*types[rng.integers(len(types))]), rnd(*types[rng.integers(len(types))]))
            for kw in ({}, {"flip": True}, {"spin": float(rng.normal())}, {"flip": True, "spin": 2.0},
                       {"second": second}, {"flip": True, "second": second}):
                r = mate(a, b, **kw)
                if not (rigid(r["M"]) and mated(a, b, r["M"], 1e-8)):
                    bad.append((ta, tb, kw))
check("fuzz: all type pairs mate", not bad, bad[:5])

print("ALL PASS" if ok else "SOME FAILED")
sys.stdout.flush()
os._exit(0 if ok else 1)
