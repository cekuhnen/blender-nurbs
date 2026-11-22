"""Writes the Move Pivot toolbar icon, blender_nurbs/icons/ops.nproto.move_pivot.dat.

    python3 tools/make_pivot_icon.py [preview.png]

A toolbar icon is Blender's triangle format ("VCO\\0", size, offset, then
uint8 x, y per corner and RGBA per corner, y up, 0..255 like the built-in
ones). The icon: a cube in the tool icons' white, the pivot (an orange ring
and dot) moved onto its lower left corner, and a dashed line back to where
it was (the cube's middle). With a path it also writes a PNG preview.
"""
import math
import os
import struct
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "blender_nurbs", "icons", "ops.nproto.move_pivot.dat")

WHITE = (229, 229, 229, 255)            # the built-in tool icons' stroke colour
GREY = (229, 229, 229, 150)
ORANGE = (255, 150, 40, 255)

tris = []       # [((x, y) * 3, colour)]


def quad(a, b, c, d, col):
    tris.append(((a, b, c), col))
    tris.append(((a, c, d), col))


def stroke(p, q, w, col):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    nx, ny = -dy / L * w / 2, dx / L * w / 2
    quad((p[0] + nx, p[1] + ny), (q[0] + nx, q[1] + ny), (q[0] - nx, q[1] - ny), (p[0] - nx, p[1] - ny), col)


def polyline(pts, w, col, closed=False):
    n = len(pts)
    for i in range(n if closed else n - 1):
        stroke(pts[i], pts[(i + 1) % n], w, col)
    for p in pts:                           # round the joins
        disc(p, w / 2, col, 8)


def disc(c, r, col, n=24):
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        tris.append(((c, (c[0] + r * math.cos(a0), c[1] + r * math.sin(a0)),
                      (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1))), col))


def ring(c, r0, r1, col, n=32):
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        P = lambda r, a: (c[0] + r * math.cos(a), c[1] + r * math.sin(a))
        quad(P(r0, a0), P(r1, a0), P(r1, a1), P(r0, a1), col)


def dashed(p, q, w, col, dash=13, gap=10):
    L = math.hypot(q[0] - p[0], q[1] - p[1])
    t = 0.0
    while t < L:
        t1 = min(L, t + dash)
        a = (p[0] + (q[0] - p[0]) * t / L, p[1] + (q[1] - p[1]) * t / L)
        b = (p[0] + (q[0] - p[0]) * t1 / L, p[1] + (q[1] - p[1]) * t1 / L)
        stroke(a, b, w, col)
        t = t1 + gap


# the cube: an upright hexagon and its three inner edges, the front corner's
# lower left vertex is where the pivot goes
cx, cy, R = 140.0, 132.0, 82.0
hexa = [(cx + R * math.cos(math.radians(90 + 60 * k)), cy + R * math.sin(math.radians(90 + 60 * k)))
        for k in range(6)]          # top, upper left, lower left, bottom, lower right, upper right
mid = (cx, cy)
polyline(hexa, 11, WHITE, closed=True)
for k in (1, 3, 5):                   # the inner Y: middle to upper left, bottom, upper right
    stroke(mid, hexa[k], 9, WHITE)
disc(mid, 4.5, WHITE, 12)

old = (cx + 10, cy - 34)            # where the pivot was: inside the cube
new = hexa[2]                       # where it goes: the lower left corner
ring(old, 10, 16, GREY)
dashed(new, old, 7, ORANGE)
ring(new, 18, 31, ORANGE)
disc(new, 9, ORANGE, 16)


def clamp8(v):
    return max(0, min(255, int(round(v))))


def write_dat(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    co = bytearray()
    col = bytearray()
    for pts, c in tris:
        for x, y in pts:
            co += bytes((clamp8(x), clamp8(y)))
            col += bytes(c)
    with open(path, "wb") as f:
        f.write(b"VCO\x00" + bytes((255, 255, 0, 0)) + bytes(co) + bytes(col))


def write_png(path, size=256, bg=(48, 48, 48)):
    """A plain rasteriser for the preview (y up as Blender draws it)."""
    img = [[list(bg) for _ in range(size)] for _ in range(size)]
    s = size / 256.0
    for pts, c in tris:
        if c[3] == 0:
            continue
        (x0, y0), (x1, y1), (x2, y2) = [(x * s, y * s) for x, y in pts]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        for py in range(max(0, int(min(y0, y1, y2))), min(size, int(max(y0, y1, y2)) + 2)):
            for px in range(max(0, int(min(x0, x1, x2))), min(size, int(max(x0, x1, x2)) + 2)):
                X, Y = px + 0.5, py + 0.5
                a = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
                b = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
                if a >= -1e-6 and b >= -1e-6 and a + b <= 1 + 1e-6:
                    row = img[size - 1 - py]
                    al = c[3] / 255.0
                    row[px] = [int(row[px][k] * (1 - al) + c[k] * al) for k in range(3)]
    raw = b"".join(b"\x00" + bytes(v for p in row for v in p) for row in img)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    write_dat(OUT)
    print(OUT, len(tris), "triangles")
    if len(sys.argv) > 1:
        write_png(sys.argv[1])
