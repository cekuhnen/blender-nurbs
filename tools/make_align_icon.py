"""Writes the Align Parts toolbar icon, blender_nurbs/icons/ops.nproto.align_parts.dat.

    python3 tools/make_align_icon.py [preview.png]

Blender's toolbar icon format as tools/make_pivot_icon.py writes it (its
helpers are reused). The icon: a peg (a cylinder in the tool icons' white)
over a slab, its end ring and the slab's ring in orange, a dashed arrow
from one to the other: the element of one part going onto the other's.
"""
import importlib.util
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "blender_nurbs", "icons", "ops.nproto.align_parts.dat")

_spec = importlib.util.spec_from_file_location("make_pivot_icon", os.path.join(ROOT, "tools", "make_pivot_icon.py"))
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)        # (draws the pivot icon into pv.tris: start afresh)
pv.tris.clear()
WHITE, ORANGE = pv.WHITE, pv.ORANGE


def ellipse(c, rx, ry, n=40):
    """A closed ellipse's points (the first not repeated)."""
    return [(c[0] + rx * math.cos(2 * math.pi * k / n), c[1] + ry * math.sin(2 * math.pi * k / n))
            for k in range(n)]


# the slab: its top face as a flat parallelogram and its front side
sl = [(28, 70), (178, 70), (228, 108), (78, 108)]
pv.polyline(sl, 10, WHITE, closed=True)
pv.polyline([(28, 70), (28, 44), (178, 44), (228, 82), (228, 108)], 10, WHITE)
pv.stroke((178, 70), (178, 44), 10, WHITE)
target = (128, 89)
pv.polyline(ellipse(target, 34, 11), 8, ORANGE, closed=True)

# the peg above: its sides, its top, and its end ring (orange) at the bottom
c_top, c_bot, rx, ry = (128, 226), (128, 160), 34, 11
pv.polyline(ellipse(c_top, rx, ry), 10, WHITE, closed=True)
pv.stroke((c_top[0] - rx, c_top[1]), (c_bot[0] - rx, c_bot[1]), 10, WHITE)
pv.stroke((c_top[0] + rx, c_top[1]), (c_bot[0] + rx, c_bot[1]), 10, WHITE)
pv.polyline(ellipse(c_bot, rx, ry), 8, ORANGE, closed=True)

# the dashed arrow: from the peg's end ring down to the slab's
pv.dashed((128, 146), (128, 116), 7, ORANGE, dash=9, gap=7)
pv.tris.append((((114, 118), (142, 118), (128, 101)), ORANGE))

if __name__ == "__main__":
    pv.write_dat(OUT)
    print(OUT, len(pv.tris), "triangles")
    if len(sys.argv) > 1:
        pv.write_png(sys.argv[1])
