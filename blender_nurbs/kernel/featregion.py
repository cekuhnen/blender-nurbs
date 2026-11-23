"""The feature a face belongs to, for Move Face: the faces enclosed by an
inner loop of the face(s) they sit in -- a hole's wall, a pocket's walls and
floor, a boss's walls and top, an embossed letter. Pure topology on the
kernel's Brep (no Blender), so the add-on (hover, preview) and the worker
(the move itself) find the same faces.

A flood fill from the picked faces crosses every edge except one that leaves
the feature into its seat: an edge of an inner loop of the face on its other
side. So features inside the feature (a hole in a boss's top) come along, and
starting on a face that is no feature (a box's side, a plate's top) floods the
whole body and finds nothing."""


def inner_loop_edges(face):
    """The edges of the face's loops other than its outer one."""
    outer = {e for e, _ in face.boundary_outer_loop()}
    return {e for loop in face.boundary_loops() for e, _ in loop if e not in outer}


def rims(brep):
    """{edge: the face it is an inner-loop edge of} over the whole body."""
    out = {}
    for f in brep.faces():
        for e in inner_loop_edges(f):
            out[e] = f
    return out


def feature_region(brep, start, rim=None):
    """(region faces, seat faces, the rim edges between them) of the feature
    holding the faces start; None when they are no feature (the fill reached
    the whole body, or never met a seat)."""
    rim = rims(brep) if rim is None else rim
    region = set(start)
    stack = list(start)
    seats, edges = set(), set()
    while stack:
        F = stack.pop()
        for e in F.edges():
            for G in e.faces():
                if G == F or G in region:
                    continue
                if rim.get(e) == G:            # leaving the feature into the face it sits in
                    seats.add(G)
                    edges.add(e)
                    continue
                region.add(G)
                stack.append(G)
    seats -= region                            # a seat reached some other way is no seat
    edges = {e for e in edges if rim.get(e) in seats}
    if not seats or len(region) >= brep.face_count():
        return None
    return region, seats, edges
