"""Plywood: a bent plywood shell textured the way the real part is -- a veneer
sheet on the front and one on the back, and the band of plies where the shell
is cut along its rim.

The shell is a NURBS solid of even thickness (as Thicken makes it) or a plain
mesh. A NURBS object is textured on its own B-rep faces (facetex: the sorting,
materials and UVs are made again at every rebuild) or meshed into a separate
render mesh like BREP to Mesh; a mesh is textured in place. Every polygon is
sorted:

  * a veneer face has the other side of the shell straight behind it, at the
    thickness, facing the other way (a ray back through the shell meets it);
  * the faces left over are the edge: the rim walls, and any rounding or
    chamfer of the rim, where the plies show.

The veneer faces split into the two sides (they meet only through the edge);
the front is the one facing up, or the other one with Swap Sides.

UVs, in metres:
  * each side is unwrapped as ONE sheet (facetex.unwrap: conformal, then as
    rigid as possible), the way a veneer is bent over the form, so the grain
    runs on across the patch boundaries; the sheet is turned so the grain runs
    along its length (plus Grain Angle);
  * the edge band gets u = length along the rim, v = depth through the shell
    from the front (0) to the back (1): any edge texture spans the plies
    exactly, however thick the shell and however the rim is rounded.

Three material slots: Front veneer, Back veneer (the front's own material
unless Back Veneer says otherwise) and Edge: plies counted from the object's
"ply_count" property, the outer plies in the veneers' colours, thin glue lines
between, or an edge photo. Each slot can hold the user's own material instead:
named in the operator, or put in the slot by hand (kept by later runs)."""

import json
import math

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import brep_mesh, store

PROP = "nproto_plywood"           # settings of the last run (JSON), on the plywood mesh
PROP_SRC = "nproto_ply_src"       # the NURBS object it was meshed from (an ID property)
FRONT, BACK, EDGE = 0, 1, 2
UV_NAME = "UVMap"

# (light, dark, pore) colours, linear
SPECIES = {
    "BIRCH": ((0.70, 0.52, 0.33), (0.55, 0.38, 0.22), (0.45, 0.30, 0.17)),
    "BEECH": ((0.72, 0.46, 0.28), (0.58, 0.34, 0.19), (0.46, 0.26, 0.14)),
    "MAPLE": ((0.78, 0.62, 0.44), (0.66, 0.50, 0.33), (0.55, 0.40, 0.26)),
    "ASH": ((0.70, 0.56, 0.38), (0.45, 0.33, 0.20), (0.30, 0.21, 0.12)),
    "OAK": ((0.55, 0.37, 0.19), (0.36, 0.22, 0.10), (0.24, 0.14, 0.06)),
    "WALNUT": ((0.23, 0.12, 0.06), (0.10, 0.05, 0.025), (0.06, 0.03, 0.015)),
    "TEAK": ((0.42, 0.22, 0.08), (0.26, 0.13, 0.045), (0.16, 0.08, 0.03)),
}
_SPECIES_ITEMS = [(k, k.title(), f"{k.title()} veneer") for k in SPECIES]
GLUE = (0.09, 0.055, 0.03)


# -- geometry --------------------------------------------------------------
def _arrays(me, M):
    """World vertices, polygon corner lists, per-polygon (centre, outward unit
    normal, area), and the corner -> polygon map."""
    nv, npoly, nl = len(me.vertices), len(me.polygons), len(me.loops)
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    P = co.reshape(-1, 3) @ np.array(M.to_3x3()).T + np.array(M.translation)
    ls = np.empty(npoly, np.int64)
    lt = np.empty(npoly, np.int64)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(nl, np.int64)
    me.loops.foreach_get("vertex_index", lv)
    le = np.empty(nl, np.int64)
    me.loops.foreach_get("edge_index", le)
    lp = np.repeat(np.arange(npoly), lt)
    # Newell normals (area vectors) in world space
    nxt = np.arange(nl) + 1
    ends = ls + lt
    nxt[ends - 1] = ls
    a, b = P[lv], P[lv[nxt]]
    cr = np.cross(a, b)
    Av = np.zeros((npoly, 3))
    np.add.at(Av, lp, cr)
    Av *= 0.5
    area = np.linalg.norm(Av, axis=1)
    N = Av / np.maximum(area, 1e-30)[:, None]
    # outward: agree with the mesh's corner normals (exact, from the B-rep, when it has them)
    try:
        cnr = np.empty(nl * 3)
        me.corner_normals.foreach_get("vector", cnr)
        cn = cnr.reshape(-1, 3) @ np.array(M.to_3x3().inverted_safe().transposed()).T
        Nc = np.zeros((npoly, 3))
        np.add.at(Nc, lp, cn)
        flip = np.einsum("ij,ij->i", Nc, N) < 0
        N[flip] *= -1
    except Exception:
        pass
    C = np.zeros((npoly, 3))
    np.add.at(C, lp, P[lv])
    C /= lt[:, None]
    polys = [lv[s:s + t].tolist() for s, t in zip(ls, lt)]
    return P, polys, C, N, area, lv, le, lp


def _ray_back(bvh, c, n, own, size):
    """First polygon behind polygon `own` (along -n from its centre), not itself:
    (index, distance) or (None, inf)."""
    d0 = 0.0
    o = Vector(c - n * 1e-6 * size)
    dv = Vector(-n)
    for _ in range(4):
        loc, _, idx, d = bvh.ray_cast(o, dv)
        if idx is None:
            return None, math.inf
        d0 += d
        if idx != own:
            return idx, d0
        o = loc + dv * (1e-6 * size)
        d0 += 1e-6 * size
    return None, math.inf


def _weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def classify(me, M, swap=False, up=(0.0, 0.0, 1.0), face_attr="brep_face"):
    """Polygon classes FRONT / BACK / EDGE and the shell's thickness (world
    units). A polygon is veneer when the ray back through the shell from its
    centre meets the other side at about the thickness, facing the other way;
    with a "brep_face" attribute each B-rep face goes by its polygons' (area-)
    majority. The veneer polygons fall into sides by connection; sides with a
    polygon straight behind each other are opposite."""
    P, polys, C, N, area, lv, le, lp = _arrays(me, M)
    npoly = len(polys)
    if npoly == 0:
        raise ValueError("the mesh has no faces")
    size = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0
    bvh = BVHTree.FromPolygons([tuple(p) for p in P], polys, all_triangles=False, epsilon=0.0)
    hit = np.full(npoly, -1, np.int64)
    dist = np.full(npoly, np.inf)
    for i in range(npoly):
        if area[i] <= 1e-14 * size * size:
            continue
        j, d = _ray_back(bvh, C[i], N[i], i, size)
        if j is not None:
            hit[i], dist[i] = j, d
    opp = (hit >= 0) & (np.einsum("ij,ij->i", N, N[np.maximum(hit, 0)]) < -0.9)
    if not opp.any():
        raise ValueError("no two faces lie opposite each other: not a shell with a thickness")
    t = _weighted_median(dist[opp], area[opp])
    skin = opp & (dist > 0.6 * t) & (dist < 1.6 * t)
    att = me.attributes.get(face_attr)
    if att is not None and att.domain == 'FACE' and len(att.data) == npoly:
        fid = np.empty(npoly, np.int64)
        att.data.foreach_get("value", fid)
        for f in np.unique(fid):
            sel = fid == f
            yes = area[sel & skin].sum()
            skin[sel] = yes > 0.5 * area[sel].sum()
    # sides: veneer polygons joined across shared mesh edges
    parent = np.arange(npoly)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_edge = {}
    for c in np.flatnonzero(skin[lp]):
        by_edge.setdefault(int(le[c]), []).append(int(lp[c]))
    for ps in by_edge.values():
        for q in ps[1:]:
            a, b = find(ps[0]), find(q)
            if a != b:
                parent[b] = a
    comp = np.array([find(i) if skin[i] else -1 for i in range(npoly)])
    roots = [r for r in np.unique(comp) if r >= 0]
    # two-colour the sides: a side and the one behind it are opposite
    links = {}
    for i in np.flatnonzero(skin):
        j = hit[i]
        if j >= 0 and skin[j] and comp[j] != comp[i]:
            k = (comp[i], comp[j])
            links[k] = links.get(k, 0.0) + area[i]
    colour = {}
    order = sorted(roots, key=lambda r: -area[comp == r].sum())
    for r0 in order:
        if r0 in colour:
            continue
        colour[r0] = 0
        stack = [r0]
        while stack:
            r = stack.pop()
            for (a, b), w in links.items():
                if a == r and b not in colour:
                    colour[b] = 1 - colour[r]
                    stack.append(b)
    cls = np.full(npoly, EDGE, np.int64)
    for r, c in colour.items():
        cls[comp == r] = c
    if not (cls == BACK).any():
        raise ValueError("found only one veneer side: is the object a closed shell with a thickness?")
    # the front faces up (most of its area), unless swapped
    u = np.asarray(up, float)
    s0 = float((area[cls == 0, None] * N[cls == 0]).sum(axis=0) @ u)
    s1 = float((area[cls == 1, None] * N[cls == 1]).sum(axis=0) @ u)
    if (s1 > s0) != bool(swap):
        f0, b0 = cls == FRONT, cls == BACK
        cls[f0], cls[b0] = BACK, FRONT
    return cls, _thickness(P, polys, lv, lp, cls, t)


def _thickness(P, polys, lv, lp, cls, rough):
    """The shell's thickness from its vertices (on the surface) to the other
    side's polygons, both ways: a curved shell's polygons cut inside its convex
    side and outside its concave side by about the same chord, so the two
    medians err opposite ways and their mean is close."""
    meds = []
    for a, b in ((FRONT, BACK), (BACK, FRONT)):
        tb = BVHTree.FromPolygons([tuple(p) for p in P], [polys[i] for i in np.flatnonzero(cls == b)], epsilon=0.0)
        vs = np.unique(lv[cls[lp] == a])
        d = _nearest_dist(tb, P[vs])
        d = d[np.isfinite(d) & (d < 2.0 * rough)]
        if len(d):
            meds.append(float(np.median(d)))
    return float(np.mean(meds)) if meds else rough


# -- UVs -------------------------------------------------------------------
def _rim_loops(P, lv, le, lp, cls, me):
    """Closed polylines (world points) along the front veneer's boundary,
    where the edge band starts."""
    ne = len(me.edges)
    ev = np.empty(ne * 2, np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    front_uses = np.zeros(ne, np.int64)
    np.add.at(front_uses, le[cls[lp] == FRONT], 1)
    bnd = np.flatnonzero(front_uses == 1)
    nb = {}
    for e in bnd:
        a, b = (int(x) for x in ev[e])
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in nb:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [x for x in nb[cur] if x != prev and (x not in seen or (x == start and len(loop) > 2))]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                loop.append(start)
                break
            loop.append(cur)
            seen.add(cur)
        if len(loop) > 2:
            loops.append(P[loop])
    return loops


def _project_on_loops(Q, loops, batch=512):
    """Arc length (with each loop's own offset) and loop index of the nearest
    point on the loops, for every point of Q; and each loop's (offset, length)."""
    segA, segB, segS, segL = [], [], [], []
    spans, off = [], 0.0
    for k, L in enumerate(loops):
        A, B = L[:-1], L[1:]
        ln = np.linalg.norm(B - A, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(ln)[:-1]])
        segA.append(A)
        segB.append(B)
        segS.append(off + cum)
        segL.append(np.full(len(A), k))
        total = float(ln.sum())
        spans.append((off, total))
        off += total + 0.05 * max(total, 1e-6)
    A, B = np.vstack(segA), np.vstack(segB)
    S0, K = np.concatenate(segS), np.concatenate(segL)
    D = B - A
    dd = np.maximum(np.einsum("ij,ij->i", D, D), 1e-30)
    ln = np.sqrt(dd)
    U = np.empty(len(Q))
    which = np.empty(len(Q), np.int64)
    for s in range(0, len(Q), batch):
        q = Q[s:s + batch]
        w = np.clip(np.einsum("qij,ij->qi", q[:, None, :] - A[None], D) / dd, 0.0, 1.0)
        X = A[None] + w[..., None] * D[None]
        dist = np.linalg.norm(q[:, None, :] - X, axis=2)
        j = np.argmin(dist, axis=1)
        U[s:s + batch] = S0[j] + w[np.arange(len(q)), j] * ln[j]
        which[s:s + batch] = K[j]
    return U, which, spans


def _nearest_dist(bvh, Q):
    out = np.empty(len(Q))
    for i, q in enumerate(Q):
        r = bvh.find_nearest(Vector(q))
        out[i] = r[3] if r[0] is not None else np.inf
    return out


def make_uvs(me, M, cls, grain_angle=0.0):
    """Veneer sheets in metres with the grain along each sheet's length, the
    edge band as (length along the rim, depth 0..1 from the front)."""
    from . import facetex
    P, polys, C, N, area, lv, le, lp = _arrays(me, M)
    npoly, nl = len(polys), len(lv)
    ls = np.empty(npoly, np.int64)
    lt = np.empty(npoly, np.int64)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    # each veneer side one sheet, unwrapped conformally (true size, long axis along u)
    island = np.where(cls == EDGE, -1, cls)
    UV, piece = facetex.unwrap(P, lv, ls, lt, island)
    lcls = cls[lp]

    # the edge band: depth from the front veneer, length along its rim
    edge_c = np.flatnonzero(lcls == EDGE)
    band_h = 0.0
    if len(edge_c):
        def side_bvh(c):
            ids = np.flatnonzero(cls == c)
            return BVHTree.FromPolygons([tuple(p) for p in P], [polys[i] for i in ids], epsilon=0.0)
        bf, bb = side_bvh(FRONT), side_bvh(BACK)
        verts = np.unique(lv[edge_c])
        df, db = _nearest_dist(bf, P[verts]), _nearest_dist(bb, P[verts])
        depth = np.zeros(len(P))
        depth[verts] = np.clip(df / np.maximum(df + db, 1e-30), 0.0, 1.0)
        loops = _rim_loops(P, lv, le, lp, cls, me)
        if loops:
            Uv, which, spans = _project_on_loops(P[verts], loops)
            along = np.zeros(len(P))
            loop_of = np.zeros(len(P), np.int64)
            along[verts], loop_of[verts] = Uv, which
            UV[edge_c, 0] = along[lv[edge_c]]
            UV[edge_c, 1] = depth[lv[edge_c]]
            # a polygon across a loop's start: its corners on one side of the jump
            for i in np.unique(lp[edge_c]):
                cs = np.arange(ls[i], ls[i] + lt[i])
                k = loop_of[lv[cs[0]]]
                off, total = spans[k]
                u = UV[cs, 0]
                if u.max() - u.min() > 0.5 * total:
                    UV[cs[u < off + 0.5 * total], 0] += total
        else:
            UV[edge_c, 0] = 0.0
            UV[edge_c, 1] = depth[lv[edge_c]]
        band_h = 1.0

    # the veneer sheets: grain along the length (plus the grain angle), front above the band, back beside it
    x0 = 0.0
    for c in (FRONT, BACK):
        rows = np.flatnonzero((lcls == c) & (piece >= 0))
        if not len(rows):
            continue
        for p in np.unique(piece[rows]):
            facetex.turn(UV, rows[piece[rows] == p], grain_angle)
        sub = np.full(len(UV), -1, np.int64)
        sub[rows] = piece[rows]
        facetex.shelf(UV, sub, origin=(x0, band_h + 0.05))
        x0 = float(UV[rows, 0].max()) + 0.05
    facetex.set_uvs(me, UV)
    me.update()


# -- materials -------------------------------------------------------------
def _node(nt, kind, name, loc, **attrs):
    n = nt.nodes.get(name)
    if n is None or n.bl_idname != kind:
        n = nt.nodes.new(kind)
        n.name = n.label = name
    n.location = loc
    for k, v in attrs.items():
        setattr(n, k, v)
    return n


def _math(nt, name, op, loc, a=None, b=None, c=None):
    n = _node(nt, "ShaderNodeMath", name, loc, operation=op)
    for k, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, n.inputs[k])
        else:
            n.inputs[k].default_value = v
    return n.outputs[0]


def _mix_col(nt, name, loc, fac, a, b, blend='MIX'):
    n = _node(nt, "ShaderNodeMix", name, loc, data_type='RGBA', blend_type=blend)
    for sock, v in ((n.inputs["Factor"], fac), (n.inputs[6], a), (n.inputs[7], b)):
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, sock)
        elif isinstance(v, (int, float)):
            sock.default_value = float(v)
        else:
            sock.default_value = tuple(v) if len(v) == 4 else (*v, 1.0)
    return n.outputs[2]


def _set_ramp(ramp, cols):
    els = ramp.color_ramp.elements
    while len(els) > 2:
        els.remove(els[-1])
    pos = [0.0, 1.0] if len(cols) == 2 else [i / (len(cols) - 1) for i in range(len(cols))]
    while len(els) < len(cols):
        els.new(0.5)
    for e, p, c in zip(els, pos, cols):
        e.position = p
        e.color = (*c, 1.0)


def _new_material(name):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except AttributeError:
        pass
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    return mat


def veneer_material(name, species, image=None, image_width=0.5):
    """Procedural veneer (grain along u, UVs in metres), or an image covering
    `image_width` metres along the grain."""
    mat = _new_material(name)
    nt = mat.node_tree
    out = _node(nt, "ShaderNodeOutputMaterial", "Output", (900, 0))
    bsdf = _node(nt, "ShaderNodeBsdfPrincipled", "Veneer BSDF", (600, 0))
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    tc = _node(nt, "ShaderNodeTexCoord", "Coordinates", (-1200, 0))
    uvm = _node(nt, "ShaderNodeUVMap", "Veneer UV", (-1200, -200), uv_map=UV_NAME)
    # figure: bands across v (grain along u), warped by noise stretched along u
    sep = _node(nt, "ShaderNodeSeparateXYZ", "Split", (-1000, 0))
    nt.links.new(uvm.outputs[0], sep.inputs[0])
    xs = _math(nt, "Along Grain", 'MULTIPLY', (-800, 100), sep.outputs[0], 0.12)
    comb = _node(nt, "ShaderNodeCombineXYZ", "Grain Space", (-600, 0))
    nt.links.new(xs, comb.inputs[0])
    nt.links.new(sep.outputs[1], comb.inputs[1])
    wave = _node(nt, "ShaderNodeTexWave", "Figure", (-400, 100), wave_type='BANDS', bands_direction='Y')
    nt.links.new(comb.outputs[0], wave.inputs["Vector"])
    wave.inputs["Scale"].default_value = 18.0
    wave.inputs["Distortion"].default_value = 9.0
    wave.inputs["Detail"].default_value = 3.0
    wave.inputs["Detail Scale"].default_value = 1.5
    ramp = _node(nt, "ShaderNodeValToRGB", "Veneer Colours", (-150, 100))
    nt.links.new(wave.outputs["Fac"], ramp.inputs[0])
    # fibres: fine streaks along u
    fs = _node(nt, "ShaderNodeVectorMath", "Fibre Space", (-600, -250), operation='MULTIPLY')
    nt.links.new(uvm.outputs[0], fs.inputs[0])
    fs.inputs[1].default_value = (6.0, 400.0, 1.0)
    fib = _node(nt, "ShaderNodeTexNoise", "Fibres", (-400, -250))
    nt.links.new(fs.outputs[0], fib.inputs["Vector"])
    fib.inputs["Scale"].default_value = 1.0
    fib.inputs["Detail"].default_value = 6.0
    fr = _node(nt, "ShaderNodeMapRange", "Fibre Strength", (-150, -250))
    nt.links.new(fib.outputs["Fac"], fr.inputs[0])
    fr.inputs[3].default_value, fr.inputs[4].default_value = 0.82, 1.08
    col = _mix_col(nt, "Fibre Tone", (100, 50), 1.0, ramp.outputs[0], (1, 1, 1, 1), blend='MULTIPLY')
    nt.links.new(fr.outputs[0], col.node.inputs[7])
    # an image instead, when given
    img = _node(nt, "ShaderNodeTexImage", "Veneer Image", (-150, -500))
    isc = _node(nt, "ShaderNodeVectorMath", "Image Scale", (-400, -500), operation='SCALE')
    nt.links.new(uvm.outputs[0], isc.inputs[0])
    nt.links.new(isc.outputs[0], img.inputs["Vector"])
    use_img = _node(nt, "ShaderNodeValue", "Use Image", (100, -400))
    pick = _mix_col(nt, "Figure or Image", (350, 0), use_img.outputs[0], col, img.outputs["Color"])
    nt.links.new(pick, bsdf.inputs["Base Color"])
    # surface: satin lacquer, fibres in the bump
    bsdf.inputs["Roughness"].default_value = 0.42
    for key in ("Coat Weight", "Coat"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = 0.25
            break
    bump = _node(nt, "ShaderNodeBump", "Fibre Bump", (350, -300))
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(fib.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    set_veneer(mat, species, image, image_width)
    return mat


def set_veneer(mat, species, image=None, image_width=0.5):
    nt = mat.node_tree
    if mat.get("nproto_species") != species and "Veneer Colours" in nt.nodes:
        light, dark, _pore = SPECIES[species]
        _set_ramp(nt.nodes["Veneer Colours"], [light, dark])
        mat["nproto_species"] = species
        mat.diffuse_color = (*light, 1.0)
    img_node = nt.nodes.get("Veneer Image")
    if img_node is not None:
        if image:
            try:
                img_node.image = bpy.data.images.load(bpy.path.abspath(image), check_existing=True)
            except Exception as exc:
                print("Plywood veneer image:", exc)
                image = None
        nt.nodes["Use Image"].outputs[0].default_value = 1.0 if image else 0.0
        nt.nodes["Image Scale"].inputs["Scale"].default_value = 1.0 / max(image_width, 1e-4)


def edge_material(name, front, back, core):
    """Ply stripes across the band's v (0 front .. 1 back): "ply_count" plies
    (an object property), the outer two in the veneers' colours, the core plies
    alternating long grain and cross grain, glue lines between."""
    mat = _new_material(name)
    nt = mat.node_tree
    out = _node(nt, "ShaderNodeOutputMaterial", "Output", (1500, 0))
    bsdf = _node(nt, "ShaderNodeBsdfPrincipled", "Edge BSDF", (1200, 0))
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = 0.6
    uvm = _node(nt, "ShaderNodeUVMap", "Edge UV", (-1400, 0), uv_map=UV_NAME)
    sep = _node(nt, "ShaderNodeSeparateXYZ", "Split", (-1200, 0))
    nt.links.new(uvm.outputs[0], sep.inputs[0])
    cnt = _node(nt, "ShaderNodeAttribute", "Ply Count", (-1400, 300), attribute_type='OBJECT',
                attribute_name="ply_count")
    glue_w = _node(nt, "ShaderNodeAttribute", "Glue Width", (-1400, 500), attribute_type='OBJECT',
                   attribute_name="ply_glue")
    n_ = _math(nt, "Plies", 'MAXIMUM', (-1200, 300), cnt.outputs["Fac"], 1.0)
    v = _math(nt, "Depth", 'MINIMUM', (-1000, 0), _math(nt, "Depth0", 'MAXIMUM', (-1100, 0), sep.outputs[1], 0.0),
              0.99999)
    d = _math(nt, "Ply Coordinate", 'MULTIPLY', (-850, 0), v, n_)
    idx = _math(nt, "Ply", 'FLOOR', (-700, 100), d)
    frac = _math(nt, "In Ply", 'SUBTRACT', (-700, -100), d, idx)
    par = _math(nt, "Cross Grain", 'FLOORED_MODULO', (-550, 200), idx, 2.0)
    top = _math(nt, "Is Front Ply", 'LESS_THAN', (-550, 350), idx, 0.5)
    last = _math(nt, "Last", 'SUBTRACT', (-700, 450), n_, 1.5)
    bot = _math(nt, "Is Back Ply", 'GREATER_THAN', (-550, 500), idx, last)
    # per ply tone
    wn = _node(nt, "ShaderNodeTexWhiteNoise", "Ply Tone", (-400, 300), noise_dimensions='1D')
    nt.links.new(idx, wn.inputs["W"])
    tone = _node(nt, "ShaderNodeMapRange", "Tone Range", (-200, 300))
    nt.links.new(wn.outputs["Value"], tone.inputs[0])
    tone.inputs[3].default_value, tone.inputs[4].default_value = 0.85, 1.12
    # texture inside a ply: long-grain streaks along u, end-grain pores
    vs = _node(nt, "ShaderNodeCombineXYZ", "Ply Space", (-550, -250))
    nt.links.new(sep.outputs[0], vs.inputs[0])
    nt.links.new(d, vs.inputs[1])
    nt.links.new(idx, vs.inputs[2])
    st_sc = _node(nt, "ShaderNodeVectorMath", "Streak Space", (-350, -200), operation='MULTIPLY')
    nt.links.new(vs.outputs[0], st_sc.inputs[0])
    st_sc.inputs[1].default_value = (8.0, 6.0, 3.7)
    streak = _node(nt, "ShaderNodeTexNoise", "Streaks", (-150, -200))
    nt.links.new(st_sc.outputs[0], streak.inputs["Vector"])
    streak.inputs["Detail"].default_value = 5.0
    po_sc = _node(nt, "ShaderNodeVectorMath", "Pore Space", (-350, -450), operation='MULTIPLY')
    nt.links.new(vs.outputs[0], po_sc.inputs[0])
    po_sc.inputs[1].default_value = (900.0, 12.0, 3.7)
    pore = _node(nt, "ShaderNodeTexNoise", "Pores", (-150, -450))
    nt.links.new(po_sc.outputs[0], pore.inputs["Vector"])
    pore.inputs["Detail"].default_value = 2.0
    tex = _node(nt, "ShaderNodeMix", "Grain Texture", (50, -300), data_type='FLOAT')
    nt.links.new(par, tex.inputs["Factor"])
    nt.links.new(streak.outputs["Fac"], tex.inputs[2])
    nt.links.new(pore.outputs["Fac"], tex.inputs[3])
    texr = _node(nt, "ShaderNodeMapRange", "Texture Strength", (250, -300))
    nt.links.new(tex.outputs[0], texr.inputs[0])
    texr.inputs[3].default_value, texr.inputs[4].default_value = 0.8, 1.1
    # colours
    c_long = _node(nt, "ShaderNodeRGB", "Core Long Grain", (-200, 700))
    c_cross = _node(nt, "ShaderNodeRGB", "Core Cross Grain", (-200, 900))
    c_front = _node(nt, "ShaderNodeRGB", "Front Ply", (-200, 1100))
    c_back = _node(nt, "ShaderNodeRGB", "Back Ply", (-200, 1300))
    c_glue = _node(nt, "ShaderNodeRGB", "Glue", (500, 900))
    core_c = _mix_col(nt, "Core", (100, 600), par, c_long.outputs[0], c_cross.outputs[0])
    with_f = _mix_col(nt, "With Front", (300, 600), top, core_c, c_front.outputs[0])
    with_b = _mix_col(nt, "With Back", (500, 600), bot, with_f, c_back.outputs[0])
    toned = _mix_col(nt, "Toned", (700, 400), 1.0, with_b, (1, 1, 1, 1), blend='MULTIPLY')
    nt.links.new(tone.outputs[0], toned.node.inputs[7])
    textured = _mix_col(nt, "Textured", (850, 200), 1.0, toned, (1, 1, 1, 1), blend='MULTIPLY')
    nt.links.new(texr.outputs[0], textured.node.inputs[7])
    # glue lines at the ply boundaries
    dist = _math(nt, "To Boundary", 'MINIMUM', (-500, -700), frac, _math(nt, "1-f", 'SUBTRACT', (-650, -750), 1.0, frac))
    gw = _math(nt, "Glue Half", 'MULTIPLY', (-500, -900), _math(nt, "Glue Default", 'MAXIMUM', (-650, -950),
                                                                glue_w.outputs["Fac"], 0.0), 0.5)
    gw_safe = _math(nt, "Glue Floor", 'MAXIMUM', (-350, -900), gw, 1e-4)
    inner = _math(nt, "Inner Line", 'GREATER_THAN', (-350, -1050), d, 0.5)  # no line on the outer faces
    inner2 = _math(nt, "Inner Line 2", 'LESS_THAN', (-350, -1200), d, _math(nt, "N-0.5", 'SUBTRACT', (-500, -1200), n_, 0.5))
    gm = _node(nt, "ShaderNodeMapRange", "Glue Line", (-150, -800), interpolation_type='SMOOTHSTEP')
    nt.links.new(dist, gm.inputs[0])
    nt.links.new(gw_safe, gm.inputs[2])
    gm.inputs[1].default_value = 0.0
    gm.inputs[3].default_value, gm.inputs[4].default_value = 1.0, 0.0
    g1 = _math(nt, "Glue Mask", 'MULTIPLY', (50, -800), gm.outputs[0], _math(nt, "Inner", 'MULTIPLY', (-150, -1100), inner, inner2))
    final = _mix_col(nt, "With Glue", (1000, 200), g1, textured, c_glue.outputs[0])
    # a photo of the ply edge instead, when given: its height across the thickness, its width
    # along the rim in proportion ("ply_thickness", an object property, in metres)
    thick = _node(nt, "ShaderNodeAttribute", "Thickness", (-1400, -1400), attribute_type='OBJECT',
                  attribute_name="ply_thickness")
    aspect = _node(nt, "ShaderNodeValue", "Edge Image Aspect", (-1400, -1600))
    span = _math(nt, "Image Span", 'MULTIPLY', (-1200, -1450), _math(nt, "Thickness Floor", 'MAXIMUM', (-1250, -1350),
                                                                   thick.outputs["Fac"], 1e-4), aspect.outputs[0])
    ix = _math(nt, "Image U", 'DIVIDE', (-1000, -1400), sep.outputs[0], span)
    isp = _node(nt, "ShaderNodeCombineXYZ", "Edge Image Space", (-800, -1400))
    nt.links.new(ix, isp.inputs[0])
    nt.links.new(sep.outputs[1], isp.inputs[1])
    eimg = _node(nt, "ShaderNodeTexImage", "Edge Image", (-600, -1400))
    nt.links.new(isp.outputs[0], eimg.inputs["Vector"])
    use_e = _node(nt, "ShaderNodeValue", "Use Edge Image", (900, -100))
    pick = _mix_col(nt, "Plies or Image", (1100, 100), use_e.outputs[0], final, eimg.outputs["Color"])
    nt.links.new(pick, bsdf.inputs["Base Color"])
    set_edge(mat, front, back, core)
    return mat


def set_edge(mat, front, back, core, image=None):
    nt = mat.node_tree
    img_node = nt.nodes.get("Edge Image")
    if img_node is not None and image is not None:
        if image:
            try:
                img_node.image = bpy.data.images.load(bpy.path.abspath(image), check_existing=True)
                w, h = img_node.image.size
                nt.nodes["Edge Image Aspect"].outputs[0].default_value = (w / h) if h else 1.0
            except Exception as exc:
                print("Plywood edge image:", exc)
                image = ""
        nt.nodes["Use Edge Image"].outputs[0].default_value = 1.0 if image else 0.0
    key = f"{front}/{back}/{core}"
    if mat.get("nproto_species") == key:
        return
    put = lambda n, c: setattr(nt.nodes[n].outputs[0], "default_value", (*c, 1.0))
    # the edge of a ply shows its cut: a little lighter than its face figure, end grain darker
    put("Core Long Grain", SPECIES[core][0])
    put("Core Cross Grain", SPECIES[core][1])
    put("Front Ply", SPECIES[front][0])
    put("Back Ply", SPECIES[back][0])
    put("Glue", GLUE)
    mat["nproto_species"] = key
    mat.diffuse_color = (*SPECIES[core][0], 1.0)


# -- the operator ----------------------------------------------------------
def _is_plywood(obj):
    """A render mesh, a textured mesh or a textured NURBS object of this tool."""
    return obj is not None and obj.type == 'MESH' and PROP in obj


def _source(obj):
    s = obj.get(PROP_SRC) if obj is not None else None
    return s if isinstance(s, bpy.types.Object) and store.is_nurbs(s) else None


def _render_mesh(src):
    return next((o for o in bpy.data.objects if _is_plywood(o) and _source(o) == src), None)


def _auto_plies(t, ply_mm=1.5):
    n = max(3, int(round(t * 1000.0 / ply_mm)))
    return n if n % 2 else n + 1


def _ours(mat, role):
    return mat is not None and mat.get("nproto_ply") == role


class NPROTO_OT_plywood(bpy.types.Operator):
    """Texture a bent plywood shell for rendering: a veneer on the front and on
    the back, each unwrapped as one sheet with the grain along it, and the ply
    layers in a band along the rim. A NURBS object is textured on its own B-rep
    faces (kept through every edit) or meshed for rendering; a mesh is textured
    in place. Each part can take your own material. Run it again to update"""
    bl_idname = "nproto.plywood"
    bl_label = "Plywood"
    bl_options = {'REGISTER', 'UNDO'}

    target: EnumProperty(
        name="Texture", default="OBJECT",
        items=[("OBJECT", "NURBS Object", "Materials and UVs on the NURBS object's own B-rep faces: it stays "
                                          "editable and keeps them through every rebuild"),
               ("MESH", "Render Mesh", "A separate mesh of the NURBS object (meshed afresh, the NURBS object "
                                       "kept and hidden): for export, modifiers or painting")])
    front: EnumProperty(name="Front Veneer", items=_SPECIES_ITEMS, default="OAK")
    back: EnumProperty(name="Back Veneer", items=[("SAME", "Same as Front", "The front's material on the back too")]
                       + _SPECIES_ITEMS, default="SAME")
    core: EnumProperty(name="Core Plies", items=_SPECIES_ITEMS, default="BIRCH",
                       description="The wood of the inner plies, seen along the rim")
    front_material: StringProperty(name="Own Front", default="",
                                   description="Your own material for the front (instead of the veneer above)")
    back_material: StringProperty(name="Own Back", default="",
                                  description="Your own material for the back")
    edge_material: StringProperty(name="Own Edge", default="",
                                  description="Your own material for the edge band (UVs: u along the rim in "
                                              "metres, v across the thickness from 0 at the front to 1 at the back)")
    plies: IntProperty(name="Plies", default=0, min=0, max=60,
                       description="Number of plies through the thickness (0: about one per 1.5 mm, odd)")
    glue: FloatProperty(name="Glue Lines", default=0.08, min=0.0, max=0.5, subtype='FACTOR',
                        description="Width of the dark line between plies, as a part of a ply")
    grain_angle: FloatProperty(name="Grain Angle", default=0.0, min=-180.0, max=180.0,
                               description="Turn of the veneer grain from the sheet's length (degrees)")
    swap: BoolProperty(name="Swap Sides", default=False,
                       description="Front and back the other way round (the front is the side facing up)")
    image: StringProperty(name="Veneer Image", subtype='FILE_PATH', default="",
                          description="A veneer photo for the front (and back, when the same) instead of the "
                                      "procedural grain; its grain along the image's width")
    image_width: FloatProperty(name="Image Width", default=0.5, min=0.001, max=100.0, unit='LENGTH',
                               description="Length of veneer the image's width covers")
    edge_image: StringProperty(name="Edge Image", subtype='FILE_PATH', default="",
                               description="A photo of a plywood edge for the band instead of the procedural "
                                           "plies: its height spans the thickness, its width runs along the rim")
    detail: EnumProperty(
        name="Detail", default="MEDIUM",
        items=[("LOW", "Low", "Chord 0.4% of the size, 15 degrees"),
               ("MEDIUM", "Medium", "Chord 0.1% of the size, 7.5 degrees"),
               ("HIGH", "High", "Chord 0.025% of the size, 3 degrees")],
        description="How finely the render mesh is made (the NURBS object uses its own display detail)")
    hide_original: BoolProperty(name="Hide NURBS Object", default=True,
                                description="Hide the NURBS object in the viewport and in renders")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(store.is_nurbs(o) or o.type == 'MESH'
                                                for o in context.selected_objects)

    def _remembered(self, context):
        """Settings not given to this run come from the last run on the active
        object (textured in place, a render mesh, or the NURBS object a render
        mesh was made from): running it again updates the result as it was."""
        o = context.active_object
        if o is not None and store.is_nurbs(o):
            # the result being updated: the object itself (textured in place) or its render mesh
            if not self.properties.is_property_set("target"):
                self.target = "OBJECT" if PROP in o or _render_mesh(o) is None else "MESH"
            if self.target == "MESH":
                o = _render_mesh(o)
        if not _is_plywood(o):
            return
        try:
            last = json.loads(o[PROP])
        except (ValueError, TypeError):
            return
        for k, v in last.items():
            if k in self._settings() and not self.properties.is_property_set(k):
                try:
                    setattr(self, k, v)
                except (TypeError, ValueError, AttributeError):
                    pass

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        if any(store.is_nurbs(o) or _source(o) is not None for o in context.selected_objects):
            lay.row().prop(self, "target", expand=True)
        col = lay.column(align=True)
        col.prop(self, "front")
        col.prop_search(self, "front_material", bpy.data, "materials")
        col = lay.column(align=True)
        col.prop(self, "back")
        col.prop_search(self, "back_material", bpy.data, "materials")
        col = lay.column(align=True)
        col.prop(self, "core")
        col.prop_search(self, "edge_material", bpy.data, "materials")
        col = lay.column(align=True)
        col.prop(self, "plies")
        col.prop(self, "glue")
        lay.prop(self, "grain_angle")
        lay.prop(self, "swap")
        lay.prop(self, "image")
        if self.image:
            lay.prop(self, "image_width")
        lay.prop(self, "edge_image")
        if self.target == "MESH":
            lay.prop(self, "detail")
            lay.prop(self, "hide_original")
        info = getattr(NPROTO_OT_plywood, "_last", None)
        if info:
            lay.label(text=info, icon='INFO')

    def _settings(self):
        return {k: getattr(self, k) for k in ("target", "detail", "front", "back", "core", "front_material",
                                               "back_material", "edge_material", "plies", "glue", "grain_angle",
                                               "swap", "image", "image_width", "edge_image")}

    def _target(self, context, src):
        """(object to texture, NURBS object it was meshed from or None, "BREP" /
        "MESH"): the NURBS object itself, a fresh mesh of it (in its render mesh
        when it has one), or a plain mesh as it is."""
        if _is_plywood(src) and _source(src) is not None:
            dst, src = src, _source(src)
        elif store.is_nurbs(src):
            if self.target == "OBJECT":
                return src, None, "BREP"
            dst = _render_mesh(src)
        else:
            return src, None, "MESH"
        brep = store.get_brep(src)
        me, m = brep_mesh.mesh_from_brep(brep, src.name + " Plywood", light=True, detail=self.detail)
        if dst is None:
            dst = bpy.data.objects.new(src.name + " Plywood", me)
            for coll in src.users_collection:
                coll.objects.link(dst)
            dst.parent = src.parent
            dst.matrix_parent_inverse = src.matrix_parent_inverse.copy()
            dst.matrix_basis = src.matrix_basis.copy()
            dst[PROP_SRC] = src
        else:
            old = dst.data
            mats = list(old.materials)
            dst.data = me
            for mt in mats:
                me.materials.append(mt)
            if old.users == 0:
                bpy.data.meshes.remove(old)
        return dst, src, "MESH"

    def _texture_brep(self, context, obj):
        """Plywood on the NURBS object's own faces: the setup goes on the object,
        the rebuild of its mesh does the sorting, materials and UVs."""
        from . import facetex
        prev = obj.get(facetex.PROP)
        d = facetex.get(obj) or {}
        d["plywood"] = {"swap": self.swap, "grain": self.grain_angle, "plies": self.plies}
        facetex.put(obj, d)
        facetex.rewrite(obj, context)
        err = facetex.LAST.get(obj.name)
        if err:
            if prev is None:
                del obj[facetex.PROP]
            else:
                obj[facetex.PROP] = prev
            facetex.rewrite(obj, context)
            raise ValueError(err)
        fc = facetex.plywood_classes(obj)
        return fc, facetex.thickness(obj)

    def execute(self, context):
        self._remembered(context)
        objs = [o for o in context.selected_objects if store.is_nurbs(o) or o.type == 'MESH']
        done, notes = [], []
        for o in objs:
            if o.type == 'MESH' and not store.is_nurbs(o) and not _is_plywood(o) and \
                    any(_is_plywood(x) and _source(x) is not None for x in objs):
                continue
            try:
                dst, src, how = self._target(context, o)
                first = PROP not in dst
                if how == "BREP":
                    self._materials(dst, first)
                    cls, t = self._texture_brep(context, dst)
                    what = "B-rep faces"
                else:
                    cls, t = classify(dst.data, dst.matrix_world, swap=self.swap)
                    make_uvs(dst.data, dst.matrix_world, cls, self.grain_angle)
                    self._materials(dst, first)
                    dst.data.polygons.foreach_set("material_index", cls.astype(np.int32))
                    dst.data.update()
                    what = "faces"
                n = self.plies or _auto_plies(t)
                dst["ply_count"] = n
                dst["ply_glue"] = self.glue
                dst["ply_thickness"] = t
                dst[PROP] = json.dumps(self._settings())
                if src is not None and self.hide_original:
                    src.hide_set(True)
                    src.hide_render = True
                done.append(dst)
                notes.append(f"{dst.name}: {t * 1000:.1f} mm, {n} plies, "
                             f"{int((cls == FRONT).sum())} front / {int((cls == BACK).sum())} back / "
                             f"{int((cls == EDGE).sum())} edge {what}"
                             + (" (its mesh before the modifiers)" if how == "MESH" and src is None
                                and len(dst.modifiers) else ""))
            except Exception as exc:
                self.report({'WARNING'}, f"{o.name}: {exc}")
        if not done:
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for o in done:
            o.select_set(True)
        context.view_layer.objects.active = done[0]
        NPROTO_OT_plywood._last = notes[0] if len(notes) == 1 else f"{len(done)} objects"
        self.report({'INFO'}, "Plywood: " + "; ".join(notes))
        return {'FINISHED'}

    def _materials(self, obj, first):
        """Slots 0 front, 1 back, 2 edge. Each takes its Own material when one is
        named; else a material you put in the slot yourself stays (empty the
        slot for the tool's own again); else the tool's procedural one, made
        once and kept (edits to it stay). On the first run the tool's own fill
        all three."""
        me = obj.data
        base = obj.name
        while len(me.materials) < 3:
            me.materials.append(None)
        own = lambda name: bpy.data.materials.get(name) if name else None
        prev_front = me.materials[FRONT]

        def fill(slot, role, named, make, update):
            cur = me.materials[slot]
            if named is not None:
                me.materials[slot] = named
            elif _ours(cur, role):
                update(cur)
            elif cur is None or first:
                m = make()
                m["nproto_ply"] = role
                me.materials[slot] = m
            return me.materials[slot]

        f = fill(FRONT, "front", own(self.front_material),
                 lambda: veneer_material(base + " Front Veneer", self.front, self.image, self.image_width),
                 lambda m: set_veneer(m, self.front, self.image, self.image_width))
        # the back slot: the front's material ("Same as Front"), the tool's own back veneer, or
        # yours -- what sits there now is yours unless it is the front's or the tool's
        cur_b = me.materials[BACK]
        named_b = own(self.back_material)
        shared = cur_b is None or cur_b == prev_front or cur_b == f or _ours(cur_b, "front")
        if named_b is not None:
            me.materials[BACK] = named_b
        elif self.back == "SAME":
            if first or shared or _ours(cur_b, "back"):
                me.materials[BACK] = f
        elif _ours(cur_b, "back"):
            set_veneer(cur_b, self.back)
        elif first or shared:
            m = veneer_material(base + " Back Veneer", self.back)
            m["nproto_ply"] = "back"
            me.materials[BACK] = m
        back = self.front if self.back == "SAME" else self.back
        fill(EDGE, "edge", own(self.edge_material),
             lambda: edge_material(base + " Edge", self.front, back, self.core),
             lambda m: None)
        e = me.materials[EDGE]
        if _ours(e, "edge"):
            set_edge(e, self.front, back, self.core, self.edge_image)


classes = (NPROTO_OT_plywood,)
