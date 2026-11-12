"""STEP (ISO 10303-21, AP214 "automotive design") export of NURBS objects.

Exact geometry: every face's surface is written as a B-spline surface (rational
when it has weights), every edge as a B-spline curve cut exactly to the edge's
range, with full B-rep topology (vertices, edge curves shared by their two
faces, oriented edges, outer / inner face bounds, face sense). Closed bodies
become MANIFOLD_SOLID_BREPs, open ones SHELL_BASED_SURFACE_MODELs. Each object
is one body, in world space, written in the chosen length unit.
"""

import datetime
import os

import numpy as np

from . import geom, store

UNITS = {"MM": ("MILLI", 1000.0), "CM": ("CENTI", 100.0), "M": (None, 1.0)}


def _vpt(v):
    q = v.point()
    return (q.x, q.y, q.z) if hasattr(q, "x") else tuple(q)


def _f(x):
    """STEP real: always with a decimal point."""
    s = f"{float(x):.15G}"
    if "E" in s:
        m, e = s.split("E")
        if "." not in m:
            m += "."
        return f"{m}E{e}"
    if "." not in s:
        s += "."
    return s


def _str(s):
    return "'" + str(s).replace("'", "''").replace("\\", "\\\\") + "'"


def _knots(U):
    """Distinct knots + multiplicities."""
    ks, ms = [], []
    for k in U:
        if ks and abs(k - ks[-1]) <= 1e-12 * (1 + abs(k)):
            ms[-1] += 1
        else:
            ks.append(float(k))
            ms.append(1)
    return ks, ms


class _Writer:
    def __init__(self):
        self.lines = []
        self.n = 0

    def add(self, text):
        self.n += 1
        self.lines.append(f"#{self.n}={text};")
        return f"#{self.n}"


class StepExporter:
    def __init__(self, scale=1.0, unit="MM"):
        self.w = _Writer()
        self.unit_prefix, unit_factor = UNITS[unit]
        self.scale = scale * unit_factor
        self.stats = {"bodies": 0, "faces": 0, "edges": 0, "solids": 0, "sheets": 0}
        self._dir_cache = {}

    # -- small geometry ----------------------------------------------------
    def point(self, p):
        return self.w.add(f"CARTESIAN_POINT('',({','.join(_f(c) for c in p)}))")

    def direction(self, d):
        key = tuple(round(float(c), 12) for c in d)
        if key not in self._dir_cache:
            self._dir_cache[key] = self.w.add(f"DIRECTION('',({','.join(_f(c) for c in d)}))")
        return self._dir_cache[key]

    def _pts(self, P):
        return "(" + ",".join(self.point(p) for p in P) + ")"

    # -- curves / surfaces -------------------------------------------------
    def bspline_curve(self, deg, P, U, W=None):
        ks, ms = _knots(U)
        pts = self._pts(P)
        mult = "(" + ",".join(str(m) for m in ms) + ")"
        kn = "(" + ",".join(_f(k) for k in ks) + ")"
        if W is None or max(abs(w - 1.0) for w in W) < 1e-12:
            return self.w.add(f"B_SPLINE_CURVE_WITH_KNOTS('',{deg},{pts},.UNSPECIFIED.,.F.,.F.,{mult},{kn},.UNSPECIFIED.)")
        ws = "(" + ",".join(_f(w) for w in W) + ")"
        return self.w.add(f"(BOUNDED_CURVE() B_SPLINE_CURVE({deg},{pts},.UNSPECIFIED.,.F.,.F.) "
                          f"B_SPLINE_CURVE_WITH_KNOTS({mult},{kn},.UNSPECIFIED.) CURVE() "
                          f"GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE({ws}) REPRESENTATION_ITEM(''))")

    def bspline_surface(self, du, dv, P, nu, nv, Uu, Uv, W=None):
        rows = []
        for i in range(nu):
            rows.append("(" + ",".join(self.point(P[i * nv + j]) for j in range(nv)) + ")")
        grid = "(" + ",".join(rows) + ")"
        ku, mu = _knots(Uu)
        kv, mv = _knots(Uv)
        mults = f"({','.join(map(str, mu))}),({','.join(map(str, mv))})"
        knots = f"({','.join(_f(k) for k in ku)}),({','.join(_f(k) for k in kv)})"
        if W is None or max(abs(w - 1.0) for w in W) < 1e-12:
            return self.w.add(f"B_SPLINE_SURFACE_WITH_KNOTS('',{du},{dv},{grid},.UNSPECIFIED.,.F.,.F.,.F.,"
                              f"{mults},{knots},.UNSPECIFIED.)")
        wrows = []
        for i in range(nu):
            wrows.append("(" + ",".join(_f(W[i * nv + j]) for j in range(nv)) + ")")
        ws = "(" + ",".join(wrows) + ")"
        return self.w.add(f"(BOUNDED_SURFACE() B_SPLINE_SURFACE({du},{dv},{grid},.UNSPECIFIED.,.F.,.F.,.F.) "
                          f"B_SPLINE_SURFACE_WITH_KNOTS({mults},{knots},.UNSPECIFIED.) "
                          f"GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE({ws}) "
                          f"REPRESENTATION_ITEM('') SURFACE())")

    # -- one body ----------------------------------------------------------
    def body(self, obj):
        """Write one NURBS object; returns the representation item (solid / shell model)."""
        from .kernel import load
        from .square import _edge_curve
        ext = load().blender_ext
        brep = store.get_brep(obj)
        M = np.array(geom.world_matrix(obj), dtype=float)
        A, t = M[:3, :3] * self.scale, M[:3, 3] * self.scale
        X = lambda p: A @ np.asarray(p, float) + t
        size = float(np.linalg.norm(np.ptp(np.array(brep.bounding_box(), float), axis=0))) or 1.0

        verts, edges = {}, {}
        # topology, not positions: a closed edge (a trim circle) whose curve
        # closes to only ~1e-6 must still start and end at ONE vertex, or its
        # loop is open and readers drop the face (Plasticity lost a fillet)
        all_v = list(brep.vertices())
        all_e = list(brep.edges())

        def index_of(items, x):
            for i, y in enumerate(items):
                if y == x:
                    return i
            return None

        def vkey(p):
            return ("p",) + tuple(np.round(np.asarray(p, float) / size, 9))

        def vertex(kv, p):
            i = index_of(all_v, kv) if kv is not None else None
            k = ("v", i) if i is not None else vkey(p)
            if k not in verts:
                q = p
                if i is not None:
                    try:
                        q = all_v[i].point()
                        q = (q.x, q.y, q.z) if hasattr(q, "x") else tuple(q)
                    except Exception:
                        q = p
                verts[k] = self.w.add(f"VERTEX_POINT('',{self.point(X(q))})")
            return verts[k]

        def edge_key(e):
            i = index_of(all_e, e)
            if i is not None:
                return ("e", i)
            t0, t1 = e.parameter_range()
            return (vkey(e.point_at(t0)), vkey(e.point_at(t1)), vkey(e.point_at(0.5 * (t0 + t1))))

        def end_vertices(e):
            try:
                return e.start_vertex(), e.end_vertex()
            except Exception:
                return None, None

        def edge_curve(e):
            k = edge_key(e)
            if k in edges:
                return edges[k]
            t0, t1 = e.parameter_range()
            c = _edge_curve(e)
            d = ext.curve_cvs(c)
            P = [X(p) for p in d["points"]]
            W = list(d["weights"]) if d.get("weights") else None
            cur = self.bspline_curve(d["degree"], P, list(d["knots"]), W)
            sv, ev = end_vertices(e)
            p0, p1 = e.point_at(t0), e.point_at(t1)
            if sv is not None and ev is not None:
                # the kernel's start vertex is where the curve starts (else swap)
                a = np.asarray(_vpt(sv), float)
                if np.linalg.norm(a - np.asarray(p0, float)) > np.linalg.norm(a - np.asarray(p1, float)) + 1e-12:
                    sv, ev = ev, sv
            v0, v1 = vertex(sv, p0), vertex(ev, p1)
            ec = self.w.add(f"EDGE_CURVE('',{v0},{v1},{cur},.T.)")
            edges[k] = ec
            self.stats["edges"] += 1
            return ec

        faces = []
        for f in brep.faces():
            surf = f.surface()
            d = ext.surface_cvs(surf)
            P = [X(p) for p in d["points"]]
            W = list(d["weights"]) if d.get("weights") else None
            srf = self.bspline_surface(d["degree_u"], d["degree_v"], P, d["count_u"], d["count_v"],
                                       list(d["knots_u"]), list(d["knots_v"]), W)
            # face sense: the face's outward normal vs the surface's own normal,
            # probed at a point inside the (trimmed) face
            from .square import _foot
            sense = ".T."
            try:
                uv = _foot(surf, np.array(f.interior_point(), float))
                n_out = np.array(f.outward_normal_at_uv(*uv), float)
                _, su, sv = surf.derivatives(*uv)
                sense = ".T." if float(np.cross(su, sv) @ n_out) >= 0 else ".F."
            except Exception:
                pass
            outer = f.boundary_outer_loop()
            outer_ids = {edge_key(e) for e, _ in outer}
            bounds = []
            for loop in f.boundary_loops():
                oes = []
                for e, fwd in loop:
                    if e.length() <= 1e-9 * size:       # degenerate (collapsed) edge: readers re-create it
                        continue
                    oes.append(self.w.add(f"ORIENTED_EDGE('',*,*,{edge_curve(e)},{'.T.' if fwd else '.F.'})"))
                if not oes:
                    continue
                lp = self.w.add(f"EDGE_LOOP('',({','.join(oes)}))")
                is_outer = {edge_key(e) for e, _ in loop} == outer_ids
                bounds.append(self.w.add(f"{'FACE_OUTER_BOUND' if is_outer else 'FACE_BOUND'}('',{lp},.T.)"))
            faces.append(self.w.add(f"ADVANCED_FACE('',({','.join(bounds)}),{srf},{sense})"))
            self.stats["faces"] += 1
        self.stats["bodies"] += 1
        name = _str(obj.name)
        try:
            solid = bool(brep.is_manifold_solid())
        except Exception:
            solid = False
        if solid:
            shell = self.w.add(f"CLOSED_SHELL('',({','.join(faces)}))")
            self.stats["solids"] += 1
            return "solid", self.w.add(f"MANIFOLD_SOLID_BREP({name},{shell})")
        shell = self.w.add(f"OPEN_SHELL('',({','.join(faces)}))")
        self.stats["sheets"] += 1
        return "sheet", self.w.add(f"SHELL_BASED_SURFACE_MODEL({name},({shell}))")

    # -- file --------------------------------------------------------------
    def write(self, path, objects, product_name=None):
        w = self.w
        name = product_name or os.path.splitext(os.path.basename(path))[0]
        app = w.add("APPLICATION_CONTEXT('automotive design')")
        w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{app})")
        pctx = w.add(f"PRODUCT_CONTEXT('',{app},'mechanical')")
        prod = w.add(f"PRODUCT({_str(name)},{_str(name)},'',({pctx}))")
        w.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,({prod}))")
        pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',{prod})")
        pdc = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{app},'design')")
        pd = w.add(f"PRODUCT_DEFINITION('design','',{pdf},{pdc})")
        pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',{pd})")
        prefix = f".{self.unit_prefix}." if self.unit_prefix else "$"
        lu = w.add(f"(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT({prefix},.METRE.))")
        au = w.add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))")
        su = w.add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())")
        unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),{lu},'distance_accuracy_value',"
                    "'confusion accuracy')")
        ctx = w.add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({unc})) "
                    f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({lu},{au},{su})) REPRESENTATION_CONTEXT('Context #1',"
                    "'3D Context with UNIT and UNCERTAINTY'))")
        ax = w.add(f"AXIS2_PLACEMENT_3D('',{self.point((0, 0, 0))},{self.direction((0, 0, 1))},"
                   f"{self.direction((1, 0, 0))})")
        bodies = []
        failed = []
        for obj in objects:
            try:
                bodies.append((obj, *self.body(obj)))
            except Exception as exc:
                failed.append((obj.name, str(exc)))
        main = w.add(f"SHAPE_REPRESENTATION({_str(name)},({ax}),{ctx})")
        w.add(f"SHAPE_DEFINITION_REPRESENTATION({pds},{main})")
        for obj, kind, item in bodies:
            if kind == "solid":
                rep = w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION({_str(obj.name)},({item},{ax}),{ctx})")
            else:
                rep = w.add(f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION({_str(obj.name)},({item},{ax}),{ctx})")
            w.add(f"SHAPE_REPRESENTATION_RELATIONSHIP('','',{main},{rep})")
        stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        header = ["ISO-10303-21;", "HEADER;",
                  "FILE_DESCRIPTION(('NURBS Prototype export'),'2;1');",
                  f"FILE_NAME({_str(os.path.basename(path))},'{stamp}',(''),(''),'NURBS Prototype (Blender)',"
                  "'USD BRep kernel','');",
                  "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(header + w.lines + ["ENDSEC;", "END-ISO-10303-21;", ""]))
        self.stats["failed"] = failed
        return self.stats


def export_step(path, objects, unit="MM", scale=1.0):
    ex = StepExporter(scale=scale, unit=unit)
    stats = ex.write(path, [o for o in objects if store.is_nurbs(o)])
    try:
        with open(path, encoding="utf-8") as fh:
            stats["open_loops"] = loop_problems(fh.read())
    except Exception:
        pass
    return stats


def loop_problems(text):
    """Edge loops of a written STEP file that don't close (an edge ending at a
    vertex the next one doesn't start from): readers drop such faces. A list
    of loop ids, empty when all are sound."""
    import re
    ent = {}
    for m in re.finditer(r"^#(\d+)=([A-Z_0-9]+)\((.*)\);\s*$", text, re.M):
        ent[int(m.group(1))] = (m.group(2), m.group(3))
    refs = lambda a: [int(x) for x in re.findall(r"#(\d+)", a)]
    bad = []
    for i, (kind, args) in ent.items():
        if kind != "EDGE_LOOP":
            continue
        chain = []
        for oe in refs(args):
            k2, a2 = ent.get(oe, ("", ""))
            if k2 != "ORIENTED_EDGE":
                continue
            ec = refs(a2)[0]
            fwd = a2.rstrip().endswith(".T.")
            k3, a3 = ent.get(ec, ("", ""))
            v0, v1 = refs(a3)[:2]
            if not a3.rstrip().endswith(".T."):
                v0, v1 = v1, v0
            chain.append((v0, v1) if fwd else (v1, v0))
        if chain and any(chain[j][1] != chain[(j + 1) % len(chain)][0] for j in range(len(chain))):
            bad.append(i)
    return bad


# --------------------------------------------------------------------------
# Blender operator: File > Export > NURBS (.step)
# --------------------------------------------------------------------------

import bpy  # noqa: E402
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty  # noqa: E402
from bpy_extras.io_utils import ExportHelper  # noqa: E402


class NPROTO_OT_export_step(bpy.types.Operator, ExportHelper):
    """Export NURBS objects as STEP (AP214): exact B-spline surfaces and curves
    with full B-rep topology; closed bodies as solids"""
    bl_idname = "nproto.export_step"
    bl_label = "Export STEP"
    bl_options = {'PRESET'}

    filename_ext = ".step"
    filter_glob: StringProperty(default="*.step;*.stp", options={'HIDDEN'})
    use_selection: BoolProperty(name="Selection Only", default=True)
    unit: EnumProperty(name="Unit", items=[("MM", "Millimetres", ""), ("CM", "Centimetres", ""),
                                           ("M", "Metres", "")], default="MM",
                       description="Length unit of the file; Blender units are taken as metres")
    global_scale: FloatProperty(name="Scale", default=1.0, min=1e-6, soft_max=1000.0)

    def execute(self, context):
        sc = context.scene.unit_settings.scale_length or 1.0      # Blender units -> metres
        objs = context.selected_objects if self.use_selection else context.scene.objects
        objs = [o for o in objs if store.is_nurbs(o) and o.visible_get()]
        if not objs:
            self.report({'WARNING'}, "No NURBS objects to export")
            return {'CANCELLED'}
        stats = export_step(self.filepath, objs, unit=self.unit, scale=self.global_scale * sc)
        msg = (f"STEP: {stats['bodies']} bodies ({stats['solids']} solid, {stats['sheets']} sheet), "
               f"{stats['faces']} faces, {stats['edges']} edges")
        if stats.get("open_loops"):
            msg += f"; {len(stats['open_loops'])} edge loop(s) don't close -- other programs may drop those faces"
        if stats.get("failed"):
            msg += f"; skipped: {', '.join(n for n, _ in stats['failed'])}"
            self.report({'WARNING'}, msg)
        elif stats.get("open_loops"):
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def menu_export(self, context):
    self.layout.operator(NPROTO_OT_export_step.bl_idname, text="NURBS (.step)")


classes = (NPROTO_OT_export_step,)


def register():
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
