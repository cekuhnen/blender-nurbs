"""The tools, sorted into four groups: one table drives the sidebar panels, the
NURBS header menu and the Add (Shift+A) menu, so they always list the same tools.

A group is (key, title, sections); a section is (label, rows) or (label, rows,
needs); a row is a list of tools; a tool is (operator id, button text or None for
the operator's own label, icon, operator properties or None), or a Dropdown of
tools. Panels draw the rows (sections separated by a gap, a Dropdown as one menu
button), menus draw the sections with their labels and every tool.

`needs` are the kinds of object a section's tools work on (CURVE, NURBS, MESH,
SUBD, SURFACE: see kinds()); a tool takes other kinds as its own fifth item
(_t(..., needs=)). A tool works on a NURBS object when it takes its edges, as
Loft or Extrude do, not only when it takes the surface. When objects are
selected and none is of a tool's kinds, the panel draws it dimmed. Dimmed buttons
still run: with nothing fitting selected a tool asks for its inputs, like
Alt+click."""

import bpy

from .kernel import load as load_kernel


def _t(op, text=None, icon='NONE', needs=None):
    return (op, text, icon, None) if needs is None else (op, text, icon, None, frozenset(needs))


def tool_needs(tool, default=()):
    """The kinds a tool works on: its own, else its section's."""
    return tool[4] if not isinstance(tool, Dropdown) and len(tool) > 4 else default


class MenuOnly(tuple):
    """A row only the menus list (NURBS header menu, Add / Shift+A), not the panel."""


def _p(kind, text=None):
    return ("nproto.add_primitive", text or kind.title(), 'NONE', {"kind": kind})


class Dropdown(tuple):
    """(text, icon, menu class name, tools): one menu button in the panel; the menus list the tools."""
    text = property(lambda self: self[0])
    icon = property(lambda self: self[1])
    menu = property(lambda self: self[2])
    tools = property(lambda self: self[3])


CURVE, NURBS, MESH, SUBD, SURFACE = "CURVE", "NURBS", "MESH", "SUBD", "SURFACE"
EDGES = {CURVE, NURBS}                                       # curves, or a surface's edges


GROUPS = (
    ("CURVES", "Create Curves", (
        ("Draw", (
            (_t("nproto.draw_curve", "CV Curve", 'CURVE_NCURVE'), _t("nproto.draw_line", "Draw Line", 'IPO_LINEAR'),
             _t("nproto.draw_arc", "Arc", 'IPO_CIRC')),
            (_t("nproto.draw_circle", "Circle", 'CURVE_NCIRCLE'), _t("nproto.draw_ellipse", "Ellipse", 'SURFACE_NCIRCLE'),
             _t("nproto.draw_rectangle", "Rectangle", 'MESH_PLANE')),
            (_t("nproto.draw_polygon", "Polygon", 'SEQ_CHROMA_SCOPE'), _t("nproto.draw_slot", "Slot", 'MESH_CAPSULE'),
             _t("nproto.draw_helix", "Helix", 'FORCE_VORTEX')),
            (_t("nproto.draw_profile", "Profile", 'MESH_CIRCLE'),),
        )),
        ("Between Curves", (
            (_t("nproto.blend_curve", icon='IPO_EASE_IN_OUT', needs=EDGES),
             _t("nproto.fillet_curves", "Fillet", 'SPHERECURVE')),
        ), {CURVE}),
        ("From Curves", (
            (_t("nproto.offset_curve", icon='MOD_OFFSET'),
             _t("nproto.offset_on_surface", icon='MOD_SHRINKWRAP', needs=EDGES)),
        ), {CURVE}),
        ("Shapes", (                                         # at the 3D cursor: the menus only (the panel
            MenuOnly((_p("LINE"), _p("ARC"), _p("CIRCLE"), _p("ELLIPSE"), _p("RECTANGLE"), _p("POLYGON"),
                      _p("HELIX"))),                         # draws them with the tools above)
        )),
        ("From Surfaces", (
            (_t("nproto.iso_curve", icon='IPO_LINEAR'), _t("nproto.dup_edge", icon='DUPLICATE')),
            (_t("nproto.project_curve", icon='MOD_SHRINKWRAP'), _t("nproto.intersect", icon='SELECT_INTERSECT')),
        ), {NURBS}),
    )),
    ("SURFACES", "Create Surfaces", (
        ("Draw", (
            (_t("nproto.draw_plane", "Plane", 'MESH_PLANE'), _t("nproto.draw_box", "Box", 'MESH_CUBE'),
             _t("nproto.draw_cylinder", "Cylinder", 'MESH_CYLINDER')),
            (Dropdown(("Primitive", 'MESH_UVSPHERE', "VIEW3D_MT_nproto_primitives",
                       (_p("PLANE"), _p("DISK"), _p("BOX"), _p("CYLINDER"), _p("CONE"), _p("SPHERE"),
                        _p("TORUS")))),),
        )),
        ("Through Curves", (
            (_t("nproto.loft", icon='MOD_SIMPLEDEFORM'), _t("nproto.loft_rails", icon='MOD_SIMPLEDEFORM'),
             _t("nproto.network", "Network", 'MESH_GRID')),
            (_t("nproto.revolve", icon='MOD_SCREW'), _t("nproto.rail_revolve", icon='MOD_SCREW')),
            (_t("nproto.sweep", icon='MOD_CURVE'), _t("nproto.pipe", icon='META_CAPSULE')),
        ), EDGES),
        ("Extrude and Fill", (
            (_t("nproto.extrude", icon='MOD_SOLIDIFY'), _t("nproto.flange", icon='MOD_SOLIDIFY')),
            (_t("nproto.planar", icon='MESH_PLANE'), _t("nproto.dome", icon='MESH_UVSPHERE')),
            (_t("nproto.boundary", icon='MESH_GRID'), _t("nproto.patch", icon='MOD_TRIANGULATE')),
        ), EDGES),
        ("From Surfaces", (
            (_t("nproto.offset", icon='MOD_OFFSET'), _t("nproto.thicken", icon='MOD_SOLIDIFY')),
            (("nproto.pattern", "Pattern", 'MOD_ARRAY', {"auto": True}, frozenset(EDGES)),  # linear / circular / path: M
             ("nproto.mirror", "Mirror", 'MOD_MIRROR', None, frozenset(EDGES)),             # mirror copy / make symmetric: S
             _t("nproto.tween", "Tween", 'IPO_EASE_IN_OUT', needs=EDGES)),                  # two objects: copies between
            (_t("nproto.unroll", "Unroll", 'UV'), _t("nproto.flow", "Flow", 'MOD_SIMPLEDEFORM', needs=EDGES)),
            (_t("nproto.from_blender_surface", icon='SURFACE_NSURFACE', needs={SURFACE}),),
        ), {NURBS}),
    )),
    ("CURVE_EDIT", "Curve Edit", (                         # in Surface Edit's order: shape, trim, join
        ("Shape", (
            (_t("nproto.rebuild_curve", "Rebuild Curve", 'FILE_REFRESH'),
             _t("nproto.extend_curve", "Extend Curve", 'FORWARD')),
            (_t("nproto.align_curve", icon='SNAP_MIDPOINT'),),
        ), {CURVE}),
        ("Trim and Split", (
            (_t("nproto.trim_curves", "Trim Curves", 'X'), _t("nproto.split_curves", "Split Curves", 'MOD_EDGESPLIT')),
        ), {CURVE}),
        ("Join", (
            (_t("nproto.join_curves", "Join Curves", 'AUTOMERGE_ON'),
             _t("nproto.unjoin_curves", "Unjoin Curves", 'AUTOMERGE_OFF')),
        ), {CURVE}),
    )),
    ("SURFACE_EDIT", "Surface Edit", (
        ("", (                                               # Tab on the surface does it: the menus only
            MenuOnly((_t("nproto.edit_cvs", "Edit CVs (Tab)", 'EDITMODE_HLT'),)),
        )),
        ("Shape", (
            (_t("nproto.rebuild_surface", icon='MOD_REMESH'), _t("nproto.extend", icon='FULLSCREEN_ENTER')),
            (_t("nproto.surface_align", icon='SNAP_EDGE'), _t("nproto.match_surface", icon='SNAP_FACE')),
            (_t("nproto.refit_trim", icon='MOD_LATTICE'), _t("nproto.merge_patches", icon='MESH_GRID')),
            (_t("nproto.flip_normal", icon='NORMALS_FACE'),),
        ), {NURBS}),
        ("Trim and Split", (
            (_t("nproto.trim", icon='MOD_BOOLEAN', needs=EDGES), _t("nproto.split", icon='MOD_EDGESPLIT', needs=EDGES)),
            (_t("nproto.untrim", icon='LOOP_BACK'), _t("nproto.untrim_region", icon='SELECT_SUBTRACT')),
            (_t("nproto.detach_iso", icon='UV_EDGESEL'),),
        ), {NURBS}),
        ("Join", (
            (_t("nproto.join", icon='AUTOMERGE_ON'), _t("nproto.unjoin", icon='AUTOMERGE_OFF')),
        ), {NURBS}),
        ("Fillets", (
            (_t("nproto.fillet", "Surface Fillet", 'MOD_BEVEL'), _t("nproto.edge_fillet", icon='MOD_BEVEL')),
        ), {NURBS}),
        ("Solids", (
            (("nproto.boolean", "Union", 'SELECT_EXTEND', {"operation": "UNION"}),
             ("nproto.boolean", "Difference", 'SELECT_SUBTRACT', {"operation": "DIFFERENCE"}),
             ("nproto.boolean", "Intersection", 'SELECT_INTERSECT', {"operation": "INTERSECTION"})),
            (_t("nproto.push_pull", icon='FACESEL'), _t("nproto.rotate_face", icon='DRIVER_ROTATIONAL_DIFFERENCE')),
            (_t("nproto.move_face", icon='ORIENTATION_GLOBAL'), _t("nproto.replace_face", icon='FACE_MAPS')),
            (_t("nproto.boundary_fill", icon='MESH_CUBE'), _t("nproto.emboss", icon='MOD_DISPLACE', needs=EDGES)),
            (_t("nproto.shell", icon='MOD_SOLIDIFY'),),
        ), {NURBS}),
    )),
    ("CONVERT", "Convert", (
        ("To Mesh", (
            (_t("nproto.brep_to_mesh", "BREP to Mesh", 'MESH_DATA'),),
        ), {NURBS}),
        ("To BREP", (
            (_t("nproto.mesh_to_brep", "Mesh to BREP", 'MESH_CUBE'),),
        ), {MESH}),
        ("", (
            (_t("nproto.subd_to_brep", "SubD to BREP", 'MOD_SUBSURF'),),
        ), {SUBD}),
    )),
    ("SCAN", "Scan", (
        ("SubD Cage", (
            (_t("nproto.scan_tube_cage", "Cage from Scan", 'MESH_CYLINDER'),),
        ), {MESH}),
        ("", (
            (_t("nproto.scan_fit_cage", "Fit Cage to Scan", 'MOD_SHRINKWRAP'),),
        ), {SUBD}),
        ("NURBS Patch", (
            (_t("nproto.scan_drape_patch", "Drape Patch to Scan", 'SURFACE_NSURFACE'),),
        ), {MESH, NURBS}),
    )),
)

GROUP = {g[0]: g for g in GROUPS}
PICK_HINT = "Alt+click a tool to pick inputs instead of using the selection."


def _flat(row):
    for tool in row:
        if isinstance(tool, Dropdown):
            yield from tool.tools
        else:
            yield tool


def tools(key=None):
    """Every tool of one group (or of all), in drawing order (a Dropdown's tools in its place)."""
    for k, _title, sections in GROUPS:
        if key is None or k == key:
            for sec in sections:
                for row in sec[1]:
                    yield from _flat(row)


def dropdowns():
    for _k, _title, sections in GROUPS:
        for sec in sections:
            for row in sec[1]:
                yield from (t for t in row if isinstance(t, Dropdown))


def kinds(obj):
    """The kinds of object obj is for the tools' `needs`."""
    from . import store
    if obj is None:
        return set()
    if obj.type == 'CURVE':
        return {CURVE}
    if obj.type == 'SURFACE':
        return {SURFACE}
    if obj.type != 'MESH':
        return set()
    if store.is_nurbs(obj):
        return {NURBS}
    out = {MESH}
    if any(m.type == 'SUBSURF' for m in obj.modifiers):
        out.add(SUBD)
    try:
        from . import cvs
        if cvs.is_cage(obj):                            # a NURBS object's CV cage, being edited
            out = {NURBS}
    except Exception:
        pass
    return out


def selected_kinds(context):
    """Kinds among the selected objects (and the active one), or None with nothing selected."""
    objs = set(getattr(context, "selected_objects", None) or ())
    act = getattr(context, "active_object", None)
    if act is not None and (act.select_get() or context.mode != 'OBJECT'):
        objs.add(act)
    if not objs:
        return None
    out = set()
    for o in objs:
        out |= kinds(o)
    return out


def fits(needs, have):
    """Whether a section with `needs` fits the selection's kinds (always, with nothing selected)."""
    return not needs or have is None or bool(set(needs) & have)


def _button(lay, tool):
    if isinstance(tool, Dropdown):
        lay.menu(tool.menu, text=tool.text, icon=tool.icon)
        return
    op, text, icon, props = tool[:4]
    kw = {"icon": icon}
    if text is not None:
        kw["text"] = text
    ptr = lay.operator(op, **kw)
    for k, v in (props or {}).items():
        setattr(ptr, k, v)


def draw_panel(lay, key, context=None):
    have = selected_kinds(context) if context is not None else None
    col = lay.column(align=True)
    first = True
    for sec in GROUP[key][2]:
        label, rows = sec[:2]
        if all(isinstance(r, MenuOnly) for r in rows):
            continue                                       # a section only the menus list
        if not first and label:
            col.separator(factor=0.6)
        first = False
        needs = sec[2] if len(sec) > 2 else ()
        for row in rows:
            if isinstance(row, MenuOnly):
                continue
            r = col.row(align=True)
            on = [fits(tool_needs(t, needs), have) for t in row]
            if len(set(on)) == 1:
                r.active = on[0]                           # dimmed, not disabled: it still runs
                for tool in row:
                    _button(r, tool)
            else:                                          # a lit and a dimmed tool side by side
                for tool, lit in zip(row, on):
                    sub = r.row(align=True)
                    sub.active = lit
                    _button(sub, tool)


def draw_menu(lay, key):
    first = True
    for sec in GROUP[key][2]:
        label, rows = sec[:2]
        if label:
            if not first:
                lay.separator()
            lay.label(text=label)
        first = False
        for row in rows:
            for tool in _flat(row):
                _button(lay, tool)


# ---------------------------------------------------------------- sidebar panels

class _ToolPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "NURBS Proto"
    group = ""
    hint = False

    @classmethod
    def poll(cls, context):
        return load_kernel() is not None

    def draw(self, context):
        draw_panel(self.layout, self.group, context)
        if self.hint:
            self.layout.label(text=PICK_HINT, icon='INFO')


class VIEW3D_PT_nproto_curves(_ToolPanel, bpy.types.Panel):
    bl_label = "Create Curves"
    bl_order = 1
    group = "CURVES"


class VIEW3D_PT_nproto_surfaces(_ToolPanel, bpy.types.Panel):
    bl_label = "Create Surfaces"
    bl_order = 2
    group = "SURFACES"


class VIEW3D_PT_nproto_curve_edit(_ToolPanel, bpy.types.Panel):
    bl_label = "Curve Edit"
    bl_order = 3
    group = "CURVE_EDIT"


class VIEW3D_PT_nproto_surface_edit(_ToolPanel, bpy.types.Panel):
    bl_label = "Surface Edit"
    bl_order = 4
    group = "SURFACE_EDIT"
    hint = True


class VIEW3D_PT_nproto_convert(_ToolPanel, bpy.types.Panel):
    bl_label = "Convert"
    bl_order = 5
    group = "CONVERT"
    bl_options = {'DEFAULT_CLOSED'}


class VIEW3D_PT_nproto_scan(_ToolPanel, bpy.types.Panel):
    """Fitting surfaces to a scan (a mesh): a SubD cage, or a single NURBS patch"""
    bl_label = "Scan"
    bl_order = 5                                           # (after Convert: registered after it)
    group = "SCAN"
    bl_options = {'DEFAULT_CLOSED'}


# ---------------------------------------------------------------- menus

class _ToolMenu:
    group = ""

    def draw(self, context):
        draw_menu(self.layout, self.group)


class VIEW3D_MT_nproto_curves(_ToolMenu, bpy.types.Menu):
    bl_label = "Create Curves"
    group = "CURVES"


class VIEW3D_MT_nproto_surfaces(_ToolMenu, bpy.types.Menu):
    bl_label = "Create Surfaces"
    group = "SURFACES"


class VIEW3D_MT_nproto_curve_edit(_ToolMenu, bpy.types.Menu):
    bl_label = "Curve Edit"
    group = "CURVE_EDIT"


class VIEW3D_MT_nproto_surface_edit(_ToolMenu, bpy.types.Menu):
    bl_label = "Surface Edit"
    group = "SURFACE_EDIT"


class VIEW3D_MT_nproto_convert(_ToolMenu, bpy.types.Menu):
    bl_label = "Convert"
    group = "CONVERT"


class VIEW3D_MT_nproto_scan(_ToolMenu, bpy.types.Menu):
    bl_label = "Scan"
    group = "SCAN"


class VIEW3D_MT_nproto_primitives(bpy.types.Menu):
    """NURBS surface primitives at the 3D cursor"""
    bl_label = "Primitive"

    def draw(self, context):
        for d in dropdowns():
            if d.menu == "VIEW3D_MT_nproto_primitives":
                for tool in d.tools:
                    _button(self.layout, tool)


class VIEW3D_MT_nproto_tools(bpy.types.Menu):
    """All NURBS tools; F3 search finds them by name through this menu"""
    bl_label = "NURBS Proto"

    def draw(self, context):
        lay = self.layout
        lay.menu("VIEW3D_MT_nproto_curves", icon='CURVE_NCURVE')
        lay.menu("VIEW3D_MT_nproto_surfaces", icon='SURFACE_NSURFACE')
        lay.separator()
        lay.menu("VIEW3D_MT_nproto_curve_edit", icon='CURVE_DATA')
        lay.menu("VIEW3D_MT_nproto_surface_edit", icon='SURFACE_DATA')
        lay.menu("VIEW3D_MT_nproto_convert", icon='MESH_DATA')
        lay.menu("VIEW3D_MT_nproto_scan", icon='MOD_SHRINKWRAP')
        lay.separator()
        lay.operator("nproto.continuity", icon='DRIVER_DISTANCE')
        lay.operator("nproto.section_add", icon='MOD_EDGESPLIT')
        lay.separator()
        lay.operator("nproto.move_pivot", icon='OBJECT_ORIGIN')
        lay.operator("nproto.align_parts", icon='SNAP_FACE')


def menu_header(self, context):
    if context.mode == 'OBJECT':
        self.layout.menu("VIEW3D_MT_nproto_tools")


def menu_add(self, context):
    lay = self.layout
    lay.separator()
    lay.menu("VIEW3D_MT_nproto_curves", text="NURBS Curves", icon='CURVE_NCURVE')
    lay.menu("VIEW3D_MT_nproto_surfaces", text="NURBS Surfaces", icon='SURFACE_NSURFACE')


classes = (VIEW3D_PT_nproto_curves, VIEW3D_PT_nproto_surfaces, VIEW3D_PT_nproto_curve_edit,
           VIEW3D_PT_nproto_surface_edit, VIEW3D_PT_nproto_convert, VIEW3D_PT_nproto_scan, VIEW3D_MT_nproto_curves,
           VIEW3D_MT_nproto_surfaces, VIEW3D_MT_nproto_curve_edit, VIEW3D_MT_nproto_surface_edit,
           VIEW3D_MT_nproto_convert, VIEW3D_MT_nproto_scan, VIEW3D_MT_nproto_primitives, VIEW3D_MT_nproto_tools)


def register():
    bpy.types.VIEW3D_MT_editor_menus.append(menu_header)
    bpy.types.VIEW3D_MT_add.append(menu_add)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(menu_add)
    bpy.types.VIEW3D_MT_editor_menus.remove(menu_header)
