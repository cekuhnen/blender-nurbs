"""The tools, sorted into four groups: one table drives the sidebar panels, the
NURBS header menu and the Add (Shift+A) menu, so they always list the same tools.

A group is (key, title, sections); a section is (label, rows); a row is a list of
tools; a tool is (operator id, button text or None for the operator's own label,
icon, operator properties or None). Panels draw the rows (sections separated by a
gap), menus draw the sections with their labels."""

import bpy

from .kernel import load as load_kernel


def _t(op, text=None, icon='NONE'):
    return (op, text, icon, None)


def _p(kind, text=None):
    return ("nproto.add_primitive", text or kind.title(), 'NONE', {"kind": kind})


GROUPS = (
    ("CURVES", "Create Curves", (
        ("Draw", (
            (_t("nproto.draw_curve", "CV Curve", 'CURVE_NCURVE'), _t("nproto.draw_line", "Draw Line", 'IPO_LINEAR'),
             _t("nproto.draw_profile", "Profile", 'MESH_CIRCLE')),
        )),
        ("Shapes", (
            (_p("LINE"), _p("ARC"), _p("CIRCLE"), _p("ELLIPSE")),
            (_p("RECTANGLE"), _p("POLYGON"), _p("HELIX")),
        )),
        ("From Surfaces", (
            (_t("nproto.iso_curve", icon='IPO_LINEAR'), _t("nproto.dup_edge", icon='DUPLICATE')),
            (_t("nproto.project_curve", icon='MOD_SHRINKWRAP'), _t("nproto.intersect", icon='SELECT_INTERSECT')),
        )),
        ("Between Curves", (
            (_t("nproto.blend_curve", icon='IPO_EASE_IN_OUT'),),
        )),
    )),
    ("SURFACES", "Create Surfaces", (
        ("Draw", (
            (_t("nproto.draw_plane", "Draw Plane", 'MESH_PLANE'),),
        )),
        ("Primitives", (
            (_p("PLANE"), _p("DISK"), _p("BOX")),
            (_p("CYLINDER"), _p("CONE"), _p("SPHERE"), _p("TORUS")),
        )),
        ("From Curves", (
            (_t("nproto.loft", icon='MOD_SIMPLEDEFORM'), _t("nproto.loft_rails", icon='MOD_SIMPLEDEFORM')),
            (_t("nproto.sweep", icon='MOD_CURVE'), _t("nproto.pipe", icon='META_CAPSULE')),
            (_t("nproto.extrude", icon='MOD_SOLIDIFY'), _t("nproto.revolve", icon='MOD_SCREW'),
             _t("nproto.planar", icon='MESH_PLANE')),
        )),
        ("Fill a Gap", (
            (_t("nproto.boundary", icon='MESH_GRID'), _t("nproto.square", icon='MESH_PLANE'),
             _t("nproto.patch", icon='MOD_TRIANGULATE')),
        )),
        ("From Surfaces", (
            (_t("nproto.offset", icon='MOD_OFFSET'), _t("nproto.thicken", icon='MOD_SOLIDIFY')),
            (_t("nproto.from_blender_surface", icon='SURFACE_NSURFACE'),),
        )),
    )),
    ("CURVE_EDIT", "Curve Edit", (
        ("Join and Trim", (
            (_t("nproto.join_curves", "Join Curves", 'AUTOMERGE_ON'),
             _t("nproto.unjoin_curves", "Unjoin Curves", 'AUTOMERGE_OFF')),
            (_t("nproto.trim_curves", "Trim Curves", 'X'), _t("nproto.split_curves", "Split Curves", 'MOD_EDGESPLIT')),
        )),
        ("Shape", (
            (_t("nproto.rebuild_curve", "Rebuild Curve", 'FILE_REFRESH'),
             _t("nproto.extend_curve", "Extend Curve", 'FORWARD')),
            (_t("nproto.align_curve", icon='SNAP_MIDPOINT'),),
        )),
    )),
    ("SURFACE_EDIT", "Surface Edit", (
        ("Shape", (
            (_t("nproto.edit_cvs", icon='EDITMODE_HLT'), _t("nproto.flip_normal", icon='NORMALS_FACE')),
            (_t("nproto.rebuild_surface", icon='MOD_REMESH'), _t("nproto.extend", icon='FULLSCREEN_ENTER')),
            (_t("nproto.surface_align", icon='SNAP_EDGE'), _t("nproto.refit_trim", icon='MOD_LATTICE')),
        )),
        ("Trim and Join", (
            (_t("nproto.trim", icon='MOD_BOOLEAN'), _t("nproto.split", icon='MOD_EDGESPLIT'),
             _t("nproto.untrim", icon='LOOP_BACK')),
            (_t("nproto.join", icon='AUTOMERGE_ON'), _t("nproto.unjoin", icon='AUTOMERGE_OFF')),
        )),
        ("Fillets", (
            (_t("nproto.fillet", "Surface Fillet", 'MOD_BEVEL'), _t("nproto.edge_fillet", icon='MOD_BEVEL')),
        )),
        ("Boolean (solids)", (
            (("nproto.boolean", "Union", 'SELECT_EXTEND', {"operation": "UNION"}),
             ("nproto.boolean", "Difference", 'SELECT_SUBTRACT', {"operation": "DIFFERENCE"}),
             ("nproto.boolean", "Intersection", 'SELECT_INTERSECT', {"operation": "INTERSECTION"})),
            (_t("nproto.boundary_fill", icon='MESH_CUBE'),),
        )),
    )),
)

GROUP = {g[0]: g for g in GROUPS}
PICK_HINT = "Alt+click a tool to pick inputs instead of using the selection."


def tools(key=None):
    """Every tool of one group (or of all), in drawing order."""
    for k, _title, sections in GROUPS:
        if key is None or k == key:
            for _label, rows in sections:
                for row in rows:
                    yield from row


def _button(lay, tool):
    op, text, icon, props = tool
    kw = {"icon": icon}
    if text is not None:
        kw["text"] = text
    ptr = lay.operator(op, **kw)
    for k, v in (props or {}).items():
        setattr(ptr, k, v)


def draw_panel(lay, key):
    col = lay.column(align=True)
    for i, (_label, rows) in enumerate(GROUP[key][2]):
        if i:
            col.separator(factor=0.6)
        for row in rows:
            r = col.row(align=True)
            for tool in row:
                _button(r, tool)


def draw_menu(lay, key):
    for i, (label, rows) in enumerate(GROUP[key][2]):
        if i:
            lay.separator()
        lay.label(text=label)
        for row in rows:
            for tool in row:
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
        draw_panel(self.layout, self.group)
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
        lay.separator()
        lay.operator("nproto.continuity", icon='DRIVER_DISTANCE')
        lay.operator("nproto.section_add", icon='MOD_EDGESPLIT')


def menu_header(self, context):
    if context.mode == 'OBJECT':
        self.layout.menu("VIEW3D_MT_nproto_tools")


def menu_add(self, context):
    lay = self.layout
    lay.separator()
    lay.menu("VIEW3D_MT_nproto_curves", text="NURBS Curves", icon='CURVE_NCURVE')
    lay.menu("VIEW3D_MT_nproto_surfaces", text="NURBS Surfaces", icon='SURFACE_NSURFACE')


classes = (VIEW3D_PT_nproto_curves, VIEW3D_PT_nproto_surfaces, VIEW3D_PT_nproto_curve_edit,
           VIEW3D_PT_nproto_surface_edit, VIEW3D_MT_nproto_curves, VIEW3D_MT_nproto_surfaces,
           VIEW3D_MT_nproto_curve_edit, VIEW3D_MT_nproto_surface_edit, VIEW3D_MT_nproto_tools)


def register():
    bpy.types.VIEW3D_MT_editor_menus.append(menu_header)
    bpy.types.VIEW3D_MT_add.append(menu_add)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(menu_add)
    bpy.types.VIEW3D_MT_editor_menus.remove(menu_header)
